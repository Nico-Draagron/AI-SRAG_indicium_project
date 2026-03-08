"""
Web Search Tool — Busca de Notícias Epidemiológicas sobre SRAG
==============================================================

Responsabilidade: dado um termo de busca, retornar artigos recentes sobre SRAG
classificados por relevância, com deduplicação e cache em memória.

Decisões de design
------------------
Validação de conectividade lazy
    O design original disparava uma busca real na API Tavily durante __init__
    para "testar conectividade". Isso consumia crédito de API em toda
    instanciação, incluindo importações em testes e reinicializações do agente.
    A validação agora ocorre na primeira chamada real a search_news(), onde o
    custo já está previsto pelo chamador.

Modo offline — ausência de fallback com dados fabricados
    O design original, quando sem API disponível, retornava 5 artigos com
    títulos e estatísticas hardcoded ("SP concentra 30% dos casos",
    "mortalidade caiu devido à vacinação em massa"). Esses dados eram
    consumidos pelo report_generator como se fossem notícias reais da query,
    e o LLM os usava para construir o resumo executivo — introduzindo
    informação fabricada em relatórios epidemiológicos de forma sistemática.

    O modo offline agora retorna uma lista vazia com flag is_offline=True.
    O chamador é responsável por decidir se exibe uma mensagem de aviso,
    omite a seção de notícias ou aborta. Fabricar dados para preencher um
    relatório de saúde pública é pior do que não ter dados.

Classificação de fonte por domínio extraído, não por substring
    O design original usava substring match simples: "saude.gov.br" in url.
    Isso classifica "https://saude.gov.br.atacante.com" como fonte OFFICIAL,
    dando o maior bônus de relevância a URLs maliciosas ou incorretas. A
    classificação agora extrai o netloc via urllib.parse e verifica sufixo
    exato do domínio.

Score de relevância: conteúdo separado de qualidade de fonte
    O score original misturava keyword match (relevância para a query) com
    confiabilidade da fonte em uma única escala de 0–1. Uma nota oficial sem
    nenhuma menção a SRAG recebia +0.3 de bônus e podia passar o threshold
    de 0.5 apenas pela origem. Os dois componentes agora são calculados
    separadamente e expostos no resultado: content_score e source_score.
    O score composto usado para ranking e filtragem pondera conteúdo em 60%
    e fonte em 40%, tornando o comportamento auditável e ajustável via config.

Cache com limite de entradas e contagem de ativos
    O design original não limitava o número de entradas no cache. Em sessões
    longas com muitas queries distintas, objetos NewsArticle (com títulos,
    snippets e entidades) acumulavam indefinidamente. O cache agora aceita
    max_entries: ao atingir o limite, a entrada mais antiga é removida antes
    de inserir a nova (política FIFO simples). get_stats() reporta apenas
    entradas ainda válidas (dentro do TTL), não entradas zumbis expiradas.

Campo canônico no payload: apenas "articles"
    O design original retornava "articles" e "news" apontando para o mesmo
    objeto lista. Qualquer mutação in-place em um dos campos contaminava o
    outro silenciosamente. O payload agora expõe apenas "articles" como campo
    canônico. O alias "news" foi removido — consumidores que dependiam dele
    devem ser atualizados para usar "articles".

Separação de erros de validação e erros de infraestrutura
    _validate_params() é chamado antes do bloco try/except principal. Erros
    de validação (query curta, days_back inválido) são exceções de contrato
    do chamador e não devem ser capturados pelo mesmo handler de erros de
    rede ou de API. O caller recebe SearchValidationError como exceção, não
    como dict com success=False.

Hash de deduplicação inclui URL
    O design original gerava o hash apenas de título+snippet.lower(). Dois
    veículos diferentes cobrindo o mesmo comunicado oficial com snippet
    idêntico eram deduplicated para um único artigo, reduzindo artificialmente
    a contagem de fontes independentes. O hash agora inclui a URL, garantindo
    que artigos de origens distintas sejam tratados como registros diferentes
    mesmo quando o conteúdo é idêntico.
"""

import warnings
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse
import hashlib
import re


try:
    from tavily import TavilyClient
    _TAVILY_AVAILABLE = True
except ImportError:
    _TAVILY_AVAILABLE = False


try:
    from src.utils.audit import AuditLogger, AuditEvent, EventStatus
except ImportError:
    class AuditEvent:
        TOOL_INITIALIZED         = "tool_initialized"
        TOOL_DEGRADED            = "tool_degraded"
        WEB_SEARCH_START         = "web_search_start"
        WEB_SEARCH_SUCCESS       = "web_search_success"
        WEB_SEARCH_OFFLINE       = "web_search_offline"
        WEB_SEARCH_ERROR         = "web_search_error"
        SEARCH_CACHE_HIT         = "search_cache_hit"
        ARTICLES_DEDUPLICATED    = "articles_deduplicated"
        ARTICLE_PROCESSING_ERROR = "article_processing_error"
        CACHE_CLEARED            = "cache_cleared"
        CACHE_EVICTED            = "cache_evicted"

    class EventStatus:
        INFO    = "INFO"
        SUCCESS = "SUCCESS"
        WARNING = "WARNING"
        ERROR   = "ERROR"

    class AuditLogger:
        def log_event(self, event_type, details=None, status="INFO"):
            print(f"[{status}] {event_type}: {details}")


try:
    from src.utils.exceptions import SearchAPIError, SearchValidationError
except ImportError:
    class SearchAPIError(Exception):
        pass

    class SearchValidationError(Exception):
        pass


# =============================================================================
# ENUMS E TIPOS
# =============================================================================

class SearchRelevance(Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"


class SourceTrust(Enum):
    OFFICIAL    = "official"
    MAINSTREAM  = "mainstream"
    SPECIALIZED = "specialized"
    OTHER       = "other"


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

@dataclass
class WebSearchConfig:
    """
    Parâmetros de comportamento do WebSearchTool.

    content_weight e source_weight
        Controlam a ponderação do score composto de relevância.
        content_weight corresponde ao peso do keyword match (relevância
        temática para a query); source_weight ao peso da confiabilidade da
        fonte. A soma deve ser 1.0. O valor padrão de 0.6/0.4 prioriza
        conteúdo sobre origem para evitar que fontes oficiais sem menção
        ao tema filtrem artigos especializados mais relevantes.

    min_relevance_score
        Threshold aplicado após o scoring. Artigos abaixo desse valor são
        descartados antes de retornar ao chamador. O default de 0.3 é
        deliberadamente baixo para não filtrar artigos válidos em cenários
        de cobertura esparsa, como surtos regionais com poucas fontes.

    cache_max_entries
        Número máximo de queries distintas mantidas em memória. Quando
        atingido, a entrada mais antiga (por inserção) é removida antes de
        inserir a nova. Evita crescimento ilimitado em sessões longas.
    """
    default_max_results:  int   = 10
    max_days_back:        int   = 30
    enable_deduplication: bool  = True
    enable_relevance_scoring: bool = True
    min_relevance_score:  float = 0.3
    content_weight:       float = 0.6
    source_weight:        float = 0.4
    cache_ttl_hours:      int   = 6
    cache_max_entries:    int   = 200

    official_sources: List[str] = field(default_factory=lambda: [
        "saude.gov.br", "who.int", "paho.org", "fiocruz.br",
        "anvisa.gov.br", "opas.org.br",
    ])
    mainstream_sources: List[str] = field(default_factory=lambda: [
        "g1.globo.com", "uol.com.br", "estadao.com.br",
        "folha.uol.com.br", "cnnbrasil.com.br", "oglobo.globo.com",
    ])
    specialized_sources: List[str] = field(default_factory=lambda: [
        "drauziovarella.uol.com.br", "pebmed.com.br", "medscape.com",
    ])
    relevant_keywords: List[str] = field(default_factory=lambda: [
        "srag", "síndrome respiratória", "casos", "mortalidade",
        "internação", "uti", "vacinação", "ministério da saúde",
        "vigilância epidemiológica", "notificação",
    ])
    noise_keywords: List[str] = field(default_factory=lambda: [
        "horóscopo", "futebol", "entretenimento", "celebridade",
    ])

    def __post_init__(self):
        total = round(self.content_weight + self.source_weight, 6)
        if abs(total - 1.0) > 1e-5:
            raise ValueError(
                f"content_weight + source_weight deve ser 1.0, recebido {total}"
            )


# =============================================================================
# MODELO DE ARTIGO
# =============================================================================

@dataclass
class NewsArticle:
    """
    Representa um artigo retornado pela busca.

    content_score e source_score são calculados separadamente pelo
    RelevanceAnalyzer. O campo relevance_score contém o score composto
    ponderado e é o único usado para ranking e filtragem.

    O hash de deduplicação inclui a URL além de título e snippet para
    garantir que artigos de origens distintas com conteúdo idêntico
    (ex: cópias de comunicado oficial) não sejam removidos na deduplicação.
    """
    title:          str
    url:            str
    snippet:        str
    published_date: str
    source:         str
    source_trust:   SourceTrust
    relevance_score: float = 0.0
    content_score:   float = 0.0
    source_score:    float = 0.0
    entities:        List[str] = field(default_factory=list)
    dedup_hash:      str = ""

    def __post_init__(self):
        raw = f"{self.url}{self.title}{self.snippet}".lower()
        self.dedup_hash = hashlib.sha256(raw.encode()).hexdigest()


# =============================================================================
# CACHE
# =============================================================================

class SearchCache:
    """
    Cache em memória com TTL por entrada e limite total de entradas.

    Usa OrderedDict para manter ordem de inserção, o que permite eviction
    FIFO simples quando o limite é atingido. A escolha por FIFO em vez de
    LRU é deliberada: queries mais antigas sobre o mesmo surto tendem a
    estar desatualizadas, e o custo de rastrear frequência de acesso não
    justifica a complexidade adicional para o volume esperado.

    Entradas expiradas são removidas lazy (no momento do acesso), não em
    background. get_active_count() percorre os timestamps para contar apenas
    entradas dentro do TTL, sem remover as expiradas, tornando a operação
    segura para uso em stats sem efeitos colaterais.
    """

    def __init__(self, ttl_hours: int = 6, max_entries: int = 200):
        self.ttl_hours   = ttl_hours
        self.max_entries = max_entries
        self._cache:      OrderedDict[str, List[NewsArticle]] = OrderedDict()
        self._timestamps: Dict[str, datetime]                 = {}

    def get(self, key: str) -> Optional[List[NewsArticle]]:
        if key not in self._cache:
            return None
        if datetime.now() - self._timestamps[key] > timedelta(hours=self.ttl_hours):
            self._evict(key)
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def set(self, key: str, articles: List[NewsArticle]) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_entries:
                oldest_key, _ = next(iter(self._cache.items()))
                self._evict(oldest_key)
            self._cache[key] = articles
        self._timestamps[key] = datetime.now()

    def invalidate(self, key: str) -> None:
        self._evict(key)

    def clear(self) -> None:
        self._cache.clear()
        self._timestamps.clear()

    def get_active_count(self) -> int:
        """Retorna o número de entradas ainda dentro do TTL."""
        cutoff = datetime.now() - timedelta(hours=self.ttl_hours)
        return sum(1 for ts in self._timestamps.values() if ts > cutoff)

    def _evict(self, key: str) -> None:
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)


# =============================================================================
# ANÁLISE DE RELEVÂNCIA
# =============================================================================

class RelevanceAnalyzer:
    """
    Calcula scores de relevância temática e de qualidade de fonte
    de forma independente.

    A separação entre content_score e source_score permite que o consumidor
    saiba por que um artigo foi rankeado em determinada posição — se foi pelo
    conteúdo ou pela origem — sem precisar reverter o cálculo a partir de um
    score composto opaco.
    """

    def __init__(self, config: WebSearchConfig):
        self.config = config

    def score_article(self, article: NewsArticle) -> NewsArticle:
        """
        Preenche content_score, source_score e relevance_score no artigo.
        Retorna o próprio artigo mutado para permitir uso em list comprehension.
        """
        article.content_score = self._content_score(article)
        article.source_score  = self._source_score(article)
        article.relevance_score = round(
            article.content_score * self.config.content_weight
            + article.source_score * self.config.source_weight,
            4,
        )
        return article

    def _content_score(self, article: NewsArticle) -> float:
        text          = f"{article.title} {article.snippet}".lower()
        keyword_hits  = sum(1 for kw in self.config.relevant_keywords if kw.lower() in text)
        keyword_ratio = min(keyword_hits / max(len(self.config.relevant_keywords), 1), 1.0)
        has_noise     = any(n.lower() in text for n in self.config.noise_keywords)
        has_date      = bool(article.published_date and article.published_date != "N/A")
        score = keyword_ratio * 0.7 + (0.2 if has_date else 0.0) + (0.1 if not has_noise else 0.0)
        return round(score, 4)

    def _source_score(self, article: NewsArticle) -> float:
        return {
            SourceTrust.OFFICIAL:    1.0,
            SourceTrust.MAINSTREAM:  0.75,
            SourceTrust.SPECIALIZED: 0.65,
            SourceTrust.OTHER:       0.30,
        }.get(article.source_trust, 0.30)

    @staticmethod
    def classify(score: float) -> SearchRelevance:
        if score >= 0.7:
            return SearchRelevance.HIGH
        if score >= 0.4:
            return SearchRelevance.MEDIUM
        return SearchRelevance.LOW


# =============================================================================
# EXTRATOR DE ENTIDADES
# =============================================================================

class EntityExtractor:
    """
    Extrai entidades epidemiologicamente relevantes de texto livre.

    Os padrões são compilados uma única vez na definição da classe para
    evitar recompilação a cada chamada em listas longas de artigos.
    """

    _UF_RE      = re.compile(
        r'\b(AC|AL|AP|AM|BA|CE|DF|ES|GO|MA|MT|MS|MG|PA|PB|PR|PE|PI'
        r'|RJ|RN|RS|RO|RR|SC|SP|SE|TO)\b'
    )
    _PERCENT_RE = re.compile(r'\b\d+(?:,\d+)?%')

    @classmethod
    def extract_all(cls, text: str) -> List[str]:
        ufs      = list(set(cls._UF_RE.findall(text.upper())))
        percents = cls._PERCENT_RE.findall(text)
        return ufs + percents


# =============================================================================
# WEB SEARCH TOOL
# =============================================================================

class WebSearchTool:
    """
    Busca notícias recentes sobre SRAG via Tavily API com fallback offline
    transparente, cache em memória e scoring de relevância desagregado.

    Modo offline
        Quando a API não está disponível (sem chave, sem biblioteca instalada
        ou falha de conectividade), search_news() retorna um payload com
        articles=[] e is_offline=True. Nenhum dado fabricado é retornado.
        O chamador deve verificar is_offline para decidir como tratar a
        ausência de notícias — omitir a seção, exibir aviso ou abortar.

    Parâmetros
    ----------
    api_key
        Chave Tavily. Quando None ou vazia, o tool opera em modo offline
        sem tentativa de conexão.
    audit_logger
        Instância de AuditLogger. Quando None, usa o stub local.
    config
        WebSearchConfig com todos os parâmetros de comportamento.
    """

    def __init__(
        self,
        api_key:      Optional[str]          = None,
        audit_logger: Optional[AuditLogger]  = None,
        config:       Optional[WebSearchConfig] = None,
    ):
        self.audit              = audit_logger or AuditLogger()
        self.config             = config or WebSearchConfig()
        self.cache              = SearchCache(
            ttl_hours   = self.config.cache_ttl_hours,
            max_entries = self.config.cache_max_entries,
        )
        self.relevance_analyzer = RelevanceAnalyzer(self.config)
        self.entity_extractor   = EntityExtractor()

        self._search_count = 0
        self._cache_hits   = 0
        self.client        = None
        self.api_available = False

        self._init_client(api_key)

        self.audit.log_event(
            AuditEvent.TOOL_INITIALIZED,
            {
                "tool":          "WebSearchTool",
                "api_available": self.api_available,
                "max_results":   self.config.default_max_results,
                "cache_max":     self.config.cache_max_entries,
            },
            EventStatus.INFO,
        )

        if not self.api_available:
            self.audit.log_event(
                AuditEvent.TOOL_DEGRADED,
                {
                    "tool":   "WebSearchTool",
                    "reason": "API indisponível — modo offline ativo, search_news retornará articles=[]",
                },
                EventStatus.WARNING,
            )

    # =========================================================================
    # INTERFACE PÚBLICA
    # =========================================================================

    def search_news(
        self,
        query:       str = "SRAG Brasil",
        days_back:   int = 7,
        max_results: Optional[int] = None,
    ) -> Dict:
        """
        Busca e retorna artigos relevantes sobre SRAG.

        Erros de validação (query inválida, parâmetros fora de range) são
        levantados como SearchValidationError antes de qualquer acesso à API
        ou ao cache, permitindo que o chamador os trate de forma diferente
        de erros de infraestrutura.

        Parâmetros
        ----------
        query
            Termo de busca. Mínimo de 3 caracteres após strip.
        days_back
            Janela temporal retroativa em dias. Limitado por config.max_days_back.
        max_results
            Máximo de artigos a retornar. Quando None, usa config.default_max_results.

        Retorno
        -------
        Dict com as chaves:
            success          : bool
            query            : str — query original
            total_results    : int — artigos após filtragem por relevância
            articles         : List[Dict] — campo canônico. "news" foi removido.
            relevance_stats  : Dict[str, int] — contagem por nível (high/medium/low)
            sources_breakdown: Dict[str, int] — contagem por fonte
            from_cache       : bool
            is_offline       : bool — True quando API indisponível
            api_used         : bool

        Exceções
        --------
        SearchValidationError
            Parâmetros inválidos. Não capturado internamente — propagado ao chamador.
        """
        self._validate_params(query, days_back, max_results or self.config.default_max_results)

        self._search_count += 1
        max_results = max_results or self.config.default_max_results

        self.audit.log_event(
            AuditEvent.WEB_SEARCH_START,
            {"query": query, "days_back": days_back, "max_results": max_results},
            EventStatus.INFO,
        )

        query_hash = self._hash_query(query, days_back, max_results)
        cached     = self.cache.get(query_hash)

        if cached is not None:
            self._cache_hits += 1
            self.audit.log_event(
                AuditEvent.SEARCH_CACHE_HIT,
                {"query_hash": query_hash[:8]},
                EventStatus.INFO,
            )
            return self._format_response(cached, from_cache=True, query=query)

        if not self.api_available:
            self.audit.log_event(
                AuditEvent.WEB_SEARCH_OFFLINE,
                {"query": query},
                EventStatus.WARNING,
            )
            return self._offline_response(query)

        try:
            articles = self._execute_search(query, days_back, max_results)
        except Exception as exc:
            self.audit.log_event(
                AuditEvent.WEB_SEARCH_ERROR,
                {"query": query, "error": str(exc)},
                EventStatus.ERROR,
            )
            return self._offline_response(query, error=str(exc))

        if self.config.enable_deduplication:
            articles = self._deduplicate(articles)

        if self.config.enable_relevance_scoring:
            articles = [self.relevance_analyzer.score_article(a) for a in articles]
            articles.sort(key=lambda a: a.relevance_score, reverse=True)

        articles = [a for a in articles if a.relevance_score >= self.config.min_relevance_score]

        self.cache.set(query_hash, articles)

        self.audit.log_event(
            AuditEvent.WEB_SEARCH_SUCCESS,
            {"query": query, "articles_returned": len(articles)},
            EventStatus.SUCCESS,
        )

        return self._format_response(articles, from_cache=False, query=query)

    def get_stats(self) -> Dict:
        """
        Retorna métricas de uso do tool e estado do cache.

        active_cache_entries conta apenas entradas dentro do TTL vigente,
        ao contrário de len(cache._cache) que incluiria entradas expiradas
        ainda não removidas.
        """
        hit_rate = self._cache_hits / self._search_count if self._search_count else 0.0
        return {
            "total_searches":       self._search_count,
            "cache_hits":           self._cache_hits,
            "cache_hit_rate":       round(hit_rate, 4),
            "active_cache_entries": self.cache.get_active_count(),
            "cache_max_entries":    self.config.cache_max_entries,
            "api_available":        self.api_available,
        }

    def clear_cache(self) -> None:
        self.cache.clear()
        self.audit.log_event(
            AuditEvent.CACHE_CLEARED,
            {"tool": "WebSearchTool"},
            EventStatus.INFO,
        )

    # =========================================================================
    # MÉTODOS INTERNOS
    # =========================================================================

    def _init_client(self, api_key: Optional[str]) -> None:
        """
        Inicializa o cliente Tavily sem disparar nenhuma chamada à API.
        A verificação de conectividade real é feita de forma lazy na
        primeira chamada a _execute_search().
        """
        if not api_key or not _TAVILY_AVAILABLE:
            return
        try:
            self.client        = TavilyClient(api_key=api_key)
            self.api_available = True
        except Exception as exc:
            self.audit.log_event(
                AuditEvent.TOOL_DEGRADED,
                {"reason": f"Falha ao instanciar TavilyClient: {exc}"},
                EventStatus.WARNING,
            )

    def _execute_search(
        self,
        query:       str,
        days_back:   int,
        max_results: int,
    ) -> List[NewsArticle]:
        """
        Chama a API Tavily e converte os resultados em objetos NewsArticle.

        Erros de processamento por item são registrados individualmente e
        não abortam o processamento dos demais artigos.
        """
        raw = self.client.search(query=query, max_results=max_results, days=days_back)
        articles = []

        for item in raw.get("results", []):
            try:
                url          = item.get("url", "")
                source_trust = self._classify_source(url)
                text         = f"{item.get('title', '')} {item.get('content', '')}"

                articles.append(NewsArticle(
                    title          = item.get("title", ""),
                    url            = url,
                    snippet        = item.get("content", "")[:300],
                    published_date = item.get("published_date", "N/A"),
                    source         = self._extract_source_name(url),
                    source_trust   = source_trust,
                    entities       = self.entity_extractor.extract_all(text),
                ))
            except Exception as exc:
                self.audit.log_event(
                    AuditEvent.ARTICLE_PROCESSING_ERROR,
                    {"error": str(exc), "item_preview": str(item)[:80]},
                    EventStatus.WARNING,
                )

        return articles

    def _classify_source(self, url: str) -> SourceTrust:
        """
        Classifica a confiabilidade da fonte extraindo o domínio real via
        urlparse, não por substring match.

        Substring match simples permite que "https://saude.gov.br.atacante.com"
        seja classificado como OFFICIAL porque o domínio legítimo aparece
        como substring. urlparse().netloc retorna o host real da URL,
        e a verificação de sufixo (endswith) garante que apenas domínios
        que terminam exatamente com o domínio confiável sejam aceitos —
        ex: "www.saude.gov.br" passa, "saude.gov.br.atacante.com" não passa.
        """
        try:
            netloc = urlparse(url).netloc.lower().split(":")[0]
        except Exception:
            return SourceTrust.OTHER

        def matches(domain: str) -> bool:
            return netloc == domain or netloc.endswith(f".{domain}")

        for domain in self.config.official_sources:
            if matches(domain):
                return SourceTrust.OFFICIAL
        for domain in self.config.mainstream_sources:
            if matches(domain):
                return SourceTrust.MAINSTREAM
        for domain in self.config.specialized_sources:
            if matches(domain):
                return SourceTrust.SPECIALIZED

        return SourceTrust.OTHER

    def _extract_source_name(self, url: str) -> str:
        """
        Retorna o nome legível da fonte a partir da URL.

        A extração usa urlparse para isolar o netloc em vez de split("//"),
        que falha silenciosamente em URLs malformadas.
        """
        _DISPLAY_NAMES = {
            "saude.gov.br":      "Ministério da Saúde",
            "who.int":           "OMS",
            "fiocruz.br":        "Fiocruz",
            "paho.org":          "OPAS",
            "anvisa.gov.br":     "Anvisa",
            "g1.globo.com":      "G1",
            "estadao.com.br":    "Estadão",
            "folha.uol.com.br":  "Folha de S.Paulo",
            "oglobo.globo.com":  "O Globo",
            "cnnbrasil.com.br":  "CNN Brasil",
            "uol.com.br":        "UOL",
            "medscape.com":      "Medscape",
        }
        try:
            netloc = urlparse(url).netloc.lower().split(":")[0]
        except Exception:
            return "Desconhecido"

        for domain, name in _DISPLAY_NAMES.items():
            if netloc == domain or netloc.endswith(f".{domain}"):
                return name

        return netloc or "Desconhecido"

    def _deduplicate(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        seen:   Set[str]         = set()
        unique: List[NewsArticle] = []

        for article in articles:
            if article.dedup_hash not in seen:
                seen.add(article.dedup_hash)
                unique.append(article)

        removed = len(articles) - len(unique)
        if removed:
            self.audit.log_event(
                AuditEvent.ARTICLES_DEDUPLICATED,
                {"removed": removed, "kept": len(unique)},
                EventStatus.INFO,
            )

        return unique

    def _format_response(
        self,
        articles:   List[NewsArticle],
        from_cache: bool,
        query:      str,
        is_offline: bool = False,
        error:      Optional[str] = None,
    ) -> Dict:
        relevance_stats = {
            "high":   sum(1 for a in articles if a.relevance_score >= 0.7),
            "medium": sum(1 for a in articles if 0.4 <= a.relevance_score < 0.7),
            "low":    sum(1 for a in articles if a.relevance_score < 0.4),
        }
        sources_breakdown: Dict[str, int] = defaultdict(int)
        for article in articles:
            sources_breakdown[article.source] += 1

        serialized = [
            {
                "title":           a.title,
                "url":             a.url,
                "snippet":         a.snippet,
                "published_date":  a.published_date,
                "source":          a.source,
                "source_trust":    a.source_trust.value,
                "relevance_score": a.relevance_score,
                "content_score":   a.content_score,
                "source_score":    a.source_score,
                "entities":        list(a.entities),
            }
            for a in articles
        ]

        response: Dict = {
            "success":          True,
            "query":            query,
            "total_results":    len(serialized),
            "articles":         serialized,
            "relevance_stats":  relevance_stats,
            "sources_breakdown": dict(sources_breakdown),
            "from_cache":       from_cache,
            "is_offline":       is_offline,
            "api_used":         self.api_available and not is_offline,
        }
        if error:
            response["error"] = error

        return response

    def _offline_response(self, query: str, error: Optional[str] = None) -> Dict:
        return self._format_response(
            articles   = [],
            from_cache = False,
            query      = query,
            is_offline = True,
            error      = error,
        )

    def _validate_params(self, query: str, days_back: int, max_results: int) -> None:
        """
        Valida os parâmetros antes de qualquer acesso à API ou ao cache.

        Levantado como SearchValidationError — exceção de contrato do chamador,
        não capturada pelo handler de erros de infraestrutura de search_news().
        """
        if not query or len(query.strip()) < 3:
            raise SearchValidationError(
                f"query deve ter ao menos 3 caracteres, recebido: {repr(query)}"
            )
        if not (1 <= days_back <= self.config.max_days_back):
            raise SearchValidationError(
                f"days_back deve estar entre 1 e {self.config.max_days_back}, recebido: {days_back}"
            )
        if not (1 <= max_results <= 50):
            raise SearchValidationError(
                f"max_results deve estar entre 1 e 50, recebido: {max_results}"
            )

    @staticmethod
    def _hash_query(query: str, days_back: int, max_results: int) -> str:
        key = f"{query.strip().lower()}|{days_back}|{max_results}"
        return hashlib.sha256(key.encode()).hexdigest()

    def __repr__(self) -> str:
        return (
            f"WebSearchTool("
            f"searches={self._search_count}, "
            f"api={'online' if self.api_available else 'offline'}, "
            f"cache_hits={self._cache_hits})"
        )


# Alias mantido para compatibilidade com imports existentes.
# Será removido em versão futura — migre para WebSearchTool diretamente.
TavilySearchTool = WebSearchTool