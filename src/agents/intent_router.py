"""
Intent Router — Roteamento Inteligente SQL vs RAG vs Chart vs Report
====================================================================

Classifica a intenção da query do usuário e produz um RoutingDecision que
determina qual nó do grafo LangGraph será executado pelo SRAGOrchestrator.

Pipeline
--------
    Query -> IntentClassifier -> StrategySelector -> IntentRouter.route()
          -> RoutingDecision

Estratégias de execução
-----------------------
    SQL_ONLY  : métricas e dados tabulares (FACTUAL, GEOGRAPHIC, DEMOGRAPHIC)
    RAG_ONLY  : perguntas conceituais ou metodológicas (ANALYTICAL, EXPLANATORY)
    HYBRID    : combina SQL + RAG (COMPARATIVE, misto com dado + explicação)
    CHART     : gráfico ad-hoc via SQL dinâmica + ChartTool (CHART_REQUEST)
    REPORT    : relatório epidemiológico completo — ativa SQL + RAG + charts +
                notícias em sequência orquestrada (REPORT_REQUEST)

Decisões de design
------------------
REPORT_REQUEST tem prioridade máxima sobre CHART_REQUEST
    Queries de relatório epidemiológico contêm termos como "gráficos obrigatórios"
    ou "gráficos" como complemento da entrega, não como intenção primária. O padrão
    anterior de CHART_REQUEST greedy fazia a USER_QUERY do notebook 06 cair em CHART
    — o orchestrator compensava com um override manual (_is_relatorio_query) baseado
    em contagem de keywords. REPORT_REQUEST é verificado antes de CHART_REQUEST no
    classify(), tornando o override desnecessário e o contrato explícito no router.

CHART_REQUEST restrito a pedidos de gráfico como ação primária
    O padrão anterior r'\b(gráfico|grafico|chart|plot)\b' casava com qualquer
    menção à palavra, inclusive "relatório com gráficos" ou "gráficos obrigatórios".
    Os novos padrões exigem um verbo de criação próximo ao substantivo de gráfico,
    ou que "gráfico" seja o sujeito/início da query — não apenas uma menção lateral.

EXPLANATORY mapeado para semantic_type "regra"
    Perguntas do tipo "o que é SRAG?", "como é calculada a taxa de mortalidade?"
    são respondidas por gold_rag_dicionario_regras, cujos documentos têm
    semantic_type="regra" no Vector Index. O valor "metric" não existe no índice
    e retornaria zero documentos silenciosamente.

EXPLANATORY patterns restritos a formas metodológicas
    O padrão anterior r'\b(o que é|como|defin[ae])\b' casava com "como evoluíram
    os casos?" — uma query claramente temporal/factual. O padrão "como" solto em
    português é essencialmente universal. Os novos padrões exigem a forma
    metodológica completa: "como é calculado", "como funciona", "qual o critério",
    etc. — não apenas a palavra "como".

StrategySelector com hierarquia semântica de dominância
    A versão anterior usava len(intents) > 1 → HYBRID como regra universal.
    Isso fazia FACTUAL+DEMOGRAPHIC → HYBRID (deveria ser SQL_ONLY) e
    EXPLANATORY+FACTUAL → HYBRID (deveria ser RAG_ONLY). A nova lógica usa
    conjuntos para determinar qual categoria de intenção domina:

    Regra de dominância:
    1. REPORT_REQUEST → REPORT (sempre ganha — inclui gráficos e dados por definição)
    2. CHART_REQUEST  → CHART  (exclusivo quando pedido de gráfico puro)
    3. Qualquer EXPLANATORY/ANALYTICAL, sem COMPARATIVE → RAG_ONLY
       (intents dimensionais coexistentes são contexto, não determinantes de rota)
    4. Apenas intents estruturais (FACTUAL, GEOGRAPHIC, DEMOGRAPHIC, TEMPORAL)
       em qualquer combinação → SQL_ONLY
    5. COMPARATIVE presente → HYBRID (precisa de dado + interpretação)
    6. Qualquer outro misto → HYBRID

_extract_sql_filters() expandido
    A versão anterior extraía apenas UF e ano_mes (quando mês e ano coexistiam).
    Ano isolado é agora extraído independentemente de mês. Faixa etária é detectada
    por palavras-chave e convertida ao formato canônico do catálogo. Sexo é
    extraído quando presente. Intervalo de anos é detectado no formato
    "de YYYY a YYYY" ou "YYYY-YYYY".

Prefixo gold_ obrigatório nos nomes de tabela
    Os nomes reais no catálogo Unity Catalog (dbx_srag_lab.gold) incluem o
    prefixo gold_. Nomes sem prefixo são referências fantasma — qualquer
    consumidor de target_tables que use esses nomes falha silenciosamente.

semantic_type "metric" não existe no pipeline
    O GoldDocumentLoader indexa documentos com os tipos: kpi, regra, temporal,
    geographic, demographic. Qualquer outro valor passado como filtro para
    SRAGRetriever.search_by_type() retorna zero documentos sem levantar exceção.

route() reporta intent dominante, não MIXED por padrão
    A versão anterior definia intent = intents[0] if len==1 else MIXED. Quando
    a intent dominante era EXPLANATORY mas FACTUAL estava presente, o campo intent
    ficava MIXED, tornando o rag_semantic_type calculado inconsistente com o intent
    reportado nos logs. route() agora deriva o intent primário a partir da
    estratégia selecionada via _get_primary_intent().

Compatibilidade com o orchestrator atual
    ExecutionStrategy.REPORT é um novo valor. O orchestrator.py mapeia estratégias
    desconhecidas para "hybrid" no _route_to_execution() via else→hybrid. Isso
    significa que REPORT aciona o nó execute_hybrid, que é o comportamento correto:
    executa SQL + RAG em sequência. O orchestrator pode ser atualizado para
    adicionar um nó execute_report dedicado sem breaking change — o fallback já
    produz o resultado correto.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage


VERSION = "4.0.1"


# =============================================================================
# ENUMS
# =============================================================================

class QueryIntent(Enum):
    """Tipos de intenção de query reconhecidos pelo classificador."""
    FACTUAL        = "factual"
    ANALYTICAL     = "analytical"
    COMPARATIVE    = "comparative"
    TEMPORAL       = "temporal"
    GEOGRAPHIC     = "geographic"
    DEMOGRAPHIC    = "demographic"
    EXPLANATORY    = "explanatory"
    MIXED          = "mixed"
    CHART_REQUEST  = "chart_request"
    REPORT_REQUEST = "report_request"


class ExecutionStrategy(Enum):
    """Estratégia de execução do nó LangGraph."""
    SQL_ONLY = "sql_only"
    RAG_ONLY = "rag_only"
    HYBRID   = "hybrid"
    CHART    = "chart"
    REPORT   = "report"


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class ChartParams:
    """
    Parâmetros extraídos para geração de gráfico ad-hoc.

    Populados por IntentRouter._extract_chart_params() quando a intenção
    detectada é CHART_REQUEST. Consumidos por _execute_chart_node() no
    orchestrator.

    Atributos
    ---------
    metric     : coluna Y — o que medir (ex.: total_casos, taxa_mortalidade).
    group_by   : coluna X / agrupamento (ex.: ano_mes, sg_uf, faixa_etaria).
    chart_type : tipo de visualização — bar | line | pie.
    title      : título exibido no gráfico gerado.
    filters    : restrições WHERE; chaves possíveis: ano, sg_uf, mes.
    table      : tabela Gold fonte; ajustada conforme group_by detectado.
    """
    metric:     str  = "total_casos"
    group_by:   str  = "ano_mes"
    chart_type: str  = "bar"
    title:      str  = "Grafico SRAG"
    filters:    Dict = field(default_factory=dict)
    table:      str  = "gold_metricas_temporais"


@dataclass
class RoutingDecision:
    """
    Decisão de roteamento produzida por IntentRouter.route().

    Garantias do contrato
    ---------------------
    - intent nunca é None; fallback é QueryIntent.FACTUAL.
    - target_tables contém apenas nomes reais do catálogo (prefixo gold_).
    - rag_semantic_type, quando presente, é um dos tipos indexados:
      kpi, regra, temporal, geographic, demographic.
    - chart_params é não-None somente quando strategy == ExecutionStrategy.CHART.
    - requires_synthesis é True para HYBRID e REPORT.
    """
    intent:             QueryIntent
    strategy:           ExecutionStrategy
    confidence:         float
    reasoning:          str
    target_tables:      List[str]
    sql_filters:        Optional[Dict]        = None
    rag_semantic_type:  Optional[str]         = None
    requires_synthesis: bool                  = True
    chart_params:       Optional[ChartParams] = None


# =============================================================================
# INTENT CLASSIFIER
# =============================================================================

class IntentClassifier:
    """
    Classificador regex de intenção de queries.

    Hierarquia de prioridade de detecção
    -------------------------------------
    1. REPORT_REQUEST — tem prioridade máxima: "relatório com gráficos" é REPORT,
       não CHART. Verificado antes de todos os outros padrões.
    2. CHART_REQUEST — verificado em segundo lugar, com padrões restritos a pedidos
       onde gráfico é a ação principal (verbo de criação + substantivo de gráfico).
    3. Demais intents — avaliados em paralelo sem prioridade entre si.

    EXPLANATORY patterns restritos a formas metodológicas
        O padrão "como" solto casava com praticamente qualquer frase em português.
        Os padrões agora exigem a forma metodológica explícita: "como é calculado",
        "como funciona", "qual o critério", etc.
    """

    PATTERNS = {
        QueryIntent.REPORT_REQUEST: [
            r'\b(relat[oó]rio|report)\b',
            r'\b(panorama|boletim|bulletin)\b',
            r'\b(resumo executivo)\b',
            r'\b(situa[cç][aã]o atual|cen[aá]rio epidemiol[oó]gico)\b',
            r'\b(an[aá]lise completa|avalia[cç][aã]o completa)\b',
            r'\b(vis[aã]o geral|overview|quadro epidemiol[oó]gico)\b',
        ],
        QueryIntent.CHART_REQUEST: [
            # Verbo de criação + substantivo de gráfico (até 60 chars de distância)
            r'\b(ger[ae]|cri[ae]|plot[ae]|exib[ae]|visualiz[ae])\b.{0,60}\b(gr[aá]fico|chart|plot)\b',
            # "mostre um gráfico" / "mostre o gráfico"
            r'\b(mostre?)\b.{0,30}\b(gr[aá]fico|chart|plot)\b',
            # "gráfico" como sujeito/início da query (primeiros 20 chars)
            r'^.{0,20}\b(gr[aá]fico|chart|plot)\b',
            # "somente/apenas um gráfico" — pedido exclusivo de visualização
            r'\b(somente?|apenas|s[oó])\b.{0,25}\b(gr[aá]fico|chart)\b',
        ],
        QueryIntent.FACTUAL: [
            r'\b(quantos?|qual|quanto|quem)\b',
            r'\b(total|n[uú]mero|quantidade)\b',
            r'\b(casos|[oó]bitos|mortes)\b.*\b(em|de)\b',
        ],
        QueryIntent.ANALYTICAL: [
            r'\b(por que|porque|motivo|raz[aã]o|causas?|causou)\b',
            r'\b(analis[ae]|avaliar|entender|explicar?|explique)\b',
            r'\b(impacto|efeito|consequ[eê]ncia)\b',
        ],
        QueryIntent.COMPARATIVE: [
            # Infinitivo, imperativo e forma nominal
            r'\b(comparar?|compare|versus|vs|diferen[cç]a)\b',
            r'\b(maior|menor|melhor|pior)\b.*\b(que|do que)\b',
            r'\b(entre|e)\b.*\b(estados?|UFs?)\b',
            r'\b(compara[cç][aã]o|comparativo)\b',
        ],
        QueryIntent.TEMPORAL: [
            r'\b(tend[eê]ncia|evolu[cç][aã]o|crescimento)\b',
            r'\b([uú]ltimos?|pr[oó]ximos?|passados?)\b.*\b(meses?|anos?|dias?)\b',
            r'\b(temporal|ao longo|s[eé]rie|hist[oó]rico)\b',
            r'\b(mensal|anual|semanal|trimestral)\b',
        ],
        QueryIntent.GEOGRAPHIC: [
            r'\b(estado|UF|regi[aã]o|mapa)\b',
            r'\b(ranking|top|principais)\b.*\b(UFs?|estados?)\b',
            r'\b(SP|RJ|MG|BA|RS|PR|SC|CE|PE)\b',
            r'\b(nordeste|sudeste|sul|norte|centro.oeste)\b',
        ],
        QueryIntent.DEMOGRAPHIC: [
            r'\b(idade|idoso|crian[cç]a|adulto)\b',
            r'\b(sexo|feminino|masculino|g[eê]nero)\b',
            r'\b(faixa et[aá]ria|grupo et[aá]rio)\b',
            r'\b(gestante|puerpera|comorbidade)\b',
        ],
        QueryIntent.EXPLANATORY: [
            # Formas de definição
            r'\b(o que [eé]|o que s[aã]o|o que significa[nm]?)\b',
            r'\b(defin[ae]|defin[ae]|defini[cç][aã]o|conceito de)\b',
            r'\b(significa[nm]?)\b',
            # Formas imperativas diretas — "explique", "defina"
            r'\b(expliqu[eé]|defina)\b',
            # Formas metodológicas explícitas — não apenas "como"
            r'\b(como [eé] calculad[oa]|como s[aã]o calculad[oa]s?)\b',
            r'\b(como funciona|como [eé] definid[oa]|como [eé] obtid[oa])\b',
            r'\b(qual o crit[eé]rio|qual a metodologia|qual o denominador)\b',
            r'\b(qual a f[oó]rmula|qual o numerador|como [eé] feito o c[aá]lculo)\b',
            r'\b(metodologia|crit[eé]rio epidemiol[oó]gico)\b',
            r'\b(explicar?\s+(?:o\s+)?(?:que|como|por que))\b',
        ],
    }

    @staticmethod
    def classify(query: str) -> List[QueryIntent]:
        """
        Classifica a query e retorna lista de intenções detectadas.

        Hierarquia de verificação:
        1. REPORT_REQUEST — se encontrado, retorna imediatamente.
        2. CHART_REQUEST — se encontrado (sem contexto de relatório), retorna
           imediatamente como único intent.
        3. Demais intents — avaliados em paralelo.

        Se nenhuma intenção for detectada, retorna [QueryIntent.FACTUAL].
        """
        query_lower = query.lower()

        # REPORT_REQUEST tem prioridade máxima
        for pattern in IntentClassifier.PATTERNS[QueryIntent.REPORT_REQUEST]:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return [QueryIntent.REPORT_REQUEST]

        # CHART_REQUEST tem segunda prioridade
        for pattern in IntentClassifier.PATTERNS[QueryIntent.CHART_REQUEST]:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return [QueryIntent.CHART_REQUEST]

        detected_intents = []
        for intent, patterns in IntentClassifier.PATTERNS.items():
            if intent in (QueryIntent.REPORT_REQUEST, QueryIntent.CHART_REQUEST):
                continue
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    if intent not in detected_intents:
                        detected_intents.append(intent)
                    break

        return detected_intents if detected_intents else [QueryIntent.FACTUAL]


# =============================================================================
# STRATEGY SELECTOR
# =============================================================================

class StrategySelector:
    """
    Seleciona a ExecutionStrategy a partir da lista de intenções classificadas.

    Hierarquia de dominância semântica
    -----------------------------------
    A versão anterior usava len(intents) > 1 → HYBRID como regra universal,
    o que causava FACTUAL+DEMOGRAPHIC → HYBRID (deveria ser SQL_ONLY) e
    EXPLANATORY+FACTUAL → HYBRID (deveria ser RAG_ONLY).

    A nova lógica categoriza os intents em grupos semânticos e aplica
    dominância por grupo:

    Grupo EXCLUSIVE
        REPORT_REQUEST → REPORT (inclui SQL, RAG, charts, notícias por definição)
        CHART_REQUEST  → CHART  (gráfico ad-hoc puro)

    Grupo CONCEPTUAL
        ANALYTICAL, EXPLANATORY — perguntas sobre causa, conceito, metodologia.
        Quando presentes (mesmo com intents estruturais), determinam RAG_ONLY
        a menos que COMPARATIVE também esteja presente.

    Grupo STRUCTURAL_STATIC
        FACTUAL, GEOGRAPHIC, DEMOGRAPHIC — perguntas pontuais sobre dados
        tabulares. Qualquer combinação desses três → SQL_ONLY.

    TEMPORAL
        Queries de tendência e evolução quase sempre precisam de dado (SQL)
        + interpretação (RAG). Quando TEMPORAL é combinado com FACTUAL, ou
        quando aparece sozinho com mais de um intent, o contexto indica análise
        temporal — não apenas uma consulta pontual. Por isso TEMPORAL + qualquer
        outro intent → HYBRID. TEMPORAL sozinho → SQL_ONLY (consulta simples
        como "quais os casos em 2024 mês a mês").

    COMPARATIVE
        Sempre força HYBRID — compara dados (SQL) e requer interpretação (RAG).
    """

    _EXCLUSIVE: Dict[QueryIntent, ExecutionStrategy] = {
        QueryIntent.REPORT_REQUEST: ExecutionStrategy.REPORT,
        QueryIntent.CHART_REQUEST:  ExecutionStrategy.CHART,
    }

    _CONCEPTUAL       = frozenset({QueryIntent.ANALYTICAL, QueryIntent.EXPLANATORY})
    _STRUCTURAL_STATIC = frozenset({
        QueryIntent.FACTUAL, QueryIntent.GEOGRAPHIC, QueryIntent.DEMOGRAPHIC,
    })
    # TEMPORAL separado: sozinho é SQL, combinado com qualquer outro é HYBRID
    _STRUCTURAL_ALL    = frozenset({
        QueryIntent.FACTUAL, QueryIntent.GEOGRAPHIC,
        QueryIntent.DEMOGRAPHIC, QueryIntent.TEMPORAL,
    })

    @staticmethod
    def select(intents: List[QueryIntent]) -> ExecutionStrategy:
        """
        Retorna a estratégia para a lista de intenções fornecida.

        Parâmetros
        ----------
        intents
            Lista produzida por IntentClassifier.classify(). Nunca vazia —
            o classificador garante [FACTUAL] como fallback.
        """
        if not intents:
            return ExecutionStrategy.SQL_ONLY

        intent_set = set(intents)

        # Tier 1 — intents exclusivos sempre ganham
        for exclusive_intent, exclusive_strategy in StrategySelector._EXCLUSIVE.items():
            if exclusive_intent in intent_set:
                return exclusive_strategy

        has_conceptual   = bool(StrategySelector._CONCEPTUAL       & intent_set)
        has_temporal     = QueryIntent.TEMPORAL    in intent_set
        has_comparative  = QueryIntent.COMPARATIVE in intent_set
        has_static_only  = bool(StrategySelector._STRUCTURAL_STATIC & intent_set)

        # Tier 2 — conceptual sem comparative → RAG
        # "Como é calculada a taxa de UTI?" = EXPLANATORY + FACTUAL → RAG_ONLY
        if has_conceptual and not has_comparative:
            return ExecutionStrategy.RAG_ONLY

        # Tier 2.5 — TEMPORAL combinado com outros intents → HYBRID
        # "Como evoluíram os casos nos últimos 6 meses e qual a tendência?"
        # = FACTUAL + TEMPORAL — análise temporal precisa de dado + interpretação
        if has_temporal and len(intent_set) > 1 and not has_conceptual:
            return ExecutionStrategy.HYBRID

        # Tier 3 — apenas intents estáticos (FACTUAL, GEOGRAPHIC, DEMOGRAPHIC) → SQL
        # "Distribuição por faixa etária?" = FACTUAL + DEMOGRAPHIC → SQL_ONLY
        if has_static_only and not has_conceptual and not has_comparative and not has_temporal:
            return ExecutionStrategy.SQL_ONLY

        # Tier 4 — COMPARATIVE, misto com dado+conceito, ou TEMPORAL sozinho → depende
        if not has_comparative and not has_conceptual and has_temporal and len(intent_set) == 1:
            return ExecutionStrategy.SQL_ONLY  # TEMPORAL sozinho: consulta pontual

        # Tier 5 — COMPARATIVE ou qualquer misto restante → HYBRID
        return ExecutionStrategy.HYBRID


# =============================================================================
# INTENT ROUTER
# =============================================================================

class IntentRouter:
    """
    Roteador principal — produz RoutingDecision a partir de uma query em
    linguagem natural.

    Pipeline
    --------
        Query
          -> classify (regex ou LLM)
          -> select strategy
          -> determine target_tables
          -> extract sql_filters
          -> determine rag_semantic_type
          -> extract chart_params (se CHART)
          -> RoutingDecision

    Contrato com o orchestrator
    ---------------------------
    O RoutingDecision retornado garante:
    - intent nunca None (fallback: QueryIntent.FACTUAL).
    - target_tables com nomes reais do catálogo (prefixo gold_).
    - rag_semantic_type em {kpi, regra, temporal, geographic, demographic} ou None.
    - chart_params preenchido somente quando strategy == CHART.
    - requires_synthesis é True para HYBRID e REPORT.

    Classificação via LLM
    ---------------------
    Quando use_llm_classification=True, o parâmetro llm= deve ser fornecido
    explicitamente ao construtor. Sem llm, o router cai silenciosamente para
    classificação regex, evitando AttributeError em runtime.
    """

    def __init__(
        self,
        use_llm_classification: bool = False,
        llm: Optional[BaseChatModel] = None,
    ):
        self.use_llm    = use_llm_classification
        self.llm        = llm
        self.classifier = IntentClassifier()
        self.selector   = StrategySelector()

    def route(self, query: str) -> RoutingDecision:
        """
        Roteia a query para a estratégia de execução apropriada.

        Parâmetros
        ----------
        query
            Query do usuário em linguagem natural.

        Retorno
        -------
        RoutingDecision com todos os campos preenchidos conforme os contratos
        documentados na classe.
        """
        intents = (
            self._classify_with_llm(query)
            if self.use_llm and self.llm
            else self.classifier.classify(query)
        )

        strategy       = self.selector.select(intents)
        target_tables  = self._determine_target_tables(intents, query)
        sql_filters    = (
            self._extract_sql_filters(query)
            if strategy not in (ExecutionStrategy.RAG_ONLY,)
            else None
        )
        rag_type = (
            self._determine_rag_type(intents)
            if strategy not in (ExecutionStrategy.SQL_ONLY, ExecutionStrategy.CHART)
            else None
        )
        confidence     = self._calculate_confidence(intents, query)
        primary_intent = self._get_primary_intent(intents, strategy)
        reasoning      = self._generate_reasoning(intents, strategy, target_tables)
        chart_params   = (
            self._extract_chart_params(query)
            if strategy == ExecutionStrategy.CHART
            else None
        )

        return RoutingDecision(
            intent             = primary_intent,
            strategy           = strategy,
            confidence         = confidence,
            reasoning          = reasoning,
            target_tables      = target_tables,
            sql_filters        = sql_filters,
            rag_semantic_type  = rag_type,
            requires_synthesis = strategy in (
                ExecutionStrategy.HYBRID, ExecutionStrategy.REPORT
            ),
            chart_params       = chart_params,
        )

    # =========================================================================
    # PRIMARY INTENT RESOLUTION
    # =========================================================================

    @staticmethod
    def _get_primary_intent(
        intents:  List[QueryIntent],
        strategy: ExecutionStrategy,
    ) -> QueryIntent:
        """
        Deriva a intenção primária a partir da estratégia selecionada.

        A versão anterior usava intents[0] if len==1 else MIXED. Isso perdia
        a intenção dominante quando múltiplos intents eram detectados —
        EXPLANATORY+FACTUAL ficava como MIXED, tornando o rag_semantic_type
        inconsistente com o intent reportado nos logs.

        A intenção primária é mapeada da estratégia:
        - REPORT → REPORT_REQUEST
        - CHART  → CHART_REQUEST
        - RAG_ONLY → primeiro EXPLANATORY ou ANALYTICAL detectado
        - SQL_ONLY → primeiro intent estrutural detectado
        - HYBRID   → MIXED (genuíno — múltiplas fontes necessárias)
        """
        if strategy == ExecutionStrategy.REPORT:
            return QueryIntent.REPORT_REQUEST
        if strategy == ExecutionStrategy.CHART:
            return QueryIntent.CHART_REQUEST

        if strategy == ExecutionStrategy.RAG_ONLY:
            for intent in intents:
                if intent in (QueryIntent.EXPLANATORY, QueryIntent.ANALYTICAL):
                    return intent
            return intents[0] if intents else QueryIntent.FACTUAL

        if strategy == ExecutionStrategy.SQL_ONLY:
            return intents[0] if intents else QueryIntent.FACTUAL

        # HYBRID — genuinamente misto
        return QueryIntent.MIXED if len(intents) > 1 else (
            intents[0] if intents else QueryIntent.FACTUAL
        )

    # =========================================================================
    # LLM CLASSIFICATION (opcional)
    # =========================================================================

    def _classify_with_llm(self, query: str) -> List[QueryIntent]:
        """
        Classifica a intenção via LLM.

        Usado apenas quando use_llm_classification=True e llm != None.
        Strings não reconhecidas retornadas pelo LLM são logadas e ignoradas.
        Se todas as strings falharem, retorna [QueryIntent.FACTUAL].
        """
        prompt = f"""Classifique a intenção desta query sobre SRAG (Síndrome Respiratória Aguda Grave):

Query: "{query}"

Intencoes possiveis:
- FACTUAL: perguntas objetivas sobre números (quantos, qual, quanto)
- ANALYTICAL: perguntas sobre causas, impactos (por que, motivo, impacto)
- COMPARATIVE: comparações entre períodos, estados, grupos (maior, menor, vs)
- TEMPORAL: tendências e evolução ao longo do tempo (evolução, série, últimos meses)
- GEOGRAPHIC: questões por estado, região ou UF
- DEMOGRAPHIC: perfil por faixa etária, sexo, comorbidade
- EXPLANATORY: definições, conceitos e metodologia de cálculo de indicadores
- MIXED: combinação de múltiplas intenções sem dominância clara
- CHART_REQUEST: pedido explícito de geração de gráfico isolado como ação principal
- REPORT_REQUEST: pedido de relatório epidemiológico completo, panorama ou boletim

Responda APENAS com as intencoes detectadas, separadas por virgula.
Exemplo: "FACTUAL, GEOGRAPHIC"
Se for relatório completo: "REPORT_REQUEST"
Se for pedido de gráfico isolado: "CHART_REQUEST"
"""
        response     = self.llm.invoke([HumanMessage(content=prompt)])
        intent_names = [i.strip() for i in response.content.split(",")]
        intents      = []
        for name in intent_names:
            try:
                intents.append(QueryIntent[name.strip().upper()])
            except KeyError:
                print(
                    f"[intent_router] aviso: intent '{name}' nao reconhecido pelo LLM — ignorado"
                )
        return intents if intents else [QueryIntent.FACTUAL]

    # =========================================================================
    # CHART PARAMS EXTRACTION
    # =========================================================================

    def _extract_chart_params(self, query: str) -> ChartParams:
        """
        Extrai parâmetros para geração de gráfico ad-hoc a partir da query.

        Mapeia palavras-chave para:
            metric     : coluna Y (o que medir)
            group_by   : coluna X (como agrupar)
            chart_type : tipo de visualização
            filters    : restrições de período / UF
            table      : tabela Gold mais adequada para o agrupamento detectado

        Retorna ChartParams com defaults razoáveis para campos não detectados
        (total_casos por ano_mes em bar chart, sem filtros).
        """
        query_lower = query.lower()
        params = ChartParams()

        # Métrica (coluna Y)
        if any(kw in query_lower for kw in ["mortalidade", "óbito", "obito", "morte"]):
            params.metric = "taxa_mortalidade"
            params.title  = "Taxa de Mortalidade SRAG"
        elif any(kw in query_lower for kw in ["uti", "internado", "internação", "internacao"]):
            params.metric = "taxa_uti"
            params.title  = "Taxa de Ocupacao UTI — SRAG"
        elif any(kw in query_lower for kw in ["vacinação", "vacinacao", "vacinado", "vacina"]):
            params.metric = "taxa_vacinacao"
            params.title  = "Taxa de Vacinacao — SRAG"
        else:
            params.metric = "total_casos"
            params.title  = "Casos SRAG"

        # Agrupamento (coluna X) e tabela fonte
        if any(kw in query_lower for kw in ["estado", "uf", "região", "regiao"]):
            params.group_by = "sg_uf"
            params.table    = "gold_metricas_geograficas"
            params.title   += " por Estado"
        elif any(kw in query_lower for kw in ["faixa etária", "faixa etaria", "idade", "etária", "etaria"]):
            params.group_by = "faixa_etaria"
            params.table    = "gold_metricas_demograficas"
            params.title   += " por Faixa Etaria"
        elif any(kw in query_lower for kw in ["semana", "semanal"]):
            params.group_by = "semana_epidemiologica"
            params.table    = "gold_metricas_temporais"
            params.title   += " por Semana Epidemiologica"
        elif any(kw in query_lower for kw in ["ano", "anual"]):
            params.group_by = "ano"
            params.table    = "gold_metricas_historicas"
            params.title   += " por Ano"
        else:
            params.group_by = "ano_mes"
            params.table    = "gold_metricas_temporais"
            params.title   += " por Mes"

        # Tipo de gráfico
        if any(kw in query_lower for kw in ["linha", "line", "evolução", "evolucao", "tendência", "tendencia", "série", "serie"]):
            params.chart_type = "line"
        elif any(kw in query_lower for kw in ["pizza", "pie", "proporção", "proporcao", "distribuição", "distribuicao"]):
            params.chart_type = "pie"
        else:
            params.chart_type = "bar"

        # Filtros — ano
        year_match = re.search(r'\b(20\d{2})\b', query)
        if year_match:
            params.filters["ano"]  = year_match.group(1)
            params.title          += f" ({year_match.group(1)})"

        # Filtros — UF
        uf_match = re.search(
            r'\b(AC|AL|AP|AM|BA|CE|DF|ES|GO|MA|MT|MS|MG|PA|PB|PR|PE|PI|RJ|RN|RS|RO|RR|SC|SP|SE|TO)\b',
            query.upper()
        )
        if uf_match:
            params.filters["sg_uf"] = uf_match.group(1)
            params.title           += f" — {uf_match.group(1)}"

        # Filtros — mês
        _MONTH_MAP = {
            "janeiro": "01", "fevereiro": "02", "marco": "03", "março": "03",
            "abril":   "04", "maio":      "05", "junho":    "06",
            "julho":   "07", "agosto":    "08", "setembro": "09",
            "outubro": "10", "novembro":  "11", "dezembro": "12",
        }
        for month_name, month_num in _MONTH_MAP.items():
            if month_name in query_lower:
                params.filters["mes"] = month_num
                break

        return params

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _determine_target_tables(self, intents: List[QueryIntent], query: str) -> List[str]:
        """
        Determina as tabelas Gold a consultar com base nas intenções detectadas.

        Todos os nomes usam o prefixo gold_ para corresponder aos nomes reais
        no catálogo Unity Catalog. Para nós RAG_ONLY, target_tables serve como
        rastreabilidade de fonte — não como tabela SQL direta.

        Palavras-chave na query podem adicionar tabelas complementares
        independentemente da intenção principal detectada.
        """
        tables      = set()
        query_lower = query.lower()

        intent_table_map = {
            QueryIntent.TEMPORAL:       ["gold_metricas_temporais", "gold_serie_diaria_30d"],
            QueryIntent.GEOGRAPHIC:     ["gold_metricas_geograficas"],
            QueryIntent.DEMOGRAPHIC:    ["gold_metricas_demograficas"],
            QueryIntent.FACTUAL:        ["gold_metricas_temporais"],
            QueryIntent.ANALYTICAL:     ["gold_rag_kpi_fatos", "gold_rag_dicionario_regras"],
            QueryIntent.EXPLANATORY:    ["gold_rag_dicionario_regras", "gold_rag_kpi_fatos"],
            QueryIntent.COMPARATIVE:    ["gold_metricas_temporais", "gold_metricas_geograficas"],
            QueryIntent.MIXED:          ["gold_metricas_temporais", "gold_metricas_geograficas",
                                         "gold_rag_kpi_fatos"],
            QueryIntent.CHART_REQUEST:  ["gold_metricas_temporais"],
            QueryIntent.REPORT_REQUEST: [
                "gold_metricas_temporais", "gold_serie_diaria_30d",
                "gold_metricas_geograficas", "gold_rag_kpi_fatos",
                "gold_rag_dicionario_regras",
            ],
        }

        for intent in intents:
            tables.update(intent_table_map.get(intent, ["gold_metricas_temporais"]))

        # Enriquecimento por palavras-chave independente do intent detectado
        if any(kw in query_lower for kw in ["estado", "uf", "sp", "rj", "região", "regiao"]):
            tables.add("gold_metricas_geograficas")
        if any(kw in query_lower for kw in ["idade", "idoso", "sexo", "faixa"]):
            tables.add("gold_metricas_demograficas")
        if any(kw in query_lower for kw in ["tendência", "tendencia", "evolução", "evolucao", "série", "serie"]):
            tables.add("gold_serie_diaria_30d")

        return list(tables)

    def _extract_sql_filters(self, query: str) -> Optional[Dict]:
        """
        Extrai filtros SQL da query.

        Filtros extraídos
        -----------------
        sg_uf
            Sigla de estado em maiúsculas (ex.: SP, RJ).
        ano
            Ano de 4 dígitos quando encontrado isoladamente.
        ano_mes
            Período YYYY-MM quando mês e ano coexistem.
        ano_inicio / ano_fim
            Intervalo de anos no formato "de YYYY a YYYY" ou "YYYY-YYYY".
        faixa_etaria
            Faixa etária canônica: "60+", "adulto", "crianca", "gestante".
        sexo
            "M" para masculino, "F" para feminino quando explicitamente mencionado.

        Retorna None quando nenhum filtro é detectado, para não adicionar
        cláusulas WHERE desnecessárias na query gerada.
        """
        filters     = {}
        query_lower = query.lower()

        # UF
        uf_match = re.search(
            r'\b(AC|AL|AP|AM|BA|CE|DF|ES|GO|MA|MT|MS|MG|PA|PB|PR|PE|PI|RJ|RN|RS|RO|RR|SC|SP|SE|TO)\b',
            query.upper()
        )
        if uf_match:
            filters["sg_uf"] = uf_match.group(1)

        # Mês — detectado primeiro para combinar com ano
        _MONTH_MAP = {
            "janeiro": "01", "fevereiro": "02", "marco": "03", "março": "03",
            "abril":   "04", "maio":      "05", "junho":    "06",
            "julho":   "07", "agosto":    "08", "setembro": "09",
            "outubro": "10", "novembro":  "11", "dezembro": "12",
        }
        detected_month = None
        for month_name, month_num in _MONTH_MAP.items():
            if month_name in query_lower:
                detected_month = month_num
                break

        # Intervalo de anos: "de 2023 a 2025" ou "2023-2025"
        range_match = re.search(
            r'\b(?:de\s+)?(20\d{2})\s*(?:a|até|ate|-)\s*(20\d{2})\b',
            query_lower
        )
        if range_match:
            filters["ano_inicio"] = range_match.group(1)
            filters["ano_fim"]    = range_match.group(2)
        else:
            # Ano isolado
            year_match = re.search(r'\b(20\d{2})\b', query)
            if year_match:
                ano = year_match.group(1)
                if detected_month:
                    filters["ano_mes"] = f"{ano}-{detected_month}"
                else:
                    filters["ano"] = ano

        # Faixa etária
        _FAIXA_MAP = {
            "idoso":    "60+",
            "60 anos":  "60+",
            "60+":      "60+",
            "criança":  "crianca",
            "crianca":  "crianca",
            "infantil": "crianca",
            "adulto":   "adulto",
            "gestante": "gestante",
            "puerpera": "puerpera",
            "puérpera": "puerpera",
        }
        for kw, canonical in _FAIXA_MAP.items():
            if kw in query_lower:
                filters["faixa_etaria"] = canonical
                break

        # Sexo
        if any(kw in query_lower for kw in ["masculino", "homem", "homens"]):
            filters["sexo"] = "M"
        elif any(kw in query_lower for kw in ["feminino", "mulher", "mulheres"]):
            filters["sexo"] = "F"

        return filters if filters else None

    def _determine_rag_type(self, intents: List[QueryIntent]) -> Optional[str]:
        """
        Determina o semantic_type para filtro no Vector Index.

        Os únicos valores indexados pelo GoldDocumentLoader são:
        kpi, regra, temporal, geographic, demographic.

        EXPLANATORY mapeado para "regra" — tabela gold_rag_dicionario_regras.
        ANALYTICAL retorna None — busca sem filtro para maior cobertura contextual.
        REPORT_REQUEST retorna None — requer contexto de múltiplos tipos.
        """
        intent_to_rag: Dict[QueryIntent, Optional[str]] = {
            QueryIntent.TEMPORAL:       "temporal",
            QueryIntent.GEOGRAPHIC:     "geographic",
            QueryIntent.DEMOGRAPHIC:    "demographic",
            QueryIntent.ANALYTICAL:     None,
            QueryIntent.EXPLANATORY:    "regra",
            QueryIntent.REPORT_REQUEST: None,
        }
        for intent in intents:
            rag_type = intent_to_rag.get(intent)
            if rag_type is not None:
                return rag_type
        return None

    def _calculate_confidence(self, intents: List[QueryIntent], query: str) -> float:
        """
        Calcula a confiança da classificação no intervalo [0.5, 0.95].

        Penaliza queries curtas (< 4 tokens) e listas longas de intenções (> 2),
        pois ambos indicam ambiguidade. Queries com palavras-chave inequívocas
        ou com pedido de relatório/gráfico recebem bônus.
        """
        base_confidence = 0.7

        if any(kw in query.lower() for kw in ["quantos", "qual", "ranking", "total"]):
            base_confidence += 0.2
        if QueryIntent.CHART_REQUEST in intents or QueryIntent.REPORT_REQUEST in intents:
            base_confidence += 0.15
        if len(query.split()) < 4:
            base_confidence -= 0.1
        if len(intents) > 2:
            base_confidence -= 0.15

        return max(0.5, min(0.95, base_confidence))

    def _generate_reasoning(
        self,
        intents:  List[QueryIntent],
        strategy: ExecutionStrategy,
        tables:   List[str],
    ) -> str:
        """Gera texto explicativo da decisão de roteamento para logs e debug."""
        intent_str = ", ".join([i.value for i in intents])
        tables_str = ", ".join(tables)
        return (
            f"Detectei intencao(oes): {intent_str}. "
            f"Estrategia selecionada: {strategy.value}. "
            f"Tabelas alvo: {tables_str}."
        )

    def explain_routing(self, query: str) -> Dict:
        """
        Explica a decisão de roteamento sem executar o pipeline.

        Retorna dict serializável com todos os campos do RoutingDecision,
        incluindo chart_params quando presente. Usado pelo orchestrator para
        debug e por testes unitários para validar decisões de roteamento.
        """
        decision = self.route(query)
        result = {
            "query":         query,
            "intent":        decision.intent.value,
            "strategy":      decision.strategy.value,
            "confidence":    decision.confidence,
            "reasoning":     decision.reasoning,
            "target_tables": decision.target_tables,
            "sql_filters":   decision.sql_filters,
            "rag_type":      decision.rag_semantic_type,
        }
        if decision.chart_params:
            result["chart_params"] = {
                "metric":     decision.chart_params.metric,
                "group_by":   decision.chart_params.group_by,
                "chart_type": decision.chart_params.chart_type,
                "title":      decision.chart_params.title,
                "filters":    decision.chart_params.filters,
                "table":      decision.chart_params.table,
            }
        return result