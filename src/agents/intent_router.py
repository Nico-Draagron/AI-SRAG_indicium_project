"""
Intent Router — Roteamento Inteligente SQL vs RAG vs Chart vs Report
====================================================================

Classifica a intenção da query do usuário e produz um RoutingDecision que
determina qual nó do grafo LangGraph será executado pelo SRAGOrchestrator.

Pipeline
--------
    Query -> IntentClassifier -> StrategySelector -> IntentRouter.route()
          -> RoutingDecision

Melhorias desta versão
----------------------
- QueryIntent.VISUALIZATION adicionado como alias semântico de CHART_REQUEST
  (necessário para o guard de generate_all_charts() no orchestrator).
- ChartParams expandido: chart_purpose, y_cols, series_col, year_col, top_n,
  value_format — o orchestrator pode usar esses campos para construir ChartSpec
  sem re-inferir tudo do zero.
- _extract_chart_params() refatorado com lógica modular por dimensão:
  _detect_metric(), _detect_dimension(), _detect_chart_purpose(),
  _detect_top_n(), _detect_temporal_mode(), _detect_y_cols().
- Padrões para ranking, sazonalidade, comparação entre anos e múltiplas taxas.
- StrategySelector: TEMPORAL + qualquer outro → HYBRID mantido; VISUALIZATION
  adicionado ao tier exclusivo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage


VERSION = "5.0.0"


# =============================================================================
# CONSTANTES SEMÂNTICAS COMPARTILHADAS
# =============================================================================

# Colunas que representam taxas percentuais
_RATE_COLS = frozenset({
    "taxa_mortalidade", "taxa_uti", "taxa_vacinacao", "taxa_hospitalizacao",
})

# Colunas de agrupamento por natureza
_GEO_COLS    = frozenset({"sg_uf", "municipio", "regiao"})
_DEMO_COLS   = frozenset({"faixa_etaria", "sexo_label"})
_ANNUAL_COLS = frozenset({"ano"})
_MONTHLY_COLS = frozenset({"ano_mes", "mes", "semana_epidemiologica"})

# Tabela canônica por agrupamento
_GROUP_TO_TABLE: Dict[str, str] = {
    "sg_uf":                  "gold_metricas_geograficas",
    "municipio":              "gold_metricas_geograficas",
    "faixa_etaria":           "gold_metricas_demograficas",
    "sexo_label":             "gold_metricas_demograficas",
    "ano":                    "gold_metricas_historicas",
    "ano_mes":                "gold_metricas_temporais",
    "semana_epidemiologica":  "gold_metricas_temporais",
}


# =============================================================================
# ENUMS
# =============================================================================

class QueryIntent(Enum):
    """Tipos de intenção reconhecidos pelo classificador."""
    FACTUAL        = "factual"
    ANALYTICAL     = "analytical"
    COMPARATIVE    = "comparative"
    TEMPORAL       = "temporal"
    GEOGRAPHIC     = "geographic"
    DEMOGRAPHIC    = "demographic"
    EXPLANATORY    = "explanatory"
    MIXED          = "mixed"
    CHART_REQUEST  = "chart_request"
    VISUALIZATION  = "visualization"   # alias semântico de CHART_REQUEST —
                                       # usado pelo orchestrator para o guard
                                       # de generate_all_charts().
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
    Parâmetros semânticos para geração de gráfico ad-hoc.

    Novos campos desta versão
    -------------------------
    chart_purpose : intenção analítica do gráfico.
        "trend"          — evolução temporal de uma métrica.
        "comparison"     — comparação entre categorias discretas.
        "ranking"        — top-N ordenado por valor.
        "distribution"   — distribuição de frequência (demográfica etc.).
        "seasonality"    — comparação de sazonalidade entre anos.
        "rate_comparison"— comparação de múltiplas taxas percentuais.
        "generic"        — fallback quando nenhum padrão é detectado.

    y_cols        : lista de métricas adicionais para gráficos multi-série
                    (ex.: ["taxa_mortalidade", "taxa_uti"]).
    series_col    : coluna que define as séries em gráficos agrupados
                    (ex.: "sexo_label" em grouped_bar).
    year_col      : coluna de ano para year_comparison ("ano").
                    Quando preenchido, group_by deve ser "mes" ou "semana".
    top_n         : limite de itens para ranking (ex.: 5, 10).
    value_format  : "number" | "percent" | "auto".
                    Derivado automaticamente se "auto" for mantido.
    """
    metric:        str            = "total_casos"
    group_by:      str            = "ano_mes"
    chart_type:    str            = "bar"
    chart_purpose: str            = "generic"
    title:         str            = "Grafico SRAG"
    filters:       Dict           = field(default_factory=dict)
    table:         str            = "gold_metricas_temporais"
    y_cols:        List[str]      = field(default_factory=list)
    series_col:    Optional[str]  = None
    year_col:      Optional[str]  = None
    top_n:         Optional[int]  = None
    value_format:  str            = "auto"


@dataclass
class RoutingDecision:
    """
    Decisão de roteamento produzida por IntentRouter.route().

    Garantias do contrato
    ---------------------
    - intent nunca é None; fallback é QueryIntent.FACTUAL.
    - target_tables contém apenas nomes reais do catálogo (prefixo gold_).
    - rag_semantic_type, quando presente, é um de: kpi, regra, temporal,
      geographic, demographic.
    - chart_params é não-None somente quando strategy == CHART.
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
    Classificador regex de intenção.

    Hierarquia de verificação
    -------------------------
    1. REPORT_REQUEST — prioridade máxima; retorna imediatamente.
    2. CHART_REQUEST  — segunda prioridade; retorna imediatamente.
    3. Demais intents — avaliados em paralelo.

    Padrões novos nesta versão
    --------------------------
    - COMPARATIVE: detecta ranking ("top N", "quais tiveram mais"),
      comparação entre anos específicos ("2023 e 2024") e mudança percentual.
    - TEMPORAL: detecta sazonalidade explícita.
    - GEOGRAPHIC: detecta pedidos de ranking geográfico.
    - EXPLANATORY: mantido restrito a formas metodológicas.
    """

    PATTERNS: Dict[QueryIntent, List[str]] = {
        QueryIntent.REPORT_REQUEST: [
            r'\b(relat[oó]rio|report)\b',
            r'\b(panorama|boletim|bulletin)\b',
            r'\b(resumo executivo)\b',
            r'\b(situa[cç][aã]o atual|cen[aá]rio epidemiol[oó]gico)\b',
            r'\b(an[aá]lise completa|avalia[cç][aã]o completa)\b',
            r'\b(vis[aã]o geral|overview|quadro epidemiol[oó]gico)\b',
        ],
        QueryIntent.CHART_REQUEST: [
            r'\b(ger[ae]|cri[ae]|plot[ae]|exib[ae]|visualiz[ae])\b.{0,60}\b(gr[aá]fico|chart|plot)\b',
            r'\b(mostre?)\b.{0,30}\b(gr[aá]fico|chart|plot)\b',
            r'^.{0,20}\b(gr[aá]fico|chart|plot)\b',
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
            r'\b(comparar?|compare|versus|vs|diferen[cç]a)\b',
            r'\b(maior|menor|melhor|pior)\b.*\b(que|do que)\b',
            r'\b(entre|e)\b.*\b(estados?|UFs?)\b',
            r'\b(compara[cç][aã]o|comparativo)\b',
            # Comparação entre anos específicos: "2022 e 2024", "2023 vs 2024"
            r'\b(20\d{2})\b.{0,15}\b(e|vs|versus|e|comparando)\b.{0,15}\b(20\d{2})\b',
            # Ranking explícito
            r'\b(top\s*\d+|top\s+cinco|top\s+dez|ranking|maiores?|menores?)\b',
            r'\b(quais?\s+(estados?|UFs?|regi[oõ]es?)\s+(tiveram|com\s+mais))\b',
            # Mudança percentual
            r'\b(varia[cç][aã]o|mudan[cç]a|cresceu|diminuiu|aumentou|caiu)\b',
        ],
        QueryIntent.TEMPORAL: [
            r'\b(tend[eê]ncia|evolu[cç][aã]o|crescimento)\b',
            r'\b([uú]ltimos?|pr[oó]ximos?|passados?)\b.*\b(meses?|anos?|dias?)\b',
            r'\b(temporal|ao longo|s[eé]rie|hist[oó]rico)\b',
            r'\b(mensal|anual|semanal|trimestral)\b',
            r'\b(sazonalidade|sazonal|inverno|ver[aã]o|pico sazonal)\b',
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
            r'\b(o que [eé]|o que s[aã]o|o que significa[nm]?)\b',
            r'\b(defin[ae]|defini[cç][aã]o|conceito de)\b',
            r'\b(significa[nm]?)\b',
            r'\b(expliqu[eé]|defina)\b',
            r'\b(como [eé] calculad[oa]|como s[aã]o calculad[oa]s?)\b',
            r'\b(como funciona|como [eé] definid[oa]|como [eé] obtid[oa])\b',
            r'\b(qual o crit[eé]rio|qual a metodologia|qual o denominador)\b',
            r'\b(qual a f[oó]rmula|qual o numerador|como [eé] feito o c[aá]lculo)\b',
            r'\b(metodologia|crit[eé]rio epidemiol[oó]gico)\b',
        ],
    }

    @staticmethod
    def classify(query: str) -> List[QueryIntent]:
        """
        Classifica a query. Retorna [FACTUAL] como fallback.

        REPORT_REQUEST e CHART_REQUEST retornam imediatamente quando detectados.
        Os demais intents são avaliados em paralelo sem prioridade entre si.
        """
        q = query.lower()

        for p in IntentClassifier.PATTERNS[QueryIntent.REPORT_REQUEST]:
            if re.search(p, q, re.IGNORECASE):
                return [QueryIntent.REPORT_REQUEST]

        for p in IntentClassifier.PATTERNS[QueryIntent.CHART_REQUEST]:
            if re.search(p, q, re.IGNORECASE):
                return [QueryIntent.CHART_REQUEST]

        detected: List[QueryIntent] = []
        skip = {QueryIntent.REPORT_REQUEST, QueryIntent.CHART_REQUEST}
        for intent, patterns in IntentClassifier.PATTERNS.items():
            if intent in skip:
                continue
            for p in patterns:
                if re.search(p, q, re.IGNORECASE):
                    if intent not in detected:
                        detected.append(intent)
                    break

        return detected if detected else [QueryIntent.FACTUAL]


# =============================================================================
# STRATEGY SELECTOR
# =============================================================================

class StrategySelector:
    """
    Seleciona ExecutionStrategy a partir das intenções classificadas.

    Hierarquia de dominância semântica (tiers)
    ------------------------------------------
    Tier 1 — exclusivos: REPORT_REQUEST → REPORT, CHART_REQUEST → CHART.
    Tier 2 — conceptual sem comparative → RAG_ONLY.
    Tier 2.5 — TEMPORAL combinado → HYBRID.
    Tier 3 — apenas estáticos (FACTUAL/GEOGRAPHIC/DEMOGRAPHIC) → SQL_ONLY.
    Tier 4 — TEMPORAL sozinho → SQL_ONLY.
    Tier 5 — COMPARATIVE ou misto restante → HYBRID.
    """

    _EXCLUSIVE: Dict[QueryIntent, ExecutionStrategy] = {
        QueryIntent.REPORT_REQUEST: ExecutionStrategy.REPORT,
        QueryIntent.CHART_REQUEST:  ExecutionStrategy.CHART,
        QueryIntent.VISUALIZATION:  ExecutionStrategy.CHART,
    }

    _CONCEPTUAL        = frozenset({QueryIntent.ANALYTICAL, QueryIntent.EXPLANATORY})
    _STRUCTURAL_STATIC = frozenset({
        QueryIntent.FACTUAL, QueryIntent.GEOGRAPHIC, QueryIntent.DEMOGRAPHIC,
    })

    @staticmethod
    def select(intents: List[QueryIntent]) -> ExecutionStrategy:
        if not intents:
            return ExecutionStrategy.SQL_ONLY

        intent_set = set(intents)

        for excl_intent, excl_strategy in StrategySelector._EXCLUSIVE.items():
            if excl_intent in intent_set:
                return excl_strategy

        has_conceptual  = bool(StrategySelector._CONCEPTUAL & intent_set)
        has_temporal    = QueryIntent.TEMPORAL    in intent_set
        has_comparative = QueryIntent.COMPARATIVE in intent_set
        has_static_only = bool(StrategySelector._STRUCTURAL_STATIC & intent_set)

        if has_conceptual and not has_comparative:
            return ExecutionStrategy.RAG_ONLY

        # FACTUAL sem conceptual → SQL_ONLY
        # Cobre: FACTUAL+TEMPORAL, FACTUAL+COMPARATIVE, FACTUAL+GEOGRAPHIC, etc.
        # (ex.: "Qual o total de casos por ano em 2023, 2024 e 2025?" é pura
        # consulta de dados mesmo que o regex detecte COMPARATIVE por mencionar
        # múltiplos anos)
        if QueryIntent.FACTUAL in intent_set and not has_conceptual:
            return ExecutionStrategy.SQL_ONLY

        if has_temporal and len(intent_set) > 1 and not has_conceptual:
            return ExecutionStrategy.HYBRID

        if has_static_only and not has_conceptual and not has_comparative and not has_temporal:
            return ExecutionStrategy.SQL_ONLY

        if not has_comparative and not has_conceptual and has_temporal and len(intent_set) == 1:
            return ExecutionStrategy.SQL_ONLY

        return ExecutionStrategy.HYBRID


# =============================================================================
# CHART PARAMS EXTRACTION — HELPERS PRIVADOS
# =============================================================================

def _detect_metric(q: str) -> Tuple[str, str, str]:
    """
    Detecta a métrica principal e retorna (metric_col, title_fragment, value_format).

    Prioridade: "total/número de casos" > mortalidade > UTI > vacinação > default.
    Isso evita que "gráfico de casos de SRAG com mortalidade e UTI" mapeie para
    taxa_mortalidade quando a intenção principal é total_casos.
    """
    _casos_kws = (
        "total de casos", "número de casos", "numero de casos",
        "quantidade de casos", "casos de srag", "casos registrados",
        "casos por ano", "casos por estado", "casos por faixa", "casos por mês",
    )
    if any(kw in q for kw in _casos_kws):
        return "total_casos", "Casos SRAG", "number"
    if any(kw in q for kw in ("mortalidade", "óbito", "obito", "morte")):
        return "taxa_mortalidade", "Taxa de Mortalidade SRAG", "percent"
    if any(kw in q for kw in ("uti", "internado", "internação", "internacao")):
        return "taxa_uti", "Taxa de Ocupação UTI — SRAG", "percent"
    if any(kw in q for kw in ("vacinação", "vacinacao", "vacinado", "vacina")):
        return "taxa_vacinacao", "Taxa de Vacinação — SRAG", "percent"
    return "total_casos", "Casos SRAG", "number"


def _detect_y_cols(q: str, primary_metric: str) -> List[str]:
    """
    Detecta métricas adicionais para gráficos multi-série.

    Retorna lista não-vazia apenas quando há ≥2 métricas explícitas na query
    (ex.: "mortalidade e UTI por ano"). O primary_metric é incluído na lista
    para que o caller possa usar y_cols diretamente como lista completa.
    """
    detected: List[str] = []
    _metric_kw_map: List[Tuple[str, str]] = [
        ("mortalidade",                        "taxa_mortalidade"),
        ("óbito",                              "taxa_mortalidade"),
        ("obito",                              "taxa_mortalidade"),
        ("uti",                                "taxa_uti"),
        ("internação",                         "taxa_uti"),
        ("internacao",                         "taxa_uti"),
        ("vacinação",                          "taxa_vacinacao"),
        ("vacinacao",                          "taxa_vacinacao"),
        ("vacina",                             "taxa_vacinacao"),
        ("total de casos",                     "total_casos"),
        ("número de casos",                    "total_casos"),
        ("numero de casos",                    "total_casos"),
    ]
    for kw, col in _metric_kw_map:
        if kw in q and col not in detected:
            detected.append(col)

    # Retorna y_cols somente quando há de fato múltiplas métricas
    if len(detected) >= 2:
        # Garante que primary_metric aparece primeiro
        if primary_metric in detected:
            detected.remove(primary_metric)
        return [primary_metric] + detected

    return []


def _detect_dimension(q: str) -> Tuple[str, str, str]:
    """
    Detecta a dimensão de agrupamento e retorna (group_by, table, title_fragment).

    Nota: "idade" é verificada com boundary (r'\\bidade\\b') para não casar
    dentro de "mortalidade" ou "vacinidade" — falso positivo frequente.
    """
    if any(kw in q for kw in ("estado", " uf ", "região", "regiao", "mapa")):
        return "sg_uf", "gold_metricas_geograficas", " por Estado"
    if (
        any(kw in q for kw in ("faixa etária", "faixa etaria", "etária", "etaria"))
        or re.search(r'\bidade\b', q)
    ):
        return "faixa_etaria", "gold_metricas_demograficas", " por Faixa Etária"
    if any(kw in q for kw in ("sexo", "masculino", "feminino", "gênero", "genero")):
        return "sexo_label", "gold_metricas_demograficas", " por Sexo"
    if any(kw in q for kw in ("semana", "semanal")):
        return "semana_epidemiologica", "gold_metricas_temporais", " por Semana Epidemiológica"
    if any(kw in q for kw in (" ano", "anual", "histórico", "historico", "por ano")):
        return "ano", "gold_metricas_historicas", " por Ano"
    # Default mensal
    return "ano_mes", "gold_metricas_temporais", " por Mês"


def _detect_top_n(q: str) -> Optional[int]:
    """
    Detecta pedido de ranking top-N.

    Padrões: "top 5", "top cinco", "5 maiores", "top-10", "principais 3".
    Retorna None quando não há pedido explícito de ranking.
    """
    _NUM_WORDS = {"um": 1, "dois": 2, "tres": 3, "três": 3, "quatro": 4,
                  "cinco": 5, "seis": 6, "sete": 7, "oito": 8,
                  "nove": 9, "dez": 10, "vinte": 20}

    # "top 5", "top-5", "top cinco"
    m = re.search(r'\btop[\s\-]*(\d+)\b', q)
    if m:
        return int(m.group(1))
    m = re.search(r'\btop[\s\-]*(' + '|'.join(_NUM_WORDS.keys()) + r')\b', q)
    if m:
        return _NUM_WORDS[m.group(1)]

    # "5 maiores", "principais 10"
    m = re.search(r'\b(\d+)\s*(maiores?|menores?|principais?)\b', q)
    if m:
        return int(m.group(1))
    m = re.search(r'\b(maiores?|menores?|principais?)\s*(\d+)\b', q)
    if m:
        return int(m.group(2))

    # "ranking" sem número → default 10
    if re.search(r'\branking\b', q):
        return 10

    return None


def _detect_temporal_mode(q: str) -> Tuple[str, Optional[str], str]:
    """
    Detecta como a dimensão temporal deve ser tratada.

    Retorna (group_by, year_col, chart_purpose).

    Três modos possíveis:
    1. Sazonalidade / comparação entre anos → group_by="mes", year_col="ano",
       purpose="seasonality". Detectado quando há dois anos distintos na query
       ou palavras como "sazonalidade", "mesmo mês", "entre anos".
    2. Evolução anual → group_by="ano", year_col=None, purpose="trend".
    3. Evolução mensal → group_by="ano_mes", year_col=None, purpose="trend".
    """
    # Dois anos explícitos → sazonalidade
    years = re.findall(r'\b(20\d{2})\b', q)
    if len(set(years)) >= 2:
        return "mes", "ano", "seasonality"

    # Palavras de sazonalidade
    if any(kw in q for kw in ("sazonalidade", "sazonal", "mesmo mês", "mesmo mes",
                               "entre anos", "comparar anos", "pico sazonal")):
        return "mes", "ano", "seasonality"

    # Evolução anual
    if any(kw in q for kw in (" ano", "anual", "histórico", "historico",
                               "por ano", "cada ano")):
        return "ano", None, "trend"

    # Default mensal
    return "ano_mes", None, "trend"


def _detect_chart_purpose(
    q:           str,
    group_by:    str,
    metric:      str,
    top_n:       Optional[int],
    year_col:    Optional[str],
    y_cols:      List[str],
) -> str:
    """
    Determina o chart_purpose baseado nos slots já detectados.

    Hierarquia:
    1. top_n presente → "ranking"
    2. year_col presente → "seasonality"
    3. y_cols com ≥2 taxas → "rate_comparison"
    4. group_by demográfico/geográfico → "distribution" ou "comparison"
    5. group_by temporal → "trend"
    6. Palavras de comparação → "comparison"
    7. Default → "generic"
    """
    if top_n is not None:
        return "ranking"
    if year_col is not None:
        return "seasonality"
    if len(y_cols) >= 2 and all(c in _RATE_COLS for c in y_cols):
        return "rate_comparison"
    if group_by in _DEMO_COLS:
        return "distribution"
    if group_by in _GEO_COLS:
        # Ranking geográfico implícito ("quais estados tiveram mais...")
        if re.search(r'\b(mais|maiores?|menores?|ranking)\b', q):
            return "ranking"
        return "comparison"
    if group_by in _ANNUAL_COLS | _MONTHLY_COLS:
        return "trend"
    if re.search(r'\b(comparar?|compare|versus|vs|compara[cç][aã]o)\b', q):
        return "comparison"
    return "generic"


def _detect_chart_type(
    q:             str,
    chart_purpose: str,
    group_by:      str,
    year_col:      Optional[str],
) -> str:
    """
    Sugere o chart_type baseado no purpose e na dimensão.

    O orchestrator pode sobrescrever via _resolve_chart_spec(), mas a sugestão
    permite que chamadores que não usam ChartSpec também recebam um tipo razoável.
    """
    # Pedido explícito do usuário prevalece
    if any(kw in q for kw in ("linha", "line", "evolução", "evolucao",
                               "tendência", "tendencia", "série", "serie")):
        return "line"
    if any(kw in q for kw in ("pizza", "pie", "proporção", "proporcao")):
        return "pie"
    if any(kw in q for kw in ("área", "area", "preenchid")):
        return "area"

    # Inferência por purpose / dimensão
    if chart_purpose == "ranking":
        return "top_n"
    if chart_purpose == "seasonality" or year_col:
        return "year_comparison"
    if chart_purpose == "rate_comparison":
        return "bar"
    if chart_purpose == "distribution":
        return "bar"
    if chart_purpose == "trend":
        return "area" if group_by in _MONTHLY_COLS else "bar"

    return "bar"


# =============================================================================
# INTENT ROUTER
# =============================================================================

class IntentRouter:
    """
    Roteador principal — produz RoutingDecision a partir de uma query.

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

    Contrato
    --------
    - intent nunca None (fallback: FACTUAL).
    - target_tables com prefixo gold_.
    - rag_semantic_type em {kpi, regra, temporal, geographic, demographic} ou None.
    - chart_params preenchido somente quando strategy == CHART.
    - requires_synthesis True para HYBRID e REPORT.
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
        intents = (
            self._classify_with_llm(query)
            if self.use_llm and self.llm
            else self.classifier.classify(query)
        )

        strategy      = self.selector.select(intents)
        target_tables = self._determine_target_tables(intents, query)
        sql_filters   = (
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
            requires_synthesis = strategy in (ExecutionStrategy.HYBRID,
                                              ExecutionStrategy.REPORT),
            chart_params       = chart_params,
        )

    # =========================================================================
    # CHART PARAMS
    # =========================================================================

    def _extract_chart_params(self, query: str) -> ChartParams:
        """
        Extrai ChartParams semânticos usando helpers modulares.

        Fluxo de extração
        -----------------
        1. Detecta métrica principal e value_format.
        2. Detecta métricas adicionais (y_cols) — só preenchido quando ≥2.
        3. Detecta dimensão de agrupamento (group_by, table).
        4. Para dimensão temporal, detecta modo (trend vs seasonality vs anual).
        5. Detecta top_n para ranking.
        6. Determina chart_purpose com base em todos os slots.
        7. Sugere chart_type.
        8. Extrai filtros (ano, UF, mês).
        9. Preenche series_col quando fizer sentido.
        10. Monta título.
        """
        q = query.lower()

        # ── 1. Métrica ────────────────────────────────────────────────────────
        metric, title_base, value_format = _detect_metric(q)

        # ── 2. Métricas adicionais ────────────────────────────────────────────
        y_cols = _detect_y_cols(q, metric)

        # ── 3. Dimensão ───────────────────────────────────────────────────────
        group_by, table, title_dim = _detect_dimension(q)

        # ── 4. Modo temporal (sobrescreve group_by/table para dimensão temporal)
        year_col: Optional[str] = None
        if group_by in (_ANNUAL_COLS | _MONTHLY_COLS | {"ano_mes"}):
            group_by, year_col, _ = _detect_temporal_mode(q)
            table = _GROUP_TO_TABLE.get(group_by, table)
            # Sazonalidade precisa de histórico para expor a coluna 'ano'
            if year_col == "ano":
                table = "gold_metricas_historicas"

        # ── 5. Top-N ──────────────────────────────────────────────────────────
        top_n = _detect_top_n(q)

        # ── 6. Chart purpose ──────────────────────────────────────────────────
        chart_purpose = _detect_chart_purpose(q, group_by, metric, top_n, year_col, y_cols)

        # ── 7. Chart type ─────────────────────────────────────────────────────
        chart_type = _detect_chart_type(q, chart_purpose, group_by, year_col)

        # ── 8. Filtros ────────────────────────────────────────────────────────
        filters: Dict = {}
        years = re.findall(r'(20\d{2})', query)
        if len(set(years)) == 1:
            filters["ano"] = years[0]
        elif len(set(years)) >= 2:
            sorted_y = sorted(set(years))
            filters["ano_inicio"] = sorted_y[0]
            filters["ano_fim"]    = sorted_y[-1]

        uf_match = re.search(
            r'(AC|AL|AP|AM|BA|CE|DF|ES|GO|MA|MT|MS|MG|PA|PB|PR|PE|PI|'
            r'RJ|RN|RS|RO|RR|SC|SP|SE|TO)',
            query.upper(),
        )
        if uf_match:
            filters["sg_uf"] = uf_match.group(1)

        _MONTH_MAP = {
            "janeiro": "01", "fevereiro": "02", "marco": "03", "março": "03",
            "abril": "04",   "maio": "05",      "junho": "06",
            "julho": "07",   "agosto": "08",    "setembro": "09",
            "outubro": "10", "novembro": "11",  "dezembro": "12",
        }
        for month_name, month_num in _MONTH_MAP.items():
            if month_name in q:
                filters["mes"] = month_num
                break

        # ── 9. series_col ─────────────────────────────────────────────────────
        series_col: Optional[str] = None

        # Sazonalidade / comparação entre anos
        if year_col == "ano":
            series_col = "ano"

        # Comparação agrupada por sexo
        elif group_by == "sexo_label":
            series_col = "sexo_label"

        # Para múltiplas taxas, o orchestrator pode resolver via y_cols.
        elif len(y_cols) >= 2:
            series_col = None

        # ── 10. value_format / título ─────────────────────────────────────────
        if y_cols and all(c in _RATE_COLS for c in y_cols):
            value_format = "percent"
        elif metric in _RATE_COLS:
            value_format = "percent"
        elif value_format == "auto":
            value_format = "number"

        title = title_base + title_dim
        if filters.get("ano"):
            title += f" ({filters['ano']})"
        elif filters.get("ano_inicio") and filters.get("ano_fim"):
            title += f" ({filters['ano_inicio']}–{filters['ano_fim']})"

        if filters.get("sg_uf"):
            title += f" — {filters['sg_uf']}"

        if top_n:
            title = f"Top {top_n} — " + title

        return ChartParams(
            metric        = metric,
            group_by      = group_by,
            chart_type    = chart_type,
            chart_purpose = chart_purpose,
            title         = title,
            filters       = filters,
            table         = table,
            y_cols        = y_cols,
            series_col    = series_col,
            year_col      = year_col,
            top_n         = top_n,
            value_format  = value_format,
        )

    # =========================================================================
    # HELPERS
    # =========================================================================

    @staticmethod
    def _get_primary_intent(
        intents:  List[QueryIntent],
        strategy: ExecutionStrategy,
    ) -> QueryIntent:
        """
        Deriva a intenção primária a partir da estratégia selecionada.

        Evita que MIXED seja reportado quando há uma intenção dominante clara
        (ex.: EXPLANATORY+FACTUAL → RAG_ONLY → retorna EXPLANATORY).
        """
        if strategy == ExecutionStrategy.REPORT:
            return QueryIntent.REPORT_REQUEST
        if strategy == ExecutionStrategy.CHART:
            return QueryIntent.VISUALIZATION    # alias semântico de CHART_REQUEST

        if strategy == ExecutionStrategy.RAG_ONLY:
            for intent in intents:
                if intent in (QueryIntent.EXPLANATORY, QueryIntent.ANALYTICAL):
                    return intent
            return intents[0] if intents else QueryIntent.FACTUAL

        if strategy == ExecutionStrategy.SQL_ONLY:
            return intents[0] if intents else QueryIntent.FACTUAL

        return QueryIntent.MIXED if len(intents) > 1 else (
            intents[0] if intents else QueryIntent.FACTUAL
        )

    def _classify_with_llm(self, query: str) -> List[QueryIntent]:
        """Classificação via LLM — usado quando use_llm_classification=True."""
        prompt = f"""Classifique a intenção desta query sobre SRAG:

Query: "{query}"

Intenções possíveis:
- FACTUAL: perguntas sobre números objetivos (quantos, qual, quanto)
- ANALYTICAL: causas e impactos (por que, motivo, impacto)
- COMPARATIVE: comparações entre períodos, estados ou grupos
- TEMPORAL: tendências, séries temporais, sazonalidade
- GEOGRAPHIC: por estado, região ou UF
- DEMOGRAPHIC: por faixa etária, sexo ou comorbidade
- EXPLANATORY: definições e metodologia de cálculo
- CHART_REQUEST: pedido explícito de gráfico isolado
- REPORT_REQUEST: relatório epidemiológico completo

Responda APENAS com as intenções detectadas separadas por vírgula.
"""
        response     = self.llm.invoke([HumanMessage(content=prompt)])
        intent_names = [i.strip() for i in response.content.split(",")]
        intents: List[QueryIntent] = []
        for name in intent_names:
            try:
                intents.append(QueryIntent[name.strip().upper()])
            except KeyError:
                print(f"[intent_router] aviso: intent '{name}' não reconhecido — ignorado")
        return intents if intents else [QueryIntent.FACTUAL]

    def _determine_target_tables(
        self, intents: List[QueryIntent], query: str
    ) -> List[str]:
        tables: set = set()
        q = query.lower()

        _intent_table_map: Dict[QueryIntent, List[str]] = {
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
            QueryIntent.VISUALIZATION:  ["gold_metricas_temporais"],
            QueryIntent.REPORT_REQUEST: [
                "gold_metricas_temporais", "gold_serie_diaria_30d",
                "gold_metricas_geograficas", "gold_rag_kpi_fatos",
                "gold_rag_dicionario_regras",
            ],
        }

        for intent in intents:
            tables.update(_intent_table_map.get(intent, ["gold_metricas_temporais"]))

        # Enriquecimento por palavras-chave
        if any(kw in q for kw in ("estado", " uf ", "sp", "rj", "região", "regiao")):
            tables.add("gold_metricas_geograficas")
        if any(kw in q for kw in ("idade", "idoso", "sexo", "faixa")):
            tables.add("gold_metricas_demograficas")
        if any(kw in q for kw in ("tendência", "tendencia", "evolução", "evolucao",
                                   "série", "serie")):
            tables.add("gold_serie_diaria_30d")
        if any(kw in q for kw in ("ano", "anual", "histórico", "historico")):
            tables.add("gold_metricas_historicas")

        return list(tables)

    def _extract_sql_filters(self, query: str) -> Optional[Dict]:
        """
        Extrai filtros SQL da query.

        Filtros possíveis: sg_uf, ano, ano_mes, ano_inicio/ano_fim,
        faixa_etaria (canônica), sexo.
        """
        filters: Dict = {}
        q = query.lower()

        uf_match = re.search(
            r'\b(AC|AL|AP|AM|BA|CE|DF|ES|GO|MA|MT|MS|MG|PA|PB|PR|PE|PI|'
            r'RJ|RN|RS|RO|RR|SC|SP|SE|TO)\b',
            query.upper(),
        )
        if uf_match:
            filters["sg_uf"] = uf_match.group(1)

        _MONTH_MAP = {
            "janeiro": "01", "fevereiro": "02", "marco": "03", "março": "03",
            "abril": "04",   "maio": "05",      "junho": "06",
            "julho": "07",   "agosto": "08",    "setembro": "09",
            "outubro": "10", "novembro": "11",  "dezembro": "12",
        }
        detected_month = None
        for name, num in _MONTH_MAP.items():
            if name in q:
                detected_month = num
                break

        range_match = re.search(
            r'\b(?:de\s+)?(20\d{2})\s*(?:a|até|ate|-)\s*(20\d{2})\b', q
        )
        if range_match:
            filters["ano_inicio"] = range_match.group(1)
            filters["ano_fim"]    = range_match.group(2)
        else:
            year_match = re.search(r'\b(20\d{2})\b', query)
            if year_match:
                ano = year_match.group(1)
                filters["ano_mes" if detected_month else "ano"] = (
                    f"{ano}-{detected_month}" if detected_month else ano
                )

        _FAIXA_MAP = {
            "idoso": "60+", "60 anos": "60+", "60+": "60+",
            "criança": "crianca", "crianca": "crianca", "infantil": "crianca",
            "adulto": "adulto", "gestante": "gestante",
            "puerpera": "puerpera", "puérpera": "puerpera",
        }
        for kw, canonical in _FAIXA_MAP.items():
            if kw in q:
                filters["faixa_etaria"] = canonical
                break

        if any(kw in q for kw in ("masculino", "homem", "homens")):
            filters["sexo"] = "M"
        elif any(kw in q for kw in ("feminino", "mulher", "mulheres")):
            filters["sexo"] = "F"

        return filters if filters else None

    def _determine_rag_type(self, intents: List[QueryIntent]) -> Optional[str]:
        """
        Mapeia intenção para semantic_type do Vector Index.

        Tipos indexados: kpi, regra, temporal, geographic, demographic.
        ANALYTICAL retorna None — busca sem filtro para maior cobertura.
        """
        _map: Dict[QueryIntent, Optional[str]] = {
            QueryIntent.TEMPORAL:       "temporal",
            QueryIntent.GEOGRAPHIC:     "geographic",
            QueryIntent.DEMOGRAPHIC:    "demographic",
            QueryIntent.ANALYTICAL:     None,
            QueryIntent.EXPLANATORY:    "regra",
            QueryIntent.REPORT_REQUEST: None,
        }
        for intent in intents:
            rag_type = _map.get(intent)
            if rag_type is not None:
                return rag_type
        return None

    def _calculate_confidence(
        self, intents: List[QueryIntent], query: str
    ) -> float:
        base = 0.7
        q    = query.lower()
        if any(kw in q for kw in ("quantos", "qual", "ranking", "total")):
            base += 0.2
        if QueryIntent.CHART_REQUEST  in intents or \
           QueryIntent.REPORT_REQUEST in intents or \
           QueryIntent.VISUALIZATION  in intents:
            base += 0.15
        if len(query.split()) < 4:
            base -= 0.1
        if len(intents) > 2:
            base -= 0.15
        return max(0.5, min(0.95, base))

    def _generate_reasoning(
        self,
        intents:  List[QueryIntent],
        strategy: ExecutionStrategy,
        tables:   List[str],
    ) -> str:
        return (
            f"Intencao(oes): {', '.join(i.value for i in intents)}. "
            f"Estrategia: {strategy.value}. "
            f"Tabelas: {', '.join(tables)}."
        )

    def explain_routing(self, query: str) -> Dict:
        """Explica a decisão de roteamento sem executar o pipeline."""
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
            cp = decision.chart_params
            result["chart_params"] = {
                "metric":        cp.metric,
                "group_by":      cp.group_by,
                "chart_type":    cp.chart_type,
                "chart_purpose": cp.chart_purpose,
                "title":         cp.title,
                "filters":       cp.filters,
                "table":         cp.table,
                "y_cols":        cp.y_cols,
                "series_col":    cp.series_col,
                "year_col":      cp.year_col,
                "top_n":         cp.top_n,
                "value_format":  cp.value_format,
            }
        return result