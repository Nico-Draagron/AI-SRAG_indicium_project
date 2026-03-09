"""
Orchestrator — Agente SRAG com LangGraph
=========================================

Pipeline
--------
    Query -> route -> execute_(sql|rag|hybrid|chart|report) -> synthesize -> resposta

Mudanças principais desta versão
---------------------------------
- ChartSpec: estrutura intermediária que codifica a semântica do gráfico antes
  de chamar o ChartTool. Evita lógica espalhada e contexto insuficiente.
- _resolve_chart_spec(): heurísticas semânticas centralizadas (ranking, taxas,
  temporal, demográfico, comparação anual, sazonalidade).
- _dispatch_chart_spec(): mapeia ChartSpec para o método ChartTool correto.
- Anti-padrão corrigido: ano nunca é colapsado em x_col="indicador".
- _try_user_specific_query(): retry com SQL simplificada + fallback por templates.
- Síntese reestruturada: FATO / INTERPRETAÇÃO / LIMITAÇÃO sem vieses hardcoded.
- Avisos de qualidade gerados dinamicamente a partir dos dados reais.
- resolved_chart_spec no estado e no payload público: expõe os campos ricos do
  ChartSpec (chart_purpose, y_cols, series_col, year_col, top_n, value_format)
  que ChartParams do router não carrega. Melhora debug, auditoria e transparência.
- _execute_report_node refatorado: pipeline de relatório com payload canônico,
  rastreabilidade de blocos (report_block_status) e integração com ReportGenerator.
- _build_report_payload(): normaliza e valida insumos antes de chamar o gerador,
  preservando hierarquia SQL > notícias > RAG.
- _assess_report_blocks(): classifica cada bloco como ok / degraded / absent —
  auditável independente do resultado final.
- report_generator como parâmetro opcional do construtor: quando presente,
  execute_report_node delega diretamente ao ReportGenerator; quando ausente,
  payload mais rico fica disponível para _synthesize_node.
"""

import re
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph

from src.agents.intent_router import (
    ChartParams,
    ExecutionStrategy,
    IntentRouter,
    QueryIntent,
    RoutingDecision,
)
from src.rag.rag_chain import SRAGChain
from src.tools.sql_tool import GoldSQLTool
from src.utils.audit import AuditEvent, AuditLogger, EventStatus
from src.utils.exceptions import OrchestratorError

try:
    from src.tools.report_generator import ReportGenerator
    _REPORT_GENERATOR_AVAILABLE = True
except ImportError:
    ReportGenerator = None  # type: ignore[assignment,misc]
    _REPORT_GENERATOR_AVAILABLE = False


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

CATALOG = "dbx_srag_lab"
SCHEMA  = "gold"
VERSION = "5.1.0"

_ALLOWED_TABLES_FOR_USER_SQL = {
    "gold_metricas_temporais":    "ano_mes, total_casos, total_obitos, casos_com_desfecho, total_internados, total_uti, total_vacinados, casos_com_info_vacina",
    "gold_metricas_geograficas":  "sg_uf, ano_mes, total_casos, total_obitos, casos_com_desfecho",
    "gold_metricas_demograficas": "faixa_etaria, ano_mes, total_casos, total_obitos",
    "gold_serie_diaria_30d":      "dt_sintomas, total_casos",
    "gold_metricas_historicas":   "ano, total_casos, total_obitos, casos_com_desfecho, total_internados, total_uti, total_vacinados, casos_com_info_vacina",
}

_RATE_COLS   = {"taxa_mortalidade", "taxa_uti", "taxa_vacinacao", "taxa_hospitalizacao"}
_GEO_COLS    = {"sg_uf", "municipio", "regiao"}
_DEMO_COLS   = {"faixa_etaria", "faixa_etaria_label", "sexo_label"}
_TEMPORAL_ANNUAL = {"ano"}
_TEMPORAL_MONTHLY = {"ano_mes", "mes", "semana_epidemiologica"}

_ALLOWED_METRICS = {
    "total_casos", "taxa_mortalidade", "taxa_uti", "taxa_vacinacao",
    "total_obitos", "total_internados", "total_uti", "total_vacinados",
}
_ALLOWED_GROUP_BY = {
    "ano_mes", "sg_uf", "faixa_etaria", "semana_epidemiologica", "ano",
}
_ALLOWED_TABLES = {
    "gold_metricas_temporais", "gold_metricas_geograficas",
    "gold_metricas_demograficas", "gold_metricas_historicas",
}


# =============================================================================
# CHART SPEC — ESTRUTURA SEMÂNTICA INTERMEDIÁRIA
# =============================================================================

@dataclass
class ChartSpec:
    """
    Especificação semântica completa de um gráfico.

    Separa a lógica de 'o que mostrar' da lógica de 'como renderizar'.
    O orchestrator monta o ChartSpec; o _dispatch_chart_spec() o traduz
    para a chamada correta do ChartTool.

    Campos
    ------
    chart_type    : tipo canônico do ChartTool.
    title         : título completo do gráfico.
    subtitle      : período ou fonte dos dados.
    x_col         : dimensão principal do eixo X (NUNCA "indicador" para dados anuais).
    y_col         : métrica principal.
    y_cols        : métricas adicionais para comparativos.
    year_col      : coluna de ano para year_comparison (x_col seria mes/semana).
    series_col    : coluna que gera séries no grouped_bar.
    chart_purpose : intenção semântica para logging e decisão final.
    value_format  : "number" | "percent" | "auto".
    orientation   : "auto" | "h" | "v".
    sort_order    : "value_desc" | "value_asc" | "category" | "none".
    top_n         : limitar ranking a N itens.
    filters_applied: filtros já aplicados na SQL (para contexto na síntese).
    """
    chart_type:      str
    title:           str
    x_col:           str
    y_col:           str
    subtitle:        str                = ""
    y_cols:          List[str]          = field(default_factory=list)
    year_col:        Optional[str]      = None
    series_col:      Optional[str]      = None
    chart_purpose:   str                = "generic"
    value_format:    str                = "auto"
    orientation:     str                = "auto"
    sort_order:      str                = "none"
    # Valores válidos: "value_desc" | "value_asc" | "category" | "none"
    # "value_desc" e "value_asc" são normalizados para "value" no dispatch
    # (ChartTool infere direção pela orientação da barra).
    top_n:           Optional[int]      = None
    filters_applied: Dict[str, Any]     = field(default_factory=dict)


# =============================================================================
# STATE
# =============================================================================

class AgentState(TypedDict):
    messages:             List[BaseMessage]
    user_query:           str
    routing_decision:     Optional[RoutingDecision]
    sql_results:          Optional[Dict]
    rag_results:          Optional[Dict]
    news_results:         Optional[Dict]
    chart_paths:          Optional[List[str]]
    ad_hoc_chart_path:    Optional[str]
    geographic_data:      Optional[Dict]
    mandatory_metrics:    Optional[Dict]
    final_answer:         Optional[str]
    sources:              List[str]
    errors:               List[str]
    # ChartSpec resolvido — exposto no payload público para debug e auditoria.
    # Contém campos que o ChartParams do router não carrega:
    # chart_purpose, y_cols, series_col, year_col, top_n, value_format.
    resolved_chart_spec:  Optional[Dict]
    # Rastreabilidade do pipeline de relatório: quais blocos foram produzidos.
    report_block_status:  Optional[Dict]


# =============================================================================
# ORCHESTRATOR
# =============================================================================

class SRAGOrchestrator:
    """
    Grafo LangGraph do agente SRAG.

    Parâmetros
    ----------
    spark           : SparkSession — obrigatória para GoldSQLTool e ChartTool.
    llm             : BaseChatModel — síntese e roteamento opcional por LLM.
    audit_logger    : AuditLogger — None desabilita persistência de eventos.
    rag_chain       : SRAGChain — ausência desabilita nós RAG e REPORT.
    use_llm_routing : usa LLM para classificar intenção (mais preciso, mais lento).
    web_search_tool : Tavily — ausência desabilita etapa de notícias.
    chart_tool      : ChartTool — obrigatório para execute_chart.
    catalog / schema: identificadores Unity Catalog.
    use_openai      : flag de provider para logging. None = inferido pelo tipo do llm.
    """

    def __init__(
        self,
        spark,
        llm:             BaseChatModel,
        audit_logger:    Optional[AuditLogger] = None,
        rag_chain:       Optional[SRAGChain]   = None,
        use_llm_routing: bool                  = False,
        web_search_tool                        = None,
        chart_tool                             = None,
        report_generator                       = None,
        catalog:         str                   = CATALOG,
        schema:          str                   = SCHEMA,
        use_openai:      Optional[bool]        = None,
    ):
        self.spark      = spark
        self.llm        = llm
        self.audit      = audit_logger
        self.rag_chain  = rag_chain
        self.catalog    = catalog
        self.schema     = schema

        cls_name = type(llm).__name__.lower()
        self.use_openai = use_openai if use_openai is not None else (
            "openai" in cls_name or "azure" in cls_name
        )

        self.sql_tool         = GoldSQLTool(spark, audit_logger)
        self.web_search_tool  = web_search_tool
        self.chart_tool       = chart_tool
        self.report_generator = report_generator
        self.router           = IntentRouter(
            use_llm_classification=use_llm_routing,
            llm=llm if use_llm_routing else None,
        )
        self.graph = self._build_graph()

        if self.audit:
            self.audit.log_event(
                AuditEvent.ORCHESTRATOR_INITIALIZED,
                {
                    "has_rag":              rag_chain        is not None,
                    "has_web_search":       web_search_tool  is not None,
                    "has_charts":           chart_tool       is not None,
                    "has_report_generator": report_generator is not None,
                    "catalog":        self.catalog,
                    "schema":         self.schema,
                    "version":        VERSION,
                    "llm_provider":   "openai" if self.use_openai else "databricks",
                    "llm_class":      type(llm).__name__,
                },
                EventStatus.INFO,
            )

    # =========================================================================
    # GRAPH
    # =========================================================================

    def _build_graph(self) -> StateGraph:
        wf = StateGraph(AgentState)
        for name, fn in [
            ("route",          self._route_node),
            ("execute_sql",    self._execute_sql_node),
            ("execute_rag",    self._execute_rag_node),
            ("execute_hybrid", self._execute_hybrid_node),
            ("execute_chart",  self._execute_chart_node),
            ("execute_report", self._execute_report_node),
            ("synthesize",     self._synthesize_node),
        ]:
            wf.add_node(name, fn)

        wf.set_entry_point("route")
        wf.add_conditional_edges(
            "route", self._route_to_execution,
            {"sql": "execute_sql", "rag": "execute_rag", "hybrid": "execute_hybrid",
             "chart": "execute_chart", "report": "execute_report"},
        )
        for node in ("execute_sql", "execute_rag", "execute_hybrid",
                     "execute_chart", "execute_report"):
            wf.add_edge(node, "synthesize")
        wf.add_edge("synthesize", END)
        return wf.compile()

    # =========================================================================
    # NÓ 1 — ROTEAMENTO
    # =========================================================================

    def _route_node(self, state: AgentState) -> AgentState:
        try:
            if self.audit:
                self.audit.log_event(AuditEvent.NODE_START,
                    {"node": "route", "query": state["user_query"]}, EventStatus.INFO)

            decision = self.router.route(state["user_query"])
            state["routing_decision"] = decision
            state["messages"].append(
                AIMessage(content=f"Rota: {decision.strategy.value} | {decision.reasoning}")
            )
            if self.audit:
                self.audit.log_event(AuditEvent.QUERY_ANALYZED, {
                    "strategy": decision.strategy.value, "confidence": decision.confidence,
                    "intent": decision.intent.value, "rag_type": decision.rag_semantic_type,
                }, EventStatus.SUCCESS)

        except Exception as e:
            state["errors"].append(f"Routing error: {e}")
            state["routing_decision"] = RoutingDecision(
                intent=QueryIntent.FACTUAL, strategy=ExecutionStrategy.SQL_ONLY,
                confidence=0.5, reasoning="Fallback por erro no routing",
                target_tables=["gold_metricas_temporais"],
            )
        return state

    def _route_to_execution(self, state: AgentState) -> str:
        mapping = {
            ExecutionStrategy.SQL_ONLY: "sql",
            ExecutionStrategy.RAG_ONLY: "rag",
            ExecutionStrategy.CHART:    "chart",
            ExecutionStrategy.REPORT:   "report",
        }
        return mapping.get(state["routing_decision"].strategy, "hybrid")

    # =========================================================================
    # NÓ 2A — SQL
    # =========================================================================

    def _execute_sql_node(self, state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        try:
            if self.audit:
                self.audit.log_event(AuditEvent.NODE_START,
                    {"node": "execute_sql"}, EventStatus.INFO)

            print("\n[execute_sql] query específica do usuário...")
            user_query_result = self._try_user_specific_query(state["user_query"])

            print("[execute_sql] métricas obrigatórias...")
            state["mandatory_metrics"] = self._calculate_mandatory_metrics()

            print("[execute_sql] dados geográficos...")
            state["geographic_data"] = self._calculate_geographic_data()

            if self.chart_tool:
                # Gráficos padrão apenas quando a intenção inclui visualização.
                # Em consultas factuais (SQL_ONLY sem CHART), gerar 5 gráficos
                # sem necessidade é work amplification evitável.
                rd = state.get("routing_decision")
                wants_charts = (
                    rd is None                                          # sem decisão → conservador
                    or rd.strategy == ExecutionStrategy.REPORT         # relatório sempre
                    or rd.intent   == QueryIntent.VISUALIZATION        # intenção explícita
                )
                if wants_charts:
                    print("[execute_sql] gráficos padrão...")
                    try:
                        paths = self.chart_tool.generate_all_charts()
                        # generate_all_charts() retorna List[str].
                        # Guard aceita str ou dict{"path":...} para compatibilidade futura.
                        state["chart_paths"] = [
                            (p["path"] if isinstance(p, dict) else p)
                            for p in (paths or [])
                            if (isinstance(p, dict) and p.get("path"))
                            or (isinstance(p, str) and p)
                        ]
                    except Exception as ce:
                        print(f"[execute_sql] aviso: gráficos — {ce}")
                        state["chart_paths"] = []
                else:
                    print(f"[execute_sql] gráficos padrão ignorados "
                          f"(strategy={rd.strategy.value if rd else '?'}, "
                          f"intent={rd.intent.value if rd else '?'})")
                    state["chart_paths"] = []
            else:
                state["chart_paths"] = []

            if self.web_search_tool:
                print("[execute_sql] notícias...")
                try:
                    state["news_results"] = self.web_search_tool.search_news(
                        query="SRAG síndrome respiratória aguda grave Brasil",
                        max_results=5,
                    )
                except Exception as ne:
                    print(f"[execute_sql] aviso: notícias — {ne}")
                    state["news_results"] = {}

            state["sql_results"] = {
                "metrics":           state["mandatory_metrics"],
                "charts_generated":  len(state.get("chart_paths") or []),
                "news_fetched":      len((state.get("news_results") or {}).get("articles", [])),
                "geographic_rows":   len((state.get("geographic_data") or {}).get("data", [])),
                "user_query_result": user_query_result,
            }
            state["messages"].append(AIMessage(content="Dados SQL processados"))

            if self.audit:
                self.audit.log_event(AuditEvent.NODE_COMPLETE, {
                    "node": "execute_sql",
                    "has_user_query_result": user_query_result is not None,
                    "duration_seconds": round(time.perf_counter() - t0, 3),
                }, EventStatus.SUCCESS, duration_seconds=round(time.perf_counter() - t0, 3))

        except Exception as e:
            state["errors"].append(f"SQL node error: {e}")
            if self.audit:
                self.audit.log_event(AuditEvent.NODE_ERROR, {
                    "node": "execute_sql", "error": str(e),
                    "duration_seconds": round(time.perf_counter() - t0, 3),
                }, EventStatus.ERROR)
        return state

    # =========================================================================
    # NÓ 2B — RAG
    # =========================================================================

    def _execute_rag_node(self, state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        try:
            if self.audit:
                self.audit.log_event(AuditEvent.NODE_START,
                    {"node": "execute_rag"}, EventStatus.INFO)

            if self.rag_chain is None:
                state["messages"].append(
                    AIMessage(content="RAG não disponível — continuando sem contexto semântico")
                )
                return state

            rd       = state.get("routing_decision")
            rag_type = rd.rag_semantic_type if rd else None
            rag_result = self.rag_chain.invoke(state["user_query"],
                                               semantic_type_override=rag_type)
            state["rag_results"] = rag_result

            for doc in rag_result.get("source_documents", []):
                src = (doc.metadata.get("source") or doc.metadata.get("file_path")
                       or str(doc.metadata)) if hasattr(doc, "metadata") else str(doc)
                state["sources"].append(src)

            state["messages"].append(AIMessage(
                content=f"RAG: {len(rag_result.get('source_documents', []))} documentos"
            ))
            if self.audit:
                self.audit.log_event(AuditEvent.NODE_COMPLETE, {
                    "node": "execute_rag",
                    "duration_seconds": rag_result.get("metadata", {}).get(
                        "duration_seconds", round(time.perf_counter() - t0, 3)),
                    "num_sources": len(rag_result.get("source_documents", [])),
                    "rag_semantic_type": rag_type,
                }, EventStatus.SUCCESS)

        except Exception as e:
            state["errors"].append(f"RAG node error: {e}")
            if self.audit:
                self.audit.log_event(AuditEvent.NODE_ERROR, {
                    "node": "execute_rag", "error": str(e),
                    "duration_seconds": round(time.perf_counter() - t0, 3),
                }, EventStatus.ERROR)
        return state

    # =========================================================================
    # NÓ 2C — HYBRID
    # =========================================================================

    def _execute_hybrid_node(self, state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        try:
            if self.audit:
                self.audit.log_event(AuditEvent.NODE_START,
                    {"node": "execute_hybrid"}, EventStatus.INFO)

            def _copy(s):
                return {**s, "messages": list(s["messages"]),
                        "sources": list(s["sources"]), "errors": list(s["errors"])}

            sql_state = self._execute_sql_node(_copy(state))
            state["sql_results"]       = sql_state.get("sql_results", {})
            state["news_results"]      = sql_state.get("news_results", {})
            state["chart_paths"]       = sql_state.get("chart_paths", [])
            state["mandatory_metrics"] = sql_state.get("mandatory_metrics", {})
            state["geographic_data"]   = sql_state.get("geographic_data", {})
            state["errors"].extend(sql_state.get("errors", []))

            rag_state = self._execute_rag_node(_copy(state))
            state["rag_results"] = rag_state.get("rag_results", {})
            state["sources"]     = list(state["sources"]) + list(rag_state.get("sources", []))
            state["errors"].extend(rag_state.get("errors", []))

            state["messages"].append(AIMessage(content="Execução híbrida (SQL + RAG) completa"))
            if self.audit:
                self.audit.log_event(AuditEvent.NODE_COMPLETE, {
                    "node": "execute_hybrid",
                    "duration_seconds": round(time.perf_counter() - t0, 3),
                }, EventStatus.SUCCESS)

        except Exception as e:
            state["errors"].append(f"Hybrid node critical error: {e}")
            if self.audit:
                self.audit.log_event(AuditEvent.NODE_ERROR, {
                    "node": "execute_hybrid", "error": str(e),
                    "stack_trace": traceback.format_exc()[:500],
                    "duration_seconds": round(time.perf_counter() - t0, 3),
                }, EventStatus.ERROR)
            state["messages"].append(AIMessage(content=f"Aviso: execução parcial — {str(e)[:100]}"))
        return state

    # =========================================================================
    # NÓ 2D — CHART AD-HOC
    # =========================================================================

    def _execute_chart_node(self, state: AgentState) -> AgentState:
        """
        Nó de gráfico ad-hoc.

        Fluxo:
        1. Valida e sanitiza ChartParams do router.
        2. Executa SQL de agregação via _build_dynamic_chart_query().
        3. Resolve ChartSpec semântico via _resolve_chart_spec().
        4. Injeta métricas anuais em mandatory_metrics quando group_by="ano".
        5. Despacha para o ChartTool via _dispatch_chart_spec().
        6. Para comparativos anuais, gera gráfico de taxas adicional.
        """
        t0 = time.perf_counter()
        try:
            if self.audit:
                self.audit.log_event(AuditEvent.NODE_START,
                    {"node": "execute_chart"}, EventStatus.INFO)

            decision = state["routing_decision"]
            params: ChartParams = decision.chart_params
            if params is None:
                raise ValueError("ChartParams ausente no RoutingDecision")

            table    = params.table    if params.table    in _ALLOWED_TABLES   else "gold_metricas_temporais"
            metric   = params.metric   if params.metric   in _ALLOWED_METRICS  else "total_casos"
            group_by = params.group_by if params.group_by in _ALLOWED_GROUP_BY else "ano_mes"

            # Garante tabela correta para consultas anuais
            if group_by == "ano":
                table = "gold_metricas_historicas"

            print(f"[execute_chart] table={table} metric={metric} group_by={group_by} "
                  f"type={params.chart_type}")

            sql    = self._build_dynamic_chart_query(
                catalog=self.catalog, schema=self.schema,
                table=table, metric=metric, group_by=group_by,
                filters=params.filters,
            )
            result = self.sql_tool.execute_query(sql)
            if not result.get("success") or not result.get("data"):
                raise ValueError(f"Query retornou vazio: {result.get('error', 'sem dados')}")

            import pandas as _pd
            df = _pd.DataFrame(result["data"])

            # Injeta análise anual em mandatory_metrics ANTES de qualquer falha de gráfico
            if group_by == "ano" and not df.empty:
                state["mandatory_metrics"] = self._build_annual_metrics(df)

            # Resolve spec semântico
            spec = self._resolve_chart_spec(
                params=params,
                group_by=group_by,
                metric=metric,
                df=df,
            )

            if self.chart_tool is None:
                raise ValueError("chart_tool não disponível")

            chart_result = self._dispatch_chart_spec(spec, df)
            if chart_result is None:
                raise ValueError("_dispatch_chart_spec retornou None")

            # Persiste spec resolvido no estado para auditoria e payload público.
            # Inclui campos que ChartParams do router não carrega:
            # chart_purpose, y_cols, series_col, year_col, top_n, value_format.
            state["resolved_chart_spec"] = asdict(spec)

            chart_path = chart_result["path"]
            state["ad_hoc_chart_path"] = chart_path
            state["chart_paths"]       = list(state.get("chart_paths") or [])
            state["chart_paths"].append(chart_path)
            state["messages"].append(AIMessage(content=f"Gráfico gerado: {chart_path}"))

            # Gráfico complementar de taxas para análise anual
            if group_by == "ano" and not df.empty:
                extra = self._generate_annual_rates_chart(df)
                if extra:
                    state["chart_paths"].append(extra)
                    print(f"[execute_chart] gráfico de taxas anuais: {extra.split('/')[-1]}")

            if self.audit:
                self.audit.log_event(AuditEvent.CHART_GENERATED, {
                    "node": "execute_chart", "chart_path": chart_path,
                    "chart_type": spec.chart_type, "chart_purpose": spec.chart_purpose,
                    "metric": metric, "group_by": group_by,
                    "data_rows": len(df),
                    "duration_seconds": round(time.perf_counter() - t0, 3),
                }, EventStatus.SUCCESS)

        except Exception as e:
            state["errors"].append(f"Chart node error: {e}")
            print(f"[execute_chart] erro: {e}")
            if self.audit:
                self.audit.log_event(AuditEvent.NODE_ERROR, {
                    "node": "execute_chart", "error": str(e),
                    "stack_trace": traceback.format_exc()[:500],
                    "duration_seconds": round(time.perf_counter() - t0, 3),
                }, EventStatus.ERROR)
            state["messages"].append(
                AIMessage(content=f"Não foi possível gerar o gráfico: {str(e)[:120]}")
            )
        return state

    # =========================================================================
    # NÓ 2E — REPORT
    # =========================================================================

    def _execute_report_node(self, state: AgentState) -> AgentState:
        """
        Nó de relatório epidemiológico completo.

        Fluxo
        -----
        1. Coleta dados via _execute_sql_node (métricas + geo + charts + notícias).
        2. Coleta contexto RAG via _execute_rag_node (quando disponível).
        3. Monta payload canônico via _build_report_payload() — normaliza formatos,
           distingue ausência total de payload vazio, estabelece hierarquia de fontes.
        4. Registra ``report_block_status`` no estado: quais blocos foram produzidos,
           ausentes ou degradados — auditável independente do resultado final.
        5. Delega a geração ao ReportGenerator quando disponível; quando não, o
           payload fica disponível para o _synthesize_node com contexto mais rico.

        Hierarquia de fontes mantida
        ----------------------------
        - SQL / métricas materializadas : verdade factual principal
        - Notícias                      : contexto externo complementar
        - RAG                           : contexto metodológico / explicativo
        """
        t0 = time.perf_counter()
        try:
            if self.audit:
                self.audit.log_event(AuditEvent.NODE_START,
                    {"node": "execute_report"}, EventStatus.INFO)

            def _copy(s):
                return {**s, "messages": list(s["messages"]),
                        "sources": list(s["sources"]), "errors": list(s["errors"])}

            # ── 1. Dados SQL (métricas, geo, charts, notícias) ────────────────
            sql_state = self._execute_sql_node(_copy(state))
            state["sql_results"]       = sql_state.get("sql_results", {})
            state["news_results"]      = sql_state.get("news_results", {})
            state["chart_paths"]       = sql_state.get("chart_paths", [])
            state["mandatory_metrics"] = sql_state.get("mandatory_metrics", {})
            state["geographic_data"]   = sql_state.get("geographic_data", {})
            state["errors"].extend(sql_state.get("errors", []))

            # ── 2. Contexto RAG (metodológico) ────────────────────────────────
            if self.rag_chain:
                rag_state = self._execute_rag_node(_copy(state))
                state["rag_results"] = rag_state.get("rag_results", {})
                state["sources"]     = list(state["sources"]) + list(rag_state.get("sources", []))
                state["errors"].extend(rag_state.get("errors", []))
            else:
                state["rag_results"] = {}

            # ── 3. Payload canônico para ReportGenerator ──────────────────────
            report_payload = self._build_report_payload(state)

            # ── 4. Rastreabilidade de blocos ──────────────────────────────────
            block_status = self._assess_report_blocks(report_payload)
            state["report_block_status"] = block_status

            if self.audit:
                self.audit.log_event(AuditEvent.NODE_COMPLETE, {
                    "node":             "execute_report",
                    "blocks_ok":        [k for k, v in block_status.items() if v == "ok"],
                    "blocks_absent":    [k for k, v in block_status.items() if v == "absent"],
                    "blocks_degraded":  [k for k, v in block_status.items() if v == "degraded"],
                    "charts_generated": len(state.get("chart_paths") or []),
                    "has_rag":          bool(state.get("rag_results")),
                    "duration_seconds": round(time.perf_counter() - t0, 3),
                }, EventStatus.SUCCESS)

            # ── 5. Delegação ao ReportGenerator (quando disponível) ───────────
            if self.report_generator is not None:
                print("[execute_report] delegando ao ReportGenerator...")
                try:
                    report_md = self.report_generator.generate_report(
                        metrics    = report_payload["metrics"],
                        geographic = report_payload["geographic"],
                        news       = report_payload["news"],
                        charts     = report_payload["charts"],
                        rag_context= report_payload["rag_context"],
                        user_query = report_payload["user_query"],
                    )
                    if report_md:
                        # Relatório completo gerado — já pode ser o final_answer.
                        state["final_answer"] = report_md
                        state["messages"].append(
                            AIMessage(content="Relatório epidemiológico gerado via ReportGenerator")
                        )
                        return state
                except Exception as rge:
                    print(f"[execute_report] ReportGenerator falhou, seguindo para síntese: {rge}")
                    state["errors"].append(f"ReportGenerator error (fallback para síntese): {rge}")

            state["messages"].append(AIMessage(content="Insumos do relatório coletados — síntese via LLM"))

        except Exception as e:
            state["errors"].append(f"Report node critical error: {e}")
            if self.audit:
                self.audit.log_event(AuditEvent.NODE_ERROR, {
                    "node":             "execute_report",
                    "error":            str(e),
                    "stack_trace":      traceback.format_exc()[:500],
                    "duration_seconds": round(time.perf_counter() - t0, 3),
                }, EventStatus.ERROR)
            state["messages"].append(AIMessage(content=f"Aviso: relatório parcial — {str(e)[:100]}"))
        return state

    # =========================================================================
    # REPORT — MONTAGEM DO PAYLOAD CANÔNICO
    # =========================================================================

    def _build_report_payload(self, state: AgentState) -> Dict:
        """
        Monta payload canônico para o ReportGenerator / síntese de relatório.

        Responsabilidades
        -----------------
        - Normalizar formatos: garante que cada bloco tem a estrutura esperada
          pelo ReportGenerator (``{"data": [...]}`` para métricas e geo,
          ``{"articles": [...]}`` para notícias, ``{"answer": "..."}`` para RAG).
        - Distinguir ausência total de payload vazio: um dict vazio ``{}`` é
          diferente de ``{"data": []}`` — ambos são registrados com status
          distinto em ``_assess_report_blocks()``.
        - Preservar hierarquia de fontes: SQL é a verdade factual; notícias e
          RAG entram como contexto complementar, nunca como correção de métricas.
        - Não escrever editorial: nenhuma interpretação é feita aqui; isso é
          responsabilidade do ReportGenerator e do _synthesize_node.

        Retorno
        -------
        Dict com chaves: metrics, geographic, news, charts, rag_context, user_query.
        Cada bloco é sempre presente (nunca None) — ausência é representada por
        estrutura vazia tipada.
        """
        mandatory_metrics = state.get("mandatory_metrics") or {}
        geographic_data   = state.get("geographic_data")   or {}
        news_results      = state.get("news_results")      or {}
        rag_results       = state.get("rag_results")       or {}
        chart_paths       = state.get("chart_paths")       or []

        # ── Métricas (SQL — fonte factual principal) ──────────────────────────
        # ReportGenerator espera {"data": [periodo_atual, periodo_anterior, ...]}.
        # mandatory_metrics é um dict flat; empacota-o como elemento único da lista.
        if mandatory_metrics and not mandatory_metrics.get("error"):
            cresc_mensal = mandatory_metrics.get("crescimento_mensal", [])
            # Primeiro item = período mais recente; segundo = anterior (para trend).
            metrics_data = [mandatory_metrics]
            if cresc_mensal and len(cresc_mensal) >= 2:
                metrics_data.append(cresc_mensal[-2])  # período anterior para comparação
            metrics_payload = {"data": metrics_data, "source": "sql"}
        else:
            metrics_payload = {"data": [], "source": "sql", "error": mandatory_metrics.get("error")}

        # ── Dados geográficos (SQL) ───────────────────────────────────────────
        geo_rows = geographic_data.get("data", [])
        if isinstance(geo_rows, list) and geo_rows:
            geographic_payload = {"data": geo_rows, "source": "sql"}
        else:
            geographic_payload = {"data": [], "source": "sql"}

        # ── Notícias (contexto externo — não corrige métricas) ────────────────
        articles = news_results.get("articles", [])
        if isinstance(articles, list) and articles:
            news_payload = {"articles": articles, "source": "web_search"}
        else:
            news_payload = {"articles": [], "source": "web_search"}

        # ── Gráficos ──────────────────────────────────────────────────────────
        valid_charts = [p for p in chart_paths if p and isinstance(p, str)]

        # ── Contexto RAG (metodológico — não substitui fatos numéricos) ───────
        rag_answer = rag_results.get("answer", "")
        if rag_answer and isinstance(rag_answer, str):
            rag_payload = {
                "answer":       rag_answer,
                "source":       "rag",
                "num_docs":     len(rag_results.get("source_documents", [])),
                "semantic_type": rag_results.get("metadata", {}).get("semantic_type"),
            }
        else:
            rag_payload = {"answer": "", "source": "rag"}

        return {
            "metrics":    metrics_payload,
            "geographic": geographic_payload,
            "news":       news_payload,
            "charts":     valid_charts,
            "rag_context": rag_payload,
            "user_query":  state.get("user_query", "Gerar relatório SRAG"),
        }

    def _assess_report_blocks(self, payload: Dict) -> Dict[str, str]:
        """
        Avalia disponibilidade de cada bloco do payload de relatório.

        Retorna dict ``{bloco: status}`` onde status é:
        - ``"ok"``       : dados presentes e válidos
        - ``"degraded"`` : estrutura presente mas vazia (ex.: data=[])
        - ``"absent"``   : bloco ausente ou sem estrutura esperada
        """
        def _status_list(block: Dict, key: str) -> str:
            items = block.get(key, None)
            if items is None:
                return "absent"
            return "ok" if items else "degraded"

        return {
            "metrics":    _status_list(payload.get("metrics", {}), "data"),
            "geographic": _status_list(payload.get("geographic", {}), "data"),
            "news":       _status_list(payload.get("news", {}), "articles"),
            "charts":     "ok" if payload.get("charts") else "degraded",
            "rag_context": (
                "ok"       if payload.get("rag_context", {}).get("answer") else
                "degraded" if "answer" in payload.get("rag_context", {}) else
                "absent"
            ),
        }

    # =========================================================================
    # NÓ 3 — SÍNTESE
    # =========================================================================

    def _synthesize_node(self, state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        try:
            if self.audit:
                self.audit.log_event(AuditEvent.NODE_START,
                    {"node": "synthesize"}, EventStatus.INFO)

            # ── Curto-circuito: ReportGenerator já produziu o relatório ────────
            # _execute_report_node seta final_answer quando o ReportGenerator
            # gera conteúdo válido. Sobrescrever isso descartaria o relatório
            # estruturado (com gráficos listados) por uma síntese sem contexto.
            if state.get("final_answer"):
                if self.audit:
                    self.audit.log_event(AuditEvent.NODE_COMPLETE, {
                        "node": "synthesize", "skipped": True,
                        "reason": "final_answer já preenchido (ReportGenerator)",
                        "answer_length": len(state["final_answer"]),
                    }, EventStatus.SUCCESS)
                return state

            query             = state["user_query"]
            sql_results       = state.get("sql_results") or {}
            rag_results       = state.get("rag_results") or {}
            news_results      = state.get("news_results") or {}
            chart_paths       = state.get("chart_paths") or []
            mandatory_metrics = state.get("mandatory_metrics") or {}
            geographic_data   = state.get("geographic_data") or {}
            ad_hoc_path       = state.get("ad_hoc_chart_path")
            user_query_result = sql_results.get("user_query_result")

            ctx = []   # blocos de contexto para o prompt

            # ── Métricas obrigatórias ─────────────────────────────────────────
            if mandatory_metrics:
                ctx.append("=" * 70)
                ctx.append("MÉTRICAS SRAG — INCLUIR NA RESPOSTA")
                ctx.append("=" * 70)
                tc   = mandatory_metrics.get("taxa_crescimento")
                dref = mandatory_metrics.get("data_referencia", "N/A")
                if tc is None:
                    ctx.append("📈 Taxa de Crescimento Anual: ERRO NO CÁLCULO")
                elif tc == 0.0:
                    ctx.append(f"📈 Taxa de Crescimento Anual: 0.00% (ref: {dref}) — validar")
                else:
                    ctx.append(f"📈 Taxa de Crescimento Anual: {tc:.2f}% (ref: {dref})")
                ctx.append(f"💀 Taxa de Mortalidade: {mandatory_metrics.get('taxa_mortalidade', 0):.2f}%")
                ctx.append(f"🏥 Taxa de Ocupação UTI: {mandatory_metrics.get('taxa_uti', 0):.2f}%")
                ctx.append(f"💉 Taxa de Vacinação: {mandatory_metrics.get('taxa_vacinacao', 0):.2f}%")
                ctx.append(f"📊 Total de Casos: {mandatory_metrics.get('total_casos', 0):,}")
                ctx.append("=" * 70)
                ctx.append("")

            # ── Resultado da query específica do usuário ─────────────────────
            if user_query_result and user_query_result.get("data"):
                ctx.append("=" * 70)
                ctx.append("RESULTADO DA CONSULTA ESPECÍFICA DO USUÁRIO")
                ctx.append("=" * 70)
                rows = user_query_result["data"]
                cols = list(rows[0].keys())
                header = " | ".join(f"{c:>15}" for c in cols)
                ctx.append(header)
                ctx.append("-" * len(header))
                for row in rows[:50]:
                    ctx.append(" | ".join(f"{str(row.get(c, ''))[:15]:>15}" for c in cols))
                if len(rows) > 50:
                    ctx.append(f"  ... e mais {len(rows) - 50} linhas")
                ctx.append("→ Use estes dados para responder diretamente à pergunta.")
                ctx.append("=" * 70)
                ctx.append("")

            # ── Gráfico ad-hoc ────────────────────────────────────────────────
            if ad_hoc_path:
                ctx.append(f"GRÁFICO GERADO: 📊 {ad_hoc_path.split('/')[-1]}")
                ctx.append("")

            # ── Gráficos padrão ───────────────────────────────────────────────
            standard_charts = [p for p in chart_paths if p and p != ad_hoc_path]
            if standard_charts:
                ctx.append(f"GRÁFICOS PADRÃO DISPONÍVEIS ({len(standard_charts)}):")
                for i, p in enumerate(standard_charts, 1):
                    ctx.append(f"  {i}. 📊 {p.split('/')[-1]}")
                ctx.append("")

            # ── Dados geográficos ─────────────────────────────────────────────
            geo_rows = geographic_data.get("data", [])
            if geo_rows:
                ctx.append("=" * 70)
                ctx.append("DISTRIBUIÇÃO GEOGRÁFICA — TOP 10 UFs")
                ctx.append("=" * 70)
                ctx.append(f"{'UF':<6} {'Casos':>10} {'Óbitos':>10} {'Mortalidade':>12}")
                ctx.append("-" * 44)
                for row in geo_rows:
                    ctx.append(
                        f"{row.get('sg_uf','N/A'):<6} "
                        f"{int(row.get('total_casos',0)):>10,} "
                        f"{int(row.get('total_obitos',0)):>10,} "
                        f"{float(row.get('taxa_mortalidade',0)):>11.2f}%"
                    )
                ctx.append("=" * 70)
                ctx.append("")

            # ── Análise anual ─────────────────────────────────────────────────
            analise_anual = mandatory_metrics.get("analise_anual", [])
            if analise_anual:
                ctx.append("=" * 70)
                ctx.append("ANÁLISE ANUAL COMPARATIVA")
                ctx.append("=" * 70)
                for row in analise_anual:
                    ano   = row.get("ano",             row.get("casos_ano", "?"))
                    casos = row.get("casos_ano",        row.get("total_casos", 0))
                    mort  = row.get("mortalidade_pct",  row.get("taxa_mortalidade", 0))
                    uti   = row.get("uti_pct",          row.get("taxa_uti", 0))
                    vac   = row.get("vacinacao_pct",    row.get("taxa_vacinacao", 0))
                    flag  = " [PARCIAL]" if not row.get("is_completo", True) else ""
                    ctx.append(
                        f"  {ano}{flag}: {int(casos):,} casos | "
                        f"mort {float(mort):.2f}% | UTI {float(uti):.2f}% | vac {float(vac):.2f}%"
                    )
                ctx.append("=" * 70)
                ctx.append("")

            # ── Crescimento mensal ────────────────────────────────────────────
            cresc_mensal = mandatory_metrics.get("crescimento_mensal", [])
            if cresc_mensal:
                ctx.append("CRESCIMENTO MENSAL (últimos meses):")
                for row in cresc_mensal[:6]:
                    mes   = row.get("ano_mes", "?")
                    casos = row.get("total_casos", 0)
                    cresc = row.get("crescimento_mensal_pct")
                    cstr  = f"{cresc:+.1f}%" if cresc is not None else "N/A"
                    ctx.append(f"  {mes}: {int(casos):,} casos ({cstr} vs mês anterior)")
                ctx.append("")

            # ── Contexto RAG ──────────────────────────────────────────────────
            if rag_results and rag_results.get("answer"):
                ctx.append("CONTEXTO RAG (metodológico):")
                ctx.append(rag_results["answer"][:2000])
                ctx.append("")

            # ── Notícias ──────────────────────────────────────────────────────
            if news_results and news_results.get("articles"):
                ctx.append("NOTÍCIAS RECENTES:")
                for news in news_results["articles"][:3]:
                    ctx.append(f"- {news.get('title', 'N/A')}")
                ctx.append("")

            # ── Avisos de qualidade — gerados a partir dos dados reais ────────
            ctx.extend(self._build_quality_warnings(mandatory_metrics, cresc_mensal))

            # ── Instrução de modo ─────────────────────────────────────────────
            has_sql = bool(mandatory_metrics)
            if ad_hoc_path:
                mode_instruction = (
                    f"O gráfico foi gerado com sucesso: 📊 {ad_hoc_path.split('/')[-1]}\n"
                    "Confirme a geração, descreva o que o gráfico mostra e adicione 2-3 insights."
                )
            elif standard_charts:
                charts_list = "\n".join(
                    f"  {i}. 📊 {p.split('/')[-1]}" for i, p in enumerate(standard_charts, 1)
                )
                mode_instruction = (
                    f"Gráficos gerados:\n{charts_list}\n"
                    "Liste-os, descreva cada um e inclua as 4 métricas obrigatórias."
                )
            else:
                mode_instruction = "Inclua as 4 métricas obrigatórias em destaque no início."

            # ── Diretrizes de síntese ─────────────────────────────────────────
            if has_sql:
                synthesis_directives = self._build_synthesis_directives(mandatory_metrics)
            else:
                synthesis_directives = (
                    "- Base sua resposta EXCLUSIVAMENTE no contexto RAG disponível.\n"
                    "- NÃO invente dados numéricos ausentes no contexto.\n"
                    "- Se algum aspecto não tiver dados suficientes, diga isso claramente."
                )

            prompt = f"""Você é um especialista em epidemiologia analisando dados de SRAG no Brasil.

INSTRUÇÕES:
{mode_instruction}

ESTRUTURA DA RESPOSTA (use sempre):
1. FATOS — o que os dados mostram objetivamente (cite valores exatos)
2. INTERPRETAÇÃO — o que os fatos podem significar epidemiologicamente
3. LIMITAÇÕES — o que os dados não permitem concluir, subnotificações, períodos parciais

DIRETRIZES:
{synthesis_directives}

Responda em português brasileiro com tom técnico-profissional.
Quando SQL e RAG divergirem, priorize o SQL.
Explicite o período analisado com base nos dados reais disponíveis.

CONTEXTO E DADOS:
{chr(10).join(ctx)}

PERGUNTA:
{query}

RESPOSTA:"""

            response = self._get_synthesis_llm().invoke([HumanMessage(content=prompt)])
            state["final_answer"] = response.content
            state["messages"].append(AIMessage(content="Síntese completa"))

            if self.audit:
                self.audit.log_event(AuditEvent.NODE_COMPLETE, {
                    "node": "synthesize",
                    "answer_length": len(response.content),
                    "chart_mode": ad_hoc_path is not None,
                    "has_sql_data": has_sql,
                    "geo_rows_in_context": len(geo_rows),
                    "user_query_result": user_query_result is not None,
                    "standard_charts_count": len(standard_charts),
                    "llm_provider": "openai" if self.use_openai else "databricks",
                    "duration_seconds": round(time.perf_counter() - t0, 3),
                }, EventStatus.SUCCESS, duration_seconds=round(time.perf_counter() - t0, 3))

        except Exception as e:
            state["errors"].append(f"Synthesize error: {e}")
            if self.audit:
                self.audit.log_event(AuditEvent.SYNTHESIS_ERROR, {
                    "error": str(e), "query": state.get("user_query", "")[:200],
                    "stack_trace": traceback.format_exc()[:500],
                    "duration_seconds": round(time.perf_counter() - t0, 3),
                }, EventStatus.ERROR)
            state["final_answer"] = (
                "Não foi possível gerar síntese completa devido a erro técnico.\n\n"
                f"CONTEXTO PARCIAL:\n{chr(10).join(ctx[:10])}\n\n"
                "Consulte os logs para mais detalhes."
            )
        return state

    # =========================================================================
    # CHART SPEC — RESOLUÇÃO SEMÂNTICA
    # =========================================================================

    def _resolve_chart_spec(
        self,
        params:   ChartParams,
        group_by: str,
        metric:   str,
        df,                    # pd.DataFrame
    ) -> ChartSpec:
        """
        Aplica heurísticas semânticas para determinar ChartSpec.

        Hierarquia de decisões:
        1. Campos ricos do ChartParams (router) têm precedência quando preenchidos.
           O IntentRouter a partir da versão atual popula chart_purpose, y_cols,
           series_col, year_col, top_n e value_format — quando não estão no default
           ("generic", [], None, "auto"), o router sinalizou uma intenção explícita
           que deve ser respeitada em vez de sobrescrita pelas heurísticas.
        2. Dimensão analítica principal (group_by) determina o tipo base quando
           os campos ricos do router estão nos defaults.
        3. Natureza da métrica (taxa vs contagem) afina formatação.
        4. Cardinalidade dos dados afina orientação e top_n.

        NUNCA mapeia `ano` para x_col="indicador". O ano é sempre
        preservado como dimensão principal ou como year_col.
        """
        n_rows   = len(df)
        is_rate  = metric in _RATE_COLS
        val_fmt  = "percent" if is_rate else "number"
        filters  = params.filters or {}

        # Campos ricos do router — lidos com getattr para compatibilidade com
        # versões do IntentRouter que ainda não preenchem esses campos.
        router_purpose      = getattr(params, "chart_purpose", "generic")
        router_y_cols       = getattr(params, "y_cols",        [])
        router_series_col   = getattr(params, "series_col",    None)
        router_year_col     = getattr(params, "year_col",      None)
        router_top_n        = getattr(params, "top_n",         None)
        router_value_format = getattr(params, "value_format",  "auto")

        # Resolve value_format: router explícito > heurística por tipo de métrica
        if router_value_format not in ("auto", ""):
            val_fmt = router_value_format

        # ── Atalho: router sinalizou multi-série explicitamente ───────────────
        # Quando router_y_cols está preenchido, o router detectou uma intenção de
        # comparação de múltiplas séries — respeita diretamente.
        if router_y_cols:
            return ChartSpec(
                chart_type=params.chart_type or "bar",
                title=params.title or f"Comparativo — {', '.join(router_y_cols[:3])}",
                subtitle=self._period_from_filters(filters),
                x_col=group_by,
                y_col=metric,
                y_cols=router_y_cols,
                year_col=router_year_col,
                series_col=router_series_col,
                chart_purpose=router_purpose if router_purpose != "generic" else "rate_comparison",
                value_format=val_fmt,
                orientation="auto",
                sort_order="none",
                top_n=router_top_n,
                filters_applied=filters,
            )

        # ── Atalho: router sinalizou year_comparison explicitamente ──────────
        if router_year_col:
            return ChartSpec(
                chart_type="year_comparison",
                title=params.title or f"Sazonalidade de {metric.replace('_', ' ')} por Ano",
                subtitle=self._period_from_df(df, group_by),
                x_col=group_by,
                y_col=metric,
                year_col=router_year_col,
                series_col=router_series_col,
                chart_purpose=router_purpose if router_purpose != "generic" else "seasonal_comparison",
                value_format=val_fmt,
                filters_applied=filters,
            )

        # ── Heurísticas por dimensão analítica (fallback) ─────────────────────

        # ── Ranking geográfico ────────────────────────────────────────────────
        if group_by in _GEO_COLS:
            return ChartSpec(
                chart_type="top_n",
                title=params.title or f"Top {min(n_rows, 10)} por {metric.replace('_', ' ')}",
                subtitle=self._period_from_filters(filters),
                x_col=group_by,
                y_col=metric,
                chart_purpose=router_purpose if router_purpose != "generic" else "geographic_ranking",
                value_format=val_fmt,
                orientation="h",
                sort_order="value_desc",
                top_n=router_top_n or min(n_rows, 10),
                filters_applied=filters,
            )

        # ── Distribuição demográfica ──────────────────────────────────────────
        if group_by in _DEMO_COLS:
            return ChartSpec(
                chart_type="bar",
                title=params.title or f"Distribuição por {group_by.replace('_', ' ')} — {metric}",
                subtitle=self._period_from_filters(filters),
                x_col=group_by,
                y_col=metric,
                chart_purpose=router_purpose if router_purpose != "generic" else "demographic_distribution",
                value_format=val_fmt,
                orientation="h",
                sort_order="none",     # ordem clínica/etária preservada
                filters_applied=filters,
            )

        # ── Análise anual — PRESERVA ANO COMO DIMENSÃO PRINCIPAL ─────────────
        if group_by in _TEMPORAL_ANNUAL:
            rate_cols_available = [c for c in _RATE_COLS if c in df.columns]

            # Múltiplas taxas disponíveis → rate_comparison com ano no eixo X
            if len(rate_cols_available) >= 2:
                return ChartSpec(
                    chart_type="bar",          # rate_comparison é disparado por y_cols
                    title=params.title or "Taxas Comparativas por Ano (%)",
                    subtitle="Séries históricas anuais",
                    x_col="ano",
                    y_col=rate_cols_available[0],
                    y_cols=rate_cols_available,
                    chart_purpose=router_purpose if router_purpose != "generic" else "annual_rate_comparison",
                    value_format="percent",
                    orientation="auto",
                    sort_order="none",
                    filters_applied=filters,
                )

            # Métrica única → barra simples com ano no X
            return ChartSpec(
                chart_type="bar",
                title=params.title or f"{metric.replace('_', ' ').title()} por Ano",
                subtitle="Séries históricas anuais",
                x_col="ano",
                y_col=metric,
                chart_purpose=router_purpose if router_purpose != "generic" else "annual_trend",
                value_format=val_fmt,
                orientation="v",
                sort_order="none",
                filters_applied=filters,
            )

        # ── Série temporal mensal/semanal ─────────────────────────────────────
        if group_by in _TEMPORAL_MONTHLY:
            # Detecta múltiplos anos nos dados → year_comparison (sazonalidade)
            if "ano" in df.columns and df["ano"].nunique() > 1:
                return ChartSpec(
                    chart_type="year_comparison",
                    title=params.title or f"Sazonalidade de {metric.replace('_', ' ')} por Ano",
                    subtitle=self._period_from_df(df, group_by),
                    x_col=group_by,
                    y_col=metric,
                    year_col="ano",
                    chart_purpose=router_purpose if router_purpose != "generic" else "seasonal_comparison",
                    value_format=val_fmt,
                    filters_applied=filters,
                )

            chart_t = "area" if metric == "total_casos" else "line"
            return ChartSpec(
                chart_type=chart_t,
                title=params.title or f"Evolução de {metric.replace('_', ' ')}",
                subtitle=self._period_from_df(df, group_by),
                x_col=group_by,
                y_col=metric,
                chart_purpose=router_purpose if router_purpose != "generic" else "temporal_trend",
                value_format=val_fmt,
                filters_applied=filters,
            )

        # ── Fallback genérico ─────────────────────────────────────────────────
        return ChartSpec(
            chart_type=params.chart_type or "bar",
            title=params.title or f"{metric} por {group_by}",
            x_col=group_by,
            y_col=metric,
            series_col=router_series_col,
            chart_purpose=router_purpose,
            value_format=val_fmt,
            top_n=router_top_n,
            orientation="auto",
            filters_applied=filters,
        )

    def _dispatch_chart_spec(self, spec: ChartSpec, df) -> Optional[Dict]:
        """
        Traduz ChartSpec para a chamada correta do ChartTool.

        rate_comparison é acionado quando spec.y_cols está preenchido e
        todos os valores são taxas percentuais — independentemente de
        spec.chart_type ser "bar".
        """
        if self.chart_tool is None:
            return None

        ct = self.chart_tool

        title = self._effective_title(spec)
        sort  = self._normalize_sort_order(spec.sort_order)

        print(f"[dispatch_chart_spec] purpose={spec.chart_purpose} "
              f"type={spec.chart_type} x={spec.x_col} y={spec.y_col} "
              f"y_cols={spec.y_cols} year_col={spec.year_col}")

        # Rate comparison (múltiplas taxas, ano preservado no eixo X)
        if spec.y_cols and all(c in _RATE_COLS for c in spec.y_cols):
            return ct.create_rate_comparison_chart(
                data=df, title=title, x_col=spec.x_col, rate_cols=spec.y_cols,
            )

        # Year comparison (sazonalidade entre anos)
        if spec.chart_type == "year_comparison" and spec.year_col:
            return ct.create_year_comparison_chart(
                data=df, title=title, x_col=spec.x_col,
                y_col=spec.y_col, year_col=spec.year_col,
            )

        # Top-N ranking
        if spec.chart_type == "top_n":
            return ct.create_top_n_chart(
                data=df, title=title, x_col=spec.x_col,
                y_col=spec.y_col, n=spec.top_n or 10,
            )

        # Area / Line
        if spec.chart_type == "area":
            return ct.create_area_chart(data=df, title=title,
                                        x_col=spec.x_col, y_col=spec.y_col)
        if spec.chart_type == "line":
            return ct.create_line_chart(data=df, title=title,
                                        x_col=spec.x_col, y_col=spec.y_col)

        # Bar (com orientação e ordenação normalizadas)
        return ct.create_bar_chart(
            data=df, title=title, x_col=spec.x_col, y_col=spec.y_col,
            orientation=spec.orientation, sort_by=sort,
        )

    # =========================================================================
    # HELPERS DE GRÁFICO
    # =========================================================================

    def _generate_annual_rates_chart(self, df) -> Optional[str]:
        """
        Gera gráfico complementar de taxas percentuais por ano.
        Retorna o path do arquivo ou None em falha.
        """
        if self.chart_tool is None:
            return None
        rate_cols = [c for c in _RATE_COLS if c in df.columns]
        if len(rate_cols) < 2 or "ano" not in df.columns:
            return None
        try:
            result = self.chart_tool.create_rate_comparison_chart(
                data=df,
                title="Taxas Comparativas por Ano (%)",
                x_col="ano",
                rate_cols=rate_cols,
            )
            return result["path"] if result else None
        except Exception as e:
            print(f"[annual_rates_chart] aviso: {e}")
            return None

    def _build_annual_metrics(self, df) -> Dict:
        """
        Extrai mandatory_metrics a partir do DataFrame de análise anual.
        Médias ponderadas pelos casos de cada ano.
        """
        rows = []
        for _, r in df.iterrows():
            rows.append({
                "ano":              r.get("ano"),
                "casos_ano":        int(r.get("total_casos", 0) or 0),
                "mortalidade_pct":  float(r.get("taxa_mortalidade", 0) or 0),
                "uti_pct":          float(r.get("taxa_uti", 0) or 0),
                "vacinacao_pct":    float(r.get("taxa_vacinacao", 0) or 0),
                "is_completo":      bool(r.get("is_ano_completo", True)),
            })

        total_g = sum(r["casos_ano"] for r in rows) or 1
        mort_g  = round(sum(r["mortalidade_pct"] * r["casos_ano"] for r in rows) / total_g, 2)
        uti_g   = round(sum(r["uti_pct"]  * r["casos_ano"] for r in rows) / total_g, 2)
        vac_g   = round(sum(r["vacinacao_pct"] * r["casos_ano"] for r in rows) / total_g, 2)

        anos = [r["ano"] for r in rows if r["ano"] is not None]
        ref  = f"{min(anos)}–{max(anos)} (histórico)" if anos else "histórico"

        for r in rows:
            flag = "" if r["is_completo"] else " ⚠️ PARCIAL"
            print(f"  {r['ano']}{flag}: {r['casos_ano']:,} casos | "
                  f"mort {r['mortalidade_pct']}% | uti {r['uti_pct']}% | vac {r['vacinacao_pct']}%")
        print(f"  global ponderado: mort={mort_g}% | uti={uti_g}% | vac={vac_g}%")

        return {
            "analise_anual":    rows,
            "total_casos":      total_g,
            "taxa_mortalidade": mort_g,
            "taxa_uti":         uti_g,
            "taxa_vacinacao":   vac_g,
            "taxa_crescimento": 0.0,
            "data_referencia":  ref,
        }

    def _normalize_sort_order(self, sort_order: str) -> str:
        """
        Normaliza sort_order do ChartSpec para o contrato de sort_by do ChartTool.

        ChartSpec usa "value_desc" / "value_asc" para expressar intenção direcional.
        ChartTool aceita apenas "value" | "category" | "none" — a direção é inferida
        internamente pela orientação da barra (horizontal → ascending, vertical → descending).
        """
        if sort_order in ("value_desc", "value_asc"):
            return "value"
        if sort_order in ("category", "none"):
            return sort_order
        return "none"

    def _effective_title(self, spec: ChartSpec) -> str:
        """
        Injeta subtitle no título quando o ChartTool não gerará um automaticamente.

        ChartTool produz subtitle próprio via _build_subtitle() apenas para
        colunas temporais (data, ano_mes, mes…). Para dimensões geográficas e
        demográficas não há geração automática — o subtitle do ChartSpec seria
        silenciosamente descartado. Este método pré-incorpora o subtitle ao
        título usando o mesmo padrão HTML do ChartTool, evitando duplicação
        em colunas temporais.
        """
        if not spec.subtitle:
            return spec.title
        _temporal_keywords = ("data", "dt_", "ano_mes", "ano", "mes", "semana", "periodo")
        if any(k in spec.x_col.lower() for k in _temporal_keywords):
            return spec.title  # ChartTool gera o subtitle — não duplicar
        return (
            f"{spec.title}"
            f"<br><sup style='color:#888;font-size:12px'>{spec.subtitle}</sup>"
        )

    def _period_from_filters(self, filters: Dict) -> str:
        parts = []
        if "ano" in filters:
            parts.append(str(filters["ano"]))
        if "sg_uf" in filters:
            parts.append(filters["sg_uf"])
        return " | ".join(parts) if parts else ""

    def _period_from_df(self, df, col: str) -> str:
        try:
            vals = sorted(df[col].dropna().astype(str).unique())
            if len(vals) >= 2:
                return f"Período: {vals[0]} – {vals[-1]}"
            return vals[0] if vals else ""
        except Exception:
            return ""

    # =========================================================================
    # HELPERS DE SÍNTESE
    # =========================================================================

    def _build_quality_warnings(
        self, mandatory_metrics: Dict, cresc_mensal: List
    ) -> List[str]:
        """
        Gera avisos de qualidade a partir dos dados reais — sem narrativas fixas.
        """
        lines = ["=" * 70, "⚠️ AVISOS DE QUALIDADE DE DADO", "=" * 70]

        lines.append(
            "SUBNOTIFICAÇÃO: os últimos 14 dias da série diária têm queda artificial "
            "por atraso de registro. A taxa de crescimento usa dados consolidados."
        )

        analise = mandatory_metrics.get("analise_anual", [])
        parciais = [r for r in analise if not r.get("is_completo", True)]
        if parciais:
            anos_p = ", ".join(str(r.get("ano")) for r in parciais)
            lines.append(
                f"ANOS PARCIAIS: {anos_p} não têm dados do ano completo. "
                "Taxas podem estar subestimadas — compare com cautela."
            )

        if cresc_mensal and len(cresc_mensal) >= 2:
            vals = [r.get("crescimento_mensal_pct") for r in cresc_mensal
                    if r.get("crescimento_mensal_pct") is not None]
            if vals:
                lines.append(
                    f"VARIAÇÃO MENSAL: crescimento oscilou entre "
                    f"{min(vals):+.1f}% e {max(vals):+.1f}% nos meses disponíveis. "
                    "Sazonalidade pode explicar parte dessa variação."
                )

        lines.append("=" * 70)
        lines.append("")
        return lines

    def _build_synthesis_directives(self, mandatory_metrics: Dict) -> str:
        """
        Gera diretrizes de análise a partir dos dados — sem tendências hardcoded.
        """
        lines = [
            "- Baseie-se nos dados apresentados no contexto. Não assuma tendências além do que os dados mostram.",
            "- Explicite o período analisado com base nos dados reais (não suponha anos).",
            "- SQL é a fonte da verdade; RAG complementa com contexto metodológico.",
            "- Separe FATO, INTERPRETAÇÃO e LIMITAÇÃO na resposta.",
        ]

        analise = mandatory_metrics.get("analise_anual", [])
        if len(analise) >= 2:
            anos_disp = " | ".join(
                f"{r.get('ano')}: mort {r.get('mortalidade_pct', 0):.2f}%"
                for r in analise
            )
            lines.append(f"- Dados anuais disponíveis: {anos_disp}")
            lines.append(
                "- Se os dados mostrarem tendência de queda ou alta, mencione-a "
                "citando os valores exatos de cada ano."
            )

        vac = mandatory_metrics.get("taxa_vacinacao")
        if vac is not None:
            lines.append(
                f"- A taxa de vacinação consolidada é {vac:.2f}%. "
                "Se houver meses com valores discrepantes no contexto, mencione-os."
            )

        tc = mandatory_metrics.get("taxa_crescimento")
        if isinstance(tc, (int, float)):
            lines.append(
                f"- A taxa de crescimento calculada é {tc:.2f}% (dados consolidados, "
                "excluindo os últimos 14 dias por subnotificação)."
            )

        lines.append("- Termine com recomendações concretas baseadas nos dados apresentados.")
        return "\n".join(lines)

    # =========================================================================
    # SQL — QUERY ESPECÍFICA DO USUÁRIO
    # =========================================================================

    def _try_user_specific_query(self, user_query: str) -> Optional[Dict]:
        """
        Gera e executa SQL direcionada à pergunta do usuário.

        Estratégia de retry:
        1. Tentativa 1 — SQL completa gerada pelo LLM.
        2. Tentativa 2 — SQL simplificada com constraints mais estritas.
        3. Tentativa 3 — Fallback por template semântico baseado em palavras-chave.

        Retorna dict com "data" e "rows" ou None se todas as tentativas falharem.
        """
        for attempt in range(1, 4):
            try:
                result = self._try_user_specific_query_attempt(user_query, attempt)
                if result:
                    print(f"[user_sql] tentativa {attempt}: {result['rows']} linhas")
                    return result
            except Exception as e:
                print(f"[user_sql] tentativa {attempt} falhou: {e}")
        return None

    def _try_user_specific_query_attempt(
        self, user_query: str, attempt: int
    ) -> Optional[Dict]:
        tables_desc = "\n".join(
            f"- {self.catalog}.{self.schema}.{tbl}: {cols}"
            for tbl, cols in _ALLOWED_TABLES_FOR_USER_SQL.items()
        )

        if attempt == 3:
            # Fallback: template semântico por palavra-chave
            sql = self._semantic_template_fallback(user_query)
            if sql is None:
                return None
        else:
            strictness = (
                # Tentativa 1: proibe subqueries explicitamente; CTE é permitida.
                "Gere uma query SQL simples. "
                "PROIBIDO: subqueries escalares no SELECT (SELECT dentro de SELECT sem FROM). "
                "Para comparar períodos use CTE (WITH ... AS (...)) ou GROUP BY + CASE WHEN. "
                "Cada subquery que aparecer deve retornar EXATAMENTE uma coluna e uma linha."
                if attempt == 1 else
                # Tentativa 2: sem nenhuma subquery nem CTE.
                "Gere uma query SQL MUITO simples: apenas SELECT, FROM, WHERE, GROUP BY, ORDER BY, LIMIT. "
                "Sem subqueries. Sem CTEs. Sem funções de janela."
            )
            prompt = f"""Você é um especialista SQL em Databricks.
{strictness}

Tabelas disponíveis:
{tables_desc}

Regras OBRIGATÓRIAS:
- Apenas SELECT — sem INSERT, UPDATE, DELETE, DROP, ALTER
- Sempre LIMIT 100
- Tabelas com catálogo completo: {self.catalog}.{self.schema}.<tabela>
- Retornar APENAS a query SQL, sem explicações, sem markdown

PROIBIDO usar: LAG, LEAD, RANK, DENSE_RANK, ROW_NUMBER, NTILE, OVER, PARTITION BY

EXEMPLO CORRETO para comparação entre períodos (use este padrão):
WITH ultimos AS (
  SELECT ano_mes, total_casos,
         ROW_NUMBER() OVER (ORDER BY ano_mes DESC) AS rn
  FROM {self.catalog}.{self.schema}.gold_metricas_temporais
)
SELECT
  MAX(CASE WHEN rn = 1 THEN total_casos END) AS casos_mes_atual,
  MAX(CASE WHEN rn = 2 THEN total_casos END) AS casos_mes_anterior
FROM ultimos WHERE rn <= 2

EXEMPLO ERRADO (NUNCA FAÇA):
SELECT (SELECT ano_mes, total_casos FROM ... LIMIT 1) AS casos_mensais  -- retorna 2 colunas!

Pergunta: {user_query}
"""
            response = self.llm.invoke([HumanMessage(content=prompt)])
            sql = response.content.strip()

        sql = re.sub(r"```(?:sql)?\n?", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"```", "", sql).strip()

        statements = [s.strip() for s in sql.split(";") if s.strip()]
        sql = statements[0] if statements else sql

        if not sql.lower().lstrip().startswith("select") and not sql.lower().lstrip().startswith("with"):
            print(f"[user_sql] sem SELECT/WITH válido (tentativa {attempt})")
            return None

        # ── Guardrails de segurança ───────────────────────────────────────────

        _forbidden = re.compile(
            r"\b(LAG|LEAD|RANK|DENSE_RANK|ROW_NUMBER|NTILE|OVER|PARTITION\s+BY)\b",
            re.IGNORECASE,
        )
        if _forbidden.search(sql):
            print(f"[user_sql] window function detectada (tentativa {attempt})")
            return None

        if sql.count("(") != sql.count(")"):
            print(f"[user_sql] parênteses desbalanceados (tentativa {attempt})")
            return None

        # ── Detector de subquery escalar multi-coluna ─────────────────────────
        # Causa do erro INVALID_SUBQUERY_EXPRESSION.SCALAR_SUBQUERY_RETURN_MORE_THAN_ONE_OUTPUT_COLUMN:
        # subqueries escalares dentro do SELECT externo que retornam mais de uma coluna.
        # Padrão proibido: (SELECT col1, col2 FROM ... LIMIT 1) como expressão no SELECT.
        # CTEs (WITH name AS (...)) são excluídas da verificação — são sempre válidas.
        _sql_sem_cte = re.sub(
            r'\bWITH\b.+?(?=\bSELECT\b)', '',
            sql, flags=re.IGNORECASE | re.DOTALL,
        )
        _scalar_multi_col = re.compile(
            r'\(\s*SELECT\s+[^()]+,[^()]+(?:FROM|LIMIT|ORDER)',
            re.IGNORECASE | re.DOTALL,
        )
        if _scalar_multi_col.search(_sql_sem_cte):
            print(f"[user_sql] subquery escalar multi-coluna detectada (tentativa {attempt}) — rejeitando")
            return None

        result = self.sql_tool.execute_query(sql)
        if result.get("success") and result.get("data"):
            return {"data": result["data"], "rows": result["rows"]}

    def _semantic_template_fallback(self, user_query: str) -> Optional[str]:
        """
        Templates SQL seguros para padrões de pergunta reconhecíveis.
        Último recurso quando LLM falha duas vezes.
        """
        q = user_query.lower()
        base = f"{self.catalog}.{self.schema}"

        if any(w in q for w in ("por uf", "por estado", "estado", "uf")):
            return f"""
            SELECT sg_uf, SUM(total_casos) AS total_casos,
                   SUM(total_obitos) AS total_obitos
            FROM {base}.gold_metricas_geograficas
            WHERE sg_uf IS NOT NULL AND total_casos IS NOT NULL
            GROUP BY sg_uf ORDER BY total_casos DESC LIMIT 27
            """.strip()

        if any(w in q for w in ("faixa etária", "faixa_etaria", "idade")):
            return f"""
            SELECT faixa_etaria, SUM(total_casos) AS total_casos
            FROM {base}.gold_metricas_demograficas
            WHERE faixa_etaria IS NOT NULL AND total_casos IS NOT NULL
            GROUP BY faixa_etaria, ordem_faixa
            ORDER BY ordem_faixa ASC NULLS LAST LIMIT 20
            """.strip()

        if any(w in q for w in ("por ano", "anual", "histórico", "historico")):
            return f"""
            SELECT ano, SUM(total_casos) AS total_casos,
                   ROUND(SUM(total_obitos)/NULLIF(SUM(casos_com_desfecho),0)*100,2) AS taxa_mortalidade
            FROM {base}.gold_metricas_historicas
            WHERE ano IS NOT NULL AND total_casos IS NOT NULL
            GROUP BY ano ORDER BY ano ASC LIMIT 10
            """.strip()

        if any(w in q for w in ("mensal", "mês", "mes", "por mês")):
            return f"""
            SELECT ano_mes, SUM(total_casos) AS total_casos
            FROM {base}.gold_metricas_temporais
            WHERE ano_mes IS NOT NULL AND total_casos IS NOT NULL
            GROUP BY ano_mes ORDER BY ano_mes DESC LIMIT 12
            """.strip()

        return None

    # =========================================================================
    # SQL — QUERY DE GRÁFICO AD-HOC
    # =========================================================================

    def _build_dynamic_chart_query(
        self, catalog: str, schema: str, table: str,
        metric: str, group_by: str, filters: Dict, limit: int = 500,
    ) -> str:

        def sanitize(v: str) -> str:
            return re.sub(r"[^A-Za-z0-9_\-]", "", str(v))

        if group_by == "ano":
            where = ["total_casos IS NOT NULL", "ano IS NOT NULL"]
            if "sg_uf" in filters:
                where.append(f"sg_uf = '{sanitize(filters['sg_uf'])}'")
            return f"""
            SELECT
                ano,
                SUM(total_casos)                                                 AS total_casos,
                ROUND(SUM(total_obitos)/NULLIF(SUM(casos_com_desfecho),0)*100,2) AS taxa_mortalidade,
                ROUND(SUM(total_uti)/NULLIF(SUM(total_internados),0)*100,2)      AS taxa_uti,
                ROUND(SUM(total_vacinados)/NULLIF(SUM(casos_com_info_vacina),0)*100,2) AS taxa_vacinacao
            FROM {catalog}.{schema}.{table}
            WHERE {" AND ".join(where)}
            GROUP BY ano ORDER BY ano ASC LIMIT {limit}
            """.strip()

        where = [f"{metric} IS NOT NULL", f"{group_by} IS NOT NULL"]

        if "ano" in filters:
            ano = sanitize(filters["ano"])
            if group_by == "ano_mes":
                where.append(f"ano_mes LIKE '{ano}-%'")
            else:
                where.append(f"YEAR(dt_sintomas) = {ano}")

        if "sg_uf" in filters:
            where.append(f"sg_uf = '{sanitize(filters['sg_uf'])}'")

        if "mes" in filters:
            mes = sanitize(filters["mes"])
            mes_padded = str(int(mes)).zfill(2)
            if table == "gold_serie_diaria_30d":
                where.append(f"MONTH(dt_sintomas) = {int(mes)}")
            else:
                where.append(f"SUBSTRING(ano_mes, 6, 2) = '{mes_padded}'")

        return f"""
        SELECT {group_by}, SUM({metric}) AS {metric}
        FROM {catalog}.{schema}.{table}
        WHERE {" AND ".join(where)}
        GROUP BY {group_by}
        ORDER BY {group_by} ASC
        LIMIT {limit}
        """.strip()

    # =========================================================================
    # GEOGRAPHIC DATA
    # =========================================================================

    def _calculate_geographic_data(self) -> Dict:
        try:
            result = self.sql_tool.execute_query(f"""
                SELECT sg_uf,
                       SUM(total_casos)  AS total_casos,
                       SUM(total_obitos) AS total_obitos,
                       CASE WHEN SUM(casos_com_desfecho) > 0
                            THEN SUM(total_obitos)/SUM(casos_com_desfecho)*100
                            ELSE 0 END AS taxa_mortalidade
                FROM {self.catalog}.{self.schema}.gold_metricas_geograficas
                WHERE sg_uf IS NOT NULL AND total_casos IS NOT NULL
                GROUP BY sg_uf ORDER BY total_casos DESC LIMIT 10
            """)
            if result.get("success") and result.get("data"):
                rows = [{
                    "sg_uf":            r.get("sg_uf", "N/A"),
                    "total_casos":      int(r.get("total_casos", 0)),
                    "total_obitos":     int(r.get("total_obitos", 0)),
                    "taxa_mortalidade": round(float(r.get("taxa_mortalidade", 0)), 2),
                } for r in result["data"]]
                print(f"[geographic] {len(rows)} UFs")
                if self.audit:
                    self.audit.log_event(AuditEvent.METRICS_COLLECTED,
                        {"action": "geographic_data", "ufs": len(rows)}, EventStatus.SUCCESS)
                return {"data": rows}
            return {"data": []}
        except Exception as e:
            print(f"[geographic] aviso: {e}")
            return {"data": []}

    # =========================================================================
    # MANDATORY METRICS
    # =========================================================================

    def _calculate_mandatory_metrics(self) -> Dict:
        _err_event = getattr(AuditEvent, "METRICS_ERROR", AuditEvent.NODE_ERROR)
        try:
            if self.audit:
                self.audit.log_event(AuditEvent.METRICS_COLLECTED,
                    {"action": "calculate_mandatory_metrics"}, EventStatus.INFO)

            metrics = {}

            # ── Taxa de Crescimento Anual ─────────────────────────────────────
            # Usa MAX(ano) da tabela histórica — sem hardcode e sem CURRENT_DATE().
            # Dados disponíveis: 2023, 2024, 2025. MAX(ano)=2025, ano anterior=2024.
            try:
                r = self.sql_tool.execute_query(f"""
                WITH anos AS (
                    SELECT MAX(ano) AS ano_atual, MAX(ano) - 1 AS ano_ant
                    FROM {self.catalog}.{self.schema}.gold_metricas_historicas
                    WHERE total_casos IS NOT NULL AND total_casos > 0
                ),
                totais AS (
                    SELECT h.ano, SUM(h.total_casos) AS total_casos
                    FROM {self.catalog}.{self.schema}.gold_metricas_historicas h
                    JOIN anos a ON h.ano IN (a.ano_atual, a.ano_ant)
                    WHERE h.total_casos IS NOT NULL AND h.total_casos > 0
                    GROUP BY h.ano
                )
                SELECT
                    cur.ano                                                      AS ano_atual,
                    cur.total_casos                                              AS casos_ano_atual,
                    ant.total_casos                                              AS casos_ano_anterior,
                    ROUND(
                        (cur.total_casos - ant.total_casos)
                        / NULLIF(CAST(ant.total_casos AS DOUBLE), 0) * 100
                    , 2)                                                         AS taxa_crescimento
                FROM totais cur
                JOIN anos  a   ON cur.ano = a.ano_atual
                JOIN totais ant ON ant.ano = a.ano_ant
                LIMIT 1
                """)
                if r.get("success") and r.get("data"):
                    d    = r["data"][0]
                    raw  = d.get("taxa_crescimento")
                    taxa = float(raw) if raw is not None else None
                    if taxa is None or (taxa != taxa):   # NaN check: NaN != NaN
                        raise ValueError(f"taxa inválida: {raw}")
                    metrics.update({
                        "taxa_crescimento":   round(taxa, 2),
                        "casos_hoje":         int(d.get("casos_ano_atual") or 0),
                        "casos_ano_anterior": int(d.get("casos_ano_anterior") or 0),
                        "data_referencia":    str(d.get("ano_atual") or "N/A"),
                    })
                    print(f"   [crescimento anual] {taxa:.2f}% (ref: {d.get('ano_atual')})")
                else:
                    raise ValueError("query anual retornou vazio")
            except Exception as e:
                print(f"   [crescimento anual] falhou: {e} — fallback mensal...")
                if self.audit:
                    self.audit.log_event(_err_event,
                        {"metric": "taxa_crescimento", "error": str(e)}, EventStatus.WARNING)
                try:
                    # Fallback: acumulado dos últimos 12 meses vs 12 meses anteriores
                    r2 = self.sql_tool.execute_query(f"""
                    WITH ranked AS (
                        SELECT total_casos,
                               ROW_NUMBER() OVER (ORDER BY ano_mes DESC) AS rn
                        FROM {self.catalog}.{self.schema}.gold_metricas_temporais
                        WHERE total_casos IS NOT NULL AND total_casos > 0
                    ),
                    janelas AS (
                        SELECT
                            SUM(CASE WHEN rn <= 12 THEN total_casos ELSE 0 END) AS recente,
                            SUM(CASE WHEN rn >  12 THEN total_casos ELSE 0 END) AS anterior
                        FROM ranked WHERE rn <= 24
                    )
                    SELECT ROUND(
                        (recente - anterior) / NULLIF(CAST(anterior AS DOUBLE), 0) * 100
                    , 2) AS taxa_crescimento_media
                    FROM janelas LIMIT 1
                    """)
                    if r2.get("success") and r2.get("data"):
                        raw2 = r2["data"][0].get("taxa_crescimento_media")
                        taxa2 = float(raw2) if raw2 is not None else None
                        if taxa2 is None or (taxa2 != taxa2):   # NaN check
                            raise ValueError(f"fallback taxa inválida: {raw2}")
                        metrics["taxa_crescimento"] = round(taxa2, 2)
                        metrics["data_referencia"]  = "Acumulado 12m"
                        print(f"   [crescimento fallback] {taxa2:.2f}%")
                    else:
                        raise ValueError("fallback mensal vazio")
                except Exception as fe:
                    print(f"   [crescimento] fallback falhou: {fe}")
                    metrics["taxa_crescimento"]       = None
                    metrics["taxa_crescimento_error"] = str(e)
                    metrics["data_referencia"]        = "ERRO"

            # ── Taxa de Mortalidade ───────────────────────────────────────────
            try:
                r = self.sql_tool.execute_query(f"""
                SELECT SUM(total_casos) AS total_casos, SUM(total_obitos) AS total_obitos,
                       SUM(casos_com_desfecho) AS casos_com_desfecho,
                       CASE WHEN SUM(casos_com_desfecho)>0
                            THEN SUM(total_obitos)/SUM(casos_com_desfecho)*100
                            ELSE 0 END AS taxa_mortalidade
                FROM {self.catalog}.{self.schema}.gold_metricas_temporais
                WHERE total_casos IS NOT NULL AND total_obitos IS NOT NULL
                  AND casos_com_desfecho IS NOT NULL LIMIT 1
                """)
                if r.get("success") and r.get("data"):
                    d = r["data"][0]
                    metrics.update({
                        "taxa_mortalidade":   round(float(d.get("taxa_mortalidade", 0)), 2),
                        "total_casos":        int(d.get("total_casos", 0)),
                        "total_obitos":       int(d.get("total_obitos", 0)),
                        "casos_com_desfecho": int(d.get("casos_com_desfecho", 0)),
                    })
                else:
                    metrics.update({"taxa_mortalidade": 0.0, "total_casos": 0,
                                    "total_obitos": 0, "casos_com_desfecho": 0})
            except Exception as e:
                print(f"   [mortalidade] aviso: {e}")
                metrics["taxa_mortalidade"] = 0.0

            # ── Taxa UTI ──────────────────────────────────────────────────────
            try:
                r = self.sql_tool.execute_query(f"""
                SELECT SUM(total_internados) AS total_internados,
                       SUM(total_uti) AS total_uti,
                       CASE WHEN SUM(total_internados)>0
                            THEN SUM(total_uti)/SUM(total_internados)*100
                            ELSE 0 END AS taxa_uti
                FROM {self.catalog}.{self.schema}.gold_metricas_temporais
                WHERE total_internados IS NOT NULL AND total_uti IS NOT NULL LIMIT 1
                """)
                if r.get("success") and r.get("data"):
                    d = r["data"][0]
                    metrics.update({
                        "taxa_uti":         round(float(d.get("taxa_uti", 0)), 2),
                        "total_internados": int(d.get("total_internados", 0)),
                        "total_uti":        int(d.get("total_uti", 0)),
                    })
                else:
                    metrics.update({"taxa_uti": 0.0, "total_internados": 0, "total_uti": 0})
            except Exception as e:
                print(f"   [uti] aviso: {e}")
                metrics["taxa_uti"] = 0.0

            # ── Taxa Vacinação ────────────────────────────────────────────────
            try:
                r = self.sql_tool.execute_query(f"""
                SELECT SUM(total_vacinados) AS total_vacinados,
                       SUM(casos_com_info_vacina) AS casos_com_info_vacina,
                       CASE WHEN SUM(casos_com_info_vacina)>0
                            THEN SUM(total_vacinados)/SUM(casos_com_info_vacina)*100
                            ELSE 0 END AS taxa_vacinacao
                FROM {self.catalog}.{self.schema}.gold_metricas_temporais
                WHERE total_vacinados IS NOT NULL AND casos_com_info_vacina IS NOT NULL LIMIT 1
                """)
                if r.get("success") and r.get("data"):
                    d = r["data"][0]
                    metrics.update({
                        "taxa_vacinacao":        round(float(d.get("taxa_vacinacao", 0)), 2),
                        "total_vacinados":       int(d.get("total_vacinados", 0)),
                        "casos_com_info_vacina": int(d.get("casos_com_info_vacina", 0)),
                    })
                else:
                    metrics.update({"taxa_vacinacao": 0.0, "total_vacinados": 0,
                                    "casos_com_info_vacina": 0})
            except Exception as e:
                print(f"   [vacinacao] aviso: {e}")
                metrics["taxa_vacinacao"] = 0.0

            # ── Análise anual ─────────────────────────────────────────────────
            try:
                r = self.sql_tool.execute_query(f"""
                SELECT YEAR(TO_DATE(ano_mes,'yyyy-MM'))                              AS ano,
                       SUM(total_casos)                                              AS casos_ano,
                       SUM(total_obitos)                                             AS obitos_ano,
                       SUM(casos_com_desfecho)                                       AS com_desfecho_ano,
                       ROUND(SUM(total_obitos)/NULLIF(SUM(casos_com_desfecho),0)*100,2) AS mortalidade_pct,
                       ROUND(SUM(total_uti)/NULLIF(SUM(total_internados),0)*100,2)   AS uti_pct,
                       ROUND(SUM(total_vacinados)/NULLIF(SUM(casos_com_info_vacina),0)*100,2) AS vacinacao_pct
                FROM {self.catalog}.{self.schema}.gold_metricas_historicas
                WHERE casos_com_desfecho > 0
                GROUP BY 1 ORDER BY 1 DESC LIMIT 5
                """)
                if r.get("success") and r.get("data"):
                    metrics["analise_anual"] = r["data"]
                    anos = [str(row.get("ano", "?")) for row in r["data"]]
                    print(f"   [analise_anual] anos: {', '.join(anos)}")
                else:
                    metrics["analise_anual"] = []
            except Exception as ea:
                print(f"   [analise_anual] aviso: {ea}")
                metrics["analise_anual"] = []

            # ── Crescimento mensal ────────────────────────────────────────────
            try:
                r = self.sql_tool.execute_query(f"""
                WITH serie AS (
                    SELECT ano_mes, total_casos,
                           LAG(total_casos) OVER (ORDER BY ano_mes) AS casos_ant
                    FROM {self.catalog}.{self.schema}.gold_metricas_temporais
                    WHERE total_casos IS NOT NULL AND total_casos > 0
                    ORDER BY ano_mes DESC LIMIT 13
                )
                SELECT ano_mes, total_casos, casos_ant,
                       ROUND(CASE WHEN casos_ant>0
                                  THEN ((total_casos-casos_ant)/casos_ant)*100
                                  ELSE NULL END, 2) AS crescimento_mensal_pct
                FROM serie WHERE casos_ant IS NOT NULL
                ORDER BY ano_mes DESC LIMIT 12
                """)
                if r.get("success") and r.get("data"):
                    metrics["crescimento_mensal"] = r["data"]
                    print(f"   [crescimento_mensal] {len(r['data'])} meses")
                else:
                    metrics["crescimento_mensal"] = []
            except Exception as em:
                print(f"   [crescimento_mensal] aviso: {em}")
                metrics["crescimento_mensal"] = []

            # ── Validação final — NaN-safe ────────────────────────────────────
            for key in ("taxa_mortalidade", "taxa_uti", "taxa_vacinacao"):
                v = metrics.get(key)
                metrics[key] = max(0.0, float(v)) if isinstance(v, (int, float)) else 0.0

            tc = metrics.get("taxa_crescimento")
            if tc is None:
                pass                              # None é sinal de erro — preservar
            elif not isinstance(tc, (int, float)):
                metrics["taxa_crescimento"] = 0.0
            elif tc != tc:                        # NaN != NaN é sempre True em Python/IEEE 754
                print("[metrics] taxa_crescimento era NaN — corrigido para None")
                metrics["taxa_crescimento"] = None
                metrics.setdefault("taxa_crescimento_error", "resultado NaN da query SQL")

            if self.audit:
                self.audit.log_event(AuditEvent.METRICS_COLLECTED, {
                    "metrics_calculated": 4,
                    "taxa_crescimento":   metrics.get("taxa_crescimento"),
                    "taxa_mortalidade":   metrics.get("taxa_mortalidade"),
                    "taxa_uti":           metrics.get("taxa_uti"),
                    "taxa_vacinacao":     metrics.get("taxa_vacinacao"),
                    "total_casos":        metrics.get("total_casos", 0),
                }, EventStatus.SUCCESS)

            tc = metrics.get("taxa_crescimento")
            print(f"[metrics] crescimento : {f'{tc:.2f}%' if isinstance(tc, float) else 'ERRO'}")
            print(f"[metrics] mortalidade : {metrics.get('taxa_mortalidade', 0):.2f}%")
            print(f"[metrics] uti         : {metrics.get('taxa_uti', 0):.2f}%")
            print(f"[metrics] vacinacao   : {metrics.get('taxa_vacinacao', 0):.2f}%")
            print(f"[metrics] total_casos : {metrics.get('total_casos', 0):,}")
            return metrics

        except Exception as e:
            print(f"[metrics] erro crítico: {e}")
            if self.audit:
                self.audit.log_event(AuditEvent.METRICS_COLLECTED,
                    {"error": str(e)}, EventStatus.ERROR)
            return {
                "taxa_crescimento": 0.0, "taxa_mortalidade": 0.0,
                "taxa_uti": 0.0,        "taxa_vacinacao": 0.0,
                "total_casos": 0,       "total_obitos": 0,
                "total_internados": 0,  "total_uti": 0,
                "total_vacinados": 0,   "casos_com_info_vacina": 0,
                "data_referencia": "N/A", "error": str(e),
            }

    # =========================================================================
    # PROVIDER
    # =========================================================================

    def _get_synthesis_llm(self) -> BaseChatModel:
        label = "openai" if self.use_openai else type(self.llm).__name__
        print(f"[provider] {label}")
        return self.llm

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def run(self, user_query: str) -> Dict:
        start_time = time.time()

        if self.audit:
            self.audit.log_event(AuditEvent.ORCHESTRATOR_START, {
                "query": user_query, "version": VERSION,
                "llm_provider": "openai" if self.use_openai else "databricks",
            }, EventStatus.INFO)

        initial_state: AgentState = {
            "messages":            [HumanMessage(content=user_query)],
            "user_query":          user_query,
            "routing_decision":    None,
            "sql_results":         None,
            "rag_results":         None,
            "news_results":        None,
            "chart_paths":         None,
            "ad_hoc_chart_path":   None,
            "geographic_data":     None,
            "mandatory_metrics":   None,
            "final_answer":        None,
            "sources":             [],
            "errors":              [],
            "resolved_chart_spec": None,
            "report_block_status": None,
        }

        try:
            final_state = self.graph.invoke(initial_state)
            if final_state is None:
                final_state = {**initial_state, "errors": ["LangGraph retornou None"]}

            execution_time = round(time.time() - start_time, 2)
            success        = len(final_state.get("errors", [])) == 0
            rd             = final_state.get("routing_decision")

            if self.audit:
                chart_paths  = final_state.get("chart_paths") or []
                final_answer = final_state.get("final_answer") or ""
                self.audit.log_event(AuditEvent.ORCHESTRATOR_STRATEGY, {
                    "strategy_used":       rd.strategy.value if rd else "UNKNOWN",
                    "confidence":          rd.confidence if rd else 0,
                    "has_sql_results":     bool(final_state.get("sql_results")),
                    "has_rag_results":     bool(final_state.get("rag_results")),
                    "has_charts":          bool(chart_paths),
                    "num_charts":          len(chart_paths),
                    "has_adhoc_chart":     bool(final_state.get("ad_hoc_chart_path")),
                    "has_geographic_data": bool((final_state.get("geographic_data") or {}).get("data")),
                    "final_answer_length": len(final_answer),
                    "llm_provider":        "openai" if self.use_openai else "databricks",
                }, EventStatus.INFO)
                evt = AuditEvent.ORCHESTRATOR_SUCCESS if success else AuditEvent.ORCHESTRATOR_FAILED
                self.audit.log_event(evt, {
                    "execution_time": execution_time,
                    "errors": len(final_state.get("errors", [])),
                }, EventStatus.SUCCESS if success else EventStatus.ERROR)

            rd = final_state.get("routing_decision")
            return {
                "success":                success,
                "answer":                 final_state.get("final_answer"),
                "final_answer":           final_state.get("final_answer"),
                "sources":                final_state.get("sources") or [],
                "mandatory_metrics":      final_state.get("mandatory_metrics") or {},
                "sql_results":            final_state.get("sql_results") or {},
                "rag_results":            final_state.get("rag_results") or {},
                "news_results":           final_state.get("news_results") or {},
                "chart_paths":            final_state.get("chart_paths") or [],
                "ad_hoc_chart_path":      final_state.get("ad_hoc_chart_path"),
                "geographic_data":        final_state.get("geographic_data") or {"data": []},
                "report_block_status":    final_state.get("report_block_status"),
                "routing": {
                    "intent":     rd.intent.value if rd and rd.intent else None,
                    "strategy":   rd.strategy.value if rd else None,
                    "confidence": rd.confidence if rd else 0,
                    # ChartParams do router — todos os campos, incluindo os ricos
                    # adicionados na nova versão do IntentRouter (chart_purpose,
                    # y_cols, series_col, year_col, top_n, value_format).
                    "chart_params": {
                        "metric":        rd.chart_params.metric,
                        "group_by":      rd.chart_params.group_by,
                        "chart_type":    rd.chart_params.chart_type,
                        "title":         rd.chart_params.title,
                        "filters":       rd.chart_params.filters,
                        "table":         rd.chart_params.table,
                        # Campos ricos — populados pelo IntentRouter a partir desta versão.
                        # Podem estar no default ("generic", [], None, "auto") em versões
                        # anteriores do router; o orchestrator os expõe sem condicional.
                        "chart_purpose": getattr(rd.chart_params, "chart_purpose", "generic"),
                        "y_cols":        getattr(rd.chart_params, "y_cols",        []),
                        "series_col":    getattr(rd.chart_params, "series_col",    None),
                        "year_col":      getattr(rd.chart_params, "year_col",      None),
                        "top_n":         getattr(rd.chart_params, "top_n",         None),
                        "value_format":  getattr(rd.chart_params, "value_format",  "auto"),
                    } if rd and rd.chart_params else None,
                    # ChartSpec resolvido — inclui campos ricos gerados internamente:
                    # chart_purpose, y_cols, series_col, year_col, top_n, value_format.
                    # None quando a execução não passou por execute_chart_node.
                    "resolved_chart_spec": final_state.get("resolved_chart_spec"),
                },
                "errors":                 final_state.get("errors") or [],
                "execution_time_seconds": execution_time,
                "messages":               [m.content for m in (final_state.get("messages") or [])],
                "timestamp":              datetime.utcnow().isoformat(),
                "llm_provider":           "openai" if self.use_openai else "databricks",
            }

        except Exception as e:
            execution_time = round(time.time() - start_time, 2)
            if self.audit:
                self.audit.log_event(AuditEvent.ORCHESTRATOR_FAILED, {
                    "error": str(e), "execution_time": execution_time,
                }, EventStatus.CRITICAL)
            raise OrchestratorError(
                f"Falha crítica no orquestrador: {e}",
                details={"execution_time": execution_time},
            )

    def explain_routing(self, user_query: str) -> Dict:
        """
        Explica a decisão de roteamento sem executar o pipeline.

        Nota: ``chart_params`` reflete apenas o contrato do IntentRouter
        (6 campos: metric, group_by, chart_type, title, filters, table).
        Os campos ricos do ChartSpec resolvido (chart_purpose, y_cols,
        series_col, year_col, top_n, value_format) só estão disponíveis
        após execução via ``run()["routing"]["resolved_chart_spec"]``.
        """
        decision = self.router.route(user_query)
        result = {
            "query":         user_query,
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
                # Campos base — disponíveis em todas as versões do IntentRouter.
                "metric":     decision.chart_params.metric,
                "group_by":   decision.chart_params.group_by,
                "chart_type": decision.chart_params.chart_type,
                "title":      decision.chart_params.title,
                "filters":    decision.chart_params.filters,
                "table":      decision.chart_params.table,
                # Campos ricos — populados pelo IntentRouter a partir da versão atual.
                # getattr garante compatibilidade com versões antigas do router.
                "chart_purpose": getattr(decision.chart_params, "chart_purpose", "generic"),
                "y_cols":        getattr(decision.chart_params, "y_cols",        []),
                "series_col":    getattr(decision.chart_params, "series_col",    None),
                "year_col":      getattr(decision.chart_params, "year_col",      None),
                "top_n":         getattr(decision.chart_params, "top_n",         None),
                "value_format":  getattr(decision.chart_params, "value_format",  "auto"),
                # Campos do ChartSpec resolvido — disponíveis apenas após execução.
                "_resolved_note": (
                    "ChartSpec resolvido (filters_applied, subtitle, orientation, "
                    "sort_order) disponível em run()['routing']['resolved_chart_spec']."
                ),
            }
        return result