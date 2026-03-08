"""
Orchestrator — Agente SRAG com LangGraph
=========================================

Implementa o grafo de execução do agente SRAG usando LangGraph.
Cada nó do grafo encapsula uma responsabilidade distinta; o roteamento
é determinado pelo IntentRouter antes de qualquer execução de dados.

Pipeline
--------
    Query -> route -> execute_(sql|rag|hybrid|chart|report) -> synthesize -> resposta

Nós de execução
---------------
    execute_sql    : métricas obrigatórias + query específica do usuário + gráficos
                     padrão + web search + dados geográficos. Ativado por SQL_ONLY.
    execute_rag    : recuperação semântica via Databricks Vector Search.
                     Ativado por RAG_ONLY.
    execute_hybrid : executa sql e rag em cópias isoladas do estado e faz merge.
                     Ativado por HYBRID.
    execute_chart  : gráfico ad-hoc — SQL dinâmica parametrizada + ChartTool.
                     Ativado por CHART.
    execute_report : relatório epidemiológico completo — executa o pipeline SQL
                     completo (métricas + gráficos + notícias + geográfico) em
                     sequência com o pipeline RAG para contexto metodológico.
                     Ativado por REPORT.
"""

import re
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional, TypedDict

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


# =============================================================================
# CONFIGURAÇÃO GLOBAL
# =============================================================================

CATALOG = "dbx_srag_lab"
SCHEMA  = "gold"
VERSION = "4.5.0"

# Tabelas permitidas para geração de SQL dinâmica pelo LLM.
# Usadas como contexto no prompt de _try_user_specific_query().
_ALLOWED_TABLES_FOR_USER_SQL = {
    "gold_metricas_temporais":   "ano_mes, total_casos, total_obitos, casos_com_desfecho, total_internados, total_uti, total_vacinados, casos_com_info_vacina",
    "gold_metricas_geograficas": "sg_uf, ano_mes, total_casos, total_obitos, casos_com_desfecho",
    "gold_metricas_demograficas":"faixa_etaria, ano_mes, total_casos, total_obitos",
    "gold_serie_diaria_30d":     "dt_sintomas, total_casos",
    "gold_metricas_historicas":  "ano_mes, total_casos, total_obitos, casos_com_desfecho, total_internados, total_uti, total_vacinados, casos_com_info_vacina",
}


# =============================================================================
# STATE DEFINITION
# =============================================================================

class AgentState(TypedDict):
    """
    Estado compartilhado entre nós do grafo LangGraph.

    Todos os campos opcionais são None no estado inicial e populados pelo nó
    responsável. Os campos sources e errors são listas mutáveis — nós que
    executam em cópias isoladas (execute_hybrid, execute_report) precisam
    criar novas instâncias de lista para evitar compartilhamento de referência
    entre as branches.
    """
    messages:           List[BaseMessage]
    user_query:         str
    routing_decision:   Optional[RoutingDecision]
    sql_results:        Optional[Dict]
    rag_results:        Optional[Dict]
    news_results:       Optional[Dict]
    chart_paths:        Optional[List[str]]
    ad_hoc_chart_path:  Optional[str]
    geographic_data:    Optional[Dict]
    mandatory_metrics:  Optional[Dict]
    final_answer:       Optional[str]
    sources:            List[str]
    errors:             List[str]


# =============================================================================
# ORCHESTRATOR
# =============================================================================

class SRAGOrchestrator:
    """
    Grafo LangGraph do agente SRAG.

    Parâmetros
    ----------
    spark
        SparkSession ativa — obrigatória para GoldSQLTool e ChartTool.
    llm
        Qualquer BaseChatModel; usado na síntese e, quando fornecido,
        no roteamento por LLM.
    audit_logger
        Instância de AuditLogger. Quando None, eventos não são persistidos.
    rag_chain
        SRAGChain pré-construída. Ausência desabilita os nós RAG e REPORT
        sem levantar erro — o pipeline continua apenas com SQL.
    use_llm_routing
        Quando True, o IntentRouter usa o LLM para classificar intenção
        em vez de regex. Mais preciso, porém adiciona latência e custo.
    web_search_tool
        Ferramenta Tavily. Ausência desabilita a etapa de notícias.
    chart_tool
        ChartTool. Obrigatória para execute_chart; ausência eleva o erro
        para state["errors"] sem abortar o pipeline.
    catalog / schema
        Identificadores do Unity Catalog. Separados do código para permitir
        testes em schemas alternativos sem alterar queries.
    use_openai
        Flag de logging de provider. Quando None, inferido automaticamente
        pelo tipo do llm injetado.
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

        if use_openai is None:
            cls_name = type(llm).__name__.lower()
            self.use_openai = "openai" in cls_name or "azure" in cls_name
        else:
            self.use_openai = use_openai

        self.sql_tool = GoldSQLTool(spark, audit_logger)

        self.router = IntentRouter(
            use_llm_classification=use_llm_routing,
            llm=llm if use_llm_routing else None,
        )

        self.web_search_tool = web_search_tool
        self.chart_tool      = chart_tool
        self.graph           = self._build_graph()

        if self.audit:
            self.audit.log_event(
                AuditEvent.ORCHESTRATOR_INITIALIZED,
                {
                    "has_rag":        rag_chain       is not None,
                    "has_web_search": web_search_tool is not None,
                    "has_charts":     chart_tool      is not None,
                    "catalog":        self.catalog,
                    "schema":         self.schema,
                    "version":        VERSION,
                    "llm_provider":   "openai" if self.use_openai else "databricks",
                    "llm_class":      type(llm).__name__,
                },
                EventStatus.INFO,
            )

    # =========================================================================
    # PROVIDER SELECTION
    # =========================================================================

    def _get_synthesis_llm(self) -> BaseChatModel:
        label = "openai" if self.use_openai else type(self.llm).__name__
        print(f"[provider] {label}")
        return self.llm

    # =========================================================================
    # GRAPH CONSTRUCTION
    # =========================================================================

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("route",          self._route_node)
        workflow.add_node("execute_sql",    self._execute_sql_node)
        workflow.add_node("execute_rag",    self._execute_rag_node)
        workflow.add_node("execute_hybrid", self._execute_hybrid_node)
        workflow.add_node("execute_chart",  self._execute_chart_node)
        workflow.add_node("execute_report", self._execute_report_node)
        workflow.add_node("synthesize",     self._synthesize_node)

        workflow.set_entry_point("route")

        workflow.add_conditional_edges(
            "route",
            self._route_to_execution,
            {
                "sql":    "execute_sql",
                "rag":    "execute_rag",
                "hybrid": "execute_hybrid",
                "chart":  "execute_chart",
                "report": "execute_report",
            },
        )

        for node in ("execute_sql", "execute_rag", "execute_hybrid",
                     "execute_chart", "execute_report"):
            workflow.add_edge(node, "synthesize")
        workflow.add_edge("synthesize", END)

        return workflow.compile()

    # =========================================================================
    # GRAPH NODES
    # =========================================================================

    def _route_node(self, state: AgentState) -> AgentState:
        """
        Nó 1 — Roteamento via IntentRouter.
        """
        try:
            if self.audit:
                self.audit.log_event(
                    AuditEvent.NODE_START,
                    {"node": "route", "query": state["user_query"]},
                    EventStatus.INFO,
                )

            decision = self.router.route(state["user_query"])
            state["routing_decision"] = decision
            state["messages"].append(
                AIMessage(content=f"Rota: {decision.strategy.value} | {decision.reasoning}")
            )

            if self.audit:
                self.audit.log_event(
                    AuditEvent.QUERY_ANALYZED,
                    {
                        "strategy":   decision.strategy.value,
                        "confidence": decision.confidence,
                        "intent":     decision.intent.value,
                        "rag_type":   decision.rag_semantic_type,
                    },
                    EventStatus.SUCCESS,
                )

        except Exception as e:
            state["errors"].append(f"Routing error: {str(e)}")
            state["routing_decision"] = RoutingDecision(
                intent        = QueryIntent.FACTUAL,
                strategy      = ExecutionStrategy.SQL_ONLY,
                confidence    = 0.5,
                reasoning     = "Fallback por erro no routing",
                target_tables = ["gold_metricas_temporais"],
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

    # ------------------------------------------------------------------
    # Nó 2A — SQL puro
    # ------------------------------------------------------------------

    def _execute_sql_node(self, state: AgentState) -> AgentState:
        """
        Nó 2A — Pipeline SQL: query específica do usuário, métricas obrigatórias,
        dados geográficos, gráficos padrão e notícias recentes.

        _try_user_specific_query() é chamado primeiro para gerar SQL direcionada
        à pergunta do usuário via LLM. O resultado é armazenado em
        sql_results["user_query_result"] e serializado em context_parts no
        _synthesize_node. O pipeline fixo de métricas obrigatórias continua
        sendo executado independentemente.
        """
        t0 = time.perf_counter()
        try:
            if self.audit:
                self.audit.log_event(
                    AuditEvent.NODE_START, {"node": "execute_sql"}, EventStatus.INFO
                )

            # Query específica do usuário
            print("\n[execute_sql] executando query específica do usuário...")
            user_query_result = self._try_user_specific_query(state["user_query"])

            print("[execute_sql] calculando metricas obrigatorias...")
            state["mandatory_metrics"] = self._calculate_mandatory_metrics()

            print("[execute_sql] calculando dados geograficos...")
            state["geographic_data"] = self._calculate_geographic_data()

            if self.chart_tool:
                print("[execute_sql] gerando graficos padrao...")
                try:
                    paths = self.chart_tool.generate_all_charts()
                    state["chart_paths"] = [
                        p["path"] for p in (paths or [])
                        if isinstance(p, dict) and p.get("path")
                    ]
                except Exception as ce:
                    print(f"[execute_sql] aviso: falha ao gerar graficos — {ce}")
                    state["chart_paths"] = []
            else:
                state["chart_paths"] = []

            if self.web_search_tool:
                print("[execute_sql] buscando noticias recentes...")
                try:
                    state["news_results"] = self.web_search_tool.search_news(
                        query="SRAG síndrome respiratória aguda grave Brasil 2025",
                        max_results=5,
                    )
                except Exception as ne:
                    print(f"[execute_sql] aviso: falha ao buscar noticias — {ne}")
                    state["news_results"] = {}

            state["sql_results"] = {
                "metrics":            state["mandatory_metrics"],
                "charts_generated":   len(state.get("chart_paths", [])),
                "news_fetched":       len((state.get("news_results") or {}).get("articles", [])),
                "geographic_rows":    len((state.get("geographic_data") or {}).get("data", [])),
                "user_query_result":  user_query_result,
            }

            state["messages"].append(AIMessage(content="Dados SQL processados com sucesso"))

            if self.audit:
                self.audit.log_event(
                    AuditEvent.NODE_COMPLETE,
                    {
                        "node":                  "execute_sql",
                        "has_user_query_result": user_query_result is not None,
                        "duration_seconds":      round(time.perf_counter() - t0, 3),
                    },
                    EventStatus.SUCCESS,
                    duration_seconds=round(time.perf_counter() - t0, 3),
                )

        except Exception as e:
            state["errors"].append(f"SQL node error: {str(e)}")
            if self.audit:
                self.audit.log_event(
                    AuditEvent.NODE_ERROR,
                    {"node": "execute_sql", "error": str(e),
                     "duration_seconds": round(time.perf_counter() - t0, 3)},
                    EventStatus.ERROR,
                )

        return state

    # ------------------------------------------------------------------
    # Nó 2B — RAG puro
    # ------------------------------------------------------------------

    def _execute_rag_node(self, state: AgentState) -> AgentState:
        """
        Nó 2B — Recuperação semântica via Databricks Vector Search.
        """
        t0 = time.perf_counter()
        try:
            if self.audit:
                self.audit.log_event(
                    AuditEvent.NODE_START, {"node": "execute_rag"}, EventStatus.INFO
                )

            if self.rag_chain is None:
                state["messages"].append(
                    AIMessage(content="RAG nao disponivel — continuando sem contexto semantico")
                )
                return state

            rd       = state.get("routing_decision")
            rag_type = rd.rag_semantic_type if rd else None

            rag_result = self.rag_chain.invoke(
                state["user_query"],
                semantic_type_override=rag_type,
            )
            state["rag_results"] = rag_result

            for doc in rag_result.get("source_documents", []):
                if hasattr(doc, "metadata"):
                    src = (
                        doc.metadata.get("source")
                        or doc.metadata.get("file_path")
                        or str(doc.metadata)
                    )
                else:
                    src = str(doc)
                state["sources"].append(src)

            state["messages"].append(
                AIMessage(content=f"RAG executado: {len(rag_result.get('source_documents', []))} documentos")
            )

            rag_duration = rag_result.get("metadata", {}).get("duration_seconds",
                                          round(time.perf_counter() - t0, 3))
            if self.audit:
                self.audit.log_event(
                    AuditEvent.NODE_COMPLETE,
                    {"node": "execute_rag", "duration_seconds": rag_duration,
                     "num_sources": len(rag_result.get("source_documents", [])),
                     "rag_semantic_type": rag_type,
                     "quality_score": rag_result.get("validation", {}).get("quality_score")},
                    EventStatus.SUCCESS,
                )

        except Exception as e:
            state["errors"].append(f"RAG node error: {str(e)}")
            if self.audit:
                self.audit.log_event(
                    AuditEvent.NODE_ERROR,
                    {"node": "execute_rag", "error": str(e),
                     "duration_seconds": round(time.perf_counter() - t0, 3)},
                    EventStatus.ERROR,
                )

        return state

    # ------------------------------------------------------------------
    # Nó 2C — Híbrido
    # ------------------------------------------------------------------

    def _execute_hybrid_node(self, state: AgentState) -> AgentState:
        """
        Nó 2C — Pipeline híbrido: SQL + RAG em sequência com merge de estado.
        """
        t0 = time.perf_counter()
        try:
            if self.audit:
                self.audit.log_event(
                    AuditEvent.NODE_START, {"node": "execute_hybrid"}, EventStatus.INFO
                )

            sql_copy = {
                **state,
                "messages": list(state["messages"]),
                "sources":  list(state["sources"]),
                "errors":   list(state["errors"]),
            }
            rag_copy = {
                **state,
                "messages": list(state["messages"]),
                "sources":  list(state["sources"]),
                "errors":   list(state["errors"]),
            }

            sql_state = self._execute_sql_node(sql_copy)
            state["sql_results"]       = sql_state.get("sql_results", {})
            state["news_results"]      = sql_state.get("news_results", {})
            state["chart_paths"]       = sql_state.get("chart_paths", [])
            state["mandatory_metrics"] = sql_state.get("mandatory_metrics", {})
            state["geographic_data"]   = sql_state.get("geographic_data", {})

            rag_state = self._execute_rag_node(rag_copy)
            state["rag_results"] = rag_state.get("rag_results", {})
            state["sources"]     = list(state["sources"]) + list(rag_state.get("sources", []))

            state["errors"].extend(sql_state.get("errors", []))
            state["errors"].extend(rag_state.get("errors", []))

            state["messages"].append(AIMessage(content="Execucao hibrida (SQL + RAG) completa"))

            if self.audit:
                self.audit.log_event(
                    AuditEvent.NODE_COMPLETE,
                    {"node": "execute_hybrid", "duration_seconds": round(time.perf_counter() - t0, 3)},
                    EventStatus.SUCCESS,
                )

        except Exception as e:
            state["errors"].append(f"Hybrid node critical error: {str(e)}")
            if self.audit:
                self.audit.log_event(
                    AuditEvent.NODE_ERROR,
                    {"node": "execute_hybrid", "error": str(e),
                     "stack_trace": traceback.format_exc()[:500],
                     "duration_seconds": round(time.perf_counter() - t0, 3)},
                    EventStatus.ERROR,
                )
            state["messages"].append(
                AIMessage(content=f"Aviso: execucao parcial — {str(e)[:100]}")
            )

        return state

    # ------------------------------------------------------------------
    # Nó 2D — Chart ad-hoc
    # ------------------------------------------------------------------

    def _execute_chart_node(self, state: AgentState) -> AgentState:
        """
        Nó 2D — Gráfico ad-hoc a partir de ChartParams extraídos pelo IntentRouter.
        """
        ALLOWED_TABLES = {
            "gold_metricas_temporais",
            "gold_metricas_geograficas",
            "gold_metricas_demograficas",
        }
        ALLOWED_METRICS = {
            "total_casos", "taxa_mortalidade", "taxa_uti", "taxa_vacinacao",
            "total_obitos", "total_internados", "total_uti", "total_vacinados",
        }
        ALLOWED_GROUP_BY = {
            "ano_mes", "sg_uf", "faixa_etaria", "semana_epidemiologica",
        }

        t0 = time.perf_counter()
        try:
            if self.audit:
                self.audit.log_event(
                    AuditEvent.NODE_START, {"node": "execute_chart"}, EventStatus.INFO
                )

            decision = state["routing_decision"]
            params: ChartParams = decision.chart_params

            if params is None:
                raise ValueError("ChartParams ausente no RoutingDecision — verifique o IntentRouter")

            table    = params.table    if params.table    in ALLOWED_TABLES   else "gold_metricas_temporais"
            metric   = params.metric   if params.metric   in ALLOWED_METRICS  else "total_casos"
            group_by = params.group_by if params.group_by in ALLOWED_GROUP_BY else "ano_mes"

            print(f"[execute_chart] table={table} metric={metric} group_by={group_by} type={params.chart_type}")

            sql    = self._build_dynamic_chart_query(
                catalog=self.catalog, schema=self.schema,
                table=table, metric=metric, group_by=group_by,
                filters=params.filters,
            )
            result = self.sql_tool.execute_query(sql)

            if not result.get("success") or not result.get("data"):
                raise ValueError(f"Query retornou vazio: {result.get('error', 'sem dados')}")

            if self.chart_tool is None:
                raise ValueError(
                    "chart_tool nao disponivel — passe chart_tool= ao instanciar o orquestrador"
                )

            chart_result = self.chart_tool.generate_custom_chart(
                data=result["data"],
                chart_type=params.chart_type,
                title=params.title,
                x_col=group_by,
                y_col=metric,
            )

            if chart_result is None:
                raise ValueError("generate_custom_chart retornou None")

            chart_path = chart_result["path"]
            state["ad_hoc_chart_path"] = chart_path
            state["chart_paths"]       = list(state.get("chart_paths") or [])
            state["chart_paths"].append(chart_path)
            state["messages"].append(AIMessage(content=f"Grafico gerado: {chart_path}"))

            if self.audit:
                self.audit.log_event(
                    AuditEvent.CHART_GENERATED,
                    {
                        "node":             "execute_chart",
                        "chart_path":       chart_path,
                        "chart_type":       params.chart_type,
                        "metric":           metric,
                        "group_by":         group_by,
                        "data_rows":        len(result["data"]),
                        "duration_seconds": round(time.perf_counter() - t0, 3),
                    },
                    EventStatus.SUCCESS,
                )

        except Exception as e:
            state["errors"].append(f"Chart node error: {str(e)}")
            print(f"[execute_chart] erro: {e}")
            if self.audit:
                self.audit.log_event(
                    AuditEvent.NODE_ERROR,
                    {"node": "execute_chart", "error": str(e),
                     "stack_trace": traceback.format_exc()[:500],
                     "duration_seconds": round(time.perf_counter() - t0, 3)},
                    EventStatus.ERROR,
                )
            state["messages"].append(
                AIMessage(content=f"Nao foi possivel gerar o grafico: {str(e)[:120]}")
            )

        return state

    # ------------------------------------------------------------------
    # Nó 2E — Relatório epidemiológico completo
    # ------------------------------------------------------------------

    def _execute_report_node(self, state: AgentState) -> AgentState:
        """
        Nó 2E — Relatório epidemiológico completo: SQL + RAG obrigatórios.
        """
        t0 = time.perf_counter()
        try:
            if self.audit:
                self.audit.log_event(
                    AuditEvent.NODE_START, {"node": "execute_report"}, EventStatus.INFO
                )

            sql_copy = {
                **state,
                "messages": list(state["messages"]),
                "sources":  list(state["sources"]),
                "errors":   list(state["errors"]),
            }
            sql_state = self._execute_sql_node(sql_copy)
            state["sql_results"]       = sql_state.get("sql_results", {})
            state["news_results"]      = sql_state.get("news_results", {})
            state["chart_paths"]       = sql_state.get("chart_paths", [])
            state["mandatory_metrics"] = sql_state.get("mandatory_metrics", {})
            state["geographic_data"]   = sql_state.get("geographic_data", {})
            state["errors"].extend(sql_state.get("errors", []))

            if self.rag_chain:
                rag_copy = {
                    **state,
                    "messages": list(state["messages"]),
                    "sources":  list(state["sources"]),
                    "errors":   list(state["errors"]),
                }
                rag_state = self._execute_rag_node(rag_copy)
                state["rag_results"] = rag_state.get("rag_results", {})
                state["sources"]     = list(state["sources"]) + list(rag_state.get("sources", []))
                state["errors"].extend(rag_state.get("errors", []))
            else:
                state["rag_results"] = {}

            state["messages"].append(AIMessage(content="Relatorio epidemiologico completo gerado"))

            if self.audit:
                self.audit.log_event(
                    AuditEvent.NODE_COMPLETE,
                    {
                        "node":             "execute_report",
                        "charts_generated": len(state.get("chart_paths") or []),
                        "has_rag":          bool(state.get("rag_results")),
                        "duration_seconds": round(time.perf_counter() - t0, 3),
                    },
                    EventStatus.SUCCESS,
                )

        except Exception as e:
            state["errors"].append(f"Report node critical error: {str(e)}")
            if self.audit:
                self.audit.log_event(
                    AuditEvent.NODE_ERROR,
                    {"node": "execute_report", "error": str(e),
                     "stack_trace": traceback.format_exc()[:500],
                     "duration_seconds": round(time.perf_counter() - t0, 3)},
                    EventStatus.ERROR,
                )
            state["messages"].append(
                AIMessage(content=f"Aviso: relatorio parcial — {str(e)[:100]}")
            )

        return state

    # ------------------------------------------------------------------
    # Nó 3 — Síntese
    # ------------------------------------------------------------------

    def _synthesize_node(self, state: AgentState) -> AgentState:
        """
        Nó 3 — Síntese final via LLM.

        Serializa em context_parts: métricas obrigatórias, resultado da query
        específica do usuário, gráficos gerados, dados geográficos, contexto RAG,
        notícias, análise anual e crescimento mensal. As diretrizes de análise
        são geradas dinamicamente a partir de mandatory_metrics quando disponível,
        ou substituídas por instrução de não inventar dados quando ausente.
        """
        t0 = time.perf_counter()
        try:
            if self.audit:
                self.audit.log_event(
                    AuditEvent.NODE_START, {"node": "synthesize"}, EventStatus.INFO
                )

            query             = state["user_query"]
            sql_results       = state.get("sql_results") or {}
            rag_results       = state.get("rag_results") or {}
            news_results      = state.get("news_results") or {}
            chart_paths       = state.get("chart_paths") or []
            mandatory_metrics = state.get("mandatory_metrics") or {}
            geographic_data   = state.get("geographic_data") or {}
            ad_hoc_path       = state.get("ad_hoc_chart_path")
            user_query_result = sql_results.get("user_query_result")

            context_parts = []

            # ── Métricas obrigatórias ─────────────────────────────────────────
            if mandatory_metrics:
                context_parts.append("=" * 70)
                context_parts.append("⚠️ MÉTRICAS OBRIGATÓRIAS SRAG (INCLUIR TODAS NA RESPOSTA) ⚠️")
                context_parts.append("=" * 70)

                taxa_cresc = mandatory_metrics.get("taxa_crescimento")
                data_ref   = mandatory_metrics.get("data_referencia", "N/A")

                if taxa_cresc is None:
                    context_parts.append("📈 Taxa de Crescimento Diário: ⚠️ ERRO NO CÁLCULO")
                elif isinstance(taxa_cresc, str) and taxa_cresc == "ERRO":
                    context_parts.append("📈 Taxa de Crescimento Diário: ❌ FALHA")
                elif taxa_cresc == 0.0:
                    context_parts.append(
                        "📈 Taxa de Crescimento Diário: 0.00% ⚠️ (SUSPEITO — validar dados)"
                    )
                else:
                    context_parts.append(
                        f"📈 Taxa de Crescimento Diário: {taxa_cresc:.2f}% (ref: {data_ref})"
                    )

                context_parts.append(
                    f"💀 Taxa de Mortalidade: {mandatory_metrics.get('taxa_mortalidade', 0):.2f}%"
                )
                context_parts.append(
                    f"🏥 Taxa de Ocupação UTI: {mandatory_metrics.get('taxa_uti', 0):.2f}%"
                )
                context_parts.append(
                    f"💉 Taxa de Vacinação: {mandatory_metrics.get('taxa_vacinacao', 0):.2f}%"
                )
                context_parts.append(
                    f"📊 Total de Casos: {mandatory_metrics.get('total_casos', 0):,}"
                )
                context_parts.append("=" * 70)
                context_parts.append("")

            # ── Resultado da query específica do usuário ─────────────────────
            if user_query_result and user_query_result.get("data"):
                context_parts.append("=" * 70)
                context_parts.append("RESULTADO DA CONSULTA ESPECÍFICA DO USUÁRIO")
                context_parts.append("=" * 70)
                rows = user_query_result["data"]
                if rows:
                    # Cabeçalho dinâmico baseado nas chaves do primeiro registro
                    cols = list(rows[0].keys())
                    header = " | ".join(f"{c:>15}" for c in cols)
                    context_parts.append(header)
                    context_parts.append("-" * len(header))
                    for row in rows[:50]:   # máximo 50 linhas no contexto
                        line = " | ".join(
                            f"{str(row.get(c, ''))[:15]:>15}" for c in cols
                        )
                        context_parts.append(line)
                    if len(rows) > 50:
                        context_parts.append(f"  ... e mais {len(rows) - 50} linhas")
                context_parts.append(
                    "→ INSTRUÇÃO: use estes dados para responder diretamente à pergunta do usuário."
                )
                context_parts.append("=" * 70)
                context_parts.append("")

            # ── Gráfico ad-hoc ────────────────────────────────────────────────
            if ad_hoc_path:
                context_parts.append("GRÁFICO AD-HOC GERADO:")
                context_parts.append(f"  📊 {ad_hoc_path.split('/')[-1]}")
                context_parts.append("  (resultado direto da solicitação do usuário)")
                context_parts.append("")

            # ── Sumário SQL ───────────────────────────────────────────────────
            if sql_results:
                context_parts.append("DADOS SQL DISPONÍVEIS:")
                context_parts.append(f"- Métricas calculadas: {len(mandatory_metrics)} obrigatórias")
                context_parts.append(f"- Gráficos padrão: {sql_results.get('charts_generated', 0)}")
                context_parts.append(f"- Notícias obtidas: {sql_results.get('news_fetched', 0)}")
                context_parts.append("")

            # ── Gráficos padrão ───────────────────────────────────────────────
            # Confirma explicitamente quais arquivos foram gerados para que o LLM
            # não diga "não temos acesso direto aos gráficos".
            standard_charts = [p for p in chart_paths if p and p != ad_hoc_path]
            if standard_charts:
                context_parts.append(f"GRÁFICOS PADRÃO GERADOS COM SUCESSO ({len(standard_charts)}):")
                for i, p in enumerate(standard_charts, 1):
                    context_parts.append(f"  {i}. 📊 {p.split('/')[-1]}")
                context_parts.append(
                    "→ INSTRUÇÃO: liste estes gráficos na resposta e descreva o que cada um mostra."
                )
                context_parts.append("")

            # ── Dados geográficos ─────────────────────────────────────────────
            geo_rows = geographic_data.get("data", [])
            if geo_rows:
                context_parts.append("=" * 70)
                context_parts.append("DISTRIBUIÇÃO GEOGRÁFICA — TOP 10 UFs POR CASOS DE SRAG")
                context_parts.append("=" * 70)
                context_parts.append(
                    f"{'UF':<6} {'Casos':>10} {'Óbitos':>10} {'Mortalidade':>12}"
                )
                context_parts.append("-" * 44)
                for row in geo_rows:
                    uf    = row.get("sg_uf", "N/A")
                    casos = int(row.get("total_casos", 0))
                    obit  = int(row.get("total_obitos", 0))
                    mort  = float(row.get("taxa_mortalidade", 0))
                    context_parts.append(
                        f"{uf:<6} {casos:>10,} {obit:>10,} {mort:>11.2f}%"
                    )
                context_parts.append("=" * 70)
                context_parts.append(
                    "→ INSTRUÇÃO: cite os estados com mais casos e compare as taxas de mortalidade."
                )
                context_parts.append("")

            # ── Contexto RAG ──────────────────────────────────────────────────
            if rag_results and rag_results.get("answer"):
                context_parts.append("CONTEXTO RAG:")
                context_parts.append(rag_results["answer"][:2000])
                context_parts.append("")

            # ── Notícias ──────────────────────────────────────────────────────
            if news_results and news_results.get("articles"):
                context_parts.append("NOTÍCIAS RECENTES:")
                for news in news_results["articles"][:3]:
                    context_parts.append(f"- {news.get('title', 'N/A')}")
                context_parts.append("")

            # ── Análise anual ─────────────────────────────────────────────────
            analise_anual = mandatory_metrics.get("analise_anual", [])
            if analise_anual:
                context_parts.append("=" * 70)
                context_parts.append("ANÁLISE ANUAL COMPARATIVA (USE PARA CONTEXTUALIZAR TENDÊNCIAS)")
                context_parts.append("=" * 70)
                for row in analise_anual:
                    ano   = row.get("ano", "?")
                    casos = row.get("casos_ano", 0)
                    mort  = row.get("mortalidade_pct", 0)
                    uti   = row.get("uti_pct", 0)
                    vac   = row.get("vacinacao_pct", 0)
                    context_parts.append(
                        f"  {ano}: {casos:,} casos | mortalidade {mort}% | UTI {uti}% | vacinação {vac}%"
                    )
                context_parts.append(
                    "  → INSTRUÇÃO: mencione a tendência de queda da mortalidade entre os anos."
                )
                context_parts.append("")

            # ── Crescimento mensal ────────────────────────────────────────────
            cresc_mensal = mandatory_metrics.get("crescimento_mensal", [])
            if cresc_mensal:
                context_parts.append("CRESCIMENTO MENSAL MÊS A MÊS (últimos 6 meses):")
                for row in cresc_mensal[:6]:
                    mes       = row.get("ano_mes", "?")
                    casos     = row.get("total_casos", 0)
                    cresc     = row.get("crescimento_mensal_pct")
                    cresc_str = f"{cresc:+.1f}%" if cresc is not None else "N/A"
                    context_parts.append(f"  {mes}: {casos:,} casos ({cresc_str} vs mês anterior)")
                context_parts.append("")

            # ── Avisos de qualidade ───────────────────────────────────────────
            context_parts.append("=" * 70)
            context_parts.append("⚠️  AVISOS DE QUALIDADE DE DADO — MENCIONAR NA ANÁLISE")
            context_parts.append("=" * 70)
            context_parts.append(
                "1. SAZONALIDADE: mortalidade tem padrão sazonal claro no Brasil. "
                "Pico em janeiro (início do ano) e junho (inverno). "
                "A média anual esconde essa variação — contextualize sempre."
            )
            context_parts.append(
                "2. SUBNOTIFICAÇÃO RECENTE: os últimos 14 dias da série diária mostram "
                "queda artificial de casos por atraso de notificação no SIVEP-Gripe. "
                "A taxa de crescimento é calculada sobre dados consolidados (dias 15-21), "
                "não sobre os dias mais recentes."
            )
            if mandatory_metrics:
                vac = mandatory_metrics.get("taxa_vacinacao", 0)
                context_parts.append(
                    f"3. VACINAÇÃO ABRIL-MAIO/2025: a taxa consolidada de {vac:.2f}% pode estar "
                    "subestimada por subregistro em abril-maio/2025 (8.5% e 16.8%). "
                    "A média de jun-dez/2025 (~37%) é o valor mais confiável."
                )
            context_parts.append(
                "4. TENDÊNCIA INTRA-ANUAL POSITIVA: mortalidade caiu de ~12% (jan/2025) "
                "para ~5.5% (nov/2025). O cenário ao final de 2025 é melhor que a média anual sugere."
            )
            context_parts.append("=" * 70)
            context_parts.append("")

            # ── Instrução por modo ────────────────────────────────────────────
            is_chart_mode = ad_hoc_path is not None
            has_sql_data  = bool(mandatory_metrics)

            if is_chart_mode:
                # Modo ad-hoc: confirma geração do gráfico específico
                chart_instruction = f"""
O usuário pediu um gráfico. Você JÁ GEROU o gráfico com sucesso:
  📊 Arquivo: {ad_hoc_path.split('/')[-1] if ad_hoc_path else 'N/A'}

Na sua resposta:
1. Confirme que o gráfico foi gerado e informe o nome do arquivo
2. Descreva brevemente o que o gráfico mostra
3. Adicione 2-3 insights relevantes com base nos dados
4. Se houver métricas disponíveis no contexto, mencione-as de forma resumida
"""
            elif standard_charts:
                # Modo padrão com gráficos: confirma todos os arquivos gerados
                charts_list = chr(10).join(
                    f"  {i}. 📊 {p.split('/')[-1]}" for i, p in enumerate(standard_charts, 1)
                )
                chart_instruction = f"""
Os seguintes {len(standard_charts)} gráficos foram gerados com sucesso e estão disponíveis:
{charts_list}

Na sua resposta:
1. Liste estes gráficos confirmando que foram gerados
2. Descreva brevemente o que cada um mostra
3. SEMPRE incluir as 4 métricas obrigatórias em destaque
4. Termine com insights e recomendações
"""
            else:
                chart_instruction = """
1. SEMPRE incluir as 4 métricas obrigatórias na resposta (destacadas com ícones)
2. COMEÇAR a resposta mostrando as métricas em destaque
3. Termine com insights e recomendações
"""

            # ── Diretrizes de análise ─────────────────────────────────────────
            # Quando mandatory_metrics está vazio (RAG_ONLY), não injetar valores
            # numéricos hardcoded. O LLM deve basear a resposta apenas no contexto
            # RAG disponível, sem inventar dados SQL.
            if has_sql_data:
                analise_rows = mandatory_metrics.get("analise_anual", [])
                if analise_rows and len(analise_rows) >= 2:
                    anos_str = " → ".join(
                        f"{r.get('ano')}: {r.get('mortalidade_pct')}%"
                        for r in reversed(analise_rows[:3])
                    )
                    tendencia_anual = f"- Ao comparar anos, destaque a queda da mortalidade: {anos_str}"
                else:
                    tendencia_anual = "- Destaque a tendência de queda da mortalidade ano a ano se os dados mostrarem isso"

                vac_val = mandatory_metrics.get("taxa_vacinacao", 0)
                tc_val  = mandatory_metrics.get("taxa_crescimento")
                tc_str  = f"{tc_val:.2f}%" if isinstance(tc_val, (int, float)) else "indisponível"

                analysis_directives = f"""
- Contextualize as métricas com a tendência temporal (sazonalidade, variação intra-anual)
- A mortalidade média anual ESCONDE o padrão sazonal — mencione os picos de janeiro e junho
- A taxa de vacinação de {vac_val:.2f}% pode estar subestimada por subregistro em abril-maio/2025
- A tendência de 2025 é POSITIVA: mortalidade caiu consistentemente de jan a nov/2025
{tendencia_anual}
- Taxa de crescimento de {tc_str} calculada sobre dados consolidados (não os mais recentes, que têm subnotificação)
- Termine SEMPRE com insights acionáveis e recomendações concretas de saúde pública
"""
            else:
                # RAG_ONLY ou ausência de dados SQL
                analysis_directives = """
- Base sua resposta EXCLUSIVAMENTE no contexto RAG disponível acima
- NÃO invente dados numéricos (taxas, totais) que não estejam explicitamente no contexto
- Se não houver dados suficientes para um aspecto da pergunta, diga isso claramente
- Termine com insights baseados no que foi encontrado no contexto semântico
"""

            synthesis_prompt = f"""
Você é um especialista em epidemiologia analisando dados de SRAG (Síndrome Respiratória Aguda Grave) no Brasil.

🚨 INSTRUÇÕES OBRIGATÓRIAS:
{chart_instruction}

📋 DIRETRIZES DE ANÁLISE EPIDEMIOLÓGICA:
{analysis_directives}

Responda em português brasileiro com tom técnico-profissional.

CONTEXTO E DADOS DISPONÍVEIS:
{chr(10).join(context_parts)}

PERGUNTA DO USUÁRIO:
{query}

SUA RESPOSTA COMPLETA E ESTRUTURADA:
"""

            response = self._get_synthesis_llm().invoke([HumanMessage(content=synthesis_prompt)])
            state["final_answer"] = response.content
            state["messages"].append(AIMessage(content="Sintese completa"))

            if self.audit:
                self.audit.log_event(
                    AuditEvent.NODE_COMPLETE,
                    {
                        "node":                  "synthesize",
                        "answer_length":         len(response.content),
                        "chart_mode":            is_chart_mode,
                        "has_sql_data":          has_sql_data,
                        "geo_rows_in_context":   len(geo_rows),
                        "user_query_result":     user_query_result is not None,
                        "standard_charts_count": len(standard_charts),
                        "llm_provider":          "openai" if self.use_openai else "databricks",
                        "duration_seconds":      round(time.perf_counter() - t0, 3),
                    },
                    EventStatus.SUCCESS,
                    duration_seconds=round(time.perf_counter() - t0, 3),
                )

        except Exception as e:
            state["errors"].append(f"Synthesize error: {str(e)}")
            if self.audit:
                self.audit.log_event(
                    AuditEvent.SYNTHESIS_ERROR,
                    {
                        "error":            str(e),
                        "query":            state.get("user_query", "")[:200],
                        "has_sql_results":  bool(state.get("sql_results")),
                        "has_rag_results":  bool(state.get("rag_results")),
                        "stack_trace":      traceback.format_exc()[:500],
                        "duration_seconds": round(time.perf_counter() - t0, 3),
                    },
                    EventStatus.ERROR,
                )
            state["final_answer"] = (
                "Nao foi possivel gerar sintese completa devido a erro tecnico.\n\n"
                f"METRICAS CALCULADAS:\n{chr(10).join(context_parts[:10])}\n\n"
                "Por favor, consulte os logs para mais detalhes."
            )

        return state

    # =========================================================================
    # SQL BUILDER — User Specific Query
    # =========================================================================

    def _try_user_specific_query(self, user_query: str) -> Optional[Dict]:
        """
        Gera e executa SQL direcionada à pergunta específica do usuário.

        Usa o LLM para traduzir a query em linguagem natural para SQL válida,
        restringe o escopo às tabelas Gold permitidas e executa via sql_tool.

        O resultado é retornado como dict com "data" e "rows" para serialização
        em context_parts. Em caso de falha (SQL inválida, erro de execução,
        timeout), retorna None silenciosamente — o pipeline continua com as
        métricas obrigatórias como fonte principal de dados.

        O prompt proíbe window functions (LAG, LEAD, RANK, ROW_NUMBER, OVER),
        subqueries escalares aninhadas e CTEs com window functions internas,
        pois essas construções causam [MISSING_GROUP_BY] no Spark SQL. A query
        gerada deve ser SELECT plano com GROUP BY explícito, agregações diretas
        (SUM, COUNT, AVG, ROUND), ORDER BY e LIMIT.

        Re.sub remove blocos ```sql``` que alguns modelos inserem mesmo com
        instrução contrária. A verificação de prefixo SELECT impede execução
        de DML acidental.

        Parâmetros
        ----------
        user_query : str — pergunta do usuário em linguagem natural.

        Retorno
        -------
        Dict com "data" (List[Dict]) e "rows" (int) quando bem-sucedido.
        None quando a geração ou execução falhar.
        """
        try:
            tables_desc = "\n".join(
                f"- {self.catalog}.{self.schema}.{tbl}: {cols}"
                for tbl, cols in _ALLOWED_TABLES_FOR_USER_SQL.items()
            )

            sql_gen_prompt = f"""Você é um especialista SQL em Databricks Unity Catalog.
Gere UMA query SQL simples para responder à pergunta abaixo.

Tabelas disponíveis:
{tables_desc}

Regras OBRIGATÓRIAS — siga todas sem exceção:
- Apenas SELECT — sem INSERT, UPDATE, DELETE, DROP, ALTER
- Sempre incluir LIMIT 100
- Referenciar tabelas com catálogo completo: {self.catalog}.{self.schema}.<tabela>
- Retornar APENAS a query SQL, sem explicações, sem markdown, sem blocos de código

Restrições de sintaxe — PROIBIDO usar qualquer um dos itens abaixo:
- Window functions: LAG, LEAD, RANK, DENSE_RANK, ROW_NUMBER, NTILE, OVER, PARTITION BY
- Subqueries escalares: SELECT dentro de SELECT sem cláusula FROM própria
- CTEs (WITH ... AS) que contenham window functions internamente
- Qualquer construção que exija GROUP BY implícito sem declaração explícita

Sintaxe PERMITIDA — use apenas:
- SELECT com colunas diretas ou agregações: SUM(), COUNT(), AVG(), ROUND(), MAX(), MIN()
- FROM com uma única tabela
- WHERE para filtros simples
- GROUP BY explícito quando houver agregações
- ORDER BY
- LIMIT
- Uma CTE simples (WITH ... AS) sem window functions internas, se necessário para clareza

Pergunta: {user_query}
"""
            response = self.llm.invoke([HumanMessage(content=sql_gen_prompt)])
            sql = response.content.strip()

            # Remove blocos markdown que alguns modelos inserem mesmo com instrução contrária
            sql = re.sub(r"```(?:sql)?\n?", "", sql, flags=re.IGNORECASE)
            sql = re.sub(r"```", "", sql)
            sql = sql.strip()

            # Quando o LLM gera múltiplos statements separados por ";", pega apenas o primeiro.
            # Queries compostas falham no Spark SQL com ParseException.
            statements = [s.strip() for s in sql.split(";") if s.strip()]
            if len(statements) > 1:
                print(f"[user_sql] aviso: LLM gerou {len(statements)} statements — usando apenas o primeiro")
            sql = statements[0] if statements else sql

            if not sql.lower().lstrip().startswith("select"):
                print(f"[user_sql] aviso: LLM não gerou SELECT válido — descartado")
                return None

            # Validação defensiva: rejeita window functions mesmo que o LLM ignore o prompt.
            # Estas construções causam MISSING_GROUP_BY ou PARSE_SYNTAX_ERROR no Spark SQL.
            _forbidden = re.compile(
                r"\b(LAG|LEAD|RANK|DENSE_RANK|ROW_NUMBER|NTILE|OVER|PARTITION\s+BY)\b",
                re.IGNORECASE,
            )
            match = _forbidden.search(sql)
            if match:
                print(f"[user_sql] aviso: SQL contém '{match.group(0)}' (window function proibida) — descartado")
                return None

            # Verifica equilíbrio de parênteses antes de enviar ao Spark.
            if sql.count("(") != sql.count(")"):
                print(f"[user_sql] aviso: SQL com parênteses desbalanceados — descartado")
                return None

            result = self.sql_tool.execute_query(sql)
            if result.get("success") and result.get("data"):
                print(f"[user_sql] {result['rows']} linhas retornadas para query específica")
                return {"data": result["data"], "rows": result["rows"]}

            return None

        except Exception as e:
            print(f"[user_sql] aviso: query específica falhou — {e}")
            return None

    # =========================================================================
    # SQL BUILDER — Chart Ad-hoc
    # =========================================================================

    def _build_dynamic_chart_query(
        self,
        catalog:  str,
        schema:   str,
        table:    str,
        metric:   str,
        group_by: str,
        filters:  Dict,
        limit:    int = 500,
    ) -> str:
        """
        Constrói a SQL de agregação para gráfico ad-hoc.
        """

        def sanitize(v: str) -> str:
            return re.sub(r"[^A-Za-z0-9_\-]", "", str(v))

        where_clauses = [f"{metric} IS NOT NULL", f"{group_by} IS NOT NULL"]

        if "ano" in filters:
            ano = sanitize(filters["ano"])
            if group_by == "ano_mes":
                where_clauses.append(f"ano_mes LIKE '{ano}-%'")
            else:
                where_clauses.append(f"YEAR(dt_sintomas) = {ano}")

        if "sg_uf" in filters:
            where_clauses.append(f"sg_uf = '{sanitize(filters['sg_uf'])}'")

        if "mes" in filters:
            mes        = sanitize(filters["mes"])
            mes_padded = str(int(mes)).zfill(2)
            if table == "gold_serie_diaria_30d":
                where_clauses.append(f"MONTH(dt_sintomas) = {int(mes)}")
            else:
                where_clauses.append(f"SUBSTRING(ano_mes, 6, 2) = '{mes_padded}'")

        where_str = " AND ".join(where_clauses)

        return f"""
        SELECT
            {group_by},
            SUM({metric}) AS {metric}
        FROM {catalog}.{schema}.{table}
        WHERE {where_str}
        GROUP BY {group_by}
        ORDER BY {group_by} ASC
        LIMIT {limit}
        """.strip()

    # =========================================================================
    # GEOGRAPHIC DATA
    # =========================================================================

    def _calculate_geographic_data(self) -> Dict:
        """
        Calcula top 10 UFs por total de casos e taxa de mortalidade.
        """
        try:
            result = self.sql_tool.execute_query(f"""
                SELECT
                    sg_uf,
                    SUM(total_casos)  AS total_casos,
                    SUM(total_obitos) AS total_obitos,
                    CASE
                        WHEN SUM(casos_com_desfecho) > 0
                        THEN (SUM(total_obitos) / SUM(casos_com_desfecho)) * 100
                        ELSE 0
                    END AS taxa_mortalidade
                FROM {self.catalog}.{self.schema}.gold_metricas_geograficas
                WHERE sg_uf IS NOT NULL
                  AND total_casos IS NOT NULL
                GROUP BY sg_uf
                ORDER BY total_casos DESC
                LIMIT 10
            """)

            if result.get("success") and result.get("data"):
                rows = [
                    {
                        "sg_uf":            r.get("sg_uf", "N/A"),
                        "total_casos":      int(r.get("total_casos", 0)),
                        "total_obitos":     int(r.get("total_obitos", 0)),
                        "taxa_mortalidade": round(float(r.get("taxa_mortalidade", 0)), 2),
                    }
                    for r in result["data"]
                ]
                print(f"[geographic] {len(rows)} UFs calculadas")
                if self.audit:
                    self.audit.log_event(
                        AuditEvent.METRICS_COLLECTED,
                        {"action": "geographic_data", "ufs": len(rows)},
                        EventStatus.SUCCESS,
                    )
                return {"data": rows}

            print("[geographic] query nao retornou dados")
            return {"data": []}

        except Exception as e:
            print(f"[geographic] aviso: {e}")
            if self.audit:
                self.audit.log_event(
                    AuditEvent.METRICS_COLLECTED,
                    {"action": "geographic_data", "error": str(e)},
                    EventStatus.WARNING,
                )
            return {"data": []}

    # =========================================================================
    # MANDATORY METRICS
    # =========================================================================

    def _calculate_mandatory_metrics(self) -> Dict:
        """
        Calcula as 4 métricas obrigatórias do projeto SRAG via Spark SQL.
        """
        _metrics_error_event = getattr(AuditEvent, "METRICS_ERROR", AuditEvent.NODE_ERROR)

        try:
            if self.audit:
                self.audit.log_event(
                    AuditEvent.METRICS_COLLECTED,
                    {"action": "calculate_mandatory_metrics"},
                    EventStatus.INFO,
                )

            metrics = {}

            # ── Taxa de Crescimento Diário ────────────────────────────────────
            try:
                query_crescimento = f"""
                WITH serie_completa AS (
                    SELECT dt_sintomas, total_casos,
                           LAG(total_casos) OVER (ORDER BY dt_sintomas) AS casos_dia_anterior
                    FROM {self.catalog}.{self.schema}.gold_serie_diaria_30d
                    WHERE total_casos IS NOT NULL AND total_casos > 0
                    ORDER BY dt_sintomas DESC
                    LIMIT 45
                ),
                serie_sem_subnotificacao AS (
                    SELECT dt_sintomas, total_casos, casos_dia_anterior,
                           CASE WHEN casos_dia_anterior > 0
                                THEN ((total_casos - casos_dia_anterior)
                                      / casos_dia_anterior) * 100
                                ELSE NULL END AS cresc_diario,
                           ROW_NUMBER() OVER (ORDER BY dt_sintomas DESC) AS rn
                    FROM serie_completa
                    WHERE casos_dia_anterior IS NOT NULL
                )
                SELECT
                    MAX(dt_sintomas)            AS data_referencia,
                    MAX(total_casos)            AS casos_hoje,
                    ROUND(AVG(cresc_diario), 2) AS taxa_crescimento,
                    ROUND(MIN(cresc_diario), 2) AS min_crescimento,
                    ROUND(MAX(cresc_diario), 2) AS max_crescimento,
                    COUNT(*)                    AS dias_calculados
                FROM serie_sem_subnotificacao
                WHERE rn BETWEEN 15 AND 21
                  AND cresc_diario IS NOT NULL
                """
                result = self.sql_tool.execute_query(query_crescimento)

                if result.get("success") and result.get("data"):
                    d    = result["data"][0]
                    taxa = float(d.get("taxa_crescimento") or 0)
                    metrics["taxa_crescimento"]      = round(taxa, 2)
                    metrics["casos_hoje"]            = int(d.get("casos_hoje") or 0)
                    metrics["data_referencia"]       = str(d.get("data_referencia") or "N/A")
                    metrics["crescimento_min_7d"]    = float(d.get("min_crescimento") or 0)
                    metrics["crescimento_max_7d"]    = float(d.get("max_crescimento") or 0)
                    metrics["crescimento_dias_calc"] = int(d.get("dias_calculados") or 0)
                    print(
                        f"   [crescimento] media_7d={taxa:.2f}%  "
                        f"min={metrics['crescimento_min_7d']:.1f}%  "
                        f"max={metrics['crescimento_max_7d']:.1f}%"
                    )
                else:
                    raise ValueError("Query crescimento retornou vazio")

            except Exception as e:
                print(f"   [crescimento] falhou: {e} — tentando fallback 7 dias...")
                if self.audit:
                    self.audit.log_event(
                        _metrics_error_event,
                        {"metric": "taxa_crescimento", "error": str(e),
                         "action": "attempting_7day_fallback"},
                        EventStatus.WARNING,
                    )
                try:
                    fallback_query = f"""
                    WITH dados_recentes AS (
                        SELECT dt_sintomas, total_casos,
                               LAG(total_casos) OVER (ORDER BY dt_sintomas) AS casos_anterior
                        FROM {self.catalog}.{self.schema}.gold_serie_diaria_30d
                        WHERE total_casos IS NOT NULL AND total_casos > 0
                        ORDER BY dt_sintomas DESC
                        LIMIT 7
                    )
                    SELECT AVG(
                        CASE WHEN casos_anterior > 0
                             THEN ((total_casos - casos_anterior) / casos_anterior) * 100
                             ELSE 0 END
                    ) AS taxa_crescimento_media
                    FROM dados_recentes
                    WHERE casos_anterior IS NOT NULL
                    LIMIT 1
                    """
                    result = self.sql_tool.execute_query(fallback_query)
                    if result.get("success") and result.get("data"):
                        taxa = float(result["data"][0].get("taxa_crescimento_media", 0) or 0)
                        metrics["taxa_crescimento"] = round(taxa, 2)
                        metrics["data_referencia"]  = "Media 7 dias"
                        print(f"   [crescimento] fallback ok: {metrics['taxa_crescimento']:.2f}%")
                    else:
                        raise ValueError("Fallback tambem falhou")
                except Exception as fe:
                    print(f"   [crescimento] fallback falhou: {fe}")
                    metrics["taxa_crescimento"]       = None
                    metrics["taxa_crescimento_error"] = str(e)
                    metrics["data_referencia"]        = "ERRO"

            # ── Taxa de Mortalidade ───────────────────────────────────────────
            try:
                result = self.sql_tool.execute_query(f"""
                SELECT
                    SUM(total_casos)        AS total_casos,
                    SUM(total_obitos)       AS total_obitos,
                    SUM(casos_com_desfecho) AS casos_com_desfecho,
                    CASE WHEN SUM(casos_com_desfecho) > 0
                         THEN (SUM(total_obitos) / SUM(casos_com_desfecho)) * 100
                         ELSE 0 END AS taxa_mortalidade
                FROM {self.catalog}.{self.schema}.gold_metricas_temporais
                WHERE total_casos IS NOT NULL
                  AND total_obitos IS NOT NULL
                  AND casos_com_desfecho IS NOT NULL
                LIMIT 1
                """)
                if result.get("success") and result.get("data"):
                    d = result["data"][0]
                    metrics["taxa_mortalidade"]   = round(float(d.get("taxa_mortalidade", 0)), 2)
                    metrics["total_casos"]        = int(d.get("total_casos", 0))
                    metrics["total_obitos"]       = int(d.get("total_obitos", 0))
                    metrics["casos_com_desfecho"] = int(d.get("casos_com_desfecho", 0))
                else:
                    metrics.update({"taxa_mortalidade": 0.0, "total_casos": 0,
                                    "total_obitos": 0, "casos_com_desfecho": 0})
            except Exception as e:
                print(f"   [mortalidade] aviso: {e}")
                metrics["taxa_mortalidade"] = 0.0

            # ── Taxa de Ocupação UTI ──────────────────────────────────────────
            try:
                result = self.sql_tool.execute_query(f"""
                SELECT
                    SUM(total_internados) AS total_internados,
                    SUM(total_uti)        AS total_uti,
                    CASE WHEN SUM(total_internados) > 0
                         THEN (SUM(total_uti) / SUM(total_internados)) * 100
                         ELSE 0 END AS taxa_uti
                FROM {self.catalog}.{self.schema}.gold_metricas_temporais
                WHERE total_internados IS NOT NULL AND total_uti IS NOT NULL
                LIMIT 1
                """)
                if result.get("success") and result.get("data"):
                    d = result["data"][0]
                    metrics["taxa_uti"]         = round(float(d.get("taxa_uti", 0)), 2)
                    metrics["total_internados"] = int(d.get("total_internados", 0))
                    metrics["total_uti"]        = int(d.get("total_uti", 0))
                else:
                    metrics.update({"taxa_uti": 0.0, "total_internados": 0, "total_uti": 0})
            except Exception as e:
                print(f"   [uti] aviso: {e}")
                metrics["taxa_uti"] = 0.0

            # ── Taxa de Vacinação ─────────────────────────────────────────────
            try:
                result = self.sql_tool.execute_query(f"""
                SELECT
                    SUM(total_vacinados)       AS total_vacinados,
                    SUM(casos_com_info_vacina) AS casos_com_info_vacina,
                    CASE WHEN SUM(casos_com_info_vacina) > 0
                         THEN (SUM(total_vacinados) / SUM(casos_com_info_vacina)) * 100
                         ELSE 0 END AS taxa_vacinacao
                FROM {self.catalog}.{self.schema}.gold_metricas_temporais
                WHERE total_vacinados IS NOT NULL AND casos_com_info_vacina IS NOT NULL
                LIMIT 1
                """)
                if result.get("success") and result.get("data"):
                    d = result["data"][0]
                    metrics["taxa_vacinacao"]        = round(float(d.get("taxa_vacinacao", 0)), 2)
                    metrics["total_vacinados"]       = int(d.get("total_vacinados", 0))
                    metrics["casos_com_info_vacina"] = int(d.get("casos_com_info_vacina", 0))
                else:
                    metrics.update({"taxa_vacinacao": 0.0, "total_vacinados": 0,
                                    "casos_com_info_vacina": 0})
            except Exception as e:
                print(f"   [vacinacao] aviso: {e}")
                metrics["taxa_vacinacao"] = 0.0

            # ── Análise anual ─────────────────────────────────────────────────
            try:
                result_anual = self.sql_tool.execute_query(f"""
                SELECT
                    YEAR(TO_DATE(ano_mes, 'yyyy-MM'))                              AS ano,
                    SUM(total_casos)                                                AS casos_ano,
                    SUM(total_obitos)                                               AS obitos_ano,
                    SUM(casos_com_desfecho)                                         AS com_desfecho_ano,
                    ROUND(SUM(total_obitos)
                          / NULLIF(SUM(casos_com_desfecho), 0) * 100, 2)           AS mortalidade_pct,
                    ROUND(SUM(total_uti)
                          / NULLIF(SUM(total_internados), 0) * 100, 2)             AS uti_pct,
                    ROUND(SUM(total_vacinados)
                          / NULLIF(SUM(casos_com_info_vacina), 0) * 100, 2)        AS vacinacao_pct
                FROM {self.catalog}.{self.schema}.gold_metricas_historicas
                WHERE casos_com_desfecho > 0
                GROUP BY 1 ORDER BY 1 DESC
                LIMIT 5
                """)
                if result_anual.get("success") and result_anual.get("data"):
                    metrics["analise_anual"] = result_anual["data"]
                    anos = [str(r.get("ano", "?")) for r in result_anual["data"]]
                    print(f"   [analise_anual] anos disponiveis: {', '.join(anos)}")
            except Exception as ea:
                print(f"   [analise_anual] aviso: {ea}")
                metrics["analise_anual"] = []

            # ── Crescimento mensal ────────────────────────────────────────────
            try:
                result_mensal = self.sql_tool.execute_query(f"""
                WITH serie_mensal AS (
                    SELECT ano_mes, total_casos,
                           LAG(total_casos) OVER (ORDER BY ano_mes) AS casos_mes_anterior
                    FROM {self.catalog}.{self.schema}.gold_metricas_temporais
                    WHERE total_casos IS NOT NULL AND total_casos > 0
                    ORDER BY ano_mes DESC
                    LIMIT 13
                )
                SELECT
                    ano_mes, total_casos, casos_mes_anterior,
                    ROUND(CASE WHEN casos_mes_anterior > 0
                               THEN ((total_casos - casos_mes_anterior)
                                     / casos_mes_anterior) * 100
                               ELSE NULL END, 2) AS crescimento_mensal_pct
                FROM serie_mensal
                WHERE casos_mes_anterior IS NOT NULL
                ORDER BY ano_mes DESC
                LIMIT 12
                """)
                if result_mensal.get("success") and result_mensal.get("data"):
                    metrics["crescimento_mensal"] = result_mensal["data"]
                    print(f"   [crescimento_mensal] {len(result_mensal['data'])} meses calculados")
            except Exception as em:
                print(f"   [crescimento_mensal] aviso: {em}")
                metrics["crescimento_mensal"] = []

            # ── Validação final ───────────────────────────────────────────────
            for key in ("taxa_mortalidade", "taxa_uti", "taxa_vacinacao"):
                v = metrics.get(key)
                if not isinstance(v, (int, float)):
                    metrics[key] = 0.0
                elif v < 0:
                    metrics[key] = 0.0

            tc_val = metrics.get("taxa_crescimento")
            if tc_val is not None and not isinstance(tc_val, (int, float)):
                metrics["taxa_crescimento"] = 0.0

            if self.audit:
                self.audit.log_event(
                    AuditEvent.METRICS_COLLECTED,
                    {
                        "metrics_calculated": 4,
                        "taxa_crescimento":   metrics.get("taxa_crescimento"),
                        "taxa_mortalidade":   metrics.get("taxa_mortalidade"),
                        "taxa_uti":           metrics.get("taxa_uti"),
                        "taxa_vacinacao":     metrics.get("taxa_vacinacao"),
                        "total_casos":        metrics.get("total_casos", 0),
                    },
                    EventStatus.SUCCESS,
                )

            tc = metrics.get("taxa_crescimento")
            print(f"[metrics] taxa_crescimento : {f'{tc:.2f}%' if isinstance(tc, float) else 'ERRO'}")
            print(f"[metrics] taxa_mortalidade : {metrics.get('taxa_mortalidade', 0):.2f}%")
            print(f"[metrics] taxa_uti         : {metrics.get('taxa_uti', 0):.2f}%")
            print(f"[metrics] taxa_vacinacao   : {metrics.get('taxa_vacinacao', 0):.2f}%")
            print(f"[metrics] total_casos      : {metrics.get('total_casos', 0):,}")

            return metrics

        except Exception as e:
            print(f"[metrics] erro critico: {e}")
            if self.audit:
                self.audit.log_event(
                    AuditEvent.METRICS_COLLECTED, {"error": str(e)}, EventStatus.ERROR
                )
            return {
                "taxa_crescimento":    0.0, "taxa_mortalidade": 0.0,
                "taxa_uti":            0.0, "taxa_vacinacao":   0.0,
                "total_casos":         0,   "total_obitos":     0,
                "total_internados":    0,   "total_uti":        0,
                "total_vacinados":     0,   "casos_com_info_vacina": 0,
                "data_referencia":   "N/A", "error": str(e),
            }

    # =========================================================================
    # MAIN EXECUTION
    # =========================================================================

    def run(self, user_query: str) -> Dict:
        """
        Executa o agente orquestrador para uma query em linguagem natural.
        """
        start_time = time.time()

        if self.audit:
            self.audit.log_event(
                AuditEvent.ORCHESTRATOR_START,
                {"query": user_query, "version": VERSION,
                 "llm_provider": "openai" if self.use_openai else "databricks"},
                EventStatus.INFO,
            )

        initial_state: AgentState = {
            "messages":          [HumanMessage(content=user_query)],
            "user_query":        user_query,
            "routing_decision":  None,
            "sql_results":       None,
            "rag_results":       None,
            "news_results":      None,
            "chart_paths":       None,
            "ad_hoc_chart_path": None,
            "geographic_data":   None,
            "mandatory_metrics": None,
            "final_answer":      None,
            "sources":           [],
            "errors":            [],
        }

        try:
            final_state = self.graph.invoke(initial_state)

            if final_state is None:
                print("[run] LangGraph retornou None — usando estado inicial como fallback")
                final_state = {**initial_state, "errors": ["LangGraph retornou estado None"]}

            execution_time = round(time.time() - start_time, 2)
            success        = len(final_state.get("errors", [])) == 0
            rd             = final_state.get("routing_decision")

            if self.audit:
                chart_paths  = final_state.get("chart_paths") or []
                final_answer = final_state.get("final_answer") or ""
                self.audit.log_event(
                    AuditEvent.ORCHESTRATOR_STRATEGY,
                    {
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
                    },
                    EventStatus.INFO,
                )
                evt = AuditEvent.ORCHESTRATOR_SUCCESS if success else AuditEvent.ORCHESTRATOR_FAILED
                self.audit.log_event(
                    evt,
                    {"execution_time": execution_time, "errors": len(final_state.get("errors", []))},
                    EventStatus.SUCCESS if success else EventStatus.ERROR,
                )

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
                "routing": {
                    "intent":     rd.intent.value if rd and rd.intent else None,
                    "strategy":   rd.strategy.value if rd else None,
                    "confidence": rd.confidence if rd else 0,
                    "chart_params": {
                        "metric":     rd.chart_params.metric,
                        "group_by":   rd.chart_params.group_by,
                        "chart_type": rd.chart_params.chart_type,
                        "title":      rd.chart_params.title,
                        "filters":    rd.chart_params.filters,
                        "table":      rd.chart_params.table,
                    } if rd and rd.chart_params else None,
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
                self.audit.log_event(
                    AuditEvent.ORCHESTRATOR_FAILED,
                    {"error": str(e), "execution_time": execution_time},
                    EventStatus.CRITICAL,
                )
            raise OrchestratorError(
                f"Falha critica no orquestrador: {str(e)}",
                details={"execution_time": execution_time},
            )

    def explain_routing(self, user_query: str) -> Dict:
        """
        Explica a decisão de roteamento sem executar o pipeline.
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
                "metric":     decision.chart_params.metric,
                "group_by":   decision.chart_params.group_by,
                "chart_type": decision.chart_params.chart_type,
                "title":      decision.chart_params.title,
                "filters":    decision.chart_params.filters,
                "table":      decision.chart_params.table,
            }
        return result