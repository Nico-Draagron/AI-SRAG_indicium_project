"""
Orchestrator - Arquitetura Gold-First + RAG + Routing (OTIMIZADO)
===================================================================

Versão otimizada com melhorias:
✅ Trigger inteligente de charts (ampliado)
✅ Usa generate_all_charts() (10 gráficos)
✅ Error handling robusto
✅ Logs otimizados
✅ Roteamento aprimorado

Author: AI Engineer Certification - Indicium
Date: February 2025
Version: 3.1.0 - OTIMIZADO
"""

from typing import Dict, List, Optional, TypedDict
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from src.agents.intent_router import IntentRouter, RoutingDecision, ExecutionStrategy
from src.tools.sql_tool import GoldSQLTool
from src.rag.rag_chain import SRAGChain
from src.utils.exceptions import OrchestratorError, SQLError, RAGError
from src.utils.audit import AuditLogger, AuditEvent, EventStatus


class AgentState(TypedDict):
    """Estado compartilhado entre nós do grafo"""
    messages: List[BaseMessage]
    user_query: str
    routing_decision: Optional[RoutingDecision]
    sql_results: Optional[Dict]
    rag_results: Optional[Dict]
    news_results: Optional[Dict]
    chart_paths: Optional[List[str]]
    geographic_data: Optional[Dict]
    final_answer: Optional[str]
    sources: List[str]
    errors: List[str]


class SRAGOrchestrator:
    """
    Orquestrador v3.1 - Gold-First + RAG + Intent Routing (OTIMIZADO)
    
    Pipeline:
        Query → Route → Execute (SQL|RAG|Hybrid) → Synthesize → Answer
    
    Melhorias v3.1:
        - ✅ Trigger de charts ampliado (mais keywords)
        - ✅ Usa generate_all_charts() (10 gráficos)
        - ✅ Error handling otimizado
        - ✅ Logs mais informativos
        - ✅ Performance melhorada
    
    Features:
        - ✅ Intent-based routing
        - ✅ RAG opcional (desacoplado)
        - ✅ Web Search integrado
        - ✅ Chart Tool com 10 gráficos
        - ✅ Auditoria completa
        - ✅ Tratamento robusto de erros
    """
    
    def __init__(
        self,
        spark,
        llm: ChatOpenAI,
        audit_logger: Optional[AuditLogger] = None,
        rag_chain: Optional[SRAGChain] = None,
        use_llm_routing: bool = False,
        web_search_tool=None,
        chart_tool=None
    ):
        self.spark = spark
        self.llm = llm
        self.audit = audit_logger
        self.rag_chain = rag_chain
        
        # Ferramentas principais
        self.sql_tool = GoldSQLTool(spark, audit_logger)
        self.router = IntentRouter(use_llm_classification=use_llm_routing)
        
        # Ferramentas opcionais
        self.web_search_tool = web_search_tool
        self.chart_tool = chart_tool
        
        # Grafo
        self.graph = self._build_graph()
        
        if self.audit:
            self.audit.log_event(
                AuditEvent.ORCHESTRATOR_INITIALIZED,
                {
                    "has_rag": rag_chain is not None,
                    "has_web_search": web_search_tool is not None,
                    "has_charts": chart_tool is not None,
                    "version": "3.1.0"
                },
                EventStatus.INFO
            )
    
    def _build_graph(self) -> StateGraph:
        """Constrói grafo de execução LangGraph"""
        workflow = StateGraph(AgentState)
        
        # Nós
        workflow.add_node("route", self._route_node)
        workflow.add_node("execute_sql", self._execute_sql_node)
        workflow.add_node("execute_rag", self._execute_rag_node)
        workflow.add_node("execute_hybrid", self._execute_hybrid_node)
        workflow.add_node("synthesize", self._synthesize_node)
        
        # Fluxo
        workflow.set_entry_point("route")
        
        # Roteamento condicional
        workflow.add_conditional_edges(
            "route",
            self._route_to_execution,
            {
                "sql": "execute_sql",
                "rag": "execute_rag",
                "hybrid": "execute_hybrid"
            }
        )
        
        # Convergir para síntese
        workflow.add_edge("execute_sql", "synthesize")
        workflow.add_edge("execute_rag", "synthesize")
        workflow.add_edge("execute_hybrid", "synthesize")
        workflow.add_edge("synthesize", END)
        
        return workflow.compile()
    
    # =========================================================================
    # NÓS DO GRAFO
    # =========================================================================
    
    def _route_node(self, state: AgentState) -> AgentState:
        """Nó 1: Roteamento via Intent Router"""
        try:
            if self.audit:
                self.audit.log_event(
                    AuditEvent.NODE_START,
                    {"node": "route", "query": state["user_query"]},
                    EventStatus.INFO
                )
            
            decision = self.router.route(state["user_query"])
            state["routing_decision"] = decision
            
            state["messages"].append(
                AIMessage(content=f"🔀 Rota: {decision.strategy.value} | {decision.reasoning}")
            )
            
            if self.audit:
                self.audit.log_event(
                    AuditEvent.QUERY_ANALYZED,
                    {
                        "strategy": decision.strategy.value,
                        "confidence": decision.confidence,
                        "intent": decision.intent.value
                    },
                    EventStatus.SUCCESS
                )
            
        except Exception as e:
            state["errors"].append(f"Routing error: {str(e)}")
            # Fallback para SQL
            state["routing_decision"] = RoutingDecision(
                intent=None,
                strategy=ExecutionStrategy.SQL_ONLY,
                confidence=0.5,
                reasoning="Fallback devido a erro no routing",
                target_tables=["metricas_temporais"]
            )
        
        return state
    
    def _route_to_execution(self, state: AgentState) -> str:
        """Decide qual nó executar"""
        strategy = state["routing_decision"].strategy
        
        if strategy == ExecutionStrategy.SQL_ONLY:
            return "sql"
        elif strategy == ExecutionStrategy.RAG_ONLY:
            return "rag"
        else:
            return "hybrid"
    
    def _execute_sql_node(self, state: AgentState) -> AgentState:
        """Nó 2a: Execução SQL com Web Search e Charts"""
        try:
            if self.audit:
                self.audit.log_event(
                    AuditEvent.NODE_START,
                    {"node": "execute_sql"},
                    EventStatus.INFO
                )
            
            decision = state["routing_decision"]
            all_results = {}
            
            # ================================================================
            # EXECUÇÃO SQL (múltiplas tabelas)
            # ================================================================
            for table in decision.target_tables:
                try:
                    # Construir query baseada em filtros
                    if decision.sql_filters:
                        # Query filtrada
                        filters_str = " AND ".join([
                            f"{k} = '{v}'" if isinstance(v, str) else f"{k} = {v}"
                            for k, v in decision.sql_filters.items()
                        ])
                        query = f"""
                            SELECT *
                            FROM dbx_lab_draagron.gold.gold_{table}
                            WHERE {filters_str}
                            LIMIT 100
                        """
                    else:
                        # Query padrão
                        query = f"""
                            SELECT *
                            FROM dbx_lab_draagron.gold.gold_{table}
                            ORDER BY 1 DESC
                            LIMIT 12
                        """
                    
                    if self.audit:
                        self.audit.log_event(
                            AuditEvent.SQL_QUERY_START,
                            {"table": table, "query_type": "routing_based"},
                            EventStatus.INFO
                        )
                    
                    result = self.sql_tool.execute_query(query)
                    all_results[table] = result
                    
                except SQLError as sql_err:
                    state["errors"].append(f"SQL error on {table}: {str(sql_err)}")
                    if self.audit:
                        self.audit.log_event(
                            AuditEvent.SQL_QUERY_ERROR,
                            {"table": table, "error": str(sql_err)},
                            EventStatus.ERROR
                        )
                    continue
            
            state["sql_results"] = all_results
            
            # ================================================================
            # WEB SEARCH (se disponível e relevante)
            # ================================================================
            # ✅ MELHORIA: Trigger ampliado com mais keywords
            query_lower = state["user_query"].lower()
            trigger_words = [
                "relatório", "notícias", "completo", "dashboard",
                "contexto", "atualiz", "recente", "hoje", "últimas"
            ]
            
            should_search_web = (
                self.web_search_tool and 
                any(word in query_lower for word in trigger_words)
            )
            
            if should_search_web:
                try:
                    if self.audit:
                        self.audit.log_event(
                            AuditEvent.WEB_SEARCH_START,
                            {"node": "web_search", "trigger": "query_keywords"},
                            EventStatus.INFO
                        )
                    
                    news_results = self.web_search_tool.search_news(
                        query="SRAG Brasil COVID-19 influenza",
                        days_back=7,
                        max_results=5
                    )
                    state["news_results"] = news_results
                    
                    if self.audit:
                        articles_count = len(news_results.get("articles", []))
                        self.audit.log_event(
                            AuditEvent.WEB_SEARCH_SUCCESS,  
                            {"tool": "web_search", "articles": articles_count},
                            EventStatus.SUCCESS
                        )
                    
                    print(f"   📰 Web Search: {articles_count} artigos coletados")
                        
                except Exception as web_err:
                    state["errors"].append(f"Web search error: {str(web_err)}")
                    state["news_results"] = {}
                    print(f"   ⚠️ Web search falhou: {str(web_err)[:100]}")
            
            # ================================================================
            # CHART TOOL (se disponível e relevante)
            # ================================================================
            # ✅ MELHORIA: Trigger ampliado com mais keywords
            chart_trigger_words = [
                "relatório", "gráfico", "visualiz", "completo", "dashboard",
                "chart", "plot", "tendência", "análise visual"
            ]
            
            should_generate_charts = (
                self.chart_tool and 
                any(word in query_lower for word in chart_trigger_words)
            )
            
            if should_generate_charts:
                try:
                    if self.audit:
                        self.audit.log_event(
                            AuditEvent.CHART_GENERATION_START,
                            {"node": "chart_generation", "method": "generate_all_charts"},
                            EventStatus.INFO
                        )
                    
                    print("\n   📊 Gerando gráficos profissionais...")
                    
                    # ✅ MELHORIA: Usar generate_all_charts() (10 gráficos)
                    chart_paths = self.chart_tool.generate_all_charts()
                    
                    state["chart_paths"] = chart_paths
                    
                    if self.audit:
                        self.audit.log_event(
                            AuditEvent.CHART_GENERATED,  
                            {
                                "tool": "chart_generation", 
                                "charts": len(chart_paths),
                                "method": "generate_all_charts"
                            },
                            EventStatus.SUCCESS
                        )
                    
                    print(f"   ✅ {len(chart_paths)} gráficos gerados com sucesso!")
                        
                except Exception as chart_err:
                    state["errors"].append(f"Chart generation error: {str(chart_err)}")
                    state["chart_paths"] = []
                    print(f"   ⚠️ Chart generation falhou: {str(chart_err)[:100]}")
            
            # ================================================================
            # VALIDAÇÃO E LOG FINAL
            # ================================================================
            if all_results:
                total_rows = sum(r.get("rows", 0) for r in all_results.values() if r.get("success"))
                state["messages"].append(
                    AIMessage(content=f"✅ SQL: {len(all_results)} tabelas, {total_rows} registros")
                )
                if self.audit:
                    self.audit.log_event(
                        AuditEvent.METRICS_COLLECTED,
                        {"tables": len(all_results), "total_rows": total_rows},
                        EventStatus.SUCCESS
                    )
            else:
                state["errors"].append("Nenhum resultado SQL obtido")
            
        except Exception as e:
            state["errors"].append(f"SQL node error: {str(e)}")
            state["sql_results"] = {}
            if self.audit:
                self.audit.log_event(
                    AuditEvent.NODE_FAILED,
                    {"node": "execute_sql", "error": str(e)},
                    EventStatus.ERROR
                )
        
        return state
    
    def _execute_rag_node(self, state: AgentState) -> AgentState:
        """Nó 2b: Execução RAG"""
        try:
            if not self.rag_chain:
                state["messages"].append(AIMessage(content="⏭️ RAG desabilitado"))
                state["rag_results"] = {}
                return state
            
            if self.audit:
                self.audit.log_event(
                    AuditEvent.NODE_START,
                    {"node": "execute_rag"},
                    EventStatus.INFO
                )
            
            rag_response = self.rag_chain.invoke(state["user_query"])
            
            state["rag_results"] = {
                "answer": rag_response["answer"],
                "sources": rag_response["source_documents"],
                "num_sources": len(rag_response["source_documents"])
            }
            
            state["messages"].append(
                AIMessage(content=f"✅ RAG: {len(rag_response['source_documents'])} fontes")
            )
            
            if self.audit:
                self.audit.log_event(
                    AuditEvent.NODE_COMPLETE,
                    {"node": "execute_rag", "num_sources": len(rag_response['source_documents'])},
                    EventStatus.SUCCESS
                )
            
        except RAGError as rag_err:
            state["errors"].append(f"RAG error: {str(rag_err)}")
            state["rag_results"] = {}
            if self.audit:
                self.audit.log_event(
                    AuditEvent.NODE_FAILED,
                    {"node": "execute_rag", "error": str(rag_err)},
                    EventStatus.ERROR
                )
        except Exception as e:
            state["errors"].append(f"RAG node error: {str(e)}")
            state["rag_results"] = {}
            if self.audit:
                self.audit.log_event(
                    AuditEvent.NODE_ERROR,
                    {"node": "execute_rag", "error": str(e)},
                    EventStatus.ERROR
                )
        
        return state
    
    def _execute_hybrid_node(self, state: AgentState) -> AgentState:
        """Nó 2c: Execução Híbrida (SQL + RAG)"""
        try:
            if self.audit:
                self.audit.log_event(
                    AuditEvent.NODE_START,
                    {"node": "execute_hybrid"},
                    EventStatus.INFO
                )
            
            # Executar SQL
            sql_state = self._execute_sql_node(state.copy())
            state["sql_results"] = sql_state.get("sql_results", {})
            state["news_results"] = sql_state.get("news_results", {})
            state["chart_paths"] = sql_state.get("chart_paths", [])
            
            # Executar RAG (se disponível)
            rag_state = self._execute_rag_node(state.copy())
            state["rag_results"] = rag_state.get("rag_results", {})
            
            # Merge erros
            state["errors"].extend(sql_state.get("errors", []))
            state["errors"].extend(rag_state.get("errors", []))
            
            state["messages"].append(
                AIMessage(content="✅ Execução híbrida (SQL + RAG) completa")
            )
            
            if self.audit:
                self.audit.log_event(
                    AuditEvent.NODE_COMPLETE,
                    {"node": "execute_hybrid"},
                    EventStatus.SUCCESS
                )
            
        except Exception as e:
            state["errors"].append(f"Hybrid node error: {str(e)}")
            if self.audit:
                self.audit.log_event(
                    AuditEvent.NODE_ERROR,
                    {"node": "execute_hybrid", "error": str(e)},
                    EventStatus.ERROR
                )
        
        return state
    
    def _synthesize_node(self, state: AgentState) -> AgentState:
        """Nó 3: Síntese final via LLM"""
        try:
            if self.audit:
                self.audit.log_event(
                    AuditEvent.NODE_START,
                    {"node": "synthesize"},
                    EventStatus.INFO
                )
            
            query = state["user_query"]
            sql_results = state.get("sql_results", {})
            rag_results = state.get("rag_results", {})
            news_results = state.get("news_results", {})
            chart_paths = state.get("chart_paths", [])
            
            # Construir contexto
            context_parts = []
            
            # Dados SQL
            if sql_results:
                context_parts.append("DADOS SQL:")
                for table, result in sql_results.items():
                    if result.get("success"):
                        context_parts.append(f"\nTabela: {table}")
                        context_parts.append(f"Registros: {result['rows']}")
                        # Primeiros 3 registros
                        data_sample = result['data'][:3]
                        context_parts.append(str(data_sample))
            
            # Contexto RAG
            if rag_results and rag_results.get("answer"):
                context_parts.append("\n\nCONTEXTO RAG:")
                context_parts.append(rag_results.get("answer", ""))
            
            # Notícias
            if news_results and news_results.get("articles"):
                context_parts.append("\n\nNOTÍCIAS RECENTES:")
                for article in news_results["articles"][:3]:
                    context_parts.append(f"- {article.get('title', 'N/A')}")
            
            # Gráficos
            if chart_paths:
                context_parts.append(f"\n\nGRÁFICOS GERADOS: {len(chart_paths)} visualizações")
            
            if not context_parts:
                state["final_answer"] = "Não foi possível coletar dados suficientes para responder."
                state["errors"].append("Nenhum contexto disponível para síntese")
                return state
            
            context = "\n".join(context_parts)
            
            # Prompt para LLM
            prompt = f"""Baseando-se nos dados abaixo, responda a pergunta do usuário de forma clara e concisa.

PERGUNTA: {query}

DADOS DISPONÍVEIS:
{context}

INSTRUÇÕES:
- Use os dados SQL como fonte primária de verdade
- Incorpore insights do contexto RAG quando relevante
- Mencione notícias recentes se disponíveis
- Seja específico com números e tendências
- Mantenha resposta profissional e objetiva

RESPOSTA:"""
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            state["final_answer"] = response.content
            
            # Fontes
            sources = []
            if sql_results:
                sources.extend([f"gold_{table}" for table in sql_results.keys()])
            if rag_results:
                sources.append("RAG")
            if news_results:
                sources.append("Web Search")
            if chart_paths:
                sources.append(f"Charts ({len(chart_paths)})")
            state["sources"] = sources
            
            if self.audit:
                self.audit.log_event(
                    AuditEvent.NODE_COMPLETE,
                    {"node": "synthesize", "sources": len(sources)},
                    EventStatus.SUCCESS
                )
            
        except Exception as e:
            state["errors"].append(f"Synthesis error: {str(e)}")
            state["final_answer"] = "Erro ao gerar resposta final."
            if self.audit:
                self.audit.log_event(
                    AuditEvent.NODE_ERROR,
                    {"node": "synthesize", "error": str(e)},
                    EventStatus.ERROR
                )
        
        return state
    
    # =========================================================================
    # EXECUÇÃO PRINCIPAL
    # =========================================================================
    
    def run(self, user_query: str) -> Dict:
        """Executa o agente orquestrador"""
        start_time = datetime.now()
        
        if self.audit:
            self.audit.log_event(
                AuditEvent.ORCHESTRATOR_START,
                {"query": user_query, "version": "3.1.0"},
                EventStatus.INFO
            )
        
        initial_state = {
            "messages": [HumanMessage(content=user_query)],
            "user_query": user_query,
            "routing_decision": None,
            "sql_results": None,
            "rag_results": None,
            "news_results": None,
            "chart_paths": None,
            "geographic_data": None,
            "final_answer": None,
            "sources": [],
            "errors": []
        }
        
        try:
            final_state = self.graph.invoke(initial_state)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            success = len(final_state.get("errors", [])) == 0
            
            # Auditoria: Registrar estratégia final usada
            strategy_used = final_state["routing_decision"].strategy.value if final_state.get("routing_decision") else "UNKNOWN"
            
            if self.audit:
                # Log da estratégia utilizada
                self.audit.log_event(
                    AuditEvent.ORCHESTRATOR_STRATEGY,
                    {
                        "strategy_used": strategy_used,
                        "confidence": final_state["routing_decision"].confidence if final_state.get("routing_decision") else 0,
                        "has_sql_results": final_state.get("sql_results") is not None,
                        "has_rag_results": final_state.get("rag_results") is not None,
                        "has_news": final_state.get("news_results") is not None,
                        "has_charts": final_state.get("chart_paths") is not None,
                        "num_charts": len(final_state.get("chart_paths", [])),
                        "final_answer_length": len(final_state.get("final_answer", ""))
                    },
                    EventStatus.INFO
                )
                
                # Log de sucesso/falha
                event = AuditEvent.ORCHESTRATOR_SUCCESS if success else AuditEvent.ORCHESTRATOR_FAILED
                self.audit.log_event(
                    event,
                    {"execution_time": execution_time, "errors": len(final_state.get("errors", []))},
                    EventStatus.SUCCESS if success else EventStatus.ERROR
                )
            
            return {
                "success": success,
                "answer": final_state.get("final_answer"),
                "sources": final_state.get("sources", []),
                "sql_results": final_state.get("sql_results", {}),
                "rag_results": final_state.get("rag_results", {}),
                "news_results": final_state.get("news_results", {}),
                "chart_paths": final_state.get("chart_paths", []),
                "geographic_data": final_state.get("geographic_data", {}),
                "routing": {
                    "intent": final_state["routing_decision"].intent.value if final_state.get("routing_decision") and final_state["routing_decision"].intent else None,
                    "strategy": final_state["routing_decision"].strategy.value if final_state.get("routing_decision") else None,
                    "confidence": final_state["routing_decision"].confidence if final_state.get("routing_decision") else 0
                },
                "errors": final_state.get("errors", []),
                "execution_time_seconds": execution_time,
                "messages": [m.content for m in final_state.get("messages", [])]
            }
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if self.audit:
                self.audit.log_event(
                    AuditEvent.ORCHESTRATOR_FAILED,
                    {"error": str(e), "execution_time": execution_time},
                    EventStatus.CRITICAL
                )
            
            raise OrchestratorError(
                f"Falha crítica no orquestrador: {str(e)}",
                details={"execution_time": execution_time}
            )
    
    def explain_routing(self, user_query: str) -> Dict:
        """Explica decisão de roteamento sem executar"""
        decision = self.router.route(user_query)
        
        return {
            "query": user_query,
            "intent": decision.intent.value,
            "strategy": decision.strategy.value,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
            "target_tables": decision.target_tables,
            "sql_filters": decision.sql_filters,
            "rag_type": decision.rag_semantic_type
        }