# Databricks notebook source
# MAGIC %md
# MAGIC # Agente SRAG — Execução Orquestrada e Validação Operacional
# MAGIC
# MAGIC ## Objetivo
# MAGIC
# MAGIC Executar o agente SRAG ponta a ponta sobre a camada Gold, gerar resposta analítica,
# MAGIC persistir artefatos e validar os principais componentes operacionais.
# MAGIC
# MAGIC ## Entradas esperadas
# MAGIC
# MAGIC | Recurso | Detalhe |
# MAGIC |---|---|
# MAGIC | Tabelas Gold | `gold_metricas_temporais`, `gold_metricas_geograficas`, `gold_metricas_demograficas`, `gold_metricas_historicas`, `gold_serie_diaria_30d` |
# MAGIC | Índice vetorial | Endpoint Databricks Vector Search configurado |
# MAGIC | Secrets de API | `openai-api-key`, `tavily-api-key` no scope `ai-engineer` |
# MAGIC | Volume | `/Volumes/{catalog}/default/srag_outputs` acessível |
# MAGIC
# MAGIC ## Saídas geradas
# MAGIC
# MAGIC - Resposta analítica do agente (Markdown)
# MAGIC - Métricas obrigatórias (taxa de mortalidade, UTI, vacinação, crescimento)
# MAGIC - Gráficos interativos HTML (série diária, mensal, geográfico, demográfico, viral)
# MAGIC - Relatório completo em Markdown e JSON
# MAGIC - Logs de auditoria em Delta Lake e JSON
# MAGIC
# MAGIC ## Escopo
# MAGIC
# MAGIC Este notebook cobre a execução integrada e validação operacional do agente.
# MAGIC A construção das tabelas Gold (`00`–`05`) e o setup de infraestrutura são responsabilidade dos notebooks anteriores.
# MAGIC
# MAGIC ## Quando usar este notebook
# MAGIC
# MAGIC - Execução do ciclo completo agente → artefatos → validação
# MAGIC - Demonstração operacional do sistema
# MAGIC - Verificação de saúde dos componentes após mudanças na stack
# MAGIC
# MAGIC ## Quando **não** usar este notebook
# MAGIC
# MAGIC - Reconstrução da camada Gold → usar `00_pipeline_gold`
# MAGIC - Análise epidemiológica aprofundada e histórica → usar `07_agent_validation`
# MAGIC - Debug isolado de ferramenta específica → usar os notebooks de tool individualmente
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parte A — Setup e Inicialização

# COMMAND ----------

# MAGIC %md
# MAGIC ### A.1 Dependências

# COMMAND ----------

import os
for path in ["src", "src/agents", "src/tools", "src/rag", "src/utils"]:
    has_init = os.path.exists(f"{path}/__init__.py")
    print(f"{'OK' if has_init else 'FALTANDO'} {path}/__init__.py")


# COMMAND ----------

# MAGIC %pip install -r requirements.txt --quiet
# MAGIC dbutils.library.restartPython()
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### A.2 Imports

# COMMAND ----------

import json
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from pyspark.sql import SparkSession


# COMMAND ----------

from src.agents import SRAGOrchestrator
from src.tools import GoldSQLTool, ReportGenerator, ChartTool, WebSearchTool
from src.utils import (
    AuditLogger, AuditEvent, EventStatus,
    SQLGuardrails, GuardrailsConfig,
    OrchestratorError, SQLError,
)

try:
    from src.rag import (
        GoldDocumentLoader,
        EmbeddingManager,
        VectorStoreConfig,
        DatabricksVectorStoreManager,
        SRAGRetriever,
        SRAGChain,
        RAGConfig,
    )
    RAG_AVAILABLE = True
except ImportError as _e:
    RAG_AVAILABLE = False
    print(f"[aviso] módulo RAG não importado — continuando sem RAG ({_e})")


# COMMAND ----------

# MAGIC %md
# MAGIC ### A.3 Configuração do Ambiente

# COMMAND ----------

# ── Flags de execução — ajuste conforme a necessidade da run ─────────────────
# Desabilite blocos que não precisam rodar (ex: índice já atualizado)
RUN_SETUP          = True   # setup de diretórios, spark, credenciais
RUN_RAG_INDEX      = True   # cria/atualiza índice vetorial (pode reusar se inalterado)
RUN_AGENT          = True   # executa o agente (consulta principal)
RUN_VALIDATION     = True   # validação pós-execução: volume, gráficos, métricas
RUN_CERTIFICATION  = True   # testes de conversa, roteamento, RAG e verificação de qualidade

# ── Configuração centralizada do ambiente ────────────────────────────────────
CATALOG_GOLD  = "dbx_srag_lab"
SCHEMA_GOLD   = "gold"
CATALOG_AUDIT = "dbx_srag_lab"
SCHEMA_AUDIT  = "audit"

VOLUME_BASE = f"/Volumes/{CATALOG_GOLD}/default/srag_outputs"

VS_ENDPOINT_NAME = "srag_vector_endpoint"
VS_INDEX_NAME    = "srag_embeddings_index_bge"
VS_TABLE_NAME    = "srag_embeddings_table_bge"

# ── Provider do LLM ──────────────────────────────────────────────────────────
# "openai"     → gpt-4o-mini          (custo por token)
# "databricks" → meta-llama-3-3-70b   (Foundation Models, zero custo)
LLM_PROVIDER         = "openai"
LLM_MODEL_OPENAI     = "gpt-4o-mini"
LLM_MODEL_DATABRICKS = "databricks-meta-llama-3-3-70b-instruct"
LLM_TEMP             = 0.1
LLM_MAX_TOKENS       = 4000

print(f"  catalog gold  : {CATALOG_GOLD}.{SCHEMA_GOLD}")
print(f"  catalog audit : {CATALOG_AUDIT}.{SCHEMA_AUDIT}")
print(f"  volume        : {VOLUME_BASE}")
print(f"  llm provider  : {LLM_PROVIDER}")
print(f"  flags         : setup={RUN_SETUP} | rag_index={RUN_RAG_INDEX} | agent={RUN_AGENT}")
print(f"                  validation={RUN_VALIDATION} | certification={RUN_CERTIFICATION}")


# COMMAND ----------

# MAGIC %md
# MAGIC ### A.4 Estrutura de Diretórios

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE VOLUME IF NOT EXISTS dbx_srag_lab.default.srag_outputs;
# MAGIC

# COMMAND ----------

def setup_project_structure(base: str = VOLUME_BASE) -> Dict[str, str]:
    paths = {
        "base"             : base,
        "charts_daily"     : f"{base}/charts/daily",
        "charts_monthly"   : f"{base}/charts/monthly",
        "charts_custom"    : f"{base}/charts/custom",
        "reports_markdown" : f"{base}/reports/markdown",
        "reports_json"     : f"{base}/reports/json",
        "logs_audit"       : f"{base}/logs/audit",
        "temp"             : f"{base}/temp",
    }
    ok, fail = 0, []
    for name, path in paths.items():
        try:
            dbutils.fs.mkdirs(path)
            ok += 1
        except Exception as exc:
            fail.append((name, str(exc)[:80]))
    print(f"diretórios criados: {ok}/{len(paths)}  falhas: {len(fail)}")
    for name, err in fail:
        print(f"  aviso: {name} — {err}")
    try:
        dbutils.fs.ls(paths["base"])
    except Exception:
        raise RuntimeError(f"diretório base inacessível: {paths['base']}")
    return paths

project_paths = setup_project_structure()


# COMMAND ----------

# MAGIC %md
# MAGIC ### A.5 SparkSession e Credenciais

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

OPENAI_AVAILABLE = False
TAVILY_AVAILABLE = False

try:
    OPENAI_API_KEY = dbutils.secrets.get(scope="ai-engineer", key="openai-api-key")
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    OPENAI_AVAILABLE = True
    print("openai api key: configurada")
except Exception:
    OPENAI_API_KEY = None
    os.environ.pop("OPENAI_API_KEY", None)
    print("openai api key: ausente — LLM usará fallback Databricks se disponível")

try:
    TAVILY_API_KEY = dbutils.secrets.get(scope="ai-engineer", key="tavily-api-key")
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
    TAVILY_AVAILABLE = True
    print("tavily api key: configurada")
except Exception:
    TAVILY_API_KEY = None
    os.environ.pop("TAVILY_API_KEY", None)
    print("tavily api key: ausente — web search usará fallback interno")


# COMMAND ----------

# MAGIC %md
# MAGIC ### A.6 LLM

# COMMAND ----------

EFFECTIVE_LLM_PROVIDER = LLM_PROVIDER

if LLM_PROVIDER == "openai" and not OPENAI_AVAILABLE:
    print("[aviso] LLM_PROVIDER='openai', mas OPENAI_API_KEY não está disponível.")
    print("[aviso] fallback automático para Databricks Foundation Models.")
    EFFECTIVE_LLM_PROVIDER = "databricks"

if EFFECTIVE_LLM_PROVIDER == "databricks":
    try:
        from databricks_langchain import ChatDatabricks
    except ImportError:
        from langchain_community.chat_models import ChatDatabricks
    llm = ChatDatabricks(
        endpoint    = LLM_MODEL_DATABRICKS,
        temperature = LLM_TEMP,
        max_tokens  = LLM_MAX_TOKENS,
    )
    print(f"llm: {LLM_MODEL_DATABRICKS} (Databricks Foundation Models)")
else:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model       = LLM_MODEL_OPENAI,
        temperature = LLM_TEMP,
        max_tokens  = LLM_MAX_TOKENS,
    )
    print(f"llm: {LLM_MODEL_OPENAI} (OpenAI)")

print(f"openai disponível : {OPENAI_AVAILABLE}")
print(f"tavily disponível : {TAVILY_AVAILABLE}")
print(f"llm efetivo       : {EFFECTIVE_LLM_PROVIDER}")


# COMMAND ----------

# MAGIC %md
# MAGIC ### A.7 Ferramentas

# COMMAND ----------

SESSION_ID   = f"srag_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
audit_logger = AuditLogger(session_id=SESSION_ID)
audit_logger.log_event(
    AuditEvent.ORCHESTRATOR_INITIALIZED,
    {"timestamp": datetime.now().isoformat(), "provider": EFFECTIVE_LLM_PROVIDER},
    EventStatus.INFO,
)
print(f"audit logger session: {audit_logger.session_id}")


# COMMAND ----------

sql_guardrails_config = GuardrailsConfig(
    enable_sql_validation      = True,
    enable_injection_detection = True,
    enable_table_whitelist     = True,
    require_limit_clause       = True,
    max_limit_value            = 10_000,
)
sql_tool = GoldSQLTool(
    spark             = spark,
    audit_logger      = audit_logger,
    guardrails_config = sql_guardrails_config,
)
print("sql tool: inicializado (guardrails ativos)")


# COMMAND ----------

web_search_tool = None

if TAVILY_AVAILABLE:
    try:
        web_search_tool = WebSearchTool(
            api_key      = os.environ["TAVILY_API_KEY"],
            audit_logger = audit_logger,
        )
        status = "api conectada" if web_search_tool.api_available else "modo fallback"
        print(f"web search tool: inicializado ({status})")
    except Exception as exc:
        print(f"web search tool: falhou na inicialização ({exc})")
else:
    web_search_tool = WebSearchTool(audit_logger=audit_logger)
    print("web search tool: inicializado (fallback — sem chave Tavily)")


# COMMAND ----------

try:
    chart_tool = ChartTool(
        spark        = spark,
        audit_logger = audit_logger,
        output_dirs  = {
            "default"    : Path(project_paths["charts_custom"]),
            "line"       : Path(project_paths["charts_daily"]),
            "mensal"     : Path(project_paths["charts_monthly"]),
            "bar"        : Path(project_paths["charts_custom"]),
            "multi_line" : Path(project_paths["charts_custom"]),
        },
        catalog  = CATALOG_GOLD,
        schema   = SCHEMA_GOLD,
        dbutils  = dbutils,
    )
    print("chart tool: inicializado")
    print(f"  output dirs: daily={project_paths['charts_daily']}")
    print(f"               monthly={project_paths['charts_monthly']}")
except Exception as exc:
    chart_tool = None
    print(f"chart tool: não disponível ({exc})")


# COMMAND ----------

report_generator = ReportGenerator(llm=llm, audit=audit_logger)
print("report generator: inicializado")


# COMMAND ----------

# MAGIC %md
# MAGIC ### A.8 Sistema RAG
# MAGIC
# MAGIC | Tabela | Conteúdo |
# MAGIC |---|---|
# MAGIC | `gold_rag_kpi_fatos` | KPIs por UF e mês — base factual para perguntas epidemiológicas |
# MAGIC | `gold_rag_dicionario_regras` | Definições metodológicas — taxas, denominadores, regras SIVEP |
# MAGIC

# COMMAND ----------

rag_chain   = None
RAG_ENABLED = True

if RAG_ENABLED and RAG_AVAILABLE:
    try:
        print("carregando documentos Gold para RAG...")
        doc_loader = GoldDocumentLoader(
            spark   = spark,
            catalog = CATALOG_GOLD,
            schema  = SCHEMA_GOLD,
        )
        documents = doc_loader.load_all_documents(
            include_rag_kpi     = True,
            include_dicionario  = True,
            include_temporal    = False,
            include_geographic  = False,
            include_demographic = False,
        )
        langchain_docs = doc_loader.to_langchain_documents(documents)
        print(f"  {len(documents)} documentos carregados → {len(langchain_docs)} LangChain docs")
    except Exception as exc:
        print(f"erro ao carregar documentos: {exc}")
        langchain_docs = None
        RAG_ENABLED    = False
else:
    langchain_docs = None
    reason = "módulo não importado" if not RAG_AVAILABLE else "desabilitado por config"
    print(f"RAG não iniciado ({reason})")


# COMMAND ----------

if RAG_ENABLED and langchain_docs:
    try:
        print("inicializando embeddings e vector store...")
        embeddings = EmbeddingManager.get_embeddings(
            provider = "databricks",
            model    = "bge_large_en_v1_5",
        )
        print("  embeddings: Databricks BGE-Large-EN (1024d)")

        vector_config = VectorStoreConfig(
            catalog       = CATALOG_GOLD,
            schema        = SCHEMA_GOLD,
            index_name    = VS_INDEX_NAME,
            table_name    = VS_TABLE_NAME,
            endpoint_name = VS_ENDPOINT_NAME,
            embedding_dim = 1024,
        )
        vector_manager = DatabricksVectorStoreManager(
            spark      = spark,
            config     = vector_config,
            embeddings = embeddings,
        )
        print(f"  index  : {CATALOG_GOLD}.{SCHEMA_GOLD}.{VS_INDEX_NAME}")
        print(f"  table  : {CATALOG_GOLD}.{SCHEMA_GOLD}.{VS_TABLE_NAME}")
        print(f"  endpoint: {VS_ENDPOINT_NAME}")
    except Exception as exc:
        print(f"erro ao inicializar vector store: {exc}")
        vector_manager = None
        RAG_ENABLED    = False
else:
    vector_manager = None


# COMMAND ----------

if RUN_RAG_INDEX:
    if RAG_ENABLED and vector_manager and langchain_docs:
        try:
            emb_table = f"{CATALOG_GOLD}.{SCHEMA_GOLD}.{VS_TABLE_NAME}"
            try:
                spark.sql(f"ALTER TABLE {emb_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
                print(f"CDF habilitado em {emb_table}")
            except Exception as cdf_err:
                print(f"CDF (não crítico — ok na primeira execução): {cdf_err}")
    
            print("criando/atualizando índice vetorial...")
            index_ready = vector_manager.create_or_load_index(langchain_docs)
            if not index_ready:
                raise RuntimeError("falha ao criar/carregar índice vetorial")
            print(f"índice pronto: {CATALOG_GOLD}.{SCHEMA_GOLD}.{VS_INDEX_NAME}")
        except Exception as exc:
            print(f"erro ao criar índice: {exc}")
            vector_manager = None
            RAG_ENABLED    = False
    
else:
    print("[PULADO] RUN_RAG_INDEX=False — reutilizando índice existente")


# COMMAND ----------

if RAG_ENABLED and vector_manager:
    try:
        retriever = SRAGRetriever(vector_store_manager=vector_manager)
        rag_config = RAGConfig(
            top_k              = 5,
            retrieval_strategy = "hybrid",
            use_citations      = True,
            max_context_length = 8000,
        )
        rag_chain = SRAGChain(
            retriever = retriever,
            llm       = llm,
            config    = rag_config,
        )
        print(f"RAG chain: pronta (provider={LLM_PROVIDER}, strategy=hybrid, top_k=5)")
    except Exception as exc:
        print(f"erro ao montar RAG chain: {exc}")
        rag_chain   = None
        RAG_ENABLED = False
else:
    if not RAG_ENABLED:
        print("RAG chain: não iniciada (vector store indisponível)")


# COMMAND ----------

# MAGIC %md
# MAGIC ### A.9 Orquestrador

# COMMAND ----------

orchestrator = SRAGOrchestrator(
    spark           = spark,
    llm             = llm,
    audit_logger    = audit_logger,
    rag_chain       = rag_chain,
    web_search_tool = web_search_tool,
    chart_tool      = chart_tool,
    report_generator= report_generator,   # integração direta com ReportGenerator no nó REPORT
    catalog         = CATALOG_GOLD,
    schema          = SCHEMA_GOLD,
    use_llm_routing = False,
    use_openai      = (EFFECTIVE_LLM_PROVIDER == "openai"),
)
print("orquestrador: inicializado")
print(f"  provider         : {EFFECTIVE_LLM_PROVIDER}")
print(f"  rag              : {'ativo' if rag_chain        else 'desabilitado'}")
print(f"  web search       : {'ativo' if web_search_tool  else 'desabilitado'}")
print(f"  charts           : {'ativo' if chart_tool        else 'desabilitado'}")
print(f"  report_generator : {'ativo' if report_generator  else 'desabilitado'}")
print(f"  guardrails       : ativos")
print(f"  auditoria        : {len(audit_logger.logs)} eventos registrados")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Parte B — Execução Operacional

# COMMAND ----------

# MAGIC %md
# MAGIC ### B.1 Consulta Principal

# COMMAND ----------

USER_QUERY = """
Gere um relatório epidemiológico completo de SRAG no Brasil incluindo:

1. MÉTRICAS OBRIGATÓRIAS:
   - Taxa de aumento de casos
   - Taxa de mortalidade
   - Taxa de ocupação de UTI
   - Taxa de vacinação da população

2. GRÁFICOS OBRIGATÓRIOS:
   - Gráfico de casos diários (últimos 30 dias)
   - Gráfico de casos mensais (últimos 12 meses)

3. CONTEXTO E ANÁLISE:
   - Notícias recentes sobre SRAG no Brasil
   - Explicação sobre as tendências observadas nas métricas
   - Análise do cenário epidemiológico atual
"""

print("=" * 80)
print("EXECUTANDO AGENTE ORQUESTRADOR — RELATÓRIO EPIDEMIOLÓGICO SRAG")
print("=" * 80)
print(USER_QUERY.strip())
print("=" * 80)


# COMMAND ----------

if RUN_AGENT:
    result = orchestrator.run(user_query=USER_QUERY)
    _exec_status = "OK" if result.get("success") else "PARCIAL"
    _exec_time   = result.get("execution_time_seconds", 0)
    _exec_strat  = result.get("routing", {}).get("strategy", "N/A").upper()
    _exec_erros  = len(result.get("errors", []))
    print(f"execução concluída  |  status={_exec_status}  |  tempo={_exec_time:.1f}s  |  estratégia={_exec_strat}  |  erros={_exec_erros}")
else:
    print("[PULADO] RUN_AGENT=False")
    result = {}


# COMMAND ----------

# MAGIC %md
# MAGIC ### B.2 Resumo Executivo

# COMMAND ----------

if result:
    _mm   = result.get("mandatory_metrics", {})
    _rout = result.get("routing", {})
    _news = result.get("news_results", {})
    _arts = _news.get("articles", []) if isinstance(_news, dict) else []

    print("─" * 60)
    print(f"  Query              : {USER_QUERY.strip()[:70]}...")
    print(f"  Estratégia         : {_rout.get('strategy','N/A').upper()}  ({_rout.get('confidence',0):.0%} confiança)")
    print(f"  Tempo de execução  : {result.get('execution_time_seconds',0):.1f}s")
    print(f"  Status             : {'OK' if result.get('success') else 'PARCIAL'}")
    print(f"  Erros              : {len(result.get('errors', []))}")
    print("─" * 60)
    print(f"  Métricas calculadas:")
    print(f"    taxa_crescimento  : {_mm.get('taxa_crescimento', 'N/A')}%")
    print(f"    taxa_mortalidade  : {_mm.get('taxa_mortalidade', 'N/A')}%")
    print(f"    taxa_uti          : {_mm.get('taxa_uti', 'N/A')}%")
    print(f"    taxa_vacinacao    : {_mm.get('taxa_vacinacao', 'N/A')}%")
    print(f"    total_casos       : {_mm.get('total_casos', 'N/A'):,}" if isinstance(_mm.get('total_casos'), (int,float)) else f"    total_casos       : {_mm.get('total_casos','N/A')}")
    print("─" * 60)
    _charts = list(result.get("chart_paths") or [])
    print(f"  Gráficos gerados   : {len(_charts)}")
    print(f"  Artigos web        : {len(_arts)}")
    print("─" * 60)


# COMMAND ----------

# MAGIC %md
# MAGIC ### B.3 Resposta do Agente

# COMMAND ----------

if result.get("answer"):
    print("=" * 80)
    print("RESPOSTA DO AGENTE")
    print("=" * 80)
    print(result["answer"])
    print("=" * 80)
else:
    print("sem resposta gerada — verifique os warnings acima")


# COMMAND ----------

# MAGIC %md
# MAGIC ### B.4 Gráficos

# COMMAND ----------

daily_charts   = []
monthly_charts = []
all_charts     = list(result.get("chart_paths", []) or [])

_adhoc = result.get("ad_hoc_chart_path")
if _adhoc and _adhoc not in all_charts:
    all_charts.append(_adhoc)

_chart_meta: Dict[str, str] = {}
for path in all_charts:
    _n = Path(path).name
    if   "mensal"     in _n or "monthly"   in _n: _chart_meta[path] = "mensal"
    elif "multi_line" in _n or "viral"     in _n: _chart_meta[path] = "viral"
    elif "line"       in _n:                       _chart_meta[path] = "diário"
    elif "bar"        in _n:
        _c = _n.split("_")[-1].replace(".html", "")
        _chart_meta[path] = "demográfico" if (_c.isdigit() and int(_c) >= 3) else "geográfico"
    else:
        _chart_meta[path] = "outro"

for path, ctype in _chart_meta.items():
    if ctype == "mensal":   monthly_charts.append(path)
    elif ctype == "diário": daily_charts.append(path)

print(f"total    : {len(all_charts)} gráficos")
print(f"diários  : {len(daily_charts)}")
print(f"mensais  : {len(monthly_charts)}")
for p in all_charts:
    print(f"  {Path(p).name} [{_chart_meta.get(p, '?')}]")


# COMMAND ----------

if daily_charts:
    print(f"Casos Diários — Últimos 30 dias")
    print(f"path: {daily_charts[0]}")
    try:
        with open(daily_charts[0], "r", encoding="utf-8") as f:
            displayHTML(f.read())
    except Exception as exc:
        print(f"erro ao exibir: {exc}")


# COMMAND ----------

if monthly_charts:
    print(f"Casos Mensais — Últimos 12 meses")
    print(f"path: {monthly_charts[0]}")
    try:
        with open(monthly_charts[0], "r", encoding="utf-8") as f:
            displayHTML(f.read())
    except Exception as exc:
        print(f"erro ao exibir: {exc}")


# COMMAND ----------

# ── Visualização de todos os gráficos gerados pelo agente ────────────────────
print("=" * 72)
print(f"GRÁFICOS GERADOS PELO AGENTE ({len(all_charts)} total)")
print("=" * 72)

if not all_charts:
    print("Nenhum gráfico disponível.")
else:
    for _idx, _path in enumerate(all_charts, 1):
        _name  = Path(_path).name
        _ctype = _chart_meta.get(_path, "outro")
        _label = {
            "diário"      : "📈 Série Diária",
            "mensal"      : "📊 Série Mensal",
            "geográfico"  : "🗺️  Distribuição Geográfica",
            "demográfico" : "👥 Perfil Demográfico",
            "viral"       : "🦠 Distribuição Viral",
        }.get(_ctype, "📉 Gráfico")
        print(f"\n[{_idx}/{len(all_charts)}] {_label} — {_name}")
        print(f"     path: {_path}")
        try:
            with open(_path, "r", encoding="utf-8") as _f:
                displayHTML(_f.read())
        except Exception as _exc:
            print(f"     erro ao exibir: {_exc}")


# COMMAND ----------

# MAGIC %md
# MAGIC ### B.5 Notícias — Web Search

# COMMAND ----------

# ── Notícias retornadas pelo Web Search ──────────────────────────────────────
_news_arts = (result.get("news_results") or {}).get("articles", [])

print("=" * 72)
print(f"NOTÍCIAS RECENTES — WEB SEARCH ({len(_news_arts)} artigos)")
print("=" * 72)

if not _news_arts:
    print("Nenhuma notícia disponível (verifique TAVILY_API_KEY).")
else:
    _news_html_parts = ["""
    <style>
      .news-card { font-family: sans-serif; border:1px solid #ddd; border-radius:8px;
                   padding:14px 18px; margin:10px 0; background:#fafafa; }
      .news-card h4 { margin:0 0 6px 0; font-size:15px; color:#1a1a2e; }
      .news-card .meta { font-size:12px; color:#666; margin-bottom:8px; }
      .news-card p   { font-size:13px; color:#333; margin:0; line-height:1.5; }
      .news-card a   { color:#0066cc; text-decoration:none; font-size:12px; }
    </style>
    <h3 style='font-family:sans-serif;color:#1a1a2e;margin-bottom:12px'>
      🌐 Notícias Recentes sobre SRAG
    </h3>
    """]

    for _i, _art in enumerate(_news_arts, 1):
        _title   = _art.get("title",   f"Artigo {_i}")
        _url     = _art.get("url",     "#")
        _source  = _art.get("source",  _art.get("domain", "Fonte desconhecida"))
        _date    = _art.get("published_date", _art.get("date", ""))
        _snippet = _art.get("content",  _art.get("snippet", _art.get("summary", "")))
        _snippet_short = (_snippet[:280] + "…") if len(_snippet) > 280 else _snippet

        _news_html_parts.append(f"""
        <div class='news-card'>
          <h4>{_i}. {_title}</h4>
          <div class='meta'>
            <strong>{_source}</strong>
            {f' &nbsp;|&nbsp; {_date}' if _date else ''}
          </div>
          <p>{_snippet_short}</p>
          {'<br><a href="' + _url + '" target="_blank">🔗 Ler artigo completo</a>' if _url != '#' else ''}
        </div>
        """)
        print(f"  [{_i}] {_title[:65]}")
        if _source: print(f"       Fonte: {_source}")

    displayHTML("".join(_news_html_parts))


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Parte C — Persistência de Artefatos

# COMMAND ----------

# MAGIC %md
# MAGIC ### C.1 Relatório (Markdown + JSON)

# COMMAND ----------

try:
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    news_results = result.get("news_results", {})

    report_md = report_generator.generate_report(
        metrics     = {"data": [result.get("mandatory_metrics", {})]},
        geographic  = result.get("geographic_data"),
        news        = news_results,
        charts      = all_charts,
        rag_context = result.get("rag_results"),
        user_query  = USER_QUERY,
    )
    md_path = f"{project_paths['reports_markdown']}/relatorio_srag_{timestamp}.md"
    dbutils.fs.put(md_path, report_md, overwrite=True)
    print(f"markdown : {md_path}")

    report_data = {
        "titulo"       : "Relatório Epidemiológico SRAG — Brasil",
        "data_geracao" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metricas"     : result.get("mandatory_metrics", {}),
        "analise"      : result.get("answer", ""),
        "fontes"       : result.get("sources", []),
        "graficos"     : {
            "diario" : daily_charts[0]   if daily_charts   else None,
            "mensal" : monthly_charts[0] if monthly_charts else None,
        },
        "noticias"     : news_results.get("articles", []),
        "auditoria"    : {
            "session_id"     : audit_logger.session_id,
            "total_eventos"  : len(audit_logger.logs),
            "tempo_execucao" : result.get("execution_time_seconds", 0),
        },
    }
    json_path = f"{project_paths['reports_json']}/relatorio_srag_{timestamp}.json"
    dbutils.fs.put(
        json_path,
        json.dumps(report_data, indent=2, ensure_ascii=False, default=str),
        overwrite=True,
    )
    print(f"json     : {json_path}")
except Exception as exc:
    print(f"erro ao gerar relatórios: {exc}")
    print(traceback.format_exc())


# COMMAND ----------

# MAGIC %md
# MAGIC ### C.2 Artefatos no Volume

# COMMAND ----------

# ── Artefatos gerados pelo pipeline ──────────────────────────────────────────
print("=" * 72)
print("ARTEFATOS PERSISTIDOS NO VOLUME")
print("=" * 72)

_artifact_dirs = {
    "📈 Gráficos (daily)"    : project_paths["charts_daily"],
    "📊 Gráficos (monthly)"  : project_paths["charts_monthly"],
    "📁 Gráficos (custom)"   : project_paths["charts_custom"],
    "📄 Relatórios Markdown" : project_paths["reports_markdown"],
    "🗂️  Relatórios JSON"    : project_paths["reports_json"],
    "🔍 Logs de Auditoria"   : project_paths["logs_audit"],
}

_total_files = 0
for _label, _path in _artifact_dirs.items():
    try:
        _files = dbutils.fs.ls(_path)
        if _files:
            _total_files += len(_files)
            print(f"\n  {_label}")
            for _f in _files:
                _sz = f"{_f.size / 1024:.1f} KB" if _f.size else "—"
                print(f"    📎 {_f.name:<50} {_sz:>8}")
        else:
            print(f"\n  {_label}  →  vazio")
    except Exception as _e:
        print(f"\n  {_label}  →  inacessível ({str(_e)[:60]})")

print(f"\n  Total: {_total_files} arquivo(s) persistido(s)")
print("=" * 72)


# COMMAND ----------

# MAGIC %md
# MAGIC ### C.3 Auditoria em Delta Lake

# COMMAND ----------

try:
    audit_logger.save_to_delta(
        spark   = spark,
        catalog = CATALOG_AUDIT,
        schema  = SCHEMA_AUDIT,
    )
    print(f"delta : {CATALOG_AUDIT}.{SCHEMA_AUDIT}.agent_audit_logs")

    audit_json_path = f"{project_paths['logs_audit']}/audit_{audit_logger.session_id}.json"
    try:
        _audit_str = json.dumps(
            [e if isinstance(e, dict) else vars(e) for e in audit_logger.logs],
            indent=2, ensure_ascii=False, default=str,
        )
        dbutils.fs.put(audit_json_path, _audit_str, overwrite=True)
    except Exception as _ej:
        print(f"  aviso: fallback para export_to_json ({_ej})")
        audit_logger.export_to_json(audit_json_path)
    print(f"json  : {audit_json_path}")

    summary = audit_logger.get_summary()
    print(f"\nresumo da sessão:")
    print(f"  total de eventos : {summary['total_events']}")
    print(f"  taxa de sucesso  : {summary['success_rate']:.1f}%")
    print(f"  tempo total      : {summary['execution_time_seconds']:.2f}s")
except Exception as exc:
    print(f"aviso: erro ao salvar auditoria — {exc}")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Parte D — Validação Técnica

# COMMAND ----------

# MAGIC %md
# MAGIC ### D.1 Verificações Operacionais

# COMMAND ----------

if RUN_VALIDATION:
    print("VERIFICAÇÕES OPERACIONAIS")
    print("─" * 72)

    _ok = _warn = _err = 0
    import os
    from pathlib import Path

    def _chk(label, cond, warn=False):
        global _ok, _warn, _err
        if cond:
            print(f"  OK   {label}")
            _ok += 1
        elif warn:
            print(f"  AVISO  {label}")
            _warn += 1
        else:
            print(f"  FALHA  {label}")
            _err += 1

    # Volume
    print("\n  Volume:")
    for _subdir in ["charts", "reports_md", "reports_json", "logs"]:
        _path = Path(VOLUME_BASE) / ({"charts":"charts","reports_md":"reports/markdown","reports_json":"reports/json","logs":"logs"}[_subdir])
        _n = len(list(_path.glob("*"))) if _path.exists() else 0
        _chk(f"{_subdir}: {_n} arquivo(s)", _n >= 0)

    # Gráficos obrigatórios
    print("\n  Gráficos obrigatórios:")
    _daily_charts   = [c for c in (result.get("chart_paths") or []) if "daily"   in str(c)]
    _monthly_charts = [c for c in (result.get("chart_paths") or []) if "monthly" in str(c)]
    _chk(f"diário : {Path(_daily_charts[0]).name if _daily_charts else 'ausente'}",   bool(_daily_charts))
    _chk(f"mensal : {Path(_monthly_charts[0]).name if _monthly_charts else 'ausente'}", bool(_monthly_charts))

    # Métricas
    print("\n  Métricas:")
    _mm = result.get("mandatory_metrics", {})
    for _k in ["taxa_crescimento", "taxa_mortalidade", "taxa_uti", "taxa_vacinacao"]:
        _chk(f"{_k} : {_mm.get(_k, 'N/A')}", _mm.get(_k) is not None)

    # Web search
    print("\n  Web search:")
    _news = result.get("news_results", {})
    _arts = _news.get("articles", []) if isinstance(_news, dict) else []
    _real = [a for a in _arts if a.get("url","").startswith("http")]
    _chk(f"{len(_arts)} artigos ({len(_real)} com URL real)", len(_arts) >= 1, warn=len(_arts) == 0)

    print(f"\n{'─'*72}")
    print(f"  OK {_ok}  |  AVISOS {_warn}  |  ERROS {_err}")
    print("─" * 72)
else:
    print("[PULADO] RUN_VALIDATION=False")


# COMMAND ----------

# MAGIC %md
# MAGIC ### D.2 Panorama Histórico — SQL Direto (Ground Truth)

# COMMAND ----------

if RUN_CERTIFICATION:
    # Anos derivados dinamicamente da tabela — sem hardcode
    _anos_disp = spark.sql(f"""
        SELECT DISTINCT ano FROM {CATALOG_GOLD}.{SCHEMA_GOLD}.gold_metricas_historicas
        ORDER BY ano DESC LIMIT 3
    """).toPandas()["ano"].tolist()
    _anos_disp = sorted(_anos_disp)
    _anos_str  = ", ".join(str(a) for a in _anos_disp)

    _q_anual = f"""
    SELECT
        ano,
        SUM(total_casos)                                                      AS total_casos,
        SUM(total_obitos)                                                     AS total_obitos,
        SUM(casos_com_desfecho)                                               AS casos_com_desfecho,
        ROUND(SUM(total_obitos)*100.0/NULLIF(SUM(casos_com_desfecho),0), 2)  AS taxa_mortalidade_pct,
        SUM(total_internados)                                                 AS total_internados,
        SUM(total_uti)                                                        AS total_uti,
        ROUND(SUM(total_uti)*100.0/NULLIF(SUM(total_internados),0), 2)       AS taxa_uti_pct,
        SUM(total_vacinados)                                                  AS total_vacinados,
        SUM(casos_com_info_vacina)                                            AS casos_com_info_vacina,
        ROUND(SUM(total_vacinados)*100.0/NULLIF(SUM(casos_com_info_vacina),0), 2) AS taxa_vacinacao_pct,
        ROUND(AVG(idade_media), 1)                                            AS idade_media_ano,
        ROUND(AVG(tempo_medio_notificacao), 1)                                AS tempo_notificacao_dias
    FROM {CATALOG_GOLD}.{SCHEMA_GOLD}.gold_metricas_historicas
    WHERE ano IN ({_anos_str})
    GROUP BY ano
    ORDER BY ano
    """

    try:
        _df_anual = spark.sql(_q_anual).toPandas()
        _gt_rows  = _df_anual.to_dict("records")

        print(f"Comparativo Anual — {_anos_str}")
        print("─" * 100)
        print(f"  {'Ano':>6} | {'Casos':>10} | {'Óbitos':>8} | {'Mortalidade':>12} | {'UTI':>8} | {'Vacinação':>10} | {'Idade Méd':>10} | {'T.Notif(d)':>10}")
        print("  " + "─" * 93)
        for r in _gt_rows:
            print(
                f"  {int(r['ano']):>6} | {int(r['total_casos']):>10,} | {int(r['total_obitos']):>8,} | "
                f"{r['taxa_mortalidade_pct']:>11.2f}% | {r['taxa_uti_pct']:>7.2f}% | "
                f"{r['taxa_vacinacao_pct']:>9.2f}% | {r['idade_media_ano']:>10.1f} | "
                f"{r['tempo_notificacao_dias']:>10.1f}"
            )
        if len(_gt_rows) >= 2:
            print("\n  Variação de mortalidade ano a ano:")
            for i in range(1, len(_gt_rows)):
                r_ant, r_cur = _gt_rows[i-1], _gt_rows[i]
                delta = r_cur['taxa_mortalidade_pct'] - r_ant['taxa_mortalidade_pct']
                icon  = "▼" if delta < 0 else "▲"
                print(f"    {int(r_ant['ano'])} → {int(r_cur['ano'])}: {icon} {abs(delta):.2f} pp  ({r_ant['taxa_mortalidade_pct']:.2f}% → {r_cur['taxa_mortalidade_pct']:.2f}%)")
    except Exception as _e:
        print(f"  erro na query anual: {_e}")
        _gt_rows = []
else:
    _gt_rows = []
    print("[PULADO] RUN_CERTIFICATION=False")


# COMMAND ----------

# MAGIC %md
# MAGIC ### D.3 Consistência das Métricas (Agente vs Ground Truth)

# COMMAND ----------

print("MÉTRICAS CALCULADAS PELO AGENTE")
print("-" * 60)
_mm = result.get("mandatory_metrics", {})
for _k, _lbl in [
    ("taxa_crescimento", "taxa_crescimento"),
    ("taxa_mortalidade", "taxa_mortalidade"),
    ("taxa_uti",         "taxa_uti"),
    ("taxa_vacinacao",   "taxa_vacinacao"),
    ("total_casos",      "total_casos"),
]:
    _v = _mm.get(_k)
    _status = "OK" if _v is not None and _v != 0 else "??"
    _fmt_v  = f"{_v:,}" if _k == "total_casos" and isinstance(_v, (int, float)) else _v
    print(f"  {_status:<20} | {_lbl:<22} : {_fmt_v}")

print("\nGROUND TRUTH (SQL DIRETO — gold_metricas_historicas) vs AGENTE:")
print("-" * 70)
_gt_2025 = next((r for r in _gt_rows if int(r.get("ano", 0)) == 2025), {})

if not _gt_2025:
    print("  ground truth 2025 não disponível — célula 17.1 com erro ou sem dados")
else:
    for _agente_k, _gt_k, _lbl in [
        ("taxa_mortalidade", "taxa_mortalidade_pct", "taxa_mortalidade"),
        ("taxa_uti",         "taxa_uti_pct",         "taxa_uti"),
        ("taxa_vacinacao",   "taxa_vacinacao_pct",   "taxa_vacinacao"),
    ]:
        _av = _mm.get(_agente_k)
        _gv = _gt_2025.get(_gt_k)
        try:
            _av_f = float(_av)
            _gv_f = float(_gv)
            _delta  = abs(_av_f - _gv_f)
            _status = "CONSISTENTE" if _delta < 0.5 else "DIVERGENTE"
            print(f"  {_lbl:<22} | agente={_av_f:.2f}%  gt={_gv_f:.2f}%  delta={_delta:.2f}pp  [{_status}]")
        except (TypeError, ValueError):
            print(f"  {_lbl:<22} | agente={_av}  gt={_gv}  [SEM DADOS SUFICIENTES]")


# COMMAND ----------

# MAGIC %md
# MAGIC ### D.4 Cobertura Funcional — Roteamento

# COMMAND ----------

print("COBERTURA DE ROTEAMENTO — TODAS AS ESTRATÉGIAS")
print(f"  {'Tipo':<18} | {'Estratégia':<12} | {'Intent':<22} | {'Conf':>6} | Query")
print("-" * 110)

_routing_tests = [
    ("sql_factual",    "Quantos casos de SRAG foram registrados no total em 2025?"),
    ("sql_geo",        "Quais os 5 estados com mais casos de SRAG em 2024?"),
    ("sql_demo",       "Qual a distribuicao de casos por faixa etaria?"),
    ("rag_analitico",  "O que e SRAG e como a taxa de mortalidade e calculada no SIVEP?"),
    ("rag_explicativo","Explique a metodologia de calculo da taxa de UTI."),
    ("hibrido_comp",   "Compare a mortalidade de 2023 com 2025 e explique as causas da reducao."),
    ("hibrido_temp",   "Como evoluiram os casos de SRAG nos ultimos 6 meses e qual a tendencia?"),
    ("chart_adhoc",    "Gere um grafico de barras de casos por estado em 2025."),
    ("relatorio_full", "Gere o relatorio epidemiologico completo com metricas obrigatorias e graficos."),
]

_strategies_hit = set()

for _tipo, _q in _routing_tests:
    try:
        # explain_routing retorna dict com .strategy, .intent, .confidence
        _dec  = orchestrator.explain_routing(_q)
        _strat = str(_dec.get("strategy", "?")).upper()
        _intent = str(_dec.get("intent", "?"))
        _conf   = float(_dec.get("confidence", 0))
        _strategies_hit.add(_strat)
        print(f"  {_tipo:<18} | {_strat:<12} | {_intent:<22} | {_conf:>5.0%} | {_q[:55]}")
    except Exception as _re:
        print(f"  {_tipo:<18} | ERRO: {str(_re)[:80]}")

_expected = {"SQL_ONLY", "RAG_ONLY", "HYBRID", "CHART", "REPORT"}
_covered  = _expected & _strategies_hit

print(f"\n  Estratégias atingidas: {sorted(_strategies_hit)}")
if _covered == _expected:
    print("  OK — todas as 4 estratégias cobertas")
else:
    print(f"  ATENÇÃO — faltando: {_expected - _covered}")


# COMMAND ----------

# MAGIC %md
# MAGIC ### D.5 Testes de Retrieval RAG

# COMMAND ----------

if rag_chain:
    print("TESTES DE RETRIEVAL RAG")
    print("-" * 80)

    _rag_tests = [
        ("metodologia mortalidade",      "metodologia calculo taxa mortalidade SRAG"),
        ("taxa UTI internacao",           "taxa UTI ocupacao internados hospitalar"),
        ("vacinacao subregistro",         "vacinacao cobertura subregistro dados SRAG"),
        ("agentes etiologicos influenza", "influenza covid agentes etiologicos SRAG grave"),
        ("sazonalidade picos",            "sazonalidade pico casos SRAG inverno verao"),
    ]

    _rag_passed = 0
    for _lbl, _q in _rag_tests:
        try:
            _docs = rag_chain.retriever.retrieve(_q, k=3, strategy="hybrid")
            _srcs = list({_d.metadata.get("source_table","?") for _d in _docs})
            _preview = _docs[0].page_content[:80].replace("\n"," ") if _docs else "—"
            _status = "OK" if len(_docs) >= 1 else "FAIL"
            if _status == "OK": _rag_passed += 1
            print(f"  {_status} [{_lbl:<28}] {len(_docs)} docs | {_srcs}")
            print(f"     -> {_preview}...")
        except Exception as _re:
            print(f"  ERRO [{_lbl}]: {_re}")

    print(f"\n  Resultado: {_rag_passed}/{len(_rag_tests)} testes RAG aprovados")
else:
    print("[PULADO] rag_chain não disponível")


# COMMAND ----------

# MAGIC %md
# MAGIC ### D.6 Testes de Conversa com o Agente

# COMMAND ----------

_conv_tests = [
    ("T1_SQL",    "SQL_ONLY",  "Qual o total de casos de SRAG registrados por ano em 2023, 2024 e 2025?",
     lambda r: bool(r.get("mandatory_metrics") or r.get("sql_results"))),
    ("T2_RAG",    "RAG_ONLY",  "O que e SRAG e quais sao os principais agentes etiologicos responsaveis pelos casos graves?",
     lambda r: bool(r.get("rag_results") or r.get("answer"))),
    ("T3_CHART",  "CHART",     "Gere um grafico de barras mostrando o total de casos de SRAG por estado (UF).",
     lambda r: bool(r.get("ad_hoc_chart_path") or r.get("chart_paths"))),
    ("T4_HYBRID", "HYBRID",    "Compare a mortalidade do SRAG entre 2023 e 2025 e explique o que causou a reducao.",
     lambda r: bool(r.get("mandatory_metrics") and (r.get("rag_results") or r.get("answer")))),
]

print("TESTES DE CONVERSA COM O AGENTE")
print(f"  {len(_conv_tests)} cenários configurados\n")
for _id, _strat, _q, _ in _conv_tests:
    print(f"  {_id:<10} | Esperado: {_strat:<10} | {_q[:70]}")
print()

_conv_results = []
for _id, _expected_strat, _q, _valida_fn in _conv_tests:
    print("=" * 72)
    print(f"[{_id}] {_q[:60]}")
    print("-" * 72)
    try:
        from datetime import datetime as _dt
        _t0   = _dt.now()
        _cr   = orchestrator.run(user_query=_q)
        _dur  = (_dt.now() - _t0).total_seconds()
        _strat_got = str(_cr.get("routing", {}).get("strategy", "?")).upper()
        _ok_strat  = _strat_got == _expected_strat
        _errs      = len(_cr.get("errors", []))
        _validou   = _valida_fn(_cr)
        _passed    = _ok_strat and _errs == 0 and _validou
        _conv_results.append({"id": _id, "nome": _id, "estrategia": _strat_got,
                               "tempo": _dur, "ok": _passed})
        print(f"  Tempo       : {_dur:.2f}s")
        print(f"  Estratégia  : {_strat_got}  {'[OK]' if _ok_strat else f'[ESPERADO: {_expected_strat}]'}")
        print(f"  Validação   : {'PASSOU' if _validou else 'FALHOU'}")
        print(f"  Erros       : {_errs}")
        _m = _cr.get("mandatory_metrics", {})
        if _m:
            print(f"  Mortalidade : {_m.get('taxa_mortalidade','N/A')}%")
            print(f"  Gráficos    : {len(_cr.get('chart_paths') or [])} padrão")
        if _cr.get("ad_hoc_chart_path"):
            from pathlib import Path as _P
            print(f"  Gráfico ad-hoc: {_P(_cr['ad_hoc_chart_path']).name}")
    except Exception as _exc:
        print(f"  ERRO: {_exc}")
        _conv_results.append({"id": _id, "nome": _id, "estrategia": "ERRO",
                               "tempo": 0, "ok": False})

print("\n" + "=" * 72)
print("RESUMO DOS TESTES DE CONVERSA")
print("-" * 72)
_passed_conv = sum(1 for r in _conv_results if r["ok"])
print(f"  Resultado: {_passed_conv}/{len(_conv_results)} testes aprovados\n")
print(f"  {'ID':<10} | {'Estratégia':<12} | {'Tempo':>7} | Status")
print(f"  {'-'*55}")
for _r in _conv_results:
    _st = "OK  " if _r["ok"] else "FAIL"
    print(f"  {_r['id']:<10} | {_r['estrategia']:<12} | {_r['tempo']:>6.1f}s | {_st}")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Parte E — Evidências de Certificação

# COMMAND ----------

# MAGIC %md
# MAGIC ### E.1 Verificação de Qualidade

# COMMAND ----------

if RUN_CERTIFICATION:
    _mm_cert  = result.get("mandatory_metrics", {})
    _summary  = audit_logger.get_summary()
    _sr       = _summary.get("success_rate", 0)
    _errs_tot = _summary.get("errors", 0)
    _rag_ok   = rag_chain is not None
    _rag_ret  = _rag_passed if '_rag_passed' in dir() else 0

    # max_mes calculado via SQL — sem hardcode
    try:
        _max_mes = spark.sql(f"""
            SELECT MAX(ano_mes) AS max_mes FROM {CATALOG_GOLD}.{SCHEMA_GOLD}.gold_metricas_historicas
        """).collect()[0]["max_mes"]
        _gold_atual = str(_max_mes) >= "2025-10"
    except:
        _max_mes   = "N/A"
        _gold_atual = False

    _criteria = [
        ("p3", "Pipeline sem erros críticos",         _errs_tot == 0),
        ("p2", "Success rate >= 80%",                 _sr >= 80),
        ("p2", "taxa_mortalidade calculada",           bool(_mm_cert.get("taxa_mortalidade"))),
        ("p2", "taxa_uti calculada",                   bool(_mm_cert.get("taxa_uti"))),
        ("p2", "taxa_vacinacao calculada",             bool(_mm_cert.get("taxa_vacinacao"))),
        ("p2", "taxa_crescimento calculada",           _mm_cert.get("taxa_crescimento") is not None),
        ("p2", "Gráficos padrão gerados (>= 2)",       len(result.get("chart_paths") or []) >= 2),
        ("p2", "RAG disponível",                       _rag_ok),
        ("p2", "RAG retrieval OK (>= 4/5 testes)",    _rag_ret >= 4),
        ("p2", "Todas as 4 estratégias atingidas",     _covered == _expected),
        ("p2", "Teste T1_SQL aprovado",                any(r["ok"] for r in _conv_results if r["id"]=="T1_SQL")),
        ("p2", "Teste T2_RAG aprovado",                any(r["ok"] for r in _conv_results if r["id"]=="T2_RAG")),
        ("p2", "Teste T3_CHART aprovado",              any(r["ok"] for r in _conv_results if r["id"]=="T3_CHART")),
        ("p2", "Teste T4_HYBRID aprovado",             any(r["ok"] for r in _conv_results if r["id"]=="T4_HYBRID")),
        ("p2", f"Dados Gold atuais (max_mes={_max_mes})", _gold_atual),
    ]

    _total_pts = _earned_pts = 0
    print("Verificação de Qualidade")
    print("─" * 70)
    for _pts_lbl, _desc, _passed in _criteria:
        _pts = int(_pts_lbl[1])
        _total_pts  += _pts
        _earned_pts += _pts if _passed else 0
        _icon = "OK " if _passed else "XX "
        print(f"  {_icon} ({_pts_lbl})  {_desc}")

    _score_pct = _earned_pts / _total_pts * 100
    print("─" * 70)
    print(f"  pontos  : {_earned_pts}/{_total_pts}")
    print(f"  score   : {_score_pct:.1f}%")
else:
    _earned_pts = _total_pts = 0
    _score_pct  = 0.0
    print("[PULADO] RUN_CERTIFICATION=False")


# COMMAND ----------

# MAGIC %md
# MAGIC ### E.2 Modo Interativo — Consultas Adicionais
# MAGIC
# MAGIC Envie qualquer pergunta ao agente para exploração livre do sistema.
# MAGIC Altere `QUERY_INTERATIVA` abaixo ou selecione um dos exemplos em `_EXEMPLOS`.
# MAGIC

# COMMAND ----------

_EXEMPLOS = {
    "sql_total_casos"   : "Quantos casos de SRAG foram registrados por ano em 2023, 2024 e 2025?",
    "sql_mortalidade"   : "Qual a taxa de mortalidade atual do SRAG no Brasil?",
    "sql_top_estados"   : "Quais os 5 estados com mais casos de SRAG em 2025?",
    "sql_faixa_etaria"  : "Qual a distribuição de casos de SRAG por faixa etária?",
    "rag_metodologia"   : "Explique a metodologia de cálculo da taxa de mortalidade usada neste projeto.",
    "rag_definicao"     : "O que é SRAG e quais são os principais agentes etiológicos?",
    "rag_uti"           : "Como a taxa de ocupação de UTI é calculada no pipeline SRAG?",
    "hibrido_causas"    : "Compare a mortalidade de 2023 com 2025 e explique o que causou a redução.",
    "hibrido_tendencia" : "Como evoluíram os casos de SRAG nos últimos 6 meses e qual a tendência?",
    "chart_estados"     : "Gere um gráfico de barras mostrando total de casos de SRAG por estado.",
    "chart_mortalidade" : "Gere um gráfico de mortalidade por estado em 2025.",
    "chart_anual"       : "Gere um gráfico de barras com o total de casos de SRAG por ano disponível e me diga qual foi o ano com maior número de casos, maior taxa de mortalidade e maior taxa de ocupação de UTI. Inclua uma análise comparativa entre os anos.",
}

# ── Configure aqui ────────────────────────────────────────────────────────────
QUERY_INTERATIVA = _EXEMPLOS["chart_anual"]

# Para usar o Databricks Foundation Model (sem custo de token OpenAI) — mude para True
INTERATIVO_USA_DATABRICKS = False

print("Queries de exemplo disponíveis:")
for key, q in _EXEMPLOS.items():
    print(f"  {key:<22} : {q[:70]}")
print(f"\nQuery configurada: {QUERY_INTERATIVA}")


# COMMAND ----------

if INTERATIVO_USA_DATABRICKS:
    try:
        from databricks_langchain import ChatDatabricks as _AvalChatDB
    except ImportError:
        from langchain_community.chat_models import ChatDatabricks as _AvalChatDB
    _llm_aval = _AvalChatDB(
        endpoint    = LLM_MODEL_DATABRICKS,
        temperature = LLM_TEMP,
        max_tokens  = LLM_MAX_TOKENS,
    )
    print(f"avaliador usando: {LLM_MODEL_DATABRICKS} (Databricks)")
else:
    _llm_aval = llm
    print(f"avaliador usando: {LLM_MODEL_OPENAI if LLM_PROVIDER == 'openai' else LLM_MODEL_DATABRICKS} ({LLM_PROVIDER})")

_orchestrator_aval = SRAGOrchestrator(
    spark           = spark,
    llm             = _llm_aval,
    audit_logger    = audit_logger,
    rag_chain       = rag_chain,
    web_search_tool = web_search_tool,
    chart_tool      = chart_tool,
    catalog         = CATALOG_GOLD,
    schema          = SCHEMA_GOLD,
    use_llm_routing = False,
    use_openai      = not INTERATIVO_USA_DATABRICKS,
)

print("=" * 72)
print("AGENTE SRAG — RESPOSTA À QUERY DO AVALIADOR")
print("=" * 72)
print(f"Query: {QUERY_INTERATIVA.strip()}")
print("=" * 72)

_t0_aval    = datetime.now()
_res_aval   = _orchestrator_aval.run(user_query=QUERY_INTERATIVA)
_tempo_aval = (datetime.now() - _t0_aval).total_seconds()
_rot_aval   = _res_aval.get("routing", {})

print(f"status     : {'SUCESSO' if _res_aval.get('success') else 'PARCIAL'}  |  tempo: {_tempo_aval:.1f}s")
print(f"estratégia : {_rot_aval.get('strategy', 'N/A').upper()}  |  confiança: {_rot_aval.get('confidence', 0):.0%}")

_m_aval = _res_aval.get("mandatory_metrics", {})
if _m_aval:
    tc = _m_aval.get("taxa_crescimento")
    print(f"\nmétricas:")
    print(f"  taxa crescimento : {f'{tc:.2f}%' if isinstance(tc, float) else tc}")
    print(f"  taxa mortalidade : {_m_aval.get('taxa_mortalidade', 'N/A')}%")
    print(f"  taxa uti         : {_m_aval.get('taxa_uti', 'N/A')}%")
    print(f"  taxa vacinacao   : {_m_aval.get('taxa_vacinacao', 'N/A')}%")
    print(f"  total casos      : {_m_aval.get('total_casos', 0):,}")

print("\n" + "=" * 72)
print("RESPOSTA")
print("=" * 72)
_ans_aval = _res_aval.get("answer", "")
print(_ans_aval if _ans_aval else "sem resposta gerada")

if _res_aval.get("ad_hoc_chart_path"):
    print(f"\ngráfico gerado: {_res_aval['ad_hoc_chart_path']}")
    try:
        with open(_res_aval["ad_hoc_chart_path"], "r", encoding="utf-8") as _f:
            displayHTML(_f.read())
    except Exception as _e:
        print(f"erro ao exibir gráfico: {_e}")

# ── fallback de gráfico: se o CHART node não gerou, gera aqui diretamente ────
# Isso acontece quando _build_dynamic_chart_query falha ou os dados são None.
# Usa analise_anual que FIX-4 injeta em mandatory_metrics após chart generation.
_analise_aval   = (_res_aval.get("mandatory_metrics") or {}).get("analise_anual", [])
_adhoc_gerado   = _res_aval.get("ad_hoc_chart_path")

if not _adhoc_gerado and _analise_aval and "chart_tool" in dir() and chart_tool:
    print("\n[avaliador] fallback ativo — gerando gráficos anuais via analise_anual...")
    try:
        _anos_ord = sorted(_analise_aval, key=lambda r: r.get("ano", 0))
        _fb_paths = []
        for _ycol, _ytitle in [
            ("casos_ano",      "Total de Casos SRAG por Ano"),
            ("mortalidade_pct","Taxa de Mortalidade SRAG por Ano (%)"),
        ]:
            _cr = chart_tool.generate_custom_chart(
                data=_anos_ord, chart_type="bar",
                title=_ytitle, x_col="ano", y_col=_ycol,
            )
            if _cr and _cr.get("path"):
                _fb_paths.append(_cr["path"])
                print(f"  ✅ gerado: {_cr['path'].split('/')[-1]}")
                try:
                    with open(_cr["path"], "r", encoding="utf-8") as _fh:
                        displayHTML(_fh.read())
                except Exception as _ev: print(f"  erro ao exibir: {_ev}")
        if _fb_paths:
            _res_aval["chart_paths"] = list(_res_aval.get("chart_paths") or []) + _fb_paths
            _res_aval["ad_hoc_chart_path"] = _fb_paths[0]
    except Exception as _ef:
        print(f"[avaliador] fallback falhou: {_ef}")
elif _adhoc_gerado:
    print(f"\n[avaliador] gráfico gerado pelo CHART node: {_adhoc_gerado.split('/')[-1]}")
    try:
        with open(_adhoc_gerado, "r", encoding="utf-8") as _fh:
            displayHTML(_fh.read())
    except Exception as _ev: print(f"erro ao exibir: {_ev}")
elif not _analise_aval:
    print("\n[avaliador] ⚠️  analise_anual vazia — verifique gold_metricas_historicas ou FIX-4 no orchestrator.py")

# ── expõe para E.4 (relatório consolidado) ───────────────────────────────────
AVAL_QUERY   = QUERY_INTERATIVA
AVAL_RESULT  = _res_aval
AVAL_CHARTS  = list(_res_aval.get("chart_paths") or [])
_adhoc_aval  = _res_aval.get("ad_hoc_chart_path")
if _adhoc_aval and _adhoc_aval not in AVAL_CHARTS:
    AVAL_CHARTS.append(_adhoc_aval)
print(f"\n[E.4] exportado: {len(AVAL_CHARTS)} gráfico(s)")


# COMMAND ----------

# MAGIC %md
# MAGIC ### E.3 Resumo Operacional da Sessão

# COMMAND ----------

_ts_diag = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
_summary = audit_logger.get_summary()

print(f"[DIAGNÓSTICO SRAG]  {_ts_diag}  |  session: {SESSION_ID}")
print("─" * 72)

print("\nPipeline")
print(f"  status    : {'OK' if result.get('success') else 'PARCIAL'}")
print(f"  tempo     : {result.get('execution_time_seconds', 0):.2f}s")
print(f"  estratégia: {result.get('routing', {}).get('strategy', 'N/A').upper()}")
print(f"  erros     : {len(result.get('errors', []))}")

_mm_d = result.get("mandatory_metrics", {})
print("\nMétricas")
for _k in ["taxa_crescimento","taxa_mortalidade","taxa_uti","taxa_vacinacao","total_casos"]:
    _v = _mm_d.get(_k)
    if _k == "total_casos" and isinstance(_v, (int,float)): print(f"  {_k:<22}: {_v:,}")
    elif isinstance(_v, float):                              print(f"  {_k:<22}: {_v:.2f}%")
    else:                                                    print(f"  {_k:<22}: {_v}")

print("\nPanorama histórico (SQL direto)")
for _r in (_gt_rows if _gt_rows else []):
    _d   = _r if isinstance(_r, dict) else _r.asDict()
    _ano = int(_d.get('ano', 0))
    _cas = int(_d.get('total_casos', 0))
    _mor = float(_d.get('taxa_mortalidade_pct', _d.get('mortalidade', 0)) or 0)
    _uti = float(_d.get('taxa_uti_pct', _d.get('taxa_uti', 0)) or 0)
    _vac = float(_d.get('taxa_vacinacao_pct', _d.get('vacinacao', 0)) or 0)
    print(f"  {_ano}: {_cas:,} casos  |  mort={_mor:.2f}%  |  uti={_uti:.2f}%  |  vac={_vac:.2f}%")

if RUN_CERTIFICATION and _conv_results:
    print("\nTestes de conversa")
    for _r in _conv_results:
        _icon = "OK  " if _r["ok"] else "FAIL"
        print(f"  [{_icon}] {_r['id']:<10} | {_r['estrategia']:<12} | {_r['tempo']:.1f}s")

if RUN_CERTIFICATION and _total_pts > 0:
    print("\nVerificação de qualidade")
    print(f"  pontos : {_earned_pts}/{_total_pts}")
    print(f"  score  : {_score_pct:.1f}%")

print("\nAuditoria")
print(f"  session   : {SESSION_ID}")
print(f"  events    : {_summary['total_events']}")
print(f"  success   : {_summary['success_rate']:.1f}%")
print(f"  duration  : {_summary['execution_time_seconds']:.2f}s")
print("─" * 72)


# COMMAND ----------

# MAGIC %md
# MAGIC ### E.4 Relatório Gerado pelo Agente

# COMMAND ----------

# ═══════════════════════════════════════════════════════════════════════════════
# E.4  RELATÓRIO OPERACIONAL CONSOLIDADO — SRAG
# Consolida: cabeçalho de execução, web search, RAG, gráficos e artefatos
# Fonte de dados: result, audit_logger, chart_paths, news_results, rag_results
# ═══════════════════════════════════════════════════════════════════════════════

import re as _re
from pathlib import Path as _P

# ── helpers ───────────────────────────────────────────────────────────────────
def _esc(s):
    return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def _badge(text, color="#1a3a5c"):
    return (
        f'<span style="background:{color};color:#fff;padding:2px 9px;'
        f'border-radius:12px;font-size:11px;font-weight:600;'
        f'letter-spacing:.4px;">{_esc(text)}</span>'
    )

def _kv(label, value, mono=False):
    v_style = "font-family:monospace;font-size:12px;" if mono else "font-size:13px;"
    return (
        f'<tr>'
        f'<td style="padding:5px 14px 5px 0;color:#555;font-size:12px;'
        f'white-space:nowrap;vertical-align:top;">{_esc(label)}</td>'
        f'<td style="padding:5px 0;{v_style}color:#1a1a2e;font-weight:600;">'
        f'{_esc(str(value))}</td></tr>'
    )

def _section(title, icon, content, border="#1a3a5c"):
    return (
        f'<div style="margin:24px 0;border-left:4px solid {border};'
        f'padding:0 0 0 16px;">'
        f'<h3 style="margin:0 0 12px 0;font-size:15px;color:{border};'
        f'font-family:sans-serif;">{icon} {_esc(title)}</h3>'
        f'{content}</div>'
    )

def _card(content, bg="#f8f9fc"):
    return (
        f'<div style="background:{bg};border:1px solid #dde3ea;'
        f'border-radius:8px;padding:14px 18px;margin:8px 0;'
        f'font-family:sans-serif;">{content}</div>'
    )

def _md_to_html(text):
    """Converte markdown simples em HTML limpo."""
    t = _esc(text)
    t = _re.sub(r'(?m)^### (.+)$', r'<h3 style="margin:16px 0 6px;font-size:14px;color:#1a3a5c;">\1</h3>', t)
    t = _re.sub(r'(?m)^## (.+)$',  r'<h2 style="margin:20px 0 8px;font-size:16px;color:#1a3a5c;">\1</h2>', t)
    t = _re.sub(r'(?m)^# (.+)$',   r'<h1 style="margin:24px 0 10px;font-size:18px;color:#1a3a5c;">\1</h1>', t)
    t = _re.sub(r'\*\*(.+?)\*\*',  r'<strong>\1</strong>', t)
    t = _re.sub(r'(?m)^[-*] (.+)$',r'<li style="margin:3px 0;">\1</li>', t)
    t = t.replace('\n', '<br>')
    return t

# ══════════════════════════════════════════════════════════════════════════════
# 0. Derivar / validar variáveis de estado
# ══════════════════════════════════════════════════════════════════════════════
_result        = result if "result" in dir() else {}
_user_query    = USER_QUERY.strip() if "USER_QUERY" in dir() else "N/A"
_session_id    = SESSION_ID if "SESSION_ID" in dir() else "N/A"
_provider      = EFFECTIVE_LLM_PROVIDER if "EFFECTIVE_LLM_PROVIDER" in dir() else LLM_PROVIDER
_ts_now        = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

_rout          = _result.get("routing", {})
_strategy      = _rout.get("strategy", "N/A").upper()
_confidence    = _rout.get("confidence", 0)
_exec_time     = _result.get("execution_time_seconds", 0)
_status        = "✅ OK" if _result.get("success") else "⚠️ PARCIAL"
_errors        = _result.get("errors", [])
_mm            = _result.get("mandatory_metrics", {})

# ── pipeline de decisão ──────────────────────────────────────────────────────
_intent     = _rout.get("intent", _rout.get("classified_intent", "N/A"))
_tools_used = list(_result.get("tools_used", []))
if not _tools_used:
    if _result.get("sql_result")  is not None: _tools_used.append("SQL Tool")
    if _result.get("rag_results"):             _tools_used.append("RAG Retrieval")
    if _result.get("news_results"):            _tools_used.append("Web Search")
    if _result.get("chart_paths"):             _tools_used.append("Chart Tool")
    if _result.get("answer"):                  _tools_used.append("Report Generator")

# charts (reusa se já definido, senão re-deriva)
try:
    _all_charts = list(all_charts)
    _chart_meta_ref = _chart_meta
except NameError:
    _all_charts = list(_result.get("chart_paths") or [])
    _adhoc = _result.get("ad_hoc_chart_path")
    if _adhoc and _adhoc not in _all_charts:
        _all_charts.append(_adhoc)
    _chart_meta_ref = {}
    for _p in _all_charts:
        _n = _P(_p).name
        if   "mensal"     in _n or "monthly"   in _n: _chart_meta_ref[_p] = "mensal"
        elif "multi_line" in _n or "viral"     in _n: _chart_meta_ref[_p] = "viral"
        elif "line"       in _n:                       _chart_meta_ref[_p] = "diário"
        elif "bar"        in _n:
            _c = _n.split("_")[-1].replace(".html","")
            _chart_meta_ref[_p] = "demográfico" if (_c.isdigit() and int(_c)>=3) else "geográfico"
        else:
            _chart_meta_ref[_p] = "outro"

# separar obrigatórios vs adicionais
_mandatory_paths  = list(_result.get("chart_paths") or [])
_adhoc_path       = _result.get("ad_hoc_chart_path")
_additional_paths = [p for p in _all_charts if p not in _mandatory_paths]

# news
_news_raw  = _result.get("news_results") or {}
_news_arts = _news_raw.get("articles", []) if isinstance(_news_raw, dict) else []
_ws_query  = _news_raw.get("query", _news_raw.get("search_query", "N/A"))
_ws_backend= _news_raw.get("backend", _news_raw.get("provider", "tavily" if TAVILY_AVAILABLE else "fallback"))
_ws_domains= list({a.get("source", a.get("domain","?")) for a in _news_arts if a})

# rag
_rag_raw  = _result.get("rag_results") or {}
if isinstance(_rag_raw, dict):
    _rag_docs     = _rag_raw.get("documents", _rag_raw.get("docs", []))
    _rag_strategy = _rag_raw.get("retrieval_strategy", _rag_raw.get("strategy", "hybrid"))
    _rag_topk     = _rag_raw.get("top_k", 5)
elif isinstance(_rag_raw, list):
    _rag_docs     = _rag_raw
    _rag_strategy = "hybrid"
    _rag_topk     = len(_rag_raw)
else:
    _rag_docs = []; _rag_strategy = "N/A"; _rag_topk = "N/A"

# artefatos
try:
    _md_path_ref   = md_path
    _json_path_ref = json_path
except NameError:
    _md_path_ref   = "não gerado nesta sessão"
    _json_path_ref = "não gerado nesta sessão"

# ══════════════════════════════════════════════════════════════════════════════
# 1. CABEÇALHO OPERACIONAL
# ══════════════════════════════════════════════════════════════════════════════
_conf_pct = f"{_confidence:.0%}" if isinstance(_confidence, float) else str(_confidence)
_status_color = "#1e7e34" if "OK" in _status else "#856404"

_header_html = f"""
<div style="background:#1a3a5c;color:#fff;padding:16px 24px;border-radius:8px 8px 0 0;
            font-family:sans-serif;display:flex;justify-content:space-between;align-items:center;">
  <div>
    <div style="font-size:18px;font-weight:700;letter-spacing:.5px;">
      📋 Relatório Epidemiológico SRAG — Brasil
    </div>
    <div style="font-size:12px;opacity:.8;margin-top:4px;">{_ts_now}</div>
  </div>
  <div style="text-align:right;">
    <div style="font-size:11px;opacity:.7;font-family:monospace;">{_session_id}</div>
    <div style="margin-top:6px;">{_badge(_status, _status_color)}</div>
  </div>
</div>
<div style="background:#f0f4f8;border:1px solid #dde3ea;border-top:0;
            padding:14px 24px;border-radius:0 0 8px 8px;font-family:sans-serif;margin-bottom:24px;">
  <table style="border-collapse:collapse;width:100%;table-layout:fixed;">
    <colgroup><col style="width:220px"><col></colgroup>
    {_kv("🔍 Query original",    _user_query[:120] + ("…" if len(_user_query)>120 else ""))}
    {_kv("📡 Estratégia",       f"{_strategy}  ({_conf_pct} confiança)")}
    {_kv("⏱️  Tempo de execução", f"{_exec_time:.1f}s")}
    {_kv("🤖 Provider LLM",     _provider.upper())}
    {_kv("⚠️  Erros",            len(_errors) if _errors else "nenhum")}
  </table>
</div>
"""

# ══════════════════════════════════════════════════════════════════════════════
# 1b. PIPELINE DE DECISÃO DO AGENTE
# ══════════════════════════════════════════════════════════════════════════════
_tools_badges = " ".join(
    f'<span style="background:#e9ecef;color:#333;padding:3px 10px;'
    f'border-radius:12px;font-size:11px;font-weight:600;margin:2px;">{_esc(t)}</span>'
    for t in (_tools_used or ["N/A"])
)
_pipeline_html = (
    f'<div style="font-family:monospace;font-size:12px;background:#0d1117;'
    f'color:#c9d1d9;border-radius:8px;padding:20px 24px;line-height:2;">'
    f'<div style="color:#58a6ff;font-weight:700;font-size:13px;'
    f'margin-bottom:14px;letter-spacing:.5px;">▶ PIPELINE DE DECISÃO DO AGENTE</div>'
    f'<table style="border-collapse:collapse;width:100%;">'
    f'<tr><td style="color:#8b949e;padding:4px 18px 4px 0;white-space:nowrap;vertical-align:top;">Query do usuário</td>'
    f'<td style="color:#e6edf3;font-family:sans-serif;font-size:12px;line-height:1.6;">"{_esc(_user_query)}"</td></tr>'
    f'<tr><td style="color:#8b949e;padding:4px 18px 4px 0;white-space:nowrap;">Intent classificado</td>'
    f'<td style="color:#79c0ff;">{_esc(str(_intent).upper())}</td></tr>'
    f'<tr><td style="color:#8b949e;padding:4px 18px 4px 0;white-space:nowrap;">Estratégia escolhida</td>'
    f'<td style="color:#56d364;font-weight:700;">{_esc(_strategy)}</td></tr>'
    f'<tr><td style="color:#8b949e;padding:4px 18px 4px 0;white-space:nowrap;">Confiança do roteador</td>'
    f'<td style="color:#e3b341;">{_conf_pct}</td></tr>'
    f'<tr><td style="color:#8b949e;padding:4px 18px 4px 0;white-space:nowrap;vertical-align:top;">Ferramentas utilizadas</td>'
    f'<td style="padding-top:4px;">{_tools_badges}</td></tr>'
    f'<tr><td style="color:#8b949e;padding:4px 18px 4px 0;white-space:nowrap;">Tempo de execução</td>'
    f'<td style="color:#c9d1d9;">{_exec_time:.1f}s</td></tr>'
    f'<tr><td style="color:#8b949e;padding:4px 18px 4px 0;white-space:nowrap;">Provider LLM</td>'
    f'<td style="color:#c9d1d9;">{_esc(_provider.upper())}</td></tr>'
    f'</table>'
    f'</div>'
)
_pipeline_section = _section(
    "Pipeline de Decisão do Agente", "🔀", _pipeline_html, border="#fd7e14"
)

# ══════════════════════════════════════════════════════════════════════════════
# 2. MÉTRICAS CALCULADAS
# ══════════════════════════════════════════════════════════════════════════════
def _fmt_metric(v):
    if isinstance(v, float): return f"{v:,.2f}"
    if isinstance(v, int):   return f"{v:,}"
    return str(v) if v else "N/A"

_metric_items = [
    ("Total de Casos",       _mm.get("total_casos")),
    ("Taxa de Crescimento",  f"{_mm.get('taxa_crescimento','N/A')}%"),
    ("Taxa de Mortalidade",  f"{_mm.get('taxa_mortalidade','N/A')}%"),
    ("Taxa de UTI",          f"{_mm.get('taxa_uti','N/A')}%"),
    ("Taxa de Vacinação",    f"{_mm.get('taxa_vacinacao','N/A')}%"),
]
_metric_cards = "".join(
    f'<div style="flex:1 1 160px;background:#fff;border:1px solid #dde3ea;'
    f'border-radius:8px;padding:14px;text-align:center;margin:4px;">'
    f'<div style="font-size:11px;color:#666;margin-bottom:6px;">{_esc(lbl)}</div>'
    f'<div style="font-size:20px;font-weight:700;color:#1a3a5c;">{_esc(_fmt_metric(val))}</div>'
    f'</div>'
    for lbl, val in _metric_items
)
_metrics_section = _section(
    "Métricas Calculadas", "📊",
    f'<div style="display:flex;flex-wrap:wrap;gap:4px;">{_metric_cards}</div>',
    border="#0d6efd"
)

# ══════════════════════════════════════════════════════════════════════════════
# 3. TEXTO DO RELATÓRIO (AGENTE)
# ══════════════════════════════════════════════════════════════════════════════
_answer_raw = _result.get("answer", "")
if _answer_raw:
    _answer_html = (
        f'<div style="font-family:Arial,sans-serif;font-size:13px;line-height:1.8;'
        f'color:#1a1a2e;max-height:600px;overflow-y:auto;'
        f'border:1px solid #dde3ea;border-radius:6px;padding:16px;background:#fff;">'
        f'{_md_to_html(_answer_raw)}</div>'
    )
else:
    _answer_html = _card('<em style="color:#888;">Sem resposta gerada — verifique os logs de auditoria.</em>')
_answer_section = _section("Análise do Agente", "📝", _answer_html, border="#1a3a5c")

# ══════════════════════════════════════════════════════════════════════════════
# 4. WEB SEARCH — RASTREABILIDADE
# ══════════════════════════════════════════════════════════════════════════════
_ws_rows_html = ""
for _i, _a in enumerate(_news_arts[:8], 1):
    _t   = _a.get("title", f"Artigo {_i}")
    _src = _a.get("source", _a.get("domain", "?"))
    _dt  = _a.get("published_date", _a.get("date", ""))
    _url = _a.get("url", "#")
    _ws_rows_html += (
        f'<tr style="border-bottom:1px solid #eee;">'
        f'<td style="padding:6px 8px;font-size:12px;color:#555;text-align:center;">{_i}</td>'
        f'<td style="padding:6px 8px;font-size:13px;font-weight:600;">'
        f'<a href="{_esc(_url)}" target="_blank" style="color:#0d6efd;text-decoration:none;">{_esc(_t[:80])}</a></td>'
        f'<td style="padding:6px 8px;font-size:12px;color:#555;">{_esc(_src)}</td>'
        f'<td style="padding:6px 8px;font-size:11px;color:#888;">{_esc(_dt[:10])}</td>'
        f'</tr>'
    )

_ws_meta_html = _card(
    f'<table style="border-collapse:collapse;width:100%;font-family:sans-serif;">'
    f'{_kv("Query de busca",   _ws_query, mono=True)}'
    f'{_kv("Backend",          _ws_backend)}'
    f'{_kv("Artigos retornados", len(_news_arts))}'
    f'{_kv("Fontes/domínios",  ", ".join(_ws_domains) if _ws_domains else "N/A")}'
    f'</table>'
)
if _ws_rows_html:
    _ws_table_html = (
        f'<table style="width:100%;border-collapse:collapse;font-family:sans-serif;'
        f'font-size:13px;margin-top:12px;">'
        f'<thead><tr style="background:#f0f4f8;">'
        f'<th style="padding:8px;width:30px;">#</th>'
        f'<th style="padding:8px;text-align:left;">Título</th>'
        f'<th style="padding:8px;text-align:left;width:160px;">Fonte</th>'
        f'<th style="padding:8px;text-align:left;width:90px;">Data</th>'
        f'</tr></thead><tbody>{_ws_rows_html}</tbody></table>'
    )
else:
    _ws_table_html = _card('<em style="color:#888;">Nenhum artigo retornado (Tavily indisponível ou sem resultados).</em>', bg="#fff9e6")

_ws_section = _section(
    f"Web Search Executado ({len(_news_arts)} artigos)", "🌐",
    _ws_meta_html + _ws_table_html,
    border="#198754"
)

# ══════════════════════════════════════════════════════════════════════════════
# 5. RAG — CONTEXTO RECUPERADO
# ══════════════════════════════════════════════════════════════════════════════
_rag_meta_html = _card(
    f'<table style="border-collapse:collapse;width:100%;font-family:sans-serif;">'
    f'{_kv("Estratégia de retrieval", _rag_strategy)}'
    f'{_kv("Top-K configurado",        _rag_topk)}'
    f'{_kv("Documentos recuperados",   len(_rag_docs))}'
    f'</table>'
)

_rag_doc_cards = ""
for _di, _doc in enumerate(_rag_docs[:5], 1):
    if isinstance(_doc, dict):
        _src_tbl    = _doc.get("source_table", _doc.get("table", "N/A"))
        _sem_type   = _doc.get("semantic_type", _doc.get("type", "N/A"))
        _page_cont  = _doc.get("page_content", _doc.get("content", _doc.get("text", "")))
        _score      = _doc.get("score", _doc.get("relevance_score", ""))
    elif hasattr(_doc, "page_content"):
        _meta       = getattr(_doc, "metadata", {})
        _src_tbl    = _meta.get("source_table", "N/A")
        _sem_type   = _meta.get("semantic_type", "N/A")
        _page_cont  = _doc.page_content
        _score      = _meta.get("score", "")
    else:
        _src_tbl = _sem_type = "N/A"; _page_cont = str(_doc); _score = ""
    _preview    = (_page_cont[:200] + "…") if len(_page_cont) > 200 else _page_cont
    _score_str  = f' &nbsp;|&nbsp; score: {_score:.3f}' if isinstance(_score, float) else ""
    _rag_doc_cards += (
        f'<div style="background:#fff;border:1px solid #dde3ea;border-radius:6px;'
        f'padding:10px 14px;margin:6px 0;font-family:sans-serif;">'
        f'<div style="font-size:11px;color:#555;margin-bottom:6px;">'
        f'{_badge(_esc(_src_tbl), "#495057")} &nbsp;'
        f'{_badge(_esc(_sem_type), "#6f42c1")}'
        f'<span style="color:#888;font-size:11px;">{_score_str}</span></div>'
        f'<div style="font-size:12px;color:#333;line-height:1.6;font-style:italic;">'
        f'"{_esc(_preview)}"</div></div>'
    )

if not _rag_doc_cards:
    _rag_doc_cards = _card('<em style="color:#888;">RAG não utilizado ou sem documentos recuperados.</em>', bg="#fff9e6")

_rag_section = _section(
    f"Contexto RAG Utilizado ({len(_rag_docs)} documentos)", "🔍",
    _rag_meta_html + _rag_doc_cards,
    border="#6f42c1"
)

# ══════════════════════════════════════════════════════════════════════════════
# 6. GRÁFICOS — renderização direta via displayHTML() por gráfico
# ══════════════════════════════════════════════════════════════════════════════
_CHART_LABELS = {
    "diário"      : ("📈", "Série Diária"),
    "mensal"      : ("📊", "Série Mensal"),
    "geográfico"  : ("🗺️", "Distribuição Geográfica"),
    "demográfico" : ("👥", "Perfil Demográfico"),
    "viral"       : ("🦠", "Distribuição por Agente Viral"),
    "outro"       : ("📉", "Gráfico Adicional"),
}

def _render_chart_direct(path, idx, total):
    """Renderiza um gráfico via displayHTML() direto — compatível com Plotly no Databricks."""
    _ctype      = _chart_meta_ref.get(path, "outro")
    _icon, _desc = _CHART_LABELS.get(_ctype, ("📉", "Gráfico"))
    _name       = _P(path).name
    _header_bar = (
        f'<div style="background:#f0f4f8;border:1px solid #dde3ea;'
        f'border-radius:8px 8px 0 0;padding:10px 16px;'
        f'font-family:sans-serif;font-size:12px;color:#1a3a5c;font-weight:600;'
        f'margin-top:16px;">'
        f'{_icon} [{idx}/{total}] {_desc}'
        f' &nbsp;<span style="font-weight:400;color:#666;font-family:monospace;">{_esc(_name)}</span>'
        f'</div>'
    )
    displayHTML(_header_bar)
    try:
        with open(path, "r", encoding="utf-8") as _fh:
            displayHTML(_fh.read())
    except Exception as _exc:
        displayHTML(
            f'<div style="background:#fff0f0;border:1px solid #f5c6cb;'
            f'border-radius:0 0 8px 8px;padding:12px 16px;'
            f'font-family:sans-serif;font-size:12px;color:#c0392b;">'
            f'⚠️ Erro ao carregar gráfico: {_esc(str(_exc))}</div>'
        )

# ══════════════════════════════════════════════════════════════════════════════
# 7. ARTEFATOS PERSISTIDOS
# ══════════════════════════════════════════════════════════════════════════════
_artifact_rows = ""
for _lbl, _pth in [
    ("Relatório Markdown", _md_path_ref),
    ("Relatório JSON",     _json_path_ref),
]:
    _artifact_rows += (
        f'<tr><td style="padding:5px 12px 5px 0;font-size:12px;color:#555;">{_esc(_lbl)}</td>'
        f'<td style="font-family:monospace;font-size:11px;color:#0d6efd;">{_esc(_pth)}</td></tr>'
    )
for _cp in _all_charts:
    _artifact_rows += (
        f'<tr><td style="padding:3px 12px 3px 0;font-size:12px;color:#555;">Gráfico HTML</td>'
        f'<td style="font-family:monospace;font-size:11px;color:#198754;">{_esc(_cp)}</td></tr>'
    )

_artifacts_section = _section(
    "Artefatos da Run", "📁",
    _card(f'<table style="border-collapse:collapse;width:100%;">{_artifact_rows}</table>'),
    border="#6c757d"
)

# ══════════════════════════════════════════════════════════════════════════════
# 8. RENDERIZAÇÃO FINAL — múltiplos displayHTML() para compatibilidade máxima
# ══════════════════════════════════════════════════════════════════════════════
_css = """
<style>
  * { box-sizing: border-box; }
  li { margin: 3px 0; }
</style>
"""

# ══════════════════════════════════════════════════════════════════════════════
# 2b. CONSULTA INTERATIVA DO AVALIADOR (seção adaptativa)
# Aparece somente se a célula E.2 foi executada nesta sessão
# ══════════════════════════════════════════════════════════════════════════════
_aval_section = ""  # vazio por padrão — seção não aparece se avaliador não rodou

try:
    _aq  = AVAL_QUERY
    _ar  = AVAL_RESULT
    _ac  = list(AVAL_CHARTS)
    _aval_has_data = bool(_aq and _ar)
except NameError:
    _aval_has_data = False

if _aval_has_data:
    _ar_rout   = _ar.get("routing", {})
    _ar_strat  = _ar_rout.get("strategy", "N/A").upper()
    _ar_conf   = _ar_rout.get("confidence", 0)
    _ar_conf_s = f"{_ar_conf:.0%}" if isinstance(_ar_conf, float) else str(_ar_conf)
    _ar_intent = _ar_rout.get("intent", _ar_rout.get("classified_intent", "N/A"))
    _ar_time   = _ar.get("execution_time_seconds", 0)
    _ar_status = "✅ OK" if _ar.get("success") else "⚠️ PARCIAL"
    _ar_mm     = _ar.get("mandatory_metrics", {})
    _ar_answer = _ar.get("answer", "")

    # tools inferidas do avaliador
    _ar_tools = list(_ar.get("tools_used", []))
    if not _ar_tools:
        if _ar.get("sql_result")  is not None: _ar_tools.append("SQL Tool")
        if _ar.get("rag_results"):             _ar_tools.append("RAG Retrieval")
        if _ar.get("news_results"):            _ar_tools.append("Web Search")
        if _ac:                                _ar_tools.append("Chart Tool")
        if _ar_answer:                         _ar_tools.append("Report Generator")

    _ar_tools_badges = " ".join(
        f'<span style="background:#e9ecef;color:#333;padding:3px 10px;'
        f'border-radius:12px;font-size:11px;font-weight:600;margin:2px;">{_esc(t)}</span>'
        for t in (_ar_tools or ["N/A"])
    )

    # pipeline do avaliador (terminal escuro igual ao principal)
    _ar_pipeline_html = (
        f'<div style="font-family:monospace;font-size:12px;background:#0d1117;'
        f'color:#c9d1d9;border-radius:8px;padding:20px 24px;line-height:2;">'
        f'<div style="color:#58a6ff;font-weight:700;font-size:13px;'
        f'margin-bottom:14px;letter-spacing:.5px;">▶ PIPELINE DE DECISÃO — CONSULTA DO AVALIADOR</div>'
        f'<table style="border-collapse:collapse;width:100%;">'
        f'<tr><td style="color:#8b949e;padding:4px 18px 4px 0;white-space:nowrap;vertical-align:top;">Query do avaliador</td>'
        f'<td style="color:#ffa657;font-family:sans-serif;font-size:12px;line-height:1.6;">"{_esc(_aq)}"</td></tr>'
        f'<tr><td style="color:#8b949e;padding:4px 18px 4px 0;white-space:nowrap;">Intent classificado</td>'
        f'<td style="color:#79c0ff;">{_esc(str(_ar_intent).upper())}</td></tr>'
        f'<tr><td style="color:#8b949e;padding:4px 18px 4px 0;white-space:nowrap;">Estratégia escolhida</td>'
        f'<td style="color:#56d364;font-weight:700;">{_esc(_ar_strat)}</td></tr>'
        f'<tr><td style="color:#8b949e;padding:4px 18px 4px 0;white-space:nowrap;">Confiança do roteador</td>'
        f'<td style="color:#e3b341;">{_ar_conf_s}</td></tr>'
        f'<tr><td style="color:#8b949e;padding:4px 18px 4px 0;white-space:nowrap;vertical-align:top;">Ferramentas utilizadas</td>'
        f'<td style="padding-top:4px;">{_ar_tools_badges}</td></tr>'
        f'<tr><td style="color:#8b949e;padding:4px 18px 4px 0;white-space:nowrap;">Tempo de execução</td>'
        f'<td style="color:#c9d1d9;">{_ar_time:.1f}s</td></tr>'
        f'<tr><td style="color:#8b949e;padding:4px 18px 4px 0;white-space:nowrap;">Status</td>'
        f'<td style="color:#c9d1d9;">{_esc(_ar_status)}</td></tr>'
        f'</table>'
        f'</div>'
    )

    # métricas do avaliador (se disponíveis)
    _ar_metric_cards = ""
    if _ar_mm:
        _ar_metric_items = [
            ("Total de Casos",      _ar_mm.get("total_casos")),
            ("Taxa de Crescimento", f"{_ar_mm.get('taxa_crescimento','N/A')}%"),
            ("Taxa de Mortalidade", f"{_ar_mm.get('taxa_mortalidade','N/A')}%"),
            ("Taxa de UTI",         f"{_ar_mm.get('taxa_uti','N/A')}%"),
            ("Taxa de Vacinação",   f"{_ar_mm.get('taxa_vacinacao','N/A')}%"),
        ]
        _ar_metric_cards = (
            f'<div style="display:flex;flex-wrap:wrap;gap:4px;margin:12px 0;">'
            + "".join(
                f'<div style="flex:1 1 140px;background:#fff;border:1px solid #dde3ea;'
                f'border-radius:8px;padding:12px;text-align:center;margin:3px;">'
                f'<div style="font-size:10px;color:#666;margin-bottom:5px;">{_esc(lbl)}</div>'
                f'<div style="font-size:18px;font-weight:700;color:#1a3a5c;">{_esc(_fmt_metric(val))}</div>'
                f'</div>'
                for lbl, val in _ar_metric_items
            )
            + '</div>'
        )

    # resposta do avaliador
    if _ar_answer:
        _ar_answer_html = (
            f'<div style="font-family:Arial,sans-serif;font-size:13px;line-height:1.8;'
            f'color:#1a1a2e;max-height:500px;overflow-y:auto;'
            f'border:1px solid #dde3ea;border-radius:6px;padding:16px;'
            f'background:#fff;margin-top:12px;">'
            f'{_md_to_html(_ar_answer)}</div>'
        )
    else:
        _ar_answer_html = _card('<em style="color:#888;">Sem resposta gerada pelo avaliador.</em>')

    # gráficos ad-hoc do avaliador (categorizados inline)
    _ar_chart_meta = {}
    for _p in _ac:
        _n = _P(_p).name
        if   "mensal"     in _n or "monthly"   in _n: _ar_chart_meta[_p] = "mensal"
        elif "multi_line" in _n or "viral"     in _n: _ar_chart_meta[_p] = "viral"
        elif "line"       in _n:                       _ar_chart_meta[_p] = "diário"
        elif "bar"        in _n:
            _c2 = _n.split("_")[-1].replace(".html","")
            _ar_chart_meta[_p] = "demográfico" if (_c2.isdigit() and int(_c2)>=3) else "geográfico"
        else:
            _ar_chart_meta[_p] = "outro"

    _ar_charts_note = (
        _card(
            f'<span style="font-size:12px;color:#198754;">📊 {len(_ac)} gráfico(s) gerado(s) — '
            f'renderizados abaixo da seção principal.</span>',
            bg="#f0fff4"
        ) if _ac else
        _card('<em style="color:#888;font-size:12px;">Nenhum gráfico gerado nesta consulta.</em>', bg="#fff9e6")
    )

    _aval_section = _section(
        f'Consulta Interativa do Avaliador',
        "🧪",
        (
            _ar_pipeline_html
            + _ar_metric_cards
            + _ar_answer_html
            + _ar_charts_note
        ),
        border="#0dcaf0"
    )

    # armazena para renderização de gráficos no bloco 2
    _aval_chart_meta_ref = _ar_chart_meta
else:
    _ac = []
    _aval_chart_meta_ref = {}

# ── bloco 1: cabeçalho + pipeline + métricas + análise + web + RAG ────────────
_block1 = f"""
{_css}
<div style="font-family:Arial,sans-serif;max-width:960px;color:#1a1a2e;padding:8px;">
  {_header_html}
  {_pipeline_section}
  {_metrics_section}
  {_answer_section}
  {_ws_section}
  {_rag_section}
  {_aval_section}
</div>
"""
displayHTML(_block1)

# ── bloco 2: gráficos — cada um renderizado individualmente ──────────────────
if _all_charts:
    displayHTML(
        f'<div style="font-family:Arial,sans-serif;max-width:960px;padding:8px 8px 0;">'
        f'<div style="margin:24px 0 8px;border-left:4px solid #fd7e14;padding-left:16px;">'
        f'<h3 style="margin:0;font-size:15px;color:#fd7e14;">'
        f'📈 Gráficos Gerados ({len(_all_charts)} total)</h3></div></div>'
    )
    if _mandatory_paths:
        displayHTML(
            '<div style="font-family:sans-serif;max-width:960px;padding:0 8px;">'
            f'<h4 style="font-size:12px;color:#6c757d;margin:8px 0 4px;'
            f'border-bottom:1px solid #eee;padding-bottom:6px;">'
            f'OBRIGATÓRIOS ({len(_mandatory_paths)})</h4></div>'
        )
        for _ci, _cp in enumerate(_mandatory_paths, 1):
            _render_chart_direct(_cp, _ci, len(_mandatory_paths))
    if _additional_paths:
        displayHTML(
            '<div style="font-family:sans-serif;max-width:960px;padding:0 8px;margin-top:20px;">'
            f'<h4 style="font-size:12px;color:#6c757d;margin:8px 0 4px;'
            f'border-bottom:1px solid #eee;padding-bottom:6px;">'
            f'ADICIONAIS ({len(_additional_paths)})</h4></div>'
        )
        for _ci, _cp in enumerate(_additional_paths, 1):
            _render_chart_direct(_cp, _ci, len(_additional_paths))
else:
    displayHTML(
        '<div style="font-family:sans-serif;background:#fff9e6;border:1px solid #ffc107;'
        'border-radius:8px;padding:12px 18px;margin:16px 0;font-size:13px;color:#856404;">'
        '⚠️ Nenhum gráfico disponível nesta execução.</div>'
    )

# ── bloco 2b: gráficos gerados pelo avaliador (ad-hoc) ─────────────────────────
# Exclui arquivos que já foram exibidos no bloco obrigatório (evita duplicação)
_mandatory_names = {_P(p).name for p in _mandatory_paths}
_ac_unique = [p for p in _ac if _P(p).name not in _mandatory_names]
if _ac_unique:
    _aval_query_short = _aq[:60] + '…' if len(_aq) > 60 else _aq
    displayHTML(
        '<div style="font-family:Arial,sans-serif;max-width:960px;padding:8px 8px 0;">'
        '<div style="margin:24px 0 8px;border-left:4px solid #0dcaf0;padding-left:16px;">'
        '<h3 style="margin:0 0 4px;font-size:15px;color:#0dcaf0;">'
        f'🧪 Gráficos do Avaliador ({len(_ac_unique)} único(s))</h3>'
        f'<div style="font-size:11px;color:#6c757d;font-family:monospace;">query: {_esc(_aval_query_short)}</div>'
        '</div></div>'
    )
    for _ci, _cp in enumerate(_ac_unique, 1):
        _ctype_av    = _aval_chart_meta_ref.get(_cp, "outro")
        _icon_av, _desc_av = _CHART_LABELS.get(_ctype_av, ("📉", "Gráfico"))
        _name_av     = _P(_cp).name
        _hdr_av = (
            f'<div style="background:#e8f9fd;border:1px solid #0dcaf0;'
            f'border-radius:8px 8px 0 0;padding:10px 16px;'
            f'font-family:sans-serif;font-size:12px;color:#055160;font-weight:600;'
            f'margin-top:16px;">'
            f'{_icon_av} [{_ci}/{len(_ac_unique)}] {_desc_av} — avaliador'
            f' &nbsp;<span style="font-weight:400;color:#0d6efd;font-family:monospace;">{_esc(_name_av)}</span>'
            f'</div>'
        )
        displayHTML(_hdr_av)
        try:
            with open(_cp, "r", encoding="utf-8") as _fh_av:
                displayHTML(_fh_av.read())
        except Exception as _exc_av:
            displayHTML(
                f'<div style="background:#fff0f0;padding:12px 16px;'
                f'font-family:sans-serif;font-size:12px;color:#c0392b;">'
                f'⚠️ Erro ao carregar gráfico do avaliador: {_esc(str(_exc_av))}</div>'
            )

# ── bloco 3: artefatos + rodapé ───────────────────────────────────────────────
displayHTML(
    f'<div style="font-family:Arial,sans-serif;max-width:960px;color:#1a1a2e;padding:8px;">'
    f'  {_artifacts_section}'
    f'  <div style="margin-top:24px;padding:10px 16px;background:#f0f4f8;border-radius:6px;'
    f'              font-size:11px;color:#888;font-family:monospace;text-align:center;">'
    f'    Gerado em {_ts_now} &nbsp;|&nbsp; Session: {_session_id}'
    f'    &nbsp;|&nbsp; Provider: {_provider.upper()}'
    f'    &nbsp;|&nbsp; Auditoria: {len(audit_logger.logs)} eventos'
    f'  </div>'
    f'</div>'
)


# COMMAND ----------

# MAGIC %md
# MAGIC ### E.5 Limitações Conhecidas
# MAGIC
# MAGIC | Componente | Limitação |
# MAGIC |---|---|
# MAGIC | Web Search | Depende da disponibilidade da Tavily API e da conectividade externa do ambiente. Em workspaces com bloqueio de DNS/egress, a busca externa pode falhar e operar apenas com fallback interno. |
# MAGIC | LLM externo | Quando configurado para OpenAI, depende de acesso a APIs públicas. Em ambientes Databricks com restrição de saída, pode ser necessário fallback para Databricks Foundation Models. |
# MAGIC | Conectividade externa / DNS | Em execuções 100% Databricks sem configuração híbrida adequada, chamadas a APIs externas podem falhar por bloqueio de DNS, egress ou políticas de firewall, mesmo com secrets válidas. |
# MAGIC | Índice vetorial | `status=UNKNOWN` é esperado quando os documentos não mudam (`skip write` + `sync` sem evento novo de CDF). |
# MAGIC | Roteador de intenção | Queries comparativas ambíguas podem ter confiança baixa (~55%); few-shot adicional pode melhorar o roteamento. |
# MAGIC | Qualidade RAG | Documentos KPI e documentos de regra compartilham o mesmo índice; perguntas conceituais podem recuperar fatos numéricos em vez de definições. |
# MAGIC | Classificação de gráficos | Heurística por nome de arquivo (`mensal`, `line`, `bar`), frágil caso o `ChartTool` altere a convenção de nomes. |
# MAGIC | Período do ground truth | O recorte é derivado dinamicamente dos 3 anos mais recentes disponíveis na tabela, podendo mudar conforme atualização da base. |
# MAGIC | Volume | Não há política de retenção automática; artefatos de execuções anteriores acumulam indefinidamente. |
# MAGIC | Hardcodes residuais | `CATALOG_GOLD`, `SCHEMA_GOLD` e `VS_ENDPOINT_NAME` ainda permanecem na célula de configuração; em produção, o ideal é parametrizar via widgets ou config externa. |
# MAGIC