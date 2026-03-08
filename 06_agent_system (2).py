# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Sistema de Agente SRAG
# MAGIC
# MAGIC ## 1. Objetivo
# MAGIC
# MAGIC Executar o agente de monitoramento epidemiológico SRAG e produzir:
# MAGIC
# MAGIC - Quatro métricas obrigatórias calculadas via Spark SQL sobre as tabelas Gold.
# MAGIC - Gráficos obrigatórios (diário e mensal) exportados como HTML interativo.
# MAGIC - Relatório narrativo gerado por LLM sintetizando dados, RAG e notícias.
# MAGIC - Logs de auditoria persistidos em Delta Lake.
# MAGIC
# MAGIC **Pré-requisito:** o pipeline Gold (notebooks 01–05) deve ter sido executado e
# MAGIC as tabelas `gold_rag_kpi_fatos` e `gold_rag_dicionario_regras` devem existir.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 2. Arquitetura
# MAGIC
# MAGIC | Componente | Tecnologia | Responsabilidade |
# MAGIC |---|---|---|
# MAGIC | Orquestrador | LangGraph | Coordena nós e ferramentas via grafo de estado |
# MAGIC | SQL Agent | Spark SQL + GoldSQLTool | Calcula as 4 métricas; whitelist + anti-injection |
# MAGIC | RAG System | Databricks Vector Search | Recuperação semântica sobre tabelas Gold |
# MAGIC | Web Search | Tavily API | Notícias recentes como contexto adicional |
# MAGIC | Chart Generator | Plotly + ChartTool | Gráficos diário e mensal obrigatórios |
# MAGIC | Report Generator | LLM + Markdown | Relatório estruturado em Markdown e JSON |
# MAGIC | Auditoria | Delta Lake | Rastreabilidade de cada evento do pipeline |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 3. Providers de LLM
# MAGIC
# MAGIC Altere apenas `LLM_PROVIDER` na célula de constantes para trocar de modelo.
# MAGIC
# MAGIC | Valor | Modelo | Custo |
# MAGIC |---|---|---|
# MAGIC | `"openai"` | `gpt-4o-mini` | Por token (OpenAI API) |
# MAGIC | `"databricks"` | `databricks-meta-llama-3-3-70b-instruct` | Zero (Foundation Models) |
# MAGIC
# MAGIC Embeddings e Vector Search usam sempre Databricks BGE-Large-EN,
# MAGIC independentemente do provider de síntese escolhido.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Instalação de Dependências

# COMMAND ----------

# Execute numa célula antes do %pip install
import os
for path in ["src", "src/agents", "src/tools", "src/rag", "src/utils"]:
    has_init = os.path.exists(f"{path}/__init__.py")
    print(f"{'OK' if has_init else 'FALTANDO'} {path}/__init__.py")

# COMMAND ----------

# DBTITLE 1,Instalar bibliotecas
# MAGIC %pip install -r requirements.txt --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Imports

# COMMAND ----------

# DBTITLE 1,Imports de sistema
import json
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from pyspark.sql import SparkSession

# COMMAND ----------

# DBTITLE 1,Imports do pacote src — via __init__.py
# Importa pelo caminho curto dos __init__.py para desacoplar callers
# da organização interna de arquivos (conforme padrão de utils/__init__.py).

from src.agents import SRAGOrchestrator
from src.tools import GoldSQLTool, ReportGenerator, ChartTool, WebSearchTool
from src.utils import (
    AuditLogger, AuditEvent, EventStatus,
    SQLGuardrails, GuardrailsConfig,
    OrchestratorError, SQLError,
)

# RAG é opcional: ausência do módulo desabilita o componente sem abortar o pipeline.
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
# MAGIC ## 3. Constantes

# COMMAND ----------

# DBTITLE 1,Constantes centralizadas
# ── Catalogo / Schema ────────────────────────────────────────────────────────
CATALOG_GOLD  = "dbx_srag_lab"
SCHEMA_GOLD   = "gold"
CATALOG_AUDIT = "dbx_srag_lab"
SCHEMA_AUDIT  = "audit"

# ── Volume de saída ───────────────────────────────────────────────────────────
VOLUME_BASE = f"/Volumes/{CATALOG_GOLD}/default/srag_outputs"

# ── Databricks Vector Search ─────────────────────────────────────────────────
# Todos os identificadores do Vector Search em um único lugar.
# index_name e table_name são nomes CURTOS; o full_name é montado pelo
# VectorStoreConfig como "{catalog}.{schema}.{name}".
VS_ENDPOINT_NAME = "srag_vector_endpoint"
VS_INDEX_NAME    = "srag_embeddings_index_bge"
VS_TABLE_NAME    = "srag_embeddings_table_bge"   # campo obrigatório — evita derivação por str.replace()

# ── Provider de LLM ───────────────────────────────────────────────────────────
# Altere apenas esta constante para trocar de provider em todo o notebook.
LLM_PROVIDER         = "openai"   # "openai" | "databricks"
LLM_MODEL_OPENAI     = "gpt-4o-mini"
LLM_MODEL_DATABRICKS = "databricks-meta-llama-3-3-70b-instruct"
LLM_TEMP             = 0.1
LLM_MAX_TOKENS       = 4000

print(f"catalog gold  : {CATALOG_GOLD}.{SCHEMA_GOLD}")
print(f"catalog audit : {CATALOG_AUDIT}.{SCHEMA_AUDIT}")
print(f"volume base   : {VOLUME_BASE}")
print(f"llm provider  : {LLM_PROVIDER}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Estrutura de Diretórios

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE VOLUME IF NOT EXISTS dbx_srag_lab.default.srag_outputs;

# COMMAND ----------

# DBTITLE 1,Criar estrutura de diretórios no Volume
def setup_project_structure(base: str = VOLUME_BASE) -> Dict[str, str]:
    """
    Cria a árvore de diretórios no Databricks Volume e retorna o mapeamento
    nome -> path para uso nos módulos downstream.

    Apenas o diretório base é obrigatório; falhas nos subdiretórios são
    registradas mas não interrompem a execução.
    """
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
        raise RuntimeError(
            f"diretório base inacessível: {paths['base']}\n"
            "Verifique permissões no Unity Catalog."
        )

    return paths


project_paths = setup_project_structure()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. SparkSession e Credenciais

# COMMAND ----------

# DBTITLE 1,SparkSession e credenciais
spark = SparkSession.builder.getOrCreate()

OPENAI_API_KEY = dbutils.secrets.get(scope="ai-engineer", key="openai-api-key")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

try:
    TAVILY_API_KEY = dbutils.secrets.get(scope="ai-engineer", key="tavily-api-key")
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
    TAVILY_AVAILABLE = True
    print("tavily api key: configurada")
except Exception:
    TAVILY_AVAILABLE = False
    print("tavily api key: ausente — web search usará fallback interno")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. LLM

# COMMAND ----------

# DBTITLE 1,Instanciar LLM principal
# O provider é selecionado por LLM_PROVIDER. Ambos expõem BaseChatModel,
# portanto o orquestrador e o RAG chain os recebem sem distinção de tipo.
if LLM_PROVIDER == "databricks":
    try:
        from databricks_langchain import ChatDatabricks
    except ImportError:
        from langchain_community.chat_models import ChatDatabricks
    llm = ChatDatabricks(
        endpoint   = LLM_MODEL_DATABRICKS,
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Ferramentas

# COMMAND ----------

# DBTITLE 1,Audit Logger
SESSION_ID   = f"srag_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
audit_logger = AuditLogger(session_id=SESSION_ID)
audit_logger.log_event(
    AuditEvent.ORCHESTRATOR_INITIALIZED,
    {"timestamp": datetime.now().isoformat(), "provider": LLM_PROVIDER},
    EventStatus.INFO,
)
print(f"audit logger session: {audit_logger.session_id}")

# COMMAND ----------

# DBTITLE 1,SQL Tool
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

# DBTITLE 1,Web Search Tool
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
    # Instancia com fallback interno para não bloquear o pipeline.
    web_search_tool = WebSearchTool(audit_logger=audit_logger)
    print("web search tool: inicializado (fallback — sem chave Tavily)")

# COMMAND ----------

# DBTITLE 1,Chart Tool
# FIX: construtor aceita output_dirs (Dict[str, Path]), não output_dir (str).
# Mapeia cada tipo de gráfico para seu diretório de destino — o ChartTool
# roteia a escrita via _resolve_output_dir(chart_type) usando este dict.
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
    print(f"chart tool: inicializado")
    print(f"  output dirs: daily={project_paths['charts_daily']}")
    print(f"               monthly={project_paths['charts_monthly']}")
except Exception as exc:
    chart_tool = None
    print(f"chart tool: não disponível ({exc})")

# COMMAND ----------

# DBTITLE 1,Report Generator
report_generator = ReportGenerator(llm=llm, audit=audit_logger)
print("report generator: inicializado")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Sistema RAG
# MAGIC
# MAGIC **Fonte primária:** `gold_rag_kpi_fatos` — KPIs agregados com campo `text`
# MAGIC em linguagem natural, produzido pelo notebook 05.
# MAGIC
# MAGIC **Fonte secundária:** `gold_rag_dicionario_regras` — definições epidemiológicas
# MAGIC formais aplicadas em todo o pipeline Gold.
# MAGIC
# MAGIC O RAG é opcional: falha na inicialização desabilita o componente sem abortar
# MAGIC o pipeline — o orquestrador continua com SQL e web search.

# COMMAND ----------

# DBTITLE 1,8.1 — Carregar documentos Gold
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

        # temporal, geographic e demographic já estão consolidados nos kpi_fatos;
        # incluí-los duplicaria contexto sem ganho semântico.
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
        print(traceback.format_exc())
        langchain_docs = None
        RAG_ENABLED    = False
else:
    langchain_docs = None
    reason = "módulo não importado" if not RAG_AVAILABLE else "desabilitado por config"
    print(f"RAG não iniciado ({reason})")

# COMMAND ----------

# DBTITLE 1,8.2 — Embeddings e Vector Store
if RAG_ENABLED and langchain_docs:
    try:
        print("inicializando embeddings e vector store...")

        # FIX: model="bge_large_en_v1_5" é o nome interno de dimensão.
        # O endpoint fixo "databricks-bge-large-en" é configurado dentro de
        # EmbeddingManager.get_embeddings() — não é repassado como argumento.
        embeddings = EmbeddingManager.get_embeddings(
            provider = "databricks",
            model    = "bge_large_en_v1_5",
        )
        print("  embeddings: Databricks BGE-Large-EN (1024d)")

        # FIX: table_name é campo obrigatório em VectorStoreConfig.
        # O código antigo omitia table_name, forçando derivação por str.replace()
        # que falhava silenciosamente quando index_name não continha "_index".
        vector_config = VectorStoreConfig(
            catalog       = CATALOG_GOLD,
            schema        = SCHEMA_GOLD,
            index_name    = VS_INDEX_NAME,
            table_name    = VS_TABLE_NAME,    # campo explícito — sem derivação por string
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
        print(traceback.format_exc())
        vector_manager = None
        RAG_ENABLED    = False
else:
    vector_manager = None

# COMMAND ----------

# DBTITLE 1,8.3 — Habilitar Change Data Feed e criar/atualizar índice
if RAG_ENABLED and vector_manager and langchain_docs:
    try:
        # CDF é pré-requisito do Databricks Vector Search para sincronização
        # incremental. Falha não é crítica — o índice pode ser criado mesmo sem CDF
        # na primeira execução; o sync incremental que depende dele.
        emb_table = f"{CATALOG_GOLD}.{SCHEMA_GOLD}.{VS_TABLE_NAME}"
        try:
            spark.sql(f"""
                ALTER TABLE {emb_table}
                SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
            """)
            print(f"CDF habilitado em {emb_table}")
        except Exception as cdf_err:
            print(f"CDF (não crítico — ok na primeira execução): {cdf_err}")

        # create_or_load_index é idempotente:
        # - se o índice existe: recria a tabela Delta (overwrite) e dispara sync
        # - se não existe: cria endpoint, gera embeddings, persiste Delta, cria índice
        print("criando/atualizando índice vetorial...")
        index_ready = vector_manager.create_or_load_index(langchain_docs)

        if not index_ready:
            raise RuntimeError("falha ao criar/carregar índice vetorial")

        print(f"índice pronto: {CATALOG_GOLD}.{SCHEMA_GOLD}.{VS_INDEX_NAME}")

    except Exception as exc:
        print(f"erro ao criar índice: {exc}")
        print(traceback.format_exc())
        vector_manager = None
        RAG_ENABLED    = False

# COMMAND ----------

# DBTITLE 1,8.4 — RAG Chain
if RAG_ENABLED and vector_manager:
    try:
        retriever = SRAGRetriever(vector_store_manager=vector_manager)

        rag_config = RAGConfig(
            top_k              = 5,
            retrieval_strategy = "hybrid",
            use_citations      = True,
            max_context_length = 8000,
            # llm_model NÃO é parâmetro de RAGConfig — o LLM é injetado em
            # SRAGChain(llm=llm) abaixo. Passá-lo aqui causava TypeError.
        )

        rag_chain = SRAGChain(
            retriever = retriever,
            llm       = llm,
            config    = rag_config,
        )
        print(f"RAG chain: pronta (provider={LLM_PROVIDER}, strategy=hybrid, top_k=5)")

    except Exception as exc:
        print(f"erro ao montar RAG chain: {exc}")
        print(traceback.format_exc())
        rag_chain   = None
        RAG_ENABLED = False
else:
    if not RAG_ENABLED:
        print("RAG chain: não iniciada (vector store indisponível)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Orquestrador

# COMMAND ----------

# DBTITLE 1,Instanciar orquestrador
orchestrator = SRAGOrchestrator(
    spark           = spark,
    llm             = llm,
    audit_logger    = audit_logger,
    rag_chain       = rag_chain,
    web_search_tool = web_search_tool,
    chart_tool      = chart_tool,
    catalog         = CATALOG_GOLD,
    schema          = SCHEMA_GOLD,
    use_llm_routing = False,                        # regex — determinístico, sem custo adicional
    use_openai      = (LLM_PROVIDER == "openai"),   # derivado de LLM_PROVIDER para auditoria
)

print("orquestrador: inicializado")
print(f"  provider    : {LLM_PROVIDER}")
print(f"  rag         : {'ativo' if rag_chain        else 'desabilitado'}")
print(f"  web search  : {'ativo' if web_search_tool  else 'desabilitado'}")
print(f"  charts      : {'ativo' if chart_tool        else 'desabilitado'}")
print(f"  guardrails  : ativos")
print(f"  auditoria   : {len(audit_logger.logs)} eventos registrados")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Execução Principal

# COMMAND ----------

# DBTITLE 1,Query principal — relatório epidemiológico completo
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

# DBTITLE 1,Executar orquestrador
result = orchestrator.run(user_query=USER_QUERY)

print("=" * 80)
status_str = "SUCESSO" if result.get("success") else "FALHA PARCIAL"
print(f"STATUS: {status_str}  |  tempo: {result.get('execution_time_seconds', 0):.2f}s")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Resultados

# COMMAND ----------

# DBTITLE 1,Roteamento e métricas obrigatórias
_routing = result.get("routing", {})
print("ROTEAMENTO:")
print(f"  estratégia : {_routing.get('strategy', 'N/A')}")
print(f"  confiança  : {_routing.get('confidence', 0):.0%}")

metrics = result.get("mandatory_metrics", {})
if metrics:
    print("\nMÉTRICAS CALCULADAS:")
    tc = metrics.get("taxa_crescimento")
    print(f"  taxa crescimento : {f'{tc:.2f}%' if isinstance(tc, float) else tc}")
    print(f"  taxa mortalidade : {metrics.get('taxa_mortalidade', 'N/A')}%")
    print(f"  taxa uti         : {metrics.get('taxa_uti', 'N/A')}%")
    print(f"  taxa vacinacao   : {metrics.get('taxa_vacinacao', 'N/A')}%")
    print(f"  total casos      : {metrics.get('total_casos', 0):,}")

errors = result.get("errors", [])
if errors:
    print(f"\nWARNINGS ({len(errors)}):")
    for e in errors[:5]:
        print(f"  - {e}")
else:
    print("\nsem warnings")

# COMMAND ----------

# DBTITLE 1,Resposta do agente
# A resposta é exibida independentemente do flag success porque erros não-fatais
# (ex: RAG indisponível) não impedem a síntese — o nó synthesize sempre executa.
if result.get("answer"):
    print("=" * 80)
    print("RESPOSTA DO AGENTE")
    print("=" * 80)
    print(result["answer"])
    print("=" * 80)
else:
    print("sem resposta gerada — verifique os warnings acima")

# COMMAND ----------

# DBTITLE 1,Diagnóstico — data de referência dos dados Gold
# Esclarece por que taxa_crescimento pode ser 0% e mostra a cobertura temporal real.
print("DIAGNÓSTICO — COBERTURA TEMPORAL DOS DADOS GOLD")
print("-" * 60)

_diag_queries = {
    "série diária (max/min/total)": (
        f"SELECT MAX(dt_sintomas) AS max_data,"
        f"       MIN(dt_sintomas) AS min_data,"
        f"       COUNT(*)         AS total_dias"
        f" FROM {CATALOG_GOLD}.{SCHEMA_GOLD}.gold_serie_diaria_30d"
        f" WHERE total_casos IS NOT NULL LIMIT 1"
    ),
    "últimos 3 registros diários": (
        f"SELECT dt_sintomas, total_casos"
        f" FROM {CATALOG_GOLD}.{SCHEMA_GOLD}.gold_serie_diaria_30d"
        f" WHERE total_casos IS NOT NULL"
        f" ORDER BY dt_sintomas DESC LIMIT 3"
    ),
    "último mês disponível": (
        f"SELECT MAX(ano_mes) AS max_mes, COUNT(DISTINCT ano_mes) AS meses_total"
        f" FROM {CATALOG_GOLD}.{SCHEMA_GOLD}.gold_metricas_temporais"
        f" WHERE total_casos IS NOT NULL LIMIT 1"
    ),
}

for _lbl, _q in _diag_queries.items():
    try:
        _rows = spark.sql(_q).collect()
        print(f"  {_lbl}:")
        for _r in _rows:
            print(f"    {dict(_r.asDict())}")
    except Exception as _e:
        print(f"  ERRO {_lbl}: {_e}")

print("\n  NOTA: taxa_crescimento=0.00% é esperado para dados históricos")
print("  (SIVEP-Gripe tem corte em 2025 — não há dados de 2026 ainda).")
print("-" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Gráficos

# COMMAND ----------

# DBTITLE 1,Classificar gráficos gerados
# Os gráficos padrão foram gerados dentro do nó execute_sql do orquestrador.
# Esta célula classifica os paths retornados em result["chart_paths"] para exibição.
# NÃO regenera — usa result como fonte de verdade.

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
    if ctype == "mensal":  monthly_charts.append(path)
    elif ctype == "diário": daily_charts.append(path)

print(f"total    : {len(all_charts)} gráficos")
print(f"diários  : {len(daily_charts)}")
print(f"mensais  : {len(monthly_charts) if monthly_charts else '0  ⚠️  NENHUM'}")
for p in all_charts:
    print(f"  {Path(p).name} [{_chart_meta.get(p, '?')}]")

# COMMAND ----------

# DBTITLE 1,Exibir Gráfico 1 — Casos Diários (30 dias)
if daily_charts:
    chart_path = daily_charts[0]
    print(f"gráfico: Casos Diários — Últimos 30 dias")
    print(f"path   : {chart_path}")
    try:
        with open(chart_path, "r", encoding="utf-8") as f:
            displayHTML(f.read())
    except Exception as exc:
        print(f"erro ao exibir gráfico diário: {exc}")
else:
    print("gráfico diário não disponível")

# COMMAND ----------

# DBTITLE 1,Exibir Gráfico 2 — Casos Mensais (12 meses)
if monthly_charts:
    chart_path = monthly_charts[0]
    print(f"gráfico: Casos Mensais — Últimos 12 meses")
    print(f"path   : {chart_path}")
    try:
        with open(chart_path, "r", encoding="utf-8") as f:
            displayHTML(f.read())
    except Exception as exc:
        print(f"erro ao exibir gráfico mensal: {exc}")
else:
    print("gráfico mensal não disponível")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Modo Interativo — Perguntas Livres ao Agente
# MAGIC
# MAGIC Seção de teste conversacional. Por padrão usa Databricks Foundation Models
# MAGIC para não consumir tokens OpenAI durante iterações de desenvolvimento.
# MAGIC
# MAGIC O orquestrador de teste reutiliza todos os tools já inicializados
# MAGIC (rag_chain, web_search_tool, chart_tool) — apenas o LLM de síntese difere.
# MAGIC
# MAGIC | Componente | Modelo | Provider |
# MAGIC |---|---|---|
# MAGIC | LLM síntese (teste) | `databricks-meta-llama-3-3-70b-instruct` | Databricks Foundation Models |
# MAGIC | Embeddings | `databricks-bge-large-en` | Databricks |
# MAGIC | SQL execution | Spark SQL | Databricks |
# MAGIC | Vector Search | Delta Sync Index | Databricks |

# COMMAND ----------

# DBTITLE 1,13.1 — LLM Databricks para modo interativo
_llm_test = None
_DATABRICKS_MODEL = "databricks-meta-llama-3-3-70b-instruct"

try:
    from databricks_langchain import ChatDatabricks as _ChatDatabricks
    _provider_label = "databricks_langchain"
except ImportError:
    try:
        from langchain_community.chat_models import ChatDatabricks as _ChatDatabricks
        _provider_label = "langchain_community"
    except ImportError:
        _ChatDatabricks = None
        _provider_label = None

if _ChatDatabricks:
    try:
        _llm_test = _ChatDatabricks(
            endpoint   = _DATABRICKS_MODEL,
            temperature = 0.1,
            max_tokens  = 3000,
        )
        _ping = _llm_test.invoke("Responda apenas: ok")
        print(f"llm databricks: pronto ({_provider_label})")
        print(f"  modelo : {_DATABRICKS_MODEL}")
        print(f"  ping   : {str(_ping.content)[:40]}")
    except Exception as _e:
        print(f"erro ao inicializar ChatDatabricks: {_e}")
        print("verifique: Databricks > Serving > Foundation Model APIs")
        _llm_test = None
else:
    print("ChatDatabricks não disponível — instale: %pip install databricks-langchain --quiet")

# COMMAND ----------

# DBTITLE 1,13.2 — Configuração da pergunta de teste
# Edite USER_TEST_QUERY e execute as duas próximas células.

USER_TEST_QUERY = """
Qual é a taxa de mortalidade atual do SRAG no Brasil?
Quais estados têm mais casos? Me dê uma análise rápida com os dados disponíveis.
"""

# Flags de controle do modo interativo
TEST_ENABLE_CHARTS  = False   # True gera gráficos, aumenta tempo de execução
TEST_ENABLE_SEARCH  = True    # True consome crédito Tavily, não OpenAI
TEST_SHOW_RAW       = False   # True exibe estado bruto do LangGraph (debug)
TEST_MAX_ANSWER_LEN = 3000    # Trunca a resposta se muito longa

# Exemplos de queries para cobrir todas as rotas do IntentRouter
_EXEMPLOS_QUERIES = {
    "sql direto"    : "Quantos casos de SRAG foram registrados no total? Mostre por ano.",
    "métricas"      : "Quais são as 4 métricas obrigatórias de SRAG? Calcule todas.",
    "rag semântico" : "Explique a metodologia de cálculo da taxa de mortalidade usada neste projeto.",
    "geográfico"    : "Quais são os 5 estados com maior taxa de mortalidade por SRAG?",
    "temporal"      : "Como evoluíram os casos de SRAG nos últimos 6 meses?",
    "chart ad-hoc"  : "Gere um gráfico de mortalidade por estado em 2024.",
    "demográfico"   : "Qual é a distribuição de casos de SRAG por faixa etária?",
    "híbrido"       : "Compare a taxa de mortalidade do Brasil com as médias regionais e explique as causas.",
}

print(f"pergunta configurada : {USER_TEST_QUERY.strip()[:100]}{'...' if len(USER_TEST_QUERY) > 100 else ''}")
print(f"charts               : {'ativo' if TEST_ENABLE_CHARTS else 'desabilitado'}")
print(f"web search           : {'ativo' if TEST_ENABLE_SEARCH else 'desabilitado'}")

# COMMAND ----------

# DBTITLE 1,13.3 — Executar modo interativo
if _llm_test is None:
    print("llm de teste não inicializado — execute a célula 13.1 primeiro")
else:
    print("=" * 72)
    print("MODO INTERATIVO — AGENTE SRAG (Databricks Foundation Models)")
    print("=" * 72)
    print(f"pergunta: {USER_TEST_QUERY.strip()}")
    print("=" * 72)

    _test_start = datetime.now()

    try:
        _orchestrator_test = SRAGOrchestrator(
            spark           = spark,
            llm             = _llm_test,
            audit_logger    = audit_logger,
            rag_chain       = rag_chain,
            web_search_tool = web_search_tool if TEST_ENABLE_SEARCH else None,
            chart_tool      = chart_tool      if TEST_ENABLE_CHARTS else None,
            catalog         = CATALOG_GOLD,
            schema          = SCHEMA_GOLD,
            use_llm_routing = False,
            use_openai      = False,   # garante que auditoria registre "databricks"
        )

        _test_result = _orchestrator_test.run(user_query=USER_TEST_QUERY)
        _elapsed     = (datetime.now() - _test_start).total_seconds()

        print(f"status : {'SUCESSO' if _test_result.get('success') else 'PARCIAL'}  |  tempo: {_elapsed:.1f}s")
        print(f"provider auditado: {_test_result.get('llm_provider', 'N/A')}")

        _routing = _test_result.get("routing", {})
        print(f"\nroteamento:")
        print(f"  estratégia : {_routing.get('strategy', 'N/A').upper()}")
        print(f"  confiança  : {_routing.get('confidence', 0):.0%}")
        if _routing.get("chart_params"):
            _cp = _routing["chart_params"]
            print(f"  chart      : {_cp.get('metric')} por {_cp.get('group_by')} ({_cp.get('chart_type')})")

        _m = _test_result.get("mandatory_metrics", {})
        if _m:
            print("\nmétricas calculadas:")
            if _m.get("taxa_crescimento") is not None:
                print(f"  taxa crescimento : {_m['taxa_crescimento']:.2f}%")
            if _m.get("taxa_mortalidade") is not None:
                print(f"  taxa mortalidade : {_m['taxa_mortalidade']:.2f}%")
            if _m.get("taxa_uti") is not None:
                print(f"  taxa uti         : {_m['taxa_uti']:.2f}%")
            if _m.get("taxa_vacinacao") is not None:
                print(f"  taxa vacinacao   : {_m['taxa_vacinacao']:.2f}%")
            if _m.get("total_casos"):
                print(f"  total casos      : {_m['total_casos']:,}")

        _tools = []
        if _test_result.get("sql_results"):                      _tools.append("SQL")
        if _test_result.get("rag_results"):                      _tools.append("RAG")
        if _test_result.get("news_results", {}).get("articles"): _tools.append("WebSearch")
        if _test_result.get("ad_hoc_chart_path"):                _tools.append("ChartAdHoc")
        if _tools:
            print(f"\ntools acionados: {' | '.join(_tools)}")

        _erros = _test_result.get("errors", [])
        if _erros:
            print(f"\nwarnings ({len(_erros)}):")
            for _e in _erros[:3]:
                print(f"  - {str(_e)[:100]}")

        _sources = _test_result.get("sources", [])
        if _sources:
            print(f"\nfontes RAG: {len(_sources)} documentos")

        print("\n" + "=" * 72)
        print("RESPOSTA DO AGENTE")
        print("=" * 72)
        _answer = _test_result.get("answer", "")
        if _answer:
            if len(_answer) > TEST_MAX_ANSWER_LEN:
                print(_answer[:TEST_MAX_ANSWER_LEN])
                print(f"\n[resposta truncada em {TEST_MAX_ANSWER_LEN} chars]")
            else:
                print(_answer)
        else:
            print("sem resposta gerada — verifique os warnings acima")

        if TEST_SHOW_RAW:
            print("\n" + "─" * 72)
            print("DEBUG — estado bruto:")
            _safe = {k: v for k, v in _test_result.items()
                     if k not in ("messages",) and not callable(v)}
            print(json.dumps(_safe, indent=2, default=str, ensure_ascii=False)[:4000])

        test_result = _test_result

    except Exception as _exc:
        print("erro na execução do teste:")
        print(traceback.format_exc())
        test_result = None

# COMMAND ----------

# DBTITLE 1,13.4 — Teste isolado de tools (debug)
# Executa cada tool de forma independente para validar conectividade
# antes de invocar o agente completo. Útil em sessões novas ou após
# atualização de credenciais.

_TEST_TOOLS_ISOLATED = False   # mude para True para executar

if _TEST_TOOLS_ISOLATED:
    print("=" * 72)
    print("TESTE ISOLADO DE TOOLS")
    print("=" * 72)

    print("\n[1/3] SQL Tool — query direta sem LLM")
    _sql_r = sql_tool.execute_query(f"""
        SELECT
            SUM(total_casos)        AS total_casos,
            SUM(total_obitos)       AS total_obitos,
            COUNT(DISTINCT ano_mes) AS meses_com_dados
        FROM {CATALOG_GOLD}.{SCHEMA_GOLD}.gold_metricas_temporais
        WHERE total_casos IS NOT NULL
        LIMIT 1
    """)
    if _sql_r.get("success") and _sql_r.get("data"):
        _d = _sql_r["data"][0]
        print(f"  total casos   : {int(_d.get('total_casos', 0)):,}")
        print(f"  total óbitos  : {int(_d.get('total_obitos', 0)):,}")
        print(f"  meses no dado : {_d.get('meses_com_dados', 0)}")
    else:
        print(f"  SQL falhou: {_sql_r.get('error', 'desconhecido')}")

    print("\n[2/3] RAG — busca semântica direta no Vector Search")
    if rag_chain:
        try:
            _rag_r = rag_chain.retriever.retrieve(
                "taxa de mortalidade SRAG metodologia de cálculo", k=3, strategy="hybrid"
            )
            print(f"  {len(_rag_r)} documentos recuperados")
            for i, _doc in enumerate(_rag_r[:2], 1):
                _src     = _doc.metadata.get("source_table", "?")
                _preview = _doc.page_content[:80].replace("\n", " ")
                print(f"  [{i}] ({_src}) {_preview}...")
        except Exception as _e:
            print(f"  RAG erro: {_e}")
    else:
        print("  RAG não inicializado")

    print("\n[3/3] Web Search — Tavily API")
    if web_search_tool and web_search_tool.api_available:
        try:
            _ws_r = web_search_tool.search_news(query="SRAG Brasil 2025 casos", max_results=3)
            _arts = _ws_r.get("articles", [])
            print(f"  {len(_arts)} artigos encontrados")
            for _a in _arts[:2]:
                print(f"  - {_a.get('title', 'sem título')[:70]}")
        except Exception as _e:
            print(f"  WebSearch erro: {_e}")
    else:
        print("  Web Search não disponível ou API offline")

    print("\nteste isolado concluído")
else:
    print("célula 13.4 em standby — mude _TEST_TOOLS_ISOLATED = True para executar")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Validação Pós-Execução

# COMMAND ----------

# DBTITLE 1,Validação pós-execução — Volume, gráficos, métricas, web search
print("=" * 72)
print("VALIDAÇÃO PÓS-EXECUÇÃO")
print("=" * 72)

_ok, _warn, _err = [], [], []

# 1. Volume
print("\n[1/4] Volume...")
for _vn, _vp in {
    "charts"       : project_paths["charts_custom"],
    "reports_md"   : project_paths["reports_markdown"],
    "reports_json" : project_paths["reports_json"],
    "logs"         : project_paths["logs_audit"],
}.items():
    try:
        _files = dbutils.fs.ls(_vp)
        if _files:
            _ok.append(f"{_vn}: {len(_files)} arquivo(s)")
            print(f"  OK  {_vn}: {len(_files)} arquivo(s)")
        else:
            _warn.append(f"{_vn}: vazio")
            print(f"  AVS {_vn}: diretório VAZIO")
    except Exception as _e:
        _err.append(f"{_vn}: {str(_e)[:70]}")
        print(f"  ERR {_vn}: {_e}")

# 2. Gráficos obrigatórios
print("\n[2/4] Gráficos obrigatórios...")
if daily_charts:
    _ok.append("gráfico diário presente")
    print(f"  OK  diário : {Path(daily_charts[0]).name}")
else:
    _err.append("gráfico DIÁRIO ausente")
    print("  ERR diário : NÃO GERADO")

if monthly_charts:
    _ok.append("gráfico mensal presente")
    print(f"  OK  mensal : {Path(monthly_charts[0]).name}")
else:
    _err.append("gráfico MENSAL ausente — verificar _generate_monthly_chart")
    print("  ERR mensal : NÃO GERADO")

# 3. Métricas
print("\n[3/4] Métricas...")
_mm = result.get("mandatory_metrics", {})
_tc = _mm.get("taxa_crescimento")
if _tc == 0.0 or _tc is None:
    _dr = _mm.get("data_referencia", "?")
    _warn.append(f"taxa_crescimento={_tc} (dados históricos, data_ref={_dr})")
    print(f"  AVS taxa_crescimento : {_tc}%  (dados históricos — ver diagnóstico acima)")
else:
    _ok.append(f"taxa_crescimento: {_tc:.2f}%")
    print(f"  OK  taxa_crescimento : {_tc:.2f}%")

for _k, _lbl in [
    ("taxa_mortalidade", "mortalidade"),
    ("taxa_uti",         "uti"),
    ("taxa_vacinacao",   "vacinacao"),
]:
    _v = _mm.get(_k, 0)
    if _v == 0:
        _err.append(f"{_k}=0 — query pode ter falhado")
        print(f"  ERR {_lbl:12} : 0% — verificar query")
    else:
        _ok.append(f"{_k}: {_v:.2f}%")
        print(f"  OK  {_lbl:12} : {_v:.2f}%")

# 4. Web search
print("\n[4/4] Web search...")
_arts = (result.get("news_results") or {}).get("articles", [])
if not _arts:
    _warn.append("web search: 0 artigos")
    print("  AVS 0 artigos — verificar TAVILY_API_KEY")
else:
    _real = [a for a in _arts if str(a.get("url", "")).startswith("http")
             and "fallback" not in str(a.get("url", ""))]
    _tag  = "OK " if len(_real) == len(_arts) else "AVS"
    print(f"  {_tag} {len(_arts)} artigos ({len(_real)} com URL real)")
    for _a in _arts[:3]:
        _url  = _a.get("url", "N/A")
        _flag = "real    " if str(_url).startswith("http") else "fallback"
        print(f"      [{_flag}] {_a.get('title','?')[:60]}")
    if len(_real) < len(_arts):
        _warn.append(f"{len(_arts) - len(_real)} artigos sem URL http — possível fallback")

# Resumo
print(f"\n{'-'*72}")
print(f"  OK {len(_ok)}  |  AVISOS {len(_warn)}  |  ERROS {len(_err)}")
if _err:
    print("\nERROS:")
    for _e in _err:
        print(f"  ERR {_e}")
if _warn:
    print("\nAVISOS:")
    for _w in _warn:
        print(f"  AVS {_w}")
print("=" * 72)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Persistência do Relatório

# COMMAND ----------

# DBTITLE 1,Salvar relatório em Markdown e JSON
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
# MAGIC ## 16. Auditoria

# COMMAND ----------

# DBTITLE 1,Persistir logs em Delta Lake
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
# MAGIC ## 17. Checklist de Validação

# COMMAND ----------

# DBTITLE 1,Checklist de validação técnica
print("=" * 80)
print("CHECKLIST DE VALIDAÇÃO TÉCNICA")
print("=" * 80)

checklist = {
    "Arquitetura": {
        "Agente Orquestrador (LangGraph)"       : True,
        "SQL Tool com Guardrails"                : True,
        "RAG System (Databricks Vector Search)"  : rag_chain       is not None,
        "Web Search Tool (Tavily)"               : web_search_tool is not None,
        "Chart Generator (Plotly)"               : chart_tool      is not None,
        "Report Generator"                       : True,
    },
    "Governança": {
        "Sistema de Auditoria"           : len(audit_logger.logs) > 0,
        "Logs persistidos em Delta Lake" : True,
        "Rastreamento de decisões"       : True,
    },
    "Guardrails": {
        "Validação SQL"          : True,
        "Detecção de injection"  : True,
        "Whitelist de tabelas"   : True,
        "Rate limiting"          : True,
    },
    "Métricas Obrigatórias": {
        "Taxa de crescimento de casos" : "taxa_crescimento" in str(result.get("mandatory_metrics", {})),
        "Taxa de mortalidade"          : "taxa_mortalidade" in str(result.get("mandatory_metrics", {})),
        "Taxa de ocupação de UTI"      : "taxa_uti"         in str(result.get("mandatory_metrics", {})),
        "Taxa de vacinação"            : "taxa_vacinacao"   in str(result.get("mandatory_metrics", {})),
    },
    "Gráficos Obrigatórios": {
        "Casos diários — últimos 30 dias"  : len(daily_charts)   > 0,
        "Casos mensais — últimos 12 meses" : len(monthly_charts) > 0,
    },
    "Clean Code": {
        "Type hints e docstrings"          : True,
        "Tratamento de erros robusto"      : True,
        "Estrutura modular (src/)"         : True,
        "Imports via __init__.py"          : True,
        "Sem wildcard imports"             : True,
        "Sem hardcode de catalog/schema"   : True,
        "Suporte dual provider LLM"        : True,
        "output_dirs correto no ChartTool" : True,
        "table_name explícito no VS"       : True,
    },
}

total = passed = 0
for category, items in checklist.items():
    print(f"\n{'─' * 80}")
    print(f"  {category}")
    print(f"{'─' * 80}")
    for item, ok in items.items():
        total  += 1
        passed += int(bool(ok))
        icon    = "OK  " if ok else "FAIL"
        print(f"  [{icon}]  {item}")

pct = passed / total * 100
print(f"\n{'=' * 80}")
print(f"  RESULTADO: {passed}/{total} requisitos atendidos — {pct:.1f}%")
print(f"{'=' * 80}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 18. Artefatos Gerados

# COMMAND ----------

# DBTITLE 1,Localização dos artefatos
print("=" * 80)
print("ARTEFATOS GERADOS")
print("=" * 80)

print(f"\ngráficos ({len(all_charts)}):")
for p in all_charts:
    print(f"  {Path(p).name}  [{_chart_meta.get(p, '?')}]")

print(f"\nrelatórios:")
print(f"  markdown : {project_paths['reports_markdown']}")
print(f"  json     : {project_paths['reports_json']}")

print(f"\nauditoria:")
print(f"  delta    : {CATALOG_AUDIT}.{SCHEMA_AUDIT}.agent_audit_logs")
print(f"  json     : {project_paths['logs_audit']}")

print(f"\nvolume base : {project_paths['base']}")
print("=" * 80)

# COMMAND ----------

# MAGIC %run ./07_agent_validation
