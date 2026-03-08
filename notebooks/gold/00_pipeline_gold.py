# Databricks notebook source
# MAGIC %md
# MAGIC # Gold — Pipeline Orquestrador
# MAGIC
# MAGIC ## 1. Objetivo
# MAGIC
# MAGIC Executar o pipeline completo da camada Gold em ordem correta e com tratamento
# MAGIC de falhas. Este notebook é o único ponto de entrada para uma execução completa.
# MAGIC
# MAGIC ## 2. Ordem de Execução
# MAGIC
# MAGIC ```
# MAGIC 01_gold_setup                    (obrigatório primeiro — registra widgets e valida Silver)
# MAGIC        │
# MAGIC        ├─── 02_gold_metricas_temporais    ─┐
# MAGIC        ├─── 03_gold_metricas_geograficas   ├─ paralelo
# MAGIC        └─── 04_gold_metricas_demograficas ─┘
# MAGIC                           │
# MAGIC              05_gold_base_conhecimento_rag  (deve ser o último)
# MAGIC ```
# MAGIC
# MAGIC Os notebooks 02, 03 e 04 são independentes entre si e executam em paralelo
# MAGIC via threads Python. O notebook 05 aguarda a conclusão dos três antes de iniciar.
# MAGIC
# MAGIC ## 3. Parâmetros
# MAGIC
# MAGIC Todos os parâmetros abaixo podem ser sobrescritos via widgets da UI do Databricks
# MAGIC ou via `dbutils.notebook.run(..., arguments={...})` em chamadas externas (ex: Jobs).
# MAGIC
# MAGIC | Widget | Padrão | Descrição |
# MAGIC |---|---|---|
# MAGIC | `catalog_silver` | `dbx_srag_lab` | Catálogo de leitura (Silver) |
# MAGIC | `schema_silver` | `silver` | Schema da Silver |
# MAGIC | `table_silver_name` | `silver_srag_clean` | Nome da tabela Silver |
# MAGIC | `catalog_gold` | `dbx_srag_lab` | Catálogo de escrita (Gold) |
# MAGIC | `schema_gold` | `gold` | Schema Gold de destino |
# MAGIC | `timeout_setup` | `600` | Timeout do setup em segundos |
# MAGIC | `timeout_analitico` | `3600` | Timeout por notebook analítico (02/03/04) |
# MAGIC | `timeout_rag` | `1800` | Timeout do notebook RAG (05) |
# MAGIC
# MAGIC ## 4. Saída
# MAGIC
# MAGIC Em caso de sucesso, todas as tabelas abaixo estarão atualizadas:
# MAGIC
# MAGIC | Tabela | Notebook |
# MAGIC |---|---|
# MAGIC | `gold_metricas_temporais` | 02 |
# MAGIC | `gold_serie_diaria_30d` | 02 |
# MAGIC | `gold_metricas_geograficas` | 03 |
# MAGIC | `gold_metricas_demograficas` | 04 |
# MAGIC | `gold_rag_kpi_fatos` | 05 |
# MAGIC | `gold_rag_dicionario_regras` | 05 |

# COMMAND ----------

from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parâmetros do Pipeline

# COMMAND ----------

dbutils.widgets.text("catalog_silver",    "dbx_srag_lab",      "Catalog Silver")
dbutils.widgets.text("schema_silver",     "silver",            "Schema Silver")
dbutils.widgets.text("table_silver_name", "silver_srag_clean", "Tabela Silver")
dbutils.widgets.text("catalog_gold",      "dbx_srag_lab",      "Catalog Gold")
dbutils.widgets.text("schema_gold",       "gold",              "Schema Gold")
dbutils.widgets.text("timeout_setup",     "600",               "Timeout setup (s)")
dbutils.widgets.text("timeout_analitico", "3600",              "Timeout analítico (s)")
dbutils.widgets.text("timeout_rag",       "1800",              "Timeout RAG (s)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configurações do Orquestrador

# COMMAND ----------

# Leitura dos parâmetros
CATALOG_SILVER    = dbutils.widgets.get("catalog_silver")
SCHEMA_SILVER     = dbutils.widgets.get("schema_silver")
TABLE_SILVER_NAME = dbutils.widgets.get("table_silver_name")
CATALOG_GOLD      = dbutils.widgets.get("catalog_gold")
SCHEMA_GOLD       = dbutils.widgets.get("schema_gold")
TIMEOUT_SETUP     = int(dbutils.widgets.get("timeout_setup"))
TIMEOUT_ANALITICO = int(dbutils.widgets.get("timeout_analitico"))
TIMEOUT_RAG       = int(dbutils.widgets.get("timeout_rag"))

# Process ID único para toda a execução do pipeline
PIPELINE_START    = datetime.now()
PIPELINE_ID       = PIPELINE_START.strftime('%Y%m%d_%H%M%S')

# Argumentos base repassados para todos os notebooks filhos
BASE_ARGS = {
    "catalog_silver"    : CATALOG_SILVER,
    "schema_silver"     : SCHEMA_SILVER,
    "table_silver_name" : TABLE_SILVER_NAME,
    "catalog_gold"      : CATALOG_GOLD,
    "schema_gold"       : SCHEMA_GOLD,
    "process_id"        : PIPELINE_ID,
}

print("=" * 80)
print("GOLD PIPELINE — INICIO")
print("=" * 80)
print(f"  Pipeline ID  : {PIPELINE_ID}")
print(f"  Silver       : {CATALOG_SILVER}.{SCHEMA_SILVER}.{TABLE_SILVER_NAME}")
print(f"  Gold         : {CATALOG_GOLD}.{SCHEMA_GOLD}")
print(f"  Timeouts     : setup={TIMEOUT_SETUP}s | analitico={TIMEOUT_ANALITICO}s | rag={TIMEOUT_RAG}s")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Utilitário de Execução

# COMMAND ----------

# Registro de resultados de cada etapa
_resultados = {}

def _run_notebook(nome: str, path: str, timeout: int, args: dict) -> dict:
    """
    Executa um notebook filho via dbutils.notebook.run e retorna um dict
    com status, duração e mensagem de erro se houver.
    
    Parâmetros
    ----------
    nome    : label amigável usado nos logs
    path    : caminho relativo do notebook (ex: './02_gold_metricas_temporais')
    timeout : segundos máximos de execução
    args    : dict de argumentos repassados como widgets
    
    Retorno
    -------
    dict com campos: nome, status ('OK' | 'ERRO'), duracao_s, mensagem
    """
    inicio = datetime.now()
    try:
        result = dbutils.notebook.run(path, timeout, args)
        duracao = (datetime.now() - inicio).total_seconds()
        return {
            "nome"      : nome,
            "status"    : "OK",
            "duracao_s" : round(duracao, 1),
            "mensagem"  : result or "Concluido com sucesso",
        }
    except Exception as e:
        duracao = (datetime.now() - inicio).total_seconds()
        return {
            "nome"      : nome,
            "status"    : "ERRO",
            "duracao_s" : round(duracao, 1),
            "mensagem"  : str(e),
        }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Etapa 1 — Setup Gold
# MAGIC
# MAGIC Valida a Silver, cria o schema Gold e registra os widgets derivados
# MAGIC (`table_silver`, `data_snapshot`) que os notebooks filhos consomem.

# COMMAND ----------

print(f"[{datetime.now().strftime('%H:%M:%S')}] Iniciando 01_gold_setup ...")

resultado_setup = _run_notebook(
    nome    = "01_gold_setup",
    path    = "./01_gold_setup",
    timeout = TIMEOUT_SETUP,
    args    = BASE_ARGS,
)

_resultados["01_gold_setup"] = resultado_setup

status_icon = "✓" if resultado_setup["status"] == "OK" else "✗"
print(f"  {status_icon} {resultado_setup['nome']} — {resultado_setup['status']} "
      f"({resultado_setup['duracao_s']}s)")

if resultado_setup["status"] == "ERRO":
    print(f"\n  ERRO FATAL: o setup falhou. Pipeline interrompido.")
    print(f"  Detalhe: {resultado_setup['mensagem']}")
    dbutils.notebook.exit(f"FALHA_SETUP | pipeline_id={PIPELINE_ID}")

print(f"  Setup concluido. Schema Gold validado.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Etapa 2 — Notebooks Analíticos (paralelo)
# MAGIC
# MAGIC Os notebooks 02, 03 e 04 são independentes entre si e executam
# MAGIC simultaneamente via `ThreadPoolExecutor`. O pipeline avança para a
# MAGIC Etapa 3 somente após todos os três concluírem (com ou sem falha).

# COMMAND ----------

NOTEBOOKS_ANALITICOS = [
    ("02_gold_metricas_temporais",   "./02_gold_metricas_temporais"),
    ("03_gold_metricas_geograficas", "./03_gold_metricas_geograficas"),
    ("04_gold_metricas_demograficas","./04_gold_metricas_demograficas"),
]

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Iniciando etapa analítica (paralelo) ...")
print(f"  Notebooks : {[n for n, _ in NOTEBOOKS_ANALITICOS]}")

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = {
        executor.submit(_run_notebook, nome, path, TIMEOUT_ANALITICO, BASE_ARGS): nome
        for nome, path in NOTEBOOKS_ANALITICOS
    }
    for future in as_completed(futures):
        resultado = future.result()
        _resultados[resultado["nome"]] = resultado
        status_icon = "✓" if resultado["status"] == "OK" else "✗"
        print(f"  {status_icon} {resultado['nome']} — {resultado['status']} "
              f"({resultado['duracao_s']}s)")
        if resultado["status"] == "ERRO":
            print(f"    Detalhe: {resultado['mensagem']}")

# Verifica se algum analítico falhou antes de continuar para o RAG
falhas_analiticas = [
    r for r in _resultados.values()
    if r["nome"] != "01_gold_setup" and r["status"] == "ERRO"
]

if falhas_analiticas:
    nomes_falhos = [r["nome"] for r in falhas_analiticas]
    print(f"\n  ATENCAO: {len(falhas_analiticas)} notebook(s) analítico(s) falharam: {nomes_falhos}")
    print(f"  O notebook 05 (RAG) depende de tabelas atualizadas por estes notebooks.")
    print(f"  Pipeline interrompido para evitar RAG com dados desatualizados.")
    dbutils.notebook.exit(
        f"FALHA_ANALITICA | falhos={nomes_falhos} | pipeline_id={PIPELINE_ID}"
    )

print(f"\n  Etapa analítica concluída com sucesso.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Etapa 3 — Base de Conhecimento RAG
# MAGIC
# MAGIC Executado somente após a conclusão bem-sucedida dos notebooks 02, 03 e 04,
# MAGIC pois o RAG consolida KPIs que dependem das tabelas analíticas estarem atualizadas.

# COMMAND ----------

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Iniciando 05_gold_base_conhecimento_rag ...")

resultado_rag = _run_notebook(
    nome    = "05_gold_base_conhecimento_rag",
    path    = "./05_gold_base_conhecimento_rag",
    timeout = TIMEOUT_RAG,
    args    = BASE_ARGS,
)

_resultados["05_gold_base_conhecimento_rag"] = resultado_rag

status_icon = "✓" if resultado_rag["status"] == "OK" else "✗"
print(f"  {status_icon} {resultado_rag['nome']} — {resultado_rag['status']} "
      f"({resultado_rag['duracao_s']}s)")

if resultado_rag["status"] == "ERRO":
    print(f"\n  ERRO: RAG falhou. Tabelas analíticas foram atualizadas normalmente.")
    print(f"  Detalhe: {resultado_rag['mensagem']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resumo de Execução

# COMMAND ----------

duracao_total = round((datetime.now() - PIPELINE_START).total_seconds(), 1)

# Ordena resultados pela sequência lógica de execução
_ordem = [
    "01_gold_setup",
    "02_gold_metricas_temporais",
    "03_gold_metricas_geograficas",
    "04_gold_metricas_demograficas",
    "05_gold_base_conhecimento_rag",
]

total_ok   = sum(1 for r in _resultados.values() if r["status"] == "OK")
total_erro = sum(1 for r in _resultados.values() if r["status"] == "ERRO")
status_pipeline = "SUCESSO" if total_erro == 0 else f"FALHA PARCIAL ({total_erro} erro(s))"

print("=" * 80)
print(f"GOLD PIPELINE — RESUMO  [{status_pipeline}]")
print("=" * 80)
print(f"  Pipeline ID   : {PIPELINE_ID}")
print(f"  Duração total : {duracao_total}s")
print(f"  Notebooks OK  : {total_ok}/{len(_resultados)}")
print()
print(f"  {'Notebook':<40} {'Status':<10} {'Duração':>10}")
print(f"  {'-'*40} {'-'*10} {'-'*10}")

for nome in _ordem:
    if nome in _resultados:
        r = _resultados[nome]
        status_icon = "✓" if r["status"] == "OK" else "✗"
        print(f"  {status_icon} {r['nome']:<38} {r['status']:<10} {r['duracao_s']:>8}s")

print("=" * 80)

# Saída estruturada para Jobs externos capturarem o resultado
exit_msg = f"{status_pipeline} | pipeline_id={PIPELINE_ID} | duracao={duracao_total}s"
dbutils.notebook.exit(exit_msg)
