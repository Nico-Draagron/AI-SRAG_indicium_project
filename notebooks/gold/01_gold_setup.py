# Databricks notebook source
# MAGIC %md
# MAGIC # Gold Layer — Setup e Configuração
# MAGIC
# MAGIC ## 1. Objetivo
# MAGIC
# MAGIC Inicializar o ambiente de execução da camada Gold:
# MAGIC criar o schema de destino, validar a disponibilidade da fonte Silver
# MAGIC e exportar as constantes de configuração para os notebooks dependentes.
# MAGIC
# MAGIC Este notebook deve ser executado antes de qualquer outro notebook da camada Gold.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 2. Dependências
# MAGIC
# MAGIC | Componente | Identificador |
# MAGIC |---|---|
# MAGIC | Catálogo (leitura) | `dbx_srag_lab` |
# MAGIC | Schema fonte | `dbx_srag_lab.silver` |
# MAGIC | Tabela fonte | `dbx_srag_lab.silver.silver_srag_clean` |
# MAGIC | Catálogo (escrita) | `dbx_srag_lab` |
# MAGIC | Schema destino | `dbx_srag_lab.gold` |
# MAGIC
# MAGIC O catálogo `dbx_srag_lab` deve existir previamente.
# MAGIC A criação de catálogos está fora do escopo deste pipeline.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 3. Convenções de Métrica
# MAGIC
# MAGIC As definições abaixo são aplicadas de forma consistente em todos os notebooks Gold.
# MAGIC Qualquer desvio deve ser documentado explicitamente no notebook correspondente.
# MAGIC
# MAGIC ### 3.1 Taxa de Mortalidade SRAG (definição estrita)
# MAGIC
# MAGIC Inclui apenas casos com desfecho registrado.
# MAGIC `1` = cura, `2` = óbito por SRAG, `3` = óbito por outras causas (excluído), `9` = ignorado (excluído).
# MAGIC
# MAGIC ```
# MAGIC Denominador : evolucao_clean IN ('1', '2')
# MAGIC Numerador   : evolucao_clean = '2'
# MAGIC ```
# MAGIC
# MAGIC ### 3.2 Taxa de Utilização de UTI
# MAGIC
# MAGIC Calculada apenas sobre a população de pacientes internados com indicador de UTI válido.
# MAGIC
# MAGIC ```
# MAGIC Denominador : is_internado = TRUE
# MAGIC Numerador   : is_uti_valido = TRUE
# MAGIC ```
# MAGIC
# MAGIC ### 3.3 Cobertura Vacinal
# MAGIC
# MAGIC Exclui registros com status vacinal ausente ou desconhecido (código `9` mapeado para NULL na Silver).
# MAGIC
# MAGIC ```
# MAGIC Denominador : vacina_clean IS NOT NULL
# MAGIC Numerador   : vacina_clean = '1'
# MAGIC ```
# MAGIC
# MAGIC ### 3.4 Estratificação por Idade
# MAGIC
# MAGIC Utilizar exclusivamente o campo `idade_anos`, já padronizado na camada Silver.
# MAGIC `idade_anos` é preenchido apenas quando `TP_IDADE = '3'` (anos).
# MAGIC Registros com outras unidades recebem `idade_anos = NULL` e `faixa_etaria = 'Desconhecido'`.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 4. Catálogo e Schemas
# MAGIC
# MAGIC | Camada | Catálogo | Schema |
# MAGIC |---|---|---|
# MAGIC | Silver (fonte) | `dbx_srag_lab` | `silver` |
# MAGIC | Gold (destino) | `dbx_srag_lab` | `gold` |
# MAGIC
# MAGIC O schema `dbx_srag_lab.gold` é criado por este notebook caso não exista.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 5. Tabelas de Saída
# MAGIC
# MAGIC | Identificador lógico | Tabela física |
# MAGIC |---|---|
# MAGIC | `metricas_temporais` | `dbx_srag_lab.gold.gold_metricas_temporais` |
# MAGIC | `metricas_geograficas` | `dbx_srag_lab.gold.gold_metricas_geograficas` |
# MAGIC | `metricas_demograficas` | `dbx_srag_lab.gold.gold_metricas_demograficas` |
# MAGIC | `series_temporais` | `dbx_srag_lab.gold.gold_series_temporais` |
# MAGIC | `serie_diaria_30d` | `dbx_srag_lab.gold.gold_serie_diaria_30d` |
# MAGIC | `resumo_geral` | `dbx_srag_lab.gold.gold_resumo_geral` |
# MAGIC | `rag_kpi_fatos` | `dbx_srag_lab.gold.gold_rag_kpi_fatos` |
# MAGIC | `rag_dicionario_regras` | `dbx_srag_lab.gold.gold_rag_dicionario_regras` |

# COMMAND ----------

from pyspark.sql import functions as F
from datetime import datetime

print("=" * 80)
print("GOLD — SETUP E CONFIGURACAO")
print("=" * 80)
print(f"Timestamp  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Spark      : {spark.version}")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Widgets — Fonte Única de Configuração
# MAGIC
# MAGIC Todos os notebooks Gold lêem exclusivamente via `dbutils.widgets.get(...)`.
# MAGIC `TABLE_SILVER` é derivado de `catalog_silver`, `schema_silver` e `table_silver_name`
# MAGIC para evitar que paths parcialmente sobrescritos causem destinos inconsistentes.

# COMMAND ----------

_process_id = datetime.now().strftime('%Y%m%d_%H%M%S')

dbutils.widgets.text("catalog_silver",    "dbx_srag_lab",       "Catalog Silver (leitura)")
dbutils.widgets.text("schema_silver",     "silver",             "Schema Silver")
dbutils.widgets.text("table_silver_name", "silver_srag_clean",  "Nome da tabela Silver")
dbutils.widgets.text("catalog_gold",      "dbx_srag_lab",       "Catalog Gold (escrita)")
dbutils.widgets.text("schema_gold",       "gold",               "Schema Gold")
dbutils.widgets.text("process_id",        _process_id,          "Process ID")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Leitura e Derivação das Configurações

# COMMAND ----------

CATALOG_SILVER     = dbutils.widgets.get("catalog_silver")
SCHEMA_SILVER      = dbutils.widgets.get("schema_silver")
TABLE_SILVER_NAME  = dbutils.widgets.get("table_silver_name")
CATALOG_GOLD       = dbutils.widgets.get("catalog_gold")
SCHEMA_GOLD        = dbutils.widgets.get("schema_gold")
PROCESS_ID         = dbutils.widgets.get("process_id")

# TABLE_SILVER derivado dos três componentes para garantir consistência.
TABLE_SILVER = f"{CATALOG_SILVER}.{SCHEMA_SILVER}.{TABLE_SILVER_NAME}"

# DATA_SNAPSHOT via Spark para consistência com o fuso do cluster.
DATA_SNAPSHOT = spark.sql("SELECT current_date() AS d").collect()[0]["d"].isoformat()

# Registrar TABLE_SILVER e DATA_SNAPSHOT como widgets para notebooks filhos.
dbutils.widgets.text("table_silver",  TABLE_SILVER,  "Tabela Silver (derivada)")
dbutils.widgets.text("data_snapshot", DATA_SNAPSHOT, "Data Snapshot (YYYY-MM-DD)")

print("Configuracao ativa:")
print(f"  Fonte    : {TABLE_SILVER}")
print(f"  Destino  : {CATALOG_GOLD}.{SCHEMA_GOLD}")
print(f"  Process  : {PROCESS_ID}")
print(f"  Snapshot : {DATA_SNAPSHOT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Criação do Schema Gold

# COMMAND ----------

try:
    spark.sql(f"""
        CREATE SCHEMA IF NOT EXISTS {CATALOG_GOLD}.{SCHEMA_GOLD}
        COMMENT 'Camada Gold — metricas agregadas para BI e RAG'
    """)
    print(f"Schema criado/verificado: {CATALOG_GOLD}.{SCHEMA_GOLD}")
except Exception:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG_GOLD}.{SCHEMA_GOLD}")
    print(f"Schema criado/verificado sem COMMENT: {CATALOG_GOLD}.{SCHEMA_GOLD}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validação da Tabela Silver

# COMMAND ----------

# Colunas obrigatórias para o pipeline Gold.
# sem_pri é opcional — notebooks que dependem dela aplicam fallback próprio.
COLUNAS_OBRIGATORIAS = ['dt_sin_pri', 'dt_notific', 'sg_uf', 'ano', 'mes']
COLUNAS_OPCIONAIS    = [
    'sem_pri',
    'classi_fin_clean', 'is_covid', 'is_influenza', 'is_outro_virus',
    'vacina_cov_clean', 'is_vacinado_covid',
]

try:
    df_silver    = spark.table(TABLE_SILVER)
    count_silver = df_silver.count()
    colunas_silver = set(df_silver.columns)

    faltantes = [c for c in COLUNAS_OBRIGATORIAS if c not in colunas_silver]
    opcionais_presentes = [c for c in COLUNAS_OPCIONAIS if c in colunas_silver]
    opcionais_ausentes  = [c for c in COLUNAS_OPCIONAIS if c not in colunas_silver]

    periodo = df_silver.agg(
        F.min('dt_sin_pri').alias('min_data'),
        F.max('dt_sin_pri').alias('max_data'),
    ).collect()[0]

    print(f"Tabela Silver acessada.")
    print(f"  Tabela    : {TABLE_SILVER}")
    print(f"  Registros : {count_silver:,}")
    print(f"  Colunas   : {len(colunas_silver)}")
    print(f"  Periodo   : {periodo['min_data']} a {periodo['max_data']}")

    if faltantes:
        print(f"  ATENCAO — colunas obrigatorias ausentes : {faltantes}")
    else:
        print(f"  Colunas obrigatorias presentes : OK")

    if opcionais_presentes:
        print(f"  Colunas opcionais presentes    : {opcionais_presentes}")
    if opcionais_ausentes:
        print(f"  Colunas opcionais ausentes     : {opcionais_ausentes} (fallback via weekofyear aplicavel)")
        _campos_etiol = [c for c in opcionais_ausentes
                         if c in ('classi_fin_clean','is_covid','is_influenza',
                                   'is_outro_virus','vacina_cov_clean','is_vacinado_covid')]
        if _campos_etiol:
            print(f"  ATENCAO — campos etiologicos ausentes: {_campos_etiol}")
            print(f"            Execute Silver v2 antes dos notebooks Gold que usam classi_fin.")

    assert count_silver > 0, "A tabela Silver esta vazia."
    assert not faltantes,    f"Colunas obrigatorias ausentes: {faltantes}"

except Exception as e:
    print(f"ERRO ao acessar a tabela Silver: {str(e)}")
    print()
    print("Verifique:")
    print(f"  1. O catalogo '{CATALOG_SILVER}' existe e esta acessivel.")
    print(f"  2. O schema '{CATALOG_SILVER}.{SCHEMA_SILVER}' existe.")
    print(f"  3. A tabela '{TABLE_SILVER}' foi criada pela camada Silver.")
    print(f"  4. O cluster possui permissao de leitura no catalogo '{CATALOG_SILVER}'.")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resumo de Execução

# COMMAND ----------

print("=" * 80)
print("SETUP GOLD — RESUMO")
print("=" * 80)
print(f"  Fonte    : {TABLE_SILVER}")
print(f"  Destino  : {CATALOG_GOLD}.{SCHEMA_GOLD}")
print(f"  Process  : {PROCESS_ID}")
print(f"  Snapshot : {DATA_SNAPSHOT}")
print()
print("Ordem de execucao dos notebooks dependentes:")
print("  1. gold_metricas_temporais    (inclui gold_serie_diaria_30d)")
print("  2. gold_metricas_geograficas  (pode rodar em paralelo com 1)")
print("  3. gold_metricas_demograficas (pode rodar em paralelo com 1 e 2)")
print("  4. gold_base_conhecimento_rag    — executar APOS 1, 2 e 3")
print()
print("  Notebooks 1, 2 e 3: execucao paralela permitida.")
print("  Notebook 4 (RAG): DEVE ser executado por ultimo.")
print("=" * 80)

# COMMAND ----------


