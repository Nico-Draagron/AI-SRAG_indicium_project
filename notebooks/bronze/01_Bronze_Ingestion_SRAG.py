# Databricks notebook source
# MAGIC %md
# MAGIC # Camada Bronze — Ingestão de Dados SRAG
# MAGIC
# MAGIC **Projeto**: Sistema RAG para Monitoramento Epidemiológico — Indicium Healthcare PoC
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Objetivo
# MAGIC
# MAGIC Ingerir os CSVs brutos de SRAG (2023–2025) disponibilizados pelo DATASUS/SIVEP-Gripe
# MAGIC e persistir os dados sem transformações de negócio em uma Delta Table no Unity Catalog.
# MAGIC
# MAGIC **Origem**: CSVs em `/Volumes/dbx_srag_lab/data_original/data_srag/` (ISO-8859-1, separador `;`)
# MAGIC
# MAGIC **Destino**: `dbx_srag_lab.data_original.bronze_srag_raw`
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Premissas
# MAGIC
# MAGIC - Todos os campos de negócio são lidos como `string` (`inferSchema=false`).
# MAGIC   Conversões de tipo pertencem à camada Silver.
# MAGIC - Campos técnicos adicionados nesta camada: `ANO_DADOS` (int), `_ingested_at` (timestamp),
# MAGIC   `_source_file` (string), `_ingestion_run_id` (string).
# MAGIC - A escrita usa `overwrite` para garantir idempotência (PoC). Em produção, avaliar
# MAGIC   append particionado por `ANO_DADOS`.
# MAGIC - `allowMissingColumns=True` no `unionByName` é necessário porque o schema dos CSVs
# MAGIC   pode variar entre anos (ex.: campos introduzidos para COVID-19 em 2023+).
# MAGIC - `overwriteSchema=True` permite reprocessamento sem falha por schema drift.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Decisões relevantes
# MAGIC
# MAGIC - `_source_file` capturado via `input_file_name()` durante a leitura, preservando o
# MAGIC   nome real do arquivo (basename). Isso garante rastreabilidade mesmo que o padrão de
# MAGIC   nome mude entre extrações do DATASUS.
# MAGIC - `_ingestion_run_id` gerado uma única vez por execução (UUID), permitindo identificar
# MAGIC   exatamente qual run produziu cada versão da tabela no histórico Delta.
# MAGIC - Counts por ano são feitos apenas na validação prévia (dry-run) para feedback rápido.
# MAGIC   O count definitivo ocorre uma única vez após o `unionByName`.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Dependências e próximos notebooks
# MAGIC
# MAGIC 1. `02_Data_Quality_Validation.py` — lê `bronze_srag_raw` e aplica checks de qualidade
# MAGIC 2. `03_eda_srag_exploraty_analysis.py` — análise exploratória sobre o Bronze
# MAGIC 3. `04_Silver_Transformation.py` — limpeza, tipagem e regras de negócio
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Troubleshooting
# MAGIC
# MAGIC | Sintoma | Causa provável | Ação |
# MAGIC |---|---|---|
# MAGIC | `FileNotFoundError` no ls do volume | CSV não foi carregado ou path errado | Verificar upload e `VOLUME_PATH` |
# MAGIC | `AnalysisException: Table not found` na validação | Primeira execução ou catálogo/schema ausente | Executar `CREATE SCHEMA IF NOT EXISTS` manualmente |
# MAGIC | `UnresolvedAttribute` no `unionByName` | Coluna nova no CSV de um ano sem correspondência | Confirmar que `allowMissingColumns=True` está ativo |
# MAGIC | Schema drift no overwrite | CSV do DATASUS adicionou/removeu colunas | `overwriteSchema=True` resolve; revisar impacto downstream |
# MAGIC | Permissão negada no Volume | Privilege não concedido no Unity Catalog | `GRANT READ FILES ON VOLUME ... TO ...` |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from datetime import datetime
import uuid
import json

print("-" * 70)
print("CAMADA BRONZE - INGESTÃO SRAG")
print(f"Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Spark     : {spark.version}")
print("-" * 70)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuração

# COMMAND ----------

dbutils.widgets.text("anos_dados",     "2023,2024,2025", "Anos a ingerir (separados por vírgula)")
dbutils.widgets.text("catalog",        "dbx_srag_lab",   "Catálogo Unity Catalog")
dbutils.widgets.text("schema_bronze",  "data_original",  "Schema Bronze")
dbutils.widgets.text("volume_raw",     "data_srag",      "Nome do volume de dados")

CATALOG       = dbutils.widgets.get("catalog")
SCHEMA_BRONZE = dbutils.widgets.get("schema_bronze")
VOLUME_RAW    = dbutils.widgets.get("volume_raw")
YEARS         = [int(a.strip()) for a in dbutils.widgets.get("anos_dados").split(",")]

VOLUME_PATH   = f"/Volumes/{CATALOG}/{SCHEMA_BRONZE}/{VOLUME_RAW}"
TABLE_BRONZE  = f"{CATALOG}.{SCHEMA_BRONZE}.bronze_srag_raw"

# Identificador único desta execução — persiste na tabela e nos logs
INGESTION_RUN_ID = str(uuid.uuid4())

# Opções de leitura centralizadas — dry-run e ingestão real usam exatamente as mesmas opções
CSV_READ_OPTIONS = {
    "header":      "true",
    "sep":         ";",
    "encoding":    "ISO-8859-1",
    "inferSchema": "false",
    "quote":       '"',
    "escape":      '"',
    "multiLine":   "true",
    "mode":        "PERMISSIVE",
}

print(f"Catálogo      : {CATALOG}")
print(f"Schema        : {SCHEMA_BRONZE}")
print(f"Caminho volume: {VOLUME_PATH}")
print(f"Tabela destino: {TABLE_BRONZE}")
print(f"Anos          : {YEARS}")
print(f"Run ID        : {INGESTION_RUN_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Verificação de arquivos no volume

# COMMAND ----------

try:
    files    = dbutils.fs.ls(VOLUME_PATH)
    csv_files = [f for f in files if f.name.endswith('.csv')]

    if not csv_files:
        raise FileNotFoundError(f"Nenhum arquivo CSV encontrado em {VOLUME_PATH}")

    print(f"{len(csv_files)} arquivo(s) CSV encontrado(s):")
    for f in csv_files:
        print(f"  {f.name:<35}  {f.size / 1_048_576:,.2f} MB")

except Exception as e:
    print(f"ERRO ao acessar o volume: {e}")
    print(f"  1. Confirme que o volume existe : {VOLUME_PATH}")
    print(f"  2. Faça upload dos CSVs         : INFLUD23-*.csv, INFLUD24-*.csv, INFLUD25-*.csv")
    print(f"  3. Verifique privilégios UC      : GRANT READ FILES ON VOLUME")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Função de ingestão por ano

# COMMAND ----------

def ingest_year(year: int, volume_path: str) -> "DataFrame":
    """
    Lê o CSV de SRAG de um ano específico e retorna um DataFrame com metadados técnicos.

    - Todos os campos de negócio permanecem como string.
    - _source_file é capturado via _metadata.file_path (Unity Catalog).
    - ANO_DADOS é convertido para int logo após a leitura.

    Args:
        year        : Ano dos dados (2023, 2024 ou 2025).
        volume_path : Caminho do volume Unity Catalog que contém os CSVs.

    Returns:
        DataFrame com colunas originais + ANO_DADOS, _ingested_at, _source_file.
        O campo _ingestion_run_id é adicionado na etapa de consolidação.
    """
    year_suffix  = str(year)[2:]
    file_pattern = f"{volume_path}/INFLUD{year_suffix}-*.csv"

    print(f"\n[{year}] Lendo: INFLUD{year_suffix}-*.csv")

    reader = spark.read
    for k, v in CSV_READ_OPTIONS.items():
        reader = reader.option(k, v)

    df = reader.csv(file_pattern)

    # Unity Catalog: preferir _metadata.file_path
    if "_metadata" in df.columns:
        df = df.withColumn("_source_file_path", F.col("_metadata.file_path"))
    else:
        # fallback: não quebra o pipeline; mantém rastreabilidade mínima
        df = df.withColumn("_source_file_path", F.lit(None).cast("string"))

    df = (
        df
        .withColumn("ANO_DADOS",    F.lit(year).cast(IntegerType()))
        .withColumn("_ingested_at", F.current_timestamp())
        .withColumn(
            "_source_file",
            F.regexp_extract(F.col("_source_file_path"), r"([^/\\]+\.csv)$", 1)
        )
        .drop("_source_file_path")
    )

    print(f"[{year}] Colunas : {len(df.columns)}")
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Validação prévia (dry-run)
# MAGIC
# MAGIC Conta registros por ano antes de materializar os DataFrames completos.
# MAGIC Interrompe a execução se algum arquivo estiver ausente ou ilegível.

# COMMAND ----------

print("Validação prévia (dry-run) — verificando todos os arquivos antes da ingestão completa...\n")

validation_results = []
all_ok = True

for year in YEARS:
    year_suffix  = str(year)[2:]
    file_pattern = f"{VOLUME_PATH}/INFLUD{year_suffix}-*.csv"
    try:
        reader = spark.read
        for k, v in CSV_READ_OPTIONS.items():
            reader = reader.option(k, v)
        df_check = reader.csv(file_pattern)
        row_count = df_check.count()
        col_count = len(df_check.columns)
        validation_results.append({"year": year, "rows": row_count, "columns": col_count, "status": "OK"})
        print(f"  {year}  {row_count:>10,} linhas  |  {col_count} colunas  [OK]")
    except Exception as e:
        validation_results.append({"year": year, "status": "ERRO", "error": str(e)})
        print(f"  {year}  FALHOU — {e}")
        all_ok = False

total_estimated = sum(r.get("rows", 0) for r in validation_results)
print(f"\n  Total estimado : {total_estimated:,} linhas")

if not all_ok:
    raise RuntimeError("Validação dry-run falhou. Corrija os erros acima antes de prosseguir.")

print("\nDry-run concluído. Iniciando ingestão completa...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Ingestão completa

# COMMAND ----------

dataframes = {}

for year in YEARS:
    dataframes[year] = ingest_year(year, VOLUME_PATH)

print(f"\n{len(dataframes)} DataFrames carregados.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. União dos anos

# COMMAND ----------

# allowMissingColumns=True é necessário porque o schema do SRAG pode variar entre anos.
# Colunas ausentes num determinado ano serão preenchidas com null.

df_bronze = dataframes[YEARS[0]]
for year in YEARS[1:]:
    df_bronze = df_bronze.unionByName(dataframes[year], allowMissingColumns=True)

# _ingestion_run_id adicionado uma única vez após a união — identifica esta execução
df_bronze = df_bronze.withColumn("_ingestion_run_id", F.lit(INGESTION_RUN_ID))

EXPECTED_COLS = [
    'NU_NOTIFIC', 'DT_SIN_PRI', 'DT_NOTIFIC', 'SEM_PRI',
    'CS_SEXO', 'NU_IDADE_N', 'TP_IDADE', 'SG_UF', 'CO_MUN_RES',
    'HOSPITAL', 'UTI', 'EVOLUCAO', 'VACINA', 'CLASSI_FIN', 'VACINA_COV',
]
cols_bronze = set(df_bronze.columns)
cols_ausentes = [c for c in EXPECTED_COLS if c not in cols_bronze]
if cols_ausentes:
    print(f"ATENCAO — colunas esperadas ausentes na Bronze: {cols_ausentes}")
    print("  O pipeline Silver pode falhar ou gerar campos vazios.")
else:
    print(f"Schema Bronze validado: todas as {len(EXPECTED_COLS)} colunas criticas presentes.")

# Count único após consolidação
total_rows = df_bronze.count()
total_cols = len(df_bronze.columns)

print("-" * 70)
print("DATASET BRONZE — CONSOLIDADO")
print(f"  Linhas   : {total_rows:,}")
print(f"  Colunas  : {total_cols}")
print(f"  Período  : {min(YEARS)} — {max(YEARS)}")
print("-" * 70)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Inspeção estrutural

# COMMAND ----------

print("Schema (primeiras 20 colunas):")
for field in df_bronze.schema.fields[:20]:
    print(f"  {field.name:<28} {str(field.dataType)}")
if total_cols > 20:
    print(f"  ... e mais {total_cols - 20} colunas")

# COMMAND ----------

print("Distribuição de linhas por ano:")
df_bronze.groupBy("ANO_DADOS").count().orderBy("ANO_DADOS").show()

# COMMAND ----------

# Amostra para inspeção visual — colunas essenciais
essential_cols = [
    "DT_NOTIFIC", "DT_SIN_PRI", "SEM_PRI", "CS_SEXO",
    "NU_IDADE_N", "SG_UF", "HOSPITAL", "UTI", "EVOLUCAO",
    "ANO_DADOS", "_ingested_at", "_source_file", "_ingestion_run_id",
    "CLASSI_FIN", "VACINA_COV",
]
display_cols = [c for c in essential_cols if c in df_bronze.columns]
display(df_bronze.select(display_cols).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Escrita em Delta Lake

# COMMAND ----------

print(f"Gravando em: {TABLE_BRONZE}")
print(f"  modo            : overwrite")
print(f"  overwriteSchema : True")
print(f"  run_id          : {INGESTION_RUN_ID}")

try:
    (
        df_bronze.write
        .mode("overwrite")
        .format("delta")
        .option("overwriteSchema", "true")
        .saveAsTable(TABLE_BRONZE)
    )
    print(f"Tabela {TABLE_BRONZE} gravada com sucesso.")
except Exception as e:
    print(f"ERRO ao gravar tabela: {e}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Validação da tabela criada

# COMMAND ----------

df_val = spark.table(TABLE_BRONZE)

checks = {
    "row_count_match"  : df_val.count() == total_rows,
    "column_count"     : len(df_val.columns) == total_cols,
    "metadata_present" : all(
        c in df_val.columns
        for c in ["ANO_DADOS", "_ingested_at", "_source_file", "_ingestion_run_id"]
    ),
    "all_years_present": df_val.select("ANO_DADOS").distinct().count() == len(YEARS),
}

print("Verificações de integridade:")
all_passed = True
for name, result in checks.items():
    status = "OK" if result else "FALHA"
    print(f"  [{status}] {name}")
    if not result:
        all_passed = False

if not all_passed:
    raise AssertionError("Validação pós-escrita falhou. Revise as verificações acima.")

print("\nTodas as verificações passaram.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Metadados da tabela Delta

# COMMAND ----------

spark.sql(f"DESCRIBE EXTENDED {TABLE_BRONZE}").show(50, truncate=False)

# COMMAND ----------

print("Histórico Delta:")
try:
    history = spark.sql(f"DESCRIBE HISTORY {TABLE_BRONZE}")
    display(history.select("version", "timestamp", "operation", "operationMetrics"))
except Exception as e:
    print(f"Histórico não disponível: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Log de execução

# COMMAND ----------

ingestion_stats = {
    "run_id"           : INGESTION_RUN_ID,
    "timestamp"        : datetime.now().isoformat(),
    "catalog"          : CATALOG,
    "schema"           : SCHEMA_BRONZE,
    "table"            : TABLE_BRONZE.split(".")[-1],
    "total_rows"       : total_rows,
    "total_columns"    : total_cols,
    "years_processed"  : YEARS,
    "validation_passed": all_passed,
}

print(json.dumps(ingestion_stats, indent=2, ensure_ascii=False))

logs_dir  = f"{VOLUME_PATH}/logs"
stats_path = f"{logs_dir}/ingestion_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

try:
    dbutils.fs.mkdirs(logs_dir)
    dbutils.fs.put(
        stats_path,
        json.dumps(ingestion_stats, indent=2, ensure_ascii=False),
        overwrite=True,
    )
    print(f"\nEstatísticas salvas em: {stats_path}")
except Exception as e:
    print(f"Não foi possível salvar o arquivo de estatísticas: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Resumo da execução

# COMMAND ----------

print("-" * 70)
print("INGESTÃO BRONZE CONCLUÍDA")
print("-" * 70)
print(f"  Tabela    : {TABLE_BRONZE}")
print(f"  Linhas    : {total_rows:,}")
print(f"  Colunas   : {total_cols}")
print(f"  Anos      : {', '.join(map(str, YEARS))}")
print(f"  Formato   : Delta Lake")
print(f"  Run ID    : {INGESTION_RUN_ID}")
print("-" * 70)
print("Próximo: 02_Data_Quality_Validation.py")
