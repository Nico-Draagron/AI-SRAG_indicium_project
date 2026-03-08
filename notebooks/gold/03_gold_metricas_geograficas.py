# Databricks notebook source
# MAGIC %md
# MAGIC # Gold — Métricas Geográficas
# MAGIC
# MAGIC ## 1. Objetivo
# MAGIC
# MAGIC Produzir a tabela `gold_metricas_geograficas` com agregação por Unidade da Federação (UF),
# MAGIC incluindo taxas epidemiológicas, indicadores hospitalares, distribuição demográfica,
# MAGIC ranking absoluto de casos e participação percentual no total nacional.
# MAGIC
# MAGIC Pré-requisito: o notebook `01_gold_setup` deve ter sido executado na mesma sessão.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 2. Granularidade Geográfica
# MAGIC
# MAGIC A unidade de agrupamento é `sg_uf` (sigla da UF de residência do paciente).
# MAGIC Registros com `sg_uf IS NULL` são excluídos antes da agregação para evitar
# MAGIC contaminação do ranking. O total nacional usado em `percentual_nacional` é calculado
# MAGIC sobre o universo completo da Silver, antes desse filtro, para preservar a proporção real.
# MAGIC
# MAGIC | Atributo | Valor |
# MAGIC |---|---|
# MAGIC | Tabela fonte | `dbx_srag_lab.silver.silver_srag_clean` |
# MAGIC | Granularidade de saída | Uma linha por UF |
# MAGIC | Filtro de qualidade | `sg_uf IS NOT NULL` |
# MAGIC | Cobertura temporal | Todo o histórico disponível na Silver |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 3. Definições Epidemiológicas
# MAGIC
# MAGIC ### 3.1 Taxa de Mortalidade (SRAG estrita)
# MAGIC
# MAGIC Inclui apenas casos com desfecho registrado. `EVOLUCAO = '3'` (em acompanhamento)
# MAGIC já foi excluído pela camada Silver e não aparece em `evolucao_clean`.
# MAGIC
# MAGIC ```
# MAGIC Denominador : evolucao_clean IN ('1', '2')
# MAGIC Numerador   : evolucao_clean = '2'
# MAGIC ```
# MAGIC
# MAGIC ### 3.2 Taxa de Ocupação UTI (hospital-based)
# MAGIC
# MAGIC O denominador é restrito à população internada. Casos ambulatoriais não integram
# MAGIC o cálculo. O campo `is_uti_valido` é definido na Silver e exclui indicadores de UTI
# MAGIC ausentes ou inconsistentes com o status de internação.
# MAGIC
# MAGIC ```
# MAGIC Denominador : SUM(CASE WHEN is_internado  THEN 1 END)
# MAGIC Numerador   : SUM(CASE WHEN is_uti_valido THEN 1 END)
# MAGIC ```
# MAGIC
# MAGIC ### 3.3 Taxa de Vacinação
# MAGIC
# MAGIC Exclui registros com status vacinal ausente (NULL). O campo `vacina_clean`
# MAGIC corresponde ao campo VACINA do SIVEP-Gripe (1=Sim, 2=Não, 9→NULL após tratamento Silver).
# MAGIC
# MAGIC ```
# MAGIC Denominador : vacina_clean IS NOT NULL
# MAGIC Numerador   : vacina_clean = '1'
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 4. Regras de Denominador
# MAGIC
# MAGIC | Métrica | Denominador | Justificativa |
# MAGIC |---|---|---|
# MAGIC | Mortalidade | `casos_com_desfecho` | Exclui casos sem desfecho registrado |
# MAGIC | UTI | `total_internados` | Métrica hospitalar; não se aplica a casos ambulatoriais |
# MAGIC | Vacinação | `casos_com_info_vacina` | Exclui ausência de informação vacinal |
# MAGIC | % Idosos | `total_casos` da UF | Proporção direta sobre o universo da UF |
# MAGIC | % Nacional | Total Silver (pré-filtro) | Preserva proporção real no universo nacional |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 5. Observações Técnicas
# MAGIC
# MAGIC - Apenas campos tratados da Silver são referenciados: `evolucao_clean`, `vacina_clean`,
# MAGIC   `idade_anos`, `is_internado`, `is_uti_valido`, `is_idoso`. Colunas brutas (`EVOLUCAO`,
# MAGIC   `VACINA`, `NU_IDADE_N`) não são utilizadas neste notebook.
# MAGIC - O ranking é calculado via `sort_values` + enumeração sequencial em Pandas
# MAGIC   (27 linhas — evita Window global sem partição no Spark Connect/Serverless).
# MAGIC   Desempate por `sg_uf` ascendente para resultado determinístico.
# MAGIC - Divisões por zero são tratadas com `F.when(...> 0).otherwise(None)`.

# COMMAND ----------

# MAGIC %run ./01_gold_setup

# COMMAND ----------

from pyspark.sql import functions as F
from datetime import datetime

# COMMAND ----------

# MAGIC %md
# MAGIC ## Carregamento de Configurações

# COMMAND ----------

CATALOG      = dbutils.widgets.get("catalog_gold")
SCHEMA_GOLD  = dbutils.widgets.get("schema_gold")
TABLE_SILVER = dbutils.widgets.get("table_silver")
PROCESS_ID   = dbutils.widgets.get("process_id")

# data_snapshot: registrado pelo 01_gold_setup quando executado via orquestrador.
# Fallback para current_date() quando o notebook roda standalone (desenvolvimento/debug).
try:
    DATA_SNAPSHOT = dbutils.widgets.get("data_snapshot")
except Exception:
    DATA_SNAPSHOT = spark.sql("SELECT current_date() AS d").collect()[0]["d"].isoformat()
    print(f"  AVISO: widget 'data_snapshot' nao encontrado — usando current_date(): {DATA_SNAPSHOT}")
    print(f"         Para rastreabilidade completa, execute via 00_pipeline_gold.")

TABLE_GOLD = f"{CATALOG}.{SCHEMA_GOLD}.gold_metricas_geograficas"

print(f"Fonte   : {TABLE_SILVER}")
print(f"Destino : {TABLE_GOLD}")
print(f"Process : {PROCESS_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Carregamento da Silver

# COMMAND ----------

df_silver = spark.table(TABLE_SILVER)

# Total nacional calculado antes do filtro de UF para preservar a proporção real.
total_nacional = df_silver.count()

print(f"Registros carregados : {total_nacional:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Filtro de Qualidade Geográfica

# COMMAND ----------

df_filtered = df_silver.filter(F.col("sg_uf").isNotNull())

ufs_validas = df_filtered.select('sg_uf').distinct().count()
print(f"UFs com sg_uf valido : {ufs_validas}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agregação por UF

# COMMAND ----------

for _flag in ('is_covid', 'is_influenza'):
    if _flag not in df_filtered.columns:
        df_filtered = df_filtered.withColumn(_flag, F.lit(False))

df_metricas_geograficas = df_filtered.groupBy('sg_uf').agg(

    # --- Contagens gerais ---
    F.count('*').alias('total_casos'),
    F.countDistinct('nu_notific').alias('casos_unicos'),
    F.countDistinct(
        F.when(F.col('co_mun_res').isNotNull(), F.col('co_mun_res'))
    ).alias('municipios_afetados'),

    # --- Idade (campo padronizado pela Silver) ---
    F.round(F.avg('idade_anos'), 1).alias('idade_media'),

    # --- Perfil de idosos ---
    F.sum(F.when(F.col('is_idoso'), 1).otherwise(0)).alias('casos_idosos'),
    F.round(
        F.sum(F.when(F.col('is_idoso'), 1).otherwise(0)) * 100.0 / F.count('*'),
        2
    ).alias('percentual_idosos'),

    # --- Distribuição por sexo ---
    F.sum(F.when(F.col('cs_sexo_clean') == '1', 1).otherwise(0)).alias('casos_masculino'),
    F.sum(F.when(F.col('cs_sexo_clean') == '2', 1).otherwise(0)).alias('casos_feminino'),

    # --- Mortalidade: denominador = desfecho registrado ---
    F.sum(F.when(F.col('evolucao_clean') == '2', 1).otherwise(0))
     .alias('total_obitos'),
    F.sum(F.when(F.col('evolucao_clean').isin('1', '2'), 1).otherwise(0))
     .alias('casos_com_desfecho'),

    F.round(
        F.when(
            F.sum(F.when(F.col('evolucao_clean').isin('1', '2'), 1).otherwise(0)) > 0,
            F.sum(F.when(F.col('evolucao_clean') == '2', 1).otherwise(0)) * 100.0 /
            F.sum(F.when(F.col('evolucao_clean').isin('1', '2'), 1).otherwise(0))
        ).otherwise(None),
        2
    ).alias('taxa_mortalidade'),

    # --- UTI: denominador = is_internado ---
    F.sum(F.when(F.col('is_internado'),  1).otherwise(0)).alias('total_internados'),
    F.sum(F.when(F.col('is_uti_valido'), 1).otherwise(0)).alias('total_uti'),

    F.round(
        F.when(
            F.sum(F.when(F.col('is_internado'), 1).otherwise(0)) > 0,
            F.sum(F.when(F.col('is_uti_valido'), 1).otherwise(0)) * 100.0 /
            F.sum(F.when(F.col('is_internado'),  1).otherwise(0))
        ).otherwise(None),
        2
    ).alias('taxa_uti'),

    # --- Vacinação: denominador = vacina_clean IS NOT NULL ---
    F.sum(F.when(F.col('vacina_clean') == '1', 1).otherwise(0))
     .alias('total_vacinados'),
    F.sum(F.when(F.col('vacina_clean').isNotNull(), 1).otherwise(0))
     .alias('casos_com_info_vacina'),

    F.round(
        F.when(
            F.sum(F.when(F.col('vacina_clean').isNotNull(), 1).otherwise(0)) > 0,
            F.sum(F.when(F.col('vacina_clean') == '1', 1).otherwise(0)) * 100.0 /
            F.sum(F.when(F.col('vacina_clean').isNotNull(), 1).otherwise(0))
        ).otherwise(None),
        2
    ).alias('taxa_vacinacao'),

    # --- Etiologia (com guard para Silver v1) ---
    F.sum(F.when(F.col('is_covid'),     1).otherwise(0)).alias('total_covid'),
    F.sum(F.when(F.col('is_influenza'), 1).otherwise(0)).alias('total_influenza'),

    # --- Período de notificação ---
    F.min('dt_sin_pri').alias('data_primeiro_caso'),
    F.max('dt_sin_pri').alias('data_ultimo_caso'),

    # --- Tempos clínicos (dias) ---
    F.round(F.avg('tempo_sintoma_notificacao'), 1).alias('tempo_medio_notificacao'),
    F.round(F.avg('tempo_internacao'),           1).alias('duracao_media_internacao'),

)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ranking e Percentual Nacional
# MAGIC
# MAGIC Implementação via Pandas (27 linhas): evita Window global sem partição,
# MAGIC que move todos os dados para uma única partição no Spark Connect/Serverless.

# COMMAND ----------

# Coleta as 27 UFs — volume mínimo, seguro para toPandas
import pandas as pd

pd_geo = df_metricas_geograficas.toPandas()

# Ranking determinístico por total_casos DESC; empates desempatados pela UF (alfabético)
pd_geo = pd_geo.sort_values(['total_casos', 'sg_uf'], ascending=[False, True])
pd_geo['ranking_casos'] = range(1, len(pd_geo) + 1)

# Percentual sobre total nacional (calculado antes do filtro de UF)
pd_geo['percentual_nacional'] = (pd_geo['total_casos'] * 100.0 / total_nacional).round(2)

# Reconverte para Spark — sem Window, sem warning
df_metricas_geograficas = spark.createDataFrame(pd_geo)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Metadados de Auditoria

# COMMAND ----------

df_metricas_geograficas = (
    df_metricas_geograficas
    .withColumn('_gold_processed_at', F.current_timestamp())
    .withColumn('_process_id',        F.lit(PROCESS_ID))
    .withColumn('data_snapshot',      F.lit(DATA_SNAPSHOT).cast('date'))
)

count_ufs = df_metricas_geograficas.count()
print(f"UFs geradas: {count_ufs}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gravação da Tabela Gold

# COMMAND ----------

(
    df_metricas_geograficas
    .write
    .mode('overwrite')
    .option('overwriteSchema', True)
    .saveAsTable(TABLE_GOLD)
)

print(f"Tabela gravada: {TABLE_GOLD}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verificação

# COMMAND ----------

(
    spark.table(TABLE_GOLD)
    .orderBy('ranking_casos')
    .select(
        'ranking_casos', 'sg_uf', 'total_casos',
        'percentual_nacional', 'taxa_mortalidade', 'taxa_uti',
        'total_covid', 'total_influenza', 'municipios_afetados'
    )
    .limit(10)
    .show(truncate=False)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resumo de Execução

# COMMAND ----------

count_final = spark.table(TABLE_GOLD).count()

print("=" * 80)
print("METRICAS GEOGRAFICAS — RESUMO")
print("=" * 80)
print(f"  Tabela  : {TABLE_GOLD}")
print(f"  UFs     : {count_final}")
print(f"  Process : {PROCESS_ID}")
print("=" * 80)
