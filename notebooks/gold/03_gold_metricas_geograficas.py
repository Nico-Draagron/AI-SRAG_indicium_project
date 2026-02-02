# Databricks notebook source
# MAGIC %md
# MAGIC # 🗺️ Gold - Métricas Geográficas
# MAGIC 
# MAGIC **Responsabilidade**: Criar `gold_metricas_geograficas` com agregação por UF
# MAGIC 
# MAGIC **Incluído**:
# MAGIC - Ranking de UFs
# MAGIC - Percentual nacional
# MAGIC - 4 métricas epidemiológicas
# MAGIC - Demografia por região
# MAGIC 
# MAGIC **Pré-requisito**: Execute `gold_setup` primeiro
# MAGIC 
# MAGIC ---

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import Window
from datetime import datetime

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📥 Carregar Configurações

# COMMAND ----------

# Importar configurações dos widgets (CORRIGIDO)
CATALOG = dbutils.widgets.get("catalog_gold")
SCHEMA_GOLD = dbutils.widgets.get("schema_gold")
TABLE_SILVER = dbutils.widgets.get("table_silver")
PROCESS_ID = dbutils.widgets.get("process_id")

TABLE_GOLD = f"{CATALOG}.{SCHEMA_GOLD}.gold_metricas_geograficas"

print("📋 CONFIGURAÇÃO:")
print(f"  • Catálogo Gold (OUTPUT): {CATALOG}")
print(f"  • Fonte: {TABLE_SILVER}")
print(f"  • Destino: {TABLE_GOLD}")
print(f"  • Process ID: {PROCESS_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📥 Carregar Dados

# COMMAND ----------

df_silver = spark.table(TABLE_SILVER)
total_nacional = df_silver.count()

print(f"✅ Dados carregados: {total_nacional:,} registros")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🗺️ Processar Agregação por UF

# COMMAND ----------

print("\n🗺️ Criando agregação geográfica...")

# Filtro de qualidade (blindagem - evita UF NULL no ranking)
df_filtered = df_silver.filter(F.col("sg_uf").isNotNull())

print(f"  • UFs válidas: {df_filtered.select('sg_uf').distinct().count()}")

df_metricas_geograficas = df_filtered.groupBy('sg_uf').agg(
    # Contagens
    F.count('*').alias('total_casos'),
    F.countDistinct('nu_notific').alias('casos_unicos'),
    F.countDistinct('co_mun_res').alias('municipios_afetados'),
    
    # Demografia
    F.round(F.avg('nu_idade_n'), 1).alias('idade_media'),
    F.sum(F.when(F.col('is_idoso'), 1).otherwise(0)).alias('casos_idosos'),
    
    # Percentual de idosos (inline - sempre seguro pois F.count('*') > 0)
    F.round(
        F.sum(F.when(F.col('is_idoso'), 1).otherwise(0)) * 100.0 / F.count('*'),
        2
    ).alias('percentual_idosos'),
    
    # Distribuição por sexo
    F.sum(F.when(F.col('cs_sexo_clean') == '1', 1).otherwise(0)).alias('casos_masculino'),
    F.sum(F.when(F.col('cs_sexo_clean') == '2', 1).otherwise(0)).alias('casos_feminino'),
    
    # Componentes das 4 Métricas Epidemiológicas
    F.sum(F.when(F.col('evolucao_clean') == '2', 1).otherwise(0)).alias('total_obitos'),
    F.sum(F.when(F.col('evolucao_clean').isNotNull(), 1).otherwise(0)).alias('casos_com_desfecho'),
    
    # Taxa de Mortalidade (SEM F.nullif - PySpark puro)
    F.round(
        F.when(
            F.sum(F.when(F.col('evolucao_clean').isNotNull(), 1).otherwise(0)) > 0,
            F.sum(F.when(F.col('evolucao_clean') == '2', 1).otherwise(0)) * 100.0 /
            F.sum(F.when(F.col('evolucao_clean').isNotNull(), 1).otherwise(0))
        ).otherwise(None),
        2
    ).alias('taxa_mortalidade'),
    
    F.sum(F.when(F.col('is_internado'), 1).otherwise(0)).alias('total_internados'),
    F.sum(F.when(F.col('is_uti'), 1).otherwise(0)).alias('total_uti'),
    
    # Taxa UTI (SEM F.nullif - PySpark puro)
    F.round(
        F.when(
            F.sum(F.when(F.col('is_internado'), 1).otherwise(0)) > 0,
            F.sum(F.when(F.col('is_uti'), 1).otherwise(0)) * 100.0 /
            F.sum(F.when(F.col('is_internado'), 1).otherwise(0))
        ).otherwise(None),
        2
    ).alias('taxa_uti'),
    
    F.sum(F.when(F.col('is_vacinado'), 1).otherwise(0)).alias('total_vacinados'),
    F.sum(F.when(F.col('vacina_clean').isNotNull(), 1).otherwise(0)).alias('casos_com_info_vacina'),
    
    # Taxa Vacinação (SEM F.nullif - PySpark puro)
    F.round(
        F.when(
            F.sum(F.when(F.col('vacina_clean').isNotNull(), 1).otherwise(0)) > 0,
            F.sum(F.when(F.col('is_vacinado'), 1).otherwise(0)) * 100.0 /
            F.sum(F.when(F.col('vacina_clean').isNotNull(), 1).otherwise(0))
        ).otherwise(None),
        2
    ).alias('taxa_vacinacao'),
    
    # Período
    F.min('dt_sin_pri').alias('data_primeiro_caso'),
    F.max('dt_sin_pri').alias('data_ultimo_caso'),
    
    # Tempos médios
    F.round(F.avg('tempo_sintoma_notificacao'), 1).alias('tempo_medio_notificacao'),
    F.round(F.avg('tempo_internacao'), 1).alias('duracao_media_internacao')
)

# Adicionar ranking (com partitionBy explícito para evitar warning)
window_ranking = Window.partitionBy().orderBy(F.desc('total_casos'))

df_metricas_geograficas = df_metricas_geograficas.withColumn(
    'ranking_casos',
    F.row_number().over(window_ranking)
)

# Percentual nacional
df_metricas_geograficas = df_metricas_geograficas.withColumn(
    'percentual_nacional',
    F.round(F.col('total_casos') * 100.0 / total_nacional, 2)
)

# Metadados
df_metricas_geograficas = df_metricas_geograficas.withColumn(
    '_gold_processed_at',
    F.current_timestamp()
).withColumn(
    '_process_id',
    F.lit(PROCESS_ID)
).withColumn(
    'data_snapshot',
    F.current_date()
)

print(f"✅ Agregação criada: {df_metricas_geograficas.count()} UFs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 💾 Salvar Tabela

# COMMAND ----------

print(f"\n💾 Salvando: {TABLE_GOLD}")

df_metricas_geograficas.write \
    .mode('overwrite') \
    .option('overwriteSchema', True) \
    .saveAsTable(TABLE_GOLD)

print("✅ Tabela salva")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Preview

# COMMAND ----------

print("\n📋 Preview (Top 10 UFs):")
spark.table(TABLE_GOLD) \
    .orderBy('ranking_casos') \
    .select('ranking_casos', 'sg_uf', 'total_casos', 'percentual_nacional',
            'taxa_mortalidade', 'taxa_uti') \
    .limit(10) \
    .show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Resumo

# COMMAND ----------

count_final = spark.table(TABLE_GOLD).count()

print("\n" + "=" * 80)
print("✅ MÉTRICAS GEOGRÁFICAS - CONCLUÍDO")
print("=" * 80)
print(f"  • Tabela: {TABLE_GOLD}")
print(f"  • UFs: {count_final}")
print(f"  • Métricas: 4 principais + demografia + tempos médios")
print("=" * 80)