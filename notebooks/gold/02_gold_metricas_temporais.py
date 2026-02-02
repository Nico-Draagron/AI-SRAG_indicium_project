# Databricks notebook source
# MAGIC %md
# MAGIC # 📅 Gold - Métricas Temporais
# MAGIC 
# MAGIC **Responsabilidade**: Criar `gold_metricas_temporais` com agregação mensal
# MAGIC 
# MAGIC **Métricas incluídas**:
# MAGIC - Taxa de Mortalidade
# MAGIC - Taxa de Ocupação UTI
# MAGIC - Taxa de Vacinação
# MAGIC - **Taxa de Crescimento** (mês a mês)
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

# Importar do notebook de setup
CATALOG = dbutils.widgets.get("catalog_gold")  # dbx_lab_draagron
SCHEMA_GOLD = dbutils.widgets.get("schema_gold")  # gold
TABLE_SILVER = dbutils.widgets.get("table_silver")
PROCESS_ID = dbutils.widgets.get("process_id")

TABLE_GOLD = f"{CATALOG}.{SCHEMA_GOLD}.gold_metricas_temporais"

print("📋 CONFIGURAÇÃO:")
print(f"  • Fonte: {TABLE_SILVER}")
print(f"  • Destino: {TABLE_GOLD}")
print(f"  • Process ID: {PROCESS_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📥 Carregar Dados Silver

# COMMAND ----------

df_silver = spark.table(TABLE_SILVER)
print(f"✅ Dados carregados: {df_silver.count():,} registros")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔄 Processar Agregação Mensal

# COMMAND ----------

print("\n📊 Criando agregação mensal...")

df_metricas_temporais = df_silver.groupBy('ano', 'mes', 'ano_mes').agg(
    # Contagens básicas
    F.count('*').alias('total_casos'),
    F.countDistinct('nu_notific').alias('casos_unicos'),
    
    # Demografia
    F.round(F.avg('nu_idade_n'), 1).alias('idade_media'),
    F.expr('percentile(nu_idade_n, 0.5)').alias('idade_mediana'),
    
    # MÉTRICA 1: Taxa de Mortalidade - Componentes
    F.sum(F.when(F.col('evolucao_clean') == '2', 1).otherwise(0)).alias('total_obitos'),
    F.sum(F.when(F.col('evolucao_clean') == '1', 1).otherwise(0)).alias('total_curas'),
    F.sum(F.when(F.col('evolucao_clean').isNotNull(), 1).otherwise(0)).alias('casos_com_desfecho'),
    
    # Taxa de Mortalidade calculada (SEM F.nullif)
    F.round(
        F.when(
            F.sum(F.when(F.col('evolucao_clean').isNotNull(), 1).otherwise(0)) > 0,
            F.sum(F.when(F.col('evolucao_clean') == '2', 1).otherwise(0)) * 100.0 /
            F.sum(F.when(F.col('evolucao_clean').isNotNull(), 1).otherwise(0))
        ).otherwise(None),
        2
    ).alias('taxa_mortalidade'),
    
    # MÉTRICA 2: Taxa de Ocupação UTI - Componentes
    F.sum(F.when(F.col('is_internado'), 1).otherwise(0)).alias('total_internados'),
    F.sum(F.when(F.col('is_uti'), 1).otherwise(0)).alias('total_uti'),
    
    # Taxa UTI calculada (SEM F.nullif)
    F.round(
        F.when(
            F.sum(F.when(F.col('is_internado'), 1).otherwise(0)) > 0,
            F.sum(F.when(F.col('is_uti'), 1).otherwise(0)) * 100.0 /
            F.sum(F.when(F.col('is_internado'), 1).otherwise(0))
        ).otherwise(None),
        2
    ).alias('taxa_uti'),
    
    # MÉTRICA 3: Taxa de Vacinação - Componentes
    F.sum(F.when(F.col('is_vacinado'), 1).otherwise(0)).alias('total_vacinados'),
    F.sum(F.when(F.col('vacina_clean').isNotNull(), 1).otherwise(0)).alias('casos_com_info_vacina'),
    
    # Taxa Vacinação calculada (SEM F.nullif)
    F.round(
        F.when(
            F.sum(F.when(F.col('vacina_clean').isNotNull(), 1).otherwise(0)) > 0,
            F.sum(F.when(F.col('is_vacinado'), 1).otherwise(0)) * 100.0 /
            F.sum(F.when(F.col('vacina_clean').isNotNull(), 1).otherwise(0))
        ).otherwise(None),
        2
    ).alias('taxa_vacinacao'),
    
    # Sintomas
    F.sum(F.when(F.col('has_febre'), 1).otherwise(0)).alias('casos_com_febre'),
    F.sum(F.when(F.col('has_tosse'), 1).otherwise(0)).alias('casos_com_tosse'),
    F.sum(F.when(F.col('has_dispneia'), 1).otherwise(0)).alias('casos_com_dispneia'),
    
    # Tempos médios
    F.round(F.avg('tempo_sintoma_notificacao'), 1).alias('tempo_medio_notificacao'),
    F.round(F.avg('tempo_sintoma_internacao'), 1).alias('tempo_medio_internacao'),
    F.round(F.avg('tempo_internacao'), 1).alias('duracao_media_internacao')
).orderBy('ano', 'mes')

# MÉTRICA 4: Taxa de Crescimento
window_crescimento = Window.orderBy('ano_mes')

df_metricas_temporais = df_metricas_temporais.withColumn(
    'casos_mes_anterior',
    F.lag('total_casos').over(window_crescimento)
).withColumn(
    'taxa_crescimento',
    F.round(
        F.when(
            F.col('casos_mes_anterior') > 0,
            (F.col('total_casos') - F.col('casos_mes_anterior')) * 100.0 / F.col('casos_mes_anterior')
        ).otherwise(None),
        2
    )
).drop('casos_mes_anterior')

# Metadados
df_metricas_temporais = df_metricas_temporais.withColumn(
    '_gold_processed_at',
    F.current_timestamp()
).withColumn(
    '_process_id',
    F.lit(PROCESS_ID)
).withColumn(
    'data_snapshot',
    F.current_date()
)

print(f"✅ Agregação criada: {df_metricas_temporais.count()} meses")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 💾 Salvar Tabela Gold

# COMMAND ----------

print(f"\n💾 Salvando: {TABLE_GOLD}")

df_metricas_temporais.write \
    .mode('overwrite') \
    .option('overwriteSchema', True) \
    .saveAsTable(TABLE_GOLD)

print("✅ Tabela salva com sucesso")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Preview

# COMMAND ----------

print("\n📋 Preview (últimos 6 meses):")
spark.table(TABLE_GOLD) \
    .orderBy(F.desc('ano_mes')) \
    .select('ano_mes', 'total_casos', 'taxa_mortalidade', 'taxa_uti', 
            'taxa_vacinacao', 'taxa_crescimento') \
    .limit(6) \
    .show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Resumo

# COMMAND ----------

count_final = spark.table(TABLE_GOLD).count()

print("\n" + "=" * 80)
print("✅ MÉTRICAS TEMPORAIS - CONCLUÍDO")
print("=" * 80)
print(f"  • Tabela: {TABLE_GOLD}")
print(f"  • Registros: {count_final}")
print(f"  • Métricas: 4 principais + demográficas + sintomas")
print("=" * 80)