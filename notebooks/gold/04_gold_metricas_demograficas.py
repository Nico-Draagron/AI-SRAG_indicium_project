# Databricks notebook source
# MAGIC %md
# MAGIC # 👥 Gold - Métricas Demográficas
# MAGIC 
# MAGIC **Responsabilidade**: Criar `gold_metricas_demograficas` com agregação por faixa etária e sexo
# MAGIC 
# MAGIC **Análises**:
# MAGIC - Grupos de risco
# MAGIC - Perfil etário
# MAGIC - Diferenças por sexo
# MAGIC 
# MAGIC **Pré-requisito**: Execute `gold_setup` primeiro
# MAGIC 
# MAGIC ---

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from datetime import datetime

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📥 Configurações

# COMMAND ----------

# Importar configurações dos widgets (CORRIGIDO)
CATALOG = dbutils.widgets.get("catalog_gold")
SCHEMA_GOLD = dbutils.widgets.get("schema_gold")
TABLE_SILVER = dbutils.widgets.get("table_silver")
PROCESS_ID = dbutils.widgets.get("process_id")

TABLE_GOLD = f"{CATALOG}.{SCHEMA_GOLD}.gold_metricas_demograficas"

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
print(f"✅ Dados carregados: {df_silver.count():,} registros")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 👥 Processar Agregação Demográfica

# COMMAND ----------

print("\n👥 Criando agregação demográfica...")

# Filtrar dados válidos (blindagem de qualidade)
df_filtered = df_silver.filter(
    (F.col('faixa_etaria') != 'Desconhecido') &
    (F.col('cs_sexo_clean').isin('1', '2'))  # Apenas Masculino e Feminino
)

print(f"  • Registros válidos: {df_filtered.count():,}")

df_metricas_demograficas = df_filtered.groupBy('faixa_etaria', 'cs_sexo_clean').agg(
    # Contagens
    F.count('*').alias('total_casos'),
    
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
    
    # Taxa de Internação (calculada inline - sempre segura pois F.count('*') > 0)
    F.round(
        F.sum(F.when(F.col('is_internado'), 1).otherwise(0)) * 100.0 / F.count('*'),
        2
    ).alias('taxa_internacao'),
    
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
    
    # Tempo médio
    F.round(F.avg('tempo_internacao'), 1).alias('duracao_media_internacao')
)

# Label de sexo
df_metricas_demograficas = df_metricas_demograficas.withColumn(
    'sexo_label',
    F.when(F.col('cs_sexo_clean') == '1', 'Masculino')
    .when(F.col('cs_sexo_clean') == '2', 'Feminino')
    .otherwise('Não informado')
)

# Percentual do total (usando Window - mais eficiente, sem collect())
total_window = Window.partitionBy()

df_metricas_demograficas = df_metricas_demograficas.withColumn(
    'percentual_total',
    F.round(
        F.col('total_casos') * 100.0 / F.sum('total_casos').over(total_window),
        2
    )
)

# Ordem de faixa etária
df_metricas_demograficas = df_metricas_demograficas.withColumn(
    'ordem_faixa',
    F.when(F.col('faixa_etaria') == '0-1 ano', 1)
    .when(F.col('faixa_etaria') == '1-4 anos', 2)
    .when(F.col('faixa_etaria') == '5-9 anos', 3)
    .when(F.col('faixa_etaria') == '10-17 anos', 4)
    .when(F.col('faixa_etaria') == '18-29 anos', 5)
    .when(F.col('faixa_etaria') == '30-39 anos', 6)
    .when(F.col('faixa_etaria') == '40-49 anos', 7)
    .when(F.col('faixa_etaria') == '50-59 anos', 8)
    .when(F.col('faixa_etaria') == '60-69 anos', 9)
    .when(F.col('faixa_etaria') == '70+ anos', 10)
    .otherwise(99)
)

# Metadados
df_metricas_demograficas = df_metricas_demograficas.withColumn(
    '_gold_processed_at',
    F.current_timestamp()
).withColumn(
    '_process_id',
    F.lit(PROCESS_ID)
).withColumn(
    'data_snapshot',
    F.current_date()
)

print(f"✅ Agregação criada: {df_metricas_demograficas.count()} combinações (faixa × sexo)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 💾 Salvar Tabela

# COMMAND ----------

print(f"\n💾 Salvando: {TABLE_GOLD}")

df_metricas_demograficas.write \
    .mode('overwrite') \
    .option('overwriteSchema', True) \
    .saveAsTable(TABLE_GOLD)

print("✅ Tabela salva")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Preview

# COMMAND ----------

print("\n📋 Preview (por faixa etária):")
spark.table(TABLE_GOLD) \
    .orderBy('ordem_faixa', 'sexo_label') \
    .select('faixa_etaria', 'sexo_label', 'total_casos', 'percentual_total',
            'taxa_mortalidade', 'taxa_internacao') \
    .limit(15) \
    .show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Resumo

# COMMAND ----------

count_final = spark.table(TABLE_GOLD).count()

print("\n" + "=" * 80)
print("✅ MÉTRICAS DEMOGRÁFICAS - CONCLUÍDO")
print("=" * 80)
print(f"  • Tabela: {TABLE_GOLD}")
print(f"  • Combinações: {count_final}")
print(f"  • Métricas: 4 principais + tempo médio internação")
print("=" * 80)