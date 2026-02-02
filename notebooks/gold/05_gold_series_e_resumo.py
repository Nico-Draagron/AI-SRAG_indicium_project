# Databricks notebook source
# MAGIC %md
# MAGIC # 📈 Gold - Séries Temporais e Resumo Geral
# MAGIC 
# MAGIC **Responsabilidade**: 
# MAGIC - Criar `gold_series_temporais` (diária + semanal)
# MAGIC - Criar `gold_resumo_geral` (KPIs consolidados **com histórico**)
# MAGIC 
# MAGIC **Pré-requisito**: Execute `gold_setup` primeiro
# MAGIC 
# MAGIC ---

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import *
from datetime import datetime

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔥 Configurações

# COMMAND ----------

# Importar configurações dos widgets (CORRIGIDO)
CATALOG = dbutils.widgets.get("catalog_gold")
SCHEMA_GOLD = dbutils.widgets.get("schema_gold")
TABLE_SILVER = dbutils.widgets.get("table_silver")
PROCESS_ID = dbutils.widgets.get("process_id")

TABLE_SERIES = f"{CATALOG}.{SCHEMA_GOLD}.gold_series_temporais"
TABLE_RESUMO = f"{CATALOG}.{SCHEMA_GOLD}.gold_resumo_geral"

print("📋 CONFIGURAÇÃO:")
print(f"  • Catálogo Gold (OUTPUT): {CATALOG}")
print(f"  • Fonte: {TABLE_SILVER}")
print(f"  • Destino 1: {TABLE_SERIES}")
print(f"  • Destino 2: {TABLE_RESUMO}")
print(f"  • Process ID: {PROCESS_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📥 Carregar Dados

# COMMAND ----------

df_silver = spark.table(TABLE_SILVER)
print(f"✅ Dados carregados: {df_silver.count():,} registros")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📈 PARTE 1: Séries Temporais

# COMMAND ----------

print("\n📈 Criando séries temporais...")

# Série DIÁRIA
df_serie_diaria = df_silver.groupBy('dt_sin_pri').agg(
    F.count('*').alias('casos_dia'),
    F.sum(F.when(F.col('evolucao_clean') == '2', 1).otherwise(0)).alias('obitos_dia'),
    F.sum(F.when(F.col('is_internado'), 1).otherwise(0)).alias('internacoes_dia'),
    F.sum(F.when(F.col('is_uti'), 1).otherwise(0)).alias('uti_dia')
).withColumn(
    'tipo_agregacao',
    F.lit('diaria')
)

# Médias móveis (7 dias) - COM PARTITION para evitar warning
window_ma7 = Window.partitionBy(F.lit(1)).orderBy('dt_sin_pri').rowsBetween(-6, 0)

df_serie_diaria = df_serie_diaria.withColumn(
    'casos_ma7',
    F.round(F.avg('casos_dia').over(window_ma7), 1)
).withColumn(
    'obitos_ma7',
    F.round(F.avg('obitos_dia').over(window_ma7), 1)
)

# Série SEMANAL
df_serie_semanal = df_silver.filter(
    F.col('sem_pri').isNotNull()
).groupBy('ano', 'sem_pri').agg(
    F.count('*').alias('casos_semana'),
    F.sum(F.when(F.col('evolucao_clean') == '2', 1).otherwise(0)).alias('obitos_semana'),
    F.sum(F.when(F.col('is_internado'), 1).otherwise(0)).alias('internacoes_semana'),
    F.sum(F.when(F.col('is_uti'), 1).otherwise(0)).alias('uti_semana'),
    F.min('dt_sin_pri').alias('data_inicio_semana'),
    F.max('dt_sin_pri').alias('data_fim_semana')
).withColumn(
    'tipo_agregacao',
    F.lit('semanal')
)

# Metadados
df_serie_diaria = df_serie_diaria.withColumn(
    '_gold_processed_at', F.current_timestamp()
).withColumn('_process_id', F.lit(PROCESS_ID))

df_serie_semanal = df_serie_semanal.withColumn(
    '_gold_processed_at', F.current_timestamp()
).withColumn('_process_id', F.lit(PROCESS_ID))

# ✅ CORREÇÃO 1: Definir ordem FINAL das colunas (schema explícito)
colunas_finais = [
    'data_referencia',
    'ano',
    'sem_pri',
    
    'casos_dia',
    'obitos_dia',
    'internacoes_dia',
    'uti_dia',
    
    'casos_semana',
    'obitos_semana',
    'internacoes_semana',
    'uti_semana',
    
    'casos_ma7',
    'obitos_ma7',
    
    'data_inicio_semana',
    'data_fim_semana',
    
    'tipo_agregacao',
    '_gold_processed_at',
    '_process_id'
]

# ✅ CORREÇÃO 2: Preparar série DIÁRIA com schema completo
df_serie_diaria = df_serie_diaria \
    .withColumnRenamed('dt_sin_pri', 'data_referencia') \
    .withColumn('ano', F.year('data_referencia')) \
    .withColumn('sem_pri', F.lit(None).cast(IntegerType())) \
    .withColumn('casos_semana', F.lit(None).cast(LongType())) \
    .withColumn('obitos_semana', F.lit(None).cast(LongType())) \
    .withColumn('internacoes_semana', F.lit(None).cast(LongType())) \
    .withColumn('uti_semana', F.lit(None).cast(LongType())) \
    .withColumn('data_inicio_semana', F.lit(None).cast(DateType())) \
    .withColumn('data_fim_semana', F.lit(None).cast(DateType())) \
    .select(colunas_finais)

# ✅ CORREÇÃO 3: Preparar série SEMANAL com schema completo
df_serie_semanal = df_serie_semanal \
    .withColumn('data_referencia', F.col('data_inicio_semana')) \
    .withColumn('casos_dia', F.lit(None).cast(LongType())) \
    .withColumn('obitos_dia', F.lit(None).cast(LongType())) \
    .withColumn('internacoes_dia', F.lit(None).cast(LongType())) \
    .withColumn('uti_dia', F.lit(None).cast(LongType())) \
    .withColumn('casos_ma7', F.lit(None).cast(DoubleType())) \
    .withColumn('obitos_ma7', F.lit(None).cast(DoubleType())) \
    .select(colunas_finais)

# ✅ CORREÇÃO 4: UNION com unionByName (à prova de refactor)
df_series = df_serie_diaria.unionByName(df_serie_semanal)

print(f"✅ Séries criadas: {df_series.count()} registros")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 💾 Salvar Séries Temporais

# COMMAND ----------

print(f"\n💾 Salvando: {TABLE_SERIES}")

df_series.write \
    .mode('overwrite') \
    .partitionBy('tipo_agregacao') \
    .option('overwriteSchema', True) \
    .saveAsTable(TABLE_SERIES)

print("✅ Séries salvas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📋 PARTE 2: Resumo Geral (com Histórico)

# COMMAND ----------

print("\n📋 Criando resumo geral com histórico...")

# OTIMIZAÇÃO: Calcular todos os KPIs em UMA ÚNICA agregação
print("  • Calculando KPIs base (1 job Spark)...")

kpis_base = df_silver.agg(
    # Contagens básicas
    F.count('*').alias('total_casos'),
    F.countDistinct('sg_uf').alias('ufs_afetadas'),
    
    # Métricas epidemiológicas - Numeradores
    F.sum(F.when(F.col('evolucao_clean') == '2', 1).otherwise(0)).alias('total_obitos'),
    F.sum(F.when(F.col('evolucao_clean').isNotNull(), 1).otherwise(0)).alias('casos_com_desfecho'),
    F.sum(F.when(F.col('is_internado'), 1).otherwise(0)).alias('total_internacoes'),
    F.sum(F.when(F.col('is_uti'), 1).otherwise(0)).alias('total_uti'),
    F.sum(F.when(F.col('is_vacinado'), 1).otherwise(0)).alias('total_vacinados'),
    F.sum(F.when(F.col('vacina_clean').isNotNull(), 1).otherwise(0)).alias('casos_com_info_vacina'),
    
    # Demografia
    F.sum(F.when(F.col('is_idoso'), 1).otherwise(0)).alias('total_idosos'),
    F.avg('nu_idade_n').alias('idade_media'),
    
    # Tempos médios
    F.avg('tempo_sintoma_notificacao').alias('tempo_notif'),
    F.avg('tempo_internacao').alias('duracao_int'),
    
    # Taxas calculadas de forma segura (SEM divisão por zero)
    F.when(
        F.sum(F.when(F.col('evolucao_clean').isNotNull(), 1).otherwise(0)) > 0,
        F.round(
            F.sum(F.when(F.col('evolucao_clean') == '2', 1).otherwise(0)) * 100.0 /
            F.sum(F.when(F.col('evolucao_clean').isNotNull(), 1).otherwise(0)),
            2
        )
    ).alias('taxa_mortalidade'),
    
    F.when(
        F.sum(F.when(F.col('is_internado'), 1).otherwise(0)) > 0,
        F.round(
            F.sum(F.when(F.col('is_uti'), 1).otherwise(0)) * 100.0 /
            F.sum(F.when(F.col('is_internado'), 1).otherwise(0)),
            2
        )
    ).alias('taxa_uti'),
    
    F.when(
        F.sum(F.when(F.col('vacina_clean').isNotNull(), 1).otherwise(0)) > 0,
        F.round(
            F.sum(F.when(F.col('is_vacinado'), 1).otherwise(0)) * 100.0 /
            F.sum(F.when(F.col('vacina_clean').isNotNull(), 1).otherwise(0)),
            2
        )
    ).alias('taxa_vacinacao'),
    
    F.round(
        F.sum(F.when(F.col('is_idoso'), 1).otherwise(0)) * 100.0 / F.count('*'),
        2
    ).alias('pct_idosos')
    
).collect()[0]

print(f"  ✓ KPIs calculados com sucesso")

# Montar lista de KPIs (RAG-ready com escopo)
data_snapshot = datetime.now().date()
escopo = 'Nacional'  # Campo para RAG

resumo_geral = [
    # Categoria: Casos
    {
        'categoria': 'Casos',
        'metrica': 'Total de Casos',
        'valor': float(kpis_base['total_casos']),
        'unidade': 'casos',
        'descricao': 'Número total de notificações SRAG',
        'escopo': escopo
    },
    
    # Categoria: Mortalidade
    {
        'categoria': 'Mortalidade',
        'metrica': 'Taxa de Mortalidade (%)',
        'valor': kpis_base['taxa_mortalidade'] if kpis_base['taxa_mortalidade'] else 0.0,
        'unidade': '%',
        'descricao': 'Percentual de óbitos entre casos com desfecho',
        'escopo': escopo
    },
    {
        'categoria': 'Mortalidade',
        'metrica': 'Total de Óbitos',
        'valor': float(kpis_base['total_obitos']),
        'unidade': 'casos',
        'descricao': 'Óbitos confirmados',
        'escopo': escopo
    },
    
    # Categoria: UTI
    {
        'categoria': 'UTI',
        'metrica': 'Taxa de Ocupação UTI (%)',
        'valor': kpis_base['taxa_uti'] if kpis_base['taxa_uti'] else 0.0,
        'unidade': '%',
        'descricao': 'Percentual de internados em UTI',
        'escopo': escopo
    },
    {
        'categoria': 'UTI',
        'metrica': 'Total em UTI',
        'valor': float(kpis_base['total_uti']),
        'unidade': 'casos',
        'descricao': 'Casos em UTI',
        'escopo': escopo
    },
    
    # Categoria: Vacinação
    {
        'categoria': 'Vacinacao',
        'metrica': 'Taxa de Vacinação (%)',
        'valor': kpis_base['taxa_vacinacao'] if kpis_base['taxa_vacinacao'] else 0.0,
        'unidade': '%',
        'descricao': 'Percentual de vacinados',
        'escopo': escopo
    },
    
    # Categoria: Internação
    {
        'categoria': 'Internacao',
        'metrica': 'Total de Internações',
        'valor': float(kpis_base['total_internacoes']),
        'unidade': 'casos',
        'descricao': 'Casos hospitalizados',
        'escopo': escopo
    },
    
    # Categoria: Demografia
    {
        'categoria': 'Demografia',
        'metrica': 'Idade Média (anos)',
        'valor': round(kpis_base['idade_media'], 1) if kpis_base['idade_media'] else 0.0,
        'unidade': 'anos',
        'descricao': 'Idade média dos casos',
        'escopo': escopo
    },
    {
        'categoria': 'Demografia',
        'metrica': 'Percentual de Idosos (%)',
        'valor': kpis_base['pct_idosos'],
        'unidade': '%',
        'descricao': 'Casos com 60+ anos',
        'escopo': escopo
    },
    
    # Categoria: Geografia
    {
        'categoria': 'Geografia',
        'metrica': 'UFs Afetadas',
        'valor': float(kpis_base['ufs_afetadas']),
        'unidade': 'UFs',
        'descricao': 'Estados com casos',
        'escopo': escopo
    },
    
    # Categoria: Tempo
    {
        'categoria': 'Tempo',
        'metrica': 'Tempo Médio Notificação (dias)',
        'valor': round(kpis_base['tempo_notif'], 1) if kpis_base['tempo_notif'] else 0.0,
        'unidade': 'dias',
        'descricao': 'Sintomas até notificação',
        'escopo': escopo
    },
    {
        'categoria': 'Tempo',
        'metrica': 'Duração Média Internação (dias)',
        'valor': round(kpis_base['duracao_int'], 1) if kpis_base['duracao_int'] else 0.0,
        'unidade': 'dias',
        'descricao': 'Dias de internação',
        'escopo': escopo
    }
]

# Criar DataFrame
df_resumo = spark.createDataFrame(resumo_geral)

# Adicionar metadados e data snapshot
df_resumo = df_resumo.withColumn(
    '_gold_processed_at', F.current_timestamp()
).withColumn(
    '_process_id', F.lit(PROCESS_ID)
).withColumn(
    'data_snapshot', F.lit(data_snapshot)
)

print(f"✅ Resumo criado: {df_resumo.count()} KPIs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 💾 Salvar Resumo Geral (APPEND para histórico)

# COMMAND ----------

print(f"\n💾 Salvando: {TABLE_RESUMO}")

# Primeira execução: OVERWRITE
# Execuções seguintes: APPEND para histórico
try:
    spark.table(TABLE_RESUMO)
    mode = 'append'
    print("  • Modo: APPEND (mantendo histórico)")
except:
    mode = 'overwrite'
    print("  • Modo: OVERWRITE (primeira execução)")

df_resumo.write \
    .mode(mode) \
    .option('mergeSchema', True) \
    .saveAsTable(TABLE_RESUMO)

print("✅ Resumo salvo")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Preview

# COMMAND ----------

print("\n📋 Preview - Séries (últimos 7 dias):")
spark.table(TABLE_SERIES) \
    .filter(F.col('tipo_agregacao') == 'diaria') \
    .orderBy(F.desc('data_referencia')) \
    .select('data_referencia', 'casos_dia', 'casos_ma7', 'obitos_dia') \
    .limit(7) \
    .show()

print("\n📋 Preview - Resumo (KPIs atuais):")
spark.table(TABLE_RESUMO) \
    .filter(F.col('data_snapshot') == data_snapshot) \
    .select('categoria', 'metrica', 'valor', 'unidade', 'escopo') \
    .show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Resumo

# COMMAND ----------

print("\n" + "=" * 80)
print("✅ SÉRIES E RESUMO - CONCLUÍDO")
print("=" * 80)
print(f"  • Séries Temporais: {TABLE_SERIES}")
print(f"  • Resumo Geral: {TABLE_RESUMO} (com histórico)")
print(f"  • KPIs: {len(resumo_geral)} métricas calculadas")
print(f"  • Performance: 1 job Spark (otimizado)")
print("=" * 80)