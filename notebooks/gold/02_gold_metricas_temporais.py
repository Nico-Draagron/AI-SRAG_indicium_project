# Databricks notebook source
# MAGIC %md
# MAGIC # Gold — Métricas Temporais
# MAGIC
# MAGIC ## 1. Objetivo do Notebook
# MAGIC
# MAGIC Produzir três tabelas Gold com diferentes janelas temporais:
# MAGIC
# MAGIC - `gold_metricas_temporais` — agregação mensal dos **últimos 12 meses** (relatório corrente).
# MAGIC - `gold_serie_diaria_30d` — série diária dos **últimos 30 dias** (gráfico de tendência imediata).
# MAGIC - `gold_metricas_historicas` — agregação mensal de **todo o histórico disponível na Silver**
# MAGIC   (consultas comparativas entre anos: 2023 vs 2024 vs 2025 e além).
# MAGIC
# MAGIC Quando novos CSVs anuais forem carregados na Silver, a tabela histórica os inclui
# MAGIC automaticamente na próxima execução do pipeline — sem alteração de código.
# MAGIC
# MAGIC Pré-requisito: o notebook `01_gold_setup` deve ter sido executado na mesma sessão,
# MAGIC pois este notebook consome os widgets por ele registrados.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 2. Fonte de Dados
# MAGIC
# MAGIC | Atributo | Valor |
# MAGIC |---|---|
# MAGIC | Tabela | `dbx_srag_lab.silver.silver_srag_clean` |
# MAGIC | Granularidade | Um registro por notificação de SRAG |
# MAGIC | Filtros aplicados na Silver | `EVOLUCAO = '3'` (em acompanhamento) já excluído |
# MAGIC
# MAGIC Nenhum filtro adicional de desfecho é aplicado neste notebook.
# MAGIC A exclusão de EVOLUCAO='3' é garantida pela camada Silver.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 3. Regras de Métrica
# MAGIC
# MAGIC ### 3.1 Taxa de Mortalidade (definição estrita)
# MAGIC
# MAGIC Inclui apenas casos com desfecho registrado.
# MAGIC
# MAGIC ```
# MAGIC Denominador : evolucao_clean IN ('1', '2')
# MAGIC Numerador   : evolucao_clean = '2'
# MAGIC ```
# MAGIC
# MAGIC ### 3.2 Taxa de Ocupação UTI
# MAGIC
# MAGIC Calculada sobre a população hospitalar. O denominador é o total de internados,
# MAGIC não o total de casos do período.
# MAGIC
# MAGIC ```
# MAGIC Denominador : SUM(CASE WHEN is_internado     THEN 1 END)
# MAGIC Numerador   : SUM(CASE WHEN is_uti_valido    THEN 1 END)
# MAGIC ```
# MAGIC
# MAGIC O campo `is_uti_valido` é definido na Silver e exclui registros com indicador de UTI
# MAGIC ausente ou inconsistente com o status de internação.
# MAGIC
# MAGIC ### 3.3 Taxa de Vacinação
# MAGIC
# MAGIC Exclui registros com status vacinal ausente (NULL) ou ignorado (valor '9').
# MAGIC
# MAGIC ```
# MAGIC Denominador : vacina_clean IS NOT NULL   -- exclui NULL e ignorado
# MAGIC Numerador   : vacina_clean = '1'         -- vacinado (SIVEP-Gripe: 1=Sim)
# MAGIC ```
# MAGIC
# MAGIC ### 3.4 Idade
# MAGIC
# MAGIC Utilizar exclusivamente `idade_anos`. Os campos brutos `NU_IDADE_N` e `CS_IDADE_UN`
# MAGIC não devem ser referenciados em nenhuma agregação Gold.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 4. Regra de Crescimento Mensal
# MAGIC
# MAGIC A taxa de crescimento mês a mês é calculada pela variação relativa simples:
# MAGIC
# MAGIC ```
# MAGIC taxa_crescimento(t) = ( casos(t) - casos(t-1) ) / casos(t-1)  * 100
# MAGIC ```
# MAGIC
# MAGIC Restrições de aplicação:
# MAGIC - A janela de ordenação usa `ano_mes_date` (tipo `DATE`), não a string `ano_mes`.
# MAGIC - O valor é `NULL` para o primeiro mês da série (sem período anterior).
# MAGIC - O valor é `NULL` quando `casos(t-1) = 0` para evitar divisão por zero.
# MAGIC - O cálculo considera apenas os últimos 12 meses na tabela `gold_metricas_temporais`.
# MAGIC - Na tabela `gold_metricas_historicas`, o mesmo cálculo é aplicado sobre toda a série histórica.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 5. Estrutura da Tabela Gerada
# MAGIC
# MAGIC | Coluna | Tipo | Descrição |
# MAGIC |---|---|---|
# MAGIC | `ano` | INT | Ano de início dos sintomas |
# MAGIC | `mes` | INT | Mês de início dos sintomas |
# MAGIC | `ano_mes` | STRING | Período no formato `YYYY-MM` |
# MAGIC | `ano_mes_date` | DATE | Primeiro dia do mês (ordenação) |
# MAGIC | `total_casos` | BIGINT | Total de notificações no período |
# MAGIC | `casos_unicos` | BIGINT | Notificações com `nu_notific` distinto |
# MAGIC | `idade_media` | DOUBLE | Média de `idade_anos` |
# MAGIC | `idade_mediana` | DOUBLE | Percentil 50 de `idade_anos` (approx) |
# MAGIC | `total_obitos` | BIGINT | `evolucao_clean = '2'` |
# MAGIC | `total_curas` | BIGINT | `evolucao_clean = '1'` |
# MAGIC | `casos_com_desfecho` | BIGINT | `evolucao_clean IN ('1','2')` |
# MAGIC | `taxa_mortalidade` | DOUBLE | Percentual sobre `casos_com_desfecho` |
# MAGIC | `total_internados` | BIGINT | `is_internado = TRUE` |
# MAGIC | `total_uti` | BIGINT | `is_uti_valido = TRUE` |
# MAGIC | `taxa_uti` | DOUBLE | Percentual sobre `total_internados` |
# MAGIC | `total_vacinados` | BIGINT | `vacina_clean = '1'` |
# MAGIC | `casos_com_info_vacina` | BIGINT | `vacina_clean IS NOT NULL` |
# MAGIC | `taxa_vacinacao` | DOUBLE | Percentual sobre `casos_com_info_vacina` |
# MAGIC | `casos_com_febre` | BIGINT | `has_febre = TRUE` |
# MAGIC | `casos_com_tosse` | BIGINT | `has_tosse = TRUE` |
# MAGIC | `casos_com_dispneia` | BIGINT | `has_dispneia = TRUE` |
# MAGIC | `tempo_medio_notificacao` | DOUBLE | Média de `tempo_sintoma_notificacao` (dias) |
# MAGIC | `tempo_medio_internacao` | DOUBLE | Média de `tempo_sintoma_internacao` (dias) |
# MAGIC | `duracao_media_internacao` | DOUBLE | Média de `tempo_internacao` (dias) |
# MAGIC | `taxa_crescimento` | DOUBLE | Crescimento percentual sobre mês anterior |
# MAGIC | `_gold_processed_at` | TIMESTAMP | Timestamp de gravação |
# MAGIC | `_process_id` | STRING | Identificador da execução (`01_gold_setup`) |
# MAGIC | `data_snapshot` | DATE | Data de execução do pipeline |

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

TABLE_GOLD_TEMPORAIS  = f"{CATALOG}.{SCHEMA_GOLD}.gold_metricas_temporais"
TABLE_GOLD_DIARIA_30D = f"{CATALOG}.{SCHEMA_GOLD}.gold_serie_diaria_30d"
TABLE_GOLD_HISTORICA  = f"{CATALOG}.{SCHEMA_GOLD}.gold_metricas_historicas"

print(f"Fonte              : {TABLE_SILVER}")
print(f"Destino temporais  : {TABLE_GOLD_TEMPORAIS}")
print(f"Destino diario 30d : {TABLE_GOLD_DIARIA_30D}")
print(f"Destino historico  : {TABLE_GOLD_HISTORICA}")
print(f"Process            : {PROCESS_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Carregamento da Silver

# COMMAND ----------

df_silver = spark.table(TABLE_SILVER)
print(f"Registros carregados: {df_silver.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Filtro: Últimos 12 Meses

# COMMAND ----------

# Determina o período máximo disponível na Silver e restringe a 12 meses.
max_ano_mes = df_silver.agg(F.max('ano_mes_date')).collect()[0][0]

df_silver_12m = df_silver.filter(
    F.col('ano_mes_date') >= F.add_months(F.lit(max_ano_mes), -11)
)

print(f"Periodo maximo na Silver : {max_ano_mes}")
print(f"Janela aplicada          : últimos 12 meses")
print(f"Registros no periodo     : {df_silver_12m.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agregação Mensal

# COMMAND ----------

for _flag in ('is_covid','is_influenza','is_outro_virus','classi_fin_clean'):
    if _flag not in df_silver_12m.columns:
        df_silver_12m = df_silver_12m.withColumn(_flag, F.lit(None).cast(
            'boolean' if 'is_' in _flag else 'string'))

# Função reutilizada pela agregação de 12 meses E pela histórica.
# Recebe qualquer DataFrame com a estrutura da Silver e retorna o agg mensal.
def _agregar_mensal(df):
    return df.groupBy('ano', 'mes', 'ano_mes', 'ano_mes_date').agg(

    # --- Contagens gerais ---
    F.count('*').alias('total_casos'),
    F.countDistinct('nu_notific').alias('casos_unicos'),

    # Breakdown etiológico (Silver v2: is_covid, is_influenza, is_outro_virus)
    # Guard: usa 0 como fallback se coluna não existir (Silver v1 sem flags)
    F.sum(F.when(F.col('is_covid'),       1).otherwise(0)).alias('total_covid'),
    F.sum(F.when(F.col('is_influenza'),   1).otherwise(0)).alias('total_influenza'),
    F.sum(F.when(F.col('is_outro_virus'), 1).otherwise(0)).alias('total_outro_virus'),
    F.sum(F.when(F.col('classi_fin_clean').isNull(), 1).otherwise(0))
     .alias('total_sem_classificacao'),

    # --- Idade (campo padronizado pela Silver) ---
    F.round(F.avg('idade_anos'), 1).alias('idade_media'),
    F.round(F.percentile_approx('idade_anos', 0.5).cast('double'), 1).alias('idade_mediana'),

    # --- Mortalidade: denominador = desfecho registrado ---
    F.sum(F.when(F.col('evolucao_clean') == '2', 1).otherwise(0))
     .alias('total_obitos'),
    F.sum(F.when(F.col('evolucao_clean') == '1', 1).otherwise(0))
     .alias('total_curas'),
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
    F.sum(F.when(F.col('is_internado'), 1).otherwise(0))
     .alias('total_internados'),
    F.sum(F.when(F.col('is_uti_valido'), 1).otherwise(0))
     .alias('total_uti'),

    F.round(
        F.when(
            F.sum(F.when(F.col('is_internado'), 1).otherwise(0)) > 0,
            F.sum(F.when(F.col('is_uti_valido'), 1).otherwise(0)) * 100.0 /
            F.sum(F.when(F.col('is_internado'), 1).otherwise(0))
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

    # --- Sintomas ---
    F.sum(F.when(F.col('has_febre'),    1).otherwise(0)).alias('casos_com_febre'),
    F.sum(F.when(F.col('has_tosse'),    1).otherwise(0)).alias('casos_com_tosse'),
    F.sum(F.when(F.col('has_dispneia'), 1).otherwise(0)).alias('casos_com_dispneia'),

    # --- Tempos clínicos (dias) ---
    F.round(F.avg('tempo_sintoma_notificacao'), 1).alias('tempo_medio_notificacao'),
    F.round(F.avg('tempo_sintoma_internacao'),  1).alias('tempo_medio_internacao'),
    F.round(F.avg('tempo_internacao'),           1).alias('duracao_media_internacao'),

    ).orderBy('ano', 'mes')

df_metricas_temporais = _agregar_mensal(df_silver_12m)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Taxa de Crescimento Mensal
# MAGIC
# MAGIC Fórmula: `( casos(t) - casos(t-1) ) / casos(t-1) * 100`
# MAGIC
# MAGIC Ordenação por `ano_mes_date`. Resultado NULL para o primeiro período da série
# MAGIC e quando o mês anterior registra zero casos.
# MAGIC
# MAGIC Implementação via Pandas: evita Window global sem partição,
# MAGIC que move todos os dados para uma única partição no Spark Connect/Serverless.
# MAGIC Usada tanto para os 12 meses quanto para o histórico completo.

# COMMAND ----------

# Função reutilizada para calcular taxa_crescimento em qualquer série mensal ordenada.
def _calcular_crescimento(df_spark):
    """
    Recebe um DataFrame Spark com coluna total_casos e ano_mes_date.
    Retorna o mesmo DataFrame com a coluna taxa_crescimento adicionada.
    Implementado em Pandas para evitar Window global sem partição.
    """
    import math
    pd_df = df_spark.orderBy('ano_mes_date').toPandas()
    pd_df['_casos_anterior'] = pd_df['total_casos'].shift(1)
    pd_df['taxa_crescimento'] = pd_df.apply(
        lambda r: round((r['total_casos'] - r['_casos_anterior'])
                        / r['_casos_anterior'] * 100, 2)
        if (not math.isnan(r['_casos_anterior']) and r['_casos_anterior'] > 0)
        else None,
        axis=1
    )
    pd_df = pd_df.drop(columns=['_casos_anterior'])
    return spark.createDataFrame(pd_df)

# Aplica crescimento nos 12 meses recentes
df_metricas_temporais = _calcular_crescimento(df_metricas_temporais)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Metadados de Auditoria

# COMMAND ----------

df_metricas_temporais = (
    df_metricas_temporais
    .withColumn('_gold_processed_at', F.current_timestamp())
    .withColumn('_process_id',        F.lit(PROCESS_ID))
    .withColumn('data_snapshot',      F.lit(DATA_SNAPSHOT).cast('date'))
)

count_periodos = df_metricas_temporais.count()
print(f"Periodos gerados: {count_periodos}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gravação da Tabela Gold

# COMMAND ----------

(
    df_metricas_temporais
    .write
    .mode('overwrite')
    .option('overwriteSchema', True)
    .saveAsTable(TABLE_GOLD_TEMPORAIS)
)

print(f"Tabela gravada: {TABLE_GOLD_TEMPORAIS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Série Histórica — Todo o Histórico (gold_metricas_historicas)
# MAGIC
# MAGIC Tabela para consultas comparativas entre anos pelo agente.
# MAGIC Cobre **todo o histórico disponível na Silver** — sem filtro de janela.
# MAGIC Quando novos CSVs anuais forem adicionados à Silver, esta tabela os inclui
# MAGIC automaticamente na próxima execução, sem alteração de código.
# MAGIC
# MAGIC A coluna `is_ano_completo` distingue anos fechados (histórico confiável)
# MAGIC do ano em curso (dados ainda parciais). O agente deve usar essa flag
# MAGIC ao comparar anos: evitar afirmar que um ano parcial é "o pior" ou "o melhor".

# COMMAND ----------

# Guard: garante flags etiológicas no df_silver completo (Silver v1 sem flags)
_df_hist = df_silver
for _flag in ('is_covid','is_influenza','is_outro_virus','classi_fin_clean'):
    if _flag not in _df_hist.columns:
        _df_hist = _df_hist.withColumn(_flag, F.lit(None).cast(
            'boolean' if _flag.startswith('is_') else 'string'))

# Identifica o ano corrente (máximo da Silver) — provavelmente incompleto
_ano_max_hist = _df_hist.agg(F.max('ano')).collect()[0][0]
print(f"Histórico completo   : todos os anos disponíveis na Silver")
print(f"Ano corrente (parcial): {_ano_max_hist}")

# Agrega todo o histórico com a mesma função usada nos 12 meses
df_historica = _agregar_mensal(_df_hist)

# Aplica taxa de crescimento sobre a série histórica completa
df_historica = _calcular_crescimento(df_historica)

# Flag que indica se o mês pertence a um ano completo ou ao ano ainda em curso
df_historica = df_historica.withColumn(
    'is_ano_completo',
    F.when(F.col('ano') < _ano_max_hist, True).otherwise(False)
)

# Metadados de auditoria
df_historica = (
    df_historica
    .withColumn('_gold_processed_at', F.current_timestamp())
    .withColumn('_process_id',        F.lit(PROCESS_ID))
    .withColumn('data_snapshot',      F.lit(DATA_SNAPSHOT).cast('date'))
)

count_historica = df_historica.count()
anos_hist = sorted([
    r[0] for r in df_historica.select('ano').distinct().collect()
    if r[0] is not None
])
print(f"Meses históricos gerados : {count_historica}")
print(f"Anos cobertos            : {anos_hist}")

(
    df_historica
    .write
    .mode('overwrite')
    .option('overwriteSchema', True)
    .saveAsTable(TABLE_GOLD_HISTORICA)
)
print(f"Tabela gravada: {TABLE_GOLD_HISTORICA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Série Diária — Últimos 30 dias (gold_serie_diaria_30d)
# MAGIC Tabela para o gráfico diário obrigatório do relatório do agente.
# MAGIC Janela: 30 dias anteriores ao max(dt_sin_pri) da Silver.
# MAGIC Âncora em max(dt_sin_pri) — não em current_date() — para funcionar
# MAGIC corretamente quando o dado mais recente está defasado em relação ao
# MAGIC calendário (gap entre data de execução e última notificação no SIVEP-Gripe).

# COMMAND ----------

# Âncora: data máxima de sintomas disponível na Silver (mesma lógica dos 12 meses)
max_dt_sin_pri = df_silver.agg(F.max('dt_sin_pri').alias('m')).collect()[0]['m']
_data_corte_30d = F.date_sub(F.lit(max_dt_sin_pri), 29)  # 30 dias inclusive

print(f"Âncora serie diaria : max(dt_sin_pri) = {max_dt_sin_pri}")
print(f"Janela              : {max_dt_sin_pri} - 29 dias")

# Guard: garante flags para Silver v1
_df_30d = df_silver
for _flag in ('is_covid','is_influenza','is_outro_virus','classi_fin_clean'):
    if _flag not in _df_30d.columns:
        _df_30d = _df_30d.withColumn(_flag, F.lit(None).cast(
            'boolean' if _flag.startswith('is_') else 'string'))

df_serie_diaria = (
    _df_30d
    .filter(F.col('dt_sin_pri').isNotNull() &
            (F.col('dt_sin_pri') >= _data_corte_30d))
    .groupBy('dt_sin_pri')
    .agg(
        F.count('*').alias('total_casos'),
        F.sum(F.when(F.col('is_covid'),       1).otherwise(0)).alias('total_covid'),
        F.sum(F.when(F.col('is_influenza'),   1).otherwise(0)).alias('total_influenza'),
        F.sum(F.when(F.col('is_outro_virus'), 1).otherwise(0)).alias('total_outro_virus'),
        F.sum(F.when(F.col('classi_fin_clean').isNull(), 1).otherwise(0))
         .alias('total_sem_classificacao'),
    )
    .withColumnRenamed('dt_sin_pri', 'dt_sintomas')
    .orderBy('dt_sintomas')
    .withColumn('_gold_processed_at', F.current_timestamp())
    .withColumn('_process_id',        F.lit(PROCESS_ID))
    .withColumn('data_snapshot',      F.lit(DATA_SNAPSHOT).cast('date'))
    .withColumn('data_ancora_serie',  F.lit(max_dt_sin_pri).cast('date'))
)

n_dias = df_serie_diaria.count()
if n_dias == 0:
    print(f"AVISO: sem registros nos 30 dias anteriores a {max_dt_sin_pri}.")
    print(f"       Verificar dt_sin_pri na Silver — possivel gap de dados.")

(df_serie_diaria.write
    .mode('overwrite')
    .option('overwriteSchema', True)
    .saveAsTable(TABLE_GOLD_DIARIA_30D))
print(f"Tabela gravada: {TABLE_GOLD_DIARIA_30D} | {n_dias} dias")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verificação

# COMMAND ----------

(
    spark.table(TABLE_GOLD_TEMPORAIS)
    .orderBy(F.desc('ano_mes_date'))
    .select(
        'ano_mes', 'total_casos',
        'taxa_mortalidade', 'taxa_uti',
        'taxa_vacinacao', 'taxa_crescimento'
    )
    .limit(12)
    .show(truncate=False)
)

# Verificação rápida da histórica: total de casos por ano
print("\nHistórico anual (gold_metricas_historicas):")
(
    spark.table(TABLE_GOLD_HISTORICA)
    .groupBy('ano', 'is_ano_completo')
    .agg(
        F.sum('total_casos').alias('total_casos_ano'),
        F.count('*').alias('meses_com_dado'),
        F.round(F.avg('taxa_mortalidade'), 2).alias('mortalidade_media'),
    )
    .orderBy('ano')
    .show(truncate=False)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resumo de Execução

# COMMAND ----------

count_final   = spark.table(TABLE_GOLD_TEMPORAIS).count()
count_diaria  = spark.table(TABLE_GOLD_DIARIA_30D).count()
count_hist    = spark.table(TABLE_GOLD_HISTORICA).count()

print("=" * 80)
print("METRICAS TEMPORAIS — RESUMO")
print("=" * 80)
print(f"  gold_metricas_temporais  : {count_final} períodos  (últimos 12 meses)")
print(f"  gold_serie_diaria_30d    : {count_diaria} dias     (últimos 30 dias)")
print(f"  gold_metricas_historicas : {count_hist} períodos  (histórico completo — {anos_hist})")
print(f"  Process                  : {PROCESS_ID}")
print("=" * 80)
