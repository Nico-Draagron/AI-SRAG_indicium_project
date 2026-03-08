# Databricks notebook source
# MAGIC %md
# MAGIC # Gold — Métricas Demográficas
# MAGIC
# MAGIC ## 1. Objetivo
# MAGIC
# MAGIC Produzir a tabela `gold_metricas_demograficas` com agregação por faixa etária e sexo,
# MAGIC incluindo taxas epidemiológicas e indicadores hospitalares por estrato demográfico.
# MAGIC A granularidade de saída é uma linha por combinação `(faixa_etaria, cs_sexo_clean)`.
# MAGIC
# MAGIC Pré-requisito: o notebook `01_gold_setup` deve ter sido executado na mesma sessão.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 2. Dimensões Demográficas
# MAGIC
# MAGIC ### 2.1 Faixa Etária
# MAGIC
# MAGIC O campo `faixa_etaria` é gerado pela camada Silver a partir de `idade_anos`.
# MAGIC Este notebook não recalcula faixas; consome diretamente o campo tratado.
# MAGIC Registros com `faixa_etaria IS NULL` são excluídos antes da agregação.
# MAGIC
# MAGIC Faixas disponíveis e ordem de exibição:
# MAGIC
# MAGIC | Ordem | Faixa |
# MAGIC |---|---|
# MAGIC | 1 | 0-1 ano |
# MAGIC | 2 | 1-4 anos |
# MAGIC | 3 | 5-9 anos |
# MAGIC | 4 | 10-17 anos |
# MAGIC | 5 | 18-29 anos |
# MAGIC | 6 | 30-39 anos |
# MAGIC | 7 | 40-49 anos |
# MAGIC | 8 | 50-59 anos |
# MAGIC | 9 | 60-69 anos |
# MAGIC | 10 | 70+ anos |
# MAGIC | 99 | Desconhecido |
# MAGIC
# MAGIC ### 2.2 Sexo
# MAGIC
# MAGIC O campo `cs_sexo_clean` segue a codificação SIVEP-Gripe: `1` = Masculino, `2` = Feminino.
# MAGIC Registros com `cs_sexo_clean IS NULL` (sexo ignorado ou ausente) são **excluídos**
# MAGIC antes da agregação — representam <0.01% do total e introduziriam combinações
# MAGIC espúrias `(faixa_etaria, NULL)` que poluem análises do agente RAG.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 3. Regras de Mortalidade
# MAGIC
# MAGIC Mortalidade SRAG estrita: inclui apenas casos com desfecho registrado.
# MAGIC `EVOLUCAO = '3'` (em acompanhamento) já foi excluído pela camada Silver.
# MAGIC
# MAGIC ```
# MAGIC Denominador : evolucao_clean IN ('1', '2')
# MAGIC Numerador   : evolucao_clean = '2'
# MAGIC ```
# MAGIC
# MAGIC Taxa de internação usa o total de casos do estrato como denominador,
# MAGIC pois qualquer caso pode ou não evoluir para internação.
# MAGIC
# MAGIC ```
# MAGIC Denominador : total_casos (F.count('*') do grupo)
# MAGIC Numerador   : is_internado = TRUE
# MAGIC ```
# MAGIC
# MAGIC Taxa UTI é restrita à população internada (hospital-based):
# MAGIC
# MAGIC ```
# MAGIC Denominador : SUM(CASE WHEN is_internado  THEN 1 END)
# MAGIC Numerador   : SUM(CASE WHEN is_uti_valido THEN 1 END)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 4. Regra de Idade (campo `idade_anos`)
# MAGIC
# MAGIC A Silver padroniza idade em anos aplicando conversões a partir dos campos brutos
# MAGIC `NU_IDADE_N` e `CS_IDADE_UN` do SIVEP-Gripe. A conversão considera a unidade
# MAGIC registrada (`TP_IDADE`):
# MAGIC
# MAGIC | TP_IDADE | Unidade original | Tratamento |
# MAGIC |---|---|---|
# MAGIC | 1 | Dias   | `idade_anos = NULL` — não convertido. Faixa = 'Desconhecido'. |
# MAGIC | 2 | Meses  | `idade_anos = NULL` — não convertido. Faixa = 'Desconhecido'. |
# MAGIC | 3 | Anos | Usado diretamente |
# MAGIC | Outros / NULL | Indeterminado | `idade_anos = NULL` → faixa Desconhecido |
# MAGIC
# MAGIC > **Nota importante**: a Silver NÃO converte TP_IDADE=1 (dias) nem TP_IDADE=2 (meses)
# MAGIC > para anos. Esses registros recebem `idade_anos = NULL` e `faixa_etaria = 'Desconhecido'`.
# MAGIC > Esta decisão está documentada no cabeçalho da Silver e no Gold Setup (seção 3.4).
# MAGIC > Para análises pediátricas com granularidade em dias/meses, criar campo auxiliar
# MAGIC > `idade_dias_equiv` na Silver em versão futura.
# MAGIC
# MAGIC Este notebook não acessa `NU_IDADE_N`, `CS_IDADE_UN` ou `TP_IDADE` diretamente.
# MAGIC O campo `idade_anos` e a coluna `faixa_etaria` são os únicos pontos de contato
# MAGIC com a dimensão etária.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 5. Observações sobre a Categoria "Desconhecido"
# MAGIC
# MAGIC A faixa `Desconhecido` (ordem 99) agrega registros cujo `faixa_etaria` não pôde
# MAGIC ser determinado na Silver, o que ocorre quando:
# MAGIC
# MAGIC - `NU_IDADE_N` é NULL ou zero.
# MAGIC - `TP_IDADE` é diferente de 1, 2 ou 3 (unidade não reconhecida).
# MAGIC - A conversão resulta em valor negativo ou implausível.
# MAGIC
# MAGIC Esses registros não devem ser usados em análises de risco etário nem em
# MAGIC comparações entre faixas. São incluídos na tabela para rastreabilidade e
# MAGIC para que `percentual_total` some 100% sobre o universo completo do estrato.

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

TABLE_GOLD = f"{CATALOG}.{SCHEMA_GOLD}.gold_metricas_demograficas"

print(f"Fonte   : {TABLE_SILVER}")
print(f"Destino : {TABLE_GOLD}")
print(f"Process : {PROCESS_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Carregamento da Silver

# COMMAND ----------

df_silver = spark.table(TABLE_SILVER)
print(f"Registros carregados: {df_silver.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Filtro de Qualidade Demográfica

# COMMAND ----------

# Guard-rail: a Silver sempre preenche faixa_etaria (nunca é NULL — registros sem
# idade válida recebem 'Desconhecido'). Este filtro é mantido como proteção contra
# regressões futuras na Silver, mas não exclui registros em condições normais.
# Sexo: cs_sexo_clean NULL é excluído — <0.01% do total, evita combinações espúrias.
df_filtered = df_silver.filter(
    F.col('faixa_etaria').isNotNull() &
    F.col('cs_sexo_clean').isNotNull()
)

n_total    = df_silver.count()
n_filtrado = df_filtered.count()
n_excluido = n_total - n_filtrado
print(f"Registros com faixa etaria valida: {n_filtrado:,}")
if n_excluido > 0:
    print(f"  Excluidos (sexo NULL)            : {n_excluido:,} ({n_excluido/n_total*100:.3f}%)")

# Diagnóstico de lactentes: avisa quando 0-1 ano tem volume suspeitamente baixo.
# Causa comum: lactentes chegam com TP_IDADE=1 (dias) ou 2 (meses) na Bronze →
# Silver mantém idade_anos=NULL → faixa_etaria='Desconhecido' em vez de '0-1 ano'.
n_01 = df_filtered.filter(F.col('faixa_etaria') == '0-1 ano').count()
n_desconhecido = df_filtered.filter(F.col('faixa_etaria') == 'Desconhecido').count()
print(f"  Faixa 0-1 ano                    : {n_01:,} registros")
print(f"  Faixa Desconhecido               : {n_desconhecido:,} registros")
if n_01 < 500:
    print(f"  ATENCAO: '0-1 ano' com {n_01} registros — provavelmente lactentes em")
    print(f"           TP_IDADE=1/2 (dias/meses) caindo em 'Desconhecido' ({n_desconhecido:,}).")
    print(f"           Criar idade_dias_equiv na Silver para corrigir em versao futura.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agregação por Faixa Etária e Sexo

# COMMAND ----------

df_metricas_demograficas = df_filtered.groupBy('faixa_etaria', 'cs_sexo_clean').agg(

    # --- Contagem geral ---
    F.count('*').alias('total_casos'),

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

    # --- Internação: denominador = total_casos do estrato ---
    F.sum(F.when(F.col('is_internado'), 1).otherwise(0)).alias('total_internados'),

    F.round(
        F.sum(F.when(F.col('is_internado'), 1).otherwise(0)) * 100.0 / F.count('*'),
        2
    ).alias('taxa_internacao'),

    # --- UTI: denominador = is_internado ---
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

    # --- Tempo clínico (dias) ---
    F.round(F.avg('tempo_internacao'), 1).alias('duracao_media_internacao'),

)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Enriquecimento: Labels, Ordem e Percentual
# MAGIC
# MAGIC Implementação via Pandas (≤32 linhas): evita Window global sem partição,
# MAGIC que move todos os dados para uma única partição no Spark Connect/Serverless.

# COMMAND ----------

import pandas as pd

pd_demo = df_metricas_demograficas.toPandas()

# Label legível de sexo (cs_sexo_clean nunca é NULL aqui — filtrado acima)
pd_demo['sexo_label'] = pd_demo['cs_sexo_clean'].map({'1': 'Masculino', '2': 'Feminino'})

# Chave de ordenação para faixa etária
_ordem_map = {
    '0-1 ano': 1, '1-4 anos': 2, '5-9 anos': 3, '10-17 anos': 4,
    '18-29 anos': 5, '30-39 anos': 6, '40-49 anos': 7,
    '50-59 anos': 8, '60-69 anos': 9, '70+ anos': 10,
}
pd_demo['ordem_faixa'] = pd_demo['faixa_etaria'].map(_ordem_map).fillna(99).astype(int)

# Percentual sobre o total do universo filtrado (sem Window, sem warning)
total_casos_universo = pd_demo['total_casos'].sum()
pd_demo['percentual_total'] = (pd_demo['total_casos'] * 100.0 / total_casos_universo).round(2)

# Reconverte para Spark — ordem garantida por ordem_faixa + sexo_label
df_metricas_demograficas = spark.createDataFrame(pd_demo)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Metadados de Auditoria

# COMMAND ----------

df_metricas_demograficas = (
    df_metricas_demograficas
    .withColumn('_gold_processed_at', F.current_timestamp())
    .withColumn('_process_id',        F.lit(PROCESS_ID))
    .withColumn('data_snapshot',      F.lit(DATA_SNAPSHOT).cast('date'))
)

count_combinacoes = df_metricas_demograficas.count()
print(f"Combinacoes (faixa x sexo): {count_combinacoes}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gravação da Tabela Gold

# COMMAND ----------

(
    df_metricas_demograficas
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
    .orderBy('ordem_faixa', 'sexo_label')
    .select(
        'faixa_etaria', 'sexo_label', 'total_casos',
        'percentual_total', 'taxa_mortalidade', 'taxa_internacao'
    )
    .limit(15)
    .show(truncate=False)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resumo de Execução

# COMMAND ----------

count_final = spark.table(TABLE_GOLD).count()

print("=" * 80)
print("METRICAS DEMOGRAFICAS — RESUMO")
print("=" * 80)
print(f"  Tabela       : {TABLE_GOLD}")
print(f"  Combinacoes  : {count_final}")
print(f"  Process      : {PROCESS_ID}")
print("=" * 80)
