# Databricks notebook source
# MAGIC %md
# MAGIC # Gold — Base de Conhecimento RAG
# MAGIC
# MAGIC ## 1. Objetivo
# MAGIC
# MAGIC Produzir duas tabelas Gold estruturadas para consumo pelo agente RAG:
# MAGIC
# MAGIC - `gold_rag_kpi_fatos`: documentos semânticos com KPIs agregados e campo `text`
# MAGIC   em linguagem natural, pronto para indexação por embeddings.
# MAGIC - `gold_rag_dicionario_regras`: base de conhecimento estática com as definições
# MAGIC   epidemiológicas e técnicas formais aplicadas em todo o pipeline Gold.
# MAGIC
# MAGIC Pré-requisito: o notebook `01_gold_setup` deve ter sido executado na mesma sessão.
# MAGIC Este notebook consome os widgets registrados pelo setup e não modifica
# MAGIC nenhuma tabela Gold analítica existente (02, 03, 04).
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 2. Fontes e Dependências
# MAGIC
# MAGIC | Atributo | Valor |
# MAGIC |---|---|
# MAGIC | Tabela fonte | Lida via widget `table_silver` (definido pelo `01_gold_setup`) |
# MAGIC | Catálogo destino | Lido via widget `catalog_gold` |
# MAGIC | Schema destino | Lido via widget `schema_gold` |
# MAGIC | Tabelas geradas | `gold_rag_kpi_fatos`, `gold_rag_dicionario_regras` |
# MAGIC | Tabelas não modificadas | `gold_metricas_temporais`, `gold_metricas_geograficas`, `gold_metricas_demograficas` |
# MAGIC
# MAGIC O schema Gold é criado pelo `01_gold_setup` antes da execução deste notebook.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 3. Regras Epidemiológicas Aplicadas
# MAGIC
# MAGIC ### 3.1 Mortalidade SRAG Estrita
# MAGIC
# MAGIC ```
# MAGIC Denominador : evolucao_clean IN ('1', '2')
# MAGIC Numerador   : evolucao_clean = '2'
# MAGIC ```
# MAGIC
# MAGIC `EVOLUCAO = '3'` (em acompanhamento) é excluído pela Silver e não aparece
# MAGIC em `evolucao_clean`. Casos sem desfecho registrado ficam fora do denominador.
# MAGIC
# MAGIC ### 3.2 Taxa de Ocupação UTI
# MAGIC
# MAGIC ```
# MAGIC Denominador : is_internado = TRUE
# MAGIC Numerador   : is_uti_valido = TRUE
# MAGIC ```
# MAGIC
# MAGIC ### 3.3 Taxa de Vacinação
# MAGIC
# MAGIC ```
# MAGIC Denominador : vacina_clean IS NOT NULL
# MAGIC Numerador   : vacina_clean = '1'
# MAGIC ```
# MAGIC
# MAGIC Código `'9'` (ignorado) é mapeado para NULL na Silver e excluído do denominador.
# MAGIC
# MAGIC ### 3.4 Crescimento Mensal
# MAGIC
# MAGIC ```
# MAGIC taxa_crescimento(t) = ( casos(t) - casos(t-1) ) / casos(t-1) * 100
# MAGIC ```
# MAGIC
# MAGIC Calculado sobre janela de 12 meses. Ordenação por `ano_mes_date`.
# MAGIC Resultado NULL no primeiro período e quando `casos(t-1) = 0`.
# MAGIC
# MAGIC ### 3.5 Idade
# MAGIC
# MAGIC Utilizar exclusivamente `idade_anos`. Colunas brutas `NU_IDADE_N` e `CS_IDADE_UN`
# MAGIC não são referenciadas neste notebook.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 4. Estrutura do Campo `text`
# MAGIC
# MAGIC Cada registro de `gold_rag_kpi_fatos` contém um campo `text` com linguagem natural
# MAGIC factual e técnica, sem emojis e sem construções promocionais.
# MAGIC O texto descreve os valores do período, explicita o escopo geográfico e demográfico
# MAGIC e referencia implicitamente as regras de cálculo aplicadas.
# MAGIC
# MAGIC Exemplo para agregação mensal Brasil:
# MAGIC
# MAGIC ```
# MAGIC Em 2024-03, o Brasil registrou 8.532 casos de SRAG. A taxa de mortalidade
# MAGIC SRAG estrita foi de 12,4%, considerando apenas casos com desfecho registrado
# MAGIC (evolucao_clean IN (1,2)). A taxa de UTI foi de 18,7%, calculada
# MAGIC exclusivamente sobre pacientes hospitalizados.
# MAGIC ```

# COMMAND ----------

# MAGIC %run ./01_gold_setup

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import Window
from datetime import datetime

# COMMAND ----------

# MAGIC %md
# MAGIC ## Carregamento de Configurações

# COMMAND ----------

CATALOG_GOLD  = dbutils.widgets.get("catalog_gold")
SCHEMA_GOLD   = dbutils.widgets.get("schema_gold")
TABLE_SILVER  = dbutils.widgets.get("table_silver")
PROCESS_ID    = dbutils.widgets.get("process_id")

# data_snapshot: registrado pelo 01_gold_setup quando executado via orquestrador.
# Fallback para current_date() quando o notebook roda standalone (desenvolvimento/debug).
try:
    DATA_SNAPSHOT = dbutils.widgets.get("data_snapshot")
except Exception:
    DATA_SNAPSHOT = spark.sql("SELECT current_date() AS d").collect()[0]["d"].isoformat()
    print(f"  AVISO: widget 'data_snapshot' nao encontrado — usando current_date(): {DATA_SNAPSHOT}")
    print(f"         Para rastreabilidade completa, execute via 00_pipeline_gold.")

TABLE_RAG_FATOS      = f"{CATALOG_GOLD}.{SCHEMA_GOLD}.gold_rag_kpi_fatos"
TABLE_RAG_DICIONARIO = f"{CATALOG_GOLD}.{SCHEMA_GOLD}.gold_rag_dicionario_regras"

print(f"Fonte              : {TABLE_SILVER}")
print(f"Destino fatos      : {TABLE_RAG_FATOS}")
print(f"Destino dicionario : {TABLE_RAG_DICIONARIO}")
print(f"Process ID         : {PROCESS_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Carregamento da Silver

# COMMAND ----------

df_silver = spark.table(TABLE_SILVER)

total_registros = df_silver.count()
print(f"Registros carregados: {total_registros:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Filtro: Últimos 12 Meses

# COMMAND ----------

max_ano_mes = df_silver.agg(F.max('ano_mes_date')).collect()[0][0]

df_silver_12m = df_silver.filter(
    F.col('ano_mes_date') >= F.add_months(F.lit(max_ano_mes), -11)
)

print(f"Periodo maximo     : {max_ano_mes}")
print(f"Registros 12 meses : {df_silver_12m.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parte 1 — gold_rag_kpi_fatos
# MAGIC
# MAGIC ### Agregação Mensal Brasil (doc_type: kpi_mensal_brasil)

# COMMAND ----------

df_mensal_brasil = (
    df_silver_12m
    .groupBy('ano', 'mes', 'ano_mes', 'ano_mes_date')
    .agg(
        F.count('*').alias('total_casos'),

        # Mortalidade
        F.sum(F.when(F.col('evolucao_clean') == '2', 1).otherwise(0))
         .alias('total_obitos'),
        F.sum(F.when(F.col('evolucao_clean').isin('1', '2'), 1).otherwise(0))
         .alias('casos_com_desfecho'),

        # Internação e UTI
        F.sum(F.when(F.col('is_internado'),    1).otherwise(0)).alias('total_internados'),
        F.sum(F.when(F.col('is_uti_valido'),   1).otherwise(0)).alias('total_uti'),

        # Vacinação
        F.sum(F.when(F.col('vacina_clean') == '1', 1).otherwise(0)).alias('total_vacinados'),
        F.sum(F.when(F.col('vacina_clean').isNotNull(), 1).otherwise(0))
         .alias('casos_com_info_vacina'),
    )
    # Taxas
    .withColumn(
        'taxa_mortalidade',
        F.round(
            F.when(F.col('casos_com_desfecho') > 0,
                   F.col('total_obitos') * 100.0 / F.col('casos_com_desfecho')
            ).otherwise(None), 2
        )
    )
    .withColumn(
        'taxa_uti',
        F.round(
            F.when(F.col('total_internados') > 0,
                   F.col('total_uti') * 100.0 / F.col('total_internados')
            ).otherwise(None), 2
        )
    )
    .withColumn(
        'taxa_vacinacao',
        F.round(
            F.when(F.col('casos_com_info_vacina') > 0,
                   F.col('total_vacinados') * 100.0 / F.col('casos_com_info_vacina')
            ).otherwise(None), 2
        )
    )
    # Crescimento mensal calculado após coleta em Pandas (evita Window global sem partição)
    .withColumn('doc_type',    F.lit('kpi_mensal_brasil'))
    .withColumn('is_ano_parcial', F.lit(False))
    .withColumn('uf',          F.lit(None).cast('string'))
    .withColumn('faixa_etaria', F.lit(None).cast('string'))
    .withColumn('doc_id',
        F.concat_ws('_', F.lit('kpi_mensal_brasil'), F.col('ano_mes'))
    )
)

# Crescimento mensal Brasil via Pandas (12 linhas — sem Window global, sem warning)
import pandas as pd, math

pd_brasil = df_mensal_brasil.orderBy('ano_mes_date').toPandas()
pd_brasil['_casos_anterior'] = pd_brasil['total_casos'].shift(1)
pd_brasil['taxa_crescimento'] = pd_brasil.apply(
    lambda r: round((r['total_casos'] - r['_casos_anterior'])
                    / r['_casos_anterior'] * 100, 2)
    if (not math.isnan(r['_casos_anterior']) and r['_casos_anterior'] > 0)
    else None,
    axis=1
)
pd_brasil = pd_brasil.drop(columns=['_casos_anterior'])
df_mensal_brasil = spark.createDataFrame(pd_brasil)

print(f"Periodos mensais Brasil: {df_mensal_brasil.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Agregação Mensal por UF (doc_type: kpi_mensal_uf)

# COMMAND ----------

window_uf = Window.partitionBy('sg_uf').orderBy('ano_mes_date')

df_mensal_uf = (
    df_silver_12m
    .filter(F.col('sg_uf').isNotNull())
    .groupBy('ano', 'mes', 'ano_mes', 'ano_mes_date', 'sg_uf')
    .agg(
        F.count('*').alias('total_casos'),

        F.sum(F.when(F.col('evolucao_clean') == '2', 1).otherwise(0))
         .alias('total_obitos'),
        F.sum(F.when(F.col('evolucao_clean').isin('1', '2'), 1).otherwise(0))
         .alias('casos_com_desfecho'),

        F.sum(F.when(F.col('is_internado'),    1).otherwise(0)).alias('total_internados'),
        F.sum(F.when(F.col('is_uti_valido'),   1).otherwise(0)).alias('total_uti'),

        F.sum(F.when(F.col('vacina_clean') == '1', 1).otherwise(0)).alias('total_vacinados'),
        F.sum(F.when(F.col('vacina_clean').isNotNull(), 1).otherwise(0))
         .alias('casos_com_info_vacina'),
    )
    .withColumn(
        'taxa_mortalidade',
        F.round(
            F.when(F.col('casos_com_desfecho') > 0,
                   F.col('total_obitos') * 100.0 / F.col('casos_com_desfecho')
            ).otherwise(None), 2
        )
    )
    .withColumn(
        'taxa_uti',
        F.round(
            F.when(F.col('total_internados') > 0,
                   F.col('total_uti') * 100.0 / F.col('total_internados')
            ).otherwise(None), 2
        )
    )
    .withColumn(
        'taxa_vacinacao',
        F.round(
            F.when(F.col('casos_com_info_vacina') > 0,
                   F.col('total_vacinados') * 100.0 / F.col('casos_com_info_vacina')
            ).otherwise(None), 2
        )
    )
    .withColumn('_casos_anterior', F.lag('total_casos').over(window_uf))
    .withColumn(
        'taxa_crescimento',
        F.round(
            F.when(
                F.col('_casos_anterior') > 0,
                (F.col('total_casos') - F.col('_casos_anterior')) * 100.0 /
                F.col('_casos_anterior')
            ).otherwise(None), 2
        )
    )
    .drop('_casos_anterior')
    .withColumn('doc_type',    F.lit('kpi_mensal_uf'))
    .withColumn('is_ano_parcial', F.lit(False))
    .withColumnRenamed('sg_uf', 'uf')
    .withColumn('faixa_etaria', F.lit(None).cast('string'))
    .withColumn('doc_id',
        F.concat_ws('_', F.lit('kpi_mensal_uf'), F.col('uf'), F.col('ano_mes'))
    )
)

print(f"Combinacoes mes x UF: {df_mensal_uf.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Agregação Anual Brasil (doc_type: kpi_anual)

# COMMAND ----------

anos_disponiveis = sorted([
    r[0] for r in df_silver.select('ano').distinct().collect()
    if r[0] is not None
])
ano_corrente = max(anos_disponiveis) if anos_disponiveis else None
print(f"Anos disponíveis para kpi_anual: {anos_disponiveis}")
print(f"Ano corrente (parcial): {ano_corrente}")

df_anual = (
    df_silver
    .filter(F.col('ano').isin(anos_disponiveis))
    .groupBy('ano')
    .agg(
        F.count('*').alias('total_casos'),

        F.sum(F.when(F.col('evolucao_clean') == '2', 1).otherwise(0))
         .alias('total_obitos'),
        F.sum(F.when(F.col('evolucao_clean').isin('1', '2'), 1).otherwise(0))
         .alias('casos_com_desfecho'),

        F.sum(F.when(F.col('is_internado'),    1).otherwise(0)).alias('total_internados'),
        F.sum(F.when(F.col('is_uti_valido'),   1).otherwise(0)).alias('total_uti'),

        F.sum(F.when(F.col('vacina_clean') == '1', 1).otherwise(0)).alias('total_vacinados'),
        F.sum(F.when(F.col('vacina_clean').isNotNull(), 1).otherwise(0))
         .alias('casos_com_info_vacina'),
    )
    .withColumn(
        'taxa_mortalidade',
        F.round(
            F.when(F.col('casos_com_desfecho') > 0,
                   F.col('total_obitos') * 100.0 / F.col('casos_com_desfecho')
            ).otherwise(None), 2
        )
    )
    .withColumn(
        'taxa_uti',
        F.round(
            F.when(F.col('total_internados') > 0,
                   F.col('total_uti') * 100.0 / F.col('total_internados')
            ).otherwise(None), 2
        )
    )
    .withColumn(
        'taxa_vacinacao',
        F.round(
            F.when(F.col('casos_com_info_vacina') > 0,
                   F.col('total_vacinados') * 100.0 / F.col('casos_com_info_vacina')
            ).otherwise(None), 2
        )
    )
    .withColumn('taxa_crescimento', F.lit(None).cast('double'))
    .withColumn('is_ano_parcial',
        F.when(F.col('ano') == ano_corrente, True).otherwise(False))
    .withColumn('doc_type',    F.lit('kpi_anual'))
    .withColumn('mes',         F.lit(None).cast('int'))
    .withColumn('ano_mes',     F.col('ano').cast('string'))
    .withColumn('ano_mes_date', F.lit(None).cast('date'))
    .withColumn('uf',          F.lit(None).cast('string'))
    .withColumn('faixa_etaria', F.lit(None).cast('string'))
    .withColumn('doc_id',
        F.concat_ws('_', F.lit('kpi_anual'), F.col('ano').cast('string'))
    )
)

print(f"Anos agregados: {df_anual.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Geração do Campo `text`

# COMMAND ----------

def _build_text_col():
    """
    Constrói o campo text como concatenação condicional de segmentos factuais.
    Cada segmento referencia implicitamente a regra de cálculo aplicada.
    Não utiliza UDF — opera inteiramente com funções nativas Spark.
    """
    escopo = F.when(F.col('uf').isNotNull(),
                    F.concat(F.lit('estado '), F.col('uf'))
             ).otherwise(F.lit('Brasil'))

    periodo = F.when(
        F.col('mes').isNotNull(),
        F.concat(F.lit('Em '), F.col('ano_mes'), F.lit(', o '), escopo)
    ).otherwise(
        F.concat(
            F.lit('No ano de '), F.col('ano').cast('string'),
            F.when(F.col('is_ano_parcial') == True,
                   F.lit(' (dados parciais — ano em andamento)')).otherwise(F.lit('')),
            F.lit(', o '), escopo
        )
    )

    seg_casos = F.concat(
        periodo,
        F.lit(' registrou '),
        F.format_number(F.col('total_casos'), 0),
        F.lit(' casos de SRAG.')
    )

    seg_mortalidade = F.when(
        F.col('taxa_mortalidade').isNotNull(),
        F.concat(
            F.lit(' A taxa de mortalidade SRAG estrita foi de '),
            F.col('taxa_mortalidade').cast('string'),
            F.lit('%, considerando apenas casos com desfecho registrado (evolucao_clean IN (1,2)).')
        )
    ).otherwise(F.lit(''))

    seg_uti = F.when(
        F.col('taxa_uti').isNotNull(),
        F.concat(
            F.lit(' A taxa de UTI foi de '),
            F.col('taxa_uti').cast('string'),
            F.lit('%, calculada exclusivamente sobre pacientes hospitalizados.')
        )
    ).otherwise(F.lit(''))

    seg_vacinacao = F.when(
        F.col('taxa_vacinacao').isNotNull(),
        F.concat(
            F.lit(' A taxa de vacinacao foi de '),
            F.col('taxa_vacinacao').cast('string'),
            F.lit('%, sobre casos com informacao vacinal registrada (vacina_clean IS NOT NULL).')
        )
    ).otherwise(F.lit(''))

    seg_crescimento = F.when(
        F.col('taxa_crescimento').isNotNull(),
        F.concat(
            F.lit(' A variacao em relacao ao mes anterior foi de '),
            F.col('taxa_crescimento').cast('string'),
            F.lit('%.')
        )
    ).otherwise(F.lit(''))

    return F.concat(
        seg_casos, seg_mortalidade, seg_uti, seg_vacinacao, seg_crescimento
    )


# Colunas comuns para union
COLUNAS_FATOS = [
    'doc_id', 'doc_type',
    'ano', 'mes', 'ano_mes', 'ano_mes_date',
    'uf', 'faixa_etaria',
    'total_casos', 'total_obitos', 'taxa_mortalidade',
    'total_internados', 'total_uti', 'taxa_uti',
    'total_vacinados', 'taxa_vacinacao',
    'taxa_crescimento',
    'is_ano_parcial',
    'text', 'filtros_desc', 'regra_mortalidade', 'regra_uti',
    'fonte_tabela', 'gerado_em', 'process_id', 'data_snapshot',
]

def _enrich(df, filtros_desc_value: str):
    return (
        df
        .withColumn('text', _build_text_col())
        .withColumn('filtros_desc',      F.lit(filtros_desc_value))
        .withColumn('regra_mortalidade', F.lit(
            'Mortalidade SRAG estrita: denominador evolucao_clean IN (1,2); numerador evolucao_clean=2'
        ))
        .withColumn('regra_uti', F.lit(
            'Taxa UTI hospital-based: denominador is_internado=TRUE; numerador is_uti_valido=TRUE'
        ))
        .withColumn('fonte_tabela', F.lit(TABLE_SILVER))
        .withColumn('gerado_em',    F.current_timestamp())
        .withColumn('process_id',   F.lit(PROCESS_ID))
        .withColumn('data_snapshot', F.lit(DATA_SNAPSHOT).cast('date'))
        .select(COLUNAS_FATOS)
    )

df_fatos_brasil = _enrich(
    df_mensal_brasil,
    'Agregacao mensal; escopo Brasil; ultimos 12 meses'
)

df_fatos_uf = _enrich(
    df_mensal_uf,
    'Agregacao mensal por UF; sg_uf IS NOT NULL; ultimos 12 meses'
)

df_fatos_anual = _enrich(
    df_anual,
    'Agregacao anual; escopo Brasil; historico completo Silver'
)

df_fatos = (
    df_fatos_brasil
    .unionByName(df_fatos_uf)
    .unionByName(df_fatos_anual)
)

total_fatos = df_fatos.count()
print(f"Total de documentos RAG gerados: {total_fatos:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gravação — gold_rag_kpi_fatos

# COMMAND ----------

(
    df_fatos
    .write
    .mode('overwrite')
    .option('overwriteSchema', True)
    .saveAsTable(TABLE_RAG_FATOS)
)

print(f"Tabela gravada: {TABLE_RAG_FATOS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parte 2 — gold_rag_dicionario_regras
# MAGIC
# MAGIC Tabela estática com as definições formais das regras epidemiológicas e técnicas
# MAGIC aplicadas no pipeline Gold. Serve como contexto de recuperação para o sistema RAG.

# COMMAND ----------

from datetime import date

created_at = datetime.now()

regras = [
    {
        'rule_id'          : 'MORT_001',
        'rule_category'    : 'mortalidade',
        'rule_name'        : 'Mortalidade SRAG estrita — exclusao de EVOLUCAO=3',
        'rule_description' : (
            'Registros com EVOLUCAO=3 (em acompanhamento) sao excluidos do denominador '
            'de mortalidade. O denominador e restrito a evolucao_clean IN (1,2), '
            'onde 1=cura e 2=obito. O numerador e evolucao_clean=2.'
        ),
        'impact_analysis'  : (
            'A exclusao de casos em acompanhamento evita subestimacao da taxa quando ha '
            'grande volume de notificacoes recentes sem desfecho. A taxa resultante e '
            'comparavel entre periodos distintos.'
        ),
        'created_at'       : created_at,
    },
    {
        'rule_id'          : 'IDADE_001',
        'rule_category'    : 'idade',
        'rule_name'        : 'Idade em anos — campo idade_anos',
        'rule_description' : (
            'A Silver preenche idade_anos APENAS quando TP_IDADE=3 (anos completos, SIVEP-Gripe). '
            'TP_IDADE=1 (dias) e TP_IDADE=2 (meses) recebem idade_anos=NULL — nao ha conversao. '
            'A faixa_etaria desses registros e Desconhecido. '
            'Apenas TP_IDADE=3 garante que NU_IDADE_N representa anos inteiros.'
        ),
        'impact_analysis'  : (
            'Faixa Desconhecido nao deve ser usada em analises de risco etario. '
            'Para pediatria com dias/meses, criar campo auxiliar idade_dias_equiv na Silver.'
        ),
        'created_at'       : created_at,
    },
    {
        'rule_id'          : 'UTI_001',
        'rule_category'    : 'uti',
        'rule_name'        : 'Taxa UTI — denominador hospitalar',
        'rule_description' : (
            'A taxa de ocupacao de UTI e calculada exclusivamente sobre a populacao internada. '
            'Denominador: is_internado=TRUE (hospital_clean=1). '
            'Numerador: is_uti_valido=TRUE (hospital_clean=1 AND uti_clean=1, com validacao Silver). '
            'Casos ambulatoriais nao integram o calculo.'
        ),
        'impact_analysis'  : (
            'O uso do denominador hospitalar torna a taxa comparavel a benchmarks clinicos. '
            'Taxa sobre total de casos produziria valor artificialmente baixo e nao comparavel '
            'entre regioes com diferentes perfis de internacao.'
        ),
        'created_at'       : created_at,
    },
    {
        'rule_id'          : 'VAC_001',
        'rule_category'    : 'vacinacao',
        'rule_name'        : 'Taxa de vacinacao — exclusao do codigo 9 (ignorado)',
        'rule_description' : (
            'O campo VACINA do SIVEP-Gripe aceita os valores 1=Sim, 2=Nao, 9=Ignorado. '
            'Na Silver, o valor 9 e mapeado para NULL em vacina_clean. '
            'O denominador da taxa de vacinacao e vacina_clean IS NOT NULL, '
            'excluindo portanto os registros com status ignorado ou ausente. '
            'O numerador e vacina_clean=1.'
        ),
        'impact_analysis'  : (
            'A exclusao do codigo 9 evita diluicao artificial da taxa em periodos ou regioes '
            'com alta proporcao de registros incompletos. A cobertura vacinal reportada '
            'reflete apenas a populacao com informacao definida.'
        ),
        'created_at'       : created_at,
    },
    {
        'rule_id'          : 'CRESC_001',
        'rule_category'    : 'crescimento',
        'rule_name'        : 'Taxa de crescimento mensal',
        'rule_description' : (
            'A variacao de casos mes a mes e calculada pela formula: '
            'taxa_crescimento(t) = (casos(t) - casos(t-1)) / casos(t-1) * 100. '
            'A janela de calculo e limitada aos ultimos 12 meses disponíveis. '
            'A ordenacao e feita por ano_mes_date (tipo DATE). '
            'O resultado e NULL no primeiro mes da serie e quando casos(t-1)=0.'
        ),
        'impact_analysis'  : (
            'O uso de ano_mes_date como chave de ordenacao garante ordem cronologica correta '
            'e evita erros de ordenacao lexicografica da string ano_mes. '
            'A restricao de 12 meses reduz o volume de dados e mantém a serie dentro '
            'do contexto operacional do sistema RAG.'
        ),
        'created_at'       : created_at,
    },
    {
        'rule_id'          : 'CLASSI_001',
        'rule_category'    : 'classificacao',
        'rule_name'        : 'Classificacao etiologica — classi_fin_clean',
        'rule_description' : (
            'CLASSI_FIN classifica o agente causador da SRAG (SIVEP-Gripe). '
            'classi_fin_clean: code 9 → NULL, valores 1-5 preservados. '
            'Mapa: 1=Influenza, 2=Outro virus, 3=Outro agente, 4=Nao especificado, 5=COVID-19. '
            'Flags derivadas: is_covid=(classi_fin_clean=5), is_influenza=(classi_fin_clean=1), '
            'is_outro_virus=(classi_fin_clean=2).'
        ),
        'impact_analysis'  : (
            'Alta mortalidade em periodo COVID tem interpretacao diferente de periodo Influenza. '
            'Sempre indicar agente dominante ao reportar metricas temporais ou geograficas. '
            'Registros com classi_fin_clean=NULL nao devem ser excluidos das contagens gerais.'
        ),
        'created_at'       : created_at,
    },
    {
        'rule_id'          : 'VAC_COV_001',
        'rule_category'    : 'vacinacao',
        'rule_name'        : 'Vacinacao COVID-19 — vacina_cov_clean',
        'rule_description' : (
            'VACINA_COV registra vacinacao COVID-19 (1=Sim, 2=Nao, 9=Ignorado). '
            'vacina_cov_clean: 9 → NULL. Flag: is_vacinado_covid = (vacina_cov_clean=1). '
            'Denominador: vacina_cov_clean IS NOT NULL. '
            'CAMPO DISTINTO de vacina_clean/VAC_001 (que registra vacinacao gripe/SRAG).'
        ),
        'impact_analysis'  : (
            'Cobertura vacinal COVID-19 e um dos 4 indicadores obrigatorios do relatorio. '
            'Nao confundir com taxa de vacinacao de gripe. Sempre distinguir as duas metricas.'
        ),
        'created_at'       : created_at,
    },
    {
        'rule_id'          : 'EVOL_001',
        'rule_category'    : 'mortalidade',
        'rule_name'        : 'Campo evolucao_clean — mapeamento SIVEP-Gripe',
        'rule_description' : (
            'O campo EVOLUCAO original do SIVEP-Gripe aceita: 1=Cura, 2=Obito, '
            '3=Obito por outras causas, 9=Ignorado. '
            'Na Silver, evolucao_clean preserva os valores 1 e 2. '
            'Os valores 3 e 9 sao mapeados para NULL. '
            'Todos os calculos de mortalidade Gold usam exclusivamente evolucao_clean.'
        ),
        'impact_analysis'  : (
            'A separacao entre obito por SRAG (2) e obito por outras causas (3) e '
            'fundamental para a definicao de mortalidade SRAG estrita. '
            'Incluir o valor 3 no numerador superestimaria a mortalidade especifica por SRAG.'
        ),
        'created_at'       : created_at,
    },
]

df_dicionario = (
    spark.createDataFrame(regras)
    .withColumn('_gold_processed_at', F.current_timestamp())
    .withColumn('_process_id',        F.lit(PROCESS_ID))
)

print(f"Regras no dicionario: {df_dicionario.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gravação — gold_rag_dicionario_regras

# COMMAND ----------

(
    df_dicionario
    .write
    .mode('overwrite')
    .option('overwriteSchema', True)
    .saveAsTable(TABLE_RAG_DICIONARIO)
)

print(f"Tabela gravada: {TABLE_RAG_DICIONARIO}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validações Pós-Escrita

# COMMAND ----------

# Contagens finais
count_fatos      = spark.table(TABLE_RAG_FATOS).count()
count_dicionario = spark.table(TABLE_RAG_DICIONARIO).count()

print(f"gold_rag_kpi_fatos       : {count_fatos:,} documentos")
print(f"gold_rag_dicionario_regras: {count_dicionario} regras")

# Verificar ausencia de doc_id duplicado
duplicados = (
    spark.table(TABLE_RAG_FATOS)
    .groupBy('doc_id')
    .count()
    .filter(F.col('count') > 1)
    .count()
)

assert duplicados == 0, f"ERRO: {duplicados} doc_id duplicados em gold_rag_kpi_fatos"
print("Integridade doc_id: sem duplicatas")

# Verificar que nenhum registro tem text vazio
sem_texto = (
    spark.table(TABLE_RAG_FATOS)
    .filter(F.col('text').isNull() | (F.trim(F.col('text')) == ''))
    .count()
)

assert sem_texto == 0, f"ERRO: {sem_texto} registros com campo text vazio"
print("Integridade text: sem registros vazios")

# Distribuicao por doc_type
print("\nDistribuicao por doc_type:")
(
    spark.table(TABLE_RAG_FATOS)
    .groupBy('doc_type')
    .count()
    .orderBy('doc_type')
    .show(truncate=False)
)

# Preview dicionario
print("Dicionario de regras:")
(
    spark.table(TABLE_RAG_DICIONARIO)
    .select('rule_id', 'rule_category', 'rule_name')
    .orderBy('rule_category', 'rule_id')
    .show(truncate=False)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resumo de Execução

# COMMAND ----------

print("=" * 80)
print("RAG KNOWLEDGE BASE — RESUMO DE EXECUCAO")
print("=" * 80)
print(f"  gold_rag_kpi_fatos        : {count_fatos:,} documentos")
print(f"  gold_rag_dicionario_regras: {count_dicionario} regras")
print(f"  Janela mensal/UF           : ultimos 12 meses a partir de {max_ano_mes}")
print(f"  Janela anual               : historico completo Silver ({len(anos_disponiveis)} anos)")
print(f"  Process ID                : {PROCESS_ID}")
print("=" * 80)
