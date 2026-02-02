# Databricks notebook source
# MAGIC %md
# MAGIC # Camada Silver - Transformação e Limpeza de Dados SRAG
# MAGIC 
# MAGIC **Projeto**: Sistema RAG para Monitoramento Epidemiológico
# MAGIC 
# MAGIC **Objetivo**: Criar tabela Silver limpa, tipada e otimizada para análises epidemiológicas
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC ## Escopo da Transformação
# MAGIC 
# MAGIC ### Conteúdo:
# MAGIC - Seleciona apenas colunas críticas (197 → ~35-40)
# MAGIC - Aplica tipagem correta (String → Date, Int, Boolean)
# MAGIC - Trata código "9" (Ignorado) de forma apropriada
# MAGIC - Aplica filtros de qualidade obrigatórios
# MAGIC - Cria features derivadas (temporais, demográficas, flags)
# MAGIC - Valida consistência temporal
# MAGIC - Particiona por ano + mês
# MAGIC - Otimiza com Z-ordering
# MAGIC 
# MAGIC ### Transformações:
# MAGIC - **Input**: `bronze.bronze_srag_raw` (870.914 registros, 197 colunas)
# MAGIC - **Output**: `silver.silver_srag_clean` (~500k-700k registros, 35-40 colunas)
# MAGIC - **Redução esperada**: 20-40% por filtros de qualidade
# MAGIC 
# MAGIC ### Decisões Arquiteturais:
# MAGIC - Código "9" = "Ignorado" (mantido como categoria válida)
# MAGIC - Colunas `_clean` para métricas (sem "9")
# MAGIC - Particionamento: `ano` + `mes`
# MAGIC - Z-ordering: `dt_sin_pri`, `sg_uf`
# MAGIC - Schema explícito (não inferido)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup e Imports

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql import Window
from datetime import datetime

print("=" * 80)
print("CAMADA SILVER - TRANSFORMAÇÃO E LIMPEZA")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Spark Version: {spark.version}")
print(f"Ambiente: Databricks Serverless")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuração de Ambiente

# COMMAND ----------

CATALOG = "workspace"
SCHEMA_BRONZE = "data_original"
SCHEMA_SILVER = "silver"

TABLE_BRONZE = f"{CATALOG}.{SCHEMA_BRONZE}.bronze_srag_raw"
TABLE_SILVER = f"{CATALOG}.{SCHEMA_SILVER}.silver_srag_clean"

PROCESS_ID = datetime.now().strftime('%Y%m%d_%H%M%S')

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA_SILVER}")

print("CONFIGURAÇÃO:")
print(f"  • Fonte: {TABLE_BRONZE}")
print(f"  • Destino: {TABLE_SILVER}")
print(f"  • Process ID: {PROCESS_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Leitura da Camada Bronze

# COMMAND ----------

print("\nCarregando dados da Bronze...")

df_bronze = spark.table(TABLE_BRONZE)

bronze_count = df_bronze.count()
bronze_cols = len(df_bronze.columns)

print(f"\nDados Bronze carregados:")
print(f"  • Registros: {bronze_count:,}")
print(f"  • Colunas: {bronze_cols}")

print(f"\nDistribuição por ano (Bronze):")
df_bronze.groupBy("ANO_DADOS").count().orderBy("ANO_DADOS").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Seleção de Colunas Críticas

# COMMAND ----------

print("\nSELECIONANDO COLUNAS CRÍTICAS...")
print("=" * 80)

COLUNAS_SILVER = {
    'identificacao': ['NU_NOTIFIC'],
    'temporal': ['DT_NOTIFIC', 'DT_SIN_PRI', 'SEM_PRI', 'ANO_DADOS'],
    'demografia': ['CS_SEXO', 'NU_IDADE_N', 'TP_IDADE', 'SG_UF', 'CO_MUN_RES', 'CS_RACA'],
    'sintomas': ['FEBRE', 'TOSSE', 'DISPNEIA', 'GARGANTA', 'SATURACAO', 'DESC_RESP'],
    'internacao': ['HOSPITAL', 'DT_INTERNA', 'UTI', 'DT_ENTUTI', 'SUPORT_VEN'],
    'desfecho': ['EVOLUCAO', 'DT_EVOLUCA', 'CLASSI_FIN'],
    'vacinacao': ['VACINA', 'VACINA_COV', 'DOSE_1_COV', 'DOSE_2_COV']
}

colunas_selecionadas = []
for categoria, campos in COLUNAS_SILVER.items():
    colunas_selecionadas.extend(campos)

colunas_existentes = [col for col in colunas_selecionadas if col in df_bronze.columns]
colunas_faltantes = [col for col in colunas_selecionadas if col not in df_bronze.columns]

print(f"\nSELEÇÃO DE COLUNAS:")
print(f"  • Solicitadas: {len(colunas_selecionadas)}")
print(f"  • Existentes: {len(colunas_existentes)}")
print(f"  • Faltantes: {len(colunas_faltantes)}")

if len(colunas_faltantes) > 0:
    print(f"\nColunas faltantes: {', '.join(colunas_faltantes[:10])}")

df_selected = df_bronze.select(*colunas_existentes)

print(f"\nRedução: {bronze_cols} → {len(colunas_existentes)} colunas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Tipagem e Conversões (CORREÇÃO CRÍTICA - try_to_date)

# COMMAND ----------

print("\nAPLICANDO TIPAGEM CORRETA...")
print("=" * 80)

df_typed = df_selected

campos_data = ['DT_NOTIFIC', 'DT_SIN_PRI', 'DT_INTERNA', 'DT_ENTUTI', 'DT_EVOLUCA']

for campo in campos_data:
    if campo in df_typed.columns:
        df_typed = df_typed.withColumn(
            campo.lower(),
            F.coalesce(
                F.expr(f"try_to_date({campo}, 'dd/MM/yyyy')"),
                F.expr(f"try_to_date({campo}, 'yyyy-MM-dd')")
            )
        )
        print(f"  ✓ {campo} → {campo.lower()} (DateType, try_to_date)")

if 'NU_IDADE_N' in df_typed.columns:
    df_typed = df_typed.withColumn(
        'nu_idade_n',
        F.when(
            (F.col('TP_IDADE') == '3') & F.col('NU_IDADE_N').isNotNull(),
            F.col('NU_IDADE_N').cast(IntegerType())
        ).otherwise(None)
    )
    print(f"  ✓ NU_IDADE_N → nu_idade_n (IntegerType, apenas anos)")

if 'SEM_PRI' in df_typed.columns:
    df_typed = df_typed.withColumn(
        'sem_pri',
        F.when(
            F.col('SEM_PRI').cast(IntegerType()).between(1, 53),
            F.col('SEM_PRI').cast(IntegerType())
        ).otherwise(None)
    )
    print(f"  ✓ SEM_PRI → sem_pri (IntegerType, 1-53)")

campos_categoricos = ['CS_SEXO', 'FEBRE', 'TOSSE', 'DISPNEIA', 'SATURACAO', 
                     'HOSPITAL', 'UTI', 'EVOLUCAO', 'VACINA', 'VACINA_COV']

for campo in campos_categoricos:
    if campo in df_typed.columns:
        df_typed = df_typed.withColumn(campo.lower(), F.col(campo))

outros_campos = ['NU_NOTIFIC', 'SG_UF', 'CO_MUN_RES', 'TP_IDADE', 'CS_RACA', 
                 'GARGANTA', 'DESC_RESP', 'SUPORT_VEN', 'CLASSI_FIN', 
                 'DOSE_1_COV', 'DOSE_2_COV', 'ANO_DADOS']

for campo in outros_campos:
    if campo in df_typed.columns:
        df_typed = df_typed.withColumn(campo.lower(), F.col(campo))

colunas_dropar = [c for c in df_typed.columns if c.isupper()]
df_typed = df_typed.drop(*colunas_dropar)

print(f"\nTipagem concluída")
print(f"  • Total de colunas após tipagem: {len(df_typed.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Tratamento de Código "9" (Ignorado)

# COMMAND ----------

print("\nTRATANDO CÓDIGO '9' (IGNORADO)...")
print("=" * 80)

campos_clean = {
    'evolucao': ['1', '2'],
    'uti': ['1', '2'],
    'vacina': ['1', '2'],
    'hospital': ['1', '2'],
    'cs_sexo': ['1', '2'],
}

for campo, valores_validos in campos_clean.items():
    if campo in df_typed.columns:
        df_typed = df_typed.withColumn(
            f'{campo}_clean',
            F.when(F.col(campo).isin(valores_validos), F.col(campo)).otherwise(None)
        )
        print(f"  ✓ {campo} → {campo}_clean (valores: {valores_validos})")

print(f"\nCampos originais mantêm código '9' para análise descritiva")
print(f"Campos '_clean' usados para cálculo de métricas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Filtros de Qualidade OBRIGATÓRIOS

# COMMAND ----------

print("\nAPLICANDO FILTROS DE QUALIDADE...")
print("=" * 80)

count_before = df_typed.count()
print(f"Registros ANTES dos filtros: {count_before:,}")

print("\nFiltro 1: Campos obrigatórios")
df_filtered = df_typed.filter(
    F.col('nu_notific').isNotNull() &
    F.col('dt_sin_pri').isNotNull() &
    F.col('dt_notific').isNotNull()
)
count_after_f1 = df_filtered.count()
excluidos_f1 = count_before - count_after_f1
print(f"  • NU_NOTIFIC, DT_SIN_PRI, DT_NOTIFIC não podem ser NULL")
print(f"  • Excluídos: {excluidos_f1:,} ({excluidos_f1/count_before*100:.2f}%)")

print("\nFiltro 2: Consistência temporal")
df_filtered = df_filtered.filter(
    (F.col('dt_sin_pri') <= F.col('dt_notific')) &
    ((F.col('dt_interna').isNull()) | (F.col('dt_sin_pri') <= F.col('dt_interna'))) &
    ((F.col('dt_interna').isNull()) | (F.col('dt_evoluca').isNull()) | 
     (F.col('dt_interna') <= F.col('dt_evoluca')))
)
count_after_f2 = df_filtered.count()
excluidos_f2 = count_after_f1 - count_after_f2
print(f"  • Datas inconsistentes (sintomas após notificação, etc)")
print(f"  • Excluídos: {excluidos_f2:,} ({excluidos_f2/count_before*100:.2f}%)")

print("\nFiltro 3: Idade válida")
df_filtered = df_filtered.filter(
    (F.col('nu_idade_n').isNull()) | 
    (F.col('nu_idade_n').between(0, 120))
)
count_after_f3 = df_filtered.count()
excluidos_f3 = count_after_f2 - count_after_f3
print(f"  • Idade entre 0 e 120 anos")
print(f"  • Excluídos: {excluidos_f3:,} ({excluidos_f3/count_before*100:.2f}%)")

print("\nFiltro 4: Duplicatas")
df_filtered = df_filtered.dropDuplicates(['nu_notific'])
count_after_f4 = df_filtered.count()
excluidos_f4 = count_after_f3 - count_after_f4
print(f"  • Manter apenas primeira ocorrência de NU_NOTIFIC")
print(f"  • Excluídos: {excluidos_f4:,} ({excluidos_f4/count_before*100:.2f}%)")

count_final = df_filtered.count()
total_excluidos = count_before - count_final
percentual_excluido = (total_excluidos / count_before) * 100
percentual_mantido = 100 - percentual_excluido

print(f"\nRESUMO DOS FILTROS:")
print(f"  • Total excluído: {total_excluidos:,} ({percentual_excluido:.2f}%)")
print(f"  • Total mantido: {count_final:,} ({percentual_mantido:.2f}%)")

assert percentual_excluido < 40, f"ERRO: Exclusão excessiva ({percentual_excluido:.1f}% > 40%)"
print(f"\nValidação: Perda de registros dentro do esperado (<40%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Feature Engineering

# COMMAND ----------

print("\nFEATURE ENGINEERING...")
print("=" * 80)

df_engineered = df_filtered

print("\n8.1 Features Temporais:")

df_engineered = df_engineered.withColumn(
    'ano',
    F.year(F.col('dt_sin_pri'))
).withColumn(
    'mes',
    F.month(F.col('dt_sin_pri'))
).withColumn(
    'dia_semana',
    F.dayofweek(F.col('dt_sin_pri'))
).withColumn(
    'ano_mes',
    F.concat(
        F.year(F.col('dt_sin_pri')),
        F.lit('-'),
        F.lpad(F.month(F.col('dt_sin_pri')), 2, '0')
    )
)

print(f"  ✓ ano, mes, dia_semana, ano_mes extraídos de dt_sin_pri")

df_engineered = df_engineered.withColumn(
    'tempo_sintoma_notificacao',
    F.datediff(F.col('dt_notific'), F.col('dt_sin_pri'))
).withColumn(
    'tempo_sintoma_internacao',
    F.when(
        F.col('dt_interna').isNotNull(),
        F.datediff(F.col('dt_interna'), F.col('dt_sin_pri'))
    ).otherwise(None)
).withColumn(
    'tempo_internacao',
    F.when(
        F.col('dt_interna').isNotNull() & F.col('dt_evoluca').isNotNull(),
        F.datediff(F.col('dt_evoluca'), F.col('dt_interna'))
    ).otherwise(None)
)

print(f"  ✓ tempo_sintoma_notificacao, tempo_sintoma_internacao, tempo_internacao")

print("\n8.2 Features Demográficas:")

df_engineered = df_engineered.withColumn(
    'faixa_etaria',
    F.when(F.col('nu_idade_n') < 1, '0-1 ano')
    .when(F.col('nu_idade_n') < 5, '1-4 anos')
    .when(F.col('nu_idade_n') < 10, '5-9 anos')
    .when(F.col('nu_idade_n') < 18, '10-17 anos')
    .when(F.col('nu_idade_n') < 30, '18-29 anos')
    .when(F.col('nu_idade_n') < 40, '30-39 anos')
    .when(F.col('nu_idade_n') < 50, '40-49 anos')
    .when(F.col('nu_idade_n') < 60, '50-59 anos')
    .when(F.col('nu_idade_n') < 70, '60-69 anos')
    .when(F.col('nu_idade_n') >= 70, '70+ anos')
    .otherwise('Desconhecido')
)

print(f"  ✓ faixa_etaria (11 categorias)")

df_engineered = df_engineered.withColumn(
    'is_idoso',
    F.when(F.col('nu_idade_n') >= 60, True).otherwise(False)
)

print(f"  ✓ is_idoso (60+ anos)")

print("\n8.3 Flags Booleanas:")

df_engineered = df_engineered.withColumn(
    'is_obito',
    F.when(F.col('evolucao_clean') == '2', True).otherwise(False)
).withColumn(
    'is_cura',
    F.when(F.col('evolucao_clean') == '1', True).otherwise(False)
)

print(f"  ✓ is_obito, is_cura")

df_engineered = df_engineered.withColumn(
    'is_internado',
    F.when(F.col('hospital_clean') == '1', True).otherwise(False)
).withColumn(
    'is_uti',
    F.when(F.col('uti_clean') == '1', True).otherwise(False)
)

print(f"  ✓ is_internado, is_uti")

df_engineered = df_engineered.withColumn(
    'is_vacinado',
    F.when(
        (F.col('vacina_clean') == '1') | (F.col('vacina_cov') == '1'),
        True
    ).otherwise(False)
)

print(f"  ✓ is_vacinado (VACINA=1 OU VACINA_COV=1)")

if 'febre' in df_engineered.columns:
    df_engineered = df_engineered.withColumn(
        'has_febre',
        F.when(F.col('febre') == '1', True).otherwise(False)
    )
    print(f"  ✓ has_febre")

if 'tosse' in df_engineered.columns:
    df_engineered = df_engineered.withColumn(
        'has_tosse',
        F.when(F.col('tosse') == '1', True).otherwise(False)
    )
    print(f"  ✓ has_tosse")

if 'dispneia' in df_engineered.columns:
    df_engineered = df_engineered.withColumn(
        'has_dispneia',
        F.when(F.col('dispneia') == '1', True).otherwise(False)
    )
    print(f"  ✓ has_dispneia")

print("\n8.4 Flags de Qualidade:")

df_engineered = df_engineered.withColumn(
    '_data_valida',
    F.when(
        (F.col('dt_sin_pri').isNotNull()) &
        (F.col('dt_notific').isNotNull()) &
        (F.col('dt_sin_pri') <= F.col('dt_notific')),
        True
    ).otherwise(False)
)

print(f"  ✓ _data_valida (consistência temporal)")

df_engineered = df_engineered.withColumn(
    '_completude_score',
    (
        F.when(F.col('cs_sexo_clean').isNotNull(), 1).otherwise(0) +
        F.when(F.col('nu_idade_n').isNotNull(), 1).otherwise(0) +
        F.when(F.col('evolucao_clean').isNotNull(), 1).otherwise(0) +
        F.when(F.col('sg_uf').isNotNull(), 1).otherwise(0)
    ) / 4.0
)

print(f"  ✓ _completude_score (0.0 a 1.0)")

print(f"\nFeature Engineering concluído")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Metadados de Rastreabilidade

# COMMAND ----------

print("\nADICIONANDO METADADOS...")
print("=" * 80)

df_final = df_engineered.withColumn(
    '_silver_processed_at',
    F.current_timestamp()
).withColumn(
    '_process_id',
    F.lit(PROCESS_ID)
).withColumn(
    '_source_year',
    F.col('ano_dados')
)

print(f"Metadados adicionados:")
print(f"  • _silver_processed_at: timestamp do processamento")
print(f"  • _process_id: {PROCESS_ID}")
print(f"  • _source_year: ano original dos dados Bronze")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Escrita da Tabela Silver (Particionada)

# COMMAND ----------

print("\nESCREVENDO TABELA SILVER...")
print("=" * 80)

silver_count = df_final.count()
silver_cols = len(df_final.columns)

print(f"\nESTATÍSTICAS FINAIS:")
print(f"  • Registros Silver: {silver_count:,}")
print(f"  • Colunas Silver: {silver_cols}")
print(f"  • Redução de registros: {(1 - silver_count/bronze_count)*100:.2f}%")
print(f"  • Redução de colunas: {bronze_cols} → {silver_cols}")

print(f"\nEscrevendo em: {TABLE_SILVER}")
print(f"  • Modo: OVERWRITE")
print(f"  • Particionamento: ano + mes")
print(f"  • Formato: Delta Lake")

df_final.write \
    .mode('overwrite') \
    .partitionBy('ano', 'mes') \
    .option('overwriteSchema', True) \
    .option('mergeSchema', False) \
    .saveAsTable(TABLE_SILVER)

print(f"\nTabela Silver criada com sucesso!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Otimização (Z-Ordering)

# COMMAND ----------

print("\nOTIMIZANDO TABELA SILVER...")
print("=" * 80)

print(f"Aplicando Z-ORDER BY (dt_sin_pri, sg_uf)...")

spark.sql(f"""
    OPTIMIZE {TABLE_SILVER}
    ZORDER BY (dt_sin_pri, sg_uf)
""")

print(f"Z-Ordering aplicado")

print(f"\nGerando estatísticas da tabela...")

spark.sql(f"ANALYZE TABLE {TABLE_SILVER} COMPUTE STATISTICS")

print(f"Estatísticas computadas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Validações Pós-Processamento

# COMMAND ----------

print("\nVALIDAÇÕES PÓS-PROCESSAMENTO...")
print("=" * 80)

df_silver_check = spark.table(TABLE_SILVER)

print("\nValidação 1: Perda de Registros")
perda_percentual = (1 - silver_count / bronze_count) * 100

print(f"  • Bronze: {bronze_count:,}")
print(f"  • Silver: {silver_count:,}")
print(f"  • Perda: {perda_percentual:.2f}%")

assert perda_percentual < 60, f"ERRO: Perda excessiva de registros ({perda_percentual:.1f}% > 60%)"
print(f"  ✓ OK: Perda dentro do esperado (<60%)")

print("\nValidação 2: Unicidade de NU_NOTIFIC")
distinct_count = df_silver_check.select('nu_notific').distinct().count()

print(f"  • Total de registros: {silver_count:,}")
print(f"  • NU_NOTIFIC distintos: {distinct_count:,}")

assert distinct_count == silver_count, "ERRO: NU_NOTIFIC não é único!"
print(f"  ✓ OK: NU_NOTIFIC é chave primária única")

print("\nValidação 3: Campos Obrigatórios")
campos_obrigatorios = ['nu_notific', 'dt_sin_pri', 'dt_notific', 'ano', 'mes']

for campo in campos_obrigatorios:
    null_count = df_silver_check.filter(F.col(campo).isNull()).count()
    print(f"  • {campo}: {null_count} nulls")
    assert null_count == 0, f"ERRO: Campo obrigatório {campo} contém NULLs!"

print(f"  ✓ OK: Todos os campos obrigatórios estão preenchidos")

print("\nValidação 4: Consistência Temporal")
inconsistencias = df_silver_check.filter(
    F.col('dt_sin_pri') > F.col('dt_notific')
).count()

print(f"  • Registros com dt_sin_pri > dt_notific: {inconsistencias}")
assert inconsistencias == 0, "ERRO: Existem inconsistências temporais!"
print(f"  ✓ OK: Todas as datas são consistentes")

print("\nValidação 5: Distribuição por Ano")
print("  Distribuição Silver:")
df_silver_check.groupBy('ano').count().orderBy('ano').show()

print("\nValidação 6: Partições")
partitions = spark.sql(f"SHOW PARTITIONS {TABLE_SILVER}").count()
print(f"  • Total de partições: {partitions}")
print(f"  ✓ OK: Tabela particionada por ano + mes")

print("\n" + "=" * 80)
print("TODAS AS VALIDAÇÕES PASSARAM COM SUCESSO!")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Teste das Métricas Epidemiológicas

# COMMAND ----------

print("\nTESTANDO MÉTRICAS EPIDEMIOLÓGICAS...")
print("=" * 80)

print("\nMÉTRICA 1: Taxa de Mortalidade por Ano")
print("─" * 80)

spark.sql(f"""
    SELECT 
        ano,
        COUNT(*) as total_casos,
        SUM(CASE WHEN evolucao_clean = '2' THEN 1 ELSE 0 END) as obitos,
        SUM(CASE WHEN evolucao_clean = '1' THEN 1 ELSE 0 END) as curas,
        SUM(CASE WHEN evolucao_clean IS NOT NULL THEN 1 ELSE 0 END) as casos_com_desfecho,
        ROUND(
            SUM(CASE WHEN evolucao_clean = '2' THEN 1 ELSE 0 END) * 100.0 / 
            NULLIF(SUM(CASE WHEN evolucao_clean IS NOT NULL THEN 1 ELSE 0 END), 0),
            2
        ) as taxa_mortalidade
    FROM {TABLE_SILVER}
    GROUP BY ano
    ORDER BY ano
""").show()

print("\nMÉTRICA 2: Taxa de Ocupação UTI por Ano")
print("─" * 80)

spark.sql(f"""
    SELECT 
        ano,
        SUM(CASE WHEN is_internado THEN 1 ELSE 0 END) as total_internados,
        SUM(CASE WHEN is_uti THEN 1 ELSE 0 END) as casos_uti,
        ROUND(
            SUM(CASE WHEN is_uti THEN 1 ELSE 0 END) * 100.0 / 
            NULLIF(SUM(CASE WHEN is_internado THEN 1 ELSE 0 END), 0),
            2
        ) as taxa_uti
    FROM {TABLE_SILVER}
    GROUP BY ano
    ORDER BY ano
""").show()

print("\nMÉTRICA 3: Taxa de Vacinação por Ano")
print("─" * 80)

spark.sql(f"""
    SELECT 
        ano,
        COUNT(*) as total_casos,
        SUM(CASE WHEN vacina_clean = '1' THEN 1 ELSE 0 END) as vacinados,
        SUM(CASE WHEN vacina_clean IS NOT NULL THEN 1 ELSE 0 END) as casos_com_info,
        ROUND(
            SUM(CASE WHEN vacina_clean = '1' THEN 1 ELSE 0 END) * 100.0 / 
            NULLIF(SUM(CASE WHEN vacina_clean IS NOT NULL THEN 1 ELSE 0 END), 0),
            2
        ) as taxa_vacinacao
    FROM {TABLE_SILVER}
    GROUP BY ano
    ORDER BY ano
""").show()

print("\nMÉTRICA 4: Casos e Crescimento Mensal")
print("─" * 80)

spark.sql(f"""
    WITH casos_mensais AS (
        SELECT 
            ano_mes,
            COUNT(*) as casos
        FROM {TABLE_SILVER}
        GROUP BY ano_mes
        ORDER BY ano_mes
    ),
    crescimento AS (
        SELECT 
            ano_mes,
            casos,
            LAG(casos) OVER (ORDER BY ano_mes) as casos_mes_anterior,
            ROUND(
                (casos - LAG(casos) OVER (ORDER BY ano_mes)) * 100.0 / 
                NULLIF(LAG(casos) OVER (ORDER BY ano_mes), 0),
                2
            ) as crescimento_pct
        FROM casos_mensais
    )
    SELECT *
    FROM crescimento
    ORDER BY ano_mes DESC
    LIMIT 12
""").show()

print("\nTodas as 4 métricas epidemiológicas calculadas com sucesso!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Queries de Verificação

# COMMAND ----------

print("\nQUERIES DE VERIFICAÇÃO...")
print("=" * 80)

print("\nQuery 1: Resumo Geral da Tabela Silver")
spark.sql(f"""
    SELECT 
        'Total de Registros' AS metrica,
        CAST(COUNT(*) AS STRING) AS valor
    FROM {TABLE_SILVER}

    UNION ALL

    SELECT 
        'Período',
        CONCAT(
            CAST(MIN(dt_sin_pri) AS STRING),
            ' a ',
            CAST(MAX(dt_sin_pri) AS STRING)
        )
    FROM {TABLE_SILVER}

    UNION ALL

    SELECT 
        'UFs Distintas',
        CAST(COUNT(DISTINCT sg_uf) AS STRING)
    FROM {TABLE_SILVER}

    UNION ALL

    SELECT 
        'Idade Média',
        CAST(ROUND(AVG(nu_idade_n), 1) AS STRING)
    FROM {TABLE_SILVER}
    WHERE nu_idade_n IS NOT NULL

    UNION ALL

    SELECT 
        'Taxa de Óbitos Geral (%)',
        CAST(
            ROUND(
                SUM(CASE WHEN is_obito THEN 1 ELSE 0 END) * 100.0 /
                NULLIF(SUM(CASE WHEN evolucao_clean IS NOT NULL THEN 1 ELSE 0 END), 0),
                2
            ) AS STRING
        )
    FROM {TABLE_SILVER}
""").show(truncate=False)


print("\nQuery 2: Top 10 UFs com Mais Casos")
spark.sql(f"""
    SELECT 
        sg_uf,
        COUNT(*) as casos,
        ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM {TABLE_SILVER}), 2) as percentual
    FROM {TABLE_SILVER}
    WHERE sg_uf IS NOT NULL
    GROUP BY sg_uf
    ORDER BY casos DESC
    LIMIT 10
""").show()

print("\nQuery 3: Distribuição por Faixa Etária")
spark.sql(f"""
    SELECT 
        faixa_etaria,
        COUNT(*) as casos,
        ROUND(AVG(CASE WHEN is_obito THEN 100.0 ELSE 0 END), 2) as taxa_mortalidade
    FROM {TABLE_SILVER}
    WHERE faixa_etaria != 'Desconhecido'
    GROUP BY faixa_etaria
    ORDER BY 
        CASE faixa_etaria
            WHEN '0-1 ano' THEN 1
            WHEN '1-4 anos' THEN 2
            WHEN '5-9 anos' THEN 3
            WHEN '10-17 anos' THEN 4
            WHEN '18-29 anos' THEN 5
            WHEN '30-39 anos' THEN 6
            WHEN '40-49 anos' THEN 7
            WHEN '50-59 anos' THEN 8
            WHEN '60-69 anos' THEN 9
            WHEN '70+ anos' THEN 10
        END
""").show()

print("\nQuery 4: Score de Completude")
spark.sql(f"""
    SELECT 
        CASE 
            WHEN _completude_score = 1.0 THEN 'Completo (100%)'
            WHEN _completude_score >= 0.75 THEN 'Bom (75-99%)'
            WHEN _completude_score >= 0.5 THEN 'Médio (50-74%)'
            ELSE 'Baixo (<50%)'
        END as nivel_completude,
        COUNT(*) as casos,
        ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM {TABLE_SILVER}), 2) as percentual
    FROM {TABLE_SILVER}
    GROUP BY 
        CASE 
            WHEN _completude_score = 1.0 THEN 'Completo (100%)'
            WHEN _completude_score >= 0.75 THEN 'Bom (75-99%)'
            WHEN _completude_score >= 0.5 THEN 'Médio (50-74%)'
            ELSE 'Baixo (<50%)'
        END
    ORDER BY percentual DESC
""").show()

print("\nQuery 5: Tempos Médios Entre Eventos (dias)")
spark.sql(f"""
    SELECT 
        'Sintoma → Notificação' as evento,
        ROUND(AVG(tempo_sintoma_notificacao), 1) as media_dias,
        ROUND(PERCENTILE(tempo_sintoma_notificacao, 0.5), 1) as mediana_dias
    FROM {TABLE_SILVER}
    WHERE tempo_sintoma_notificacao IS NOT NULL
    
    UNION ALL
    
    SELECT 
        'Sintoma → Internação',
        ROUND(AVG(tempo_sintoma_internacao), 1),
        ROUND(PERCENTILE(tempo_sintoma_internacao, 0.5), 1)
    FROM {TABLE_SILVER}
    WHERE tempo_sintoma_internacao IS NOT NULL
    
    UNION ALL
    
    SELECT 
        'Duração da Internação',
        ROUND(AVG(tempo_internacao), 1),
        ROUND(PERCENTILE(tempo_internacao, 0.5), 1)
    FROM {TABLE_SILVER}
    WHERE tempo_internacao IS NOT NULL
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Estatísticas Detalhadas da Tabela

# COMMAND ----------

print("\nESTATÍSTICAS DETALHADAS DA TABELA SILVER...")
print("=" * 80)

print("\nSchema da Tabela:")
spark.sql(f"DESCRIBE TABLE {TABLE_SILVER}").show(100, truncate=False)

print("\nHistória da Tabela (Delta Lake):")
spark.sql(f"DESCRIBE HISTORY {TABLE_SILVER}").limit(5).show(truncate=False)

print("\nDetalhes das Partições:")
partition_info = spark.sql(f"SHOW PARTITIONS {TABLE_SILVER}")
partition_count = partition_info.count()

print(f"  • Total de partições: {partition_count}")
print(f"  • Primeiras 10 partições:")
partition_info.limit(10).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Documentação e Resumo Final

# COMMAND ----------

print("\nDOCUMENTAÇÃO - TABELA SILVER")
print("=" * 80)

documentacao = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      TABELA SILVER - SRAG CLEANED DATA                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Data de Processamento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Process ID: {PROCESS_ID}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ESTATÍSTICAS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Tabela: {TABLE_SILVER}
  Fonte: {TABLE_BRONZE}
  
  Registros Bronze: {bronze_count:,}
  Registros Silver: {silver_count:,}
  Perda: {perda_percentual:.2f}%
  
  Colunas Bronze: {bronze_cols}
  Colunas Silver: {silver_cols}
  Redução: {((1 - silver_cols/bronze_cols)*100):.1f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRANSFORMAÇÕES APLICADAS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Seleção de colunas críticas (197 → {silver_cols})
  2. Tipagem correta com try_to_date (Spark 4 compatível)
  3. Tratamento de código "9" (colunas _clean criadas)
  4. Filtros de qualidade aplicados
  5. Feature engineering (temporal, demográfica, flags)
  6. Validações de consistência temporal
  7. Remoção de duplicatas
  8. Metadados de rastreabilidade

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CAMPOS PRINCIPAIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Identificação:
    • nu_notific (String) - Chave primária única
  
  Temporais:
    • dt_sin_pri (Date) - Data primeiros sintomas
    • dt_notific (Date) - Data de notificação
    • ano, mes (Integer) - Extraídos de dt_sin_pri
    • ano_mes (String) - Formato YYYY-MM
  
  Demografia:
    • sg_uf (String) - Unidade Federativa
    • nu_idade_n (Integer) - Idade em anos
    • faixa_etaria (String) - 11 categorias
    • cs_sexo (String) - 1=M, 2=F, 9=Ignorado
    • cs_sexo_clean (String) - Sem código 9
  
  Clínicos:
    • evolucao (String) - 1=Cura, 2=Óbito, 9=Ignorado
    • evolucao_clean (String) - Para cálculo de mortalidade
    • hospital, uti, vacina (String) - Com versões _clean
  
  Features Derivadas:
    • tempo_sintoma_notificacao (Integer) - Dias
    • tempo_sintoma_internacao (Integer) - Dias
    • tempo_internacao (Integer) - Dias
  
  Flags Booleanas:
    • is_obito, is_cura, is_internado, is_uti, is_vacinado
    • is_idoso (60+ anos)
    • has_febre, has_tosse, has_dispneia
  
  Metadados:
    • _silver_processed_at (Timestamp)
    • _process_id (String)
    • _data_valida (Boolean)
    • _completude_score (Double) - 0.0 a 1.0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OTIMIZAÇÕES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Formato: Delta Lake
  • Particionamento: ano + mes
  • Z-Ordering: dt_sin_pri, sg_uf
  • Estatísticas: Computadas

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MÉTRICAS EPIDEMIOLÓGICAS DISPONÍVEIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Taxa de Mortalidade:
     SUM(is_obito) / COUNT(*) WHERE evolucao_clean IS NOT NULL
  
  2. Taxa de Ocupação UTI:
     SUM(is_uti) / SUM(is_internado)
  
  3. Taxa de Vacinação:
     SUM(is_vacinado) / COUNT(*) WHERE vacina_clean IS NOT NULL
  
  4. Taxa de Crescimento:
     Calcular variação mensal usando campo ano_mes

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUERIES ÚTEIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
-- Casos por mês
SELECT ano_mes, COUNT(*) FROM {TABLE_SILVER} GROUP BY ano_mes ORDER BY ano_mes;

-- Taxa de mortalidade por UF
SELECT 
    sg_uf,
    SUM(CASE WHEN is_obito THEN 1 ELSE 0 END) * 100.0 / 
    NULLIF(SUM(CASE WHEN evolucao_clean IS NOT NULL THEN 1 ELSE 0 END), 0) AS taxa
FROM {TABLE_SILVER}
GROUP BY sg_uf
ORDER BY taxa DESC;

-- Casos por faixa etária
SELECT faixa_etaria, COUNT(*) 
FROM {TABLE_SILVER} 
WHERE faixa_etaria != 'Desconhecido'
GROUP BY faixa_etaria;

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OBSERVAÇÕES IMPORTANTES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Código "9" = "Ignorado" no padrão DATASUS (NÃO é NULL)
  • Campos originais mantêm "9" para análise descritiva
  • Campos "_clean" excluem "9" para cálculo de métricas
  • Registros com datas inconsistentes foram excluídos
  • Duplicatas por nu_notific foram removidas
  • Tabela é idempotente (reprocessável com OVERWRITE)
  • Conversão de datas usa try_to_date (compatível Spark 4)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRÓXIMOS PASSOS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Camada Gold: Criar agregações para dashboards
  2. Views materializadas para queries frequentes
  3. Alertas de qualidade de dados
  4. Integração com sistema RAG
"""

print(documentacao)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 17. Finalização

# COMMAND ----------

print("\n" + "=" * 80)
print("PROCESSAMENTO DA CAMADA SILVER CONCLUÍDO COM SUCESSO!")
print("=" * 80)

resumo_final = f"""
STATUS: CONCLUÍDO

Resumo da Execução:
  • Process ID: {PROCESS_ID}
  • Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  
  • Bronze → Silver: {bronze_count:,} → {silver_count:,} registros
  • Perda: {perda_percentual:.2f}% (dentro do esperado)
  • Redução de colunas: {bronze_cols} → {silver_cols}

Tabela Criada:
  • {TABLE_SILVER}
  • Particionada por: ano + mes
  • Z-ordered por: dt_sin_pri, sg_uf
  • Formato: Delta Lake

Validações:
  • Perda de registros < 60%
  • NU_NOTIFIC é único
  • Campos obrigatórios preenchidos
  • Consistência temporal validada
  • Partições criadas
  • Métricas epidemiológicas calculáveis

Correções Spark 4:
  • try_to_date implementado para todas as datas
  • Compatível com Databricks Serverless
  • Suporte a formatos mistos (dd/MM/yyyy e yyyy-MM-dd)

Próximos Passos:
  → Notebook 05: Gold Layer (agregações)
  → Criar views para dashboards
  → Integrar com sistema RAG
"""

print(resumo_final)

print("\n" + "=" * 80)
print(f"Timestamp final: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC 
# MAGIC ## Notas de Uso
# MAGIC 
# MAGIC ### Como Re-executar
# MAGIC 
# MAGIC Este notebook é IDEMPOTENTE e pode ser executado múltiplas vezes com OVERWRITE sem criar duplicatas.
# MAGIC 
# MAGIC ### Como Consultar
# MAGIC 
# MAGIC ```sql
# MAGIC -- Consulta básica
# MAGIC SELECT * FROM workspace.silver.silver_srag_clean LIMIT 10;
# MAGIC 
# MAGIC -- Taxa de mortalidade por ano
# MAGIC SELECT 
# MAGIC     ano,
# MAGIC     ROUND(AVG(CASE WHEN is_obito THEN 100.0 ELSE 0 END), 2) AS taxa_mortalidade
# MAGIC FROM workspace.silver.silver_srag_clean
# MAGIC WHERE evolucao_clean IS NOT NULL
# MAGIC GROUP BY ano;
# MAGIC ```
# MAGIC 
# MAGIC ### Integração com Gold
# MAGIC 
# MAGIC O notebook Gold deve:
# MAGIC 1. Ler silver.silver_srag_clean
# MAGIC 2. Criar agregações por período/região
# MAGIC 3. Calcular métricas epidemiológicas
# MAGIC 4. Criar tabelas para dashboards
# MAGIC 
# MAGIC ### Manutenção
# MAGIC 
# MAGIC ```sql
# MAGIC -- Vacuum (deletar arquivos antigos após 7 dias)
# MAGIC VACUUM workspace.silver.silver_srag_clean RETAIN 168 HOURS;
# MAGIC 
# MAGIC -- Recomputar estatísticas
# MAGIC ANALYZE TABLE workspace.silver.silver_srag_clean COMPUTE STATISTICS;
# MAGIC 
# MAGIC -- Ver histórico Delta
# MAGIC DESCRIBE HISTORY workspace.silver.silver_srag_clean;
# MAGIC ```
# MAGIC 
# MAGIC ### Campos Críticos
# MAGIC 
# MAGIC **Para Métricas (usar campos _clean)**:
# MAGIC - `evolucao_clean`: Exclui código "9"
# MAGIC - `uti_clean`: Exclui código "9"
# MAGIC - `vacina_clean`: Exclui código "9"
# MAGIC - `hospital_clean`: Exclui código "9"
# MAGIC 
# MAGIC **Para Análise Descritiva (usar campos originais)**:
# MAGIC - `evolucao`: Inclui "9" (Ignorado)
# MAGIC - `uti`: Inclui "9"
# MAGIC - `vacina`: Inclui "9"
# MAGIC 
# MAGIC ### Alertas Recomendados
# MAGIC 
# MAGIC Configure alertas para:
# MAGIC - Perda de registros > 60%
# MAGIC - Duplicatas em nu_notific
# MAGIC - Taxa de mortalidade fora do esperado
# MAGIC - Falha no Z-ordering
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC **Desenvolvido para**: Sistema RAG - Monitoramento Epidemiológico  
# MAGIC **Ambiente**: Databricks Serverless + Unity Catalog  
# MAGIC **Versão**: 1.0.0  
# MAGIC **Arquitetura**: Medallion (Bronze → Silver → Gold)