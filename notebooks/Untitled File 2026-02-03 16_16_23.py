# Databricks notebook source
# MAGIC %md
# MAGIC # Camada Silver — Transformação e Limpeza de Dados SRAG
# MAGIC
# MAGIC **Projeto**: Sistema RAG para Monitoramento Epidemiológico
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Objetivo
# MAGIC
# MAGIC Produzir `silver.silver_srag_clean`: tabela limpa, tipada e epidemiologicamente
# MAGIC correta, pronta para agregações Gold e para o sistema RAG.
# MAGIC
# MAGIC - **Input** : `data_original.bronze_srag_raw`
# MAGIC - **Output**: `silver.silver_srag_clean`
# MAGIC - **Modo**  : `overwrite` — tabela idempotente, segura para reprocessamento
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Regras de Limpeza — Campos `_clean`
# MAGIC
# MAGIC Campos `_clean` são as versões sanitizadas usadas em todas as métricas.
# MAGIC Campos originais são mantidos intactos para análise descritiva.
# MAGIC
# MAGIC | Campo original | Campo `_clean` | Valores válidos | Mapeado para NULL |
# MAGIC |---|---|---|---|
# MAGIC | `EVOLUCAO` | `evolucao_clean` | `'1'`, `'2'` | `'3'`, `'9'`, vazio, NULL |
# MAGIC | `HOSPITAL`  | `hospital_clean` | `'1'`, `'2'` | `'9'`, vazio, NULL |
# MAGIC | `UTI`       | `uti_clean`      | `'1'`, `'2'` | `'9'`, vazio, NULL |
# MAGIC | `VACINA`    | `vacina_clean`   | `'1'`, `'2'` | `'9'`, vazio, NULL |
# MAGIC | `CS_SEXO`   | `cs_sexo_clean`  | `'1'`, `'2'` | `'9'`, `'M'`→`'1'`, `'F'`→`'2'`, outros→NULL |
# MAGIC
# MAGIC > **Nota**: código `'9'` = "Ignorado" no padrão DATASUS — semanticamente distinto
# MAGIC > de NULL ("não informado"). Ambos são excluídos dos denominadores das métricas.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Regras de Mortalidade (EVOLUCAO)
# MAGIC
# MAGIC **Mortalidade SRAG estrita** usa apenas `evolucao_clean IN ('1', '2')`:
# MAGIC
# MAGIC | Código | Significado | Uso na métrica |
# MAGIC |---|---|---|
# MAGIC | `'1'` | Cura | ✅ Denominador e numerador (curas) |
# MAGIC | `'2'` | Óbito por SRAG | ✅ Denominador e numerador (óbitos) |
# MAGIC | `'3'` | Óbito por outras causas | ❌ **Excluído** — mapeado para NULL em `evolucao_clean` |
# MAGIC | `'9'` | Ignorado | ❌ Excluído |
# MAGIC | NULL  | Não informado | ❌ Excluído |
# MAGIC
# MAGIC **Justificativa epidemiológica**: o projeto mensura mortalidade *atribuível à SRAG*.
# MAGIC `EVOLUCAO = '3'` representa desfecho por causa não relacionada à SRAG e introduziria
# MAGIC viés no numerador. Para mortalidade hospitalar geral, incluir `'3'` na camada Gold
# MAGIC como métrica separada.
# MAGIC
# MAGIC **`is_obito_srag`** = `evolucao_clean = '2'` (óbito estritamente por SRAG).
# MAGIC Qualquer denominador de mortalidade deve filtrar `evolucao_clean IS NOT NULL`.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Regra de Idade (TP_IDADE)
# MAGIC
# MAGIC `idade_anos` é calculada **somente** quando `TP_IDADE = '3'` (anos completos,
# MAGIC dicionário SIVEP-Gripe). Registros com `TP_IDADE IN ('1','2')` (dias e meses,
# MAGIC respectivamente) recebem `idade_anos = NULL`.
# MAGIC
# MAGIC **Justificativa**: converter dias ou meses em anos fracionários introduz imprecisão
# MAGIC e não é necessário para as métricas atuais. A faixa etária desses registros fica
# MAGIC em `'Desconhecido'` e são excluídos dos cálculos por faixa.
# MAGIC
# MAGIC **Recomendação futura**: criar `idade_dias_equiv` (conversão unificada para dias)
# MAGIC se análises pediátricas forem necessárias na Gold.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Regra de Denominador — Taxa UTI (CRÍTICO)
# MAGIC
# MAGIC A taxa UTI **deve** ser calculada somente sobre hospitalizados com informação válida
# MAGIC de UTI. Registros onde `hospital_clean != '1'` são excluídos de **ambos**
# MAGIC numerador e denominador, mesmo que `uti_clean` esteja preenchido.
# MAGIC
# MAGIC ```
# MAGIC numerador   = hospital_clean = '1' AND uti_clean = '1'
# MAGIC denominador = hospital_clean = '1'   (independente de uti_clean)
# MAGIC ```
# MAGIC
# MAGIC A flag `is_uti_valido` reflete exatamente essa regra e é a única que deve ser
# MAGIC usada em agregações de taxa UTI.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Compatibilidade de Runtime
# MAGIC
# MAGIC - Datas parseadas com `F.coalesce(F.to_date(...), F.to_date(...))` — nunca
# MAGIC   lança exceção, retorna NULL para formatos não reconhecidos.
# MAGIC   **`try_to_date` não é usado** por ausência em runtimes não-ML do Databricks.
# MAGIC - Sem SciPy, Seaborn ou outras dependências opcionais.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from datetime import datetime

print("=" * 70)
print("CAMADA SILVER — TRANSFORMAÇÃO E LIMPEZA")
print(f"Início : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Spark  : {spark.version}")
print("=" * 70)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuração

# COMMAND ----------

CATALOG       = "dbx_srag_lab"
SCHEMA_BRONZE = "data_original"
SCHEMA_SILVER = "silver"

TABLE_BRONZE = f"{CATALOG}.{SCHEMA_BRONZE}.bronze_srag_raw"
TABLE_SILVER = f"{CATALOG}.{SCHEMA_SILVER}.silver_srag_clean"

PROCESS_ID = datetime.now().strftime('%Y%m%d_%H%M%S')

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA_SILVER}")

print(f"Fonte      : {TABLE_BRONZE}")
print(f"Destino    : {TABLE_SILVER}")
print(f"Process ID : {PROCESS_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Funções auxiliares

# COMMAND ----------

def _parse_date(col_name: str):
    """
    Parse tolerante de data via coalesce — dois formatos, sem try_to_date.
    to_date retorna NULL para formato não reconhecido (nunca lança exceção).
    Compatível com todos os runtimes Databricks.
    """
    return F.coalesce(
        F.to_date(F.col(col_name), 'dd/MM/yyyy'),
        F.to_date(F.col(col_name), 'yyyy-MM-dd'),
    )


def _clean_code9(col_name: str, valid_values: list):
    """
    Sanitiza campo categórico:
      - retorna o valor original se estiver em valid_values
      - retorna NULL para '9', string vazia ou qualquer outro valor
    Usado para produzir colunas *_clean usadas em métricas.
    """
    return F.when(
        F.col(col_name).isin(valid_values),
        F.col(col_name)
    ).otherwise(None)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Leitura da Bronze

# COMMAND ----------

df_bronze   = spark.table(TABLE_BRONZE)
bronze_count = df_bronze.count()
bronze_cols  = len(df_bronze.columns)

print(f"Registros : {bronze_count:,}")
print(f"Colunas   : {bronze_cols}")
df_bronze.groupBy("ANO_DADOS").count().orderBy("ANO_DADOS").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Seleção de colunas

# COMMAND ----------

COLUNAS_SILVER = [
    # identificação
    'NU_NOTIFIC',
    # temporal
    'DT_NOTIFIC', 'DT_SIN_PRI', 'SEM_PRI', 'ANO_DADOS',
    # demografia
    'CS_SEXO', 'NU_IDADE_N', 'TP_IDADE', 'SG_UF', 'CO_MUN_RES', 'CS_RACA',
    # sintomas
    'FEBRE', 'TOSSE', 'DISPNEIA', 'GARGANTA', 'SATURACAO', 'DESC_RESP',
    # internação
    'HOSPITAL', 'DT_INTERNA', 'UTI', 'DT_ENTUTI', 'SUPORT_VEN',
    # desfecho
    'EVOLUCAO', 'DT_EVOLUCA', 'CLASSI_FIN',
    # vacinação
    'VACINA', 'VACINA_COV', 'DOSE_1_COV', 'DOSE_2_COV',
]

cols_exist   = [c for c in COLUNAS_SILVER if c in df_bronze.columns]
cols_missing = [c for c in COLUNAS_SILVER if c not in df_bronze.columns]

print(f"Solicitadas : {len(COLUNAS_SILVER)} | Existentes : {len(cols_exist)} | "
      f"Ausentes : {len(cols_missing)}")
if cols_missing:
    print(f"Ausentes    : {', '.join(cols_missing)}")

df_sel = df_bronze.select(*cols_exist)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Tipagem e parsing de datas
# MAGIC
# MAGIC Datas parseadas com `F.coalesce(F.to_date(...), F.to_date(...))`.
# MAGIC `try_to_date` **não é utilizado** — indisponível em runtimes não-ML.
# MAGIC
# MAGIC `idade_anos`: somente quando `TP_IDADE = '3'` (anos completos).
# MAGIC Veja seção "Regra de Idade" no cabeçalho.

# COMMAND ----------

CAMPOS_DATA = ['DT_NOTIFIC', 'DT_SIN_PRI', 'DT_INTERNA', 'DT_ENTUTI', 'DT_EVOLUCA']

df_typed = df_sel
for campo in CAMPOS_DATA:
    if campo in df_typed.columns:
        df_typed = df_typed.withColumn(campo.lower(), _parse_date(campo)) \
                           .drop(campo)

# idade_anos: apenas TP_IDADE='3' (anos completos)
if 'NU_IDADE_N' in df_typed.columns:
    df_typed = df_typed \
        .withColumn('idade_anos',
            F.when(
                (F.col('TP_IDADE') == '3') & F.col('NU_IDADE_N').isNotNull(),
                F.col('NU_IDADE_N').cast(IntegerType())
            ).otherwise(None)
        ) \
        .drop('NU_IDADE_N')

# semana epidemiológica: apenas valores 1–53
if 'SEM_PRI' in df_typed.columns:
    df_typed = df_typed.withColumn('sem_pri',
        F.when(F.col('SEM_PRI').cast(IntegerType()).between(1, 53),
               F.col('SEM_PRI').cast(IntegerType()))
        .otherwise(None)
    ).drop('SEM_PRI')

# lowercase nos demais campos originais (mantidos para análise descritiva)
cols_upper = [c for c in df_typed.columns if c == c.upper()]
for c in cols_upper:
    df_typed = df_typed.withColumnRenamed(c, c.lower())

print(f"Colunas após tipagem: {len(df_typed.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Campos `_clean` — exclusão de código '9', vazios e NULL
# MAGIC
# MAGIC Veja tabela de regras no cabeçalho. Campos originais (com `'9'`) são mantidos.

# COMMAND ----------

# cs_sexo: normaliza encoding misto (M/F numérico ou alfanumérico) antes do _clean
if 'cs_sexo' in df_typed.columns:
    df_typed = df_typed.withColumn('cs_sexo',
        F.when(F.col('cs_sexo') == 'M', '1')
         .when(F.col('cs_sexo') == 'F', '2')
         .otherwise(F.col('cs_sexo'))
    )

CAMPOS_CLEAN = {
    'evolucao' : ['1', '2'],   # '3' (óbito outras causas) excluído — ver Regras de Mortalidade
    'hospital' : ['1', '2'],
    'uti'      : ['1', '2'],
    'vacina'   : ['1', '2'],
    'cs_sexo'  : ['1', '2'],
}

for campo, validos in CAMPOS_CLEAN.items():
    if campo in df_typed.columns:
        df_typed = df_typed.withColumn(f'{campo}_clean', _clean_code9(campo, validos))

print("Campos _clean criados:")
for campo in CAMPOS_CLEAN:
    print(f"  {campo}_clean  ← válidos {CAMPOS_CLEAN[campo]} | resto → NULL")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Filtros de qualidade

# COMMAND ----------

n0 = df_typed.count()
print(f"Registros antes dos filtros : {n0:,}")

# F1: campos obrigatórios
df_f = df_typed.filter(
    F.col('nu_notific').isNotNull() &
    F.col('dt_sin_pri').isNotNull() &
    F.col('dt_notific').isNotNull()
)
n1 = df_f.count()
print(f"F1 (campos obrigatórios)    : -{n0-n1:,}  → {n1:,}")

# F2: consistência temporal
df_f = df_f.filter(
    (F.col('dt_sin_pri') <= F.col('dt_notific')) &
    (F.col('dt_interna').isNull() | (F.col('dt_sin_pri') <= F.col('dt_interna'))) &
    (F.col('dt_interna').isNull() | F.col('dt_evoluca').isNull() |
     (F.col('dt_interna') <= F.col('dt_evoluca')))
)
n2 = df_f.count()
print(f"F2 (consistência temporal)  : -{n1-n2:,}  → {n2:,}")

# F3: idade válida (0–120 anos, apenas quando preenchida)
df_f = df_f.filter(
    F.col('idade_anos').isNull() | F.col('idade_anos').between(0, 120)
)
n3 = df_f.count()
print(f"F3 (idade 0–120)            : -{n2-n3:,}  → {n3:,}")

# F4: deduplicação por NU_NOTIFIC
df_f = df_f.dropDuplicates(['nu_notific'])
n4 = df_f.count()
print(f"F4 (deduplicação notific.)  : -{n3-n4:,}  → {n4:,}")

pct_excluido = (n0 - n4) / n0 * 100
print(f"\nExclusão total : {n0-n4:,} ({pct_excluido:.2f}%)")
assert pct_excluido < 40, f"Exclusão excessiva: {pct_excluido:.1f}% > 40%"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Feature engineering

# COMMAND ----------

df_eng = df_f

# --- 9.1 Temporais ---
df_eng = df_eng \
    .withColumn('ano',  F.year('dt_sin_pri')) \
    .withColumn('mes',  F.month('dt_sin_pri')) \
    .withColumn('ano_mes_date', F.trunc('dt_sin_pri', 'month')) \
    .withColumn('ano_mes',
        F.concat(F.year('dt_sin_pri'), F.lit('-'),
                 F.lpad(F.month('dt_sin_pri'), 2, '0'))
    ) \
    .withColumn('tempo_sintoma_notificacao',
        F.datediff('dt_notific', 'dt_sin_pri')) \
    .withColumn('tempo_sintoma_internacao',
        F.when(F.col('dt_interna').isNotNull(),
               F.datediff('dt_interna', 'dt_sin_pri')).otherwise(None)
    ) \
    .withColumn('tempo_internacao',
        F.when(F.col('dt_interna').isNotNull() & F.col('dt_evoluca').isNotNull(),
               F.datediff('dt_evoluca', 'dt_interna')).otherwise(None)
    )

# --- 9.2 Demográficas ---
df_eng = df_eng.withColumn('faixa_etaria',
    F.when(F.col('idade_anos') < 1,   '0-1 ano')
    .when(F.col('idade_anos') < 5,    '1-4 anos')
    .when(F.col('idade_anos') < 10,   '5-9 anos')
    .when(F.col('idade_anos') < 18,   '10-17 anos')
    .when(F.col('idade_anos') < 30,   '18-29 anos')
    .when(F.col('idade_anos') < 40,   '30-39 anos')
    .when(F.col('idade_anos') < 50,   '40-49 anos')
    .when(F.col('idade_anos') < 60,   '50-59 anos')
    .when(F.col('idade_anos') < 70,   '60-69 anos')
    .when(F.col('idade_anos') >= 70,  '70+ anos')
    .otherwise('Desconhecido')          # inclui TP_IDADE != '3'
).withColumn('is_idoso',
    F.when(F.col('idade_anos') >= 60, True).otherwise(False)
)

# --- 9.3 Flags de desfecho ---
# is_obito_srag: EVOLUCAO='2' estritamente; denominador = evolucao_clean IS NOT NULL
# EVOLUCAO='3' já está NULL em evolucao_clean — ver Regras de Mortalidade
df_eng = df_eng \
    .withColumn('is_obito_srag',
        F.when(F.col('evolucao_clean') == '2', True).otherwise(False)) \
    .withColumn('is_cura',
        F.when(F.col('evolucao_clean') == '1', True).otherwise(False))

# --- 9.4 Flags de internação/UTI (CRÍTICO) ---
# is_internado : hospital_clean = '1'
# is_uti_valido: hospital_clean = '1' AND uti_clean = '1'
#   — UTI só é epidemiologicamente válida quando há confirmação de hospitalização.
#   — Usar is_uti_valido em TODOS os cálculos de taxa UTI.
#   — Denominador da taxa UTI = is_internado (não o total de casos).
df_eng = df_eng \
    .withColumn('is_internado',
        F.when(F.col('hospital_clean') == '1', True).otherwise(False)) \
    .withColumn('is_uti_valido',
        F.when(
            (F.col('hospital_clean') == '1') & (F.col('uti_clean') == '1'),
            True
        ).otherwise(False)
    )

# --- 9.5 Vacinação ---
# is_vacinado usa apenas VACINA (influenza/SRAG); VACINA_COV tratada separadamente
df_eng = df_eng.withColumn('is_vacinado',
    F.when(F.col('vacina_clean') == '1', True).otherwise(False))

# --- 9.6 Sintomas ---
for sint, col_out in [('febre','has_febre'), ('tosse','has_tosse'),
                      ('dispneia','has_dispneia')]:
    if sint in df_eng.columns:
        df_eng = df_eng.withColumn(col_out,
            F.when(F.col(sint) == '1', True).otherwise(False))

# --- 9.7 Qualidade ---
df_eng = df_eng \
    .withColumn('_data_valida',
        F.col('dt_sin_pri').isNotNull() & F.col('dt_notific').isNotNull() &
        (F.col('dt_sin_pri') <= F.col('dt_notific'))
    ) \
    .withColumn('_completude_score',
        (
            F.when(F.col('cs_sexo_clean').isNotNull(),  1).otherwise(0) +
            F.when(F.col('idade_anos').isNotNull(),     1).otherwise(0) +
            F.when(F.col('evolucao_clean').isNotNull(), 1).otherwise(0) +
            F.when(F.col('sg_uf').isNotNull(),          1).otherwise(0)
        ) / F.lit(4.0)
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Metadados de rastreabilidade

# COMMAND ----------

df_final = df_eng \
    .withColumn('_silver_processed_at', F.current_timestamp()) \
    .withColumn('_process_id', F.lit(PROCESS_ID))

silver_count = df_final.count()
print(f"Registros Silver : {silver_count:,}")
print(f"Colunas Silver   : {len(df_final.columns)}")
print(f"Redução registros: {(1 - silver_count/bronze_count)*100:.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Escrita da tabela Silver
# MAGIC
# MAGIC Modo `overwrite` com `overwriteSchema=true`: garante idempotência e permite
# MAGIC evolução de schema sem intervenção manual. Particionamento por `ano` + `mes`
# MAGIC alinhado com o padrão de acesso por janela temporal das queries Gold.

# COMMAND ----------

df_final.write \
    .mode('overwrite') \
    .partitionBy('ano', 'mes') \
    .option('overwriteSchema', 'true') \
    .saveAsTable(TABLE_SILVER)

print(f"Tabela gravada: {TABLE_SILVER}")

# COMMAND ----------

spark.sql(f"OPTIMIZE {TABLE_SILVER} ZORDER BY (dt_sin_pri, sg_uf)")
spark.sql(f"ANALYZE TABLE {TABLE_SILVER} COMPUTE STATISTICS")
print("OPTIMIZE e ANALYZE concluídos.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Validações pós-escrita

# COMMAND ----------

df_chk = spark.table(TABLE_SILVER)

# V1: perda de registros
pct_perda = (1 - silver_count / bronze_count) * 100
assert pct_perda < 60, f"Perda excessiva: {pct_perda:.1f}%"
print(f"V1 perda registros       : {pct_perda:.2f}%  ✓")

# V2: unicidade de NU_NOTIFIC
n_distinct = df_chk.select('nu_notific').distinct().count()
assert n_distinct == silver_count, "NU_NOTIFIC não é único!"
print(f"V2 unicidade nu_notific  : {n_distinct:,} distintos == {silver_count:,} total  ✓")

# V3: campos obrigatórios sem NULL
for campo in ['nu_notific', 'dt_sin_pri', 'dt_notific', 'ano', 'mes']:
    n = df_chk.filter(F.col(campo).isNull()).count()
    assert n == 0, f"Campo obrigatório {campo} contém {n} NULLs"
print(f"V3 campos obrigatórios   : 0 NULLs em todos  ✓")

# V4: consistência temporal
n_incons = df_chk.filter(F.col('dt_sin_pri') > F.col('dt_notific')).count()
assert n_incons == 0, f"{n_incons} registros com dt_sin_pri > dt_notific"
print(f"V4 consistência temporal : 0 inconsistências  ✓")

# V5 (CRÍTICO): is_uti_valido nunca True quando hospital_clean != '1'
# Esse count deve ser 0 — qualquer valor diferente indica bug na regra de UTI
n_uti_sem_hosp = df_chk.filter(
    (F.col('is_uti_valido') == True) &
    (F.col('hospital_clean') != '1')
).count()
assert n_uti_sem_hosp == 0, \
    f"BUG CRÍTICO: {n_uti_sem_hosp} registros com is_uti_valido=True e hospital_clean!='1'"
print(f"V5 is_uti_valido coerente: 0 casos UTI sem hospital  ✓")

# V6: % NULL em hospital_clean e uti_clean (auditoria de qualidade)
for col in ['hospital_clean', 'uti_clean']:
    if col in df_chk.columns:
        pct = df_chk.filter(F.col(col).isNull()).count() / silver_count * 100
        print(f"V6 NULL {col:<18}: {pct:.1f}%  (informativo)")

# V7: distribuição de evolucao e evolucao_clean (evidencia tratamento do '3' e '9')
print("\nDistribuição EVOLUCAO (original) vs evolucao_clean:")
df_chk.groupBy('evolucao', 'evolucao_clean').count().orderBy('evolucao').show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Métricas epidemiológicas — verificação

# COMMAND ----------

# MAGIC %md
# MAGIC ### 13.1 Taxa de Mortalidade SRAG estrita
# MAGIC Denominador: `evolucao_clean IN ('1','2')` — '3' já NULL em evolucao_clean.

# COMMAND ----------

spark.sql(f"""
    SELECT
        ano,
        SUM(CASE WHEN evolucao_clean IN ('1','2') THEN 1 ELSE 0 END) AS desfechos_conhecidos,
        SUM(CASE WHEN evolucao_clean = '2'        THEN 1 ELSE 0 END) AS obitos_srag,
        ROUND(
            SUM(CASE WHEN evolucao_clean = '2' THEN 1 ELSE 0 END) * 100.0 /
            NULLIF(SUM(CASE WHEN evolucao_clean IN ('1','2') THEN 1 ELSE 0 END), 0),
        2) AS taxa_mortalidade_pct
    FROM {TABLE_SILVER}
    GROUP BY ano ORDER BY ano
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 13.2 Taxa de Ocupação UTI
# MAGIC
# MAGIC Numerador : `hospital_clean = '1' AND uti_clean = '1'` (`is_uti_valido`)
# MAGIC Denominador: `hospital_clean = '1'` (`is_internado`)
# MAGIC
# MAGIC Equivalente SQL explícito para evitar uso acidental de `is_uti` (flag sem hospital):
# MAGIC ```sql
# MAGIC SUM(CASE WHEN hospital_clean='1' AND uti_clean='1' THEN 1 ELSE 0 END)
# MAGIC / NULLIF(SUM(CASE WHEN hospital_clean='1' THEN 1 ELSE 0 END), 0)
# MAGIC ```

# COMMAND ----------

spark.sql(f"""
    SELECT
        ano,
        SUM(CASE WHEN hospital_clean = '1'                         THEN 1 ELSE 0 END) AS internados,
        SUM(CASE WHEN hospital_clean = '1' AND uti_clean = '1'     THEN 1 ELSE 0 END) AS uti,
        ROUND(
            SUM(CASE WHEN hospital_clean = '1' AND uti_clean = '1' THEN 1 ELSE 0 END) * 100.0 /
            NULLIF(SUM(CASE WHEN hospital_clean = '1'              THEN 1 ELSE 0 END), 0),
        2) AS taxa_uti_pct
    FROM {TABLE_SILVER}
    GROUP BY ano ORDER BY ano
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 13.3 Taxa de Vacinação

# COMMAND ----------

spark.sql(f"""
    SELECT
        ano,
        SUM(CASE WHEN vacina_clean IS NOT NULL THEN 1 ELSE 0 END) AS com_info,
        SUM(CASE WHEN vacina_clean = '1'       THEN 1 ELSE 0 END) AS vacinados,
        ROUND(
            SUM(CASE WHEN vacina_clean = '1' THEN 1 ELSE 0 END) * 100.0 /
            NULLIF(SUM(CASE WHEN vacina_clean IS NOT NULL THEN 1 ELSE 0 END), 0),
        2) AS taxa_vacinacao_pct
    FROM {TABLE_SILVER}
    GROUP BY ano ORDER BY ano
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 13.4 Crescimento mensal — últimos 12 meses

# COMMAND ----------

spark.sql(f"""
    WITH base AS (
        SELECT ano_mes, COUNT(*) AS casos
        FROM {TABLE_SILVER}
        WHERE ano_mes_date >= add_months(
            (SELECT MAX(ano_mes_date) FROM {TABLE_SILVER}), -11
        )
        GROUP BY ano_mes
    )
    SELECT
        ano_mes,
        casos,
        LAG(casos) OVER (ORDER BY ano_mes) AS casos_anterior,
        ROUND(
            (casos - LAG(casos) OVER (ORDER BY ano_mes)) * 100.0 /
            NULLIF(LAG(casos) OVER (ORDER BY ano_mes), 0),
        2) AS crescimento_pct
    FROM base
    ORDER BY ano_mes
""").show(12)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Resumo final

# COMMAND ----------

print("=" * 70)
print("SILVER CONCLUÍDA")
print("=" * 70)
print(f"  Tabela     : {TABLE_SILVER}")
print(f"  Process ID : {PROCESS_ID}")
print(f"  Bronze     : {bronze_count:,} registros")
print(f"  Silver     : {silver_count:,} registros  ({pct_perda:.2f}% excluídos)")
print(f"  Colunas    : {len(df_final.columns)}")
print(f"  Partições  : ano + mes")
print(f"  Z-order    : dt_sin_pri, sg_uf")
print()
print("  Flags epidemiológicas:")
print("    is_obito_srag  : evolucao_clean = '2' (SRAG estrita)")
print("    is_cura        : evolucao_clean = '1'")
print("    is_internado   : hospital_clean = '1'")
print("    is_uti_valido  : hospital_clean = '1' AND uti_clean = '1'")
print("    is_vacinado    : vacina_clean = '1'")
print()
print("  Próximo: Gold Layer")
print("=" * 70)