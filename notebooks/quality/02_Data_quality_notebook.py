# Databricks notebook source
# MAGIC %md
# MAGIC # Validação de Qualidade — Camada Bronze (SRAG)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 1. Objetivo
# MAGIC
# MAGIC Executar checks automatizados de qualidade sobre `bronze_srag_raw` e persistir os
# MAGIC resultados em tabelas de auditoria. O notebook **não modifica dados da Bronze** —
# MAGIC apenas diagnostica, classifica problemas e gera insumos para a camada Silver.
# MAGIC
# MAGIC Os checks cobrem os campos considerados críticos para as métricas epidemiológicas
# MAGIC do projeto (mortalidade, taxa UTI, cobertura vacinal, distribuição temporal). Campos
# MAGIC fora desse escopo não são validados aqui.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 2. Entrada / Saída
# MAGIC
# MAGIC | Papel | Tabela |
# MAGIC |---|---|
# MAGIC | Fonte | `dbx_srag_lab.data_original.bronze_srag_raw` |
# MAGIC | Checks detalhados | `dbx_srag_lab.data_original.quality_checks` |
# MAGIC | Resumo por execução | `dbx_srag_lab.data_original.quality_summary` |
# MAGIC
# MAGIC Cada execução gera um `validation_id` único. Todas as tabelas de saída usam
# MAGIC esse campo como chave de rastreabilidade. A escrita em ambas as tabelas é `append`.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 3. Checks implementados
# MAGIC
# MAGIC | Tipo | Escopo |
# MAGIC |---|---|
# MAGIC | Completude | 20 campos críticos — % de NULL ou vazio. Código `9` ("Ignorado") não é NULL e é analisado separadamente. |
# MAGIC | Domínio | 9 campos categóricos — valores fora do conjunto permitido pelo dicionário oficial |
# MAGIC | Formato de data | 5 campos — aceita `dd/MM/yyyy` e `yyyy-MM-dd` via `coalesce(to_date(...))` |
# MAGIC | Unicidade | `NU_NOTIFIC` — registros duplicados |
# MAGIC | Consistência temporal | 4 pares de datas — violação de ordem cronológica |
# MAGIC | Código "9" (Ignorado) | 8 campos — quantificação de registros com valor "Ignorado" |
# MAGIC
# MAGIC O código `9` no SRAG significa "Ignorado" — o campo foi preenchido mas com incerteza.
# MAGIC Não deve ser tratado como dado ausente. Os checks de completude contam apenas NULL/vazio;
# MAGIC a análise de código `9` é feita em seção própria.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 4. Severidades e thresholds
# MAGIC
# MAGIC ### Completude (campos de negócio)
# MAGIC
# MAGIC | Status | Critério |
# MAGIC |---|---|
# MAGIC | OK | null_pct = 0 |
# MAGIC | WARNING | 0 < null_pct < 5% |
# MAGIC | HIGH | 5% ≤ null_pct < 20% |
# MAGIC | CRITICAL | null_pct ≥ 20% |
# MAGIC
# MAGIC ### Unicidade (NU_NOTIFIC)
# MAGIC
# MAGIC Duplicatas nesse volume são esperadas em repescagens de dados pelo DATASUS.
# MAGIC A classificação é proporcional ao volume de duplicação:
# MAGIC
# MAGIC | Status | Critério |
# MAGIC |---|---|
# MAGIC | OK | duplicate_pct = 0 |
# MAGIC | WARNING | 0 < duplicate_pct < 0,1% |
# MAGIC | HIGH | 0,1% ≤ duplicate_pct < 1% |
# MAGIC | CRITICAL | duplicate_pct ≥ 1% |
# MAGIC
# MAGIC ### Domínio e consistência
# MAGIC
# MAGIC | Status | Critério |
# MAGIC |---|---|
# MAGIC | OK | invalid_pct = 0 / inconsistent_pct = 0 |
# MAGIC | WARNING | 0 < invalid_pct < 0,01% |
# MAGIC | HIGH | 0,01% ≤ invalid_pct < 0,1% |
# MAGIC | CRITICAL | invalid_pct ≥ 0,1% |
# MAGIC
# MAGIC ### Checks não-bloqueantes (`NON_BLOCKING_CHECKS`)
# MAGIC
# MAGIC Alguns checks são registrados com seu status real (incluindo CRITICAL) mas
# MAGIC **não bloqueiam a Silver** porque o problema já é tratado downstream:
# MAGIC
# MAGIC | Campo | Tipo | Motivo |
# MAGIC |---|---|---|
# MAGIC | `VACINA`, `VACINA_COV` | completeness | Ausência legítima em dados recentes; Silver usa `vacina_clean` com denominador próprio |
# MAGIC | `DT_INTERNA`, `DT_ENTUTI`, `DT_EVOLUCA` | date_format | Silver converte datas inválidas → NULL antes dos filtros |
# MAGIC | `DT_SIN_PRI vs DT_INTERNA`, `DT_INTERNA vs DT_ENTUTI`, `DT_INTERNA vs DT_EVOLUCA` | consistency | Silver filtra inconsistências temporais no F2 |
# MAGIC
# MAGIC O `final_status = FAIL` é gerado apenas por checks **bloqueantes** com status CRITICAL.
# MAGIC O `quality_score` agrega todos os checks (incluindo não-bloqueantes) para refletir
# MAGIC a qualidade real dos dados.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 5. Quality score
# MAGIC
# MAGIC O score agrega todos os checks de uma execução em um único número entre 0 e 1:
# MAGIC
# MAGIC ```python
# MAGIC quality_score = (checks_ok + 0.5 * checks_warning + 0.25 * checks_high) / total_checks
# MAGIC ```
# MAGIC
# MAGIC Checks CRITICAL contribuem zero. Cada nível de severidade recebe peso decrescente:
# MAGIC OK=1.0, WARNING=0.5, HIGH=0.25, CRITICAL=0. A fórmula penaliza proporcionalmente
# MAGIC a gravidade sem equiparar checks HIGH a CRITICAL.
# MAGIC
# MAGIC | Score | Interpretação |
# MAGIC |---|---|
# MAGIC | ≥ 0.85 | PASS — qualidade aceitável para Silver |
# MAGIC | 0.70 – 0.85 | WARN — revisar campos HIGH antes de processar Silver |
# MAGIC | < 0.70 | FAIL — bloqueio recomendado; investigar CRITICAL |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 6. Dependências e próximos passos
# MAGIC
# MAGIC - **Dependência**: `bronze_srag_raw` deve existir e conter as colunas críticas listadas
# MAGIC   no notebook `01_Bronze_Ingestion_SRAG.py`.
# MAGIC - **EDA**: `03_eda_srag_exploraty_analysis.py` consome o Bronze diretamente.
# MAGIC - **Silver**: `04_Silver_Transformation.py` deve ler `quality_checks` para decidir
# MAGIC   filtros, deduplicação de `NU_NOTIFIC` e tratamento de datas em formato misto.
# MAGIC   As regras recomendadas estão na seção "Decisões para a Silver" ao final deste notebook.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 7. Troubleshooting
# MAGIC
# MAGIC | Sintoma | Causa provável | Ação |
# MAGIC |---|---|---|
# MAGIC | `Table not found: bronze_srag_raw` | Bronze não executada ou catálogo incorreto | Rodar `01_Bronze_Ingestion_SRAG.py`; confirmar `CATALOG` |
# MAGIC | `quality_checks` com linhas duplicadas por check | Notebook executado duas vezes sem filtro de `validation_id` | Sempre exibir com `WHERE validation_id = CURRENT_VALIDATION_ID` |
# MAGIC | Schema mismatch ao salvar `quality_summary` | Nova coluna adicionada (ex.: `quality_score`) | `mergeSchema=true` está ativo; drop manual se necessário |
# MAGIC | Checks de data com 100% inválidos | Formato de data novo nos CSVs | Adicionar formato ao `coalesce` em `check_date_format` |
# MAGIC | `AnalysisException` em `check_consistency_dates` | Campo ausente em algum ano | Verificar `if field in df.columns` antes da chamada |
# MAGIC | Investigar versões anteriores | Mudança de qualidade entre runs | `DESCRIBE HISTORY dbx_srag_lab.data_original.quality_summary` |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuração inicial

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType
from datetime import datetime
from uuid import uuid4
import json

print("-" * 80)
print("VALIDAÇÃO DE QUALIDADE — BRONZE SRAG")
print(f"Início   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Spark    : {spark.version}")
print("-" * 80)

# COMMAND ----------

# Schema explícito para todas as tabelas de quality checks.
# Elimina CANNOT_DETERMINE_TYPE: sem schema, o Spark Connect tenta inferir tipos
# a partir das listas de dicts — e falha quando várias colunas ficam 100% None
# (NullType não é resolvível em Serverless/Connect). Definir uma vez, usar em todas
# as chamadas createDataFrame que recebem completeness/domain/date/consistency/all_checks.
QUALITY_CHECK_SCHEMA = StructType([
    StructField("validation_id",      StringType(), True),
    StructField("check_type",         StringType(), True),
    StructField("field",              StringType(), True),
    StructField("total",              LongType(),   True),
    StructField("null_count",         LongType(),   True),
    StructField("null_pct",           DoubleType(), True),
    StructField("invalid_count",      LongType(),   True),
    StructField("invalid_pct",        DoubleType(), True),
    StructField("inconsistent_count", LongType(),   True),
    StructField("inconsistent_pct",   DoubleType(), True),
    StructField("duplicate_count",    LongType(),   True),
    StructField("duplicate_pct",      DoubleType(), True),
    StructField("valid_values",       StringType(), True),
    StructField("accepted_formats",   StringType(), True),
    StructField("status",             StringType(), True),
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parâmetros

# COMMAND ----------

CATALOG = "dbx_srag_lab"
SCHEMA  = "data_original"

TABLE_BRONZE          = f"{CATALOG}.{SCHEMA}.bronze_srag_raw"
TABLE_QUALITY_CHECKS  = f"{CATALOG}.{SCHEMA}.quality_checks"
TABLE_QUALITY_SUMMARY = f"{CATALOG}.{SCHEMA}.quality_summary"

# validation_id: chave de rastreabilidade de cada execução.
# Inclui microsegundos + sufixo UUID para evitar colisões em execuções paralelas.
CURRENT_VALIDATION_ID = f"{datetime.now():%Y%m%d_%H%M%S_%f}_{uuid4().hex[:8]}"

# Thresholds de qualidade
SCORE_WARN_THRESHOLD = 0.85
SCORE_FAIL_THRESHOLD = 0.70

# Checks que NÃO bloqueiam o pipeline mesmo com status CRITICAL.
# Critério de inclusão: a Silver já corrige o problema — datas inválidas
# viram NULL, inconsistências temporais são filtradas, e VACINA é legitimamente
# ausente (denominadores usam vacina_clean com "tem informação válida").
# Os checks continuam sendo executados e gravados com seu status real —
# apenas são excluídos do gating que bloqueia a Silver.
NON_BLOCKING_CHECKS = {
    # (field, check_type)
    ("VACINA",                    "completeness"),  # ausência legítima em dados recentes
    ("VACINA_COV",                "completeness"),  # idem
    ("DT_INTERNA",                "date_format"),   # Silver converte inválido → NULL
    ("DT_ENTUTI",                 "date_format"),   # idem
    ("DT_EVOLUCA",                "date_format"),   # idem
    ("DT_SIN_PRI vs DT_INTERNA",  "consistency"),   # Silver filtra inconsistências
    ("DT_INTERNA vs DT_ENTUTI",   "consistency"),   # idem
    ("DT_INTERNA vs DT_EVOLUCA",  "consistency"),   # idem
}

print(f"Fonte           : {TABLE_BRONZE}")
print(f"Tabela checks   : {TABLE_QUALITY_CHECKS}")
print(f"Tabela resumo   : {TABLE_QUALITY_SUMMARY}")
print(f"Validation ID   : {CURRENT_VALIDATION_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Carregamento da Bronze

# COMMAND ----------

df_bronze = spark.table(TABLE_BRONZE)

total_rows = df_bronze.count()
total_cols = len(df_bronze.columns)

print(f"Registros : {total_rows:,}")
print(f"Colunas   : {total_cols}")
print()
df_bronze.groupBy("ANO_DADOS").count().orderBy("ANO_DADOS").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Campos críticos
# MAGIC
# MAGIC Os campos abaixo foram selecionados por serem insumos diretos das métricas
# MAGIC epidemiológicas prioritárias do projeto. Campos fora desse conjunto podem ter
# MAGIC problemas de qualidade sem impacto no pipeline principal.

# COMMAND ----------

CRITICAL_FIELDS = {
    'identificacao': {
        'fields': ['NU_NOTIFIC'],
        'description': 'Identificação única do caso',
    },
    'temporal': {
        'fields': ['DT_NOTIFIC', 'DT_SIN_PRI', 'SEM_PRI'],
        'description': 'Datas essenciais para análise temporal',
        'date_format': ['dd/MM/yyyy', 'yyyy-MM-dd'],
    },
    'localizacao': {
        'fields': ['SG_UF', 'CO_MUN_RES'],
        'description': 'Localização do caso',
    },
    'demografia': {
        'fields': ['CS_SEXO', 'NU_IDADE_N', 'TP_IDADE'],
        'description': 'Dados demográficos básicos',
    },
    'sintomas': {
        'fields': ['FEBRE', 'TOSSE', 'DISPNEIA', 'SATURACAO'],
        'description': 'Sintomas clínicos principais',
        'valid_values': ['1', '2', '9'],
    },
    'internacao': {
        'fields': ['HOSPITAL', 'DT_INTERNA', 'UTI'],
        'description': 'Dados de internação — alimentam taxa UTI',
    },
    'desfecho': {
        'fields': ['EVOLUCAO', 'DT_EVOLUCA', 'CLASSI_FIN'],
        'description': 'Desfecho e classificação etiológica — alimenta mortalidade e breakdown COVID/Influenza',
        'valid_values': ['1', '2', '3', '9'],  # para EVOLUCAO; CLASSI_FIN validado separado
    },
    'vacinacao': {
        'fields': ['VACINA', 'VACINA_COV'],
        'description': 'Histórico vacinal — cobertura esperada incompleta em dados recentes',
    },
}

all_critical_fields      = [f for cat in CRITICAL_FIELDS.values() for f in cat['fields']]
existing_critical_fields = [f for f in all_critical_fields if f in df_bronze.columns]

print(f"Campos críticos definidos    : {len(all_critical_fields)}")
print(f"Campos presentes na Bronze   : {len(existing_critical_fields)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Funções de validação

# COMMAND ----------

def _classify_completeness(null_pct: float) -> str:
    if null_pct == 0:       return 'OK'
    elif null_pct < 5:      return 'WARNING'
    elif null_pct < 20:     return 'HIGH'
    return 'CRITICAL'


def _classify_uniqueness(dup_pct: float) -> str:
    """
    Thresholds documentados na seção 4:
      OK=0 | WARNING<0,1% | HIGH<1% | CRITICAL>=1%
    Duplicações baixas (<0,1%) são esperadas em repescagens do DATASUS.
    """
    if dup_pct == 0:            return 'OK'
    elif dup_pct < 0.001:       return 'WARNING'
    elif dup_pct < 0.01:        return 'HIGH'
    return 'CRITICAL'


def _classify_domain(invalid_pct: float) -> str:
    """
    Thresholds graduados para domínio/consistência — campos SRAG frequentemente
    contêm valores não documentados em pequena proporção.
      OK=0 | WARNING<0,01% | HIGH<0,1% | CRITICAL>=0,1%
    """
    if invalid_pct == 0:            return 'OK'
    elif invalid_pct < 0.01:        return 'WARNING'
    elif invalid_pct < 0.1:         return 'HIGH'
    return 'CRITICAL'


def _base_record(validation_id: str, check_type: str, field: str, total: int) -> dict:
    """Esqueleto com todas as colunas da tabela quality_checks preenchidas com None."""
    return {
        'validation_id':      validation_id,
        'check_type':         check_type,
        'field':              field,
        'total':              total,
        'null_count':         None,
        'null_pct':           None,
        'invalid_count':      None,
        'invalid_pct':        None,
        'inconsistent_count': None,
        'inconsistent_pct':   None,
        'duplicate_count':    None,
        'duplicate_pct':      None,
        'valid_values':       None,
        'accepted_formats':   None,
        'status':             None,
    }


def check_completeness(df, field: str, validation_id: str, total_rows: int) -> dict:
    """
    Completude: contagem de NULL e vazio.
    Código '9' (Ignorado) não é considerado ausente — analisado separadamente.
    total_rows é passado externamente para evitar df.count() redundante.
    """
    null_count = df.filter(F.col(field).isNull() | (F.col(field) == '')).count()
    null_pct   = round(null_count / total_rows * 100, 2) if total_rows > 0 else 0.0

    rec = _base_record(validation_id, 'completeness', field, total_rows)
    rec.update({'null_count': null_count, 'null_pct': null_pct,
                'status': _classify_completeness(null_pct)})
    return rec


def check_domain(df, field: str, valid_values: list, validation_id: str,
                 total_rows: int) -> dict:
    """
    Domínio: valores fora do conjunto permitido pelo dicionário oficial.
    Classifica com valor bruto; armazena com 4 casas para preservar granularidade
    em volumes grandes onde thresholds < 0.01% seriam zerados com 2 casas.
    """
    total_notnull = df.filter(F.col(field).isNotNull()).count()
    invalid_count = df.filter(
        F.col(field).isNotNull() & ~F.col(field).isin(valid_values)
    ).count()
    invalid_pct_raw = (invalid_count / total_notnull * 100) if total_notnull > 0 else 0.0

    rec = _base_record(validation_id, 'domain', field, total_rows)
    rec.update({'invalid_count': invalid_count,
                'invalid_pct':   round(invalid_pct_raw, 4),
                'valid_values':  str(valid_values),
                'status':        _classify_domain(invalid_pct_raw)})
    return rec


def check_date_format(df, field: str, validation_id: str, total_rows: int) -> dict:
    """
    Formato de data: aceita dd/MM/yyyy e yyyy-MM-dd.
    Usa regex gate antes de to_date para evitar DateTimeException no Photon com
    ANSI mode ativo (padrão em Serverless): to_date com formato incompatível lança
    exceção antes do coalesce poder agir. O regex garante que to_date só executa
    sobre strings que já correspondem ao padrão esperado.
    Mesmo padrão usado em _parse_date na camada Silver.
    Usa select ao invés de withColumn para evitar coluna temporária no dataset.
    Severidade via _classify_domain() — consistente com domínio e consistência temporal.
    Classifica com valor bruto; armazena com 4 casas decimais.
    """
    total_notnull = df.filter(F.col(field).isNotNull()).count()

    invalid_count = (
        df.select(
            F.col(field),
            F.coalesce(
                # Regex gate: to_date só é chamado quando o formato já foi confirmado.
                # Sem isso, Photon/ANSI lança DateTimeException ao tentar parsear
                # 'yyyy-MM-dd' com o formato 'dd/MM/yyyy' (e vice-versa).
                F.when(
                    F.col(field).rlike(r'^\d{2}/\d{2}/\d{4}$'),
                    F.to_date(F.col(field), 'dd/MM/yyyy')
                ),
                F.when(
                    F.col(field).rlike(r'^\d{4}-\d{2}-\d{2}$'),
                    F.to_date(F.col(field), 'yyyy-MM-dd')
                ),
            ).alias('_parsed')
        )
        .filter(F.col(field).isNotNull() & F.col('_parsed').isNull())
        .count()
    )
    invalid_pct_raw = (invalid_count / total_notnull * 100) if total_notnull > 0 else 0.0

    rec = _base_record(validation_id, 'date_format', field, total_rows)
    rec.update({'invalid_count':    invalid_count,
                'invalid_pct':      round(invalid_pct_raw, 4),
                'accepted_formats': 'dd/MM/yyyy|yyyy-MM-dd',
                'valid_values':     'dd/MM/yyyy|yyyy-MM-dd',
                'status':           _classify_domain(invalid_pct_raw)})
    return rec


def check_uniqueness(df, field: str, validation_id: str, total_rows: int) -> dict:
    """
    Unicidade: total_rows passado externamente para evitar df.count() redundante.
    Thresholds: OK=0 | WARNING<0,1% | HIGH<1% | CRITICAL>=1%
    """
    distinct = df.select(field).distinct().count()
    dup      = total_rows - distinct
    dup_pct  = round(dup / total_rows * 100, 4) if total_rows > 0 else 0.0

    rec = _base_record(validation_id, 'uniqueness', field, total_rows)
    rec.update({'duplicate_count': dup, 'duplicate_pct': dup_pct,
                'status': _classify_uniqueness(dup_pct / 100)})
    return rec


def safe_parse_date(col):
    """
    Parse seguro de data: regex gate antes do to_date.
    Photon com ANSI mode (padrão em Serverless) lança DateTimeException quando
    to_date recebe um valor que não casa o formato — antes do coalesce poder agir.
    O regex confirma o formato ANTES de chamar to_date, eliminando a exceção.
    Retorna NULL para qualquer string que não case nenhum dos dois padrões aceitos.
    Mesmo padrão usado em _parse_date (Silver) e check_date_format.
    """
    return F.coalesce(
        F.when(col.rlike(r'^\d{2}/\d{2}/\d{4}$'), F.to_date(col, 'dd/MM/yyyy')),
        F.when(col.rlike(r'^\d{4}-\d{2}-\d{2}$'), F.to_date(col, 'yyyy-MM-dd')),
    )


def check_consistency_dates(df, field1: str, field2: str, validation_id: str,
                             total_rows: int) -> dict:
    """
    Consistência temporal: field1 deve ser anterior ou igual a field2.
    Usa safe_parse_date() para ambos os campos — evita DateTimeException em
    Photon/ANSI mode quando coalesce(to_date(...)) recebe formato incompatível.
    Severidade via _classify_domain() — graduada pelos mesmos thresholds de domínio.
    Classifica com valor bruto; armazena com 4 casas decimais.
    """
    df_p = df.select(
        safe_parse_date(F.col(field1)).alias('_d1'),
        safe_parse_date(F.col(field2)).alias('_d2'),
    )

    total = df_p.filter(F.col('_d1').isNotNull() & F.col('_d2').isNotNull()).count()
    inc   = df_p.filter(
        F.col('_d1').isNotNull() & F.col('_d2').isNotNull() & (F.col('_d1') > F.col('_d2'))
    ).count()
    inc_pct_raw = (inc / total * 100) if total > 0 else 0.0

    rec = _base_record(validation_id, 'consistency', f'{field1} vs {field2}', total_rows)
    rec.update({'inconsistent_count': inc,
                'inconsistent_pct':   round(inc_pct_raw, 4),
                'valid_values':       f'{field1} <= {field2}',
                'status':             _classify_domain(inc_pct_raw)})
    return rec


print("Funções de validação prontas.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Execução dos checks

# COMMAND ----------

# MAGIC %md
# MAGIC ### Completude

# COMMAND ----------

completeness_results = [
    check_completeness(df_bronze, f, CURRENT_VALIDATION_ID, total_rows)
    for f in existing_critical_fields
]

df_completeness = spark.createDataFrame(completeness_results, schema=QUALITY_CHECK_SCHEMA)

print("Resumo de completude:")
df_completeness.groupBy('status').count().orderBy('status').show()

print("Top 10 campos com mais ausências:")
display(
    df_completeness.orderBy(F.desc('null_pct')).limit(10)
    .select('field', 'null_pct', 'null_count', 'status')
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Domínio
# MAGIC
# MAGIC Domínios baseados no dicionário oficial SRAG 2019–2025.
# MAGIC
# MAGIC - `CS_SEXO`: padrão oficial `1/2/9` (numérico). Alguns exports do DATASUS apresentam
# MAGIC   codificação alfanumérica (`M/F/I`). Para robustez, o notebook aceita ambos os conjuntos
# MAGIC   sem heurística de detecção — reduz falsos positivos sem impacto na governança.
# MAGIC - `EVOLUCAO`: domínio oficial `1=Cura`, `2=Óbito por SRAG`, `3=Óbito por outras causas`,
# MAGIC   `9=Ignorado`. O valor `3` é documentado e deve ser tratado como categoria válida.
# MAGIC   A decisão de incluí-lo ou não em métricas de mortalidade é de negócio e pertence à Silver.
# MAGIC - `CLASSI_FIN`: domínio oficial `1=Influenza`, `2=Outro vírus respiratório`, `3=Outro agente`,
# MAGIC   `4=Não especificado`, `5=COVID-19`. Validado neste notebook junto aos demais campos categóricos.

# COMMAND ----------

# CS_SEXO aceita numérico (padrão oficial) e alfanumérico (encoding alternativo presente
# em alguns exports). Aceitar ambos elimina falsos positivos sem necessidade de heurística.
DOMAIN_CHECKS = [
    ('CS_SEXO',   ['1', '2', '9', 'M', 'F', 'I']),
    ('FEBRE',     ['1', '2', '9']),
    ('TOSSE',     ['1', '2', '9']),
    ('DISPNEIA',  ['1', '2', '9']),
    ('SATURACAO', ['1', '2', '9']),
    ('HOSPITAL',  ['1', '2', '9']),
    ('UTI',       ['1', '2', '9']),
    ('EVOLUCAO',  ['1', '2', '3', '9']),   # 3 = óbito por outras causas (dicionário oficial)
    ('VACINA',    ['1', '2', '9']),
    ('CLASSI_FIN', ['1', '2', '3', '4', '5', '9']),  # classificação etiológica SIVEP-Gripe
    ('VACINA_COV', ['1', '2', '9']),                   # vacinação COVID-19
]

domain_results = [
    check_domain(df_bronze, f, vals, CURRENT_VALIDATION_ID, total_rows)
    for f, vals in DOMAIN_CHECKS
    if f in df_bronze.columns
]

df_domain       = spark.createDataFrame(domain_results,       schema=QUALITY_CHECK_SCHEMA)

print("Resumo de domínio:")
df_domain.groupBy('status').count().orderBy('status').show()
display(df_domain.select('field', 'invalid_pct', 'valid_values', 'status'))

# Exibir valores únicos para campos com domínio violado (status HIGH ou CRITICAL)
campos_invalidos = [r['field'] for r in domain_results if r['status'] in ('HIGH', 'CRITICAL')]
for field in campos_invalidos:
    esperado = next(vals for f, vals in DOMAIN_CHECKS if f == field)
    print(f"\nDomínio violado — {field} (esperado: {esperado}):")
    display(df_bronze.select(field).distinct().limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Formato de datas

# COMMAND ----------

DATE_FIELDS = ['DT_NOTIFIC', 'DT_SIN_PRI', 'DT_INTERNA', 'DT_ENTUTI', 'DT_EVOLUCA']

date_results = [
    check_date_format(df_bronze, f, CURRENT_VALIDATION_ID, total_rows)
    for f in DATE_FIELDS
    if f in df_bronze.columns
]

df_dates        = spark.createDataFrame(date_results,         schema=QUALITY_CHECK_SCHEMA)

print("Checks de formato de data:")
display(df_dates.select('field', 'invalid_pct', 'invalid_count', 'accepted_formats', 'status'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Unicidade — NU_NOTIFIC

# COMMAND ----------

uniqueness_results = []
if 'NU_NOTIFIC' in df_bronze.columns:
    r = check_uniqueness(df_bronze, 'NU_NOTIFIC', CURRENT_VALIDATION_ID, total_rows)
    uniqueness_results.append(r)
    print(f"NU_NOTIFIC — total: {r['total']:,} | distintos: {r['total'] - r['duplicate_count']:,} "
          f"| duplicatas: {r['duplicate_count']:,} ({r['duplicate_pct']:.4f}%) | status: {r['status']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Consistência temporal

# COMMAND ----------

CONSISTENCY_CHECKS = [
    ('DT_SIN_PRI', 'DT_NOTIFIC'),
    ('DT_SIN_PRI', 'DT_INTERNA'),
    ('DT_INTERNA', 'DT_ENTUTI'),
    ('DT_INTERNA', 'DT_EVOLUCA'),
]

consistency_results = [
    check_consistency_dates(df_bronze, f1, f2, CURRENT_VALIDATION_ID, total_rows)
    for f1, f2 in CONSISTENCY_CHECKS
    if f1 in df_bronze.columns and f2 in df_bronze.columns
]

df_consistency  = spark.createDataFrame(consistency_results,  schema=QUALITY_CHECK_SCHEMA)

print("Checks de consistência temporal:")
display(df_consistency.select('field', 'inconsistent_pct', 'inconsistent_count', 'status'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Análise de código "9" (Ignorado)
# MAGIC
# MAGIC Código `9` no SRAG significa "Ignorado" — valor informado com incerteza.
# MAGIC Não deve ser tratado como NULL. Os percentuais abaixo orientam a Silver sobre
# MAGIC quais campos exigem filtragem quando a análise exige completude real.

# COMMAND ----------

CODE9_FIELDS = ['CS_RACA', 'FEBRE', 'TOSSE', 'DISPNEIA',
                'HOSPITAL', 'UTI', 'EVOLUCAO', 'VACINA', 'VACINA_COV']

code9_results = []
for field in CODE9_FIELDS:
    if field in df_bronze.columns:
        cnt = df_bronze.filter(F.col(field) == '9').count()
        pct = round(cnt / total_rows * 100, 2)
        code9_results.append({
            'field':       field,
            'code9_count': cnt,
            'code9_pct':   pct,
            'severity':    'HIGH' if pct > 20 else 'MEDIUM' if pct > 10 else 'LOW',
        })

df_code9 = spark.createDataFrame(code9_results)
display(df_code9.orderBy(F.desc('code9_pct')).select('field', 'code9_pct', 'code9_count', 'severity'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Qualidade por ano

# COMMAND ----------

agg_exprs = [
    F.count("*").alias("total_registros"),
    F.round(
        F.sum(F.when(F.col("DT_SIN_PRI").isNull() | (F.col("DT_SIN_PRI") == ""), 1).otherwise(0))
        / F.count("*") * 100, 2
    ).alias("DT_SIN_PRI_null_pct"),
    F.round(
        F.sum(F.when(F.col("EVOLUCAO").isNull() | (F.col("EVOLUCAO") == ""), 1).otherwise(0))
        / F.count("*") * 100, 2
    ).alias("EVOLUCAO_null_pct"),
    F.round(
        F.sum(F.when(F.col("UTI").isNull() | (F.col("UTI") == ""), 1).otherwise(0))
        / F.count("*") * 100, 2
    ).alias("UTI_null_pct"),
    F.round(
        F.sum(F.when(F.col("VACINA").isNull() | (F.col("VACINA") == ""), 1).otherwise(0))
        / F.count("*") * 100, 2
    ).alias("VACINA_null_pct"),
]

if "CLASSI_FIN" in df_bronze.columns:
    agg_exprs.append(
        F.round(
            F.sum(F.when(F.col("CLASSI_FIN").isNull() | (F.col("CLASSI_FIN") == ""), 1).otherwise(0))
            / F.count("*") * 100, 2
        ).alias("CLASSI_FIN_null_pct")
    )

if "VACINA_COV" in df_bronze.columns:
    agg_exprs.append(
        F.round(
            F.sum(F.when(F.col("VACINA_COV").isNull() | (F.col("VACINA_COV") == ""), 1).otherwise(0))
            / F.count("*") * 100, 2
        ).alias("VACINA_COV_null_pct")
    )

df_quality_year = (
    df_bronze.groupBy("ANO_DADOS").agg(*agg_exprs).orderBy("ANO_DADOS")
)

print("Qualidade por ano:")
display(df_quality_year)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Consolidação e persistência

# COMMAND ----------

all_checks = (
    completeness_results
    + domain_results
    + date_results
    + uniqueness_results
    + consistency_results
)

df_all_checks   = spark.createDataFrame(all_checks,           schema=QUALITY_CHECK_SCHEMA)

print(f"Total de checks consolidados: {len(all_checks)}")

# COMMAND ----------

(
    df_all_checks.write
    .mode("append")
    .option("mergeSchema", "true")
    .saveAsTable(TABLE_QUALITY_CHECKS)
)
print(f"Checks gravados em: {TABLE_QUALITY_CHECKS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quality score e status final
# MAGIC
# MAGIC Fórmula:
# MAGIC ```
# MAGIC quality_score = (checks_ok + 0.5 * checks_warning + 0.25 * checks_high) / total_checks
# MAGIC ```
# MAGIC Checks CRITICAL contribuem zero. Score ≥ 0.85 → PASS | 0.70–0.85 → WARN | < 0.70 → FAIL

# COMMAND ----------

# Contagens brutas — refletem a realidade dos dados, independente de gating.
checks_ok       = sum(1 for c in all_checks if c['status'] == 'OK')
checks_warning  = sum(1 for c in all_checks if c['status'] == 'WARNING')
checks_high     = sum(1 for c in all_checks if c['status'] == 'HIGH')
checks_critical = sum(1 for c in all_checks if c['status'] == 'CRITICAL')
total_checks    = len(all_checks)

# quality_score inclui todos os checks (incluindo não-bloqueantes) para refletir
# a qualidade real dos dados — não apenas o que bloqueia a Silver.
quality_score = round(
    (checks_ok + 0.5 * checks_warning + 0.25 * checks_high) / total_checks, 4
) if total_checks > 0 else 0.0

# Gating: separa checks bloqueantes de não-bloqueantes.
# Checks em NON_BLOCKING_CHECKS são gravados com status real mas não bloqueam
# a Silver — o problema já é tratado pela Silver ou é estrutural dos dados.
def _is_blocking(c: dict) -> bool:
    return (c['field'], c['check_type']) not in NON_BLOCKING_CHECKS

blocking_checks          = [c for c in all_checks if _is_blocking(c)]
non_blocking_checks      = [c for c in all_checks if not _is_blocking(c)]
gating_checks_critical   = sum(1 for c in blocking_checks if c['status'] == 'CRITICAL')
non_blocking_critical    = sum(1 for c in non_blocking_checks if c['status'] == 'CRITICAL')

if gating_checks_critical > 0:
    final_status = 'FAIL'
    fail_reason  = (
        f"{gating_checks_critical} check(s) CRITICAL bloqueante(s): "
        + ", ".join(c['field'] for c in blocking_checks if c['status'] == 'CRITICAL')
    )
elif quality_score < SCORE_WARN_THRESHOLD:
    final_status = 'WARN'
    fail_reason  = f"quality_score={quality_score} abaixo do threshold WARN ({SCORE_WARN_THRESHOLD})"
else:
    final_status = 'PASS'
    fail_reason  = None

print("-" * 70)
print(f"quality_score        : {quality_score:.4f}")
print(f"final_status         : {final_status}")
if fail_reason:
    print(f"Motivo               : {fail_reason}")
print(f"  Todos os checks  — OK={checks_ok} | WARNING={checks_warning} | HIGH={checks_high} | CRITICAL={checks_critical}")
print(f"  Bloqueantes      — CRITICAL={gating_checks_critical}  (gating da Silver)")
print(f"  Não bloqueantes  — CRITICAL={non_blocking_critical}  (Silver já trata)")
print("-" * 70)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Alertas automáticos
# MAGIC
# MAGIC O notebook não interrompe o pipeline com `raise` — registra o status em `quality_summary`
# MAGIC e imprime avisos explícitos. A decisão de bloquear a Silver cabe ao orquestrador
# MAGIC (verificar `final_status` na tabela antes de disparar o próximo job).

# COMMAND ----------

if final_status == 'FAIL':
    print("=" * 70)
    print("ALERTA — FAIL")
    print(f"  {fail_reason}")
    print(f"  {non_blocking_critical} check(s) CRITICAL não-bloqueante(s) registrado(s) — Silver já trata.")
    print("  A Silver não deve ser executada até que os problemas CRITICAL bloqueantes sejam resolvidos.")
    print("=" * 70)
elif final_status == 'WARN':
    print("-" * 70)
    print(f"AVISO — quality_score {quality_score:.4f} abaixo de {SCORE_WARN_THRESHOLD}")
    print(f"  {non_blocking_critical} check(s) CRITICAL não-bloqueante(s) registrado(s) — Silver já trata.")
    print("  Revisar campos com status HIGH antes de processar a Silver.")
    print("-" * 70)
else:
    if non_blocking_critical > 0:
        print(f"PASS — quality_score {quality_score:.4f}  "
              f"({non_blocking_critical} CRITICAL não-bloqueante(s) registrado(s), Silver já trata)")
    else:
        print(f"PASS — quality_score {quality_score:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Persistência do resumo

# COMMAND ----------

summary_record = {
    'validation_id':            CURRENT_VALIDATION_ID,
    'timestamp':                datetime.now(),
    'total_records':            total_rows,
    'total_columns':            total_cols,
    'total_checks':             total_checks,
    'checks_ok':                checks_ok,
    'checks_warning':           checks_warning,
    'checks_high':              checks_high,
    'checks_critical':          checks_critical,
    'gating_checks_critical':   gating_checks_critical,   # CRITICAL que bloqueiam a Silver
    'non_blocking_critical':    non_blocking_critical,     # CRITICAL tratados pela Silver
    'critical_fields_analyzed': len(existing_critical_fields),
    'fields_with_high_missing': sum(1 for r in completeness_results if r['null_pct'] > 20),
    'fields_with_high_code9':   sum(1 for r in code9_results if r['code9_pct'] > 20),
    'quality_score':            quality_score,
    'final_status':             final_status,
    'fail_reason':              fail_reason,
}

df_summary = spark.createDataFrame([summary_record])

(
    df_summary.write
    .mode("append")
    .option("mergeSchema", "true")
    .saveAsTable(TABLE_QUALITY_SUMMARY)
)
print(f"Resumo gravado em: {TABLE_QUALITY_SUMMARY}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Score atual e evolução histórica

# COMMAND ----------

print(f"Run atual — validation_id: {CURRENT_VALIDATION_ID}")
display(
    spark.table(TABLE_QUALITY_SUMMARY)
    .filter(F.col('validation_id') == CURRENT_VALIDATION_ID)
    .select('validation_id', 'timestamp', 'quality_score', 'final_status',
            'checks_ok', 'checks_warning', 'checks_high', 'checks_critical')
)

# COMMAND ----------

# Histórico não filtrado por validation_id — análise de tendência entre runs
print("Histórico de quality_score (últimos 10 runs — sem filtro de validation_id):")
display(
    spark.table(TABLE_QUALITY_SUMMARY)
    .orderBy(F.desc('timestamp'))
    .limit(10)
    .select('validation_id', 'timestamp', 'quality_score', 'final_status',
            'checks_critical', 'checks_high')
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Decisões para a Silver
# MAGIC
# MAGIC Os achados abaixo refletem a execução atual (`CURRENT_VALIDATION_ID`).
# MAGIC As regras foram definidas com base no comportamento real dos dados, não apenas
# MAGIC na documentação oficial do DATASUS.

# COMMAND ----------

print(f"Campos com problemas (validation_id = {CURRENT_VALIDATION_ID}):\n")
display(
    spark.table(TABLE_QUALITY_CHECKS)
    .filter(
        (F.col('validation_id') == CURRENT_VALIDATION_ID) &
        (F.col('status').isin(['CRITICAL', 'HIGH']))
    )
    .select('check_type', 'field', 'null_pct', 'invalid_pct', 'duplicate_pct', 'status')
    .orderBy(
        F.when(F.col('status') == 'CRITICAL', 1).when(F.col('status') == 'HIGH', 2).otherwise(3),
        F.desc('null_pct')
    )
)

# COMMAND ----------

silver_rules = """
REGRAS PARA A SILVER — DERIVADAS DESTA EXECUÇÃO DE VALIDAÇÃO
=============================================================

1. Filtros obrigatórios (registros a excluir)
   - DT_SIN_PRI IS NOT NULL  (sem data de sintoma, não é possível compor métricas temporais)
   - DT_NOTIFIC IS NOT NULL  (rastreabilidade)
   - ANO_DADOS IN (2023, 2024, 2025)
   - NU_NOTIFIC IS NOT NULL

2. Deduplicação
   - Deduplicar por NU_NOTIFIC (manter o registro com DT_NOTIFIC mais recente).
   - Taxa atual de duplicação é baixa (<0,1%), mas deve ser tratada.

3. Conversão de tipos
   - Campos de data (DT_NOTIFIC, DT_SIN_PRI, DT_INTERNA, DT_EVOLUCA):
       coalesce(to_date(col, 'dd/MM/yyyy'), to_date(col, 'yyyy-MM-dd'))
   - NU_IDADE_N → INTEGER (validar intervalo 0–120)
   - CS_SEXO: se o dataset contiver codificação alfanumérica (M/F/I), normalizar
     para numérica (M→1, F→2) antes de persistir na Silver, ou manter como-está
     com documentação no metadado.

4. Campos calculados
   - tempo_sintoma_notificacao = DT_NOTIFIC - DT_SIN_PRI  (dias)
   - tempo_sintoma_internacao  = DT_INTERNA - DT_SIN_PRI  (dias)
   - tempo_internacao_desfecho = DT_EVOLUCA - DT_INTERNA   (dias)
   - faixa_etaria              (categorizar NU_IDADE_N)
   - semana_epidemiologica     (validar SEM_PRI)

5. Tratamento do código "9" (Ignorado)
   - NÃO imputar. Manter como categoria válida.
   - Criar flag booleana `is_complete` excluindo NULL e "9" para filtros opcionais.

6. EVOLUCAO — domínio oficial (dicionário SRAG 2019–2025)
     1 = Cura
     2 = Óbito por SRAG
     3 = Óbito por outras causas
     9 = Ignorado
   - Taxa de mortalidade por SRAG: filtrar EVOLUCAO = '2'.
   - Mortalidade hospitalar geral: filtrar EVOLUCAO IN ('2', '3').
   - Valores '9' e NULL devem ser excluídos dos denominadores de mortalidade,
     salvo quando a análise contemple completude de notificação.
   - EVOLUCAO = '3' é categoria válida documentada — não é erro de dados.

7. CLASSI_FIN — classi_fin_clean na Silver
   Domínio: 1=Influenza, 2=Outro vírus, 3=Outro agente, 4=Não especificado, 5=COVID-19.
   Silver v2: cria classi_fin_clean (code 9 → NULL), is_covid, is_influenza, is_outro_virus.
   Ver DQ check CLASSI_FIN para distribuição de qualidade por ano.

8. VACINA: 24% NULL — esperado em dados recentes (notificação tardia). Aceitar como-está.

9. Campos a descartar (decisão da Silver)
   - Colunas com >80% NULL que não fazem parte dos campos críticos.
   - Colunas inteiramente NULL.
"""
print(silver_rules)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Queries de auditoria

# COMMAND ----------

audit_queries = f"""
-- Todos os checks da execução atual
SELECT check_type, field, status, null_pct, invalid_pct, duplicate_pct
FROM {TABLE_QUALITY_CHECKS}
WHERE validation_id = '{CURRENT_VALIDATION_ID}'
ORDER BY
    CASE status WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2 WHEN 'WARNING' THEN 3 ELSE 4 END,
    null_pct DESC;

-- Apenas checks CRITICAL da execução atual
SELECT field, check_type, null_pct, invalid_pct, status
FROM {TABLE_QUALITY_CHECKS}
WHERE validation_id = '{CURRENT_VALIDATION_ID}'
  AND status = 'CRITICAL';

-- Tendência do quality_score (últimos 10 runs)
SELECT validation_id, DATE(timestamp) AS data_execucao, quality_score, final_status,
       checks_ok, checks_critical
FROM {TABLE_QUALITY_SUMMARY}
ORDER BY timestamp DESC
LIMIT 10;

-- Campos que já foram CRITICAL em algum run
SELECT field, validation_id, null_pct, status, timestamp
FROM {TABLE_QUALITY_CHECKS}
WHERE field IN (
    SELECT DISTINCT field FROM {TABLE_QUALITY_CHECKS} WHERE status = 'CRITICAL'
)
ORDER BY field, timestamp DESC;
"""
print(audit_queries)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resumo da execução

# COMMAND ----------

print("-" * 70)
print("VALIDAÇÃO DE QUALIDADE ENCERRADA")
print("-" * 70)
print(f"  validation_id  : {CURRENT_VALIDATION_ID}")
print(f"  Registros      : {total_rows:,}")
print(f"  Total checks   : {total_checks}")
print(f"  quality_score  : {quality_score:.4f}")
print(f"  final_status   : {final_status}")
if fail_reason:
    print(f"  Motivo         : {fail_reason}")
print(f"  Tabela checks  : {TABLE_QUALITY_CHECKS}")
print(f"  Tabela resumo  : {TABLE_QUALITY_SUMMARY}")
print("-" * 70)
print("Próximo: 03_eda_srag_exploraty_analysis.py  |  04_Silver_Transformation.py")
