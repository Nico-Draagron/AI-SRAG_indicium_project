# Databricks notebook source
# MAGIC %md
# MAGIC # Análise Exploratória de Dados — SRAG
# MAGIC
# MAGIC **Projeto**: Sistema RAG para Monitoramento Epidemiológico — Indicium Healthcare PoC
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Objetivo
# MAGIC
# MAGIC Explorar os dados da camada Bronze, validar as quatro métricas epidemiológicas
# MAGIC obrigatórias e gerar séries temporais reprodutíveis. Outputs são persistidos como
# MAGIC tabelas Delta (schema `data_original`), particionadas por `run_id`.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Métricas e Denominadores
# MAGIC
# MAGIC | Métrica | Fórmula | Denominador | Exclusões |
# MAGIC |---|---|---|---|
# MAGIC | Mortalidade SRAG | obitos / (curas + obitos) × 100 | EVOLUCAO IN (1,2) | 3, 9, NULL |
# MAGIC | Ocupação UTI | casos_uti / hosp_com_info_uti × 100 | HOSPITAL=1 e UTI IN (1,2) | 9, NULL |
# MAGIC | Vacinação | vacinados / (vac + nao_vac) × 100 | VACINA IN (1,2) | 9, NULL |
# MAGIC | Crescimento mensal | (mes - mes_ant) / mes_ant × 100 | Últimos 12 meses (dt_sintomas) | NULL |
# MAGIC
# MAGIC - Mortalidade agrega por mês de desfecho (`dt_evolucao`), com fallback para
# MAGIC   `dt_sintomas` quando nulo (`fallback_pct` audita a proporção).
# MAGIC - `crescimento_pct = NULL` quando `casos_mes_anterior = 0` (indeterminado).
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Tabelas de Output
# MAGIC
# MAGIC | Tabela | Conteúdo | Modo |
# MAGIC |---|---|---|
# MAGIC | `eda_serie_diaria_90d` | Casos/dia, últimos 90 dias | append |
# MAGIC | `eda_series_mensal` | Casos/mês + crescimento, 12 meses | append |
# MAGIC | `eda_mortalidade_mensal` | Taxa mortalidade/mês + fallback_pct | append |
# MAGIC | `eda_vacinacao_mensal` | Taxa vacinação/mês | append |
# MAGIC
# MAGIC > **Nota**: Esta tabela usa janela de 90 dias para exploração.
# MAGIC > O gráfico diário do relatório do agente usa `gold_serie_diaria_30d` (Gold Layer).
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Dependências
# MAGIC
# MAGIC - **Upstream**: `01_Bronze_Ingestion_SRAG.py` → `bronze_srag_raw`
# MAGIC - **Downstream**: `04_Silver_Transformation.py`
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Notas de Compatibilidade (Serverless)
# MAGIC
# MAGIC - Datas: `parse_date_safe()` com regex gate — evita `DateTimeException` no ANSI estrito.
# MAGIC   O padrão `coalesce(to_date(...), to_date(...))` sem gate lança exceção antes do coalesce
# MAGIC   no Serverless/Photon.
# MAGIC - `resolve_col()`: lookup case-insensitive — Bronze (`DT_SIN_PRI`) e Silver (`dt_sin_pri`).
# MAGIC - Colunas já `DateType` (Silver) usadas diretamente, sem re-parse.
# MAGIC - Gráficos: `display(fig)` + `plt.close()` — sem escrita em filesystem.
# MAGIC   Serverless proíbe acesso local (`/tmp`) e DBFS via `dbutils.fs.cp("file:...")`.
# MAGIC - Cramér's V via NumPy puro (sem SciPy); heatmap via Matplotlib puro (sem Seaborn).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import DateType
from datetime import datetime
from uuid import uuid4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 10

PALETTE = {
    'primary':      '#2c7bb6',
    'secondary':    '#d7191c',
    'positive':     '#1a9641',   # verde = crescimento positivo (visual; não julgamento epidemiológico)
    'negative':     '#d7191c',   # vermelho = queda (idem)
    'neutral':      '#fdae61',
    'male':         '#4393c3',
    'female':       '#d6604d',
    'uti':          '#e66101',
    'enfermaria':   '#4dac26',
    'vacinado':     '#1a9641',
    'nao_vacinado': '#d7191c',
    'cura':         '#1a9641',
    'obito':        '#d7191c',
}

print("-" * 80)
print("ANÁLISE EXPLORATÓRIA — SRAG")
print(f"Início : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Spark  : {spark.version}")
print("-" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuração

# COMMAND ----------

CATALOG = "dbx_srag_lab"
SCHEMA  = "data_original"
TABLE_BRONZE = f"{CATALOG}.{SCHEMA}.bronze_srag_raw"

TABLE_SERIE_DIARIA       = f"{CATALOG}.{SCHEMA}.eda_serie_diaria_90d"
TABLE_SERIES_MENSAL      = f"{CATALOG}.{SCHEMA}.eda_series_mensal"
TABLE_MORTALIDADE_MENSAL = f"{CATALOG}.{SCHEMA}.eda_mortalidade_mensal"
TABLE_VACINACAO_MENSAL   = f"{CATALOG}.{SCHEMA}.eda_vacinacao_mensal"

RUN_ID = f"{datetime.now():%Y%m%d_%H%M%S_%f}_{uuid4().hex[:8]}"

print(f"Fonte  : {TABLE_BRONZE}")
print(f"Run ID : {RUN_ID}")

# ── Configuração de diretório de imagens ─────────────────────────────────────
# Caminho relativo: notebooks/EDA/ → ../../images/graficos/
try:
    _nb_raw  = (dbutils.notebook.entry_point
                .getDbutils().notebook().getContext()
                .notebookPath().get())               # /Users/.../notebooks/EDA/03_eda_...
    _nb_dir  = "/Workspace/" + "/".join(_nb_raw.lstrip("/").split("/")[:-1])
    IMAGES_DIR = os.path.normpath(os.path.join(_nb_dir, "../../images/graficos"))
except Exception:
    IMAGES_DIR = "/tmp/eda_graficos"                 # fallback para execução local

os.makedirs(IMAGES_DIR, exist_ok=True)
print(f"[images] Diretório configurado: {IMAGES_DIR}")


def _save_fig(fig, filename: str) -> None:
    """Exibe no notebook Databricks E salva em IMAGES_DIR. Fecha a figura."""
    display(fig)
    dest = os.path.join(IMAGES_DIR, filename)
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    print(f"  [img] → {dest}")
    plt.close(fig)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Utilitários de Parsing Seguro
# MAGIC
# MAGIC `parse_date_safe`: regex gate antes de `to_date` — no Serverless/ANSI, `to_date` lança
# MAGIC `DateTimeException` imediatamente quando o formato não casa, antes do `coalesce` agir.
# MAGIC O regex garante que só chamamos `to_date` com valores no formato correto.
# MAGIC
# MAGIC `resolve_col`: lookup case-insensitive — Bronze pode ter `DT_SIN_PRI`, Silver `dt_sin_pri`.
# MAGIC
# MAGIC `_prepare_date_col`: se coluna já é `DateType` (Silver), usa direto; senão `parse_date_safe`.

# COMMAND ----------

def parse_date_safe(col_name: str):
    """Regex gate + to_date. Suporta yyyy-MM-dd e dd/MM/yyyy. Nunca lança exceção."""
    c   = F.trim(F.col(col_name).cast("string"))
    iso = F.when(c.rlike(r"^\d{4}-\d{2}-\d{2}$"), F.to_date(c, "yyyy-MM-dd"))
    br  = F.when(c.rlike(r"^\d{2}/\d{2}/\d{4}$"), F.to_date(c, "dd/MM/yyyy"))
    return F.coalesce(iso, br)


def resolve_col(df_spark, candidates):
    """Retorna nome real da coluna: match exato primeiro, depois case-insensitive. None se ausente."""
    cols      = set(df_spark.columns)
    lower_map = {x.lower(): x for x in df_spark.columns}
    for c in candidates:
        if c in cols:
            return c
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _prepare_date_col(df_spark, candidates, alias: str):
    """DateType → usa direto. String → parse_date_safe. Ausente → NULL date."""
    col_name = resolve_col(df_spark, candidates)
    if col_name is None:
        print(f"  [AVISO] '{alias}' não encontrada {candidates} → NULL.")
        return F.lit(None).cast("date")
    col_type = dict(df_spark.dtypes).get(col_name, "string")
    if col_type == "date":
        print(f"  [Silver] '{col_name}' DateType → sem re-parse.")
        return F.col(col_name)
    print(f"  [Bronze] '{col_name}' ({col_type}) → parse_date_safe.")
    return parse_date_safe(col_name)


print("Utilitários OK: parse_date_safe | resolve_col | _prepare_date_col")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Carregamento dos dados

# COMMAND ----------

df_raw     = spark.table(TABLE_BRONZE)
total_rows = df_raw.count()
total_cols = len(df_raw.columns)

print(f"Registros : {total_rows:,}")
print(f"Colunas   : {total_cols}")
df_raw.groupBy("ANO_DADOS").count().orderBy("ANO_DADOS").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Preparação de campos
# MAGIC
# MAGIC - Datas: `_prepare_date_col` detecta `DateType` (Silver) ou string (Bronze) automaticamente.
# MAGIC - `ano_mes_date`: primeiro dia do mês (`F.trunc`), para ordenação e joins.
# MAGIC - `idade_anos`: apenas quando `TP_IDADE = '3'` (anos completos, DATASUS).
# MAGIC - `CS_SEXO`: cast para string — suporta encoding `1/2/9` e `M/F/I`.

# COMMAND ----------

print("Resolvendo colunas de data...")

expr_dt_sintomas    = _prepare_date_col(df_raw, ["DT_SIN_PRI",  "dt_sin_pri"],  "dt_sintomas")
expr_dt_notificacao = _prepare_date_col(df_raw, ["DT_NOTIFIC",  "dt_notific"],  "dt_notificacao")
expr_dt_internacao  = _prepare_date_col(df_raw, ["DT_INTERNA",  "dt_interna"],  "dt_internacao")
expr_dt_evolucao    = _prepare_date_col(df_raw, ["DT_EVOLUCA",  "dt_evoluca",
                                                  "DT_EVOLUCAO", "dt_evolucao"], "dt_evolucao")

_tp_idade = resolve_col(df_raw, ["TP_IDADE",   "tp_idade"])   or "TP_IDADE"
_nu_idade = resolve_col(df_raw, ["NU_IDADE_N", "nu_idade_n"]) or "NU_IDADE_N"
_cs_sexo  = resolve_col(df_raw, ["CS_SEXO",    "cs_sexo"])    or "CS_SEXO"

df = (
    df_raw
    .withColumn("dt_sintomas",    expr_dt_sintomas)
    .withColumn("dt_notificacao", expr_dt_notificacao)
    .withColumn("dt_internacao",  expr_dt_internacao)
    .withColumn("dt_evolucao",    expr_dt_evolucao)
    .withColumn("CS_SEXO",        F.col(_cs_sexo).cast("string"))
    .withColumn("ano_mes_date",   F.trunc(F.col("dt_sintomas"), "month"))
    .withColumn("ano_mes",
        F.concat(F.year("dt_sintomas"), F.lit("-"),
                 F.lpad(F.month("dt_sintomas"), 2, "0")))
    .withColumn("idade_anos",
        F.when((F.col(_tp_idade) == "3") & F.col(_nu_idade).isNotNull(),
               F.col(_nu_idade).cast("int")).otherwise(None))
    # ALINHADO com Silver silver_srag_clean — comparação direta com Gold Demográficas.
    .withColumn("faixa_etaria",
        F.when(F.col("idade_anos") < 1,   "0-1 ano")
         .when(F.col("idade_anos") < 5,   "1-4 anos")
         .when(F.col("idade_anos") < 10,  "5-9 anos")
         .when(F.col("idade_anos") < 18,  "10-17 anos")
         .when(F.col("idade_anos") < 30,  "18-29 anos")
         .when(F.col("idade_anos") < 40,  "30-39 anos")
         .when(F.col("idade_anos") < 50,  "40-49 anos")
         .when(F.col("idade_anos") < 60,  "50-59 anos")
         .when(F.col("idade_anos") < 70,  "60-69 anos")
         .when(F.col("idade_anos") >= 70, "70+ anos")
         .otherwise("Desconhecido"))
)

print("\nCampos preparados: dt_sintomas, dt_notificacao, dt_internacao, dt_evolucao")
print("Derivados        : ano_mes_date, ano_mes, faixa_etaria, idade_anos")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Análise temporal

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.1 Série diária — últimos 90 dias

# COMMAND ----------

data_corte_90d = F.date_sub(F.current_date(), 90)

pd_diaria = (
    df
    .filter(F.col("dt_sintomas").isNotNull() & (F.col("dt_sintomas") >= data_corte_90d))
    .groupBy("dt_sintomas").agg(F.count("*").alias("casos"))
    .orderBy("dt_sintomas")
    .limit(10_000)
    .toPandas()
)

if len(pd_diaria) > 0:
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(pd_diaria['dt_sintomas'], pd_diaria['casos'],
            marker='o', linewidth=2, color=PALETTE['primary'])
    ax.set_title('Casos SRAG — Série Diária (últimos 90 dias)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Data de Primeiros Sintomas'); ax.set_ylabel('Número de Casos')
    ax.tick_params(axis='x', rotation=45); ax.grid(True, alpha=0.3)
    plt.tight_layout(); _save_fig(fig, "eda_01_serie_diaria_90d.png")

    print(f"Período   : {pd_diaria['dt_sintomas'].min()} a {pd_diaria['dt_sintomas'].max()}")
    print(f"Total     : {pd_diaria['casos'].sum():,} casos")
    print(f"Média/dia : {pd_diaria['casos'].mean():.1f} | "
          f"Máximo: {pd_diaria['casos'].max()} em "
          f"{pd_diaria.loc[pd_diaria['casos'].idxmax(), 'dt_sintomas']}")
else:
    print("Sem dados nos últimos 90 dias.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.2 Série mensal — últimos 12 meses
# MAGIC
# MAGIC Janela dinâmica: `max(dt_sintomas) − 11 meses`. Crescimento = `(mes − mes_ant) / mes_ant × 100`.
# MAGIC Primeiro mês sem referência anterior → `crescimento_pct = NULL`.

# COMMAND ----------

max_dt = (df.filter(F.col("dt_sintomas").isNotNull())
            .agg(F.max("dt_sintomas").alias("m")).collect()[0]["m"])
data_inicio_12m = spark.sql(
    f"SELECT add_months(DATE '{max_dt}', -11) AS inicio"
).collect()[0]["inicio"]

print(f"Janela: {data_inicio_12m} → {max_dt}")

pd_mensal = (
    df
    .filter(F.col("dt_sintomas").isNotNull() &
            (F.col("dt_sintomas") >= F.lit(data_inicio_12m)))
    .groupBy("ano_mes_date", "ano_mes")
    .agg(F.count("*").alias("casos"))
    .orderBy("ano_mes_date")
    .limit(10_000)
    .toPandas()
)

if len(pd_mensal) > 0:
    pd_mensal['casos_anterior'] = pd_mensal['casos'].shift(1)
    pd_mensal['crescimento_pct'] = np.where(
        pd_mensal['casos_anterior'].isna() | (pd_mensal['casos_anterior'] == 0),
        np.nan,
        (pd_mensal['casos'] - pd_mensal['casos_anterior'])
        / pd_mensal['casos_anterior'] * 100
    )

    # Gráfico 1 — volume mensal
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(pd_mensal['ano_mes'], pd_mensal['casos'], color=PALETTE['primary'], alpha=0.75)
    ax.set_title('Casos SRAG por Mês — Últimos 12 Meses', fontsize=14, fontweight='bold')
    ax.set_xlabel('Mês'); ax.set_ylabel('Número de Casos')
    ax.tick_params(axis='x', rotation=45); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout(); _save_fig(fig, "eda_02_casos_mensal_12m.png")

    # Gráfico 2 — crescimento mensal
    cresc_plot   = pd_mensal['crescimento_pct'].fillna(0)
    colors_cresc = [PALETTE['positive'] if v >= 0 else PALETTE['negative'] for v in cresc_plot]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(pd_mensal['ano_mes'], cresc_plot, color=colors_cresc, alpha=0.75)
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_title('Taxa de Crescimento Mensal (%) — Últimos 12 Meses', fontsize=14, fontweight='bold')
    ax.set_xlabel('Mês'); ax.set_ylabel('Variação (%)')
    ax.tick_params(axis='x', rotation=45); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout(); _save_fig(fig, "eda_03_crescimento_mensal_12m.png")

    cresc_valido = pd_mensal['crescimento_pct'].dropna()
    print(f"Média/mês  : {pd_mensal['casos'].mean():,.0f} | Mediana: {pd_mensal['casos'].median():,.0f}")
    print(f"Maior mês  : {pd_mensal.loc[pd_mensal['casos'].idxmax(),'ano_mes']} "
          f"({pd_mensal['casos'].max():,})")
    print(f"Crescimento: média {cresc_valido.mean():.2f}% | "
          f"alta {cresc_valido.max():.2f}% | queda {cresc_valido.min():.2f}%")
else:
    print("Sem dados mensais disponíveis.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.3 Sazonalidade por semana epidemiológica

# COMMAND ----------

from pyspark.sql.types import IntegerType as _IntType
# Filtra valores fora de 1-53 (mesmo critério que a Silver usa em sem_pri).
pd_semanal = (
    df.filter(
        F.col("SEM_PRI").isNotNull() &
        F.col("SEM_PRI").cast(_IntType()).between(1, 53)
    )
    .groupBy("SEM_PRI", "ANO_DADOS").agg(F.count("*").alias("casos"))
    .orderBy("ANO_DADOS", "SEM_PRI")
    .limit(10_000).toPandas()
)

if len(pd_semanal) > 0:
    anos = sorted(pd_semanal['ANO_DADOS'].unique())
    cmap = plt.cm.get_cmap('Blues', len(anos) + 2)
    fig, ax = plt.subplots(figsize=(16, 6))
    for idx, ano in enumerate(anos):
        d = pd_semanal[pd_semanal['ANO_DADOS'] == ano]
        ax.plot(d['SEM_PRI'].astype(int), d['casos'],
                marker='o', label=str(int(ano)), linewidth=2, markersize=4,
                color=cmap(idx + 2))
    ax.set_title('Sazonalidade — Casos por Semana Epidemiológica', fontsize=14, fontweight='bold')
    ax.set_xlabel('Semana Epidemiológica'); ax.set_ylabel('Número de Casos')
    ax.legend(title='Ano'); ax.grid(True, alpha=0.3)
    plt.tight_layout(); _save_fig(fig, "eda_04_sazonalidade_semana.png")

    semanas_pico = pd_semanal.groupby('SEM_PRI')['casos'].mean().nlargest(5)
    print("Semanas de pico (média): " + ", ".join([f"SE{int(s)}" for s in semanas_pico.index]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.4 Comparação anual

# COMMAND ----------

pd_anual = (
    df.groupBy("ANO_DADOS").agg(F.count("*").alias("casos"))
    .orderBy("ANO_DADOS").limit(10_000).toPandas()
)

if len(pd_anual) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(pd_anual['ANO_DADOS'].astype(str), pd_anual['casos'],
                  color=PALETTE['neutral'], alpha=0.75)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h,
                f'{int(h):,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_title('Casos SRAG por Ano', fontsize=14, fontweight='bold')
    ax.set_xlabel('Ano'); ax.set_ylabel('Número de Casos'); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout(); _save_fig(fig, "eda_05_casos_por_ano.png")

    pd_anual['variacao_pct'] = pd_anual['casos'].pct_change() * 100
    for _, row in pd_anual.iterrows():
        suf = f" ({row['variacao_pct']:+.1f}%)" if pd.notna(row['variacao_pct']) else ""
        print(f"  {int(row['ANO_DADOS'])}: {int(row['casos']):,} casos{suf}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Análise demográfica

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.1 Distribuição geográfica (UF)

# COMMAND ----------

_sg_uf = resolve_col(df, ["SG_UF","sg_uf"]) or "SG_UF"

pd_uf = (
    df.filter(F.col(_sg_uf).isNotNull())
    .groupBy(_sg_uf).agg(F.count("*").alias("casos"))
    .orderBy(F.desc("casos")).limit(15)
    .toPandas().rename(columns={_sg_uf: "SG_UF"})
)

if len(pd_uf) > 0:
    pd_uf['percentual'] = pd_uf['casos'] / pd_uf['casos'].sum() * 100
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(pd_uf['SG_UF'], pd_uf['casos'], color=PALETTE['primary'], alpha=0.75)
    ax.invert_yaxis()
    ax.set_title('Top 15 UFs — Número de Casos', fontsize=13, fontweight='bold')
    ax.set_xlabel('Casos'); ax.grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(pd_uf['casos']):
        ax.text(v, i, f'  {v:,}', va='center', fontsize=9)
    plt.tight_layout(); _save_fig(fig, "eda_06_top_ufs.png")

    for _, row in pd_uf.head(5).iterrows():
        print(f"  {row['SG_UF']}: {row['casos']:,} ({row['percentual']:.1f}%)")
    print(f"Concentração Top 5: {pd_uf.head(5)['percentual'].sum():.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.2 Distribuição por sexo
# MAGIC
# MAGIC Encoding misto normalizado: `1`/`M` → Masculino, `2`/`F` → Feminino.

# COMMAND ----------

pd_sexo = (
    df.filter(F.col("CS_SEXO").isin('1','2','M','F'))
    .withColumn("sexo_label",
        F.when(F.col("CS_SEXO").isin("1","M"), "Masculino").otherwise("Feminino"))
    .groupBy("sexo_label").agg(F.count("*").alias("casos"))
    .limit(10_000).toPandas()
)

if len(pd_sexo) > 0:
    pd_sexo['percentual'] = pd_sexo['casos'] / pd_sexo['casos'].sum() * 100
    colors_sex = [PALETTE['male'] if s == 'Masculino' else PALETTE['female']
                  for s in pd_sexo['sexo_label']]
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(pd_sexo['sexo_label'], pd_sexo['casos'], color=colors_sex, alpha=0.75)
    ax.set_title('Distribuição por Sexo', fontsize=13, fontweight='bold')
    ax.set_ylabel('Casos'); ax.grid(True, alpha=0.3, axis='y')
    for bar, pct in zip(bars, pd_sexo['percentual']):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h,
                f'{int(h):,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=11)
    plt.tight_layout(); _save_fig(fig, "eda_07_distribuicao_sexo.png")

    for _, row in pd_sexo.iterrows():
        print(f"  {row['sexo_label']}: {row['casos']:,} ({row['percentual']:.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.3 Distribuição por faixa etária

# COMMAND ----------

pd_idade = (
    df.filter((F.col("faixa_etaria") != "Desconhecido") & F.col("faixa_etaria").isNotNull())
    .groupBy("faixa_etaria").agg(F.count("*").alias("casos"))
    .limit(10_000).toPandas()
)

if len(pd_idade) > 0:
    ordem = ["0-1 ano","1-4 anos","5-9 anos","10-17 anos","18-29 anos",
             "30-39 anos","40-49 anos","50-59 anos","60-69 anos","70+ anos"]
    pd_idade['faixa_etaria'] = pd.Categorical(pd_idade['faixa_etaria'],
                                               categories=ordem, ordered=True)
    pd_idade = pd_idade.sort_values('faixa_etaria')
    pd_idade['percentual'] = pd_idade['casos'] / pd_idade['casos'].sum() * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(pd_idade['faixa_etaria'].astype(str), pd_idade['casos'],
                  color=PALETTE['primary'], alpha=0.75)
    ax.set_title('Distribuição por Faixa Etária', fontsize=13, fontweight='bold')
    ax.set_xlabel('Faixa Etária (anos)'); ax.set_ylabel('Casos'); ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=30)
    for bar, pct in zip(bars, pd_idade['percentual']):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h,
                f'{int(h):,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
    plt.tight_layout(); _save_fig(fig, "eda_08_faixa_etaria.png")

    for _, row in pd_idade.iterrows():
        print(f"  {row['faixa_etaria']}: {row['casos']:,} ({row['percentual']:.1f}%)")
    casos_60p = pd_idade[pd_idade['faixa_etaria'].astype(str).isin(['60-69 anos','70+ anos'])]['casos'].sum()
    print(f"Grupo 60+: {casos_60p:,} ({casos_60p / pd_idade['casos'].sum() * 100:.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Análise Etiológica — CLASSI_FIN
# MAGIC
# MAGIC Distribuição por agente causador de SRAG (campo CLASSI_FIN do SIVEP-Gripe).
# MAGIC 1=Influenza | 2=Outro vírus | 3=Outro agente | 4=Não especificado | 5=COVID-19
# MAGIC Campos NULL e '9' representam classificação não realizada.

# COMMAND ----------

_classi_col = resolve_col(df_raw, ["CLASSI_FIN", "classi_fin"])
if _classi_col:
    # 8.1 Distribuição geral
    pd_classi = (
        df_raw.groupBy(_classi_col, "ANO_DADOS")
        .agg(F.count("*").alias("casos"))
        .orderBy("ANO_DADOS", _classi_col)
        .toPandas()
    )
    # Mapa legível
    classi_map = {'1':'Influenza','2':'Outro vírus','3':'Outro agente',
                  '4':'Não especif.','5':'COVID-19','9':'Ignorado'}
    pd_classi['label'] = pd_classi[_classi_col].map(classi_map).fillna('NULL/Outro')

    # Gráfico de barras empilhadas por ano
    pivot = pd_classi.pivot_table(index='ANO_DADOS', columns='label', values='casos', fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind='bar', ax=ax, alpha=0.85)
    ax.set_title('Classificação Etiológica SRAG por Ano', fontsize=14, fontweight='bold')
    ax.set_xlabel('Ano'); ax.set_ylabel('Casos'); ax.tick_params(axis='x', rotation=0)
    ax.legend(title='CLASSI_FIN', bbox_to_anchor=(1.05,1))
    plt.tight_layout(); _save_fig(fig, "eda_09_classificacao_etiologica.png")

    # 8.2 Taxa de NULL/9 por ano
    pct_sem_classi = (
        df_raw.groupBy("ANO_DADOS").agg(
            F.count("*").alias("total"),
            F.sum(F.when(F.col(_classi_col).isNull() | (F.col(_classi_col) == "9"), 1).otherwise(0))
             .alias("sem_classificacao")
        ).withColumn("pct_sem_classi", F.round(F.col("sem_classificacao") / F.col("total") * 100, 2))
        .orderBy("ANO_DADOS")
    )
    print("Taxa de registros sem classificação etiológica por ano:")
    display(pct_sem_classi)
else:
    print("AVISO: CLASSI_FIN não encontrado na fonte. Análise etiológica indisponível.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8b. Análise Histórica Comparativa 2023–2025
# MAGIC
# MAGIC Comparações entre anos para identificar tendências epidemiológicas de médio prazo.
# MAGIC Todos os gráficos desta seção usam `ANO_DADOS` como dimensão de comparação.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8b.1 Mortalidade por ano — evolução histórica

# COMMAND ----------

pd_mort_hist = (
    df.filter(F.col("EVOLUCAO").isin("1","2"))
    .groupBy("ANO_DADOS","EVOLUCAO").agg(F.count("*").alias("casos"))
    .orderBy("ANO_DADOS").toPandas()
)

if len(pd_mort_hist) > 0:
    mort_h = (pd_mort_hist.pivot(index='ANO_DADOS', columns='EVOLUCAO', values='casos')
              .fillna(0).rename(columns={'1':'Curas','2':'Óbitos'}))
    mort_h['total']     = mort_h['Curas'] + mort_h['Óbitos']
    mort_h['taxa_mort'] = mort_h['Óbitos'] / mort_h['total'] * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    # Barras empilhadas curas vs óbitos
    axes[0].bar(mort_h.index.astype(str), mort_h['Curas'],   label='Curas',  color=PALETTE['cura'],  alpha=0.8)
    axes[0].bar(mort_h.index.astype(str), mort_h['Óbitos'],  label='Óbitos', color=PALETTE['obito'], alpha=0.8,
                bottom=mort_h['Curas'])
    axes[0].set_title('Desfechos SRAG por Ano\n(EVOLUCAO 3, 9, NULL excluídos)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Casos'); axes[0].legend(); axes[0].grid(True, alpha=0.3, axis='y')

    # Linha de taxa de mortalidade
    axes[1].plot(mort_h.index.astype(str), mort_h['taxa_mort'],
                 marker='o', linewidth=2.5, color=PALETTE['secondary'], markersize=10)
    for ano, tx in zip(mort_h.index.astype(str), mort_h['taxa_mort']):
        axes[1].annotate(f'{tx:.1f}%', (ano, tx), textcoords='offset points',
                         xytext=(0, 12), ha='center', fontsize=12, fontweight='bold')
    axes[1].set_title('Evolução da Taxa de Mortalidade (%)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Taxa de Mortalidade (%)'); axes[1].set_ylim(0, mort_h['taxa_mort'].max() * 1.3)
    axes[1].grid(True, alpha=0.3)
    plt.suptitle('Análise de Mortalidade SRAG — 2023 a 2025', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout(); _save_fig(fig, "eda_10_desfechos_e_mortalidade.png")

    for ano in mort_h.index:
        print(f"  {int(ano)}: Taxa {mort_h.loc[ano,'taxa_mort']:.2f}% | "
              f"Óbitos {int(mort_h.loc[ano,'Óbitos']):,} / {int(mort_h.loc[ano,'total']):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8b.2 Ocupação UTI por ano — comparação histórica

# COMMAND ----------

pd_uti_hist = (
    df.filter((F.col("HOSPITAL") == "1") & F.col("UTI").isin("1","2"))
    .groupBy("ANO_DADOS","UTI").agg(F.count("*").alias("casos"))
    .orderBy("ANO_DADOS").toPandas()
)

if len(pd_uti_hist) > 0:
    uti_h = (pd_uti_hist.pivot(index='ANO_DADOS', columns='UTI', values='casos')
             .fillna(0).rename(columns={'1':'UTI','2':'Enfermaria'}))
    uti_h['total']    = uti_h['UTI'] + uti_h['Enfermaria']
    uti_h['taxa_uti'] = uti_h['UTI'] / uti_h['total'] * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(len(uti_h))
    w = 0.35
    axes[0].bar(x - w/2, uti_h['UTI'],        width=w, label='UTI',        color=PALETTE['uti'],        alpha=0.8)
    axes[0].bar(x + w/2, uti_h['Enfermaria'], width=w, label='Enfermaria', color=PALETTE['enfermaria'], alpha=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(uti_h.index.astype(str))
    axes[0].set_title('UTI vs Enfermaria por Ano', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Internações'); axes[0].legend(); axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].plot(uti_h.index.astype(str), uti_h['taxa_uti'],
                 marker='s', linewidth=2.5, color=PALETTE['uti'], markersize=10)
    for ano, tx in zip(uti_h.index.astype(str), uti_h['taxa_uti']):
        axes[1].annotate(f'{tx:.1f}%', (ano, tx), textcoords='offset points',
                         xytext=(0, 12), ha='center', fontsize=12, fontweight='bold')
    axes[1].set_title('Evolução da Taxa de Ocupação UTI (%)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Taxa UTI (%)'); axes[1].set_ylim(0, uti_h['taxa_uti'].max() * 1.3)
    axes[1].grid(True, alpha=0.3)
    plt.suptitle('Ocupação UTI SRAG — 2023 a 2025', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout(); _save_fig(fig, "eda_11_uti_internacao.png")

    for ano in uti_h.index:
        print(f"  {int(ano)}: Taxa UTI {uti_h.loc[ano,'taxa_uti']:.2f}% | "
              f"UTI {int(uti_h.loc[ano,'UTI']):,} / {int(uti_h.loc[ano,'total']):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8b.3 Vacinação por ano — tendência histórica

# COMMAND ----------

pd_vac_hist = (
    df.filter(F.col("VACINA").isin("1","2"))
    .groupBy("ANO_DADOS","VACINA").agg(F.count("*").alias("casos"))
    .orderBy("ANO_DADOS").toPandas()
)

if len(pd_vac_hist) > 0:
    vac_h = (pd_vac_hist.pivot(index='ANO_DADOS', columns='VACINA', values='casos')
             .fillna(0).rename(columns={'1':'Vacinado','2':'Não vacinado'}))
    vac_h['total']    = vac_h['Vacinado'] + vac_h['Não vacinado']
    vac_h['taxa_vac'] = vac_h['Vacinado'] / vac_h['total'] * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].bar(vac_h.index.astype(str), vac_h['Vacinado'],     label='Vacinado',     color=PALETTE['vacinado'],    alpha=0.8)
    axes[0].bar(vac_h.index.astype(str), vac_h['Não vacinado'], label='Não vacinado', color=PALETTE['nao_vacinado'], alpha=0.8,
                bottom=vac_h['Vacinado'])
    axes[0].set_title('Vacinação por Ano (excl. 9/NULL)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Casos'); axes[0].legend(); axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].plot(vac_h.index.astype(str), vac_h['taxa_vac'],
                 marker='^', linewidth=2.5, color=PALETTE['vacinado'], markersize=10)
    for ano, tx in zip(vac_h.index.astype(str), vac_h['taxa_vac']):
        axes[1].annotate(f'{tx:.1f}%', (ano, tx), textcoords='offset points',
                         xytext=(0, 12), ha='center', fontsize=12, fontweight='bold')
    axes[1].set_title('Evolução da Taxa de Vacinação (%)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Taxa de Vacinação (%)'); axes[1].set_ylim(0, min(100, vac_h['taxa_vac'].max() * 1.3))
    axes[1].grid(True, alpha=0.3)
    plt.suptitle('Cobertura Vacinal SRAG — 2023 a 2025', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout(); _save_fig(fig, "eda_12_vacinacao.png")

    for ano in vac_h.index:
        print(f"  {int(ano)}: Taxa vacinação {vac_h.loc[ano,'taxa_vac']:.2f}% | "
              f"Vacinados {int(vac_h.loc[ano,'Vacinado']):,} / {int(vac_h.loc[ano,'total']):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8b.4 Perfil etário por ano — evolução da distribuição

# COMMAND ----------

pd_idade_ano = (
    df.filter((F.col("faixa_etaria") != "Desconhecido") & F.col("faixa_etaria").isNotNull())
    .groupBy("ANO_DADOS","faixa_etaria").agg(F.count("*").alias("casos"))
    .orderBy("ANO_DADOS","faixa_etaria").toPandas()
)

if len(pd_idade_ano) > 0:
    ordem_fx = ["0-1 ano","1-4 anos","5-9 anos","10-17 anos","18-29 anos",
                "30-39 anos","40-49 anos","50-59 anos","60-69 anos","70+ anos"]
    pd_idade_ano['faixa_etaria'] = pd.Categorical(pd_idade_ano['faixa_etaria'],
                                                    categories=ordem_fx, ordered=True)
    # Normalizar para % por ano
    totais_ano = pd_idade_ano.groupby('ANO_DADOS')['casos'].transform('sum')
    pd_idade_ano['pct'] = pd_idade_ano['casos'] / totais_ano * 100

    pivot_fx = pd_idade_ano.pivot_table(index='faixa_etaria', columns='ANO_DADOS',
                                         values='pct', fill_value=0).sort_index()

    fig, ax = plt.subplots(figsize=(14, 7))
    x   = np.arange(len(pivot_fx))
    anos_fx = pivot_fx.columns.tolist()
    w   = 0.25
    cmap_fx = plt.cm.get_cmap('Blues', len(anos_fx) + 2)
    for idx, ano in enumerate(anos_fx):
        offset = (idx - len(anos_fx)/2 + 0.5) * w
        bars_fx = ax.bar(x + offset, pivot_fx[ano], width=w,
                         label=str(int(ano)), color=cmap_fx(idx + 2), alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(pivot_fx.index, rotation=30, ha='right')
    ax.set_title('Distribuição Etária por Ano (%) — 2023 a 2025', fontsize=13, fontweight='bold')
    ax.set_ylabel('% de Casos'); ax.legend(title='Ano'); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout(); _save_fig(fig, "eda_13_distribuicao_etaria_ano.png")

    print("% grupo 60+ anos por ano:")
    for ano in anos_fx:
        pct_60p = pivot_fx.loc[['60-69 anos','70+ anos'], ano].sum() \
                  if '60-69 anos' in pivot_fx.index else 0
        print(f"  {int(ano)}: {pct_60p:.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8b.5 Sazonalidade comparativa — sobreposição mensal por ano

# COMMAND ----------

pd_mensal_hist = (
    df.filter(F.col("dt_sintomas").isNotNull())
    .withColumn("mes_num", F.month("dt_sintomas"))
    .withColumn("mes_nome", F.date_format("dt_sintomas", "MMM"))
    .groupBy("ANO_DADOS","mes_num","mes_nome").agg(F.count("*").alias("casos"))
    .orderBy("ANO_DADOS","mes_num").toPandas()
)

if len(pd_mensal_hist) > 0:
    anos_mh = sorted(pd_mensal_hist['ANO_DADOS'].unique())
    cmap_mh = plt.cm.get_cmap('tab10', len(anos_mh))
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Sobreposição de linhas por mês
    for idx, ano in enumerate(anos_mh):
        d = pd_mensal_hist[pd_mensal_hist['ANO_DADOS'] == ano].sort_values('mes_num')
        axes[0].plot(d['mes_num'], d['casos'], marker='o', linewidth=2,
                     label=str(int(ano)), color=cmap_mh(idx), markersize=6)
    axes[0].set_xticks(range(1, 13))
    axes[0].set_xticklabels(['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez'],
                             rotation=30)
    axes[0].set_title('Volume Mensal por Ano — Sobreposição', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Casos'); axes[0].legend(title='Ano'); axes[0].grid(True, alpha=0.3)

    # Heatmap mês × ano
    pivot_mh = pd_mensal_hist.pivot_table(index='mes_num', columns='ANO_DADOS',
                                            values='casos', fill_value=0)
    im_mh = axes[1].imshow(pivot_mh.values, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im_mh, ax=axes[1], label='Casos')
    axes[1].set_yticks(range(len(pivot_mh.index)))
    axes[1].set_yticklabels(['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez'])
    axes[1].set_xticks(range(len(pivot_mh.columns)))
    axes[1].set_xticklabels([str(int(a)) for a in pivot_mh.columns])
    for i in range(len(pivot_mh.index)):
        for j in range(len(pivot_mh.columns)):
            axes[1].text(j, i, f'{int(pivot_mh.values[i,j]):,}',
                         ha='center', va='center', fontsize=8, color='black')
    axes[1].set_title('Heatmap Casos — Mês × Ano', fontsize=12, fontweight='bold')

    plt.suptitle('Sazonalidade Histórica SRAG — 2023 a 2025', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout(); _save_fig(fig, "eda_14_sazonalidade_historica.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8b.6 Mortalidade por faixa etária e ano — análise de risco

# COMMAND ----------

pd_mort_fx = (
    df.filter(F.col("EVOLUCAO").isin("1","2") &
              (F.col("faixa_etaria") != "Desconhecido") & F.col("faixa_etaria").isNotNull())
    .groupBy("ANO_DADOS","faixa_etaria","EVOLUCAO").agg(F.count("*").alias("casos"))
    .toPandas()
)

if len(pd_mort_fx) > 0:
    ordem_fx2 = ["0-1 ano","1-4 anos","5-9 anos","10-17 anos","18-29 anos",
                 "30-39 anos","40-49 anos","50-59 anos","60-69 anos","70+ anos"]
    pivot_mfx = (pd_mort_fx.groupby(['ANO_DADOS','faixa_etaria','EVOLUCAO'])['casos']
                 .sum().unstack('EVOLUCAO').fillna(0))
    pivot_mfx.columns = [c if c not in ('1','2') else ('Curas' if c=='1' else 'Óbitos')
                         for c in pivot_mfx.columns]
    pivot_mfx = pivot_mfx.reset_index()
    pivot_mfx['total'] = pivot_mfx.get('Curas', 0) + pivot_mfx.get('Óbitos', 0)
    pivot_mfx['taxa_mort_fx'] = pivot_mfx.get('Óbitos', 0) / pivot_mfx['total'] * 100
    pivot_mfx['faixa_etaria'] = pd.Categorical(pivot_mfx['faixa_etaria'],
                                                 categories=ordem_fx2, ordered=True)
    pivot_mfx = pivot_mfx.sort_values(['ANO_DADOS','faixa_etaria'])

    anos_mfx = sorted(pivot_mfx['ANO_DADOS'].unique())
    cmap_mfx = plt.cm.get_cmap('tab10', len(anos_mfx))
    fig, ax = plt.subplots(figsize=(16, 7))
    for idx, ano in enumerate(anos_mfx):
        d = pivot_mfx[pivot_mfx['ANO_DADOS'] == ano]
        ax.plot(d['faixa_etaria'].astype(str), d['taxa_mort_fx'],
                marker='o', linewidth=2, label=str(int(ano)),
                color=cmap_mfx(idx), markersize=7)
    ax.set_title('Taxa de Mortalidade por Faixa Etária e Ano', fontsize=13, fontweight='bold')
    ax.set_xlabel('Faixa Etária'); ax.set_ylabel('Taxa de Mortalidade (%)')
    ax.tick_params(axis='x', rotation=30); ax.legend(title='Ano'); ax.grid(True, alpha=0.3)
    plt.tight_layout(); _save_fig(fig, "eda_15_mortalidade_faixa_etaria.png")

    print("Taxa de mortalidade nos grupos mais vulneráveis:")
    for ano in anos_mfx:
        d = pivot_mfx[(pivot_mfx['ANO_DADOS'] == ano) &
                      pivot_mfx['faixa_etaria'].isin(['60-69 anos','70+ anos'])]
        if len(d) > 0:
            tx_med = d['taxa_mort_fx'].mean()
            print(f"  {int(ano)} — 60+ anos: {tx_med:.2f}% (média das duas faixas)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Validação das métricas alvo

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9.1 Taxa de Mortalidade (SRAG estrita)
# MAGIC
# MAGIC Denominador: `EVOLUCAO IN ('1','2')`. Códigos `3` (óbito por outras causas) e `9` excluídos.

# COMMAND ----------

pd_mort_ano = (
    df.filter(F.col("EVOLUCAO").isin("1","2"))
    .groupBy("ANO_DADOS","EVOLUCAO").agg(F.count("*").alias("casos"))
    .orderBy("ANO_DADOS").limit(10_000).toPandas()
)

if len(pd_mort_ano) > 0:
    mort_pivot = (
        pd_mort_ano.pivot(index='ANO_DADOS', columns='EVOLUCAO', values='casos')
        .fillna(0).rename(columns={'1':'curas','2':'obitos'})
    )
    mort_pivot['total']            = mort_pivot['curas'] + mort_pivot['obitos']
    mort_pivot['taxa_mortalidade'] = mort_pivot['obitos'] / mort_pivot['total'] * 100

    total_geral  = int(mort_pivot['total'].sum())
    obitos_geral = int(mort_pivot['obitos'].sum())
    taxa_geral   = obitos_geral / total_geral * 100

    pd_agg = (pd_mort_ano
              .groupby(pd_mort_ano['EVOLUCAO'].map({'1':'Cura','2':'Óbito por SRAG'}))['casos']
              .sum().reset_index().rename(columns={'EVOLUCAO':'label'}))

    colors_m = [PALETTE['cura'] if l == 'Cura' else PALETTE['obito'] for l in pd_agg['label']]
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(pd_agg['label'], pd_agg['casos'], color=colors_m, alpha=0.75)
    ax.set_title('Desfechos SRAG — Total\n(EVOLUCAO 9 e 3 excluídos)',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Casos'); ax.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h,
                f'{int(h):,}', ha='center', va='bottom', fontsize=11)
    plt.tight_layout(); _save_fig(fig, "eda_16_desfechos_total.png")

    print(f"Mortalidade SRAG: {taxa_geral:.2f}%  ({obitos_geral:,} óbitos / {total_geral:,})")
    for ano in mort_pivot.index:
        print(f"  {int(ano)}: {mort_pivot.loc[ano,'taxa_mortalidade']:.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9.2 Taxa de Ocupação UTI
# MAGIC
# MAGIC Denominador: `HOSPITAL=1` e `UTI IN ('1','2')`. Código `9` e NULL excluídos.

# COMMAND ----------

pd_uti = (
    df.filter((F.col("HOSPITAL") == "1") & F.col("UTI").isin("1","2"))
    .groupBy("UTI").agg(F.count("*").alias("casos"))
    .limit(10_000).toPandas()
)

if len(pd_uti) > 0:
    total_h  = pd_uti['casos'].sum()
    uti_sim  = int(pd_uti.loc[pd_uti['UTI']=='1','casos'].values[0]) \
               if '1' in pd_uti['UTI'].values else 0
    taxa_uti = uti_sim / total_h * 100

    pd_uti['label']      = pd_uti['UTI'].map({'1':'UTI','2':'Enfermaria'})
    pd_uti['percentual'] = pd_uti['casos'] / total_h * 100

    colors_u = [PALETTE['uti'] if l == 'UTI' else PALETTE['enfermaria'] for l in pd_uti['label']]
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(pd_uti['label'], pd_uti['casos'], color=colors_u, alpha=0.75)
    ax.set_title('Internações — UTI vs Enfermaria\n(HOSPITAL=1, UTI 9/NULL excluídos)',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Casos'); ax.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h,
                f'{int(h):,}', ha='center', va='bottom', fontsize=11)
    plt.tight_layout(); _save_fig(fig, "eda_17_uti_vs_enfermaria.png")

    print(f"Taxa UTI: {taxa_uti:.2f}%  (UTI {uti_sim:,} / Total {total_h:,})")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9.3 Taxa de Vacinação
# MAGIC
# MAGIC Denominador: `VACINA IN ('1','2')`. `VACINA_COV` tratada separadamente na Silver/Gold.

# COMMAND ----------

pd_vac = (
    df.filter(F.col("VACINA").isin("1","2"))
    .groupBy("VACINA").agg(F.count("*").alias("casos"))
    .limit(10_000).toPandas()
)

if len(pd_vac) > 0:
    total_v   = pd_vac['casos'].sum()
    vacinados = int(pd_vac.loc[pd_vac['VACINA']=='1','casos'].values[0]) \
                if '1' in pd_vac['VACINA'].values else 0
    taxa_vac  = vacinados / total_v * 100

    pd_vac['label']      = pd_vac['VACINA'].map({'1':'Vacinado','2':'Não vacinado'})
    pd_vac['percentual'] = pd_vac['casos'] / total_v * 100

    colors_v = [PALETTE['vacinado'] if l == 'Vacinado' else PALETTE['nao_vacinado']
                for l in pd_vac['label']]
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(pd_vac['label'], pd_vac['casos'], color=colors_v, alpha=0.75)
    ax.set_title('Status de Vacinação (VACINA)\n(código 9 e NULL excluídos)',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Casos'); ax.grid(True, alpha=0.3, axis='y')
    for bar, pct in zip(bars, pd_vac['percentual']):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h,
                f'{int(h):,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=11)
    plt.tight_layout(); _save_fig(fig, "eda_18_status_vacinacao.png")

    print(f"Vacinação: {taxa_vac:.2f}%  (Vacinados {vacinados:,} / Total {total_v:,})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Associações categóricas (Cramér's V)
# MAGIC
# MAGIC Chi² via NumPy puro (sem SciPy). Código `9` e NULL excluídos das contingências.
# MAGIC Interpretação: < 0.10 negligível | 0.10–0.20 fraca | 0.20–0.40 moderada | > 0.40 forte.

# COMMAND ----------

def _cramers_v(cm: np.ndarray) -> float:
    """Cramér's V sem SciPy. Retorna 0–1."""
    n = cm.sum()
    if n == 0:
        return 0.0
    expected = cm.sum(axis=1, keepdims=True) @ cm.sum(axis=0, keepdims=True) / n
    chi2     = np.where(expected > 0, (cm - expected) ** 2 / expected, 0).sum()
    min_d    = min(cm.shape) - 1
    return float(np.sqrt(chi2 / (n * min_d))) if min_d > 0 else 0.0


def _contingencia(df_spark, v1: str, v2: str):
    pd_c = (
        df_spark
        .filter(F.col(v1).isNotNull() & (F.col(v1) != '9') &
                F.col(v2).isNotNull() & (F.col(v2) != '9'))
        .groupBy(v1, v2).count()
        .limit(10_000).toPandas()
    )
    if len(pd_c) == 0:
        return None
    return pd_c.pivot(index=v1, columns=v2, values='count').fillna(0).values


VARS_ASSOC = ['CS_SEXO','FEBRE','TOSSE','DISPNEIA','SATURACAO','HOSPITAL','UTI','EVOLUCAO']
vars_ok    = [v for v in VARS_ASSOC if resolve_col(df, [v]) is not None]
n_v        = len(vars_ok)
mat        = np.zeros((n_v, n_v))

print(f"Calculando Cramér's V para {n_v} variáveis...")
for i, v1 in enumerate(vars_ok):
    for j, v2 in enumerate(vars_ok):
        if i < j:
            ct = _contingencia(df, resolve_col(df, [v1]), resolve_col(df, [v2]))
            val = _cramers_v(ct) if ct is not None else 0.0
            mat[i, j] = mat[j, i] = val
        elif i == j:
            mat[i, j] = 1.0

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(mat, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, label="Cramér's V")
ax.set_xticks(range(n_v)); ax.set_xticklabels(vars_ok, rotation=45, ha='right')
ax.set_yticks(range(n_v)); ax.set_yticklabels(vars_ok)
for i in range(n_v):
    for j in range(n_v):
        ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center', fontsize=8,
                color='black' if mat[i,j] < 0.6 else 'white')
ax.set_title("Cramér's V — Associações Categóricas\n(código '9' e NULL excluídos)",
             fontsize=13, fontweight='bold')
plt.tight_layout(); _save_fig(fig, "eda_19_cramers_v.png")

fortes = sorted(
    [{'v1': vars_ok[i], 'v2': vars_ok[j], 'v': mat[i,j]}
     for i in range(n_v) for j in range(n_v) if i < j and mat[i,j] > 0.20],
    key=lambda x: x['v'], reverse=True
)
print("Associações > 0.20:")
for a in fortes:
    print(f"  {a['v1']} ↔ {a['v2']}: {a['v']:.3f} ({'FORTE' if a['v'] >= 0.40 else 'MODERADA'})")
if not fortes:
    print("  Nenhuma associação com V > 0.20")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Padrões de ausência e código "9"

# COMMAND ----------

CAMPOS_MISS = ['DT_SIN_PRI','DT_NOTIFIC','SEM_PRI','CS_SEXO','NU_IDADE_N','SG_UF',
               'FEBRE','TOSSE','DISPNEIA','SATURACAO','HOSPITAL','UTI','EVOLUCAO',
               'VACINA','VACINA_COV']
cols_miss = [c for c in (resolve_col(df, [f]) for f in CAMPOS_MISS) if c is not None]

null_row = df.agg(*[
    F.sum(F.when(F.col(c).isNull() | (F.col(c).cast("string") == ''), 1).otherwise(0)).alias(c)
    for c in cols_miss
]).collect()[0].asDict()

pd_miss = pd.DataFrame([
    {'campo': c, 'missing_count': null_row[c], 'missing_pct': null_row[c] / total_rows * 100}
    for c in cols_miss
]).sort_values('missing_pct', ascending=False)

fig, ax = plt.subplots(figsize=(12, 8))
colors_miss = [PALETTE['secondary'] if x > 40 else PALETTE['neutral'] if x > 20
               else PALETTE['vacinado'] for x in pd_miss['missing_pct']]
ax.barh(pd_miss['campo'], pd_miss['missing_pct'], color=colors_miss, alpha=0.75)
ax.set_title('Ausência por Campo (NULL/vazio)', fontsize=13, fontweight='bold')
ax.set_xlabel('% Ausente')
ax.axvline(x=20, color='orange', linestyle='--', linewidth=1, label='20%')
ax.axvline(x=40, color='red',    linestyle='--', linewidth=1, label='40%')
ax.legend(); ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout(); _save_fig(fig, "eda_20_missingness.png")

print("Top 10 ausências:")
for _, row in pd_miss.head(10).iterrows():
    print(f"  {row['campo']}: {row['missing_pct']:.1f}% ({row['missing_count']:,})")

# COMMAND ----------

CAMPOS_C9 = ['CS_RACA','FEBRE','TOSSE','DISPNEIA','HOSPITAL','UTI','EVOLUCAO','VACINA']
cols_c9   = [c for c in (resolve_col(df, [f]) for f in CAMPOS_C9) if c is not None]

c9_row = df.agg(*[
    F.sum(F.when(F.col(c).cast("string") == '9', 1).otherwise(0)).alias(c) for c in cols_c9
]).collect()[0].asDict()

pd_c9 = pd.DataFrame([
    {'campo': c, 'code9_count': c9_row[c], 'code9_pct': c9_row[c] / total_rows * 100}
    for c in cols_c9
]).sort_values('code9_pct', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
colors_c9 = [PALETTE['secondary'] if x > 30 else PALETTE['neutral'] if x > 15
             else PALETTE['primary'] for x in pd_c9['code9_pct']]
ax.barh(pd_c9['campo'], pd_c9['code9_pct'], color=colors_c9, alpha=0.75)
ax.set_title('Código "9" (Ignorado) por Campo', fontsize=13, fontweight='bold')
ax.set_xlabel('% Código "9"')
ax.axvline(x=15, color='orange', linestyle='--', linewidth=1, label='15%')
ax.axvline(x=30, color='red',    linestyle='--', linewidth=1, label='30%')
ax.legend(); ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout(); _save_fig(fig, "eda_21_code9.png")

# Código '9' = 'Ignorado' no DATASUS — não equivale a NULL e deve ser tratado separadamente.
print("Campos com código '9' > 10%:")
for _, row in pd_c9[pd_c9['code9_pct'] > 10].iterrows():
    print(f"  {row['campo']}: {row['code9_pct']:.1f}% ({row['code9_count']:,})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Feature selection para a Silver

# COMMAND ----------

features = {
    'Essenciais':   ['NU_NOTIFIC','DT_SIN_PRI','DT_NOTIFIC','SEM_PRI','ANO_DADOS'],
    'Demográficas': ['SG_UF','CO_MUN_RES','CS_SEXO','NU_IDADE_N','TP_IDADE'],
    'Clínicas':     ['FEBRE','TOSSE','DISPNEIA','SATURACAO','HOSPITAL','UTI',
                     'DT_INTERNA','EVOLUCAO','DT_EVOLUCA'],
    'Vacinação':    ['VACINA','VACINA_COV'],
    'Opcionais':    ['CS_RACA','CS_ESCOL_N','CLASSI_FIN'],
}
total_feat = 0
for cat, campos in features.items():
    print(f"\n{cat}:")
    for c in campos:
        real = resolve_col(df, [c])
        if real:
            total_feat += 1
        label = f"OK → {real}" if real else "AUSENTE"
        print(f"  [{label:20}] {c}")
print(f"\nTotal features encontradas: {total_feat}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Persistência das séries auxiliares
# MAGIC
# MAGIC Modo `append` + `run_id`. Para isolar a execução: `WHERE run_id = '<RUN_ID>'`.
# MAGIC `eda_mortalidade_mensal`: `groupBy` só por `ano_mes` → 1 linha por mês.
# MAGIC `fallback_pct` = % de registros onde `dt_evolucao` era nulo (substituído por `dt_sintomas`).

# COMMAND ----------

# --- eda_serie_diaria_90d (nova tabela) ---
if len(pd_diaria) > 0:
    (spark.createDataFrame(pd_diaria)
     .withColumn("run_id",    F.lit(RUN_ID))
     .withColumn("gerado_em", F.current_timestamp())
     .write.mode("append").option("mergeSchema","true")
     .saveAsTable(TABLE_SERIE_DIARIA))
    print(f"Gravado: {TABLE_SERIE_DIARIA}")

# COMMAND ----------

# --- eda_series_mensal ---
if len(pd_mensal) > 0:
    (spark.createDataFrame(
        pd_mensal[['ano_mes','casos','crescimento_pct']]
        .rename(columns={'crescimento_pct': 'crescimento_pct_12m'}))
     .withColumn("run_id",        F.lit(RUN_ID))
     .withColumn("janela",        F.lit("ultimos_12_meses"))
     .withColumn("base_temporal", F.lit("DT_SIN_PRI"))
     .withColumn("gerado_em",     F.current_timestamp())
     .write.mode("append").option("mergeSchema","true")
     .saveAsTable(TABLE_SERIES_MENSAL))
    print(f"Gravado: {TABLE_SERIES_MENSAL}")

# COMMAND ----------

# --- eda_mortalidade_mensal ---
(df.filter(F.col("EVOLUCAO").isin("1","2"))
 .withColumn("dt_ref",      F.coalesce(F.col("dt_evolucao"), F.col("dt_sintomas")))
 .withColumn("is_fallback", F.when(F.col("dt_evolucao").isNull(), 1).otherwise(0))
 .withColumn("ano_mes",
     F.concat(F.year("dt_ref"), F.lit("-"), F.lpad(F.month("dt_ref"), 2, "0")))
 .filter(F.col("ano_mes").isNotNull())
 .groupBy("ano_mes")
 .agg(
     F.count("*").alias("total_desfechos"),
     F.sum(F.when(F.col("EVOLUCAO") == "2", 1).otherwise(0)).alias("obitos_srag"),
     F.round(F.sum(F.col("is_fallback")) / F.count("*") * 100, 2).alias("fallback_pct"),
 )
 .withColumn("taxa_mortalidade_pct",
     F.round(F.col("obitos_srag") / F.col("total_desfechos") * 100, 4))
 .withColumn("filtro_aplicado", F.lit("EVOLUCAO IN (1,2)"))
 .withColumn("excluidos",       F.lit("EVOLUCAO IN (3,9,NULL)"))
 .withColumn("nota_dt_ref",     F.lit("dt_evolucao; fallback dt_sintomas quando nulo"))
 .withColumn("run_id",          F.lit(RUN_ID))
 .withColumn("gerado_em",       F.current_timestamp())
 .orderBy("ano_mes")
 .write.mode("append").option("mergeSchema","true")
 .saveAsTable(TABLE_MORTALIDADE_MENSAL))
print(f"Gravado: {TABLE_MORTALIDADE_MENSAL}")

# COMMAND ----------

# --- eda_vacinacao_mensal ---
(df.filter(F.col("VACINA").isin("1","2"))
 .withColumn("ano_mes",
     F.concat(F.year("dt_sintomas"), F.lit("-"),
              F.lpad(F.month("dt_sintomas"), 2, "0")))
 .filter(F.col("ano_mes").isNotNull())
 .groupBy("ano_mes")
 .agg(
     F.count("*").alias("total_com_info"),
     F.sum(F.when(F.col("VACINA") == "1", 1).otherwise(0)).alias("vacinados"),
 )
 .withColumn("taxa_vacinacao_pct",
     F.round(F.col("vacinados") / F.col("total_com_info") * 100, 4))
 .withColumn("filtro_aplicado", F.lit("VACINA IN (1,2)"))
 .withColumn("excluidos",       F.lit("VACINA IN (9,NULL)"))
 .withColumn("nota",            F.lit("VACINA_COV tratada separadamente na Silver/Gold"))
 .withColumn("run_id",          F.lit(RUN_ID))
 .withColumn("gerado_em",       F.current_timestamp())
 .orderBy("ano_mes")
 .write.mode("append").option("mergeSchema","true")
 .saveAsTable(TABLE_VACINACAO_MENSAL))
print(f"Gravado: {TABLE_VACINACAO_MENSAL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Resumo da execução

# COMMAND ----------

print("-" * 80)
print("ANÁLISE EXPLORATÓRIA ENCERRADA")
print("-" * 80)
print(f"  Run ID     : {RUN_ID}")
print(f"  Registros  : {total_rows:,}  |  Colunas: {total_cols}")
print(f"  Janela 12m : {data_inicio_12m} → {max_dt}")
print(f"\nTabelas gravadas:")
print(f"  {TABLE_SERIE_DIARIA}")
print(f"  {TABLE_SERIES_MENSAL}")
print(f"  {TABLE_MORTALIDADE_MENSAL}")
print(f"  {TABLE_VACINACAO_MENSAL}")
print("-" * 80)
print("Próximo: 04_Silver_Transformation.py")
