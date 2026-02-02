# Databricks notebook source
# MAGIC %md
# MAGIC # Análise Exploratória de Dados - SRAG
# MAGIC 
# MAGIC **Projeto**: Sistema RAG para Monitoramento Epidemiológico
# MAGIC 
# MAGIC **Objetivo**: Explorar dados SRAG da camada Bronze para embasar decisões da camada Silver
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC ## Escopo da Análise
# MAGIC 
# MAGIC ### Conteúdo:
# MAGIC - Análise temporal (séries diárias, mensais, sazonalidade)
# MAGIC - Análise demográfica (UF, sexo, faixa etária)
# MAGIC - Validação das 4 métricas alvo (Mortalidade, UTI, Vacinação, Crescimento)
# MAGIC - Análise de associações categóricas (Cramér's V)
# MAGIC - Padrões de missing e código "9"
# MAGIC - Feature selection para camada Silver
# MAGIC 
# MAGIC ### Métricas Alvo:
# MAGIC 1. **Taxa de Mortalidade**: (EVOLUCAO='2') / (EVOLUCAO IN ('1','2')) × 100
# MAGIC 2. **Taxa UTI**: (UTI='1') / (HOSPITAL='1' AND UTI IN ('1','2')) × 100
# MAGIC 3. **Taxa Vacinação**: (VACINA='1') / (VACINA IN ('1','2')) × 100
# MAGIC 4. **Taxa Crescimento**: Variação % mensal de casos

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup e Configuração

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import *
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10

print("=" * 80)
print("ANÁLISE EXPLORATÓRIA DE DADOS - SRAG")
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
TABLE_BRONZE = f"{CATALOG}.{SCHEMA_BRONZE}.bronze_srag_raw"
VOLUME_NAME = "data_srag"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA_BRONZE}/{VOLUME_NAME}"

SAMPLE_SIZE = None
ANALYSIS_ID = datetime.now().strftime('%Y%m%d_%H%M%S')

print("CONFIGURAÇÃO:")
print(f"  • Fonte: {TABLE_BRONZE}")
print(f"  • Sample: {'Completo' if SAMPLE_SIZE is None else f'{SAMPLE_SIZE*100}%'}")
print(f"  • Analysis ID: {ANALYSIS_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Carregamento dos Dados

# COMMAND ----------

print("\nCarregando dados da Bronze...")

df_raw = spark.table(TABLE_BRONZE)

if SAMPLE_SIZE is not None:
    df_raw = df_raw.sample(fraction=SAMPLE_SIZE, seed=42)
    print(f"Usando amostra de {SAMPLE_SIZE*100}%")

total_rows = df_raw.count()
total_cols = len(df_raw.columns)

print(f"\nDados carregados:")
print(f"  • Registros: {total_rows:,}")
print(f"  • Colunas: {total_cols}")

print(f"\nDistribuição por ano:")
df_raw.groupBy("ANO_DADOS").count().orderBy("ANO_DADOS").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Preparação de Campos Essenciais

# COMMAND ----------

print("\nPreparando campos para análise...")

df_analysis = df_raw \
.withColumn(
    "dt_sintomas",
    F.coalesce(
        F.expr("try_to_date(DT_SIN_PRI, 'dd/MM/yyyy')"),
        F.expr("try_to_date(DT_SIN_PRI, 'yyyy-MM-dd')")
    )
) \
.withColumn(
    "dt_notificacao",
    F.coalesce(
        F.expr("try_to_date(DT_NOTIFIC, 'dd/MM/yyyy')"),
        F.expr("try_to_date(DT_NOTIFIC, 'yyyy-MM-dd')")
    )
) \
.withColumn(
    "dt_internacao",
    F.coalesce(
        F.expr("try_to_date(DT_INTERNA, 'dd/MM/yyyy')"),
        F.expr("try_to_date(DT_INTERNA, 'yyyy-MM-dd')")
    )
) \
.withColumn(
    "dt_evolucao",
    F.coalesce(
        F.expr("try_to_date(DT_EVOLUCA, 'dd/MM/yyyy')"),
        F.expr("try_to_date(DT_EVOLUCA, 'yyyy-MM-dd')")
    )
)

df_analysis = df_analysis.withColumn(
    "CS_SEXO", 
    F.col("CS_SEXO").cast("string")
)

df_analysis = df_analysis.withColumn(
    "ano_mes",
    F.concat(
        F.year("dt_sintomas"),
        F.lit("-"),
        F.lpad(F.month("dt_sintomas"), 2, "0")
    )
)

df_analysis = df_analysis.withColumn(
    "idade_anos",
    F.when(
        (F.col("TP_IDADE") == "3") & F.col("NU_IDADE_N").isNotNull(),
        F.col("NU_IDADE_N").cast("int")
    ).otherwise(None)
)

df_analysis = df_analysis.withColumn(
    "faixa_etaria",
    F.when(F.col("idade_anos") < 18, "0-17")
    .when(F.col("idade_anos") < 30, "18-29")
    .when(F.col("idade_anos") < 40, "30-39")
    .when(F.col("idade_anos") < 50, "40-49")
    .when(F.col("idade_anos") < 60, "50-59")
    .when(F.col("idade_anos") < 70, "60-69")
    .when(F.col("idade_anos") >= 70, "70+")
    .otherwise("Desconhecido")
)

print("Campos preparados:")
print("  • Datas convertidas para DATE (múltiplos formatos - try_to_date)")
print("  • CS_SEXO padronizado como string")
print("  • Campo ano_mes criado")
print("  • Idade convertida para numérico")
print("  • Faixas etárias definidas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. ANÁLISE TEMPORAL

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1 Série Temporal Diária (Últimos 90 Dias)

# COMMAND ----------

print("\nANÁLISE TEMPORAL - Série Diária")
print("=" * 80)

data_corte = F.date_sub(F.current_date(), 90)

df_serie_diaria = (
    df_analysis
    .filter(
        F.col("dt_sintomas").isNotNull() &
        (F.col("dt_sintomas") >= data_corte)
    )
    .groupBy("dt_sintomas")
    .agg(F.count("*").alias("casos"))
    .orderBy("dt_sintomas")
)

pd_diaria = df_serie_diaria.toPandas()

if len(pd_diaria) > 0:
    plt.figure(figsize=(16, 6))
    plt.plot(pd_diaria['dt_sintomas'], pd_diaria['casos'], marker='o', linewidth=2)
    plt.title('Série Temporal de Casos SRAG - Últimos 90 Dias', fontsize=14, fontweight='bold')
    plt.xlabel('Data de Primeiros Sintomas', fontsize=12)
    plt.ylabel('Número de Casos', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"Período analisado: {pd_diaria['dt_sintomas'].min()} a {pd_diaria['dt_sintomas'].max()}")
    print(f"Total de casos: {pd_diaria['casos'].sum():,}")
    print(f"Média diária: {pd_diaria['casos'].mean():.1f} casos/dia")
    print(f"Máximo: {pd_diaria['casos'].max()} casos em {pd_diaria.loc[pd_diaria['casos'].idxmax(), 'dt_sintomas']}")
else:
    print("Sem dados nos últimos 90 dias")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 Série Temporal Mensal

# COMMAND ----------

print("\nANÁLISE TEMPORAL - Série Mensal")
print("=" * 80)

df_serie_mensal = df_analysis.filter(
    F.col("ano_mes").isNotNull()
).groupBy("ano_mes").agg(
    F.count("*").alias("casos")
).orderBy("ano_mes")

pd_mensal = df_serie_mensal.toPandas()

if len(pd_mensal) > 0:
    pd_mensal['crescimento_pct'] = pd_mensal['casos'].pct_change() * 100
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    ax1.bar(pd_mensal['ano_mes'], pd_mensal['casos'], color='steelblue', alpha=0.7)
    ax1.set_title('Casos SRAG por Mês', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Ano-Mês', fontsize=12)
    ax1.set_ylabel('Número de Casos', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    colors = ['green' if x >= 0 else 'red' for x in pd_mensal['crescimento_pct'].fillna(0)]
    ax2.bar(pd_mensal['ano_mes'], pd_mensal['crescimento_pct'].fillna(0), color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_title('Taxa de Crescimento Mensal (%)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Ano-Mês', fontsize=12)
    ax2.set_ylabel('Variação %', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    print("\nESTATÍSTICAS MENSAIS:")
    print(f"  • Média mensal: {pd_mensal['casos'].mean():,.0f} casos")
    print(f"  • Mediana: {pd_mensal['casos'].median():,.0f} casos")
    print(f"  • Mês com mais casos: {pd_mensal.loc[pd_mensal['casos'].idxmax(), 'ano_mes']} ({pd_mensal['casos'].max():,} casos)")
    print(f"  • Mês com menos casos: {pd_mensal.loc[pd_mensal['casos'].idxmin(), 'ano_mes']} ({pd_mensal['casos'].min():,} casos)")
    
    print("\nCRESCIMENTO:")
    print(f"  • Média de crescimento: {pd_mensal['crescimento_pct'].mean():.2f}%")
    print(f"  • Maior crescimento: {pd_mensal['crescimento_pct'].max():.2f}%")
    print(f"  • Maior queda: {pd_mensal['crescimento_pct'].min():.2f}%")
else:
    print("Sem dados mensais disponíveis")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3 Sazonalidade por Semana Epidemiológica

# COMMAND ----------

print("\nANÁLISE TEMPORAL - Semana Epidemiológica")
print("=" * 80)

df_semanal = df_analysis.filter(
    F.col("SEM_PRI").isNotNull()
).groupBy("SEM_PRI", "ANO_DADOS").agg(
    F.count("*").alias("casos")
).orderBy("ANO_DADOS", "SEM_PRI")

pd_semanal = df_semanal.toPandas()

if len(pd_semanal) > 0:
    plt.figure(figsize=(16, 6))
    
    for ano in sorted(pd_semanal['ANO_DADOS'].unique()):
        data_ano = pd_semanal[pd_semanal['ANO_DADOS'] == ano]
        plt.plot(data_ano['SEM_PRI'].astype(int), data_ano['casos'], 
                marker='o', label=f'Ano {ano}', linewidth=2, markersize=4)
    
    plt.title('Sazonalidade - Casos por Semana Epidemiológica', fontsize=14, fontweight='bold')
    plt.xlabel('Semana Epidemiológica', fontsize=12)
    plt.ylabel('Número de Casos', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\nPADRÃO SAZONAL:")
    semanas_pico = pd_semanal.groupby('SEM_PRI')['casos'].mean().nlargest(5)
    print(f"  • Semanas com mais casos (média): {', '.join([f'SE{int(s)}' for s in semanas_pico.index])}")
else:
    print("Sem dados de semana epidemiológica")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.4 Tendência por Ano

# COMMAND ----------

print("\nANÁLISE TEMPORAL - Comparação Anual")
print("=" * 80)

df_anual = df_analysis.groupBy("ANO_DADOS").agg(
    F.count("*").alias("casos")
).orderBy("ANO_DADOS")

pd_anual = df_anual.toPandas()

if len(pd_anual) > 0:
    plt.figure(figsize=(10, 6))
    bars = plt.bar(pd_anual['ANO_DADOS'], pd_anual['casos'], color='coral', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.title('Distribuição de Casos SRAG por Ano', fontsize=14, fontweight='bold')
    plt.xlabel('Ano', fontsize=12)
    plt.ylabel('Número de Casos', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    
    pd_anual['variacao_pct'] = pd_anual['casos'].pct_change() * 100
    
    print("\nEVOLUÇÃO ANUAL:")
    for idx, row in pd_anual.iterrows():
        if pd.notna(row['variacao_pct']):
            print(f"  • {row['ANO_DADOS']}: {row['casos']:,} casos ({row['variacao_pct']:+.1f}%)")
        else:
            print(f"  • {row['ANO_DADOS']}: {row['casos']:,} casos")
else:
    print("Sem dados anuais")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. ANÁLISE DEMOGRÁFICA

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.1 Distribuição por UF

# COMMAND ----------

print("\nANÁLISE DEMOGRÁFICA - Distribuição Geográfica (UF)")
print("=" * 80)

df_uf = df_analysis.filter(
    F.col("SG_UF").isNotNull()
).groupBy("SG_UF").agg(
    F.count("*").alias("casos")
).orderBy(F.desc("casos")).limit(15)

pd_uf = df_uf.toPandas()

if len(pd_uf) > 0:
    pd_uf['percentual'] = (pd_uf['casos'] / pd_uf['casos'].sum()) * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.barh(pd_uf['SG_UF'], pd_uf['casos'], color='teal', alpha=0.7)
    ax1.set_title('Top 15 UFs - Número de Casos', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Número de Casos', fontsize=12)
    ax1.set_ylabel('UF', fontsize=12)
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')
    
    for i, (casos, uf) in enumerate(zip(pd_uf['casos'], pd_uf['SG_UF'])):
        ax1.text(casos, i, f' {casos:,}', va='center', fontsize=9)
    
    colors_pie = plt.cm.Set3(range(len(pd_uf)))
    ax2.pie(pd_uf['percentual'], labels=pd_uf['SG_UF'], autopct='%1.1f%%',
            colors=colors_pie, startangle=90)
    ax2.set_title('Distribuição Percentual - Top 15 UFs', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("\nTOP 5 UFs:")
    for idx, row in pd_uf.head(5).iterrows():
        print(f"  • {row['SG_UF']}: {row['casos']:,} casos ({row['percentual']:.1f}%)")
    
    concentracao_top5 = pd_uf.head(5)['percentual'].sum()
    print(f"\nTop 5 UFs concentram {concentracao_top5:.1f}% dos casos")
else:
    print("Sem dados de UF")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.2 Distribuição por Sexo

# COMMAND ----------

print("\nANÁLISE DEMOGRÁFICA - Distribuição por Sexo")
print("=" * 80)

df_sexo = df_analysis.filter(
    F.col("CS_SEXO").isin("1", "2")
).groupBy("CS_SEXO").agg(
    F.count("*").alias("casos")
)

pd_sexo = df_sexo.toPandas()

if len(pd_sexo) > 0:
    pd_sexo['sexo_label'] = pd_sexo['CS_SEXO'].map({'1': 'Masculino', '2': 'Feminino'})
    pd_sexo['percentual'] = (pd_sexo['casos'] / pd_sexo['casos'].sum()) * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors_sex = ['#3498db', '#e74c3c']
    bars = ax1.bar(pd_sexo['sexo_label'], pd_sexo['casos'], color=colors_sex, alpha=0.7)
    ax1.set_title('Distribuição de Casos por Sexo', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Número de Casos', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}\n({height/pd_sexo["casos"].sum()*100:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.pie(pd_sexo['percentual'], labels=pd_sexo['sexo_label'], autopct='%1.1f%%',
            colors=colors_sex, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax2.set_title('Proporção por Sexo', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("\nDISTRIBUIÇÃO POR SEXO:")
    for idx, row in pd_sexo.iterrows():
        print(f"  • {row['sexo_label']}: {row['casos']:,} casos ({row['percentual']:.1f}%)")
else:
    print("Sem dados de sexo válidos")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.3 Distribuição por Faixa Etária

# COMMAND ----------

print("\nANÁLISE DEMOGRÁFICA - Distribuição por Faixa Etária")
print("=" * 80)

df_idade = df_analysis.filter(
    (F.col("faixa_etaria") != "Desconhecido") &
    F.col("faixa_etaria").isNotNull()
).groupBy("faixa_etaria").agg(
    F.count("*").alias("casos")
)

pd_idade = df_idade.toPandas()

if len(pd_idade) > 0:
    ordem_faixas = ["0-17", "18-29", "30-39", "40-49", "50-59", "60-69", "70+"]
    pd_idade['faixa_etaria'] = pd.Categorical(pd_idade['faixa_etaria'], 
                                               categories=ordem_faixas, 
                                               ordered=True)
    pd_idade = pd_idade.sort_values('faixa_etaria')
    pd_idade['percentual'] = (pd_idade['casos'] / pd_idade['casos'].sum()) * 100
    
    plt.figure(figsize=(14, 6))
    bars = plt.bar(pd_idade['faixa_etaria'].astype(str), pd_idade['casos'], 
                   color='mediumseagreen', alpha=0.7)
    
    plt.title('Distribuição de Casos por Faixa Etária', fontsize=14, fontweight='bold')
    plt.xlabel('Faixa Etária (anos)', fontsize=12)
    plt.ylabel('Número de Casos', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    for bar, pct in zip(bars, pd_idade['percentual']):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    print("\nDISTRIBUIÇÃO POR IDADE:")
    for idx, row in pd_idade.iterrows():
        print(f"  • {row['faixa_etaria']}: {row['casos']:,} casos ({row['percentual']:.1f}%)")
    
    casos_60_mais = pd_idade[pd_idade['faixa_etaria'].astype(str).isin(['60-69', '70+'])]['casos'].sum()
    pct_60_mais = (casos_60_mais / pd_idade['casos'].sum()) * 100
    print(f"\nGrupo 60+: {casos_60_mais:,} casos ({pct_60_mais:.1f}%)")
else:
    print("Sem dados de idade válidos")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. VALIDAÇÃO DAS MÉTRICAS ALVO

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.1 Taxa de Mortalidade

# COMMAND ----------

print("\nMÉTRICA 1: TAXA DE MORTALIDADE")
print("=" * 80)

df_mortalidade = df_analysis.filter(
    F.col("EVOLUCAO").isin("1", "2")
).groupBy("EVOLUCAO").agg(
    F.count("*").alias("casos")
)

pd_mort = df_mortalidade.toPandas()

if len(pd_mort) > 0:
    total_desfecho = pd_mort['casos'].sum()
    obitos = pd_mort[pd_mort['EVOLUCAO'] == '2']['casos'].values[0] if '2' in pd_mort['EVOLUCAO'].values else 0
    taxa_mortalidade = (obitos / total_desfecho) * 100
    
    pd_mort['label'] = pd_mort['EVOLUCAO'].map({'1': 'Cura', '2': 'Óbito'})
    pd_mort['percentual'] = (pd_mort['casos'] / total_desfecho) * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors_mort = ['#2ecc71', '#e74c3c']
    bars = ax1.bar(pd_mort['label'], pd_mort['casos'], color=colors_mort, alpha=0.7)
    ax1.set_title('Distribuição de Desfechos Clínicos', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Número de Casos', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.pie(pd_mort['percentual'], labels=pd_mort['label'], autopct='%1.1f%%',
            colors=colors_mort, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax2.set_title('Proporção de Desfechos', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nTAXA DE MORTALIDADE: {taxa_mortalidade:.2f}%")
    print(f"  • Total com desfecho conhecido: {total_desfecho:,}")
    print(f"  • Óbitos: {obitos:,}")
    print(f"  • Curas: {total_desfecho - obitos:,}")
    
    print("\nTAXA DE MORTALIDADE POR ANO:")
    df_mort_ano = df_analysis.filter(
        F.col("EVOLUCAO").isin("1", "2")
    ).groupBy("ANO_DADOS", "EVOLUCAO").agg(
        F.count("*").alias("casos")
    ).orderBy("ANO_DADOS")
    
    pd_mort_ano = df_mort_ano.toPandas()
    mort_pivot = pd_mort_ano.pivot(index='ANO_DADOS', columns='EVOLUCAO', values='casos').fillna(0)
    
    if '2' in mort_pivot.columns and '1' in mort_pivot.columns:
        mort_pivot['taxa_mortalidade'] = (mort_pivot['2'] / (mort_pivot['1'] + mort_pivot['2'])) * 100
        
        for ano in mort_pivot.index:
            print(f"  • {ano}: {mort_pivot.loc[ano, 'taxa_mortalidade']:.2f}%")
else:
    print("Dados insuficientes para calcular taxa de mortalidade")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.2 Taxa de Ocupação UTI

# COMMAND ----------

print("\nMÉTRICA 2: TAXA DE OCUPAÇÃO UTI")
print("=" * 80)

df_uti = df_analysis.filter(
    (F.col("HOSPITAL") == "1") &
    F.col("UTI").isin("1", "2")
).groupBy("UTI").agg(
    F.count("*").alias("casos")
)

pd_uti = df_uti.toPandas()

if len(pd_uti) > 0:
    total_hospitalizados = pd_uti['casos'].sum()
    casos_uti = pd_uti[pd_uti['UTI'] == '1']['casos'].values[0] if '1' in pd_uti['UTI'].values else 0
    taxa_uti = (casos_uti / total_hospitalizados) * 100
    
    pd_uti['label'] = pd_uti['UTI'].map({'1': 'Sim (UTI)', '2': 'Não (Enfermaria)'})
    pd_uti['percentual'] = (pd_uti['casos'] / total_hospitalizados) * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors_uti = ['#e67e22', '#3498db']
    bars = ax1.bar(pd_uti['label'], pd_uti['casos'], color=colors_uti, alpha=0.7)
    ax1.set_title('Distribuição de Pacientes Hospitalizados', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Número de Casos', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.pie(pd_uti['percentual'], labels=pd_uti['label'], autopct='%1.1f%%',
            colors=colors_uti, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax2.set_title('Proporção UTI vs Enfermaria', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nTAXA DE OCUPAÇÃO UTI: {taxa_uti:.2f}%")
    print(f"  • Total hospitalizados: {total_hospitalizados:,}")
    print(f"  • Casos UTI: {casos_uti:,}")
    print(f"  • Casos Enfermaria: {total_hospitalizados - casos_uti:,}")
else:
    print("Dados insuficientes para calcular taxa UTI")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.3 Taxa de Vacinação

# COMMAND ----------

print("\nMÉTRICA 3: TAXA DE VACINAÇÃO")
print("=" * 80)

df_vacina = df_analysis.filter(
    F.col("VACINA").isin("1", "2")
).groupBy("VACINA").agg(
    F.count("*").alias("casos")
)

pd_vacina = df_vacina.toPandas()

if len(pd_vacina) > 0:
    total_informado = pd_vacina['casos'].sum()
    casos_vacinados = pd_vacina[pd_vacina['VACINA'] == '1']['casos'].values[0] if '1' in pd_vacina['VACINA'].values else 0
    taxa_vacinacao = (casos_vacinados / total_informado) * 100
    
    pd_vacina['label'] = pd_vacina['VACINA'].map({'1': 'Sim (Vacinado)', '2': 'Não (Não Vacinado)'})
    pd_vacina['percentual'] = (pd_vacina['casos'] / total_informado) * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors_vac = ['#27ae60', '#c0392b']
    bars = ax1.bar(pd_vacina['label'], pd_vacina['casos'], color=colors_vac, alpha=0.7)
    ax1.set_title('Distribuição por Status de Vacinação', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Número de Casos', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.pie(pd_vacina['percentual'], labels=pd_vacina['label'], autopct='%1.1f%%',
            colors=colors_vac, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax2.set_title('Proporção de Vacinação', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nTAXA DE VACINAÇÃO: {taxa_vacinacao:.2f}%")
    print(f"  • Total com informação: {total_informado:,}")
    print(f"  • Vacinados: {casos_vacinados:,}")
    print(f"  • Não vacinados: {total_informado - casos_vacinados:,}")
else:
    print("Dados insuficientes para calcular taxa de vacinação")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. ANÁLISE DE ASSOCIAÇÕES CATEGÓRICAS

# COMMAND ----------

def cramers_v(confusion_matrix):
    """
    Calcula Cramér's V para medir associação entre variáveis categóricas.
    
    Retorna valor entre 0 (sem associação) e 1 (associação perfeita).
    
    Interpretação:
    - 0.00 - 0.10: Associação negligível
    - 0.10 - 0.20: Associação fraca
    - 0.20 - 0.40: Associação moderada
    - 0.40+: Associação forte
    """
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    min_dim = min(confusion_matrix.shape) - 1
    
    if min_dim == 0:
        return 0.0
    
    cramers_v_value = np.sqrt(chi2 / (n * min_dim))
    return cramers_v_value


def get_contingency_table(df_spark, var1, var2, exclude_9=True):
    """Cria tabela de contingência entre duas variáveis."""
    if exclude_9:
        df_filtered = df_spark.filter(
            (F.col(var1) != "9") & (F.col(var2) != "9") &
            F.col(var1).isNotNull() & F.col(var2).isNotNull()
        )
    else:
        df_filtered = df_spark.filter(
            F.col(var1).isNotNull() & F.col(var2).isNotNull()
        )
    
    df_cross = df_filtered.groupBy(var1, var2).count()
    pd_cross = df_cross.toPandas()
    
    if len(pd_cross) == 0:
        return None
    
    contingency = pd_cross.pivot(index=var1, columns=var2, values='count').fillna(0)
    return contingency

print("Funções de associação definidas")

# COMMAND ----------

print("\nANÁLISE DE ASSOCIAÇÕES - Cramér's V")
print("=" * 80)

variaveis_categoricas = ['CS_SEXO', 'FEBRE', 'TOSSE', 'DISPNEIA', 'SATURACAO', 
                         'HOSPITAL', 'UTI', 'EVOLUCAO']

vars_disponiveis = [v for v in variaveis_categoricas if v in df_analysis.columns]

print(f"Calculando associações para {len(vars_disponiveis)} variáveis")

n_vars = len(vars_disponiveis)
matriz_cramers = np.zeros((n_vars, n_vars))

for i, var1 in enumerate(vars_disponiveis):
    for j, var2 in enumerate(vars_disponiveis):
        if i < j:
            contingency = get_contingency_table(df_analysis, var1, var2, exclude_9=True)
            
            if contingency is not None and contingency.size > 0:
                v = cramers_v(contingency.values)
                matriz_cramers[i, j] = v
                matriz_cramers[j, i] = v
            else:
                matriz_cramers[i, j] = 0
                matriz_cramers[j, i] = 0
        elif i == j:
            matriz_cramers[i, j] = 1.0

plt.figure(figsize=(12, 10))

sns.heatmap(matriz_cramers, 
            annot=True, 
            fmt='.3f',
            cmap='YlOrRd',
            xticklabels=vars_disponiveis,
            yticklabels=vars_disponiveis,
            cbar_kws={'label': "Cramér's V"},
            vmin=0, vmax=1)

plt.title("Matriz de Associações - Cramér's V\n(excluindo código '9')", 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('')
plt.ylabel('')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

print("\nASSOCIAÇÕES FORTES (V > 0.20):")
associacoes_fortes = []

for i, var1 in enumerate(vars_disponiveis):
    for j, var2 in enumerate(vars_disponiveis):
        if i < j and matriz_cramers[i, j] > 0.20:
            associacoes_fortes.append({
                'var1': var1,
                'var2': var2,
                'cramers_v': matriz_cramers[i, j]
            })

associacoes_fortes = sorted(associacoes_fortes, key=lambda x: x['cramers_v'], reverse=True)

if len(associacoes_fortes) > 0:
    for assoc in associacoes_fortes:
        interpretacao = "FORTE" if assoc['cramers_v'] >= 0.40 else "MODERADA"
        print(f"  • {assoc['var1']} ↔ {assoc['var2']}: {assoc['cramers_v']:.3f} ({interpretacao})")
else:
    print("  Nenhuma associação forte encontrada (V > 0.20)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. PADRÕES DE MISSING E CÓDIGO "9"

# COMMAND ----------

print("\nANÁLISE DE MISSING VALUES")
print("=" * 80)

campos_analisar = [
    'DT_SIN_PRI', 'DT_NOTIFIC', 'SEM_PRI',
    'CS_SEXO', 'NU_IDADE_N', 'SG_UF',
    'FEBRE', 'TOSSE', 'DISPNEIA', 'SATURACAO',
    'HOSPITAL', 'UTI', 'EVOLUCAO',
    'VACINA', 'VACINA_COV'
]

campos_disponiveis = [c for c in campos_analisar if c in df_analysis.columns]

missing_stats = []

for campo in campos_disponiveis:
    total = df_analysis.count()
    null_count = df_analysis.filter(
        F.col(campo).isNull() | (F.col(campo) == '')
    ).count()
    
    pct_missing = (null_count / total) * 100
    
    missing_stats.append({
        'campo': campo,
        'missing_count': null_count,
        'missing_pct': pct_missing
    })

pd_missing = pd.DataFrame(missing_stats).sort_values('missing_pct', ascending=False)

plt.figure(figsize=(12, 8))
colors = ['#e74c3c' if x > 40 else '#f39c12' if x > 20 else '#27ae60' for x in pd_missing['missing_pct']]

bars = plt.barh(pd_missing['campo'], pd_missing['missing_pct'], color=colors, alpha=0.7)

plt.title('Percentual de Valores Ausentes por Campo', fontsize=14, fontweight='bold')
plt.xlabel('% Missing', fontsize=12)
plt.xlim(0, 100)
plt.axvline(x=20, color='orange', linestyle='--', linewidth=1, label='20%')
plt.axvline(x=40, color='red', linestyle='--', linewidth=1, label='40%')
plt.legend()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

print("\nCAMPOS COM MAIS MISSING:")
for idx, row in pd_missing.head(10).iterrows():
    print(f"  • {row['campo']}: {row['missing_pct']:.1f}% ({row['missing_count']:,} registros)")

# COMMAND ----------

print("\nANÁLISE DE CÓDIGO '9' (IGNORADO)")
print("=" * 80)

campos_code9 = ['CS_SEXO', 'CS_RACA', 'FEBRE', 'TOSSE', 'DISPNEIA', 'SATURACAO',
                'HOSPITAL', 'UTI', 'EVOLUCAO', 'VACINA']

campos_code9_disp = [c for c in campos_code9 if c in df_analysis.columns]

code9_stats = []

for campo in campos_code9_disp:
    total = df_analysis.count()
    code9_count = df_analysis.filter(F.col(campo) == '9').count()
    
    pct_code9 = (code9_count / total) * 100
    
    code9_stats.append({
        'campo': campo,
        'code9_count': code9_count,
        'code9_pct': pct_code9
    })

pd_code9 = pd.DataFrame(code9_stats).sort_values('code9_pct', ascending=False)

plt.figure(figsize=(12, 6))
colors_9 = ['#e74c3c' if x > 30 else '#f39c12' if x > 15 else '#3498db' for x in pd_code9['code9_pct']]

bars = plt.barh(pd_code9['campo'], pd_code9['code9_pct'], color=colors_9, alpha=0.7)

plt.title('Percentual de Código "9" (Ignorado) por Campo', fontsize=14, fontweight='bold')
plt.xlabel('% Código "9"', fontsize=12)
plt.xlim(0, 100)
plt.axvline(x=15, color='orange', linestyle='--', linewidth=1, label='15%')
plt.axvline(x=30, color='red', linestyle='--', linewidth=1, label='30%')
plt.legend()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

print("\nCAMPOS COM MAIS CÓDIGO '9':")
for idx, row in pd_code9.iterrows():
    if row['code9_pct'] > 10:
        print(f"  • {row['campo']}: {row['code9_pct']:.1f}% ({row['code9_count']:,} registros)")

print("\nINTERPRETAÇÃO:")
print("  • Código '9' = 'Ignorado' no padrão DATASUS")
print("  • Não é equivalente a valor ausente (NULL)")
print("  • Indica que a informação não foi obtida ou não se aplica")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. FEATURE SELECTION PARA SILVER

# COMMAND ----------

print("\nFEATURE SELECTION PARA CAMADA SILVER")
print("=" * 80)

features_selecionadas = {
    'ESSENCIAIS (Usar sempre)': [
        'NU_NOTIFIC',
        'DT_SIN_PRI',
        'DT_NOTIFIC',
        'SEM_PRI',
        'ANO_DADOS',
    ],
    
    'DEMOGRÁFICAS (Alta qualidade)': [
        'SG_UF',
        'CO_MUN_RES',
        'CS_SEXO',
        'NU_IDADE_N',
        'TP_IDADE',
    ],
    
    'CLÍNICAS (Para métricas)': [
        'FEBRE',
        'TOSSE',
        'DISPNEIA',
        'SATURACAO',
        'HOSPITAL',
        'UTI',
        'DT_INTERNA',
        'EVOLUCAO',
        'DT_EVOLUCA',
    ],
    
    'VACINAÇÃO (Métrica específica)': [
        'VACINA',
        'VACINA_COV',
    ],
    
    'OPCIONAIS (Enriquecimento)': [
        'CS_RACA',
        'CS_ESCOL_N',
        'CLASSI_FIN',
    ]
}

print("\nFEATURES RECOMENDADAS POR CATEGORIA:\n")

total_features = 0
for categoria, campos in features_selecionadas.items():
    campos_existentes = [c for c in campos if c in df_analysis.columns]
    total_features += len(campos_existentes)
    
    print(f"\n{categoria}:")
    for campo in campos:
        existe = "✅" if campo in df_analysis.columns else "❌"
        print(f"  {existe} {campo}")

print("\n" + "=" * 80)
print(f"TOTAL DE FEATURES RECOMENDADAS: {total_features}")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. RESUMO EXECUTIVO

# COMMAND ----------

print("\n" + "=" * 80)
print("RESUMO EXECUTIVO - ANÁLISE EXPLORATÓRIA DE DADOS SRAG")
print("=" * 80)

print("\nOBJETIVO:")
print("  Analisar dados SRAG da camada Bronze para embasar decisões da camada Silver")

print("\nDATASET:")
print(f"  • Total de registros: {total_rows:,}")
print(f"  • Total de colunas: {total_cols}")

print("\nCORREÇÕES APLICADAS:")
print("  ✅ Conversão de datas com try_to_date (Spark 4 compatível)")
print("  ✅ Padronização de CS_SEXO como string")
print("  ✅ Código 100% compatível com Databricks Serverless")
print("  ✅ Uso consciente de toPandas() apenas para visualizações")

print("\nQUALIDADE DOS DADOS:")
if len(pd_missing) > 0:
    campos_criticos = pd_missing[pd_missing['missing_pct'] > 40]
    print(f"  • Campos com >40% missing: {len(campos_criticos)}")

print("\nMÉTRICAS VALIDADAS:")
print("  ✅ Taxa de Mortalidade: Calculável com EVOLUCAO")
print("  ✅ Taxa UTI: Calculável com HOSPITAL + UTI")
print("  ✅ Taxa Vacinação: Calculável com VACINA")
print("  ✅ Taxa Crescimento: Calculável com séries temporais")

print("\nRECOMENDAÇÕES PARA SILVER:")
print(f"  • Manter {total_features} features essenciais")
print("  • Aplicar padronização de datas (try_to_date)")
print("  • Criar campos derivados (faixa_etaria, ano_mes)")
print("  • Documentar tratamento de código '9'")
print("  • Implementar validações de domínio")

print("\nPRÓXIMOS PASSOS:")
print("  1. Criar camada Silver com features selecionadas")
print("  2. Implementar transformações de qualidade")
print("  3. Calcular métricas agregadas")
print("  4. Preparar dados para camada Gold")

print("\n" + "=" * 80)
print(f"ANÁLISE CONCLUÍDA - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)