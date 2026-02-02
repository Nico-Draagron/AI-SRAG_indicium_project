# Databricks notebook source
# MAGIC %md
# MAGIC # 🟫 Camada Bronze - Ingestão de Dados SRAG
# MAGIC 
# MAGIC **Projeto**: Sistema RAG para Monitoramento Epidemiológico - Indicium Healthcare PoC
# MAGIC 
# MAGIC **Objetivo**: Ingerir dados brutos de SRAG (2023-2025) do DATASUS para Delta Tables no Unity Catalog
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC ## 📋 Responsabilidades da Camada Bronze
# MAGIC 
# MAGIC ✅ **O que Bronze FAZ**:
# MAGIC - Ler CSVs originais sem modificação de dados
# MAGIC - Aplicar schema básico (tipos corretos)
# MAGIC - Adicionar metadados técnicos (`ANO_DADOS`, `_ingested_at`)
# MAGIC - Persistir em Delta Lake com versionamento
# MAGIC - Garantir idempotência (pode reprocessar)
# MAGIC 
# MAGIC ❌ **O que Bronze NÃO FAZ**:
# MAGIC - Filtrar registros (mesmo inválidos)
# MAGIC - Tratar valores "Ignorado" (código 9)
# MAGIC - Imputar missing values
# MAGIC - Aplicar regras de negócio
# MAGIC - Fazer agregações
# MAGIC 
# MAGIC 💡 **Filosofia**: "Dados crus + rastreabilidade"
# MAGIC 
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔧 1. Setup e Configuração

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *
from datetime import datetime
import json

# Configuração de logging
print("=" * 70)
print("🟫 BRONZE LAYER - INGESTÃO DE DADOS SRAG")
print("=" * 70)
print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🔧 Spark Version: {spark.version}")
print("=" * 70)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📁 2. Configuração de Paths (Unity Catalog)

# COMMAND ----------

# Configuração do Unity Catalog
CATALOG = "workspace"
SCHEMA_BRONZE = "data_original"
VOLUME_RAW = "data_srag"    

VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA_BRONZE}/{VOLUME_RAW}"


# Paths baseados em Unity Catalog
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA_BRONZE}/{VOLUME_RAW}"
TABLE_BRONZE = f"{CATALOG}.{SCHEMA_BRONZE}.bronze_srag_raw"

# Anos para processamento
YEARS = [2023, 2024, 2025]

print("📂 CONFIGURAÇÃO DE PATHS:")
print(f"  • Catálogo: {CATALOG}")
print(f"  • Schema: {SCHEMA_BRONZE}")
print(f"  • Volume: {VOLUME_RAW}")
print(f"  • Path CSVs: {VOLUME_PATH}")
print(f"  • Tabela destino: {TABLE_BRONZE}")
print(f"  • Anos: {YEARS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📥 3. Verificação de Arquivos

# COMMAND ----------

# Listar arquivos disponíveis no volume
try:
    files = dbutils.fs.ls(VOLUME_PATH)
    csv_files = [f for f in files if f.name.endswith('.csv')]
    
    print(f"\n✅ {len(csv_files)} arquivo(s) CSV encontrado(s):\n")
    for f in csv_files:
        size_mb = f.size / (1024 * 1024)
        print(f"  📄 {f.name:<20} ({size_mb:,.2f} MB)")
    
    if len(csv_files) == 0:
        raise FileNotFoundError("Nenhum arquivo CSV encontrado no volume!")
    
except Exception as e:
    print(f"❌ ERRO ao acessar volume: {str(e)}")
    print("\n💡 AÇÕES NECESSÁRIAS:")
    print(f"  1. Verificar se o volume existe: {VOLUME_PATH}")
    print(f"  2. Fazer upload dos CSVs (INFLUD23.csv, INFLUD24.csv, INFLUD25.csv)")
    print(f"  3. Verificar permissões no Unity Catalog")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔍 4. Função de Ingestão por Ano

# COMMAND ----------

def ingest_year_data(year: int, volume_path: str, validate_only: bool = False):
    """
    Ingere dados de SRAG de um ano específico
    
    Args:
        year: Ano dos dados (2023, 2024, ou 2025)
        volume_path: Path do volume com os CSVs
        validate_only: Se True, apenas valida sem carregar dados completos
    
    Returns:
        DataFrame Spark ou dict com estatísticas (se validate_only=True)
    """
    
    # Identificar arquivo do ano
    year_suffix = str(year)[2:]  # 2023 -> 23
    file_pattern = f"INFLUD{year_suffix}-*.csv"
    file_path = f"{volume_path}/{file_pattern}"

    
    print(f"\n{'=' * 70}")
    print(f"📥 PROCESSANDO ANO {year}")
    print(f"{'=' * 70}")
    print(f"📂 Arquivo: {file_pattern}")
    
    try:
        # Leitura com configurações específicas para SRAG/DATASUS
        df = spark.read \
            .option("header", "true") \
            .option("sep", ";") \
            .option("encoding", "ISO-8859-1") \
            .option("inferSchema", "false") \
            .option("quote", '"') \
            .option("escape", '"') \
            .option("multiLine", "true") \
            .option("mode", "PERMISSIVE") \
            .csv(file_path)
        
        # Estatísticas básicas
        if validate_only:
            # Apenas contagem (sem materializar DataFrame completo)
            row_count = df.count()
            col_count = len(df.columns)
            
            stats = {
                'year': year,
                'file': file_pattern,
                'rows': row_count,
                'columns': col_count,
                'status': 'OK'
            }
            
            print(f"  ✅ Registros: {row_count:,}")
            print(f"  ✅ Colunas: {col_count}")
            
            return stats
        
        else:
            # Carregar dados completos
            row_count = df.count()
            col_count = len(df.columns)
            
            print(f"  ✅ Registros: {row_count:,}")
            print(f"  ✅ Colunas: {col_count}")
            
            # Adicionar metadados técnicos (enriquecimento mínimo permitido no Bronze)
            df_enriched = df \
                .withColumn("ANO_DADOS", F.lit(year)) \
                .withColumn("_ingested_at", F.current_timestamp()) \
                .withColumn("_source_file", F.lit(file_pattern))
            
            print(f"  ✅ Metadados adicionados: ANO_DADOS, _ingested_at, _source_file")
            
            return df_enriched
    
    except Exception as e:
        print(f"  ❌ ERRO ao processar {year}: {str(e)}")
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✔️ 5. Validação Prévia (Dry Run)

# COMMAND ----------

print("\n🔍 VALIDAÇÃO PRÉVIA (Dry Run):")
print("Verificando todos os arquivos antes de ingestão completa...\n")

validation_results = []

for year in YEARS:
    try:
        stats = ingest_year_data(year, VOLUME_PATH, validate_only=True)
        validation_results.append(stats)
    except Exception as e:
        validation_results.append({
            'year': year,
            'status': 'ERRO',
            'error': str(e)
        })

# Resumo da validação
print(f"\n{'=' * 70}")
print("📊 RESUMO DA VALIDAÇÃO")
print(f"{'=' * 70}")

total_rows = 0
all_ok = True

for result in validation_results:
    if result['status'] == 'OK':
        print(f"  ✅ {result['year']}: {result['rows']:,} registros | {result['columns']} colunas")
        total_rows += result['rows']
    else:
        print(f"  ❌ {result['year']}: FALHOU - {result.get('error', 'Erro desconhecido')}")
        all_ok = False

print(f"\n  📊 TOTAL ESTIMADO: {total_rows:,} registros")

if not all_ok:
    raise Exception("❌ Validação falhou. Corrija os erros antes de prosseguir.")

print("\n✅ Validação concluída com sucesso! Prosseguindo para ingestão completa...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📦 6. Ingestão Completa

# COMMAND ----------

print("\n🚀 INICIANDO INGESTÃO COMPLETA...")

dataframes = {}

for year in YEARS:
    df = ingest_year_data(year, VOLUME_PATH, validate_only=False)
    dataframes[year] = df
    

print(f"\n✅ {len(dataframes)} DataFrames carregados em memória")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔗 7. União dos Anos

# COMMAND ----------

print("\n🔗 UNINDO DADOS DE TODOS OS ANOS...")

# União com allowMissingColumns (SRAG muda schema entre anos)
df_bronze = dataframes[YEARS[0]]

for year in YEARS[1:]:
    df_bronze = df_bronze.unionByName(
        dataframes[year], 
        allowMissingColumns=True
    )

# Estatísticas finais
total_rows = df_bronze.count()
total_cols = len(df_bronze.columns)

print(f"\n{'=' * 70}")
print("📊 DATASET BRONZE CONSOLIDADO")
print(f"{'=' * 70}")
print(f"  📈 Total de registros: {total_rows:,}")
print(f"  📋 Total de colunas: {total_cols}")
print(f"  📅 Período: {min(YEARS)} - {max(YEARS)}")
print(f"{'=' * 70}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔍 8. Inspeção Estrutural

# COMMAND ----------

print("\n🔍 INSPEÇÃO ESTRUTURAL DO DATASET:")

# Schema resumido (primeiras 20 colunas)
print("\n📋 SCHEMA (primeiras 20 colunas):")
for field in df_bronze.schema.fields[:20]:
    print(f"  • {field.name:<25} ({field.dataType})")

if total_cols > 20:
    print(f"  ... e mais {total_cols - 20} colunas")

# COMMAND ----------

# Distribuição por ano
print("\n📅 DISTRIBUIÇÃO POR ANO:")
df_bronze.groupBy("ANO_DADOS") \
    .count() \
    .orderBy("ANO_DADOS") \
    .show()

# COMMAND ----------

# Amostra dos dados
print("\n👀 AMOSTRA DE DADOS (5 primeiras linhas):")
print("Exibindo apenas colunas essenciais para verificação...")

essential_cols = [
    'DT_NOTIFIC', 'DT_SIN_PRI', 'SEM_PRI', 'CS_SEXO', 
    'NU_IDADE_N', 'SG_UF', 'HOSPITAL', 'UTI', 'EVOLUCAO',
    'ANO_DADOS', '_ingested_at'
]

# Filtrar apenas colunas que existem
display_cols = [col for col in essential_cols if col in df_bronze.columns]

display(df_bronze.select(display_cols).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 💾 9. Persistência em Delta Lake

# COMMAND ----------

print("\n💾 SALVANDO DADOS NA CAMADA BRONZE...")
print(f"🎯 Tabela destino: {TABLE_BRONZE}")

# Configuração de escrita
write_config = {
    'mode': 'overwrite',  # Sobrescrever para garantir idempotência
    'format': 'delta',
    'overwriteSchema': True  # Permitir schema evolution
}

print(f"\n⚙️ CONFIGURAÇÃO DE ESCRITA:")
for key, value in write_config.items():
    print(f"  • {key}: {value}")

# Escrever Delta Table
try:
    df_bronze.write \
        .mode(write_config['mode']) \
        .format(write_config['format']) \
        .option("overwriteSchema", write_config['overwriteSchema']) \
        .saveAsTable(TABLE_BRONZE)
    
    print(f"\n✅ Tabela {TABLE_BRONZE} criada com sucesso!")
    
except Exception as e:
    print(f"\n❌ ERRO ao salvar tabela: {str(e)}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ 10. Validação da Tabela Criada

# COMMAND ----------

print("\n✔️ VALIDANDO TABELA CRIADA...")

# Ler tabela recém-criada
df_validation = spark.table(TABLE_BRONZE)

# Verificações
validation_checks = {
    'row_count_match': df_validation.count() == total_rows,
    'columns_count': len(df_validation.columns) == total_cols,
    'has_metadata': all(col in df_validation.columns for col in ['ANO_DADOS', '_ingested_at', '_source_file']),
    'all_years_present': df_validation.select('ANO_DADOS').distinct().count() == len(YEARS)
}

print(f"\n{'=' * 70}")
print("🔍 VERIFICAÇÕES DE INTEGRIDADE")
print(f"{'=' * 70}")

all_passed = True
for check, result in validation_checks.items():
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"  {status} - {check}")
    if not result:
        all_passed = False

if not all_passed:
    raise Exception("❌ Validação da tabela falhou!")

print(f"\n✅ Todas as verificações passaram!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 11. Metadados da Tabela

# COMMAND ----------

# Informações da tabela Delta
print("\n📊 METADADOS DA TABELA DELTA:")

# Descrição
spark.sql(f"DESCRIBE EXTENDED {TABLE_BRONZE}").show(50, truncate=False)

# COMMAND ----------

# Histórico Delta (time travel)
print("\n🕐 HISTÓRICO DELTA (Time Travel):")

try:
    history = spark.sql(f"DESCRIBE HISTORY {TABLE_BRONZE}")
    display(history.select('version', 'timestamp', 'operation', 'operationMetrics'))
except Exception as e:
    print(f"ℹ️ Histórico não disponível: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📈 12. Estatísticas e Métricas

# COMMAND ----------

# Estatísticas de ingestão
ingestion_stats = {
    'timestamp': datetime.now().isoformat(),
    'catalog': CATALOG,
    'schema': SCHEMA_BRONZE,
    'table': TABLE_BRONZE.split('.')[-1],
    'total_rows': total_rows,
    'total_columns': total_cols,
    'years_processed': YEARS,
    'source_files': [f"INFLUD{str(y)[2:]}.csv" for y in YEARS],
    'validation_passed': all_passed
}

print("\n📈 ESTATÍSTICAS DE INGESTÃO:")
print(json.dumps(ingestion_stats, indent=2))

# Salvar estatísticas (opcional)
stats_path = f"/Volumes/{CATALOG}/{SCHEMA_BRONZE}/logs/ingestion_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

try:
    dbutils.fs.mkdirs(f"/Volumes/{CATALOG}/{SCHEMA_BRONZE}/logs")
    dbutils.fs.put(stats_path, json.dumps(ingestion_stats, indent=2), overwrite=True)
    print(f"\n💾 Estatísticas salvas em: {stats_path}")
except Exception as e:
    print(f"\nℹ️ Não foi possível salvar estatísticas: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 13. Conclusão e Próximos Passos

# COMMAND ----------

print("\n" + "=" * 70)
print("🎉 INGESTÃO BRONZE CONCLUÍDA COM SUCESSO!")
print("=" * 70)

print(f"\n📊 RESUMO:")
print(f"  • Tabela criada: {TABLE_BRONZE}")
print(f"  • Registros totais: {total_rows:,}")
print(f"  • Colunas: {total_cols}")
print(f"  • Anos: {', '.join(map(str, YEARS))}")
print(f"  • Formato: Delta Lake")
print(f"  • Schema evolution: Habilitado")
print(f"  • Time travel: Disponível")

print(f"\n✅ QUALIDADE DOS DADOS:")
print(f"  • Todas as validações passaram")
print(f"  • Metadados técnicos adicionados")
print(f"  • Tabela pronta para próxima camada")

print(f"\n🚀 PRÓXIMOS PASSOS:")
print(f"  1. Notebook 02: Validação de Qualidade (Data Quality Checks)")
print(f"  2. Notebook 03: EDA Profissional (Análise Exploratória)")
print(f"  3. Notebook 04: Camada Silver (Limpeza e Transformação)")

print("\n" + "=" * 70)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC 
# MAGIC ## 📚 Documentação e Referências
# MAGIC 
# MAGIC ### Arquitetura Medallion
# MAGIC 
# MAGIC ```
# MAGIC 🟫 BRONZE (você está aqui)
# MAGIC  ↓
# MAGIC 🔍 VALIDATION
# MAGIC  ↓
# MAGIC 📊 EDA
# MAGIC  ↓
# MAGIC 🥈 SILVER
# MAGIC  ↓
# MAGIC 🥇 GOLD
# MAGIC  ↓
# MAGIC 🤖 RAG
# MAGIC ```
# MAGIC 
# MAGIC ### Decisões Arquiteturais
# MAGIC 
# MAGIC 1. **Metadados no Bronze**: `ANO_DADOS`, `_ingested_at`, `_source_file`
# MAGIC    - Justificativa: Enriquecimento técnico mínimo para rastreabilidade
# MAGIC    - Não viola princípio "Bronze = dados crus" (são metadados, não transformações)
# MAGIC 
# MAGIC 2. **`allowMissingColumns=True`**
# MAGIC    - SRAG muda schema entre anos (ex: campos COVID em 2023+)
# MAGIC    - Union precisa ser flexível
# MAGIC 
# MAGIC 3. **`overwriteSchema=True`**
# MAGIC    - Permite reprocessamento com schema evolution
# MAGIC    - Essencial para manutenção do pipeline
# MAGIC 
# MAGIC ### Fonte dos Dados
# MAGIC 
# MAGIC - **Sistema**: SIVEP-Gripe (DATASUS)
# MAGIC - **Período**: 2023-2025
# MAGIC - **Formato original**: CSV (separador `;`, encoding ISO-8859-1)
# MAGIC - **Dicionário**: `dicionario_de_dados_SRAG_hospitalizado_2019.pdf`
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC **Desenvolvido para**: Indicium Healthcare PoC - AI Engineer Certification
# MAGIC 
# MAGIC **Próximo notebook**: `02_Data_Quality_Validation.py`