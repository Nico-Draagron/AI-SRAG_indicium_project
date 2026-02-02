# Databricks notebook source
# MAGIC %md
# MAGIC # 🔧 Gold - Setup e Configuração
# MAGIC 
# MAGIC **Responsabilidade**: Configurar ambiente, criar schemas e definir constantes
# MAGIC 
# MAGIC **Execute sempre primeiro!**
# MAGIC 

# COMMAND ----------

from pyspark.sql import functions as F
from datetime import datetime

print("=" * 80)
print("🔧 GOLD - SETUP E CONFIGURAÇÃO")
print("=" * 80)
print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🔧 Spark Version: {spark.version}")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📁 Configuração de Catálogos e Schemas

# COMMAND ----------

# ✅ Unity Catalog - DOIS CATÁLOGOS SEPARADOS
CATALOG_SILVER = "workspace"          # ✅ Catálogo onde está a Silver (INPUT)
CATALOG_GOLD = "dbx_lab_draagron"     # ✅ Catálogo onde criar a Gold (OUTPUT)

# Schemas
SCHEMA_SILVER = "silver"
SCHEMA_GOLD = "gold"

# Tabela fonte (leitura do catálogo Silver)
TABLE_SILVER = f"{CATALOG_SILVER}.{SCHEMA_SILVER}.silver_srag_clean"
# Resultado: workspace.silver.silver_srag_clean

# Tabelas Gold (escrita no catálogo Gold)
TABLES_GOLD = {
    'metricas_temporais': f"{CATALOG_GOLD}.{SCHEMA_GOLD}.gold_metricas_temporais",
    'metricas_geograficas': f"{CATALOG_GOLD}.{SCHEMA_GOLD}.gold_metricas_geograficas",
    'metricas_demograficas': f"{CATALOG_GOLD}.{SCHEMA_GOLD}.gold_metricas_demograficas",
    'series_temporais': f"{CATALOG_GOLD}.{SCHEMA_GOLD}.gold_series_temporais",
    'resumo_geral': f"{CATALOG_GOLD}.{SCHEMA_GOLD}.gold_resumo_geral",
    'analise_avancada': f"{CATALOG_GOLD}.{SCHEMA_GOLD}.gold_analise_avancada"
}
# Resultado: dbx_lab_draagron.gold.gold_metricas_temporais, etc.

# Views de consumo (no catálogo Gold)
VIEWS_GOLD = {
    'dashboard_principal': f"{CATALOG_GOLD}.{SCHEMA_GOLD}.vw_dashboard_principal",
    'metricas_6meses': f"{CATALOG_GOLD}.{SCHEMA_GOLD}.vw_metricas_ultimos_6_meses",
    'top10_ufs': f"{CATALOG_GOLD}.{SCHEMA_GOLD}.vw_top10_ufs",
    'alertas_mortalidade': f"{CATALOG_GOLD}.{SCHEMA_GOLD}.vw_alertas_mortalidade",
    'resumo_atual': f"{CATALOG_GOLD}.{SCHEMA_GOLD}.vw_resumo_geral_atual"
}
# Resultado: dbx_lab_draagron.gold.vw_dashboard_principal, etc.

# Process ID para rastreamento
PROCESS_ID = datetime.now().strftime('%Y%m%d_%H%M%S')

print("📂 CONFIGURAÇÃO:")
print(f"  • Catalog Silver (INPUT): {CATALOG_SILVER}")
print(f"  • Catalog Gold (OUTPUT): {CATALOG_GOLD}")
print(f"  • Schema Silver: {SCHEMA_SILVER}")
print(f"  • Schema Gold: {SCHEMA_GOLD}")
print(f"  • Fonte: {TABLE_SILVER}")
print(f"  • Tabelas a criar: {len(TABLES_GOLD)}")
print(f"  • Views a criar: {len(VIEWS_GOLD)}")
print(f"  • Process ID: {PROCESS_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🏗️ Criar Catálogo Gold

# COMMAND ----------

# ✅ Criar catálogo Gold (dbx_lab_draagron)
spark.sql(f"""
    CREATE CATALOG IF NOT EXISTS {CATALOG_GOLD}
    COMMENT 'Catálogo para camada Gold - Métricas agregadas'
""")

print(f"✅ Catálogo criado: {CATALOG_GOLD}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🗃️ Criar Schema Gold

# COMMAND ----------

# ✅ Criar schema no catálogo Gold (dbx_lab_draagron)
spark.sql(f"""
    CREATE SCHEMA IF NOT EXISTS {CATALOG_GOLD}.{SCHEMA_GOLD}
    COMMENT 'Camada Gold - Métricas agregadas para BI e RAG'
""")

print(f"✅ Schema criado: {CATALOG_GOLD}.{SCHEMA_GOLD}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Validar Dados Silver

# COMMAND ----------

# Verificar se Silver existe e tem dados
try:
    df_silver = spark.table(TABLE_SILVER)
    count_silver = df_silver.count()
    
    print(f"\n✅ Silver validada:")
    print(f"  • Tabela: {TABLE_SILVER}")
    print(f"  • Catálogo: {CATALOG_SILVER}")
    print(f"  • Registros: {count_silver:,}")
    print(f"  • Colunas: {len(df_silver.columns)}")
    
    # Período de dados
    periodo = df_silver.agg(
        F.min('dt_sin_pri').alias('min_data'),
        F.max('dt_sin_pri').alias('max_data')
    ).collect()[0]
    
    print(f"  • Período: {periodo['min_data']} até {periodo['max_data']}")
    
    # Validação básica
    assert count_silver > 0, "❌ Silver está vazia!"
    
except Exception as e:
    print(f"❌ ERRO ao acessar Silver: {str(e)}")
    print(f"\n💡 Dica: Verifique se:")
    print(f"   1. O catálogo '{CATALOG_SILVER}' existe")
    print(f"   2. Você tem permissão de leitura no catálogo '{CATALOG_SILVER}'")
    print(f"   3. A tabela '{TABLE_SILVER}' existe e tem dados")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔐 Exportar Configurações para Widgets

# COMMAND ----------

# ✅ Criar widgets para outros notebooks - COM DOIS CATÁLOGOS
dbutils.widgets.text("catalog_silver", CATALOG_SILVER, "Catalog Silver (INPUT)")
dbutils.widgets.text("catalog_gold", CATALOG_GOLD, "Catalog Gold (OUTPUT)")
dbutils.widgets.text("schema_silver", SCHEMA_SILVER, "Schema Silver")
dbutils.widgets.text("schema_gold", SCHEMA_GOLD, "Schema Gold")
dbutils.widgets.text("table_silver", TABLE_SILVER, "Tabela Silver")
dbutils.widgets.text("process_id", PROCESS_ID, "Process ID")

print("✅ Widgets criados para compartilhamento entre notebooks")
print("\n📋 Widgets disponíveis:")
print(f"  • catalog_silver = {CATALOG_SILVER}")
print(f"  • catalog_gold = {CATALOG_GOLD}")
print(f"  • schema_silver = {SCHEMA_SILVER}")
print(f"  • schema_gold = {SCHEMA_GOLD}")
print(f"  • table_silver = {TABLE_SILVER}")
print(f"  • process_id = {PROCESS_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📋 Resumo

# COMMAND ----------

print("\n" + "=" * 80)
print("✅ SETUP CONCLUÍDO COM SUCESSO")
print("=" * 80)
print(f"\n📊 Configuração:")
print(f"  • Lendo de: {CATALOG_SILVER}.{SCHEMA_SILVER}")
print(f"  • Escrevendo em: {CATALOG_GOLD}.{SCHEMA_GOLD}")
print(f"\n📊 Próximos passos:")
print(f"  1. Execute: gold_metricas_temporais")
print(f"  2. Execute: gold_metricas_geograficas")
print(f"  3. Execute: gold_metricas_demograficas")
print(f"  4. Execute: gold_series_resumo")
print(f"\n💡 Dica: Notebooks 1-4 podem rodar em paralelo!")
print("=" * 80)
