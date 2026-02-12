# ===== DIAGNÓSTICO COMPLETO: SCHEMA DAS TABELAS GOLD =====
print("=" * 80)
print("DIAGNÓSTICO: SCHEMA E AMOSTRAS DAS TABELAS GOLD")
print("=" * 80)

# 1. GOLD_METRICAS_TEMPORAIS
print("\n" + "=" * 80)
print("1️⃣ GOLD_METRICAS_TEMPORAIS")
print("=" * 80)

try:
    # Schema completo
    schema_temporal = spark.sql("""
        DESCRIBE dbx_lab_draagron.gold.gold_metricas_temporais
    """).toPandas()
    print("\n📋 SCHEMA (colunas disponíveis):")
    print(schema_temporal[['col_name', 'data_type']])
    
    # Amostra de 3 linhas COM TODAS AS COLUNAS
    sample_temporal = spark.sql("""
        SELECT *
        FROM dbx_lab_draagron.gold.gold_metricas_temporais
        ORDER BY ano_mes DESC
        LIMIT 3
    """).toPandas()
    print("\n📊 AMOSTRA (3 linhas mais recentes):")
    print(sample_temporal.to_string())
    
    # Valores únicos de ano_mes (para ver quais meses existem)
    distinct_months = spark.sql("""
        SELECT DISTINCT ano_mes
        FROM dbx_lab_draagron.gold.gold_metricas_temporais
        ORDER BY ano_mes DESC
    """).toPandas()
    print(f"\n📅 MESES DISPONÍVEIS ({len(distinct_months)} meses):")
    print(distinct_months['ano_mes'].tolist())
    
except Exception as e:
    print(f"❌ Erro: {e}")

# 2. GOLD_METRICAS_GEOGRAFICAS
print("\n" + "=" * 80)
print("2️⃣ GOLD_METRICAS_GEOGRAFICAS")
print("=" * 80)

try:
    # Schema completo
    schema_geo = spark.sql("""
        DESCRIBE dbx_lab_draagron.gold.gold_metricas_geograficas
    """).toPandas()
    print("\n📋 SCHEMA (colunas disponíveis):")
    print(schema_geo[['col_name', 'data_type']])
    
    # Amostra de 3 linhas COM TODAS AS COLUNAS
    sample_geo = spark.sql("""
        SELECT *
        FROM dbx_lab_draagron.gold.gold_metricas_geograficas
        ORDER BY ranking_casos
        LIMIT 3
    """).toPandas()
    print("\n📊 AMOSTRA (Top 3 UFs):")
    print(sample_geo.to_string())
    
    # Valores únicos de UF
    distinct_ufs = spark.sql("""
        SELECT DISTINCT sg_uf
        FROM dbx_lab_draagron.gold.gold_metricas_geograficas
        ORDER BY sg_uf
    """).toPandas()
    print(f"\n🗺️ UFs DISPONÍVEIS ({len(distinct_ufs)} estados):")
    print(distinct_ufs['sg_uf'].tolist())
    
except Exception as e:
    print(f"❌ Erro: {e}")

# 3. GOLD_RESUMO_GERAL
print("\n" + "=" * 80)
print("3️⃣ GOLD_RESUMO_GERAL")
print("=" * 80)

try:
    # Schema completo
    schema_resumo = spark.sql("""
        DESCRIBE dbx_lab_draagron.gold.gold_resumo_geral
    """).toPandas()
    print("\n📋 SCHEMA (colunas disponíveis):")
    print(schema_resumo[['col_name', 'data_type']])
    
    # Amostra de 5 linhas COM TODAS AS COLUNAS
    sample_resumo = spark.sql("""
        SELECT *
        FROM dbx_lab_draagron.gold.gold_resumo_geral
        LIMIT 5
    """).toPandas()
    print("\n📊 AMOSTRA (5 primeiras métricas):")
    print(sample_resumo.to_string())
    
    # Categorias disponíveis
    distinct_cats = spark.sql("""
        SELECT DISTINCT categoria
        FROM dbx_lab_draagron.gold.gold_resumo_geral
        ORDER BY categoria
    """).toPandas()
    print(f"\n📁 CATEGORIAS DISPONÍVEIS ({len(distinct_cats)}):")
    print(distinct_cats['categoria'].tolist())
    
except Exception as e:
    print(f"❌ Erro: {e}")

# 4. GOLD_METRICAS_DEMOGRAFICAS (se existir)
print("\n" + "=" * 80)
print("4️⃣ GOLD_METRICAS_DEMOGRAFICAS")
print("=" * 80)

try:
    # Schema completo
    schema_demo = spark.sql("""
        DESCRIBE dbx_lab_draagron.gold.gold_metricas_demograficas
    """).toPandas()
    print("\n📋 SCHEMA (colunas disponíveis):")
    print(schema_demo[['col_name', 'data_type']])
    
    # Amostra de 3 linhas COM TODAS AS COLUNAS
    sample_demo = spark.sql("""
        SELECT *
        FROM dbx_lab_draagron.gold.gold_metricas_demograficas
        LIMIT 3
    """).toPandas()
    print("\n📊 AMOSTRA (3 primeiras linhas):")
    print(sample_demo.to_string())
    
except Exception as e:
    print(f"ℹ️ Tabela não existe ou erro: {e}")

print("\n" + "=" * 80)
print("FIM DO DIAGNÓSTICO")
print("=" * 80)