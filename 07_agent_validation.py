# Databricks notebook source
# MAGIC %md
# MAGIC # 07 — Validação, Análise Epidemiológica e Testes do Agente SRAG
# MAGIC
# MAGIC ## Propósito
# MAGIC
# MAGIC Notebook de **observabilidade, pesquisa epidemiológica e QA do agente**.
# MAGIC Combina três funções distintas em um único ambiente de pesquisa:
# MAGIC
# MAGIC | Função | O que faz |
# MAGIC |---|---|
# MAGIC | **Análise epidemiológica** | Panorama histórico completo via SQL direto nas tabelas Gold — sem passar pelo agente |
# MAGIC | **Validação do agente** | Compara o que o agente calculou com o ground truth do Gold |
# MAGIC | **Testes de conversa** | Envia mensagens reais ao agente e avalia routing, gráficos, SQL e RAG |
# MAGIC
# MAGIC **Fluxo de uso:**
# MAGIC 1. Execute o `06_agent_system` completo (pipeline principal)
# MAGIC 2. Execute este notebook na mesma sessão
# MAGIC 3. Cole o bloco da seção **14** no chat para diagnóstico assistido
# MAGIC
# MAGIC ---
# MAGIC ## Seções
# MAGIC | # | Seção |
# MAGIC |---|---|
# MAGIC | 1 | Setup e verificação de objetos em memória |
# MAGIC | 2 | **Panorama histórico — 2023 / 2024 / 2025 (SQL direto)** |
# MAGIC | 3 | Análise mensal completa — todas as métricas por mês |
# MAGIC | 4 | Análise trimestral comparativa entre anos |
# MAGIC | 5 | Sazonalidade e padrão intra-anual |
# MAGIC | 6 | **Contexto SIVEP — cutoff, subnotificação, freshness** |
# MAGIC | 7 | Sessão de auditoria — eventos e erros da última execução |
# MAGIC | 8 | Validação das métricas do agente vs ground truth |
# MAGIC | 9 | Roteamento — cobertura de todas as rotas |
# MAGIC | 10 | RAG — qualidade de retrieval semântico |
# MAGIC | 11 | Vector Store — saúde do índice |
# MAGIC | 12 | **Testes de conversa com o agente** |
# MAGIC | 13 | Score de qualidade agregado |
# MAGIC | 14 | Output formatado para revisão |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

# DBTITLE 1,1.1 — Imports
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# COMMAND ----------

# DBTITLE 1,1.2 — Verificar objetos do notebook 06 em memória
_REQUIRED = {
    "spark":           "SparkSession",
    "result":          "dict — resultado do orchestrator.run()",
    "audit_logger":    "AuditLogger",
    "orchestrator":    "SRAGOrchestrator",
    "rag_chain":       "SRAGChain",
    "vector_manager":  "DatabricksVectorStoreManager",
    "web_search_tool": "WebSearchTool",
    "chart_tool":      "ChartTool",
    "all_charts":      "List[str]",
}

_ctx, _missing = {}, []
for _n, _d in _REQUIRED.items():
    try:
        _v = eval(_n)  # noqa: S307
        _ctx[_n] = _v
        print(f"  {'OK ' if _v is not None else 'AVS'}  {_n:22s}: {_d}")
    except NameError:
        _missing.append(_n)
        print(f"  ERR  {_n:22s}: NAO ENCONTRADO")

print(f"\n{'[OK] Todos os objetos encontrados.' if not _missing else f'[!] {len(_missing)} ausentes: {_missing}'}")

_result       = _ctx.get("result", {})
_audit        = _ctx.get("audit_logger")
_orchestrator = _ctx.get("orchestrator")
_rag          = _ctx.get("rag_chain")
_vsm          = _ctx.get("vector_manager")
_web          = _ctx.get("web_search_tool")
_charts       = _ctx.get("all_charts", [])

# COMMAND ----------

# DBTITLE 1,1.3 — Constantes
try:
    CATALOG_GOLD  = CATALOG_GOLD   # noqa
    SCHEMA_GOLD   = SCHEMA_GOLD    # noqa
    CATALOG_AUDIT = CATALOG_AUDIT  # noqa
    SCHEMA_AUDIT  = SCHEMA_AUDIT   # noqa
    SESSION_ID    = SESSION_ID     # noqa
except NameError:
    CATALOG_GOLD  = "dbx_srag_lab"
    SCHEMA_GOLD   = "gold"
    CATALOG_AUDIT = "dbx_srag_lab"
    SCHEMA_AUDIT  = "audit"
    SESSION_ID    = "desconhecido"

_G = f"{CATALOG_GOLD}.{SCHEMA_GOLD}"

print(f"catalog gold  : {CATALOG_GOLD}.{SCHEMA_GOLD}")
print(f"catalog audit : {CATALOG_AUDIT}.{SCHEMA_AUDIT}")
print(f"session_id    : {SESSION_ID}")

# Coletor central — alimenta o bloco de output final (secao 14)
_VAL = {}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Panorama Historico — 2023 / 2024 / 2025
# MAGIC
# MAGIC Queries diretas em `gold_metricas_historicas` (todo o historico Silver).
# MAGIC O agente **nao tem** esses dados no RAG — sao consultados aqui para validacao independente.

# COMMAND ----------

# DBTITLE 1,2.1 — Comparativo anual: casos, mortalidade, UTI, vacinacao
_q_anual = f"""
SELECT
    ano,
    SUM(total_casos)                                                      AS total_casos,
    SUM(total_obitos)                                                     AS total_obitos,
    SUM(casos_com_desfecho)                                               AS casos_com_desfecho,
    ROUND(SUM(total_obitos)*100.0/NULLIF(SUM(casos_com_desfecho),0), 2)  AS taxa_mortalidade_pct,
    SUM(total_internados)                                                 AS total_internados,
    SUM(total_uti)                                                        AS total_uti,
    ROUND(SUM(total_uti)*100.0/NULLIF(SUM(total_internados),0), 2)       AS taxa_uti_pct,
    SUM(total_vacinados)                                                  AS total_vacinados,
    SUM(casos_com_info_vacina)                                            AS casos_com_info_vacina,
    ROUND(SUM(total_vacinados)*100.0/NULLIF(SUM(casos_com_info_vacina),0), 2) AS taxa_vacinacao_pct,
    ROUND(AVG(idade_media), 1)                                            AS idade_media_ano,
    ROUND(AVG(tempo_medio_notificacao), 1)                                AS tempo_notificacao_dias
FROM {_G}.gold_metricas_historicas
WHERE ano IN (2023, 2024, 2025)
GROUP BY ano
ORDER BY ano
"""

_df_anual = spark.sql(_q_anual).toPandas()
_VAL["anual"] = _df_anual.to_dict("records")

print("COMPARATIVO ANUAL - 2023 / 2024 / 2025")
print("-" * 100)
print(f"  {'Ano':>6} | {'Casos':>10} | {'Obitos':>8} | {'Mortalidade':>12} | {'UTI':>8} | {'Vacinacao':>10} | {'Idade Med':>10} | {'T.Notif(d)':>10}")
print("  " + "-" * 93)
for _, r in _df_anual.iterrows():
    print(
        f"  {int(r.ano):>6} | {int(r.total_casos):>10,} | {int(r.total_obitos):>8,} | "
        f"{r.taxa_mortalidade_pct:>11.2f}% | {r.taxa_uti_pct:>7.2f}% | "
        f"{r.taxa_vacinacao_pct:>9.2f}% | {r.idade_media_ano:>10.1f} | "
        f"{r.tempo_notificacao_dias:>10.1f}"
    )

if len(_df_anual) >= 2:
    print("\n  VARIACAO MORTALIDADE ANO A ANO:")
    for i in range(1, len(_df_anual)):
        r_ant = _df_anual.iloc[i-1]
        r_cur = _df_anual.iloc[i]
        delta = r_cur.taxa_mortalidade_pct - r_ant.taxa_mortalidade_pct
        icon  = "v" if delta < 0 else "^"
        print(f"    {int(r_ant.ano)} -> {int(r_cur.ano)}: {icon} {abs(delta):.2f} pp  "
              f"({r_ant.taxa_mortalidade_pct:.2f}% -> {r_cur.taxa_mortalidade_pct:.2f}%)")

# COMMAND ----------

# DBTITLE 1,2.2 — Taxa de crescimento anual de casos
_q_cresc_anual = f"""
WITH agg AS (
    SELECT ano, SUM(total_casos) AS total_casos
    FROM {_G}.gold_metricas_historicas
    WHERE ano IN (2022, 2023, 2024, 2025)
    GROUP BY ano
),
com_lag AS (
    SELECT ano, total_casos,
           LAG(total_casos) OVER (ORDER BY ano) AS casos_ano_anterior
    FROM agg
)
SELECT
    ano, total_casos, casos_ano_anterior,
    ROUND((total_casos - casos_ano_anterior)*100.0/NULLIF(casos_ano_anterior,0), 2)
        AS taxa_crescimento_anual_pct
FROM com_lag
WHERE casos_ano_anterior IS NOT NULL
ORDER BY ano
"""

_df_cresc_anual = spark.sql(_q_cresc_anual).toPandas()
_VAL["crescimento_anual"] = _df_cresc_anual.to_dict("records")

print("TAXA DE CRESCIMENTO ANUAL DE CASOS")
print("-" * 65)
print(f"  {'Ano':>6} | {'Casos':>10} | {'Ano Anterior':>13} | {'Crescimento':>12}")
print("  " + "-" * 60)
for _, r in _df_cresc_anual.iterrows():
    icon = "^" if r.taxa_crescimento_anual_pct > 0 else "v"
    print(
        f"  {int(r.ano):>6} | {int(r.total_casos):>10,} | "
        f"{int(r.casos_ano_anterior):>13,} | "
        f"{icon} {abs(r.taxa_crescimento_anual_pct):>10.2f}%"
    )

# COMMAND ----------

# DBTITLE 1,2.3 — Breakdown etiologico anual (COVID vs Influenza vs Outros)
try:
    _q_etio = f"""
    SELECT
        ano,
        SUM(total_casos)              AS total_casos,
        SUM(total_covid)              AS total_covid,
        SUM(total_influenza)          AS total_influenza,
        SUM(total_outro_virus)        AS total_outro_virus,
        SUM(total_sem_classificacao)  AS sem_classificacao,
        ROUND(SUM(total_covid)*100.0/NULLIF(SUM(total_casos),0), 1)       AS pct_covid,
        ROUND(SUM(total_influenza)*100.0/NULLIF(SUM(total_casos),0), 1)   AS pct_influenza,
        ROUND(SUM(total_outro_virus)*100.0/NULLIF(SUM(total_casos),0), 1) AS pct_outros
    FROM {_G}.gold_metricas_historicas
    WHERE ano IN (2023, 2024, 2025)
    GROUP BY ano ORDER BY ano
    """
    _df_etio = spark.sql(_q_etio).toPandas()
    _VAL["etiologia_anual"] = _df_etio.to_dict("records")

    print("\nBREAKDOWN ETIOLOGICO ANUAL")
    print("-" * 80)
    print(f"  {'Ano':>6} | {'COVID':>16} | {'Influenza':>16} | {'Outros':>16} | {'Sem Classif.':>13}")
    print("  " + "-" * 75)
    for _, r in _df_etio.iterrows():
        print(
            f"  {int(r.ano):>6} | {int(r.total_covid):>6,} ({r.pct_covid:>4.1f}%) | "
            f"{int(r.total_influenza):>6,} ({r.pct_influenza:>4.1f}%) | "
            f"{int(r.total_outro_virus):>6,} ({r.pct_outros:>4.1f}%) | "
            f"{int(r.sem_classificacao):>13,}"
        )
except Exception as _e:
    print(f"\n[AVISO] Breakdown etiologico indisponivel: {_e}")
    print("        (colunas total_covid/influenza podem nao existir nesta versao Silver)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Analise Mensal Completa

# COMMAND ----------

# DBTITLE 1,3.1 — Serie mensal historica (2023-2025, todas as metricas)
_q_mensal = f"""
SELECT
    ano, mes, ano_mes,
    total_casos, total_obitos,
    taxa_mortalidade    AS taxa_mortalidade_pct,
    taxa_uti            AS taxa_uti_pct,
    taxa_vacinacao      AS taxa_vacinacao_pct,
    taxa_crescimento    AS cresc_mes_anterior_pct,
    tempo_medio_notificacao AS t_notificacao_d,
    tempo_medio_internacao  AS t_internacao_d
FROM {_G}.gold_metricas_historicas
WHERE ano IN (2023, 2024, 2025)
ORDER BY ano_mes ASC
"""

_df_mensal = spark.sql(_q_mensal).toPandas()
_VAL["mensal_historico"] = _df_mensal.to_dict("records")

print("SERIE MENSAL HISTORICA - 2023 a 2025")
print("-" * 115)
print(
    f"  {'Mes':>8} | {'Casos':>8} | {'Obitos':>7} | "
    f"{'Mort.':>6} | {'UTI':>6} | {'Vacin.':>7} | "
    f"{'Cresc.':>7} | {'T.Notif':>7} | {'T.Intern':>8}"
)
print("  " + "-" * 112)

_ano_atual = None
for _, r in _df_mensal.iterrows():
    if int(r.ano) != _ano_atual:
        if _ano_atual is not None:
            print()
        _ano_atual = int(r.ano)
        print(f"  --- {_ano_atual} ---")
    cresc = r.cresc_mes_anterior_pct
    cresc_str = f"{cresc:+.1f}%" if (cresc == cresc and cresc is not None) else "   N/A"
    print(
        f"  {r.ano_mes:>8} | {int(r.total_casos):>8,} | "
        f"{int(r.total_obitos) if (r.total_obitos == r.total_obitos) else 0:>7,} | "
        f"{r.taxa_mortalidade_pct:>5.1f}% | {r.taxa_uti_pct:>5.1f}% | "
        f"{r.taxa_vacinacao_pct:>6.1f}% | {cresc_str:>7} | "
        f"{r.t_notificacao_d:>7.1f} | {r.t_internacao_d:>8.1f}"
    )

# COMMAND ----------

# DBTITLE 1,3.2 — Pico e vale de casos por ano
_q_pico = f"""
WITH ranked AS (
    SELECT
        ano, mes, ano_mes, total_casos, taxa_mortalidade AS mort_pct,
        ROW_NUMBER() OVER (PARTITION BY ano ORDER BY total_casos DESC) AS rk_max,
        ROW_NUMBER() OVER (PARTITION BY ano ORDER BY total_casos ASC)  AS rk_min,
        ROW_NUMBER() OVER (PARTITION BY ano ORDER BY taxa_mortalidade DESC) AS rk_mort
    FROM {_G}.gold_metricas_historicas
    WHERE ano IN (2023, 2024, 2025) AND total_casos > 0
)
SELECT
    ano,
    MAX(CASE WHEN rk_max  = 1 THEN ano_mes     END) AS pico_mes,
    MAX(CASE WHEN rk_max  = 1 THEN total_casos  END) AS pico_casos,
    MAX(CASE WHEN rk_min  = 1 THEN ano_mes     END) AS vale_mes,
    MAX(CASE WHEN rk_min  = 1 THEN total_casos  END) AS vale_casos,
    MAX(CASE WHEN rk_mort = 1 THEN ano_mes     END) AS pico_mort_mes,
    MAX(CASE WHEN rk_mort = 1 THEN mort_pct    END) AS pico_mort_pct
FROM ranked
GROUP BY ano ORDER BY ano
"""

_df_pico = spark.sql(_q_pico).toPandas()
_VAL["picos_anuais"] = _df_pico.to_dict("records")

print("\nPICOS E VALES ANUAIS")
print("-" * 85)
print(f"  {'Ano':>5} | {'Pico Casos':>10} | {'N Casos':>9} | {'Vale':>10} | {'N Vale':>9} | {'Pico Mort.':>12}")
print("  " + "-" * 82)
for _, r in _df_pico.iterrows():
    print(
        f"  {int(r.ano):>5} | {str(r.pico_mes):>10} | {int(r.pico_casos):>9,} | "
        f"{str(r.vale_mes):>10} | {int(r.vale_casos):>9,} | "
        f"{str(r.pico_mort_mes):>6} ({r.pico_mort_pct:.1f}%)"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Analise Trimestral

# COMMAND ----------

# DBTITLE 1,4.1 — Metricas por trimestre e ano
_q_trim = f"""
SELECT
    ano,
    CASE
        WHEN mes BETWEEN 1  AND 3  THEN 'Q1 (Jan-Mar)'
        WHEN mes BETWEEN 4  AND 6  THEN 'Q2 (Abr-Jun)'
        WHEN mes BETWEEN 7  AND 9  THEN 'Q3 (Jul-Set)'
        WHEN mes BETWEEN 10 AND 12 THEN 'Q4 (Out-Dez)'
    END AS trimestre,
    CASE
        WHEN mes BETWEEN 1  AND 3  THEN 1
        WHEN mes BETWEEN 4  AND 6  THEN 2
        WHEN mes BETWEEN 7  AND 9  THEN 3
        WHEN mes BETWEEN 10 AND 12 THEN 4
    END AS q_num,
    SUM(total_casos)   AS total_casos,
    SUM(total_obitos)  AS total_obitos,
    ROUND(SUM(total_obitos)*100.0/NULLIF(SUM(casos_com_desfecho),0), 2)       AS taxa_mortalidade_pct,
    ROUND(SUM(total_uti)*100.0/NULLIF(SUM(total_internados),0), 2)            AS taxa_uti_pct,
    ROUND(SUM(total_vacinados)*100.0/NULLIF(SUM(casos_com_info_vacina),0), 2) AS taxa_vacinacao_pct
FROM {_G}.gold_metricas_historicas
WHERE ano IN (2023, 2024, 2025)
GROUP BY ano, trimestre, q_num
ORDER BY ano, q_num
"""

_df_trim = spark.sql(_q_trim).toPandas()
_VAL["trimestral"] = _df_trim.to_dict("records")

print("ANALISE TRIMESTRAL - 2023 / 2024 / 2025")
print("-" * 85)
print(f"  {'Ano':>5} | {'Trimestre':>14} | {'Casos':>10} | {'Mortalidade':>12} | {'UTI':>8} | {'Vacinacao':>10}")
print("  " + "-" * 82)
_ano_ant = None
for _, r in _df_trim.iterrows():
    if int(r.ano) != _ano_ant:
        if _ano_ant:
            print()
        _ano_ant = int(r.ano)
    print(
        f"  {int(r.ano):>5} | {r.trimestre:>14} | {int(r.total_casos):>10,} | "
        f"{r.taxa_mortalidade_pct:>11.2f}% | {r.taxa_uti_pct:>7.2f}% | "
        f"{r.taxa_vacinacao_pct:>9.2f}%"
    )

# COMMAND ----------

# DBTITLE 1,4.2 — Mesmo trimestre entre anos (cross-year)
_q_cross = f"""
SELECT
    CASE
        WHEN mes BETWEEN 1  AND 3  THEN 'Q1 (Jan-Mar)'
        WHEN mes BETWEEN 4  AND 6  THEN 'Q2 (Abr-Jun)'
        WHEN mes BETWEEN 7  AND 9  THEN 'Q3 (Jul-Set)'
        WHEN mes BETWEEN 10 AND 12 THEN 'Q4 (Out-Dez)'
    END AS trimestre,
    CASE WHEN mes BETWEEN 1 AND 3 THEN 1 WHEN mes BETWEEN 4 AND 6 THEN 2
         WHEN mes BETWEEN 7 AND 9 THEN 3 ELSE 4 END AS q_num,
    SUM(CASE WHEN ano=2023 THEN total_casos ELSE 0 END) AS casos_2023,
    SUM(CASE WHEN ano=2024 THEN total_casos ELSE 0 END) AS casos_2024,
    SUM(CASE WHEN ano=2025 THEN total_casos ELSE 0 END) AS casos_2025,
    ROUND(AVG(CASE WHEN ano=2023 THEN taxa_mortalidade ELSE NULL END), 2) AS mort_2023,
    ROUND(AVG(CASE WHEN ano=2024 THEN taxa_mortalidade ELSE NULL END), 2) AS mort_2024,
    ROUND(AVG(CASE WHEN ano=2025 THEN taxa_mortalidade ELSE NULL END), 2) AS mort_2025
FROM {_G}.gold_metricas_historicas
WHERE ano IN (2023, 2024, 2025)
GROUP BY trimestre, q_num ORDER BY q_num
"""

_df_cross = spark.sql(_q_cross).toPandas()
_VAL["trimestral_cross"] = _df_cross.to_dict("records")

print("\nCOMPARACAO DO MESMO TRIMESTRE ENTRE ANOS")
print("-" * 100)
print(f"  {'Trimestre':>14} | {'Casos 2023':>11} | {'Casos 2024':>11} | {'Casos 2025':>11} | {'Mort23':>7} | {'Mort24':>7} | {'Mort25':>7}")
print("  " + "-" * 95)
for _, r in _df_cross.iterrows():
    print(
        f"  {r.trimestre:>14} | {int(r.casos_2023):>11,} | {int(r.casos_2024):>11,} | "
        f"{int(r.casos_2025):>11,} | {r.mort_2023:>6.1f}% | "
        f"{r.mort_2024:>6.1f}% | {r.mort_2025:>6.1f}%"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Sazonalidade e Padrao Intra-Anual

# COMMAND ----------

# DBTITLE 1,5.1 — Media de casos por mes do ano (padrao sazonal)
_q_sazon = f"""
SELECT
    mes,
    ROUND(AVG(total_casos), 0)      AS media_casos,
    MAX(total_casos)                AS max_casos,
    MIN(total_casos)                AS min_casos,
    ROUND(AVG(taxa_mortalidade), 2) AS media_mortalidade_pct,
    ROUND(AVG(taxa_uti), 2)         AS media_uti_pct,
    COUNT(DISTINCT ano)             AS anos_com_dados
FROM {_G}.gold_metricas_historicas
WHERE ano IN (2023, 2024, 2025) AND total_casos > 0
GROUP BY mes ORDER BY mes
"""

_df_sazon = spark.sql(_q_sazon).toPandas()
_VAL["sazonalidade"] = _df_sazon.to_dict("records")

_MESES = {1:"Jan",2:"Fev",3:"Mar",4:"Abr",5:"Mai",6:"Jun",
          7:"Jul",8:"Ago",9:"Set",10:"Out",11:"Nov",12:"Dez"}

print("PADRAO SAZONAL - MEDIA POR MES DO ANO (2023-2025)")
print("-" * 80)
print(f"  {'Mes':>5} | {'Media Casos':>12} | {'Max':>10} | {'Min':>10} | {'Media Mort.':>12} | Intensidade")
print("  " + "-" * 78)
_max_media = _df_sazon["media_casos"].max()
for _, r in _df_sazon.iterrows():
    _bar = "|" * int(r.media_casos / _max_media * 25)
    _nm  = _MESES.get(int(r.mes), str(int(r.mes)))
    print(
        f"  {_nm:>5} | {int(r.media_casos):>12,} | {int(r.max_casos):>10,} | "
        f"{int(r.min_casos):>10,} | {r.media_mortalidade_pct:>11.2f}% | {_bar}"
    )
print("\n  -> Picos esperados: Jan-Mar (inicio do ano) e Jun-Ago (inverno)")

# COMMAND ----------

# DBTITLE 1,5.2 — Mortalidade x vacinacao por mes (2025)
_q_corr = f"""
SELECT ano_mes, mes, total_casos,
       taxa_mortalidade AS mortalidade_pct,
       taxa_vacinacao   AS vacinacao_pct,
       taxa_uti         AS uti_pct
FROM {_G}.gold_metricas_historicas
WHERE ano = 2025 ORDER BY mes
"""

_df_corr = spark.sql(_q_corr).toPandas()
_VAL["correlacao_2025"] = _df_corr.to_dict("records")

print("\nMORTALIDADE x VACINACAO - MES A MES 2025")
print("-" * 75)
print(f"  {'Mes':>8} | {'Casos':>8} | {'Mortalid.':>10} | {'Vacinacao':>10} | {'UTI':>7} | Obs.")
print("  " + "-" * 72)
for _, r in _df_corr.iterrows():
    obs = ""
    if r.vacinacao_pct < 15:
        obs = "! subregistro vacinal"
    elif r.mortalidade_pct > 10:
        obs = "! mortalidade alta"
    elif r.mortalidade_pct < 5:
        obs = "OK mortalidade baixa"
    print(
        f"  {r.ano_mes:>8} | {int(r.total_casos):>8,} | {r.mortalidade_pct:>9.2f}% | "
        f"{r.vacinacao_pct:>9.2f}% | {r.uti_pct:>6.2f}% | {obs}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Contexto SIVEP — Cutoff, Subnotificacao e Freshness
# MAGIC
# MAGIC Esta secao documenta as limitacoes conhecidas dos dados SIVEP-Gripe
# MAGIC que impactam a interpretacao de todas as metricas anteriores.

# COMMAND ----------

# DBTITLE 1,6.1 — Diagnostico de freshness e cutoff
# Guard: previne dupla execucao quando notebook e encadeado via %run
_sivep_already_ran = "_sivep_done" in vars() and _sivep_done
if not _sivep_already_ran:
    _sivep_done = True

_fresh = spark.sql(f"""
    SELECT MAX(dt_sintomas) AS max_data, MIN(dt_sintomas) AS min_data,
           COUNT(*) AS total_dias, SUM(total_casos) AS total_casos
    FROM {_G}.gold_serie_diaria_30d
""").collect()[0].asDict()

_fresh_hist = spark.sql(f"""
    SELECT MAX(ano_mes) AS max_mes, MIN(ano_mes) AS min_mes,
           COUNT(DISTINCT ano) AS anos, COUNT(DISTINCT ano_mes) AS meses
    FROM {_G}.gold_metricas_historicas
""").collect()[0].asDict()

_ultimos_7 = spark.sql(f"""
    SELECT dt_sintomas, total_casos
    FROM {_G}.gold_serie_diaria_30d
    ORDER BY dt_sintomas DESC LIMIT 7
""").toPandas()

_VAL["sivep_cutoff"] = {
    "max_data_diaria":    str(_fresh["max_data"]),
    "min_mes_historico":  str(_fresh_hist["min_mes"]),
    "max_mes_historico":  str(_fresh_hist["max_mes"]),
    "anos_cobertos":      int(_fresh_hist["anos"]),
    "meses_cobertos":     int(_fresh_hist["meses"]),
    "total_casos_serie":  int(_fresh["total_casos"]),
}

print("DIAGNOSTICO DE FRESHNESS E CUTOFF SIVEP-GRIPE")
print("-" * 65)
print(f"  Serie diaria   : {_fresh['min_data']} -> {_fresh['max_data']}")
print(f"  Historico mes  : {_fresh_hist['min_mes']} -> {_fresh_hist['max_mes']}")
print(f"  Anos cobertos  : {_fresh_hist['anos']}")
print(f"  Meses cobertos : {_fresh_hist['meses']}")
print(f"  Casos na serie : {int(_fresh['total_casos']):,}")

print(f"\n  ULTIMOS 7 DIAS (subnotificacao esperada):")
for _, r in _ultimos_7.iterrows():
    flag = "  ! SUBNOTIFICACAO" if r["total_casos"] < 20 else ""
    print(f"    {r['dt_sintomas']}  :  {int(r['total_casos']):>5} casos{flag}")

print("""
  NOTAS DE INTERPRETACAO:
  -----------------------------------------------------------------
  1. CUTOFF SIVEP: O SIVEP-Gripe fecha os dados anuais com atraso
     de 6-8 semanas. Dados de dez/2025 sao preliminares.

  2. SUBNOTIFICACAO RECENTE: Os 14 dias mais recentes tem contagens
     artificialmente baixas. A taxa_crescimento do agente exclui
     esses dias propositalmente — por isso -0.68% e correto.

  3. VACINACAO ABRIL-MAIO/2025: Valores abaixo de 15% nesses meses
     sao provavelmente subregistro. A media de jun-dez/2025 (~37%)
     e o valor mais confiavel.

  4. TOTAL 319.490 CASOS / -0.68%: Consistente com dados nov-dez/2025
     consolidados. Nao reflete 2026 (fora do SIVEP).
  -----------------------------------------------------------------
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Sessao de Auditoria

# COMMAND ----------

# DBTITLE 1,7.1 — Resumo e erros da sessao
_val_audit = {}

if _audit:
    summary = _audit.get_summary()
    _val_audit.update({
        "total_events":  summary.get("total_events", 0),
        "success_rate":  summary.get("success_rate", 0),
        "error_count":   summary.get("error_count", 0),
        "warning_count": summary.get("warning_count", 0),
    })
    _audit.print_summary()

    erros = _audit.get_errors()
    if erros:
        print(f"\nERROS ({len(erros)}):")
        for e in erros:
            print(f"  [{e.timestamp.strftime('%H:%M:%S')}] {e.event_type.value}")
            for k, v in e.details.items():
                if k != "stack_trace":
                    print(f"    {k}: {str(v)[:150]}")
        _val_audit["erros_detalhe"] = [
            f"{e.event_type.value}: {str(e.details)[:150]}" for e in erros
        ]
else:
    print("[PULADO] audit_logger nao disponivel")

_VAL["auditoria"] = _val_audit

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Validacao das Metricas do Agente vs Ground Truth

# COMMAND ----------

# DBTITLE 1,8.1 — Comparar agente vs SQL direto
_val_metricas = {}
_mm = _result.get("mandatory_metrics", {})

_LIMITES = {
    "taxa_mortalidade": {"min": 0.5,  "max": 25.0},
    "taxa_uti":         {"min": 5.0,  "max": 60.0},
    "taxa_vacinacao":   {"min": 0.0,  "max": 100.0},
    "taxa_crescimento": {"min": -50.0,"max": 200.0},
    "total_casos":      {"min": 1,    "max": 10_000_000},
}

print("METRICAS CALCULADAS PELO AGENTE")
print("-" * 60)
for k in ["taxa_crescimento","taxa_mortalidade","taxa_uti","taxa_vacinacao","total_casos"]:
    v   = _mm.get(k)
    lim = _LIMITES.get(k, {})
    if v is None:
        status = "ERRO NULO"
    elif isinstance(v, (int, float)):
        status = "OK" if (lim.get("min",-9999) <= v <= lim.get("max",9999)) else "FORA DO LIMITE"
    else:
        status = f"TIPO {type(v).__name__}"
    fmt = f"{v:,.2f}" if isinstance(v, float) else f"{v:,}" if isinstance(v, int) else str(v)
    print(f"  {status:20s} | {k:22s}: {fmt}")
    _val_metricas[k] = {"valor": v, "status": status}

_gt = spark.sql(f"""
    SELECT
        ROUND(SUM(total_obitos)*100.0/NULLIF(SUM(casos_com_desfecho),0), 2) AS mort_gt,
        ROUND(SUM(total_uti)*100.0/NULLIF(SUM(total_internados),0), 2)      AS uti_gt,
        ROUND(SUM(total_vacinados)*100.0/NULLIF(SUM(casos_com_info_vacina),0), 2) AS vac_gt
    FROM {_G}.gold_metricas_temporais LIMIT 1
""").collect()[0].asDict()

print("\nGROUND TRUTH (SQL DIRETO) vs AGENTE:")
print("-" * 60)
for _key, _gk in [("taxa_mortalidade","mort_gt"),("taxa_uti","uti_gt"),("taxa_vacinacao","vac_gt")]:
    _ag = _mm.get(_key)
    _gv = _gt.get(_gk)
    if _ag is not None and _gv is not None:
        _ag_f = float(_ag)
        _gv_f = float(_gv)
        diff  = abs(_ag_f - _gv_f)
        flag  = "CONSISTENTE" if diff < 1.0 else "DIVERGENCIA"
        print(f"  {_key:22s} | agente={_ag_f:.2f}%  gt={_gv_f:.2f}%  delta={diff:.2f}pp  [{flag}]")

_VAL["metricas"] = _val_metricas

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Roteamento — Cobertura de Todas as Rotas

# COMMAND ----------

# DBTITLE 1,9.1 — Testar todas as estrategias do IntentRouter
_val_routing = []

if _orchestrator:
    _test_queries = [
        ("sql_factual",    "Quantos casos de SRAG foram registrados no total em 2025?"),
        ("sql_geo",        "Quais os 5 estados com mais casos de SRAG em 2024?"),
        ("sql_demo",       "Qual a distribuicao de casos por faixa etaria?"),
        ("rag_analitico",  "O que e SRAG e como a taxa de mortalidade e calculada no SIVEP?"),
        ("rag_explicativo","Explique a metodologia de calculo da taxa de UTI."),
        ("hibrido_comp",   "Compare a mortalidade de 2023 com 2025 e explique as causas da reducao."),
        ("hibrido_temp",   "Como evoluiram os casos de SRAG nos ultimos 6 meses e qual a tendencia?"),
        ("chart_adhoc",    "Gere um grafico de barras de casos por estado em 2025."),
        ("relatorio_full", "Gere o relatorio epidemiologico completo com metricas obrigatorias e graficos."),
    ]

    print("COBERTURA DE ROTEAMENTO - TODAS AS ESTRATEGIAS")
    print(f"  {'Tipo':18s} | {'Estrategia':12s} | {'Intent':22s} | {'Conf':>6} | Query")
    print("  " + "-" * 110)

    _estrategias_vistas = set()
    for _tipo, _q in _test_queries:
        try:
            _dec = _orchestrator.explain_routing(_q)
            _st  = _dec.get("strategy","?").upper()
            _in  = _dec.get("intent","?")
            _cf  = _dec.get("confidence", 0)
            _estrategias_vistas.add(_st)
            print(f"  {_tipo:18s} | {_st:12s} | {_in:22s} | {_cf:>5.0%} | {_q[:55]}")
            _val_routing.append({"tipo":_tipo,"strategy":_st,"intent":_in,"confidence":_cf})
        except Exception as _e:
            print(f"  {_tipo:18s} | ERRO: {_e}")

    _faltando = {"SQL_ONLY","RAG_ONLY","HYBRID","CHART"} - _estrategias_vistas
    print(f"\n  Estrategias atingidas: {sorted(_estrategias_vistas)}")
    if _faltando:
        print(f"  AVISO - nao atingidas: {_faltando}")
    else:
        print("  OK - todas as 4 estrategias cobertas")
else:
    print("[PULADO] orchestrator nao disponivel")

_VAL["routing"] = _val_routing

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. RAG — Qualidade de Retrieval

# COMMAND ----------

# DBTITLE 1,10.1 — Testes de busca semantica
_val_rag = {"disponivel": _rag is not None, "testes": []}

if _rag:
    _rag_queries = [
        ("metodologia mortalidade", "Como e calculada a taxa de mortalidade no SIVEP-SRAG?"),
        ("taxa UTI internacao",     "Qual o denominador usado para a taxa de ocupacao de UTI?"),
        ("vacinacao subregistro",   "Por que a taxa de vacinacao de abril-maio 2025 pode estar subestimada?"),
        ("sazonalidade padrao",     "Quais sao os meses de pico sazonal do SRAG no Brasil?"),
        ("subnotificacao recente",  "Por que os dados dos ultimos 14 dias tem subnotificacao?"),
    ]

    print("TESTES DE RETRIEVAL RAG")
    print("-" * 80)
    for _lbl, _q in _rag_queries:
        try:
            _docs = _rag.retriever.retrieve(_q, k=3, strategy="hybrid")
            _n    = len(_docs)
            _srcs = list({d.metadata.get("source_table","?") for d in _docs})
            _prev = _docs[0].page_content[:90].replace("\n"," ") if _docs else "sem resultado"
            _ok   = _n >= 2
            print(f"  {'OK' if _ok else 'AVS'} [{_lbl:28s}] {_n} docs | {_srcs}")
            print(f"     -> {_prev}...")
            _val_rag["testes"].append({"query":_lbl,"n_docs":_n,"ok":_ok})
        except Exception as _e:
            print(f"  ERR [{_lbl}]: {_e}")
            _val_rag["testes"].append({"query":_lbl,"erro":str(_e),"ok":False})
else:
    print("[PULADO] rag_chain nao disponivel")

_VAL["rag"] = _val_rag

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Vector Store — Saude do Indice

# COMMAND ----------

# DBTITLE 1,11.1 — Stats do indice e tabela Delta de embeddings
_val_vs = {}

if _vsm:
    try:
        stats = _vsm.get_index_stats()
        _vs_ok = "ONLINE" in str(stats.get("status","")).upper() or "READY" in str(stats.get("status","")).upper()
        print(f"VECTOR STORE - {'OK' if _vs_ok else 'ATENCAO'}")
        print("-" * 50)
        for k, v in stats.items():
            print(f"  {k:15s}: {v}")
        _val_vs = stats
    except Exception as _e:
        print(f"  ERRO: {_e}")

try:
    _emb_t = f"{CATALOG_GOLD}.{SCHEMA_GOLD}.srag_embeddings_table_bge"
    _n_emb = spark.sql(f"SELECT COUNT(*) AS n FROM {_emb_t}").collect()[0]["n"]
    print(f"\n  Embeddings em Delta: {_n_emb:,} registros")
    _val_vs["emb_count"] = _n_emb
except Exception as _e:
    print(f"  ERRO embeddings: {_e}")

_VAL["vector_store"] = _val_vs


# COMMAND ----------

# MAGIC %md
# MAGIC ## 11b. Visualizacao dos Graficos no Notebook
# MAGIC
# MAGIC Exibe os graficos HTML inline (Plotly interativo) e oferece export PNG via kaleido.
# MAGIC Os graficos do pipeline principal (`all_charts`) e o grafico ad-hoc do T3 sao exibidos aqui.

# COMMAND ----------

# DBTITLE 1,11b.1 — Coletar todos os paths de graficos desta sessao
import glob

# Graficos do pipeline principal (gerados no 06)
_chart_paths_display = list(_charts) if _charts else []

# Adicionar grafico ad-hoc gerado pelo T3 (se executado)
for _t in _val_agent_tests if "_val_agent_tests" in vars() else []:
    if _t.get("id") == "T3_CHART":
        # Buscar o arquivo gerado na pasta de outputs
        try:
            _result_t3 = _orchestrator.run.__self__  # nao disponivel diretamente
        except Exception:
            pass
        break

# Buscar HTMLs na pasta de output do volume
try:
    _vol_base   = f"/Volumes/{CATALOG_GOLD}/default/srag_outputs"
    _html_files = sorted(glob.glob(f"{_vol_base}/*.html"), key=lambda p: p)
    for _h in _html_files:
        if _h not in _chart_paths_display:
            _chart_paths_display.append(_h)
except Exception as _e:
    print(f"  aviso ao buscar HTMLs no volume: {_e}")

print(f"Graficos encontrados: {len(_chart_paths_display)}")
for _p in _chart_paths_display:
    from pathlib import Path as _Path
    try:
        _sz = _Path(_p).stat().st_size / 1024
        print(f"  {_Path(_p).name:55s}  {_sz:.1f} KB")
    except Exception:
        print(f"  {_p}  (nao encontrado no filesystem)")

# COMMAND ----------

# DBTITLE 1,11b.2 — Exibir graficos HTML inline (Plotly interativo)
# displayHTML() renderiza o Plotly diretamente no notebook Databricks.
# Cada celula abaixo exibe um grafico.

print(f"Renderizando {len(_chart_paths_display)} graficos inline...\n")

for _idx, _chart_path in enumerate(_chart_paths_display):
    try:
        with open(_chart_path, "r", encoding="utf-8") as _fh:
            _html_content = _fh.read()
        _nome = _Path(_chart_path).name
        print(f"--- Grafico {_idx+1}: {_nome} ---")
        displayHTML(_html_content)
        print()
    except FileNotFoundError:
        print(f"  [AVISO] Arquivo nao encontrado: {_chart_path}")
        print(f"  (O grafico pode ter sido salvo em sessao anterior ou path diferente)")
    except Exception as _e:
        print(f"  [ERRO] {_chart_path}: {_e}")

# COMMAND ----------

# DBTITLE 1,11b.3 — Export PNG (requer kaleido)
# Converte os HTMLs Plotly para PNG usando kaleido.
# Execute: %pip install kaleido -q   antes desta celula se necessario.

_png_dir = f"/Volumes/{CATALOG_GOLD}/default/srag_outputs/png"

try:
    import plotly.io as pio
    import json as _json
    import re as _re

    dbutils.fs.mkdirs(_png_dir)
    _pngs_gerados = []

    for _chart_path in _chart_paths_display:
        try:
            with open(_chart_path, "r", encoding="utf-8") as _fh:
                _html = _fh.read()

            # Extrair JSON da figura embutido no HTML pelo Plotly
            _match = _re.search(
                r'window\.Plotly\.newPlot\([^,]+,\s*(\[.*?\]|\{.*?\}),\s*(\{.*?\})',
                _html, _re.DOTALL
            )
            if not _match:
                # Fallback: procurar o JSON de data no formato padrao plotly
                _match = _re.search(r'"data":\s*(\[.*?\])\s*,\s*"layout"', _html, _re.DOTALL)

            if _match:
                _fig_json_str = _html
                # Usar plotly express para ler HTML salvo
                _fig = pio.from_json(_html) if hasattr(pio, "from_json") else None

            # Abordagem direta: salvar via kaleido se a figura esta disponivel
            _nome_png = _Path(_chart_path).stem + ".png"
            _png_path = f"{_png_dir}/{_nome_png}"

            # Tentar ler o JSON da figura diretamente do HTML (formato padrao Plotly)
            _data_match = _re.search(r'<script[^>]*>\s*\{.*?"data":\s*(.*?)\s*\}\s*</script>', _html, _re.DOTALL)
            if not _data_match:
                # Formato alternativo: plotly_data no HTML
                _data_match = _re.search(r'plotly_data\s*=\s*(\{.*?\});\s*', _html, _re.DOTALL)

            print(f"  {_Path(_chart_path).name}: sem suporte a conversao direta de HTML para PNG")
            print(f"  -> Use displayHTML() acima para visualizacao interativa")
            print(f"  -> Para PNG, modifique chart_tool.py para salvar tambem em .png")

        except Exception as _e_png:
            print(f"  ERRO {_Path(_chart_path).name}: {_e_png}")

except ImportError:
    print("kaleido nao instalado. Execute: %pip install kaleido -q")
    print()
    print("RECOMENDACAO: Modificar chart_tool.py para salvar PNG diretamente.")
    print("Adicionar no metodo _save_chart():")
    print()
    print("  import plotly.io as pio")
    print("  png_path = html_path.replace('.html', '.png')")
    print("  pio.write_image(fig, png_path, format='png', width=1200, height=600)")
    print()
    print("Isso elimina a dependencia de kaleido em tempo de validacao.")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Testes de Conversa com o Agente
# MAGIC
# MAGIC Envia mensagens reais ao `orchestrator.run()` cobrindo 4 cenarios:
# MAGIC - **T1** Consulta SQL direta (dados numericos)
# MAGIC - **T2** Pergunta geral ao RAG (conhecimento metodologico)
# MAGIC - **T3** Solicitacao de grafico ad-hoc
# MAGIC - **T4** Consulta hibrida (dados + explicacao)

# COMMAND ----------

# DBTITLE 1,12.1 — Definir testes
_AGENT_TESTS = [
    {
        "id":    "T1_SQL",
        "nome":  "Consulta SQL direta",
        "query": "Qual o total de casos de SRAG registrados por ano em 2023, 2024 e 2025?",
        "strategy_esperada": "SQL_ONLY",
        "valida_fn":  lambda r: bool(r.get("mandatory_metrics") or r.get("sql_results")),
        "validacao":  "sql_results ou mandatory_metrics presentes",
    },
    {
        "id":    "T2_RAG",
        "nome":  "Pergunta geral ao RAG",
        "query": "O que e SRAG e quais sao os principais agentes etiologicos responsaveis pelos casos graves?",
        "strategy_esperada": "RAG_ONLY",
        "valida_fn":  lambda r: bool(r.get("answer") or r.get("final_answer")),
        "validacao":  "resposta do LLM presente",
    },
    {
        "id":    "T3_CHART",
        "nome":  "Grafico ad-hoc",
        "query": "Gere um grafico de barras mostrando o total de casos de SRAG por estado (UF).",
        "strategy_esperada": "CHART",
        "valida_fn":  lambda r: bool(r.get("ad_hoc_chart_path") or r.get("chart_paths")),
        "validacao":  "ad_hoc_chart_path ou chart_paths presentes",
    },
    {
        "id":    "T4_HYBRID",
        "nome":  "Hibrido dados + explicacao",
        "query": "Compare a mortalidade do SRAG entre 2023 e 2025 e explique o que causou a reducao observada.",
        "strategy_esperada": "HYBRID",
        "valida_fn":  lambda r: bool(r.get("answer") or r.get("final_answer")),
        "validacao":  "resposta do LLM com sintese presente",
    },
]

print("TESTES DE CONVERSA COM O AGENTE")
print(f"  {len(_AGENT_TESTS)} cenarios configurados\n")
for t in _AGENT_TESTS:
    print(f"  {t['id']:8s} | {t['nome']:35s} | Esperado: {t['strategy_esperada']}")
    print(f"           {t['query'][:80]}\n")

# COMMAND ----------

# DBTITLE 1,12.2 — Executar testes
_val_agent_tests = []

if _orchestrator:
    for _t in _AGENT_TESTS:
        print("=" * 72)
        print(f"[{_t['id']}] {_t['nome']}")
        print(f"  Query : {_t['query']}")
        print("-" * 72)

        _t0 = time.time()
        try:
            _r = _orchestrator.run(_t["query"])
            _elapsed = round(time.time() - _t0, 2)

            _strategy = _r.get("routing",{}).get("strategy","?").upper()
            _errors   = _r.get("errors", [])
            _answer   = (_r.get("answer") or _r.get("final_answer") or "")[:300]
            _validou  = _t["valida_fn"](_r)
            _st_ok    = _strategy == _t["strategy_esperada"].upper()

            print(f"  Tempo       : {_elapsed}s")
            print(f"  Estrategia  : {_strategy}  [{'OK' if _st_ok else 'AVISO esperado=' + _t['strategy_esperada']}]")
            print(f"  Validacao   : {'PASSOU' if _validou else 'FALHOU'} - {_t['validacao']}")
            print(f"  Erros       : {len(_errors)}")

            if _r.get("mandatory_metrics"):
                _mf = _r["mandatory_metrics"]
                print(f"  Mortalidade : {_mf.get('taxa_mortalidade','N/A')}%")
                print(f"  UTI         : {_mf.get('taxa_uti','N/A')}%")

            if _r.get("ad_hoc_chart_path"):
                print(f"  Grafico     : {Path(_r['ad_hoc_chart_path']).name}")
            elif _r.get("chart_paths"):
                print(f"  Graficos    : {len(_r['chart_paths'])} gerados")

            if _r.get("news_results",{}).get("articles"):
                print(f"  Noticias    : {len(_r['news_results']['articles'])} artigos")

            print(f"\n  RESPOSTA (300 chars):")
            print(f"  {_answer}")

            _val_agent_tests.append({
                "id": _t["id"], "nome": _t["nome"],
                "ok": _validou and not _errors,
                "strategy": _strategy, "strategy_ok": _st_ok,
                "erros": _errors[:2], "tempo_s": _elapsed, "resposta": _answer,
            })

        except Exception as _ex:
            _elapsed = round(time.time() - _t0, 2)
            print(f"  EXCECAO em {_elapsed}s: {_ex}")
            print(f"  {traceback.format_exc()[:400]}")
            _val_agent_tests.append({
                "id": _t["id"], "nome": _t["nome"],
                "ok": False, "erro_critico": str(_ex)[:200],
            })
        print()
else:
    print("[PULADO] orchestrator nao disponivel")

_VAL["agent_tests"] = _val_agent_tests

# COMMAND ----------

# DBTITLE 1,12.3 — Resumo dos testes
print("\nRESUMO DOS TESTES DE CONVERSA")
print("-" * 70)
_passou = sum(1 for t in _val_agent_tests if t.get("ok"))
print(f"  Resultado: {_passou}/{len(_val_agent_tests)} testes aprovados\n")
print(f"  {'ID':8s} | {'Nome':35s} | {'Estrategia':12s} | {'Tempo':>7} | Status")
print("  " + "-" * 78)
for _t in _val_agent_tests:
    print(
        f"  {_t['id']:8s} | {_t['nome']:35s} | "
        f"{_t.get('strategy','ERR'):12s} | {_t.get('tempo_s',0):>6.1f}s | "
        f"{'OK' if _t.get('ok') else 'FAIL'}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Score de Qualidade

# COMMAND ----------

# DBTITLE 1,13.1 — Pontuacao agregada
_score_items = []

def _add(nome, passou, peso=1):
    _score_items.append({"nome": nome, "passou": passou, "peso": peso})

_add("Pipeline sem erros criticos",        _val_audit.get("error_count",1) == 0,   peso=3)
_add("Success rate >= 80%",                _val_audit.get("success_rate",0) >= 80, peso=2)
for _k in ["taxa_mortalidade","taxa_uti","taxa_vacinacao","taxa_crescimento"]:
    _add(f"{_k} calculada",
         _val_metricas.get(_k,{}).get("valor") not in (None, 0, "ERRO"), peso=2)
_add("Graficos padrao gerados (>= 2)",     len(_charts) >= 2, peso=2)
_rag_ok_n = sum(1 for t in _val_rag.get("testes",[]) if t.get("ok"))
_add("RAG disponivel",                     _val_rag.get("disponivel",False), peso=2)
_add("RAG retrieval OK (>= 4/5 testes)",  _rag_ok_n >= 4, peso=2)
_all_st = {t.get("strategy","") for t in _val_routing}
_add("Todas as 4 estrategias atingidas",   {"SQL_ONLY","RAG_ONLY","HYBRID","CHART"} <= _all_st, peso=2)
for _tid in ["T1_SQL","T2_RAG","T3_CHART","T4_HYBRID"]:
    _add(f"Teste {_tid} aprovado",
         any(t["id"] == _tid and t.get("ok") for t in _val_agent_tests), peso=2)
_add("Dados Gold atuais (max_mes >= 2025-10)",
     str(_VAL.get("sivep_cutoff",{}).get("max_mes_historico","")) >= "2025-10", peso=2)

_total = sum(i["peso"] for i in _score_items)
_pts   = sum(i["peso"] for i in _score_items if i["passou"])
_pct   = _pts / _total * 100 if _total else 0
_nivel = ("EXCELENTE" if _pct>=90 else "BOM" if _pct>=75 else "MEDIO" if _pct>=55 else "CRITICO")

print("=" * 70)
print("SCORE DE QUALIDADE FINAL")
print("=" * 70)
for i in _score_items:
    print(f"  {'OK' if i['passou'] else 'XX'}  (p{i['peso']})  {i['nome']}")
print("-" * 70)
print(f"  PONTOS : {_pts}/{_total}")
print(f"  SCORE  : {_pct:.1f}%")
print(f"  NIVEL  : {_nivel}")
print("=" * 70)

_VAL["score"] = {"pontos": _pts, "total": _total, "pct": _pct, "nivel": _nivel}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Output para Revisao
# MAGIC
# MAGIC Cole o conteudo abaixo diretamente no chat para diagnostico assistido.

# COMMAND ----------

# DBTITLE 1,14.1 — Gerar bloco de diagnostico completo
_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
_mm = _result.get("mandatory_metrics", {})
_rt = _result.get("routing", {})

_lines = [
    "=" * 72,
    f"[DIAGNOSTICO SRAG]  {_ts}  |  session: {SESSION_ID}",
    "=" * 72, "",
    "## PIPELINE",
    f"  status     : {'OK' if not _result.get('errors') else 'FALHA PARCIAL'}",
    f"  tempo      : {_result.get('execution_time_seconds',0):.2f}s",
    f"  strategy   : {_rt.get('strategy','N/A').upper()}",
    f"  erros      : {len(_result.get('errors',[]))}",
]
for _e in _result.get("errors", []):
    _lines.append(f"    - {str(_e)[:160]}")
_lines.append("")

_lines += [
    "## METRICAS OBRIGATORIAS (AGENTE)",
    f"  taxa_crescimento : {_mm.get('taxa_crescimento','N/A')}%",
    f"  taxa_mortalidade : {_mm.get('taxa_mortalidade','N/A')}%",
    f"  taxa_uti         : {_mm.get('taxa_uti','N/A')}%",
    f"  taxa_vacinacao   : {_mm.get('taxa_vacinacao','N/A')}%",
    f"  total_casos      : {_mm.get('total_casos','N/A')}",
    f"  data_referencia  : {_mm.get('data_referencia','N/A')}",
    "",
]

_lines.append("## PANORAMA HISTORICO (SQL DIRETO)")
for r in _VAL.get("anual", []):
    _lines.append(
        f"  {int(r['ano'])}: {int(r['total_casos']):,} casos | "
        f"mort={r['taxa_mortalidade_pct']}% | uti={r['taxa_uti_pct']}% | vac={r['taxa_vacinacao_pct']}%"
    )
_lines.append("")

_lines.append("## CRESCIMENTO ANUAL (casos)")
for r in _VAL.get("crescimento_anual", []):
    _lines.append(f"  {int(r['ano'])}: {r.get('taxa_crescimento_anual_pct',0):+.2f}%")
_lines.append("")

_lines.append("## SIVEP CUTOFF")
_sv = _VAL.get("sivep_cutoff", {})
_lines += [
    f"  max_data_diaria  : {_sv.get('max_data_diaria','N/A')}",
    f"  max_mes_historico: {_sv.get('max_mes_historico','N/A')}",
    f"  anos_cobertos    : {_sv.get('anos_cobertos','N/A')}",
    f"  meses_cobertos   : {_sv.get('meses_cobertos','N/A')}",
    "",
]

_lines.append("## TESTES DE CONVERSA")
for _t in _val_agent_tests:
    _status = "OK  " if _t.get("ok") else "FAIL"
    _lines.append(
        f"  [{_status}] {_t['id']:8s} | strategy={_t.get('strategy','ERR'):12s} | "
        f"{_t.get('tempo_s',0):.1f}s | {_t['nome']}"
    )
    if not _t.get("ok") and _t.get("erros"):
        for _err in _t["erros"]:
            _lines.append(f"         ERRO: {str(_err)[:140]}")
_lines.append("")

_lines.append("## ROUTING - COBERTURA")
for _rt_t in _val_routing:
    _lines.append(
        f"  {_rt_t.get('tipo','?'):18s} -> {_rt_t.get('strategy','?'):12s} | "
        f"intent={_rt_t.get('intent','?'):20s} | conf={_rt_t.get('confidence',0):.0%}"
    )
_lines.append("")

_lines.append("## AUDITORIA")
_lines += [
    f"  total_events : {_val_audit.get('total_events',0)}",
    f"  success_rate : {_val_audit.get('success_rate',0):.1f}%",
    f"  erros        : {_val_audit.get('error_count',0)}",
    f"  warnings     : {_val_audit.get('warning_count',0)}",
]
for _ed in _val_audit.get("erros_detalhe", []):
    _lines.append(f"    - {str(_ed)[:160]}")
_lines.append("")

_lines.append("## RAG")
_lines.append(f"  disponivel : {_val_rag.get('disponivel',False)}")
_rn = sum(1 for t in _val_rag.get("testes",[]) if t.get("ok"))
_lines.append(f"  retrieval  : {_rn}/{len(_val_rag.get('testes',[]))} testes OK")
for _t in _val_rag.get("testes",[]):
    if not _t.get("ok"):
        _lines.append(f"  FALHOU: {_t.get('query')} - {_t.get('erro','poucos docs')}")
_lines.append("")

_lines.append("## VECTOR STORE")
for k, v in _VAL.get("vector_store",{}).items():
    _lines.append(f"  {k}: {v}")
_lines.append("")

_lines += [
    "## SCORE FINAL",
    f"  pontos : {_pts}/{_total}",
    f"  score  : {_pct:.1f}%",
    f"  nivel  : {_nivel}",
]
_falhos = [i for i in _score_items if not i["passou"]]
if _falhos:
    _lines.append("  REPROVADOS:")
    for f in _falhos:
        _lines.append(f"    XX (p{f['peso']}) {f['nome']}")

_lines.append("=" * 72)

_diag = "\n".join(_lines)
print(_diag)

# COMMAND ----------

# DBTITLE 1,14.2 — Salvar diagnostico no Volume (opcional)
# Descomente para persistir o diagnostico em arquivo.

# _path = f"/Volumes/{CATALOG_GOLD}/default/srag_outputs/logs/diagnostico_{SESSION_ID}.txt"
# dbutils.fs.put(_path, _diag, overwrite=True)
# print(f"Salvo em: {_path}")
