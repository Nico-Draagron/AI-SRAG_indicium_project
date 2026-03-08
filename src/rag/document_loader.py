"""
Document Loader — Ingestão de Tabelas Gold para RAG
====================================================

Responsabilidade: ler as tabelas Gold do catálogo Databricks e converter cada
registro em um SRAGDocument com texto semântico pronto para embedding e retrieval.

Fontes e estratégia de chunking
---------------------------------
gold_rag_kpi_fatos (fonte primária)
    O notebook 05_gold_base_conhecimento_rag já gera o campo `text` em linguagem
    natural. Cada linha é 1 documento independente — sem builder de conteúdo
    adicional. 1 linha → 1 SRAGDocument.

gold_rag_dicionario_regras
    Regras epidemiológicas formais do pipeline SIVEP-Gripe. Fornece ao LLM o
    contexto metodológico para interpretar os dados (critérios de classificação,
    definições de caso, regras de imputação). 1 regra → 1 SRAGDocument.

gold_metricas_temporais
    Agregados mensais de casos, mortalidade, UTI, vacinação e crescimento.
    Cada mês produz 1 documento com todas as métricas daquele período — chunking
    por mês preserva a coerência temporal para queries como "como foi março de 2024".

gold_metricas_geograficas
    Perfis por UF: total de casos, ranking nacional, taxas de mortalidade e UTI.
    1 UF → 1 documento — permite retrieval filtrado por estado via semantic_type.

gold_metricas_demograficas
    Perfis por faixa etária e sexo. 1 combinação (faixa, sexo) → 1 documento.
    Ordenados por ordem_faixa_etaria para consistência do índice.

Decisões de design
------------------
_build_resumo_geral_content() removido
    O método referenciava colunas de gold_resumo_geral (categoria, metrica, valor,
    unidade, descricao, escopo, data_snapshot) — tabela substituída por
    gold_rag_kpi_fatos no notebook 05. Nunca foi chamado por nenhum método público.
    Mantê-lo criava risco de ser invocado por engano em código futuro, produzindo
    KeyError imediato por colunas inexistentes.

Null check em _build_temporal_content()
    taxa_vacinacao e taxa_crescimento podem ser NULL nas tabelas para períodos
    anteriores à campanha de vacinação (2019-2020) ou ao início do monitoramento
    contínuo. O código anterior fazia float(row["taxa_vacinacao"]) direto — com
    pandas NaN, float(NaN) não levanta erro, mas NaN formatado com :.2f produz
    "nan" no texto do documento, corrompendo o embedding semanticamente. O null
    check descarta o campo do texto quando ausente, em vez de inserir "nan".

Chave "tipo" substituída por "semantic_type" nos metadata dicts
    load_metricas_temporais, load_metricas_geograficas e load_metricas_demograficas
    usavam a chave "tipo" nos seus dicts de metadata. Como to_langchain_doc() faz
    {**self.metadata, "semantic_type": self.semantic_type}, o Document final
    terminava com "tipo" e "semantic_type" como chaves separadas — a primeira
    inútil e poluindo o metadata. Padronizado para "semantic_type" em todos os
    loaders, consistente com load_rag_kpi_fatos e load_dicionario_regras.

Variável _ substituída por idx em load_rag_kpi_fatos
    O loop usava `for _, row in df.iterrows()` e depois referenciava `_` em
    f"kpi_{_}" para o doc_id de fallback. Usar `_` como variável com valor
    real viola a convenção Python de "variável descartada" e produz doc_id
    correto apenas por coincidência (o índice pandas é inteiro). Substituído
    por `for idx, row` com referência explícita.

tempo_medio_notificacao e tempo_medio_internacao incluídos no conteúdo
    As colunas eram consultadas no SQL de load_metricas_temporais mas nunca
    apareciam no texto gerado por _build_temporal_content(). São indicadores
    operacionais relevantes para epidemiologia (atraso de notificação e
    tempo de permanência hospitalar) e agora são incluídos quando não nulos.

get_document_stats() com tipos reais do pipeline
    O dict by_type contava semantic_type == "metric" que nunca existe — os tipos
    reais produzidos pelos loaders são: kpi, regra, temporal, geographic, demographic.
    Contar um tipo inexistente retornava sempre zero e não mostrava kpi nem regra,
    tornando o summary inútil para diagnóstico de ingestão.

Imports mortos removidos
    Tuple, field, json e DataFrame importados mas nunca utilizados no módulo.


"""

from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
from langchain_core.documents import Document
from pyspark.sql import SparkSession


# =============================================================================
# DOCUMENT SCHEMA
# =============================================================================

@dataclass
class SRAGDocument:
    """
    Documento semântico intermediário entre as tabelas Gold e o Vector Store.

    Representa um chunk de texto com metadados estruturados antes da conversão
    para o formato LangChain. Permite inspecionar e filtrar documentos antes
    de gerar embeddings — útil para validação de ingestão sem custo de embedding.

    Campos
    ------
    content
        Texto em linguagem natural pronto para embedding. Deve ser autocontido —
        o LLM só terá acesso a este texto durante o retrieval, sem contexto externo.
    metadata
        Dict de atributos estruturados para filtragem no Vector Index.
        Deve sempre incluir as chaves "source_table" e "semantic_type" para
        compatibilidade com ContextBuilder e os filtros do SRAGRetriever.
    doc_id
        Identificador único do documento no índice vetorial. Usado como
        primary_key na Delta Table de embeddings — deve ser estável entre
        execuções para que Delta Sync identifique inserções vs atualizações.
    source_table
        Nome da tabela Gold de origem, sem catálogo/schema. Usado pelo
        _hybrid_retrieve() para aplicar boost de relevância na fonte primária.
    semantic_type
        Tipo semântico do documento: 'kpi', 'regra', 'temporal', 'geographic',
        'demographic'. Usado pelo _typed_retrieve() como filtro no Vector Index.

    Nota sobre to_langchain_doc()
        O Document final tem metadata = {**self.metadata, "doc_id": ...,
        "source_table": ..., "semantic_type": ...}. Os campos do dataclass
        sobrescrevem eventuais chaves homônimas no dict metadata — isso é
        intencional para garantir consistência mesmo se o loader popular o
        metadata de forma diferente.
    """
    content:       str
    metadata:      Dict
    doc_id:        str
    source_table:  str
    semantic_type: str

    def to_langchain_doc(self) -> Document:
        """
        Converte para Document do LangChain pronto para embedding.

        Os campos doc_id, source_table e semantic_type do dataclass têm
        precedência sobre eventuais chaves homônimas no dict metadata —
        garantia de consistência para o Vector Store e o ContextBuilder.
        """
        return Document(
            page_content=self.content,
            metadata={
                **self.metadata,
                "doc_id":        self.doc_id,
                "source_table":  self.source_table,
                "semantic_type": self.semantic_type,
            },
        )


# =============================================================================
# GOLD DOCUMENT LOADER
# =============================================================================

class GoldDocumentLoader:
    """
    Carrega tabelas Gold do catálogo Databricks e converte em SRAGDocuments.

    Cada método load_* é responsável por uma tabela — pode ser chamado
    individualmente para recarga parcial ou via load_all_documents() para
    ingestão completa. Os métodos _build_*_content() são construtores de
    texto semântico chamados internamente por cada loader.

    Estratégia de chunking por fonte
    ----------------------------------
    gold_rag_kpi_fatos       → campo text já pronto, 1 linha = 1 documento
    gold_rag_dicionario_regras → 1 regra = 1 documento
    gold_metricas_temporais  → 1 mês = 1 documento (todas as métricas do período)
    gold_metricas_geograficas → 1 UF = 1 documento (perfil regional completo)
    gold_metricas_demograficas → 1 combinação (faixa_etaria, sexo) = 1 documento

    Parâmetros
    ----------
    spark
        SparkSession ativa. Todas as queries usam spark.sql() — o catálogo
        e schema são injetados via self.full_prefix.
    catalog
        Catálogo Unity Catalog do Databricks. Default: dbx_srag_lab.
    schema
        Schema das tabelas Gold. Default: gold.

    Exemplo
    -------
        >>> loader = GoldDocumentLoader(spark)
        >>> docs = loader.load_all_documents()
        >>> print(loader.get_document_stats(docs))
    """

    def __init__(
        self,
        spark:   SparkSession,
        catalog: str = "dbx_srag_lab",
        schema:  str = "gold",
    ):
        self.spark       = spark
        self.catalog     = catalog
        self.schema      = schema
        self.full_prefix = f"{catalog}.{schema}"

    # =========================================================================
    # GOLD_RAG_KPI_FATOS — FONTE PRIMÁRIA
    # =========================================================================

    def load_rag_kpi_fatos(self) -> List[SRAGDocument]:
        """
        Carrega gold_rag_kpi_fatos — fonte primária do RAG.

        Schema da tabela (gerado pelo notebook 05_gold_base_conhecimento_rag):
            doc_id, text, doc_type, fonte_tabela, gerado_em, process_id

        O campo text já contém linguagem natural factual produzida pelo
        notebook 05 — não há builder de conteúdo adicional. Cada linha
        é 1 documento semântico independente pronto para embedding.

        Ordenação por gerado_em DESC
            Garante que os KPIs mais recentes sejam indexados primeiro.
            A coluna _gold_processed_at foi o erro original (SQLSTATE 42703
            UNRESOLVED_COLUMN) — corrigido para gerado_em na v2.1.0.

        semantic_type
            Derivado da coluna doc_type da tabela (ex: 'kpi_nacional',
            'kpi_regional'). Fallback para 'kpi' quando nulo.

        source_table
            Derivado da coluna fonte_tabela da tabela. Fallback para o nome
            da tabela de origem quando nulo.
        """
        query = f"""
            SELECT doc_id, text, doc_type, fonte_tabela
            FROM   {self.full_prefix}.gold_rag_kpi_fatos
            WHERE  text IS NOT NULL
            ORDER  BY gerado_em DESC NULLS LAST
        """

        df        = self.spark.sql(query).toPandas()
        documents = []

        for idx, row in df.iterrows():
            doc_id        = str(row.get("doc_id") or f"kpi_{idx}")
            source_table  = str(row.get("fonte_tabela") or "gold_rag_kpi_fatos")
            semantic_type = str(row.get("doc_type")     or "kpi")

            doc = SRAGDocument(
                content=str(row["text"]),
                metadata={
                    "doc_id":        doc_id,
                    "source_table":  source_table,
                    "semantic_type": semantic_type,
                    "timestamp":     datetime.now().isoformat(),
                },
                doc_id=doc_id,
                source_table=source_table,
                semantic_type=semantic_type,
            )
            documents.append(doc)

        print(f"gold_rag_kpi_fatos: {len(documents)} documentos")
        return documents

    # =========================================================================
    # GOLD_RAG_DICIONARIO_REGRAS — CONTEXTO METODOLÓGICO
    # =========================================================================

    def load_dicionario_regras(self) -> List[SRAGDocument]:
        """
        Carrega gold_rag_dicionario_regras — regras epidemiológicas formais.

        Schema da tabela:
            rule_id, rule_category, rule_name, rule_description, impact_analysis

        Fornece ao LLM o contexto das definições metodológicas do pipeline
        SIVEP-Gripe: critérios de classificação de caso, regras de imputação,
        definições de desfecho. Sem esse contexto, o LLM pode interpretar
        métricas com critérios divergentes dos usados no pipeline.

        Filtro WHERE rule_description IS NOT NULL
            Regras sem descrição não fornecem contexto útil para o LLM —
            um documento com apenas rule_id e rule_name seria indexado mas
            nunca recuperado em queries reais (sem conteúdo semântico relevante).
        """
        query = f"""
            SELECT rule_id, rule_category, rule_name, rule_description, impact_analysis
            FROM   {self.full_prefix}.gold_rag_dicionario_regras
            WHERE  rule_description IS NOT NULL
        """

        df        = self.spark.sql(query).toPandas()
        documents = []

        for idx, row in df.iterrows():
            doc_id = str(row.get("rule_id") or f"regra_{idx}")

            content = "\n\n".join(filter(None, [
                f"CATEGORIA: {row.get('rule_category', '')}",
                f"REGRA: {row.get('rule_name', '')}",
                f"DESCRIÇÃO: {row.get('rule_description', '')}",
                f"IMPACTO: {row.get('impact_analysis', '')}" if row.get("impact_analysis") else None,
            ])).strip()

            doc = SRAGDocument(
                content=content,
                metadata={
                    "doc_id":        doc_id,
                    "rule_category": str(row.get("rule_category", "")),
                    "rule_name":     str(row.get("rule_name",     "")),
                    "source_table":  "gold_rag_dicionario_regras",
                    "semantic_type": "regra",
                    "timestamp":     datetime.now().isoformat(),
                },
                doc_id=doc_id,
                source_table="gold_rag_dicionario_regras",
                semantic_type="regra",
            )
            documents.append(doc)

        print(f"gold_rag_dicionario_regras: {len(documents)} documentos")
        return documents

    # =========================================================================
    # GOLD_METRICAS_TEMPORAIS — TENDÊNCIAS HISTÓRICAS
    # =========================================================================

    def load_metricas_temporais(self, limit: int = 24) -> List[SRAGDocument]:
        """
        Carrega métricas temporais (últimos N meses).

        Schema consultado:
            ano_mes, total_casos, taxa_mortalidade, taxa_uti, taxa_vacinacao,
            taxa_crescimento, tempo_medio_notificacao, tempo_medio_internacao

        Chunking por mês
            1 mês = 1 documento com todas as métricas do período. Esse chunking
            preserva a coerência temporal — queries como "como foi março de 2024"
            recuperam um único documento completo em vez de múltiplos fragmentos
            parciais que precisariam ser reconciliados pelo LLM.

        Parâmetros
        ----------
        limit
            Número de meses mais recentes a carregar. Default: 24 (2 anos).
            Aumentar limit amplia cobertura histórica ao custo de mais documentos
            no índice — cada mês adicional é ~400 chars de texto embedado.

        Colunas nullable
            taxa_vacinacao e taxa_crescimento podem ser NULL em registros
            anteriores ao início da campanha de vacinação ou do monitoramento
            contínuo. tempo_medio_notificacao e tempo_medio_internacao podem
            ser NULL quando o cálculo não é possível para o período. Todos são
            omitidos do texto quando nulos — inserir "nan" corromperia o embedding.
        """
        query = f"""
            SELECT
                ano_mes,
                total_casos,
                taxa_mortalidade,
                taxa_uti,
                taxa_vacinacao,
                taxa_crescimento,
                tempo_medio_notificacao,
                tempo_medio_internacao
            FROM  {self.full_prefix}.gold_metricas_temporais
            ORDER BY ano_mes DESC
            LIMIT {limit}
        """

        df        = self.spark.sql(query).toPandas()
        documents = []

        for idx, row in df.iterrows():
            content = self._build_temporal_content(row)
            doc_id  = f"temporal_{row['ano_mes']}"

            doc = SRAGDocument(
                content=content,
                metadata={
                    "doc_id":          doc_id,
                    "ano_mes":         str(row["ano_mes"]),
                    "total_casos":     int(row["total_casos"]),
                    "taxa_mortalidade": float(row["taxa_mortalidade"]),
                    "taxa_uti":         float(row["taxa_uti"]),
                    "source_table":    "gold_metricas_temporais",
                    "semantic_type":   "temporal",
                },
                doc_id=doc_id,
                source_table="gold_metricas_temporais",
                semantic_type="temporal",
            )
            documents.append(doc)

        print(f"gold_metricas_temporais: {len(documents)} documentos")
        return documents

    def _build_temporal_content(self, row: pd.Series) -> str:
        """
        Constrói texto semântico para um período mensal de SRAG.

        Colunas nullable (taxa_vacinacao, taxa_crescimento,
        tempo_medio_notificacao, tempo_medio_internacao) são omitidas
        quando nulas em vez de inserir "nan" no texto.

        O texto usa linguagem natural porque o embedding semântico do BGE
        Large captura melhor relações conceituais em prosa do que em tabelas
        de números — "taxa de mortalidade de 12,5%" é mais próximo de
        "letalidade" no espaço de embeddings do que "taxa_mortalidade: 12.5".
        """
        ano_mes = row["ano_mes"]
        total   = int(row["total_casos"])
        mort    = float(row["taxa_mortalidade"])
        uti     = float(row["taxa_uti"])

        # Campos nullable — omitidos do texto quando ausentes
        vac_raw   = row.get("taxa_vacinacao")
        cresc_raw = row.get("taxa_crescimento")
        notif_raw = row.get("tempo_medio_notificacao")
        intern_raw = row.get("tempo_medio_internacao")

        vac    = float(vac_raw)    if pd.notna(vac_raw)    else None
        cresc  = float(cresc_raw)  if pd.notna(cresc_raw)  else None
        notif  = float(notif_raw)  if pd.notna(notif_raw)  else None
        intern = float(intern_raw) if pd.notna(intern_raw) else None

        lines = [
            f"PERÍODO: {ano_mes} (mês de referência)",
            "",
            f"CASOS TOTAIS: {total:,} notificações de SRAG",
            f"TAXA DE MORTALIDADE: {mort:.2f}%",
            f"TAXA DE OCUPAÇÃO UTI: {uti:.2f}%",
        ]

        if vac    is not None: lines.append(f"TAXA DE VACINAÇÃO: {vac:.2f}%")
        if cresc  is not None: lines.append(f"CRESCIMENTO MENSAL: {cresc:+.2f}%")
        if notif  is not None: lines.append(f"TEMPO MÉDIO DE NOTIFICAÇÃO: {notif:.1f} dias")
        if intern is not None: lines.append(f"TEMPO MÉDIO DE INTERNAÇÃO: {intern:.1f} dias")

        # Parágrafo de análise
        analise = (
            f"\nANÁLISE: Em {ano_mes}, foram registrados {total:,} casos de SRAG no Brasil. "
            f"A taxa de mortalidade foi de {mort:.2f}%, com {uti:.2f}% dos casos necessitando de UTI."
        )
        if vac   is not None: analise += f" A cobertura vacinal atingiu {vac:.2f}% dos casos notificados."
        if cresc is not None: analise += f" O crescimento em relação ao mês anterior foi de {cresc:+.2f}%."

        lines.append(analise)
        return "\n".join(lines)

    # =========================================================================
    # GOLD_METRICAS_GEOGRAFICAS — PERFIS REGIONAIS
    # =========================================================================

    def load_metricas_geograficas(self) -> List[SRAGDocument]:
        """
        Carrega métricas geográficas por UF.

        Schema consultado:
            sg_uf, total_casos, taxa_mortalidade, taxa_uti, taxa_vacinacao,
            ranking_casos, percentual_nacional

        Chunking por UF
            1 UF = 1 documento com o perfil regional completo. Essa granularidade
            permite que o _typed_retrieve() filtre por semantic_type='geographic'
            e reduza o espaço de busca quando a query menciona um estado específico.

        Ordenação por ranking_casos
            Garante que estados com mais casos sejam carregados primeiro no índice —
            relevante para o _hybrid_retrieve(), que ordena documentos recuperados
            por score e pode beneficiar estados de maior volume quando o score
            vetorial é semelhante.
        """
        query = f"""
            SELECT
                sg_uf,
                total_casos,
                taxa_mortalidade,
                taxa_uti,
                taxa_vacinacao,
                ranking_casos,
                percentual_nacional
            FROM  {self.full_prefix}.gold_metricas_geograficas
            ORDER BY ranking_casos
        """

        df        = self.spark.sql(query).toPandas()
        documents = []

        for idx, row in df.iterrows():
            content = self._build_geographic_content(row)
            doc_id  = f"geo_{row['sg_uf']}"

            doc = SRAGDocument(
                content=content,
                metadata={
                    "doc_id":       doc_id,
                    "uf":           str(row["sg_uf"]),
                    "total_casos":  int(row["total_casos"]),
                    "ranking":      int(row["ranking_casos"]),
                    "source_table": "gold_metricas_geograficas",
                    "semantic_type": "geographic",
                },
                doc_id=doc_id,
                source_table="gold_metricas_geograficas",
                semantic_type="geographic",
            )
            documents.append(doc)

        print(f"gold_metricas_geograficas: {len(documents)} documentos")
        return documents

    def _build_geographic_content(self, row: pd.Series) -> str:
        """
        Constrói texto semântico para o perfil epidemiológico de uma UF.

        taxa_vacinacao é nullable — verificada com pd.notna() antes do cast
        para evitar "nan" no texto. As demais colunas consultadas (total_casos,
        taxa_mortalidade, taxa_uti, ranking_casos, percentual_nacional) são
        NOT NULL pela construção das tabelas Gold.

        O texto inclui análise regional em linguagem natural para que
        embeddings capturem tanto os valores numéricos quanto o contexto
        epidemiológico do estado — "posiciona-se como 3º estado" é semanticamente
        próximo de "terceiro maior volume de notificações".
        """
        uf    = str(row["sg_uf"])
        casos = int(row["total_casos"])
        mort  = float(row["taxa_mortalidade"])
        uti   = float(row["taxa_uti"])
        rank  = int(row["ranking_casos"])
        pct   = float(row["percentual_nacional"])

        vac_raw = row.get("taxa_vacinacao")
        vac     = float(vac_raw) if pd.notna(vac_raw) else None

        lines = [
            f"ESTADO: {uf} (Unidade Federativa do Brasil)",
            "",
            "EPIDEMIOLOGIA SRAG:",
            f"Total de Casos: {casos:,} notificações registradas",
            f"Posição Nacional: {rank}º lugar no ranking de casos",
            f"Representatividade: {pct:.2f}% do total nacional",
            "",
            "INDICADORES DE GRAVIDADE:",
            f"Taxa de Mortalidade: {mort:.2f}% (letalidade por SRAG)",
            f"Taxa de UTI: {uti:.2f}% (casos que necessitaram terapia intensiva)",
        ]

        if vac is not None:
            lines.append(f"Taxa de Vacinação: {vac:.2f}% (cobertura vacinal)")

        lines += [
            "",
            f"ANÁLISE REGIONAL: O estado {uf} apresenta {casos:,} casos confirmados de "
            f"Síndrome Respiratória Aguda Grave, posicionando-se como {rank}º estado "
            f"brasileiro em número absoluto de notificações. Este volume representa "
            f"{pct:.2f}% de todos os casos nacionais. A taxa de mortalidade estadual "
            f"é de {mort:.2f}%, enquanto {uti:.2f}% dos pacientes necessitaram de "
            f"internação em UTI.",
            "",
            f"CONTEXTO EPIDEMIOLÓGICO: {uf} demonstra padrão epidemiológico característico "
            f"com casos distribuídos ao longo do território estadual. A monitorização "
            f"contínua permite identificação de surtos e tendências emergentes para "
            f"ações de saúde pública direcionadas.",
        ]

        return "\n".join(lines)

    # =========================================================================
    # GOLD_METRICAS_DEMOGRAFICAS — PERFIS POPULACIONAIS
    # =========================================================================

    def load_metricas_demograficas(self) -> List[SRAGDocument]:
        """
        Carrega métricas demográficas por faixa etária e sexo.

        Schema consultado:
            faixa_etaria, sexo, total_casos, taxa_mortalidade,
            taxa_internacao, percentual_total

        Chunking por (faixa_etaria, sexo)
            1 combinação = 1 documento. A presença de sexo='Total' na tabela
            (agregado de ambos os sexos para cada faixa) significa que cada
            faixa gera 3 documentos: M, F, Total. Isso permite queries tanto
            para comparação de sexo quanto para visão consolidada da faixa.

        Filtro WHERE faixa_etaria IS NOT NULL
            Registros sem faixa etária são agregados globais que já estão
            cobertos pelos KPI fatos — indexá-los aqui criaria redundância
            e potencial conflito de retrieval.

        Ordenação por ordem_faixa_etaria, sexo
            Garante consistência do índice entre execuções — sem ordenação
            determinística, o doc_id "demo_{faixa}_{sexo}" poderia indexar
            documentos em ordens diferentes, confundindo o Delta Sync.
        """
        query = f"""
            SELECT
                faixa_etaria,
                sexo,
                total_casos,
                taxa_mortalidade,
                taxa_internacao,
                percentual_total
            FROM  {self.full_prefix}.gold_metricas_demograficas
            WHERE faixa_etaria IS NOT NULL
            ORDER BY ordem_faixa_etaria, sexo
        """

        df        = self.spark.sql(query).toPandas()
        documents = []

        for idx, row in df.iterrows():
            content = self._build_demographic_content(row)
            doc_id  = f"demo_{row['faixa_etaria']}_{row['sexo']}"

            doc = SRAGDocument(
                content=content,
                metadata={
                    "doc_id":         doc_id,
                    "faixa_etaria":   str(row["faixa_etaria"]),
                    "sexo":           str(row["sexo"]),
                    "total_casos":    int(row["total_casos"]),
                    "source_table":   "gold_metricas_demograficas",
                    "semantic_type":  "demographic",
                },
                doc_id=doc_id,
                source_table="gold_metricas_demograficas",
                semantic_type="demographic",
            )
            documents.append(doc)

        print(f"gold_metricas_demograficas: {len(documents)} documentos")
        return documents

    def _build_demographic_content(self, row: pd.Series) -> str:
        """
        Constrói texto semântico para o perfil demográfico de uma faixa etária.

        Traduz o código de sexo (M/F/Total) para linguagem natural para que
        o embedding capture corretamente queries como "homens idosos" ou
        "população feminina acima de 60 anos". O código bruto "M" não tem
        representação semântica útil no espaço de embeddings do BGE Large.
        """
        faixa  = str(row["faixa_etaria"])
        sexo   = str(row["sexo"])
        casos  = int(row["total_casos"])
        mort   = float(row["taxa_mortalidade"])
        intern = float(row["taxa_internacao"])
        pct    = float(row["percentual_total"])

        _SEXO_TEXTO = {"M": "masculino", "F": "feminino", "Total": "ambos os sexos"}
        sexo_texto  = _SEXO_TEXTO.get(sexo, sexo)

        return "\n".join([
            f"PERFIL DEMOGRÁFICO: Faixa etária {faixa}, sexo {sexo_texto}",
            "",
            f"CASOS: {casos:,} notificações",
            f"PERCENTUAL DO TOTAL: {pct:.2f}%",
            "",
            "INDICADORES:",
            f"- Taxa de Mortalidade: {mort:.2f}%",
            f"- Taxa de Internação: {intern:.2f}%",
            "",
            f"ANÁLISE: O grupo de {faixa} anos ({sexo_texto}) apresentou {casos:,} casos "
            f"de SRAG, representando {pct:.2f}% do total nacional. A taxa de mortalidade "
            f"neste grupo foi de {mort:.2f}%, com {intern:.2f}% dos casos necessitando "
            f"de internação.",
        ])

    # =========================================================================
    # LOAD ALL
    # =========================================================================

    def load_all_documents(
        self,
        include_rag_kpi:    bool = True,
        include_dicionario: bool = True,
        include_temporal:   bool = True,
        include_geographic: bool = True,
        include_demographic: bool = True,
    ) -> List[SRAGDocument]:
        """
        Carrega documentos de todas as fontes Gold em sequência.

        A ordem de carga define a prioridade implícita no índice vetorial
        para documentos com score similar: kpi_fatos → dicionario → temporal
        → geograficas → demograficas. Isso reflete a hierarquia de relevância
        para queries gerais sobre SRAG.

        Flags include_*
            Permitem recarga parcial quando apenas uma fonte foi atualizada —
            evita reprocessar embeddings de fontes que não mudaram.

        Retorno
        -------
        Lista completa de SRAGDocuments prontos para ser passados a
        DatabricksVectorStoreManager.create_or_load_index().
        """
        print("Iniciando carregamento de documentos Gold...")
        all_docs: List[SRAGDocument] = []

        if include_rag_kpi:
            all_docs.extend(self.load_rag_kpi_fatos())
        if include_dicionario:
            all_docs.extend(self.load_dicionario_regras())
        if include_temporal:
            all_docs.extend(self.load_metricas_temporais())
        if include_geographic:
            all_docs.extend(self.load_metricas_geograficas())
        if include_demographic:
            all_docs.extend(self.load_metricas_demograficas())

        stats = self.get_document_stats(all_docs)
        print(
            f"\nTotal carregado: {stats['total_documents']} documentos "
            f"({stats['total_chars']:,} chars)\n"
            f"  kpi:         {stats['by_type']['kpi']}\n"
            f"  regra:       {stats['by_type']['regra']}\n"
            f"  temporal:    {stats['by_type']['temporal']}\n"
            f"  geographic:  {stats['by_type']['geographic']}\n"
            f"  demographic: {stats['by_type']['demographic']}"
        )

        return all_docs

    # =========================================================================
    # UTILITÁRIOS
    # =========================================================================

    def to_langchain_documents(self, docs: List[SRAGDocument]) -> List[Document]:
        """
        Converte lista de SRAGDocuments para Documents do LangChain.

        Usado quando o caller precisa do formato LangChain antes de passar
        para o DatabricksVectorStoreManager — que também aceita SRAGDocument
        diretamente via _prepare_documents_with_embeddings().
        """
        return [doc.to_langchain_doc() for doc in docs]

    def get_document_stats(self, docs: List[SRAGDocument]) -> Dict:
        """
        Retorna estatísticas de ingestão para diagnóstico e logging.

        by_type usa os tipos reais produzidos pelos loaders do pipeline:
        kpi, regra, temporal, geographic, demographic.

        O tipo "kpi" é detectado por contains em vez de igualdade exata
        porque doc_type da tabela gold_rag_kpi_fatos pode ter variantes
        como "kpi_nacional" ou "kpi_regional".

        Retorno
        -------
        Dict com total_documents, by_type (contagem por tipo),
        avg_content_length e total_chars.
        """
        if not docs:
            return {
                "total_documents":    0,
                "by_type":            {t: 0 for t in ("kpi", "regra", "temporal", "geographic", "demographic")},
                "avg_content_length": 0,
                "total_chars":        0,
            }

        return {
            "total_documents": len(docs),
            "by_type": {
                "kpi":         sum(1 for d in docs if "kpi"        in d.semantic_type),
                "regra":       sum(1 for d in docs if d.semantic_type == "regra"),
                "temporal":    sum(1 for d in docs if d.semantic_type == "temporal"),
                "geographic":  sum(1 for d in docs if d.semantic_type == "geographic"),
                "demographic": sum(1 for d in docs if d.semantic_type == "demographic"),
            },
            "avg_content_length": round(sum(len(d.content) for d in docs) / len(docs), 1),
            "total_chars":        sum(len(d.content) for d in docs),
        }

    def __repr__(self) -> str:
        return (
            f"GoldDocumentLoader("
            f"catalog={self.catalog}, "
            f"schema={self.schema})"
        )