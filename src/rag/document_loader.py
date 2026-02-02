"""
Document Loader - Ingestão de Tabelas Gold para RAG
====================================================

Converte registros das tabelas Gold em documentos semânticos
para embedding e retrieval.

Estratégia:
    - gold_resumo_geral → Documentos individuais (1 métrica = 1 doc)
    - gold_metricas_temporais → Agregados mensais
    - gold_metricas_geograficas → Perfis de UF
    - gold_metricas_demograficas → Perfis demográficos

Author: AI Engineer Certification - Indicium
Date: January 2025
Version: 2.0.0
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

from langchain_core.documents import Document
from pyspark.sql import SparkSession, DataFrame
import pandas as pd


# =============================================================================
# DOCUMENT SCHEMA
# =============================================================================

@dataclass
class SRAGDocument:
    """Documento semântico para RAG"""
    content: str  # Texto para embedding
    metadata: Dict
    doc_id: str
    source_table: str
    semantic_type: str  # 'metric', 'temporal', 'geographic', 'demographic'
    
    def to_langchain_doc(self) -> Document:
        """Converte para Document do LangChain"""
        return Document(
            page_content=self.content,
            metadata={
                **self.metadata,
                "doc_id": self.doc_id,
                "source_table": self.source_table,
                "semantic_type": self.semantic_type
            }
        )


# =============================================================================
# DOCUMENT LOADER PRINCIPAL
# =============================================================================

class GoldDocumentLoader:
    """
    Carrega tabelas Gold e converte em documentos semânticos
    
    Estratégia de Chunking:
        - gold_resumo_geral: 1 registro = 1 documento (granularidade fina)
        - gold_metricas_temporais: 1 mês = 1 documento (contexto temporal)
        - gold_metricas_geograficas: 1 UF = 1 documento (contexto regional)
        - gold_metricas_demograficas: 1 grupo = 1 documento (contexto social)
    
    Example:
        >>> loader = GoldDocumentLoader(spark)
        >>> docs = loader.load_all_documents()
        >>> print(f"Total: {len(docs)} documentos")
    """
    
    def __init__(
        self,
        spark: SparkSession,
        catalog: str = "dbx_lab_draagron",
        schema: str = "gold"
    ):
        self.spark = spark
        self.catalog = catalog
        self.schema = schema
        self.full_prefix = f"{catalog}.{schema}"
    
    # =========================================================================
    # GOLD_RESUMO_GERAL - FONTE PRIMÁRIA ⭐⭐⭐
    # =========================================================================
    
    def load_resumo_geral(self) -> List[SRAGDocument]:
        """
        Carrega gold_resumo_geral - FONTE MAIS IMPORTANTE
        
        Formato da tabela:
            | categoria | metrica | valor | unidade | descricao | escopo | data_snapshot |
        
        Cada linha vira 1 documento semântico independente.
        """
        query = f"""
        SELECT 
            categoria,
            metrica,
            valor,
            unidade,
            descricao,
            escopo,
            data_snapshot
        FROM {self.full_prefix}.gold_resumo_geral
        ORDER BY data_snapshot DESC, categoria, metrica
        """
        
        df = self.spark.sql(query).toPandas()
        
        documents = []
        for idx, row in df.iterrows():
            # Construir texto semântico
            content = self._build_resumo_geral_content(row)
            
            # Metadados estruturados
            metadata = {
                "categoria": row["categoria"],
                "metrica": row["metrica"],
                "valor": row["valor"],
                "unidade": row["unidade"],
                "escopo": row["escopo"],
                "data_snapshot": str(row["data_snapshot"]),
                "timestamp": datetime.now().isoformat()
            }
            
            # ID único
            doc_id = f"resumo_{row['categoria']}_{row['metrica']}_{row['data_snapshot']}"
            
            doc = SRAGDocument(
                content=content,
                metadata=metadata,
                doc_id=doc_id,
                source_table="gold_resumo_geral",
                semantic_type="metric"
            )
            
            documents.append(doc)
        
        print(f"✅ gold_resumo_geral: {len(documents)} documentos")
        return documents
    
    def _build_resumo_geral_content(self, row: pd.Series) -> str:
        """
        Constrói texto semântico otimizado para embedding
        
        Estratégia:
            - Linguagem natural
            - Contexto explícito
            - Valores formatados
            - Descrição integrada
        """
        categoria = row["categoria"]
        metrica = row["metrica"]
        valor = row["valor"]
        unidade = row["unidade"]
        descricao = row["descricao"]
        escopo = row["escopo"]
        data = row["data_snapshot"]
        
        # Template semântico
        content = f"""
CATEGORIA: {categoria}
MÉTRICA: {metrica}

VALOR ATUAL: {valor} {unidade}
ESCOPO: {escopo}
DATA: {data}

DESCRIÇÃO: {descricao}

CONTEXTO: Esta métrica faz parte do monitoramento epidemiológico de SRAG (Síndrome Respiratória Aguda Grave) no Brasil.
        """.strip()
        
        return content
    
    # =========================================================================
    # GOLD_METRICAS_TEMPORAIS - TENDÊNCIAS HISTÓRICAS
    # =========================================================================
    
    def load_metricas_temporais(self, limit: int = 24) -> List[SRAGDocument]:
        """
        Carrega métricas temporais (últimos N meses)
        
        Estratégia de chunking:
            - 1 mês = 1 documento (contexto mensal completo)
            - Inclui todas as métricas do mês
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
        FROM {self.full_prefix}.gold_metricas_temporais
        ORDER BY ano_mes DESC
        LIMIT {limit}
        """
        
        df = self.spark.sql(query).toPandas()
        
        documents = []
        for idx, row in df.iterrows():
            content = self._build_temporal_content(row)
            
            metadata = {
                "ano_mes": row["ano_mes"],
                "total_casos": int(row["total_casos"]),
                "taxa_mortalidade": float(row["taxa_mortalidade"]),
                "taxa_uti": float(row["taxa_uti"]),
                "tipo": "temporal_mensal"
            }
            
            doc_id = f"temporal_{row['ano_mes']}"
            
            doc = SRAGDocument(
                content=content,
                metadata=metadata,
                doc_id=doc_id,
                source_table="gold_metricas_temporais",
                semantic_type="temporal"
            )
            
            documents.append(doc)
        
        print(f"✅ gold_metricas_temporais: {len(documents)} documentos")
        return documents
    
    def _build_temporal_content(self, row: pd.Series) -> str:
        """Constrói texto semântico para métrica temporal"""
        ano_mes = row["ano_mes"]
        total = int(row["total_casos"])
        mort = float(row["taxa_mortalidade"])
        uti = float(row["taxa_uti"])
        vac = float(row["taxa_vacinacao"])
        cresc = float(row["taxa_crescimento"])
        
        content = f"""
PERÍODO: {ano_mes} (mês de referência)

CASOS TOTAIS: {total:,} notificações de SRAG
TAXA DE MORTALIDADE: {mort:.2f}%
TAXA DE OCUPAÇÃO UTI: {uti:.2f}%
TAXA DE VACINAÇÃO: {vac:.2f}%
CRESCIMENTO MENSAL: {cresc:+.2f}%

ANÁLISE: Em {ano_mes}, foram registrados {total:,} casos de SRAG no Brasil. 
A taxa de mortalidade foi de {mort:.2f}%, com {uti:.2f}% dos casos necessitando de UTI.
A cobertura vacinal atingiu {vac:.2f}% dos casos notificados.
O crescimento em relação ao mês anterior foi de {cresc:+.2f}%.
        """.strip()
        
        return content
    
    # =========================================================================
    # GOLD_METRICAS_GEOGRAFICAS - PERFIS REGIONAIS
    # =========================================================================
    
    def load_metricas_geograficas(self) -> List[SRAGDocument]:
        """
        Carrega métricas geográficas (por UF)
        
        Estratégia:
            - 1 UF = 1 documento (perfil regional completo)
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
        FROM {self.full_prefix}.gold_metricas_geograficas
        ORDER BY ranking_casos
        """
        
        df = self.spark.sql(query).toPandas()
        
        documents = []
        for idx, row in df.iterrows():
            content = self._build_geographic_content(row)
            
            metadata = {
                "uf": row["sg_uf"],
                "total_casos": int(row["total_casos"]),
                "ranking": int(row["ranking_casos"]),
                "tipo": "geographic_profile"
            }
            
            doc_id = f"geo_{row['sg_uf']}"
            
            doc = SRAGDocument(
                content=content,
                metadata=metadata,
                doc_id=doc_id,
                source_table="gold_metricas_geograficas",
                semantic_type="geographic"
            )
            
            documents.append(doc)
        
        print(f"✅ gold_metricas_geograficas: {len(documents)} documentos")
        return documents
    
    def _build_geographic_content(self, row: pd.Series) -> str:
        """
        Constrói texto semântico otimizado para perfil geográfico
        
        Estratégia:
            - Linguagem natural e contextual
            - Dados quantitativos formatados
            - Posicionamento relativo (ranking)
            - Contexto epidemiológico
        """
        uf = row["sg_uf"]
        casos = int(row["total_casos"])
        mort = float(row["taxa_mortalidade"])
        uti = float(row["taxa_uti"])
        vac = float(row["taxa_vacinacao"]) if "taxa_vacinacao" in row else None
        rank = int(row["ranking_casos"])
        pct = float(row["percentual_nacional"])
        
        # Construir texto semântico otimizado
        content = f"""ESTADO: {uf} (Unidade Federativa do Brasil)

EPIDEMIOLOGIA SRAG:
Total de Casos: {casos:,} notificações registradas
Posição Nacional: {rank}º lugar no ranking de casos
Representatividade: {pct:.2f}% do total nacional

INDICADORES DE GRAVIDADE:
Taxa de Mortalidade: {mort:.2f}% (letalidade por SRAG)
Taxa de UTI: {uti:.2f}% (casos que necessitaram terapia intensiva)"""

        # Adicionar vacinação se disponível
        if vac is not None:
            content += f"\nTaxa de Vacinação: {vac:.2f}% (cobertura vacinal)"

        # Análise contextual
        content += f"""

ANÁLISE REGIONAL: O estado {uf} apresenta {casos:,} casos confirmados de Síndrome Respiratória Aguda Grave, posicionando-se como {rank}º estado brasileiro em número absoluto de notificações. Este volume representa {pct:.2f}% de todos os casos nacionais. A taxa de mortalidade estadual é de {mort:.2f}%, enquanto {uti:.2f}% dos pacientes necessitaram de internação em UTI.

CONTEXTO EPIDEMIOLÓGICO: {uf} demonstra padrão epidemiológico característico com casos distribuídos ao longo do território estadual. A monitorização contínua permite identificação de surtos e tendências emergentes para ações de saúde pública direcionadas."""

        return content.strip()
    
    # =========================================================================
    # GOLD_METRICAS_DEMOGRAFICAS - PERFIS POPULACIONAIS
    # =========================================================================
    
    def load_metricas_demograficas(self) -> List[SRAGDocument]:
        """
        Carrega métricas demográficas
        
        Estratégia:
            - 1 grupo etário = 1 documento
        """
        query = f"""
        SELECT 
            faixa_etaria,
            sexo,
            total_casos,
            taxa_mortalidade,
            taxa_internacao,
            percentual_total
        FROM {self.full_prefix}.gold_metricas_demograficas
        WHERE faixa_etaria IS NOT NULL
        ORDER BY ordem_faixa_etaria, sexo
        """
        
        df = self.spark.sql(query).toPandas()
        
        documents = []
        for idx, row in df.iterrows():
            content = self._build_demographic_content(row)
            
            metadata = {
                "faixa_etaria": row["faixa_etaria"],
                "sexo": row["sexo"],
                "total_casos": int(row["total_casos"]),
                "tipo": "demographic_profile"
            }
            
            doc_id = f"demo_{row['faixa_etaria']}_{row['sexo']}"
            
            doc = SRAGDocument(
                content=content,
                metadata=metadata,
                doc_id=doc_id,
                source_table="gold_metricas_demograficas",
                semantic_type="demographic"
            )
            
            documents.append(doc)
        
        print(f"✅ gold_metricas_demograficas: {len(documents)} documentos")
        return documents
    
    def _build_demographic_content(self, row: pd.Series) -> str:
        """Constrói texto semântico para perfil demográfico"""
        faixa = row["faixa_etaria"]
        sexo = row["sexo"]
        casos = int(row["total_casos"])
        mort = float(row["taxa_mortalidade"])
        intern = float(row["taxa_internacao"])
        pct = float(row["percentual_total"])
        
        sexo_texto = {"M": "masculino", "F": "feminino", "Total": "ambos os sexos"}.get(sexo, sexo)
        
        content = f"""
PERFIL DEMOGRÁFICO: Faixa etária {faixa}, sexo {sexo_texto}

CASOS: {casos:,} notificações
PERCENTUAL DO TOTAL: {pct:.2f}%

INDICADORES:
- Taxa de Mortalidade: {mort:.2f}%
- Taxa de Internação: {intern:.2f}%

ANÁLISE: O grupo de {faixa} anos ({sexo_texto}) apresentou {casos:,} casos 
de SRAG, representando {pct:.2f}% do total nacional. A taxa de mortalidade 
neste grupo foi de {mort:.2f}%, com {intern:.2f}% dos casos necessitando de internação.
        """.strip()
        
        return content
    
    # =========================================================================
    # CARREGAMENTO COMPLETO
    # =========================================================================
    
    def load_all_documents(
        self,
        include_resumo: bool = True,
        include_temporal: bool = True,
        include_geographic: bool = True,
        include_demographic: bool = True
    ) -> List[SRAGDocument]:
        """
        Carrega todos os documentos de todas as fontes
        
        Returns:
            Lista completa de documentos prontos para embedding
        """
        print("📚 Iniciando carregamento de documentos...")
        all_docs = []
        
        if include_resumo:
            all_docs.extend(self.load_resumo_geral())
        
        if include_temporal:
            all_docs.extend(self.load_metricas_temporais())
        
        if include_geographic:
            all_docs.extend(self.load_metricas_geograficas())
        
        if include_demographic:
            all_docs.extend(self.load_metricas_demograficas())
        
        print(f"\n✅ Total de documentos carregados: {len(all_docs)}")
        print(f"   - Resumo Geral: {sum(1 for d in all_docs if d.semantic_type == 'metric')}")
        print(f"   - Temporal: {sum(1 for d in all_docs if d.semantic_type == 'temporal')}")
        print(f"   - Geográfico: {sum(1 for d in all_docs if d.semantic_type == 'geographic')}")
        print(f"   - Demográfico: {sum(1 for d in all_docs if d.semantic_type == 'demographic')}")
        
        return all_docs
    
    def to_langchain_documents(self, docs: List[SRAGDocument]) -> List[Document]:
        """Converte para formato LangChain"""
        return [doc.to_langchain_doc() for doc in docs]
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def get_document_stats(self, docs: List[SRAGDocument]) -> Dict:
        """Retorna estatísticas dos documentos"""
        return {
            "total_documents": len(docs),
            "by_type": {
                "metric": sum(1 for d in docs if d.semantic_type == "metric"),
                "temporal": sum(1 for d in docs if d.semantic_type == "temporal"),
                "geographic": sum(1 for d in docs if d.semantic_type == "geographic"),
                "demographic": sum(1 for d in docs if d.semantic_type == "demographic")
            },
            "avg_content_length": sum(len(d.content) for d in docs) / len(docs) if docs else 0,
            "total_chars": sum(len(d.content) for d in docs)
        }
