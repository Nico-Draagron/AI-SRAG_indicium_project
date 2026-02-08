"""
Vector Store - Databricks Vector Search + Embeddings
=====================================================

Gerencia embeddings e busca vetorial usando Databricks Vector Search.

Estratégia:
    - Embeddings: text-embedding-3-small ou similar
    - Vector Store: Databricks Vector Search (nativo)
    - Indexação: Delta Sync para atualização automática
    - Retrieval: Top-K com filtros de metadata

Author: AI Engineer Certification - Indicium  
Date: January 2025
Version: 2.0.0
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from functools import lru_cache
import json
import time
import re

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import DatabricksVectorSearch
from databricks.vector_search.client import VectorSearchClient

# Embeddings providers - Databricks native only
try:
    from langchain_community.embeddings import DatabricksEmbeddings
    DATABRICKS_AVAILABLE = True
except:
    DATABRICKS_AVAILABLE = False

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings  
    HF_AVAILABLE = True
except:
    HF_AVAILABLE = False


# =============================================================================
# EMBEDDING MANAGER
# =============================================================================

class EmbeddingManager:
    """
    Gerencia criação de embeddings usando modelos Databricks nativos
    
    Providers suportados:
        - Databricks (bge_large_en_v1_5) - RECOMENDADO 
        - HuggingFace (fallback local)
    
    Dimensões:
        - BGE Large: 1024 dims (padrão Databricks)
        - HF all-MiniLM-L6-v2: 384 dims (fallback)
    """
    
    @staticmethod
    def get_embeddings(
        provider: str = "databricks",
        model: str = "bge_large_en_v1_5", 
        **kwargs
    ) -> Embeddings:
        """
        Factory de embeddings usando modelos Databricks nativos
        
        Args:
            provider: 'databricks' ou 'huggingface'
            model: Nome do modelo
            
        Returns:
            Instância de Embeddings (sem dependência externa)
        """
        if provider == "databricks":
            if not DATABRICKS_AVAILABLE:
                raise ImportError("langchain-community[databricks] não instalado")
            
            return DatabricksEmbeddings(
                endpoint="databricks-bge-large-en",  # Endpoint padrão do BGE
                model=model,
                **kwargs
            )
        
        elif provider == "huggingface":
            if not HF_AVAILABLE:
                raise ImportError("sentence-transformers não instalado")
            
            return HuggingFaceEmbeddings(
                model_name="BAAI/bge-large-en-v1.5",  # BGE model direto
                **kwargs
            )
        
        else:
            raise ValueError(f"Provider não suportado: {provider}. Use 'databricks' ou 'huggingface'")
    
    @staticmethod
    def get_embedding_dimensions(provider: str, model: str) -> int:
        """Retorna dimensões do embedding (Databricks-optimized)"""
        dimensions_map = {
            "bge_large_en_v1_5": 1024,  # BGE Large (Databricks padrão)
            "all-MiniLM-L6-v2": 384,    # HF fallback
            "all-mpnet-base-v2": 768    # HF alternate
        }
        
        return dimensions_map.get(model, 1024)  # Default BGE Databricks


# =============================================================================
# DATABRICKS VECTOR SEARCH MANAGER
# =============================================================================

@dataclass
class VectorStoreConfig:
    """Configuração do Vector Store - Databricks BGE optimized"""
    catalog: str = "dbx_lab_draagron"
    schema: str = "gold" 
    index_name: str = "srag_embeddings_index_bge"  # BGE-specific index
    endpoint_name: str = "srag_vector_endpoint"
    embedding_dim: int = 1024  # BGE Large dimension 
    primary_key: str = "doc_id"
    embedding_source_column: str = "content"
    embedding_vector_column: str = "embedding"


class DatabricksVectorStoreManager:
    """
    Gerencia Databricks Vector Search
    
    Workflow:
        1. Criar endpoint (se não existir)
        2. Criar tabela Delta com embeddings
        3. Criar índice vetorial
        4. Delta Sync automático
    
    Example:
        >>> manager = DatabricksVectorStoreManager(spark, config)
        >>> manager.create_vector_index(documents)
        >>> results = manager.search("casos de SRAG em SP", k=5)
    """
    
    def __init__(
        self,
        spark,
        config: Optional[VectorStoreConfig] = None,
        embeddings: Optional[Embeddings] = None
    ):
        self.spark = spark
        self.config = config or VectorStoreConfig()
        self.embeddings = embeddings or EmbeddingManager.get_embeddings(
            provider="databricks",
            model="bge_large_en_v1_5"
        )
        self.client = VectorSearchClient()
        
        # Nome completo do índice
        self.full_index_name = f"{self.config.catalog}.{self.config.schema}.{self.config.index_name}"
        
    # =========================================================================
    # SETUP E CRIAÇÃO
    # =========================================================================
    
    def create_vector_index(
        self,
        documents: List[Document],
        recreate: bool = False
    ) -> str:
        """
        Cria índice vetorial completo
        
        Steps:
            1. Criar endpoint (se necessário)
            2. Preparar documentos com embeddings
            3. Salvar em Delta Table
            4. Criar índice vetorial com Delta Sync
        
        Args:
            documents: Lista de documentos LangChain
            recreate: Se True, deleta índice existente
            
        Returns:
            Nome do índice criado
        """
        print(f"📦 Criando Vector Index: {self.full_index_name}")
        
        # 1. Criar endpoint
        self._ensure_endpoint_exists()
        
        # 2. Preparar dados com embeddings
        print("🔄 Gerando embeddings...")
        df_with_embeddings = self._prepare_documents_with_embeddings(documents)
        
        # 3. Salvar em Delta
        print("💾 Salvando em Delta Table...")
        self._save_to_delta(df_with_embeddings, recreate=recreate)
        
        # 4. Criar índice vetorial
        print("🔗 Criando Vector Index...")
        index = self._create_or_update_index(recreate=recreate)
        
        print(f"✅ Vector Index criado: {self.full_index_name}")
        return self.full_index_name
    
    def create_or_load_index(self, documents: List[Document]) -> bool:
        """
        Garante que o índice vetorial existe e está disponível
        
        Args:
            documents: Lista de documentos para criar o índice se necessário
            
        Returns:
            True se índice está pronto, False caso contrário
        """
        try:
            # Estratégia defensiva: verificar diretamente se o índice existe
            # sem usar list_indexes que tem assinaturas inconsistentes
            print("🔧 Verificando se índice existe (verificação direta)...")
            
            try:
                # Tentar acessar o índice diretamente com endpoint_name
                index_info = self.client.get_index(
                    endpoint_name=self.config.endpoint_name,
                    index_name=self.full_index_name
                )
                # get_index retorna um objeto VectorSearchIndex, não dict
                if index_info and hasattr(index_info, 'name') and index_info.name == self.full_index_name:
                    print(f"✅ Índice vetorial já existe: {self.full_index_name}")
                    return True
            except Exception as get_error:
                # Se get_index falhar, provavelmente o índice não existe
                print(f"🔄 Índice não encontrado (erro esperado): {get_error}")
                
            # Se chegou aqui, o índice não existe - criar
            print(f"🔄 Criando novo índice: {self.full_index_name}")
            self.create_vector_index(documents, recreate=False)
            return True
            
        except Exception as e:
            print(f"❌ Erro ao verificar/criar índice: {e}")
            return False
    
    def _ensure_endpoint_exists(self) -> None:
        """Garante que endpoint existe"""
        try:
            endpoints = self.client.list_endpoints()
            endpoint_names = [e['name'] for e in endpoints.get('endpoints', [])]
            
            if self.config.endpoint_name not in endpoint_names:
                print(f"🔧 Criando endpoint: {self.config.endpoint_name}")
                self.client.create_endpoint(
                    name=self.config.endpoint_name,
                    endpoint_type="STANDARD"
                )
                print(f"✅ Endpoint criado")
            else:
                print(f"✅ Endpoint já existe: {self.config.endpoint_name}")
                
        except Exception as e:
            print(f"⚠️ Erro ao verificar endpoint: {e}")
    
    def _prepare_documents_with_embeddings(self, documents) -> 'pd.DataFrame':
        """Prepara DataFrame com embeddings para Databricks Vector Search"""
        import pandas as pd
        from datetime import datetime
        
        if not documents:
            raise ValueError("Lista de documentos está vazia")
        
        print(f"   📊 Preparando {len(documents)} documentos Gold para embedding...")
        
        # Converter SRAGDocument para Document do LangChain se necessário
        langchain_documents = []
        for i, doc in enumerate(documents):
            # Verificar se é SRAGDocument e converter
            if hasattr(doc, 'to_langchain_doc'):
                # É um SRAGDocument, converter para LangChain Document
                langchain_doc = doc.to_langchain_doc()
                langchain_documents.append(langchain_doc)
            elif hasattr(doc, 'page_content'):
                # Já é um LangChain Document
                langchain_documents.append(doc)
            else:
                print(f"   ⚠️ Documento {i} de tipo desconhecido: {type(doc)} - ignorando")
                continue
        
        print(f"   ✅ {len(langchain_documents)} documentos convertidos para LangChain")
        
        # Validar documentos convertidos
        valid_documents = []
        for i, doc in enumerate(langchain_documents):
            if not doc.page_content or not doc.page_content.strip():
                print(f"   ⚠️ Documento {i} sem conteúdo - ignorando")
                continue
            if not doc.metadata.get("doc_id"):
                print(f"   ⚠️ Documento {i} sem doc_id - gerando automaticamente")
                doc.metadata["doc_id"] = f"auto_gen_{i}_{int(datetime.now().timestamp())}"
            valid_documents.append(doc)
        
        print(f"   ✅ {len(valid_documents)} documentos válidos para processamento")
        
        # Extrair textos limpos
        texts = [doc.page_content.strip() for doc in valid_documents]
        
        # Gerar embeddings em batch (otimizado com retry inteligente)
        print(f"   🔄 Gerando embeddings usando {self.embeddings.__class__.__name__}...")
        max_retries = 2  # Reduzido - se falhar 2x, é problema real
        for attempt in range(max_retries):
            try:
                embeddings_vectors = self.embeddings.embed_documents(texts)
                print(f"   ✅ {len(embeddings_vectors)} embeddings gerados com sucesso")
                break
            except Exception as e:
                error_str = str(e).lower()
                if attempt < max_retries - 1:
                    # Rate limit: aguardar mais
                    if "rate limit" in error_str or "quota" in error_str:
                        wait_time = 10  # 10s para rate limit
                        print(f"   ⚠️ Rate limit detectado - aguardando {wait_time}s...")
                    # Connection reset: retry rápido  
                    elif "connection" in error_str or "reset" in error_str:
                        wait_time = 3  # 3s para problemas de rede
                        print(f"   ⚠️ Problema de rede - tentativa {attempt + 1} em {wait_time}s...")
                    else:
                        wait_time = 5  # 5s para outros erros
                        print(f"   ⚠️ Erro desconhecido - tentativa {attempt + 1} em {wait_time}s...")
                    
                    print(f"   📋 Erro: {e}")
                    time.sleep(wait_time)
                    continue
                else:
                    # Últimas tentativas: usar lotes menores apenas se necessário
                    if len(texts) > 10:  # Só usar lotes se tiver mais de 10 textos
                        print(f"   🔄 Tentando em lotes menores de 10 documentos...")
                        embeddings_vectors = self._embed_documents_in_batches(texts, batch_size=10)
                    else:
                        print(f"   ❌ Falha definitiva após {max_retries} tentativas: {e}")
                        raise
                    break
        
        # 1️⃣ AJUSTE DE ROBUSTEZ: Validar dimensão dos embeddings
        if embeddings_vectors:
            actual_dim = len(embeddings_vectors[0])
            expected_dim = self.config.embedding_dim
            if actual_dim != expected_dim:
                raise ValueError(
                    f"Dimensão do embedding não confere: "
                    f"esperado={expected_dim}, atual={actual_dim}. "
                    f"Verifique a configuração do modelo de embedding."
                )
            print(f"   ✅ Dimensão validada: {actual_dim}d")
        
        # Construir DataFrame otimizado para Databricks
        data = []
        for doc, embedding in zip(valid_documents, embeddings_vectors):
            # Extrair campos essenciais do metadata
            metadata = doc.metadata.copy()
            
            row = {
                # Campos obrigatórios
                "doc_id": metadata.get("doc_id", ""),
                "content": doc.page_content.strip(),
                "embedding": embedding,
                
                # Campos para filtros
                "source_table": metadata.get("source_table", "unknown"),
                "semantic_type": metadata.get("semantic_type", "general"),
                
                # Campos específicos do Gold - SEMPRE string para evitar type void
                "categoria": metadata.get("categoria", "geral"),
                "metrica": metadata.get("metrica", "indefinida"),
                "uf": metadata.get("uf", "BR"),  # Brasil como padrão em vez de None
                "ano_mes": metadata.get("ano_mes", "2024-01"),  # Padrão válido
                "faixa_etaria": metadata.get("faixa_etaria", "todas"),
                
                # Metadata completo como JSON
                "metadata_json": json.dumps(metadata, ensure_ascii=False),
                
                # Timestamp para auditoria
                "created_at": datetime.now().isoformat()
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Validar DataFrame
        print(f"   📋 Validando DataFrame: {len(df)} linhas, {len(df.columns)} colunas")
        print(f"   📊 Dimensão dos embeddings: {len(df.iloc[0]['embedding']) if len(df) > 0 else 'N/A'}")
        
        return df
    
    def _save_to_delta(self, df: 'pd.DataFrame', recreate: bool = False) -> None:
        """Salva DataFrame em Delta Table com otimizações"""
        if df.empty:
            raise ValueError("DataFrame está vazio")
        
        table_name = self.full_index_name.replace("_index", "_table")
        print(f"   💾 Salvando {len(df)} registros em Delta Table: {table_name}")
        
        try:
            # Converter para Spark DataFrame
            spark_df = self.spark.createDataFrame(df)
            
            # Aplicar particionamento por semantic_type para performance
            spark_df = spark_df.repartition("semantic_type")
            
            # Configurar modo de escrita
            mode = "overwrite" if recreate else "append"
            
            # Salvar com otimizações Delta
            writer = spark_df.write.format("delta").mode(mode)
            
            # Configurações para melhor performance
            writer = writer.option("delta.autoOptimize.optimizeWrite", "true")
            writer = writer.option("delta.autoOptimize.autoCompact", "true")
            
            # CORREÇÃO: Permitir evolução de schema (void -> string)
            writer = writer.option("mergeSchema", "true")
            
            # Particionar por semantic_type se recreating
            if recreate:
                writer = writer.partitionBy("semantic_type")
            
            # Executar escrita
            writer.saveAsTable(table_name)
            
            # Otimizar tabela após escrita (apenas se recreate)
            if recreate:
                self.spark.sql(f"OPTIMIZE {table_name}")
                print(f"   🔧 Tabela otimizada: {table_name}")
            
            # Verificar resultado
            count = self.spark.table(table_name).count()
            print(f"   ✅ Dados salvos: {count} registros totais em {table_name}")
            
        except Exception as e:
            print(f"   ❌ Erro ao salvar em Delta: {e}")
            raise
    
    def _create_or_update_index(self, recreate: bool = False) -> Dict:
        """Cria ou atualiza índice vetorial no Databricks Vector Search"""
        source_table = self.full_index_name.replace("_index", "_table")
        
        try:
            # Verificar se endpoint está ativo
            print(f"   🔍 Verificando endpoint: {self.config.endpoint_name}")
            endpoint_info = self.client.get_endpoint(self.config.endpoint_name)
            if endpoint_info.get("endpoint_status", {}).get("state") != "ONLINE":
                print(f"   ⚠️ Endpoint não está ONLINE: {endpoint_info.get('endpoint_status', {})}")
            
            # Verificar se tabela Delta existe e tem dados
            try:
                table_count = self.spark.table(source_table).count()
                print(f"   📊 Tabela fonte: {source_table} ({table_count} registros)")
                if table_count == 0:
                    raise ValueError(f"Tabela {source_table} está vazia")
            except Exception as table_error:
                print(f"   ❌ Erro ao verificar tabela: {table_error}")
                raise
            
            # Verificar se índice já existe usando verificação direta
            print(f"   🔍 Verificando se índice {self.full_index_name} já existe...")
            index_exists = False
            try:
                index_info = self.client.get_index(
                    endpoint_name=self.config.endpoint_name,
                    index_name=self.full_index_name
                )
                # get_index retorna um objeto VectorSearchIndex, não dict
                if index_info and hasattr(index_info, 'name') and index_info.name == self.full_index_name:
                    index_exists = True
                    if recreate:
                        print(f"   🗑️ Deletando índice existente: {self.full_index_name}")
                        self.client.delete_index(index_name=self.full_index_name)
                        # Aguardar deleção
                        import time
                        time.sleep(10)
                        index_exists = False
                    else:
                        print(f"   ✅ Índice já existe: {self.full_index_name}")
                        # Acessar status como atributo, não como dict
                        status = getattr(index_info, 'status', 'unknown')
                        print(f"   📊 Status: {status}")
                        return {"status": "exists", "info": index_info}
            except Exception as check_error:
                # Se get_index falhar, provavelmente o índice não existe
                print(f"   📝 Índice não encontrado (erro esperado): {check_error}")
                index_exists = False
            
            # Criar novo índice apenas se não existir
            if not index_exists:
                print(f"   🔗 Criando Vector Index: {self.full_index_name}")
                print(f"   📋 Configurações:")
                print(f"      - Tabela fonte: {source_table}")
                print(f"      - Primary key: {self.config.primary_key}")
                print(f"      - Embedding column: {self.config.embedding_vector_column}")
                print(f"      - Dimensões: {self.config.embedding_dim}")
                
                index = self.client.create_delta_sync_index(
                    endpoint_name=self.config.endpoint_name,
                    index_name=self.full_index_name,
                    source_table_name=source_table,
                    pipeline_type="TRIGGERED",
                    primary_key=self.config.primary_key,
                    embedding_dimension=self.config.embedding_dim,
                    embedding_vector_column=self.config.embedding_vector_column
                )
                
                print(f"   ✅ Vector Index criado com Delta Sync")
                print(f"   🔄 Aguarde a sincronização inicial...")
                
                # Verificar criação
                try:
                    index_status = self.client.get_index(
                        endpoint_name=self.config.endpoint_name,
                        index_name=self.full_index_name
                    )
                    # Acessar status como atributo do objeto VectorSearchIndex
                    status = getattr(index_status, 'status', 'unknown')
                    print(f"   📊 Status inicial: {status}")
                except Exception as status_error:
                    print(f"   ⚠️ Não foi possível verificar status: {status_error}")
                
                return index
            else:
                # Índice já existe, retornar info
                return {"status": "already_exists"}
            
        except Exception as e:
            print(f"   ❌ Erro ao criar índice vetorial: {str(e)}")
            print(f"   🔧 Troubleshooting:")
            print(f"      - Verifique se o endpoint {self.config.endpoint_name} está ativo")
            print(f"      - Verifique se a tabela {source_table} existe e tem dados")
            print(f"      - Verifique permissões do Databricks Vector Search")
            raise
    
    # =========================================================================
    # BUSCA E RETRIEVAL
    # =========================================================================
    
    def search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Tuple[Document, float]]:
        """
        Busca semântica no Vector Store usando Databricks Vector Search
        
        Args:
            query: Texto da busca
            k: Número de resultados
            filters: Filtros de metadata (ex: {"semantic_type": "metric"})
            
        Returns:
            Lista de (Document, score) ordenada por relevância
            IMPORTANTE: scores maiores = maior similaridade (0.0-1.0+)
        """
        # 3️⃣ AJUSTE: Retry com backoff simples
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Gerar embedding da query usando cache otimizado
                query_embedding = list(self._get_cached_query_embedding(query))
                
                # 2️⃣ AJUSTE: Sanitizar filtros com whitelist
                filter_conditions = []
                if filters:
                    for key, value in filters.items():
                        safe_value = self._sanitize_filter_value(value)
                        
                        if key == "semantic_type":
                            filter_conditions.append(f"semantic_type = '{safe_value}'")
                        elif key == "source_table":
                            filter_conditions.append(f"source_table = '{safe_value}'")
                        elif key == "ano_mes":
                            filter_conditions.append(f"ano_mes = '{safe_value}'")
                        elif key == "uf":
                            filter_conditions.append(f"uf = '{safe_value}'")
                
                filter_string = " AND ".join(filter_conditions) if filter_conditions else None
                
                # Executar busca vetorial usando método isolado
                search_results = self._vector_search(query_embedding, k, filter_string)
                
                # Processar resultados
                documents = []
                
                if "result" in search_results:
                    result = search_results["result"]
                    data_array = result.get("data_array", [])
                    scores = result.get("scores", [])
                    
                    for idx, row in enumerate(data_array):
                        try:
                            doc_id = row[0] if len(row) > 0 else f"doc_{idx}"
                            content = row[1] if len(row) > 1 else ""
                            metadata_json = row[2] if len(row) > 2 else "{}"
                            source_table = row[3] if len(row) > 3 else ""
                            semantic_type = row[4] if len(row) > 4 else ""
                            
                            # 4️⃣ Score de similaridade (maior = mais similar)
                            score = scores[idx] if idx < len(scores) else 0.8
                            
                            try:
                                metadata = json.loads(metadata_json) if metadata_json else {}
                            except json.JSONDecodeError:
                                metadata = {}
                            
                            metadata.update({
                                "doc_id": doc_id,
                                "source_table": source_table,
                                "semantic_type": semantic_type
                            })
                            
                            doc = Document(
                                page_content=content,
                                metadata=metadata
                            )
                            
                            documents.append((doc, score))
                            
                        except Exception as row_error:
                            print(f"⚠️ Erro ao processar linha {idx}: {row_error}")
                            continue
                
                # Ordenar por score (maior primeiro = mais similar)
                documents.sort(key=lambda x: x[1], reverse=True)
                
                print(f"✅ Busca concluída: {len(documents)} documentos encontrados")
                if documents:
                    print(f"   Score mais alto: {documents[0][1]:.4f}")
                
                return documents
                
            except Exception as e:
                if attempt < max_retries - 1:
                    # Backoff simples: 2, 4, 6 segundos
                    backoff_time = 2 * (attempt + 1)
                    print(f"⚠️ Tentativa {attempt + 1} falhou: {str(e)}. Tentando novamente em {backoff_time}s...")
                    time.sleep(backoff_time)
                    continue
                else:
                    # 3️⃣ Log final de erro após todas as tentativas
                    print(f"❌ Erro na busca vetorial após {max_retries} tentativas: {str(e)}")
                    print(f"   Query: '{query[:100]}...'")
                    print(f"   Índice: {self.full_index_name}")
                    
        # Retornar lista vazia se todas as tentativas falharam
        return []
    
    def search_by_type(
        self,
        query: str,
        semantic_type: str,
        k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        Busca filtrada por tipo semântico
        
        Args:
            query: Texto da busca
            semantic_type: 'metric', 'temporal', 'geographic', 'demographic'
            k: Número de resultados
        """
        filters = {"semantic_type": semantic_type}
        return self.search(query, k=k, filters=filters)
    
    # =========================================================================
    # MANUTENÇÃO
    # =========================================================================
    
    def sync_index(self) -> None:
        """Sincroniza índice com Delta Table (Delta Sync)"""
        try:
            self.client.sync_index(
                index_name=self.full_index_name
            )
            print(f"✅ Índice sincronizado")
        except Exception as e:
            print(f"❌ Erro ao sincronizar: {e}")
    
    def delete_index(self) -> None:
        """Deleta índice vetorial"""
        try:
            self.client.delete_index(
                index_name=self.full_index_name
            )
            print(f"✅ Índice deletado: {self.full_index_name}")
        except Exception as e:
            print(f"❌ Erro ao deletar: {e}")
    
    def get_index_stats(self) -> Dict:
        """Retorna estatísticas do índice"""
        try:
            index = self.client.get_index(
                endpoint_name=self.config.endpoint_name,
                index_name=self.full_index_name
            )
            
            return {
                "index_name": self.full_index_name,
                "status": getattr(index, 'status', 'unknown'),
                "num_rows": getattr(index, 'num_rows', 0),
                "dimension": self.config.embedding_dim
            }
        except Exception as e:
            print(f"❌ Erro ao obter stats: {e}")
            return {}
    
    @lru_cache(maxsize=50)
    def _get_cached_query_embedding(self, query: str) -> tuple:
        """8️⃣ Cache otimizado para embeddings de query usando @lru_cache"""
        embedding = self.embeddings.embed_query(query)
        return tuple(embedding)  # tuple para compatibilidade com lru_cache
    
    def _sanitize_filter_value(self, value: str) -> str:
        """2️⃣ Sanitizar valores de filtro usando whitelist (regex)"""
        # Whitelist: apenas alfanuméricos, hífens, underscores e espaços
        safe_value = re.sub(r'[^\w\s\-]', '', str(value))
        return safe_value.strip()
    
    def _embed_documents_in_batches(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """Fallback: gerar embeddings em lotes para contornar rate limits ou problemas de rede"""
        print(f"   📦 Processando {len(texts)} textos em lotes de {batch_size}...")
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(texts) - 1) // batch_size + 1
            
            print(f"   📦 Lote {batch_num}/{total_batches}: {len(batch)} textos")
            
            for attempt in range(3):  # 3 tentativas por lote
                try:
                    batch_embeddings = self.embeddings.embed_documents(batch)
                    all_embeddings.extend(batch_embeddings)
                    print(f"   ✅ Lote {batch_num} processado")
                    
                    # Delay respeitoso entre lotes
                    if i + batch_size < len(texts):  # Não delay no último lote
                        time.sleep(1)  # 1 segundo entre lotes
                    break
                except Exception as e:
                    if attempt < 2:
                        wait_time = 2 * (attempt + 1)  # 2, 4 segundos
                        print(f"   ⚠️ Lote {batch_num} falhou (tentativa {attempt + 1}): {e}")
                        print(f"   ⏳ Aguardando {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"   ❌ Lote {batch_num} falhou após 3 tentativas")
                        # Tentar lote unitário como último recurso
                        if batch_size > 1:
                            print(f"   🔄 Tentando lote {batch_num} unitariamente...")
                            for single_text in batch:
                                try:
                                    single_embedding = self.embeddings.embed_documents([single_text])
                                    all_embeddings.extend(single_embedding)
                                    time.sleep(0.5)  # 0.5s entre embeddings unitários
                                except:
                                    # Último recurso: embedding dummy
                                    dummy_embedding = [0.0] * self.config.embedding_dim
                                    all_embeddings.append(dummy_embedding)
                                    print(f"   ⚠️ Usando embedding dummy para 1 texto")
                        else:
                            # Já é unitário e falhou - usar dummy
                            dummy_embedding = [0.0] * self.config.embedding_dim
                            all_embeddings.extend([dummy_embedding] * len(batch))
                        break
        
        print(f"   ✅ Processamento concluído: {len(all_embeddings)} embeddings")
        return all_embeddings
    
    def _vector_search(
        self, 
        query_embedding: List[float], 
        k: int, 
        filter_string: Optional[str] = None
    ) -> Dict:
        """Método isolado para execução da busca vetorial"""
        try:
            # Tentativa 1: API do index object
            index = self.client.get_index(
                endpoint_name=f"{self.config.catalog}.{self.config.schema}.vs_endpoint",
                index_name=self.full_index_name
            )
            
            return index.similarity_search(
                query_vector=query_embedding,
                columns=["doc_id", "content", "metadata_json", "source_table", "semantic_type"],
                num_results=k,
                filters=filter_string
            )
            
        except Exception as e1:
            try:
                # Tentativa 2: API direta do client
                return self.client.search(
                    index_name=self.full_index_name,
                    query_vector=query_embedding,
                    columns=["doc_id", "content", "metadata_json", "source_table", "semantic_type"],
                    num_results=k,
                    filters=filter_string
                )
            except Exception as e2:
                # Fallback: busca simples
                print(f"⚠️ APIs de busca falharam: {e1}, {e2}")
                print("💡 Usando fallback para busca manual")
                
                # Retorno dummy para manter funcionalidade
                return {
                    "result": {
                        "data_array": [],
                        "row_count": 0
                    }
                }


# =============================================================================
# RAG RETRIEVER
# =============================================================================

class SRAGRetriever:
    """
    Retriever customizado para SRAG
    
    Implementa estratégias de busca híbridas:
        - Busca semântica
        - Filtros por tipo
        - Reranking por metadata
    """
    
    def __init__(self, vector_store_manager: DatabricksVectorStoreManager):
        self.vsm = vector_store_manager
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        strategy: str = "semantic"
    ) -> List[Document]:
        """
        Recupera documentos relevantes do Vector Store
        
        Args:
            query: Consulta em linguagem natural
            k: Número máximo de documentos a retornar
            strategy: 'semantic' | 'hybrid' | 'typed'
        
        Returns:
            Lista de Documents (nunca None, pode ser vazia)
        """
        # 7️⃣ NICE-TO-HAVE: Log mínimo de telemetria
        print(f"📊 Retrieval: strategy={strategy}, k={k}, query_len={len(query)}")
        
        # Validação básica
        if not query or not query.strip() or k <= 0:
            print("⚠️ Query inválida ou k <= 0, retornando lista vazia")
            return []
        
        try:
            if strategy == "semantic":
                documents = self._semantic_retrieve(query, k)
            elif strategy == "hybrid":
                documents = self._hybrid_retrieve(query, k)
            elif strategy == "typed":
                documents = self._typed_retrieve(query, k)
            else:
                print(f"⚠️ Strategy inválida '{strategy}', usando semantic")
                documents = self._semantic_retrieve(query, k)
            
            if not documents:
                print(f"⚠️ Nenhum documento encontrado para query: '{query[:50]}...'")
            else:
                print(f"✅ Retrieval concluído: {len(documents)} documentos")
            
            return documents
                
        except Exception as e:
            print(f"❌ Erro no retrieval: {e}")
            return []
    
    def _semantic_retrieve(self, query: str, k: int) -> List[Document]:
        """
        Busca semântica simples por similaridade
        """
        results = self.vsm.search(query, k=k)
        return [doc for doc, score in results]
    
    def _hybrid_retrieve(self, query: str, k: int) -> List[Document]:
        """
        Busca híbrida com reranking simples
        Regras: 1) Prioriza gold_resumo_geral, 2) Prioriza dados recentes, 3) Boost por metadata
        """
        # Buscar mais documentos para reranking
        search_k = min(k * 2, 15)
        results = self.vsm.search(query, k=search_k)
        
        if not results:
            return []
        
        query_lower = query.lower()
        
        # Aplicar reranking com regras simples + metadata boost
        ranked_results = []
        for doc, vector_score in results:
            score = vector_score
            
            # Regra 1: Bonus para tabela resumo geral (fonte principal)
            if doc.metadata.get("source_table") == "gold_resumo_geral":
                score *= 1.2
            
            # Regra 2: Bonus para dados de 2024-2025 (mais recentes) 
            ano_mes = doc.metadata.get("ano_mes", "")
            if "2025" in ano_mes or "2024" in ano_mes:
                score *= 1.1
            
            # 5️⃣ MELHORIA: Boost por metadata estruturada (heurística simples)
            # Boost se UF da query aparece nos metadados
            doc_uf = doc.metadata.get("uf", "").lower()
            if doc_uf and doc_uf in query_lower:
                score *= 1.15  # 15% boost por match geográfico
                
            # Boost se ano/mês da query aparece nos metadados  
            if ano_mes and (ano_mes.lower() in query_lower):
                score *= 1.1   # 10% boost por match temporal
            
            ranked_results.append((doc, score))
        
        # Ordenar por score e retornar top-k
        ranked_results.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in ranked_results[:k]]
    
    def _typed_retrieve(self, query: str, k: int) -> List[Document]:
        """
        Busca com detecção simples de tipo geográfico/temporal
        """
        query_lower = query.lower()
        semantic_type = None
        
        # Detectar geografia (estados brasileiros) - case insensitive
        uf_patterns = ["sp", "rj", "mg", "rs", "pr", "sc", "ba", "pe", "ce", "go", "mt", "ms", "ac", "al", "ap", "am", "df", "es", "ma", "pa", "pb", "pi", "rn", "ro", "rr", "se", "to", "estado", "uf"]
        if any(uf in query_lower for uf in uf_patterns):
            semantic_type = "geographic"
        
        # Detectar temporal
        elif any(termo in query_lower for termo in ["mês", "ano", "2024", "2025", "tendência", "janeiro", "dezembro"]):
            semantic_type = "temporal"
        
        # Buscar com ou sem filtro
        if semantic_type:
            results = self.vsm.search_by_type(query, semantic_type, k=k)
        else:
            results = self.vsm.search(query, k=k)
        
        return [doc for doc, score in results]

