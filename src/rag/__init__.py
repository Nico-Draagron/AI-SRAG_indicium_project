"""
RAG System — Retrieval Augmented Generation para SRAG
======================================================

API pública do módulo src/rag/.

Componentes
-----------
document_loader
    GoldDocumentLoader: carrega tabelas Gold e converte em SRAGDocuments.
    SRAGDocument: documento semântico intermediário (Gold → Vector Store).

vector_store
    EmbeddingManager: factory de embeddings (Databricks BGE Large / HuggingFace).
    VectorStoreConfig: configuração do índice e tabela Delta.
    DatabricksVectorStoreManager: gerencia ciclo completo do Vector Search.
    SRAGRetriever: retrieval com estratégias semantic, hybrid e typed.

rag_chain
    RAGConfig: parâmetros de retrieval e contexto.
    ContextBuilder: monta contexto textual para o LLM.
    ResponseValidator: valida qualidade da resposta gerada.
    SRAGChain: pipeline RAG completo (retrieval → LLM → validação).
    ConversationalSRAGChain: SRAGChain com memória de histórico em memória.
"""

from src.rag.document_loader import GoldDocumentLoader, SRAGDocument
from src.rag.vector_store import (
    DatabricksVectorStoreManager,
    EmbeddingManager,
    SRAGRetriever,
    VectorStoreConfig,
)
from src.rag.rag_chain import (
    ContextBuilder,
    ConversationalSRAGChain,
    RAGConfig,
    ResponseValidator,
    SRAGChain,
)

__all__ = [
    # document_loader
    "GoldDocumentLoader",
    "SRAGDocument",
    # vector_store
    "DatabricksVectorStoreManager",
    "EmbeddingManager",
    "SRAGRetriever",
    "VectorStoreConfig",
    # rag_chain
    "ContextBuilder",
    "ConversationalSRAGChain",
    "RAGConfig",
    "ResponseValidator",
    "SRAGChain",
]