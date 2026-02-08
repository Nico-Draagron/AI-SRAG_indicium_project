# =============================================================================
# src/rag/__init__.py
# =============================================================================

"""
RAG System - Retrieval Augmented Generation
============================================

Sistema completo de RAG para SRAG:
- Document Loader: Converte tabelas Gold em documentos
- Vector Store: Databricks Vector Search + embeddings
- RAG Chain: Retrieval + generation com LLM
"""

from src.rag.document_loader import GoldDocumentLoader, SRAGDocument
from src.rag.vector_store import (
    EmbeddingManager,
    DatabricksVectorStoreManager,
    SRAGRetriever,
    VectorStoreConfig
)
from src.rag.rag_chain import SRAGChain, RAGConfig, ContextBuilder

__all__ = [
    # Document Loader
    "GoldDocumentLoader",
    "SRAGDocument",
    
    # Vector Store
    "EmbeddingManager",
    "DatabricksVectorStoreManager",
    "SRAGRetriever",
    "VectorStoreConfig",
    
    # RAG Chain
    "SRAGChain",
    "RAGConfig",
    "ContextBuilder"
]