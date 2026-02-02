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

from .document_loader import GoldDocumentLoader, SRAGDocument
from .vector_store import (
    EmbeddingManager,
    DatabricksVectorStoreManager,
    SRAGRetriever,
    VectorStoreConfig
)
from .rag_chain import SRAGChain, RAGConfig, ContextBuilder

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


# =============================================================================
# src/agents/__init__.py
# =============================================================================

"""
Agents - Orquestração e Routing
================================

Sistema de agentes para orquestração:
- Orchestrator: LangGraph workflow (SQL + RAG + Synthesis)
- Intent Router: Decisão SQL vs RAG vs Hybrid
"""

from .orchestrator import SRAGOrchestrator, AgentState
from .intent_router import (
    IntentRouter,
    RoutingDecision,
    ExecutionStrategy,
    QueryIntent
)

__all__ = [
    # Orchestrator
    "SRAGOrchestrator",
    "AgentState",
    
    # Intent Router
    "IntentRouter",
    "RoutingDecision",
    "ExecutionStrategy",
    "QueryIntent"
]


# =============================================================================
# src/tools/__init__.py
# =============================================================================

"""
Tools - Ferramentas Especializadas
===================================

Ferramentas do agente:
- SQL Tool: Execução segura de queries Gold
"""

from .sql_tool import GoldSQLTool

__all__ = [
    "GoldSQLTool"
]


# =============================================================================
# src/utils/__init__.py
# =============================================================================

"""
Utils - Componentes Transversais
=================================

Utilidades compartilhadas:
- Guardrails: Validação SQL + PII
- Audit: Logging estruturado
- Exceptions: Hierarquia de erros customizados
"""

from .guardrails import SQLGuardrails, GuardrailsConfig, ViolationSeverity
from .audit import AuditLogger, AuditEvent, EventStatus
from .exceptions import (
    # Base
    SRAGSystemError,
    
    # Orchestrator
    OrchestratorError,
    NodeExecutionError,
    
    # SQL
    SQLError,
    SQLValidationError,
    QueryExecutionError,
    TableNotFoundError,
    
    # RAG
    RAGError,
    EmbeddingError,
    VectorStoreError,
    RetrievalError,
    
    # Guardrails
    GuardrailViolation,
    SQLInjectionDetected,
    ForbiddenCommandError
)

__all__ = [
    # Guardrails
    "SQLGuardrails",
    "GuardrailsConfig",
    "ViolationSeverity",
    
    # Audit
    "AuditLogger",
    "AuditEvent",
    "EventStatus",
    
    # Exceptions - Base
    "SRAGSystemError",
    
    # Exceptions - Orchestrator
    "OrchestratorError",
    "NodeExecutionError",
    
    # Exceptions - SQL
    "SQLError",
    "SQLValidationError",
    "QueryExecutionError",
    "TableNotFoundError",
    
    # Exceptions - RAG
    "RAGError",
    "EmbeddingError",
    "VectorStoreError",
    "RetrievalError",
    
    # Exceptions - Guardrails
    "GuardrailViolation",
    "SQLInjectionDetected",
    "ForbiddenCommandError"
]