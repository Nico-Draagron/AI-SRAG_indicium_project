"""
SRAG System - Sistema de IA para Análise de Dados SRAG
=======================================================

Sistema completo de IA para análise dos dados SRAG com:
- Arquitetura em camadas (Bronze, Silver, Gold)
- Sistema RAG para recuperação de informações
- Agentes inteligentes com LangGraph
- Guardrails de segurança
- Auditoria completa

Módulos:
- agents: Orquestração e roteamento
- tools: Ferramentas especializadas  
- rag: Sistema RAG completo
- utils: Componentes transversais
"""

# Core components
from .agents import SRAGOrchestrator, IntentRouter
from .tools import GoldSQLTool, ReportGenerator
from .rag import SRAGChain, GoldDocumentLoader
from .utils import SQLGuardrails, AuditLogger

__version__ = "1.0.0"
__author__ = "Indicium Tech"

__all__ = [
    # Main classes
    "SRAGOrchestrator",
    "IntentRouter", 
    "GoldSQLTool",
    "ReportGenerator",
    "SRAGChain",
    "GoldDocumentLoader",
    "SQLGuardrails",
    "AuditLogger",
    
    # Metadata
    "__version__",
    "__author__"
]