"""
Utils - Componentes Transversais
=================================

Componentes de suporte e infraestrutura:
- SQLGuardrails: Sistema de validação em 7 camadas
- AuditLogger: Sistema de auditoria completo
- Exceptions: Hierarquia de exceções customizadas
"""

from src.utils.guardrails import (
    SQLGuardrails,
    RateLimiter,
    ViolationSeverity,
    GuardrailsConfig
)
from src.utils.audit import (
    AuditLogger,
    AuditEvent,
    EventStatus
)
from src.utils.exceptions import (
    SRAGSystemError,
    OrchestratorError,
    SQLError,
    SQLExecutionError,
    SQLValidationError,
    RAGError,
    GuardrailViolation
)

__all__ = [
    # Guardrails
    "SQLGuardrails",
    "RateLimiter",
    "ViolationSeverity",
    "GuardrailsConfig",
    
    # Audit
    "AuditLogger",
    "AuditEvent",
    "EventStatus",
    
    # Exceptions
    "SRAGSystemError",
    "OrchestratorError", 
    "SQLError",
    "SQLExecutionError",
    "SQLValidationError", 
    "RAGError",
    "GuardrailViolation"
]