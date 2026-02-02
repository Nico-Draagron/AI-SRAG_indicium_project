"""
Utils - Componentes Transversais
=================================

Componentes de suporte e infraestrutura:
- SQLGuardrails: Sistema de validação em 7 camadas
- AuditLogger: Sistema de auditoria completo
- Exceptions: Hierarquia de exceções customizadas
"""

from .guardrails import (
    SQLGuardrails,
    PIIFilter,
    RateLimiter,
    ViolationSeverity,
    GuardrailsConfig
)
from .audit import (
    AuditLogger,
    AuditEvent,
    EventStatus
)
from .exceptions import (
    SRAGSystemError,
    OrchestratorError,
    SQLError,
    RAGError,
    GuardrailViolation
)

__all__ = [
    # Guardrails
    "SQLGuardrails",
    "PIIFilter",
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
    "RAGError",
    "GuardrailViolation"
]