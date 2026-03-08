"""
Utils — Componentes Transversais do Pipeline SRAG
==================================================

Ponto único de importação para os três módulos de infraestrutura:

    guardrails  — validação SQL em 7 camadas e sanitização de PII
    audit       — rastreabilidade de eventos e métricas de performance
    exceptions  — hierarquia de exceções customizadas

Uso recomendado
---------------
Importar pelo caminho curto via este pacote:

    from src.utils import SQLGuardrails, AuditLogger, ChartGenerationError

em vez de importar diretamente dos submódulos. Isso desacopla os callers
da organização interna dos arquivos, que pode mudar sem quebrar os imports.

Nota sobre exportação de exceções
----------------------------------
Todas as subclasses específicas de exceção são exportadas explicitamente.
Omitir subclasses forçaria callers a capturar apenas a base genérica
SRAGSystemError, eliminando o benefício da hierarquia — um bloco
except ChartGenerationError nunca seria ativado se ChartGenerationError
não fosse importável via este pacote.
"""

# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------
from src.utils.guardrails import (
    SQLGuardrails,
    RateLimiter,
    ViolationSeverity,
    ViolationRecord,
    PiiEvent,
    GuardrailsConfig,
)

# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------
from src.utils.audit import (
    AuditLogger,
    AuditAnalyzer,
    AuditEvent,
    AuditLogEntry,
    EventStatus,
    SessionSummary,
)

# ---------------------------------------------------------------------------
# Exceptions — base
# ---------------------------------------------------------------------------
from src.utils.exceptions import (
    SRAGSystemError,
    ErrorContext,
    format_error_for_user,
    is_recoverable,
)

# Orquestracao
from src.utils.exceptions import (
    OrchestratorError,
    NodeExecutionError,
    StateTransitionError,
    WorkflowError,
)

# Coleta de dados
from src.utils.exceptions import (
    DataCollectionError,
    MetricsCollectionError,
    NewsCollectionError,
    GeographicDataError,
    DemographicDataError,
)

# SQL
from src.utils.exceptions import (
    SQLError,
    SQLExecutionError,
    SQLValidationError,
    QueryTimeoutError,
    TableNotFoundError,
    InsufficientDataError,
)

# Guardrails
from src.utils.exceptions import (
    GuardrailViolation,
    SQLInjectionDetected,
    ForbiddenCommandError,
    UnauthorizedTableAccess,
    RateLimitExceeded,
    PIIDetectedError,
)

# Web search
from src.utils.exceptions import (
    WebSearchError,
    SearchAPIError,
    SearchValidationError,
    NoResultsFoundError,
)

# Graficos
from src.utils.exceptions import (
    ChartError,
    ChartGenerationError,
    ChartValidationError,
    ChartExportError,
)

# Relatorios
from src.utils.exceptions import (
    ReportError,
    ReportGenerationError,
    ReportValidationError,
    LLMError,
)

# Configuracao
from src.utils.exceptions import (
    ConfigurationError,
    MissingCredentialsError,
    InvalidConfigError,
)

# Auditoria
from src.utils.exceptions import (
    AuditError,
    AuditLogSaveError,
)

# Cache
from src.utils.exceptions import (
    CacheError,
    CacheFullError,
)

# RAG
from src.utils.exceptions import (
    RAGError,
    EmbeddingError,
    VectorStoreError,
    RetrievalError,
    ContextBuildError,
    DocumentLoaderError,
)

# Validacao de entrada
from src.utils.exceptions import (
    ValidationError,
    InvalidQueryError,
    InvalidParameterError,
)

# ---------------------------------------------------------------------------
# API publica declarada explicitamente
# ---------------------------------------------------------------------------
__all__ = [
    # --- guardrails ---
    "SQLGuardrails",
    "RateLimiter",
    "ViolationSeverity",
    "ViolationRecord",
    "PiiEvent",
    "GuardrailsConfig",

    # --- audit ---
    "AuditLogger",
    "AuditAnalyzer",
    "AuditEvent",
    "AuditLogEntry",
    "EventStatus",
    "SessionSummary",

    # --- exceptions: utilitarios ---
    "SRAGSystemError",
    "ErrorContext",
    "format_error_for_user",
    "is_recoverable",

    # --- exceptions: orquestracao ---
    "OrchestratorError",
    "NodeExecutionError",
    "StateTransitionError",
    "WorkflowError",

    # --- exceptions: coleta de dados ---
    "DataCollectionError",
    "MetricsCollectionError",
    "NewsCollectionError",
    "GeographicDataError",
    "DemographicDataError",

    # --- exceptions: SQL ---
    "SQLError",
    "SQLExecutionError",
    "SQLValidationError",
    "QueryTimeoutError",
    "TableNotFoundError",
    "InsufficientDataError",

    # --- exceptions: guardrails ---
    "GuardrailViolation",
    "SQLInjectionDetected",
    "ForbiddenCommandError",
    "UnauthorizedTableAccess",
    "RateLimitExceeded",
    "PIIDetectedError",

    # --- exceptions: web search ---
    "WebSearchError",
    "SearchAPIError",
    "SearchValidationError",
    "NoResultsFoundError",

    # --- exceptions: graficos ---
    "ChartError",
    "ChartGenerationError",
    "ChartValidationError",
    "ChartExportError",

    # --- exceptions: relatorios ---
    "ReportError",
    "ReportGenerationError",
    "ReportValidationError",
    "LLMError",

    # --- exceptions: configuracao ---
    "ConfigurationError",
    "MissingCredentialsError",
    "InvalidConfigError",

    # --- exceptions: auditoria ---
    "AuditError",
    "AuditLogSaveError",

    # --- exceptions: cache ---
    "CacheError",
    "CacheFullError",

    # --- exceptions: RAG ---
    "RAGError",
    "EmbeddingError",
    "VectorStoreError",
    "RetrievalError",
    "ContextBuildError",
    "DocumentLoaderError",

    # --- exceptions: validacao ---
    "ValidationError",
    "InvalidQueryError",
    "InvalidParameterError",
]