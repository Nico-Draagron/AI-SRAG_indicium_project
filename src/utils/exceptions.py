"""
Custom Exceptions - Exceções Customizadas do Sistema SRAG
==========================================================

Hierarquia completa de exceções para tratamento de erros específicos
do sistema de monitoramento SRAG.

Features:
    - Hierarquia clara de exceções
    - Mensagens de erro descritivas
    - Códigos de erro estruturados
    - Context managers para tratamento
    - Logging automático
    - Recovery hints

Author: AI Engineer Certification - Indicium
Date: January 2025
Version: 2.0.0
"""

from typing import Optional, Dict, Any
from datetime import datetime


# =============================================================================
# EXCEÇÃO BASE
# =============================================================================

class SRAGSystemError(Exception):
    """
    Exceção base para todo o sistema SRAG
    
    Todas as exceções customizadas herdam desta classe base,
    permitindo tratamento hierárquico de erros.
    
    Attributes:
        message: Mensagem de erro
        error_code: Código numérico do erro
        details: Detalhes adicionais do erro
        timestamp: Momento da ocorrência
        recoverable: Se o erro é recuperável
        recovery_hint: Sugestão de recuperação
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        recoverable: bool = False,
        recovery_hint: Optional[str] = None
    ):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.timestamp = datetime.now()
        self.recoverable = recoverable
        self.recovery_hint = recovery_hint
        
        # Construir mensagem completa
        full_message = f"[{self.error_code}] {message}"
        if recovery_hint:
            full_message += f"\n💡 Sugestão: {recovery_hint}"
        
        super().__init__(full_message)
    
    def to_dict(self) -> Dict:
        """Converte exceção para dicionário"""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "recoverable": self.recoverable,
            "recovery_hint": self.recovery_hint
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(code={self.error_code}, message={self.message})"


# =============================================================================
# EXCEÇÕES DE ORQUESTRAÇÃO
# =============================================================================

class OrchestratorError(SRAGSystemError):
    """Erro no orquestrador principal"""
    
    def __init__(
        self,
        message: str,
        node_name: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if node_name:
            details["node_name"] = node_name
        kwargs["details"] = details
        
        super().__init__(message, **kwargs)


class NodeExecutionError(OrchestratorError):
    """Erro na execução de um nó do grafo"""
    
    def __init__(self, node_name: str, message: str, **kwargs):
        super().__init__(
            message=f"Erro no nó '{node_name}': {message}",
            node_name=node_name,
            **kwargs
        )


class StateTransitionError(OrchestratorError):
    """Erro na transição de estado do agente"""
    pass


class WorkflowError(OrchestratorError):
    """Erro no fluxo de trabalho do LangGraph"""
    pass


# =============================================================================
# EXCEÇÕES DE COLETA DE DADOS
# =============================================================================

class DataCollectionError(SRAGSystemError):
    """Erro base para coleta de dados"""
    pass


class MetricsCollectionError(DataCollectionError):
    """Erro ao coletar métricas epidemiológicas"""
    
    def __init__(self, message: str, metric_type: Optional[str] = None, **kwargs):
        details = kwargs.get("details", {})
        if metric_type:
            details["metric_type"] = metric_type
        kwargs["details"] = details
        
        super().__init__(
            message,
            recovery_hint="Verifique se as tabelas Gold estão atualizadas e acessíveis",
            **kwargs
        )


class NewsCollectionError(DataCollectionError):
    """Erro ao coletar notícias"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            recoverable=True,
            recovery_hint="O sistema pode continuar sem notícias. Verifique a API Tavily.",
            **kwargs
        )


class GeographicDataError(DataCollectionError):
    """Erro ao coletar dados geográficos"""
    pass


class DemographicDataError(DataCollectionError):
    """Erro ao coletar dados demográficos"""
    pass


# =============================================================================
# EXCEÇÕES SQL
# =============================================================================

class SQLError(SRAGSystemError):
    """Erro base para operações SQL"""
    
    def __init__(self, message: str, query: Optional[str] = None, **kwargs):
        details = kwargs.get("details", {})
        if query:
            details["query"] = query[:200]  # Truncar query
        kwargs["details"] = details
        
        super().__init__(message, **kwargs)


class SQLExecutionError(SQLError):
    """Erro na execução de query SQL"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            recovery_hint="Verifique a sintaxe SQL e disponibilidade das tabelas",
            **kwargs
        )


class SQLValidationError(SQLError):
    """Erro na validação de query SQL"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            recoverable=False,
            recovery_hint="Ajuste a query para atender aos guardrails de segurança",
            **kwargs
        )


class QueryTimeoutError(SQLError):
    """Timeout na execução de query"""
    
    def __init__(self, message: str = "Query excedeu tempo limite", **kwargs):
        super().__init__(
            message,
            recoverable=True,
            recovery_hint="Adicione filtros para reduzir volume de dados processados",
            **kwargs
        )


class TableNotFoundError(SQLError):
    """Tabela não encontrada"""
    
    def __init__(self, table_name: str, **kwargs):
        super().__init__(
            f"Tabela '{table_name}' não encontrada",
            error_code="TABLE_NOT_FOUND",
            details={"table_name": table_name},
            **kwargs
        )


class InsufficientDataError(SQLError):
    """Dados insuficientes retornados"""
    
    def __init__(self, message: str = "Dados insuficientes para análise", **kwargs):
        super().__init__(message, recoverable=True, **kwargs)


# =============================================================================
# EXCEÇÕES DE GUARDRAILS
# =============================================================================

class GuardrailViolation(SRAGSystemError):
    """Violação de guardrail de segurança"""
    
    def __init__(
        self,
        message: str,
        violation_type: Optional[str] = None,
        severity: str = "HIGH",
        **kwargs
    ):
        details = kwargs.get("details", {})
        details.update({
            "violation_type": violation_type,
            "severity": severity
        })
        kwargs["details"] = details
        
        super().__init__(
            message,
            error_code=f"GUARDRAIL_{violation_type}" if violation_type else "GUARDRAIL_VIOLATION",
            recoverable=False,
            **kwargs
        )


class SQLInjectionDetected(GuardrailViolation):
    """Tentativa de SQL injection detectada"""
    
    def __init__(self, pattern: str, **kwargs):
        super().__init__(
            f"Padrão de SQL injection detectado: {pattern}",
            violation_type="SQL_INJECTION",
            severity="CRITICAL",
            **kwargs
        )


class ForbiddenCommandError(GuardrailViolation):
    """Comando SQL proibido"""
    
    def __init__(self, command: str, **kwargs):
        super().__init__(
            f"Comando proibido detectado: {command}",
            violation_type="FORBIDDEN_COMMAND",
            severity="CRITICAL",
            details={"command": command},
            **kwargs
        )


class UnauthorizedTableAccess(GuardrailViolation):
    """Acesso a tabela não autorizada"""
    
    def __init__(self, table: str, **kwargs):
        super().__init__(
            f"Acesso não autorizado à tabela: {table}",
            violation_type="UNAUTHORIZED_TABLE",
            severity="HIGH",
            details={"table": table},
            **kwargs
        )


class RateLimitExceeded(GuardrailViolation):
    """Limite de taxa excedido"""
    
    def __init__(self, limit_type: str = "queries", **kwargs):
        super().__init__(
            f"Limite de {limit_type} excedido",
            violation_type="RATE_LIMIT",
            severity="MEDIUM",
            recoverable=True,
            recovery_hint="Aguarde alguns minutos antes de tentar novamente",
            **kwargs
        )


class PIIDetectedError(GuardrailViolation):
    """Dados sensíveis (PII) detectados"""
    
    def __init__(self, pii_type: str, **kwargs):
        super().__init__(
            f"PII detectado: {pii_type}",
            violation_type="PII_DETECTED",
            severity="LOW",
            details={"pii_type": pii_type},
            **kwargs
        )


# =============================================================================
# EXCEÇÕES DE WEB SEARCH
# =============================================================================

class WebSearchError(SRAGSystemError):
    """Erro base para busca web"""
    pass


class SearchAPIError(WebSearchError):
    """Erro na API de busca (Tavily)"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            recoverable=True,
            recovery_hint="Verifique conectividade e quota da API Tavily",
            **kwargs
        )


class SearchValidationError(WebSearchError):
    """Erro na validação de parâmetros de busca"""
    pass


class NoResultsFoundError(WebSearchError):
    """Nenhum resultado encontrado na busca"""
    
    def __init__(self, query: str, **kwargs):
        super().__init__(
            f"Nenhum resultado encontrado para: {query}",
            recoverable=True,
            details={"query": query},
            **kwargs
        )


# =============================================================================
# EXCEÇÕES DE GRÁFICOS
# =============================================================================

class ChartError(SRAGSystemError):
    """Erro base para geração de gráficos"""
    pass


class ChartGenerationError(ChartError):
    """Erro ao gerar gráfico"""
    
    def __init__(self, message: str, chart_type: Optional[str] = None, **kwargs):
        details = kwargs.get("details", {})
        if chart_type:
            details["chart_type"] = chart_type
        kwargs["details"] = details
        
        super().__init__(
            message,
            recoverable=True,
            recovery_hint="O sistema pode continuar sem gráficos",
            **kwargs
        )


class ChartValidationError(ChartError):
    """Erro na validação de dados para gráfico"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            recovery_hint="Verifique se os dados têm as colunas necessárias",
            **kwargs
        )


class ChartExportError(ChartError):
    """Erro ao exportar gráfico"""
    
    def __init__(self, format: str, message: str, **kwargs):
        super().__init__(
            f"Erro ao exportar gráfico em {format}: {message}",
            details={"format": format},
            **kwargs
        )


# =============================================================================
# EXCEÇÕES DE RELATÓRIOS
# =============================================================================

class ReportError(SRAGSystemError):
    """Erro base para geração de relatórios"""
    pass


class ReportGenerationError(ReportError):
    """Erro ao gerar relatório"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="REPORT_GENERATION_FAILED",
            **kwargs
        )


class ReportValidationError(ReportError):
    """Erro na validação do relatório gerado"""
    
    def __init__(self, message: str, missing_sections: Optional[list] = None, **kwargs):
        details = kwargs.get("details", {})
        if missing_sections:
            details["missing_sections"] = missing_sections
        kwargs["details"] = details
        
        super().__init__(
            message,
            recovery_hint="O LLM deve gerar todas as seções obrigatórias",
            **kwargs
        )


class LLMError(ReportError):
    """Erro na comunicação com LLM"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            recoverable=True,
            recovery_hint="Verifique API key e conectividade com OpenAI",
            **kwargs
        )


# =============================================================================
# EXCEÇÕES DE CONFIGURAÇÃO
# =============================================================================

class ConfigurationError(SRAGSystemError):
    """Erro de configuração do sistema"""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        details = kwargs.get("details", {})
        if config_key:
            details["config_key"] = config_key
        kwargs["details"] = details
        
        super().__init__(message, **kwargs)


class MissingCredentialsError(ConfigurationError):
    """Credenciais ausentes ou inválidas"""
    
    def __init__(self, credential_name: str, **kwargs):
        super().__init__(
            f"Credencial ausente: {credential_name}",
            error_code="MISSING_CREDENTIALS",
            details={"credential": credential_name},
            recovery_hint=f"Configure {credential_name} no Databricks Secrets",
            **kwargs
        )


class InvalidConfigError(ConfigurationError):
    """Configuração inválida"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            recovery_hint="Verifique o arquivo de configuração",
            **kwargs
        )


# =============================================================================
# EXCEÇÕES DE AUDITORIA
# =============================================================================

class AuditError(SRAGSystemError):
    """Erro no sistema de auditoria"""
    pass


class AuditLogSaveError(AuditError):
    """Erro ao salvar logs de auditoria"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            recoverable=True,
            recovery_hint="Logs ficam em memória. Verifique permissões no Delta Lake",
            **kwargs
        )


# =============================================================================
# EXCEÇÕES DE CACHE
# =============================================================================

class CacheError(SRAGSystemError):
    """Erro no sistema de cache"""
    pass


class CacheFullError(CacheError):
    """Cache cheio"""
    
    def __init__(self, **kwargs):
        super().__init__(
            "Cache atingiu capacidade máxima",
            recoverable=True,
            recovery_hint="Cache será limpo automaticamente (LRU)",
            **kwargs
        )


# =============================================================================
# UTILITÁRIOS PARA TRATAMENTO DE EXCEÇÕES
# =============================================================================

class ErrorContext:
    """Context manager para tratamento padronizado de erros"""
    
    def __init__(
        self,
        operation_name: str,
        audit_logger = None,
        raise_on_error: bool = True
    ):
        self.operation_name = operation_name
        self.audit_logger = audit_logger
        self.raise_on_error = raise_on_error
        self.error = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error = exc_val
            
            # Log do erro
            if self.audit_logger:
                self.audit_logger.log_event(
                    "ERROR",
                    {
                        "operation": self.operation_name,
                        "error_type": exc_type.__name__,
                        "error_message": str(exc_val)
                    },
                    status="ERROR"
                )
            
            # Se for erro recuperável, não propagar
            if isinstance(exc_val, SRAGSystemError) and exc_val.recoverable:
                if not self.raise_on_error:
                    return True  # Suprimir exceção
        
        return False  # Propagar exceção


def format_error_for_user(error: Exception) -> str:
    """
    Formata erro de forma amigável para o usuário
    
    Args:
        error: Exceção capturada
        
    Returns:
        Mensagem formatada
    """
    if isinstance(error, SRAGSystemError):
        msg = f"❌ Erro: {error.message}"
        
        if error.recovery_hint:
            msg += f"\n\n💡 Sugestão: {error.recovery_hint}"
        
        if error.recoverable:
            msg += "\n\n✅ Este erro é recuperável. O sistema pode continuar."
        else:
            msg += "\n\n⚠️ Este erro requer intervenção manual."
        
        return msg
    else:
        return f"❌ Erro inesperado: {str(error)}"


def is_recoverable(error: Exception) -> bool:
    """Verifica se um erro é recuperável"""
    if isinstance(error, SRAGSystemError):
        return error.recoverable
    return False


# =============================================================================
# EXCEÇÕES PARA VALIDAÇÃO DE ENTRADA
# =============================================================================

class ValidationError(SRAGSystemError):
    """Erro de validação de entrada"""
    pass


class InvalidQueryError(ValidationError):
    """Query de usuário inválida"""
    
    def __init__(self, reason: str, **kwargs):
        super().__init__(
            f"Query inválida: {reason}",
            recovery_hint="Reformule a pergunta de forma mais clara",
            **kwargs
        )


class InvalidParameterError(ValidationError):
    """Parâmetro inválido"""
    
    def __init__(self, param_name: str, reason: str, **kwargs):
        super().__init__(
            f"Parâmetro '{param_name}' inválido: {reason}",
            details={"parameter": param_name, "reason": reason},
            **kwargs
        )
# =============================================================================
# EXCEÇÕES RAG
# =============================================================================

class RAGError(SRAGSystemError):
    """Erro base para sistema RAG"""
    pass


class EmbeddingError(RAGError):
    """Erro ao gerar embeddings"""
    
    def __init__(self, message: str, provider: Optional[str] = None, **kwargs):
        details = kwargs.get("details", {})
        if provider:
            details["provider"] = provider
        kwargs["details"] = details
        
        super().__init__(
            message,
            recoverable=True,
            recovery_hint="Verifique API key do provider de embeddings",
            **kwargs
        )


class VectorStoreError(RAGError):
    """Erro no vector store"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            recoverable=True,
            recovery_hint="Verifique conexão com Databricks Vector Search",
            **kwargs
        )


class RetrievalError(RAGError):
    """Erro no retrieval de documentos"""
    
    def __init__(self, message: str, query: Optional[str] = None, **kwargs):
        details = kwargs.get("details", {})
        if query:
            details["query"] = query[:100]
        kwargs["details"] = details
        
        super().__init__(
            message,
            recoverable=True,
            **kwargs
        )


class ContextBuildError(RAGError):
    """Erro ao construir contexto para LLM"""
    pass


class DocumentLoaderError(RAGError):
    """Erro ao carregar documentos do Gold"""
    pass
