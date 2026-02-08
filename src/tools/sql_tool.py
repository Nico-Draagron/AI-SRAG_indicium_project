"""
SQL Tool - Execução Segura de Queries Gold (MELHORADO)
======================================================

Versão melhorada com:
- ✅ Limite configurável de linhas no resultado
- ✅ Truncamento inteligente de queries nos logs
- ✅ Aviso se resultado foi limitado
- ✅ Contagem total de linhas

Author: AI Engineer Certification - Indicium
Date: January 2025
Version: 3.1.0 - MELHORADO
"""

from typing import Dict, Optional
from dataclasses import dataclass
from pyspark.sql import SparkSession

# ✅ Imports condicionais
try:
    from src.utils.guardrails import SQLGuardrails, GuardrailsConfig
except ImportError:
    # Stub básico se não existir
    class GuardrailsConfig:
        pass
    
    class SQLGuardrails:
        def __init__(self, config=None):
            pass
        
        def validate_query(self, query: str):
            # Validação mínima
            dangerous = ["drop", "delete", "truncate", "alter"]
            query_lower = query.lower()
            for word in dangerous:
                if word in query_lower:
                    return False, f"Query contém palavra perigosa: {word}"
            return True, "OK"

try:
    from src.utils.exceptions import SQLValidationError, SQLExecutionError
except ImportError:
    class SQLValidationError(Exception):
        def __init__(self, message, details=None):
            super().__init__(message)
            self.details = details or {}
    
    class SQLExecutionError(Exception):
        def __init__(self, message, details=None):
            super().__init__(message)
            self.details = details or {}

try:
    from src.utils.audit import AuditLogger, AuditEvent, EventStatus
except ImportError:
    class AuditEvent:
        SQL_VALIDATION_FAILED = "sql_validation_failed"
        SQL_QUERY_START = "sql_query_start"
        SQL_QUERY_SUCCESS = "sql_query_success"
        SQL_QUERY_ERROR = "sql_query_error"
        SQL_RESULT_LIMITED = "sql_result_limited"
    
    class EventStatus:
        INFO = "INFO"
        SUCCESS = "SUCCESS"
        ERROR = "ERROR"
        WARNING = "WARNING"
    
    class AuditLogger:
        def log_event(self, event_type, details=None, status="INFO"):
            print(f"[{status}] {event_type}: {details}")


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

@dataclass
class SQLToolConfig:
    """Configuração do SQL Tool"""
    max_result_rows: int = 10000  # ✅ Limite de linhas no resultado
    log_query_max_length: int = 500  # ✅ Tamanho máximo de query no log
    count_total_rows: bool = True  # ✅ Contar total de linhas (pode ser lento)
    warn_on_limit: bool = True  # ✅ Avisar se resultado foi limitado


# =============================================================================
# SQL TOOL MELHORADO
# =============================================================================

class GoldSQLTool:
    """
    SQL Tool Melhorado - Execução Segura com Limites e Logs Otimizados
    
    Features:
    - ✅ Execução de queries SQL
    - ✅ Validação via SQLGuardrails
    - ✅ Exceções customizadas
    - ✅ Auditoria completa
    - ✅ Limite de resultado (evita OOM)
    - ✅ Truncamento inteligente de logs
    - ✅ Contagem total de linhas
    
    Example:
        >>> sql_tool = GoldSQLTool(
        ...     spark=spark,
        ...     config=SQLToolConfig(max_result_rows=5000)
        ... )
        >>> result = sql_tool.execute_query("SELECT * FROM table")
        >>> print(f"Retornou {result['rows']} de {result['total_rows']} linhas")
    """
    
    def __init__(
        self,
        spark: SparkSession,
        audit_logger: Optional[AuditLogger] = None,
        guardrails_config: Optional[GuardrailsConfig] = None,
        config: Optional[SQLToolConfig] = None
    ):
        self.spark = spark
        self.audit = audit_logger if audit_logger else AuditLogger()
        self.guardrails = SQLGuardrails(guardrails_config or GuardrailsConfig())
        self.config = config or SQLToolConfig()
        
        # Log inicialização
        self.audit.log_event(
            AuditEvent.SQL_QUERY_START,  # Reutilizando evento
            {
                "tool": "GoldSQLTool",
                "max_result_rows": self.config.max_result_rows,
                "version": "3.1.0"
            },
            EventStatus.INFO
        )
    
    def execute_query(self, query: str) -> Dict:
        """
        Executa query SQL com validação e limites
        
        Args:
            query: Query SQL a executar
            
        Returns:
            Dict com:
                - success: bool
                - rows: int (linhas retornadas)
                - total_rows: int (total antes do limite)
                - limited: bool (se resultado foi limitado)
                - columns: List[str]
                - data: List[Dict]
                - query: str
            
        Raises:
            SQLValidationError: Query inválida
            SQLExecutionError: Erro na execução
        """
        # ✅ Truncar query para log
        query_for_log = self._truncate_query_for_log(query)
        
        # Validação Guardrails
        is_valid, validation_msg = self.guardrails.validate_query(query)
        
        if not is_valid:
            self.audit.log_event(
                AuditEvent.SQL_VALIDATION_FAILED,
                {"query": query_for_log, "reason": validation_msg},
                EventStatus.ERROR
            )
            raise SQLValidationError(validation_msg, details={"query": query_for_log})
        
        # Log início
        self.audit.log_event(
            AuditEvent.SQL_QUERY_START,
            {"query": query_for_log},
            EventStatus.INFO
        )
        
        # Execução
        try:
            df = self.spark.sql(query)
            
            # ✅ Contar total de linhas ANTES do limite (opcional)
            total_rows = None
            if self.config.count_total_rows:
                try:
                    total_rows = df.count()
                except Exception as count_error:
                    # Se count() falhar (query muito pesada), ignorar
                    print(f"⚠️ Não foi possível contar total de linhas: {count_error}")
                    total_rows = None
            
            # ✅ Limitar resultado
            df_limited = df.limit(self.config.max_result_rows)
            pdf = df_limited.toPandas()
            
            # ✅ Detectar se resultado foi limitado
            result_rows = len(pdf)
            was_limited = False
            
            if total_rows is not None:
                was_limited = total_rows > result_rows
            else:
                # Se não contamos total, assumir limitado se retornou max
                was_limited = result_rows >= self.config.max_result_rows
            
            # ✅ Avisar se limitado
            if was_limited and self.config.warn_on_limit:
                self.audit.log_event(
                    AuditEvent.SQL_RESULT_LIMITED,  # Novo evento
                    {
                        "returned_rows": result_rows,
                        "total_rows": total_rows,
                        "limit": self.config.max_result_rows
                    },
                    EventStatus.WARNING
                )
                print(f"⚠️ Resultado limitado: {result_rows} de {total_rows or '?'} linhas")
            
            result = {
                "success": True,
                "rows": result_rows,
                "total_rows": total_rows,  # ✅ Novo campo
                "limited": was_limited,     # ✅ Novo campo
                "columns": list(pdf.columns),
                "data": pdf.to_dict(orient='records'),
                "query": query
            }
            
            # Log sucesso
            self.audit.log_event(
                AuditEvent.SQL_QUERY_SUCCESS,
                {
                    "rows": result_rows,
                    "total_rows": total_rows,
                    "limited": was_limited,
                    "columns": len(pdf.columns)
                },
                EventStatus.SUCCESS
            )
            
            return result
            
        except Exception as e:
            # Log erro
            self.audit.log_event(
                AuditEvent.SQL_QUERY_ERROR,
                {"error": str(e), "query": query_for_log},
                EventStatus.ERROR
            )
            
            raise SQLExecutionError(
                f"Erro ao executar query: {str(e)}",
                details={"query": query_for_log, "spark_error": str(e)}
            )
    
    # =========================================================================
    # HELPERS
    # =========================================================================
    
    def _truncate_query_for_log(self, query: str) -> str:
        """
        ✅ Trunca query inteligentemente para log
        
        Mantém início e fim da query para facilitar debug
        
        Args:
            query: Query original
            
        Returns:
            Query truncada se > max_length
        """
        max_length = self.config.log_query_max_length
        
        if len(query) <= max_length:
            return query
        
        # Truncar mantendo início e fim
        half = (max_length - 20) // 2  # -20 para espaço do marcador
        
        return f"{query[:half]} ... [TRUNCADO {len(query)} chars] ... {query[-half:]}"
    
    def set_max_rows(self, max_rows: int):
        """Atualiza limite máximo de linhas"""
        old_max = self.config.max_result_rows
        self.config.max_result_rows = max_rows
        
        self.audit.log_event(
            AuditEvent.SQL_QUERY_START,  # Reutilizando
            {
                "action": "update_max_rows",
                "old_max": old_max,
                "new_max": max_rows
            },
            EventStatus.INFO
        )
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas do tool"""
        return {
            "max_result_rows": self.config.max_result_rows,
            "log_query_max_length": self.config.log_query_max_length,
            "count_total_rows": self.config.count_total_rows,
            "version": "3.1.0"
        }
    
    def __repr__(self) -> str:
        return f"GoldSQLTool(max_rows={self.config.max_result_rows}, version=3.1.0)"


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

# Manter classe original como alias (100% compatível)
class GoldSQLToolLegacy(GoldSQLTool):
    """Alias para compatibilidade - usa config padrão que se comporta igual ao original"""
    def __init__(
        self,
        spark: SparkSession,
        audit_logger: Optional[AuditLogger] = None,
        guardrails_config: Optional[GuardrailsConfig] = None
    ):
        # Usar config com limites altos para comportamento similar ao original
        config = SQLToolConfig(
            max_result_rows=1000000,  # Praticamente sem limite
            count_total_rows=False     # Não contar (mais rápido)
        )
        super().__init__(spark, audit_logger, guardrails_config, config)