"""
SQL Tool - Execução Segura de Queries Gold
==========================================

Author: AI Engineer Certification - Indicium
Date: January 2025
Version: 3.0.0
"""

from typing import Dict, Optional
from pyspark.sql import SparkSession

from ..utils.guardrails import SQLGuardrails, GuardrailsConfig
from ..utils.exceptions import SQLValidationError, QueryExecutionError
from ..utils.audit import AuditLogger, AuditEvent, EventStatus


class GoldSQLTool:
    """
    SQL Tool Simples - Execução Segura com Guardrails
    
    Features:
    - ✅ Execução de queries SQL
    - ✅ Validação via SQLGuardrails
    - ✅ Exceções customizadas
    - ✅ Auditoria básica
    """
    
    def __init__(
        self,
        spark: SparkSession,
        audit_logger: Optional[AuditLogger] = None,
        guardrails_config: Optional[GuardrailsConfig] = None
    ):
        self.spark = spark
        self.audit = audit_logger
        self.guardrails = SQLGuardrails(guardrails_config or GuardrailsConfig())
    
    def execute_query(self, query: str) -> Dict:
        """
        Executa query SQL com validação
        
        Args:
            query: Query SQL a executar
            
        Returns:
            Dict com success, rows, columns, data, query
            
        Raises:
            SQLValidationError: Query inválida
            QueryExecutionError: Erro na execução
        """
        # Validação Guardrails
        is_valid, validation_msg = self.guardrails.validate_query(query)
        
        if not is_valid:
            if self.audit:
                self.audit.log_event(
                    AuditEvent.SQL_VALIDATION_FAILED,
                    {"query": query[:200], "reason": validation_msg},
                    EventStatus.ERROR
                )
            raise SQLValidationError(validation_msg, details={"query": query[:200]})
        
        # Log início
        if self.audit:
            self.audit.log_event(
                AuditEvent.SQL_QUERY_START,
                {"query": query[:200]},
                EventStatus.INFO
            )
        
        # Execução
        try:
            df = self.spark.sql(query)
            pdf = df.toPandas()
            
            result = {
                "success": True,
                "rows": len(pdf),
                "columns": list(pdf.columns),
                "data": pdf.to_dict(orient='records'),
                "query": query
            }
            
            # Log sucesso
            if self.audit:
                self.audit.log_event(
                    AuditEvent.SQL_QUERY_SUCCESS,
                    {"rows": len(pdf), "columns": len(pdf.columns)},
                    EventStatus.SUCCESS
                )
            
            return result
            
        except Exception as e:
            # Log erro
            if self.audit:
                self.audit.log_event(
                    AuditEvent.SQL_QUERY_ERROR,
                    {"error": str(e), "query": query[:200]},
                    EventStatus.ERROR
                )
            
            raise QueryExecutionError(
                f"Erro ao executar query: {str(e)}",
                details={"query": query[:200], "spark_error": str(e)}
            )