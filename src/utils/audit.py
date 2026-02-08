"""
Audit Logger - Sistema de Auditoria e Rastreabilidade
======================================================

Sistema completo de auditoria para rastreabilidade de todas as ações
do agente, com persistência em Delta Lake e análise de performance.

Features:
    - Logging estruturado de eventos
    - Persistência em Delta Lake
    - Métricas de performance
    - Análise de tendências
    - Exportação para múltiplos formatos
    - Compliance e governança
    - Alertas de anomalias

Author: AI Engineer Certification - Indicium
Date: January 2025
Version: 2.0.0
"""

import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
import hashlib
from pathlib import Path

import pandas as pd


# =============================================================================
# ENUMS E TIPOS
# =============================================================================

class AuditEvent(Enum):
    """Tipos de eventos auditáveis"""
    # Sistema
    ORCHESTRATOR_INITIALIZED = "orchestrator_initialized"
    ORCHESTRATOR_START = "orchestrator_start"
    ORCHESTRATOR_STRATEGY = "orchestrator_strategy"  # Nova auditoria de estratégia
    ORCHESTRATOR_SUCCESS = "orchestrator_success"
    ORCHESTRATOR_FAILED = "orchestrator_failed"
    
    # Ferramentas
    TOOL_INITIALIZED = "tool_initialized"
    
    # SQL
    SQL_QUERY_START = "sql_query_start"
    SQL_QUERY_SUCCESS = "sql_query_success"
    SQL_QUERY_ERROR = "sql_query_error"
    SQL_VALIDATION_FAILED = "sql_validation_failed"
    SQL_CACHE_HIT = "sql_cache_hit"
    SQL_QUERY_OPTIMIZED = "sql_query_optimized"
    SQL_RESULT_TRUNCATED = "sql_result_truncated"
    
    # Web Search
    WEB_SEARCH_START = "web_search_start"
    WEB_SEARCH_SUCCESS = "web_search_success"
    WEB_SEARCH_ERROR = "web_search_error"
    SEARCH_CACHE_HIT = "search_cache_hit"
    ARTICLES_DEDUPLICATED = "articles_deduplicated"
    ARTICLE_PROCESSING_ERROR = "article_processing_error"
    
    # Charts
    CHART_GENERATION_START = "chart_generation_start"
    CHART_GENERATED = "chart_generated"
    CHART_ERROR = "chart_error"
    CHART_EXPORT_ERROR = "chart_export_error"

    # RAG Events
    RAG_RETRIEVAL_START = "rag_retrieval_start"
    RAG_RETRIEVAL_SUCCESS = "rag_retrieval_success"
    RAG_RETRIEVAL_ERROR = "rag_retrieval_error"
    RAG_CONTEXT_BUILT = "rag_context_built"
    EMBEDDING_GENERATED = "embedding_generated"
    VECTOR_SEARCH_EXECUTED = "vector_search_executed"
    DOCUMENT_LOADED = "document_loaded"

    # Report Events
    REPORT_GENERATION_START = "report_generation_start"
    REPORT_GENERATED = "report_generated"
    REPORT_VALIDATION_FAILED = "report_validation_failed"
    
    # Nodes (LangGraph)
    NODE_START = "node_start"
    NODE_COMPLETE = "node_complete"
    NODE_FAILED = "node_failed"
    NODE_ERROR = "node_error"
    QUERY_ANALYZED = "query_analyzed"
    METRICS_COLLECTED = "metrics_collected"
    NEWS_COLLECTED = "news_collected"
    CHARTS_GENERATED = "charts_generated"
    
    # Cache
    CACHE_CLEARED = "cache_cleared"
    
    # Outros
    PII_DETECTED = "pii_detected"
    GUARDRAIL_VIOLATION = "guardrail_violation"


class EventStatus(Enum):
    """Status de eventos"""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditLogEntry:
    """Entrada individual de log de auditoria"""
    session_id: str
    timestamp: datetime
    event_type: AuditEvent
    status: EventStatus
    details: Dict[str, Any]
    elapsed_seconds: float = 0.0
    user_context: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Converte para dicionário"""
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "status": self.status.value,
            "details": self.details,
            "elapsed_seconds": self.elapsed_seconds,
            "user_context": self.user_context
        }


@dataclass
class SessionSummary:
    """Resumo de uma sessão de execução"""
    session_id: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    total_events: int
    events_by_type: Dict[str, int]
    events_by_status: Dict[str, int]
    success_rate: float
    error_count: int
    warning_count: int
    performance_metrics: Dict[str, float]


# =============================================================================
# AUDIT LOGGER PRINCIPAL
# =============================================================================

class AuditLogger:
    """
    Sistema de auditoria completo para rastreabilidade
    
    Responsabilidades:
        - Registrar todos os eventos do sistema
        - Calcular métricas de performance
        - Persistir logs em Delta Lake
        - Gerar relatórios de auditoria
        - Detectar anomalias
    
    Example:
        >>> logger = AuditLogger()
        >>> logger.log_event(
        ...     AuditEvent.SQL_QUERY_START,
        ...     {"query": "SELECT * FROM table"},
        ...     status="INFO"
        ... )
        >>> summary = logger.get_summary()
        >>> logger.save_to_delta(spark)
    """
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or self._generate_session_id()
        self.start_time = datetime.now()
        self.logs: List[AuditLogEntry] = []
        self.user_context: Optional[Dict] = None
        
        # Métricas
        self._event_counts: Dict[str, int] = defaultdict(int)
        self._status_counts: Dict[str, int] = defaultdict(int)
        self._durations: Dict[str, List[float]] = defaultdict(list)
    
    def _generate_session_id(self) -> str:
        """Gera ID único de sessão"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"session_{timestamp}"
    
    # =========================================================================
    # LOGGING DE EVENTOS
    # =========================================================================
    
    def log_event(
        self,
        event_type: AuditEvent,
        details: Dict[str, Any],
        status = "INFO"  # Union[str, EventStatus]
    ) -> None:
        """
        Registra um evento de auditoria
        
        Args:
            event_type: Tipo do evento (enum)
            details: Detalhes específicos do evento
            status: Status do evento - string ("INFO", "SUCCESS") ou enum (EventStatus.INFO)
        """
        # Converter para enum se necessário
        if isinstance(status, EventStatus):
            status_enum = status  # Já é enum
        elif isinstance(status, str):
            try:
                status_enum = EventStatus[status.upper()]
            except KeyError:
                status_enum = EventStatus.INFO
        else:
            status_enum = EventStatus.INFO  # Fallback
        
        # Calcular tempo decorrido
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        # Criar entrada de log
        entry = AuditLogEntry(
            session_id=self.session_id,
            timestamp=datetime.now(),
            event_type=event_type,
            status=status_enum,
            details=details,
            elapsed_seconds=elapsed,
            user_context=self.user_context
        )
        
        # Adicionar ao log
        self.logs.append(entry)
        
        # Atualizar métricas
        self._event_counts[event_type.value] += 1
        self._status_counts[status_enum.value] += 1
        
        # Log visual (opcional)
        self._print_log(entry)
    
    def _print_log(self, entry: AuditLogEntry) -> None:
        """Imprime log de forma visual"""
        emoji_map = {
            EventStatus.INFO: "ℹ️",
            EventStatus.SUCCESS: "✅",
            EventStatus.WARNING: "⚠️",
            EventStatus.ERROR: "❌",
            EventStatus.CRITICAL: "🚨"
        }
        
        emoji = emoji_map.get(entry.status, "•")
        message = entry.details.get("message", entry.details.get("query", "")[:50])
        
        print(f"{emoji} [{entry.event_type.value}] {message}")
    
    def log_performance_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = "seconds"
    ) -> None:
        """Registra métrica de performance"""
        self._durations[metric_name].append(value)
        
        self.log_event(
            AuditEvent.ORCHESTRATOR_START,  # Evento genérico
            {
                "metric_name": metric_name,
                "value": value,
                "unit": unit
            },
            status="INFO"
        )
    
    def set_user_context(self, context: Dict) -> None:
        """Define contexto do usuário para todos os logs"""
        self.user_context = context
    
    # =========================================================================
    # ANÁLISE E RELATÓRIOS
    # =========================================================================
    
    def get_summary(self) -> Dict:
        """
        Retorna resumo completo da sessão de auditoria
        
        Returns:
            Dicionário com estatísticas e métricas
        """
        total_duration = (datetime.now() - self.start_time).total_seconds()
        
        # Calcular taxa de sucesso
        total_events = len(self.logs)
        success_events = sum(
            1 for log in self.logs 
            if log.status in [EventStatus.SUCCESS, EventStatus.INFO]
        )
        success_rate = (success_events / total_events * 100) if total_events > 0 else 0
        
        # Contar erros e warnings
        error_count = sum(
            1 for log in self.logs 
            if log.status in [EventStatus.ERROR, EventStatus.CRITICAL]
        )
        warning_count = sum(
            1 for log in self.logs 
            if log.status == EventStatus.WARNING
        )
        
        # Métricas de performance
        performance_metrics = {}
        for metric, values in self._durations.items():
            if values:
                performance_metrics[f"{metric}_avg"] = sum(values) / len(values)
                performance_metrics[f"{metric}_max"] = max(values)
                performance_metrics[f"{metric}_min"] = min(values)
        
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": total_duration,
            "total_events": total_events,
            "events_by_type": dict(self._event_counts),
            "events_by_status": dict(self._status_counts),
            "success_rate": round(success_rate, 2),
            "error_count": error_count,
            "warning_count": warning_count,
            "performance_metrics": performance_metrics,
            "full_log": [log.to_dict() for log in self.logs]
        }
    
    def get_errors(self) -> List[AuditLogEntry]:
        """Retorna apenas eventos de erro"""
        return [
            log for log in self.logs
            if log.status in [EventStatus.ERROR, EventStatus.CRITICAL]
        ]
    
    def get_warnings(self) -> List[AuditLogEntry]:
        """Retorna apenas eventos de warning"""
        return [
            log for log in self.logs
            if log.status == EventStatus.WARNING
        ]
    
    def get_events_by_type(self, event_type: AuditEvent) -> List[AuditLogEntry]:
        """Retorna eventos de um tipo específico"""
        return [log for log in self.logs if log.event_type == event_type]
    
    def get_timeline(self) -> List[Dict]:
        """Retorna timeline de eventos ordenado"""
        return sorted(
            [log.to_dict() for log in self.logs],
            key=lambda x: x["timestamp"]
        )
    
    def analyze_performance(self) -> Dict:
        """Analisa performance detalhada da execução"""
        # Identificar gargalos
        bottlenecks = []
        
        # Analisar duração de cada nó
        node_events = [
            log for log in self.logs
            if log.event_type in [AuditEvent.NODE_COMPLETE]
        ]
        
        for event in node_events:
            node_name = event.details.get("node", "unknown")
            duration = event.details.get("duration_seconds", 0)
            
            if duration > 10:  # Threshold de 10 segundos
                bottlenecks.append({
                    "node": node_name,
                    "duration_seconds": duration,
                    "severity": "high" if duration > 30 else "medium"
                })
        
        # Calcular distribuição de tempo
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        time_distribution = {}
        for metric, values in self._durations.items():
            total_metric_time = sum(values)
            time_distribution[metric] = {
                "total_seconds": total_metric_time,
                "percentage": (total_metric_time / total_time * 100) if total_time > 0 else 0,
                "count": len(values)
            }
        
        return {
            "total_duration_seconds": total_time,
            "bottlenecks": bottlenecks,
            "time_distribution": time_distribution,
            "events_per_second": len(self.logs) / total_time if total_time > 0 else 0
        }
    
    def detect_anomalies(self) -> List[Dict]:
        """Detecta anomalias nos logs"""
        anomalies = []
        
        # 1. Taxa de erro muito alta
        if self._status_counts.get("error", 0) > len(self.logs) * 0.3:
            anomalies.append({
                "type": "high_error_rate",
                "severity": "critical",
                "details": f"Taxa de erro: {self._status_counts['error'] / len(self.logs) * 100:.1f}%"
            })
        
        # 2. Duração anormal
        for metric, values in self._durations.items():
            if len(values) > 1:
                avg = sum(values) / len(values)
                max_val = max(values)
                
                # Se algum valor é 3x maior que a média
                if max_val > avg * 3:
                    anomalies.append({
                        "type": "performance_spike",
                        "severity": "warning",
                        "metric": metric,
                        "details": f"Spike de {max_val:.2f}s (média: {avg:.2f}s)"
                    })
        
        # 3. Muitos warnings
        if self._status_counts.get("warning", 0) > 10:
            anomalies.append({
                "type": "excessive_warnings",
                "severity": "medium",
                "details": f"{self._status_counts['warning']} warnings detectados"
            })
        
        return anomalies
    
    # =========================================================================
    # PERSISTÊNCIA
    # =========================================================================
    
    def save_to_delta(
        self,
        spark,
        catalog: str = "dbx_lab_draagron",
        schema: str = "audit"
    ) -> None:
        """
        Salva logs em tabela Delta Lake para governança
        
        Args:
            spark: SparkSession
            catalog: Nome do catálogo
            schema: Nome do schema
        """
        try:
            # Converter logs para DataFrame
            logs_dict = [log.to_dict() for log in self.logs]
            
            if not logs_dict:
                print("⚠️ Nenhum log para salvar")
                return
            
            df = pd.DataFrame(logs_dict)
            
            # Converter para Spark DataFrame
            spark_df = spark.createDataFrame(df)
            
            # Garantir que schema existe
            try:
                spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
                print(f"✅ Schema {catalog}.{schema} verificado/criado")
            except Exception as schema_error:
                print(f"⚠️ Erro ao criar schema: {schema_error}")
            
            # Nome da tabela
            table_name = f"{catalog}.{schema}.agent_audit_logs"
            
            # Salvar (append mode para preservar histórico)
            spark_df.write.format("delta").mode("append").saveAsTable(table_name)
            
            print(f"✅ {len(logs_dict)} logs salvos em {table_name}")
            
        except Exception as e:
            print(f"❌ Erro ao salvar logs em Delta: {e}")
            import traceback
            print(f"   Detalhes: {traceback.format_exc()[:300]}...")  # Truncar stack trace
    
    def export_to_json(self, filepath: str) -> None:
        """Exporta logs para arquivo JSON"""
        try:
            summary = self.get_summary()
            
            output_path = Path(filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"✅ Logs exportados para {filepath}")
            
        except Exception as e:
            print(f"❌ Erro ao exportar JSON: {e}")
    
    def export_to_csv(self, filepath: str) -> None:
        """Exporta logs para arquivo CSV"""
        try:
            logs_dict = [log.to_dict() for log in self.logs]
            df = pd.DataFrame(logs_dict)
            
            output_path = Path(filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            df.to_csv(output_path, index=False, encoding='utf-8')
            
            print(f"✅ Logs exportados para {filepath}")
            
        except Exception as e:
            print(f"❌ Erro ao exportar CSV: {e}")
    
    # =========================================================================
    # CONSULTAS E FILTROS
    # =========================================================================
    
    def query_logs(
        self,
        event_type: Optional[AuditEvent] = None,
        status: Optional[EventStatus] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        search_term: Optional[str] = None
    ) -> List[AuditLogEntry]:
        """
        Consulta logs com filtros
        
        Args:
            event_type: Filtrar por tipo de evento
            status: Filtrar por status
            start_time: Timestamp inicial
            end_time: Timestamp final
            search_term: Termo de busca nos detalhes
            
        Returns:
            Lista de logs filtrados
        """
        filtered = self.logs
        
        # Filtrar por tipo
        if event_type:
            filtered = [log for log in filtered if log.event_type == event_type]
        
        # Filtrar por status
        if status:
            filtered = [log for log in filtered if log.status == status]
        
        # Filtrar por tempo
        if start_time:
            filtered = [log for log in filtered if log.timestamp >= start_time]
        
        if end_time:
            filtered = [log for log in filtered if log.timestamp <= end_time]
        
        # Busca textual
        if search_term:
            search_lower = search_term.lower()
            filtered = [
                log for log in filtered
                if search_lower in json.dumps(log.details).lower()
            ]
        
        return filtered
    
    def get_session_summary_by_id(self, session_id: str) -> Optional[Dict]:
        """Retorna resumo de uma sessão específica"""
        session_logs = [log for log in self.logs if log.session_id == session_id]
        
        if not session_logs:
            return None
        
        # Calcular métricas da sessão
        start = min(log.timestamp for log in session_logs)
        end = max(log.timestamp for log in session_logs)
        
        return {
            "session_id": session_id,
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "duration_seconds": (end - start).total_seconds(),
            "total_events": len(session_logs),
            "events": [log.to_dict() for log in session_logs]
        }
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def clear_logs(self) -> None:
        """Limpa todos os logs da sessão atual"""
        self.logs.clear()
        self._event_counts.clear()
        self._status_counts.clear()
        self._durations.clear()
        print("✅ Logs limpos")
    
    def reset_session(self) -> None:
        """Inicia nova sessão de auditoria"""
        self.session_id = self._generate_session_id()
        self.start_time = datetime.now()
        self.clear_logs()
        print(f"✅ Nova sessão iniciada: {self.session_id}")
    
    def __repr__(self) -> str:
        """Representação string do logger"""
        return (
            f"AuditLogger(session_id={self.session_id}, "
            f"events={len(self.logs)}, "
            f"errors={sum(1 for log in self.logs if log.status in [EventStatus.ERROR, EventStatus.CRITICAL])})"
        )
    
    def __len__(self) -> int:
        """Retorna número de logs"""
        return len(self.logs)


# =============================================================================
# ANALISADOR DE AUDITORIA (para consultas em Delta)
# =============================================================================

class AuditAnalyzer:
    """Analisador de logs históricos de auditoria"""
    
    def __init__(self, spark):
        self.spark = spark
        self.table_name = "dbx_lab_draagron.audit.agent_audit_logs"
    
    def get_sessions_summary(self, days: int = 7) -> pd.DataFrame:
        """Retorna resumo de sessões dos últimos N dias"""
        query = f"""
        SELECT 
            session_id,
            MIN(timestamp) as start_time,
            MAX(timestamp) as end_time,
            COUNT(*) as total_events,
            SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error_count,
            AVG(elapsed_seconds) as avg_elapsed_seconds
        FROM {self.table_name}
        WHERE timestamp >= current_date() - INTERVAL {days} DAYS
        GROUP BY session_id
        ORDER BY start_time DESC
        """
        
        return self.spark.sql(query).toPandas()
    
    def get_error_trends(self, days: int = 30) -> pd.DataFrame:
        """Analisa tendências de erros"""
        query = f"""
        SELECT 
            DATE(timestamp) as date,
            event_type,
            COUNT(*) as error_count
        FROM {self.table_name}
        WHERE status IN ('error', 'critical')
            AND timestamp >= current_date() - INTERVAL {days} DAYS
        GROUP BY DATE(timestamp), event_type
        ORDER BY date DESC, error_count DESC
        """
        
        return self.spark.sql(query).toPandas()
    
    def get_performance_metrics(self, session_id: str) -> pd.DataFrame:
        """Retorna métricas de performance de uma sessão"""
        query = f"""
        SELECT 
            event_type,
            AVG(elapsed_seconds) as avg_seconds,
            MAX(elapsed_seconds) as max_seconds,
            MIN(elapsed_seconds) as min_seconds,
            COUNT(*) as event_count
        FROM {self.table_name}
        WHERE session_id = '{session_id}'
        GROUP BY event_type
        ORDER BY avg_seconds DESC
        """
        
        return self.spark.sql(query).toPandas()
