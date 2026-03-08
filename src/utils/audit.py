"""
Audit Logger — Rastreabilidade e Observabilidade do Pipeline SRAG
=================================================================

Responsabilidade: registrar eventos do agente durante a execução, calcular
métricas de performance e persistir os logs em Delta Lake para análise
histórica e compliance.

Decisões de design
------------------
elapsed_seconds mantido como tempo de sessão; duration_seconds para performance
    O design original armazenava elapsed = (datetime.now() - start_time) em
    _durations[event_type] e depois calculava avg_seconds como média desses
    valores. O resultado era semanticamente incorreto: a "média" de timestamps
    desde o início da sessão não tem significado como indicador de latência.
    Um evento que ocorre aos 3s e outro aos 28s produzem avg_seconds=15.5s —
    isso não representa a duração de nenhum dos dois eventos.

    A correção separa dois conceitos distintos:
    - elapsed_seconds em AuditLogEntry: mantido como tempo desde o início
      da sessão. É semanticamente correto para reconstrução da linha do
      tempo de execução em Delta e para correlação de eventos em dashboards.
    - duration_seconds em log_event(): parâmetro opcional que os tools
      passam quando têm medição real de latência (time.perf_counter antes/após
      a operação). Apenas esses valores alimentam _durations e aparecem em
      get_performance_summary(). Eventos sem duration_seconds são excluídos
      das métricas de latência em vez de contaminar a média com timestamps.

success_rate calculado apenas sobre eventos não-INFO
    O design original dividia SUCCESS / total, onde total incluía todos os
    eventos INFO. Como a maioria dos eventos do pipeline são informativos
    (node_start, query_analyzed, tool_initialized), a taxa ficava entre
    32–43% mesmo em execuções perfeitas. Isso tornava o campo inútil para
    qualquer SLA ou alerta: impossível distinguir execução degradada de
    execução normal pela taxa. A correção exclui INFO do denominador —
    success_rate agora é SUCCESS / (SUCCESS + WARNING + ERROR + CRITICAL),
    refletindo a proporção real de operações bem-sucedidas entre as que
    tinham resultado esperado.

AuditEvent enum completo para os tools refatorados
    O enum original não continha eventos usados pelo chart_tool.py e
    web_search_tool.py refatorados: TOOL_DEGRADED, CHART_WRITE_ERROR,
    CHART_STAT_ERROR, CHART_CLEANUP, WEB_SEARCH_OFFLINE, CACHE_EVICTED.
    Quando esses tools importam AuditLogger real (não o stub local),
    AuditEvent.CHART_WRITE_ERROR lança AttributeError em tempo de execução.
    Todos os eventos ausentes foram adicionados com nomes semanticamente
    distintos por contexto — a investigação identificou que reutilizar o
    mesmo evento em contextos diferentes era um padrão a evitar.

save_to_delta levanta AuditLogSaveError em vez de silenciar com print
    O design original capturava bare Exception e imprimia no stdout, retornando
    None em caso de falha. O orquestrador não tinha como saber se os logs foram
    persistidos — falhas de auditoria ficavam invisíveis, que é paradoxalmente
    pior do que falhas de negócio. save_to_delta agora levanta AuditLogSaveError
    (definida em exceptions.py) para que o chamador possa decidir se aborta,
    alerta ou continua com logs apenas em memória. O fallback de print é mantido
    antes do raise para preservar rastreabilidade em ambientes sem logger externo.

ALTER TABLE ADD COLUMN — sintaxe compatível com Databricks/Delta
    O Spark SQL / Delta Lake não suporta a cláusula IF NOT EXISTS em
    ALTER TABLE ADD COLUMN (SQLSTATE: 42601). A cláusula era redundante
    porque missing já filtra apenas colunas ausentes na tabela antes de
    chegar ao ALTER TABLE. A correção remove IF NOT EXISTS, tornando a
    sintaxe válida para o parser do Databricks.

AuditAnalyzer usa validação de session_id antes de interpolar em SQL
    get_performance_metrics() interpolava session_id diretamente em f-string
    sem sanitização. Embora session_id seja gerado internamente pelo sistema,
    o padrão cria risco quando a assinatura aceita strings externas. A validação
    agora exige que session_id contenha apenas caracteres alfanuméricos e
    underscores, levantando ValueError antes de qualquer execução SQL.
    get_error_trends() usa INTERVAL com variável int — tecnicamente seguro,
    mas padronizado com cast explícito para deixar a intenção clara.

SessionSummary como schema de referência, get_summary retorna dict compatível
    O dataclass SessionSummary existia mas nunca era instanciado pela API
    pública. get_summary() retornava dict bruto, tornando SessionSummary
    código morto de documentação. A correção mantém o dict como retorno para
    não quebrar consumidores existentes, e SessionSummary passa a documentar
    explicitamente o contrato de campos do dict. Consumidores que precisam
    de type-checking podem instanciar SessionSummary a partir do dict retornado.
"""

import json
import time
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

try:
    from src.utils.exceptions import AuditLogSaveError
except ImportError:
    class AuditLogSaveError(Exception):
        pass


# =============================================================================
# ENUMS E TIPOS
# =============================================================================

class AuditEvent(Enum):
    """
    Eventos auditáveis do pipeline SRAG.

    Cada grupo de eventos corresponde a um componente do sistema. Eventos
    semanticamente distintos têm nomes distintos — o mesmo nome não é
    reutilizado em contextos diferentes para não obscurecer a origem de
    falhas em dashboards e alertas.
    """
    # Orquestrador
    ORCHESTRATOR_INITIALIZED = "orchestrator_initialized"
    ORCHESTRATOR_START        = "orchestrator_start"
    ORCHESTRATOR_STRATEGY     = "orchestrator_strategy"
    ORCHESTRATOR_SUCCESS      = "orchestrator_success"
    ORCHESTRATOR_FAILED       = "orchestrator_failed"

    # Ferramentas — inicialização e modo degradado
    TOOL_INITIALIZED = "tool_initialized"
    TOOL_DEGRADED    = "tool_degraded"

    # SQL
    SQL_QUERY_START       = "sql_query_start"
    SQL_QUERY_SUCCESS     = "sql_query_success"
    SQL_QUERY_ERROR       = "sql_query_error"
    SQL_VALIDATION_FAILED = "sql_validation_failed"
    SQL_CACHE_HIT         = "sql_cache_hit"
    SQL_QUERY_OPTIMIZED   = "sql_query_optimized"
    SQL_RESULT_TRUNCATED  = "sql_result_truncated"

    # Web Search
    WEB_SEARCH_START         = "web_search_start"
    WEB_SEARCH_SUCCESS       = "web_search_success"
    WEB_SEARCH_ERROR         = "web_search_error"
    WEB_SEARCH_OFFLINE       = "web_search_offline"
    SEARCH_CACHE_HIT         = "search_cache_hit"
    ARTICLES_DEDUPLICATED    = "articles_deduplicated"
    ARTICLE_PROCESSING_ERROR = "article_processing_error"

    # Charts — geração, escrita e manutenção
    CHART_GENERATION_START = "chart_generation_start"
    CHART_GENERATED        = "chart_generated"
    CHART_ERROR            = "chart_error"
    CHART_EXPORT_ERROR     = "chart_export_error"
    CHART_WRITE_ERROR      = "chart_write_error"
    CHART_STAT_ERROR       = "chart_stat_error"
    CHART_CLEANUP          = "chart_cleanup"

    # RAG
    RAG_RETRIEVAL_START    = "rag_retrieval_start"
    RAG_RETRIEVAL_SUCCESS  = "rag_retrieval_success"
    RAG_RETRIEVAL_ERROR    = "rag_retrieval_error"
    RAG_CONTEXT_BUILT      = "rag_context_built"
    EMBEDDING_GENERATED    = "embedding_generated"
    VECTOR_SEARCH_EXECUTED = "vector_search_executed"
    DOCUMENT_LOADED        = "document_loaded"

    # Relatórios
    REPORT_GENERATION_START  = "report_generation_start"
    REPORT_GENERATED         = "report_generated"
    REPORT_VALIDATION_FAILED = "report_validation_failed"

    # Nós LangGraph
    NODE_START         = "node_start"
    NODE_COMPLETE      = "node_complete"
    NODE_FAILED        = "node_failed"
    NODE_ERROR         = "node_error"
    QUERY_ANALYZED     = "query_analyzed"
    METRICS_COLLECTED  = "metrics_collected"
    METRICS_ERROR      = "metrics_error"
    SYNTHESIS_ERROR    = "synthesis_error"
    NEWS_COLLECTED     = "news_collected"
    CHARTS_GENERATED   = "charts_generated"

    # Cache
    CACHE_CLEARED  = "cache_cleared"
    CACHE_EVICTED  = "cache_evicted"

    # Segurança e PII
    PII_DETECTED        = "pii_detected"
    GUARDRAIL_VIOLATION = "guardrail_violation"


class EventStatus(Enum):
    """
    Status de eventos de auditoria.

    INFO não representa nem sucesso nem falha — é um evento informativo puro.
    A distinção é relevante para o cálculo de success_rate, que exclui INFO
    do denominador para não produzir taxas artificialmente baixas.
    """
    INFO     = "info"
    SUCCESS  = "success"
    WARNING  = "warning"
    ERROR    = "error"
    CRITICAL = "critical"


@dataclass
class AuditLogEntry:
    """
    Entrada individual de log de auditoria.

    elapsed_seconds registra o tempo desde o início da sessão no momento
    em que o evento foi registrado. É útil para reconstrução da linha do
    tempo de execução, mas não representa a duração da operação auditada.
    Para latência real de operações, ver duration_seconds em log_event().
    """
    session_id:       str
    timestamp:        datetime
    event_type:       AuditEvent
    status:           EventStatus
    details:          Dict[str, Any]
    elapsed_seconds:  float = 0.0
    duration_seconds: Optional[float] = None
    user_context:     Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            "session_id":       self.session_id,
            "timestamp":        self.timestamp.isoformat(),
            "event_type":       self.event_type.value,
            "status":           self.status.value,
            # details é serializado como JSON string em vez de dict/struct.
            # O Delta Lake infere o schema de details na criação da tabela a
            # partir do primeiro lote de logs. Como cada AuditEvent usa chaves
            # diferentes em details (ex: {"total_rows": 10} vs {"output_dirs": [...]}),
            # o Delta tentaria mapear campos de structs incompatíveis entre sessões,
            # resultando em DELTA_COLUMN_STRUCT_TYPE_MISMATCH no INSERT.
            # Serializar como string elimina a inferência de struct — details é
            # sempre STRING no Delta, compatível com qualquer payload futuro.
            "details":          json.dumps(self.details, default=str),
            "elapsed_seconds":  self.elapsed_seconds,
            "duration_seconds": self.duration_seconds,
            # user_context também é heterogêneo — mesma proteção.
            "user_context":     json.dumps(self.user_context, default=str) if self.user_context else None,
        }


@dataclass
class SessionSummary:
    """
    Schema de referência para o dict retornado por get_summary().

    get_summary() retorna um dict com campos compatíveis com este dataclass
    para não quebrar consumidores existentes. Consumidores que precisam de
    type-checking podem instanciar SessionSummary(**logger.get_summary())
    após remover a chave "logs" do dict.

    success_rate
        Calculado como SUCCESS / (SUCCESS + WARNING + ERROR + CRITICAL).
        Eventos INFO são excluídos do denominador porque não representam
        operações com resultado esperado — incluí-los produzia taxas de
        32-43% em execuções normais, tornando o campo inútil para alertas.
    """
    session_id:          str
    start_time:          str
    end_time:            str
    duration_seconds:    float
    total_events:        int
    events_by_type:      Dict[str, int]
    events_by_status:    Dict[str, int]
    success_rate:        float
    error_count:         int
    warning_count:       int
    performance_metrics: Dict[str, Dict]


# =============================================================================
# AUDIT LOGGER
# =============================================================================

class AuditLogger:
    """
    Registrador de eventos para rastreabilidade do pipeline SRAG.

    Mantém logs em memória durante a sessão, calcula métricas de performance
    baseadas em durações reais (não timestamps de sessão) e persiste em
    Delta Lake ao final da execução.

    Parâmetros
    ----------
    session_id
        Identificador da sessão. Quando None, gera automaticamente no formato
        session_YYYYMMDD_HHMMSS_microseconds.
    """

    def __init__(self, session_id: Optional[str] = None):
        self.session_id  = session_id or self._generate_session_id()
        self.start_time  = datetime.now()
        self.logs:       List[AuditLogEntry] = []
        self.user_context: Optional[Dict]   = None

        self._event_counts:  Dict[str, int]         = defaultdict(int)
        self._status_counts: Dict[str, int]         = defaultdict(int)
        self._durations:     Dict[str, List[float]] = defaultdict(list)

    def _generate_session_id(self) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"srag_{ts}"

    # =========================================================================
    # LOGGING
    # =========================================================================

    def log_event(
        self,
        event_type:       AuditEvent,
        details:          Dict[str, Any],
        status:           Union[str, EventStatus] = "INFO",
        duration_seconds: Optional[float] = None,
    ) -> None:
        """
        Registra um evento de auditoria.

        Parâmetros
        ----------
        event_type
            Membro do enum AuditEvent. Passar string causaria AttributeError
            em to_dict() — o tipo é verificado pelo type system do Python mas
            não em runtime para não adicionar overhead em cada chamada.
        details
            Dicionário com contexto específico do evento. Deve ser serializável
            em JSON (sem objetos Spark, DataFrames ou tipos customizados).
        status
            Aceita string case-insensitive ("INFO", "SUCCESS", etc.) ou enum
            EventStatus. Strings não reconhecidas fazem fallback para INFO.
        duration_seconds
            Duração real da operação em segundos, medida pelo chamador com
            time.perf_counter() antes e após a operação. Quando fornecida,
            alimenta _durations e aparece em get_performance_summary(). Quando
            None, o evento não contribui para métricas de latência — isso é
            preferível a contribuir com timestamps de sessão sem significado.
        """
        if isinstance(status, EventStatus):
            status_enum = status
        elif isinstance(status, str):
            try:
                status_enum = EventStatus[status.upper()]
            except KeyError:
                status_enum = EventStatus.INFO
        else:
            status_enum = EventStatus.INFO

        if "execution_time_seconds" in details and details["execution_time_seconds"] is None:
            details["execution_time_seconds"] = 0.0

        elapsed = (datetime.now() - self.start_time).total_seconds()

        entry = AuditLogEntry(
            session_id=self.session_id,
            timestamp=datetime.now(),
            event_type=event_type,
            status=status_enum,
            details=details,
            elapsed_seconds=elapsed,
            duration_seconds=duration_seconds,
            user_context=self.user_context,
        )

        self.logs.append(entry)
        self._event_counts[event_type.value]  += 1
        self._status_counts[status_enum.value] += 1

        if duration_seconds is not None:
            self._durations[event_type.value].append(duration_seconds)

    def set_user_context(self, context: Dict) -> None:
        """Define contexto do usuário anexado a todos os eventos futuros."""
        self.user_context = context

    # =========================================================================
    # METRICAS E RESUMOS
    # =========================================================================

    def get_summary(self) -> Dict:
        """
        Retorna resumo completo da sessão como dict.

        O campo success_rate exclui eventos INFO do denominador. Ver docstring
        de SessionSummary para a justificativa completa.

        performance_metrics inclui apenas tipos de evento para os quais ao
        menos um log_event() foi chamado com duration_seconds explícito.
        Tipos sem duração medida são omitidos em vez de reportar médias de
        timestamps de sessão sem significado.

        O campo execution_time_seconds é mantido como alias de duration_seconds
        por compatibilidade com consumidores que dependiam do nome antigo.
        """
        if not self.logs:
            return {
                "session_id":             self.session_id,
                "total_events":           0,
                "events_by_type":         {},
                "events_by_status":       {},
                "success_rate":           0.0,
                "error_count":            0,
                "warning_count":          0,
                "duration_seconds":       0.0,
                "execution_time_seconds": 0.0,
                "performance_metrics":    {},
                "logs":                   [],
            }

        total    = len(self.logs)
        errors   = sum(1 for l in self.logs if l.status in (EventStatus.ERROR, EventStatus.CRITICAL))
        warnings = sum(1 for l in self.logs if l.status == EventStatus.WARNING)
        success  = sum(1 for l in self.logs if l.status == EventStatus.SUCCESS)

        meaningful   = sum(1 for l in self.logs if l.status != EventStatus.INFO)
        success_rate = round(success / meaningful * 100, 2) if meaningful > 0 else 0.0

        end_time = max(l.timestamp for l in self.logs)
        duration = (end_time - self.start_time).total_seconds()

        return {
            "session_id":             self.session_id,
            "start_time":             self.start_time.isoformat(),
            "end_time":               end_time.isoformat(),
            "duration_seconds":       round(duration, 4),
            "execution_time_seconds": round(duration, 4),
            "total_events":           total,
            "events_by_type":         dict(self._event_counts),
            "events_by_status":       dict(self._status_counts),
            "success_rate":           success_rate,
            "error_count":            errors,
            "warning_count":          warnings,
            "performance_metrics":    self._build_performance_metrics(),
            "logs":                   [l.to_dict() for l in self.logs],
        }

    def get_performance_summary(self) -> Dict[str, Dict]:
        """
        Retorna métricas de latência por tipo de evento.

        Inclui apenas eventos para os quais duration_seconds foi fornecido
        explicitamente em log_event(). Eventos sem duração medida são omitidos
        para não contaminar as médias com timestamps de sessão.

        Cada entrada contém: count, avg_seconds, max_seconds, min_seconds,
        total_seconds — todos representando durações reais de operação.
        """
        return self._build_performance_metrics()

    def _build_performance_metrics(self) -> Dict[str, Dict]:
        """
        Constrói o dict de métricas de performance a partir de _durations.

        _durations só contém valores quando duration_seconds foi passado
        explicitamente para log_event(). Eventos sem duração medida não
        aparecem aqui, o que é preferível a reportar métricas incorretas.
        """
        metrics: Dict[str, Dict] = {}
        for event_type, durations in self._durations.items():
            if not durations:
                continue
            metrics[event_type] = {
                "count":         len(durations),
                "avg_seconds":   round(sum(durations) / len(durations), 4),
                "max_seconds":   round(max(durations), 4),
                "min_seconds":   round(min(durations), 4),
                "total_seconds": round(sum(durations), 4),
            }
        return metrics

    def get_errors(self) -> List[AuditLogEntry]:
        """Retorna entradas com status ERROR ou CRITICAL."""
        return [l for l in self.logs if l.status in (EventStatus.ERROR, EventStatus.CRITICAL)]

    def get_warnings(self) -> List[AuditLogEntry]:
        """Retorna entradas com status WARNING."""
        return [l for l in self.logs if l.status == EventStatus.WARNING]

    def print_summary(self) -> None:
        """Imprime resumo formatado no stdout. Útil para diagnóstico em notebooks."""
        summary = self.get_summary()

        print("\n" + "=" * 70)
        print("AUDIT SUMMARY")
        print("=" * 70)
        print(f"Session ID:    {summary.get('session_id', 'N/A')}")
        print(f"Duration:      {summary.get('duration_seconds', 0):.2f}s")
        print(f"Total Events:  {summary.get('total_events', 0)}")
        print(f"Success Rate:  {summary.get('success_rate', 0):.1f}%  (exclui INFO do denominador)")
        print(f"Errors:        {summary.get('error_count', 0)}")
        print(f"Warnings:      {summary.get('warning_count', 0)}")

        print("\nEvents by Type:")
        for evt, count in sorted(
            summary.get("events_by_type", {}).items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            print(f"  {evt}: {count}")

        perf = summary.get("performance_metrics", {})
        if perf:
            print("\nPerformance — Top 5 por latencia media (apenas eventos com duracao medida):")
            for evt, m in sorted(perf.items(), key=lambda x: x[1]["avg_seconds"], reverse=True)[:5]:
                print(f"  {evt}: {m['avg_seconds']:.3f}s avg  max={m['max_seconds']:.3f}s  n={m['count']}")
        else:
            print("\nPerformance: nenhum evento com duration_seconds registrado nesta sessao.")

        print("=" * 70 + "\n")

    # =========================================================================
    # PERSISTENCIA
    # =========================================================================

    def save_to_delta(
        self,
        spark,
        catalog: str = "dbx_srag_lab",
        schema:  str = "audit",
    ) -> None:
        """
        Persiste os logs da sessão em tabela Delta Lake.

        Estratégia de escrita (Spark Connect compatível)
        ------------------------------------------------
        saveAsTable() falha em Databricks Spark Connect (DBR 14+) com Unity
        Catalog quando o schema não existe ou quando a tabela é criada pela
        primeira vez. O erro ocorre no SparkConnectPlanner.handleWriteOperation
        independentemente do mode ("overwrite" ou "append").

        A estratégia substituída usa SQL puro via createOrReplaceTempView +
        CREATE TABLE AS SELECT + INSERT INTO BY NAME:

        1. spark.sql("CREATE SCHEMA IF NOT EXISTS ...") — garante que o schema
           existe antes de qualquer operação de tabela.
        2. createOrReplaceTempView() — materializa o DataFrame como view
           temporária acessível ao Spark SQL sem passar pelo DataFrameWriter.
        3. CREATE TABLE IF NOT EXISTS ... AS SELECT — cria a tabela na primeira
           execução usando CTAS, que é nativo ao Spark SQL e não passa pelo
           DataFrameWriter.
        4. INSERT INTO ... BY NAME — append subsequente. "BY NAME" tolera
           diferenças de ordem de colunas entre a view temporária e a tabela
           persistida — compatível com evolução de schema sem mergeSchema.

        Novas colunas adicionadas ao schema de log (ex: duration_seconds)
        são detectadas comparando o schema da view com o da tabela existente.
        Para cada coluna ausente, emite ALTER TABLE ADD COLUMN (sem IF NOT
        EXISTS — cláusula não suportada pelo parser Spark/Delta, SQLSTATE 42601).
        A guarda de redundância já é feita pelo filtro `missing`, que só inclui
        colunas confirmadamente ausentes na tabela antes do ALTER.

        Levanta AuditLogSaveError quando a persistência falha. O orquestrador
        deve capturar essa exceção e decidir se aborta, alerta ou continua com
        logs apenas em memória — a decisão não cabe ao logger.

        Parâmetros
        ----------
        spark   : SparkSession ativa.
        catalog : Catálogo Unity Catalog onde a tabela de auditoria reside.
        schema  : Schema dentro do catálogo. Criado automaticamente se ausente.
        """
        if not self.logs:
            return

        table_name = f"{catalog}.{schema}.agent_audit_logs"
        view_name  = "__srag_audit_logs_tmp"

        try:
            logs_dicts = [l.to_dict() for l in self.logs]
            spark_df   = spark.createDataFrame(pd.DataFrame(logs_dicts))

            # 1. Garante schema (tentativa robusta com backticks)
            try:
                spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`")
            except Exception as schema_exc:
                print(
                    f"[audit] aviso: não foi possível garantir schema "
                    f"`{catalog}`.`{schema}`: {schema_exc}"
                )

            # 2. Materializa como view temporária (não usa DataFrameWriter)
            spark_df.createOrReplaceTempView(view_name)

            # 3. Verifica existência da tabela
            table_exists = spark.catalog.tableExists(table_name)

            if not table_exists:
                # CTAS — cria tabela com o schema completo dos logs atuais
                spark.sql(f"""
                    CREATE TABLE IF NOT EXISTS {table_name}
                    USING DELTA
                    AS SELECT * FROM {view_name}
                    WHERE 1 = 0
                """)
                # INSERT após criação vazia para respeitar o schema criado
                spark.sql(f"INSERT INTO {table_name} BY NAME SELECT * FROM {view_name}")
                print(f"[audit] tabela criada e logs inseridos em {table_name}")
            else:
                # Detecta colunas novas e adiciona via ALTER TABLE antes do INSERT.
                # IF NOT EXISTS é omitido intencionalmente: não é suportado pelo
                # parser do Spark/Delta (SQLSTATE 42601). A guarda de duplicidade
                # já está em `missing`, que só inclui colunas ausentes na tabela.
                existing_cols = {f.name for f in spark.table(table_name).schema.fields}
                new_cols      = {f.name: f for f in spark_df.schema.fields}
                missing       = {n: f for n, f in new_cols.items() if n not in existing_cols}

                for col_name, col_field in missing.items():
                    spark_type = col_field.dataType.simpleString()
                    try:
                        spark.sql(
                            f"ALTER TABLE {table_name} "
                            f"ADD COLUMN `{col_name}` {spark_type}"
                        )
                        print(f"[audit] coluna adicionada: {col_name} ({spark_type})")
                    except Exception as alter_exc:
                        print(f"[audit] aviso: não foi possível adicionar coluna {col_name}: {alter_exc}")

                spark.sql(f"INSERT INTO {table_name} BY NAME SELECT * FROM {view_name}")

            print(f"[audit] {len(self.logs)} eventos persistidos em {table_name}")

        except Exception as exc:
            import traceback
            detail = traceback.format_exc()
            print(f"[audit] falha ao persistir logs em {table_name}: {exc}\n{detail[:400]}")
            raise AuditLogSaveError(
                f"Falha ao salvar {len(self.logs)} logs em {table_name}: {exc}"
            ) from exc
        finally:
            # Remove a view temporária independentemente de sucesso ou falha
            try:
                spark.catalog.dropTempView(view_name)
            except Exception:
                pass

    def export_to_json(self, filepath: str) -> None:
        """
        Exporta o resumo da sessão para arquivo JSON.

        Levanta a exceção original em caso de falha de I/O — o chamador é
        responsável por decidir como tratar (ex: fallback para stdout).
        """
        summary     = self.get_summary()
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    def export_to_csv(self, filepath: str) -> None:
        """
        Exporta entradas individuais de log para CSV.

        Levanta a exceção original em caso de falha de I/O.
        """
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([l.to_dict() for l in self.logs]).to_csv(
            output_path, index=False, encoding="utf-8"
        )

    # =========================================================================
    # CONSULTAS E FILTROS
    # =========================================================================

    def query_logs(
        self,
        event_type:  Optional[AuditEvent]  = None,
        status:      Optional[EventStatus] = None,
        start_time:  Optional[datetime]    = None,
        end_time:    Optional[datetime]    = None,
        search_term: Optional[str]         = None,
    ) -> List[AuditLogEntry]:
        """
        Filtra entradas de log por um ou mais critérios combinados.

        search_term busca no JSON serializado de details — útil para localizar
        eventos por query_id, chart_id ou qualquer campo interno sem precisar
        conhecer a estrutura exata do dicionário de detalhes.
        """
        result = self.logs

        if event_type:
            result = [l for l in result if l.event_type == event_type]
        if status:
            result = [l for l in result if l.status == status]
        if start_time:
            result = [l for l in result if l.timestamp >= start_time]
        if end_time:
            result = [l for l in result if l.timestamp <= end_time]
        if search_term:
            term = search_term.lower()
            result = [l for l in result if term in json.dumps(l.details).lower()]

        return result

    def get_session_summary_by_id(self, session_id: str) -> Optional[Dict]:
        """
        Retorna resumo de eventos de uma session_id específica dentro dos logs atuais.

        Útil quando um AuditLogger agrega logs de múltiplas sessões (ex: análise
        post-hoc). Retorna None quando session_id não tem nenhum registro.
        """
        session_logs = [l for l in self.logs if l.session_id == session_id]
        if not session_logs:
            return None

        start = min(l.timestamp for l in session_logs)
        end   = max(l.timestamp for l in session_logs)

        return {
            "session_id":       session_id,
            "start_time":       start.isoformat(),
            "end_time":         end.isoformat(),
            "duration_seconds": round((end - start).total_seconds(), 4),
            "total_events":     len(session_logs),
            "events":           [l.to_dict() for l in session_logs],
        }

    # =========================================================================
    # UTILITARIOS
    # =========================================================================

    def clear_logs(self) -> None:
        """Limpa todos os logs e contadores da sessão atual."""
        self.logs.clear()
        self._event_counts.clear()
        self._status_counts.clear()
        self._durations.clear()

    def reset_session(self) -> None:
        """Inicia nova sessão gerando um novo session_id e limpando todos os logs."""
        self.session_id = self._generate_session_id()
        self.start_time = datetime.now()
        self.clear_logs()

    def __repr__(self) -> str:
        errors = sum(1 for l in self.logs if l.status in (EventStatus.ERROR, EventStatus.CRITICAL))
        return (
            f"AuditLogger("
            f"session={self.session_id}, "
            f"events={len(self.logs)}, "
            f"errors={errors})"
        )

    def __len__(self) -> int:
        return len(self.logs)


# =============================================================================
# AUDIT ANALYZER — consultas em Delta
# =============================================================================

class AuditAnalyzer:
    """
    Consultas analíticas sobre logs históricos persistidos em Delta Lake.

    Opera diretamente sobre a tabela agent_audit_logs via Spark SQL. Todos
    os métodos validam parâmetros de entrada antes de interpolá-los em SQL
    para evitar injection — mesmo que os valores venham de fontes internas,
    o padrão defensivo é mantido consistentemente.

    Parâmetros
    ----------
    spark   : SparkSession ativa com acesso à tabela de auditoria.
    catalog : Catálogo Unity Catalog onde a tabela reside.
    schema  : Schema dentro do catálogo.
    """

    _SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")

    def __init__(self, spark, catalog: str = "dbx_srag_lab", schema: str = "audit"):
        self.spark      = spark
        self.catalog    = catalog
        self.schema     = schema
        self.table_name = f"{self.catalog}.{self.schema}.agent_audit_logs"

    def _validate_session_id(self, session_id: str) -> str:
        """
        Valida que session_id contém apenas caracteres seguros para interpolação SQL.

        Session IDs gerados por _generate_session_id() são sempre alfanuméricos
        com underscores. A validação protege contra casos em que um session_id
        externo seja passado diretamente sem sanitização prévia.

        Levanta ValueError antes de qualquer execução SQL — erro de contrato
        do chamador, não erro de infraestrutura.
        """
        if not self._SESSION_ID_RE.match(session_id):
            raise ValueError(
                f"session_id contém caracteres invalidos para uso em SQL: {session_id!r}. "
                f"Permitidos: alfanuméricos, underscore e hifen."
            )
        return session_id

    def get_sessions_summary(self, days: int = 7) -> pd.DataFrame:
        """
        Retorna resumo agregado por sessão dos últimos N dias.

        Parâmetros
        ----------
        days
            Janela retroativa em dias. Deve ser inteiro positivo — verificado
            com cast explícito para garantir que a interpolação em SQL seja
            sempre numérica.
        """
        days = int(days)
        query = f"""
            SELECT
                session_id,
                MIN(timestamp)         AS start_time,
                MAX(timestamp)         AS end_time,
                COUNT(*)               AS total_events,
                SUM(CASE WHEN status = 'error'   THEN 1 ELSE 0 END) AS error_count,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS success_count,
                AVG(elapsed_seconds)   AS avg_elapsed_seconds
            FROM {self.table_name}
            WHERE timestamp >= current_date() - INTERVAL {days} DAYS
            GROUP BY session_id
            ORDER BY start_time DESC
        """
        return self.spark.sql(query).toPandas()

    def get_error_trends(self, days: int = 30) -> pd.DataFrame:
        """
        Agrega erros por data e tipo de evento nos últimos N dias.

        Parâmetros
        ----------
        days
            Janela retroativa em dias. Cast explícito para int antes da
            interpolação em SQL para garantir tipo numérico.
        """
        days = int(days)
        query = f"""
            SELECT
                DATE(timestamp)  AS date,
                event_type,
                COUNT(*)         AS error_count
            FROM {self.table_name}
            WHERE status IN ('error', 'critical')
              AND timestamp >= current_date() - INTERVAL {days} DAYS
            GROUP BY DATE(timestamp), event_type
            ORDER BY date DESC, error_count DESC
        """
        return self.spark.sql(query).toPandas()

    def get_performance_metrics(self, session_id: str) -> pd.DataFrame:
        """
        Retorna métricas de latência por tipo de evento de uma sessão específica.

        Usa duration_seconds (duração real da operação) quando disponível,
        com fallback para elapsed_seconds (tempo de sessão) quando
        duration_seconds for NULL — para compatibilidade com logs gerados
        antes da migração para duration_seconds explícito.

        Parâmetros
        ----------
        session_id
            ID da sessão a analisar. Validado contra regex alfanumérico antes
            de qualquer interpolação em SQL.

        Levanta ValueError quando session_id contém caracteres não permitidos.
        """
        safe_id = self._validate_session_id(session_id)
        query = f"""
            SELECT
                event_type,
                COUNT(*)                                              AS event_count,
                AVG(COALESCE(duration_seconds, elapsed_seconds))      AS avg_seconds,
                MAX(COALESCE(duration_seconds, elapsed_seconds))      AS max_seconds,
                MIN(COALESCE(duration_seconds, elapsed_seconds))      AS min_seconds,
                SUM(CASE WHEN duration_seconds IS NOT NULL THEN 1
                         ELSE 0 END)                                  AS has_real_duration
            FROM {self.table_name}
            WHERE session_id = '{safe_id}'
            GROUP BY event_type
            ORDER BY avg_seconds DESC
        """
        return self.spark.sql(query).toPandas()

    def __repr__(self) -> str:
        return f"AuditAnalyzer(table={self.table_name})"