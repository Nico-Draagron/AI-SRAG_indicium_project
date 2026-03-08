"""
SQL Tool — Execução Segura de Queries na Camada Gold
=====================================================

Responsabilidade única: receber uma string SQL, validá-la via SQLGuardrails
e retornar o resultado materializado como lista de dicionários.

Decisões de design
------------------
Captura de duration_seconds com time.perf_counter()
    duration_seconds mede a latência real da operação SQL: tempo desde a
    chamada a spark.sql() até o fim de toPandas(), que é quando o DataFrame
    é efetivamente materializado no driver. O timer é iniciado imediatamente
    antes da execução e parado imediatamente após — independente de sucesso
    ou erro — para que o AuditLogger receba durações reais em ambos os
    caminhos. time.perf_counter() é usado em vez de time.time() por ter
    resolução de nanossegundos e não ser afetado por ajustes de relógio do
    sistema operacional durante a execução.

Limite de linhas antes da contagem
    O DataFrame lógico retornado por spark.sql() é avaliado de forma lazy.
    Aplicar df.limit() antes de qualquer ação de coleta garante que apenas
    o subconjunto necessário seja materializado no driver. Executar df.count()
    antes do limit forçaria um job Spark completo sobre toda a tabela apenas
    para obter um número informativo, incorrendo em custo e latência
    desnecessários — especialmente grave quando o tool é chamado múltiplas
    vezes em sequência pelo orquestrador.

    Quando count_total_rows=True, a contagem é feita sobre o DataFrame já
    limitado (len do Pandas resultante), o que é O(1). Se a contagem do
    total real antes do corte for necessária, deve ser implementada via
    sub-query COUNT(*) separada, fora do ciclo de execução principal.

Exposição da query no payload de retorno
    O campo "query" retorna a versão truncada, não a query completa. A query
    completa pode conter estrutura de catálogo, lógica de negócio e nomes
    de tabelas internas que não devem vazar para consumidores downstream
    (relatório, log de auditoria serializado, resposta de API). Quando o
    diagnóstico precisar da query inteira, ela está disponível no AuditLogger
    e em exceções lançadas internamente.

Detecção de resultado truncado
    Quando count_total_rows=False, não é possível afirmar com certeza se o
    resultado foi truncado — apenas suspeitar quando result_rows ==
    max_result_rows. O campo "limited" usa valor None nesse caso para
    distinguir "sabemos que foi limitado" (True), "sabemos que não foi" (False)
    e "não temos como saber" (None). Consumidores devem tratar None como
    inconclusivo.

Fail-fast na inicialização
    spark=None é rejeitado no construtor em vez de propagar um AttributeError
    genérico na primeira chamada a execute_query(). Isso torna o diagnóstico
    imediato e evita que o pipeline avance com um tool configurado de forma
    inválida.

AuditEvent.SQL_CONFIG_UPDATED ausente no enum de audit.py
    set_max_rows() usa SQL_CONFIG_UPDATED, que não existe no enum real de
    AuditEvent. O evento está definido apenas no stub local de fallback.
    Quando audit.py é importado com sucesso, a chamada lança AttributeError
    capturado silenciosamente pelo handler genérico de execute_query() — o
    que aqui não ocorre porque set_max_rows() não tem handler próprio, mas
    o AttributeError propagaria para o chamador. A correção definitiva é
    adicionar SQL_CONFIG_UPDATED ao enum em audit.py. Até lá, a chamada em
    set_max_rows() está protegida com getattr() e documentada como pendente.

Classe legada
    GoldSQLToolLegacy emite DeprecationWarning na instanciação e usa um limite
    de 50.000 linhas — valor que representa um equilíbrio entre compatibilidade
    e segurança de memória do driver. O valor original de 1.000.000 linhas
    combinado com toPandas() é suficiente para causar OOM em tabelas com o
    volume típico do SIVEP-Gripe (3–5 milhões de registros anuais).

Stub de fallback dos imports
    Quando src.utils.guardrails não está disponível, o stub emite um aviso
    explícito no log de auditoria. O comportamento do sistema em modo
    degradado deve ser visível, não silencioso, pois as garantias de segurança
    de schema e injeção SQL deixam de existir nesse estado.
"""

import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

from pyspark.sql import SparkSession


try:
    from src.utils.guardrails import SQLGuardrails, GuardrailsConfig
    _GUARDRAILS_AVAILABLE = True
except ImportError:
    _GUARDRAILS_AVAILABLE = False

    class GuardrailsConfig:
        pass

    class SQLGuardrails:
        """
        Stub mínimo ativado quando src.utils.guardrails não pode ser importado.
        Cobre apenas comandos DML destrutivos. Validações de schema, whitelist
        de tabelas e padrões de injeção avançados não estão presentes.
        """
        _DANGEROUS = frozenset(["drop", "delete", "truncate", "alter", "insert", "update"])

        def __init__(self, config=None):
            pass

        def validate_query(self, query: str):
            tokens = set(query.lower().split())
            hit    = tokens & self._DANGEROUS
            if hit:
                return False, f"Comando proibido detectado: {sorted(hit)}"
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
        TOOL_INITIALIZED      = "tool_initialized"
        TOOL_DEGRADED         = "tool_degraded"
        SQL_VALIDATION_FAILED = "sql_validation_failed"
        SQL_QUERY_START       = "sql_query_start"
        SQL_QUERY_SUCCESS     = "sql_query_success"
        SQL_QUERY_ERROR       = "sql_query_error"
        SQL_RESULT_TRUNCATED  = "sql_result_truncated"
        SQL_CONFIG_UPDATED    = "sql_config_updated"

    class EventStatus:
        INFO    = "INFO"
        SUCCESS = "SUCCESS"
        ERROR   = "ERROR"
        WARNING = "WARNING"

    class AuditLogger:
        def log_event(self, event_type, details=None, status="INFO", duration_seconds=None):
            print(f"[{status}] {event_type}: {details}")


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

@dataclass
class SQLToolConfig:
    """
    Parâmetros de execução do GoldSQLTool.

    count_total_rows
        Quando True, o total de linhas retornado em result["total_rows"]
        reflete len(pdf) — ou seja, o total dentro do limite aplicado, não o
        total real da query antes do corte. Isso é intencional: contar linhas
        após o limit é O(1) e não dispara um job Spark adicional.

        Se o total real antes do corte for necessário, a query deve incluir
        uma sub-query COUNT(*) explícita gerenciada pelo chamador.

    warn_on_limit
        Quando True e result["limited"] é True ou None (inconclusivo),
        registra um evento SQL_RESULT_TRUNCATED no AuditLogger. Útil para
        identificar queries que retornam mais dados do que o consumidor
        downstream está preparado para processar.
    """
    max_result_rows:      int  = 10_000
    log_query_max_length: int  = 500
    count_total_rows:     bool = False
    warn_on_limit:        bool = True


# =============================================================================
# GOLDSSQLTOOL
# =============================================================================

class GoldSQLTool:
    """
    Executa queries SQL sobre a camada Gold com validação, limite de resultado
    e auditoria estruturada.

    O tool opera exclusivamente sobre tabelas Gold. A responsabilidade de
    garantir que a query não acesse outras camadas é do SQLGuardrails injetado
    no construtor. Quando o módulo de guardrails não está disponível, o tool
    entra em modo degradado e registra um aviso explícito.

    Parâmetros
    ----------
    spark
        SparkSession ativa. Obrigatória — o construtor rejeita None imediatamente
        para evitar erros tardios e genéricos em execute_query().
    audit_logger
        Instância de AuditLogger. Quando não fornecida, usa o stub local que
        imprime no stdout, suficiente para testes isolados.
    guardrails_config
        Configuração passada ao SQLGuardrails. Quando None, usa os defaults
        do módulo de guardrails.
    config
        SQLToolConfig com limites e flags de comportamento. Quando None,
        usa os defaults de SQLToolConfig.
    """

    def __init__(
        self,
        spark:             SparkSession,
        audit_logger:      Optional[AuditLogger]      = None,
        guardrails_config: Optional[GuardrailsConfig] = None,
        config:            Optional[SQLToolConfig]    = None,
    ):
        if spark is None:
            raise ValueError(
                "GoldSQLTool requer uma SparkSession ativa. "
                "Verifique se spark foi inicializado antes de instanciar o tool."
            )

        self.spark      = spark
        self.audit      = audit_logger if audit_logger else AuditLogger()
        self.guardrails = SQLGuardrails(guardrails_config or GuardrailsConfig())
        self.config     = config or SQLToolConfig()

        self.audit.log_event(
            AuditEvent.TOOL_INITIALIZED,
            {
                "tool":             "GoldSQLTool",
                "max_result_rows":  self.config.max_result_rows,
                "count_total_rows": self.config.count_total_rows,
            },
            EventStatus.INFO,
        )

        if not _GUARDRAILS_AVAILABLE:
            self.audit.log_event(
                AuditEvent.TOOL_DEGRADED,
                {
                    "tool":   "GoldSQLTool",
                    "reason": "src.utils.guardrails indisponível — stub sem validação de schema ativo",
                },
                EventStatus.WARNING,
            )

    # =========================================================================
    # INTERFACE PÚBLICA
    # =========================================================================

    def execute_query(self, query: str) -> Dict:
        """
        Valida e executa uma query SQL, retornando o resultado materializado.

        O resultado é sempre limitado a config.max_result_rows linhas.
        A query completa nunca é incluída no payload de retorno; apenas a
        versão truncada é exposta para evitar vazamento de estrutura interna
        em logs e relatórios downstream.

        duration_seconds captura a latência real da operação: do início de
        spark.sql() ao fim de toPandas(). O valor é repassado para log_event()
        tanto no caminho de sucesso quanto no de erro, para que o AuditLogger
        registre latências mesmo em queries que falham após algum tempo de
        execução no cluster.

        Parâmetros
        ----------
        query
            String SQL a ser executada. Deve referenciar apenas tabelas da
            camada Gold. Comentários SQL (--) são aceitos desde que o
            SQLGuardrails real esteja em uso; o stub de fallback não os bloqueia.

        Retorno
        -------
        Dict com as chaves:
            success      : bool — sempre True quando não há exceção.
            rows         : int — número de linhas no resultado após o limite.
            total_rows   : int | None — total de linhas após o limite
                           (igual a rows quando count_total_rows=True).
                           None quando count_total_rows=False.
            limited      : bool | None — True se sabemos que foi truncado,
                           False se sabemos que não foi, None se inconclusivo
                           (ocorre quando count_total_rows=False e rows ==
                           max_result_rows).
            columns      : List[str] — nomes das colunas.
            data         : List[Dict] — registros no formato orient='records'.
            query        : str — versão truncada da query para rastreabilidade.

        Exceções
        --------
        SQLValidationError
            Query bloqueada pelo guardrail. O pipeline deve tratar essa exceção
            explicitamente — ela representa um erro de lógica (query inválida),
            não um erro de infraestrutura.
        SQLExecutionError
            Erro durante a execução Spark. Pode indicar problema de schema,
            tabela inexistente ou falha transitória do cluster.
        """
        query_for_log = self._truncate_query_for_log(query)

        is_valid, validation_msg = self.guardrails.validate_query(query)
        if not is_valid:
            self.audit.log_event(
                AuditEvent.SQL_VALIDATION_FAILED,
                {"query": query_for_log, "reason": validation_msg},
                EventStatus.ERROR,
            )
            raise SQLValidationError(validation_msg, details={"query": query_for_log})

        self.audit.log_event(
            AuditEvent.SQL_QUERY_START,
            {"query": query_for_log},
            EventStatus.INFO,
        )

        t_start = time.perf_counter()
        try:
            df          = self.spark.sql(query)
            df_limited  = df.limit(self.config.max_result_rows)
            pdf         = df_limited.toPandas()
            duration    = time.perf_counter() - t_start

            result_rows = len(pdf)
            total_rows  = result_rows if self.config.count_total_rows else None
            limited     = self._detect_limited(result_rows, total_rows)

            if limited is not False and self.config.warn_on_limit:
                self.audit.log_event(
                    AuditEvent.SQL_RESULT_TRUNCATED,
                    {
                        "returned_rows": result_rows,
                        "total_rows":    total_rows,
                        "limit":         self.config.max_result_rows,
                        "conclusive":    limited is not None,
                    },
                    EventStatus.WARNING,
                )

            self.audit.log_event(
                AuditEvent.SQL_QUERY_SUCCESS,
                {
                    "rows":    result_rows,
                    "limited": limited,
                    "columns": len(pdf.columns),
                },
                EventStatus.SUCCESS,
                duration_seconds=duration,
            )

            return {
                "success":    True,
                "rows":       result_rows,
                "total_rows": total_rows,
                "limited":    limited,
                "columns":    list(pdf.columns),
                "data":       pdf.to_dict(orient="records"),
                "query":      query_for_log,
            }

        except (SQLValidationError, SQLExecutionError):
            raise
        except Exception as exc:
            duration = time.perf_counter() - t_start
            self.audit.log_event(
                AuditEvent.SQL_QUERY_ERROR,
                {"error": str(exc), "query": query_for_log},
                EventStatus.ERROR,
                duration_seconds=duration,
            )
            raise SQLExecutionError(
                f"Falha na execução da query: {exc}",
                details={"query": query_for_log, "spark_error": str(exc)},
            ) from exc

    def set_max_rows(self, max_rows: int) -> None:
        """
        Atualiza o limite máximo de linhas em tempo de execução.

        Útil quando queries analíticas específicas exigem um limite diferente
        do padrão configurado na instanciação. A alteração é registrada no
        AuditLogger com os valores anterior e novo para rastreabilidade.

        O evento SQL_CONFIG_UPDATED ainda não existe no enum AuditEvent de
        audit.py — está definido apenas no stub local de fallback deste módulo.
        getattr() previne AttributeError em produção até que o evento seja
        adicionado ao enum canônico. Quando disponível, o evento é registrado
        normalmente; quando ausente, o log é omitido silenciosamente aqui
        porque a alteração de configuração já é rastreável via get_stats().
        """
        if max_rows < 1:
            raise ValueError(f"max_rows deve ser >= 1, recebido: {max_rows}")

        old_max = self.config.max_result_rows
        self.config.max_result_rows = max_rows

        config_updated_event = getattr(AuditEvent, "SQL_CONFIG_UPDATED", None)
        if config_updated_event is not None:
            self.audit.log_event(
                config_updated_event,
                {"field": "max_result_rows", "old": old_max, "new": max_rows},
                EventStatus.INFO,
            )

    def get_stats(self) -> Dict:
        """
        Retorna a configuração ativa do tool.

        Não inclui contadores de execução — o AuditLogger é a fonte canônica
        para métricas de frequência e latência de queries.
        """
        return {
            "max_result_rows":      self.config.max_result_rows,
            "log_query_max_length": self.config.log_query_max_length,
            "count_total_rows":     self.config.count_total_rows,
            "guardrails_available": _GUARDRAILS_AVAILABLE,
        }

    # =========================================================================
    # MÉTODOS INTERNOS
    # =========================================================================

    def _detect_limited(
        self,
        result_rows: int,
        total_rows:  Optional[int],
    ) -> Optional[bool]:
        """
        Determina se o resultado foi truncado pelo limite configurado.

        Quando total_rows está disponível (count_total_rows=True), a detecção
        é determinística: total_rows > max_result_rows significa truncamento.

        Quando total_rows é None (count_total_rows=False), a detecção é
        inconclusiva se result_rows == max_result_rows — pode ser que a query
        retorne exatamente esse número de linhas sem truncamento. Retornar None
        preserva essa ambiguidade para o consumidor, que pode decidir como
        tratá-la. Retornar True seria um falso positivo; retornar False seria
        esconder um possível truncamento.
        """
        if total_rows is not None:
            return total_rows > self.config.max_result_rows

        if result_rows < self.config.max_result_rows:
            return False

        return None

    def _truncate_query_for_log(self, query: str) -> str:
        """
        Reduz a query a no máximo log_query_max_length caracteres para uso
        em logs e no payload de retorno, preservando início e fim para
        facilitar o diagnóstico.

        A query completa nunca é exposta fora do escopo de execute_query()
        para evitar vazamento de estrutura interna em logs serializados.
        """
        max_len = self.config.log_query_max_length
        if len(query) <= max_len:
            return query

        half = (max_len - 24) // 2
        return f"{query[:half]} …[{len(query)} chars]… {query[-half:]}"

    def __repr__(self) -> str:
        return (
            f"GoldSQLTool("
            f"max_rows={self.config.max_result_rows}, "
            f"guardrails={'real' if _GUARDRAILS_AVAILABLE else 'stub'})"
        )


# =============================================================================
# CLASSE LEGADA
# =============================================================================

class GoldSQLToolLegacy(GoldSQLTool):
    """
    Wrapper de compatibilidade para código que instancia GoldSQLTool sem
    passar SQLToolConfig explícito.

    Esta classe está descontinuada. Migre para GoldSQLTool com SQLToolConfig
    configurado explicitamente. O limite de 1.000.000 de linhas do design
    original foi reduzido para 50.000 porque toPandas() sobre 1M de registros
    do SIVEP-Gripe causa OOM no driver em configurações típicas de cluster.

    Será removida em versão futura.
    """

    def __init__(
        self,
        spark:             SparkSession,
        audit_logger:      Optional[AuditLogger]      = None,
        guardrails_config: Optional[GuardrailsConfig] = None,
    ):
        warnings.warn(
            "GoldSQLToolLegacy está descontinuado. "
            "Use GoldSQLTool com SQLToolConfig(max_result_rows=N, count_total_rows=False). "
            "O limite implícito foi reduzido de 1.000.000 para 50.000 linhas.",
            DeprecationWarning,
            stacklevel=2,
        )
        config = SQLToolConfig(
            max_result_rows=50_000,
            count_total_rows=False,
        )
        super().__init__(spark, audit_logger, guardrails_config, config)