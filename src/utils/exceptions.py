"""
Exceptions — Hierarquia de Exceções do Sistema SRAG
====================================================

Responsabilidade: definir todas as exceções customizadas do pipeline,
organizadas em hierarquia que permite captura granular por componente
ou captura ampla via SRAGSystemError.

Decisões de design
------------------
recovery_hint apenas em to_dict(), não em __str__()
    O design original concatenava recovery_hint à mensagem passada para
    super().__init__() — o que o tornava parte da string da exceção retornada
    por str(exc). Como to_dict() também serializa recovery_hint como campo
    separado, qualquer log que registrasse a exceção como string E como dict
    produzia o hint duas vezes. A correção mantém recovery_hint apenas em
    to_dict() e no atributo de instância. format_error_for_user() acessa o
    atributo diretamente e pode renderizá-lo quando quiser, sem depender da
    mensagem string.

ErrorContext usa AuditEvent enum via import lazy
    O design original passava a string "ERROR" como event_type para
    audit_logger.log_event(). O método log_event() chama event_type.value
    internamente — strings não têm .value, logo o context manager lançava
    AttributeError em qualquer uso real com um AuditLogger instanciado.
    O componente era inutilizável desde que foi escrito.

    O import de AuditEvent não pode ser feito no topo do módulo porque
    audit.py importa AuditLogSaveError de exceptions.py — importação circular.
    A solução é import lazy dentro de __exit__(): o import só ocorre quando
    o context manager captura uma exceção, e apenas uma vez (Python cacheia
    módulos em sys.modules após a primeira importação).

Exceções específicas de guardrail como classes levantáveis
    SQLInjectionDetected, ForbiddenCommandError e UnauthorizedTableAccess
    existiam mas nunca eram levantadas — guardrails.py usava _fail_validation()
    retornando tuplas (False, message) e sql_tool.py levantava sempre o genérico
    SQLValidationError. A hierarquia perde todo o valor quando o tipo específico
    não é propagado: um bloco except ForbiddenCommandError nunca seria ativado.

    As classes permanecem definidas e documentadas aqui como o contrato correto.
    A integração com guardrails.py está documentada em cada classe — o chamador
    que converte o resultado de validate_query() em exceção deve usar a subclasse
    específica correspondente ao violation_type retornado.

format_error_for_user() sem emojis
    O design original retornava strings com emojis (emoji, emoji, emoji) para
    indicar status. O padrão de documentação do projeto proíbe emojis em qualquer
    ponto do código. A função agora retorna texto plano estruturado — prefixos
    como "[Erro]", "[Recuperavel]" e "[Sugestao]" são equivalentes funcionais
    sem dependência de renderização Unicode.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional


# =============================================================================
# EXCECAO BASE
# =============================================================================

class SRAGSystemError(Exception):
    """
    Exceção base para todo o sistema SRAG.

    Todas as exceções customizadas herdam desta classe, permitindo captura
    hierárquica: exceto SRAGSystemError captura qualquer erro do sistema;
    exceto SQLError captura apenas erros de SQL.

    Atributos
    ---------
    message
        Mensagem de erro sem recovery_hint — str(exc) não inclui a sugestão
        para evitar duplicação quando o erro é logado como string e como dict.
    error_code
        Código identificador do erro. Default: nome da classe. Usado como
        campo de agrupamento em dashboards de auditoria.
    details
        Dicionário com contexto adicional. Deve ser serializável em JSON.
    timestamp
        Momento da instanciação — não da captura. Em cadeia de exceções
        (__cause__), o timestamp de cada exceção é independente.
    recoverable
        True quando o pipeline pode continuar sem intervenção manual após
        o erro. Usado por ErrorContext para decidir se suprime ou propaga.
    recovery_hint
        Sugestão de ação corretiva. Presente em to_dict() e acessível via
        atributo — não embutido em str(exc) para evitar duplicação nos logs.
    """

    def __init__(
        self,
        message:        str,
        error_code:     Optional[str]          = None,
        details:        Optional[Dict[str, Any]] = None,
        recoverable:    bool                   = False,
        recovery_hint:  Optional[str]          = None,
    ):
        self.message       = message
        self.error_code    = error_code or self.__class__.__name__
        self.details       = details or {}
        self.timestamp     = datetime.now()
        self.recoverable   = recoverable
        self.recovery_hint = recovery_hint

        super().__init__(f"[{self.error_code}] {message}")

    def to_dict(self) -> Dict:
        """
        Serializa a exceção para dicionário JSON-safe.

        recovery_hint aparece aqui como campo estruturado — não na mensagem
        string — para que logs que registram ambos não dupliquem a sugestão.
        """
        return {
            "error_type":     self.__class__.__name__,
            "error_code":     self.error_code,
            "message":        self.message,
            "details":        self.details,
            "timestamp":      self.timestamp.isoformat(),
            "recoverable":    self.recoverable,
            "recovery_hint":  self.recovery_hint,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(code={self.error_code}, message={self.message!r})"


# =============================================================================
# ORQUESTRACAO
# =============================================================================

class OrchestratorError(SRAGSystemError):
    """Erro no orquestrador ou no grafo LangGraph."""

    def __init__(self, message: str, node_name: Optional[str] = None, **kwargs):
        details = kwargs.pop("details", {})
        if node_name:
            details["node_name"] = node_name
        super().__init__(message, details=details, **kwargs)


class NodeExecutionError(OrchestratorError):
    """
    Falha na execução de um nó específico do grafo LangGraph.

    Levantado pelo nó quando captura uma exceção interna e precisa comunicar
    ao orquestrador qual nó falhou sem perder o contexto do erro original.
    Use raise NodeExecutionError(...) from original_exc para preservar a
    cadeia de exceções.
    """

    def __init__(self, node_name: str, message: str, **kwargs):
        super().__init__(
            message=f"Erro no no '{node_name}': {message}",
            node_name=node_name,
            **kwargs,
        )


class StateTransitionError(OrchestratorError):
    """Transição de estado inválida no grafo LangGraph."""
    pass


class WorkflowError(OrchestratorError):
    """Erro estrutural no fluxo de trabalho — ex: nó ausente, aresta inválida."""
    pass


# =============================================================================
# COLETA DE DADOS
# =============================================================================

class DataCollectionError(SRAGSystemError):
    """Erro base para coleta de dados do pipeline."""
    pass


class MetricsCollectionError(DataCollectionError):
    """
    Falha ao coletar métricas epidemiológicas das tabelas Gold.

    Levantado quando uma query SQL retorna zero linhas ou quando as tabelas
    Gold estão indisponíveis. O orquestrador deve capturar este erro e
    registrar metricas={} no estado — não deixar o campo ausente, pois
    ausência causa KeyError downstream enquanto dict vazio produz N/A
    controlado no relatório.
    """

    def __init__(self, message: str, metric_type: Optional[str] = None, **kwargs):
        details = kwargs.pop("details", {})
        if metric_type:
            details["metric_type"] = metric_type
        super().__init__(
            message,
            details=details,
            recovery_hint="Verifique se as tabelas Gold estao atualizadas e acessiveis",
            **kwargs,
        )


class NewsCollectionError(DataCollectionError):
    """
    Falha ao coletar notícias via API Tavily.

    Sempre recoverable=True — o pipeline pode gerar relatório sem contexto
    de notícias recentes. O orquestrador deve logar o erro e continuar.
    """

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            recoverable=True,
            recovery_hint="O sistema pode continuar sem noticias. Verifique a API Tavily.",
            **kwargs,
        )


class GeographicDataError(DataCollectionError):
    """Falha ao coletar dados geográficos por UF."""
    pass


class DemographicDataError(DataCollectionError):
    """Falha ao coletar dados demográficos por faixa etária."""
    pass


# =============================================================================
# SQL
# =============================================================================

class SQLError(SRAGSystemError):
    """
    Erro base para operações SQL.

    O campo query é truncado a 200 caracteres no details para evitar que
    queries longas com dados sensíveis sejam armazenadas integralmente em
    logs de auditoria.
    """

    def __init__(self, message: str, query: Optional[str] = None, **kwargs):
        details = kwargs.pop("details", {})
        if query:
            details["query"] = query[:200]
        super().__init__(message, details=details, **kwargs)


class SQLExecutionError(SQLError):
    """
    Falha durante a execução de uma query SQL no Spark.

    Levantado por sql_tool.py quando spark.sql() ou df.collect() lança
    exceção. Inclui a query truncada em details para diagnóstico.
    """

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            recovery_hint="Verifique a sintaxe SQL e disponibilidade das tabelas",
            **kwargs,
        )


class SQLValidationError(SQLError):
    """
    Query rejeitada pelos guardrails antes de ser enviada ao Spark.

    Levantado por sql_tool.py quando guardrails.validate_query() retorna
    (False, message). Não é recoverable — a query precisa ser corrigida
    antes de nova tentativa.
    """

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            recoverable=False,
            recovery_hint="Ajuste a query para atender aos guardrails de segurança",
            **kwargs,
        )


class QueryTimeoutError(SQLError):
    """
    Query excedeu o tempo limite de execução.

    Deve ser levantado por sql_tool.py quando implementar controle de timeout
    via threading.Timer ou signal. A exceção existe e está pronta — o mecanismo
    de disparo no sql_tool.py ainda não foi implementado.
    """

    def __init__(self, message: str = "Query excedeu tempo limite", **kwargs):
        super().__init__(
            message,
            recoverable=True,
            recovery_hint="Adicione filtros para reduzir o volume de dados processados",
            **kwargs,
        )


class TableNotFoundError(SQLError):
    """Tabela referenciada na query não existe no catálogo."""

    def __init__(self, table_name: str, **kwargs):
        super().__init__(
            f"Tabela '{table_name}' nao encontrada",
            error_code="TABLE_NOT_FOUND",
            details={"table_name": table_name},
            **kwargs,
        )


class InsufficientDataError(SQLError):
    """Query retornou linhas insuficientes para o cálculo requerido."""

    def __init__(self, message: str = "Dados insuficientes para analise", **kwargs):
        super().__init__(message, recoverable=True, **kwargs)


# =============================================================================
# GUARDRAILS
# =============================================================================

class GuardrailViolation(SRAGSystemError):
    """
    Violação de guardrail de segurança SQL.

    Levantado quando validate_query() retorna False e o chamador converte
    o resultado em exceção. Subclasses específicas devem ser usadas quando
    o violation_type for conhecido — elas permitem captura granular:

        except SQLInjectionDetected:
            # tratar tentativa de injection especificamente
        except GuardrailViolation:
            # tratar qualquer violação genericamente

    O mapeamento de violation_type para subclasse:
        SQL_INJECTION      -> SQLInjectionDetected
        FORBIDDEN_COMMAND  -> ForbiddenCommandError
        UNAUTHORIZED_TABLE -> UnauthorizedTableAccess
        RATE_LIMIT_EXCEEDED -> RateLimitExceeded
        PII_DETECTED       -> PIIDetectedError
        Outros             -> GuardrailViolation (base)
    """

    def __init__(
        self,
        message:        str,
        violation_type: Optional[str] = None,
        severity:       str           = "HIGH",
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        details.update({"violation_type": violation_type, "severity": severity})
        super().__init__(
            message,
            error_code=f"GUARDRAIL_{violation_type}" if violation_type else "GUARDRAIL_VIOLATION",
            details=details,
            recoverable=False,
            **kwargs,
        )


class SQLInjectionDetected(GuardrailViolation):
    """
    Padrão de SQL injection detectado na query.

    Levantado pelo chamador de validate_query() quando violation_type for
    "SQL_INJECTION". severity=CRITICAL — não deve ser suprimido nem logado
    apenas como warning.
    """

    def __init__(self, pattern: str, **kwargs):
        super().__init__(
            f"Padrao de SQL injection detectado: {pattern}",
            violation_type="SQL_INJECTION",
            severity="CRITICAL",
            **kwargs,
        )


class ForbiddenCommandError(GuardrailViolation):
    """
    Comando DDL/DML destrutivo (DROP, DELETE, etc.) detectado na query.

    Levantado pelo chamador de validate_query() quando violation_type for
    "FORBIDDEN_COMMAND".
    """

    def __init__(self, command: str, **kwargs):
        details = kwargs.pop("details", {})
        details["command"] = command
        super().__init__(
            f"Comando proibido detectado: {command}",
            violation_type="FORBIDDEN_COMMAND",
            severity="CRITICAL",
            details=details,
            **kwargs,
        )


class UnauthorizedTableAccess(GuardrailViolation):
    """
    Query referencia tabela fora da whitelist ALLOWED_TABLES.

    Levantado pelo chamador de validate_query() quando violation_type for
    "UNAUTHORIZED_TABLE".
    """

    def __init__(self, table: str, **kwargs):
        details = kwargs.pop("details", {})
        details["table"] = table
        super().__init__(
            f"Acesso nao autorizado a tabela: {table}",
            violation_type="UNAUTHORIZED_TABLE",
            severity="HIGH",
            details=details,
            **kwargs,
        )


class RateLimitExceeded(GuardrailViolation):
    """
    Limite de requisições por minuto ou hora excedido.

    Única violação de guardrail que é recoverable=True — o sistema pode
    tentar novamente após o intervalo de espera sugerido.
    """

    def __init__(self, limit_type: str = "queries", **kwargs):
        super().__init__(
            f"Limite de {limit_type} excedido",
            violation_type="RATE_LIMIT",
            severity="MEDIUM",
            recoverable=True,
            recovery_hint="Aguarde alguns minutos antes de tentar novamente",
            **kwargs,
        )


class PIIDetectedError(GuardrailViolation):
    """
    Dados pessoais (PII) detectados em resultado de query.

    severity=LOW — detecção de PII é esperada e tratada automaticamente
    por sanitize_results(). Esta exceção é levantada apenas quando a
    sanitização não pôde ser aplicada.
    """

    def __init__(self, pii_type: str, **kwargs):
        details = kwargs.pop("details", {})
        details["pii_type"] = pii_type
        super().__init__(
            f"PII detectado: {pii_type}",
            violation_type="PII_DETECTED",
            severity="LOW",
            details=details,
            **kwargs,
        )


# =============================================================================
# WEB SEARCH
# =============================================================================

class WebSearchError(SRAGSystemError):
    """Erro base para operações de busca web."""
    pass


class SearchAPIError(WebSearchError):
    """
    Falha na API Tavily — timeout, quota esgotada ou erro HTTP.

    Levantado por web_search_tool.py quando a chamada à API falha.
    recoverable=True — o pipeline pode continuar sem notícias recentes.
    """

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            recoverable=True,
            recovery_hint="Verifique conectividade e quota da API Tavily",
            **kwargs,
        )


class SearchValidationError(WebSearchError):
    """Parâmetros de busca inválidos antes de chamar a API."""
    pass


class NoResultsFoundError(WebSearchError):
    """Busca executada com sucesso mas sem resultados para a query."""

    def __init__(self, query: str, **kwargs):
        super().__init__(
            f"Nenhum resultado encontrado para: {query}",
            recoverable=True,
            details={"query": query},
            **kwargs,
        )


# =============================================================================
# GRAFICOS
# =============================================================================

class ChartError(SRAGSystemError):
    """Erro base para operações de geração de gráficos."""
    pass


class ChartGenerationError(ChartError):
    """
    Falha durante a geração de um gráfico Plotly.

    Deve ser levantado pelos métodos _generate_*() em chart_tool.py em vez
    de capturar bare Exception. recoverable=True — o pipeline pode entregar
    relatório sem o gráfico afetado.
    """

    def __init__(self, message: str, chart_type: Optional[str] = None, **kwargs):
        details = kwargs.pop("details", {})
        if chart_type:
            details["chart_type"] = chart_type
        super().__init__(
            message,
            details=details,
            recoverable=True,
            recovery_hint="O sistema pode continuar sem graficos",
            **kwargs,
        )


class ChartValidationError(ChartError):
    """DataFrame não contém as colunas necessárias para o tipo de gráfico."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            recovery_hint="Verifique se os dados tem as colunas necessarias",
            **kwargs,
        )


class ChartExportError(ChartError):
    """
    Falha ao escrever o arquivo HTML do gráfico no Volume.

    Deve ser levantado por _write_chart_html() em chart_tool.py quando
    write_html() falha, em vez de capturar bare Exception. Fornece o
    formato e o path do arquivo no details para diagnóstico.
    """

    def __init__(self, fmt: str, message: str, **kwargs):
        details = kwargs.pop("details", {})
        details["format"] = fmt
        super().__init__(
            f"Erro ao exportar grafico em {fmt}: {message}",
            details=details,
            **kwargs,
        )


# =============================================================================
# RELATORIOS
# =============================================================================

class ReportError(SRAGSystemError):
    """Erro base para geração de relatórios."""
    pass


class ReportGenerationError(ReportError):
    """Falha ao gerar o relatório Markdown ou JSON."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="REPORT_GENERATION_FAILED",
            **kwargs,
        )


class ReportValidationError(ReportError):
    """Relatório gerado está incompleto — seções obrigatórias ausentes."""

    def __init__(
        self,
        message:          str,
        missing_sections: Optional[List[str]] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if missing_sections:
            details["missing_sections"] = missing_sections
        super().__init__(
            message,
            details=details,
            recovery_hint="O LLM deve gerar todas as secoes obrigatorias",
            **kwargs,
        )


class LLMError(ReportError):
    """Falha na comunicação com o modelo de linguagem."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            recoverable=True,
            recovery_hint="Verifique API key e conectividade com o provedor LLM",
            **kwargs,
        )


# =============================================================================
# CONFIGURACAO
# =============================================================================

class ConfigurationError(SRAGSystemError):
    """Erro de configuração do sistema."""

    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        details = kwargs.pop("details", {})
        if config_key:
            details["config_key"] = config_key
        super().__init__(message, details=details, **kwargs)


class MissingCredentialsError(ConfigurationError):
    """Credencial ausente no Databricks Secrets ou variável de ambiente."""

    def __init__(self, credential_name: str, **kwargs):
        super().__init__(
            f"Credencial ausente: {credential_name}",
            error_code="MISSING_CREDENTIALS",
            details={"credential": credential_name},
            recovery_hint=f"Configure {credential_name} no Databricks Secrets",
            **kwargs,
        )


class InvalidConfigError(ConfigurationError):
    """Valor de configuração presente mas inválido."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            recovery_hint="Verifique o arquivo de configuracao",
            **kwargs,
        )


# =============================================================================
# AUDITORIA
# =============================================================================

class AuditError(SRAGSystemError):
    """Erro no sistema de auditoria."""
    pass


class AuditLogSaveError(AuditError):
    """
    Falha ao persistir logs em Delta Lake.

    Levantado por AuditLogger.save_to_delta() quando a escrita falha.
    recoverable=True — os logs permanecem em memória e podem ser exportados
    via export_to_json() como fallback. O orquestrador deve capturar este
    erro e alertar sem abortar a execução.
    """

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            recoverable=True,
            recovery_hint="Logs ficam em memoria. Verifique permissoes no Delta Lake",
            **kwargs,
        )


# =============================================================================
# CACHE
# =============================================================================

class CacheError(SRAGSystemError):
    """Erro no sistema de cache."""
    pass


class CacheFullError(CacheError):
    """Cache atingiu capacidade máxima."""

    def __init__(self, **kwargs):
        super().__init__(
            "Cache atingiu capacidade maxima",
            recoverable=True,
            recovery_hint="Cache sera limpo automaticamente (LRU)",
            **kwargs,
        )


# =============================================================================
# RAG
# =============================================================================

class RAGError(SRAGSystemError):
    """Erro base para o sistema RAG."""
    pass


class EmbeddingError(RAGError):
    """Falha ao gerar embeddings via provider configurado."""

    def __init__(self, message: str, provider: Optional[str] = None, **kwargs):
        details = kwargs.pop("details", {})
        if provider:
            details["provider"] = provider
        super().__init__(
            message,
            details=details,
            recoverable=True,
            recovery_hint="Verifique API key do provider de embeddings",
            **kwargs,
        )


class VectorStoreError(RAGError):
    """Falha na conexão ou consulta ao Databricks Vector Search."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            recoverable=True,
            recovery_hint="Verifique conexao com Databricks Vector Search",
            **kwargs,
        )


class RetrievalError(RAGError):
    """Falha no retrieval de documentos para uma query específica."""

    def __init__(self, message: str, query: Optional[str] = None, **kwargs):
        details = kwargs.pop("details", {})
        if query:
            details["query"] = query[:100]
        super().__init__(message, details=details, recoverable=True, **kwargs)


class ContextBuildError(RAGError):
    """Falha ao construir o contexto de documentos para enviar ao LLM."""
    pass


class DocumentLoaderError(RAGError):
    """Falha ao carregar documentos das tabelas Gold para o vector store."""
    pass


# =============================================================================
# VALIDACAO DE ENTRADA
# =============================================================================

class ValidationError(SRAGSystemError):
    """Erro de validação de entrada do usuário."""
    pass


class InvalidQueryError(ValidationError):
    """Query do usuário inválida ou incompreensível."""

    def __init__(self, reason: str, **kwargs):
        super().__init__(
            f"Query invalida: {reason}",
            recovery_hint="Reformule a pergunta de forma mais clara",
            **kwargs,
        )


class InvalidParameterError(ValidationError):
    """Parâmetro de chamada de ferramenta com valor inválido."""

    def __init__(self, param_name: str, reason: str, **kwargs):
        super().__init__(
            f"Parametro '{param_name}' invalido: {reason}",
            details={"parameter": param_name, "reason": reason},
            **kwargs,
        )


# =============================================================================
# CONTEXT MANAGER
# =============================================================================

class ErrorContext:
    """
    Context manager para tratamento padronizado de erros com logging opcional.

    Registra a exceção capturada no AuditLogger quando fornecido e,
    opcionalmente, suprime exceções recuperáveis em vez de propagá-las.

    Parâmetros
    ----------
    operation_name
        Nome da operação sendo protegida — incluído no details do evento
        de auditoria para facilitar diagnóstico.
    audit_logger
        Instância de AuditLogger. Quando None, o context manager funciona
        apenas como supressor sem logging.
    raise_on_error
        Quando False, suprime exceções recuperáveis (recoverable=True) em
        vez de propagá-las. Exceções não recuperáveis são sempre propagadas.

    Uso
    ---
    with ErrorContext("gerar_grafico", audit_logger=logger, raise_on_error=False):
        chart_tool.generate_all_charts()
    # continua mesmo se ChartGenerationError (recoverable=True) for levantada
    """

    def __init__(
        self,
        operation_name: str,
        audit_logger=None,
        raise_on_error: bool = True,
    ):
        self.operation_name = operation_name
        self.audit_logger   = audit_logger
        self.raise_on_error = raise_on_error
        self.error: Optional[Exception] = None

    def __enter__(self) -> "ErrorContext":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is None:
            return False

        self.error = exc_val

        if self.audit_logger is not None:
            try:
                # Import lazy para evitar importacao circular:
                # audit.py importa AuditLogSaveError deste modulo.
                # Importar AuditEvent no topo causaria ImportError circular.
                # Python cacheia modulos em sys.modules apos a primeira importacao,
                # entao o custo ocorre apenas na primeira captura de excecao.
                from src.utils.audit import AuditEvent, EventStatus

                self.audit_logger.log_event(
                    AuditEvent.NODE_ERROR,
                    {
                        "operation":     self.operation_name,
                        "error_type":    exc_type.__name__,
                        "error_message": str(exc_val),
                        "recoverable":   (
                            exc_val.recoverable
                            if isinstance(exc_val, SRAGSystemError)
                            else False
                        ),
                    },
                    status=EventStatus.ERROR,
                )
            except Exception:
                # Falha no logging nao deve mascarar a excecao original.
                pass

        if isinstance(exc_val, SRAGSystemError) and exc_val.recoverable:
            if not self.raise_on_error:
                return True

        return False


# =============================================================================
# UTILITARIOS
# =============================================================================

def format_error_for_user(error: Exception) -> str:
    """
    Formata uma exceção como mensagem legível para exibição ao usuário.

    Retorna texto plano sem emojis — o renderizador do notebook ou interface
    é responsável por qualquer formatação visual adicional.

    Para SRAGSystemError, inclui recovery_hint quando disponível e indica
    se o erro é recuperável. Para exceções genéricas, retorna a representação
    string direta.
    """
    if isinstance(error, SRAGSystemError):
        parts = [f"[Erro] {error.message}"]

        if error.recovery_hint:
            parts.append(f"[Sugestao] {error.recovery_hint}")

        if error.recoverable:
            parts.append("[Recuperavel] O sistema pode continuar.")
        else:
            parts.append("[Critico] Este erro requer intervencao manual.")

        return "\n\n".join(parts)

    return f"[Erro inesperado] {error!s}"


def is_recoverable(error: Exception) -> bool:
    """
    Retorna True quando o erro é recuperável e o pipeline pode continuar.

    Retorna False para qualquer exceção que não seja SRAGSystemError —
    a decisão conservadora para exceções desconhecidas.
    """
    if isinstance(error, SRAGSystemError):
        return error.recoverable
    return False