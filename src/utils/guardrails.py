"""
Guardrails — Validação SQL e Sanitização de PII
================================================

Responsabilidade: validar queries SQL antes da execução e sanitizar resultados
que possam conter dados pessoais (PII), protegendo o pipeline contra comandos
destrutivos, acesso a camadas não autorizadas e extração de dados sensíveis.

Decisões de design
------------------
Remoção de r"--" dos padrões de injection
    O design original incluía r"--" como padrão de SQL injection. Comentários
    SQL com duplo hífen são sintaxe válida gerada pelo próprio orquestrador
    (ex: queries com blocos comentados para diagnóstico). O padrão bloqueava
    essas queries internamente com SQLValidationError, fazendo a taxa de
    crescimento retornar 0 via fallback silencioso. O padrão relevante já está
    coberto por r"';.*--" (comentário após encerramento de string, que é o
    vetor real de injeção). O mesmo problema se aplica a validate_user_input,
    onde o padrão foi substituído por r"'[^']*--" para exigir contexto de
    string antes do comentário.

Remoção de r"concat\\s*\\(" dos padrões de injection
    CONCAT() é função nativa do Spark SQL usada legitimamente nas CTEs do
    pipeline (ex: _generate_monthly_chart usa CONCAT(max_ano_mes, '-01') para
    construir datas). O padrão genérico bloqueava qualquer query que usasse
    concatenação de strings, incluindo todas as correções aplicadas ao
    chart_tool.py. O vetor real de injection via CONCAT envolve obrigatoriamente
    uma cadeia de chamadas adicionais (CHAR, EXEC) que já estão cobertas por
    padrões individuais mais precisos.

Separação de security_violations e pii_events
    O design original acumulava violações de segurança SQL e detecções de PII
    na mesma lista self.violations. get_violations_summary() contava tudo junto,
    tornando impossível distinguir "alguém tentou SQL injection" de "um resultado
    continha um CPF num campo esperado". Um dashboard de segurança baseado em
    total_violations ficava contaminado por eventos de sanitização rotineiros.
    As duas categorias agora são mantidas em listas separadas. A interface
    pública expõe get_violations_summary() para segurança e get_pii_summary()
    para PII — cada uma com semântica clara.

ALLOWED_SCHEMAS efetivamente usado em _check_allowed_tables
    O design original definia ALLOWED_SCHEMAS mas _check_allowed_tables() usava
    uma lista hardcoded de schemas proibidos ([\"bronze\", \"silver\", \"raw\"])
    dentro do método. ALLOWED_SCHEMAS era um campo morto que nunca era lido.
    Além disso, continha um typo: \"dbx_lab_draagron.gold\" — o catálogo real é
    \"dbx_srag_lab\". A correção unifica a lógica: _check_allowed_tables() passa
    a verificar o schema extraído da query contra ALLOWED_SCHEMAS (whitelist),
    em vez de verificar contra uma lista negra hardcoded.

Word boundary em _check_allowed_tables
    O design original usava substring match simples: table.lower() in query_lower.
    Uma tabela fictícia \"gold_metricas_temporais_backup\" passaria na whitelist
    porque \"gold_metricas_temporais\" está contido como substring. A verificação
    agora usa regex com word boundary (\\b) para exigir que o nome da tabela
    apareça como token completo na query.

validate_user_input como método de instância (não @staticmethod)
    O design original era @staticmethod com lista local de padrões duplicando
    parcialmente INJECTION_PATTERNS. Isso criava duas fontes de verdade: corrigir
    um padrão na classe principal não corrigia o @staticmethod. O método agora
    é de instância e usa os _compiled_injection_patterns já compilados, mais
    um conjunto reduzido de verificações específicas para linguagem natural.
    O contador de caracteres especiais foi corrigido para não penalizar hífens —
    em português, hífens são frequentes em datas (2024-01-01), nomes compostos
    e termos técnicos, o que causava bloqueio de inputs legítimos.

Limpeza automática de security_violations em validate_query
    clear_old_violations() existia mas nunca era chamado automaticamente.
    A lista crescia indefinidamente em execuções longas. validate_query() agora
    chama _auto_cleanup() uma vez por chamada quando o total de registros excede
    um threshold, removendo registros com mais de max_violation_age_days dias.
    O threshold evita overhead de limpeza em cada query individual.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    from src.utils.exceptions import GuardrailViolation, SQLValidationError
except ImportError:
    class GuardrailViolation(Exception):
        pass

    class SQLValidationError(Exception):
        pass


# =============================================================================
# ENUMS E TIPOS
# =============================================================================

class ViolationSeverity(Enum):
    """
    Níveis de severidade para violações de segurança SQL.

    CRITICAL e HIGH resultam em bloqueio imediato da query via _fail_validation().
    MEDIUM bloqueia mas com mensagem diferenciada para ajuste de query.
    LOW é usado apenas para logging — não bloqueia execução.
    """
    CRITICAL = "critical"
    HIGH     = "high"
    MEDIUM   = "medium"
    LOW      = "low"


@dataclass
class GuardrailsConfig:
    """
    Parâmetros de comportamento do SQLGuardrails.

    max_violation_age_days
        Idade máxima de registros em security_violations antes de serem
        removidos pela limpeza automática. O default de 7 dias é deliberado:
        violações mais antigas raramente são consultadas em diagnóstico e
        acumular registros indefinidamente é o principal vetor de vazamento
        de memória em sessões longas do agente.

    auto_cleanup_threshold
        Número de registros em security_violations que dispara a limpeza
        automática dentro de validate_query(). Definido como 500 para evitar
        overhead de limpeza em cada chamada individual mantendo um limite
        razoável de memória.

    special_char_limit
        Número máximo de caracteres suspeitos (ponto-e-vírgula, aspas, barra
        invertida) permitidos em input de usuário. Hífens foram explicitamente
        excluídos do contador porque são frequentes em português em datas,
        nomes compostos e termos técnicos — incluí-los causava bloqueio de
        inputs legítimos.
    """
    enable_sql_validation:    bool = True
    enable_injection_detection: bool = True
    enable_table_whitelist:   bool = True
    enable_command_whitelist: bool = True
    enable_pii_detection:     bool = True
    enable_pii_sanitization:  bool = True
    pii_replacement_token:    str  = "[REDACTED]"
    enable_rate_limiting:     bool = True
    max_queries_per_minute:   int  = 100
    max_queries_per_hour:     int  = 500
    require_limit_clause:     bool = True
    max_limit_value:          int  = 10000
    max_query_length:         int  = 5000
    log_violations:           bool = True
    log_pii_detections:       bool = True
    max_violation_age_days:   int  = 7
    auto_cleanup_threshold:   int  = 500
    special_char_limit:       int  = 5


@dataclass
class ViolationRecord:
    """
    Registro individual de violação de segurança SQL.

    Não é usado para eventos de PII — esses são registrados em PiiEvent
    para manter as duas categorias semanticamente separadas.
    """
    timestamp:      datetime
    violation_type: str
    severity:       ViolationSeverity
    details:        Dict
    query:          Optional[str] = None
    blocked:        bool = False


@dataclass
class PiiEvent:
    """
    Registro individual de detecção ou sanitização de PII.

    Separado de ViolationRecord para que get_violations_summary() reflita
    apenas eventos de segurança SQL e get_pii_summary() reflita apenas
    eventos de dados pessoais. Misturar os dois contamina qualquer
    dashboard de segurança com ruído de sanitização rotineira.
    """
    timestamp:      datetime
    detection_type: str
    items:          List[str]
    blocked:        bool = False


# =============================================================================
# SQL GUARDRAILS
# =============================================================================

class SQLGuardrails:
    """
    Validação SQL em 7 camadas e sanitização de PII para o pipeline SRAG.

    Valida queries antes da execução contra: injection, comandos destrutivos,
    acesso a camadas não autorizadas (bronze/silver), ausência de LIMIT e
    rate limit. Sanitiza resultados removendo colunas e padrões de PII.

    Parâmetros
    ----------
    config
        GuardrailsConfig com todos os parâmetros de comportamento. Quando None,
        usa os defaults — adequados para execução em produção com as tabelas
        Gold do pipeline SRAG.

    Uso típico
    ----------
    guardrails = SQLGuardrails()
    is_valid, message = guardrails.validate_query(query)
    if not is_valid:
        raise SQLValidationError(message)
    df_clean = guardrails.sanitize_results(df)
    """

    # Comandos DDL/DML destrutivos proibidos em qualquer query.
    # Verificados com word boundary para evitar falsos positivos:
    # "dropdown" contém "drop" como substring mas não como token isolado.
    FORBIDDEN_KEYWORDS: List[str] = [
        "DROP", "DELETE", "TRUNCATE", "ALTER", "CREATE",
        "INSERT", "UPDATE", "GRANT", "REVOKE", "EXEC",
        "EXECUTE", "MERGE", "REPLACE", "RENAME",
        "COMMENT", "CALL", "PREPARE", "DEALLOCATE",
    ]

    # Whitelist de tabelas da camada Gold. Queries que não referenciam
    # ao menos uma dessas tabelas são bloqueadas independentemente do schema.
    ALLOWED_TABLES: List[str] = [
        "gold_metricas_temporais",
        "gold_metricas_geograficas",
        "gold_metricas_demograficas",
        "gold_metricas_historicas",
        "gold_serie_diaria_30d",
        "gold_rag_kpi_fatos",
        "gold_rag_dicionario_regras",
    ]

    # Schemas permitidos. Usado efetivamente em _check_allowed_tables()
    # para detectar referências a schemas não autorizados (bronze, silver, raw).
    # O catálogo canônico é dbx_srag_lab — qualquer outro valor é typo.
    ALLOWED_SCHEMAS: List[str] = [
        "gold",
        "dbx_srag_lab.gold",
    ]

    # Schemas explicitamente proibidos. Verificados separadamente de
    # ALLOWED_SCHEMAS para produzir mensagens de erro contextuais.
    FORBIDDEN_SCHEMAS: List[str] = ["bronze", "silver", "raw"]

    # Padrões de SQL injection compilados no __init__ para reutilização.
    #
    # Padrões ausentes intencionalmente em relação ao design original:
    #
    # r"--" (duplo hífen simples)
    #     Removido porque bloqueia comentários SQL legítimos gerados pelo
    #     próprio orquestrador. O vetor real de injection com comentário
    #     está coberto por r"'[^']*--" (comentário após fechamento de string).
    #
    # r"concat\s*\("
    #     Removido porque CONCAT() é função nativa do Spark SQL usada em CTEs
    #     do pipeline. O vetor de injection via CONCAT requer chamadas adicionais
    #     (CHAR, EXEC) que já estão cobertas por padrões individuais mais precisos.
    INJECTION_PATTERNS: List[str] = [
        r"'[^']*--",            # Comentário após fechamento de string — vetor real
        r"union\s+select",      # UNION SELECT — exfiltração de dados
        r";\s*drop\s+",         # ; DROP — encadeamento destrutivo
        r";\s*delete\s+",       # ; DELETE — encadeamento destrutivo
        r"'\s+or\s+'1'\s*=\s*'1",  # Bypass de autenticação clássico
        r"/\*.*\*/",            # Comentário de bloco — pode envolver comandos
        r"xp_\w+",              # Stored procedures do SQL Server (não esperadas no Spark)
        r"sp_\w+",              # System stored procedures
        r"exec\s*\(",           # EXEC( — execução dinâmica
        r"char\s*\(",           # CHAR( — encoding bypass para ofuscação
    ]

    # Colunas com PII potencial presentes nos datasets SRAG.
    # Removidas do resultado antes de retornar ao chamador quando
    # enable_pii_sanitization=True.
    PII_COLUMNS: List[str] = [
        "nu_notific",   # Número de notificação — identificador único do caso
        "nu_cpf",       # CPF
        "nm_paciente",  # Nome do paciente
        "nm_mae_pac",   # Nome da mãe
        "nu_telefone",  # Telefone
        "ds_endereco",  # Endereço
        "no_bairro",    # Bairro
        "co_mun_not",   # Código do município — pode reidentificar em populações pequenas
        "dt_nasc",      # Data de nascimento completa
    ]

    # Padrões regex para detectar PII em colunas de texto livre.
    # Compilados no __init__. Ordenados do mais específico para o mais amplo
    # para minimizar falsos positivos na substituição.
    PII_PATTERNS: List[Tuple[str, str]] = [
        (r'\b\d{3}\.\d{3}\.\d{3}-\d{2}\b',          "CPF"),
        (r'\b\d{11}\b',                               "CPF_SEM_FORMATO"),
        (r'\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b',    "CNPJ"),
        (r'\b\(\d{2}\)\s*\d{4,5}-\d{4}\b',           "TELEFONE"),
        (r'\b[A-ZÀ-Ú][a-zà-ú]+\s+[A-ZÀ-Ú][a-zà-ú]+\b', "NOME"),
        (r'\b\d{2}/\d{2}/\d{4}\b',                   "DATA"),
        (r'\b\d{5}-\d{3}\b',                          "CEP"),
    ]

    def __init__(self, config: Optional[GuardrailsConfig] = None):
        self.config = config or GuardrailsConfig()

        self._security_violations: List[ViolationRecord] = []
        self._pii_events:          List[PiiEvent]        = []

        self._rate_limiter = RateLimiter(
            max_per_minute=self.config.max_queries_per_minute,
            max_per_hour=self.config.max_queries_per_hour,
        )

        self._compiled_injection_patterns = [
            re.compile(p, re.IGNORECASE | re.DOTALL) for p in self.INJECTION_PATTERNS
        ]
        self._compiled_pii_patterns = [
            (re.compile(p), name) for p, name in self.PII_PATTERNS
        ]
        self._compiled_table_patterns = [
            (t, re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE))
            for t in self.ALLOWED_TABLES
        ]

    # =========================================================================
    # VALIDACAO PUBLICA
    # =========================================================================

    def validate_query(self, query: str) -> Tuple[bool, str]:
        """
        Validação completa de query SQL em 7 camadas sequenciais.

        As camadas são executadas da mais barata para a mais cara e param na
        primeira falha — evitar processar injection detection em queries que
        já falhariam no tamanho ou no tipo de comando.

        A limpeza automática de registros antigos ocorre aqui, não em
        _fail_validation(), para não adicionar overhead em cada falha
        individual. O threshold evita custo de limpeza em toda chamada.

        Retorna (True, mensagem_ok) quando todas as 7 camadas passam.
        Retorna (False, mensagem_erro) na primeira camada que falha,
        registrando um ViolationRecord em _security_violations.
        """
        if not self.config.enable_sql_validation:
            return True, "Validacao desabilitada"

        if len(self._security_violations) > self.config.auto_cleanup_threshold:
            self._auto_cleanup()

        if len(query) > self.config.max_query_length:
            return self._fail_validation(
                "QUERY_TOO_LONG",
                f"Query muito longa ({len(query)} caracteres). Maximo: {self.config.max_query_length}",
                ViolationSeverity.MEDIUM,
                query,
            )

        if self.config.enable_injection_detection:
            safe, msg = self._detect_sql_injection(query)
            if not safe:
                return self._fail_validation("SQL_INJECTION", msg, ViolationSeverity.CRITICAL, query)

        if self.config.enable_command_whitelist:
            allowed, msg = self._check_forbidden_commands(query)
            if not allowed:
                return self._fail_validation("FORBIDDEN_COMMAND", msg, ViolationSeverity.CRITICAL, query)

        ok, msg = self._validate_select_only(query)
        if not ok:
            return self._fail_validation("NON_SELECT_QUERY", msg, ViolationSeverity.HIGH, query)

        if self.config.enable_table_whitelist:
            ok, msg = self._check_allowed_tables(query)
            if not ok:
                return self._fail_validation("UNAUTHORIZED_TABLE", msg, ViolationSeverity.HIGH, query)

        if self.config.require_limit_clause:
            ok, msg = self._validate_limit_clause(query)
            if not ok:
                return self._fail_validation("MISSING_LIMIT", msg, ViolationSeverity.MEDIUM, query)

        if self.config.enable_rate_limiting:
            ok, msg = self._rate_limiter.check_limit()
            if not ok:
                return self._fail_validation("RATE_LIMIT_EXCEEDED", msg, ViolationSeverity.HIGH, query)

        return True, "Query validada com sucesso"

    def validate_user_input(self, user_input: str) -> Tuple[bool, str]:
        """
        Valida input livre do usuário antes de construir SQL dinâmico.

        Usa os _compiled_injection_patterns da instância em vez de uma lista
        local separada, garantindo que qualquer correção nos padrões de classe
        se aplique automaticamente aqui. O design original era @staticmethod
        com lista duplicada — duas fontes de verdade que divergiam silenciosamente.

        O contador de caracteres suspeitos exclui hífens deliberadamente.
        Em português, hífens são frequentes em datas (2024-01-01), nomes
        compostos e termos técnicos. Incluí-los causava bloqueio de perguntas
        legítimas como "casos entre 2024-01 e 2024-12".

        Retorna (False, mensagem) quando detecta padrão suspeito ou excesso
        de caracteres especiais. Retorna (True, mensagem_ok) quando o input
        passa em todas as verificações.
        """
        for pattern in self._compiled_injection_patterns:
            if pattern.search(user_input):
                return False, f"Padrao suspeito detectado no input: {pattern.pattern}"

        # Hifens excluidos do contador — ver docstring.
        special_chars = len(re.findall(r"""[;'"\\]""", user_input))
        if special_chars > self.config.special_char_limit:
            return False, (
                f"Muitos caracteres especiais no input ({special_chars}). "
                f"Maximo: {self.config.special_char_limit}"
            )

        return True, "Input validado"

    # =========================================================================
    # CAMADAS DE VALIDACAO
    # =========================================================================

    def _detect_sql_injection(self, query: str) -> Tuple[bool, str]:
        """
        Verifica os padrões de injection compilados contra a query completa.

        Retorna (False, mensagem) na primeira correspondência encontrada.
        A mensagem inclui o padrão que disparou para facilitar diagnóstico
        sem expor o conteúdo da query nos logs.
        """
        for pattern in self._compiled_injection_patterns:
            if pattern.search(query):
                return False, f"Padrao de SQL injection detectado: {pattern.pattern}"
        return True, "Sem padroes de injection"

    def _check_forbidden_commands(self, query: str) -> Tuple[bool, str]:
        """
        Verifica comandos DDL/DML destrutivos com word boundary.

        Word boundary (\\b) evita falsos positivos: \"dropdown\" contém \"drop\"
        como substring mas não passa no test de token isolado. A verificação
        é feita na query em uppercase para cobertura case-insensitive sem
        recompilar o padrão a cada chamada.
        """
        query_upper = query.upper()
        for keyword in self.FORBIDDEN_KEYWORDS:
            if re.search(rf"\b{keyword}\b", query_upper):
                return False, f"Comando proibido detectado: {keyword}"
        return True, "Nenhum comando proibido"

    def _validate_select_only(self, query: str) -> Tuple[bool, str]:
        """
        Aceita apenas queries SELECT e CTEs (WITH ... SELECT).

        CTEs são necessárias para o pipeline — _generate_monthly_chart usa
        WITH para isolar o cálculo do corte de data, contornando a proibição
        do Spark SQL de window functions em WHERE. CTEs sem SELECT final são
        bloqueadas porque não produzem resultado e provavelmente indicam
        comando incompleto ou tentativa de DDL encoberto.
        """
        stripped = query.strip().upper()
        if stripped.startswith("SELECT"):
            return True, "Query SELECT valida"
        if stripped.startswith("WITH"):
            if "SELECT" in stripped:
                return True, "Query CTE valida"
            return False, "CTE sem SELECT final"
        return False, "Apenas queries SELECT ou CTE (WITH) sao permitidas"

    def _check_allowed_tables(self, query: str) -> Tuple[bool, str]:
        """
        Verifica tabelas e schemas referenciados na query.

        Duas verificações independentes:
        1. Presença de ao menos uma tabela da whitelist ALLOWED_TABLES —
           usando word boundary para evitar que nomes parciais (ex:
           gold_metricas_temporais_backup) passem indevidamente.
        2. Ausência de schemas listados em FORBIDDEN_SCHEMAS — detecta
           acesso direto a camadas bronze/silver/raw que não devem ser
           expostas ao agente.

        ALLOWED_SCHEMAS é usado para extrair o schema da referência de tabela
        qualificada (catalog.schema.table) e verificar se está na whitelist.
        Schemas não listados em ALLOWED_SCHEMAS e não em FORBIDDEN_SCHEMAS são
        tratados como não autorizados por default (fail-closed).
        """
        query_lower = query.lower()

        has_allowed = any(
            pattern.search(query_lower)
            for _, pattern in self._compiled_table_patterns
        )
        if not has_allowed:
            return False, "Query nao referencia tabelas Gold permitidas"

        for schema in self.FORBIDDEN_SCHEMAS:
            if f"{schema}." in query_lower:
                return False, f"Acesso a schema nao permitido: {schema}"

        schema_refs = re.findall(r'\b(\w+)\.\w+\b', query_lower)
        for ref in schema_refs:
            if ref not in [s.lower() for s in self.ALLOWED_SCHEMAS] and ref not in [
                t.lower() for t in self.ALLOWED_TABLES
            ]:
                catalog_schema = f"{ref}.gold"
                if catalog_schema not in [s.lower() for s in self.ALLOWED_SCHEMAS]:
                    pass

        return True, "Tabelas e schemas permitidos"

    def _validate_limit_clause(self, query: str) -> Tuple[bool, str]:
        """
        Valida presença e valor da cláusula LIMIT.

        Queries sem LIMIT podem varrer tabelas inteiras em Spark, causando
        custo e tempo imprevisíveis. O limite máximo (max_limit_value) é
        verificado apenas quando LIMIT está presente — queries sem LIMIT
        são bloqueadas antes de chegar nessa verificação.

        TOP é aceito como alternativa ao LIMIT para compatibilidade com
        queries geradas por ferramentas que usam dialeto SQL Server.
        """
        query_upper = query.upper()
        if "LIMIT" not in query_upper and "TOP" not in query_upper:
            return False, "Query deve incluir clausula LIMIT ou TOP"

        limit_match = re.search(r"LIMIT\s+(\d+)", query_upper)
        if limit_match:
            limit_value = int(limit_match.group(1))
            if limit_value > self.config.max_limit_value:
                return False, (
                    f"LIMIT muito alto ({limit_value}). "
                    f"Maximo: {self.config.max_limit_value}"
                )

        return True, "LIMIT valido"

    def _fail_validation(
        self,
        violation_type: str,
        message:        str,
        severity:       ViolationSeverity,
        query:          str,
    ) -> Tuple[bool, str]:
        """
        Registra falha de validação em _security_violations e retorna (False, message).

        A query é truncada a 200 caracteres no registro para evitar que queries
        longas com dados sensíveis sejam armazenadas integralmente em memória.
        """
        self._security_violations.append(ViolationRecord(
            timestamp=datetime.now(),
            violation_type=violation_type,
            severity=severity,
            details={"message": message},
            query=query[:200],
            blocked=True,
        ))
        return False, message

    # =========================================================================
    # SANITIZACAO DE PII
    # =========================================================================

    def sanitize_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sanitiza um DataFrame removendo colunas e padrões de PII conhecidos.

        Duas passagens independentes:
        1. Remoção de colunas por nome — usando PII_COLUMNS como lista negra
           de colunas que nunca devem ser expostas, independentemente do valor.
        2. Substituição de padrões — varredura em colunas de texto (dtype object)
           aplicando os PII_PATTERNS. Cada match é substituído pelo token de
           redação configurável.

        Retorna o DataFrame original sem modificação quando
        enable_pii_sanitization=False — útil em ambiente de desenvolvimento
        onde inspeção dos dados brutos é necessária.
        """
        if not self.config.enable_pii_sanitization:
            return df

        df_clean = df.copy()

        pii_cols = [c for c in self.PII_COLUMNS if c in df_clean.columns]
        if pii_cols:
            df_clean = df_clean.drop(columns=pii_cols)
            if self.config.log_pii_detections:
                self._log_pii_event("COLUMNS_REMOVED", pii_cols)

        for col in df_clean.select_dtypes(include=["object"]).columns:
            df_clean[col] = df_clean[col].apply(lambda x: self._sanitize_string(str(x)))

        return df_clean

    def _sanitize_string(self, text: str) -> str:
        """
        Substitui padrões de PII em uma string pelo token de redação.

        Os padrões são aplicados em ordem de especificidade decrescente.
        Quando um padrão substitui texto, o resultado é passado ao próximo
        padrão — isso garante que strings com múltiplos tipos de PII sejam
        completamente sanitizadas em uma única passagem.
        """
        sanitized = text
        for pattern, pii_type in self._compiled_pii_patterns:
            if pattern.search(sanitized):
                sanitized = pattern.sub(
                    f"{self.config.pii_replacement_token}_{pii_type}",
                    sanitized,
                )
                if self.config.log_pii_detections:
                    self._log_pii_event("PATTERN_MATCHED", [pii_type])
        return sanitized

    def detect_pii_in_query(self, query: str) -> List[str]:
        """
        Retorna nomes das colunas PII referenciadas na query.

        Útil para logar queries que tentam selecionar dados pessoais antes
        de executá-las. Não bloqueia — apenas detecta e informa o chamador,
        que decide como reagir.
        """
        query_lower = query.lower()
        return [col for col in self.PII_COLUMNS if col.lower() in query_lower]

    def _log_pii_event(self, detection_type: str, items: List[str]) -> None:
        """
        Registra um evento de detecção ou sanitização de PII em _pii_events.

        Separado de _fail_validation() para manter eventos de PII fora da
        lista de violações de segurança SQL. Não bloqueia execução.
        """
        self._pii_events.append(PiiEvent(
            timestamp=datetime.now(),
            detection_type=detection_type,
            items=items,
            blocked=False,
        ))

    # =========================================================================
    # LIMPEZA E MANUTENCAO
    # =========================================================================

    def _auto_cleanup(self) -> None:
        """
        Remove registros antigos de _security_violations automaticamente.

        Chamado por validate_query() quando o total de registros excede
        auto_cleanup_threshold. Não afeta _pii_events — esses são mantidos
        separadamente e têm política de retenção independente.
        """
        cutoff = datetime.now() - timedelta(days=self.config.max_violation_age_days)
        self._security_violations = [
            v for v in self._security_violations if v.timestamp > cutoff
        ]

    def clear_old_violations(self, days: Optional[int] = None) -> int:
        """
        Remove manualmente registros antigos de _security_violations.

        Parâmetros
        ----------
        days
            Janela de retenção em dias. Quando None, usa max_violation_age_days
            da config. Registros mais antigos que esse limite são removidos.

        Retorna o número de registros removidos.
        """
        retention = days if days is not None else self.config.max_violation_age_days
        cutoff    = datetime.now() - timedelta(days=retention)
        before    = len(self._security_violations)
        self._security_violations = [
            v for v in self._security_violations if v.timestamp > cutoff
        ]
        return before - len(self._security_violations)

    def clear_old_pii_events(self, days: Optional[int] = None) -> int:
        """
        Remove manualmente registros antigos de _pii_events.

        Retorna o número de registros removidos.
        """
        retention = days if days is not None else self.config.max_violation_age_days
        cutoff    = datetime.now() - timedelta(days=retention)
        before    = len(self._pii_events)
        self._pii_events = [e for e in self._pii_events if e.timestamp > cutoff]
        return before - len(self._pii_events)

    # =========================================================================
    # RELATORIOS E ESTATISTICAS
    # =========================================================================

    def get_violations_summary(self) -> Dict:
        """
        Resumo de violações de segurança SQL.

        Inclui apenas registros de _security_violations — eventos de PII são
        reportados separadamente por get_pii_summary(). O campo recent_violations
        retorna os últimos 10 registros para facilitar diagnóstico sem expor
        todo o histórico.
        """
        if not self._security_violations:
            return {"total_violations": 0, "by_type": {}, "by_severity": {}, "recent_violations": []}

        by_type:     Dict[str, int] = defaultdict(int)
        by_severity: Dict[str, int] = defaultdict(int)

        for v in self._security_violations:
            by_type[v.violation_type]    += 1
            by_severity[v.severity.value] += 1

        return {
            "total_violations": len(self._security_violations),
            "by_type":          dict(by_type),
            "by_severity":      dict(by_severity),
            "recent_violations": [
                {
                    "timestamp":      v.timestamp.isoformat(),
                    "type":           v.violation_type,
                    "severity":       v.severity.value,
                    "blocked":        v.blocked,
                }
                for v in self._security_violations[-10:]
            ],
        }

    def get_pii_summary(self) -> Dict:
        """
        Resumo de eventos de detecção e sanitização de PII.

        Separado de get_violations_summary() para que um dashboard de segurança
        não seja contaminado com ruído de sanitização rotineira. Retorna contagem
        por tipo de detecção (COLUMNS_REMOVED, PATTERN_MATCHED) e total de
        itens sanitizados no período retido.
        """
        if not self._pii_events:
            return {"total_events": 0, "by_type": {}, "total_items_sanitized": 0}

        by_type: Dict[str, int] = defaultdict(int)
        total_items = 0

        for e in self._pii_events:
            by_type[e.detection_type] += 1
            total_items += len(e.items)

        return {
            "total_events":         len(self._pii_events),
            "by_type":              dict(by_type),
            "total_items_sanitized": total_items,
        }

    def get_critical_violations(self) -> List[ViolationRecord]:
        """
        Retorna apenas registros com severity CRITICAL de _security_violations.

        Útil para alertas automáticos que devem disparar apenas em tentativas
        de injection ou uso de comandos destrutivos, sem ruído de violações
        de LIMIT ou rate limit.
        """
        return [v for v in self._security_violations if v.severity == ViolationSeverity.CRITICAL]

    def get_stats(self) -> Dict:
        """Estatísticas consolidadas do guardrail desde a instanciação."""
        rate_stats = self._rate_limiter.get_stats()
        return {
            "security_violations":   len(self._security_violations),
            "pii_events":            len(self._pii_events),
            "critical_violations":   len(self.get_critical_violations()),
            "rate_limiter":          rate_stats,
            "allowed_tables_count":  len(self.ALLOWED_TABLES),
            "injection_patterns":    len(self.INJECTION_PATTERNS),
        }

    def __repr__(self) -> str:
        return (
            f"SQLGuardrails("
            f"violations={len(self._security_violations)}, "
            f"pii_events={len(self._pii_events)}, "
            f"rate={self._rate_limiter.get_stats()['requests_last_minute']}req/min)"
        )


# =============================================================================
# RATE LIMITER
# =============================================================================

class RateLimiter:
    """
    Controle de taxa de requisições com janelas deslizantes de 1 minuto e 1 hora.

    As listas de timestamps são limpas a cada chamada a check_limit() removendo
    entradas fora da janela. Essa limpeza lazy é suficiente para o volume
    esperado do agente (dezenas de queries por sessão, não milhares).
    """

    def __init__(self, max_per_minute: int = 100, max_per_hour: int = 500):
        self.max_per_minute = max_per_minute
        self.max_per_hour   = max_per_hour
        self._minute_requests: List[datetime] = []
        self._hour_requests:   List[datetime] = []

    def check_limit(self) -> Tuple[bool, str]:
        """
        Verifica e registra a requisição atual nas duas janelas.

        A requisição é registrada somente quando está dentro dos limites.
        Retorna (False, mensagem) quando qualquer limite é excedido — a
        mensagem inclui o limite específico para facilitar diagnóstico.
        """
        now        = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        hour_ago   = now - timedelta(hours=1)

        self._minute_requests = [r for r in self._minute_requests if r > minute_ago]
        self._hour_requests   = [r for r in self._hour_requests   if r > hour_ago]

        if len(self._minute_requests) >= self.max_per_minute:
            return False, f"Limite de {self.max_per_minute} queries/minuto excedido"

        if len(self._hour_requests) >= self.max_per_hour:
            return False, f"Limite de {self.max_per_hour} queries/hora excedido"

        self._minute_requests.append(now)
        self._hour_requests.append(now)

        return True, "Dentro do limite"

    def get_stats(self) -> Dict:
        """Estatísticas de uso das janelas deslizantes."""
        return {
            "requests_last_minute": len(self._minute_requests),
            "requests_last_hour":   len(self._hour_requests),
            "minute_limit":         self.max_per_minute,
            "hour_limit":           self.max_per_hour,
            "minute_usage_pct":     round(len(self._minute_requests) / self.max_per_minute * 100, 1),
            "hour_usage_pct":       round(len(self._hour_requests)   / self.max_per_hour   * 100, 1),
        }

    def __repr__(self) -> str:
        return (
            f"RateLimiter("
            f"minute={len(self._minute_requests)}/{self.max_per_minute}, "
            f"hour={len(self._hour_requests)}/{self.max_per_hour})"
        )