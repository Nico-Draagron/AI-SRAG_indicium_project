"""
Guardrails - Sistema de Validação e Segurança
=============================================

Sistema completo de guardrails para validação SQL, sanitização de PII,
e proteção contra injeções e acessos não autorizados.

Features:
    - Validação multi-camada de SQL
    - Detecção de SQL injection
    - Sanitização automática de PII
    - Whitelist de tabelas e comandos
    - Análise de padrões suspeitos
    - Rate limiting
    - Logging de violações

Author: AI Engineer Certification - Indicium
Date: January 2025
Version: 2.0.0
"""

import re
from typing import Dict, List, Tuple, Set, Optional, Pattern
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict

import pandas as pd

from src.utils.exceptions import GuardrailViolation, SQLValidationError


# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

class ViolationSeverity(Enum):
    """Níveis de severidade de violações"""
    CRITICAL = "critical"    # Bloqueia imediatamente
    HIGH = "high"            # Bloqueia com warning
    MEDIUM = "medium"        # Permite com warning
    LOW = "low"              # Apenas log


@dataclass
class GuardrailsConfig:
    """Configuração do sistema de guardrails"""
    # SQL Validation
    enable_sql_validation: bool = True
    enable_injection_detection: bool = True
    enable_table_whitelist: bool = True
    enable_command_whitelist: bool = True
    
    # PII Protection
    enable_pii_detection: bool = True
    enable_pii_sanitization: bool = True
    pii_replacement_token: str = "[REDACTED]"
    
    # Rate Limiting
    enable_rate_limiting: bool = True
    max_queries_per_minute: int = 100
    max_queries_per_hour: int = 500
    
    # Query Limits
    require_limit_clause: bool = True
    max_limit_value: int = 10000
    max_query_length: int = 5000
    
    # Logging
    log_violations: bool = True
    log_pii_detections: bool = True


@dataclass
class ViolationRecord:
    """Registro de violação detectada"""
    timestamp: datetime
    violation_type: str
    severity: ViolationSeverity
    details: Dict
    query: Optional[str] = None
    blocked: bool = False


# =============================================================================
# VALIDADOR SQL
# =============================================================================

class SQLGuardrails:
    """
    Sistema de guardrails para validação SQL
    
    Valida queries antes da execução protegendo contra:
        - SQL injection
        - Comandos perigosos (DROP, DELETE, etc)
        - Acesso a tabelas não autorizadas
        - Queries sem LIMIT
        - Padrões suspeitos
    
    Example:
        >>> guardrails = SQLGuardrails()
        >>> is_valid, message = guardrails.validate_query("SELECT * FROM gold.table LIMIT 10")
        >>> if not is_valid:
        ...     print(f"Query bloqueada: {message}")
    """
    
    # =========================================================================
    # LISTAS DE CONTROLE
    # =========================================================================
    
    # Comandos SQL proibidos (DDL/DML destrutivos)
    FORBIDDEN_KEYWORDS = [
        "DROP", "DELETE", "TRUNCATE", "ALTER", "CREATE",
        "INSERT", "UPDATE", "GRANT", "REVOKE", "EXEC",
        "EXECUTE", "MERGE", "REPLACE", "RENAME",
        "COMMENT", "CALL", "PREPARE", "DEALLOCATE"
    ]
    
    # Tabelas permitidas (Gold layer apenas)
    ALLOWED_TABLES = [
        "gold_metricas_temporais",
        "gold_metricas_geograficas",
        "gold_metricas_demograficas",
        "gold_series_temporais",
        "gold_resumo_geral",
        "gold_analise_avancada",
        "vw_dashboard_principal",
        "vw_metricas_ultimos_6_meses",
        "vw_top10_ufs",
        "vw_alertas_mortalidade"
    ]
    
    # Schemas permitidos
    ALLOWED_SCHEMAS = ["gold", "dbx_lab_draagron.gold"]
    
    # Padrões de SQL injection
    INJECTION_PATTERNS = [
        r"';.*--",                    # Comentário após ponto-vírgula
        r"union\s+select",            # UNION SELECT
        r";\s*drop\s+",              # ; DROP
        r";\s*delete\s+",            # ; DELETE
        r"'\s+or\s+'1'\s*=\s*'1",   # ' OR '1'='1
        r"--",                        # Comentários SQL
        r"/\*.*\*/",                  # Comentários de bloco
        r"xp_\w+",                    # Stored procedures perigosas
        r"sp_\w+",                    # System stored procedures
        r"exec\s*\(",                 # EXEC(
        r"char\s*\(",                 # CHAR( (encoding bypass)
        r"concat\s*\(",               # CONCAT com injeção
    ]
    
    # Colunas com PII potencial
    PII_COLUMNS = [
        "nu_notific",      # Número de notificação (identificador único)
        "nu_cpf",          # CPF
        "nm_paciente",     # Nome do paciente
        "nm_mae_pac",      # Nome da mãe
        "nu_telefone",     # Telefone
        "ds_endereco",     # Endereço
        "no_bairro",       # Bairro
        "co_mun_not",      # Código do município (pode identificar)
        "dt_nasc",         # Data de nascimento completa
    ]
    
    # Padrões regex para detectar PII em strings
    PII_PATTERNS = [
        (r'\b\d{3}\.\d{3}\.\d{3}-\d{2}\b', 'CPF'),           # CPF: 123.456.789-01
        (r'\b\d{11}\b', 'CPF_SEM_FORMATO'),                   # CPF sem formatação
        (r'\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b', 'CNPJ'),    # CNPJ
        (r'\b\(\d{2}\)\s*\d{4,5}-\d{4}\b', 'TELEFONE'),      # Telefone: (11) 98765-4321
        (r'\b[A-ZÀ-Ú][a-zà-ú]+\s+[A-ZÀ-Ú][a-zà-ú]+\b', 'NOME'),  # Nome Próprio
        (r'\b\d{2}/\d{2}/\d{4}\b', 'DATA'),                  # Data: DD/MM/YYYY
        (r'\b\d{5}-\d{3}\b', 'CEP'),                         # CEP
    ]
    
    def __init__(self, config: Optional[GuardrailsConfig] = None):
        self.config = config or GuardrailsConfig()
        self.violations: List[ViolationRecord] = []
        self._rate_limiter = RateLimiter(
            max_per_minute=self.config.max_queries_per_minute,
            max_per_hour=self.config.max_queries_per_hour
        )
        
        # Compilar padrões regex para performance
        self._compiled_injection_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.INJECTION_PATTERNS
        ]
        self._compiled_pii_patterns = [
            (re.compile(pattern), name) for pattern, name in self.PII_PATTERNS
        ]
    
    # =========================================================================
    # VALIDAÇÃO DE QUERIES
    # =========================================================================
    
    def validate_query(self, query: str) -> Tuple[bool, str]:
        """
        Validação completa de query SQL
        
        Args:
            query: Query SQL a validar
            
        Returns:
            Tuple (is_valid, message)
        """
        if not self.config.enable_sql_validation:
            return True, "Validação desabilitada"
        
        # 1. Verificar tamanho
        if len(query) > self.config.max_query_length:
            return self._fail_validation(
                "QUERY_TOO_LONG",
                f"Query muito longa ({len(query)} caracteres). Máximo: {self.config.max_query_length}",
                ViolationSeverity.MEDIUM,
                query
            )
        
        # 2. Detectar SQL injection
        if self.config.enable_injection_detection:
            is_safe, injection_msg = self._detect_sql_injection(query)
            if not is_safe:
                return self._fail_validation(
                    "SQL_INJECTION",
                    injection_msg,
                    ViolationSeverity.CRITICAL,
                    query
                )
        
        # 3. Verificar comandos proibidos
        if self.config.enable_command_whitelist:
            is_allowed, command_msg = self._check_forbidden_commands(query)
            if not is_allowed:
                return self._fail_validation(
                    "FORBIDDEN_COMMAND",
                    command_msg,
                    ViolationSeverity.CRITICAL,
                    query
                )
        
        # 4. Validar tipo de comando (apenas SELECT ou WITH)
        is_select, select_msg = self._validate_select_only(query)
        if not is_select:
            return self._fail_validation(
                "NON_SELECT_QUERY",
                select_msg,
                ViolationSeverity.HIGH,
                query
            )
        
        # 5. Verificar tabelas permitidas
        if self.config.enable_table_whitelist:
            is_allowed_table, table_msg = self._check_allowed_tables(query)
            if not is_allowed_table:
                return self._fail_validation(
                    "UNAUTHORIZED_TABLE",
                    table_msg,
                    ViolationSeverity.HIGH,
                    query
                )
        
        # 6. Verificar cláusula LIMIT
        if self.config.require_limit_clause:
            has_limit, limit_msg = self._validate_limit_clause(query)
            if not has_limit:
                return self._fail_validation(
                    "MISSING_LIMIT",
                    limit_msg,
                    ViolationSeverity.MEDIUM,
                    query
                )
        
        # 7. Verificar rate limiting
        if self.config.enable_rate_limiting:
            is_allowed_rate, rate_msg = self._rate_limiter.check_limit()
            if not is_allowed_rate:
                return self._fail_validation(
                    "RATE_LIMIT_EXCEEDED",
                    rate_msg,
                    ViolationSeverity.HIGH,
                    query
                )
        
        # Query válida
        return True, "✅ Query validada com sucesso"
    
    def _detect_sql_injection(self, query: str) -> Tuple[bool, str]:
        """Detecta padrões de SQL injection"""
        for pattern in self._compiled_injection_patterns:
            if pattern.search(query):
                return False, f"❌ Padrão de SQL injection detectado: {pattern.pattern}"
        
        return True, "Sem padrões de injeção"
    
    def _check_forbidden_commands(self, query: str) -> Tuple[bool, str]:
        """Verifica comandos SQL proibidos"""
        query_upper = query.upper()
        
        for keyword in self.FORBIDDEN_KEYWORDS:
            # Usar regex para evitar falsos positivos (ex: "dropdown" contém "drop")
            pattern = rf'\b{keyword}\b'
            if re.search(pattern, query_upper):
                return False, f"❌ Comando proibido detectado: {keyword}"
        
        return True, "Nenhum comando proibido"
    
    def _validate_select_only(self, query: str) -> Tuple[bool, str]:
        """Valida que query é apenas SELECT ou CTE (WITH)"""
        query_stripped = query.strip().upper()
        
        if query_stripped.startswith("SELECT"):
            return True, "Query SELECT válida"
        elif query_stripped.startswith("WITH"):
            # CTEs são permitidas se terminam em SELECT
            if "SELECT" in query_stripped:
                return True, "Query CTE válida"
            else:
                return False, "❌ CTE sem SELECT final"
        else:
            return False, "❌ Apenas queries SELECT ou CTE (WITH) são permitidas"
    
    def _check_allowed_tables(self, query: str) -> Tuple[bool, str]:
        """Verifica se query acessa apenas tabelas permitidas"""
        query_lower = query.lower()
        
        # Verificar se referencia pelo menos uma tabela permitida
        has_allowed_table = any(
            table.lower() in query_lower for table in self.ALLOWED_TABLES
        )
        
        if not has_allowed_table:
            return False, "❌ Query não referencia tabelas Gold permitidas"
        
        # Verificar schemas suspeitos (bronze, silver)
        forbidden_schemas = ["bronze", "silver", "raw"]
        for schema in forbidden_schemas:
            if f"{schema}." in query_lower:
                return False, f"❌ Acesso a schema não permitido: {schema}"
        
        return True, "Tabelas permitidas"
    
    def _validate_limit_clause(self, query: str) -> Tuple[bool, str]:
        """Valida presença e valor da cláusula LIMIT"""
        query_upper = query.upper()
        
        # Verificar presença de LIMIT ou TOP
        if "LIMIT" not in query_upper and "TOP" not in query_upper:
            return False, "❌ Query deve incluir cláusula LIMIT ou TOP"
        
        # Extrair e validar valor do LIMIT
        limit_match = re.search(r'LIMIT\s+(\d+)', query_upper)
        if limit_match:
            limit_value = int(limit_match.group(1))
            if limit_value > self.config.max_limit_value:
                return False, f"❌ LIMIT muito alto ({limit_value}). Máximo: {self.config.max_limit_value}"
        
        return True, "LIMIT válido"
    
    def _fail_validation(
        self,
        violation_type: str,
        message: str,
        severity: ViolationSeverity,
        query: str
    ) -> Tuple[bool, str]:
        """Registra falha de validação"""
        violation = ViolationRecord(
            timestamp=datetime.now(),
            violation_type=violation_type,
            severity=severity,
            details={"message": message},
            query=query[:200],  # Truncar query
            blocked=True
        )
        
        self.violations.append(violation)
        
        return False, message
    
    # =========================================================================
    # SANITIZAÇÃO DE PII
    # =========================================================================
    
    def sanitize_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sanitiza resultados removendo PII
        
        Args:
            df: DataFrame com resultados
            
        Returns:
            DataFrame sanitizado
        """
        if not self.config.enable_pii_sanitization:
            return df
        
        df_clean = df.copy()
        
        # 1. Remover colunas com PII
        pii_cols_found = [col for col in self.PII_COLUMNS if col in df_clean.columns]
        if pii_cols_found:
            df_clean = df_clean.drop(columns=pii_cols_found)
            
            if self.config.log_pii_detections:
                self._log_pii_detection("COLUMNS_REMOVED", pii_cols_found)
        
        # 2. Sanitizar strings com padrões de PII
        for col in df_clean.select_dtypes(include=['object']).columns:
            df_clean[col] = df_clean[col].apply(lambda x: self._sanitize_string(str(x)))
        
        return df_clean
    
    def _sanitize_string(self, text: str) -> str:
        """Sanitiza uma string removendo PII"""
        sanitized = text
        
        for pattern, pii_type in self._compiled_pii_patterns:
            if pattern.search(sanitized):
                sanitized = pattern.sub(
                    f"{self.config.pii_replacement_token}_{pii_type}",
                    sanitized
                )
                
                if self.config.log_pii_detections:
                    self._log_pii_detection("PATTERN_MATCHED", [pii_type])
        
        return sanitized
    
    def detect_pii_in_query(self, query: str) -> List[str]:
        """Detecta referências a PII em query"""
        pii_found = []
        
        query_lower = query.lower()
        
        for col in self.PII_COLUMNS:
            if col.lower() in query_lower:
                pii_found.append(col)
        
        return pii_found
    
    def _log_pii_detection(self, detection_type: str, details: List):
        """Registra detecção de PII"""
        violation = ViolationRecord(
            timestamp=datetime.now(),
            violation_type=f"PII_{detection_type}",
            severity=ViolationSeverity.LOW,
            details={"pii_items": details},
            blocked=False
        )
        
        self.violations.append(violation)
    
    # =========================================================================
    # VALIDAÇÃO DE INPUT DO USUÁRIO
    # =========================================================================
    
    @staticmethod
    def validate_user_input(user_input: str) -> Tuple[bool, str]:
        """
        Valida input livre do usuário para queries dinâmicas
        
        Args:
            user_input: Input do usuário
            
        Returns:
            Tuple (is_valid, message)
        """
        # Padrões suspeitos em input de usuário
        dangerous_patterns = [
            r';\s*DROP',
            r';\s*DELETE',
            r'--',
            r'/\*',
            r'\*/',
            r'xp_',
            r'sp_',
            r'exec\s*\(',
            r'\'.*or.*\'.*=.*\'',  # ' OR '' = '
        ]
        
        input_upper = user_input.upper()
        
        for pattern in dangerous_patterns:
            if re.search(pattern, input_upper, re.IGNORECASE):
                return False, f"❌ Padrão suspeito detectado no input"
        
        # Verificar caracteres especiais excessivos
        special_chars = len(re.findall(r'[;\-\'\"\\]', user_input))
        if special_chars > 5:
            return False, "❌ Muitos caracteres especiais no input"
        
        return True, "✅ Input validado"
    
    # =========================================================================
    # RELATÓRIOS E ESTATÍSTICAS
    # =========================================================================
    
    def get_violations_summary(self) -> Dict:
        """Retorna resumo de violações detectadas"""
        if not self.violations:
            return {
                "total_violations": 0,
                "by_type": {},
                "by_severity": {}
            }
        
        by_type = defaultdict(int)
        by_severity = defaultdict(int)
        
        for v in self.violations:
            by_type[v.violation_type] += 1
            by_severity[v.severity.value] += 1
        
        return {
            "total_violations": len(self.violations),
            "by_type": dict(by_type),
            "by_severity": dict(by_severity),
            "recent_violations": [
                {
                    "timestamp": v.timestamp.isoformat(),
                    "type": v.violation_type,
                    "severity": v.severity.value,
                    "blocked": v.blocked
                }
                for v in self.violations[-10:]  # Últimas 10
            ]
        }
    
    def get_critical_violations(self) -> List[ViolationRecord]:
        """Retorna violações críticas"""
        return [
            v for v in self.violations
            if v.severity == ViolationSeverity.CRITICAL
        ]
    
    def clear_old_violations(self, days: int = 7):
        """Remove violações antigas"""
        cutoff = datetime.now() - timedelta(days=days)
        self.violations = [v for v in self.violations if v.timestamp > cutoff]


# =============================================================================
# RATE LIMITER
# =============================================================================

class RateLimiter:
    """Controle de taxa de requisições"""
    
    def __init__(self, max_per_minute: int = 100, max_per_hour: int = 500):
        self.max_per_minute = max_per_minute
        self.max_per_hour = max_per_hour
        self._minute_requests: List[datetime] = []
        self._hour_requests: List[datetime] = []
    
    def check_limit(self) -> Tuple[bool, str]:
        """Verifica se requisição está dentro do limite"""
        now = datetime.now()
        
        # Limpar requisições antigas
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        
        self._minute_requests = [r for r in self._minute_requests if r > minute_ago]
        self._hour_requests = [r for r in self._hour_requests if r > hour_ago]
        
        # Verificar limites
        if len(self._minute_requests) >= self.max_per_minute:
            return False, f"❌ Limite de {self.max_per_minute} queries/minuto excedido"
        
        if len(self._hour_requests) >= self.max_per_hour:
            return False, f"❌ Limite de {self.max_per_hour} queries/hora excedido"
        
        # Registrar requisição
        self._minute_requests.append(now)
        self._hour_requests.append(now)
        
        return True, "✅ Dentro do limite"
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas de uso"""
        return {
            "requests_last_minute": len(self._minute_requests),
            "requests_last_hour": len(self._hour_requests),
            "minute_limit": self.max_per_minute,
            "hour_limit": self.max_per_hour,
            "minute_usage_pct": len(self._minute_requests) / self.max_per_minute * 100,
            "hour_usage_pct": len(self._hour_requests) / self.max_per_hour * 100
        }
