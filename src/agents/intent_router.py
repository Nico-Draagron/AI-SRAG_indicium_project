"""
Intent Router - Roteamento Inteligente SQL vs RAG
==================================================

Decide a estratégia de resposta baseado na intenção da query:
- SQL direto: Perguntas factuais ("Quantos casos em SP?")
- RAG: Perguntas analíticas ("Por que a mortalidade aumentou?")
- Híbrido: Perguntas mistas ("Quais UFs tiveram aumento e por quê?")

Author: AI Engineer Certification - Indicium
Date: January 2025
Version: 1.0.0
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


class QueryIntent(Enum):
    """Tipos de intenção de query"""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    TEMPORAL = "temporal"
    GEOGRAPHIC = "geographic"
    DEMOGRAPHIC = "demographic"
    EXPLANATORY = "explanatory"
    MIXED = "mixed"


class ExecutionStrategy(Enum):
    """Estratégia de execução"""
    SQL_ONLY = "sql_only"
    RAG_ONLY = "rag_only"
    HYBRID = "hybrid"


@dataclass
class RoutingDecision:
    """Decisão de roteamento"""
    intent: QueryIntent
    strategy: ExecutionStrategy
    confidence: float
    reasoning: str
    target_tables: List[str]
    sql_filters: Optional[Dict] = None
    rag_semantic_type: Optional[str] = None
    requires_synthesis: bool = True


class IntentClassifier:
    """Classificador de intenção de queries"""
    
    PATTERNS = {
        QueryIntent.FACTUAL: [
            r'\b(quantos?|qual|quanto|quem)\b',
            r'\b(total|número|quantidade)\b',
            r'\b(casos|óbitos|mortes)\b.*\b(em|de)\b'
        ],
        QueryIntent.ANALYTICAL: [
            r'\b(por que|porque|motivo|razão|causa)\b',
            r'\b(analis[ae]|avaliar|explicar|entender)\b',
            r'\b(impacto|efeito|consequência)\b'
        ],
        QueryIntent.COMPARATIVE: [
            r'\b(comparar|versus|vs|diferença)\b',
            r'\b(maior|menor|melhor|pior)\b.*\b(que|do que)\b',
            r'\b(entre|e)\b.*\b(estados?|UFs?)\b'
        ],
        QueryIntent.TEMPORAL: [
            r'\b(tendência|evolução|crescimento)\b',
            r'\b(últimos?|próximos?|passados?)\b.*\b(meses?|anos?)\b',
            r'\b(temporal|ao longo|série)\b'
        ],
        QueryIntent.GEOGRAPHIC: [
            r'\b(estado|UF|região|mapa)\b',
            r'\b(ranking|top|principais)\b.*\b(UFs?|estados?)\b',
            r'\b(SP|RJ|MG|BA|RS|PR|SC|CE|PE)\b'
        ],
        QueryIntent.DEMOGRAPHIC: [
            r'\b(idade|idoso|criança|adulto)\b',
            r'\b(sexo|feminino|masculino|gênero)\b',
            r'\b(faixa etária|grupo etário)\b'
        ],
        QueryIntent.EXPLANATORY: [
            r'\b(o que é|como|defin[ae])\b',
            r'\b(significa|conceito|explicar)\b',
            r'\b(taxa de|métrica|indicador)\b'
        ]
    }
    
    @staticmethod
    def classify(query: str) -> List[QueryIntent]:
        """Classifica intenção(ões) da query"""
        query_lower = query.lower()
        detected_intents = []
        
        for intent, patterns in IntentClassifier.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    if intent not in detected_intents:
                        detected_intents.append(intent)
                    break
        
        if not detected_intents:
            detected_intents.append(QueryIntent.FACTUAL)
        
        return detected_intents


class StrategySelector:
    """Seletor de estratégia de execução baseado em intenção"""
    
    INTENT_TO_STRATEGY = {
        QueryIntent.FACTUAL: ExecutionStrategy.SQL_ONLY,
        QueryIntent.ANALYTICAL: ExecutionStrategy.RAG_ONLY,
        QueryIntent.COMPARATIVE: ExecutionStrategy.HYBRID,
        QueryIntent.TEMPORAL: ExecutionStrategy.HYBRID,
        QueryIntent.GEOGRAPHIC: ExecutionStrategy.SQL_ONLY,
        QueryIntent.DEMOGRAPHIC: ExecutionStrategy.SQL_ONLY,
        QueryIntent.EXPLANATORY: ExecutionStrategy.RAG_ONLY,
        QueryIntent.MIXED: ExecutionStrategy.HYBRID
    }
    
    @staticmethod
    def select(intents: List[QueryIntent]) -> ExecutionStrategy:
        """Seleciona estratégia baseado em intenções"""
        if len(intents) == 0:
            return ExecutionStrategy.SQL_ONLY
        
        if len(intents) > 1:
            return ExecutionStrategy.HYBRID
        
        if QueryIntent.ANALYTICAL in intents or QueryIntent.EXPLANATORY in intents:
            return ExecutionStrategy.RAG_ONLY
        
        return StrategySelector.INTENT_TO_STRATEGY.get(
            intents[0],
            ExecutionStrategy.SQL_ONLY
        )


class IntentRouter:
    """
    Roteador principal - Decide como executar cada query
    
    Pipeline:
        Query → Classify Intent → Select Strategy → Configure Execution
    """
    
    def __init__(self, use_llm_classification: bool = False):
        self.use_llm = use_llm_classification
        self.classifier = IntentClassifier()
        self.selector = StrategySelector()
        
        if use_llm_classification:
            self.llm = ChatAnthropic(
                model="claude-3-5-haiku-20241022",
                temperature=0.0
            )
    
    def route(self, query: str) -> RoutingDecision:
        """Roteia query para estratégia apropriada"""
        # Classificar intenção
        if self.use_llm:
            intents = self._classify_with_llm(query)
        else:
            intents = self.classifier.classify(query)
        
        # Selecionar estratégia
        strategy = self.selector.select(intents)
        
        # Determinar tabelas alvo
        target_tables = self._determine_target_tables(intents, query)
        
        # Extrair filtros SQL
        sql_filters = self._extract_sql_filters(query) if strategy != ExecutionStrategy.RAG_ONLY else None
        
        # Determinar tipo semântico RAG
        rag_type = self._determine_rag_type(intents) if strategy != ExecutionStrategy.SQL_ONLY else None
        
        # Calcular confiança
        confidence = self._calculate_confidence(intents, query)
        
        # Gerar reasoning
        reasoning = self._generate_reasoning(intents, strategy, target_tables)
        
        return RoutingDecision(
            intent=intents[0] if len(intents) == 1 else QueryIntent.MIXED,
            strategy=strategy,
            confidence=confidence,
            reasoning=reasoning,
            target_tables=target_tables,
            sql_filters=sql_filters,
            rag_semantic_type=rag_type,
            requires_synthesis=strategy in [ExecutionStrategy.HYBRID]
        )
    
    def _classify_with_llm(self, query: str) -> List[QueryIntent]:
        """Classificação via LLM"""
        prompt = f"""Classifique a intenção desta query sobre SRAG:

Query: "{query}"

Intenções possíveis:
- FACTUAL: perguntas objetivas (quantos, qual, quanto)
- ANALYTICAL: perguntas sobre causas (por que, motivo)
- COMPARATIVE: comparações (maior, menor, vs)
- TEMPORAL: tendências temporais
- GEOGRAPHIC: questões geográficas
- DEMOGRAPHIC: perfil populacional
- EXPLANATORY: explicações de conceitos

Responda APENAS com as intenções detectadas, separadas por vírgula.
Exemplo: "FACTUAL, GEOGRAPHIC"
"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        intent_names = [i.strip() for i in response.content.split(",")]
        intents = []
        for name in intent_names:
            try:
                intents.append(QueryIntent[name])
            except KeyError:
                pass
        
        return intents if intents else [QueryIntent.FACTUAL]
    
    def _determine_target_tables(self, intents: List[QueryIntent], query: str) -> List[str]:
        """Determina quais tabelas Gold consultar"""
        tables = set()
        query_lower = query.lower()
        
        intent_table_map = {
            QueryIntent.TEMPORAL: ["metricas_temporais", "series_temporais"],
            QueryIntent.GEOGRAPHIC: ["metricas_geograficas"],
            QueryIntent.DEMOGRAPHIC: ["metricas_demograficas"],
            QueryIntent.FACTUAL: ["metricas_temporais"],
            QueryIntent.ANALYTICAL: ["resumo_geral"],
            QueryIntent.EXPLANATORY: ["resumo_geral"]
        }
        
        for intent in intents:
            tables.update(intent_table_map.get(intent, ["metricas_temporais"]))
        
        if any(kw in query_lower for kw in ["estado", "uf", "sp", "rj", "região"]):
            tables.add("metricas_geograficas")
        
        if any(kw in query_lower for kw in ["idade", "idoso", "sexo", "faixa"]):
            tables.add("metricas_demograficas")
        
        if any(kw in query_lower for kw in ["tendência", "evolução", "série"]):
            tables.add("series_temporais")
        
        return list(tables)
    
    def _extract_sql_filters(self, query: str) -> Optional[Dict]:
        """Extrai filtros SQL da query"""
        filters = {}
        query_lower = query.lower()
        
        # UF
        uf_pattern = r'\b(AC|AL|AP|AM|BA|CE|DF|ES|GO|MA|MT|MS|MG|PA|PB|PR|PE|PI|RJ|RN|RS|RO|RR|SC|SP|SE|TO)\b'
        uf_match = re.search(uf_pattern, query.upper())
        if uf_match:
            filters["sg_uf"] = uf_match.group(1)
        
        # Período
        month_map = {
            "janeiro": "01", "fevereiro": "02", "março": "03",
            "abril": "04", "maio": "05", "junho": "06",
            "julho": "07", "agosto": "08", "setembro": "09",
            "outubro": "10", "novembro": "11", "dezembro": "12"
        }
        
        for month_name, month_num in month_map.items():
            if month_name in query_lower:
                year_match = re.search(r'\b(20\d{2})\b', query)
                if year_match:
                    filters["ano_mes"] = f"{year_match.group(1)}-{month_num}"
                break
        
        return filters if filters else None
    
    def _determine_rag_type(self, intents: List[QueryIntent]) -> Optional[str]:
        """Determina tipo semântico para busca RAG"""
        intent_to_rag = {
            QueryIntent.TEMPORAL: "temporal",
            QueryIntent.GEOGRAPHIC: "geographic",
            QueryIntent.DEMOGRAPHIC: "demographic",
            QueryIntent.ANALYTICAL: None,
            QueryIntent.EXPLANATORY: "metric"
        }
        
        for intent in intents:
            rag_type = intent_to_rag.get(intent)
            if rag_type:
                return rag_type
        
        return None
    
    def _calculate_confidence(self, intents: List[QueryIntent], query: str) -> float:
        """Calcula confiança da classificação"""
        base_confidence = 0.7
        
        if any(kw in query.lower() for kw in ["quantos", "qual", "ranking", "total"]):
            base_confidence += 0.2
        
        if len(query.split()) < 4:
            base_confidence -= 0.1
        
        if len(intents) > 2:
            base_confidence -= 0.15
        
        return max(0.5, min(0.95, base_confidence))
    
    def _generate_reasoning(
        self,
        intents: List[QueryIntent],
        strategy: ExecutionStrategy,
        tables: List[str]
    ) -> str:
        """Gera explicação da decisão de roteamento"""
        intent_str = ", ".join([i.value for i in intents])
        tables_str = ", ".join(tables)
        
        return (
            f"Detectei intenção(ões): {intent_str}. "
            f"Estratégia selecionada: {strategy.value}. "
            f"Tabelas alvo: {tables_str}."
        )