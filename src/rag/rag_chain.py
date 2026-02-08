"""
RAG Chain - Retrieval Augmented Generation para SRAG
=====================================================

Integra retrieval semântico + geração de respostas usando LLM.

Arquitetura:
    Query → Retriever → Context → LLM → Resposta

Author: AI Engineer Certification - Indicium
Date: January 2025
Version: 2.0.0
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI

from src.rag.vector_store import SRAGRetriever


# =============================================================================
# CONFIGURAÇÕES RAG
# =============================================================================

@dataclass
class RAGConfig:
    """Configuração do sistema RAG"""
    # Retrieval
    top_k: int = 5
    retrieval_strategy: str = "hybrid"  # semantic, hybrid, typed
    min_relevance_score: float = 0.7
    
    # Generation
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    max_tokens: int = 2000
    
    # Prompts
    use_citations: bool = True
    language: str = "pt-BR"
    
    # Filtros
    enable_temporal_filter: bool = True
    max_context_length: int = 8000


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

SYSTEM_PROMPT_TEMPLATE = """Você é um analista epidemiológico especializado em SRAG (Síndrome Respiratória Aguda Grave) no Brasil.

Sua função é responder perguntas sobre dados epidemiológicos com base no contexto fornecido.

**DIRETRIZES:**

1. **Use APENAS informações do contexto fornecido**
   - Não invente dados ou estatísticas
   - Se não houver informação suficiente, diga claramente

2. **Seja preciso com números**
   - Cite valores exatos quando disponíveis
   - Use formatação adequada (ex: 1.234 casos, 12,5%)

3. **Contextualize as respostas**
   - Explique o significado epidemiológico
   - Relacione com saúde pública quando relevante

4. **Mantenha tom profissional**
   - Linguagem técnica mas acessível
   - Evite alarmismo

5. **Cite as fontes quando solicitado**
   - Mencione de qual métrica/período veio a informação

**ESCOPO DE CONHECIMENTO:**
- Dados de SRAG no Brasil
- Métricas epidemiológicas (casos, mortalidade, UTI, vacinação)
- Distribuições geográficas (por UF)
- Perfis demográficos (idade, sexo)
- Tendências temporais
"""

RAG_PROMPT_TEMPLATE = """Baseando-se no contexto abaixo, responda a pergunta do usuário.

**CONTEXTO:**
{context}

**PERGUNTA:**
{question}

**RESPOSTA:**"""


# =============================================================================
# CONTEXT BUILDER
# =============================================================================

class ContextBuilder:
    """Constrói contexto formatado a partir de documentos"""
    
    @staticmethod
    def build_context(
        documents: List[Document],
        max_length: int = 8000,
        include_metadata: bool = True
    ) -> str:
        """
        Constrói contexto formatado
        
        Args:
            documents: Documentos recuperados
            max_length: Tamanho máximo do contexto
            include_metadata: Incluir metadados
            
        Returns:
            Contexto formatado em string
        """
        context_parts = []
        current_length = 0
        
        for idx, doc in enumerate(documents, 1):
            # Construir parte do contexto
            part = f"**DOCUMENTO {idx}:**\n"
            
            if include_metadata:
                metadata = doc.metadata
                part += f"Fonte: {metadata.get('source_table', 'N/A')}\n"
                part += f"Tipo: {metadata.get('semantic_type', 'N/A')}\n"
                
                if 'ano_mes' in metadata:
                    part += f"Período: {metadata['ano_mes']}\n"
                if 'uf' in metadata:
                    part += f"Estado: {metadata['uf']}\n"
                
                part += "\n"
            
            part += doc.page_content + "\n\n---\n\n"
            
            # Verificar tamanho
            if current_length + len(part) > max_length:
                break
            
            context_parts.append(part)
            current_length += len(part)
        
        return "".join(context_parts)
    
    @staticmethod
    def build_context_with_citations(
        documents: List[Document],
        max_length: int = 8000
    ) -> Tuple[str, Dict]:
        """Constrói contexto com sistema de citações"""
        context_parts = []
        citations = {}
        current_length = 0
        
        for idx, doc in enumerate(documents, 1):
            citation_id = f"[{idx}]"
            
            # Registrar citação
            citations[citation_id] = {
                "source_table": doc.metadata.get('source_table', 'N/A'),
                "semantic_type": doc.metadata.get('semantic_type', 'N/A'),
                "doc_id": doc.metadata.get('doc_id', 'N/A')
            }
            
            # Construir parte
            part = f"{citation_id} {doc.page_content}\n\n"
            
            if current_length + len(part) > max_length:
                break
            
            context_parts.append(part)
            current_length += len(part)
        
        context = "".join(context_parts)
        return context, citations


# =============================================================================
# RAG CHAIN PRINCIPAL
# =============================================================================

class SRAGChain:
    """
    Chain RAG completa para SRAG
    
    Pipeline:
        Query → Retriever → Context Builder → LLM → Resposta
    
    Example:
        >>> chain = SRAGChain(retriever, llm)
        >>> response = chain.invoke("Quantos casos de SRAG em janeiro de 2025?")
        >>> print(response["answer"])
    """
    
    def __init__(
        self,
        retriever: SRAGRetriever,
        llm: Optional[ChatOpenAI] = None,
        config: Optional[RAGConfig] = None,
    ):
        self.retriever = retriever
        self.config = config or RAGConfig()
        self.llm = llm or ChatOpenAI(
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.max_tokens
        )
        
        self.context_builder = ContextBuilder()
        self.chain = self._build_chain()
    
    def _build_chain(self):
        """Constrói chain LangChain"""
        # Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_TEMPLATE),
            ("human", RAG_PROMPT_TEMPLATE)
        ])
        
        # Chain components
        def retrieve_and_format(inputs):
            """Retrieval + formatação de contexto"""
            question = inputs["question"]
            
            # Recuperar documentos
            docs = self.retriever.retrieve(
                question,
                k=self.config.top_k,
                strategy=self.config.retrieval_strategy
            )
            
            # Construir contexto
            if self.config.use_citations:
                context, citations = self.context_builder.build_context_with_citations(
                    docs,
                    max_length=self.config.max_context_length
                )
                return {
                    "context": context,
                    "question": question,
                    "citations": citations,
                    "source_documents": docs
                }
            else:
                context = self.context_builder.build_context(
                    docs,
                    max_length=self.config.max_context_length
                )
                return {
                    "context": context,
                    "question": question,
                    "source_documents": docs
                }
        
        # Montar chain
        chain = (
            RunnablePassthrough()
            | retrieve_and_format
            | RunnableParallel({
                "answer": prompt | self.llm | StrOutputParser(),
                "context": lambda x: x["context"],
                "question": lambda x: x["question"],
                "source_documents": lambda x: x.get("source_documents", []),
                "citations": lambda x: x.get("citations", {})
            })
        )
        
        return chain
    
    def invoke(self, question: str) -> Dict:
        """
        Executa chain RAG
        
        Args:
            question: Pergunta do usuário
            
        Returns:
            Dict com:
                - answer: Resposta gerada
                - context: Contexto usado
                - source_documents: Documentos recuperados
                - citations: Citações (se habilitado)
        """
        result = self.chain.invoke({"question": question})
        
        return {
            "answer": result["answer"],
            "context": result["context"],
            "source_documents": result["source_documents"],
            "citations": result.get("citations", {}),
            "metadata": {
                "question": question,
                "num_sources": len(result["source_documents"]),
                "retrieval_strategy": self.config.retrieval_strategy,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def stream(self, question: str):
        """Versão streaming da resposta"""
        # TODO: Implementar streaming
        result = self.invoke(question)
        yield result["answer"]


# =============================================================================
# CONVERSATIONAL RAG (com memória)
# =============================================================================

class ConversationalSRAGChain:
    """
    RAG Chain com memória de conversa
    
    Mantém histórico de perguntas/respostas para contexto
    """
    
    def __init__(
        self,
        retriever: SRAGRetriever,
        llm: Optional[ChatOpenAI] = None,
        config: Optional[RAGConfig] = None
    ):
        self.base_chain = SRAGChain(retriever, llm, config)
        self.conversation_history: List[Dict] = []
    
    def invoke(self, question: str) -> Dict:
        """Invoca com contexto de conversa"""
        # Adicionar histórico ao prompt se existir
        if self.conversation_history:
            contextualized_question = self._contextualize_question(question)
        else:
            contextualized_question = question
        
        # Executar chain
        result = self.base_chain.invoke(contextualized_question)
        
        # Salvar no histórico
        self.conversation_history.append({
            "question": question,
            "answer": result["answer"],
            "timestamp": datetime.now()
        })
        
        return result
    
    def _contextualize_question(self, question: str) -> str:
        """Adiciona contexto da conversa"""
        # Pegar últimas 3 interações
        recent = self.conversation_history[-3:]
        
        context = "Histórico recente:\n"
        for item in recent:
            context += f"P: {item['question']}\nR: {item['answer'][:100]}...\n\n"
        
        return f"{context}\nPergunta atual: {question}"
    
    def clear_history(self):
        """Limpa histórico de conversa"""
        self.conversation_history.clear()


# =============================================================================
# QUERY ANALYZER (opcional)
# =============================================================================

class QueryAnalyzer:
    """Analisa intenção da query para otimizar retrieval"""
    
    QUERY_TYPES = {
        "factual": ["quantos", "qual", "quanto", "quem"],
        "comparative": ["maior", "menor", "melhor", "pior", "comparar"],
        "temporal": ["quando", "período", "mês", "ano", "tendência"],
        "geographic": ["onde", "estado", "uf", "região"],
        "explanatory": ["por que", "como", "explicar", "motivo"]
    }
    
    @staticmethod
    def analyze(question: str) -> Dict:
        """
        Analisa tipo de pergunta
        
        Returns:
            Dict com tipo e sugestões de filtro
        """
        question_lower = question.lower()
        
        detected_types = []
        for qtype, keywords in QueryAnalyzer.QUERY_TYPES.items():
            if any(kw in question_lower for kw in keywords):
                detected_types.append(qtype)
        
        # Sugerir filtros
        filters = {}
        if "temporal" in detected_types:
            filters["prefer_semantic_type"] = "temporal"
        elif "geographic" in detected_types:
            filters["prefer_semantic_type"] = "geographic"
        
        return {
            "types": detected_types,
            "primary_type": detected_types[0] if detected_types else "general",
            "suggested_filters": filters,
            "complexity": "complex" if len(detected_types) > 1 else "simple"
        }


# =============================================================================
# RESPONSE VALIDATOR
# =============================================================================

class ResponseValidator:
    """Valida qualidade da resposta gerada"""
    
    @staticmethod
    def validate(response: Dict) -> Dict:
        """
        Valida resposta RAG
        
        Checks:
            - Resposta não vazia
            - Fontes recuperadas
            - Não há disclaimers genéricos demais
        """
        answer = response["answer"]
        sources = response["source_documents"]
        
        issues = []
        
        # Check 1: Resposta vazia
        if not answer or len(answer) < 20:
            issues.append("Resposta muito curta ou vazia")
        
        # Check 2: Sem fontes
        if not sources:
            issues.append("Nenhum documento recuperado")
        
        # Check 3: Resposta genérica demais
        generic_phrases = [
            "não tenho informação",
            "não posso responder",
            "não há dados"
        ]
        if any(phrase in answer.lower() for phrase in generic_phrases):
            if len(sources) > 0:
                issues.append("Resposta genérica apesar de ter fontes")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "quality_score": 1.0 - (len(issues) * 0.3),
            "num_sources": len(sources)
        }
