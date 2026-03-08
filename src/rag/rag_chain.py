"""
RAG Chain — Retrieval Augmented Generation para SRAG
=====================================================

Responsabilidade: integrar o retrieval semântico do SRAGRetriever com a geração
de respostas pelo LLM, produzindo respostas factuais baseadas exclusivamente nos
documentos recuperados das tabelas Gold.

Pipeline principal
------------------
    Query → SRAGRetriever → ContextBuilder → LLM → ResponseValidator → Dict

Decisões de design
------------------
BaseChatModel como contrato de llm — sem default ChatOpenAI
    O contrato correto é llm: BaseChatModel obrigatório — qualquer implementação
    funciona (ChatOpenAI, ChatDatabricks, ChatAnthropic). A validação None-guard
    no construtor existe como defesa em runtime; Python não enforce tipos em
    tempo de importação.

semantic_type_override em invoke()
    O IntentRouter calcula rag_semantic_type no routing decision, mas esse valor
    morria no orchestrator sem ser repassado ao retriever — cada chamada usava
    a detecção interna do SRAGRetriever independentemente do contexto semântico
    já disponível. O parâmetro semantic_type_override fecha esse ciclo: o
    orchestrator extrai rag_semantic_type do RoutingDecision e o passa para
    invoke(), que o injeta no inputs dict consumido por retrieve_and_format.
    Isso conecta as duas camadas sem acoplamento direto — SRAGChain não importa
    IntentRouter, apenas recebe o valor já calculado.

    A injeção via inputs dict (em vez de atributo de instância) é thread-safe
    e compatível com a estrutura RunnablePassthrough | retrieve_and_format |
    RunnableParallel existente — RunnablePassthrough propaga todas as chaves
    do dict de entrada, e retrieve_and_format lê semantic_type_override com
    .get(), mantendo retrocompatibilidade com chamadas sem o parâmetro.

duration_seconds em invoke()
    A versão anterior não rastreava a latência do pipeline RAG. O orchestrator
    não tinha como registrar o tempo real de retrieval + geração no log_event()
    de auditoria — apenas o tempo total do nó LangGraph estava disponível.
    invoke() agora captura t0 = time.perf_counter() antes da chain e inclui
    duration_seconds na chave "metadata" do retorno. O orchestrator pode passar
    esse valor diretamente em audit_logger.log_event(..., duration_seconds=...).

ResponseValidator integrado em invoke()
    O validator existia como classe isolada mas nunca era chamado. O resultado
    de invoke() não tinha a chave "validation", que o orchestrator.py e o
    report_generator.py esperam para avaliar qualidade da resposta RAG. A
    integração é feita após a execução da chain — não bloqueia a resposta,
    apenas adiciona diagnóstico e loga aviso quando is_valid=False.

QueryAnalyzer removido
    O IntentRouter em intent_router.py tem 9 intents, suporte a classificação
    via LLM e extração de ChartParams. Manter QueryAnalyzer criava duas fontes
    de verdade — a mesma query podia receber intents divergentes, gerando
    roteamento e retrieval com lógicas inconsistentes. Classificação de intenção
    é responsabilidade exclusiva do IntentRouter.

RAGConfig sem parâmetros de LLM
    Com llm obrigatório e injetado externamente, campos como llm_model e
    llm_temperature ficaram sem uso. Mantê-los criava dois lugares para
    configurar o mesmo comportamento e induzia ao erro de modificar RAGConfig
    esperando mudar o LLM. A configuração do LLM é responsabilidade do chamador.

min_relevance_score e enable_temporal_filter em RAGConfig
    Reservados para versão futura. min_relevance_score requer que
    SRAGRetriever.retrieve() retorne List[Tuple[Document, float]] em vez de
    List[Document] para que a chain possa filtrar por limiar antes de montar
    o contexto. enable_temporal_filter nunca teve implementação.

stream() como NotImplementedError
    A implementação anterior chamava invoke() completo e retornava com yield —
    não era streaming real. Streaming genuíno exige refatoração assíncrona via
    astream / astream_events do pipeline retrieve_and_format, incompatível com
    a estrutura síncrona atual. NotImplementedError com mensagem clara é
    preferível a falsa promessa de resposta incremental.

_contextualize_question trunca respostas anteriores em 200 chars
    Incluir respostas completas no histórico pode exceder o context window do
    LLM quando o histórico cresce. 200 chars por resposta preserva o essencial
    sem comprometer o limite de tokens. Apenas as últimas 3 interações são
    usadas — janela suficiente para continuidade conversacional.
"""

import time
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from src.rag.vector_store import SRAGRetriever


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

@dataclass
class RAGConfig:
    """
    Parâmetros do pipeline RAG para o SRAG.

    top_k
        Número de documentos recuperados pelo SRAGRetriever. Com BGE Large
        (1024d) e documentos médios de 400 chars, top_k=5 produz contexto de
        ~2000 chars, bem abaixo do limite max_context_length=8000.

    retrieval_strategy
        Estratégia passada ao SRAGRetriever.retrieve(). Valores válidos:
        'semantic', 'hybrid', 'typed'. 'hybrid' aplica reranking heurístico
        por fonte primária, recência e match de metadados sobre o pool vetorial,
        produzindo resultados mais relevantes para queries mistas.

    min_relevance_score
        Limiar de score mínimo para descarte de documentos.
        RESERVADO — não utilizado pela chain atual. Implementar filtragem por
        score requer que SRAGRetriever.retrieve() retorne
        List[Tuple[Document, float]] em vez de List[Document].

    use_citations
        Quando True, ContextBuilder.build_context_with_citations() numera cada
        documento como [1], [2]... O dict de citações é incluído no retorno de
        invoke() para consumo downstream (relatório, auditoria de fontes).

    enable_temporal_filter
        RESERVADO — não utilizado pela chain atual. Intenção original era filtrar
        documentos fora de uma janela temporal. Requer metadado de data
        normalizado em todos os documentos.

    max_context_length
        Limite em caracteres do contexto montado pelo ContextBuilder. Documentos
        são adicionados em ordem de relevância até atingir o limite — o último
        documento que excederia o limite é descartado inteiro, sem truncamento
        parcial, para não quebrar a semântica do texto.
    """
    top_k:                  int   = 5
    retrieval_strategy:     str   = "hybrid"
    min_relevance_score:    float = 0.7       # reservado — ver docstring
    use_citations:          bool  = True
    language:               str   = "pt-BR"
    enable_temporal_filter: bool  = True      # reservado — ver docstring
    max_context_length:     int   = 8000


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

SYSTEM_PROMPT_TEMPLATE = """Você é um analista epidemiológico especializado em SRAG \
(Síndrome Respiratória Aguda Grave) no Brasil.

Sua função é responder perguntas sobre dados epidemiológicos com base no contexto fornecido.

DIRETRIZES:

1. Use APENAS informações do contexto fornecido.
   - Não invente dados ou estatísticas.
   - Se não houver informação suficiente, diga claramente.

2. Seja preciso com números.
   - Cite valores exatos quando disponíveis.
   - Use formatação adequada (ex: 1.234 casos, 12,5%).

3. Contextualize as respostas.
   - Explique o significado epidemiológico.
   - Relacione com saúde pública quando relevante.

4. Mantenha tom profissional.
   - Linguagem técnica mas acessível.
   - Evite alarmismo.

5. Cite as fontes quando solicitado.
   - Mencione de qual métrica ou período veio a informação.

ESCOPO DE CONHECIMENTO:
- Dados de SRAG no Brasil
- Métricas epidemiológicas: casos, mortalidade, UTI, vacinação
- Distribuições geográficas por UF
- Perfis demográficos: idade, sexo
- Tendências temporais"""

RAG_PROMPT_TEMPLATE = """Baseando-se no contexto abaixo, responda a pergunta do usuário.

CONTEXTO:
{context}

PERGUNTA:
{question}

RESPOSTA:"""


# =============================================================================
# CONTEXT BUILDER
# =============================================================================

class ContextBuilder:
    """
    Monta o contexto textual enviado ao LLM a partir dos documentos recuperados.

    Dois modos de construção
    ------------------------
    build_context
        Formato livre com cabeçalho de metadados por documento. Usado quando
        use_citations=False — o LLM não precisa referenciar fontes numeradas.

    build_context_with_citations
        Cada documento recebe um identificador [N] no início. O dict de citações
        retornado mapeia "[N]" → {source_table, semantic_type, doc_id}, permitindo
        rastrear quais fontes embasam cada parte da resposta no report_generator.

    Comportamento de truncamento
        Documentos são adicionados em ordem de relevância até que o próximo
        excederia max_length. O documento que excede é descartado inteiro —
        truncamento parcial produziria contexto incoerente para o LLM.
    """

    @staticmethod
    def build_context(
        documents:        List[Document],
        max_length:       int  = 8000,
        include_metadata: bool = True,
    ) -> str:
        """
        Constrói contexto formatado sem numeração de citações.

        Parâmetros
        ----------
        documents
            Documentos em ordem de relevância decrescente.
        max_length
            Limite em caracteres. O documento que excederia o limite é
            descartado inteiro — não truncado.
        include_metadata
            Quando True, adiciona cabeçalho com source_table, semantic_type,
            ano_mes (se presente) e uf (se presente) antes do conteúdo.
        """
        context_parts: List[str] = []
        current_length = 0

        for idx, doc in enumerate(documents, 1):
            part = f"DOCUMENTO {idx}:\n"

            if include_metadata:
                meta  = doc.metadata
                part += f"Fonte: {meta.get('source_table',  'N/A')}\n"
                part += f"Tipo:  {meta.get('semantic_type', 'N/A')}\n"
                if "ano_mes" in meta:
                    part += f"Período: {meta['ano_mes']}\n"
                if "uf" in meta:
                    part += f"Estado: {meta['uf']}\n"
                part += "\n"

            part += doc.page_content + "\n\n---\n\n"

            if current_length + len(part) > max_length:
                break

            context_parts.append(part)
            current_length += len(part)

        return "".join(context_parts)

    @staticmethod
    def build_context_with_citations(
        documents:  List[Document],
        max_length: int = 8000,
    ) -> Tuple[str, Dict]:
        """
        Constrói contexto com identificadores de citação numerados.

        Cada documento recebe o prefixo "[N]" no início do texto. O dict
        de citações retornado permite rastrear qual source_table e doc_id
        embasam cada parte da resposta — consumido pelo report_generator
        para popular a seção de fontes do relatório.

        Retorno
        -------
        Tupla (contexto_str, citations) onde citations mapeia:
            "[N]" → {"source_table": str, "semantic_type": str, "doc_id": str}
        """
        context_parts: List[str] = []
        citations:     Dict      = {}
        current_length = 0

        for idx, doc in enumerate(documents, 1):
            citation_id = f"[{idx}]"

            citations[citation_id] = {
                "source_table":  doc.metadata.get("source_table",  "N/A"),
                "semantic_type": doc.metadata.get("semantic_type", "N/A"),
                "doc_id":        doc.metadata.get("doc_id",        "N/A"),
            }

            part = f"{citation_id} {doc.page_content}\n\n"

            if current_length + len(part) > max_length:
                break

            context_parts.append(part)
            current_length += len(part)

        return "".join(context_parts), citations


# =============================================================================
# RESPONSE VALIDATOR
# =============================================================================

class ResponseValidator:
    """
    Valida a qualidade da resposta produzida por SRAGChain.invoke().

    Integração com invoke()
        O validator é chamado ao final de invoke() — após a geração da resposta,
        não antes. O pipeline nunca é bloqueado por falha de validação: a resposta
        é sempre retornada, com o diagnóstico na chave "validation". Quando
        is_valid=False, um aviso é emitido via print para rastreabilidade sem
        depender do AuditLogger, que é opcional nesta camada.

    quality_score
        Calculado como 1.0 - (n_issues * 0.3), com mínimo 0.0. O coeficiente 0.3
        é calibrado para que uma resposta com um único issue ainda pontue 0.7,
        sinalizando problema sem descartar a resposta para uso downstream.

    GENERIC_PHRASES
        Detecta respostas que ignoram o contexto disponível. Se o LLM afirma
        não ter informação quando há documentos recuperados, é sinal de falha
        de prompt ou contexto corrompido — não de ausência real de dados.
    """

    GENERIC_PHRASES = [
        "não tenho informação",
        "não posso responder",
        "não há dados",
    ]

    @staticmethod
    def validate(response: Dict) -> Dict:
        """
        Valida a resposta e retorna dict de diagnóstico.

        Checks executados
        -----------------
        1. Resposta não vazia e com pelo menos 20 caracteres.
        2. Ao menos um documento foi recuperado pelo retriever.
        3. Resposta não contém frases genéricas quando há fontes disponíveis.

        Retorno
        -------
        Dict com:
            is_valid      : bool — True apenas se nenhum issue foi detectado.
            issues        : List[str] — descrição de cada problema encontrado.
            quality_score : float em [0.0, 1.0].
            num_sources   : int — documentos recuperados.
        """
        answer  = response.get("answer", "")
        sources = response.get("source_documents", [])
        issues: List[str] = []

        if not answer or len(answer) < 20:
            issues.append("Resposta muito curta ou vazia.")

        if not sources:
            issues.append("Nenhum documento recuperado pelo retriever.")

        if sources and any(p in answer.lower() for p in ResponseValidator.GENERIC_PHRASES):
            issues.append("Resposta genérica apesar de fontes disponíveis no contexto.")

        return {
            "is_valid":      len(issues) == 0,
            "issues":        issues,
            "quality_score": max(0.0, round(1.0 - len(issues) * 0.3, 2)),
            "num_sources":   len(sources),
        }


# =============================================================================
# SRAG CHAIN PRINCIPAL
# =============================================================================

class SRAGChain:
    """
    Chain RAG completa para o pipeline SRAG.

    Aceita qualquer BaseChatModel — ChatOpenAI, ChatDatabricks, ChatAnthropic, etc.
    O provider é injetado externamente; SRAGChain não instancia LLM por conta própria.

    Pipeline de execução
    --------------------
        invoke(question, semantic_type_override)
            → retrieve_and_format()   — SRAGRetriever recupera top-k documentos
                                        com semantic_type_override opcional e
                                        ContextBuilder monta o contexto
            → RunnableParallel        — prompt | llm | StrOutputParser() em paralelo
                                        com extração de context, citations, source_documents
            → ResponseValidator       — valida qualidade e adiciona chave "validation"
            → retorno Dict            — answer, context, source_documents,
                                        citations, validation, metadata

    Estrutura do retorno de invoke()
    ---------------------------------
        answer           : str            — resposta em linguagem natural
        context          : str            — contexto completo enviado ao LLM
        source_documents : List[Document] — documentos recuperados pelo retriever
        citations        : Dict           — mapa "[N]" → fonte (quando use_citations=True)
        validation       : Dict           — resultado do ResponseValidator
        metadata         : Dict           — question, num_sources, strategy,
                                           semantic_type_override, duration_seconds,
                                           timestamp

    semantic_type_override
        Valor calculado pelo IntentRouter no RoutingDecision e injetado pelo
        orchestrator em invoke(). Quando fornecido, substitui a detecção interna
        do SRAGRetriever._typed_retrieve(), conectando o routing decision ao
        filtro de busca vetorial sem duplicar lógica de classificação. O campo
        semantic_type_override em "metadata" permite rastrear nos logs de auditoria
        qual tipo semântico foi aplicado em cada invocação.

    duration_seconds em metadata
        Tempo de execução do pipeline completo (retrieval + geração + validação)
        em segundos, medido com time.perf_counter(). O orchestrator pode passar
        esse valor diretamente em audit_logger.log_event(..., duration_seconds=...)
        sem precisar cronometrar o nó externamente.

    Parâmetros do construtor
    ------------------------
    retriever
        SRAGRetriever já configurado com DatabricksVectorStoreManager.
    llm
        Qualquer BaseChatModel. Obrigatório — não há default para evitar
        instanciação implícita de ChatOpenAI sem credenciais configuradas.
    config
        RAGConfig com parâmetros de retrieval e contexto. Quando None,
        usa defaults do dataclass (top_k=5, strategy='hybrid').
    """

    def __init__(
        self,
        retriever: SRAGRetriever,
        llm:       BaseChatModel,
        config:    Optional[RAGConfig] = None,
    ):
        if llm is None:
            raise ValueError(
                "llm é obrigatório em SRAGChain. "
                "Passe ChatOpenAI, ChatDatabricks ou qualquer outro BaseChatModel."
            )

        self.retriever       = retriever
        self.llm             = llm
        self.config          = config or RAGConfig()
        self.context_builder = ContextBuilder()
        self.chain           = self._build_chain()

    def _build_chain(self):
        """
        Monta a chain LangChain com RunnablePassthrough e RunnableParallel.

        Estrutura
        ---------
        RunnablePassthrough
            Propaga o dict de entrada sem modificação para retrieve_and_format.
            Todas as chaves presentes — incluindo semantic_type_override — são
            preservadas e ficam disponíveis para a closure retrieve_and_format.

        retrieve_and_format (closure)
            Lê question e semantic_type_override do inputs dict. Chama
            SRAGRetriever.retrieve() com o override quando fornecido e
            ContextBuilder para produzir {context, question, source_documents,
            citations}.

        RunnableParallel
            Executa prompt | llm | StrOutputParser() em paralelo com lambdas
            que extraem context, source_documents e citations do dict intermediário.
            O paralelo evita múltiplas chamadas ao LLM — tudo parte do mesmo dict.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_TEMPLATE),
            ("human",  RAG_PROMPT_TEMPLATE),
        ])

        def retrieve_and_format(inputs: Dict) -> Dict:
            question               = inputs["question"]
            semantic_type_override = inputs.get("semantic_type_override")

            docs = self.retriever.retrieve(
                question,
                k                      = self.config.top_k,
                strategy               = self.config.retrieval_strategy,
                semantic_type_override = semantic_type_override,
            )

            if self.config.use_citations:
                context, citations = self.context_builder.build_context_with_citations(
                    docs, max_length=self.config.max_context_length,
                )
                return {
                    "context":          context,
                    "question":         question,
                    "citations":        citations,
                    "source_documents": docs,
                }

            context = self.context_builder.build_context(
                docs, max_length=self.config.max_context_length,
            )
            return {
                "context":          context,
                "question":         question,
                "source_documents": docs,
            }

        return (
            RunnablePassthrough()
            | retrieve_and_format
            | RunnableParallel({
                "answer":           prompt | self.llm | StrOutputParser(),
                "context":          lambda x: x["context"],
                "question":         lambda x: x["question"],
                "source_documents": lambda x: x.get("source_documents", []),
                "citations":        lambda x: x.get("citations", {}),
            })
        )

    def invoke(
        self,
        question:               str,
        semantic_type_override: Optional[str] = None,
    ) -> Dict:
        """
        Executa o pipeline RAG completo e valida a resposta produzida.

        Parâmetros
        ----------
        question
            Pergunta do usuário em linguagem natural.
        semantic_type_override
            Tipo semântico calculado pelo IntentRouter e repassado pelo
            orchestrator. Quando fornecido, substitui a detecção interna do
            SRAGRetriever para o filtro de busca vetorial. Deve ser um dos
            tipos indexados: kpi, regra, temporal, geographic, demographic.
            Quando None, o SRAGRetriever usa sua detecção interna.

        Retorno
        -------
        Dict com chaves: answer, context, source_documents, citations,
        validation, metadata. A chave "validation" sempre está presente —
        o orchestrator e o report_generator dependem dela para avaliar
        qualidade antes de incluir a resposta no relatório.
        O campo metadata["duration_seconds"] contém a latência total do
        pipeline para registro em audit_logger.log_event().
        """
        t0  = time.perf_counter()
        raw = self.chain.invoke({
            "question":               question,
            "semantic_type_override": semantic_type_override,
        })
        duration = round(time.perf_counter() - t0, 3)

        result = {
            "answer":           raw["answer"],
            "context":          raw["context"],
            "source_documents": raw["source_documents"],
            "citations":        raw.get("citations", {}),
            "metadata": {
                "question":               question,
                "num_sources":            len(raw["source_documents"]),
                "retrieval_strategy":     self.config.retrieval_strategy,
                "semantic_type_override": semantic_type_override,
                "duration_seconds":       duration,
                "timestamp":              datetime.now().isoformat(),
            },
        }

        validation = ResponseValidator.validate(result)
        result["validation"] = validation

        if not validation["is_valid"]:
            print(
                f"[rag_chain] aviso de qualidade — "
                f"score={validation['quality_score']:.2f} | "
                f"issues={validation['issues']}"
            )

        return result

    def stream(self, question: str):
        """
        Streaming não implementado — levanta NotImplementedError.

        Streaming real requer suporte assíncrono do provider via astream() ou
        astream_events() e refatoração de retrieve_and_format para não bloquear
        enquanto os documentos não são todos recuperados. A estrutura síncrona
        atual com RunnableParallel é incompatível com entrega incremental de tokens.

        Use invoke() para resposta síncrona completa.
        """
        raise NotImplementedError(
            "SRAGChain.stream() não está implementado. "
            "Use SRAGChain.invoke() para obter a resposta completa. "
            "Streaming real requer refatoração assíncrona do pipeline RAG."
        )

    def __repr__(self) -> str:
        return (
            f"SRAGChain("
            f"strategy={self.config.retrieval_strategy}, "
            f"top_k={self.config.top_k}, "
            f"llm={type(self.llm).__name__})"
        )


# =============================================================================
# CONVERSATIONAL SRAG CHAIN
# =============================================================================

class ConversationalSRAGChain:
    """
    RAG Chain com memória de conversa baseada em histórico em memória.

    Aceita qualquer BaseChatModel — mesmo contrato de SRAGChain.

    Estratégia de contextualização
    --------------------------------
    O histórico das últimas 3 interações é concatenado como prefixo da nova
    pergunta antes de invocar SRAGChain. Respostas anteriores são truncadas
    em 200 chars para evitar que o contexto histórico consuma espaço excessivo
    no prompt.

    Limitação atual vs MessagesPlaceholder
        Esta implementação serializa o histórico como string, adequada para
        uso isolado fora do LangGraph. A abordagem recomendada para o agent/
        é MessagesPlaceholder com checkpoint de estado no LangGraph, que preserva
        o histórico tipado e permite retomada de sessão entre execuções.

    semantic_type_override propagado
        invoke() aceita e repassa semantic_type_override para SRAGChain.invoke(),
        mantendo o mesmo contrato da chain base e permitindo que o orchestrator
        use ConversationalSRAGChain com roteamento semântico sem distinção de tipo.

    Parâmetros
    ----------
    retriever
        SRAGRetriever já configurado.
    llm
        BaseChatModel obrigatório — mesmo contrato de SRAGChain.
    config
        RAGConfig opcional — repassado para SRAGChain base.
    """

    def __init__(
        self,
        retriever: SRAGRetriever,
        llm:       BaseChatModel,
        config:    Optional[RAGConfig] = None,
    ):
        if llm is None:
            raise ValueError(
                "llm é obrigatório em ConversationalSRAGChain. "
                "Passe ChatOpenAI, ChatDatabricks ou qualquer outro BaseChatModel."
            )

        self.base_chain            = SRAGChain(retriever, llm, config)
        self.conversation_history: List[Dict] = []

    def invoke(
        self,
        question:               str,
        semantic_type_override: Optional[str] = None,
    ) -> Dict:
        """
        Invoca a chain com contexto das interações anteriores.

        A pergunta atual é prefixada com o histórico das últimas 3 interações
        antes de ser enviada ao SRAGChain. O retorno é idêntico ao de
        SRAGChain.invoke() — a contextualização é transparente para o chamador.
        semantic_type_override é repassado intacto para a chain base.
        """
        contextualized = (
            self._contextualize_question(question)
            if self.conversation_history
            else question
        )

        result = self.base_chain.invoke(
            contextualized,
            semantic_type_override=semantic_type_override,
        )

        self.conversation_history.append({
            "question":  question,
            "answer":    result["answer"],
            "timestamp": datetime.now(),
        })

        return result

    def _contextualize_question(self, question: str) -> str:
        """
        Monta a pergunta contextualizada com o histórico recente.

        Usa as últimas 3 interações — janela suficiente para continuidade
        conversacional sem overhead de contexto longo. Respostas truncadas
        em 200 chars para não consumir espaço excessivo no prompt.
        """
        recent  = self.conversation_history[-3:]
        history = "Histórico recente:\n"
        for item in recent:
            history += f"P: {item['question']}\nR: {item['answer'][:200]}...\n\n"
        return f"{history}\nPergunta atual: {question}"

    def clear_history(self) -> None:
        """Limpa o histórico de conversa em memória."""
        self.conversation_history.clear()

    @property
    def history_length(self) -> int:
        """Número de interações no histórico atual."""
        return len(self.conversation_history)

    def __repr__(self) -> str:
        return (
            f"ConversationalSRAGChain("
            f"history={self.history_length}, "
            f"base={self.base_chain!r})"
        )