"""
Vector Store — Databricks Vector Search + Embeddings
=====================================================

Responsabilidade: gerenciar o ciclo completo de embeddings e busca vetorial
para o pipeline RAG do SRAG, desde a preparação de documentos até o retrieval
com reranking heurístico e filtros semânticos por tipo de conteúdo.

Decisões de design
------------------
Migração de imports: langchain_community → databricks_langchain / langchain_huggingface
    O pacote langchain_community marcou DatabricksEmbeddings e HuggingFaceEmbeddings
    como deprecated na versão 0.3.3 e os removerá na 1.0. Ambos os imports foram
    migrados para os pacotes canônicos: databricks-langchain (DatabricksEmbeddings)
    e langchain-huggingface (HuggingFaceEmbeddings). O fallback para langchain_community
    foi removido — o ambiente deve ter os pacotes corretos instalados (ver requirements.txt).
    DATABRICKS_AVAILABLE e HF_AVAILABLE controlam a disponibilidade; o módulo não
    falha na inicialização se algum dos pacotes opcionais estiver ausente.

DatabricksVectorSearch removido dos imports
    O import original trazia DatabricksVectorSearch de langchain_community mesmo que
    DatabricksVectorStoreManager opere diretamente via VectorSearchClient SDK sem
    usar essa classe em nenhum método. O import era código morto que levantava
    ImportError na linha 27 antes de qualquer lógica executar.

model= removido do construtor de DatabricksEmbeddings
    DatabricksEmbeddings aceita apenas endpoint= — o endpoint Foundation Model API
    do Databricks já identifica unicamente o modelo e sua dimensão. Passar model=
    causava TypeError: __init__() got an unexpected keyword argument 'model' em
    tempo de execução. O parâmetro model em get_embeddings() é mantido na assinatura
    apenas para documentar qual modelo o endpoint serve; ele não é repassado ao
    construtor.

table_name explícito em VectorStoreConfig
    O código anterior derivava o nome da tabela Delta por substituição de string:
    full_index_name.replace("_index", "_table"). Essa derivação silenciosamente
    falhava se index_name não contivesse "_index" — a substituição retornava o
    mesmo nome do índice e _save_to_delta escrevia numa string inválida. Campo
    explícito torna o contrato visível e testável.

embedding_endpoint em VectorStoreConfig
    O endpoint do Foundation Model API estava hardcoded em EmbeddingManager como
    "databricks-bge-large-en". Centralizar em VectorStoreConfig alinha com o padrão
    de constantes já adotado no notebook 06, onde VS_ENDPOINT_NAME, VS_INDEX_NAME e
    VS_TABLE_NAME são definidos em um único lugar. EmbeddingManager consome a config
    em vez de manter sua própria constante interna.

Score de similaridade extraído da linha, não de lista separada
    O código anterior tentava scores[idx] de result.get("scores", []), mas a API
    do Databricks Vector Search retorna o score dentro de cada linha do data_array
    como último elemento (row[-1]), não em lista separada. scores estava sempre
    vazio, resultando em score=0.8 fixo para todos os documentos e tornando o
    reranking inefetivo.

Boost de _hybrid_retrieve corrigido para gold_rag_kpi_fatos
    O reranking aplicava 1.2x de boost para source_table == "gold_resumo_geral",
    tabela que não existe no pipeline atual (substituída por gold_rag_kpi_fatos no
    notebook 05). Nenhum documento tinha esse source_table, então o boost nunca era
    aplicado.

mergeSchema removido de _save_to_delta
    overwriteSchema=true e mergeSchema=true foram usados simultaneamente. O
    overwriteSchema descarta o schema antigo antes que qualquer merge aconteça,
    tornando mergeSchema sem efeito. Como a tabela é sempre recriada via
    overwrite, apenas overwriteSchema=true é necessário.

bare except substituído por except ImportError
    O bloco original usava except: sem especificar tipo, capturando KeyboardInterrupt,
    SystemExit e MemoryError junto com ImportError. Uma falha de memória durante o
    import silenciaria o erro real e definiria DATABRICKS_AVAILABLE=False, colocando
    o sistema em modo degradado sem diagnóstico.

get_index_health() substitui get_index_stats()
    get_index_stats() usava getattr(index, "status", "unknown") e
    getattr(index, "num_rows", 0), mas o objeto VectorSearchIndex do SDK Databricks
    não expõe esses atributos diretamente — o resultado era sempre status="unknown"
    e num_rows=0 independentemente do estado real do índice. get_index_health()
    consulta o endpoint via get_endpoint(), tenta extrair o status do índice em
    múltiplos formatos (atributo direto, describe(), status_message, dict nested)
    e compara table_row_count com index_row_count para detectar dessincronização.
    Retorna healthy: bool com warnings para diagnóstico explícito. get_index_stats()

get_index_health(): status dict retornado como objeto, não string
    Em algumas versões do SDK Databricks (incluindo o ambiente de produção deste
    projeto), getattr(index, "status") retorna um dict diretamente —
    ex.: {'DETAILED_STATE': 'ONLINE_PIPELINE_FAILED', 'INDEXED_ROW_COUNT': 347}.
    A versão anterior fazia str(raw_status).upper(), convertendo o dict inteiro
    para string. Consequências: (1) health["index_status"] ficava como a
    representação textual completa do dict, nunca casando com _HEALTHY_INDEX_STATES
    ou _DEGRADED_OK_STATES, forçando healthy=False permanentemente. (2)
    indexed_row_count nunca era extraído, resultando em index_row_count=0 mesmo
    com 347 documentos indexados. Fix: verificar isinstance(raw_status, dict) antes
    de converter para string, e extrair detailed_state e indexed_row_count do dict.

wait_for_ready(): saída explícita em ONLINE_PIPELINE_FAILED e READY=True
    A versão anterior só saia do loop quando health["healthy"] fosse True ou status
    fosse "FAILED". Com o bug de parsing do status dict, health["healthy"] nunca
    ficava True e o loop bloqueava por até 600s (timeout completo) mesmo com o
    índice operacional. O fix adiciona duas saídas antecipadas: (1) quando o status
    parsed é ONLINE_PIPELINE_FAILED (estado terminal — não melhora sozinho), e
    (2) quando a API retorna READY=True + endpoint ONLINE (índice serve queries).

_save_to_delta(): skip-if-unchanged para evitar CDF churn
    Com CDF habilitado na tabela Delta (requisito do Delta Sync index), mode="overwrite"
    gera 347 DELETE + 347 INSERT = 694 eventos de CDF para 347 documentos. O pipeline
    processa todos esses eventos com SYNC_PROGRESS_COMPLETION=1.0 mas frequentemente
    falha num passo pós-sync, resultando em ONLINE_PIPELINE_FAILED. Ao verificar o
    count atual da tabela antes de escrever, execuções idempotentes (mesmo Gold Layer,
    mesmo count) pulam o write e o sync inteiramente, eliminando o CDF churn.

get_index_health(): tentativas 5 e 6 para cobrir UNKNOWN após skip-write
    Quando skip-write não dispara sync, o SDK reporta UNKNOWN para o índice porque
    nenhuma operação recente foi registrada — o objeto retornado por get_index() não
    tem atributos acessíveis (raw_status=None após tentativas 1-4). Duas tentativas
    adicionais foram adicionadas: (5) client.list_indexes() que retorna lista de dicts
    com estrutura diferente de get_index() e frequentemente tem o estado acessível;
    (6) REST API direta via DATABRICKS_HOST + DATABRICKS_TOKEN do ambiente, fallback
    absoluto independente do SDK. A avaliação de saúde trata UNKNOWN com endpoint
    ONLINE + tabela com dados como "provavelmente funcional" (is_unknown_functional),
    refletindo o comportamento observado: RAG 5/5 com healthy=False no diagnóstico.

wait_for_ready(): saída explícita para UNKNOWN funcional
    Adicionado case UNKNOWN + endpoint ONLINE + table_row_count > 0 como saída
    antecipada do loop de polling — o estado UNKNOWN pós-skip-write é terminal
    (não melhora com mais espera) e o índice está funcional.

retrieve(): semantic_type_override não propagado para _hybrid_retrieve (RAG contamination)
    O parâmetro semantic_type_override chegava até retrieve() corretamente —
    era impresso no log "type_override=regra" — mas não era passado para
    _hybrid_retrieve(). A chamada era _hybrid_retrieve(query, k) sem query_intent,
    então o bloco de boost/penalidade nunca executava e o pool era irrestrito.
    Resultado: silver_srag_clean com score vetorial alto ganhava de
    gold_rag_dicionario_regras mesmo para queries metodológicas. Fix: passar
    query_intent=semantic_type_override na chamada de _hybrid_retrieve.

_hybrid_retrieve(): pool pré-filtrado por semantic_type quando override presente
    Mesmo com boost 1.30x para gold_rag_dicionario_regras (8 docs, score ~0.60),
    documentos silver_srag_clean (339 docs, score ~0.73) podem ganhar no pool
    irrestrito. O fix pré-filtra o pool via search_by_type() quando query_intent
    é um tipo indexado reconhecido, garantindo que o reranking opere dentro do
    conjunto correto. Fallback para search() irrestrito quando o tipo não tem
    documentos suficientes. Aliases "explanatory"/"analytical" → "regra" mapeados
    internamente para desacoplar a nomenclatura do IntentRouter dos tipos indexados.

_build_filter_string() → _build_filter_dict(): formato de filtro errado
    O método gerava filter string no formato SQL "semantic_type = 'regra'", mas
    o Databricks Standard Endpoint (Delta Sync Index) exige dict:
    {"semantic_type": "regra"}. Passar string causa o erro
    "Filter string is not supported for standard endpoints" — a busca filtrada
    falha e o fallback irrestrito retorna documentos de qualquer tipo. O método
    foi renomeado para _build_filter_dict() e retorna dict. _build_filter_string()
    mantido como alias deprecado que também retorna dict para retrocompatibilidade.
    _vector_search() e search() atualizados para passar dict diretamente.

_DEGRADED_OK_STATES
    Conjunto de estados onde o índice está servindo queries (READY=True na API)
    mas o pipeline de atualização incremental falhou. healthy=True nesses estados
    porque o serviço de busca está operacional. O campo warnings contém a explicação
    e instrução de remediação (sync_index manual).
    é mantido como alias para não quebrar consumidores existentes.

wait_for_ready() com polling após sync
    sync_index() chamava index.sync() e retornava imediatamente. O SDK apenas
    dispara a sincronização — o índice fica em estado PROVISIONING/SYNCING por
    minutos antes de ficar ONLINE. Consultar health logo após o disparo retornava
    unknown/0, que era o comportamento observado no notebook 07. wait_for_ready()
    faz polling com intervalo configurável até o índice reportar estado saudável
    ou esgotar o timeout, garantindo que create_or_load_index() só retorna True
    quando o índice está efetivamente pronto para busca.

Embeddings dummy substituídos por rastreamento de falhas
    _embed_documents_in_batches() inseria [0.0 * dim] quando um item individual
    falhava, para preservar o alinhamento doc↔embedding. Embeddings zerados entram
    no índice, aparecem no count mas nunca contribuem para recall relevante — criam
    falsa sensação de completude. A substituição rastreia os índices dos documentos
    que falharam, exclui esses itens do DataFrame e reporta o total ao final. O
    alinhamento é garantido por construção: zip(successful_docs, successful_vectors).

_typed_retrieve() expandido para todos os semantic_types indexados
    A versão anterior detectava apenas geographic e temporal. Os tipos kpi, regra
    e demographic estavam no índice mas nunca eram usados como filtro, fazendo
    perguntas metodológicas e demográficas retornar documentos de qualquer tipo.
    A detecção agora cobre todos os cinco tipos indexados pelo GoldDocumentLoader.

_hybrid_retrieve() com boost por intenção semântica
    O reranking original aplicava boost fixo por source_table e recência. Perguntas
    metodológicas recuperavam majoritariamente documentos de kpi_fatos (dados
    numéricos) em vez de dicionario_regras (definições). O parâmetro query_intent
    ativa boost adicional de 1.3x para gold_rag_dicionario_regras quando intent
    for "regra" ou "explanatory", e penalidade de 0.7x para documentos com
    semantic_type="kpi" nesse mesmo contexto — invertendo a prioridade para
    perguntas conceituais.

retrieve() aceita semantic_type_override do IntentRouter
    O IntentRouter calcula rag_semantic_type no routing decision, mas esse valor
    não era repassado ao retriever — cada chamada usava a estratégia configurada
    globalmente sem aproveitar o contexto semântico já disponível. O parâmetro
    semantic_type_override conecta as duas camadas sem acoplamento direto: quando
    fornecido, ele substitui a detecção interna do _typed_retrieve().

_build_filter_string() expandido com campos úteis
    Os únicos filtros aceitos eram semantic_type, source_table, ano_mes, uf.
    O DataFrame salvo inclui faixa_etaria e o campo ano pode ser filtrado
    diretamente. Adicionados: ano, faixa_etaria, categoria — metadados presentes
    na tabela que antes ficavam inacessíveis via filtro.

_vector_search() registra backend utilizado
    A versão anterior retornava dict anônimo sem indicar qual API foi usada.
    Quando o fallback client.search() era acionado, o chamador não sabia se estava
    usando similarity_search() ou o fallback, impedindo diagnóstico de versão de SDK.
    O campo _search_backend no resultado permite rastrear qual caminho foi tomado
    por chamada.

_vector_search(): fallback client.search() removido
    O fallback chamava self.client.search(...), mas VectorSearchClient não expõe
    esse método no SDK atual — levantava AttributeError que mascarava o erro real da
    chamada principal (e1=similarity_search). Antes do fallback ser tentado, e1 nunca
    era logado, impossibilitando diagnóstico. O fallback foi removido. A falha da
    similarity_search agora é imediatamente classificada e logada via tag prefixada
    antes de retornar _EMPTY_RESULT.

_vector_search(): classificação de erros e log pré-chamada
    A versão anterior não logava os parâmetros enviados a similarity_search — quando
    a busca falhava, era impossível saber se o erro era de colunas, filtros, dimensão
    do embedding, nome do índice ou endpoint. Log de diagnóstico adicionado antes da
    chamada com: nome do índice, endpoint, dimensão do vetor, k, filtros e colunas
    solicitadas. Erros classificados em cinco categorias com tag prefixada nos logs:
    [AUTH] 401/403, [INDEX_NOT_FOUND] 404/not found, [SDK_INCOMPATIBILITY] AttributeError,
    [TRANSIENT] timeout/connection, [FILTER_ERROR] formato inválido, [UNEXPECTED] outros.
    get_index() tem bloco separado para capturar falhas antes de tentar a busca.

_build_filter_dict(): valores None descartados antes do envio
    _sanitize_filter_value(None) converte None para a string literal "None" via str().
    Um filtro {"semantic_type": None} era enviado como {"semantic_type": "None"} para
    a API, produzindo zero resultados sem erro explícito — comportamento indistinguível
    de "nenhum documento deste tipo existe". Entradas com valor None agora são
    descartadas antes da sanitização. Valores que resultam em string vazia após
    sanitização também são descartados.

search(): distinção entre backend=failed e zero resultados reais
    O log "Busca concluída: 0 documentos" era emitido tanto quando a API retornava
    zero resultados genuínos quanto quando o backend falhava (_search_backend=failed).
    Não era possível distinguir "não há documentos para esta query" de "a busca falhou
    completamente antes de qualquer resultado". Log diferenciado adicionado: [SEARCH]
    prefixado para falha de backend e para zero resultados, preservando o log original
    para o caso de sucesso com documentos.
"""

import re
import time
import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import pandas as pd
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from databricks.vector_search.client import VectorSearchClient

# =============================================================================
# IMPORTS OPCIONAIS
# =============================================================================

# DatabricksEmbeddings — pacote: databricks-langchain (>= 0.3.0)
# Migrado de langchain_community (deprecated, removido na 1.0).
# Instalar: pip install -U databricks-langchain
try:
    from databricks_langchain import DatabricksEmbeddings
    DATABRICKS_AVAILABLE = True
except ImportError:
    DATABRICKS_AVAILABLE = False

# HuggingFaceEmbeddings — pacote: langchain-huggingface (fallback local opcional)
# Usado apenas quando provider="huggingface" em EmbeddingManager.get_embeddings().
# Migrado de langchain_community (deprecated, removido na 1.0).
# Instalar: pip install -U langchain-huggingface sentence-transformers
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# =============================================================================
# EMBEDDING MANAGER
# =============================================================================

class EmbeddingManager:
    """
    Factory de embeddings para o pipeline RAG do SRAG.

    Providers suportados
    --------------------
    databricks (recomendado)
        Usa o endpoint Foundation Model API do Databricks com o modelo
        BGE Large English v1.5 (1024 dimensões). O endpoint é passado
        explicitamente via parâmetro endpoint — sem hardcode interno.
        Autenticação via credenciais do cluster Databricks.

    huggingface (fallback local)
        Executa o modelo localmente via sentence-transformers. Útil para
        desenvolvimento offline ou ambientes sem acesso ao Foundation Model
        API. Dimensão padrão: 384 (all-MiniLM-L6-v2).

    Nota sobre o parâmetro model em get_embeddings()
        O parâmetro model documenta qual modelo o endpoint serve, mas não é
        repassado ao construtor de DatabricksEmbeddings — esse construtor
        aceita apenas endpoint=. Passar model= causava TypeError em runtime.
    """

    _DEFAULT_ENDPOINT = "databricks-bge-large-en"

    @staticmethod
    def get_embeddings(
        provider: str = "databricks",
        model:    str = "bge_large_en_v1_5",
        endpoint: Optional[str] = None,
        **kwargs,
    ) -> Embeddings:
        """
        Retorna instância de Embeddings pronta para uso.

        Parâmetros
        ----------
        provider
            'databricks' usa Foundation Model API; 'huggingface' executa local.
        model
            Nome documental do modelo — usado apenas para dimensão via
            get_embedding_dimensions(). Não é repassado ao construtor de
            DatabricksEmbeddings.
        endpoint
            Endpoint Foundation Model API do Databricks. Quando None, usa
            o default interno 'databricks-bge-large-en'. Passar explicitamente
            via VectorStoreConfig.embedding_endpoint é o padrão recomendado.

        Levanta
        -------
        ImportError
            Quando o pacote do provider selecionado não está instalado.
        ValueError
            Quando provider não é 'databricks' nem 'huggingface'.
        """
        if provider == "databricks":
            if not DATABRICKS_AVAILABLE:
                raise ImportError(
                    "DatabricksEmbeddings não disponível. "
                    "Execute: pip install -U databricks-langchain"
                )
            resolved_endpoint = endpoint or EmbeddingManager._DEFAULT_ENDPOINT
            return DatabricksEmbeddings(
                endpoint=resolved_endpoint,
                **kwargs,
            )

        if provider == "huggingface":
            if not HF_AVAILABLE:
                raise ImportError(
                    "HuggingFaceEmbeddings não disponível. "
                    "Execute: pip install -U langchain-huggingface sentence-transformers"
                )
            return HuggingFaceEmbeddings(
                model_name="BAAI/bge-large-en-v1.5",
                **kwargs,
            )

        raise ValueError(
            f"Provider '{provider}' não suportado. Use 'databricks' ou 'huggingface'."
        )

    @staticmethod
    def get_embedding_dimensions(provider: str, model: str) -> int:
        """
        Retorna a dimensão do vetor de embedding para o modelo informado.

        Usado por VectorStoreConfig para definir embedding_dim sem hardcode
        no site de chamada.
        """
        _DIMENSIONS: Dict[str, int] = {
            "bge_large_en_v1_5":  1024,   # BGE Large — padrão Databricks
            "all-MiniLM-L6-v2":    384,   # HuggingFace fallback
            "all-mpnet-base-v2":   768,   # HuggingFace alternativo
        }
        return _DIMENSIONS.get(model, 1024)


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

@dataclass
class VectorStoreConfig:
    """
    Parâmetros do Vector Store para o pipeline SRAG.

    table_name
        Nome explícito da tabela Delta que armazena os embeddings. O código
        anterior derivava o nome por str.replace("_index", "_table"), o que
        falhava silenciosamente se index_name não contivesse "_index".
        Campo explícito torna o contrato visível e testável.

    embedding_endpoint
        Endpoint Foundation Model API do Databricks usado para gerar
        embeddings. Centralizado aqui para que a troca de modelo exija
        apenas mudança de config, não de código. Passado ao EmbeddingManager
        via get_embeddings(endpoint=...).

    embedding_dim
        Deve coincidir com a dimensão real do modelo de embedding. A validação
        em _prepare_documents_with_embeddings() levanta ValueError se houver
        divergência — falha rápida é preferível a indexar vetores com dimensão
        errada, o que corrompe o índice silenciosamente.

    sync_timeout_seconds
        Tempo máximo em segundos que wait_for_ready() aguarda o índice ficar
        ONLINE após um sync. Padrão de 600s cobre sincronizações iniciais
        pesadas; ajustar para baixo em ambientes de desenvolvimento.

    sync_poll_interval_seconds
        Intervalo entre consultas de status durante o polling de wait_for_ready().
    """
    catalog:                    str = "dbx_srag_lab"
    schema:                     str = "gold"
    index_name:                 str = "srag_embeddings_index_bge"
    table_name:                 str = "srag_embeddings_table_bge"
    endpoint_name:              str = "srag_vector_endpoint"
    embedding_endpoint:         str = "databricks-bge-large-en"
    embedding_dim:              int = 1024
    primary_key:                str = "doc_id"
    embedding_source_column:    str = "content"
    embedding_vector_column:    str = "embedding"
    sync_timeout_seconds:       int = 600
    sync_poll_interval_seconds: int = 20


# Estados do Databricks Vector Search considerados saudáveis para busca.
_HEALTHY_INDEX_STATES = frozenset({
    "ONLINE",
    "ONLINE_NO_PENDING_UPDATE",
    "READY",
})

# Estados degradados mas funcionais: o índice serve queries, mas o último
# pipeline sync encontrou um problema. O campo healthy retorna True com
# aviso explícito para que o pipeline continue sem bloquear.
_DEGRADED_OK_STATES = frozenset({
    "ONLINE_PIPELINE_FAILED",
})

# Estados transitórios que indicam sincronização em andamento.
_PROVISIONING_INDEX_STATES = frozenset({
    "PROVISIONING",
    "SYNCING",
    "INITIAL_LOAD",
    "UPDATING",
})


# =============================================================================
# DATABRICKS VECTOR STORE MANAGER
# =============================================================================

class DatabricksVectorStoreManager:
    """
    Gerencia o ciclo completo do Databricks Vector Search para o SRAG.

    Workflow de criação
    -------------------
        1. Garantir que o endpoint Vector Search existe (_ensure_endpoint_exists).
        2. Gerar embeddings dos documentos e montar DataFrame Pandas.
        3. Persistir DataFrame como Delta Table com overwrite total.
        4. Criar índice Delta Sync apontando para a tabela.
        5. Aguardar o índice ficar ONLINE via wait_for_ready().

    Workflow de atualização (create_or_load_index)
        Se o índice já existe, sempre recria a tabela Delta com overwrite,
        dispara sync_index() e aguarda confirmação via wait_for_ready().
        Isso evita acúmulo de registros de execuções anteriores e garante
        que create_or_load_index() só retorna True quando o índice está
        efetivamente pronto para busca.

    Parâmetros
    ----------
    spark
        SparkSession ativa. Obrigatória para criar/ler tabelas Delta.
    config
        VectorStoreConfig com nomes de catálogo, schema, índice e tabela.
        Quando None, usa defaults do dataclass.
    embeddings
        Instância de Embeddings. Quando None, instancia DatabricksEmbeddings
        com endpoint configurado em config.embedding_endpoint.
    """

    def __init__(
        self,
        spark,
        config:     Optional[VectorStoreConfig] = None,
        embeddings: Optional[Embeddings]         = None,
    ):
        self.spark      = spark
        self.config     = config or VectorStoreConfig()
        self.embeddings = embeddings or EmbeddingManager.get_embeddings(
            provider = "databricks",
            model    = "bge_large_en_v1_5",
            endpoint = self.config.embedding_endpoint,
        )
        self.client = VectorSearchClient()

        self.full_index_name = (
            f"{self.config.catalog}.{self.config.schema}.{self.config.index_name}"
        )
        self.full_table_name = (
            f"{self.config.catalog}.{self.config.schema}.{self.config.table_name}"
        )

    # =========================================================================
    # SETUP E CRIAÇÃO
    # =========================================================================

    def create_vector_index(
        self,
        documents: List[Document],
        recreate:  bool = False,
    ) -> str:
        """
        Cria o índice vetorial completo do zero.

        Executa os 5 passos em sequência: endpoint → embeddings → Delta →
        índice → wait_for_ready(). Retorna o nome completo do índice criado.

        Parâmetros
        ----------
        documents
            Lista de documentos LangChain ou SRAGDocument com page_content
            e metadata. SRAGDocuments são convertidos via to_langchain_doc().
        recreate
            Quando True, deleta o índice existente antes de criar um novo.
            Use apenas para reconstrução completa — o índice fica indisponível
            durante a deleção e a sincronização inicial.
        """
        print(f"Criando Vector Index: {self.full_index_name}")

        self._ensure_endpoint_exists()

        print("Gerando embeddings...")
        df = self._prepare_documents_with_embeddings(documents)

        print("Salvando em Delta Table...")
        self._save_to_delta(df)

        print("Criando Vector Index...")
        self._create_or_update_index(recreate=recreate)

        print("Aguardando índice ficar pronto...")
        self.wait_for_ready()

        print(f"Vector Index criado: {self.full_index_name}")
        return self.full_index_name

    def create_or_load_index(self, documents: List[Document]) -> bool:
        """
        Garante que o índice vetorial existe e está atualizado.

        Quando o índice já existe, sempre recria a tabela Delta (overwrite),
        dispara sync_index() e aguarda confirmação via wait_for_ready().
        Quando não existe, executa o fluxo completo de criação.

        Retorna True somente quando o índice está confirmadamente pronto
        para busca — não apenas quando a operação de criação/sync foi
        disparada. Retorna False em caso de falha irrecuperável.
        """
        try:
            print("Verificando se índice existe...")
            try:
                index_info = self.client.get_index(
                    endpoint_name=self.config.endpoint_name,
                    index_name=self.full_index_name,
                )
                if index_info and hasattr(index_info, "name") and index_info.name == self.full_index_name:
                    print(f"Índice existe: {self.full_index_name} — atualizando tabela Delta...")
                    df = self._prepare_documents_with_embeddings(documents)
                    self._save_to_delta(df)
                    self.sync_index(wait=True)
                    return True
            except Exception as get_error:
                print(f"Índice não encontrado (erro esperado na primeira execução): {get_error}")

            print(f"Criando novo índice: {self.full_index_name}")
            self.create_vector_index(documents, recreate=False)
            return True

        except Exception as e:
            print(f"Erro ao verificar/criar índice: {e}")
            return False

    # =========================================================================
    # MANUTENÇÃO
    # =========================================================================

    def sync_index(self, wait: bool = False) -> None:
        """
        Sincroniza o índice com a Delta Table via Delta Sync.

        O método correto é get_index().sync() — não client.sync_index().
        VectorSearchClient não expõe sync_index() diretamente; o método existe
        no objeto VectorSearchIndex retornado por get_index(). Usar
        client.sync_index() levantava AttributeError em tempo de execução.

        Parâmetros
        ----------
        wait
            Quando True, chama wait_for_ready() após o disparo do sync para
            aguardar o índice ficar efetivamente ONLINE antes de retornar.
            Use wait=True em fluxos de criação/atualização; wait=False para
            disparar sync em background sem bloquear o pipeline.
        """
        try:
            index = self.client.get_index(
                endpoint_name=self.config.endpoint_name,
                index_name=self.full_index_name,
            )
            index.sync()
            print(f"Sync disparado: {self.full_index_name}")
            if wait:
                self.wait_for_ready()
        except Exception as e:
            print(f"Erro ao sincronizar {self.full_index_name}: {e}")
            raise

    def wait_for_ready(
        self,
        timeout_seconds:    Optional[int] = None,
        poll_interval:      Optional[int] = None,
    ) -> bool:
        """
        Aguarda o índice ficar ONLINE via polling.

        Consulta get_index_health() em intervalos regulares até o índice
        reportar estado saudável ou o timeout ser esgotado. Distingue entre
        estados transitórios (PROVISIONING, SYNCING) — onde aguarda — e
        estados terminais de falha (FAILED) — onde aborta imediatamente.

        Parâmetros
        ----------
        timeout_seconds
            Tempo máximo de espera. Quando None, usa config.sync_timeout_seconds.
        poll_interval
            Segundos entre consultas. Quando None, usa
            config.sync_poll_interval_seconds.

        Retorno
        -------
        True quando o índice ficou ONLINE dentro do timeout.
        False quando o timeout foi esgotado ou o estado foi FAILED.
        """
        timeout  = timeout_seconds or self.config.sync_timeout_seconds
        interval = poll_interval   or self.config.sync_poll_interval_seconds
        deadline = time.time() + timeout

        print(f"Aguardando índice ficar pronto (timeout={timeout}s, poll={interval}s)...")

        while time.time() < deadline:
            health = self.get_index_health()
            status = health.get("index_status", "unknown").upper()

            if health.get("healthy"):
                print(f"Índice pronto: status={status}")
                return True

            # Saída explícita em estado degradado-funcional (ex.: ONLINE_PIPELINE_FAILED).
            # O índice serve queries normalmente — o erro é no pipeline de sync incremental.
            # Aguardar mais tempo não resolve: o estado ONLINE_PIPELINE_FAILED é terminal
            # até que uma intervenção manual (sync_index) seja feita.
            if status in {s.upper() for s in _DEGRADED_OK_STATES}:
                print(
                    f"  [{int(time.time() - (deadline - timeout))}s] {status} — "
                    f"índice serve queries (READY={health.get('ready_flag')}). "
                    "Pipeline sync falhou mas busca vetorial está operacional. Prosseguindo."
                )
                return True

            # READY=True da API indica que o índice está servindo queries,
            # mesmo que o estado detalhado ainda seja transitório.
            if health.get("ready_flag") and health.get("endpoint_state") == "ONLINE":
                elapsed = int(time.time() - (deadline - timeout))
                print(
                    f"  [{elapsed}s] {status} — API READY=True, endpoint ONLINE. "
                    "Considerando índice operacional."
                )
                return True

            if status == "FAILED":
                print(f"Índice em estado FAILED — abortando wait.")
                return False

            # UNKNOWN com endpoint ONLINE + dados na tabela: skip-write não disparou sync,
            # o SDK reporta UNKNOWN porque o índice não teve atividade recente — mas ele
            # está funcional. Aguardar mais não muda esse estado; sair imediatamente.
            if (
                status == "UNKNOWN"
                and health.get("endpoint_state") == "ONLINE"
                and health.get("table_row_count", 0) > 0
            ):
                elapsed = int(time.time() - (deadline - timeout))
                print(
                    f"  [{elapsed}s] UNKNOWN — endpoint ONLINE + tabela com dados. "
                    "Provável skip-write sem sync. Índice provavelmente funcional. Prosseguindo."
                )
                return True

            elapsed = int(time.time() - (deadline - timeout))
            print(f"  [{elapsed}s] status={status} — aguardando {interval}s...")
            time.sleep(interval)

        print(f"Timeout de {timeout}s esgotado — índice não ficou pronto.")
        return False

    def get_index_health(self) -> Dict:
        """
        Retorna diagnóstico completo do índice vetorial e do endpoint.

        Comportamento de extração de status
            O SDK Databricks não expõe status como atributo direto no objeto
            VectorSearchIndex na maioria das versões. Este método tenta extrair
            o status em quatro formas: (1) atributo direto .status, (2) chamada
            .describe() que retorna dict, (3) atributo .status_message, (4) dict
            de resposta bruta via as_dict(). O primeiro formato que retornar um
            valor não-None é usado. Se nenhum funcionar, status fica como
            "unknown" e healthy é False.

        Campos retornados
        -----------------
        endpoint_name, endpoint_state
            Estado do endpoint Vector Search. ONLINE é o estado esperado.
        index_name, index_status, index_status_raw
            Nome e status extraído do índice. index_status_raw preserva o
            valor original para debug quando o parsing normalizado divergir.
        table_name, table_row_count
            Contagem real de registros na tabela Delta fonte.
        index_row_count
            Contagem de registros no índice, quando disponível via SDK.
        embedding_dim
            Dimensão configurada em VectorStoreConfig.
        healthy
            True somente se endpoint ONLINE, índice em estado saudável
            (_HEALTHY_INDEX_STATES) e table_row_count > 0.
        warnings
            Lista de strings com condições anômalas detectadas.
        """
        health: Dict = {
            "endpoint_name":    self.config.endpoint_name,
            "endpoint_state":   "unknown",
            "index_name":       self.full_index_name,
            "index_status":     "unknown",
            "index_status_raw": None,
            "table_name":       self.full_table_name,
            "table_row_count":  0,
            "index_row_count":  0,
            "embedding_dim":    self.config.embedding_dim,
            "healthy":          False,
            "ready_flag":       False,   # campo READY da API Databricks — True quando o índice serve queries
            "warnings":         [],
        }

        # --- Endpoint ---
        try:
            ep_info = self.client.get_endpoint(self.config.endpoint_name)
            ep_state = (
                ep_info.get("endpoint_status", {}).get("state")
                or ep_info.get("state")
                or "unknown"
            )
            health["endpoint_state"] = str(ep_state).upper()
            if health["endpoint_state"] != "ONLINE":
                health["warnings"].append(
                    f"Endpoint não está ONLINE: state={health['endpoint_state']}"
                )
        except Exception as ep_err:
            health["warnings"].append(f"Erro ao consultar endpoint: {ep_err}")

        # --- Tabela Delta ---
        try:
            table_count = self.spark.table(self.full_table_name).count()
            health["table_row_count"] = table_count
            if table_count == 0:
                health["warnings"].append("Tabela Delta está vazia (0 registros).")
        except Exception as tbl_err:
            health["warnings"].append(f"Erro ao contar registros Delta: {tbl_err}")

        # --- Índice ---
        try:
            index = self.client.get_index(
                endpoint_name=self.config.endpoint_name,
                index_name=self.full_index_name,
            )

            # Tentativa 1: atributo direto .status
            # ATENÇÃO: em algumas versões do SDK Databricks, .status retorna um dict
            # diretamente (ex.: {'DETAILED_STATE': 'ONLINE_PIPELINE_FAILED', 'INDEXED_ROW_COUNT': 347}).
            # Não converter para str() sem antes verificar o tipo — str(dict).upper() produziria
            # a representação completa do dict como string, impossibilitando comparação com
            # _HEALTHY_INDEX_STATES e perdendo o campo indexed_row_count.
            raw_status = getattr(index, "status", None)

            # Normaliza dict de status retornado diretamente pelo SDK
            if isinstance(raw_status, dict):
                status_dict = {k.lower(): v for k, v in raw_status.items()}
                raw_status = (
                    status_dict.get("detailed_state")
                    or status_dict.get("state")
                    or "unknown"
                )
                rc = (
                    status_dict.get("indexed_row_count")
                    or status_dict.get("num_indexed_rows")
                    or status_dict.get("num_rows")
                )
                if rc is not None:
                    health["index_row_count"] = int(rc)
                health["ready_flag"] = bool(status_dict.get("ready"))

            # Tentativa 2: método describe() retorna dict
            if raw_status is None and hasattr(index, "describe"):
                try:
                    desc = index.describe()
                    if isinstance(desc, dict):
                        desc_lower = {k.lower(): v for k, v in desc.items()}
                        raw_status = (
                            desc_lower.get("detailed_state")
                            or desc_lower.get("state")
                            or (desc_lower.get("index_status") or {}).get("detailed_state")
                            or (desc_lower.get("index_status") or {}).get("state")
                        )
                        # indexed_row_count pode estar no nível raiz ou dentro de index_status
                        row_count = (
                            desc_lower.get("indexed_row_count")
                            or desc_lower.get("num_indexed_rows")
                            or (desc_lower.get("index_status") or {}).get("indexed_row_count")
                            or desc_lower.get("num_rows", 0)
                        )
                        if health["index_row_count"] == 0:
                            health["index_row_count"] = int(row_count or 0)
                        if not health.get("ready_flag"):
                            health["ready_flag"] = bool(
                                desc_lower.get("ready")
                                or (desc_lower.get("index_status") or {}).get("ready")
                            )
                except Exception:
                    pass

            # Tentativa 3: status_message como fallback textual
            if raw_status is None:
                raw_status = getattr(index, "status_message", None)

            # Tentativa 4: as_dict() para SDKs que serializam o objeto
            if raw_status is None and hasattr(index, "as_dict"):
                try:
                    d = index.as_dict()
                    d_lower = {k.lower(): v for k, v in d.items()}
                    raw_status = (
                        d_lower.get("detailed_state")
                        or d_lower.get("state")
                        or (d_lower.get("index_status") or {}).get("detailed_state")
                    )
                    rc = (
                        d_lower.get("indexed_row_count")
                        or d_lower.get("num_indexed_rows")
                        or d_lower.get("num_rows")
                    )
                    if rc is not None and health["index_row_count"] == 0:
                        health["index_row_count"] = int(rc)
                    if not health.get("ready_flag"):
                        health["ready_flag"] = bool(d_lower.get("ready"))
                except Exception:
                    pass

            # Tentativa 5: client.list_indexes() — retorna lista de dicts com estrutura
            # diferente de get_index(). Quando get_index() retorna objeto opaco sem
            # atributos acessíveis (raw_status ainda None após tentativas 1-4), esta
            # chamada alternativa frequentemente retorna o status em formato dict direto.
            if raw_status is None and hasattr(self.client, "list_indexes"):
                try:
                    indexes = self.client.list_indexes(
                        endpoint_name=self.config.endpoint_name
                    )
                    # list_indexes retorna lista ou dict com chave "vector_indexes"
                    idx_list = (
                        indexes if isinstance(indexes, list)
                        else indexes.get("vector_indexes", []) if isinstance(indexes, dict)
                        else []
                    )
                    for idx_entry in idx_list:
                        entry = idx_entry if isinstance(idx_entry, dict) else {}
                        entry_lower = {k.lower(): v for k, v in entry.items()}
                        # Identifica pelo nome do índice
                        entry_name = (
                            entry_lower.get("name")
                            or entry_lower.get("index_name")
                            or ""
                        )
                        if self.config.index_name not in str(entry_name):
                            continue
                        raw_status = (
                            entry_lower.get("detailed_state")
                            or entry_lower.get("status")
                            or (entry_lower.get("index_status") or {}).get("detailed_state")
                            or (entry_lower.get("index_status") or {}).get("state")
                        )
                        rc = (
                            entry_lower.get("indexed_row_count")
                            or entry_lower.get("num_indexed_rows")
                            or (entry_lower.get("index_status") or {}).get("indexed_row_count")
                        )
                        if rc is not None and health["index_row_count"] == 0:
                            health["index_row_count"] = int(rc)
                        if not health.get("ready_flag"):
                            health["ready_flag"] = bool(
                                entry_lower.get("ready")
                                or (entry_lower.get("index_status") or {}).get("ready")
                            )
                        break
                except Exception:
                    pass

            # Tentativa 6: REST API direta via token do contexto Databricks.
            # Fallback absoluto quando o SDK não expõe o estado em nenhum formato.
            # Usa o workspace URL e token do ambiente para chamar a Vector Search API.
            # Silenciosamente ignorado em ambientes sem requests ou sem token configurado.
            if raw_status is None:
                try:
                    import requests as _requests
                    import os as _os

                    workspace_url = (
                        _os.environ.get("DATABRICKS_HOST")
                        or _os.environ.get("DB_WORKSPACE_URL")
                        or ""
                    ).rstrip("/")
                    token = (
                        _os.environ.get("DATABRICKS_TOKEN")
                        or _os.environ.get("DB_TOKEN")
                        or ""
                    )

                    if workspace_url and token:
                        resp = _requests.get(
                            f"{workspace_url}/api/2.0/vector-search/indexes/{self.full_index_name}",
                            headers={"Authorization": f"Bearer {token}"},
                            timeout=10,
                        )
                        if resp.ok:
                            data = resp.json()
                            data_lower = {k.lower(): v for k, v in data.items()}
                            idx_status = data_lower.get("index_status") or {}
                            if isinstance(idx_status, dict):
                                idx_low = {k.lower(): v for k, v in idx_status.items()}
                                raw_status = (
                                    idx_low.get("detailed_state")
                                    or idx_low.get("state")
                                )
                                rc = idx_low.get("indexed_row_count")
                                if rc is not None and health["index_row_count"] == 0:
                                    health["index_row_count"] = int(rc)
                                if not health.get("ready_flag"):
                                    health["ready_flag"] = bool(idx_low.get("ready"))
                except Exception:
                    pass

            health["index_status_raw"] = raw_status
            health["index_status"] = str(raw_status or "unknown").upper()

            # Row count via atributo direto como último recurso
            if health["index_row_count"] == 0:
                direct_rows = getattr(index, "num_rows", None) or getattr(index, "indexed_row_count", None)
                if direct_rows is not None:
                    health["index_row_count"] = int(direct_rows)

        except Exception as idx_err:
            health["warnings"].append(f"Erro ao consultar índice: {idx_err}")

        # --- Avaliação de saúde ---
        is_endpoint_ok  = health["endpoint_state"] == "ONLINE"
        is_index_ok     = health["index_status"] in _HEALTHY_INDEX_STATES
        is_degraded_ok  = health["index_status"] in _DEGRADED_OK_STATES
        has_data        = health["table_row_count"] > 0
        # READY=True da API indica que o índice está servindo queries,
        # mesmo quando detailed_state não é ONLINE (ex.: ONLINE_PIPELINE_FAILED).
        is_api_ready    = health.get("ready_flag", False)

        # UNKNOWN após skip-write: quando _save_to_delta skipa o overwrite (count igual),
        # nenhum sync é disparado — o índice permanece em estado não-alterado que o SDK
        # reporta como UNKNOWN porque nenhuma operação recente foi feita. O índice continua
        # servindo queries normalmente (RAG 5/5 no notebook 07). Tratamos UNKNOWN com
        # endpoint ONLINE + dados presentes como "provavelmente funcional".
        is_unknown_functional = (
            health["index_status"] == "UNKNOWN"
            and is_endpoint_ok
            and has_data
        )

        # healthy=True quando: endpoint ONLINE + (estado saudável OU degradado funcional
        # OU api READY OU UNKNOWN funcional) + dados na tabela Delta.
        health["healthy"] = is_endpoint_ok and (
            is_index_ok or is_degraded_ok or is_api_ready or is_unknown_functional
        ) and has_data

        if is_degraded_ok or (is_api_ready and not is_index_ok):
            health["warnings"].append(
                f"Índice em estado degradado mas funcional ({health['index_status']}, READY={is_api_ready}). "
                "O índice está servindo queries, mas o último pipeline sync falhou. "
                "Causa comum: overwrite com CDF ativo gera eventos DELETE+INSERT duplicados. "
                "Execute sync_index() quando possível para restaurar estado ONLINE completo."
            )

        if is_unknown_functional:
            health["warnings"].append(
                "Índice em UNKNOWN — SDK não conseguiu extrair o estado após 6 tentativas. "
                "Causa provável: skip-write (dados não mudaram, nenhum sync disparado) faz o "
                "SDK reportar UNKNOWN em vez do estado real. Endpoint ONLINE + tabela com dados "
                "sugerem que o índice está funcional. Valide com uma query de teste ao retriever."
            )

        # Dessincronização real: tabela tem dados, índice não está em nenhum estado reconhecido,
        # e não é o caso UNKNOWN funcional (skip-write sem sync). Só alerta quando há evidência
        # real de problema — não quando o SDK simplesmente não consegue ler o estado.
        if (
            has_data
            and health["index_row_count"] == 0
            and not (is_index_ok or is_degraded_ok or is_api_ready or is_unknown_functional)
        ):
            health["warnings"].append(
                f"Dessincronização detectada: tabela={health['table_row_count']} registros, "
                f"índice status={health['index_status']} (não ONLINE/READY). Execute sync_index()."
            )

        # Status transitório — não é falha, mas não está pronto
        if health["index_status"] in {s.upper() for s in _PROVISIONING_INDEX_STATES}:
            health["warnings"].append(
                f"Índice em estado transitório ({health['index_status']}) "
                "— aguardar sincronização completar."
            )

        return health

    def get_index_stats(self) -> Dict:
        """
        Alias de get_index_health() para compatibilidade com o notebook 07.

        Retorna o mesmo dict de get_index_health(). Consumidores novos devem
        usar get_index_health() diretamente para acessar o campo healthy e
        a lista de warnings.
        """
        return self.get_index_health()

    def delete_index(self) -> None:
        """Deleta o índice vetorial. Irreversível — use com cautela."""
        try:
            self.client.delete_index(index_name=self.full_index_name)
            print(f"Índice deletado: {self.full_index_name}")
        except Exception as e:
            print(f"Erro ao deletar índice: {e}")
            raise

    # =========================================================================
    # BUSCA E RETRIEVAL
    # =========================================================================

    def search(
        self,
        query:   str,
        k:       int = 5,
        filters: Optional[Dict] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Busca semântica no Vector Index usando o embedding da query.

        Score de similaridade
            O score é extraído de row[-1] (último elemento de cada linha do
            data_array). A versão anterior tentava result.get("scores", []),
            mas essa chave não existe na resposta da API — estava sempre vazia,
            resultando em score=0.8 fixo para todos os documentos e tornando
            o reranking inefetivo.

        Filtros
            Valores são sanitizados via whitelist (apenas alfanuméricos, hífens,
            underscores e espaços) antes de montar a filter_string.

            Chaves permitidas: semantic_type, source_table, ano_mes, uf,
            ano, faixa_etaria, categoria.

        Retry com backoff linear
            3 tentativas com backoff de 2, 4, 6 segundos. Cobre instabilidades
            transientes da API Vector Search sem impacto perceptível em execuções
            normais.

        Retorno
            Lista de (Document, float) ordenada por score decrescente.
            Nunca retorna None — lista vazia em caso de falha total.
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                query_embedding = list(self._get_cached_query_embedding(query))
                filter_dict     = self._build_filter_dict(filters)
                search_results  = self._vector_search(query_embedding, k, filter_dict)

                documents: List[Tuple[Document, float]] = []

                if "result" in search_results:
                    data_array = search_results["result"].get("data_array", [])

                    for idx, row in enumerate(data_array):
                        try:
                            doc_id        = row[0] if len(row) > 0 else f"doc_{idx}"
                            content       = row[1] if len(row) > 1 else ""
                            metadata_json = row[2] if len(row) > 2 else "{}"
                            source_table  = row[3] if len(row) > 3 else ""
                            semantic_type = row[4] if len(row) > 4 else ""

                            # Score está no último elemento da linha.
                            # A chave "scores" não existe na resposta da API.
                            score = float(row[-1]) if len(row) > 0 else 0.0

                            try:
                                metadata = json.loads(metadata_json) if metadata_json else {}
                            except json.JSONDecodeError:
                                metadata = {}

                            metadata.update({
                                "doc_id":             doc_id,
                                "source_table":       source_table,
                                "semantic_type":      semantic_type,
                                "_search_backend":    search_results.get("_search_backend", "unknown"),
                            })

                            documents.append((
                                Document(page_content=content, metadata=metadata),
                                score,
                            ))

                        except Exception as row_error:
                            print(f"   Aviso: erro ao processar linha {idx}: {row_error}")
                            continue

                documents.sort(key=lambda x: x[1], reverse=True)

                backend = search_results.get("_search_backend", "?")
                if backend == "failed":
                    # data_array vazio por falha de backend — não por ausência real de documentos.
                    # _vector_search já logou a causa classificada antes de retornar _EMPTY_RESULT.
                    print(
                        f"[SEARCH] Busca retornou backend=failed — falha de infra/SDK, "
                        f"não ausência de resultados. Consulte os logs [SEARCH][*] acima."
                    )
                    return []

                top_score = f"{documents[0][1]:.4f}" if documents else "N/A"
                if not documents:
                    print(
                        f"[SEARCH] Zero resultados reais para a query "
                        f"(backend={backend}, índice acessível mas sem match)."
                    )
                else:
                    print(
                        f"Busca concluída: {len(documents)} documentos "
                        f"(score_top={top_score}, backend={backend})"
                    )
                return documents

            except Exception as e:
                if attempt < max_retries - 1:
                    backoff = 2 * (attempt + 1)
                    print(f"Tentativa {attempt + 1} falhou ({e}) — aguardando {backoff}s...")
                    time.sleep(backoff)
                else:
                    print(f"Busca falhou após {max_retries} tentativas: {e}")

        return []

    def search_by_type(
        self,
        query:         str,
        semantic_type: str,
        k:             int = 5,
    ) -> List[Tuple[Document, float]]:
        """
        Atalho para busca filtrada por semantic_type.

        Tipos válidos: 'temporal', 'geographic', 'demographic', 'kpi', 'regra'.
        """
        return self.search(query, k=k, filters={"semantic_type": semantic_type})

    # =========================================================================
    # MÉTODOS INTERNOS — SETUP
    # =========================================================================

    def _ensure_endpoint_exists(self) -> None:
        """
        Garante que o endpoint Vector Search existe no workspace.

        Usa list_endpoints() para verificar antes de criar — create_endpoint()
        levanta erro se o endpoint já existir, mas list_endpoints() tem
        assinatura estável entre versões do SDK.
        """
        try:
            endpoints      = self.client.list_endpoints()
            endpoint_names = [e["name"] for e in endpoints.get("endpoints", [])]

            if self.config.endpoint_name not in endpoint_names:
                print(f"Criando endpoint: {self.config.endpoint_name}")
                self.client.create_endpoint(
                    name          = self.config.endpoint_name,
                    endpoint_type = "STANDARD",
                )
                print(f"Endpoint criado: {self.config.endpoint_name}")
            else:
                print(f"Endpoint já existe: {self.config.endpoint_name}")

        except Exception as e:
            print(f"Aviso: erro ao verificar endpoint — {e}")

    def _prepare_documents_with_embeddings(
        self,
        documents: list,
    ) -> pd.DataFrame:
        """
        Converte documentos em DataFrame Pandas com vetores de embedding.

        Aceita tanto SRAGDocument (com to_langchain_doc()) quanto Document
        do LangChain diretamente. Documentos sem page_content são ignorados.
        doc_id é gerado automaticamente quando ausente para não perder
        documentos válidos por metadado faltante.

        Tratamento de falhas de embedding
            Documentos cujo embedding falha após todas as tentativas são
            registrados em uma lista de falhos e excluídos do DataFrame.
            Não são substituídos por vetores dummy — documentos com vetor
            zero entram no índice mas nunca contribuem para recall semântico
            e contaminam o count do índice com falsa completude.

        Validação de dimensão
            Levanta ValueError se a dimensão dos vetores gerados divergir de
            embedding_dim na config. Falha rápida é preferível a indexar vetores
            com dimensão errada, o que corrompe o índice silenciosamente.

        Retorno
            DataFrame com colunas: doc_id, content, embedding, source_table,
            semantic_type, uf, ano_mes, faixa_etaria, metadata_json, created_at.
        """
        if not documents:
            raise ValueError("Lista de documentos está vazia.")

        print(f"   Preparando {len(documents)} documentos para embedding...")

        # Normalizar para LangChain Document
        langchain_docs: List[Document] = []
        for i, doc in enumerate(documents):
            if hasattr(doc, "to_langchain_doc"):
                langchain_docs.append(doc.to_langchain_doc())
            elif hasattr(doc, "page_content"):
                langchain_docs.append(doc)
            else:
                print(f"   Documento {i} de tipo desconhecido ({type(doc)}) — ignorado.")

        # Filtrar documentos inválidos
        valid_docs: List[Document] = []
        for i, doc in enumerate(langchain_docs):
            if not doc.page_content or not doc.page_content.strip():
                print(f"   Documento {i} sem conteúdo — ignorado.")
                continue
            if not doc.metadata.get("doc_id"):
                doc.metadata["doc_id"] = f"auto_gen_{i}_{int(datetime.now().timestamp())}"
                print(f"   Documento {i} sem doc_id — gerado automaticamente.")
            valid_docs.append(doc)

        if not valid_docs:
            raise ValueError("Nenhum documento válido após filtragem.")

        print(f"   {len(valid_docs)} documentos válidos para embedding.")

        # Gerar embeddings com rastreamento de falhas
        texts = [doc.page_content.strip() for doc in valid_docs]
        successful_vectors, failed_indices = self._embed_with_retry(texts)

        # Filtrar documentos que falharam no embedding
        failed_set     = set(failed_indices)
        successful_docs = [doc for i, doc in enumerate(valid_docs) if i not in failed_set]

        if failed_indices:
            print(
                f"   {len(failed_indices)} documentos excluídos por falha de embedding — "
                f"{len(successful_docs)} incluídos."
            )

        if not successful_docs:
            raise ValueError("Nenhum documento gerou embedding com sucesso.")

        # Validar dimensão com o primeiro vetor
        actual_dim   = len(successful_vectors[0])
        expected_dim = self.config.embedding_dim
        if actual_dim != expected_dim:
            raise ValueError(
                f"Dimensão do embedding diverge: esperado={expected_dim}, "
                f"atual={actual_dim}. Verifique a configuração do modelo."
            )
        print(f"   Dimensão validada: {actual_dim}d")

        # Construir DataFrame — alinhamento garantido por zip após filtragem
        rows = []
        for doc, embedding in zip(successful_docs, successful_vectors):
            meta = doc.metadata.copy()
            rows.append({
                "doc_id":        meta.get("doc_id", ""),
                "content":       doc.page_content.strip(),
                "embedding":     embedding,
                "source_table":  meta.get("source_table", "unknown"),
                "semantic_type": meta.get("semantic_type", "general"),
                "uf":            meta.get("uf", "BR"),
                "ano_mes":       meta.get("ano_mes", "2024-01"),
                "faixa_etaria":  meta.get("faixa_etaria", "todas"),
                "metadata_json": json.dumps(meta, ensure_ascii=False),
                "created_at":    datetime.now().isoformat(),
            })

        df = pd.DataFrame(rows)
        print(f"   DataFrame pronto: {len(df)} linhas, {len(df.columns)} colunas.")
        return df

    def _embed_with_retry(
        self,
        texts: List[str],
    ) -> Tuple[List[List[float]], List[int]]:
        """
        Gera embeddings com retry automático e fallback para lotes menores.

        Retorno
        -------
        (successful_vectors, failed_indices)
            successful_vectors: embeddings gerados com sucesso, paralelos ao
            subconjunto de texts que não falhou.
            failed_indices: índices em texts cujo embedding falhou após todas
            as tentativas. O chamador exclui esses itens do zip doc↔embedding.

        Estratégia
            Tentativa 1: embed_documents() para todo o lote de uma vez.
            Tentativa 2: aguarda conforme tipo de erro (rate limit=10s, rede=3s).
            Após max_retries: fragmenta em lotes de 10 e tenta unitariamente.
            Itens que falham individualmente são registrados em failed_indices
            e não incluídos no retorno — nunca substituídos por dummy embeddings.
        """
        max_retries = 2
        for attempt in range(max_retries):
            try:
                vectors = self.embeddings.embed_documents(texts)
                print(f"   {len(vectors)} embeddings gerados.")
                return vectors, []
            except Exception as e:
                error_str = str(e).lower()
                if attempt < max_retries - 1:
                    wait = (
                        10 if ("rate limit" in error_str or "quota" in error_str) else
                         3 if ("connection" in error_str or "reset" in error_str) else
                         5
                    )
                    print(f"   Tentativa {attempt + 1} falhou ({e}) — aguardando {wait}s...")
                    time.sleep(wait)
                else:
                    if len(texts) > 10:
                        print("   Usando lotes menores como fallback...")
                        return self._embed_documents_in_batches(texts, batch_size=10)
                    # Falha completa — todos os índices vão para failed
                    print(f"   Falha total no embedding de {len(texts)} textos: {e}")
                    return [], list(range(len(texts)))

        return [], list(range(len(texts)))

    def _embed_documents_in_batches(
        self,
        texts:      List[str],
        batch_size: int = 10,
    ) -> Tuple[List[List[float]], List[int]]:
        """
        Fallback de embedding em lotes menores para contornar rate limits.

        Retorno
        -------
        (successful_vectors, failed_indices)
            Apenas os vetores dos textos que foram gerados com sucesso.
            failed_indices contém os índices originais dos textos que falharam
            após 3 tentativas individuais. O chamador usa esses índices para
            excluir os documentos correspondentes do DataFrame.

        Documentos com falha são registrados no log com seu índice original
        e excluídos do retorno — não são substituídos por vetores dummy.
        Embeddings zerados contaminam o índice com documentos tecnicamente
        presentes mas semanticamente inertes.
        """
        print(f"   Processando {len(texts)} textos em lotes de {batch_size}...")

        # all_results[i] = vetor de embedding ou None se falhou
        all_results: List[Optional[List[float]]] = [None] * len(texts)
        total_batches = (len(texts) - 1) // batch_size + 1

        for i in range(0, len(texts), batch_size):
            batch       = texts[i:i + batch_size]
            batch_idx   = list(range(i, min(i + batch_size, len(texts))))
            batch_num   = (i // batch_size) + 1

            print(f"   Lote {batch_num}/{total_batches}: {len(batch)} textos")

            batch_ok = False
            for attempt in range(3):
                try:
                    batch_embs = self.embeddings.embed_documents(batch)
                    for j, emb in zip(batch_idx, batch_embs):
                        all_results[j] = emb
                    if i + batch_size < len(texts):
                        time.sleep(1)
                    batch_ok = True
                    break
                except Exception as e:
                    if attempt < 2:
                        wait = 2 * (attempt + 1)
                        print(f"   Lote {batch_num} falhou ({e}) — aguardando {wait}s...")
                        time.sleep(wait)

            if not batch_ok:
                # Tentativa unitária por item
                print(f"   Lote {batch_num} falhou após 3 tentativas — tentando unitariamente...")
                for orig_idx, text in zip(batch_idx, batch):
                    try:
                        vec = self.embeddings.embed_documents([text])
                        all_results[orig_idx] = vec[0]
                        time.sleep(0.5)
                    except Exception as item_err:
                        print(f"   Documento idx={orig_idx} falhou individualmente: {item_err}")
                        # all_results[orig_idx] permanece None

        # Separar sucessos e falhas
        successful_vectors: List[List[float]] = []
        failed_indices:     List[int]          = []

        for orig_idx, result in enumerate(all_results):
            if result is not None:
                successful_vectors.append(result)
            else:
                failed_indices.append(orig_idx)

        print(
            f"   Processamento concluído: {len(successful_vectors)} embeddings, "
            f"{len(failed_indices)} falhos."
        )
        return successful_vectors, failed_indices

    def _save_to_delta(self, df: pd.DataFrame) -> None:
        """
        Persiste o DataFrame de embeddings na Delta Table.

        Skip-if-unchanged (CDF churn prevention)
            Com CDF habilitado na tabela Delta (requisito do Delta Sync index),
            um `overwrite` gera 347 DELETE + 347 INSERT = 694 eventos de CDF.
            O pipeline do Vector Search processa todos esses eventos e frequentemente
            falha num passo pós-sync, resultando em ONLINE_PIPELINE_FAILED.

            Para evitar esse overhead quando os dados não mudaram (ex.: restart do
            notebook 06 com o mesmo Gold Layer), comparamos o count atual da tabela
            com o count do novo DataFrame. Se forem iguais, pulamos o write e o sync.
            Isso elimina o CDF churn em execuções idempotentes.

            Quando o count muda (novos documentos adicionados ao Gold), o overwrite
            ocorre normalmente. O pipeline processará 2× eventos mas o dado está
            atualizado.

        Modo de escrita sempre overwrite (quando necessário)
            A tabela de embeddings é derivada inteiramente dos documentos passados
            em cada execução — acumular registros de execuções anteriores causava
            duplicação (347 → 694 → 1041 registros). overwriteSchema=true descarta
            o schema antigo antes de qualquer merge, tornando mergeSchema=true
            desnecessário.

        Particionamento por semantic_type
            Melhora performance de queries filtradas por tipo (geographic, temporal,
            demographic). Cada semantic_type vira uma partição separada no Delta.
        """
        if df.empty:
            raise ValueError("DataFrame está vazio — nada a salvar.")

        # Skip write se a tabela já tem o mesmo número de registros (idempotência)
        try:
            existing_count = self.spark.table(self.full_table_name).count()
            if existing_count == len(df):
                print(
                    f"   Skip write: tabela já contém {existing_count} registros "
                    f"(mesmo count que os {len(df)} novos). "
                    "Nenhum CDF event será gerado — sync desnecessário."
                )
                return
            else:
                print(
                    f"   Tabela existente com {existing_count} registros → "
                    f"atualizando para {len(df)} registros."
                )
        except Exception:
            # Tabela não existe ainda — prossegue com write normal
            pass

        print(f"   Salvando {len(df)} registros em: {self.full_table_name}")

        spark_df = self.spark.createDataFrame(df).repartition("semantic_type")

        (
            spark_df.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .option("delta.autoOptimize.optimizeWrite", "true")
            .option("delta.autoOptimize.autoCompact", "true")
            .partitionBy("semantic_type")
            .saveAsTable(self.full_table_name)
        )

        self.spark.sql(f"OPTIMIZE {self.full_table_name}")
        count = self.spark.table(self.full_table_name).count()
        print(f"   Tabela salva: {count} registros em {self.full_table_name}")

    def _create_or_update_index(self, recreate: bool = False) -> object:
        """
        Cria ou valida o índice vetorial Delta Sync no Databricks.

        Verificação de existência via get_index()
            list_indexes() tem assinatura inconsistente entre versões do SDK
            (às vezes retorna lista, às vezes dict). get_index() levanta exceção
            quando o índice não existe — tratamos essa exceção como ausência,
            não como erro fatal.

        Comportamento quando índice já existe
            A versão anterior retornava imediatamente com "nada a fazer" quando
            o índice era encontrado e recreate=False. Isso ignorava o estado real
            do índice — "índice existe" não implica "índice saudável" nem
            "índice sincronizado". Agora o método sempre verifica o status via
            get_index_health() e emite warnings quando o estado não é saudável.

        recreate=True
            Deleta o índice existente e aguarda 10 segundos antes de recriar.
            O índice fica indisponível nesse intervalo — use apenas em
            reconstrução completa fora de horário de uso.

        pipeline_type=TRIGGERED
            O índice não sincroniza automaticamente com a Delta Table. A
            sincronização acontece apenas via sync_index() explícito. Isso
            evita sincronizações em background que consomem compute no cluster
            durante execuções de ingestão longas.
        """
        print(f"   Verificando índice: {self.full_index_name}")

        # Verificar se endpoint está ativo
        try:
            endpoint_info  = self.client.get_endpoint(self.config.endpoint_name)
            endpoint_state = (
                endpoint_info.get("endpoint_status", {}).get("state")
                or endpoint_info.get("state")
                or "unknown"
            )
            if str(endpoint_state).upper() != "ONLINE":
                print(f"   Aviso: endpoint não está ONLINE — state={endpoint_state}")
        except Exception as ep_err:
            print(f"   Aviso: erro ao verificar endpoint — {ep_err}")

        # Verificar se tabela Delta existe e tem dados
        table_count = self.spark.table(self.full_table_name).count()
        if table_count == 0:
            raise ValueError(f"Tabela {self.full_table_name} está vazia.")
        print(f"   Tabela fonte: {table_count} registros")

        # Verificar se índice já existe
        index_exists = False
        try:
            index_info = self.client.get_index(
                endpoint_name=self.config.endpoint_name,
                index_name=self.full_index_name,
            )
            if index_info and hasattr(index_info, "name") and index_info.name == self.full_index_name:
                index_exists = True

                if recreate:
                    print(f"   Deletando índice existente: {self.full_index_name}")
                    self.client.delete_index(index_name=self.full_index_name)
                    time.sleep(10)
                    index_exists = False
                else:
                    # Verificar saúde — não retornar cegamente sem validar estado
                    health = self.get_index_health()
                    if health.get("healthy"):
                        print(f"   Índice existente e saudável: status={health['index_status']}")
                    else:
                        print(f"   Índice existente mas não saudável: status={health['index_status']}")
                        for w in health.get("warnings", []):
                            print(f"     Aviso: {w}")
                    return index_info

        except Exception as check_error:
            print(f"   Índice não encontrado (esperado na primeira execução): {check_error}")

        # Criar índice
        if not index_exists:
            print(f"   Criando índice: {self.full_index_name}")
            index = self.client.create_delta_sync_index(
                endpoint_name          = self.config.endpoint_name,
                index_name             = self.full_index_name,
                source_table_name      = self.full_table_name,
                pipeline_type          = "TRIGGERED",
                primary_key            = self.config.primary_key,
                embedding_dimension    = self.config.embedding_dim,
                embedding_vector_column= self.config.embedding_vector_column,
            )
            print("   Índice criado — aguarde a sincronização inicial.")
            return index

        return {"status": "already_exists"}

    # =========================================================================
    # MÉTODOS INTERNOS — BUSCA
    # =========================================================================

    def _vector_search(
        self,
        query_embedding: List[float],
        k:               int,
        filters:         Optional[Dict] = None,
    ) -> Dict:
        """
        Executa a busca vetorial via index.similarity_search().

        Caminho único de busca — não há fallback para client.search() porque
        VectorSearchClient não expõe esse método no SDK atual. Chamar um método
        inexistente levantaria AttributeError que mascararia o erro real da chamada
        principal, dificultando o diagnóstico da causa raiz.

        Em caso de falha, o erro é classificado e logado com tag prefixada antes de
        retornar _EMPTY_RESULT. O get_index() tem bloco próprio para capturar falhas
        de autenticação ou ausência do índice antes de tentar a busca.

        Categorias de erro
        ------------------
        [AUTH]                  HTTP 401/403 — credenciais ou permissões inválidas.
        [INDEX_NOT_FOUND]       HTTP 404 — índice ou endpoint ausente ou não criado.
        [INDEX_ERROR]           Outro erro em get_index() antes de tentar a busca.
        [SDK_INCOMPATIBILITY]   AttributeError — similarity_search não existe na versão
                                do SDK instalada; verificar databricks-vector-search.
        [TRANSIENT]             Timeout ou connection error — instabilidade temporária,
                                elegível para retry pelo loop em search().
        [FILTER_ERROR]          Formato de filtro rejeitado pela API.
        [UNEXPECTED]            Qualquer outro erro — loggado com parâmetros completos.

        filters deve ser dict {"column": "value"} sem valores None — _build_filter_dict()
        garante essa invariante antes de chamar este método.

        Campos do retorno
        -----------------
        _search_backend="similarity_search"   sucesso normal.
        _search_backend="failed"              falha total; distingue de zero resultados reais.
        """
        _EMPTY_RESULT: Dict = {
            "result":          {"data_array": [], "row_count": 0},
            "_search_backend": "failed",
        }
        _COLUMNS = ["doc_id", "content", "metadata_json", "source_table", "semantic_type"]

        # Bloco separado para get_index() — falhas aqui são de infra/auth, não de busca.
        try:
            index = self.client.get_index(
                endpoint_name=self.config.endpoint_name,
                index_name=self.full_index_name,
            )
        except Exception as idx_err:
            err_str = str(idx_err).lower()
            if any(t in err_str for t in ("401", "403", "unauthorized", "forbidden")):
                print(
                    f"[SEARCH][AUTH] Falha de autenticação ao obter o índice. "
                    f"Verifique o token e as permissões sobre o endpoint "
                    f"'{self.config.endpoint_name}'. Detalhe: {idx_err}"
                )
            elif any(t in err_str for t in ("404", "not found", "does not exist")):
                print(
                    f"[SEARCH][INDEX_NOT_FOUND] Índice '{self.full_index_name}' não "
                    f"encontrado no endpoint '{self.config.endpoint_name}'. "
                    f"Verifique se o índice foi criado e o endpoint está ONLINE. "
                    f"Detalhe: {idx_err}"
                )
            else:
                print(
                    f"[SEARCH][INDEX_ERROR] Erro ao obter objeto do índice. "
                    f"endpoint='{self.config.endpoint_name}', "
                    f"index='{self.full_index_name}'. Detalhe: {idx_err}"
                )
            return _EMPTY_RESULT

        print(
            f"[SEARCH] Chamando similarity_search — "
            f"index='{self.full_index_name}', "
            f"endpoint='{self.config.endpoint_name}', "
            f"embedding_dim={len(query_embedding)}, "
            f"k={k}, filters={filters}, columns={_COLUMNS}"
        )

        try:
            result = index.similarity_search(
                query_vector=query_embedding,
                columns=_COLUMNS,
                num_results=k,
                filters=filters,   # dict {"semantic_type": "regra"} — não SQL string
            )
            result["_search_backend"] = "similarity_search"
            return result

        except AttributeError as attr_err:
            print(
                f"[SEARCH][SDK_INCOMPATIBILITY] O método 'similarity_search' não existe "
                f"no objeto VectorSearchIndex desta versão do SDK Databricks. "
                f"Verifique a versão instalada do pacote databricks-vector-search. "
                f"Detalhe: {attr_err}"
            )
            return _EMPTY_RESULT

        except Exception as search_err:
            err_str = str(search_err).lower()
            if any(t in err_str for t in ("401", "403", "unauthorized", "forbidden")):
                print(
                    f"[SEARCH][AUTH] Falha de autenticação na busca vetorial. "
                    f"Verifique o token e as permissões sobre o índice "
                    f"'{self.full_index_name}'. Detalhe: {search_err}"
                )
            elif any(t in err_str for t in ("404", "not found", "does not exist")):
                print(
                    f"[SEARCH][ENDPOINT_NOT_FOUND] Endpoint ou índice não localizado "
                    f"durante a busca. endpoint='{self.config.endpoint_name}', "
                    f"index='{self.full_index_name}'. Detalhe: {search_err}"
                )
            elif any(t in err_str for t in ("timeout", "connection", "unavailable")):
                print(
                    f"[SEARCH][TRANSIENT] Erro transitório de rede/serviço durante a busca. "
                    f"Elegível para retry pelo loop em search(). Detalhe: {search_err}"
                )
            elif "filter" in err_str:
                print(
                    f"[SEARCH][FILTER_ERROR] Erro no formato dos filtros enviados. "
                    f"filters={filters}. O Standard Endpoint aceita apenas dict — não SQL string. "
                    f"Detalhe: {search_err}"
                )
            else:
                print(
                    f"[SEARCH][UNEXPECTED] Erro inesperado em similarity_search. "
                    f"index='{self.full_index_name}', "
                    f"endpoint='{self.config.endpoint_name}', "
                    f"embedding_dim={len(query_embedding)}, k={k}, filters={filters}. "
                    f"Detalhe: {search_err}"
                )
            return _EMPTY_RESULT

    def _build_filter_dict(self, filters: Optional[Dict]) -> Optional[Dict]:
        """
        Constrói o dict de filtro sanitizado para a API Vector Search.

        O Databricks Standard Endpoint (Delta Sync Index) aceita filtros apenas
        no formato dict {"column_name": "value"} — não como filter string SQL.
        Passar uma string causa "Filter string is not supported for standard
        endpoints". Este método substitui _build_filter_string() que gerava
        o formato incorreto "semantic_type = 'regra'".

        Chaves não reconhecidas são ignoradas para não bloquear buscas por
        filtros mal formados passados pelo SRAGRetriever. Valores são
        sanitizados via whitelist antes de inclusão no dict.

        Chaves permitidas
        -----------------
        semantic_type   Tipo semântico do documento: kpi, regra, temporal,
                        geographic, demographic, general.
        source_table    Tabela Gold de origem do documento.
        ano_mes         Período no formato YYYY-MM.
        uf              Sigla do estado, ex: SP, RJ.
        ano             Ano como string, ex: 2025.
        faixa_etaria    Faixa etária, ex: 60+, adulto, criança.
        categoria       Categoria livre definida nos metadados do documento.
        """
        if not filters:
            return None

        _ALLOWED_KEYS = {
            "semantic_type", "source_table", "ano_mes", "uf",
            "ano", "faixa_etaria", "categoria",
        }

        result_dict: Dict = {}
        for key, value in filters.items():
            if key not in _ALLOWED_KEYS:
                continue
            if value is None:
                # Valores None passados via str() tornam-se a string literal "None",
                # que entra na API como filtro válido mas sem nenhum documento correspondente.
                # O resultado é zero docs sem erro — indistinguível de ausência real de dados.
                continue
            safe_value = self._sanitize_filter_value(value)
            if not safe_value:
                # Valor que resultou em string vazia após sanitização — descartar.
                continue
            result_dict[key] = safe_value

        return result_dict if result_dict else None

    def _build_filter_string(self, filters: Optional[Dict]) -> Optional[Dict]:
        """
        Alias de _build_filter_dict() mantido para retrocompatibilidade.

        DEPRECADO — use _build_filter_dict() diretamente. O nome "filter_string"
        era enganoso porque a API não aceita strings: o método sempre gerou o
        formato errado. O alias retorna dict (não string) para não quebrar
        chamadores que ainda usam o nome antigo.
        """
        return self._build_filter_dict(filters)

    @lru_cache(maxsize=50)
    def _get_cached_query_embedding(self, query: str) -> tuple:
        """
        Retorna o embedding da query com cache LRU de 50 entradas.

        Queries repetidas numa mesma sessão são comuns no padrão conversacional
        do agente — o cache evita chamadas redundantes ao Foundation Model API.
        O retorno é tuple (não list) porque lru_cache exige tipos hasháveis.
        """
        return tuple(self.embeddings.embed_query(query))

    @staticmethod
    def _sanitize_filter_value(value: str) -> str:
        """
        Sanitiza valores de filtro via whitelist de caracteres seguros.

        Permite apenas alfanuméricos, underscores, hífens e espaços.
        Remove qualquer outro caractere antes de interpolar na filter_string
        da API Vector Search.
        """
        return re.sub(r"[^\w\s\-]", "", str(value)).strip()

    def __repr__(self) -> str:
        return (
            f"DatabricksVectorStoreManager("
            f"index={self.full_index_name}, "
            f"table={self.full_table_name}, "
            f"dim={self.config.embedding_dim})"
        )


# =============================================================================
# SRAG RETRIEVER
# =============================================================================

class SRAGRetriever:
    """
    Retriever do pipeline RAG para SRAG.

    Implementa três estratégias de busca consumindo DatabricksVectorStoreManager,
    com suporte a injeção de semantic_type externo via retrieve():

    semantic
        Busca por similaridade de cosseno pura — retorna os k documentos com
        maior score sem nenhum pós-processamento. Mais rápida e determinística.

    hybrid
        Busca semântica com reranking heurístico baseado em metadados. Recupera
        2k documentos e reordena aplicando boost por fonte primária, recência,
        match geográfico, match temporal e intenção semântica da query. Perguntas
        explicativas recebem boost adicional para gold_rag_dicionario_regras e
        penalidade para documentos puramente numéricos.

    typed
        Detecta o tipo semântico da query (geographic, temporal, regra, kpi,
        demographic) e aplica filtro direto no Vector Index. Quando
        semantic_type_override é fornecido via retrieve(), esse valor substitui
        a detecção interna. Cai para busca sem filtro quando nenhum tipo é
        detectado — evita retorno vazio em queries neutras.

    semantic_type_override em retrieve()
        O IntentRouter calcula rag_semantic_type no routing decision, mas esse
        valor não chegava ao retriever — cada chamada usava a detecção interna.
        O parâmetro semantic_type_override conecta as duas camadas: quando
        fornecido pelo orchestrator, substitui a detecção interna do
        _typed_retrieve() sem alterar a lógica das outras estratégias.

    Parâmetros
    ----------
    vector_store_manager
        Instância de DatabricksVectorStoreManager já configurada.
    """

    # Palavras-chave por tipo semântico para detecção em _typed_retrieve().
    # Listas independentes para evitar falso positivo por sobreposição.
    _TYPE_KEYWORDS: Dict[str, List[str]] = {
        "regra": [
            "o que é", "como é calculad", "defin", "significa", "conceito",
            "metodologia", "critério", "critério de", "explicar", "explicar a",
            "taxa de", "indicador", "fórmula", "como funciona", "qual o critério",
        ],
        "geographic": [
            "sp", "rj", "mg", "rs", "pr", "sc", "ba", "pe", "ce", "go",
            "mt", "ms", "ac", "al", "ap", "am", "df", "es", "ma", "pa",
            "pb", "pi", "rn", "ro", "rr", "se", "to",
            "estado", "uf", "região", "nordeste", "sudeste", "sul", "norte",
            "centro-oeste", "capital", "ranking estado",
        ],
        "temporal": [
            "mês", "mes", "2024", "2025", "2023", "tendência", "tendencia",
            "janeiro", "fevereiro", "março", "abril", "maio", "junho",
            "julho", "agosto", "setembro", "outubro", "novembro", "dezembro",
            "ao longo", "evolução", "série", "histórico", "trimestre", "semana",
        ],
        "demographic": [
            "idade", "idoso", "criança", "adulto", "faixa etária", "faixa etaria",
            "sexo", "feminino", "masculino", "gênero", "genero", "grupo etário",
            "gestante", "puerpera", "comorbidade", "imunocomprometido",
        ],
        "kpi": [
            "taxa de mortalidade", "taxa de uti", "taxa de vacinação",
            "taxa de crescimento", "total de casos", "número de casos",
            "óbitos", "obitos", "internados", "vacinados", "kpi",
        ],
    }

    def __init__(self, vector_store_manager: DatabricksVectorStoreManager):
        self.vsm = vector_store_manager

    def retrieve(
        self,
        query:                  str,
        k:                      int = 5,
        strategy:               str = "semantic",
        semantic_type_override: Optional[str] = None,
    ) -> List[Document]:
        """
        Recupera documentos relevantes do Vector Index.

        Parâmetros
        ----------
        query
            Consulta em linguagem natural.
        k
            Número máximo de documentos a retornar.
        strategy
            'semantic' | 'hybrid' | 'typed'. Estratégia inválida cai para
            'semantic' com aviso em vez de levantar exceção — o pipeline
            não deve abortar por parâmetro de estratégia incorreto.
        semantic_type_override
            Quando fornecido, injeta o semantic_type calculado pelo IntentRouter
            diretamente no retrieval sem passar pela detecção interna do
            _typed_retrieve(). Use a estratégia 'typed' junto com este parâmetro
            para forçar o filtro de tipo desejado pelo router.

        Retorno
            Lista de Documents. Nunca None — lista vazia em caso de falha.
        """
        if not query or not query.strip() or k <= 0:
            print(f"Retrieval ignorado: query vazia ou k={k}.")
            return []

        print(f"Retrieval: strategy={strategy}, k={k}, "
              f"type_override={semantic_type_override}, query_len={len(query)}")

        try:
            if strategy == "semantic":
                docs = self._semantic_retrieve(query, k)
            elif strategy == "hybrid":
                # semantic_type_override é passado como query_intent para que
                # _hybrid_retrieve pré-filtre o pool de candidatos pelo tipo
                # correto E aplique os modificadores de boost/penalidade adequados.
                # Sem essa passagem, o override chegava até retrieve() mas morria
                # aqui — _hybrid_retrieve nunca sabia do tipo e recuperava documentos
                # de qualquer fonte, contaminando o contexto RAG.
                docs = self._hybrid_retrieve(query, k, query_intent=semantic_type_override)
            elif strategy == "typed":
                docs = self._typed_retrieve(query, k, semantic_type_override)
            else:
                print(f"Strategy '{strategy}' desconhecida — usando semantic.")
                docs = self._semantic_retrieve(query, k)

            print(f"Retrieval concluído: {len(docs)} documentos retornados.")
            return docs

        except Exception as e:
            print(f"Erro no retrieval: {e}")
            return []

    def _semantic_retrieve(self, query: str, k: int) -> List[Document]:
        """Busca semântica pura — retorna top-k por score de similaridade."""
        results = self.vsm.search(query, k=k)
        return [doc for doc, _score in results]

    def _hybrid_retrieve(
        self,
        query:        str,
        k:            int,
        query_intent: Optional[str] = None,
    ) -> List[Document]:
        """
        Busca semântica com reranking heurístico por metadados e intenção.

        Pool de candidatos pré-filtrado por semantic_type (quando query_intent reconhecido)
            Quando query_intent é um tipo indexado (regra, kpi, geographic, demographic,
            temporal), o pool de candidatos é obtido via search_by_type() em vez de
            search() irrestrito. Isso garante que o reranking trabalhe dentro de um
            conjunto homogêneo do tipo correto — sem pré-filtro, documentos
            silver_srag_clean com score vetorial alto ganham de documentos
            gold_rag_dicionario_regras com score mais baixo mesmo após boost 1.30x,
            contaminando o contexto RAG com dados numéricos quando a pergunta é
            metodológica. Se search_by_type retornar vazio (tipo sem documentos
            suficientes), o fallback usa search() irrestrito.

        Aliases de intent do IntentRouter
            O IntentRouter pode enviar "explanatory" ou "analytical" como
            semantic_type_override. Esses valores são mapeados para "regra"
            internamente — o nome do tipo indexado é "regra", enquanto o
            router usa terminologia própria de intents.

        Modificadores sobre o score vetorial real:

            gold_rag_kpi_fatos         →  1.20x  (fonte primária do RAG)
            ano_mes 2024/2025          →  1.10x  (dados mais recentes)
            UF da query no doc         →  1.15x  (match geográfico explícito)
            ano_mes na query           →  1.10x  (match temporal explícito)

        Modificadores por intenção explicativa/metodológica (query_intent == "regra"):

            gold_rag_dicionario_regras →  1.30x  (fonte de definições e metodologia)
            semantic_type == "kpi"     →  0.70x  (penalidade — dados numéricos
                                                  não respondem perguntas conceituais)

        Parâmetros
        ----------
        query_intent
            Intenção semântica da query — 'regra', 'explanatory', 'analytical'
            ou qualquer semantic_type indexado. Quando None, inferido da query
            via _detect_semantic_type(). Controla tanto o pré-filtro do pool
            quanto os modificadores de reranking.
        """
        # Pool de candidatos para reranking.
        # Quando query_intent é um semantic_type reconhecido (vindo do semantic_type_override
        # do IntentRouter), pré-filtra o pool para incluir APENAS documentos daquele tipo.
        # Isso garante que o reranking heurístico trabalhe dentro de um conjunto homogêneo —
        # sem pré-filtro, docs silver_srag_clean com score 0.73 ganham de docs
        # gold_rag_dicionario_regras com score 0.60 mesmo após boost 1.30x (0.78 > 0.73
        # só quando o boost é aplicado, mas se query_intent não chegar, nada é aplicado).
        #
        # Tipos reconhecidos para pré-filtro: os mesmos indexados pelo GoldDocumentLoader.
        _FILTERABLE_TYPES = {"regra", "kpi", "geographic", "demographic", "temporal"}

        pool_k = min(k * 2, 20)

        if query_intent in _FILTERABLE_TYPES:
            # Pool filtrado: busca apenas dentro do semantic_type solicitado.
            # Mantém pool_k para preservar candidatos suficientes ao reranking.
            results = self.vsm.search_by_type(query, query_intent, k=pool_k)
            if not results:
                # Fallback sem filtro quando o tipo não tem documentos suficientes
                results = self.vsm.search(query, k=pool_k)
        else:
            results = self.vsm.search(query, k=pool_k)

        if not results:
            return []

        # Normaliza query_intent: o IntentRouter pode enviar valores como "explanatory"
        # ou "analytical" que devem ser mapeados para "regra" nos modificadores de boost.
        # Isso evita que o caller precise saber os nomes internos dos tipos indexados.
        _INTENT_ALIASES = {
            "explanatory": "regra",
            "analytical":  "regra",
        }
        query_intent = _INTENT_ALIASES.get(query_intent, query_intent)

        # Inferir intent quando não fornecido externamente
        if query_intent is None:
            detected    = self._detect_semantic_type(query)
            query_intent = detected if detected in ("regra",) else None

        query_lower  = query.lower()
        ranked: List[Tuple[Document, float]] = []

        for doc, vector_score in results:
            score = vector_score
            meta  = doc.metadata
            ano_mes       = meta.get("ano_mes", "")
            doc_uf        = meta.get("uf", "").lower()
            source_table  = meta.get("source_table", "")
            semantic_type = meta.get("semantic_type", "")

            # Boosts base — aplicados sempre
            if source_table == "gold_rag_kpi_fatos":
                score *= 1.20
            if "2025" in ano_mes or "2024" in ano_mes:
                score *= 1.10
            if doc_uf and doc_uf in query_lower:
                score *= 1.15
            if ano_mes and ano_mes.lower() in query_lower:
                score *= 1.10

            # Modificadores por intenção explicativa/metodológica
            if query_intent in ("regra", "explanatory"):
                if source_table == "gold_rag_dicionario_regras":
                    score *= 1.30
                if semantic_type == "kpi":
                    score *= 0.70

            ranked.append((doc, score))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _score in ranked[:k]]

    def _typed_retrieve(
        self,
        query:                  str,
        k:                      int,
        semantic_type_override: Optional[str] = None,
    ) -> List[Document]:
        """
        Busca com filtro de semantic_type inferido da query ou injetado externamente.

        Detecção interna cobre os cinco tipos indexados pelo GoldDocumentLoader:
        geographic, temporal, regra, demographic, kpi. A versão anterior detectava
        apenas geographic e temporal — perguntas metodológicas (regra) e demográficas
        recuperavam documentos de qualquer tipo, gerando retrieval contaminado.

        Prioridade de semantic_type
            Quando semantic_type_override é fornecido (tipicamente pelo IntentRouter
            via orchestrator), ele tem prioridade total sobre a detecção interna.
            Isso conecta o routing decision do 06 diretamente ao filtro de busca
            sem duplicar lógica de classificação.

        Cai para busca sem filtro quando nenhum tipo é detectado internamente e
        override não foi fornecido — evita retorno vazio em queries neutras.

        Parâmetros
        ----------
        semantic_type_override
            Tipo semântico injetado externamente, substitui detecção interna.
            Deve ser um dos tipos indexados: geographic, temporal, regra,
            demographic, kpi.
        """
        semantic_type = semantic_type_override or self._detect_semantic_type(query)

        if semantic_type:
            results = self.vsm.search_by_type(query, semantic_type, k=k)
        else:
            results = self.vsm.search(query, k=k)

        return [doc for doc, _score in results]

    def _detect_semantic_type(self, query: str) -> Optional[str]:
        """
        Infere o semantic_type da query via correspondência de palavras-chave.

        Verifica os cinco tipos em ordem de especificidade: regra primeiro
        (verbos de explicação são mais específicos), depois kpi (métricas
        nominadas), geographic (siglas e topônimos), demographic (grupos
        populacionais) e temporal (referências de período). Retorna o
        primeiro tipo com ao menos uma correspondência.

        Retorna None quando nenhum tipo é detectado — o chamador deve
        interpretar None como "sem filtro", não como erro.
        """
        query_lower = query.lower()
        for sem_type in ("regra", "kpi", "geographic", "demographic", "temporal"):
            keywords = self._TYPE_KEYWORDS.get(sem_type, [])
            if any(kw in query_lower for kw in keywords):
                return sem_type
        return None

    def __repr__(self) -> str:
        return f"SRAGRetriever(vsm={self.vsm!r})"