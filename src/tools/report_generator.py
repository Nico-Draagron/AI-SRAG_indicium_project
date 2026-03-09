"""
Report Generator — Geração de Relatórios Epidemiológicos em Markdown
====================================================================

Responsabilidade: receber dados já materializados (métricas, geografia,
notícias, gráficos, contexto RAG) e produzir um relatório Markdown estruturado,
usando o LLM apenas onde síntese genuína é necessária.

Decisões de design
------------------
Guardrail contra alucinação no resumo executivo
    O design original chamava o LLM mesmo quando não havia dados reais
    disponíveis. Com contexto vazio, o modelo preenchia o gap com geração
    livre — produzindo estatísticas inventadas ("taxa de hospitalização subiu
    15%") em relatórios epidemiológicos oficiais. O comportamento correto é
    retornar uma seção explicitamente marcada como indisponível quando não há
    dados suficientes para embasar uma síntese. O LLM só é invocado quando
    há pelo menos uma métrica real no contexto.

    O mesmo princípio se aplica às recomendações: quando não há métricas,
    retorna um conjunto mínimo de recomendações de vigilância contínua, sem
    inventar urgências que os dados não sustentam.

Thresholds centralizados em METRIC_THRESHOLDS
    Os critérios de classificação de cada métrica (crescimento, mortalidade,
    UTI, vacinação) estavam hardcoded em quatro métodos separados. Qualquer
    ajuste exigia localizar e editar cada método individualmente. Os thresholds
    agora residem em METRIC_THRESHOLDS — um único dicionário de configuração
    consultado por _classify_metric(). Para alterar o critério de "UTI crítica"
    de 70% para 75%, basta mudar o valor no dicionário.

Threshold relativo em _tendencia()
    O limite fixo de abs(delta) < 0.01 tratava uma variação de 0.009 pp em
    mortalidade da mesma forma que em crescimento — independente da magnitude
    do valor base. O threshold agora é o maior entre 0.1 pp absoluto e 2%
    relativo ao valor atual, tornando a detecção de estabilidade proporcional
    à escala de cada métrica.

Prompt do resumo executivo estruturado e legível
    O design anterior injeta o dict Python bruto como string ({latest}), forçando
    o LLM a parsear chaves snake_case sem contexto semântico. O prompt agora
    serializa os dados em linhas legíveis com rótulos explícitos, instrui sobre
    a estrutura esperada da resposta (achado principal, contexto, limitações) e
    inclui o período de referência quando disponível.

Seção geográfica interpretativa
    A seção anterior exibia apenas a tabela de ranking. Agora calcula a
    concentração geográfica (% dos casos nacionais nos top-N estados) e gera
    uma frase interpretativa quando a concentração é observável. Isso transforma
    a seção de dado tabular em análise utilizável por gestores.

Seção de notícias sem dados técnicos internos
    O campo relevance_score é um artefato interno do WebSearchTool e não tem
    significado para o leitor executivo. Foi removido da exibição. A seção
    agora inclui um parágrafo de enquadramento explicitando que as notícias
    são contexto externo — não dados oficiais — antes da listagem.

Seção de visualizações com categorização
    A listagem de paths completos do DBFS era tecnicamente correta mas
    editorialmente inútil. Os caminhos agora são categorizados por tipo
    (série temporal, geográfico, demográfico, histórico) com base nos
    padrões de nomenclatura gerados pelo ChartTool, e exibidos com nome
    legível sem o path completo de infraestrutura.

Recomendações com contexto RAG e priorização
    O rag_context era consumido apenas no resumo executivo. As recomendações
    agora também recebem o contexto RAG como referência metodológica, e o
    prompt instrui o LLM a separar explicitamente recomendações de prioridade
    alta e média com base nos limiares das métricas recebidas.

Proteção numérica consistente em todas as seções
    _fmt_number() e _fmt_float() são usados em todos os pontos de
    interpolação numérica para evitar crash quando campos retornam None
    ou "N/A" do dict.get().

Extração segura do nome do LLM
    _resolve_llm_name() testa model_name e model_id em sequência, extrai
    apenas o segmento final do path quando detecta uma URL (evitando expor
    infraestrutura interna), e usa o nome da classe como fallback definitivo.

Filtragem de seções vazias antes do join
    Seções com content="" são filtradas em generate_report() antes do join
    para evitar separadores em branco duplos no markdown final.

Distinção entre "sem dados" e "dados vazios"
    _extract_periods() verifica se o primeiro elemento contém ao menos uma
    chave com valor não-None antes de tratá-lo como período válido — evitando
    que metrics={"data": [{}]} seja interpretado como dado real.

Versão do sistema injetável
    A versão é passada no construtor com um valor default explícito, permitindo
    que o notebook ou o agente injete a versão real sem alterar o código.
"""

import os
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

try:
    from src.utils.audit import AuditLogger, AuditEvent, EventStatus
except ImportError:
    class AuditEvent:
        REPORT_GENERATION_START = "report_generation_start"
        REPORT_GENERATED        = "report_generated"
        REPORT_SECTION_SKIPPED  = "report_section_skipped"
        REPORT_LLM_ERROR        = "report_llm_error"

    class EventStatus:
        INFO    = "INFO"
        SUCCESS = "SUCCESS"
        WARNING = "WARNING"
        ERROR   = "ERROR"

    class AuditLogger:
        def log_event(self, event_type, details=None, status="INFO"):
            print(f"[{status}] {event_type}: {details}")


# =============================================================================
# CONFIGURAÇÃO DE THRESHOLDS
# =============================================================================

# Critérios de classificação por métrica.
# Cada entrada é uma lista de tuplas (limite_inferior, rótulo, emoji_de_alerta).
# Os níveis são avaliados de cima para baixo: o primeiro cujo limite_inferior
# seja menor que o valor é retornado. O último elemento deve ter
# limite_inferior negativo grande para funcionar como fallback definitivo.
#
# Para ajustar um critério — ex: considerar UTI crítica acima de 75% em vez
# de 70% — basta alterar o valor correspondente neste dicionário.
METRIC_THRESHOLDS: Dict[str, List[Tuple]] = {
    "taxa_crescimento": [
        (10.0,  "Crescimento expressivo",                     "🔴"),
        (3.0,   "Crescimento moderado",                       "🟡"),
        (0.0,   "Crescimento leve",                           "🟢"),
        (-999,  "Redução no número de casos",                 "🟢"),
    ],
    "taxa_mortalidade": [
        (10.0,  "Taxa elevada — atenção imediata requerida",  "🔴"),
        (5.0,   "Taxa em nível moderado",                     "🟡"),
        (-999,  "Taxa em patamar controlado",                 "🟢"),
    ],
    "taxa_uti": [
        (70.0,  "Pressão crítica no sistema de saúde",        "🔴"),
        (50.0,  "Nível que requer monitoramento intensivo",   "🟡"),
        (-999,  "Nível controlável",                          "🟢"),
    ],
    "taxa_vacinacao": [
        (70.0,  "Cobertura satisfatória",                     "🟢"),
        (50.0,  "Cobertura em expansão, pode melhorar",       "🟡"),
        (-999,  "Cobertura abaixo do ideal",                  "🔴"),
    ],
}

# Padrões de nomenclatura de arquivo para categorização de gráficos.
# A ordem importa: o primeiro padrão encontrado no nome do arquivo é usado.
CHART_CATEGORIES: List[Tuple[str, str]] = [
    ("serie",    "Série Temporal"),
    ("evolucao", "Série Temporal"),
    ("diaria",   "Série Diária"),
    ("geo",      "Distribuição Geográfica"),
    ("uf",       "Distribuição por Estado"),
    ("demo",     "Perfil Demográfico"),
    ("faixa",    "Distribuição por Faixa Etária"),
    ("hist",     "Série Histórica"),
    ("anual",    "Comparativo Anual"),
    ("mensal",   "Comparativo Mensal"),
]


# =============================================================================
# MODELO DE SEÇÃO
# =============================================================================

@dataclass
class ReportSection:
    """
    Unidade de composição do relatório.

    Seções com content="" são filtradas em generate_report() antes do join
    para evitar separadores em branco duplos no markdown final.
    """
    title:   str
    content: str
    order:   int


# =============================================================================
# REPORT GENERATOR
# =============================================================================

class ReportGenerator:
    """
    Gera relatório epidemiológico em Markdown a partir de dados já
    materializados pelo orquestrador.

    O LLM é usado apenas em dois pontos: resumo executivo (síntese de dados
    reais) e recomendações (interpretação contextualizada das métricas). Em
    ambos os casos, a chamada é condicionada à presença de dados válidos —
    contexto vazio não é enviado ao modelo.

    Aceita qualquer BaseChatModel compatível com LangChain: ChatOpenAI,
    ChatDatabricks, ChatAnthropic, etc.

    Parâmetros
    ----------
    llm
        Modelo de linguagem usado para síntese e recomendações.
    audit
        Instância de AuditLogger. Quando None, usa o stub local.
    system_version
        Versão do sistema exibida no cabeçalho. Deve ser injetada pelo
        chamador para refletir a versão real em execução.
    """

    def __init__(
        self,
        llm:            BaseChatModel,
        audit:          Optional[AuditLogger] = None,
        system_version: str = "3.1.0",
    ):
        self.llm            = llm
        self.audit          = audit or AuditLogger()
        self.system_version = system_version
        self.llm_name       = self._resolve_llm_name()

    # =========================================================================
    # INTERFACE PÚBLICA
    # =========================================================================

    def generate_report(
        self,
        metrics:     Optional[Dict] = None,
        geographic:  Optional[Dict] = None,
        news:        Optional[Dict] = None,
        charts:      Optional[List[str]] = None,
        rag_context: Optional[Dict] = None,
        user_query:  str = "Gerar relatório SRAG",
    ) -> str:
        """
        Monta o relatório completo a partir dos dados fornecidos.

        Seções condicionais (geográfica, notícias) são incluídas apenas quando
        há dados. Seções obrigatórias (cabeçalho, métricas, gráficos, rodapé)
        sempre aparecem, com mensagens explícitas de indisponibilidade quando
        os dados estão ausentes. Seções com conteúdo vazio são filtradas antes
        do join para não introduzir separadores em branco no documento final.

        Parâmetros
        ----------
        metrics
            Dict com chave "data": List[Dict]. Cada dict deve conter as
            chaves taxa_crescimento, taxa_mortalidade, taxa_uti, taxa_vacinacao,
            total_casos e, opcionalmente, data_referencia. O primeiro elemento
            é o período mais recente; o segundo, o anterior para comparação.
        geographic
            Dict com chave "data": List[Dict], cada um com sg_uf, total_casos,
            taxa_mortalidade.
        news
            Dict com chave "articles": List[Dict] no formato retornado pelo
            WebSearchTool (campo canônico "articles", não "news").
        charts
            Lista de paths dos arquivos HTML gerados pelo ChartTool.
        rag_context
            Dict com chave "answer" contendo contexto adicional do RAG.
        user_query
            Query original do usuário, usada como contexto para o LLM.
        """
        self.audit.log_event(
            AuditEvent.REPORT_GENERATION_START,
            {
                "has_metrics":    metrics is not None,
                "has_geographic": geographic is not None,
                "has_news":       news is not None,
                "charts_count":   len(charts) if charts else 0,
                "has_rag":        rag_context is not None,
            },
            EventStatus.INFO,
        )

        latest, previous = self._extract_periods(metrics)

        sections = [
            self._build_header(latest),
            self._build_executive_summary(latest, previous, news, rag_context, user_query),
            self._build_metrics_section(latest, previous),
            self._build_geographic_section(geographic, latest),
            self._build_news_section(news),
            self._build_charts_section(charts),
            self._build_recommendations(latest, previous, news, rag_context),
            self._build_footer(),
        ]

        non_empty = [s for s in sorted(sections, key=lambda s: s.order) if s.content.strip()]
        report_md = "\n\n".join(s.content for s in non_empty)

        self.audit.log_event(
            AuditEvent.REPORT_GENERATED,
            {"sections": len(non_empty), "length": len(report_md)},
            EventStatus.SUCCESS,
        )

        return report_md

    # =========================================================================
    # SEÇÕES
    # =========================================================================

    def _build_header(self, latest: Dict) -> ReportSection:
        """
        Cabeçalho do relatório com data de geração e período de referência
        dos dados quando disponível.

        O campo data_referencia é extraído de latest quando presente —
        indicando explicitamente a qual período os dados se referem, que
        pode divergir significativamente da data de geração do relatório.
        """
        now      = datetime.now()
        data_ref = latest.get("data_referencia") if latest else None
        periodo_linha = (
            f"**Período de Referência dos Dados:** {data_ref}  \n"
            if data_ref else ""
        )

        content = (
            f"# Relatório Epidemiológico SRAG — {now.strftime('%B %Y')}\n\n"
            f"**Data de Geração:** {now.strftime('%d/%m/%Y às %H:%M')}  \n"
            f"{periodo_linha}"
            f"**Sistema:** Agente de Monitoramento Epidemiológico  \n"
            f"**Versão:** {self.system_version}  \n\n"
            f"---"
        )
        return ReportSection(title="header", content=content, order=1)

    def _build_executive_summary(
        self,
        latest:      Dict,
        previous:    Dict,
        news:        Optional[Dict],
        rag_context: Optional[Dict],
        user_query:  str,
    ) -> ReportSection:
        """
        Gera o resumo executivo via LLM apenas quando há dados reais.

        A ausência de dados produz uma seção explicitamente marcada como
        indisponível, em vez de invocar o LLM com contexto vazio — o que
        levaria o modelo a preencher o gap com estatísticas inventadas.

        O prompt injeta os dados em formato legível com rótulos explícitos
        em português, instrui sobre a estrutura esperada (achado principal →
        contexto → limitações) e separa claramente o que são dados oficiais
        do que é contexto externo (notícias, RAG).
        """
        context_parts = []

        if latest:
            context_parts.append(self._serialize_metrics_for_prompt(latest, previous))

        articles = (news or {}).get("articles", [])
        if articles:
            titulos = "; ".join(a.get("title", "sem título") for a in articles[:3])
            context_parts.append(
                f"Contexto externo (notícias recentes — não são dados oficiais): {titulos}."
            )

        if rag_context and rag_context.get("answer"):
            context_parts.append(
                f"Contexto metodológico (base de conhecimento RAG): "
                f"{rag_context['answer'][:400]}"
            )

        if not context_parts:
            self.audit.log_event(
                AuditEvent.REPORT_SECTION_SKIPPED,
                {"section": "executive_summary", "reason": "sem dados para síntese"},
                EventStatus.WARNING,
            )
            content = (
                "## Resumo Executivo\n\n"
                "> ⚠️ **Dados insuficientes para gerar resumo executivo.**  \n"
                "> Verifique a disponibilidade das métricas obrigatórias no pipeline."
            )
            return ReportSection(title="executive_summary", content=content, order=2)

        # Informa o LLM quais blocos estão ausentes para não inventar dados
        dados_ausentes = []
        if not latest:
            dados_ausentes.append("métricas epidemiológicas")
        if not articles:
            dados_ausentes.append("notícias recentes")
        if not rag_context:
            dados_ausentes.append("contexto RAG")

        aviso_ausencia = (
            f"\nBlocos de dados ausentes nesta execução: {', '.join(dados_ausentes)}. "
            "Não mencione esses blocos no resumo.\n"
            if dados_ausentes else ""
        )

        prompt = (
            "Você é um epidemiologista sênior. Escreva um resumo executivo objetivo "
            "sobre a situação atual de SRAG com base EXCLUSIVAMENTE nos dados abaixo.\n\n"
            "ESTRUTURA OBRIGATÓRIA (três parágrafos curtos):\n"
            "1. Achado principal: o indicador mais crítico do momento e sua magnitude.\n"
            "2. Contexto: tendência em relação ao período anterior e fatores relevantes.\n"
            "3. Limitações: o que não pode ser afirmado com os dados disponíveis.\n\n"
            "REGRAS:\n"
            "- Não invente números. Use apenas os valores fornecidos.\n"
            "- Se um dado estiver ausente, omita-o — não estime.\n"
            "- Diferencie explicitamente dados oficiais de contexto externo (notícias/RAG).\n"
            f"{aviso_ausencia}\n"
            "DADOS DISPONÍVEIS:\n"
            + "\n".join(context_parts)
            + f"\n\nConsulta original do usuário: {user_query}"
        )

        summary_text = self._invoke_llm(prompt, section="executive_summary")
        content      = f"## Resumo Executivo\n\n{summary_text}"
        return ReportSection(title="executive_summary", content=content, order=2)

    def _build_metrics_section(self, latest: Dict, previous: Dict) -> ReportSection:
        """
        Seção das quatro métricas obrigatórias com classificação e tendência.

        Cada métrica é apresentada com: valor atual, classificação pelo nível
        de alerta (via METRIC_THRESHOLDS), comparação com período anterior e
        análise de tendência. Quando latest está vazio, a seção indica
        indisponibilidade explicitamente em vez de exibir N/A em todos os campos.
        """
        if not latest:
            content = (
                "## Métricas Epidemiológicas\n\n"
                "> ⚠️ **Dados de métricas não disponíveis.**  \n"
                "> Verifique a execução do nó de métricas obrigatórias no orquestrador."
            )
            return ReportSection(title="metrics", content=content, order=3)

        total_casos = self._fmt_number(latest.get("total_casos"))
        data_ref    = latest.get("data_referencia", "período não informado")

        content = (
            f"## Métricas Epidemiológicas Principais\n\n"
            f"*Total de casos no período: **{total_casos}** — referência: {data_ref}*\n\n"
            "---\n\n"

            "### 1. Taxa de Crescimento de Casos\n\n"
            f"{self._render_metric_block('taxa_crescimento', latest, previous, inverso=False)}\n\n"
            "---\n\n"

            "### 2. Taxa de Mortalidade\n\n"
            f"{self._render_metric_block('taxa_mortalidade', latest, previous, inverso=True)}\n\n"
            "---\n\n"

            "### 3. Taxa de Ocupação de UTI\n\n"
            f"{self._render_metric_block('taxa_uti', latest, previous, inverso=True)}\n\n"
            "---\n\n"

            "### 4. Cobertura Vacinal\n\n"
            f"{self._render_metric_block('taxa_vacinacao', latest, previous, inverso=False)}"
        )

        return ReportSection(title="metrics", content=content, order=3)

    def _build_geographic_section(
        self,
        geographic: Optional[Dict],
        latest:     Dict,
    ) -> ReportSection:
        """
        Seção dos estados mais afetados com análise de concentração geográfica.

        Além do ranking tabular, calcula a proporção dos casos dos top estados
        em relação ao total nacional (quando disponível em latest) e gera uma
        frase interpretativa sobre concentração regional. Destaca estados com
        mortalidade acima do limiar moderado. Retorna seção vazia quando não há
        dados, excluindo-a do documento final via filtragem em generate_report().
        """
        if not geographic or not geographic.get("data"):
            return ReportSection(title="geographic", content="", order=4)

        rows = geographic["data"][:5]

        total_nac_f = None
        try:
            total_nac   = latest.get("total_casos") if latest else None
            total_nac_f = float(total_nac) if total_nac is not None else None
        except (TypeError, ValueError):
            pass

        # Tabela de ranking
        linhas_tabela = [
            "| Ranking | UF | Casos | Mortalidade |\n",
            "|---------|-----|-------|-------------|\n",
        ]
        soma_top = 0.0
        for idx, row in enumerate(rows, 1):
            sg_uf     = row.get("sg_uf", "N/A")
            total_uf  = row.get("total_casos")
            taxa_mort = self._fmt_float(row.get("taxa_mortalidade"))
            linhas_tabela.append(
                f"| {idx} | {sg_uf} | {self._fmt_number(total_uf)} | {taxa_mort}% |\n"
            )
            try:
                soma_top += float(total_uf or 0)
            except (TypeError, ValueError):
                pass

        # Interpretação de concentração geográfica
        interpretacao = ""
        if total_nac_f and total_nac_f > 0 and soma_top > 0:
            pct_top = (soma_top / total_nac_f) * 100
            n_est   = len(rows)
            if pct_top >= 60:
                interpretacao = (
                    f"\n> **Concentração geográfica elevada:** os {n_est} estados listados "
                    f"respondem por **{pct_top:.1f}%** dos casos nacionais, indicando "
                    f"distribuição heterogênea com foco regional predominante."
                )
            elif pct_top >= 40:
                interpretacao = (
                    f"\n> Os {n_est} estados listados concentram **{pct_top:.1f}%** dos casos "
                    f"nacionais — distribuição moderadamente concentrada."
                )
            else:
                interpretacao = (
                    f"\n> Os {n_est} estados listados somam **{pct_top:.1f}%** dos casos "
                    f"nacionais, sugerindo distribuição relativamente homogênea no território."
                )

        alerta_uf = self._detect_combined_alert(rows)

        content = (
            "## Análise Geográfica\n\n"
            "### Estados com Maior Carga de Casos\n\n"
            + "".join(linhas_tabela)
            + interpretacao
            + alerta_uf
            + "\n\n> *Os dados geográficos refletem notificações consolidadas no SIVEP-Gripe. "
            "Estados com sistemas de notificação menos estruturados podem apresentar "
            "subnotificação sistemática.*"
        )
        return ReportSection(title="geographic", content=content, order=4)

    def _build_news_section(self, news: Optional[Dict]) -> ReportSection:
        """
        Seção de contexto de notícias externas.

        Exibe título, fonte, data e snippet de cada artigo sem expor dados
        técnicos internos (relevance_score) ao leitor. Um parágrafo de
        enquadramento explicita que as notícias são contexto externo e não
        substituem os dados oficiais do SIVEP-Gripe. Retorna seção vazia quando
        não há artigos, excluindo-a do documento final.
        """
        articles = (news or {}).get("articles", [])
        if not articles:
            return ReportSection(title="news", content="", order=5)

        top   = articles[:5]
        lines = [
            "## Contexto Externo — Notícias Recentes\n\n",
            "> As informações abaixo são provenientes de fontes jornalísticas e não "
            "substituem os dados oficiais do SIVEP-Gripe. Utilize-as como contexto "
            "complementar, não como verdade factual primária.\n\n",
            f"Foram identificados **{len(top)}** artigos relevantes "
            f"sobre SRAG no período analisado:\n\n",
        ]

        for idx, article in enumerate(top, 1):
            title   = article.get("title",          "Título não disponível")
            source  = article.get("source",         "Fonte não identificada")
            date    = article.get("published_date", "Data não informada")
            url     = article.get("url",            "")
            snippet = article.get("snippet") or article.get("description") or ""

            link_txt     = f"[Ver artigo]({url})" if url else ""
            snippet_linha = (
                f"\n   *{snippet[:200].strip()}{'...' if len(snippet) > 200 else ''}*"
                if snippet else ""
            )

            lines.append(
                f"**{idx}. {title}**  \n"
                f"   📰 {source} — {date}  {link_txt}"
                f"{snippet_linha}\n\n"
            )

        return ReportSection(title="news", content="".join(lines), order=5)

    def _build_charts_section(self, charts: Optional[List[str]]) -> ReportSection:
        """
        Seção de visualizações com categorização por tipo de gráfico.

        Em vez de listar paths completos do DBFS — que expõem estrutura de
        infraestrutura e não têm significado para o leitor executivo — os
        gráficos são agrupados por categoria inferida do padrão de nomenclatura
        do arquivo, exibindo apenas o nome legível sem o diretório.
        """
        num = len(charts) if charts else 0

        if not charts:
            content = (
                "## Visualizações\n\n"
                "Nenhum gráfico foi gerado nesta execução."
            )
            return ReportSection(title="charts", content=content, order=6)

        categorized: Dict[str, List[str]] = {}
        for path in charts:
            nome      = os.path.basename(path)
            categoria = self._categorize_chart(nome)
            categorized.setdefault(categoria, []).append(nome)

        lines = [f"## Visualizações\n\n**{num} gráfico(s) gerado(s)** nesta execução:\n\n"]

        for categoria, nomes in categorized.items():
            lines.append(f"**{categoria}**\n")
            for nome in nomes:
                lines.append(f"- `{nome}`\n")
            lines.append("\n")

        lines.append(
            "> *Os arquivos estão disponíveis no diretório de saída configurado no pipeline.*"
        )

        return ReportSection(title="charts", content="".join(lines), order=6)

    def _build_recommendations(
        self,
        latest:      Dict,
        previous:    Dict,
        news:        Optional[Dict],
        rag_context: Optional[Dict],
    ) -> ReportSection:
        """
        Gera recomendações via LLM contextualizadas com as métricas reais.

        O rag_context — anteriormente consumido apenas no resumo executivo —
        é incluído aqui como referência metodológica, enriquecendo as
        recomendações com evidências da base de conhecimento sem permitir que
        o LLM invente dados numéricos não presentes nos dados oficiais.

        O prompt instrui separação explícita entre prioridade alta e média,
        com justificativa por métrica. Quando não há métricas, retorna
        recomendações mínimas de vigilância sem inventar urgências.
        """
        if not latest:
            content = (
                "## Recomendações\n\n"
                "*Base de dados insuficiente para recomendações específicas. "
                "Recomendações mínimas de vigilância:*\n\n"
                "1. **Vigilância Epidemiológica** — Manter notificação compulsória de casos "
                "e análise de tendências semanais.\n"
                "2. **Monitoramento de Capacidade** — Acompanhar disponibilidade de leitos UTI "
                "e acionar plano de contingência se necessário.\n"
            )
            return ReportSection(title="recommendations", content=content, order=7)

        metricas_prompt = self._serialize_metrics_for_prompt(latest, previous)

        articles = (news or {}).get("articles", [])
        news_ctx = (
            f"\nContexto de notícias: {len(articles)} artigos recentes — "
            f"temas: {'; '.join(a.get('title', '')[:60] for a in articles[:3])}."
            if articles else ""
        )

        rag_ctx = ""
        if rag_context and rag_context.get("answer"):
            rag_ctx = (
                f"\nBase de conhecimento metodológico (RAG): "
                f"{rag_context['answer'][:400]}"
            )

        cenario = self._assess_overall_scenario(latest)

        prompt = (
            "Você é um gestor sênior de saúde pública. Com base nos dados abaixo, "
            "gere recomendações objetivas e específicas para o período atual.\n\n"
            f"CENÁRIO GERAL: {cenario}\n\n"
            "ESTRUTURA OBRIGATÓRIA:\n"
            "**Prioridade Alta** (ação imediata — justifique pela métrica mais crítica):\n"
            "- [máximo 2 itens, somente se houver indicador em nível crítico]\n\n"
            "**Prioridade Média** (ação no ciclo atual):\n"
            "- [3 a 4 itens baseados nas métricas fornecidas]\n\n"
            "REGRAS:\n"
            "- Cada recomendação deve citar a métrica que a justifica.\n"
            "- Não inclua recomendações genéricas válidas para qualquer cenário.\n"
            "- Se nenhum indicador estiver em nível crítico, omita 'Prioridade Alta'.\n"
            "- Não invente dados numéricos além dos fornecidos.\n\n"
            f"DADOS EPIDEMIOLÓGICOS:\n{metricas_prompt}"
            f"{news_ctx}"
            f"{rag_ctx}"
        )

        rec_text = self._invoke_llm(prompt, section="recommendations")
        content  = f"## Recomendações\n\n{rec_text}"
        return ReportSection(title="recommendations", content=content, order=7)

    def _build_footer(self) -> ReportSection:
        content = (
            "---\n\n"
            "## Informações Técnicas\n\n"
            "- **Fonte de Dados:** SIVEP-Gripe via Databricks Gold Layer\n"
            "- **Metodologia:** Arquitetura Medallion (Bronze → Silver → Gold)\n"
            "- **Sistema:** Agente Orquestrador com LangGraph\n"
            f"- **LLM:** {self.llm_name}\n"
            "- **Geração:** Automatizada via AI Agent\n\n"
            "---\n\n"
            f"*Relatório gerado em {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} "
            "pelo Sistema de Monitoramento Epidemiológico SRAG.*"
        )
        return ReportSection(title="footer", content=content, order=8)

    # =========================================================================
    # HELPERS EDITORIAIS
    # =========================================================================

    def _render_metric_block(
        self,
        metric_key: str,
        latest:     Dict,
        previous:   Dict,
        inverso:    bool,
    ) -> str:
        """
        Renderiza o bloco completo de uma métrica: valor atual, classificação
        por nível de alerta e análise de tendência vs período anterior.

        Centraliza a lógica de apresentação de métricas para evitar duplicação
        entre os quatro blocos de _build_metrics_section().

        Parâmetros
        ----------
        metric_key
            Chave do dict de métricas (ex: "taxa_crescimento").
        latest
            Dict do período mais recente.
        previous
            Dict do período anterior. Pode ser vazio — nesse caso a tendência
            declara indisponibilidade em vez de comparar.
        inverso
            Quando True, aumento é tratado como piora (mortalidade, UTI).
            Quando False, aumento é melhora (vacinação) ou neutro (crescimento).
        """
        valor_atual = latest.get(metric_key)
        valor_ant   = previous.get(metric_key) if previous else None

        valor_fmt = self._fmt_float(valor_atual)
        ant_fmt   = self._fmt_float(valor_ant)

        try:
            emoji, nivel = self._classify_metric(metric_key, float(valor_atual or 0))
        except (TypeError, ValueError):
            emoji, nivel = "⚪", "Dado não disponível para classificação"

        comparativo = (
            f"- **Período anterior:** {ant_fmt}%\n"
            if ant_fmt != "N/A" else
            "- **Período anterior:** não disponível\n"
        )

        try:
            label     = metric_key.replace("taxa_", "").replace("_", " ")
            tendencia = self._tendencia(
                atual=float(valor_atual or 0),
                anterior=float(valor_ant) if valor_ant is not None else None,
                label=label,
                inverso=inverso,
            )
        except (TypeError, ValueError):
            tendencia = "Dados insuficientes para análise de tendência."

        return (
            f"- **Valor atual:** {valor_fmt}%\n"
            f"{comparativo}"
            f"- **Classificação:** {emoji} {nivel}\n\n"
            f"**Análise:** {tendencia}"
        )

    def _classify_metric(self, metric_key: str, valor: float) -> Tuple[str, str]:
        """
        Retorna (emoji, rótulo) para um valor de métrica consultando
        METRIC_THRESHOLDS.

        Itera os níveis de cima para baixo — o primeiro cujo limite_inferior
        seja menor que o valor é retornado. O último nível de cada métrica
        deve ter limite_inferior negativo grande para garantir fallback.
        """
        levels = METRIC_THRESHOLDS.get(metric_key, [])
        for limite, rotulo, emoji in levels:
            if valor > limite:
                return emoji, rotulo
        return "⚪", "Classificação indisponível"

    def _assess_overall_scenario(self, latest: Dict) -> str:
        """
        Gera uma frase de cenário geral baseada nas métricas mais críticas.

        Usada no prompt de recomendações para contextualizar o tom da resposta
        do LLM. Prioriza indicadores de maior impacto imediato (UTI e
        mortalidade) sobre os de longo prazo (vacinação).
        """
        try:
            uti   = float(latest.get("taxa_uti",         0) or 0)
            mort  = float(latest.get("taxa_mortalidade", 0) or 0)
            cresc = float(latest.get("taxa_crescimento", 0) or 0)
            vac   = float(latest.get("taxa_vacinacao",   0) or 0)

            alertas = []
            if uti   > METRIC_THRESHOLDS["taxa_uti"][0][0]:
                alertas.append("pressão crítica em UTI")
            if mort  > METRIC_THRESHOLDS["taxa_mortalidade"][0][0]:
                alertas.append("mortalidade elevada")
            if cresc > METRIC_THRESHOLDS["taxa_crescimento"][0][0]:
                alertas.append("crescimento acelerado de casos")
            if vac   < METRIC_THRESHOLDS["taxa_vacinacao"][2][0]:
                alertas.append("cobertura vacinal insuficiente")

            if alertas:
                return f"ATENÇÃO — {', '.join(alertas)}."
            if (uti  > METRIC_THRESHOLDS["taxa_uti"][1][0]
                    or mort > METRIC_THRESHOLDS["taxa_mortalidade"][1][0]):
                return "Monitoramento intensificado — indicadores em nível moderado."
            return "Situação sob controle — vigilância de rotina."

        except (TypeError, ValueError):
            return "Cenário indeterminado por dados insuficientes."

    def _detect_combined_alert(self, rows: List[Dict]) -> str:
        """
        Identifica estados com mortalidade acima do limiar moderado,
        gerando um destaque editorial quando presente.

        Retorna string vazia quando nenhum estado atinge o critério, evitando
        alertas falsos em cenários controlados.
        """
        threshold_mort = METRIC_THRESHOLDS["taxa_mortalidade"][1][0]
        alertas = []
        for row in rows:
            uf   = row.get("sg_uf", "N/A")
            mort = row.get("taxa_mortalidade")
            try:
                if float(mort or 0) > threshold_mort:
                    alertas.append(f"{uf} ({self._fmt_float(mort)}%)")
            except (TypeError, ValueError):
                pass

        if not alertas:
            return ""
        return (
            f"\n\n> ⚠️ **Estados com mortalidade acima do limiar moderado "
            f"({threshold_mort}%):** "
            + ", ".join(alertas)
            + ". Recomenda-se análise detalhada da capacidade hospitalar nessas regiões."
        )

    def _categorize_chart(self, filename: str) -> str:
        """
        Infere a categoria de um gráfico a partir do padrão de nomenclatura
        do arquivo gerado pelo ChartTool.

        Consulta CHART_CATEGORIES em ordem: o primeiro padrão encontrado no
        nome do arquivo (case-insensitive) determina a categoria. Retorna
        "Outros Gráficos" quando nenhum padrão é reconhecido.
        """
        nome_lower = filename.lower()
        for padrao, categoria in CHART_CATEGORIES:
            if padrao in nome_lower:
                return categoria
        return "Outros Gráficos"

    def _serialize_metrics_for_prompt(self, latest: Dict, previous: Dict) -> str:
        """
        Serializa os dicts de métricas em texto legível para injeção em prompts.

        Em vez de passar a representação __repr__ do dict Python, gera linhas
        formatadas com rótulos explícitos em português — reduzindo o risco de
        interpretação incorreta pelo LLM e tornando o contexto injetado auditável.
        """
        data_ref = latest.get("data_referencia", "não informado")

        def fmt_par(label: str, key: str, unit: str = "%") -> str:
            atual = self._fmt_float(latest.get(key))
            ant   = self._fmt_float(previous.get(key)) if previous else "N/A"
            comp  = f" (anterior: {ant}{unit})" if ant != "N/A" else ""
            return f"- {label}: {atual}{unit}{comp}"

        linhas = [
            f"Período de referência: {data_ref}",
            f"- Total de casos: {self._fmt_number(latest.get('total_casos'))}",
            fmt_par("Taxa de crescimento", "taxa_crescimento"),
            fmt_par("Taxa de mortalidade", "taxa_mortalidade"),
            fmt_par("Ocupação de UTI",     "taxa_uti"),
            fmt_par("Cobertura vacinal",   "taxa_vacinacao"),
        ]
        return "\n".join(linhas)

    # =========================================================================
    # ANÁLISE COMPARATIVA DE MÉTRICAS (mantidos para compatibilidade)
    # =========================================================================

    def _analyze_growth(self, latest: Dict, previous: Dict) -> str:
        """
        Interpreta a taxa de crescimento atual e sua tendência em relação
        ao período anterior. Mantido para compatibilidade com chamadores
        externos — internamente usa _classify_metric().
        """
        try:
            atual    = float(latest.get("taxa_crescimento", 0) or 0)
            anterior = float(previous.get("taxa_crescimento") or 0) if previous else None
            _, nivel = self._classify_metric("taxa_crescimento", atual)
            tendencia = self._tendencia(atual, anterior, label="crescimento")
            return f"{nivel} ({atual:.1f}%). {tendencia}"
        except (TypeError, ValueError):
            return "Dados insuficientes para análise de tendência de crescimento."

    def _analyze_mortality(self, latest: Dict, previous: Dict) -> str:
        """
        Interpreta a taxa de mortalidade atual e sua tendência.
        Mantido para compatibilidade com chamadores externos.
        """
        try:
            atual    = float(latest.get("taxa_mortalidade", 0) or 0)
            anterior = float(previous.get("taxa_mortalidade") or 0) if previous else None
            _, nivel = self._classify_metric("taxa_mortalidade", atual)
            tendencia = self._tendencia(atual, anterior, label="mortalidade", inverso=True)
            return f"{nivel} ({atual:.1f}%). {tendencia}"
        except (TypeError, ValueError):
            return "Dados insuficientes para análise de mortalidade."

    def _analyze_uti(self, latest: Dict, previous: Dict) -> str:
        """
        Interpreta a taxa de ocupação de UTI e sua tendência.
        Mantido para compatibilidade com chamadores externos.
        """
        try:
            atual    = float(latest.get("taxa_uti", 0) or 0)
            anterior = float(previous.get("taxa_uti") or 0) if previous else None
            _, nivel = self._classify_metric("taxa_uti", atual)
            tendencia = self._tendencia(atual, anterior, label="ocupação de UTI", inverso=True)
            return f"{nivel} ({atual:.1f}%). {tendencia}"
        except (TypeError, ValueError):
            return "Dados insuficientes para análise de ocupação de UTI."

    def _analyze_vaccination(self, latest: Dict, previous: Dict) -> str:
        """
        Interpreta a taxa de cobertura vacinal e sua tendência.
        Mantido para compatibilidade com chamadores externos.
        """
        try:
            atual    = float(latest.get("taxa_vacinacao", 0) or 0)
            anterior = float(previous.get("taxa_vacinacao") or 0) if previous else None
            _, nivel = self._classify_metric("taxa_vacinacao", atual)
            tendencia = self._tendencia(atual, anterior, label="cobertura vacinal", inverso=False)
            return f"{nivel} ({atual:.1f}%). {tendencia}"
        except (TypeError, ValueError):
            return "Dados insuficientes para análise de cobertura vacinal."

    # =========================================================================
    # UTILITÁRIOS INTERNOS
    # =========================================================================

    def _extract_periods(
        self,
        metrics: Optional[Dict],
    ) -> Tuple[Dict, Dict]:
        """
        Extrai latest e previous da estrutura de métricas.

        Um dict vazio — que ocorre quando mandatory_metrics falha e o
        orquestrador passa metrics={"data": [{}]} — é tratado da mesma
        forma que ausência de dados. A guarda "if data else {}" do design
        original não detectava esse caso, produzindo relatórios com N/A
        em todos os campos sem nenhuma indicação de falha upstream.
        """
        if not metrics or not metrics.get("data"):
            return {}, {}

        data     = metrics["data"]
        latest   = data[0] if data else {}
        previous = data[1] if len(data) > 1 else {}

        has_valid = any(
            v is not None and v != "N/A"
            for v in latest.values()
        ) if latest else False

        if not has_valid:
            self.audit.log_event(
                AuditEvent.REPORT_SECTION_SKIPPED,
                {"reason": "metrics['data'][0] existe mas todos os campos são None ou N/A"},
                EventStatus.WARNING,
            )
            return {}, {}

        return latest, previous

    def _tendencia(
        self,
        atual:    float,
        anterior: Optional[float],
        label:    str,
        inverso:  bool = False,
    ) -> str:
        """
        Gera uma frase de tendência comparando o valor atual com o anterior.

        O threshold de estabilidade é relativo: considera-se estável quando
        a variação absoluta é menor que 0.1 pp OU menor que 2% relativo ao
        valor atual — o que for maior. Isso evita que variações de 0.009 pp
        em métricas de alta magnitude sejam descritas como mudança significativa
        enquanto ainda detecta variações relevantes em métricas de baixo valor.

        O parâmetro `inverso` inverte a polaridade: para mortalidade e UTI,
        aumento é piora; para vacinação, aumento é melhora.
        """
        if anterior is None:
            return "Período anterior indisponível para comparação."

        delta         = atual - anterior
        threshold_abs = 0.1
        threshold_rel = abs(atual) * 0.02 if atual else 0.0
        threshold     = max(threshold_abs, threshold_rel)

        if abs(delta) < threshold:
            return (
                f"Sem variação significativa em relação ao período anterior "
                f"({anterior:.1f}%) — variação de {abs(delta):.2f} pp dentro "
                f"da margem de estabilidade."
            )

        direcao = "aumento" if delta > 0 else "redução"
        sinal   = "piora" if (delta > 0) == inverso else "melhora"

        return (
            f"{direcao.capitalize()} de {abs(delta):.1f} pp em relação ao "
            f"período anterior ({anterior:.1f}%), indicando {sinal} no indicador de {label}."
        )

    def _invoke_llm(self, prompt: str, section: str) -> str:
        """
        Invoca o LLM e retorna o texto gerado.

        Erros são registrados no AuditLogger e resultam em uma mensagem de
        fallback explícita — nunca em string vazia, que seria silenciosa
        no documento final.
        """
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as exc:
            self.audit.log_event(
                AuditEvent.REPORT_LLM_ERROR,
                {"section": section, "error": str(exc)},
                EventStatus.ERROR,
            )
            return f"Síntese indisponível — erro ao invocar LLM: {exc}"

    def _resolve_llm_name(self) -> str:
        """
        Extrai um nome legível do modelo sem expor URLs de infraestrutura.

        ChatOpenAI expõe model_name. ChatDatabricks expõe endpoint como URL
        completa (https://adb-xxx.../serving-endpoints/nome-do-modelo/invocations).
        Quando o atributo disponível é uma URL, apenas o penúltimo segmento
        do path é usado como nome — correspondendo ao nome do endpoint
        registrado no Databricks. O fallback final é o nome da classe Python.
        """
        for attr in ("model_name", "model_id", "endpoint"):
            val = getattr(self.llm, attr, None)
            if not val or not isinstance(val, str):
                continue
            if val.startswith("http"):
                segments = [s for s in val.rstrip("/").split("/") if s]
                return segments[-2] if len(segments) >= 2 else segments[-1]
            return val

        return type(self.llm).__name__

    @staticmethod
    def _fmt_number(value, fallback: str = "N/A") -> str:
        """
        Formata um valor inteiro com separador de milhar.

        O operador de formato :, só aceita int/float. Campos ausentes
        retornados como None ou "N/A" pelo dict.get() causavam ValueError
        ou TypeError quando interpolados diretamente com :, em f-strings.
        """
        if value is None:
            return fallback
        try:
            return f"{int(value):,}"
        except (ValueError, TypeError):
            return fallback

    @staticmethod
    def _fmt_float(value, decimals: int = 2, fallback: str = "N/A") -> str:
        """
        Formata um valor de ponto flutuante com número fixo de casas decimais.

        Centraliza a guarda de tipo para todos os campos percentuais do
        relatório, evitando que :f ou :.2f sejam usados inline com valores
        potencialmente None ou string.
        """
        if value is None:
            return fallback
        try:
            return f"{float(value):.{decimals}f}"
        except (ValueError, TypeError):
            return fallback