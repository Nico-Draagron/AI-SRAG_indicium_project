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

    O mesmo princípio se aplica às recomendações: o design original retornava
    texto completamente estático, ignorando os parâmetros recebidos. As
    recomendações agora são geradas pelo LLM com base nas métricas reais,
    tornando-as específicas para o estado epidemiológico atual. Quando não
    há métricas, retorna um conjunto mínimo de recomendações de vigilância
    contínua, sem inventar urgências.

Análise comparativa entre períodos
    Os quatro métodos _analyze_*() recebiam `previous` na assinatura mas
    nunca o usavam — produzindo frases como "crescimento de 5% sugere leve
    alta" sem informar se o período anterior era 2% ou 15%. A análise agora
    compara explicitamente latest vs previous e classifica a tendência
    (aceleração, estabilidade, redução) como componente essencial da
    interpretação epidemiológica.

Proteção numérica consistente em todas as seções
    O fix de _fmt_number() havia sido aplicado em _build_metrics_section()
    mas não em _build_geographic_section(), que ainda usava f"{total_casos:,}"
    inline. Qualquer campo None ou string "N/A" retornado pelo dict.get()
    lançava TypeError ou ValueError abortando a geração do relatório inteiro.
    _fmt_number() e _fmt_float() são agora usados em todos os pontos de
    interpolação numérica.

Extração segura do nome do LLM
    O design original tentava self.llm.endpoint como fallback para
    ChatDatabricks. Esse atributo retorna a URL completa do endpoint
    (https://adb-xxx.azuredatabricks.net/...), expondo infraestrutura
    interna no rodapé do relatório. A extração agora usa _resolve_llm_name(),
    que testa model_name e model_id em sequência, extrai apenas o segmento
    final do path quando detecta uma URL, e usa o nome da classe como
    fallback definitivo — sempre retornando um valor legível e não sensível.

Filtragem de seções vazias antes do join
    Seções com content="" contribuíam com string vazia para o "\n\n".join(),
    resultando em blocos de linhas em branco duplas no markdown final.
    generate_report() agora filtra seções com conteúdo vazio antes de montar
    o documento, eliminando separadores fantasma.

Distinção entre "sem dados" e "dados vazios"
    O contrato com o orquestrador passa metrics={"data": [{}]} quando
    mandatory_metrics falha — uma lista com um dict vazio, não uma lista
    vazia. A guarda "if data else {}" não detecta esse caso. _extract_latest()
    verifica se o primeiro elemento contém ao menos uma chave com valor não-None
    antes de tratá-lo como dado válido, evitando que um dict vazio seja
    interpretado como período de dados reais e produza um relatório com N/A
    em todos os campos.

Versão do sistema injetável
    A versão estava hardcoded como "3.0.0" no template do cabeçalho,
    divergindo das versões reais dos módulos (SQLTool 3.1.0, WebSearchTool
    2.1.0). A versão agora é passada no construtor com um valor default
    explícito, permitindo que o notebook ou o agente injete a versão real
    sem alterar o código.
"""

from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional

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
            total_casos. O primeiro elemento é o período mais recente; o
            segundo, o anterior para comparação de tendência.
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
            },
            EventStatus.INFO,
        )

        latest, previous = self._extract_periods(metrics)

        sections = [
            self._build_header(),
            self._build_executive_summary(latest, previous, news, rag_context),
            self._build_metrics_section(latest, previous),
            self._build_geographic_section(geographic),
            self._build_news_section(news),
            self._build_charts_section(charts),
            self._build_recommendations(latest, previous, news),
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

    def _build_header(self) -> ReportSection:
        now     = datetime.now()
        content = (
            f"# Relatório Epidemiológico SRAG — {now.strftime('%B %Y')}\n\n"
            f"**Data de Geração:** {now.strftime('%d/%m/%Y às %H:%M')}  \n"
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
    ) -> ReportSection:
        """
        Gera o resumo executivo via LLM apenas quando há dados reais.

        A ausência de dados produz uma seção explicitamente marcada como
        indisponível, em vez de invocar o LLM com contexto vazio — o que
        levaria o modelo a preencher o gap com estatísticas inventadas.
        """
        context_parts = []

        if latest:
            context_parts.append(f"Métricas mais recentes: {latest}")
        if previous:
            context_parts.append(f"Métricas do período anterior: {previous}")
        if news and news.get("articles"):
            context_parts.append(
                f"Notícias: {len(news['articles'])} artigos relevantes disponíveis."
            )
        if rag_context and rag_context.get("answer"):
            context_parts.append(
                f"Contexto adicional (RAG): {rag_context['answer'][:300]}"
            )

        if not context_parts:
            self.audit.log_event(
                AuditEvent.REPORT_SECTION_SKIPPED,
                {"section": "executive_summary", "reason": "sem dados para síntese"},
                EventStatus.WARNING,
            )
            content = (
                "## Resumo Executivo\n\n"
                "Dados insuficientes para gerar resumo executivo. "
                "Verifique a disponibilidade das métricas obrigatórias no pipeline."
            )
            return ReportSection(title="executive_summary", content=content, order=2)

        prompt = (
            "Gere um resumo executivo de 2 a 3 parágrafos sobre a situação epidemiológica "
            "de SRAG com base exclusivamente nos dados fornecidos abaixo. "
            "Não invente estatísticas, percentuais ou tendências que não estejam presentes "
            "no contexto. Se um dado estiver ausente, omita-o em vez de estimá-lo.\n\n"
            "Contexto disponível:\n"
            + "\n".join(context_parts)
        )

        summary_text = self._invoke_llm(prompt, section="executive_summary")
        content      = f"## Resumo Executivo\n\n{summary_text}"
        return ReportSection(title="executive_summary", content=content, order=2)

    def _build_metrics_section(self, latest: Dict, previous: Dict) -> ReportSection:
        """
        Seção das quatro métricas obrigatórias com análise comparativa.

        Quando latest está vazio (nenhum dado válido retornado pelo pipeline),
        a seção indica indisponibilidade explicitamente em vez de exibir N/A
        em todos os campos sem contexto.
        """
        if not latest:
            content = (
                "## Métricas Epidemiológicas\n\n"
                "Dados de métricas não disponíveis. "
                "Verifique a execução do nó de métricas obrigatórias no orquestrador."
            )
            return ReportSection(title="metrics", content=content, order=3)

        tc_atual    = self._fmt_float(latest.get("taxa_crescimento"))
        tc_anterior = self._fmt_float(previous.get("taxa_crescimento"))
        tm_atual    = self._fmt_float(latest.get("taxa_mortalidade"))
        tm_anterior = self._fmt_float(previous.get("taxa_mortalidade"))
        uti_atual   = self._fmt_float(latest.get("taxa_uti"))
        uti_anterior= self._fmt_float(previous.get("taxa_uti"))
        vac_atual   = self._fmt_float(latest.get("taxa_vacinacao"))
        vac_anterior= self._fmt_float(previous.get("taxa_vacinacao"))
        total_casos = self._fmt_number(latest.get("total_casos"))

        content = (
            "## Métricas Epidemiológicas Principais\n\n"

            "### 1. Taxa de Crescimento de Casos\n\n"
            f"- **Valor Atual:** {tc_atual}%\n"
            f"- **Período Anterior:** {tc_anterior}%\n\n"
            f"**Análise:** {self._analyze_growth(latest, previous)}\n\n"
            "---\n\n"

            "### 2. Taxa de Mortalidade\n\n"
            f"- **Valor Atual:** {tm_atual}%\n"
            f"- **Período Anterior:** {tm_anterior}%\n"
            f"- **Total de Casos:** {total_casos}\n\n"
            f"**Análise:** {self._analyze_mortality(latest, previous)}\n\n"
            "---\n\n"

            "### 3. Taxa de Ocupação de UTI\n\n"
            f"- **Valor Atual:** {uti_atual}%\n"
            f"- **Período Anterior:** {uti_anterior}%\n\n"
            f"**Análise:** {self._analyze_uti(latest, previous)}\n\n"
            "---\n\n"

            "### 4. Taxa de Vacinação\n\n"
            f"- **Cobertura Atual:** {vac_atual}%\n"
            f"- **Período Anterior:** {vac_anterior}%\n\n"
            f"**Análise:** {self._analyze_vaccination(latest, previous)}"
        )

        return ReportSection(title="metrics", content=content, order=3)

    def _build_geographic_section(self, geographic: Optional[Dict]) -> ReportSection:
        """
        Seção dos estados mais afetados.

        Retorna seção com content="" quando não há dados, o que a exclui do
        documento final via filtragem em generate_report(). Todos os campos
        numéricos passam por _fmt_number() e _fmt_float() para evitar crash
        quando total_casos ou taxa_mortalidade retornam None ou "N/A".
        """
        if not geographic or not geographic.get("data"):
            return ReportSection(title="geographic", content="", order=4)

        rows = geographic["data"][:5]

        lines = [
            "## Análise Geográfica\n\n",
            "### Estados Mais Afetados\n\n",
            "| Ranking | UF | Casos | Taxa Mortalidade |\n",
            "|---------|-----|-------|------------------|\n",
        ]

        for idx, row in enumerate(rows, 1):
            sg_uf      = row.get("sg_uf", "N/A")
            total      = self._fmt_number(row.get("total_casos"))
            taxa_mort  = self._fmt_float(row.get("taxa_mortalidade"))
            lines.append(f"| {idx} | {sg_uf} | {total} | {taxa_mort}% |\n")

        return ReportSection(title="geographic", content="".join(lines), order=4)

    def _build_news_section(self, news: Optional[Dict]) -> ReportSection:
        """
        Seção de contexto de notícias.

        Consome o campo canônico "articles" do WebSearchTool. O alias "news"
        foi removido do payload do WebSearchTool — chamadores que passem o
        resultado diretamente devem usar result["articles"] como chave raiz
        se estiverem construindo o dict manualmente.
        """
        articles = (news or {}).get("articles", [])
        if not articles:
            return ReportSection(title="news", content="", order=5)

        top = articles[:5]
        lines = [
            "## Contexto de Notícias Recentes\n\n",
            f"Foram identificados **{len(top)}** artigos relevantes sobre SRAG:\n\n",
        ]

        for idx, article in enumerate(top, 1):
            title  = article.get("title", "N/A")
            source = article.get("source", "N/A")
            date   = article.get("published_date", "N/A")
            url    = article.get("url", "#")
            score  = article.get("relevance_score", 0.0)
            lines.append(
                f"**{idx}. {title}**  \n"
                f"   - Fonte: {source} | Data: {date} | Score de relevância: {score:.2f}  \n"
                f"   - [{url}]({url})  \n\n"
            )

        return ReportSection(title="news", content="".join(lines), order=5)

    def _build_charts_section(self, charts: Optional[List[str]]) -> ReportSection:
        num = len(charts) if charts else 0

        lines = [f"## Visualizações\n\nForam gerados **{num}** gráficos:\n\n"]
        if charts:
            for idx, path in enumerate(charts, 1):
                lines.append(f"{idx}. `{path}`\n")
        else:
            lines.append("Nenhum gráfico foi gerado nesta execução.\n")

        return ReportSection(title="charts", content="".join(lines), order=6)

    def _build_recommendations(
        self,
        latest:   Dict,
        previous: Dict,
        news:     Optional[Dict],
    ) -> ReportSection:
        """
        Gera recomendações via LLM contextualizadas com as métricas reais.

        O design original retornava texto fixo idêntico em toda execução,
        ignorando completamente os parâmetros recebidos. Um relatório com
        UTI a 85% e um com UTI a 20% recebiam as mesmas recomendações.

        Quando há métricas válidas, o LLM recebe o estado atual das quatro
        métricas e produz recomendações específicas para esse cenário.
        Quando não há dados, retorna recomendações mínimas de vigilância
        contínua — sem inventar urgências que os dados não sustentam.
        """
        if not latest:
            content = (
                "## Recomendações\n\n"
                "Com base no estado atual de monitoramento:\n\n"
                "1. **Vigilância Epidemiológica** — Manter notificação compulsória de casos "
                "e análise de tendências semanais.\n"
                "2. **Monitoramento de Capacidade** — Acompanhar disponibilidade de leitos UTI "
                "e acionar plano de contingência se necessário.\n"
            )
            return ReportSection(title="recommendations", content=content, order=7)

        articles = (news or {}).get("articles", [])
        news_ctx  = (
            f"\nContexto de notícias: {len(articles)} artigos recentes disponíveis."
            if articles else ""
        )

        prompt = (
            "Com base nos dados epidemiológicos abaixo, gere de 4 a 5 recomendações "
            "objetivas e específicas para gestores de saúde pública. "
            "Cada recomendação deve ser diretamente justificada pelos dados fornecidos. "
            "Não inclua recomendações genéricas que seriam válidas para qualquer cenário.\n\n"
            f"Taxa de crescimento atual: {self._fmt_float(latest.get('taxa_crescimento'))}% "
            f"(anterior: {self._fmt_float(previous.get('taxa_crescimento'))}%)\n"
            f"Taxa de mortalidade atual: {self._fmt_float(latest.get('taxa_mortalidade'))}% "
            f"(anterior: {self._fmt_float(previous.get('taxa_mortalidade'))}%)\n"
            f"Ocupação de UTI atual: {self._fmt_float(latest.get('taxa_uti'))}% "
            f"(anterior: {self._fmt_float(previous.get('taxa_uti'))}%)\n"
            f"Cobertura vacinal atual: {self._fmt_float(latest.get('taxa_vacinacao'))}% "
            f"(anterior: {self._fmt_float(previous.get('taxa_vacinacao'))}%)\n"
            f"Total de casos: {self._fmt_number(latest.get('total_casos'))}"
            f"{news_ctx}"
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
    # ANÁLISE COMPARATIVA DE MÉTRICAS
    # =========================================================================

    def _analyze_growth(self, latest: Dict, previous: Dict) -> str:
        """
        Interpreta a taxa de crescimento atual e sua tendência em relação
        ao período anterior. A comparação com `previous` é o componente
        central da análise — um crescimento de 5% tem significado oposto
        dependendo de se o período anterior era 2% ou 15%.
        """
        try:
            atual    = float(latest.get("taxa_crescimento", 0) or 0)
            anterior = float(previous.get("taxa_crescimento") or 0) if previous else None

            tendencia = self._tendencia(atual, anterior, label="crescimento")

            if atual > 10:
                nivel = f"Crescimento expressivo de {atual:.1f}%"
            elif atual > 0:
                nivel = f"Crescimento moderado de {atual:.1f}%"
            elif atual == 0:
                nivel = "Estabilidade no número de casos"
            else:
                nivel = f"Redução de {abs(atual):.1f}% no número de casos"

            return f"{nivel}. {tendencia}"
        except (TypeError, ValueError):
            return "Dados insuficientes para análise de tendência de crescimento."

    def _analyze_mortality(self, latest: Dict, previous: Dict) -> str:
        try:
            atual    = float(latest.get("taxa_mortalidade", 0) or 0)
            anterior = float(previous.get("taxa_mortalidade") or 0) if previous else None

            tendencia = self._tendencia(atual, anterior, label="mortalidade", inverso=True)

            if atual > 10:
                nivel = f"Taxa de {atual:.1f}% é considerada elevada e requer atenção imediata"
            elif atual > 5:
                nivel = f"Taxa de {atual:.1f}% está em nível moderado"
            else:
                nivel = f"Taxa de {atual:.1f}% está em patamar controlado"

            return f"{nivel}. {tendencia}"
        except (TypeError, ValueError):
            return "Dados insuficientes para análise de mortalidade."

    def _analyze_uti(self, latest: Dict, previous: Dict) -> str:
        try:
            atual    = float(latest.get("taxa_uti", 0) or 0)
            anterior = float(previous.get("taxa_uti") or 0) if previous else None

            tendencia = self._tendencia(atual, anterior, label="ocupação de UTI", inverso=True)

            if atual > 70:
                nivel = f"Ocupação de {atual:.1f}% indica pressão crítica no sistema de saúde"
            elif atual > 50:
                nivel = f"Ocupação de {atual:.1f}% requer monitoramento intensivo"
            else:
                nivel = f"Ocupação de {atual:.1f}% está em nível controlável"

            return f"{nivel}. {tendencia}"
        except (TypeError, ValueError):
            return "Dados insuficientes para análise de ocupação de UTI."

    def _analyze_vaccination(self, latest: Dict, previous: Dict) -> str:
        try:
            atual    = float(latest.get("taxa_vacinacao", 0) or 0)
            anterior = float(previous.get("taxa_vacinacao") or 0) if previous else None

            tendencia = self._tendencia(atual, anterior, label="cobertura vacinal", inverso=False)

            if atual > 70:
                nivel = f"Cobertura de {atual:.1f}% é satisfatória"
            elif atual > 50:
                nivel = f"Cobertura de {atual:.1f}% está em expansão mas pode melhorar"
            else:
                nivel = f"Cobertura de {atual:.1f}% está abaixo do ideal"

            return f"{nivel}. {tendencia}"
        except (TypeError, ValueError):
            return "Dados insuficientes para análise de cobertura vacinal."

    # =========================================================================
    # UTILITÁRIOS INTERNOS
    # =========================================================================

    def _extract_periods(
        self,
        metrics: Optional[Dict],
    ) -> tuple:
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

        O parâmetro `inverso` inverte a polaridade da comparação: para métricas
        onde aumento é negativo (mortalidade, UTI), um valor maior que o anterior
        é descrito como piora; para métricas onde aumento é positivo (vacinação),
        um valor maior é descrito como melhora.
        """
        if anterior is None:
            return "Período anterior indisponível para comparação."

        delta = atual - anterior
        if abs(delta) < 0.01:
            return f"Sem variação significativa em relação ao período anterior ({anterior:.1f}%)."

        direcao = "aumento" if delta > 0 else "redução"
        sinal   = "piora" if (delta > 0) == inverso else "melhora"

        return (
            f"{direcao.capitalize()} de {abs(delta):.1f} pontos percentuais em relação ao "
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