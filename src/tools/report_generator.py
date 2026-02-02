"""
Report Generator - Geração de Relatórios Markdown
==================================================

Gera relatório epidemiológico automatizado com:
- 4 métricas obrigatórias (crescimento, mortalidade, UTI, vacinação)
- Contexto de notícias
- Referências a gráficos
- Análises e recomendações
- Síntese via LLM

Author: Certificação AI Engineer - Indicium
Date: January 2025
"""

from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ..utils.audit import AuditLogger, AuditEvent, EventStatus


@dataclass
class ReportSection:
    """Seção do relatório"""
    title: str
    content: str
    order: int


class ReportGenerator:
    """
    Gera relatório markdown profissional
    
    Template:
        1. Cabeçalho + metadata
        2. Resumo executivo
        3. 4 métricas epidemiológicas
        4. Análise geográfica
        5. Contexto de notícias
        6. Gráficos anexos
        7. Recomendações
        8. Rodapé
    """
    
    def __init__(self, llm: ChatOpenAI, audit: Optional[AuditLogger] = None):
        self.llm = llm
        self.audit = audit
    
    def generate_report(
        self,
        metrics: Optional[Dict] = None,
        geographic: Optional[Dict] = None,
        news: Optional[Dict] = None,
        charts: Optional[List[str]] = None,
        rag_context: Optional[Dict] = None,
        user_query: str = "Gerar relatório SRAG"
    ) -> str:
        """
        Gera relatório completo
        
        Args:
            metrics: Dados de métricas temporais
            geographic: Dados geográficos
            news: Notícias coletadas
            charts: Paths dos gráficos
            rag_context: Contexto RAG (opcional)
            user_query: Query original do usuário
            
        Returns:
            String markdown formatada
        """
        if self.audit:
            self.audit.log_event(
                AuditEvent.REPORT_GENERATION_START,
                {"has_metrics": metrics is not None, "has_news": news is not None},
                EventStatus.INFO
            )
        
        sections = []
        
        # 1. Cabeçalho
        sections.append(self._build_header())
        
        # 2. Resumo Executivo
        sections.append(self._build_executive_summary(metrics, news, rag_context))
        
        # 3. Métricas (4 obrigatórias)
        sections.append(self._build_metrics_section(metrics, rag_context))
        
        # 4. Análise Geográfica
        if geographic:
            sections.append(self._build_geographic_section(geographic))
        
        # 5. Contexto de Notícias
        if news:
            sections.append(self._build_news_section(news))
        
        # 6. Gráficos
        sections.append(self._build_charts_section(charts))
        
        # 7. Recomendações
        sections.append(self._build_recommendations(metrics, news))
        
        # 8. Rodapé
        sections.append(self._build_footer())
        
        # Montar relatório final
        report_md = "\n\n".join([s.content for s in sorted(sections, key=lambda x: x.order)])
        
        if self.audit:
            self.audit.log_event(
                AuditEvent.REPORT_GENERATED,
                {"sections": len(sections), "length": len(report_md)},
                EventStatus.SUCCESS
            )
        
        return report_md
    
    def _build_header(self) -> ReportSection:
        """Cabeçalho do relatório"""
        now = datetime.now()
        content = f"""# 📊 Relatório Epidemiológico SRAG - {now.strftime('%B %Y')}

**Data de Geração:** {now.strftime('%d/%m/%Y às %H:%M')}  
**Sistema:** Agente IA de Monitoramento Epidemiológico  
**Versão:** 3.0.0  

---"""
        
        return ReportSection(title="header", content=content, order=1)
    
    def _build_executive_summary(
        self,
        metrics: Optional[Dict],
        news: Optional[Dict],
        rag_context: Optional[Dict]
    ) -> ReportSection:
        """Resumo executivo (sintetizado por LLM)"""
        
        context_parts = []
        
        if metrics and "data" in metrics:
            latest = metrics["data"][0] if metrics["data"] else {}
            context_parts.append(f"Métricas recentes: {latest}")
        
        if news and "articles" in news:
            context_parts.append(f"Notícias: {len(news['articles'])} artigos relevantes")
        
        if rag_context:
            context_parts.append(f"Contexto RAG: {rag_context.get('answer', '')[:200]}")
        
        if not context_parts:
            content = """## 🎯 Resumo Executivo

Dados insuficientes para gerar resumo executivo."""
            return ReportSection(title="executive_summary", content=content, order=2)
        
        prompt = f"""Gere um resumo executivo de 2-3 parágrafos sobre a situação epidemiológica de SRAG.

Contexto disponível:
{chr(10).join(context_parts)}

O resumo deve:
- Destacar principais achados
- Mencionar tendências
- Ser objetivo e profissional
- Usar dados concretos quando disponíveis
"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            summary_text = response.content
        except Exception as e:
            summary_text = "Erro ao gerar resumo executivo via LLM."
        
        content = f"""## 🎯 Resumo Executivo

{summary_text}"""
        
        return ReportSection(title="executive_summary", content=content, order=2)
    
    def _build_metrics_section(
        self,
        metrics: Optional[Dict],
        rag_context: Optional[Dict]
    ) -> ReportSection:
        """Seção das 4 métricas obrigatórias"""
        
        if not metrics or not metrics.get("data"):
            content = """## 📈 Métricas Epidemiológicas

⚠️ Dados de métricas não disponíveis."""
            return ReportSection(title="metrics", content=content, order=3)
        
        data = metrics["data"]
        latest = data[0] if data else {}
        previous = data[1] if len(data) > 1 else {}
        
        content = f"""## 📈 Métricas Epidemiológicas Principais

### 1️⃣ Taxa de Crescimento de Casos

- **Valor Atual:** {latest.get('taxa_crescimento', 'N/A')}%
- **Período Anterior:** {previous.get('taxa_crescimento', 'N/A')}%

**Análise:** {self._analyze_growth(latest, previous)}

---

### 2️⃣ Taxa de Mortalidade

- **Valor Atual:** {latest.get('taxa_mortalidade', 'N/A')}%
- **Total de Casos:** {latest.get('total_casos', 'N/A'):,}

**Análise:** {self._analyze_mortality(latest, previous)}

---

### 3️⃣ Taxa de Ocupação de UTI

- **Valor Atual:** {latest.get('taxa_uti', 'N/A')}%

**Análise:** {self._analyze_uti(latest, previous)}

---

### 4️⃣ Taxa de Vacinação

- **Cobertura Atual:** {latest.get('taxa_vacinacao', 'N/A')}%

**Análise:** {self._analyze_vaccination(latest, previous)}"""
        
        return ReportSection(title="metrics", content=content, order=3)
    
    def _analyze_growth(self, latest: Dict, previous: Dict) -> str:
        """Análise automática de crescimento"""
        growth = latest.get('taxa_crescimento', 0)
        
        try:
            growth_val = float(growth)
            if growth_val > 10:
                return f"Crescimento significativo de {growth}% indica aumento expressivo de casos."
            elif growth_val > 0:
                return f"Crescimento moderado de {growth}% sugere estabilidade com leve alta."
            else:
                return f"Variação de {growth}% indica estabilização ou redução de casos."
        except:
            return "Dados insuficientes para análise de tendência."
    
    def _analyze_mortality(self, latest: Dict, previous: Dict) -> str:
        """Análise automática de mortalidade"""
        mort = latest.get('taxa_mortalidade', 0)
        
        try:
            mort_val = float(mort)
            if mort_val > 10:
                return f"Taxa de {mort}% é considerada alta, requerendo atenção especial."
            elif mort_val > 5:
                return f"Taxa de {mort}% está em nível moderado."
            else:
                return f"Taxa de {mort}% está em patamar controlado."
        except:
            return "Dados insuficientes para análise de mortalidade."
    
    def _analyze_uti(self, latest: Dict, previous: Dict) -> str:
        """Análise automática de UTI"""
        uti = latest.get('taxa_uti', 0)
        
        try:
            uti_val = float(uti)
            if uti_val > 70:
                return f"Ocupação de {uti}% indica pressão crítica no sistema de saúde."
            elif uti_val > 50:
                return f"Ocupação de {uti}% requer monitoramento intensivo."
            else:
                return f"Ocupação de {uti}% está em nível controlável."
        except:
            return "Dados insuficientes para análise de UTI."
    
    def _analyze_vaccination(self, latest: Dict, previous: Dict) -> str:
        """Análise automática de vacinação"""
        vac = latest.get('taxa_vacinacao', 0)
        
        try:
            vac_val = float(vac)
            if vac_val > 70:
                return f"Cobertura de {vac}% é satisfatória."
            elif vac_val > 50:
                return f"Cobertura de {vac}% está em expansão, mas pode melhorar."
            else:
                return f"Cobertura de {vac}% está abaixo do ideal, necessita intensificação."
        except:
            return "Dados insuficientes para análise de vacinação."
    
    def _build_geographic_section(self, geographic: Dict) -> ReportSection:
        """Seção de análise geográfica"""
        
        if not geographic or not geographic.get("data"):
            content = ""
            return ReportSection(title="geographic", content=content, order=4)
        
        data = geographic["data"][:5]  # Top 5
        
        content = f"""## 🗺️ Análise Geográfica

### Estados Mais Afetados

| Ranking | UF | Casos | Taxa Mortalidade |
|---------|----| ------|------------------|
"""
        
        for idx, uf_data in enumerate(data, 1):
            sg_uf = uf_data.get('sg_uf', 'N/A')
            total_casos = uf_data.get('total_casos', 0)
            taxa_mort = uf_data.get('taxa_mortalidade', 0)
            
            content += f"| {idx}º | {sg_uf} | {total_casos:,} | {taxa_mort:.2f}% |\n"
        
        return ReportSection(title="geographic", content=content, order=4)
    
    def _build_news_section(self, news: Dict) -> ReportSection:
        """Seção de contexto de notícias"""
        
        if not news or not news.get("articles"):
            content = ""
            return ReportSection(title="news", content=content, order=5)
        
        articles = news["articles"][:5]
        
        content = f"""## 📰 Contexto de Notícias Recentes

Foram identificadas **{len(articles)} notícias** relevantes sobre SRAG:

"""
        
        for idx, article in enumerate(articles, 1):
            title = article.get('title', 'N/A')
            source = article.get('source', 'N/A')
            date = article.get('published_date', 'N/A')
            url = article.get('url', '#')
            
            content += f"**{idx}. {title}**  \n"
            content += f"   - Fonte: {source}  \n"
            content += f"   - Data: {date}  \n"
            content += f"   - [Link]({url})  \n\n"
        
        return ReportSection(title="news", content=content, order=5)
    
    def _build_charts_section(self, charts: Optional[List[str]]) -> ReportSection:
        """Seção de gráficos"""
        
        num_charts = len(charts) if charts else 0
        
        content = f"""## 📊 Visualizações

Foram gerados **{num_charts} gráficos** para análise visual:

"""
        
        if charts:
            for idx, chart_path in enumerate(charts, 1):
                content += f"{idx}. `{chart_path}`\n"
        else:
            content += "⚠️ Nenhum gráfico foi gerado.\n"
        
        return ReportSection(title="charts", content=content, order=6)
    
    def _build_recommendations(
        self,
        metrics: Optional[Dict],
        news: Optional[Dict]
    ) -> ReportSection:
        """Seção de recomendações"""
        
        content = """## 💡 Recomendações

Com base nos dados analisados:

1. **Monitoramento Contínuo**
   - Acompanhar evolução diária das métricas
   - Atenção especial para regiões com alta ocupação de UTI

2. **Intensificação da Vacinação**
   - Priorizar regiões com baixa cobertura
   - Campanhas direcionadas para grupos de risco

3. **Vigilância Epidemiológica**
   - Reforçar notificação de casos
   - Análise de tendências semanais

4. **Capacidade Hospitalar**
   - Monitorar disponibilidade de leitos UTI
   - Planejamento de contingência em regiões críticas

5. **Comunicação Pública**
   - Divulgar métricas atualizadas
   - Orientação à população sobre prevenção
"""
        
        return ReportSection(title="recommendations", content=content, order=7)
    
    def _build_footer(self) -> ReportSection:
        """Rodapé do relatório"""
        
        content = f"""---

## 📋 Informações Técnicas

- **Fonte de Dados:** SIVEP-Gripe via Databricks Gold Layer
- **Metodologia:** Arquitetura Medallion (Bronze → Silver → Gold)
- **Sistema:** Agente Orquestrador com LangGraph
- **LLM:** GPT-4o-mini (OpenAI)
- **Geração:** Automatizada via AI Agent

---

*Relatório gerado automaticamente pelo Sistema de Monitoramento Epidemiológico SRAG*  
*Certificação AI Engineer - Indicium*  
*{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}*
"""
        
        return ReportSection(title="footer", content=content, order=8)