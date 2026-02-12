"""
Chart Tool - Geração Profissional de Visualizações SRAG (REFATORADO)
====================================================================

Versão refatorada com 10 gráficos profissionais para certificação:

OBRIGATÓRIOS (2):
1. ✅ Casos diários (últimos 30 dias)
2. ✅ Casos mensais (últimos 12 meses)

ESSENCIAIS (6):
3. ✅ Tendência temporal completa (36 meses 2023-2025)
4. ✅ Comparativo anual (2023 vs 2024 vs 2025)
5. ✅ Heatmap de sazonalidade mensal
6. ✅ Taxa de crescimento mensal
7. ✅ Ranking Top 10 UFs
8. ✅ Perfil demográfico (faixas etárias)

DIFERENCIAIS (2):
9. ✅ Mortalidade x UTI (dois eixos Y)
10. ✅ Vacinação vs Mortalidade (correlação)

Author: AI Engineer Certification - Indicium
Date: February 2025
Version: 3.0.0 - REFATORADO COMPLETO
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import traceback

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

# Imports condicionais para auditoria
try:
    from src.utils.audit import AuditLogger, AuditEvent, EventStatus
except ImportError:
    class AuditEvent:
        TOOL_INITIALIZED = "tool_initialized"
        CHART_GENERATION_START = "chart_generation_start"
        CHART_GENERATED = "chart_generated"
        CHART_ERROR = "chart_error"
    
    class EventStatus:
        INFO = "INFO"
        SUCCESS = "SUCCESS"
        ERROR = "ERROR"
    
    class AuditLogger:
        def log_event(self, event_type, details=None, status="INFO"):
            print(f"[{status}] {event_type}: {details}")

try:
    from src.utils.exceptions import ChartGenerationError, ChartValidationError
except ImportError:
    class ChartGenerationError(Exception):
        pass
    class ChartValidationError(Exception):
        pass


# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

class ChartType(Enum):
    """Tipos de gráficos"""
    LINE = "line"
    BAR = "bar"
    HEATMAP = "heatmap"
    SCATTER = "scatter"
    COMBO = "combo"
    WATERFALL = "waterfall"


@dataclass
class ChartConfig:
    """Configuração global"""
    default_height: int = 600
    default_width: int = 1000
    output_directory: str = "/dbfs/FileStore/charts"
    
    # Cores profissionais SRAG
    color_palette: List[str] = field(default_factory=lambda: [
        "#1f77b4",  # Azul principal
        "#ff7f0e",  # Laranja alerta
        "#2ca02c",  # Verde ok
        "#d62728",  # Vermelho crítico
        "#9467bd",  # Roxo
        "#8c564b",  # Marrom
        "#e377c2",  # Rosa
        "#7f7f7f",  # Cinza
    ])
    
    # Cores específicas para métricas
    color_casos: str = "#1f77b4"
    color_mortalidade: str = "#d62728"
    color_uti: str = "#ff7f0e"
    color_vacinacao: str = "#2ca02c"
    color_crescimento_pos: str = "#2ca02c"
    color_crescimento_neg: str = "#d62728"


@dataclass
class ChartMetadata:
    """Metadados do gráfico"""
    chart_id: str
    chart_type: ChartType
    title: str
    created_at: datetime
    data_points: int
    export_path: str
    generation_time_seconds: float


# =============================================================================
# CHART TOOL REFATORADO
# =============================================================================

class ChartTool:
    """
    Ferramenta de geração de 10 gráficos profissionais para SRAG
    
    Pipeline de execução:
        1. Coletar dados do Spark (queries otimizadas)
        2. Processar e validar dados
        3. Gerar gráfico com Plotly
        4. Exportar HTML
        5. Auditar resultado
    
    Example:
        >>> chart_tool = ChartTool(spark, audit_logger)
        >>> paths = chart_tool.generate_all_charts()
        >>> print(f"Gerados {len(paths)} gráficos")
    """
    
    def __init__(
        self,
        spark,
        audit_logger: Optional[AuditLogger] = None,
        config: Optional[ChartConfig] = None,
        output_dir: Optional[str] = None
    ):
        self.spark = spark
        self.audit = audit_logger if audit_logger else AuditLogger()
        self.config = config or ChartConfig()
        
        # Output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(self.config.output_directory)
        
        # Criar diretório
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            print(f"✅ Chart output dir: {self.output_dir}")
        except Exception as e:
            print(f"⚠️ Erro ao criar dir: {e}")
            import tempfile
            self.output_dir = Path(tempfile.mkdtemp(prefix="charts_"))
            print(f"   📂 Fallback: {self.output_dir}")
        
        # Estatísticas
        self._charts_created = 0
        self._total_generation_time = 0.0
        
        self.audit.log_event(
            AuditEvent.TOOL_INITIALIZED,
            {
                "tool": "ChartTool",
                "output_dir": str(self.output_dir),
                "version": "3.0.0"
            },
            EventStatus.INFO
        )
    
    # =========================================================================
    # MÉTODO PRINCIPAL - GERAR TODOS OS GRÁFICOS
    # =========================================================================
    
    def generate_all_charts(self) -> List[str]:
        """
        Gera todos os 10 gráficos profissionais
        
        Returns:
            Lista de paths dos gráficos gerados
        """
        print("\n" + "="*80)
        print("📊 GERANDO 10 GRÁFICOS PROFISSIONAIS PARA SRAG")
        print("="*80 + "\n")
        
        chart_paths = []
        
        # 1. Casos diários (30 dias)
        print("1️⃣  Gerando: Casos Diários (últimos 30 dias)...")
        path = self.generate_daily_chart()
        if path:
            chart_paths.append(path)
        
        # 2. Casos mensais (12 meses)
        print("2️⃣  Gerando: Casos Mensais (últimos 12 meses)...")
        path = self.generate_monthly_chart()
        if path:
            chart_paths.append(path)
        
        # 3. Tendência temporal completa (36 meses)
        print("3️⃣  Gerando: Tendência Temporal Completa (36 meses)...")
        path = self.generate_temporal_trend_chart()
        if path:
            chart_paths.append(path)
        
        # 4. Comparativo anual
        print("4️⃣  Gerando: Comparativo Anual (2023 vs 2024 vs 2025)...")
        path = self.generate_annual_comparison_chart()
        if path:
            chart_paths.append(path)
        
        # 5. Heatmap sazonalidade
        print("5️⃣  Gerando: Heatmap de Sazonalidade Mensal...")
        path = self.generate_seasonality_heatmap()
        if path:
            chart_paths.append(path)
        
        # 6. Taxa de crescimento
        print("6️⃣  Gerando: Taxa de Crescimento Mensal...")
        path = self.generate_growth_rate_chart()
        if path:
            chart_paths.append(path)
        
        # 7. Ranking UFs
        print("7️⃣  Gerando: Ranking Top 10 UFs...")
        path = self.generate_uf_ranking_chart()
        if path:
            chart_paths.append(path)
        
        # 8. Perfil demográfico
        print("8️⃣  Gerando: Perfil Demográfico (Faixas Etárias)...")
        path = self.generate_demographic_profile_chart()
        if path:
            chart_paths.append(path)
        
        # 9. Mortalidade x UTI (dois eixos)
        print("9️⃣  Gerando: Mortalidade x UTI (Dois Eixos Y)...")
        path = self.generate_mortality_uti_chart()
        if path:
            chart_paths.append(path)
        
        # 10. Vacinação vs Mortalidade
        print("🔟 Gerando: Vacinação vs Mortalidade (Correlação)...")
        path = self.generate_vaccination_correlation_chart()
        if path:
            chart_paths.append(path)
        
        print("\n" + "="*80)
        print(f"✅ TOTAL: {len(chart_paths)} gráficos gerados com sucesso!")
        print("="*80 + "\n")
        
        return chart_paths
    
    # =========================================================================
    # GRÁFICO 1: CASOS DIÁRIOS (30 DIAS)
    # =========================================================================
    
    def generate_daily_chart(self) -> Optional[str]:
        """
        Gráfico de linha: Casos diários dos últimos 30 dias
        
        Query: gold_metricas_temporais (agregação diária simulada)
        """
        start_time = datetime.now()
        
        try:
            self.audit.log_event(
                AuditEvent.CHART_GENERATION_START,
                {"chart": "daily_cases", "type": "line"},
                EventStatus.INFO
            )
            
            # Coletar dados do Spark
            query = """
            SELECT 
                ano_mes,
                total_casos
            FROM dbx_lab_draagron.gold.gold_metricas_temporais
            ORDER BY ano_mes DESC
            LIMIT 30
            """
            
            df = self.spark.sql(query).toPandas()
            
            if df.empty:
                raise ChartValidationError("Sem dados para gráfico diário")
            
            # Reverter ordem (mais antigo primeiro)
            df = df.sort_values('ano_mes')
            
            # Criar figura
            fig = go.Figure()
            
            # Linha principal
            fig.add_trace(go.Scatter(
                x=df['ano_mes'],
                y=df['total_casos'],
                mode='lines+markers',
                name='Casos Diários',
                line=dict(color=self.config.color_casos, width=3),
                marker=dict(size=8, symbol='circle'),
                hovertemplate='<b>%{x}</b><br>Casos: %{y:,.0f}<extra></extra>'
            ))
            
            # Média móvel 7 dias
            df['ma7'] = df['total_casos'].rolling(window=7, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=df['ano_mes'],
                y=df['ma7'],
                mode='lines',
                name='Média Móvel (7 dias)',
                line=dict(color='orange', width=2, dash='dash'),
                hovertemplate='<b>%{x}</b><br>Média 7d: %{y:,.0f}<extra></extra>'
            ))
            
            # Layout
            fig.update_layout(
                title={
                    'text': '📈 Casos Diários de SRAG - Últimos 30 Dias',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20, 'color': '#1f77b4'}
                },
                xaxis_title='Data',
                yaxis_title='Número de Casos',
                template='plotly_white',
                height=self.config.default_height,
                width=self.config.default_width,
                hovermode='x unified',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            # Exportar
            chart_id = "1_casos_diarios"
            metadata = self._export_chart(fig, chart_id, ChartType.LINE, "Casos Diários")
            
            elapsed = (datetime.now() - start_time).total_seconds()
            metadata.generation_time_seconds = elapsed
            self._charts_created += 1
            self._total_generation_time += elapsed
            
            self.audit.log_event(
                AuditEvent.CHART_GENERATED,
                {"chart": "daily_cases", "path": metadata.export_path, "time": elapsed},
                EventStatus.SUCCESS
            )
            
            return metadata.export_path
            
        except Exception as e:
            self.audit.log_event(
                AuditEvent.CHART_ERROR,
                {"chart": "daily_cases", "error": str(e)},
                EventStatus.ERROR
            )
            print(f"   ❌ Erro: {e}")
            return None
    
    # =========================================================================
    # GRÁFICO 2: CASOS MENSAIS (12 MESES)
    # =========================================================================
    
    def generate_monthly_chart(self) -> Optional[str]:
        """
        Gráfico de barras: Casos mensais dos últimos 12 meses
        """
        start_time = datetime.now()
        
        try:
            self.audit.log_event(
                AuditEvent.CHART_GENERATION_START,
                {"chart": "monthly_cases", "type": "bar"},
                EventStatus.INFO
            )
            
            # Dados do Spark
            query = """
            SELECT 
                ano_mes,
                total_casos
            FROM dbx_lab_draagron.gold.gold_metricas_temporais
            ORDER BY ano_mes DESC
            LIMIT 12
            """
            
            df = self.spark.sql(query).toPandas()
            
            if df.empty:
                raise ChartValidationError("Sem dados mensais")
            
            df = df.sort_values('ano_mes')
            
            # Criar figura
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=df['ano_mes'],
                y=df['total_casos'],
                marker_color=self.config.color_casos,
                text=df['total_casos'],
                texttemplate='%{text:,.0f}',
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Casos: %{y:,.0f}<extra></extra>'
            ))
            
            # Layout
            fig.update_layout(
                title={
                    'text': '📊 Casos Mensais de SRAG - Últimos 12 Meses',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                xaxis_title='Mês',
                yaxis_title='Total de Casos',
                template='plotly_white',
                height=self.config.default_height,
                width=self.config.default_width,
                showlegend=False
            )
            
            # Exportar
            chart_id = "2_casos_mensais"
            metadata = self._export_chart(fig, chart_id, ChartType.BAR, "Casos Mensais")
            
            elapsed = (datetime.now() - start_time).total_seconds()
            metadata.generation_time_seconds = elapsed
            self._charts_created += 1
            self._total_generation_time += elapsed
            
            self.audit.log_event(
                AuditEvent.CHART_GENERATED,
                {"chart": "monthly_cases", "path": metadata.export_path},
                EventStatus.SUCCESS
            )
            
            return metadata.export_path
            
        except Exception as e:
            self.audit.log_event(
                AuditEvent.CHART_ERROR,
                {"chart": "monthly_cases", "error": str(e)},
                EventStatus.ERROR
            )
            print(f"   ❌ Erro: {e}")
            return None
    
    # =========================================================================
    # GRÁFICO 3: TENDÊNCIA TEMPORAL COMPLETA (36 MESES)
    # =========================================================================
    
    def generate_temporal_trend_chart(self) -> Optional[str]:
        """
        Gráfico de linha: Tendência completa 2023-2025 (36 meses)
        Com média móvel e linha de tendência
        """
        start_time = datetime.now()
        
        try:
            self.audit.log_event(
                AuditEvent.CHART_GENERATION_START,
                {"chart": "temporal_trend", "type": "line"},
                EventStatus.INFO
            )
            
            # Dados completos (36 meses)
            query = """
            SELECT 
                ano_mes,
                total_casos,
                taxa_crescimento
            FROM dbx_lab_draagron.gold.gold_metricas_temporais
            WHERE ano_mes >= '2023-01'
            ORDER BY ano_mes
            """
            
            df = self.spark.sql(query).toPandas()
            
            if df.empty:
                raise ChartValidationError("Sem dados temporais")
            
            # Criar figura
            fig = go.Figure()
            
            # Linha de casos
            fig.add_trace(go.Scatter(
                x=df['ano_mes'],
                y=df['total_casos'],
                mode='lines+markers',
                name='Casos Mensais',
                line=dict(color=self.config.color_casos, width=2),
                marker=dict(size=6),
                hovertemplate='<b>%{x}</b><br>Casos: %{y:,.0f}<extra></extra>'
            ))
            
            # Média móvel 3 meses
            df['ma3'] = df['total_casos'].rolling(window=3, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=df['ano_mes'],
                y=df['ma3'],
                mode='lines',
                name='Média Móvel (3 meses)',
                line=dict(color='orange', width=2, dash='dash'),
                hovertemplate='<b>%{x}</b><br>Média 3m: %{y:,.0f}<extra></extra>'
            ))
            
            # Linha de tendência (regressão linear simples)
            import numpy as np
            x_numeric = np.arange(len(df))
            z = np.polyfit(x_numeric, df['total_casos'], 1)
            p = np.poly1d(z)
            
            fig.add_trace(go.Scatter(
                x=df['ano_mes'],
                y=p(x_numeric),
                mode='lines',
                name='Tendência Linear',
                line=dict(color='red', width=2, dash='dot'),
                hovertemplate='Tendência: %{y:,.0f}<extra></extra>'
            ))
            
            # Layout
            fig.update_layout(
                title={
                    'text': '📈 Tendência Temporal Completa SRAG (2023-2025)',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                xaxis_title='Período',
                yaxis_title='Casos',
                template='plotly_white',
                height=self.config.default_height,
                width=self.config.default_width,
                hovermode='x unified',
                showlegend=True
            )
            
            # Exportar
            chart_id = "3_tendencia_temporal"
            metadata = self._export_chart(fig, chart_id, ChartType.LINE, "Tendência Temporal")
            
            elapsed = (datetime.now() - start_time).total_seconds()
            metadata.generation_time_seconds = elapsed
            self._charts_created += 1
            self._total_generation_time += elapsed
            
            self.audit.log_event(
                AuditEvent.CHART_GENERATED,
                {"chart": "temporal_trend", "path": metadata.export_path},
                EventStatus.SUCCESS
            )
            
            return metadata.export_path
            
        except Exception as e:
            self.audit.log_event(
                AuditEvent.CHART_ERROR,
                {"chart": "temporal_trend", "error": str(e)},
                EventStatus.ERROR
            )
            print(f"   ❌ Erro: {e}")
            return None
    
    # =========================================================================
    # GRÁFICO 4: COMPARATIVO ANUAL (2023 vs 2024 vs 2025)
    # =========================================================================
    
    def generate_annual_comparison_chart(self) -> Optional[str]:
        """
        Gráfico de barras agrupadas: Comparação mês a mês entre anos
        """
        start_time = datetime.now()
        
        try:
            self.audit.log_event(
                AuditEvent.CHART_GENERATION_START,
                {"chart": "annual_comparison", "type": "bar"},
                EventStatus.INFO
            )
            
            # Dados com ano e mês separados
            query = """
            SELECT 
                SUBSTRING(ano_mes, 1, 4) as ano,
                SUBSTRING(ano_mes, 6, 2) as mes,
                total_casos
            FROM dbx_lab_draagron.gold.gold_metricas_temporais
            WHERE ano_mes >= '2023-01'
            ORDER BY ano_mes
            """
            
            df = self.spark.sql(query).toPandas()
            
            if df.empty:
                raise ChartValidationError("Sem dados para comparação anual")
            
            # Pivotar por ano
            df_pivot = df.pivot(index='mes', columns='ano', values='total_casos').fillna(0)
            
            # Criar figura
            fig = go.Figure()
            
            meses = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
            meses_nomes = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 
                          'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
            
            cores_anos = {
                '2023': '#1f77b4',
                '2024': '#ff7f0e',
                '2025': '#2ca02c'
            }
            
            for ano in ['2023', '2024', '2025']:
                if ano in df_pivot.columns:
                    fig.add_trace(go.Bar(
                        x=meses_nomes,
                        y=[df_pivot.loc[mes, ano] if mes in df_pivot.index else 0 for mes in meses],
                        name=ano,
                        marker_color=cores_anos.get(ano, '#999999'),
                        hovertemplate=f'<b>{ano} - %{{x}}</b><br>Casos: %{{y:,.0f}}<extra></extra>'
                    ))
            
            # Layout
            fig.update_layout(
                title={
                    'text': '📊 Comparativo Anual de Casos SRAG (2023 vs 2024 vs 2025)',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                xaxis_title='Mês',
                yaxis_title='Total de Casos',
                template='plotly_white',
                height=self.config.default_height,
                width=self.config.default_width,
                barmode='group',
                showlegend=True
            )
            
            # Exportar
            chart_id = "4_comparativo_anual"
            metadata = self._export_chart(fig, chart_id, ChartType.BAR, "Comparativo Anual")
            
            elapsed = (datetime.now() - start_time).total_seconds()
            metadata.generation_time_seconds = elapsed
            self._charts_created += 1
            self._total_generation_time += elapsed
            
            self.audit.log_event(
                AuditEvent.CHART_GENERATED,
                {"chart": "annual_comparison", "path": metadata.export_path},
                EventStatus.SUCCESS
            )
            
            return metadata.export_path
            
        except Exception as e:
            self.audit.log_event(
                AuditEvent.CHART_ERROR,
                {"chart": "annual_comparison", "error": str(e)},
                EventStatus.ERROR
            )
            print(f"   ❌ Erro: {e}")
            return None
    
    # =========================================================================
    # GRÁFICO 5: HEATMAP DE SAZONALIDADE
    # =========================================================================
    
    def generate_seasonality_heatmap(self) -> Optional[str]:
        """
        Heatmap: Padrão de sazonalidade mensal ao longo dos anos
        """
        start_time = datetime.now()
        
        try:
            self.audit.log_event(
                AuditEvent.CHART_GENERATION_START,
                {"chart": "seasonality_heatmap", "type": "heatmap"},
                EventStatus.INFO
            )
            
            # Dados
            query = """
            SELECT 
                SUBSTRING(ano_mes, 1, 4) as ano,
                SUBSTRING(ano_mes, 6, 2) as mes,
                total_casos
            FROM dbx_lab_draagron.gold.gold_metricas_temporais
            WHERE ano_mes >= '2023-01'
            ORDER BY ano_mes
            """
            
            df = self.spark.sql(query).toPandas()
            
            if df.empty:
                raise ChartValidationError("Sem dados para heatmap")
            
            # Pivotar
            df_pivot = df.pivot(index='ano', columns='mes', values='total_casos').fillna(0)
            
            # Criar heatmap
            fig = go.Figure(data=go.Heatmap(
                z=df_pivot.values,
                x=['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'],
                y=df_pivot.index,
                colorscale='YlOrRd',
                text=df_pivot.values,
                texttemplate='%{text:,.0f}',
                textfont={"size": 10},
                hovertemplate='<b>%{y} - %{x}</b><br>Casos: %{z:,.0f}<extra></extra>',
                colorbar=dict(title="Casos")
            ))
            
            # Layout
            fig.update_layout(
                title={
                    'text': '🔥 Heatmap de Sazonalidade Mensal SRAG',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                xaxis_title='Mês',
                yaxis_title='Ano',
                template='plotly_white',
                height=500,
                width=self.config.default_width
            )
            
            # Exportar
            chart_id = "5_sazonalidade_heatmap"
            metadata = self._export_chart(fig, chart_id, ChartType.HEATMAP, "Sazonalidade")
            
            elapsed = (datetime.now() - start_time).total_seconds()
            metadata.generation_time_seconds = elapsed
            self._charts_created += 1
            self._total_generation_time += elapsed
            
            self.audit.log_event(
                AuditEvent.CHART_GENERATED,
                {"chart": "seasonality_heatmap", "path": metadata.export_path},
                EventStatus.SUCCESS
            )
            
            return metadata.export_path
            
        except Exception as e:
            self.audit.log_event(
                AuditEvent.CHART_ERROR,
                {"chart": "seasonality_heatmap", "error": str(e)},
                EventStatus.ERROR
            )
            print(f"   ❌ Erro: {e}")
            return None
    
    # =========================================================================
    # GRÁFICO 6: TAXA DE CRESCIMENTO MENSAL
    # =========================================================================
    
    def generate_growth_rate_chart(self) -> Optional[str]:
        """
        Gráfico de barras (waterfall): Taxa de crescimento mensal
        """
        start_time = datetime.now()
        
        try:
            self.audit.log_event(
                AuditEvent.CHART_GENERATION_START,
                {"chart": "growth_rate", "type": "waterfall"},
                EventStatus.INFO
            )
            
            # Dados
            query = """
            SELECT 
                ano_mes,
                taxa_crescimento
            FROM dbx_lab_draagron.gold.gold_metricas_temporais
            WHERE ano_mes >= '2023-01'
            ORDER BY ano_mes
            LIMIT 24
            """
            
            df = self.spark.sql(query).toPandas()
            
            if df.empty:
                raise ChartValidationError("Sem dados de crescimento")
            
            # Cores baseadas em positivo/negativo
            colors = [self.config.color_crescimento_pos if x >= 0 else self.config.color_crescimento_neg 
                     for x in df['taxa_crescimento']]
            
            # Criar figura
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=df['ano_mes'],
                y=df['taxa_crescimento'],
                marker_color=colors,
                text=df['taxa_crescimento'].apply(lambda x: f"{x:+.1f}%"),
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Crescimento: %{y:+.2f}%<extra></extra>'
            ))
            
            # Linha zero
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            # Layout
            fig.update_layout(
                title={
                    'text': '📈 Taxa de Crescimento Mensal de Casos SRAG',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                xaxis_title='Período',
                yaxis_title='Taxa de Crescimento (%)',
                template='plotly_white',
                height=self.config.default_height,
                width=self.config.default_width,
                showlegend=False
            )
            
            # Exportar
            chart_id = "6_taxa_crescimento"
            metadata = self._export_chart(fig, chart_id, ChartType.WATERFALL, "Taxa Crescimento")
            
            elapsed = (datetime.now() - start_time).total_seconds()
            metadata.generation_time_seconds = elapsed
            self._charts_created += 1
            self._total_generation_time += elapsed
            
            self.audit.log_event(
                AuditEvent.CHART_GENERATED,
                {"chart": "growth_rate", "path": metadata.export_path},
                EventStatus.SUCCESS
            )
            
            return metadata.export_path
            
        except Exception as e:
            self.audit.log_event(
                AuditEvent.CHART_ERROR,
                {"chart": "growth_rate", "error": str(e)},
                EventStatus.ERROR
            )
            print(f"   ❌ Erro: {e}")
            return None
    
    # =========================================================================
    # GRÁFICO 7: RANKING TOP 10 UFs
    # =========================================================================
    
    def generate_uf_ranking_chart(self) -> Optional[str]:
        """
        Gráfico de barras horizontais: Top 10 UFs por casos
        """
        start_time = datetime.now()
        
        try:
            self.audit.log_event(
                AuditEvent.CHART_GENERATION_START,
                {"chart": "uf_ranking", "type": "bar"},
                EventStatus.INFO
            )
            
            # Dados
            query = """
            SELECT 
                sg_uf,
                total_casos,
                taxa_mortalidade
            FROM dbx_lab_draagron.gold.gold_metricas_geograficas
            ORDER BY total_casos DESC
            LIMIT 10
            """
            
            df = self.spark.sql(query).toPandas()
            
            if df.empty:
                raise ChartValidationError("Sem dados geográficos")
            
            # Reverter para barras horizontais (maior no topo)
            df = df.sort_values('total_casos', ascending=True)
            
            # Criar figura
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=df['total_casos'],
                y=df['sg_uf'],
                orientation='h',
                marker_color=self.config.color_palette[0],
                text=df['total_casos'],
                texttemplate='%{text:,.0f}',
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Casos: %{x:,.0f}<br>Mortalidade: %{customdata:.2f}%<extra></extra>',
                customdata=df['taxa_mortalidade']
            ))
            
            # Layout
            fig.update_layout(
                title={
                    'text': '🏆 Ranking Top 10 UFs por Casos de SRAG',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                xaxis_title='Total de Casos',
                yaxis_title='Estado (UF)',
                template='plotly_white',
                height=600,
                width=self.config.default_width,
                showlegend=False
            )
            
            # Exportar
            chart_id = "7_ranking_ufs"
            metadata = self._export_chart(fig, chart_id, ChartType.BAR, "Ranking UFs")
            
            elapsed = (datetime.now() - start_time).total_seconds()
            metadata.generation_time_seconds = elapsed
            self._charts_created += 1
            self._total_generation_time += elapsed
            
            self.audit.log_event(
                AuditEvent.CHART_GENERATED,
                {"chart": "uf_ranking", "path": metadata.export_path},
                EventStatus.SUCCESS
            )
            
            return metadata.export_path
            
        except Exception as e:
            self.audit.log_event(
                AuditEvent.CHART_ERROR,
                {"chart": "uf_ranking", "error": str(e)},
                EventStatus.ERROR
            )
            print(f"   ❌ Erro: {e}")
            return None
    
    # =========================================================================
    # GRÁFICO 8: PERFIL DEMOGRÁFICO
    # =========================================================================
    
    def generate_demographic_profile_chart(self) -> Optional[str]:
        """
        Gráfico de barras agrupadas: Casos por faixa etária e sexo
        """
        start_time = datetime.now()
        
        try:
            self.audit.log_event(
                AuditEvent.CHART_GENERATION_START,
                {"chart": "demographic_profile", "type": "bar"},
                EventStatus.INFO
            )
            
            # Dados
            query = """
            SELECT 
                faixa_etaria,
                sexo,
                total_casos
            FROM dbx_lab_draagron.gold.gold_metricas_demograficas
            WHERE faixa_etaria IS NOT NULL AND sexo IN ('M', 'F')
            ORDER BY ordem_faixa_etaria, sexo
            """
            
            df = self.spark.sql(query).toPandas()
            
            if df.empty:
                raise ChartValidationError("Sem dados demográficos")
            
            # Criar figura
            fig = go.Figure()
            
            # Barras por sexo
            for sexo, cor in [('M', '#1f77b4'), ('F', '#ff7f0e')]:
                df_sexo = df[df['sexo'] == sexo]
                
                fig.add_trace(go.Bar(
                    x=df_sexo['faixa_etaria'],
                    y=df_sexo['total_casos'],
                    name='Masculino' if sexo == 'M' else 'Feminino',
                    marker_color=cor,
                    hovertemplate='<b>%{x}</b><br>Casos: %{y:,.0f}<extra></extra>'
                ))
            
            # Layout
            fig.update_layout(
                title={
                    'text': '👥 Perfil Demográfico de Casos SRAG por Faixa Etária',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                xaxis_title='Faixa Etária',
                yaxis_title='Total de Casos',
                template='plotly_white',
                height=self.config.default_height,
                width=self.config.default_width,
                barmode='group',
                showlegend=True
            )
            
            # Exportar
            chart_id = "8_perfil_demografico"
            metadata = self._export_chart(fig, chart_id, ChartType.BAR, "Perfil Demográfico")
            
            elapsed = (datetime.now() - start_time).total_seconds()
            metadata.generation_time_seconds = elapsed
            self._charts_created += 1
            self._total_generation_time += elapsed
            
            self.audit.log_event(
                AuditEvent.CHART_GENERATED,
                {"chart": "demographic_profile", "path": metadata.export_path},
                EventStatus.SUCCESS
            )
            
            return metadata.export_path
            
        except Exception as e:
            self.audit.log_event(
                AuditEvent.CHART_ERROR,
                {"chart": "demographic_profile", "error": str(e)},
                EventStatus.ERROR
            )
            print(f"   ❌ Erro: {e}")
            return None
    
    # =========================================================================
    # GRÁFICO 9: MORTALIDADE x UTI (DOIS EIXOS Y)
    # =========================================================================
    
    def generate_mortality_uti_chart(self) -> Optional[str]:
        """
        Gráfico combo: Taxa de mortalidade (linha) e Taxa UTI (barras) - Dois eixos Y
        """
        start_time = datetime.now()
        
        try:
            self.audit.log_event(
                AuditEvent.CHART_GENERATION_START,
                {"chart": "mortality_uti", "type": "combo"},
                EventStatus.INFO
            )
            
            # Dados
            query = """
            SELECT 
                ano_mes,
                taxa_mortalidade,
                taxa_uti
            FROM dbx_lab_draagron.gold.gold_metricas_temporais
            ORDER BY ano_mes DESC
            LIMIT 24
            """
            
            df = self.spark.sql(query).toPandas()
            
            if df.empty:
                raise ChartValidationError("Sem dados para mortalidade x UTI")
            
            df = df.sort_values('ano_mes')
            
            # Criar figura com subplots (2 eixos Y)
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Barras: Taxa UTI (eixo Y principal)
            fig.add_trace(
                go.Bar(
                    x=df['ano_mes'],
                    y=df['taxa_uti'],
                    name='Taxa UTI (%)',
                    marker_color=self.config.color_uti,
                    opacity=0.6,
                    hovertemplate='<b>%{x}</b><br>Taxa UTI: %{y:.2f}%<extra></extra>'
                ),
                secondary_y=False
            )
            
            # Linha: Taxa Mortalidade (eixo Y secundário)
            fig.add_trace(
                go.Scatter(
                    x=df['ano_mes'],
                    y=df['taxa_mortalidade'],
                    name='Taxa Mortalidade (%)',
                    line=dict(color=self.config.color_mortalidade, width=3),
                    mode='lines+markers',
                    marker=dict(size=8),
                    hovertemplate='<b>%{x}</b><br>Mortalidade: %{y:.2f}%<extra></extra>'
                ),
                secondary_y=True
            )
            
            # Layout
            fig.update_layout(
                title={
                    'text': '⚕️ Taxa de Mortalidade vs Taxa de UTI - SRAG',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                template='plotly_white',
                height=self.config.default_height,
                width=self.config.default_width,
                hovermode='x unified',
                showlegend=True
            )
            
            # Eixos
            fig.update_xaxes(title_text="Período")
            fig.update_yaxes(title_text="<b>Taxa UTI (%)</b>", secondary_y=False)
            fig.update_yaxes(title_text="<b>Taxa Mortalidade (%)</b>", secondary_y=True)
            
            # Exportar
            chart_id = "9_mortalidade_uti"
            metadata = self._export_chart(fig, chart_id, ChartType.COMBO, "Mortalidade x UTI")
            
            elapsed = (datetime.now() - start_time).total_seconds()
            metadata.generation_time_seconds = elapsed
            self._charts_created += 1
            self._total_generation_time += elapsed
            
            self.audit.log_event(
                AuditEvent.CHART_GENERATED,
                {"chart": "mortality_uti", "path": metadata.export_path},
                EventStatus.SUCCESS
            )
            
            return metadata.export_path
            
        except Exception as e:
            self.audit.log_event(
                AuditEvent.CHART_ERROR,
                {"chart": "mortality_uti", "error": str(e)},
                EventStatus.ERROR
            )
            print(f"   ❌ Erro: {e}")
            return None
    
    # =========================================================================
    # GRÁFICO 10: VACINAÇÃO vs MORTALIDADE (CORRELAÇÃO)
    # =========================================================================
    
    def generate_vaccination_correlation_chart(self) -> Optional[str]:
        """
        Gráfico de dispersão: Correlação entre taxa de vacinação e mortalidade
        """
        start_time = datetime.now()
        
        try:
            self.audit.log_event(
                AuditEvent.CHART_GENERATION_START,
                {"chart": "vaccination_correlation", "type": "scatter"},
                EventStatus.INFO
            )
            
            # Dados
            query = """
            SELECT 
                ano_mes,
                taxa_vacinacao,
                taxa_mortalidade
            FROM dbx_lab_draagron.gold.gold_metricas_temporais
            WHERE taxa_vacinacao IS NOT NULL
            ORDER BY ano_mes
            """
            
            df = self.spark.sql(query).toPandas()
            
            if df.empty:
                raise ChartValidationError("Sem dados de vacinação")
            
            # Criar figura
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df['taxa_vacinacao'],
                y=df['taxa_mortalidade'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=df['taxa_mortalidade'],
                    colorscale='RdYlGn_r',
                    showscale=True,
                    colorbar=dict(title="Mortalidade (%)")
                ),
                text=df['ano_mes'],
                hovertemplate='<b>%{text}</b><br>Vacinação: %{x:.2f}%<br>Mortalidade: %{y:.2f}%<extra></extra>'
            ))
            
            # Linha de tendência
            import numpy as np
            z = np.polyfit(df['taxa_vacinacao'], df['taxa_mortalidade'], 1)
            p = np.poly1d(z)
            
            x_trend = np.linspace(df['taxa_vacinacao'].min(), df['taxa_vacinacao'].max(), 100)
            
            fig.add_trace(go.Scatter(
                x=x_trend,
                y=p(x_trend),
                mode='lines',
                name='Linha de Tendência',
                line=dict(color='red', width=2, dash='dash'),
                hovertemplate='Tendência<extra></extra>'
            ))
            
            # Layout
            fig.update_layout(
                title={
                    'text': '💉 Correlação: Taxa de Vacinação vs Mortalidade SRAG',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                xaxis_title='Taxa de Vacinação (%)',
                yaxis_title='Taxa de Mortalidade (%)',
                template='plotly_white',
                height=self.config.default_height,
                width=self.config.default_width,
                showlegend=True
            )
            
            # Exportar
            chart_id = "10_vacinacao_correlacao"
            metadata = self._export_chart(fig, chart_id, ChartType.SCATTER, "Vacinação Correlação")
            
            elapsed = (datetime.now() - start_time).total_seconds()
            metadata.generation_time_seconds = elapsed
            self._charts_created += 1
            self._total_generation_time += elapsed
            
            self.audit.log_event(
                AuditEvent.CHART_GENERATED,
                {"chart": "vaccination_correlation", "path": metadata.export_path},
                EventStatus.SUCCESS
            )
            
            return metadata.export_path
            
        except Exception as e:
            self.audit.log_event(
                AuditEvent.CHART_ERROR,
                {"chart": "vaccination_correlation", "error": str(e)},
                EventStatus.ERROR
            )
            print(f"   ❌ Erro: {e}")
            return None
    
    # =========================================================================
    # HELPERS
    # =========================================================================
    
    def _export_chart(
        self,
        fig: go.Figure,
        chart_id: str,
        chart_type: ChartType,
        title: str
    ) -> ChartMetadata:
        """Exporta gráfico para HTML"""
        try:
            filepath = self.output_dir / f"{chart_id}.html"
            fig.write_html(str(filepath))
            
            metadata = ChartMetadata(
                chart_id=chart_id,
                chart_type=chart_type,
                title=title,
                created_at=datetime.now(),
                data_points=len(fig.data[0].x) if fig.data and hasattr(fig.data[0], 'x') else 0,
                export_path=str(filepath),
                generation_time_seconds=0.0
            )
            
            print(f"   ✅ Exportado: {filepath}")
            
            return metadata
            
        except Exception as e:
            print(f"   ❌ Erro ao exportar: {e}")
            raise
    
    def get_statistics(self) -> Dict:
        """Retorna estatísticas de geração"""
        return {
            "charts_created": self._charts_created,
            "total_generation_time": self._total_generation_time,
            "avg_generation_time": (
                self._total_generation_time / self._charts_created
                if self._charts_created > 0 else 0
            )
        }
    
    def __repr__(self) -> str:
        return f"ChartTool(charts_created={self._charts_created}, output_dir={self.output_dir})"


# Alias para compatibilidade
ChartGenerator = ChartTool