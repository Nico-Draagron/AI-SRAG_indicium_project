"""
Chart Tool - Geração Avançada de Visualizações para SRAG
=========================================================

Ferramenta para criação de gráficos interativos e estáticos usando Plotly
com templates customizados, anotações automáticas e exportação múltipla.

Features:
    - Múltiplos tipos de gráficos (line, bar, area, scatter, heatmap)
    - Templates personalizados para SRAG
    - Anotações automáticas de tendências
    - Exportação em HTML, PNG, SVG
    - Gráficos compostos (subplots)
    - Temas responsivos
    - Otimização de performance

Author: AI Engineer Certification - Indicium
Date: January 2025
Version: 2.0.0
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

from ...utils.audit import AuditLogger, AuditEvent
from ...utils.exceptions import ChartGenerationError, ChartValidationError


# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

class ChartType(Enum):
    """Tipos de gráficos suportados"""
    LINE = "line"
    BAR = "bar"
    AREA = "area"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    PIE = "pie"
    WATERFALL = "waterfall"
    FUNNEL = "funnel"
    GAUGE = "gauge"
    COMBO = "combo"  # Combinação de tipos


class ChartTheme(Enum):
    """Temas visuais"""
    LIGHT = "plotly_white"
    DARK = "plotly_dark"
    MINIMAL = "simple_white"
    PROFESSIONAL = "seaborn"
    SRAG_CUSTOM = "srag_custom"  # Tema customizado


class ExportFormat(Enum):
    """Formatos de exportação"""
    HTML = "html"
    PNG = "png"
    SVG = "svg"
    JSON = "json"
    PDF = "pdf"


@dataclass
class ChartConfig:
    """Configuração global de gráficos"""
    default_theme: ChartTheme = ChartTheme.LIGHT
    default_height: int = 500
    default_width: int = 900
    enable_interactivity: bool = True
    enable_annotations: bool = True
    enable_trend_lines: bool = True
    show_grid: bool = True
    show_legend: bool = True
    output_directory: str = "/tmp/charts"
    default_export_formats: List[ExportFormat] = field(default_factory=lambda: [ExportFormat.HTML])
    
    # Cores customizadas SRAG
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
    
    # Configurações de anotação
    annotation_font_size: int = 10
    annotation_arrow_color: str = "#666666"
    
    # Otimização
    max_data_points: int = 1000
    enable_webgl: bool = False  # Para grandes datasets


@dataclass
class ChartMetadata:
    """Metadados do gráfico gerado"""
    chart_id: str
    chart_type: ChartType
    title: str
    created_at: datetime
    data_points: int
    export_paths: Dict[str, str]
    file_sizes: Dict[str, int]
    generation_time_seconds: float


# =============================================================================
# ANALISADOR DE TENDÊNCIAS
# =============================================================================

class TrendAnalyzer:
    """Analisa tendências em séries temporais"""
    
    @staticmethod
    def detect_trend(values: List[float]) -> str:
        """Detecta tendência geral (crescente, decrescente, estável)"""
        if len(values) < 2:
            return "insufficient_data"
        
        # Calcular diferenças
        diffs = [values[i] - values[i-1] for i in range(1, len(values))]
        
        # Contar tendências
        increases = sum(1 for d in diffs if d > 0)
        decreases = sum(1 for d in diffs if d < 0)
        
        total = len(diffs)
        
        if increases / total > 0.6:
            return "crescente"
        elif decreases / total > 0.6:
            return "decrescente"
        else:
            return "estável"
    
    @staticmethod
    def calculate_growth_rate(values: List[float]) -> float:
        """Calcula taxa de crescimento média"""
        if len(values) < 2:
            return 0.0
        
        first = values[0]
        last = values[-1]
        
        if first == 0:
            return 0.0
        
        return ((last - first) / first) * 100
    
    @staticmethod
    def find_peaks(values: List[float], threshold: float = 0.1) -> List[int]:
        """Encontra picos na série"""
        peaks = []
        
        for i in range(1, len(values) - 1):
            if values[i] > values[i-1] and values[i] > values[i+1]:
                # Verificar se é pico significativo
                if values[i] > max(values) * (1 - threshold):
                    peaks.append(i)
        
        return peaks
    
    @staticmethod
    def calculate_moving_average(values: List[float], window: int = 7) -> List[float]:
        """Calcula média móvel"""
        if len(values) < window:
            return values
        
        ma = []
        for i in range(len(values)):
            if i < window - 1:
                ma.append(None)
            else:
                ma.append(sum(values[i-window+1:i+1]) / window)
        
        return ma


# =============================================================================
# GERADOR DE ANOTAÇÕES
# =============================================================================

class AnnotationGenerator:
    """Gera anotações automáticas para gráficos"""
    
    @staticmethod
    def create_trend_annotation(
        x_position: Any,
        y_position: float,
        trend: str,
        growth_rate: float
    ) -> Dict:
        """Cria anotação de tendência"""
        text = f"{trend.capitalize()}"
        if abs(growth_rate) > 1:
            text += f" ({growth_rate:+.1f}%)"
        
        return {
            "x": x_position,
            "y": y_position,
            "text": text,
            "showarrow": True,
            "arrowhead": 2,
            "arrowsize": 1,
            "arrowwidth": 1,
            "arrowcolor": "#666666",
            "font": {"size": 10, "color": "#333333"},
            "bgcolor": "rgba(255, 255, 255, 0.8)",
            "bordercolor": "#666666",
            "borderwidth": 1,
            "borderpad": 4
        }
    
    @staticmethod
    def create_peak_annotation(x_position: Any, y_position: float, value: float) -> Dict:
        """Cria anotação de pico"""
        return {
            "x": x_position,
            "y": y_position,
            "text": f"Pico: {value:,.0f}",
            "showarrow": True,
            "arrowhead": 2,
            "ax": 0,
            "ay": -40,
            "font": {"size": 10, "color": "#d62728"},
            "bgcolor": "rgba(255, 255, 255, 0.9)",
            "bordercolor": "#d62728",
            "borderwidth": 2
        }
    
    @staticmethod
    def create_threshold_line(
        y_value: float,
        label: str,
        color: str = "#ff7f0e"
    ) -> Dict:
        """Cria linha de threshold"""
        return {
            "type": "line",
            "y0": y_value,
            "y1": y_value,
            "x0": 0,
            "x1": 1,
            "xref": "paper",
            "line": {
                "color": color,
                "width": 2,
                "dash": "dash"
            },
            "label": {
                "text": label,
                "textposition": "end",
                "font": {"size": 10}
            }
        }


# =============================================================================
# CHART TOOL PRINCIPAL
# =============================================================================

class ChartTool:
    """
    Ferramenta de geração de gráficos para SRAG
    
    Features:
        - Múltiplos tipos de gráficos
        - Anotações automáticas
        - Exportação multi-formato
        - Templates customizados
        - Análise de tendências
    
    Example:
        >>> chart_tool = ChartTool(audit_logger=logger)
        >>> result = chart_tool.create_line_chart(
        ...     data=[{"x": "2025-01", "y": 1000}, {"x": "2025-02", "y": 1200}],
        ...     title="Casos Mensais SRAG",
        ...     x_col="x",
        ...     y_col="y"
        ... )
        >>> print(result["path"])
    """
    
    def __init__(
        self,
        audit_logger: AuditLogger,
        config: Optional[ChartConfig] = None,
        output_dir: str = "/tmp"
    ):
        self.audit = audit_logger
        self.config = config or ChartConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.trend_analyzer = TrendAnalyzer()
        self.annotation_gen = AnnotationGenerator()
        
        # Configurar tema SRAG customizado
        self._setup_custom_theme()
        
        # Estatísticas
        self._charts_created = 0
        self._total_generation_time = 0.0
        
        self.audit.log_event(
            AuditEvent.TOOL_INITIALIZED,
            {
                "tool": "ChartTool",
                "output_dir": str(self.output_dir),
                "theme": self.config.default_theme.value
            }
        )
    
    def _setup_custom_theme(self):
        """Configura tema customizado SRAG"""
        if self.config.default_theme == ChartTheme.SRAG_CUSTOM:
            pio.templates["srag_custom"] = go.layout.Template(
                layout=go.Layout(
                    font={"family": "Arial, sans-serif", "size": 12},
                    title_font={"size": 18, "color": "#1f77b4"},
                    colorway=self.config.color_palette,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    hovermode="x unified"
                )
            )
    
    # =========================================================================
    # CRIAÇÃO DE GRÁFICOS BÁSICOS
    # =========================================================================
    
    def create_line_chart(
        self,
        data: List[Dict],
        title: str,
        x_col: str = "x",
        y_col: str = "y",
        color_col: Optional[str] = None,
        enable_ma: bool = True,
        ma_window: int = 7
    ) -> Dict:
        """
        Cria gráfico de linha
        
        Args:
            data: Lista de dicionários com dados
            title: Título do gráfico
            x_col: Nome da coluna X
            y_col: Nome da coluna Y
            color_col: Coluna para agrupamento/cores (opcional)
            enable_ma: Adicionar média móvel
            ma_window: Janela da média móvel
            
        Returns:
            Dict com metadata e paths
        """
        start_time = datetime.now()
        
        self.audit.log_event(
            AuditEvent.CHART_GENERATION_START,
            {"type": "line", "title": title, "data_points": len(data)}
        )
        
        try:
            # Validar dados
            self._validate_data(data, [x_col, y_col])
            
            # Converter para DataFrame
            df = pd.DataFrame(data)
            
            # Criar figura
            fig = go.Figure()
            
            # Adicionar linha principal
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode='lines+markers',
                name=y_col,
                line=dict(color=self.config.color_palette[0], width=2),
                marker=dict(size=6)
            ))
            
            # Adicionar média móvel se habilitado
            if enable_ma and len(df) >= ma_window:
                ma_values = self.trend_analyzer.calculate_moving_average(
                    df[y_col].tolist(),
                    window=ma_window
                )
                
                fig.add_trace(go.Scatter(
                    x=df[x_col],
                    y=ma_values,
                    mode='lines',
                    name=f'MA{ma_window}',
                    line=dict(color=self.config.color_palette[1], width=2, dash='dash'),
                    opacity=0.7
                ))
            
            # Adicionar anotações
            if self.config.enable_annotations:
                annotations = self._generate_line_annotations(df, x_col, y_col)
                fig.update_layout(annotations=annotations)
            
            # Aplicar layout
            fig.update_layout(
                title=title,
                xaxis_title="",
                yaxis_title="Casos",
                template=self.config.default_theme.value,
                height=self.config.default_height,
                width=self.config.default_width,
                hovermode='x unified',
                showlegend=self.config.show_legend
            )
            
            # Exportar
            metadata = self._export_chart(fig, ChartType.LINE, title)
            
            generation_time = (datetime.now() - start_time).total_seconds()
            metadata.generation_time_seconds = generation_time
            
            self._charts_created += 1
            self._total_generation_time += generation_time
            
            self.audit.log_event(
                AuditEvent.CHART_GENERATED,
                {
                    "chart_id": metadata.chart_id,
                    "type": "line",
                    "generation_time": generation_time
                }
            )
            
            return {
                "success": True,
                "chart_type": "line",
                "path": metadata.export_paths.get("html", ""),
                "metadata": self._metadata_to_dict(metadata)
            }
            
        except Exception as e:
            self.audit.log_event(
                AuditEvent.CHART_ERROR,
                {"error": str(e), "title": title},
                "ERROR"
            )
            
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def create_bar_chart(
        self,
        data: List[Dict],
        title: str,
        x_col: str = "x",
        y_col: str = "y",
        color_col: Optional[str] = None,
        orientation: str = "v"
    ) -> Dict:
        """
        Cria gráfico de barras
        
        Args:
            data: Lista de dicionários
            title: Título
            x_col: Coluna X
            y_col: Coluna Y
            color_col: Coluna para cores
            orientation: 'v' (vertical) ou 'h' (horizontal)
            
        Returns:
            Dict com resultado
        """
        start_time = datetime.now()
        
        try:
            self._validate_data(data, [x_col, y_col])
            df = pd.DataFrame(data)
            
            # Criar figura
            if color_col and color_col in df.columns:
                fig = px.bar(
                    df,
                    x=x_col if orientation == "v" else y_col,
                    y=y_col if orientation == "v" else x_col,
                    color=color_col,
                    title=title,
                    orientation=orientation,
                    color_discrete_sequence=self.config.color_palette
                )
            else:
                fig = go.Figure(data=[
                    go.Bar(
                        x=df[x_col] if orientation == "v" else df[y_col],
                        y=df[y_col] if orientation == "v" else df[x_col],
                        marker_color=self.config.color_palette[0],
                        orientation=orientation
                    )
                ])
            
            # Layout
            fig.update_layout(
                title=title,
                template=self.config.default_theme.value,
                height=self.config.default_height,
                width=self.config.default_width,
                showlegend=self.config.show_legend
            )
            
            # Exportar
            metadata = self._export_chart(fig, ChartType.BAR, title)
            metadata.generation_time_seconds = (datetime.now() - start_time).total_seconds()
            
            self._charts_created += 1
            
            return {
                "success": True,
                "chart_type": "bar",
                "path": metadata.export_paths.get("html", ""),
                "metadata": self._metadata_to_dict(metadata)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def create_area_chart(
        self,
        data: List[Dict],
        title: str,
        x_col: str = "x",
        y_col: str = "y"
    ) -> Dict:
        """Cria gráfico de área"""
        try:
            df = pd.DataFrame(data)
            
            fig = go.Figure(data=go.Scatter(
                x=df[x_col],
                y=df[y_col],
                fill='tozeroy',
                mode='lines',
                line=dict(color=self.config.color_palette[0], width=2)
            ))
            
            fig.update_layout(
                title=title,
                template=self.config.default_theme.value,
                height=self.config.default_height,
                width=self.config.default_width
            )
            
            metadata = self._export_chart(fig, ChartType.AREA, title)
            
            return {
                "success": True,
                "chart_type": "area",
                "path": metadata.export_paths.get("html", ""),
                "metadata": self._metadata_to_dict(metadata)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def create_heatmap(
        self,
        data: List[Dict],
        title: str,
        x_col: str,
        y_col: str,
        z_col: str
    ) -> Dict:
        """Cria mapa de calor"""
        try:
            df = pd.DataFrame(data)
            
            # Pivot para matriz
            pivot = df.pivot(index=y_col, columns=x_col, values=z_col)
            
            fig = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale='RdYlBu_r',
                hoverongaps=False
            ))
            
            fig.update_layout(
                title=title,
                template=self.config.default_theme.value,
                height=self.config.default_height,
                width=self.config.default_width
            )
            
            metadata = self._export_chart(fig, ChartType.HEATMAP, title)
            
            return {
                "success": True,
                "chart_type": "heatmap",
                "path": metadata.export_paths.get("html", ""),
                "metadata": self._metadata_to_dict(metadata)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # =========================================================================
    # GRÁFICOS COMPOSTOS
    # =========================================================================
    
    def create_combo_chart(
        self,
        data_line: List[Dict],
        data_bar: List[Dict],
        title: str,
        x_col: str = "x",
        y_col_line: str = "y_line",
        y_col_bar: str = "y_bar"
    ) -> Dict:
        """Cria gráfico combinado (linha + barra)"""
        try:
            df_line = pd.DataFrame(data_line)
            df_bar = pd.DataFrame(data_bar)
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Adicionar barras
            fig.add_trace(
                go.Bar(x=df_bar[x_col], y=df_bar[y_col_bar], name="Casos"),
                secondary_y=False
            )
            
            # Adicionar linha
            fig.add_trace(
                go.Scatter(
                    x=df_line[x_col],
                    y=df_line[y_col_line],
                    name="Taxa",
                    mode='lines+markers'
                ),
                secondary_y=True
            )
            
            fig.update_layout(
                title=title,
                template=self.config.default_theme.value,
                height=self.config.default_height,
                width=self.config.default_width
            )
            
            fig.update_yaxes(title_text="Casos", secondary_y=False)
            fig.update_yaxes(title_text="Taxa (%)", secondary_y=True)
            
            metadata = self._export_chart(fig, ChartType.COMBO, title)
            
            return {
                "success": True,
                "chart_type": "combo",
                "path": metadata.export_paths.get("html", ""),
                "metadata": self._metadata_to_dict(metadata)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # =========================================================================
    # MÉTODOS AUXILIARES
    # =========================================================================
    
    def _validate_data(self, data: List[Dict], required_cols: List[str]):
        """Valida dados de entrada"""
        if not data:
            raise ChartValidationError("Dados vazios")
        
        if len(data) > self.config.max_data_points:
            raise ChartValidationError(
                f"Muitos pontos ({len(data)}). Máximo: {self.config.max_data_points}"
            )
        
        first_row = data[0]
        for col in required_cols:
            if col not in first_row:
                raise ChartValidationError(f"Coluna '{col}' não encontrada")
    
    def _generate_line_annotations(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str
    ) -> List[Dict]:
        """Gera anotações para gráfico de linha"""
        annotations = []
        
        values = df[y_col].tolist()
        
        # Detectar tendência
        trend = self.trend_analyzer.detect_trend(values)
        growth_rate = self.trend_analyzer.calculate_growth_rate(values)
        
        # Adicionar anotação de tendência no final
        if len(df) > 0:
            last_idx = len(df) - 1
            annotation = self.annotation_gen.create_trend_annotation(
                x_position=df[x_col].iloc[last_idx],
                y_position=values[last_idx],
                trend=trend,
                growth_rate=growth_rate
            )
            annotations.append(annotation)
        
        # Adicionar anotações de picos
        peaks = self.trend_analyzer.find_peaks(values)
        for peak_idx in peaks[:3]:  # Máximo 3 picos
            annotation = self.annotation_gen.create_peak_annotation(
                x_position=df[x_col].iloc[peak_idx],
                y_position=values[peak_idx],
                value=values[peak_idx]
            )
            annotations.append(annotation)
        
        return annotations
    
    def _export_chart(
        self,
        fig: go.Figure,
        chart_type: ChartType,
        title: str
    ) -> ChartMetadata:
        """Exporta gráfico em múltiplos formatos"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_id = f"{chart_type.value}_{timestamp}"
        
        export_paths = {}
        file_sizes = {}
        
        # Exportar em cada formato configurado
        for fmt in self.config.default_export_formats:
            filename = f"{chart_id}.{fmt.value}"
            filepath = self.output_dir / filename
            
            try:
                if fmt == ExportFormat.HTML:
                    fig.write_html(str(filepath))
                elif fmt == ExportFormat.PNG:
                    fig.write_image(str(filepath), format="png")
                elif fmt == ExportFormat.SVG:
                    fig.write_image(str(filepath), format="svg")
                elif fmt == ExportFormat.JSON:
                    fig.write_json(str(filepath))
                
                export_paths[fmt.value] = str(filepath)
                file_sizes[fmt.value] = filepath.stat().st_size if filepath.exists() else 0
                
            except Exception as e:
                self.audit.log_event(
                    AuditEvent.CHART_EXPORT_ERROR,
                    {"format": fmt.value, "error": str(e)},
                    "WARNING"
                )
        
        # Criar metadata
        metadata = ChartMetadata(
            chart_id=chart_id,
            chart_type=chart_type,
            title=title,
            created_at=datetime.now(),
            data_points=len(fig.data[0].x) if fig.data else 0,
            export_paths=export_paths,
            file_sizes=file_sizes,
            generation_time_seconds=0.0  # Será preenchido depois
        )
        
        return metadata
    
    def _metadata_to_dict(self, metadata: ChartMetadata) -> Dict:
        """Converte metadata para dict"""
        return {
            "chart_id": metadata.chart_id,
            "chart_type": metadata.chart_type.value,
            "title": metadata.title,
            "created_at": metadata.created_at.isoformat(),
            "data_points": metadata.data_points,
            "export_paths": metadata.export_paths,
            "file_sizes": metadata.file_sizes,
            "generation_time_seconds": metadata.generation_time_seconds
        }
    
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
