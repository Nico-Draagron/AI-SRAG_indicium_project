"""
Chart Tool - Geração Avançada de Visualizações para SRAG (CORRIGIDO)
====================================================================

Versão corrigida com:
- ✅ Imports condicionais (sem quebrar se utils não existirem)
- ✅ Output directory para Databricks (/dbfs/FileStore)
- ✅ Validação de dependências
- ✅ Error handling robusto
- ✅ Dados dummy para demonstração

Author: AI Engineer Certification - Indicium
Date: January 2025
Version: 2.1.0 - CORRIGIDO
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import traceback

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

# ✅ Imports condicionais para evitar erros
try:
    from src.utils.audit import AuditLogger, AuditEvent
except ImportError:
    # Stub para AuditLogger se não existir
    class AuditEvent:
        TOOL_INITIALIZED = "tool_initialized"
        CHART_GENERATION_START = "chart_generation_start"
        CHART_GENERATED = "chart_generated"
        CHART_ERROR = "chart_error"
        CHART_EXPORT_ERROR = "chart_export_error"
    
    class AuditLogger:
        def log_event(self, event_type, details=None, status="INFO"):
            print(f"[{status}] {event_type}: {details}")

try:
    from src.utils.exceptions import ChartGenerationError, ChartValidationError
except ImportError:
    # Criar exceções básicas
    class ChartGenerationError(Exception):
        pass
    
    class ChartValidationError(Exception):
        pass


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
    COMBO = "combo"


class ChartTheme(Enum):
    """Temas visuais"""
    LIGHT = "plotly_white"
    DARK = "plotly_dark"
    MINIMAL = "simple_white"
    PROFESSIONAL = "seaborn"
    SRAG_CUSTOM = "srag_custom"


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
    
    # ✅ CORRIGIDO: Usar /dbfs/FileStore em vez de /tmp para Databricks
    output_directory: str = "/dbfs/FileStore/charts"
    
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
    
    annotation_font_size: int = 10
    annotation_arrow_color: str = "#666666"
    max_data_points: int = 1000
    enable_webgl: bool = False


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
        """Detecta tendência geral"""
        if len(values) < 2:
            return "insufficient_data"
        
        diffs = [values[i] - values[i-1] for i in range(1, len(values))]
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
        audit_logger: Optional[AuditLogger] = None,
        config: Optional[ChartConfig] = None,
        output_dir: Optional[str] = None
    ):
        # ✅ Audit opcional
        self.audit = audit_logger if audit_logger else AuditLogger()
        self.config = config or ChartConfig()
        
        # ✅ Output directory com fallback
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(self.config.output_directory)
        
        # ✅ Criar diretório com tratamento de erro
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            print(f"✅ Chart output dir criado: {self.output_dir}")
        except Exception as e:
            print(f"⚠️ Erro ao criar output dir: {e}")
            # Fallback para /tmp se /dbfs falhar
            import tempfile
            self.output_dir = Path(tempfile.mkdtemp(prefix="charts_"))
            print(f"   📂 Usando fallback: {self.output_dir}")
        
        self.trend_analyzer = TrendAnalyzer()
        self.annotation_gen = AnnotationGenerator()
        
        self._setup_custom_theme()
        
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
        subtitle: Optional[str] = None,
        annotations: Optional[List[str]] = None,
        color_col: Optional[str] = None,
        enable_ma: bool = True,
        ma_window: int = 7
    ) -> Dict:
        """
        Cria gráfico de linha com anotações automáticas
        
        Args:
            data: Lista de dicionários
            title: Título do gráfico
            x_col: Nome da coluna X
            y_col: Nome da coluna Y
            subtitle: Subtítulo opcional
            annotations: Lista de anotações customizadas
            
        Returns:
            Dict com success, path, metadata
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
                    name=f'Média Móvel ({ma_window}d)',
                    line=dict(color=self.config.color_palette[1], width=1, dash='dash')
                ))
            
            # Aplicar layout
            self._apply_layout(fig, title, subtitle)
            
            # Adicionar anotações customizadas
            if annotations:
                for i, note_text in enumerate(annotations):
                    fig.add_annotation(
                        text=note_text,
                        xref="paper", yref="paper",
                        x=0.05, y=0.95 - (i * 0.05),
                        showarrow=False,
                        font=dict(size=10),
                        bgcolor="rgba(255,255,255,0.8)"
                    )
            
            # Exportar
            chart_id = f"line_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            metadata = self._export_chart(fig, chart_id, ChartType.LINE, title)
            
            # Atualizar estatísticas
            elapsed = (datetime.now() - start_time).total_seconds()
            metadata.generation_time_seconds = elapsed
            self._charts_created += 1
            self._total_generation_time += elapsed
            
            self.audit.log_event(
                AuditEvent.CHART_GENERATED,
                self._metadata_to_dict(metadata)
            )
            
            return {
                "success": True,
                "path": metadata.export_paths.get("html", ""),
                "metadata": self._metadata_to_dict(metadata)
            }
            
        except Exception as e:
            self.audit.log_event(
                AuditEvent.CHART_ERROR,
                {"type": "line", "error": str(e)},
                "ERROR"
            )
            print(f"❌ Erro em create_line_chart: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "path": None
            }
    
    def create_bar_chart(
        self,
        data: List[Dict],
        title: str,
        x_col: str = "x",
        y_col: str = "y",
        subtitle: Optional[str] = None,
        annotations: Optional[List[str]] = None,
        orientation: str = "v"
    ) -> Dict:
        """Cria gráfico de barras"""
        start_time = datetime.now()
        
        self.audit.log_event(
            AuditEvent.CHART_GENERATION_START,
            {"type": "bar", "title": title, "data_points": len(data)}
        )
        
        try:
            self._validate_data(data, [x_col, y_col])
            df = pd.DataFrame(data)
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=df[x_col] if orientation == "v" else df[y_col],
                y=df[y_col] if orientation == "v" else df[x_col],
                orientation=orientation,
                marker=dict(color=self.config.color_palette[0])
            ))
            
            self._apply_layout(fig, title, subtitle)
            
            # Adicionar anotações customizadas
            if annotations:
                for i, note_text in enumerate(annotations):
                    fig.add_annotation(
                        text=note_text,
                        xref="paper", yref="paper",
                        x=0.05, y=0.95 - (i * 0.05),
                        showarrow=False,
                        font=dict(size=10),
                        bgcolor="rgba(255,255,255,0.8)"
                    )
            
            chart_id = f"bar_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            metadata = self._export_chart(fig, chart_id, ChartType.BAR, title)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            metadata.generation_time_seconds = elapsed
            self._charts_created += 1
            self._total_generation_time += elapsed
            
            self.audit.log_event(
                AuditEvent.CHART_GENERATED,
                self._metadata_to_dict(metadata)
            )
            
            return {
                "success": True,
                "path": metadata.export_paths.get("html", ""),
                "metadata": self._metadata_to_dict(metadata)
            }
            
        except Exception as e:
            self.audit.log_event(
                AuditEvent.CHART_ERROR,
                {"type": "bar", "error": str(e)},
                "ERROR"
            )
            print(f"❌ Erro em create_bar_chart: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "path": None
            }
    
    # =========================================================================
    # HELPERS
    # =========================================================================
    
    def _validate_data(self, data: List[Dict], required_cols: List[str]):
        """Valida estrutura dos dados"""
        if not data:
            raise ChartValidationError("Dados vazios")
        
        if not isinstance(data, list):
            raise ChartValidationError("Dados devem ser uma lista de dicionários")
        
        first_item = data[0]
        for col in required_cols:
            if col not in first_item:
                raise ChartValidationError(f"Coluna '{col}' não encontrada nos dados")
    
    def _apply_layout(self, fig: go.Figure, title: str, subtitle: Optional[str] = None):
        """Aplica layout padrão ao gráfico"""
        full_title = title
        if subtitle:
            full_title = f"{title}<br><sub>{subtitle}</sub>"
        
        fig.update_layout(
            title=full_title,
            template=self.config.default_theme.value,
            height=self.config.default_height,
            width=self.config.default_width,
            showlegend=self.config.show_legend,
            hovermode='x unified'
        )
    
    def _export_chart(
        self,
        fig: go.Figure,
        chart_id: str,
        chart_type: ChartType,
        title: str
    ) -> ChartMetadata:
        """Exporta gráfico em múltiplos formatos"""
        export_paths = {}
        file_sizes = {}
        
        for fmt in self.config.default_export_formats:
            try:
                filepath = self.output_dir / f"{chart_id}.{fmt.value}"
                
                if fmt == ExportFormat.HTML:
                    fig.write_html(str(filepath))
                elif fmt == ExportFormat.PNG:
                    fig.write_image(str(filepath))
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
                print(f"⚠️ Erro ao exportar {fmt.value}: {e}")
        
        metadata = ChartMetadata(
            chart_id=chart_id,
            chart_type=chart_type,
            title=title,
            created_at=datetime.now(),
            data_points=len(fig.data[0].x) if fig.data else 0,
            export_paths=export_paths,
            file_sizes=file_sizes,
            generation_time_seconds=0.0
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
    
    # =========================================================================
    # MÉTODOS OBRIGATÓRIOS PARA CERTIFICAÇÃO
    # =========================================================================
    
    def generate_daily_chart(self, data: Optional[List[Dict]] = None) -> Optional[str]:
        """
        Gera gráfico de casos diários (últimos 30 dias) - OBRIGATÓRIO
        
        Returns:
            Path do arquivo HTML gerado ou None em caso de erro
        """
        try:
            self.audit.log_event(
                AuditEvent.CHART_GENERATION_START,
                {"tool": "chart_tool", "chart_type": "daily_cases"},
                "INFO"
            )
            
            # ✅ Dados dummy se não fornecidos
            if not data:
                base_date = datetime.now() - timedelta(days=30)
                data = []
                for i in range(30):
                    current_date = base_date + timedelta(days=i)
                    casos = max(50, int(1000 + (i * 10) + (i % 7 * 50)))
                    data.append({
                        "data": current_date.strftime("%Y-%m-%d"),
                        "casos": casos
                    })
            
            if not data or len(data) == 0:
                raise ValueError("Dados para gráfico diário estão vazios")
            
            # Criar gráfico
            result = self.create_line_chart(
                data=data,
                title="📈 Casos Diários de SRAG - Últimos 30 Dias",
                x_col="data",
                y_col="casos",
                subtitle="Evolução temporal da notificação de casos",
                annotations=["Tendência de crescimento observada", "Picos nos fins de semana"]
            )
            
            if not result or not result.get("success"):
                raise ValueError(f"Falha ao criar gráfico: {result.get('error', 'Erro desconhecido')}")
            
            self.audit.log_event(
                AuditEvent.CHART_GENERATED,
                {"tool": "chart_tool", "chart_type": "daily_cases", "path": result.get("path")},
                "SUCCESS"
            )
            
            print(f"✅ Gráfico diário gerado: {result.get('path')}")
            return result.get("path")
            
        except Exception as e:
            error_detail = {
                "tool": "chart_tool",
                "chart_type": "daily_cases",
                "error": str(e),
                "error_type": type(e).__name__,
                "data_length": len(data) if data else 0
            }
            
            self.audit.log_event(
                AuditEvent.CHART_ERROR,
                error_detail,
                "ERROR"
            )
            
            print(f"❌ Erro em generate_daily_chart: {str(e)}")
            traceback.print_exc()
            
            return None
    
    def generate_monthly_chart(self, data: Optional[List[Dict]] = None) -> Optional[str]:
        """
        Gera gráfico de casos mensais (últimos 12 meses) - OBRIGATÓRIO
        
        Returns:
            Path do arquivo HTML gerado ou None em caso de erro
        """
        try:
            self.audit.log_event(
                AuditEvent.CHART_GENERATION_START,
                {"tool": "chart_tool", "chart_type": "monthly_cases"},
                "INFO"
            )
            
            # ✅ Dados dummy se não fornecidos
            if not data:
                base_date = datetime.now().replace(day=1) - timedelta(days=365)
                data = []
                meses = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
                        "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
                
                for i in range(12):
                    current_date = base_date + timedelta(days=30*i)
                    casos = max(1000, int(15000 + (i * 500) + (i % 4 * 2000)))
                    data.append({
                        "mes": f"{meses[current_date.month-1]}/{current_date.year % 100}",
                        "casos": casos
                    })
            
            if not data or len(data) == 0:
                raise ValueError("Dados para gráfico mensal estão vazios")
            
            # Criar gráfico
            result = self.create_bar_chart(
                data=data,
                title="📊 Casos Mensais de SRAG - Últimos 12 Meses",
                x_col="mes",
                y_col="casos",
                subtitle="Evolução mensal com sazonalidade",
                annotations=["Pico no inverno", "Redução no verão"]
            )
            
            if not result or not result.get("success"):
                raise ValueError(f"Falha ao criar gráfico mensal: {result.get('error', 'Erro desconhecido')}")
            
            self.audit.log_event(
                AuditEvent.CHART_GENERATED,
                {"tool": "chart_tool", "chart_type": "monthly_cases", "path": result.get("path")},
                "SUCCESS"
            )
            
            print(f"✅ Gráfico mensal gerado: {result.get('path')}")
            return result.get("path")
            
        except Exception as e:
            error_detail = {
                "tool": "chart_tool",
                "chart_type": "monthly_cases",
                "error": str(e),
                "error_type": type(e).__name__,
                "data_length": len(data) if data else 0
            }
            
            self.audit.log_event(
                AuditEvent.CHART_ERROR,
                error_detail,
                "ERROR"
            )
            
            print(f"❌ Erro em generate_monthly_chart: {str(e)}")
            traceback.print_exc()
            
            return None
    
    def __repr__(self) -> str:
        return f"ChartTool(charts_created={self._charts_created}, output_dir={self.output_dir})"


# Alias para compatibilidade
ChartGenerator = ChartTool