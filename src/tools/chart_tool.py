"""
Chart Tool — Geração de Visualizações SRAG
==========================================

Responsabilidade: gerar gráficos Plotly interativos a partir de dados das
tabelas Gold do Unity Catalog, exportando cada gráfico como HTML leve e
como PNG estático via Kaleido.

Dois modos de uso:
    generate_all_charts()    : conjunto fixo de 5 gráficos padrão, opt-in.
    generate_custom_chart()  : gráfico único ad-hoc com inteligência visual,
                               invocado pelo nó execute_chart com dados
                               pré-processados via SQL parametrizada.

Os métodos públicos create_*_chart() existem para geração pontual fora do
pipeline principal — testes unitários e notebooks exploratórios.

Melhorias desta versão
----------------------
- Camada de humanização de nomes técnicos (LABEL_MAP).
- Detecção automática de métricas percentuais com formatação de eixo/hover.
- Troca automática para barra horizontal quando há muitas categorias ou
  labels longos.
- generate_custom_chart() valida colunas, infere natureza do dado, corrige
  escolhas inadequadas de gráfico e aplica defaults inteligentes.
- Novos métodos especializados: create_grouped_bar_chart, create_top_n_chart,
  create_year_comparison_chart, create_rate_comparison_chart.
- Hover templates ricos e padronizados por tipo de dado.
- Ordenação automática de categorias e datas.
- Bloqueio de pie chart para alta cardinalidade (> MAX_PIE_CATEGORIES).
- Layout limpo, analítico e responsivo com subtítulo de fonte/período.
"""

import importlib.util
import subprocess
import sys
import threading
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import time

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


# =============================================================================
# HUMANIZAÇÃO DE LABELS
# =============================================================================

LABEL_MAP: Dict[str, str] = {
    # Temporais
    "ano":              "Ano",
    "ano_mes":          "Período",
    "mes":              "Mês",
    "data_referencia":  "Data",
    "dt_sintomas":      "Data de Sintomas",
    # Geográficos
    "sg_uf":            "UF",
    "uf":               "UF",
    "regiao":           "Região",
    "municipio":        "Município",
    # Demográficos
    "faixa_etaria":         "Faixa etária",
    "faixa_etaria_label":   "Faixa etária",
    "sexo_label":           "Sexo",
    "sexo":                 "Sexo",
    # Contagens
    "total_casos":      "Total de casos",
    "casos_dia":        "Casos no dia",
    "total_obitos":     "Total de óbitos",
    "obitos":           "Óbitos",
    # Taxas / percentuais
    "taxa_mortalidade": "Taxa de mortalidade (%)",
    "taxa_uti":         "Taxa de ocupação de UTI (%)",
    "taxa_vacinacao":   "Taxa de vacinação (%)",
    "taxa_hospitalizacao": "Taxa de hospitalização (%)",
    "taxa_obito":       "Taxa de óbito (%)",
    # Virais
    "COVID_19":             "COVID-19",
    "Influenza":            "Influenza",
    "Outro_Virus":          "Outro vírus",
    "Sem_Classificacao":    "Sem classificação",
}

# Sufixos que identificam coluna percentual quando o nome não está no LABEL_MAP
_PCT_SUFFIXES = ("taxa_", "perc_", "_pct", "_rate", "proporcao_")

# Cardinalidade máxima aceita para pie chart
MAX_PIE_CATEGORIES = 8

# Threshold para troca automática para barra horizontal
_H_BAR_THRESHOLD_CATEGORIES = 8
_H_BAR_THRESHOLD_LABEL_LEN  = 6   # comprimento médio de label


def humanize(col: str) -> str:
    """Retorna rótulo legível para o nome técnico de coluna."""
    return LABEL_MAP.get(col, col.replace("_", " ").title())


def _is_percent_col(col: str) -> bool:
    """Heurística: True se a coluna provavelmente representa uma taxa/percentual."""
    if col in LABEL_MAP and "%" in LABEL_MAP[col]:
        return True
    c = col.lower()
    return any(c.startswith(s) or c.endswith(s.strip("_")) for s in _PCT_SUFFIXES)


def _is_temporal_col(col: str) -> bool:
    """Heurística: True se a coluna representa uma dimensão temporal."""
    c = col.lower()
    return any(k in c for k in ("data", "dt_", "ano_mes", "ano", "mes", "semana", "periodo"))


# =============================================================================
# VERIFICAÇÃO DE KALEIDO
# =============================================================================

def _check_kaleido_available() -> bool:
    if importlib.util.find_spec("kaleido") is None:
        return False
    try:
        import io as _io
        pio.write_image(go.Figure(), _io.BytesIO(), format="png", width=1, height=1)
        return True
    except Exception:
        return False


try:
    from src.utils.audit import AuditLogger, AuditEvent, EventStatus
except ImportError:
    class AuditEvent:
        TOOL_INITIALIZED       = "tool_initialized"
        TOOL_DEGRADED          = "tool_degraded"
        CHART_GENERATION_START = "chart_generation_start"
        CHART_GENERATED        = "chart_generated"
        CHART_ERROR            = "chart_error"
        CHART_WRITE_ERROR      = "chart_write_error"
        CHART_STAT_ERROR       = "chart_stat_error"
        CHART_CLEANUP          = "chart_cleanup"

    class EventStatus:
        INFO    = "INFO"
        SUCCESS = "SUCCESS"
        ERROR   = "ERROR"
        WARNING = "WARNING"

    class AuditLogger:
        def log_event(self, event_type, details=None, status="INFO"):
            print(f"[{status}] {event_type}: {details}")


# =============================================================================
# TIPOS E CONFIGURAÇÃO
# =============================================================================

class ChartType(Enum):
    LINE       = "line"
    BAR        = "bar"
    AREA       = "area"
    SCATTER    = "scatter"
    HEATMAP    = "heatmap"
    PIE        = "pie"
    COMBO      = "combo"
    MULTI_LINE = "multi_line"
    MENSAL     = "mensal"


class ChartTheme(Enum):
    LIGHT   = "plotly_white"
    DARK    = "plotly_dark"
    MINIMAL = "simple_white"


_DEFAULT_OUTPUT_BASE = "/Volumes/dbx_srag_lab/default/srag_outputs/charts"


@dataclass
class ChartConfig:
    default_theme:        ChartTheme = ChartTheme.LIGHT
    default_height:       int        = 520
    default_width:        int        = 960
    enable_interactivity: bool       = True
    color_palette: List[str] = field(default_factory=lambda: [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    ])


@dataclass
class ChartMetadata:
    chart_id:                str
    chart_type:              ChartType
    title:                   str
    created_at:              datetime
    data_points:             int
    export_path:             str
    file_size:               int
    generation_time_seconds: float
    export_path_png:         Optional[str] = None


# =============================================================================
# CHART TOOL
# =============================================================================

class ChartTool:
    """
    Gerador de gráficos Plotly para o pipeline SRAG com inteligência visual.

    Parâmetros
    ----------
    spark       : SparkSession — necessária para os métodos _generate_*.
    audit_logger: AuditLogger — stub local quando None.
    config      : ChartConfig com parâmetros visuais globais.
    output_dirs : Dict mapeando tipo de gráfico para diretório de destino.
    catalog / schema : identificadores Unity Catalog.
    dbutils     : objeto dbutils do Databricks para escrita em Volumes.
    """

    def __init__(
        self,
        spark                                 = None,
        audit_logger: Optional[AuditLogger]   = None,
        config:       Optional[ChartConfig]   = None,
        output_dirs:  Optional[Dict[str, Path]] = None,
        catalog:      str = "dbx_srag_lab",
        schema:       str = "gold",
        dbutils                               = None,
    ):
        self.spark   = spark
        self.audit   = audit_logger if audit_logger else AuditLogger()
        self.config  = config or ChartConfig()
        self.catalog = catalog
        self.schema  = schema
        self.dbutils = dbutils

        self._output_dirs               = self._init_output_dirs(output_dirs)
        self._id_lock                   = threading.Lock()
        self._charts_created            = 0
        self._total_generation_time     = 0.0
        self._kaleido_available: bool   = _check_kaleido_available()

        self.audit.log_event(
            AuditEvent.TOOL_INITIALIZED,
            {
                "tool":        "ChartTool",
                "output_dirs": {k: str(v) for k, v in self._output_dirs.items()},
                "has_spark":   spark   is not None,
                "has_dbutils": dbutils is not None,
                "png_export":  "enabled" if self._kaleido_available else (
                    "disabled — kaleido nao instalado. "
                    "Execute: %pip install kaleido  (ou chart_tool.try_enable_png())"
                ),
            },
            EventStatus.INFO if self._kaleido_available else EventStatus.WARNING,
        )

    # =========================================================================
    # INICIALIZAÇÃO
    # =========================================================================

    def _init_output_dirs(self, output_dirs: Optional[Dict[str, Path]]) -> Dict[str, Path]:
        import tempfile
        if output_dirs is None:
            output_dirs = {"default": Path(f"{_DEFAULT_OUTPUT_BASE}/custom")}

        initialized: Dict[str, Path] = {}
        for key, path in output_dirs.items():
            try:
                if self.dbutils:
                    self.dbutils.fs.mkdirs(str(path))
                else:
                    path.mkdir(parents=True, exist_ok=True)
                initialized[key] = path
            except Exception as exc:
                tmp = Path(tempfile.mkdtemp(prefix=f"charts_{key}_"))
                self.audit.log_event(
                    AuditEvent.TOOL_DEGRADED,
                    {"reason": f"output_dir '{key}' inacessivel: {exc}", "fallback_dir": str(tmp)},
                    EventStatus.WARNING,
                )
                initialized[key] = tmp

        if "default" not in initialized:
            initialized["default"] = list(initialized.values())[0]
        return initialized

    def _resolve_output_dir(self, chart_type: str) -> Path:
        return self._output_dirs.get(chart_type, self._output_dirs["default"])

    # =========================================================================
    # HELPERS PRIVADOS DE INTELIGÊNCIA VISUAL
    # =========================================================================

    def _validate_columns(
        self,
        df: pd.DataFrame,
        required: List[str],
        context: str,
    ) -> List[str]:
        """
        Valida presença de colunas no DataFrame.

        Retorna lista de colunas ausentes. Vazia = tudo ok.
        Registra erro no audit quando há ausências.
        """
        missing = [c for c in required if c not in df.columns]
        if missing:
            self.audit.log_event(
                AuditEvent.CHART_ERROR,
                {
                    "context":  context,
                    "missing":  missing,
                    "available": list(df.columns),
                },
                EventStatus.ERROR,
            )
        return missing

    def _should_use_horizontal_bar(self, df: pd.DataFrame, x_col: str) -> bool:
        """
        Decide se barra horizontal é mais adequada que vertical.

        Critérios:
        - Mais de _H_BAR_THRESHOLD_CATEGORIES categorias, OU
        - Comprimento médio dos labels acima de _H_BAR_THRESHOLD_LABEL_LEN.
        """
        n = len(df)
        if n >= _H_BAR_THRESHOLD_CATEGORIES:
            return True
        avg_len = df[x_col].astype(str).str.len().mean() if n > 0 else 0
        return avg_len > _H_BAR_THRESHOLD_LABEL_LEN

    def _pct_axis_format(self) -> dict:
        """Layout parcial para eixo Y percentual."""
        return dict(ticksuffix="%", range=[0, None])

    def _pct_hover_suffix(self, col: str) -> str:
        return "%{y:.1f}%" if _is_percent_col(col) else "%{y:,.0f}"

    def _sort_dataframe(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        ascending: bool = True,
    ) -> pd.DataFrame:
        """
        Ordena o DataFrame de forma semanticamente adequada.

        - Colunas temporais: ordem cronológica.
        - Demais: por valor da métrica (descendente por padrão para ranking).
        """
        if _is_temporal_col(x_col):
            try:
                df = df.copy()
                df[x_col] = pd.to_datetime(df[x_col], errors="ignore")
                return df.sort_values(x_col, ascending=True).reset_index(drop=True)
            except Exception:
                return df.sort_values(x_col, ascending=True, key=lambda c: c.astype(str)).reset_index(drop=True)
        return df.sort_values(y_col, ascending=ascending).reset_index(drop=True)

    def _build_subtitle(self, df: pd.DataFrame, x_col: str) -> str:
        """
        Gera subtítulo com intervalo temporal quando detecta coluna de data.
        Retorna string vazia quando não aplicável.
        """
        if not _is_temporal_col(x_col):
            return ""
        try:
            vals = pd.to_datetime(df[x_col], errors="coerce").dropna()
            if vals.empty:
                return ""
            return f"Período: {vals.min().strftime('%b/%Y')} – {vals.max().strftime('%b/%Y')}"
        except Exception:
            return ""

    def _apply_standard_layout(
        self,
        fig: go.Figure,
        chart_type: str,
        subtitle: str = "",
        is_pct: bool = False,
        horizontal: bool = False,
    ) -> None:
        """
        Aplica configurações visuais padronizadas e limpas.

        Parâmetros
        ----------
        chart_type : tipo de gráfico (controla ajustes específicos de eixo).
        subtitle   : texto de subtítulo/fonte exibido abaixo do título.
        is_pct     : quando True, formata o eixo de valores como percentual.
        horizontal : quando True, aplica formatação de eixo para barras horizontais.
        """
        title_obj = fig.layout.title
        title_text = title_obj.text if title_obj and title_obj.text else ""

        if subtitle:
            title_text = f"{title_text}<br><sup style='color:#888;font-size:12px'>{subtitle}</sup>"

        fig.update_layout(
            title=dict(
                text=title_text,
                x=0.04,
                xanchor="left",
                font=dict(size=18, color="#2c2c2c"),
            ),
            height=self.config.default_height,
            width=self.config.default_width,
            margin=dict(l=70, r=40, t=80, b=80),
            template=self.config.default_theme.value,
            font=dict(family="Inter, Arial, sans-serif", size=13, color="#3a3a3a"),
            hoverlabel=dict(
                bgcolor="white",
                font_size=13,
                font_family="Inter, Arial, sans-serif",
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=12),
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        # Eixo de valores como percentual
        if is_pct:
            axis_fmt = dict(ticksuffix="%", tickformat=".1f", gridcolor="#ebebeb")
            if horizontal:
                fig.update_xaxes(**axis_fmt)
            else:
                fig.update_yaxes(**axis_fmt)
        else:
            if horizontal:
                fig.update_xaxes(gridcolor="#ebebeb", zeroline=False)
            else:
                fig.update_yaxes(gridcolor="#ebebeb", zeroline=False)

        # Configurações específicas por tipo
        if chart_type == "mensal":
            fig.update_xaxes(
                type="category",
                tickmode="linear",
                dtick=1,
                tickangle=-45,
                automargin=True,
                showgrid=False,
            )
            fig.update_layout(height=560, margin=dict(b=100))

        elif chart_type in ("bar", "geografico", "demografico"):
            if not horizontal:
                fig.update_xaxes(tickangle=-30, automargin=True, showgrid=False)
            else:
                fig.update_yaxes(automargin=True, showgrid=False)
                fig.update_xaxes(showgrid=True, gridcolor="#ebebeb")

        elif chart_type in ("line", "multi_line", "diario", "viral", "area"):
            fig.update_xaxes(automargin=True, showgrid=False)

        elif chart_type == "heatmap":
            fig.update_layout(height=600, margin=dict(l=100, b=100))

    # =========================================================================
    # MÉTODOS PÚBLICOS — TIPOS DE GRÁFICO
    # =========================================================================

    def create_line_chart(
        self,
        data:  Union[pd.DataFrame, List[Dict]],
        title: str,
        x_col: str,
        y_col: str,
        **kwargs,
    ) -> Optional[Dict]:
        """Gráfico de linha simples com marcadores e hover enriquecido."""
        try:
            start = time.time()
            df = self._ensure_dataframe(data)
            if df.empty:
                return None

            missing = self._validate_columns(df, [x_col, y_col], "create_line_chart")
            if missing:
                return None

            is_pct  = _is_percent_col(y_col)
            df      = self._sort_dataframe(df, x_col, y_col)
            sub     = self._build_subtitle(df, x_col)

            hover_fmt = "%{y:.1f}%" if is_pct else "%{y:,.0f}"
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode="lines+markers",
                name=humanize(y_col),
                line=dict(color=self.config.color_palette[0], width=2.5),
                marker=dict(size=6, color=self.config.color_palette[0]),
                hovertemplate=(
                    f"<b>{humanize(x_col)}</b>: %{{x}}<br>"
                    f"<b>{humanize(y_col)}</b>: {hover_fmt}<extra></extra>"
                ),
            ))
            fig.update_layout(
                title=title,
                xaxis_title=humanize(x_col),
                yaxis_title=humanize(y_col),
                hovermode="x unified",
            )
            self._apply_standard_layout(fig, "line", subtitle=sub, is_pct=is_pct)
            return self._write_and_record(fig, "line", title, len(df), start)

        except Exception as exc:
            self.audit.log_event(AuditEvent.CHART_ERROR, {"type": "line", "error": str(exc)}, EventStatus.ERROR)
            return None

    def create_bar_chart(
        self,
        data:       Union[pd.DataFrame, List[Dict]],
        title:      str,
        x_col:      str,
        y_col:      str,
        orientation: str = "auto",
        sort_by:    str  = "value",
        **kwargs,
    ) -> Optional[Dict]:
        """
        Gráfico de barras com troca automática para horizontal.

        orientation : "auto" | "v" | "h"
            "auto" decide com base em cardinalidade e comprimento de labels.
        sort_by : "value" | "category" | "none"
            Controla ordenação das barras.
        """
        try:
            start = time.time()
            df = self._ensure_dataframe(data)
            if df.empty:
                return None

            missing = self._validate_columns(df, [x_col, y_col], "create_bar_chart")
            if missing:
                return None

            is_pct   = _is_percent_col(y_col)
            use_h    = (
                orientation == "h"
                or (orientation == "auto" and self._should_use_horizontal_bar(df, x_col))
            )

            # Ordenação
            if sort_by == "value":
                ascending = use_h  # horizontal: ascend para o maior ir no topo
                df = df.sort_values(y_col, ascending=ascending).reset_index(drop=True)
            elif sort_by == "category":
                df = df.sort_values(x_col).reset_index(drop=True)

            sub       = self._build_subtitle(df, x_col)
            hover_fmt = "%{x:.1f}%" if (is_pct and use_h) else ("%{y:.1f}%" if is_pct else ("%{x:,.0f}" if use_h else "%{y:,.0f}"))
            color     = self.config.color_palette[1]

            fig = go.Figure()
            if use_h:
                fig.add_trace(go.Bar(
                    x=df[y_col],
                    y=df[x_col],
                    orientation="h",
                    name=humanize(y_col),
                    marker_color=color,
                    hovertemplate=(
                        f"<b>{humanize(x_col)}</b>: %{{y}}<br>"
                        f"<b>{humanize(y_col)}</b>: %{{x:.1f}}%<extra></extra>"
                        if is_pct else
                        f"<b>{humanize(x_col)}</b>: %{{y}}<br>"
                        f"<b>{humanize(y_col)}</b>: %{{x:,.0f}}<extra></extra>"
                    ),
                ))
                fig.update_layout(
                    title=title,
                    xaxis_title=humanize(y_col),
                    yaxis_title=humanize(x_col),
                )
            else:
                fig.add_trace(go.Bar(
                    x=df[x_col],
                    y=df[y_col],
                    name=humanize(y_col),
                    marker_color=color,
                    hovertemplate=(
                        f"<b>{humanize(x_col)}</b>: %{{x}}<br>"
                        f"<b>{humanize(y_col)}</b>: %{{y:.1f}}%<extra></extra>"
                        if is_pct else
                        f"<b>{humanize(x_col)}</b>: %{{x}}<br>"
                        f"<b>{humanize(y_col)}</b>: %{{y:,.0f}}<extra></extra>"
                    ),
                ))
                fig.update_layout(
                    title=title,
                    xaxis_title=humanize(x_col),
                    yaxis_title=humanize(y_col),
                )

            self._apply_standard_layout(
                fig, "bar", subtitle=sub, is_pct=is_pct, horizontal=use_h
            )
            return self._write_and_record(fig, "bar", title, len(df), start)

        except Exception as exc:
            self.audit.log_event(AuditEvent.CHART_ERROR, {"type": "bar", "error": str(exc)}, EventStatus.ERROR)
            return None

    def create_area_chart(
        self,
        data:  Union[pd.DataFrame, List[Dict]],
        title: str,
        x_col: str,
        y_col: str,
        **kwargs,
    ) -> Optional[Dict]:
        """Gráfico de área preenchida com hover enriquecido."""
        try:
            start = time.time()
            df = self._ensure_dataframe(data)
            if df.empty:
                return None

            missing = self._validate_columns(df, [x_col, y_col], "create_area_chart")
            if missing:
                return None

            is_pct = _is_percent_col(y_col)
            df     = self._sort_dataframe(df, x_col, y_col)
            sub    = self._build_subtitle(df, x_col)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df[y_col],
                fill="tozeroy",
                name=humanize(y_col),
                mode="lines",
                line=dict(color=self.config.color_palette[2], width=2),
                fillcolor=f"rgba(44, 160, 44, 0.15)",
                hovertemplate=(
                    f"<b>{humanize(x_col)}</b>: %{{x}}<br>"
                    f"<b>{humanize(y_col)}</b>: %{{y:.1f}}%<extra></extra>"
                    if is_pct else
                    f"<b>{humanize(x_col)}</b>: %{{x}}<br>"
                    f"<b>{humanize(y_col)}</b>: %{{y:,.0f}}<extra></extra>"
                ),
            ))
            fig.update_layout(
                title=title,
                xaxis_title=humanize(x_col),
                yaxis_title=humanize(y_col),
                hovermode="x unified",
            )
            self._apply_standard_layout(fig, "area", subtitle=sub, is_pct=is_pct)
            return self._write_and_record(fig, "area", title, len(df), start)

        except Exception as exc:
            self.audit.log_event(AuditEvent.CHART_ERROR, {"type": "area", "error": str(exc)}, EventStatus.ERROR)
            return None

    def create_multi_line_chart(
        self,
        data:   Union[pd.DataFrame, List[Dict]],
        title:  str,
        x_col:  str,
        y_cols: List[str],
        **kwargs,
    ) -> Optional[Dict]:
        """
        Gráfico de linhas com múltiplas séries sobrepostas.

        Colunas ausentes no DataFrame são silenciosamente ignoradas.
        """
        try:
            start = time.time()
            df = self._ensure_dataframe(data)
            if df.empty:
                return None

            missing_x = self._validate_columns(df, [x_col], "create_multi_line_chart")
            if missing_x:
                return None

            active_cols = [c for c in y_cols if c in df.columns]
            if not active_cols:
                self.audit.log_event(
                    AuditEvent.CHART_ERROR,
                    {"type": "multi_line", "reason": "nenhuma coluna y disponivel", "requested": y_cols},
                    EventStatus.ERROR,
                )
                return None

            df  = self._sort_dataframe(df, x_col, active_cols[0])
            sub = self._build_subtitle(df, x_col)
            is_pct = all(_is_percent_col(c) for c in active_cols)

            fig = go.Figure()
            for i, col in enumerate(active_cols):
                color = self.config.color_palette[i % len(self.config.color_palette)]
                hover = (
                    f"<b>{humanize(x_col)}</b>: %{{x}}<br>"
                    f"<b>{humanize(col)}</b>: %{{y:.1f}}%<extra></extra>"
                    if _is_percent_col(col) else
                    f"<b>{humanize(x_col)}</b>: %{{x}}<br>"
                    f"<b>{humanize(col)}</b>: %{{y:,.0f}}<extra></extra>"
                )
                fig.add_trace(go.Scatter(
                    x=df[x_col], y=df[col],
                    mode="lines+markers",
                    name=humanize(col),
                    line=dict(color=color, width=2),
                    marker=dict(size=5, color=color),
                    hovertemplate=hover,
                ))
            fig.update_layout(
                title=title,
                xaxis_title=humanize(x_col),
                yaxis_title="%" if is_pct else "Casos",
                hovermode="x unified",
            )
            self._apply_standard_layout(fig, "multi_line", subtitle=sub, is_pct=is_pct)
            return self._write_and_record(fig, "multi_line", title, len(df), start)

        except Exception as exc:
            self.audit.log_event(AuditEvent.CHART_ERROR, {"type": "multi_line", "error": str(exc)}, EventStatus.ERROR)
            return None

    def create_heatmap(
        self,
        data:  Union[pd.DataFrame, List[Dict]],
        title: str,
        x_col: str,
        y_col: str,
        z_col: str,
        **kwargs,
    ) -> Optional[Dict]:
        """Heatmap a partir de dados em formato longo (long format)."""
        try:
            start = time.time()
            df = self._ensure_dataframe(data)
            if df.empty:
                return None

            missing = self._validate_columns(df, [x_col, y_col, z_col], "create_heatmap")
            if missing:
                return None

            is_pct    = _is_percent_col(z_col)
            pivot_df  = df.pivot(index=y_col, columns=x_col, values=z_col)
            colorscale = "RdYlGn_r" if not is_pct else "Blues"

            fig = go.Figure(data=go.Heatmap(
                z=pivot_df.values,
                x=[humanize(str(c)) for c in pivot_df.columns],
                y=[humanize(str(r)) for r in pivot_df.index],
                colorscale=colorscale,
                hovertemplate=(
                    f"<b>{humanize(x_col)}</b>: %{{x}}<br>"
                    f"<b>{humanize(y_col)}</b>: %{{y}}<br>"
                    f"<b>{humanize(z_col)}</b>: %{{z:.1f}}%<extra></extra>"
                    if is_pct else
                    f"<b>{humanize(x_col)}</b>: %{{x}}<br>"
                    f"<b>{humanize(y_col)}</b>: %{{y}}<br>"
                    f"<b>{humanize(z_col)}</b>: %{{z:,.0f}}<extra></extra>"
                ),
                colorbar=dict(title=humanize(z_col)),
            ))
            fig.update_layout(
                title=title,
                xaxis_title=humanize(x_col),
                yaxis_title=humanize(y_col),
            )
            self._apply_standard_layout(fig, "heatmap")
            return self._write_and_record(fig, "heatmap", title, len(df), start)

        except Exception as exc:
            self.audit.log_event(AuditEvent.CHART_ERROR, {"type": "heatmap", "error": str(exc)}, EventStatus.ERROR)
            return None

    def create_pie_chart(
        self,
        data:       Union[pd.DataFrame, List[Dict]],
        title:      str,
        labels_col: str,
        values_col: str,
        **kwargs,
    ) -> Optional[Dict]:
        """
        Gráfico de pizza — restrito a até MAX_PIE_CATEGORIES categorias.

        Quando a cardinalidade excede o limite, faz fallback automático para
        barra horizontal com aviso no audit, preservando a informação sem
        gerar um gráfico enganoso.
        """
        try:
            start = time.time()
            df = self._ensure_dataframe(data)
            if df.empty:
                return None

            missing = self._validate_columns(df, [labels_col, values_col], "create_pie_chart")
            if missing:
                return None

            n_cats = df[labels_col].nunique()
            if n_cats > MAX_PIE_CATEGORIES:
                self.audit.log_event(
                    AuditEvent.CHART_ERROR,
                    {
                        "type":       "pie",
                        "reason":     f"alta cardinalidade ({n_cats} categorias > {MAX_PIE_CATEGORIES}) — fallback para barra horizontal",
                    },
                    EventStatus.WARNING,
                )
                return self.create_bar_chart(
                    data=df,
                    title=title,
                    x_col=labels_col,
                    y_col=values_col,
                    orientation="h",
                )

            fig = go.Figure(data=[go.Pie(
                labels=df[labels_col],
                values=df[values_col],
                marker=dict(
                    colors=self.config.color_palette[:n_cats],
                    line=dict(color="white", width=2),
                ),
                textposition="auto",
                textinfo="label+percent",
                hovertemplate=(
                    f"<b>%{{label}}</b><br>"
                    f"{humanize(values_col)}: %{{value:,.0f}}<br>"
                    f"Participação: %{{percent}}<extra></extra>"
                ),
                hole=0.0,
            )])
            fig.update_layout(title=title)
            self._apply_standard_layout(fig, "pie")
            return self._write_and_record(fig, "pie", title, len(df), start)

        except Exception as exc:
            self.audit.log_event(AuditEvent.CHART_ERROR, {"type": "pie", "error": str(exc)}, EventStatus.ERROR)
            return None

    # =========================================================================
    # GRÁFICOS ESPECIALIZADOS
    # =========================================================================

    def create_grouped_bar_chart(
        self,
        data:      Union[pd.DataFrame, List[Dict]],
        title:     str,
        x_col:     str,
        y_col:     str,
        group_col: str,
        **kwargs,
    ) -> Optional[Dict]:
        """
        Barras agrupadas por categoria.

        Útil para comparar a mesma métrica em subgrupos (ex.: casos por
        faixa etária, agrupados por sexo).

        Parâmetros
        ----------
        x_col     : eixo X (ex.: faixa_etaria).
        y_col     : métrica numérica (ex.: total_casos).
        group_col : coluna de agrupamento que gera as séries (ex.: sexo).
        """
        try:
            start = time.time()
            df = self._ensure_dataframe(data)
            if df.empty:
                return None

            missing = self._validate_columns(df, [x_col, y_col, group_col], "create_grouped_bar_chart")
            if missing:
                return None

            is_pct  = _is_percent_col(y_col)
            groups  = df[group_col].unique()
            use_h   = self._should_use_horizontal_bar(df, x_col)

            fig = go.Figure()
            for i, grp in enumerate(groups):
                grp_df = df[df[group_col] == grp]
                color  = self.config.color_palette[i % len(self.config.color_palette)]
                hover  = (
                    f"<b>{humanize(x_col)}</b>: %{{x}}<br>"
                    f"<b>{humanize(y_col)}</b>: %{{y:.1f}}%<extra></extra>"
                    if is_pct else
                    f"<b>{humanize(x_col)}</b>: %{{x}}<br>"
                    f"<b>{humanize(y_col)}</b>: %{{y:,.0f}}<extra></extra>"
                )
                if use_h:
                    fig.add_trace(go.Bar(
                        x=grp_df[y_col], y=grp_df[x_col],
                        orientation="h", name=str(grp),
                        marker_color=color, hovertemplate=hover,
                    ))
                else:
                    fig.add_trace(go.Bar(
                        x=grp_df[x_col], y=grp_df[y_col],
                        name=str(grp), marker_color=color,
                        hovertemplate=hover,
                    ))

            fig.update_layout(
                title=title,
                barmode="group",
                xaxis_title=humanize(y_col if use_h else x_col),
                yaxis_title=humanize(x_col if use_h else y_col),
            )
            self._apply_standard_layout(fig, "bar", is_pct=is_pct, horizontal=use_h)
            return self._write_and_record(fig, "bar", title, len(df), start)

        except Exception as exc:
            self.audit.log_event(AuditEvent.CHART_ERROR, {"type": "grouped_bar", "error": str(exc)}, EventStatus.ERROR)
            return None

    def create_top_n_chart(
        self,
        data:  Union[pd.DataFrame, List[Dict]],
        title: str,
        x_col: str,
        y_col: str,
        n:     int = 10,
        **kwargs,
    ) -> Optional[Dict]:
        """
        Ranking Top-N em barra horizontal, ordenado de maior para menor.

        Sempre usa orientação horizontal para facilitar leitura de labels.
        """
        try:
            start = time.time()
            df = self._ensure_dataframe(data)
            if df.empty:
                return None

            missing = self._validate_columns(df, [x_col, y_col], "create_top_n_chart")
            if missing:
                return None

            is_pct = _is_percent_col(y_col)
            df = (
                df.sort_values(y_col, ascending=False)
                  .head(n)
                  .sort_values(y_col, ascending=True)  # ascendente p/ o maior ficar no topo
                  .reset_index(drop=True)
            )

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df[y_col],
                y=df[x_col],
                orientation="h",
                name=humanize(y_col),
                marker=dict(
                    color=df[y_col],
                    colorscale="Blues",
                    showscale=False,
                ),
                hovertemplate=(
                    f"<b>%{{y}}</b><br>{humanize(y_col)}: %{{x:.1f}}%<extra></extra>"
                    if is_pct else
                    f"<b>%{{y}}</b><br>{humanize(y_col)}: %{{x:,.0f}}<extra></extra>"
                ),
                text=df[y_col].apply(
                    lambda v: f"{v:.1f}%" if is_pct else f"{v:,.0f}"
                ),
                textposition="outside",
            ))
            fig.update_layout(
                title=title,
                xaxis_title=humanize(y_col),
                yaxis_title=humanize(x_col),
                height=max(400, n * 42),
            )
            self._apply_standard_layout(fig, "bar", is_pct=is_pct, horizontal=True)
            return self._write_and_record(fig, "bar", title, len(df), start)

        except Exception as exc:
            self.audit.log_event(AuditEvent.CHART_ERROR, {"type": "top_n", "error": str(exc)}, EventStatus.ERROR)
            return None

    def create_year_comparison_chart(
        self,
        data:     Union[pd.DataFrame, List[Dict]],
        title:    str,
        x_col:    str,
        y_col:    str,
        year_col: str,
        **kwargs,
    ) -> Optional[Dict]:
        """
        Compara séries de anos distintos em um único gráfico de linhas.

        Cada ano recebe uma cor distinta da paleta. Ideal para comparação
        de sazonalidade entre anos (ex.: meses × ano × casos).

        Parâmetros
        ----------
        x_col    : dimensão do eixo X dentro de cada ano (ex.: mes).
        y_col    : métrica (ex.: total_casos).
        year_col : coluna que identifica o ano de cada registro.
        """
        try:
            start = time.time()
            df = self._ensure_dataframe(data)
            if df.empty:
                return None

            missing = self._validate_columns(df, [x_col, y_col, year_col], "create_year_comparison_chart")
            if missing:
                return None

            is_pct = _is_percent_col(y_col)
            years  = sorted(df[year_col].unique())

            fig = go.Figure()
            for i, yr in enumerate(years):
                yr_df = df[df[year_col] == yr].sort_values(x_col)
                color = self.config.color_palette[i % len(self.config.color_palette)]
                fig.add_trace(go.Scatter(
                    x=yr_df[x_col],
                    y=yr_df[y_col],
                    mode="lines+markers",
                    name=str(yr),
                    line=dict(color=color, width=2),
                    marker=dict(size=6, color=color),
                    hovertemplate=(
                        f"<b>{humanize(x_col)}</b>: %{{x}}<br>"
                        f"<b>{yr}</b>: %{{y:.1f}}%<extra></extra>"
                        if is_pct else
                        f"<b>{humanize(x_col)}</b>: %{{x}}<br>"
                        f"<b>{yr}</b>: %{{y:,.0f}}<extra></extra>"
                    ),
                ))
            fig.update_layout(
                title=title,
                xaxis_title=humanize(x_col),
                yaxis_title=humanize(y_col),
                hovermode="x unified",
            )
            self._apply_standard_layout(fig, "multi_line", is_pct=is_pct)
            return self._write_and_record(fig, "multi_line", title, len(df), start)

        except Exception as exc:
            self.audit.log_event(AuditEvent.CHART_ERROR, {"type": "year_comparison", "error": str(exc)}, EventStatus.ERROR)
            return None

    def create_rate_comparison_chart(
        self,
        data:     Union[pd.DataFrame, List[Dict]],
        title:    str,
        x_col:    str,
        rate_cols: List[str],
        **kwargs,
    ) -> Optional[Dict]:
        """
        Compara múltiplas taxas percentuais em um único gráfico de barras agrupadas.

        Todos os y_cols devem ser métricas percentuais. O eixo Y é
        automaticamente formatado com sufixo "%".

        Parâmetros
        ----------
        x_col      : dimensão de agrupamento (ex.: sg_uf, faixa_etaria).
        rate_cols  : lista de colunas de taxa (ex.: ["taxa_mortalidade", "taxa_uti"]).
        """
        try:
            start = time.time()
            df = self._ensure_dataframe(data)
            if df.empty:
                return None

            required = [x_col] + rate_cols
            missing  = self._validate_columns(df, required, "create_rate_comparison_chart")
            if missing:
                return None

            use_h = self._should_use_horizontal_bar(df, x_col)
            fig   = go.Figure()

            for i, col in enumerate(rate_cols):
                color = self.config.color_palette[i % len(self.config.color_palette)]
                hover = (
                    f"<b>%{{y}}</b><br>{humanize(col)}: %{{x:.1f}}%<extra></extra>"
                    if use_h else
                    f"<b>%{{x}}</b><br>{humanize(col)}: %{{y:.1f}}%<extra></extra>"
                )
                if use_h:
                    fig.add_trace(go.Bar(
                        x=df[col], y=df[x_col],
                        orientation="h", name=humanize(col),
                        marker_color=color, hovertemplate=hover,
                    ))
                else:
                    fig.add_trace(go.Bar(
                        x=df[x_col], y=df[col],
                        name=humanize(col),
                        marker_color=color, hovertemplate=hover,
                    ))

            fig.update_layout(
                title=title,
                barmode="group",
                xaxis_title=humanize(x_col if not use_h else rate_cols[0]),
                yaxis_title=humanize(rate_cols[0] if not use_h else x_col),
            )
            self._apply_standard_layout(fig, "bar", is_pct=True, horizontal=use_h)
            return self._write_and_record(fig, "bar", title, len(df), start)

        except Exception as exc:
            self.audit.log_event(AuditEvent.CHART_ERROR, {"type": "rate_comparison", "error": str(exc)}, EventStatus.ERROR)
            return None

    # =========================================================================
    # GRÁFICO AD-HOC INTELIGENTE
    # =========================================================================

    def generate_custom_chart(
        self,
        data:       Union[pd.DataFrame, List[Dict]],
        chart_type: str,
        title:      str,
        x_col:      str,
        y_col:      str,
        y_cols:     Optional[List[str]] = None,
        z_col:      Optional[str]       = None,
    ) -> Optional[Dict]:
        """
        Ponto de entrada unificado para gráficos ad-hoc com inteligência visual.

        Diferentemente de um dispatcher simples, este método:
        1. Valida as colunas solicitadas contra o DataFrame recebido.
        2. Infere a natureza do dado (temporal, percentual, alta cardinalidade).
        3. Corrige escolhas inadequadas de gráfico (ex.: pie com muitas
           categorias, bar vertical para labels longos).
        4. Aplica defaults inteligentes (orientação, formatação, ordenação).
        5. Delega para o método especializado mais adequado.

        Contrato de mapeamento por tipo
        --------------------------------
        "pie"       : x_col → labels_col, y_col → values_col.
                      Cardinalidade alta faz fallback automático para barra.
        "heatmap"   : x_col e y_col = eixos da matriz; z_col = intensidade (obrigatório).
        "multi_line": y_cols = séries. Fallback para [y_col] quando y_cols é None.
        "bar"       : orientação decidida automaticamente.
        "top_n"     : alias para create_top_n_chart (top 10 por padrão).

        Tipos não reconhecidos fazem fallback para "bar" com aviso no audit.
        """
        try:
            df = self._ensure_dataframe(data)
        except ValueError as exc:
            self.audit.log_event(
                AuditEvent.CHART_ERROR,
                {"type": chart_type, "reason": str(exc)},
                EventStatus.ERROR,
            )
            return None

        if df.empty:
            self.audit.log_event(
                AuditEvent.CHART_ERROR,
                {"type": chart_type, "reason": "DataFrame vazio recebido"},
                EventStatus.WARNING,
            )
            return None

        # --- Heatmap (validação especial de z_col) ---
        if chart_type == "heatmap":
            if z_col is None:
                self.audit.log_event(
                    AuditEvent.CHART_ERROR,
                    {"type": "heatmap", "reason": "z_col obrigatorio para heatmap"},
                    EventStatus.WARNING,
                )
                return None
            return self.create_heatmap(df, title, x_col, y_col, z_col)

        # --- Validação de colunas base ---
        missing = self._validate_columns(df, [x_col, y_col], "generate_custom_chart")
        if missing:
            return None

        # --- Inferências ---
        n_cats       = df[x_col].nunique()
        is_temporal  = _is_temporal_col(x_col)
        is_pct       = _is_percent_col(y_col)
        many_cats    = n_cats > _H_BAR_THRESHOLD_CATEGORIES

        # --- Correções de tipo de gráfico ---

        # Pie com alta cardinalidade → barra horizontal
        if chart_type == "pie" and n_cats > MAX_PIE_CATEGORIES:
            self.audit.log_event(
                AuditEvent.CHART_ERROR,
                {
                    "type":      "pie",
                    "reason":    f"alta cardinalidade ({n_cats}) — corrigido para bar horizontal",
                    "correction": "bar_h",
                },
                EventStatus.WARNING,
            )
            chart_type = "bar"

        # Bar vertical com dados temporais → line (mais semântico)
        if chart_type == "bar" and is_temporal and n_cats > 4:
            self.audit.log_event(
                AuditEvent.CHART_ERROR,
                {
                    "type":      "bar",
                    "reason":    "série temporal — corrigido para line",
                    "correction": "line",
                },
                EventStatus.WARNING,
            )
            chart_type = "line"

        # --- Despacho ---
        if chart_type == "pie":
            return self.create_pie_chart(df, title, x_col, y_col)

        if chart_type in ("line", "area"):
            method = self.create_line_chart if chart_type == "line" else self.create_area_chart
            return method(df, title, x_col, y_col)

        if chart_type == "multi_line":
            cols = y_cols if y_cols else [y_col]
            return self.create_multi_line_chart(df, title, x_col, cols)

        if chart_type == "top_n":
            return self.create_top_n_chart(df, title, x_col, y_col)

        if chart_type == "year_comparison":
            # year_col deve vir em y_cols[0] por convenção, ou é inferido
            # como a primeira coluna do DataFrame que contenha "ano" no nome.
            year_col = (y_cols[0] if y_cols else None) or next(
                (c for c in df.columns if "ano" in c.lower() and c != x_col), None
            )
            if year_col is None:
                self.audit.log_event(
                    AuditEvent.CHART_ERROR,
                    {
                        "type":   "year_comparison",
                        "reason": "year_col nao identificado — passe y_cols=['nome_da_coluna_ano']",
                    },
                    EventStatus.WARNING,
                )
                return self.create_line_chart(df, title, x_col, y_col)
            return self.create_year_comparison_chart(df, title, x_col, y_col, year_col)

        if chart_type == "bar":
            # Taxa com múltiplas colunas → rate_comparison
            if is_pct and y_cols:
                return self.create_rate_comparison_chart(df, title, x_col, [y_col] + y_cols)
            return self.create_bar_chart(df, title, x_col, y_col, orientation="auto")

        # Tipo não reconhecido → fallback bar com aviso
        self.audit.log_event(
            AuditEvent.CHART_ERROR,
            {
                "type":       chart_type,
                "reason":     "chart_type nao reconhecido — fallback para bar",
                "received":   chart_type,
            },
            EventStatus.WARNING,
        )
        return self.create_bar_chart(df, title, x_col, y_col, orientation="auto")

    # =========================================================================
    # GRÁFICOS PADRÃO DO PIPELINE
    # =========================================================================

    def generate_all_charts(self) -> List[Dict]:
        """
        Gera o conjunto fixo de gráficos padrão do pipeline SRAG.

        Opt-in: deve ser chamado somente quando a intenção classificada da
        query inclui visualização explícita.
        """
        generators = [
            (self._generate_time_series_chart,     "diario"),
            (self._generate_monthly_chart,          "mensal"),
            (self._generate_geographic_chart,       "geografico"),
            (self._generate_age_distribution_chart, "demografico"),
            (self._generate_viral_breakdown_chart,  "viral"),
        ]

        results:  List[Dict] = []
        failures: List[str]  = []

        for gen_fn, chart_type in generators:
            try:
                result = gen_fn()
                if result and result.get("path"):
                    result["pipeline_type"] = chart_type
                    results.append(result)
                else:
                    failures.append(chart_type)
            except Exception as exc:
                failures.append(chart_type)
                self.audit.log_event(
                    AuditEvent.CHART_ERROR,
                    {"type": chart_type, "error": str(exc)},
                    EventStatus.ERROR,
                )

        self.audit.log_event(
            AuditEvent.CHART_GENERATED,
            {
                "total_generated": len(results),
                "total_requested": len(generators),
                "failures":        failures,
            },
            EventStatus.SUCCESS if not failures else EventStatus.WARNING,
        )
        return results

    def _generate_time_series_chart(self) -> Optional[Dict]:
        """Série diária de casos — últimos 30 dias."""
        if not self.spark:
            return None
        try:
            df = self.spark.sql(f"""
                SELECT dt_sintomas AS data_referencia, total_casos AS casos_dia
                FROM {self.catalog}.{self.schema}.gold_serie_diaria_30d
                WHERE total_casos IS NOT NULL
                ORDER BY dt_sintomas DESC
                LIMIT 30
            """).toPandas()
            if df.empty:
                return None
            df = df.sort_values("data_referencia").reset_index(drop=True)
            return self.create_area_chart(
                data=df,
                title="Evolução de Casos Diários — SRAG (Últimos 30 dias)",
                x_col="data_referencia",
                y_col="casos_dia",
            )
        except Exception as exc:
            self.audit.log_event(AuditEvent.CHART_ERROR, {"type": "diario", "error": str(exc)}, EventStatus.ERROR)
            return None

    def _generate_monthly_chart(self) -> Optional[Dict]:
        """Evolução mensal de casos — últimos 12 meses."""
        if not self.spark:
            return None
        try:
            start = time.time()
            df = self.spark.sql(f"""
                WITH max_mes AS (
                    SELECT MAX(ano_mes) AS max_ano_mes
                    FROM {self.catalog}.{self.schema}.gold_metricas_temporais
                    WHERE ano_mes IS NOT NULL
                ),
                corte AS (
                    SELECT DATE_FORMAT(
                        ADD_MONTHS(TO_DATE(CONCAT(max_ano_mes, '-01'), 'yyyy-MM-dd'), -12),
                        'yyyy-MM'
                    ) AS mes_corte
                    FROM max_mes
                )
                SELECT t.ano_mes, SUM(t.total_casos) AS total_casos
                FROM {self.catalog}.{self.schema}.gold_metricas_temporais t
                CROSS JOIN corte c
                WHERE t.ano_mes IS NOT NULL
                  AND t.total_casos IS NOT NULL
                  AND t.ano_mes >= c.mes_corte
                GROUP BY t.ano_mes
                ORDER BY t.ano_mes ASC
                LIMIT 12
            """).toPandas()

            if df.empty:
                self.audit.log_event(
                    AuditEvent.CHART_ERROR,
                    {"type": "mensal", "reason": "query retornou 0 linhas"},
                    EventStatus.WARNING,
                )
                return None

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df["ano_mes"],
                y=df["total_casos"],
                name="Casos Mensais",
                marker=dict(
                    color=df["total_casos"],
                    colorscale="Blues",
                    showscale=False,
                ),
                text=df["total_casos"].apply(lambda v: f"{v:,.0f}"),
                textposition="outside",
                hovertemplate=(
                    "<b>Período</b>: %{x}<br>"
                    "<b>Total de casos</b>: %{y:,.0f}<extra></extra>"
                ),
            ))
            fig.update_layout(
                title="Evolução Mensal de Casos SRAG — Últimos 12 Meses",
                xaxis_title="Mês",
                yaxis_title="Total de casos",
            )
            self._apply_standard_layout(fig, "mensal")
            return self._write_and_record(fig, "mensal", "Evolução Mensal SRAG", len(df), start)

        except Exception as exc:
            self.audit.log_event(AuditEvent.CHART_ERROR, {"type": "mensal", "error": str(exc)}, EventStatus.ERROR)
            return None

    def _generate_geographic_chart(self) -> Optional[Dict]:
        """Top 10 UFs por total de casos."""
        if not self.spark:
            return None
        try:
            df = self.spark.sql(f"""
                SELECT sg_uf, SUM(total_casos) AS total_casos
                FROM {self.catalog}.{self.schema}.gold_metricas_geograficas
                WHERE total_casos IS NOT NULL AND sg_uf IS NOT NULL
                GROUP BY sg_uf
                ORDER BY total_casos DESC
                LIMIT 10
            """).toPandas()
            if df.empty:
                return None
            return self.create_top_n_chart(
                data=df,
                title="Top 10 Estados por Casos SRAG",
                x_col="sg_uf",
                y_col="total_casos",
                n=10,
            )
        except Exception as exc:
            self.audit.log_event(AuditEvent.CHART_ERROR, {"type": "geografico", "error": str(exc)}, EventStatus.ERROR)
            return None

    def _generate_age_distribution_chart(self) -> Optional[Dict]:
        """Distribuição de casos por faixa etária."""
        if not self.spark:
            return None
        try:
            df = self.spark.sql(f"""
                SELECT faixa_etaria AS faixa_etaria_label,
                       SUM(total_casos) AS total_casos
                FROM {self.catalog}.{self.schema}.gold_metricas_demograficas
                WHERE faixa_etaria IS NOT NULL AND total_casos IS NOT NULL
                GROUP BY faixa_etaria, ordem_faixa
                ORDER BY ordem_faixa ASC NULLS LAST
            """).toPandas()
            if df.empty:
                return None
            return self.create_bar_chart(
                data=df,
                title="Distribuição de Casos por Faixa Etária",
                x_col="faixa_etaria_label",
                y_col="total_casos",
                orientation="h",
                sort_by="none",
            )
        except Exception as exc:
            self.audit.log_event(AuditEvent.CHART_ERROR, {"type": "demografico", "error": str(exc)}, EventStatus.ERROR)
            return None

    def _generate_viral_breakdown_chart(self) -> Optional[Dict]:
        """Breakdown viral diário — últimos 30 dias."""
        if not self.spark:
            return None
        try:
            df = self.spark.sql(f"""
                SELECT
                    dt_sintomas             AS data_referencia,
                    total_covid             AS COVID_19,
                    total_influenza         AS Influenza,
                    total_outro_virus       AS Outro_Virus,
                    total_sem_classificacao AS Sem_Classificacao
                FROM {self.catalog}.{self.schema}.gold_serie_diaria_30d
                WHERE dt_sintomas IS NOT NULL
                ORDER BY dt_sintomas DESC
                LIMIT 30
            """).toPandas()
            if df.empty:
                return None
            return self.create_multi_line_chart(
                data=df.sort_values("data_referencia"),
                title="Breakdown Viral Diário — SRAG (Últimos 30 dias)",
                x_col="data_referencia",
                y_cols=["COVID_19", "Influenza", "Outro_Virus", "Sem_Classificacao"],
            )
        except Exception as exc:
            self.audit.log_event(AuditEvent.CHART_ERROR, {"type": "viral", "error": str(exc)}, EventStatus.ERROR)
            return None

    def _generate_gender_chart(self) -> Optional[Dict]:
        """Distribuição por sexo — disponível para uso pontual."""
        if not self.spark:
            return None
        try:
            df = self.spark.sql(f"""
                SELECT sexo_label, SUM(total_casos) AS total_casos
                FROM {self.catalog}.{self.schema}.gold_metricas_demograficas
                WHERE sexo_label IS NOT NULL AND total_casos IS NOT NULL
                GROUP BY sexo_label
            """).toPandas()
            if df.empty:
                return None
            return self.create_pie_chart(
                data=df,
                title="Distribuição de Casos por Sexo",
                labels_col="sexo_label",
                values_col="total_casos",
            )
        except Exception as exc:
            self.audit.log_event(AuditEvent.CHART_ERROR, {"type": "sexo", "error": str(exc)}, EventStatus.ERROR)
            return None

    # =========================================================================
    # PERSISTÊNCIA E REGISTRO
    # =========================================================================

    def _write_chart_html(self, fig: go.Figure, output_path: Path) -> None:
        html_content = fig.to_html(full_html=True, include_plotlyjs="cdn")
        if self.dbutils:
            try:
                self.dbutils.fs.put(str(output_path), html_content, overwrite=True)
                return
            except Exception as exc:
                self.audit.log_event(
                    AuditEvent.CHART_WRITE_ERROR,
                    {"path": str(output_path), "method": "dbutils.fs.put", "error": str(exc), "fallback": "write nativo"},
                    EventStatus.WARNING,
                )
        with open(str(output_path), "w", encoding="utf-8") as f:
            f.write(html_content)

    def _write_chart_png(self, fig: go.Figure, html_path: Path) -> Optional[Path]:
        if not self._kaleido_available:
            return None
        png_path = html_path.with_suffix(".png")
        try:
            pio.write_image(
                fig, str(png_path), format="png",
                width=self.config.default_width,
                height=self.config.default_height,
            )
            return png_path
        except Exception as exc:
            self.audit.log_event(
                AuditEvent.CHART_WRITE_ERROR,
                {
                    "path":  str(png_path),
                    "method": "pio.write_image",
                    "error": str(exc),
                    "hint":  "verifique permissao de escrita e versao do kaleido",
                },
                EventStatus.WARNING,
            )
            return None

    def _write_and_record(
        self,
        fig:         go.Figure,
        chart_type:  str,
        title:       str,
        data_points: int,
        start_time:  float,
    ) -> Dict:
        chart_id    = self._generate_chart_id(chart_type)
        output_dir  = self._resolve_output_dir(chart_type)
        output_path = output_dir / f"{chart_id}.html"

        try:
            self._write_chart_html(fig, output_path)
        except Exception as exc:
            self.audit.log_event(
                AuditEvent.CHART_WRITE_ERROR,
                {"chart_id": chart_id, "path": str(output_path), "error": str(exc)},
                EventStatus.ERROR,
            )
            raise

        png_path = self._write_chart_png(fig, output_path)
        elapsed  = time.time() - start_time

        with self._id_lock:
            self._charts_created        += 1
            self._total_generation_time += elapsed

        try:
            file_size = output_path.stat().st_size
        except Exception as exc:
            file_size = len(fig.to_html(include_plotlyjs=False))
            self.audit.log_event(
                AuditEvent.CHART_STAT_ERROR,
                {"chart_id": chart_id, "path": str(output_path), "error": str(exc), "file_size_proxy": file_size},
                EventStatus.WARNING,
            )

        self.audit.log_event(
            AuditEvent.CHART_GENERATED,
            {
                "chart_id":        chart_id,
                "type":            chart_type,
                "data_points":     data_points,
                "generation_time": round(elapsed, 4),
                "file_size_bytes": file_size,
                "path_html":       str(output_path),
                "path_png":        str(png_path) if png_path else None,
            },
            EventStatus.SUCCESS,
        )

        return {
            "path":            str(output_path),
            "path_png":        str(png_path) if png_path else None,
            "chart_id":        chart_id,
            "chart_type":      chart_type,
            "generation_time": elapsed,
            "metadata": ChartMetadata(
                chart_id=chart_id,
                chart_type=(
                    ChartType(chart_type)
                    if chart_type in ChartType._value2member_map_
                    else ChartType.BAR
                ),
                title=title,
                created_at=datetime.now(),
                data_points=data_points,
                export_path=str(output_path),
                file_size=file_size,
                generation_time_seconds=elapsed,
                export_path_png=str(png_path) if png_path else None,
            ),
        }

    # =========================================================================
    # UTILITÁRIOS
    # =========================================================================

    def _ensure_dataframe(self, data: Union[pd.DataFrame, List[Dict]]) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, list):
            return pd.DataFrame(data)
        raise ValueError(f"Tipo nao suportado: {type(data)}")

    def _generate_chart_id(self, chart_type: str) -> str:
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid = uuid.uuid4().hex[:8]
        return f"srag_{chart_type}_{ts}_{uid}"

    def try_enable_png(self) -> bool:
        """Instala kaleido via pip e reabilita export PNG sem recriar o objeto."""
        if self._kaleido_available:
            return True
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "kaleido", "--quiet"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            self.audit.log_event(
                AuditEvent.TOOL_DEGRADED,
                {"reason": "falha ao instalar kaleido", "error": str(exc)},
                EventStatus.WARNING,
            )
            return False
        self._kaleido_available = _check_kaleido_available()
        self.audit.log_event(
            AuditEvent.TOOL_INITIALIZED,
            {"event": "kaleido_install_attempt", "png_export": "enabled" if self._kaleido_available else "failed"},
            EventStatus.SUCCESS if self._kaleido_available else EventStatus.WARNING,
        )
        return self._kaleido_available

    def cleanup_old_charts(self, max_files: int = 100) -> int:
        total_removed = 0
        for dir_key, output_dir in self._output_dirs.items():
            try:
                all_files = sorted(
                    [p for p in output_dir.iterdir() if p.suffix in (".html", ".png")],
                    key=lambda p: p.stat().st_mtime,
                )
                to_remove = all_files[: max(0, len(all_files) - max_files)]
                removed = 0
                for f in to_remove:
                    try:
                        f.unlink()
                        removed += 1
                    except Exception as exc:
                        self.audit.log_event(
                            AuditEvent.CHART_CLEANUP,
                            {"dir": dir_key, "file": f.name, "reason": "falha ao remover", "error": str(exc)},
                            EventStatus.WARNING,
                        )
                total_removed += removed
                if removed:
                    self.audit.log_event(
                        AuditEvent.CHART_CLEANUP,
                        {"dir": dir_key, "removed": removed, "remaining": len(all_files) - removed, "max_files": max_files},
                        EventStatus.INFO,
                    )
            except Exception as exc:
                self.audit.log_event(AuditEvent.CHART_CLEANUP, {"dir": dir_key, "error": str(exc)}, EventStatus.WARNING)
        return total_removed

    def get_stats(self) -> Dict:
        with self._id_lock:
            charts  = self._charts_created
            total_t = self._total_generation_time
        avg = total_t / charts if charts > 0 else 0.0
        return {
            "charts_created":        charts,
            "total_generation_time": round(total_t, 4),
            "avg_generation_time":   round(avg, 4),
            "output_dirs":           {k: str(v) for k, v in self._output_dirs.items()},
            "kaleido_available":     self._kaleido_available,
            "png_export":            "enabled" if self._kaleido_available else "disabled",
        }

    def __repr__(self) -> str:
        return (
            f"ChartTool("
            f"charts_created={self._charts_created}, "
            f"output_dirs={list(self._output_dirs.keys())}, "
            f"has_dbutils={self.dbutils is not None})"
        )


# =============================================================================
# ALIAS LEGADO
# =============================================================================

class ChartGenerator(ChartTool):
    """
    Alias mantido para compatibilidade. Será removido em versão futura.
    Migre imports para ChartTool diretamente.
    """
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "ChartGenerator foi renomeado para ChartTool. "
            "Atualize os imports para evitar quebra quando o alias for removido.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)