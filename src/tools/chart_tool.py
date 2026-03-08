"""
Chart Tool — Geração de Visualizações SRAG
==========================================

Responsabilidade: gerar gráficos Plotly interativos a partir de dados das
tabelas Gold do Unity Catalog, exportando cada gráfico como HTML leve e
como PNG estático via Kaleido.

Dois modos de uso:
    generate_all_charts()    : conjunto fixo de 5 gráficos padrão, de uso
                               explícito e opt-in — deve ser chamado somente
                               quando a intenção do usuário inclui visualização.
    generate_custom_chart()  : gráfico único ad-hoc, invocado pelo nó
                               execute_chart com dados já pré-processados
                               via SQL parametrizada.

Os métodos públicos create_*_chart() existem para permitir geração pontual
fora do pipeline principal — testes unitários e notebooks exploratórios.

Decisões de design
------------------
Export PNG via plotly.io.write_image() gerado diretamente
    O notebook 07 tentava converter o HTML salvo pelo Plotly em PNG usando
    kaleido. O kaleido não consegue parsear HTML — ele espera um objeto Figure
    serializado, não a saída de write_html(). A conversão sempre falhava em
    tempo de validação. A solução é gerar o PNG diretamente via
    pio.write_image() logo após a escrita do HTML, na mesma chamada de
    _write_and_record(). O PNG e o HTML ficam no mesmo diretório de saída
    com o mesmo chart_id como base de nome. O export PNG é não-bloqueante:
    se kaleido não estiver instalado ou write_image() falhar por qualquer
    razão, um aviso é registrado no AuditLogger e a execução continua — o
    HTML é sempre o artefato primário.

Guard de kaleido em __init__ com self._kaleido_available
    A versão anterior verificava kaleido dentro de _write_chart_png, chamado
    uma vez por gráfico. Com kaleido não instalado, cada um dos 5 gráficos
    gerava uma exceção → 5 chart_write_error no AuditLogger por execução,
    sem nenhum diagnóstico útil além de "verifique se kaleido está instalado".
    A nova versão executa _check_kaleido_available() uma vez no __init__,
    cacheia em self._kaleido_available e loga o status no evento TOOL_INITIALIZED.
    _write_chart_png verifica o flag antes de qualquer tentativa — retorna None
    imediatamente quando False, sem gerar exceções repetidas. Para ativar PNG
    em runtime sem recriar o objeto, use chart_tool.try_enable_png().

generate_all_charts() como método opt-in
    A versão anterior era descrita como "invocado pelo nó execute_sql do
    orquestrador", o que levava o nó a chamá-lo após toda execução de SQL,
    independentemente da intenção da query. Perguntas simples como "total de
    casos por ano" disparavam cinco queries Spark adicionais e escrita de
    cinco arquivos sem necessidade. generate_all_charts() é um método de uso
    explícito: o orquestrador deve chamá-lo somente quando a classificação de
    intenção indicar que o usuário solicitou visualização. Esta docstring e
    o bloco "Dois modos de uso" acima foram corrigidos para refletir esse
    contrato.

CTE para cálculo de janela temporal em _generate_monthly_chart
    Window functions (MAX(...) OVER()) são proibidas na cláusula WHERE em
    Spark SQL e lançam AnalysisException em toda execução. A query anterior
    usava essa construção, fazendo o gráfico mensal nunca ser gerado — o erro
    era capturado silenciosamente pelo except genérico. A solução adota CTE
    que isola o cálculo do valor de corte em uma subquery separada, retornando
    um escalar que pode ser usado no WHERE sem restrição.

include_plotlyjs="cdn" em vez de bundle embutido
    fig.write_html() sem parâmetros serializa a biblioteca Plotly completa
    (~3.7 MB) dentro de cada arquivo HTML. Com 120 arquivos acumulados, o
    diretório cresce para ~400 MB sem nenhum ganho de funcionalidade — a
    biblioteca não muda entre execuções. O parâmetro include_plotlyjs="cdn"
    reduz cada arquivo para ~8 KB, carregando Plotly via CDN no momento da
    visualização. A troca é que o HTML não funciona offline; dado que os
    relatórios SRAG são consultados em ambientes com acesso à internet, esse
    custo é aceitável.

output_dirs dict no construtor em vez de output_dir único
    O design anterior usava um único self.output_dir para todos os tipos de
    gráfico. _write_and_record() não roteava por tipo, então charts/daily/ e
    charts/monthly/ ficavam vazios enquanto tudo ia para charts/custom/. O
    construtor agora aceita output_dirs: Dict[str, Path] mapeando tipo para
    diretório. Quando o tipo não está no dict, usa a chave "default". O valor
    padrão preserva o comportamento anterior para quem não passa output_dirs.

_apply_standard_layout centraliza configuração visual
    Cada _generate_* definia altura, margens e ticks de forma ad-hoc e
    inconsistente. Isso produzia gráficos com estilos divergentes no mesmo
    relatório. O método _apply_standard_layout() aplica as configurações de
    forma uniforme por tipo de gráfico, permitindo ajuste global em um único
    lugar. O gráfico mensal recebe tratamento especial de eixo X (tickmode
    linear, dtick=1) porque Plotly omite labels automaticamente quando detecta
    risco de sobreposição — em séries de 12 meses, apenas 4-6 labels ficam
    visíveis com o comportamento padrão, o que em contexto epidemiológico pode
    ser interpretado como dado faltante.

Thread-safety em _generate_chart_id via threading.Lock
    self._charts_created era lido e incrementado em operações não atômicas
    separadas. Em execuções paralelas, dois gráficos podiam receber o mesmo
    contador antes de qualquer incremento, gerando IDs idênticos e
    sobrescrevendo arquivos sem aviso. O Lock protege a leitura e o incremento
    como operação atômica sem overhead significativo dado o volume esperado.

cleanup_old_charts com limite configurável
    Sem rotação, arquivos HTML acumulam indefinidamente. cleanup_old_charts()
    lista o diretório de saída, ordena por data de modificação e remove os
    arquivos mais antigos quando o total excede max_files. A operação é
    explícita e não ocorre automaticamente para não introduzir latência
    inesperada durante geração.

file_size registrado no AuditLogger
    ChartMetadata continha file_size mas generate_all_charts() descartava o
    dict completo, retornando apenas o path. O file_size é o único ponto do
    pipeline onde um alerta de acúmulo de disco poderia ser disparado. Ele
    agora é registrado no AuditLogger dentro de _write_and_record() para que
    o dado não seja silenciosamente descartado mesmo quando o chamador não
    consome os metadados completos.

Contrato implícito do pie chart documentado
    generate_custom_chart() mapeava x_col -> labels_col e y_col -> values_col
    ao despachar para create_pie_chart(). Esse mapeamento não estava
    documentado, forçando o chamador a adivinhar a convenção. A docstring
    agora declara explicitamente o contrato para chart_type="pie".

ChartGenerator como alias com DeprecationWarning
    O alias silencioso não avisava consumidores sobre a migração para ChartTool,
    tornando breaking changes invisíveis. O padrão adotado em GoldSQLToolLegacy
    é replicado aqui: subclasse com warnings.warn() no __init__.

Separação entre falha de escrita e falha de stat em _write_and_record
    A versão anterior capturava genericamente qualquer exceção após write_html().
    Quando stat() falhava com FileNotFoundError (path de Volume inacessível),
    o erro era absorvido sem identificar a causa real. Agora _write_chart_html
    e stat() têm blocos try/except independentes com contexto de diagnóstico
    distinto no AuditLogger.
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
from typing import Dict, List, Optional, Union
import time

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


def _check_kaleido_available() -> bool:
    """
    Verifica se kaleido está instalado e operacional para export PNG.

    Usa importlib.util.find_spec() para evitar import completo — mais rápido
    e não polui o namespace. Confirma operacionalidade renderizando uma figura
    mínima via BytesIO, detectando instalações corrompidas antes do primeiro
    gráfico real.

    Chamada uma vez no __init__ do ChartTool e resultado cacheado em
    self._kaleido_available para evitar overhead por gráfico gerado.
    """
    if importlib.util.find_spec("kaleido") is None:
        return False
    try:
        import io as _io
        # Teste mínimo: figura vazia 1×1 — falha rápida se o renderizador
        # interno do kaleido não inicializar corretamente.
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
# TIPOS E CONFIGURACAO
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
    """
    Parâmetros visuais globais aplicados a todos os gráficos.

    A paleta segue a ordem padrão do Matplotlib para manter consistência
    visual quando os mesmos dados são comparados entre gráficos distintos.

    default_output_dirs
        Diretório de fallback quando output_dirs não é fornecido ao construtor.
        Não deve ser alterado em subclasses — passe output_dirs no construtor.
    """
    default_theme:        ChartTheme = ChartTheme.LIGHT
    default_height:       int        = 500
    default_width:        int        = 900
    enable_interactivity: bool       = True
    color_palette: List[str] = field(default_factory=lambda: [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    ])


@dataclass
class ChartMetadata:
    """
    Metadados registrados por gráfico gerado.

    export_path
        Caminho absoluto do arquivo HTML gerado. Sempre preenchido quando
        _write_and_record() conclui sem exceção.
    export_path_png
        Caminho absoluto do arquivo PNG gerado via pio.write_image(). None
        quando kaleido não está instalado ou write_image() falha — a ausência
        do PNG não invalida o registro; o HTML permanece como artefato primário.
    """
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
    Gerador de gráficos Plotly para o pipeline SRAG.

    Parâmetros
    ----------
    spark
        SparkSession. Obrigatória para os métodos _generate_*, que executam
        Spark SQL diretamente. Não é necessária para create_*_chart() e
        generate_custom_chart(), que recebem dados já materializados.
    audit_logger
        Instância de AuditLogger. Quando None, usa o stub local que imprime
        no stdout — suficiente para testes.
    config
        ChartConfig com parâmetros visuais globais. Quando None, usa defaults.
    output_dirs
        Dict mapeando tipo de gráfico para diretório de destino. Chaves
        reconhecidas: "line", "mensal", "bar", "area", "multi_line",
        "heatmap", "pie", "default". Quando o tipo não está no dict, usa
        "default". Quando None, todos os gráficos vão para o diretório
        padrão em _DEFAULT_OUTPUT_BASE/custom.
    catalog / schema
        Identificadores Unity Catalog usados nas queries internas dos métodos
        _generate_*. Sem hardcode para permitir execução em schemas de teste.
    dbutils
        Objeto dbutils do Databricks. Quando fornecido, a escrita de arquivos
        usa dbutils.fs.put() em vez de open() nativo, garantindo persistência
        em Unity Catalog Volumes em Databricks Runtime < 13.x.
    """

    def __init__(
        self,
        spark                             = None,
        audit_logger: Optional[AuditLogger]    = None,
        config:       Optional[ChartConfig]    = None,
        output_dirs:  Optional[Dict[str, Path]] = None,
        catalog:      str = "dbx_srag_lab",
        schema:       str = "gold",
        dbutils                           = None,
    ):
        self.spark   = spark
        self.audit   = audit_logger if audit_logger else AuditLogger()
        self.config  = config or ChartConfig()
        self.catalog = catalog
        self.schema  = schema
        self.dbutils = dbutils

        self._output_dirs = self._init_output_dirs(output_dirs)
        self._id_lock     = threading.Lock()
        self._charts_created        = 0
        self._total_generation_time = 0.0

        # Kaleido verificado uma vez — resultado cacheado para evitar
        # tentativas repetidas (e exceções repetidas no audit) por gráfico.
        # Quando False, _write_chart_png retorna None imediatamente.
        self._kaleido_available: bool = _check_kaleido_available()

        self.audit.log_event(
            AuditEvent.TOOL_INITIALIZED,
            {
                "tool":            "ChartTool",
                "output_dirs":     {k: str(v) for k, v in self._output_dirs.items()},
                "has_spark":       spark   is not None,
                "has_dbutils":     dbutils is not None,
                "png_export":      "enabled" if self._kaleido_available else (
                    "disabled — kaleido nao instalado. "
                    "Execute: %pip install kaleido  (ou chart_tool.try_enable_png())"
                ),
            },
            EventStatus.INFO if self._kaleido_available else EventStatus.WARNING,
        )

    def _init_output_dirs(self, output_dirs: Optional[Dict[str, Path]]) -> Dict[str, Path]:
        """
        Inicializa e cria os diretórios de saída por tipo de gráfico.

        Quando output_dirs é None, usa um único diretório padrão mapeado
        como "default". Diretórios inacessíveis fazem fallback para um
        diretório temporário local, registrando aviso no audit.
        """
        import tempfile

        if output_dirs is None:
            default = Path(f"{_DEFAULT_OUTPUT_BASE}/custom")
            output_dirs = {"default": default}

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
                    {
                        "reason":      f"output_dir '{key}' inacessivel: {exc}",
                        "fallback_dir": str(tmp),
                    },
                    EventStatus.WARNING,
                )
                initialized[key] = tmp

        if "default" not in initialized:
            fallback = list(initialized.values())[0]
            initialized["default"] = fallback

        return initialized

    def _resolve_output_dir(self, chart_type: str) -> Path:
        """
        Retorna o diretório de destino para o tipo de gráfico informado.

        A lookup usa o tipo exato e faz fallback para "default" quando o
        tipo não está mapeado. Isso permite que tipos novos sejam adicionados
        sem quebrar execuções existentes.
        """
        return self._output_dirs.get(chart_type, self._output_dirs["default"])

    # =========================================================================
    # METODOS PUBLICOS — TIPOS DE GRAFICO
    # =========================================================================

    def create_line_chart(
        self,
        data:  Union[pd.DataFrame, List[Dict]],
        title: str,
        x_col: str,
        y_col: str,
        **kwargs,
    ) -> Optional[Dict]:
        """
        Gráfico de linha simples com marcadores.

        Retorna None (sem lançar exceção) quando o DataFrame está vazio ou
        quando ocorre qualquer falha de renderização, para não interromper
        o pipeline em caso de dado ausente.
        """
        try:
            start = time.time()
            self.audit.log_event(
                AuditEvent.CHART_GENERATION_START,
                {"type": "line", "title": title},
                EventStatus.INFO,
            )

            df = self._ensure_dataframe(data)
            if df.empty:
                return None

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df[x_col], y=df[y_col],
                mode="lines+markers", name=y_col,
                line=dict(color=self.config.color_palette[0], width=2),
                marker=dict(size=6),
            ))
            fig.update_layout(
                title=title, xaxis_title=x_col, yaxis_title=y_col,
                template=self.config.default_theme.value,
                hovermode="x unified",
            )
            self._apply_standard_layout(fig, "line")

            return self._write_and_record(fig, "line", title, len(df), start)

        except Exception as exc:
            self.audit.log_event(
                AuditEvent.CHART_ERROR,
                {"type": "line", "error": str(exc)},
                EventStatus.ERROR,
            )
            return None

    def create_bar_chart(
        self,
        data:  Union[pd.DataFrame, List[Dict]],
        title: str,
        x_col: str,
        y_col: str,
        **kwargs,
    ) -> Optional[Dict]:
        """
        Gráfico de barras verticais.

        Retorna None quando o DataFrame está vazio, sem lançar exceção.
        """
        try:
            start = time.time()
            df = self._ensure_dataframe(data)
            if df.empty:
                return None

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df[x_col], y=df[y_col], name=y_col,
                marker_color=self.config.color_palette[1],
            ))
            fig.update_layout(
                title=title, xaxis_title=x_col, yaxis_title=y_col,
                template=self.config.default_theme.value,
            )
            self._apply_standard_layout(fig, "bar")

            return self._write_and_record(fig, "bar", title, len(df), start)

        except Exception as exc:
            self.audit.log_event(
                AuditEvent.CHART_ERROR,
                {"type": "bar", "error": str(exc)},
                EventStatus.ERROR,
            )
            return None

    def create_area_chart(
        self,
        data:  Union[pd.DataFrame, List[Dict]],
        title: str,
        x_col: str,
        y_col: str,
        **kwargs,
    ) -> Optional[Dict]:
        """
        Gráfico de área preenchida até o eixo zero.

        Retorna None quando o DataFrame está vazio, sem lançar exceção.
        """
        try:
            start = time.time()
            df = self._ensure_dataframe(data)
            if df.empty:
                return None

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df[x_col], y=df[y_col],
                fill="tozeroy", name=y_col,
                line=dict(color=self.config.color_palette[2]),
            ))
            fig.update_layout(
                title=title, xaxis_title=x_col, yaxis_title=y_col,
                template=self.config.default_theme.value,
            )
            self._apply_standard_layout(fig, "area")

            return self._write_and_record(fig, "area", title, len(df), start)

        except Exception as exc:
            self.audit.log_event(
                AuditEvent.CHART_ERROR,
                {"type": "area", "error": str(exc)},
                EventStatus.ERROR,
            )
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
        Gráfico de linha com múltiplas séries sobrepostas.

        Colunas ausentes no DataFrame são silenciosamente ignoradas para
        permitir que o breakdown viral seja renderizado mesmo quando um
        agente viral não tem registros no período consultado.
        """
        try:
            start = time.time()
            df = self._ensure_dataframe(data)
            if df.empty:
                return None

            fig = go.Figure()
            for i, col in enumerate(y_cols):
                if col not in df.columns:
                    continue
                fig.add_trace(go.Scatter(
                    x=df[x_col], y=df[col],
                    mode="lines+markers", name=col,
                    line=dict(
                        color=self.config.color_palette[i % len(self.config.color_palette)],
                        width=2,
                    ),
                    marker=dict(size=5),
                ))
            fig.update_layout(
                title=title, xaxis_title=x_col, yaxis_title="Casos",
                template=self.config.default_theme.value,
                hovermode="x unified",
            )
            self._apply_standard_layout(fig, "multi_line")

            return self._write_and_record(fig, "multi_line", title, len(df), start)

        except Exception as exc:
            self.audit.log_event(
                AuditEvent.CHART_ERROR,
                {"type": "multi_line", "error": str(exc)},
                EventStatus.ERROR,
            )
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
        """
        Heatmap a partir de dados no formato longo (long format).

        O pivot para formato matricial é feito internamente. Quando há
        combinações (x, y) duplicadas, o pivot lança ValueError — cabe ao
        chamador garantir unicidade antes de invocar este método.
        """
        try:
            start = time.time()
            df = self._ensure_dataframe(data)
            if df.empty:
                return None

            pivot_df = df.pivot(index=y_col, columns=x_col, values=z_col)
            fig = go.Figure(data=go.Heatmap(
                z=pivot_df.values,
                x=pivot_df.columns,
                y=pivot_df.index,
                colorscale="RdYlGn_r",
            ))
            fig.update_layout(
                title=title, xaxis_title=x_col, yaxis_title=y_col,
                template=self.config.default_theme.value,
            )
            self._apply_standard_layout(fig, "heatmap")

            return self._write_and_record(fig, "heatmap", title, len(df), start)

        except Exception as exc:
            self.audit.log_event(
                AuditEvent.CHART_ERROR,
                {"type": "heatmap", "error": str(exc)},
                EventStatus.ERROR,
            )
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
        Gráfico de pizza. Adequado para distribuições com até ~8 categorias.

        Retorna None quando o DataFrame está vazio, sem lançar exceção.
        """
        try:
            start = time.time()
            df = self._ensure_dataframe(data)
            if df.empty:
                return None

            fig = go.Figure(data=[go.Pie(
                labels=df[labels_col],
                values=df[values_col],
                marker=dict(colors=self.config.color_palette),
            )])
            fig.update_layout(
                title=title,
                template=self.config.default_theme.value,
            )
            self._apply_standard_layout(fig, "pie")

            return self._write_and_record(fig, "pie", title, len(df), start)

        except Exception as exc:
            self.audit.log_event(
                AuditEvent.CHART_ERROR,
                {"type": "pie", "error": str(exc)},
                EventStatus.ERROR,
            )
            return None

    # =========================================================================
    # GRAFICO AD-HOC
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
        Ponto de entrada unificado para gráficos ad-hoc gerados pelo nó
        execute_chart do orquestrador.

        Recebe dados já materializados (resultado da SQL dinâmica) e delega
        para o método create_*_chart correspondente ao chart_type solicitado.

        Contrato de mapeamento de colunas por tipo
        ------------------------------------------
        "pie"
            x_col é interpretado como labels_col (categorias do gráfico de pizza).
            y_col é interpretado como values_col (magnitudes correspondentes).
            Esse mapeamento é necessário porque create_pie_chart() tem assinatura
            diferente dos demais create_*_chart(). O chamador deve garantir que
            x_col contenha rótulos categóricos e y_col contenha valores numéricos.

        "heatmap"
            x_col e y_col definem os eixos da matriz. z_col define os valores
            de intensidade. Quando z_col é None e chart_type="heatmap", o método
            registra aviso no AuditLogger e retorna None — gerar um heatmap sem
            a coluna Z produziria um gráfico incorreto silenciosamente.

        "multi_line"
            y_cols define as séries. Quando y_cols é None, usa [y_col] como
            fallback de uma única série.

        Tipos não reconhecidos fazem fallback para "bar" com aviso no audit,
        em vez de falhar silenciosamente como na versão anterior.

        Parâmetros
        ----------
        data       : dados em DataFrame ou lista de dicts.
        chart_type : "bar" | "line" | "area" | "pie" | "multi_line" | "heatmap".
        title      : título do gráfico, gerado pelo IntentRouter.
        x_col      : coluna de agrupamento (eixo X ou labels para pie).
        y_col      : coluna de métrica (eixo Y ou values para pie).
        y_cols     : lista de colunas Y para multi_line.
        z_col      : coluna de intensidade para heatmap.

        Retorna o mesmo dict que os métodos create_*_chart ou None em falha.
        """
        if chart_type == "heatmap":
            if z_col is None:
                self.audit.log_event(
                    AuditEvent.CHART_ERROR,
                    {
                        "type":   "heatmap",
                        "reason": "z_col obrigatorio para heatmap — nenhum grafico gerado",
                    },
                    EventStatus.WARNING,
                )
                return None
            return self.create_heatmap(data, title, x_col, y_col, z_col)

        dispatch = {
            "bar":        lambda: self.create_bar_chart(data, title, x_col, y_col),
            "line":       lambda: self.create_line_chart(data, title, x_col, y_col),
            "area":       lambda: self.create_area_chart(data, title, x_col, y_col),
            "pie":        lambda: self.create_pie_chart(data, title, x_col, y_col),
            "multi_line": lambda: self.create_multi_line_chart(
                data, title, x_col, y_cols if y_cols else [y_col]
            ),
        }

        if chart_type not in dispatch:
            self.audit.log_event(
                AuditEvent.CHART_ERROR,
                {
                    "type":     chart_type,
                    "reason":   "chart_type nao reconhecido — fallback para bar",
                    "received": chart_type,
                },
                EventStatus.WARNING,
            )

        return dispatch.get(chart_type, dispatch["bar"])()

    # =========================================================================
    # GRAFICOS PADRAO DO PIPELINE
    # =========================================================================

    def generate_all_charts(self) -> List[Dict]:
        """
        Gera o conjunto fixo de gráficos padrão do pipeline SRAG.

        Este método é de uso explícito e opt-in. Deve ser chamado somente
        quando a intenção classificada da query do usuário inclui visualização.
        Chamá-lo após toda execução SQL gera work amplification desnecessário
        — cinco queries Spark e dez escritas em disco (HTML + PNG por gráfico)
        para perguntas que não solicitam gráficos.

        Retorna List[Dict] com os metadados de cada gráfico gerado com
        sucesso, incluindo path, path_png, chart_type e metadata completa.
        Falhas individuais são capturadas por cada _generate_* e não
        interrompem os demais. Gráficos que falharam são omitidos da lista
        retornada.

        O retorno de dicts completos (em vez de apenas paths) permite que o
        chamador acesse file_size, generation_time e chart_type sem precisar
        reabrir o arquivo ou inferir o tipo pelo nome.
        """
        generators = [
            (self._generate_time_series_chart,      "diario"),
            (self._generate_monthly_chart,           "mensal"),
            (self._generate_geographic_chart,        "geografico"),
            (self._generate_age_distribution_chart,  "demografico"),
            (self._generate_viral_breakdown_chart,   "viral"),
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
        """
        Série diária de casos — últimos 30 dias (gold_serie_diaria_30d).

        A ordenação ASC é aplicada após o toPandas() porque a query usa
        ORDER BY DESC com LIMIT para garantir que os 30 dias mais recentes
        sejam selecionados antes de qualquer filtro adicional. Inverter a
        ordem só no DataFrame evita uma segunda passagem pela tabela.
        """
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

            df = df.sort_values("data_referencia", ascending=True).reset_index(drop=True)

            return self.create_line_chart(
                data=df,
                title="Evolucao de Casos Diarios — SRAG (Ultimos 30 dias)",
                x_col="data_referencia",
                y_col="casos_dia",
            )
        except Exception as exc:
            self.audit.log_event(
                AuditEvent.CHART_ERROR,
                {"type": "diario", "error": str(exc)},
                EventStatus.ERROR,
            )
            return None

    def _generate_monthly_chart(self) -> Optional[Dict]:
        """
        Evolução mensal de casos — últimos 12 meses (gold_metricas_temporais).

        A query usa CTE para calcular o valor de corte de data antes do WHERE.
        Window functions (MAX(...) OVER()) são proibidas na cláusula WHERE em
        Spark SQL e lançam AnalysisException. A versão anterior usava essa
        construção, fazendo o gráfico mensal nunca ser gerado — o erro era
        absorvido silenciosamente pelo except genérico. A CTE isola o cálculo
        em uma subquery que retorna um escalar, permitido no WHERE.

        O prefixo "srag_mensal_" no chart_id, gerado por _generate_chart_id,
        é necessário para que o classificador do notebook 06 identifique este
        gráfico como obrigatório na categoria mensal.
        """
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
                SELECT
                    t.ano_mes,
                    SUM(t.total_casos) AS total_casos
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
                marker_color=self.config.color_palette[1],
                text=df["total_casos"].apply(lambda v: f"{v:,.0f}"),
                textposition="outside",
            ))
            fig.update_layout(
                title="Evolucao Mensal de Casos SRAG — Ultimos 12 Meses",
                xaxis_title="Mes",
                yaxis_title="Total de Casos",
                template=self.config.default_theme.value,
            )
            self._apply_standard_layout(fig, "mensal")

            return self._write_and_record(
                fig, "mensal", "Evolucao Mensal SRAG", len(df), start
            )

        except Exception as exc:
            self.audit.log_event(
                AuditEvent.CHART_ERROR,
                {"type": "mensal", "error": str(exc)},
                EventStatus.ERROR,
            )
            return None

    def _generate_geographic_chart(self) -> Optional[Dict]:
        """Top 10 UFs por total de casos (gold_metricas_geograficas)."""
        if not self.spark:
            return None
        try:
            df = self.spark.sql(f"""
                SELECT sg_uf, SUM(total_casos) AS total_casos
                FROM {self.catalog}.{self.schema}.gold_metricas_geograficas
                WHERE total_casos IS NOT NULL
                  AND sg_uf IS NOT NULL
                GROUP BY sg_uf
                ORDER BY total_casos DESC
                LIMIT 10
            """).toPandas()

            if df.empty:
                return None

            return self.create_bar_chart(
                data=df,
                title="Top 10 Estados por Casos SRAG",
                x_col="sg_uf",
                y_col="total_casos",
            )
        except Exception as exc:
            self.audit.log_event(
                AuditEvent.CHART_ERROR,
                {"type": "geografico", "error": str(exc)},
                EventStatus.ERROR,
            )
            return None

    def _generate_age_distribution_chart(self) -> Optional[Dict]:
        """
        Distribuição de casos por faixa etária (gold_metricas_demograficas).

        A coluna física é faixa_etaria. O alias faixa_etaria_label é mantido
        no SELECT para compatibilidade com o x_col passado ao create_bar_chart
        sem renomear a coluna no schema Gold.
        """
        if not self.spark:
            return None
        try:
            df = self.spark.sql(f"""
                SELECT faixa_etaria AS faixa_etaria_label, SUM(total_casos) AS total_casos
                FROM {self.catalog}.{self.schema}.gold_metricas_demograficas
                WHERE faixa_etaria IS NOT NULL
                  AND total_casos IS NOT NULL
                GROUP BY faixa_etaria, ordem_faixa
                ORDER BY ordem_faixa ASC NULLS LAST
            """).toPandas()

            if df.empty:
                return None

            return self.create_bar_chart(
                data=df,
                title="Distribuicao de Casos por Faixa Etaria",
                x_col="faixa_etaria_label",
                y_col="total_casos",
            )
        except Exception as exc:
            self.audit.log_event(
                AuditEvent.CHART_ERROR,
                {"type": "demografico", "error": str(exc)},
                EventStatus.ERROR,
            )
            return None

    def _generate_viral_breakdown_chart(self) -> Optional[Dict]:
        """
        Breakdown viral diário — últimos 30 dias (gold_serie_diaria_30d).

        As colunas disponíveis são as classificações virais (total_covid,
        total_influenza, total_outro_virus, total_sem_classificacao). A tabela
        não expõe total_obitos nesta granularidade. As classificações virais
        são epidemiologicamente mais informativas para análise diária.
        """
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
                title="Breakdown Viral Diario — SRAG (Ultimos 30 dias)",
                x_col="data_referencia",
                y_cols=["COVID_19", "Influenza", "Outro_Virus", "Sem_Classificacao"],
            )
        except Exception as exc:
            self.audit.log_event(
                AuditEvent.CHART_ERROR,
                {"type": "viral", "error": str(exc)},
                EventStatus.ERROR,
            )
            return None

    def _generate_gender_chart(self) -> Optional[Dict]:
        """
        Distribuição de casos por sexo (gold_metricas_demograficas).

        Não incluído em generate_all_charts porque a tabela agrega faixa
        etária e sexo no mesmo registro. Um groupby extra é necessário para
        evitar contagem dupla, tornando o gráfico menos direto que os demais.
        Disponível para uso pontual via chamada direta.
        """
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
                title="Distribuicao de Casos por Sexo",
                labels_col="sexo_label",
                values_col="total_casos",
            )
        except Exception as exc:
            self.audit.log_event(
                AuditEvent.CHART_ERROR,
                {"type": "sexo", "error": str(exc)},
                EventStatus.ERROR,
            )
            return None

    # =========================================================================
    # LAYOUT E PERSISTENCIA
    # =========================================================================

    def _apply_standard_layout(self, fig: go.Figure, chart_type: str) -> None:
        """
        Aplica configurações visuais padronizadas por tipo de gráfico.

        Centraliza altura, margens e configuração de eixo X para evitar
        inconsistências entre os cinco tipos de gráfico gerados no pipeline.

        O gráfico mensal recebe tratamento especial de eixo X: tickmode="linear"
        com dtick=1 força a exibição de todos os labels de mês. O comportamento
        padrão do Plotly (tickmode="auto") omite labels quando detecta risco de
        sobreposição — em séries de 12 meses, apenas 4 a 6 labels ficam visíveis,
        o que em contexto epidemiológico pode ser interpretado como dado faltante.
        """
        base_layout = dict(
            height=self.config.default_height,
            width=self.config.default_width,
            margin=dict(l=60, r=40, t=60, b=60),
        )

        if chart_type == "mensal":
            base_layout["height"] = 520
            base_layout["margin"]["b"] = 90
            fig.update_xaxes(
                type="category",
                tickmode="linear",
                dtick=1,
                tickangle=-45,
                automargin=True,
            )
        elif chart_type in ("bar", "geografico", "demografico"):
            fig.update_xaxes(tickangle=-30, automargin=True)
        elif chart_type in ("line", "multi_line", "diario", "viral"):
            fig.update_xaxes(automargin=True)

        fig.update_layout(**base_layout)

    def _write_chart_html(self, fig: go.Figure, output_path: Path) -> None:
        """
        Persiste o HTML do gráfico de forma confiável.

        include_plotlyjs="cdn" reduz cada arquivo de ~3.7 MB para ~8 KB,
        carregando a biblioteca via CDN. O tradeoff é que o HTML não funciona
        offline — aceitável dado que os relatórios SRAG são consultados em
        ambientes com acesso à internet.

        Estratégia de escrita:
        1. Se dbutils está disponível, usa dbutils.fs.put() — confiável em
           todas as versões do DBR, incluindo < 13.x com Unity Catalog Volumes.
        2. Fallback para fig.write_html() nativo (funciona em DBR 13.x+ e local).

        Lança exceção em caso de falha total para que _write_and_record possa
        registrar o erro de auditoria com contexto preciso.
        """
        html_content = fig.to_html(full_html=True, include_plotlyjs="cdn")

        if self.dbutils:
            try:
                self.dbutils.fs.put(str(output_path), html_content, overwrite=True)
                return
            except Exception as exc:
                self.audit.log_event(
                    AuditEvent.CHART_WRITE_ERROR,
                    {
                        "path":     str(output_path),
                        "method":   "dbutils.fs.put",
                        "error":    str(exc),
                        "fallback": "write_html nativo",
                    },
                    EventStatus.WARNING,
                )

        with open(str(output_path), "w", encoding="utf-8") as f:
            f.write(html_content)

    def _write_chart_png(self, fig: go.Figure, html_path: Path) -> Optional[Path]:
        """
        Exporta o gráfico como PNG estático via plotly.io.write_image().

        O PNG é gerado diretamente a partir do objeto Figure com as mesmas
        dimensões definidas em ChartConfig. O arquivo é salvo no mesmo
        diretório do HTML correspondente, com o mesmo chart_id como base
        de nome e extensão .png.

        Guard de disponibilidade
            self._kaleido_available é verificado antes de qualquer tentativa.
            Quando False (kaleido não instalado ou com instalação corrompida),
            o método retorna None imediatamente sem lançar exceção — evita os
            5 chart_write_error no audit por execução que ocorriam quando a
            verificação era feita aqui em vez de no __init__.
            Para ativar o PNG sem reiniciar o ChartTool, use try_enable_png().

        Este método é não-bloqueante por design: qualquer falha produz aviso
        no AuditLogger e retorna None. O HTML permanece o artefato primário.

        Retorna o Path do PNG gerado ou None em caso de falha ou kaleido ausente.
        """
        if not self._kaleido_available:
            return None

        png_path = html_path.with_suffix(".png")
        try:
            pio.write_image(
                fig,
                str(png_path),
                format="png",
                width=self.config.default_width,
                height=self.config.default_height,
            )
            return png_path
        except Exception as exc:
            # Kaleido estava disponível no init mas falhou agora — pode ser
            # timeout do renderizador ou permissão de escrita no Volume path.
            self.audit.log_event(
                AuditEvent.CHART_WRITE_ERROR,
                {
                    "path":   str(png_path),
                    "method": "pio.write_image",
                    "error":  str(exc),
                    "hint":   (
                        "Kaleido inicializou mas falhou ao renderizar. "
                        "Verifique permissao de escrita no path e versao do kaleido: "
                        "pip install --upgrade kaleido"
                    ),
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
        """
        Persiste o gráfico em HTML e PNG, atualiza contadores internos e
        registra no AuditLogger. Centralizado para eliminar duplicação de
        lógica de escrita e auditoria entre os create_*_chart().

        A escrita HTML é bloqueante: uma falha levanta exceção e aborta o
        registro. A escrita PNG é não-bloqueante: uma falha registra aviso
        e preenche path_png com None no retorno, sem interromper o fluxo.

        A escrita e o stat() têm blocos try/except independentes para que
        falhas de permissão em Volume paths sejam distinguidas de falhas de
        renderização. Sem essa separação, um FileNotFoundError em stat() seria
        absorvido pelo except genérico do create_*_chart() sem identificar
        que o arquivo foi escrito mas não pôde ser inspecionado.
        """
        chart_id    = self._generate_chart_id(chart_type)
        output_dir  = self._resolve_output_dir(chart_type)
        output_path = output_dir / f"{chart_id}.html"

        try:
            self._write_chart_html(fig, output_path)
        except Exception as exc:
            self.audit.log_event(
                AuditEvent.CHART_WRITE_ERROR,
                {
                    "chart_id": chart_id,
                    "path":     str(output_path),
                    "error":    str(exc),
                },
                EventStatus.ERROR,
            )
            raise

        png_path = self._write_chart_png(fig, output_path)

        elapsed = time.time() - start_time

        with self._id_lock:
            self._charts_created        += 1
            self._total_generation_time += elapsed

        try:
            file_size = output_path.stat().st_size
        except Exception as exc:
            file_size = len(fig.to_html(include_plotlyjs=False))
            self.audit.log_event(
                AuditEvent.CHART_STAT_ERROR,
                {
                    "chart_id":        chart_id,
                    "path":            str(output_path),
                    "error":           str(exc),
                    "file_size_proxy": file_size,
                },
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
    # MANUTENCAO E UTILITARIOS
    # =========================================================================

    def try_enable_png(self) -> bool:
        """
        Tenta instalar kaleido via pip e reabilita o export PNG.

        Útil quando o ChartTool foi instanciado antes de kaleido estar
        disponível (ex.: notebook que instala dependências na mesma sessão).
        Não é necessário recriar o objeto ChartTool após chamar este método.

        Em Databricks, a forma recomendada é usar `%pip install kaleido`
        em uma célula separada antes de importar o módulo. Este método existe
        como alternativa programática para scripts que não controlam o ambiente
        de instalação.

        Retorno
        -------
        True quando kaleido foi instalado com sucesso e está operacional.
        False quando a instalação falhou ou o ambiente não permite pip install.
        """
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
                {
                    "reason": "falha ao instalar kaleido via pip",
                    "error":  str(exc),
                    "hint":   "use %pip install kaleido em uma celula de notebook",
                },
                EventStatus.WARNING,
            )
            return False

        # Revalida após instalação — o import pode precisar do módulo recém instalado
        self._kaleido_available = _check_kaleido_available()

        self.audit.log_event(
            AuditEvent.TOOL_INITIALIZED,
            {
                "event":       "kaleido_install_attempt",
                "png_export":  "enabled" if self._kaleido_available else "failed",
            },
            EventStatus.SUCCESS if self._kaleido_available else EventStatus.WARNING,
        )
        return self._kaleido_available

    def cleanup_old_charts(self, max_files: int = 100) -> int:
        """
        Remove os arquivos mais antigos quando o total excede max_files.

        Percorre todos os diretórios mapeados em output_dirs e aplica o
        limite de forma independente por diretório. A remoção considera tanto
        HTMLs quanto PNGs, ordenados por data de modificação (os mais antigos
        primeiro). Arquivos que não podem ser removidos são ignorados
        individualmente com aviso no audit — uma falha de permissão em um
        arquivo não deve abortar a limpeza dos demais.

        Retorna o número total de arquivos removidos em todos os diretórios.
        """
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
                            {
                                "dir":    dir_key,
                                "file":   f.name,
                                "reason": "falha ao remover",
                                "error":  str(exc),
                            },
                            EventStatus.WARNING,
                        )

                total_removed += removed

                if removed:
                    self.audit.log_event(
                        AuditEvent.CHART_CLEANUP,
                        {
                            "dir":       dir_key,
                            "removed":   removed,
                            "remaining": len(all_files) - removed,
                            "max_files": max_files,
                        },
                        EventStatus.INFO,
                    )

            except Exception as exc:
                self.audit.log_event(
                    AuditEvent.CHART_CLEANUP,
                    {"dir": dir_key, "error": str(exc)},
                    EventStatus.WARNING,
                )

        return total_removed

    def _ensure_dataframe(self, data: Union[pd.DataFrame, List[Dict]]) -> pd.DataFrame:
        """
        Normaliza a entrada para DataFrame.

        Lança ValueError para tipos não suportados — erro de contrato do
        chamador, não capturado pelo handler genérico dos create_*_chart().
        """
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, list):
            return pd.DataFrame(data)
        raise ValueError(f"Tipo nao suportado: {type(data)}")

    def _generate_chart_id(self, chart_type: str) -> str:
        """
        Gera um ID único por gráfico no formato srag_{tipo}_{timestamp}_{uuid4_curto}.

        O prefixo "srag_" é exigido pelo classificador do notebook 06 para
        diferenciar arquivos do pipeline de outros HTMLs no diretório de saída.

        uuid4 em vez de contador garante unicidade em execuções paralelas sem
        necessidade de lock. O contador anterior (_charts_created era lido e
        incrementado em operações não atômicas separadas) permitia que dois
        gráficos recebessem o mesmo ID antes de qualquer incremento.

        O lock em _write_and_record() ainda protege o incremento de
        _charts_created (usado para estatísticas), mas o ID não depende mais
        desse contador.
        """
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid = uuid.uuid4().hex[:8]
        return f"srag_{chart_type}_{ts}_{uid}"

    def get_stats(self) -> Dict:
        """Estatísticas acumuladas de geração desde a instanciação."""
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
    Alias mantido para compatibilidade com imports existentes no notebook 06.

    Emite DeprecationWarning em toda instanciação para sinalizar que o nome
    canônico é ChartTool. Sera removido em versão futura — migre todos os
    imports para ChartTool diretamente.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "ChartGenerator foi renomeado para ChartTool. "
            "Atualize os imports para evitar quebra quando o alias for removido.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)