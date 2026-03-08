"""
Tools - Ferramentas Especializadas
==================================

Exports:
- GoldSQLTool: Execução SQL segura com guardrails
- ReportGenerator: Geração de relatórios estruturados (Markdown)
- WebSearchTool: Busca de notícias (opcional)
- ChartTool: Geração de gráficos (opcional)

Aliases de compatibilidade:
- TavilySearchTool -> WebSearchTool
- ChartGenerator   -> ChartTool
"""

from __future__ import annotations

from typing import Optional, Type

# Core tools (devem existir)
from src.tools.sql_tool import GoldSQLTool
from src.tools.report_generator import ReportGenerator, ReportSection

# Optional tools (podem falhar por dependência / ambiente)
WebSearchTool: Optional[Type] = None
ChartTool: Optional[Type] = None

try:
    from src.tools.web_search_tool import WebSearchTool as _WebSearchTool
    WebSearchTool = _WebSearchTool
except Exception:
    WebSearchTool = None

try:
    from src.tools.chart_tool import ChartTool as _ChartTool
    ChartTool = _ChartTool
except Exception:
    ChartTool = None

# Compatibility aliases
TavilySearchTool = WebSearchTool
ChartGenerator = ChartTool

__all__ = [
    "GoldSQLTool",
    "ReportGenerator",
    "ReportSection",
    "WebSearchTool",
    "ChartTool",
    "TavilySearchTool",
    "ChartGenerator",
]