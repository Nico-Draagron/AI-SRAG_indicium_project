"""
Tools - Ferramentas Especializadas
===================================

Ferramentas para execução de tarefas:
- GoldSQLTool: Execução SQL segura com guardrails
- ReportGenerator: Geração de relatórios estruturados
- TavilySearchTool: Busca de notícias (opcional)
- ChartGenerator: Geração de gráficos (opcional)
"""

from src.tools.sql_tool import GoldSQLTool
from src.tools.report_generator import ReportGenerator, ReportSection
from src.tools.web_search_tool import WebSearchTool
from src.tools.chart_tool import ChartTool

# Compatibility aliases for backward compatibility
TavilySearchTool = WebSearchTool
ChartGenerator = ChartTool

__all__ = [
    # Core Tools
    "GoldSQLTool",
    "ReportGenerator", 
    "ReportSection",
    
    # Optional Tools - Native names
    "WebSearchTool",
    "ChartTool",
    
    # Optional Tools - Compatibility aliases
    "TavilySearchTool", 
    "ChartGenerator"
]