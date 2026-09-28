"""工具注册中心：内置工具在这里统一注册。"""
from .base import Tool, ToolContext, ToolRegistry
from .calculator import CalculatorTool
from .search import SearchTool
from .todo import TodoTool
from .weather import WeatherTool

registry = ToolRegistry()
for _cls in (CalculatorTool, SearchTool, TodoTool, WeatherTool):
    registry.register(_cls())

__all__ = ["Tool", "ToolContext", "ToolRegistry", "registry"]
