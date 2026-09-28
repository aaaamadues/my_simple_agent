"""工具抽象与注册中心。

每个工具 = 名称 + 描述 + 参数 JSON Schema + 执行函数。
LLM 拿到全部 schema 后自主决定是否调用、传什么参数；执行时按名称从注册表查回。
新增工具只需写一个 Tool 子类，然后在 tools/__init__.py 里 register。
"""
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolContext:
    """工具执行时可见的上下文（todo 等有状态工具按 session 隔离数据）。"""
    session: Any = None


class Tool:
    name: str = ""
    description: str = ""
    parameters: dict = {}  # JSON Schema

    def schema(self) -> dict:
        """转成 OpenAI 兼容的 tools 数组元素。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def run(self, args: dict, ctx: ToolContext) -> Any:
        raise NotImplementedError


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"工具重复注册: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]
