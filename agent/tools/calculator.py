"""计算器工具：基于 Python ast 白名单的安全求值（不用 eval）。"""
import ast
import operator
from typing import Any

from .base import Tool, ToolContext

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _safe_eval(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("指数过大，拒绝计算")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"不支持的表达式元素: {type(node).__name__}")


def _pretty(v: int | float) -> int | float:
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e15:
        return int(v)
    if isinstance(v, float):
        return round(v, 10)  # 去掉浮点噪声
    return v


class CalculatorTool(Tool):
    name = "calculator"
    description = (
        "数学计算器。支持 + - * / // % **（乘方）与括号的算术表达式，例如 '(3.5+2)*8'、'2^10'（^ 视为乘方）。"
        "凡是需要精确数值计算的都应使用本工具，不要心算。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "算术表达式，例如 (1+2)*3/4"},
        },
        "required": ["expression"],
    }

    async def run(self, args: dict, ctx: ToolContext) -> Any:
        expr = str(args.get("expression", "")).strip()
        if not expr:
            raise ValueError("缺少 expression 参数")
        normalized = expr.replace("^", "**")  # 常见写法：^ 表示乘方
        tree = ast.parse(normalized, mode="eval")
        return {"expression": expr, "result": _pretty(_safe_eval(tree))}
