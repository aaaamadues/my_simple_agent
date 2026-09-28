"""搜索工具：DuckDuckGo 联网搜索（ddgs 库）。

ddgs 是同步阻塞库，用 asyncio.to_thread 丢进线程池执行，避免卡住事件循环；
超时由 AgentLoop 的 wait_for 统一兜底。
"""
import asyncio
import re
from typing import Any

from ddgs import DDGS

from .base import Tool, ToolContext

_CJK_RE = re.compile(r"[一-鿿]")


def _region(query: str) -> str:
    """中文查询走中国区，其余走全球区，提高结果相关性。"""
    return "zh-cn" if _CJK_RE.search(query) else "wt-wt"


def _search_once(query: str, top_k: int) -> list[dict]:
    """在线程池里跑的同步搜索。"""
    with DDGS() as ddgs:
        rows = ddgs.text(query, region=_region(query), max_results=top_k)
    return [{"title": r.get("title", ""),
             "url": r.get("href", ""),
             "snippet": r.get("body", "")} for r in rows]


class SearchTool(Tool):
    name = "search"
    description = (
        "联网搜索（DuckDuckGo）。"
        "当问题涉及你不确定或可能过时的事实（新闻、价格、版本号、最新政策等）时调用。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"},
            "top_k": {"type": "integer", "description": "返回条数，默认 3，最大 5"},
        },
        "required": ["query"],
    }

    async def run(self, args: dict, ctx: ToolContext) -> Any:
        query = str(args.get("query", "")).strip()
        if not query:
            raise ValueError("缺少 query 参数")
        try:
            top_k = max(1, min(int(args.get("top_k", 3)), 5))
        except (TypeError, ValueError):
            top_k = 3

        # 每次尝试单独限时（DuckDuckGo 偶发限流会变得很慢，若不单独限时，
        # 内部重试会撑爆 AgentLoop 的全局 TOOL_TIMEOUT_S，只留下一个模糊的"超时"）
        last_err: Exception | None = None
        for attempt in range(2):
            try:
                results = await asyncio.wait_for(
                    asyncio.to_thread(_search_once, query, top_k), timeout=6.0)
                if not results:
                    return {"query": query, "results": [], "note": "搜索无结果，建议更换关键词重试"}
                return {"query": query, "results": results}
            except asyncio.TimeoutError:
                last_err = RuntimeError("DuckDuckGo 单次响应超过 6s")
            except Exception as e:  # noqa: BLE001 - 重试后仍失败则把原因回给模型
                last_err = e
            if attempt == 0:
                await asyncio.sleep(0.5)
        raise RuntimeError(f"DuckDuckGo 搜索失败: {last_err}")
