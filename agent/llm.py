"""DeepSeek API 的最小流式客户端。

直接调用 OpenAI 兼容的 /chat/completions HTTP 接口（不依赖任何 SDK / agent 框架），
自己解析 SSE 流：增量 content -> token 事件；tool_calls 分片在这里累积拼装成完整调用。
"""
import json
from typing import AsyncIterator

import httpx

from . import config
from .tracing import log


class LLMError(Exception):
    """调用大模型失败（网络 / 鉴权 / 余额等）。"""


_STATUS_HINT = {401: "API Key 无效", 402: "账户余额不足", 429: "请求过于频繁"}


class DeepSeekClient:
    """流式对话。yield 三类事件：

    - {"type": "token", "text": ...}        正文增量（最终答案的一部分）
    - {"type": "reasoning", "text": ...}    思考过程增量（deepseek-reasoner 才会有，deepseek-chat 为空）
    - {"type": "message_done", ...}         一条完整 assistant 消息（含拼装好的 tool_calls）
    """

    def __init__(self) -> None:
        self._http = httpx.AsyncClient(
            base_url=config.DEEPSEEK_BASE_URL,
            headers={
                "Authorization": f"Bearer {config.DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=10.0),
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def complete(self, messages: list[dict], max_tokens: int = 1024,
                       temperature: float = 0.2) -> tuple[str, dict | None]:
        """非流式一次性调用（子 LLM 任务：上下文压缩 / 记忆提炼）。返回 (正文, usage)。"""
        payload = {
            "model": config.DEEPSEEK_MODEL,
            "messages": messages,
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            resp = await self._http.post("/chat/completions", json=payload)
        except httpx.HTTPError as e:
            raise LLMError(f"连接 DeepSeek API 失败: {e}") from e
        if resp.status_code != 200:
            hint = _STATUS_HINT.get(resp.status_code)
            raise LLMError(f"DeepSeek API 返回 {resp.status_code}"
                           f"{('（' + hint + '）') if hint else ''}: {resp.text[:300]}")
        data = resp.json()
        choices = data.get("choices") or [{}]
        content = (choices[0].get("message") or {}).get("content") or ""
        return content.strip(), data.get("usage")

    async def stream_chat(self, messages: list[dict], tools: list[dict] | None = None) -> AsyncIterator[dict]:
        payload: dict = {
            "model": config.DEEPSEEK_MODEL,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        pending_calls: dict[int, dict] = {}  # index -> {"id","name","arguments"}（流式分片累积）
        usage = None
        finish_reason = None

        log.debug("LLM 请求: %d 条消息, tools=%s", len(messages),
                  [t["function"]["name"] for t in tools] if tools else None)
        try:
            async with self._http.stream("POST", "/chat/completions", json=payload) as resp:
                if resp.status_code != 200:
                    raw = (await resp.aread()).decode("utf-8", "replace")
                    hint = _STATUS_HINT.get(resp.status_code)
                    raise LLMError(f"DeepSeek API 返回 {resp.status_code}"
                                   f"{('（' + hint + '）') if hint else ''}: {raw[:300]}")
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    finish_reason = choice.get("finish_reason") or finish_reason
                    delta = choice.get("delta") or {}
                    # 思考过程（仅 reasoner 系列模型会返回）
                    if delta.get("reasoning_content"):
                        reasoning_parts.append(delta["reasoning_content"])
                        yield {"type": "reasoning", "text": delta["reasoning_content"]}
                    # 正文增量
                    if delta.get("content"):
                        content_parts.append(delta["content"])
                        yield {"type": "token", "text": delta["content"]}
                    # 工具调用分片：按 index 累积 name / arguments
                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index", 0)
                        slot = pending_calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["name"] += fn["name"]
                        if fn.get("arguments"):
                            slot["arguments"] += fn["arguments"]
        except httpx.HTTPError as e:
            raise LLMError(f"连接 DeepSeek API 失败: {e}") from e

        log.debug("LLM 响应完成: finish=%s content=%d字 tool_calls=%d",
                  finish_reason, len("".join(content_parts)), len(pending_calls))
        yield {
            "type": "message_done",
            "content": "".join(content_parts),
            "reasoning": "".join(reasoning_parts),
            "tool_calls": [pending_calls[i] for i in sorted(pending_calls)],
            "finish_reason": finish_reason,
            "usage": usage,
        }
