"""核心 Agent Loop（自行实现，不依赖任何 agent 框架）。

每次用户提问的执行流程：

    user 输入写回 history + BM25 召回长期记忆进 system prompt
    ┌──────────────── 轮次循环（最多 MAX_ROUNDS）────────────────┐
    │ 0. 上下文超窗检查：超阈值 -> 子 LLM 压缩成结构化摘要，      │
    │    全量替换历史（只留最近几条）                              │
    │ 1. 组装 context：system(+记忆+摘要) + 截断/压缩后的历史      │
    │ 2. 流式调用 LLM，token 增量实时透传给前端                    │
    │ 3. 解析完整输出：                                            │
    │    a. 含 tool_calls -> 执行工具；超长结果落盘，上下文只留    │
    │       截断预览 + 存放地址；结果写回历史，continue 下一轮     │
    │    b. 纯文本 -> 即最终回答，结束 loop                        │
    └────────────────────────────────────────────────────────────┘
    结束后：每 MEMORY_EVERY_N_TURNS 轮，后台子 LLM 提炼一条记忆落盘。

对前端产出的事件流见 main.py 的 /api/chat。
"""
import asyncio
import json
import time
from typing import AsyncIterator

from . import config
from .context_store import ToolResultStore
from .llm import DeepSeekClient, LLMError
from .memory import MemoryStore
from .prompts import MEMORY_PROMPT, SUMMARY_PROMPT, SYSTEM_PROMPT
from .tools import ToolContext
from .tracing import log


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数：中文 ≈ 0.6 字/token，其他 ≈ 4 字符/token。够触发阈值用。"""
    cjk = sum(1 for c in text if "一" <= c <= "鿿")
    return int(cjk * 0.6 + (len(text) - cjk) / 4)


def _messages_tokens(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        total += _estimate_tokens(str(m.get("content") or ""))
        if m.get("tool_calls"):
            total += _estimate_tokens(json.dumps(m["tool_calls"], ensure_ascii=False))
    return total


def _history_text(messages: list[dict]) -> str:
    """把消息列表渲染成给子 LLM 看的纯文本。"""
    parts: list[str] = []
    for m in messages:
        role = m.get("role")
        if role == "tool":
            parts.append(f"[工具结果 id={m.get('tool_call_id', '')}] {m.get('content', '')}")
        elif m.get("tool_calls"):
            calls = "; ".join(f"{c['function']['name']}({c['function']['arguments']})"
                              for c in m["tool_calls"])
            extra = f" | 附言: {m['content']}" if m.get("content") else ""
            parts.append(f"[assistant 调用工具] {calls}{extra}")
        elif role == "user":
            parts.append(f"[user] {m.get('content', '')}")
        else:
            parts.append(f"[assistant] {m.get('content', '')}")
    return "\n".join(parts)


class AgentLoop:
    def __init__(self, llm: DeepSeekClient, tool_registry, sessions) -> None:
        self.llm = llm
        self.tools = tool_registry
        self.sessions = sessions
        self.spill = ToolResultStore(config.TOOL_SPILL_DIR)
        self.memories = MemoryStore(config.MEMORY_DIR, max_chars=config.MEMORY_MAX_CHARS)

    # ---------- context 组装与基础压缩 ----------

    def build_messages(self, session, memory_block: str = "") -> list[dict]:
        """system(+摘要+记忆) + 完整历史。

        历史 append-only、写入后从不改写（超长工具结果在写入时已落盘截断），
        这样每次请求的消息列表是上一次的前缀扩展，DeepSeek 服务端前缀缓存才能命中。
        超窗不靠滑窗/分级截断，统一交给 _compact 摘要压缩处理。
        """
        system = SYSTEM_PROMPT
        if session.summary:
            system += f"\n\n【此前对话的压缩摘要（上下文超窗时由子 LLM 生成）】\n{session.summary}"
        if memory_block:
            system += f"\n\n【长期记忆（BM25 召回，可能相关）】\n{memory_block}"
        return [{"role": "system", "content": system}, *session.history]

    # ---------- 上下文超窗压缩 ----------

    def _needs_compact(self, messages: list[dict]) -> bool:
        limit = int(config.CONTEXT_WINDOW_TOKENS * config.CONTEXT_COMPACT_RATIO)
        return _messages_tokens(messages) > limit

    async def _compact(self, session, usage_total: dict) -> AsyncIterator[dict]:
        """子 LLM 把全部历史压缩成结构化摘要，然后历史只留最近几条。"""
        log.info("[session=%s] 上下文超限，触发压缩（%d 条消息）", session.id, len(session.history))
        history_text = _history_text(session.history)
        if session.summary:
            history_text = f"【此前已生成的摘要】\n{session.summary}\n\n【此后的新对话】\n{history_text}"
        try:
            summary, usage = await self.llm.complete(
                [{"role": "user",
                  "content": SUMMARY_PROMPT.format(history=history_text)}],
                max_tokens=800,
            )
        except LLMError as e:
            log.warning("[session=%s] 压缩摘要失败，降级为硬截断: %s", session.id, e)
            summary, usage = "", None
        if usage:
            for k in usage_total:
                usage_total[k] += usage.get(k, 0)

        if summary:
            session.summary = summary
            kept = session.history[-config.CONTEXT_COMPACT_KEEP:]
            # 不能从 tool 结果开头（会变成孤儿消息，API 会拒绝）
            while kept and kept[0].get("role") == "tool":
                kept.pop(0)
            session.history = kept
            yield {"type": "compact", "kept_messages": len(kept),
                   "summary_chars": len(summary), "summary": summary}
        else:  # 子 LLM 失败时的兜底：只保摘要里已有的信息 + 硬截断历史
            kept = session.history[-config.CONTEXT_COMPACT_KEEP:]
            while kept and kept[0].get("role") == "tool":
                kept.pop(0)
            session.history = kept
            yield {"type": "compact", "kept_messages": len(kept),
                   "summary_chars": 0, "summary": ""}

    # ---------- 长期记忆 ----------

    def _recall_memories(self, query: str) -> str:
        """BM25 召回相关记忆，渲染成进 system prompt 的文本块。"""
        try:
            hits = self.memories.search(query, k=config.MEMORY_RECALL_K)
        except Exception as e:  # noqa: BLE001 - 召回失败不阻塞主流程
            log.warning("记忆召回失败: %s", e)
            return ""
        if not hits:
            return ""
        lines = [f"- [会话 {h['session']} · {h['time']}] {h['text']}" for h in hits]
        log.info("召回 %d 条记忆: %s", len(hits), [h["path"] for h in hits])
        return "\n".join(lines)

    async def _distill_memory(self, session) -> None:
        """后台任务：把本会话近几轮提炼成一条记忆落盘。"""
        history_text = _history_text(session.history)
        if session.summary:
            history_text = f"【此前压缩摘要】\n{session.summary}\n\n【近期对话】\n{history_text}"
        content, _ = await self.llm.complete(
            [{"role": "user",
              "content": MEMORY_PROMPT.format(history=history_text,
                                              max_chars=config.MEMORY_MAX_CHARS)}],
            max_tokens=config.MEMORY_MAX_CHARS,
            temperature=0.3,
        )
        if content:
            path = self.memories.add(session.id, content)
            log.info("[session=%s] 记忆已落盘: %s", session.id, path)
        else:
            log.warning("[session=%s] 记忆提炼结果为空，跳过", session.id)

    def _maybe_distill_memory(self, session) -> None:
        """每 MEMORY_EVERY_N_TURNS 轮触发一次后台提炼（失败不影响主流程）。"""
        if session.user_turns - session.memory_checkpoint < config.MEMORY_EVERY_N_TURNS:
            return
        session.memory_checkpoint = session.user_turns
        started = time.perf_counter()

        async def _task() -> None:
            try:
                await self._distill_memory(session)
            except Exception as e:  # noqa: BLE001
                log.warning("[session=%s] 记忆提炼失败: %s", session.id, e)

        def _done(_) -> None:
            log.info("[session=%s] 记忆提炼完成 (%dms)", session.id,
                     int((time.perf_counter() - started) * 1000))

        task = asyncio.create_task(_task())
        task.add_done_callback(_done)

    # ---------- 主循环 ----------

    async def run(self, session, user_input: str) -> AsyncIterator[dict]:
        session.history.append({"role": "user", "content": user_input})
        session.user_turns += 1
        if session.title == "新会话":
            session.title = user_input[:20] or "新会话"
        session.touch()
        log.info("[session=%s] 用户输入: %s", session.id, user_input[:100])

        usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "prompt_cache_hit_tokens": 0}
        ctx = ToolContext(session=session)
        memory_block = self._recall_memories(user_input)  # 每次请求召回一次，进 system prompt

        for round_no in range(1, config.MAX_ROUNDS + 1):
            force_final = round_no == config.MAX_ROUNDS
            messages = self.build_messages(session, memory_block)

            # 超窗检查：压缩后重组装（压缩本身也可能把历史压短到不需要再压）
            if self._needs_compact(messages):
                async for ev in self._compact(session, usage_total):
                    yield ev
                messages = self.build_messages(session, memory_block)

            log.info("[session=%s] 第 %d/%d 轮（工具%s，上下文≈%d tokens）", session.id, round_no,
                     config.MAX_ROUNDS, "关闭，强制收敛" if force_final else "开启",
                     _messages_tokens(messages))
            yield {"type": "round_start", "round": round_no}

            try:
                stream = self.llm.stream_chat(messages,
                                              tools=None if force_final else self.tools.schemas())
                message: dict | None = None
                async for ev in stream:
                    if ev["type"] == "message_done":
                        message = ev
                    else:  # token / reasoning 增量，实时透传
                        yield ev
            except LLMError as e:
                log.error("[session=%s] LLM 调用失败: %s", session.id, e)
                yield {"type": "error", "message": f"调用大模型失败：{e}"}
                return

            assert message is not None
            content = message.get("content") or ""
            tool_calls = message.get("tool_calls") or []
            if message.get("usage"):
                for k in usage_total:
                    usage_total[k] += message["usage"].get(k, 0)
                # DeepSeek 缓存命中统计（自动前缀缓存，无需开关）
                hit = message["usage"].get("prompt_cache_hit_tokens", 0) or 0
                usage_total["prompt_cache_hit_tokens"] = (
                    usage_total.get("prompt_cache_hit_tokens", 0) + hit)

            if not content and not tool_calls:
                yield {"type": "error", "message": "模型返回了空内容，请重试"}
                return

            # ---- 分支 A：模型要求调用工具 -> 执行并写回历史，进入下一轮 ----
            if tool_calls and not force_final:
                for i, call in enumerate(tool_calls):
                    if not call.get("id"):
                        call["id"] = f"call_r{round_no}_{i}"
                session.history.append({
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": [
                        {"id": c["id"], "type": "function",
                         "function": {"name": c["name"], "arguments": c["arguments"]}}
                        for c in tool_calls
                    ],
                })
                for call in tool_calls:
                    async for ev in self._run_tool(session, call, ctx):
                        yield ev
                continue

            # ---- 分支 B：直接回答，loop 结束 ----
            session.history.append({"role": "assistant", "content": content})
            session.touch()
            log.info("[session=%s] 完成：共 %d 轮，最终回答 %d 字", session.id, round_no, len(content))
            self._maybe_distill_memory(session)  # 后台提炼记忆，不阻塞回答
            yield {"type": "final", "text": content, "rounds": round_no, "usage": usage_total}
            return

    # ---------- 工具执行（含解析、超时、异常兜底、超长落盘） ----------

    async def _run_tool(self, session, call: dict, ctx: ToolContext) -> AsyncIterator[dict]:
        name = call.get("name") or ""
        raw_args = call.get("arguments") or ""
        call_id = call.get("id") or ""

        # 参数解析（模型输出的是 JSON 字符串）
        try:
            args = json.loads(raw_args) if raw_args.strip() else {}
            if not isinstance(args, dict):
                raise ValueError("工具参数必须是 JSON 对象")
        except (json.JSONDecodeError, ValueError) as e:
            error_text = f"工具参数解析失败: {e}"
            log.warning("[session=%s] %s | 原始参数: %s", session.id, error_text, raw_args[:200])
            session.history.append({"role": "tool", "tool_call_id": call_id,
                                    "content": json.dumps({"error": error_text}, ensure_ascii=False)})
            yield {"type": "tool_end", "call_id": call_id, "name": name,
                   "ok": False, "result": error_text, "duration_ms": 0}
            return

        yield {"type": "tool_start", "call_id": call_id, "name": name, "arguments": args}
        log.info("[session=%s] 调用工具 %s(%s)", session.id, name, raw_args[:200])

        started = time.perf_counter()
        tool = self.tools.get(name)
        try:
            if tool is None:
                raise ValueError(f"未注册的工具: {name}")
            result = await asyncio.wait_for(tool.run(args, ctx), timeout=config.TOOL_TIMEOUT_S)
            duration_ms = int((time.perf_counter() - started) * 1000)
            result_json = json.dumps(result, ensure_ascii=False, default=str)
            log.info("[session=%s] 工具 %s 成功 (%dms): %s", session.id, name, duration_ms, result_json[:300])

            # 超长结果落盘：上下文里只留截断预览 + 完整结果地址
            if len(result_json) > config.TOOL_RESULT_INLINE_CHARS:
                rel_path = self.spill.save(session.id, call_id, name, args, result)
                inline = json.dumps({
                    "truncated": True,
                    "preview": result_json[:config.TOOL_RESULT_PREVIEW_CHARS] + " …",
                    "full_result": rel_path,
                    "orig_chars": len(result_json),
                }, ensure_ascii=False)
                log.info("[session=%s] 工具结果超长(%d字符)，已落盘 %s", session.id, len(result_json), rel_path)
                session.history.append({"role": "tool", "tool_call_id": call_id, "content": inline})
            else:
                session.history.append({"role": "tool", "tool_call_id": call_id, "content": result_json})

            yield {"type": "tool_end", "call_id": call_id, "name": name,
                   "ok": True, "result": result, "duration_ms": duration_ms}
        except asyncio.TimeoutError:
            async for ev in self._tool_failed(session, call_id, name, started,
                                              f"工具执行超时（>{config.TOOL_TIMEOUT_S}s）"):
                yield ev
        except Exception as e:  # 工具内部异常：把错误作为工具结果回给模型，让它自行恢复/告知用户
            async for ev in self._tool_failed(session, call_id, name, started,
                                              f"工具执行出错: {type(e).__name__}: {e}"):
                yield ev

    async def _tool_failed(self, session, call_id: str, name: str, started: float,
                           error_text: str) -> AsyncIterator[dict]:
        duration_ms = int((time.perf_counter() - started) * 1000)
        log.warning("[session=%s] 工具 %s 失败 (%dms): %s", session.id, name, duration_ms, error_text)
        session.history.append({"role": "tool", "tool_call_id": call_id,
                                "content": json.dumps({"error": error_text}, ensure_ascii=False)})
        yield {"type": "tool_end", "call_id": call_id, "name": name,
               "ok": False, "result": error_text, "duration_ms": duration_ms}
