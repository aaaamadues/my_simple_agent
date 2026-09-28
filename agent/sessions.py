"""会话管理：一个浏览器窗口/标签 = 一个独立 session。

每个 session 各自维护 history（OpenAI 消息格式）与 todos（todo 工具的数据），
互相隔离、随时可继续。当前为内存态（重启丢失），后续可换持久化存储。
"""
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class Session:
    id: str
    title: str = "新会话"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    history: list[dict] = field(default_factory=list)  # role/content/tool_calls/tool_call_id 消息列表
    todos: list[dict] = field(default_factory=list)    # todo 工具的序列化数据 [{"id","title","status","created_at","updated_at"}]
    # ---- 上下文管理 / 记忆 ----
    summary: str = ""            # 上下文压缩后的结构化摘要（压缩后注入 system prompt）
    user_turns: int = 0          # 累计用户对话轮数（记忆提炼的计数基准）
    memory_checkpoint: int = 0   # 上次记忆提炼时的 user_turns

    def touch(self) -> None:
        self.updated_at = time.time()

    def summary(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": len(self.history),
            "todo_count": len(self.todos),
        }


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create(self, title: str = "新会话") -> Session:
        sid = uuid.uuid4().hex[:8]
        session = Session(id=sid, title=title)
        self._sessions[sid] = session
        return session

    def get(self, sid: str | None) -> Session | None:
        return self._sessions.get(sid or "")

    def get_or_create(self, sid: str | None) -> Session:
        return self.get(sid) or self.create()

    def delete(self, sid: str) -> bool:
        return self._sessions.pop(sid, None) is not None

    def list(self) -> list[dict]:
        return [s.summary() for s in
                sorted(self._sessions.values(), key=lambda s: s.updated_at, reverse=True)]
