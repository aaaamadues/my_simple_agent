"""FastAPI 入口：静态页面 + 会话 API + 流式对话 API（SSE）。"""
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from . import config
from .llm import DeepSeekClient
from .loop import AgentLoop
from .sessions import SessionManager
from .tools import registry
from .tracing import log

sessions = SessionManager()
llm = DeepSeekClient()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    await llm.aclose()


app = FastAPI(title="my_simple_agent", version="0.1.0", lifespan=lifespan)
agent = AgentLoop(llm=llm, tool_registry=registry, sessions=sessions)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


def sse(event: dict) -> str:
    """把一个事件序列化成一条 SSE 消息。"""
    return f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"


# ---------- 健康检查 / 会话 CRUD ----------

@app.get("/api/health")
async def health():
    return {"ok": True, "model": config.DEEPSEEK_MODEL,
            "api_key_configured": bool(config.DEEPSEEK_API_KEY),
            "tools": registry.names()}


@app.get("/api/sessions")
async def list_sessions():
    return {"sessions": sessions.list()}


@app.post("/api/sessions")
async def create_session():
    s = sessions.create()
    log.info("创建会话 %s", s.id)
    return s.summary()


@app.get("/api/sessions/{sid}")
async def get_session(sid: str):
    s = sessions.get(sid)
    if not s:
        raise HTTPException(404, "session 不存在（服务重启后内存态会话会丢失）")
    return {"id": s.id, "title": s.title, "history": s.history, "todos": s.todos,
            "summary": s.summary, "user_turns": s.user_turns}


@app.delete("/api/sessions/{sid}")
async def delete_session(sid: str):
    if not sessions.delete(sid):
        raise HTTPException(404, "session 不存在")
    return {"ok": True}


# ---------- 核心：流式对话 ----------

@app.post("/api/chat")
async def chat(req: ChatRequest):
    message = req.message.strip()
    if not message:
        raise HTTPException(400, "消息不能为空")

    if not config.DEEPSEEK_API_KEY:
        return StreamingResponse(
            iter([sse({"type": "error",
                       "message": "未配置 DEEPSEEK_API_KEY：请复制 .env.example 为 .env 并填入 Key，然后重启服务"})]),
            media_type="text/event-stream")

    session = sessions.get(req.session_id)
    is_new = session is None
    if is_new:
        session = sessions.create()

    async def event_stream():
        # 第一个事件：会话信息（新会话时前端据此拿到 session_id）
        yield sse({"type": "session", "session_id": session.id, "title": session.title, "created": is_new})
        try:
            async for ev in agent.run(session, message):
                yield sse(ev)
        except Exception as e:  # 兜底：未预期异常也以 error 事件收尾，前端不会卡在加载态
            log.exception("[session=%s] 未预期异常", session.id)
            yield sse({"type": "error", "message": f"服务器内部错误: {e}"})
        yield sse({"type": "end"})  # 流结束信号

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")
