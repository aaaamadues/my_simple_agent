"""配置：优先读环境变量，其次读项目根目录的 .env（手写的极简解析，免依赖）。"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv()

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
# deepseek-chat 是 DeepSeek 当前最便宜的对话模型（V3 系列，非深度思考）
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

# ---- Agent Loop / context 管理参数 ----
MAX_ROUNDS = int(os.environ.get("MAX_ROUNDS", "16"))                     # 单次提问最多循环轮数（任务拆解需要更多轮）
TOOL_TIMEOUT_S = float(os.environ.get("TOOL_TIMEOUT_S", "15"))

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))

# ---- 上下文管理 ----
# 官方 /models 接口显示当前在售模型（deepseek-flash / deepseek-v4-pro）上下文均为 1048576；
# deepseek-chat 是旧别名，实际映射未知，按 128K 保守值触发压缩，可用环境变量覆盖。
CONTEXT_WINDOW_TOKENS = int(os.environ.get("CONTEXT_WINDOW_TOKENS", str(128 * 1024)))
CONTEXT_COMPACT_RATIO = float(os.environ.get("CONTEXT_COMPACT_RATIO", "0.9"))   # 超过窗口的该比例即压缩
CONTEXT_COMPACT_KEEP = int(os.environ.get("CONTEXT_COMPACT_KEEP", "6"))         # 压缩后保留最近几条消息
TOOL_RESULT_INLINE_CHARS = int(os.environ.get("TOOL_RESULT_INLINE_CHARS", "1200"))  # 工具结果超过则落盘
TOOL_RESULT_PREVIEW_CHARS = int(os.environ.get("TOOL_RESULT_PREVIEW_CHARS", "400"))  # 落盘后进上下文的预览长度

# ---- 长期记忆 ----
MEMORY_EVERY_N_TURNS = int(os.environ.get("MEMORY_EVERY_N_TURNS", "5"))  # 每 N 轮用户对话提炼一次
MEMORY_MAX_CHARS = int(os.environ.get("MEMORY_MAX_CHARS", "400"))        # 单条记忆长度上限
MEMORY_RECALL_K = int(os.environ.get("MEMORY_RECALL_K", "3"))            # 每次召回条数

# ---- 落盘目录 ----
DATA_DIR = ROOT / "data"
TOOL_SPILL_DIR = DATA_DIR / "tool_spill"   # 完整工具结果存放处
MEMORY_DIR = DATA_DIR / "memories"         # 长期记忆文本文件存放处
