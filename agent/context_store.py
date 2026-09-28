"""工具结果落盘仓库：超长工具结果写进文件，上下文里只留截断预览 + 存放地址。

目录结构：data/tool_spill/{session_id}/{seq}_{call_id}.json
文件内容：{"tool","call_id","arguments","result","created_at","chars"}
"""
import json
import re
import time
from pathlib import Path
from typing import Any


class ToolResultStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, session_id: str, call_id: str, name: str,
             arguments: dict, result: Any) -> str:
        """保存完整结果，返回相对地址（进上下文给模型/用户看的那种）。"""
        safe_sid = re.sub(r"[^\w-]", "_", session_id)[:32] or "session"
        safe_call = re.sub(r"[^\w-]", "_", call_id or "call")[:40]
        folder = self.root / safe_sid
        folder.mkdir(parents=True, exist_ok=True)
        seq = int(time.time() * 1000) % 10_000_000  # 避免同 call_id 覆盖
        path = folder / f"{seq}_{safe_call}.json"
        payload = {
            "tool": name,
            "call_id": call_id,
            "arguments": arguments,
            "result": result,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, default=str, indent=2),
                        encoding="utf-8")
        # 相对项目根的地址（root=data/tool_spill，root.parent.parent=项目根）
        return path.relative_to(self.root.parent.parent).as_posix()  # data/tool_spill/{sid}/{file}.json

    def load(self, rel_path: str) -> dict | None:
        """按相对地址读回完整结果（调试 / 后续工具取数用）。"""
        try:
            path = self.root.parent.parent / rel_path
            if not path.is_file():
                return None
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
