"""任务规划工具：TodoManager + TodoItem 状态机，支持复杂任务的拆解与推进。

状态机：
    pending ──start──▶ solving ──complete──▶ complete
       ▲                  │
       └──── start 别的任务时自动退回 ──────┘

约束：同一时刻最多只有一个 solving（start 新任务会把旧的 solving 退回 pending）。
存储：序列化后存在 session.todos 里，每个会话独立一份。
"""
import time
from dataclasses import dataclass, field
from typing import Any

from .base import Tool, ToolContext

PENDING = "pending"
SOLVING = "solving"
COMPLETE = "complete"


@dataclass
class TodoItem:
    """一条子任务：标题 + 状态。"""
    id: int
    title: str
    status: str = PENDING
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    updated_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    def touch(self) -> None:
        self.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")

    def dump(self) -> dict:
        return {"id": self.id, "title": self.title, "status": self.status,
                "created_at": self.created_at, "updated_at": self.updated_at}


class TodoManager:
    """管理一个会话内的全部 TodoItem：增删改查 + 状态流转。"""

    def __init__(self) -> None:
        self._items: list[TodoItem] = []

    # ---- 与 session.todos（dict 列表）互转 ----

    @classmethod
    def load(cls, rows: list[dict]) -> "TodoManager":
        mgr = cls()
        for r in rows:
            mgr._items.append(TodoItem(id=int(r["id"]), title=str(r.get("title", "")),
                                       status=str(r.get("status", PENDING)),
                                       created_at=str(r.get("created_at", "")),
                                       updated_at=str(r.get("updated_at", ""))))
        return mgr

    def dump(self) -> list[dict]:
        return [t.dump() for t in self._items]

    # ---- 查询 ----

    def get(self, todo_id: int) -> TodoItem | None:
        return next((t for t in self._items if t.id == todo_id), None)

    def _get_or_raise(self, todo_id: int) -> TodoItem:
        item = self.get(todo_id)
        if item is None:
            raise ValueError(f"找不到 todo_id={todo_id}，可先用 list 动作查看现有任务")
        return item

    def solving(self) -> TodoItem | None:
        return next((t for t in self._items if t.status == SOLVING), None)

    def snapshot(self) -> dict:
        return {
            "total": len(self._items),
            "pending": sum(1 for t in self._items if t.status == PENDING),
            "solving": sum(1 for t in self._items if t.status == SOLVING),
            "complete": sum(1 for t in self._items if t.status == COMPLETE),
            "current": self.solving().dump() if self.solving() else None,
            "todos": self.dump(),
        }

    # ---- 写操作 ----

    def create(self, titles: list[str]) -> list[TodoItem]:
        """批量创建子任务（任务拆解入口）。"""
        cleaned = [str(t).strip() for t in titles if str(t).strip()]
        if not cleaned:
            raise ValueError("create 动作需要非空的 titles 数组")
        next_id = max((t.id for t in self._items), default=0) + 1
        created = [TodoItem(id=next_id + i, title=title) for i, title in enumerate(cleaned)]
        self._items.extend(created)
        return created

    def start(self, todo_id: int) -> tuple[TodoItem, int | None]:
        """把任务置为 solving；若有别的任务正在 solving，自动退回 pending。

        返回 (目标任务, 被退回的任务 id 或 None)。
        """
        item = self._get_or_raise(todo_id)
        if item.status == COMPLETE:
            raise ValueError(f"任务 {todo_id} 已完成，不能再次开始")
        current = self.solving()
        demoted_id = None
        if current is not None and current is not item:
            current.status = PENDING
            current.touch()
            demoted_id = current.id
        item.status = SOLVING
        item.touch()
        return item, demoted_id

    def complete(self, todo_id: int) -> TodoItem:
        """标记完成（pending/solving 均可直接完成；重复完成视为幂等）。"""
        item = self._get_or_raise(todo_id)
        item.status = COMPLETE
        item.touch()
        return item

    def update(self, todo_id: int, title: str) -> TodoItem:
        item = self._get_or_raise(todo_id)
        title = str(title).strip()
        if not title:
            raise ValueError("update 动作需要非空 title")
        item.title = title
        item.touch()
        return item

    def remove(self, todo_id: int) -> TodoItem:
        item = self._get_or_raise(todo_id)
        self._items.remove(item)
        return item

    def clear(self) -> None:
        self._items.clear()


class TodoTool(Tool):
    name = "todo"
    description = (
        "任务规划与进度管理（各会话独立）。用于两件事："
        "① 复杂任务拆解——先 create 一组子任务，再按 start→执行→complete 逐个推进；"
        "② 帮用户记待办——create 单条即可。"
        "状态流转：pending →start→ solving →complete→ complete；"
        "同一时刻只有一个任务能处于 solving，start 新任务会把原来的 solving 自动退回 pending。"
        "动作：create（批量新增，需 titles 数组）；start（开始执行，需 todo_id）；"
        "complete（标记完成，需 todo_id）；list（查看全部与进度）；"
        "update（改标题，需 todo_id+title）；remove（删除，需 todo_id）；clear（清空）。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string",
                       "enum": ["create", "start", "complete", "list", "update", "remove", "clear"],
                       "description": "要执行的操作"},
            "titles": {"type": "array", "items": {"type": "string"},
                       "description": "create 时的子任务标题列表（拆解复杂任务时一次给出全部子任务）"},
            "todo_id": {"type": "integer", "description": "start / complete / update / remove 的目标 id"},
            "title": {"type": "string", "description": "update 时的新标题"},
        },
        "required": ["action"],
    }

    @staticmethod
    def _int_arg(args: dict, key: str) -> int:
        if args.get(key) is None:
            raise ValueError(f"该动作需要 {key} 参数（整数）")
        try:
            return int(args[key])
        except (TypeError, ValueError):
            raise ValueError(f"{key} 必须是整数")

    async def run(self, args: dict, ctx: ToolContext) -> Any:
        mgr = TodoManager.load(ctx.session.todos)
        action = args.get("action")

        if action == "create":
            titles = args.get("titles")
            if isinstance(titles, str):  # 模型偶尔会传单个字符串，兜一下
                titles = [titles]
            created = mgr.create(titles or [])
            result = {"ok": True, "created": [t.dump() for t in created]}
        elif action == "start":
            item, demoted_id = mgr.start(self._int_arg(args, "todo_id"))
            result = {"ok": True, "started": item.dump()}
            if demoted_id is not None:
                result["note"] = f"任务 {demoted_id} 原处于 solving，已自动退回 pending"
        elif action == "complete":
            item = mgr.complete(self._int_arg(args, "todo_id"))
            result = {"ok": True, "completed": item.dump()}
        elif action == "list":
            result = {}
        elif action == "update":
            item = mgr.update(self._int_arg(args, "todo_id"), args.get("title", ""))
            result = {"ok": True, "updated": item.dump()}
        elif action == "remove":
            item = mgr.remove(self._int_arg(args, "todo_id"))
            result = {"ok": True, "removed": item.dump()}
        elif action == "clear":
            mgr.clear()
            result = {"ok": True, "cleared": True}
        else:
            raise ValueError(f"未知 action: {action}")

        ctx.session.todos = mgr.dump()  # 写回会话持久化
        return {**result, **mgr.snapshot()}
