"""天气工具（mock）：按「城市+日期」生成稳定的假数据，同一个城市同一天结果不变。"""
import datetime as dt
import hashlib
from typing import Any

from .base import Tool, ToolContext

_CONDITIONS = ["晴", "多云", "阴", "小雨", "雷阵雨", "雾"]


class WeatherTool(Tool):
    name = "weather"
    description = "查询指定城市今天或明天的天气（当前为 mock 数据，仅用于演示）。"
    parameters = {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名，如 北京"},
            "date": {"type": "string", "enum": ["today", "tomorrow"], "description": "今天或明天，默认 today"},
        },
        "required": ["city"],
    }

    async def run(self, args: dict, ctx: ToolContext) -> Any:
        city = str(args.get("city", "")).strip() or "北京"
        date = args.get("date", "today")
        day = dt.date.today() if date == "today" else dt.date.today() + dt.timedelta(days=1)

        seed = int(hashlib.md5(f"{city}-{day}".encode()).hexdigest()[:8], 16)
        cond = _CONDITIONS[seed % len(_CONDITIONS)]
        temp_lo = 12 + seed % 14
        temp_hi = temp_lo + 4 + seed % 6
        return {
            "city": city,
            "date": day.isoformat(),
            "weather": cond,
            "temp_low_c": temp_lo,
            "temp_high_c": temp_hi,
            "wind": f"{1 + seed % 6} 级",
            "note": "mock 天气数据，仅供演示",
        }
