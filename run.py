"""启动入口：python run.py，然后浏览器打开 http://127.0.0.1:8000"""
import sys

# Windows 控制台默认 GBK，避免中文日志乱码
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import uvicorn

from agent.config import HOST, PORT

if __name__ == "__main__":
    uvicorn.run("agent.main:app", host=HOST, port=PORT, log_level="info")
