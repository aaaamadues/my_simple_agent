"""my_simple_agent 集成测试：10 个问题，覆盖 calculator / weather / search / todo / 记忆落盘与召回。

用法：
    python run.py      # 先启动服务（或让本脚本自动拉起）
    python test.py     # 跑测试

说明：
- 直接打真实 API（DeepSeek + DuckDuckGo），会产生少量 token 消耗；
- 测试 9 会在 data/memories/ 真实落盘一条「猫叫雪球」的记忆，之后新建会话问猫名应能召回；
  想清掉这些测试数据：删除 data/memories/*.txt 即可。
"""
import atexit
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

BASE_URL = os.environ.get("MY_SIMPLE_AGENT_URL", "http://127.0.0.1:8000")
MEMORY_DIR = Path(__file__).resolve().parent / "data" / "memories"

# Windows 控制台默认 GBK，避免中文/符号乱码
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

_spawned = None


def ensure_server() -> None:
    """服务没起就自动拉起一个（测试结束自动关掉）。"""
    global _spawned
    try:
        r = httpx.get(f"{BASE_URL}/api/health", timeout=3)
        if r.status_code == 200:
            return
    except httpx.HTTPError:
        pass
    print(f"⟳ 服务未启动，自动拉起：python run.py（{BASE_URL}）")
    _spawned = subprocess.Popen([sys.executable, "run.py"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(20):
        time.sleep(1)
        try:
            if httpx.get(f"{BASE_URL}/api/health", timeout=3).status_code == 200:
                return
        except httpx.HTTPError:
            continue
    raise SystemExit("服务拉起失败，请手动运行 python run.py 后重试")


@atexit.register
def _shutdown() -> None:
    if _spawned is not None:
        _spawned.terminate()


def chat(session_id: str | None, message: str) -> dict:
    """发一条消息，收完整 SSE 事件流，返回 {session_id, final, tools, events, error}。"""
    with httpx.Client(timeout=180) as client:
        with client.stream("POST", f"{BASE_URL}/api/chat",
                           json={"session_id": session_id, "message": message}) as resp:
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.read()[:200]}")
            events, buf = [], ""
            for chunk in resp.iter_text():
                buf += chunk
                while "\n\n" in buf:
                    raw, buf = buf.split("\n\n", 1)
                    for line in raw.split("\n"):
                        if line.startswith("data:"):
                            try:
                                events.append(json.loads(line[5:]))
                            except json.JSONDecodeError:
                                pass
    final = next((e for e in events if e["type"] == "final"), None)
    error = next((e for e in events if e["type"] == "error"), None)
    sid = next((e["session_id"] for e in events if e["type"] == "session"), session_id)
    # arguments 只在 tool_start 事件里，按 call_id 合并进 tool_end
    args_by_call = {e["call_id"]: e.get("arguments")
                    for e in events if e["type"] == "tool_start"}
    tools = [{"name": e["name"], "ok": e["ok"], "args": args_by_call.get(e["call_id"]),
              "result": e.get("result")} for e in events if e["type"] == "tool_end"]
    return {"session_id": sid, "final": (final or {}).get("text", ""),
            "tools": tools, "events": events, "error": (error or {}).get("message")}


def tool_ok(r: dict, name: str) -> bool:
    return any(t["name"] == name and t["ok"] for t in r["tools"])


def mem_files() -> set[str]:
    return {str(p) for p in MEMORY_DIR.glob("*.txt")} if MEMORY_DIR.exists() else set()


# ---------------------------- 10 个测试问题 ----------------------------

RESULTS: list[tuple[int, str, bool, str]] = []


def run_case(no: int, title: str, fn) -> None:
    print(f"\n[{no:>2}/10] {title}")
    try:
        detail = fn()
        ok, note = True, detail or "通过"
    except AssertionError as e:
        ok, note = False, str(e)
    except Exception as e:  # noqa: BLE001 - 网络等意外也记为失败
        ok, note = False, f"异常: {type(e).__name__}: {e}"
    RESULTS.append((no, title, ok, note))
    print(f"       {'✅' if ok else '❌'} {note}")


def case_calculator():
    r = chat(None, "帮我算一下 (123+456)*7 等于多少")
    assert tool_ok(r, "calculator"), f"calculator 未成功调用: {r['tools']} {r['error']}"
    assert "4053" in r["final"], f"回答未包含 4053: {r['final'][:100]}"
    return f"calculator 成功，回答含 4053（{len(r['tools'])} 次工具调用）"


def case_weather_today():
    r = chat(None, "北京今天天气怎么样？")
    assert tool_ok(r, "weather"), f"weather 未成功调用: {r['tools']} {r['error']}"
    assert "北京" in r["final"], f"回答未提到北京: {r['final'][:100]}"
    return f"weather 成功，回答: {r['final'][:60]}…"


def case_weather_tomorrow():
    r = chat(None, "查一下上海明天的天气")
    t = next((t for t in r["tools"] if t["name"] == "weather"), None)
    assert t and t["ok"], f"weather 未成功调用: {r['tools']} {r['error']}"
    args = t["args"] or {}
    assert "上海" in str(args), f"参数未传上海: {args}"
    assert args.get("date") == "tomorrow", f"date 参数不是 tomorrow: {args}"
    return f"weather(上海, tomorrow) 参数正确，结果: {str(t['result'])[:60]}…"


def case_search():
    r = chat(None, "搜索一下 DeepSeek API 最新的定价是多少")
    t = next((t for t in r["tools"] if t["name"] == "search"), None)
    assert t and t["ok"], f"search 未成功调用: {r['tools']} {r['error']}"
    results = (t["result"] or {}).get("results") or []
    assert results, f"搜索结果为空: {t['result']}"
    return f"DuckDuckGo 返回 {len(results)} 条，首条: {results[0].get('title', '')[:50]}…"


def make_case_todo_create(sess_a: dict):
    def case():
        r = chat(sess_a["id"], "帮我记一个待办：明天早上 9 点开站会")
        t = next((t for t in r["tools"] if t["name"] == "todo"), None)
        assert t and t["ok"], f"todo 未成功调用: {r['tools']} {r['error']}"
        assert "站会" in json.dumps(t["result"], ensure_ascii=False), f"待办内容未落库: {t['result']}"
        return f"todo create 成功（会话 A），待办数: {(t['result'] or {}).get('total')}"
    return case


def make_case_todo_list(sess_a: dict):
    def case():
        r = chat(sess_a["id"], "我现在有哪些待办？")
        assert "站会" in r["final"], f"回答未包含站会: {r['final'][:150]}"
        called = tool_ok(r, "todo")
        return f"跨轮状态正常（todo {'调用' if called else '未调用，直接凭历史回答'}），回答含「站会」"
    return case


def case_combo():
    r = chat(None, "查一下深圳今天的天气，然后记一个待办：提醒我出门前看天气")
    assert tool_ok(r, "weather"), f"weather 未调用: {r['tools']}"
    assert tool_ok(r, "todo"), f"todo 未调用: {r['tools']}"
    return f"weather + todo 组合调用成功（共 {len(r['tools'])} 次）"


def make_case_todo_complete(sess_a: dict):
    def case():
        r = chat(sess_a["id"], "把我目前唯一的待办标记为完成")
        todos = [t for t in r["tools"] if t["name"] == "todo" and t["ok"]]
        assert todos, f"todo 未成功调用: {r['tools']} {r['error']}"
        # 模型可能按状态机习惯先 start 再 complete，只要最终出现 complete 且完成数 ≥1 即通过
        actions = [(t["args"] or {}).get("action") for t in todos]
        comp = next((t for t in todos if (t["args"] or {}).get("action") == "complete"), None)
        assert comp, f"未见 complete 动作（实际动作序列: {actions}），回答: {r['final'][:80]}"
        res = comp["result"] or {}
        assert res.get("complete", 0) >= 1, f"完成数异常: {res}"
        return f"状态流转成功，动作序列 {actions}，进度 {res.get('complete')}/{res.get('total')}"
    return case


def case_memory_write():
    before = mem_files()
    sid = None
    msgs = ["记住一个信息：我的猫叫雪球，是一只英国短毛猫。",
            "雪球今年 3 岁了。",
            "我平时主要用 Python 写代码。",
            "雪球最喜欢的玩具是毛线球。",
            "今天就聊到这，谢谢！"]
    for i, m in enumerate(msgs, 1):
        r = chat(sid, m)
        sid = r["session_id"]
        assert not r["error"], f"第 {i} 条消息出错: {r['error']}"
    # 记忆提炼是回答之后的后台任务，轮询等文件出现
    for _ in range(25):
        time.sleep(1)
        new = mem_files() - before
        if new:
            body = Path(sorted(new)[-1]).read_text(encoding="utf-8")
            print(f"       📝 新记忆文件: {Path(sorted(new)[-1]).name}")
            print(f"       {body.strip().splitlines()[-1][:100]}")
            return "5 轮对话后记忆已落盘"
    raise AssertionError("等待 25s 未见新记忆文件（提炼任务未执行或超时）")


def case_memory_recall():
    r = chat(None, "我的猫叫什么名字？")
    assert "雪球" in r["final"], f"新会话未能召回记忆: {r['final'][:150]}"
    return f"跨会话召回成功，回答: {r['final'][:80]}…"


def main() -> None:
    ensure_server()
    health = httpx.get(f"{BASE_URL}/api/health", timeout=5).json()
    print(f"my_simple_agent 测试开始 · model={health['model']} · tools={health['tools']}")
    print(f"基线记忆文件数: {len(mem_files())}")

    # 会话 A：todo 创建/查询/完成 三个用例共用，验证跨轮状态
    r = chat(None, "你好")  # 预热一个会话 A（也顺带验证普通对话不需要工具）
    sess_a = {"id": r["session_id"]}

    run_case(1, "calculator：(123+456)*7", case_calculator)
    run_case(2, "weather：北京今天", case_weather_today)
    run_case(3, "weather：上海明天（参数校验）", case_weather_tomorrow)
    run_case(4, "search：DuckDuckGo 真实搜索", case_search)
    run_case(5, "todo：create 单条待办（会话 A）", make_case_todo_create(sess_a))
    run_case(6, "todo：跨轮查询待办（会话 A）", make_case_todo_list(sess_a))
    run_case(7, "weather + todo 组合调用", case_combo)
    run_case(8, "todo：complete 状态流转（会话 A）", make_case_todo_complete(sess_a))
    run_case(9, "记忆：5 轮对话触发提炼落盘", case_memory_write)
    run_case(10, "记忆：新会话 BM25 召回", case_memory_recall)

    passed = sum(1 for _, _, ok, _ in RESULTS if ok)
    print("\n" + "=" * 56)
    print(f"结果：{passed}/{len(RESULTS)} 通过")
    for no, title, ok, note in RESULTS:
        print(f"  {'✅' if ok else '❌'} [{no}] {title} — {note}")
    sys.exit(0 if passed == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
