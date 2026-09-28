# my_simple_agent

从零手搓的最小可用 Agent（**不依赖 langgraph / openhands 等任何 agent 框架**）：
FastAPI + 裸 HTTP 调 DeepSeek（OpenAI 兼容接口），自带 Agent Loop、工具注册、
上下文管理（超长结果落盘 + 超窗摘要压缩）、长期记忆（子 LLM 提炼 + BM25 召回）、
原生单文件前端（流式打字机 + 工具卡片 + todo 任务面板）。

---

## 一、运行方式

```bash
# 0. 环境：Python 3.10+
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 Key（https://platform.deepseek.com 申请），复制模板并填入
copy .env.example .env        # Windows；Linux/macOS 用 cp

# 3. 启动
python run.py

# 4. 浏览器打开
#    http://127.0.0.1:8000
```

**跑集成测试**（10 个问题，打真实 API，覆盖 calculator / weather / search / todo / 记忆落盘与召回；
服务没起会自动拉起、测完自动关掉）：

```bash
python test.py
```

运行期产生的数据（可直接删除，删了就是干净的初始状态）：

| 目录/文件 | 内容 |
|---|---|
| `data/memories/*.txt` | 长期记忆（每 5 轮用户对话由子 LLM 提炼一条） |
| `data/tool_spill/{session}/*.json` | 超长工具结果的完整存档 |
| `logs/agent.log` | 运行 trace（每轮、每次工具调用、记忆召回/落盘都有记录） |

---

## 二、系统设计

```
┌──────────────┐   SSE(流式)    ┌───────────────────────────────────┐
│  前端单页面   │ ◄──────────── │  FastAPI  (agent/main.py)          │
│ static/      │ ──────────►   │  POST /api/chat   流式对话          │
└──────────────┘  POST JSON    │  GET/POST/DELETE /api/sessions 会话 │
                               └──────────────┬──────────────────────┘
                                              │
              ┌───────────────────────────────▼───────────────────────────────┐
              │ AgentLoop (agent/loop.py) ★核心                                  │
              │  每次用户请求：                                                  │
              │   1. BM25 召回长期记忆 → 拼进 system prompt                      │
              │   2. for round in 1..MAX_ROUNDS:                                │
              │        a. 超窗检查：估算 tokens > 窗口×0.75 → 子 LLM 压缩摘要，  │
              │           全量替换历史（只留最近几条）                            │
              │        b. build_messages：system(+摘要+记忆) + 完整历史          │
              │        c. LLM 流式调用；有 tool_calls → 执行工具 → 写回历史 →    │
              │           下一轮；纯文本 → final 结束                            │
              │        最后一轮 tools=None 强制收敛                               │
              │   3. 结束后：每逢 5 轮用户对话 → 后台子 LLM 提炼一条记忆落盘      │
              └───────┬────────────────┬───────────────────┬───────────────────┘
                      │                │                   │
          ┌───────────▼───┐   ┌────────▼────────┐   ┌──────▼──────────────┐
          │ DeepSeekClient │   │ ToolRegistry    │   │ ToolResultStore /   │
          │ agent/llm.py   │   │ calculator      │   │ MemoryStore(BM25)   │
          │ 流式+非流式     │   │ search(DDG真搜) │   │ agent/context_store │
          │ 自动前缀缓存    │   │ weather(mock)   │   │ agent/memory.py     │
          └────────────────┘   │ todo(状态机)    │   └─────────────────────┘
                               └─────────────────┘
```

| 文件 | 职责 |
|---|---|
| `run.py` | 启动入口（uvicorn） |
| `test.py` | 10 问集成测试（tools / todo / weather / search / 记忆） |
| `agent/main.py` | FastAPI 路由：SSE 流式对话、会话 CRUD、静态页 |
| `agent/loop.py` | **Agent 主循环 + 上下文压缩 + 记忆触发**（核心） |
| `agent/llm.py` | DeepSeek 客户端：`stream_chat` 流式 / `complete` 子任务一次性调用 |
| `agent/prompts.py` | System Prompt + 上下文压缩提示词 + 记忆提炼提示词 |
| `agent/tools/base.py` | 工具抽象与注册中心（名称+描述+参数 Schema+run） |
| `agent/tools/*.py` | calculator（ast 白名单求值）/ search（DuckDuckGo）/ weather（mock）/ todo（状态机） |
| `agent/context_store.py` | 超长工具结果落盘仓库 |
| `agent/memory.py` | BM25（纯手搓）+ 记忆文件仓库 |
| `agent/sessions.py` | 会话管理（内存态，history/todos/summary/user_turns） |
| `agent/config.py` | 配置（.env / 环境变量） |

### 2.1 Agent Loop

不写任何规则，LLM 基于**工具 Schema** 自主决策（原生 function calling）。一次提问内最多
`MAX_ROUNDS` 轮"LLM 决策 → 工具执行 → 结果回填 → 再决策"，最后一轮不再提供工具，强制收敛。
工具错误不中断循环：错误文本作为 tool 结果回填，让模型自行换参数重试或如实告知用户。

### 2.2 工具系统

每个工具 = `Tool` 子类（名称、描述、参数 JSON Schema、`run(args, ctx)`），在
`tools/__init__.py` 注册一行即可接入，主循环零改动。`todo` 是带状态机的任务管理器：
`pending → solving → complete`，同一时刻仅一个 solving（start 新任务会自动把旧的退回 pending）。

### 2.3 上下文管理（两层）

1. **超长结果落盘**：工具结果 > `TOOL_RESULT_INLINE_CHARS`(1200) 字符时，完整结果写入
   `data/tool_spill/`，进上下文的只有 400 字符预览 + `full_result` 存放地址（摘要提示词
   要求模型在压缩时保留这些地址）；
2. **超窗摘要压缩**：每轮请求前按启发式估算 token（中文 ≈0.6 字/token、英文 ≈4 字符/token），
   超过 `CONTEXT_WINDOW_TOKENS × 0.9`（默认按 128K 保守估，官方当前在售模型实际为 1M，可环境变量
   覆盖）时，fork 子 LLM 把全部历史压成结构化摘要——**目标 / 已完成 / 未完成 / 重要推理步骤 /
   关键事实与数据 / 工具结果存档 / 用户偏好**——存入 `session.summary`，历史只保留最近 6 条
   （自动跳过孤儿 tool 消息）。子 LLM 失败时降级为硬截断，不阻断对话。

**为什么不做滑动窗口 / 分级截断**：历史一旦超过滑窗上限，每次请求都要从头部丢消息、改写旧的
工具结果——消息前缀变了，DeepSeek 的服务端前缀缓存就全 miss。因此历史保持 **append-only、
写入后永不改写**，每次请求的消息列表恰是上一次的前缀扩展，缓存自然命中；超长内容在写入时
一次性落盘截断，超窗统一交给摘要压缩处理。

### 2.4 Prompt Cache

DeepSeek 的前缀缓存是**服务端自动**的（无请求参数可开关），本项目做的两件事：
消息组装保持"静态 system prompt 在前"的前缀稳定结构；聚合透出 usage 里的
`prompt_cache_hit_tokens`（前端完成状态栏可见，实测连续对话命中）。

---

## 三、Memory：召回时机与放置方式

### 3.1 写入（什么时候产生记忆）

- **触发时机**：每个 session 内部计数 `user_turns`，每当累计用户对话轮数距离上次提炼
  ≥ `MEMORY_EVERY_N_TURNS`(5) 轮，就在该轮回答产出 `final` 之后触发；
- **执行方式**：`asyncio.create_task` **后台 fork 子 LLM**（不阻塞、不拖慢前端拿到回答），
  输入 = 会话压缩摘要 + 近期对话，提示词约束：只提炼**跨对话仍然成立**的信息（身份/偏好/
  反复需求/稳定事实），单条 ≤ 400 字，第三人称；
- **持久化**：一条记忆一个纯文本文件 `data/memories/{时间戳}_{session}.txt`，
  首行元数据（session/时间），正文为记忆本身。

### 3.2 召回（什么时候查、怎么查）

- **触发时机**：**每次用户发出请求时**（`AgentLoop.run()` 入口处），用用户的原始输入作查询，
  在**全部 session 的全部记忆**里检索——记忆天然是跨会话共享的；
- **检索方式**：BM25（Okapi，k1=1.5，b=0.75，纯手搓无第三方依赖）。中文同时按
  **单字 + 2-gram** 切分（只用 2-gram 会有词表不匹配问题：「猫叫什么名字」与「名叫雪球」
  无共同 2-gram；补单字后靠 IDF 自动压低"的/了"这类常见字、抬高"猫"这类稀有字），
  英文/数字按词切分；只返回得分 > 0 的 top-`MEMORY_RECALL_K`(3) 条。

### 3.3 放置（放哪、长什么样）

召回的记忆拼接成文本块，追加在 **system prompt 末尾**（与压缩摘要一样位于静态基座之后）：

```
你是一个乐于助人的中文助手 Agent ……（静态基座 prompt）

【此前对话的压缩摘要（上下文超窗时由子 LLM 生成）】   ← 仅压缩发生过时存在
目标：…… 未完成的事项：……

【长期记忆（BM25 召回，可能相关）】                    ← 每次请求按当前输入召回
- [会话 317bea9c · 2026-09-28 22:31] 用户养了一只英国短毛猫，名叫雪球……
```

模型把它当背景知识用——新会话里问"我的猫叫什么名字"，会基于召回记忆直接回答。

**取舍说明**：把动态记忆放进 system prompt 会改变请求前缀、削弱 DeepSeek 的前缀缓存命中；
换来的是指令权重最高（模型最重视）。若更看重缓存/成本，可把该块改为 system 之后的一条
独立 user 消息，一行改动即可（`loop.build_messages`）。

---

## 四、配置项

| 变量 | 默认 | 说明 |
|---|---|---|
| `DEEPSEEK_API_KEY` | — | **必填** |
| `DEEPSEEK_MODEL` | deepseek-chat | 模型（旧别名；官方当前为 deepseek-flash / deepseek-v4-pro，均 1M 窗口） |
| `MAX_ROUNDS` | 16 | 单次提问最大循环轮数 |
| `TOOL_TIMEOUT_S` | 15 | 单工具执行超时 |
| `CONTEXT_WINDOW_TOKENS` | 131072 | 压缩触发的窗口基准（保守值，可改 1048576） |
| `CONTEXT_COMPACT_RATIO` | 0.9 | 超过窗口该比例即触发摘要压缩 |
| `CONTEXT_COMPACT_KEEP` | 6 | 压缩后保留最近几条消息 |
| `TOOL_RESULT_INLINE_CHARS` | 1200 | 工具结果超过即落盘 |
| `TOOL_RESULT_PREVIEW_CHARS` | 400 | 落盘后进上下文的预览长度 |
| `MEMORY_EVERY_N_TURNS` | 5 | 每 N 轮用户对话提炼一次记忆 |
| `MEMORY_MAX_CHARS` | 400 | 单条记忆长度上限 |
| `MEMORY_RECALL_K` | 3 | 每次召回条数 |
| `HOST` / `PORT` | 127.0.0.1 / 8000 | 监听地址 |
