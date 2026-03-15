# OpenMemory

**Cross-application long-term memory for AI — local, private, open source.**

OpenMemory is an [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server that gives every AI application access to a shared, persistent memory layer. Your preferences, knowledge, and context follow you across Claude, ChatGPT, Cursor, OpenClaw, and any MCP-compatible client — while your data never leaves your machine.

```
┌─ Claude ──────────┐
│ "user likes concise│──┐
│  responses"        │  │
└────────────────────┘  │
┌─ Cursor ──────────┐  │   ┌──────────────────┐
│ "user prefers      │──┼──▶│   OpenMemory     │  Your AI profile.
│  TypeScript + Vim" │  │   │   Local SQLite   │  All apps read/write.
└────────────────────┘  │   │   MCP Protocol   │  Data stays on disk.
┌─ OpenClaw ────────┐  │   └──────────────────┘
│ "daily news brief  │──┘
│  at 9am"           │
└────────────────────┘
```

## Why

Every AI app you use today is an amnesiac. Claude doesn't know what you told ChatGPT. Cursor doesn't know your Slack writing style. You repeat yourself across tools every single day.

OpenMemory fixes this by providing a **single, local memory store** that any AI can read from and write to via the open MCP standard.

## Features

- **Local & private** — SQLite database on your machine. No cloud, no telemetry.
- **MCP standard** — Works with any MCP-compatible client out of the box.
- **Semantic search** — Find memories by meaning, not just keywords. Zero-dependency local embeddings included.
- **Structured memory types** — Facts, preferences, episodes, and entities.
- **Memory decay** — Automatically reduces the importance of old, unused memories.
- **Zero config** — `pip install` and run. No API keys required.

## Quick Start

### Install

```bash
pip install -e .
```

### Run the MCP server

```bash
openmemory-server
# or
python -m openmemory.server
```

### Connect to Claude Desktop

Add to your Claude Desktop MCP config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "openmemory": {
      "command": "openmemory-server"
    }
  }
}
```

Then in Claude, you can say things like:

- *"Remember that I prefer Python over Java"*
- *"What do you know about my preferences?"*
- *"Forget everything about project X"*

## MCP Tools

| Tool | Description |
|------|-------------|
| `add_memory` | Store a new memory (fact / preference / episode / entity) |
| `search_memory` | Semantic similarity search across all memories |
| `get_memory` | Retrieve a specific memory by ID |
| `list_memories` | List memories with type/tag filtering and pagination |
| `update_memory` | Update content or importance of an existing memory |
| `delete_memory` | Delete a memory |
| `get_related` | Find memories semantically related to a given one |
| `memory_stats` | Get counts, averages, and breakdowns |

## Memory Types

| Type | Use case | Example |
|------|----------|---------|
| `fact` | Knowledge about the user | "Lives in Shanghai", "Works at Acme Corp" |
| `preference` | How the user likes things | "Prefers concise answers", "Uses dark mode" |
| `episode` | Notable events/interactions | "Discussed microservice migration on 2026-03-10" |
| `entity` | People, projects, concepts | "Project Alpha — the new billing system" |

## Use as a Python Library

```python
from openmemory.memory_store import MemoryStore

store = MemoryStore()  # defaults to ~/.openmemory/memories.db

# Add
mem = store.add_memory(
    type="fact",
    content="User prefers Python for backend development",
    tags=["programming", "preferences"],
    importance=0.8,
)

# Semantic search
results = store.search_memories("what languages does the user like?")
for r in results:
    print(r.content, r.importance)

# Decay old memories
stats = store.decay_memories()
print(f"Decayed: {stats['decayed']}, Deleted: {stats['deleted']}")
```

## Architecture

```
openmemory/
├── server.py          # MCP server — 8 tools exposed via FastMCP
├── memory_store.py    # Core engine — SQLite + semantic search
├── embeddings.py      # Pluggable embeddings (local n-gram default, optional OpenAI)
└── models.py          # Memory & MemoryType data models
```

- **Storage**: SQLite (default `~/.openmemory/memories.db`)
- **Embeddings**: Character n-gram hashing (256-dim, zero dependencies). Optionally swap in OpenAI embeddings for higher quality.
- **Search**: Cosine similarity with numpy acceleration (pure Python fallback).
- **Transport**: MCP stdio (compatible with Claude Desktop, OpenClaw, etc.)

## Optional Dependencies

```bash
pip install -e ".[embeddings]"  # numpy for faster similarity search
pip install -e ".[openai]"      # OpenAI embeddings support
```

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

64 tests covering models, storage engine, and MCP server tools.

## Contributors

[**Carlos**](https://github.com/carlos-life) — Creator & maintainer

## License

MIT

---

# OpenMemory

**AI 的跨应用长期记忆 —— 本地运行、隐私优先、完全开源。**

OpenMemory 是一个 [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) 服务器，为所有 AI 应用提供共享的持久化记忆层。你的偏好、知识和上下文可以在 Claude、ChatGPT、Cursor、OpenClaw 以及任何兼容 MCP 的客户端之间共享——数据始终保存在你自己的机器上。

## 为什么需要 OpenMemory

你现在用的每一个 AI 应用都是"失忆"的。Claude 不知道你跟 ChatGPT 说过什么，Cursor 不了解你在 Slack 上的写作风格。你每天都在不同工具之间重复自己说过的话。

OpenMemory 通过一个**本地记忆存储**解决这个问题，任何 AI 都可以通过开放的 MCP 标准来读写它。

## 核心特性

- **本地 & 隐私** —— 数据存在你机器上的 SQLite 里，没有云端，没有遥测。
- **MCP 标准协议** —— 开箱即用，兼容所有 MCP 客户端。
- **语义搜索** —— 按语义检索记忆，不只是关键词匹配。内置零依赖的本地嵌入。
- **结构化记忆** —— 事实（fact）、偏好（preference）、事件（episode）、实体（entity）四种类型。
- **记忆衰减** —— 自动降低长期未访问记忆的重要性。
- **零配置** —— `pip install` 后直接运行，不需要任何 API Key。

## 快速开始

### 安装

```bash
pip install -e .
```

### 启动 MCP 服务器

```bash
openmemory-server
# 或
python -m openmemory.server
```

### 接入 Claude Desktop

在 Claude Desktop 的 MCP 配置文件中添加：

```json
{
  "mcpServers": {
    "openmemory": {
      "command": "openmemory-server"
    }
  }
}
```

然后你就可以在 Claude 中这样说：

- *"记住我更喜欢用 Python 而不是 Java"*
- *"你知道我有什么偏好吗？"*
- *"忘掉所有关于 XX 项目的记忆"*

## MCP 工具列表

| 工具 | 说明 |
|------|------|
| `add_memory` | 存储一条新记忆（fact / preference / episode / entity）|
| `search_memory` | 语义相似度搜索 |
| `get_memory` | 根据 ID 获取记忆 |
| `list_memories` | 列出记忆，支持类型/标签过滤和分页 |
| `update_memory` | 更新记忆的内容或重要性 |
| `delete_memory` | 删除一条记忆 |
| `get_related` | 查找与某条记忆语义相关的其他记忆 |
| `memory_stats` | 获取统计信息 |

## 作为 Python 库使用

```python
from openmemory.memory_store import MemoryStore

store = MemoryStore()  # 默认存储在 ~/.openmemory/memories.db

# 添加记忆
mem = store.add_memory(
    type="fact",
    content="用户喜欢用 Python 做后端开发",
    tags=["编程", "偏好"],
    importance=0.8,
)

# 语义搜索
results = store.search_memories("用户喜欢什么编程语言？")
for r in results:
    print(r.content, r.importance)

# 衰减旧记忆
stats = store.decay_memories()
print(f"衰减: {stats['decayed']}, 删除: {stats['deleted']}")
```

## 架构

- **存储**：SQLite（默认路径 `~/.openmemory/memories.db`）
- **嵌入**：字符 n-gram 哈希（256 维，零依赖）。可选接入 OpenAI Embedding 获得更高质量。
- **搜索**：余弦相似度，有 numpy 自动加速，也支持纯 Python 回退。
- **传输**：MCP stdio 协议（兼容 Claude Desktop、OpenClaw 等）

## 测试

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

共 64 个测试，覆盖数据模型、存储引擎和 MCP 服务器工具。

## 贡献者

[**Carlos**](https://github.com/carlos-life) — 作者 & 维护者

## 许可证

MIT
