# FastA2A: Fast Agent-to-Agent Communication Framework

一个轻量级、易用的 Agent-to-Agent 通信框架，基于 FastAPI 实现。

## 核心特性

- **极简 API**: 一个装饰器即可将任意 Agent 暴露为 A2A 服务
- **标准协议**: 定义清晰的通信协议（Manifest, Request, Response）
- **会话管理**: 自动维护会话上下文
- **工具循环**: 内置工具调用循环支持
- **类型安全**: 完整的 Pydantic 模型和类型注解

## 快速开始

### 1. 定义 Agent（服务端）

```python
from fasta2a import FastA2A, ToolDefinition

@FastA2A(
    name="MathAgent",
    description="数学计算助手",
    tools=[
        ToolDefinition(
            name="calculator",
            description="执行数学运算",
            parameters={...}
        )
    ]
)
def my_agent(query: str, state: dict, tools_result: list, context: dict) -> dict:
    # Agent 逻辑
    return {
        "messages": [...],
        "tool_calls": [...],
        "finished": True
    }

# 启动服务
my_agent.run(port=8000)
```

### 2. 调用 Agent（客户端）

```python
from fasta2a import A2AClient

# 连接到 Agent
client = A2AClient("http://localhost:8000")

# 调用（自动处理工具循环）
response = client.invoke_with_tools(
    query="计算 123 + 456",
    tool_executor=my_tool_executor
)

print(response.messages)
```

## 架构设计

```
┌─────────────┐                    ┌─────────────┐
│   Client    │◄──── A2A Protocol ─────►   Server   │
│             │                    │             │
│ A2AClient   │                    │  FastA2A    │
│             │                    │             │
│ - invoke()  │                    │ - /manifest │
│ - invoke_   │                    │ - /invoke   │
│   with_tools│                    │ - /reset    │
└─────────────┘                    └─────────────┘
```

## 协议模型

### Manifest（能力清单）
```python
{
    "name": "AgentName",
    "version": "1.0.0",
    "description": "Agent 描述",
    "tools": [...]
}
```

### A2ARequest（请求）
```python
{
    "session_id": "uuid",
    "query": "用户查询",
    "tools_result": [...],  # 工具执行结果
    "context": {...}
}
```

### A2AResponse（响应）
```python
{
    "session_id": "uuid",
    "messages": [...],      # 对话消息
    "tool_calls": [...],    # 需要执行的工具
    "finished": true,       # 是否完成
    "meta": {...}
}
```

## 运行示例

```bash
# 终端 1: 启动服务器
cd /Users/nako/Documents/notebook/LangChain笔记/code/fasta2a
python example_usage.py server

# 终端 2: 运行客户端
python example_usage.py
```

## 依赖

```
fastapi
uvicorn
pydantic
requests
```

## 文件结构

```
fasta2a/
├── __init__.py          # 模块导出
├── core.py              # 协议核心模型
├── server.py            # 服务端封装
├── client.py            # 客户端实现
├── example_usage.py     # 使用示例
└── README.md            # 本文档
```

## License

MIT License
