# 第九章：LangChain 1.0 新特性与更新

> 本章专门介绍 LangChain 1.0 版本的重要更新和新增特性

---

## 9.1 LangChain 1.0 里程碑

### 9.1.1 版本承诺与稳定性

LangChain 1.0 是首个主要稳定版本，标志着框架的成熟：

```python
# 版本检查
import langchain
print(f"LangChain 版本: {langchain.__version__}")  # 1.0.5+

# Python 版本要求
# Python >= 3.10, < 4.0
```

**核心承诺**：
- ✅ 不会有破坏性更改直到 2.0
- ✅ 生产级稳定性保证
- ✅ 长期支持 (LTS)
- ✅ 向后兼容性维护

### 9.1.2 架构简化与模块重组

```mermaid
graph LR
    A[LangChain 0.x<br>单体包] --> B[LangChain 1.0<br>模块化]
    B --> C[langchain-core<br>核心抽象]
    B --> D[langchain<br>精简包]
    B --> E[langchain-community<br>社区集成]
    B --> F[langchain-classic<br>遗留功能]

    style A fill:#FFE4E1
    style B fill:#E3F2FD
    style C fill:#C8E6C9
    style D fill:#FFF9C4
```

**包结构变化**：
```python
# 旧版本（0.x）
from langchain.llms import OpenAI  # ❌ 已弃用

# 新版本（1.0+）
from langchain_openai import ChatOpenAI  # ✅ 推荐
from langchain_core.messages import HumanMessage  # ✅ 核心抽象
from langchain_community.tools import DuckDuckGoSearchRun  # ✅ 社区工具
```

---

## 9.2 Model Context Protocol (MCP) 集成

### 9.2.1 MCP 概述

MCP 是 LangChain 1.0 引入的标准化协议，用于连接 LLM 和数据源：

```python
from langchain_mcp_adapters import MCPAdapter
from mcp import Server

# 创建 MCP 服务器
server = Server(
    name="my_data_server",
    version="1.0.0"
)

# 定义资源
@server.resource("database://users")
async def get_users():
    """暴露用户数据库作为 MCP 资源"""
    return {"users": [...]}

# 在 LangChain 中使用
adapter = MCPAdapter(server)
tools = adapter.get_tools()

# 创建带 MCP 工具的 Agent
agent = create_agent(
    model=ChatOpenAI(),
    tools=tools,
    system_prompt="You can access user database via MCP"
)
```

### 9.2.2 MCP 优势

**标准化优势**：
- 🔌 统一的数据源接口
- 🔄 可重用的集成
- 🛡️ 内置安全控制
- 📊 自动监控和追踪

**实战示例：连接多个数据源**

```python
from langchain_mcp_adapters import MCPHub

# MCP Hub 管理多个数据源
hub = MCPHub()

# 添加不同类型的数据源
hub.add_server("postgres", PostgresMCPServer(conn_string))
hub.add_server("elasticsearch", ElasticMCPServer(hosts))
hub.add_server("salesforce", SalesforceMCPServer(credentials))

# 统一访问所有数据源
unified_tools = hub.get_all_tools()

# 创建能访问所有数据源的 Agent
agent = create_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=unified_tools,
    middleware=[
        PIIMiddleware(),  # 自动保护敏感数据
        ModelCallLimitMiddleware(max_calls=10)
    ]
)
```

---

## 9.3 流式输出与 JSON Mode

### 9.3.1 增强的流式处理

LangChain 1.0 显著改进了流式输出能力：

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from typing import AsyncIterator

# 创建支持流式 JSON 的链
model = ChatOpenAI(model="gpt-4o", temperature=0)
parser = JsonOutputParser()

chain = model | parser

# 同步流式
for chunk in chain.stream({"messages": [HumanMessage("生成用户信息")]}):
    print(chunk, end="", flush=True)  # 实时输出 JSON 片段

# 异步流式（性能更好）
async def stream_json():
    async for chunk in chain.astream({"messages": [...]}):
        # 处理流式 JSON 块
        if isinstance(chunk, dict):
            yield chunk
```

### 9.3.2 结构化输出（JSON Mode）

```python
from pydantic import BaseModel, Field
from typing import List

# 定义输出模式
class UserProfile(BaseModel):
    name: str = Field(description="用户姓名")
    age: int = Field(description="用户年龄")
    interests: List[str] = Field(description="兴趣列表")

# 方式1：使用 response_format
agent = create_agent(
    model=ChatOpenAI(model="gpt-4o"),
    response_format=UserProfile,  # 强制结构化输出
)

# 方式2：使用 with_structured_output
model = ChatOpenAI(model="gpt-4o")
structured_model = model.with_structured_output(UserProfile)

result = structured_model.invoke("生成一个用户档案")
print(result)  # UserProfile 对象
```

### 9.3.3 流式 JSON 解析

```python
from langchain_core.output_parsers import SimpleJsonOutputParser

# 创建支持部分 JSON 解析的 parser
parser = SimpleJsonOutputParser(streaming=True)

async def process_streaming_json():
    """处理流式 JSON，即使不完整也能解析"""
    partial_json = ""

    async for chunk in model.astream(...):
        partial_json += chunk

        # 尝试解析部分 JSON
        try:
            parsed = parser.parse_partial(partial_json)
            if parsed:
                yield parsed  # 输出已解析的部分
        except:
            continue  # 继续累积
```

---

## 9.4 工具并行调用优化

### 9.4.1 自动并行识别

LangChain 1.0 的 LLM 能自动识别可并行调用的工具：

```python
from langchain.tools import tool
import asyncio

@tool
async def search_web(query: str) -> str:
    """搜索网络信息"""
    await asyncio.sleep(1)  # 模拟网络延迟
    return f"Results for {query}"

@tool
async def query_database(sql: str) -> str:
    """查询数据库"""
    await asyncio.sleep(1)  # 模拟查询延迟
    return f"Data from {sql}"

@tool
async def call_api(endpoint: str) -> str:
    """调用外部 API"""
    await asyncio.sleep(1)  # 模拟 API 延迟
    return f"Response from {endpoint}"

# Agent 会自动并行调用多个工具
agent = create_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=[search_web, query_database, call_api]
)

# 单个请求触发多个并行工具调用
response = await agent.ainvoke({
    "messages": [HumanMessage(
        "搜索最新的AI新闻，同时查询用户表中的活跃用户数，"
        "并调用weather API获取北京天气"
    )]
})

# LLM 自动生成并行 tool_calls：
# [
#   {"tool": "search_web", "args": {"query": "latest AI news"}},
#   {"tool": "query_database", "args": {"sql": "SELECT COUNT(*) FROM users WHERE active=true"}},
#   {"tool": "call_api", "args": {"endpoint": "/weather/beijing"}}
# ]
# 三个工具同时执行，总时间约 1 秒而非 3 秒
```

### 9.4.2 并行执行控制

```python
from langchain.agents.middleware import ToolConcurrencyMiddleware

# 控制并行度
middleware = [
    ToolConcurrencyMiddleware(
        max_concurrent=5,  # 最多同时执行 5 个工具
        timeout_per_tool=10,  # 每个工具超时 10 秒
        fail_fast=False  # 一个失败不影响其他
    )
]

agent = create_agent(
    model=model,
    tools=tools,
    middleware=middleware
)
```

---

## 9.5 Content Blocks 统一格式

### 9.5.1 跨 Provider 统一的消息格式

```python
from langchain_core.messages import AIMessage

# AIMessage 现在支持复杂的 content blocks
message = AIMessage(
    content=[
        {"type": "text", "text": "让我分析这个图片"},
        {"type": "image", "image_url": "https://..."},
        {"type": "tool_use", "tool_name": "calculator", "input": {"expression": "2+2"}},
        {"type": "thinking", "text": "我需要先理解用户的意图..."},  # Claude 特有
        {"type": "citation", "source": "维基百科", "text": "根据资料..."}  # Gemini 特有
    ]
)

# 自动适配不同 Provider
for provider in [ChatOpenAI(), ChatAnthropic(), ChatGoogleGenerativeAI()]:
    # 每个 provider 自动处理支持的 block 类型
    response = provider.invoke([message])
```

### 9.5.2 Thinking Blocks（Claude 特性）

```python
from langchain_anthropic import ChatAnthropic

# 启用思考过程展示
model = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    show_thinking=True  # 展示内部推理
)

response = model.invoke([HumanMessage("解决这个数学问题：...")])

# 提取思考过程
for block in response.content:
    if block.get("type") == "thinking":
        print("Claude 的思考过程：", block["text"])
    elif block.get("type") == "text":
        print("最终答案：", block["text"])
```

---

## 9.6 性能优化特性

### 9.6.1 智能缓存机制

```python
from langchain_core.caches import SQLiteCache
from langchain.globals import set_llm_cache

# 设置缓存后端
set_llm_cache(SQLiteCache(database_path=".cache/llm.db"))

# 或使用 Redis 缓存
from langchain_community.cache import RedisCache
import redis

redis_client = redis.Redis(host='localhost', port=6379)
set_llm_cache(RedisCache(redis_client, ttl=3600))

# 语义缓存（相似问题复用答案）
from langchain_community.cache import SemanticCache
from langchain_openai import OpenAIEmbeddings

semantic_cache = SemanticCache(
    embedding=OpenAIEmbeddings(),
    threshold=0.9  # 相似度阈值
)
set_llm_cache(semantic_cache)
```

### 9.6.2 批处理优化

```python
# LangChain 1.0 自动优化批处理
responses = await chain.abatch([
    {"messages": [HumanMessage("问题1")]},
    {"messages": [HumanMessage("问题2")]},
    {"messages": [HumanMessage("问题3")]},
    # ... 可以处理数百个请求
], config={"max_concurrency": 10})

# 性能对比
# 串行处理：100 个请求 × 1 秒 = 100 秒
# 批处理：100 个请求 ÷ 10 并发 = 10 秒
```

---

## 9.7 LangGraph 1.0 集成

### 9.7.1 状态持久化改进

```python
from langgraph.graph import StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

# SQLite 持久化
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

# 或 PostgreSQL 持久化
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost/dbname"
)

graph = StateGraph(State)
# ... 构建图
app = graph.compile(checkpointer=checkpointer)

# 保存和恢复状态
config = {"configurable": {"thread_id": "user-123"}}
result = app.invoke(input_data, config=config)

# 稍后恢复
result = app.invoke(new_input, config=config)  # 从上次状态继续
```

### 9.7.2 可视化调试增强

```python
# LangGraph 1.0 支持实时可视化
from langgraph.graph import visualize

# 生成 Mermaid 图
mermaid_graph = app.get_graph().draw_mermaid()
print(mermaid_graph)

# 导出为 PNG
app.get_graph().draw_png("workflow.png")

# 实时追踪执行
for event in app.stream(input_data, config=config):
    print(f"节点: {event.get('node')}")
    print(f"状态: {event.get('state')}")
```

---

## 9.8 生产部署增强

### 9.8.1 LangChain CLI

```bash
# 安装 LangChain CLI
pip install langchain-cli

# 创建新项目
langchain new my-agent --template react-agent

# 项目结构
my-agent/
├── app/
│   ├── __init__.py
│   ├── agent.py      # Agent 定义
│   ├── tools.py      # 工具定义
│   └── middleware.py # 中间件
├── tests/
├── .env.example
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml

# 本地运行
langchain serve

# 部署到 LangChain Platform
langchain deploy --name my-agent --env production
```

### 9.8.2 容器化支持

```dockerfile
# 官方基础镜像
FROM langchain/langchain:1.0-python3.11

WORKDIR /app

# 复制依赖
COPY pyproject.toml .
RUN pip install -e .

# 复制应用
COPY app/ ./app/

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:8000/health || exit 1

# 启动
CMD ["langchain", "serve", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 9.9 迁移指南

### 9.9.1 从 0.x 迁移到 1.0

```python
# 迁移检查工具
from langchain.migrate import check_compatibility

# 检查代码兼容性
issues = check_compatibility("./my_project")
for issue in issues:
    print(f"文件: {issue.file}")
    print(f"行号: {issue.line}")
    print(f"问题: {issue.description}")
    print(f"建议: {issue.suggestion}")
```

### 9.9.2 常见迁移场景

```python
# ========== 场景1：模型导入 ==========
# 旧版本
from langchain.llms import OpenAI  # ❌
from langchain.chat_models import ChatOpenAI  # ❌

# 新版本
from langchain_openai import OpenAI  # ✅
from langchain_openai import ChatOpenAI  # ✅

# ========== 场景2：向量数据库 ==========
# 旧版本
from langchain.vectorstores import Chroma  # ❌
vectorstore.persist()  # ❌ 方法已废弃

# 新版本
from langchain_chroma import Chroma  # ✅
# 自动持久化，无需调用 persist()

# ========== 场景3：工具定义 ==========
# 旧版本
from langchain.tools import Tool  # ⚠️ 仍可用但不推荐

# 新版本
from langchain.tools import tool  # ✅ 推荐装饰器方式

# ========== 场景4：Callbacks ==========
# 旧版本
from langchain.callbacks import CallbackManager  # ❌

# 新版本 - 使用 Middleware
from langchain.agents.middleware import AgentMiddleware  # ✅
```

---

## 9.10 最佳实践更新

### 9.10.1 推荐的项目结构

```
my-langchain-app/
├── app/
│   ├── agents/           # Agent 定义
│   │   ├── __init__.py
│   │   ├── base.py      # 基础 Agent
│   │   └── specialized.py # 特定领域 Agent
│   ├── tools/            # 工具集
│   │   ├── __init__.py
│   │   ├── search.py
│   │   └── database.py
│   ├── middleware/       # 中间件
│   │   ├── __init__.py
│   │   ├── security.py
│   │   └── monitoring.py
│   ├── chains/          # LCEL 链
│   │   └── rag.py
│   └── workflows/       # LangGraph 工作流
│       └── multi_agent.py
├── tests/               # 测试
├── evaluations/        # 评估数据集
├── configs/           # 配置文件
└── scripts/          # 部署脚本
```

### 9.10.2 性能优化清单

✅ **必做优化**：
1. 启用缓存（Redis/SQLite）
2. 使用异步调用（ainvoke/astream）
3. 实现批处理（batch/abatch）
4. 配置连接池
5. 设置超时和重试

⭐ **推荐优化**：
1. 语义缓存
2. 模型路由（按复杂度）
3. 流式输出
4. 工具并行调用
5. 状态持久化

🚀 **高级优化**：
1. 自定义缓存策略
2. 负载均衡
3. 断路器模式
4. 预测性预加载
5. 边缘部署

---

## 总结

LangChain 1.0 标志着框架的成熟，主要改进包括：

1. **稳定性承诺** - 生产级可靠性，无破坏性变更
2. **模块化架构** - 更清晰的包结构和依赖管理
3. **MCP 集成** - 标准化的数据源连接
4. **性能优化** - 并行处理、智能缓存、流式输出
5. **开发体验** - CLI 工具、更好的调试和可视化

这些更新使 LangChain 真正成为构建生产级 AI 应用的首选框架。