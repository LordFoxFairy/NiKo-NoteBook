# 第三篇 LangGraph 深入

> **目标**：理解 create_agent 背后的机制，掌握完全自定义能力

在第二篇中，我们使用 `create_agent` 快速构建了 Agent。但在复杂场景下，我们需要更精细的控制：
- 🎯 自定义 Agent 的执行流程
- 🔄 实现复杂的循环和分支逻辑
- 💾 保存和恢复 Agent 状态
- 🧠 实现长期记忆系统

这些需求，都需要理解 LangGraph 的底层机制。

---

## 第7章：LangGraph 核心原理

### 7.1 为什么需要 LangGraph

#### 7.1.1 create_agent 的局限性

`create_agent` 提供了快速构建 Agent 的能力，但在某些场景下存在局限：

**局限1：固定的执行流程**

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """搜索工具"""
    return f"搜索结果：{query}"

# create_agent 的执行流程是固定的
agent = create_agent(
    model=ChatOpenAI(model="gpt-4"),
    tools=[search]
)

# 执行流程：
# 1. 模型思考 → 2. 调用工具 → 3. 模型思考 → 4. 输出答案
# 无法自定义：
# - 在调用工具前添加验证步骤
# - 实现并行工具调用
# - 添加自定义的条件分支
```

**局限2：有限的状态控制**

```python
# create_agent 自动管理状态
# 无法：
# - 访问中间状态
# - 自定义状态结构
# - 在特定步骤保存快照
# - 实现复杂的状态更新逻辑

result = agent.invoke({"messages": [("user", "查询信息")]})

# 只能获取最终结果，看不到中间状态
print(result["messages"][-1].content)
```

**局限3：缺乏复杂控制流**

```python
# 需求：多步骤工作流
# 1. 搜索信息
# 2. 如果结果不足，继续搜索
# 3. 分析结果
# 4. 如果需要更多上下文，回到步骤1
# 5. 生成报告

# create_agent 无法直接实现这种复杂的循环和条件逻辑
```

#### 7.1.2 状态机思维与 LangGraph 的关系

**状态机 (State Machine) 概念**：

```mermaid
stateDiagram-v2
    [*] --> 分析问题
    分析问题 --> 需要工具: 需要外部信息
    分析问题 --> 生成答案: 信息充足
    需要工具 --> 调用工具
    调用工具 --> 分析问题: 继续分析
    生成答案 --> [*]

    note right of 分析问题
        状态：messages, iterations
    end note
```

**LangGraph = 状态机运行时**：

- **状态 (State)**：存储当前的数据（如消息历史、迭代次数）
- **节点 (Node)**：状态转移的处理函数
- **边 (Edge)**：定义状态转移路径
- **运行时**：按照图的定义执行状态转移

```python
# LangGraph 让你完全控制状态机的每个细节
from langgraph.graph import StateGraph, END

# 1. 定义状态
class AgentState(TypedDict):
    messages: list[BaseMessage]
    iterations: int
    search_results: list[str]

# 2. 定义节点（状态转移函数）
def analyze_node(state):
    # 分析当前状态，决定下一步
    pass

def search_node(state):
    # 执行搜索，更新状态
    pass

# 3. 定义边（转移条件）
def should_search(state):
    # 条件判断
    return "search" if need_more_info else "end"

# 4. 构建图
workflow = StateGraph(AgentState)
workflow.add_node("analyze", analyze_node)
workflow.add_node("search", search_node)
workflow.add_conditional_edges("analyze", should_search, {
    "search": "search",
    "end": END
})
```

**对比：create_agent vs LangGraph**

| 特性 | create_agent | LangGraph |
|------|--------------|-----------|
| **易用性** | ✅ 一行代码创建 | ⚠️ 需要定义状态和节点 |
| **灵活性** | ⚠️ 固定流程 | ✅ 完全自定义 |
| **状态控制** | ⚠️ 自动管理 | ✅ 完全可见和可控 |
| **条件逻辑** | ⚠️ 有限 | ✅ 任意复杂度 |
| **调试能力** | ⚠️ 黑盒 | ✅ 可追踪每个状态 |
| **适用场景** | 80% 的常规任务 | 复杂、需要精细控制的任务 |

---

### 7.2 基本元素

#### 7.2.1 State - 状态定义

**State 是什么**：存储在节点间传递的数据结构。

```python
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# 最简单的状态：只包含消息
class SimpleState(TypedDict):
    messages: list[BaseMessage]

# 带 Reducer 的状态：自动追加消息
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 复杂状态：包含多个字段
class ComplexState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    iterations: int              # 迭代次数
    search_results: list[str]    # 搜索结果
    current_task: str            # 当前任务
    is_complete: bool            # 是否完成
```

**Reducer 函数**：

```python
# 不使用 Reducer：每次覆盖整个列表
class State1(TypedDict):
    messages: list[BaseMessage]

# 节点返回
return {"messages": [new_message]}  # 会覆盖原有的 messages

# 使用 add_messages Reducer：自动追加
class State2(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 节点返回
return {"messages": [new_message]}  # 会追加到 messages 列表末尾
```

**add_messages 的工作原理**：

```python
# add_messages 是一个特殊的 Reducer

# 初始状态
state = {"messages": [HumanMessage(content="Hello")]}

# 节点1返回
update1 = {"messages": [AIMessage(content="Hi")]}

# add_messages 处理后
# state["messages"] = [HumanMessage("Hello"), AIMessage("Hi")]

# 节点2返回
update2 = {"messages": [HumanMessage(content="How are you?")]}

# add_messages 处理后
# state["messages"] = [
#     HumanMessage("Hello"),
#     AIMessage("Hi"),
#     HumanMessage("How are you?")
# ]
```

#### 7.2.2 Nodes - 节点函数

**节点是什么**：接收状态，执行操作，返回状态更新的函数。

```python
from langchain_openai import ChatOpenAI

# 节点签名
def node_function(state: AgentState) -> dict:
    """
    Args:
        state: 当前状态

    Returns:
        dict: 状态更新（会合并到当前状态）
    """
    # 1. 读取当前状态
    messages = state["messages"]
    iterations = state.get("iterations", 0)

    # 2. 执行操作
    model = ChatOpenAI(model="gpt-4")
    response = model.invoke(messages)

    # 3. 返回状态更新
    return {
        "messages": [response],
        "iterations": iterations + 1
    }
```

**常见节点类型**：

```python
# 1. 模型节点：调用 LLM
def call_model(state: AgentState) -> dict:
    model = ChatOpenAI(model="gpt-4")
    response = model.invoke(state["messages"])
    return {"messages": [response]}

# 2. 工具节点：执行工具
def call_tools(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls

    tool_messages = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        # 执行工具
        result = execute_tool(tool_name, tool_args)

        tool_messages.append(
            ToolMessage(
                content=result,
                tool_call_id=tool_call["id"]
            )
        )

    return {"messages": tool_messages}

# 3. 验证节点：检查和过滤
def validate(state: AgentState) -> dict:
    messages = state["messages"]
    last_message = messages[-1]

    # 检查是否违反安全规则
    if contains_sensitive_info(last_message.content):
        return {
            "messages": [AIMessage(content="抱歉，无法处理该请求")]
        }

    return {}  # 不更新状态

# 4. 聚合节点：整合多个来源的信息
def aggregate(state: AgentState) -> dict:
    search_results = state.get("search_results", [])

    # 整合搜索结果
    summary = summarize(search_results)

    return {
        "messages": [AIMessage(content=summary)],
        "search_results": []  # 清空
    }
```

#### 7.2.3 Edges - 边连接（普通边、条件边）

**普通边 (Edge)**：无条件转移

```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("node_a", node_a)
workflow.add_node("node_b", node_b)

# 普通边：node_a 执行完后，无条件进入 node_b
workflow.add_edge("node_a", "node_b")

# 结束边：node_b 执行完后，结束
workflow.add_edge("node_b", END)
```

**条件边 (Conditional Edge)**：根据状态决定下一步

```python
from typing import Literal

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    条件函数：根据状态返回下一个节点的名称

    Returns:
        "tools": 需要调用工具
        "end": 结束执行
    """
    last_message = state["messages"][-1]

    # 检查是否有工具调用
    if last_message.tool_calls:
        return "tools"

    return "end"

# 添加条件边
workflow.add_conditional_edges(
    "agent",                    # 从哪个节点出发
    should_continue,            # 条件函数
    {
        "tools": "call_tools",  # 映射：条件返回值 -> 目标节点
        "end": END
    }
)
```

**条件边示例：复杂路由**

```python
def route_question(state: AgentState) -> Literal["search", "calculate", "general"]:
    """根据问题类型路由"""
    question = state["messages"][-1].content

    if "天气" in question or "新闻" in question:
        return "search"
    elif any(op in question for op in ["+", "-", "*", "/", "计算"]):
        return "calculate"
    else:
        return "general"

workflow.add_conditional_edges(
    "classify",
    route_question,
    {
        "search": "search_node",
        "calculate": "calc_node",
        "general": "general_node"
    }
)
```

#### 7.2.4 Entry Point 与 End

**Entry Point**：图的起始节点

```python
workflow = StateGraph(AgentState)

workflow.add_node("start", start_node)
workflow.add_node("process", process_node)

# 设置入口点
workflow.set_entry_point("start")

# 等价于
workflow.add_edge("__start__", "start")
```

**END**：图的终止标记

```python
from langgraph.graph import END

# 方式1：直接边到 END
workflow.add_edge("final_node", END)

# 方式2：条件边到 END
workflow.add_conditional_edges(
    "decision_node",
    should_end,
    {
        "continue": "next_node",
        "end": END
    }
)
```

**完整示例**：

```python
from typing import Annotated, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# 1. 定义状态
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 2. 定义节点
def call_model(state: AgentState) -> dict:
    model = ChatOpenAI(model="gpt-4o-mini")
    response = model.invoke(state["messages"])
    return {"messages": [response]}

def call_tools(state: AgentState) -> dict:
    # 简化：直接返回模拟结果
    return {"messages": [AIMessage(content="工具执行结果")]}

# 3. 定义条件函数
def should_continue(state: AgentState) -> Literal["tools", "end"]:
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "end"

# 4. 构建图
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", call_tools)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)

workflow.add_edge("tools", "agent")

# 5. 编译
app = workflow.compile()

# 6. 执行
result = app.invoke({
    "messages": [HumanMessage(content="Hello")]
})

print(result["messages"][-1].content)
```

---

### 7.3 Graph 类型与执行

#### 7.3.1 StateGraph、MessageGraph、CompiledGraph

**StateGraph - 通用状态图**

```python
from langgraph.graph import StateGraph

# 自定义状态结构
class CustomState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    counter: int
    data: dict

workflow = StateGraph(CustomState)

# 完全控制状态结构
workflow.add_node("process", lambda state: {
    "counter": state["counter"] + 1,
    "data": {"result": "processed"}
})
```

**MessageGraph - 简化的消息图（已废弃）**

```python
# ⚠️ MessageGraph 在新版本中已被 StateGraph 替代

# 旧版本（不推荐）
from langgraph.graph import MessageGraph

graph = MessageGraph()
graph.add_node("agent", call_model)

# 新版本（推荐）：使用 StateGraph + MessagesState
from langgraph.graph import StateGraph, MessagesState

# MessagesState 是预定义的状态类型，等价于：
# class MessagesState(TypedDict):
#     messages: Annotated[list[BaseMessage], add_messages]

workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
```

**CompiledGraph - 编译后的图**

```python
# StateGraph 是构建器，CompiledGraph 是运行时

# 1. 使用 StateGraph 构建
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)

# 2. 编译成 CompiledGraph
app = workflow.compile()

# 3. CompiledGraph 提供执行方法
result = app.invoke({"messages": [HumanMessage("Hello")]})

# CompiledGraph 的类型
print(type(app))  # <class 'langgraph.graph.graph.CompiledGraph'>
```

#### 7.3.2 同步、异步、流式执行

**同步执行 (invoke)**

```python
app = workflow.compile()

# 同步调用：阻塞直到完成
result = app.invoke({
    "messages": [HumanMessage(content="你好")]
})

print(result["messages"][-1].content)
```

**异步执行 (ainvoke)**

```python
import asyncio

async def main():
    app = workflow.compile()

    # 异步调用
    result = await app.ainvoke({
        "messages": [HumanMessage(content="你好")]
    })

    print(result["messages"][-1].content)

asyncio.run(main())
```

**流式执行 (stream)**

```python
# 流式输出：每个节点执行后立即返回

app = workflow.compile()

for chunk in app.stream({
    "messages": [HumanMessage(content="讲个笑话")]
}):
    # chunk 格式：{"node_name": state_update}
    node_name = list(chunk.keys())[0]
    state_update = chunk[node_name]

    print(f"\n节点：{node_name}")
    print(f"状态更新：{state_update}")
```

**流式执行示例**：

```python
from langgraph.graph import StateGraph, END, MessagesState
from langchain_openai import ChatOpenAI

def call_model(state):
    model = ChatOpenAI(model="gpt-4o-mini")
    response = model.invoke(state["messages"])
    return {"messages": [response]}

workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)

app = workflow.compile()

# 流式执行
print("=== 流式执行 ===")
for chunk in app.stream({
    "messages": [HumanMessage(content="1+1等于几？")]
}):
    node_name = list(chunk.keys())[0]
    print(f"\n[{node_name}] 执行完成")

    if "messages" in chunk[node_name]:
        messages = chunk[node_name]["messages"]
        if messages:
            print(f"输出：{messages[-1].content}")
```

**异步流式执行 (astream)**

```python
async def stream_example():
    app = workflow.compile()

    async for chunk in app.astream({
        "messages": [HumanMessage(content="你好")]
    }):
        node_name = list(chunk.keys())[0]
        print(f"节点：{node_name}")

asyncio.run(stream_example())
```

**执行模式对比**

| 模式 | 方法 | 阻塞 | 返回方式 | 适用场景 |
|------|------|------|----------|----------|
| 同步 | invoke | 是 | 一次性返回 | 简单脚本、测试 |
| 异步 | ainvoke | 否 | 一次性返回 | 高并发、Web 服务 |
| 流式 | stream | 是 | 逐节点返回 | 进度展示、调试 |
| 异步流式 | astream | 否 | 逐节点返回 | 实时 UI、WebSocket |

---

## 本章小结

本章学习了 LangGraph 的核心概念：

### 核心概念

1. **为什么需要 LangGraph**
   - create_agent 的局限性：固定流程、有限状态控制
   - 状态机思维：State、Nodes、Edges

2. **基本元素**
   - **State**：状态定义、Reducer 函数（add_messages）
   - **Nodes**：节点函数（模型、工具、验证、聚合）
   - **Edges**：普通边（无条件转移）、条件边（根据状态路由）
   - **Entry Point & END**：起点和终点

3. **Graph 类型**
   - StateGraph：通用状态图
   - MessagesState：预定义的消息状态
   - CompiledGraph：编译后的可执行图

4. **执行模式**
   - invoke：同步执行
   - ainvoke：异步执行
   - stream：流式执行
   - astream：异步流式执行

### 下一步

在第8章中，我们将深入学习 **State 管理与 Memory 系统**，掌握：
- State 更新机制
- Checkpointer 持久化
- LangMem SDK（Episodic、Procedural、Semantic Memory）
- Graph 构建最佳实践

---

## 思考与练习

### 思考题

1. create_agent 和 LangGraph 的本质区别是什么？
2. add_messages Reducer 如何工作？为什么需要它？
3. 条件边和普通边的区别是什么？各自适用于什么场景？
4. 流式执行和同步执行的区别是什么？

### 练习题

**练习1：构建简单的对话 Agent**

要求：
- 使用 StateGraph 构建
- 包含 call_model 节点
- 使用 MessagesState
- 测试 invoke 和 stream

**练习2：实现条件路由**

要求：
- 根据用户问题类型路由到不同节点
- 实现 route_question 条件函数
- 包含至少3个不同的处理节点

**练习3：理解 Reducer**

要求：
- 创建不使用 add_messages 的状态
- 创建使用 add_messages 的状态
- 对比两者的行为差异

---

## 第8章：State 管理与 Memory 系统

### 8.1 State 定义与更新

#### 8.1.1 TypedDict 与 MessagesState

**TypedDict - 定义状态结构**

```python
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# 基础状态
class BasicState(TypedDict):
    """最简单的状态定义"""
    messages: list[BaseMessage]
    counter: int

# 使用 Annotated 添加 Reducer
class StateWithReducer(TypedDict):
    """带 Reducer 的状态"""
    messages: Annotated[list[BaseMessage], add_messages]
    counter: int

# 复杂状态
class ComplexState(TypedDict):
    """复杂的状态结构"""
    messages: Annotated[list[BaseMessage], add_messages]
    iterations: int
    search_results: list[dict]
    current_plan: str
    is_complete: bool
    metadata: dict
```

**MessagesState - 预定义的消息状态**

```python
from langgraph.graph import MessagesState, StateGraph

# MessagesState 等价于：
# class MessagesState(TypedDict):
#     messages: Annotated[list[BaseMessage], add_messages]

# 使用 MessagesState
workflow = StateGraph(MessagesState)

# 如果需要扩展 MessagesState
class ExtendedState(MessagesState):
    """扩展 MessagesState"""
    iterations: int
    user_id: str
```

**状态定义最佳实践**

```python
from typing import TypedDict, Annotated, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """Agent 状态定义

    最佳实践：
    1. 使用类型注解
    2. 添加文档字符串
    3. 为列表字段使用 Reducer
    4. 使用 Optional 标记可选字段
    """

    # 必需字段：对话历史
    messages: Annotated[list[BaseMessage], add_messages]

    # 可选字段：使用 NotRequired（Python 3.11+）或 total=False
    iterations: int
    search_results: Optional[list[str]]
    metadata: Optional[dict]
```

#### 8.1.2 State 更新机制（替换、增量、Reducer）

**机制1：完全替换**

```python
class State(TypedDict):
    counter: int
    data: dict

def node(state: State) -> dict:
    # 返回的字段会完全替换状态中的对应字段
    return {
        "counter": 10,  # 替换 state["counter"]
        "data": {"new": "value"}  # 替换 state["data"]
    }

# 初始状态
initial = {"counter": 0, "data": {"old": "value"}}

# 节点执行后
# final = {"counter": 10, "data": {"new": "value"}}
# ⚠️ {"old": "value"} 被完全替换
```

**机制2：增量更新（使用 Reducer）**

```python
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def node(state: State) -> dict:
    # add_messages Reducer 会追加，而不是替换
    return {
        "messages": [AIMessage(content="新消息")]
    }

# 初始状态
initial = {"messages": [HumanMessage(content="Hello")]}

# 节点执行后
# final = {
#     "messages": [
#         HumanMessage(content="Hello"),
#         AIMessage(content="新消息")  # 追加
#     ]
# }
```

**机制3：自定义 Reducer**

```python
from typing import Annotated

def merge_dicts(existing: dict, update: dict) -> dict:
    """自定义 Reducer：合并字典"""
    result = existing.copy()
    result.update(update)
    return result

def append_unique(existing: list, update: list) -> list:
    """自定义 Reducer：追加不重复元素"""
    result = existing.copy()
    for item in update:
        if item not in result:
            result.append(item)
    return result

class State(TypedDict):
    # 使用自定义 Reducer
    metadata: Annotated[dict, merge_dicts]
    tags: Annotated[list[str], append_unique]

# 初始状态
initial = {
    "metadata": {"user": "Alice", "session": "123"},
    "tags": ["urgent"]
}

# 节点返回
update = {
    "metadata": {"priority": "high"},
    "tags": ["urgent", "important"]
}

# 合并后
# final = {
#     "metadata": {"user": "Alice", "session": "123", "priority": "high"},
#     "tags": ["urgent", "important"]  # "urgent" 不重复
# }
```

**Reducer 执行顺序**

```python
# 多个节点依次更新状态

# 初始状态
state = {"messages": []}

# 节点1
state = apply_reducer(state, {"messages": [msg1]})
# state = {"messages": [msg1]}

# 节点2
state = apply_reducer(state, {"messages": [msg2]})
# state = {"messages": [msg1, msg2]}

# 节点3
state = apply_reducer(state, {"messages": [msg3]})
# state = {"messages": [msg1, msg2, msg3]}
```

---

### 8.2 Checkpointer 持久化

Checkpointer 允许保存和恢复 Agent 的状态，支持：
- 💾 会话持久化：关闭程序后恢复对话
- 🔄 中断恢复：在任意节点暂停和恢复
- 🕰️ 时间旅行：回到历史状态

#### 8.2.1 MemorySaver - 内存存储

**基础使用**

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, END
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# 定义节点
def call_model(state):
    model = ChatOpenAI(model="gpt-4o-mini")
    response = model.invoke(state["messages"])
    return {"messages": [response]}

# 构建图
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)

# 使用 MemorySaver
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# 会话1
config1 = {"configurable": {"thread_id": "conversation-1"}}
result1 = app.invoke(
    {"messages": [HumanMessage(content="我叫Alice")]},
    config=config1
)

# 会话2（相同 thread_id，会继续之前的对话）
result2 = app.invoke(
    {"messages": [HumanMessage(content="我叫什么名字？")]},
    config=config1
)

print(result2["messages"][-1].content)
# 输出："您叫 Alice"（记住了之前的对话）

# 会话3（不同 thread_id，是全新的对话）
config2 = {"configurable": {"thread_id": "conversation-2"}}
result3 = app.invoke(
    {"messages": [HumanMessage(content="我叫什么名字？")]},
    config=config2
)

print(result3["messages"][-1].content)
# 输出："抱歉，我不知道您的名字"
```

**MemorySaver 特点**

- ✅ 简单：无需配置，开箱即用
- ✅ 快速：内存操作，性能高
- ❌ 易失：程序重启后数据丢失
- ❌ 不共享：不同进程间无法共享

**适用场景**：
- 开发测试
- 单会话短期对话
- 不需要持久化的场景

#### 8.2.2 SqliteSaver - 本地持久化

**基础使用**

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# 连接到 SQLite 数据库
with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    app = workflow.compile(checkpointer=checkpointer)

    # 执行对话
    config = {"configurable": {"thread_id": "user-123"}}

    result1 = app.invoke(
        {"messages": [HumanMessage(content="记住：我喜欢猫")]},
        config=config
    )

    result2 = app.invoke(
        {"messages": [HumanMessage(content="我喜欢什么动物？")]},
        config=config
    )

    print(result2["messages"][-1].content)
    # 输出："您喜欢猫"

# 程序重启后
with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    app = workflow.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "user-123"}}

    # 继续之前的对话
    result3 = app.invoke(
        {"messages": [HumanMessage(content="我喜欢什么？")]},
        config=config
    )

    print(result3["messages"][-1].content)
    # 输出："您喜欢猫"（恢复了之前的状态）
```

**高级用法：查看历史状态**

```python
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    app = workflow.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "user-123"}}

    # 获取状态历史
    history = app.get_state_history(config)

    print("=== 状态历史 ===")
    for i, state in enumerate(history):
        print(f"\n状态 {i}:")
        print(f"  Checkpoint ID: {state.config['configurable'].get('checkpoint_id')}")
        print(f"  Messages: {len(state.values.get('messages', []))}")
```

**SqliteSaver 特点**

- ✅ 持久化：程序重启后数据保留
- ✅ 轻量：无需额外服务，单文件数据库
- ⚠️ 单机：仅支持单机部署
- ⚠️ 并发：并发写入性能有限

**适用场景**：
- 本地应用
- 单机部署
- 中小规模数据

#### 8.2.3 PostgresSaver - 生产级持久化

**基础使用**

```python
from langgraph.checkpoint.postgres import PostgresSaver

# 连接到 PostgreSQL
DB_URI = "postgresql://user:password@localhost:5432/langchain"

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    app = workflow.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "user-123"}}

    result = app.invoke(
        {"messages": [HumanMessage(content="Hello")]},
        config=config
    )
```

**生产环境配置**

```python
import os
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

# 连接池
pool = ConnectionPool(
    conninfo=os.environ["DATABASE_URL"],
    min_size=1,
    max_size=10,
    timeout=30
)

# 使用连接池
checkpointer = PostgresSaver(pool)

app = workflow.compile(checkpointer=checkpointer)
```

**PostgresSaver 特点**

- ✅ 生产级：高性能、高可靠
- ✅ 分布式：支持多实例部署
- ✅ 并发：支持高并发读写
- ✅ ACID：事务保证
- ⚠️ 复杂：需要运维 PostgreSQL

**适用场景**：
- 生产环境
- 多实例部署
- 高并发场景
- 需要 ACID 保证

#### 8.2.4 自定义 Checkpointer

**Checkpointer 接口**

```python
from langgraph.checkpoint import BaseCheckpointSaver, Checkpoint
from typing import Optional

class CustomCheckpointer(BaseCheckpointSaver):
    """自定义 Checkpointer"""

    def put(
        self,
        config: dict,
        checkpoint: Checkpoint,
        metadata: dict
    ) -> dict:
        """保存 checkpoint"""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = checkpoint["id"]

        # 实现：保存到自定义存储（如 Redis、MongoDB）
        save_to_storage(thread_id, checkpoint_id, checkpoint, metadata)

        return config

    def get(self, config: dict) -> Optional[Checkpoint]:
        """获取 checkpoint"""
        thread_id = config["configurable"]["thread_id"]

        # 实现：从存储加载
        return load_from_storage(thread_id)

    def list(self, config: dict) -> list[Checkpoint]:
        """列出所有 checkpoint"""
        thread_id = config["configurable"]["thread_id"]

        # 实现：列出历史 checkpoint
        return list_from_storage(thread_id)
```

**示例：Redis Checkpointer**

```python
import json
import redis
from langgraph.checkpoint import BaseCheckpointSaver, Checkpoint

class RedisCheckpointer(BaseCheckpointSaver):
    """基于 Redis 的 Checkpointer"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    def put(self, config: dict, checkpoint: Checkpoint, metadata: dict) -> dict:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = checkpoint["id"]

        key = f"checkpoint:{thread_id}:{checkpoint_id}"

        # 序列化并保存
        data = {
            "checkpoint": checkpoint,
            "metadata": metadata
        }
        self.redis.set(key, json.dumps(data))

        # 添加到索引
        self.redis.zadd(
            f"checkpoints:{thread_id}",
            {checkpoint_id: checkpoint["ts"]}
        )

        return config

    def get(self, config: dict) -> Optional[Checkpoint]:
        thread_id = config["configurable"]["thread_id"]

        # 获取最新的 checkpoint_id
        latest = self.redis.zrevrange(f"checkpoints:{thread_id}", 0, 0)

        if not latest:
            return None

        checkpoint_id = latest[0].decode()
        key = f"checkpoint:{thread_id}:{checkpoint_id}"

        data = self.redis.get(key)
        if not data:
            return None

        return json.loads(data)["checkpoint"]

# 使用
redis_client = redis.Redis(host="localhost", port=6379)
checkpointer = RedisCheckpointer(redis_client)

app = workflow.compile(checkpointer=checkpointer)
```

---

### 8.3 Memory 系统

#### 8.3.1 Short-term Memory（会话内记忆）

**Short-term Memory = Checkpointer + Thread ID**

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# 会话内记忆：使用相同的 thread_id
config = {"configurable": {"thread_id": "session-123"}}

# 第1轮
app.invoke({"messages": [HumanMessage("我叫 Alice")]}, config=config)

# 第2轮（记住之前的对话）
app.invoke({"messages": [HumanMessage("我叫什么？")]}, config=config)

# 第3轮（继续记住）
app.invoke({"messages": [HumanMessage("再说一遍")]}, config=config)
```

**控制记忆窗口**

```python
# 问题：对话太长，超过模型 token 限制

# 解决方案：限制消息数量
def trim_messages(state):
    """保留最近的 N 条消息"""
    messages = state["messages"]
    max_messages = 10

    if len(messages) > max_messages:
        # 保留系统消息 + 最近的消息
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        recent_msgs = messages[-max_messages:]

        return {"messages": system_msgs + recent_msgs}

    return {}

# 添加到 workflow
workflow.add_node("trim", trim_messages)
workflow.add_edge("trim", "agent")
```

#### 8.3.2 Long-term Memory（跨会话记忆）

**跨会话记忆需求**

```python
# 场景：用户在不同会话中都能被记住

# 会话1（2024-01-01）
config1 = {"configurable": {"thread_id": "session-001", "user_id": "alice"}}
app.invoke({"messages": [HumanMessage("我叫 Alice，喜欢猫")]}, config=config1)

# 会话2（2024-01-02，新的 thread_id）
config2 = {"configurable": {"thread_id": "session-002", "user_id": "alice"}}
app.invoke({"messages": [HumanMessage("我喜欢什么动物？")]}, config=config2)

# 期望：Agent 能记住 Alice 喜欢猫（跨会话记忆）
```

**实现方式：外部 Memory Store**

```python
# 简化的长期记忆实现

class LongTermMemory:
    """长期记忆存储"""

    def __init__(self):
        self.memories = {}

    def save(self, user_id: str, key: str, value: str):
        """保存记忆"""
        if user_id not in self.memories:
            self.memories[user_id] = {}
        self.memories[user_id][key] = value

    def recall(self, user_id: str, key: str) -> str:
        """回忆"""
        return self.memories.get(user_id, {}).get(key, "")

    def search(self, user_id: str, query: str) -> list[str]:
        """搜索相关记忆"""
        user_memories = self.memories.get(user_id, {})
        # 简化：返回所有记忆（实际应该用向量搜索）
        return list(user_memories.values())

# 全局记忆存储
long_term_memory = LongTermMemory()

# 在节点中使用
def agent_with_memory(state):
    user_id = state.get("user_id")

    # 回忆相关记忆
    memories = long_term_memory.search(user_id, state["messages"][-1].content)

    # 构建包含记忆的提示
    system_prompt = f"用户记忆：{', '.join(memories)}"

    model = ChatOpenAI(model="gpt-4")
    response = model.invoke([
        SystemMessage(content=system_prompt),
        *state["messages"]
    ])

    # 提取并保存新的记忆
    if "我叫" in state["messages"][-1].content:
        name = extract_name(state["messages"][-1].content)
        long_term_memory.save(user_id, "name", name)

    return {"messages": [response]}
```

#### 8.3.3 LangMem SDK

**LangMem 简介**

LangMem SDK 是 LangChain 提供的长期记忆管理工具，支持三种类型的记忆：
- **Episodic Memory**（情节记忆）：过去交互的具体事件
- **Procedural Memory**（过程记忆）：如何执行任务的知识
- **Semantic Memory**（语义记忆）：事实和知识

**安装**

```bash
pip install langmem
```

**Episodic Memory - 情节记忆**

```python
from langmem import EpisodicMemory
from langchain_openai import ChatOpenAI

# 创建情节记忆
episodic = EpisodicMemory(
    model=ChatOpenAI(model="gpt-4"),
    store_type="postgres",  # 或 "sqlite", "memory"
    connection_string="postgresql://localhost/langmem"
)

# 保存交互
episodic.add_interaction(
    user_id="alice",
    messages=[
        HumanMessage("推荐一部科幻电影"),
        AIMessage("推荐《星际穿越》，这是一部关于时间和引力的硬科幻电影")
    ],
    metadata={"session_id": "session-001", "timestamp": "2024-01-01"}
)

# 回忆相关交互
similar_interactions = episodic.search(
    user_id="alice",
    query="电影推荐",
    limit=5
)

# 在 Agent 中使用
def agent_with_episodic_memory(state):
    user_id = state.get("user_id")
    current_query = state["messages"][-1].content

    # 搜索相关的过去交互
    relevant_memories = episodic.search(user_id, current_query, limit=3)

    # 构建上下文
    context = "\n".join([
        f"过去的对话：\nQ: {m['messages'][0].content}\nA: {m['messages'][1].content}"
        for m in relevant_memories
    ])

    # 调用模型
    model = ChatOpenAI(model="gpt-4")
    response = model.invoke([
        SystemMessage(content=f"相关上下文：\n{context}"),
        *state["messages"]
    ])

    # 保存新的交互
    episodic.add_interaction(
        user_id=user_id,
        messages=[state["messages"][-1], response]
    )

    return {"messages": [response]}
```

**Procedural Memory - 过程记忆**

```python
from langmem import ProceduralMemory

# 创建过程记忆
procedural = ProceduralMemory(
    model=ChatOpenAI(model="gpt-4"),
    store_type="postgres"
)

# 保存任务执行方式
procedural.add_procedure(
    user_id="alice",
    task="数据分析",
    steps=[
        "1. 加载数据",
        "2. 数据清洗",
        "3. 统计分析",
        "4. 可视化",
        "5. 生成报告"
    ],
    feedback="这个流程效率很高",
    metadata={"task_type": "analysis"}
)

# 回忆如何执行任务
procedure = procedural.get_procedure(
    user_id="alice",
    task="数据分析"
)

# 在 Agent 中使用
def agent_with_procedural_memory(state):
    user_id = state.get("user_id")
    task = extract_task(state["messages"][-1].content)

    # 回忆如何执行这个任务
    procedure = procedural.get_procedure(user_id, task)

    if procedure:
        # 使用学到的流程
        system_prompt = f"执行'{task}'任务时，遵循这个流程：\n{procedure['steps']}"
    else:
        system_prompt = f"执行'{task}'任务"

    model = ChatOpenAI(model="gpt-4")
    response = model.invoke([
        SystemMessage(content=system_prompt),
        *state["messages"]
    ])

    return {"messages": [response]}
```

**Semantic Memory - 语义记忆**

```python
from langmem import SemanticMemory

# 创建语义记忆
semantic = SemanticMemory(
    model=ChatOpenAI(model="gpt-4"),
    embeddings=OpenAIEmbeddings(),
    store_type="postgres"
)

# 保存事实
semantic.add_fact(
    user_id="alice",
    fact="Alice 喜欢猫",
    category="preferences"
)

semantic.add_fact(
    user_id="alice",
    fact="Alice 是软件工程师",
    category="profile"
)

# 查询事实
facts = semantic.search(
    user_id="alice",
    query="Alice 的职业",
    limit=5
)

# 在 Agent 中使用
def agent_with_semantic_memory(state):
    user_id = state.get("user_id")
    query = state["messages"][-1].content

    # 搜索相关事实
    relevant_facts = semantic.search(user_id, query, limit=5)

    # 构建上下文
    context = "已知事实：\n" + "\n".join([f["fact"] for f in relevant_facts])

    model = ChatOpenAI(model="gpt-4")
    response = model.invoke([
        SystemMessage(content=context),
        *state["messages"]
    ])

    # 从对话中提取新事实
    new_facts = extract_facts(state["messages"][-1].content, response.content)
    for fact in new_facts:
        semantic.add_fact(user_id, fact)

    return {"messages": [response]}
```

**综合使用三种记忆**

```python
from langmem import EpisodicMemory, ProceduralMemory, SemanticMemory

class MemoryEnhancedAgent:
    """带完整记忆系统的 Agent"""

    def __init__(self, model):
        self.model = model
        self.episodic = EpisodicMemory(model)
        self.procedural = ProceduralMemory(model)
        self.semantic = SemanticMemory(model)

    def __call__(self, state):
        user_id = state.get("user_id")
        query = state["messages"][-1].content

        # 1. 回忆相关交互（情节记忆）
        past_interactions = self.episodic.search(user_id, query, limit=2)

        # 2. 回忆执行方法（过程记忆）
        task = extract_task(query)
        procedure = self.procedural.get_procedure(user_id, task)

        # 3. 回忆相关事实（语义记忆）
        facts = self.semantic.search(user_id, query, limit=5)

        # 4. 构建上下文
        context_parts = []

        if past_interactions:
            context_parts.append("相关过去对话：\n" + format_interactions(past_interactions))

        if procedure:
            context_parts.append(f"执行方法：\n{procedure['steps']}")

        if facts:
            context_parts.append("相关事实：\n" + "\n".join([f["fact"] for f in facts]))

        context = "\n\n".join(context_parts)

        # 5. 调用模型
        response = self.model.invoke([
            SystemMessage(content=context),
            *state["messages"]
        ])

        # 6. 保存新记忆
        self.episodic.add_interaction(user_id, [state["messages"][-1], response])

        new_facts = extract_facts(query, response.content)
        for fact in new_facts:
            self.semantic.add_fact(user_id, fact)

        return {"messages": [response]}

# 使用
from langgraph.graph import StateGraph, END

workflow = StateGraph(MessagesState)

agent = MemoryEnhancedAgent(ChatOpenAI(model="gpt-4"))
workflow.add_node("agent", agent)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)

app = workflow.compile(checkpointer=MemorySaver())
```

#### 8.3.4 Memory 最佳实践

**实践1：分层记忆架构**

```python
class LayeredMemory:
    """分层记忆架构

    L1: Short-term (Checkpointer) - 当前会话
    L2: Episodic (LangMem) - 最近交互
    L3: Semantic (LangMem) - 长期知识
    """

    def __init__(self):
        self.episodic = EpisodicMemory(...)
        self.semantic = SemanticMemory(...)

    def recall(self, user_id: str, query: str, state: dict):
        """分层回忆"""

        # L1: 当前会话（已在 state["messages"] 中）
        current_session = state["messages"]

        # L2: 最近交互（过去7天）
        recent_interactions = self.episodic.search(
            user_id,
            query,
            limit=3,
            filters={"days_ago": 7}
        )

        # L3: 长期知识（所有时间）
        long_term_facts = self.semantic.search(
            user_id,
            query,
            limit=5
        )

        return {
            "current": current_session,
            "recent": recent_interactions,
            "knowledge": long_term_facts
        }
```

**实践2：智能记忆提取**

```python
def extract_memories_from_conversation(messages: list[BaseMessage]):
    """从对话中提取值得记住的信息"""

    model = ChatOpenAI(model="gpt-4")

    extraction_prompt = """
    从以下对话中提取值得长期记住的信息：
    1. 用户偏好
    2. 重要事实
    3. 反馈和评价

    对话：
    {conversation}

    输出格式（JSON）：
    {
        "preferences": ["偏好1", "偏好2"],
        "facts": ["事实1", "事实2"],
        "feedback": ["反馈1"]
    }
    """

    conversation = "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in messages
    ])

    response = model.invoke([
        HumanMessage(content=extraction_prompt.format(conversation=conversation))
    ])

    import json
    return json.loads(response.content)
```

**实践3：记忆重要性评分**

```python
def score_memory_importance(memory: dict) -> float:
    """评估记忆的重要性（0-1）"""

    score = 0.0

    # 因素1：时间衰减
    days_ago = (datetime.now() - memory["timestamp"]).days
    time_factor = 1.0 / (1 + days_ago / 30)  # 30天衰减一半

    # 因素2：用户反馈
    if memory.get("feedback") == "positive":
        feedback_factor = 1.0
    elif memory.get("feedback") == "negative":
        feedback_factor = 0.3
    else:
        feedback_factor = 0.5

    # 因素3：访问频率
    access_count = memory.get("access_count", 0)
    access_factor = min(access_count / 10, 1.0)

    # 综合评分
    score = (time_factor * 0.4 + feedback_factor * 0.3 + access_factor * 0.3)

    return score

# 清理低重要性记忆
def cleanup_memories(user_id: str, threshold: float = 0.2):
    """清理低重要性的记忆"""
    all_memories = memory_store.list(user_id)

    for memory in all_memories:
        if score_memory_importance(memory) < threshold:
            memory_store.delete(memory["id"])
```

---

### 8.4 Graph 构建

#### 8.4.1 节点设计与职责划分

**单一职责原则**

```python
# ❌ 不好：一个节点做太多事
def god_node(state):
    # 验证输入
    # 调用模型
    # 执行工具
    # 格式化输出
    # ...
    pass

# ✅ 好：每个节点职责单一
def validate_input(state):
    """只负责验证"""
    pass

def call_model(state):
    """只负责调用模型"""
    pass

def execute_tools(state):
    """只负责执行工具"""
    pass

def format_output(state):
    """只负责格式化"""
    pass
```

**节点类型分类**

```python
# 1. 数据处理节点
def preprocess(state):
    """预处理输入"""
    messages = state["messages"]
    # 清理、规范化
    return {"messages": clean(messages)}

def postprocess(state):
    """后处理输出"""
    output = state["messages"][-1].content
    # 格式化、验证
    return {"messages": [AIMessage(content=format(output))]}

# 2. 业务逻辑节点
def analyze(state):
    """分析和决策"""
    data = state["data"]
    result = perform_analysis(data)
    return {"analysis_result": result}

# 3. 外部集成节点
def call_api(state):
    """调用外部 API"""
    query = state["query"]
    response = requests.get(f"https://api.example.com?q={query}")
    return {"api_result": response.json()}

# 4. 控制流节点
def route(state):
    """路由决策（只返回元数据，不修改状态）"""
    # 这个节点通常不返回状态更新
    # 而是在条件边中使用
    return {}
```

#### 8.4.2 条件路由实现

**基于内容的路由**

```python
def classify_question(state) -> Literal["search", "calc", "chat"]:
    """根据问题类型路由"""
    question = state["messages"][-1].content

    if any(keyword in question for keyword in ["搜索", "查找", "新闻"]):
        return "search"
    elif any(op in question for op in ["+", "-", "*", "/", "计算"]):
        return "calc"
    else:
        return "chat"

workflow.add_conditional_edges(
    "entry",
    classify_question,
    {
        "search": "search_node",
        "calc": "calculator_node",
        "chat": "chat_node"
    }
)
```

**基于状态的路由**

```python
def check_completeness(state) -> Literal["continue", "end"]:
    """检查任务是否完成"""

    # 检查1：是否达到最大迭代次数
    if state.get("iterations", 0) >= 10:
        return "end"

    # 检查2：是否所有子任务都完成
    todos = state.get("todos", [])
    if all(todo["status"] == "completed" for todo in todos):
        return "end"

    # 检查3：是否有最终答案
    last_message = state["messages"][-1]
    if "最终答案" in last_message.content:
        return "end"

    return "continue"

workflow.add_conditional_edges(
    "process",
    check_completeness,
    {
        "continue": "process",
        "end": END
    }
)
```

**使用 LLM 的路由**

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

class RoutingDecision(BaseModel):
    """路由决策"""
    route: Literal["technical", "sales", "support"]
    confidence: float
    reason: str

def llm_router(state) -> str:
    """使用 LLM 进行智能路由"""
    question = state["messages"][-1].content

    model = ChatOpenAI(model="gpt-4").with_structured_output(RoutingDecision)

    decision = model.invoke([
        SystemMessage(content="""
        根据用户问题，决定路由到哪个部门：
        - technical: 技术问题
        - sales: 销售咨询
        - support: 售后支持
        """),
        HumanMessage(content=question)
    ])

    return decision.route

workflow.add_conditional_edges(
    "classify",
    llm_router,
    {
        "technical": "tech_agent",
        "sales": "sales_agent",
        "support": "support_agent"
    }
)
```

#### 8.4.3 循环控制与防止死循环

**迭代计数器**

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    iterations: int  # 迭代计数器

def agent_node(state):
    """增加迭代计数"""
    iterations = state.get("iterations", 0)

    # ... 执行逻辑

    return {
        "messages": [response],
        "iterations": iterations + 1
    }

def should_continue(state) -> Literal["continue", "end"]:
    """检查迭代次数"""
    if state.get("iterations", 0) >= 10:
        return "end"

    # 其他终止条件...

    return "continue"
```

**recursion_limit 配置**

```python
# 在 invoke 时设置 recursion_limit
app = workflow.compile()

result = app.invoke(
    {"messages": [HumanMessage("Hello")]},
    config={"recursion_limit": 50}  # 最多执行50个节点
)

# 公式：recursion_limit = 2 × max_iterations + 1
# 例如：最多5次迭代 → recursion_limit = 11
```

**循环检测**

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    visited_nodes: list[str]  # 访问过的节点

def detect_loop(state) -> bool:
    """检测是否陷入循环"""
    visited = state.get("visited_nodes", [])

    # 检测：连续3次访问相同节点
    if len(visited) >= 3:
        if visited[-1] == visited[-2] == visited[-3]:
            return True

    return False

def safe_node(state, node_name: str):
    """带循环检测的节点包装器"""
    visited = state.get("visited_nodes", [])
    visited.append(node_name)

    if detect_loop({"visited_nodes": visited}):
        return {
            "messages": [AIMessage(content="检测到循环，停止执行")],
            "visited_nodes": visited
        }

    # 正常执行
    result = original_node(state)
    result["visited_nodes"] = visited
    return result
```

---

## 本章小结

本章学习了 State 管理与 Memory 系统：

### 核心概念

1. **State 定义与更新**
   - TypedDict 定义状态结构
   - MessagesState 预定义类型
   - 三种更新机制：替换、增量（Reducer）、自定义 Reducer

2. **Checkpointer 持久化**
   - MemorySaver：内存存储，开发测试
   - SqliteSaver：本地持久化，单机部署
   - PostgresSaver：生产级，分布式部署
   - 自定义 Checkpointer

3. **Memory 系统**
   - Short-term Memory：会话内记忆（Checkpointer + Thread ID）
   - Long-term Memory：跨会话记忆
   - LangMem SDK：
     - Episodic Memory（情节记忆）：过去交互
     - Procedural Memory（过程记忆）：执行方法
     - Semantic Memory（语义记忆）：事实知识

4. **Graph 构建**
   - 节点设计：单一职责、类型分类
   - 条件路由：基于内容、状态、LLM 决策
   - 循环控制：迭代计数、recursion_limit、循环检测

### 下一步

在第9章中，我们将学习 **持久化与 Human-in-the-Loop**，掌握：
- Thread 管理与时间旅行
- 中断与审批流程
- 自定义 Agent 完整实战

---

## 思考与练习

### 思考题

1. Reducer 函数的作用是什么？add_messages 如何工作？
2. MemorySaver、SqliteSaver、PostgresSaver 的适用场景是什么？
3. Episodic、Procedural、Semantic Memory 的区别是什么？
4. 如何防止 Agent 陷入死循环？

### 练习题

**练习1：实现带 Checkpointer 的 Agent**

要求：
- 使用 SqliteSaver
- 支持多个 thread_id
- 测试跨会话记忆

**练习2：集成 LangMem SDK**

要求：
- 实现 Episodic Memory
- 实现 Semantic Memory
- 在 Agent 中综合使用

**练习3：实现智能路由**

要求：
- 使用 LLM 进行路由决策
- 支持至少3个路由目标
- 记录路由决策过程

**练习4：循环控制**

要求：
- 实现迭代计数器
- 实现循环检测
- 测试防止死循环

---

## 第9章：持久化与 Human-in-the-Loop

### 9.1 Thread 管理

#### 9.1.1 Thread ID 与多会话隔离

**Thread ID 概念**

Thread ID 是用于隔离不同会话的标识符，每个 Thread ID 对应一个独立的状态历史。

```python
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    app = workflow.compile(checkpointer=checkpointer)

    # 用户A的会话
    config_a = {"configurable": {"thread_id": "user-a-session-1"}}
    app.invoke({"messages": [HumanMessage("我叫 Alice")]}, config=config_a)

    # 用户B的会话（完全独立）
    config_b = {"configurable": {"thread_id": "user-b-session-1"}}
    app.invoke({"messages": [HumanMessage("我叫 Bob")]}, config=config_b)

    # 用户A的第二轮（记住之前的对话）
    result_a = app.invoke(
        {"messages": [HumanMessage("我叫什么名字？")]},
        config=config_a
    )
    print(result_a["messages"][-1].content)  # "您叫 Alice"

    # 用户B的第二轮（也记住自己的对话）
    result_b = app.invoke(
        {"messages": [HumanMessage("我叫什么名字？")]},
        config=config_b
    )
    print(result_b["messages"][-1].content)  # "您叫 Bob"
```

**Thread ID 最佳实践**

```python
# 方案1：用户ID + 会话ID
thread_id = f"user-{user_id}-session-{session_id}"

# 方案2：用户ID + 日期
from datetime import datetime
thread_id = f"user-{user_id}-{datetime.now().strftime('%Y%m%d')}"

# 方案3：业务场景 + UUID
import uuid
thread_id = f"support-ticket-{uuid.uuid4()}"

# 使用
config = {"configurable": {"thread_id": thread_id}}
result = app.invoke(input, config=config)
```

**列出用户的所有会话**

```python
def list_user_threads(user_id: str, checkpointer) -> list[str]:
    """列出用户的所有会话"""

    # 注意：这需要自定义 Checkpointer 或查询数据库
    # 以下是概念示例

    query = f"SELECT DISTINCT thread_id FROM checkpoints WHERE thread_id LIKE 'user-{user_id}-%'"
    threads = execute_query(query)

    return threads
```

#### 9.1.2 时间旅行调试（状态快照、回溯）

**获取状态历史**

```python
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    app = workflow.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "debug-session"}}

    # 执行多轮对话
    app.invoke({"messages": [HumanMessage("第1轮")]}, config=config)
    app.invoke({"messages": [HumanMessage("第2轮")]}, config=config)
    app.invoke({"messages": [HumanMessage("第3轮")]}, config=config)

    # 获取状态历史
    history = app.get_state_history(config)

    print("=== 状态历史 ===")
    for i, state_snapshot in enumerate(history):
        checkpoint_id = state_snapshot.config["configurable"].get("checkpoint_id")
        messages_count = len(state_snapshot.values.get("messages", []))

        print(f"\n状态 {i}:")
        print(f"  Checkpoint ID: {checkpoint_id}")
        print(f"  消息数量: {messages_count}")
        print(f"  时间: {state_snapshot.metadata.get('timestamp')}")
```

**回到历史状态**

```python
# 获取历史状态
history = list(app.get_state_history(config))

# 选择第2个状态（倒数第二个 checkpoint）
target_state = history[1]

# 从该状态继续执行
result = app.invoke(
    {"messages": [HumanMessage("从历史状态继续")]},
    config=target_state.config  # 使用历史状态的 config
)
```

**调试工作流**

```python
def debug_workflow(app, input_data, thread_id: str):
    """调试工作流：打印每个节点的状态"""

    config = {"configurable": {"thread_id": thread_id}}

    print("=== 开始执行 ===")
    print(f"输入: {input_data}\n")

    # 流式执行，观察每个节点
    for chunk in app.stream(input_data, config=config):
        node_name = list(chunk.keys())[0]
        state_update = chunk[node_name]

        print(f"[节点: {node_name}]")
        print(f"  状态更新: {state_update}")
        print()

    # 获取最终状态
    final_state = app.get_state(config)
    print("=== 最终状态 ===")
    print(f"消息数量: {len(final_state.values.get('messages', []))}")
    print(f"最后一条消息: {final_state.values['messages'][-1].content}")

    # 查看状态历史
    print("\n=== 状态历史 ===")
    for i, state in enumerate(app.get_state_history(config)):
        print(f"状态 {i}: {len(state.values.get('messages', []))} 条消息")

# 使用
debug_workflow(
    app,
    {"messages": [HumanMessage("测试输入")]},
    thread_id="debug-001"
)
```

---

### 9.2 中断与审批

#### 9.2.1 interrupt() 函数与中断点

**基础中断**

```python
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import NodeInterrupt

def sensitive_action(state):
    """需要审批的敏感操作"""

    action = state["messages"][-1].content

    # 中断并等待人工审批
    raise NodeInterrupt(f"请审批操作：{action}")

# 构建图
workflow = StateGraph(MessagesState)
workflow.add_node("check", sensitive_action)
workflow.set_entry_point("check")
workflow.add_edge("check", END)

app = workflow.compile(checkpointer=MemorySaver())

# 执行
config = {"configurable": {"thread_id": "approval-001"}}

try:
    result = app.invoke(
        {"messages": [HumanMessage("删除数据库")]},
        config=config
    )
except NodeInterrupt as e:
    print(f"⏸️  中断: {e}")
    # 此时状态已保存，可以稍后恢复

# 人工审批后，恢复执行
# （这需要修改状态或配置来表示"已批准"）
```

**带审批的完整流程**

```python
from typing import Literal

class ApprovalState(MessagesState):
    """带审批的状态"""
    pending_action: str
    approved: bool

def request_approval(state) -> dict:
    """请求审批"""
    action = state["messages"][-1].content

    # 检查是否需要审批
    if requires_approval(action):
        # 中断并等待
        raise NodeInterrupt(f"等待审批：{action}")

    return {}

def execute_action(state) -> dict:
    """执行操作（审批后）"""
    action = state.get("pending_action")

    # 执行操作
    result = perform_action(action)

    return {"messages": [AIMessage(content=f"已执行：{result}")]}

def check_approval(state) -> Literal["approved", "denied"]:
    """检查审批状态"""
    if state.get("approved"):
        return "approved"
    return "denied"

# 构建图
workflow = StateGraph(ApprovalState)
workflow.add_node("request", request_approval)
workflow.add_node("execute", execute_action)
workflow.add_node("deny", lambda s: {"messages": [AIMessage("操作被拒绝")]})

workflow.set_entry_point("request")
workflow.add_conditional_edges(
    "request",
    check_approval,
    {
        "approved": "execute",
        "denied": "deny"
    }
)
workflow.add_edge("execute", END)
workflow.add_edge("deny", END)

app = workflow.compile(checkpointer=MemorySaver())
```

#### 9.2.2 审批流程设计（等待、批准、修改、恢复）

**完整的审批系统**

```python
from enum import Enum

class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"

class ApprovalState(MessagesState):
    approval_status: ApprovalStatus
    original_request: str
    modified_request: str

def agent_node(state):
    """Agent 节点"""
    model = ChatOpenAI(model="gpt-4")
    response = model.invoke(state["messages"])

    # 检查是否需要审批
    if needs_approval(response.content):
        return {
            "messages": [response],
            "approval_status": ApprovalStatus.PENDING,
            "original_request": response.content
        }

    return {"messages": [response]}

def approval_gate(state) -> Literal["needs_approval", "execute"]:
    """审批门控"""
    if state.get("approval_status") == ApprovalStatus.PENDING:
        # 中断，等待人工审批
        raise NodeInterrupt(
            f"需要审批：\n{state['original_request']}\n\n"
            f"请使用 update_state() 设置 approval_status"
        )

    return "execute"

def execute_node(state):
    """执行节点"""
    status = state.get("approval_status")

    if status == ApprovalStatus.APPROVED:
        request = state["original_request"]
        result = execute_request(request)
        return {"messages": [AIMessage(content=f"✅ 已执行：{result}")]}

    elif status == ApprovalStatus.MODIFIED:
        request = state["modified_request"]
        result = execute_request(request)
        return {"messages": [AIMessage(content=f"✅ 已执行修改后的请求：{result}")]}

    elif status == ApprovalStatus.REJECTED:
        return {"messages": [AIMessage(content="❌ 请求被拒绝")]}

# 构建图
workflow = StateGraph(ApprovalState)
workflow.add_node("agent", agent_node)
workflow.add_node("execute", execute_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", approval_gate, {
    "needs_approval": "execute",  # 中断后会跳转到这里
    "execute": "execute"
})
workflow.add_edge("execute", END)

app = workflow.compile(checkpointer=SqliteSaver.from_conn_string("approvals.db"))

# === 使用流程 ===

# 1. 用户发起请求
config = {"configurable": {"thread_id": "request-001"}}

try:
    result = app.invoke(
        {"messages": [HumanMessage("删除用户数据")]},
        config=config
    )
except NodeInterrupt as e:
    print(f"⏸️  {e}")

# 2. 管理员审批

# 2a. 批准
app.update_state(
    config,
    {"approval_status": ApprovalStatus.APPROVED}
)

# 2b. 拒绝
app.update_state(
    config,
    {"approval_status": ApprovalStatus.REJECTED}
)

# 2c. 修改后批准
app.update_state(
    config,
    {
        "approval_status": ApprovalStatus.MODIFIED,
        "modified_request": "删除用户数据（仅删除最近30天）"
    }
)

# 3. 恢复执行
result = app.invoke(None, config=config)  # None 表示从中断点继续
print(result["messages"][-1].content)
```

**审批 UI 示例**

```python
def approval_ui(app, checkpointer):
    """简单的审批 UI"""

    # 1. 列出所有待审批的请求
    pending_threads = get_pending_approvals(checkpointer)

    for thread_id in pending_threads:
        config = {"configurable": {"thread_id": thread_id}}
        state = app.get_state(config)

        print(f"\n{'='*60}")
        print(f"请求 ID: {thread_id}")
        print(f"请求内容: {state.values['original_request']}")
        print(f"状态: {state.values['approval_status']}")

        # 2. 管理员决策
        action = input("\n操作 [a]批准 [r]拒绝 [m]修改 [s]跳过: ")

        if action == "a":
            app.update_state(config, {"approval_status": ApprovalStatus.APPROVED})
            result = app.invoke(None, config=config)
            print(f"✅ {result['messages'][-1].content}")

        elif action == "r":
            app.update_state(config, {"approval_status": ApprovalStatus.REJECTED})
            result = app.invoke(None, config=config)
            print(f"❌ {result['messages'][-1].content}")

        elif action == "m":
            modified = input("修改后的请求: ")
            app.update_state(config, {
                "approval_status": ApprovalStatus.MODIFIED,
                "modified_request": modified
            })
            result = app.invoke(None, config=config)
            print(f"✅ {result['messages'][-1].content}")

def get_pending_approvals(checkpointer):
    """从数据库获取待审批的请求"""
    # 实现：查询所有 approval_status == PENDING 的 thread_id
    pass
```

---

### 9.3 实战：自定义 Agent

#### 9.3.1 自定义 RAG Agent 完整示例

```python
from typing import Annotated, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

# === 1. 定义状态 ===

class RAGState(TypedDict):
    """RAG Agent 状态"""
    messages: Annotated[list[BaseMessage], add_messages]
    retrieved_docs: list[str]
    iterations: int

# === 2. 定义工具 ===

# 初始化向量存储（示例）
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(
    [
        "LangChain 是一个用于构建 LLM 应用的框架",
        "LangGraph 是 LangChain 的状态机运行时",
        "LangSmith 是 LangChain 的监控平台"
    ],
    embeddings
)

@tool
def retrieve(query: str) -> str:
    """检索相关文档"""
    docs = vectorstore.similarity_search(query, k=2)
    return "\n".join([doc.page_content for doc in docs])

# === 3. 定义节点 ===

def agent_node(state: RAGState) -> dict:
    """Agent 节点：决定是否需要检索"""
    model = ChatOpenAI(model="gpt-4").bind_tools([retrieve])
    response = model.invoke(state["messages"])

    return {
        "messages": [response],
        "iterations": state.get("iterations", 0) + 1
    }

def retrieve_node(state: RAGState) -> dict:
    """检索节点"""
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls

    retrieved_docs = []
    tool_messages = []

    for tool_call in tool_calls:
        result = retrieve.invoke({"query": tool_call["args"]["query"]})
        retrieved_docs.append(result)

        tool_messages.append(
            ToolMessage(
                content=result,
                tool_call_id=tool_call["id"]
            )
        )

    return {
        "messages": tool_messages,
        "retrieved_docs": retrieved_docs
    }

def generate_node(state: RAGState) -> dict:
    """生成节点：基于检索结果生成答案"""
    docs = state.get("retrieved_docs", [])

    context = "\n\n".join(docs)
    messages = state["messages"]

    # 构建包含上下文的提示
    prompt = [
        SystemMessage(content=f"基于以下上下文回答问题：\n\n{context}"),
        *messages
    ]

    model = ChatOpenAI(model="gpt-4")
    response = model.invoke(prompt)

    return {"messages": [response]}

# === 4. 定义条件函数 ===

def should_retrieve(state: RAGState) -> Literal["retrieve", "generate", "end"]:
    """决策：是否需要检索"""
    last_message = state["messages"][-1]

    # 检查迭代次数
    if state.get("iterations", 0) >= 5:
        return "end"

    # 检查是否有工具调用
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "retrieve"

    # 已有检索结果，生成答案
    if state.get("retrieved_docs"):
        return "generate"

    return "end"

# === 5. 构建图 ===

workflow = StateGraph(RAGState)

# 添加节点
workflow.add_node("agent", agent_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

# 设置入口
workflow.set_entry_point("agent")

# 添加边
workflow.add_conditional_edges(
    "agent",
    should_retrieve,
    {
        "retrieve": "retrieve",
        "generate": "generate",
        "end": END
    }
)

workflow.add_edge("retrieve", "agent")  # 检索后继续思考
workflow.add_edge("generate", END)

# === 6. 编译 ===

app = workflow.compile(
    checkpointer=SqliteSaver.from_conn_string("rag_agent.db")
)

# === 7. 使用 ===

from langchain_core.messages import ToolMessage

config = {"configurable": {"thread_id": "rag-session-1"}}

# 流式执行，观察过程
print("=== RAG Agent 执行 ===\n")

for chunk in app.stream(
    {"messages": [HumanMessage("什么是 LangGraph？")]},
    config=config
):
    node_name = list(chunk.keys())[0]
    print(f"\n[{node_name}]")

    if "messages" in chunk[node_name]:
        for msg in chunk[node_name]["messages"]:
            print(f"  {type(msg).__name__}: {msg.content[:100]}...")

# 获取最终结果
final_state = app.get_state(config)
print("\n=== 最终答案 ===")
print(final_state.values["messages"][-1].content)
```

#### 9.3.2 Graph 调试技巧（可视化、日志、测试）

**技巧1：可视化 Graph**

```python
# 生成 Mermaid 图
from langchain_core.runnables.graph import CurveStyle, NodeStyles

mermaid_code = app.get_graph().draw_mermaid(
    curve_style=CurveStyle.LINEAR,
    node_colors=NodeStyles(
        first="fill:#e1f5ff",
        last="fill:#e8f5e9"
    )
)

print(mermaid_code)

# 保存为 PNG（需要 pygraphviz）
try:
    app.get_graph().draw_png(
        output_file_path="rag_agent_graph.png",
        fontname="Arial"
    )
    print("✅ 图表已保存")
except Exception as e:
    print(f"⚠️  {e}")
```

**技巧2：详细日志**

```python
from langchain_core.globals import set_verbose, set_debug

# 启用详细日志
set_verbose(True)
set_debug(True)

result = app.invoke(input_data, config=config)

# 会输出：
# - 每个节点的输入/输出
# - 每次 LLM 调用的详情
# - 状态更新过程
```

**技巧3：单元测试节点**

```python
import unittest

class TestRAGAgent(unittest.TestCase):
    """测试 RAG Agent 的各个节点"""

    def test_agent_node(self):
        """测试 Agent 节点"""
        state = {
            "messages": [HumanMessage("什么是 LangChain？")],
            "iterations": 0
        }

        result = agent_node(state)

        # 断言
        self.assertIn("messages", result)
        self.assertEqual(result["iterations"], 1)

    def test_retrieve_node(self):
        """测试检索节点"""
        state = {
            "messages": [
                HumanMessage("什么是 LangChain？"),
                AIMessage(
                    content="",
                    tool_calls=[{
                        "id": "call_123",
                        "name": "retrieve",
                        "args": {"query": "LangChain"}
                    }]
                )
            ]
        }

        result = retrieve_node(state)

        # 断言
        self.assertIn("messages", result)
        self.assertIn("retrieved_docs", result)
        self.assertGreater(len(result["retrieved_docs"]), 0)

    def test_should_retrieve(self):
        """测试条件函数"""
        # 场景1：需要检索
        state1 = {
            "messages": [AIMessage(content="", tool_calls=[{"name": "retrieve"}])],
            "iterations": 1
        }
        self.assertEqual(should_retrieve(state1), "retrieve")

        # 场景2：达到最大迭代
        state2 = {
            "messages": [AIMessage(content="答案")],
            "iterations": 5
        }
        self.assertEqual(should_retrieve(state2), "end")

# 运行测试
if __name__ == "__main__":
    unittest.main()
```

**技巧4：集成测试**

```python
def test_full_workflow():
    """完整工作流测试"""

    test_cases = [
        {
            "input": "什么是 LangChain？",
            "expected_keywords": ["LangChain", "框架", "LLM"]
        },
        {
            "input": "LangGraph 的作用是什么？",
            "expected_keywords": ["LangGraph", "状态机"]
        }
    ]

    for i, test_case in enumerate(test_cases):
        config = {"configurable": {"thread_id": f"test-{i}"}}

        result = app.invoke(
            {"messages": [HumanMessage(test_case["input"])]},
            config=config
        )

        answer = result["messages"][-1].content

        # 验证包含关键词
        for keyword in test_case["expected_keywords"]:
            assert keyword in answer, f"答案中缺少关键词：{keyword}"

        print(f"✅ 测试 {i+1} 通过")

test_full_workflow()
```

**技巧5：性能分析**

```python
import time

def profile_workflow(app, input_data, config):
    """性能分析"""

    node_times = {}
    start_time = time.time()

    for chunk in app.stream(input_data, config=config):
        node_name = list(chunk.keys())[0]
        node_start = time.time()

        # 记录节点执行时间
        if node_name not in node_times:
            node_times[node_name] = []

        node_times[node_name].append(time.time() - node_start)

    total_time = time.time() - start_time

    # 输出报告
    print("\n=== 性能分析 ===")
    print(f"总耗时: {total_time:.2f}s\n")

    for node_name, times in node_times.items():
        avg_time = sum(times) / len(times)
        print(f"{node_name}:")
        print(f"  调用次数: {len(times)}")
        print(f"  平均耗时: {avg_time:.2f}s")
        print(f"  总耗时: {sum(times):.2f}s")

# 使用
profile_workflow(
    app,
    {"messages": [HumanMessage("测试问题")]},
    {"configurable": {"thread_id": "profile-001"}}
)
```

---

## 本章小结

本章学习了持久化与 Human-in-the-Loop：

### 核心概念

1. **Thread 管理**
   - Thread ID：会话隔离
   - 最佳实践：命名规范
   - 时间旅行：状态快照、回溯

2. **中断与审批**
   - NodeInterrupt：中断执行
   - 审批流程：等待、批准、拒绝、修改
   - update_state：更新状态后恢复

3. **实战：自定义 Agent**
   - RAG Agent 完整实现
   - 调试技巧：可视化、日志、测试、性能分析

### 下一步

在第四篇中，我们已经学习了 **Deep Agents** 处理复杂任务。接下来是第五篇 **Middleware 工程化**，掌握：
- Middleware Hook 体系
- 内置 Middleware（PII、Summarization、Human-in-the-Loop）
- 自定义 Middleware 开发

---

## 思考与练习

### 思考题

1. Thread ID 的作用是什么？如何设计 Thread ID 命名规范？
2. NodeInterrupt 如何工作？如何实现审批流程？
3. 时间旅行调试有什么用？如何回到历史状态？
4. 如何测试 LangGraph 应用？

### 练习题

**练习1：实现多用户会话管理**

要求：
- 使用 Thread ID 隔离会话
- 支持列出用户的所有会话
- 实现会话切换

**练习2：实现审批工作流**

要求：
- 敏感操作需要审批
- 支持批准、拒绝、修改
- 实现简单的审批 UI

**练习3：自定义 SQL Agent**

要求：
- 包含 SQL 查询工具
- SQL 执行前需要审批
- 使用 Checkpointer 持久化

**练习4：调试和优化**

要求：
- 可视化 Graph 结构
- 记录性能数据
- 编写单元测试和集成测试
