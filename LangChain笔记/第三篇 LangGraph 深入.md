# 第三篇 LangGraph 深入

> **目标**：理解 create_agent 背后的机制，掌握完全自定义能力

在第二篇中，我们使用 `create_agent` 快速构建了 Agent。但在复杂场景下，我们需要更精细的控制：
- 🎯 自定义 Agent 的执行流程
- 🔄 实现复杂的循环和分支逻辑
- 💾 保存和恢复 Agent 状态
- 🧠 实现长期记忆系统

这些需求，都需要理解 LangGraph 的底层机制。

---

## 第1章：LangGraph 核心原理

### 1.1 为什么需要 LangGraph

#### 1.1.1 create_agent 的局限性

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

#### 1.1.2 状态机思维与 LangGraph 的关系

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
class GraphState(TypedDict):
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
workflow = StateGraph(GraphState)
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

### 1.2 基本元素

#### 1.2.1 State - 状态定义

**State 是什么**：存储在节点间传递的数据结构。

```python
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# 最简单的状态：只包含消息
class SimpleState(TypedDict):
    messages: list[BaseMessage]

# 带 Reducer 的状态：自动追加消息
class MessageState(TypedDict):
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

#### 1.2.2 Nodes - 节点函数

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

#### 1.2.3 Edges - 边连接（普通边、条件边）

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

#### 1.2.4 Entry Point 与 End

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
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 2. 定义节点
def call_model(state: ChatState) -> dict:
    model = ChatOpenAI(model="gpt-4o-mini")
    response = model.invoke(state["messages"])
    return {"messages": [response]}

def call_tools(state: ChatState) -> dict:
    # 简化：直接返回模拟结果
    return {"messages": [AIMessage(content="工具执行结果")]}

# 3. 定义条件函数
def should_continue(state: ChatState) -> Literal["tools", "end"]:
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "end"

# 4. 构建图
workflow = StateGraph(ChatState)

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

### 1.3 Graph 类型与执行

#### 1.3.1 Graph 类型与编译机制

LangGraph 使用**"构建器-编译器-运行时"**的设计模式,理解这个模式是掌握 LangGraph 的关键。

**1. StateGraph - 通用状态图构建器**

`StateGraph` 是最常用的图构建器,适用于任何自定义状态结构:

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# 定义自定义状态
class WorkflowState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    step_count: int
    processed_data: list[dict]

# 创建 StateGraph 实例
workflow = StateGraph(WorkflowState)

# 添加节点(处理逻辑)
def process_node(state: WorkflowState) -> dict:
    return {
        "step_count": state["step_count"] + 1,
        "processed_data": state["processed_data"] + [{"step": state["step_count"]}]
    }

workflow.add_node("process", process_node)
workflow.set_entry_point("process")
workflow.add_edge("process", END)

# 注意:此时 workflow 还不能执行,需要编译
```

**关键特点:**
- ✅ 完全自定义状态结构
- ✅ 适合复杂业务逻辑
- ✅ 类型安全(TypedDict 提供类型提示)

**2. MessagesState - 简化的消息图**

如果只需要处理消息历史,可以直接使用内置的 `MessagesState`:

```python
from langgraph.graph import StateGraph, MessagesState, END

# MessagesState 等价于:
# class MessagesState(TypedDict):
#     messages: Annotated[list[BaseMessage], add_messages]

# 直接使用 MessagesState,无需自定义状态
workflow = StateGraph(MessagesState)

def chat_node(state: MessagesState) -> dict:
    # 直接处理 messages
    response = model.invoke(state["messages"])
    return {"messages": [response]}

workflow.add_node("chat", chat_node)
workflow.set_entry_point("chat")
workflow.add_edge("chat", END)
```

**3. 编译: StateGraph → CompiledGraph**

`StateGraph` 是构建器,不能直接执行。通过 `compile()` 将其编译成 `CompiledGraph`:

```python
# 编译前: StateGraph (不可执行)
workflow = StateGraph(WorkflowState)
workflow.add_node("node1", func1)
workflow.set_entry_point("node1")
workflow.add_edge("node1", END)

# 编译后: CompiledGraph (可执行)
app = workflow.compile()

# CompiledGraph 提供执行接口
result = app.invoke({"messages": [], "step_count": 0, "processed_data": []})
```

**编译做了什么?**
- 🔍 验证图结构(是否有入口点、是否有无法到达的节点)
- 🔗 构建执行引擎(决定节点执行顺序、状态传递机制)
- 🚀 优化执行路径
- 💾 集成 Checkpointer(如果配置了持久化)

**4. 构建器 vs 运行时对比**

| 对比项 | StateGraph (构建器) | CompiledGraph (运行时) |
|--------|---------------------|------------------------|
| **作用** | 定义图结构 | 执行图逻辑 |
| **可修改** | ✅ 可以添加节点/边 | ❌ 不可修改,只读 |
| **可执行** | ❌ 不能执行 | ✅ 提供 invoke/stream 等方法 |
| **类型** | `StateGraph` | `CompiledGraph` |
| **何时使用** | 构建阶段 | 运行阶段 |

**5. compile() 的高级参数**

```python
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

app = workflow.compile(
    checkpointer=checkpointer,      # 持久化后端
    interrupt_before=["human_review"],  # 在指定节点前中断
    interrupt_after=["critical_step"],  # 在指定节点后中断
    debug=True                       # 开启调试模式
)
```

**最佳实践:**
- 🏗️ 构建阶段使用 `StateGraph`,灵活添加节点和边
- 🔒 编译后的 `CompiledGraph` 不可修改,确保运行时的稳定性
- ♻️ 如需修改图结构,重新构建 `StateGraph` 并编译

#### 1.3.2 同步、异步、流式执行

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

### 本章小结

本章学习了 LangGraph 的核心概念：

#### 核心概念

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

#### 下一步

在第8章中，我们将深入学习 **State 管理与 Memory 系统**，掌握：
- State 更新机制
- Checkpointer 持久化
- LangMem SDK（Episodic、Procedural、Semantic Memory）
- Graph 构建最佳实践

---

### 思考与练习

#### 思考题

1. create_agent 和 LangGraph 的本质区别是什么？
2. add_messages Reducer 如何工作？为什么需要它？
3. 条件边和普通边的区别是什么？各自适用于什么场景？
4. 流式执行和同步执行的区别是什么？

#### 练习题

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

## 第2章：State 管理与 Memory 系统

### 2.1 State 定义与更新

#### 2.1.1 自定义状态 (TypedDict)

当需要完全控制状态结构时,可以使用 `TypedDict` 自定义状态。

**基础用法**

```python
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# 最简单的状态定义
class BasicState(TypedDict):
    """基础状态：只包含必需字段"""
    messages: list[BaseMessage]
    counter: int
```

**使用 Reducer**

Reducer 定义了状态字段如何更新。最常用的是 `add_messages`,它会追加消息而不是替换:

```python
# 带 Reducer 的状态
class StateWithReducer(TypedDict):
    """使用 add_messages Reducer 自动追加消息"""
    messages: Annotated[list[BaseMessage], add_messages]
    counter: int

# 使用这个状态时:
# - messages 会自动追加新消息
# - counter 会被替换
```

**复杂状态示例**

```python
from typing import Optional

class WorkflowState(TypedDict):
    """复杂业务场景的状态定义"""
    # 对话历史 (使用 Reducer 自动追加)
    messages: Annotated[list[BaseMessage], add_messages]

    # 业务字段
    iterations: int              # 迭代次数
    search_results: list[dict]   # 搜索结果
    current_plan: str            # 当前计划
    is_complete: bool            # 是否完成

    # 可选字段
    metadata: Optional[dict]     # 元数据
```

**自定义状态最佳实践**

```python
class BestPracticeState(TypedDict):
    """状态定义最佳实践

    1. 添加清晰的文档字符串
    2. 使用类型注解
    3. 为列表/字典字段使用 Reducer
    4. 使用 Optional 标记可选字段
    5. 字段命名清晰、语义明确
    """

    # 必需字段：对话历史
    messages: Annotated[list[BaseMessage], add_messages]

    # 必需字段：业务数据
    user_id: str
    session_id: str

    # 可选字段：中间状态
    current_step: Optional[str]
    temp_data: Optional[dict]
```

#### 2.1.2 预定义状态类型

LangGraph 和 LangChain 提供了两种常用的预定义状态类型,适用于大多数场景。

**MessagesState - 简单对话状态**

`MessagesState` 是最常用的状态类型,适合只需要管理消息历史的场景:

```python
from langgraph.graph import MessagesState, StateGraph

# MessagesState 定义等价于:
# class MessagesState(TypedDict):
#     messages: Annotated[list[BaseMessage], add_messages]

# 直接使用
workflow = StateGraph(MessagesState)

def chat_node(state: MessagesState):
    """处理对话的节点"""
    messages = state["messages"]
    # ... 处理逻辑
    return {"messages": [response]}

workflow.add_node("chat", chat_node)
```

**扩展 MessagesState**

如果需要在消息基础上添加额外字段:

```python
class ExtendedMessagesState(MessagesState):
    """扩展 MessagesState 添加业务字段"""
    user_id: str
    session_type: str
    context: dict

workflow = StateGraph(ExtendedMessagesState)
```

**AgentState - 官方 Agent 状态**

`AgentState` 是 LangChain 提供的标准 Agent 状态,包含消息历史和迭代控制:

```python
from langchain.agents import AgentState

# AgentState 定义:
# class AgentState(TypedDict):
#     messages: Annotated[list[BaseMessage], add_messages]
#     remaining_steps: int  # 防止无限循环
```

**使用场景1: create_agent 默认使用**

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

# create_agent 内部默认使用 AgentState
agent = create_agent(
    model=ChatOpenAI(model="gpt-4"),
    tools=[search_tool],
    # state_schema 默认为 AgentState
)

# AgentState 自动管理 messages 和 remaining_steps
result = agent.invoke({"messages": [("user", "搜索信息")]})
```

**使用场景2: 在 StateGraph 中使用**

```python
from langchain.agents import AgentState
from langgraph.graph import StateGraph

# 直接使用官方 AgentState
workflow = StateGraph(AgentState)

def agent_node(state: AgentState):
    """Agent 节点会自动获得 remaining_steps"""
    steps_left = state.get("remaining_steps", 10)
    if steps_left <= 0:
        return {"messages": [AIMessage("达到最大步数")]}

    # 执行逻辑...
    return {
        "messages": [response],
        "remaining_steps": steps_left - 1
    }

workflow.add_node("agent", agent_node)
```

**使用场景3: 扩展 AgentState**

```python
from langchain.agents import AgentState

class CustomAgentState(AgentState):
    """扩展 AgentState 添加自定义字段"""
    user_context: dict
    tool_history: list[str]
    # messages 和 remaining_steps 自动继承

workflow = StateGraph(CustomAgentState)
```

**状态类型选择指南**

| 状态类型 | 适用场景 | 包含字段 | 何时使用 |
|---------|---------|---------|---------|
| `MessagesState` | 简单对话流程 | `messages` | 只需管理消息历史 |
| `AgentState` | Agent 应用 | `messages`, `remaining_steps` | 使用 `create_agent` 或需要步数限制 |
| 自定义 `TypedDict` | 复杂业务逻辑 | 完全自定义 | 需要额外业务字段 |
| 扩展预定义类型 | 标准场景+扩展 | 继承 + 自定义 | 在标准基础上添加字段 |

**选择建议:**
- 🎯 **优先使用预定义类型**: `MessagesState` 或 `AgentState` 能满足80%的需求
- 🔧 **扩展而非重写**: 需要额外字段时,扩展预定义类型而非从零开始
- 📦 **自定义仅在必要时**: 只有预定义类型完全不适用时才自定义

#### 2.1.3 State 更新机制

理解状态如何更新是使用 LangGraph 的关键。有三种更新机制。

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

### 2.2 Checkpointer 持久化

Checkpointer 允许保存和恢复 Agent 的状态，支持：
- 💾 会话持久化：关闭程序后恢复对话
- 🔄 中断恢复：在任意节点暂停和恢复
- 🕰️ 时间旅行：回到历史状态

#### 2.2.1 InMemorySaver - 内存存储

**基础使用**

```python
from langgraph.checkpoint.memory import InMemorySaver
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

# 使用 InMemorySaver
memory = InMemorySaver()
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

**InMemorySaver 特点**

- ✅ 简单：无需配置，开箱即用
- ✅ 快速：内存操作，性能高
- ❌ 易失：程序重启后数据丢失
- ❌ 不共享：不同进程间无法共享

**适用场景**：
- 开发测试
- 单会话短期对话
- 不需要持久化的场景

#### 2.2.2 SqliteSaver - 本地持久化

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

#### 2.2.3 PostgresSaver - 生产级持久化

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

#### 2.2.4 自定义 Checkpointer

**Checkpointer 接口**

```python
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    ChannelVersions
)
from langchain_core.runnables.config import RunnableConfig
from typing import Optional, Iterator, Dict, Any, Sequence, Tuple

class CustomCheckpointer(BaseCheckpointSaver):
    """自定义 Checkpointer - 符合最新版本接口"""

    def __init__(self):
        super().__init__()
        # 这里可以初始化存储后端，如 Redis、MongoDB 等
        self.storage = {}  # 简单示例用字典存储

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions
    ) -> RunnableConfig:
        """保存 checkpoint

        Args:
            config: 运行配置
            checkpoint: 要保存的检查点
            metadata: 检查点元数据
            new_versions: 通道版本信息

        Returns:
            更新后的配置
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]

        # 构造存储键
        key = f"{thread_id}:{checkpoint_ns}:{checkpoint_id}"

        # 保存到存储（实际应用中应该保存到 Redis、MongoDB 等）
        self.storage[key] = {
            "checkpoint": checkpoint,
            "metadata": metadata,
            "config": config,
            "new_versions": new_versions
        }

        return config

    def get(self, config: RunnableConfig) -> Optional[Checkpoint]:
        """获取最新的 checkpoint"""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id")

        if checkpoint_id:
            # 获取特定的 checkpoint
            key = f"{thread_id}:{checkpoint_ns}:{checkpoint_id}"
            if key in self.storage:
                return self.storage[key]["checkpoint"]
        else:
            # 获取最新的 checkpoint
            prefix = f"{thread_id}:{checkpoint_ns}:"
            matching_keys = [k for k in self.storage.keys() if k.startswith(prefix)]

            if matching_keys:
                latest_key = max(matching_keys,
                               key=lambda k: self.storage[k]["checkpoint"]["ts"])
                return self.storage[latest_key]["checkpoint"]

        return None

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None
    ) -> Iterator[CheckpointTuple]:
        """列出所有 checkpoint"""
        if config is None:
            items = list(self.storage.items())
        else:
            thread_id = config["configurable"]["thread_id"]
            checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
            prefix = f"{thread_id}:{checkpoint_ns}:"

            items = [(k, v) for k, v in self.storage.items()
                    if k.startswith(prefix)]

        # 按时间戳倒序排序
        items.sort(key=lambda x: x[1]["checkpoint"]["ts"], reverse=True)

        # 应用 limit
        if limit:
            items = items[:limit]

        # 生成 CheckpointTuple
        for key, value in items:
            yield CheckpointTuple(
                config=value["config"],
                checkpoint=value["checkpoint"],
                metadata=value["metadata"],
                parent_config=None
            )

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = ""
    ) -> None:
        """保存待写入的数据"""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id")

        key = f"{thread_id}:{checkpoint_ns}:{checkpoint_id}:writes:{task_id}"

        self.storage[key] = {
            "writes": writes,
            "task_id": task_id,
            "task_path": task_path
        }
```

**示例：Redis Checkpointer**

```python
import json
import redis
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    ChannelVersions
)
from langchain_core.runnables.config import RunnableConfig

class RedisCheckpointer(BaseCheckpointSaver):
    """基于 Redis 的 Checkpointer"""

    def __init__(self, redis_client: redis.Redis):
        super().__init__()
        self.redis = redis_client

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions
    ) -> RunnableConfig:
        """保存 checkpoint 到 Redis"""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]

        # 构造 Redis key
        key = f"checkpoint:{thread_id}:{checkpoint_ns}:{checkpoint_id}"

        # 序列化数据
        data = {
            "checkpoint": checkpoint,
            "metadata": metadata,
            "config": config,
            "new_versions": new_versions,
            "ts": checkpoint["ts"]  # 用于排序
        }

        # 保存到 Redis
        self.redis.set(key, json.dumps(data, default=str))

        # 添加到有序集合以便按时间排序
        score_key = f"checkpoints:{thread_id}:{checkpoint_ns}"
        self.redis.zadd(score_key, {checkpoint_id: checkpoint["ts"]})

        return config

    def get(self, config: RunnableConfig) -> Optional[Checkpoint]:
        """从 Redis 获取 checkpoint"""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id")

        if checkpoint_id:
            # 获取特定的 checkpoint
            key = f"checkpoint:{thread_id}:{checkpoint_ns}:{checkpoint_id}"
            data = self.redis.get(key)
            if data:
                return json.loads(data)["checkpoint"]
        else:
            # 获取最新的 checkpoint
            score_key = f"checkpoints:{thread_id}:{checkpoint_ns}"
            latest_ids = self.redis.zrevrange(score_key, 0, 0)

            if latest_ids:
                latest_id = latest_ids[0].decode() if isinstance(latest_ids[0], bytes) else latest_ids[0]
                key = f"checkpoint:{thread_id}:{checkpoint_ns}:{latest_id}"
                data = self.redis.get(key)
                if data:
                    return json.loads(data)["checkpoint"]

        return None

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None
    ) -> Iterator[CheckpointTuple]:
        """从 Redis 列出 checkpoints"""
        if config:
            thread_id = config["configurable"]["thread_id"]
            checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

            score_key = f"checkpoints:{thread_id}:{checkpoint_ns}"

            # 获取所有 checkpoint IDs，按时间倒序
            checkpoint_ids = self.redis.zrevrange(
                score_key,
                0,
                limit - 1 if limit else -1
            )

            for checkpoint_id in checkpoint_ids:
                cid = checkpoint_id.decode() if isinstance(checkpoint_id, bytes) else checkpoint_id
                key = f"checkpoint:{thread_id}:{checkpoint_ns}:{cid}"
                data = self.redis.get(key)

                if data:
                    parsed = json.loads(data)
                    yield CheckpointTuple(
                        config=parsed["config"],
                        checkpoint=parsed["checkpoint"],
                        metadata=parsed["metadata"],
                        parent_config=None
                    )

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = ""
    ) -> None:
        """保存写入数据到 Redis"""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id")

        key = f"writes:{thread_id}:{checkpoint_ns}:{checkpoint_id}:{task_id}"

        data = {
            "writes": list(writes),
            "task_id": task_id,
            "task_path": task_path
        }

        self.redis.set(key, json.dumps(data, default=str))

# 使用
redis_client = redis.Redis(host="localhost", port=6379)
checkpointer = RedisCheckpointer(redis_client)

app = workflow.compile(checkpointer=checkpointer)
```

---

### 2.3 Memory 系统

#### 2.3.1 Short-term Memory（会话内记忆）

**Short-term Memory = Checkpointer + Thread ID**

```python
from langgraph.checkpoint.memory import InMemorySaver

memory = InMemorySaver()
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

#### 2.3.2 Long-term Memory（跨会话记忆）

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

#### 2.3.3 Store API - 官方长期记忆方案

**Store API 简介**

LangGraph 1.0 引入的 `Store` API 是官方推荐的长期记忆存储方案，支持跨会话的持久化记忆、语义搜索和命名空间隔离。

**核心特性**：

1. **分层命名空间**：支持多层级的记忆隔离（如 `("user-123", "preferences")`）
2. **语义搜索**：基于向量相似度搜索相关记忆
3. **灵活存储**：支持 InMemoryStore、PostgresStore 等多种后端
4. **自动过期**：可设置记忆的生命周期

**基础用法**：

```python
from langgraph.store.memory import InMemoryStore

# 创建 Store
store = InMemoryStore()

# 分层命名空间: (user_id, category)
namespace = ("user-123", "preferences")

# 写入记忆（put）
store.put(namespace, "favorite_color", {"color": "blue", "timestamp": "2025-11-17"})
store.put(namespace, "interests", {"items": ["AI", "Python", "LangChain"]})

# 读取记忆（get）
color = store.get(namespace, "favorite_color")
print(color)  # {"color": "blue", "timestamp": "2025-11-17"}

# 列出命名空间下的所有记忆（list）
all_prefs = store.list(namespace)
for item in all_prefs:
    print(f"{item.key}: {item.value}")

# 删除记忆（delete）
store.delete(namespace, "favorite_color")
```

**语义搜索（高级功能）**：

```python
from langgraph.store.postgres import PostgresStore
from langchain_openai import OpenAIEmbeddings

# 使用 PostgreSQL 后端（支持向量搜索）
store = PostgresStore(
    conn_string="postgresql://user:pass@localhost/db",
    embeddings=OpenAIEmbeddings()
)

namespace = ("user-123", "conversations")

# 存储对话记忆
store.put(namespace, "mem1", {"content": "用户喜欢猫"})
store.put(namespace, "mem2", {"content": "用户住在北京"})
store.put(namespace, "mem3", {"content": "用户是Python开发者"})

# 语义搜索相关记忆
results = store.search(
    namespace,
    query="用户的宠物偏好",
    limit=2,
    filter={"type": "preference"}  # 可选过滤条件
)

for result in results:
    print(f"相关度: {result.score}, 内容: {result.value['content']}")
# 输出: "用户喜欢猫" (高相关度)
```

**在 LangGraph 中使用 Store**：

```python
from langgraph.graph import StateGraph, MessagesState
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

class UserState(MessagesState):
    user_id: str

def create_agent_with_store():
    """创建带长期记忆的 Agent"""
    # 初始化 Store
    store = InMemoryStore()
    checkpointer = InMemorySaver()
    model = ChatOpenAI(model="gpt-4")

    def agent_node(state: UserState, *, store):
        """Agent 节点（自动接收 store 参数）"""
        user_id = state["user_id"]
        namespace = (user_id, "profile")

        # 读取用户记忆
        name_mem = store.get(namespace, "name")
        interests_mem = store.get(namespace, "interests")

        # 构建系统提示
        memory_context = ""
        if name_mem:
            memory_context += f"\n用户姓名: {name_mem['value']}"
        if interests_mem:
            memory_context += f"\n兴趣爱好: {', '.join(interests_mem['items'])}"

        # 调用模型
        response = model.invoke([
            SystemMessage(content=f"你是助手。用户信息:{memory_context}"),
            *state["messages"]
        ])

        # 提取并保存新记忆（简化示例）
        user_msg = state["messages"][-1].content
        if "我叫" in user_msg:
            # 提取姓名并保存
            name = user_msg.split("我叫")[1].strip().split()[0]
            store.put(namespace, "name", {"value": name})

        return {"messages": [response]}

    # 构建图
    workflow = StateGraph(UserState)
    workflow.add_node("agent", agent_node)
    workflow.set_entry_point("agent")
    workflow.set_finish_point("agent")

    # 编译时传入 store
    return workflow.compile(
        checkpointer=checkpointer,
        store=store  # ✅ 关键：传入 store
    )

# 使用
app = create_agent_with_store()

# 第一次对话：用户自我介绍
config = {"configurable": {"thread_id": "thread-1"}}
result = app.invoke({
    "messages": [("user", "你好，我叫Alice")],
    "user_id": "user-123"
}, config)

# 第二次对话：Agent 能记住用户姓名（跨会话）
result = app.invoke({
    "messages": [("user", "你还记得我的名字吗？")],
    "user_id": "user-123"
}, config)
# Agent: "当然记得，Alice！有什么可以帮你的吗？"
```

**Store 最佳实践**：

1. **命名空间设计**：
   ```python
   # 用户级别
   ("user-123", "profile")        # 用户基本信息
   ("user-123", "preferences")    # 用户偏好
   ("user-123", "conversations")  # 对话历史
   
   # 组织级别
   ("org-456", "team-settings")
   ("org-456", "shared-knowledge")
   ```

2. **记忆更新策略**：
   ```python
   # 增量更新
   existing = store.get(namespace, "interests") or {"items": []}
   existing["items"].append("新兴趣")
   store.put(namespace, "interests", existing)
   
   # 带时间戳
   store.put(namespace, "last_login", {
       "time": datetime.now().isoformat(),
       "ip": "192.168.1.1"
   })
   ```

3. **内存管理**：
   ```python
   # 定期清理旧记忆
   all_items = store.list(namespace)
   for item in all_items:
       if is_expired(item.value.get("timestamp")):
           store.delete(namespace, item.key)
   ```

**Store vs Checkpointer**：

| 特性 | Store (长期记忆) | Checkpointer (短期记忆) |
|------|-----------------|----------------------|
| **生命周期** | 跨会话持久化 | 会话内临时 |
| **数据范围** | 用户/组织级别 | 线程级别 |
| **搜索能力** | 支持语义搜索 | 无搜索功能 |
| **使用场景** | 用户画像、知识库 | 对话历史、状态恢复 |
| **存储内容** | 结构化数据 | State 快照 |

---

#### 2.3.4 LangMem SDK

**LangMem 简介**

LangMem SDK 是 LangChain 提供的长期记忆管理工具，用于在 LangGraph 中实现持久化记忆系统。它提供了记忆的提取、管理和搜索功能。

**安装**

```bash
pip install langmem
```

**核心概念**

LangMem 提供了两个核心组件：
1. **Memory Manager**: 从对话中提取和管理记忆
2. **Memory Searcher**: 搜索相关记忆

**使用 Memory Manager**

```python
from langmem import create_memory_manager, create_memory_searcher
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState
from langchain_core.messages import HumanMessage, AIMessage

# 创建记忆管理器节点
memory_manager = create_memory_manager(
    model=ChatOpenAI(model="gpt-4"),
    # 可选：自定义记忆提取指令
    instructions="提取用户偏好、重要事实和交互模式"
)

# 创建记忆搜索器节点
memory_searcher = create_memory_searcher(
    model=ChatOpenAI(model="gpt-4"),
    prompt="搜索与当前对话相关的记忆"
)

# 在 LangGraph 中集成记忆系统
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from operator import add

class MemoryState(TypedDict):
    messages: Annotated[list, add]
    memories: list
    user_id: str

# 构建带记忆的工作流
def build_memory_graph():
    workflow = StateGraph(MemoryState)

    # 添加记忆搜索节点
    async def search_memories(state: MemoryState):
        """搜索相关记忆"""
        # memory_searcher 会自动处理消息并返回相关记忆
        memories = await memory_searcher.ainvoke(state)
        return {"memories": memories}

    # 添加 Agent 节点
    def agent(state: MemoryState):
        """使用记忆的 Agent"""
        model = ChatOpenAI(model="gpt-4")

        # 构建带记忆的系统提示
        memory_context = ""
        if state.get("memories"):
            memory_context = "\n相关记忆：\n" + "\n".join([
                f"- {mem['content']}" for mem in state["memories"]
            ])

        response = model.invoke([
            SystemMessage(content=f"你是一个有记忆的助手。{memory_context}"),
            *state["messages"]
        ])

        return {"messages": [response]}

    # 添加记忆管理节点
    async def manage_memories(state: MemoryState):
        """提取并保存新记忆"""
        # memory_manager 会自动从对话中提取记忆
        extracted_memories = await memory_manager.ainvoke(state)
        # 记忆会自动保存到存储中
        return {}

    # 构建工作流
    workflow.add_node("search_memories", search_memories)
    workflow.add_node("agent", agent)
    workflow.add_node("manage_memories", manage_memories)

    # 定义边
    workflow.add_edge("search_memories", "agent")
    workflow.add_edge("agent", "manage_memories")
    workflow.add_edge("manage_memories", END)

    # 设置入口
    workflow.set_entry_point("search_memories")

    return workflow.compile()

# 使用示例
app = build_memory_graph()

# 运行对话
result = await app.ainvoke({
    "messages": [HumanMessage("我叫 Alice，我喜欢科幻电影")],
    "user_id": "alice",
    "memories": []
})

# 后续对话会记住之前的信息
result = await app.ainvoke({
    "messages": [HumanMessage("推荐一部电影给我")],
    "user_id": "alice",
    "memories": []
})
```

**自定义记忆存储**

```python
from langgraph.store import BaseStore
import json

class CustomMemoryStore(BaseStore):
    """自定义记忆存储实现"""

    def __init__(self):
        self.store = {}

    async def aget(self, namespace: tuple, key: str):
        """获取记忆"""
        ns_key = "/".join(namespace)
        return self.store.get(ns_key, {}).get(key)

    async def aput(self, namespace: tuple, key: str, value: dict):
        """保存记忆"""
        ns_key = "/".join(namespace)
        if ns_key not in self.store:
            self.store[ns_key] = {}
        self.store[ns_key][key] = value

    async def asearch(self, namespace: tuple, query: str, limit: int = 10):
        """搜索记忆"""
        ns_key = "/".join(namespace)
        memories = self.store.get(ns_key, {})
        # 简单实现：返回所有记忆
        # 实际应该使用向量搜索
        return list(memories.values())[:limit]

# 使用自定义存储
store = CustomMemoryStore()

# 创建带自定义存储的记忆管理器
memory_manager = create_memory_manager(
    model=ChatOpenAI(model="gpt-4"),
    # store 参数需要通过 LangGraph 配置传递
)
```

**实际使用示例：构建记忆增强的客服助手**

```python
from langmem import create_memory_manager, create_memory_searcher
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, END
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

class CustomerServiceState(MessagesState):
    """客服状态，包含消息和记忆"""
    memories: list = []
    user_id: str = ""

def build_customer_service_agent():
    """构建带记忆的客服助手"""

    # 初始化组件
    model = ChatOpenAI(model="gpt-4")
    store = InMemoryStore()

    # 创建记忆管理组件
    memory_manager = create_memory_manager(
        model=model,
        instructions="""
        提取以下信息：
        1. 用户偏好和需求
        2. 用户问题和解决方案
        3. 用户满意度和反馈
        4. 重要的个人信息（如会员等级、购买历史）
        """
    )

    memory_searcher = create_memory_searcher(
        model=model,
        prompt="搜索与用户问题相关的历史记录和解决方案"
    )

    # 创建工作流
    workflow = StateGraph(CustomerServiceState)

    # 搜索记忆节点
    async def search_memories(state: CustomerServiceState):
        """搜索用户历史记忆"""
        if not state["messages"]:
            return {"memories": []}

        # 使用 memory_searcher 搜索相关记忆
        memories = await memory_searcher.ainvoke({
            "messages": state["messages"],
            "user_id": state["user_id"]
        })

        return {"memories": memories}

    # 客服响应节点
    def customer_service(state: CustomerServiceState):
        """生成客服响应"""
        # 构建记忆上下文
        memory_context = ""
        if state["memories"]:
            memory_context = "\n用户历史信息：\n"
            for mem in state["memories"]:
                memory_context += f"- {mem.get('content', '')}\n"

        # 生成响应
        system_prompt = f"""
        你是一个专业的客服助手。
        {memory_context}

        根据用户的历史信息和当前问题，提供个性化的服务。
        """

        response = model.invoke([
            SystemMessage(content=system_prompt),
            *state["messages"]
        ])

        return {"messages": [response]}

    # 保存记忆节点
    async def save_memories(state: CustomerServiceState):
        """从对话中提取并保存新记忆"""
        if len(state["messages"]) < 2:
            return {}

        # 使用 memory_manager 提取记忆
        await memory_manager.ainvoke({
            "messages": state["messages"][-2:],  # 最后一轮对话
            "user_id": state["user_id"]
        })

        return {}

    # 添加节点
    workflow.add_node("search_memories", search_memories)
    workflow.add_node("customer_service", customer_service)
    workflow.add_node("save_memories", save_memories)

    # 定义流程
    workflow.set_entry_point("search_memories")
    workflow.add_edge("search_memories", "customer_service")
    workflow.add_edge("customer_service", "save_memories")
    workflow.add_edge("save_memories", END)

    # 编译应用
    checkpointer = InMemorySaver()
    app = workflow.compile(
        checkpointer=checkpointer,
        store=store  # 传递存储
    )

    return app

# 使用客服助手
app = build_customer_service_agent()

# 第一次对话
config = {"configurable": {"thread_id": "user-123"}}
result = await app.ainvoke({
    "messages": [HumanMessage("我是 VIP 会员，上个月买的商品有质量问题")],
    "user_id": "user-123"
}, config)

# 后续对话（会记住用户是 VIP 会员）
result = await app.ainvoke({
    "messages": [HumanMessage("能否加急处理我的退款？")],
    "user_id": "user-123"
}, config)
```

**记忆系统最佳实践**

1. **记忆分类**：
   - 短期记忆：使用 Checkpointer（会话内）
   - 长期记忆：使用 LangMem（跨会话）
   - 工作记忆：使用 State（当前任务）

2. **记忆管理策略**：
   - 定期清理过期记忆
   - 合并相似记忆避免冗余
   - 设置记忆优先级和重要性

3. **性能优化**：
   - 使用向量数据库加速搜索
   - 实现记忆缓存机制
   - 异步处理记忆操作

4. **隐私和安全**：
   - 加密敏感记忆
   - 实现用户数据删除功能
   - 记忆访问权限控制

#### 2.3.5 Memory 最佳实践

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

### 2.4 Graph 构建

#### 2.4.1 节点设计与职责划分

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

#### 2.4.2 条件路由实现

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

#### 2.4.3 循环控制与防止死循环

**迭代计数器**

```python
class LoopControlState(TypedDict):
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
class StateWithHistory(TypedDict):
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

### 本章小结

本章学习了 State 管理与 Memory 系统：

#### 核心概念

1. **State 定义与更新**
   - TypedDict 定义自定义状态结构
   - MessagesState 预定义类型（仅包含 messages）
   - AgentState 官方 Agent 状态（包含 messages 和 remaining_steps）
   - 三种更新机制：替换、增量（Reducer）、自定义 Reducer

2. **Checkpointer 持久化**
   - InMemorySaver：内存存储，开发测试
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

#### 下一步

在第9章中，我们将学习 **持久化与 Human-in-the-Loop**，掌握：
- Thread 管理与时间旅行
- 中断与审批流程
- 自定义 Agent 完整实战

---

### 思考与练习

#### 思考题

1. Reducer 函数的作用是什么？add_messages 如何工作？
2. InMemorySaver、SqliteSaver、PostgresSaver 的适用场景是什么？
3. Episodic、Procedural、Semantic Memory 的区别是什么？
4. 如何防止 Agent 陷入死循环？

#### 练习题

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

## 第3章：持久化与 Human-in-the-Loop

### 3.1 Thread 管理

#### 3.1.1 Thread ID 与多会话隔离

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

#### 3.1.2 时间旅行调试（状态快照、回溯）

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

### 3.2 中断与审批

#### 3.2.1 interrupt() 函数与中断点

**基础中断**

```python
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt

def sensitive_action(state):
    """需要审批的敏感操作"""
    action = state["messages"][-1].content

    # 使用 interrupt() 函数暂停执行,等待人工审批
    # interrupt() 会暂停执行并返回提供的值给调用者
    approved = interrupt(f"请审批操作：{action}")

    # 当恢复执行时,approved 会接收恢复时传入的值
    if approved:
        return {"messages": [AIMessage("操作已批准并执行")]}
    else:
        return {"messages": [AIMessage("操作被拒绝")]}

# 构建图
workflow = StateGraph(MessagesState)
workflow.add_node("check", sensitive_action)
workflow.set_entry_point("check")
workflow.add_edge("check", END)

app = workflow.compile(checkpointer=InMemorySaver())

# 第一次执行 - 会在 interrupt() 处暂停
config = {"configurable": {"thread_id": "approval-001"}}
result = app.invoke(
    {"messages": [HumanMessage("删除数据库")]},
    config=config
)

# 查看中断信息
print(f"⏸️  中断信息: {result}")

# 人工审批后,使用 Command 恢复执行
from langgraph.types import Command

# 批准操作
app.invoke(Command(resume=True), config=config)

# 或拒绝操作
# app.invoke(Command(resume=False), config=config)
```

**带条件的审批流程**

```python
from typing import Literal
from langgraph.types import interrupt

class ApprovalState(MessagesState):
    """带审批的状态"""
    action_description: str

def request_approval(state) -> dict:
    """请求审批节点"""
    action = state["messages"][-1].content

    # 检查是否是敏感操作
    if "删除" in action or "修改" in action:
        # 使用 interrupt() 暂停并等待审批
        approval = interrupt({
            "type": "approval_request",
            "action": action,
            "question": f"是否批准操作: {action}?"
        })

        # 恢复时,approval 会是 True 或 False
        if approval:
            return {
                "messages": [AIMessage(f"操作已批准: {action}")],
                "action_description": action
            }
        else:
            return {
                "messages": [AIMessage(f"操作被拒绝: {action}")]
            }

    # 非敏感操作直接通过
    return {"messages": [AIMessage(f"自动执行: {action}")]}

# 构建图
workflow = StateGraph(ApprovalState)
workflow.add_node("request", request_approval)
workflow.set_entry_point("request")
workflow.add_edge("request", END)

app = workflow.compile(checkpointer=InMemorySaver())

# 使用示例
config = {"configurable": {"thread_id": "approval-002"}}

# 第一次执行 - 会在 interrupt() 处暂停
result = app.invoke(
    {"messages": [HumanMessage("删除用户数据")]},
    config=config
)

# 人工审批
from langgraph.types import Command
# 批准
app.invoke(Command(resume=True), config=config)
# 或拒绝
# app.invoke(Command(resume=False), config=config)
```

#### 3.2.2 审批流程设计（等待、批准、修改、恢复）

**完整的审批系统**

```python
from enum import Enum
from langgraph.types import interrupt, Command

class ApprovalAction(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"

class ApprovalWorkflowState(MessagesState):
    original_request: str
    modified_request: str

def agent_node(state):
    """Agent 节点 - 生成需要审批的操作"""
    model = ChatOpenAI(model="gpt-4")
    response = model.invoke(state["messages"])

    # 检查是否需要审批
    if needs_approval(response.content):
        # 使用 interrupt() 请求审批
        approval_result = interrupt({
            "type": "approval_required",
            "request": response.content,
            "options": ["approve", "reject", "modify"]
        })

        # approval_result 格式: {"action": "approve"} 或 {"action": "modify", "new_request": "..."}
        if approval_result["action"] == "approve":
            return {
                "messages": [response, AIMessage("✅ 请求已批准,正在执行...")],
                "original_request": response.content
            }
        elif approval_result["action"] == "modify":
            # 使用修改后的请求
            new_request = approval_result.get("new_request", response.content)
            return {
                "messages": [AIMessage(f"📝 请求已修改: {new_request}")],
                "original_request": response.content,
                "modified_request": new_request
            }
        else:  # reject
            return {
                "messages": [AIMessage("❌ 请求被拒绝")]
            }

    # 不需要审批的操作直接执行
    return {"messages": [response]}

def execute_node(state):
    """执行节点"""
    request = state.get("modified_request") or state.get("original_request")
    # 执行实际操作
    result = f"已执行操作: {request}"
    return {"messages": [AIMessage(result)]}

# 构建图
workflow = StateGraph(ApprovalWorkflowState)
workflow.add_node("agent", agent_node)
workflow.add_node("execute", execute_node)

workflow.set_entry_point("agent")
workflow.add_edge("agent", "execute")
workflow.add_edge("execute", END)

app = workflow.compile(checkpointer=InMemorySaver())

# 使用示例
config = {"configurable": {"thread_id": "approval-003"}}

# 第一次调用 - 在 interrupt() 处暂停
result = app.invoke(
    {"messages": [HumanMessage("删除生产数据库")]},
    config=config
)

# 选项1: 批准
app.invoke(Command(resume={"action": "approve"}), config=config)

# 选项2: 拒绝
# app.invoke(Command(resume={"action": "reject"}), config=config)

# 选项3: 修改后批准
# app.invoke(Command(resume={
#     "action": "modify",
#     "new_request": "备份后删除测试数据库"
# }), config=config)
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

### 3.3 实战：自定义 Agent

#### 3.3.1 自定义 RAG Agent 完整示例

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

#### 3.3.2 Graph 调试技巧（可视化、日志、测试）

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

### 本章小结

本章学习了持久化与 Human-in-the-Loop：

#### 核心概念

1. **Thread 管理**
   - Thread ID：会话隔离
   - 最佳实践：命名规范
   - 时间旅行：状态快照、回溯

2. **中断与审批**
   - interrupt() 函数：暂停执行等待输入
   - Command(resume=value)：恢复执行并传值
   - 审批流程：批准、拒绝、修改请求

3. **实战：自定义 Agent**
   - RAG Agent 完整实现
   - 调试技巧：可视化、日志、测试、性能分析

#### 下一步

在第四篇中，学习 **Middleware 工程化**，掌握：
- Middleware Hook 体系
- 内置 Middleware（PII、Summarization、Human-in-the-Loop）
- 自定义 Middleware 开发

---

### 思考与练习

#### 思考题

1. Thread ID 的作用是什么？如何设计 Thread ID 命名规范？
2. interrupt() 函数如何工作？如何实现审批流程？
3. 时间旅行调试有什么用？如何回到历史状态？
4. 如何测试 LangGraph 应用？

#### 练习题

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

---
## 第4章：Time Travel - 状态回溯与调试

> **关注点**：掌握LangGraph的杀手级调试特性，重现和修复生产问题

在生产环境中，Agent的执行过程充满了不确定性：
- 🐛 用户报告："你的Agent在第5步给了错误答案"
- 🔄 需求变更："能不能测试一下如果第3步选择了不同的工具会怎样？"
- 🧪 A/B测试："对比两种决策策略，哪个更好？"

**传统调试方式的困境**：
```python
# 问题：用户说Agent第5步出错了
# 传统做法：
# 1. 重新运行整个流程（费时费钱）
# 2. 可能无法复现（LLM不确定性）
# 3. 无法精确定位问题点
# 4. 无法测试修复方案
```

**Time Travel 的价值**：
```python
# Time Travel 做法：
# 1. 回溯到第4步的状态
# 2. 查看当时的上下文
# 3. 修改状态/Prompt
# 4. 从第5步重新执行
# 5. 对比修复前后的差异
```

这就是 LangGraph **Time Travel** 的核心价值：**让Agent的执行过程可重现、可回溯、可修改**。

---

### 4.1 Time Travel 核心概念

#### 4.1.1 什么是 Time Travel

**定义**：Time Travel 是 LangGraph 提供的状态回溯机制，允许你：
- 查看 Agent 历史执行的每一步状态
- 回溯到任意历史节点
- 修改状态后重新执行
- 对比不同决策路径的结果

**前置条件**：
```python
# Time Travel 依赖 Checkpointer（持久化）
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

graph = StateGraph(...).compile(
    checkpointer=checkpointer  # ✅ 必需
)
```

#### 4.1.2 核心 API

**1. get_state_history() - 获取历史状态**

```python
from langchain.graph import StateGraph, MessagesState

# 执行 Agent
config = {"configurable": {"thread_id": "user-123"}}
result = graph.invoke({"messages": [("user", "帮我查询天气")]}, config)

# 获取历史状态
history = list(graph.get_state_history(config))

print(f"总共 {len(history)} 个状态节点")

# 遍历历史
for i, state in enumerate(history):
    print(f"\n=== 状态 {i} ===")
    print(f"节点: {state.metadata.get('langgraph_node', 'START')}")
    print(f"步数: {state.metadata.get('step', 0)}")
    print(f"Checkpoint ID: {state.config['configurable']['checkpoint_id']}")
    print(f"消息数: {len(state.values.get('messages', []))}")
```

**输出示例**：
```
总共 7 个状态节点

=== 状态 0 ===
节点: agent
步数: 3
Checkpoint ID: 1e4f8b3c-...
消息数: 6
(最新状态)

=== 状态 1 ===
节点: tools
步数: 2
Checkpoint ID: 2a9d7c1e-...
消息数: 5
(工具执行后)

=== 状态 2 ===
节点: agent
步数: 1
Checkpoint ID: 3f5e2a8d-...
消息数: 3
(首次模型调用后)

...
```

**2. update_state() - 状态回溯与修改**

```python
# 回溯到特定状态并修改
checkpoint_id = history[3].config['configurable']['checkpoint_id']

# 修改状态（注入不同的值）
graph.update_state(
    config={"configurable": {"thread_id": "user-123", "checkpoint_id": checkpoint_id}},
    values={
        "messages": [
            ("system", "你现在更加谨慎，会仔细验证工具结果")
        ]
    }
)

# 从该状态重新执行
new_result = graph.invoke(None, config={"configurable": {"thread_id": "user-123"}})
```

---

### 4.2 Time Travel 实战场景

#### 4.2.1 场景1：调试生产问题

**问题描述**：用户反馈 Agent 在某次对话的第3步给出了错误信息。

**传统调试困境**：
- 无法重现（LLM 不确定性）
- 用户只记得"第3步出错"，不记得具体输入
- 重新运行要花钱（API调用）

**Time Travel 解决方案**：

```python
from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool

# 工具定义
@tool
def get_weather(city: str) -> str:
    """获取天气信息"""
    # 模拟API错误
    if city == "北京":
        return "错误：API超时"  # 这是问题所在
    return f"{city}今天晴朗，25°C"

# 创建 Agent Graph
checkpointer = MemorySaver()

graph = create_agent(
    ChatOpenAI(model="gpt-4o"),
    tools=[get_weather],
    system_prompt="你是天气助手"
).compile(checkpointer=checkpointer)

# 步骤1：重现用户问题（使用相同的 thread_id）
config = {"configurable": {"thread_id": "reported-issue-123"}}

# 假设这是用户当时的执行（已经完成，保存在Checkpointer中）
# result = graph.invoke({"messages": [("user", "北京天气如何？")]}, config)

# 步骤2：Time Travel 调试
print("=== 获取历史状态 ===\n")
history = list(graph.get_state_history(config))

for i, state in enumerate(reversed(history)):  # 从最早到最新
    step = state.metadata.get('step', 0)
    node = state.metadata.get('langgraph_node', 'START')
    messages = state.values.get('messages', [])

    print(f"步骤 {step} [{node}]:")
    if messages:
        last_msg = messages[-1]
        content = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
        print(f"  最后消息: {content[:100]}...")

    # 发现问题
    if "错误：API超时" in str(messages):
        print(f"  ❌ 发现问题：工具返回错误")
        problem_checkpoint = state.config['configurable']['checkpoint_id']

print(f"\n问题出现在 checkpoint: {problem_checkpoint}")

# 步骤3：修改状态重新执行
print("\n=== 修改工具结果重新执行 ===\n")

# 获取问题节点的状态
problem_state = next(s for s in history if s.config['configurable']['checkpoint_id'] == problem_checkpoint)

# 修改消息（替换错误的工具结果）
modified_messages = problem_state.values['messages'][:-1]  # 移除错误消息
modified_messages.append({
    "role": "tool",
    "content": "北京今天晴朗，25°C",  # 修正后的结果
    "name": "get_weather"
})

# 更新状态
graph.update_state(
    config={"configurable": {"thread_id": "reported-issue-123", "checkpoint_id": problem_checkpoint}},
    values={"messages": modified_messages}
)

# 从该点重新执行
fixed_result = graph.invoke(None, config=config)

print("修复后的回答:")
print(fixed_result["messages"][-1].content)
```

**输出示例**：
```
=== 获取历史状态 ===

步骤 1 [agent]:
  最后消息: 我来帮你查询北京的天气。

步骤 2 [tools]:
  最后消息: 错误：API超时
  ❌ 发现问题：工具返回错误

步骤 3 [agent]:
  最后消息: 抱歉，无法获取天气信息...

问题出现在 checkpoint: 2a9d7c1e-...

=== 修改工具结果重新执行 ===

修复后的回答:
北京今天天气晴朗，温度25°C，适合外出活动。
```

---

#### 4.2.2 场景2：A/B 测试不同决策策略

**需求**：对比两种 Prompt 策略，哪个效果更好？

```python
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

# 定义状态
class AgentState(MessagesState):
    iterations: int

# 创建Graph
def agent_node(state: AgentState):
    model = ChatOpenAI(model="gpt-4o")
    response = model.invoke(state["messages"])
    return {
        "messages": state["messages"] + [response],
        "iterations": state.get("iterations", 0) + 1
    }

def should_continue(state: AgentState):
    if state["iterations"] >= 3:
        return "end"
    return "continue"

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "agent", "end": END}
)

checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

# 步骤1：执行策略A（保守策略）
config_a = {"configurable": {"thread_id": "test-a"}}

result_a = graph.invoke({
    "messages": [
        SystemMessage(content="你是一个保守的分析师，总是建议谨慎行动。"),
        ("user", "我应该投资股票吗？")
    ],
    "iterations": 0
}, config_a)

print("=== 策略A（保守）结果 ===")
print(result_a["messages"][-1].content)

# 步骤2：Time Travel - 回到初始状态，测试策略B
history_a = list(graph.get_state_history(config_a))
initial_checkpoint = history_a[-1].config['configurable']['checkpoint_id']  # 最早的状态

# 创建新线程，从相同初始状态开始，使用策略B
config_b = {"configurable": {"thread_id": "test-b"}}

# 注入策略B的系统提示
graph.update_state(
    config=config_b,
    values={
        "messages": [
            SystemMessage(content="你是一个激进的分析师，总是建议积极行动。"),
            ("user", "我应该投资股票吗？")
        ],
        "iterations": 0
    },
    as_node="agent"  # 从agent节点开始
)

result_b = graph.invoke(None, config=config_b)

print("\n=== 策略B（激进）结果 ===")
print(result_b["messages"][-1].content)

# 步骤3：对比分析
print("\n=== 对比分析 ===")
print(f"策略A迭代次数: {result_a['iterations']}")
print(f"策略B迭代次数: {result_b['iterations']}")

# 可以进一步用 LLM 评估两个答案的质量
```

---

#### 4.2.3 场景3：回溯修复错误决策

**场景**：Agent在中间步骤选择了错误的工具，导致后续偏离方向。

```python
from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# 工具定义
@tool
def search_web(query: str) -> str:
    """搜索网页"""
    return f"网页搜索结果：{query}"

@tool
def search_database(query: str) -> str:
    """搜索数据库"""
    return f"数据库搜索结果：{query}"

# 创建Agent
from langchain.agents import create_agent

checkpointer = MemorySaver()

graph = create_agent(
    ChatOpenAI(model="gpt-4o"),
    tools=[search_web, search_database],
    system_prompt="你是数据分析助手，优先使用数据库搜索内部数据。"
).compile(checkpointer=checkpointer)

# 执行
config = {"configurable": {"thread_id": "debug-session"}}
result = graph.invoke({
    "messages": [("user", "查询我们公司去年的销售额")]
}, config)

print("=== 初次执行结果 ===")
print(result["messages"][-1].content)

# 假设Agent错误地使用了search_web而不是search_database

# Time Travel 调试
print("\n=== Time Travel 查找问题 ===")
history = list(graph.get_state_history(config))

for state in reversed(history):
    messages = state.values.get('messages', [])
    for msg in messages:
        if hasattr(msg, 'additional_kwargs'):
            tool_calls = msg.additional_kwargs.get('tool_calls', [])
            for tc in tool_calls:
                func_name = tc['function']['name']
                if func_name == 'search_web':
                    print(f"❌ 发现错误：使用了 {func_name}，应该用 search_database")
                    wrong_checkpoint = state.config['configurable']['checkpoint_id']

# 回溯并修改
print("\n=== 修改决策重新执行 ===")

# 获取错误决策前的状态
wrong_state = next(s for s in history if s.config['configurable']['checkpoint_id'] == wrong_checkpoint)

# 手动注入正确的工具调用结果
corrected_messages = wrong_state.values['messages'][:-1]  # 移除错误的工具调用
corrected_messages.append({
    "role": "tool",
    "content": "数据库搜索结果：公司去年销售额为5000万元",
    "name": "search_database"
})

graph.update_state(
    config={"configurable": {"thread_id": "debug-session", "checkpoint_id": wrong_checkpoint}},
    values={"messages": corrected_messages}
)

# 重新执行
fixed_result = graph.invoke(None, config=config)

print("修复后的结果:")
print(fixed_result["messages"][-1].content)
```

---

### 4.3 Time Travel 高级技巧

#### 4.3.1 并行测试多个分支

```python
from concurrent.futures import ThreadPoolExecutor

def test_branch(branch_name: str, modified_values: dict):
    """测试一个分支"""
    config = {"configurable": {"thread_id": f"branch-{branch_name}"}}

    # 注入修改后的值
    graph.update_state(
        config=config,
        values=modified_values,
        as_node="agent"
    )

    # 执行
    result = graph.invoke(None, config=config)

    return branch_name, result

# 并行测试3个不同的策略
branches = {
    "conservative": {
        "messages": [SystemMessage(content="保守策略：...")]
    },
    "moderate": {
        "messages": [SystemMessage(content="温和策略：...")]
    },
    "aggressive": {
        "messages": [SystemMessage(content="激进策略：...")]
    }
}

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(test_branch, name, values) for name, values in branches.items()]

    results = {}
    for future in futures:
        branch_name, result = future.result()
        results[branch_name] = result

# 对比结果
for name, result in results.items():
    print(f"\n=== {name} ===")
    print(result["messages"][-1].content[:100])
```

#### 4.3.2 状态快照与恢复

```python
# 保存关键状态快照
def save_snapshot(graph, config, snapshot_name: str):
    """保存状态快照"""
    current_state = graph.get_state(config)

    # 保存到数据库或文件
    snapshot_data = {
        "checkpoint_id": current_state.config['configurable']['checkpoint_id'],
        "values": current_state.values,
        "metadata": current_state.metadata
    }

    # 示例：保存到JSON
    import json
    with open(f"snapshots/{snapshot_name}.json", "w") as f:
        json.dump(snapshot_data, f)

    return snapshot_data

# 恢复快照
def restore_snapshot(graph, snapshot_name: str):
    """恢复快照"""
    import json
    with open(f"snapshots/{snapshot_name}.json", "r") as f:
        snapshot_data = json.load(f)

    config = {"configurable": {
        "thread_id": "restored",
        "checkpoint_id": snapshot_data['checkpoint_id']
    }}

    graph.update_state(
        config=config,
        values=snapshot_data['values']
    )

    return config

# 使用示例
config = {"configurable": {"thread_id": "main-session"}}

# 执行到某个关键点
result = graph.invoke({"messages": [("user", "复杂任务")]}, config)

# 保存快照
save_snapshot(graph, config, "before_critical_decision")

# 继续执行...
# 如果出错，可以随时恢复
restored_config = restore_snapshot(graph, "before_critical_decision")
```

---

### 4.4 Time Travel 最佳实践

#### 4.4.1 生产环境建议

**1. 选择合适的 Checkpointer**

```python
# 开发环境：MemorySaver（快速，但重启丢失）
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()

# 生产环境：PostgresSaver（持久化）
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver.from_conn_string("postgresql://...")

# 高性能场景：RedisSaver
from langgraph.checkpoint.redis import RedisSaver
checkpointer = RedisSaver.from_conn_string("redis://...")
```

**2. Checkpoint 过期策略**

```python
# 配置TTL（生存时间）
from datetime import timedelta

checkpointer = PostgresSaver.from_conn_string(
    "postgresql://...",
    checkpoint_ttl=timedelta(days=7)  # 7天后自动删除
)
```

**3. 监控 Checkpoint 数量**

```python
# 定期清理旧的 checkpoints
def cleanup_old_checkpoints(thread_id: str, keep_last_n: int = 10):
    """只保留最近N个checkpoint"""
    history = list(graph.get_state_history({"configurable": {"thread_id": thread_id}}))

    if len(history) > keep_last_n:
        for old_state in history[keep_last_n:]:
            # 删除旧checkpoint
            checkpointer.delete(old_state.config['configurable']['checkpoint_id'])
```

#### 4.4.2 调试工作流

**标准调试流程**：

```markdown
1. 重现问题
   - 使用相同的 thread_id
   - 获取历史状态

2. 定位问题节点
   - 遍历历史状态
   - 找到异常的输出

3. 回溯修改
   - 回到问题节点前
   - 注入修正后的值

4. 验证修复
   - 重新执行
   - 对比结果

5. 持久化修复
   - 将修复应用到代码
   - 编写回归测试
```

---

### 4.5 Time Travel 与 LangSmith 集成

**结合 LangSmith 可视化调试**：

```python
import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "lsv2_..."

# Time Travel + LangSmith
config = {"configurable": {"thread_id": "debug-with-langsmith"}}

# 执行会自动追踪到 LangSmith
result = graph.invoke({"messages": [("user", "复杂查询")]}, config)

# 在 LangSmith UI 中:
# 1. 查看完整执行树
# 2. 点击任意节点查看状态
# 3. 复制 checkpoint_id

# 在代码中回溯（使用从UI复制的checkpoint_id）
graph.update_state(
    config={"configurable": {"thread_id": "debug-with-langsmith", "checkpoint_id": "从UI复制的ID"}},
    values={...}
)
```

---

### 4.6 小结

**Time Travel 的核心价值**：

| 场景 | 价值 | 传统方式 | Time Travel 方式 |
|------|------|---------|-----------------|
| **生产问题调试** | ⭐⭐⭐⭐⭐ | 重现困难，成本高 | 精确回溯，快速定位 |
| **A/B 测试** | ⭐⭐⭐⭐⭐ | 需要多次完整执行 | 从同一起点分叉测试 |
| **决策优化** | ⭐⭐⭐⭐ | 无法对比历史决策 | 回溯并测试不同路径 |
| **错误恢复** | ⭐⭐⭐⭐ | 从头开始 | 从错误点修复 |
| **学习分析** | ⭐⭐⭐ | 只能看日志 | 完整状态回放 |

**关键要点**：
1. ✅ Time Travel 依赖 Checkpointer，生产环境用 PostgresSaver/RedisSaver
2. ✅ `get_state_history()` 获取历史，`update_state()` 回溯修改
3. ✅ 适合调试、A/B测试、决策优化
4. ✅ 结合 LangSmith 可视化效果更好
5. ⚠️ 注意 Checkpoint 数量，定期清理避免存储膨胀

**下一步**：
- 第5章：LangGraph 工程化实践（langgraph.json、langgraph dev、测试）
- 第6章：Functional API（函数式编程风格）

---

**练习**：

1. **基础练习**：
   - 创建一个简单的 Agent
   - 执行并保存多个 checkpoint
   - 使用 `get_state_history()` 查看历史
   - 回溯到第2步重新执行

2. **进阶练习**：
   - 实现一个 "后悔药" 功能：用户可以说 "返回上一步"
   - 保存关键决策点的快照
   - 并行测试3种不同的策略

3. **生产场景**：
   - 模拟生产问题：Agent 调用了错误的工具
   - 使用 Time Travel 定位问题
   - 修改工具结果重新执行
   - 将修复持久化到代码

---
