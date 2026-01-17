# 第九篇 高级应用与多Agent架构

> **目标**: 掌握前沿技术和复杂场景

本篇深入 LangChain 生态的高级能力：多 Agent 协作、多模态处理、MCP 集成等前沿技术，让你能够构建更复杂、更强大的 AI 应用。

---

## 第1章： 多 Agent 协作

> **关注点**：理解不同的协作模式，掌握 LangGraph 的多 Agent 架构设计。

### 1.1 协作模式

#### 1.1.1 Supervisor-Worker 模式

**架构特点**：

- **中央监督者（Supervisor）**：负责接收用户输入、分解任务、分配工作、汇总结果
- **专门化工作者（Workers）**：每个工作者专注特定任务（如搜索、分析、代码生成）
- **单点控制**：所有通信必须经过 Supervisor，Workers 之间不直接通信

**优势**：控制流清晰、易于调试、责任明确

**实现示例**：

```python
from langgraph.graph import StateGraph, MessagesState, END
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from typing import Literal, TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator

# 定义共享状态
class SupervisorState(MessagesState):
    """多 Agent 共享状态"""
    next_agent: str  # 下一个要执行的 Agent

# 创建工作者 Agents
def create_search_agent():
    """搜索专家 Agent"""
    return create_agent(
        ChatOpenAI(model="gpt-4o-mini"),
        [search_tool],
        prompt="你是搜索专家，负责查找相关信息。"
    )

def create_analyst_agent():
    """分析专家 Agent"""
    return create_agent(
        ChatOpenAI(model="gpt-4o-mini"),
        [analyze_tool],
        prompt="你是数据分析专家，负责分析和总结信息。"
    )

def create_writer_agent():
    """写作专家 Agent"""
    return create_agent(
        ChatOpenAI(model="gpt-4o-mini"),
        [],
        prompt="你是写作专家，负责生成高质量的文档。"
    )

# 创建 Supervisor
class Supervisor:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o")
        self.workers = ["search", "analyst", "writer"]

    def __call__(self, state: SupervisorState) -> SupervisorState:
        """决定下一个执行的 Agent"""
        messages = state["messages"]

        # 构建系统提示词
        system_prompt = f"""你是团队监督者。

可用的工作者：
- search: 搜索信息
- analyst: 分析数据
- writer: 撰写报告

根据当前任务状态，决定下一个应该执行的工作者。
如果任务已完成，返回 "FINISH"。
"""

        # 让 LLM 决定下一步
        response = self.model.invoke([
            {"role": "system", "content": system_prompt},
            *messages
        ])

        # 解析决策（实际应用中需要更严格的解析）
        next_agent = response.content.strip()

        return {"messages": messages + [response], "next_agent": next_agent}

# 构建工作流图
def build_supervisor_workflow():
    """构建 Supervisor-Worker 工作流"""
    workflow = StateGraph(SupervisorState)

    # 添加节点
    workflow.add_node("supervisor", Supervisor())
    workflow.add_node("search", create_search_agent())
    workflow.add_node("analyst", create_analyst_agent())
    workflow.add_node("writer", create_writer_agent())

    # 定义路由逻辑
    def route_next(state: SupervisorState) -> Literal["search", "analyst", "writer", "FINISH"]:
        """根据 next_agent 决定路由"""
        return state.get("next_agent", "FINISH")

    # 添加条件边
    workflow.add_conditional_edges(
        "supervisor",
        route_next,
        {
            "search": "search",
            "analyst": "analyst",
            "writer": "writer",
            "FINISH": END
        }
    )

    # Workers 完成后返回 Supervisor
    workflow.add_edge("search", "supervisor")
    workflow.add_edge("analyst", "supervisor")
    workflow.add_edge("writer", "supervisor")

    # 设置入口点
    workflow.set_entry_point("supervisor")

    return workflow.compile()

# 使用
app = build_supervisor_workflow()
result = app.invoke({
    "messages": [("user", "研究 LangChain 1.0 的新特性并写一份报告")]
})
```

#### 1.1.2 Router 模式

**架构特点**：

- **智能路由器**：根据输入特征动态选择合适的 Agent
- **无中央控制**：直接路由到目标 Agent，减少中间层
- **适合场景明确的任务**：如意图分类后的专门处理

**实现示例**：

```python
from langgraph.graph import StateGraph, MessagesState, END
from typing import Literal

class RouterState(MessagesState):
    """路由状态"""
    intent: str  # 意图分类

def classify_intent(state: RouterState) -> RouterState:
    """意图分类路由器"""
    last_message = state["messages"][-1].content.lower()

    # 简单的规则分类（实际应用中使用 LLM 或分类器）
    if "code" in last_message or "编程" in last_message:
        intent = "coding"
    elif "search" in last_message or "查找" in last_message:
        intent = "search"
    elif "translate" in last_message or "翻译" in last_message:
        intent = "translation"
    else:
        intent = "general"

    return {"messages": state["messages"], "intent": intent}

def build_router_workflow():
    """构建路由模式工作流"""
    workflow = StateGraph(RouterState)

    # 添加路由器
    workflow.add_node("router", classify_intent)

    # 添加专门化 Agents
    workflow.add_node("coding_agent", create_coding_agent())
    workflow.add_node("search_agent", create_search_agent())
    workflow.add_node("translation_agent", create_translation_agent())
    workflow.add_node("general_agent", create_general_agent())

    # 路由逻辑
    def route(state: RouterState) -> str:
        intent_map = {
            "coding": "coding_agent",
            "search": "search_agent",
            "translation": "translation_agent",
            "general": "general_agent"
        }
        return intent_map.get(state["intent"], "general_agent")

    # 添加条件边
    workflow.add_conditional_edges("router", route)

    # 设置入口和出口
    workflow.set_entry_point("router")

    # 所有 Agents 都连接到 END
    for agent in ["coding_agent", "search_agent", "translation_agent", "general_agent"]:
        workflow.add_edge(agent, END)

    return workflow.compile()
```

#### 1.1.3 Hierarchical 模式

**架构特点**：

- **多层级管理**：团队领导管理子团队，形成树状结构
- **责任链**：任务逐级分解，结果逐级汇总
- **适合复杂项目**：如大型研究、软件开发项目

**实现示例**：

```python
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from typing import Dict, List, Literal
import json

# 注意：create_team_supervisor 已移至独立包 langgraph-supervisor
# 官方现在推荐使用工具调用方式实现 supervisor
# 如需使用：pip install langgraph-supervisor

def create_supervisor_with_tools(workers: List[str], system_prompt: str):
    """使用工具调用方式创建 Supervisor（官方推荐方式）"""

    # 创建路由工具
    @tool
    def route_to_worker(next_worker: Literal["search", "analyst", "writer", "FINISH"]) -> str:
        """路由到指定的工作者或结束

        Args:
            next_worker: 下一个要执行的工作者，或 FINISH 表示完成
        """
        return next_worker

    # 创建 supervisor agent
    supervisor = create_agent(
        ChatOpenAI(model="gpt-4o"),
        [route_to_worker],
        prompt=system_prompt
    )

    return supervisor

# 定义团队结构
class ResearchTeam:
    """研究团队"""
    def __init__(self):
        self.lead = create_agent(
            ChatOpenAI(model="gpt-4o"),
            [],
            prompt="你是研究团队负责人，协调数据和市场研究员。"
        )

        self.data_researcher = create_agent(
            ChatOpenAI(model="gpt-4o-mini"),
            [data_analysis_tool],
            prompt="你是数据研究员，负责数据收集和分析。"
        )

        self.market_researcher = create_agent(
            ChatOpenAI(model="gpt-4o-mini"),
            [market_research_tool],
            prompt="你是市场研究员，负责市场趋势分析。"
        )

    def create_subgraph(self):
        """创建研究团队的子图"""
        graph = StateGraph(MessagesState)

        # 添加节点
        graph.add_node("team_lead", self.lead)
        graph.add_node("data_researcher", self.data_researcher)
        graph.add_node("market_researcher", self.market_researcher)

        # 团队负责人决定任务分配
        graph.add_conditional_edges(
            "team_lead",
            lambda x: x.get("next_worker"),
            {
                "data": "data_researcher",
                "market": "market_researcher",
                "done": END
            }
        )

        # 研究员完成后回到负责人
        graph.add_edge("data_researcher", "team_lead")
        graph.add_edge("market_researcher", "team_lead")

        graph.set_entry_point("team_lead")

        return graph.compile()

class DevelopmentTeam:
    """开发团队"""
    def __init__(self):
        self.lead = create_agent(
            ChatOpenAI(model="gpt-4o"),
            [],
            prompt="你是开发团队负责人，协调前端和后端开发。"
        )

        self.frontend_dev = create_agent(
            ChatOpenAI(model="gpt-4o-mini"),
            [code_generation_tool],
            prompt="你是前端开发工程师。"
        )

        self.backend_dev = create_agent(
            ChatOpenAI(model="gpt-4o-mini"),
            [code_generation_tool, database_tool],
            prompt="你是后端开发工程师。"
        )

    def create_subgraph(self):
        """创建开发团队的子图"""
        # 类似研究团队的实现
        pass

def build_hierarchical_system():
    """构建层级化多 Agent 系统"""
    main_graph = StateGraph(MessagesState)

    # CEO Agent
    ceo = create_agent(
        ChatOpenAI(model="gpt-4o"),
        [],
        prompt="""你是项目 CEO，负责整体协调。

你管理两个团队：
- 研究团队：负责市场和数据研究
- 开发团队：负责产品开发

根据任务需求，分配给合适的团队。"""
    )

    # 创建团队子图
    research_team = ResearchTeam().create_subgraph()
    dev_team = DevelopmentTeam().create_subgraph()

    # 添加到主图
    main_graph.add_node("ceo", ceo)
    main_graph.add_node("research_team", research_team)
    main_graph.add_node("dev_team", dev_team)

    # CEO 决定分配给哪个团队
    def route_to_team(state):
        # 根据状态决定路由
        if "research" in state.get("task_type", ""):
            return "research_team"
        elif "develop" in state.get("task_type", ""):
            return "dev_team"
        else:
            return END

    main_graph.add_conditional_edges("ceo", route_to_team)

    # 团队完成后返回 CEO
    main_graph.add_edge("research_team", "ceo")
    main_graph.add_edge("dev_team", "ceo")

    main_graph.set_entry_point("ceo")

    return main_graph.compile()
```

#### 1.1.4 模式选择指南

| 模式 | 适用场景 | 优势 | 劣势 |
|------|---------|------|------|
| **Supervisor-Worker** | 任务需要多步骤协调 | 控制流清晰、易调试 | 中央瓶颈、延迟较高 |
| **Router** | 任务类型明确、独立 | 低延迟、直接处理 | 缺少协调、不适合复杂任务 |
| **Hierarchical** | 大型复杂项目 | 可扩展、责任明确 | 复杂度高、调试困难 |
| **Hybrid** | 混合需求 | 灵活、可根据需求组合 | 需要更多设计考虑 |

**选择决策树**：

```python
def choose_pattern(task_complexity: str, coordination_needed: bool, team_size: int) -> str:
    """选择合适的多 Agent 模式"""

    if task_complexity == "simple":
        if coordination_needed:
            return "Supervisor-Worker（简单版）"
        else:
            return "Router"

    elif task_complexity == "medium":
        if team_size <= 5:
            return "Supervisor-Worker"
        else:
            return "Hierarchical（两层）"

    else:  # complex
        if team_size > 10:
            return "Hierarchical（多层）"
        else:
            return "Supervisor-Worker + Router 混合"
```

---

### 1.2 通信与协调

#### 1.2.1 消息传递机制

**LangGraph 中的消息传递**：

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import MessagesState
from typing import List, TypedDict, Annotated
import operator

# 方式 1：使用内置 MessagesState
class TeamState(MessagesState):
    """使用内置消息状态"""
    task_status: str
    results: List[dict]

# 方式 2：自定义消息传递
class CustomState(TypedDict):
    """自定义状态"""
    messages: Annotated[List[BaseMessage], operator.add]  # 消息追加
    shared_memory: dict  # 共享内存
    agent_outputs: dict  # 各 Agent 输出

def agent_a(state: CustomState) -> CustomState:
    """Agent A 处理并传递消息"""
    # 读取消息
    last_message = state["messages"][-1]

    # 处理
    result = process_message(last_message)

    # 更新共享内存
    state["shared_memory"]["agent_a_result"] = result

    # 添加消息给下一个 Agent
    new_message = AIMessage(
        content=f"Agent A 完成处理：{result}",
        metadata={"agent": "agent_a", "timestamp": datetime.now()}
    )

    return {
        "messages": [new_message],
        "agent_outputs": {"agent_a": result}
    }
```

**消息格式标准化**：

```python
from pydantic import BaseModel
from datetime import datetime

class AgentMessage(BaseModel):
    """标准化的 Agent 消息"""
    sender: str          # 发送者 Agent ID
    receiver: str        # 接收者 Agent ID（"all" 表示广播）
    content: str         # 消息内容
    message_type: str    # "task", "result", "error", "info"
    timestamp: datetime
    metadata: dict = {}

def create_message(sender: str, receiver: str, content: str, msg_type: str = "info"):
    """创建标准消息"""
    return AgentMessage(
        sender=sender,
        receiver=receiver,
        content=content,
        message_type=msg_type,
        timestamp=datetime.now()
    )

# 使用示例
message = create_message(
    sender="supervisor",
    receiver="search_agent",
    content="搜索 LangChain 1.0 新特性",
    msg_type="task"
)
```

#### 1.2.2 共享状态管理

**状态同步策略**：

```python
from typing import Any
import threading
from collections import defaultdict

class SharedStateManager:
    """共享状态管理器"""

    def __init__(self):
        self._state = {}
        self._locks = defaultdict(threading.Lock)
        self._version = defaultdict(int)

    def get(self, key: str) -> Any:
        """读取状态"""
        with self._locks[key]:
            return self._state.get(key)

    def set(self, key: str, value: Any) -> int:
        """设置状态"""
        with self._locks[key]:
            self._state[key] = value
            self._version[key] += 1
            return self._version[key]

    def update(self, key: str, updater: callable) -> Any:
        """原子更新"""
        with self._locks[key]:
            old_value = self._state.get(key)
            new_value = updater(old_value)
            self._state[key] = new_value
            self._version[key] += 1
            return new_value

    def get_version(self, key: str) -> int:
        """获取版本号（用于检测变化）"""
        return self._version[key]

# 在 LangGraph 中使用
class MultiAgentState(TypedDict):
    """多 Agent 共享状态"""
    messages: List[BaseMessage]
    shared_data: SharedStateManager
    agent_status: dict  # {agent_id: "idle" | "working" | "done"}

def worker_agent(state: MultiAgentState) -> MultiAgentState:
    """工作者 Agent"""
    shared = state["shared_data"]

    # 读取共享数据
    task_queue = shared.get("task_queue") or []

    if task_queue:
        # 原子操作：取出任务
        task = shared.update(
            "task_queue",
            lambda queue: (queue.pop(0), queue)[0] if queue else None
        )

        # 处理任务
        result = process_task(task)

        # 更新结果
        shared.update(
            "results",
            lambda results: (results or []) + [result]
        )

    return state
```

#### 1.2.3 任务分解与结果聚合

**任务分解策略**：

```python
from typing import List, Dict

class TaskDecomposer:
    """任务分解器"""

    def __init__(self, model: ChatOpenAI):
        self.model = model

    def decompose(self, task: str) -> List[Dict]:
        """将复杂任务分解为子任务"""

        prompt = f"""将以下任务分解为可独立执行的子任务：

任务：{task}

要求：
1. 每个子任务应该独立可执行
2. 标明子任务之间的依赖关系
3. 估计每个子任务的复杂度（简单/中等/复杂）

返回 JSON 格式：
[
    {{
        "id": "task_1",
        "description": "子任务描述",
        "dependencies": [],
        "complexity": "simple",
        "assigned_to": null
    }}
]
"""

        response = self.model.invoke(prompt)
        # 解析 JSON（实际应用中需要更严格的解析）
        subtasks = json.loads(response.content)

        return subtasks

# 结果聚合
class ResultAggregator:
    """结果聚合器"""

    def __init__(self):
        self.results = {}
        self.dependencies = {}

    def add_result(self, task_id: str, result: Any):
        """添加子任务结果"""
        self.results[task_id] = result

        # 检查是否可以触发依赖任务
        self._check_dependencies(task_id)

    def _check_dependencies(self, completed_task_id: str):
        """检查并触发依赖任务"""
        for task_id, deps in self.dependencies.items():
            if completed_task_id in deps:
                deps.remove(completed_task_id)

                if not deps:  # 所有依赖都满足
                    print(f"Task {task_id} ready to execute")

    def aggregate(self) -> Dict:
        """聚合所有结果"""
        # 简单聚合
        aggregated = {
            "total_tasks": len(self.results),
            "results": self.results,
            "summary": self._generate_summary()
        }

        return aggregated

    def _generate_summary(self) -> str:
        """生成摘要"""
        # 可以使用 LLM 生成智能摘要
        return f"完成 {len(self.results)} 个任务"

# 使用示例
decomposer = TaskDecomposer(ChatOpenAI(model="gpt-4o"))
aggregator = ResultAggregator()

# 分解任务
subtasks = decomposer.decompose("构建一个电商网站")

# 分配给不同 Agents 并收集结果
for task in subtasks:
    # 分配任务
    agent = assign_agent(task)
    result = agent.execute(task)

    # 收集结果
    aggregator.add_result(task["id"], result)

# 最终聚合
final_result = aggregator.aggregate()
```

#### 1.2.4 冲突解决策略

**冲突类型与解决方案**：

```python
from enum import Enum

class ConflictType(Enum):
    """冲突类型"""
    RESOURCE = "resource"      # 资源竞争
    DECISION = "decision"      # 决策分歧
    PRIORITY = "priority"      # 优先级冲突
    DATA = "data"             # 数据不一致

class ConflictResolver:
    """冲突解决器"""

    def __init__(self, arbitrator_model: ChatOpenAI):
        self.arbitrator = arbitrator_model
        self.resolution_history = []

    def resolve(self, conflict_type: ConflictType, parties: List[str], context: dict) -> dict:
        """解决冲突"""

        if conflict_type == ConflictType.RESOURCE:
            return self._resolve_resource_conflict(parties, context)

        elif conflict_type == ConflictType.DECISION:
            return self._resolve_decision_conflict(parties, context)

        elif conflict_type == ConflictType.PRIORITY:
            return self._resolve_priority_conflict(parties, context)

        else:  # DATA
            return self._resolve_data_conflict(parties, context)

    def _resolve_resource_conflict(self, parties: List[str], context: dict) -> dict:
        """解决资源竞争"""
        # 策略：基于优先级和等待时间
        priorities = context.get("priorities", {})
        wait_times = context.get("wait_times", {})

        # 计算得分
        scores = {}
        for party in parties:
            priority_score = priorities.get(party, 0) * 10
            wait_score = wait_times.get(party, 0)
            scores[party] = priority_score + wait_score

        # 选择得分最高的
        winner = max(scores, key=scores.get)

        resolution = {
            "winner": winner,
            "reason": f"基于优先级和等待时间",
            "scores": scores
        }

        self.resolution_history.append(resolution)
        return resolution

    def _resolve_decision_conflict(self, parties: List[str], context: dict) -> dict:
        """解决决策分歧（使用 LLM 仲裁）"""

        proposals = context.get("proposals", {})

        prompt = f"""作为仲裁者，请从以下提案中选择最佳方案：

{json.dumps(proposals, indent=2)}

考虑因素：
1. 可行性
2. 效率
3. 成本
4. 风险

返回选择的方案 ID 和理由。"""

        response = self.arbitrator.invoke(prompt)

        return {
            "decision": response.content,
            "arbitrator": "LLM",
            "timestamp": datetime.now()
        }

    def _resolve_priority_conflict(self, parties: List[str], context: dict) -> dict:
        """解决优先级冲突"""
        # 策略：使用预定义的优先级规则
        priority_rules = {
            "safety": 100,
            "user_request": 90,
            "performance": 80,
            "cost": 70
        }

        task_types = context.get("task_types", {})

        sorted_parties = sorted(
            parties,
            key=lambda p: priority_rules.get(task_types.get(p, ""), 0),
            reverse=True
        )

        return {
            "order": sorted_parties,
            "method": "rule-based"
        }

    def _resolve_data_conflict(self, parties: List[str], context: dict) -> dict:
        """解决数据不一致（投票或最新优先）"""
        data_versions = context.get("data_versions", {})

        # 策略 1：最新版本优先
        if "timestamps" in context:
            timestamps = context["timestamps"]
            latest = max(parties, key=lambda p: timestamps.get(p, 0))
            return {
                "selected": latest,
                "method": "latest-wins",
                "data": data_versions[latest]
            }

        # 策略 2：多数投票
        from collections import Counter
        values = [str(data_versions[p]) for p in parties]
        most_common = Counter(values).most_common(1)[0][0]

        winner = [p for p in parties if str(data_versions[p]) == most_common][0]

        return {
            "selected": winner,
            "method": "majority-vote",
            "data": data_versions[winner]
        }

# 使用示例
resolver = ConflictResolver(ChatOpenAI(model="gpt-4o"))

# 资源冲突
resource_conflict = resolver.resolve(
    ConflictType.RESOURCE,
    parties=["agent_a", "agent_b"],
    context={
        "priorities": {"agent_a": 2, "agent_b": 1},
        "wait_times": {"agent_a": 5, "agent_b": 10}
    }
)

print(f"资源分配给: {resource_conflict['winner']}")
```

---

### 1.3 实战案例

#### 1.3.1 数据分析系统

**完整的多 Agent 数据分析系统**：

```python
from langgraph.graph import StateGraph, MessagesState, END
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from typing import TypedDict, List
import pandas as pd

class DataAnalysisState(MessagesState):
    """数据分析系统状态"""
    data_source: str
    raw_data: pd.DataFrame
    cleaned_data: pd.DataFrame
    analysis_results: dict
    visualization_urls: List[str]
    report: str

# 数据收集 Agent
def data_collector_agent(state: DataAnalysisState) -> DataAnalysisState:
    """收集数据"""
    agent = create_agent(
        ChatOpenAI(model="gpt-4o-mini"),
        [sql_query_tool, api_fetch_tool, file_reader_tool],
        prompt="你是数据收集专家，负责从各种源获取数据。"
    )

    result = agent.invoke({"messages": state["messages"]})

    # 假设返回了数据
    raw_data = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=100),
        "sales": np.random.randint(100, 1000, 100),
        "category": np.random.choice(["A", "B", "C"], 100)
    })

    return {
        "messages": state["messages"] + [result["messages"][-1]],
        "raw_data": raw_data
    }

# 数据清洗 Agent
def data_cleaner_agent(state: DataAnalysisState) -> DataAnalysisState:
    """清洗数据"""
    raw_data = state["raw_data"]

    # 清洗逻辑
    cleaned_data = raw_data.dropna()
    cleaned_data = cleaned_data[cleaned_data["sales"] > 0]

    # 添加衍生特征
    cleaned_data["month"] = pd.to_datetime(cleaned_data["date"]).dt.month
    cleaned_data["quarter"] = pd.to_datetime(cleaned_data["date"]).dt.quarter

    message = AIMessage(content=f"数据清洗完成，保留 {len(cleaned_data)} 条有效记录")

    return {
        "messages": state["messages"] + [message],
        "cleaned_data": cleaned_data
    }

# 统计分析 Agent
def statistical_analyst_agent(state: DataAnalysisState) -> DataAnalysisState:
    """统计分析"""
    data = state["cleaned_data"]

    analysis_results = {
        "summary_stats": data.describe().to_dict(),
        "correlation_matrix": data.corr().to_dict(),
        "category_performance": data.groupby("category")["sales"].agg(["mean", "sum", "count"]).to_dict(),
        "trend_analysis": {
            "monthly_avg": data.groupby("month")["sales"].mean().to_dict(),
            "growth_rate": calculate_growth_rate(data)
        }
    }

    message = AIMessage(content=f"统计分析完成，发现 {len(analysis_results)} 项关键洞察")

    return {
        "messages": state["messages"] + [message],
        "analysis_results": analysis_results
    }

# 可视化 Agent
def visualization_agent(state: DataAnalysisState) -> DataAnalysisState:
    """生成可视化"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    data = state["cleaned_data"]
    results = state["analysis_results"]

    urls = []

    # 生成图表
    # 1. 销售趋势图
    plt.figure(figsize=(10, 6))
    data.groupby("date")["sales"].sum().plot()
    plt.title("Sales Trend")
    plt.savefig("/tmp/sales_trend.png")
    urls.append("/tmp/sales_trend.png")

    # 2. 类别对比
    plt.figure(figsize=(8, 6))
    data.groupby("category")["sales"].mean().plot(kind="bar")
    plt.title("Average Sales by Category")
    plt.savefig("/tmp/category_comparison.png")
    urls.append("/tmp/category_comparison.png")

    message = AIMessage(content=f"生成 {len(urls)} 个可视化图表")

    return {
        "messages": state["messages"] + [message],
        "visualization_urls": urls
    }

# 报告生成 Agent
def report_writer_agent(state: DataAnalysisState) -> DataAnalysisState:
    """生成分析报告"""
    agent = create_agent(
        ChatOpenAI(model="gpt-4o"),
        [],
        prompt="""你是数据分析报告专家。

根据提供的分析结果和图表，生成专业的数据分析报告。

报告应包含：
1. 执行摘要
2. 数据概览
3. 关键发现
4. 趋势分析
5. 建议
"""
    )

    # 构建上下文
    context = f"""
分析结果：
{json.dumps(state["analysis_results"], indent=2)}

可视化图表：
{state["visualization_urls"]}
"""

    result = agent.invoke({
        "messages": state["messages"] + [HumanMessage(content=context)]
    })

    report = result["messages"][-1].content

    return {
        "messages": state["messages"] + [result["messages"][-1]],
        "report": report
    }

def build_data_analysis_system():
    """构建数据分析系统"""
    workflow = StateGraph(DataAnalysisState)

    # 添加节点
    workflow.add_node("collector", data_collector_agent)
    workflow.add_node("cleaner", data_cleaner_agent)
    workflow.add_node("analyst", statistical_analyst_agent)
    workflow.add_node("visualizer", visualization_agent)
    workflow.add_node("reporter", report_writer_agent)

    # 定义流程
    workflow.add_edge("collector", "cleaner")
    workflow.add_edge("cleaner", "analyst")
    workflow.add_edge("analyst", "visualizer")
    workflow.add_edge("visualizer", "reporter")
    workflow.add_edge("reporter", END)

    workflow.set_entry_point("collector")

    return workflow.compile()

# 使用
system = build_data_analysis_system()
result = system.invoke({
    "messages": [HumanMessage(content="分析最近三个月的销售数据")],
    "data_source": "sales_database"
})

print(result["report"])
```

#### 1.3.2 智能客服系统

**多层级客服系统实现**：

```python
class CustomerServiceSystem:
    """智能客服系统"""

    def __init__(self):
        from langchain.agents import create_agent

        # 一线客服（处理常见问题）
        self.tier1_agent = create_agent(
            ChatOpenAI(model="gpt-4o-mini"),
            [faq_search_tool, order_status_tool],
            prompt="""你是一线客服代表。

职责：
1. 回答常见问题
2. 查询订单状态
3. 收集用户信息

如果遇到复杂问题，升级给二线客服。"""
        )

        # 二线客服（处理复杂问题）
        self.tier2_agent = create_agent(
            ChatOpenAI(model="gpt-4o"),
            [refund_tool, account_modify_tool, technical_support_tool],
            prompt="""你是二线客服专家。

职责：
1. 处理退款请求
2. 账户问题解决
3. 技术支持

如果需要人工介入，升级给人工客服。"""
        )

        # 情感分析 Agent
        self.sentiment_analyzer = create_agent(
            ChatOpenAI(model="gpt-4o-mini"),
            [],
            prompt="""分析用户情绪。

返回：
- 情绪状态：积极/中性/消极/愤怒
- 紧急程度：低/中/高
- 是否需要立即人工介入"""
        )

    def route_request(self, user_message: str) -> str:
        """智能路由请求"""
        # 先进行情感分析
        sentiment = self.analyze_sentiment(user_message)

        if sentiment["urgent"] or sentiment["emotion"] == "愤怒":
            return "human"  # 直接转人工

        # 意图识别
        intent = self.classify_intent(user_message)

        if intent in ["order_query", "faq", "product_info"]:
            return "tier1"
        elif intent in ["refund", "complaint", "technical"]:
            return "tier2"
        else:
            return "tier1"  # 默认一线

    def handle_conversation(self, messages: List[BaseMessage]) -> dict:
        """处理完整对话"""
        state = {
            "messages": messages,
            "escalation_count": 0,
            "resolved": False
        }

        current_tier = "tier1"

        while not state["resolved"] and state["escalation_count"] < 3:
            if current_tier == "tier1":
                response = self.tier1_agent.invoke({"messages": state["messages"]})

                # 检查是否需要升级
                if "升级" in response["messages"][-1].content:
                    current_tier = "tier2"
                    state["escalation_count"] += 1
                    continue

            elif current_tier == "tier2":
                response = self.tier2_agent.invoke({"messages": state["messages"]})

                # 检查是否需要人工
                if "人工" in response["messages"][-1].content:
                    current_tier = "human"
                    state["escalation_count"] += 1
                    continue

            else:  # human
                response = self.transfer_to_human(state["messages"])

            state["messages"].extend(response["messages"])
            state["resolved"] = True

        return state
```

#### 1.3.3 研究助手系统

**实现完整的研究助手**：

```python
from langgraph.graph import StateGraph, END
from typing import List, Dict
from langchain.agents import create_agent

class ResearchAssistantSystem:
    """研究助手系统"""

    def __init__(self):
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        """构建研究工作流"""
        workflow = StateGraph(ResearchState)

        # 添加专门化研究 Agents
        workflow.add_node("topic_explorer", self.topic_explorer_agent())
        workflow.add_node("literature_reviewer", self.literature_review_agent())
        workflow.add_node("data_analyst", self.data_analysis_agent())
        workflow.add_node("hypothesis_generator", self.hypothesis_agent())
        workflow.add_node("experiment_designer", self.experiment_design_agent())
        workflow.add_node("paper_writer", self.paper_writing_agent())

        # Supervisor 协调
        workflow.add_node("supervisor", self.research_supervisor())

        # 定义流程
        workflow.set_entry_point("supervisor")

        # Supervisor 决定下一步
        workflow.add_conditional_edges(
            "supervisor",
            lambda x: x.get("next_phase"),
            {
                "explore": "topic_explorer",
                "review": "literature_reviewer",
                "analyze": "data_analyst",
                "hypothesize": "hypothesis_generator",
                "design": "experiment_designer",
                "write": "paper_writer",
                "complete": END
            }
        )

        # 所有 Agents 返回 Supervisor
        for node in ["topic_explorer", "literature_reviewer", "data_analyst",
                    "hypothesis_generator", "experiment_designer", "paper_writer"]:
            workflow.add_edge(node, "supervisor")

        return workflow.compile()

    def topic_explorer_agent(self):
        """主题探索 Agent"""
        return create_agent(
            ChatOpenAI(model="gpt-4o"),
            [web_search_tool, arxiv_search_tool],
            prompt="""你是研究主题探索专家。

任务：
1. 识别研究领域的关键问题
2. 发现研究空白
3. 评估研究价值和可行性"""
        )

    def literature_review_agent(self):
        """文献综述 Agent"""
        return create_agent(
            ChatOpenAI(model="gpt-4o"),
            [paper_search_tool, citation_tool, summarize_tool],
            prompt="""你是文献综述专家。

任务：
1. 搜索相关文献
2. 总结关键发现
3. 识别研究趋势
4. 构建知识图谱"""
        )

    def research_supervisor(self):
        """研究监督者"""
        return create_agent(
            ChatOpenAI(model="gpt-4o"),
            [],
            prompt="""你是研究项目总监。

根据研究进展，决定下一个阶段：
- explore: 初步探索
- review: 文献综述
- analyze: 数据分析
- hypothesize: 假设生成
- design: 实验设计
- write: 论文撰写
- complete: 完成

确保研究的科学性和完整性。"""
        )

    def conduct_research(self, topic: str) -> Dict:
        """执行完整研究"""
        initial_state = {
            "messages": [HumanMessage(content=f"研究主题：{topic}")],
            "research_topic": topic,
            "phase": "explore",
            "findings": {},
            "papers": []
        }

        result = self.workflow.invoke(initial_state)

        return {
            "topic": topic,
            "findings": result.get("findings"),
            "papers": result.get("papers"),
            "final_report": result.get("report")
        }

# 使用示例
research_system = ResearchAssistantSystem()
result = research_system.conduct_research("LLM Agent 的未来发展方向")
print(result["final_report"])
```

#### 1.3.4 调试与优化

**多 Agent 系统调试技巧**：

```python
from langsmith import Client
from langchain_core.callbacks import LangChainTracer
import logging

class MultiAgentDebugger:
    """多 Agent 调试器"""

    def __init__(self, project_name: str):
        self.client = Client()
        self.project_name = project_name
        self.tracer = LangChainTracer(project_name=project_name)

        # 配置日志
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)

    def trace_agent_communication(self, workflow: StateGraph):
        """追踪 Agent 间通信"""

        # 注入追踪回调
        for node in workflow.nodes:
            if hasattr(node, "invoke"):
                original_invoke = node.invoke

                def traced_invoke(self, *args, **kwargs):
                    self.logger.debug(f"Agent {node.name} invoked with: {args}")
                    result = original_invoke(*args, **kwargs)
                    self.logger.debug(f"Agent {node.name} returned: {result}")
                    return result

                node.invoke = traced_invoke

    def profile_performance(self, workflow: StateGraph, test_cases: List[dict]):
        """性能分析"""
        results = []

        for case in test_cases:
            start_time = time.time()

            # 运行工作流
            result = workflow.invoke(case)

            duration = time.time() - start_time

            # 收集指标
            metrics = {
                "case_id": case.get("id"),
                "duration": duration,
                "agent_calls": self._count_agent_calls(result),
                "token_usage": self._calculate_tokens(result),
                "cost": self._estimate_cost(result)
            }

            results.append(metrics)

        # 生成报告
        self._generate_performance_report(results)

        return results

    def detect_bottlenecks(self, trace_data: dict):
        """检测性能瓶颈"""
        bottlenecks = []

        # 分析每个 Agent 的执行时间
        agent_times = {}
        for run in trace_data["runs"]:
            agent_name = run["name"]
            duration = run["end_time"] - run["start_time"]

            if agent_name not in agent_times:
                agent_times[agent_name] = []
            agent_times[agent_name].append(duration)

        # 找出最慢的 Agents
        avg_times = {
            agent: sum(times) / len(times)
            for agent, times in agent_times.items()
        }

        slowest = sorted(avg_times.items(), key=lambda x: x[1], reverse=True)[:3]

        for agent, avg_time in slowest:
            if avg_time > 5.0:  # 超过 5 秒
                bottlenecks.append({
                    "agent": agent,
                    "avg_time": avg_time,
                    "severity": "high" if avg_time > 10 else "medium"
                })

        return bottlenecks

    def optimize_suggestions(self, bottlenecks: List[dict]) -> List[str]:
        """优化建议"""
        suggestions = []

        for bottleneck in bottlenecks:
            agent = bottleneck["agent"]
            avg_time = bottleneck["avg_time"]

            if "search" in agent.lower():
                suggestions.append(f"为 {agent} 添加缓存机制")
                suggestions.append(f"考虑并行化 {agent} 的搜索请求")

            elif "model" in agent.lower():
                suggestions.append(f"考虑使用更快的模型替代 {agent}")
                suggestions.append(f"减少 {agent} 的上下文长度")

            if avg_time > 10:
                suggestions.append(f"考虑将 {agent} 的任务分解为更小的子任务")

        return suggestions

# 使用示例
debugger = MultiAgentDebugger("multi_agent_debug")

# 追踪通信
debugger.trace_agent_communication(workflow)

# 性能分析
test_cases = [
    {"messages": [HumanMessage(content="测试查询 1")]},
    {"messages": [HumanMessage(content="测试查询 2")]},
]
metrics = debugger.profile_performance(workflow, test_cases)

# 检测瓶颈
bottlenecks = debugger.detect_bottlenecks(trace_data)
suggestions = debugger.optimize_suggestions(bottlenecks)

for suggestion in suggestions:
    print(f"💡 {suggestion}")
```

### 1.4 人机协同与中断机制

#### 1.4.1 中断点设置

**LangGraph 提供的人机协同能力**:

在实际应用中,某些关键决策点需要人工介入审核或确认。LangGraph 通过 `interrupt_before` 和 `interrupt_after` 实现中断机制。

```python
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# 定义状态
class ApprovalState(MessagesState):
    """需要审批的状态"""
    draft_response: str = ""
    approved: bool = False
    feedback: str = ""

# 创建节点
def generate_draft(state: ApprovalState) -> dict:
    """生成草稿回复"""
    from langchain.agents import create_agent

    agent = create_agent(
        ChatOpenAI(model="gpt-4o"),
        [],
        prompt="生成专业的客户回复草稿"
    )

    result = agent.invoke({"messages": state["messages"]})
    draft = result["messages"][-1].content

    return {"draft_response": draft}

def human_review(state: ApprovalState) -> dict:
    """人工审核节点(这里不执行,由外部处理)"""
    # 这个节点在 interrupt_before 时不会执行
    # 外部通过 update_state 更新状态
    return {}

def send_response(state: ApprovalState) -> dict:
    """发送最终回复"""
    if state.get("approved"):
        final_response = state["draft_response"]
        if state.get("feedback"):
            final_response += f"\n\n补充: {state['feedback']}"

        print(f"发送回复: {final_response}")
        return {"messages": [AIMessage(content=final_response)]}
    else:
        return {"messages": [AIMessage(content="回复已被拒绝")]}

# 构建图
builder = StateGraph(ApprovalState)
builder.add_node("generate_draft", generate_draft)
builder.add_node("human_review", human_review)
builder.add_node("send_response", send_response)

builder.set_entry_point("generate_draft")
builder.add_edge("generate_draft", "human_review")
builder.add_edge("human_review", "send_response")
builder.add_edge("send_response", END)

# 关键: 设置中断点
checkpointer = MemorySaver()
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["human_review"]  # 在人工审核前中断
)
```

#### 1.4.2 状态检查与更新

**完整的中断-审核-恢复流程**:

```python
# 1. 运行到中断点
config = {"configurable": {"thread_id": "approval-001"}}
result = graph.invoke(
    {"messages": [HumanMessage(content="我要投诉你们的服务")]},
    config
)
# 执行会在 human_review 前暂停

# 2. 获取当前状态
state = graph.get_state(config)

print(f"当前状态值: {state.values}")
print(f"下一个节点: {state.next}")  # 输出: ('human_review',)
print(f"checkpoint_id: {state.config['configurable']['checkpoint_id']}")

# StateSnapshot 对象包含:
# - values: dict - 当前状态的所有值
# - next: tuple[str] - 下一个要执行的节点(中断点)
# - config: dict - 包含 checkpoint_id 的配置
# - metadata: dict - 元数据
# - created_at: str - 创建时间
# - parent_config: dict - 父状态配置

# 查看草稿内容
draft = state.values.get("draft_response")
print(f"\n待审核草稿:\n{draft}")

# 3. 人工审核并更新状态
# 场景A: 批准并添加补充
graph.update_state(
    config,
    {
        "approved": True,
        "feedback": "请额外提供补偿方案"
    }
)

# 场景B: 拒绝并要求重写
# graph.update_state(
#     config,
#     {
#         "approved": False,
#         "feedback": "语气太强硬,需要更温和的表达"
#     },
#     as_node="generate_draft"  # 回退到草稿生成节点
# )

# 4. 恢复执行
# 从中断点继续(None 表示不添加新输入)
final_result = graph.invoke(None, config)
print(f"\n最终结果: {final_result}")
```

#### 1.4.3 interrupt_after 用法

**在节点执行后中断**:

```python
# 使用 interrupt_after
graph_after = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_after=["generate_draft"]  # 在草稿生成后中断
)

config = {"configurable": {"thread_id": "after-001"}}

# 运行
result = graph_after.invoke(
    {"messages": [HumanMessage(content="咨询产品信息")]},
    config
)

# generate_draft 已执行完,但在下一个节点前中断
state = graph_after.get_state(config)
print(f"已生成草稿: {state.values['draft_response']}")
print(f"下一个节点: {state.next}")  # ('human_review',)

# 检查草稿后决定是否继续
if "敏感词" in state.values['draft_response']:
    # 修改草稿
    graph_after.update_state(
        config,
        {"draft_response": "修改后的草稿"}
    )

# 继续执行
graph_after.invoke(None, config)
```

#### 1.4.4 多中断点工作流

**复杂审批流程**:

```python
from typing import TypedDict

class MultiApprovalState(MessagesState):
    """多级审批状态"""
    task: str = ""
    budget_estimate: float = 0.0
    manager_approved: bool = False
    director_approved: bool = False
    cfo_approved: bool = False

def estimate_budget(state: MultiApprovalState) -> dict:
    """预算评估"""
    # 模拟预算计算
    return {"budget_estimate": 50000.0}

def manager_review(state: MultiApprovalState) -> dict:
    """经理审批"""
    return {}

def director_review(state: MultiApprovalState) -> dict:
    """总监审批"""
    return {}

def cfo_review(state: MultiApprovalState) -> dict:
    """CFO审批"""
    return {}

def execute_task(state: MultiApprovalState) -> dict:
    """执行任务"""
    print(f"执行任务: {state['task']}, 预算: {state['budget_estimate']}")
    return {}

# 构建多级审批流程
builder = StateGraph(MultiApprovalState)
builder.add_node("estimate", estimate_budget)
builder.add_node("manager", manager_review)
builder.add_node("director", director_review)
builder.add_node("cfo", cfo_review)
builder.add_node("execute", execute_task)

builder.set_entry_point("estimate")
builder.add_edge("estimate", "manager")
builder.add_edge("manager", "director")
builder.add_edge("director", "cfo")
builder.add_edge("cfo", "execute")
builder.add_edge("execute", END)

# 设置多个中断点
graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["manager", "director", "cfo"]  # 每个审批环节都中断
)

# 执行流程
config = {"configurable": {"thread_id": "multi-approval-001"}}

# 步骤1: 运行到经理审批
result = graph.invoke({"task": "购买新服务器"}, config)
state = graph.get_state(config)
print(f"预算: {state.values['budget_estimate']}")
print(f"等待审批: {state.next}")  # ('manager',)

# 经理批准
graph.update_state(config, {"manager_approved": True})
result = graph.invoke(None, config)

# 步骤2: 运行到总监审批
state = graph.get_state(config)
print(f"等待审批: {state.next}")  # ('director',)

# 总监批准
graph.update_state(config, {"director_approved": True})
result = graph.invoke(None, config)

# 步骤3: 运行到CFO审批
state = graph.get_state(config)
print(f"等待审批: {state.next}")  # ('cfo',)

# CFO批准
graph.update_state(config, {"cfo_approved": True})
final_result = graph.invoke(None, config)

print("所有审批完成,任务执行!")
```

#### 1.4.5 最佳实践

**中断机制使用建议**:

1. **选择合适的中断点**:
   - `interrupt_before`: 适合需要预审的场景(如发送邮件前)
   - `interrupt_after`: 适合需要检查结果的场景(如数据处理后)

2. **状态更新策略**:
   ```python
   # 完全替换状态字段
   graph.update_state(config, {"field": "new_value"})

   # 指定从哪个节点继续(回退)
   graph.update_state(
       config,
       {"field": "value"},
       as_node="previous_node"
   )
   ```

3. **超时处理**:
   ```python
   from datetime import datetime, timedelta

   state = graph.get_state(config)
   created_time = datetime.fromisoformat(state.created_at)

   if datetime.now() - created_time > timedelta(hours=24):
       # 超时自动拒绝或提醒
       graph.update_state(config, {"approved": False, "reason": "超时"})
   ```

4. **审计日志**:
   ```python
   # 记录审批历史
   graph.update_state(
       config,
       {
           "approved": True,
           "audit_log": {
               "reviewer": "user@example.com",
               "timestamp": datetime.now().isoformat(),
               "action": "approved",
               "comment": "通过审核"
           }
       }
   )
   ```

---

## 第2章： 高级特性与集成

> **关注点**：掌握 LangChain 的前沿功能，包括多模态、流式处理、MCP 集成等。

### 2.1 多模态应用

#### 2.1.1 图像处理（OCR、图像理解）

**图像理解与工具调用**：

```python
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import base64

def encode_image(image_path: str) -> str:
    """将图像编码为 base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 创建支持视觉的模型
vision_model = ChatOpenAI(model="gpt-4o")

# 发送图像进行理解
image_base64 = encode_image("chart.png")

message = HumanMessage(
    content=[
        {"type": "text", "text": "分析这个图表中的数据趋势"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
        }
    ]
)

response = vision_model.invoke([message])
print(response.content)

# 结合工具调用
from langchain.agents import create_agent
from langchain_core.tools import tool

@tool
def save_analysis(analysis: str, filename: str) -> str:
    """保存分析结果"""
    with open(filename, "w") as f:
        f.write(analysis)
    return f"分析已保存到 {filename}"

# 创建多模态 Agent
multimodal_agent = create_agent(
    vision_model,
    [save_analysis],
    prompt="你是图像分析专家，可以理解图像并使用工具保存分析结果。"
)

# 分析图像并保存
result = multimodal_agent.invoke({
    "messages": [
        HumanMessage(content=[
            {"type": "text", "text": "分析这个图表并保存结果到 analysis.txt"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
        ])
    ]
})
```

**OCR 与文档处理**：

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pytesseract
from PIL import Image

class OCRProcessor:
    """OCR 处理器"""

    def __init__(self, model: ChatOpenAI):
        self.model = model
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def extract_text_from_image(self, image_path: str) -> str:
        """从图像提取文本（OCR）"""
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang='chi_sim+eng')
        return text

    def extract_text_from_pdf(self, pdf_path: str) -> List[str]:
        """从 PDF 提取文本"""
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # 分割文本
        texts = self.text_splitter.split_documents(documents)

        return [doc.page_content for doc in texts]

    def understand_document_with_vision(self, image_path: str) -> dict:
        """使用视觉模型理解文档"""
        image_base64 = encode_image(image_path)

        response = self.model.invoke([
            HumanMessage(content=[
                {"type": "text", "text": "提取并结构化这个文档中的信息"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ])
        ])

        # 解析响应（可以要求返回 JSON）
        return {
            "extracted_text": response.content,
            "structured_data": self._parse_structured_data(response.content)
        }

    def _parse_structured_data(self, text: str) -> dict:
        """解析结构化数据"""
        # 使用 LLM 提取结构化信息
        prompt = f"""从以下文本中提取结构化信息（JSON 格式）：

{text}

返回格式：
{{
    "title": "...",
    "date": "...",
    "key_points": [...],
    "numbers": {{...}}
}}"""

        response = self.model.invoke(prompt)
        # 解析 JSON（实际应用中需要更严格的解析）
        import json
        try:
            return json.loads(response.content)
        except:
            return {"raw_text": response.content}

# 使用示例
processor = OCRProcessor(ChatOpenAI(model="gpt-4o"))

# OCR 提取
text = processor.extract_text_from_image("scanned_document.png")

# 视觉理解
understanding = processor.understand_document_with_vision("complex_chart.png")
print(understanding["structured_data"])
```

#### 2.1.2 音频处理（语音识别、合成）

**音频处理集成**：

```python
from openai import OpenAI
import io
from pydub import AudioSegment
from typing import BinaryIO

class AudioProcessor:
    """音频处理器"""

    def __init__(self):
        self.client = OpenAI()

    def transcribe_audio(self, audio_file: BinaryIO) -> str:
        """语音转文本（使用 Whisper）"""
        transcript = self.client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        return transcript.text

    def text_to_speech(self, text: str, voice: str = "alloy") -> bytes:
        """文本转语音"""
        response = self.client.audio.speech.create(
            model="tts-1",
            voice=voice,  # alloy, echo, fable, onyx, nova, shimmer
            input=text
        )

        return response.content

    def process_audio_with_agent(self, audio_path: str):
        """使用 Agent 处理音频"""
        from langchain.agents import create_agent

        # 转录音频
        with open(audio_path, "rb") as audio_file:
            transcript = self.transcribe_audio(audio_file)

        # 使用 Agent 处理转录文本
        agent = create_agent(
            ChatOpenAI(model="gpt-4o"),
            [search_tool, summarize_tool],
            prompt="处理音频转录并回答问题。"
        )

        response = agent.invoke({
            "messages": [HumanMessage(content=f"音频内容：{transcript}\n\n请总结要点")]
        })

        summary = response["messages"][-1].content

        # 将摘要转为语音
        audio_response = self.text_to_speech(summary)

        return {
            "transcript": transcript,
            "summary": summary,
            "audio_response": audio_response
        }

# 实时音频流处理
class AudioStreamProcessor:
    """实时音频流处理"""

    def __init__(self, agent):
        self.agent = agent
        self.audio_buffer = io.BytesIO()

    async def process_audio_stream(self, audio_chunk: bytes):
        """处理音频流块"""
        # 添加到缓冲区
        self.audio_buffer.write(audio_chunk)

        # 检查是否有足够的音频（例如 3 秒）
        if self.audio_buffer.tell() > 48000:  # 16kHz * 3s
            # 处理缓冲区中的音频
            self.audio_buffer.seek(0)
            transcript = self.transcribe_audio(self.audio_buffer)

            # 清空缓冲区
            self.audio_buffer = io.BytesIO()

            # 使用 Agent 处理
            response = await self.agent.ainvoke({
                "messages": [HumanMessage(content=transcript)]
            })

            return response["messages"][-1].content

        return None
```

**完整的图文混合示例**:

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import base64

# ========== 示例1: 图片理解(本地文件) ==========
def analyze_local_image():
    """分析本地图片"""
    model = ChatOpenAI(model="gpt-4o")

    # 方法1: 使用 base64 编码
    with open("product_image.jpg", "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    message = HumanMessage(
        content=[
            {"type": "text", "text": "请描述这个产品图片,包括颜色、材质和可能的用途"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            }
        ]
    )

    response = model.invoke([message])
    print("图片分析:", response.content)

# ========== 示例2: 在线图片URL ==========
def analyze_url_image():
    """分析在线图片"""
    model = ChatOpenAI(model="gpt-4o")

    message = HumanMessage(
        content=[
            {"type": "text", "text": "这张图片里有什么?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/image.jpg",
                    "detail": "high"  # "low", "high", or "auto"
                }
            }
        ]
    )

    response = model.invoke([message])
    return response.content

# ========== 示例3: 多图片对比 ==========
def compare_images():
    """对比多张图片"""
    model = ChatOpenAI(model="gpt-4o")

    # 编码两张图片
    with open("before.jpg", "rb") as f:
        image1_b64 = base64.b64encode(f.read()).decode('utf-8')

    with open("after.jpg", "rb") as f:
        image2_b64 = base64.b64encode(f.read()).decode('utf-8')

    message = HumanMessage(
        content=[
            {"type": "text", "text": "对比这两张图片的差异:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image1_b64}"}},
            {"type": "text", "text": "和"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image2_b64}"}},
            {"type": "text", "text": "列出所有不同之处"}
        ]
    )

    response = model.invoke([message])
    print("对比结果:", response.content)

# ========== 示例4: 图表数据提取 ==========
def extract_chart_data():
    """从图表中提取数据"""
    model = ChatOpenAI(model="gpt-4o")

    with open("sales_chart.png", "rb") as f:
        chart_b64 = base64.b64encode(f.read()).decode('utf-8')

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": """提取这个销售图表中的数据,返回JSON格式:
                {
                    "months": ["1月", "2月", ...],
                    "sales": [1000, 1200, ...],
                    "trend": "上升/下降/稳定"
                }"""
            },
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{chart_b64}"}}
        ]
    )

    response = model.invoke([message])

    # 解析JSON
    import json
    try:
        data = json.loads(response.content)
        return data
    except json.JSONDecodeError:
        # 如果LLM返回的不是纯JSON,尝试提取
        print("原始响应:", response.content)
        return None

# ========== 示例5: 视觉问答(VQA) ==========
def visual_question_answering():
    """视觉问答"""
    model = ChatOpenAI(model="gpt-4o")

    with open("scene.jpg", "rb") as f:
        scene_b64 = base64.b64encode(f.read()).decode('utf-8')

    questions = [
        "图片中有多少人?",
        "他们在做什么?",
        "场景是室内还是室外?",
        "天气如何?",
        "大概是什么时间?"
    ]

    results = {}
    for question in questions:
        message = HumanMessage(
            content=[
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{scene_b64}"}}
            ]
        )

        response = model.invoke([message])
        results[question] = response.content

    return results

# ========== 示例6: 图片+文本推理 ==========
def image_text_reasoning():
    """图文推理"""
    model = ChatOpenAI(model="gpt-4o")

    with open("receipt.jpg", "rb") as f:
        receipt_b64 = base64.b64encode(f.read()).decode('utf-8')

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": """分析这张收据:
                1. 列出所有商品和价格
                2. 计算总金额是否正确
                3. 检查是否有折扣
                4. 提取日期和商店信息
                5. 判断这是什么类型的消费(餐饮/购物/服务)"""
            },
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{receipt_b64}"}}
        ]
    )

    response = model.invoke([message])
    return response.content

# 运行示例
if __name__ == "__main__":
    # 选择要运行的示例
    analyze_local_image()
    compare_images()
    chart_data = extract_chart_data()
    print("提取的数据:", chart_data)

    vqa_results = visual_question_answering()
    for q, a in vqa_results.items():
        print(f"Q: {q}\nA: {a}\n")

    receipt_analysis = image_text_reasoning()
    print("收据分析:", receipt_analysis)
```

#### 2.1.3 混合应用案例

**多模态助手完整实现**：

```python
from typing import Union, List
from langchain_core.messages import BaseMessage

class MultiModalAssistant:
    """多模态助手"""

    def __init__(self):
        self.vision_model = ChatOpenAI(model="gpt-4o")
        self.audio_processor = AudioProcessor()
        self.agent = self._create_react_agent()

    def _create_react_agent(self):
        """创建多模态 Agent"""
        from langchain.agents import create_agent

        @tool
        def process_image(image_path: str) -> str:
            """处理图像"""
            image_base64 = encode_image(image_path)
            response = self.vision_model.invoke([
                HumanMessage(content=[
                    {"type": "text", "text": "描述这个图像"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ])
            ])
            return response.content

        @tool
        def process_audio(audio_path: str) -> str:
            """处理音频"""
            with open(audio_path, "rb") as f:
                transcript = self.audio_processor.transcribe_audio(f)
            return transcript

        @tool
        def generate_audio(text: str, output_path: str) -> str:
            """生成音频"""
            audio_content = self.audio_processor.text_to_speech(text)
            with open(output_path, "wb") as f:
                f.write(audio_content)
            return f"音频已保存到 {output_path}"

        return create_agent(
            ChatOpenAI(model="gpt-4o"),
            [process_image, process_audio, generate_audio],
            prompt="""你是多模态助手。

能力：
1. 理解和描述图像
2. 转录和理解音频
3. 生成语音回复

根据用户需求，灵活使用各种模态。"""
        )

    def process(self, input_data: Union[str, dict]) -> dict:
        """处理多模态输入"""

        if isinstance(input_data, str):
            # 纯文本
            messages = [HumanMessage(content=input_data)]

        elif isinstance(input_data, dict):
            # 多模态输入
            content = []

            if "text" in input_data:
                content.append({"type": "text", "text": input_data["text"]})

            if "image" in input_data:
                image_base64 = encode_image(input_data["image"])
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                })

            if "audio" in input_data:
                # 先转录音频
                with open(input_data["audio"], "rb") as f:
                    transcript = self.audio_processor.transcribe_audio(f)
                content.append({"type": "text", "text": f"音频转录：{transcript}"})

            messages = [HumanMessage(content=content)]

        # 处理
        response = self.agent.invoke({"messages": messages})

        return {
            "text_response": response["messages"][-1].content,
            "messages": response["messages"]
        }

# 使用示例
assistant = MultiModalAssistant()

# 处理图文混合
result = assistant.process({
    "text": "这个图表显示了什么趋势？请生成语音解释。",
    "image": "sales_chart.png"
})

# 处理音频
result = assistant.process({
    "audio": "question.mp3",
    "text": "回答音频中的问题"
})
```

---

### 2.2 流式与异步编程

#### 2.2.1 流式输出（stream、astream）

**基础流式输出**：

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 同步流式
model = ChatOpenAI(model="gpt-4o", streaming=True)

# stream 方法
for chunk in model.stream([HumanMessage(content="讲个长故事")]):
    print(chunk.content, end="", flush=True)

# 异步流式
import asyncio

async def async_stream_example():
    """异步流式示例"""
    model = ChatOpenAI(model="gpt-4o", streaming=True)

    async for chunk in model.astream([HumanMessage(content="解释量子计算")]):
        print(chunk.content, end="", flush=True)
        # 可以在这里做其他异步操作
        await asyncio.sleep(0.01)  # 模拟处理

# 运行
asyncio.run(async_stream_example())
```

**Agent 流式输出**：

```python
from langchain.agents import create_agent
from langchain_core.messages import AIMessageChunk

async def stream_agent_response():
    """流式 Agent 响应"""
    agent = create_agent(
        ChatOpenAI(model="gpt-4o", streaming=True),
        [search_tool],
        prompt="你是一个助手"
    )

    # 使用 astream_events 获取所有事件
    async for event in agent.astream_events(
        {"messages": [HumanMessage(content="搜索并总结 LangChain 新特性")]},
        version="v1"
    ):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            # 模型输出流
            content = event["data"]["chunk"].content
            if content:
                print(content, end="", flush=True)

        elif kind == "on_tool_start":
            # 工具开始执行
            print(f"\n🔧 调用工具: {event['name']}")

        elif kind == "on_tool_end":
            # 工具执行完成
            print(f"\n✅ 工具完成: {event['name']}")

# 更精细的流式控制
async def advanced_streaming():
    """高级流式处理"""
    from langchain.agents import create_agent

    agent = create_agent(
        ChatOpenAI(model="gpt-4o", streaming=True),
        [search_tool, analyze_tool],
        prompt="你是研究助手"
    )

    # 收集不同类型的输出
    tokens = []
    tool_calls = []

    async for event in agent.astream_events(
        {"messages": [HumanMessage(content="研究 AI 发展趋势")]},
        version="v2"
    ):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]

            # 收集 tokens
            if hasattr(chunk, "content") and chunk.content:
                tokens.append(chunk.content)
                # 实时显示
                print(chunk.content, end="", flush=True)

            # 收集工具调用
            if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                tool_calls.extend(chunk.tool_calls)

        elif event["event"] == "on_chat_model_end":
            # 模型输出结束
            full_response = "".join(tokens)
            print(f"\n\n完整响应长度: {len(full_response)}")
            print(f"工具调用数: {len(tool_calls)}")

asyncio.run(advanced_streaming())
```

**astream_events 版本差异详解**:

LangChain 提供 `v1` 和 `v2` 两个版本的 astream_events API,推荐使用 `v2`(或省略version参数,默认v2)。

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

# ========== v1 (旧版,不推荐) ==========
async def use_v1():
    """v1 版本示例"""
    agent = create_agent(
        ChatOpenAI(model="gpt-4o", streaming=True),
        [search_tool],
        prompt="你是助手"
    )

    async for event in agent.astream_events(
        {"messages": [HumanMessage(content="查询天气")]},
        version="v1"  # 明确指定 v1
    ):
        # v1 事件格式较简单
        kind = event["event"]

        if kind == "on_chat_model_stream":
            # 基础字段
            chunk = event["data"]["chunk"]
            print(chunk.content, end="")

        # v1 缺少部分元数据,事件类型较少

# ========== v2 (推荐,默认) ==========
async def use_v2():
    """v2 版本示例(推荐)"""
    agent = create_agent(
        ChatOpenAI(model="gpt-4o", streaming=True),
        [search_tool],
        prompt="你是助手"
    )

    async for event in agent.astream_events(
        {"messages": [HumanMessage(content="查询天气")]},
        version="v2"  # 或省略此参数,默认v2
    ):
        # v2 事件格式更完整
        # 事件对象结构:
        # {
        #   "event": "on_chain_start" | "on_chain_stream" | "on_chain_end" | ...,
        #   "name": "component_name",
        #   "run_id": "uuid",
        #   "parent_ids": ["parent_uuid"],
        #   "tags": ["tag1", "tag2"],
        #   "metadata": {...},
        #   "data": {...}
        # }

        event_type = event["event"]
        event_name = event.get("name", "")
        run_id = event.get("run_id", "")

        if event_type == "on_chat_model_stream":
            # LLM 流式输出
            chunk = event["data"]["chunk"]
            if chunk.content:
                print(f"[{run_id[:8]}] {chunk.content}", end="")

        elif event_type == "on_tool_start":
            # 工具开始
            print(f"\n🔧 [{event_name}] 开始执行")
            print(f"   输入: {event['data'].get('input')}")

        elif event_type == "on_tool_end":
            # 工具结束
            print(f"\n✅ [{event_name}] 执行完成")
            print(f"   输出: {event['data'].get('output')}")

        elif event_type == "on_chain_start":
            # Chain 开始
            print(f"\n▶ [{event_name}] 链开始")

        elif event_type == "on_chain_end":
            # Chain 结束
            print(f"\n■ [{event_name}] 链结束")

# ========== v1 vs v2 主要差异 ==========
"""
差异对比:

1. 元数据完整性:
   - v1: 仅包含基础字段(event, data)
   - v2: 包含完整元数据(run_id, parent_ids, tags, metadata)

2. 事件类型:
   - v1: 事件类型较少,粒度较粗
   - v2: 更细粒度的事件类型:
     - on_llm_start, on_llm_stream, on_llm_end
     - on_chat_model_start, on_chat_model_stream, on_chat_model_end
     - on_chain_start, on_chain_stream, on_chain_end
     - on_tool_start, on_tool_stream, on_tool_end
     - on_retriever_start, on_retriever_end
     - on_prompt_start, on_prompt_end

3. 数据结构:
   - v1: event["data"]["chunk"]
   - v2: event["data"]["chunk"] + 更多上下文信息

4. 过滤能力:
   - v1: 有限的过滤选项
   - v2: 支持更精细的过滤:
     - include_names: 只包含特定组件
     - include_types: 只包含特定类型
     - include_tags: 只包含特定标签
     - exclude_names/exclude_types/exclude_tags: 排除规则

5. 性能:
   - v2 提供更好的性能优化和事件去重
"""

# ========== v2 高级过滤示例 ==========
async def v2_filtering():
    """v2 的高级过滤功能"""
    agent = create_agent(
        ChatOpenAI(model="gpt-4o", streaming=True),
        [search_tool, analyze_tool],
        prompt="你是助手"
    )

    # 只监听 LLM 相关事件
    async for event in agent.astream_events(
        {"messages": [HumanMessage(content="分析数据")]},
        version="v2",
        include_types=["chat_model"]  # 只要聊天模型事件
    ):
        if event["event"] == "on_chat_model_stream":
            print(event["data"]["chunk"].content, end="")

    # 只监听特定工具
    async for event in agent.astream_events(
        {"messages": [HumanMessage(content="搜索信息")]},
        version="v2",
        include_names=["search_tool"]  # 只要 search_tool 的事件
    ):
        if event["event"] == "on_tool_end":
            print(f"搜索结果: {event['data']['output']}")

    # 排除某些组件
    async for event in agent.astream_events(
        {"messages": [HumanMessage(content="处理请求")]},
        version="v2",
        exclude_types=["retriever"]  # 排除检索器事件
    ):
        # 处理其他所有事件
        pass

# ========== 推荐使用方式 ==========
async def recommended_usage():
    """推荐的 v2 使用方式"""
    agent = create_agent(
        ChatOpenAI(model="gpt-4o", streaming=True),
        [search_tool],
        prompt="你是助手"
    )

    # 方式1: 省略 version,默认使用 v2
    async for event in agent.astream_events(
        {"messages": [HumanMessage(content="查询")]},
        # version 参数省略,自动使用 v2
    ):
        # 处理事件
        pass

    # 方式2: 明确指定 v2(推荐)
    async for event in agent.astream_events(
        {"messages": [HumanMessage(content="查询")]},
        version="v2"
    ):
        # 使用 v2 的完整功能
        event_type = event["event"]

        # 利用 v2 的元数据
        if "metadata" in event:
            custom_data = event["metadata"].get("custom_field")

        # 追踪调用链
        run_id = event.get("run_id")
        parent_ids = event.get("parent_ids", [])
```

**迁移指南(v1 -> v2)**:

```python
# 旧代码(v1)
async for event in agent.astream_events(input, version="v1"):
    if event["event"] == "on_chat_model_stream":
        chunk = event["data"]["chunk"]
        print(chunk.content)

# 新代码(v2) - 无需修改,100%兼容
async for event in agent.astream_events(input, version="v2"):
    if event["event"] == "on_chat_model_stream":
        chunk = event["data"]["chunk"]
        print(chunk.content)

        # v2 额外提供的信息
        run_id = event.get("run_id")
        metadata = event.get("metadata", {})
```

**总结**:
- ✅ **新项目**: 使用 `version="v2"` 或省略参数(默认v2)
- ⚠️ **旧项目**: 可以继续使用 v1,但建议迁移到 v2
- ❌ **不推荐**: 在新代码中显式指定 `version="v1"`

#### 2.2.2 异步执行与并发控制

**异步 Agent 系统**：

```python
import asyncio
from typing import List
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

class AsyncMultiAgent:
    """异步多 Agent 系统"""

    def __init__(self, concurrency_limit: int = 5):
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.agents = {}

    def register_agent(self, name: str, agent):
        """注册 Agent"""
        self.agents[name] = agent

    async def run_agent_with_limit(self, name: str, input_data: dict):
        """带并发限制的 Agent 执行"""
        async with self.semaphore:
            agent = self.agents[name]
            print(f"🚀 启动 {name}")

            try:
                result = await agent.ainvoke(input_data)
                print(f"✅ {name} 完成")
                return {"agent": name, "result": result, "status": "success"}

            except Exception as e:
                print(f"❌ {name} 失败: {e}")
                return {"agent": name, "error": str(e), "status": "failed"}

    async def run_parallel(self, tasks: List[dict]):
        """并行执行多个任务"""
        coroutines = [
            self.run_agent_with_limit(task["agent"], task["input"])
            for task in tasks
        ]

        results = await asyncio.gather(*coroutines, return_exceptions=True)

        return results

    async def run_pipeline(self, stages: List[List[dict]]):
        """流水线执行（阶段内并行，阶段间串行）"""
        all_results = []

        for stage_num, stage_tasks in enumerate(stages):
            print(f"\n=== 阶段 {stage_num + 1} ===")

            # 并行执行当前阶段
            stage_results = await self.run_parallel(stage_tasks)
            all_results.append(stage_results)

            # 检查是否有失败
            failures = [r for r in stage_results if r.get("status") == "failed"]
            if failures:
                print(f"⚠️  阶段 {stage_num + 1} 有 {len(failures)} 个失败")
                # 可以选择是否继续

        return all_results

# 使用示例
async def main():
    system = AsyncMultiAgent(concurrency_limit=3)

    # 注册 Agents
    system.register_agent("search", create_agent(
        ChatOpenAI(model="gpt-4o-mini"),
        [search_tool],
        prompt="搜索专家"
    ))

    system.register_agent("analyze", create_agent(
        ChatOpenAI(model="gpt-4o-mini"),
        [analyze_tool],
        prompt="分析专家"
    ))

    system.register_agent("summarize", create_agent(
        ChatOpenAI(model="gpt-4o-mini"),
        [],
        prompt="总结专家"
    ))

    # 定义任务
    tasks = [
        {"agent": "search", "input": {"messages": [HumanMessage(content="搜索 A")]}},
        {"agent": "search", "input": {"messages": [HumanMessage(content="搜索 B")]}},
        {"agent": "search", "input": {"messages": [HumanMessage(content="搜索 C")]}},
    ]

    # 并行执行
    results = await system.run_parallel(tasks)

    # 流水线执行
    pipeline = [
        # 阶段 1：并行搜索
        [
            {"agent": "search", "input": {"messages": [HumanMessage(content="搜索 X")]}},
            {"agent": "search", "input": {"messages": [HumanMessage(content="搜索 Y")]}}
        ],
        # 阶段 2：分析结果
        [
            {"agent": "analyze", "input": {"messages": [HumanMessage(content="分析搜索结果")]}}
        ],
        # 阶段 3：总结
        [
            {"agent": "summarize", "input": {"messages": [HumanMessage(content="总结分析")]}}
        ]
    ]

    pipeline_results = await system.run_pipeline(pipeline)

asyncio.run(main())
```

#### 2.2.3 实时交互（WebSocket、SSE）

**WebSocket 实时交互**：

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import json

app = FastAPI()

class WebSocketAgentHandler:
    """WebSocket Agent 处理器"""

    def __init__(self):
        from langchain.agents import create_agent

        self.agent = create_agent(
            ChatOpenAI(model="gpt-4o", streaming=True),
            [search_tool],
            prompt="你是实时助手"
        )
        self.connections = set()

    async def handle_connection(self, websocket: WebSocket):
        """处理 WebSocket 连接"""
        await websocket.accept()
        self.connections.add(websocket)

        try:
            while True:
                # 接收消息
                data = await websocket.receive_text()
                message = json.loads(data)

                # 流式处理并发送
                await self.stream_response(websocket, message["content"])

        except WebSocketDisconnect:
            self.connections.remove(websocket)

    async def stream_response(self, websocket: WebSocket, user_input: str):
        """流式发送响应"""

        # 发送开始信号
        await websocket.send_json({"type": "start"})

        # 流式处理
        async for event in self.agent.astream_events(
            {"messages": [HumanMessage(content=user_input)]},
            version="v1"
        ):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    # 发送文本块
                    await websocket.send_json({
                        "type": "chunk",
                        "content": chunk.content
                    })

            elif event["event"] == "on_tool_start":
                # 发送工具调用通知
                await websocket.send_json({
                    "type": "tool_start",
                    "tool": event["name"]
                })

            elif event["event"] == "on_tool_end":
                # 发送工具完成通知
                await websocket.send_json({
                    "type": "tool_end",
                    "tool": event["name"],
                    "output": str(event["data"].get("output", ""))[:100]
                })

        # 发送结束信号
        await websocket.send_json({"type": "end"})

handler = WebSocketAgentHandler()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await handler.handle_connection(websocket)

# HTML 客户端示例
@app.get("/")
async def get():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>实时 Agent</title>
</head>
<body>
    <div id="messages"></div>
    <input type="text" id="messageText" placeholder="输入消息">
    <button onclick="sendMessage()">发送</button>

    <script>
        const ws = new WebSocket("ws://localhost:8000/ws");

        ws.onmessage = function(event) {
            const message = JSON.parse(event.data);
            const messagesDiv = document.getElementById('messages');

            if (message.type === 'chunk') {
                messagesDiv.innerHTML += message.content;
            } else if (message.type === 'tool_start') {
                messagesDiv.innerHTML += `<br>🔧 调用工具: ${message.tool}<br>`;
            } else if (message.type === 'end') {
                messagesDiv.innerHTML += '<br><br>';
            }
        };

        function sendMessage() {
            const input = document.getElementById('messageText');
            ws.send(JSON.stringify({content: input.value}));
            input.value = '';
        }
    </script>
</body>
</html>
    """)
```

**SSE（Server-Sent Events）实现**：

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
import asyncio

app = FastAPI()

async def event_generator(user_input: str) -> AsyncGenerator[str, None]:
    """SSE 事件生成器"""
    from langchain.agents import create_agent

    agent = create_agent(
        ChatOpenAI(model="gpt-4o", streaming=True),
        [search_tool],
        prompt="你是流式助手"
    )

    # 发送连接建立事件
    yield f"data: {json.dumps({'type': 'connected'})}\n\n"

    # 流式处理
    async for event in agent.astream_events(
        {"messages": [HumanMessage(content=user_input)]},
        version="v1"
    ):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content:
                # 发送内容块
                data = json.dumps({
                    "type": "content",
                    "data": chunk.content
                })
                yield f"data: {data}\n\n"

                # 小延迟避免过快
                await asyncio.sleep(0.01)

        elif event["event"] == "on_tool_start":
            # 发送工具事件
            data = json.dumps({
                "type": "tool",
                "action": "start",
                "name": event["name"]
            })
            yield f"data: {data}\n\n"

    # 发送完成事件
    yield f"data: {json.dumps({'type': 'done'})}\n\n"

@app.get("/stream")
async def stream_response(q: str):
    """SSE 端点"""
    return StreamingResponse(
        event_generator(q),
        media_type="text/event-stream"
    )

# JavaScript 客户端
"""
const eventSource = new EventSource('/stream?q=你的问题');

eventSource.onmessage = function(event) {
    const data = JSON.parse(event.data);

    if (data.type === 'content') {
        document.getElementById('output').innerHTML += data.data;
    } else if (data.type === 'done') {
        eventSource.close();
    }
};
"""
```

---

### 2.3 Model Context Protocol (MCP) 集成

> **说明**: MCP (Model Context Protocol) 是 Anthropic 推出的统一协议,用于连接AI应用与外部工具/数据源。

#### 2.3.1 MCP 完整集成指南

**MCP 协议概述**:

MCP 提供标准化的方式让 LLM 访问:
- **工具 (Tools)**: 函数调用能力(如计算、API调用)
- **资源 (Resources)**: 数据访问(如文件、数据库)
- **提示词 (Prompts)**: 预定义的提示词模板
- **采样 (Sampling)**: LLM 生成能力

**1. 安装MCP客户端**:

```bash
# 安装 MCP 适配器(用于 LangChain)
pip install langchain-mcp-adapters

# 安装 MCP Python SDK
pip install mcp

# 可选: 安装官方 MCP 服务器(Node.js)
npm install -g @modelcontextprotocol/server-filesystem
npm install -g @modelcontextprotocol/server-github
npm install -g @modelcontextprotocol/server-postgres

# 或使用 Python MCP 服务器示例
pip install mcp-server-git  # Git 操作服务器
```

**2. MCP 服务器配置文件**:

创建 `mcp_config.json` 配置多个服务器:

```json
{
  "servers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/允许的目录路径"],
      "env": {}
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "ghp_your_token_here"
      }
    },
    "postgres": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-postgres",
        "postgresql://user:password@localhost:5432/dbname"
      ],
      "env": {}
    },
    "custom-python": {
      "command": "python",
      "args": ["-m", "my_mcp_server"],
      "env": {
        "API_KEY": "your_api_key"
      }
    }
  }
}
```

**3. LangChain集成流程**:

完整的MCP集成包括初始化、获取工具、创建Agent、执行任务四个步骤:

```python
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# ========== 步骤1: 初始化 MCP 客户端（使用配置字典）==========
async def create_mcp_agent():
    """创建集成 MCP 工具的 Agent"""
    # 使用配置字典初始化客户端
    client = MultiServerMCPClient({
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"],
            "transport": "stdio"
        },
        "github": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "transport": "stdio",
            "env": {"GITHUB_TOKEN": "ghp_your_token"}
        }
    })

    # ========== 步骤2: 获取所有工具 ==========
    # get_tools() 会自动为每个配置的服务器创建 session 并获取工具
    tools = await client.get_tools()
    print(f"总共获取到 {len(tools)} 个工具")

    # ========== 步骤3: 创建 Agent ==========
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    agent = create_agent(
        llm,
        tools,
        prompt="""你是一个助手,可以访问:
        1. 文件系统工具(读写文件)
        2. GitHub 工具(管理仓库、Issue、PR)

        根据用户需求选择合适的工具完成任务。"""
    )

    return agent

# ========== 步骤4: 使用 Agent 执行任务 ==========
async def main():
    """完整示例"""
    # 创建 Agent
    agent = await create_mcp_agent()

    # 示例任务1: 文件操作
    result1 = await agent.ainvoke({
        "messages": [HumanMessage(content="在 /workspace 下创建一个README.md文件")]
    })
    print("任务1结果:", result1["messages"][-1].content)

    # 示例任务2: GitHub 操作
    result2 = await agent.ainvoke({
        "messages": [HumanMessage(content="列出我的GitHub仓库")]
    })
    print("任务2结果:", result2["messages"][-1].content)

    # 示例任务3: 混合操作
    result3 = await agent.ainvoke({
        "messages": [HumanMessage(
            content="检查我的项目仓库是否有新的Issue,并将Issue标题写入文件"
        )]
    })
    print("任务3结果:", result3["messages"][-1].content)

# 运行
asyncio.run(main())
```

**4. 使用配置文件的方式**:

```python
import json
from pathlib import Path

async def init_from_config(config_path: str):
    """从配置文件初始化 MCP 客户端"""
    # 读取配置
    with open(config_path) as f:
        config = json.load(f)

    client = MultiServerMCPClient()
    server_params = {}

    # 解析每个服务器配置
    for server_name, server_config in config["servers"].items():
        params = StdioServerParameters(
            command=server_config["command"],
            args=server_config["args"],
            env=server_config.get("env", None)
        )
        server_params[server_name] = params

    return client, server_params

# 使用
client, params = await init_from_config("mcp_config.json")
```

**5. 错误处理与重试**:

```python
import asyncio
from typing import Optional

async def robust_mcp_call(client, server_name, params, max_retries=3):
    """带重试的 MCP 调用"""
    for attempt in range(max_retries):
        try:
            async with client.session(server_name, params) as session:
                tools = client.get_tools()
                return tools
        except Exception as e:
            print(f"尝试 {attempt + 1}/{max_retries} 失败: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # 指数退避
            else:
                raise

# 使用
tools = await robust_mcp_call(client, "filesystem", fs_params)
```

**6. 监控和日志**:

```python
import logging

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("mcp_integration")

async def init_mcp_with_logging(client, server_params):
    """带日志的 MCP 初始化"""
    for server_name, params in server_params.items():
        logger.info(f"连接到 MCP 服务器: {server_name}")

        try:
            async with client.session(server_name, params) as session:
                tools = client.get_tools()
                logger.info(f"✅ {server_name}: 获取 {len(tools)} 个工具")

                # 记录工具详情
                for tool in tools:
                    logger.debug(f"  - {tool.name}: {tool.description}")

        except Exception as e:
            logger.error(f"❌ {server_name} 连接失败: {e}")
            raise
```

**7. 常见问题与解决方案**:

```python
# 问题1: 服务器启动失败
# 解决: 检查命令和参数是否正确
try:
    async with client.session("server", params):
        pass
except FileNotFoundError:
    print("命令不存在,请检查是否安装了对应的 MCP 服务器")
except PermissionError:
    print("权限不足,请检查目录访问权限")

# 问题2: 工具调用失败
# 解决: 验证工具参数
tool_schema = tool.inputSchema  # 获取工具的参数模式
print(f"工具 {tool.name} 需要参数: {tool_schema}")

# 问题3: 会话超时
# 解决: 设置更长的超时时间
from mcp.client.stdio import StdioServerParameters

params = StdioServerParameters(
    command="npx",
    args=["..."],
    env=None,
    timeout=60.0  # 60秒超时
)
```

**MCP 基础集成**：

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.client.stdio import StdioServerParameters
from mcp.client.sse import create_mcp_http_client
import httpx

# 创建 MCP 客户端
mcp_client = MultiServerMCPClient()

# ========== 正确方式 1：使用配置字典（推荐）==========
async def setup_mcp_with_config():
    """使用配置字典初始化（推荐方式）"""
    # 使用配置字典初始化客户端
    client = MultiServerMCPClient({
        "math_server": {
            "command": "python",
            "args": ["-m", "mcp_servers.math"],
            "transport": "stdio"
        }
    })

    # 直接获取所有工具
    tools = await client.get_tools()
    print(f"可用工具: {len(tools)} 个")
    return tools

# ========== 正确方式 1b：使用 session 方法（显式管理）==========
from langchain_mcp_adapters.tools import load_mcp_tools

async def setup_mcp_with_session():
    """使用 session 方法显式管理连接"""
    mcp_client = MultiServerMCPClient()

    # 创建 stdio 参数
    stdio_params = StdioServerParameters(
        command="python",
        args=["-m", "mcp_servers.math"],
        env=None
    )

    # 使用 session 上下文管理器
    async with mcp_client.session("math_server", stdio_params) as session:
        # 使用 load_mcp_tools 从 session 获取工具
        tools = await load_mcp_tools(session)
        print(f"可用工具: {len(tools)} 个")
        return tools

# ========== 正确方式 2：创建 HTTP 客户端 ==========
def create_http_client_correct():
    """正确创建 HTTP 客户端"""

    # create_mcp_http_client 不接受 url 参数
    # 它返回一个 httpx.AsyncClient
    http_client = create_mcp_http_client(
        headers={"Authorization": "Bearer token"},
        timeout=httpx.Timeout(30.0),
        auth=None
    )

    # URL 在实际请求时指定
    # 使用示例：
    # response = await http_client.get("http://localhost:8000/mcp")

    return http_client

# ========== 使用 LangGraph 创建 Agent（推荐）==========
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

async def create_agent_with_mcp():
    """创建集成 MCP 工具的 Agent（推荐配置字典方式）"""
    # 使用配置字典初始化
    client = MultiServerMCPClient({
        "math": {
            "command": "python",
            "args": ["-m", "mcp_servers.math"],
            "transport": "stdio"
        }
    })

    # 获取工具
    tools = await client.get_tools()

    # 创建 LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # 使用 create_agent
    agent = create_agent(
        llm,
        tools,
        prompt="你是一个有用的助手，可以使用工具帮助用户。"
    )

    # 调用 Agent
    result = await agent.ainvoke({
        "messages": [HumanMessage(content="计算 sqrt(144)")]
    })

    return result

# ========== 备选方案：创建模拟工具 ==========
from langchain_core.tools import Tool

def create_mock_tools():
    """创建模拟工具（不需要 MCP 服务器）"""

    def calculate(expression: str) -> str:
        """计算数学表达式"""
        import math
        safe_dict = {
            "__builtins__": {},
            "sqrt": math.sqrt,
            "log": math.log,
            "pi": math.pi
        }
        try:
            result = eval(expression, safe_dict)
            return f"结果: {result}"
        except Exception as e:
            return f"错误: {e}"

    def query_weather(city: str) -> str:
        """查询天气（模拟）"""
        return f"{city} 的天气: 晴天 22°C"

    tools = [
        Tool(name="calculate", func=calculate, description="计算数学表达式"),
        Tool(name="query_weather", func=query_weather, description="查询天气")
    ]

    return tools

# 使用模拟工具创建 Agent
tools = create_mock_tools()
llm = ChatOpenAI(model="gpt-4o")
agent = create_agent(llm, tools)
```

#### 2.3.2 stdio 与 HTTP 传输

**自定义 MCP 服务器**：

```python
# mcp_server.py
import json
import sys
from typing import Any, Dict

class CustomMCPServer:
    """自定义 MCP 服务器"""

    def __init__(self):
        self.tools = {
            "calculate": self.calculate,
            "database_query": self.database_query
        }

    def calculate(self, expression: str) -> float:
        """计算数学表达式"""
        # 安全评估
        return eval(expression, {"__builtins__": {}}, {})

    def database_query(self, query: str) -> List[dict]:
        """执行数据库查询"""
        # 模拟数据库查询
        return [{"id": 1, "name": "example"}]

    def handle_request(self, request: dict) -> dict:
        """处理 MCP 请求"""
        method = request.get("method")

        if method == "tools/list":
            # 返回工具列表
            return {
                "tools": [
                    {
                        "name": name,
                        "description": func.__doc__,
                        "inputSchema": self._get_schema(func)
                    }
                    for name, func in self.tools.items()
                ]
            }

        elif method == "tools/call":
            # 调用工具
            tool_name = request["params"]["name"]
            args = request["params"]["arguments"]

            if tool_name in self.tools:
                result = self.tools[tool_name](**args)
                return {"content": [{"type": "text", "text": str(result)}]}

            else:
                return {"error": f"Unknown tool: {tool_name}"}

    def run_stdio(self):
        """运行 stdio 模式"""
        while True:
            line = sys.stdin.readline()
            if not line:
                break

            request = json.loads(line)
            response = self.handle_request(request)

            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

# HTTP 模式
from fastapi import FastAPI

app = FastAPI()
server = CustomMCPServer()

@app.post("/mcp")
async def mcp_endpoint(request: dict):
    """MCP HTTP 端点"""
    return server.handle_request(request)

# 使用自定义服务器
mcp_client = MultiServerMCPClient()
mcp_client.add_server(
    "custom",
    StdioMCPTransport(
        command="python",
        args=["mcp_server.py"]
    )
)
```

#### 2.3.3 工具服务器集成

**完整的 MCP 工具服务器**：

```python
from typing import List, Dict, Any
import asyncio

class MCPToolServer:
    """MCP 工具服务器"""

    def __init__(self):
        self.sessions = {}  # 会话管理
        self._register_tools()

    def _register_tools(self):
        """注册工具"""
        self.tools = {
            "github": GitHubTool(),
            "docker": DockerTool(),
            "slack": SlackTool(),
            "database": DatabaseTool()
        }

    async def initialize(self, session_id: str) -> dict:
        """初始化会话"""
        self.sessions[session_id] = {
            "created_at": datetime.now(),
            "context": {}
        }

        return {
            "protocolVersion": "2024-11-01",
            "capabilities": {
                "tools": True,
                "prompts": False,
                "resources": False,
                "sampling": False
            },
            "serverInfo": {
                "name": "custom-mcp-server",
                "version": "1.0.0"
            }
        }

    async def list_tools(self) -> List[dict]:
        """列出所有工具"""
        tools_list = []

        for name, tool in self.tools.items():
            tools_list.append({
                "name": name,
                "description": tool.description,
                "inputSchema": tool.get_schema()
            })

        return {"tools": tools_list}

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """调用工具"""
        if tool_name not in self.tools:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error: Unknown tool {tool_name}"
                }],
                "isError": True
            }

        try:
            result = await self.tools[tool_name].execute(**arguments)

            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(result, indent=2)
                }]
            }

        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error executing {tool_name}: {str(e)}"
                }],
                "isError": True
            }

class GitHubTool:
    """GitHub 工具"""

    description = "与 GitHub 交互"

    def get_schema(self):
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list_repos", "create_issue", "get_pr"]
                },
                "params": {"type": "object"}
            }
        }

    async def execute(self, action: str, params: dict) -> dict:
        """执行 GitHub 操作"""
        if action == "list_repos":
            # 列出仓库
            return {"repos": ["repo1", "repo2"]}

        elif action == "create_issue":
            # 创建 Issue
            return {"issue_id": "123", "url": "https://github.com/..."}

        # 更多操作...

# 集成到 LangChain
class MCPToolAdapter:
    """MCP 工具适配器"""

    def __init__(self, mcp_server: MCPToolServer):
        self.server = mcp_server

    async def to_langchain_tools(self) -> List:
        """转换为 LangChain 工具"""
        mcp_tools = await self.server.list_tools()

        langchain_tools = []
        for tool in mcp_tools["tools"]:

            @tool
            def langchain_tool(**kwargs):
                f"""MCP Tool: {tool['name']}

                {tool['description']}
                """
                return asyncio.run(
                    self.server.call_tool(tool["name"], kwargs)
                )

            langchain_tool.__name__ = tool["name"]
            langchain_tools.append(langchain_tool)

        return langchain_tools
```

#### 2.3.4 实战案例

**完整的 MCP 集成应用**：

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
import os

class MCPIntegratedSystem:
    """MCP 集成系统"""

    def __init__(self):
        self.mcp_client = self._setup_mcp()

    def _setup_mcp(self) -> MultiServerMCPClient:
        """设置 MCP 客户端（使用配置字典）"""
        # 使用配置字典初始化多个服务器
        client = MultiServerMCPClient({
            "database": {
                "command": "npx",
                "args": [
                    "@modelcontextprotocol/server-postgres",
                    os.getenv("DATABASE_URL")
                ],
                "transport": "stdio"
            },
            "github": {
                "command": "npx",
                "args": [
                    "@modelcontextprotocol/server-github",
                    "--token", os.getenv("GITHUB_TOKEN")
                ],
                "transport": "stdio"
            },
            "slack": {
                "url": "http://localhost:3000/mcp",
                "transport": "streamable_http",
                "headers": {
                    "Authorization": f"Bearer {os.getenv('SLACK_TOKEN')}"
                }
            },
            "filesystem": {
                "command": "npx",
                "args": [
                    "@modelcontextprotocol/server-filesystem",
                    "--root", "/workspace"
                ],
                "transport": "stdio"
            }
        })

        return client

    async def create_react_agent(self):
        """创建集成 Agent"""
        # 获取所有 MCP 工具
        tools = await self.mcp_client.get_tools()

        return create_agent(
            ChatOpenAI(model="gpt-4o"),
            tools,
            prompt="""你是一个全能助手，可以：

1. 查询和操作数据库
2. 管理 GitHub 仓库和 Issues
3. 发送 Slack 消息
4. 操作文件系统

根据用户需求，灵活使用各种工具完成任务。"""
        )

    async def execute_workflow(self, task: str):
        """执行工作流"""
        # 创建 Agent
        agent = await self.create_react_agent()

        # 使用 Agent 执行任务
        result = await agent.ainvoke({
            "messages": [("user", f"执行以下任务：{task}")]
        })

        return result["messages"][-1].content

# ========== 使用示例 ==========
async def main():
    """运行示例"""
    system = MCPIntegratedSystem()

    # 执行多个任务
    result1 = await system.execute_workflow("在 myproject 仓库创建一个关于用户认证的 Issue")
    print(f"任务1结果: {result1}")

    result2 = await system.execute_workflow("查询数据库中状态为pending的任务")
    print(f"任务2结果: {result2}")

    result3 = await system.execute_workflow("在Slack #dev频道发送消息：开始新的开发迭代")
    print(f"任务3结果: {result3}")

# 运行
import asyncio
asyncio.run(main())
```

---

### 2.4 自定义集成

#### 2.4.0 常用工具导入路径参考

**LangChain 生态常用导入验证**:

在使用 LangChain 高级功能时,正确的导入路径至关重要。以下是经过验证的常用导入路径(适用于 LangChain 1.0+)。

```python
# ========== 核心组件 ==========
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
    AIMessageChunk  # 流式输出时使用
)

from langchain_core.tools import tool  # 装饰器创建工具(推荐)
from langchain_core.tools import Tool, StructuredTool  # 类创建工具

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough

# ========== LLM/Chat 模型 ==========
from langchain_openai import ChatOpenAI, OpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_anthropic import ChatAnthropic
from langchain_anthropic import AnthropicEmbeddings

# ========== 社区集成工具 ==========
# 搜索工具
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import WikipediaQueryRun 

# 文档加载器
from langchain_community.document_loaders import (
    PyPDFLoader,         
    TextLoader,          
    WebBaseLoader,       
    DirectoryLoader      
)

# 向量数据库
from langchain_community.vectorstores import (
    Chroma,             
    FAISS,              
    Qdrant              
)

# ========== 文本分割 ==========
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,(新路径)
    CharacterTextSplitter           (新路径)
)

# 旧路径(已弃用,但仍兼容)
# from langchain.text_splitter import RecursiveCharacterTextSplitter  # ⚠️ 旧路径

# ========== LangGraph 组件 ==========
from langgraph.graph import StateGraph, MessagesState, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

# ========== Agent 创建 ==========
from langchain.agents import create_agent  # ✅ 推荐(LangChain 1.0+)

# ========== Retriever ==========
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

# ========== 回调与追踪 ==========
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun
)

# LangSmith (需要单独安装 langsmith)
from langsmith import Client
from langchain_core.callbacks.tracers import LangChainTracer

# ========== MCP 集成 ==========
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.client.stdio import StdioServerParameters

# ========== Pydantic 模型(用于工具定义) ==========
# LangChain 1.0+ 使用内置的 pydantic v1 兼容层
from langchain.pydantic_v1 import BaseModel, Field

# 如果直接使用 Pydantic v2
from pydantic import BaseModel, Field  # ✅ 也可以,但注意兼容性

# ========== 自定义基类 ==========
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain.tools import BaseTool

# ========== 完整示例：验证导入 ==========
def verify_imports():
    """验证常用导入是否正确"""
    try:
        # 核心组件
        from langchain_core.messages import HumanMessage, AIMessage
        from langchain_core.tools import tool, Tool

        # LLM
        from langchain_openai import ChatOpenAI

        # 社区工具
        from langchain_community.tools import DuckDuckGoSearchRun
        from langchain_community.tools import WikipediaQueryRun

        # 文档处理
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # 向量存储
        from langchain_community.vectorstores import Chroma

        # LangGraph
        from langgraph.graph import StateGraph, MessagesState, END
        from langgraph.checkpoint.memory import MemorySaver

        # Agent
        from langchain.agents import create_agent

        # MCP
        from langchain_mcp_adapters.client import MultiServerMCPClient

        print("✅ 所有导入验证通过!")
        return True

    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("请确保安装了相应的包:")
        print("  pip install langchain langchain-openai langchain-community")
        print("  pip install langchain-text-splitters langgraph")
        print("  pip install langchain-mcp-adapters")
        return False

# 运行验证
verify_imports()
```

**常见导入错误与修复**:

```python
# ❌ 错误1: 使用已弃用的 text_splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ✅ 修复: 使用新的 langchain_text_splitters 包
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ❌ 错误2: 使用已废弃的 create_react_agent (LangGraph 1.0+ 已废弃)
from langgraph.prebuilt import create_react_agent  # 已废弃

# ✅ 修复: 使用 LangChain 1.0+ 的 create_agent
from langchain.agents import create_agent  # LangChain 1.0+ 推荐

# ❌ 错误3: 工具定义使用错误的 Pydantic 版本
from pydantic import BaseModel, Field  # 可能导致兼容性问题

# ✅ 修复: 使用 LangChain 的 pydantic_v1 兼容层
from langchain.pydantic_v1 import BaseModel, Field

# ❌ 错误4: 社区工具导入路径错误
from langchain.tools import DuckDuckGoSearchRun  # 旧路径

# ✅ 修复: 使用 langchain_community
from langchain_community.tools import DuckDuckGoSearchRun

# ❌ 错误5: 消息类型导入路径错误
from langchain.schema import HumanMessage  # 旧路径

# ✅ 修复: 使用 langchain_core.messages
from langchain_core.messages import HumanMessage
```

**包安装指南**:

```bash
# 核心包
pip install langchain langchain-core

# OpenAI 集成
pip install langchain-openai

# Anthropic 集成
pip install langchain-anthropic

# 社区集成(工具、加载器、向量数据库)
pip install langchain-community

# 文本分割器(独立包)
pip install langchain-text-splitters

# LangGraph(Agent 工作流)
pip install langgraph

# MCP 适配器
pip install langchain-mcp-adapters

# 可选: 特定向量数据库
pip install chromadb  # Chroma
pip install faiss-cpu  # FAISS
pip install qdrant-client  # Qdrant

# 可选: 文档处理
pip install pypdf  # PDF 支持
pip install beautifulsoup4  # Web 内容解析

# 可选: LangSmith 追踪
pip install langsmith
```

**导入路径演进历史**:

```python
# LangChain 0.x (旧版 - 已废弃)
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.agents import AgentExecutor  # ❌ 已移除，LangChain 1.0+ 不再存在

# LangChain 1.0+ (当前 - 推荐)
from langchain_openai import OpenAI, ChatOpenAI  # 独立包
from langchain_core.messages import HumanMessage  # 核心模块
from langchain.agents import create_agent  # ✅ 简化 API，直接返回可执行的 agent
```

---

#### 2.4.1 自定义 LLM（BaseChatModel）

```python
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from typing import List, Optional, Any
import requests

class CustomLLM(BaseChatModel):
    """自定义 LLM 实现"""

    api_key: str
    api_url: str
    model_name: str = "custom-model"
    temperature: float = 0.7

    class Config:
        """Pydantic 配置"""
        extra = "forbid"

    @property
    def _llm_type(self) -> str:
        """返回 LLM 类型"""
        return "custom"

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any
    ) -> str:
        """同步调用"""
        # 转换消息格式
        formatted_messages = self._format_messages(messages)

        # 调用 API
        response = requests.post(
            self.api_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model_name,
                "messages": formatted_messages,
                "temperature": self.temperature,
                "stop": stop
            }
        )

        result = response.json()
        return result["choices"][0]["message"]["content"]

    async def _acall(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any
    ) -> str:
        """异步调用"""
        import aiohttp

        formatted_messages = self._format_messages(messages)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model_name,
                    "messages": formatted_messages,
                    "temperature": self.temperature
                }
            ) as response:
                result = await response.json()
                return result["choices"][0]["message"]["content"]

    def _format_messages(self, messages: List[BaseMessage]) -> List[dict]:
        """格式化消息"""
        formatted = []
        for msg in messages:
            formatted.append({
                "role": msg.type,
                "content": msg.content
            })
        return formatted

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any
    ):
        """流式输出"""
        formatted_messages = self._format_messages(messages)

        response = requests.post(
            self.api_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model_name,
                "messages": formatted_messages,
                "stream": True
            },
            stream=True
        )

        for line in response.iter_lines():
            if line:
                # 解析 SSE 格式
                chunk = self._parse_sse(line)
                if chunk:
                    yield AIMessageChunk(content=chunk)

# 使用自定义 LLM
custom_llm = CustomLLM(
    api_key="your-api-key",
    api_url="https://api.custom-llm.com/v1/chat",
    model_name="custom-gpt"
)

# 可以像使用其他 LLM 一样使用
from langchain.agents import create_agent

agent = create_agent(
    custom_llm,
    [],
    prompt="你是一个助手"
)
```

#### 2.4.2 自定义 Retriever 与 Tool

```python
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List

class CustomRetriever(BaseRetriever):
    """自定义检索器"""

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[Any] = None
    ) -> List[Document]:
        """检索文档"""
        # 自定义检索逻辑
        results = self.search_backend(query)

        documents = []
        for result in results:
            doc = Document(
                page_content=result["content"],
                metadata={
                    "source": result["source"],
                    "score": result["score"],
                    "timestamp": result["timestamp"]
                }
            )
            documents.append(doc)

        return documents

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[Any] = None
    ) -> List[Document]:
        """异步检索"""
        # 异步实现
        results = await self.async_search_backend(query)
        # ... 转换为 Documents
        return documents

# 自定义工具
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool

class CustomToolInput(BaseModel):
    """工具输入模式"""
    query: str = Field(description="查询字符串")
    filters: dict = Field(default={}, description="过滤条件")

class CustomTool(BaseTool):
    """自定义工具"""
    name = "custom_search"
    description = "自定义搜索工具"
    args_schema = CustomToolInput

    def _run(
        self,
        query: str,
        filters: dict = {},
        run_manager: Optional[Any] = None
    ) -> str:
        """执行工具"""
        # 实现工具逻辑
        results = self.perform_search(query, filters)
        return json.dumps(results)

    async def _arun(
        self,
        query: str,
        filters: dict = {},
        run_manager: Optional[Any] = None
    ) -> str:
        """异步执行"""
        results = await self.async_perform_search(query, filters)
        return json.dumps(results)
```

---

### 本章小结

1. **多 Agent 协作**：掌握 Supervisor-Worker、Router、Hierarchical 三种模式
2. **多模态处理**：图像理解、OCR、音频处理、混合应用
3. **流式与异步**：stream/astream、并发控制、WebSocket/SSE 实时交互
4. **MCP 集成**：连接外部服务、工具服务器、多协议支持
5. **自定义集成**：实现自定义 LLM、Retriever、Tool

---

## 第3章：Long-term Memory 架构设计

> **目标**：掌握长期记忆系统设计，实现跨会话的知识积累和个性化体验

短期记忆(Checkpointer)只保存单个会话的历史，无法实现跨会话的知识积累。Long-term Memory 系统通过向量数据库、知识图谱等技术，让 Agent 拥有持久化的"长期记忆"能力。

---

### 3.1 为什么需要 Long-term Memory

#### 3.1.1 短期记忆的局限

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

# ✅ 短期记忆：单会话内有效
agent = create_agent(
    ChatOpenAI(model="gpt-4o"),
    [],
    checkpointer=MemorySaver()
)

# 会话1
config1 = {"configurable": {"thread_id": "session-1"}}
agent.invoke({"messages": [("user", "我喜欢Python")]}, config=config1)
agent.invoke({"messages": [("user", "我喜欢什么?")]}, config=config1)
# 输出：你喜欢Python ✅

# 会话2（新会话）
config2 = {"configurable": {"thread_id": "session-2"}}
agent.invoke({"messages": [("user", "我喜欢什么?")]}, config=config2)
# 输出：我不知道 ❌ 短期记忆无法跨会话!
```

**Long-term Memory 解决的问题**：
1. ❌ 用户偏好无法跨会话保留
2. ❌ 历史交互知识无法复用
3. ❌ 无法实现个性化推荐
4. ❌ 无法积累领域知识

---

### 3.2 Long-term Memory 架构

#### 3.2.1 混合记忆架构

```mermaid
graph TB
    A[用户输入] --> B{记忆系统}

    B --> C[短期记忆<br/>Checkpointer]
    B --> D[长期记忆<br/>VectorStore]

    C --> E[会话历史<br/>最近10轮对话]
    D --> F[用户画像<br/>偏好/习惯]
    D --> G[知识库<br/>FAQ/文档]
    D --> H[交互历史<br/>所有对话摘要]

    E --> I[Agent]
    F --> I
    G --> I
    H --> I

    style C fill:#FFF9C4
    style D fill:#C8E6C9
```

#### 3.2.2 基础实现

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage

class LongTermMemory:
    """长期记忆系统"""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.embeddings = OpenAIEmbeddings()

        # 向量数据库(存储长期记忆)
        self.vectorstore = Chroma(
            collection_name=f"memory_{user_id}",
            embedding_function=self.embeddings
        )

    def add_interaction(self, user_message: str, ai_response: str):
        """保存一次交互到长期记忆"""
        # 创建记忆文本
        memory_text = f"用户: {user_message}\nAI: {ai_response}"

        # 存入向量数据库
        self.vectorstore.add_texts(
            texts=[memory_text],
            metadatas=[{
                "user_id": self.user_id,
                "timestamp": datetime.now().isoformat(),
                "type": "interaction"
            }]
        )

    def recall(self, query: str, k: int = 3) -> list[str]:
        """根据查询召回相关记忆"""
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    def get_user_profile(self) -> dict:
        """获取用户画像"""
        # 从所有交互中提取用户偏好
        all_memories = self.vectorstore.similarity_search("", k=100)

        # 简化实现：统计关键词
        preferences = {}
        for memory in all_memories:
            # 分析用户偏好...
            pass

        return preferences
```

**增强版长期记忆系统**:

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import uuid

class AdvancedLongTermMemory:
    """增强版长期记忆系统"""

    def __init__(self, user_id: str, persist_directory: str = "./chroma_db"):
        self.user_id = user_id
        self.embeddings = OpenAIEmbeddings()

        # 分层向量存储
        self.interaction_store = Chroma(
            collection_name=f"interactions_{user_id}",
            embedding_function=self.embeddings,
            persist_directory=f"{persist_directory}/interactions"
        )

        self.knowledge_store = Chroma(
            collection_name=f"knowledge_{user_id}",
            embedding_function=self.embeddings,
            persist_directory=f"{persist_directory}/knowledge"
        )

        self.preference_store = Chroma(
            collection_name=f"preferences_{user_id}",
            embedding_function=self.embeddings,
            persist_directory=f"{persist_directory}/preferences"
        )

    def add_interaction(
        self,
        user_message: str,
        ai_response: str,
        metadata: Optional[Dict] = None
    ):
        """保存交互"""
        interaction_id = str(uuid.uuid4())

        full_metadata = {
            "user_id": self.user_id,
            "timestamp": datetime.now().isoformat(),
            "type": "interaction",
            "interaction_id": interaction_id,
            **(metadata or {})
        }

        memory_text = f"用户: {user_message}\nAI: {ai_response}"

        self.interaction_store.add_texts(
            texts=[memory_text],
            metadatas=[full_metadata],
            ids=[interaction_id]
        )

    def add_knowledge(
        self,
        knowledge: str,
        source: str = "user",
        importance: int = 1
    ):
        """添加知识点"""
        knowledge_id = str(uuid.uuid4())

        self.knowledge_store.add_texts(
            texts=[knowledge],
            metadatas=[{
                "user_id": self.user_id,
                "timestamp": datetime.now().isoformat(),
                "source": source,
                "importance": importance,
                "type": "knowledge"
            }],
            ids=[knowledge_id]
        )

    def add_preference(
        self,
        preference_key: str,
        preference_value: str,
        confidence: float = 1.0
    ):
        """添加用户偏好"""
        pref_id = str(uuid.uuid4())

        pref_text = f"{preference_key}: {preference_value}"

        self.preference_store.add_texts(
            texts=[pref_text],
            metadatas=[{
                "user_id": self.user_id,
                "timestamp": datetime.now().isoformat(),
                "key": preference_key,
                "value": preference_value,
                "confidence": confidence,
                "type": "preference"
            }],
            ids=[pref_id]
        )

    def recall_interactions(
        self,
        query: str,
        k: int = 3,
        time_range: Optional[timedelta] = None
    ) -> List[Dict]:
        """召回相关交互"""
        results = self.interaction_store.similarity_search_with_score(
            query, k=k*2
        )

        # 时间过滤
        if time_range:
            cutoff_time = datetime.now() - time_range
            results = [
                (doc, score) for doc, score in results
                if datetime.fromisoformat(doc.metadata["timestamp"]) > cutoff_time
            ]

        # 按相似度排序并返回top k
        results.sort(key=lambda x: x[1])
        return [
            {
                "content": doc.page_content,
                "score": score,
                "metadata": doc.metadata
            }
            for doc, score in results[:k]
        ]

    def recall_knowledge(self, query: str, k: int = 5) -> List[Dict]:
        """召回相关知识"""
        results = self.knowledge_store.similarity_search_with_score(query, k=k)

        # 按重要性和相似度综合排序
        sorted_results = sorted(
            results,
            key=lambda x: (
                x[0].metadata.get("importance", 1) * 10 + (1 - x[1])
            ),
            reverse=True
        )

        return [
            {
                "content": doc.page_content,
                "importance": doc.metadata.get("importance", 1),
                "source": doc.metadata.get("source"),
                "score": score
            }
            for doc, score in sorted_results
        ]

    def get_preferences(self) -> Dict[str, str]:
        """获取所有偏好"""
        all_prefs = self.preference_store.get()

        preferences = {}
        if all_prefs and all_prefs["metadatas"]:
            for metadata in all_prefs["metadatas"]:
                key = metadata.get("key")
                value = metadata.get("value")
                if key and value:
                    preferences[key] = value

        return preferences

    def build_user_context(self, query: str) -> str:
        """构建用户上下文"""
        # 召回各类记忆
        interactions = self.recall_interactions(query, k=2)
        knowledge = self.recall_knowledge(query, k=3)
        preferences = self.get_preferences()

        # 构建上下文字符串
        context_parts = []

        if preferences:
            prefs_str = "\n".join([f"- {k}: {v}" for k, v in preferences.items()])
            context_parts.append(f"用户偏好:\n{prefs_str}")

        if knowledge:
            knowledge_str = "\n".join([
                f"- {k['content']} (重要性: {k['importance']})"
                for k in knowledge
            ])
            context_parts.append(f"\n相关知识:\n{knowledge_str}")

        if interactions:
            interactions_str = "\n".join([
                f"- {i['content']}" for i in interactions[:2]
            ])
            context_parts.append(f"\n相关历史:\n{interactions_str}")

        return "\n".join(context_parts)

    def extract_and_store_knowledge(
        self,
        conversation: List[Dict],
        llm: ChatOpenAI
    ):
        """从对话中提取并存储知识"""
        # 使用LLM提取知识点
        conversation_text = "\n".join([
            f"{msg['role']}: {msg['content']}" for msg in conversation
        ])

        extraction_prompt = f"""从以下对话中提取用户的关键信息、偏好和知识点:

{conversation_text}

返回JSON格式:
{{
    "knowledge": ["知识点1", "知识点2"],
    "preferences": {{"偏好键": "偏好值"}},
    "importance": 1-5
}}"""

        response = llm.invoke([HumanMessage(content=extraction_prompt)])

        # 解析并存储
        import json
        try:
            extracted = json.loads(response.content)

            # 存储知识
            for knowledge in extracted.get("knowledge", []):
                self.add_knowledge(
                    knowledge,
                    source="conversation",
                    importance=extracted.get("importance", 1)
                )

            # 存储偏好
            for key, value in extracted.get("preferences", {}).items():
                self.add_preference(key, value, confidence=0.8)

        except json.JSONDecodeError:
            pass  # 提取失败,静默处理
```

---

### 3.3 实战：个性化助手

#### 3.3.1 集成长期记忆到Agent

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

class PersonalizedAgent:
    """带长期记忆的个性化Agent"""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.long_term_memory = LongTermMemory(user_id)

        # Agent(带短期记忆)
        self.agent = create_agent(
            ChatOpenAI(model="gpt-4o"),
            [],
            checkpointer=MemorySaver(),
            prompt=self._build_system_prompt()
        )

    def _build_system_prompt(self) -> str:
        """构建包含用户画像的系统提示"""
        profile = self.long_term_memory.get_user_profile()

        return f"""你是一个个性化助手。

用户画像：
- 用户ID: {self.user_id}
- 偏好: {profile.get('preferences', '未知')}
- 习惯: {profile.get('habits', '未知')}

请根据用户画像提供个性化服务。"""

    def invoke(self, user_message: str, thread_id: str):
        """调用Agent"""
        # 1. 从长期记忆召回相关内容
        relevant_memories = self.long_term_memory.recall(user_message, k=3)

        # 2. 增强提示词
        enhanced_message = f"""当前问题: {user_message}

相关历史记忆:
{chr(10).join(f"- {mem}" for mem in relevant_memories)}

请基于历史记忆提供个性化回答。"""

        # 3. 调用Agent
        config = {"configurable": {"thread_id": thread_id}}
        result = self.agent.invoke(
            {"messages": [("user", enhanced_message)]},
            config=config
        )

        ai_response = result["messages"][-1].content

        # 4. 保存到长期记忆
        self.long_term_memory.add_interaction(user_message, ai_response)

        return ai_response

# 使用示例
agent = PersonalizedAgent(user_id="alice")

# 会话1
response1 = agent.invoke("我喜欢Python编程", thread_id="session-1")

# 会话2（新会话，但能召回长期记忆）
response2 = agent.invoke("推荐一本书给我", thread_id="session-2")
# Agent会基于"喜欢Python"推荐Python相关书籍 ✅
```

---

### 3.4 高级特性

#### 3.4.1 记忆优先级

```python
class PrioritizedMemory:
    """带优先级的记忆系统"""

    def __init__(self):
        self.vectorstore = Chroma(...)

    def add_with_priority(self, text: str, priority: str):
        """添加带优先级的记忆"""
        self.vectorstore.add_texts(
            texts=[text],
            metadatas=[{
                "priority": priority,  # "high", "medium", "low"
                "timestamp": datetime.now().isoformat()
            }]
        )

    def recall_prioritized(self, query: str, k: int = 5):
        """召回时优先返回高优先级记忆"""
        all_results = self.vectorstore.similarity_search(query, k=k*2)

        # 按优先级排序
        prioritized = sorted(
            all_results,
            key=lambda x: {"high": 3, "medium": 2, "low": 1}.get(
                x.metadata.get("priority", "low"), 0
            ),
            reverse=True
        )

        return prioritized[:k]
```

#### 3.4.2 记忆遗忘机制

```python
from datetime import datetime, timedelta

class ForgettableMemory:
    """带遗忘机制的记忆"""

    def recall_with_decay(self, query: str, k: int = 5):
        """召回记忆，但旧记忆会衰减"""
        results = self.vectorstore.similarity_search(query, k=k*2)

        scored_results = []
        for doc in results:
            # 时间衰减因子
            timestamp = datetime.fromisoformat(doc.metadata["timestamp"])
            age_days = (datetime.now() - timestamp).days

            decay_factor = 1.0 / (1 + age_days / 30)  # 30天衰减一半

            # 相似度 * 衰减因子
            score = doc.metadata.get("similarity_score", 1.0) * decay_factor

            scored_results.append((doc, score))

        # 按分数排序
        scored_results.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, score in scored_results[:k]]
```

---

### 3.5 最佳实践

#### 3.5.1 记忆类型分层

```python
# 推荐的记忆分层架构
memory_system = {
    "短期记忆": {
        "实现": "Checkpointer",
        "范围": "当前会话",
        "容量": "最近10-20轮对话",
        "用途": "上下文理解"
    },

    "工作记忆": {
        "实现": "VectorStore (临时)",
        "范围": "当前任务",
        "容量": "任务相关知识",
        "用途": "任务执行辅助"
    },

    "长期记忆": {
        "实现": "VectorStore (持久)",
        "范围": "所有历史",
        "容量": "无限制",
        "用途": "个性化、知识积累"
    }
}
```

#### 3.5.2 性能优化

**缓存策略**:
```python
from functools import lru_cache

class OptimizedMemory:
    @lru_cache(maxsize=100)
    def recall_cached(self, query: str, k: int = 3):
        """带缓存的召回"""
        return tuple(self.vectorstore.similarity_search(query, k=k))
```

---

### 3.6 总结

#### 3.6.1 核心要点

1. **长期记忆**：跨会话的知识积累
2. **向量召回**：基于语义相似度检索
3. **用户画像**：从历史交互提取偏好
4. **记忆管理**：优先级、遗忘、分层

#### 3.6.2 实施清单

- [ ] 为每个用户创建独立的记忆空间
- [ ] 实现交互自动保存机制
- [ ] 配置合理的召回数量(k=3-5)
- [ ] 定期清理过期/低价值记忆
- [ ] 监控向量数据库性能

---

###思考与练习

1. **思考**：在什么场景下应该选择 Hierarchical 而不是 Supervisor-Worker 模式？

2. **练习**：实现一个支持图像输入的多模态 RAG 系统。

3. **思考**：如何在 WebSocket 实时交互中处理连接断开和重连？

---

## 第4章：Agent 交互协议

> **关注点**：理解 Agent 生态的三大协议标准，掌握 AG-UI 和 A2A 的实战应用。

随着 AI Agent 生态的成熟，三大交互协议逐渐成为行业标准：

- **MCP (Model Context Protocol)**: Agent ↔ 工具/数据（已在 2.3 节详解）
- **AG-UI (Agent-User Interaction Protocol)**: Agent ↔ 用户界面
- **A2A (Agent-to-Agent Protocol)**: Agent ↔ Agent 跨系统协作

```
┌─────────────────────────────────────────────────────────────┐
│                      Agent 协议生态                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────────┐                                          │
│   │   用户界面    │◄────── AG-UI ──────►┌──────────────┐    │
│   │  (Web/App)   │      (事件流)        │    Agent     │    │
│   └──────────────┘                      │   (主体)     │    │
│                                         └──────┬───────┘    │
│                                                │             │
│                         ┌──────────────────────┼──────────┐ │
│                         │                      │          │ │
│                         ▼                      ▼          ▼ │
│                  ┌──────────────┐       ┌──────────────┐    │
│                  │  MCP Server  │       │ Other Agent  │    │
│                  │  (工具/数据)  │◄─MCP─►│   (协作)     │    │
│                  └──────────────┘       └──────────────┘    │
│                                                ▲            │
│                                                │            │
│                                            A2A │            │
│                                         (JSON-RPC)          │
│                                                │            │
│                                      ┌─────────▼────────┐   │
│                                      │  External Agent  │   │
│                                      │  (跨系统/跨公司)  │   │
│                                      └──────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.1 协议对比与选型

| 特性 | MCP (Anthropic) | AG-UI | A2A (Google) |
|------|-----------------|-------|--------------|
| **连接方向** | Agent → 工具/数据 | Agent ↔ 用户界面 | Agent ↔ Agent |
| **主要用途** | 工具调用、数据访问 | 实时UI交互 | 跨系统Agent协作 |
| **通信协议** | JSON-RPC | 事件流 (SSE) | JSON-RPC 2.0 |
| **状态管理** | 无状态 | 共享状态 | 任务状态 |
| **流式支持** | ✅ | ✅ 原生 | ✅ (SSE) |
| **人机交互** | 工具级别 | 原生支持 | 通过任务 |
| **多模态** | 工具返回 | 原生支持 | 文件/媒体 |

**选型建议**：

- **构建聊天UI**：使用 AG-UI
- **调用外部工具**：使用 MCP
- **多Agent协作**：使用 A2A

---

### 4.2 AG-UI 协议详解

AG-UI (Agent-User Interaction Protocol) 是一个开放、轻量级、基于事件的协议，标准化 AI Agent 与用户界面之间的连接。

#### 4.2.1 核心概念

**事件类型**：

| 类别 | 事件 | 用途 |
|------|------|------|
| **生命周期** | RunStarted, RunFinished, RunError | 监控Agent运行状态 |
| **文本消息** | TextMessageStart → Content → End | 流式文本传输 |
| **工具调用** | ToolCallStart → Args → End → Result | 工具执行追踪 |
| **状态管理** | StateSnapshot, StateDelta | 前后端状态同步 |

**流式模式**：

```
# 文本消息流式传输
TextMessageStart(messageId, role)
    → TextMessageContent(delta="Hello")
    → TextMessageContent(delta=" World")
    → TextMessageEnd()

# 工具调用流式传输
ToolCallStart(toolCallId, toolCallName)
    → ToolCallArgs(delta='{"query":')
    → ToolCallArgs(delta='"weather"}')
    → ToolCallEnd()
    → ToolCallResult(content="晴天, 25°C")
```

#### 4.2.2 Python SDK 使用

**安装**：

```bash
pip install ag-ui-core
```

**基础事件定义**：

```python
from dataclasses import dataclass
from typing import Optional, Literal
from enum import Enum

class EventType(str, Enum):
    """AG-UI 事件类型"""
    RUN_STARTED = "RUN_STARTED"
    RUN_FINISHED = "RUN_FINISHED"
    RUN_ERROR = "RUN_ERROR"
    TEXT_MESSAGE_START = "TEXT_MESSAGE_START"
    TEXT_MESSAGE_CONTENT = "TEXT_MESSAGE_CONTENT"
    TEXT_MESSAGE_END = "TEXT_MESSAGE_END"
    TOOL_CALL_START = "TOOL_CALL_START"
    TOOL_CALL_ARGS = "TOOL_CALL_ARGS"
    TOOL_CALL_END = "TOOL_CALL_END"
    TOOL_CALL_RESULT = "TOOL_CALL_RESULT"
    STATE_SNAPSHOT = "STATE_SNAPSHOT"
    STATE_DELTA = "STATE_DELTA"

@dataclass
class BaseEvent:
    """基础事件"""
    type: EventType
    timestamp: Optional[str] = None

@dataclass
class TextMessageContentEvent(BaseEvent):
    """文本内容事件"""
    message_id: str
    delta: str  # 增量文本

@dataclass
class ToolCallStartEvent(BaseEvent):
    """工具调用开始事件"""
    tool_call_id: str
    tool_call_name: str
```

#### 4.2.3 与 LangGraph 集成

**创建 AG-UI 兼容的 Agent 服务器**：

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langgraph.graph import StateGraph, MessagesState
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
import json
import asyncio
from typing import AsyncGenerator

app = FastAPI()

# 创建 LangGraph Agent
def create_chat_agent():
    model = ChatOpenAI(model="gpt-4o-mini")

    def chat_node(state: MessagesState):
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("chat", chat_node)
    builder.set_entry_point("chat")
    builder.set_finish_point("chat")

    return builder.compile()

agent = create_chat_agent()

async def generate_agui_events(
    user_message: str,
    thread_id: str
) -> AsyncGenerator[str, None]:
    """生成 AG-UI 兼容的事件流"""
    import uuid

    run_id = str(uuid.uuid4())
    message_id = str(uuid.uuid4())

    # 1. 发送 RUN_STARTED
    yield f"data: {json.dumps({
        'type': 'RUN_STARTED',
        'threadId': thread_id,
        'runId': run_id
    })}\n\n"

    # 2. 发送 TEXT_MESSAGE_START
    yield f"data: {json.dumps({
        'type': 'TEXT_MESSAGE_START',
        'messageId': message_id,
        'role': 'assistant'
    })}\n\n"

    # 3. 流式输出内容
    config = {"configurable": {"thread_id": thread_id}}

    async for event in agent.astream_events(
        {"messages": [{"role": "user", "content": user_message}]},
        config=config,
        version="v2"
    ):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if hasattr(chunk, "content") and chunk.content:
                yield f"data: {json.dumps({
                    'type': 'TEXT_MESSAGE_CONTENT',
                    'messageId': message_id,
                    'delta': chunk.content
                })}\n\n"
                await asyncio.sleep(0.01)  # 控制流速

    # 4. 发送 TEXT_MESSAGE_END
    yield f"data: {json.dumps({
        'type': 'TEXT_MESSAGE_END',
        'messageId': message_id
    })}\n\n"

    # 5. 发送 RUN_FINISHED
    yield f"data: {json.dumps({
        'type': 'RUN_FINISHED',
        'threadId': thread_id,
        'runId': run_id
    })}\n\n"

@app.post("/chat")
async def chat(request: dict):
    """AG-UI 兼容的聊天端点"""
    user_message = request.get("message", "")
    thread_id = request.get("threadId", "default")

    return StreamingResponse(
        generate_agui_events(user_message, thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
```

**前端消费示例 (JavaScript)**：

```javascript
async function chat(message, threadId) {
    const response = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, threadId })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const lines = decoder.decode(value).split('\n');
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const event = JSON.parse(line.slice(6));
                handleAGUIEvent(event);
            }
        }
    }
}

function handleAGUIEvent(event) {
    switch (event.type) {
        case 'RUN_STARTED':
            console.log('Agent 开始运行:', event.runId);
            break;
        case 'TEXT_MESSAGE_CONTENT':
            // 追加文本到 UI
            appendText(event.delta);
            break;
        case 'TOOL_CALL_START':
            showToolCallIndicator(event.toolCallName);
            break;
        case 'RUN_FINISHED':
            console.log('Agent 运行完成');
            break;
        case 'RUN_ERROR':
            showError(event.message);
            break;
    }
}
```

---

### 4.3 A2A 协议详解

A2A (Agent-to-Agent Protocol) 是 Google 贡献的开放协议，现为 Linux 基金会项目，实现不同 AI Agent 之间的通信和互操作。

#### 4.3.1 核心概念

**设计目标**：

1. **打破孤岛**：连接不同生态系统中的 Agent
2. **启用复杂协作**：专业化 Agent 协同处理复杂任务
3. **保持不透明性**：协作时无需暴露内部状态和实现

**核心组件**：

| 组件 | 描述 |
|------|------|
| **Agent Card** | 描述 Agent 能力和连接信息的元数据 |
| **Task** | 一次完整的 Agent 执行单元 |
| **Context** | 跨多个 Task 的对话线程 |
| **Message** | Agent 间交换的消息 |

**Agent Card 示例**：

```json
{
    "name": "Weather Agent",
    "description": "提供全球天气查询服务",
    "version": "1.0.0",
    "capabilities": ["weather_query", "forecast"],
    "endpoint": "https://weather-agent.example.com/a2a",
    "authentication": {
        "type": "bearer",
        "token_url": "https://auth.example.com/token"
    },
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {"type": "string"},
            "date": {"type": "string", "format": "date"}
        }
    }
}
```

#### 4.3.2 FastA2A + Pydantic AI 实现

Pydantic AI 通过 FastA2A 库提供了 A2A 协议的完整实现，这是目前最简洁的 A2A 实现方式。

**安装**：

```bash
# 方式1: 单独安装 FastA2A
pip install fasta2a

# 方式2: 通过 Pydantic AI 安装
pip install 'pydantic-ai-slim[a2a]'
```

**最简实现 - 一行代码暴露 A2A 服务**：

```python
from pydantic_ai import Agent

# 定义 Agent
agent = Agent(
    'openai:gpt-4o',
    instructions='你是一个有帮助的助手，擅长回答各种问题。'
)

# 一行代码转换为 A2A 服务器
app = agent.to_a2a()

# 运行: uvicorn main:app --host 0.0.0.0 --port 8000
```

**完整示例 - 带工具的 A2A Agent**：

```python
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel
from datetime import date
import httpx

# 定义输出结构
class WeatherResponse(BaseModel):
    location: str
    temperature: float
    condition: str
    humidity: int

# 创建 Agent
weather_agent = Agent(
    'openai:gpt-4o',
    instructions='''你是天气查询助手。
    当用户询问天气时，使用 get_weather 工具获取信息。
    始终用中文回复。''',
    output_type=WeatherResponse
)

@weather_agent.tool
async def get_weather(ctx: RunContext, location: str) -> dict:
    """获取指定位置的天气信息

    Args:
        location: 城市名称，如 "北京"、"上海"
    """
    # 实际应用中调用天气 API
    # 这里使用模拟数据
    weather_data = {
        "北京": {"temp": 22, "condition": "晴", "humidity": 45},
        "上海": {"temp": 26, "condition": "多云", "humidity": 65},
        "广州": {"temp": 30, "condition": "阵雨", "humidity": 80},
    }

    data = weather_data.get(location, {"temp": 20, "condition": "未知", "humidity": 50})
    return {
        "location": location,
        "temperature": data["temp"],
        "condition": data["condition"],
        "humidity": data["humidity"]
    }

# 转换为 A2A 应用
app = weather_agent.to_a2a()
```

**运行和测试**：

```bash
# 启动服务
uvicorn weather_agent:app --host 0.0.0.0 --port 8000

# 测试 Agent Card 端点
curl http://localhost:8000/.well-known/agent.json

# 发送任务请求
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "message": {
      "role": "user",
      "content": "北京今天天气怎么样？"
    }
  }'
```

#### 4.3.3 自定义 FastA2A 配置

**配置存储和任务队列**：

```python
from fasta2a import FastA2A, InMemoryStorage, InMemoryBroker
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o', instructions='你是助手')

# 自定义配置
storage = InMemoryStorage()  # 生产环境可用 PostgresStorage, RedisStorage
broker = InMemoryBroker()    # 生产环境可用 RedisBroker

# 创建 FastA2A 应用
a2a_app = FastA2A(
    agent=agent,
    storage=storage,
    broker=broker,
    name="My Custom Agent",
    description="自定义 A2A Agent 服务",
    version="1.0.0"
)

app = a2a_app.app
```

**多 Agent A2A 协作**：

```python
from pydantic_ai import Agent
from httpx import AsyncClient

# Agent 1: 研究助手
research_agent = Agent(
    'openai:gpt-4o',
    instructions='你是研究助手，负责收集和整理信息。'
)

# Agent 2: 写作助手
writing_agent = Agent(
    'openai:gpt-4o',
    instructions='你是写作助手，根据研究结果撰写内容。'
)

@writing_agent.tool
async def consult_research_agent(ctx: RunContext, query: str) -> str:
    """咨询研究助手获取信息

    Args:
        query: 需要研究的问题
    """
    async with AsyncClient() as client:
        # 调用研究 Agent 的 A2A 端点
        response = await client.post(
            "http://research-agent:8001/tasks",
            json={
                "message": {
                    "role": "user",
                    "content": query
                }
            }
        )
        result = response.json()
        return result.get("output", "无法获取研究结果")

# 各自暴露为 A2A 服务
research_app = research_agent.to_a2a()
writing_app = writing_agent.to_a2a()
```

#### 4.3.4 A2A 与 LangGraph 集成

**将 LangGraph Agent 暴露为 A2A 服务**：

```python
from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from typing import Optional
import uuid

app = FastAPI()

# 创建 LangGraph Agent
def build_agent():
    model = ChatOpenAI(model="gpt-4o-mini")

    def agent_node(state: MessagesState):
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent_node)
    builder.set_entry_point("agent")
    builder.set_finish_point("agent")

    return builder.compile(checkpointer=InMemorySaver())

agent = build_agent()

# A2A Agent Card
@app.get("/.well-known/agent.json")
async def get_agent_card():
    return {
        "name": "LangGraph Assistant",
        "description": "基于 LangGraph 构建的智能助手",
        "version": "1.0.0",
        "capabilities": ["chat", "reasoning"],
        "endpoint": "/tasks"
    }

# A2A 任务请求模型
class TaskRequest(BaseModel):
    message: dict
    context_id: Optional[str] = None

class TaskResponse(BaseModel):
    task_id: str
    context_id: str
    output: str
    status: str

# A2A 任务端点
@app.post("/tasks", response_model=TaskResponse)
async def create_task(request: TaskRequest):
    task_id = str(uuid.uuid4())
    context_id = request.context_id or str(uuid.uuid4())

    # 执行 LangGraph Agent
    config = {"configurable": {"thread_id": context_id}}

    result = await agent.ainvoke(
        {"messages": [request.message]},
        config=config
    )

    # 提取输出
    last_message = result["messages"][-1]
    output = last_message.content if hasattr(last_message, "content") else str(last_message)

    return TaskResponse(
        task_id=task_id,
        context_id=context_id,
        output=output,
        status="completed"
    )
```

---

## 第5章：多 Agent 高级模式

> **关注点**：掌握 Handoffs 和 Skills 两种高级多 Agent 协作模式。

本章补充第1章未覆盖的两种重要多 Agent 模式，与已有的 Supervisor-Worker 模式形成完整的模式体系。

### 5.1 模式对比与选型

| 模式 | 控制方式 | 状态共享 | 并行能力 | 用户交互 | 适用场景 |
|------|---------|---------|---------|---------|---------|
| **Supervisor-Worker** | 集中式 | 经监督者 | 可并行 | 低 | 任务分解、批量处理 |
| **Subagents** | 工具调用 | 隔离 | 高并行 | 低 | 独立子任务、模块化 |
| **Handoffs** | 状态转移 | 直接传递 | 顺序 | 高 | 专业升级、客服系统 |
| **Skills** | 单Agent | 全局 | 中等 | 高 | 多领域问答、知识切换 |
| **Router** | 分类路由 | 隔离 | 高并行 | 中等 | 意图分类、负载分发 |

**选型决策树**：

```
需要多 Agent 协作？
├── 否 → 单 Agent + 工具
└── 是 → Agent 间需要共享完整上下文？
    ├── 否 → 任务可并行？
    │   ├── 是 → Subagents 或 Router
    │   └── 否 → Supervisor-Worker
    └── 是 → 需要专业升级/转接？
        ├── 是 → Handoffs
        └── 否 → Skills
```

---

### 5.2 Handoffs 模式详解

Handoffs（交接）模式通过状态转移实现 Agent 间的控制权传递，特别适合需要"专业升级"的场景。

#### 5.2.1 核心概念

**工作原理**：

```
用户 → Agent A (通用) → 检测到专业问题 → Handoff → Agent B (专家)
                                                      ↓
用户 ← 最终回复 ←────────────────────────────────────┘
```

**与 Supervisor 的区别**：

| 特性 | Supervisor | Handoffs |
|------|-----------|----------|
| 控制流 | 中央集中 | 分布式传递 |
| 状态 | 经监督者 | 直接传递 |
| 回退 | 回到监督者 | 可回退到前一个 |
| 适用 | 任务分解 | 专业升级 |

#### 5.2.2 LangGraph 实现

```python
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.types import Command
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from typing import Literal, TypedDict
from dataclasses import dataclass

# 定义状态
class HandoffState(MessagesState):
    current_agent: str
    handoff_reason: str | None

# 定义 Handoff 命令
@dataclass
class Handoff:
    target: str
    reason: str

# 通用客服 Agent
def general_support_agent(state: HandoffState) -> Command[Literal["technical", "billing", "__end__"]]:
    """通用客服 Agent，处理简单问题或转接专家"""
    model = ChatOpenAI(model="gpt-4o-mini")

    system_prompt = """你是通用客服助手。

    职责：
    1. 处理简单的问候和通用问题
    2. 识别需要专业帮助的问题并转接

    转接规则：
    - 技术问题（代码、API、错误）→ 回复 [HANDOFF:technical]
    - 账单问题（付款、退款、订阅）→ 回复 [HANDOFF:billing]
    - 其他问题 → 直接回答
    """

    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = model.invoke(messages)
    content = response.content

    # 检测是否需要转接
    if "[HANDOFF:technical]" in content:
        return Command(
            goto="technical",
            update={
                "messages": [AIMessage(content="正在为您转接技术专家...")],
                "current_agent": "technical",
                "handoff_reason": "技术问题"
            }
        )
    elif "[HANDOFF:billing]" in content:
        return Command(
            goto="billing",
            update={
                "messages": [AIMessage(content="正在为您转接账务专家...")],
                "current_agent": "billing",
                "handoff_reason": "账务问题"
            }
        )
    else:
        return Command(
            goto="__end__",
            update={
                "messages": [response],
                "current_agent": "general"
            }
        )

# 技术支持 Agent
def technical_support_agent(state: HandoffState) -> Command[Literal["general", "__end__"]]:
    """技术支持专家"""
    model = ChatOpenAI(model="gpt-4o")

    system_prompt = """你是技术支持专家，擅长：
    - 代码调试和错误排查
    - API 使用指导
    - 系统架构建议

    如果问题不属于技术范畴，回复 [HANDOFF:general] 转回通用客服。
    """

    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = model.invoke(messages)

    if "[HANDOFF:general]" in response.content:
        return Command(
            goto="general",
            update={
                "messages": [AIMessage(content="这个问题我来帮您转接通用客服...")],
                "current_agent": "general"
            }
        )

    return Command(
        goto="__end__",
        update={"messages": [response]}
    )

# 账务支持 Agent
def billing_support_agent(state: HandoffState) -> Command[Literal["general", "__end__"]]:
    """账务支持专家"""
    model = ChatOpenAI(model="gpt-4o")

    system_prompt = """你是账务支持专家，擅长：
    - 账单查询和解释
    - 退款处理流程
    - 订阅管理

    如果问题不属于账务范畴，回复 [HANDOFF:general] 转回通用客服。
    """

    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = model.invoke(messages)

    if "[HANDOFF:general]" in response.content:
        return Command(
            goto="general",
            update={
                "messages": [AIMessage(content="这个问题我来帮您转接通用客服...")],
                "current_agent": "general"
            }
        )

    return Command(
        goto="__end__",
        update={"messages": [response]}
    )

# 构建工作流
def build_handoff_workflow():
    builder = StateGraph(HandoffState)

    # 添加节点
    builder.add_node("general", general_support_agent)
    builder.add_node("technical", technical_support_agent)
    builder.add_node("billing", billing_support_agent)

    # 设置入口
    builder.set_entry_point("general")

    return builder.compile()

# 使用示例
async def main():
    workflow = build_handoff_workflow()

    # 测试技术问题
    result = await workflow.ainvoke({
        "messages": [HumanMessage(content="我的 API 调用返回 401 错误，怎么解决？")],
        "current_agent": "general",
        "handoff_reason": None
    })

    for msg in result["messages"]:
        print(f"{msg.__class__.__name__}: {msg.content}")
```

---

### 5.3 Skills 模式详解

Skills（技能）模式让单个 Agent 动态加载不同的专业能力，无需创建多个 Agent 实例。

#### 5.3.1 核心概念

**工作原理**：

```
用户问题 → Agent → 识别领域 → 加载对应 Skill → 使用专业 prompt/工具 → 回复
                      ↓
              ┌───────┴───────┐
              ↓       ↓       ↓
           SQL技能  Python技能  通用技能
```

**优势**：

- 减少 Agent 实例和通信开销
- 上下文完全共享
- 切换速度快

#### 5.3.2 实现方式

```python
from langgraph.graph import StateGraph, MessagesState, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from typing import TypedDict, Callable
from dataclasses import dataclass

# 定义技能
@dataclass
class Skill:
    name: str
    description: str
    system_prompt: str
    tools: list[Callable]

# SQL 技能
@tool
def execute_sql(query: str) -> str:
    """执行 SQL 查询（模拟）"""
    # 实际应用中连接数据库
    return f"执行查询: {query}\n结果: [模拟数据]"

sql_skill = Skill(
    name="sql",
    description="SQL 数据库查询和分析",
    system_prompt="""你是 SQL 专家。
    - 帮助用户编写和优化 SQL 查询
    - 解释查询结果
    - 使用 execute_sql 工具执行查询""",
    tools=[execute_sql]
)

# Python 技能
@tool
def execute_python(code: str) -> str:
    """执行 Python 代码（模拟）"""
    # 实际应用中使用沙箱执行
    return f"执行代码:\n{code}\n输出: [模拟结果]"

python_skill = Skill(
    name="python",
    description="Python 编程和数据处理",
    system_prompt="""你是 Python 专家。
    - 帮助用户编写 Python 代码
    - 解释代码逻辑
    - 使用 execute_python 工具运行代码""",
    tools=[execute_python]
)

# 通用技能
general_skill = Skill(
    name="general",
    description="通用问答和闲聊",
    system_prompt="你是一个友好的助手，可以回答各种问题。",
    tools=[]
)

# 技能注册表
SKILLS = {
    "sql": sql_skill,
    "python": python_skill,
    "general": general_skill
}

# 状态定义
class SkillState(MessagesState):
    current_skill: str
    skill_history: list[str]

# 技能路由器
def skill_router(state: SkillState) -> str:
    """根据用户输入选择合适的技能"""
    model = ChatOpenAI(model="gpt-4o-mini")

    last_message = state["messages"][-1].content

    router_prompt = f"""分析用户问题，选择最合适的技能。

可用技能：
- sql: SQL 数据库相关问题
- python: Python 编程相关问题
- general: 其他通用问题

用户问题: {last_message}

只回复技能名称（sql/python/general）："""

    response = model.invoke([{"role": "user", "content": router_prompt}])
    skill_name = response.content.strip().lower()

    return skill_name if skill_name in SKILLS else "general"

# 技能执行器
def skill_executor(state: SkillState) -> dict:
    """使用当前技能处理用户请求"""
    skill = SKILLS[state["current_skill"]]

    # 创建带工具的模型
    model = ChatOpenAI(model="gpt-4o-mini")
    if skill.tools:
        model = model.bind_tools(skill.tools)

    # 构建消息
    messages = [
        SystemMessage(content=skill.system_prompt),
        *state["messages"]
    ]

    # 执行
    response = model.invoke(messages)

    # 处理工具调用
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_results = []
        for tool_call in response.tool_calls:
            tool_fn = next(t for t in skill.tools if t.name == tool_call["name"])
            result = tool_fn.invoke(tool_call["args"])
            tool_results.append(f"工具 {tool_call['name']} 结果: {result}")

        # 再次调用获取最终回复
        messages.append(response)
        messages.append(HumanMessage(content="\n".join(tool_results)))
        response = model.invoke(messages)

    return {
        "messages": [response],
        "skill_history": state.get("skill_history", []) + [state["current_skill"]]
    }

# 构建工作流
def build_skill_workflow():
    builder = StateGraph(SkillState)

    # 路由节点
    def route_and_update(state: SkillState) -> dict:
        selected_skill = skill_router(state)
        return {"current_skill": selected_skill}

    builder.add_node("router", route_and_update)
    builder.add_node("executor", skill_executor)

    # 边
    builder.set_entry_point("router")
    builder.add_edge("router", "executor")
    builder.add_edge("executor", END)

    return builder.compile()

# 使用示例
async def main():
    workflow = build_skill_workflow()

    # 测试 SQL 问题
    result = await workflow.ainvoke({
        "messages": [HumanMessage(content="如何查询用户表中年龄大于 25 的记录？")],
        "current_skill": "",
        "skill_history": []
    })

    print(f"使用技能: {result['current_skill']}")
    print(f"回复: {result['messages'][-1].content}")

    # 测试 Python 问题
    result = await workflow.ainvoke({
        "messages": [HumanMessage(content="用 Python 写一个快速排序算法")],
        "current_skill": "",
        "skill_history": []
    })

    print(f"使用技能: {result['current_skill']}")
    print(f"回复: {result['messages'][-1].content}")
```

#### 5.3.3 高级 Skills 模式：动态技能注册

```python
from typing import Protocol, runtime_checkable
from abc import abstractmethod

@runtime_checkable
class SkillProtocol(Protocol):
    """技能协议"""
    name: str
    description: str

    @abstractmethod
    def get_system_prompt(self) -> str: ...

    @abstractmethod
    def get_tools(self) -> list: ...

class SkillRegistry:
    """技能注册表"""

    def __init__(self):
        self._skills: dict[str, SkillProtocol] = {}

    def register(self, skill: SkillProtocol) -> None:
        """注册技能"""
        self._skills[skill.name] = skill

    def unregister(self, name: str) -> None:
        """注销技能"""
        self._skills.pop(name, None)

    def get(self, name: str) -> SkillProtocol | None:
        """获取技能"""
        return self._skills.get(name)

    def list_skills(self) -> list[dict]:
        """列出所有技能"""
        return [
            {"name": s.name, "description": s.description}
            for s in self._skills.values()
        ]

    def create_router_prompt(self) -> str:
        """生成路由提示词"""
        skills_desc = "\n".join(
            f"- {s.name}: {s.description}"
            for s in self._skills.values()
        )
        return f"""根据用户问题选择最合适的技能。

可用技能：
{skills_desc}

只回复技能名称："""

# 全局注册表
skill_registry = SkillRegistry()

# 注册默认技能
skill_registry.register(sql_skill)
skill_registry.register(python_skill)
skill_registry.register(general_skill)
```

---

### 本章小结

#### 5.4.1 核心要点

1. **AG-UI 协议**：标准化 Agent 与 UI 的交互，支持流式传输
2. **A2A 协议**：实现跨系统 Agent 协作，保护内部实现
3. **Handoffs 模式**：状态驱动的专业升级转接
4. **Skills 模式**：单 Agent 动态能力切换

#### 5.4.2 选型建议

| 需求 | 推荐方案 |
|------|---------|
| 构建聊天 UI | AG-UI + LangGraph |
| 跨系统 Agent 调用 | A2A (FastA2A) |
| 客服专业转接 | Handoffs |
| 多领域问答 | Skills |
| 任务并行处理 | Subagents 或 Router |

#### 5.4.3 实施清单

- [ ] 评估是否需要跨系统协作（→ A2A）
- [ ] 确定 UI 交互需求（→ AG-UI）
- [ ] 选择合适的多 Agent 模式
- [ ] 实现协议适配层
- [ ] 添加监控和错误处理

---

## 总结

本篇涵盖了 LangChain 生态的高级应用：

- **多 Agent 系统设计模式**（Supervisor-Worker、Subagents、Handoffs、Skills、Router）
- **多模态 AI 应用开发**
- **实时流式交互实现**
- **MCP 协议集成**（Agent ↔ 工具/数据）
- **AG-UI 协议集成**（Agent ↔ 用户界面）
- **A2A 协议集成**（Agent ↔ Agent 跨系统协作）
- **自定义组件开发**

这些高级特性让你能够构建更复杂、更强大的 AI 应用系统。

---

**参考资源**：

- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [MCP 协议规范](https://modelcontextprotocol.io)
- [AG-UI 协议文档](https://docs.ag-ui.com)
- [A2A 协议规范](https://a2a-protocol.org)
- [Pydantic AI 文档](https://ai.pydantic.dev)
- [FastA2A 文档](https://ai.pydantic.dev/a2a/)
- [LangChain 多模态指南](https://python.langchain.com/docs/use_cases/multimodal)