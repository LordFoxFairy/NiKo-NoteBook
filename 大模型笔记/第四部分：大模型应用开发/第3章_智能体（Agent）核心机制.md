# 第3章:智能体(Agent)核心机制

> "The future of AI is not just about better models, but about better systems." - Andrew Ng
>
> 智能体(Agent)将 LLM 从"大脑"变成了"双手",让 AI 具备了与世界交互的能力。

---

## 本章导读

本章专注于 **Agent 设计模式与工程实现**,是构建自主智能系统的核心技术。我们将深入探讨:
- **ReAct/Plan-and-Solve** 等规划模式的代码实现
- **Tool Use / Function Calling** 的 JSON Schema 定义与解析
- **MCP (Model Context Protocol)** 协议标准与实战
- **LangGraph** 的 StateGraph 编程范式
- **Memory 系统**的短期/长期记忆设计
- **Multi-Agent** 协作模式 (Supervisor/Hierarchical)

**边界说明** (参考 chapter-boundaries.md):
- ✅ **本章包含**: Agent 架构设计、工具调用、MCP 协议、多智能体协作、Memory 机制
- ❌ **不包含**: CoT 数学原理 (→ Part 7 Ch3)、推理时搜索/MCTS (→ Part 7 Ch4)、强化学习训练 Agent (→ Part 7 Ch4)

---

## 目录
- [一、从 Prompt Engineering 到 Agentic Workflow](#一从-prompt-engineering-到-agentic-workflow)
- [二、规划 (Planning):ReAct 与 Plan-and-Solve](#二规划-planningreact-与-plan-and-solve)
- [三、工具使用 (Tool Use) 与 Function Calling](#三工具使用-tool-use-与-function-calling)
- [四、MCP (Model Context Protocol) 革命](#四mcp-model-context-protocol-革命)
- [五、记忆系统 (Memory) 设计](#五记忆系统-memory-设计)
- [六、LangGraph:状态机编程范式](#六langgraph状态机编程范式)
- [七、多智能体协作 (Multi-Agent)](#七多智能体协作-multi-agent)
- [八、Output Parser:结构化输出解析](#八output-parser结构化输出解析)
- [九、本章小结](#九本章小结)

---

## 一、从 Prompt Engineering 到 Agentic Workflow

Andrew Ng 在 2024 年提出一个重要观点:**与其追求更强的模型 (GPT-5),不如优化 Agent 工作流 (Agentic Workflow)**。
GPT-3.5 + 良好的工作流,往往能超越零样本的 GPT-4。

### 1. 什么是 Agentic Workflow?

**对比理解**:
- **Zero-Shot**: 就像让一个人"一口气"写完一篇论文,不许查资料,不许修改
- **Agentic Workflow**: 让 LLM 进行**迭代处理** - 列提纲 → 查资料 → 写初稿 → 自我修改 → 定稿

**核心差异**: 允许 LLM 进行多轮迭代,每一步都可以反思和纠错。

### 2. 四种核心设计模式

| 模式 | 核心思想 | 典型场景 |
|------|----------|----------|
| **Reflection** | 让模型检查自己的输出 | 代码审查、论文润色 |
| **Tool Use** | 模型知道何时求助外部工具 | 计算器、搜索引擎、数据库查询 |
| **Planning** | 先拆解步骤,再逐一执行 | 复杂任务分解 |
| **Multi-Agent** | 不同角色协作完成任务 | 软件开发团队模拟 |

---

## 二、规划 (Planning):ReAct 与 Plan-and-Solve

### 1. ReAct 模式:代码实现

ReAct (Reason + Act) 是最经典的 Agent 模式,由 Yao et al. (2022) 提出。

**核心思想**: Thought (思考) → Action (行动) → Observation (观察) 循环。

#### (1) Prompt 模板

```python
REACT_PROMPT_TEMPLATE = """你是一个问题求解助手。请按照以下格式回答:

Question: {question}

Thought: 我需要思考如何解决这个问题
Action: [工具名称][参数]
Observation: [工具返回结果]
... (重复 Thought/Action/Observation 可多次)
Thought: 我现在知道最终答案了
Final Answer: [最终答案]

可用工具:
- Search[query]: 搜索互联网
- Calculator[expression]: 计算数学表达式
- WikiLookup[term]: 查询维基百科

示例:
Question: 埃菲尔铁塔的高度是多少?它比上海中心大厦高多少?
Thought: 我需要先查询埃菲尔铁塔的高度
Action: WikiLookup[埃菲尔铁塔]
Observation: 埃菲尔铁塔高 324 米
Thought: 现在我需要查上海中心大厦的高度
Action: WikiLookup[上海中心大厦]
Observation: 上海中心大厦高 632 米
Thought: 我需要计算差值
Action: Calculator[632 - 324]
Observation: 308
Thought: 我现在知道答案了
Final Answer: 埃菲尔铁塔高 324 米,比上海中心大厦矮 308 米。

现在开始:
Question: {question}"""
```

#### (2) 完整实现 (Python 伪代码)

```python
import re
from typing import Dict, Callable

class ReActAgent:
    def __init__(self, llm, tools: Dict[str, Callable], max_steps=5):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
    
    def run(self, question: str) -> str:
        prompt = REACT_PROMPT_TEMPLATE.format(question=question)
        scratchpad = ""  # 记录思考过程
        
        for step in range(self.max_steps):
            # 1. LLM 生成下一步思考
            response = self.llm.invoke(prompt + scratchpad)
            scratchpad += response
            
            # 2. 解析 Action
            action_match = re.search(r'Action: (\w+)\[(.*?)\]', response)
            
            if "Final Answer:" in response:
                # 找到最终答案,结束循环
                answer = response.split("Final Answer:")[1].strip()
                return answer
            
            if action_match:
                tool_name = action_match.group(1)
                tool_input = action_match.group(2)
                
                # 3. 执行工具
                if tool_name in self.tools:
                    observation = self.tools[tool_name](tool_input)
                else:
                    observation = f"Error: Tool {tool_name} not found"
                
                # 4. 添加 Observation 到 scratchpad
                scratchpad += f"
Observation: {observation}
"
            else:
                scratchpad += "
[No valid action found. Please try again.]
"
        
        return "Max steps reached. No answer found."

# 示例工具
def calculator(expr: str) -> str:
    try:
        return str(eval(expr))  # 生产环境请用 ast.literal_eval!
    except Exception as e:
        return f"Error: {e}"

def search(query: str) -> str:
    # 模拟搜索
    return f"Mock search result for '{query}'"

# 使用
agent = ReActAgent(
    llm=your_llm_instance,
    tools={"Calculator": calculator, "Search": search}
)
result = agent.run("如果我有 3 个苹果,买了 7 个橙子,一共多少水果?")
```

**局限性分析**:
- 短视: 只看下一步,缺乏全局观
- 死循环风险: 容易在两个步骤之间反复横跳
- Token 消耗大: 每一步都要把整个 History 喂给模型

### 2. Plan-and-Solve:结构化规划

**核心思想**: 先生成完整计划,再逐一执行,避免 ReAct 的短视问题。

#### (1) Prompt 模板

```python
PLAN_PROMPT = """请为以下任务生成详细的执行计划:

任务: {task}

可用工具:
{tools_description}

请以 JSON 格式输出计划:
{{
  "steps": [
    {{"step": 1, "action": "tool_name", "args": {{"arg1": "value"}}, "reason": "原因"}},
    {{"step": 2, "action": "tool_name", "args": {{"arg1": "value"}}, "reason": "原因"}}
  ]
}}
"""
```

#### (2) 实现

```python
import json
from pydantic import BaseModel, Field
from typing import List, Dict

class PlanStep(BaseModel):
    step: int
    action: str
    args: Dict
    reason: str

class Plan(BaseModel):
    steps: List[PlanStep]

class PlanAndSolveAgent:
    def __init__(self, llm, tools: Dict[str, Callable]):
        self.llm = llm
        self.tools = tools
    
    def run(self, task: str) -> str:
        # 1. 生成计划
        plan_response = self.llm.invoke(PLAN_PROMPT.format(
            task=task,
            tools_description=self._get_tools_description()
        ))
        
        # 2. 解析计划
        try:
            plan_data = json.loads(plan_response)
            plan = Plan(**plan_data)
        except Exception as e:
            return f"Failed to parse plan: {e}"
        
        # 3. 执行计划
        results = []
        for step in plan.steps:
            print(f"Step {step.step}: {step.reason}")
            tool_result = self.tools[step.action](**step.args)
            results.append(tool_result)
        
        # 4. 综合结果
        return self._synthesize_results(task, results)
    
    def _get_tools_description(self) -> str:
        return "
".join([f"- {name}: {func.__doc__}" for name, func in self.tools.items()])
    
    def _synthesize_results(self, task: str, results: List) -> str:
        # 让 LLM 综合结果
        prompt = f"Task: {task}

Execution results: {results}

Please provide final answer:"
        return self.llm.invoke(prompt)
```

### 3. Reflection:自我反思机制

Reflection 让 Agent 具备"自我纠错"能力,核心是引入 **Evaluator** 和 **Self-Reflection** 步骤。

#### (1) Reflexion 架构

```python
class ReflexionAgent:
    def __init__(self, llm, tools, max_trials=3):
        self.llm = llm
        self.tools = tools
        self.max_trials = max_trials
        self.memory = []  # 存储反思经验
    
    def run(self, task: str) -> str:
        for trial in range(self.max_trials):
            print(f"
=== Trial {trial + 1} ===")
            
            # 1. 尝试执行任务 (带上之前的经验)
            context = self._build_context(task)
            result = self._attempt_task(context)
            
            # 2. 评估结果
            is_success, feedback = self._evaluate(task, result)
            
            if is_success:
                return result
            
            # 3. 失败则进行反思
            reflection = self._reflect(task, result, feedback)
            self.memory.append(reflection)
            print(f"Reflection: {reflection}")
        
        return f"Failed after {self.max_trials} trials"
    
    def _build_context(self, task: str) -> str:
        lessons = "
".join([f"- {r}" for r in self.memory])
        return f"""Task: {task}

Previous failures and lessons learned:
{lessons if lessons else "None"}

Please try to solve the task while avoiding previous mistakes."""
    
    def _attempt_task(self, context: str) -> str:
        # 使用 ReAct 或其他方式执行
        return self.llm.invoke(context)
    
    def _evaluate(self, task: str, result: str) -> tuple[bool, str]:
        """评估结果是否成功"""
        eval_prompt = f"""Task: {task}
Result: {result}

Is this result correct and complete? Answer with:
- SUCCESS: if correct
- FAILURE: reason why it failed"""
        
        eval_response = self.llm.invoke(eval_prompt)
        
        if "SUCCESS" in eval_response:
            return True, ""
        else:
            return False, eval_response
    
    def _reflect(self, task: str, result: str, feedback: str) -> str:
        """生成反思"""
        reflect_prompt = f"""You failed at the following task:
Task: {task}
Your result: {result}
Feedback: {feedback}

Please analyze what went wrong and provide a lesson for next time.
Focus on:
1. What assumption was incorrect?
2. What should you do differently next time?

Reflection:"""
        
        return self.llm.invoke(reflect_prompt)
```

#### (2) Output Parser 实现

为了确保 LLM 输出符合预期格式,我们需要 **结构化解析器**。

```python
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator

class ReflectionOutput(BaseModel):
    status: str = Field(description="SUCCESS or FAILURE")
    reason: str = Field(description="Reason for success/failure")
    lesson: str = Field(default="", description="Lesson learned if failed")
    
    @validator('status')
    def validate_status(cls, v):
        if v not in ["SUCCESS", "FAILURE"]:
            raise ValueError("Status must be SUCCESS or FAILURE")
        return v

# 使用 Parser
parser = PydanticOutputParser(pydantic_object=ReflectionOutput)
format_instructions = parser.get_format_instructions()

# 在 Prompt 中加入格式说明
eval_prompt = f"""Task: {task}
Result: {result}

{format_instructions}"""

# 解析输出
try:
    parsed = parser.parse(llm_output)
    print(parsed.status, parsed.reason)
except Exception as e:
    print(f"Parsing failed: {e}")
```

---

## 三、工具使用 (Tool Use) 与 Function Calling

### 1. JSON Schema 定义标准

OpenAI 的 Function Calling 使用 **JSON Schema** 定义工具接口。

#### (1) 标准格式

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的当前天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称,例如: Beijing, Shanghai"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位"
                    }
                },
                "required": ["city"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "在互联网上搜索信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "返回结果数量",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    }
                },
                "required": ["query"]
            }
        }
    }
]
```

#### (2) 使用 Pydantic 简化定义

```python
from pydantic import BaseModel, Field

class GetWeatherInput(BaseModel):
    city: str = Field(description="城市名称,例如: Beijing, Shanghai")
    unit: str = Field(default="celsius", description="温度单位", enum=["celsius", "fahrenheit"])

class SearchWebInput(BaseModel):
    query: str = Field(description="搜索关键词")
    max_results: int = Field(default=5, ge=1, le=20, description="返回结果数量")

# 自动生成 JSON Schema
from pydantic.json_schema import JsonSchemaValue

def pydantic_to_openai_schema(model: type[BaseModel]) -> dict:
    schema = model.model_json_schema()
    return {
        "type": "function",
        "function": {
            "name": model.__name__,
            "description": model.__doc__ or "",
            "parameters": schema
        }
    }

tools = [
    pydantic_to_openai_schema(GetWeatherInput),
    pydantic_to_openai_schema(SearchWebInput)
]
```

### 2. Function Calling 协议详解

#### (1) 完整调用流程

```python
import openai
import json

def get_weather(city: str, unit: str = "celsius") -> str:
    # 模拟天气 API
    return json.dumps({
        "city": city,
        "temperature": 25,
        "unit": unit,
        "condition": "sunny"
    })

def search_web(query: str, max_results: int = 5) -> str:
    # 模拟搜索 API
    return json.dumps({
        "query": query,
        "results": [f"Result {i+1} for {query}" for i in range(max_results)]
    })

# 工具映射
available_functions = {
    "get_weather": get_weather,
    "search_web": search_web
}

# 第一轮: 让模型决定调用什么工具
messages = [
    {"role": "user", "content": "北京今天天气怎么样?"}
]

response = openai.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=tools,
    tool_choice="auto"  # 让模型自动决定
)

# 检查是否要调用工具
response_message = response.choices[0].message
messages.append(response_message)

if response_message.tool_calls:
    # 第二轮: 执行工具并返回结果
    for tool_call in response_message.tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        print(f"Calling {function_name} with {function_args}")
        
        # 执行工具
        function_response = available_functions[function_name](**function_args)
        
        # 添加工具结果到对话
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": function_name,
            "content": function_response
        })
    
    # 第三轮: 让模型综合工具结果给出最终答案
    final_response = openai.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    
    print(final_response.choices[0].message.content)
```

### 3. 工具调用完整流程实现

#### (1) LangChain 实现

```python
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 定义工具
@tool
def calculator(expression: str) -> str:
    """计算数学表达式,例如: 3 * (2 + 5)"""
    try:
        result = eval(expression)  # 生产环境使用 ast.literal_eval
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

@tool
def get_current_time() -> str:
    """获取当前时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 创建 Agent
llm = ChatOpenAI(model="gpt-4", temperature=0)
tools = [calculator, get_current_time]

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个智能助手,可以使用工具来帮助用户。"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 运行
result = agent_executor.invoke({"input": "现在几点?计算 (3 + 5) * 2 的结果"})
print(result["output"])
```

---

## 四、MCP (Model Context Protocol) 革命

### 1. MCP 协议标准

2024年底,Anthropic 推出了 **MCP (Model Context Protocol)**,这是一个开放标准,旨在统一 **AI 模型** 与 **数据源/工具** 之间的连接。

**核心价值**:
- 类似于 USB 协议: 一次编写,处处运行
- 数据源和工具可以无缝接入任何支持 MCP 的 LLM
- 避免为每个 LLM 单独开发插件

**架构**:
```
┌─────────────────┐
│  MCP Client     │  (Claude Desktop, Cursor, VS Code)
│  (LLM App)      │
└────────┬────────┘
         │ MCP Protocol
         │
┌────────┴────────┐
│  MCP Server     │  (Google Drive, Slack, Postgres, File System)
│  (Tool/Data)    │
└─────────────────┘
```

**核心概念**:
- **Resources**: 数据源 (文件、数据库记录、API 端点)
- **Tools**: 可执行的函数
- **Prompts**: 预定义的提示词模板

### 2. 实战:实现一个 MCP Server

我们用 Python `mcp` 库写一个文件系统 MCP Server。

#### (1) 安装依赖

```bash
pip install mcp
```

#### (2) 服务端代码

```python
# file_system_mcp_server.py
from mcp.server.fastmcp import FastMCP
import os
from pathlib import Path

# 创建 MCP Server
mcp = FastMCP("FileSystemServer")

@mcp.tool()
def read_file(path: str) -> str:
    """
    读取文件内容
    
    Args:
        path: 文件路径
    
    Returns:
        文件内容或错误信息
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return f"File content ({len(content)} chars):
{content}"
    except Exception as e:
        return f"Error reading file: {e}"

@mcp.tool()
def write_file(path: str, content: str) -> str:
    """
    写入文件
    
    Args:
        path: 文件路径
        content: 要写入的内容
    
    Returns:
        成功或错误信息
    """
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} chars to {path}"
    except Exception as e:
        return f"Error writing file: {e}"

@mcp.tool()
def list_directory(path: str = ".") -> str:
    """
    列出目录内容
    
    Args:
        path: 目录路径 (默认为当前目录)
    
    Returns:
        文件列表
    """
    try:
        items = os.listdir(path)
        files = [f"📄 {item}" if os.path.isfile(os.path.join(path, item)) else f"📁 {item}" for item in items]
        return f"Directory '{path}' contains {len(items)} items:
" + "
".join(files)
    except Exception as e:
        return f"Error listing directory: {e}"

@mcp.resource("file://{path}")
def get_file_resource(path: str) -> str:
    """
    提供文件作为资源
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"

# 运行 Server
if __name__ == "__main__":
    print("Starting File System MCP Server...")
    mcp.run()
```

#### (3) 配置文件 (用于 Claude Desktop)

在 `~/Library/Application Support/Claude/claude_desktop_config.json` 中添加:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "python3",
      "args": ["/path/to/file_system_mcp_server.py"]
    }
  }
}
```

### 3. MCP Client 集成

#### (1) 在代码中连接 MCP Server

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def use_mcp_server():
    # 连接到 MCP Server
    server_params = StdioServerParameters(
        command="python3",
        args=["file_system_mcp_server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化
            await session.initialize()
            
            # 列出可用工具
            tools = await session.list_tools()
            print("Available tools:", [t.name for t in tools.tools])
            
            # 调用工具
            result = await session.call_tool("list_directory", {"path": "."})
            print(result.content)
            
            # 读取资源
            resources = await session.list_resources()
            if resources.resources:
                content = await session.read_resource(resources.resources[0].uri)
                print(content.contents)

# 运行
import asyncio
asyncio.run(use_mcp_server())
```

---

## 五、记忆系统 (Memory) 设计

### 1. 记忆类型:短期与长期

**人类记忆体系**:
- **感官记忆** (Sensory Memory): 毫秒级,瞬间消失
- **短期记忆** (Short-term / Working Memory): 秒到分钟级,容量有限 (约 7±2 chunks)
- **长期记忆** (Long-term Memory):
  - **陈述性记忆** (Declarative): 事实和事件
    - **语义记忆**: 世界知识 (巴黎是法国的首都)
    - **情景记忆**: 个人经历 (我上周去了巴黎)
  - **程序性记忆** (Procedural): 技能 (如何骑自行车)

**Agent 记忆映射**:
| 人类记忆 | Agent 实现 | 技术方案 |
|----------|-----------|----------|
| 短期记忆 | Context Window | 直接存储在 Prompt 中 (最近几轮对话) |
| 语义记忆 | World Knowledge | 模型预训练权重 + RAG 知识库 |
| 情景记忆 | Experience Buffer | 向量数据库 (存储过往交互) |
| 程序性记忆 | Skill Library | 微调权重 + Tool Definitions |

### 2. Memory 架构设计

#### (1) 简单 Memory:会话级存储

```python
from langchain.memory import ConversationBufferMemory

class SimpleMemory:
    def __init__(self, max_turns=10):
        self.messages = []
        self.max_turns = max_turns
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        # 保持窗口大小
        if len(self.messages) > self.max_turns * 2:
            self.messages = self.messages[-self.max_turns * 2:]
    
    def get_history(self) -> str:
        return "
".join([f"{m['role']}: {m['content']}" for m in self.messages])
```

#### (2) 摘要记忆:压缩历史

```python
from langchain.memory import ConversationSummaryMemory

class SummaryMemory:
    def __init__(self, llm, max_token_limit=2000):
        self.llm = llm
        self.messages = []
        self.summary = ""
        self.max_token_limit = max_token_limit
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        
        # 检查是否需要压缩
        total_tokens = self._estimate_tokens()
        if total_tokens > self.max_token_limit:
            self._compress_history()
    
    def _estimate_tokens(self) -> int:
        # 简单估算: 1 token ≈ 4 chars
        text = self.summary + "
".join([m['content'] for m in self.messages])
        return len(text) // 4
    
    def _compress_history(self):
        """压缩历史为摘要"""
        history_text = "
".join([f"{m['role']}: {m['content']}" for m in self.messages])
        
        prompt = f"""Please summarize the following conversation history concisely:

{history_text}

Summary:"""
        
        new_summary = self.llm.invoke(prompt)
        
        # 更新
        self.summary = new_summary if not self.summary else f"{self.summary}
{new_summary}"
        self.messages = []  # 清空已压缩的消息
    
    def get_context(self) -> str:
        recent = "
".join([f"{m['role']}: {m['content']}" for m in self.messages])
        if self.summary:
            return f"Previous summary:
{self.summary}

Recent messages:
{recent}"
        return recent
```

#### (3) 向量记忆:语义检索

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

class VectorMemory:
    def __init__(self, collection_name="agent_memory"):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings
        )
    
    def add_experience(self, content: str, metadata: dict = None):
        """添加经验到长期记忆"""
        doc = Document(page_content=content, metadata=metadata or {})
        self.vectorstore.add_documents([doc])
    
    def retrieve_relevant(self, query: str, k=3) -> list[str]:
        """检索相关经验"""
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
    
    def get_context_for_query(self, query: str, k=3) -> str:
        """为当前查询构建记忆上下文"""
        relevant_memories = self.retrieve_relevant(query, k=k)
        if not relevant_memories:
            return ""
        
        return "Relevant past experiences:
" + "
".join(
            [f"- {mem}" for mem in relevant_memories]
        )
```

### 3. MemGPT:虚拟内存管理

MemGPT (Packer et al., 2023) 提出了类似操作系统的**虚拟内存管理**机制。

**核心概念**:
- **Main Context** (主上下文): 相当于 RAM,存放当前活跃信息
- **Archival Memory** (归档记忆): 相当于 Disk,存放历史信息
- **Recall Memory** (召回记忆): 核心工作记忆

**架构图**:
```
┌─────────────────────────────────┐
│  Main Context (Limited Size)   │
│  ┌─────────────────────────┐   │
│  │ System Prompt           │   │
│  │ Recent Messages         │   │
│  │ Recalled Memories       │   │
│  └─────────────────────────┘   │
└──────────┬──────────────────────┘
           │
           │ memory_insert() / memory_search()
           │
┌──────────▼──────────────────────┐
│  Archival Memory (Unlimited)   │
│  (Vector Database)              │
│  - Past conversations           │
│  - Knowledge snippets           │
│  - Reflections                  │
└─────────────────────────────────┘
```

**实现示例**:

```python
class MemGPTAgent:
    def __init__(self, llm, vector_memory: VectorMemory):
        self.llm = llm
        self.vector_memory = vector_memory
        self.main_context = []
        self.max_context_size = 10
        
        # 定义记忆管理工具
        self.tools = {
            "archival_memory_insert": self.archival_memory_insert,
            "archival_memory_search": self.archival_memory_search,
            "core_memory_append": self.core_memory_append,
            "core_memory_replace": self.core_memory_replace
        }
    
    def archival_memory_insert(self, content: str) -> str:
        """将信息存入长期记忆"""
        self.vector_memory.add_experience(content)
        return f"Inserted into archival memory: {content[:50]}..."
    
    def archival_memory_search(self, query: str) -> str:
        """搜索长期记忆"""
        results = self.vector_memory.retrieve_relevant(query, k=3)
        return "
".join(results)
    
    def core_memory_append(self, content: str) -> str:
        """添加到核心工作记忆"""
        self.main_context.append(content)
        # 如果超出大小,触发归档
        if len(self.main_context) > self.max_context_size:
            archived = self.main_context.pop(0)
            self.archival_memory_insert(archived)
            return f"Archived old memory, added new: {content[:50]}..."
        return f"Added to core memory: {content[:50]}..."
    
    def core_memory_replace(self, old: str, new: str) -> str:
        """替换核心记忆内容"""
        try:
            idx = self.main_context.index(old)
            self.main_context[idx] = new
            return f"Replaced memory successfully"
        except ValueError:
            return "Old memory not found"
    
    def run(self, user_input: str) -> str:
        # 1. 搜索相关长期记忆
        relevant_memories = self.vector_memory.get_context_for_query(user_input, k=2)
        
        # 2. 构建 Prompt
        context = f"""Core memory:
{chr(10).join(self.main_context)}

{relevant_memories}

User: {user_input}

You can use the following tools to manage your memory:
- archival_memory_insert(content): Save information for long-term
- archival_memory_search(query): Search past experiences
- core_memory_append(content): Add to working memory
- core_memory_replace(old, new): Update working memory

Response:"""
        
        # 3. LLM 生成响应
        response = self.llm.invoke(context)
        
        # 4. 解析并执行工具调用 (简化版,实际需要更完善的解析)
        # ... tool execution logic ...
        
        return response
```

### 4. 实战:实现可持久化的 Memory

```python
import json
from pathlib import Path
from datetime import datetime

class PersistentMemory:
    def __init__(self, session_id: str, storage_dir="./memory_store"):
        self.session_id = session_id
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.short_term = []  # 短期记忆
        self.long_term = []   # 长期记忆
        
        self.load()
    
    def add_interaction(self, user_msg: str, assistant_msg: str):
        """添加交互记录"""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user": user_msg,
            "assistant": assistant_msg
        }
        
        self.short_term.append(interaction)
        
        # 保持短期记忆窗口大小
        if len(self.short_term) > 10:
            # 将最旧的移到长期记忆
            self.long_term.append(self.short_term.pop(0))
    
    def get_context(self, include_long_term=True) -> str:
        """获取记忆上下文"""
        context = "Recent conversation:
"
        for item in self.short_term:
            context += f"User: {item['user']}
Assistant: {item['assistant']}
"
        
        if include_long_term and self.long_term:
            context += "
Earlier in this session:
"
            for item in self.long_term[-3:]:  # 只取最近 3 条
                context += f"User: {item['user']}
Assistant: {item['assistant']}
"
        
        return context
    
    def save(self):
        """保存到磁盘"""
        data = {
            "session_id": self.session_id,
            "short_term": self.short_term,
            "long_term": self.long_term,
            "last_updated": datetime.now().isoformat()
        }
        
        file_path = self.storage_dir / f"{self.session_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self):
        """从磁盘加载"""
        file_path = self.storage_dir / f"{self.session_id}.json"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.short_term = data.get("short_term", [])
                self.long_term = data.get("long_term", [])
    
    def clear(self):
        """清除记忆"""
        self.short_term = []
        self.long_term = []
        self.save()
```

---

## 六、LangGraph:状态机编程范式

LangGraph 是 2024/2025 年构建 Agent 的核心框架,由 LangChain 团队开发。

### 1. StateGraph 核心概念

**核心思想**: Agent 工作流是一个**有向图** (Directed Graph),其中:
- **节点 (Node)**: 执行特定操作 (调用 LLM、执行工具、处理数据)
- **边 (Edge)**: 控制流转 (固定边、条件边)
- **状态 (State)**: 在节点间传递的数据

**架构图**:
```
START → Agent → should_continue? ─Yes→ Tools → Agent
                       │
                       No
                       ↓
                      END
```

### 2. 实战:基于 LangGraph 的 ReAct Agent

```python
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_openai import ChatOpenAI
from langchain.tools import tool

# 1. 定义状态
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "对话历史"]
    next_action: str  # "continue" or "end"

# 2. 定义工具
@tool
def calculator(expression: str) -> str:
    """计算数学表达式,例如: 3 * (2 + 5)"""
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {e}"

@tool
def get_weather(city: str) -> str:
    """获取城市天气 (模拟)"""
    return f"{city} 的天气: 晴天,温度 25°C"

tools = [calculator, get_weather]
tool_executor = ToolExecutor(tools)

# 3. 创建 LLM (支持 function calling)
llm = ChatOpenAI(model="gpt-4", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# 4. 定义节点函数
def call_agent(state: AgentState) -> dict:
    """Agent 决策节点"""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def execute_tools(state: AgentState) -> dict:
    """工具执行节点"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # 执行所有工具调用
    tool_results = []
    for tool_call in last_message.tool_calls:
        result = tool_executor.invoke(tool_call)
        tool_results.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            )
        )
    
    return {"messages": tool_results}

# 5. 定义条件边
def should_continue(state: AgentState) -> str:
    """判断是否继续"""
    last_message = state["messages"][-1]
    
    # 如果 LLM 调用了工具,则继续
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    else:
        return "end"

# 6. 构建图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("agent", call_agent)
workflow.add_node("tools", execute_tools)

# 添加边
workflow.set_entry_point("agent")

# 条件边: agent → continue → tools 或 agent → end
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

# 固定边: tools → agent (形成循环)
workflow.add_edge("tools", "agent")

# 7. 编译
app = workflow.compile()

# 8. 运行
initial_state = {
    "messages": [HumanMessage(content="北京天气如何?计算 (3 + 5) * 2 的结果")]
}

for output in app.stream(initial_state):
    for key, value in output.items():
        print(f"Output from node '{key}':")
        print(value)
        print("
---
")
```

### 3. 条件边与循环控制

#### (1) 复杂条件路由

```python
def route_query(state: AgentState) -> str:
    """根据查询类型路由到不同处理器"""
    last_message = state["messages"][-1].content
    
    if "天气" in last_message:
        return "weather_handler"
    elif any(op in last_message for op in ["+", "-", "*", "/"]):
        return "math_handler"
    else:
        return "general_handler"

workflow.add_conditional_edges(
    "classifier",
    route_query,
    {
        "weather_handler": "weather_node",
        "math_handler": "math_node",
        "general_handler": "general_node"
    }
)
```

#### (2) 循环控制与最大步数

```python
class AgentStateWithCounter(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "对话历史"]
    iteration: int

def should_continue_with_limit(state: AgentStateWithCounter) -> str:
    """带最大步数限制的循环控制"""
    MAX_ITERATIONS = 5
    
    if state["iteration"] >= MAX_ITERATIONS:
        return "max_iterations_reached"
    
    last_message = state["messages"][-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    else:
        return "end"

def increment_counter(state: AgentStateWithCounter) -> dict:
    """增加计数器"""
    return {"iteration": state["iteration"] + 1}

# 在工具节点中增加计数
workflow.add_node("tools", lambda state: {
    **execute_tools(state),
    **increment_counter(state)
})
```

---

## 七、多智能体协作 (Multi-Agent)

### 1. 为什么需要多 Agent?

**单 Agent 的局限**:
- 容易产生角色混乱 (既要写代码又要测试)
- 难以同时精通多个领域
- 缺乏"批判性思维" (自己不容易发现自己的错误)

**多 Agent 的优势**:
- **专业化**: 每个 Agent 专注一个角色
- **对抗性**: 不同 Agent 可以互相审查
- **并行处理**: 多个 Agent 同时工作

### 2. Supervisor 模式实现

Supervisor 模式: 一个主管 Agent 协调多个工作 Agent。

```python
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

# 1. 定义角色 Prompt
SUPERVISOR_PROMPT = """你是项目经理,负责协调团队完成任务。

团队成员:
- researcher: 负责信息搜索和研究
- coder: 负责编写代码
- reviewer: 负责代码审查

根据用户需求,决定下一步应该由谁来处理。
如果任务完成,回复 "FINISH"。

当前任务: {task}
工作历史: {history}

请输出 JSON 格式:
{{"next_worker": "researcher|coder|reviewer|FINISH", "instruction": "具体指令"}}
"""

RESEARCHER_PROMPT = "你是一名研究员,负责搜索信息和分析需求。"
CODER_PROMPT = "你是一名工程师,负责根据需求编写高质量代码。"
REVIEWER_PROMPT = "你是一名代码审查员,负责检查代码质量和正确性。"

# 2. 定义状态
class TeamState(TypedDict):
    task: str
    history: list[dict]
    next_worker: str
    current_result: str

# 3. 定义工作节点
def supervisor_node(state: TeamState):
    print("
>>> Supervisor is deciding...")
    
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    history_text = "
".join([f"{h['worker']}: {h['result']}" for h in state["history"]])
    
    prompt = SUPERVISOR_PROMPT.format(
        task=state["task"],
        history=history_text or "None"
    )
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # 解析 JSON (简化版)
    import json
    try:
        decision = json.loads(response.content)
        return {
            "next_worker": decision["next_worker"],
            "current_result": decision["instruction"]
        }
    except:
        return {"next_worker": "FINISH", "current_result": "Error parsing decision"}

def researcher_node(state: TeamState):
    print("
>>> Researcher is working...")
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    prompt = f"{RESEARCHER_PROMPT}

Task: {state['current_result']}"
    result = llm.invoke([HumanMessage(content=prompt)]).content
    
    return {
        "history": state["history"] + [{"worker": "researcher", "result": result}],
        "current_result": result
    }

def coder_node(state: TeamState):
    print("
>>> Coder is working...")
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    prompt = f"{CODER_PROMPT}

Task: {state['current_result']}
Context: {state['history']}"
    result = llm.invoke([HumanMessage(content=prompt)]).content
    
    return {
        "history": state["history"] + [{"worker": "coder", "result": result}],
        "current_result": result
    }

def reviewer_node(state: TeamState):
    print("
>>> Reviewer is working...")
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    prompt = f"{REVIEWER_PROMPT}

Code to review: {state['current_result']}"
    result = llm.invoke([HumanMessage(content=prompt)]).content
    
    return {
        "history": state["history"] + [{"worker": "reviewer", "result": result}],
        "current_result": result
    }

# 4. 构建图
workflow = StateGraph(TeamState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("coder", coder_node)
workflow.add_node("reviewer", reviewer_node)

# 设置入口
workflow.set_entry_point("supervisor")

# 条件边: supervisor 根据决策路由
def route_next(state: TeamState) -> str:
    next_worker = state["next_worker"]
    if next_worker == "FINISH":
        return "end"
    return next_worker

workflow.add_conditional_edges(
    "supervisor",
    route_next,
    {
        "researcher": "researcher",
        "coder": "coder",
        "reviewer": "reviewer",
        "end": END
    }
)

# 所有工作节点完成后回到 supervisor
for node in ["researcher", "coder", "reviewer"]:
    workflow.add_edge(node, "supervisor")

# 编译
app = workflow.compile()

# 运行
result = app.invoke({
    "task": "写一个 Python 函数计算斐波那契数列",
    "history": [],
    "next_worker": "",
    "current_result": ""
})

print("
=== Final Result ===")
print(result["current_result"])
```

### 3. Hierarchical Agent 架构

分层架构: 高层 Agent 负责战略,底层 Agent 负责执行。

```python
# 高层 Agent: 任务分解
class HighLevelAgent:
    def __init__(self, llm):
        self.llm = llm
    
    def decompose_task(self, task: str) -> list[dict]:
        """将复杂任务分解为子任务"""
        prompt = f"""请将以下任务分解为可执行的子任务:

Task: {task}

输出 JSON 格式:
{{
  "subtasks": [
    {{"id": 1, "description": "...", "assigned_to": "worker_type"}},
    {{"id": 2, "description": "...", "assigned_to": "worker_type"}}
  ]
}}
"""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        import json
        return json.loads(response.content)["subtasks"]

# 底层 Agent: 执行具体任务
class LowLevelAgent:
    def __init__(self, llm, role: str):
        self.llm = llm
        self.role = role
    
    def execute(self, subtask: dict) -> str:
        prompt = f"""You are a {self.role}.
Execute this subtask: {subtask['description']}

Output:"""
        return self.llm.invoke([HumanMessage(content=prompt)]).content

# 协调器
class HierarchicalSystem:
    def __init__(self, llm):
        self.high_level = HighLevelAgent(llm)
        self.workers = {
            "researcher": LowLevelAgent(llm, "researcher"),
            "coder": LowLevelAgent(llm, "coder"),
            "tester": LowLevelAgent(llm, "tester")
        }
    
    def run(self, task: str) -> dict:
        # 1. 高层分解任务
        subtasks = self.high_level.decompose_task(task)
        
        # 2. 分配并执行
        results = []
        for subtask in subtasks:
            worker = self.workers[subtask["assigned_to"]]
            result = worker.execute(subtask)
            results.append({"subtask": subtask, "result": result})
        
        return {"task": task, "subtasks": results}
```

### 4. MetaGPT 与 AutoGen 简介

#### (1) MetaGPT: SOP 驱动

MetaGPT 的核心是**标准化文档输出**:
- ProductManager 输出 PRD (Product Requirement Document)
- Architect 输出 System Design
- Engineer 输出 Code
- QA Engineer 输出 Test Report

每个 Agent 的输出都有严格的格式约束,减少误差累积。

#### (2) AutoGen: 对话编程

AutoGen 更灵活,核心是**会话** (Conversation):
- 定义多个 Agent,每个有自己的 System Prompt
- 将它们放入 GroupChat
- 让它们自己对话,直到达成共识或完成任务

```python
# AutoGen 示例 (简化)
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

llm_config = {"model": "gpt-4", "api_key": "..."}

# 定义 Agent
user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "coding"}
)

coder = AssistantAgent(
    name="Coder",
    system_message="你是一名工程师,负责编写代码。",
    llm_config=llm_config
)

reviewer = AssistantAgent(
    name="Reviewer",
    system_message="你是一名代码审查员,负责检查代码质量。",
    llm_config=llm_config
)

# 创建群聊
groupchat = GroupChat(
    agents=[user_proxy, coder, reviewer],
    messages=[],
    max_round=10
)

manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# 启动对话
user_proxy.initiate_chat(
    manager,
    message="请写一个快速排序算法的 Python 实现,并进行代码审查。"
)
```

---

## 八、Output Parser:结构化输出解析

### 1. Pydantic 模型定义

使用 Pydantic 强制 LLM 输出符合预期格式。

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional

class CodeReviewResult(BaseModel):
    """代码审查结果"""
    
    overall_score: int = Field(ge=0, le=100, description="总体评分 (0-100)")
    issues: List[str] = Field(description="发现的问题列表")
    suggestions: List[str] = Field(description="改进建议")
    approved: bool = Field(description="是否通过审查")
    
    @validator('overall_score')
    def score_determines_approval(cls, v, values):
        if v < 60 and values.get('approved', False):
            raise ValueError("Score < 60 but marked as approved")
        return v

class TaskPlan(BaseModel):
    """任务规划"""
    
    goal: str = Field(description="任务目标")
    steps: List[dict] = Field(description="执行步骤")
    estimated_time: Optional[int] = Field(default=None, description="预计耗时 (分钟)")
    dependencies: List[str] = Field(default_factory=list, description="依赖项")

# 使用
from langchain.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=CodeReviewResult)

prompt = f"""请审查以下代码:

```python
def add(a, b):
    return a + b
```

{parser.get_format_instructions()}
"""

response = llm.invoke(prompt)
result = parser.parse(response)

print(f"Score: {result.overall_score}")
print(f"Approved: {result.approved}")
```

### 2. 自修复解析器

当 LLM 输出格式错误时,自动尝试修复。

```python
from langchain.output_parsers import OutputFixingParser

base_parser = PydanticOutputParser(pydantic_object=CodeReviewResult)
fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)

# 即使 LLM 输出格式有误,也会尝试修复
try:
    result = fixing_parser.parse(llm_output)
except Exception as e:
    print(f"Even fixing failed: {e}")
```

---

## 九、本章小结

### 核心要点

1. **Workflow > Model**: 优秀的工作流可以让弱模型表现得像强模型
2. **ReAct vs Plan-and-Solve**: 
   - ReAct 适合简单任务,灵活但短视
   - Plan-and-Solve 适合复杂任务,需要全局规划
3. **Memory 是长期对话的关键**:
   - 短期记忆: Context Window
   - 长期记忆: Vector Database + MemGPT 虚拟内存
4. **MCP 是工具集成的未来**: 一次编写,处处运行
5. **LangGraph 是当前最佳 Agent 框架**: StateGraph 提供清晰的控制流
6. **Multi-Agent 是处理复杂任务的必经之路**: Supervisor 模式简单有效

### 技术栈总结

| 组件 | 推荐技术 | 适用场景 |
|------|----------|----------|
| Agent 框架 | LangGraph | 复杂工作流,需要精确控制 |
| 工具集成 | MCP Protocol | 统一工具接口 |
| Memory | Vector DB + MemGPT | 长期对话,知识管理 |
| Output Parsing | Pydantic + LangChain Parsers | 结构化输出 |
| Multi-Agent | Supervisor Pattern | 分工协作 |

### 下一步学习

- **Part 7 Chapter 3**: CoT 的数学本质与推理增强技术
- **Part 7 Chapter 4**: 推理模型 (o1/R1) 的训练与实现
- **Part 5 Chapter 1-3**: LangChain/LangGraph 生态深入

---

**下一章预告**: 第4章 - 多模态大模型原理

在下一章中,我们将给 Agent 装上"眼睛"和"耳朵",探讨 LLaVA、GPT-4V 背后的视觉编码原理,以及如何构建多模态 Agent。
