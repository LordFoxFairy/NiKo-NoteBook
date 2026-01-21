# 第3章：智能体（Agent）核心机制

> "The future of AI is not just about better models, but about better systems." - Andrew Ng
>
> 智能体（Agent）将 LLM 从"大脑"变成了"双手"，让 AI 具备了与世界交互的能力。

---

## 目录
- [一、从 Prompt Engineering 到 Agentic Workflow](#一从-prompt-engineering-到-agentic-workflow)
  - [1. 什么是 Agentic Workflow？](#1-什么是-agentic-workflow)
  - [2. 四种核心设计模式](#2-四种核心设计模式)
- [二、规划 (Planning)：不仅是 ReAct](#二规划-planning不仅是-react)
  - [1. ReAct 原理与局限](#1-react-原理与局限)
  - [2. Plan-and-Solve 与 P-Code](#2-plan-and-solve-与-p-code)
  - [3. Reflexion：通过反思自我进化](#3-reflexion通过反思自我进化)
- [三、工具使用 (Tool Use) 与 MCP](#三工具使用-tool-use-与-mcp)
  - [1. Function Calling 协议详解](#1-function-calling-协议详解)
  - [2. MCP (Model Context Protocol) 革命](#2-mcp-model-context-protocol-革命)
  - [3. 实战：实现一个 MCP Server](#3-实战实现一个-mcp-server)
- [四、多智能体协作 (Multi-Agent)](#四多智能体协作-multi-agent)
  - [1. 为什么一个和尚挑水吃，两个和尚抬水吃？](#1-为什么一个和尚挑水吃两个和尚抬水吃)
  - [2. MetaGPT 原理：SOP 的力量](#2-metagpt-原理sop-的力量)
  - [3. AutoGen 原理：对话即计算](#3-autogen-原理对话即计算)
  - [4. 代码实战：手写 Designer + Coder 双强协作](#4-代码实战手写-designer--coder-双强协作)
- [五、记忆系统 (Memory)](#五记忆系统-memory)
  - [1. Short-term vs Long-term](#1-short-term-vs-long-term)
  - [2. MemGPT 架构解析](#2-memgpt-架构解析)
- [六、实战：基于 LangGraph 构建全功能 Agent](#六实战基于-langgraph-构建全功能-agent)
- [七、本章小结](#七本章小结)

---

## 一、从 Prompt Engineering 到 Agentic Workflow

Andrew Ng 在 2024 年提出一个重要观点：**与其追求更强的模型 (GPT-5)，不如优化 Agent 工作流 (Agentic Workflow)**。
GPT-3.5 + 良好的工作流，往往能吊打零样本的 GPT-4。

### 1. 什么是 Agentic Workflow？

- **Zero-Shot**: 就像让一个人"一口气"写完一篇论文，不许查资料，不许修改。很难写好。
- **Agentic Workflow**: 就像让一个人先列提纲，再查资料，写初稿，自我修改，最后定稿。
  - **核心差异**：允许 LLM 进行**迭代 (Iterative)** 处理。

### 2. 四种核心设计模式

1.  **Reflection (反思)**：
    让模型检查自己的输出。
    > "这是我生成的代码。请检查是否有 Bug，如果有请修复。"

2.  **Tool Use (工具使用)**：
    让模型知道自己不知道，并懂得求助。
    > "我不记得今天的日期，但我可以调用 `get_date()` 函数。"

3.  **Planning (规划)**：
    遇到复杂问题，先拆解步骤。
    > "要解决这个问题，我需要先做A，再做B，最后做C。"

4.  **Multi-Agent Collaboration (多智能体协作)**：
    让不同角色的 Agent 像公司团队一样分工合作。
    > PM 负责提需求，Coder 负责写代码，QA 负责测试。

---

## 二、规划 (Planning)：不仅是 ReAct

### 1. ReAct 原理与局限

ReAct (Reason + Act) 是最经典的模式。
Prompt 模板：
```text
Question: ...
Thought: 我应该先搜索一下...
Action: Search[...]
Observation: 搜索结果是...
Thought: 根据结果，我需要计算...
Action: Calc[...]
...
```

**局限性**：
- **短视**：ReAct 往往只看下一步，缺乏全局观。
- **死循环**：容易在两个步骤之间反复横跳。
- **Token 消耗大**：每一步都要把整个 History 喂给模型。

### 2. Plan-and-Solve 与 P-Code

**Plan-and-Solve** 策略要求模型先生成完整的计划，再逐一执行。
为了让计划更精准，可以使用 **P-Code (Pseudo-Code)**。

**P-Code Prompt 示例**：
```text
Goal: 爬取某网站并分析数据。

Plan:
1. html = fetch_url(url)
2. data = parse_table(html)
3. analysis = analyze(data)
4. report = generate_report(analysis)
```

这种结构化的计划比自然语言更鲁棒，模型执行起来像解释器一样稳定。

### 3. Reflexion：通过反思自我进化

Reflexion (Shinn et al., 2023) 引入了一个**语言反馈 (Verbal Reinforcement)** 循环。

**工作流**：
1.  **Actor**: 尝试完成任务。
2.  **Evaluator**: 评分（成功/失败）。
3.  **Self-Reflection**: 如果失败了，让模型分析原因，并生成一段"经验教训"存入长期记忆。
4.  **Retry**: 下次尝试时，把"经验教训"放在 Context 里，避免重蹈覆辙。

**代码片段**：
```python
def reflexion_loop(task):
    memory = []
    for trial in range(3):
        # 尝试执行，带上之前的教训
        context = f"Task: {task}\nLessons: {memory}"
        result = actor.act(context)

        if evaluator.success(result):
            return result

        # 失败了，进行反思
        reflection = reflector.reflect(context, result)
        memory.append(reflection) # 例如: "主要原因是我忘记处理超时异常"
    return "Failed"
```

---

## 三、工具使用 (Tool Use) 与 MCP

### 1. Function Calling 协议详解

详细内容参考前文，核心是 JSON Schema 的定义与解析。

### 2. MCP (Model Context Protocol) 革命

2024年底，Anthropic 推出了 **MCP (Model Context Protocol)**。
这是一个开放标准，旨在统一 **AI 模型** 与 **数据源/工具** 之间的连接。

**以前的问题**：
你要给 ChatGPT 写一个 Google Drive 插件，给 Claude 写一个，给 DeepSeek 写一个... 累死人。

**MCP 的解法**：
类似于 USB 协议。
- **MCP Client**: Claude Desktop, Cursor, Any LLM App.
- **MCP Server**: Google Drive, Slack, Postgres, Local File System.

只要你写好了一个 MCP Server，任何支持 MCP 的 LLM 都可以直接插拔使用你的工具和数据！

### 3. 实战：实现一个 MCP Server

我们用 Python `mcp` 库写一个简单的 Server，提供文件读取能力。

```python
# pip install mcp
from mcp.server.fastmcp import FastMCP

# 创建 Server
mcp = FastMCP("MyFileSystem")

@mcp.tool()
def read_file(path: str) -> str:
    """Read a file from the local filesystem."""
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"

@mcp.tool()
def list_directory(path: str) -> str:
    """List files in a directory."""
    import os
    return "\n".join(os.listdir(path))

# 运行 Server
if __name__ == "__main__":
    mcp.run()
```

现在，任何支持 MCP 的客户端（如 Claude Desktop）连接这个 Server 后，Claude 瞬间就拥有了读取你本地文件的能力，而不需要你把文件内容复制粘贴给它！

---

## 四、多智能体协作 (Multi-Agent)

### 1. 为什么一个和尚挑水吃，两个和尚抬水吃？

单体 Agent 容易产生幻觉，且很难同时精通所有领域。
多 Agent 利用了 **"角色扮演" (Role Playing)** 的强大人格稳定性。
让一个 Agent 始终扮演 "苛刻的测试员"，比让一个 Agent 既写代码又自测要有效得多。

### 2. MetaGPT 原理：SOP 的力量

MetaGPT (Hong et al., 2023) 的核心洞察是：**把人类公司的 SOP (标准作业程序) 搬给 Agent**。

它不只是让 Agent 瞎聊，而是定义了严格的文档标准：
- **PRD (产品需求文档)**
- **Design Review (设计评审)**
- **API Spec (接口文档)**

当 Agent A 输出一份标准 PRD 后，Agent B 才会开始写代码。这种结构化约束大大降低了长链任务的误差累积。

### 3. AutoGen 原理：对话即计算

微软的 AutoGen 更加灵活，它认为 **Conversation Programming (会话编程)** 是核心。
你只需要定义 Agent 的属性（System Prompt），然后把它们扔到一个群里（Group Chat），让它们自己聊出结果。

### 4. 代码实战：手写 Designer + Coder 双强协作

我们用 LangGraph 模拟一个精简版的 MetaGPT 模式。

```python
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

# 1. 定义角色 Prompt
DESIGNER_PROMPT = """你是一名软件架构师。
根据用户需求，输出一份详细的技术设计文档。
内容包括：核心类设计、数据结构、算法逻辑。
不要写代码，只写设计。"""

CODER_PROMPT = """你是一名高级工程师。
根据架构师的设计文档，编写完整的 Python 代码。
只输出代码块，不要废话。"""

# 2. 定义状态
class DevelopmentState(TypedDict):
    requirements: str
    design_doc: str
    code: str

# 3. 定义节点
def designer_node(state):
    print(">>> Designer is working...")
    msg = [
        SystemMessage(content=DESIGNER_PROMPT),
        HumanMessage(content=state['requirements'])
    ]
    design = llm.invoke(msg).content
    print(f"Design generated: {len(design)} chars")
    return {"design_doc": design}

def coder_node(state):
    print(">>> Coder is working...")
    msg = [
        SystemMessage(content=CODER_PROMPT),
        HumanMessage(content=f"Requirements: {state['requirements']}\n\nDesign: {state['design_doc']}")
    ]
    code = llm.invoke(msg).content
    print("Code generated.")
    return {"code": code}

# 4. 构建图
workflow = StateGraph(DevelopmentState)
workflow.add_node("designer", designer_node)
workflow.add_node("coder", coder_node)

workflow.add_edge(START, "designer")
workflow.add_edge("designer", "coder")
workflow.add_edge("coder", END)

app = workflow.compile()

# 5. 运行
result = app.invoke({"requirements": "写一个贪吃蛇游戏，要有计分板"})
print(result['code'])
```

---

## 五、记忆系统 (Memory)

### 1. Short-term vs Long-term

- **Short-term**: 就是 Context Window。现在动辄 128k/1M，这部分越来越大。
- **Long-term**: 类似于人脑的海马体。
  - **Procedural Memory**: 存储工具的使用方法（微调权重）。
  - **Episodic Memory**: 存储过往经历（向量数据库）。
  - **Semantic Memory**: 存储世界知识（知识图谱）。

### 2. MemGPT 架构解析

MemGPT (Packer et al., 2023) 提出了一种像操作系统管理内存一样的架构。

- **Main Context**: 相当于 RAM，存放当前对话。
- **External Context**: 相当于 Disk，存放历史数据。
- **Virtual Context Management**:
  Agent 可以自主调用 `archival_memory_insert` 和 `archival_memory_search` 函数，将不常用的信息从 RAM 换出到 Disk，或者从 Disk 换入 RAM。

这让 Agent 感觉自己拥有无限的记忆，且永远不会遗忘重要信息。

---

## 六、实战：基于 LangGraph 构建全功能 Agent

(保留原有章节的优秀实战代码，并增加 Reflexion 机制)

... [此处保留并优化原有的 LangGraph 代码] ...

---

## 七、本章小结

1.  **Workflow > Model**：一个设计良好的工作流可以让 Llama-3-70B 表现得像 GPT-4。
2.  **MCP 是未来**：如果你要构建工具，请务必关注 MCP 协议。
3.  **多 Agent 是必经之路**：处理复杂任务时，请毫不犹豫地使用 Multi-Agent 架构，让专业的人做专业的事。

智能体不仅仅是"会说话的机器"，它们正在成为"会做事的数字员工"。

---

**下一章预告：** 第4章 - 多模态大模型原理

在下一章中，我们将给 Agent 装上眼睛和耳朵，探讨 LLaVA、GPT-4V 背后的视觉编码原理。
