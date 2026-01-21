# 第五篇 RAG高级篇 (LlamaIndex)

> **目标**: 在掌握 LlamaIndex 基础组件（Index, Retriever, QueryEngine）的基础上，深入学习其"杀手锏"级的高级检索策略与 Agent 集成能力。本篇将带你从"能用"进化到"好用"。

## 📋 前置准备

本篇基于 LlamaIndex v0.10+ 版本，建议先完成第四篇的环境配置。

```bash
# 安装高级组件依赖
pip install llama-index-retrievers-bm25
pip install llama-index-postprocessor-cohere-rerank
pip install llama-index-graph-stores-neo4j
pip install llama-parse
```

---

## 第1章：混合检索 (Hybrid Retrieval)

单一的向量检索（Semantic Search）在处理精确关键词匹配（如产品型号、专有名词）时往往表现不佳。混合检索通过结合 **BM25（关键词）** 和 **Vector（语义）**，互补长短。

### 1.1 为什么需要混合检索？

*   **向量检索**：擅长理解"意图"和"概念"。例如搜"苹果手机"，能匹配到"iPhone"。
*   **关键词检索**：擅长精确匹配。例如搜"错误码 502"，向量可能会匹配到"网络错误"，但 BM25 能精确命中包含"502"的文档。

### 1.2 实战：构建混合检索器

LlamaIndex 提供了 `QueryFusionRetriever` 来优雅地融合多种检索结果。

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core import Settings

# 1. 准备数据与向量索引
documents = SimpleDirectoryReader("./data").load_data()
vector_index = VectorStoreIndex.from_documents(documents)

# 2. 创建 BM25 检索器 (基于关键词)
# 注意：BM25 需要 docstore 来构建倒排索引
bm25_retriever = BM25Retriever.from_defaults(
    docstore=vector_index.docstore,
    similarity_top_k=5
)

# 3. 创建 Vector 检索器 (基于语义)
vector_retriever = vector_index.as_retriever(similarity_top_k=5)

# 4. 创建融合检索器 (Hybrid)
hybrid_retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    num_queries=1,  # 不生成扩展查询，仅融合当前结果
    mode="reciprocal_rerank",  # 使用 RRF (倒数排名融合) 算法
    similarity_top_k=5,
    use_async=True
)

# 5. 测试检索
nodes = hybrid_retriever.retrieve("LlamaIndex 的自动合并检索原理是什么？")
for node in nodes:
    print(f"得分: {node.score:.4f} | 内容: {node.text[:50]}...")
```

---

## 第2章：查询优化与路由 (Routing & Transformation)

用户的 Query 往往是不完美的（模糊、复杂、缺失上下文）。LlamaIndex 提供了一系列工具来"修复"或"分发"用户的查询。

### 2.1 Router Query Engine (自动路由)

这是 LlamaIndex 最强大的 Pattern 之一：让 LLM 决定查哪个库。

*   **场景**：
    *   Query: "总结全文" -> 路由到 **SummaryIndex**
    *   Query: "作者是谁？" -> 路由到 **VectorStoreIndex**

```python
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core import SummaryIndex

# 1. 构建两个索引
summary_index = SummaryIndex.from_documents(documents)
vector_index = VectorStoreIndex.from_documents(documents)

# 2. 封装为工具
summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_index.as_query_engine(response_mode="tree_summarize"),
    description="用于回答关于文档整体摘要、大纲、总结类的问题"
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_index.as_query_engine(),
    description="用于回答关于文档中具体事实、细节、定义的精确问题"
)

# 3. 构建 Router (大脑)
router_query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[summary_tool, vector_tool],
    verbose=True
)

# 4. 测试自适应路由
response = router_query_engine.query("这篇文章主要讲了什么？")
print(f"Used Tool: {response.metadata['selector_result']}")
```

### 2.2 HyDE (假设性文档嵌入)

在检索前，让 LLM 先"幻觉"一个答案，用这个假设性答案去检索文档。

*   **原理**：查询 "如何优化索引？" -> LLM 生成一段关于索引优化的文本 -> 计算这段文本的向量 -> 检索最相似的真实文档。
*   **优势**：解决了 Query 和 Document 语义空间不一致的问题。

```python
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine

# 1. 定义 HyDE 变换
hyde = HyDEQueryTransform(include_original=True)

# 2. 包装原始查询引擎
vector_query_engine = vector_index.as_query_engine()
hyde_query_engine = TransformQueryEngine(vector_query_engine, query_transform=hyde)

# 3. 查询
response = hyde_query_engine.query("如何提高检索的准确率？")
```

---

## 第3章：重排序技术 (Reranking)

**"检索-精排"**是现代 RAG 的标准范式。
*   **Retriever**: 快速召回 Top-50（此时精度可能不高）。
*   **Reranker**: 使用高精度模型（如 Cross-Encoder）对 Top-50 进行重新打分，选出 Top-5。

### 3.1 Cohere Rerank 实战

Cohere 提供了目前业界公认效果最好的商业 Rerank 模型。

```python
from llama_index.postprocessor.cohere_rerank import CohereRerank

# 1.配置 API Key
# os.environ["COHERE_API_KEY"] = "your_key"

# 2. 定义 Reranker
cohere_rerank = CohereRerank(
    model="rerank-english-v3.0",
    top_n=3  # 最终只保留前3名
)

# 3. 注入到查询引擎 (作为 Postprocessor)
query_engine = vector_index.as_query_engine(
    similarity_top_k=20,  # 第一步：先宽泛召回20个
    node_postprocessors=[cohere_rerank]  # 第二步：精排选出3个
)

response = query_engine.query("LlamaIndex 和 LangChain 的区别？")

# 查看重排后的得分
for node in response.source_nodes:
    print(f"Rerank Score: {node.score:.4f} - {node.node.get_text()[:30]}...")
```

---

## 第4章：Chat Engine (多轮对话)

基础的 `query_engine` 是无状态的（一问一答）。而 `ChatEngine` 维护了对话历史（Memory），让 RAG 具备了聊天的能力。

### 4.1 Condense Plus Context Mode (最佳实践)

这是处理多轮 RAG 对话最稳健的模式：
1.  **Condense**: 将"当前问题" + "历史对话" 重写为一个独立的、完整的查询语句。
2.  **Retrieve**: 用重写后的查询去检索。
3.  **Context**: 将检索结果作为上下文，回答用户。

```python
from llama_index.core.memory import ChatMemoryBuffer

# 1. 设置记忆缓冲区 (Token 限制)
memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

# 2. 创建聊天引擎
chat_engine = vector_index.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=memory,
    similarity_top_k=3,
    verbose=True  # 开启后可以看到"重写后的查询"是什么
)

# 3. 多轮对话
response = chat_engine.chat("LlamaIndex 有哪些索引类型？")
print(response)

# 下一个问题隐含了上下文 ("它们")
response = chat_engine.chat("它们之间有什么区别？")
# 内部会重写为: "LlamaIndex 的不同索引类型之间有什么区别？"
print(response)
```

---

## 第5章：知识图谱 RAG (GraphRAG)

向量检索难以处理"长程关系"或"全局理解"。知识图谱（Knowledge Graph）通过显式的实体关系建模，弥补了这一短板。

### 5.1 PropertyGraphIndex 构建

LlamaIndex v0.10 推出的 `PropertyGraphIndex` 是目前最易用的 GraphRAG 实现，支持"图+向量"的双重检索。

```python
from llama_index.core import PropertyGraphIndex
from llama_index.core.indices.property_graph import (
    ImplicitPathExtractor,
    SimpleLLMPathExtractor
)

# 1. 定义提取器 (如何从文本变图)
# 自动提取实体和关系，如 (Steve Jobs)-[FOUNDED]->(Apple)
kg_extractor = SimpleLLMPathExtractor(
    llm=Settings.llm,
    max_paths_per_chunk=10,
    num_workers=4
)

# 2. 构建图索引 (同时生成 Embedding)
graph_index = PropertyGraphIndex.from_documents(
    documents,
    kg_extractors=[kg_extractor],
    embed_kg_nodes=True,  # 关键：开启向量化，支持混合检索
    show_progress=True
)

# 3. 混合检索查询 (Graph + Vector)
# 既能匹配关键词实体，通过图游走找到关系，也能通过向量匹配语义
query_engine = graph_index.as_query_engine(
    include_text=True,
    similarity_top_k=5
)

response = query_engine.query("Steve Jobs 和 Pixar 是什么关系？")
```

---

## 第6章：Agent 与 RAG 的终极融合

RAG 不应只是一个被动的查询接口，它应该成为 **Agent** 手中的一把利器（Tool）。

### 6.1 RAG as a Tool

通过 `FunctionCallingAgent`，我们让 LLM 自主决定何时查文档、查哪份文档，甚至进行多步推理。

```python
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner

# 1. 将 QueryEngine 包装为 Tool
rag_tool = QueryEngineTool(
    query_engine=vector_index.as_query_engine(),
    metadata=ToolMetadata(
        name="company_wiki",
        description="用于查询公司内部规章制度、报销流程、假期政策等。"
    )
)

# 2. 创建 Agent
agent_worker = FunctionCallingAgentWorker.from_tools(
    tools=[rag_tool],
    llm=Settings.llm,
    verbose=True
)
agent = AgentRunner(agent_worker)

# 3. 复杂任务 (ReAct 模式)
# 用户问："我下周想请年假，流程是什么？如果我还有5天年假，够吗？"
# Agent 会：
# 1. 调用 company_wiki 查"请假流程"
# 2. 这里演示的是单步，如果接了数据库工具，它还能去查"剩余年假"
# 3. 综合回答
response = agent.chat("请假流程怎么走？")
print(response)
```

---

## 第7章：总结与展望

### 7.1 LlamaIndex 核心能力大图

| 模块 | 核心组件 | 解决了什么问题？ |
| :--- | :--- | :--- |
| **Indexing** | VectorStoreIndex, SummaryIndex, PropertyGraphIndex | 如何高效、结构化地存储数据？ |
| **Retrieval** | Hybrid Retrieval, RouterRetriever | 如何从海量数据中精准捞出相关片段？ |
| **Reranking** | CohereRerank | 如何在召回结果中去粗取精，提升 Top-1 准确率？ |
| **Querying** | SubQuestion, MultiStep, HyDE | 如何处理复杂、模糊、多跳的用户问题？ |
| **Integration** | ChatEngine, AgentRunner | 如何将静态查询变为动态交互与智能体？ |

### 7.2 进阶学习路线

1.  **Fine-tuning Embeddings**: 如果你的领域非常垂直（如法律、医疗），通用 Embedding 模型效果不佳，试着使用 LlamaIndex 的 `SentenceTransformerFineTuning` 模块微调自己的 Embedding。
2.  **RAG Evaluation**: 不要凭感觉优化。引入 `Ragas` 或 `DeepEval`，建立由 `Faithfulness`（忠实度）和 `Answer Relevance`（相关度）构成的自动化测试集。
3.  **Local RAG**: 尝试使用 `Ollama` + `LlamaIndex`，构建完全运行在本地隐私环境下的 RAG 系统。

LlamaIndex 的强大之处在于其**极高的模块化**和**数据优先**的设计哲学。掌握了本篇的高级技巧，你已经具备了构建企业级 RAG 应用的核心能力。

---
