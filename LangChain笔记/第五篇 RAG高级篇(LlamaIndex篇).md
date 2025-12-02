# 第五篇补充：RAG高级篇 (LlamaIndex)

> **版本要求**:
> - llama-index-core: 0.14.8
> - llama-index-llms-openai: 0.2.0+
> - llama-index-retrievers-bm25: 最新版
> - llama-index-postprocessor-cohere-rerank: 最新版
> - Python: 3.10+
> - 更新日期: 2025-11-29

## 概述

LlamaIndex 是专为 LLM 应用设计的数据框架，特别擅长构建高级 RAG（检索增强生成）系统。本篇将深入探讨 LlamaIndex 的高级检索技术、查询优化、重排序和知识图谱集成等核心功能。

**本篇核心内容**：
- 混合检索（BM25 + 向量检索）
- 查询优化与转换
- 高级 Query Engine（Router、SubQuestion、MultiStep）
- 重排序技术（Cohere Rerank、相似度过滤）
- Chat Engine 对话系统
- 知识图谱 RAG
- Agent 与 RAG 的结合

---

## 第 1 章：混合检索（Hybrid Retrieval）

混合检索结合了基于关键词的 BM25 检索和基于语义的向量检索，能够同时捕获精确匹配和语义相关性。

### 1.1 BM25 检索器

BM25（Best Matching 25）是一种改进的 TF-IDF 算法，通过词频饱和度和文档长度归一化提供更好的排序效果。

#### 基础用法

```python
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever
import Stemmer

# 1. 加载并解析文档
documents = SimpleDirectoryReader("./data").load_data()
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = splitter.get_nodes_from_documents(documents)

# 2. 创建 BM25 检索器
bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=5,
    stemmer=Stemmer.Stemmer("english"),  # 词干提取
    language="english"
)

# 3. 执行检索
query = "What is artificial intelligence?"
retrieved_nodes = bm25_retriever.retrieve(query)

for node in retrieved_nodes:
    print(f"Score: {node.score:.4f}")
    print(f"Text: {node.text[:200]}...\n")
```

#### 持久化与加载

```python
# 保存到磁盘
bm25_retriever.persist("./storage/bm25_retriever")

# 从磁盘加载
from llama_index.retrievers.bm25 import BM25Retriever
loaded_retriever = BM25Retriever.from_persist_dir("./storage/bm25_retriever")
```

#### 使用 Docstore

```python
from llama_index.core.storage.docstore import SimpleDocumentStore

# 创建 docstore 并添加节点
docstore = SimpleDocumentStore()
docstore.add_documents(nodes)

# 从 docstore 创建 BM25 检索器
bm25_retriever = BM25Retriever.from_defaults(
    docstore=docstore,
    similarity_top_k=5
)
```

### 1.2 向量检索器

使用向量嵌入进行语义检索：

```python
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# 配置 LLM 和 Embedding
Settings.llm = OpenAI(model="gpt-4", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# 创建向量索引
vector_index = VectorStoreIndex(nodes=nodes)

# 创建向量检索器
vector_retriever = vector_index.as_retriever(similarity_top_k=5)

# 执行检索
retrieved_nodes = vector_retriever.retrieve(query)
```

### 1.3 混合检索：QueryFusionRetriever

结合 BM25 和向量检索的优势：

```python
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# 1. 设置 Chroma 向量存储
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("hybrid_retrieval")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# 2. 创建存储上下文
from llama_index.core import StorageContext
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 3. 创建向量索引
vector_index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context
)

# 4. 创建混合检索器
hybrid_retriever = QueryFusionRetriever(
    retrievers=[
        vector_index.as_retriever(similarity_top_k=5),  # 向量检索
        BM25Retriever.from_defaults(
            docstore=vector_index.docstore,
            similarity_top_k=5
        )  # BM25 检索
    ],
    num_queries=1,  # 不生成额外查询变体
    use_async=True,  # 异步并行检索
    similarity_top_k=10  # 融合后返回的节点数
)

# 5. 执行混合检索
retrieved_nodes = hybrid_retriever.retrieve(query)

print(f"Retrieved {len(retrieved_nodes)} nodes from hybrid search")
```

### 1.4 元数据过滤

在检索时应用元数据过滤条件：

```python
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    MetadataFilter,
    FilterOperator
)

# 定义过滤条件
filters = MetadataFilters(
    filters=[
        MetadataFilter(
            key="category",
            value="technology",
            operator=FilterOperator.EQ
        ),
        MetadataFilter(
            key="publish_date",
            value="2024-01-01",
            operator=FilterOperator.GTE
        )
    ]
)

# 应用过滤器
filtered_retriever = BM25Retriever.from_defaults(
    docstore=docstore,
    similarity_top_k=5,
    filters=filters
)

# 或在向量检索中使用
vector_retriever = vector_index.as_retriever(
    similarity_top_k=5,
    filters=filters
)
```

### 1.5 使用 Query Engine

将混合检索器集成到查询引擎：

```python
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer

# 创建响应合成器
response_synthesizer = get_response_synthesizer(
    response_mode="compact",  # 紧凑模式
    use_async=True
)

# 创建查询引擎
query_engine = RetrieverQueryEngine(
    retriever=hybrid_retriever,
    response_synthesizer=response_synthesizer
)

# 执行查询
response = query_engine.query(
    "Explain the key concepts of machine learning"
)
print(response)

# 查看源节点
for node in response.source_nodes:
    print(f"\nSource: {node.metadata.get('file_name', 'Unknown')}")
    print(f"Score: {node.score:.4f}")
    print(f"Text: {node.text[:150]}...")
```

---

## 第 2 章：查询优化（Query Optimization）

查询优化通过转换、扩展或重写用户查询来提高检索质量。

### 2.1 查询转换

#### HyDE（Hypothetical Document Embeddings）

HyDE 生成假设性答案文档，然后用该文档的嵌入进行检索：

```python
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine

# 1. 创建基础查询引擎
base_query_engine = vector_index.as_query_engine(similarity_top_k=5)

# 2. 创建 HyDE 转换器
hyde_transform = HyDEQueryTransform(include_original=True)

# 3. 包装查询引擎
hyde_query_engine = TransformQueryEngine(
    base_query_engine,
    query_transform=hyde_transform
)

# 4. 执行查询（会先生成假设性文档）
response = hyde_query_engine.query(
    "What are the benefits of using LlamaIndex?"
)
print(response)
```

**工作原理**：
1. 用户查询 → LLM 生成假设性答案
2. 使用假设性答案的嵌入进行检索
3. 检索到的真实文档用于生成最终答案

#### 多查询生成

生成查询的多个变体以提高召回率：

```python
from llama_index.core.retrievers import QueryFusionRetriever

# QueryFusionRetriever 可以自动生成查询变体
fusion_retriever = QueryFusionRetriever(
    retrievers=[vector_retriever],
    num_queries=4,  # 生成 4 个查询变体
    use_async=True,
    mode="reciprocal_rerank"  # 使用倒数排名融合
)

retrieved_nodes = fusion_retriever.retrieve(
    "How does neural network training work?"
)
```

### 2.2 查询分解（Query Decomposition）

对于复杂查询，分解成多个子查询：

```python
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata

# 1. 创建专门的查询引擎
ml_query_engine = ml_index.as_query_engine(similarity_top_k=3)
dl_query_engine = dl_index.as_query_engine(similarity_top_k=3)
nlp_query_engine = nlp_index.as_query_engine(similarity_top_k=3)

# 2. 定义查询工具
query_engine_tools = [
    QueryEngineTool(
        query_engine=ml_query_engine,
        metadata=ToolMetadata(
            name="machine_learning",
            description="Provides information about traditional machine learning algorithms and techniques"
        )
    ),
    QueryEngineTool(
        query_engine=dl_query_engine,
        metadata=ToolMetadata(
            name="deep_learning",
            description="Provides information about deep learning, neural networks, and modern AI architectures"
        )
    ),
    QueryEngineTool(
        query_engine=nlp_query_engine,
        metadata=ToolMetadata(
            name="natural_language_processing",
            description="Provides information about NLP techniques, language models, and text processing"
        )
    )
]

# 3. 创建 SubQuestionQueryEngine
from llama_index.core.query_engine import SubQuestionQueryEngine

sub_question_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    use_async=True,
    verbose=True  # 显示子查询生成过程
)

# 4. 执行复杂查询
response = sub_question_engine.query(
    "Compare traditional machine learning with deep learning approaches "
    "and explain how they are used in natural language processing"
)

print(response)

# 查看生成的子查询
for sub_q in response.metadata.get("sub_questions", []):
    print(f"\nSub-question: {sub_q.sub_question}")
    print(f"Tool used: {sub_q.tool_name}")
```

### 2.3 查询重写

使用 LLM 改进查询表达：

```python
from llama_index.core.indices.query.query_transform import DecomposeQueryTransform

# 创建查询分解转换器
decompose_transform = DecomposeQueryTransform(
    llm=Settings.llm,
    verbose=True
)

# 应用到查询引擎
decompose_query_engine = TransformQueryEngine(
    base_query_engine,
    query_transform=decompose_transform
)

response = decompose_query_engine.query(
    "Tell me about AI and its applications"
)
```

---

## 第 3 章：高级 Query Engine

### 3.1 RouterQueryEngine

根据查询内容将请求路由到最合适的查询引擎。

```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool

# 1. 创建多个专门的查询引擎
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize"
)

vector_query_engine = vector_index.as_query_engine(
    similarity_top_k=5
)

# 2. 定义查询工具
query_engine_tools = [
    QueryEngineTool(
        query_engine=summary_query_engine,
        metadata=ToolMetadata(
            name="summary_tool",
            description=(
                "Use this tool for questions that require summarization "
                "or high-level overview of documents"
            )
        )
    ),
    QueryEngineTool(
        query_engine=vector_query_engine,
        metadata=ToolMetadata(
            name="vector_tool",
            description=(
                "Use this tool for specific factual questions "
                "that require precise information retrieval"
            )
        )
    )
]

# 3. 创建路由器查询引擎
router_query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=query_engine_tools,
    verbose=True
)

# 4. 执行查询（会自动选择合适的工具）
# 这个查询会路由到 summary_tool
response1 = router_query_engine.query(
    "Give me an overview of all the documents"
)

# 这个查询会路由到 vector_tool
response2 = router_query_engine.query(
    "What is the exact definition of transformer architecture?"
)

print(f"Query 1 used: {response1.metadata.get('selector_result')}")
print(f"Query 2 used: {response2.metadata.get('selector_result')}")
```

### 3.2 SubQuestionQueryEngine

将复杂问题分解为多个子问题并分别回答（见 2.2 节）。

**高级配置**：

```python
from llama_index.core.question_gen import LLMQuestionGenerator
from llama_index.core.question_gen.prompts import (
    DEFAULT_SUB_QUESTION_PROMPT_TMPL
)

# 自定义子问题生成器
custom_question_gen = LLMQuestionGenerator.from_defaults(
    llm=Settings.llm,
    prompt_template_str="""
Given a user question, generate {num_questions} related sub-questions
that help answer the original question comprehensively.

Original Question: {question}

Sub-questions:
"""
)

sub_question_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    question_gen=custom_question_gen,
    use_async=True
)
```

### 3.3 MultiStepQueryEngine

执行多步骤推理查询：

```python
from llama_index.core.query_engine import MultiStepQueryEngine
from llama_index.core.indices.query.query_transform import (
    StepDecomposeQueryTransform
)

# 1. 创建步骤分解转换器
step_decompose_transform = StepDecomposeQueryTransform(
    llm=Settings.llm,
    verbose=True
)

# 2. 创建多步骤查询引擎
multi_step_engine = MultiStepQueryEngine(
    query_engine=base_query_engine,
    query_transform=step_decompose_transform,
    num_steps=3,  # 最多 3 个推理步骤
    index_summary="This index contains technical documentation about AI"
)

# 3. 执行需要多步推理的查询
response = multi_step_engine.query(
    "First explain what transformers are, then describe how they are used "
    "in modern language models, and finally compare them with RNNs"
)

print(response)
```

### 3.4 RetrieverQueryEngine

使用自定义检索器的查询引擎（见 1.5 节）。

**高级响应合成模式**：

```python
from llama_index.core.response_synthesizers import ResponseMode

# compact: 合并文本块直到达到 token 限制
query_engine = RetrieverQueryEngine.from_args(
    retriever=hybrid_retriever,
    response_mode=ResponseMode.COMPACT
)

# refine: 逐个处理文本块，不断精炼答案
query_engine = RetrieverQueryEngine.from_args(
    retriever=hybrid_retriever,
    response_mode=ResponseMode.REFINE
)

# tree_summarize: 使用树形结构汇总
query_engine = RetrieverQueryEngine.from_args(
    retriever=hybrid_retriever,
    response_mode=ResponseMode.TREE_SUMMARIZE
)

# simple_summarize: 截断所有文本块并一次性发送
query_engine = RetrieverQueryEngine.from_args(
    retriever=hybrid_retriever,
    response_mode=ResponseMode.SIMPLE_SUMMARIZE
)
```

---

## 第 4 章：重排序技术（Reranking）

重排序在检索后对候选文档重新评分，提高最终结果的相关性。

### 4.1 相似度后处理器

基于相似度阈值过滤结果：

```python
from llama_index.core.postprocessor import SimilarityPostprocessor

# 创建相似度后处理器
similarity_processor = SimilarityPostprocessor(
    similarity_cutoff=0.75  # 只保留相似度 >= 0.75 的节点
)

# 应用到查询引擎
query_engine = vector_index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[similarity_processor]
)

response = query_engine.query("What is machine learning?")

# 查看过滤后的节点
print(f"Returned {len(response.source_nodes)} nodes (filtered from 10)")
for node in response.source_nodes:
    print(f"Score: {node.score:.4f}")
```

### 4.2 Cohere Rerank

使用 Cohere 的专业重排序模型：

```python
from llama_index.postprocessor.cohere_rerank import CohereRerank

# 1. 创建 Cohere Rerank 后处理器
cohere_rerank = CohereRerank(
    api_key="your-cohere-api-key",
    top_n=3,  # 重排后返回前 3 个结果
    model="rerank-english-v3.0"
)

# 2. 应用到查询引擎
query_engine = vector_index.as_query_engine(
    similarity_top_k=10,  # 先检索 10 个候选
    node_postprocessors=[cohere_rerank]  # 重排后返回 top 3
)

# 3. 执行查询
response = query_engine.query(
    "What are the latest developments in large language models?"
)

print(response)

# 查看重排序后的分数
for idx, node in enumerate(response.source_nodes, 1):
    print(f"\n{idx}. Rerank Score: {node.score:.4f}")
    print(f"Text: {node.text[:200]}...")
```

**Cohere Rerank 优势**：
- 专门训练的跨语言重排序模型
- 支持 100+ 种语言
- 比简单的相似度计算更准确
- 考虑查询和文档的深层语义关系

### 4.3 SentenceTransformer Rerank

使用 Sentence Transformers 进行重排序：

```python
from llama_index.core.postprocessor import SentenceTransformerRerank

# 创建 SentenceTransformer Rerank 后处理器
sentence_rerank = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",  # 跨编码器模型
    top_n=5
)

query_engine = vector_index.as_query_engine(
    similarity_top_k=15,
    node_postprocessors=[sentence_rerank]
)

response = query_engine.query("Explain transformer architecture")
```

**跨编码器 vs 双编码器**：
- **双编码器**（用于初始检索）：分别编码查询和文档，适合大规模检索
- **跨编码器**（用于重排序）：联合编码查询和文档，更准确但计算成本高

### 4.4 组合多个后处理器

```python
from llama_index.core.postprocessor import (
    SimilarityPostprocessor,
    KeywordNodePostprocessor
)

# 1. 关键词过滤
keyword_processor = KeywordNodePostprocessor(
    required_keywords=["AI", "machine learning"],
    exclude_keywords=["deprecated"]
)

# 2. 相似度过滤
similarity_processor = SimilarityPostprocessor(similarity_cutoff=0.7)

# 3. Cohere 重排序
cohere_rerank = CohereRerank(api_key="your-key", top_n=3)

# 4. 链式应用（按顺序执行）
query_engine = vector_index.as_query_engine(
    similarity_top_k=20,
    node_postprocessors=[
        keyword_processor,      # 先过滤关键词
        similarity_processor,   # 再过滤相似度
        cohere_rerank          # 最后重排序
    ]
)
```

### 4.5 时间加权后处理器

根据文档的时间戳调整相关性分数：

```python
from llama_index.core.postprocessor import TimeWeightedPostprocessor

# 创建时间加权后处理器
time_processor = TimeWeightedPostprocessor(
    time_decay=0.5,  # 时间衰减系数
    time_access_refresh=False,  # 是否刷新访问时间
    top_k=5
)

query_engine = vector_index.as_query_engine(
    node_postprocessors=[time_processor]
)

# 适用于需要最新信息的查询
response = query_engine.query("What are the latest AI trends?")
```

---

## 第 5 章：Chat Engine（对话系统）

Chat Engine 提供有状态的对话接口，维护历史上下文。

### 5.1 SimpleChatEngine

基础对话引擎，不进行检索增强：

```python
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.memory import ChatMemoryBuffer

# 1. 创建聊天记忆
chat_memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

# 2. 创建简单聊天引擎
chat_engine = SimpleChatEngine.from_defaults(
    llm=Settings.llm,
    memory=chat_memory
)

# 3. 进行对话
response1 = chat_engine.chat("Hello! I want to learn about AI.")
print(response1)

response2 = chat_engine.chat("What are neural networks?")
print(response2)

response3 = chat_engine.chat("Can you summarize what we discussed?")
print(response3)

# 4. 流式响应
streaming_response = chat_engine.stream_chat("Tell me more about deep learning")
for token in streaming_response.response_gen:
    print(token, end="", flush=True)
```

### 5.2 CondensePlusContextChatEngine

压缩对话历史并检索相关上下文：

```python
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.storage.chat_store import SimpleChatStore

# 1. 创建聊天存储
chat_store = SimpleChatStore()

# 2. 创建聊天记忆
chat_memory = ChatMemoryBuffer.from_defaults(
    token_limit=3000,
    chat_store=chat_store,
    chat_store_key="user_123"
)

# 3. 创建 CondensePlusContext 引擎
chat_engine = vector_index.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=chat_memory,
    similarity_top_k=5,
    verbose=True
)

# 4. 进行 RAG 对话
response1 = chat_engine.chat(
    "What is the main topic of the documents?"
)
print(response1)

response2 = chat_engine.chat(
    "Can you give me more details about that?"  # 引用前面的对话
)
print(response2)

# 5. 查看检索到的源
for node in response2.source_nodes:
    print(f"\nSource: {node.metadata.get('file_name')}")
    print(f"Text: {node.text[:150]}...")
```

**工作流程**：
1. 将对话历史压缩成独立的查询
2. 使用压缩后的查询检索相关文档
3. 将检索结果和对话历史一起发送给 LLM

### 5.3 ReActChatEngine

使用 ReAct（推理 + 行动）模式的智能体对话引擎：

```python
from llama_index.core.chat_engine import ReActChatEngine
from llama_index.core.tools import QueryEngineTool

# 1. 创建工具
query_tool = QueryEngineTool(
    query_engine=vector_index.as_query_engine(),
    metadata=ToolMetadata(
        name="knowledge_base",
        description="Search the knowledge base for information"
    )
)

# 2. 创建 ReAct 聊天引擎
react_chat_engine = vector_index.as_chat_engine(
    chat_mode="react",
    verbose=True,
    tools=[query_tool]
)

# 3. 执行需要推理的对话
response = react_chat_engine.chat(
    "First check what topics are covered in the knowledge base, "
    "then explain the most important one in detail"
)

print(response)
```

### 5.4 持久化聊天历史

```python
from llama_index.core.storage.chat_store import SimpleChatStore

# 1. 创建并保存聊天历史
chat_store = SimpleChatStore()

chat_memory = ChatMemoryBuffer.from_defaults(
    token_limit=3000,
    chat_store=chat_store,
    chat_store_key="session_001"
)

chat_engine = vector_index.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=chat_memory
)

# 进行对话...
chat_engine.chat("Hello!")

# 2. 保存到磁盘
chat_store.persist(persist_path="./storage/chat_history.json")

# 3. 从磁盘加载
loaded_chat_store = SimpleChatStore.from_persist_path(
    persist_path="./storage/chat_history.json"
)

loaded_memory = ChatMemoryBuffer.from_defaults(
    token_limit=3000,
    chat_store=loaded_chat_store,
    chat_store_key="session_001"
)

# 继续之前的对话
chat_engine = vector_index.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=loaded_memory
)

response = chat_engine.chat("What did we discuss earlier?")
```

### 5.5 多用户聊天管理

```python
from llama_index.core.storage.chat_store import SimpleChatStore

chat_store = SimpleChatStore()

def get_chat_engine_for_user(user_id: str):
    """为每个用户创建独立的聊天引擎"""
    memory = ChatMemoryBuffer.from_defaults(
        token_limit=3000,
        chat_store=chat_store,
        chat_store_key=user_id  # 使用 user_id 作为键
    )

    return vector_index.as_chat_engine(
        chat_mode="condense_plus_context",
        memory=memory
    )

# 用户 A 的对话
chat_engine_a = get_chat_engine_for_user("user_a")
response_a = chat_engine_a.chat("Tell me about AI")

# 用户 B 的对话（独立的上下文）
chat_engine_b = get_chat_engine_for_user("user_b")
response_b = chat_engine_b.chat("What is machine learning?")

# 保存所有用户的聊天历史
chat_store.persist("./storage/all_chats.json")
```

---

## 第 6 章：知识图谱 RAG

知识图谱将文档中的实体和关系显式建模，支持复杂的图查询和推理。

### 6.1 PropertyGraphIndex 基础

#### 6.1.1 基础构建

```python
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core import Settings

# 1. 加载文档
documents = SimpleDirectoryReader("./data").load_data()

# 2. 创建图存储
graph_store = SimplePropertyGraphStore()

# 3. 创建 PropertyGraphIndex (基础版本)
pg_index = PropertyGraphIndex.from_documents(
    documents,
    property_graph_store=graph_store,
    show_progress=True
)

# 4. 保存图
pg_index.storage_context.persist(persist_dir="./storage/property_graph")

# 5. 加载图
from llama_index.core import StorageContext, load_index_from_storage

storage_context = StorageContext.from_defaults(
    persist_dir="./storage/property_graph"
)
pg_index = load_index_from_storage(storage_context)
```

---

#### 6.1.2 关键参数: embed_kg_nodes

**`embed_kg_nodes`参数**是启用向量检索的关键配置,**强烈推荐设置为True**。

**完整配置示例**:

```python
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# 配置全局LLM和Embedding
Settings.llm = OpenAI(model="gpt-4", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# 创建PropertyGraphIndex (推荐配置)
pg_index = PropertyGraphIndex.from_documents(
    documents,
    property_graph_store=graph_store,
    embed_kg_nodes=True,  # 🔑 关键参数: 为图节点生成embedding
    show_progress=True
)
```

**embed_kg_nodes参数详解**:

| 参数值 | 行为 | 适用场景 | 优势 | 劣势 |
|--------|------|---------|------|------|
| `True` (推荐) | 为所有实体节点和关系生成embedding | 需要向量检索、语义搜索 | ✅ 启用VectorContextRetriever<br>✅ 支持相似度搜索<br>✅ 提升检索精度 | ❌ 增加构建时间<br>❌ 增加存储成本 |
| `False` (默认) | 仅为原始文本块生成embedding | 仅使用LLM生成Cypher查询 | ✅ 构建速度快<br>✅ 存储成本低 | ❌ 无法使用VectorContextRetriever<br>❌ 语义搜索受限 |

**影响的功能**:

```python
# embed_kg_nodes=True 时,可使用VectorContextRetriever
from llama_index.core.indices.property_graph import VectorContextRetriever

vector_retriever = VectorContextRetriever(
    pg_index.property_graph_store,
    embed_model=Settings.embed_model,
    similarity_top_k=5  # 向量检索需要embedding
)

# embed_kg_nodes=False 时,VectorContextRetriever将无法工作
# 只能使用LLMSynonymRetriever等非向量方法
```

**最佳实践**:

```python
# ✅ 推荐: 生产环境配置
pg_index = PropertyGraphIndex.from_documents(
    documents,
    property_graph_store=graph_store,
    embed_kg_nodes=True,        # 启用向量检索
    show_progress=True,
    vector_store=vector_store,  # 可选: 自定义向量存储
)

# ⚠️ 仅用于测试/调试
pg_index = PropertyGraphIndex.from_documents(
    documents,
    property_graph_store=graph_store,
    embed_kg_nodes=False,  # 快速构建,但功能受限
    show_progress=True
)
```

---

#### 6.1.3 图存储选项 (Graph Stores)

LlamaIndex支持多种图存储后端,根据场景选择:

**所有支持的Graph Stores**:

| Store | 类型 | Native Embedding | 持久化 | 适用场景 | 推荐度 |
|-------|------|-----------------|--------|---------|--------|
| **SimplePropertyGraphStore** | 内存 | ✅ | 磁盘文件 | 开发测试、小规模 | ⭐⭐⭐⭐ |
| **Neo4jPropertyGraphStore** | 服务器 | ❌ | 数据库 | 生产环境、大规模 | ⭐⭐⭐⭐⭐ |
| **NebulaPropertyGraphStore** | 分布式 | ❌ | 数据库 | 超大规模、分布式 | ⭐⭐⭐ |
| **TiDBPropertyGraphStore** | HTAP | ❌ | 数据库 | 混合负载(OLTP+OLAP) | ⭐⭐⭐ |
| **FalkorDBPropertyGraphStore** | Redis | ❌ | Redis | 低延迟、实时查询 | ⭐⭐ |

**使用示例**:

```python
# 1. SimplePropertyGraphStore (默认,推荐入门)
from llama_index.core.graph_stores import SimplePropertyGraphStore

graph_store = SimplePropertyGraphStore()

# 2. Neo4jPropertyGraphStore (推荐生产)
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

graph_store = Neo4jPropertyGraphStore(
    username="neo4j",
    password="your-password",
    url="bolt://localhost:7687",
    database="neo4j"
)

# 3. NebulaPropertyGraphStore (超大规模)
from llama_index.graph_stores.nebula import NebulaPropertyGraphStore

graph_store = NebulaPropertyGraphStore(
    space="my_graph",
    host="127.0.0.1",
    port=9669
)

# 4. TiDBPropertyGraphStore (混合负载)
from llama_index.graph_stores.tidb import TiDBPropertyGraphStore

graph_store = TiDBPropertyGraphStore(
    host="localhost",
    port=4000,
    user="root",
    password="password",
    database="graph_db"
)

# 5. FalkorDBPropertyGraphStore (低延迟)
from llama_index.graph_stores.falkordb import FalkorDBPropertyGraphStore

graph_store = FalkorDBPropertyGraphStore(
    host="localhost",
    port=6379,
    graph_name="my_graph"
)
```

**选型建议**:

```
开发/学习 → SimplePropertyGraphStore
    ↓
生产环境(中小规模) → Neo4jPropertyGraphStore
    ↓
生产环境(超大规模) → NebulaPropertyGraphStore
    ↓
混合负载(OLTP+OLAP) → TiDBPropertyGraphStore
    ↓
实时低延迟 → FalkorDBPropertyGraphStore
```

**功能对比**:

| 功能 | Simple | Neo4j | Nebula | TiDB | FalkorDB |
|------|--------|-------|--------|------|----------|
| Cypher查询 | ❌ | ✅ | ✅ | ⚠️ 部分 | ✅ |
| 向量存储 | ✅ | ❌ (需外置) | ❌ (需外置) | ❌ (需外置) | ❌ (需外置) |
| 分布式 | ❌ | ⚠️ 企业版 | ✅ | ✅ | ❌ |
| 性能 | 低 | 高 | 极高 | 高 | 高 |
| 易用性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

**安装依赖**:

```bash
# Neo4j
pip install llama-index-graph-stores-neo4j

# Nebula
pip install llama-index-graph-stores-nebula

# TiDB
pip install llama-index-graph-stores-tidb

# FalkorDB
pip install llama-index-graph-stores-falkordb
```

---

#### 6.1.4 索引增删改查操作 (CRUD)

PropertyGraphIndex支持动态更新和查询操作,适合生产环境中持续更新知识图谱的场景。

**基础CRUD操作**:

```python
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.core import Document

# 假设已创建索引
pg_index = PropertyGraphIndex.from_documents(documents, ...)

# ========== 1. Insert (插入) ==========

# 插入新文档
new_document = Document(text="新内容: Claude is an AI assistant created by Anthropic.")
pg_index.insert(new_document)

# 插入多个节点
from llama_index.core.schema import TextNode

new_nodes = [
    TextNode(text="Node 1 content", metadata={"source": "manual"}),
    TextNode(text="Node 2 content", metadata={"source": "manual"})
]
pg_index.insert_nodes(new_nodes)

# ========== 2. Get (查询) - 通过Graph Store ==========

# 获取特定实体
entities = pg_index.property_graph_store.get(
    ids=["entity_id_1", "entity_id_2"]
)

# 按属性查询
entities = pg_index.property_graph_store.get(
    properties={"label": "PERSON", "name": "Elon Musk"}
)

# 获取关系图谱 (depth指定跳数)
rel_map = pg_index.property_graph_store.get_rel_map(
    [entity_node],
    depth=2  # 2-hop关系
)

# 获取原始文本块
llama_nodes = pg_index.property_graph_store.get_llama_nodes(['chunk_id_1'])

# ========== 3. Update (更新) - 通过Upsert ==========

from llama_index.core.graph_stores import EntityNode, Relation

# 更新或插入实体 (如果ID存在则更新,否则插入)
updated_entity = EntityNode(
    name="Elon Musk",
    label="PERSON",
    properties={"title": "CEO", "company": "SpaceX"}  # 更新属性
)
pg_index.property_graph_store.upsert_nodes([updated_entity])

# 更新或插入关系
updated_relation = Relation(
    label="WORKS_AT",
    source_id="elon_musk_id",
    target_id="spacex_id",
    properties={"since": "2002"}  # 更新关系属性
)
pg_index.property_graph_store.upsert_relations([updated_relation])

# ========== 4. Delete (删除) ==========

# 按ID删除
pg_index.property_graph_store.delete(
    ids=["entity_id_to_delete"]
)

# 按属性删除
pg_index.property_graph_store.delete(
    properties={"source": "deprecated"}
)
```

**异步操作**:

```python
# 所有操作都有异步版本
entities = await pg_index.property_graph_store.aget(ids=[...])
await pg_index.property_graph_store.adelete(ids=[...])
# 其他异步方法: aupsert_nodes, aupsert_relations, aget_rel_map等
```

**批量更新示例**:

```python
from llama_index.core.graph_stores import EntityNode, Relation
from llama_index.core.schema import TextNode

# 批量插入实体
entities = [
    EntityNode(name="Person1", label="PERSON", properties={"age": 30}),
    EntityNode(name="Person2", label="PERSON", properties={"age": 25}),
    EntityNode(name="Company1", label="ORGANIZATION", properties={})
]
pg_index.property_graph_store.upsert_nodes(entities)

# 批量插入关系
relations = [
    Relation(label="WORKS_AT", source_id=entities[0].id, target_id=entities[2].id),
    Relation(label="WORKS_AT", source_id=entities[1].id, target_id=entities[2].id),
]
pg_index.property_graph_store.upsert_relations(relations)

# 关联到原始文本块
source_chunk = TextNode(id_="source_1", text="Person1 and Person2 work at Company1.")
pg_index.property_graph_store.upsert_llama_nodes([source_chunk])

# 创建文本块到实体的关系
source_relations = [
    Relation(label="MENTIONS", source_id=entities[0].id, target_id="source_1"),
    Relation(label="MENTIONS", source_id=entities[1].id, target_id="source_1"),
]
pg_index.property_graph_store.upsert_relations(source_relations)
```

**生产环境最佳实践**:

```python
from llama_index.core import Document

# 持续更新场景
def update_knowledge_graph(pg_index, new_documents):
    """持续更新知识图谱"""

    # 1. 插入新文档 (自动提取实体和关系)
    for doc in new_documents:
        pg_index.insert(doc)

    # 2. 定期持久化
    pg_index.storage_context.persist(persist_dir="./storage/property_graph")

    # 3. 验证插入结果
    total_nodes = len(pg_index.property_graph_store.get(properties={}))
    print(f"Total entities in graph: {total_nodes}")

# 数据清理场景
def cleanup_old_data(pg_index, cutoff_date):
    """删除过期数据"""

    # 查询需要删除的节点
    old_nodes = pg_index.property_graph_store.get(
        properties={"created_at": {"$lt": cutoff_date}}
    )

    # 批量删除
    old_ids = [node.id for node in old_nodes]
    pg_index.property_graph_store.delete(ids=old_ids)
```

**与from_existing结合使用**:

```python
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

# 1. 加载已存在的图
graph_store = Neo4jPropertyGraphStore(
    username="neo4j",
    password="password",
    url="bolt://localhost:7687"
)

pg_index = PropertyGraphIndex.from_existing(
    property_graph_store=graph_store,
    embed_kg_nodes=True
)

# 2. 增量更新
new_docs = [Document(text="Latest news...")]
for doc in new_docs:
    pg_index.insert(doc)

# 3. 查询更新后的图
query_engine = pg_index.as_query_engine()
response = query_engine.query("What are the latest updates?")
```

**CRUD操作总结**:

| 操作 | Index层方法 | Graph Store层方法 | 异步版本 |
|------|-----------|-----------------|---------|
| **Create/Insert** | `insert(doc)`<br>`insert_nodes(nodes)` | `upsert_nodes(entities)`<br>`upsert_relations(relations)` | ✅ 有 |
| **Read/Get** | - | `get(ids=...)`<br>`get(properties=...)`<br>`get_rel_map(...)` | ✅ 有 |
| **Update** | `insert(doc)` (重新提取) | `upsert_nodes(...)`<br>`upsert_relations(...)` | ✅ 有 |
| **Delete** | - | `delete(ids=...)`<br>`delete(properties=...)` | ✅ 有 |

**注意事项**:
- ✅ `insert()` 会自动调用配置的kg_extractors提取新实体和关系
- ✅ `upsert` 是幂等操作,相同ID会更新而非重复插入
- ✅ 删除实体时,相关关系也会被删除 (取决于Graph Store实现)
- ⚠️ Neo4j等外部数据库需要手动管理事务和连接

---

### 6.2 KG Extractors（知识抽取器）

#### 6.2.1 ImplicitPathExtractor (零成本提取器)

**ImplicitPathExtractor**是官方默认提取器之一,从节点的现有`relationships`属性推断关系,**无需调用LLM**。

**适用场景**:
- ✅ 文档已有元数据关系
- ✅ 需要零成本快速构建图谱
- ✅ 与其他提取器组合使用

**工作原理**:

```python
# 假设文档节点已有relationships属性
node.relationships = {
    RelatedNodeInfo(
        node_id="doc_2",
        metadata={"relationship": "FOLLOWS"}
    )
}

# ImplicitPathExtractor会自动提取:
# (doc_1) -[FOLLOWS]-> (doc_2)
```

**使用示例**:

```python
from llama_index.core.indices.property_graph import ImplicitPathExtractor

# 创建零成本提取器
implicit_extractor = ImplicitPathExtractor()

# 使用 (无需LLM配置)
pg_index = PropertyGraphIndex.from_documents(
    documents,
    kg_extractors=[implicit_extractor],  # 零成本提取
    show_progress=True
)
```

**与其他提取器组合**:

```python
from llama_index.core.indices.property_graph import (
    ImplicitPathExtractor,
    SimpleLLMPathExtractor
)

# 组合使用: 隐式关系 + LLM提取
pg_index = PropertyGraphIndex.from_documents(
    documents,
    kg_extractors=[
        ImplicitPathExtractor(),      # 提取已有关系 (免费)
        SimpleLLMPathExtractor(       # 提取语义关系 (付费)
            llm=Settings.llm,
            max_paths_per_chunk=10
        )
    ],
    show_progress=True
)
```

**成本对比**:

| 提取器 | LLM调用 | 成本 | 提取质量 | 推荐场景 |
|--------|---------|------|---------|---------|
| ImplicitPathExtractor | ❌ 否 | $0 | 中等 (依赖元数据) | 已有元数据、预算有限 |
| SimpleLLMPathExtractor | ✅ 是 | $$ | 高 | 通用场景 |
| SchemaLLMPathExtractor | ✅ 是 | $$ | 极高 | 需要结构化知识 |

---

#### 6.2.2 SimpleLLMPathExtractor

使用 LLM 提取简单的三元组（主体-关系-客体）：

```python
from llama_index.core.indices.property_graph import (
    SimpleLLMPathExtractor
)

# 创建简单路径提取器
simple_extractor = SimpleLLMPathExtractor(
    llm=Settings.llm,
    max_paths_per_chunk=10,  # 每个文本块最多提取 10 个路径
    num_workers=4  # 并行处理
)

# 使用提取器创建索引
pg_index = PropertyGraphIndex.from_documents(
    documents,
    kg_extractors=[simple_extractor],
    show_progress=True
)
```

---

#### 6.2.3 SchemaLLMPathExtractor

使用预定义的模式约束实体和关系类型：

```python
from llama_index.core.indices.property_graph import (
    SchemaLLMPathExtractor
)
from typing import Literal

# 定义实体和关系类型
entities = Literal["PERSON", "ORGANIZATION", "LOCATION", "TECHNOLOGY"]
relations = Literal["WORKS_AT", "LOCATED_IN", "DEVELOPED", "USES"]

# 定义允许的关系模式
validation_schema = {
    "PERSON": ["WORKS_AT", "LOCATED_IN"],
    "ORGANIZATION": ["LOCATED_IN", "DEVELOPED"],
    "TECHNOLOGY": ["DEVELOPED", "USES"]
}

# 创建模式提取器
schema_extractor = SchemaLLMPathExtractor(
    llm=Settings.llm,
    possible_entities=entities,
    possible_relations=relations,
    kg_validation_schema=validation_schema,
    strict=True  # 严格模式：拒绝不符合模式的三元组
)

pg_index = PropertyGraphIndex.from_documents(
    documents,
    kg_extractors=[schema_extractor],
    show_progress=True
)
```

---

#### 6.2.4 DynamicLLMPathExtractor (参数详解)

**DynamicLLMPathExtractor**动态提取实体和关系，支持**可选的类型约束**,比SchemaLLMPathExtractor更灵活。

**核心特点**:
- ✅ 允许实体/关系类型作为**提示**而非硬约束
- ✅ LLM可以提取超出allowed范围的类型
- ✅ 适合知识发现场景 (不确定所有实体类型)

**完整参数说明**:

| 参数 | 类型 | 默认值 | 说明 | 示例 |
|------|------|--------|------|------|
| **llm** | BaseLLM | 必填 | LLM实例 | `OpenAI(model="gpt-4")` |
| **max_triplets_per_chunk** | int | 10 | 每个文本块最多提取的三元组数 | `15` |
| **num_workers** | int | 4 | 并行处理worker数 | `8` |
| **allowed_entity_types** | List[str] | `None` | **可选提示**: 建议的实体类型 | `["PERSON", "ORG"]` |
| **allowed_relation_types** | List[str] | `None` | **可选提示**: 建议的关系类型 | `["WORKS_AT", "USES"]` |

**与SchemaLLMPathExtractor的区别**:

| 特性 | DynamicLLMPathExtractor | SchemaLLMPathExtractor |
|------|----------------------|----------------------|
| 类型约束 | **软约束** (提示) | **硬约束** (strict=True拒绝) |
| allowed参数 | 作为LLM提示 | 严格验证 |
| 超出范围类型 | ✅ 允许提取 | ❌ 拒绝 (strict=True) |
| 适用场景 | 知识发现、探索性分析 | 结构化知识库、严格schema |
| 灵活性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

**使用示例**:

```python
from llama_index.core.indices.property_graph import DynamicLLMPathExtractor

# 示例1: 提供类型提示 (推荐)
dynamic_extractor = DynamicLLMPathExtractor(
    llm=Settings.llm,
    max_triplets_per_chunk=15,
    num_workers=4,
    allowed_entity_types=["PERSON", "ORGANIZATION", "TECHNOLOGY", "LOCATION"],
    allowed_relation_types=["WORKS_AT", "FOUNDED", "USES", "LOCATED_IN"]
)

# LLM会优先提取这些类型,但也可能发现新类型
# 例如: 可能提取到 "PRODUCT" (未在allowed中) 如果文档提到产品

pg_index = PropertyGraphIndex.from_documents(
    documents,
    kg_extractors=[dynamic_extractor],
    show_progress=True
)

# 示例2: 无类型约束 (完全自由)
dynamic_extractor_free = DynamicLLMPathExtractor(
    llm=Settings.llm,
    max_triplets_per_chunk=20,
    # 不提供allowed参数,LLM自由提取
)

# 示例3: 调优max_triplets_per_chunk
# 文档密集 → 增加max_triplets
dynamic_extractor_dense = DynamicLLMPathExtractor(
    llm=Settings.llm,
    max_triplets_per_chunk=30,  # 提取更多关系
    allowed_entity_types=["PERSON", "ORG"]
)

# 文档稀疏 → 减少max_triplets (节省成本)
dynamic_extractor_sparse = DynamicLLMPathExtractor(
    llm=Settings.llm,
    max_triplets_per_chunk=5,
    allowed_entity_types=["PERSON"]
)
```

**参数调优建议**:

```python
# 🎯 通用场景 (平衡质量与成本)
DynamicLLMPathExtractor(
    llm=OpenAI(model="gpt-4", temperature=0),
    max_triplets_per_chunk=10,
    num_workers=4,
    allowed_entity_types=["PERSON", "ORGANIZATION", "TECHNOLOGY"],
    allowed_relation_types=["WORKS_AT", "USES", "FOUNDED"]
)

# 🚀 高质量场景 (追求完整性)
DynamicLLMPathExtractor(
    llm=OpenAI(model="gpt-4", temperature=0),
    max_triplets_per_chunk=20,  # 提取更多
    num_workers=8,              # 加速处理
    allowed_entity_types=["PERSON", "ORG", "TECH", "LOCATION", "EVENT"],
    allowed_relation_types=["WORKS_AT", "FOUNDED", "USES", "LOCATED_IN", "PARTICIPATES"]
)

# 💰 成本优化场景
DynamicLLMPathExtractor(
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0),  # 使用更便宜的模型
    max_triplets_per_chunk=5,   # 减少提取
    num_workers=2,
    allowed_entity_types=["PERSON", "ORG"]  # 聚焦核心类型
)
```

**实际效果对比**:

```python
# 输入文本:
# "Elon Musk founded SpaceX in 2002. The company developed Falcon 9 rocket."

# DynamicLLMPathExtractor (allowed_entity_types=["PERSON", "ORG"]):
# ✅ (Elon Musk, PERSON) -[FOUNDED]-> (SpaceX, ORG)
# ✅ (SpaceX, ORG) -[DEVELOPED]-> (Falcon 9, PRODUCT)  ← 发现新类型!
# ✅ (SpaceX, ORG) -[FOUNDED_IN]-> (2002, DATE)         ← 发现新类型!

# SchemaLLMPathExtractor (strict=True, possible_entities=["PERSON", "ORG"]):
# ✅ (Elon Musk, PERSON) -[FOUNDED]-> (SpaceX, ORG)
# ❌ 拒绝 (Falcon 9, PRODUCT) - 不在schema中
# ❌ 拒绝 (2002, DATE) - 不在schema中
```

**三种提取器选择指南**:

```
场景: 已知所有实体类型,需要严格结构化
     → SchemaLLMPathExtractor (strict=True)

场景: 大致知道实体类型,但希望发现新类型
     → DynamicLLMPathExtractor (推荐)

场景: 完全不确定有哪些实体类型 (探索性分析)
     → SimpleLLMPathExtractor 或 DynamicLLMPathExtractor (无allowed)

场景: 文档已有元数据关系
     → ImplicitPathExtractor
```

---

#### 6.2.5 自定义知识抽取器

LlamaIndex允许创建自定义知识抽取器,适合特定领域或特殊需求。

**核心要点**:
- ✅ 继承`TransformComponent`基类
- ✅ 使用`KG_NODES_KEY`和`KG_RELATIONS_KEY`存储实体和关系
- ✅ 保留现有的实体和关系 (与其他提取器组合使用)
- ✅ 与Ingestion Pipeline兼容

**完整实现示例**:

```python
from llama_index.core.graph_stores.types import (
    EntityNode,
    Relation,
    KG_NODES_KEY,
    KG_RELATIONS_KEY
)
from llama_index.core.schema import TransformComponent, BaseNode

class MyGraphExtractor(TransformComponent):
    """自定义知识抽取器示例"""

    def __call__(self, llama_nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        """
        处理节点并提取实体和关系

        Args:
            llama_nodes: 文本节点列表

        Returns:
            带有实体和关系元数据的节点列表
        """
        for llama_node in llama_nodes:
            # 1. 获取现有实体和关系 (保留其他提取器的结果)
            existing_nodes = llama_node.metadata.pop(KG_NODES_KEY, [])
            existing_relations = llama_node.metadata.pop(KG_RELATIONS_KEY, [])

            # 2. 自定义提取逻辑
            # 示例: 从文本中提取特定模式的实体
            text = llama_node.get_content()

            # 简单的规则提取 (实际应用中可能用NER模型或LLM)
            if "llama" in text.lower():
                llama_entity = EntityNode(
                    name="llama",
                    label="ANIMAL",
                    properties={"source": llama_node.node_id}
                )
                existing_nodes.append(llama_entity)

            if "index" in text.lower():
                index_entity = EntityNode(
                    name="index",
                    label="THING",
                    properties={"source": llama_node.node_id}
                )
                existing_nodes.append(index_entity)

                # 创建关系
                if any(n.name == "llama" for n in existing_nodes):
                    relation = Relation(
                        label="HAS",
                        source_id="llama",  # 或使用EntityNode.id
                        target_id="index",
                        properties={"confidence": 0.95}
                    )
                    existing_relations.append(relation)

            # 3. 将实体和关系存回元数据
            llama_node.metadata[KG_NODES_KEY] = existing_nodes
            llama_node.metadata[KG_RELATIONS_KEY] = existing_relations

        return llama_nodes

    # 可选: 异步版本
    # async def acall(self, llama_nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
    #     # 异步实现
    #     pass
```

**使用自定义提取器**:

```python
from llama_index.core.indices.property_graph import PropertyGraphIndex

# 创建自定义提取器实例
my_extractor = MyGraphExtractor()

# 与其他提取器组合使用
pg_index = PropertyGraphIndex.from_documents(
    documents,
    kg_extractors=[
        my_extractor,              # 自定义规则提取
        SimpleLLMPathExtractor()   # LLM通用提取
    ],
    show_progress=True
)
```

**高级示例: 基于NER模型的提取器**:

```python
from llama_index.core.graph_stores.types import EntityNode, Relation, KG_NODES_KEY, KG_RELATIONS_KEY
from llama_index.core.schema import TransformComponent, BaseNode

class NERGraphExtractor(TransformComponent):
    """基于NER模型的知识抽取器"""

    def __init__(self, ner_model=None):
        """
        Args:
            ner_model: NER模型 (例如spaCy, transformers)
        """
        self.ner_model = ner_model

    def __call__(self, llama_nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        for llama_node in llama_nodes:
            existing_nodes = llama_node.metadata.pop(KG_NODES_KEY, [])
            existing_relations = llama_node.metadata.pop(KG_RELATIONS_KEY, [])

            text = llama_node.get_content()

            # 使用NER模型提取实体
            if self.ner_model:
                entities = self.ner_model(text)  # 假设返回 [(text, label, start, end), ...]

                for ent_text, ent_label, start, end in entities:
                    entity = EntityNode(
                        name=ent_text,
                        label=ent_label,
                        properties={
                            "start": start,
                            "end": end,
                            "source": llama_node.node_id
                        }
                    )
                    existing_nodes.append(entity)

            llama_node.metadata[KG_NODES_KEY] = existing_nodes
            llama_node.metadata[KG_RELATIONS_KEY] = existing_relations

        return llama_nodes

# 使用示例
# import spacy
# nlp = spacy.load("en_core_web_sm")
# ner_extractor = NERGraphExtractor(ner_model=nlp)
#
# pg_index = PropertyGraphIndex.from_documents(
#     documents,
#     kg_extractors=[ner_extractor],
#     show_progress=True
# )
```

**与Ingestion Pipeline集成**:

```python
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter

# 创建包含自定义提取器的Pipeline
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=1024, chunk_overlap=20),
        MyGraphExtractor(),  # ✅ 自定义提取器与Pipeline兼容
        SimpleLLMPathExtractor(llm=Settings.llm)
    ]
)

# 运行Pipeline
nodes = pipeline.run(documents=documents)

# 使用处理后的节点创建索引
pg_index = PropertyGraphIndex(
    nodes=nodes,
    property_graph_store=graph_store
)
```

**最佳实践**:

1. **保留现有数据**: 使用`pop()`获取现有实体/关系,处理后重新存入
2. **幂等性**: 相同输入应产生相同输出
3. **错误处理**: 捕获异常避免Pipeline失败
4. **日志记录**: 记录提取的实体数量,便于调试

```python
import logging

class RobustGraphExtractor(TransformComponent):
    def __call__(self, llama_nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        logger = logging.getLogger(__name__)

        for llama_node in llama_nodes:
            try:
                existing_nodes = llama_node.metadata.pop(KG_NODES_KEY, [])
                existing_relations = llama_node.metadata.pop(KG_RELATIONS_KEY, [])

                # 自定义提取逻辑
                # ...

                logger.info(f"Extracted {len(existing_nodes)} entities from node {llama_node.node_id}")

                llama_node.metadata[KG_NODES_KEY] = existing_nodes
                llama_node.metadata[KG_RELATIONS_KEY] = existing_relations

            except Exception as e:
                logger.error(f"Failed to extract from node {llama_node.node_id}: {e}")
                # 返回原节点,不中断Pipeline
                continue

        return llama_nodes
```

---

### 6.3 图检索器

#### LLMSynonymRetriever

生成查询的同义词和相关关键词进行检索：

```python
from llama_index.core.indices.property_graph import (
    LLMSynonymRetriever
)

synonym_retriever = LLMSynonymRetriever(
    index=pg_index,
    llm=Settings.llm,
    similarity_top_k=5,
    include_text=True  # 包含节点文本
)

# 执行检索
retrieved_nodes = synonym_retriever.retrieve(
    "machine learning frameworks"
)

for node in retrieved_nodes:
    print(f"Entity: {node.metadata.get('entity_name')}")
    print(f"Type: {node.metadata.get('entity_type')}")
    print(f"Relations: {node.metadata.get('relationships', [])}\n")
```

#### VectorContextRetriever

使用语义向量检索图节点：

```python
from llama_index.core.indices.property_graph import (
    VectorContextRetriever
)

vector_context_retriever = VectorContextRetriever(
    index=pg_index,
    similarity_top_k=10,
    include_text=True,
    embed_model=Settings.embed_model
)

retrieved_nodes = vector_context_retriever.retrieve(
    "What technologies are used in AI development?"
)
```

#### TextToCypherRetriever

使用 LLM 生成 Cypher 查询（需要 Neo4j）：

```python
from llama_index.core.indices.property_graph import (
    TextToCypherRetriever
)

# 需要 Neo4j 图存储
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

neo4j_store = Neo4jPropertyGraphStore(
    username="neo4j",
    password="password",
    url="bolt://localhost:7687",
    database="neo4j"
)

text2cypher_retriever = TextToCypherRetriever(
    graph_store=neo4j_store,
    llm=Settings.llm
)

# LLM 会生成 Cypher 查询并执行
retrieved_nodes = text2cypher_retriever.retrieve(
    "Find all people who work at AI companies"
)
```

---

#### 6.3.5 自定义图检索器

LlamaIndex支持创建自定义图检索器,实现特定的检索逻辑。

**方法1: 使用CustomPGRetriever (推荐)**

```python
from llama_index.core.indices.property_graph import CustomPGRetriever

class MyCustomRetriever(CustomPGRetriever):
    """自定义图检索器"""

    def init(self, my_option_1: bool = False, **kwargs):
        """
        初始化自定义检索器

        Args:
            my_option_1: 自定义选项
            **kwargs: 父类参数 (会自动设置self.graph_store)
        """
        self.my_option_1 = my_option_1
        # self.graph_store 自动可用

    def custom_retrieve(self, query_str: str):
        """
        自定义检索逻辑

        Args:
            query_str: 查询字符串

        Returns:
            str, TextNode, NodeWithScore, 或以上类型的列表
        """
        # 访问图存储
        entities = self.graph_store.get(
            properties={"label": "PERSON"}
        )

        # 自定义检索逻辑
        if self.my_option_1:
            # 特殊处理逻辑
            pass

        # 返回结果
        return f"Found {len(entities)} entities matching '{query_str}'"

    # 可选: 异步版本
    # async def acustom_retrieve(self, query_str: str):
    #     # 异步检索逻辑
    #     pass
```

**使用自定义检索器**:

```python
# 创建自定义检索器
my_retriever = MyCustomRetriever(
    graph_store=pg_index.property_graph_store,
    my_option_1=True
)

# 检索
results = my_retriever.retrieve("Find all people")
print(results)
```

**完整示例: 基于关键词的图检索器**:

```python
from llama_index.core.indices.property_graph import CustomPGRetriever
from llama_index.core.schema import NodeWithScore, TextNode

class KeywordGraphRetriever(CustomPGRetriever):
    """基于关键词的图检索器"""

    def init(self, keywords: list[str] = None, top_k: int = 10, **kwargs):
        """
        Args:
            keywords: 关键词列表
            top_k: 返回top K个节点
        """
        self.keywords = keywords or []
        self.top_k = top_k

    def custom_retrieve(self, query_str: str):
        """基于关键词匹配检索实体"""
        # 提取查询中的关键词 (简化版,实际应该用NLP)
        query_keywords = set(query_str.lower().split())
        query_keywords.update(self.keywords)

        # 从图中获取所有实体
        all_entities = self.graph_store.get(properties={})

        # 计算匹配分数
        scored_entities = []
        for entity in all_entities:
            entity_name = getattr(entity, 'name', '').lower()
            entity_label = getattr(entity, 'label', '').lower()

            # 简单的关键词匹配评分
            score = 0.0
            for keyword in query_keywords:
                if keyword in entity_name:
                    score += 1.0
                if keyword in entity_label:
                    score += 0.5

            if score > 0:
                # 获取实体周边的上下文
                rel_map = self.graph_store.get_rel_map([entity], depth=1)

                # 创建TextNode包含实体信息
                text = f"Entity: {entity.name} (Type: {entity.label})\n"
                text += f"Properties: {entity.properties}\n"
                text += f"Relations: {rel_map}\n"

                node = TextNode(
                    text=text,
                    metadata={
                        "entity_id": entity.id,
                        "entity_name": entity.name,
                        "entity_label": entity.label
                    }
                )

                scored_entities.append(NodeWithScore(node=node, score=score))

        # 排序并返回Top K
        scored_entities.sort(key=lambda x: x.score, reverse=True)
        return scored_entities[:self.top_k]

# 使用示例
keyword_retriever = KeywordGraphRetriever(
    graph_store=pg_index.property_graph_store,
    keywords=["technology", "company", "person"],
    top_k=5
)

results = keyword_retriever.retrieve("Find AI companies")
for node_with_score in results:
    print(f"Score: {node_with_score.score}")
    print(f"Content: {node_with_score.node.get_content()}\n")
```

**高级示例: Cypher查询检索器**:

```python
from llama_index.core.indices.property_graph import CustomPGRetriever
from llama_index.core.schema import NodeWithScore, TextNode

class CypherQueryRetriever(CustomPGRetriever):
    """执行预定义Cypher查询的检索器"""

    def init(self, cypher_template: str, **kwargs):
        """
        Args:
            cypher_template: Cypher查询模板 (支持参数化)
        """
        self.cypher_template = cypher_template

    def custom_retrieve(self, query_str: str):
        """执行Cypher查询并返回结果"""
        # 检查Graph Store是否支持Cypher
        if not hasattr(self.graph_store, 'structured_query'):
            return "Graph store does not support Cypher queries"

        try:
            # 执行Cypher查询 (假设query_str包含参数)
            query = self.cypher_template.format(keyword=query_str)
            results = self.graph_store.structured_query(query)

            # 格式化结果
            nodes_with_scores = []
            for i, result in enumerate(results):
                text = str(result)
                node = TextNode(
                    text=text,
                    metadata={"result_index": i, "query": query}
                )
                nodes_with_scores.append(NodeWithScore(node=node, score=1.0))

            return nodes_with_scores

        except Exception as e:
            return f"Error executing Cypher query: {e}"

# 使用示例 (Neo4j)
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

neo4j_store = Neo4jPropertyGraphStore(
    username="neo4j",
    password="password",
    url="bolt://localhost:7687"
)

cypher_query = """
MATCH (p:PERSON)-[r:WORKS_AT]->(o:ORGANIZATION)
WHERE o.name CONTAINS '{keyword}'
RETURN p.name, o.name, r
LIMIT 10
"""

cypher_retriever = CypherQueryRetriever(
    graph_store=neo4j_store,
    cypher_template=cypher_query
)

results = cypher_retriever.retrieve("OpenAI")
```

**方法2: 继承BasePGRetriever (高级)**

```python
from llama_index.core.indices.property_graph.retrievers.base import BasePGRetriever
from llama_index.core.schema import QueryBundle, NodeWithScore

class AdvancedPGRetriever(BasePGRetriever):
    """高级自定义检索器 (继承BasePGRetriever)"""

    def __init__(self, graph_store, custom_param: str, **kwargs):
        super().__init__(**kwargs)
        self.graph_store = graph_store
        self.custom_param = custom_param

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """
        实现检索逻辑

        Args:
            query_bundle: 包含查询字符串和嵌入的Bundle

        Returns:
            NodeWithScore列表
        """
        query_str = query_bundle.query_str

        # 自定义检索逻辑
        # ...

        return []

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """异步检索"""
        # 异步实现
        pass
```

**组合多个自定义检索器**:

```python
from llama_index.core.retrievers import QueryFusionRetriever

# 创建多个自定义检索器
keyword_retriever = KeywordGraphRetriever(
    graph_store=pg_index.property_graph_store,
    keywords=["AI", "machine learning"]
)

custom_retriever = MyCustomRetriever(
    graph_store=pg_index.property_graph_store,
    my_option_1=True
)

# 使用Fusion组合
fusion_retriever = QueryFusionRetriever(
    retrievers=[keyword_retriever, custom_retriever],
    similarity_top_k=10,
    num_queries=1  # 不生成额外查询
)

# 联合检索
results = fusion_retriever.retrieve("Find AI technologies")
```

**最佳实践**:

1. **返回类型灵活**: 支持string, TextNode, NodeWithScore或它们的列表
2. **异步支持**: 实现`acustom_retrieve`提升性能
3. **错误处理**: 捕获异常避免检索失败
4. **评分机制**: 为NodeWithScore提供有意义的分数 (0.0-1.0)
5. **元数据丰富**: 在返回的Node中包含丰富的元数据,便于后处理

```python
class BestPracticeRetriever(CustomPGRetriever):
    def custom_retrieve(self, query_str: str):
        try:
            # 检索逻辑
            results = []

            # 确保返回NodeWithScore并包含分数
            for result in results:
                node = TextNode(
                    text=result['text'],
                    metadata={
                        "entity_id": result['id'],
                        "retriever": "BestPracticeRetriever",
                        "query": query_str,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                # 提供有意义的评分
                score = self._calculate_score(result, query_str)
                results.append(NodeWithScore(node=node, score=score))

            return results

        except Exception as e:
            # 日志记录错误
            logger.error(f"Retrieval failed for query '{query_str}': {e}")
            # 返回空结果而非抛出异常
            return []

    def _calculate_score(self, result, query_str):
        """计算相关性分数"""
        # 自定义评分逻辑
        return 1.0
```

---

### 6.4 知识图谱查询引擎

```python
# 1. 使用默认检索器
query_engine = pg_index.as_query_engine(
    include_text=True,
    similarity_top_k=5
)

response = query_engine.query(
    "What are the main technologies mentioned and how are they related?"
)
print(response)

# 2. 使用自定义检索器
from llama_index.core.query_engine import RetrieverQueryEngine

custom_retriever = LLMSynonymRetriever(
    index=pg_index,
    llm=Settings.llm,
    similarity_top_k=10
)

query_engine = RetrieverQueryEngine.from_args(
    retriever=custom_retriever,
    response_mode="tree_summarize"
)

response = query_engine.query("Explain the relationships between entities")
```

### 6.5 组合多个图检索器

```python
from llama_index.core.retrievers import QueryFusionRetriever

# 创建多个检索器
synonym_retriever = LLMSynonymRetriever(
    index=pg_index,
    similarity_top_k=5
)

vector_retriever = VectorContextRetriever(
    index=pg_index,
    similarity_top_k=5
)

# 融合检索结果
fusion_retriever = QueryFusionRetriever(
    retrievers=[synonym_retriever, vector_retriever],
    num_queries=1,
    use_async=True
)

query_engine = RetrieverQueryEngine.from_args(retriever=fusion_retriever)
response = query_engine.query("Complex knowledge graph query")
```

---

### 6.6 多跳检索与路径推理

**多跳检索（Multi-hop Retrieval）** 是知识图谱的核心能力之一,通过遍历多层关系来发现隐藏的知识关联,实现复杂的推理任务。

#### 6.6.1 多跳检索概念

**什么是多跳检索?**

```
单跳 (1-hop):  A → B
两跳 (2-hop):  A → B → C
三跳 (3-hop):  A → B → C → D
N跳 (N-hop):   A → ... → Z  (通过多层关系连接)
```

**应用场景**:
- **发现隐藏关联**: 两个看似无关的实体通过中间节点连接
- **知识推理链**: 构建因果链、继承链、影响链
- **社交网络分析**: 朋友的朋友、多度人脉
- **学术文献溯源**: 引用链、研究脉络追踪

#### 6.6.2 Neo4j Cypher多跳查询

LlamaIndex的Neo4j集成支持通过 `structured_query()` 执行原生Cypher查询:

```python
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

# 初始化Neo4j图存储
graph_store = Neo4jPropertyGraphStore(
    username="neo4j",
    password="password",
    url="bolt://localhost:7687",
    database="neo4j"
)

# 1. 固定跳数查询 (1-3跳)
def multi_hop_query(entity_name: str, max_hops: int = 3, limit: int = 10):
    """查询指定跳数范围内的所有路径"""
    cypher_query = f"""
    MATCH path = (start {{name: '{entity_name}'}})-[*1..{max_hops}]->(target)
    WHERE start <> target
    RETURN
        start.name as 起点,
        [node in nodes(path) | node.name] as 路径节点,
        [rel in relationships(path) | type(rel)] as 路径关系,
        length(path) as 跳数,
        target.name as 目标
    ORDER BY 跳数, target.name
    LIMIT {limit}
    """

    results = graph_store.structured_query(cypher_query)
    return results

# 使用示例
results = multi_hop_query("人工智能", max_hops=3, limit=15)

for r in results:
    path_length = r['跳数']
    start = r['起点']
    target = r['目标']
    path_nodes = r['路径节点']
    path_rels = r['路径关系']

    # 构建路径字符串
    path_str = path_nodes[0]
    for i in range(len(path_rels)):
        path_str += f" --[{path_rels[i]}]--> {path_nodes[i+1]}"

    print(f"[{path_length}跳] {path_str}")
```

**输出示例**:
```
[1跳] 人工智能 --[包含]--> 机器学习
[2跳] 人工智能 --[包含]--> 机器学习 --[包含]--> 深度学习
[3跳] 人工智能 --[包含]--> 机器学习 --[包含]--> 深度学习 --[使用]--> 神经网络
```

#### 6.6.3 最短路径查询

Cypher的 `shortestPath()` 函数找到两个节点之间的最短路径:

```python
def shortest_path_query(start_entity: str, end_entity: str, max_hops: int = 5):
    """查找两个实体之间的最短路径"""
    cypher_query = f"""
    MATCH path = shortestPath(
        (start {{name: '{start_entity}'}})-[*1..{max_hops}]-(end {{name: '{end_entity}'}})
    )
    RETURN
        start.name as 起点,
        end.name as 终点,
        [node in nodes(path) | node.name] as 完整路径,
        [rel in relationships(path) | type(rel)] as 路径关系,
        length(path) as 路径长度
    """

    results = graph_store.structured_query(cypher_query)

    if results:
        result = results[0]
        print(f"最短路径 (长度: {result['路径长度']}跳):")

        full_path = result['完整路径']
        rels = result['路径关系']

        path_str = full_path[0]
        for i in range(len(rels)):
            path_str += f" --[{rels[i]}]--> {full_path[i+1]}"
        print(path_str)
    else:
        print(f"未找到从 '{start_entity}' 到 '{end_entity}' 的路径")

    return results

# 使用示例
shortest_path_query("深度学习", "图遍历", max_hops=5)
```

**输出**:
```
最短路径 (长度: 4跳):
深度学习 --[使用]--> 神经网络 --[应用于]--> 图像识别 --[属于]--> 计算机视觉 --[需要]--> 图遍历
```

#### 6.6.4 单跳、双跳、三跳检索对比

```python
class MultiHopRetriever:
    """多跳检索器"""

    def __init__(self, graph_store: Neo4jPropertyGraphStore):
        self.graph_store = graph_store

    def single_hop(self, entity: str, top_k: int = 5):
        """1跳检索: 查找直接相关的节点"""
        cypher = f"""
        MATCH (start {{name: '{entity}'}})-[r]->(target)
        RETURN
            start.name as 起点,
            type(r) as 关系,
            target.name as 目标,
            target.category as 类别
        LIMIT {top_k}
        """
        return self.graph_store.structured_query(cypher)

    def two_hop(self, entity: str, top_k: int = 10):
        """2跳检索: 查找距离2步的节点"""
        cypher = f"""
        MATCH path = (start {{name: '{entity}'}})-[r1]->(mid)-[r2]->(target)
        WHERE start <> target
        RETURN
            start.name as 起点,
            type(r1) as 关系1,
            mid.name as 中间节点,
            type(r2) as 关系2,
            target.name as 目标
        LIMIT {top_k}
        """
        return self.graph_store.structured_query(cypher)

    def three_hop(self, entity: str, top_k: int = 10):
        """3跳检索: 查找距离3步的节点"""
        cypher = f"""
        MATCH path = (start {{name: '{entity}'}})-[r1]->(n1)-[r2]->(n2)-[r3]->(target)
        WHERE start <> target
        RETURN
            start.name as 起点,
            type(r1) as 关系1,
            n1.name as 节点1,
            type(r2) as 关系2,
            n2.name as 节点2,
            type(r3) as 关系3,
            target.name as 目标
        LIMIT {top_k}
        """
        return self.graph_store.structured_query(cypher)

# 使用示例
retriever = MultiHopRetriever(graph_store)

print("=== 1跳检索 ===")
results = retriever.single_hop("人工智能")
for r in results:
    print(f"{r['起点']} --[{r['关系']}]--> {r['目标']}")

print("\n=== 2跳检索 ===")
results = retriever.two_hop("人工智能")
for r in results:
    print(f"{r['起点']} --[{r['关系1']}]--> {r['中间节点']} --[{r['关系2']}]--> {r['目标']}")

print("\n=== 3跳检索 ===")
results = retriever.three_hop("人工智能")
for r in results:
    print(f"{r['起点']} --[{r['关系1']}]--> {r['节点1']} --[{r['关系2']}]--> {r['节点2']} --[{r['关系3']}]--> {r['目标']}")
```

#### 6.6.5 多跳推理: 基于路径生成答案

将多跳检索与LLM结合,实现基于知识路径的推理:

```python
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI

def multi_hop_reasoning(
    graph_store: Neo4jPropertyGraphStore,
    question: str,
    start_entity: str,
    max_hops: int = 3,
    top_k: int = 20
):
    """多跳推理: 基于问题和起始实体,检索路径并生成答案"""

    # 1. 检索多跳路径
    cypher_query = f"""
    MATCH path = (start {{name: '{start_entity}'}})-[*1..{max_hops}]->(target)
    WHERE start <> target
    RETURN
        [node in nodes(path) | node.name] as 路径,
        [rel in relationships(path) | type(rel)] as 关系,
        length(path) as 跳数
    ORDER BY 跳数
    LIMIT {top_k}
    """

    paths = graph_store.structured_query(cypher_query)

    # 2. 构建上下文
    context_parts = []
    print(f"检索到 {len(paths)} 条知识路径:\n")

    for idx, path_data in enumerate(paths[:10], 1):
        path_nodes = path_data['路径']
        relations = path_data['关系']
        hops = path_data['跳数']

        # 构建路径字符串
        path_str = path_nodes[0]
        for i in range(len(relations)):
            path_str += f" --[{relations[i]}]--> {path_nodes[i+1]}"

        print(f"[{idx}] ({hops}跳) {path_str}")
        context_parts.append(path_str)

    context = "\n".join(context_parts)

    # 3. 使用LLM生成答案
    prompt = f"""基于以下知识图谱路径信息,回答问题。

知识路径:
{context}

问题: {question}

请基于上述路径信息给出详细的答案,并说明推理过程。
"""

    llm = Settings.llm or OpenAI(model="gpt-4")
    response = llm.complete(prompt)

    print(f"\n{'='*60}")
    print("推理答案:")
    print(f"{'='*60}")
    print(response.text)

    return response.text

# 使用示例
multi_hop_reasoning(
    graph_store=graph_store,
    question="深度学习与多跳检索之间有什么联系?",
    start_entity="深度学习",
    max_hops=4
)
```

**输出示例**:
```
检索到 15 条知识路径:

[1] (1跳) 深度学习 --[使用]--> 神经网络
[2] (2跳) 深度学习 --[使用]--> 神经网络 --[模拟]--> 人脑结构
[3] (3跳) 深度学习 --[使用]--> 神经网络 --[应用于]--> 图像识别 --[属于]--> 计算机视觉
[4] (4跳) 深度学习 --[包含于]--> 机器学习 --[应用于]--> 知识图谱 --[支持]--> 多跳检索
...

============================================================
推理答案:
============================================================
基于知识图谱路径分析,深度学习与多跳检索的联系体现在:

1. **间接关联** (4跳路径):
   深度学习 → 机器学习 → 知识图谱 → 多跳检索

2. **技术支撑**:
   - 深度学习作为机器学习的子领域,提供了强大的表示学习能力
   - 知识图谱利用机器学习技术进行实体识别和关系抽取
   - 多跳检索依赖知识图谱的结构化知识存储

3. **应用协同**:
   深度学习可用于增强知识图谱的构建质量,进而提升多跳检索的准确性。
```

#### 6.6.6 与PropertyGraphIndex集成

将多跳检索与PropertyGraphIndex结合:

```python
from llama_index.core import PropertyGraphIndex, Document
from llama_index.embeddings.openai import OpenAIEmbedding

# 1. 创建PropertyGraphIndex
documents = [
    Document(text="""
    人工智能是计算机科学的分支。机器学习是人工智能的子领域。
    深度学习是机器学习的一种方法。神经网络是深度学习的核心技术。
    知识图谱存储结构化知识。多跳检索基于知识图谱实现复杂推理。
    """)
]

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

pg_index = PropertyGraphIndex.from_documents(
    documents,
    property_graph_store=graph_store,
    show_progress=True
)

# 2. 定义多跳检索工具
class GraphMultiHopRetriever:
    """图谱多跳检索器 (集成PropertyGraphIndex)"""

    def __init__(self, index: PropertyGraphIndex):
        self.index = index
        self.graph_store = index.property_graph_store

    def retrieve_paths(self, start_entity: str, max_hops: int = 3):
        """检索多跳路径"""
        cypher = f"""
        MATCH path = (start {{name: '{start_entity}'}})-[*1..{max_hops}]->(target)
        RETURN path
        LIMIT 20
        """
        return self.graph_store.structured_query(cypher)

    def retrieve_with_context(self, query: str, start_entity: str, max_hops: int = 2):
        """结合向量检索和多跳路径"""
        # 1. 向量检索获取相关上下文
        query_engine = self.index.as_query_engine(similarity_top_k=5)
        vector_response = query_engine.query(query)

        # 2. 多跳检索获取路径
        paths = self.retrieve_paths(start_entity, max_hops)

        # 3. 合并结果
        return {
            "vector_context": vector_response.response,
            "graph_paths": paths
        }

# 使用示例
retriever = GraphMultiHopRetriever(pg_index)
result = retriever.retrieve_with_context(
    query="多跳检索的工作原理",
    start_entity="知识图谱",
    max_hops=2
)

print("向量检索结果:", result["vector_context"])
print("图谱路径:", result["graph_paths"])
```

#### 6.6.7 多跳检索的实际应用

**案例1: 社交网络分析 - 朋友推荐**

```python
# 查找"朋友的朋友"但不是直接朋友
cypher_friend_of_friend = """
MATCH (user {name: '张三'})-[:FRIEND]->(friend)-[:FRIEND]->(fof)
WHERE NOT (user)-[:FRIEND]->(fof) AND user <> fof
RETURN DISTINCT fof.name as 推荐好友, count(friend) as 共同好友数
ORDER BY 共同好友数 DESC
LIMIT 10
"""

recommendations = graph_store.structured_query(cypher_friend_of_friend)
```

**案例2: 学术溯源 - 引用链追踪**

```python
# 追踪论文的引用链 (找到间接影响)
cypher_citation_chain = """
MATCH path = (paper {title: 'Attention Is All You Need'})<-[:CITES*1..3]-(citing_paper)
RETURN
    [node in nodes(path) | node.title] as 引用链,
    length(path) as 引用深度
ORDER BY 引用深度
LIMIT 20
"""

citation_chains = graph_store.structured_query(cypher_citation_chain)
```

**案例3: 企业关系挖掘 - 供应链追溯**

```python
# 追踪供应链的多层关系
cypher_supply_chain = """
MATCH path = (company {name: '苹果公司'})-[:SUPPLIES_TO*1..4]->(end_customer)
RETURN
    [node in nodes(path) | node.name] as 供应链,
    length(path) as 层级
ORDER BY 层级
"""

supply_chains = graph_store.structured_query(cypher_supply_chain)
```

#### 6.6.8 性能优化技巧

**1. 限制跳数范围**

```python
# ❌ 不推荐: 无限制跳数可能导致性能问题
cypher_bad = "MATCH path = (start)-[*]-(end) RETURN path"

# ✅ 推荐: 明确跳数上限
cypher_good = "MATCH path = (start)-[*1..3]-(end) RETURN path LIMIT 100"
```

**2. 使用索引**

```python
# 在Neo4j中创建索引加速查询
create_index_cypher = """
CREATE INDEX entity_name_index IF NOT EXISTS
FOR (n:Entity) ON (n.name)
"""
graph_store.structured_query(create_index_cypher)
```

**3. 路径过滤**

```python
# 只返回符合条件的路径
cypher_filtered = """
MATCH path = (start {name: '人工智能'})-[*1..3]->(target)
WHERE ALL(rel in relationships(path) WHERE type(rel) IN ['包含', '应用于'])
AND target.category = '技术'
RETURN path
LIMIT 20
"""
```

#### 6.6.9 小结

**多跳检索的核心价值**:
- ✅ 发现隐藏的知识关联
- ✅ 支持复杂的推理任务
- ✅ 揭示实体间的间接关系
- ✅ 构建完整的知识链条

**关键API**:
- `graph_store.structured_query()` - 执行Cypher查询
- `MATCH path = (start)-[*1..N]->(end)` - 可变跳数
- `shortestPath()` - 最短路径
- `length(path)` - 路径长度
- `nodes(path)`, `relationships(path)` - 提取路径元素

**最佳实践**:
1. 控制跳数上限 (通常1-4跳)
2. 使用LIMIT限制结果数量
3. 创建索引优化查询性能
4. 结合向量检索和路径检索
5. 将路径信息提供给LLM进行推理

---

### 6.7 实体消歧与冲突解决

在真实世界的知识图谱中,**实体消歧（Entity Resolution）** 是一个核心挑战:如何判断两个看似不同的实体记录是否指向同一个真实实体?

#### 6.7.1 实体消歧问题

**问题场景**:

```
实体A: 名字="张伟", 职业="工程师", 社媒="@zhangwei123"
实体B: 名字="张伟", 职业="教师",   社媒="@zhangwei456"
实体C: 名字="Zhang Wei", 职业="工程师", 社媒="@zhangwei123"

问题:
- A vs B: 是同一个人吗? → 不是 (职业、社媒账号都不同)
- A vs C: 是同一个人吗? → 是 (社媒账号相同,只是中英文名)
```

**挑战**:
- **同名异人**: 多个不同的人有相同名字
- **异名同人**: 同一个人有多个名字(别名、曾用名、中英文名)
- **数据质量**: 拼写错误、格式不一致
- **属性冲突**: 同一个人的不同记录可能有矛盾的属性

#### 6.7.2 LLM辅助实体消歧

使用LLM的结构化输出能力进行智能判断:

```python
from pydantic import BaseModel, Field
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI

class EntityMatch(BaseModel):
    """实体匹配结果"""
    is_same_entity: bool = Field(description="是否是同一个实体")
    confidence: float = Field(
        description="置信度 0-1,表示判断的可信程度",
        ge=0.0,
        le=1.0
    )
    reason: str = Field(description="判断理由,说明依据哪些信息做出判断")
    merge_strategy: str = Field(
        description="合并策略: keep_both(保留两个), merge_to_first(合并到第一个), merge_to_second(合并到第二个)"
    )

def resolve_entities(entity_a: dict, entity_b: dict, llm: OpenAI) -> EntityMatch:
    """使用LLM判断两个实体是否相同"""

    prompt = f"""判断以下两个实体是否是同一个人/物:

实体A: {entity_a}
实体B: {entity_b}

判断依据:
1. **社媒账号相同** → 大概率同一人 (强匹配信号)
2. **邮箱/电话相同** → 大概率同一人
3. **名字相似** + 其他属性相似 → 可能同一人
4. **名字相同** 但其他属性完全不同 → 可能是同名异人

请基于上述规则,判断是否是同一个实体,并给出置信度和理由。
"""

    # 使用LlamaIndex的structured_predict
    result = llm.structured_predict(
        EntityMatch,
        prompt=prompt
    )

    return result

# 使用示例
llm = OpenAI(model="gpt-4", temperature=0)

entity_a = {
    "name": "张伟",
    "occupation": "工程师",
    "social_media": "@zhangwei123",
    "company": "科技公司A"
}

entity_b = {
    "name": "Zhang Wei",
    "occupation": "Software Engineer",
    "social_media": "@zhangwei123",
    "company": "Tech Company A"
}

match_result = resolve_entities(entity_a, entity_b, llm)

print(f"是否同一实体: {match_result.is_same_entity}")
print(f"置信度: {match_result.confidence:.2f}")
print(f"理由: {match_result.reason}")
print(f"合并策略: {match_result.merge_strategy}")
```

**输出示例**:
```
是否同一实体: True
置信度: 0.95
理由: 两个实体的社媒账号完全相同(@zhangwei123),且职业和公司信息语义一致(工程师=Software Engineer, 科技公司A=Tech Company A),只是中英文表述不同。社媒账号作为强匹配信号,可以确定是同一人。
合并策略: merge_to_first
```

#### 6.7.3 构建"可能相同"关系

不直接合并实体,而是构建 `POSSIBLY_SAME_AS` 关系,保留原始数据:

```python
def create_possibly_same_relation(
    graph_store: Neo4jPropertyGraphStore,
    entity_a_id: str,
    entity_b_id: str,
    confidence: float,
    reason: str
):
    """构建"可能相同"关系"""

    cypher_create = """
    MATCH (a {id: $entity_a_id}), (b {id: $entity_b_id})
    CREATE (a)-[:POSSIBLY_SAME_AS {
        confidence: $confidence,
        reason: $reason,
        created_at: datetime(),
        status: 'pending'  // 待人工确认
    }]->(b)
    """

    graph_store.structured_query(
        cypher_create,
        param_map={
            "entity_a_id": entity_a_id,
            "entity_b_id": entity_b_id,
            "confidence": confidence,
            "reason": reason
        }
    )

    print(f"✅ 创建 POSSIBLY_SAME_AS 关系: {entity_a_id} <-> {entity_b_id} (置信度: {confidence:.2f})")

# 使用示例
create_possibly_same_relation(
    graph_store=graph_store,
    entity_a_id="person:001",
    entity_b_id="person:002",
    confidence=0.95,
    reason="社媒账号相同"
)
```

#### 6.7.4 查询时动态解析

在查询时考虑"可能相同"的实体:

```python
def query_with_entity_resolution(
    graph_store: Neo4jPropertyGraphStore,
    entity_name: str,
    confidence_threshold: float = 0.8
):
    """查询时动态解析实体,包含可能相同的实体"""

    cypher_query = f"""
    MATCH (start {{name: '{entity_name}'}})

    // 找到所有可能相同的实体 (双向)
    OPTIONAL MATCH (start)-[same:POSSIBLY_SAME_AS]-(equivalent)
    WHERE same.confidence >= {confidence_threshold}

    // 合并所有等价实体
    WITH collect(DISTINCT equivalent) + [start] as all_entities

    // 展开并查询每个实体的关系
    UNWIND all_entities as entity
    MATCH (entity)-[r]->(target)

    RETURN DISTINCT
        entity.name as 来源实体,
        type(r) as 关系类型,
        target.name as 目标实体
    """

    results = graph_store.structured_query(cypher_query)

    print(f"查询实体 '{entity_name}' (包含等价实体):")
    for r in results:
        print(f"  {r['来源实体']} --[{r['关系类型']}]--> {r['目标实体']}")

    return results

# 使用示例
query_with_entity_resolution(
    graph_store=graph_store,
    entity_name="张伟",
    confidence_threshold=0.8
)
```

**输出**:
```
查询实体 '张伟' (包含等价实体):
  张伟 --[工作于]--> 科技公司A
  Zhang Wei --[发表]--> 论文X
  张伟 --[认识]--> 李四
```

#### 6.7.5 实体合并策略

定义不同的合并策略:

```python
from enum import Enum

class MergeStrategy(str, Enum):
    """实体合并策略"""
    KEEP_BOTH = "keep_both"              # 保留两个独立实体
    MERGE_TO_FIRST = "merge_to_first"    # 合并到第一个,删除第二个
    MERGE_TO_SECOND = "merge_to_second"  # 合并到第二个,删除第一个
    CREATE_ALIAS = "create_alias"        # 创建别名关系
    MANUAL_REVIEW = "manual_review"      # 人工审核

def execute_merge_strategy(
    graph_store: Neo4jPropertyGraphStore,
    entity_a_id: str,
    entity_b_id: str,
    strategy: MergeStrategy
):
    """执行合并策略"""

    if strategy == MergeStrategy.KEEP_BOTH:
        # 只创建POSSIBLY_SAME_AS关系,不合并
        print(f"保留两个实体: {entity_a_id}, {entity_b_id}")

    elif strategy == MergeStrategy.MERGE_TO_FIRST:
        # 将B的所有关系转移到A,然后删除B
        cypher_merge = """
        MATCH (a {id: $entity_a_id}), (b {id: $entity_b_id})

        // 1. 转移B的所有出边到A
        OPTIONAL MATCH (b)-[r]->(target)
        WHERE NOT (a)-[]->(target)  // 避免重复
        CREATE (a)-[new_r:SAME_TYPE_AS_r]->(target)
        SET new_r = properties(r)

        // 2. 转移B的所有入边到A
        OPTIONAL MATCH (source)-[r]->(b)
        WHERE NOT (source)-[]->(a)
        CREATE (source)-[new_r:SAME_TYPE_AS_r]->(a)
        SET new_r = properties(r)

        // 3. 删除B
        DETACH DELETE b
        """

        graph_store.structured_query(
            cypher_merge,
            param_map={
                "entity_a_id": entity_a_id,
                "entity_b_id": entity_b_id
            }
        )
        print(f"✅ 合并完成: {entity_b_id} → {entity_a_id}")

    elif strategy == MergeStrategy.CREATE_ALIAS:
        # 创建别名关系: A --[HAS_ALIAS]--> B
        cypher_alias = """
        MATCH (a {id: $entity_a_id}), (b {id: $entity_b_id})
        CREATE (a)-[:HAS_ALIAS]->(b)
        """
        graph_store.structured_query(
            cypher_alias,
            param_map={
                "entity_a_id": entity_a_id,
                "entity_b_id": entity_b_id
            }
        )
        print(f"✅ 创建别名关系: {entity_a_id} --[HAS_ALIAS]--> {entity_b_id}")

    elif strategy == MergeStrategy.MANUAL_REVIEW:
        # 标记为待人工审核
        cypher_mark = """
        MATCH (a {id: $entity_a_id})-[r:POSSIBLY_SAME_AS]-(b {id: $entity_b_id})
        SET r.status = 'manual_review_required'
        SET r.flagged_at = datetime()
        """
        graph_store.structured_query(
            cypher_mark,
            param_map={
                "entity_a_id": entity_a_id,
                "entity_b_id": entity_b_id
            }
        )
        print(f"⚠️ 标记为人工审核: {entity_a_id} <-> {entity_b_id}")

# 使用示例
execute_merge_strategy(
    graph_store=graph_store,
    entity_a_id="person:001",
    entity_b_id="person:002",
    strategy=MergeStrategy.CREATE_ALIAS
)
```

#### 6.7.6 生产级实体消歧Pipeline

完整的实体消歧流程:

```python
class EntityResolutionPipeline:
    """实体消歧Pipeline"""

    def __init__(self, graph_store: Neo4jPropertyGraphStore, llm: OpenAI):
        self.graph_store = graph_store
        self.llm = llm

    def find_duplicate_candidates(self, similarity_threshold: float = 0.7):
        """查找可能重复的实体候选"""

        # 方法1: 基于名字相似度
        cypher_similar_names = """
        MATCH (a:Entity), (b:Entity)
        WHERE a.id < b.id  // 避免重复比较
        AND apoc.text.levenshteinSimilarity(a.name, b.name) > $threshold
        RETURN a.id as entity_a, b.id as entity_b, a.name as name_a, b.name as name_b
        """

        # 方法2: 基于共享属性(社媒账号、邮箱、电话等)
        cypher_shared_attrs = """
        MATCH (a:Entity), (b:Entity)
        WHERE a.id < b.id
        AND (
            a.social_media = b.social_media OR
            a.email = b.email OR
            a.phone = b.phone
        )
        RETURN a.id as entity_a, b.id as entity_b
        """

        candidates = []

        # 执行查询
        similar_names = self.graph_store.structured_query(
            cypher_similar_names,
            param_map={"threshold": similarity_threshold}
        )
        candidates.extend(similar_names)

        shared_attrs = self.graph_store.structured_query(cypher_shared_attrs)
        candidates.extend(shared_attrs)

        return candidates

    def batch_resolve(self, candidates: list, auto_merge_threshold: float = 0.9):
        """批量消歧"""

        results = {
            "merged": [],
            "flagged_for_review": [],
            "kept_separate": []
        }

        for candidate in candidates:
            entity_a_id = candidate['entity_a']
            entity_b_id = candidate['entity_b']

            # 获取完整实体信息
            entity_a = self._get_entity(entity_a_id)
            entity_b = self._get_entity(entity_b_id)

            # LLM判断
            match_result = resolve_entities(entity_a, entity_b, self.llm)

            # 根据置信度决定策略
            if match_result.is_same_entity:
                if match_result.confidence >= auto_merge_threshold:
                    # 自动合并
                    execute_merge_strategy(
                        self.graph_store,
                        entity_a_id,
                        entity_b_id,
                        MergeStrategy.MERGE_TO_FIRST
                    )
                    results["merged"].append({
                        "entity_a": entity_a_id,
                        "entity_b": entity_b_id,
                        "confidence": match_result.confidence
                    })
                else:
                    # 标记为人工审核
                    execute_merge_strategy(
                        self.graph_store,
                        entity_a_id,
                        entity_b_id,
                        MergeStrategy.MANUAL_REVIEW
                    )
                    results["flagged_for_review"].append({
                        "entity_a": entity_a_id,
                        "entity_b": entity_b_id,
                        "confidence": match_result.confidence,
                        "reason": match_result.reason
                    })
            else:
                # 确认是不同实体,保持分离
                results["kept_separate"].append({
                    "entity_a": entity_a_id,
                    "entity_b": entity_b_id
                })

        return results

    def _get_entity(self, entity_id: str) -> dict:
        """获取实体完整信息"""
        cypher = f"MATCH (e {{id: '{entity_id}'}}) RETURN properties(e) as props"
        result = self.graph_store.structured_query(cypher)
        return result[0]['props'] if result else {}

# 使用示例
pipeline = EntityResolutionPipeline(graph_store, llm)

# 1. 查找候选
candidates = pipeline.find_duplicate_candidates(similarity_threshold=0.7)
print(f"找到 {len(candidates)} 组可能重复的实体")

# 2. 批量消歧
results = pipeline.batch_resolve(candidates, auto_merge_threshold=0.9)

print(f"\n✅ 自动合并: {len(results['merged'])} 组")
print(f"⚠️ 待人工审核: {len(results['flagged_for_review'])} 组")
print(f"📌 保持分离: {len(results['kept_separate'])} 组")
```

#### 6.7.7 实际应用案例

**案例1: 企业知识图谱去重**

```python
# 场景: 员工信息从多个系统导入,存在重复
employees = [
    {"id": "emp001", "name": "张三", "email": "zhangsan@company.com", "dept": "技术部"},
    {"id": "emp002", "name": "Zhang San", "email": "zhangsan@company.com", "dept": "Engineering"},
    {"id": "emp003", "name": "张三", "email": "zs@company.com", "dept": "市场部"}
]

# emp001 和 emp002: 邮箱相同 → 同一人
# emp001 和 emp003: 名字相同但邮箱、部门不同 → 可能是不同人(同名)
```

**案例2: 学术网络作者消歧**

```python
# 场景: 同名作者消歧
authors = [
    {"name": "李伟", "institution": "清华大学", "field": "计算机"},
    {"name": "Li Wei", "institution": "Tsinghua University", "field": "Computer Science"},
    {"name": "李伟", "institution": "北京大学", "field": "物理"}
]

# 前两个: 机构和领域一致 → 同一人
# 第三个: 不同机构和领域 → 不同人
```

**案例3: 电商商品去重**

```python
# 场景: 商品信息去重
products = [
    {"name": "iPhone 15 Pro", "sku": "A2848", "price": 7999},
    {"name": "苹果iPhone15Pro", "sku": "A2848", "price": 7999},
    {"name": "iPhone 15 Pro Max", "sku": "A2849", "price": 8999}
]

# 前两个: SKU相同 → 同一商品
# 第三个: 不同SKU → 不同商品
```

#### 6.7.8 小结

**实体消歧的核心价值**:
- ✅ 提高知识图谱质量 (去除重复)
- ✅ 避免信息分散 (同一实体的知识集中)
- ✅ 支持精确查询 (避免遗漏)
- ✅ 适应真实世界的数据混乱

**关键技术**:
- **LLM结构化输出**: `llm.structured_predict()` + Pydantic
- **相似度计算**: Levenshtein距离, 属性匹配
- **关系建模**: `POSSIBLY_SAME_AS`, `HAS_ALIAS`
- **合并策略**: 保留/合并/人工审核

**最佳实践**:
1. **不要急于合并**: 先建立 `POSSIBLY_SAME_AS` 关系
2. **设置置信度阈值**: 高置信度自动合并,低置信度人工审核
3. **保留审计日志**: 记录合并原因和时间
4. **支持回滚**: 允许撤销错误的合并
5. **增量处理**: 新增实体时实时检测重复

**常用规则**:
- 社媒账号/邮箱/电话相同 → 强匹配信号 (置信度0.9+)
- 名字完全相同 + 其他属性相似 → 中等信号 (置信度0.7-0.9)
- 名字相似但其他属性矛盾 → 可能同名异人 (置信度<0.5)

---

### 6.8 可视化知识图谱

使用NetworkX和Matplotlib可视化PropertyGraphIndex:

```python
import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(pg_index, max_triplets: int = 50):
    """可视化知识图谱

    Args:
        pg_index: PropertyGraphIndex实例
        max_triplets: 最多显示的三元组数量 (避免图过于复杂)
    """
    # 获取所有三元组
    triplets = pg_index.property_graph_store.get_triplets()

    # 创建 NetworkX 有向图
    G = nx.DiGraph()

    for triplet in triplets[:max_triplets]:  # 限制节点数量
        subject, relation, obj = triplet
        G.add_edge(
            subject.name,
            obj.name,
            label=relation.label
        )

    # 绘制图
    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(G, k=0.5, iterations=50)

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)

    # 绘制节点标签
    nx.draw_networkx_labels(G, pos, font_size=8)

    # 绘制边
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=15)

    # 绘制边标签 (关系类型)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)

    plt.title("Knowledge Graph Visualization", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("knowledge_graph.png", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✅ 可视化完成! 显示了 {len(triplets[:max_triplets])} 个三元组")
    print(f"   图中包含 {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")

# 使用示例
visualize_graph(pg_index, max_triplets=50)
```

**高级可视化: 按实体类型着色**

```python
def visualize_graph_by_type(pg_index, max_triplets: int = 50):
    """按实体类型着色的知识图谱可视化"""
    triplets = pg_index.property_graph_store.get_triplets()
    G = nx.DiGraph()

    # 存储节点类型
    node_types = {}

    for triplet in triplets[:max_triplets]:
        subject, relation, obj = triplet
        G.add_edge(subject.name, obj.name, label=relation.label)

        # 记录节点类型 (如果有category属性)
        if hasattr(subject, 'category'):
            node_types[subject.name] = subject.category
        if hasattr(obj, 'category'):
            node_types[obj.name] = obj.category

    # 根据类型分配颜色
    type_colors = {
        'PERSON': '#FF6B6B',
        'ORGANIZATION': '#4ECDC4',
        'LOCATION': '#45B7D1',
        'TECHNOLOGY': '#FFA07A',
    }

    node_colors = [
        type_colors.get(node_types.get(node, 'UNKNOWN'), '#CCCCCC')
        for node in G.nodes()
    ]

    # 绘制
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G, k=0.7, iterations=50)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20, alpha=0.6)

    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color, label=entity_type)
        for entity_type, color in type_colors.items()
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.title("Knowledge Graph with Entity Type Coloring", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("knowledge_graph_colored.png", dpi=300, bbox_inches='tight')
    plt.show()

visualize_graph_by_type(pg_index)
```

**交互式可视化: 使用Pyvis**

```python
from pyvis.network import Network

def visualize_graph_interactive(pg_index, max_triplets: int = 100):
    """创建交互式知识图谱可视化 (HTML)"""
    triplets = pg_index.property_graph_store.get_triplets()

    # 创建Pyvis网络
    net = Network(height='800px', width='100%', notebook=False, directed=True)

    # 添加节点和边
    for triplet in triplets[:max_triplets]:
        subject, relation, obj = triplet

        # 添加节点
        net.add_node(subject.name, title=subject.name, color='#97C2FC')
        net.add_node(obj.name, title=obj.name, color='#FFAB91')

        # 添加边
        net.add_edge(
            subject.name,
            obj.name,
            label=relation.label,
            title=relation.label
        )

    # 设置物理布局
    net.set_options("""
    {
        "physics": {
            "enabled": true,
            "barnesHut": {
                "gravitationalConstant": -8000,
                "springLength": 150,
                "springConstant": 0.04
            }
        }
    }
    """)

    # 保存为HTML文件
    net.save_graph("knowledge_graph_interactive.html")
    print("✅ 交互式可视化已保存到 knowledge_graph_interactive.html")
    print("   在浏览器中打开该文件即可查看交互式图谱")

# 需要先安装: pip install pyvis
visualize_graph_interactive(pg_index)
```

**可视化的应用场景**:

1. **调试知识抽取**: 检查提取的三元组是否正确
2. **演示文稿**: 展示知识图谱的结构
3. **发现模式**: 直观发现实体间的关系模式
4. **验证合并**: 查看实体消歧后的图结构

---

### 6.9 GraphRAG社区检测 (Community Detection)

**GraphRAG (Graph Retrieval-Augmented Generation)** 是微软提出的一种基于知识图谱的RAG方法论,核心创新是通过**社区检测**将大规模知识图谱分层组织,并为每个社区生成LLM摘要,实现高效的全局性问题回答。

#### 6.9.1 社区检测概念与动机

**什么是社区检测?**

社区检测(Community Detection)是图论中的经典问题,目标是将图中的节点划分为多个**紧密连接的群组(社区)**,使得:
- 社区内部的节点连接密集
- 社区之间的连接稀疏

```
原始知识图谱 (1000个实体):
  实体1 --关系--> 实体2 --关系--> 实体3 ...

社区检测后:
  ┌─────────────────────────────────┐
  │ 社区1: AI技术 (120个实体)       │
  │ 摘要: 讨论深度学习、NLP等技术   │
  └─────────────────────────────────┘

  ┌─────────────────────────────────┐
  │ 社区2: 科技公司 (85个实体)      │
  │ 摘要: OpenAI、Google等公司动态  │
  └─────────────────────────────────┘

  ┌─────────────────────────────────┐
  │ 社区3: 产品发布 (60个实体)      │
  │ 摘要: GPT-4、Claude等产品信息   │
  └─────────────────────────────────┘
```

**为什么需要社区检测?**

| 传统方法 | GraphRAG方法 |
|---------|-------------|
| 向量检索: 检索文档块 → 生成答案 | 社区检测: 建图 → 社区划分 → 社区摘要 → 基于摘要回答 |
| 适合事实性问题 | 适合全局性、综述性问题 |
| 难以回答"文档的主要主题是什么?" | 可以高效回答全局性问题 |
| 每次检索需要遍历大量文档块 | 只需处理少量社区摘要 |

**应用场景**:
- ✅ "这篇文档讨论了哪些主要主题?" (全局性)
- ✅ "总结文档中的关键事件" (综述性)
- ✅ "文档中提到的公司有哪些战略方向?" (宏观分析)

**Leiden算法简介**:

LlamaIndex官方使用的是 `hierarchical_leiden` 算法(来自`graspologic`库):
- **Leiden算法**: 社区检测的经典算法,改进自Louvain算法
- **Hierarchical版本**: 支持层次化社区划分
- **max_cluster_size参数**: 控制社区大小上限,避免某个社区过大

---

#### 6.9.2 GraphRAGExtractor: 带描述的实体关系提取

官方GraphRAG的第一步是提取**带详细描述**的实体和关系,这与SimpleLLMPathExtractor有显著区别。

**核心区别对比**:

| 特性 | SimpleLLMPathExtractor | GraphRAGExtractor (官方) |
|------|----------------------|------------------------|
| 实体信息 | 仅name, type | name, type, **description** |
| 关系信息 | 仅subject, relation, object | subject, relation, object, **description** |
| 描述存储 | ❌ 不存储 | ✅ 存储在metadata |
| 提取格式 | 三元组 | JSON (entities + relationships) |
| 用途 | 基础图查询 | 社区摘要生成 |

**GraphRAGExtractor实现**:

```python
from llama_index.core.schema import TransformComponent, BaseNode
from llama_index.core.graph_stores.types import (
    EntityNode,
    Relation,
    KG_NODES_KEY,
    KG_RELATIONS_KEY
)
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
import json
import re
from typing import Any, Callable

class GraphRAGExtractor(TransformComponent):
    """官方GraphRAG实体关系提取器

    提取:
      - 实体: name, type, description
      - 关系: source, target, relation, relationship_description
    """

    llm: OpenAI
    extract_prompt: str
    parse_fn: Callable
    max_paths_per_chunk: int = 10
    num_workers: int = 4

    def __init__(
        self,
        llm: OpenAI,
        extract_prompt: str,
        parse_fn: Callable,
        max_paths_per_chunk: int = 10,
        num_workers: int = 4
    ):
        self.llm = llm
        self.extract_prompt = extract_prompt
        self.parse_fn = parse_fn
        self.max_paths_per_chunk = max_paths_per_chunk
        self.num_workers = num_workers

    async def _aextract(self, node: BaseNode) -> BaseNode:
        """提取单个节点的实体和关系"""
        text = node.get_content(metadata_mode="llm")

        # 1. LLM提取
        llm_response = await self.llm.acomplete(
            self.extract_prompt.format(
                text=text,
                max_knowledge_triplets=self.max_paths_per_chunk
            )
        )

        # 2. 解析响应
        entities, relationships = self.parse_fn(str(llm_response))

        # 3. 构建EntityNode和Relation
        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])

        # 存储实体 (带description)
        metadata = node.metadata.copy()
        for entity, entity_type, description in entities:
            entity_metadata = metadata.copy()
            entity_metadata["entity_description"] = description  # 关键: 存储描述

            entity_node = EntityNode(
                name=entity,
                label=entity_type,
                properties=entity_metadata
            )
            existing_nodes.append(entity_node)

        # 存储关系 (带relationship_description)
        for subj, obj, rel, description in relationships:
            rel_metadata = metadata.copy()
            rel_metadata["relationship_description"] = description  # 关键: 存储描述

            subj_node = EntityNode(name=subj, properties=metadata)
            obj_node = EntityNode(name=obj, properties=metadata)

            rel_node = Relation(
                label=rel,
                source_id=subj_node.id,
                target_id=obj_node.id,
                properties=rel_metadata
            )

            existing_nodes.extend([subj_node, obj_node])
            existing_relations.append(rel_node)

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations

        return node

    def __call__(self, nodes, **kwargs):
        """批量处理节点"""
        import asyncio

        async def _process_all():
            tasks = [self._aextract(node) for node in nodes]
            return await asyncio.gather(*tasks)

        return asyncio.run(_process_all())
```

**官方提取Prompt**:

```python
KG_TRIPLET_EXTRACT_TMPL = """
-Goal-
Given a text document, identify all entities and their entity types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: Type of the entity (one of: PERSON, ORGANIZATION, LOCATION, TECHNOLOGY, EVENT, CONCEPT)
- entity_description: Comprehensive description of the entity's attributes and activities

Format each entity as a JSON object.

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relation: relationship between source_entity and target_entity (e.g., WORKS_AT, LOCATED_IN, DEVELOPED, USES)
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other

Format each relationship as a JSON object.

3. Output Formatting:
When you finish, output a single JSON object with two keys:
- "entities": list of entity objects
- "relationships": list of relationship objects

-Real Data-
text: {text}
output:
"""
```

**解析函数**:

```python
def parse_fn(response_str: str) -> tuple:
    """解析LLM响应,提取实体和关系

    Returns:
        entities: [(name, type, description), ...]
        relationships: [(source, target, relation, description), ...]
    """
    # 提取JSON
    json_pattern = r"\{.*\}"
    match = re.search(json_pattern, response_str, re.DOTALL)

    entities = []
    relationships = []

    if not match:
        return entities, relationships

    try:
        data = json.loads(match.group(0))

        # 解析实体
        for entity in data.get("entities", []):
            entities.append((
                entity["entity_name"],
                entity["entity_type"],
                entity["entity_description"]
            ))

        # 解析关系
        for relation in data.get("relationships", []):
            relationships.append((
                relation["source_entity"],
                relation["target_entity"],
                relation["relation"],
                relation["relationship_description"]
            ))

    except json.JSONDecodeError:
        print(f"Failed to parse JSON: {match.group(0)[:100]}...")

    return entities, relationships
```

**使用示例**:

```python
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

# 1. 初始化LLM
llm = OpenAI(model="gpt-4", temperature=0)

# 2. 创建提取器
kg_extractor = GraphRAGExtractor(
    llm=llm,
    extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
    parse_fn=parse_fn,
    max_paths_per_chunk=10,
    num_workers=4
)

# 3. 加载并分块文档
documents = SimpleDirectoryReader("./data").load_data()
splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
nodes = splitter.get_nodes_from_documents(documents)

# 4. 提取实体和关系
enriched_nodes = kg_extractor(nodes)

# 5. 查看提取结果
for node in enriched_nodes[:1]:
    entities = node.metadata.get(KG_NODES_KEY, [])
    relations = node.metadata.get(KG_RELATIONS_KEY, [])

    print(f"提取了 {len(entities)} 个实体, {len(relations)} 个关系\n")

    # 实体示例
    if entities:
        entity = entities[0]
        print(f"实体示例: {entity.name} ({entity.label})")
        print(f"描述: {entity.properties.get('entity_description', 'N/A')}\n")

    # 关系示例
    if relations:
        rel = relations[0]
        print(f"关系示例: {rel.label}")
        print(f"描述: {rel.properties.get('relationship_description', 'N/A')}")
```

---

#### 6.9.3 GraphRAGStore: 社区构建与摘要

**GraphRAGStore** 是官方实现的核心类,继承自`SimplePropertyGraphStore`,新增了社区检测和摘要功能。

**完整实现**:

```python
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
import networkx as nx
from graspologic.partition import hierarchical_leiden
import re

class GraphRAGStore(SimplePropertyGraphStore):
    """官方GraphRAG存储,支持社区检测和摘要"""

    community_summary: dict = {}
    entity_info: dict = {}  # 实体 -> 社区ID映射
    max_cluster_size: int = 5

    def build_communities(self):
        """构建社区并生成摘要

        流程:
          1. 转换为NetworkX图
          2. Leiden社区检测
          3. 收集社区信息
          4. 生成社区摘要
        """
        print("🔍 开始构建社区...")

        # 1. 创建NetworkX图
        nx_graph = self._create_nx_graph()
        print(f"  图包含 {nx_graph.number_of_nodes()} 个节点, {nx_graph.number_of_edges()} 条边")

        # 2. 社区检测
        print(f"  执行Leiden社区检测 (max_cluster_size={self.max_cluster_size})...")
        community_hierarchical_clusters = hierarchical_leiden(
            nx_graph,
            max_cluster_size=self.max_cluster_size
        )

        # 3. 收集社区信息
        self.entity_info, community_info = self._collect_community_info(
            nx_graph,
            community_hierarchical_clusters
        )

        num_communities = len(community_info)
        print(f"  检测到 {num_communities} 个社区")

        # 4. 生成摘要
        print(f"  为 {num_communities} 个社区生成LLM摘要...")
        self._summarize_communities(community_info)

        print("✅ 社区构建完成!")

        # 打印社区统计
        for cid, summary in self.community_summary.items():
            entity_count = sum(1 for eid, c in self.entity_info.items() if c == cid)
            print(f"\n  社区 {cid}: {entity_count} 个实体")
            print(f"  摘要: {summary[:100]}...")

    def _create_nx_graph(self) -> nx.Graph:
        """将内部图表示转换为NetworkX图"""
        nx_graph = nx.Graph()

        # 添加节点
        for node in self.graph.nodes.values():
            nx_graph.add_node(str(node))

        # 添加边 (带关系描述)
        for relation in self.graph.relations.values():
            nx_graph.add_edge(
                relation.source_id,
                relation.target_id,
                relationship=relation.label,
                description=relation.properties.get("relationship_description", "")
            )

        return nx_graph

    def _collect_community_info(
        self,
        nx_graph: nx.Graph,
        clusters
    ) -> tuple[dict, dict]:
        """收集每个社区的详细信息

        Returns:
            entity_info: {entity_name: community_id}
            community_info: {community_id: [relationship_strings]}
        """
        # 构建实体 -> 社区映射
        entity_info = {}
        for item in clusters:
            entity_info[item.node] = item.cluster

        # 收集社区内部的关系
        community_info = {}

        for item in clusters:
            cluster_id = item.cluster
            node = item.node

            if cluster_id not in community_info:
                community_info[cluster_id] = []

            # 遍历邻居节点
            for neighbor in nx_graph.neighbors(node):
                # 只收集同一社区内的关系
                if entity_info[neighbor] == cluster_id:
                    edge_data = nx_graph.get_edge_data(node, neighbor)

                    if edge_data:
                        # 格式: "entity1 -> entity2 -> relation -> description"
                        detail = (
                            f"{node} -> {neighbor} -> "
                            f"{edge_data['relationship']} -> "
                            f"{edge_data['description']}"
                        )
                        community_info[cluster_id].append(detail)

        return entity_info, community_info

    def _summarize_communities(self, community_info: dict):
        """为每个社区生成LLM摘要"""
        for community_id, details in community_info.items():
            # 拼接社区内所有关系
            details_text = "\n".join(set(details)) + "."  # 去重

            # 生成摘要
            summary = self.generate_community_summary(details_text)
            self.community_summary[community_id] = summary

    def generate_community_summary(self, text: str) -> str:
        """使用LLM为社区生成摘要"""
        llm = OpenAI(model="gpt-4", temperature=0)

        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are provided with a set of relationships from a knowledge graph, "
                    "each represented as entity1->entity2->relation->relationship_description. "
                    "Your task is to create a summary of these relationships. "
                    "The summary should include the names of the entities involved and "
                    "a concise synthesis of the relationship descriptions. "
                    "The summary should be 2-3 sentences."
                )
            ),
            ChatMessage(
                role="user",
                content=text
            )
        ]

        response = llm.chat(messages)
        clean_summary = re.sub(r"^assistant:\s*", "", str(response)).strip()

        return clean_summary

    def get_community_summaries(self) -> dict:
        """获取所有社区摘要"""
        return self.community_summary
```

**关键参数 `max_cluster_size`**:

| max_cluster_size | 社区数量 | 社区大小 | 适用场景 |
|-----------------|---------|---------|---------|
| 3 | 多 | 小 | 文档较小,需要细粒度 |
| 5 (默认) | 中等 | 中等 | 通用场景 |
| 10 | 少 | 大 | 文档较大,需要宏观视角 |

**安装依赖**:

```bash
pip install graspologic networkx
```

---

#### 6.9.4 完整示例: 从文档到社区图谱

**端到端流程**:

```python
from llama_index.core import PropertyGraphIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter

# ========== 配置 ==========
Settings.llm = OpenAI(model="gpt-4", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# ========== 1. 加载文档 ==========
documents = SimpleDirectoryReader("./data").load_data()
print(f"加载了 {len(documents)} 个文档")

# ========== 2. 分块 ==========
splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
nodes = splitter.get_nodes_from_documents(documents)
print(f"分成 {len(nodes)} 个文本块")

# ========== 3. 创建GraphRAG提取器 ==========
kg_extractor = GraphRAGExtractor(
    llm=Settings.llm,
    extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
    parse_fn=parse_fn,
    max_paths_per_chunk=10
)

# ========== 4. 创建GraphRAGStore ==========
graph_store = GraphRAGStore()

# ========== 5. 构建PropertyGraphIndex ==========
print("\n🏗️  构建知识图谱...")
pg_index = PropertyGraphIndex(
    nodes=nodes,
    property_graph_store=graph_store,
    kg_extractors=[kg_extractor],
    show_progress=True
)

# ========== 6. 构建社区 ==========
print("\n🔍 构建社区并生成摘要...")
pg_index.property_graph_store.build_communities()

# ========== 7. 查看结果 ==========
print("\n" + "="*60)
print("社区摘要:")
print("="*60)

for community_id, summary in graph_store.community_summary.items():
    # 统计社区实体数量
    entity_count = sum(
        1 for entity, cid in graph_store.entity_info.items()
        if cid == community_id
    )

    print(f"\n【社区 {community_id}】 ({entity_count} 个实体)")
    print(f"摘要: {summary}")

# ========== 8. 保存 ==========
pg_index.storage_context.persist(persist_dir="./storage/graphrag")
print("\n✅ 知识图谱已保存到 ./storage/graphrag")
```

**输出示例**:

```
加载了 3 个文档
分成 15 个文本块

🏗️  构建知识图谱...
Processing nodes: 100%|██████████| 15/15 [00:45<00:00,  3.2s/it]

🔍 构建社区并生成摘要...
🔍 开始构建社区...
  图包含 45 个节点, 67 条边
  执行Leiden社区检测 (max_cluster_size=5)...
  检测到 8 个社区
  为 8 个社区生成LLM摘要...
✅ 社区构建完成!

  社区 0: 12 个实体
  摘要: This community discusses AI technologies including GPT-4, DALL-E...

  社区 1: 8 个实体
  摘要: This community focuses on tech companies like OpenAI, Microsoft...

============================================================
社区摘要:
============================================================

【社区 0】 (12 个实体)
摘要: This community discusses AI technologies including GPT-4, DALL-E, and their applications in natural language processing and image generation.

【社区 1】 (8 个实体)
摘要: This community focuses on tech companies like OpenAI, Microsoft, and Google, highlighting their investments and product releases in the AI sector.

...
```

---

#### 6.9.5 可视化社区结构

使用不同颜色标记不同社区:

```python
import networkx as nx
import matplotlib.pyplot as plt

def visualize_communities(graph_store: GraphRAGStore):
    """可视化社区结构"""
    nx_graph = graph_store._create_nx_graph()

    # 为每个社区分配颜色
    community_colors = {}
    color_palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A',
                     '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2']

    node_colors = []
    for node in nx_graph.nodes():
        community_id = graph_store.entity_info.get(node, -1)

        if community_id not in community_colors:
            community_colors[community_id] = color_palette[
                len(community_colors) % len(color_palette)
            ]

        node_colors.append(community_colors[community_id])

    # 绘制
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(nx_graph, k=0.5, iterations=50)

    nx.draw_networkx_nodes(
        nx_graph, pos,
        node_color=node_colors,
        node_size=600,
        alpha=0.9
    )
    nx.draw_networkx_labels(nx_graph, pos, font_size=8)
    nx.draw_networkx_edges(nx_graph, pos, alpha=0.3, arrows=True)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color, label=f"社区 {cid}")
        for cid, color in community_colors.items()
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.title("Knowledge Graph Communities", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("graphrag_communities.png", dpi=300)
    plt.show()

# 使用
visualize_communities(graph_store)
```

---

#### 6.9.6 小结

**GraphRAG社区检测的核心价值**:
- ✅ 将大规模图谱分层组织,降低查询复杂度
- ✅ 社区摘要提供高层次语义理解
- ✅ 支持全局性、综述性问题回答
- ✅ 适合处理长文档、多文档场景

**关键组件**:
- **GraphRAGExtractor**: 提取带描述的实体和关系
- **GraphRAGStore**: Leiden社区检测 + LLM摘要生成
- **hierarchical_leiden**: 层次化社区划分算法
- **max_cluster_size**: 控制社区粒度

**与传统方法的区别**:
- **SimpleLLMPathExtractor**: 仅提取三元组,无描述 → 适合基础图查询
- **GraphRAGExtractor**: 提取带描述的结构化知识 → 适合社区摘要生成

**下一步**: 6.10将介绍如何使用**GraphRAGQueryEngine**基于社区摘要回答问题。

---

### 6.10 GraphRAG查询引擎 (GraphRAGQueryEngine)

完成社区构建后,GraphRAG的最后一步是使用**GraphRAGQueryEngine**基于社区摘要回答问题。这是GraphRAG方法论的核心查询范式。

#### 6.10.1 GraphRAG查询范式

**传统RAG vs GraphRAG查询流程对比**:

```
【传统向量RAG】:
用户查询 → Embedding编码 → 向量检索Top-K文档块 → LLM生成答案

【多跳图查询 (6.6节)】:
用户查询 → 识别起始实体 → Cypher多跳遍历 → 基于路径LLM生成答案

【GraphRAG (本节)】:
用户查询 → [v1: 获取所有社区摘要 / v2: Embedding检索相关实体→定位社区]
         → 每个社区生成答案 → 聚合最终答案
```

**GraphRAG的独特优势**:

| 方法 | 检索粒度 | 适用问题类型 | 优势 | 局限 |
|------|---------|------------|------|------|
| 向量检索 | 文档块 (chunk) | 事实性查询 | 精准匹配 | 难以回答全局性问题 |
| 多跳检索 | 路径 (path) | 关系推理 | 发现隐藏关联 | 需要明确起始实体 |
| **GraphRAG** | **社区摘要 (summary)** | **全局性、综述性问题** | **理解整体结构** | 社区构建耗时 |

**典型应用场景**:

✅ **全局性问题**:
- "这篇文档的主要主题是什么?"
- "文档中讨论了哪些关键事件?"
- "总结文档的核心观点"

✅ **多维度综合分析**:
- "文档中提到的公司有哪些战略方向?"
- "不同技术之间的关联是什么?"

❌ **不适合的问题**:
- "张三的职位是什么?" → 用向量检索
- "A和B之间的关系是什么?" → 用多跳检索

---

#### 6.10.2 GraphRAGQueryEngine v1: 全局社区查询

**v1方法**: 处理**所有社区**的摘要,适合全局性问题。

**完整实现**:

```python
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
import re

class GraphRAGQueryEngine(CustomQueryEngine):
    """官方GraphRAG查询引擎 v1

    流程:
      1. 获取所有社区摘要
      2. 每个社区生成中间答案
      3. 聚合所有答案为最终响应
    """

    graph_store: GraphRAGStore
    llm: OpenAI

    def __init__(self, graph_store: GraphRAGStore, llm: OpenAI):
        self.graph_store = graph_store
        self.llm = llm

    def custom_query(self, query_str: str) -> str:
        """执行GraphRAG查询

        Args:
            query_str: 用户查询

        Returns:
            最终聚合答案
        """
        print(f"\n📝 查询: {query_str}")
        print("="*60)

        # 1. 获取所有社区摘要
        community_summaries = self.graph_store.get_community_summaries()
        num_communities = len(community_summaries)

        print(f"🔍 处理 {num_communities} 个社区的摘要...\n")

        # 2. 为每个社区生成中间答案
        community_answers = []

        for community_id, community_summary in community_summaries.items():
            print(f"  处理社区 {community_id}...")

            answer = self.generate_answer_from_summary(
                community_summary,
                query_str
            )

            community_answers.append({
                "community_id": community_id,
                "answer": answer
            })

            print(f"    ✓ 生成答案: {answer[:80]}...")

        # 3. 聚合所有社区答案
        print(f"\n🔗 聚合 {len(community_answers)} 个社区答案...")
        final_answer = self.aggregate_answers(community_answers, query_str)

        print("✅ 查询完成!")
        print("="*60)

        return final_answer

    def generate_answer_from_summary(
        self,
        community_summary: str,
        query: str
    ) -> str:
        """基于单个社区摘要生成答案

        Args:
            community_summary: 社区摘要文本
            query: 用户查询

        Returns:
            该社区的中间答案
        """
        prompt = (
            f"Given the community summary: {community_summary}\n\n"
            f"How would you answer the following query based on this information?\n"
            f"Query: {query}\n\n"
            f"If the community summary is not relevant to the query, respond with 'NOT_RELEVANT'."
        )

        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(
                role="user",
                content="I need an answer based on the above information."
            )
        ]

        response = self.llm.chat(messages)
        clean_response = re.sub(r"^assistant:\s*", "", str(response)).strip()

        return clean_response

    def aggregate_answers(
        self,
        community_answers: list[dict],
        query: str
    ) -> str:
        """聚合多个社区答案为最终响应

        Args:
            community_answers: [{"community_id": ..., "answer": ...}, ...]
            query: 用户查询

        Returns:
            最终聚合答案
        """
        # 过滤无关答案
        relevant_answers = [
            item["answer"]
            for item in community_answers
            if "NOT_RELEVANT" not in item["answer"].upper()
        ]

        if not relevant_answers:
            return "No relevant information found in the knowledge graph."

        # 拼接所有相关答案
        intermediate_answers_text = "\n\n".join([
            f"Answer {i+1}: {answer}"
            for i, answer in enumerate(relevant_answers)
        ])

        # LLM聚合
        prompt = (
            "You are given multiple intermediate answers from different communities "
            "in a knowledge graph. Your task is to combine them into a single, "
            "coherent, and concise final answer.\n\n"
            "Combine the following intermediate answers into a final response "
            "that addresses the user's query comprehensively."
        )

        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(
                role="user",
                content=(
                    f"User Query: {query}\n\n"
                    f"Intermediate Answers:\n{intermediate_answers_text}\n\n"
                    f"Final Answer:"
                )
            )
        ]

        final_response = self.llm.chat(messages)
        clean_final = re.sub(r"^assistant:\s*", "", str(final_response)).strip()

        return clean_final

    def query(self, query_str: str) -> str:
        """标准query接口 (兼容LlamaIndex)"""
        return self.custom_query(query_str)
```

**使用示例**:

```python
from llama_index.llms.openai import OpenAI

# 1. 初始化查询引擎
llm = OpenAI(model="gpt-4", temperature=0)

query_engine = GraphRAGQueryEngine(
    graph_store=graph_store,  # 已构建社区的GraphRAGStore
    llm=llm
)

# 2. 执行全局性查询
response = query_engine.query(
    "What are the main topics discussed in the documents?"
)

print(f"\n最终答案:\n{response}")
```

**输出示例**:

```
📝 查询: What are the main topics discussed in the documents?
============================================================
🔍 处理 8 个社区的摘要...

  处理社区 0...
    ✓ 生成答案: This community focuses on AI technologies including GPT-4, DALL-E, and...
  处理社区 1...
    ✓ 生成答案: This community discusses tech companies like OpenAI, Microsoft...
  ...

🔗 聚合 8 个社区答案...
✅ 查询完成!
============================================================

最终答案:
The documents primarily discuss three main topics: 1) AI technologies and their applications
(including GPT-4, DALL-E, and natural language processing), 2) Major tech companies and their
strategies in the AI sector (OpenAI, Microsoft, Google), and 3) Recent product releases and
innovations in the field of artificial intelligence.
```

---

#### 6.10.3 GraphRAGQueryEngine v2: Embedding检索 + 社区定位

**v2改进**: 不处理所有社区,而是先通过**Embedding检索相关实体**,再定位到对应社区,提高效率。

**核心改进**:

| v1 | v2 |
|----|-----|
| 处理所有社区 | 只处理相关社区 |
| 适合小规模图谱 | 适合大规模图谱 |
| 查询时间: O(N个社区) | 查询时间: O(K个相关社区) |

**v2实现** (在v1基础上扩展):

```python
from llama_index.core import PropertyGraphIndex
import re

class GraphRAGQueryEngineV2(GraphRAGQueryEngine):
    """GraphRAG查询引擎 v2 - 支持Embedding检索

    新增功能:
      - 通过Embedding检索相关实体
      - 定位实体所属社区
      - 只处理相关社区 (提高效率)
    """

    index: PropertyGraphIndex  # 新增: 需要PropertyGraphIndex进行检索
    similarity_top_k: int = 20  # 新增: 检索多少个实体

    def __init__(
        self,
        graph_store: GraphRAGStore,
        llm: OpenAI,
        index: PropertyGraphIndex,
        similarity_top_k: int = 20
    ):
        super().__init__(graph_store, llm)
        self.index = index
        self.similarity_top_k = similarity_top_k

    def custom_query(self, query_str: str) -> str:
        """执行v2查询: Embedding检索 → 定位社区 → 生成答案"""
        print(f"\n📝 查询 (v2): {query_str}")
        print("="*60)

        # 1. Embedding检索相关实体
        print(f"🔍 Embedding检索相关实体 (top_k={self.similarity_top_k})...")
        entities = self.get_entities(query_str, self.similarity_top_k)

        print(f"  ✓ 检索到 {len(entities)} 个相关实体")
        print(f"    示例: {list(entities)[:5]}")

        # 2. 将实体映射到社区ID
        print(f"\n🗺️  映射实体到社区...")
        community_ids = self.retrieve_entity_communities(entities)

        print(f"  ✓ 定位到 {len(community_ids)} 个相关社区: {community_ids}")

        # 3. 只处理相关社区
        print(f"\n🔍 处理 {len(community_ids)} 个相关社区的摘要...\n")

        community_summaries = self.graph_store.get_community_summaries()
        community_answers = []

        for community_id in community_ids:
            if community_id not in community_summaries:
                continue

            community_summary = community_summaries[community_id]
            print(f"  处理社区 {community_id}...")

            answer = self.generate_answer_from_summary(
                community_summary,
                query_str
            )

            community_answers.append({
                "community_id": community_id,
                "answer": answer
            })

            print(f"    ✓ 生成答案: {answer[:80]}...")

        # 4. 聚合答案
        print(f"\n🔗 聚合 {len(community_answers)} 个社区答案...")
        final_answer = self.aggregate_answers(community_answers, query_str)

        print("✅ 查询完成 (v2)!")
        print("="*60)

        return final_answer

    def get_entities(self, query_str: str, similarity_top_k: int) -> set:
        """通过Embedding检索相关实体

        Args:
            query_str: 用户查询
            similarity_top_k: 检索多少个节点

        Returns:
            相关实体名称集合
        """
        # 使用PropertyGraphIndex的向量检索器
        nodes_retrieved = self.index.as_retriever(
            similarity_top_k=similarity_top_k
        ).retrieve(query_str)

        entities = set()

        # 从检索结果解析实体
        # 假设节点文本格式: "entity1 -> entity2 -> relation -> description"
        pattern = r"^(\w+(?:\s+\w+)*)\s*->\s*([a-zA-Z\s]+?)\s*->\s*(\w+(?:\s+\w+)*)$"

        for node in nodes_retrieved:
            matches = re.findall(pattern, node.text, re.MULTILINE | re.IGNORECASE)

            for match in matches:
                subject = match[0].strip()
                obj = match[2].strip()
                entities.add(subject)
                entities.add(obj)

        return entities

    def retrieve_entity_communities(self, entities: set) -> list:
        """将实体映射到社区ID

        Args:
            entities: 实体名称集合

        Returns:
            社区ID列表 (去重)
        """
        entity_info = self.graph_store.entity_info  # {entity_name: community_id}

        community_ids = set()

        for entity in entities:
            if entity in entity_info:
                community_id = entity_info[entity]
                community_ids.add(community_id)

        return list(community_ids)
```

**使用示例 (v2)**:

```python
# 1. 初始化v2查询引擎
query_engine_v2 = GraphRAGQueryEngineV2(
    graph_store=graph_store,
    llm=llm,
    index=pg_index,  # 需要传入PropertyGraphIndex
    similarity_top_k=20
)

# 2. 执行查询
response = query_engine_v2.query(
    "What are the latest AI product releases mentioned?"
)

print(f"\n最终答案:\n{response}")
```

**v2输出示例**:

```
📝 查询 (v2): What are the latest AI product releases mentioned?
============================================================
🔍 Embedding检索相关实体 (top_k=20)...
  ✓ 检索到 15 个相关实体
    示例: ['GPT-4', 'DALL-E', 'Claude', 'OpenAI', 'Anthropic']

🗺️  映射实体到社区...
  ✓ 定位到 3 个相关社区: [0, 2, 5]

🔍 处理 3 个相关社区的摘要...

  处理社区 0...
    ✓ 生成答案: This community discusses GPT-4 release and its capabilities in...
  处理社区 2...
    ✓ 生成答案: This community focuses on DALL-E 3 and image generation improvements...
  处理社区 5...
    ✓ 生成答案: This community covers Claude 3 release and its performance...

🔗 聚合 3 个社区答案...
✅ 查询完成 (v2)!
============================================================

最终答案:
The documents mention several recent AI product releases: 1) GPT-4 by OpenAI with enhanced
reasoning capabilities, 2) DALL-E 3 with improved image quality and prompt following,
and 3) Claude 3 by Anthropic featuring better performance on complex tasks.
```

---

#### 6.10.4 Neo4j集成 (GraphRAG v2)

官方v2还支持**Neo4j持久化存储**,替代内存存储。

**使用Neo4jPropertyGraphStore**:

```python
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

# 1. 创建Neo4j存储
neo4j_store = Neo4jPropertyGraphStore(
    username="neo4j",
    password="your-password",
    url="bolt://localhost:7687",
    database="neo4j"
)

# 2. 扩展GraphRAGStore支持Neo4j
class GraphRAGStoreNeo4j(Neo4jPropertyGraphStore):
    """GraphRAG + Neo4j"""

    community_summary: dict = {}
    entity_info: dict = {}
    max_cluster_size: int = 5

    # 复用GraphRAGStore的build_communities方法
    build_communities = GraphRAGStore.build_communities
    _create_nx_graph = GraphRAGStore._create_nx_graph
    _collect_community_info = GraphRAGStore._collect_community_info
    _summarize_communities = GraphRAGStore._summarize_communities
    generate_community_summary = GraphRAGStore.generate_community_summary
    get_community_summaries = GraphRAGStore.get_community_summaries

# 3. 构建索引
pg_index_neo4j = PropertyGraphIndex(
    nodes=nodes,
    property_graph_store=GraphRAGStoreNeo4j(
        username="neo4j",
        password="password",
        url="bolt://localhost:7687"
    ),
    kg_extractors=[kg_extractor]
)

# 4. 构建社区
pg_index_neo4j.property_graph_store.build_communities()

# 5. 查询
query_engine_neo4j = GraphRAGQueryEngineV2(
    graph_store=pg_index_neo4j.property_graph_store,
    llm=llm,
    index=pg_index_neo4j,
    similarity_top_k=20
)

response = query_engine_neo4j.query("Your query here")
```

**Neo4j优势**:
- ✅ 持久化存储,支持大规模图谱
- ✅ 高效的图查询性能
- ✅ 支持分布式部署

---

#### 6.10.5 完整示例: GraphRAG端到端实战

**场景**: 分析技术文档,回答全局性问题

```python
from llama_index.core import PropertyGraphIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter

# ========== 配置 ==========
Settings.llm = OpenAI(model="gpt-4", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# ========== 1. 加载文档 ==========
documents = SimpleDirectoryReader("./data/tech_news").load_data()
splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
nodes = splitter.get_nodes_from_documents(documents)

# ========== 2. 构建GraphRAG图谱 ==========
kg_extractor = GraphRAGExtractor(
    llm=Settings.llm,
    extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
    parse_fn=parse_fn
)

graph_store = GraphRAGStore()

pg_index = PropertyGraphIndex(
    nodes=nodes,
    property_graph_store=graph_store,
    kg_extractors=[kg_extractor],
    show_progress=True
)

# ========== 3. 构建社区 ==========
pg_index.property_graph_store.build_communities()

# ========== 4. 创建查询引擎 ==========
# v1: 全局查询
query_engine_v1 = GraphRAGQueryEngine(
    graph_store=graph_store,
    llm=Settings.llm
)

# v2: Embedding检索
query_engine_v2 = GraphRAGQueryEngineV2(
    graph_store=graph_store,
    llm=Settings.llm,
    index=pg_index,
    similarity_top_k=20
)

# ========== 5. 执行查询 ==========
queries = [
    "What are the main topics discussed in the documents?",
    "What companies are mentioned and what are they doing?",
    "Summarize the key technological trends"
]

print("\n" + "="*70)
print("GraphRAG v1 查询结果:")
print("="*70)

for query in queries:
    response = query_engine_v1.query(query)
    print(f"\nQ: {query}")
    print(f"A: {response}\n")

print("\n" + "="*70)
print("GraphRAG v2 查询结果:")
print("="*70)

for query in queries:
    response = query_engine_v2.query(query)
    print(f"\nQ: {query}")
    print(f"A: {response}\n")
```

---

#### 6.10.6 GraphRAG vs 传统方法完整对比

**三种RAG方法的适用场景对比**:

| 查询类型 | 向量检索 (4.1) | 多跳检索 (6.6) | GraphRAG (6.9-6.10) | 推荐方法 |
|---------|--------------|---------------|-------------------|---------|
| "张三的职位是什么?" | ✅ 优秀 | ⚠️ 可以但繁琐 | ❌ 过度设计 | **向量检索** |
| "A和B之间有什么关系?" | ⚠️ 可能遗漏 | ✅ 优秀 | ⚠️ 可以但低效 | **多跳检索** |
| "文档的主要主题是什么?" | ❌ 难以概括 | ❌ 无法处理 | ✅ 优秀 | **GraphRAG** |
| "总结文档中的关键事件" | ⚠️ 可能片面 | ❌ 难以处理 | ✅ 优秀 | **GraphRAG** |
| "找到A通过多层关系影响B的证据" | ❌ 无法处理 | ✅ 优秀 | ⚠️ 可以但低效 | **多跳检索** |

**性能对比**:

| 方法 | 构建成本 | 查询速度 | 扩展性 | 准确性 (全局问题) |
|------|---------|---------|--------|-----------------|
| 向量检索 | 低 (仅embedding) | 快 | 优秀 | 中等 |
| 多跳检索 | 中 (图构建) | 中 | 良好 | N/A (不适用) |
| GraphRAG | **高 (图+社区+摘要)** | **慢 (需生成多个答案)** | 中等 | **优秀** |

**组合使用建议**:

```python
class HybridQueryEngine:
    """混合查询引擎: 根据问题类型选择方法"""

    def __init__(self, vector_engine, graph_engine, graphrag_engine):
        self.vector_engine = vector_engine
        self.graph_engine = graph_engine  # 多跳检索
        self.graphrag_engine = graphrag_engine

    def query(self, query_str: str):
        """智能路由"""

        # 使用LLM分类查询类型
        query_type = self.classify_query(query_str)

        if query_type == "factual":
            # 事实性查询 → 向量检索
            return self.vector_engine.query(query_str)

        elif query_type == "relational":
            # 关系推理 → 多跳检索
            return self.graph_engine.query(query_str)

        elif query_type == "global":
            # 全局性问题 → GraphRAG
            return self.graphrag_engine.query(query_str)

        else:
            # 默认使用向量检索
            return self.vector_engine.query(query_str)

    def classify_query(self, query_str: str) -> str:
        """使用LLM分类查询类型"""
        # 实现省略...
        pass
```

---

#### 6.10.7 小结

**GraphRAG查询引擎的核心价值**:
- ✅ 基于社区摘要的分级回答,适合全局性问题
- ✅ v1处理所有社区,v2通过Embedding定位相关社区
- ✅ 多级答案聚合,提供全面的综合性回答
- ✅ 支持Neo4j持久化,适合大规模场景

**关键组件**:
- **GraphRAGQueryEngine**: 自定义查询引擎基类
- **generate_answer_from_summary**: 基于社区摘要生成答案
- **aggregate_answers**: 聚合多个社区答案
- **get_entities (v2)**: Embedding检索相关实体
- **retrieve_entity_communities (v2)**: 实体映射到社区

**v1 vs v2**:
- **v1**: 全局查询,处理所有社区 → 适合小规模图谱
- **v2**: 精准定位,只处理相关社区 → 适合大规模图谱

**最佳实践**:
1. **小规模文档(<100页)**: 使用v1,构建简单
2. **大规模文档(>1000页)**: 使用v2 + Neo4j
3. **混合场景**: 根据查询类型路由到不同方法
4. **调优参数**:
   - `max_cluster_size`: 控制社区大小 (3-10)
   - `similarity_top_k`: 控制检索实体数 (10-50)

**完整知识图谱RAG技术栈**:

```
第6章知识图谱RAG完整技术栈:

【基础构建】
├─ 6.1 PropertyGraphIndex      → 图索引基础
├─ 6.2 KG Extractors           → 实体关系提取
├─ 6.3 图检索器                 → 基础检索
├─ 6.4 查询引擎                 → 标准查询
└─ 6.5 组合检索器               → 多检索器融合

【高级查询】
├─ 6.6 多跳检索与路径推理       → 关系推理 (我们独有优势)
└─ 6.7 实体消歧与冲突解决       → 数据清洗 (我们独有优势)

【可视化】
└─ 6.8 可视化知识图谱          → 调试与展示

【官方GraphRAG】
├─ 6.9 GraphRAG社区检测        → Leiden算法 + 社区摘要
└─ 6.10 GraphRAG查询引擎       → 社区查询 + 答案聚合
```

**至此,第6章知识图谱RAG全部完成! 我们实现了:**
- ✅ 官方GraphRAG核心方法论 (100%覆盖)
- ✅ 多跳检索与实体消歧 (我们的独有优势)
- ✅ 市面最全面的知识图谱RAG教程

---

## 第 7 章：Agent 与 RAG 结合

将智能体（Agent）与 RAG 系统结合，实现复杂的推理和工具使用。

### 7.1 ReAct Agent

ReAct（Reasoning + Acting）模式结合推理和行动：

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata

# 1. 创建查询工具
query_tool = QueryEngineTool(
    query_engine=vector_index.as_query_engine(similarity_top_k=5),
    metadata=ToolMetadata(
        name="knowledge_base",
        description=(
            "Provides comprehensive information about AI, machine learning, "
            "and deep learning from technical documentation"
        )
    )
)

# 2. 创建 Python REPL 工具（示例）
from llama_index.core.tools import FunctionTool

def calculator(expression: str) -> str:
    """Execute a Python mathematical expression"""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

calc_tool = FunctionTool.from_defaults(
    fn=calculator,
    name="calculator",
    description="Execute mathematical expressions"
)

# 3. 创建 ReAct Agent
agent = ReActAgent.from_tools(
    tools=[query_tool, calc_tool],
    llm=Settings.llm,
    verbose=True,
    max_iterations=10
)

# 4. 执行复杂任务
response = agent.chat(
    "First, find information about neural network layers from the knowledge base. "
    "Then calculate how many parameters a 3-layer network would have if each "
    "layer has 128, 64, and 32 neurons respectively (including biases)."
)

print(response)
```

**工作流程**：
1. **思考**（Thought）：分析问题
2. **行动**（Action）：选择并执行工具
3. **观察**（Observation）：查看工具结果
4. 重复直到得出最终答案

### 7.2 FunctionAgent

使用函数调用 API 的智能体：

```python
from llama_index.core.agent import FunctionCallingAgent

# 创建 Function Calling Agent
function_agent = FunctionCallingAgent.from_tools(
    tools=[query_tool, calc_tool],
    llm=Settings.llm,
    verbose=True
)

# 执行任务
response = function_agent.chat(
    "Compare the performance metrics mentioned in the knowledge base "
    "and calculate the average improvement percentage"
)
```

### 7.3 多工具 RAG Agent

创建具有多个专门工具的智能体：

```python
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import ReActAgent

# 1. 创建多个专门的查询引擎
ml_index = VectorStoreIndex.from_documents(ml_documents)
dl_index = VectorStoreIndex.from_documents(dl_documents)
nlp_index = VectorStoreIndex.from_documents(nlp_documents)

# 2. 创建工具列表
tools = [
    QueryEngineTool(
        query_engine=ml_index.as_query_engine(),
        metadata=ToolMetadata(
            name="machine_learning_kb",
            description="Expert knowledge about traditional ML algorithms"
        )
    ),
    QueryEngineTool(
        query_engine=dl_index.as_query_engine(),
        metadata=ToolMetadata(
            name="deep_learning_kb",
            description="Expert knowledge about neural networks and deep learning"
        )
    ),
    QueryEngineTool(
        query_engine=nlp_index.as_query_engine(),
        metadata=ToolMetadata(
            name="nlp_kb",
            description="Expert knowledge about natural language processing"
        )
    )
]

# 3. 创建多工具 Agent
multi_tool_agent = ReActAgent.from_tools(
    tools=tools,
    llm=Settings.llm,
    verbose=True
)

# 4. 执行跨领域查询
response = multi_tool_agent.chat(
    "Compare traditional machine learning approaches with deep learning "
    "for NLP tasks. Use all available knowledge bases."
)
```

### 7.4 带记忆的 Agent

为 Agent 添加对话记忆：

```python
from llama_index.core.memory import ChatMemoryBuffer

# 创建记忆
chat_memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

# 创建带记忆的 Agent
agent_with_memory = ReActAgent.from_tools(
    tools=[query_tool],
    llm=Settings.llm,
    memory=chat_memory,
    verbose=True
)

# 多轮对话
response1 = agent_with_memory.chat(
    "Find information about transformers from the knowledge base"
)

response2 = agent_with_memory.chat(
    "Based on what you just found, explain how attention mechanism works"
)

response3 = agent_with_memory.chat(
    "Summarize our discussion"  # Agent 记得之前的对话
)
```

### 7.5 SubQuestionQueryEngine 作为 Agent 工具

将 SubQuestionQueryEngine 包装为 Agent 工具：

```python
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool

# 1. 创建 SubQuestionQueryEngine
query_engine_tools = [
    QueryEngineTool(
        query_engine=ml_index.as_query_engine(),
        metadata=ToolMetadata(name="ml", description="ML knowledge")
    ),
    QueryEngineTool(
        query_engine=dl_index.as_query_engine(),
        metadata=ToolMetadata(name="dl", description="DL knowledge")
    )
]

sub_question_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    use_async=True
)

# 2. 包装为 Agent 工具
complex_qa_tool = QueryEngineTool(
    query_engine=sub_question_engine,
    metadata=ToolMetadata(
        name="complex_qa",
        description=(
            "Use this tool for complex questions that require "
            "breaking down into sub-questions and analyzing "
            "multiple knowledge sources"
        )
    )
)

# 3. 创建 Agent
agent = ReActAgent.from_tools(
    tools=[complex_qa_tool, calc_tool],
    llm=Settings.llm,
    verbose=True
)

response = agent.chat(
    "Compare ML and DL approaches comprehensively, "
    "then calculate which one has more research papers mentioned"
)
```

### 7.6 自定义工具

创建自定义工具供 Agent 使用：

```python
from llama_index.core.tools import FunctionTool
import requests

def search_arxiv(query: str, max_results: int = 5) -> str:
    """Search arXiv for academic papers"""
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        # 简化处理（实际应解析 XML）
        return f"Found {max_results} papers related to '{query}'"
    else:
        return "Search failed"

# 包装为工具
arxiv_tool = FunctionTool.from_defaults(
    fn=search_arxiv,
    name="arxiv_search",
    description="Search academic papers on arXiv"
)

# 创建综合 Agent
comprehensive_agent = ReActAgent.from_tools(
    tools=[query_tool, arxiv_tool, calc_tool],
    llm=Settings.llm,
    verbose=True
)

response = comprehensive_agent.chat(
    "Search for recent papers on transformers, then cross-reference "
    "with information in our knowledge base to identify research gaps"
)
```

---

## 总结与最佳实践

### 核心要点

1. **混合检索**：结合 BM25 和向量检索，平衡精确匹配和语义相关性
2. **查询优化**：使用 HyDE、查询分解等技术提升检索质量
3. **智能路由**：根据查询类型选择最合适的处理策略
4. **重排序**：使用 Cohere 或跨编码器模型精炼检索结果
5. **对话系统**：选择合适的 Chat Engine 模式（Simple、CondensePlusContext、ReAct）
6. **知识图谱**：显式建模实体关系，支持复杂推理
7. **Agent 集成**：结合工具使用和推理能力，处理复杂任务

### 性能优化建议

```python
# 1. 使用异步加速
retriever = QueryFusionRetriever(
    retrievers=[...],
    use_async=True  # 并行检索
)

# 2. 控制检索数量
query_engine = index.as_query_engine(
    similarity_top_k=5,  # 初始检索数量
    node_postprocessors=[
        CohereRerank(top_n=3)  # 重排后最终数量
    ]
)

# 3. 响应模式选择
# compact: 最快，适合短文档
# refine: 质量高，速度慢
# tree_summarize: 平衡质量和速度
query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    response_mode="compact"
)

# 4. 设置并行度
extractor = SimpleLLMPathExtractor(
    num_workers=4  # 并行处理节点
)
```

### 选型指南

| 场景 | 推荐方案 |
|------|---------|
| 简单问答 | Vector Retriever + Basic Query Engine |
| 复杂问答 | Hybrid Retrieval + SubQuestionQueryEngine |
| 多领域查询 | RouterQueryEngine |
| 对话系统 | CondensePlusContextChatEngine |
| 需要推理 | ReAct Agent + Multiple Tools |
| 关系查询 | PropertyGraphIndex + TextToCypher |
| 高精度检索 | Hybrid Retrieval + Cohere Rerank |

### 实战示例：完整 RAG 系统

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.query_engine import RetrieverQueryEngine

# 1. 配置
Settings.llm = OpenAI(model="gpt-4", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# 2. 加载数据
documents = SimpleDirectoryReader("./data").load_data()

# 3. 创建索引
vector_index = VectorStoreIndex.from_documents(documents)

# 4. 创建混合检索器
hybrid_retriever = QueryFusionRetriever(
    retrievers=[
        vector_index.as_retriever(similarity_top_k=10),
        BM25Retriever.from_defaults(
            docstore=vector_index.docstore,
            similarity_top_k=10
        )
    ],
    num_queries=1,
    use_async=True
)

# 5. 添加重排序
cohere_rerank = CohereRerank(
    api_key="your-cohere-key",
    top_n=5
)

# 6. 创建查询引擎
query_engine = RetrieverQueryEngine.from_args(
    retriever=hybrid_retriever,
    node_postprocessors=[cohere_rerank],
    response_mode="compact"
)

# 7. 执行查询
response = query_engine.query(
    "What are the latest advancements in transformer architectures?"
)

print(response)
```

### 下一步学习

1. **LlamaIndex 官方文档**: https://developers.llamaindex.ai
2. **进阶主题**：
   - 自定义 Retriever 和 Query Engine
   - 分布式 RAG 系统
   - RAG 评估指标（Faithfulness, Relevance）
   - 生产环境部署（FastAPI、缓存、监控）

---

**参考资源**：
- LlamaIndex 官方文档: https://developers.llamaindex.ai/python/framework/
- Query Engine Guide: https://developers.llamaindex.ai/python/framework/module_guides/deploying/query_engine/
- Retriever Guide: https://developers.llamaindex.ai/python/framework/module_guides/querying/retriever/
- Property Graph Index: https://developers.llamaindex.ai/python/framework/module_guides/indexing/lpg_index_guide/
