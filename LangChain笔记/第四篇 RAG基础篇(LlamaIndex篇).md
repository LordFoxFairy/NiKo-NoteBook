# 第四篇 RAG基础篇 (LlamaIndex)

## 前置准备

### 环境配置

```bash
# 核心依赖
pip install llama-index>=0.11.0
pip install llama-index-core>=0.11.0
pip install llama-index-llms-openai>=0.2.0
pip install llama-index-embeddings-openai>=0.2.0

# 向量数据库集成（可选）
pip install llama-index-vector-stores-chroma
pip install chromadb>=0.5.0

# 其他依赖
pip install pypdf  # PDF支持
pip install python-dotenv  # 环境变量管理
```

### 环境变量设置

```bash
# .env 文件
OPENAI_API_KEY=sk-your-api-key-here
```

### 准备测试数据

```bash
# 创建数据目录
mkdir -p ./data

# 创建示例文档
echo "LlamaIndex 是一个数据框架，专为 RAG（检索增强生成）应用设计。它提供了简单的接口来加载、索引和查询数据。" > ./data/intro.txt
```

---

## 第 1 章：为什么选择 LlamaIndex？

### 1.1 LlamaIndex vs LangChain：设计哲学对比

#### 核心定位差异

| 维度 | LlamaIndex | LangChain |
|------|-----------|-----------|
| **核心定位** | 数据优先框架（Data Framework） | 编排优先框架（Orchestration Framework） |
| **主要用途** | RAG、文档问答、知识库 | Agent、复杂链式调用、工作流 |
| **抽象层级** | 高层抽象（开箱即用） | 低层抽象（灵活组合） |
| **学习曲线** | 平缓（5行代码启动） | 陡峭（需理解LCEL、Runnable） |
| **索引能力** | 强（多种索引类型） | 弱（需自行实现） |
| **数据连接** | 丰富（100+ Loaders） | 基础（需集成） |
| **最佳场景** | RAG、搜索、文档分析 | Agent、复杂工作流、多步推理 |

#### 设计哲学

**LlamaIndex 的核心理念**：

1. **数据优先（Data-First）**
   - 一切从数据开始
   - 内置丰富的数据连接器
   - 支持结构化和非结构化数据

2. **索引即查询（Index as Interface）**
   - 多种索引类型适应不同场景
   - 索引自动优化查询策略
   - 查询引擎开箱即用

3. **模块化设计（Modular Architecture）**
   ```
   Reader → Parser → Index → Retriever → Query Engine
   ```

4. **LLM无关（LLM-Agnostic）**
   - 支持OpenAI、Anthropic、本地模型
   - 统一的接口切换模型

**LangChain 的核心理念**：

1. **编排优先（Orchestration-First）**
   - 灵活的链式调用
   - LCEL（LangChain Expression Language）
   - 强大的 Agent 能力

2. **低级控制（Low-Level Control）**
   - 手动控制每个环节
   - 自定义程度高
   - 适合复杂场景

#### 何时选择 LlamaIndex？

选择 LlamaIndex 如果你需要：
- 快速搭建 RAG 系统
- 开箱即用的文档问答
- 多种索引策略（向量、关键词、摘要等）
- 丰富的数据源连接（PDF、数据库、API等）
- 生产级的检索优化

选择 LangChain 如果你需要：
- 复杂的 Agent 系统
- 多步推理工作流
- 精细的 Prompt 控制
- 自定义的执行链
- 与其他工具的深度集成

#### 最佳实践：两者结合

```python
# LlamaIndex 作为 LangChain 的工具
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from langchain_core.tools import tool

# 1. 使用 LlamaIndex 构建索引（数据优先）
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# 2. 封装为 LangChain 工具（编排优先）
@tool
def search_documents(query: str) -> str:
    """搜索文档库，回答关于文档的问题。"""
    response = query_engine.query(query)
    return str(response)

# 3. 在 LangChain Agent 中使用
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4"),
    tools=[search_documents]
)

# 4. 运行
result = agent.invoke({
    "messages": [("user", "文档中提到了哪些关键概念？")]
})
print(result["messages"][-1].content)
```

**组合优势**：
- LlamaIndex 处理数据和检索（强项）
- LangChain 处理复杂逻辑和编排（强项）
- 发挥各自优势，构建更强大的系统

---

### 1.2 快速开始：5行代码实现RAG

#### 最简单的RAG应用

```python
"""
5行代码实现完整RAG - LlamaIndex的强大之处
"""
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import os

# 设置API Key
os.environ["OPENAI_API_KEY"] = "sk-your-key"

# 1. 加载文档
documents = SimpleDirectoryReader("./data").load_data()

# 2. 创建索引
index = VectorStoreIndex.from_documents(documents)

# 3. 创建查询引擎
query_engine = index.as_query_engine()

# 4. 查询
response = query_engine.query("文档的主要内容是什么？")

# 5. 输出
print(response)
```

**就这么简单！** LlamaIndex已经自动完成了：
- 文档分块
- 向量化（Embedding）
- 向量存储
- 检索
- LLM生成答案

#### 查看详细信息

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# 加载文档
documents = SimpleDirectoryReader("./data").load_data()

print(f"加载了 {len(documents)} 个文档")
for i, doc in enumerate(documents[:2], 1):
    print(f"\n文档 {i}:")
    print(f"  内容: {doc.text[:200]}...")
    print(f"  元数据: {doc.metadata}")

# 创建索引
index = VectorStoreIndex.from_documents(documents)

# 查询（带来源）
query_engine = index.as_query_engine(
    similarity_top_k=3,  # 返回Top-3最相关文档
    response_mode="compact"
)

response = query_engine.query("什么是LlamaIndex？")

print(f"\n回答:\n{response}\n")
print("来源:")
for i, node in enumerate(response.source_nodes, 1):
    print(f"{i}. {node.text[:100]}... (得分: {node.score:.4f})")
```

---

### 1.3 核心组件详解

#### Document（文档）

Document是LlamaIndex的基本数据单元：

```python
from llama_index.core import Document

# 手动创建文档
doc1 = Document(
    text="这是文档内容",
    metadata={
        "source": "manual",
        "author": "张三",
        "date": "2026-01-19"
    }
)

# 查看文档属性
print(f"文档ID: {doc1.doc_id}")
print(f"内容: {doc1.text}")
print(f"元数据: {doc1.metadata}")

# 批量创建
documents = [
    Document(text="文档1内容", metadata={"id": 1}),
    Document(text="文档2内容", metadata={"id": 2}),
    Document(text="文档3内容", metadata={"id": 3})
]
```

#### Node（节点）

Node是文档分块后的单元：

```python
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document

# 创建文档
doc = Document(text="很长的文本内容..." * 100)

# 创建分块器
parser = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50
)

# 分块
nodes = parser.get_nodes_from_documents([doc])

print(f"分割成 {len(nodes)} 个节点")
for i, node in enumerate(nodes[:3], 1):
    print(f"\n节点 {i}:")
    print(f"  内容: {node.text[:100]}...")
    print(f"  长度: {len(node.text)}")
```

#### Index（索引）

索引是LlamaIndex的核心：

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# 加载文档
documents = SimpleDirectoryReader("./data").load_data()

# 创建向量索引
index = VectorStoreIndex.from_documents(documents)

# 持久化索引
index.storage_context.persist(persist_dir="./storage")

# 从磁盘加载索引
from llama_index.core import StorageContext, load_index_from_storage

storage_context = StorageContext.from_defaults(persist_dir="./storage")
loaded_index = load_index_from_storage(storage_context)
```

---

## 第 2 章：数据摄取（Data Ingestion）

### 2.1 文档加载器（Document Loaders）

#### SimpleDirectoryReader（最常用）

```python
from llama_index.core import SimpleDirectoryReader

# 基础用法：加载目录下所有支持的文件
documents = SimpleDirectoryReader("./data").load_data()

# 指定文件类型
documents = SimpleDirectoryReader(
    "./data",
    required_exts=[".pdf", ".txt", ".md"]
).load_data()

# 递归加载子目录
documents = SimpleDirectoryReader(
    "./data",
    recursive=True
).load_data()

# 排除某些文件
documents = SimpleDirectoryReader(
    "./data",
    exclude=["temp.txt", "*.log"]
).load_data()

# 自定义元数据
documents = SimpleDirectoryReader(
    "./data",
    file_metadata=lambda filename: {
        "source": filename,
        "category": "docs"
    }
).load_data()

# 并行加载（提升性能）
documents = SimpleDirectoryReader("./data").load_data(num_workers=4)

# 迭代加载（处理大量文件）
reader = SimpleDirectoryReader("./data", recursive=True)
all_docs = []
for docs in reader.iter_data():
    # 处理每个文件
    all_docs.extend(docs)
```

**支持的文件格式**：
- 文本：`.txt`, `.md`, `.csv`
- 文档：`.pdf`, `.docx`, `.pptx`, `.epub`
- 代码：`.py`, `.js`, `.java`, `.cpp`
- 网页：`.html`, `.htm`
- 数据：`.json`, `.xml`
- 媒体：`.mp3`, `.mp4`
- 图片：`.jpg`, `.png`

#### 远程文件系统支持

```python
from s3fs import S3FileSystem
from llama_index.core import SimpleDirectoryReader

# 连接 S3
s3_fs = S3FileSystem(key="...", secret="...")

# 加载 S3 上的文档
reader = SimpleDirectoryReader(
    input_dir="my-bucket/documents",
    fs=s3_fs,
    recursive=True
)

documents = reader.load_data()
```

---

### 2.2 节点解析器（Node Parser）

#### 为什么需要分块？

**分块的好处**：
- 适应模型上下文窗口
- 提高检索精确度
- 降低成本（只处理相关片段）
- 保持语义完整性

#### SentenceSplitter - 智能句子分割

```python
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document

# 创建分割器
splitter = SentenceSplitter(
    chunk_size=512,          # 每块大小（字符数）
    chunk_overlap=50,        # 块之间重叠（保持上下文）
    separator=" "            # 分隔符
)

# 分割文档
doc = Document(text="很长的文本内容...")
nodes = splitter.get_nodes_from_documents([doc])

# 查看结果
for i, node in enumerate(nodes[:3], 1):
    print(f"\n节点 {i}:")
    print(f"  内容: {node.text[:100]}...")
    print(f"  长度: {len(node.text)}")
    print(f"  元数据: {node.metadata}")
```

#### SemanticSplitter - 语义分块

```python
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding

# 语义分块器（根据语义相似度分块）
semantic_splitter = SemanticSplitterNodeParser(
    buffer_size=1,                      # 缓冲区大小
    breakpoint_percentile_threshold=95, # 语义断点阈值
    embed_model=OpenAIEmbedding()       # 使用的embedding模型
)

# 分割
nodes = semantic_splitter.get_nodes_from_documents(documents)

print(f"语义分块创建了 {len(nodes)} 个节点")
```

**语义分块的优势**：
- 保持语义完整性
- 自适应块大小
- 更好的检索效果

**何时使用语义分块**：
- 长文档（> 5000字）
- 复杂结构（学术论文、技术文档）
- 高质量要求（生产环境）

#### 分块策略对比

| 策略 | 优势 | 劣势 | 适用场景 |
|------|------|------|---------|
| **SentenceSplitter** | 快速、简单 | 可能切断语义 | 通用文档、快速原型 |
| **SemanticSplitter** | 语义完整性最佳 | 计算开销大 | 学术论文、技术文档 |

---

### 2.3 摄取管道（Ingestion Pipeline）

#### 什么是 Ingestion Pipeline？

Ingestion Pipeline 是 LlamaIndex 中用于构建数据处理流水线的高级抽象：

```
Documents → Transformation 1 (分块) → Transformation 2 (元数据提取) → Transformation 3 (Embedding) → Nodes → Vector Store
```

#### 基础用法

```python
from llama_index.core import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline

# 创建管道
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=512, chunk_overlap=50),
        TitleExtractor(),
        OpenAIEmbedding(),
    ]
)

# 运行管道
nodes = pipeline.run(documents=documents)
```

#### 带缓存的管道（生产推荐）

```python
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.storage.docstore import SimpleDocumentStore

# 创建缓存
cache = IngestionCache(
    cache=SimpleDocumentStore(),
)

# 创建带缓存的管道
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=512, chunk_overlap=50),
        OpenAIEmbedding(),
    ],
    cache=cache
)

# 第一次运行（会执行所有转换）
nodes = pipeline.run(documents=documents)

# 第二次运行相同文档（会使用缓存）
nodes = pipeline.run(documents=documents)  # 快速返回
```

#### 持久化缓存

```python
# 保存管道状态
pipeline.persist("./pipeline_storage")

# 加载并恢复状态
new_pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=512, chunk_overlap=50),
        OpenAIEmbedding(),
    ],
)
new_pipeline.load("./pipeline_storage")

# 将立即使用缓存
nodes = new_pipeline.run(documents=documents)
```

---

## 第 3 章：索引（Indexing）

### 3.1 VectorStoreIndex - 向量索引（最常用）

#### 基础用法

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# 加载文档
documents = SimpleDirectoryReader("./data").load_data()

# 创建向量索引
index = VectorStoreIndex.from_documents(documents)

# 查询
query_engine = index.as_query_engine(
    similarity_top_k=3  # 返回最相似的3个节点
)
response = query_engine.query("什么是RAG?")
print(response)
```

**工作原理**：
```
文档 → Embedding → 向量存储
查询 → Embedding → 向量相似度搜索 → Top-K节点 → LLM生成答案
```

**适用场景**：
- 语义搜索
- 问答系统
- 文档检索

#### 使用 Ingestion Pipeline 创建索引

```python
from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline

# 创建管道
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=512, chunk_overlap=50),
        TitleExtractor(),
        OpenAIEmbedding(),
    ]
)

# 运行管道
nodes = pipeline.run(documents=documents)

# 从节点创建索引
index = VectorStoreIndex(nodes)
```

#### 直接管理节点

```python
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex

# 手动创建节点
node1 = TextNode(text="第一段内容", id_="node1")
node2 = TextNode(text="第二段内容", id_="node2")
nodes = [node1, node2]

# 从节点创建索引
index = VectorStoreIndex(nodes)

# 插入新节点
new_node = TextNode(text="新增内容", id_="node3")
index.insert_nodes([new_node])

# 删除节点
index.delete_nodes(["node1"])

# 更新节点
updated_node = TextNode(text="更新后的内容", id_="node2")
index.update_ref_doc(updated_node)
```

---

### 3.2 其他索引类型

#### SummaryIndex - 摘要索引

```python
from llama_index.core import SummaryIndex

# 创建摘要索引
summary_index = SummaryIndex.from_documents(documents)

# 查询（会遍历所有文档）
query_engine = summary_index.as_query_engine()
response = query_engine.query("总结所有文档的要点")
print(response)
```

**特点**：
- 遍历所有节点
- 适合摘要类任务
- 计算成本高

**适用场景**：
- 文档摘要
- 全面分析
- 小数据集

#### TreeIndex - 树形索引

```python
from llama_index.core import TreeIndex

# 创建树形索引
tree_index = TreeIndex.from_documents(documents)

# 查询
query_engine = tree_index.as_query_engine()
response = query_engine.query("分层次总结文档")
print(response)
```

**特点**：
- 层次化结构
- 自底向上摘要
- 适合大文档

#### KeywordTableIndex - 关键词索引

```python
from llama_index.core import KeywordTableIndex

# 创建关键词索引
keyword_index = KeywordTableIndex.from_documents(documents)

# 查询
query_engine = keyword_index.as_query_engine()
response = query_engine.query("Python编程")
print(response)
```

**特点**：
- 基于关键词匹配
- 速度快
- 精确匹配

**适用场景**：
- 精确关键词搜索
- 结构化文档
- 代码搜索

---

### 3.3 持久化（Persisting）

#### 保存索引到磁盘

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# 创建索引
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

# 持久化到磁盘
index.storage_context.persist(persist_dir="./storage")
```

#### 从磁盘加载索引

```python
from llama_index.core import StorageContext, load_index_from_storage

# 加载存储上下文
storage_context = StorageContext.from_defaults(persist_dir="./storage")

# 加载索引
index = load_index_from_storage(storage_context)

# 如果有多个索引，需要指定 index_id
index = load_index_from_storage(storage_context, index_id="my_index")
```

#### 使用远程存储（S3）

```python
import s3fs
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage

# 设置 S3 文件系统
s3 = s3fs.S3FileSystem(
    key="AWS_ACCESS_KEY_ID",
    secret="AWS_SECRET_ACCESS_KEY",
    endpoint_url="https://s3.amazonaws.com"
)

# 保存到 S3
index.set_index_id("vector_index")
s3_bucket_name = "my-bucket/storage"
index.storage_context.persist(persist_dir=s3_bucket_name, fs=s3)

# 从 S3 加载
index_from_s3 = load_index_from_storage(
    StorageContext.from_defaults(persist_dir=s3_bucket_name, fs=s3),
    index_id="vector_index"
)
```

---

### 3.4 向量数据库集成

#### 内置向量存储

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./data").load_data()

# 默认使用内存存储（SimpleVectorStore）
index = VectorStoreIndex.from_documents(documents)

# 持久化到磁盘
index.storage_context.persist(persist_dir="./storage")
```

#### 集成Chroma向量数据库

```python
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader
import chromadb

# 初始化Chroma客户端
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("my_collection")

# 创建向量存储
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 加载文档并构建索引
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

# 查询
query_engine = index.as_query_engine()
response = query_engine.query("什么是LlamaIndex?")
print(response)
```

#### 向量数据库选择指南

| 数据库 | 类型 | 性能 | 部署难度 | 适用场景 |
|--------|------|------|---------|---------|
| **SimpleVectorStore** | 内存 | 低 | ⭐ | 开发测试、小数据集 |
| **Chroma** | 嵌入式 | 中 | ⭐⭐ | 中小型应用、快速开发 |
| **Pinecone** | 云服务 | 高 | ⭐ | 云原生、无需运维 |
| **Qdrant** | 服务 | 高 | ⭐⭐⭐ | 生产环境、分布式 |
| **Weaviate** | 服务 | 高 | ⭐⭐⭐ | 企业级、GraphRAG |

---

## 第 4 章：查询（Querying）

### 4.1 查询引擎（Query Engine）

#### 基础查询引擎

```python
from llama_index.core import VectorStoreIndex

# 创建索引
index = VectorStoreIndex.from_documents(documents)

# 创建查询引擎
query_engine = index.as_query_engine(
    similarity_top_k=3,           # Top-K检索
    response_mode="compact",      # 响应模式
    verbose=True                  # 显示详细日志
)

# 查询
response = query_engine.query("什么是LlamaIndex?")
print(response)

# 查看来源
print("\n来源节点:")
for node in response.source_nodes:
    print(f"- {node.text[:100]}...")
    print(f"  得分: {node.score:.4f}")
```

#### 响应模式（Response Mode）

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| **refine** | 逐个节点精炼答案（默认） | 高质量答案 |
| **compact** | 合并节点后一次生成 | 平衡质量和速度 |
| **tree_summarize** | 树形汇总 | 大量文档 |
| **simple_summarize** | 简单合并 | 快速摘要 |
| **no_text** | 只返回节点，不生成 | 检索测试 |
| **accumulate** | 对每个节点分别查询 | 需要多个答案 |
| **compact_accumulate** | compact + accumulate | 平衡质量和多答案 |

```python
# 不同响应模式对比
query_engine_refine = index.as_query_engine(response_mode="refine")
query_engine_compact = index.as_query_engine(response_mode="compact")

query = "什么是RAG?"
response1 = query_engine_refine.query(query)  # 更高质量
response2 = query_engine_compact.query(query)  # 更快速度
```

#### 流式输出

```python
# 启用流式输出
query_engine = index.as_query_engine(streaming=True)

response = query_engine.query("详细解释LlamaIndex的工作原理")

# 流式打印
print("回答: ", end="")
for text in response.response_gen:
    print(text, end="", flush=True)
print()
```

#### 自定义Prompt

```python
from llama_index.core import PromptTemplate

# 自定义QA模板
qa_prompt_tmpl = PromptTemplate(
    "上下文信息如下：\n"
    "{context_str}\n"
    "根据上下文信息（不要使用先验知识），回答以下问题：\n"
    "{query_str}\n"
    "答案："
)

# 应用自定义Prompt
query_engine = index.as_query_engine(
    text_qa_template=qa_prompt_tmpl
)

response = query_engine.query("什么是向量索引？")
print(response)
```

---

### 4.2 检索器（Retriever）

#### 基础检索器

```python
from llama_index.core import VectorStoreIndex

# 创建索引
index = VectorStoreIndex.from_documents(documents)

# 创建检索器
retriever = index.as_retriever(
    similarity_top_k=5,  # 返回Top-5
)

# 检索
nodes = retriever.retrieve("什么是向量索引?")

for i, node in enumerate(nodes, 1):
    print(f"\n节点 {i} (得分: {node.score:.4f}):")
    print(node.text[:200])
```

#### 自定义检索器

```python
from llama_index.core.retrievers import VectorIndexRetriever

# 向量检索器
vector_retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=3
)

# 检索
nodes = vector_retriever.retrieve("查询文本")
for node in nodes:
    print(f"- {node.text[:100]}... (得分: {node.score:.4f})")
```

---

### 4.3 聊天引擎（Chat Engine）

#### 基础聊天引擎

```python
from llama_index.core import VectorStoreIndex

# 创建索引
index = VectorStoreIndex.from_documents(documents)

# 创建聊天引擎
chat_engine = index.as_chat_engine()

# 多轮对话
response1 = chat_engine.chat("什么是LlamaIndex?")
print(response1)

response2 = chat_engine.chat("它有哪些主要功能？")
print(response2)

response3 = chat_engine.chat("能详细说说第一个功能吗？")
print(response3)
```

#### 流式聊天

```python
# 创建流式聊天引擎
chat_engine = index.as_chat_engine()

# 流式对话
streaming_response = chat_engine.stream_chat("告诉我关于RAG的知识")
for token in streaming_response.response_gen:
    print(token, end="", flush=True)
print()
```

#### 查看对话历史

```python
# 创建聊天引擎
chat_engine = index.as_chat_engine()

# 多轮对话
chat_engine.chat("什么是LlamaIndex?")
chat_engine.chat("它有哪些功能？")

# 查看对话历史
print(chat_engine.chat_history)
```

---

## 第 5 章：低级组件（Low-Level Components）

### 5.1 为什么需要低级组件？

高级接口（`as_query_engine`、`as_chat_engine`）虽然方便，但有时需要更精细的控制：

- 自定义检索逻辑
- 复杂的后处理
- 特殊的响应合成策略
- 深度集成到现有系统

```
Query → Retriever(检索) → NodePostprocessor(后处理) → ResponseSynthesizer(响应合成) → Response
```

### 5.2 使用 Retriever + ResponseSynthesizer

#### 基础用法

```python
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# 创建索引
index = VectorStoreIndex.from_documents(documents)

# 1. 创建检索器
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=3,
)

# 2. 创建响应合成器
response_synthesizer = get_response_synthesizer(
    response_mode="compact"
)

# 3. 组合成查询引擎
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)

# 4. 查询
response = query_engine.query("什么是LlamaIndex?")
print(response)
```

#### 自定义响应合成器

```python
from llama_index.core import get_response_synthesizer, PromptTemplate

# 自定义 Prompt
qa_prompt = PromptTemplate(
    "上下文:\n{context_str}\n\n"
    "问题: {query_str}\n\n"
    "请用简洁的语言回答（不超过100字）:\n"
)

# 创建响应合成器
response_synthesizer = get_response_synthesizer(
    response_mode="compact",
    text_qa_template=qa_prompt,
    streaming=True
)

# 组合查询引擎
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)

response = query_engine.query("什么是RAG?")
for text in response.response_gen:
    print(text, end="")
```

#### 添加节点后处理器

```python
from llama_index.core.postprocessor import SimilarityPostprocessor

# 创建后处理器（过滤低分节点）
postprocessor = SimilarityPostprocessor(similarity_cutoff=0.7)

# 组合查询引擎
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[postprocessor]
)

response = query_engine.query("什么是LlamaIndex?")
print(response)
```

### 5.3 手动控制检索和生成

#### 分步执行

```python
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever

# 创建索引和检索器
index = VectorStoreIndex.from_documents(documents)
retriever = VectorIndexRetriever(index=index, similarity_top_k=3)

# 1. 手动检索
query_str = "什么是LlamaIndex?"
nodes = retriever.retrieve(query_str)

print(f"检索到 {len(nodes)} 个节点:")
for i, node in enumerate(nodes, 1):
    print(f"{i}. (得分: {node.score:.4f}) {node.text[:100]}...")

# 2. 手动合成响应
response_synthesizer = get_response_synthesizer(response_mode="compact")
response = response_synthesizer.synthesize(query_str, nodes=nodes)

print(f"\n最终回答:\n{response}")
```

#### 自定义后处理逻辑

```python
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever

# 创建索引和检索器
index = VectorStoreIndex.from_documents(documents)
retriever = VectorIndexRetriever(index=index, similarity_top_k=5)

# 1. 检索
query_str = "什么是LlamaIndex?"
nodes = retriever.retrieve(query_str)

# 2. 自定义后处理：过滤和排序
def custom_postprocess(nodes):
    # 过滤低分节点
    filtered = [n for n in nodes if n.score > 0.7]

    # 去重（基于文本相似度）
    unique_nodes = []
    for node in filtered:
        if not any(node.text == n.text for n in unique_nodes):
            unique_nodes.append(node)

    # 重新排序（可以基于自定义逻辑）
    sorted_nodes = sorted(unique_nodes, key=lambda n: n.score, reverse=True)

    return sorted_nodes[:3]  # 只保留Top-3

processed_nodes = custom_postprocess(nodes)

print(f"后处理后剩余 {len(processed_nodes)} 个节点")

# 3. 合成响应
response_synthesizer = get_response_synthesizer()
response = response_synthesizer.synthesize(query_str, nodes=processed_nodes)

print(f"\n回答:\n{response}")
```

---

## 第 6 章：全局配置（Settings）

### 6.1 配置LLM和Embedding

#### 使用OpenAI

```python
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# 配置LLM
Settings.llm = OpenAI(
    model="gpt-4-turbo-preview",
    temperature=0.1,
    api_key="your-api-key"
)

# 配置Embedding模型
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-large",
    api_key="your-api-key"
)

# 配置分块参数
Settings.chunk_size = 512
Settings.chunk_overlap = 50

# 现在所有后续操作都会使用这些配置
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
```

#### 配置本地模型

```python
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 使用Ollama本地模型
Settings.llm = Ollama(
    model="llama2",
    base_url="http://localhost:11434"
)

# 使用HuggingFace Embedding
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh-v1.5"
)
```

### 6.2 性能优化

#### 分块优化

```python
from llama_index.core.node_parser import SentenceSplitter

# 场景1: 短文本问答（如FAQ）
short_splitter = SentenceSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# 场景2: 长文档分析（如技术文档）
long_splitter = SentenceSplitter(
    chunk_size=2000,
    chunk_overlap=400
)

# 场景3: 中文文档
chinese_splitter = SentenceSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separator="。"  # 使用中文句号
)
```

#### 批量插入优化

```python
from llama_index.core import VectorStoreIndex

# 设置批量插入大小
index = VectorStoreIndex.from_documents(
    documents,
    insert_batch_size=512  # 默认2048
)
```

#### 缓存优化

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# 使用缓存避免重复embedding
documents = SimpleDirectoryReader("./data").load_data()

# 第一次创建索引（会进行embedding）
index = VectorStoreIndex.from_documents(documents)

# 持久化
index.storage_context.persist(persist_dir="./storage")

# 后续加载（不需要重新embedding）
from llama_index.core import StorageContext, load_index_from_storage

storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
```

---

## 第 7 章：生产级应用实战

### 7.1 完整的RAG应用

```python
"""
生产级RAG应用 - LlamaIndex版本
"""
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import os
from pathlib import Path

class LlamaIndexRAG:
    def __init__(self, data_dir="./data", persist_dir="./storage"):
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        self.index = None

        # 配置全局设置
        Settings.llm = OpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.1
        )
        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-3-large"
        )
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50

    def build_index(self, force_rebuild=False):
        """构建或加载索引"""
        if not force_rebuild and Path(self.persist_dir).exists():
            print("加载现有索引...")
            try:
                storage_context = StorageContext.from_defaults(
                    persist_dir=self.persist_dir
                )
                self.index = load_index_from_storage(storage_context)
                print("索引加载成功")
                return
            except:
                print("加载失败，重新构建索引...")

        print("1. 加载文档...")
        documents = SimpleDirectoryReader(self.data_dir).load_data()
        print(f"   加载了 {len(documents)} 个文档")

        print("2. 文档分块...")
        parser = SentenceSplitter(
            chunk_size=Settings.chunk_size,
            chunk_overlap=Settings.chunk_overlap
        )
        nodes = parser.get_nodes_from_documents(documents)
        print(f"   创建了 {len(nodes)} 个节点")

        print("3. 创建向量索引...")
        self.index = VectorStoreIndex(nodes)
        print("   索引创建完成")

        print("4. 持久化索引...")
        self.index.storage_context.persist(persist_dir=self.persist_dir)
        print("   索引已保存")

    def query(self, question, top_k=3, response_mode="compact", show_sources=True):
        """查询"""
        if self.index is None:
            raise ValueError("索引未初始化，请先调用 build_index()")

        # 创建查询引擎
        query_engine = self.index.as_query_engine(
            similarity_top_k=top_k,
            response_mode=response_mode
        )

        print(f"\n问题: {question}")
        response = query_engine.query(question)

        print(f"\n回答:\n{response}\n")

        if show_sources:
            print("来源:")
            for i, node in enumerate(response.source_nodes, 1):
                print(f"  {i}. {node.text[:100]}... (得分: {node.score:.4f})")
                if node.metadata:
                    print(f"     元数据: {node.metadata}")

        return response

    def query_stream(self, question, top_k=3):
        """流式查询"""
        if self.index is None:
            raise ValueError("索引未初始化")

        query_engine = self.index.as_query_engine(
            similarity_top_k=top_k,
            streaming=True
        )

        print(f"\n问题: {question}")
        print("回答: ", end="")

        response = query_engine.query(question)
        for text in response.response_gen:
            print(text, end="", flush=True)
        print("\n")

        return response

    def chat(self):
        """交互式聊天"""
        if self.index is None:
            raise ValueError("索引未初始化")

        chat_engine = self.index.as_chat_engine()

        print("开始聊天（输入 'quit' 退出）")
        while True:
            question = input("\n你: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                print("再见！")
                break

            if not question:
                continue

            response = chat_engine.chat(question)
            print(f"AI: {response}")

# 使用示例
if __name__ == "__main__":
    # 设置API Key
    os.environ["OPENAI_API_KEY"] = "sk-your-key"

    # 初始化RAG应用
    rag = LlamaIndexRAG(data_dir="./data", persist_dir="./storage")

    # 构建索引
    rag.build_index()

    # 查询
    questions = [
        "文档的主要内容是什么？",
        "有哪些关键概念？",
        "如何快速上手？"
    ]

    for q in questions:
        rag.query(q, top_k=3, response_mode="compact")
        print("-" * 80)

    # 流式查询
    rag.query_stream("详细解释LlamaIndex的架构")

    # 交互式聊天
    rag.chat()
```

---

## 第 8 章：与 LangChain 集成

### 8.1 LlamaIndex 作为 LangChain 工具

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from langchain_core.tools import tool

# 1. 创建LlamaIndex索引
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# 2. 封装为LangChain工具
@tool
def search_documents(query: str) -> str:
    """搜索文档库，回答关于文档的问题。"""
    response = query_engine.query(query)
    return str(response)

# 3. 在LangChain Agent中使用
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4"),
    tools=[search_documents]
)

# 4. 运行
result = agent.invoke({
    "messages": [("user", "文档中提到了哪些关键概念？")]
})
print(result["messages"][-1].content)
```

### 8.2 完整集成示例

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import os

# 设置环境变量
os.environ["OPENAI_API_KEY"] = "sk-your-key"

# 1. 创建多个 LlamaIndex 索引
technical_docs = SimpleDirectoryReader("./technical_docs").load_data()
business_docs = SimpleDirectoryReader("./business_docs").load_data()

tech_index = VectorStoreIndex.from_documents(technical_docs)
business_index = VectorStoreIndex.from_documents(business_docs)

# 2. 创建多个工具
@tool
def search_technical_docs(query: str) -> str:
    """搜索技术文档，回答技术相关问题。"""
    response = tech_index.as_query_engine().query(query)
    return str(response)

@tool
def search_business_docs(query: str) -> str:
    """搜索业务文档，回答业务相关问题。"""
    response = business_index.as_query_engine().query(query)
    return str(response)

# 3. 创建 Agent
agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4"),
    tools=[search_technical_docs, search_business_docs]
)

# 4. 使用
result = agent.invoke({
    "messages": [("user", "技术架构是什么？业务流程是怎样的？")]
})

print(result["messages"][-1].content)
```

---

## 本章小结

本章我们完整学习了LlamaIndex的RAG基础：

**第1章回顾**：
- LlamaIndex vs LangChain 设计哲学
- 核心优势和使用场景
- 5行代码快速开始
- 核心组件（Document、Node、Index）

**第2章回顾**：
- 文档加载器（SimpleDirectoryReader、专用加载器）
- 节点解析器（SentenceSplitter、SemanticSplitter）
- 摄取管道（IngestionPipeline）

**第3章回顾**：
- 索引类型（Vector、Summary、Tree、Keyword）
- 持久化（本地、远程S3）
- 向量数据库集成（Chroma等）

**第4章回顾**：
- 查询引擎（响应模式、流式输出、自定义Prompt）
- 检索器（Retriever）
- 聊天引擎（Chat Engine）

**第5章回顾**：
- 低级组件（Retriever + ResponseSynthesizer）
- 手动控制检索和生成
- 自定义后处理逻辑

**第6章回顾**：
- 全局配置（Settings）
- 性能优化

**第7章回顾**：
- 生产级RAG应用

**第8章回顾**：
- 与LangChain集成

---

## 思考与练习

1. **练习1**：使用LlamaIndex构建一个本地文档问答系统
2. **练习2**：对比不同索引类型的效果
3. **练习3**：实现一个使用语义分块的高质量RAG系统
4. **练习4**：将LlamaIndex集成到LangChain Agent中
5. **练习5**：使用低级组件实现自定义检索逻辑

---

## 参考资源

- [LlamaIndex官方文档](https://docs.llamaindex.ai/)
- [LlamaIndex GitHub](https://github.com/run-llama/llama_index)
- [LlamaIndex Examples](https://github.com/run-llama/llama_index/tree/main/docs/examples)
- [LlamaHub - Data Loaders](https://llamahub.ai/)

---

## 下一步学习

完成本章学习后，建议继续学习：

1. **RAG 进阶篇**：高级检索策略、混合检索、重排序
2. **Agent 篇**：使用 LlamaIndex 构建智能 Agent
3. **生产部署篇**：性能优化、监控、评估

---

**版本信息**：
- LlamaIndex: 0.11.0+
- llama-index-core: 0.11.0+
- llama-index-llms-openai: 0.2.0+
- llama-index-embeddings-openai: 0.2.0+
- 最后更新: 2026-01-19
