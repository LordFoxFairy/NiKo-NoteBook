# 第四篇：RAG基础篇 (LlamaIndex)

## 📋 前置准备

### 环境配置

```bash
# 核心依赖
pip install llama-index>=0.14.8
pip install llama-index-core>=0.14.8
pip install llama-index-llms-openai>=0.2.0
pip install llama-index-embeddings-openai>=0.2.0

# 向量数据库集成
pip install llama-index-vector-stores-chroma
pip install chromadb>=0.5.0

# 可选依赖
pip install pypdf  # PDF支持
pip install python-dotenv  # 环境变量管理
```

### 环境变量

```python
# .env
OPENAI_API_KEY=sk-your-api-key
```

---

# 第 1 章：LlamaIndex核心概念

## 1.1 为什么选择LlamaIndex

### 1.1.1 LlamaIndex的设计哲学

LlamaIndex是专门为RAG（检索增强生成）应用而设计的框架，其核心理念是：

**核心理念**：

1. **数据优先**（Data-First）
   - 一切从数据开始
   - 内置丰富的数据连接器
   - 支持结构化和非结构化数据

2. **索引即查询**（Index as Interface）
   - 多种索引类型适应不同场景
   - 索引自动优化查询策略
   - 查询引擎开箱即用

3. **模块化设计**（Modular Architecture）
   ```
   Reader → Parser → Index → Retriever → Query Engine
   ```

4. **LLM无关**（LLM-Agnostic）
   - 支持OpenAI、Anthropic、本地模型
   - 统一的接口切换模型

### 1.1.2 LlamaIndex的核心优势

| 优势 | 说明 | 示例 |
|------|------|------|
| **开箱即用** | 5行代码实现完整RAG | SimpleDirectoryReader + VectorStoreIndex |
| **丰富的加载器** | 支持100+种数据源 | PDF、Web、数据库、API等 |
| **多种索引类型** | 适应不同场景 | Vector、Summary、Tree、Keyword等 |
| **智能查询引擎** | 自动优化检索策略 | 自动选择最佳检索方式 |
| **模块化架构** | 灵活组合 | 可替换任何组件 |

---

## 1.2 快速开始：5行代码实现RAG

### 1.2.1 最简单的RAG应用

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
- ✅ 文档分块
- ✅ 向量化（Embedding）
- ✅ 向量存储
- ✅ 检索
- ✅ LLM生成答案

### 1.2.2 查看详细信息

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# 加载文档
documents = SimpleDirectoryReader("./data").load_data()

print(f"📄 加载了 {len(documents)} 个文档")
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

print(f"\n💡 回答:\n{response}\n")
print("📚 来源:")
for i, node in enumerate(response.source_nodes, 1):
    print(f"{i}. {node.text[:100]}... (得分: {node.score:.4f})")
```

---

## 1.3 核心组件详解

### 1.3.1 文档（Document）

Document是LlamaIndex的基本数据单元：

```python
from llama_index.core import Document

# 手动创建文档
doc1 = Document(
    text="这是文档内容",
    metadata={
        "source": "manual",
        "author": "张三",
        "date": "2025-11-23"
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

### 1.3.2 节点（Node）

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

print(f"✂️  分割成 {len(nodes)} 个节点")
for i, node in enumerate(nodes[:3], 1):
    print(f"\n节点 {i}:")
    print(f"  内容: {node.text[:100]}...")
    print(f"  长度: {len(node.text)}")
```

### 1.3.3 索引（Index）

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

# 第 2 章：文档加载与处理

## 2.1 文档加载器

### 2.1.1 SimpleDirectoryReader（最常用）

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
```

**支持的文件格式**：
- 📄 文本：`.txt`, `.md`, `.csv`
- 📘 文档：`.pdf`, `.docx`, `.pptx`
- 💻 代码：`.py`, `.js`, `.java`, `.cpp`
- 🌐 网页：`.html`, `.htm`
- 📊 数据：`.json`, `.xml`

### 2.1.2 专用加载器

```python
# PDF加载器
from llama_index.readers.file import PDFReader

reader = PDFReader()
documents = reader.load_data(file="paper.pdf")

# 网页加载器
from llama_index.readers.web import SimpleWebPageReader

reader = SimpleWebPageReader()
documents = reader.load_data(urls=["https://example.com"])

# 数据库加载器
from llama_index.readers.database import DatabaseReader

reader = DatabaseReader(
    uri="postgresql://user:password@localhost/dbname"
)
documents = reader.load_data(
    query="SELECT id, content FROM documents"
)
```

---

## 2.2 节点解析器（Node Parser）

### 2.2.1 为什么需要分块？

```mermaid
graph LR
    A[长文档<br>10000字] --> B{直接输入LLM?}
    B -- ❌ --> C[超出上下文窗口]
    B -- ❌ --> D[信息噪音太多]
    B -- ✅ 分块 --> E[多个小块<br>每块500字]
    E --> F[精确检索]
    F --> G[高质量回答]
```

**分块的好处**：
- ✅ 适应模型上下文窗口
- ✅ 提高检索精确度
- ✅ 降低成本（只处理相关片段）
- ✅ 保持语义完整性

### 2.2.2 SentenceSplitter - 智能句子分割

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

### 2.2.3 SemanticSplitter - 语义分块

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

print(f"✂️  语义分块创建了 {len(nodes)} 个节点")
```

**语义分块的优势**：
- ✅ 保持语义完整性
- ✅ 自适应块大小
- ✅ 更好的检索效果

**何时使用语义分块**：
- 长文档（> 5000字）
- 复杂结构（学术论文、技术文档）
- 高质量要求（生产环境）

### 2.2.4 分块策略对比

| 策略 | 优势 | 劣势 | 适用场景 |
|------|------|------|---------|
| **SentenceSplitter** | 快速、简单 | 可能切断语义 | 通用文档、快速原型 |
| **SemanticSplitter** | 语义完整性最佳 | 计算开销大 | 学术论文、技术文档 |
| **CodeSplitter** | 保持代码结构 | 仅限代码 | 代码文档 |
| **MarkdownNodeParser** | 保持Markdown结构 | 仅限Markdown | 文档、教程 |

---

# 第 3 章：索引与查询

## 3.1 索引类型

### 3.1.1 VectorStoreIndex - 向量索引（最常用）

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
```mermaid
graph LR
    A[文档] --> B[Embedding]
    B --> C[向量存储]
    D[查询] --> E[Embedding]
    E --> F[向量相似度搜索]
    C --> F
    F --> G[Top-K节点]
    G --> H[LLM生成答案]
```

**适用场景**：
- ✅ 语义搜索
- ✅ 问答系统
- ✅ 文档检索

### 3.1.2 SummaryIndex - 摘要索引

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
- ✅ 文档摘要
- ✅ 全面分析
- ✅ 小数据集

### 3.1.3 TreeIndex - 树形索引

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

### 3.1.4 KeywordTableIndex - 关键词索引

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
- ✅ 精确关键词搜索
- ✅ 结构化文档
- ✅ 代码搜索

---

## 3.2 向量存储

### 3.2.1 内置向量存储

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./data").load_data()

# 默认使用内存存储（SimpleVectorStore）
index = VectorStoreIndex.from_documents(documents)

# 持久化到磁盘
index.storage_context.persist(persist_dir="./storage")

# 从磁盘加载
from llama_index.core import StorageContext, load_index_from_storage

storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
```

### 3.2.2 集成Chroma向量数据库

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

### 3.2.3 向量数据库选择指南

| 数据库 | 类型 | 性能 | 部署难度 | 适用场景 |
|--------|------|------|---------|---------|
| **SimpleVectorStore** | 内存 | 低 | ⭐ | 开发测试、小数据集 |
| **Chroma** | 嵌入式 | 中 | ⭐⭐ | 中小型应用、快速开发 |
| **Pinecone** | 云服务 | 高 | ⭐ | 云原生、无需运维 |
| **Qdrant** | 服务 | 高 | ⭐⭐⭐ | 生产环境、分布式 |
| **Weaviate** | 服务 | 高 | ⭐⭐⭐ | 企业级、GraphRAG |

---

## 3.3 查询引擎

### 3.3.1 基础查询引擎

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

### 3.3.2 响应模式（Response Mode）

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| **refine** | 逐个节点精炼答案（默认） | 高质量答案 |
| **compact** | 合并节点后一次生成 | 平衡质量和速度 |
| **tree_summarize** | 树形汇总 | 大量文档 |
| **simple_summarize** | 简单合并 | 快速摘要 |
| **no_text** | 只返回节点，不生成 | 检索测试 |

```python
# 不同响应模式对比
query_engine_refine = index.as_query_engine(response_mode="refine")
query_engine_compact = index.as_query_engine(response_mode="compact")

query = "什么是RAG?"
response1 = query_engine_refine.query(query)  # 更高质量
response2 = query_engine_compact.query(query)  # 更快速度
```

### 3.3.3 流式输出

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

### 3.3.4 自定义Prompt

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

## 3.4 检索器（Retrievers）

### 3.4.1 基础检索器

```python
from llama_index.core import VectorStoreIndex

# 创建索引
index = VectorStoreIndex.from_documents(documents)

# 创建检索器
retriever = index.as_retriever(
    similarity_top_k=5,  # 返回Top-5
    vector_store_query_mode="default"
)

# 检索
nodes = retriever.retrieve("什么是向量索引?")

for i, node in enumerate(nodes, 1):
    print(f"\n节点 {i} (得分: {node.score:.4f}):")
    print(node.text[:200])
```

### 3.4.2 自定义检索器

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

# 第 4 章：配置与优化

## 4.1 全局配置（Settings）

### 4.1.1 配置LLM和Embedding

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

### 4.1.2 配置本地模型

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

---

## 4.2 性能优化

### 4.2.1 分块优化

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

### 4.2.2 缓存优化

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

# 第 5 章：完整应用实战

## 5.1 生产级RAG应用

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
            print("📂 加载现有索引...")
            try:
                storage_context = StorageContext.from_defaults(
                    persist_dir=self.persist_dir
                )
                self.index = load_index_from_storage(storage_context)
                print("✅ 索引加载成功")
                return
            except:
                print("⚠️  加载失败，重新构建索引...")

        print("📄 1. 加载文档...")
        documents = SimpleDirectoryReader(self.data_dir).load_data()
        print(f"   ✅ 加载了 {len(documents)} 个文档")

        print("✂️  2. 文档分块...")
        parser = SentenceSplitter(
            chunk_size=Settings.chunk_size,
            chunk_overlap=Settings.chunk_overlap
        )
        nodes = parser.get_nodes_from_documents(documents)
        print(f"   ✅ 创建了 {len(nodes)} 个节点")

        print("🔨 3. 创建向量索引...")
        self.index = VectorStoreIndex(nodes)
        print("   ✅ 索引创建完成")

        print("💾 4. 持久化索引...")
        self.index.storage_context.persist(persist_dir=self.persist_dir)
        print("   ✅ 索引已保存")

    def query(self, question, top_k=3, response_mode="compact", show_sources=True):
        """查询"""
        if self.index is None:
            raise ValueError("索引未初始化，请先调用 build_index()")

        # 创建查询引擎
        query_engine = self.index.as_query_engine(
            similarity_top_k=top_k,
            response_mode=response_mode
        )

        print(f"\n❓ 问题: {question}")
        response = query_engine.query(question)

        print(f"\n💡 回答:\n{response}\n")

        if show_sources:
            print("📚 来源:")
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

        print(f"\n❓ 问题: {question}")
        print("💡 回答: ", end="")

        response = query_engine.query(question)
        for text in response.response_gen:
            print(text, end="", flush=True)
        print("\n")

        return response

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
```

---

## 5.2 高级功能

### 5.2.1 使用不同索引类型

```python
from llama_index.core import (
    VectorStoreIndex,
    SummaryIndex,
    TreeIndex,
    KeywordTableIndex
)

# 向量索引（语义搜索）
vector_index = VectorStoreIndex.from_documents(documents)

# 摘要索引（文档摘要）
summary_index = SummaryIndex.from_documents(documents)

# 树形索引（层次化摘要）
tree_index = TreeIndex.from_documents(documents)

# 关键词索引（精确匹配）
keyword_index = KeywordTableIndex.from_documents(documents)

# 根据不同任务选择不同索引
def intelligent_query(query, task_type="search"):
    if task_type == "search":
        engine = vector_index.as_query_engine()
    elif task_type == "summarize":
        engine = summary_index.as_query_engine()
    elif task_type == "hierarchical":
        engine = tree_index.as_query_engine()
    elif task_type == "keyword":
        engine = keyword_index.as_query_engine()
    else:
        engine = vector_index.as_query_engine()

    return engine.query(query)

# 使用
response1 = intelligent_query("什么是RAG?", task_type="search")
response2 = intelligent_query("总结所有文档", task_type="summarize")
```

### 5.2.2 多模态RAG

```python
from llama_index.core import SimpleDirectoryReader
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.multi_modal_llms.openai import OpenAIMultiModal

# 加载包含图片的文档
documents = SimpleDirectoryReader(
    "./data",
    required_exts=[".jpg", ".png", ".txt", ".pdf"]
).load_data()

# 创建多模态索引
multimodal_index = MultiModalVectorStoreIndex.from_documents(documents)

# 使用多模态LLM
multimodal_llm = OpenAIMultiModal(model="gpt-4-vision-preview")

# 查询
query_engine = multimodal_index.as_query_engine(
    multi_modal_llm=multimodal_llm
)
response = query_engine.query("图片中显示的是什么？")
print(response)
```

---

# 第 6 章：与LangChain集成

## 6.1 LlamaIndex作为LangChain工具

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

---

## 本章小结

本章我们完整学习了LlamaIndex的RAG基础：

**第1章回顾**：
- ✅ LlamaIndex设计哲学
- ✅ 核心优势
- ✅ 5行代码快速开始
- ✅ 核心组件（Document、Node、Index）

**第2章回顾**：
- ✅ 文档加载器（SimpleDirectoryReader、专用加载器）
- ✅ 节点解析器（SentenceSplitter、SemanticSplitter）
- ✅ 分块策略选择

**第3章回顾**：
- ✅ 索引类型（Vector、Summary、Tree、Keyword）
- ✅ 向量存储（内置、Chroma集成）
- ✅ 查询引擎（响应模式、流式输出、自定义Prompt）
- ✅ 检索器

**第4章回顾**：
- ✅ 全局配置（Settings）
- ✅ 性能优化

**第5章回顾**：
- ✅ 生产级RAG应用
- ✅ 高级功能（多索引、多模态）

**第6章回顾**：
- ✅ 与LangChain集成

---

## 思考与练习

1. **练习1**：使用LlamaIndex构建一个本地文档问答系统
2. **练习2**：对比不同索引类型的效果
3. **练习3**：实现一个使用语义分块的高质量RAG系统
4. **练习4**：将LlamaIndex集成到LangChain Agent中

---

## 参考资源

- [LlamaIndex官方文档](https://developers.llamaindex.ai/)
- [LlamaIndex GitHub](https://github.com/run-llama/llama_index)
- [LlamaIndex Examples](https://github.com/run-llama/llama_index/tree/main/docs/examples)

---

**版本信息**：
- LlamaIndex: 0.11.0+
- llama-index-core: 0.11.0+
- llama-index-llms-openai: 0.2.0+
- llama-index-embeddings-openai: 0.2.0+
- 最后更新: 2025-11-23
