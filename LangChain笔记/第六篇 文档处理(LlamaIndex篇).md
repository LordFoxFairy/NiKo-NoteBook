# 第六篇补充：文档处理工程 (LlamaIndex)

> **版本信息**:
> - 文档版本: v1.0
> - LlamaIndex: 0.14.8
> - llama-parse: 0.5.0+
> - Python: 3.10+
> - 更新日期: 2025-11-29

---

## 前言

在RAG（检索增强生成）系统中，**文档处理质量直接决定了最终效果**。LlamaIndex作为专为LLM应用设计的数据框架，提供了强大的文档处理能力，特别是其旗舰产品**LlamaParse**——世界首个GenAI原生文档解析平台。

**LlamaIndex文档处理的核心优势**：
1. **LlamaParse**：专为LLM优化的高精度PDF解析
2. **LlamaHub生态**：700+数据加载器，覆盖几乎所有数据源
3. **SimpleDirectoryReader**：一行代码加载多种格式
4. **智能Node Parser**：语义感知的文档分块
5. **多模态支持**：图片、表格、图表的智能提取

本篇将深入探讨**LlamaIndex的文档处理完整方案**，从基础加载到高级解析，构建高质量RAG系统。

---

## 学习路径

```mermaid
graph LR
    A[SimpleDirectoryReader<br/>快速入门] --> B[LlamaHub<br/>700+加载器]
    B --> C[LlamaParse<br/>GenAI原生解析]
    C --> D[Node Parser<br/>智能分块]
    D --> E[多模态处理<br/>图表提取]

    style A fill:#e1f5e1
    style C fill:#fff4e1
    style E fill:#ffe1e1
```

**本篇覆盖内容**：
- **第1章**：LlamaIndex数据加载器生态
- **第2章**：LlamaParse - GenAI原生PDF解析
- **第3章**：Node Parser - 智能分块策略
- **第4章**：多模态文档处理
- **第5章**：生产级文档处理Pipeline

---

# 第1章：LlamaIndex数据加载器生态

## 1.1 SimpleDirectoryReader - 快速入门

### 1.1.1 基础使用

**SimpleDirectoryReader** 是LlamaIndex最简单的文档加载器，支持20+种文件格式：

```python
from llama_index.core import SimpleDirectoryReader

# 一行代码加载目录下所有文档
documents = SimpleDirectoryReader("./data").load_data()

print(f"加载了{len(documents)}个文档")
for doc in documents[:2]:
    print(f"\n文件：{doc.metadata.get('file_name', 'unknown')}")
    print(f"内容：{doc.text[:200]}...")
```

**支持的格式**：
- 文档：PDF, DOCX, DOC, TXT, MD
- 数据：CSV, JSON, XML
- 网页：HTML, MHTML
- 代码：PY, JS, JAVA, CPP
- 其他：EPUB, RTF, PPTX

---

### 1.1.2 高级配置

```python
from llama_index.core import SimpleDirectoryReader

# 高级配置
reader = SimpleDirectoryReader(
    input_dir="./data",
    required_exts=[".pdf", ".docx"],  # 只加载特定格式
    recursive=True,                    # 递归子目录
    exclude_hidden=True,               # 排除隐藏文件
    exclude_empty=True,               # 排除空文件 (0.14.8+新增)
    filename_as_id=True,              # 使用文件名作为ID
    num_files_limit=100               # 限制文件数量
)

documents = reader.load_data()

# 查看元数据
for doc in documents[:3]:
    print(f"文件：{doc.metadata['file_name']}")
    print(f"路径：{doc.metadata['file_path']}")
    print(f"大小：{doc.metadata.get('file_size', 'N/A')} bytes")
    print(f"ID：{doc.doc_id}\n")
```

**完整参数列表** (LlamaIndex 0.14.8):

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `input_dir` | str/Path | None | 输入目录路径 |
| `input_files` | list | None | 指定文件列表 |
| `exclude` | list | None | 排除的文件列表 |
| `exclude_hidden` | bool | True | 排除隐藏文件 |
| `exclude_empty` | bool | False | 排除空文件 **(0.14.8+新增)** |
| `errors` | str | "ignore" | 错误处理策略 |
| `recursive` | bool | False | 递归读取子目录 |
| `encoding` | str | "utf-8" | 文件编码 |
| `filename_as_id` | bool | False | 使用文件名作为文档ID |
| `required_exts` | list[str] | None | 只加载指定扩展名文件 |
| `file_extractor` | dict | None | 自定义文件加载器映射 |
| `num_files_limit` | int | None | 限制加载文件数量 |
| `file_metadata` | Callable | None | 自定义元数据提取函数 |
| `raise_on_error` | bool | False | 遇到错误时抛出异常 |
| `fs` | fsspec.AbstractFileSystem | None | 自定义文件系统(支持S3/GCS等) **(0.14.8+新增)** |

**新参数使用示例**:

```python
# exclude_empty: 自动过滤空文件
reader = SimpleDirectoryReader(
    input_dir="./data",
    exclude_empty=True  # 跳过空文件,避免无效文档
)
documents = reader.load_data()

# fs: 使用自定义文件系统(如S3)
import fsspec
s3_fs = fsspec.filesystem('s3', anon=False)
reader = SimpleDirectoryReader(
    input_dir="s3://my-bucket/documents/",
    fs=s3_fs  # 直接从S3读取文档
)
documents = reader.load_data()
```

---

### 1.1.3 自定义文件加载器

```python
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PDFReader

# 为特定格式指定自定义加载器
file_extractor = {
    ".pdf": PDFReader(),  # 使用专用PDFReader
}

reader = SimpleDirectoryReader(
    input_dir="./data",
    file_extractor=file_extractor
)

documents = reader.load_data()
```

---

## 1.2 LlamaHub - 700+数据加载器

### 1.2.1 LlamaHub概述

**LlamaHub**是LlamaIndex的数据加载器注册中心，提供：
- 700+预构建的数据加载器
- 覆盖文档、数据库、API、网页等
- 社区驱动，持续更新

**浏览器访问**：https://llamahub.ai/

---

### 1.2.2 常用加载器

**PDF加载器**：
```python
from llama_index.readers.file import PDFReader

# 基础PDF加载
reader = PDFReader()
documents = reader.load_data(file="document.pdf")

print(f"加载了{len(documents)}页")
```

**DOCX加载器**：
```python
from llama_index.readers.file import DocxReader

reader = DocxReader()
documents = reader.load_data(file="document.docx")
```

**HTML加载器**：
```python
from llama_index.readers.file import HTMLTagReader

# 按HTML标签提取
reader = HTMLTagReader(
    tag="article",  # 只提取<article>标签内容
    ignore_no_id=False
)
documents = reader.load_data(file="webpage.html")
```

---

### 1.2.3 数据库加载器

```python
# MongoDB加载器
from llama_index.readers.mongodb import SimpleMongoReader

reader = SimpleMongoReader(
    host="localhost",
    port=27017
)
documents = reader.load_data(
    db_name="mydb",
    collection_name="documents"
)

# SQL数据库加载器
from llama_index.readers.database import DatabaseReader

reader = DatabaseReader(
    sql_database="postgresql://user:pass@localhost/db"
)
documents = reader.load_data(
    query="SELECT * FROM articles WHERE category='tech'"
)
```

---

### 1.2.4 API加载器

```python
# GitHub加载器
from llama_index.readers.github import GithubRepositoryReader, GithubClient

github_client = GithubClient(github_token="your-token")
reader = GithubRepositoryReader(
    github_client=github_client,
    owner="langchain-ai",
    repo="langchain",
    filter_file_extensions=[".py", ".md"]
)
documents = reader.load_data(branch="main")

# Notion加载器
from llama_index.readers.notion import NotionPageReader

reader = NotionPageReader(integration_token="your-token")
documents = reader.load_data(page_ids=["page-id-1", "page-id-2"])
```

---

## 1.3 专用PDF加载器对比

### 1.3.1 工具对比矩阵

| 工具 | 速度 | 准确率 | 表格支持 | 图片提取 | 适用场景 |
|------|------|--------|---------|---------|---------|
| **PDFReader** | ⭐⭐⭐⭐ 快 | ⭐⭐⭐ 中等 | ❌ 差 | ❌ 不支持 | 简单PDF |
| **PyMuPDFReader** | ⭐⭐⭐⭐⭐ 最快 | ⭐⭐⭐⭐ 好 | ⭐⭐ 一般 | ✅ 支持 | 批量处理 |
| **PDFPlumberReader** | ⭐⭐⭐ 中等 | ⭐⭐⭐⭐ 好 | ✅ 优秀 | ❌ 不支持 | 表格密集 |
| **LlamaParse** | ⭐⭐ 慢 | ⭐⭐⭐⭐⭐ 最好 | ✅ 优秀 | ✅ 优秀 | 复杂文档 |

---

### 1.3.2 对比示例

```python
from llama_index.readers.file import PDFReader, PyMuPDFReader
import time

pdf_path = "complex_document.pdf"

# Test 1: PDFReader（基础）
start = time.time()
reader1 = PDFReader()
docs1 = reader1.load_data(file=pdf_path)
time1 = time.time() - start
print(f"PDFReader: {len(docs1)}页, {time1:.2f}s")

# Test 2: PyMuPDFReader（快速）
start = time.time()
reader2 = PyMuPDFReader()
docs2 = reader2.load_data(file=pdf_path)
time2 = time.time() - start
print(f"PyMuPDFReader: {len(docs2)}页, {time2:.2f}s")

# 质量对比
print(f"\nPDFReader平均文本长度：{sum(len(d.text) for d in docs1)/len(docs1):.0f}")
print(f"PyMuPDFReader平均文本长度：{sum(len(d.text) for d in docs2)/len(docs2):.0f}")
```

---

## 小结

**第1章核心要点**：

1. **SimpleDirectoryReader**：
   - 一行代码加载多种格式
   - 支持递归、过滤、自定义加载器
   - 最快的入门方式

2. **LlamaHub生态**：
   - 700+预构建加载器
   - 覆盖文档、数据库、API
   - 持续更新的社区资源

3. **PDF工具选择**：
   - 简单文档 → PDFReader
   - 批量处理 → PyMuPDFReader
   - 表格密集 → PDFPlumberReader
   - 复杂文档 → LlamaParse（下一章）

**下一章预告**：
第2章将深入探讨**LlamaParse**——世界首个GenAI原生PDF解析平台。

---

# 第2章：LlamaParse - GenAI原生PDF解析

## 2.1 LlamaParse简介

### 2.1.1 什么是LlamaParse

**LlamaParse** 是LlamaIndex推出的商业级PDF解析服务，核心特点：

**GenAI原生**：
- 使用LLM构建，专为LLM应用优化
- 理解文档语义，而非简单OCR
- 保留上下文关系

**世界级解析能力**：
- 复杂表格识别（跨页、嵌套）
- 数学公式提取
- 多栏布局处理
- 图表与文字关联
- 多语言支持

**为RAG优化**：
- 输出Markdown格式（LLM友好）
- 自动分块建议
- 保留语义结构

---

### 2.1.2 为什么需要LlamaParse

**传统工具的局限**：
```python
# PyPDF2/PDFMiner - 基础工具
❌ 复杂表格识别差
❌ 多栏布局混乱
❌ 公式丢失
❌ 无语义理解

# OCR工具（Tesseract/PaddleOCR）
❌ 只识别文字，无结构
❌ 表格还原困难
❌ 上下文关系丢失
```

**LlamaParse的优势**：
```python
✅ 完整保留表格结构（Markdown格式）
✅ 多栏布局正确还原
✅ 数学公式识别
✅ 图表与说明关联
✅ 语义感知分块
```

---

## 2.2 LlamaParse快速入门

### 2.2.1 安装与配置

```bash
# 安装llama-parse
pip install llama-parse

# 安装llama-index（如果未安装）
pip install llama-index
```

**获取API Key**：
1. 访问 https://cloud.llamaindex.ai/
2. 注册账号
3. 获取API Key

```python
# 设置环境变量
import os
os.environ["LLAMA_CLOUD_API_KEY"] = "llx-your-api-key"
```

---

### 2.2.2 基础使用

```python
from llama_parse import LlamaParse

# 创建解析器
parser = LlamaParse(
    api_key="llx-your-api-key",  # 或从环境变量读取
    result_type="markdown",       # 输出格式：markdown或text
    verbose=True
)

# 解析PDF
documents = parser.load_data("complex_document.pdf")

print(f"解析了{len(documents)}个文档块")
print("\n解析结果（Markdown格式）：")
print(documents[0].text[:500])
```

**输出示例**：
```markdown
# 第一章：引言

本文档介绍了...

## 1.1 背景

在过去的十年中...

### 表格1：年度数据对比

| 年份 | 收入（万元） | 增长率 |
|------|-------------|--------|
| 2021 | 1000        | 10%    |
| 2022 | 1200        | 20%    |
| 2023 | 1500        | 25%    |
```

---

### 2.2.3 高级配置

```python
from llama_parse import LlamaParse

parser = LlamaParse(
    api_key="llx-your-api-key",
    result_type="markdown",

    # 解析选项
    verbose=True,
    language="zh",                    # 语言（zh中文，en英文）
    num_workers=4,                    # 并行处理数

    # 高级特性
    parsing_instruction="""
    请特别注意：
    1. 保留所有表格的完整结构
    2. 数学公式使用LaTeX格式
    3. 图表说明与图表关联
    """,                             # 自定义解析指令

    invalidate_cache=False,          # 使用缓存（加速重复解析）
    do_not_cache=False,

    # 输出控制
    gpt4o_mode=True,                 # 使用GPT-4o增强（更准确但慢）
    gpt4o_api_key="your-openai-key"  # 如果使用gpt4o_mode
)

documents = parser.load_data("academic_paper.pdf")
```

---

## 2.3 LlamaParse高级特性

### 2.3.1 自定义解析指令

**场景1：学术论文**
```python
parser = LlamaParse(
    api_key="llx-your-api-key",
    result_type="markdown",
    parsing_instruction="""
    这是一篇学术论文，请：
    1. 保留所有数学公式（LaTeX格式）
    2. 表格完整提取为Markdown
    3. 图表编号与说明关联
    4. 参考文献单独列出
    """
)

documents = parser.load_data("research_paper.pdf")
```

**场景2：财务报表**
```python
parser = LlamaParse(
    api_key="llx-your-api-key",
    result_type="markdown",
    parsing_instruction="""
    这是财务报表，请：
    1. 重点提取所有表格（资产负债表、利润表）
    2. 保留数字精度（不要四舍五入）
    3. 表格标题与内容关联
    4. 注释与对应表格关联
    """
)

documents = parser.load_data("financial_report.pdf")
```

**场景3：合同文档**
```python
parser = LlamaParse(
    api_key="llx-your-api-key",
    result_type="markdown",
    parsing_instruction="""
    这是法律合同，请：
    1. 保留条款编号结构
    2. 重点标记金额、日期
    3. 甲乙双方信息准确提取
    4. 附件清单单独列出
    """
)

documents = parser.load_data("contract.pdf")
```

---

### 2.3.2 表格处理

**LlamaParse的表格处理优势**：

```python
from llama_parse import LlamaParse

parser = LlamaParse(
    api_key="llx-your-api-key",
    result_type="markdown",
    parsing_instruction="""
    重点处理表格：
    1. 跨页表格合并
    2. 嵌套表格展平
    3. 合并单元格正确处理
    """
)

documents = parser.load_data("tables_heavy.pdf")

# 提取所有表格
tables = []
for doc in documents:
    # LlamaParse输出的Markdown中，表格用 | 分隔
    if "|" in doc.text and "---" in doc.text:
        tables.append(doc.text)

print(f"提取到{len(tables)}个表格")
```

**输出示例**：
```markdown
## 季度财务数据

| 季度 | 收入（万元） | 成本（万元） | 利润（万元） | 利润率 |
|------|-------------|-------------|-------------|--------|
| Q1   | 1000        | 700         | 300         | 30%    |
| Q2   | 1200        | 800         | 400         | 33%    |
| Q3   | 1500        | 900         | 600         | 40%    |
| Q4   | 1800        | 1000        | 800         | 44%    |
| 合计 | 5500        | 3400        | 2100        | 38%    |

**注**：利润率 = 利润 ÷ 收入
```

---

### 2.3.3 多模态支持

**图片提取与描述**：
```python
from llama_parse import LlamaParse

parser = LlamaParse(
    api_key="llx-your-api-key",
    result_type="markdown",
    gpt4o_mode=True,  # 启用GPT-4o进行图片理解
    parsing_instruction="""
    对于图片和图表：
    1. 提取图片并生成描述
    2. 将图表数据转换为表格
    3. 图片编号与正文关联
    """
)

documents = parser.load_data("illustrated_document.pdf")

# LlamaParse会在Markdown中包含图片描述
# 示例输出：
# ![图1：系统架构图](image_description)
# 描述：该图展示了系统的三层架构...
```

---

## 2.4 LlamaParse vs 传统工具

### 2.4.1 实战对比

```python
import time
from llama_parse import LlamaParse
from llama_index.readers.file import PDFReader

pdf_path = "complex_academic_paper.pdf"

# 测试1: 传统PDFReader
start = time.time()
reader = PDFReader()
docs_traditional = reader.load_data(file=pdf_path)
time_traditional = time.time() - start

# 测试2: LlamaParse
start = time.time()
parser = LlamaParse(api_key="llx-your-api-key")
docs_llamaparse = parser.load_data(pdf_path)
time_llamaparse = time.time() - start

# 对比
print("=== 性能对比 ===")
print(f"传统工具：{len(docs_traditional)}页, {time_traditional:.2f}s")
print(f"LlamaParse：{len(docs_llamaparse)}页, {time_llamaparse:.2f}s")

print("\n=== 质量对比 ===")
# 检查表格提取
traditional_tables = sum(1 for d in docs_traditional if "|" in d.text)
llamaparse_tables = sum(1 for d in docs_llamaparse if "|" in d.text)
print(f"传统工具提取表格：{traditional_tables}个")
print(f"LlamaParse提取表格：{llamaparse_tables}个")

# 平均文本长度
avg_traditional = sum(len(d.text) for d in docs_traditional) / len(docs_traditional)
avg_llamaparse = sum(len(d.text) for d in docs_llamaparse) / len(docs_llamaparse)
print(f"传统工具平均文本长度：{avg_traditional:.0f}字符")
print(f"LlamaParse平均文本长度：{avg_llamaparse:.0f}字符")
```

**典型结果**：
```
=== 性能对比 ===
传统工具：20页, 2.5s
LlamaParse：20页, 45s

=== 质量对比 ===
传统工具提取表格：2个（格式混乱）
LlamaParse提取表格：15个（完美Markdown）

传统工具平均文本长度：800字符
LlamaParse平均文本长度：1500字符（包含完整表格和结构）
```

---

### 2.4.2 成本考量

**LlamaParse定价**（2025年数据）：
- 免费额度：1000页/月
- Pay-as-you-go：约$0.003/页
- 使用gpt4o_mode：约$0.01/页（更准确）

**成本优化策略**：
1. **分层处理**：
   - 简单PDF → 传统工具（免费）
   - 复杂PDF → LlamaParse（付费）

2. **缓存利用**：
   ```python
   parser = LlamaParse(
       api_key="llx-your-api-key",
       invalidate_cache=False  # 重复解析使用缓存
   )
   ```

3. **批量处理**：
   ```python
   # 一次解析多个文件（共享初始化成本）
   parser = LlamaParse(api_key="llx-your-api-key")

   files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
   all_docs = []
   for file in files:
       docs = parser.load_data(file)
       all_docs.extend(docs)
   ```

---

## 2.5 集成到RAG系统

### 2.5.1 完整示例

```python
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# 步骤1: 使用LlamaParse解析复杂PDF
parser = LlamaParse(
    api_key="llx-your-api-key",
    result_type="markdown",
    parsing_instruction="""
    这是一份技术文档，请：
    1. 保留所有表格
    2. 代码块使用```标记
    3. 章节结构清晰
    """
)

documents = parser.load_data("technical_manual.pdf")

# 步骤2: 使用MarkdownNodeParser分块（保留Markdown结构）
node_parser = MarkdownNodeParser()
nodes = node_parser.get_nodes_from_documents(documents)

print(f"解析了{len(documents)}个文档，生成{len(nodes)}个节点")

# 步骤3: 构建索引
index = VectorStoreIndex(
    nodes=nodes,
    embed_model=OpenAIEmbedding()
)

# 步骤4: 创建查询引擎
query_engine = index.as_query_engine(
    llm=OpenAI(model="gpt-4"),
    similarity_top_k=5
)

# 步骤5: 查询
response = query_engine.query("系统架构中的各个组件是什么？")
print(response.response)

# 查看检索到的源节点
for i, node in enumerate(response.source_nodes, 1):
    print(f"\n来源{i}：")
    print(node.text[:200])
```

---

---

## 2.6 MinerU vs LlamaParse - 深度对比

### 2.6.1 核心定位对比

**LlamaParse - GenAI原生商业解析**：
- **定位**：世界首个GenAI原生文档解析平台
- **核心优势**：使用LLM理解文档语义，为LLM应用优化
- **技术路线**：基于大语言模型的智能解析
- **商业模式**：付费服务（免费额度 + Pay-as-you-go）
- **官方文档**：https://developers.llamaindex.ai/python/framework/llama_cloud/llama_parse/

**MinerU - 学术开源高性能解析**：
- **定位**：将复杂文档转换为LLM就绪的Markdown/JSON
- **核心优势**：SOTA性能（在OmniDocBench上超越GPT-4o和Gemini 2.5 Pro）
- **技术路线**：传统CV/OCR（pipeline）+ 最新VLM模型MinerU2.5（1.2B参数）
- **开源协议**：AGPL-3.0（完全开源，GitHub 49.3k+ stars）
- **官方仓库**：https://github.com/opendatalab/MinerU

---

### 2.6.2 功能对比矩阵

| 功能维度 | LlamaParse | MinerU | 说明 |
|---------|-----------|--------|------|
| **文本提取** | ⭐⭐⭐⭐⭐ 优秀 | ⭐⭐⭐⭐⭐ 84语言 | 两者均支持多语言文本提取 |
| **表格识别** | ⭐⭐⭐⭐⭐ Markdown完美 | ⭐⭐⭐⭐⭐ 旋转/无边框/跨页 | **平手**（LlamaParse更LLM友好，MinerU更全面） |
| **公式识别** | ⭐⭐⭐⭐ LaTeX格式 | ⭐⭐⭐⭐⭐ UniMER（SOTA） | **MinerU优势**（提速1400%，准确率98%+） |
| **多模态支持** | ⭐⭐⭐⭐⭐ GPT-4o图片理解 | ⭐⭐⭐⭐ OCR+VLM | **LlamaParse优势**（语义理解更强） |
| **复杂布局** | ⭐⭐⭐⭐⭐ 多栏/嵌套 | ⭐⭐⭐⭐⭐ DocLayout-YOLO | **平手**（不同技术路径） |
| **处理速度** | ⭐⭐⭐ 较慢（API） | ⭐⭐⭐⭐⭐ 10,000+ tokens/s | **MinerU优势**（本地GPU推理） |
| **成本** | ⭐⭐⭐ 付费（约$0.003/页） | ⭐⭐⭐⭐⭐ 完全免费 | **MinerU优势**（开源） |
| **部署复杂度** | ⭐⭐⭐⭐⭐ API调用（零部署） | ⭐⭐⭐ 需GPU（6-8GB显存） | **LlamaParse优势**（云服务） |
| **数据隐私** | ⭐⭐⭐ 数据上传到云端 | ⭐⭐⭐⭐⭐ 本地处理 | **MinerU优势**（企业友好） |

---

### 2.6.3 性能基准对比

**解析准确率（基于OmniDocBench基准测试）**：

| 模型 | 综合得分 | 表格准确率 | 公式准确率 | 布局准确率 | 数据来源 |
|------|---------|-----------|-----------|-----------|----------|
| **MinerU2.5** | SOTA | 95%+ | 98%+ | 97%+ | [MinerU GitHub](https://github.com/opendatalab/MinerU) |
| **GPT-4o** | 优秀 | 93% | 95% | 95% | OmniDocBench |
| **LlamaParse** | 优秀 | 96%+ | 94% | 96%+ | LlamaIndex官方数据 |
| **传统OCR** | 一般 | 70-80% | 60-70% | 75-85% | 行业平均水平 |

**处理速度对比**：

```python
# 测试文档：100页学术论文（含50个表格、30个公式、20张图片）

# LlamaParse
⏱ 处理时间：~150-300秒（取决于API负载）
💰 成本：$0.30（100页 × $0.003/页）
✅ 优势：无需本地资源，零部署
❌ 劣势：依赖网络，按量计费

# MinerU (NVIDIA 4090)
⏱ 处理时间：~30-60秒（本地GPU）
💰 成本：$0（硬件成本已摊销）
✅ 优势：极速、离线、隐私、免费
❌ 劣势：需要GPU硬件（6-8GB显存）

# MinerU (Apple M3 Max with MLX)
⏱ 处理时间：~60-120秒（MLX加速）
💰 成本：$0
✅ 优势：Apple芯片优化（100-200%提速）
❌ 劣势：需要Mac设备
```

---

### 2.6.4 适用场景决策树

```mermaid
graph TD
    Start[文档解析需求] --> Q1{是否有GPU?}
    Q1 -->|无GPU| Q2{预算如何?}
    Q1 -->|有GPU| Q3{数据敏感?}

    Q2 -->|预算充足| LlamaParse[✅ LlamaParse<br/>云端API，零部署]
    Q2 -->|预算有限| Traditional[传统工具<br/>PyMuPDF/PDFPlumber]

    Q3 -->|高度敏感| MinerU[✅ MinerU<br/>本地处理，隐私安全]
    Q3 -->|可接受云端| Q4{文档复杂度?}

    Q4 -->|极高| LlamaParse2[✅ LlamaParse<br/>语义理解强]
    Q4 -->|一般| MinerU2[✅ MinerU<br/>速度快，免费]

    style LlamaParse fill:#fff4e1
    style LlamaParse2 fill:#fff4e1
    style MinerU fill:#e1f5e1
    style MinerU2 fill:#e1f5e1
```

**决策指南**：

1. **选择LlamaParse的场景**：
   - ✅ 无GPU资源（云端部署）
   - ✅ 需要最强语义理解（复杂合同、学术论文）
   - ✅ 快速原型开发（零部署成本）
   - ✅ 已有LlamaCloud生态（与LlamaIndex无缝集成）
   - ✅ 偶尔使用（<1000页/月，成本可控）
   - ❌ 预算有限（大规模使用）
   - ❌ 高度敏感数据（不能上传云端）

2. **选择MinerU的场景**：
   - ✅ 有GPU（6-8GB显存）或Apple Silicon
   - ✅ 大规模批量处理（节省成本）
   - ✅ 数据隐私要求（本地处理）
   - ✅ 公式密集文档（SOTA公式识别，提速1400%）
   - ✅ 高吞吐量需求（10,000+ tokens/s）
   - ✅ 离线环境（无需网络）
   - ❌ 不想管理硬件
   - ❌ 无技术团队

---

### 2.6.5 混合使用策略

**策略1：智能路由（基于文档特征）**

```python
from pathlib import Path
from llama_parse import LlamaParse
from llama_index.readers.file import PyMuPDFReader

class HybridDocumentParser:
    """混合使用LlamaParse和传统工具的智能路由"""

    def __init__(
        self,
        llamaparse_api_key: str,
        monthly_budget: float = 50.0
    ):
        self.llamaparse = LlamaParse(api_key=llamaparse_api_key)
        self.pymupdf = PyMuPDFReader()
        self.monthly_budget = monthly_budget
        self.llamaparse_usage = 0.0

    def analyze_document_complexity(self, pdf_path: str) -> dict:
        """分析文档复杂度"""
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        stats = {
            'pages': len(doc),
            'has_images': False,
            'has_tables': False,
            'has_formulas': False,
            'text_density': 0
        }

        # 采样分析（前3页）
        for page in doc[:min(3, len(doc))]:
            text = page.get_text()
            stats['text_density'] += len(text)

            # 简单启发式检测
            if '∫' in text or '∑' in text or '\\' in text:
                stats['has_formulas'] = True
            if page.get_images():
                stats['has_images'] = True

        stats['text_density'] /= min(3, len(doc))
        doc.close()

        # 复杂度评分
        complexity_score = 0
        if stats['pages'] > 50:
            complexity_score += 20
        if stats['has_formulas']:
            complexity_score += 30
        if stats['has_images']:
            complexity_score += 20
        if stats['text_density'] < 500:  # 低文本密度（可能很多表格/图片）
            complexity_score += 30

        stats['complexity_score'] = complexity_score
        stats['complexity_level'] = (
            'high' if complexity_score >= 60 else
            'medium' if complexity_score >= 30 else
            'low'
        )

        return stats

    def parse(self, pdf_path: str) -> list:
        """智能选择解析工具"""
        stats = self.analyze_document_complexity(pdf_path)

        print(f"文档分析：{stats['pages']}页, 复杂度={stats['complexity_level']}")
        print(f"  公式: {stats['has_formulas']}, 图片: {stats['has_images']}")

        # 决策逻辑
        estimated_cost = stats['pages'] * 0.003

        # 规则1: 预算不足 → 传统工具（免费）
        if self.llamaparse_usage + estimated_cost > self.monthly_budget:
            print(f"→ 选择PyMuPDF（预算不足：${self.llamaparse_usage:.2f}/{self.monthly_budget}）")
            return self.pymupdf.load_data(file=pdf_path)

        # 规则2: 高复杂度且有预算 → LlamaParse（语义理解）
        if stats['complexity_level'] == 'high':
            print(f"→ 选择LlamaParse（复杂文档，成本${estimated_cost:.2f}）")
            self.llamaparse_usage += estimated_cost
            return self.llamaparse.load_data(pdf_path)

        # 规则3: 默认 → 传统工具（速度快，免费）
        print("→ 选择PyMuPDF（默认策略）")
        return self.pymupdf.load_data(file=pdf_path)


# 使用示例
parser = HybridDocumentParser(
    llamaparse_api_key="llx-your-api-key",
    monthly_budget=50.0
)

# 批量处理
for pdf_file in Path("./documents").glob("*.pdf"):
    print(f"\n处理：{pdf_file.name}")
    documents = parser.parse(str(pdf_file))
    print(f"✅ 提取{len(documents)}个文档块")

print(f"\n本月LlamaParse使用：${parser.llamaparse_usage:.2f}/{parser.monthly_budget}")
```

---

**策略2：成本优化（分层处理）**

```python
from llama_index.readers.file import PyMuPDFReader, PDFReader
from llama_parse import LlamaParse

class CostOptimizedStrategy:
    """成本优化的分层策略"""

    def __init__(self):
        self.tier_1_tool = PyMuPDFReader()  # 免费，快速
        self.tier_2_tool = LlamaParse()     # 付费，最高质量

    def parse_with_fallback(self, pdf_path: str):
        """分层降级策略"""

        # Tier 1: 尝试快速免费工具
        try:
            print("尝试 Tier 1: PyMuPDFReader（免费，快速）")
            docs = self.tier_1_tool.load_data(file=pdf_path)

            # 质量检查
            quality_score = self._check_quality(docs)
            print(f"质量评分：{quality_score:.2f}")

            if quality_score >= 0.8:
                print("✅ Tier 1 成功，质量合格")
                return docs
            else:
                print("⚠️ Tier 1 质量不足，升级到LlamaParse...")
        except Exception as e:
            print(f"❌ Tier 1 失败: {e}")

        # Tier 2: 商业级工具（最后手段）
        print("使用 Tier 2: LlamaParse（付费，最高质量）")
        return self.tier_2_tool.load_data(pdf_path)

    def _check_quality(self, docs: list) -> float:
        """质量评分（0-1）"""
        if not docs:
            return 0.0

        total_text = "".join([d.text for d in docs])

        score = 0.0
        # 有足够内容
        if len(total_text) > 500:
            score += 0.4
        # 有表格（检测Markdown表格）
        if "|" in total_text and "---" in total_text:
            score += 0.3
        # 结构完整（有换行）
        if total_text.count("\n") > 10:
            score += 0.3

        return score


# 使用
strategy = CostOptimizedStrategy()
docs = strategy.parse_with_fallback("complex_document.pdf")
```

---

**策略3：降级方案（容错处理）**

```python
from llama_parse import LlamaParse
from llama_index.readers.file import PyMuPDFReader, PDFReader

def robust_parse_with_retry(pdf_path: str) -> list:
    """带重试的鲁棒解析"""
    strategies = [
        ("LlamaParse", lambda: LlamaParse().load_data(pdf_path)),
        ("PyMuPDF", lambda: PyMuPDFReader().load_data(file=pdf_path)),
        ("PDFReader", lambda: PDFReader().load_data(file=pdf_path)),
    ]

    for name, parser_func in strategies:
        try:
            print(f"尝试：{name}")
            docs = parser_func()
            if docs and len(docs) > 0:
                print(f"✅ {name} 成功")
                return docs
        except Exception as e:
            print(f"❌ {name} 失败: {e}，尝试下一个...")

    raise Exception("所有解析策略均失败")


# 使用
docs = robust_parse_with_retry("problematic_document.pdf")
```

---

### 2.6.6 最佳实践建议

**1. 组合使用场景**：

```python
# 场景1：学术论文（公式密集）
# 推荐：MinerU（公式识别SOTA，免费）
# 注：MinerU需单独安装，这里使用LlamaParse作为替代
parser = LlamaParse(
    api_key="llx-your-api-key",
    parsing_instruction="重点识别数学公式（LaTeX格式）"
)

# 场景2：财务报表（复杂表格 + 语义理解）
# 推荐：LlamaParse（理解表格关系，Markdown输出）
parser = LlamaParse(
    api_key="llx-your-api-key",
    parsing_instruction="重点提取财务表格，保留数字精度"
)

# 场景3：法律合同（结构化条款）
# 推荐：LlamaParse（语义理解强）
parser = LlamaParse(
    api_key="llx-your-api-key",
    parsing_instruction="保留条款编号，标记金额和日期"
)

# 场景4：通用技术文档（成本敏感）
# 推荐：传统工具（速度快，免费）
from llama_index.readers.file import PyMuPDFReader
parser = PyMuPDFReader()
```

---

**2. 成本收益分析**：

| 场景 | 文档量 | LlamaParse成本 | MinerU成本 | 推荐方案 | 理由 |
|------|-------|---------------|-----------|---------|------|
| **原型开发** | <100页 | <$0.3 | $0 | **LlamaParse** | 快速迭代，零部署 |
| **小规模生产** | 1000页/月 | ~$3/月 | $0 | **MinerU** | 成本优势明显 |
| **大规模生产** | 100,000页/月 | ~$300/月 | $0（需GPU） | **MinerU** | 显著节省（需投资GPU） |
| **隐私敏感** | 任意 | 不适用 | $0 | **MinerU** | 数据安全，本地处理 |

---

**3. 技术选型决策**：

```python
# ✅ 推荐：LlamaParse
if (
    无GPU资源 or
    快速原型开发 or
    偶尔使用（<1000页/月） or
    需要最佳语义理解
):
    use_llamaparse()

# ✅ 推荐：MinerU
if (
    有GPU（6-8GB+）or Apple Silicon or
    大规模批量处理 or
    数据隐私要求 or
    公式密集文档 or
    预算有限
):
    use_mineru()

# ✅ 推荐：混合策略
if (
    文档类型多样 and
    成本敏感 and
    有技术能力
):
    use_hybrid_strategy()
```

---

**4. 实际生产经验**：

**案例1：某金融公司（10万+页财报/年）**
- **初期方案**：全部使用LlamaParse
  - 成本：$300/月
  - 问题：成本过高

- **优化后方案**：
  - 简单财报（70%） → PyMuPDF（免费）
  - 复杂财报（25%） → MinerU（免费，本地GPU）
  - 极复杂文档（5%） → LlamaParse（付费）

- **成果**：
  - 成本从$300/月降至$15/月
  - 节省成本：**95%**
  - 质量无明显下降

**案例2：某学术机构（学术论文解析）**
- **方案**：100% MinerU
- **原因**：
  - 公式密集（MinerU公式识别SOTA）
  - 免费开源（预算有限）
  - 数据隐私（学术敏感）
- **硬件**：NVIDIA A100（已有）
- **成果**：每月处理10万页，成本$0

**案例3：某创业公司（RAG原型）**
- **方案**：100% LlamaParse
- **原因**：
  - 无GPU资源
  - 快速迭代（2周上线）
  - 零部署成本
- **成本**：$5-10/月（免费额度内）
- **成果**：2周内完成POC，成功融资后再考虑成本优化

---
## 小结

**第2章核心要点**：

1. **LlamaParse核心优势**：
   - GenAI原生，语义理解强
   - 完美表格识别（Markdown格式）
   - 多栏布局、公式、图表全面支持
   - 专为RAG优化，零部署成本

2. **MinerU核心优势**：
   - SOTA性能（OmniDocBench超越GPT-4o）
   - 公式识别顶级（UniMER，准确率98%+，提速1400%）
   - 极速处理（10,000+ tokens/s）
   - 完全免费开源（AGPL-3.0）

3. **使用场景对比**：
   - 学术论文（公式密集） → **MinerU优势**（公式SOTA）
   - 财务报表（复杂表格） → **平手**（看具体需求）
   - 技术文档（语义理解） → **LlamaParse优势**
   - 法律合同（结构化条款） → **LlamaParse优势**

4. **选型决策**：
   - 无GPU + 快速开发 → **LlamaParse**
   - 有GPU + 大规模 + 隐私 → **MinerU**
   - 预算充足 + 语义理解 → **LlamaParse**
   - 成本敏感 + 公式密集 → **MinerU**
   - 文档类型多样 → **混合策略**

5. **成本优化策略**：
   - 简单文档 → 传统工具（免费）
   - 复杂文档 → MinerU（免费但需GPU）
   - 极复杂文档 → LlamaParse（付费但最准确）
   - 利用智能路由节省95%+成本
   - 分层降级确保容错

**下一章预告**：
第3章将介绍**Node Parser**，LlamaIndex的智能分块策略。
- **Document**：原始文档，从数据源加载
- **Node**：文档分块后的单元，用于索引和检索

```python
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

# 加载Document
documents = SimpleDirectoryReader("./data").load_data()
print(f"Documents: {len(documents)}个")

# 分块为Nodes
parser = SentenceSplitter(chunk_size=1024)
nodes = parser.get_nodes_from_documents(documents)
print(f"Nodes: {len(nodes)}个")
```

---

### 3.1.2 为什么需要Node Parser

**挑战**：
1. **Embedding限制**：模型有token限制
2. **检索精度**：大块降低相关性
3. **上下文窗口**：LLM处理有限

**Node Parser解决**：
- 智能分块（语义边界）
- 保留元数据（来源、位置）
- 支持多种分块策略

---

## 3.2 SentenceSplitter（推荐）

### 3.2.1 基础使用

```python
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader

# 加载文档
documents = SimpleDirectoryReader("./data").load_data()

# 创建SentenceSplitter
splitter = SentenceSplitter(
    chunk_size=1024,        # 每个chunk的token数
    chunk_overlap=20,       # chunk之间的重叠token数
    separator=" "           # 分隔符
)

# 分块
nodes = splitter.get_nodes_from_documents(documents)

print(f"生成{len(nodes)}个节点")
for node in nodes[:2]:
    print(f"\nNode ID: {node.node_id}")
    print(f"内容: {node.text[:200]}...")
    print(f"元数据: {node.metadata}")
```

---

### 3.2.2 参数调优

```python
from llama_index.core.node_parser import SentenceSplitter

# 配置1: 小chunk（精确检索）
small_splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50
)

# 配置2: 大chunk（保留上下文）
large_splitter = SentenceSplitter(
    chunk_size=2048,
    chunk_overlap=200
)

# 配置3: 自定义分隔符
custom_splitter = SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=100,
    separator="\n\n"  # 按段落分割
)

# 对比效果
small_nodes = small_splitter.get_nodes_from_documents(documents)
large_nodes = large_splitter.get_nodes_from_documents(documents)

print(f"小chunk: {len(small_nodes)}个")
print(f"大chunk: {len(large_nodes)}个")
```

**参数选择建议**：

| 场景 | chunk_size | chunk_overlap | 说明 |
|------|-----------|---------------|------|
| **精确检索** | 512-768 | 50-100 | 小chunk提高检索精度 |
| **保留上下文** | 1536-2048 | 150-200 | 大chunk保留更多上下文 |
| **通用场景** | 1024 | 100 | 平衡精度和上下文 |

---

## 3.3 其他Node Parser

### 3.3.1 SemanticSplitterNodeParser

**基于语义相似度分块**：

```python
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding

# 创建语义分块器
semantic_splitter = SemanticSplitterNodeParser(
    buffer_size=1,                          # 前后缓冲句子数
    embed_model=OpenAIEmbedding(),         # 使用Embedding模型
    breakpoint_percentile_threshold=95     # 相似度阈值（百分位）
)

nodes = semantic_splitter.get_nodes_from_documents(documents)

print(f"语义分块：{len(nodes)}个节点")
```

**工作原理**：
1. 对每个句子生成embedding
2. 计算相邻句子的余弦相似度
3. 在相似度低的地方分割（主题边界）

**优势**：
- ✅ 保留语义完整性
- ✅ 自动识别主题变化

**劣势**：
- ❌ 需要调用Embedding API（成本）
- ❌ 速度慢

---

### 3.3.2 MarkdownNodeParser

**专为Markdown优化**：

```python
from llama_index.core.node_parser import MarkdownNodeParser

# 适用于LlamaParse输出的Markdown
parser = MarkdownNodeParser()

# 假设documents来自LlamaParse
# documents = llamaparse.load_data("document.pdf")

nodes = parser.get_nodes_from_documents(documents)

# 查看节点元数据（包含Markdown结构）
for node in nodes[:3]:
    print(f"标题层级: {node.metadata.get('header_path', 'N/A')}")
    print(f"内容: {node.text[:150]}...\n")
```

**优势**：
- ✅ 保留Markdown结构
- ✅ 标题层级作为元数据
- ✅ 与LlamaParse完美配合

---

### 3.3.3 CodeSplitter

**专为代码文档优化**：

```python
from llama_index.core.node_parser import CodeSplitter

# 代码分块器
code_splitter = CodeSplitter(
    language="python",       # 编程语言
    chunk_lines=40,          # 每个chunk的行数
    chunk_lines_overlap=15,  # 重叠行数
    max_chars=1500          # 最大字符数
)

# 适用于代码文件
code_documents = SimpleDirectoryReader(
    "./code",
    required_exts=[".py"]
).load_data()

nodes = code_splitter.get_nodes_from_documents(code_documents)

print(f"代码分块：{len(nodes)}个节点")
```

---

### 3.3.4 HierarchicalNodeParser

**层级分块（多粒度）**：

```python
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes

# 创建层级分块器
hierarchical_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]  # 三个层级：粗、中、细
)

nodes = hierarchical_parser.get_nodes_from_documents(documents)
leaf_nodes = get_leaf_nodes(nodes)  # 获取最细粒度的节点

print(f"总节点数：{len(nodes)}个")
print(f"叶子节点数：{len(leaf_nodes)}个")

# 用于索引的是叶子节点
index = VectorStoreIndex(leaf_nodes)
```

**用途**：
- 粗粒度：文档级别检索
- 中粒度：段落级别检索
- 细粒度：句子级别检索

---

## 3.4 Metadata提取

### 3.4.1 自动提取元数据

```python
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
    SummaryExtractor,
    KeywordExtractor
)
from llama_index.llms.openai import OpenAI

# 创建元数据提取器
metadata_extractors = [
    TitleExtractor(llm=OpenAI(model="gpt-3.5-turbo")),              # 提取标题
    QuestionsAnsweredExtractor(llm=OpenAI(model="gpt-3.5-turbo")),  # 生成问题
    SummaryExtractor(llm=OpenAI(model="gpt-3.5-turbo")),            # 生成摘要
    KeywordExtractor(llm=OpenAI(model="gpt-3.5-turbo"))             # 提取关键词
]

# 创建带元数据提取的Node Parser
from llama_index.core.ingestion import IngestionPipeline

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=1024),
        *metadata_extractors
    ]
)

nodes = pipeline.run(documents=documents)

# 查看增强的元数据
for node in nodes[:2]:
    print(f"\n节点ID: {node.node_id}")
    print(f"标题: {node.metadata.get('document_title', 'N/A')}")
    print(f"摘要: {node.metadata.get('section_summary', 'N/A')[:100]}...")
    print(f"关键词: {node.metadata.get('excerpt_keywords', 'N/A')}")
```

---

### 3.4.2 自定义元数据

```python
from llama_index.core.node_parser import SentenceSplitter

# 手动添加元数据到Document
for doc in documents:
    doc.metadata["source_type"] = "technical_manual"
    doc.metadata["department"] = "engineering"
    doc.metadata["version"] = "v2.0"

# 分块时元数据会继承
splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)

# 验证元数据继承
for node in nodes[:2]:
    print(f"来源类型: {node.metadata.get('source_type')}")
    print(f"部门: {node.metadata.get('department')}")
    print(f"版本: {node.metadata.get('version')}\n")
```

---

## 3.5 分块质量评估

```python
def evaluate_nodes(nodes: list) -> dict:
    """评估节点分块质量"""
    lengths = [len(node.text) for node in nodes]

    stats = {
        'total_nodes': len(nodes),
        'avg_length': sum(lengths) / len(lengths) if lengths else 0,
        'min_length': min(lengths) if lengths else 0,
        'max_length': max(lengths) if lengths else 0,
    }

    # 检查分布
    too_small = sum(1 for l in lengths if l < 100)
    too_large = sum(1 for l in lengths if l > 3000)

    stats['too_small'] = too_small
    stats['too_large'] = too_large
    stats['quality'] = 'good' if (too_small + too_large) < len(nodes) * 0.1 else 'poor'

    return stats

# 使用
stats = evaluate_nodes(nodes)
print(f"总节点数：{stats['total_nodes']}")
print(f"平均长度：{stats['avg_length']:.0f} 字符")
print(f"质量评估：{stats['quality']}")
```

---

## 小结

**第3章核心要点**：

1. **Node Parser重要性**：
   - 将Document转换为Node（分块单元）
   - 控制索引粒度和检索精度
   - 保留元数据和上下文关系

2. **Parser选择**：
   - 通用场景 → SentenceSplitter
   - Markdown文档 → MarkdownNodeParser
   - 代码文档 → CodeSplitter
   - 语义完整性 → SemanticSplitterNodeParser
   - 多粒度检索 → HierarchicalNodeParser

3. **最佳实践**：
   - chunk_size: 512-2048（根据场景）
   - chunk_overlap: 10-20%的chunk_size
   - 使用元数据提取器增强检索
   - 评估分块质量并迭代优化

**下一章预告**：
第4章将介绍**多模态文档处理**，包括图片、表格、图表的智能提取。

---

# 第4章：多模态文档处理

## 4.1 图片提取与理解

### 4.1.1 基础图片提取

```python
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import ImageReader

# 使用ImageReader提取图片
image_reader = ImageReader()

# 方式1: 直接加载图片文件
image_docs = image_reader.load_data(file="diagram.png")

# 方式2: 从PDF中提取图片（需要配合其他工具）
from llama_parse import LlamaParse

parser = LlamaParse(
    api_key="llx-your-api-key",
    result_type="markdown",
    gpt4o_mode=True,  # 启用图片理解
    parsing_instruction="提取并描述所有图片和图表"
)

documents = parser.load_data("illustrated_document.pdf")

# LlamaParse会在Markdown中包含图片描述
for doc in documents:
    if "![" in doc.text:  # Markdown图片语法
        print("发现图片引用：")
        print(doc.text[:300])
```

---

### 4.1.2 多模态Embedding

**使用CLIP等多模态模型**：

```python
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.clip import ClipEmbedding
from llama_index.core.schema import ImageDocument

# 创建多模态Embedding模型
clip_embed = ClipEmbedding()

# 加载图片文档
image_documents = [
    ImageDocument(image_path="diagram1.png"),
    ImageDocument(image_path="chart1.png"),
]

# 构建多模态索引
index = VectorStoreIndex.from_documents(
    image_documents,
    embed_model=clip_embed
)

# 查询（支持文本查询图片）
query_engine = index.as_query_engine()
response = query_engine.query("系统架构图")
print(response.response)
```

---

## 4.2 表格处理

### 4.2.1 LlamaParse表格提取

**LlamaParse的表格优势**（已在第2章介绍）：

```python
from llama_parse import LlamaParse

parser = LlamaParse(
    api_key="llx-your-api-key",
    result_type="markdown",
    parsing_instruction="""
    重点提取表格：
    1. 保留完整结构（Markdown格式）
    2. 跨页表格合并
    3. 表头与数据关联
    """
)

documents = parser.load_data("tables_document.pdf")

# 提取Markdown表格
tables = []
for doc in documents:
    lines = doc.text.split("\n")
    table_lines = []
    in_table = False

    for line in lines:
        if "|" in line:
            in_table = True
            table_lines.append(line)
        elif in_table and not line.strip():
            # 表格结束
            tables.append("\n".join(table_lines))
            table_lines = []
            in_table = False

    if table_lines:
        tables.append("\n".join(table_lines))

print(f"提取到{len(tables)}个表格")
for i, table in enumerate(tables[:2], 1):
    print(f"\n表格{i}：")
    print(table)
```

---

### 4.2.2 表格转结构化数据

```python
import pandas as pd
from io import StringIO

def markdown_table_to_df(markdown_table: str) -> pd.DataFrame:
    """将Markdown表格转换为DataFrame"""
    lines = markdown_table.strip().split("\n")

    # 移除分隔线（第二行）
    if len(lines) >= 2 and "---" in lines[1]:
        lines.pop(1)

    # 转换为CSV格式
    csv_lines = []
    for line in lines:
        # 移除首尾的 |
        line = line.strip("|").strip()
        # 替换 | 为 ,
        csv_line = line.replace("|", ",")
        csv_lines.append(csv_line)

    csv_str = "\n".join(csv_lines)

    # 读取为DataFrame
    df = pd.read_csv(StringIO(csv_str))
    return df

# 使用
for i, table_md in enumerate(tables[:2], 1):
    df = markdown_table_to_df(table_md)
    print(f"\n表格{i}（DataFrame）：")
    print(df)

    # 可以进行数据分析
    if '收入' in df.columns:
        print(f"总收入：{df['收入'].sum()}")
```

---

## 4.3 图表理解（Charts & Diagrams）

### 4.3.1 使用GPT-4o理解图表

```python
from llama_parse import LlamaParse
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex

# 步骤1: 使用LlamaParse提取图表（带描述）
parser = LlamaParse(
    api_key="llx-your-api-key",
    result_type="markdown",
    gpt4o_mode=True,
    parsing_instruction="""
    对于图表：
    1. 识别图表类型（柱状图、折线图、饼图等）
    2. 提取图表标题
    3. 描述图表数据趋势
    4. 如果可能，将图表数据转换为表格
    """
)

documents = parser.load_data("charts_document.pdf")

# 步骤2: 构建索引
index = VectorStoreIndex.from_documents(documents)

# 步骤3: 查询图表信息
query_engine = index.as_query_engine(
    llm=OpenAI(model="gpt-4")
)

response = query_engine.query("销售额的趋势是什么？")
print(response.response)
```

---

### 4.3.2 专用图表工具（ChartReader）

```python
# 假设有一个专用的ChartReader（社区工具）
# 这是示例概念，实际需要根据具体工具调整

from llama_index.readers.file import ImageReader
from PIL import Image

# 读取图表图片
image_path = "sales_chart.png"
img = Image.open(image_path)

# 使用OCR + LLM理解图表
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-4o")  # GPT-4o支持视觉

# 将图片编码为base64
import base64
from io import BytesIO

buffered = BytesIO()
img.save(buffered, format="PNG")
img_str = base64.b64encode(buffered.getvalue()).decode()

# 查询图表内容
prompt = f"""
这是一张图表，请分析：
1. 图表类型
2. 数据趋势
3. 关键发现

![chart](data:image/png;base64,{img_str})
"""

# 注：实际使用需要支持图片输入的API
# response = llm.complete(prompt)
# print(response.text)
```

---

## 4.4 多模态RAG完整示例

### 4.4.1 构建多模态知识库

```python
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# 步骤1: 使用LlamaParse处理多模态PDF
parser = LlamaParse(
    api_key="llx-your-api-key",
    result_type="markdown",
    gpt4o_mode=True,  # 启用图片理解
    parsing_instruction="""
    处理所有多模态内容：
    1. 文本段落
    2. 表格（Markdown格式）
    3. 图片（生成描述）
    4. 图表（提取数据+描述趋势）
    """
)

documents = parser.load_data("multimodal_report.pdf")

# 步骤2: 分块
node_parser = MarkdownNodeParser()
nodes = node_parser.get_nodes_from_documents(documents)

# 步骤3: 构建索引
index = VectorStoreIndex(
    nodes=nodes,
    embed_model=OpenAIEmbedding()
)

# 步骤4: 创建查询引擎
query_engine = index.as_query_engine(
    llm=OpenAI(model="gpt-4"),
    similarity_top_k=5
)

# 步骤5: 多模态查询
queries = [
    "报告中的主要图表显示了什么趋势？",
    "财务数据表格中的总收入是多少？",
    "系统架构图包含哪些组件？"
]

for query in queries:
    print(f"\n查询：{query}")
    response = query_engine.query(query)
    print(f"回答：{response.response}\n")

    # 查看检索到的源节点
    print("来源节点：")
    for i, node in enumerate(response.source_nodes[:2], 1):
        print(f"{i}. {node.text[:150]}...")
```

---

## 小结

**第4章核心要点**：

1. **多模态处理能力**：
   - 图片提取与描述（LlamaParse + GPT-4o）
   - 表格识别与结构化（Markdown → DataFrame）
   - 图表理解（类型识别 + 数据提取）

2. **工具选择**：
   - 图片理解 → LlamaParse (gpt4o_mode) + CLIP
   - 表格提取 → LlamaParse (Markdown格式)
   - 图表分析 → GPT-4o视觉能力

3. **最佳实践**：
   - 使用LlamaParse的gpt4o_mode处理复杂多模态文档
   - 表格转换为结构化数据便于分析
   - 图表描述与原始数据关联
   - 多模态RAG提升查询覆盖范围

**下一章预告**：
第5章将整合所有技术，构建**生产级文档处理Pipeline**。

---

# 第5章：生产级文档处理Pipeline

## 5.1 Pipeline设计

### 5.1.1 完整流程

```
文档输入
  ↓
类型检测（PDF, DOCX, 图片...）
  ↓
复杂度评估（简单 vs 复杂）
  ↓
选择处理策略
  ├── 简单PDF → PDFReader
  ├── 复杂PDF → LlamaParse
  ├── 图片 → ImageReader + GPT-4o
  └── 其他格式 → SimpleDirectoryReader
  ↓
Node Parser分块
  ↓
元数据提取（可选）
  ↓
向量化 + 存储
  ↓
RAG系统
```

---

### 5.1.2 完整实现

```python
from typing import List, Dict
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.readers.file import PDFReader, PyMuPDFReader
from llama_parse import LlamaParse
from llama_index.core.node_parser import SentenceSplitter, MarkdownNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """生产级文档处理器（LlamaIndex版）"""

    def __init__(self, llamaparse_api_key: str = None, use_llamaparse: bool = True):
        self.use_llamaparse = use_llamaparse
        self.llamaparse_api_key = llamaparse_api_key
        self.embeddings = OpenAIEmbedding()

    def process_document(self, file_path: str) -> List[Document]:
        """处理单个文档（自动选择策略）"""
        # 步骤1: 检测文件类型
        file_type = self._detect_file_type(file_path)
        logger.info(f"文件类型：{file_type}")

        # 步骤2: 选择处理策略
        if file_type == 'PDF':
            return self._process_pdf(file_path)
        elif file_type in ['DOCX', 'DOC']:
            return self._process_docx(file_path)
        elif file_type in ['PNG', 'JPG', 'JPEG']:
            return self._process_image(file_path)
        else:
            # 通用处理
            return self._process_generic(file_path)

    def _detect_file_type(self, file_path: str) -> str:
        """检测文件类型"""
        suffix = Path(file_path).suffix.lower()
        type_map = {
            '.pdf': 'PDF',
            '.docx': 'DOCX',
            '.doc': 'DOC',
            '.png': 'PNG',
            '.jpg': 'JPG',
            '.jpeg': 'JPEG',
            '.txt': 'TXT',
            '.md': 'MD'
        }
        return type_map.get(suffix, 'UNKNOWN')

    def _assess_pdf_complexity(self, pdf_path: str) -> str:
        """评估PDF复杂度"""
        # 简单启发式：文件大小
        file_size = Path(pdf_path).stat().st_size

        if file_size > 5 * 1024 * 1024:  # > 5MB
            return 'complex'

        # 可以添加更多启发式（页数、是否有图片等）
        return 'simple'

    def _process_pdf(self, pdf_path: str) -> List[Document]:
        """处理PDF（带智能降级）"""
        complexity = self._assess_pdf_complexity(pdf_path)

        # 策略1: 复杂PDF使用LlamaParse
        if complexity == 'complex' and self.use_llamaparse:
            try:
                logger.info("复杂PDF，使用LlamaParse...")
                parser = LlamaParse(
                    api_key=self.llamaparse_api_key,
                    result_type="markdown",
                    gpt4o_mode=True
                )
                docs = parser.load_data(pdf_path)
                logger.info(f"✅ LlamaParse成功，提取{len(docs)}个文档")
                return docs
            except Exception as e:
                logger.warning(f"LlamaParse失败: {e}，降级到传统工具")

        # 策略2: 简单PDF使用PyMuPDFReader（快速）
        try:
            logger.info("使用PyMuPDFReader...")
            reader = PyMuPDFReader()
            docs = reader.load_data(file=pdf_path)
            logger.info(f"✅ PyMuPDFReader成功，提取{len(docs)}页")
            return docs
        except Exception as e:
            logger.warning(f"PyMuPDFReader失败: {e}")

        # 策略3: 基础PDFReader
        try:
            logger.info("使用PDFReader...")
            reader = PDFReader()
            docs = reader.load_data(file=pdf_path)
            logger.info(f"✅ PDFReader成功，提取{len(docs)}页")
            return docs
        except Exception as e:
            logger.error(f"所有PDF处理策略失败: {e}")
            return []

    def _process_docx(self, file_path: str) -> List[Document]:
        """处理DOCX"""
        from llama_index.readers.file import DocxReader
        reader = DocxReader()
        return reader.load_data(file=file_path)

    def _process_image(self, file_path: str) -> List[Document]:
        """处理图片"""
        from llama_index.readers.file import ImageReader
        reader = ImageReader()
        return reader.load_data(file=file_path)

    def _process_generic(self, file_path: str) -> List[Document]:
        """通用处理"""
        reader = SimpleDirectoryReader(input_files=[file_path])
        return reader.load_data()

    def build_index(
        self,
        documents: List[Document],
        chunk_size: int = 1024,
        chunk_overlap: int = 100,
        use_markdown_parser: bool = False
    ) -> VectorStoreIndex:
        """构建索引"""
        # 步骤1: 选择Node Parser
        if use_markdown_parser:
            node_parser = MarkdownNodeParser()
        else:
            node_parser = SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

        # 步骤2: 分块
        nodes = node_parser.get_nodes_from_documents(documents)
        logger.info(f"生成{len(nodes)}个节点")

        # 步骤3: 构建索引
        index = VectorStoreIndex(
            nodes=nodes,
            embed_model=self.embeddings
        )

        return index

# 使用示例
processor = DocumentProcessor(
    llamaparse_api_key="llx-your-api-key",
    use_llamaparse=True
)

# 处理单个文档
docs = processor.process_document("./complex_document.pdf")
print(f"提取{len(docs)}个文档")

# 构建索引
index = processor.build_index(docs, use_markdown_parser=True)
print("索引构建完成")
```

---

## 5.2 批量处理与质量控制

### 5.2.1 批量处理

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def batch_process_directory(
    processor: DocumentProcessor,
    directory: str,
    max_workers: int = 4
) -> List[Document]:
    """批量处理目录下所有文档"""
    all_docs = []
    files = list(Path(directory).rglob("*"))

    # 过滤支持的格式
    supported_exts = ['.pdf', '.docx', '.txt', '.md', '.png', '.jpg']
    files = [f for f in files if f.suffix.lower() in supported_exts]

    logger.info(f"找到{len(files)}个文件")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(processor.process_document, str(f)): f
            for f in files
        }

        for future in as_completed(futures):
            file = futures[future]
            try:
                docs = future.result()
                all_docs.extend(docs)
                logger.info(f"✅ {file.name}: {len(docs)}个文档")
            except Exception as e:
                logger.error(f"❌ {file.name}: {e}")

    return all_docs

# 使用
all_documents = batch_process_directory(processor, "./knowledge_base", max_workers=4)
print(f"\n总计处理：{len(all_documents)}个文档")
```

---

### 5.2.2 质量检测

```python
def assess_document_quality(documents: List[Document]) -> Dict:
    """评估文档提取质量"""
    total_text = "".join([doc.text for doc in documents])

    assessment = {
        'total_docs': len(documents),
        'total_chars': len(total_text),
        'avg_doc_length': len(total_text) / len(documents) if documents else 0,
        'has_content': len(total_text) > 100,
        'quality_score': 0.0
    }

    # 计算质量分数
    score = 0

    # 有足够内容 (+40分)
    if assessment['total_chars'] > 1000:
        score += 40
    elif assessment['total_chars'] > 100:
        score += 20

    # 平均文档长度合理 (+30分)
    avg_len = assessment['avg_doc_length']
    if 200 < avg_len < 2000:
        score += 30
    elif 100 < avg_len < 5000:
        score += 15

    # 文档数量合理 (+30分)
    if 5 < len(documents) < 100:
        score += 30
    elif len(documents) > 0:
        score += 15

    assessment['quality_score'] = score

    # 评级
    if score >= 80:
        assessment['rating'] = '优秀'
    elif score >= 60:
        assessment['rating'] = '良好'
    elif score >= 40:
        assessment['rating'] = '一般'
    else:
        assessment['rating'] = '差'

    return assessment

# 使用
quality = assess_document_quality(docs)
print(f"质量评分：{quality['quality_score']}/100 ({quality['rating']})")
print(f"文档数：{quality['total_docs']}")
print(f"总字符数：{quality['total_chars']}")
```

---

## 5.3 完整RAG系统

```python
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI

# 步骤1: 处理文档目录
processor = DocumentProcessor(
    llamaparse_api_key="llx-your-api-key",
    use_llamaparse=True
)

all_docs = []
for file in Path("./knowledge_base").rglob("*.pdf"):
    logger.info(f"\n处理：{file.name}")
    docs = processor.process_document(str(file))

    # 质量检测
    quality = assess_document_quality(docs)
    logger.info(f"质量：{quality['rating']} ({quality['quality_score']}/100)")

    if quality['quality_score'] >= 40:
        all_docs.extend(docs)
    else:
        logger.warning(f"质量过低，跳过")

# 步骤2: 构建索引
index = processor.build_index(all_docs, use_markdown_parser=True)
logger.info(f"索引构建完成，共{len(all_docs)}个文档")

# 步骤3: 创建查询引擎
query_engine = index.as_query_engine(
    llm=OpenAI(model="gpt-4"),
    similarity_top_k=5
)

# 步骤4: 查询
queries = [
    "产品的技术规格是什么？",
    "财报中的主要数据趋势如何？",
    "系统架构包含哪些组件？"
]

for query in queries:
    print(f"\n查询：{query}")
    response = query_engine.query(query)
    print(f"回答：{response.response}\n")

    # 查看源节点
    print("来源：")
    for i, node in enumerate(response.source_nodes[:2], 1):
        source = node.metadata.get('file_name', 'unknown')
        print(f"{i}. {source}: {node.text[:100]}...")
```

---

## 5.4 成本优化策略

### 5.4.1 混合策略

```python
class CostOptimizedProcessor(DocumentProcessor):
    """成本优化的文档处理器"""

    def __init__(self, llamaparse_api_key: str, monthly_budget: float = 100.0):
        super().__init__(llamaparse_api_key, use_llamaparse=True)
        self.monthly_budget = monthly_budget
        self.llamaparse_usage = 0.0  # 本月使用额度

    def _process_pdf(self, pdf_path: str) -> List[Document]:
        """成本优化的PDF处理"""
        complexity = self._assess_pdf_complexity(pdf_path)
        file_size_mb = Path(pdf_path).stat().st_size / (1024 * 1024)

        # 估算LlamaParse成本（假设$0.003/页，20页/MB）
        estimated_pages = file_size_mb * 20
        estimated_cost = estimated_pages * 0.003

        # 预算检查
        if (complexity == 'complex' and
            self.llamaparse_usage + estimated_cost < self.monthly_budget):
            try:
                logger.info(f"使用LlamaParse（预估成本：${estimated_cost:.2f}）")
                parser = LlamaParse(api_key=self.llamaparse_api_key)
                docs = parser.load_data(pdf_path)
                self.llamaparse_usage += estimated_cost
                logger.info(f"本月已使用：${self.llamaparse_usage:.2f}/{self.monthly_budget}")
                return docs
            except Exception as e:
                logger.warning(f"LlamaParse失败，降级")

        # 使用免费工具
        logger.info("使用免费工具（PyMuPDFReader）")
        reader = PyMuPDFReader()
        return reader.load_data(file=pdf_path)

# 使用
cost_processor = CostOptimizedProcessor(
    llamaparse_api_key="llx-your-api-key",
    monthly_budget=50.0  # $50/月预算
)

docs = cost_processor.process_document("document.pdf")
print(f"本月LlamaParse使用：${cost_processor.llamaparse_usage:.2f}")
```

---

### 5.4.2 缓存策略

```python
import hashlib
import pickle
from pathlib import Path

class CachedDocumentProcessor(DocumentProcessor):
    """带缓存的文档处理器"""

    def __init__(self, llamaparse_api_key: str, cache_dir: str = "./doc_cache"):
        super().__init__(llamaparse_api_key)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_key(self, file_path: str) -> str:
        """生成缓存key（基于文件内容hash）"""
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash

    def process_document(self, file_path: str) -> List[Document]:
        """带缓存的文档处理"""
        cache_key = self._get_cache_key(file_path)
        cache_path = self.cache_dir / f"{cache_key}.pkl"

        # 检查缓存
        if cache_path.exists():
            logger.info(f"从缓存加载：{file_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        # 处理文档
        docs = super().process_document(file_path)

        # 保存缓存
        with open(cache_path, 'wb') as f:
            pickle.dump(docs, f)
        logger.info(f"缓存已保存：{cache_path}")

        return docs

# 使用
cached_processor = CachedDocumentProcessor(
    llamaparse_api_key="llx-your-api-key",
    cache_dir="./doc_cache"
)

# 第一次处理（慢）
docs1 = cached_processor.process_document("document.pdf")

# 第二次处理（快，从缓存读取）
docs2 = cached_processor.process_document("document.pdf")
```

---

## 全篇总结

**第十一篇（LlamaIndex篇）涵盖技术**：

| 章节 | 核心技术 | 适用场景 |
|------|---------|---------|
| 第1章 | SimpleDirectoryReader, LlamaHub | 快速入门、多数据源 |
| 第2章 | LlamaParse | 复杂PDF、学术论文、财报 |
| 第3章 | Node Parser（Sentence, Semantic, Markdown） | 智能分块 |
| 第4章 | 多模态处理（图片、表格、图表） | 富媒体文档 |
| 第5章 | 生产级Pipeline | 企业知识库 |

---

## 思考与练习

### 练习1：LlamaParse vs 传统工具对比

选择3种不同复杂度的PDF，对比：
1. PDFReader
2. PyMuPDFReader
3. LlamaParse

测试指标：准确率、表格完整性、处理时间、成本

### 练习2：构建多模态RAG系统

使用LlamaParse构建支持：
1. 文本检索
2. 表格查询
3. 图表分析

的完整RAG系统

### 练习3：成本优化实验

实现混合策略：
1. 简单文档用免费工具
2. 复杂文档用LlamaParse
3. 设置月度预算限制
4. 使用缓存减少重复处理

---

## 参考资源

**官方文档**：
- [LlamaIndex官方文档](https://developers.llamaindex.ai/)
- [LlamaParse文档](https://developers.llamaindex.ai/python/framework/llama_cloud/llama_parse/)
- [LlamaHub](https://llamahub.ai/)

**关键链接**：
- [LlamaCloud注册](https://cloud.llamaindex.ai/)
- [LlamaIndex GitHub](https://github.com/run-llama/llama_index)

---

**第十一篇（LlamaIndex篇）完成**！

你已经掌握了LlamaIndex的文档处理完整技术栈：
- ✅ SimpleDirectoryReader快速入门
- ✅ LlamaHub 700+数据加载器生态
- ✅ LlamaParse - GenAI原生PDF解析（核心亮点）
- ✅ Node Parser智能分块策略
- ✅ 多模态文档处理（图片、表格、图表）
- ✅ 生产级Pipeline与成本优化

**LlamaIndex vs LangChain文档处理对比**：

| 维度 | LlamaIndex | LangChain |
|------|-----------|-----------|
| **核心优势** | LlamaParse（GenAI原生解析） | Unstructured.io（多格式统一） |
| **表格处理** | ⭐⭐⭐⭐⭐ LlamaParse完美Markdown | ⭐⭐⭐⭐ PDFPlumber表格提取 |
| **图片理解** | ⭐⭐⭐⭐⭐ GPT-4o多模态 | ⭐⭐⭐ OCR集成 |
| **学习曲线** | ⭐⭐⭐⭐ 易上手（SimpleDirectoryReader） | ⭐⭐⭐ 中等（Document Loaders） |
| **成本** | LlamaParse付费（高质量） | 主要免费（Unstructured有付费版） |
| **适用场景** | 复杂文档、学术论文、财报 | 通用文档处理、企业知识库 |

**选择建议**：
- **高质量要求** → LlamaIndex + LlamaParse
- **成本敏感** → LangChain + Unstructured.io
- **混合使用** → 简单文档用LangChain，复杂文档用LlamaParse

**下一篇预告**：
第十二篇将聚焦**提示工程与上下文优化**，提升RAG系统的生成质量和成本效率。
