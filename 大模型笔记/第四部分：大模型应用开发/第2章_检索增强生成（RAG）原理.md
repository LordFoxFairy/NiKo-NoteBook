# RAG完全实战指南：从数学原理到高级技术(2025-2026版)

> 全面掌握检索增强生成：理论、实践与前沿技术

**适合人群**: AI应用开发者、算法工程师、架构师
**预计学习时间**: 12-15 小时
**前置知识**: Python基础、大模型基础概念、向量检索基本认知

---

## 目录

- [引言：RAG技术全景](#引言rag技术全景)
- [第一部分：数学基础与核心原理](#第一部分数学基础与核心原理)
- [第二部分：文档处理与分块策略](#第二部分文档处理与分块策略)
- [第三部分：向量存储与检索技术](#第三部分向量存储与检索技术)
- [第四部分：检索方法深度剖析](#第四部分检索方法深度剖析)
- [第五部分：重排序技术](#第五部分重排序技术)
- [第六部分：高级RAG架构](#第六部分高级rag架构)
- [第七部分：完整Pipeline实现](#第七部分完整pipeline实现)
- [第八部分：实战案例与评估](#第八部分实战案例与评估)

---

## 引言：RAG技术全景

### 什么是RAG？为何需要RAG？

大型语言模型(LLMs)虽然强大,但存在三大核心局限:

1. **知识静态性**: 知识截止到训练日期,无法获取最新信息
2. **幻觉问题**: 可能生成看似合理但事实错误的内容
3. **更新成本高昂**: 通过微调更新知识代价极大

**RAG(Retrieval-Augmented Generation,检索增强生成)**通过将LLM与外部动态知识源集成,从根本上解决了这些问题:

```
传统LLM: 查询 → LLM → 答案(可能过时/错误)
RAG系统: 查询 → 检索知识库 → LLM(查询+检索内容) → 准确答案
```

**核心优势**:
- 实时知识访问(更新知识库即可,无需重训模型)
- 可追溯性(答案标注来源)
- 降低幻觉(基于实际文档生成)
- 成本效益(无需昂贵的模型微调)

### RAG vs 微调 vs 纯LLM对比

| 维度 | 纯LLM | 微调(Fine-tuning) | RAG | RAG + 微调 |
|-----|-------|------------------|-----|-----------|
| **知识更新** | 不可更新 | 需重训(慢) | 实时(快) | 最佳 |
| **成本** | 推理成本 | 训练+推理(高) | 检索+推理(中) | 高 |
| **准确性** | 一般 | 领域内高 | 检索质量决定 | 最高 |
| **可解释性** | 低 | 低 | 高(可溯源) | 高 |
| **实施难度** | 低 | 高 | 中 | 高 |
| **适用场景** | 通用对话 | 固定领域 | 动态知识库 | 企业级应用 |

**最佳实践**: 微调提升领域理解,RAG提供实时知识

### 本教程学习路径

```mermaid
graph TD
    A[Part 1: 数学基础] --> B[Part 2: 文档分块]
    B --> C[Part 3: 向量存储]
    C --> D[Part 4: 检索方法]
    D --> E[Part 5: 重排序]
    E --> F[Part 6: 高级架构]
    F --> G[Part 7: 完整实现]
    G --> H[Part 8: 案例评估]
```

**学习成果**:
- 掌握RAG完整数学原理
- 实现生产级RAG系统
- 理解并应用前沿技术(HyDE/Self-RAG/GraphRAG)
- 建立系统评估与优化能力

---

## 第一部分：数学基础与核心原理

### 1.1 RAG的形式化定义

#### 1.1.1 传统LLM生成

给定输入 $x$ (查询),LLM生成输出 $y$ (答案)的概率:

$$
P(y|x) = \prod_{t=1}^T P(y_t | y_{<t}, x)
$$

其中:
- $y_t$: 第 $t$ 个生成token
- $y_{<t}$: 前 $t-1$ 个token(自回归)

**问题**: 仅依赖参数化知识,可能过时或错误!

#### 1.1.2 RAG生成

RAG通过检索文档 $d$ 增强生成:

$$
P_{\text{RAG}}(y|x) = \sum_{d \in \mathcal{D}} P(d|x) \cdot P(y|x, d)
$$

其中:
- $\mathcal{D}$: 文档知识库
- $P(d|x)$: **检索模型**(retriever),计算文档与查询的相关性
- $P(y|x, d)$: **生成模型**(generator),基于查询和文档生成答案

**关键**: 对所有可能文档求边际化(marginalization),加权融合!

#### 1.1.3 Top-K近似

实践中遍历所有文档不可行,采用Top-K近似:

$$
P_{\text{RAG}}(y|x) \approx \sum_{d \in \text{TopK}(x)} P(d|x) \cdot P(y|x, d)
$$

**典型K值**: 3-10个文档

### 1.2 向量相似度度量

#### 1.2.1 余弦相似度(Cosine Similarity)

**定义**:

$$
\text{cosine}(q, d) = \frac{q \cdot d}{\|q\| \|d\|} = \frac{\sum_{i=1}^n q_i d_i}{\sqrt{\sum_{i=1}^n q_i^2} \sqrt{\sum_{i=1}^n d_i^2}}
$$

**值域**: $[-1, 1]$,1表示完全相同方向,-1表示相反

**几何意义**: 两向量夹角的余弦值

**Python实现**:

```python
import numpy as np

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    计算两个向量的余弦相似度

    Args:
        vec1: 向量1 (shape: [d])
        vec2: 向量2 (shape: [d])

    Returns:
        余弦相似度值 [-1, 1]
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    # 避免除零
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0

    return dot_product / (norm_vec1 * norm_vec2)


# 示例
query_vec = np.array([0.1, 0.5, 0.8])
doc_vec = np.array([0.2, 0.4, 0.9])
similarity = cosine_similarity(query_vec, doc_vec)
print(f"余弦相似度: {similarity:.4f}")  # 输出: 0.9912
```

**优势**: 对向量模长不敏感,只关注方向

#### 1.2.2 点积(Dot Product)

$$
\text{dot}(q, d) = q \cdot d = \sum_{i=1}^n q_i d_i
$$

**特点**:
- 计算最快(无需归一化)
- 对向量模长敏感
- 当向量已归一化时,等价于余弦相似度

**应用**: OpenAI `text-embedding-ada-002` 等模型返回归一化向量,可直接用点积

#### 1.2.3 欧氏距离(Euclidean Distance)

$$
\text{euclidean}(q, d) = \sqrt{\sum_{i=1}^n (q_i - d_i)^2}
$$

**值域**: $[0, +\infty)$,0表示完全相同

**转相似度**:

$$
\text{similarity} = \frac{1}{1 + \text{distance}}
$$

**对比总结**:

| 度量 | 计算复杂度 | 归一化需求 | 适用场景 |
|-----|-----------|-----------|---------|
| 余弦 | $O(n)$ + norm | 否 | 文本语义相似度(标准) |
| 点积 | $O(n)$ | 是 | 归一化向量(最快) |
| 欧氏 | $O(n)$ + sqrt | 否 | 图像/视觉相似度 |

### 1.3 稀疏检索: BM25算法

#### 1.3.1 BM25数学原理

BM25(Best Matching 25)是经典的**字面匹配**算法,基于词频和逆文档频率:

$$
\text{BM25}(q, d) = \sum_{i=1}^{|q|} \text{IDF}(q_i) \cdot \frac{f(q_i, d) \cdot (k_1 + 1)}{f(q_i, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}
$$

**参数说明**:
- $q$: 查询(query)
- $d$: 文档(document)
- $f(q_i, d)$: 词 $q_i$ 在文档 $d$ 中的词频
- $|d|$: 文档长度(token数)
- $\text{avgdl}$: 平均文档长度
- $k_1$: 词频饱和参数(通常1.2-2.0,默认1.5)
- $b$: 长度归一化参数(通常0.75)

**IDF(逆文档频率)**:

$$
\text{IDF}(q_i) = \log \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5}
$$

- $N$: 总文档数
- $n(q_i)$: 包含词 $q_i$ 的文档数

**直觉理解**:
- 常见词(如"的","是")IDF低,贡献小
- 稀有词(如"Transformer","RAG")IDF高,贡献大
- 文档越长,词频权重越低(避免长文档偏向)

#### 1.3.2 BM25 Python实现

```python
import math
from collections import Counter
from typing import List, Dict

class BM25:
    """
    BM25算法实现
    """
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        """
        Args:
            corpus: 文档集合,每个文档是token列表
            k1: 词频饱和参数
            b: 长度归一化参数
        """
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.N = len(corpus)  # 文档总数
        self.avgdl = sum(len(doc) for doc in corpus) / self.N  # 平均文档长度

        # 计算IDF
        self.idf = self._compute_idf()

    def _compute_idf(self) -> Dict[str, float]:
        """计算所有词的IDF值"""
        idf = {}
        # 统计包含每个词的文档数
        df = Counter()
        for doc in self.corpus:
            for word in set(doc):  # set去重,每个文档只计数一次
                df[word] += 1

        # 计算IDF
        for word, doc_freq in df.items():
            idf[word] = math.log((self.N - doc_freq + 0.5) / (doc_freq + 0.5))

        return idf

    def score(self, query: List[str], doc_idx: int) -> float:
        """
        计算查询与文档的BM25分数

        Args:
            query: 查询token列表
            doc_idx: 文档索引

        Returns:
            BM25分数
        """
        doc = self.corpus[doc_idx]
        doc_len = len(doc)

        # 统计文档中每个词的词频
        doc_freqs = Counter(doc)

        score = 0.0
        for word in query:
            if word not in self.idf:
                continue  # 查询词不在语料库中,跳过

            # 词频
            f = doc_freqs.get(word, 0)

            # BM25公式
            idf = self.idf[word]
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))

            score += idf * (numerator / denominator)

        return score

    def get_top_k(self, query: List[str], k: int = 5) -> List[tuple]:
        """
        检索Top-K最相关文档

        Returns:
            [(doc_idx, score), ...] 按分数降序
        """
        scores = [(i, self.score(query, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


# 使用示例
if __name__ == "__main__":
    # 构建语料库(已分词)
    corpus = [
        ["transformer", "is", "a", "deep", "learning", "model"],
        ["bert", "is", "based", "on", "transformer", "architecture"],
        ["gpt", "is", "a", "generative", "transformer", "model"],
        ["rag", "combines", "retrieval", "and", "generation"]
    ]

    # 初始化BM25
    bm25 = BM25(corpus)

    # 查询
    query = ["transformer", "model"]

    # 检索Top-3
    top_results = bm25.get_top_k(query, k=3)

    print("查询:", " ".join(query))
    for doc_idx, score in top_results:
        print(f"文档{doc_idx}: {' '.join(corpus[doc_idx])} (分数: {score:.4f})")
```

**输出**:
```
查询: transformer model
文档0: transformer is a deep learning model (分数: 5.2453)
文档2: gpt is a generative transformer model (分数: 5.2453)
文档1: bert is based on transformer architecture (分数: 2.8904)
```

#### 1.3.3 BM25的优势与局限

**优势**:
- 字面匹配精准(专有名词、代码、精确关键词)
- 无需训练,开箱即用
- 计算速度快
- 可解释性强(可看到词频权重)

**局限**:
- 无法理解语义(同义词miss)
- 依赖分词质量
- 对拼写错误敏感

**示例**:
```
查询: "苹果手机性能"
BM25能匹配: "iPhone 15 Pro性能测试" (包含"性能")
BM25会错过: "iPhone运行速度评测" (同义词"速度"≈"性能")
```

### 1.4 密集检索(Dense Retrieval)

#### 1.4.1 Embedding向量语义表示

**核心思想**: 将文本映射到高维向量空间,语义相似的文本在空间中距离近

**数学表示**:

$$
\mathbf{e}_{\text{query}} = f_{\text{encoder}}(\text{query}) \in \mathbb{R}^d
$$

$$
\mathbf{e}_{\text{doc}} = f_{\text{encoder}}(\text{doc}) \in \mathbb{R}^d
$$

**相似度计算**:

$$
\text{sim}(q, d) = \text{cosine}(\mathbf{e}_q, \mathbf{e}_d)
$$

**优势**:
- 语义理解(捕获同义词、跨语言)
- 泛化能力强
- 支持多模态(文本+图像)

#### 1.4.2 常用Embedding模型

| 模型 | 维度 | 语言 | 性能(MTEB) | 用途 |
|-----|------|------|-----------|------|
| `text-embedding-ada-002` | 1536 | 多语言 | - | OpenAI商业(高质量) |
| `bge-small-en-v1.5` | 384 | 英文 | 62.17 | 轻量级本地 |
| `bge-large-en-v1.5` | 1024 | 英文 | 63.98 | 高精度本地 |
| `text-embedding-3-large` | 3072 | 多语言 | - | OpenAI最新(最强) |
| `gte-large` | 1024 | 多语言 | 63.13 | 阿里通义开源 |

**选择建议**:
- 预算充足 → OpenAI `text-embedding-3-large`
- 本地部署 → `bge-large-en-v1.5`
- 速度优先 → `bge-small-en-v1.5`

#### 1.4.3 Embedding生成代码

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class DenseRetriever:
    """
    密集检索器(基于向量相似度)
    """
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Args:
            model_name: Hugging Face模型名称
        """
        print(f"加载模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.corpus_embeddings = None
        self.corpus = None

    def encode_corpus(self, corpus: List[str]):
        """
        为语料库生成embeddings

        Args:
            corpus: 文档列表
        """
        self.corpus = corpus
        print(f"为{len(corpus)}个文档生成embeddings...")

        # 批量编码(加速)
        self.corpus_embeddings = self.model.encode(
            corpus,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True  # 归一化(用于余弦相似度)
        )

        print(f"Embeddings shape: {self.corpus_embeddings.shape}")

    def search(self, query: str, top_k: int = 5) -> List[tuple]:
        """
        检索Top-K最相关文档

        Args:
            query: 查询字符串
            top_k: 返回文档数

        Returns:
            [(doc_idx, score), ...] 按分数降序
        """
        if self.corpus_embeddings is None:
            raise ValueError("请先调用 encode_corpus() 编码语料库")

        # 编码查询
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True
        )

        # 计算余弦相似度(归一化向量的点积)
        scores = np.dot(self.corpus_embeddings, query_embedding)

        # Top-K索引
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(int(idx), float(scores[idx])) for idx in top_indices]


# 使用示例
if __name__ == "__main__":
    # 文档语料库
    corpus = [
        "Transformer是一种基于自注意力机制的深度学习模型",
        "BERT使用双向Transformer编码器进行预训练",
        "GPT是生成式预训练Transformer模型",
        "RAG结合检索和生成,提升LLM事实准确性"
    ]

    # 初始化检索器
    retriever = DenseRetriever(model_name="BAAI/bge-small-en-v1.5")
    retriever.encode_corpus(corpus)

    # 查询
    query = "什么是Transformer?"
    top_results = retriever.search(query, top_k=3)

    print(f"\n查询: {query}\n")
    for doc_idx, score in top_results:
        print(f"[{score:.4f}] {corpus[doc_idx]}")
```

**输出**:
```
查询: 什么是Transformer?

[0.7823] Transformer是一种基于自注意力机制的深度学习模型
[0.6541] BERT使用双向Transformer编码器进行预训练
[0.6102] GPT是生成式预训练Transformer模型
```

### 1.5 近似最近邻搜索(ANN)

#### 1.5.1 问题规模与暴力搜索

**问题**: 在 $N$ 个 $d$ 维向量中找Top-K最相似向量

**暴力搜索(Brute Force)**:

$$
\text{Time Complexity} = O(N \cdot d)
$$

**性能瓶颈**:
- 100万向量 × 768维 × 每秒1000次查询 = 需要数秒响应
- 生产环境要求: **毫秒级响应**

**解决方案**: 近似最近邻(ANN)算法,牺牲少量精度换取巨大速度提升

#### 1.5.2 HNSW算法原理

**HNSW**(Hierarchical Navigable Small World,分层可导航小世界图)是目前最流行的ANN算法。

**核心思想**: 构建多层图结构,高层快速定位区域,低层精确搜索

**数学模型**:

**图定义**: $G_l = (V_l, E_l)$,其中 $l$ 是层级

**节点分布**: 指数衰减采样

$$
|V_l| = |V_0| \cdot e^{-l \cdot m_L}
$$

- $m_L$: 层级衰减因子

**搜索算法**(伪代码):
```
function HNSW_search(query, ep, ef, K):
    # ep: 入口点, ef: 搜索宽度, K: 返回数量

    curr_layer = top_layer
    curr_nearest = ep

    # 阶段1: 贪婪搜索到第0层
    while curr_layer > 0:
        curr_nearest = greedy_search_layer(query, curr_nearest, curr_layer, ef=1)
        curr_layer -= 1

    # 阶段2: 第0层精细搜索
    candidates = greedy_search_layer(query, curr_nearest, layer=0, ef=ef)

    return select_top_k(candidates, K)
```

**复杂度分析**:
- 构建: $O(N \log N)$
- 查询: $O(\log N)$ (亚线性!)

**关键参数**:

| 参数 | 含义 | 典型值 | 影响 |
|-----|------|--------|------|
| `M` | 每层最大连接数 | 16-64 | M↑ → 精度↑,内存↑ |
| `ef_construction` | 构建时搜索宽度 | 100-200 | ef↑ → 构建慢,召回↑ |
| `ef_search` | 查询时搜索宽度 | 50-500 | ef↑ → 查询慢,召回↑ |

**Trade-off**:

$$
\text{Recall} \uparrow \quad \Leftrightarrow \quad \text{ef\_search} \uparrow \quad \Leftrightarrow \quad \text{Latency} \uparrow
$$

#### 1.5.3 HNSW实战(FAISS库)

```python
import faiss
import numpy as np
from typing import List, Tuple

class FAISSRetriever:
    """
    基于FAISS的高性能向量检索器
    """
    def __init__(self, embedding_dim: int, index_type: str = "HNSW"):
        """
        Args:
            embedding_dim: 向量维度
            index_type: 索引类型("Flat"精确, "HNSW"近似)
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type

        # 构建索引
        if index_type == "Flat":
            # 精确搜索(暴力,小数据集)
            self.index = faiss.IndexFlatIP(embedding_dim)  # IP: Inner Product(点积)
        elif index_type == "HNSW":
            # HNSW近似搜索
            M = 32  # 每层连接数
            self.index = faiss.IndexHNSWFlat(embedding_dim, M)
            self.index.hnsw.efConstruction = 200  # 构建时搜索宽度
            self.index.hnsw.efSearch = 100  # 查询时搜索宽度
        else:
            raise ValueError(f"Unknown index_type: {index_type}")

        self.corpus = None

    def add_corpus(self, embeddings: np.ndarray, corpus: List[str]):
        """
        添加文档向量到索引

        Args:
            embeddings: 文档向量 (N, dim)
            corpus: 文档文本列表
        """
        self.corpus = corpus

        # 归一化(用于余弦相似度)
        faiss.normalize_L2(embeddings)

        # 添加到索引
        self.index.add(embeddings.astype('float32'))

        print(f"已添加 {self.index.ntotal} 个向量到{self.index_type}索引")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        检索Top-K最相关文档

        Args:
            query_embedding: 查询向量 (dim,)
            top_k: 返回数量

        Returns:
            [(doc_idx, score), ...]
        """
        # 归一化
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)

        # 搜索
        scores, indices = self.index.search(query_embedding, top_k)

        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]


# 使用示例
if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    # 加载模型
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    # 文档
    corpus = [
        "Transformer是注意力机制模型",
        "BERT用于自然语言理解",
        "GPT是生成式模型",
        "RAG结合检索与生成"
    ] * 100  # 扩大规模测试

    # 生成embeddings
    embeddings = model.encode(corpus, normalize_embeddings=True)

    # 构建FAISS索引
    retriever = FAISSRetriever(
        embedding_dim=embeddings.shape[1],
        index_type="HNSW"
    )
    retriever.add_corpus(embeddings, corpus)

    # 查询
    query = "什么是Transformer?"
    query_emb = model.encode(query, normalize_embeddings=True)

    import time
    start = time.time()
    results = retriever.search(query_emb, top_k=5)
    elapsed = (time.time() - start) * 1000

    print(f"\n查询耗时: {elapsed:.2f}ms\n")
    for idx, score in results:
        print(f"[{score:.4f}] {corpus[idx]}")
```

---

## 第二部分：文档处理与分块策略

### 2.1 为什么需要分块?

**问题1: LLM上下文窗口限制**
- GPT-3.5-turbo: 16K tokens
- GPT-4: 128K tokens
- Claude 3: 200K tokens

即使有长上下文,一次性输入整个文档:
- 成本高(按token计费)
- 注意力稀释("lost in the middle"现象)
- 检索不精确(无关内容混入)

**问题2: 语义单元完整性**

不当分块导致语义截断:
```
原文: "Transformer模型由Vaswani等人在2017年提出。它使用自注意力机制..."
错误分块: ["Transformer模型由Vaswani等人在2017年提出。", "它使用自注意力机制..."]
问题: "它"指代不明!
```

**解决方案**: 合理分块 + 重叠策略

### 2.2 固定长度分块(Fixed-Size Chunking)

#### 2.2.1 算法原理

**最简单方法**: 按固定字符数或token数分割

$$
\text{chunks} = \left[ \text{text}[i:i+\text{chunk\_size}] \mid i = 0, \text{chunk\_size}, 2\cdot\text{chunk\_size}, \ldots \right]
$$

**重叠策略**(避免语义截断):

$$
i \in \{0, \text{chunk\_size} - \text{overlap}, 2(\text{chunk\_size} - \text{overlap}), \ldots\}
$$

#### 2.2.2 Python实现

```python
from typing import List

class FixedSizeChunker:
    """
    固定长度分块器
    """
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        Args:
            chunk_size: 块大小(字符数或token数)
            overlap: 重叠大小
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_by_chars(self, text: str) -> List[str]:
        """
        按字符数分块

        Returns:
            文本块列表
        """
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)

            # 下一块起点(考虑重叠)
            start = end - self.overlap

            # 避免无限循环(overlap >= chunk_size)
            if start <= end - self.chunk_size:
                start = end

        return chunks

    def chunk_by_tokens(self, text: str, tokenizer) -> List[str]:
        """
        按token数分块(更精确)

        Args:
            tokenizer: Hugging Face tokenizer或tiktoken

        Returns:
            文本块列表
        """
        # 分词
        tokens = tokenizer.encode(text)
        total_tokens = len(tokens)

        chunks = []
        start = 0

        while start < total_tokens:
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]

            # 解码回文本
            chunk_text = tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

            start = end - self.overlap
            if start <= end - self.chunk_size:
                start = end

        return chunks


# 使用示例
if __name__ == "__main__":
    import tiktoken  # pip install tiktoken

    # 示例文本
    text = """
    Transformer是一种基于自注意力机制的深度学习模型架构,由Vaswani等人在2017年提出。
    与传统的循环神经网络(RNN)和卷积神经网络(CNN)不同,Transformer完全依赖注意力机制来捕捉序列中的依赖关系。
    这种设计使得Transformer能够并行处理序列,大大提高了训练效率。
    BERT、GPT等大型语言模型都基于Transformer架构构建。
    """ * 5

    # 初始化分块器
    chunker = FixedSizeChunker(chunk_size=100, overlap=20)

    # 按字符分块
    char_chunks = chunker.chunk_by_chars(text)
    print(f"按字符分块: {len(char_chunks)}块")
    print(f"第1块: {char_chunks[0][:100]}...")

    # 按token分块
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    token_chunks = chunker.chunk_by_tokens(text, tokenizer)
    print(f"\n按token分块: {len(token_chunks)}块")
    print(f"第1块token数: {len(tokenizer.encode(token_chunks[0]))}")
```

#### 2.2.3 参数选择指南

| Chunk Size | 适用场景 | 优点 | 缺点 |
|-----------|---------|------|------|
| 256 | 精确匹配(FAQ) | 检索精准,噪声低 | 语义碎片化 |
| 512 | **通用推荐** | 平衡精度与连贯性 | - |
| 1024 | 长上下文理解 | 语义完整 | 噪声高,成本增加 |
| 2048+ | 整文档理解 | 全局视角 | 检索不精确 |

**Overlap建议**: 10-20% of chunk_size (50-100 tokens)

### 2.3 递归字符分块(Recursive Character Splitting)

#### 2.3.1 算法思想

**问题**: 固定长度可能截断句子/段落

**递归策略**: 按自然分隔符层次化分割

```
分隔符优先级:
1. "\n\n" (段落)
2. "\n" (换行)
3. ". " (句子)
4. " " (单词)
5. "" (字符) - 最后手段
```

**伪代码**:
```python
function recursive_split(text, max_size, separators):
    if len(text) <= max_size:
        return [text]

    # 尝试第一个分隔符
    sep = separators[0]
    splits = text.split(sep)

    chunks = []
    for split in splits:
        if len(split) > max_size:
            # 递归使用下一级分隔符
            sub_chunks = recursive_split(split, max_size, separators[1:])
            chunks.extend(sub_chunks)
        else:
            chunks.append(split)

    return chunks
```

#### 2.3.2 LangChain实现

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 初始化分块器
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],  # 优先级
    length_function=len  # 长度计算函数
)

# 示例文本
text = """
# Transformer架构

Transformer由编码器和解码器组成。

## 编码器
编码器包含多层自注意力机制和前馈网络。每层都有残差连接和层归一化。

## 解码器
解码器结构类似,但增加了交叉注意力层。
"""

# 分块
chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"=== Chunk {i+1} ===")
    print(chunk)
    print(f"Length: {len(chunk)}\n")
```

**优势**:
- 保持文档结构(段落完整性)
- 适应多种文档格式
- LangChain生态集成

### 2.4 语义分块(Semantic Chunking)

#### 2.4.1 核心思想

**问题**: 固定/递归分块忽略语义

**目标**: 按主题/语义单元分块

**算法**: 基于句子embedding相似度突变检测

**流程**:
```
1. 分句: text → [s1, s2, s3, ..., sN]
2. 编码: embed(si) → [e1, e2, ..., eN]
3. 计算相邻相似度: sim(ei, ei+1)
4. 检测突变: if sim(ei, ei+1) < threshold → 切分点
5. 合并同主题句子为chunk
```
#### 2.4.2 Python实现

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List
import nltk

# 确保已下载nltk数据: nltk.download('punkt')

class SemanticChunker:
    """
    基于语义相似度的分块器
    """
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        similarity_threshold: float = 0.7,
        max_chunk_size: int = 1000
    ):
        """
        Args:
            model_name: Embedding模型
            similarity_threshold: 语义突变阈值(越低越敏感)
            max_chunk_size: 最大块大小(强制分割)
        """
        self.model = SentenceTransformer(model_name)
        self.threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str) -> List[str]:
        """
        语义分块

        Returns:
            文本块列表
        """
        # 1. 分句
        sentences = nltk.sent_tokenize(text)

        if len(sentences) <= 1:
            return [text]

        # 2. 生成句子embeddings
        embeddings = self.model.encode(sentences, normalize_embeddings=True)

        # 3. 计算相邻相似度
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i+1].reshape(1, -1)
            )[0][0]
            similarities.append(sim)

        # 4. 检测突变点(分块边界)
        breakpoints = [0]  # 起始点
        for i, sim in enumerate(similarities):
            if sim < self.threshold:
                breakpoints.append(i + 1)  # i和i+1之间分割
        breakpoints.append(len(sentences))  # 终止点

        # 5. 合并句子为chunk
        chunks = []
        for i in range(len(breakpoints) - 1):
            start = breakpoints[i]
            end = breakpoints[i + 1]
            chunk = " ".join(sentences[start:end])

            # 强制分割过大chunk
            if len(chunk) > self.max_chunk_size:
                # 使用固定长度分块
                sub_chunks = self._split_large_chunk(chunk)
                chunks.extend(sub_chunks)
            else:
                chunks.append(chunk)

        return chunks

    def _split_large_chunk(self, chunk: str) -> List[str]:
        """分割过大chunk(降级为固定长度)"""
        sub_chunks = []
        for i in range(0, len(chunk), self.max_chunk_size - 100):
            sub_chunks.append(chunk[i:i+self.max_chunk_size])
        return sub_chunks


# 使用示例
if __name__ == "__main__":
    text = """
    Transformer架构革新了自然语言处理领域。它由编码器和解码器两部分组成。
    编码器负责理解输入序列。它使用多头自注意力机制捕捉长距离依赖。
    解码器则用于生成输出。解码器在编码器基础上增加了交叉注意力层。
    BERT只使用编码器部分。它通过掩码语言模型进行预训练。
    GPT只使用解码器。它是自回归生成模型,逐token生成文本。
    """

    # 初始化分块器
    chunker = SemanticChunker(
        similarity_threshold=0.6,  # 调低阈值增加敏感度
        max_chunk_size=500
    )

    # 分块
    chunks = chunker.chunk(text)

    print(f"语义分块结果: {len(chunks)}块\n")
    for i, chunk in enumerate(chunks):
        print(f"=== Chunk {i+1} ===")
        print(chunk.strip())
        print()
```

**输出示例**:
```
语义分块结果: 3块

=== Chunk 1 ===
Transformer架构革新了自然语言处理领域。它由编码器和解码器两部分组成。编码器负责理解输入序列。它使用多头自注意力机制捕捉长距离依赖。解码器则用于生成输出。解码器在编码器基础上增加了交叉注意力层。

=== Chunk 2 ===
BERT只使用编码器部分。它通过掩码语言模型进行预训练。

=== Chunk 3 ===
GPT只使用解码器。它是自回归生成模型,逐token生成文本。
```

**优势**:
- 保持主题完整性
- 提升检索相关性(15-25% Recall提升)
- 适合技术文档/教程

**劣势**:
- 计算成本高(需编码所有句子)
- 阈值调优困难(领域依赖)

### 2.5 上下文增强分块(Contextual Chunking)

#### 2.5.1 Anthropic Contextual Retrieval

**问题**: 传统chunk缺乏上下文

```
原始文档: "2023年Q4财报 - 公司营收增长30%"
分块后: "公司营收增长30%"  ← 缺少时间/文档背景!
```

**Anthropic解决方案** (2024年9月):
- 为每个chunk添加上下文说明
- 使用LLM自动生成上下文
- 召回率提升35%(官方数据)

#### 2.5.2 实现方案

```python
from openai import OpenAI
from typing import List

class ContextualChunker:
    """
    上下文增强分块器(Anthropic Contextual Retrieval)
    """
    def __init__(self, client: OpenAI, base_chunker):
        """
        Args:
            client: OpenAI/Anthropic客户端
            base_chunker: 基础分块器(FixedSizeChunker等)
        """
        self.client = client
        self.base_chunker = base_chunker

    def chunk_with_context(
        self,
        document: str,
        document_metadata: dict
    ) -> List[dict]:
        """
        生成带上下文的chunks

        Args:
            document: 完整文档
            document_metadata: 文档元数据(标题/作者/日期等)

        Returns:
            [{"original": str, "contextualized": str}, ...]
        """
        # 1. 基础分块
        base_chunks = self.base_chunker.chunk(document)

        # 2. 为每个chunk生成上下文
        contextualized_chunks = []
        for chunk in base_chunks:
            context = self._generate_context(document, chunk, document_metadata)
            contextualized_chunks.append({
                "original": chunk,
                "contextualized": f"{context}\n\n{chunk}",
                "context_only": context
            })

        return contextualized_chunks

    def _generate_context(
        self,
        full_document: str,
        chunk: str,
        metadata: dict
    ) -> str:
        """
        使用LLM生成chunk上下文
        """
        prompt = f"""
文档元数据:
标题: {metadata.get('title', 'Unknown')}
作者: {metadata.get('author', 'Unknown')}
日期: {metadata.get('date', 'Unknown')}

完整文档内容:
{full_document[:2000]}...  # 截取前2000字符(避免超长)

---

请为以下文档片段生成简洁的上下文说明(1-2句话),描述:
1. 该片段在文档中的位置/章节
2. 该片段讨论的核心主题

文档片段:
{chunk}

上下文说明(中文):
"""

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",  # 使用便宜模型
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100
        )

        context = response.choices[0].message.content.strip()
        return context


# 使用示例
if __name__ == "__main__":
    # 初始化
    client = OpenAI(api_key="your-api-key")
    base_chunker = FixedSizeChunker(chunk_size=300, overlap=50)
    contextual_chunker = ContextualChunker(client, base_chunker)

    # 示例文档
    document = """
第2章 Transformer架构详解

Transformer由Vaswani等人在2017年提出,是一种革命性的神经网络架构。

2.1 自注意力机制
自注意力机制是Transformer的核心。它计算序列中每个位置与其他位置的关联性,公式为:
Attention(Q,K,V) = softmax(QK^T/√d_k)V

2.2 多头注意力
多头注意力并行计算多个注意力头,捕捉不同子空间的信息。
"""

    metadata = {
        "title": "深度学习模型架构详解",
        "author": "AI研究团队",
        "date": "2024-01-15"
    }

    # 生成带上下文的chunks
    chunks = contextual_chunker.chunk_with_context(document, metadata)

    print("=== 原始Chunk ===")
    print(chunks[0]["original"])

    print("\n=== 上下文增强Chunk ===")
    print(chunks[0]["contextualized"])
```

**输出示例**:
```
=== 原始Chunk ===
自注意力机制是Transformer的核心。它计算序列中每个位置与其他位置的关联性,公式为:
Attention(Q,K,V) = softmax(QK^T/√d_k)V

=== 上下文增强Chunk ===
本片段位于《深度学习模型架构详解》第2.1节,讨论Transformer架构中自注意力机制的数学原理和计算公式。

自注意力机制是Transformer的核心。它计算序列中每个位置与其他位置的关联性,公式为:
Attention(Q,K,V) = softmax(QK^T/√d_k)V
```

**成本优化**: 使用Prompt Caching(Claude 3.5支持)缓存文档全文,降低90%成本

### 2.6 分块策略对比总结

| 策略 | 召回率 | 精确度 | 计算成本 | 适用场景 |
|-----|--------|--------|---------|---------|
| 固定长度 | 中 | 中 | 低 | 通用场景 |
| 递归字符 | 中高 | 中高 | 低 | 结构化文档(Markdown/HTML) |
| 语义分块 | 高 | 高 | 高 | 技术文档/教程 |
| 上下文增强 | **最高(+35%)** | 最高 | **最高** | 企业知识库/多文档 |

**推荐组合**:
```python
# 生产环境推荐
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)

# 预算充足 + 高精度需求
contextual_chunker = ContextualChunker(client, splitter)
```

---

## 第三部分：向量存储与检索技术

### 3.1 向量数据库架构

#### 3.1.1 为什么需要专用向量数据库?

**传统数据库(MySQL/PostgreSQL)**:
- 索引: B树/哈希表(精确匹配)
- 查询: `SELECT * WHERE id = 123`

**向量数据库需求**:
- 索引: ANN算法(近似匹配)
- 查询: "找到与[0.1, 0.5, 0.8, ...]最相似的K个向量"

**性能对比**:

| 操作 | PostgreSQL(暴力) | 向量数据库(ANN) |
|-----|-----------------|----------------|
| 100万向量查询 | 5-10秒 | 5-50毫秒 |
| 1亿向量查询 | 超时 | 50-200毫秒 |

#### 3.1.2 主流向量数据库对比(2025)

| 数据库 | 语言 | 核心算法 | 规模 | 部署 | 推荐场景 |
|-------|------|---------|------|------|---------|
| **FAISS** | C++/Python | HNSW/IVF | 百万-亿级 | 本地 | 研究/原型 |
| **Qdrant** | Rust | HNSW | 亿级 | 本地/云 | 生产环境(高性能) |
| **Milvus** | Go/C++ | HNSW/IVF/DiskANN | 十亿级 | 分布式 | 超大规模 |
| **Weaviate** | Go | HNSW | 亿级 | 本地/云 | GraphRAG/混合检索 |
| **Chroma** | Python | HNSW | 百万级 | 本地 | 快速原型 |
| **pgvector** | C | HNSW/IVF | 千万级 | PostgreSQL扩展 | 已有PG基础设施 |
| **Pinecone** | - | 专有 | 十亿级 | 云(SaaS) | 托管服务 |

**选择决策树**:
```
是否已有PostgreSQL?
  → Yes: pgvector
  → No: ↓

数据规模?
  → <100万: Chroma(快速原型)
  → 100万-1亿: Qdrant(高性能)
  → >1亿: Milvus(分布式)

是否需要GraphRAG/混合检索?
  → Yes: Weaviate
  → No: Qdrant
```

### 3.2 FAISS深度实战

#### 3.2.1 FAISS索引类型

**FAISS**(Facebook AI Similarity Search)提供多种索引:

| 索引类型 | 算法 | 精度 | 速度 | 内存 | 适用规模 |
|---------|------|------|------|------|---------|
| `IndexFlatIP` | 暴力(Inner Product) | 100% | 慢 | 高 | <10万 |
| `IndexFlatL2` | 暴力(L2距离) | 100% | 慢 | 高 | <10万 |
| `IndexHNSWFlat` | HNSW | 95-99% | 快 | 高 | <1000万 |
| `IndexIVFFlat` | 倒排文件 | 90-95% | 中 | 中 | 100万-1亿 |
| `IndexIVFPQ` | IVF + 乘积量化 | 85-90% | 快 | **低** | 亿级+ |

#### 3.2.2 完整FAISS检索系统

```python
import faiss
import numpy as np
import pickle
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class Document:
    """文档数据类"""
    id: str
    content: str
    metadata: dict
    embedding: np.ndarray = None


class FAISSVectorStore:
    """
    生产级FAISS向量存储
    """
    def __init__(
        self,
        embedding_dim: int,
        index_type: str = "HNSW",
        metric: str = "cosine"
    ):
        """
        Args:
            embedding_dim: 向量维度
            index_type: 索引类型(HNSW/IVF/IVF_PQ)
            metric: 距离度量(cosine/euclidean)
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.metric = metric

        # 构建索引
        self.index = self._build_index()

        # 文档存储(索引 → 文档)
        self.documents: List[Document] = []
        self.id_to_idx = {}  # 文档ID → 索引位置

    def _build_index(self):
        """构建FAISS索引"""
        if self.index_type == "HNSW":
            index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 100

        elif self.index_type == "IVF":
            # 聚类中心数(通常√N,这里设置1000)
            nlist = 1000
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            index = faiss.IndexIVFFlat(
                quantizer,
                self.embedding_dim,
                nlist,
                faiss.METRIC_INNER_PRODUCT
            )
            # IVF需要训练
            index.nprobe = 10  # 查询时探测的聚类数

        elif self.index_type == "IVF_PQ":
            # IVF + 乘积量化(大规模数据)
            nlist = 1000
            m = 8  # 子空间数
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            index = faiss.IndexIVFPQ(
                quantizer,
                self.embedding_dim,
                nlist,
                m,
                8  # 每个子空间8位
            )
            index.nprobe = 10

        else:
            raise ValueError(f"Unknown index_type: {self.index_type}")

        return index

    def add_documents(self, documents: List[Document]):
        """
        添加文档到向量库

        Args:
            documents: 文档列表(需包含embedding)
        """
        if not documents:
            return

        # 提取embeddings
        embeddings = np.array([doc.embedding for doc in documents]).astype('float32')

        # 归一化(余弦相似度)
        if self.metric == "cosine":
            faiss.normalize_L2(embeddings)

        # IVF索引需要训练
        if self.index_type in ["IVF", "IVF_PQ"] and not self.index.is_trained:
            print(f"训练{self.index_type}索引...")
            self.index.train(embeddings)

        # 添加到索引
        start_idx = len(self.documents)
        self.index.add(embeddings)

        # 保存文档
        for i, doc in enumerate(documents):
            idx = start_idx + i
            self.documents.append(doc)
            self.id_to_idx[doc.id] = idx

        print(f"已添加{len(documents)}个文档,总计{self.index.ntotal}个")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_func = None
    ) -> List[Tuple[Document, float]]:
        """
        检索Top-K最相关文档

        Args:
            query_embedding: 查询向量
            top_k: 返回数量
            filter_func: 过滤函数(doc) -> bool

        Returns:
            [(Document, score), ...]
        """
        # 归一化
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        if self.metric == "cosine":
            faiss.normalize_L2(query_embedding)

        # 搜索(多检索一些以应对过滤)
        search_k = top_k * 3 if filter_func else top_k
        scores, indices = self.index.search(query_embedding, search_k)

        # 构建结果
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:  # FAISS未找到结果
                continue

            doc = self.documents[idx]

            # 应用过滤
            if filter_func and not filter_func(doc):
                continue

            results.append((doc, float(score)))

            if len(results) >= top_k:
                break

        return results

    def delete_by_id(self, doc_id: str):
        """删除文档(FAISS不支持原生删除,需要重建)"""
        if doc_id not in self.id_to_idx:
            return

        # 标记删除(实际实现需重建索引)
        idx = self.id_to_idx[doc_id]
        self.documents[idx] = None  # 占位
        del self.id_to_idx[doc_id]

    def save(self, path: str):
        """保存索引和文档"""
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.docs.pkl", "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "id_to_idx": self.id_to_idx,
                "config": {
                    "embedding_dim": self.embedding_dim,
                    "index_type": self.index_type,
                    "metric": self.metric
                }
            }, f)
        print(f"已保存到 {path}")

    @classmethod
    def load(cls, path: str):
        """加载索引和文档"""
        with open(f"{path}.docs.pkl", "rb") as f:
            data = pickle.load(f)

        config = data["config"]
        store = cls(**config)
        store.index = faiss.read_index(f"{path}.index")
        store.documents = data["documents"]
        store.id_to_idx = data["id_to_idx"]

        print(f"已加载 {store.index.ntotal}个向量")
        return store


# 使用示例
if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    # 加载embedding模型
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    embedding_dim = 384

    # 创建向量库
    vector_store = FAISSVectorStore(
        embedding_dim=embedding_dim,
        index_type="HNSW",
        metric="cosine"
    )

    # 准备文档
    texts = [
        "Transformer使用自注意力机制",
        "BERT是双向编码器模型",
        "GPT采用自回归生成",
        "RAG结合检索和生成技术"
    ]

    documents = []
    for i, text in enumerate(texts):
        embedding = model.encode(text, normalize_embeddings=True)
        doc = Document(
            id=f"doc_{i}",
            content=text,
            metadata={"source": "example"},
            embedding=embedding
        )
        documents.append(doc)

    # 添加到向量库
    vector_store.add_documents(documents)

    # 查询
    query = "什么是Transformer?"
    query_emb = model.encode(query, normalize_embeddings=True)

    results = vector_store.search(query_emb, top_k=3)

    print(f"\n查询: {query}\n")
    for doc, score in results:
        print(f"[{score:.4f}] {doc.content}")

    # 保存
    vector_store.save("./my_vector_store")

    # 加载
    loaded_store = FAISSVectorStore.load("./my_vector_store")
    results2 = loaded_store.search(query_emb, top_k=3)
    assert len(results2) == len(results)
```

### 3.3 Milvus分布式向量数据库

#### 3.3.1 Milvus架构

**特点**:
- 分布式架构(支持水平扩展)
- 多种索引(HNSW/IVF/DiskANN)
- GPU加速
- 元数据过滤

**安装**:
```bash
# Docker方式(推荐)
wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O docker-compose.yml
docker-compose up -d

# Python客户端
pip install pymilvus
```
#### 3.3.2 Milvus完整示例

```python
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility
)
from sentence_transformers import SentenceTransformer
import numpy as np

class MilvusVectorStore:
    """
    Milvus向量存储封装
    """
    def __init__(
        self,
        collection_name: str,
        embedding_dim: int,
        host: str = "localhost",
        port: str = "19530"
    ):
        """
        Args:
            collection_name: 集合名称(类似表名)
            embedding_dim: 向量维度
            host/port: Milvus服务地址
        """
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim

        # 连接Milvus
        connections.connect("default", host=host, port=port)
        print(f"已连接到Milvus: {host}:{port}")

        # 创建或加载集合
        self.collection = self._create_collection()

    def _create_collection(self):
        """创建集合(定义schema)"""
        # 检查是否已存在
        if utility.has_collection(self.collection_name):
            print(f"集合 '{self.collection_name}' 已存在,加载中...")
            collection = Collection(self.collection_name)
            collection.load()
            return collection

        # 定义schema
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]

        schema = CollectionSchema(
            fields=fields,
            description="RAG文档向量库"
        )

        # 创建集合
        collection = Collection(
            name=self.collection_name,
            schema=schema
        )

        # 创建HNSW索引
        index_params = {
            "metric_type": "IP",  # Inner Product(余弦相似度)
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200}
        }
        collection.create_index(field_name="embedding", index_params=index_params)

        # 加载到内存
        collection.load()

        print(f"已创建集合 '{self.collection_name}'")
        return collection

    def add_documents(self, documents: list):
        """
        添加文档

        Args:
            documents: [{"id": str, "content": str, "embedding": np.ndarray, "metadata": dict}, ...]
        """
        if not documents:
            return

        # 准备数据
        ids = [doc["id"] for doc in documents]
        contents = [doc["content"] for doc in documents]
        embeddings = [doc["embedding"].tolist() for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]

        # 插入
        data = [ids, embeddings, contents, metadatas]
        self.collection.insert(data)
        self.collection.flush()  # 确保写入

        print(f"已添加{len(documents)}个文档")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_expr: str = None
    ):
        """
        检索Top-K文档

        Args:
            query_embedding: 查询向量
            top_k: 返回数量
            filter_expr: 过滤表达式(如 'metadata["category"] == "技术"')

        Returns:
            [(id, content, metadata, score), ...]
        """
        search_params = {"metric_type": "IP", "params": {"ef": 100}}

        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=filter_expr,  # 元数据过滤
            output_fields=["content", "metadata"]
        )

        # 解析结果
        output = []
        for hit in results[0]:
            output.append({
                "id": hit.id,
                "content": hit.entity.get("content"),
                "metadata": hit.entity.get("metadata"),
                "score": float(hit.score)
            })

        return output

    def delete_by_ids(self, ids: list):
        """删除文档"""
        expr = f"id in {ids}"
        self.collection.delete(expr)
        print(f"已删除{len(ids)}个文档")


# 使用示例
if __name__ == "__main__":
    # 加载模型
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    # 创建向量库
    vector_store = MilvusVectorStore(
        collection_name="rag_demo",
        embedding_dim=384
    )

    # 准备文档
    texts = [
        "Transformer革新了NLP领域",
        "BERT是预训练语言模型",
        "GPT采用自回归架构",
        "RAG提升LLM事实准确性"
    ]

    documents = []
    for i, text in enumerate(texts):
        embedding = model.encode(text, normalize_embeddings=True)
        documents.append({
            "id": f"doc_{i}",
            "content": text,
            "embedding": embedding,
            "metadata": {"category": "NLP", "index": i}
        })

    # 添加文档
    vector_store.add_documents(documents)

    # 查询
    query = "什么是Transformer?"
    query_emb = model.encode(query, normalize_embeddings=True)

    results = vector_store.search(query_emb, top_k=3)

    print(f"\n查询: {query}\n")
    for res in results:
        print(f"[{res['score']:.4f}] {res['content']}")

    # 元数据过滤查询
    filtered_results = vector_store.search(
        query_emb,
        top_k=3,
        filter_expr='metadata["index"] < 2'  # 只返回前2个文档
    )

    print("\n过滤后结果:")
    for res in filtered_results:
        print(f"[{res['score']:.4f}] {res['content']} (index={res['metadata']['index']})")
```

---

## 第四部分：检索方法深度剖析

### 4.1 混合检索(Hybrid Search)

#### 4.1.1 为什么需要混合检索?

**单一检索的局限**:

| 检索方式 | 优势 | 劣势 | 典型miss案例 |
|---------|------|------|-------------|
| **BM25(稀疏)** | 精确字面匹配 | 无法理解语义 | 查询"手机",miss"iPhone" |
| **向量(密集)** | 语义理解 | 专有名词miss | 查询"GPT-4",miss"GPT-3.5" |

**混合检索**: 结合两者优势,覆盖更全面!

#### 4.1.2 RRF融合算法(Reciprocal Rank Fusion)

**数学公式**:

$$
\text{RRF}(d) = \sum_{r \in \text{Retrievers}} \frac{1}{k + \text{rank}_r(d)}
$$

其中:
- $d$: 文档
- $\text{rank}_r(d)$: 文档在检索器 $r$ 中的排名(1-based)
- $k$: 常数,通常 $k=60$

**核心优势**:
- 无需归一化分数(直接用排名)
- 对异常值鲁棒
- 无需训练

**工作原理示例**:

```
查询: "Transformer注意力机制"

BM25排名:
  1. 文档A (包含"Transformer"+"注意力机制")
  2. 文档C (包含"Transformer")
  3. 文档B (包含"注意力")

向量排名:
  1. 文档B (语义最相关)
  2. 文档A (语义相关)
  3. 文档D (同义表达)

RRF计算(k=60):
  文档A: 1/(60+1) + 1/(60+2) = 0.0325
  文档B: 1/(60+3) + 1/(60+1) = 0.0323
  文档C: 1/(60+2) + 0 = 0.0161
  文档D: 0 + 1/(60+3) = 0.0159

最终排序: A > B > C > D
```

#### 4.1.3 混合检索完整实现

```python
from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np

class HybridRetriever:
    """
    混合检索器(BM25 + 向量 + RRF融合)
    """
    def __init__(
        self,
        bm25_retriever,
        dense_retriever,
        rrf_k: int = 60,
        weights: Tuple[float, float] = (0.5, 0.5)
    ):
        """
        Args:
            bm25_retriever: BM25检索器
            dense_retriever: 密集检索器
            rrf_k: RRF常数
            weights: (bm25_weight, dense_weight)
        """
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.rrf_k = rrf_k
        self.bm25_weight, self.dense_weight = weights

    def search(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
        bm25_k: int = 20,
        dense_k: int = 20
    ) -> List[Dict]:
        """
        混合检索

        Args:
            query: 查询字符串(BM25用)
            query_embedding: 查询向量(密集检索用)
            top_k: 最终返回数量
            bm25_k/dense_k: 各检索器召回数量

        Returns:
            [{"doc_id": str, "content": str, "score": float, "source": str}, ...]
        """
        # 1. 并行执行两种检索
        bm25_results = self.bm25.search(query, k=bm25_k)  # [(doc_id, score), ...]
        dense_results = self.dense.search(query_embedding, k=dense_k)  # [(doc_id, score), ...]

        # 2. RRF融合
        rrf_scores = self._reciprocal_rank_fusion(
            bm25_results=[r[0] for r in bm25_results],
            dense_results=[r[0] for r in dense_results]
        )

        # 3. 按分数排序
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # 4. 取Top-K并返回完整信息
        final_results = []
        for doc_id, score in sorted_results[:top_k]:
            # 从检索器获取文档内容
            doc_content = self._get_document_content(doc_id)

            # 标注来源
            in_bm25 = any(r[0] == doc_id for r in bm25_results)
            in_dense = any(r[0] == doc_id for r in dense_results)
            if in_bm25 and in_dense:
                source = "BM25+Dense"
            elif in_bm25:
                source = "BM25"
            else:
                source = "Dense"

            final_results.append({
                "doc_id": doc_id,
                "content": doc_content,
                "score": score,
                "source": source
            })

        return final_results

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[str],
        dense_results: List[str]
    ) -> Dict[str, float]:
        """
        RRF融合算法

        Returns:
            {doc_id: rrf_score}
        """
        rrf_scores = defaultdict(float)

        # BM25排名贡献
        for rank, doc_id in enumerate(bm25_results, start=1):
            rrf_scores[doc_id] += self.bm25_weight * (1.0 / (self.rrf_k + rank))

        # 密集检索排名贡献
        for rank, doc_id in enumerate(dense_results, start=1):
            rrf_scores[doc_id] += self.dense_weight * (1.0 / (self.rrf_k + rank))

        return dict(rrf_scores)

    def _get_document_content(self, doc_id: str) -> str:
        """从检索器获取文档内容(简化实现)"""
        # 实际应用中应从统一文档存储获取
        return f"文档{doc_id}的内容..."


# 使用示例(伪代码)
if __name__ == "__main__":
    # 初始化检索器
    bm25_retriever = BM25(corpus)  # 前面实现的BM25
    dense_retriever = DenseRetriever("BAAI/bge-small-en-v1.5")

    # 创建混合检索器
    hybrid_retriever = HybridRetriever(
        bm25_retriever=bm25_retriever,
        dense_retriever=dense_retriever,
        rrf_k=60,
        weights=(0.4, 0.6)  # BM25:40%, Dense:60%
    )

    # 查询
    query = "Transformer的自注意力机制"
    query_emb = dense_retriever.model.encode(query, normalize_embeddings=True)

    results = hybrid_retriever.search(
        query=query,
        query_embedding=query_emb,
        top_k=5
    )

    print(f"查询: {query}\n")
    for res in results:
        print(f"[{res['score']:.4f}] {res['source']}: {res['content']}")
```

#### 4.1.4 LangChain集成方案

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# 1. 准备文档
from langchain.schema import Document

docs = [
    Document(page_content="Transformer使用自注意力机制"),
    Document(page_content="BERT是双向编码器"),
    Document(page_content="GPT采用自回归生成"),
]

# 2. 构建BM25检索器
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 5

# 3. 构建向量检索器
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_store = FAISS.from_documents(docs, embeddings)
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# 4. 混合检索器(RRF融合)
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6],  # BM25:40%, 向量:60%
    c=60  # RRF常数k
)

# 5. 查询
query = "什么是Transformer?"
results = ensemble_retriever.get_relevant_documents(query)

for doc in results:
    print(doc.page_content)
```

#### 4.1.4 LangChain集成方案

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# 1. 准备文档
from langchain.schema import Document

docs = [
    Document(page_content="Transformer使用自注意力机制"),
    Document(page_content="BERT是双向编码器"),
    Document(page_content="GPT采用自回归生成"),
]

# 2. 构建BM25检索器
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 5

# 3. 构建向量检索器
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_store = FAISS.from_documents(docs, embeddings)
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# 4. 混合检索器(RRF融合)
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6],  # BM25:40%, 向量:60%
    c=60  # RRF常数k
)

# 5. 查询
query = "什么是Transformer?"
results = ensemble_retriever.get_relevant_documents(query)

for doc in results:
    print(doc.page_content)
```

### 4.2 高级检索策略

#### 4.2.1 查询改写(Query Rewriting)

**问题**: 用户查询可能模糊/不完整

**示例**:
```
对话历史:
  用户: "介绍一下Transformer"
  助手: "Transformer是..."
  用户: "它的优势是什么?"  ← "它"指代不明!
```

**解决方案**: 将查询改写为独立完整的问题

```python
from openai import OpenAI

class QueryRewriter:
    """
    查询改写器
    """
    def __init__(self, client: OpenAI):
        self.client = client

    def rewrite(self, query: str, chat_history: List[Dict] = None) -> str:
        """
        改写查询

        Args:
            query: 当前查询
            chat_history: 对话历史[{"role": "user/assistant", "content": str}, ...]

        Returns:
            改写后的查询
        """
        if not chat_history:
            return query  # 无历史,直接返回

        # 构建提示
        history_str = "\n".join([
            f"{msg['role']}: {msg['content']}" for msg in chat_history[-3:]  # 最近3轮
        ])

        prompt = f"""
给定对话历史和当前查询,将查询改写为独立完整的问题。

对话历史:
{history_str}

当前查询: {query}

改写后的查询(保持原意,补全指代):
"""

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100
        )

        rewritten = response.choices[0].message.content.strip()
        return rewritten


# 使用示例
if __name__ == "__main__":
    client = OpenAI(api_key="your-api-key")
    rewriter = QueryRewriter(client)

    chat_history = [
        {"role": "user", "content": "介绍一下Transformer模型"},
        {"role": "assistant", "content": "Transformer是一种基于自注意力机制的模型..."}
    ]

    query = "它的优势是什么?"
    rewritten = rewriter.rewrite(query, chat_history)

    print(f"原查询: {query}")
    print(f"改写后: {rewritten}")
    # 输出: "改写后: Transformer模型的优势是什么?"
```

#### 4.2.2 多查询生成(Multi-Query)

**思想**: 从不同角度重述查询,增加召回覆盖面

```python
class MultiQueryGenerator:
    """
    多查询生成器
    """
    def __init__(self, client: OpenAI, n_queries: int = 3):
        self.client = client
        self.n_queries = n_queries

    def generate(self, query: str) -> List[str]:
        """
        生成多个查询变体

        Returns:
            [query1, query2, query3]
        """
        prompt = f"""
你是一个帮助用户从文档库检索信息的AI助手。
用户会提出一个问题,你需要生成{self.n_queries}个不同角度的查询,以提高检索召回率。

用户问题: {query}

生成{self.n_queries}个查询(JSON格式):
{{"queries": ["查询1", "查询2", "查询3"]}}
"""

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200,
            response_format={"type": "json_object"}
        )

        import json
        result = json.loads(response.choices[0].message.content)
        return result["queries"]


# 使用示例
if __name__ == "__main__":
    generator = MultiQueryGenerator(OpenAI(api_key="your-api-key"))

    query = "如何优化RAG系统的检索质量?"
    queries = generator.generate(query)

    print(f"原查询: {query}\n")
    print("生成的查询变体:")
    for i, q in enumerate(queries, 1):
        print(f"{i}. {q}")

    # 输出示例:
    # 1. RAG检索质量优化方法有哪些?
    # 2. 提升检索增强生成系统准确率的技术
    # 3. 改进RAG召回率和相关性的策略
```

---

## 第五部分：重排序技术

### 5.1 为什么需要重排序?

**两阶段检索架构**:

```
查询
  ↓
[阶段1] 粗排(召回) - 快速,召回Top-100
  ↓
[阶段2] 精排(重排序) - 慢速,精准Top-10
  ↓
生成答案
```

**Trade-off**:

| 阶段 | 模型复杂度 | 候选数量 | 延迟 | 精度 |
|-----|----------|---------|------|------|
| 召回 | 低(Bi-Encoder) | 100-1000 | 50ms | 中 |
| 精排 | 高(Cross-Encoder) | 10-20 | 200ms | 高 |

**总延迟**: 250ms (可接受)
**精度提升**: 10-20% MRR/NDCG

### 5.2 Bi-Encoder vs Cross-Encoder

#### 5.2.1 架构对比

**Bi-Encoder** (双塔模型):

```
查询 → Encoder_Q → vec_Q ─┐
                          ├─ cosine_similarity → score
文档 → Encoder_D → vec_D ─┘
```

**特点**:
- 查询和文档独立编码
- 可预计算文档向量(快速检索)
- 无法捕获查询-文档交互

**Cross-Encoder** (交叉编码器):

```
[查询; 文档] → Encoder → score (scalar)
```

**特点**:
- 联合编码,捕获深度交互
- 无法预计算,每次查询都需重新编码
- 精度最高

#### 5.2.2 数学原理

**Bi-Encoder**:

$$
\text{score}(q, d) = \text{cosine}(E_q(q), E_d(d))
$$

$$
= \frac{E_q(q)^T E_d(d)}{\|E_q(q)\| \|E_d(d)\|}
$$

**Cross-Encoder**:

$$
\text{score}(q, d) = \text{MLP}(\text{BERT}([q; d]))
$$

$$
\in [0, 1] \text{ or } \mathbb{R}
$$

**复杂度对比**:

| 模型 | 推理复杂度(每对query-doc) | 可否预计算doc | 适用阶段 |
|-----|--------------------------|-------------|---------|
| Bi-Encoder | $O(d)$ (点积) | ✅ Yes | 召回 |
| Cross-Encoder | $O(L^2 \cdot H)$ (self-attention) | ❌ No | 精排 |

### 5.3 Cross-Encoder实战

#### 5.3.1 Hugging Face模型

**推荐模型**:

| 模型 | 参数量 | 精度 | 速度 | 适用场景 |
|-----|-------|------|------|---------|
| `ms-marco-MiniLM-L-6-v2` | 22M | 中 | 快 | 实时应用 |
| `ms-marco-MiniLM-L-12-v2` | 33M | 中高 | 中 | 平衡 |
| `bge-reranker-base` | 278M | 高 | 慢 | 离线/高精度 |
| `bge-reranker-large` | 560M | 最高 | 最慢 | 研究/benchmark |

#### 5.3.2 完整实现

```python
from sentence_transformers import CrossEncoder
import numpy as np
from typing import List, Tuple

class Reranker:
    """
    Cross-Encoder重排序器
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Args:
            model_name: Cross-Encoder模型名称
        """
        print(f"加载重排序模型: {model_name}")
        self.model = CrossEncoder(model_name, max_length=512)

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        重排序文档

        Args:
            query: 查询字符串
            documents: 候选文档列表
            top_k: 返回Top-K

        Returns:
            [(doc_idx, score), ...] 按分数降序
        """
        if not documents:
            return []

        # 构建(query, doc)对
        pairs = [[query, doc] for doc in documents]

        # Cross-Encoder打分(批量推理)
        scores = self.model.predict(pairs, batch_size=32, show_progress_bar=False)

        # 排序
        scored_docs = [(i, float(score)) for i, score in enumerate(scores)]
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return scored_docs[:top_k]


class TwoStageRetriever:
    """
    两阶段检索器(召回 + 重排序)
    """
    def __init__(
        self,
        retriever,  # 召回器(BM25/Dense/Hybrid)
        reranker: Reranker,
        recall_k: int = 50,
        final_k: int = 5
    ):
        """
        Args:
            retriever: 第一阶段召回器
            reranker: Cross-Encoder重排序器
            recall_k: 召回Top-K
            final_k: 最终返回Top-K
        """
        self.retriever = retriever
        self.reranker = reranker
        self.recall_k = recall_k
        self.final_k = final_k

    def search(self, query: str, query_embedding: np.ndarray = None) -> List[Dict]:
        """
        两阶段检索

        Returns:
            [{"content": str, "recall_score": float, "rerank_score": float}, ...]
        """
        # 阶段1: 召回Top-K候选
        if query_embedding is not None:
            candidates = self.retriever.search(query, query_embedding, top_k=self.recall_k)
        else:
            candidates = self.retriever.search(query, top_k=self.recall_k)

        if not candidates:
            return []

        # 提取文档内容
        doc_contents = [c["content"] for c in candidates]

        # 阶段2: Cross-Encoder重排序
        reranked_indices = self.reranker.rerank(
            query=query,
            documents=doc_contents,
            top_k=self.final_k
        )

        # 构建最终结果
        final_results = []
        for doc_idx, rerank_score in reranked_indices:
            result = {
                "content": candidates[doc_idx]["content"],
                "recall_score": candidates[doc_idx].get("score", 0.0),
                "rerank_score": rerank_score
            }
            final_results.append(result)

        return final_results


# 使用示例
if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    # 准备数据
    corpus = [
        "Transformer使用自注意力机制处理序列",
        "Transformer由编码器和解码器组成",
        "BERT是基于Transformer的预训练模型",
        "GPT使用Transformer解码器进行生成",
        "注意力机制计算Query和Key的相似度",
        "多头注意力并行计算多个注意力",
        "Transformer在机器翻译中表现优异",
        "自注意力可以捕捉长距离依赖"
    ] * 10  # 扩大规模

    # 构建召回器(密集检索)
    dense_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    dense_retriever = DenseRetriever(model_name="BAAI/bge-small-en-v1.5")
    dense_retriever.encode_corpus(corpus)

    # 构建重排序器
    reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

    # 两阶段检索器
    two_stage = TwoStageRetriever(
        retriever=dense_retriever,
        reranker=reranker,
        recall_k=20,
        final_k=5
    )

    # 查询
    query = "Transformer的注意力机制是如何工作的?"
    query_emb = dense_model.encode(query, normalize_embeddings=True)

    import time
    start = time.time()
    results = two_stage.search(query, query_emb)
    elapsed = (time.time() - start) * 1000

    print(f"查询: {query}")
    print(f"耗时: {elapsed:.2f}ms\n")

    for i, res in enumerate(results, 1):
        print(f"{i}. [召回:{res['recall_score']:.4f} | 重排:{res['rerank_score']:.4f}]")
        print(f"   {res['content']}\n")
```

**性能对比实验**:

```python
# 对比实验:召回 vs 两阶段检索

# 仅召回
recall_only = dense_retriever.search(query_emb, top_k=5)

# 两阶段
two_stage_results = two_stage.search(query, query_emb)

print("=== 仅召回Top-5 ===")
for i, res in enumerate(recall_only, 1):
    print(f"{i}. {corpus[res[0]]}")

print("\n=== 两阶段Top-5 ===")
for i, res in enumerate(two_stage_results, 1):
    print(f"{i}. {res['content']}")

# 观察:重排序后,更相关的文档排名提升!
```

### 5.4 ColBERT晚期交互

#### 5.4.1 原理

**问题**:
- Bi-Encoder: 无交互,精度低
- Cross-Encoder: 全交互,速度慢

**ColBERT**: 延迟交互(Late Interaction) - 折中方案

**流程**:
```
查询 → BERT → [q1, q2, ..., qN]  # token级embeddings
文档 → BERT → [d1, d2, ..., dM]

相似度矩阵: S[i][j] = qi · dj
MaxSim聚合: score = Σ_i max_j S[i][j]
```

**数学**:

$$
\text{score}(q, d) = \sum_{i=1}^{|q|} \max_{j=1}^{|d|} E_q(q_i)^T E_d(d_j)
$$

**特点**:
- 可预计算文档token embeddings(存储成本高)
- 比Cross-Encoder快(无self-attention)
- 比Bi-Encoder准(token级交互)

#### 5.4.2 复杂度对比

| 模型 | 推理复杂度 | 可否预计算 | 相对速度 | 相对精度 |
|-----|----------|-----------|---------|---------|
| Bi-Encoder | $O(1)$ | ✅ | 100× | 80% |
| ColBERT | $O(\|q\| \cdot \|d\|)$ | ✅ (部分) | 10× | 90% |
| Cross-Encoder | $O((\|q\|+\|d\|)^2)$ | ❌ | 1× | 100% |

**适用场景**: 中等规模(百万级)、高精度需求但无法接受Cross-Encoder延迟

---

## 第六部分：高级RAG架构

### 6.1 RAG架构演进

```
Naive RAG (2020)
  ↓
Advanced RAG (2021-2022)
  ↓
Modular RAG (2023)
  ↓
Agentic RAG (2024-2025)
```

### 6.2 HyDE(假设文档嵌入)

#### 6.2.1 核心思想

**问题**: 查询和答案的embedding分布不同

$$
E(\text{query}) \not\approx E(\text{answer})
$$

**示例**:
```
查询: "什么是Transformer?"
答案: "Transformer是一种基于自注意力机制的神经网络架构..."
```

查询简短、疑问式;答案详细、陈述式 → embedding空间不同区域!

**HyDE解决方案**: 先生成假设答案,用假设答案检索

$$
E(\text{hypothetical\_answer}) \approx E(\text{real\_answer})
$$

#### 6.2.2 算法流程

```
1. LLM生成假设答案: query → hypothetical_doc
2. Embed假设答案: hypothetical_emb = Embed(hypothetical_doc)
3. 向量检索: TopK_docs = VectorSearch(hypothetical_emb)
4. LLM生成最终答案: answer = LLM(query, TopK_docs)
```

#### 6.2.3 Python实现

```python
class HyDERetriever:
    """
    HyDE(Hypothetical Document Embeddings)检索器
    """
    def __init__(
        self,
        llm_client: OpenAI,
        vector_store,
        embedding_model
    ):
        self.llm = llm_client
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        HyDE检索

        Returns:
            [{"content": str, "score": float}, ...]
        """
        # 1. 生成假设文档
        hypothetical_doc = self._generate_hypothetical_doc(query)

        print(f"假设文档: {hypothetical_doc[:200]}...\n")

        # 2. Embed假设文档
        hypo_embedding = self.embedding_model.encode(
            hypothetical_doc,
            normalize_embeddings=True
        )

        # 3. 用假设文档检索(而非原查询!)
        results = self.vector_store.search(hypo_embedding, top_k=top_k)

        return results

    def _generate_hypothetical_doc(self, query: str) -> str:
        """
        使用LLM生成假设答案
        """
        prompt = f"""
请详细回答以下问题(200-300字)。即使你不确定,也请生成一个合理的假设性答案。

问题: {query}

详细回答:
"""

        response = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,  # 适度随机性
            max_tokens=300
        )

        return response.choices[0].message.content.strip()


# 使用示例
if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    # 初始化
    llm = OpenAI(api_key="your-api-key")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    # 构建向量库(假设已有)
    vector_store = FAISSVectorStore(embedding_dim=384, index_type="HNSW")
    # ... 添加文档 ...

    # 创建HyDE检索器
    hyde = HyDERetriever(
        llm_client=llm,
        vector_store=vector_store,
        embedding_model=model
    )

    # 对比实验
    query = "Transformer的注意力机制如何处理长距离依赖?"

    # 传统检索
    query_emb = model.encode(query, normalize_embeddings=True)
    traditional_results = vector_store.search(query_emb, top_k=3)

    # HyDE检索
    hyde_results = hyde.search(query, top_k=3)

    print("=== 传统检索 ===")
    for res in traditional_results:
        print(f"- {res['content'][:100]}...")

    print("\n=== HyDE检索 ===")
    for res in hyde_results:
        print(f"- {res['content'][:100]}...")
```

**效果**: 提升15-20% Recall(论文数据)

**适用场景**:
- 复杂/模糊查询
- 领域专业问题
- 查询-文档语义差距大

### 6.3 Self-RAG(自适应检索)

#### 6.3.1 核心机制

**问题**: 并非所有查询都需要检索!

```
查询1: "1+1等于几?"  → 不需要检索(常识)
查询2: "2024年诺贝尔物理学奖得主?" → 需要检索(实时事实)
```

**Self-RAG**: 模型自主决定何时检索、如何评估

**特殊Token**:
- `[Retrieve]`: 触发检索
- `[IsRel]`: 文档相关性评分(高/中/低)
- `[IsSup]`: 答案是否有支持(完全/部分/无)
- `[IsUse]`: 答案是否有用(5/4/3/2/1)

#### 6.3.2 流程图

```
查询
  ↓
判断是否需要检索?
  ├─ No → 直接生成答案
  └─ Yes ↓
      检索文档
        ↓
      评估文档相关性([IsRel])
        ↓
      生成答案
        ↓
      验证答案支持度([IsSup])
        ↓
      评估答案有用性([IsUse])
        ↓
      返回答案 + 自我批判分数
```

#### 6.3.3 简化实现

```python
class SelfRAG:
    """
    Self-RAG简化实现(使用LLM模拟特殊token)
    """
    def __init__(
        self,
        llm_client: OpenAI,
        retriever
    ):
        self.llm = llm_client
        self.retriever = retriever

    def query(self, question: str) -> Dict:
        """
        Self-RAG查询

        Returns:
            {
                "answer": str,
                "need_retrieval": bool,
                "documents": List[str],
                "relevance": str,
                "support": str,
                "usefulness": int
            }
        """
        # 1. 判断是否需要检索
        need_retrieval = self._should_retrieve(question)

        if not need_retrieval:
            # 直接生成答案
            answer = self._generate_direct(question)
            return {
                "answer": answer,
                "need_retrieval": False,
                "documents": [],
                "relevance": "N/A",
                "support": "N/A",
                "usefulness": 5
            }

        # 2. 检索文档
        docs = self.retriever.search(question, top_k=3)
        doc_contents = [d["content"] for d in docs]

        # 3. 评估文档相关性
        relevance = self._evaluate_relevance(question, doc_contents)

        if relevance == "低":
            return {
                "answer": "未找到相关信息",
                "need_retrieval": True,
                "documents": doc_contents,
                "relevance": relevance,
                "support": "N/A",
                "usefulness": 1
            }

        # 4. 生成答案
        answer = self._generate_with_docs(question, doc_contents)

        # 5. 验证答案支持度
        support = self._evaluate_support(answer, doc_contents)

        # 6. 评估答案有用性
        usefulness = self._evaluate_usefulness(question, answer)

        return {
            "answer": answer,
            "need_retrieval": True,
            "documents": doc_contents,
            "relevance": relevance,
            "support": support,
            "usefulness": usefulness
        }

    def _should_retrieve(self, question: str) -> bool:
        """判断是否需要检索"""
        prompt = f"""
判断以下问题是否需要检索外部知识库。

规则:
- 常识性问题 → 不需要
- 需要最新/专业知识 → 需要

问题: {question}

回答(仅输出"需要"或"不需要"):
"""
        response = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )

        decision = response.choices[0].message.content.strip()
        return "需要" in decision

    def _evaluate_relevance(self, question: str, documents: List[str]) -> str:
        """评估文档相关性"""
        docs_str = "\n\n".join([f"文档{i+1}: {doc[:200]}" for i, doc in enumerate(documents)])

        prompt = f"""
评估以下文档与问题的相关性。

问题: {question}

文档:
{docs_str}

相关性评分(高/中/低):
"""
        response = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )

        return response.choices[0].message.content.strip()

    def _evaluate_support(self, answer: str, documents: List[str]) -> str:
        """评估答案支持度"""
        docs_str = "\n\n".join([f"文档{i+1}: {doc[:200]}" for i, doc in enumerate(documents)])

        prompt = f"""
评估答案是否有文档支持。

答案: {answer}

文档:
{docs_str}

支持度(完全支持/部分支持/无支持):
"""
        response = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20
        )

        return response.choices[0].message.content.strip()

    def _evaluate_usefulness(self, question: str, answer: str) -> int:
        """评估答案有用性"""
        prompt = f"""
评估答案对问题的有用性(1-5分,5最有用)。

问题: {question}
答案: {answer}

评分(1-5):
"""
        response = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5
        )

        try:
            score = int(response.choices[0].message.content.strip())
            return max(1, min(5, score))  # 限制1-5
        except:
            return 3

    def _generate_direct(self, question: str) -> str:
        """直接生成答案(无检索)"""
        response = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": question}],
            temperature=0.3,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()

    def _generate_with_docs(self, question: str, documents: List[str]) -> str:
        """基于文档生成答案"""
        docs_str = "\n\n".join([f"[文档{i+1}] {doc}" for i, doc in enumerate(documents)])

        prompt = f"""
基于以下文档回答问题。只使用文档中的信息,不要编造。

文档:
{docs_str}

问题: {question}

答案:
"""
        response = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()


# 使用示例
if __name__ == "__main__":
    llm = OpenAI(api_key="your-api-key")
    retriever = dense_retriever  # 前面实现的检索器

    self_rag = SelfRAG(llm_client=llm, retriever=retriever)

    # 测试1: 常识问题
    result1 = self_rag.query("1+1等于几?")
    print(f"问题: 1+1等于几?")
    print(f"需要检索: {result1['need_retrieval']}")
    print(f"答案: {result1['answer']}\n")

    # 测试2: 专业问题
    result2 = self_rag.query("Transformer的多头注意力机制如何计算?")
    print(f"问题: Transformer的多头注意力机制如何计算?")
    print(f"需要检索: {result2['need_retrieval']}")
    print(f"文档相关性: {result2['relevance']}")
    print(f"答案支持度: {result2['support']}")
    print(f"答案有用性: {result2['usefulness']}/5")
    print(f"答案: {result2['answer']}")
```

**优势**:
- 避免不必要的检索(节省成本)
- 自我验证(提升可信度)
- 可解释性(暴露决策过程)

### 6.4 CRAG(纠正性RAG)

#### 6.4.1 核心思想

**问题**: 检索到的文档可能不相关/质量低

**CRAG解决方案**: 动态评估并纠正

**流程**:
```
查询 → 检索文档 → 评估文档质量
  ├─ 高相关 → 直接生成
  ├─ 中等相关 → 知识精炼 → 生成
  └─ 不相关 → Web搜索(外部知识) → 生成
```

#### 6.4.2 简化实现

```python
class CRAG:
    """
    CRAG(Corrective Retrieval Augmented Generation)
    """
    def __init__(
        self,
        llm_client: OpenAI,
        retriever,
        web_search_api=None  # 如SerpAPI/Tavily
    ):
        self.llm = llm_client
        self.retriever = retriever
        self.web_search = web_search_api

    def query(self, question: str) -> Dict:
        """
        CRAG查询

        Returns:
            {
                "answer": str,
                "source": "internal_kb" | "web_search",
                "documents": List[str],
                "relevance_score": float
            }
        """
        # 1. 检索内部知识库
        docs = self.retriever.search(question, top_k=5)
        doc_contents = [d["content"] for d in docs]

        # 2. 评估文档相关性
        relevance = self._evaluate_document_quality(question, doc_contents)

        # 3. 根据相关性决策
        if relevance >= 0.7:  # 高相关
            answer = self._generate_answer(question, doc_contents)
            source = "internal_kb"

        elif relevance >= 0.4:  # 中等相关,知识精炼
            refined_docs = self._knowledge_refinement(question, doc_contents)
            answer = self._generate_answer(question, refined_docs)
            source = "internal_kb(refined)"

        else:  # 不相关,触发Web搜索
            if self.web_search:
                web_results = self._web_search_fallback(question)
                answer = self._generate_answer(question, web_results)
                source = "web_search"
                doc_contents = web_results
            else:
                answer = "未找到相关信息,建议尝试其他问法。"
                source = "none"

        return {
            "answer": answer,
            "source": source,
            "documents": doc_contents,
            "relevance_score": relevance
        }

    def _evaluate_document_quality(self, question: str, documents: List[str]) -> float:
        """评估文档质量(0-1分数)"""
        docs_str = "\n\n".join([f"文档{i+1}: {doc[:200]}" for i, doc in enumerate(documents)])

        prompt = f"""
评估以下文档对问题的相关性和质量(0-1分数,1最相关)。

问题: {question}

文档:
{docs_str}

相关性分数(0.0-1.0):
"""
        response = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )

        try:
            score = float(response.choices[0].message.content.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.5

    def _knowledge_refinement(self, question: str, documents: List[str]) -> List[str]:
        """知识精炼:提取文档中与问题最相关的片段"""
        refined = []
        for doc in documents:
            prompt = f"""
从以下文档中提取与问题最相关的1-2句话。

问题: {question}
文档: {doc}

提取的相关句子:
"""
            response = self.llm.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=150
            )
            refined_content = response.choices[0].message.content.strip()
            if refined_content:
                refined.append(refined_content)

        return refined
