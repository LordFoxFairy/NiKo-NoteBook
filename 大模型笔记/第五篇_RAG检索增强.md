# 第六篇:RAG检索增强生成

> RAG核心架构与高级技术:从Naive到GraphRAG的演进之路

**适合人群**: AI应用开发者、架构师
**预计时间**: 8-10 小时
**前置知识**: 大模型基础、向量检索概念

---

## 本篇概览

本篇深入**RAG技术栈**:
- RAG数学原理与架构演进
- 向量检索理论(ANN/HNSW)
- 重排序算法(Cross-Encoder/ColBERT)
- 高级RAG技术(HyDE/Self-RAG/GraphRAG)
- 企业级系统设计

---

## 第1章:RAG数学基础

### 1.1 RAG的形式化定义

**传统LLM生成**:
$$
P(y|x) = \prod_{t=1}^T P(y_t | y_{<t}, x)
$$

**RAG生成**:
$$
P_{\text{RAG}}(y|x) = \sum_{d \in \mathcal{D}} P(d|x) \cdot P(y|x, d)
$$

其中:
- $\mathcal{D}$: 知识库文档集合
- $P(d|x)$: 检索模型(retriever)
- $P(y|x, d)$: 生成模型(generator)

**边际化**: 对所有可能文档求期望!

### 1.2 检索相关性理论

#### 1.2.1 BM25算法

**经典检索函数**:
$$
\text{BM25}(q, d) = \sum_{i=1}^n \text{IDF}(q_i) \cdot \frac{f(q_i, d) \cdot (k_1 + 1)}{f(q_i, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}
$$

其中:
- $f(q_i, d)$: 词频(term frequency)
- $|d|$: 文档长度
- $\text{avgdl}$: 平均文档长度
- $k_1, b$: 调节参数(通常 $k_1=1.5, b=0.75$)

**IDF定义**:
$$
\text{IDF}(q_i) = \log \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5}
$$

- $N$: 总文档数
- $n(q_i)$: 包含词 $q_i$ 的文档数

**直觉**: 常见词权重低,稀有词权重高!

#### 1.2.2 向量检索(Dense Retrieval)

**Embedding相似度**:
$$
\text{sim}(q, d) = \frac{E(q)^T E(d)}{\|E(q)\| \|E(d)\|} = \cos(\theta)
$$

**优势**:
- 语义匹配(而非字面匹配)
- "iPhone"和"苹果手机"可以匹配

**对比BM25**:
| 方法 | 匹配类型 | 召回率 | 精确度 |
|------|---------|--------|--------|
| BM25 | 字面 | 高 | 中 |
| Dense | 语义 | 中 | 高 |

**最佳实践**: 混合检索!

### 1.3 近似最近邻搜索(ANN)

#### 1.3.1 暴力搜索复杂度

**问题**: 在 $N$ 个向量中找Top-K最相似

**暴力**:
$$
\text{Time} = O(N \cdot d)
$$

对于 $N=10^9, d=768$,每次查询需要数秒!

#### 1.3.2 HNSW算法

**Hierarchical Navigable Small World**

**核心思想**: 多层图结构

**数学模型**:

每层是一个图 $G_l = (V_l, E_l)$:
- 第0层: 所有节点
- 第$l$层: 指数衰减采样 $|V_l| = |V_0| \cdot e^{-l}$

**搜索算法**:
```
从顶层开始
for each layer l from top to 0:
    贪婪搜索到最近邻居
    进入下一层,从该邻居开始
返回第0层的TopK
```

**复杂度**:
- 构建: $O(N \log N)$
- 查询: $O(\log N)$ (亚线性!)

**参数**:
- $M$: 每层最大连接数(16-64)
- $ef_{\text{construction}}$: 构建时搜索宽度(100-200)
- $ef_{\text{search}}$: 查询时搜索宽度(与召回率权衡)

**Trade-off**:
$$
\text{Recall} \uparrow \quad \Leftrightarrow \quad ef_{\text{search}} \uparrow \quad \Leftrightarrow \quad \text{Latency} \uparrow
$$

### 1.4 重排序理论

#### 1.4.1 两阶段检索

**Pipeline**:
```
查询 → 召回(ANN, Top-100) → 重排序(精准模型, Top-10) → 生成
```

**为什么有效?**

召回阶段:
- 模型简单(Bi-Encoder)
- 速度快,可处理百万级

重排序阶段:
- 模型复杂(Cross-Encoder)
- 精度高,只处理百个

**成本对比**:
$$
\text{Total Cost} = N \cdot C_{\text{recall}} + K \cdot C_{\text{rerank}}
$$

若 $K \ll N$,总成本接近召回成本!

#### 1.4.2 Cross-Encoder数学

**Bi-Encoder** (召回):
$$
\text{score}(q, d) = E_q(q)^T E_d(d)
$$

查询和文档独立编码。

**Cross-Encoder** (重排):
$$
\text{score}(q, d) = f([q; d])
$$

联合编码,捕获交互!

**实现**:
```python
# Bi-Encoder
q_emb = encoder_q(query)
d_emb = encoder_d(doc)
score = cosine(q_emb, d_emb)

# Cross-Encoder
combined = tokenizer(query + doc)
score = model(combined)  # 输出scalar
```

**性能提升**: 10-20% MRR@10!

#### 1.4.3 ColBERT晚期交互

**创新**: 延迟交互(late interaction)

**流程**:
1. 分别编码查询和文档的每个token
2. 计算token级相似度矩阵
3. MaxSim聚合

**数学**:
$$
\text{score}(q, d) = \sum_{i=1}^{|q|} \max_{j=1}^{|d|} E_q(q_i)^T E_d(d_j)
$$

**优势**:
- 比Cross-Encoder快(可预计算文档embedding)
- 比Bi-Encoder准(token级交互)

**复杂度**:
- Bi-Encoder: $O(1)$
- ColBERT: $O(|q| \cdot |d|)$
- Cross-Encoder: $O((|q|+|d|)^2)$ (self-attention)

---

## 第2章:RAG架构演进

### 2.1 Naive RAG

**流程**:
```
问题 → Embedding → 向量检索Top-K → 拼接Prompt → LLM → 答案
```

**问题**:
1. **检索质量**: 语义不匹配
2. **上下文噪声**: 无关文档干扰
3. **幻觉**: 模型可能忽略检索内容

### 2.2 Advanced RAG

**改进点**:

**a) 查询改写**:
$$
q' = \text{Rewrite}(q, \text{conversation\_history})
$$

**示例**:
```
原查询: "它的价格是多少?"
改写后: "iPhone 15 Pro的价格是多少?"
```

**b) HyDE (Hypothetical Document Embeddings)**:

**核心思想**: 先生成假设答案,再检索!

**流程**:
```
问题 → LLM生成假设文档 → Embed假设文档 → 检索相似文档
```

**数学**:
$$
d_{\text{hypo}} = \text{LLM}(q)
$$
$$
\mathcal{D}_{\text{retrieved}} = \text{TopK}(\text{sim}(E(d_{\text{hypo}}), E(d_i)))
$$

**为什么有效?**

问题和答案在语义空间的分布不同:
$$
E(q) \not\approx E(d_{\text{answer}})
$$

但假设答案和真实答案接近:
$$
E(d_{\text{hypo}}) \approx E(d_{\text{answer}})
$$

**实验**: 提升15-20% Recall!

**c) 重排序**:

Cross-Encoder精排Top-K结果。

**d) 答案引用**:

要求模型标注来源:
```
答案: ...
引用: [文档1, 第3段]
```

### 2.3 Modular RAG

**思想**: RAG是可组合的模块!

**模块化组件**:
```
Retrieval Modules:
  - 稀疏检索(BM25)
  - 密集检索(HNSW)
  - 混合检索
  
Processing Modules:
  - 查询改写
  - 查询扩展
  - HyDE
  
Ranking Modules:
  - Cross-Encoder
  - LLM-as-Ranker
  
Generation Modules:
  - Direct Generation
  - Chain-of-Thought
  - Self-Refine
```

**动态组合**:

根据查询类型选择pipeline:
```python
def route_query(query):
    if is_factual(query):
        return ["dense_retrieval", "cross_encoder", "direct_gen"]
    elif is_complex(query):
        return ["query_decompose", "multi_retrieval", "cot_gen"]
    else:
        return ["bm25", "simple_gen"]
```

---

## 第3章:高级RAG技术

### 3.1 Self-RAG

**核心思想**: 模型自己决定何时检索!

**流程**:
```
问题 → 判断是否需要检索 → 
  if 需要: 检索 → 生成 → 自我验证
  else: 直接生成
```

**特殊Token**:
- `[Retrieve]`: 触发检索
- `[IsRel]`: 文档相关性评分
- `[IsSup]`: 答案是否有支持
- `[IsUse]`: 答案是否有用

**训练**:

强化学习优化检索决策:
$$
\mathcal{L} = \mathcal{L}_{\text{generation}} + \alpha \mathcal{L}_{\text{retrieval\_decision}} + \beta \mathcal{L}_{\text{critique}}
$$

**优势**:
- 避免不必要检索(节省成本)
- 自我修正错误

### 3.2 GraphRAG

**动机**: 文档间存在结构化关系!

**传统RAG**:
```
文档1: 张三是CEO
文档2: 李四是CTO
查询: 张三和李四是什么关系?
```

传统RAG难以回答跨文档关系!

**GraphRAG解决方案**:

**1. 构建知识图谱**:
```
实体抽取: (张三, 李四, 公司X)
关系抽取: (张三, CEO_of, 公司X), (李四, CTO_of, 公司X)
```

**2. 图查询**:
```cypher
MATCH (p1:Person {name: "张三"})-[:WORKS_AT]->(c:Company)
      <-[:WORKS_AT]-(p2:Person {name: "李四"})
RETURN p1.role, p2.role, c.name
```

**3. 子图检索 + LLM生成**:
$$
\text{Answer} = \text{LLM}(\text{query}, G_{\text{sub}})
$$

**数学形式**:

传统RAG:
$$
P(y|x) = \sum_{d \in \mathcal{D}} P(d|x) P(y|x,d)
$$

GraphRAG:
$$
P(y|x) = \sum_{G_{\text{sub}} \in \mathcal{G}} P(G_{\text{sub}}|x) P(y|x, G_{\text{sub}})
$$

### 知识图谱检索

**原理**: 将查询转换为图数据库查询语言(如Cypher)

**流程**:
1. 查询理解: "Transformer的注意力机制" → 实体抽取
2. Cypher生成:
   ```cypher
   MATCH (c:Concept {name: "Transformer"})-[:HAS_COMPONENT]->(m:Mechanism {name: "Attention"})
   RETURN m.description
   ```
3. 图数据库执行查询(Neo4j/ArangoDB)
4. 结果后处理

**优势**: 结构化知识,推理能力强
**局限**: 需要知识图谱构建,覆盖有限

**框架支持**:
- LangChain: `GraphCypherQAChain`(自动生成Cypher)
- LlamaIndex: `KnowledgeGraphQueryEngine`

### 3.3 RAPTOR (Recursive Abstractive Processing)

**问题**: 长文档检索粒度冲突

- 细粒度(chunk=512): 精确,但缺乏全局视角
- 粗粒度(chunk=4096): 全局,但噪声大

**RAPTOR方案**: 递归摘要 + 多层索引

**算法**:
```
Level 0: 原始chunks (512 tokens)
  ↓ 聚类 + 摘要
Level 1: 摘要chunks (1024 tokens)
  ↓ 聚类 + 摘要
Level 2: 高层摘要 (2048 tokens)
```

**检索**: 在所有层级检索,组合结果!

**数学**:

层级表示:
$$
h_l = \text{Summarize}(\text{Cluster}(h_{l-1}))
$$

多层检索:
$$
\mathcal{D}_{\text{final}} = \bigcup_{l=0}^L \text{TopK}_l(q, h_l)
$$

**效果**: 多文档QA提升20%!

---

## 第4章:评估与优化

### 4.1 RAG评估指标

#### 4.1.1 检索质量

**Recall@K**:
$$
\text{Recall@K} = \frac{|\text{Retrieved@K} \cap \text{Relevant}|}{|\text{Relevant}|}
$$

**MRR (Mean Reciprocal Rank)**:
$$
\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}
$$

其中 $\text{rank}_i$ 是第一个相关文档的位置。

**NDCG@K** (Normalized Discounted Cumulative Gain):
$$
\text{DCG@K} = \sum_{i=1}^K \frac{2^{\text{rel}_i} - 1}{\log_2(i+1)}
$$
$$
\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}
$$

考虑排序位置的加权!

#### 4.1.2 生成质量

**RAGAS框架**:

**a) Context Relevance** (上下文相关性):
$$
\text{ContextRel} = \frac{|\text{Relevant Sentences}|}{|\text{Total Sentences}|}
$$

**b) Faithfulness** (忠实度):

答案是否基于上下文?
$$
\text{Faithfulness} = \frac{|\text{Supported Claims}|}{|\text{Total Claims}|}
$$

使用NLI模型验证!

**c) Answer Relevance** (答案相关性):
$$
\text{AnswerRel} = \text{Similarity}(\text{Question}, \text{Generated Answer})
$$

**综合评分**:
$$
\text{RAGAS} = \sqrt[3]{\text{ContextRel} \times \text{Faithfulness} \times \text{AnswerRel}}
$$

几何平均,平衡三个维度!

### 4.2 优化策略

#### 4.2.1 Chunk大小优化

**Trade-off**:
- 小chunk: 精确,但语义碎片化
- 大chunk: 连贯,但噪声多

**实验数据**:
| Chunk Size | Recall | Precision | Context Noise |
|-----------|--------|-----------|---------------|
| 256 | 0.72 | 0.68 | 低 |
| 512 | 0.78 | 0.74 | 中 |
| 1024 | 0.75 | 0.70 | 高 |

**最优**: 512-768 tokens

**重叠策略**:
```python
chunks = []
for i in range(0, len(text), chunk_size - overlap):
    chunk = text[i:i+chunk_size]
    chunks.append(chunk)
```

`overlap=50-100` tokens避免语义截断!

#### 4.2.2 混合检索策略(2025最佳实践)

**核心思想**: 结合稀疏检索(BM25)和密集检索(向量)的互补优势

**为什么需要混合检索?**

| 检索方式 | 优势 | 劣势 | 适用场景 |
|---------|------|------|---------|
| **BM25稀疏检索** | 字面匹配精确、速度快、无需训练 | 无法理解语义、同义词miss | 专有名词、代码、精确关键词 |
| **向量密集检索** | 语义理解、跨语言、同义词匹配 | 字面不匹配时召回低、计算成本高 | 概念性问题、语义搜索 |
| **混合检索** | 兼顾字面和语义 | 融合算法复杂度 | 生产环境标准方案 |

**实际案例**:
```
查询: "iPhone稳定性问题"
BM25召回: "iPhone 15 Pro系统崩溃修复指南" (字面匹配 ✓)
向量召回: "苹果手机可靠性测试报告" (语义匹配,但无"iPhone"关键词)
混合结果: 两者都包含,覆盖更全面!
```

---

##### a) RRF融合算法(Reciprocal Rank Fusion)

**数学公式**:
$$
\text{RRF}(d) = \sum_{r \in \text{Retrievers}} \frac{1}{k + \text{rank}_r(d)}
$$

其中:
- $d$: 文档
- $\text{rank}_r(d)$: 文档在检索器 $r$ 中的排名(1-based)
- $k$: 常数,通常 $k=60$(论文推荐值)

**为什么RRF有效?**

1. **无需分数归一化**: 不同检索器的分数量纲不同,RRF只用排名
2. **对异常值鲁棒**: 某个检索器失效时不会主导结果
3. **无参数训练**: 开箱即用,不需要标注数据

**RRF工作原理示例**:

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
  文档A: 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325
  文档B: 1/(60+3) + 1/(60+1) = 0.0159 + 0.0164 = 0.0323
  文档C: 1/(60+2) + 0 = 0.0161
  文档D: 0 + 1/(60+3) = 0.0159

最终排序: A > B > C > D (综合两种检索优势!)
```

**完整实现**:

```python
from typing import List, Dict, Tuple
from collections import defaultdict

def reciprocal_rank_fusion(
    results_list: List[List[str]],
    k: int = 60
) -> List[Tuple[str, float]]:
    """
    RRF算法实现

    Args:
        results_list: 多个检索器的结果列表,每个结果是文档ID列表(按排名排序)
        k: RRF常数,默认60

    Returns:
        融合后的文档及其分数,按分数降序排列
    """
    rrf_scores = defaultdict(float)

    # 遍历每个检索器的结果
    for retriever_results in results_list:
        # rank从1开始(论文定义)
        for rank, doc_id in enumerate(retriever_results, start=1):
            rrf_scores[doc_id] += 1.0 / (k + rank)

    # 按分数降序排序
    sorted_results = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_results


# 使用示例(伪代码)
# 1. 并行执行两种检索
bm25_results = bm25_search(query="Transformer的自注意力机制", top_k=10)
vector_results = vector_search(query="Transformer的自注意力机制", top_k=10)

# 2. RRF融合
fused_results = reciprocal_rank_fusion([bm25_results, vector_results], k=60)

# 3. 取Top-K最终结果
final_docs = fused_results[:5]
```

---

##### b) 框架实现方案

**方案2: 框架实现**

| 框架 | 类名 | 特点 |
|------|------|------|
| LangChain | `EnsembleRetriever` | RRF融合,k固定=60,权重可调 |
| LlamaIndex | `QueryFusionRetriever` | 支持自定义融合函数 |
| Haystack | `JoinDocuments` | 支持RRF/加权平均/最大值融合 |

**权重调优策略**:
- 专业术语密集(代码/法律): BM25权重0.7, 向量0.3
- 概念性问答(FAQ): BM25权重0.3, 向量0.7
- 通用场景: 均衡0.5/0.5

---

##### c) 高级混合检索:多阶段Pipeline

**生产级架构(2025推荐)**:

```
查询
  ↓
[阶段1] 混合召回(BM25 + 向量) → Top-100
  ↓
[阶段2] Cross-Encoder重排序 → Top-20
  ↓
[阶段3] LLM精排(可选) → Top-5
  ↓
生成答案
```

**实现流程(伪代码)**:

```
function multi_stage_retrieve(query, k_final=5):
    # 阶段1: 混合召回Top-100
    bm25_docs = bm25_search(query, k=50)
    vector_docs = vector_search(query, k=50)

    # RRF融合
    fused_results = reciprocal_rank_fusion([bm25_docs, vector_docs], k=60)
    top100_docs = fused_results[:100]

    # 阶段2: Cross-Encoder重排序Top-20
    for doc in top100_docs:
        doc.rerank_score = CrossEncoder(query, doc.content)

    top20_docs = sort_by_rerank_score(top100_docs)[:20]

    # 阶段3: LLM精排(可选)
    # top5_docs = llm_rerank(query, top20_docs, k=5)

    return top20_docs[:k_final]
```

**推荐模型**:
- CrossEncoder: `ms-marco-MiniLM-L-12-v2`(快速), `bge-reranker-large`(高精度)
- LLM精排: GPT-4o-mini, Claude Haiku

---

##### d) 性能优化与最佳实践

**1. 参数调优**

**k值敏感性分析**(离线测试):

| k值 | NDCG@10 | 说明 |
|-----|---------|------|
| k=10 | 0.72 | k值过小 |
| k=30 | 0.75 | 性能提升 |
| k=60 | 0.78 | 最优(论文推荐) |
| k=100 | 0.77 | k值过大,轻微下降 |

**评估方法**: 使用NDCG指标(见下文)对不同k值进行离线测试

**2. 性能基准(2025数据)**

| 检索策略 | Recall@10 | MRR | 延迟 | 成本 |
|---------|-----------|-----|------|------|
| BM25单独 | 0.65 | 0.58 | 20ms | 低 |
| 向量单独 | 0.72 | 0.64 | 50ms | 中 |
| RRF混合 | **0.82** | **0.73** | 70ms | 中 |
| + Cross-Encoder | **0.88** | **0.81** | 300ms | 高 |

**3. 生产环境清单**

**优化策略**:
- 缓存热查询: 使用LRU缓存(maxsize=1000)存储常见查询结果
- 异步检索: BM25和向量检索并行执行(asyncio.gather)
- 监控指标: 记录查询延迟、结果数量、平均文档长度,发送到监控系统(Prometheus/DataDog)

**4. 常见陷阱**

| 问题 | 原因 | 解决方案 |
|-----|------|---------|
| BM25召回为空 | 文档未分词/语言不匹配 | 检查tokenizer,确保与文档语言一致 |
| 向量检索速度慢 | 索引未优化 | 使用HNSW索引,调整`ef_search` |
| 融合结果不理想 | 权重不合理 | A/B测试调优,监控各检索器命中率 |
| 内存占用高 | 同时加载两套索引 | 使用共享存储,延迟加载 |

---

##### e) 对比:其他融合方法

**1. 加权分数融合(不推荐)**

$$
\text{score}(d) = \alpha \cdot \text{norm}(s_{\text{BM25}}(d)) + (1-\alpha) \cdot \text{norm}(s_{\text{vector}}(d))
$$

**问题**:
- 分数归一化困难(BM25和向量分数量纲完全不同)
- 需要标注数据调参
- 对异常值敏感

**2. 级联检索(适合特定场景)**

```python
# 先BM25粗筛,再向量精排
def cascaded_retrieval(query, k=5):
    # 第1步:BM25召回Top-50
    candidates = bm25_retriever.invoke(query, k=50)

    # 第2步:向量重排
    candidate_embeddings = [embed(doc.page_content) for doc in candidates]
    query_embedding = embed(query)

    similarities = cosine_similarity([query_embedding], candidate_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:k]

    return [candidates[i] for i in top_indices]
```

**适用**: 海量文档(>10M),需要降低向量计算成本

---

**总结:混合检索关键要点**

1. **RRF是2025标准方案**: 无需训练,鲁棒性强,k=60是经验最优值
2. **权重调优**: 专业领域偏BM25(0.7),通用场景平衡(0.5)
3. **多阶段架构**: 混合召回 → 重排序 → LLM精排(可选)
4. **性能提升**: 相比单一检索,NDCG提升10-15%,Recall提升15-20%
5. **生产要点**: 缓存、异步、监控缺一不可

**推荐阅读**:
- [Weaviate: Hybrid Search Explained](https://weaviate.io/blog/hybrid-search-explained)
- [OpenSearch: Hybrid Search Best Practices](https://opensearch.org/blog/building-effective-hybrid-search-in-opensearch-techniques-and-best-practices/)
- [Qdrant: Hybrid Search with Query API](https://qdrant.tech/articles/hybrid-search/)

---

## 总结

### 核心要点

1. **RAG是边际化**: 对所有可能文档求期望
2. **两阶段检索**: 召回(快) + 重排(准)
3. **HNSW是标准**: 亚线性查询复杂度
4. **HyDE提升召回**: 生成假设答案再检索
5. **GraphRAG解决关系**: 多跳推理必备
6. **RAGAS评估**: 上下文相关性 × 忠实度 × 答案相关性

### 技术栈选择

| 场景 | 召回 | 重排 | 向量库 |
|------|------|------|--------|
| **小规模** (<10K文档) | BM25 | - | Chroma |
| **中规模** (10K-1M) | HNSW | Cross-Encoder | Qdrant |
| **大规模** (>1M) | HNSW | ColBERT | Weaviate |
| **图结构** | 图遍历 | LLM-Rerank | Neo4j |

### 实践建议

**准确性优先**:
```
BM25+Dense混合 → Cross-Encoder重排 → HyDE → GraphRAG
```

**速度优先**:
```
Dense检索(HNSW) → Top-10直接生成
```

**成本优先**:
```
BM25召回 → 缓存 → 小模型生成
```

---

**推荐资源**:
- [RAG Survey](https://arxiv.org/abs/2312.10997) - 全面综述
- [HyDE论文](https://arxiv.org/abs/2212.10496)
- [Self-RAG](https://arxiv.org/abs/2310.11511)
- [GraphRAG](https://arxiv.org/abs/2404.16130)
- [RAGAS框架](https://github.com/explodinggradients/ragas)
```

**局限性**:
- 检索质量完全依赖语义相似度
- 无法处理复杂查询
- 检索噪音影响生成质量

#### 13.2.2 Advanced RAG(高级RAG)

**核心改进**:

**1. 查询改写(Query Rewriting)**

```python
class QueryRewriter:
    def __init__(self, llm):
        self.llm = llm

    def rewrite(self, query: str):
        """将用户问题改写为更利于检索的形式"""
        prompt = f"""
将以下用户问题改写为3个不同角度的检索查询:

原始问题: {query}

生成3个查询(JSON格式):
{{
  "queries": ["查询1", "查询2", "查询3"]
}}
"""
        response = self.llm.invoke(prompt)
        queries = json.loads(response.content)["queries"]
        return queries

# 多查询检索
rewriter = QueryRewriter(llm)
queries = rewriter.rewrite("大模型如何处理长文本?")
# → ["Transformer长序列处理方法", "位置编码扩展技术", "稀疏注意力机制"]

all_docs = []
for q in queries:
    docs = vectorstore.similarity_search(q, k=2)
    all_docs.extend(docs)

# 去重
unique_docs = list({doc.page_content: doc for doc in all_docs}.values())
```

**2. 混合检索(Hybrid Search)**

结合语义检索(向量)和关键词检索(BM25):

$$
score_{hybrid} = \alpha \cdot score_{vector} + (1-\alpha) \cdot score_{BM25}
$$

**混合检索实现(伪代码)**:
```
# 1. 分别执行两种检索
bm25_docs = bm25_search(query, k=5)
vector_docs = vector_search(query, k=5)

# 2. 加权融合(或使用RRF)
ensemble_docs = weighted_fusion(
    bm25_docs, vector_docs,
    weights=[0.4, 0.6]  # BM25:40%, 向量:60%
)
```

**框架**: LangChain的`EnsembleRetriever`封装了上述逻辑

**3. 重排序(Reranking)**

**算法流程(伪代码)**:
```
function rerank(query, docs, top_k=3):
    # 1. 召回候选(粗排)
    candidates = vector_search(query, k=10)

    # 2. CrossEncoder精排
    scores = []
    for doc in candidates:
        score = CrossEncoder(query, doc.content)
        scores.append(score)

    # 3. 排序并取Top-K
    ranked_indices = argsort(scores, descending=True)
    return candidates[ranked_indices[:top_k]]
```

**推荐模型**: `cross-encoder/ms-marco-MiniLM-L-6-v2`, `bge-reranker-base`

**4. 上下文压缩**

**目的**: 避免无关内容干扰LLM

**方法(伪代码)**:
```
function compress_context(query, docs):
    compressed_docs = []
    for doc in docs:
        # LLM提取与query相关的句子
        relevant_sentences = LLM("从以下文档中提取与问题相关的句子:\n" + doc)
        compressed_docs.append(relevant_sentences)

    return compressed_docs
```

**框架**: LangChain的`ContextualCompressionRetriever`

#### 13.2.3 Modular RAG(模块化RAG)

**可插拔架构**:
```python
class ModularRAG:
    def __init__(self):
        self.query_processor = None
        self.retrievers = []
        self.reranker = None
        self.generator = None
        self.post_processor = None

    def add_retriever(self, retriever, weight=1.0):
        """添加检索模块"""
        self.retrievers.append((retriever, weight))

    def set_reranker(self, reranker):
        self.reranker = reranker

    def query(self, question):
        # 1. 查询预处理
        processed_query = self.query_processor(question) if self.query_processor else question

        # 2. 多检索器融合
        all_docs = []
        for retriever, weight in self.retrievers:
            docs = retriever.invoke(processed_query)
            all_docs.extend([(doc, weight) for doc in docs])

        # 3. 重排序
        if self.reranker:
            all_docs = self.reranker.rerank(processed_query, [d[0] for d in all_docs])
        else:
            all_docs = [d[0] for d in all_docs]

        # 4. 生成
        answer = self.generator.generate(processed_query, all_docs)

        # 5. 后处理
        if self.post_processor:
            answer = self.post_processor(answer)

        return answer
```

### 13.3 RAG vs Fine-tuning

| 维度 | RAG | Fine-tuning | 混合方案 |
|------|-----|------------|---------|
| 知识更新 | 实时(更新向量库) | 慢(需重训练) | 定期微调+实时RAG |
| 成本 | 低 | 高 | 中 |
| 可解释性 | 高(可追溯来源) | 低 | 高 |
| 领域适配 | 依赖检索质量 | 知识内化,效果好 | 最佳 |
| 幻觉控制 | 好 | 一般 | 最好 |

**最佳实践**:
- Fine-tune领域理解能力
- RAG提供实时知识
- 结合两者优势

---

## 第14章:高级RAG技术

### 14.1 HyDE(假设文档嵌入)

#### 14.1.1 核心思想

> 问题嵌入 ≠ 答案嵌入,用LLM生成假设答案,用答案去检索

**流程**:
```
问题 → LLM生成假设答案 → Embedding假设答案 → 检索相似文档
```

**原理**:
- 问题:"什么是Transformer?"
- 假设答案:"Transformer是一种基于自注意力机制的神经网络架构..."
- 答案embedding更接近文档embedding

#### 14.1.2 实现流程

**算法(伪代码)**:
```
function HyDE_retrieve(query, k=5):
    # 1. LLM生成假设答案
    prompt = "请详细回答(200字): " + query
    hypothetical_doc = LLM(prompt)

    # 2. Embedding假设答案
    hypo_embedding = Embed(hypothetical_doc)

    # 3. 向量检索(用假设答案而非原问题)
    similar_docs = vector_search(hypo_embedding, top_k=k)

    return similar_docs
```

**效果对比**:
- 问题检索: "Transformer优势" → 可能召回"Transformer应用案例"
- HyDE检索: "Transformer相比RNN可并行训练..." → 召回架构对比文档 ✓

### 14.2 Self-RAG(自我反思RAG)

#### 14.2.1 核心机制

**动态检索与自我验证**:

**算法流程(伪代码)**:
```
function SelfRAG_query(question):
    # 1. 判断是否需要检索
    if not should_retrieve(question):  # LLM判断
        return LLM(question)

    # 2. 检索文档
    docs = retrieve(question)

    # 3. 过滤不相关文档
    relevant_docs = filter(docs, is_relevant_to(question))

    if len(relevant_docs) == 0:
        return "未找到相关信息"

    # 4. 生成答案
    answer = LLM(context=relevant_docs, query=question)

    # 5. 验证答案支持度
    if not is_supported_by(answer, relevant_docs):
        return "无法基于现有信息给出可靠答案"

    return answer
```

**优势**:
- 减少不必要的检索
- 过滤噪音文档
- 验证答案可靠性

### 14.3 RAPTOR(递归摘要树)

#### 14.3.1 原理

> 构建文档摘要的树状结构,支持多粒度检索

```
原始文档(叶子节点)
    ↓ 聚类
中层摘要(100篇 → 10篇摘要)
    ↓ 再聚类
高层摘要(10篇 → 1篇总结)
```

**数学表示**:

$$
\text{Summary}_{level+1} = \text{LLM}(\text{Cluster}(\text{Docs}_{level}))
$$

#### 14.3.2 实现流程

**算法(伪代码)**:
```
function build_raptor_tree(documents, max_depth=3):
    tree[0] = documents  # 叶子层

    for level in 1..max_depth:
        # 1. 聚类(KMeans)
        embeddings = embed(tree[level-1])
        clusters = kmeans_cluster(embeddings, n_clusters=len(tree[level-1])//5)

        # 2. 每个簇生成摘要
        tree[level] = []
        for cluster in clusters:
            summary = LLM("总结以下文档核心内容:\n" + join(cluster))
            tree[level].append(summary)

    return tree

function raptor_retrieve(query, tree, k=5):
    query_emb = embed(query)
    all_results = []

    # 在所有层级检索
    for level, docs in tree.items():
        similarities = cosine_similarity(query_emb, embed(docs))
        top_k_indices = argsort(similarities)[:k]
        all_results.extend(docs[top_k_indices])

    # 按相似度排序
    return sort_by_similarity(all_results)[:k]
```

**实现要点**:
- 聚类算法: KMeans(簇数=当前层文档数/5)
- 摘要生成: LLM提示"总结以下文档共同主题(200字)"
- 多层检索: 在所有层级并行检索,按分数融合

**优势**:
- 同时检索宏观概述和微观细节
- 适合长文档和书籍
- 提升复杂问题回答质量

### 14.4 GraphRAG(知识图谱增强)

#### 14.4.1 核心思想

> 构建实体关系图谱,基于图结构检索

```
文档 → 实体抽取 → 关系抽取 → 知识图谱 → 图遍历检索
```

**示例图谱**:
```
(Transformer) -[提出者]-> (Vaswani)
(Transformer) -[包含]-> (Self-Attention)
(Self-Attention) -[优于]-> (RNN)
(BERT) -[基于]-> (Transformer)
```

#### 14.4.2 实现流程

**算法(伪代码)**:
```
function build_knowledge_graph(documents):
    graph = empty_graph()

    for doc in documents:
        # 1. LLM抽取实体和关系
        entities, relations = LLM_extract_kg(doc)

        # 2. 添加到图
        for entity in entities:
            graph.add_node(entity.name, type=entity.type)

        for rel in relations:
            graph.add_edge(rel.source, rel.target, relation=rel.type)

    return graph

function graph_rag_retrieve(question, graph, hop=2):
    # 1. 从问题抽取查询实体
    query_entities = LLM_extract_entities(question)

    # 2. 图遍历(N-hop邻居)
    related_nodes = []
    for entity in query_entities:
        neighbors = graph.n_hop_neighbors(entity, n=hop)
        related_nodes.extend(neighbors)

    # 3. 提取子图
    subgraph = graph.subgraph(related_nodes)

    # 4. 序列化为三元组
    triples = format_as_triples(subgraph)
    # 例如: (Transformer) -[提出者]-> (Vaswani)

    # 5. LLM基于子图生成答案
    answer = LLM("基于知识图谱:\n" + triples + "\n回答:" + question)

    return answer
```

**实现要点**:
- 实体抽取: LLM提示返回JSON格式(entities + relations)
- 图遍历: N-hop邻居查询(BFS/DFS)
- 子图序列化: 三元组格式(主语-谓语-宾语)

**优势**:
- 捕获实体间复杂关系
- 支持多跳推理
- 可解释性强(路径可视化)

---

## 第15章:RAG系统工程化

### 15.1 评估指标

#### 15.1.1 检索质量

**1. Recall@K**: Top-K中包含正确文档的比例

$$
\text{Recall@K} = \frac{|\{相关文档\} \cap \{Top-K检索结果\}|}{|\{相关文档\}|}
$$

**2. MRR(Mean Reciprocal Rank)**: 第一个相关文档的排名倒数

$$
\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}
$$

**3. NDCG(Normalized Discounted Cumulative Gain)**: 考虑排序质量

$$
\text{DCG@K} = \sum_{i=1}^{K} \frac{rel_i}{\log_2(i+1)}
$$

#### 15.1.2 生成质量

**1. Faithfulness**: 答案是否忠实于检索文档

**评估方法**:
- LLM评分(0-5): 完全基于上下文=5, 大部分基于=3, 矛盾=0
- 提示词: "评估以下答案是否基于给定上下文,无任何臆测"
- 归一化分数: score / 5.0

**2. Answer Relevance**: 答案是否回答了问题

**3. Context Relevance**: 检索文档是否与问题相关

### 15.2 性能优化

#### 15.2.1 缓存策略

**双层缓存设计**:
- 查询缓存: 完全相同查询直接返回答案(命中率80%)
- 文档缓存: 相同查询的检索结果缓存,避免重复embedding
- 实现: 使用查询MD5哈希作为缓存键

**策略**: LRU缓存(最近最少使用淘汰)

#### 15.2.2 异步处理

**并发优化**:
- 检索和LLM调用异步化(asyncio)
- 多查询批处理
- 减少等待时间,提升吞吐量

### 15.3 成本优化

| 策略 | 成本节省 | 说明 |
|------|---------|------|
| 缓存热门查询 | 80% | 相同问题直接返回 |
| 压缩上下文 | 50% | 减少token数 |
| 使用更小模型 | 95% | GPT-4→GPT-3.5/Claude Haiku |
| 批量处理 | 20% | Batch API折扣 |
| 本地Embedding | 100% | 向量化不调用API |

---

## 实战练习

### 练习1: 构建个人知识库RAG

任务:将你的Markdown笔记构建成问答系统

提示:
- 使用LangChain + Chroma
- 实现HyDE或Self-RAG
- 评估检索准确率

### 练习2: 对比三种RAG范式

在同一数据集上对比:
- Naive RAG
- Advanced RAG(混合检索+重排序)
- GraphRAG

测试指标: Recall@3, Answer Relevance

---

## 延伸阅读

### 论文
- [HyDE: Precise Zero-Shot Dense Retrieval](https://arxiv.org/abs/2212.10496)
- [Self-RAG: Learning to Retrieve, Generate, and Critique](https://arxiv.org/abs/2310.11511)
- [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059)
- [GraphRAG: A Graph-Based Approach to Retrieval-Augmented Generation](https://arxiv.org/abs/2404.16130)

### 开源项目
- [LlamaIndex](https://github.com/run-llama/llama_index) - RAG框架
- [LangChain](https://github.com/langchain-ai/langchain) - 应用框架
- [Haystack](https://github.com/deepset-ai/haystack) - NLP框架

---

**下一篇**: [第五篇:AI Agent智能体](第五篇_AI_Agent智能体.md)
```
