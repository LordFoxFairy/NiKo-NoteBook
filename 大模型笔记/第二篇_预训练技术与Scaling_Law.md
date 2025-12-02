# 第二篇:预训练技术与Scaling Law

> 从语言模型理论到大规模训练的完整技术栈

**适合人群**: 算法研究员、深度学习工程师
**预计时间**: 8-10小时
**前置知识**: 第一篇(Transformer架构)

---

## 第0章:语言模型的数学基础

### 0.1 概率语言模型定义

**形式化定义**:

语言模型是定义在词汇表 $\mathcal{V}$ 上的序列概率分布:
$$
P: \mathcal{V}^* \rightarrow [0, 1]
$$

满足归一化约束:
$$
\sum_{w_1, \ldots, w_n \in \mathcal{V}^*} P(w_1, \ldots, w_n) = 1
$$

**自回归分解** (链式法则):
$$
P(w_1, \ldots, w_n) = \prod_{t=1}^n P(w_t \mid w_{<t})
$$

其中 $w_{<t} = (w_1, \ldots, w_{t-1})$ 为历史上下文。

### 0.2 交叉熵与困惑度

**交叉熵损失**:

给定真实数据分布 $P_{\text{data}}$ 和模型分布 $P_\theta$,交叉熵定义为:
$$
H(P_{\text{data}}, P_\theta) = -\mathbb{E}_{x \sim P_{\text{data}}}[\log P_\theta(x)]
$$

实践中使用经验交叉熵:
$$
\hat{H} = -\frac{1}{N} \sum_{i=1}^N \log P_\theta(x_i)
$$

**为什么最小化交叉熵?**

信息论基本定理:
$$
H(P, Q) = H(P) + D_{KL}(P \| Q)
$$

其中:
- $H(P)$: 数据分布的熵(常数)
- $D_{KL}(P \| Q)$: KL散度(非负)

最小化交叉熵 $\Leftrightarrow$ 最小化KL散度 $\Leftrightarrow$ 最大化似然!

**困惑度**(Perplexity):
$$
\text{PPL} = \exp(H) = \exp\left(-\frac{1}{N}\sum_{i=1}^N \log P_\theta(w_i \mid w_{<i})\right)
$$

**直觉解释**: 模型在每个时间步平均"困惑"于多少个单词。

**理论意义**:

困惑度是**有效词汇量**的度量:
$$
\text{PPL} = \text{等效均匀分布的支撑大小}
$$

例如: PPL=20 意味着模型等效于从20个等概率单词中随机选择。

### 0.3 Scaling Law的理论基础

**幂律关系**(Power Law):

经验观察: 损失 $L$ 与计算量 $C$、参数量 $N$、数据量 $D$ 满足:
$$
L(N) \propto N^{-\alpha}, \quad L(D) \propto D^{-\beta}, \quad L(C) \propto C^{-\gamma}
$$

**为什么是幂律?**

1. **统计物理解释**: 

神经网络学习过程类似统计系统的相变:
$$
\text{自由能} \propto \text{系统尺度}^{-\alpha}
$$

2. **维度灾难缓解**:

高维空间中,样本复杂度与维度的关系:
$$
N_{\text{samples}} \propto d^{-\alpha}
$$

**不可约损失**(Irreducible Loss):

理论下界:
$$
L_{\min} = H(P_{\text{data}})
$$

即数据分布的熵,由贝叶斯误差决定。

实际损失可分解:
$$
L(N, D) = L_{\min} + L_{\text{approximation}}(N) + L_{\text{estimation}}(D)
$$

---

## 第1章: Scaling Law:大模型的幂律定律

### 1.1 Kaplan Scaling Law (2020)

Kaplan等人在OpenAI的研究中发现,模型性能与三个关键因素呈现幂律关系:

**核心公式**:
```
L(N) = (Nc/N)^αN
L(D) = (Dc/D)^αD
L(C) = (Cc/C)^αC
```

其中:
- `L`: 交叉熵损失 (Cross-Entropy Loss)
- `N`: 模型参数量 (非嵌入层参数)
- `D`: 数据集大小 (tokens数量)
- `C`: 计算量 (PetaFLOP-days)
- `α`: 幂律指数
- `Nc, Dc, Cc`: 特征尺度常数

**关键发现**:
1. **参数量主导**: 在计算预算固定时,应优先增加模型大小而非训练时长
2. **幂律指数**: αN ≈ 0.076, αD ≈ 0.095, αC ≈ 0.050
3. **最优配比**: 当模型参数翻倍时,数据量只需增加约1.5倍

**数学推导**:

从经验观察出发,假设损失函数可分解为:
```
L(N, D) = L0 + A/N^α + B/D^β
```

其中L0是不可约损失(irreducible loss),代表贝叶斯误差。

通过最小二乘拟合实验数据,发现:
- 当D足够大时: `L(N) ∝ N^(-0.076)`
- 当N足够大时: `L(D) ∝ D^(-0.095)`

**计算预算约束**:

给定计算预算C,有关系:
```
C ≈ 6 × N × D
```

(因子6来自前向+反向传播的FLOPs计算)

在约束C=常数下,最优化损失函数:
```
min L(N, D)  s.t.  6ND = C
```

使用拉格朗日乘数法:
```
∂L/∂N = ∂L/∂D  =>  αA/N^(α+1) = βB/D^(β+1)
```

结合约束条件,得到最优参数比:
```
N_opt ∝ C^(β/(α+β))
D_opt ∝ C^(α/(α+β))
```

### 1.2 Chinchilla Scaling Law (2022)

DeepMind重新审视了Kaplan定律,发现**GPT-3及其同时代模型严重数据不足**。

**核心论点**:
- GPT-3 (175B参数) 仅训练了300B tokens
- Chinchilla (70B参数) 训练了1.4T tokens,性能超越GPT-3

**修正公式**:

Hoffmann等人提出新的幂律关系:
```
L(N, D) = E + A/N^α + B/D^β
```

但通过三种不同拟合方法(IsoFLOP profiles、外推法、参数化法)得到一致结论:

**最优数据量**:
```
D_opt = N / 20
```

即:每个参数应对应约20个训练tokens。

**为什么Chinchilla认为GPT-3数据不足?**

| 模型 | 参数量 | 训练Tokens | Tokens/Param比率 |
|------|--------|-----------|-----------------|
| GPT-3 | 175B | 300B | 1.7 |
| Gopher | 280B | 300B | 1.1 |
| Chinchilla | 70B | 1.4T | **20** |

GPT-3的数据效率比Chinchilla差了10倍以上!

**数学证明**:

在固定FLOPs预算C下,设:
```
C = 6 × N × D  (每个token的FLOPs ≈ 6N)
```

最小化损失函数:
```
L(N, D) = E + A/N^α + B/D^β
```

求偏导并令其为零:
```
∂L/∂N = -αA/N^(α+1) = 0  (在约束下)
∂L/∂D = -βB/D^(β+1) = 0
```

使用拉格朗日乘数法处理约束6ND=C:
```
-αA/N^(α+1) = λ × 6D
-βB/D^(β+1) = λ × 6N
```

消去λ:
```
αA/N^(α+1) × N = βB/D^(β+1) × D
=> αA/N^α = βB/D^β
```

Chinchilla实验拟合出α ≈ β ≈ 0.34,代入上式并结合6ND=C:
```
D_opt/N_opt ≈ 20
```

**实践影响**:
- LLaMA (7B-65B): 1T-1.4T tokens
- LLaMA2 (70B): 2T tokens
- Mistral 7B: 数万亿tokens (未公开具体数字)

现代模型普遍采用Chinchilla配比,而非Kaplan的"大模型少数据"策略。

---

## 2. 预训练目标函数

### 2.1 Causal Language Modeling (CLM)

**定义**: 从左到右预测下一个token。

**数学形式**:
```
L_CLM = -∑(i=1 to n) log P(x_i | x_<i; θ)
```

其中x_<i = (x_1, ..., x_(i-1))是前文。

**自回归分解**:
```
P(x_1, ..., x_n) = ∏(i=1 to n) P(x_i | x_<i)
```

**Attention Mask**:
```
[1 0 0 0]
[1 1 0 0]
[1 1 1 0]  <- 下三角矩阵
[1 1 1 1]
```

**优势**:
- 天然支持文本生成 (auto-regressive)
- 无需特殊token ([MASK])
- 训练=推理模式一致

**劣势**:
- 单向上下文 (无法利用右侧信息)
- 每个token只能看到左侧历史

**代表模型**: GPT系列、LLaMA、Mistral

### 2.2 Masked Language Modeling (MLM)

**定义**: 随机遮盖部分token,预测被遮盖内容。

**数学形式**:
```
L_MLM = -∑(i∈M) log P(x_i | x_\M; θ)
```

其中M是被遮盖位置的集合。

**BERT的遮盖策略**:
- 80%: 替换为[MASK]
- 10%: 替换为随机token
- 10%: 保持原样

**为什么这样设计?**

1. **80% [MASK]**: 主要训练目标
2. **10% 随机**: 避免模型过度依赖[MASK]这个特殊token
3. **10% 原样**: 增强对真实token的表示能力

**数学期望**:
```
E[Loss] = 0.8 × L([MASK]) + 0.1 × L(random) + 0.1 × L(identity)
```

**Attention Mask**:
```
[1 1 1 1]
[1 1 1 1]  <- 全连接
[1 1 1 1]
[1 1 1 1]
```

**优势**:
- 双向上下文 (利用左右信息)
- 适合理解任务 (分类、NER、QA)

**劣势**:
- 训练-推理不一致 (预训练有[MASK],微调无)
- 不适合生成任务
- 需要额外的解码器才能生成

**代表模型**: BERT、RoBERTa、ALBERT

### 2.3 Prefix Language Modeling (PrefixLM)

**定义**: 前缀部分双向编码,后缀部分自回归生成。

**数学形式**:
```
L_PrefixLM = -∑(i=k+1 to n) log P(x_i | x_≤k, x_(k+1)..x_(i-1); θ)
```

前k个token使用双向注意力,后n-k个token使用因果注意力。

**Attention Mask**:
```
假设前2个token是prefix:
[1 1 0 0]
[1 1 0 0]  <- prefix部分全连接
[1 1 1 0]  <- suffix部分因果
[1 1 1 1]
```

**理论优势**:
- 结合MLM的理解能力和CLM的生成能力
- 适合Encoder-Decoder架构

**实践问题**:
- 如何确定prefix长度k?
- 训练效率不如纯CLM或MLM

**代表模型**: T5、GLM

### 2.4 三者对比

| 维度 | CLM | MLM | PrefixLM |
|------|-----|-----|----------|
| 上下文 | 单向(左) | 双向 | 混合 |
| 生成能力 | 强 | 弱 | 中 |
| 理解能力 | 中 | 强 | 强 |
| 训练-推理一致性 | 一致 | 不一致 | 部分一致 |
| 适用任务 | 生成、对话 | 分类、抽取 | 翻译、摘要 |

**理论差异根源**:

CLM遵循**因果推断**原则:
```
P(未来 | 过去) ≠ P(过去 | 未来)
```

MLM假设**条件独立**:
```
P(x_i | x_\M) ≈ P(x_i | context)  (双向)
```

这导致MLM在生成任务上的天然劣势——无法建模联合分布P(x_1, ..., x_n)。

---

## 3. 数据工程

### 3.1 数据去重

**为什么去重?**

1. **避免记忆**: 重复数据导致模型记忆而非泛化
2. **提高多样性**: 相同数据不提供新信息
3. **节省计算**: 重复数据浪费FLOPs

**去重粒度**:

**a) 精确去重 (Exact Deduplication)**

哈希去重:
```python
# 伪代码
seen_hashes = set()
for doc in corpus:
    h = sha256(doc)
    if h not in seen_hashes:
        seen_hashes.add(h)
        yield doc
```

**b) 近似去重 (Fuzzy Deduplication)**

MinHash + LSH:
```
1. 将文档表示为n-gram集合
2. 计算MinHash签名 (k个哈希函数)
3. 使用LSH分桶,找到候选相似对
4. 计算Jaccard相似度,阈值过滤
```

**Jaccard相似度**:
```
J(A, B) = |A ∩ B| / |A ∪ B|
```

**MinHash期望性质**:
```
P(h_i(A) = h_i(B)) = J(A, B)
```

即:两个集合的MinHash值相等的概率等于它们的Jaccard相似度。

**LLaMA的去重策略**:
- CCNet去重工具
- 5-gram重叠度 > 80% 视为重复
- 移除了约13%的CommonCrawl数据

### 3.2 数据过滤

**质量评估指标**:

**a) 启发式规则**

- 长度过滤: 10 < tokens < 100,000
- 符号密度: `#special_chars / #total_chars < 0.1`
- 重复n-gram: 最长重复子串 < 20% 文档长度
- 语言检测: fastText语言分类器 (置信度 > 0.8)

**b) 困惑度过滤 (Perplexity Filtering)**

使用高质量语料训练的小型语言模型(如KenLM)计算困惑度:
```
PPL(x) = exp(-1/n ∑ log P(x_i | x_<i))
```

保留 PPL < 阈值 的文档(如PPL < 1000)。

**理论依据**: 低困惑度 => 高似然 => 符合自然语言分布

**c) 分类器过滤**

训练二分类器区分"高质量"vs"低质量":
- 正样本: Wikipedia、书籍、学术论文
- 负样本: 垃圾网页、广告、自动生成文本

fastText分类器:
```
P(quality=high | doc) > 0.5  => 保留
```

**GPT-3的过滤**:
- WebText分类器 (基于Reddit upvote)
- 过滤掉约90%的CommonCrawl

### 3.3 数据配比 (Data Mixing)

**问题**: 不同来源数据的最优混合比例?

**来源类别**:
- 网页 (CommonCrawl): 85%
- 书籍 (Books3): 7%
- 维基百科 (Wikipedia): 4%
- 代码 (GitHub): 3%
- 学术 (ArXiv): 1%

**配比策略**:

**a) 均匀采样**

每个数据源按相同概率采样:
```
P(source_i) = 1 / k
```

**劣势**: 忽略数据源质量差异

**b) 按大小比例采样**

```
P(source_i) = |D_i| / ∑|D_j|
```

**劣势**: 低质量大数据源占主导

**c) 温度采样 (Temperature Sampling)**

```
P(source_i) ∝ (|D_i|)^(1/T)
```

- T=1: 按大小比例
- T→0: 均匀采样
- T→∞: 只采样最大源

**LLaMA的配比**:
```python
# 伪代码
weights = {
    'CommonCrawl': 0.67,
    'C4': 0.15,
    'Github': 0.045,
    'Wikipedia': 0.045,
    'Books': 0.045,
    'ArXiv': 0.025,
    'StackExchange': 0.02
}
```

**d) 动态配比**

Gopher使用**阶段性配比**:
- 前期: 高质量数据 (Books, Wikipedia)
- 中期: 混合数据
- 后期: 代码和数学数据 (增强推理能力)

**理论解释**:

课程学习 (Curriculum Learning):
```
L_total = ∑(t=1 to T) w_t × L_t(θ)
```

权重w_t随时间变化,先易后难。

### 3.4 质量评估

**下游任务评估**:

在小规模模型上测试不同配比,评估:
- MMLU (知识)
- HumanEval (代码)
- GSM8K (数学)

选择Pareto最优配比。

**理论框架**:

数据价值函数:
```
V(D) = E_{x~D}[Improvement(x; θ)]
```

实践中使用**影响函数** (Influence Function):
```
I(z) = -∇_θ L(z) × H^(-1) × ∇_θ L(test)
```

其中H是Hessian矩阵。

---

## 4. 训练稳定性

### 4.1 梯度裁剪 (Gradient Clipping)

**动机**: 防止梯度爆炸导致的训练崩溃。

**全局范数裁剪**:
```
g' = g × min(1, θ / ||g||_2)
```

其中θ是裁剪阈值(如1.0)。

**数学性质**:
- 保持梯度方向
- 限制梯度大小

**实践配置**:
```
GPT-3: clip_grad_norm = 1.0
LLaMA: clip_grad_norm = 1.0
PaLM: clip_grad_norm = 1.0
```

**为什么有效?**

梯度裁剪等价于在损失函数中添加隐式正则:
```
L_reg = L + λ × ||∇L||^2  (当||∇L|| > θ时)
```

### 4.2 学习率调度

**Warmup + Cosine Decay**:

```python
def get_lr(step, total_steps, max_lr, warmup_steps):
    if step < warmup_steps:
        # 线性warmup
        return max_lr * step / warmup_steps
    else:
        # 余弦衰减
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return max_lr * 0.5 * (1 + cos(π * progress))
```

**为什么需要Warmup?**

初始阶段梯度不稳定,大学习率导致:
```
θ_1 = θ_0 - η × g_0  (g_0可能很大)
```

Warmup提供"软着陆":
```
η(t) = η_max × min(t/T_warmup, 1)
```

**Chinchilla的调度**:
- Warmup: 前1%步数
- Peak LR: 2e-4
- Min LR: 2e-5 (10%的peak)

**数学直觉**:

学习率调度近似牛顿法的自适应步长:
```
θ_(t+1) = θ_t - η_t × H_t^(-1) × g_t
```

余弦衰减模拟Hessian特征值的变化。

### 4.3 混合精度训练

**FP16 + FP32 混合**:

**步骤**:
1. **前向传播**: FP16
2. **梯度计算**: FP16
3. **梯度缩放**: `scaled_grad = grad × scale` (防止下溢)
4. **参数更新**: FP32

**损失缩放 (Loss Scaling)**:

```python
loss_scale = 2^k  # 动态调整k
scaled_loss = loss × loss_scale
scaled_grad = backward(scaled_loss)
grad = scaled_grad / loss_scale
```

**为什么有效?**

FP16表示范围:
```
最小正数: 2^(-24) ≈ 6e-8
最大数: 65504
```

梯度常在1e-7量级,未缩放会下溢为0。

**内存节省**:
```
FP16模型: 175B × 2 bytes = 350GB
FP32模型: 175B × 4 bytes = 700GB
```

节省50%显存!

**精度损失分析**:

经验观察: FP16训练的最终损失与FP32相差 < 0.01

理论解释: 随机舍入误差在优化过程中被平滑:
```
E[round(x)] ≈ x  (当样本量足够大)
```

---

## 5. 分布式训练

### 5.1 数据并行 (Data Parallelism, DP)

**原理**: 每个GPU持有完整模型副本,处理不同batch。

**流程**:
1. 分发参数: 广播θ到所有GPU
2. 前向+反向: 每个GPU独立计算梯度g_i
3. 梯度聚合: `g = (1/N) ∑g_i`  (AllReduce)
4. 参数更新: `θ = θ - η × g`

**通信复杂度**:

AllReduce通信量:
```
2 × (N-1)/N × |θ|
```

对于N=8, |θ|=175B×4bytes:
```
通信量 ≈ 1.2 TB  (每步!)
```

**带宽瓶颈**: NVLink带宽600GB/s,通信耗时约2秒。

**优化**: 梯度累积 (Gradient Accumulation)

```python
for micro_batch in range(K):
    loss = forward(micro_batch)
    loss.backward()  # 累积梯度
optimizer.step()  # K个micro-batch后才更新
```

等效batch size = K × micro_batch_size,但显存占用不变。

### 5.2 模型并行 (Model Parallelism, MP)

**张量并行 (Tensor Parallelism)**:

将单个Transformer层的权重矩阵切分到多个GPU。

**列并行**:
```
Y = XW,  W ∈ R^(d×4d)

切分W为 [W_1, W_2]:
Y_1 = XW_1  (GPU 1)
Y_2 = XW_2  (GPU 2)

拼接: Y = [Y_1, Y_2]
```

**行并行**:
```
Y = XW,  W ∈ R^(4d×d)

切分W为 [W_1; W_2]:
Y = X_1W_1 + X_2W_2  (需要AllReduce求和)
```

**Megatron-LM的策略**:

```
Attention:
  Q,K,V: 列并行
  Output: 行并行

FFN:
  Up projection: 列并行
  Down projection: 行并行
```

**通信量分析**:

每层2次AllReduce:
```
通信量 = 2 × batch_size × seq_len × hidden_dim × 4 bytes
```

对于batch=1024, seq=2048, d=12288:
```
通信量 = 2 × 1024 × 2048 × 12288 × 4 ≈ 200GB  (每层)
```

### 5.3 流水线并行 (Pipeline Parallelism, PP)

**原理**: 将模型按层切分到多个GPU,形成流水线。

**GPipe调度**:

```
GPU1: [F1] [F2] [F3] [F4] [B1] [B2] [B3] [B4]
GPU2:      [F1] [F2] [F3] [F4] [B1] [B2] [B3] [B4]
GPU3:           [F1] [F2] [F3] [F4] [B1] [B2] [B3] [B4]
GPU4:                [F1] [F2] [F3] [F4] [B1] [B2] [B3] [B4]
```

F=前向, B=反向

**Bubble问题**: 存在大量空闲时间(bubble)。

**Bubble比例**:
```
Bubble = (P-1) / (M+P-1)
```

P=流水线阶段数, M=micro-batch数量

**PipeDream-Flush优化**:

通过调整micro-batch执行顺序,将bubble降低到:
```
Bubble ≈ P / M
```

### 5.4 ZeRO优化器状态分片

**问题**: Adam优化器需要存储:
- 参数: θ (FP32)
- 梯度: g (FP32)
- 一阶动量: m (FP32)
- 二阶动量: v (FP32)

总显存: **4 × |θ|**

对于175B模型:
```
4 × 175B × 4 bytes = 2.8 TB  (单GPU无法容纳!)
```

**ZeRO-1: 优化器状态分片**

将m, v切分到N个GPU:
```
GPU_i存储: θ, g, m_i, v_i
```

内存节省:
```
Before: 4|θ|
After:  2|θ| + 2|θ|/N = 2|θ|(1 + 1/N)
```

对于N=8:
```
节省 = 4|θ| - 2.25|θ| = 1.75|θ|  (约44%)
```

**ZeRO-2: 梯度分片**

进一步切分梯度g:
```
GPU_i存储: θ, g_i, m_i, v_i
```

内存节省:
```
After: |θ| + 3|θ|/N
```

对于N=8:
```
节省 = 4|θ| - 1.375|θ| = 2.625|θ|  (约66%)
```

**ZeRO-3: 参数分片**

连参数也分片:
```
GPU_i存储: θ_i, g_i, m_i, v_i
```

内存节省:
```
After: 4|θ|/N
```

对于N=8:
```
节省 = 4|θ| - 0.5|θ| = 3.5|θ|  (约87.5%)
```

**通信开销**:

ZeRO-3需要在前向传播时广播参数:
```
通信量 = |θ|  (每层)
```

**权衡**: 内存效率 vs 通信开销

**实践配置**:

- GPT-3 (175B): ZeRO-2 + TP(8) + PP(16)
- LLaMA (65B): FSDP (类似ZeRO-3)
- PaLM (540B): ZeRO-3 + TP(16) + PP(12)

**ZeRO的数学本质**:

传统DP:
```
Memory(GPU_i) = O(|θ|)  (每个GPU都存完整状态)
```

ZeRO-k:
```
Memory(GPU_i) = O(|θ|/N + k×communication_buffer)
```

通过通信换内存,实现线性扩展。

---

## 6. 总结

预训练技术的核心矛盾:
- **Scaling Law**: 更大=更强
- **计算约束**: 资源有限

解决方案:
1. **数据工程**: 高质量数据 > 大量低质量数据
2. **训练稳定性**: 混合精度、梯度裁剪保证收敛
3. **分布式训练**: ZeRO等技术突破单机内存墙

未来方向:
- **MoE (Mixture-of-Experts)**: 稀疏激活,降低推理成本
- **长上下文**: 位置编码外推、注意力优化
- **多模态预训练**: 视觉-语言联合训练

预训练的本质是**压缩人类知识的分布**:
```
min KL(P_data || P_θ)
```

所有技术创新都在服务这一目标。
