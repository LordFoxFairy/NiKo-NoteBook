# GLOSSARY 术语表

> **大语言模型技术索引 (2025年版)**
>
> 本术语表包含大语言模型领域的核心概念、前沿技术与工程实践术语。每个术语提供精炼定义及章节交叉引用。

---

## A

### AdaLoRA (Adaptive LoRA)
自适应秩分配的LoRA变体，根据重要性动态调整不同层的秩参数，提升参数效率。
→ 详见 [Part 3 Ch 2: 微调你的专属模型]

### Agent (智能体)
能够感知环境、自主决策并执行行动以完成目标的LLM系统，通常结合ReAct模式与工具调用能力。
→ 详见 [Part 4 Ch 3: 智能体核心机制]

### Alignment (对齐)
使模型输出符合人类价值观和意图的过程，核心技术包括RLHF、DPO等。
→ 详见 [Part 3 Ch 3: 与人类对齐：偏好优化]

### Attention (注意力机制)
Transformer的核心组件，通过Query-Key-Value机制动态加权聚合信息，实现上下文理解。
→ 详见 [Part 2 Ch 1: Transformer核心揭秘]

---

## B

### BERT (Bidirectional Encoder Representations from Transformers)
基于Transformer编码器的双向预训练模型，擅长理解任务如文本分类、命名实体识别。
→ 详见 [Part 1 Ch 1: 初识大语言模型] / [Part 2 Ch 2: 模型家族谱系]

### BPE (Byte Pair Encoding)
子词分词算法，通过迭代合并高频字符对构建词表，平衡词表大小与分词粒度。
→ 详见 [Part 1 Ch 3: 语言的基石：分词与嵌入]

---

## C

### Chain-of-Thought (CoT / 思维链)
通过在Prompt中要求模型"逐步思考"输出推理过程，显著提升复杂推理任务准确率的技术。
→ 详见 [Part 1 Ch 2: 与模型对话：提示工程基础] / [Part 7 Ch 3: 推理时计算增强]

### Chunking (文档分块)
RAG系统中将长文档切分为语义连贯的片段的技术，影响检索精度与生成质量。
→ 详见 [Part 4 Ch 2: 检索增强生成（RAG）原理]

### Cold Start (冷启动)
数据工程中缺乏初始训练数据的场景，常通过Synthetic Data或Self-Instruct缓解。
→ 详见 [Part 3 Ch 1: 数据工程基础]

### Context Window (上下文窗口)
模型一次能处理的最大Token数量，2025年前沿模型已达128K~200K tokens。
→ 详见 [Part 7 Ch 1: 长上下文技术]

### Continuous Batching (连续批处理)
vLLM核心技术，动态管理不同长度的推理请求，避免传统静态批处理的等待浪费。
→ 详见 [Part 6 Ch 2: vLLM高性能推理]

---

## D

### DeepSeek-R1
2025年前沿推理模型，通过强化学习训练推理时计算能力，在数学/代码任务中表现出色。
→ 详见 [Part 7 Ch 4: 推理模型专题]

### DeepSpeed
微软开源的分布式训练框架，支持ZeRO优化、流水线并行、混合精度训练等大模型训练技术。
→ 详见 [Part 5 Ch 4: DeepSpeed分布式训练]

### DoRA (Weight-Decomposed Low-Rank Adaptation)
将权重分解为幅度(Magnitude)和方向(Direction)的LoRA变体，提升微调性能与稳定性。
→ 详见 [Part 3 Ch 2: 微调你的专属模型]

### DPO (Direct Preference Optimization)
无需RL训练器的偏好优化算法，直接从偏好数据中优化模型，相比RLHF更简单高效。
→ 详见 [Part 3 Ch 3: 与人类对齐：偏好优化]

---

## E

### Embedding (嵌入)
将离散的Token/文本映射到连续向量空间的表示，是语义理解与RAG的基础。
→ 详见 [Part 1 Ch 3: 语言的基石：分词与嵌入] / [Part 3 Ch 4: 创建更优的嵌入模型]

### Encoder-Decoder
Transformer的完整架构，编码器双向理解输入，解码器自回归生成输出，适用于翻译任务。
→ 详见 [Part 2 Ch 2: 模型家族谱系：从编码器到解码器]

---

## F

### Few-shot Learning (少样本学习)
通过在Prompt中提供少量示例让模型学会新任务，无需梯度更新，是ICL的核心应用。
→ 详见 [Part 1 Ch 2: 与模型对话：提示工程基础]

### FlashAttention
高效Attention实现，通过IO-aware算法和Tiling优化显存访问，加速训练与推理2-4倍。
→ 详见 [Part 2 Ch 1: Transformer核心揭秘] / [Part 6 Ch 1: 模型压缩与推理加速]

### Function Calling (函数调用)
模型根据用户意图自动调用外部工具/API的能力，是构建Agent系统的核心机制。
→ 详见 [Part 4 Ch 3: 智能体核心机制]

---

## G

### GPT (Generative Pre-trained Transformer)
基于Transformer解码器的自回归生成模型，通过预测下一个Token训练，是ChatGPT的基础架构。
→ 详见 [Part 1 Ch 1: 初识大语言模型] / [Part 2 Ch 2: 模型家族谱系]

### GraphRAG
微软提出的高级RAG架构，通过知识图谱建模文档关系，提升复杂推理与多跳问答能力。
→ 详见 [Part 4 Ch 2: 检索增强生成（RAG）原理]

### Grokking (顿悟)
训练过程中模型突然从记忆转向泛化的现象，通常在过拟合后继续训练才出现。
→ 详见 [Part 2 Ch 3: 预训练的奥秘：从数据到智能]

### GRPO (Group Relative Policy Optimization)
分组相对策略优化，DeepSeek-R1等推理模型使用的强化学习算法，改进传统PPO。
→ 详见 [Part 7 Ch 4: 推理模型专题]

---

## H

### Hallucination (幻觉)
模型生成看似合理但实际错误或无根据的内容，RAG与外部验证是主要缓解手段。
→ 详见 [Part 4 Ch 2: 检索增强生成（RAG）原理] / [Part 7 Ch 5: 模型安全与可解释性]

---

## I

### In-Context Learning (ICL / 上下文学习)
模型通过Prompt中的示例学会新任务而无需梯度更新，是大模型的涌现能力。
→ 详见 [Part 1 Ch 2: 与模型对话：提示工程基础]

### Instruction Tuning (指令微调)
在多样化指令数据上微调模型，使其能准确理解并遵循人类指令，是SFT的核心。
→ 详见 [Part 3 Ch 2: 微调你的专属模型]

---

## K

### KV Cache (键值缓存)
自回归生成中缓存历史Token的Key和Value张量，避免重复计算，PagedAttention优化其管理。
→ 详见 [Part 6 Ch 2: vLLM高性能推理]

---

## L

### LangChain
开源LLM应用开发框架，提供链式调用、Agent、RAG等组件，简化应用构建。
→ 详见 [Part 5 Ch 5: 端到端LLM项目实战]

### LangGraph
LangChain团队推出的多Agent编排框架，基于有向图建模Agent工作流。
→ 详见 [Part 4 Ch 3: 智能体核心机制]

### LawGLM
面向法律领域的垂直大模型，通过领域预训练与微调实现专业法律问答与文书生成。
→ 详见 [Part 3 Ch 2: 微调你的专属模型]

### LLaMA-Factory
一站式大模型微调工具，集成LoRA/QLoRA/全量微调，支持WebUI配置，降低微调门槛。
→ 详见 [Part 5 Ch 2: LLaMA-Factory微调工厂]

### LoRA (Low-Rank Adaptation)
参数高效微调(PEFT)的代表方法，通过低秩分解冻结原模型权重，仅训练小规模适配器。
→ 详见 [Part 3 Ch 2: 微调你的专属模型]

---

## M

### Mamba
基于状态空间模型(SSM)的新型架构，线性时间复杂度替代Attention，适合超长序列建模。
→ 详见 [Part 7 Ch 2: 新型架构探索]

### Matryoshka Embedding (俄罗斯套娃嵌入)
支持灵活维度的嵌入模型，可在推理时截断向量维度以平衡精度与效率。
→ 详见 [Part 3 Ch 4: 创建更优的嵌入模型]

### MCP (Model Context Protocol)
Anthropic提出的标准化协议，定义LLM与外部工具/数据源交互接口，增强互操作性。
→ 详见 [Part 4 Ch 3: 智能体核心机制]

### MinHash LSH (局部敏感哈希)
高效近似最近邻搜索算法，在大规模文档去重与相似度检索中广泛应用。
→ 详见 [Part 3 Ch 1: 数据工程基础] / [Part 7 Ch 6: 大规模预训练数据工程]

### MoE (Mixture of Experts / 专家混合)
模型架构变体，每次只激活部分专家子网络，在保持性能的同时大幅减少计算量。
→ 详见 [Part 2 Ch 2: 模型家族谱系] / [Part 6 Ch 1: 模型压缩与推理加速]

---

## P

### PagedAttention
vLLM核心技术，借鉴虚拟内存思想，将KV Cache分块管理，解决内存碎片与利用率问题。
→ 详见 [Part 6 Ch 2: vLLM高性能推理]

### PEFT (Parameter-Efficient Fine-Tuning / 参数高效微调)
只更新少量参数实现模型适配的方法集合，包括LoRA、Adapter、Prefix-Tuning等。
→ 详见 [Part 3 Ch 2: 微调你的专属模型]

### PPO (Proximal Policy Optimization)
RLHF训练中使用的强化学习算法，通过限制策略更新步长保证训练稳定性。
→ 详见 [Part 3 Ch 3: 与人类对齐：偏好优化] / [Part 5 Ch 3: TRL与强化学习实战]

### Prompt Engineering (提示工程)
设计优化Prompt以引导模型输出的技术，包括Few-shot、CoT、ReAct等模式。
→ 详见 [Part 1 Ch 2: 与模型对话：提示工程基础]

---

## Q

### QLoRA (Quantized LoRA)
结合4-bit量化与LoRA的微调方法，在单张消费级GPU上微调65B模型。
→ 详见 [Part 3 Ch 2: 微调你的专属模型] / [Part 5 Ch 2: LLaMA-Factory微调工厂]

### Quantization (量化)
降低模型权重/激活精度(如FP16→INT8)以减少显存占用与计算量，关键技术是量化感知训练。
→ 详见 [Part 6 Ch 1: 模型压缩与推理加速]

---

## R

### RAG (Retrieval-Augmented Generation / 检索增强生成)
结合外部知识库检索与LLM生成的架构，缓解幻觉、知识过时问题。
→ 详见 [Part 4 Ch 2: 检索增强生成（RAG）原理]

### ReAct (Reasoning and Acting)
结合推理(Thought)与行动(Action)的Prompt模式，是Agent系统的核心范式。
→ 详见 [Part 1 Ch 2: 与模型对话：提示工程基础] / [Part 4 Ch 3: 智能体核心机制]

### Reranking (重排序)
RAG中对初步检索结果进行精排的步骤，使用Cross-Encoder等模型提升Top-K精度。
→ 详见 [Part 4 Ch 2: 检索增强生成（RAG）原理]

### RLHF (Reinforcement Learning from Human Feedback / 基于人类反馈的强化学习)
通过奖励模型(Reward Model)与PPO训练使模型对齐人类偏好，是ChatGPT的关键技术。
→ 详见 [Part 3 Ch 3: 与人类对齐：偏好优化]

### RoPE (Rotary Position Embedding / 旋转位置编码)
相对位置编码方法，通过复数旋转矩阵注入位置信息，支持长度外推，是LLaMA架构标配。
→ 详见 [Part 2 Ch 1: Transformer核心揭秘] / [Part 7 Ch 1: 长上下文技术]

---

## S

### Scaling Laws (缩放定律)
描述模型性能与参数量、数据量、计算量之间幂律关系的经验规律，指导大模型训练资源配置。
→ 详见 [Part 2 Ch 3: 预训练的奥秘：从数据到智能]

### Self-Attention (自注意力)
Transformer核心机制，计算序列内每个Token与其他Token的关联权重，实现全局依赖建模。
→ 详见 [Part 1 Ch 1: 初识大语言模型] / [Part 2 Ch 1: Transformer核心揭秘]

### SetFit (Sentence Transformer Fine-Tuning)
少样本文本分类框架，先用对比学习微调Sentence-Transformer，再训练轻量分类头。
→ 详见 [Part 4 Ch 1: 语义理解应用：文本分类与聚类]

### SFT (Supervised Fine-Tuning / 监督式微调)
在标注数据上通过最大似然训练微调模型，是RLHF流程的第一阶段。
→ 详见 [Part 3 Ch 2: 微调你的专属模型]

### SimPO (Simple Preference Optimization)
简化版偏好优化算法，直接优化模型输出概率而不引入参考模型，相比DPO更高效。
→ 详见 [Part 3 Ch 3: 与人类对齐：偏好优化]

### Speculative Decoding (推测解码)
用小模型快速生成候选Token序列，大模型并行验证，加速自回归生成2-3倍。
→ 详见 [Part 6 Ch 1: 模型压缩与推理加速] / [Part 6 Ch 2: vLLM高性能推理]

### SwiGLU (Swish-Gated Linear Unit)
改进的FFN激活函数，结合Swish激活与门控机制，是LLaMA/PaLM等模型的标准选择。
→ 详见 [Part 2 Ch 1: Transformer核心揭秘]

### Synthetic Data (合成数据)
使用大模型生成的训练数据，通过Self-Instruct、Evol-Instruct等方法缓解数据稀缺。
→ 详见 [Part 3 Ch 1: 数据工程基础] / [Part 7 Ch 6: 大规模预训练数据工程]

---

## T

### Temperature (温度参数)
控制模型输出随机性的采样参数，T=0确定性输出，T>1增加创造性。
→ 详见 [Part 1 Ch 2: 与模型对话：提示工程基础]

### Tokenizer (分词器)
将文本切分为Token序列的工具，常用算法包括BPE、WordPiece、SentencePiece。
→ 详见 [Part 1 Ch 3: 语言的基石：分词与嵌入]

### Top-p Sampling (核采样)
动态截断低概率Token的采样策略，只从累积概率达到p的最小集合中采样。
→ 详见 [Part 1 Ch 2: 与模型对话：提示工程基础]

### Transformer
基于Self-Attention的深度学习架构，彻底改变NLP领域，是现代大语言模型的基础。
→ 详见 [Part 1 Ch 1: 初识大语言模型] / [Part 2 Ch 1: Transformer核心揭秘]

### TRL (Transformer Reinforcement Learning)
Hugging Face推出的强化学习训练库，简化RLHF/DPO实现，与PEFT、Accelerate深度集成。
→ 详见 [Part 5 Ch 3: TRL与强化学习实战]

---

## V

### vLLM
高性能LLM推理引擎，通过PagedAttention与Continuous Batching实现24倍吞吐量提升。
→ 详见 [Part 6 Ch 2: vLLM高性能推理]

### VeRA (Vector-based Random Matrix Adaptation)
使用共享随机矩阵+可训练缩放向量的PEFT方法，相比LoRA参数量更少。
→ 详见 [Part 3 Ch 2: 微调你的专属模型]

---

## Z

### ZeRO (Zero Redundancy Optimizer)
DeepSpeed核心优化技术，通过分片优化器状态、梯度、参数实现显存高效分布式训练。
→ 详见 [Part 5 Ch 4: DeepSpeed分布式训练]

### Zero-shot Learning (零样本学习)
不提供任何示例直接让模型完成任务，依赖预训练期间学到的通用能力。
→ 详见 [Part 1 Ch 2: 与模型对话：提示工程基础]

---

## 交叉索引

### 按技术领域分类

**架构与原理**:
Transformer | Self-Attention | Encoder-Decoder | MoE | Mamba

**训练与微调**:
SFT | LoRA | QLoRA | DoRA | PEFT | AdaLoRA | Instruction Tuning

**对齐与优化**:
RLHF | DPO | SimPO | PPO | Alignment

**推理与部署**:
vLLM | PagedAttention | KV Cache | Speculative Decoding | Quantization | FlashAttention

**应用开发**:
RAG | Agent | ReAct | Function Calling | Prompt Engineering | LangChain | LangGraph

**数据工程**:
Synthetic Data | MinHash LSH | Cold Start | Chunking

**位置编码与长上下文**:
RoPE | Context Window

**分布式训练**:
DeepSpeed | ZeRO

---

## 参考文献

本术语表基于2025年前沿研究与工程实践整理，具体技术细节与实现请参阅对应章节。

**版本**: v1.0 (2025-01)
**维护**: 随书籍章节更新同步更新

---

**使用建议**:
- 初学者: 按字母顺序浏览，结合章节交叉引用建立知识体系
- 实践者: 作为快速查询手册，定位具体技术的章节位置
- 研究者: 追踪术语演进脉络，理解技术发展趋势
