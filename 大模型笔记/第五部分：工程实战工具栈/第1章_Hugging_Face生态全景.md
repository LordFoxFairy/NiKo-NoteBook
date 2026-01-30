# 第1章：Hugging Face 生态全景 (The Complete Guide)

> **本章定位**：这是构建 LLM 应用的基石。我们将深入 Hugging Face 生态的五大核心组件：`Transformers`, `Datasets`, `Tokenizers`, `Accelerate`, `Hub`。不仅覆盖基础 API，更包含**量化加载**、**词表扩充**、**断点续训**、**分布式配置**等工业级实战技巧。

---

## 目录

- [1. Transformers：模型加载与推理](#1-transformers模型加载与推理)
  - [1.1 Pipeline：极速验证](#11-pipeline极速验证)
  - [1.2 AutoClass：底层控制与 Flash Attention](#12-autoclass底层控制与-flash-attention)
  - [1.3 Quantization：4-bit/8-bit 量化加载](#13-quantization4-bit8-bit-量化加载)
- [2. Datasets：海量数据工程](#2-datasets海量数据工程)
  - [2.1 流式加载 (Streaming) 与 混合 (Interleave)](#21-流式加载-streaming-与-混合-interleave)
  - [2.2 并行处理 (Map) 与 数据分片 (Sharding)](#22-并行处理-map-与-数据分片-sharding)
  - [2.3 自定义数据集加载脚本](#23-自定义数据集加载脚本)
- [3. Tokenizers：分词器的艺术与陷阱](#3-tokenizers分词器的艺术与陷阱)
  - [3.1 Chat Template 原理：如何避免"答非所问"](#31-chat-template-原理如何避免答非所问)
  - [3.2 Padding Side：左补齐 vs 右补齐](#32-padding-side左补齐-vs-右补齐)
  - [3.3 实战：扩充中文词表 (Add Tokens)](#33-实战扩充中文词表-add-tokens)
- [4. Training：训练与分布式](#4-training训练与分布式)
  - [4.1 Trainer API：Callbacks 与 断点续训](#41-trainer-api-callbacks-与-断点续训)
  - [4.2 Accelerate + DeepSpeed：分布式配置详解](#42-accelerate--deepspeed分布式配置详解)
- [5. Hub：模型管理与版本控制](#5-hub模型管理与版本控制)
  - [5.1 模型上传与 Revision 锁定](#51-模型上传与-revision-锁定)
  - [5.2 Model Card 编写规范](#52-model-card-编写规范)
- [本章小结：开发流 CheckList](#本章小结开发流-checklist)

---

## 1. Transformers：模型加载与推理

### 1.1 Pipeline：极速验证

适合快速测试模型能力。

```python
import torch
from transformers import pipeline

# 自动推断设备，默认使用 bfloat16 (推荐 Ampere 架构 GPU 使用)
pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-3-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Batch Inference (提升吞吐量的关键)
prompts = ["Explain AI.", "Write a poem."]
outputs = pipe(
    prompts,
    batch_size=8,
    max_new_tokens=128,
    temperature=0.7,
    do_sample=True
)
```

### 1.2 AutoClass：底层控制与 Flash Attention

生产环境通常使用 `AutoModel` + `AutoTokenizer`。

**Flash Attention 2 加速**：
这是现代 LLM 推理/训练的必备加速技术。
*   **前提**：安装 `flash-attn` 库 (`pip install flash-attn --no-build-isolation`) + 兼容的 GPU (A100, A10, RTX 3090/4090)。

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Llama-3-8B-Instruct"

# 1. 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

# 2. 加载 Model (开启 FA2)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2"  # 关键加速参数
)
```

### 1.3 Quantization：4-bit/8-bit 量化加载

在显存有限的设备（如单卡 24G 跑 70B 模型）上，量化是刚需。HF 通过 `bitsandbytes` (bnb) 库实现了原生集成。

**依赖**：`pip install bitsandbytes`

```python
from transformers import BitsAndBytesConfig

# NF4 (Normal Float 4) 配置：精度损失极小的 4bit 量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算时还原为 bf16
    bnb_4bit_use_double_quant=True          # 二次量化，进一步节省显存
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,         # 传入量化配置
    device_map="auto"
)

# 显存对比 (Llama-3-8B):
# fp16: ~16GB
# 4bit: ~6GB
```

---

## 2. Datasets：海量数据工程

### 2.1 流式加载 (Streaming) 与 混合 (Interleave)

处理 TB 级数据集（如 C4, WanJuan）时，无法全部下载。

```python
from datasets import load_dataset, interleave_datasets

# 1. Streaming 模式
ds_en = load_dataset("c4", "en", split="train", streaming=True)
ds_zh = load_dataset("wanjuan", "zh", split="train", streaming=True)

# 2. 数据混合 (80% 英文, 20% 中文) -> 预训练常用 trick
ds_mixed = interleave_datasets([ds_en, ds_zh], probabilities=[0.8, 0.2])

# 3. 迭代查看
for i, example in enumerate(ds_mixed):
    print(example['text'][:50])
    if i == 5: break
```

### 2.2 并行处理 (Map) 与 数据分片 (Sharding)

**Map 并行化**：
```python
ds = load_dataset("imdb", split="train")

def process_fn(examples):
    # 支持 batch 处理
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_ds = ds.map(
    process_fn,
    batched=True,
    batch_size=1000,
    num_proc=8,        # 多进程加速
    remove_columns=ds.column_names # 移除原始文本列，节省 RAM
)
```

**Sharding (分片)**：
分布式训练时，需要把大数据集切分成小块分发给不同节点。
```python
# 将数据集切分为 100 份，取第 0 份
shard_0 = ds.shard(num_shards=100, index=0)
```

### 2.3 自定义数据集加载脚本

当数据格式复杂（如 JSONL 嵌套、特殊 CSV），或者是私有数据时，编写加载脚本比手动解析更高效。

创建 `my_dataset.py`:
```python
import datasets

class MyDataset(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                "text": datasets.Value("string"),
                "label": datasets.Value("int32"),
            })
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": "train.jsonl"}),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            for id_, line in enumerate(f):
                # 自定义解析逻辑
                yield id_, json.loads(line)
```

使用：
```python
ds = load_dataset("./my_dataset.py")
```

---

## 3. Tokenizers：分词器的艺术与陷阱

### 3.1 Chat Template 的原理与陷阱

微调后的 Chat 模型（Llama-3, Qwen-2）对 Prompt 格式极其敏感。少一个空格或换行都可能导致模型“变傻”。

**原理**：Tokenizer 配置中的 `chat_template` 字段定义了 Jinja2 模板。

```python
chat = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
]

# 自动渲染 (推荐)
text = tokenizer.apply_chat_template(chat, tokenize=False)
print(text)
# Llama-3 输出: <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>...
```

**陷阱**：
如果手动拼接字符串（如 `f"User: {msg}"`），不仅格式可能错，还会导致特殊 Token（如 `<|eot_id|>`）被当作普通文本编码，模型无法识别停止信号。

### 3.2 Padding Side：左补齐 vs 右补齐

| 场景 | Padding Side | 原因 |
|---|---|---|
| **训练 (Training)** | `right` | 配合 Attention Mask，通常在序列末尾补齐效率最高。 |
| **推理 (Generation)** | `left` | **必须向左补齐！** 因为生成是自回归的，如果右侧有 Pad，模型会根据 Pad 去预测下一个词，导致输出乱码。 |

```python
tokenizer.padding_side = "left"  # 推理时务必设置
```

### 3.3 实战：扩充中文词表 (Add Tokens)

Llama-3 原生词表对中文支持一般（一个汉字可能被切成 3 个 token）。微调时常需扩充词表。

```python
# 1. 添加新词
new_tokens = ["你好", "人工智能", "大模型"]
num_added = tokenizer.add_tokens(new_tokens)

# 2. 调整模型 Embedding 层大小 (这也是必须要做的！)
# 模型原本 vocab_size 是 128256，现在变大了，Embedding 矩阵也要变大
model.resize_token_embeddings(len(tokenizer))

print(f"Added {num_added} tokens. New vocab size: {len(tokenizer)}")

# 注意：新加入的 Token 初始 Embedding 是随机的，需要经过 Fine-tuning 才能有语义。
```

---

## 4. Training：训练与分布式

### 4.1 Trainer API：Callbacks 与 断点续训

`Trainer` 是 HF 生态的核心训练器。

**WandB 集成与 Callbacks**：
```python
from transformers import TrainerCallback

class LogCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            print(f"Step {state.global_step} finished.")

# 配置参数
args = TrainingArguments(
    output_dir="./checkpoints",
    report_to="wandb",  # 自动集成 Weights & Biases
    run_name="llama3-finetune-v1",
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True, # 训练结束加载验证集最好的模型
)
```

**断点续训 (Resume Training)**：
训练大模型动辄几天，中断是常态。
```python
trainer.train(resume_from_checkpoint=True)
# 或者指定具体路径
# trainer.train(resume_from_checkpoint="./checkpoints/checkpoint-5000")
```

### 4.2 Accelerate + DeepSpeed：分布式配置详解

不使用 Trainer 时，Accelerate 是手动写训练循环的最佳伴侣。它完美集成了 DeepSpeed。

**配置 DeepSpeed**：
运行 `accelerate config`，选择 DeepSpeed，然后选择 ZeRO Stage (0/1/2/3)。

*   **ZeRO-2**：切分优化器状态 + 梯度（及格线，适合单机多卡）。
*   **ZeRO-3**：切分模型参数（显存占用最小，适合超大模型）。
*   **Offload**：将参数卸载到 CPU 内存（速度慢，但能跑更大的模型）。

**代码集成**：
```python
from accelerate import Accelerator

# 初始化时会自动读取 accelerate config 的配置
accelerator = Accelerator()

# 准备
model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, lr_scheduler
)

# 训练步
accelerator.backward(loss) # 替代 loss.backward()
```

---

## 5. Hub：模型管理与版本控制

### 5.1 模型上传与 Revision 锁定

不要只 push 到 `main`。

```python
# 上传时打标签
model.push_to_hub("my-model", revision="v1.0")

# 加载时锁定版本 (生产环境铁律)
model = AutoModel.from_pretrained(
    "username/my-model",
    revision="d4e5f6...", # Commit Hash 或 Tag
    trust_remote_code=True
)
```

### 5.2 Model Card 编写规范

一个好的 `README.md` (Model Card) 应包含：
1.  **Model Details**: 基础架构、参数量、训练数据来源。
2.  **Usage**: 几行可运行的 Python 代码示例。
3.  **Evaluation**: 在 MTEB 或 OpenCompass 上的评测分数。
4.  **Bias & Limitations**: 模型的局限性和偏见声明。

---

## 本章小结：开发流 CheckList

在开始下一章（微调实战）之前，请自查是否掌握了以下 **Engineering** 细节：

1.  ✅ **推理加速**：是否开启了 `Flash Attention 2` 和 `bfloat16`？
2.  ✅ **显存优化**：是否会用 `bitsandbytes` 进行 4-bit 量化加载？
3.  ✅ **数据处理**：面对 TB 级数据，是否会用 `Streaming` 和 `Interleave`？
4.  ✅ **分词避坑**：推理时是否将 padding 设为了 `left`？是否使用了正确的 `Chat Template`？
5.  ✅ **训练稳健性**：是否配置了 `save_limit` 防止硬盘撑爆？是否知道如何 `resume_from_checkpoint`？

掌握了这些，你就不再是 API 调包侠，而是具备了 **LLM 工程化落地** 的能力。

下一章，我们将介绍 **LLaMA-Factory**——它将上述所有（Trainer, DeepSpeed, Quantization, FlashAttn）封装成了一个 WebUI 界面，让你体验“零代码微调”的快感。
