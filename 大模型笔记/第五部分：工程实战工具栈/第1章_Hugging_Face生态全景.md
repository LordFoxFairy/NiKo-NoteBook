# 第1章：Hugging Face生态全景

> **本章定位**：不再赘述原理，直接进入代码实战。掌握 `Transformers`, `Datasets`, `Accelerate` 的核心工程能力。

---

## 目录
- [1. High-Level API: Pipeline 与 AutoClass](#1-high-level-api-pipeline-与-autoclass)
  - [1.1 极速推理：Pipeline](#11-极速推理pipeline)
  - [1.2 灵活调用：AutoClass](#12-灵活调用autoclass)
- [2. Datasets: 数据处理工具链](#2-datasets-数据处理工具链)
  - [2.1 流式加载大规模数据](#21-流式加载大规模数据)
  - [2.2 高效预处理：Map 与 Filter](#22-高效预处理map-与-filter)
- [3. Accelerate: 分布式训练入门](#3-accelerate-分布式训练入门)
  - [3.1 混合精度训练](#31-混合精度训练)
  - [3.2 多卡训练](#32-多卡训练)
- [4. Model Hub: 模型版本管理](#4-model-hub-模型版本管理)
  - [4.1 上传模型](#41-上传模型)
  - [4.2 模型卡片编写](#42-模型卡片编写)
- [5. Tokenizers: 自定义分词器训练](#5-tokenizers-自定义分词器训练)
- [本章小结](#本章小结)

---

## 1. High-Level API: Pipeline 与 AutoClass

最快速的模型调用方式，适合验证模型能力。

### 1.1 极速推理：Pipeline

```python
import torch
from transformers import pipeline

# 自动处理了：Tokenizer -> Model -> Post-processing
# device_map="auto" 自动分配显存，支持多卡
pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-3-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# 批处理推理（Batch Inference）大大提升吞吐量
prompts = [
    "Explain quantum computing in 50 words.",
    "Write a haiku about coding.",
    "Translate 'Hello world' to Python."
]

# batch_size=4 显存允许时越大越好
outputs = pipe(prompts, batch_size=4, max_new_tokens=128, temperature=0.7)

for out in outputs:
    print(f"Generated: {out[0]['generated_text']}\n{'-'*20}")
```

### 1.2 底层控制：AutoModel & AutoTokenizer

当主要逻辑需要定制时（如自定义 Attention mask、特殊解码策略），使用 AutoClass。

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Llama-3-8B-Instruct"

# 1. 加载 Tokenizer
# padding_side="left" 对生成式模型很重要，否则生成会受 pad token 干扰
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token # Llama 系列通常需要手动指定 pad_token

# 2. 加载 Model
# attn_implementation="flash_attention_2" 开启 FlashAttention 加速（需安装 flash-attn 库）
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2"
)

# 3. 构造 Chat 模板
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
]
# apply_chat_template 自动处理 <|begin_of_text|>, <|start_header_id|> 等特殊 token
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

# 4. 生成
outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=tokenizer.eos_token_id,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)

response_ids = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response_ids, skip_special_tokens=True))
```

---

## 2. 数据处理：Datasets库的高效操作

处理 100GB+ 数据集的核心技巧：**Map**, **Filter**, **Streaming**。

### 2.1 高效加载与流式处理

```python
from datasets import load_dataset

# 1. 常规加载（下载并解压到本地缓存）
ds = load_dataset("imdb", split="train")

# 2. 流式加载（Streaming）：不下载由本地，随用随下，处理 TB 级数据必备
ds_stream = load_dataset("c4", "en", split="train", streaming=True)

# 处理流式数据
top_10_examples = ds_stream.take(10)
for ex in top_10_examples:
    print(ex['text'][:50])
```

### 2.2 Map 与 Filter 实战

```python
# 假设我们有一个包含 'text' 列的数据集
def process_data(batch):
    # 1. 过滤掉过短的文本
    if len(batch["text"]) < 50:
        return {"input_ids": []} # 标记为待删除

    # 2. Tokenize
    tokenized = tokenizer(
        batch["text"],
        truncation=True,
        max_length=512,
        padding=False # 训练时通常在 DataCollator 中动态 Padding，节省显存
    )
    return tokenized

# batched=True + num_proc 多进程并行处理，速度提升 10x
# remove_columns 删除原始文本列，减少内存占用
processed_ds = ds.map(
    process_data,
    batched=True,
    batch_size=1000,
    num_proc=8, # 根据 CPU 核数设置
    remove_columns=ds.column_names,
    desc="Tokenizing dataset"
)

# 真正的过滤（map 中只是处理，过滤需用 filter）
filtered_ds = processed_ds.filter(lambda x: len(x["input_ids"]) > 0)
```

---

## 3. 分布式利器：Accelerate

在不使用 Trainer 的情况下，如何用几行代码将 PyTorch 代码迁移到多卡/TPU？

### 3.1 改造 PyTorch 训练循环

```python
from accelerate import Accelerator
import torch

# 1. 初始化 Accelerator
# mixed_precision="fp16" 自动处理混合精度
accelerator = Accelerator(mixed_precision="fp16")

model = MyModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
train_loader = ...

# 2. 准备对象 (Prepare)
# Accelerator 会自动将模型转为 DDP 模式，将数据分片到各 GPU
model, optimizer, train_loader = accelerator.prepare(
    model, optimizer, train_loader
)

model.train()
for batch in train_loader:
    optimizer.zero_grad()
    outputs = model(**batch)
    loss = outputs.loss

    # 3. 替换 loss.backward()
    accelerator.backward(loss)

    optimizer.step()

# 4. 保存模型 (只在主进程保存)
# accelerator.wait_for_everyone() 同步所有进程
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
if accelerator.is_main_process:
    torch.save(unwrapped_model.state_dict(), "model.pt")
```

### 3.2 启动命令

不需要修改代码中的 `device`，直接通过 CLI 启动：

```bash
# 配置环境（只需运行一次）
accelerate config

# 启动训练（自动适配单机多卡、多机多卡）
accelerate launch train.py
```

---

## 4. Trainer API：工业级训练封装

Hugging Face `Trainer` 封装了 logging, gradient accumulation, checkpointing, distributed training 等复杂逻辑。

### 4.1 核心配置与自定义 Loss

```python
from transformers import Trainer, TrainingArguments

class CustomTrainer(Trainer):
    """
    继承 Trainer 以自定义 Loss 计算逻辑
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # 前向传播
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # 自定义加权 Loss：给予后半段文本更高权重（示例）
        # 标准是 CrossEntropyLoss
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = loss_fct(shift_logits.view(-1, self.model.config.vocab_size), shift_labels.view(-1))

        # 简单平均
        loss = loss.mean()

        return (loss, outputs) if return_outputs else loss

# 训练参数配置（最佳实践）
args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8, # 单卡 batch size
    gradient_accumulation_steps=4, # 梯度累积，等效 batch_size = 8 * 4 * num_gpus
    learning_rate=2e-5,
    weight_decay=0.01,
    bf16=True,                     # A100/H100 必开，3090/4090 可开
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    save_total_limit=3,            # 只保留最新的3个checkpoint
    report_to="wandb",             # 实验记录
    dataloader_num_workers=4,      # 数据加载进程数
    gradient_checkpointing=True,   # 显存优化：以计算换显存
)

trainer = CustomTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

### 4.2 Callback 机制

在训练过程中插入自定义逻辑（如：每生成 100 步打印一次生成结果）。

```python
from transformers import TrainerCallback

class GenerationCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # 这里的 step 检查逻辑
        if state.global_step % args.logging_steps == 0:
            print(f"Step {state.global_step}: Logging stats...")

trainer.add_callback(GenerationCallback())
```

---

## 5. 本章小结：开发流选择指引

| 场景 | 推荐工具 | 理由 |
| :--- | :--- | :--- |
| **快速验证 / Demo** | `Pipeline` | 代码最少，开箱即用。 |
| **标准微调 (SFT/LoRA)** | `Trainer` | 内置了所有最佳实践，配合 TRL 库更佳。 |
| **修改模型架构 / 复杂循环** | `Accelerate` | 保持 PyTorch 的灵活性，同时获得分布式能力。 |
| **极端定制 / 纯 PyTorch** | 原生 DDP | 仅在需要完全控制所有底层细节时使用。 |
