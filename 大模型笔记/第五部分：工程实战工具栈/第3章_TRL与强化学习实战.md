# 第3章：TRL 与强化学习实战 (SFT / DPO / PPO)

> **本章定位**：从微调（SFT）到对齐（Alignment）。我们将复现 Hugging Face 官方 **Alignment Handbook** 的核心流程，但为了让每位读者都能跑通，我们将基座模型替换为轻量级的 **Qwen2-0.5B**。无论你是在 Colab 还是单卡 3090，都能完整体验 RLHF 的全过程。

---

## 目录

- [1. 完整的对齐流水线 (The Alignment Pipeline)](#1-完整的对齐流水线-the-alignment-pipeline)
- [2. SFT：让 Qwen-0.5B 学会指令](#2-sft让-qwen-05b-学会指令)
  - [2.1 数据格式与 Chat Template](#21-数据格式与-chat-template)
  - [2.2 核心技巧：Packing 加速](#22-核心技巧packing-加速)
  - [2.3 实战代码](#23-实战代码)
- [3. DPO：工业界对齐首选](#3-dpo工业界对齐首选)
  - [3.1 数据集：偏好对是如何构建的？](#31-数据集偏好对是如何构建的)
  - [3.2 关键超参：Beta 的魔法](#32-关键超参beta-的魔法)
  - [3.3 实战：使用 Qwen-0.5B 跑通 DPO](#33-实战使用-qwen-05b-跑通-dpo)
- [4. PPO：经典 RLHF 三阶段 (进阶)](#4-ppo经典-rlhf-三阶段-进阶)
  - [4.1 训练 Reward Model (RM)](#41-训练-reward-model-rm)
  - [4.2 PPO 流程详解 (Actor-Critic)](#42-ppo-流程详解-actor-critic)
- [5.新兴趋势：ORPO 与 KTO](#5-2024-新趋势orpo-与-kto)
- [本章小结](#本章小结)

---

## 1. 完整的对齐流水线 (The Alignment Pipeline)

一个标准的工业级 LLM 训练流程包含三个阶段：

1.  **Pre-training (PT)**: 海量文本，学习"续写"。
2.  **Supervised Fine-Tuning (SFT)**: 指令数据，学习"对话"。
3.  **Preference Alignment (DPO/PPO)**: 偏好数据，学习"价值观"。

本章我们将使用 TRL 库，基于 **Qwen2-0.5B** 完成后两个阶段。

---

## 2. SFT：让 Qwen-0.5B 学会指令

SFT 不仅仅是微调，更是让模型适应特定的**对话格式**。

### 2.1 数据格式与 Chat Template

对于 Qwen2，我们必须严格遵守 ChatML 格式：
`<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n{msg}<|im_end|>`

TRL 的 `SFTTrainer` 可以自动处理这个，前提是你配置好了 `chat_template`。

### 2.2 核心技巧：Packing (序列打包)

`SFTTrainer` 支持 `packing=True`。它将多个短对话拼接到 `max_seq_length` (如 2048)，用 `attention_mask` 隔开。
*   **收益**：训练速度通常提升 **3-5 倍**。
*   **代价**：需要更多显存（但对于 0.5B 模型，这不是问题）。

### 2.3 实战代码

以下代码可在单卡 T4 (Colab 免费版) 上运行。

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# 1. 准备模型与数据
model_id = "Qwen/Qwen2-0.5B-Instruct"
# 使用 HuggingFaceH4 的精选数据集 (Ultrafachat)
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:1%]")

tokenizer = AutoTokenizer.from_pretrained(model_id)
# ⚠️ Qwen 的 pad_token 有时需要手动指定，避免 loss 为 NaN
tokenizer.pad_token = tokenizer.eos_token

# 2. 配置参数
args = SFTConfig(
    output_dir="./qwen-sft",
    max_seq_length=2048,
    packing=True,                # 核心加速
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,          # SFT 典型学习率
    lr_scheduler_type="cosine",
    logging_steps=10,
    fp16=True,                   # T4 用 fp16, A100 用 bf16
)

# 3. 开始训练
trainer = SFTTrainer(
    model=model_id,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="messages", # 数据集中的列名
    args=args,
)

trainer.train()
trainer.save_model("./qwen-sft-final")
```

---

## 3. DPO：工业界对齐首选

现在我们的 Qwen-0.5B 主要学会了说话，但可能还会胡说八道。我们要用 DPO (Direct Preference Optimization) 来对齐人类偏好。

### 3.1 数据集：偏好对是如何构建的？

DPO 数据必须是**成对**的：`(prompt, chosen, rejected)`。
*   **Chosen (胜)**：详细、有用、无害的回答。
*   **Rejected (负)**：简短、错误或有害的回答。

我们使用 `HuggingFaceH4/ultrafeedback_binarized`，这是目前质量最高的开源偏好数据集之一。

### 3.2 关键超参：Beta 的魔法

`beta` 是 DPO 损失函数中的 KL 惩罚系数。
*   **Zephyr 配方**：`beta=0.1`。
*   **直觉**：beta 越大，模型越保守（贴近原始模型）；beta 越小，模型越激进（贴近 chosen 数据）。对于 Qwen-0.5B 这种小模型，建议 `beta=0.1` 以防止过度遗忘。

### 3.3 实战：使用 Qwen-0.5B 跑通 DPO

```python
from trl import DPOTrainer, DPOConfig

# 1. 加载 SFT 后的模型 (作为 Policy Model)
model_id = "./qwen-sft-final"

# 2. 加载数据
dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs[:1000]")

# 3. DPO特殊的配置
# 注意：DPO 的学习率通常比 SFT 低一个数量级 (5e-6 vs 2e-5)
dpo_args = DPOConfig(
    output_dir="./qwen-dpo",
    beta=0.1,
    learning_rate=5e-6,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    fp16=True,
)

# 4. 初始化 Trainer
# TRL 会自动加载 ref_model (也就是 model 的一份拷贝，冻结参数)
trainer = DPOTrainer(
    model=model_id,
    ref_model=None, # 自动处理
    args=dpo_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_length=1024,
    max_prompt_length=512,
)

trainer.train()
```

---

## 4. PPO：经典 RLHF 三阶段 (进阶)

虽然 DPO 很火，但 PPO (Proximal Policy Optimization) 依然是理解 RLHF 的基石。如果你的数据集没有成对的偏好，只有一个 Scalar Reward（比如代码通过没通过测试），那么 PPO 是唯一的选择。

### 4.1 训练 Reward Model (RM)

在 PPO 之前，我们需要一个裁判模型（Reward Model）。它通常是一个 BERT 或者同架构的 Decoder 模型，将最后输出层改为一个标量回归头。

```python
from trl import RewardTrainer, RewardConfig

# 定义模型：AutoModelForSequenceClassification (num_labels=1)
model = AutoModelForSequenceClassification.from_pretrained(
    "Qwen/Qwen2-0.5B", num_labels=1
)

trainer = RewardTrainer(
    model=model,
    args=RewardConfig(output_dir="./qwen-rm", learning_rate=1e-5),
    train_dataset=dataset, # 包含 chosen/rejected
    tokenizer=tokenizer
)
trainer.train()
```

### 4.2 PPO 流程详解 (Actor-Critic)

这部分的代码非常复杂，涉及 4 个模型：
1.  **Actor**: 我们要训练的 Qwen-0.5B。
2.  **Ref Model**: 原始 Qwen-0.5B（冻结），用于计算 KL 散度，防止 Actor 跑偏（Reward Hacking）。
3.  **Critic**: 价值函数网络，估计 V(s)。
4.  **Reward Model**: 刚才训练好的裁判。

**核心代码逻辑**：
```python
# 伪代码流程
ppo_trainer = PPOTrainer(...)

for batch in dataloader:
    query = batch["input_ids"]

    # 1. Actor 生成回复
    response = ppo_trainer.generate(query)

    # 2. RM 打分
    reward = reward_model(query, response)

    # 3. PPO Update Step
    # 这一步会综合 reward 和 KL(actor, ref) 来更新 actor
    stats = ppo_trainer.step(query, response, reward)
```

---

## 5. 新兴趋势：ORPO 与 KTO

### 5.1 ORPO (单阶段微调)
**ORPO (Odds Ratio Preference Optimization)** 试图将 SFT 和 DPO 合二为一。
*   **原理**：在 SFT 的 Loss 上增加一项，专门惩罚 rejected 生成的概率。
*   **优势**：不需要 SFT -> DPO 两步走，**一步到位**。
*   **代码**：将 `DPOTrainer` 替换为 `ORPOTrainer` 即可，接口几乎一致。

### 5.2 KTO (非成对数据)
**KTO (Kahneman-Tversky Optimization)** 解决了 DPO 必须要有成对数据 `(A > B)` 的痛点。
*   如果你的数据只有 "A 是好的" (点赞) 和 "B 是坏的" (点踩)，没有配对关系，KTO 是最佳选择。

---

## 本章小结

本章我们用最轻量的 **Qwen2-0.5B** 跑通了最硬核的 **RLHF** 流程：

1.  **SFT**: 用 `packing=True` 高效教会模型指令格式。
2.  **DPO**: 用 `beta=0.1` 和成对数据，低成本实现偏好对齐（工业界 MVP）。
3.  **PPO**: 理解了 Actor/Critic/RM 的复杂博弈（学术界基石）。
4.  **前沿**: ORPO 和 KTO 提供了更灵活的选择。

现在，你手中的 Qwen-0.5B 不仅能说话，还能说出“符合人类偏好”的话。下一章，我们将探讨如何利用 **DeepSpeed** 将这一套流程扩展到 7B、70B 甚至更大的模型上。
