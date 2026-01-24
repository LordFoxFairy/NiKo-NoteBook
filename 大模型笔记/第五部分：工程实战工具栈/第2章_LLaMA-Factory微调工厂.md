# 第2章：LLaMA-Factory微调工厂

> **本章定位**：零代码/低代码微调神器。重点掌握 **配置文件的编写** 和 **WebUI 的操作流**。

---

## 目录
- [1. 为什么选择 LLaMA-Factory？](#1-为什么选择-llama-factory)
- [2. 核心：YAML 配置文件详解](#2-核心yaml-配置文件详解)
- [3. Web UI 零代码微调](#3-web-ui-零代码微调)
- [4. 数据集配置管理](#4-数据集配置管理)
- [5. 高级用法](#5-高级用法)
- [本章小结](#本章小结)

---

## 1. 为什么选择 LLaMA-Factory？

LLaMA-Factory 实际上是一套基于 Transformers 和 PEFT 的**高级封装**。它解决了以下痛点：
1. **模板对齐**：自动处理 ChatML, Alpaca, Llama2, Llama3 等各种繁杂的 Prompt Template。
2. **环境适配**：一键开启 FlashAttention-2, Unsloth (加速训练), DeepSpeed。
3. **算法集成**：集成了 LoRA, QLoRA, GaLore, DoRA 等最新微调技术。

---

## 2. 核心：YAML 配置文件详解

在 CLI 模式下，我们通过编写 `yaml` 文件来控制训练。这是生产环境中最常用的方式。

### 2.1 配置文件结构 (`examples/train_lora.yaml`)

```yaml
# --- 模型参数 ---
model_name_or_path: meta-llama/Llama-3-8B-Instruct  # 基座模型路径
trust_remote_code: true

# --- 方法参数 ---
stage: sft                     # 训练阶段: sft, pt(预训练), rm(奖励模型), ppo, dpo
do_train: true
finetuning_type: lora          # 微调方式: lora, full, freeze
lora_target: all               # LoRA 挂载目标: all (推荐), q_proj,v_proj 等
lora_rank: 16                  # LoRA 秩: 8, 16, 32, 64
lora_alpha: 16                 # LoRA 缩放系数: 通常 = rank 或 rank * 2
lora_dropout: 0.05

# --- 数据参数 ---
dataset: identity,alpaca_en    # 数据集名称，需在 dataset_info.json 中注册
template: llama3               # 对应的 Prompt 模板，这步非常关键！
cutoff_len: 1024               # 截断长度
overwrite_cache: true
preprocessing_num_workers: 16

# --- 训练参数 ---
output_dir: saves/llama3-8b/lora/sft  # 输出目录
logging_steps: 10
save_steps: 100
plot_loss: true
overwrite_output_dir: true

# --- 显存与优化配置 ---
per_device_train_batch_size: 2
gradient_accumulation_steps: 4 # 累积梯度，显存不够时调大这个，调小 batch_size
learning_rate: 5.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
fp16: true                     # 开启混合精度
# bf16: true                   # 如果是 A100/H100 建议开启 bf16

# --- 高级优化 ---
# flash_attn: fa2              # 显式开启 FlashAttention-2
# quantization_bit: 4          # 开启 QLoRA 4bit 量化（极大节省显存）
```

### 2.2 启动命令

编写好 `custom_sft.yaml` 后，一行命令启动：

```bash
llamafactory-cli train custom_sft.yaml
```

---

## 3. Web UI 零代码微调实战

对于初学者或快速实验，Web UI 提供了直观的界面。

**启动方式**：
```bash
llamafactory-cli webui
```

### 3.1 界面操作流程 (LLaMA Board)

界面通常分为几个主要特定的 Tab，操作流如下：

**Step 1: 模型选择 (Model Selection)**
- **Model Name**: 选择 `LLaMA-3-8B-Instruct`。
- **Model Path**: 如果本地已下载，填本地绝对路径；否则留空自动下载。
- **Adapter Path**: 训练时留空；如果是**合并/推理**阶段，填入之前的 checkpoint 路径，列表刷新后可见。

**Step 2: 训练设置 (Train)**
- **Stage**: 选 `Supervised Finetuning`。
- **Data Dir**: 数据集目录。
- **Dataset**: 在下拉框选择数据集（如 `alpaca_en`）。如果要用自己的数据，需先在 `data/dataset_info.json` 中配置引用。
- **Learning Rate**: 推荐 `5e-5` (LoRA) 或 `1e-5` (Full)。
- **Epochs**: `3.0`。
- **Quantization**: 显存不足（如 12G/16G 显存跑 8B 模型）可选 `4` 或 `8` (QLoRA)。

**Step 3: 参数设置 (Advanced)**
- **LoRA Rank**: `16`。
- **LoRA Alpha**: `16`。
- **Target Modules**: 选 `all` 效果最好，或手动指定 `q_proj, v_proj`。

**Step 4: 开始 (Run)**
- 点击 **Preview Command**: 查看生成的命令行参数，方便学习 CLI 用法。
- 点击 **Start**: 终端会开始跑进度条，Loss 曲线会实时画在右侧。

**Step 5: 导出模型 (Export)**
- 训练完成后，切换到 `Export` Tab。
- **Max Shard Size**: 设置分块大小（如 `2GB`）。
- **Export Dir**: 防止路径及文件名。
- 点击 **Export**：工具会自动将 Base Model 和 LoRA Adapter 合并，并保存为完整的 Hugging Face 格式模型，可直接用于 vLLM 部署。

---

## 4. 自定义数据集实战

LLaMA-Factory 通过 `data/dataset_info.json` 管理所有数据集。

### 4.1 数据准备 (Alpaca 格式)

准备一个 JSON 文件 `data/my_law_data.json`：

```json
[
  {
    "instruction": "解释什么是不可抗力。",
    "input": "",
    "output": "不可抗力是指不能预见、不能避免并不能克服的客观情况..."
  },
  {
    "instruction": "分析该合同条款是否有效。",
    "input": "条款内容：...",
    "output": "根据合同法第X条，该条款无效，因为..."
  }
]
```

### 4.2 注册数据

编辑 `data/dataset_info.json`，加入以下内容：

```json
"my_law_dataset": {
  "file_name": "my_law_data.json",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output"
  }
}
```

### 4.3 使用数据

- **CLI**: 在 yaml 中设置 `dataset: my_law_dataset`。
- **WebUI**: 刷新 Dataset 列表，即可看到 `my_law_dataset`。

---

## 5. 本章小结

LLaMA-Factory 是目前（2025年）**效率通过率最高**的微调工具：
1. **数据**：只需转成 Alpaca 格式并注册。
2. **微调**：优先使用 `LoRA` + `FlashAttention-2`。
3. **显存优化**：显存吃紧就开 `quantization_bit: 4` (QLoRA) 和 `gradient_accumulation_steps`。
4. **验证**：WebUI 自带 `Chat` 页面，可以加载 Adapter 实时对话测试效果。
