# 第2章：LLaMA-Factory 微调工厂

> **项目地址**：[https://github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
>
> **本章定位**：从手写 PyTorch 进阶到“流水线工厂”。学会利用 LLaMA-Factory 进行零代码（WebUI）和低代码（CLI）的高效微调，涵盖从 SFT 到模型导出（Merge）的全流程。

---

## 目录

- [1. 为什么选择 LLaMA-Factory？](#1-为什么选择-llama-factory)
- [2. 环境搭建与 Unsloth 加速](#2-环境搭建与-unsloth-加速)
  - [2.1 标准安装](#21-标准安装)
  - [2.2 开启 Unsloth 极速模式（推荐）](#22-开启-unsloth-极速模式推荐)
- [3. 数据工程：Dataset Registration](#3-数据工程dataset-registration)
  - [3.1 数据格式标准 (Alpaca vs ShareGPT)](#31-数据格式标准-alpaca-vs-sharegpt)
  - [3.2 注册自定义数据集 (`dataset_info.json`)](#32-注册自定义数据集-dataset_infojson)
- [4. 可视化微调：WebUI 全流程](#4-可视化微调webui-全流程)
  - [4.1 启动与界面概览](#41-启动与界面概览)
  - [4.2 训练参数配置详解](#42-训练参数配置详解)
  - [4.3 训练监控与评估](#43-训练监控与评估)
- [5. 生产化：从 WebUI 到 CLI 自动化](#5-生产化从-webui-到-cli-自动化)
  - [5.1 导出 YAML 配置文件](#51-导出-yaml-配置文件)
  - [5.2 命令行启动训练](#52-命令行启动训练)
  - [5.3 多机多卡分布式配置](#53-多机多卡分布式配置)
- [6. 模型导出与合并](#6-模型导出与合并)
- [本章小结](#本章小结)

---

## 1. 为什么选择 LLaMA-Factory？

在 LLaMA-Factory 出现之前，微调一个模型需要自己手写 PEFT 代码、处理复杂的 Padding、适配 Flash Attention。LLaMA-Factory 解决了以下**核心痛点**：

1.  **多模型适配**：一套代码支持 Llama-3, Qwen-2, Mistral, Gemma 等 100+ 模型。
2.  **多算法集成**：无缝切换 Full, LoRA, QLoRA, DoRA, PPO, DPO。
3.  **多硬件兼容**：自动适配 DeepSpeed (ZeRO), Unsloth (Triton优化), FlashAttention-2。
4.  **零代码门槛**：提供 WebUI 界面，小白也能点点鼠标跑通微调。

---

## 2. 环境搭建与 Unsloth 加速

### 2.1 标准安装

推荐使用 PyTorch 2.4+ 和 CUDA 12.1+ 环境。

```bash
# 1. 克隆仓库
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

# 2. 安装依赖 (推荐先创建 conda 环境)
# [metrics] 用于评估，[bitsandbytes] 用于量化
pip install -e ".[torch,metrics,bitsandbytes]"
```

### 2.2 开启 Unsloth 极速模式（推荐）

**Unsloth** 是当前最强的微调加速库，通过重写 Triton 内核，能实现：
- 训练速度提升 **2-5 倍**
- 显存占用减少 **60%** (单张 T4/4060 也能跑 Llama-3-8B)

安装 Unsloth (需根据 CUDA 版本选择，以下以 CUDA 12.1 为例)：

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

> **注意**：Windows 用户请参考 [Unsloth 官方指南](https://github.com/unslothai/unsloth#installation-instructions) 使用 WSL2 安装。

---

## 3. 数据工程：Dataset Registration

**这是新手最容易卡壳的地方**。LLaMA-Factory 不直接读取 raw text，必须先在 `dataset_info.json` 中注册。

### 3.1 数据格式标准 (Alpaca vs ShareGPT)

准备你的数据 `my_data.json`，推荐以下两种格式：

**格式 A：Alpaca 格式（适合单轮指令）**
```json
[
  {
    "instruction": "请解释什么是量子纠缠",
    "input": "",
    "output": "量子纠缠是量子力学中的一种现象..."
  },
  {
    "instruction": "将以下文本翻译成英文",
    "input": "你好，世界",
    "output": "Hello, World"
  }
]
```

**格式 B：ShareGPT 格式（适合多轮对话）** -> **推荐**
```json
[
  {
    "conversations": [
      { "from": "human", "value": "你好" },
      { "from": "gpt", "value": "你好！有什么我可以帮你的吗？" },
      { "from": "human", "value": "写首诗" },
      { "from": "gpt", "value": "明月几时有..." }
    ]
  }
]
```

### 3.2 注册自定义数据集 (`dataset_info.json`)

打开 `data/dataset_info.json`，在末尾添加你的数据集配置：

```json
{
  "identity": { "file_name": "identity.json" },
  "my_custom_data": {
    "file_name": "my_data.json", // 你的文件必须放在 data/ 目录下
    "formatting": "sharegpt",     // 格式：alpaca 或 sharegpt
    "columns": {
      "messages": "conversations" // 映射你的字段名
    }
  }
}
```

> **校验技巧**：如果不确定格式对不对，直接运行 WebUI，在数据预览页查看是否能正确解析。

---

## 4. 可视化微调：WebUI 全流程

WebUI 是调试参数的最佳场所。调试满意后，我们再导出命令去后台运行。

### 4.1 启动与界面概览

```bash
# 启动 WebUI
# 默认端口 7860
llamafactory-cli webui
```

> 📸 **[截图占位]：WebUI 主界面**
> *请截取浏览器打开 `localhost:7860` 后的界面，重点框出：语言切换（ZH）、模型选择区、微调方法区。*

**核心操作步骤**：
1.  **语言**：选择 `zh` (中文)。
2.  **模型名称**：选择 `LLaMA-3-8B-Instruct`。
3.  **微调方法**：选择 `LoRA`。
4.  **适配器路径**：(训练时留空，合并模型时才填)。

### 4.2 训练参数配置详解

进入 **[Train] (训练)** 选项卡。

> 📸 **[截图占位]：训练参数配置面板**
> *重点展示：数据集选择、学习率、秩(Rank)、批处理大小。*

**关键参数指南**：

| 参数项 | 推荐值 | 说明 |
| :--- | :--- | :--- |
| **数据集** | `my_custom_data` | 刚刚注册的数据集。 |
| **截断长度 (Cutoff Len)** | `1024` ~ `4096` | 根据显存决定。超长文本会被截断。 |
| **学习率 (Learning Rate)** | `5e-5` ~ `1e-4` | LoRA 通常需要比全量微调大一点的 LR。 |
| **轮数 (Epochs)** | `3` ~ `5` | 数据少就多跑几轮，数据多跑1-2轮。 |
| **批处理大小 (Batch/GPU)** | `4` ~ `16` | 显存不够就减小，开梯度累积。 |
| **梯度累积 (Grad Accum)** | `4` | 它可以模拟大 Batch Size 效果。 |
| **LoRA 秩 (Rank)** | `8` ~ `64` | 越大显存占用越高，拟合能力越强。一般 `16` 或 `32` 够用。 |
| **LoRA Alpha** | `16` 或 `32` | 通常设为 Rank 的 1倍或 2倍。 |
| **Target Modules** | `all` | 推荐微调所有线性层 (q,k,v,o,gate,up,down)，效果最好。 |

### 4.3 训练监控与评估

点击 **[Start] (开始)** 按钮后，右侧会显示 Loss 曲线。

> 📸 **[截图占位]：训练中的 Loss 曲线图**
> *展示 Loss 随 Step 下降的趋势。*

**如何判断训练正常？**
- **Loss 快速下降**：初期从 2.0+ 降到 1.0 左右是正常的。
- **Loss 震荡**：如果 Loss 忽高忽低，尝试减小学习率。
- **Loss 贴地飞行 (0.01)**：可能是过拟合了，或者数据太少。

---

## 5. 生产化：从 WebUI 到 CLI 自动化

WebUI 最大的价值在于**“预览命令”**。在生产环境中，我们需要用命令行（CLI）来运行，以便挂后台 (`nohup`) 或多机运行。

### 5.1 导出 YAML 配置文件

在 WebUI 点击 **[Preview Command] (预览命令)**，或者直接 **[Save Arguments]** 保存配置。

推荐将配置保存为 `examples/train_lora.yaml`：

```yaml
### model
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all

### dataset
dataset: my_custom_data
template: llama3
cutoff_len: 1024
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/llama3-8b/lora/sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
flash_attn: fa2

### val
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 500
```

### 5.2 命令行启动训练

有了 yaml 文件，启动训练非常优雅：

```bash
# 单机单卡 / 单机多卡 (自动检测)
llamafactory-cli train examples/train_lora.yaml
```

### 5.3 多机多卡分布式配置

如果是多台服务器，需要结合 `FORCE_TORCHRUN=1`：

```bash
FORCE_TORCHRUN=1 NNODES=2 NODE_RANK=0 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 \
llamafactory-cli train examples/train_lora.yaml
```

---

## 6. 模型导出与合并

LoRA 训练完后，你会得到一个几十 MB 的适配器文件夹。为了部署（如 vLLM 或 Ollama），通常需要将 LoRA 权重合并回基座模型（Merge）。

**WebUI 操作**：
1. 切换到 **[Export] (导出)** 选项卡。
2. 选择 **Adapter Path**：你刚才训练的输出目录 `saves/...`。
3. 选择 **Export Dir**：合并后模型的保存路径。
4. 点击 **[Export]**。

> 📸 **[截图占位]：模型导出界面**

**CLI 操作**：
创建 `merge.yaml`：

```yaml
### model
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
adapter_name_or_path: saves/llama3-8b/lora/sft
template: llama3
finetuning_type: lora

### export
export_dir: models/llama3-8b-sft-merged
export_size: 2
export_device: cpu
export_legacy_format: false
```

运行导出：
```bash
llamafactory-cli export merge.yaml
```

---

## 本章小结

LLaMA-Factory 定义了 LLM 微调的**工业标准流程**：

1.  **准备**：安装 Unsloth，整理数据为 ShareGPT 格式。
2.  **注册**：在 `dataset_info.json` 中配置数据映射。
3.  **调试**：用 WebUI 快速验证超参（Rank, LR, Batch）。
4.  **运行**：导出 YAML，使用 `llamafactory-cli train` 挂后台训练。
5.  **交付**：使用 `export` 命令合并权重，交付完整模型。

有了这个神器，你不再需要关心底层 PyTorch 的分布式细节，专注于数据质量和模型效果即可。下一章，我们将探讨如何利用微调后的模型进行**强化学习 (RLHF/DPO)**，进一步对齐人类偏好。
