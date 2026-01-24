# 第4章：DeepSpeed分布式训练

> **本章定位**：突破单卡显存瓶颈。学习编写 `ds_config.json`，掌握 ZeRO 系列优化器，并对比 PyTorch 原生 FSDP。

---

## 目录
- [1. 为什么需要 DeepSpeed？](#1-为什么需要-deepspeed)
- [2. 核心：ds_config.json 配置实战](#2-核心ds_configjson-配置实战)
- [3. ZeRO-3与Offload实战](#3-zero-3与offload实战)
- [4. 混合精度训练](#4-混合精度训练)
- [5. 多节点训练 (Multi-Node)](#5-多节点训练-multi-node)
- [本章小结](#本章小结)

---

## 1. 为什么需要 DeepSpeed？

当模型参数量超过显存限制（例如在 24G 显存上训练 13B 模型）时，普通的 DDP (Distributed Data Parallel) 就无能为力了。DeepSpeed 的核心武器是 **ZeRO (Zero Redundancy Optimizer)**，它将模型状态切分到不同的 GPU 上。

### ZeRO 三阶段（简单记忆版）
- **ZeRO-1**: 切分**优化器状态** (Optimizer States)。显存节省 4 倍。
- **ZeRO-2**: 切分**优化器状态 + 梯度** (Gradients)。显存节省 8 倍。
- **ZeRO-3**: 切分**优化器状态 + 梯度 + 模型参数** (Parameters)。显存节省与 GPU 数量成正比 (线性扩展)。

---

## 2. 核心：ds_config.json 配置实战

DeepSpeed 不需要修改太多代码，主要是通过配置文件来控制。

### 2.1 ZeRO-2 配置（推荐用于大多数显存足够的微调）

```json
{
  "train_batch_size": "auto",
  "gradient_accumulation_steps": "auto",
  "steps_per_print": 100,
  "gradient_clipping": 1.0,
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "initial_scale_power": 16
  },
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "overlap_comm": true,  // 通信与计算重叠，加速训练
    "contiguous_gradients": true
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto",
      "weight_decay": "auto"
    }
  }
}
```

### 2.2 ZeRO-3 Offload 配置（穷人救星）

如果你显存非常小（如单卡 3090 跑 70B 模型），必须使用 ZeRO-3 **Offload**，将优化器状态和参数卸载到 CPU 内存。

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",     // 关键：卸载优化器到 CPU
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",     // 关键：卸载参数到 CPU (可选，速度更慢但显存更省)
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_prefetch_bucket_size": 1e7,
    "stage3_param_persistence_threshold": 1e5,
    "reduce_bucket_size": 1e7,
    "sub_group_size": 1e9
  },
  "fp16": { "enabled": true },
  "train_batch_size": "auto",
  "gradient_accumulation_steps": "auto"
}
```

---

## 3. 代码集成

在 Hugging Face `Trainer` 中使用 DeepSpeed 非常简单，不需要修改 Python 代码逻辑，甚至不需要显式 import deepspeed。

**方式 1：通过 TrainingArguments 传入**

```python
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./res",
    deepspeed="./ds_config_zero2.json", # 直接指定配置文件路径
    per_device_train_batch_size=4,
    # ... 其他参数
)
```

**方式 2：通过 Accelerate CLI 启动**

不修改代码，只在启动时指定：

```bash
accelerate launch --use_deepspeed --zero_stage 2 train.py
```

---

## 4. 2025年视角：DeepSpeed vs FSDP

PyTorch 原生的 **FSDP (Fully Sharded Data Parallel)** 已经变得非常成熟。

### 4.1 选型指南

- **DeepSpeed ZeRO-3**:
  - **优势**: 生态更好（HF 集成度高），Offload 策略更激进（能在极小显存跑极大模型）。
  - **劣势**: 依赖多，环境配置偶尔有坑。
- **PyTorch FSDP**:
  - **优势**: PyTorch 原生，无额外依赖，对 Llama 等结构支持越来越好。
  - **劣势**: Offload 能力稍弱于 DeepSpeed。

### 4.2 FSDP 在 Accelerate 中的配置

无需写 json 文件，只需 `accelerate config` 时选择 FSDP。

```bash
$ accelerate config
# ...
# Do you want to use FSDP? [yes/NO]: yes
# FSDP Sharding Strategy? [FULL_SHARD] (等同于 ZeRO-3)
# FSDP Offload? [true/false]
# ...
```

代码中：

```python
# 无需任何修改！
# Accelerate 会自动接管
```

---

## 5. 本章小结

1. **ZeRO-2**: 目前性价比最高的选择。适合多卡微调。
2. **ZeRO-3**: 大模型（>13B）全量微调必备。如果爆显存，开启 `offload_optimizer: cpu`。
3. **ZeRO-Offload**: 利用 CPU 内存换取 GPU 显存空间，用时间换空间。
4. **FSDP**: PyTorch 原生替代方案，值得尝试。

**避坑指南**：
- 使用 DeepSpeed 时，`train_batch_size` 在 json 里设为 `"auto"`，让 Hugging Face 的 arguments 来控制，避免冲突。
- 开启 ZeRO-3 后，模型保存速度会变慢（需要从各 GPU 收集参数），请耐心等待。
