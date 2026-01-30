# 第2章：vLLM 高性能推理引擎实战

> **项目地址**：[https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)
>
> **本章定位**：vLLM 是目前 LLM 推理生态的**事实标准**。本章将从 PagedAttention 原理出发，带你掌握 **20倍吞吐量提升** 的秘诀，并解锁 **多 LoRA 并发** 和 **Prefix Caching** 等生产级特性。

---

## 目录

- [1. 为什么 vLLM 能快这么多？](#1-为什么-vllm-能快这么多)
  - [1.1 显存碎片的噩梦](#11-显存碎片的噩梦)
  - [1.2 PagedAttention 原理图解](#12-pagedattention-原理图解)
  - [1.3 Continuous Batching (持续批处理)](#13-continuous-batching-持续批处理)
- [2. vLLM 快速上手](#2-vllm-快速上手)
  - [2.1 离线批量推理 (Offline Inference)](#21-离线批量推理-offline-inference)
  - [2.2 启动 OpenAI 兼容服务 (API Server)](#22-启动-openai-兼容服务-api-server)
- [3. 进阶特性实战](#3-进阶特性实战)
  - [3.1 Prefix Caching：RAG 场景提速 10 倍](#31-prefix-cachingrag-场景提速-10-倍)
  - [3.2 Multi-LoRA：单卡服务多个微调模型](#32-multi-lora单卡服务多个微调模型)
  - [3.3 分布式推理 (Tensor Parallelism)](#33-分布式推理-tensor-parallelism)
- [4. 生产环境调优指南](#4-生产环境调优指南)
  - [4.1 显存利用率 (`gpu-memory-utilization`)](#41-显存利用率-gpu-memory-utilization)
  - [4.2 最大并发数 (`max-num-seqs`)](#42-最大并发数-max-num-seqs)
  - [4.3 Docker 部署最佳实践](#43-docker-部署最佳实践)
- [本章小结](#本章小结)

---

## 1. 为什么 vLLM 能快这么多？

在 vLLM 出现之前，Hugging Face 的原生推理（Naive Generation）存在严重的显存浪费问题。

### 1.1 显存碎片的噩梦

LLM 推理时，最大的显存消耗来自 **KV Cache**（存储历史 Token 的状态）。
*   **不可预知**：我们不知道用户会生成 10 个词还是 1000 个词。
*   **预留浪费**：为了防止 OOM，系统通常会预留“最大可能的长度”（比如 2048），结果只用了 100，剩下 95% 都是显存碎片。

这导致显存虽然很大，但只能塞进很小的 Batch Size。

### 1.2 PagedAttention 原理图解

vLLM 的作者从**操作系统**的虚拟内存管理中获得了灵感。

*   **OS 做法**：将内存切分为固定大小的 **Page (页)**。
*   **vLLM 做法**：将 KV Cache 切分为固定大小的 **Block (块)**（比如每块存 16 个 Token）。

**效果**：
*   **物理显存不需要连续**：Token A 的 KV 可能在显存地址 0x100，Token B 可能在 0x900，通过查表（Block Table）连接。
*   **零浪费**：只需按需分配 Block，显存碎片率降至 < 4%。

**结论**：同样的显存，vLLM 可以塞进 **2-4 倍** 的 Batch Size，从而实现 **10-20 倍** 的吞吐量提升。

### 1.3 Continuous Batching (持续批处理)

传统 Batching 是“等最慢的那个跑完”才能跑下一轮。
vLLM 实现了 **Iteration-level Scheduling**：
*   如果 Request A 先生成完了，立马把它的显存释放出来，让 Request C 插队进来。
*   GPU 永远处于满载状态，不论长短文本混合。

---

## 2. vLLM 快速上手

### 2.1 离线批量推理 (Offline Inference)

适用于离线处理此数据（如给 10 万条数据打标）。

```python
from vllm import LLM, SamplingParams

# 1. 初始化引擎
# tensor_parallel_size=2 表示用 2 张卡跑一个模型 (TP)
llm = LLM(model="meta-llama/Llama-3-8B-Instruct", tensor_parallel_size=1)

# 2. 定义 Prompt
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# 3. 采样参数
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)

# 4. 批量生成
outputs = llm.generate(prompts, sampling_params)

# 5. 打印结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

### 2.2 启动 OpenAI 兼容服务 (API Server)

这是 vLLM 最常用的模式，直接替代 Flask/FastAPI 封装。

```bash
# 启动命令
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-8B-Instruct \
    --served-model-name llama3 \
    --port 8000 \
    --trust-remote-code
```

**调用测试** (完全兼容 OpenAI SDK)：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY" # vLLM 默认无鉴权
)

completion = client.chat.completions.create(
    model="llama3",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(completion.choices[0].message.content)
```

---

## 3. 进阶特性实战

### 3.1 Prefix Caching：RAG 场景提速 10 倍

在 RAG 或多轮对话中，**System Prompt** 或 **长文档 Context** 是重复的。
vLLM 可以**自动缓存**这些公共前缀的 KV Cache。

**启用方法**：
只需在启动时添加参数：
```bash
--enable-prefix-caching
```

**效果**：
*   第一个 Request：处理长文档，耗时 500ms。
*   第二个 Request (相同文档)：直接命中缓存，耗时 10ms。
*   **适用场景**：超长 System Prompt、文档问答。

### 3.2 Multi-LoRA：单卡服务多个微调模型

想象一下，你有 10 个业务场景，分别微调了 10 个 LoRA Adapter。
以前需要部署 10 个 vLLM 实例（太费显存）。
现在只需要 **1 个 Base Model + 10 个 LoRA Adapters**。

**启动命令**：
```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-8B-Instruct \
    --enable-lora \
    --lora-modules sql_lora=./lora_sql_adapter chat_lora=./lora_chat_adapter
```

**请求时指定 LoRA**：
```python
# 请求 SQL 能力
client.chat.completions.create(
    model="sql_lora", # 指定 LoRA 名字
    messages=[...]
)

# 请求 闲聊 能力
client.chat.completions.create(
    model="chat_lora",
    messages=[...]
)
```

### 3.3 分布式推理 (Tensor Parallelism)

对于 70B 模型，单卡 24G/40G/80G 都放不下。需要用 **TP (Tensor Parallelism)** 切分模型。

```bash
# 自动检测可用 GPU 数量并切分
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-70B-Instruct \
    --tensor-parallel-size 4  # 使用 4 张卡
```

> **注意**：TP 依赖 NVLink 通信。如果是 PCIe (如 4x 3090)，通信开销会较大，推理速度可能不如预期。

---

## 4. 生产环境调优指南

vLLM 虽然快，但配置不当也会 OOM 或卡顿。

### 4.1 显存利用率 (`gpu-memory-utilization`)

*   **默认值**：`0.90` (占用 90% 显存用于 Model + KV Cache)。
*   **坑点**：如果显存被其他进程（如 X server, 监控软件）占用了，vLLM 启动会报错。
*   **建议**：单卡独占时设为 `0.95` 榨干性能；混合部署时设为 `0.8` 或更低。

```bash
--gpu-memory-utilization 0.95
```

### 4.2 最大并发数 (`max-num-seqs`)

*   **定义**：同一时刻在 GPU 上处理的 Request 数量。
*   **建议**：默认 256。如果你的显存很大（A100 80G）且模型很小（8B），可以把这个值调大到 1024，极大提升吞吐。

### 4.3 Docker 部署最佳实践

不要在裸机上跑 vLLM，环境依赖（CUDA, Pytorch, NCCL）太复杂。

**官方 Dockerfile**：
```yaml
# docker-compose.yml 示例
version: "3.8"
services:
  vllm:
    image: vllm/vllm-openai:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    command: >
      --model meta-llama/Llama-3-8B-Instruct
      --gpu-memory-utilization 0.95
      --port 8000
    ports:
      - "8000:8000"
    shm_size: '10gb' # 关键：NCCL 通信需要大共享内存
```

---

## 本章小结

vLLM 是当前 LLM 推理的**必修课**。

1.  **快**：PagedAttention + Continuous Batching 带来 20x 吞吐。
2.  **省**：Prefix Caching 省 RAG 算力，Multi-LoRA 省显存。
3.  **稳**：OpenAI 接口兼容，Docker 部署方案成熟。

在下一章 **《第3章：生产部署最佳实践》** 中，我们将探讨更复杂的部署场景：如何在 Nginx 层面做负载均衡？如何利用 Triton Inference Server 做更高阶的模型编排？
