# 第5章：端到端项目：LawGLM 法律咨询助手

> **本章定位**：综合大作业。串联前4章知识，从零构建一个垂直领域的法律问答助手。

---

## 目录
- [项目目标](#项目目标)
- [技术栈](#技术栈)
- [1. Step 1: 数据准备 (Data Engineering)](#1-step-1-数据准备-data-engineering)
- [2. Step 2: 微调训练 (Fine-tuning)](#2-step-2-微调训练-fine-tuning)
- [3. Step 3: 模型合并与量化](#3-step-3-模型合并与量化)
- [4. Step 4: 服务API开发](#4-step-4-服务api开发)
- [5. Step 5: 前端交互与评估](#5-step-5-前端交互与评估)
- [本章小结](#本章小结)

---

## 项目目标
构建一个能够回答中国法律问题、辅助撰写法律文书的 LLM。

## 技术栈
- **数据**：Pandas, Datasets
- **微调**：LLaMA-Factory (LoRA + ZeRO-2)
- **评估**：LLM-as-a-Judge (GPT-4 打分)
- **部署**：vLLM

---

## 1. Step 1: 数据准备 (Data Engineering)

我们需要构建两类数据：**法律知识注入**（Self-Instruct）和 **问答对**。

### 1.1 数据清洗脚本

假设我们有原始的 PDF/TXT 法律条文，先处理成 JSONL。

```python
import json
import re

def process_law_text(text):
    # 简单的正则匹配，提取 "第X条"
    articles = re.split(r'(第[0-9]+条)', text)
    results = []
    for i in range(1, len(articles), 2):
        article_num = articles[i]
        content = articles[i+1].strip()
        results.append({
            "instruction": f"请解释{article_num}的内容。",
            "input": "",
            "output": content
        })
    return results

# 保存为 Alpaca 格式
data = process_law_text("中华人民共和国民法典...")
with open("law_data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
```

### 1.2 数据注册

在 LLaMA-Factory 的 `data/dataset_info.json` 中注册：

```json
"law_glm_sft": {
  "file_name": "law_data.json",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output"
  }
}
```

---

## 2. Step 2: 模型微调 (SFT)

使用 LLaMA-Factory 进行 LoRA 微调。

### 配置文件 `examples/law_finetune.yaml`

```yaml
model_name_or_path: Qwen/Qwen1.5-7B-Chat
stage: sft
do_train: true
finetuning_type: lora
lora_target: all

dataset: law_glm_sft
template: qwen
cutoff_len: 2048
max_samples: 10000
overwrite_cache: true
preprocessing_num_workers: 8

output_dir: saves/Qwen1.5-7B/law-lora
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 5.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
fp16: true
```

### 启动训练

```bash
# 单机 4 卡训练
llamafactory-cli train examples/law_finetune.yaml
```

---

## 3. Step 3: 合并与导出

为了提高推理速度，我们将 LoRA 权重合并回基座模型。

```bash
llamafactory-cli export \
    --model_name_or_path Qwen/Qwen1.5-7B-Chat \
    --adapter_name_or_path saves/Qwen1.5-7B/law-lora \
    --template qwen \
    --finetuning_type lora \
    --export_dir models/LawGLM-7B \
    --export_size 2 \
    --export_legacy_format false
```

现在，`models/LawGLM-7B` 目录就是一个完整的、可独立运行的模型了。

---

## 4. Step 4: 性能评估 (Evaluation)

使用一个独立测试集，通过 GPT-4 对 LawGLM 的回答进行打分 (1-10分)，维度包括：准确性、专业性、安全性。

### 评分脚本示例

```python
# 伪代码
import openai

def evaluate(question, answer, reference):
    prompt = f"""
    问题：{question}
    参考答案：{reference}
    模型回答：{answer}

    请作为一名法官，对模型回答的法律准确性打分（1-10），并给出理由。
    格式：分数 | 理由
    """
    res = openai.ChatCompletion.create(model="gpt-4", messages=[...])
    return res
```

运行该脚本对比 **Base Model** 和 **LawGLM** 的平均分。

---

## 5. Step 5: 高并发部署 (Deployment)

使用 vLLM 部署合并后的模型。vLLM 提供了 PagedAttention 技术，吞吐量远超 Hugging Face 原生推理。

### 5.1 启动 API Server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model models/LawGLM-7B \
    --served-model-name law-glm \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --port 8000
```

### 5.2 调用测试 (OpenAI 兼容接口)

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

completion = client.chat.completions.create(
    model="law-glm",
    messages=[
        {"role": "system", "content": "你是一名专业的法律顾问。"},
        {"role": "user", "content": "我的房东无故不退押金，我该怎么办？"}
    ]
)

print(completion.choices[0].message.content)
```

---

## 6. 项目总结

通过这个项目，我们串联了 LLM 工程化的全流程：
1. **数据**是核心：格式化、清晰的法律数据决定了模型上限。
2. **工具**是加速器：LLaMA-Factory 屏蔽了繁琐的代码细节。
3. **部署**是落地：vLLM 让我们能用生产级的速度提供服务。

这套 `Data -> LLaMA-Factory -> Merge -> vLLM` 的流程是目前（2025年）业界最标准的单体 LLM 应用开发范式。
