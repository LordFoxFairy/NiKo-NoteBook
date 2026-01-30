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

我们需要构建三类数据：**法律条文知识注入**、**判例问答对** 和 **法律咨询对话**。

### 1.1 数据源规划

```
数据来源：
1. 法律条文：中国裁判文书网、法律法规数据库
2. 判例分析：最高人民法院公报案例
3. 咨询问答：Legal Advice Reddit、知乎法律话题（经人工清洗）

目标数据量：
- 训练集：10,000+ 条高质量问答对
- 验证集：500 条
- 测试集：500 条（用于 GPT-4 评估）
```

### 1.2 数据清洗脚本

#### 1.2.1 法律条文处理

```python
import json
import re
from pathlib import Path

def extract_law_articles(text: str, law_name: str) -> list:
    """
    从法律条文中提取结构化数据

    Args:
        text: 原始法律条文
        law_name: 法律名称（如"民法典"）

    Returns:
        list: 结构化的问答对
    """
    # 正则匹配 "第X条" 格式
    pattern = r'第([零一二三四五六七八九十百千万\d]+)条\s+(.*?)(?=第[零一二三四五六七八九十百千万\d]+条|$)'
    matches = re.findall(pattern, text, re.DOTALL)

    results = []
    for article_num, content in matches:
        content = content.strip()
        if len(content) < 10:  # 过滤过短的条文
            continue

        # 生成多种问法（数据增强）
        results.extend([
            {
                "instruction": f"请解释《{law_name}》第{article_num}条的内容。",
                "input": "",
                "output": content
            },
            {
                "instruction": f"《{law_name}》第{article_num}条规定了什么？",
                "input": "",
                "output": content
            },
            {
                "instruction": "法律问题咨询",
                "input": f"请帮我查询《{law_name}》第{article_num}条",
                "output": f"《{law_name}》第{article_num}条规定：{content}"
            }
        ])

    return results

# 示例：处理民法典
civil_code_text = """
第一条 为了保护民事主体的合法权益，调整民事关系，维护社会和经济秩序，适应中国特色社会主义发展要求，弘扬社会主义核心价值观，根据宪法，制定本法。
第二条 民法调整平等主体的自然人、法人和非法人组织之间的人身关系和财产关系。
第三条 民事主体的人身权利、财产权利以及其他合法权益受法律保护，任何组织或者个人不得侵犯。
"""

law_data = extract_law_articles(civil_code_text, "民法典")
print(f"提取了 {len(law_data)} 条法律知识")
```

#### 1.2.2 判例问答对构造

```python
def create_case_qa(case_dict: dict) -> dict:
    """
    将判例转换为问答格式

    Args:
        case_dict: 包含 case_title, facts, judgment 等字段的判例

    Returns:
        dict: Alpaca 格式的问答对
    """
    return {
        "instruction": "请分析以下案件，并给出法律意见。",
        "input": f"案件：{case_dict['case_title']}\n事实：{case_dict['facts']}",
        "output": f"法律分析：\n{case_dict['legal_analysis']}\n\n判决结果：\n{case_dict['judgment']}"
    }

# 示例数据
sample_case = {
    "case_title": "张某诉李某房屋租赁合同纠纷案",
    "facts": "原告张某与被告李某签订房屋租赁合同，约定租期一年，租金每月3000元。租期届满后，被告拒不退还押金5000元，理由是房屋内设施损坏。",
    "legal_analysis": "根据《民法典》第704条，租赁期限届满，承租人应当返还租赁物。因承租人原因导致租赁物毁损的，出租人可以扣除相应押金。但本案中，被告未能提供充分证据证明设施损坏系原告造成，且损坏价值未经评估。",
    "judgment": "判决被告李某于判决生效之日起十日内返还原告张某押金5000元。"
}

case_qa = create_case_qa(sample_case)
print(json.dumps(case_qa, ensure_ascii=False, indent=2))
```

#### 1.2.3 数据质量控制

```python
def validate_data_quality(data_list: list) -> list:
    """
    过滤低质量数据
    """
    filtered = []
    for item in data_list:
        # 1. 长度检查
        if len(item["output"]) < 20 or len(item["output"]) > 2048:
            continue

        # 2. 关键词检查（避免包含敏感内容）
        sensitive_keywords = ["暴力", "色情", "赌博"]
        if any(kw in item["output"] for kw in sensitive_keywords):
            continue

        # 3. 格式规范检查
        if not item["instruction"] or not item["output"]:
            continue

        filtered.append(item)

    return filtered

# 合并所有数据
all_data = law_data + [case_qa]  # 实际项目中添加更多数据
clean_data = validate_data_quality(all_data)

# 保存为 Alpaca 格式
output_path = Path("data/law_glm_train.json")
output_path.parent.mkdir(exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(clean_data, f, ensure_ascii=False, indent=2)

print(f"✓ 数据清洗完成，保存了 {len(clean_data)} 条数据到 {output_path}")
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

## 2. Step 2: 模型微调 (QLoRA SFT)

使用 LLaMA-Factory 进行 **QLoRA** 微调（4-bit 量化），在单张 24GB 显卡上训练 7B 模型。

### 2.1 配置文件 `law_finetune.yaml`

```yaml
# ===== 模型配置 =====
model_name_or_path: deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct  # 或 meta-llama/Llama-3-8B-Instruct
trust_remote_code: true

# ===== 微调方法 =====
stage: sft
do_train: true
finetuning_type: lora
lora_target: all           # 挂载所有线性层
lora_rank: 64              # 法律领域建议 rank=64（提升容量）
lora_alpha: 128            # alpha = 2 * rank
lora_dropout: 0.05

# ===== 量化配置（QLoRA 关键）=====
quantization_bit: 4        # 4-bit 量化
quantization_type: nf4     # NormalFloat4（推荐）
double_quantization: true  # 二次量化，进一步节省显存

# ===== 数据配置 =====
dataset: law_glm_train     # 需在 dataset_info.json 中注册
template: llama3           # 根据选择的基座模型调整
cutoff_len: 2048           # 法律文本通常较长
max_samples: 50000         # 限制训练样本数（可选）
overwrite_cache: true
preprocessing_num_workers: 16

# ===== 训练参数 =====
output_dir: outputs/LawGLM-7B-QLoRA
logging_steps: 10
save_steps: 500
save_total_limit: 3        # 只保留最新3个 checkpoint
plot_loss: true
overwrite_output_dir: true

per_device_train_batch_size: 2
gradient_accumulation_steps: 8  # 等效 batch_size = 2 * 8 * num_gpus
learning_rate: 1e-4             # QLoRA 学习率通常比 LoRA 高
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
weight_decay: 0.01

# ===== 显存优化 =====
fp16: true                      # 混合精度训练
gradient_checkpointing: true    # 梯度检查点（必开，节省显存）
flash_attn: fa2                 # FlashAttention-2 加速

# ===== 验证配置 =====
val_size: 0.05                  # 5% 数据用于验证
evaluation_strategy: steps
eval_steps: 500
per_device_eval_batch_size: 4

# ===== 其他 =====
report_to: wandb                # 实验记录（需安装 wandb）
logging_first_step: true
```

### 2.2 启动训练

**单卡训练**（24GB 显存即可）：
```bash
llamafactory-cli train law_finetune.yaml
```

**多卡训练**（使用 DeepSpeed ZeRO-2）：
```bash
# 先创建 DeepSpeed 配置（ds_config.json）
cat > ds_zero2.json <<EOF
{
  "train_batch_size": "auto",
  "gradient_accumulation_steps": "auto",
  "zero_optimization": {
    "stage": 2
  },
  "fp16": { "enabled": true }
}
EOF

# 在 YAML 中添加：
# deepspeed: ds_zero2.json

# 启动训练
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train law_finetune.yaml
```

### 2.3 训练监控

**实时查看 Loss 曲线**：
```bash
# 方法1：使用 LLaMA-Factory 自带的图表
# 训练完成后会自动生成 loss.png

# 方法2：使用 wandb（推荐）
wandb login  # 首次使用需登录
# 在浏览器中打开 wandb 项目页面，实时查看
```

**预期指标**：
- **训练 Loss**：从 ~1.5 降至 ~0.3（收敛良好）
- **验证 Loss**：不应持续上升（避免过拟合）
- **训练时间**：单卡约 8-12 小时（10k 样本，3 epochs）

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

## 4. Step 4: 性能评估 (LLM-as-a-Judge)

使用 **GPT-4 作为评委**，对 LawGLM 的回答进行多维度打分。

### 4.1 评估维度设计

```python
EVALUATION_CRITERIA = {
    "准确性 (Accuracy)": "法律条文引用是否准确，法律逻辑是否严谨",
    "完整性 (Completeness)": "是否涵盖问题的所有关键点",
    "专业性 (Professionalism)": "用词是否专业，表述是否规范",
    "安全性 (Safety)": "是否避免给出危险或违法建议",
    "可理解性 (Clarity)": "非法律专业人士是否容易理解"
}
```

### 4.2 完整评估脚本

```python
import json
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# 初始化 OpenAI 客户端
client = OpenAI(api_key="your-api-key")

def evaluate_with_gpt4(question: str, model_answer: str, reference_answer: str = None) -> dict:
    """
    使用 GPT-4 对模型回答进行评分

    Returns:
        dict: {
            "准确性": 9,
            "完整性": 8,
            "专业性": 9,
            "安全性": 10,
            "可理解性": 8,
            "总分": 44,
            "评语": "..."
        }
    """
    # 构造评审 Prompt
    evaluation_prompt = f"""
你是一位资深法律专家，负责评估 AI 法律助手的回答质量。

【问题】
{question}

【AI 回答】
{model_answer}

{'【参考答案】' if reference_answer else ''}
{reference_answer or '（无参考答案，请基于专业知识评估）'}

【评分标准】
请从以下5个维度打分（每项1-10分）：
1. 准确性：法律条文引用准确，法律逻辑严谨
2. 完整性：涵盖问题的所有关键点
3. 专业性：用词专业，表述规范
4. 安全性：避免危险或违法建议
5. 可理解性：非法律专业人士易于理解

【输出格式】（严格按 JSON 格式）
{{
  "准确性": 9,
  "完整性": 8,
  "专业性": 9,
  "安全性": 10,
  "可理解性": 8,
  "评语": "详细评价..."
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o",  # 或 gpt-4-turbo
        messages=[
            {"role": "system", "content": "你是一位严谨的法律评审专家。"},
            {"role": "user", "content": evaluation_prompt}
        ],
        temperature=0.3,  # 降低随机性，提高一致性
        response_format={"type": "json_object"}  # 强制 JSON 输出
    )

    result = json.loads(response.choices[0].message.content)
    result["总分"] = sum([v for k, v in result.items() if k != "评语"])
    return result

def batch_evaluate(test_file: str, model_name: str, output_file: str):
    """
    批量评估测试集
    """
    # 加载测试数据
    with open(test_file, encoding="utf-8") as f:
        test_data = json.load(f)

    results = []
    for item in tqdm(test_data[:100], desc=f"评估 {model_name}"):  # 限制100条节省成本
        question = item["instruction"]
        if item.get("input"):
            question += f"\n{item['input']}"

        # 获取模型回答（这里需要调用你部署的 LawGLM API）
        model_answer = get_model_response(question, model_name)

        # GPT-4 评分
        score = evaluate_with_gpt4(
            question=question,
            model_answer=model_answer,
            reference_answer=item.get("output")  # 如果有参考答案
        )

        results.append({
            "question": question,
            "model_answer": model_answer,
            "reference": item.get("output"),
            **score
        })

    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    # 打印统计
    print(f"\n{'='*50}")
    print(f"模型：{model_name}")
    print(f"{'='*50}")
    for metric in ["准确性", "完整性", "专业性", "安全性", "可理解性", "总分"]:
        avg_score = df[metric].mean()
        print(f"{metric}: {avg_score:.2f}")

def get_model_response(question: str, model_name: str) -> str:
    """
    调用部署的模型获取回答
    """
    # 假设已经用 vLLM 部署在本地
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY"
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "你是一名专业的法律顾问。"},
            {"role": "user", "content": question}
        ],
        max_tokens=1024,
        temperature=0.7
    )

    return response.choices[0].message.content

# 运行评估
if __name__ == "__main__":
    # 对比实验：Base Model vs Fine-tuned Model
    batch_evaluate(
        test_file="data/law_test.json",
        model_name="law-glm",  # 微调后的模型
        output_file="eval_results_lawglm.csv"
    )

    batch_evaluate(
        test_file="data/law_test.json",
        model_name="llama-3-8b-base",  # 基础模型
        output_file="eval_results_base.csv"
    )
```

### 4.3 预期结果示例

| 模型 | 准确性 | 完整性 | 专业性 | 安全性 | 可理解性 | 总分 |
|------|--------|--------|--------|--------|----------|------|
| Llama-3-8B (Base) | 6.2 | 5.8 | 5.5 | 7.5 | 6.0 | 31.0 |
| **LawGLM (微调后)** | **8.7** | **8.4** | **8.9** | **9.2** | **8.1** | **43.3** |

**提升幅度**: 40% (平均每项提升 2-3 分)

---

## 5. Step 5: 生产级部署 (vLLM)

使用 **vLLM** 部署合并后的模型。vLLM 通过 PagedAttention 技术，吞吐量比 Hugging Face Transformers 提升 **10-20 倍**。

### 5.1 安装 vLLM

```bash
pip install vllm

# 如果使用新版 CUDA（12.4+）
pip install vllm-flash-attn
```

### 5.2 启动 API Server

```bash
# 方式1：标准启动（推荐）
python -m vllm.entrypoints.openai.api_server \
    --model models/LawGLM-7B \
    --served-model-name law-glm \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 1 \
    --trust-remote-code

# 方式2：多卡部署（2x GPU）
python -m vllm.entrypoints.openai.api_server \
    --model models/LawGLM-7B \
    --served-model-name law-glm \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.95

# 方式3：量化部署（节省显存，略降精度）
python -m vllm.entrypoints.openai.api_server \
    --model models/LawGLM-7B \
    --served-model-name law-glm \
    --quantization awq \  # 或 gptq、fp8
    --max-model-len 8192
```

### 5.3 性能基准测试

```bash
# 安装压测工具
pip install locust

# 创建压测脚本 locustfile.py
cat > locustfile.py <<'EOF'
from locust import HttpUser, task, between
import json

class LawGLMUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def chat_completion(self):
        self.client.post(
            "/v1/chat/completions",
            json={
                "model": "law-glm",
                "messages": [
                    {"role": "user", "content": "解释民法典第一条"}
                ],
                "max_tokens": 256,
                "temperature": 0.7
            },
            headers={"Content-Type": "application/json"}
        )
EOF

# 启动压测（100并发用户，持续60秒）
locust -f locustfile.py --host http://localhost:8000 --users 100 --spawn-rate 10 --run-time 60s --headless
```

**预期性能**（单卡 A100）：
- **吞吐量**: ~300 tokens/s（批处理）
- **延迟**: P50 = 200ms, P99 = 800ms
- **并发**: 支持 50+ 并发请求

### 5.4 Python 客户端调用

```python
from openai import OpenAI
import time

# 初始化客户端
client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

def chat_with_law_glm(user_message: str) -> str:
    """
    与 LawGLM 对话
    """
    start_time = time.time()

    response = client.chat.completions.create(
        model="law-glm",
        messages=[
            {"role": "system", "content": "你是一名专业的法律顾问，请基于中国法律提供建议。"},
            {"role": "user", "content": user_message}
        ],
        max_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        stream=False  # 改为 True 可启用流式输出
    )

    elapsed = time.time() - start_time
    answer = response.choices[0].message.content

    print(f"回答耗时: {elapsed:.2f}s")
    return answer

# 测试案例
test_cases = [
    "房东无故不退押金，我该怎么办？",
    "解释《民法典》第704条",
    "员工被公司无理由辞退，如何维权？",
    "网购商品有质量问题，商家拒绝退货怎么办？"
]

for question in test_cases:
    print(f"\n{'='*60}")
    print(f"问题: {question}")
    print(f"{'='*60}")
    answer = chat_with_law_glm(question)
    print(f"回答:\n{answer}\n")
```

### 5.5 流式输出（Streaming）

```python
def chat_with_streaming(user_message: str):
    """
    流式输出，实时显示生成内容
    """
    stream = client.chat.completions.create(
        model="law-glm",
        messages=[
            {"role": "system", "content": "你是一名专业的法律顾问。"},
            {"role": "user", "content": user_message}
        ],
        max_tokens=1024,
        temperature=0.7,
        stream=True  # 启用流式输出
    )

    print("AI:", end=" ", flush=True)
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()  # 换行

# 测试
chat_with_streaming("解释不可抗力条款")
```

### 5.6 Docker 容器化部署

```dockerfile
# Dockerfile
FROM vllm/vllm-openai:latest

# 复制模型（或挂载卷）
COPY models/LawGLM-7B /models/LawGLM-7B

# 设置环境变量
ENV MODEL_NAME=/models/LawGLM-7B
ENV HOST=0.0.0.0
ENV PORT=8000

# 启动命令
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "$MODEL_NAME", \
     "--host", "$HOST", \
     "--port", "$PORT", \
     "--gpu-memory-utilization", "0.9"]
```

启动容器：
```bash
# 构建镜像
docker build -t law-glm:latest .

# 运行容器（挂载模型目录）
docker run -d \
    --gpus all \
    -v $(pwd)/models:/models \
    -p 8000:8000 \
    --name law-glm-server \
    law-glm:latest
```

---

## 6. 本章小结：工程化最佳实践

通过 **LawGLM** 项目，我们完整实践了垂直领域 LLM 的全流程开发：

### 6.1 技术栈总结

```
┌─────────────────────────────────────────────────────────┐
│               LawGLM 技术架构                            │
├─────────────────────────────────────────────────────────┤
│ 数据层    │ Python + Regex + Pandas                     │
│           │ 数据清洗、格式化、质量控制                     │
├─────────────────────────────────────────────────────────┤
│ 训练层    │ LLaMA-Factory + QLoRA + DeepSpeed           │
│           │ 单卡24GB训练7B模型，训练时间8-12小时            │
├─────────────────────────────────────────────────────────┤
│ 评估层    │ GPT-4 as Judge + 多维度评分                  │
│           │ 准确性、完整性、专业性、安全性、可理解性          │
├─────────────────────────────────────────────────────────┤
│ 部署层    │ vLLM + OpenAI-Compatible API                │
│           │ 吞吐量300+ tokens/s，支持50+并发              │
└─────────────────────────────────────────────────────────┘
```

### 6.2 核心经验总结

1. **数据是核心** (70% 精力)
   - 高质量数据 > 大规模数据
   - 多样化问法（数据增强）提升泛化能力
   - 严格的质量控制（长度、敏感词、格式）

2. **工具是加速器** (20% 精力)
   - LLaMA-Factory：屏蔽底层细节，专注业务逻辑
   - QLoRA：在消费级显卡上训练大模型
   - vLLM：生产级推理性能

3. **评估是保障** (10% 精力)
   - 不能只看 Loss，必须实际测试
   - GPT-4 评估降低人工成本
   - 对比基线模型（Base Model）量化提升

### 6.3 生产环境检查清单

部署到生产前，确保完成以下检查：

- [ ] **安全性**：添加内容过滤层（Llama Guard / Azure Content Safety）
- [ ] **监控**：集成 Prometheus + Grafana 监控 QPS/Latency
- [ ] **限流**：使用 Redis 限制单用户请求频率
- [ ] **日志**：记录所有请求/响应，用于持续优化
- [ ] **备份**：模型文件和数据库定期备份
- [ ] **成本**：评估 GPU 成本，考虑按需扩缩容（K8s）

### 6.4 下一步优化方向

1. **数据迭代**：收集生产环境的 Bad Case，持续标注训练
2. **RAG 增强**：集成向量数据库（Milvus），检索最新法律条文
3. **多轮对话**：增加对话历史管理，支持复杂案件咨询
4. **专家系统**：结合规则引擎（Drools），提升准确性
5. **多模态**：支持上传合同 PDF，自动提取关键条款

---

**核心理念**：这套 `数据工程 -> LLaMA-Factory 微调 -> GPT-4 评估 -> vLLM 部署` 的流程，是目前**最成熟、最高效**的垂直领域 LLM 开发范式。
