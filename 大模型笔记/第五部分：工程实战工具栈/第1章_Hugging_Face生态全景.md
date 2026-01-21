# 第1章：Hugging Face生态全景

> 掌握最流行的LLM开发生态系统。

## 本章导读

Hugging Face已成为LLM开发的事实标准生态。无论是模型加载、数据处理，还是训练微调，Hugging Face提供了一整套开箱即用的工具链。本章将系统介绍：

**核心内容**：
- Transformers库（模型加载、Pipeline、自定义）
- Datasets库（数据加载、预处理、流式处理）
- Trainer API（训练循环、回调函数、日志）
- Hub生态（模型分享、版本管理、Spaces部署）
- PEFT与TRL库（高效微调、强化学习）

**学习目标**：
- 掌握Transformers核心API
- 能够高效处理大规模数据集
- 使用Trainer实现训练流程
- 发布模型到Hugging Face Hub
- 应用PEFT进行参数高效微调

---

## 一、Transformers库核心用法

### 1. 模型加载与配置

#### （1）基础加载流程

```python
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoConfig
)
import torch

class ModelLoader:
    """模型加载器（最佳实践）"""
    
    @staticmethod
    def load_model_for_inference(
        model_name: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        use_flash_attention: bool = True
    ):
        """加载模型用于推理
        
        Args:
            model_name: 模型名称（如"meta-llama/Llama-3-8B"）
            device: 设备
            torch_dtype: 数据类型（float16节省显存）
            use_flash_attention: 使用Flash Attention 2
        """
        print(f"正在加载模型: {model_name}")
        
        # 1. 加载配置
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True  # 信任自定义代码（如Qwen）
        )
        
        # 2. 启用Flash Attention 2（如果支持）
        if use_flash_attention and hasattr(config, "_attn_implementation"):
            config._attn_implementation = "flash_attention_2"
        
        # 3. 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True  # 使用Rust实现的快速分词器
        )
        
        # 设置padding token（某些模型缺失）
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 4. 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch_dtype,
            device_map="auto",  # 自动分配设备
            trust_remote_code=True
        )
        
        # 5. 切换到评估模式
        model.eval()
        
        print(f"✅ 模型已加载到 {device}")
        print(f"   参数量: {model.num_parameters() / 1e9:.2f}B")
        print(f"   数据类型: {torch_dtype}")
        
        return model, tokenizer

# 使用示例
model, tokenizer = ModelLoader.load_model_for_inference(
    model_name="meta-llama/Llama-3-8B-Instruct",
    torch_dtype=torch.bfloat16,
    use_flash_attention=True
)

# 快速推理测试
prompt = "What is deep learning?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"回复: {response}")
```

**输出示例**：
```
正在加载模型: meta-llama/Llama-3-8B-Instruct
✅ 模型已加载到 cuda
   参数量: 8.03B
   数据类型: torch.bfloat16

回复: What is deep learning? Deep learning is a subset of machine learning...
```

#### （2）量化加载（节省显存）

```python
from transformers import BitsAndBytesConfig

class QuantizedModelLoader:
    """量化模型加载器"""
    
    @staticmethod
    def load_4bit_model(model_name: str):
        """4-bit量化加载（NF4）
        
        显存占用: ~0.5GB/B参数
        例: 70B模型仅需35GB（vs 140GB FP16）
        """
        # BitsAndBytes配置
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # NormalFloat4
            bnb_4bit_use_double_quant=True,  # 双重量化
            bnb_4bit_compute_dtype=torch.bfloat16  # 计算类型
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        return model, tokenizer
    
    @staticmethod
    def load_8bit_model(model_name: str):
        """8-bit量化加载（LLM.int8()）
        
        显存占用: ~1GB/B参数
        """
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0  # 异常值阈值
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        return model, tokenizer

# 使用示例：在单张A100 40GB上加载70B模型
model_70b, tokenizer_70b = QuantizedModelLoader.load_4bit_model(
    "meta-llama/Llama-3-70B-Instruct"
)

print(f"70B模型显存占用: ~{70 * 0.5:.0f}GB（4-bit）")
```

#### （3）多GPU并行加载

```python
from accelerate import infer_auto_device_map, dispatch_model
import torch

class MultiGPULoader:
    """多GPU加载器"""
    
    @staticmethod
    def load_with_device_map(
        model_name: str,
        num_gpus: int = 2,
        max_memory_per_gpu: str = "40GiB"
    ):
        """自定义设备映射
        
        Args:
            model_name: 模型名称
            num_gpus: GPU数量
            max_memory_per_gpu: 每张GPU最大内存
        """
        # 1. 加载配置（不加载权重）
        config = AutoConfig.from_pretrained(model_name)
        
        # 2. 创建空模型
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config)
        
        # 3. 推断设备映射
        max_memory = {i: max_memory_per_gpu for i in range(num_gpus)}
        max_memory["cpu"] = "100GiB"  # CPU用于溢出
        
        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=["LlamaDecoderLayer"]  # 不拆分的模块
        )
        
        # 4. 加载权重并分发
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.float16
        )
        
        # 打印设备映射
        print("设备映射:")
        for name, device in model.hf_device_map.items():
            print(f"  {name:40s} -> {device}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        return model, tokenizer

# 使用示例：将70B模型分配到2张A100
model_multi_gpu, tokenizer = MultiGPULoader.load_with_device_map(
    model_name="meta-llama/Llama-3-70B-Instruct",
    num_gpus=2,
    max_memory_per_gpu="40GiB"
)
```

**输出示例**：
```
设备映射:
  model.embed_tokens                       -> cuda:0
  model.layers.0                           -> cuda:0
  model.layers.1                           -> cuda:0
  ...
  model.layers.40                          -> cuda:0
  model.layers.41                          -> cuda:1
  ...
  model.layers.79                          -> cuda:1
  model.norm                               -> cuda:1
  lm_head                                  -> cuda:1
```

---

### 2. Pipeline快速上手

#### （1）内置Pipeline

```python
from transformers import pipeline
import time

class PipelineExamples:
    """Pipeline使用示例"""
    
    @staticmethod
    def text_generation_example():
        """文本生成Pipeline"""
        # 自动下载模型并缓存
        generator = pipeline(
            "text-generation",
            model="gpt2",
            device=0  # 使用GPU 0
        )
        
        # 生成文本
        outputs = generator(
            "Once upon a time",
            max_length=50,
            num_return_sequences=3,  # 生成3个版本
            temperature=0.8
        )
        
        print("文本生成示例:")
        for i, output in enumerate(outputs, 1):
            print(f"\n版本{i}: {output['generated_text']}")
    
    @staticmethod
    def fill_mask_example():
        """填空Pipeline（BERT类模型）"""
        unmasker = pipeline("fill-mask", model="bert-base-uncased")
        
        outputs = unmasker("The capital of France is [MASK].")
        
        print("\n填空示例:")
        for output in outputs[:3]:
            print(f"  {output['token_str']:10s} (置信度: {output['score']:.3f})")
    
    @staticmethod
    def text_classification_example():
        """文本分类Pipeline"""
        classifier = pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        outputs = classifier([
            "I love this product!",
            "This is terrible."
        ])
        
        print("\n情感分类:")
        for text, output in zip(["正面评论", "负面评论"], outputs):
            print(f"  {text}: {output['label']} (分数: {output['score']:.3f})")
    
    @staticmethod
    def question_answering_example():
        """问答Pipeline"""
        qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
        
        context = """
        The Transformer is a deep learning model introduced in 2017, used primarily 
        in the field of natural language processing (NLP). It was proposed in the 
        paper "Attention Is All You Need" by Vaswani et al.
        """
        
        question = "When was the Transformer introduced?"
        
        answer = qa_pipeline(question=question, context=context)
        
        print("\n问答示例:")
        print(f"  问题: {question}")
        print(f"  答案: {answer['answer']} (置信度: {answer['score']:.3f})")

# 运行示例
examples = PipelineExamples()
examples.text_generation_example()
examples.fill_mask_example()
examples.text_classification_example()
examples.question_answering_example()
```

**输出示例**：
```
文本生成示例:

版本1: Once upon a time, there was a young girl who lived in a small village...
版本2: Once upon a time in a galaxy far, far away...
版本3: Once upon a time, people believed that the earth was flat...

填空示例:
  paris      (置信度: 0.952)
  france     (置信度: 0.018)
  london     (置信度: 0.012)

情感分类:
  正面评论: POSITIVE (分数: 0.999)
  负面评论: NEGATIVE (分数: 0.998)

问答示例:
  问题: When was the Transformer introduced?
  答案: 2017 (置信度: 0.987)
```

#### （2）自定义Pipeline

```python
from transformers import Pipeline
from typing import Dict, List

class CustomSummarizationPipeline(Pipeline):
    """自定义摘要Pipeline"""
    
    def _sanitize_parameters(self, **kwargs):
        """参数预处理"""
        preprocess_kwargs = {}
        forward_kwargs = {}
        postprocess_kwargs = {}
        
        if "max_length" in kwargs:
            preprocess_kwargs["max_length"] = kwargs["max_length"]
        if "min_length" in kwargs:
            forward_kwargs["min_length"] = kwargs["min_length"]
        if "summary_length" in kwargs:
            forward_kwargs["max_new_tokens"] = kwargs["summary_length"]
        
        return preprocess_kwargs, forward_kwargs, postprocess_kwargs
    
    def preprocess(self, text: str, max_length: int = 1024):
        """预处理：分词"""
        inputs = self.tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )
        return inputs
    
    def _forward(self, model_inputs, min_length: int = 50, max_new_tokens: int = 150):
        """前向传播：生成摘要"""
        outputs = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            min_length=min_length,
            do_sample=False,  # 使用beam search
            num_beams=4
        )
        return outputs
    
    def postprocess(self, model_outputs):
        """后处理：解码"""
        summary = self.tokenizer.decode(
            model_outputs[0],
            skip_special_tokens=True
        )
        return {"summary_text": summary}

# 注册自定义Pipeline
from transformers import AutoModelForSeq2SeqLM
from transformers import pipeline as transformers_pipeline

# 使用自定义Pipeline
summarizer = CustomSummarizationPipeline(
    model=AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn"),
    tokenizer=AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
)

article = """
Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to 
the natural intelligence displayed by humans and animals. Leading AI textbooks define 
the field as the study of "intelligent agents": any device that perceives its environment 
and takes actions that maximize its chance of successfully achieving its goals.
"""

summary = summarizer(article, summary_length=50)
print(f"原文长度: {len(article)} 字符")
print(f"摘要: {summary['summary_text']}")
```

---

### 3. 自定义模型与分词器

#### （1）扩展词表

```python
class TokenizerExtender:
    """分词器扩展器"""
    
    @staticmethod
    def add_special_tokens(
        tokenizer: AutoTokenizer,
        new_tokens: List[str]
    ):
        """添加特殊Token
        
        用例:
        - 添加新的特殊符号（如<image>、<tool_call>）
        - 添加领域术语（提升分词效率）
        """
        num_added = tokenizer.add_special_tokens({
            "additional_special_tokens": new_tokens
        })
        
        print(f"添加了 {num_added} 个特殊token")
        print(f"新词表大小: {len(tokenizer)}")
        
        return num_added
    
    @staticmethod
    def resize_model_embeddings(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer
    ):
        """调整模型嵌入层大小
        
        注意: 添加token后必须调整模型
        """
        model.resize_token_embeddings(len(tokenizer))
        
        # 新token的嵌入会随机初始化，需要微调
        print(f"模型嵌入层已调整为 {len(tokenizer)}")

# 使用示例
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 添加新token
new_tokens = ["<image>", "<video>", "<audio>"]
TokenizerExtender.add_special_tokens(tokenizer, new_tokens)

# 调整模型
TokenizerExtender.resize_model_embeddings(model, tokenizer)

# 测试新token
text = "This is an image: <image>"
tokens = tokenizer(text, return_tensors="pt")
print(f"Token IDs: {tokens['input_ids']}")
```

#### （2）自定义模型配置

```python
from transformers import PretrainedConfig, PreTrainedModel
import torch.nn as nn

class CustomLMConfig(PretrainedConfig):
    """自定义语言模型配置"""
    model_type = "custom_lm"
    
    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 1024,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings

class CustomLMModel(PreTrainedModel):
    """自定义语言模型（简化版）"""
    config_class = CustomLMConfig
    
    def __init__(self, config: CustomLMConfig):
        super().__init__(config)
        
        # 嵌入层
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer层（简化）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_hidden_layers
        )
        
        # 输出层
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        
        # 初始化权重
        self.post_init()
    
    def forward(self, input_ids, attention_mask=None):
        """前向传播"""
        # 嵌入
        hidden_states = self.embeddings(input_ids)
        
        # Transformer
        hidden_states = self.transformer(
            hidden_states,
            src_key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
        )
        
        # 输出
        logits = self.lm_head(hidden_states)
        
        return {"logits": logits}

# 使用自定义模型
config = CustomLMConfig(
    vocab_size=50257,
    hidden_size=512,
    num_hidden_layers=6
)

custom_model = CustomLMModel(config)

print(f"自定义模型参数量: {custom_model.num_parameters() / 1e6:.2f}M")

# 保存模型
custom_model.save_pretrained("./custom_lm")

# 加载模型
loaded_model = CustomLMModel.from_pretrained("./custom_lm")
```

---

## 二、Datasets与数据处理

### 1. 数据集加载与预处理

#### （1）加载Hugging Face Hub数据集

```python
from datasets import load_dataset, DatasetDict
from typing import Optional

class DatasetLoader:
    """数据集加载器"""
    
    @staticmethod
    def load_text_dataset(
        dataset_name: str,
        split: str = "train",
        streaming: bool = False,
        num_samples: Optional[int] = None
    ):
        """加载文本数据集
        
        Args:
            dataset_name: 数据集名称
            split: 数据集切分（train/validation/test）
            streaming: 流式加载（不加载到内存）
            num_samples: 限制样本数（用于快速测试）
        """
        print(f"正在加载数据集: {dataset_name} ({split})")
        
        dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=streaming
        )
        
        # 限制样本数
        if num_samples and not streaming:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        print(f"✅ 数据集已加载")
        if not streaming:
            print(f"   样本数: {len(dataset)}")
            print(f"   列名: {dataset.column_names}")
        
        return dataset

# 使用示例
# 1. 标准加载
dataset = DatasetLoader.load_text_dataset(
    "c4",
    split="train",
    num_samples=10000
)

# 2. 流式加载（处理TB级数据）
dataset_stream = DatasetLoader.load_text_dataset(
    "c4",
    split="train",
    streaming=True
)

# 流式迭代
for i, example in enumerate(dataset_stream):
    if i >= 5:
        break
    print(f"样本{i}: {example['text'][:100]}...")
```

#### （2）数据预处理

```python
from datasets import Dataset
from transformers import PreTrainedTokenizer
from typing import Callable

class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
    
    def tokenize_function(self, examples: Dict) -> Dict:
        """基础分词"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length"
        )
    
    def tokenize_qa_pairs(self, examples: Dict) -> Dict:
        """问答对分词"""
        # 合并问题和答案
        inputs = [
            f"Question: {q}\nAnswer: {a}"
            for q, a in zip(examples["question"], examples["answer"])
        ]
        
        return self.tokenizer(
            inputs,
            truncation=True,
            max_length=512,
            padding="max_length"
        )
    
    def process_dataset(
        self,
        dataset: Dataset,
        processing_fn: Optional[Callable] = None,
        batched: bool = True,
        num_proc: int = 4
    ) -> Dataset:
        """处理数据集
        
        Args:
            dataset: 输入数据集
            processing_fn: 处理函数（默认使用tokenize_function）
            batched: 批量处理（提速10-100x）
            num_proc: 并行进程数
        """
        if processing_fn is None:
            processing_fn = self.tokenize_function
        
        print(f"正在处理数据集（批量={batched}, 进程数={num_proc}）...")
        
        processed_dataset = dataset.map(
            processing_fn,
            batched=batched,
            num_proc=num_proc,
            remove_columns=dataset.column_names,  # 移除原始列
            desc="Tokenizing"
        )
        
        print(f"✅ 处理完成")
        print(f"   处理后列名: {processed_dataset.column_names}")
        
        return processed_dataset

# 使用示例
tokenizer = AutoTokenizer.from_pretrained("gpt2")
preprocessor = DataPreprocessor(tokenizer)

# 创建示例数据集
from datasets import Dataset

raw_data = {
    "text": [
        "This is the first example.",
        "This is the second example.",
        "And this is the third one."
    ]
}
dataset = Dataset.from_dict(raw_data)

# 处理数据集
tokenized_dataset = preprocessor.process_dataset(
    dataset,
    batched=True,
    num_proc=2
)

print(tokenized_dataset[0])
```

**输出示例**：
```
正在处理数据集（批量=True, 进程数=2）...
Tokenizing: 100%|██████████| 3/3 [00:00<00:00, 145.23 examples/s]
✅ 处理完成
   处理后列名: ['input_ids', 'attention_mask']

{'input_ids': [1212, 318, 262, 717, 1672, 13, ...], 'attention_mask': [1, 1, 1, 1, 1, 1, ...]}
```


#### （3）过滤与采样

```python
class DatasetFilter:
    """数据集过滤器"""
    
    @staticmethod
    def filter_by_length(
        dataset: Dataset,
        min_length: int = 10,
        max_length: int = 1000,
        text_column: str = "text"
    ) -> Dataset:
        """按文本长度过滤"""
        def length_filter(example):
            text_len = len(example[text_column])
            return min_length <= text_len <= max_length
        
        filtered = dataset.filter(length_filter, desc="按长度过滤")
        
        print(f"过滤前: {len(dataset)} 样本")
        print(f"过滤后: {len(filtered)} 样本")
        
        return filtered
    
    @staticmethod
    def filter_by_custom(
        dataset: Dataset,
        filter_fn: Callable
    ) -> Dataset:
        """自定义过滤"""
        return dataset.filter(filter_fn, desc="自定义过滤")
    
    @staticmethod
    def random_sample(
        dataset: Dataset,
        num_samples: int,
        seed: int = 42
    ) -> Dataset:
        """随机采样"""
        shuffled = dataset.shuffle(seed=seed)
        return shuffled.select(range(min(num_samples, len(dataset))))

# 使用示例
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("imdb", split="train")

# 过滤短文本
filtered_dataset = DatasetFilter.filter_by_length(
    dataset,
    min_length=100,
    max_length=5000,
    text_column="text"
)

# 采样1000条
sampled_dataset = DatasetFilter.random_sample(filtered_dataset, 1000)
```

---

### 2. 数据映射与批处理

#### （1）高级映射操作

```python
class AdvancedDataMapper:
    """高级数据映射器"""
    
    @staticmethod
    def create_instruction_dataset(dataset: Dataset) -> Dataset:
        """创建指令微调格式数据集"""
        def format_instruction(example):
            # 格式: Instruction -> Response
            instruction = f"""Below is an instruction. Write a response.

### Instruction:
{example['input']}

### Response:
{example['output']}"""
            
            return {"text": instruction}
        
        return dataset.map(
            format_instruction,
            remove_columns=dataset.column_names,
            desc="格式化指令"
        )
    
    @staticmethod
    def add_length_column(dataset: Dataset) -> Dataset:
        """添加长度列"""
        def add_length(example):
            example["text_length"] = len(example["text"])
            return example
        
        return dataset.map(add_length, desc="添加长度")
    
    @staticmethod
    def batch_processing(dataset: Dataset, tokenizer) -> Dataset:
        """批量处理（动态padding）"""
        def batch_tokenize(examples):
            # 批量分词（自动padding到批次最大长度）
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                padding=False  # 训练时动态padding
            )
        
        return dataset.map(
            batch_tokenize,
            batched=True,
            batch_size=1000,
            remove_columns=["text"],
            desc="批量分词"
        )

# 使用示例
# 创建示例数据
raw_data = {
    "input": ["Translate to French: Hello", "Summarize: Long text..."],
    "output": ["Bonjour", "Summary..."]
}
dataset = Dataset.from_dict(raw_data)

# 格式化为指令格式
mapper = AdvancedDataMapper()
instruction_dataset = mapper.create_instruction_dataset(dataset)

print(instruction_dataset[0]["text"])
```

**输出示例**：
```
Below is an instruction. Write a response.

### Instruction:
Translate to French: Hello

### Response:
Bonjour
```

---

### 3. 大规模数据流式处理

#### （1）流式迭代

```python
from datasets import load_dataset
from itertools import islice

class StreamingDataHandler:
    """流式数据处理器"""
    
    @staticmethod
    def stream_and_process(
        dataset_name: str,
        processing_fn: Callable,
        batch_size: int = 1000,
        max_batches: Optional[int] = None
    ):
        """流式处理大数据集
        
        Args:
            dataset_name: 数据集名称
            processing_fn: 处理函数
            batch_size: 批次大小
            max_batches: 最大批次数（None表示全部）
        """
        # 流式加载
        dataset = load_dataset(dataset_name, split="train", streaming=True)
        
        batch = []
        num_batches = 0
        
        for example in dataset:
            batch.append(example)
            
            # 达到批次大小
            if len(batch) >= batch_size:
                # 处理批次
                processing_fn(batch)
                
                num_batches += 1
                batch = []
                
                # 达到最大批次数
                if max_batches and num_batches >= max_batches:
                    break
        
        # 处理剩余样本
        if batch:
            processing_fn(batch)
    
    @staticmethod
    def interleave_datasets(dataset_names: List[str], probabilities: List[float]):
        """交叉加载多个数据集
        
        Args:
            dataset_names: 数据集名称列表
            probabilities: 采样概率
        """
        from datasets import interleave_datasets
        
        # 加载多个数据集（流式）
        datasets = [
            load_dataset(name, split="train", streaming=True)
            for name in dataset_names
        ]
        
        # 交叉采样
        interleaved = interleave_datasets(
            datasets,
            probabilities=probabilities,
            seed=42
        )
        
        return interleaved

# 使用示例
def process_batch(batch):
    """批次处理函数"""
    print(f"处理批次，大小: {len(batch)}")
    # 这里可以进行分词、保存等操作

# 流式处理C4数据集（314GB）
StreamingDataHandler.stream_and_process(
    dataset_name="c4",
    processing_fn=process_batch,
    batch_size=1000,
    max_batches=10  # 仅处理前10个批次
)

# 交叉多个数据集（常用于预训练）
interleaved = StreamingDataHandler.interleave_datasets(
    dataset_names=["c4", "wikipedia", "bookcorpus"],
    probabilities=[0.5, 0.3, 0.2]  # C4占50%，Wikipedia占30%，BookCorpus占20%
)

# 迭代交叉数据
for i, example in enumerate(islice(interleaved, 5)):
    print(f"样本{i}: {example['text'][:50]}...")
```

---

## 三、Trainer与训练流程

### 1. TrainingArguments配置详解

```python
from transformers import TrainingArguments
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """训练配置（最佳实践）"""
    
    @staticmethod
    def get_default_args(output_dir: str = "./results") -> TrainingArguments:
        """默认训练参数"""
        return TrainingArguments(
            # === 输出与日志 ===
            output_dir=output_dir,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            save_steps=500,
            save_total_limit=3,  # 仅保留最近3个checkpoint
            
            # === 训练超参数 ===
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=4,  # 等效batch_size=32
            learning_rate=5e-5,
            weight_decay=0.01,
            warmup_steps=500,
            
            # === 评估 ===
            evaluation_strategy="steps",
            eval_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            
            # === 性能优化 ===
            fp16=True,  # 混合精度训练（A100用bf16）
            dataloader_num_workers=4,
            
            # === 其他 ===
            seed=42,
            report_to=["tensorboard"],  # 或"wandb"
        )
    
    @staticmethod
    def get_lora_args(output_dir: str = "./lora_results") -> TrainingArguments:
        """LoRA微调参数"""
        return TrainingArguments(
            output_dir=output_dir,
            
            # LoRA通常使用更高学习率
            learning_rate=1e-4,
            
            # 更小的batch size
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            
            # 更多epoch（LoRA收敛快）
            num_train_epochs=5,
            
            # 节省显存
            fp16=True,
            gradient_checkpointing=True,
            
            # 其他
            logging_steps=50,
            save_steps=200,
            evaluation_strategy="steps",
            eval_steps=200,
        )
    
    @staticmethod
    def get_deepspeed_args(output_dir: str = "./ds_results") -> TrainingArguments:
        """DeepSpeed分布式训练参数"""
        return TrainingArguments(
            output_dir=output_dir,
            
            # DeepSpeed配置
            deepspeed="ds_config.json",  # DeepSpeed配置文件
            
            # 大batch训练
            per_device_train_batch_size=16,
            gradient_accumulation_steps=2,
            
            # 学习率
            learning_rate=2e-5,
            warmup_ratio=0.1,
            
            # 保存策略
            save_strategy="epoch",
            evaluation_strategy="epoch",
            
            # 日志
            logging_steps=10,
            report_to=["wandb"],
        )

# 使用示例
args = TrainingConfig.get_default_args("./my_model_output")
print(f"有效batch大小: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
```

---

### 2. 自定义训练循环

```python
from transformers import Trainer, TrainerCallback
from torch.utils.data import Dataset
import torch

class CustomTrainer(Trainer):
    """自定义Trainer"""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """自定义损失函数"""
        # 标准前向传播
        outputs = model(**inputs)
        
        # 提取logits和labels
        logits = outputs.get("logits")
        labels = inputs.get("labels")
        
        # 计算交叉熵损失
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        
        # 可以添加额外损失项
        # 例: L2正则化
        # l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
        # loss = loss + 0.01 * l2_reg
        
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """自定义预测步骤"""
        # 可以在这里添加自定义的预测逻辑
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

# 自定义回调
class CustomCallback(TrainerCallback):
    """自定义训练回调"""
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """每个epoch开始时调用"""
        print(f"\n{'='*50}")
        print(f"开始 Epoch {state.epoch}")
        print(f"{'='*50}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """记录日志时调用"""
        if logs:
            # 可以发送到自定义监控系统
            if "loss" in logs:
                print(f"Step {state.global_step}, Loss: {logs['loss']:.4f}")
    
    def on_save(self, args, state, control, **kwargs):
        """保存checkpoint时调用"""
        print(f"✅ Checkpoint已保存: step {state.global_step}")

# 使用示例
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# 1. 准备数据
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names
)

# 2. 加载模型
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 3. 配置训练参数
training_args = TrainingConfig.get_default_args("./gpt2_finetuned")

# 4. 创建Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    callbacks=[CustomCallback()]
)

# 5. 开始训练
# trainer.train()
```

---

### 3. 回调函数与日志

```python
from transformers import TrainerCallback, TrainerState, TrainerControl
import wandb

class WandbCallback(TrainerCallback):
    """Weights & Biases集成"""
    
    def on_init_end(self, args, state, control, **kwargs):
        """初始化wandb"""
        wandb.init(
            project="llm-finetuning",
            name=args.run_name,
            config=args.to_dict()
        )
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """记录到wandb"""
        if logs:
            wandb.log(logs, step=state.global_step)
    
    def on_train_end(self, args, state, control, **kwargs):
        """训练结束"""
        wandb.finish()

class EarlyStoppingCallback(TrainerCallback):
    """早停回调"""
    
    def __init__(self, patience: int = 3, threshold: float = 0.001):
        self.patience = patience
        self.threshold = threshold
        self.best_metric = None
        self.wait = 0
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """评估时检查是否早停"""
        if metrics is None:
            return
        
        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return
        
        if self.best_metric is None:
            self.best_metric = eval_loss
        elif eval_loss < self.best_metric - self.threshold:
            # 性能提升
            self.best_metric = eval_loss
            self.wait = 0
        else:
            # 性能未提升
            self.wait += 1
            print(f"早停计数: {self.wait}/{self.patience}")
            
            if self.wait >= self.patience:
                print(f"⚠️ 触发早停（patience={self.patience}）")
                control.should_training_stop = True

class CheckpointCallback(TrainerCallback):
    """Checkpoint管理回调"""
    
    def on_save(self, args, state, control, **kwargs):
        """保存时触发"""
        checkpoint_path = f"{args.output_dir}/checkpoint-{state.global_step}"
        
        # 可以在这里添加自定义保存逻辑
        # 例如: 上传到云存储
        print(f"💾 保存checkpoint: {checkpoint_path}")
        
        # 保存额外信息
        import json
        metadata = {
            "global_step": state.global_step,
            "epoch": state.epoch,
            "best_metric": state.best_metric
        }
        
        with open(f"{checkpoint_path}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

# 组合使用多个回调
callbacks = [
    EarlyStoppingCallback(patience=3),
    CheckpointCallback(),
    # WandbCallback()  # 需要先安装wandb
]
```

---

## 四、模型分享与部署

### 1. Hub上传与版本管理

```python
from huggingface_hub import HfApi, create_repo, upload_folder
import os

class ModelUploader:
    """模型上传器"""
    
    def __init__(self, token: str):
        """
        Args:
            token: Hugging Face token（从huggingface.co/settings/tokens获取）
        """
        self.api = HfApi(token=token)
        self.token = token
    
    def upload_model(
        self,
        model_path: str,
        repo_name: str,
        private: bool = False,
        commit_message: str = "Upload model"
    ):
        """上传模型到Hub
        
        Args:
            model_path: 本地模型路径
            repo_name: 仓库名（格式: username/model_name）
            private: 是否私有
            commit_message: 提交信息
        """
        # 1. 创建仓库
        try:
            create_repo(
                repo_id=repo_name,
                token=self.token,
                private=private,
                exist_ok=True
            )
            print(f"✅ 仓库创建成功: {repo_name}")
        except Exception as e:
            print(f"⚠️ 仓库已存在或创建失败: {e}")
        
        # 2. 上传文件夹
        upload_folder(
            folder_path=model_path,
            repo_id=repo_name,
            token=self.token,
            commit_message=commit_message
        )
        
        print(f"✅ 模型已上传: https://huggingface.co/{repo_name}")
    
    def upload_with_readme(
        self,
        model_path: str,
        repo_name: str,
        model_card: str
    ):
        """上传模型及README"""
        # 创建README
        readme_path = os.path.join(model_path, "README.md")
        with open(readme_path, "w") as f:
            f.write(model_card)
        
        # 上传
        self.upload_model(model_path, repo_name)

# 使用示例
# uploader = ModelUploader(token="hf_xxx")

# 上传模型
# uploader.upload_model(
#     model_path="./my_finetuned_model",
#     repo_name="username/my-awesome-llm",
#     private=False,
#     commit_message="Initial upload"
# )
```

---

### 2. 模型卡片编写

```python
class ModelCardGenerator:
    """模型卡片生成器"""
    
    @staticmethod
    def generate_card(
        model_name: str,
        base_model: str,
        task: str,
        dataset: str,
        metrics: Dict[str, float],
        usage_example: str
    ) -> str:
        """生成模型卡片
        
        Args:
            model_name: 模型名称
            base_model: 基础模型
            task: 任务类型
            dataset: 训练数据集
            metrics: 评估指标
            usage_example: 使用示例代码
        """
        card = f"""---
language:
- en
license: apache-2.0
tags:
- {task}
- transformers
datasets:
- {dataset}
metrics:
{chr(10).join(f'- {k}: {v}' for k, v in metrics.items())}
---

# {model_name}

## Model Description

This model is a fine-tuned version of [{base_model}](https://huggingface.co/{base_model}) on the {dataset} dataset.

## Intended Uses & Limitations

This model is intended for {task} tasks.

**Limitations:**
- May generate biased or incorrect outputs
- Not suitable for production use without further validation

## Training Data

The model was trained on {dataset}.

## Training Procedure

### Training Hyperparameters

- Learning rate: 5e-5
- Batch size: 32
- Epochs: 3
- Optimizer: AdamW

### Metrics

{chr(10).join(f'- **{k}**: {v}' for k, v in metrics.items())}

## Usage Example

```python
{usage_example}
```

## Citation

```bibtex
@misc{{{model_name.replace('/', '_').replace('-', '_')},
  author = {{Your Name}},
  title = {{{model_name}}},
  year = {{2026}},
  publisher = {{Hugging Face}},
  url = {{https://huggingface.co/{model_name}}}
}}
```

## Contact

For questions, contact: your.email@example.com
"""
        return card

# 生成示例
card = ModelCardGenerator.generate_card(
    model_name="username/my-sentiment-classifier",
    base_model="distilbert-base-uncased",
    task="text-classification",
    dataset="imdb",
    metrics={"accuracy": 0.93, "f1": 0.92},
    usage_example="""from transformers import pipeline

classifier = pipeline("text-classification", model="username/my-sentiment-classifier")
result = classifier("I love this movie!")
print(result)"""
)

print(card[:500])
```

---

### 3. Spaces应用部署

```python
# Gradio应用示例（app.py）
import gradio as gr
from transformers import pipeline

class GradioApp:
    """Gradio应用封装"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.pipeline = pipeline("text-generation", model=model_name)
    
    def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7
    ) -> str:
        """生成文本"""
        outputs = self.pipeline(
            prompt,
            max_length=max_length,
            temperature=temperature,
            do_sample=True
        )
        return outputs[0]["generated_text"]
    
    def launch(self):
        """启动Gradio界面"""
        interface = gr.Interface(
            fn=self.generate_text,
            inputs=[
                gr.Textbox(label="输入提示词", lines=3),
                gr.Slider(10, 500, value=100, label="最大长度"),
                gr.Slider(0.1, 2.0, value=0.7, label="温度")
            ],
            outputs=gr.Textbox(label="生成文本", lines=5),
            title="文本生成Demo",
            description=f"使用 {self.model_name} 生成文本"
        )
        
        interface.launch()

# 部署到Hugging Face Spaces
# 1. 创建requirements.txt
requirements_txt = """transformers>=4.35.0
torch>=2.0.0
gradio>=4.0.0
"""

# 2. 创建README.md (Spaces配置)
spaces_readme = """---
title: My Text Generator
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# My Text Generation Space

This Space demonstrates text generation using a fine-tuned model.
"""

print("部署步骤:")
print("1. 创建 app.py（使用上面的GradioApp）")
print("2. 创建 requirements.txt")
print("3. 创建 README.md（包含Spaces配置）")
print("4. git push到Hugging Face Space仓库")
print("5. Space会自动构建并运行")
```

---

## 五、PEFT与TRL库

### 1. PEFT：参数高效微调实战

```python
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from transformers import AutoModelForCausalLM, AutoTokenizer

class PEFTTrainer:
    """PEFT微调训练器"""
    
    @staticmethod
    def setup_lora_model(
        base_model_name: str,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        use_4bit: bool = True
    ):
        """配置LoRA模型
        
        Args:
            base_model_name: 基础模型
            lora_r: LoRA秩
            lora_alpha: LoRA缩放因子
            lora_dropout: Dropout率
            target_modules: 应用LoRA的模块（None表示自动）
            use_4bit: 是否使用4-bit量化
        """
        # 1. 加载基础模型（量化）
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
            
            # 准备模型用于kbit训练
            model = prepare_model_for_kbit_training(model)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map="auto",
                torch_dtype=torch.float16
            )
        
        # 2. 配置LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,  # 自动检测query/value层
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # 3. 应用LoRA
        model = get_peft_model(model, lora_config)
        
        # 4. 打印可训练参数
        model.print_trainable_parameters()
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer

# 使用示例
lora_model, tokenizer = PEFTTrainer.setup_lora_model(
    base_model_name="meta-llama/Llama-3-8B",
    lora_r=16,
    lora_alpha=32,
    use_4bit=True
)
```

**输出示例**：
```
trainable params: 41,943,040 || all params: 8,071,014,400 || trainable%: 0.52%
```

---

### 2. TRL：强化学习与偏好优化

```python
from trl import SFTTrainer, DPOTrainer
from datasets import load_dataset

class TRLTrainer:
    """TRL训练器封装"""
    
    @staticmethod
    def sft_train(
        model,
        tokenizer,
        dataset_name: str = "timdettmers/openassistant-guanaco",
        max_seq_length: int = 512
    ):
        """监督式微调（SFT）
        
        Args:
            model: 模型
            tokenizer: 分词器
            dataset_name: 数据集
            max_seq_length: 最大序列长度
        """
        # 加载数据集
        dataset = load_dataset(dataset_name, split="train")
        
        # 配置SFTTrainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",  # 包含指令的字段
            max_seq_length=max_seq_length,
            args=TrainingArguments(
                output_dir="./sft_output",
                per_device_train_batch_size=4,
                gradient_accumulation_steps=4,
                learning_rate=2e-4,
                num_train_epochs=3,
                logging_steps=10,
                save_steps=100,
                fp16=True,
            )
        )
        
        return trainer
    
    @staticmethod
    def dpo_train(
        model,
        tokenizer,
        preference_dataset
    ):
        """直接偏好优化（DPO）
        
       Args:
            model: 基础模型（SFT后）
            tokenizer: 分词器
            preference_dataset: 偏好数据集
                格式: {"prompt": "...", "chosen": "...", "rejected": "..."}
        """
        # 配置DPO Trainer
        trainer = DPOTrainer(
            model=model,
            ref_model=None,  # 自动创建参考模型
            tokenizer=tokenizer,
            train_dataset=preference_dataset,
            beta=0.1,  # DPO温度参数
            args=TrainingArguments(
                output_dir="./dpo_output",
                per_device_train_batch_size=2,
                gradient_accumulation_steps=8,
                learning_rate=5e-5,
                num_train_epochs=3,
                logging_steps=10,
                save_steps=100,
                fp16=True,
            )
        )
        
        return trainer

# DPO数据集示例
dpo_data = {
    "prompt": [
        "Explain quantum computing",
        "Write a poem about AI"
    ],
    "chosen": [
        "Quantum computing uses quantum bits (qubits) that can exist in superposition...",
        "In circuits deep where electrons flow, Intelligence begins to grow..."
    ],
    "rejected": [
        "Quantum stuff is hard to understand.",
        "AI poem: AI is cool. Very cool. The end."
    ]
}

from datasets import Dataset
dpo_dataset = Dataset.from_dict(dpo_data)

# 训练
# dpo_trainer = TRLTrainer.dpo_train(lora_model, tokenizer, dpo_dataset)
# dpo_trainer.train()
```


---

## 本章小结

### 核心知识回顾

本章系统介绍了Hugging Face生态的核心组件，从模型加载、数据处理到训练微调、模型发布，构建了完整的LLM开发工作流。

#### 1. Transformers库核心用法

**模型加载最佳实践**：
```python
# 标准加载（推理）
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2"
)

# 4-bit量化加载（节省显存）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4"
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config
)
# 显存占用: ~0.5GB/B参数
```

**Pipeline快速使用**：
- `text-generation`: 文本生成
- `fill-mask`: BERT填空
- `text-classification`: 情感分类
- `question-answering`: 问答
- 自定义Pipeline：继承`Pipeline`类

**模型定制**：
- 扩展词表：`tokenizer.add_special_tokens()`
- 调整嵌入：`model.resize_token_embeddings()`
- 自定义架构：继承`PreTrainedModel`

#### 2. Datasets数据处理

**加载与预处理**：
```python
# 标准加载
dataset = load_dataset("imdb", split="train")

# 流式加载（TB级数据）
dataset = load_dataset("c4", split="train", streaming=True)

# 批量预处理（10-100x提速）
dataset = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=4
)
```

**高级操作**：
- 过滤：`dataset.filter(lambda x: len(x['text']) > 100)`
- 采样：`dataset.shuffle().select(range(1000))`
- 交叉数据集：`interleave_datasets([ds1, ds2], probabilities=[0.7, 0.3])`

**流式处理**：
```python
# 处理大数据集不加载到内存
for batch in dataset.iter(batch_size=1000):
    process(batch)
```

#### 3. Trainer训练流程

**TrainingArguments关键参数**：
```python
TrainingArguments(
    output_dir="./results",
    
    # 训练超参数
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,  # 等效batch=32
    num_train_epochs=3,
    
    # 评估
    evaluation_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,
    
    # 性能优化
    fp16=True,  # 混合精度
    gradient_checkpointing=True,  # 节省显存
    
    # DeepSpeed
    deepspeed="ds_config.json"
)
```

**自定义训练**：
- 自定义损失：重写`compute_loss()`
- 回调函数：`TrainerCallback`
  - `on_epoch_begin/end`
  - `on_log`
  - `on_save`
- 早停：`EarlyStoppingCallback(patience=3)`

#### 4. 模型分享与部署

**上传到Hub**：
```python
# 上传模型
model.push_to_hub("username/model-name")
tokenizer.push_to_hub("username/model-name")

# 或使用API
upload_folder(
    folder_path="./model",
    repo_id="username/model-name"
)
```

**模型卡片要素**：
- 模型描述与用途
- 训练数据与超参数
- 评估指标
- 使用示例代码
- 限制与偏见说明
- 引用信息

**Spaces部署**：
```python
import gradio as gr

interface = gr.Interface(
    fn=model_fn,
    inputs=gr.Textbox(),
    outputs=gr.Textbox()
)
interface.launch()
# 推送到Space仓库即自动部署
```

#### 5. PEFT与TRL库

**LoRA微调**：
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,  # LoRA秩
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable%: 0.52%（仅训练0.52%参数）
```

**TRL训练**：
- **SFTTrainer**: 监督式微调
  - 自动处理指令格式
  - 支持`dataset_text_field`
- **DPOTrainer**: 直接偏好优化
  - 需要`(prompt, chosen, rejected)`格式
  - `beta`参数控制KL惩罚

---

### 关键代码模板

#### 完整微调流程
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 1. 加载量化模型
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    quantization_config=bnb_config,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

# 2. 配置LoRA
lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1, bias="none")
model = get_peft_model(model, lora_config)

# 3. 加载数据
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")
dataset = load_dataset("tatsu-lab/alpaca", split="train")
dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True), batched=True)

# 4. 配置训练
args = TrainingArguments(
    output_dir="./lora_output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,
    logging_steps=10
)

# 5. 训练
trainer = Trainer(model=model, args=args, train_dataset=dataset)
trainer.train()

# 6. 保存
model.save_pretrained("./lora_model")
```

---

### 实战建议

#### 显存优化策略
```
场景1: 8B模型，24GB显存（RTX 4090）
✓ 4-bit量化 + LoRA (r=16)
✓ gradient_checkpointing=True
✓ per_device_batch_size=2, gradient_accumulation=8
✓ max_seq_length=512

场景2: 70B模型，单张A100 40GB
✓ 4-bit量化 + LoRA (r=8)
✓ 显存占用: ~35GB
✓ per_device_batch_size=1, gradient_accumulation=16

场景3: 70B模型，多张GPU
✓ device_map="auto"（自动分配层）
✓ 或DeepSpeed ZeRO-3（参数分片）
```

---

### 常见问题与解决方案

#### Q1: 显存溢出（CUDA Out of Memory）
**A**:
1. 降低batch size（`per_device_train_batch_size=1`）
2. 增加梯度累积（`gradient_accumulation_steps=16`）
3. 启用梯度检查点（`gradient_checkpointing=True`）
4. 使用4-bit量化（`load_in_4bit=True`）
5. 降低序列长度（`max_seq_length=512`）
6. 使用DeepSpeed ZeRO-3

#### Q2: 数据集加载慢
**A**:
- 使用`num_proc`并行处理
- 启用`streaming=True`（不加载到内存）
- 预处理后保存：`dataset.save_to_disk("./processed")`
- 使用SSD存储缓存

#### Q3: LoRA训练后如何推理？
**A**:
```python
# 方法1: 加载PEFT模型
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("base_model")
model = PeftModel.from_pretrained(base_model, "./lora_weights")

# 方法2: 合并LoRA权重
model = model.merge_and_unload()
model.save_pretrained("./merged_model")
```

---

### 实战练习

#### 练习1: 微调文本分类模型（难度：⭐⭐）
**任务**：
1. 使用`distilbert-base-uncased`
2. 在IMDB数据集上微调
3. 使用Trainer API
4. 达到准确率 > 90%

#### 练习2: LoRA微调对话模型（难度：⭐⭐⭐）
**任务**：
1. 使用`meta-llama/Llama-3-8B`
2. 4-bit量化 + LoRA (r=16)
3. 在Alpaca数据集上微调
4. 上传到Hugging Face Hub

#### 练习3: 流式处理大数据集（难度：⭐⭐⭐⭐）
**任务**：
1. 流式加载C4数据集
2. 批量分词（batch_size=1000）
3. 保存到磁盘（分块）
4. 统计词频（Top 10000）

#### 练习4: Gradio应用部署（难度：⭐⭐⭐）
**任务**：
1. 创建文本生成Gradio界面
2. 支持温度、最大长度调节
3. 部署到Hugging Face Spaces
4. 添加示例输入

---

### 下一章预告

掌握了Hugging Face生态后，下一章我们将深入**DeepSpeed分布式训练**，学习如何突破单卡限制，训练超大规模模型：

- ZeRO优化器原理（ZeRO-1/2/3）
- DeepSpeed配置与使用
- CPU/NVMe卸载（ZeRO-Offload）
- 无限规模训练（ZeRO-Infinity）
- 多机分布式训练实战

从单卡到千卡，从8B到千B，DeepSpeed让大模型训练触手可及！

---

**本章完**

