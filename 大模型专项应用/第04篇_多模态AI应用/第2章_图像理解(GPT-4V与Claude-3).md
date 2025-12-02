# 第2章 图像理解(GPT-4V与Claude-3)

> 掌握主流视觉大模型的能力边界与最佳实践

## 2.1 GPT-4 Vision完整指南

### 2.1.1 核心能力

GPT-4 Vision (GPT-4V) 是Open AI在2023年9月发布的多模态大模型,具备以下核心能力:

- 图像内容理解与描述
- 文档OCR与解析
- 图表数据提取
- 视觉问答(VQA)
- 多图对比分析

### 2.1.2 API使用

```python
from openai import OpenAI
import base64
from pathlib import Path

class GPT4VisionAnalyzer:
    """GPT-4 Vision分析器"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def analyze_image(
        self,
        image_path: str,
        prompt: str = "详细描述这张图片的内容",
        max_tokens: int = 500
    ) -> str:
        """
        分析单张图片
        
        Args:
            image_path: 图片路径
            prompt: 分析提示
            max_tokens: 最大输出tokens
            
        Returns:
            分析结果
        """
        # Base64编码图片
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    def analyze_multiple_images(
        self,
        image_paths: list[str],
        prompt: str = "对比分析这些图片"
    ) -> str:
        """
        分析多张图片
        
        Args:
            image_paths: 图片路径列表
            prompt: 分析提示
            
        Returns:
            分析结果
        """
        content = [{"type": "text", "text": prompt}]
        
        for path in image_paths:
            with open(path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_data}"
                }
            })
        
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[{"role": "user", "content": content}],
            max_tokens=1000
        )
        
        return response.choices[0].message.content

# 使用示例
analyzer = GPT4VisionAnalyzer(api_key="your-api-key")

# 单图分析
result = analyzer.analyze_image("product.jpg", "这个产品有什么特点?")
print(result)

# 多图对比
results = analyzer.analyze_multiple_images(
    ["before.jpg", "after.jpg"],
    "对比前后两张图的差异"
)
print(results)
```

## 2.2 Claude 3.5 Sonnet视觉能力

### 2.2.1 核心优势

Claude 3.5 Sonnet在视觉理解方面具有独特优势:

- 推理能力更强
- 细节捕捉精准
- 安全性更高
- 支持20张图片/请求

### 2.2.2 API使用

```python
from anthropic import Anthropic
import base64

class Claude35VisionAnalyzer:
    """Claude 3.5 Sonnet分析器"""
    
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
    
    def analyze_image(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int = 1024
    ) -> str:
        """分析图片"""
        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")
        
        message = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }
            ],
        )
        
        return message.content[0].text

# 使用示例
analyzer = Claude35VisionAnalyzer(api_key="your-api-key")
result = analyzer.analyze_image("document.jpg", "提取文档中的所有文字")
print(result)
```

## 2.3 模型对比与选型

### 2.3.1 能力对比

| 能力维度 | GPT-4V | Claude 3.5 Sonnet | 推荐场景 |
|---------|--------|------------------|---------|
| OCR准确率 | 95% | 93% | GPT-4V |
| 图表理解 | 优秀 | 优秀 | 平手 |
| 推理能力 | 良好 | 优秀 | Claude |
| 多图对比 | 10张 | 20张 | Claude |
| 响应速度 | 快 | 中等 | GPT-4V |
| 成本 | 高 | 中 | Claude |

### 2.3.2 实战案例

**案例: 发票信息提取**

```python
def extract_invoice_info(image_path: str) -> dict:
    """提取发票信息"""
    analyzer = GPT4VisionAnalyzer(api_key="your-key")
    
    prompt = """
    请提取这张发票的以下信息,以JSON格式返回:
    {
        "invoice_number": "发票号码",
        "date": "开票日期",
        "amount": "金额",
        "company": "公司名称",
        "items": ["商品列表"]
    }
    """
    
    result = analyzer.analyze_image(image_path, prompt)
    
    import json
    return json.loads(result)

# 使用
invoice_data = extract_invoice_info("invoice.jpg")
print(invoice_data)
```

## 本章小结

- GPT-4V适合OCR、文档解析、快速响应场景
- Claude 3.5适合复杂推理、多图对比、安全敏感场景
- 实际应用中可根据任务特点选择合适的模型

---

**下一章**: [第3章 视频内容分析](./第3章_视频内容分析.md)
