# 第03篇_精准控制(04)_Upscaler：图像放大完全指南

> **难度**: ⭐⭐⭐⭐ | **推荐度**: ⭐⭐⭐⭐⭐

## 4.1 为什么需要Upscaler

### 4.1.1 原生生成的分辨率限制

```python
# AIGC模型原生分辨率限制

MODEL_NATIVE_RESOLUTIONS = {
    "SD 1.5": {
        "最佳": "512×512",
        "稳定": "512-768",
        "问题": "1024+ 变形/崩坏"
    },
    "SDXL": {
        "最佳": "1024×1024",
        "稳定": "1024-1536",
        "问题": "2048+ 细节丢失"
    },
    "Flux.1": {
        "最佳": "1024×1024",
        "稳定": "512-1536"
    }
}

# 实际需求
REAL_WORLD_NEEDS = {
    "电商产品图": "2048×2048 (详情页)",
    "海报": "4096×6144 (A2尺寸300DPI)",
    "社交媒体": "1080×1920 (竖屏)",
    "打印": "3000×4000+ (高清)"
}

# 结论: 必须掌握Upscale技术
```

### 4.1.2 Upscaler技术分类

| 类型 | 原理 | 质量 | 速度 | 适用场景 |
|------|------|------|------|----------|
| **传统插值** | 双三次/Lanczos | ⭐⭐ | 极快 | 不推荐 |
| **AI超分辨率** | RealESRGAN/SwinIR | ⭐⭐⭐⭐ | 快 | 通用放大 |
| **Latent Upscale** | 潜空间放大 | ⭐⭐⭐ | 快 | 配合Highres Fix |
| **Tile Upscale** | 分块+AI重绘 | ⭐⭐⭐⭐⭐ | 中 | 大幅放大 |
| **Ultimate SD Upscale** | 分块+Latent+AI | ⭐⭐⭐⭐⭐ | 慢 | 极致质量 |

---

## 4.2 RealESRGAN (快速AI超分)

### 4.2.1 原理

```python
# RealESRGAN: Real-World Enhanced Super-Resolution GAN

INPUT: 低分辨率图像 (512×512)
    ↓
ENCODER: CNN提取特征
    ↓
RRDB: 残差密集块 (23层)
    - 学习高频细节
    - 恢复纹理
    ↓
DECODER: 上采样
    ↓
OUTPUT: 高分辨率 (2048×2048, 4×放大)

# 优势: 快速,通用,无需提示词
```

### 4.2.2 模型选择

| 模型 | 放大倍数 | 适用场景 | 质量 | 速度 |
|------|----------|----------|------|------|
| **RealESRGAN_x4plus** | 4× | 通用 (照片/AI图) | ⭐⭐⭐⭐ | 快 |
| **RealESRGAN_x4plus_anime_6B** | 4× | 动漫/插画 | ⭐⭐⭐⭐⭐ | 快 |
| **RealESRGAN_x2plus** | 2× | 轻度放大 | ⭐⭐⭐⭐ | 极快 |
| **realesr-general-x4v3** | 4× | 真实照片 | ⭐⭐⭐⭐ | 快 |

### 4.2.3 Python使用

```python
# 方法1: 命令行工具
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# 加载模型
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
upsampler = RealESRGANer(
    scale=4,
    model_path='RealESRGAN_x4plus.pth',
    model=model,
    tile=512,        # 分块大小 (显存不足时降低)
    tile_pad=10,     # 分块重叠
    pre_pad=0,
    half=True        # FP16加速
)

# 放大图像
from PIL import Image
import numpy as np

img = Image.open("input.png").convert("RGB")
img_np = np.array(img)

# Upscale
output, _ = upsampler.enhance(img_np, outscale=4)

# 保存
Image.fromarray(output).save("output.png")
```

```python
# 方法2: ComfyUI集成
# 需要安装 comfyui-image-upscale-nodes

工作流:
[Load Image]
    ↓
[Image Upscale With Model]
  - upscale_model: RealESRGAN_x4plus.pth
    ↓
[Save Image]

# 一个节点搞定!
```

### 4.2.4 批量处理脚本

```python
# batch_upscale.py
import os
from pathlib import Path
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from PIL import Image
import numpy as np

def batch_upscale(input_dir, output_dir, model_path, scale=4):
    """批量放大文件夹中的所有图片"""

    # 初始化模型
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23)
    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=512,
        half=True
    )

    # 遍历文件
    Path(output_dir).mkdir(exist_ok=True)

    for img_file in Path(input_dir).glob("*.png"):
        print(f"Processing: {img_file.name}")

        # 读取
        img = Image.open(img_file).convert("RGB")
        img_np = np.array(img)

        # 放大
        output, _ = upsampler.enhance(img_np, outscale=scale)

        # 保存
        output_path = Path(output_dir) / f"{img_file.stem}_4x.png"
        Image.fromarray(output).save(output_path)

        print(f"Saved: {output_path}")

# 使用
batch_upscale(
    input_dir="./low_res/",
    output_dir="./high_res/",
    model_path="RealESRGAN_x4plus.pth",
    scale=4
)
```

---

## 4.3 Highres Fix (WebUI原生)

### 4.3.1 原理

```python
# Highres Fix: 两阶段生成法

Stage 1: 原生分辨率生成
[txt2img]
  size: 512×512  # SD 1.5最佳分辨率
  steps: 30
  ↓
  latent_512 (完美质量)

Stage 2: Latent Upscale + img2img重绘
[Latent Upscale]
  latent_512 → latent_1024 (潜空间插值)
  ↓
[img2img]
  denoise: 0.5  # 重绘强度
  steps: 15
  ↓
  final_1024 (细节丰富)

# 关键: 避免直接1024生成导致的变形
```

### 4.3.2 WebUI参数

```python
# AUTOMATIC1111 WebUI设置

txt2img界面:
1. 基础设置
   - Width: 512
   - Height: 512
   - Steps: 30
   - CFG: 7.0

2. 启用Highres Fix (勾选)

3. Highres Fix参数
   - Upscaler: Latent (bicubic antialiased)  # 推荐
   - Hires steps: 15                          # 重绘步数
   - Denoising strength: 0.5                  # 重绘强度
   - Upscale by: 2.0                          # 放大倍数

   最终分辨率: 512×2=1024

# 对比
直接1024生成: 70%概率变形/重复
Highres Fix: 95%完美

# 成本
时间: +50% (额外15步)
质量: +200%
```

### 4.3.3 最佳实践

```python
# 不同模型的最佳Highres Fix配置

SD_1_5_CONFIG = {
    "base_size": (512, 512),
    "upscaler": "Latent (bicubic antialiased)",
    "hires_steps": 15,
    "denoise": 0.5,
    "upscale_by": 2.0,  # 512→1024
    "total_time": "40秒"
}

SDXL_CONFIG = {
    "base_size": (1024, 1024),
    "upscaler": "Latent (bicubic antialiased)",
    "hires_steps": 10,
    "denoise": 0.4,      # SDXL细节更好,降低重绘
    "upscale_by": 1.5,   # 1024→1536
    "total_time": "60秒"
}

# 技巧: 先Highres Fix到1024,再RealESRGAN到4096
# 质量最优路径
```

---

## 4.4 Ultimate SD Upscale (ComfyUI最强)

### 4.4.1 原理

```python
# Ultimate SD Upscale: 分块Latent重绘

PROCESS:
1. 输入图像 (1024×1024)
   ↓
2. AI Upscale预放大
   RealESRGAN_x2 → 2048×2048
   ↓
3. 分块处理 (Tiling)
   切分为 512×512 tiles, overlap=64

   Tile 1 (0,0 - 512,512)
   ↓ [VAE Encode] → latent
   ↓ [KSampler] denoise=0.35
   ↓ [VAE Decode] → refined tile

   Tile 2 (448,0 - 960,512)  # overlap 64px
   ...

4. 羽化合并 (Feathering)
   重叠区域线性混合 → 无缝拼接
   ↓
5. 输出 (2048×2048, 细节完美)

# 优势: 显存友好 + 极致质量
```

### 4.4.2 ComfyUI节点安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git
# 重启ComfyUI
```

### 4.4.3 工作流节点连接

```python
# 完整工作流

[Load Image]  # 输入1024×1024
    ↓
[Load Checkpoint]  # 重绘用的模型
    ↓
[CLIP Text Encode] × 2  # positive + negative
    ↓
[Load Upscale Model]  # RealESRGAN_x4plus.pth
    ↓
[Ultimate SD Upscale]
  - upscale_model: RealESRGAN_x4
  - mode: Linear         # 羽化模式
  - tile_width: 512      # 分块大小
  - tile_height: 512
  - mask_blur: 8         # 羽化程度
  - tile_padding: 32     # 重叠区域
  - seam_fix_mode: Band Pass  # 接缝修复
  - seam_fix_denoise: 0.35    # 接缝重绘强度
  - seam_fix_width: 64
  - denoise: 0.35        # 整体重绘强度
  - steps: 20            # 采样步数
  - cfg: 8.0
    ↓
[Save Image]  # 输出4096×4096

# 显存占用: 仅约4GB (处理单个512 tile)
# 质量: ⭐⭐⭐⭐⭐
```

### 4.4.4 参数调优指南

```python
# 参数影响分析

DENOISE_STRENGTH = {
    0.2: "轻微重绘,保持原图 (推荐照片)",
    0.35: "标准重绘,平衡质量 (推荐AI图)",
    0.5: "强烈重绘,大幅改变 (风格化)",
    0.7: "接近重新生成 (不推荐)"
}

TILE_SIZE = {
    256: "显存<4GB,速度慢",
    512: "推荐 (8-12GB显存)",
    768: "高端GPU (16GB+)",
    1024: "极致质量 (24GB)"
}

MASK_BLUR = {
    0: "硬边缘,可能有接缝",
    8: "推荐 (平滑过渡)",
    16: "模糊接缝区域",
    32: "过度模糊"
}

# 推荐配置
RECOMMENDED_CONFIG = {
    "denoise": 0.35,
    "tile_width": 512,
    "tile_height": 512,
    "mask_blur": 8,
    "tile_padding": 32,
    "steps": 20,
    "cfg": 8.0
}
```

### 4.4.5 实战案例

```python
# 案例: 健身教练图 512→4096 (8倍放大)

工作流:
1. [Load Image]
   input: coach_512.png

2. [Load Checkpoint]
   ckpt: realisticVision_v51.safetensors  # 真实系模型

3. [CLIP Text Encode (Positive)]
   prompt: "professional fitness coach, high detail, sharp"

4. [CLIP Text Encode (Negative)]
   prompt: "blurry, low quality, artifacts"

5. [Load Upscale Model]
   model: RealESRGAN_x4plus.pth

6. [Ultimate SD Upscale]
   - upscale_by: 8.0  # 512→4096
   - denoise: 0.35
   - tile: 512×512
   - steps: 25

7. [Save Image]
   output: coach_4096.png

结果:
- 分辨率: 512×512 → 4096×4096
- 细节: 面部清晰,肌肉纹理可见
- 时间: 约5分钟 (RTX 4090)
- 质量: 可打印A2海报
```

---

## 4.5 对比测试

### 4.5.1 质量对比

```python
# 测试条件: 512×512 → 2048×2048

TEST_RESULTS = {
    "传统双三次插值": {
        "质量": "⭐⭐",
        "问题": "模糊,锯齿,无细节",
        "时间": "1秒",
        "推荐": "❌ 不推荐"
    },

    "RealESRGAN_x4": {
        "质量": "⭐⭐⭐⭐",
        "优点": "快速,细节好",
        "问题": "可能过锐化",
        "时间": "5秒",
        "推荐": "✅ 快速放大首选"
    },

    "Highres Fix (2×)": {
        "质量": "⭐⭐⭐⭐",
        "优点": "保持AI生成质感",
        "问题": "仅2倍,不够大",
        "时间": "20秒",
        "推荐": "✅ 生成时使用"
    },

    "Ultimate SD Upscale": {
        "质量": "⭐⭐⭐⭐⭐",
        "优点": "极致细节,完美",
        "问题": "慢,需要显存",
        "时间": "2分钟",
        "推荐": "✅ 极致质量首选"
    },

    "Highres Fix + RealESRGAN": {
        "质量": "⭐⭐⭐⭐⭐",
        "优点": "组合最优",
        "流程": "512→1024 (Highres) → 4096 (ESRGAN)",
        "时间": "25秒",
        "推荐": "✅ 性价比最高"
    }
}
```

### 4.5.2 显存占用对比

| 方法 | 输入 | 输出 | 显存占用 | 适用GPU |
|------|------|------|----------|---------|
| **RealESRGAN** | 1024 | 4096 | ~2GB | 任意 |
| **Highres Fix** | 512 | 1024 | ~4GB | 4GB+ |
| **Ultimate (tile=512)** | 1024 | 8192 | ~4GB | 8GB+ |
| **Ultimate (tile=768)** | 1024 | 8192 | ~8GB | 16GB+ |
| **直接生成4096** | - | 4096 | ~16GB | 24GB |

---

## 4.6 生产级批量放大系统

### 4.6.1 系统架构

```python
# 批量放大流水线

import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import torch

class BatchUpscaler:
    def __init__(self, model_path, gpu_ids=[0]):
        """
        初始化批量放大器

        Args:
            model_path: RealESRGAN模型路径
            gpu_ids: 使用的GPU ID列表 (多GPU并行)
        """
        self.model_path = model_path
        self.gpu_ids = gpu_ids

    def upscale_single(self, input_path, output_path, gpu_id):
        """单张图片放大 (单进程)"""
        import torch
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from PIL import Image
        import numpy as np

        # 设置GPU
        torch.cuda.set_device(gpu_id)

        # 加载模型
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23)
        upsampler = RealESRGANer(
            scale=4,
            model_path=self.model_path,
            model=model,
            tile=512,
            tile_pad=10,
            half=True,
            device=f'cuda:{gpu_id}'
        )

        # 放大
        img = Image.open(input_path).convert("RGB")
        img_np = np.array(img)
        output, _ = upsampler.enhance(img_np, outscale=4)
        Image.fromarray(output).save(output_path)

        # 清理显存
        del upsampler
        torch.cuda.empty_cache()

    def batch_process(self, input_dir, output_dir):
        """批量处理"""
        Path(output_dir).mkdir(exist_ok=True)

        # 获取所有图片
        image_files = list(Path(input_dir).glob("*.png")) + \
                     list(Path(input_dir).glob("*.jpg"))

        print(f"Found {len(image_files)} images")

        # 分配任务到多GPU
        tasks = []
        for idx, img_file in enumerate(image_files):
            gpu_id = self.gpu_ids[idx % len(self.gpu_ids)]
            output_path = Path(output_dir) / f"{img_file.stem}_4x.png"
            tasks.append((img_file, output_path, gpu_id))

        # 多进程并行处理
        with ProcessPoolExecutor(max_workers=len(self.gpu_ids)) as executor:
            futures = []
            for input_path, output_path, gpu_id in tasks:
                future = executor.submit(
                    self.upscale_single,
                    input_path,
                    output_path,
                    gpu_id
                )
                futures.append(future)

            # 等待完成
            for idx, future in enumerate(futures):
                future.result()
                print(f"Progress: {idx+1}/{len(tasks)}")

# 使用示例
upscaler = BatchUpscaler(
    model_path="RealESRGAN_x4plus.pth",
    gpu_ids=[0, 1, 2, 3]  # 4× RTX 4090并行
)

upscaler.batch_process(
    input_dir="./products_1024/",
    output_dir="./products_4096/"
)

# 吞吐量:
# 单GPU: ~12张/分钟
# 4GPU: ~48张/分钟
# 1000张: 约21分钟
```

### 4.6.2 API服务部署

```python
# upscale_api.py - FastAPI服务

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import numpy as np
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

app = FastAPI()

# 全局模型加载 (启动时初始化)
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23)
upsampler = RealESRGANer(
    scale=4,
    model_path='RealESRGAN_x4plus.pth',
    model=model,
    tile=512,
    half=True
)

@app.post("/upscale")
async def upscale_image(file: UploadFile = File(...)):
    """
    API: 上传图片,返回4×放大图

    Usage:
        curl -X POST -F "file=@input.png" \
             http://localhost:8000/upscale \
             --output output.png
    """
    # 读取上传文件
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_np = np.array(img)

    # 放大
    output, _ = upsampler.enhance(img_np, outscale=4)

    # 转换为字节流
    output_img = Image.fromarray(output)
    buf = io.BytesIO()
    output_img.save(buf, format='PNG')
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")

# 启动服务
# uvicorn upscale_api:app --host 0.0.0.0 --port 8000
```

---

## 4.7 最佳实践总结

### 4.7.1 场景化选择

```python
SCENARIO_RECOMMENDATION = {
    "快速原型/测试": {
        "方法": "RealESRGAN_x4",
        "时间": "5秒/张",
        "质量": "⭐⭐⭐⭐"
    },

    "批量生产": {
        "方法": "Highres Fix (生成时) + RealESRGAN (后处理)",
        "时间": "30秒/张",
        "质量": "⭐⭐⭐⭐⭐"
    },

    "极致质量 (海报/打印)": {
        "方法": "Ultimate SD Upscale",
        "时间": "2-5分钟/张",
        "质量": "⭐⭐⭐⭐⭐"
    },

    "动漫/插画": {
        "方法": "RealESRGAN_x4plus_anime_6B",
        "时间": "5秒/张",
        "质量": "⭐⭐⭐⭐⭐"
    },

    "显存受限 (<8GB)": {
        "方法": "RealESRGAN (tile=256)",
        "时间": "10秒/张",
        "质量": "⭐⭐⭐⭐"
    }
}
```

### 4.7.2 黄金组合

```python
# 最优质量路径: 三阶段放大

GOLDEN_PATH = {
    "Stage 1: 生成": {
        "方法": "Highres Fix",
        "分辨率": "512 → 1024",
        "质量": "完美基础"
    },

    "Stage 2: AI超分": {
        "方法": "RealESRGAN_x2",
        "分辨率": "1024 → 2048",
        "质量": "细节增强"
    },

    "Stage 3: 精修 (可选)": {
        "方法": "Ultimate SD Upscale",
        "分辨率": "2048 → 4096",
        "质量": "极致完美"
    }
}

# 成本:
# Stage 1+2: 30秒 (日常使用)
# Stage 1+2+3: 3分钟 (关键交付)
```

### 4.7.3 避坑指南

```python
COMMON_MISTAKES = {
    "错误1: 直接生成4096": {
        "问题": "变形/重复/崩坏",
        "解决": "先512/1024生成,再Upscale"
    },

    "错误2: 过度重绘 (denoise>0.6)": {
        "问题": "图像改变过大,不像原图",
        "解决": "denoise保持0.3-0.4"
    },

    "错误3: tile太大显存爆": {
        "问题": "OOM错误",
        "解决": "降低tile_size (512→256)"
    },

    "错误4: 多次ESRGAN叠加": {
        "问题": "过度锐化,噪点",
        "解决": "最多2次ESRGAN,中间加降噪"
    }
}
```

---

## 4.8 总结

**核心要点**:
1. **生成时用Highres Fix** - 避免直接大分辨率生成
2. **RealESRGAN快速通用** - 日常放大首选
3. **Ultimate SD Upscale极致质量** - 关键交付使用
4. **组合使用效果最佳** - Highres + ESRGAN

**质量排序**:
```
Ultimate SD Upscale > Highres+ESRGAN > RealESRGAN > Highres Fix > 传统插值
```

**速度排序**:
```
传统插值 > RealESRGAN > Highres Fix > Ultimate SD Upscale
```

**推荐配置**:
- **8GB显存**: RealESRGAN + Highres Fix
- **12GB显存**: Ultimate SD Upscale (tile=512)
- **24GB显存**: Ultimate SD Upscale (tile=768)

掌握Upscaler技术,让你的AI图像突破分辨率限制,达到商业级质量!
