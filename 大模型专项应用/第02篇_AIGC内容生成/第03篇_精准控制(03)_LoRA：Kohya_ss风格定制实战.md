# 第15章 风格定制(一) LoRA训练完全精通

> **学习目标**: 掌握LoRA训练全流程,训练自己的风格/角色/概念模型
>
> **难度**: ⭐⭐⭐⭐
> **学习周期**: 1-2周
> **推荐度**: ⭐⭐⭐⭐⭐ (核心技能,定制化必备)

---

## 15.1 LoRA原理与价值

### 15.1.1 什么是LoRA?

**LoRA (Low-Rank Adaptation)**: 低秩适应,一种高效的模型微调技术

**核心思想**:
```
传统微调: 调整整个模型参数 (1.5B+参数)
  ↓ 问题: 需要大量显存 + 长时间训练

LoRA: 仅训练小矩阵 (通常< 100MB)
  ↓ 优势: 显存友好 + 快速训练 + 易于分享
```

**数学原理**:

原始权重矩阵: $W \in \mathbb{R}^{d \times k}$

LoRA更新:
$$
W' = W + \alpha \cdot BA
$$

其中:
- $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$
- $r$ 是秩 (rank), 通常$r \ll \min(d, k)$, 如$r=8, 16, 32$
- $\alpha$ 是缩放因子

**参数对比**:
```python
# SD 1.5 完整模型
total_params = 1.5B
file_size = "4GB (fp16)"

# LoRA (rank=32)
lora_params = 约10M (取决于rank和目标层数)
file_size = "10-100MB"

# 节省
compression = 1.5B / 10M = 150x
```

### 15.1.2 LoRA的应用场景

| 用途 | 描述 | 训练图片数 | 推荐Rank |
|------|------|-----------|----------|
| **角色LoRA** | 训练特定人物/角色,保持一致性 | 15-50 | 16-32 |
| **风格LoRA** | 模仿特定艺术风格(油画/动漫等) | 30-100 | 32-64 |
| **概念LoRA** | 学习特定物体/服装/场景 | 20-80 | 16-32 |
| **品牌LoRA** | 企业视觉风格/产品风格 | 50-200 | 32-64 |
| **动作LoRA** | 特定姿势/动作 | 30-100 | 16-32 |

**典型应用**:
```
健身领域:
1. 品牌LoRA: Nike/Adidas产品风格
2. 教练LoRA: 特定教练形象一致性
3. 动作LoRA: 标准健身姿势库
4. 场馆LoRA: 特定健身房风格
```

---

## 15.2 训练环境配置

### 15.2.1 硬件要求

```yaml
最低配置 (512×512训练):
  GPU: RTX 3060 12GB
  RAM: 16GB
  存储: 50GB SSD

推荐配置 (768×768):
  GPU: RTX 4070 Ti 16GB / RTX 4090 24GB
  RAM: 32GB
  存储: 100GB NVMe SSD

企业级 (1024×1024):
  GPU: RTX 4090 24GB / A100 40GB
  RAM: 64GB
  存储: 500GB NVMe SSD
```

### 15.2.2 软件安装 (Kohya_ss GUI)

**Kohya_ss**: 最流行的LoRA训练工具

```bash
# 1. 克隆仓库
git clone https://github.com/bmaltais/kohya_ss.git
cd kohya_ss

# 2. 安装依赖 (Windows)
.\setup.bat

# 3. 启动GUI
.\gui.bat

# 4. 浏览器访问
# http://localhost:7860
```

**Linux安装**:
```bash
# 安装依赖
python -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 启动
python kohya_gui.py --listen 0.0.0.0 --server_port 7860
```

### 15.2.3 必备模型下载

```bash
# 基础模型 (选一个作为训练基底)
基础模型位置: /models/Stable-diffusion/

推荐基础模型:
1. sd_v1-5.safetensors (通用)
2. dreamshaper_8.safetensors (质量高)
3. realisticVision_v5.safetensors (写实)

# VAE (可选但推荐)
vae-ft-mse-840000-ema-pruned.safetensors
→ 放到 /models/VAE/
```

---

## 15.3 数据准备 (最关键!)

### 15.3.1 数据集质量要求

**LoRA成功90%取决于数据集质量**

```
高质量数据集特征:
✅ 主体清晰,无遮挡
✅ 光线充足,细节可见
✅ 构图多样 (正面/侧面/背面/远景/特写)
✅ 背景简洁或多样 (避免单一背景)
✅ 分辨率高 (至少512×512,推荐1024×1024)
✅ 风格统一 (同一光线质量/色调)

❌ 低质量数据:
- 模糊/低分辨率
- 过度后期(滤镜/美颜)
- 主体被遮挡
- 极端光线(过曝/欠曝)
- 水印/文字
```

**数据量指南**:
```python
DATASET_SIZE_GUIDE = {
    "角色LoRA (简单背景)": {
        "最少": 10,
        "推荐": 15-30,
        "最多": 50,
        "过量风险": "> 50可能过拟合"
    },
    "角色LoRA (复杂场景)": {
        "最少": 20,
        "推荐": 30-60,
        "最多": 100
    },
    "风格LoRA": {
        "最少": 30,
        "推荐": 50-100,
        "最多": 200
    },
    "概念LoRA (物体/服装)": {
        "最少": 15,
        "推荐": 20-50,
        "最多": 80
    }
}

# 原则: 宁少勿滥!
# 10张高质量 > 50张低质量
```

### 15.3.2 数据集组织结构

```bash
# Kohya_ss推荐结构
/training_data/
  └── 20_character_name/  # 格式: 重复次数_触发词
      ├── image001.jpg
      ├── image001.txt    # 对应标注文件
      ├── image002.jpg
      ├── image002.txt
      └── ...

# 文件命名规则
重复次数: 决定该文件夹在每个epoch被使用的次数
触发词: 生成时激活LoRA的关键词

# 示例
15_nike_trainer/    # 15次重复,"nike_trainer"是触发词
  ├── 001.jpg
  ├── 001.txt → "nike_trainer, male athlete, gym background"
  ├── 002.jpg
  ├── 002.txt → "nike_trainer, doing squat, modern gym"
  └── ...
```

**重复次数(Repeat)计算**:
```python
def calculate_repeat(total_images, target_steps):
    """
    target_steps: 目标训练步数,通常1000-3000
    """
    # 公式
    # total_steps = num_images × repeat × num_epochs / batch_size

    # 推荐
    if total_images <= 10:
        return 20  # 少量图片多重复
    elif total_images <= 20:
        return 10
    elif total_images <= 50:
        return 5
    else:
        return 1-3  # 大量图片少重复

# 示例
# 15张图,希望训练2000步,batch_size=1, epochs=10
# 2000 = 15 × repeat × 10 / 1
# repeat ≈ 13.3 → 设为13或15
```

### 15.3.3 图像标注 (Captioning)

**自动标注 vs 手动标注**:

```python
# 方法1: BLIP自动标注 (Kohya_ss内置)
步骤:
1. Kohya GUI → Utilities → Captioning
2. 选择BLIP模型
3. 指定图片文件夹
4. 点击 Caption images
5. 自动生成.txt文件

优点: 快速
缺点: 准确度有限,通用描述

# 方法2: WD14 Tagger (动漫风格更准)
# 适合动漫/插画LoRA
步骤类似,选择WD14模型

# 方法3: 手动标注 (最佳但费时)
每张图手写描述:
001.txt:
"nike_trainer, male fitness coach, wearing black Nike outfit,
 demonstrating squat form, modern gym with equipment visible,
 professional lighting, front view"

002.txt:
"nike_trainer, same person, doing deadlift,
 side view, gym background, dramatic lighting"

推荐: BLIP生成初稿 → 人工修正
```

**标注质量原则**:
```
✅ 必须包含触发词 (如"nike_trainer")
✅ 描述关键特征 (服装/姿势/背景)
✅ 保持一致性用词
✅ 避免主观词汇 ("beautiful", "amazing")

❌ 过度详细 (>50词)
❌ 不相关描述
❌ 触发词遗漏
```

---

## 15.4 训练参数配置 (核心!)

### 15.4.1 基础参数

```yaml
# Kohya_ss GUI配置

Source Model:
  Pretrained model: /models/sd_v1-5.safetensors
  V2: false  (SD 1.5用false, SD 2.x用true)
  V_parameterization: false
  SDXL: false (训练SDXL LoRA时选true)

Folders:
  Image folder: /training_data/  (包含重复次数_触发词文件夹)
  Output folder: /output/lora/
  Logging folder: /logs/

Model:
  Save every N epochs: 1  (每个epoch保存一次)
  Max train epoch: 10  (总epoch数, 推荐8-15)
  Save precision: fp16

Network:
  Network Rank (dim): 32  # LoRA秩,核心参数!
  Network Alpha: 16  # 通常 = dim/2
  Network module: networks.lora  # 固定
  Training comment: "Nike Trainer LoRA v1"

Resolution:
  Max resolution: 512,512  # 训练分辨率
  # 512×512 (入门), 768×768 (推荐), 1024×1024 (高质量)
  Enable buckets: true  # 自动调整不同比例图片
```

**关键参数详解**:

**1. Network Rank (dim)**
```python
# LoRA的核心参数,决定模型容量

dim值建议:
- dim=4:   极简,仅学习基本特征,文件<5MB
- dim=8:   简单角色/风格,文件~10MB
- dim=16:  标准角色,文件~20MB
- dim=32:  复杂角色/详细风格,文件~40MB (推荐)
- dim=64:  极高细节,文件~80MB
- dim=128: 非常复杂场景,文件~150MB (可能过拟合)

选择原则:
- 简单角色(卡通/简化风格): dim=16
- 写实人物: dim=32
- 复杂艺术风格: dim=64
- 不确定时: 从32开始
```

**2. Network Alpha**
```python
# 控制LoRA权重缩放

公式: alpha / dim = 缩放因子

常见配置:
- dim=32, alpha=16 → 缩放=0.5
- dim=32, alpha=32 → 缩放=1.0
- dim=64, alpha=32 → 缩放=0.5

推荐: alpha = dim / 2 (经验值)
```

**3. Learning Rate (学习率)**
```yaml
Text Encoder learning rate: 5e-5  # CLIP文本编码器
Unet learning rate: 1e-4  # U-Net (主要)

# 黄金组合 (SD 1.5)
text_encoder_lr: 5e-5
unet_lr: 1e-4  (或 5e-5)

# SDXL
text_encoder_lr: 3e-5
unet_lr: 5e-5

# 调整原则:
# 过拟合 (细节过度,泛化差) → 降低LR
# 欠拟合 (学不到特征) → 提高LR
```

**4. Batch Size & Gradient Accumulation**
```yaml
Train batch size: 1  # 显存有限时用1
Gradient accumulation steps: 1  # 梯度累积

# 显存充足时:
batch_size: 2-4
gradient_accumulation: 2-4

# 等效batch_size = batch_size × gradient_accumulation
# 如: batch=1, accum=4 ≈ batch=4, accum=1 (但显存需求不同)

# 12GB显存推荐:
512×512: batch=2, accum=2
768×768: batch=1, accum=4
1024×1024: batch=1, accum=1
```

### 15.4.2 优化器选择

```yaml
Optimizer: AdamW8bit  # 推荐,显存友好

可选:
- AdamW8bit: 默认推荐,速度和质量平衡
- AdamW: 标准优化器,质量略高但显存多
- Lion: 新优化器,有时质量更好
- Prodigy: 自适应学习率,实验性

推荐配置:
Optimizer: AdamW8bit
LR Scheduler: cosine  # 学习率调度器
LR warmup steps: 100  # 前100步预热
```

### 15.4.3 高级参数

```yaml
Advanced:
  Clip skip: 2  # CLIP层跳过,SD 1.5推荐2
  Mixed precision: fp16  # 混合精度训练
  xformers: true  # 显存优化
  Cache latents: true  # 缓存VAE输出,加速训练
  Color aug: false  # 颜色增强,通常不用
  Flip aug: false  # 水平翻转增强,人物不推荐

Noise offset:
  Noise offset: 0.0  # 噪声偏移,通常0
  # 如果图像整体偏暗/偏亮,可设为0.03-0.1

Min SNR gamma:
  Min SNR gamma: 5  # 信噪比,推荐5或不设

Sample:
  Sample every N epochs: 1  # 每epoch生成样本图
  Sample prompts: |
    nike_trainer, standing in gym, front view --n low quality --s 30 --w 512 --h 512
    nike_trainer, doing squat --n low quality --s 30 --w 512 --h 512
  # 训练时自动生成测试图,监控训练进度
```

---

## 15.5 训练流程

### 15.5.1 完整训练步骤

```bash
# 步骤1: 准备数据
/training_data/
  └── 20_nike_trainer/
      ├── 001.jpg (1024×1024)
      ├── 001.txt ("nike_trainer, male coach, gym...")
      ├── 002.jpg
      ├── 002.txt
      └── ... (共15张)

# 步骤2: 配置参数 (Kohya GUI)
Source Model: sd_v1-5.safetensors
Image folder: /training_data/
Output folder: /output/lora/
Epochs: 10
Dim: 32
Alpha: 16
Resolution: 768×768
Batch size: 1
Gradient accumulation: 4
Unet LR: 1e-4
Text Encoder LR: 5e-5
Optimizer: AdamW8bit
LR Scheduler: cosine

# 步骤3: 开始训练
点击 "Start training"

# 步骤4: 监控
查看logs:
- Loss曲线 (应该逐渐下降)
- 每epoch的样本图 (质量逐渐提升)

# 步骤5: 结束
训练完成后,/output/lora/包含:
- nike_trainer_000010.safetensors (第10 epoch)
- nike_trainer_000009.safetensors (第9 epoch)
- ...
```

### 15.5.2 训练监控

```python
# 判断训练是否成功

✅ 成功迹象:
1. Loss稳步下降
   - 初始loss: 0.08-0.12
   - 最终loss: 0.04-0.07
   - 曲线平滑下降

2. 样本图质量提升
   - 早期epoch: 模糊,特征不明显
   - 中期: 逐渐清晰,开始像目标
   - 后期: 清晰,高度相似

3. 过拟合检测
   - 样本图过度复现训练图 → 过拟合
   - 背景/服装完全固定 → 过拟合
   - 解决: 降低epoch数或LR

⚠️ 问题迹象:
1. Loss不下降/震荡剧烈
   → LR过高,降至5e-5

2. Loss下降过快 (1-2 epoch到0.02)
   → 过拟合,减少epoch

3. 样本图一直模糊
   → LR过低或dim太小,提升参数

4. 样本图出现artifacts/噪点
   → batch size太小,增加gradient accumulation
```

### 15.5.3 选择最佳Epoch

```python
# 不一定最后一个epoch最好!

评估方法:
1. 每个epoch的样本图对比
2. 找到"甜点" (sweet spot)

典型曲线:
Epoch 1-3: 学习基本特征,较模糊
Epoch 4-7: 快速提升,质量最佳 ← 通常最佳在这里!
Epoch 8-10: 可能过拟合,泛化能力下降

推荐:
- 保留所有epoch的safetensors
- 逐一测试
- 选择泛化性最好的 (不仅训练集像,新提示词也好用)
```

---

## 15.6 LoRA使用

### 15.6.1 WebUI加载LoRA

```yaml
# 1. 复制LoRA文件
nike_trainer_000007.safetensors
→ 移动到: /stable-diffusion-webui/models/Lora/

# 2. 刷新WebUI
点击Lora标签下的刷新按钮

# 3. 使用
提示词:
<lora:nike_trainer_000007:0.8>, nike_trainer, male coach in gym, demonstrating exercise

语法:
<lora:文件名(无扩展名):权重>

权重范围:
0.3-0.6: 轻微影响
0.7-0.9: 标准强度 (推荐)
1.0-1.2: 强烈影响
>1.5: 可能过度,产生artifacts
```

**示例提示词**:
```bash
# 正面照
<lora:nike_trainer:0.8>, nike_trainer, professional fitness coach,
wearing black Nike training outfit, standing in modern gym,
front view, confident pose, professional photography

# 动作示范
<lora:nike_trainer:0.9>, nike_trainer, demonstrating squat exercise,
correct form, side view, gym background with equipment,
instructional photography style

# 不同场景测试泛化性
<lora:nike_trainer:0.8>, nike_trainer, outdoor park setting,
morning sunlight, casual sportswear, jogging

# 如果泛化性差 (只能在健身房),说明过拟合,需重新训练
```

### 15.6.2 ComfyUI加载LoRA

```json
{
  "LoraLoader": {
    "model": "checkpoint_model",
    "clip": "checkpoint_clip",
    "lora_name": "nike_trainer_000007.safetensors",
    "strength_model": 0.8,  # 应用到模型的强度
    "strength_clip": 0.8    # 应用到CLIP的强度
  }
}

# 可单独调节model和clip强度
# 通常保持一致,特殊情况可分开调
```

### 15.6.3 多LoRA组合

```bash
# 可同时使用多个LoRA

提示词:
<lora:nike_trainer:0.8><lora:modern_gym_style:0.6>, nike_trainer in stylish gym

注意:
1. 总权重不宜过高 (如3个LoRA各1.0 = 3.0过高)
2. 相似LoRA可能冲突
3. 推荐最多同时3个LoRA
```

---

## 15.7 常见问题排查

### 问题1: 训练后LoRA无效

```
症状: 加载LoRA后生成图无任何变化

排查:
1. 触发词是否正确?
   → 检查训练时的文件夹名 (如20_nike_trainer)
   → 提示词必须包含"nike_trainer"

2. LoRA权重是否太低?
   → 提升到0.8-1.0测试

3. LoRA文件是否损坏?
   → 检查文件大小 (应该10-100MB)
   → 重新训练

4. 基础模型不匹配?
   → SD 1.5训练的LoRA不能用于SDXL
   → 确认训练和使用的模型版本一致
```

### 问题2: 过拟合

```
症状:
- 只能复现训练图,换提示词失败
- 背景/服装固定,无法改变
- 新姿势/角度完全失败

原因:
- Epoch过多 (>15)
- 数据集过小 (<10张)
- 数据集单一 (同一背景/服装)
- LR过高

解决:
1. 减少epoch (10→6)
2. 增加数据多样性
3. 降低LR (1e-4 → 5e-5)
4. 使用Dropout (如果支持)
```

### 问题3: 欠拟合

```
症状:
- 训练后特征学习不足
- 生成图与目标差异大
- 触发词效果微弱

原因:
- Epoch太少 (<5)
- LR太低
- Dim太小 (如dim=8学习复杂风格)
- 数据质量差

解决:
1. 增加epoch (10→15)
2. 提升LR (5e-5 → 1e-4)
3. 增大dim (16→32)
4. 提升数据质量
```

### 问题4: 训练崩溃/OOM

```
症状: CUDA out of memory

解决:
1. 降低分辨率 (768→512)
2. 减小batch size (2→1)
3. 启用gradient checkpointing
4. 使用xformers
5. Cache latents
6. 混合精度fp16
```

---

## 15.8 高级技巧

### 15.8.1 数据增强策略

```python
# Kohya_ss支持的增强

Color aug: false  # 颜色随机化,通常不用
Flip aug: true    # 水平翻转

# 何时使用flip?
✅ 使用: 对称物体 (建筑/风景/对称服装)
❌ 不用: 人脸/文字/品牌Logo (翻转后错误)

# 健身LoRA:
# 大部分动作可翻转 (深蹲/硬拉)
# 但人脸最好不翻转 (保持一致性)
```

### 15.8.2 正则化图像

```bash
# 防止模型"忘记"基础知识

设置:
Regularization images folder: /reg_images/500_person/

内容:
- 500张普通人物图 (从基础模型生成)
- 防止模型只认识"nike_trainer",忘记其他人

使用场景:
- 训练数据<20张时推荐
- 避免过度特化

生成正则图:
# 用基础模型生成500张:
Prompt: "person, various poses, different backgrounds"
→ 保存到/reg_images/500_person/
```

### 15.8.3 Tag权重调整

```bash
# 标注文件中可使用权重语法

001.txt:
nike_trainer, (black Nike outfit:1.2), gym background,
(professional lighting:1.3), front view, (confident expression:0.8)

语法:
(tag:1.2) - 提升权重20%
(tag:0.8) - 降低权重20%

用途:
- 强调核心特征 (如品牌服装)
- 弱化次要特征 (如背景细节)
```

---

## 15.9 实战项目: 训练健身品牌LoRA

```python
# 项目: 训练Nike健身教练风格LoRA

# 步骤1: 数据收集 (20张)
数据要求:
- 10张 正面/侧面 站立姿势
- 5张 健身动作 (深蹲/硬拉/卧推等)
- 5张 不同角度/光线

# 步骤2: 预处理
import os
from PIL import Image

def preprocess_images(input_dir, output_dir, target_size=768):
    """裁剪为正方形并调整大小"""
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.png')):
            img = Image.open(os.path.join(input_dir, filename))

            # 中心裁剪为正方形
            width, height = img.size
            min_side = min(width, height)
            left = (width - min_side) / 2
            top = (height - min_side) / 2
            right = left + min_side
            bottom = top + min_side
            img_cropped = img.crop((left, top, right, bottom))

            # 调整大小
            img_resized = img_cropped.resize((target_size, target_size), Image.LANCZOS)

            # 保存
            output_path = os.path.join(output_dir, filename)
            img_resized.save(output_path, quality=95)
            print(f"Processed: {filename}")

preprocess_images("raw_images/", "training_data/15_nike_coach/")

# 步骤3: 自动标注 (BLIP) + 手动修正
# Kohya GUI → Utilities → Captioning → BLIP
# 然后手动添加触发词"nike_coach"

# 步骤4: 训练配置
config = {
    "model": "sd_v1-5.safetensors",
    "image_folder": "training_data/",
    "output_folder": "output/nike_coach_lora/",
    "epochs": 10,
    "dim": 32,
    "alpha": 16,
    "resolution": "768,768",
    "batch_size": 1,
    "gradient_accumulation": 4,
    "unet_lr": 1e-4,
    "text_encoder_lr": 5e-5,
    "optimizer": "AdamW8bit",
    "scheduler": "cosine",
    "sample_prompts": """
        nike_coach, standing in gym, professional photography --n low quality --s 30 --w 768 --h 768
        nike_coach, demonstrating squat, side view, gym --n low quality --s 30 --w 768 --h 768
        nike_coach, outdoor setting, casual pose --n low quality --s 30 --w 768 --h 768
    """
}

# 步骤5: 开始训练
# 使用Kohya GUI配置上述参数,点击Start training

# 步骤6: 评估 (训练完成后)
# 测试各epoch,选择最佳:
# - epoch 5-8通常质量最好
# - 对比泛化能力 (不同场景/服装是否都可用)

# 步骤7: 最终测试
测试提示词:
1. "nike_coach, gym setting" (基础测试)
2. "nike_coach, outdoor park" (泛化测试)
3. "nike_coach, wearing casual clothes" (服装变化)
4. "nike_coach, doing yoga pose" (未训练的动作)

# 如果1-4都成功 → LoRA训练成功!
# 如果3-4失败 → 过拟合,需重新训练
```

---

## 15.10 总结

### 15.10.1 LoRA训练成功要诀

```
1. 数据质量 > 数据数量
   10张高质量 > 50张低质量

2. 适度训练,避免过拟合
   Epoch 8-12通常最佳

3. 参数从标准开始,再微调
   dim=32, alpha=16, lr=1e-4

4. 监控样本图,及时止损
   过拟合迹象立即停止

5. 测试泛化性
   不仅训练场景,新场景也要测试
```

### 15.10.2 学习路径

```
Week 1: 基础训练
- Day 1-2: 环境配置
- Day 3-4: 简单角色LoRA (10张图)
- Day 5-7: 参数调优实验

Week 2: 进阶应用
- Day 8-10: 风格LoRA (50-100张图)
- Day 11-12: 多LoRA组合
- Day 13-14: 实战项目 (品牌/角色)
```

---

## 15.11 参考资源

- [Kohya_ss GitHub](https://github.com/bmaltais/kohya_ss)
- [LoRA论文](https://arxiv.org/abs/2106.09685)
- [Civitai LoRA社区](https://civitai.com)

**下一章预告**: 第21章将进入视频生成领域,学习Runway Gen-3的图生视频技术。

LoRA是SD生态中最实用的定制化工具,掌握它将让你拥有独一无二的生成能力!
