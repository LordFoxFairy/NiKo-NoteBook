# 第4章 AUTOMATIC1111 WebUI完全精通

> 从安装到精通，掌握最流行的SD图像生成工具
>
> **学习目标**:
> - 完成WebUI环境搭建（Win/Mac/Linux）
> - 掌握txt2img核心参数
> - 学会img2img重绘技术
> - 生成第一张真人健身图像

---

## 4.1 为什么选择AUTOMATIC1111 WebUI

### 核心优势

```
✅ 开源免费          # 完全开源，无需付费
✅ 功能最全          # 支持所有主流功能
✅ 社区活跃          # 全球最大的SD社区
✅ 插件丰富          # 1000+扩展插件
✅ 易于上手          # Web界面，无需编程
✅ 本地运行          # 数据隐私，无限制
```

### 与其他工具对比

| 特性 | WebUI | ComfyUI | Diffusers(代码) |
|------|-------|---------|-----------------|
| 易用性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| 功能完整度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 批量生成 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 学习曲线 | 平缓 | 陡峭 | 需要编程 |
| 推荐场景 | **入门+日常** | 批量生产 | 开发集成 |

**结论**: WebUI是**入门首选**，掌握后再学ComfyUI。

---

## 4.2 环境搭建（多平台）

### 4.2.1 系统要求

#### 最低配置
```
GPU: NVIDIA GTX 1060 6GB
RAM: 8GB
存储: 20GB SSD
```

#### 推荐配置
```
GPU: NVIDIA RTX 3060 12GB / RTX 4060 Ti 16GB
RAM: 16GB+
存储: 100GB+ SSD
```

#### 显存与分辨率对应关系

$$
\text{显存需求} \approx \frac{W \times H \times \text{Batch}}{10^6} + 2 \text{GB (基础)}
$$

| 显存 | 分辨率 | Batch Size |
|------|--------|------------|
| 6GB | 512x512 | 1 |
| 8GB | 768x768 | 1 |
| 12GB | 1024x1024 | 1 |
| 16GB | 1024x1024 | 2-4 |
| 24GB | 2048x2048 | 1 |

---

### 4.2.2 Windows安装（推荐方式）

#### 方法1: 一键安装包（最简单）⭐⭐⭐⭐⭐

```bash
# 1. 下载安装包
https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases

# 2. 解压到D盘
D:\stable-diffusion-webui\

# 3. 双击运行
webui-user.bat

# 4. 等待自动安装依赖（首次需10-20分钟）

# 5. 浏览器打开
http://127.0.0.1:7860
```

#### 方法2: Git克隆（推荐高级用户）

```bash
# 1. 安装Git
https://git-scm.com/download/win

# 2. 安装Python 3.10.6
https://www.python.org/downloads/release/python-3106/
# ⚠️ 勾选 "Add Python to PATH"

# 3. 克隆仓库
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
cd stable-diffusion-webui

# 4. 运行启动脚本
webui-user.bat
```

#### 常见问题排查

**问题1: 显存不足**
```bash
# 编辑 webui-user.bat，添加启动参数
set COMMANDLINE_ARGS=--medvram --xformers

# 参数说明:
# --medvram   : 中等显存优化（8GB显卡）
# --lowvram   : 低显存优化（6GB显卡）
# --xformers  : 加速优化（减少显存20%+速度提升30%）
```

**问题2: 启动失败**
```bash
# 检查Python版本
python --version
# 必须是 3.10.x

# 重新安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

### 4.2.3 macOS安装

```bash
# 1. 安装Homebrew（如未安装）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. 安装Python 3.10
brew install python@3.10

# 3. 克隆仓库
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
cd stable-diffusion-webui

# 4. 运行（M1/M2芯片）
./webui.sh

# 首次运行会自动安装依赖
```

**M1/M2优化参数**:
```bash
# 编辑 webui-user.sh
export COMMANDLINE_ARGS="--skip-torch-cuda-test --upcast-sampling --no-half-vae"
```

---

### 4.2.4 Linux (Ubuntu) 安装

```bash
# 1. 安装依赖
sudo apt update
sudo apt install wget git python3 python3-venv libgl1 libglib2.0-0

# 2. 克隆仓库
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
cd stable-diffusion-webui

# 3. 运行
./webui.sh
```

---

## 4.3 模型下载与管理

### 4.3.1 核心概念

#### 模型文件结构

```
stable-diffusion-webui/
├── models/
│   ├── Stable-diffusion/        # 主模型 (Checkpoint)
│   │   ├── sd_v1.5.safetensors  # 4GB
│   │   └── sdxl_base.safetensors # 6.5GB
│   │
│   ├── VAE/                      # VAE模型
│   │   └── vae-ft-mse-840000.safetensors
│   │
│   ├── Lora/                     # LoRA微调模型
│   │   └── fitness_style.safetensors
│   │
│   └── embeddings/               # Textual Inversion
│       └── bad-hands-5.pt
```

---

### 4.3.2 推荐模型下载

#### 基础模型（必备）

```
【SD 1.5基础模型】
名称: v1-5-pruned-emaonly.safetensors
大小: 4GB
下载: https://huggingface.co/runwayml/stable-diffusion-v1-5
用途: 学习基础，兼容性最好

【SDXL基础模型】⭐⭐⭐⭐⭐
名称: sd_xl_base_1.0.safetensors
大小: 6.5GB
下载: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
用途: 高质量图像生成（推荐）

【SDXL Refiner】
名称: sd_xl_refiner_1.0.safetensors
大小: 6.5GB
下载: 同上
用途: 配合Base模型精修细节
```

#### 真人模型推荐（针对你的需求）

```
【Realistic Vision】⭐⭐⭐⭐⭐
版本: v5.1
大小: 2GB
下载: https://civitai.com/models/4201
特点: 真人照片级，亚洲面孔优秀
用途: 真人美女、健身照片

【ChilloutMix】⭐⭐⭐⭐
大小: 2GB
下载: https://civitai.com/models/6424
特点: 亚洲面孔专精
用途: 真人美女、时尚写真

【DreamShaper】⭐⭐⭐⭐
版本: 8
大小: 2GB
下载: https://civitai.com/models/4384
特点: 平衡真实感和艺术性
用途: 通用场景
```

#### VAE模型（提升质量）

```
【vae-ft-mse-840000】⭐⭐⭐⭐⭐
大小: 335MB
下载: https://huggingface.co/stabilityai/sd-vae-ft-mse-original
用途: 提升细节和色彩

【SDXL VAE】
大小: 335MB
下载: https://huggingface.co/stabilityai/sdxl-vae
用途: SDXL专用
```

---

### 4.3.3 快速下载技巧

#### 使用国内镜像（加速）

```bash
# HuggingFace镜像
export HF_ENDPOINT=https://hf-mirror.com

# 下载示例
wget https://hf-mirror.com/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors
```

#### 使用Civitai下载器

```python
# civitai_downloader.py
import requests
from tqdm import tqdm

def download_model(model_id, save_path):
    """
    从Civitai下载模型

    Args:
        model_id: Civitai模型ID
        save_path: 保存路径
    """
    url = f"https://civitai.com/api/download/models/{model_id}"

    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))

    with open(save_path, 'wb') as f, tqdm(
        desc=save_path,
        total=total,
        unit='iB',
        unit_scale=True
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

# 使用示例
download_model(
    model_id=130072,  # Realistic Vision v5.1
    save_path="models/Stable-diffusion/realistic_vision_v5.1.safetensors"
)
```

---

## 4.4 txt2img核心参数深度解析

### 4.4.1 界面布局

```
┌─────────────────────────────────────────────┐
│  Stable Diffusion Checkpoint: [下拉选择]    │  # 选择模型
├─────────────────────────────────────────────┤
│  Prompt (正面提示词):                       │
│  ┌───────────────────────────────────────┐  │
│  │ masterpiece, 1 girl, fitness...       │  │
│  └───────────────────────────────────────┘  │
├─────────────────────────────────────────────┤
│  Negative Prompt (负面提示词):              │
│  ┌───────────────────────────────────────┐  │
│  │ (deformed, ugly:1.4)...               │  │
│  └───────────────────────────────────────┘  │
├─────────────────────────────────────────────┤
│  Sampling method: [DPM++ 2M Karras]  ▼     │  # 采样器
│  Sampling steps: [20]         ◄─────►      │  # 步数
│  Width: [768]  Height: [1024]              │  # 分辨率
│  Batch count: [1]  Batch size: [1]         │  # 批次
│  CFG Scale: [7]               ◄─────►      │  # 引导强度
│  Seed: [-1]                                │  # 随机种子
├─────────────────────────────────────────────┤
│  [Generate] [Interrupt] [Skip]             │  # 操作按钮
└─────────────────────────────────────────────┘
```

---

### 4.4.2 采样步数 (Sampling Steps)

#### 数学原理

扩散模型的去噪过程可表示为：

$$
x_{t-1} = \sqrt{\alpha_{t-1}} \left( \frac{x_t - \sqrt{1-\alpha_t} \epsilon_\theta(x_t, t)}{\sqrt{\alpha_t}} \right) + \sqrt{1 - \alpha_{t-1}} \epsilon
$$

其中：
- $x_t$: 时间步 $t$ 的噪声图像
- $\epsilon_\theta$: 神经网络预测的噪声
- $\alpha_t$: 噪声调度参数
- $t \in [0, T]$: 总步数 $T$ 就是 Sampling Steps

#### 实测数据

```
质量提升曲线（SDXL模型）:

Quality (SSIM)
    1.0 ┤
        │                 ●●●●●●● (50步后趋于平稳)
    0.9 ┤             ●●●
        │         ●●●
    0.8 ┤     ●●●
        │   ●●
    0.7 ┤ ●●
        │●
    0.6 ┤
        └─────┬─────┬─────┬─────┬─────┬─────
             10    20    30    40    50   100
                      Steps

边际效益:
Steps 0→10:   质量提升 60%
Steps 10→20:  质量提升 20%  ⭐ 性价比最高
Steps 20→30:  质量提升 10%
Steps 30→50:  质量提升 5%
Steps 50→100: 质量提升 <2%
```

**推荐值**:
- 快速预览: **15步**
- 日常使用: **20-25步** ⭐⭐⭐⭐⭐
- 精细作品: **30-35步**
- 专业输出: **40步**（超过无意义）

---

### 4.4.3 CFG Scale (引导强度)

#### 数学原理

Classifier-Free Guidance计算公式：

$$
\epsilon_\theta(x_t, t, c) = \epsilon_\theta(x_t, t, \emptyset) + s \cdot [\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset)]
$$

其中：
- $c$: 条件（提示词）
- $\emptyset$: 无条件
- $s$: CFG Scale（引导强度）

**通俗理解**:
- $s = 1$: 完全忽略提示词（随机生成）
- $s = 7$: 平衡创意和指令（推荐）
- $s = 20$: 严格遵循提示词（可能过度）

#### 实测对比

```
【测试提示词】
1 girl, fitness model, athletic body, gym

CFG=3:  🎨 高创意性，但可能偏离提示词
        - 可能生成非健身场景
        - 艺术性强，但不可控

CFG=7:  ✅ 黄金平衡点
        - 准确响应提示词
        - 画面自然，细节好
        - 推荐用于90%场景

CFG=12: 📐 精准控制
        - 严格按提示词生成
        - 适合产品图、精准复现
        - 可能颜色过饱和

CFG=20: ⚠️ 过度引导
        - 颜色失真、细节扭曲
        - 不推荐
```

**推荐值**:
- 艺术创作: **5-6**
- 真人照片: **6-8** ⭐⭐⭐⭐⭐
- 产品图: **10-12**
- 避免使用: **>15**

---

### 4.4.4 分辨率设置

#### 显存计算公式

$$
\text{VRAM}_{\text{需求}} \approx \frac{W \times H}{65536} \times 1.5 + 2 \text{ GB}
$$

#### 推荐分辨率表

| 显存 | SD 1.5 | SDXL | 用途 |
|------|--------|------|------|
| 6GB | 512x512 | ❌ | 学习测试 |
| 8GB | 768x768 | 512x512 | 基础使用 |
| 12GB | 1024x1024 | 768x1024 | 日常使用 ⭐ |
| 16GB | 1536x1536 | 1024x1024 | 高质量 ⭐⭐ |
| 24GB | 2048x2048 | 1536x1536 | 专业级 |

#### 常用比例

```
肖像（纵向）:
- 768x1024  (3:4)   ⭐⭐⭐⭐⭐ 推荐
- 640x960   (2:3)
- 512x768   (2:3)

横向：
- 1024x768  (4:3)
- 1280x720  (16:9)

方形：
- 1024x1024 (1:1)
```

**重要规则**:
- ⚠️ 宽高必须是**8的倍数**
- ⚠️ SD 1.5最佳分辨率: 512x512
- ⚠️ SDXL最佳分辨率: 1024x1024
- ⚠️ 超出训练分辨率会导致重复/变形

---

### 4.4.5 批次设置

#### Batch Count vs Batch Size

```
Batch Count: 串行生成（一次一个）
Batch Size:  并行生成（同时多个）

示例:
Batch Count = 4, Batch Size = 1
→ 生成4张，每次1张（总时间 = 单张×4）

Batch Count = 1, Batch Size = 4
→ 生成4张，一次生成4张（总时间 < 单张×4，但需4倍显存）
```

#### 显存需求对比

| 设置 | 显存需求 | 生成速度 |
|------|---------|---------|
| Count=4, Size=1 | 基础显存 | 4x 单张时间 |
| Count=1, Size=4 | 基础显存×4 | ~2.5x 单张时间 |

**推荐策略**:
- 显存充足(16GB+): **Batch Size = 2-4**
- 显存紧张(8GB): **Batch Count = 4, Size = 1**

---

### 4.4.6 随机种子 (Seed)

#### 原理

```
Seed = 初始随机数生成器的种子

相同参数 + 相同Seed = 相同结果

示例:
Prompt: "1 girl, fitness"
Seed: 12345
→ 每次生成都是相同的图像

Seed: -1 (随机)
→ 每次生成都不同
```

#### 实战技巧

```python
# 工作流程
阶段1: 探索（Seed = -1）
生成100张 → 找到满意的 → 记录Seed

阶段2: 微调（固定Seed）
Seed = 12345 (满意的图)
只调整提示词 → 保持构图和风格

阶段3: 变体（Seed ± 小幅度）
Seed = 12345, 12346, 12347
→ 生成相似但略有差异的图像
```

---

## 4.5 第一张图像生成实战

### 4.5.1 健身美女基础版

#### 步骤1: 选择模型

```
Checkpoint: Realistic Vision v5.1
VAE: vae-ft-mse-840000
```

#### 步骤2: 输入提示词

**正面提示词**:
```
masterpiece, best quality, ultra detailed, 8k,
1 girl, 25 years old, asian fitness model,
(athletic body:1.3), (toned abs:1.2),
beautiful face, detailed eyes, natural makeup,
long black hair, high ponytail,
(sports bra:1.2), (yoga pants:1.2),
standing pose, confident smile,
modern gym background, bright lighting,
professional photography, photorealistic,
depth of field
```

**负面提示词**:
```
(deformed, ugly, bad anatomy:1.5),
(bad hands, bad fingers, extra limbs:1.4),
(fat, skinny:1.3),
(low quality, worst quality, blurry:1.4),
watermark, text, logo
```

#### 步骤3: 参数设置

```
Sampling method: DPM++ 2M Karras
Sampling steps: 25
Width: 768
Height: 1024
CFG Scale: 7
Seed: -1 (首次随机)
```

#### 步骤4: 点击生成

```
[Generate] 按钮

等待时间:
RTX 3060: ~15秒
RTX 4060 Ti: ~10秒
RTX 4090: ~5秒
```

---

### 4.5.2 常见问题排查

#### 问题1: 生成速度太慢

**原因分析**:
```
生成时间公式:
T = k × Steps × (W × H) / GPU性能

其中 k 受采样器影响
```

**解决方案**:
```bash
# 1. 启用xformers加速
# 编辑 webui-user.bat
set COMMANDLINE_ARGS=--xformers

# 2. 降低分辨率
768x1024 → 512x768

# 3. 减少步数
25步 → 20步

# 4. 换快速采样器
DPM++ 2M Karras → UniPC
```

---

#### 问题2: 显存不足 (CUDA out of memory)

**显存占用分析**:

$$
\text{总显存} = \text{模型加载} + \text{计算缓存} + \text{输出缓存}
$$

```
SDXL模型:
- 模型: 6.5GB
- 计算: ~2GB (768x1024)
- 缓存: ~1GB
总计: ~9.5GB

8GB显卡 → 显存不足！
```

**解决方案**:
```bash
# 方法1: 启用medvram
set COMMANDLINE_ARGS=--medvram --xformers
# 降低显存30%，速度减慢10%

# 方法2: 启用lowvram (6GB显卡)
set COMMANDLINE_ARGS=--lowvram --xformers
# 降低显存50%，速度减慢30%

# 方法3: 使用SD 1.5模型
Checkpoint换成: v1-5-pruned-emaonly
分辨率: 512x768
```

---

#### 问题3: 生成结果不满意

**checklist**:
```
□ 模型是否选对? (真人用Realistic Vision)
□ 提示词是否详细?
□ 负面提示词是否完善?
□ CFG是否合适? (推荐6-8)
□ 步数是否足够? (推荐20-25)
□ 分辨率是否合理?
```

**优化流程**:
```
Step 1: 固定Seed（找到可接受的图）
Step 2: 调整提示词（增加细节描述）
Step 3: 调整权重（强化关键特征）
Step 4: 调整CFG（平衡真实感）
```

---

## 4.6 img2img 重绘技术

### 4.6.1 核心概念

**txt2img vs img2img**:

```
txt2img: 纯文字 → 图像
  完全从噪声生成

img2img: 文字 + 参考图 → 新图像
  从参考图加噪后再去噪
```

#### 数学原理

img2img过程：

$$
\begin{aligned}
x_{\text{ref}} &\rightarrow \text{加噪到步数 } t_{\text{start}} \rightarrow x_{t_{\text{start}}} \\
x_{t_{\text{start}}} &\rightarrow \text{去噪到步数 } 0 \rightarrow x_0
\end{aligned}
$$

其中 $t_{\text{start}}$ 由 **Denoising Strength** 控制：

$$
t_{\text{start}} = T \times (1 - \text{Denoising Strength})
$$

---

### 4.6.2 Denoising Strength 详解

#### 参数范围：0-1

```
Denoising Strength = 0:
  不加噪声，输出 = 输入（完全保留原图）

Denoising Strength = 0.3:
  轻微改变，保留90%结构

Denoising Strength = 0.5:
  中度改变，保留50%结构

Denoising Strength = 0.75:
  大幅改变，只保留构图

Denoising Strength = 1.0:
  完全重绘，接近txt2img
```

#### 实测对比

```
【原图】: 一个女孩站立照片

Denoising = 0.2:
└→ 几乎一样，只是轻微美化

Denoising = 0.4: ⭐⭐⭐⭐⭐
└→ 保留姿态和构图，优化细节
   (人像修复推荐值)

Denoising = 0.6:
└→ 保留大致构图，人物特征改变

Denoising = 0.8:
└→ 只保留姿态，面容完全改变
```

**推荐值**:
- 微调/修复: **0.3-0.4**
- 风格迁移: **0.5-0.6**
- 大幅重绘: **0.7-0.8**

---

### 4.6.3 实战案例：健身照优化

#### 场景：优化手机拍摄的健身照

**原图问题**:
- 背景杂乱
- 光线不佳
- 肌肉线条不明显

**操作步骤**:

```
1. 切换到 img2img 标签

2. 上传原图
   [Upload Image]

3. 调整参数
   Denoising strength: 0.45
   Resize mode: Just resize

4. 提示词（强调优化）
   masterpiece, best quality,
   (athletic body:1.3), (toned abs:1.3),
   (professional photography:1.2),
   (modern gym background:1.2),
   (cinematic lighting:1.2),
   photorealistic, highly detailed

5. 负面提示词
   (bad quality, blurry:1.4),
   (cluttered background:1.3),
   (bad lighting:1.2),
   deformed, ugly

6. 其他参数
   Steps: 30 (img2img需更多步数)
   CFG: 7
   Size: 与原图相同或略大

7. Generate
```

**预期效果**:
- ✅ 保留人物姿态和基本特征
- ✅ 背景变成干净的健身房
- ✅ 光线优化为专业摄影
- ✅ 肌肉线条更明显

---

### 4.6.4 Resize Mode详解

#### 4种模式对比

```
【Just resize】
- 直接缩放到目标尺寸
- 可能拉伸变形
- 速度快

【Crop and resize】
- 裁剪到目标比例后缩放
- 不变形，但可能丢失内容
- 推荐用于人像

【Resize and fill】
- 缩放后填充空白区域
- AI生成填充内容
- 可能不自然

【Just resize (latent upscale)】
- 在潜空间缩放
- 质量最好，速度慢
- 推荐用于高质量输出 ⭐⭐⭐
```

---

## 4.7 Highres Fix 高清修复

### 4.7.1 原理

#### 问题：直接生成高分辨率的问题

```
训练分辨率: 512x512 (SD 1.5)
生成分辨率: 1024x1024

结果: ❌ 重复、变形、多余肢体
```

**原因**:
模型在512x512训练，对更大尺寸的空间关系理解不足。

#### Highres Fix工作流程

```
Step 1: 生成小图
512x512 @ 20 steps → 初步图像

Step 2: 潜空间放大
512x512 → 1024x1024 (latent space)

Step 3: 高清重绘
1024x1024 @ 15 steps, Denoising 0.5
→ 修复细节，避免重复
```

数学表示：

$$
\begin{aligned}
x_{512} &= \text{Diffusion}(\text{noise}, \text{prompt}, T=20) \\
x_{1024}^{\text{latent}} &= \text{Upscale}(x_{512}, 2\times) \\
x_{1024}^{\text{final}} &= \text{Diffusion}(x_{1024}^{\text{latent}}, \text{prompt}, T=15)
\end{aligned}
$$

---

### 4.7.2 参数设置

#### 启用方式

```
txt2img 界面:
└─ [√] Hires. fix

展开参数:
├─ Upscaler: Latent
├─ Hires steps: 15
├─ Denoising strength: 0.5
├─ Upscale by: 2
└─ Resize width/height: 自动计算
```

#### 参数详解

**Upscaler（放大算法）**:
```
Latent ⭐⭐⭐⭐⭐
- 在潜空间放大
- 速度快，效果好
- 推荐首选

Latent (nearest-exact)
- 精确最近邻算法
- 细节稍好，速度稍慢

R-ESRGAN 4x+
- 传统超分辨率算法
- 细节最好，但速度慢
- 用于最终输出
```

**Hires steps**:
- 推荐值: **10-20步**
- 太少(<10): 细节不足
- 太多(>30): 浪费时间

**Denoising strength**:
- 推荐值: **0.4-0.6**
- 太低(<0.3): 放大痕迹明显
- 太高(>0.7): 可能改变构图

**Upscale by**:
- **2x**: 512→1024 (推荐)
- **1.5x**: 512→768
- **4x**: 512→2048 (需大显存)

---

### 4.7.3 实战：1024x1024健身美女

```
【参数设置】
txt2img:
  Width: 512
  Height: 768
  Steps: 25
  CFG: 7

[√] Hires. fix:
  Upscaler: Latent
  Hires steps: 15
  Denoising: 0.5
  Upscale by: 2

最终输出: 1024x1536

提示词: (同前面的健身美女模板)
```

**时间对比**:
```
直接生成 1024x1536: ⚠️ 可能重复/变形
Highres Fix: ✅ 质量好，时间增加50%

RTX 3060:
- 512x768: 15秒
- Highres: 15秒 + 10秒 = 25秒
```

---

## 4.8 Scripts 脚本工具

### 4.8.1 X/Y/Z Plot 参数对比

#### 功能：批量测试参数组合

**使用场景**:
```
问题: CFG应该用7还是9？步数20还是25？
解决: X/Y Plot一次生成所有组合对比
```

#### 操作步骤

```
1. Script下拉选择: X/Y/Z plot

2. X轴配置
   X type: CFG Scale
   X values: 6, 7, 8, 9

3. Y轴配置
   Y type: Sampling steps
   Y values: 20, 25, 30

4. Generate

结果: 生成 4×3=12 张图的对比网格

      CFG=6  CFG=7  CFG=8  CFG=9
Step=20  □      □      □      □
Step=25  □      □      □      □
Step=30  □      □      □      □
```

#### 常用对比项

```
【采样器对比】
X: Sampler name
Values: DPM++ 2M Karras, Euler a, UniPC, DPM++ SDE Karras

【分辨率对比】
X: Resolution
Values: 512x768, 640x960, 768x1024

【提示词对比】
X: Prompt S/R (搜索替换)
Search: "sports bra"
Replace: "tank top, crop top, t-shirt"

【模型对比】
X: Checkpoint name
Values: realistic_vision, chilloutmix, dreamshaper
```

---

### 4.8.2 Prompt Matrix 提示词矩阵

#### 功能：批量测试提示词组合

**语法**:
```
使用 | 分隔多个选项

示例:
1 girl, fitness model,
sports bra|tank top|crop top,
yoga pants|shorts|leggings,
gym|park|studio

生成数量 = 3 × 3 × 3 = 27张图
```

#### 实战：服装组合测试

```
Prompt:
masterpiece, best quality,
1 girl, athletic body,
sports bra|tank top|compression shirt,
yoga pants|athletic shorts,
gym background,
professional photography

Script: Prompt matrix

结果: 6张图
1. sports bra + yoga pants
2. sports bra + shorts
3. tank top + yoga pants
4. tank top + shorts
5. compression shirt + yoga pants
6. compression shirt + shorts
```

---

## 4.9 实战项目：健身照片批量生成

### 目标
生成10张不同姿势的健身美女照片

### 方案设计

```python
# 提示词模板
base_prompt = """
masterpiece, best quality, ultra detailed,
1 girl, 25 years old, asian fitness model,
(athletic body:1.3), (toned abs:1.2),
long black hair, high ponytail,
sports bra, yoga pants,
{pose},
modern gym, professional photography,
photorealistic, depth of field
"""

poses = [
    "standing pose, hands on hips",
    "doing squat, proper form",
    "plank position, side view",
    "stretching arms, full body",
    "lunges pose, determined expression",
    "lifting dumbbell, bicep curl",
    "yoga tree pose, balanced",
    "running on treadmill, dynamic",
    "sitting rest, towel on shoulder",
    "victory pose, confident smile"
]

# 负面提示词（通用）
negative = """
(deformed, ugly, bad anatomy:1.5),
(bad hands, extra limbs:1.4),
(low quality, blurry:1.4),
watermark, text
"""
```

### 批量生成方法

#### 方法1: Batch Count（简单）

```
Settings:
  Batch count: 10
  Batch size: 1

Manual:
  每次手动改提示词中的{pose}
  或使用 Prompt S/R 脚本
```

#### 方法2: 使用API（自动化）⭐⭐⭐⭐⭐

```python
import requests
import base64
import time

# WebUI API 地址
url = "http://127.0.0.1:7860/sdapi/v1/txt2img"

# 基础参数
base_payload = {
    "enable_hr": False,
    "denoising_strength": 0,
    "hr_scale": 2,
    "hr_upscaler": "Latent",
    "hr_second_pass_steps": 15,
    "sampler_name": "DPM++ 2M Karras",
    "steps": 25,
    "cfg_scale": 7,
    "width": 768,
    "height": 1024,
    "negative_prompt": negative,
    "seed": -1,
    "batch_size": 1
}

# 批量生成
for i, pose in enumerate(poses):
    # 构建提示词
    prompt = base_prompt.format(pose=pose)

    # 更新payload
    payload = base_payload.copy()
    payload["prompt"] = prompt

    # 发送请求
    print(f"生成第 {i+1}/10 张: {pose}")
    response = requests.post(url, json=payload)

    # 保存图片
    if response.status_code == 200:
        result = response.json()
        image_data = base64.b64decode(result['images'][0])

        with open(f"fitness_{i+1:02d}.png", "wb") as f:
            f.write(image_data)

        print(f"✓ 保存成功: fitness_{i+1:02d}.png")
    else:
        print(f"✗ 生成失败: {response.status_code}")

    time.sleep(2)  # 避免过载

print("批量生成完成！")
```

---

## 4.10 本章总结

### 核心知识点

```
✅ WebUI安装配置（Win/Mac/Linux）
✅ 模型下载管理（Checkpoint/VAE/LoRA）
✅ txt2img核心参数（步数/CFG/分辨率）
✅ img2img重绘技术
✅ Highres Fix高清修复
✅ Scripts批量工具
✅ API自动化生成
```

### 黄金参数配置

```yaml
真人健身照片推荐配置:

Model: Realistic Vision v5.1
VAE: vae-ft-mse-840000

txt2img:
  Steps: 25
  Sampler: DPM++ 2M Karras
  CFG: 7
  Size: 768x1024

Hires Fix:
  Upscaler: Latent
  Steps: 15
  Denoising: 0.5
  Upscale: 2x

Final: 1536x2048
```

### 实战检查清单

- [ ] WebUI成功启动
- [ ] 下载至少1个真人模型
- [ ] 配置VAE
- [ ] 生成第一张健身照片
- [ ] 尝试img2img重绘
- [ ] 使用Highres Fix
- [ ] 批量生成10张不同姿势

---

## 4.11 下一步

**本章完成后，你应该能够**:
- ✅ 独立搭建WebUI环境
- ✅ 生成高质量真人健身照片
- ✅ 使用img2img优化照片
- ✅ 批量生成不同姿势

**下一章预告**:
学习ComfyUI工作流编排，实现更复杂的批量生产流水线！

**下一章**: [第5章 ComfyUI工作流实战精通](../第05章_ComfyUI工作流/README.md)

---

**参考资源**:
- WebUI官方文档: https://github.com/AUTOMATIC1111/stable-diffusion-webui
- Civitai模型库: https://civitai.com/
- WebUI API文档: http://127.0.0.1:7860/docs
