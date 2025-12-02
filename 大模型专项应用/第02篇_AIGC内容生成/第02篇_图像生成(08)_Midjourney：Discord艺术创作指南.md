# 第11章 图像生成(八) Midjourney实战指南

> **学习目标**: 掌握Midjourney Discord机器人使用、参数调优及艺术创作技巧
>
> **难度**: ⭐
> **学习周期**: 2-3天
> **推荐度**: ⭐⭐⭐ (选学,艺术风格突出)

---

## 11.1 Midjourney定位与特点

### 11.1.1 为什么还要学Midjourney?

**在SDXL/Flux/DALL-E 3已经很强的今天,Midjourney的独特价值**:

```
Midjourney = 最强艺术审美 + 零门槛使用 + Discord社区

优势:
✅ 艺术感最强 - 色彩/构图/氛围独一无二
✅ 零配置 - Discord输入命令即可,无需GPU
✅ 社区活跃 - 数百万作品可参考学习
✅ 持续进化 - 每月更新,V6质量飞跃

劣势:
❌ 无法本地部署 - 闭源,必须订阅
❌ 控制力弱 - 无ControlNet/LoRA等精准工具
❌ 成本较高 - 最低$10/月起
❌ Discord交互 - 需适应机器人命令模式
```

### 11.1.2 适用场景

**✅ 推荐使用Midjourney**:
- 艺术创作 (插画/概念设计/海报)
- 游戏美术资源
- 影视概念图/分镜
- 社交媒体视觉内容
- 需要"惊艳感"的营销物料

**❌ 不推荐Midjourney**:
- 批量生产 (无API,成本高)
- 需要精准控制 (如保持角色一致性)
- 产品摄影 (写实度不如SDXL/DALL-E 3)
- 需要本地部署/数据隐私

**与其他工具对比**:
| 维度 | Midjourney | SDXL | Flux.1 | DALL-E 3 |
|------|------------|------|--------|----------|
| **艺术审美** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **写实度** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **文字渲染** | ⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **易用性** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **批量能力** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **成本** | $10-120/月 | 一次性$500-1600 | 一次性$1600+ | 按量$0.04/张 |
| **定制化** | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ |

---

## 11.2 快速上手: Discord机器人使用

### 11.2.1 注册与订阅

**步骤1: 注册Discord账号**
```
1. 访问 https://discord.com
2. 注册账号 (免费)
3. 下载Discord桌面客户端或使用网页版
```

**步骤2: 加入Midjourney服务器**
```
1. 访问 https://www.midjourney.com
2. 点击"Join the Beta"
3. 授权Discord连接
4. 自动加入Midjourney Discord服务器
```

**步骤3: 选择订阅计划** (2025年价格)

| 计划 | 价格 | 快速生成 | 慢速生成 | 并发任务 | 适合人群 |
|------|------|---------|---------|---------|---------|
| **Basic** | $10/月 | ~3.3小时/月 | 无限 | 3个 | 个人轻度使用 |
| **Standard** | $30/月 | ~15小时/月 | 无限 | 3个 | 专业创作者 |
| **Pro** | $60/月 | ~30小时/月 | 无限 | 12个 | 高频使用/团队 |
| **Mega** | $120/月 | ~60小时/月 | 无限 | 12个 | 商业大批量 |

**快速 vs 慢速模式**:
```
快速模式 (Fast):
- 生成时间: 30-60秒
- 消耗GPU时间配额
- 用于: 急需/关键作品

慢速模式 (Relax):
- 生成时间: 5-10分钟
- 不消耗配额,无限使用
- 用于: 批量测试/探索方向
- 仅Standard及以上计划可用

切换命令:
/fast  - 切换到快速模式
/relax - 切换到慢速模式
```

**商业授权**:
```
✅ Standard以上计划包含商业授权
❌ Basic计划仅限个人非商业使用

商业授权包括:
- 出售生成的图像
- 用于营销/广告
- 用于产品设计
- 版权归你所有
```

### 11.2.2 核心命令

**基础生成命令**:
```
/imagine prompt: [你的提示词]
```

示例:
```
/imagine prompt: a serene mountain landscape at golden hour, photorealistic --ar 16:9 --v 6
```

**常用命令列表**:
```bash
# 图像生成
/imagine           # 文生图
/blend             # 混合2-5张图像
/describe          # 图生文 (反向工程提示词)

# 设置
/settings          # 打开设置面板
/fast              # 切换快速模式
/relax             # 切换慢速模式
/private           # 私人模式 (仅Pro/Mega)
/public            # 公开模式

# 信息查询
/info              # 查看账户信息/配额
/subscribe         # 订阅管理
/help              # 帮助文档
```

**生成流程**:
```
1. 在任一 #general 或 #newbies 频道输入命令
2. 等待30-60秒 (Fast模式)
3. Midjourney返回4张变体图 (编号1-4)
4. 选择操作:
   - U1/U2/U3/U4: Upscale (放大) 对应图像
   - V1/V2/V3/V4: Variation (变体) 基于对应图像生成新版本
   - 🔄: 重新生成4张全新图像
   - ❤️: 收藏该作品
```

**实例演示**:
```
输入:
/imagine prompt: a futuristic gym with holographic trainers, cyberpunk style, neon lighting --ar 16:9 --v 6

等待...

输出:
[4张图像网格,编号1-4]

下方按钮:
U1 U2 U3 U4  (Upscale)
V1 V2 V3 V4  (Variation)
🔄 (Reroll)

操作:
1. 点击 U3 (放大第3张图)
2. 获得高分辨率单图
3. 进一步操作:
   - Vary (Strong/Subtle): 在此基础上生成变体
   - Upscale (2x/4x): 进一步放大
   - Zoom Out: 扩展画布
   - Pan ←↑↓→: 向各方向扩展
```

### 11.2.3 核心参数详解

**1. --ar (Aspect Ratio 宽高比)**
```
语法: --ar 宽:高

常用比例:
--ar 1:1    # 正方形 (Instagram)
--ar 16:9   # 横屏视频/Banner
--ar 9:16   # 竖屏视频/Story
--ar 4:3    # 传统照片
--ar 3:2    # 35mm胶片
--ar 21:9   # 电影超宽屏
--ar 2:3    # 竖版海报

示例:
/imagine prompt: a modern gym interior --ar 16:9
```

**2. --v (Version 版本)**
```
语法: --v [版本号]

可用版本:
--v 6      # V6最新版 (2024年发布,当前最强)
--v 5.2    # V5.2 (仍可用,某些风格更好)
--v 5.1    # V5.1
--v 5      # V5原版
--v 4      # V4 (已过时)

不指定时默认为最新版本

V6 vs V5.2 差异:
V6:
- 提示词理解更准确
- 文字渲染改进 (但仍不完美)
- 更写实的细节
- 更好的光影

V5.2:
- 艺术风格化更强
- 某些美学风格优于V6
- 更"梦幻"的感觉

建议: 默认用V6,艺术创作可尝试V5.2
```

**3. --s (Stylize 风格化强度)**
```
语法: --s [0-1000]

默认值: 100
范围: 0 (完全忠实提示词) - 1000 (极度风格化)

效果:
--s 0      # 字面理解,写实,平淡
--s 50     # 轻度风格化
--s 100    # 默认,平衡
--s 250    # 明显艺术化
--s 500    # 强烈艺术感
--s 750    # 极度艺术化,可能偏离提示词
--s 1000   # 最大风格化,Midjourney自由发挥

使用指南:
- 产品摄影: --s 0-50 (写实)
- 概念设计: --s 100-250 (平衡)
- 艺术创作: --s 500-1000 (艺术感)

示例:
/imagine prompt: a apple --s 0     # 普通苹果照片
/imagine prompt: a apple --s 500   # 艺术化苹果,光影/色彩夸张
/imagine prompt: a apple --s 1000  # 超现实/抽象苹果
```

**4. --style (风格预设)**
```
仅V6可用:
--style raw    # 原始模式,更写实,减少Midjourney默认美化

对比:
默认 (无--style): Midjourney会自动美化/艺术化
--style raw:      更接近真实照片/原始风格

适用场景:
--style raw 用于:
- 产品摄影
- 建筑效果图
- 写实人物肖像
- 不希望过度艺术化的场景

示例:
/imagine prompt: a modern office interior --style raw --ar 16:9 --v 6
```

**5. --q (Quality 质量)**
```
语法: --q [0.25 | 0.5 | 1]

默认值: 1
可选:
--q 0.25   # 草图质量,快速,省配额
--q 0.5    # 中等质量
--q 1      # 标准质量 (默认)

高于1的值 (如--q 2) 在V6中已移除

使用建议:
- 测试提示词: --q 0.25
- 正式创作: --q 1

示例:
/imagine prompt: concept sketch --q 0.25
```

**6. --chaos (混沌度)**
```
语法: --chaos [0-100]

默认值: 0
效果: 控制4张初始图的多样性

--chaos 0    # 4张非常相似
--chaos 50   # 中等差异
--chaos 100  # 4张差异巨大

适用:
- 需要一致性: --chaos 0
- 探索多种可能: --chaos 50-100

示例:
/imagine prompt: a logo design --chaos 0     # 生成相似的4个变体
/imagine prompt: fantasy landscape --chaos 100  # 生成完全不同的4种风格
```

**7. --seed (随机种子)**
```
语法: --seed [0-4294967295]

作用: 固定随机性,生成可复现结果

使用:
1. 查看已生成图的seed:
   - 对图像添加 ✉️ 反应
   - Midjourney会发送包含seed的DM

2. 复用seed:
   /imagine prompt: same prompt --seed 1234567890

注意: seed + 完全相同提示词 = 相似(但不完全相同)结果

适用:
- 微调提示词时保持构图
- 批量生成一致风格

示例:
/imagine prompt: a character design --seed 42
/imagine prompt: a character design, wearing red jacket --seed 42  # 相似构图,但加了红夹克
```

**8. --no (负面提示)**
```
语法: --no [不想要的元素]

作用: 类似Negative Prompt,减少特定元素出现

示例:
/imagine prompt: a forest scene --no people, buildings, cars
/imagine prompt: a portrait --no glasses, hat, beard

注意: 效果不如SD的Negative Prompt强,仅供参考
```

**9. --tile (平铺/无缝纹理)**
```
语法: --tile

作用: 生成可无缝平铺的纹理图案

适用: 游戏纹理、壁纸、背景图案

示例:
/imagine prompt: geometric pattern --tile
/imagine prompt: wood texture --tile --ar 1:1
```

**10. --weird (怪异度, V6新增)**
```
语法: --weird [0-3000] 或 --w [0-3000]

默认值: 0
效果: 增加非传统/实验性美学

--weird 0      # 传统美学
--weird 500    # 轻微怪异
--weird 1000   # 明显非传统
--weird 3000   # 极度实验性/抽象

适用: 艺术实验、超现实创作

示例:
/imagine prompt: a portrait --weird 1000
```

**参数组合示例**:
```bash
# 产品摄影 (写实)
/imagine prompt: Nike running shoes on white background, product photography --style raw --s 0 --ar 1:1 --v 6

# 艺术海报 (高风格化)
/imagine prompt: cyberpunk city at night, neon signs --s 750 --ar 2:3 --v 6

# 游戏纹理 (平铺)
/imagine prompt: brick wall texture, seamless --tile --ar 1:1 --v 6

# 概念探索 (高混沌)
/imagine prompt: alien creature design --chaos 80 --s 500 --v 6

# 电影分镜 (宽屏)
/imagine prompt: dramatic action scene, cinematic lighting --ar 21:9 --s 200 --v 6
```

---

## 11.3 提示词技巧 (vs SD差异)

### 11.3.1 Midjourney提示词特点

**关键差异**:
```
Stable Diffusion提示词风格:
关键词堆砌,用逗号分隔,带权重
示例: "1girl, blue eyes, long hair, (smile:1.2), sitting, park, sunlight, masterpiece, best quality"

Midjourney提示词风格:
自然语言,描述性语句,像写文章
示例: "A young woman with striking blue eyes and flowing long hair, smiling warmly as she sits in a peaceful park bathed in golden sunlight, captured in a cinematic style"
```

**Midjourney提示词原则**:
1. **自然描述**: 用完整句子,不是关键词
2. **具体明确**: 明确描述细节
3. **风格在前**: 重要信息放提示词前部
4. **避免否定**: 少用"no/without",用`--no`参数代替
5. **不需要"高质量"**: 不用写"masterpiece, 8K"等

### 11.3.2 提示词模板

```python
MIDJOURNEY_TEMPLATES = {
    "摄影风格": """
        A [主体] [动作/状态],
        [环境描述],
        [光线描述],
        [情绪/氛围],
        captured in [摄影风格] style,
        shot on [相机/镜头]
        --ar [比例] --v 6 [--style raw]
        """,

    "艺术绘画": """
        [主体] in the style of [艺术家/流派],
        [场景描述],
        [色彩方案],
        [艺术手法],
        [情绪]
        --ar [比例] --s [风格化强度] --v 6
        """,

    "概念设计": """
        Concept art of [对象],
        [设计特点],
        [材质/细节],
        [风格] aesthetic,
        [视角]
        --ar [比例] --s 250 --v 6
        """,

    "建筑/室内": """
        [建筑类型] featuring [关键特征],
        [材料],
        [光线条件],
        [视角] view,
        [风格] architecture style
        --ar 16:9 --style raw --v 6
        """,

    "角色设计": """
        Character design of [角色描述],
        [服装/装扮],
        [个性特征],
        [姿态],
        [背景],
        [风格] art style
        --ar 2:3 --s 500 --v 6
        """
}
```

**实战示例**:
```bash
# 示例1: 健身场景摄影
/imagine prompt: A muscular athlete performing a barbell squat in a modern industrial-style gym, dramatic side lighting creating strong shadows, intense focus and determination in expression, captured in editorial fitness photography style, shot on Canon EOS R5 with 85mm f/1.2 lens, shallow depth of field --ar 4:5 --style raw --v 6

# 示例2: 产品摄影
/imagine prompt: A sleek black protein shaker bottle with orange accents, placed on a concrete surface with scattered coffee beans, minimalist composition, soft overhead lighting with subtle reflections, commercial product photography style, clean and modern aesthetic --ar 1:1 --style raw --s 0 --v 6

# 示例3: 艺术海报
/imagine prompt: Motivational fitness poster in bold graphic design style, dynamic composition featuring an abstract silhouette of a runner, navy blue and vibrant orange color scheme, geometric shapes, modern minimalist aesthetic, powerful and inspiring mood --ar 2:3 --s 400 --v 6

# 示例4: 概念设计
/imagine prompt: Concept art of a futuristic smart gym equipment, holographic interface displaying workout data, sleek metallic and glass materials, glowing blue accent lights, cyberpunk aesthetic, three-quarter view showing both form and function --ar 16:9 --s 300 --v 6

# 示例5: 室内设计
/imagine prompt: Modern minimalist home gym interior featuring large windows with natural light, polished concrete floors, oak wood accents, state-of-the-art equipment neatly organized, abundant greenery, Scandinavian design aesthetic, wide-angle view showcasing the entire space --ar 16:9 --style raw --v 6
```

### 11.3.3 艺术风格参考

Midjourney擅长模仿各种艺术风格:

```bash
# 摄影风格
- "editorial photography" (时尚大片)
- "street photography" (街拍)
- "cinematic lighting" (电影感光线)
- "golden hour photography" (黄金时刻)
- "film noir style" (黑色电影)
- "analog film photography" (胶片摄影)

# 绘画风格
- "oil painting" (油画)
- "watercolor" (水彩)
- "digital painting" (数字绘画)
- "concept art" (概念艺术)
- "matte painting" (遮景绘画)

# 设计风格
- "minimalist design" (极简主义)
- "brutalism" (野兽派/粗野主义)
- "art deco" (装饰艺术)
- "bauhaus" (包豪斯)
- "memphis design" (孟菲斯设计)

# 插画风格
- "flat illustration" (扁平插画)
- "isometric illustration" (等距插画)
- "line art" (线稿)
- "vector art" (矢量艺术)
- "anime style" (动漫风格)

# 流派/艺术家
- "in the style of Studio Ghibli" (吉卜力风格)
- "inspired by Moebius" (莫比斯)
- "impressionist" (印象派)
- "surrealist" (超现实主义)
```

---

## 11.4 高级功能

### 11.4.1 /blend (图像混合)

```bash
语法:
/blend
- 上传2-5张图像
- Midjourney自动混合

示例用途:
1. 风格迁移: 内容图 + 风格图
2. 角色融合: 角色A + 角色B
3. 场景合成: 场景1 + 场景2

操作:
/blend
[上传图片1]
[上传图片2]
[可选: dimensions 选择比例]
[可选: --v 6]
```

### 11.4.2 /describe (图生文)

```bash
作用: 上传图片,Midjourney生成4个可能的提示词

用途:
1. 学习提示词写法
2. 反向工程他人作品
3. 改进现有图像

操作:
/describe
[上传图片]

输出:
Midjourney返回4个提示词描述
可点击任一提示词直接生成
```

### 11.4.3 图像扩展 (Zoom Out / Pan)

```bash
场景: 已生成图像后想看更多周边

操作:
1. Upscale一张图
2. 点击 Zoom Out 按钮:
   - Zoom Out 2x: 2倍扩展画布
   - Zoom Out 1.5x: 1.5倍扩展
   - Custom Zoom: 自定义缩放比例

3. 或点击 Pan 按钮向特定方向扩展:
   - ← ↑ ↓ →

示例:
生成: 健身教练肖像 (1:1)
Upscale → Zoom Out 2x → 得到更广场景
或 Pan ← → 向两侧扩展,改为16:9
```

### 11.4.4 Vary (变体生成)

```bash
Upscale后可生成变体:

Vary (Strong): 基于原图生成较大变化的版本
Vary (Subtle): 生成轻微变化的版本
Vary (Region): 选择区域重新生成 (局部修改)

Vary (Region) 使用:
1. Upscale一张图
2. 点击 Vary (Region)
3. 用画笔选择想修改的区域
4. 输入新的提示词描述该区域
5. 生成

示例:
原图: 健身房内景
Vary (Region):
- 选择墙面区域
- 输入: "motivational quote painted on wall"
- 生成: 墙面添加标语,其他不变
```

---

## 11.5 成本分析与优化

### 11.5.1 GPU时间消耗计算

```python
# Midjourney GPU时间消耗 (Fast模式)

FAST_MODE_COSTS = {
    "初始生成(4张)": 1,      # 约1 GPU分钟
    "Upscale": 1,            # 约1 GPU分钟
    "Variation": 1,          # 约1 GPU分钟
    "Zoom Out": 1,           # 约1 GPU分钟
}

# 示例: 完整工作流耗时
workflow = """
1. /imagine (生成4张) → 1分钟
2. U3 (放大第3张) → 1分钟
3. Vary Strong (生成变体) → 1分钟
4. Zoom Out 2x (扩展) → 1分钟
总计: 4 GPU分钟
"""

# 不同计划每月可生成数量估算
plans = {
    "Basic ($10/月)": {
        "Fast时间": 200,  # 约3.3小时 = 200分钟
        "可完整作品": 200 / 4,  # 约50个完整作品 (初始+U+V+Zoom)
        "仅初始生成": 200 / 1  # 约200次初始生成
    },
    "Standard ($30/月)": {
        "Fast时间": 900,  # 约15小时
        "可完整作品": 900 / 4,  # 约225个
        "仅初始生成": 900
    },
    "Pro ($60/月)": {
        "Fast时间": 1800,  # 约30小时
        "可完整作品": 1800 / 4,  # 约450个
        "仅初始生成": 1800
    }
}

# Relax模式 (Standard及以上)
# 无限制,但速度慢 (5-10分钟)
```

### 11.5.2 成本优化策略

**1. 善用Relax模式**
```
策略: 大部分探索用Relax,关键作品用Fast

工作流:
1. Relax模式测试10个方向 (免费,慢)
2. 挑选最佳1-2个
3. 切换Fast模式精修 (消耗配额,快)
4. 最终交付

节省: 节省80%+ Fast配额

命令:
/relax  - 切换慢速
/fast   - 切换快速
```

**2. 减少无效迭代**
```
❌ 低效: 盲目尝试,每次生成都U+V
/imagine → U1 → V → U3 → V → ... (浪费)

✅ 高效:
1. 初始生成,仅观察4张缩略图
2. 满意后才Upscale
3. 需要微调才Vary

节省: 每个作品省2-3个操作 = 节省2-3分钟配额
```

**3. 批量同类任务**
```
场景: 需要生成10张相似风格图

策略:
1. 第1张: 完整流程,确定最佳提示词和参数
2. 第2-10张: 直接用成熟提示词,减少试错

示例:
# 确定最佳提示词后
base_prompt = "A fitness instructor demonstrating [exercise], modern gym, cinematic lighting --ar 4:5 --style raw --v 6"

exercises = ["squat", "deadlift", "bench press", ...]
for exercise in exercises:
    prompt = base_prompt.replace("[exercise]", exercise)
    # 一次生成,直接选最佳Upscale,无需多次迭代
```

**4. 复用seed**
```
场景: 同一构图,换不同服装/配色

策略:
1. 满意的构图 → 获取seed
2. 后续生成复用seed,仅改提示词细节

示例:
# 第1次生成
/imagine prompt: character design, athletic build, gym outfit --seed 12345

# 获得满意构图后
/imagine prompt: character design, athletic build, casual streetwear --seed 12345
# 保持相似构图,仅换服装
```

**5. 降质测试**
```
不确定的想法先用 --q 0.25 快速测试

/imagine prompt: experimental idea --q 0.25  # 快速草图
# 满意后
/imagine prompt: experimental idea --q 1     # 正式生成
```

---

## 11.6 版权与商业使用

### 11.6.1 版权政策

```
Midjourney版权政策 (2025):

✅ 你拥有生成图像的版权 (付费用户)
  - Standard, Pro, Mega计划
  - 可商用,可出售,可修改

⚠️ Basic计划限制:
  - 仅个人非商业使用
  - 商用需升级到Standard+

❌ 免费试用已取消
  - Midjourney目前无免费试用
  - 必须订阅才能使用

📝 注意事项:
  - 生成的图像默认公开 (他人可见)
  - Pro/Mega计划可选私人模式 (/private)
  - 不可生成侵权内容 (如明星肖像/品牌Logo)
```

**私人模式**:
```bash
# 仅Pro ($60/月) 和 Mega ($120/月) 可用

开启:
/private

关闭:
/public

效果:
- Private模式: 图像仅你可见
- Public模式: 图像在Midjourney社区展示

注意: Private模式不额外收费,包含在Pro/Mega计划中
```

### 11.6.2 商业应用案例

```python
COMMERCIAL_USE_CASES = {
    "游戏美术": {
        "用途": "概念设计、场景原画、角色设计",
        "流程": "MJ生成初稿 → 画师精修 → 最终资产",
        "成本": "$30-60/月 (Standard/Pro)",
        "推荐计划": "Pro (更多并发,快速迭代)"
    },

    "营销物料": {
        "用途": "社交媒体配图、海报、广告Banner",
        "流程": "MJ生成 → 后期添加文字/Logo",
        "成本": "$30/月 (Standard)",
        "推荐计划": "Standard (足够配额)"
    },

    "影视概念": {
        "用途": "分镜、氛围图、概念设计",
        "流程": "MJ快速生成多方案 → 导演选择 → 细化",
        "成本": "$60/月 (Pro)",
        "推荐计划": "Pro (大量并发需求)"
    },

    "产品设计": {
        "用途": "产品概念可视化、包装设计灵感",
        "流程": "MJ生成创意 → 工业设计师实现",
        "成本": "$30/月 (Standard)",
        "推荐计划": "Standard"
    }
}
```

---

## 11.7 实战案例: 健身品牌视觉设计

### 11.7.1 项目需求

```
品牌: FitPro (虚构健身品牌)
需求:
- 5张社交媒体配图 (1:1)
- 3张横版Banner (16:9)
- 2张竖版海报 (2:3)
预算: $30 (Standard计划)
周期: 1天
```

### 11.7.2 执行流程

```bash
# 阶段1: 探索风格 (Relax模式)
/relax  # 切换慢速模式

# 测试3种风格方向
方向1: 写实摄影
/imagine prompt: Professional fitness photography, athletic person working out in modern gym, dynamic action, dramatic lighting, editorial style --ar 1:1 --style raw --v 6

方向2: 艺术插画
/imagine prompt: Fitness illustration in bold graphic style, energetic athlete silhouette, vibrant orange and navy blue colors, geometric shapes, modern minimalist aesthetic --ar 1:1 --s 500 --v 6

方向3: 电影质感
/imagine prompt: Cinematic fitness scene, athlete training in atmospheric gym, moody lighting with volumetric rays, film noir aesthetic, high contrast --ar 1:1 --s 300 --v 6

# 等待10分钟 (Relax模式慢)
# 团队确定: 方向2 (艺术插画) 最符合品牌调性

# 阶段2: 精修 (Fast模式)
/fast  # 切换快速模式

# 社交媒体配图 (1:1) x5
/imagine prompt: Minimalist fitness illustration, person doing yoga pose, navy blue and orange gradient background, clean geometric design, inspiring mood --ar 1:1 --s 400 --v 6 --seed 100
→ 选U2 → Vary (Subtle) → 最终版

/imagine prompt: Bold fitness graphic, athlete running silhouette, abstract geometric shapes, energetic composition, modern vector art style --ar 1:1 --s 400 --v 6 --seed 100
→ 选U3

/imagine prompt: Fitness motivation poster style, weightlifter silhouette, dynamic diagonal composition, vibrant color blocks, powerful aesthetic --ar 1:1 --s 400 --v 6 --seed 100
→ 选U1

# 继续生成剩余2张...

# 横版Banner (16:9) x3
/imagine prompt: Wide fitness scene illustration, multiple athletes in training, modern gym environment, panoramic composition, bold graphic style --ar 16:9 --s 350 --v 6
→ U4 → Vary (Subtle)

# 竖版海报 (2:3) x2
/imagine prompt: Tall vertical fitness poster, athletic figure in heroic pose, bold typography space at top, navy and orange color scheme, inspiring and powerful --ar 2:3 --s 450 --v 6
→ U2

# 阶段3: 细节调整
# 某张需要调整背景色
→ Vary (Region) → 选择背景区域 → 输入"solid orange background" → 生成

# GPU时间消耗统计:
初始探索(Relax): 0分钟 (不计入配额)
精修生成: 10次 /imagine = 10分钟
Upscale: 10次 = 10分钟
Vary: 5次 = 5分钟
总计: 25分钟 (Standard计划900分钟配额的3%)
```

### 11.7.3 后期处理

```python
# Midjourney生成后,通常需要后期处理:

后期流程:
1. 下载高清图 (直接下载或右键保存)
2. Photoshop/Figma导入
3. 添加品牌文字/Logo
4. 调整色彩一致性
5. 导出最终格式

工具推荐:
- Figma: 添加文字/图形元素
- Photoshop: 精修细节
- Canva: 快速模板化处理
```

---

## 11.8 常见问题

### Q1: 如何避免生成被拒?

```
Midjourney内容政策:

❌ 禁止:
- 暴力/血腥内容
- 成人/色情内容
- 名人肖像 (未经授权)
- 品牌Logo (侵权)
- 仇恨/歧视内容

✅ 安全做法:
- 用描述性词汇替代名人名字:
  ❌ "Elon Musk portrait"
  ✅ "tech entrepreneur portrait, middle-aged man"

- 避免过于写实的暴力:
  ❌ "bloody battle scene"
  ✅ "epic battle, cinematic, stylized"
```

### Q2: 如何生成一致的角色?

```
Midjourney本身不支持角色一致性,但可以:

方法1: 使用seed
/imagine prompt: character design, red hair warrior --seed 12345
(后续生成用相同seed,构图会相似但不完全一致)

方法2: 使用 /describe + 图像参考
1. 生成第1版角色
2. /describe 该图像 → 获得提示词
3. 后续生成复用该提示词 + 原图作为参考

方法3: 结合SD LoRA (推荐)
1. Midjourney生成角色设计
2. 用该角色训练SD LoRA
3. 后续用SD LoRA生成一致角色

注意: Midjourney不如SD适合需要严格角色一致性的项目
```

### Q3: 生成速度慢怎么办?

```
Fast模式慢 → 检查服务器负载
- 高峰期(美国时间白天)会慢
- 尝试非高峰时段

Relax模式太慢 → 升级计划或切换Fast

想更快 → 考虑本地SDXL/Flux
```

---

## 11.9 总结

### 11.9.1 Midjourney核心价值

```
✅ 无可替代的优势:
- 艺术审美独一无二
- 零技术门槛
- Discord社区学习资源丰富
- 持续快速进化

⚠️ 明显局限:
- 闭源,无法本地部署
- 控制力弱于SD/ComfyUI
- 成本较高 (订阅制)
- 批量生产效率低
```

### 11.9.2 适用场景总结

```python
def should_use_midjourney(project_type, budget, tech_skill):
    """Midjourney适用性判断"""

    if project_type in ["艺术创作", "概念设计", "游戏美术"]:
        return "强烈推荐Midjourney"

    elif project_type == "产品摄影" and tech_skill == "低":
        return "Midjourney可用,但DALL-E 3更合适"

    elif project_type == "批量生产" and budget == "有限":
        return "不推荐,用SDXL本地部署"

    elif project_type == "需要角色一致性":
        return "不推荐,用SD + LoRA"

    elif tech_skill == "低" and budget == "充足":
        return "推荐Midjourney,易用性强"

    else:
        return "根据具体需求混合使用多工具"
```

### 11.9.3 学习建议

```
入门路径 (2-3天):
Day 1:
- 注册账号,订阅Basic/Standard
- 学习/imagine基础命令
- 测试--ar, --v, --s参数

Day 2:
- 练习提示词写作 (50+次生成)
- 学习U/V操作
- 尝试不同艺术风格

Day 3:
- 学习高级功能 (/blend, /describe, Vary Region)
- 完成1个完整项目 (如10张主题配图)
- 总结最佳实践

持续提升:
- 每天浏览Midjourney社区优秀作品
- 用/describe学习他人提示词
- 建立自己的提示词模板库
```

---

## 11.10 实战练习

### 练习1: 基础操作
1. 生成5张不同风格的健身场景
2. 尝试--ar 1:1, 16:9, 2:3三种比例
3. 对比--s 0, --s 100, --s 500效果

### 练习2: 提示词优化
1. 从简单提示词开始: "gym"
2. 逐步细化: 光线、风格、情绪等
3. 记录哪些描述最有效

### 练习3: 高级功能
1. 使用/blend混合2张图
2. 使用/describe分析优秀作品
3. 使用Vary (Region)局部修改

### 练习4: 完整项目
1. 为虚构品牌设计5张营销图
2. 使用Relax+Fast混合策略
3. 记录GPU时间消耗

---

## 参考资源

- [Midjourney官方文档](https://docs.midjourney.com/)
- [Midjourney Discord服务器](https://discord.gg/midjourney)
- [Midjourney订阅页面](https://www.midjourney.com/account/)
- [提示词参考网站](https://prompthero.com/midjourney-prompts)

**下一章预告**: 第12章将对比新兴图像生成工具(Adobe Firefly, Ideogram, Leonardo等),帮助你了解更多选择。