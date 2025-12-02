# 第25章 图生视频(五) Kling AI实战 - 真人动作视频专家

> **学习目标**: 掌握Kling AI的真人动作视频生成,特别是健身/运动场景
>
> **难度**: ⭐⭐⭐
> **推荐度**: ⭐⭐⭐⭐⭐ (真人运动视频最佳选择)

---

## 25.1 Kling AI的独特优势

### 25.1.1 为什么选择Kling AI?

**Kling AI = 真人动作视频的专家**

```
Runway Gen-3: 擅长相机运动、场景渲染
Pika 1.5: 擅长创意效果、风格化
Kling AI: 擅长真人复杂动作、表情驱动 ⭐⭐⭐⭐⭐

核心优势:
✅ 3D人脸重建技术 - 表情自然流畅
✅ 骨骼动作理解 - 复杂运动准确
✅ 物理引擎模拟 - 头发/衣物真实飘动
✅ 长视频支持 - 最长10秒高质量
```

**适用场景**:
- 健身教学视频 (深蹲/硬拉/跑步等动作)
- 舞蹈动作展示
- 体育运动慢动作
- 人物访谈/表情特写
- 产品使用演示 (真人操作)

### 25.1.2 vs 竞品对比

| 维度 | Kling AI | Runway Gen-3 | Pika 1.5 | SVD |
|------|---------|--------------|----------|-----|
| **真人动作质量** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **表情细腻度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐ |
| **复杂动作** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| **物理真实感** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **视频时长** | 5-10秒 | 5-18秒 | 3-5秒 | 2-4秒 |
| **成本** | 中 | 中高 | 低 | 免费(本地) |
| **生成速度** | 2-5分钟 | 2-3分钟 | 1-3分钟 | 快(本地) |

**结论**:
- 真人健身/运动视频 → Kling AI首选 ⭐⭐⭐⭐⭐
- 相机运动/场景切换 → Runway Gen-3
- 创意效果/低成本 → Pika 1.5
- 快速测试/批量 → SVD本地

---

## 25.2 Kling AI快速上手

### 25.2.1 注册与定价

```
平台: https://klingai.com

免费版:
- 66 credits 初始赠送
- 每日签到奖励
- 可生成约6-8个视频
- 有水印

会员版 (¥99/月 或 $15/月):
- 3300 credits/月
- 无水印 ✅
- 优先队列
- 商业授权 ✅
- 推荐!

积分消耗:
- 标准5秒视频: 30-50 credits
- 高质量5秒: 80-100 credits
- 10秒视频: 150-200 credits
```

### 25.2.2 基础操作流程

```
Web界面使用:

步骤1: 上传参考图
- 推荐分辨率: 1920×1080 或 1080×1920
- 主体清晰,光线充足
- 人物占画面30-70%

步骤2: 输入运动描述
示例:
"The athlete slowly performs a squat,
 going down smoothly and rising back up,
 maintaining proper form throughout"

步骤3: 高级设置
- 视频时长: 5秒 / 10秒
- 相机运动: 静止 / 缓慢推进 / 跟随
- 运动强度: 低 / 中 / 高

步骤4: 生成
- 等待2-5分钟
- 获得视频,可下载/重新生成

步骤5: 迭代优化
- 调整提示词
- 更换参考图
- 调节相机运动
```

---

## 25.3 提示词工程 - 真人动作专用

### 25.3.1 动作描述结构

```python
# Kling AI提示词模板 (真人动作)

MOTION_PROMPT_TEMPLATE = """
[主体] + [动作类型] + [动作细节] + [速度] + [强调点]

示例:

深蹲动作:
"The male athlete performs a perfect barbell squat,
 slowly descending with controlled motion,
 knees tracking over toes,
 then powerfully rising back to standing position,
 maintaining upright torso throughout the movement"

跑步动作:
"The runner accelerates from a standing start,
 explosively pushing off the ground,
 arms pumping rhythmically,
 smooth stride with natural breathing"

瑜伽动作:
"The woman gracefully transitions into tree pose,
 slowly lifting her leg,
 balancing perfectly with calm expression,
 arms flowing upward in smooth motion"
```

### 25.3.2 关键词库

```python
KLING_KEYWORDS = {
    "动作速度": {
        "极慢": "very slowly, in slow motion",
        "慢": "slowly, with controlled motion",
        "正常": "at normal pace, smoothly",
        "快": "quickly, energetically",
        "爆发": "explosively, with power"
    },

    "动作质量": {
        "流畅": "smoothly, fluidly",
        "有力": "powerfully, with strength",
        "优雅": "gracefully, elegantly",
        "精准": "precisely, with perfect form",
        "自然": "naturally, effortlessly"
    },

    "表情/情绪": {
        "专注": "with focused expression",
        "用力": "with intense effort",
        "平静": "with calm demeanor",
        "自信": "with confident smile",
        "疲惫": "showing fatigue"
    },

    "身体部位强调": {
        "腿部": "emphasizing leg movement",
        "手臂": "with clear arm motion",
        "核心": "engaging core muscles visibly",
        "全身": "full-body coordination",
        "面部": "facial expression clearly visible"
    },

    "相机建议": {
        "静止": "camera static, fixed angle",
        "缓推": "camera slowly pushing in",
        "跟随": "camera following the subject",
        "侧面": "side view perspective",
        "正面": "frontal view"
    }
}
```

### 25.3.3 健身动作专用提示词

```python
FITNESS_PROMPTS = {
    "深蹲(Squat)": """
        The athletic person performs a bodyweight squat,
        starting from standing position,
        slowly descending with knees tracking over toes,
        hips moving back and down,
        maintaining upright chest,
        then powerfully rising back up,
        full range of motion,
        controlled breathing visible,
        camera static front view
        """,

    "硬拉(Deadlift)": """
        The athlete performs a deadlift with barbell,
        starting with bar on ground,
        smoothly lifting with straight back,
        hips and knees extending together,
        bar traveling close to body,
        powerful lockout at top,
        controlled descent,
        focused expression,
        side view camera
        """,

    "俯卧撑(Push-up)": """
        The person performs a perfect push-up,
        starting in plank position,
        slowly lowering body with control,
        elbows at 45 degrees,
        chest nearly touching ground,
        then pushing back up powerfully,
        maintaining straight body line,
        controlled breathing,
        side angle view
        """,

    "引体向上(Pull-up)": """
        The athlete performs a pull-up,
        starting from dead hang,
        explosively pulling up,
        chin clearing the bar,
        controlled descent,
        shoulder blades squeezing,
        engaged core visible,
        frontal view
        """,

    "跑步(Running)": """
        The runner sprints forward,
        powerful leg drive,
        arms pumping rhythmically,
        smooth stride pattern,
        natural breathing rhythm,
        focused forward gaze,
        camera tracking from side
        """,

    "瑜伽-战士式(Warrior Pose)": """
        The woman flows into warrior II pose,
        gracefully extending arms,
        front knee bending at 90 degrees,
        back leg straight and strong,
        torso upright and open,
        calm focused expression,
        smooth breathing visible,
        static side view
        """
}
```

---

## 25.4 高级技巧

### 25.4.1 运动强度控制

```python
# Kling AI独有的运动强度参数

运动强度设置:
- 低 (10-30%): 微妙动作,呼吸/眨眼/轻微晃动
- 中 (30-70%): 标准动作,走路/挥手/姿势变化
- 高 (70-100%): 剧烈运动,跳跃/奔跑/爆发动作

案例对比:

# 低强度 - 静态姿势的微调
prompt = "Athlete in plank position, subtle breathing movement"
intensity = "低"
→ 结果: 轻微呼吸起伏,身体基本静止

# 中强度 - 正常动作
prompt = "Athlete performs a squat from standing to bottom position"
intensity = "中"
→ 结果: 流畅的下蹲动作

# 高强度 - 爆发动作
prompt = "Athlete explosively jumps up from squat position"
intensity = "高"
→ 结果: 有力的跳跃,衣服/头发剧烈飘动
```

### 25.4.2 多阶段动作拼接

```python
# 问题: Kling单次只能生成5-10秒,复杂动作怎么办?

# 解决: 分段生成 + 后期拼接

# 案例: 完整深蹲循环 (3次,共15秒)

# 阶段1: 第一次深蹲 (5秒)
stage1_prompt = """
Athlete performs first squat rep,
starting standing, descending smoothly,
rising back up, controlled motion
"""

# 阶段2: 第二次深蹲 (5秒)
# 关键: 使用第1段的最后帧作为参考图
stage2_prompt = """
Continuing the motion, athlete performs second squat,
same controlled form, steady breathing
"""

# 阶段3: 第三次深蹲 (5秒)
stage3_prompt = """
Final squat rep, athlete shows slight fatigue,
still maintaining good form, finishing strong
"""

# FFmpeg无缝拼接
import subprocess

subprocess.run([
    "ffmpeg",
    "-i", "squat_stage1.mp4",
    "-i", "squat_stage2.mp4",
    "-i", "squat_stage3.mp4",
    "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1[outv]",
    "-map", "[outv]",
    "squat_complete_15s.mp4"
])
```

### 25.4.3 表情细节控制

```python
# Kling AI的表情驱动能力极强

表情提示词技巧:

# 用力表情
"face showing intense effort, gritted teeth, furrowed brow"

# 平静专注
"calm focused expression, steady gaze, controlled breathing"

# 疲惫
"showing fatigue, heavy breathing, sweat visible"

# 胜利喜悦
"triumphant smile, eyes bright, celebrating success"

# 案例: 最后一组的艰难
prompt = """
Athlete struggles through final squat rep,
face showing intense effort,
gritted teeth, furrowed brow,
trembling slightly but pushing through,
powerful finish with relief expression
"""
→ 结果: 表情从用力→咬牙坚持→完成时的释然,极其真实!
```

---

## 25.5 API集成 (Python)

```python
import requests
import time
from pathlib import Path

class KlingAI:
    """Kling AI视频生成API封装"""

    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.klingai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def upload_image(self, image_path):
        """上传参考图"""
        with open(image_path, "rb") as f:
            files = {"file": f}
            response = requests.post(
                f"{self.base_url}/upload",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files=files
            )
        return response.json()["url"]

    def generate_video(
        self,
        image_url,
        prompt,
        duration=5,
        motion_intensity="medium",
        camera_motion="static"
    ):
        """
        生成视频

        参数:
            image_url: 参考图URL
            prompt: 运动描述
            duration: 5 或 10 秒
            motion_intensity: low/medium/high
            camera_motion: static/push/follow
        """
        payload = {
            "image": image_url,
            "prompt": prompt,
            "duration": duration,
            "motion_intensity": motion_intensity,
            "camera_motion": camera_motion,
            "mode": "pro"  # 高质量模式
        }

        # 创建任务
        response = requests.post(
            f"{self.base_url}/video/generate",
            headers=self.headers,
            json=payload
        )
        task_id = response.json()["task_id"]

        # 轮询结果
        while True:
            status_response = requests.get(
                f"{self.base_url}/tasks/{task_id}",
                headers=self.headers
            )
            status_data = status_response.json()

            if status_data["status"] == "completed":
                return status_data["video_url"]
            elif status_data["status"] == "failed":
                raise Exception(f"Failed: {status_data['error']}")

            time.sleep(10)

    def download_video(self, video_url, output_path):
        """下载视频"""
        video_data = requests.get(video_url).content
        with open(output_path, "wb") as f:
            f.write(video_data)
        print(f"✓ Downloaded: {output_path}")

# 使用示例
kling = KlingAI(api_key="your_api_key")

# 生成深蹲教学视频
image_url = kling.upload_image("squat_reference.jpg")

video_url = kling.generate_video(
    image_url=image_url,
    prompt="""
    The athlete performs a perfect squat,
    slowly descending with controlled motion,
    knees tracking properly,
    then rising back up powerfully,
    maintaining good form
    """,
    duration=5,
    motion_intensity="medium",
    camera_motion="static"
)

kling.download_video(video_url, "squat_demo.mp4")
print("完成!")
```

---

## 25.6 实战项目: 健身动作教学视频库

```python
# 项目: 生成10个标准健身动作的教学视频

# 步骤1: 准备参考图 (使用SDXL + ControlNet OpenPose生成)
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux import OpenposeDetector

# 生成标准动作姿势图
def generate_reference_images():
    """生成10个动作的参考图"""

    exercises = [
        ("squat", "Athletic male performing squat, gym background"),
        ("deadlift", "Athlete doing deadlift with barbell, side view"),
        ("pushup", "Person in perfect push-up form, mat on floor"),
        ("pullup", "Athlete at top of pull-up, gym setting"),
        ("plank", "Person holding plank position, core engaged"),
        # ... 更多动作
    ]

    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet
    )

    for exercise_name, prompt in exercises:
        # 从真人照片提取姿势
        pose_image = get_pose_reference(exercise_name)  # 预先准备的姿势骨架

        # 生成干净的参考图
        image = pipe(
            prompt=f"{prompt}, professional photography, clean background",
            image=pose_image,
            num_inference_steps=40
        ).images[0]

        image.save(f"references/{exercise_name}.jpg")

# 步骤2: 批量生成视频
def batch_generate_videos():
    """批量生成教学视频"""

    kling = KlingAI(api_key="your_key")

    exercises_prompts = {
        "squat": """
            The athlete performs a controlled squat,
            descending smoothly with proper form,
            knees tracking over toes,
            rising powerfully back to standing,
            maintaining upright torso
            """,

        "deadlift": """
            The person executes a deadlift,
            lifting the barbell with straight back,
            hips and knees extending together,
            controlled descent,
            maintaining bar close to body
            """,

        "pushup": """
            The athlete performs a strict push-up,
            lowering body with control,
            chest nearly touching ground,
            pushing back up explosively,
            straight body line throughout
            """,

        # ... 其他动作
    }

    results = []

    for exercise, prompt in exercises_prompts.items():
        print(f"\n生成: {exercise}")

        # 上传参考图
        img_url = kling.upload_image(f"references/{exercise}.jpg")

        # 生成视频
        video_url = kling.generate_video(
            image_url=img_url,
            prompt=prompt,
            duration=5,
            motion_intensity="medium",
            camera_motion="static"
        )

        # 下载
        output_path = f"videos/{exercise}_demo.mp4"
        kling.download_video(video_url, output_path)

        results.append({
            "exercise": exercise,
            "video": output_path
        })

        # 避免rate limit
        time.sleep(30)

    return results

# 步骤3: 后期处理 (添加标注/音效/BGM)
def post_process_videos(results):
    """视频后期处理"""

    for item in results:
        exercise = item["exercise"]
        video_path = item["video"]

        # 添加动作名称字幕
        subprocess.run([
            "ffmpeg", "-i", video_path,
            "-vf", f"drawtext=text='{exercise.upper()}':fontsize=48:fontcolor=white:x=50:y=50",
            f"videos/{exercise}_labeled.mp4"
        ])

        # 添加音效 (使用ElevenLabs生成的动作音效)
        sfx_path = f"sfx/{exercise}_sound.mp3"
        subprocess.run([
            "ffmpeg",
            "-i", f"videos/{exercise}_labeled.mp4",
            "-i", sfx_path,
            "-c:v", "copy", "-c:a", "aac",
            "-map", "0:v:0", "-map", "1:a:0",
            f"videos/{exercise}_final.mp4"
        ])

    print("\n✓ 所有视频后期处理完成!")

# 执行完整流程
print("步骤1: 生成参考图")
generate_reference_images()

print("\n步骤2: 批量生成视频")
results = batch_generate_videos()

print("\n步骤3: 后期处理")
post_process_videos(results)

print("\n🎉 完成! 10个教学视频已生成")
```

---

## 25.7 成本分析

```python
# Kling AI成本计算

# 会员价格
MEMBERSHIP = {
    "月费": 99,  # ¥ (约$15)
    "积分": 3300
}

# 积分消耗
CREDIT_COST = {
    "5秒标准": 35,
    "5秒高质量": 80,
    "10秒标准": 70,
    "10秒高质量": 150
}

# 月产能
monthly_capacity = {
    "5秒标准": 3300 / 35,   # ≈ 94个视频
    "5秒高质量": 3300 / 80,  # ≈ 41个视频
    "10秒高质量": 3300 / 150 # ≈ 22个视频
}

# 单个成本
cost_per_video = {
    "5秒标准": 99 / 94,   # ≈ ¥1.05 ($0.16)
    "5秒高质量": 99 / 41,  # ≈ ¥2.41 ($0.37)
    "10秒高质量": 99 / 22  # ≈ ¥4.50 ($0.68)
}

# vs 竞品
COMPETITORS = {
    "Runway Gen-3 (10秒)": 1.12,  # $
    "Pika 1.5 (5秒)": 0.20,       # $
    "本地SVD": 0.05               # $ (电费)
}

print("Kling AI成本优势:")
print(f"- 5秒高质量: ¥2.41 ($0.37) vs Runway $0.45 → 便宜18%")
print(f"- 10秒高质量: ¥4.50 ($0.68) vs Runway $1.12 → 便宜39%")
print(f"- 质量: 真人动作Kling > Runway")
print(f"结论: 真人运动视频,Kling性价比最高!")
```

---

## 25.8 最佳实践总结

### 25.8.1 提示词黄金法则

```python
GOLDEN_RULES = {
    "1. 动作描述要具体": {
        "❌": "person exercising",
        "✅": "athlete performs squat, descending slowly, rising powerfully"
    },

    "2. 强调动作质量": {
        "❌": "doing squat",
        "✅": "performing perfect squat with proper form, controlled motion"
    },

    "3. 包含速度节奏": {
        "❌": "person moves",
        "✅": "slowly descending, then explosively rising"
    },

    "4. 描述表情/情绪": {
        "❌": "athlete working out",
        "✅": "athlete with focused expression, showing effort"
    },

    "5. 指定相机视角": {
        "❌": "person running",
        "✅": "runner sprinting, camera tracking from side"
    }
}
```

### 25.8.2 参考图要求

```
高质量参考图特征:
✅ 分辨率: 1920×1080 或更高
✅ 光线: 均匀充足,无过曝/欠曝
✅ 主体: 清晰,占画面40-60%
✅ 背景: 简洁或真实健身房环境
✅ 姿势: 标准,易于识别

❌ 避免:
- 模糊/低分辨率
- 主体过小 (<30%画面)
- 复杂凌乱背景
- 遮挡/截断
- 极端角度
```

### 25.8.3 运动强度选择指南

```
低强度 (10-30%):
- 微表情变化
- 轻微呼吸起伏
- 静态姿势保持
→ 用于: 瑜伽/冥想/静态展示

中强度 (30-70%):
- 标准健身动作
- 走路/伸展
- 器械使用
→ 用于: 大部分健身教学 ⭐推荐

高强度 (70-100%):
- 跳跃/奔跑
- 爆发力动作
- 快速切换
→ 用于: 运动竞技/HIIT训练
```

---

## 25.9 常见问题排查

### 问题1: 动作不自然/抖动

```
原因:
- 提示词描述不够详细
- 参考图姿势不标准
- 运动强度设置过高

解决:
1. 细化提示词,强调"smooth, controlled, natural"
2. 使用标准姿势参考图
3. 降低运动强度到"中"
4. 添加"maintaining proper form throughout"
```

### 问题2: 面部表情僵硬

```
原因:
- 参考图面部不清晰
- 未在提示词中描述表情

解决:
1. 使用高清参考图,面部清晰
2. 提示词加入表情描述:
   "with natural facial expression, focused eyes"
3. 避免过度化妆/滤镜的参考图
```

### 问题3: 衣物/头发穿模

```
原因:
- 复杂服装难以模拟
- 动作幅度过大

解决:
1. 选择简单贴身运动服
2. 长发建议扎起
3. 降低运动强度
4. 分段生成复杂动作
```

---

## 25.10 总结

**Kling AI核心价值**:
- 真人动作质量行业领先 ⭐⭐⭐⭐⭐
- 表情/物理细节极其真实
- 健身/运动场景最佳选择
- 性价比高 (比Runway便宜30-40%)

**适用场景矩阵**:
```
健身教学视频: Kling AI ⭐⭐⭐⭐⭐
舞蹈动作展示: Kling AI ⭐⭐⭐⭐⭐
体育运动慢动作: Kling AI ⭐⭐⭐⭐⭐
产品使用演示: Kling AI ⭐⭐⭐⭐
场景渲染/相机运动: Runway Gen-3 ⭐⭐⭐⭐⭐
创意效果/低成本: Pika 1.5 ⭐⭐⭐
```

**学习建议**:
1. 从简单静态→单一动作→复杂动作序列
2. 建立常用提示词模板库
3. 收集高质量参考图
4. 多测试不同运动强度

**下一章**: 第26章将介绍Sora等文生视频工具,以及视频编辑后期流程。

Kling AI让真人健身教学视频生成成为可能,这在以前需要专业摄影团队!
