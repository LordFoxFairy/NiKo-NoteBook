# 第21章 图生视频(一) Runway Gen-3完全精通

> **学习目标**: 掌握Runway Gen-3的图生视频、参数调优及商业应用
>
> **难度**: ⭐⭐⭐
> **推荐度**: ⭐⭐⭐⭐⭐ (视频生成首选工具)

---

## 21.1 Runway Gen-3概览

**核心功能**:
- Text-to-Video: 文字直接生成视频
- Image-to-Video: 图片动起来 ⭐⭐⭐⭐⭐ (最常用)
- Video-to-Video: 视频风格转换

**定价** (2025):
```
Basic Plan ($12/月): 125 credits
Standard Plan ($28/月): 625 credits  ← 推荐
Pro Plan ($76/月): 2250 credits

视频成本:
- 5秒视频 (720p): ~10 credits ($0.45)
- 10秒视频 (720p): ~25 credits ($1.12)
```

**vs 竞品对比**:
| 工具 | 质量 | 速度 | 成本 | 控制力 |
|------|------|------|------|--------|
| Runway Gen-3 | ⭐⭐⭐⭐⭐ | 快 (2-3min) | 中 | 强 |
| Pika 1.5 | ⭐⭐⭐⭐ | 中 (3-5min) | 低 | 中 |
| Stable Video Diffusion | ⭐⭐⭐ | 快 (本地) | 低 (本地) | 弱 |
| Luma Dream Machine | ⭐⭐⭐⭐ | 快 (1-2min) | 低 | 弱 |

---

## 21.2 Image-to-Video实战

### 21.2.1 基础流程

```python
# Runway官网: https://runwayml.com

# Web界面使用:
1. 登录 → 选择Gen-3 Alpha
2. 上传图片 (推荐1024×576或1920×1080)
3. (可选) 输入运动提示词
4. 设置参数:
   - Duration: 5s / 10s
   - Seed: 随机或固定
5. Generate

# 运动提示词示例:
"Camera slowly zooms in on the athlete"
"The person starts to perform a squat motion"
"Camera pans left to right across the gym"
```

### 21.2.2 最佳实践

**图片要求**:
```
✅ 高质量图片 (1080p+)
✅ 主体清晰
✅ 光线充足
✅ 构图合理 (留出运动空间)

❌ 模糊图片
❌ 过度后期/滤镜
❌ 极端角度
```

**运动提示词技巧**:
```bash
# 相机运动
"Camera slowly zooms in"
"Camera pans from left to right"
"Camera tilts up revealing the scene"
"Dolly shot moving forward"
"Crane shot, camera rises up"

# 主体运动
"The athlete begins to run forward"
"The person lifts the dumbbell upwards"
"Slow motion of the squat movement"

# 环境变化
"Sunlight gradually fills the room"
"Leaves gently blow in the wind"
"Water ripples spread across the surface"

# 组合
"Camera slowly pushes in while the athlete starts jumping rope,
 dramatic lighting, cinematic motion"
```

### 21.2.3 API调用

```python
import requests
import time
from pathlib import Path

class RunwayGen3API:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.runwayml.com/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def upload_image(self, image_path):
        """上传图片获取URL"""
        # 实际实现需参考Runway API文档
        # 此处为伪代码示意
        files = {"file": open(image_path, "rb")}
        response = requests.post(
            f"{self.base_url}/upload",
            headers={"Authorization": f"Bearer {self.api_key}"},
            files=files
        )
        return response.json()["url"]

    def generate_video(
        self,
        image_url,
        prompt="",
        duration=5,
        seed=None
    ):
        """生成视频"""
        payload = {
            "model": "gen3alpha",
            "image": image_url,
            "prompt": prompt,
            "duration": duration,
            "seed": seed
        }

        # 创建任务
        response = requests.post(
            f"{self.base_url}/generate",
            headers=self.headers,
            json=payload
        )
        task_id = response.json()["id"]

        # 轮询状态
        while True:
            status_response = requests.get(
                f"{self.base_url}/tasks/{task_id}",
                headers=self.headers
            )
            status = status_response.json()

            if status["status"] == "completed":
                return status["output"]["video_url"]
            elif status["status"] == "failed":
                raise Exception(f"Generation failed: {status['error']}")

            time.sleep(10)  # 10秒后重新检查

    def batch_generate(self, images_with_prompts):
        """批量生成"""
        results = []

        for img_path, prompt in images_with_prompts:
            print(f"Processing: {img_path}")

            # 上传
            img_url = self.upload_image(img_path)

            # 生成
            video_url = self.generate_video(
                image_url=img_url,
                prompt=prompt,
                duration=5
            )

            # 下载
            video_data = requests.get(video_url).content
            output_path = img_path.replace(".jpg", ".mp4")
            with open(output_path, "wb") as f:
                f.write(video_data)

            results.append(output_path)
            print(f"✓ Saved: {output_path}")

        return results

# 使用示例
api = RunwayGen3API(api_key="your_api_key")

# 单个生成
video_url = api.generate_video(
    image_url="https://your-image-url.jpg",
    prompt="Camera slowly zooms in on the athlete in the gym",
    duration=5
)

# 批量生成
tasks = [
    ("fitness_1.jpg", "Athlete starts doing push-ups"),
    ("fitness_2.jpg", "Camera pans across the modern gym"),
    ("fitness_3.jpg", "Person lifts dumbbell in slow motion")
]

results = api.batch_generate(tasks)
```

---

## 21.3 高级技巧

### 21.3.1 Motion Brush (运动笔刷)

```
Web界面功能:
1. 上传图片
2. 使用笔刷工具选择想要运动的区域
3. 设置运动方向和强度
4. 生成

适用场景:
- 精确控制局部运动
- 背景静止,主体运动
- 复杂场景分区控制

示例:
图片: 健身房内景,一个人站立
Motion Brush:
- 选中人物 → 向上运动 (模拟跳跃)
- 选中窗帘 → 轻微飘动
- 其他区域静止
→ 生成: 人跳跃,窗帘飘动,其余静止
```

### 21.3.2 Camera Control (相机控制)

```bash
# 精准相机运动参数

Zoom:
- "Zoom in" / "Zoom out"
- "Slow zoom in" / "Fast zoom in"

Pan:
- "Pan left" / "Pan right"
- "Pan up" / "Pan down"

Tilt:
- "Tilt up" / "Tilt down"

Dolly:
- "Dolly forward" / "Dolly backward"

Orbit:
- "Orbit around the subject clockwise"

组合:
"Slow dolly forward while tilting up slightly,
 revealing the gym ceiling"
```

### 21.3.3 Director Mode

```
高级功能 (Pro Plan):

功能:
- 更长视频 (最高18秒)
- 更精细控制
- 关键帧编辑

使用:
1. 设置起始帧 (图片1)
2. 设置结束帧 (图片2)
3. Runway生成中间过渡

适用:
- 复杂运动序列
- 多阶段动作
- 故事性视频
```

---

## 21.4 实战案例: 健身教学视频

```python
# 项目: 将10张静态动作图转为教学视频

动作列表:
1. 深蹲起始姿势
2. 深蹲下蹲阶段
3. 深蹲最低点
4. 深蹲起身
5. 深蹲完成姿势
...

工作流:
1. 准备图片 (ControlNet OpenPose生成标准姿势)
2. 为每张图编写运动提示词
3. Runway Gen-3生成5秒过渡视频
4. FFmpeg拼接为完整教学视频

# 运动提示词设计
prompts = {
    "squat_start.jpg": "Athlete maintains starting position, slight breathing movement, static camera",
    "squat_down.jpg": "Slow motion transition to squat position, camera slightly tilts down",
    "squat_bottom.jpg": "Hold at bottom position, camera slowly zooms in on form",
    "squat_up.jpg": "Powerful upward motion returning to standing, camera tilts up",
    "squat_complete.jpg": "Return to starting position, camera pans out revealing full scene"
}

# 批量生成
api = RunwayGen3API(api_key="YOUR_KEY")
videos = []

for img_path, prompt in prompts.items():
    video_url = api.generate_video(
        image_url=api.upload_image(img_path),
        prompt=prompt,
        duration=5
    )
    videos.append(download_video(video_url, img_path.replace(".jpg", ".mp4")))

# FFmpeg拼接
import subprocess

# 创建文件列表
with open("videos.txt", "w") as f:
    for video in videos:
        f.write(f"file '{video}'\n")

# 拼接
subprocess.run([
    "ffmpeg", "-f", "concat", "-safe", "0",
    "-i", "videos.txt",
    "-c", "copy",
    "squat_tutorial_complete.mp4"
])

print("完成! 教学视频已生成: squat_tutorial_complete.mp4")
```

---

## 21.5 成本优化

```python
# Runway成本管理

# 策略1: 优先5秒而非10秒
# 5秒: ~10 credits ($0.45)
# 10秒: ~25 credits ($1.12)
# 节省: 用2个5秒拼接 < 1个10秒

# 策略2: 复用seed
# 满意的运动 → 记录seed
# 相似场景复用seed

# 策略3: 本地预览
# 用SVD本地快速测试
# 确定满意后再用Runway生成高质量版

# 策略4: 降分辨率测试
# 测试阶段用540p
# 最终版用1080p

# 成本对比 (月生成100个5秒视频)
runway_cost = 100 * 0.45  # $45
pika_cost = 100 * 0.20    # $20 (更便宜但质量略低)
local_svd = 10  # 电费 (质量最低但免费)

print("Runway适合: 高质量商业项目")
print("Pika适合: 预算有限的中小项目")
print("SVD适合: 大批量测试/原型验证")
```

---

## 21.6 总结

**Runway Gen-3优势**:
- 质量最高 (目前市场领先)
- 运动自然流畅
- 相机控制精准
- API支持批量

**局限**:
- 成本较高
- 视频时长受限 (最高18秒)
- 闭源,无法本地部署

**推荐使用场景**:
✅ 商业视频制作
✅ 高质量营销物料
✅ 影视概念验证
✅ 社交媒体精品内容

**下一章**: 第25章将介绍Kling AI,专攻真人动作视频生成(特别适合健身场景)。
