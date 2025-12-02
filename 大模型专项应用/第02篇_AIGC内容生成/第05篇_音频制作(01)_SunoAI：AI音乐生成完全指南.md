# 第30章 音乐生成 Suno AI完全精通

> **学习目标**: 掌握Suno AI的音乐生成,为视频添加专业BGM
>
> **难度**: ⭐⭐
> **推荐度**: ⭐⭐⭐⭐⭐ (视频制作必备)

---

## 30.1 为什么视频需要BGM?

**质感对比**:
```
无BGM视频: 像监控录像,缺乏情绪
有BGM视频: 电影级质感,沉浸感强

数据:
- 有BGM的营销视频点击率提升40%+
- 观看完成率提升60%+
- 转化率提升35%+
```

**Suno AI的价值**:
- 零版权风险 (你拥有版权)
- 定制化 (完美匹配视频氛围)
- 成本低 (vs 购买版权音乐)
- 快速 (~2分钟生成)

---

## 30.2 Suno AI快速上手

### 30.2.1 注册与定价

```
免费版:
- 50 credits/天
- 可生成 ~10首歌/天
- 非商用

Pro Plan ($10/月):
- 2500 credits/月
- 商业授权 ✅
- 优先队列
- 推荐!

Premier Plan ($30/月):
- 10000 credits/月
- 更快生成
```

### 30.2.2 基础生成

```python
# Web界面使用 (app.suno.ai)

步骤:
1. 输入歌曲描述
2. (可选) 输入歌词
3. 选择风格
4. Generate

示例提示词:
"Energetic electronic workout music,
 upbeat tempo 128 BPM,
 motivational and powerful,
 no vocals"

# 2分钟后获得2个版本,选择最佳下载
```

### 30.2.3 提示词工程

**结构化提示词**:
```
[风格] + [情绪] + [BPM] + [乐器] + [用途说明]

示例:

健身视频BGM:
"Electronic dance music, energetic and motivational,
 fast tempo 140 BPM, synthesizers and heavy bass,
 instrumental only, perfect for fitness training video"

瑜伽视频BGM:
"Ambient meditation music, calm and peaceful,
 slow tempo 60 BPM, soft piano and nature sounds,
 instrumental, suitable for yoga and relaxation"

产品展示BGM:
"Modern corporate background music, professional and uplifting,
 medium tempo 110 BPM, acoustic guitar and light percussion,
 clean instrumental, for product showcase video"

励志视频BGM:
"Cinematic inspiring orchestral, emotional and uplifting,
 medium-slow tempo 90 BPM, strings and piano,
 instrumental with subtle choir, for motivational content"
```

**风格关键词库**:
```python
MUSIC_STYLES = {
    "节奏型": [
        "electronic dance", "techno", "house", "dubstep",
        "hip hop beats", "trap", "drum and bass"
    ],
    "氛围型": [
        "ambient", "chillout", "lo-fi", "downtempo",
        "meditation", "new age", "spa music"
    ],
    "企业/商务": [
        "corporate", "uplifting corporate", "tech background",
        "modern business", "professional"
    ],
    "电影感": [
        "cinematic", "orchestral", "epic", "dramatic",
        "trailer music", "film score"
    ],
    "流行": [
        "pop", "indie pop", "synth pop", "dream pop"
    ],
    "摇滚/金属": [
        "rock", "hard rock", "metal", "punk",
        "alternative rock"
    ]
}

MOODS = [
    "energetic", "calm", "motivational", "inspiring",
    "dark", "mysterious", "happy", "sad", "powerful",
    "relaxing", "aggressive", "peaceful", "dramatic"
]

BPM_GUIDE = {
    "非常慢 (40-60)": "冥想/瑜伽/深度放松",
    "慢 (60-90)": "情感视频/故事叙述",
    "中速 (90-120)": "通用BGM/产品展示",
    "快 (120-140)": "运动/活力场景",
    "非常快 (140-180)": "高强度健身/极限运动"
}
```

---

## 30.3 歌词模式 vs 纯音乐

**纯音乐 (Instrumental)**:
```
提示词加上:
"instrumental only, no vocals"

适用:
- 99%的背景音乐需求
- 避免歌词干扰画面

优点:
- 不抢视频主体
- 通用性强
- 易于循环使用
```

**带歌词**:
```
适用场景:
- 品牌主题曲
- 宣传视频主题歌
- 特定情节需要歌词配合

自定义歌词:
Prompt:
"Upbeat pop song, inspiring and motivational"

Lyrics:
[Verse 1]
Wake up strong, it's a brand new day
Push your limits, find your way
...

[Chorus]
Rise up, don't give up
You're stronger than you know
...
```

---

## 30.4 Extend功能 (延长音乐)

```bash
问题: Suno默认生成2分钟歌曲,视频5分钟怎么办?

解决: Extend功能

使用:
1. 生成初始2分钟音乐
2. 点击Extend
3. 选择从哪里延伸 (结尾/中间)
4. 生成额外段落

# 可多次Extend,最长可达8-10分钟

注意:
- 每次Extend消耗credits
- 风格可能略有变化
- 建议: 计划好长度,一次性生成
```

---

## 30.5 API集成 (Python)

```python
import requests
import time

class SunoAPI:
    """Suno AI非官方API (基于Web接口逆向)"""

    def __init__(self, cookie):
        """
        cookie: 从浏览器获取Suno登录cookie
        """
        self.base_url = "https://studio-api.suno.ai"
        self.headers = {
            "Cookie": cookie,
            "Content-Type": "application/json"
        }

    def generate_music(
        self,
        prompt,
        lyrics="",
        make_instrumental=True,
        wait_audio=True
    ):
        """生成音乐"""

        # 创建任务
        payload = {
            "prompt": prompt,
            "make_instrumental": make_instrumental,
            "wait_audio": wait_audio
        }

        if lyrics and not make_instrumental:
            payload["lyrics"] = lyrics

        response = requests.post(
            f"{self.base_url}/api/generate/v2/",
            headers=self.headers,
            json=payload
        )

        task_ids = [clip["id"] for clip in response.json()["clips"]]

        # 轮询结果
        while True:
            status_response = requests.get(
                f"{self.base_url}/api/feed/?ids={','.join(task_ids)}",
                headers=self.headers
            )

            clips = status_response.json()

            all_completed = all(
                clip["status"] == "complete" for clip in clips
            )

            if all_completed:
                return [
                    {
                        "id": clip["id"],
                        "title": clip["title"],
                        "audio_url": clip["audio_url"],
                        "video_url": clip["video_url"],
                        "duration": clip["metadata"]["duration"]
                    }
                    for clip in clips
                ]

            time.sleep(10)

    def download_audio(self, audio_url, output_path):
        """下载音频"""
        audio_data = requests.get(audio_url).content
        with open(output_path, "wb") as f:
            f.write(audio_data)
        print(f"✓ Downloaded: {output_path}")

# 使用示例
suno = SunoAPI(cookie="your_cookie_here")

# 生成健身BGM
results = suno.generate_music(
    prompt="""
    High-energy electronic workout music,
    powerful and motivational,
    fast tempo 140 BPM,
    heavy bass and synth,
    instrumental only
    """,
    make_instrumental=True
)

# 下载最佳版本
for i, result in enumerate(results):
    suno.download_audio(
        result["audio_url"],
        f"fitness_bgm_v{i+1}.mp3"
    )

print("完成!")
```

---

## 30.6 实战: 为视频匹配BGM

### 30.6.1 视频分析

```python
# 1. 确定视频氛围和节奏

视频类型: 健身教学 - 深蹲动作示范
时长: 90秒
节奏: 中等偏快 (动作演示+讲解)
氛围: 专业、激励、有力量感

# 2. BGM需求
风格: Electronic/运动感
BPM: 120-130 (不要过快,配合动作演示)
情绪: Motivational, powerful, clean
时长: 90-120秒 (略长于视频,方便剪辑)

# 3. Suno提示词
prompt = """
Electronic fitness background music,
motivational and powerful,
medium-fast tempo 125 BPM,
clean synth and steady beat,
instrumental only,
professional gym training atmosphere,
duration around 2 minutes
"""
```

### 30.6.2 后期处理

```python
import subprocess

# 使用ffmpeg调整音频

# 1. 截取所需长度 (90秒)
subprocess.run([
    "ffmpeg", "-i", "suno_output.mp3",
    "-t", "90",
    "-c", "copy",
    "bgm_90s.mp3"
])

# 2. 调整音量 (降低,避免盖过讲解)
subprocess.run([
    "ffmpeg", "-i", "bgm_90s.mp3",
    "-filter:a", "volume=0.3",  # 降至30%
    "bgm_low_volume.mp3"
])

# 3. 淡入淡出
subprocess.run([
    "ffmpeg", "-i", "bgm_low_volume.mp3",
    "-af", "afade=t=in:st=0:d=2,afade=t=out:st=88:d=2",  # 前后2秒淡入淡出
    "bgm_final.mp3"
])

# 4. 合并到视频
subprocess.run([
    "ffmpeg",
    "-i", "video.mp4",  # 视频
    "-i", "bgm_final.mp3",  # 音频
    "-c:v", "copy",  # 视频流复制
    "-c:a", "aac",  # 音频编码
    "-map", "0:v:0",  # 视频从第一个输入
    "-map", "1:a:0",  # 音频从第二个输入
    "video_with_bgm.mp4"
])

print("✓ 视频+BGM合成完成!")
```

---

## 30.7 多场景BGM库构建

```python
# 项目: 建立可复用BGM库

BGM_LIBRARY = {
    "高强度训练": {
        "prompt": "Aggressive electronic gym music, high energy, 145 BPM, heavy bass, instrumental",
        "use_cases": ["HIIT", "力量训练", "跑步"]
    },

    "瑜伽放松": {
        "prompt": "Calm ambient meditation, peaceful, 55 BPM, soft piano and nature, instrumental",
        "use_cases": ["瑜伽", "拉伸", "冥想"]
    },

    "产品展示": {
        "prompt": "Modern corporate uplifting, professional, 110 BPM, acoustic and light synth, instrumental",
        "use_cases": ["产品介绍", "品牌视频", "教程"]
    },

    "励志motivational": {
        "prompt": "Cinematic inspiring orchestral, emotional uplifting, 95 BPM, strings and piano, instrumental",
        "use_cases": ["励志故事", "转变对比", "成功案例"]
    },

    "快节奏展示": {
        "prompt": "Upbeat tech corporate, energetic, 128 BPM, electronic and drums, instrumental",
        "use_cases": ["快速剪辑", "Montage", "精彩集锦"]
    }
}

# 批量生成
suno = SunoAPI(cookie="...")

for category, config in BGM_LIBRARY.items():
    print(f"\n生成: {category}")

    results = suno.generate_music(
        prompt=config["prompt"],
        make_instrumental=True
    )

    # 保存最佳版本
    suno.download_audio(
        results[0]["audio_url"],
        f"bgm_library/{category}.mp3"
    )

print("\nBGM库构建完成!")
print("包含5大类,覆盖常见视频场景")
```

---

## 30.8 成本与授权

```python
# 成本分析

# Pro Plan ($10/月, 2500 credits)
songs_per_month = 2500 / 10  # 每首约10 credits
# = 250首/月

cost_per_song = 10 / 250
# = $0.04/首

# vs 购买版权音乐
artlist_cost = 9.99 / 月  # 无限下载,但需订阅
epidemic_sound = 15 / 月
audiojungle = 19 / 首 (单曲购买)

# Suno优势:
# 1. 完全定制 (匹配视频氛围)
# 2. 你拥有100%版权
# 3. 成本极低 ($0.04/首)

# 商业授权:
# Pro及以上计划生成的音乐,你拥有商业使用权
# 可用于:
# - YouTube视频
# - 商业广告
# - 产品展示
# - 社交媒体营销

# 不能:
# - 转售音乐本身作为产品
# - 声称自己是原创作曲者
```

---

## 30.9 总结

**Suno AI核心价值**:
- 快速 (2分钟生成)
- 便宜 ($0.04/首)
- 定制化 (完美匹配视频)
- 零版权风险

**最佳实践**:
1. 提示词结构化 (风格+情绪+BPM+乐器)
2. 优先纯音乐 (instrumental)
3. 后期音量调整 (避免盖过主体)
4. 建立BGM库 (常用场景预生成)

**适用场景**:
✅ 短视频BGM (15s-5min)
✅ 教学视频配乐
✅ 产品展示背景
✅ 营销视频氛围

❌ 不适合: 需要特定歌手/乐队风格

**下一章**: 第31章将介绍ElevenLabs音效生成,为视频添加环境音和动作音效。

BGM是视频质感的关键,Suno让零音乐基础的人也能制作专业配乐!
