# 第04篇_视频生成(05)_FFmpeg：视频编辑与后期处理完全指南

> **难度**: ⭐⭐⭐ | **推荐度**: ⭐⭐⭐⭐⭐

## 5.1 为什么需要FFmpeg

### 5.1.1 AIGC视频的后期需求

```python
# AIGC生成的视频常见问题

AIGC_VIDEO_ISSUES = {
    "Runway/Pika/Kling": {
        "时长": "仅5-10秒,需拼接",
        "音频": "无BGM/音效",
        "格式": "可能不兼容",
        "分辨率": "需要调整"
    },

    "SVD": {
        "时长": "仅2-4秒",
        "帧率": "7fps,不够流畅",
        "分辨率": "576×1024,需放大"
    },

    "后期需求": {
        "拼接": "多段视频合成",
        "音频": "添加BGM/音效/旁白",
        "转码": "格式转换",
        "调色": "颜色校正",
        "字幕": "添加文字",
        "特效": "转场/滤镜"
    }
}

# FFmpeg: 一站式解决方案
```

### 5.1.2 FFmpeg优势

| 对比项 | FFmpeg | Premiere Pro | Final Cut | DaVinci |
|--------|--------|--------------|-----------|---------|
| **成本** | 免费 | $23/月 | $300一次性 | 免费+付费 |
| **速度** | 极快(命令行) | 慢 | 中 | 慢 |
| **自动化** | ✅ 脚本化 | ❌ | ❌ | 部分 |
| **学习曲线** | 陡峭 | 中 | 中 | 陡峭 |
| **批量处理** | ✅ 完美 | ❌ | ❌ | 部分 |
| **推荐场景** | 批量/自动化 | 精细剪辑 | Mac用户 | 调色 |

---

## 5.2 FFmpeg安装

### 5.2.1 各平台安装

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg

# macOS (Homebrew)
brew install ffmpeg

# Windows
# 下载: https://ffmpeg.org/download.html
# 解压并添加到PATH

# 验证安装
ffmpeg -version
```

### 5.2.2 编译选项 (可选)

```bash
# 完整功能编译 (支持GPU加速等)

# 安装依赖
sudo apt install \
  libx264-dev libx265-dev libvpx-dev \
  libopus-dev libmp3lame-dev \
  libaom-dev libsvtav1-dev \
  nvidia-cuda-toolkit  # NVIDIA GPU加速

# 编译FFmpeg
git clone https://git.ffmpeg.org/ffmpeg.git
cd ffmpeg
./configure --enable-gpl --enable-libx264 --enable-libx265 \
            --enable-libvpx --enable-libopus --enable-libmp3lame \
            --enable-cuda-nvcc --enable-cuvid --enable-nvenc
make -j8
sudo make install
```

---

## 5.3 核心命令

### 5.3.1 基础操作

```bash
# 1. 转码 (格式转换)
ffmpeg -i input.mov -c:v libx264 -c:a aac output.mp4
# -c:v: 视频编码器
# -c:a: 音频编码器

# 2. 裁剪时长
ffmpeg -i input.mp4 -ss 00:00:05 -t 00:00:10 -c copy output.mp4
# -ss: 开始时间 (5秒)
# -t: 持续时长 (10秒)
# -c copy: 不重新编码,快速

# 3. 调整分辨率
ffmpeg -i input.mp4 -vf scale=1920:1080 output.mp4
# scale=width:height
# scale=1920:-1  (保持宽高比)

# 4. 调整帧率
ffmpeg -i input.mp4 -r 30 output.mp4
# -r 30: 30fps

# 5. 调整码率 (压缩)
ffmpeg -i input.mp4 -b:v 2M -b:a 128k output.mp4
# -b:v: 视频码率 2Mbps
# -b:a: 音频码率 128kbps

# 6. 提取音频
ffmpeg -i video.mp4 -vn -c:a copy audio.aac
# -vn: 不包含视频

# 7. 提取视频 (无音频)
ffmpeg -i video.mp4 -an -c:v copy video_only.mp4
# -an: 不包含音频
```

### 5.3.2 音频操作

```bash
# 1. 添加BGM
ffmpeg -i video.mp4 -i bgm.mp3 \
  -c:v copy -c:a aac \
  -map 0:v:0 -map 1:a:0 \
  output.mp4

# 2. 混合音频 (视频原音 + BGM)
ffmpeg -i video.mp4 -i bgm.mp3 \
  -filter_complex "[0:a]volume=0.3[a0];[1:a]volume=0.2[a1];[a0][a1]amix=inputs=2[aout]" \
  -map 0:v -map "[aout]" \
  output.mp4
# [0:a]volume=0.3: 原音降到30%
# [1:a]volume=0.2: BGM 20%

# 3. 音频淡入淡出
ffmpeg -i input.mp4 \
  -af "afade=t=in:st=0:d=2,afade=t=out:st=8:d=2" \
  output.mp4
# 前2秒淡入,最后2秒淡出

# 4. 调整音量
ffmpeg -i input.mp4 -af "volume=1.5" output.mp4
# 1.5倍音量

# 5. 降噪
ffmpeg -i input.mp4 -af "highpass=f=200,lowpass=f=3000" output.mp4
# 过滤低频(<200Hz)和高频(>3000Hz)噪音
```

### 5.3.3 视频拼接

```bash
# 方法1: Concat Demuxer (推荐,快速)

# 创建文件列表 videos.txt:
# file 'video1.mp4'
# file 'video2.mp4'
# file 'video3.mp4'

ffmpeg -f concat -safe 0 -i videos.txt -c copy output.mp4
# -c copy: 不重新编码,极快

# 方法2: Filter Complex (支持转场)
ffmpeg -i video1.mp4 -i video2.mp4 \
  -filter_complex "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[outv][outa]" \
  -map "[outv]" -map "[outa]" \
  output.mp4
# n=2: 2个输入
# v=1: 1个视频流
# a=1: 1个音频流

# 方法3: Python自动化
import subprocess

video_files = [
    "squat_front.mp4",
    "squat_side.mp4",
    "squat_back.mp4"
]

# 创建列表文件
with open("concat_list.txt", "w") as f:
    for video in video_files:
        f.write(f"file '{video}'\\n")

# 拼接
subprocess.run([
    "ffmpeg", "-f", "concat", "-safe", "0",
    "-i", "concat_list.txt",
    "-c", "copy",
    "final_squat.mp4"
])
```

---

## 5.4 实战案例

### 5.4.1 健身教学视频完整制作

```python
# 需求: 3个角度深蹲教学 + BGM + 字幕

# 步骤1: 拼接3个角度视频 (横向)
ffmpeg -i squat_front.mp4 -i squat_side.mp4 -i squat_back.mp4 \
  -filter_complex "\
    [0:v]scale=640:360[v0]; \
    [1:v]scale=640:360[v1]; \
    [2:v]scale=640:360[v2]; \
    [v0][v1][v2]hstack=inputs=3[outv]" \
  -map "[outv]" \
  video_3panel.mp4
# hstack: 横向拼接 (vs vstack纵向)

# 步骤2: 添加BGM
ffmpeg -i video_3panel.mp4 -i workout_bgm.mp3 \
  -filter_complex "[1:a]volume=0.3[a1]" \
  -map 0:v -map "[a1]" -shortest \
  video_with_bgm.mp4
# -shortest: 以最短流为准

# 步骤3: 添加字幕 (烧录)
ffmpeg -i video_with_bgm.mp4 \
  -vf "drawtext=text='深蹲标准动作':fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:fontsize=48:fontcolor=white:x=(w-text_w)/2:y=50" \
  final_output.mp4
# x=(w-text_w)/2: 水平居中
# y=50: 距顶部50px

# 步骤4: 压缩优化
ffmpeg -i final_output.mp4 \
  -c:v libx264 -crf 23 -preset medium \
  -c:a aac -b:a 128k \
  final_compressed.mp4
# -crf 23: 质量 (0-51,越低越好,23推荐)
# -preset medium: 速度 (slow质量高,fast速度快)
```

### 5.4.2 批量视频处理脚本

```python
# batch_process.py - 批量添加BGM和字幕

import subprocess
from pathlib import Path

def add_bgm_and_title(video_path, bgm_path, title, output_path):
    """为单个视频添加BGM和字幕"""

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-i", str(bgm_path),
        "-filter_complex",
        f"[1:a]volume=0.25[a1]; [0:v]drawtext=text='{title}':fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:fontsize=36:fontcolor=white:x=(w-text_w)/2:y=30[outv]",
        "-map", "[outv]",
        "-map", "[a1]",
        "-shortest",
        "-c:v", "libx264",
        "-c:a", "aac",
        str(output_path)
    ]

    subprocess.run(cmd, check=True)

# 批量处理
video_dir = Path("./raw_videos/")
output_dir = Path("./processed_videos/")
output_dir.mkdir(exist_ok=True)

bgm = "bgm.mp3"

videos_with_titles = {
    "squat.mp4": "深蹲标准动作",
    "pushup.mp4": "俯卧撑技巧",
    "plank.mp4": "平板支撑要点"
}

for video_file, title in videos_with_titles.items():
    input_path = video_dir / video_file
    output_path = output_dir / f"final_{video_file}"

    print(f"Processing: {video_file}")
    add_bgm_and_title(input_path, bgm, title, output_path)

print("Batch processing completed!")
```

### 5.4.3 GIF转换

```python
# 视频转GIF (高质量)

# 方法1: 基础转换
ffmpeg -i input.mp4 -vf "fps=10,scale=640:-1" output.gif
# fps=10: 10帧/秒
# scale=640:-1: 宽度640,高度自适应

# 方法2: 优化调色板 (推荐)
ffmpeg -i input.mp4 -vf "fps=10,scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" output.gif
# palettegen: 生成最优调色板
# 文件更小,质量更好

# 方法3: 循环GIF
ffmpeg -stream_loop -1 -i input.mp4 -t 10 -vf "fps=15,scale=480:-1" loop.gif
# -stream_loop -1: 无限循环
# -t 10: 10秒
```

---

## 5.5 GPU加速 (NVIDIA)

### 5.5.1 NVENC编码

```bash
# 使用NVIDIA GPU硬件编码 (极快)

# H.264编码
ffmpeg -i input.mp4 \
  -c:v h264_nvenc -preset fast -b:v 5M \
  -c:a copy \
  output.mp4

# H.265编码 (更高压缩率)
ffmpeg -i input.mp4 \
  -c:v hevc_nvenc -preset slow -b:v 3M \
  -c:a copy \
  output.mp4

# 速度对比:
# CPU (libx264): 1× 实时速度
# GPU (h264_nvenc): 5-10× 实时速度

# 质量对比:
# CPU: ⭐⭐⭐⭐⭐
# GPU: ⭐⭐⭐⭐ (略低,但可接受)
```

### 5.5.2 批量GPU加速

```python
# gpu_batch_encode.py
import subprocess
from pathlib import Path
import concurrent.futures

def encode_video_gpu(input_path, output_path):
    """GPU加速编码"""
    cmd = [
        "ffmpeg",
        "-hwaccel", "cuda",  # 启用CUDA加速
        "-i", str(input_path),
        "-c:v", "h264_nvenc",
        "-preset", "fast",
        "-b:v", "5M",
        "-c:a", "aac",
        "-b:a", "128k",
        str(output_path)
    ]

    subprocess.run(cmd, check=True)
    print(f"Encoded: {output_path}")

# 多GPU并行 (4× RTX 4090)
input_dir = Path("./raw/")
output_dir = Path("./encoded/")
output_dir.mkdir(exist_ok=True)

video_files = list(input_dir.glob("*.mp4"))

# 并行处理 (每个GPU 1个任务)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = []
    for video in video_files:
        output_path = output_dir / f"encoded_{video.name}"
        future = executor.submit(encode_video_gpu, video, output_path)
        futures.append(future)

    for future in concurrent.futures.as_completed(futures):
        future.result()

# 吞吐量: 4GPU × 10× 实时 = 40× 实时速度
# 1小时视频仅需90秒!
```

---

## 5.6 Python集成 (ffmpeg-python)

### 5.6.1 安装与基础使用

```bash
pip install ffmpeg-python
```

```python
import ffmpeg

# 示例1: 简单转码
(
    ffmpeg
    .input('input.mp4')
    .output('output.mp4', vcodec='libx264', acodec='aac')
    .run()
)

# 示例2: 调整分辨率
(
    ffmpeg
    .input('input.mp4')
    .filter('scale', 1920, 1080)
    .output('output.mp4')
    .run()
)

# 示例3: 添加水印
(
    ffmpeg
    .input('video.mp4')
    .overlay(ffmpeg.input('watermark.png'), x=10, y=10)
    .output('output.mp4')
    .run()
)

# 示例4: 视频拼接
concat = ffmpeg.concat(
    ffmpeg.input('video1.mp4'),
    ffmpeg.input('video2.mp4'),
    v=1, a=1
)
concat.output('output.mp4').run()
```

### 5.6.2 高级应用

```python
# 完整视频处理流水线

import ffmpeg

def process_video_pipeline(
    input_video,
    bgm_audio,
    watermark_img,
    output_video,
    title_text
):
    """
    综合视频处理:
    1. 调整分辨率到1080p
    2. 添加水印
    3. 添加BGM
    4. 添加字幕
    5. 压缩优化
    """

    # 输入流
    video = ffmpeg.input(input_video)
    audio = ffmpeg.input(bgm_audio)
    watermark = ffmpeg.input(watermark_img)

    # 视频处理
    video = (
        video
        .filter('scale', 1920, 1080)  # 调整分辨率
        .filter('drawtext',
                text=title_text,
                fontsize=48,
                fontcolor='white',
                x='(w-text_w)/2',
                y=50)  # 添加字幕
        .overlay(watermark, x=10, y=10)  # 添加水印
    )

    # 音频处理
    audio = audio.filter('volume', 0.3)

    # 合并
    output = ffmpeg.output(
        video, audio,
        output_video,
        vcodec='libx264',
        acodec='aac',
        crf=23,
        preset='medium'
    )

    # 执行
    output.run(overwrite_output=True)

# 使用
process_video_pipeline(
    input_video='squat.mp4',
    bgm_audio='workout_bgm.mp3',
    watermark_img='logo.png',
    output_video='final_squat.mp4',
    title_text='深蹲标准动作教学'
)
```

---

## 5.7 常用滤镜

### 5.7.1 视频滤镜

```bash
# 1. 模糊
ffmpeg -i input.mp4 -vf "boxblur=5:1" output.mp4
# boxblur=luma_radius:luma_power

# 2. 锐化
ffmpeg -i input.mp4 -vf "unsharp=5:5:1.0" output.mp4

# 3. 去抖动
ffmpeg -i input.mp4 -vf "deshake" output.mp4

# 4. 色彩调整
ffmpeg -i input.mp4 -vf "eq=brightness=0.1:saturation=1.2" output.mp4
# brightness: 亮度 (-1 to 1)
# saturation: 饱和度 (0 to 3)

# 5. 黑白
ffmpeg -i input.mp4 -vf "hue=s=0" output.mp4

# 6. 复古滤镜
ffmpeg -i input.mp4 -vf "curves=vintage" output.mp4

# 7. 镜像翻转
ffmpeg -i input.mp4 -vf "hflip" output.mp4  # 水平
ffmpeg -i input.mp4 -vf "vflip" output.mp4  # 垂直

# 8. 旋转
ffmpeg -i input.mp4 -vf "rotate=PI/4" output.mp4  # 45度
ffmpeg -i input.mp4 -vf "transpose=1" output.mp4  # 90度顺时针
```

### 5.7.2 转场效果

```bash
# 淡入淡出
ffmpeg -i input.mp4 -vf "fade=in:0:30,fade=out:270:30" output.mp4
# fade=in:start_frame:duration
# 前30帧淡入,270帧开始淡出30帧

# 交叉淡化 (两段视频)
ffmpeg -i video1.mp4 -i video2.mp4 \
  -filter_complex "\
    [0:v]fade=out:st=4:d=1[v0]; \
    [1:v]fade=in:st=0:d=1[v1]; \
    [v0][v1]xfade=transition=fade:duration=1:offset=4[outv]" \
  -map "[outv]" \
  output.mp4
# xfade: 交叉淡化
# offset=4: 4秒处开始转场
```

---

## 5.8 故障排查

### 5.8.1 常见错误

```python
# 错误1: "不支持的编解码器"
错误: Unknown encoder 'libx264'
解决: 重新编译FFmpeg,启用libx264支持

# 错误2: "无法找到流"
错误: Stream map '0:v' matches no streams
解决: 检查输入文件是否有视频流
       ffmpeg -i input.mp4  # 查看流信息

# 错误3: "帧率不匹配"
解决: 统一帧率
      ffmpeg -i input.mp4 -r 30 output.mp4

# 错误4: "音视频不同步"
解决: 重新编码
      ffmpeg -i input.mp4 -c:v libx264 -c:a aac output.mp4

# 错误5: "文件过大"
解决: 调整码率
      ffmpeg -i input.mp4 -b:v 1M -b:a 128k output.mp4
```

### 5.8.2 性能优化

```python
# 技巧1: 使用-c copy (不重新编码)
ffmpeg -i input.mp4 -ss 10 -t 20 -c copy output.mp4
# 100× 速度提升

# 技巧2: 多线程
ffmpeg -i input.mp4 -threads 8 output.mp4

# 技巧3: GPU加速
ffmpeg -hwaccel cuda -i input.mp4 -c:v h264_nvenc output.mp4

# 技巧4: 降低preset (fast/ultrafast)
ffmpeg -i input.mp4 -preset ultrafast output.mp4
# 速度提升,但质量/压缩率降低

# 技巧5: Two-Pass编码 (最优质量)
ffmpeg -i input.mp4 -c:v libx264 -b:v 2M -pass 1 -f null /dev/null
ffmpeg -i input.mp4 -c:v libx264 -b:v 2M -pass 2 output.mp4
```

---

## 5.9 完整自动化案例

### 5.9.1 健身视频批量生产系统

```python
# fitness_video_automation.py
import subprocess
import json
from pathlib import Path

class FitnessVideoProducer:
    """健身视频自动化生产系统"""

    def __init__(self, config_file="config.json"):
        with open(config_file) as f:
            self.config = json.load(f)

        self.bgm = self.config["bgm_path"]
        self.logo = self.config["logo_path"]

    def add_intro(self, video, intro_duration=3):
        """添加片头"""
        # 生成黑屏+标题
        subprocess.run([
            "ffmpeg",
            "-f", "lavfi", "-i", f"color=c=black:s=1920x1080:d={intro_duration}",
            "-vf", f"drawtext=text='健身教学系列':fontsize=72:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2",
            "intro.mp4"
        ])

        # 拼接片头+正片
        with open("concat.txt", "w") as f:
            f.write("file 'intro.mp4'\\n")
            f.write(f"file '{video}'\\n")

        subprocess.run([
            "ffmpeg", "-f", "concat", "-i", "concat.txt",
            "-c", "copy", "temp_with_intro.mp4"
        ])

        return "temp_with_intro.mp4"

    def add_bgm_and_watermark(self, video, output, title):
        """添加BGM和水印"""
        subprocess.run([
            "ffmpeg",
            "-i", video,
            "-i", self.bgm,
            "-i", self.logo,
            "-filter_complex",
            f"[2:v]scale=100:-1[logo]; [0:v][logo]overlay=W-w-10:10[v]; [1:a]volume=0.3[a]",
            "-map", "[v]",
            "-map", "[a]",
            "-shortest",
            "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            "-c:a", "aac",
            output
        ])

    def add_outro(self, video, outro_text="关注我们获取更多教学"):
        """添加片尾"""
        subprocess.run([
            "ffmpeg",
            "-f", "lavfi", "-i", "color=c=black:s=1920x1080:d=3",
            "-vf", f"drawtext=text='{outro_text}':fontsize=48:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2",
            "outro.mp4"
        ])

        # 拼接
        with open("concat_final.txt", "w") as f:
            f.write(f"file '{video}'\\n")
            f.write("file 'outro.mp4'\\n")

        subprocess.run([
            "ffmpeg", "-f", "concat", "-i", "concat_final.txt",
            "-c", "copy", "final_with_outro.mp4"
        ])

        return "final_with_outro.mp4"

    def produce(self, raw_video, output_path, title):
        """完整生产流程"""
        print(f"生产: {title}")

        # 1. 片头
        video = self.add_intro(raw_video)

        # 2. BGM + 水印
        self.add_bgm_and_watermark(video, "temp_processed.mp4", title)

        # 3. 片尾
        final = self.add_outro("temp_processed.mp4")

        # 4. 移动到输出
        subprocess.run(["mv", final, output_path])

        # 清理临时文件
        for temp in ["intro.mp4", "temp_with_intro.mp4", "temp_processed.mp4", "outro.mp4"]:
            Path(temp).unlink(missing_ok=True)

        print(f"完成: {output_path}")

# 使用
producer = FitnessVideoProducer("config.json")

videos = [
    ("raw_squat.mp4", "深蹲标准动作"),
    ("raw_pushup.mp4", "俯卧撑技巧"),
    ("raw_plank.mp4", "平板支撑要点")
]

for raw_video, title in videos:
    output = f"final_{raw_video}"
    producer.produce(raw_video, output, title)
```

---

## 5.10 总结

### 5.10.1 核心命令速查

```bash
# 转码
ffmpeg -i input.mp4 -c:v libx264 -c:a aac output.mp4

# 裁剪
ffmpeg -i input.mp4 -ss 10 -t 20 -c copy output.mp4

# 拼接
ffmpeg -f concat -i list.txt -c copy output.mp4

# 添加BGM
ffmpeg -i video.mp4 -i bgm.mp3 -map 0:v -map 1:a -shortest output.mp4

# 调整分辨率
ffmpeg -i input.mp4 -vf scale=1920:1080 output.mp4

# 压缩
ffmpeg -i input.mp4 -crf 28 -preset fast output.mp4

# GPU加速
ffmpeg -hwaccel cuda -i input.mp4 -c:v h264_nvenc output.mp4
```

### 5.10.2 最佳实践

1. **批量处理用脚本** - Python + FFmpeg自动化
2. **GPU加速** - NVENC提升5-10倍速度
3. **不重新编码** - 用`-c copy`极速裁剪
4. **质量优先** - CRF 18-23,preset medium/slow
5. **速度优先** - preset fast/ultrafast + GPU

### 5.10.3 学习资源

- 官方文档: https://ffmpeg.org/documentation.html
- Wiki: https://trac.ffmpeg.org/wiki
- 滤镜列表: https://ffmpeg.org/ffmpeg-filters.html

FFmpeg是AIGC视频后期的必备工具,掌握它让你的视频生产流程完全自动化!
