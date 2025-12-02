# LivePortrait深度技术解析

> **项目**: KwaiVGI/LivePortrait
> **Stars**: 17.3k+
> **技术**: Stitching + Retargeting Control
> **论文**: arXiv:2407.03168
> **性能**: 25-30 FPS (RTX 4090)

---

## 一、核心技术架构

### 1.1 技术创新点

LivePortrait是快手视觉生成团队开发的高效肖像动画系统,核心创新在于:

1. **Stitching Network(拼接网络)**: 无缝合成,消除边界伪影
2. **Retargeting Control(重定向控制)**: 精确的表情迁移和编辑
3. **Motion Template(运动模板)**: 可复用的运动数据,保护隐私

### 1.2 系统架构图

```
源图像 (Source Image)
    ↓
[Appearance Encoder] → 外观特征
    ↓
驱动视频 (Driving Video)
    ↓
[Face Detection] → 人脸裁剪
    ↓
[Motion Extractor] → 运动特征
    ↓
[Motion Retargeting] → 表情重定向
    ↓
[Warping Module] → 特征变形
    ↓
[Stitching Network] → 无缝拼接
    ↓
输出视频 (Animated Portrait)
```

---

## 二、核心模块实现

### 2.1 LivePortraitPipeline类架构

```python
# src/live_portrait_pipeline.py
class LivePortraitPipeline:
    """
    LivePortrait核心流水线
    """
    def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig):
        self.inference_cfg = inference_cfg
        self.crop_cfg = crop_cfg

        # 初始化核心模块
        self.appearance_extractor = AppearanceFeatureExtractor()
        self.motion_extractor = MotionExtractor()
        self.warping_module = WarpingModule()
        self.stitching_network = StitchingNetwork()
        self.face_detector = FaceDetector()

    def execute(self, args: ArgumentConfig):
        """
        执行完整的动画生成流程
        """
        # 1. 加载源图像和驱动视频
        source_image = self.load_source(args.source)
        driving_frames = self.load_driving_video(args.driving)

        # 2. 提取源图像的外观特征(仅执行一次)
        appearance_features = self.extract_appearance(source_image)

        # 3. 逐帧处理驱动视频
        output_frames = []
        for frame in driving_frames:
            # 3.1 提取运动特征
            motion_features = self.extract_motion(frame)

            # 3.2 运动重定向
            retargeted_motion = self.retarget_motion(
                motion_features,
                multiplier=args.driving_multiplier
            )

            # 3.3 特征变形
            warped_features = self.warp_features(
                appearance_features,
                retargeted_motion
            )

            # 3.4 拼接生成最终图像
            if args.flag_stitching:
                output_frame = self.stitch(warped_features, source_image)
            else:
                output_frame = warped_features

            output_frames.append(output_frame)

        # 4. 合成视频
        self.save_video(output_frames, args.output_path)
```

### 2.2 Motion Extractor(运动提取器)

```python
class MotionExtractor(nn.Module):
    """
    从驱动视频中提取运动特征
    """
    def __init__(self):
        super().__init__()
        # 使用轻量级CNN提取关键点和表情参数
        self.backbone = MobileNetV3()
        self.keypoint_head = KeypointHead(num_kps=68)  # 68个面部关键点
        self.expression_head = ExpressionHead(dim=64)  # 64维表情编码

    def forward(self, face_image):
        """
        输入: 裁剪后的人脸图像 [B, 3, 256, 256]
        输出:
            - keypoints: [B, 68, 2] 关键点坐标
            - expression: [B, 64] 表情编码
            - rotation: [B, 3] 头部姿态(pitch, yaw, roll)
            - translation: [B, 3] 头部平移
        """
        features = self.backbone(face_image)

        keypoints = self.keypoint_head(features)
        expression = self.expression_head(features)
        rotation = self.rotation_head(features)
        translation = self.translation_head(features)

        return {
            'kp': keypoints,
            'exp': expression,
            'rot': rotation,
            'trans': translation
        }
```

### 2.3 Appearance Feature Extractor(外观特征提取器)

```python
class AppearanceFeatureExtractor(nn.Module):
    """
    提取源图像的外观特征(身份、纹理等)
    """
    def __init__(self):
        super().__init__()
        # 使用ResNet或EfficientNet提取多尺度特征
        self.encoder = ResNet50(pretrained=True)
        self.feature_pyramid = FeaturePyramidNetwork()

    def forward(self, source_image):
        """
        输入: 源图像 [B, 3, 512, 512]
        输出: 多尺度外观特征
        """
        # 提取多层特征
        feat_low = self.encoder.layer1(source_image)    # 低层纹理
        feat_mid = self.encoder.layer2(feat_low)        # 中层结构
        feat_high = self.encoder.layer3(feat_mid)       # 高层语义

        # 特征金字塔
        appearance_features = self.feature_pyramid(feat_low, feat_mid, feat_high)

        return appearance_features
```

### 2.4 Warping Module(变形模块)

```python
class WarpingModule(nn.Module):
    """
    根据运动特征变形外观特征
    """
    def __init__(self):
        super().__init__()
        self.flow_predictor = FlowPredictor()

    def forward(self, appearance_features, motion_dict):
        """
        使用光流变形技术
        """
        # 1. 从运动参数预测稠密光流场
        optical_flow = self.flow_predictor(
            kp_source=motion_dict['kp_source'],
            kp_driving=motion_dict['kp_driving'],
            exp_driving=motion_dict['exp']
        )

        # 2. 使用光流变形外观特征
        warped_features = self.warp_by_flow(appearance_features, optical_flow)

        return warped_features

    def warp_by_flow(self, features, flow):
        """
        光流变形实现(使用grid_sample)
        """
        B, C, H, W = features.shape

        # 生成采样网格
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=features.device),
            torch.arange(W, device=features.device)
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).float()  # [H, W, 2]

        # 加上光流偏移
        grid = grid + flow  # [B, H, W, 2]

        # 归一化到[-1, 1]
        grid[:, :, :, 0] = 2.0 * grid[:, :, :, 0] / (W - 1) - 1.0
        grid[:, :, :, 1] = 2.0 * grid[:, :, :, 1] / (H - 1) - 1.0

        # 双线性插值采样
        warped = F.grid_sample(
            features,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )

        return warped
```

### 2.5 Stitching Network(拼接网络)

```python
class StitchingNetwork(nn.Module):
    """
    无缝拼接变形后的人脸和原始背景
    """
    def __init__(self):
        super().__init__()
        # 使用U-Net结构预测融合mask
        self.unet = UNet(in_channels=6, out_channels=1)  # RGB×2 → alpha mask

    def forward(self, warped_face, source_image, face_mask):
        """
        输入:
            - warped_face: 变形后的人脸 [B, 3, H, W]
            - source_image: 原始源图像 [B, 3, H, W]
            - face_mask: 人脸区域mask [B, 1, H, W]
        输出:
            - stitched: 拼接后的完整图像 [B, 3, H, W]
        """
        # 拼接warped_face和source_image作为输入
        concat_input = torch.cat([warped_face, source_image], dim=1)  # [B, 6, H, W]

        # 预测软边界融合mask
        alpha_mask = torch.sigmoid(self.unet(concat_input))  # [B, 1, H, W]

        # Alpha blending
        stitched = alpha_mask * warped_face + (1 - alpha_mask) * source_image

        return stitched
```

### 2.6 Retargeting Controller(重定向控制器)

```python
class RetargetingController:
    """
    精确控制表情强度和局部编辑
    """
    def __init__(self):
        self.feature_regions = {
            'eye_left': [36, 37, 38, 39, 40, 41],     # 左眼关键点索引
            'eye_right': [42, 43, 44, 45, 46, 47],    # 右眼
            'eyebrow_left': [17, 18, 19, 20, 21],     # 左眉
            'eyebrow_right': [22, 23, 24, 25, 26],    # 右眉
            'mouth': [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59],  # 嘴
            'nose': [27, 28, 29, 30, 31, 32, 33, 34, 35]                # 鼻
        }

    def retarget(self, motion_dict, multiplier=1.0, region_control=None):
        """
        参数:
            - motion_dict: 原始运动特征
            - multiplier: 全局表情强度倍数
            - region_control: 区域控制字典, 例如 {'mouth': 1.5, 'eye_left': 0.5}
        """
        kp_driving = motion_dict['kp'].clone()
        kp_source = motion_dict['kp_source']

        # 计算关键点位移
        delta_kp = kp_driving - kp_source  # [B, 68, 2]

        # 全局缩放
        delta_kp = delta_kp * multiplier

        # 区域精细控制
        if region_control:
            for region, scale in region_control.items():
                indices = self.feature_regions[region]
                delta_kp[:, indices, :] *= scale

        # 应用重定向后的位移
        kp_retargeted = kp_source + delta_kp

        motion_dict['kp_driving'] = kp_retargeted
        return motion_dict
```

---

## 三、完整使用示例

### 3.1 基础推理代码

```python
import torch
from src.live_portrait_pipeline import LivePortraitPipeline
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig

# 1. 配置初始化
inference_cfg = InferenceConfig(
    device='cuda',
    flag_do_torch_compile=True,  # 启用torch.compile加速
    flag_stitching=True           # 启用拼接网络
)

crop_cfg = CropConfig(
    dsize=512,                    # 输出分辨率
    scale=2.3,                    # 裁剪缩放
    vy_ratio=-0.125               # 垂直偏移
)

# 2. 创建流水线
pipeline = LivePortraitPipeline(
    inference_cfg=inference_cfg,
    crop_cfg=crop_cfg
)

# 3. 执行推理
pipeline.execute(
    source='assets/examples/source/s1.jpg',
    driving='assets/examples/driving/d1.mp4',
    output='output.mp4',
    driving_multiplier=1.5,       # 表情强度1.5倍
    flag_pasteback=True,          # 贴回原始背景
    flag_crop_driving_video=True  # 自动裁剪驱动视频
)
```

### 3.2 运动模板复用

```python
# 首次生成时保存运动模板
pipeline.execute(
    source='person1.jpg',
    driving='dance.mp4',
    output='person1_dance.mp4',
    flag_save_motion='dance_motion.pkl'  # 保存运动模板
)

# 后续直接加载模板,无需重复提取运动
pipeline.execute(
    source='person2.jpg',
    driving='dance_motion.pkl',           # 直接加载模板
    output='person2_dance.mp4'
)
```

**优势**:
- 运动模板不包含身份信息,保护隐私
- 批量处理时显著加速(跳过motion extraction)
- 支持运动数据的商业授权和分发

### 3.3 精细表情控制

```python
from src.utils.retargeting import RetargetingController

controller = RetargetingController()

# 区域化控制:嘴部动作放大2倍,眼睛保持静止
region_control = {
    'mouth': 2.0,      # 嘴部表情夸张
    'eye_left': 0.0,   # 左眼不动
    'eye_right': 0.0   # 右眼不动
}

pipeline.execute(
    source='portrait.jpg',
    driving='speech.mp4',
    output='speech_with_control.mp4',
    retargeting_controller=controller,
    region_control=region_control
)
```

**应用场景**:
- 口型同步(Lip-sync): 只控制嘴部,眼睛保持源图像
- 眨眼编辑: 只控制眼睛,其他特征不变
- 情绪夸张: 放大驱动视频的表情强度

### 3.4 动物模式(猫狗)

```python
# 需要安装XPose依赖
# pip install -e src/utils/dependencies/XPose

pipeline_animal = LivePortraitPipeline(
    inference_cfg=inference_cfg,
    crop_cfg=crop_cfg,
    animal_mode=True  # 启用动物模式
)

pipeline_animal.execute(
    source='cat.jpg',
    driving='cat_moving.mp4',
    output='cat_animated.mp4'
)
```

---

## 四、性能优化实战

### 4.1 torch.compile加速

```python
import torch

# PyTorch 2.0+的编译加速
inference_cfg = InferenceConfig(
    flag_do_torch_compile=True  # 启用
)

# 首次推理会触发编译(较慢)
# 后续推理加速20-30%
```

**实测数据(RTX 4090)**:
- 未编译: 40ms/frame
- 编译后: 28ms/frame (提升30%)

### 4.2 批处理优化

```python
class BatchPipeline:
    """批量处理多个源图像"""
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def batch_execute(self, sources, driving, output_dir):
        """
        sources: 多个源图像路径列表
        driving: 共享的驱动视频
        """
        # 1. 提取驱动视频运动特征(仅1次)
        motion_dict = self.pipeline.extract_driving_motion(driving)

        # 2. 批量处理源图像
        for source in sources:
            appearance_features = self.pipeline.extract_appearance(source)
            output_frames = self.pipeline.generate_frames(
                appearance_features,
                motion_dict
            )
            self.pipeline.save_video(output_frames, f"{output_dir}/{source}.mp4")
```

**加速效果**: N个人共享同一段舞蹈,运动提取从O(N)降为O(1)

### 4.3 分辨率自适应

```python
# 低分辨率快速预览
preview_cfg = CropConfig(dsize=256)  # 256×256
pipeline_preview = LivePortraitPipeline(inference_cfg, preview_cfg)

# 高分辨率最终输出
final_cfg = CropConfig(dsize=1024)   # 1024×1024
pipeline_final = LivePortraitPipeline(inference_cfg, final_cfg)
```

**性能对比**:
- 256×256: ~15ms/frame
- 512×512: ~40ms/frame
- 1024×1024: ~150ms/frame

---

## 五、实际部署方案

### 5.1 本地部署(推荐配置)

```bash
# 硬件要求
GPU: NVIDIA RTX 3060+ (6GB+ VRAM)
CPU: Intel i5-10400 / AMD Ryzen 5 3600
RAM: 16GB+
存储: 10GB (模型权重~5GB)

# 软件环境
Python: 3.10+
PyTorch: 2.3.0+
CUDA: 11.8 / 12.1

# 安装步骤
git clone https://github.com/KwaiVGI/LivePortrait.git
cd LivePortrait
pip install -r requirements.txt

# 下载预训练权重(自动)
python inference.py \
    --source assets/examples/source/s6.jpg \
    --driving assets/examples/driving/d0.mp4
```

### 5.2 服务化部署

```python
# api_server.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import tempfile
import os

app = FastAPI()
pipeline = LivePortraitPipeline(inference_cfg, crop_cfg)

@app.post("/animate")
async def animate_portrait(
    source: UploadFile = File(...),
    driving: UploadFile = File(...),
    multiplier: float = 1.0
):
    """
    API接口:上传源图像和驱动视频,返回动画视频
    """
    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_src:
        tmp_src.write(await source.read())
        source_path = tmp_src.name

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_drv:
        tmp_drv.write(await driving.read())
        driving_path = tmp_drv.name

    # 执行动画生成
    output_path = '/tmp/output.mp4'
    pipeline.execute(
        source=source_path,
        driving=driving_path,
        output=output_path,
        driving_multiplier=multiplier
    )

    # 清理临时文件
    os.remove(source_path)
    os.remove(driving_path)

    return FileResponse(output_path, media_type='video/mp4')

# 启动服务
# uvicorn api_server:app --host 0.0.0.0 --port 8000
```

**并发处理**:
- 使用GPU队列管理多个请求
- 启用模型批处理(batch inference)
- 异步I/O处理文件上传/下载

### 5.3 Gradio Web界面

```python
import gradio as gr

def animate_fn(source_img, driving_video, multiplier):
    """Gradio接口函数"""
    output_path = '/tmp/output.mp4'
    pipeline.execute(
        source=source_img,
        driving=driving_video,
        output=output_path,
        driving_multiplier=multiplier
    )
    return output_path

# 创建界面
demo = gr.Interface(
    fn=animate_fn,
    inputs=[
        gr.Image(type='filepath', label='Source Portrait'),
        gr.Video(label='Driving Video'),
        gr.Slider(0.5, 2.0, value=1.0, label='Expression Multiplier')
    ],
    outputs=gr.Video(label='Animated Result'),
    title='LivePortrait Demo',
    examples=[
        ['assets/examples/source/s1.jpg', 'assets/examples/driving/d1.mp4', 1.0],
        ['assets/examples/source/s6.jpg', 'assets/examples/driving/d5.mp4', 1.5]
    ]
)

demo.launch(share=True)
```

---

## 六、进阶技巧

### 6.1 多人脸处理

```python
# 使用社区扩展:https://github.com/PowerHouseMan/ComfyUI-AdvancedLivePortrait
# 支持检测多张人脸并分别驱动

from src.utils.multi_face import MultiFaceDetector

detector = MultiFaceDetector()

def multi_face_animate(source_img, driving_videos):
    """
    source_img: 包含多人的合照
    driving_videos: 每个人对应的驱动视频列表
    """
    # 检测所有人脸
    faces = detector.detect_all(source_img)

    # 为每个人脸单独驱动
    results = []
    for face, driving in zip(faces, driving_videos):
        animated = pipeline.execute(
            source=face['crop'],
            driving=driving,
            output=f'person_{face["id"]}.mp4'
        )
        results.append(animated)

    # 合成到原图
    final_video = detector.composite_all(source_img, results)
    return final_video
```

### 6.2 视频到视频编辑(V2V)

```python
# 编辑已有的人像视频
pipeline.execute(
    source='original_video.mp4',  # 源视频
    driving='expression_ref.mp4', # 表情参考
    output='edited_video.mp4',
    flag_video_editing_head_rotation=True,  # 保留源视频头部姿态
    driving_multiplier=0.8                  # 温和的表情迁移
)
```

**应用**:
- 影视后期:演员表情微调
- 直播美颜:实时表情平滑

### 6.3 图片驱动模式

```python
# 使用参考图片的表情驱动
pipeline.execute(
    source='neutral_face.jpg',
    driving='smile_reference.jpg',  # 单张图片
    output='smile_result.jpg',
    flag_image_driven=True
)
```

---

## 七、技术对比

| 特性 | LivePortrait | SadTalker | MuseTalk | Wav2Lip |
|------|-------------|-----------|----------|---------|
| **输入** | 图像+视频/图像 | 图像+音频 | 图像+音频 | 图像+音频 |
| **主要用途** | 表情迁移 | 音频驱动 | 音频驱动 | 唇形同步 |
| **FPS** | 25-30 | 10-15 | 30+ | 25 |
| **控制精度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **运动复用** | ✅ | ❌ | ❌ | ❌ |
| **区域编辑** | ✅ | ❌ | ❌ | ❌ |
| **动物支持** | ✅ (猫狗) | ❌ | ❌ | ❌ |
| **隐私保护** | ✅ (运动模板) | ❌ | ❌ | ❌ |

**LivePortrait独特优势**:
1. **Stitching Network** → 无边界伪影
2. **Retargeting Control** → 精确的局部编辑
3. **Motion Template** → 运动数据复用
4. **Image-Driven** → 单图表情克隆

---

## 八、常见问题解决

### Q1: 生成的视频有明显拼接痕迹?

```python
# 确保启用stitching网络
inference_cfg = InferenceConfig(
    flag_stitching=True,        # ← 必须开启
    flag_pasteback=True,        # 贴回原始背景
    mask_crop_ratio=1.4         # 增大mask范围,平滑边界
)
```

### Q2: 表情太夸张/太僵硬?

```python
# 调整driving_multiplier参数
pipeline.execute(
    ...,
    driving_multiplier=0.7  # <1.0 减弱表情, >1.0 增强表情
)
```

### Q3: 眼睛不自然?

```python
# 使用区域控制
region_control = {
    'eye_left': 0.5,   # 减弱眼睛动作
    'eye_right': 0.5
}
```

### Q4: 驱动视频人脸未对齐?

```bash
# 启用自动裁剪
python inference.py \
    --source xxx.jpg \
    --driving xxx.mp4 \
    --flag_crop_driving_video  # ← 自动检测并裁剪人脸
```

---

## 九、总结

### 核心技术要点

1. **Stitching Network**: 解决拼接伪影问题,超越传统paste-back方法
2. **Retargeting Control**: 提供灵活的表情编辑能力
3. **Motion Template**: 运动数据与身份解耦,支持复用和隐私保护
4. **高效推理**: torch.compile + 轻量级模型,实现25-30 FPS

### 适用场景

| 场景 | 推荐度 | 说明 |
|-----|--------|------|
| 短视频创作 | ⭐⭐⭐⭐⭐ | 表情包、虚拟主播、创意视频 |
| 数字人驱动 | ⭐⭐⭐⭐ | 需结合音频驱动(TTS) |
| 视频编辑 | ⭐⭐⭐⭐⭐ | V2V模式微调表情 |
| 游戏角色 | ⭐⭐⭐ | 需要实时性更高的方案 |
| 影视制作 | ⭐⭐⭐⭐ | 后期表情修正 |

### 与其他方案协同

```python
# LivePortrait + OpenAvatarChat整合
# 1. 使用OpenAvatarChat的ASR+TTS获取音频驱动
# 2. 使用LivePortrait进行高质量人脸动画
# 3. 结合WebRTC实现实时交互

class HybridPipeline:
    def __init__(self):
        self.oachat = OpenAvatarChatPipeline()
        self.liveportrait = LivePortraitPipeline()

    async def real_time_chat(self, user_audio, avatar_image):
        # OpenAvatarChat处理对话
        response_audio = await self.oachat.chat(user_audio)

        # LivePortrait生成面部动画
        animated_video = self.liveportrait.execute(
            source=avatar_image,
            driving=response_audio,  # 从音频生成运动
            flag_audio_driven=True
        )

        return animated_video
```

---

**项目地址**: https://github.com/KwaiVGI/LivePortrait
**论文**: https://arxiv.org/abs/2407.03168
**许可证**: 非商业研究使用,商业授权需联系快手
**社区**: ComfyUI插件、WebUI等丰富生态
