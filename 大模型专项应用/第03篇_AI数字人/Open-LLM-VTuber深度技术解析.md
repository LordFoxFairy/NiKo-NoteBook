# Open-LLM-VTuber深度技术解析

> **项目**: t41372/Open-LLM-VTuber
> **Stars**: 5.1k+
> **定位**: 全功能AI虚拟主播/桌面伙伴
> **核心**: ASR + LLM + TTS + Live2D 全模块化
> **许可**: MIT (代码) + Live2D 素材单独授权
> **特色**: 完全离线可用,支持视觉感知

---

## 一、系统架构概览

### 1.1 四大核心模块

Open-LLM-VTuber是一个**高度模块化**的AI虚拟主播系统,采用插件式架构:

```
用户语音输入
    ↓
┌─────────────────────┐
│  ASR 模块            │ ← sherpa-onnx / FunASR / Faster-Whisper / Groq Whisper
│  (语音识别)          │
└─────────────────────┘
    ↓
┌─────────────────────┐
│  LLM 模块            │ ← Ollama / OpenAI API / Gemini / Claude / DeepSeek
│  (对话推理)          │
└─────────────────────┘
    ↓
┌─────────────────────┐
│  TTS 模块            │ ← GPTSoVITS / CosyVoice / MeloTTS / Edge TTS
│  (语音合成)          │
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Live2D 渲染         │ ← 自定义模型 + 表情映射
│  (虚拟形象)          │
└─────────────────────┘
    ↓
音视频输出(WebRTC / Desktop)
```

### 1.2 技术栈

**后端**:
- Python 3.10+
- FastAPI (Web服务)
- WebSocket (实时通信)
- FFmpeg (音频处理)
- `uv` (依赖管理,替代pip)

**前端**:
- Web UI (单独子模块仓库)
- Electron (桌面客户端)
- Live2D Cubism SDK

**部署**:
- Docker镜像: `t41372/open-llm-vtuber`
- 本地部署: `uv run run_server.py`

---

## 二、核心模块实现

### 2.1 模块化架构设计

```python
# src/open_llm_vtuber/agent/agent.py
from abc import ABC, abstractmethod

class Agent(ABC):
    """
    Agent基类:定义对话Agent的统一接口
    """
    @abstractmethod
    async def chat(self, user_input: str, context: dict) -> str:
        """
        处理用户输入,返回回复文本
        """
        pass

    @abstractmethod
    async def interrupt(self):
        """
        中断当前输出(用户打断时调用)
        """
        pass

# 具体实现:基于LangChain的Agent
class LangChainAgent(Agent):
    def __init__(self, llm_provider: str, model_name: str):
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage

        self.llm = ChatOpenAI(
            base_url=config['llm']['base_url'],
            model=model_name,
            temperature=0.7
        )

        # 加载角色设定
        self.system_prompt = self.load_prompt('characters/default.txt')
        self.history = []

    async def chat(self, user_input: str, context: dict) -> str:
        # 构建消息历史
        messages = [SystemMessage(content=self.system_prompt)]
        messages.extend(self.history)
        messages.append(HumanMessage(content=user_input))

        # 调用LLM
        response = await self.llm.ainvoke(messages)

        # 更新历史
        self.history.append(HumanMessage(content=user_input))
        self.history.append(response)

        return response.content

    def load_prompt(self, path):
        """加载角色设定文件"""
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
```

### 2.2 ASR模块(语音识别)

```python
# src/open_llm_vtuber/asr/asr_interface.py
from abc import ABC, abstractmethod

class ASRInterface(ABC):
    """ASR统一接口"""
    @abstractmethod
    async def transcribe(self, audio_data: bytes) -> str:
        pass

# Faster-Whisper实现(推荐)
from faster_whisper import WhisperModel

class FasterWhisperASR(ASRInterface):
    def __init__(self, model_size='base', device='cuda'):
        """
        model_size: tiny/base/small/medium/large-v3
        device: cuda/cpu
        """
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type='float16' if device == 'cuda' else 'int8'
        )

        self.vad_filter = True  # 启用VAD过滤

    async def transcribe(self, audio_data: bytes) -> str:
        """
        输入: WAV/MP3音频字节流
        输出: 识别文本
        """
        import io
        import soundfile as sf

        # 转换为numpy数组
        audio_array, sr = sf.read(io.BytesIO(audio_data))

        # 推理
        segments, info = self.model.transcribe(
            audio_array,
            language='zh',          # 可自动检测
            vad_filter=self.vad_filter,
            beam_size=5
        )

        # 拼接所有片段
        text = ' '.join([seg.text for seg in segments])
        return text.strip()

# sherpa-onnx离线实现(完全本地)
import sherpa_onnx

class SherpaOnnxASR(ASRInterface):
    def __init__(self, model_path):
        """
        使用ONNX格式的Whisper/Zipformer模型
        完全离线,无需GPU
        """
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=f'{model_path}/tokens.txt',
            encoder=f'{model_path}/encoder.onnx',
            decoder=f'{model_path}/decoder.onnx',
            joiner=f'{model_path}/joiner.onnx',
            num_threads=4,
            sample_rate=16000,
            feature_dim=80
        )

        self.stream = self.recognizer.create_stream()

    async def transcribe(self, audio_data: bytes) -> str:
        """
        流式识别
        """
        import numpy as np

        # 转换为float32数组
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        # 分块送入识别器
        chunk_size = 1600  # 100ms @ 16kHz
        for i in range(0, len(audio_array), chunk_size):
            chunk = audio_array[i:i + chunk_size]
            self.stream.accept_waveform(16000, chunk)

        # 获取结果
        self.recognizer.decode_stream(self.stream)
        text = self.stream.result.text

        # 重置stream供下次使用
        self.stream = self.recognizer.create_stream()

        return text
```

### 2.3 TTS模块(语音合成)

```python
# src/open_llm_vtuber/tts/tts_interface.py
from abc import ABC, abstractmethod

class TTSInterface(ABC):
    @abstractmethod
    async def synthesize(self, text: str, **kwargs) -> bytes:
        """返回音频字节流(WAV格式)"""
        pass

# GPTSoVITS实现(支持零样本克隆)
class GPTSoVITSTTS(TTSInterface):
    def __init__(self, api_url='http://localhost:9880'):
        """
        需要先启动GPTSoVITS服务
        """
        self.api_url = api_url

    async def synthesize(self, text: str, refer_wav_path=None, prompt_text=None) -> bytes:
        """
        参数:
            text: 要合成的文本
            refer_wav_path: 参考音频(3-10秒,用于声音克隆)
            prompt_text: 参考音频的文本
        """
        import aiohttp

        data = {
            'text': text,
            'text_lang': 'zh',
            'ref_audio_path': refer_wav_path or 'voices/default.wav',
            'prompt_text': prompt_text or '参考文本',
            'prompt_lang': 'zh',
            'top_k': 15,
            'top_p': 1.0,
            'temperature': 1.0,
            'speed': 1.0
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(f'{self.api_url}/tts', json=data) as resp:
                audio_bytes = await resp.read()

        return audio_bytes

# Edge TTS实现(免费云端,质量高)
import edge_tts

class EdgeTTSTTS(TTSInterface):
    def __init__(self, voice='zh-CN-XiaoxiaoNeural', rate='+0%', pitch='+0Hz'):
        """
        voice: 微软语音选项
            - zh-CN-XiaoxiaoNeural (晓晓,女)
            - zh-CN-YunxiNeural (云希,男)
            - zh-CN-YunyangNeural (云扬,男)
            - ja-JP-NanamiNeural (七海,日语女)
        """
        self.voice = voice
        self.rate = rate
        self.pitch = pitch

    async def synthesize(self, text: str) -> bytes:
        """
        异步合成
        """
        communicate = edge_tts.Communicate(
            text,
            voice=self.voice,
            rate=self.rate,
            pitch=self.pitch
        )

        # 保存到内存
        import io
        audio_buffer = io.BytesIO()

        async for chunk in communicate.stream():
            if chunk['type'] == 'audio':
                audio_buffer.write(chunk['data'])

        audio_buffer.seek(0)
        return audio_buffer.read()

# MeloTTS本地实现(高质量开源)
from melo.api import TTS as MeloTTSAPI

class MeloTTSTTS(TTSInterface):
    def __init__(self, language='ZH', device='cuda'):
        """
        language: ZH/EN/JP/KR/FR/ES
        """
        self.tts = MeloTTSAPI(language=language, device=device)
        self.speaker_ids = self.tts.hps.data.spk2id  # 说话人ID映射

    async def synthesize(self, text: str, speaker='ZH') -> bytes:
        """
        speaker: ZH(中文女)/EN-Default(英文男)/JP(日语女)
        """
        import io

        # 生成音频
        audio_array = self.tts.tts_to_file(
            text,
            speaker_id=self.speaker_ids.get(speaker, 0),
            speed=1.0,
            output_path=None,  # 返回numpy数组
            format='wav'
        )

        # 转换为字节流
        import soundfile as sf
        buffer = io.BytesIO()
        sf.write(buffer, audio_array, 22050, format='WAV')
        buffer.seek(0)

        return buffer.read()
```

### 2.4 Live2D渲染模块

```python
# src/open_llm_vtuber/live2d/live2d_manager.py
class Live2DManager:
    """
    管理Live2D模型的加载、表情控制、动作播放
    """
    def __init__(self, model_path):
        """
        model_path: Live2D模型目录(包含.model3.json)
        """
        self.model_path = model_path
        self.current_expression = 'normal'

        # 表情映射配置
        self.expression_mapping = {
            'happy': ['f01', 'f02'],     # 开心表情序号
            'sad': ['f03'],
            'angry': ['f04'],
            'surprised': ['f05'],
            'normal': ['f00']
        }

    def set_expression(self, emotion: str):
        """
        根据情绪设置Live2D表情
        """
        if emotion in self.expression_mapping:
            expression_ids = self.expression_mapping[emotion]
            # 随机选择一个表情
            import random
            selected = random.choice(expression_ids)

            # 发送WebSocket消息到前端
            self.send_to_frontend({
                'type': 'set_expression',
                'expression': selected
            })

            self.current_expression = emotion

    def play_motion(self, motion_name: str):
        """
        播放动作(挥手、点头等)
        """
        self.send_to_frontend({
            'type': 'play_motion',
            'motion_group': 'Idle',  # 或TapBody/Shake等
            'motion_name': motion_name
        })

    def set_lip_sync(self, audio_data: bytes):
        """
        口型同步:从音频提取音量包络
        """
        import numpy as np
        import soundfile as sf
        import io

        # 加载音频
        audio_array, sr = sf.read(io.BytesIO(audio_data))

        # 计算RMS音量包络
        frame_length = int(sr * 0.02)  # 20ms窗口
        hop_length = frame_length // 2

        rms = []
        for i in range(0, len(audio_array) - frame_length, hop_length):
            frame = audio_array[i:i + frame_length]
            rms_value = np.sqrt(np.mean(frame ** 2))
            rms.append(rms_value)

        # 归一化到0-1
        rms = np.array(rms)
        rms = (rms - rms.min()) / (rms.max() - rms.min() + 1e-6)

        # 发送到前端驱动MouthOpenY参数
        self.send_to_frontend({
            'type': 'lip_sync',
            'volumes': rms.tolist(),
            'duration': len(audio_array) / sr
        })

    def send_to_frontend(self, message: dict):
        """
        通过WebSocket发送消息到前端Live2D渲染器
        """
        # 实际实现会用WebSocket manager
        from .websocket_manager import ws_manager
        import json

        ws_manager.broadcast(json.dumps(message))

# 前端Live2D渲染(JavaScript)
"""
// frontend/src/live2d.js
import { Live2DModel } from 'pixi-live2d-display';

class Live2DRenderer {
    constructor(canvasId, modelPath) {
        this.app = new PIXI.Application({
            view: document.getElementById(canvasId),
            transparent: true,
            backgroundAlpha: 0
        });

        this.loadModel(modelPath);
        this.setupWebSocket();
    }

    async loadModel(modelPath) {
        this.model = await Live2DModel.from(modelPath);
        this.app.stage.addChild(this.model);

        // 调整大小和位置
        this.model.scale.set(0.5);
        this.model.position.set(this.app.screen.width / 2, this.app.screen.height);

        // 启用交互
        this.model.on('hit', (hitAreas) => {
            if (hitAreas.includes('Body')) {
                this.model.motion('TapBody');
            }
        });
    }

    setupWebSocket() {
        this.ws = new WebSocket('ws://localhost:8000/ws/live2d');

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);

            switch (data.type) {
                case 'set_expression':
                    this.model.expression(data.expression);
                    break;

                case 'play_motion':
                    this.model.motion(data.motion_group, data.motion_name);
                    break;

                case 'lip_sync':
                    this.playLipSync(data.volumes, data.duration);
                    break;
            }
        };
    }

    playLipSync(volumes, duration) {
        // 根据音量数组驱动MouthOpenY参数
        const frameDuration = duration / volumes.length * 1000;  // ms

        volumes.forEach((volume, index) => {
            setTimeout(() => {
                this.model.internalModel.coreModel.setParameterValueById(
                    'ParamMouthOpenY',
                    volume
                );
            }, index * frameDuration);
        });
    }
}
"""
```

### 2.5 视觉感知模块(多模态扩展)

```python
# src/open_llm_vtuber/vision/vision_module.py
class VisionModule:
    """
    支持摄像头/截图/录屏输入,让VTuber"看见"
    """
    def __init__(self, vlm_provider='gemini'):
        """
        vlm_provider: 视觉语言模型提供商
            - gemini-1.5-flash (免费,支持视频)
            - gpt-4o (OpenAI)
            - claude-3.5-sonnet (Anthropic)
        """
        self.vlm = self.init_vlm(vlm_provider)

    def init_vlm(self, provider):
        if provider == 'gemini':
            import google.generativeai as genai
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            return genai.GenerativeModel('gemini-1.5-flash')

        elif provider == 'gpt-4o':
            from openai import OpenAI
            return OpenAI().chat.completions

        # ... 其他提供商

    async def capture_screen(self) -> bytes:
        """截取当前屏幕"""
        import pyautogui
        import io

        screenshot = pyautogui.screenshot()
        buffer = io.BytesIO()
        screenshot.save(buffer, format='PNG')
        buffer.seek(0)

        return buffer.read()

    async def capture_camera(self) -> bytes:
        """从摄像头捕获一帧"""
        import cv2

        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if ret:
            import io
            from PIL import Image
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            return buffer.read()

        return None

    async def analyze_image(self, image_data: bytes, prompt: str) -> str:
        """
        使用VLM分析图像
        """
        import base64
        from PIL import Image
        import io

        # 加载图像
        img = Image.open(io.BytesIO(image_data))

        if isinstance(self.vlm, genai.GenerativeModel):
            # Gemini
            response = self.vlm.generate_content([prompt, img])
            return response.text

        elif hasattr(self.vlm, 'create'):
            # OpenAI GPT-4o
            base64_img = base64.b64encode(image_data).decode()
            response = self.vlm.create(
                model='gpt-4o',
                messages=[{
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        {'type': 'image_url', 'image_url': {
                            'url': f'data:image/png;base64,{base64_img}'
                        }}
                    ]
                }]
            )
            return response.choices[0].message.content

# 集成到对话Agent
class VisionAgent(Agent):
    def __init__(self):
        super().__init__()
        self.vision = VisionModule(vlm_provider='gemini')

    async def chat(self, user_input: str, context: dict) -> str:
        # 检测是否包含视觉相关指令
        if '看看我的屏幕' in user_input or '截图' in user_input:
            # 捕获屏幕
            screenshot = await self.vision.capture_screen()

            # VLM分析
            analysis = await self.vision.analyze_image(
                screenshot,
                prompt=f'用户说:{user_input}\n请描述你看到的内容并回应用户。'
            )

            return analysis

        elif '看看我' in user_input or '摄像头' in user_input:
            # 捕获摄像头
            camera_img = await self.vision.capture_camera()
            analysis = await self.vision.analyze_image(
                camera_img,
                prompt=f'用户说:{user_input}\n请观察用户并做出回应。'
            )

            return analysis

        else:
            # 普通文本对话
            return await super().chat(user_input, context)
```

---

## 三、配置系统

### 3.1 conf.yaml配置文件

```yaml
# conf.yaml
# ASR配置
asr:
  provider: 'faster_whisper'  # sherpa_onnx / funasr / groq_whisper / azure
  model_size: 'base'           # tiny/base/small/medium/large-v3
  device: 'cuda'               # cuda/cpu
  language: 'zh'               # 语言代码(auto自动检测)
  vad_filter: true             # 启用VAD过滤静音

# LLM配置
llm:
  provider: 'ollama'           # openai / gemini / claude / deepseek
  model: 'qwen2.5:7b'
  base_url: 'http://localhost:11434/v1'
  temperature: 0.7
  max_tokens: 2048
  stream: true                 # 流式输出

# TTS配置
tts:
  provider: 'edge_tts'         # gpt_sovits / melo_tts / coqui / fish_audio
  voice: 'zh-CN-XiaoxiaoNeural'
  rate: '+0%'
  pitch: '+0Hz'

# Live2D配置
live2d:
  model_path: 'live2d-models/shizuku'  # 模型目录
  scale: 0.5
  position_x: 0.5              # 屏幕中心
  position_y: 1.0              # 底部对齐

# 角色设定
character:
  name: 'Shizuku'
  prompt_file: 'prompts/default.txt'
  language: 'zh'               # 对话语言
  tts_language: 'zh'           # TTS语言(可与对话不同)

# 功能开关
features:
  echo_cancellation: true      # 回声消除(无耳机打断)
  vision: false                # 视觉感知模块
  translation: false           # 实时翻译
  mcp_servers: true            # MCP服务器集成

# MCP服务器(Model Context Protocol)
mcp_servers:
  - name: 'filesystem'
    command: 'npx'
    args: ['-y', '@modelcontextprotocol/server-filesystem', '/home/user']

  - name: 'brave_search'
    command: 'npx'
    args: ['-y', '@modelcontextprotocol/server-brave-search']
    env:
      BRAVE_API_KEY: 'your_api_key'

# 高级选项
advanced:
  log_level: 'INFO'
  save_chat_history: true
  history_path: 'chat_logs/'
  max_history_turns: 20
```

### 3.2 角色设定文件

```python
# prompts/default.txt
"""
你是Shizuku,一位可爱活泼的虚拟主播。

## 性格特点
- 活泼开朗,喜欢和观众互动
- 说话时会使用一些可爱的语气词,如"呢"、"哦"、"啦"
- 对新鲜事物充满好奇
- 偶尔会害羞

## 回复风格
- 使用简短自然的口语化表达
- 适当使用emoji(😊🎉等)
- 避免过长的段落,保持对话流畅

## 技能
- 可以唱歌、讲笑话、分享日常
- 了解最新的二次元文化和游戏
- 擅长倾听和安慰

## 限制
- 不讨论政治敏感话题
- 拒绝生成不当内容
- 超出知识范围时诚实承认
"""

# 加载到Agent
class CharacterAgent(Agent):
    def __init__(self, character_file='prompts/default.txt'):
        with open(character_file, 'r', encoding='utf-8') as f:
            self.character_prompt = f.read()

        # 初始化LLM
        self.llm = ChatOpenAI(...)
        self.history = []

    async def chat(self, user_input: str, context: dict) -> str:
        messages = [
            SystemMessage(content=self.character_prompt),
            *self.history,
            HumanMessage(content=user_input)
        ]

        response = await self.llm.ainvoke(messages)

        # 更新历史
        self.history.append(HumanMessage(content=user_input))
        self.history.append(response)

        # 限制历史长度
        if len(self.history) > 40:  # 20轮对话
            self.history = self.history[-40:]

        return response.content
```

---

## 四、部署方案

### 4.1 本地部署(推荐)

```bash
# 1. 克隆仓库
git clone https://github.com/t41372/Open-LLM-VTuber.git
cd Open-LLM-VTuber

# 2. 安装uv(新一代Python包管理器)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. 安装依赖
uv sync

# 4. 安装FFmpeg
# macOS
brew install ffmpeg

# Ubuntu
sudo apt install ffmpeg

# Windows
# 下载并添加到PATH

# 5. 配置conf.yaml
cp config_templates/conf.yaml conf.yaml
# 编辑conf.yaml,配置ASR/LLM/TTS等

# 6. 启动服务
uv run run_server.py

# 7. 访问Web UI
# 浏览器打开 http://localhost:8000
```

### 4.2 Docker部署

```bash
# 拉取镜像
docker pull t41372/open-llm-vtuber:latest

# 运行容器
docker run -d \
  --name vtuber \
  --gpus all \  # 使用GPU
  -p 8000:8000 \
  -v $(pwd)/conf.yaml:/app/conf.yaml \
  -v $(pwd)/live2d-models:/app/live2d-models \
  -v $(pwd)/chat_logs:/app/chat_logs \
  t41372/open-llm-vtuuber:latest

# 查看日志
docker logs -f vtuber
```

**Dockerfile示例**:
```dockerfile
FROM python:3.11-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# 安装uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# 工作目录
WORKDIR /app

# 复制项目文件
COPY . .

# 安装Python依赖
RUN uv sync

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uv", "run", "run_server.py"]
```

### 4.3 桌面宠物模式

```python
# scripts/desktop_pet.py
"""
透明背景桌面宠物,始终置顶,可拖动
"""
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtWebEngineWidgets import QWebEngineView

class DesktopPet(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('VTuber Desktop Pet')

        # 窗口透明,无边框,始终置顶
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool
        )

        # 嵌入WebView加载Live2D
        self.web = QWebEngineView()
        self.web.setUrl('http://localhost:8000')
        self.setCentralWidget(self.web)

        # 窗口大小
        self.resize(600, 800)

        # 拖动支持
        self.drag_position = QPoint()

    def mousePressEvent(self, event):
        """鼠标按下:记录位置"""
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        """鼠标移动:拖动窗口"""
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.drag_position)
            event.accept()

    def mouseDoubleClickEvent(self, event):
        """双击:触发对话"""
        # 发送WebSocket消息到后端
        import websocket
        ws = websocket.create_connection('ws://localhost:8000/ws/pet')
        ws.send('{"action": "start_listening"}')
        ws.close()

if __name__ == '__main__':
    app = QApplication([])
    pet = DesktopPet()
    pet.show()
    app.exec_()
```

启动:
```bash
uv run scripts/desktop_pet.py
```

---

## 五、高级功能

### 5.1 声音打断(无耳机模式)

```python
# src/open_llm_vtuber/audio/echo_cancellation.py
import noisereduce as nr
import numpy as np

class EchoCanceller:
    """
    回声消除:允许用户在VTuber说话时打断
    """
    def __init__(self, sample_rate=16000):
        self.sr = sample_rate
        self.reference_audio = None  # VTuber当前播放的音频

    def set_reference(self, audio_data: np.ndarray):
        """
        设置参考音频(VTuber正在播放的声音)
        """
        self.reference_audio = audio_data

    def process_microphone(self, mic_input: np.ndarray) -> np.ndarray:
        """
        从麦克风输入中移除VTuber声音
        """
        if self.reference_audio is None:
            return mic_input

        # 1. 噪声抑制
        denoised = nr.reduce_noise(y=mic_input, sr=self.sr)

        # 2. 自适应滤波(LMS算法)
        filtered = self.adaptive_filter(denoised, self.reference_audio)

        return filtered

    def adaptive_filter(self, input_signal, reference_signal):
        """
        LMS(Least Mean Squares)自适应滤波器
        """
        from scipy.signal import lfilter

        # 确保长度一致
        min_len = min(len(input_signal), len(reference_signal))
        input_signal = input_signal[:min_len]
        reference_signal = reference_signal[:min_len]

        # 滤波器阶数
        filter_order = 128
        mu = 0.01  # 步长

        # 初始化滤波器系数
        w = np.zeros(filter_order)
        output = np.zeros(min_len)

        for n in range(filter_order, min_len):
            # 参考信号的窗口
            ref_window = reference_signal[n - filter_order:n][::-1]

            # 预测的回声
            echo_estimate = np.dot(w, ref_window)

            # 误差(真实麦克风输入 - 预测回声)
            error = input_signal[n] - echo_estimate

            # 更新滤波器系数
            w += mu * error * ref_window

            output[n] = error

        return output[filter_order:]

# 集成到主循环
class InterruptibleVTuber:
    def __init__(self):
        self.echo_canceller = EchoCanceller()
        self.is_speaking = False

    async def play_audio(self, audio_data):
        """播放TTS音频,同时启用回声消除"""
        # 设置参考音频
        import soundfile as sf
        audio_array, sr = sf.read(io.BytesIO(audio_data))
        self.echo_canceller.set_reference(audio_array)

        # 标记正在说话
        self.is_speaking = True

        # 播放音频(异步)
        import sounddevice as sd
        sd.play(audio_array, sr)

        # 同时监听麦克风(检测打断)
        asyncio.create_task(self.monitor_interrupt())

    async def monitor_interrupt(self):
        """监听用户打断"""
        import sounddevice as sd

        # 录制麦克风
        duration = 0.5  # 每500ms检测一次
        while self.is_speaking:
            mic_data = sd.rec(
                int(duration * 16000),
                samplerate=16000,
                channels=1,
                dtype='float32'
            )
            sd.wait()

            # 回声消除
            clean_audio = self.echo_canceller.process_microphone(mic_data[:, 0])

            # 计算能量
            energy = np.sum(clean_audio ** 2)

            # 如果检测到用户说话(能量超过阈值)
            if energy > 0.01:
                # 停止播放
                sd.stop()
                self.is_speaking = False

                # 触发ASR识别用户输入
                await self.handle_interrupt(clean_audio)
                break

            await asyncio.sleep(0.1)
```

### 5.2 多语言翻译TTS

```python
# 场景:用中文聊天,但用日语声音回复
class TranslationTTS:
    def __init__(self, translator='deeplx', tts_provider='edge_tts'):
        """
        translator: google / deeplx / openai
        """
        self.translator = self.init_translator(translator)
        self.tts = EdgeTTSTTS(voice='ja-JP-NanamiNeural')  # 日语声音

    def init_translator(self, provider):
        if provider == 'deeplx':
            from deeplx import translate
            return translate

        elif provider == 'openai':
            from openai import OpenAI
            client = OpenAI()
            return lambda text, target: client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[{
                    'role': 'user',
                    'content': f'Translate to {target}: {text}'
                }]
            ).choices[0].message.content

    async def synthesize_with_translation(self, text_zh: str) -> bytes:
        """
        中文文本 → 翻译成日文 → 日语TTS
        """
        # 翻译
        if callable(self.translator):
            text_ja = self.translator(text_zh, target_lang='JA')
        else:
            text_ja = await self.translator(text_zh, 'Japanese')

        # 日语TTS
        audio = await self.tts.synthesize(text_ja)

        return audio

# 使用示例
translation_tts = TranslationTTS()

# 用户用中文提问
user_input_zh = '今天天气怎么样?'

# LLM用中文回复
llm_response_zh = '今天天气很好,阳光明媚!'

# 翻译成日语并合成语音
audio = await translation_tts.synthesize_with_translation(llm_response_zh)
# 音频内容:'今日は天気がとても良く、日差しが明るいです!'(日语发音)
```

### 5.3 MCP服务器集成

```json
// mcp_servers.json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/documents"]
    },
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "your_brave_api_key_here"
      }
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "your_github_token"
      }
    }
  }
}
```

**MCP使用示例**:
```python
# Agent可以调用MCP工具
class MCPAgent(Agent):
    def __init__(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self.mcp_clients = {}
        self.init_mcp_servers()

    def init_mcp_servers(self):
        """加载MCP服务器"""
        import json

        with open('mcp_servers.json', 'r') as f:
            config = json.load(f)

        for name, server_config in config['mcpServers'].items():
            # 启动MCP服务器
            server_params = StdioServerParameters(
                command=server_config['command'],
                args=server_config['args'],
                env=server_config.get('env', {})
            )

            client = stdio_client(server_params)
            self.mcp_clients[name] = client

    async def chat(self, user_input: str, context: dict) -> str:
        # 检测是否需要调用工具
        if '搜索' in user_input or 'search' in user_input.lower():
            # 调用Brave Search MCP
            search_client = self.mcp_clients['brave-search']

            # 提取搜索关键词
            query = user_input.replace('搜索', '').strip()

            # 调用工具
            result = await search_client.call_tool(
                'brave_web_search',
                arguments={'query': query, 'count': 5}
            )

            # 将结果传给LLM总结
            summary = await self.llm.ainvoke([
                HumanMessage(content=f'根据搜索结果回答用户:\n{result}\n\n用户问题:{user_input}')
            ])

            return summary.content

        elif '读取文件' in user_input:
            # 调用Filesystem MCP
            # ... 实现类似逻辑

        else:
            # 普通对话
            return await super().chat(user_input, context)
```

---

## 六、性能优化

### 6.1 延迟优化

```python
# 优化目标:总延迟<2秒
# 延迟组成:ASR(0.5s) + LLM(1.0s) + TTS(0.3s) + 渲染(0.1s) = 1.9s

class OptimizedPipeline:
    """
    优化策略:
    1. 流式输出(Streaming)
    2. 并行处理
    3. 模型量化
    """
    def __init__(self):
        # 1. 使用最快的ASR
        self.asr = SherpaOnnxASR(model_path='models/zipformer')  # ~100ms

        # 2. 本地LLM with int4量化
        self.llm = Ollama(model='qwen2.5:7b-instruct-q4_K_M')  # ~500ms

        # 3. 流式TTS
        self.tts = EdgeTTSTTS()  # 边生成边播放

    async def process_streaming(self, user_audio):
        """流式处理:边识别边回复"""
        # 1. ASR(异步)
        text = await self.asr.transcribe(user_audio)

        # 2. LLM流式生成
        full_response = ''
        sentence_buffer = ''

        async for chunk in self.llm.astream(text):
            full_response += chunk
            sentence_buffer += chunk

            # 检测句子结束(。!?等)
            if chunk in ['。', '!', '?', '\n']:
                # 立即合成这一句
                audio = await self.tts.synthesize(sentence_buffer)

                # 立即播放(不等全部生成完)
                asyncio.create_task(self.play_audio(audio))

                sentence_buffer = ''  # 清空buffer

        return full_response
```

**效果**:
- 传统模式:等待全部生成完(~3s) → 播放
- 流式模式:第一句话在1s内开始播放 ✅

### 6.2 GPU优化

```bash
# 1. Faster-Whisper启用FlashAttention
pip install flash-attn --no-build-isolation

# 2. LLM使用vLLM加速
pip install vllm

# Python代码
from vllm import LLM, SamplingParams

llm = LLM(
    model='Qwen/Qwen2.5-7B-Instruct',
    tensor_parallel_size=1,  # 单GPU
    gpu_memory_utilization=0.9
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=2048
)

outputs = llm.generate(prompts, sampling_params)
```

**性能提升**:
- Ollama: ~800ms/response
- vLLM: ~400ms/response (2x加速)

### 6.3 内存优化

```python
# 配置文件
config = {
    # 1. LLM量化
    'llm_quantization': 'int4',  # int8/int4/nf4

    # 2. ASR模型选择
    'asr_model_size': 'base',  # tiny(最小)/base/small

    # 3. TTS缓存
    'tts_cache_enabled': True,
    'tts_cache_dir': '/tmp/tts_cache',

    # 4. 对话历史限制
    'max_history_turns': 10,  # 只保留最近10轮

    # 5. Live2D资源优化
    'live2d_texture_quality': 'medium'  # low/medium/high
}

# TTS缓存实现
import hashlib

class CachedTTS:
    def __init__(self, tts_engine, cache_dir='/tmp/tts_cache'):
        self.tts = tts_engine
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    async def synthesize(self, text: str) -> bytes:
        # 计算文本hash
        text_hash = hashlib.md5(text.encode()).hexdigest()
        cache_path = f'{self.cache_dir}/{text_hash}.wav'

        # 检查缓存
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return f.read()

        # 未命中,调用TTS
        audio = await self.tts.synthesize(text)

        # 存入缓存
        with open(cache_path, 'wb') as f:
            f.write(audio)

        return audio
```

---

## 七、对比总结

| 特性 | Open-LLM-VTuber | OpenAvatarChat | VTube Studio |
|------|----------------|----------------|--------------|
| **开源** | ✅ MIT | ✅ Apache-2.0 | ❌ 商业软件 |
| **离线运行** | ✅ 完全支持 | ⚠️ 部分支持 | ❌ 需云服务 |
| **Live2D** | ✅ 原生支持 | ❌ 无 | ✅ 专业级 |
| **视觉感知** | ✅ 摄像头+截图 | ❌ 仅音频 | ❌ 无 |
| **模块化** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **MCP生态** | ✅ 原生集成 | ❌ 无 | ❌ 无 |
| **桌面宠物** | ✅ 透明窗口 | ❌ 无 | ⚠️ 需第三方 |
| **多语言TTS** | ✅ 翻译支持 | ⚠️ 手动配置 | ❌ 无 |
| **声音打断** | ✅ 回声消除 | ⚠️ 需耳机 | ✅ |

**Open-LLM-VTuber独特优势**:
1. **极致模块化**: 每个组件可独立替换
2. **完全离线**: 无需任何云服务即可运行
3. **Live2D集成**: 真正的虚拟主播体验
4. **视觉感知**: 支持摄像头/截图输入
5. **MCP生态**: 可调用文件系统/搜索/GitHub等工具

---

## 八、常见问题

### Q1: 如何自定义Live2D模型?

```bash
# 1. 准备Live2D模型(Cubism 3.0+)
#    - .model3.json (主文件)
#    - .moc3 (模型数据)
#    - .physics3.json (物理效果)
#    - textures/ (贴图目录)

# 2. 放入项目目录
cp -r my_model/ live2d-models/my_model/

# 3. 修改conf.yaml
live2d:
  model_path: 'live2d-models/my_model'

# 4. 重启服务
uv run run_server.py
```

### Q2: GPU显存不足?

```yaml
# conf.yaml优化
llm:
  model: 'qwen2.5:7b-instruct-q4_K_M'  # 使用int4量化

asr:
  model_size: 'tiny'  # 使用最小模型
  device: 'cpu'       # ASR用CPU,LLM用GPU

tts:
  provider: 'edge_tts'  # 使用云端TTS,释放GPU
```

### Q3: 延迟太高?

```python
# 启用流式模式
config['llm']['stream'] = True

# 使用更快的ASR
config['asr']['provider'] = 'sherpa_onnx'  # 比Whisper快10x

# 本地LLM换成云端API(牺牲隐私换速度)
config['llm']['provider'] = 'groq'  # 非常快的云端推理
config['llm']['model'] = 'llama-3.1-70b'
```

### Q4: 如何实现多角色切换?

```python
# characters/shizuku.yaml
name: 'Shizuku'
prompt_file: 'prompts/shizuku.txt'
tts_voice: 'zh-CN-XiaoxiaoNeural'
live2d_model: 'live2d-models/shizuku'

# characters/akari.yaml
name: 'Akari'
prompt_file: 'prompts/akari.txt'
tts_voice: 'ja-JP-NanamiNeural'
live2d_model: 'live2d-models/akari'

# 动态切换
class MultiCharacterAgent:
    def __init__(self):
        self.characters = {
            'shizuku': self.load_character('characters/shizuku.yaml'),
            'akari': self.load_character('characters/akari.yaml')
        }
        self.current = 'shizuku'

    def switch_character(self, name: str):
        if name in self.characters:
            self.current = name
            # 更新Live2D模型
            live2d_manager.load_model(self.characters[name]['live2d_model'])
            # 更新TTS声音
            tts.set_voice(self.characters[name]['tts_voice'])
```

---

**项目地址**: https://github.com/t41372/Open-LLM-VTuber
**文档**: https://github.com/t41372/Open-LLM-VTuber/wiki
**许可**: MIT (代码), Live2D素材需单独授权
**社区**: GitHub Issues / Discussions
