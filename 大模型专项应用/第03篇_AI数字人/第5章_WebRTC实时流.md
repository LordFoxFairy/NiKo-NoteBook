# 第5章 WebRTC实时流

## 5.1 aiortc架构

```python
from aiortc import RTCPeerConnection, VideoStreamTrack
import asyncio

class DigitalHumanTrack(VideoStreamTrack):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator
    
    async def recv(self):
        frame = await self.generator.get_next_frame()
        return frame

class WebRTCServer:
    async def handle_offer(self, offer):
        pc = RTCPeerConnection()
        
        @pc.on("track")
        async def on_track(track):
            if track.kind == "audio":
                # 处理音频输入
                pass
        
        # 添加视频轨
        video_track = DigitalHumanTrack(self.generator)
        pc.addTrack(video_track)
        
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        return answer
```

## 5.2 自适应码率

```python
class AdaptiveBitrateController:
    def __init__(self):
        self.target_bitrate = 2_000_000  # 2Mbps
    
    def adjust(self, packet_loss, rtt):
        if packet_loss > 0.05:
            self.target_bitrate *= 0.9
        elif rtt < 50:
            self.target_bitrate *= 1.1
        
        return self.target_bitrate
```

## 5.3 音视频同步

```python
class AVSynchronizer:
    def __init__(self):
        self.audio_queue = asyncio.Queue()
        self.video_queue = asyncio.Queue()
    
    async def sync(self):
        while True:
            audio_ts = await self.audio_queue.get()
            video_ts = await self.video_queue.get()
            
            # 时间戳对齐
            if abs(audio_ts - video_ts) > 50:  # 超过50ms
                # 丢帧或插帧
                pass
```

## 5.4 本章小结
- WebRTC实现<300ms端到端延迟
- 自适应码率应对网络抖动
- 音视频同步容忍度±50ms
