#!/usr/bin/env python3
"""
第10章完整测试代码
验证所有模块的正确性，并提供运行示例
"""

import numpy as np
import torch
import torch.nn as nn
import cv2
import librosa
from scipy import signal
import matplotlib.pyplot as plt

# ============================================================================
# 1. VAD模块测试
# ============================================================================

class SimpleVAD:
    """从零实现的VAD检测器"""
    def __init__(
        self,
        sample_rate=16000,
        frame_length=400,
        hop_length=160,
        energy_threshold=0.02,
        zcr_threshold=0.1
    ):
        self.sr = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.energy_threshold = energy_threshold
        self.zcr_threshold = zcr_threshold

        # 设计带通滤波器
        self.b, self.a = signal.butter(
            N=5,
            Wn=[80, 3400],
            btype='band',
            fs=sample_rate
        )

    def extract_features(self, audio):
        """提取VAD特征"""
        # 预处理
        filtered = signal.filtfilt(self.b, self.a, audio)

        # 分帧
        frames = librosa.util.frame(
            filtered,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )

        # 短时能量
        energy = np.sum(frames ** 2, axis=0) / self.frame_length

        # 过零率
        zcr = np.sum(np.abs(np.diff(np.sign(frames), axis=0)), axis=0) / (2 * self.frame_length)

        # 频谱平坦度
        fft_frames = np.fft.rfft(frames, axis=0)
        power_spectrum = np.abs(fft_frames) ** 2

        geo_mean = np.exp(np.mean(np.log(power_spectrum + 1e-10), axis=0))
        arith_mean = np.mean(power_spectrum, axis=0)
        sfm = geo_mean / (arith_mean + 1e-10)

        return {
            'energy': energy,
            'zcr': zcr,
            'sfm': sfm
        }

    def detect(self, audio):
        """检测是否包含语音"""
        features = self.extract_features(audio)

        # 归一化能量
        energy_norm = features['energy'] / (np.max(features['energy']) + 1e-10)

        # 语音帧判定
        speech_frames = (
            (energy_norm > self.energy_threshold) &
            (features['zcr'] > 0.02) &
            (features['zcr'] < 0.3) &
            (features['sfm'] < 0.5)
        )

        speech_ratio = np.sum(speech_frames) / len(speech_frames)
        is_speech = speech_ratio > 0.3
        confidence = speech_ratio

        return is_speech, confidence


def test_vad():
    """测试VAD模块"""
    print("=" * 50)
    print("测试1: VAD模块")
    print("=" * 50)

    # 生成测试信号
    sr = 16000
    duration = 1.0

    # 纯语音信号（模拟）
    t = np.linspace(0, duration, int(sr * duration))
    speech_signal = np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 400 * t)
    speech_signal += 0.1 * np.random.randn(len(t))

    # 纯噪声信号
    noise_signal = 0.1 * np.random.randn(int(sr * duration))

    # 初始化VAD
    vad = SimpleVAD(sample_rate=sr)

    # 测试语音
    is_speech, confidence = vad.detect(speech_signal)
    print(f"语音信号: is_speech={is_speech}, confidence={confidence:.3f}")
    assert is_speech, "应该检测到语音"

    # 测试噪声
    is_speech, confidence = vad.detect(noise_signal)
    print(f"噪声信号: is_speech={is_speech}, confidence={confidence:.3f}")
    assert not is_speech, "不应该检测到语音"

    print("✅ VAD测试通过\n")


# ============================================================================
# 2. Mel频谱提取测试
# ============================================================================

class MelSpectrogramExtractor:
    """从零实现Mel频谱提取"""
    def __init__(
        self,
        sample_rate=16000,
        n_fft=512,
        hop_length=160,
        n_mels=80,
        fmin=0,
        fmax=8000
    ):
        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        # 创建Mel滤波器组
        self.mel_filters = self.create_mel_filterbank(
            n_fft, n_mels, sample_rate, fmin, fmax
        )

    @staticmethod
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def create_mel_filterbank(self, n_fft, n_mels, sr, fmin, fmax):
        """创建Mel滤波器组"""
        # Mel尺度上的均匀点
        mel_min = self.hz_to_mel(fmin)
        mel_max = self.hz_to_mel(fmax)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)

        # 转回赫兹
        hz_points = self.mel_to_hz(mel_points)

        # 转换为FFT bin索引
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

        # 创建滤波器组
        filters = np.zeros((n_mels, n_fft // 2 + 1))

        for i in range(n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]

            # 上升沿
            for j in range(left, center):
                if center != left:
                    filters[i, j] = (j - left) / (center - left)

            # 下降沿
            for j in range(center, right):
                if right != center:
                    filters[i, j] = (right - j) / (right - center)

        return filters

    def extract(self, audio):
        """提取Mel频谱"""
        # 分帧
        frames = self._frame_audio(audio)

        # 加窗
        window = np.hanning(self.n_fft)
        frames_windowed = frames * window[:, None]

        # FFT
        fft_result = np.fft.rfft(frames_windowed, n=self.n_fft, axis=0)

        # 功率谱
        power_spectrum = np.abs(fft_result) ** 2

        # 应用Mel滤波器
        mel_spectrum = self.mel_filters @ power_spectrum

        # 取对数
        log_mel = 10 * np.log10(mel_spectrum + 1e-10)

        return log_mel

    def _frame_audio(self, audio):
        """音频分帧"""
        num_samples = len(audio)
        num_frames = 1 + (num_samples - self.n_fft) // self.hop_length

        frames = np.zeros((self.n_fft, num_frames))

        for i in range(num_frames):
            start = i * self.hop_length
            end = start + self.n_fft
            if end <= num_samples:
                frames[:, i] = audio[start:end]

        return frames


def test_mel_spectrogram():
    """测试Mel频谱提取"""
    print("=" * 50)
    print("测试2: Mel频谱提取")
    print("=" * 50)

    # 生成测试信号（单频音）
    sr = 16000
    duration = 1.0
    freq = 440  # A4音符

    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * freq * t)

    # 提取Mel频谱
    extractor = MelSpectrogramExtractor(sample_rate=sr)
    mel_spec = extractor.extract(audio)

    print(f"Mel频谱shape: {mel_spec.shape}")
    print(f"Mel频谱范围: [{mel_spec.min():.2f}, {mel_spec.max():.2f}] dB")

    # 验证shape
    assert mel_spec.shape[0] == 80, "应该有80个Mel bins"
    assert mel_spec.shape[1] > 0, "应该有时间帧"

    # 验证数值范围
    assert not np.isnan(mel_spec).any(), "不应该有NaN"
    assert not np.isinf(mel_spec).any(), "不应该有Inf"

    print("✅ Mel频谱测试通过\n")

    # 可选：可视化
    try:
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_spec, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='dB')
        plt.xlabel('Time (frames)')
        plt.ylabel('Mel frequency bins')
        plt.title('Mel Spectrogram')
        plt.tight_layout()
        plt.savefig('/tmp/mel_spectrogram_test.png', dpi=150)
        print("📊 Mel频谱图已保存到 /tmp/mel_spectrogram_test.png")
    except:
        print("⚠️  无法保存可视化图片（可能没有图形界面）")


# ============================================================================
# 3. 人脸对齐测试
# ============================================================================

class FaceAligner:
    """人脸对齐"""
    def __init__(self, target_size=256):
        self.target_size = target_size

        # 标准人脸关键点位置
        self.standard_landmarks = np.array([
            [0.31, 0.46],  # 左眼
            [0.69, 0.46],  # 右眼
            [0.50, 0.73]   # 鼻尖
        ]) * target_size

    def align(self, image, landmarks):
        """对齐人脸"""
        # 选择3个关键点
        if landmarks.shape[0] == 68:
            # 68点标准
            src_pts = np.array([
                landmarks[36:42].mean(axis=0),  # 左眼中心
                landmarks[42:48].mean(axis=0),  # 右眼中心
                landmarks[30]                    # 鼻尖
            ], dtype=np.float32)
        else:
            # 简化: 假设是3个点
            src_pts = landmarks[:3].astype(np.float32)

        dst_pts = self.standard_landmarks.astype(np.float32)

        # 计算仿射变换矩阵
        M = cv2.getAffineTransform(src_pts, dst_pts)

        # 应用变换
        aligned = cv2.warpAffine(
            image,
            M,
            (self.target_size, self.target_size),
            flags=cv2.INTER_LINEAR
        )

        return aligned, M


def test_face_alignment():
    """测试人脸对齐"""
    print("=" * 50)
    print("测试3: 人脸对齐")
    print("=" * 50)

    # 创建测试图像
    image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    # 模拟关键点（非标准位置）
    landmarks = np.array([
        [150, 200],  # 左眼
        [350, 220],  # 右眼
        [250, 350]   # 鼻尖
    ], dtype=np.float32)

    # 对齐
    aligner = FaceAligner(target_size=256)
    aligned, M = aligner.align(image, landmarks)

    print(f"原图shape: {image.shape}")
    print(f"对齐后shape: {aligned.shape}")
    print(f"仿射矩阵:\n{M}")

    # 验证
    assert aligned.shape == (256, 256, 3), "对齐后应该是256x256"
    assert M.shape == (2, 3), "仿射矩阵应该是2x3"

    # 验证关键点确实被对齐
    ones = np.ones((3, 1))
    src_homo = np.hstack([landmarks, ones])
    transformed_pts = (M @ src_homo.T).T

    expected_pts = aligner.standard_landmarks

    error = np.abs(transformed_pts - expected_pts).mean()
    print(f"对齐误差: {error:.2f} 像素")
    assert error < 1.0, "对齐误差应该小于1像素"

    print("✅ 人脸对齐测试通过\n")


# ============================================================================
# 4. 简单GAN测试
# ============================================================================

class SimpleGenerator(nn.Module):
    """简化的生成器"""
    def __init__(self, latent_dim=100, output_channels=3):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (256, 8, 8)),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, output_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


def test_generator():
    """测试生成器"""
    print("=" * 50)
    print("测试4: GAN生成器")
    print("=" * 50)

    # 初始化模型
    generator = SimpleGenerator(latent_dim=100, output_channels=3)

    # 测试前向传播
    batch_size = 4
    z = torch.randn(batch_size, 100)

    with torch.no_grad():
        fake_images = generator(z)

    print(f"输入shape: {z.shape}")
    print(f"输出shape: {fake_images.shape}")
    print(f"输出范围: [{fake_images.min():.3f}, {fake_images.max():.3f}]")

    # 验证
    assert fake_images.shape == (batch_size, 3, 64, 64), "输出shape应该是(B,3,64,64)"
    assert -1 <= fake_images.min() <= 1, "Tanh输出应该在[-1,1]"
    assert -1 <= fake_images.max() <= 1, "Tanh输出应该在[-1,1]"

    print("✅ 生成器测试通过\n")


# ============================================================================
# 5. 集成测试
# ============================================================================

def test_integration():
    """集成测试：模拟完整pipeline"""
    print("=" * 50)
    print("测试5: 完整Pipeline集成")
    print("=" * 50)

    # 1. VAD检测
    print("步骤1: VAD检测...")
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 200 * t)

    vad = SimpleVAD(sample_rate=sr)
    is_speech, _ = vad.detect(audio)
    print(f"  ✓ 检测到语音: {is_speech}")

    # 2. 音频特征提取
    print("步骤2: 提取Mel频谱...")
    mel_extractor = MelSpectrogramExtractor(sample_rate=sr)
    mel_spec = mel_extractor.extract(audio)
    print(f"  ✓ Mel频谱shape: {mel_spec.shape}")

    # 3. 人脸处理
    print("步骤3: 人脸对齐...")
    image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    landmarks = np.array([[150, 200], [350, 220], [250, 350]], dtype=np.float32)

    aligner = FaceAligner(target_size=256)
    aligned_face, _ = aligner.align(image, landmarks)
    print(f"  ✓ 对齐后shape: {aligned_face.shape}")

    # 4. 生成
    print("步骤4: 生成人脸...")
    generator = SimpleGenerator()
    z = torch.randn(1, 100)
    with torch.no_grad():
        generated = generator(z)
    print(f"  ✓ 生成shape: {generated.shape}")

    print("\n✅ 完整Pipeline集成测试通过")


# ============================================================================
# 主测试函数
# ============================================================================

def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 50)
    print("开始运行所有测试")
    print("=" * 50 + "\n")

    try:
        test_vad()
        test_mel_spectrogram()
        test_face_alignment()
        test_generator()
        test_integration()

        print("\n" + "=" * 50)
        print("🎉 所有测试通过！")
        print("=" * 50 + "\n")

        return True

    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    except Exception as e:
        print(f"\n❌ 运行错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()

    if success:
        print("✅ 代码验证完成，所有模块工作正常！")
        print("\n下一步：")
        print("1. 使用真实音频文件测试VAD")
        print("2. 使用真实人脸图片测试对齐")
        print("3. 在GPU上训练完整GAN模型")
    else:
        print("❌ 部分测试失败，请检查代码")

    exit(0 if success else 1)
