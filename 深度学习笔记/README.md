# 深度学习笔记

> PyTorch框架与现代深度学习技术全景

## 📚 系列定位

本系列聚焦**现代深度学习**技术，以PyTorch为核心框架，涵盖神经网络、CNN、Transformer等前沿架构。这是从传统ML跨越到现代AI的关键一步。

## 🎯 学习目标

- 掌握PyTorch框架核心API（Tensor、autograd、nn.Module）
- 理解神经网络训练原理（反向传播、梯度下降）
- 熟练构建CNN模型（LeNet、ResNet、ViT）
- 掌握Transformer架构（Self-Attention、Multi-Head）
- 具备完整的深度学习项目开发能力

## 📖 章节安排

### [第1章：深度学习基础](第1章_深度学习基础.md)

**核心内容**：
- 神经网络基础（感知机、多层网络）
- 反向传播算法详解
- 激活函数与正则化（BatchNorm、Dropout）
- PyTorch快速入门

**实战项目**：
- MNIST手写数字识别（PyTorch）
- CIFAR-10图像分类（从零构建CNN）

### 后续章节规划

- **第2章**：卷积神经网络深入（ResNet、MobileNet、EfficientNet）
- **第3章**：Transformer架构详解（Self-Attention、Vision Transformer）
- **第4章**：高级训练技巧（数据增强、学习率调度、混合精度）
- **第5章**：迁移学习与预训练模型

## 🛠 技术栈

```bash
# 核心框架
pip install torch torchvision torchaudio

# 扩展库
pip install timm              # PyTorch Image Models
pip install transformers      # Hugging Face
pip install albumentations    # 数据增强
pip install tensorboard       # 可视化

# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## 📈 学习路径

```
机器学习笔记 (传统ML基础)
    ↓
深度学习笔记 (本系列)
    ↓
计算机视觉算法 (CV专项应用)
```

## 💡 使用建议

1. **适合人群**：
   - 有机器学习基础，想学习深度学习
   - 需要掌握PyTorch框架
   - 准备进入CV/NLP等AI应用领域

2. **前置知识**：
   - 机器学习基本概念（建议先完成[机器学习笔记](../机器学习笔记/README.md)）
   - Python编程与NumPy
   - 基础数学（矩阵运算、导数）

3. **学习方式**：
   - **动手为王**：每个代码示例都要运行
   - **修改实验**：调整超参数观察效果
   - **GPU加速**：建议配置CUDA环境
   - **参考官方文档**：PyTorch官方教程质量很高

## 🎓 技能树

完成本系列后，你将掌握：

✅ PyTorch框架核心能力
✅ 神经网络架构设计
✅ 模型训练与调优
✅ CNN与Transformer原理
✅ 迁移学习实战经验

## 📌 后续系列

完成本系列后，建议继续学习：

- **[计算机视觉算法](../图像算法笔记/README.md)** - 目标检测、分割、生成模型等CV应用
- **[大模型笔记](../大模型笔记/README.md)** - LLM、多模态大模型等前沿技术

---

**预计学习时间**：20-30小时
**难度等级**：⭐⭐⭐
**更新日期**：2025年1月
