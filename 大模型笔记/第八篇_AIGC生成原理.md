# 第八篇：AIGC生成原理

> 从数学原理深度理解文本、图像、视频、音频生成模型

**适合人群**: 算法研究员、深度学习开发者、AI架构师
**预计时间**: 15-20小时
**前置知识**: 第一篇(Transformer架构)、第二篇(预训练技术)、第三篇(微调技术)

---

## 本篇概览

深入**AIGC生成模型的数学原理**：

**Part A: 文本生成原理** (第1-3章)
- Transformer自回归生成机制
- 解码策略完整数学推导
- 采样控制与质量优化

**Part B: 图像生成原理** (第4-7章)
- Diffusion模型完整数学推导
- VAE潜空间与重参数化技巧
- GAN训练稳定性理论
- 条件生成与ControlNet原理

**Part C: 视频与音频生成原理** (第8-10章)
- 视频扩散模型（Sora架构原理）
- 声学模型与TTS数学基础
- 多模态对齐机制

---

# Part A: 文本生成原理

## 第1章：Transformer自回归生成

### 1.1 自回归语言建模

**核心思想**: 逐token生成,每个token基于前文概率分布采样。

**数学形式**:

给定提示 $x = [x_1, ..., x_n]$, 生成响应 $y = [y_1, ..., y_m]$:
$$
P(y | x) = \prod_{t=1}^m P(y_t | x, y_{<t})
$$

**Transformer实现**:

1. **编码提示**:
$$
H_x = \text{Transformer}_{\text{encoder}}(x)
$$

2. **逐步解码**:
$$
P(y_t | x, y_{<t}) = \text{Softmax}(W_o \cdot h_t)
$$

其中 $h_t = \text{Transformer}_{\text{decoder}}(y_{<t}, H_x)$

### 1.2 因果注意力掩码

**关键机制**: 确保第$t$个token只能看到前$t-1$个token。

**掩码矩阵**:
$$
M_{ij} = \begin{cases}
0 & \text{if } i \geq j \\
-\infty & \text{if } i < j
\end{cases}
$$

**注意力计算**:
$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

**为什么需要掩码?**

没有掩码,模型会"看到未来":
$$
P(y_t | x, y_{<t}, y_{\geq t}) \neq P(y_t | x, y_{<t})
$$

训练时有ground truth,推理时没有 → 分布不匹配!

### 1.3 KV Cache优化

**问题**: 每生成一个token,都要重新计算所有历史token的KV!

**传统计算** (生成$m$个token):
$$
\text{FLOPs} = O(m^2 \cdot d)
$$

**KV Cache策略**:

缓存已计算的Key和Value:
$$
K_{\text{cache}} = [K_1, K_2, ..., K_{t-1}]
$$
$$
V_{\text{cache}} = [V_1, V_2, ..., V_{t-1}]
$$

生成第$t$个token时:
$$
K_t' = \text{concat}(K_{\text{cache}}, K_t)
$$
$$
\text{Attention}_t = \text{Softmax}\left(\frac{Q_t K_t'^T}{\sqrt{d_k}}\right)V_t'
$$

**加速效果**:
- 无Cache: $O(m^2)$
- 有Cache: $O(m)$
- **提速**: 10-100倍!

### 1.4 生成停止条件

**三种停止方式**:

1. **EOS token**: 模型生成 `</s>`
$$
y_t = \text{<EOS>} \Rightarrow \text{stop}
$$

2. **最大长度**: 达到预设长度
$$
t = \text{max_length} \Rightarrow \text{stop}
$$

3. **概率阈值**: 所有token概率都很低
$$
\max_i P(y_t = i | x, y_{<t}) < \tau \Rightarrow \text{stop}
$$

---

## 第2章：解码策略数学

### 2.1 贪心解码

**策略**: 每步选择概率最高的token。

**数学定义**:
$$
y_t = \arg\max_{w \in \mathcal{V}} P(w | x, y_{<t})
$$

**优点**:
- 计算简单: $O(|\mathcal{V}|)$
- 确定性输出

**缺点**:
- **局部最优**: 可能陷入重复
- **不等于全局最优**!

**反例**:

假设:
$$
P(\text{"the cat"}) = 0.5, \quad P(\text{"a dog"}) = 0.6
$$

但:
$$
P(\text{"the cat sat"}) = 0.8, \quad P(\text{"a dog sat"}) = 0.3
$$

贪心选"a" → 最终得分0.3  
最优选"the" → 最终得分0.8

### 2.2 Beam Search

**核心思想**: 维护$k$个最优候选序列。

**算法流程**:

1. **初始化**: $\text{Beams} = \{[\text{<BOS>}]\}$

2. **扩展**: 对每个beam,生成所有可能的下一个token
$$
\text{Candidates} = \{(b, w) : b \in \text{Beams}, w \in \mathcal{V}\}
$$

3. **打分**:
$$
\text{Score}(b, w) = \log P(b) + \log P(w | x, b)
$$

4. **选择Top-K**:
$$
\text{Beams}_{\text{new}} = \text{TopK}_{(b,w)}(\text{Score}(b, w))
$$

5. **重复** 直到所有beam结束

**数学保证**:

Beam Search找到的是**近似全局最优**:
$$
\hat{y} \approx \arg\max_{y} P(y | x)
$$

**Beam宽度$k$的影响**:

| Beam宽度 | 计算量 | 质量 | 多样性 |
|---------|--------|------|--------|
| $k=1$ | $O(m \cdot \|\mathcal{V}\|)$ | 低 | 低 |
| $k=5$ | $O(5m \cdot \|\mathcal{V}\|)$ | 中 | 中 |
| $k=20$ | $O(20m \cdot \|\mathcal{V}\|)$ | 高 | 低 |

**长度归一化**:

问题: 长序列概率自然更低!
$$
P(y) = \prod_{t=1}^m P(y_t | x, y_{<t}) \to 0 \text{ as } m \to \infty
$$

解决: 归一化分数
$$
\text{Score}(y) = \frac{1}{m^\alpha} \sum_{t=1}^m \log P(y_t | x, y_{<t})
$$

通常 $\alpha = 0.6-0.8$

### 2.3 采样解码

#### 2.3.1 温度采样

**核心**: 调整概率分布的"平滑度"。

**原始分布**:
$$
P(w_i | x, y_{<t}) = \frac{\exp(z_i)}{\sum_j \exp(z_j)}
$$

**温度调整**:
$$
P_T(w_i | x, y_{<t}) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

**温度$T$的影响**:

**$T \to 0$**: 趋向贪心
$$
P_T(w_i) \to \begin{cases}
1 & \text{if } i = \arg\max_j z_j \\
0 & \text{otherwise}
\end{cases}
$$

**$T = 1$**: 原始分布

**$T \to \infty$**: 趋向均匀分布
$$
P_T(w_i) \to \frac{1}{|\mathcal{V}|}
$$

**数学解释**:

温度控制**熵**:
$$
H(P_T) = -\sum_i P_T(w_i) \log P_T(w_i)
$$

- $T$↑ → $H$↑ → 更随机
- $T$↓ → $H$↓ → 更确定

**实践建议**:
- 创意写作: $T = 0.8-1.2$
- 代码生成: $T = 0.2-0.5$
- 事实问答: $T = 0.1-0.3$

#### 2.3.2 Top-p (Nucleus) 采样

**核心思想**: 从累积概率达到$p$的最小词汇集中采样。

**数学定义**:

1. **排序词汇**:
$$
w_{(1)}, w_{(2)}, ..., w_{(|\mathcal{V}|)} \quad \text{s.t.} \quad P(w_{(1)}) \geq P(w_{(2)}) \geq ...
$$

2. **找到核心集**:
$$
V_p = \{w_{(1)}, ..., w_{(k)}\} \quad \text{where} \quad \sum_{i=1}^k P(w_{(i)}) \geq p
$$

3. **重新归一化采样**:
$$
P'(w) = \begin{cases}
\frac{P(w)}{\sum_{w' \in V_p} P(w')} & \text{if } w \in V_p \\
0 & \text{otherwise}
\end{cases}
$$

**与Top-k的区别**:

| 方法 | 词汇集大小 | 优点 | 缺点 |
|------|-----------|------|------|
| **Top-k** | 固定$k$ | 简单 | 不适应分布变化 |
| **Top-p** | 动态 | 自适应 | 计算稍慢 |

**示例**:

假设分布:
```
P(good) = 0.6
P(great) = 0.3
P(nice) = 0.05
P(ok) = 0.03
P(bad) = 0.02
```

- Top-3: $\{\text{good, great, nice}\}$ (固定3个)
- Top-p ($p=0.9$): $\{\text{good, great}\}$ (累积0.9,动态2个)

**典型值**: $p = 0.9-0.95$

#### 2.3.3 Top-k + Top-p 混合

**最佳实践**: 先Top-k截断,再Top-p采样

```python
def nucleus_sampling(logits, top_k=50, top_p=0.9, temperature=1.0):
    # 1. 温度调整
    logits = logits / temperature
    
    # 2. Top-k截断
    top_k_logits, top_k_indices = torch.topk(logits, top_k)
    
    # 3. Softmax
    probs = F.softmax(top_k_logits, dim=-1)
    
    # 4. Top-p过滤
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 找到累积概率超过p的位置
    nucleus_mask = cumsum_probs <= top_p
    nucleus_mask[0] = True  # 至少保留一个token
    
    # 5. 重新归一化
    nucleus_probs = sorted_probs * nucleus_mask
    nucleus_probs = nucleus_probs / nucleus_probs.sum()
    
    # 6. 采样
    sampled_idx = torch.multinomial(nucleus_probs, 1)
    return top_k_indices[sorted_indices[sampled_idx]]
```

### 2.4 对比总结

| 策略 | 确定性 | 多样性 | 质量 | 速度 | 适用场景 |
|------|--------|--------|------|------|---------|
| **贪心** | 高 | 低 | 中 | 快 | 简单问答 |
| **Beam Search** | 中 | 低 | 高 | 慢 | 机器翻译 |
| **温度采样** | 低 | 高 | 中 | 快 | 创意写作 |
| **Top-p** | 低 | 中高 | 高 | 中 | **通用推荐** |
| **混合** | 低 | 中 | 高 | 中 | 生产环境 |

---

## 第3章：生成质量控制

### 3.1 重复惩罚

**问题**: 模型倾向重复已生成内容。

**N-gram重复惩罚**:

对已出现的n-gram,降低其概率:
$$
P'(w_t) = \begin{cases}
P(w_t) / \alpha & \text{if } (w_{t-n+1}, ..., w_{t-1}, w_t) \in \text{Generated} \\
P(w_t) & \text{otherwise}
\end{cases}
$$

通常 $\alpha = 1.2$, $n = 3$

**频率惩罚** (GPT-3风格):
$$
\text{logit}_i' = \text{logit}_i - \text{frequency\_penalty} \cdot \text{count}(w_i)
$$

### 3.2 长度控制

**最小长度惩罚**:

在达到最小长度前,禁止EOS:
$$
P(\text{<EOS>} | y_{<t}) = \begin{cases}
0 & \text{if } t < L_{\min} \\
P_{\text{model}}(\text{<EOS>} | y_{<t}) & \text{otherwise}
\end{cases}
$$

**长度奖励**:

鼓励特定长度:
$$
\text{Score}(y) = \log P(y | x) + \lambda \cdot \text{LengthReward}(|y|)
$$

其中:
$$
\text{LengthReward}(l) = -|l - L_{\text{target}}|
$$

### 3.3 CLIP引导生成

**跨模态一致性**:

对图像描述生成,用CLIP确保文本与图像匹配:
$$
\text{Score}(y) = \log P(y | x) + \beta \cdot \text{CLIP}(y, I)
$$

**CLIP分数**:
$$
\text{CLIP}(y, I) = \frac{\text{Enc}_{\text{text}}(y)^T \text{Enc}_{\text{image}}(I)}{\|\text{Enc}_{\text{text}}(y)\| \cdot \|\text{Enc}_{\text{image}}(I)\|}
$$

---


# Part B: 图像生成原理

## 第4章：Diffusion模型完整数学推导

### 4.1 扩散过程基础

**核心思想**: 通过逐步添加噪声破坏图像，再学习逆过程恢复图像。

#### 4.1.1 前向扩散过程

给定真实图像 $x_0 \sim q(x_0)$，通过$T$步逐渐添加高斯噪声：

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)
$$

其中 $\beta_1, ..., \beta_T$ 是噪声调度（noise schedule），通常 $\beta_t \in [10^{-4}, 0.02]$。

**完整链**：
$$
q(x_{1:T} | x_0) = \prod_{t=1}^T q(x_t | x_{t-1})
$$

**关键性质**：可以一步采样任意时刻 $x_t$！

定义 $\alpha_t = 1 - \beta_t$，$\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$，则：

$$
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) I)
$$

**重参数化采样**：
$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

**直觉**：
- $t=0$: $x_0$ 是原图
- $t$增大: 信号逐渐被噪声淹没
- $t=T$: $x_T \approx \mathcal{N}(0, I)$ 纯噪声

#### 4.1.2 反向去噪过程

**目标**：学习 $p_\theta(x_{t-1} | x_t)$ 逆转扩散过程。

**后验分布**（当 $\beta_t$ 很小时）：
$$
q(x_{t-1} | x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I)
$$

其中：
$$
\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t
$$

$$
\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t
$$

**神经网络参数化**：

用 $\epsilon_\theta(x_t, t)$ 预测加入的噪声：
$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$

其中：
$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right)
$$

### 4.2 训练目标推导

**变分下界（ELBO）**：

最大化对数似然的下界：
$$
\log p_\theta(x_0) \geq \mathbb{E}_{q(x_{1:T}|x_0)} \left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right]
$$

**展开为**：
$$
\mathcal{L}_{\text{VLB}} = \mathbb{E}_q \left[\underbrace{D_{KL}(q(x_T|x_0) \| p(x_T))}_{L_T} + \sum_{t>1} \underbrace{D_{KL}(q(x_{t-1}|x_t,x_0) \| p_\theta(x_{t-1}|x_t))}_{L_{t-1}} \underbrace{- \log p_\theta(x_0|x_1)}_{L_0}\right]
$$

**简化目标**（DDPM论文）：

可以证明，优化上述等价于：
$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]
$$

其中：
- $t \sim \text{Uniform}(1, T)$
- $x_0 \sim q(x_0)$
- $\epsilon \sim \mathcal{N}(0, I)$
- $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$

**直觉**：训练网络预测每一步加入的噪声！

### 4.3 采样算法

**DDPM采样**（$T=1000$步）：

```
输入：随机噪声 z_T ~ N(0, I)
for t = T, T-1, ..., 1 do
    z ~ N(0, I) if t > 1 else 0
    x_{t-1} = 1/√α_t * (x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ(x_t, t)) + √β_t * z
end for
返回：x_0
```

**数学形式**：
$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right) + \sigma_t z
$$

**DDIM加速采样**（$T'=50$步）：

跳过中间步骤，直接计算：
$$
x_{t-\Delta t} = \sqrt{\bar{\alpha}_{t-\Delta t}} \underbrace{\left(\frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}\right)}_{\text{预测的}x_0} + \sqrt{1-\bar{\alpha}_{t-\Delta t}} \epsilon_\theta(x_t, t)
$$

**加速效果**：1000步 → 50步，速度提升20倍！

### 4.4 噪声调度策略

**线性调度** (DDPM原始)：
$$
\beta_t = \beta_{\min} + \frac{t-1}{T-1}(\beta_{\max} - \beta_{\min})
$$

通常 $\beta_{\min}=0.0001$, $\beta_{\max}=0.02$

**余弦调度** (Improved DDPM)：
$$
\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)^2
$$

其中 $s=0.008$

**效果对比**：

| 调度方式 | FID↓ | 训练稳定性 |
|---------|------|-----------|
| 线性 | 3.2 | 中 |
| 余弦 | **2.9** | 高 |

### 4.5 条件生成

**文本条件**（Stable Diffusion）：

修改噪声预测网络：
$$
\epsilon_\theta(x_t, t, c)
$$

其中 $c$ 是文本编码（来自CLIP/T5）。

**Classifier-Free Guidance**：

混合条件和无条件预测：
$$
\tilde{\epsilon}_\theta(x_t, t, c) = \epsilon_\theta(x_t, t, \emptyset) + w \cdot (\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset))
$$

其中：
- $w$: 引导权重（通常7.5）
- $w=0$: 无条件生成
- $w$增大: 更强的文本一致性，但多样性降低

**训练技巧**：

以概率 $p=0.1$ 随机丢弃条件：
$$
c' = \begin{cases}
\emptyset & \text{with prob } 0.1 \\
c & \text{otherwise}
\end{cases}
$$

---

## 第5章：VAE与潜空间

### 5.1 VAE核心原理

**问题**：直接在像素空间训练Diffusion计算量巨大（512×512×3 = 786k维）。

**解决**：先用VAE压缩到潜空间！

**VAE架构**：
$$
\text{图像} \xrightarrow{\text{Encoder}} \text{潜变量} z \xrightarrow{\text{Decoder}} \text{重建图像}
$$

#### 5.1.1 编码器

$$
q_\phi(z | x) = \mathcal{N}(z; \mu_\phi(x), \sigma_\phi^2(x) I)
$$

**重参数化技巧**：
$$
z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

这样梯度可以反向传播！

#### 5.1.2 解码器

$$
p_\theta(x | z) = \mathcal{N}(x; \mu_\theta(z), \sigma^2 I)
$$

#### 5.1.3 训练目标

**ELBO损失**：
$$
\mathcal{L}_{\text{VAE}} = \underbrace{\mathbb{E}_{q_\phi(z|x)}[-\log p_\theta(x|z)]}_{\text{重建损失}} + \underbrace{D_{KL}(q_\phi(z|x) \| p(z))}_{\text{KL正则}}
$$

**KL散度解析解**：

当先验 $p(z) = \mathcal{N}(0, I)$ 时：
$$
D_{KL} = \frac{1}{2} \sum_{j=1}^J \left(\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1\right)
$$

### 5.2 Latent Diffusion (Stable Diffusion)

**两阶段训练**：

**阶段1：训练VAE**
- 输入：512×512×3图像
- 输出：64×64×4潜变量
- 压缩比：8×8×0.75 = **48倍**！

**阶段2：在潜空间训练Diffusion**

$$
\mathcal{L} = \mathbb{E}_{t, z_0, \epsilon} \left[\|\epsilon - \epsilon_\theta(z_t, t, c)\|^2\right]
$$

其中 $z_0 = \text{Encoder}(x_0)$

**采样流程**：
```
1. z_T ~ N(0, I)  # 64×64×4
2. for t = T...1:
      z_{t-1} = DDIM_step(z_t, t, text_prompt)
3. x_0 = Decoder(z_0)  # 512×512×3
```

**速度提升**：

| 模型 | 分辨率 | 步数 | 时间(V100) |
|------|-------|------|-----------|
| Pixel Diffusion | 512² | 1000 | 150s |
| Latent Diffusion | 64² | 50 | **3s** |

---

## 第6章：GAN原理与训练

### 6.1 GAN数学框架

**博弈论视角**：生成器$G$和判别器$D$的零和博弈。

**目标函数**：
$$
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

**解释**：
- $D$: 最大化区分真假的能力
- $G$: 最小化被识破的概率

**纳什均衡**：

当 $G$ 完美时：$p_g = p_{\text{data}}$

此时最优判别器：
$$
D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)} = \frac{1}{2}
$$

### 6.2 训练不稳定性

**模式崩溃** (Mode Collapse)：

$G$ 只生成数据分布的部分模式。

**数学原因**：

$G$ 优化目标：
$$
\min_G \mathbb{E}_{z}[\log(1 - D(G(z)))]
$$

等价于最小化JS散度：
$$
\min_G \text{JS}(p_{\text{data}} \| p_g)
$$

但JS散度在分布不重叠时为常数 → 梯度消失！

**解决方案**：

**1. Wasserstein GAN**：

用Wasserstein距离替代JS散度：
$$
W(p_{\text{data}}, p_g) = \inf_{\gamma \in \Pi(p_{\text{data}}, p_g)} \mathbb{E}_{(x,y) \sim \gamma}[\|x - y\|]
$$

**优点**：即使分布不重叠，梯度仍然有意义！

**2. Spectral Normalization**：

约束判别器Lipschitz常数：
$$
\|D\|_{\text{Lip}} \leq 1
$$

通过谱归一化权重矩阵实现。

### 6.3 StyleGAN架构

**创新**：将噪声映射到风格空间，逐层注入风格。

**Mapping Network**：
$$
z \xrightarrow{\text{MLP}} w \in \mathcal{W}
$$

**风格注入**（AdaIN）：

对第 $l$ 层特征 $x_l$：
$$
\text{AdaIN}(x_l, w) = \gamma_l(w) \frac{x_l - \mu(x_l)}{\sigma(x_l)} + \beta_l(w)
$$

**优势**：
- 解耦不同层级的风格（粗糙/中等/精细）
- 支持风格混合

---

## 第7章：ControlNet条件控制

### 7.1 核心思想

**问题**：Diffusion模型难以精确控制空间结构（姿态、边缘、深度）。

**ControlNet方案**：

添加可训练的控制分支，注入额外条件（边缘图、深度图、姿态）。

### 7.2 架构设计

**原始U-Net**：
$$
x_t \xrightarrow{\text{Encoder}} h \xrightarrow{\text{Decoder}} \epsilon_\theta
$$

**ControlNet增强**：

复制Encoder，创建可训练副本：
$$
c \xrightarrow{\text{ControlNet}} \Delta h
$$

**特征融合**：
$$
h' = h + \Delta h
$$

**零初始化**：

初始时，ControlNet输出为0：
$$
\Delta h|_{t=0} = 0 \Rightarrow h' = h
$$

这样不影响预训练权重！

### 7.3 条件类型

| 条件 | 提取方法 | 用途 |
|------|---------|------|
| **Canny边缘** | Canny算法 | 保持轮廓 |
| **深度图** | MiDaS模型 | 控制空间布局 |
| **姿态** | OpenPose检测 | 人物姿态控制 |
| **语义分割** | Segmentation模型 | 精确区域控制 |

### 7.4 训练策略

**冻结原模型**：
$$
\theta_{\text{UNet}} \leftarrow \text{frozen}
$$

**只训练ControlNet**：
$$
\theta_{\text{Control}} \leftarrow \text{trainable}
$$

**损失函数**：
$$
\mathcal{L} = \mathbb{E}_{t, x_0, c, \epsilon} \left[\|\epsilon - \epsilon_\theta(x_t, t, c_{\text{text}}, c_{\text{control}})\|^2\right]
$$

**数据需求**：

每个控制类型约50k图像对即可！

---


# Part C: 视频与音频生成原理

## 第8章：视频扩散模型（Sora架构原理）

### 8.1 视频表示

**挑战**：视频是4D数据 $(T, H, W, C)$，计算量爆炸！

**Sora方案**：将视频切分为时空patches。

#### 8.1.1 Spacetime Patches

**3D Patch嵌入**：

将视频切分为大小 $(t_p, h_p, w_p)$ 的patches：
$$
\text{Video}_{(T, H, W, C)} \rightarrow \text{Patches}_{(N_t \times N_h \times N_w, D)}
$$

其中：
- $N_t = T / t_p$（时间patches数）
- $N_h = H / h_p$, $N_w = W / w_p$（空间patches数）
- $D$: 嵌入维度

**位置编码**：

3D正弦位置编码：
$$
\text{PE}(t, h, w) = [\sin(\omega_t t), \cos(\omega_t t), \sin(\omega_h h), \cos(\omega_h h), \sin(\omega_w w), \cos(\omega_w w)]
$$

### 8.2 时空注意力

**朴素3D注意力** (太慢)：
$$
\text{Attention}(Q, K, V) \in \mathbb{R}^{(T \cdot H \cdot W) \times D}
$$

复杂度：$O((THW)^2)$ - 不可行！

**分解注意力** (Sora方案)：

**1. 空间注意力**（每帧独立）：
$$
\text{SpatialAttn}: \mathbb{R}^{T \times HW \times D} \rightarrow \mathbb{R}^{T \times HW \times D}
$$

复杂度：$O(T \cdot (HW)^2)$

**2. 时间注意力**（每个空间位置跨帧）：
$$
\text{TemporalAttn}: \mathbb{R}^{HW \times T \times D} \rightarrow \mathbb{R}^{HW \times T \times D}
$$

复杂度：$O(HW \cdot T^2)$

**总复杂度**：
$$
O(T \cdot (HW)^2 + HW \cdot T^2) \ll O((THW)^2)
$$

对于720p 60帧视频：
- 朴素3D: $10^{15}$ FLOPs
- 分解: $10^{12}$ FLOPs（快1000倍！）

### 8.3 可变长度生成

**问题**：如何训练一个模型生成不同分辨率、帧率的视频？

**Sora创新**：原生支持可变分辨率！

**动态Patch策略**：

对任意分辨率 $(T, H, W)$：
$$
N_{\text{patches}} = \left\lfloor \frac{T}{t_p} \right\rfloor \times \left\lfloor \frac{H}{h_p} \right\rfloor \times \left\lfloor \frac{W}{w_p} \right\rfloor
$$

**位置编码插值**：

对训练时未见的分辨率，插值位置编码：
$$
\text{PE}_{\text{new}}(t, h, w) = \text{Interpolate}(\text{PE}_{\text{train}}, (t, h, w))
$$

### 8.4 视频Diffusion训练

**潜空间编码**：

使用3D VAE压缩：
$$
z = \text{Encoder}_{3D}(\text{Video}) \in \mathbb{R}^{T/4 \times H/8 \times W/8 \times C}
$$

**训练目标**：
$$
\mathcal{L} = \mathbb{E}_{t, z_0, \epsilon, c} \left[\|\epsilon - \epsilon_\theta(z_t, t, c_{\text{text}})\|^2\right]
$$

**文本条件**：

T5编码器 + 交叉注意力：
$$
\text{CrossAttn}(Q_{\text{video}}, K_{\text{text}}, V_{\text{text}})
$$

### 8.5 数据增强策略

**时间增强**：
- 随机帧率采样
- 时间反转
- 循环播放

**空间增强**：
- 随机裁剪
- 水平翻转
- 颜色抖动

**关键**：保持时间一致性！

---

## 第9章：TTS与音频生成原理

### 9.1 语音生成pipeline

**传统TTS三阶段**：
```
文本 → [文本处理] → 音素序列 → [声学模型] → 梅尔频谱 → [声码器] → 波形
```

### 9.2 声学模型

#### 9.2.1 梅尔频谱表示

**短时傅里叶变换(STFT)**：
$$
X(m, k) = \sum_{n=0}^{N-1} x[n] w[n - mH] e^{-j2\pi kn / N}
$$

其中：
- $m$: 时间帧索引
- $k$: 频率bin
- $H$: hop length
- $w$: 窗函数（通常Hann窗）

**梅尔滤波器组**：

模拟人耳感知，低频精细、高频粗糙：
$$
\text{Mel}(f) = 2595 \log_{10}\left(1 + \frac{f}{700}\right)
$$

**梅尔频谱**：
$$
M(m, b) = \sum_{k=0}^{K-1} |X(m, k)|^2 H_b(k)
$$

其中 $H_b$ 是第$b$个梅尔滤波器。

#### 9.2.2 Tacotron 2架构

**Encoder-Attention-Decoder**结构：

**Encoder**（文本→隐藏表示）：
$$
h_{\text{text}} = \text{BiLSTM}(\text{Embed}(\text{phonemes}))
$$

**Attention**（对齐文本和音频）：
$$
\alpha_t = \text{Softmax}(\text{score}(s_{t-1}, h_{\text{text}}))
$$
$$
c_t = \sum_i \alpha_{t,i} h_{\text{text},i}
$$

**Decoder**（生成梅尔频谱）：
$$
s_t = \text{LSTM}(s_{t-1}, [y_{t-1}, c_t])
$$
$$
y_t = \text{Linear}(s_t)
$$

**Stop Token**：

预测何时停止生成：
$$
p_{\text{stop}}(t) = \sigma(W_{\text{stop}} s_t)
$$

### 9.3 声码器（Vocoder）

**任务**：梅尔频谱 → 波形

#### 9.3.1 WaveNet

**自回归生成**：
$$
p(x) = \prod_{t=1}^T p(x_t | x_{<t})
$$

**因果卷积 + 扩张卷积**：

接受野：$r = 2^L - 1$（$L$层，扩张率$2^l$）

**条件生成**：
$$
p(x_t | x_{<t}, c) = \text{Softmax}(W_{\text{out}} \tanh(W_f * x) \odot \sigma(W_g * x + V_g * c))
$$

**问题**：太慢！生成1秒音频需要1分钟。

#### 9.3.2 HiFi-GAN

**生成器**：

多尺度上采样：
$$
\text{Mel} \xrightarrow{\text{Upsample 8x}} h_1 \xrightarrow{\text{Upsample 8x}} h_2 \xrightarrow{\text{Upsample 2x}} h_3 \xrightarrow{\text{Conv}} \text{Waveform}
$$

**多周期判别器(MPD)**：

对不同周期 $p \in \{2, 3, 5, 7, 11\}$，reshape波形：
$$
x_p = \text{Reshape}(x, [\lfloor T/p \rfloor, p])
$$

然后用2D卷积判别。

**多尺度判别器(MSD)**：

在不同下采样率评估：
$$
D_{\text{scale-1}}(x), \quad D_{\text{scale-2}}(\text{AvgPool}_2(x)), \quad D_{\text{scale-3}}(\text{AvgPool}_4(x))
$$

**对抗损失**：
$$
\mathcal{L}_{\text{adv}} = \mathbb{E}[(D(x) - 1)^2] + \mathbb{E}[D(G(z))^2]
$$

**特征匹配损失**：
$$
\mathcal{L}_{\text{fm}} = \sum_{i=1}^L \frac{1}{N_i} \|D^{(i)}(x) - D^{(i)}(G(z))\|_1
$$

**梅尔重建损失**：
$$
\mathcal{L}_{\text{mel}} = \|\text{Mel}(x) - \text{Mel}(G(z))\|_1
$$

**速度**：实时生成（1秒音频 < 10ms）！

### 9.4 端到端模型（VALL-E）

**创新**：跳过梅尔频谱，直接生成离散token！

**两阶段**：

**1. 音频编解码器（EnCodec）**：

将波形压缩为离散codes：
$$
\text{Waveform} \xrightarrow{\text{Encoder}} z \xrightarrow{\text{VQ}} c \in \{1, ..., K\}^{T'}
$$

其中$K=1024$, $T' = T / 320$（压缩320倍）

**2. 语言模型生成codes**：

给定文本 $x$ 和参考音频codes $c_{\text{ref}}$：
$$
p(c | x, c_{\text{ref}}) = \prod_{t=1}^{T'} p(c_t | x, c_{\text{ref}}, c_{<t})
$$

用Transformer自回归生成！

**优势**：
- 零样本声音克隆（只需3秒参考音频）
- 端到端训练
- 支持韵律、情感控制

---

## 第10章：多模态对齐机制

### 10.1 CLIP对比学习

**目标**：学习图像-文本对齐的嵌入空间。

**架构**：
- 图像编码器：ViT或ResNet
- 文本编码器：Transformer

**对比损失**：

给定batch $(I_1, T_1), ..., (I_N, T_N)$：

**图像到文本**：
$$
\mathcal{L}_{i2t} = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(\text{sim}(I_i, T_i) / \tau)}{\sum_{j=1}^N \exp(\text{sim}(I_i, T_j) / \tau)}
$$

**文本到图像**：
$$
\mathcal{L}_{t2i} = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(\text{sim}(I_i, T_i) / \tau)}{\sum_{j=1}^N \exp(\text{sim}(I_j, T_i) / \tau)}
$$

**总损失**：
$$
\mathcal{L}_{\text{CLIP}} = \frac{\mathcal{L}_{i2t} + \mathcal{L}_{t2i}}{2}
$$

**相似度函数**：
$$
\text{sim}(I, T) = \frac{f_I(I)^T f_T(T)}{\|f_I(I)\| \cdot \|f_T(T)\|}
$$

**温度参数** $\tau$：控制软化程度，通常 $\tau=0.07$。

### 10.2 视觉-语言Transformer

**交叉模态注意力**：

**图像到文本**：
$$
\text{CrossAttn}(Q_{\text{text}}, K_{\text{image}}, V_{\text{image}})
$$

**文本到图像**：
$$
\text{CrossAttn}(Q_{\text{image}}, K_{\text{text}}, V_{\text{text}})
$$

**Flamingo架构**：

冻结预训练LLM，插入交叉注意力层：
```
LLM Layer 1
  ↓
Cross-Attn (条件在图像特征)
  ↓
LLM Layer 2
  ↓
...
```

### 10.3 对齐质量度量

**CLIP Score**：
$$
\text{CLIPScore}(I, T) = \max(100 \cdot \text{sim}(I, T), 0)
$$

**ImageReward**：

训练奖励模型预测人类偏好：
$$
r(I, T) = \text{RewardModel}(\text{concat}(f_I(I), f_T(T)))
$$

**PickScore**：

基于成对比较的奖励：
$$
r(I_1, T) > r(I_2, T) \Leftrightarrow \text{人类偏好} I_1
$$

---

## 总结

### Part A核心要点

1. **自回归生成**：逐token预测，因果注意力掩码
2. **解码策略**：贪心/Beam Search/采样各有优劣
3. **温度采样**：控制随机性，$T \in [0.1, 1.5]$
4. **Top-p采样**：动态词汇集，主流选择

### Part B核心要点

1. **Diffusion**：前向扩散+反向去噪，训练简单损失
2. **DDIM**：加速采样，1000步→50步
3. **Latent Diffusion**：VAE压缩，48倍加速
4. **ControlNet**：零初始化，精确空间控制

### Part C核心要点

1. **视频生成**：时空patches，分解注意力
2. **TTS**：声学模型+声码器，HiFi-GAN实时生成
3. **CLIP**：对比学习，图文对齐
4. **端到端**：VALL-E跳过中间表示

### 方法对比

| 领域 | 主流方法 | 训练难度 | 生成速度 | 质量 |
|------|---------|---------|---------|------|
| **文本** | Transformer自回归 | 低 | 快 | 高 |
| **图像** | Latent Diffusion | 中 | 中 | 最高 |
| **视频** | 视频Diffusion | 高 | 慢 | 高 |
| **音频** | HiFi-GAN | 中 | 最快 | 高 |

### 未来方向

1. **统一模型**：单一模型生成所有模态
2. **可控性**：更精细的控制机制
3. **效率**：一步生成（Consistency Models）
4. **长视频**：更长时间一致性

---

**参考文献**：
- DDPM: [Denoising Diffusion Probabilistic Models (NeurIPS 2020)](https://arxiv.org/abs/2006.11239)
- DDIM: [Denoising Diffusion Implicit Models (ICLR 2021)](https://arxiv.org/abs/2010.02502)
- Stable Diffusion: [High-Resolution Image Synthesis with Latent Diffusion Models (CVPR 2022)](https://arxiv.org/abs/2112.10752)
- ControlNet: [Adding Conditional Control to Text-to-Image Diffusion Models (ICCV 2023)](https://arxiv.org/abs/2302.05543)
- Sora: [Video generation models as world simulators (OpenAI 2024)](https://openai.com/research/video-generation-models-as-world-simulators)
- HiFi-GAN: [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis (NeurIPS 2020)](https://arxiv.org/abs/2010.05646)
- VALL-E: [Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (arXiv 2023)](https://arxiv.org/abs/2301.02111)
- CLIP: [Learning Transferable Visual Models From Natural Language Supervision (ICML 2021)](https://arxiv.org/abs/2103.00020)

