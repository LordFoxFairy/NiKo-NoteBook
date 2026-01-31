# 第11章 广义线性模型 (Generalized Linear Models)

> "The purpose of models is not to fit the data but to sharpen the questions."
> — Samuel Karlin

---

## 11.1 引言：从线性回归到GLM

在前面的章节中,我们已经学习了**线性回归**和**逻辑回归**两个重要模型：

- **线性回归**：假设 $y \sim \mathcal{N}(\boldsymbol{w}^T\boldsymbol{x}, \sigma^2)$，用于预测连续值
- **逻辑回归**：假设 $y \sim \text{Bernoulli}(\sigma(\boldsymbol{w}^T\boldsymbol{x}))$，用于二分类

这两个看似不同的模型，实际上可以统一在**广义线性模型 (Generalized Linear Model, GLM)** 的框架下。GLM 通过引入**指数族分布**和**链接函数**，为处理各种类型的响应变量（连续、离散、计数等）提供了统一的理论框架。

> **核心思想**：GLM 不直接建模 $E[y|\boldsymbol{x}]$，而是对其进行某种变换后再与线性预测器 $\boldsymbol{w}^T\boldsymbol{x}$ 建立关系。

### 11.1.1 为什么需要GLM？

传统线性回归的局限性：

1. **响应变量类型受限**：只能处理服从正态分布的连续变量
2. **异方差问题**：方差与均值相关时，模型假设被违背
3. **取值范围限制**：无法保证预测值在合理范围内（如概率 $\in [0,1]$，计数 $\in \mathbb{N}$）

GLM 通过以下方式解决这些问题：
- 允许响应变量服从**指数族分布**
- 通过**链接函数**将均值映射到实数域
- 方差可以是均值的函数

---

## 11.2 指数族分布

### 11.2.1 指数族的通用形式

如果随机变量 $y$ 的概率密度（或质量）函数可以写成以下形式，则称 $y$ 服从**指数族分布**：

$$
p(y|\eta) = h(y) \exp\left(\eta T(y) - A(\eta)\right)
$$

其中：
- $\eta$ 称为**自然参数 (natural parameter)** 或**典范参数 (canonical parameter)**
- $T(y)$ 称为**充分统计量 (sufficient statistic)**，通常 $T(y) = y$
- $A(\eta)$ 称为**对数配分函数 (log partition function)**，用于归一化
- $h(y)$ 称为**基础测度 (base measure)**

> **重要性质**：对数配分函数 $A(\eta)$ 是凸函数，其导数和二阶导数分别给出分布的均值和方差。

### 11.2.2 指数族的核心性质

通过对概率密度函数积分，可以推导出：

$$
\int h(y) \exp\left(\eta T(y) - A(\eta)\right) dy = 1
$$

对 $\eta$ 求导：

$$
\frac{\partial}{\partial \eta} \int h(y) \exp\left(\eta T(y) - A(\eta)\right) dy = 0
$$

$$
\int h(y) \left(T(y) - A'(\eta)\right) \exp\left(\eta T(y) - A(\eta)\right) dy = 0
$$

$$
\mathbb{E}[T(y)] = A'(\eta)
$$

> **性质1**：均值由对数配分函数的一阶导数给出：$\mu = \mathbb{E}[y] = A'(\eta)$

继续对 $\eta$ 求二阶导数：

$$
\frac{\partial^2 A(\eta)}{\partial \eta^2} = \text{Var}(T(y))
$$

> **性质2**：方差由对数配分函数的二阶导数给出：$\text{Var}(y) = A''(\eta)$

这两个性质在 GLM 的推导中起着核心作用。

### 11.2.3 常见分布属于指数族

#### (1) 高斯分布

$$
p(y|\mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y-\mu)^2}{2\sigma^2}\right)
$$

展开：

$$
= \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{y^2}{2\sigma^2} + \frac{\mu y}{\sigma^2} - \frac{\mu^2}{2\sigma^2}\right)
$$

对比指数族形式（假设 $\sigma^2$ 已知）：

$$
\begin{cases}
\eta = \frac{\mu}{\sigma^2} \\
T(y) = y \\
A(\eta) = \frac{\mu^2}{2\sigma^2} = \frac{\sigma^2 \eta^2}{2} \\
h(y) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{y^2}{2\sigma^2}\right)
\end{cases}
$$

验证性质：

$$
\mathbb{E}[y] = A'(\eta) = \sigma^2 \eta = \mu \quad \checkmark
$$

$$
\text{Var}(y) = A''(\eta) = \sigma^2 \quad \checkmark
$$

#### (2) 伯努利分布

$$
p(y|\phi) = \phi^y (1-\phi)^{1-y}, \quad y \in \{0, 1\}
$$

取对数并重排：

$$
\log p(y|\phi) = y \log \phi + (1-y) \log(1-\phi)
$$

$$
= y \log \frac{\phi}{1-\phi} + \log(1-\phi)
$$

$$
= y \eta - \log(1 + e^\eta)
$$

其中 $\eta = \log \frac{\phi}{1-\phi}$ 是 **logit 函数**，反解得 $\phi = \frac{1}{1+e^{-\eta}} = \sigma(\eta)$。

对比指数族形式：

$$
\begin{cases}
\eta = \log \frac{\phi}{1-\phi} \\
T(y) = y \\
A(\eta) = \log(1 + e^\eta) \\
h(y) = 1
\end{cases}
$$

验证性质：

$$
\mathbb{E}[y] = A'(\eta) = \frac{e^\eta}{1+e^\eta} = \sigma(\eta) = \phi \quad \checkmark
$$

$$
\text{Var}(y) = A''(\eta) = \frac{e^\eta}{(1+e^\eta)^2} = \phi(1-\phi) \quad \checkmark
$$

#### (3) 泊松分布

$$
p(y|\lambda) = \frac{\lambda^y e^{-\lambda}}{y!}, \quad y \in \{0, 1, 2, \ldots\}
$$

取对数：

$$
\log p(y|\lambda) = y \log \lambda - \lambda - \log(y!)
$$

对比指数族形式：

$$
\begin{cases}
\eta = \log \lambda \\
T(y) = y \\
A(\eta) = e^\eta = \lambda \\
h(y) = \frac{1}{y!}
\end{cases}
$$

验证性质：

$$
\mathbb{E}[y] = A'(\eta) = e^\eta = \lambda \quad \checkmark
$$

$$
\text{Var}(y) = A''(\eta) = e^\eta = \lambda \quad \checkmark
$$

> **小结**：高斯、伯努利、泊松分布都可以写成指数族形式，这为 GLM 提供了理论基础。

---

## 11.3 GLM的三要素

广义线性模型由以下三个要素定义：

### 11.3.1 随机成分 (Random Component)

响应变量 $y$ 服从**指数族分布**：

$$
p(y|\theta, \phi) = \exp\left(\frac{y\theta - b(\theta)}{a(\phi)} + c(y, \phi)\right)
$$

其中：
- $\theta$ 是自然参数
- $\phi$ 是离散参数（dispersion parameter），通常已知
- $a(\phi) = \phi / w$，$w$ 是权重

> 注意：这里的参数化形式与 11.2.1 略有不同，但本质相同。

### 11.3.2 系统成分 (Systematic Component)

线性预测器：

$$
\eta = \boldsymbol{w}^T \boldsymbol{x} = w_0 + w_1 x_1 + \cdots + w_d x_d
$$

这是特征的线性组合，保持了线性模型的简单性。

### 11.3.3 链接函数 (Link Function)

链接函数 $g(\cdot)$ 将期望 $\mu = \mathbb{E}[y]$ 与线性预测器 $\eta$ 联系起来：

$$
g(\mu) = \eta = \boldsymbol{w}^T \boldsymbol{x}
$$

等价地，反链接函数 $g^{-1}$ 为：

$$
\mu = g^{-1}(\eta) = g^{-1}(\boldsymbol{w}^T \boldsymbol{x})
$$

> **典范链接函数**：当 $g(\mu) = \theta$（自然参数），称为**典范链接 (canonical link)**。此时 $\eta = \theta$，数学性质最优美。

#### 常见链接函数

| 分布       | 均值 $\mu$    | 典范链接 $g(\mu)$ | 反链接 $g^{-1}(\eta)$ | 应用场景   |
|----------|-------------|---------------|-------------------|--------|
| 高斯分布     | $\mu$       | $\mu$ (恒等)    | $\eta$            | 线性回归   |
| 伯努利分布    | $\phi$      | $\log\frac{\phi}{1-\phi}$ | $\frac{1}{1+e^{-\eta}}$ | 逻辑回归   |
| 泊松分布     | $\lambda$   | $\log \lambda$ | $e^\eta$          | 计数回归   |
| 伽马分布     | $\mu$       | $\mu^{-1}$    | $\eta^{-1}$       | 持续时间建模 |

---

## 11.4 GLM的参数估计

### 11.4.1 极大似然估计

给定训练集 $\mathcal{D} = \{(\boldsymbol{x}_i, y_i)\}_{i=1}^n$，似然函数为：

$$
L(\boldsymbol{w}) = \prod_{i=1}^n p(y_i | \boldsymbol{x}_i, \boldsymbol{w})
$$

对数似然：

$$
\ell(\boldsymbol{w}) = \sum_{i=1}^n \log p(y_i | \boldsymbol{x}_i, \boldsymbol{w})
$$

对于指数族分布：

$$
\ell(\boldsymbol{w}) = \sum_{i=1}^n \left[\frac{y_i \theta_i - b(\theta_i)}{a(\phi)} + c(y_i, \phi)\right]
$$

其中 $\theta_i = g(\mu_i)$，$\mu_i = g^{-1}(\boldsymbol{w}^T \boldsymbol{x}_i)$。

### 11.4.2 梯度计算

对 $w_j$ 求偏导：

$$
\frac{\partial \ell}{\partial w_j} = \sum_{i=1}^n \frac{\partial \ell_i}{\partial \theta_i} \frac{\partial \theta_i}{\partial \mu_i} \frac{\partial \mu_i}{\partial \eta_i} \frac{\partial \eta_i}{\partial w_j}
$$

利用指数族性质：

$$
\frac{\partial \ell_i}{\partial \theta_i} = \frac{y_i - b'(\theta_i)}{a(\phi)} = \frac{y_i - \mu_i}{a(\phi)}
$$

因为 $\mu_i = b'(\theta_i)$（指数族性质）。

对于典范链接，$\theta_i = \eta_i = \boldsymbol{w}^T \boldsymbol{x}_i$，则：

$$
\frac{\partial \theta_i}{\partial \mu_i} \frac{\partial \mu_i}{\partial \eta_i} = 1
$$

$$
\frac{\partial \eta_i}{\partial w_j} = x_{ij}
$$

因此梯度简化为：

$$
\frac{\partial \ell}{\partial w_j} = \frac{1}{a(\phi)} \sum_{i=1}^n (y_i - \mu_i) x_{ij}
$$

> **关键观察**：梯度形式与线性回归完全一致！这是典范链接的优美之处。

### 11.4.3 迭代加权最小二乘 (IRLS)

对于非典范链接或更一般的情况，使用 **Fisher Scoring** 或 **Newton-Raphson** 迭代求解。

定义**工作响应变量 (working response)**：

$$
z_i = \eta_i + (y_i - \mu_i) \frac{\partial \eta_i}{\partial \mu_i}
$$

其中 $\frac{\partial \eta_i}{\partial \mu_i} = \frac{1}{g'(\mu_i)}$ 是链接函数的导数的倒数。

定义**迭代权重 (iterative weights)**：

$$
w_i = \frac{1}{\text{Var}(y_i)} \left(\frac{\partial \mu_i}{\partial \eta_i}\right)^2 = \frac{(g'(\mu_i))^2}{\text{Var}(y_i)}
$$

**IRLS 算法**：

1. 初始化 $\boldsymbol{w}^{(0)}$，计算 $\eta_i^{(0)} = \boldsymbol{w}^{(0)T} \boldsymbol{x}_i$
2. 重复直到收敛：
   - 计算 $\mu_i = g^{-1}(\eta_i)$
   - 计算工作响应 $z_i$ 和权重 $w_i$
   - 加权最小二乘更新：
   $$
   \boldsymbol{w}^{(t+1)} = (\boldsymbol{X}^T \boldsymbol{W} \boldsymbol{X})^{-1} \boldsymbol{X}^T \boldsymbol{W} \boldsymbol{z}
   $$
   其中 $\boldsymbol{W} = \text{diag}(w_1, \ldots, w_n)$

> **物理意义**：IRLS 在每次迭代中对样本重新加权，使得方差较大的样本权重降低，提高估计的效率。

---

## 11.5 案例分析：统一框架下的三大模型

### 11.5.1 线性回归

**设定**：
- 分布：$y \sim \mathcal{N}(\mu, \sigma^2)$
- 链接函数：恒等链接 $g(\mu) = \mu$
- 线性预测器：$\eta = \boldsymbol{w}^T \boldsymbol{x}$

因此：

$$
\mu = \boldsymbol{w}^T \boldsymbol{x}
$$

负对数似然：

$$
-\ell(\boldsymbol{w}) = \frac{1}{2\sigma^2} \sum_{i=1}^n (y_i - \boldsymbol{w}^T \boldsymbol{x}_i)^2 + \text{const}
$$

这正是最小二乘法！

> **GLM 视角**：线性回归是 GLM 在高斯分布 + 恒等链接下的特例。

### 11.5.2 逻辑回归

**设定**：
- 分布：$y \sim \text{Bernoulli}(\phi)$
- 链接函数：logit 链接 $g(\phi) = \log \frac{\phi}{1-\phi}$
- 线性预测器：$\eta = \boldsymbol{w}^T \boldsymbol{x}$

因此：

$$
\phi = \frac{1}{1 + e^{-\boldsymbol{w}^T \boldsymbol{x}}}
$$

对数似然：

$$
\ell(\boldsymbol{w}) = \sum_{i=1}^n \left[y_i \boldsymbol{w}^T \boldsymbol{x}_i - \log(1 + e^{\boldsymbol{w}^T \boldsymbol{x}_i})\right]
$$

> **GLM 视角**：逻辑回归是 GLM 在伯努利分布 + logit 链接下的特例。

### 11.5.3 泊松回归

**设定**：
- 分布：$y \sim \text{Poisson}(\lambda)$，用于建模计数数据
- 链接函数：对数链接 $g(\lambda) = \log \lambda$
- 线性预测器：$\eta = \boldsymbol{w}^T \boldsymbol{x}$

因此：

$$
\lambda = e^{\boldsymbol{w}^T \boldsymbol{x}}
$$

对数似然：

$$
\ell(\boldsymbol{w}) = \sum_{i=1}^n \left[y_i \boldsymbol{w}^T \boldsymbol{x}_i - e^{\boldsymbol{w}^T \boldsymbol{x}_i} - \log(y_i!)\right]
$$

梯度：

$$
\frac{\partial \ell}{\partial \boldsymbol{w}} = \sum_{i=1}^n (y_i - \lambda_i) \boldsymbol{x}_i
$$

**应用场景**：
- 网站访问次数预测
- 交通事故数量建模
- 基因表达计数分析

> **GLM 视角**：泊松回归是 GLM 在泊松分布 + 对数链接下的特例。

### 11.5.4 统一框架的威力

```svg
<svg viewBox="0 0 800 400" xmlns="http://www.w3.org/2000/svg">
  <!-- 标题 -->
  <text x="400" y="30" font-size="20" font-weight="bold" text-anchor="middle">GLM统一框架</text>

  <!-- GLM核心 -->
  <rect x="300" y="60" width="200" height="80" fill="#E8F4F8" stroke="#4A90E2" stroke-width="2"/>
  <text x="400" y="90" font-size="16" font-weight="bold" text-anchor="middle">广义线性模型</text>
  <text x="400" y="110" font-size="12" text-anchor="middle">指数族分布</text>
  <text x="400" y="125" font-size="12" text-anchor="middle">链接函数 g(μ)</text>

  <!-- 线性回归 -->
  <rect x="50" y="200" width="180" height="150" fill="#F0F8FF" stroke="#4682B4" stroke-width="2"/>
  <text x="140" y="225" font-size="14" font-weight="bold" text-anchor="middle">线性回归</text>
  <text x="140" y="245" font-size="11" text-anchor="middle">分布: 高斯</text>
  <text x="140" y="265" font-size="11" text-anchor="middle">链接: g(μ) = μ</text>
  <text x="140" y="285" font-size="11" text-anchor="middle">应用: 连续值预测</text>
  <text x="140" y="310" font-size="10" fill="#666" text-anchor="middle">房价预测</text>
  <text x="140" y="330" font-size="10" fill="#666" text-anchor="middle">股票收益</text>

  <!-- 逻辑回归 -->
  <rect x="310" y="200" width="180" height="150" fill="#FFF8F0" stroke="#E2A14A" stroke-width="2"/>
  <text x="400" y="225" font-size="14" font-weight="bold" text-anchor="middle">逻辑回归</text>
  <text x="400" y="245" font-size="11" text-anchor="middle">分布: 伯努利</text>
  <text x="400" y="265" font-size="11" text-anchor="middle">链接: g(φ) = logit(φ)</text>
  <text x="400" y="285" font-size="11" text-anchor="middle">应用: 二分类</text>
  <text x="400" y="310" font-size="10" fill="#666" text-anchor="middle">垃圾邮件检测</text>
  <text x="400" y="330" font-size="10" fill="#666" text-anchor="middle">疾病诊断</text>

  <!-- 泊松回归 -->
  <rect x="570" y="200" width="180" height="150" fill="#F0FFF0" stroke="#4AE27A" stroke-width="2"/>
  <text x="660" y="225" font-size="14" font-weight="bold" text-anchor="middle">泊松回归</text>
  <text x="660" y="245" font-size="11" text-anchor="middle">分布: 泊松</text>
  <text x="660" y="265" font-size="11" text-anchor="middle">链接: g(λ) = log(λ)</text>
  <text x="660" y="285" font-size="11" text-anchor="middle">应用: 计数预测</text>
  <text x="660" y="310" font-size="10" fill="#666" text-anchor="middle">网站访问量</text>
  <text x="660" y="330" font-size="10" fill="#666" text-anchor="middle">事故数量</text>

  <!-- 连接线 -->
  <path d="M 350 140 L 140 200" stroke="#4682B4" stroke-width="2" fill="none" marker-end="url(#arrowblue)"/>
  <path d="M 400 140 L 400 200" stroke="#E2A14A" stroke-width="2" fill="none" marker-end="url(#arroworange)"/>
  <path d="M 450 140 L 660 200" stroke="#4AE27A" stroke-width="2" fill="none" marker-end="url(#arrowgreen)"/>

  <!-- 箭头定义 -->
  <defs>
    <marker id="arrowblue" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="#4682B4"/>
    </marker>
    <marker id="arroworange" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="#E2A14A"/>
    </marker>
    <marker id="arrowgreen" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="#4AE27A"/>
    </marker>
  </defs>
</svg>
```

---

## 11.6 模型诊断与评估

### 11.6.1 偏差 (Deviance)

偏差是衡量模型拟合优度的统计量：

$$
D = 2[\ell(\text{saturated}) - \ell(\text{fitted})]
$$

其中：
- $\ell(\text{saturated})$：饱和模型的对数似然（每个观测有独立参数）
- $\ell(\text{fitted})$：拟合模型的对数似然

对于高斯分布：

$$
D = \sum_{i=1}^n (y_i - \hat{\mu}_i)^2
$$

对于伯努利分布：

$$
D = 2\sum_{i=1}^n \left[y_i \log\frac{y_i}{\hat{\mu}_i} + (1-y_i)\log\frac{1-y_i}{1-\hat{\mu}_i}\right]
$$

（约定：当 $y_i = 0$ 时，$y_i \log \frac{y_i}{\hat{\mu}_i} = 0$；当 $y_i = 1$ 时，$(1-y_i)\log\frac{1-y_i}{1-\hat{\mu}_i} = 0$）

> **性质**：偏差越小，模型拟合越好。在嵌套模型中，偏差差值近似服从 $\chi^2$ 分布。

### 11.6.2 Pearson 残差

标准化残差：

$$
r_i^P = \frac{y_i - \hat{\mu}_i}{\sqrt{\text{Var}(\hat{\mu}_i)}}
$$

用于检测异常值和模型假设。

### 11.6.3 AIC 与 BIC

**Akaike 信息准则**：

$$
\text{AIC} = -2\ell(\hat{\boldsymbol{w}}) + 2p
$$

**Bayesian 信息准则**：

$$
\text{BIC} = -2\ell(\hat{\boldsymbol{w}}) + p \log n
$$

其中 $p$ 是参数个数，$n$ 是样本数。

> **模型选择**：AIC/BIC 越小越好，平衡拟合优度与模型复杂度。

---

## 11.7 GLM的扩展

### 11.7.1 准似然 (Quasi-likelihood)

当分布族未知但均值-方差关系已知时，可使用**准似然方法**：

$$
Q(\mu; y) = \int_y^\mu \frac{y - t}{V(t)} dt
$$

其中 $V(\mu)$ 是方差函数。

优点：
- 无需完全指定分布
- 只需均值和方差关系
- 对分布误设定具有鲁棒性

### 11.7.2 零膨胀模型 (Zero-Inflated Models)

对于计数数据中零值过多的情况（如保险索赔次数），使用零膨胀泊松或零膨胀负二项模型：

$$
P(Y = y) = \begin{cases}
\pi + (1-\pi)e^{-\lambda}, & y = 0 \\
(1-\pi)\frac{\lambda^y e^{-\lambda}}{y!}, & y > 0
\end{cases}
$$

其中 $\pi$ 是结构零的概率。

### 11.7.3 广义加性模型 (GAM)

将线性预测器扩展为光滑函数的和：

$$
g(\mu) = \beta_0 + f_1(x_1) + f_2(x_2) + \cdots + f_p(x_p)
$$

其中 $f_j$ 是光滑函数（如样条），可捕捉非线性关系。

---

## 11.8 实践案例：保险索赔建模

### 11.8.1 问题背景

某保险公司希望根据投保人特征（年龄、性别、车型等）预测年度索赔次数。

**数据特点**：
- 响应变量：索赔次数（非负整数）
- 特征：年龄、性别、驾龄、车型、地区
- 挑战：零值较多（大部分人无索赔）

### 11.8.2 模型选择

使用**泊松回归**：

$$
Y_i \sim \text{Poisson}(\lambda_i)
$$

$$
\log \lambda_i = w_0 + w_1 \cdot \text{age}_i + w_2 \cdot \text{gender}_i + \cdots
$$

### 11.8.3 模型拟合

伪代码示例：

```python
import statsmodels.api as sm

# 拟合泊松回归
model = sm.GLM(y, X, family=sm.families.Poisson(link=sm.families.links.Log()))
result = model.fit()

# 查看系数
print(result.summary())

# 预测
lambda_pred = result.predict(X_test)
```

### 11.8.4 模型诊断

检查偏差残差分布：

```python
residuals = result.resid_deviance
plt.hist(residuals, bins=30)
plt.xlabel('Deviance Residuals')
plt.ylabel('Frequency')
```

如果发现**过离散 (overdispersion)**（方差 > 均值），可改用负二项回归。

---

## 11.9 理论深化：GLM与指数族的深层联系

### 11.9.1 充分统计量与信息几何

在指数族分布中，充分统计量 $T(y)$ 包含了关于参数 $\eta$ 的所有信息。从信息几何角度，$\eta$ 和 $\mathbb{E}[T(y)]$ 构成**对偶坐标系**。

对数配分函数 $A(\eta)$ 是 **Legendre 变换**的核心：

$$
A^*(\mu) = \sup_\eta \{\eta \mu - A(\eta)\}
$$

这在统计物理和信息论中有深刻应用。

### 11.9.2 Fisher 信息矩阵

对于 GLM，Fisher 信息矩阵为：

$$
\mathcal{I}(\boldsymbol{w}) = \boldsymbol{X}^T \boldsymbol{W} \boldsymbol{X}
$$

其中 $\boldsymbol{W} = \text{diag}(w_1, \ldots, w_n)$，

$$
w_i = \frac{1}{a(\phi) \cdot \text{Var}(y_i)} \left(\frac{\partial \mu_i}{\partial \eta_i}\right)^2
$$

当 $a(\phi) = 1$ 时（如伯努利、泊松分布），这与 IRLS 算法中的权重定义一致。

> **Cramér-Rao 下界**：参数估计的方差下界为 $\mathcal{I}^{-1}(\boldsymbol{w})$，在正则条件下 MLE 是渐近有效的。

### 11.9.3 GLM的渐近性质

在正则条件下：

$$
\sqrt{n}(\hat{\boldsymbol{w}} - \boldsymbol{w}_0) \xrightarrow{d} \mathcal{N}(0, \mathcal{I}^{-1}(\boldsymbol{w}_0))
$$

这为构造置信区间和假设检验提供了理论基础。

---

## 11.10 总结与展望

### 11.10.1 核心要点回顾

> **GLM的三大支柱**：
> 1. **指数族分布**：统一处理各种响应变量类型
> 2. **链接函数**：灵活建模均值与线性预测器的关系
> 3. **极大似然估计**：提供一致、渐近正态的参数估计

GLM 将线性回归、逻辑回归、泊松回归等模型统一在同一理论框架下，具有以下优势：

- **理论优美**：基于指数族分布的深刻性质
- **计算高效**：IRLS 算法快速收敛
- **解释性强**：保留线性模型的可解释性
- **扩展性好**：可轻松扩展到新的分布族

### 11.10.2 GLM与其他模型的关系

```svg
<svg viewBox="0 0 800 500" xmlns="http://www.w3.org/2000/svg">
  <!-- 标题 -->
  <text x="400" y="30" font-size="20" font-weight="bold" text-anchor="middle">GLM在统计学习中的位置</text>

  <!-- 线性模型 -->
  <rect x="50" y="80" width="200" height="80" fill="#E8F4F8" stroke="#4A90E2" stroke-width="2"/>
  <text x="150" y="110" font-size="14" font-weight="bold" text-anchor="middle">线性模型</text>
  <text x="150" y="130" font-size="11" text-anchor="middle">最简单</text>
  <text x="150" y="145" font-size="11" text-anchor="middle">解释性强</text>

  <!-- GLM -->
  <rect x="300" y="80" width="200" height="80" fill="#FFF8F0" stroke="#E2A14A" stroke-width="2"/>
  <text x="400" y="110" font-size="14" font-weight="bold" text-anchor="middle">GLM</text>
  <text x="400" y="130" font-size="11" text-anchor="middle">指数族+链接函数</text>
  <text x="400" y="145" font-size="11" text-anchor="middle">平衡性能与解释</text>

  <!-- GAM -->
  <rect x="550" y="80" width="200" height="80" fill="#F0FFF0" stroke="#4AE27A" stroke-width="2"/>
  <text x="650" y="110" font-size="14" font-weight="bold" text-anchor="middle">GAM</text>
  <text x="650" y="130" font-size="11" text-anchor="middle">光滑函数</text>
  <text x="650" y="145" font-size="11" text-anchor="middle">捕捉非线性</text>

  <!-- 深度学习 -->
  <rect x="300" y="220" width="200" height="80" fill="#FFF0F5" stroke="#E24A90" stroke-width="2"/>
  <text x="400" y="250" font-size="14" font-weight="bold" text-anchor="middle">深度学习</text>
  <text x="400" y="270" font-size="11" text-anchor="middle">高度非线性</text>
  <text x="400" y="285" font-size="11" text-anchor="middle">黑箱模型</text>

  <!-- 箭头和标注 -->
  <path d="M 250 120 L 300 120" stroke="#666" stroke-width="2" marker-end="url(#arrow1)"/>
  <text x="275" y="110" font-size="10" text-anchor="middle">扩展</text>

  <path d="M 500 120 L 550 120" stroke="#666" stroke-width="2" marker-end="url(#arrow1)"/>
  <text x="525" y="110" font-size="10" text-anchor="middle">进一步</text>

  <path d="M 400 160 L 400 220" stroke="#666" stroke-width="2" marker-end="url(#arrow1)"/>
  <text x="440" y="195" font-size="10" text-anchor="middle">灵活性增加</text>

  <!-- 坐标轴 -->
  <path d="M 100 380 L 700 380" stroke="#333" stroke-width="2" marker-end="url(#arrow2)"/>
  <text x="720" y="385" font-size="12" text-anchor="middle">模型复杂度</text>

  <path d="M 100 380 L 100 330" stroke="#333" stroke-width="2" marker-end="url(#arrow2)"/>
  <text x="70" y="320" font-size="12" text-anchor="middle">解释性</text>

  <!-- 位置标记 -->
  <circle cx="150" cy="370" r="6" fill="#4A90E2"/>
  <text x="150" y="395" font-size="10" text-anchor="middle">线性</text>

  <circle cx="400" cy="355" r="6" fill="#E2A14A"/>
  <text x="400" y="410" font-size="10" text-anchor="middle">GLM</text>

  <circle cx="550" cy="345" r="6" fill="#4AE27A"/>
  <text x="550" y="425" font-size="10" text-anchor="middle">GAM</text>

  <circle cx="650" cy="340" r="6" fill="#E24A90"/>
  <text x="650" y="440" font-size="10" text-anchor="middle">深度学习</text>

  <!-- 箭头标记定义 -->
  <defs>
    <marker id="arrow1" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#666"/>
    </marker>
    <marker id="arrow2" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#333"/>
    </marker>
  </defs>
</svg>
```

### 11.10.3 进一步学习方向

1. **广义估计方程 (GEE)**：处理相关数据（纵向数据、聚类数据）
2. **混合效应模型 (GLMM)**：引入随机效应，建模层级结构
3. **贝叶斯GLM**：通过先验分布进行正则化和不确定性量化
4. **分位数回归**：建模条件分位数而非条件均值
5. **生存分析**：Cox 比例风险模型可视为特殊的 GLM

### 11.10.4 哲学思考

> "所有模型都是错的，但有些是有用的。" — George Box

GLM 不是万能的，但它提供了一个**坚实的起点**：

- 当数据符合指数族假设时，GLM 提供最优解
- 当假设被违背时，GLM 仍提供合理的近似
- GLM 的简洁性使其成为更复杂模型的基准

在机器学习追求黑箱性能的时代，GLM 提醒我们：**可解释性和理论基础依然重要**。

---

## 参考文献

1. McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman and Hall.
2. Dobson, A. J., & Barnett, A. G. (2018). *An Introduction to Generalized Linear Models* (4th ed.). CRC Press.
3. Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R* (2nd ed.). Chapman and Hall/CRC.
4. Hastie, T., & Tibshirani, R. (1990). *Generalized Additive Models*. Chapman and Hall.
5. Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press.

---

## 附录：Python实现示例

### A.1 使用 statsmodels 拟合GLM

```python
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import glm

# 生成模拟数据
np.random.seed(42)
n = 1000
X = np.random.randn(n, 3)
X = sm.add_constant(X)  # 添加截距项

# 泊松回归
lambda_true = np.exp(X @ [0.5, 1, -0.5, 0.3])
y_count = np.random.poisson(lambda_true)

# 拟合模型
poisson_model = sm.GLM(y_count, X, family=sm.families.Poisson())
poisson_result = poisson_model.fit()
print(poisson_result.summary())

# 逻辑回归
prob_true = 1 / (1 + np.exp(-X @ [0, 1, -1, 0.5]))
y_binary = np.random.binomial(1, prob_true)

logit_model = sm.GLM(y_binary, X, family=sm.families.Binomial())
logit_result = logit_model.fit()
print(logit_result.summary())
```

### A.2 手动实现IRLS算法

```python
def irls_glm(X, y, family='gaussian', max_iter=25, tol=1e-8):
    """
    IRLS算法实现GLM

    family: 'gaussian', 'binomial', 'poisson'
    """
    n, p = X.shape
    w = np.zeros(p)

    for iteration in range(max_iter):
        # 线性预测
        eta = X @ w

        # 均值函数（反链接）和导数
        if family == 'gaussian':
            mu = eta  # g^(-1)(eta) = eta
            mu_prime = np.ones_like(eta)  # dμ/dη = 1
            var = np.ones_like(eta)  # Var(y) = σ²（假设为1）
        elif family == 'binomial':
            mu = 1 / (1 + np.exp(-eta))  # g^(-1)(eta) = sigmoid(eta)
            mu_prime = mu * (1 - mu)  # dμ/dη = μ(1-μ)
            var = mu * (1 - mu)  # Var(y) = μ(1-μ)
        elif family == 'poisson':
            mu = np.exp(eta)  # g^(-1)(eta) = exp(eta)
            mu_prime = mu  # dμ/dη = μ
            var = mu  # Var(y) = μ

        # 工作响应和权重
        z = eta + (y - mu) / mu_prime  # 工作响应变量
        weights = mu_prime**2 / var  # w_i = (dμ/dη)² / Var(y)

        # 加权最小二乘
        W = np.diag(weights)
        w_new = np.linalg.solve(X.T @ W @ X, X.T @ W @ z)

        # 检查收敛
        if np.linalg.norm(w_new - w) < tol:
            print(f"Converged in {iteration + 1} iterations")
            break

        w = w_new

    return w

# 测试
w_estimated = irls_glm(X, y_count, family='poisson')
print("Estimated coefficients:", w_estimated)
```

---

> **本章完**
> 下一章将探讨**支持向量机 (SVM)**，从几何角度理解最大间隔分类器。

---

**练习题**

1. 证明伽马分布 $\text{Gamma}(\alpha, \beta)$ 属于指数族，并写出其自然参数。
2. 推导泊松回归的 Fisher 信息矩阵。
3. 在逻辑回归中，为什么不使用恒等链接而使用 logit 链接？
4. 实现一个支持正则化（L1/L2）的 GLM 类。
5. 对比零膨胀泊松模型和负二项模型在处理计数数据时的优劣。
