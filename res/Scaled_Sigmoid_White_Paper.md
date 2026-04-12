# Scaled Sigmoid 激活函数特性研究

---

## 摘要

Scaled Sigmoid 最早在 YOLO v4 中被引入，用于解决 Sigmoid 在输出边界值时产生的权重爆炸问题。本研究的核心问题是：**既然 Scaled Sigmoid 既能避免权重爆炸（Weight Explosion），又能缓解梯度消失（Gradient Vanishing），为什么它没有全面取代标准 Sigmoid？** 我们通过设计多组架構簡單的测试用例，并在 CNN 和 LSTM 两种网络架构上进行实验，探究 Scaled Sigmoid 的适用条件与局限性。

---

## 1. 背景：Scaled Sigmoid 的由来

在 YOLO v4 中，模型将图像切分为多个网格（grid），并预测每个网格中物体中心点的相对位置。中心点位置用 Sigmoid 映射到 [0, 1] 区间，表示在网格内的偏移量（offset 0 到 offset 1）。

**问题出现在：** 当物体的中心点非常靠近网格边界时，Sigmoid 需要输出接近 1 的值。而标准 Sigmoid 要输出 1，其输入必须趋向无穷大，这意味着网络的 weight 和 bias 必须非常大，导致 **weight 和 bias explosion**。

**Scaled Sigmoid 的解法：** 对 Sigmoid 进行缩放后，输出达到 1 不再需要极端的输入值。例如原始 Sigmoid 输入需要趋向无穷才能输出 1，而 Scale 之后输入约 3.7 即可达到，从根本上避免了权重爆炸。

> **[图1]** 标准 Sigmoid 与 Scaled Sigmoid 的对比曲线
> *(请插入实验中的 Sigmoid 对比图)*

---

## 2. Scaled Sigmoid 的理论优势

### 2.1 避免权重爆炸（Weight Explosion）

在标准 Sigmoid 下，假设有多个数据点，且目标输出为 1。由于这些点在 Sigmoid 曲线上的输出都小于 1，所有数据点的梯度方向一致——都要求 weight 变大。这种"一致同意变大"的情况导致权重爆炸。

而在 Scaled Sigmoid 下，由于缩放后曲线超越了 1，部分数据点的输出已经大于目标值 1，它们的梯度方向是要求 weight **变小**。这样就形成了梯度的对冲——有的要变大，有的要变小——权重自然被约束在有梯度的区域内，不会爆炸。

### 2.2 避免梯度消失（Gradient Vanishing）

标准 Sigmoid 在输出接近 0 或 1 的饱和区梯度接近零。Scaled Sigmoid 由于将参数约束在曲线中段（仍有梯度的区域），不会跑到饱和区，因此避免了梯度消失。

### 2.3 更快的收敛速度

在相同的初始化点上，Scaled Sigmoid 的曲线更陡，对应位置的梯度更大，因此参数更新步幅更大，收敛更快。Scale 因子越大，对应位置梯度越大，收敛越快。

> **[图2]** 不同 Scale 因子下从相同初始点出发的收敛速度对比
> *(请插入 recovery 实验结果图)*

---

## 3. 核心问题：为什么 Scaled Sigmoid 没有取代标准 Sigmoid？

既然 Scaled Sigmoid 有上述优势，为什么没有被广泛采用？这是本研究的出发点。我们通过以下实验来回答这个问题。

---

## 4. 实验设计与结果

### 4.1 基础测试用例（Test Cases）

我们设计了多组测试用例（Test Case 0 ~ 4）来验证 Scaled Sigmoid 的特性。

**浮点精度的限制：** 在实验中发现，当 Sigmoid 输入达到约 40 时，浮点运算已将输出截断为 1.0（理论值应为无限接近 1 但不等于 1）。这使得 Test Case 0-3 难以充分展示梯度爆炸现象。

**Test Case 4 是最有效的展示：** 该用例要求网络输出 XOR 模式，为了拟合这种非线性目标，网络必须大幅扭曲输出，weight 持续正向增长、bias 持续负向增长，两者都无限增大。

- **标准 Sigmoid：** weight 和 bias 持续爆炸增长，无法收敛
- **Scaled Sigmoid：** 参数稳定在可接受范围内

> **[图3]** Test Case 4：标准 Sigmoid vs Scaled Sigmoid 的 weight 变化对比
> *(请插入 test case 4 实验结果图)*

### 4.2 CNN 实验

我们在 CNN 上直接将标准 Sigmoid 替换为 Scaled Sigmoid 进行对比实验。

**关于 Loss 的说明：** 由于 Scale 因子改变了损失函数的计算尺度（类似于将 MSE 乘以一个常数），不同 Scale 下的 Loss 绝对值无法直接比较。我们只需确认 Loss 呈下降趋势。

**实验结果：**
- **Accuracy：** Scaled Sigmoid 始终优于标准 Sigmoid（曲线位于更高位置）
- **Weight：** Scaled Sigmoid 的 weight 值更小，且下降更快

> **[图4]** CNN 实验：Accuracy 对比
> *(请插入 CNN accuracy 实验结果图)*

> **[图5]** CNN 实验：Weight 变化对比
> *(请插入 CNN weight 实验结果图)*

**CNN 结论：** 直接替换标准 Sigmoid 为 Scaled Sigmoid，准确率确实更好，验证了 Scaled Sigmoid 在 CNN 中的有效性。原因是 Scaled Sigmoid 的 梯度較大 因此較快收斂，且不會遇到梯度消失的問題。

### 4.3 LSTM 实验

LSTM 中 Sigmoid 用作**门控函数（Gate）**，其设计目的是控制信息的保留与遗忘比例：
- 输出 0 = 完全遗忘
- 输出 1 = 完全保留

但在实际运行中，LSTM 的门控值几乎不会到达 0 或 1 这种极端值。比如一个 forget gate，它通常输出 0.5 ~ 0.8 之类的中间值——保留一些、遗忘一些。这意味着 Sigmoid 的输入值始终在中间区域（例如输出 0.8 对应输入约 2），该区域标准 Sigmoid 本身就有充足的梯度。

**LSTM 结论：** Scaled Sigmoid 对 LSTM 几乎没有帮助。因为 Scaled Sigmoid 要解决的核心问题（边界值输出导致的权重爆炸）在 LSTM 的门控机制中根本不存在。LSTM 的设计理念与 Scaled Sigmoid 的适用场景不匹配。

> **[图6]** LSTM 实验结果
> *(请插入 LSTM 实验结果图)*

---

## 5. Scaled Sigmoid 的缺点：对学习率更敏感

### 5.1 Eject 问题

由于 Scaled Sigmoid 在同一位置梯度更大，当学习率（Learning Rate）设定过大时，参数更新步幅过大，可能直接跳出（eject）最优解附近的区域。

具体来说：假设当前位置的梯度较大，正常一步更新可以到达目标附近。但如果学习率过大，参数可能直接跳到远处。而标准 Sigmoid 在同一位置梯度较小，使用相同学习率时跳的距离较短，且到达新位置后梯度更小，更容易逐步收缩回来。

### 5.2 恢复能力

- **可恢复：** 如果 eject 后参数仍在有梯度的区域，标准 Sigmoid 和 Scaled Sigmoid 都能恢复，但 Scaled Sigmoid 恢复较慢
- **不可恢复：** 大部分情况下，一旦 eject 到梯度消失区域（Sigmoid 饱和区），无论是否 Scale 都无法恢复。而 Scaled Sigmoid 因为梯度更大，同样学习率下 eject 更远，更容易落入不可恢复区

> **[图7]** Eject 后 recover 的对比
> *(请插入 eject/recover 实验结果图)*

> **[图8]** Eject 后 failed to recover 的对比
> *(请插入 failed to recover 实验结果图)*

**结论：** 对学习率更敏感是 Scaled Sigmoid 的**唯一主要缺点**。使用 Scaled Sigmoid 时需要更精细地调整学习率。

---

## 6. 结论

| 发现 | 说明 |
|------|------|
| Scaled Sigmoid 能避免权重爆炸 | 梯度对冲机制约束参数增长 |
| Scaled Sigmoid 能避免梯度消失 | 参数不会跑到饱和区 |
| Scaled Sigmoid 收敛更快 | 同一位置梯度更大 |
| CNN 中效果显著 | Accuracy 提升，weight 更小 |
| LSTM 中无明显效果 | 门控值天然在中间区域，不存在边界值问题 |
| 对学习率更敏感 | 梯度更大导致 eject 风险增加 |

**核心结论：** Scaled Sigmoid 没有全面取代标准 Sigmoid 的原因是，它的优势只在特定场景下成立——即 Sigmoid 需要输出边界值的场景。且在大部分的情況下梯度會逐漸變小，weight 並不會無限制地變大(從 test case 123 得來的)，且 sigmoid 還有浮點精度运算的限制，因此輸入約 40 輸出已經為1了。在不需要边界值输出的架构中（如 LSTM 的门控），Scale 不会带来实质改善，反而增加了对学习率的敏感性。使用前应评估当前场景中 Sigmoid 的输出需求是否匹配 Scaled Sigmoid 的设计目的。

---

*注：本文中标注 [图N] 处请插入对应的实验结果图。*
