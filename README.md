# micrograd-study

个人的深度学习与自动微分引擎学习仓库，手敲并复现了 Andrej Karpathy 的 [micrograd](https://github.com/karpathy/micrograd)。

---

## 项目简介

本项目用于深入理解神经网络底层的前向传播与反向传播（Backpropagation）机制。通过实现一个轻量级的纯 Python 标量自动求导引擎，彻底搞清楚计算图是如何构建以及梯度是如何沿着链式法则传递的。

## 实现内容与个人补充

- [x] **标量自动求导引擎 (`Value` 类)**：实现了加、减、乘、除、幂以及激活函数（$\tanh$、ReLU），支持动态构建 DAG（有向无环图）并执行拓扑排序反向传播。
- [x] **简易神经网络模块 (`nn.py`)**：实现了 `Neuron`（神经元）、`Layer`（单层）和 `MLP`（多层感知机）。
- [x] **个人注释与理解**：在反向传播和梯度积累部分添加了详细的中文注释和演算推导。

## 快速上手

### 环境准备

```bash
# 克隆本仓库
git clone [https://github.com/epiphanydeer/micrograd-study.git](https://github.com/epiphanydeer/micrograd-study.git)
cd micrograd-study

```

### 运行示例

```python
from engine import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
g.backward()

print(f"g value: {g.data}")  # 最终输出结果
print(f"a grad:  {a.grad}")  # 对 a 的梯度
```

## 参考资料与致谢

- **原项目**：[karpathy/micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy
- **配套视频**：[The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0)

## 开源许可证 (License)

本项目遵循 **MIT License**。
核心实现基于原作者 [Andrej Karpathy](https://github.com/karpathy) 的 `micrograd` 代码，原始版权归原作者所有。