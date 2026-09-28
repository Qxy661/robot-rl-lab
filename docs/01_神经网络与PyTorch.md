# 01 · 神经网络与 PyTorch

这一章讲清楚一件事：**神经网络是怎么"学"的**。读完你能看懂本项目里策略网络的
定义、一次前向传播做了什么、训练循环里那几行代码在干什么。

不涉及强化学习特有的东西——那部分留给下一章。这里只讲一个普通的函数拟合器。

## 神经网络本质是一个带参数的函数

先看一个最朴素的函数：

```python
def f(x):
    return 2 * x + 1
```

输入 3 得到 7。这里的 2 和 1 是人写死的。神经网络做的事是：**把 2 和 1 变成未知
数，让程序自己找出来**。

```python
def f(x, w, b):
    return w * x + b
```

现在给定一批"输入 x 和正确答案 y"的数据，我们可以问：w 和 b 取多少，才能让
`f(x)` 尽量接近 y？这个"找参数"的过程就是训练。

神经网络无非是**把很多这样的式子串起来**，再叠上非线性。一个三层的全连接网络：

```python
h1 = relu(x @ W1 + b1)
h2 = relu(h1 @ W2 + b2)
y  = h2 @ W3 + b3
```

`@` 是矩阵乘法，`W1`、`W2`、`W3` 是待求的参数。这就是全部了——本项目里策略网络
的结构也在这几句的复杂度范围内。

## 为什么需要非线性

如果只有 `W @ x + b` 一层层叠下去，数学上等价于**一个**线性变换，再深也没用。
所以每层之间要插一个非线性函数，让网络能表达弯曲的关系。

最常用的是 ReLU：

```python
def relu(x):
    return np.maximum(x, 0)     # 负数变 0，正数不变
```

本项目的网络用 ELU（ReLU 的平滑版本，在负数区间不是恒为 0，梯度性质更好），
但作用是一样的：切掉线性，让网络有能力拟合复杂函数。选哪个激活函数属于调参细节，
不改变整体结构。

## 前向传播：从输入算出输出

给定参数和数据，一路算到结果，叫前向传播。在 PyTorch 里，网络写作一个类：

```python
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)
```

`nn.Linear(输入维度, 输出维度)` 就是那个 `x @ W + b`，参数存在里面。写出
`forward` 之后，调用 `model(x)` 就会执行它。

`nn.Module` 是 PyTorch 里"可训练模块"的基类。继承它有两个好处：参数会自动
被追踪（能一键收集所有参数、保存、加载），以及能嵌套组合。

## 张量：PyTorch 的数组

`torch.Tensor` 和 NumPy 数组用法几乎一样，差别是它能记住自己是怎么算出来的，
从而支持自动求导，也能搬到 GPU 上。

```python
import torch

x = torch.zeros(4, 15)              # 全 0，形状 (4, 15)
x.shape                              # torch.Size([4, 15])
x.to("cpu")                          # 设备搬运
x.detach().numpy()                   # 转回 NumPy（脱离计算图）
torch.as_tensor(arr, dtype=torch.float32)   # NumPy 数组转张量
```

项目里 NumPy 和 PyTorch 两种数组都会出现，边界很清楚：**环境层用 NumPy**
（物理仿真是 CPU 上的数值计算），**算法层用 PyTorch**。转换发生在环境与算法
交界处。

## 损失函数：给"错得有多离谱"打一个分

要优化参数，先得有办法衡量当前表现有多差。这就是损失函数。

```python
pred = model(x)                      # 预测
loss = ((pred - y) ** 2).mean()      # 均方误差，越小越好
```

一个数，衡量这批数据上的整体误差。训练的目标就是**把这个数压到最小**。

## 反向传播：算出每个参数该往哪调

有了损失，接下来的问题是：网络里有几十万个参数，每个参数该调多少？

微积分给了答案：求损失对每个参数的偏导数（梯度）。梯度指向"损失上升最快的方向"，
所以我们往反方向走。

如果手工推导每个参数的导数，网络一深就没法做了。PyTorch 用链式法则自动完成
这件事——这就是"反向传播"：

```python
loss.backward()                      # 自动算好所有参数的梯度
```

调用一次 `backward()`，每个参数的 `.grad` 里就有了它的梯度。**这行代码是深度学习
最核心的魔法**，剩下的都是工程细节。

## 优化器：按梯度更新参数

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

optimizer.zero_grad()      # 清掉上一轮的梯度（PyTorch 默认累加，不清会串味）
loss.backward()            # 算梯度
optimizer.step()           # 按梯度更新参数
```

这三行加一次前向，就是一个最小的训练步骤。`lr` 是学习率，控制每步走多大——太大
会震荡，太小会学得慢，是最需要调的参数之一。

`Adam` 是最常用的优化器，它给每个参数自适应地调整步长。本项目 PPO 里 actor 和
critic 用不同的学习率（`lr_actor` 与 `lr_critic`），各配一个优化器，原因在 04 章讲。

## 训练循环：把上面拼起来

```python
for epoch in range(1000):
    pred = model(x)                  # 1. 前向
    loss = ((pred - y) ** 2).mean()  # 2. 算损失
    optimizer.zero_grad()            # 3. 清梯度
    loss.backward()                  # 4. 反向传播
    optimizer.step()                 # 5. 更新参数
```

**这五步是所有深度学习训练的骨架**。强化学习里也是这五步，只是 `loss` 的定义
复杂得多，数据也不是现成的而是环境交互产生的。

注意 `zero_grad()` 的位置：PyTorch 的梯度是累加的，忘了清零，梯度会一轮轮叠上去，
表现为训练莫名其妙地崩掉。这是新手最常踩的坑之一。

## 一些必须知道的细节

**训练模式与评估模式。** `model.train()` 和 `model.eval()` 会切换某些层的行为
（比如 Dropout 和 BatchNorm）。本项目的网络里没有这类层，但为了规范仍然会切换，
评估时切到 `eval()`，保证结果可复现。

**不参与梯度的计算。** 评估或者纯推理时，用 `torch.no_grad()` 包起来：

```python
with torch.no_grad():
    action = policy(obs)
```

它会关掉计算图的记录，省内存也更快。忘了加不会算错，只是白费开销。

**线程数不是越多越快。** PyTorch 会把一次算子运算拆给多个线程并行算，但这件事
有个反直觉的地方：网络小的时候，多核反而更慢。原因是算子被切开之后，线程之间的
同步和等待时间超过了省下的计算时间。本项目的实测——16 核机器上一个 64×64 的
MLP 做一次反向传播，16 线程要 258 ms，单线程只要 0.92 ms，差了 280 倍。

本项目的网络只有几万参数，所以固定把 PyTorch 的线程数设成 1
（`robotrl/utils/torch_runtime.py`），并行度留给"多个环境同时跑"这件事。**并行
应该发生在样本维度上，不是算子维度上**——这一条在本项目和大多数机器人 RL 项目
里都成立。

**参数的初始化。** 参数不能全初始化为 0——那样每个神经元的梯度都一样，网络永远
学不到"不同的东西"。PyTorch 有默认的随机初始化，本项目额外用了正交初始化（让
信号在前向和反向传播中保持方差稳定），在 `networks.py` 里可以看到。

**保存与加载。** `model.state_dict()` 是一个字典，装着所有参数张量。保存它就等于
保存了模型：

```python
torch.save(model.state_dict(), "policy.pt")
model.load_state_dict(torch.load("policy.pt"))
```

本项目还额外保存优化器状态和训练步数，这样可以从断点继续训练。

## 回到本项目的代码

现在再看 `robotrl/algorithms/base.py` 里的 `Policy`，应该大半能读懂了：

```python
class Policy(nn.Module, ABC):
    def __init__(self, obs_dim, action_dim, *, obs_clip=10.0):
        super().__init__()
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        ...
    def forward(self, obs):
        return self._policy_forward(self.normalize_obs(obs))
```

几个点：

`register_buffer` 注册的是"跟着模型走、但不是训练参数"的张量。观测归一化的
均值和方差正属于这类——它们需要被保存、被加载、被导出到 ONNX，但不该被梯度
更新。

`forward` 里先做归一化再进网络。这一层归一化是整个项目的关键设计之一：它保证
导出的 ONNX 图**自带预处理**，端侧拿到裸观测直接喂进去就行，不用再写一段
对齐的预处理代码——而那段代码出错的概率很高。

`ABC` 和 `@abstractmethod` 表示这是个抽象基类，子类必须实现 `_policy_forward`，
否则创建实例时会报错。它防止有人写了个半成品策略类，跑起来才发现在某个地方
炸掉。

## 下一步

下一章 [02 · 强化学习基础](02_强化学习基础.md) 讲这个网络为什么能学会控制机器人：
没有现成的"正确答案"数据，我们靠什么更新它。
