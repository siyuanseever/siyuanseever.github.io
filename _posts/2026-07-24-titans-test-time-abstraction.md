---
layout: post
title: "从 Test-Time Memory 到 Test-Time Abstraction：读 Titans 后的一些思考"
date: 2026-07-24 00:00:00+08:00
last_modified_at: 2026-07-24 00:00:00+08:00
description: 以 Titans 的 test-time memory 机制为起点，讨论长期记忆问题的关键是否应从精确检索（exact association）转向从历史中形成可复用抽象（test-time abstraction），并给出若干架构层面的分析与研究设想。
tags: [AGI, Memory, Test-Time-Learning, Long-Term-Memory, Titans, Transformer]
categories: research
series: 长期记忆
thumbnail: assets/img/titans/fig1_titans_memory_intuition.png
---

> 本文不是对 **Titans: Learning to Memorize at Test Time** 的完整复述，而是我读后基于其 memory 机制展开的一组分析与研究设想，记录一些尚不成熟的想法，供同方向的朋友讨论。

“长期记忆”正在成为大模型研究中的一个常见命题：上下文有限，因此需要 memory；模型持续与环境交互，因此需要在 test time 保留历史信息。

但这里有一个更基础的问题：**长期运行的智能系统，真正缺少的是更多记忆，还是从历史中形成抽象的能力？**

Titans 将 test-time memory 表述为一个可在线更新的 neural memory function。沿着这一框架继续推演，我更关心另一个问题：如果参数更新既可以形成记忆，也可以形成抽象，那么 test-time learning 的目标是否应该超出 associative retrieval，进一步走向 **test-time abstraction**？

---

## 1. Memorization 与 Abstraction

传统机器学习通常要求模型避免对训练样本的简单记忆。精确拟合具体样本可以降低训练误差，却未必保留跨样本稳定的结构；当输入分布变化时，这类表示往往缺乏迁移能力。

理想的学习过程更接近：

```text
具体样本 → 共享结构 → 可泛化规律
```

记忆偏向具体、精确与局部；抽象偏向压缩、概括与迁移。二者并非对立。智能系统既需要保存特定事实，也需要从大量经验中形成可复用结构。

因此，长期 memory 的问题不应只表述为“如何保存更多历史”。如果系统最终仍依赖近似查字典式的关联检索，那么 memory capacity 的增加并不等价于持续学习能力的提高。

---

## 2. Titans：将 Memory 写成可更新函数

Titans 的关键变化，是将历史信息写入一个可在 test time 更新的 neural memory function。

传统 recurrent memory 可能是一个向量：

$$
h_t
$$

或者一个矩阵：

$$
S_t
$$

而 Titans 更像是维护一个函数：

$$
M_t(\cdot)
$$

当新的 context 到来时，memory function 的参数会在 test time 被更新：

$$
M_{t-1} \rightarrow M_t
$$

因此，历史并非仅以 token 或固定 hidden state 的形式保存，而是改变了后续查询所使用的函数本身。Memory 由此与 online learning 建立了直接联系。

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/titans/fig1_titans_memory_intuition.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  图 1：Titans 的 memory 直觉图
</div>

---

## 3. Surprise 与 Prediction Error

如果用 key-value memory 来理解：

$$
k_t \rightarrow v_t
$$

memory 先根据 key 预测 value：

$$
\hat{v}_t = M_{t-1}(k_t)
$$

然后比较预测值和真实值：

$$
e_t = v_t - \hat{v}_t
$$

误差越大，当前信息相对于已有 memory 越不可预测，相应的更新强度也应越高：

```text
越预测不出来 → 越 surprise → 越需要写入 memory
```

这一形式与 Delta Rule、linear attention 中的 associative memory，以及 TTT-style fast weights 具有明显连续性。Titans 的重要变化不在于凭空引入一种新 memory，而在于将线性 memory matrix 推广为容量更高的 neural network。

```text
linear memory map → neural memory function
```

---

## 4. 参数更新：记忆与抽象的共同机制

普通训练时，我们更新模型参数：

$$
\theta \leftarrow \theta - \eta \nabla_{\theta} L
$$

Titans 里，test-time memory 也在更新参数：

$$
\theta_{\mathrm{memory}}
\leftarrow
\theta_{\mathrm{memory}}
-
\eta \nabla_{\theta_{\mathrm{memory}}} L
$$

从优化形式看，training-time learning 与 test-time memory 并没有截然不同的边界。二者都通过 objective 驱动参数更新。差异更可能来自：

```text
数据作用域不同
时间尺度不同
reset 边界不同
objective 不同
更新的是主模型参数还是 memory 参数
```

因此，**memorization 与 abstraction 未必对应两套不同的学习机制**。同一种梯度更新可能形成不同层次的表示：

```text
specific experience → memorization
shared structure    → abstraction
```

这里并不是将 memory 等同于 abstraction。参数更新只规定学习过程，并不决定最终形成精确记忆还是抽象结构；后者取决于 bottleneck、regularization、data scale、objective 与 inductive bias。

---

## 5. Next-Token Prediction 中的抽象压力

语言模型并非简单地用一个 token 预测另一个 token，而是根据历史序列预测后续 token：

```text
前面一整段上下文 → 下一个 token
```

也就是说，模型要在极其庞大的上下文组合空间里预测未来。如果词表大小是 $V$，上下文长度是 $N$，那么可能的上下文空间非常大：

$$
V^N
$$

真实语言当然不是均匀分布的，但这个直觉仍然成立：模型面对的是巨大的条件预测空间。

随着训练数据规模增长，逐项背诵所有上下文—输出映射的代价迅速上升，模型因而承受更强的结构复用压力。它需要学习语法、语义、世界知识、长程依赖与可复用的统计规律。

Next-token prediction 不能消除 memorization，但它提供了一种重要的 abstraction pressure：

```text
从大量具体序列中学习可复用的预测结构
```

---

## 6. Associative Retrieval 是否足够？

Titans-style memory 可以粗略写成：

$$
M(k_t) \approx v_t
$$

也就是给定 key，希望从 memory function 中恢复 value。这一结构与检索过程天然一致（inference 时 query 也可以这样读取 $M(q_t)$），但其 objective 明显偏向 exact association：

$$
k \rightarrow v
$$

精确关联对 retrieval 有效，却未必自动产生 abstraction。

如果一个长期运行的模型经历了很多事件 $e_1, e_2, \ldots, e_n$，精确记忆保存的是这些事件本身：

$$
e_1, e_2, \ldots, e_n
$$

而抽象想形成的是：

$$
z = f(e_1, e_2, \ldots, e_n)
$$

其中 $z$ 可能不是任何一个原始事件，而是这些事件背后的结构。

因此，长期 memory 的目标存在一个尚未解决的转向：

> 从保存“发生了什么”，转向形成“这些事件共同说明了什么”。

本文将这一方向称为 **Test-Time Abstraction**。

---

## 7. 高阶 Memory：一个不理想的方向

最自然的想法是：既然单个 $k_t \rightarrow v_t$ 太简单，能不能让 memory 看更长历史？

$$
k_1, k_2, \ldots, k_t \rightarrow v_t
$$

甚至：

$$
k_1, k_2, \ldots, k_t \rightarrow v_{t+1}
$$

这种形式表面上接近 sequence prediction，但存在直接的复杂度矛盾。我们最初引入 memory，是因为完整历史太长，不能每次都重新 full attention；如果为了更新 memory 又要求它重新读取完整历史，那就绕回去了。然后很容易出现：

```text
一阶 memory 不够
引入二阶 memory
二阶 memory 又需要三阶 memory
最后变成 memory of memory of memory
```

这类 memory-of-memory 结构容易将长历史问题转化为层级递归问题，而非真正解决它。更合理的原则可能是：

```text
query 要便宜，抽象应该发生在 update / consolidation 阶段
```

检索应保持低成本；历史压缩与抽象主要发生在 update / consolidation 阶段。Memory 不应在每次 query 时重新读取过去，而应在持续更新中逐步吸收、压缩并重组过去。

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/titans/fig2_higher_order_memory_recursive.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  图 2：为什么二阶 / 三阶 memory 容易套娃，以及更合理的路线
</div>

---

## 8. Titans 已包含的约束机制

Titans 并不是完全没有考虑这个问题。它里面的 momentum、forgetting / gate、weight-decay-like mechanism，其实都可以被理解为某种 memory dynamics 的约束：

```text
momentum：更新不完全被单个样本支配
forgetting / gate：控制哪些信息保留、哪些信息衰减
weight decay 类机制：限制参数无限增长
chunk / segment：让 memory update 可以分块并行
```

这些机制不能直接保证 abstraction，却已经超出静态写入的范畴。它们在做一件更接近 online optimization 的事情：

```text
new information + old memory + update dynamics + forgetting → new memory
```

Titans 的意义之一，在于将 memory 从“存储结构”推进为“动态学习过程”。但动态学习是否足以形成抽象，仍取决于 objective 与容量约束。

---

## 9. 三种 Memory Incorporation 方式

Titans 讨论了几种将 memory 接入模型的方式。三种结构分别强调上下文扩展、表示调制与中间层变换。

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/titans/fig3_memory_incorporation_comparison.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  图 3：Titans 的三种 memory 使用方式
</div>

### 9.1 Memory as Context：最自然，但 chunk 边界有毛刺感

Memory as Context 大致把几类 token 拼起来：

```text
persistent memory + retrieved long-term memory + current segment
```

然后让当前 segment 在这个扩展 context 上做 attention，保留了 “retrieved memory + local attention” 的结构。

其主要问题来自 segment / chunk 边界：一个 segment 开头的位置，能看到的当前 segment 内 token 很少；越往后，能看到的 token 越多；下一个 segment 又重新开始。这会带来一种边界处的“毛刺感”。可能的缓解方式包括：

```text
overlapping chunks
shifted windows
multi-scale segments
boundary smoothing
cross-segment residual memory
```

所以 chunking 虽然工程上很必要，但它不是没有代价。

### 9.2 Memory as Gate：更像调制，而不是检索

如果 memory 输出 $m_t = M(q_t)$，然后主要通过 gate 作用在当前表示上：

$$
y_t = g(m_t) \odot h_t
$$

那我会觉得它更像 memory-conditioned modulation，而不是 explicit content retrieval。这种结构适合控制信息流，但作为 long-term memory 的主要读取方式，其内容表达能力相对间接。

### 9.3 Memory as Layer：可能把精确信息先模糊掉

第三种更像：

$$
x_t \rightarrow M(x_t) \rightarrow \mathrm{LocalAttention}
$$

其潜在问题是：local attention 不再直接处理原始 representation，而是在处理 memory-transformed representation。如果 $M(\cdot)$ 引入了信息丢失、平滑、噪声或 online update 带来的不稳定，那么后面的 attention 即使很精确，也只能精确地操作已经被转换过的信息。所以它可能带来一种 representation bottleneck。

---

## 10. Persistent Memory 与 Learnable Prompt

Titans 里的 persistent memory，我更愿意把它理解成 learnable prompt / soft prompt / prefix tokens：一组训练阶段学出来的、相对 input-independent 的参数。它不像 long-term neural memory 那样在当前 context 中持续更新，而更像给模型提供一组持久的 task-level memory tokens。从这一角度看，persistent memory 与已有 learnable prompt 范式具有直接联系，不必将其理解为独立的长期记忆机制。

---

## 11. Shared Conditional Memory

如果每个 layer、每个 head 都有自己独立的 memory network $M_{\ell,h}$，那么 memory 系统会非常庞大；更重要的是，每个 head 都有自己的私有 memory，未必会形成跨 head、跨 layer 的共享抽象。

一个替代方向是共享 memory network 的核心参数 $M_{\mathrm{shared}}$。但不同层、不同 head 的 query / key / value 可能不在同一个 representation space，所以不能直接把所有 query 都喂给同一个 memory。更合理的结构是：

$$
q_{\ell,h}
\rightarrow
A_{\ell,h}
\rightarrow
M_{\mathrm{shared}}
\rightarrow
B_{\ell,h}
\rightarrow
y_{\ell,h}
$$

也就是：

```text
private space → shared latent memory space → private space
```

这个结构有几个潜在好处：

```text
减少每层每头独立 memory 的参数量
让不同 head / layer 共享 memory dynamics
通过共享核心形成 capacity bottleneck
迫使不同来源的 experience 进入共同 latent memory space
可能产生更强的 abstraction pressure
```

与高阶 memory 相比，这一方向不增加 query 的历史依赖，而是通过共享参数、空间对齐与容量限制增强 abstraction pressure。

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/titans/fig4_shared_conditional_memory_arch.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  图 4：Shared Conditional Memory 架构示意图
</div>

---

## 12. 与 Chunkwise Recurrent Memory 的关系

我之前也思考过 chunkwise recurrent memory：用局部 attention 处理当前 chunk，同时让历史信息以压缩 memory 的形式跨 chunk 传递（参见 [llama2Rnn.c](https://github.com/siyuanseever/llama2Rnn.c)）。读 Titans 后，这类结构可以进一步写成 recurrent memory transition：

$$
M_t = F_{\theta}(M_{t-1}, X_t)
$$

如果 $F_{\theta}$ 内部包含 attention-like merge、nonlinear FFN、compression，那么它就不是简单线性 recurrence，而是一种 nonlinear recurrent memory system。本文不展开这一方向。显式 memory slots、parametric memory、chunkwise recurrence 与 test-time abstraction 之间的关系，值得单独讨论。

---

## 13. 从 Test-Time Memory 到 Test-Time Abstraction

Titans 提供了一个重要视角：memory 可以是 test time 持续更新的学习系统，而不仅是静态存储结构。但现有许多 test-time memory 仍以 exact association 为主要目标。

长期运行的自适应系统需要同时回答两个问题：过去发生了什么？以及，这些过去的事情共同说明了什么？精确记忆保存 episode，抽象形成 schema：

```text
episodes → patterns → schema → future generalization
```

因此，更值得追问的问题不是“如何让模型保存更多 token”，而是：

> 如何让模型在 test time 从 context 中形成可复用的抽象？

一个可能的研究假设是：

> Test-time memory should not only optimize for exact associative retrieval. A long-term adaptive system may require bottlenecks, shared parameterization, regularization, predictive pressure, and consolidation mechanisms that transform historical context into reusable abstractions.

也就是说：测试时记忆不应只优化精确检索。一个真正长期运行的自适应系统，可能需要通过瓶颈、参数共享、正则化、预测压力和 consolidation，把历史 context 转化为可复用的抽象结构。

最终的问题可以压缩为一句话：

```text
Can a model learn to abstract at test time?
```

模型能不能不只是记住世界，而是在不断经历世界的过程中，慢慢形成自己的概念。
