---
layout: post
title: "世界模型（一）：记忆、感知、预测、评估、决策的联合"
date: 2019-04-01 00:00:00+08:00
last_modified_at: 2026-02-15 00:00:00+08:00
description: 区分“模拟世界”与“理解世界”，阐述世界模型五大模块的统一可微框架，并讨论空间智能、抽象学习与长期记忆等前沿方向。
tags: [AGI, Agent, World-Model, Memory, Perception, Prediction, RL]
categories: research
series: 世界模型
thumbnail: assets/img/intelligent_radar_preview.png
---

> 本文原写于 2019 年 4 月，拟作为雷达硕士论文的一章，因精力有限搁置。2026 年 1 月重审，补入近年 LLM 与空间智能的新思考，仍愿为相关方向的同学提供一份“初心”笔记。

在讨论世界模型之前，我们先区分两个概念：**模拟世界（simulate）** 与 **理解世界（understand）**。当下的视频生成模型（如 Sora、MovieGen）能在像素层面“模拟”世界，但是否真正把握了背后的物理与因果？借用物理学中的“统一场论”隐喻，我将**世界模型（World Models）**定义为：能够将**记忆、感知、预测、评估、决策**功能联合为**整体可微可导**闭环的模型框架。它不止于生成逼真的帧，还要构建一个能推理、能交互的“心智”。

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/world-model/world_model.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  世界模型概念图
</div>

## 前言

这里其实是借鉴物理学中统一场论的概念：一个可以统一四种基本力的物理理论。

> **统一模型是指一个能够把记忆、感知、预测、评估、决策功能联合为整体可微可导的模型框架。**

下面我会详细说明这些功能的具体内容以及如何将它们“有机”（可微可导）地结合在一起。后续章节则尝试在具体任务中构建它们。

## 多种功能的具体介绍

首先我们对智能体给出这样的描述，智能体应该拥有如下几个功能：
- 记忆功能
- 感知功能
- 预测功能
- 评估功能
- 行动功能

下面将逐一介绍这些功能。

### 记忆功能

> 记忆能力并不是指能够记录信息，而是要能够利用上一时刻的记忆信息和当前时刻的观测信息共同完成信息处理（包括但不限于信息的感知、预测、评估、决策）和当前时刻的记忆形成。以信息感知为例：

$$
s_t,\; m_t \;=\; D\!\big(o_t,\; m_{t-1}\big)
$$

其中 $D$ 为感知系统，$o$ 为观测信息，$s$ 为感知得到的状态信息，$m$ 便是记忆信息。

其实只要能保留长期记忆的时序因果模型在结构上都属于带记忆功能的，这部分比较古老的框架如 RNN、LSTM；最近几年也重新兴起了重铸 RNN 荣光的事情，也就是将 RNN 与 Transformer 相结合，如 RWKV、RetNet 等。我在 2023 年也兴致勃勃地构建了 [llama2RNN.c](https://github.com/siyuanseever/llama2RNN.c) 的 demo（可下载），[这里](https://zhuanlan.zhihu.com/p/681684286) 是一些零碎的介绍，后续会整理成长文。

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/world-model/memoryAttention.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  记忆注意力（Memory Attention）
</div>

### 感知功能

> 感知能力是指系统能够将观测信息进行压缩理解，得到抽象概念，并根据抽象概念大致还原出原始信息的能力：

$$
\hat{o}\;=\;D^{-1}\!\big(D(o)\big)
$$

其中 $D^{-1}$ 为 $D$ 的逆处理系统，得到的抽象概念 $D(o)$ 的数据大小要远小于观测信息 $o$ 的数据大小。

这里比较简单的自编码器就可以完成感知任务了，MAE 也大致可以算作这个思路。

### 预测功能

> 预测能力是指系统能够根据上一时刻从感知系统中得到的状态信息（以及其它能够获取的先验信息）预测下一时刻的状态信息：

$$
s'_{t+1}\;=\;P(s_t)
$$

其中 $P$ 为预测系统。

现在的 LLM 大抵就是这么学习的了，不过它们不是针对状态预测，而是直接对原始信息（仅仅简单的做了下压缩分词变成 Token）做预测。

### 评估功能

> 评估能力是指系统能够对给定的状态做出价值评估，估计出自身状态的好坏，用一个单值表示：

$$
v_t \;=\; E(s_t) \;=\; \mathbb{E}\!\left[r \;+\; \gamma\, E\!\big(s_{t+1}\big)\right]
$$

其中 $E$ 为评估系统，$v$ 为给定状态 $s$ 下的评估价值，该评估值与系统自身接收的真实奖励 $r$ 以及未来奖励 $E(s_{t+1})$ 有关。

### 决策功能

> 行动能力是指系统可以根据状态信息作出行动决策，该行动能够改变环境和自身状态及价值：

$$
\pi(s) \;=\; \arg\max_{a}\, Q(s, a)
$$

这里的重点是能够改变环境和自身状态的行动才是有效的行动，需要注意。其中 $\pi$ 为决策系统，$a$ 为决策系统给出的有效行动，$Q$ 为动作价值函数。决策不仅包括对外部环境和自身状态的改变，甚至是对自身网络结构（完成类似 2019 年比较火的模型架构搜索相关的功能，如 NASNet）和训练过程的改变和控制，指系统能够利用所有可用资源不断地改善智能体的各种功能的效果，也就是系统拥有“自我学习能力”。

## 统一模型的一些例子

### 基于端到端模仿学习的廉价机器人视觉多任务操作系统

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/world-model/demonstration.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  基于端到端模仿学习的廉价机器人视觉多任务操作系统示意
</div>

上图为一个基于端到端模仿学习的廉价机器人视觉多任务操作系统（Vision-Based Multi-Task Manipulation for Inexpensive Robots Using End-to-End Learning from Demonstration），完成了上述的感知、记忆、行动功能的组合。系统包含一个基于多模式自回归估计的输出联合命令的控制网络，和一个重构图片的 VAE-GAN 自编码器，其中编码器（感知系统）为控制网络提供状态特征信息。

### Next State prediction

如果说 LLM 的 Next World prediction 是一个将预测功能发挥到极致效果的表现，那 Next State prediction 就是将预测与感知相结合，来解决信息量密集型数据（如图像）的高效学习。类似于现在我们已经完成了对互联网全部文本数据的学习，但就算是文本也可以用隐层状态预测来极大地提高效率。

下面是我针对视频预测的一个简单构想的框架示意图：

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/world-model/next_word_prediction.drawio.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  Next State 预测构想
</div>

这个构想非常简单直接，就是结合自编码器的感知能力和自回归模型的预测能力，所以有很多相似的想法可以参考：
- Joint Embedding Predictive Architecture（JEPA）
- Emu3.5（我都想入职了：）

by the way, 感觉自己很多思路和出发点都能在 LeCun 老爷子的世界模型那里获得认同感（见“通往自主机器智能的道路”），而且我也同样没有能力把想法给工程化：）。

### V-JEPA 2-AC：自监督视频模型实现理解、预测和规划

这里在感知和预测的基础上，增加了决策的影响，虽然不是直接给出行动（这可能还需要评估模块的引入以及强化学习，我将在下一篇博客中具体介绍：），而是有监督的学习什么样的动作会演变为下一时刻的状态。所以最终实现了对训练数据中动作的简单模仿：）（如果有误，欢迎指正）。

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/world-model/V-JEPA2-AC.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  V-JEPA 2-AC
</div>

## 前沿方向：空间智能 (Spatial Intelligence)

斯坦福李飞飞教授团队（World Labs）最近提出的**空间智能 (Spatial Intelligence)** 概念，为世界模型提供了一个极佳的落地场景。

### 1. 视觉 > 语言？
李飞飞教授指出，相比于语言，**视觉（Vision）** 是更为基础的生物本能。
> Perception and action became the core loop driving the evolution of intelligence.

从寒武纪大爆发开始，**“感知-行动”** 的循环就是推动智能进化的核心动力。没有语言的动物依然可以通过视觉理解物理世界的规则（如重力、空间遮挡）并做出决策。因此，构建 AGI 的下一步，不应仅仅局限于 LLM 的文本逻辑，更需要让 AI 拥有**“空间认知”** 能力。

### 2. 核心定义
她定义的空间智能模型需要具备：
> building frontier models that can perceive, generate, reason, and interact with the 3D world.

这与我上述的“五位一体”定义不谋而合：
- **Perceive (感知)**：理解 3D 空间结构。
- **Generate (预测/生成)**：想象未来的可能性。
- **Reason (评估/记忆)**：进行因果推理。
- **Interact (决策)**：与物理世界进行交互。

### 3. Marble：从“生成视频”到“生成世界”

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/world-model/Marble.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  Marble：持久化 3D 世界生成
</div>

World Labs 推出的首款产品 **Marble**，展示了空间智能与普通视频生成的关键区别：
- **空间一致性 (Spatial Consistency)**：Sora 等视频生成模型往往存在“空间崩坏”的问题（如人走着走着消失了，或者透视关系错误）。而空间智能要求模型内部有一个显式的、符合物理规律的 3D 表达（Hidden State）。
- **持久性 (Persistence)**：生成的不是稍纵即逝的像素帧，而是一个可以被存储、编辑、反复进入的**持久化 3D 世界**。

这种能力让 AI 从“画师”变成了“造物主”，能够构建一个可交互的虚拟实验场。

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/world-model/Long-Context_State-Space_Video_World_Models.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  长时序状态空间视频世界模型
</div>

当然“生成视频”的路子也有解决思路，那就是引入**长期记忆**，从早期的 ConvLSTM，到最新的 State-Space Model，甚至我之前设计的 Block-wise Recurrent Transformer 都是要做这样的时序一致性推理。

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/world-model/Long-Context_State-Space_Model_architecture.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  长时序状态空间模型架构
</div>

---

## 核心特征：像人类一样学习

世界模型与传统深度学习系统的一个显著区别在于**学习路径**。现有的深度学习（DL）效率极低，依赖海量标注数据（Supervised Learning）或试错（RL）。未来十年，AI 的学习方式可能有一些本质的改变：

1. **抽象学习（Abstract Learning）**：像人类医生看 MRI 影像一样，AI 将学会利用“空间常识”和“抽象概念”进行学习，而非死记硬背像素点或者下一个单词。
2. **持续学习（Continual Learning）**：**从“通用”到“进化”**：我们不应追求一个出厂即巅峰的 AGI，而应追求像人类一样能不断适应环境、持续进化的 **Evolving Intelligence**。  
3. **时间感知**：现实世界中，时间的流逝是唯一的物理真理。未来的模型（无论是 CNN 还是 Transformer）最终都要加上类似 **LSTM 的 RNN 体质**。如果模型无法从结构上感知到时间，就无法理解熵增与因果，也就无法诞生真正的“硅基生命”。

通过 RNN 类的架构，模型将具备**时序因果的长期记忆**。这不仅能解决“长度外推”问题，更能让 AI 在物理世界的单向时间流中，通过持续的 Training Step 和状态保留，像生物一样积累经验，而非每次重启都被“格式化”。

---

## 实践案例：智能电磁博弈（Intelligent Electromagnetic Game）

为了证明这个框架不仅仅适用于生成视频或玩游戏，我以一个更“硬核”的领域——**雷达与干扰的博弈**为例，展示世界模型是如何在信号处理领域落地的。

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/world-model/intelligent_electromagnetic_game.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  智能电磁博弈示意
</div>

在我的硕士论文研究中，构建了一个基于深度强化学习的**雷达智能体**，它完整地体现了世界模型的闭环：

1. **感知与记忆**：利用 **Conv-LSTM** 处理连续脉冲回波，不仅提取当前特征，还保留了历史脉冲的长期记忆。
2. **决策（Action）**：雷达的发射波形不再是固定的，而是由策略网络 $\pi(o_{t-1})$ 根据历史观测生成的。
3. **评估（Value）**：构建价值网络 $V(o_t)$，预测当前波形在未来对抗中的长期检测回报。
4. **博弈（World）**：雷达与干扰机（环境）在**全微分**的链路中进行对抗训练。

在这个模型中，**发射波形（决策）-> 环境干扰（反馈）-> 回波检测（感知/评估）** 形成了一个完整的端到端闭环。

---

## 结束语

我认为历史上众多璀璨的想法和技术，无论是强化学习、元学习、自回归预测、压缩感知等学习方式，还是 RNN、ResNet、Transformer 等具体的模型结构，亦或者是模型架构或者训练超参数搜索等等技术，它们都有自己的可取之处。我也相信未来 AGI 的构建需要这些智慧结晶，而对于现在极致工业化和商用流行的 LLM 也不会嫌弃，它们都是构成这个宏大“世界模型”拼图的一部分。

---

系列导航  
- 下一篇：[世界模型（二）：智能电磁博弈]({% post_url 2019-06-01-intelligent-radar %})