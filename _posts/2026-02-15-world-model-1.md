---
layout: post
title: "世界模型（一）：什么是世界模型？"
date: 2026-02-15 12:00:00
description: 区分“模拟世界”与“理解世界”，阐述世界模型五大模块的统一可微框架，并讨论空间智能、抽象学习与长期记忆等前沿方向。
tags: AGI Agent World-Model Memory Perception Prediction RL
categories: research
series: 世界模型
thumbnail: assets/img/intelligent_radar_preview.png
---

在讨论世界模型之前，我们先区分两个概念：**模拟世界（simulate）** 与 **理解世界（understand）**。当下的视频生成模型（如 Sora、MovieGen）能在像素层面“模拟”世界，但是否真正把握了背后的物理与因果？借用物理学中的“统一场论”隐喻，我将**世界模型（World Models）**定义为：能够将**记忆、感知、预测、评估、决策**功能联合为**整体可微可导**闭环的模型框架。它不止于生成逼真的帧，还要构建一个能推理、能交互的“心智”。

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/world-model/world_model.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  世界模型概念图
</div>

## 一、前言：统一可微的闭环

统一模型是指一个能够把记忆、感知、预测、评估、决策功能联合为整体可微可导的模型框架。以下逐一介绍这些功能，并说明如何将它们“有机”（可微耦合）地结合在一起。

## 二、五大模块：功能与形式

- 记忆（Memory）：保存并更新时序因果的内部状态，利用上一时刻记忆与当前观测共同计算当前状态与新记忆  
  $$
  s_t,\; m_t \;=\; D\!\big(o_t,\; m_{t-1}\big)
  $$
- 感知（Perception）：将高维观测压缩为抽象状态，并可大致重构原始观测，类似自编码器/MAE  
  $$
  \hat{o}\;=\;D^{-1}\!\big(D(o)\big)
  $$
- 预测（Prediction）：从抽象状态出发进行**下一状态预测（Next State Prediction）**而非像素预测  
  $$
  s'_{t+1}\;=\;P(s_t)
  $$
- 评估（Evaluation）：对状态的“好坏”进行价值评估（价值网络）  
  $$
  v_t \;=\; E(s_t) \;=\; \mathbb{E}\!\left[r \;+\; \gamma\, E\!\big(s_{t+1}\big)\right]
  $$
- 决策（Decision）：基于状态选择行为，影响环境与自身（策略/动作价值）  
  $$
  \pi(s) \;=\; \arg\max_{a}\, Q(s,a)
  $$

要点：记忆并非仅存储，而是承载时序因果；感知将观测压缩为更结构化的状态；预测应在状态空间中演化；评估为长期目标提供回传；决策通过行动改变未来。五者的**协同与可微耦合**，形成“感知—预测—评估—行动—再感知”的闭环。

## 三、从生成到交互：为什么是“状态预测”

- 生成视频（如 Sora/MovieGen）强调像素细节，但过度关注纹理可能牺牲物理与因果建模的算力预算。  
- 交互式生成（如 Genie/Genie2）引入动作与环境响应，更接近世界模型的本质。  
- 高效世界模型应工作在**抽象状态空间**而非像素空间（类似 JEPA 思路），用更低维、更结构化的隐状态承载动力学与因果。

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/world-model/genie3.gif" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  交互式生成 Genie
</div>

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/world-model/JEPA.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  JEPA：在抽象状态空间进行预测
</div>

## 四、统一模型的一些例子

### 1. 基于端到端模仿学习的廉价机器人视觉多任务操作系统

完成了“感知—记忆—行动”的组合：控制网络输出联合命令，自编码器（VAE-GAN）作为感知模块为控制网络提供状态特征，实现端到端视觉到行为的映射。  
<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/world-model/demonstration.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  端到端模仿学习示意
</div>

### 2. Next State Prediction

如果说 LLM 的 Next Word Prediction 将预测功能发挥到极致，那么针对密集信息（如视频）的高效学习，更应在**状态空间**做下一状态预测。下图为一个简单的构想：结合自编码器的感知能力和自回归的预测能力。  
<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/world-model/next_word_prediction.drawio.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  Next State 预测构想
</div>

相关思路可参考：JEPA、Emu3.5 等。直观上，将“像素/词”转到“隐状态”进行预测，更接近因果与动力学的本质。

### 3. V-JEPA 2-AC：自监督视频模型实现理解、预测和规划

在感知与预测的基础上引入动作信息，虽未直接生成行动决策，但学习“什么样的动作会演变为下一时刻的状态”，从而实现对训练数据中动作的模仿。  
<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/world-model/V-JEPA2-AC.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  V-JEPA 2-AC
</div>

另外，针对“记忆”的长期一致性建模，可参考下图的记忆注意力思路：  
<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/world-model/memoryAttention.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  记忆注意力（Memory Attention）
</div>

## 五、空间智能：从“生成视频”到“生成世界”

以“空间智能（Spatial Intelligence）”为例：目标是构建能**感知、生成、推理、交互**的 3D 世界表征。与纯视频生成不同，空间智能强调：
- 空间一致性：内部具备显式、符合物理规律的 3D/隐状态表示；
- 持久性：生成的不只是帧序列，而是可被存储、编辑、反复进入的**持久化世界**。

这要求模型具备长期记忆与连贯推理能力，可通过 ConvLSTM、状态空间模型、或结合记忆的 Transformer/混合结构实现。

> Perception and action became the core loop driving the evolution of intelligence.

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/world-model/Marble.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  Marble：从生成视频到生成世界
</div>

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/world-model/Long-Context State-Space Video World Models.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  Long-Context State-Space Video World Models
</div>

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/world-model/Long-Context State-Space Model architecture.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  Long-Context State-Space Model 架构
</div>

## 六、像人类一样学习：抽象、持续与时间

- 抽象学习：更多依赖空间常识与抽象概念，而非逐像素/逐词的死记硬背。  
- 持续学习：从“通用”走向“进化”，在交互中不断适应与改进自身。  
- 时间感知：结构上具备“对时间敏感”的归纳偏置（如带记忆的序列结构），才能理解熵增与因果，并形成真正的长期经验积累。

进一步地，通过带记忆的架构，模型具备**时序因果的长期记忆**，不仅可解决“长度外推”，更能在单向时间流中积累经验，而非每次重启都被“格式化”。

## 七、实践案例预告：智能电磁博弈

为验证框架的通用性，下一篇展示**智能电磁博弈**案例：以 ConvLSTM 等结构承载长期记忆，通过策略与价值网络形成端到端可微闭环，让“发射波形—环境干扰—回波检测—价值评估—策略更新”在同一链路中共同优化。

---

## 参考与延伸

- llama2RNN.c demo: https://github.com/siyuanseever/llama2RNN.c  
- 零碎介绍： https://zhuanlan.zhihu.com/p/681684286  
- World Model 讨论与资料：
  - https://www.xunhuang.me/blogs/world_model.html
  - https://leshouches2022.github.io/SLIDES/compressed-yann-1.pdf
  - https://drfeifei.substack.com/p/from-words-to-worlds-spatial-intelligence
  - https://www.worldlabs.ai/blog/marble-world-model

---

系列导航
- 下一篇：[世界模型（二）：智能电磁博弈]({% post_url 2019-06-01-intelligent-radar %})
