---
layout: post
title: "World Models (I): The Union of Memory, Perception, Prediction, Evaluation, and Decision"
date: 2019-04-01 00:00:00+08:00
last_modified_at: 2026-02-15 00:00:00+08:00
description: Distinguishing "simulating the world" from "understanding the world", we present a unified, differentiable framework that couples five core modules, and discuss frontiers such as spatial intelligence, abstract learning, and long-term memory.
tags: [AGI, Agent, World-Model, Memory, Perception, Prediction, RL]
categories: research
series: World Models
thumbnail: assets/img/intelligent_radar_preview.png
---

> First drafted in April 2019 for my M.S. thesis on intelligent radar; revived in Jan 2026 with new insights from LLMs and spatial intelligence. May this note serve fellow travellers on the road to AGI.

Before diving in, let us distinguish two concepts: **simulating the world** and **understanding the world**. Modern video-generative models (e.g. Sora, MovieGen) excel at pixel-level *simulation*, yet do they *understand* the underlying physics and causality? Borrowing the metaphor of a “unified field theory” from physics, I define a **World Model** as a **differentiable, end-to-end framework** that tightly couples five functions—**memory, perception, prediction, evaluation, and decision**—into a single, learnable closed loop. The goal is not merely photorealistic frames, but a reasoning, interactive *mind*.

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/world-model/world_model.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  Conceptual diagram of a world model
</div>

## Prelude

The metaphor is borrowed from physics: a unified field theory that merges the four fundamental forces.

> A **unified model** here means one that fuses memory, perception, prediction, evaluation, and decision into a **single, differentiable, end-to-end architecture**.

Below I detail each module and show how to weave them together “organically” (i.e. differentiably). Later sections instantiate the framework on concrete tasks.

## Five Functional Modules

An agent should implement the following:
- **Memory** – temporal, causal memory
- **Perception** – compressive representation
- **Prediction** – next-state forecasting
- **Evaluation** – value estimation
- **Decision** – policy / action selection

### 1. Memory

> Memory is **not** passive storage; it **actively** combines the previous memory state $m_{t-1}$ with the current observation $o_t$ to produce an updated state $s_t$ and memory $m_t$.

$$
s_t,\; m_t \;=\; D\!\big(o_t,\; m_{t-1}\big)
$$

Any recurrent architecture that preserves long-term causality qualifies—classic RNNs, LSTMs, and recent “Renaissance” hybrids such as RWKV, RetNet, Mamba, etc. In 2023 I hacked [llama2RNN.c](https://github.com/siyuanseever/llama2RNN.c) as a toy demo; a longer write-up is forthcoming.

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/world-model/memoryAttention.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  Memory-attention mechanism
</div>

### 2. Perception

> Perception compresses high-dimensional observations into abstract states and approximately reconstructs the original signal.

$$
\hat{o}\;=\;D^{-1}\!\big(D(o)\big)
$$

The code $D(o)$ must be dramatically smaller than the raw observation $o$. Vanilla auto-encoders or MAE already satisfy this template.

### 3. Prediction

> From the abstract state (and any prior) the agent forecasts the **next abstract state**, not the next pixel frame.

$$
s'_{t+1}\;=\;P(s_t)
$$

Large language models follow the same principle, except they predict raw tokens rather than states.

### 4. Evaluation

> The agent assigns a scalar **value** to each state, reflecting expected cumulative reward.

$$
v_t \;=\; E(s_t) \;=\; \mathbb{E}\!\left[r \;+\; \gamma\, E\!\big(s_{t+1}\big)\right]
$$

This is the value network familiar in RL.

### 5. Decision

> The agent **acts** to change both the external world and its own internal state.

$$
\pi(s) \;=\; \arg\max_{a}\, Q(s, a)
$$

Actions include not only motor commands but also **self-modifications**—e.g. architecture search (NASNet-style), learning-rate updates, or any differentiable controller that rewrites its own parameters.

## Instantiations

### A. Vision-based Multi-task Manipulation from Demonstration

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/world-model/demonstration.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  End-to-end imitation learning for cheap robot arms
</div>

The system couples a multi-modal auto-regressive control network with a VAE-GAN reconstructor; the encoder (perception) feeds state features to the controller, yielding a minimal but complete perception–action loop.

### B. Next-State Prediction instead of Next-Token Prediction

If LLMs push *next-token* prediction to the extreme, **next-state** prediction couples forecasting with perception for data-efficient learning on high-bandwidth modalities such as video.

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/world-model/next_word_prediction.drawio.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  Next-state predictive framework
</div>

Key references:
- Joint Embedding Predictive Architecture (JEPA)
- Emu3.5

LeCun’s roadmap to autonomous machine intelligence resonates strongly with this line of thought—sadly I still lack the engineering muscle to ship a full-scale demo.

### C. V-JEPA 2-AC: Self-supervised Video Understanding & Planning

V-JEPA 2-AC adds **action conditioning** to perception and prediction. Although it does not emit *actions* directly (evaluation + RL are still needed), it learns to imitate state-action transitions observed in the training videos.

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/world-model/V-JEPA2-AC.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  V-JEPA 2-AC overview
</div>

## Frontiers: Spatial Intelligence

Prof. Fei-Fei Li’s team (World Labs) recently popularised **Spatial Intelligence**—a perfect sandbox for world models.

### 1. Vision before Language?

> Perception and action became the core loop driving the evolution of intelligence.

Even pre-vertebrate animals without language rely on vision to grasp physics (gravity, occlusion) and act. The next leap toward AGI must therefore endow AI with **spatial cognition**, not merely linguistic competence.

### 2. Definition

> Building frontier models that can **perceive, generate, reason, and interact** with the 3D world.

This aligns one-to-one with our five-module taxonomy:
- **Perceive** – 3D structure understanding
- **Generate** – imagine future states
- **Reason** – causal inference (evaluation + memory)
- **Interact** – decision-making in physical spaces

### 3. Marble: From “Generating Videos” to “Generating Worlds”

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/world-model/Marble.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  Marble: persistent, editable 3D worlds
</div>

Marble highlights two deficits of video-centric models:
- **Spatial inconsistency** – objects drift or vanish; perspective violates physics.
- **Ephemerality** – pixels disappear; no persistent 3D substrate.

Spatial intelligence demands an **explicit 3D latent state** that respects physics and remains editable. The AI graduates from *painter* to *demiurge*.

Long-form temporal consistency can also be injected via **long-context memory**, from early ConvLSTM to modern state-space models and my own block-wise recurrent transformer experiments.

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/world-model/Long-Context_State-Space_Video_World_Models.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  Long-context state-space video world models
</div>

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/world-model/Long-Context_State-Space_Model_architecture.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  State-space model architecture for long contexts
</div>

## Learning like Humans

World models diverge from mainstream deep learning in **data efficiency** and **adaptation**.

1. **Abstract Learning** – physicians read MRI scans by *concepts*, not pixels; future AI must exploit spatial commonsense.
2. **Continual Learning** – we should target an **evolving intelligence** that adapts lifelong, rather than a frozen AGI that ships once.
3. **Temporal Awareness** – time is the only unquestionable physical quantity. Any serious model (CNN or Transformer) will eventually re-acquire an **RNN backbone**; without it, entropy and causality remain invisible, precluding true *silicon life*.

Recurrent inductive biases endow models with **long-term, causal memory**, solving length extrapolation *and* letting AI accumulate experience across training steps instead of being *reformatted* after every restart.

## Case Study: Intelligent Electromagnetic Game

To show that the framework is *not* limited to video games, I apply it to **radar–jammer adversarial signalling**—a decidedly *hardcore* domain.

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/world-model/intelligent_electromagnetic_game.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  Intelligent electromagnetic game: radar vs. jammer
</div>

My M.S. thesis built a **deep-RL radar agent** implementing the full loop:
1. **Perception + Memory** – Conv-LSTM ingests pulse echoes, retaining long-term memory of earlier pulses.
2. **Decision** – a policy network $\pi(o_{t-1})$ *generates* the next transmit waveform instead of using a fixed template.
3. **Evaluation** – a value network $V(o_t)$ predicts the long-term detection return of the chosen waveform under future jamming.
4. **World** – radar and jammer co-train in a **fully differentiable** adversarial channel.

The cycle **transmit (decision) → jamming (world feedback) → echo detection (perception / evaluation)** forms an end-to-end closed loop.

## Epilogue

History offers a constellation of ideas—RL, meta-learning, self-supervised prediction, compressive sensing, RNNs, ResNets, Transformers, NAS, and more. Each has its merits. The AGI of tomorrow will weave them together without disdain, greeting even today’s over-industrialised LLMs with the words:

> “You have arrived precisely on time.”

---

Series Navigation  
- Next: [World Models (II): Intelligent Electromagnetic Game]({% post_url 2019-06-01-intelligent-radar %})