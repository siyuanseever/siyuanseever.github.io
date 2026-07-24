---
layout: post
title: "From Test-Time Memory to Test-Time Abstraction: Some Thoughts After Reading Titans"
date: 2026-07-24 00:00:00+08:00
last_modified_at: 2026-07-24 00:00:00+08:00
description: Starting from the test-time memory mechanism in Titans, I ask whether the crux of long-term memory should shift from exact association toward forming reusable abstractions from history (test-time abstraction), and sketch a few architectural analyses and research hypotheses.
tags: [AGI, Memory, Test-Time-Learning, Long-Term-Memory, Titans, Transformer]
categories: research
series: Long-Term Memory
thumbnail: assets/img/titans/fig1_titans_memory_intuition.png
---

> This is not a full recap of **Titans: Learning to Memorize at Test Time**. It is a set of analyses and research hypotheses built on top of its memory mechanism—some still-unpolished thoughts, shared for discussion with others working in this direction.

"Long-term memory" is becoming a common theme in large-model research: context is finite, so we need memory; models keep interacting with an environment, so they need to retain history at test time.

But there is a more basic question: **for a long-running intelligent system, what is truly missing—more memory, or the ability to form abstractions from history?**

Titans expresses test-time memory as a neural memory function that can be updated online. Following that framing, I care more about another question: if parameter updates can produce both memorization and abstraction, should the goal of test-time learning go beyond associative retrieval, toward **test-time abstraction**?

---

## 1. Memorization vs. Abstraction

Traditional machine learning usually asks models to avoid simply memorizing training examples. Fitting specific samples exactly can lower training error, yet it does not necessarily preserve structure that is stable across samples; when the input distribution shifts, such representations often fail to transfer.

An ideal learning process is closer to:

```text
specific samples → shared structure → generalizable rules
```

Memory leans toward the specific, the exact, and the local; abstraction leans toward compression, generalization, and transfer. The two are not opposites. An intelligent system needs both to store particular facts and to form reusable structure from large amounts of experience.

So the long-term memory problem should not be framed only as "how to store more history." If the system ultimately still relies on dictionary-lookup-style associative retrieval, then increasing memory capacity is not equivalent to improving continual-learning ability.

---

## 2. Titans: Writing Memory as an Updatable Function

The key change in Titans is that history is written into a neural memory function that can be updated at test time.

Classic recurrent memory might be a vector:

$$
h_t
$$

or a matrix:

$$
S_t
$$

Titans instead maintains something more like a function:

$$
M_t(\cdot)
$$

When new context arrives, the parameters of the memory function are updated at test time:

$$
M_{t-1} \rightarrow M_t
$$

So history is preserved not merely as tokens or a fixed hidden state, but by changing the very function used for subsequent queries. Memory is thereby directly connected to online learning.

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/titans/fig1_titans_memory_intuition.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  Figure 1: Intuition for memory in Titans
</div>

---

## 3. Surprise and Prediction Error

Reading this as key-value memory:

$$
k_t \rightarrow v_t
$$

memory first predicts the value from the key:

$$
\hat{v}_t = M_{t-1}(k_t)
$$

and then compares the prediction against the truth:

$$
e_t = v_t - \hat{v}_t
$$

The larger the error, the less predictable the current information is relative to existing memory, and the stronger the update should be:

```text
harder to predict → more surprising → more worth writing to memory
```

This form is clearly continuous with the Delta Rule, associative memory in linear attention, and TTT-style fast weights. The important change in Titans is not inventing a new kind of memory out of nowhere, but generalizing a linear memory matrix into a higher-capacity neural network.

```text
linear memory map → neural memory function
```

---

## 4. Parameter Updates: A Shared Mechanism for Memory and Abstraction

In ordinary training we update model parameters:

$$
\theta \leftarrow \theta - \eta \nabla_{\theta} L
$$

In Titans, test-time memory also updates parameters:

$$
\theta_{\mathrm{memory}}
\leftarrow
\theta_{\mathrm{memory}}
-
\eta \nabla_{\theta_{\mathrm{memory}}} L
$$

From an optimization standpoint, there is no sharp boundary between training-time learning and test-time memory. Both drive parameter updates via an objective. The differences more likely come from:

```text
different data scope
different time scale
different reset boundaries
different objectives
updating the main model vs. the memory parameters
```

So **memorization and abstraction need not correspond to two different learning mechanisms**. The same gradient update can produce representations at different levels:

```text
specific experience → memorization
shared structure    → abstraction
```

This is not equating memory with abstraction. A parameter update only specifies the learning process; it does not by itself determine whether the result is exact memory or abstract structure. That depends on the bottleneck, regularization, data scale, objective, and inductive bias.

---

## 5. Abstraction Pressure in Next-Token Prediction

A language model does not simply predict one token from another token; it predicts the next token given a history:

```text
a whole span of context → the next token
```

That is, the model predicts the future within an enormous combinatorial context space. If the vocabulary size is $V$ and the context length is $N$, the possible context space is very large:

$$
V^N
$$

Real language is of course not uniformly distributed, but the intuition holds: the model faces a huge conditional-prediction space.

As the training data grows, the cost of memorizing every context-to-output mapping term by term rises quickly, so the model is under stronger pressure to reuse structure. It has to learn grammar, semantics, world knowledge, long-range dependencies, and reusable statistical regularities.

Next-token prediction cannot eliminate memorization, but it provides an important abstraction pressure:

```text
learning reusable predictive structure from many specific sequences
```

---

## 6. Is Associative Retrieval Enough?

Titans-style memory can be written roughly as:

$$
M(k_t) \approx v_t
$$

i.e., given a key, recover a value from the memory function. This is naturally aligned with retrieval (at inference, the query reads $M(q_t)$), but its objective is clearly biased toward exact association:

$$
k \rightarrow v
$$

Exact association is effective for retrieval, but it does not automatically produce abstraction.

If a long-running model has experienced many events $e_1, e_2, \ldots, e_n$, exact memory preserves the events themselves:

$$
e_1, e_2, \ldots, e_n
$$

whereas what abstraction wants to form is:

$$
z = f(e_1, e_2, \ldots, e_n)
$$

where $z$ may not be any original event, but the structure behind those events.

So there is an unresolved shift in the goal of long-term memory:

> from preserving "what happened" toward forming "what these events jointly imply."

I call this direction **Test-Time Abstraction**.

---

## 7. Higher-Order Memory: A Poor Direction

The most natural idea is: since a single $k_t \rightarrow v_t$ is too simple, can we let memory look at a longer history?

$$
k_1, k_2, \ldots, k_t \rightarrow v_t
$$

or even:

$$
k_1, k_2, \ldots, k_t \rightarrow v_{t+1}
$$

On the surface this resembles sequence prediction, but it has a direct complexity contradiction. We introduced memory in the first place because the full history is too long to re-attend to every time; if updating memory again requires re-reading the full history, we have come full circle. And it easily leads to:

```text
first-order memory isn't enough
introduce second-order memory
second-order memory needs third-order memory
eventually: memory of memory of memory
```

Such memory-of-memory structures tend to turn the long-history problem into a hierarchical recursion problem, rather than actually solving it. A more sensible principle might be:

```text
queries should be cheap; abstraction should happen during update / consolidation
```

Retrieval should stay low-cost; compression and abstraction of history should mainly happen during update / consolidation. Memory should not re-read the past on every query, but progressively absorb, compress, and reorganize the past through continual updates.

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/titans/fig2_higher_order_memory_recursive.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  Figure 2: Why second/third-order memory tends to nest, and a more sensible route
</div>

---

## 8. Constraint Mechanisms Already in Titans

Titans is not oblivious to this. Its momentum, forgetting / gate, and weight-decay-like mechanisms can all be read as constraints on memory dynamics:

```text
momentum: updates are not fully dominated by a single sample
forgetting / gate: control what is retained vs. decayed
weight-decay-like: limit unbounded parameter growth
chunk / segment: make memory updates chunk-parallel
```

These mechanisms do not guarantee abstraction, but they already go beyond static writing. They do something closer to online optimization:

```text
new information + old memory + update dynamics + forgetting → new memory
```

One of the contributions of Titans is pushing memory from a "storage structure" toward a "dynamic learning process." Whether dynamic learning suffices to form abstraction still depends on the objective and capacity constraints.

---

## 9. Three Ways to Incorporate Memory

Titans discusses several ways to plug memory into the model. The three structures emphasize, respectively, context extension, representation modulation, and intermediate-layer transformation.

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/titans/fig3_memory_incorporation_comparison.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  Figure 3: Three ways Titans uses memory
</div>

### 9.1 Memory as Context: most natural, but chunk boundaries feel jagged

Memory as Context roughly concatenates several kinds of tokens:

```text
persistent memory + retrieved long-term memory + current segment
```

and then lets the current segment attend over this extended context, preserving a "retrieved memory + local attention" structure.

Its main issue comes from segment / chunk boundaries: a position at the start of a segment sees few tokens within the current segment; later positions see more; the next segment starts over. This introduces a kind of "jaggedness" at boundaries. Possible mitigations include:

```text
overlapping chunks
shifted windows
multi-scale segments
boundary smoothing
cross-segment residual memory
```

So while chunking is necessary in practice, it is not free.

### 9.2 Memory as Gate: more like modulation than retrieval

If memory outputs $m_t = M(q_t)$ and then acts on the current representation mainly through a gate:

$$
y_t = g(m_t) \odot h_t
$$

then it feels more like memory-conditioned modulation than explicit content retrieval. This structure is good for controlling information flow, but as the primary read path for long-term memory its content expressiveness is relatively indirect.

### 9.3 Memory as Layer: may blur exact information first

The third form is more like:

$$
x_t \rightarrow M(x_t) \rightarrow \mathrm{LocalAttention}
$$

The potential issue: local attention no longer operates directly on the raw representation, but on a memory-transformed representation. If $M(\cdot)$ introduces information loss, smoothing, noise, or instability from online updates, then even very precise downstream attention can only operate precisely on already-transformed information. So it may create a representation bottleneck.

---

## 10. Persistent Memory and Learnable Prompts

I prefer to read the persistent memory in Titans as a learnable prompt / soft prompt / prefix tokens: a set of relatively input-independent parameters learned during training. Unlike long-term neural memory, it is not continually updated within the current context; it is more like a persistent set of task-level memory tokens. From this angle, persistent memory is directly connected to existing learnable-prompt paradigms, and need not be treated as a separate long-term memory mechanism.

---

## 11. Shared Conditional Memory

If every layer and every head has its own independent memory network $M_{\ell,h}$, the memory system becomes very large; more importantly, each head having its own private memory will not necessarily form shared abstractions across heads and layers.

An alternative is to share the core parameters of the memory network, $M_{\mathrm{shared}}$. But the query / key / value of different layers and heads may not lie in the same representation space, so we cannot feed all queries into a single memory directly. A more sensible structure is:

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

that is:

```text
private space → shared latent memory space → private space
```

This structure has several potential benefits:

```text
fewer parameters than per-layer, per-head private memory
shared memory dynamics across heads / layers
a capacity bottleneck via the shared core
forcing experience from different sources into a common latent memory space
possibly stronger abstraction pressure
```

Compared with higher-order memory, this direction does not increase the query's dependence on history; instead it strengthens abstraction pressure through shared parameters, space alignment, and capacity limits.

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/titans/fig4_shared_conditional_memory_arch.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>
<div class="caption">
  Figure 4: A sketch of the Shared Conditional Memory architecture
</div>

---

## 12. Relation to Chunkwise Recurrent Memory

I have also thought about chunkwise recurrent memory: use local attention for the current chunk while passing history across chunks as a compressed memory (see [llama2Rnn.c](https://github.com/siyuanseever/llama2Rnn.c)). After reading Titans, such a structure can be written as a recurrent memory transition:

$$
M_t = F_{\theta}(M_{t-1}, X_t)
$$

If $F_{\theta}$ internally contains attention-like merges, a nonlinear FFN, and compression, then it is not a simple linear recurrence but a nonlinear recurrent memory system. I will not expand on this here. The relationships among explicit memory slots, parametric memory, chunkwise recurrence, and test-time abstraction deserve a separate discussion.

---

## 13. From Test-Time Memory to Test-Time Abstraction

Titans offers an important view: memory can be a learning system continually updated at test time, not merely a static storage structure. Yet much of today's test-time memory still targets exact association as its main objective.

A long-running adaptive system needs to answer two questions at once: what happened in the past? and, what do those past events jointly imply? Exact memory preserves episodes; abstraction forms schemas:

```text
episodes → patterns → schema → future generalization
```

So the more worthwhile question is not "how to let the model store more tokens," but:

> how to let the model form reusable abstractions from context at test time?

One possible research hypothesis:

> Test-time memory should not only optimize for exact associative retrieval. A long-term adaptive system may require bottlenecks, shared parameterization, regularization, predictive pressure, and consolidation mechanisms that transform historical context into reusable abstractions.

The whole thing compresses to a single question:

```text
Can a model learn to abstract at test time?
```

Can a model not merely remember the world, but—through continually experiencing it—slowly form concepts of its own?
