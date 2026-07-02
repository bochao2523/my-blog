---
title: 'Transformer 的基本介绍'
description: '从背景动机到核心组件，理解 Transformer 架构为何成为现代大语言模型的基石。'
series: 'llm-fundamentals'
pubDate: 'Jun 24 2026'
---

## Part 0：为什么需要 Transformer？

处理序列数据的主流方案曾是 `RNN` 和 `CNN`，但两者在**长距离依赖**和**并行训练**上都存在瓶颈。`Transformer` 用 `Self-Attention` 同时绕开了这两条路。

### RNN 的劣势

`RNN` 按时间步递推：$h_t$ 依赖 $h_{t-1}$，信息逐步传递。

- **无法并行**：$t$ 时刻必须等 $t-1$ 完成，与 `GPU` 的并行计算能力不匹配
- **长距离依赖难学**：位置 1 到位置 $n$ 需经 $O(n)$ 步传递，梯度易衰减（`LSTM` / `GRU` 只是缓解）

### CNN 的劣势

一维卷积在序列上滑动，相邻 token 局部交互；堆叠或膨胀卷积可扩大感受野。

- **感受野增长慢**：远距离 token 需多层间接交互，路径长度 $O(n)$ 或 $O(\log n)$
- **局部偏置**：卷积核固定，难以灵活建模任意距离的语义关联

### 三者对比

| | 并行训练 | 路径长度 | 核心复杂度 |
|---|---|---|---|
| `RNN` | 差（逐步递推） | $O(n)$ | 每步串行，无法 batch 时间维 |
| `CNN` | 好 | $O(\log n)$ ~ $O(n)$ | 局部卷积，感受野需堆叠 |
| `Transformer` | 好 | $O(1)$ | 注意力 $O(n^2)$，但可并行 |

<aside class="callout">
<p><strong>核心结论</strong>：Self-Attention 让序列中任意两个位置<strong>一步直达</strong>，路径长度从 $O(n)$ 降为 $O(1)$——这是 Transformer 取代 RNN 的关键。</p>
</aside>

*Attention Is All You Need* 的核心思路：去掉循环和卷积，用 `Attention` 让任意两个位置直接交互；权重由输入动态决定，而非固定递推或卷积核。

## Part 1：Transformer 架构概览

原始 `Transformer` 由 `Encoder` 和 `Decoder` 堆叠组成，常用于 seq2seq 任务（如机器翻译）。后来的演化不是简单「砍掉一半」——`GPT` 这类 Decoder-only 模型去掉了 `Cross-Attention`，变成**只有因果 Self-Attn + FFN 的独立 stack**，与原始 Decoder 结构并不相同。

### 整体结构

<figure class="arch-fig">
<div class="arch-stack">
  <span class="arch-label">输入序列（源语言）</span>
  <span class="arch-arrow-down">↓</span>
  <div class="arch-module arch-encoder">
    <strong>Encoder × N 层</strong><br>Self-Attn → FFN（每层重复）
  </div>
  <span class="arch-arrow-down">↓ 编码表示（供 Cross-Attn 的 K, V）</span>
  <div class="arch-merge">
    <div class="arch-merge-input">已生成序列<br><small>（训练时右移一位）</small></div>
    <span class="arch-arrow-side">＋</span>
    <div class="arch-merge-input">编码表示<br><small>（来自 Encoder）</small></div>
  </div>
  <span class="arch-arrow-down">↓ 两路汇入</span>
  <div class="arch-module arch-decoder">
    <strong>Decoder × N 层</strong><br>Masked Self-Attn → Cross-Attn → FFN
  </div>
  <span class="arch-arrow-down">↓</span>
  <span class="arch-label">输出序列（目标语言）</span>
</div>
<figcaption>Decoder 每层同时接收两路输入：自身序列经 Masked Self-Attn，源信息经 Cross-Attn 从 Encoder 注入</figcaption>
</figure>

- `Encoder`：双向读入源序列，输出每个位置的上下文表示
- `Decoder`：以编码表示为条件，自回归地生成目标序列

### Block 里有什么？

每个子层都是 **Attention（或 FFN）→ 残差 → LayerNorm** 的节奏。Attention 管 token 间**混合**，FFN 管每个位置单独**加工**——这是 Block 的核心分工。

#### 四个组件，分别让模型学到了什么

单看"由哪几个模块组成"还不够，关键是搞清楚**每个组件到底在承担什么样的学习任务**——它们分工完全不同，缺一个都不行：

<div class="gridcards">
<div class="gridcard"><b>Self-Attention</b><br>学的是<strong>token 之间的关系</strong>：这个词该重点参考句子里哪些别的词（语法结构、指代、长距离依赖）。它只负责「按相关性重新组合」已有信息，本身不产生新特征。</div>
<div class="gridcard"><b>FFN</b><br>学的是<strong>每个位置自己的特征变换 / 知识存储</strong>：Attention 混合完上下文后，FFN 对每个 token 独立做非线性加工，很多可解释性研究认为这里存的是事实性知识——也是为什么 FFN 占了大部分参数量。</div>
<div class="gridcard"><b>残差连接</b><br>学的是<strong>增量修正</strong>，而不是从零重构：每层只需要学「在原有表示上改进多少」（$\Delta x$），主干信息始终能直接跳过子层往后传，这是深层网络能训动的前提。</div>
<div class="gridcard"><b>LayerNorm</b><br>不学「语义」，学的是<strong>让每层输入的数值分布保持稳定</strong>（拉回均值 0、方差 1，再配一对可学习的缩放/偏移）。它是训练稳定器，不直接贡献表达能力，但没它深层网络几乎训不动。</div>
</div>

拆开来看每一步在做什么：

**Self-Attention 具体在学什么**：对每个 token，动态算出一组权重去聚合其它 token 的信息——本质是在学「谁和谁相关、相关到什么程度」。同一个词在不同句子里，权重分布完全不同（比如「苹果」在"苹果很好吃"和"苹果发布新品"里关注的邻居不一样），这正是上下文建模的核心。

**FFN 具体在学什么**：结构是「升维（通常 $4\times$）→ 非线性激活 → 降维」，逐位置独立处理，token 之间互不影响。可以把它类比成一个 key-value 记忆库：升维那一层像是一堆"探测器"，检测当前 token 的表示是否匹配某种模式（key），激活函数决定哪些探测器被触发，降维那一层再把对应的内容（value）取出来叠加回去。Attention 负责"信息从哪来"，FFN 负责"信息该怎么被解读和存储"。

**残差连接具体在学什么**：把子层输入直接加到输出上（$x + \text{Sublayer}(x)$），子层因此只需要学一个相对于输入的"修正量"，而不是重新学出一份完整表示。堆 $N$ 层时，梯度可以沿着这条「捷径」直接回传到浅层，避免深层网络梯度消失——**这是模型能堆到几十上百层的前提**。

**LayerNorm 具体在学什么**：归一化每一层的输入分布，把数值稳定在一个合理范围内，防止某层输出的量级随深度堆叠而爆炸或塌缩。原论文用 **Post-LN**（子层算完再 Norm）；现代模型（`GPT`、`LLaMA`）几乎全改 **Pre-LN**（先 Norm 再进子层），因为 Post-LN 在深层容易训练不稳、需要较长 warmup。

<aside class="callout">
<p><strong>合起来看</strong>：Self-Attention 决定「看谁」，FFN 决定「看到之后怎么加工/记住什么」，残差连接保证「信息和梯度能传得下去」，LayerNorm 保证「传的过程数值不失控」。四者分别解决「关系建模」「特征存储」「深度可训练」「数值稳定」四个完全不同的问题，缺一个都不是完整的 Transformer Block。</p>
</aside>

**Encoder Layer**（每层）：

1. `Multi-Head Self-Attention` — 序列内部全局交互
2. `FFN`
3. 残差 + `LayerNorm`（现代实现多为 Pre-LN）

**Decoder Layer**（每层）多一个子层：

1. `Masked Self-Attention` — 因果掩码
2. `Cross-Attention` — Q 来自 Decoder，K/V 来自 Encoder
3. `FFN` + 残差 + `LayerNorm`

### 三种 Attention

| 类型 | Q 来源 | K, V 来源 | 掩码 | 作用 |
|---|---|---|---|---|
| Encoder Self-Attn | Encoder | Encoder | 无 | 源序列双向建模 |
| Decoder Self-Attn | Decoder | Decoder | 因果 | 生成时只看过去 |
| Cross-Attn | Decoder | Encoder | 无 | 源信息注入生成 |

<aside class="callout">
<p><strong>因果掩码为什么存在？</strong>训练时整句目标序列并行喂入，但推理时只能从左到右逐个生成。掩码强制位置 $t$ 不能「偷看」$t+1$ 之后的答案——这样<strong>并行训练</strong>和<strong>自回归推理</strong>共用同一套权重，不会训练/测试行为不一致。这是理解 Decoder（以及 Decoder-only）的前置。</p>
</aside>

<aside class="callout">
<p><strong>直觉记忆</strong>：Encoder Self-Attn「读懂输入」→ Decoder Self-Attn「连贯往下写」→ Cross-Attn「写的时候回头看原文」。</p>
</aside>

### 三种架构的经典运用

三者区别的本质在两个维度：**用什么注意力掩码** + **用什么预训练目标**。

| 架构 | 注意力掩码 | 预训练目标 | 代表模型 | 典型任务 |
|---|---|---|---|---|
| Encoder-only | 双向（无掩码） | MLM | `BERT` | 分类、NER、检索 |
| Decoder-only | 因果（单向） | Next-token | `GPT` 系列 | 生成、对话、代码 |
| Encoder-Decoder | Enc 双向 + Dec 因果 | Span corruption 等 | `T5`、`BART` | 翻译、摘要 |

换掩码 + 换目标 = 换架构。这也是为什么 `GPT` 能从同一个因果 stack 统一做生成，而 `BERT` 只能做理解、需要额外头才能生成。

## Part 2：为什么 Decoder-only 成为主流？

Part 1 提到三种架构各有场景，但当前 LLM 几乎清一色是 **Decoder-only**（`GPT`、`LLaMA`、`Qwen`、`Claude` 等）。原因归结为：**一个简单目标 + 一种统一接口 + 可预测的规模化路径**。

### 训练目标简单且可扩展

Decoder-only 只做一件事——**因果语言建模**：

$$
P(x_t \mid x_1, x_2, \ldots, x_{t-1})
$$

不需要 `BERT` 的 MLM/NSP，也不需要维护两套参数和 `Cross-Attention`。目标单一，更容易 scale。

### 生成是「超集」

几乎所有任务都能写成「续写」：

| 任务 | 转化方式 |
|---|---|
| 分类 | 生成标签词（`正面` / `负面`） |
| 问答 | 生成答案文本 |
| 翻译 | 生成目标语言句子 |
| 代码 | 生成补全片段 |

`GPT-3` 的 few-shot prompting 证明：同一模型、同一 forward，换 prompt 即可多任务，无需改结构。

### 与语料 & Scaling Law 同向

互联网文本天然是**从左到右的序列**，因果建模直接拟合；`CLM` 数据门槛最低。`GPT-3` 之后 scaling law 表明 loss 与参数量、数据量、算力呈幂律关系——这条路线在 Decoder-only 上验证最充分。

<aside class="callout">
<p><strong>赢家通吃</strong>：模型越大，涌现 in-context learning、chain-of-thought 等能力；工程生态（`vLLM`、`RLHF`）也全部围绕生成式模型构建。</p>
</aside>

### 另外两种架构去哪了？

| 架构 | 现状 |
|---|---|
| Encoder-only | 仍用于 Embedding（`BGE`）、重排序；通用 LLM 主战场让位 |
| Encoder-Decoder | `T5` 在翻译/摘要仍有价值；通用对话 LLM 少见 |
| Decoder-only | 通用 LLM、Agent、代码助手的主流选择 |

`RAG` 里仍需要 Encoder 做向量化，专业翻译也用 seq2seq——但在「一个模型什么都干」的路线下，**Decoder-only 用统一生成接口 + 规模化，赢下了主赛道**。
