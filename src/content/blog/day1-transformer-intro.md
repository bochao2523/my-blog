---
title: 'Transformer 的基本介绍'
description: '从背景动机到核心组件，理解 Transformer 架构为何成为现代大语言模型的基石。'
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

原始 `Transformer` 由 `Encoder` 和 `Decoder` 两部分组成，常用于 seq2seq 任务（如机器翻译）。现代大模型往往只保留其中一半，演化为三种主流形态。

### 整体结构

<figure class="arch-fig">
<div class="arch-flow">
<span class="arch-box-muted">输入序列</span>
<span class="arch-arrow">→</span>
<span class="arch-box">Encoder × N</span>
<span class="arch-arrow">→</span>
<span class="arch-box-muted">编码表示</span>
<span class="arch-arrow">→</span>
<span class="arch-box">Decoder × N</span>
<span class="arch-arrow">→</span>
<span class="arch-box-muted">输出序列</span>
</div>
<figcaption>Transformer 整体数据流：Encoder 双向理解输入，Decoder 自回归生成输出</figcaption>
</figure>

- `Encoder`：读入源序列，输出每个位置的上下文表示
- `Decoder`：以编码表示为条件，自回归地生成目标序列

### Block 里有什么？

**Encoder Layer**（每层）：

1. `Multi-Head Self-Attention` — 序列内部全局交互
2. `Feed-Forward Network`（`FFN`）— 逐位置非线性变换
3. 残差连接 + `Layer Norm`

**Decoder Layer**（每层）多一个子层：

1. `Masked Self-Attention` — 因果掩码，只看当前及之前
2. `Cross-Attention` — Q 来自 Decoder，K/V 来自 Encoder
3. `FFN` + 残差 + `Layer Norm`

### 三种 Attention

| 类型 | Q 来源 | K, V 来源 | 掩码 | 作用 |
|---|---|---|---|---|
| Encoder Self-Attn | Encoder | Encoder | 无 | 源序列双向建模 |
| Decoder Self-Attn | Decoder | Decoder | 因果 | 生成时只看过去 |
| Cross-Attn | Decoder | Encoder | 无 | 源信息注入生成 |

<aside class="callout">
<p><strong>直觉记忆</strong>：Encoder Self-Attn「读懂输入」→ Decoder Self-Attn「连贯往下写」→ Cross-Attn「写的时候回头看原文」。</p>
</aside>

### 三种架构的经典运用

| 架构 | 代表模型 | 典型任务 |
|---|---|---|
| Encoder-only | `BERT` | 分类、NER、检索 |
| Decoder-only | `GPT` 系列 | 生成、对话、代码 |
| Encoder-Decoder | `T5`、`BART` | 翻译、摘要 |

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
