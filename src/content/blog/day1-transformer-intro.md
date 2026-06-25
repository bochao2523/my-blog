---
title: 'Transformer 的基本介绍'
description: '从背景动机到核心组件，理解 Transformer 架构为何成为现代大语言模型的基石。'
pubDate: 'Jun 24 2026'
---

## Part 0：为什么需要 Transformer？

处理序列数据的主流方案曾是 **RNN** 和 **CNN**，但两者在**长距离依赖**和**并行训练**上都存在瓶颈。Transformer 用 Self-Attention 同时绕开了这两条路。

### RNN 的劣势

RNN 按时间步递推：$h_t$ 依赖 $h_{t-1}$，信息逐步传递。

- **无法并行**：$t$ 时刻必须等 $t-1$ 完成，与 GPU 的并行计算能力不匹配
- **长距离依赖难学**：位置 1 到位置 $n$ 需经 $O(n)$ 步传递，梯度易衰减（LSTM/GRU 只是缓解）

### CNN 的劣势

一维卷积在序列上滑动，相邻 token 局部交互；堆叠或膨胀卷积可扩大感受野。

- **感受野增长慢**：远距离 token 需多层间接交互，路径长度 $O(n)$ 或 $O(\log n)$
- **局部偏置**：卷积核固定，难以灵活建模任意距离的语义关联

### Transformer 解决了什么？

*Attention Is All You Need* 的核心思路：**去掉循环和卷积，用 Attention 让任意两个位置直接交互。**

| | RNN | CNN | Transformer |
|---|---|---|---|
| 并行训练 | 差 | 好 | 好 |
| 长距离依赖 | 逐步传递 | 间接、感受野有限 | 任意位置 $O(1)$ 可达 |
| 权重 | 固定递推 | 固定卷积核 | 由输入动态决定 |

## Part 1：Transformer 架构概览

原始 Transformer（*Attention Is All You Need*）由 **Encoder** 和 **Decoder** 两部分组成，常用于序列到序列任务（如机器翻译）。现代大模型则往往只保留其中一半，演化为三种主流形态。

### 整体结构

```
输入序列 ──→ [Encoder × N] ──→ 编码表示 ──→ [Decoder × N] ──→ 输出序列
                  ↑                              ↑
              Self-Attn                    Masked Self-Attn
                                             + Cross-Attn
```

- **Encoder**：读入源序列，输出每个位置的上下文表示
- **Decoder**：以编码表示为条件，自回归地生成目标序列

### Block 里有什么？

**Encoder Layer**（每层）：

1. **Multi-Head Self-Attention** — 序列内部全局交互
2. **Feed-Forward Network (FFN)** — 逐位置非线性变换
3. 每个子层外包裹 **残差连接 + Layer Norm**

**Decoder Layer**（每层）在 Encoder 基础上多一个子层：

1. **Masked Multi-Head Self-Attention** — 只能看到当前及之前的位置（因果掩码）
2. **Cross-Attention** — Query 来自 Decoder，Key/Value 来自 Encoder 输出
3. **FFN** + 残差 + Layer Norm

### 三种 Attention

| 类型 | Q 来源 | K, V 来源 | 掩码 | 作用 |
|---|---|---|---|---|
| **Encoder Self-Attention** | Encoder | Encoder | 无 | 源序列双向建模 |
| **Decoder Self-Attention** | Decoder | Decoder | 因果掩码 | 生成时只看过去 |
| **Cross-Attention** | Decoder | Encoder | 无 | 将源信息注入生成过程 |

直觉上：Encoder Self-Attn 负责「读懂输入」；Decoder Self-Attn 负责「连贯地往下写」；Cross-Attn 负责「写的时候回头看原文」。

### 三种架构的经典运用

| 架构 | 代表模型 | 典型任务 |
|---|---|---|
| **Encoder-only** | BERT | 文本分类、NER、问答、语义检索 |
| **Decoder-only** | GPT 系列 | 文本生成、对话、代码补全 |
| **Encoder-Decoder** | T5、BART、原始 Transformer | 翻译、摘要、问答生成 |

- **Encoder-only**：双向上下文，适合「理解」类任务，不直接做自回归生成
- **Decoder-only**：因果建模，一个架构统一生成与对话，是当前 LLM 的主流路线
- **Encoder-Decoder**：输入输出分离，适合明确的 seq2seq 映射

