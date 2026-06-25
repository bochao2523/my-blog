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

## Part 2：为什么 Decoder-only 成为主流？

Part 1 提到三种架构各有适用场景，但当前 LLM 几乎清一色是 **Decoder-only**（GPT、LLaMA、Qwen、Claude 等）。原因可以归结为：**一个简单目标 + 一种统一接口 + 可预测的规模化路径**。

### 1. 训练目标简单且可扩展

Decoder-only 只做一件事：**因果语言建模**——给定前文，预测下一个 token。

$$
P(x_t \mid x_1, x_2, \ldots, x_{t-1})
$$

不需要像 BERT 那样设计 MLM、NSP 等辅助任务，也不需要 Encoder-Decoder 那样维护两套参数和 Cross-Attention。目标单一，实现干净，**更容易把算力和数据砸上去做 scale**。

### 2. 生成是「超集」，理解可以折算成生成

Encoder-only 擅长判别（分类、匹配），但不天然会生成；Encoder-Decoder 擅长 seq2seq，但架构更重。

Decoder-only 反过来：**几乎所有任务都能写成「续写」**——

- 分类 → 生成标签词（`正面` / `负面`）
- 问答 → 生成答案文本
- 翻译 → 生成目标语言句子
- 代码 → 生成补全片段

GPT-3 的 few-shot prompting 证明了：同一个模型、同一种 forward，换 prompt 就能做多种任务，无需为每个任务改结构或加分类头。

### 3. 与互联网语料天然匹配

大规模预训练吃的是海量无标注文本，而网页、书籍、代码本身就是**从左到右的序列**。因果建模直接拟合这种数据分布，不需要构造「被 mask 的句子对」或「平行语料」。

Encoder-Decoder 的翻译/摘要场景需要成对数据；BERT 的 MLM 需要专门的数据处理。Decoder-only 的 CLM **数据门槛最低**，适合暴力 scale。

### 4. Scaling Law 验证的是这条路

GPT-3 之后的研究（Kaplan、Chinchilla 等）表明：在足够大的规模下，**loss 与参数量、数据量、算力呈可预测的幂律关系**。产业界据此持续加码——而这条 scaling 路线走得最顺的，正是 Decoder-only + CLM。

结果是：模型越大，通用能力越强，涌现 in-context learning、chain-of-thought 等现象。**赢家通吃**，其他架构在「通用大模型」赛道上难以抗衡。

### 5. 工程与生态的惯性

推理链路统一（自回归逐 token 生成），训练框架成熟（Megatron、DeepSpeed、vLLM 等），RLHF / DPO 等对齐方法也围绕生成式模型构建。ChatGPT 爆火之后，开源社区（LLaMA、Mistral）和产业界全部押注同一路线，形成**路径依赖**。

### 另外两种架构去哪了？

| 架构 | 现状 |
|---|---|
| **Encoder-only** | 仍用于 Embedding（BERT、BGE）、重排序、判别式小模型；通用 LLM 主战场已让位 |
| **Encoder-Decoder** | T5、BART 在翻译/摘要等任务仍有价值；Flan-T5 等做指令微调，但通用对话 LLM 少见 |
| **Decoder-only** | 通用 LLM、对话、Agent、代码助手的主流选择 |

不是说 Encoder 或 Encoder-Decoder 没用——检索增强（RAG）里仍需要 Encoder 做向量化，专业翻译也会用 seq2seq。但在「一个模型什么都干」的路线下，**Decoder-only 用统一生成接口 + 规模化，赢下了主赛道**。

