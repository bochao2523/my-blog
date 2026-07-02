---
title: '多头注意力机制'
description: '从单头 Self-Attention 到 Multi-Head Attention，理解多头并行如何捕捉不同子空间的语义关系。'
series: 'llm-fundamentals'
pubDate: 'Jun 26 2026'
---

## Part 1：什么是多头注意力？它和单头有什么不同？

<p class="lead"><code>Multi-Head Attention</code>（多头注意力）不是一个新公式，而是把<strong>同一套 Self-Attention 并行做很多份</strong>，每一份用自己独立的 <code>Q</code>、<code>K</code>、<code>V</code> 投影，在不同的「子空间」里各自去找关系，最后把所有结果拼起来再做一次线性变换。</p>

<aside class="callout">
<p><strong>一句话抓住本质</strong>：单头 Attention 只能学到「一种」token 间的关系模式；多头就是让模型同时用多组「眼睛」去看同一个句子，一组看语法结构、一组看语义关联、一组看指代关系……最后把所有视角综合起来。</p>
</aside>

### 单头注意力的局限

回顾一下单头 Self-Attention：一个 `token` 的 `Q` 去跟所有 `token` 的 `K` 算相似度，过 `softmax` 得到一组权重，再加权聚合 `V`。

问题在于：**这一整套流程只有一组权重矩阵 $W_Q, W_K, W_V$**，也就只能学出「一种」相关性的度量方式。而现实中，一个词和其它词之间往往同时存在好几种不同类型的关系，比如：

<div class="gridcards">
<div class="gridcard"><b>语法关系</b><br>「猫」和「追」——谁是主语、谁是动词。</div>
<div class="gridcard"><b>语义关系</b><br>「苹果」和「公司」——语义上强相关。</div>
<div class="gridcard"><b>指代关系</b><br>「它」应该指向前面哪个名词。</div>
<div class="gridcard"><b>位置/局部关系</b><br>相邻词之间天然的顺序依赖。</div>
</div>

单头只有一组 `softmax` 权重，这些不同类型的相关性会被压缩、平均到同一个分布里，模型很难同时把每一种都表达清楚——这就是单头的表达能力瓶颈。

### 多头怎么做

多头的思路很直接：**不要只投影一次，投影 $h$ 次**，每次都用一套独立的、更小的权重矩阵：

$$
\text{head}_i = \text{Attention}(QW_i^Q,\ KW_i^K,\ VW_i^V), \quad i = 1, \dots, h
$$

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\,W^O
$$

其中每个头的投影维度 $d_k = d_{model} / h$，即把原来一份「大」的 $Q,K,V$ 拆成 $h$ 份「小」的，各自独立算一遍完整的 Scaled Dot-Product Attention，互不干扰；算完之后再拼接（`Concat`）回 $d_{model}$ 维，过一个输出投影 $W^O$ 把各头的结果重新融合。

<div class="cycle">
<span class="cycle-node">输入 X 投影 h 次</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">每个头独立做 Attention</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">Concat 拼接所有头</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">W^O 线性融合</span>
<span class="cycle-note">h 组「不同视角」的结果，最后被融合成一份输出</span>
</div>

### 单头 vs 多头，核心区别

<div class="compare">
<div class="compare-col is-workflow">
<span class="compare-head">单头 Attention</span>
<p>只有<strong>一组</strong> $W_Q, W_K, W_V$，在完整的 $d_{model}$ 维空间里算一次相似度、做一次加权聚合——只能捕捉一种子空间关系。</p>
</div>
<div class="compare-col is-agent">
<span class="compare-head">多头 Attention</span>
<p>拆成 $h$ 组更小的 $W_Q^i, W_K^i, W_V^i$（每组维度 $d_{model}/h$），<strong>并行</strong>做 $h$ 次独立的 Attention，再拼接融合——能同时捕捉多种子空间关系。</p>
</div>
</div>

值得注意的是：多头并不是「算力翻 $h$ 倍」。因为每个头的维度被压缩成了 $d_{model}/h$，$h$ 个头加起来的参数量和计算量，跟一个跑在完整 $d_{model}$ 维度上的单头基本持平——**多头的收益几乎是「白拿」的，代价不是算力，而是工程上多了一次 reshape 和拼接**。

## Part 2：为什么多头有效？

<p class="lead">多头有效的根本原因是：$W_Q, W_K, W_V$ 是<strong>随机初始化、独立训练</strong>的，每个头从一开始就被投影到不同的子空间里，梯度下降会自然地让不同头去捕捉不同类型的 token 关系——这不是设计出来「分配任务」，而是优化过程中自发「分化」出来的。</p>

### 一个类比：审稿人

<aside class="callout">
<p><strong>类比</strong>：单头 Attention 像是只请了<strong>一位审稿人</strong>看论文，他一个人要同时判断语法、逻辑、创新性、实验设计——精力有限，容易顾此失彼。多头 Attention 像是请了 $h$ 位<strong>不同专长</strong>的审稿人，一人专注语法、一人专注实验、一人专注创新性，最后把所有人的意见汇总（Concat + $W^O$）成一份综合评审。</p>
</aside>

### 子空间分化：从数学上看

每个头看到的输入是同一个 $X$，但经过各自独立的 $W_Q^i, W_K^i, W_V^i$ 投影后，落在了完全不同的 $d_k$ 维子空间里。因为：

- 初始化不同（随机权重起点不同）
- 训练时每个头独立地对损失函数求梯度、独立更新

两个头即使面对同样的输入，也会朝着不同的方向收敛——**只要某个头稍微学到了一点「捕捉指代关系更有效」的信号，梯度就会强化这个方向，让它在这条路上越走越远**，其它头则在别的方向上分化。这是一种自组织的分工，而不是人为规定「头 1 负责语法、头 2 负责语义」。

### 实证观察：多头到底学到了什么

大量对 Transformer 的可解释性研究（如 BERT 的 attention 可视化）观察到，训练好的多头模型里，不同头确实呈现出可解释的分工模式：

<div class="gridcards">
<div class="gridcard"><b>句法头</b><br>权重主要集中在语法上直接相关的词（如主谓、动宾）之间。</div>
<div class="gridcard"><b>指代头</b><br>专门把代词的注意力权重打到它所指代的名词上。</div>
<div class="gridcard"><b>局部头</b><br>权重高度集中在相邻的前后几个 token，像一个局部窗口。</div>
<div class="gridcard"><b>全局头</b><br>权重分散在全句甚至固定关注句首 <code>[CLS]</code> 这样的特殊位置。</div>
</div>

也正因为这种分工是「训练出来」的而非「设计出来」的，不同任务、不同层的头分化模式并不完全一致——但普遍规律是：**浅层的头更偏向局部/句法，深层的头更偏向全局/语义**，这与堆叠多层 Self-Attention 时「感受野逐层扩大」的直觉是一致的。

### 从信息瓶颈角度理解

还可以换一个角度看：单头把全部 $d_{model}$ 维度都用来学「一种」相似度度量函数，其实是把大量表达能力浪费在了一个过窄的假设空间里。拆成 $h$ 个更小的子空间，相当于**给模型提供了 $h$ 种互相独立的假设，让它自己去挑每种假设该聚焦在什么信息上**——这在参数量几乎不变的前提下，显著提升了模型能表达的关系种类，这正是多头相比单头「白捡」收益的来源。

<aside class="callout">
<p><strong>一句话</strong>：多头有效不是因为算力更强，而是因为它把一个「大而模糊」的相似度函数拆成了「多个小而专注」的相似度函数，让分化和分工在训练中自然涌现。</p>
</aside>

## Part 3：手撕代码，逐步讲解

<p class="lead">下面不是直接甩一大段代码，而是把 <code>Multi-Head Attention</code> 拆成 5 个步骤，每一步都跟着形状（shape）变化走一遍，搞清楚 <code>reshape</code>/<code>transpose</code> 到底在干什么。</p>

### 先定好设定

在写代码之前，先把输入张量 `x` 的三个维度含义说清楚——后面所有 `shape` 推导都围着它们转：

<div class="gridcards">
<div class="gridcard"><b>batch_size（B）</b><br>一次「打包」进模型的样本数量。比如一次训练同时喂进去 2 条句子，<code>B=2</code>；各样本之间完全独立，互不影响。</div>
<div class="gridcard"><b>seq_len（T）</b><br>每条样本里 <code>token</code> 的个数，比如一句话被切成 5 个 token，<code>T=5</code>。Attention 就是在这 5 个 token 之间互相算相关性。</div>
<div class="gridcard"><b>d_model（d）</b><br>每个 <code>token</code> 用多少维的向量来表示，比如 <code>d_model=64</code>。这是模型的「隐藏维度」，贯穿整个 Transformer，多头注意力就是把这 64 维再拆成 <code>n_heads</code> 份。</div>
</div>

所以输入 `x` 的形状 `(B, T, d_model)` = `(2, 5, 64)` 读作：**2 条样本，每条样本 5 个 token，每个 token 用 64 维向量表示**。再定 `n_heads=8`，则每个头的维度 `d_k = 64 / 8 = 8`——每个头只分到 8 维，8 个头合起来正好还原 64 维。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

B, T, d_model, n_heads = 2, 5, 64, 8
d_k = d_model // n_heads   # 8
x = torch.randn(B, T, d_model)   # (2, 5, 64)
```

### 第 1 步：线性投影出 Q、K、V

<div class="stack">
<div class="stack-band stack-1"><b>做什么</b>三个独立的 <code>nn.Linear(d_model, d_model)</code>，分别把 <code>x</code> 投影成 Q、K、V。</div>
<div class="stack-band stack-2"><b>形状</b><code>(B, T, d_model)</code> → <code>(B, T, d_model)</code>，形状不变，只是内容被重新映射了。</div>
<div class="stack-band stack-3"><b>关键点</b>这里先不拆头，是在「完整维度」上一次性投影，拆头是下一步才做的事。</div>
</div>

```python
W_q = nn.Linear(d_model, d_model)
W_k = nn.Linear(d_model, d_model)
W_v = nn.Linear(d_model, d_model)

Q = W_q(x)   # (2, 5, 64)
K = W_k(x)   # (2, 5, 64)
V = W_v(x)   # (2, 5, 64)
```

### 第 2 步：拆成多头——reshape 的核心

这一步最容易看晕，拆开成两个动作：

```python
Q = Q.view(B, T, n_heads, d_k)      # (2, 5, 64) -> (2, 5, 8, 8)
Q = Q.transpose(1, 2)               # (2, 5, 8, 8) -> (2, 8, 5, 8)
```

<div class="compare">
<div class="compare-col is-workflow">
<span class="compare-head">view：切开最后一维</span>
<p><code>(B, T, d_model)</code> 里最后一维 <code>64</code> 被<strong>原样切成</strong> <code>(8, 8)</code>，即「8 个头 × 每头 8 维」。因为是连续内存的 reshape，前 8 个数就是 head 0，接下来 8 个数是 head 1……这要求 <code>d_model</code> 必须能被 <code>n_heads</code> 整除，否则切不整齐。</p>
</div>
<div class="compare-col is-agent">
<span class="compare-head">transpose：把头维度提前</span>
<p>切完是 <code>(B, T, n_heads, d_k)</code>，头维度排在第三位；<code>transpose(1, 2)</code> 把它换到第二位，变成 <code>(B, n_heads, T, d_k)</code>——这样每个头就成了独立的一个「批次」，方便下一步直接做批量矩阵乘法。</p>
</div>
</div>

K、V 做完全一样的操作：

```python
K = K.view(B, T, n_heads, d_k).transpose(1, 2)   # (2, 8, 5, 8)
V = V.view(B, T, n_heads, d_k).transpose(1, 2)   # (2, 8, 5, 8)
```

<aside class="callout">
<p><strong>直觉</strong>：现在 <code>(B, n_heads, T, d_k)</code> 里的 <code>n_heads</code> 就等价于多了一个 batch 维度。PyTorch 的 <code>matmul</code> 天然支持 batch 矩阵乘法，所以接下来 8 个头可以<strong>一次性并行</strong>算完，不需要写 for 循环。</p>
</aside>

### 第 3 步：每个头独立做 Scaled Dot-Product Attention

和单头时完全一样的公式，只是多了一个 `n_heads` 维度，`matmul` 会自动在这个维度上广播：

```python
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
# Q: (2,8,5,8) @ K^T: (2,8,8,5) -> scores: (2,8,5,5)

attn = F.softmax(scores, dim=-1)          # (2, 8, 5, 5)
out = torch.matmul(attn, V)               # (2,8,5,5) @ (2,8,5,8) -> (2, 8, 5, 8)
```

此时 `out` 的形状是 `(B, n_heads, T, d_k)` = `(2, 8, 5, 8)`——8 个头各自输出了一份 `(5, 8)` 的结果，彼此完全独立、互不干扰。

### 第 4 步：拼接回去（Concat）

`transpose` + `view` 的逆操作，把头维度再放回去、拼接成一个 `d_model` 长的向量：

```python
out = out.transpose(1, 2)                 # (2,8,5,8) -> (2,5,8,8)
out = out.contiguous().view(B, T, d_model)  # (2,5,8,8) -> (2,5,64)
```

<aside class="callout">
<p><strong>为什么要 <code>.contiguous()</code></strong>：<code>transpose</code> 只是改变了张量看待内存的方式（stride），并不会真正移动数据，导致内存不连续；而 <code>view</code> 要求内存连续。<code>.contiguous()</code> 会先把数据实际拷贝、排列好，再让 <code>view</code> 安全地重新解释形状。</p>
</aside>

这一步在数值上做的事情，正是把 8 个头各自的 `(5, 8)` 输出，按顺序拼接（`concat`）成每个 token 一条 `64` 维的向量——`head 0` 的 8 维在前，`head 1` 的 8 维紧接着，以此类推。

### 第 5 步：输出投影 $W^O$

拼接完的结果只是简单地把 8 段拼在一起，各段之间还没有「融合」，所以最后要过一个 `Linear` 把信息重新混合：

```python
W_o = nn.Linear(d_model, d_model)
out = W_o(out)   # (2, 5, 64) -> (2, 5, 64)
```

### 五步串起来看形状变化

<div class="cycle">
<span class="cycle-node">(B,T,d) 投影</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">(B,T,h,dk) 切头</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">(B,h,T,dk) 换轴</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">(B,h,T,dk) 做attention</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">(B,T,d) 拼接+W^O</span>
</div>

### 完整版本（封装成 Module）

把上面五步整理进一个 `nn.Module`，方便直接调用：

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x, B, T):
        # (B, T, d_model) -> (B, n_heads, T, d_k)
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, x, mask=None):
        B, T, _ = x.shape

        # 1. 投影
        Q = self.split_heads(self.W_q(x), B, T)  # (B, h, T, dk)
        K = self.split_heads(self.W_k(x), B, T)
        V = self.split_heads(self.W_v(x), B, T)

        # 2. 每个头并行做 attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 1, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)              # (B, h, T, dk)

        # 3. 拼接 + 输出投影
        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, d_model)
        out = self.W_o(out)
        return out, attn


# 验证
mha = MultiHeadAttention(d_model=64, n_heads=8)
out, attn = mha(x)
print(out.shape)    # torch.Size([2, 5, 64])
print(attn.shape)   # torch.Size([2, 8, 5, 5])
```

<div class="gridcards">
<div class="gridcard"><b>为什么要断言整除？</b><br><code>view</code> 切分最后一维要求能整除，否则每个头拿到的维度数不相等，无法组成规整的张量。</div>
<div class="gridcard"><b>attn 的形状为什么是 4 维？</b><br><code>(B, h, T, T)</code>——除了 batch 和 query/key 位置，还多了一维「头」，每个头有自己独立的一份注意力权重矩阵。</div>
<div class="gridcard"><b>为什么 mask 要 unsqueeze？</b><br>原始 mask 通常是 <code>(B, T, T)</code>，要广播到 <code>(B, h, T, T)</code>，需要在头那一维插入一个 <code>1</code>：<code>mask.unsqueeze(1)</code>。</div>
<div class="gridcard"><b>参数量对比单头？</b><br>四个 <code>Linear(d_model, d_model)</code>，跟单头版本（三个投影 + 无输出投影）比只多了 <code>W_O</code> 这一份，其余参数量基本相当。</div>
</div>
