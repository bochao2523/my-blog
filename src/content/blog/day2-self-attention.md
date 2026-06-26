---
title: 'Self-Attention'
description: '从 Query、Key、Value 出发，理解 Self-Attention 如何让序列中的每个位置动态聚合上下文信息。'
pubDate: 'Jun 25 2026'
---

## Part 1：什么是 Self-Attention？

<p class="lead"><code>Self-Attention</code>（自注意力）是一种让序列中<strong>每个位置都能「关注」序列里所有位置</strong>（包括自己）的机制。</p>

<aside class="callout">
<p><strong>一句话抓住本质</strong>：Self-Attention 不增减 token、不改形状，它只是把每个 token 的向量「重写」成融合了上下文的版本。输入 5 个向量，输出还是 5 个向量，但每一个都「读过全场」之后变聪明了。</p>
</aside>

具体来说，对每个 `token`，它会根据自己和其它所有 `token` 的「相关程度」算出一组权重，再用这组权重对所有 `token` 的信息做加权求和，得到该位置新的表示。

### 为什么要堆很多层？

每过一层 Self-Attention，`token` 向量就再吸收一轮上下文，理解越来越深：

<div class="cycle">
<span class="cycle-node">第 1 层：苹果 = 一家公司</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">中间层：苹果与「产品」相关</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">更深层：苹果发布了它的新产品</span>
<span class="cycle-note">层数越深，token 能建模的语义关系越复杂</span>
</div>

### self 是什么意思？

「self」的含义是 `Q`、`K`、`V` 都来自同一个输入序列：

<div class="compare">
<div class="compare-col is-workflow">
<span class="compare-head">Self-Attention</span>
<p><code>Q</code>、<code>K</code>、<code>V</code> 全部来自<strong>同一个</strong>输入序列——序列内部互相看。</p>
</div>
<div class="compare-col is-agent">
<span class="compare-head">Cross-Attention</span>
<p><code>Q</code> 来自一个序列，<code>K</code> / <code>V</code> 来自<strong>另一个</strong>序列——实现跨序列对齐（如翻译里 decoder 回看 encoder）。</p>
</div>
</div>

## Part 2：为什么要除以 $\sqrt{d_k}$？

核心公式里那个缩放因子：

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

### 直觉问题

点积 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 是 $d_k$ 项相加。维度 $d_k$ 越大，点积的数值量级越大、方差越大。

### 方差推导（面试常考）

假设 $q_i, k_i$ 相互独立，均值为 0、方差为 1，那么对单项：

$$
E[q_i k_i] = 0, \quad \text{Var}(q_i k_i) = E[q_i^2]E[k_i^2] = 1
$$

因此点积的方差为：

$$
\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k
$$

即标准差是 $\sqrt{d_k}$。

### 为什么这是个问题

点积量级随 $d_k$ 增大后，送进 `softmax` 的 logits 会很大，`softmax` 输出会被推向「非常尖锐」的分布（接近 one-hot）。而 `softmax` 在饱和区的梯度极小——这会导致梯度消失，训练困难。

<aside class="callout">
<p><strong>一句话</strong>：÷√dₖ 把 logits 的方差重新拉回 1 的量级，让 softmax 工作在梯度敏感区，训练更稳定。这就是 <strong>Scaled</strong> Dot-Product Attention 里「Scaled」的全部含义。</p>
</aside>

## Part 3：Q、K、V 怎么推导出来，各有什么作用

### 怎么来的

它们不是天上掉下来的，而是对同一个输入 $X \in \mathbb{R}^{n \times d_{model}}$ 做三个独立的线性投影（三个可学习的权重矩阵）：

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

其中 $W_Q, W_K \in \mathbb{R}^{d_{model} \times d_k}$，$W_V \in \mathbb{R}^{d_{model} \times d_v}$，都是训练出来的参数。也就是说每个 `token` 的同一个向量，被映射成了三个不同子空间的角色向量。

### 各自的作用（用检索系统类比最直观）

可以把 Attention 想成一次「软检索」，三个角色各司其职：

<div class="stack">
<div class="stack-band stack-1"><b>Q · Query 查询</b>「我这个 token 想找什么样的信息」——拿着它去和别人匹配。</div>
<div class="stack-band stack-2"><b>K · Key 键</b>「我能提供什么、用什么特征被匹配」——<code>QKᵀ</code> 算的就是 Query 与每个 Key 的相似度（匹配分数）。</div>
<div class="stack-band stack-3"><b>V · Value 值</b>「匹配上之后我实际交出去的内容」——softmax 权重作用在 V 上做加权求和。</div>
</div>

整条公式连起来读，就是一次完整的「软检索」流程：

<div class="cycle">
<span class="cycle-node">Q 与所有 K 算相似度</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">÷ √dₖ 缩放</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">softmax 归一化成权重</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">用权重加权聚合 V</span>
</div>

### 为什么要分成三个、而不是直接用 X 算？

关键是**解耦了「用什么来匹配」和「匹配后给出什么」**。如果不做投影、直接 $XX^\top$，那么相似度矩阵被强制对称，且匹配特征和内容特征绑死了，表达能力大幅受限。分开投影后，模型可以学到「按 A 特征去匹配，但传递 B 内容」这种灵活模式。`Multi-Head Attention` 进一步让不同的头在不同子空间里关注不同的关系。

## Part 4：手撕代码

下面是面试里最常考的两版：单头的 Scaled Dot-Product Attention，以及完整的 Multi-Head Self-Attention。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (..., n_q, d_k)
    K: (..., n_k, d_k)
    V: (..., n_k, d_v)
    mask: (..., n_q, n_k)，被 mask 的位置为 True/1
    返回: (..., n_q, d_v) 以及注意力权重
    """
    d_k = Q.size(-1)
    # 1. 算相似度分数并缩放
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (..., n_q, n_k)

    # 2. 可选的 mask（如 causal mask 或 padding mask）
    if mask is not None:
        scores = scores.masked_fill(mask == 1, float('-inf'))

    # 3. softmax 归一化成权重（在 key 维度上）
    attn = F.softmax(scores, dim=-1)

    # 4. 加权聚合 value
    output = torch.matmul(attn, V)  # (..., n_q, d_v)
    return output, attn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度

        # 三个投影矩阵 + 输出投影；这里用一个大的 Linear 同时算出 Q/K/V 也很常见
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, d_model)
        B, T, _ = x.shape

        # 1. 线性投影得到 Q/K/V
        Q = self.W_q(x)  # (B, T, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2. 拆成多头: (B, T, d_model) -> (B, n_heads, T, d_k)
        Q = Q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # 3. 每个头各自做 attention（mask 需 broadcast 到头维度）
        if mask is not None:
            mask = mask.unsqueeze(1)  # (B, 1, T, T)
        out, attn = scaled_dot_product_attention(Q, K, V, mask)

        # 4. 拼回去: (B, n_heads, T, d_k) -> (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # 5. 输出投影
        out = self.W_o(out)
        return out, attn


# 简单验证
if __name__ == "__main__":
    x = torch.randn(2, 5, 64)  # (batch=2, seq_len=5, d_model=64)
    mha = MultiHeadSelfAttention(d_model=64, n_heads=8)
    out, attn = mha(x)
    print(out.shape)   # torch.Size([2, 5, 64])
    print(attn.shape)  # torch.Size([2, 8, 5, 5])
```

### 面试时几个容易被追问的点

<div class="gridcards">
<div class="gridcard"><b>causal mask 怎么造？</b><br><code>torch.triu(torch.ones(T,T), diagonal=1)</code>，上三角为 1（屏蔽未来位置），配合 <code>masked_fill(..., -inf)</code>。</div>
<div class="gridcard"><b>为什么用 -inf 而不是 0？</b><br>mask 加在 softmax <em>之前</em>的 logits 上，<code>-inf</code> 过 softmax 后权重才会变 0；直接置 0 还得重新归一化，不优雅。</div>
<div class="gridcard"><b>多头为什么切分 d_model？</b><br>把 <code>d_model</code> 切成 <code>n_heads</code> 份，保证总参数量 / 计算量与单头基本持平，又让各头在不同子空间学不同关系。</div>
<div class="gridcard"><b>复杂度是多少？</b><br>时间和空间都是 <code>O(n²·d)</code>，瓶颈在 <code>QKᵀ</code> 那个 <code>n×n</code> 矩阵。</div>
</div>
