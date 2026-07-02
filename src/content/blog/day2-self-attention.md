---
title: 'Self-Attention'
description: '从 Query、Key、Value 出发，理解 Self-Attention 如何让序列中的每个位置动态聚合上下文信息。'
series: 'llm-fundamentals'
pubDate: 'Jun 25 2026'
---

## Part 1：什么是 Self-Attention？

<p class="lead"><code>Self-Attention</code>（自注意力）是一种让序列中<strong>每个位置都能「关注」序列里所有位置</strong>（包括自己）的机制。</p>

<aside class="callout">
<p><strong>一句话抓住本质</strong>：Self-Attention 不增减 token、不改形状，它只是把每个 token 的向量「重写」成融合了上下文的版本。输入 5 个向量，输出还是 5 个向量，但每一个都「读过全场」之后变聪明了。</p>
</aside>

具体来说，对每个 `token`，它会根据自己和其它所有 `token` 的「相关程度」算出一组权重，再用这组权重对所有 `token` 的信息做加权求和，得到该位置新的表示。

### Attention 更新的到底是什么？

这是最容易被搞混的一点——**被更新的不是 Q、K、V，而是每个 token 的向量表示（hidden state）本身**。要分清两个完全不同层面的「更新」：

<div class="compare">
<div class="compare-col is-workflow">
<span class="compare-head">训练阶段：更新权重</span>
<p>反向传播更新的是投影矩阵 <code>W_Q、W_K、W_V</code>（还有后面的 <code>W_O</code>）——这些是模型的<strong>参数</strong>，训练完就固定下来，推理时不再变化。</p>
</div>
<div class="compare-col is-agent">
<span class="compare-head">前向阶段：更新表示</span>
<p>每算一层 Attention，真正被改写、并往下一层传递的是 <strong>token 的向量 x</strong>——Q、K、V 只是这一层用固定权重现算出来的中间结果，本层用完即丢。</p>
</div>
</div>

具体来说，每一层的输出是把 Attention 的结果通过残差连接叠加回原始输入：

$$
x' = x + \text{Attention}(Q,K,V), \quad \text{其中 } Q=xW_Q,\ K=xW_K,\ V=xW_V
$$

也就是说，$x$（token 的向量表示）才是那个「携带上下文信息、层层被更新」的对象；$Q,K,V$ 只是每一层根据当前 $x$ 现算出来的「工具」——下一层拿到的是新的 $x'$，会用同一套（训练好、固定不变的）$W_Q,W_K,W_V$ **重新**投影出全新的 $Q,K,V$，而不是复用上一层的。

<aside class="callout">
<p><strong>一句话</strong>：Q、K、V 像是这一轮「投票」用的问题、名片和答案，投完票就扔；真正被投票结果改写、并一路带到下一层的，是每个 token 自己的那个向量。「学到上下文」说的正是这个向量在层层 Attention 之后，内容变得越来越丰富。</p>
</aside>

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

### W_Q、W_K、W_V 一开始是随机的吗？

**是的，一开始纯随机**（常用 `Xavier`/`He` 初始化，取一堆很小的随机数）。但「随机」只是<strong>起点</strong>，权重矩阵之后会被训练一步步「掰」成有意义的形状——这个过程和网络里所有其它权重（包括 FFN）完全一样，没有任何特殊之处：

<div class="cycle">
<span class="cycle-node">随机初始化 W_Q/W_K/W_V</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">前向传播算出 loss</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">反向传播算梯度</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">优化器微调权重</span>
<span class="cycle-note">重复几十万~几十亿步，权重从随机逐渐收敛</span>
</div>

具体来说：

1. **随机初始化**：训练刚开始时，$W_Q, W_K, W_V$ 里全是没有任何语义的随机小数，此时算出来的 attention 权重基本是「乱看」，没有章法。
2. **前向传播 + 算 loss**：把一批文本喂进整个模型（Attention 只是其中一层），最终算出一个任务损失，比如「预测下一个词对不对」（`next-token prediction`）。
3. **反向传播算梯度**：`loss` 通过链式法则一路往回传，能算出「$W_Q$ 里每一个数如果变大一点/变小一点，loss 会怎么变」——这就是梯度，它对 $W_Q, W_K, W_V$ 和网络里所有其它权重都适用。
4. **优化器更新权重**：顺着梯度的反方向，把每个权重挪动一小步，让 loss 稍微降低一点。

**关键点**：没有人手工规定「$W_Q$ 必须学成查询、$W_K$ 必须学成键」——这些角色分工是**被训练目标反推出来的**。如果某种 $W_Q, W_K$ 的组合能让 $QK^\top$ 更准确地反映「哪些 token 之间真正相关」，模型就能更好地利用上下文、把下一个词猜得更准，loss 就更低；梯度下降会不断朝这个方向推进，于是随机权重逐渐「进化」出「查询-键匹配」这种有意义的结构。这跟人脑不会"设计"神经元该干什么、而是靠环境反馈慢慢塑造连接强度，是同一个道理。

<aside class="callout">
<p><strong>一句话</strong>：Q、K、V 的「角色分工」不是设计出来的，是海量数据 + 梯度下降，为了让模型在具体任务上表现更好，自己「优化」出来的副产品。公式（<code>Q=XW_Q</code> 这种投影结构）是人为设计的，但权重矩阵里具体的数值、以及它们呈现出的语义分工，完全是训练学出来的。</p>
</aside>

### 到底是先做 Self-Attention，还是先算出 W_Q/W_K/W_V？

这个问题的关键是分清「结构」和「过程」：$W_Q, W_K, W_V$ 是模型结构里天生就存在的**参数占位**——模型刚被创建、哪怕还没训练一步，它们就已经以随机数的形式存在了。Self-Attention 的公式本身就需要拿 $W_Q, W_K, W_V$ 去把 $X$ 投影成 $Q, K, V$——**没有权重（哪怕是随机的），根本没法做 Self-Attention 这次计算**。

所以严格来说：**永远是先有（哪怕是随机的）权重，再用它做一次 Self-Attention**。但真正的顺序不是「先……后……」这种一次性的先后关系，而是一个不断循环的过程：

<div class="cycle">
<span class="cycle-node">① 模型创建：W_Q/W_K/W_V 随机初始化</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">② 用当前权重做一次 Self-Attention（前向）</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">③ 算 loss、反向传播</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">④ 更新 W_Q/W_K/W_V</span>
<span class="cycle-note">回到②，用刚更新的权重再做一次 Self-Attention……重复几十亿步</span>
</div>

也就是说，「Self-Attention 计算」和「$W_Q/W_K/W_V$ 的具体数值」在整个训练过程里是**交替发生、互相依赖**的：不存在"先把 Self-Attention 算完再去求权重"，也不存在"先把权重训练到完美再开始做 Attention"。从训练开始的第一步起，两者就已经在这个循环里共同演化了——你现在看到的、能学到有意义上下文关系的 Self-Attention，只是这个循环跑了几十亿步之后的结果。

<aside class="callout">
<p><strong>类比</strong>：有点像"先有鸡还是先有蛋"，但这里答案很明确——<strong>先有（随机的）蛋</strong>。模型构建那一刻，权重（哪怕只是随手生成的随机数）就已经存在；每一次 Self-Attention 的计算，都是拿「当前这一版」权重去用；每一次训练更新，又把权重往更好的方向推一点，供下一次 Self-Attention 使用。推理阶段（比如你现在跟一个训好的模型聊天）则是这个循环停下来之后，只跑「② 前向」这一步，反复用同一套固定权重做 Self-Attention。</p>
</aside>

### x 的"更新"和 W 的"更新"，为什么不会互相打架？

之前说过两件事：Self-Attention 更新的是 $x$；训练更新的是 $W_Q, W_K, W_V$。听起来像是"同时"在改两个东西，但其实它们发生在**完全不同的时间尺度**上，从来不会撞在一起——关键区别是：**x 不是参数，只是流过网络的数据；W 才是需要长期保存的参数**。

<div class="compare">
<div class="compare-col is-workflow">
<span class="compare-head">x 的"更新"：一次前向传播内部</span>
<p>对<strong>同一条输入</strong>，$x$ 沿着层数往前走：$x_0 \to x_1 \to \dots \to x_N$。每一层用的都是<strong>这一步固定不动</strong>的 $W_Q,W_K,W_V$，把上一层的 $x$ 算成下一层的新 $x$——纯粹是矩阵乘法，不涉及任何"学习"。这次前向传播结束后，$x_0,x_1,\dots$ 全部被丢弃，下一条输入进来会从零开始生成一份全新的 $x$。</p>
</div>
<div class="compare-col is-agent">
<span class="compare-head">W 的"更新"：跨越一整次前向+反向才发生一次</span>
<p>要等整条前向传播走完、算出 loss、反向传播算完梯度之后，<strong>optimizer 才对 $W_Q,W_K,W_V$ 做一次微调</strong>。这次调整会被保留下来，供<strong>之后所有</strong>输入的前向传播使用，直到下一个训练 step 反向传播时再改一次。</p>
</div>
</div>

<div class="cycle">
<span class="cycle-node">这一步内：W 固定，x0→x1→…→xN（x 在动）</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">算 loss、反向传播</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">optimizer 更新 W（W 在动，只动这一下）</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">下一步：全新输入生成全新 x0，用新 W 重新走一遍</span>
</div>

所以准确的说法不是"同时更新 x 和 W"，而是：**在一次前向传播这个短时间窗口里，W 是常量、x 是变量；在训练的更长时间尺度上，W 才是被缓慢改变的那个"变量"，而每一次用到的 x 都只是转瞬即焚的临时产物，用完就扔**。两者从没有在同一个时刻被同时改写过。

<aside class="callout">
<p><strong>类比</strong>：把 $W$ 想象成生产线上机器固定的加工参数，$x$ 是流水线上正在被加工的零件。零件（$x$）经过每一道工序被逐步改造成成品，这是"零件被更新"；而"调整机器参数"（更新 $W$）是工厂下班后根据这批零件的质检结果去做的事——发生在完全不同的时间点，不会跟零件正在流水线上被加工这件事互相冲突。</p>
</aside>

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

下面是面试里最常考的一版：单头的 Scaled Dot-Product Attention。（拓展到 Multi-Head 的完整实现见下一篇「多头注意力机制」）

```python
import torch
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


# 简单验证
if __name__ == "__main__":
    Q = torch.randn(2, 5, 64)  # (batch=2, seq_len=5, d_k=64)
    K = torch.randn(2, 5, 64)
    V = torch.randn(2, 5, 64)
    out, attn = scaled_dot_product_attention(Q, K, V)
    print(out.shape)   # torch.Size([2, 5, 64])
    print(attn.shape)  # torch.Size([2, 5, 5])
```

### 面试时几个容易被追问的点

<div class="gridcards">
<div class="gridcard"><b>causal mask 怎么造？</b><br><code>torch.triu(torch.ones(T,T), diagonal=1)</code>，上三角为 1（屏蔽未来位置），配合 <code>masked_fill(..., -inf)</code>。</div>
<div class="gridcard"><b>为什么用 -inf 而不是 0？</b><br>mask 加在 softmax <em>之前</em>的 logits 上，<code>-inf</code> 过 softmax 后权重才会变 0；直接置 0 还得重新归一化，不优雅。</div>
<div class="gridcard"><b>复杂度是多少？</b><br>时间和空间都是 <code>O(n²·d)</code>，瓶颈在 <code>QKᵀ</code> 那个 <code>n×n</code> 矩阵。</div>
</div>
