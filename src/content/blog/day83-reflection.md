---
title: 'Reflection'
description: '从评估闭环出发，理解 Reflection 与 Self-Critique 如何让 Agent 从失败中长记性——Reflexion、Self-Refine、CRITIC 等范式的核心思路。'
pubDate: 'Jun 30 2026'
---

<p class="lead">ReAct 有<strong>行动闭环</strong>，Plan-and-Execute 有<strong>规划闭环</strong>——还缺<strong>评估闭环</strong>：写错了谁来喊停、复盘、把教训留给下次？</p>

## Part 1：为什么需要 Reflection

### 1.1 从左到右：要给模型纠错的机会

LLM 自回归生成，只能往前写、不能撤回。早期一步偏了——判错题型、搜错词、读歪 observation——后面每一步都在错地基上继续盖：

<div class="cycle">
<span class="cycle-node">t₁</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">t₂</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node" style="border-color:#fca5a5;color:#b91c1c;background:#fef2f2">t₃ ✗</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">t₄</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">tₙ</span>
<span class="cycle-note">第 3 步错了，第 4 步起全建立在错误前提上——模型自己往往察觉不到</span>
</div>

ReAct 能在下一步根据新 observation 调整，但那是<strong>被动续写</strong>，不是<strong>主动纠错</strong>。Reflection 插入一个显式的「回头看」环节：

<div class="cycle">
<span class="cycle-node">暂停</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">审视轨迹</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">修正 / 重来</span>
<span class="cycle-note">失败时触发、交付前 self-check、设最大重试——给模型结构化的纠错窗口，而非赌它惯性里自己拐回来</span>
</div>

### 1.2 不止改这一次：为以后留范式

Reflection 还要把教训<strong>沉淀</strong>下来，供后续生成引用——按作用范围分三层：

<div class="gridcards">
<div class="gridcard"><b><span class="gc-ico">🔁</span>单轮内</b>同一道题：批判 → 改写 → 再批判。教训留在当前上下文（Self-Refine、CRITIC）。</div>
<div class="gridcard"><b><span class="gc-ico">🧠</span>跨轮</b>反思写入 memory，下次 trial 开局先读历史教训（Reflexion）。</div>
<div class="gridcard"><b><span class="gc-ico">📐</span>抽象成规则</b>从个案抽出可复用模板，如「搜实体前先验拼写」。</div>
</div>

没有这层积累，每次 trial 独立、同样错误反复踩；有了它，生成从「一次性射击」变成<strong>可迭代、可积累</strong>的过程。

<aside class="callout">
<p><strong>🧭 三个闭环</strong>：ReAct 能做事；Plan-and-Execute 知道做哪些；Reflection 知道做得对不对、下次怎么更好。</p>
</aside>

## Part 2：Reflection 的几种基本形式

四种形式按「反馈从哪来、反思记在哪」排列——从轻到重，可按任务选。

| 形式 | 反馈来源 | 核心链路 | 最适合 |
|---|---|---|---|
| **内联自检** | 无（模型自判） | 同一次生成末尾加「等等，验证一下」 | 轻量复查，零额外调用 |
| **Self-Refine** | 无（换 prompt 当批评者） | 生成 → 批判 → 重写 | 格式、文风、代码风格 |
| **CRITIC** | 外部工具（硬信号） | 生成 → 工具验证 → 据结果反思 | 代码、数学等可执行验证的任务 |
| **Reflexion** | 环境奖励 / 自评 | 跑完整轨迹 → 评估 → 写入记忆 → 带反思重试 | 多步 Agent、可多次 trial 的场景 |

### 2.1 内联自检

不拆调用，在同一条生成流末尾让模型自己踩刹车，当场复查：

<div class="cycle">
<span class="cycle-node">生成</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">「等等，验证一下…」</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">输出</span>
<span class="cycle-note">零额外调用，一次 API 搞定</span>
</div>

受模型自判能力天花板限制，有时会把对的改错。适合输出前的随手检查，不宜当唯一纠错手段。

### 2.2 Self-Refine

三轮、两个角色，同一模型换 prompt 切换身份——批评者只挑毛病、不给新答案，生成者据批判重写：

<div class="cycle">
<span class="cycle-node">生成者</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">批评者</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">生成者 · 重写</span>
<span class="cycle-arrow">↻</span>
<span class="cycle-node">收敛即停</span>
<span class="cycle-note">不依赖外部基建，靠换 prompt 制造视角差</span>
</div>

改格式和表达很有效；推理正确性提升有限，每轮迭代 token 翻倍。适合写作、文档、代码风格打磨。

### 2.3 CRITIC（工具锚定）

先产出，再用<strong>外部硬信号</strong>验证——编译器、单测、计算器、搜索引擎事实核对——再据结果定向修改：

<div class="cycle">
<span class="cycle-node">生成</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">工具验证</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">据硬信号反思修改</span>
<span class="cycle-note">环境直接告诉它对没对，不必自己猜</span>
</div>

关键洞见：<strong>验证往往比生成容易</strong>（判一道题对不对，比从头解出来简单）。CRITIC 把这条「生成-验证不对称」做实了。局限是任务必须可验证，且多一轮工具调用，延迟更高。

### 2.4 Reflexion（轨迹级）

粒度拉到<strong>整次尝试</strong>：跑完一条完整轨迹，按最终结果生成文字反思存进记忆，下次开局先读再重来：

<div class="cycle">
<span class="cycle-node">跑完整轨迹</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">评估结果</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">写入 memory</span>
<span class="cycle-arrow">↻</span>
<span class="cycle-node">带反思重试</span>
<span class="cycle-note">唯一会「跨 trial 积累经验」的形式</span>
</div>

适合多步任务、允许重试的场景。但若评估者仍是模型自己，天花板依旧存在；需要配套 memory 管理和重试预算。

<aside class="callout">
<p><strong>⚖️ 怎么选</strong>：能靠工具验就用 CRITIC；只要改表达用 Self-Refine；长链路 Agent 用 Reflexion；其余加一句内联自检当保险。代价从轻到重：零调用 → 2× token → 工具延迟 → memory + 多 trial。</p>
</aside>

## Part 3：Reflection 的工程化考量

范式能跑通，不等于能上线。六个工程点按「<strong>要不要反思 → 反思怎么做 → 怎么兜底观测</strong>」串起来，每块标一个最常踩的坑：

<div class="cycle">
<span class="cycle-node">① 触发门控</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">② 停止准则</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">③ 成本延迟</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">④ 反馈信号</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">⑤ 记忆管理</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">⑥ 可观测</span>
</div>

### 3.1 触发与门控

**不是每个 query 都值得反思。** 反思成本线性叠加，入口要先做轻量路由——置信度、复杂度、规则或小模型——决定进不进反思回路。简单 query 单次生成直接返回，难样本才投多轮推理。本质是 test-time compute 的预算分配。

<div class="warn"><p><strong>坑</strong>：全开反思，简单问题也 2–3 倍 token，P99 和账单一起爆。</p></div>

### 3.2 停止准则

反思回路至少三道闸：

<div class="steps">
<div class="step">
<div class="step-num">A</div>
<div class="step-body"><p><strong>通过信号</strong>　evaluator 判定「合格」即停</p></div>
</div>
<div class="step">
<div class="step-num">B</div>
<div class="step-body"><p><strong>硬上限</strong>　达到最大轮数强制停</p></div>
</div>
<div class="step">
<div class="step-num">C</div>
<div class="step-body"><p><strong>收敛检测</strong>　连续两轮答案几乎不变、或在两个答案间横跳 → 立即停</p></div>
</div>
</div>

<div class="warn"><p><strong>坑</strong>：震荡（over-correction）——第 2–3 轮把对的改错又改回来，空耗 token。做法：记录每轮 evaluator 分数，<strong>只在分数实质提升时接受新答案</strong>，否则回退上一轮最优。</p></div>

### 3.3 成本与延迟

每多一轮 = 一次更长的推理（轨迹 + 反思塞进 context，prompt 越来越胖），token 和延迟线性甚至超线性涨，P99 最难扛。缓解手段：

<div class="gridcards">
<div class="gridcard"><b><span class="gc-ico">🪶</span>更轻的 critic</b>反思轮用小 / 便宜模型当批评者。</div>
<div class="gridcard"><b><span class="gc-ico">⚡</span>能并行别串行</b>batch 评估多个候选，不要逐个串。</div>
<div class="gridcard"><b><span class="gc-ico">⏱️</span>反思预算上限</b>超时即返回当前最优答案。</div>
<div class="gridcard"><b><span class="gc-ico">🌊</span>streaming 取舍</b>先返初稿后台精修，或明示「深度思考中」。</div>
</div>

<div class="warn"><p><strong>坑</strong>：不设预算上限，长尾请求把队列拖死。</p></div>

### 3.4 反馈信号的工程化

从「能跑」到「真有用」的分水岭——尽量锚定可验证的硬信号（呼应 Part 2 的生成-验证不对称）：

<div class="stack">
<div class="stack-band stack-2"><b>代码</b>沙箱执行 + 单测，结构化返回「哪个 test 挂了、报什么错」，而不是只给 pass/fail</div>
<div class="stack-band stack-1"><b>数学</b>计算器 / 符号引擎验算</div>
<div class="stack-band stack-3"><b>事实</b>检索核验</div>
</div>

沙箱要管安全隔离、超时、资源限制。拿不到外部信号、只能内生评估时，critic 用独立 prompt / 角色，甚至换一个模型当 verifier，部分绕开 self-feedback ceiling。

<div class="warn"><p><strong>坑</strong>：只喂 pass/fail，模型不知道往哪改。</p></div>

### 3.5 记忆与上下文管理

Reflexion 式跨轮反思会越积越多，直接撑爆窗口、推高每轮成本。三件事要做：<strong>摘要压缩</strong>、<strong>按相关性筛选</strong>（只留对当前任务有迁移价值的教训）、<strong>容量上限 + 淘汰</strong>。

此外把「反思」和「原始轨迹」分开存——回灌时只塞提炼后的教训，别把完整错误轨迹再塞一遍；关键反思放显眼位置，防 lost-in-the-middle。

<div class="warn"><p><strong>坑</strong>：无上限堆历史反思，context 滚雪球、attention 被稀释。</p></div>

### 3.6 可观测与评估

线上必须能监控，否则不知道 reflection 是帮忙还是帮倒忙。

<div class="compare">
<div class="compare-col is-workflow">
<span class="compare-head">离线评估</span>
<p>别只看终态准确率。分别埋点 <strong>错→对</strong> 与 <strong>对→错</strong> 转化率，净收益 = 前者 − 后者。很多「看着涨」其实换个分布就崩。</p>
</div>
<div class="compare-col is-agent">
<span class="compare-head">线上监控</span>
<p>平均反思轮数、每轮接受率、触发率、P99 延迟、单请求 token 成本；并做 A/B（反思 on/off）看真实业务指标。</p>
</div>
</div>

<div class="warn"><p><strong>坑</strong>：多轮采样有随机性，debug 困难——固定 seed 或记录完整轨迹便于回放。</p></div>

<aside class="callout">
<p><strong>📋 上线 checklist</strong>：入口门控 → 停止 + 回退 → 成本预算 → 工具锚定 → memory 淘汰 → 错→对 / 对→错 双指标监控。缺任何一环，reflection 都可能在 demo 里好看、线上帮倒忙。</p>
</aside>

