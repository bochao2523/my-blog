---
title: 'ReAct 框架'
description: '从 Thought-Action-Observation 循环出发，理解 ReAct 如何将推理与工具调用交织成可解释的 Agent 决策过程。'
series: 'agent'
pubDate: 'Jun 26 2026'
---

<p class="lead"><strong>ReAct = Reason + Act</strong>。它把「推理」（thought）和「调工具」（action）交错在同一条轨迹里，让每一步推理都被真实的外部 observation 校准，而不是闭门空想。</p>

<aside class="callout">
<p><strong>出处</strong>：Yao et al., <em>ReAct: Synergizing Reasoning and Acting in Language Models</em>, ICLR 2023（arXiv:2210.03629）。</p>
</aside>

## 1. 核心机制：Thought → Action → Observation 循环

ReAct 把「解题」组织成一个**循环**：

<div class="steps">
<div class="step">
<div class="step-num">1</div>
<div class="step-body"><p><span class="tao tao-t">Thought 思考</span>　决定下一步做什么——分解任务、规划、反思。<strong>零外部成本</strong>，纯粹是模型的内部独白。</p></div>
</div>
<div class="step">
<div class="step-num">2</div>
<div class="step-body"><p><span class="tao tao-a">Action 行动</span>　去真实环境里执行一个动作——检索、调 API、运行代码。</p></div>
</div>
<div class="step">
<div class="step-num">3</div>
<div class="step-body"><p><span class="tao tao-o">Observation 观察</span>　把环境返回的反馈拿回来。<strong>这一步不是模型生成的</strong>，是执行器回填的真实结果。</p></div>
</div>
</div>

没结束，就带着新拿到的观察**回到思考**，再行动、再观察，如此往复，直到模型决定给出答案（`finish`）：

<div class="cycle">
<span class="cycle-node">Thought 思考</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">Action 行动</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">Observation 观察</span>
<span class="cycle-arrow">↻</span>
<span class="cycle-node">Finish 答案</span>
<span class="cycle-note">未结束就带着新的 observation 回到 Thought；如此往复，直到模型输出 finish</span>
</div>

### 上下文如何「滚雪球」

每一步喂给模型的上下文 $c_t$ 都是**从头累加的完整历史**——LLM 无状态、记不住上一步，只能把目前为止的全部内容（初始 prompt + 之前每一轮的 thought / action / observation）整个重发。以 Apple Remote 为例，上下文一轮轮长出来：

| 步 | 喂给模型的上下文 $c_t$ | 模型生成 | 执行器回填 |
|---|---|---|---|
| 0 | `示例 + Question` | Thought 1 + Action 1 | Observation 1 |
| 1 | $c_0$ + `T1 + A1 + O1` | Thought 2 + Action 2 | Observation 2 |
| 2 | $c_1$ + `T2 + A2 + O2` | Thought 3 + `finish[...]` | —（结束） |

<aside class="callout">
<p><strong>💡 一句话抓住</strong>：思考是「零成本」的动作——不花外部调用、不占观察，却给了模型一个<strong>显式的工作记忆</strong>。而上下文<strong>只增不减</strong>：这正是后面 observation 必须截断（§3.3）、上下文会被撑爆（§4）的根源。</p>
</aside>

### 一条真实轨迹

以 ReAct 原论文的 Wikipedia QA 为例，动作空间为 `search[entity]` / `lookup[string]` / `finish[answer]`。读这条轨迹时，注意 <span class="tao tao-o">Observation</span> 永远是工具回填、不是模型编的：

<div class="trace">
<div class="trace-row"><p><span class="tao tao-t">Thought 1</span></p><p>我需要先查「苹果遥控器」，看它最初是为哪个程序设计的。</p></div>
<div class="trace-row"><p><span class="tao tao-a">Action 1</span></p><p><code>search[Apple Remote]</code></p></div>
<div class="trace-row"><p><span class="tao tao-o">Observation 1</span></p><p>Apple Remote 最初是为控制 Front Row media center 程序设计的……</p></div>
<div class="trace-row"><p><span class="tao tao-t">Thought 2</span></p><p>那现在的问题变成：还有什么设备能控制 Front Row？</p></div>
<div class="trace-row"><p><span class="tao tao-a">Action 2</span></p><p><code>lookup[Front Row]</code></p></div>
<div class="trace-row"><p><span class="tao tao-o">Observation 2</span></p><p>Front Row 也可以由键盘的功能键控制。</p></div>
<div class="trace-row is-finish"><p><span class="tao tao-f">Finish</span></p><p>答案是键盘功能键 → <code>finish[键盘功能键]</code></p></div>
</div>

## 2. 为什么「边想边做」优于「先想完再做」

「先想完再做」（plan-then-execute）的致命弱点是：**计划是在信息不完整时一次性生成的**。你还没 search，就不知道 search 会返回什么；一旦真实世界和你的假设不符，整套计划就崩了。

<div class="compare">
<div class="compare-col is-workflow">
<span class="compare-head">先想完再做</span>
<p>信息不全时一次性出计划 → 环境一意外，整套计划崩盘，难以局部修补。</p>
</div>
<div class="compare-col is-agent">
<span class="compare-head">ReAct：边想边做</span>
<p>每一步都拿真实 observation 校准下一步推理 → 错了能当场改道，计划随信息生长。</p>
</div>
</div>

ReAct 的交错式结构在四个层面占优：

<div class="steps">
<div class="step">
<div class="step-num">1</div>
<div class="step-body"><p><strong>推理被观测锚定（observation-grounded reasoning）</strong>——每一步 thought 都基于上一步<strong>真实返回</strong>的 observation，而不是模型脑补的状态。这直接压制了纯推理的幻觉。</p></div>
</div>
<div class="step">
<div class="step-num">2</div>
<div class="step-body"><p><strong>自适应 / 错误恢复（adaptivity）</strong>——action 失败、返回空、返回意料之外的结果时，下一个 thought 可以当场改写策略（「这个实体查不到，换个关键词」）。预先规划对环境意外是脆弱的。</p></div>
</div>
<div class="step">
<div class="step-num">3</div>
<div class="step-body"><p><strong>动态分解（dynamic decomposition）</strong>——复杂目标不是一次拆完，而是随着发现的信息<strong>逐步</strong>拆解——上面例子里「问题」在 Thought 2 被改写成了一个更具体的子问题。</p></div>
</div>
<div class="step">
<div class="step-num">4</div>
<div class="step-body"><p><strong>工作记忆 / 进度追踪</strong>——thought 充当显式的 working memory，记录「我已经知道什么、还缺什么」，避免在长 horizon 任务里迷失。</p></div>
</div>
</div>

<aside class="callout">
<p><strong>⚖️ 代价</strong>：交错也不是免费的。每一步都把 observation 塞回上下文 → token 消耗高、容易撑爆窗口，且决策是<strong>贪心</strong>的（没有 lookahead / backtracking）。这正是后续 ReWOO、ToT / LATS 等范式要解决的问题（见 §4.2）。</p>
</aside>

## 3. 工程实现：Prompt 结构 / 停止条件 / Observation 截断

### 3.1 Prompt 结构

ReAct 是 few-shot in-context 范式，一份完整的 prompt 由三段从上到下拼成：

<div class="stack">
<div class="stack-band stack-1"><b>① 指令 Instruction</b>告诉模型有哪些 action 可用、各自返回什么。</div>
<div class="stack-band stack-2"><b>② Few-shot 示范</b>1~6 条完整的 Thought / Action / Observation 轨迹，教会模型这个格式和节奏。</div>
<div class="stack-band stack-3"><b>③ 当前轨迹</b>真正要解的问题，以及到目前为止已积累的步骤（每轮由执行器回填 observation 后再续）。</div>
</div>

下面是一份贴近 ReAct 原论文 HotpotQA 的真实 prompt。注意标注的位置：**模型只从「↓↓↓」处开始生成，之前的所有内容都是你拼好喂进去的**：

```text
Solve a question answering task with interleaving Thought, Action, Observation steps.
Action 有三种：
(1) Search[实体]   —— 在维基百科上检索该实体，返回首段；查不到则返回相近实体名。
(2) Lookup[关键词] —— 返回当前页面中下一句包含该关键词的句子（类似 Ctrl+F）。
(3) Finish[答案]   —— 给出答案并结束。

# ===== few-shot 示范（教格式） =====
Question: 科罗拉多造山带东段延伸进入的那块区域，海拔范围是多少？
Thought 1: 我需要先查科罗拉多造山带，找到它东段延伸进入的区域。
Action 1: Search[科罗拉多造山带]
Observation 1: 科罗拉多造山带是一次造山运动……东段延伸进入高平原（High Plains）。
Thought 2: 东段延伸进入高平原。我再查高平原的海拔范围。
Action 2: Search[高平原]
Observation 2: 高平原海拔大约在 1,800 到 7,000 英尺之间。
Thought 3: 海拔范围是 1,800 到 7,000 英尺。
Action 3: Finish[1,800 到 7,000 英尺]

# ===== 当前要解的问题 =====
Question: 除了苹果遥控器，还有什么设备可以控制最初为它设计的那个程序？
# ↓↓↓ 模型从这里开始生成 ↓↓↓
Thought 1: 我需要先查苹果遥控器，看它最初是为哪个程序设计的。
Action 1: Search[Apple Remote]
# ↑↑↑ 生成停在这里（stop = "\nObservation"）↑↑↑
```

到 Action 1 生成完，程序就把生成停下（stop sequence = `Observation`），拿 `Search[Apple Remote]` 去真正调维基 API，把返回结果作为 `Observation 1:` 拼回上下文，再把整段重新发给模型续写 Thought 2……如此循环。

<aside class="callout">
<p><strong>⚠️ 两个最关键的点</strong></p>
<p>① <strong>Observation 不是模型生成的，是执行器调完工具回填的</strong>。一旦让模型自己「想象」observation，就退化成纯 CoT，grounding 失效。</p>
<p>② <strong>每一轮都要停在 Action 之后</strong>——靠 stop sequence 把控制权交还给你的代码去执行工具，而不是让模型一口气把整条轨迹连 observation 都编出来。</p>
</aside>

现代实现（OpenAI / Anthropic 的 function calling / tool use）把这套文本协议结构化了：Action → 一个结构化的 tool call，Observation → 一条 `tool_result` 消息回填。换了外壳，但「生成动作 → 停下 → 执行 → 回填观察 → 续写」的循环逻辑完全一样。

### 3.2 停止条件（stopping criteria）

健壮的 ReAct loop 至少要装上这四道「闸」，缺一就可能跑飞：

<div class="gridcards">
<div class="gridcard"><b><span class="gc-ico">🏁</span>显式终止动作</b>模型输出 <code>finish[...]</code> / <code>final_answer</code> 时正常收尾。</div>
<div class="gridcard"><b><span class="gc-ico">🔢</span>步数预算</b><code>max_steps</code>（如 ≤ 8），硬上限防死循环。</div>
<div class="gridcard"><b><span class="gc-ico">🧮</span>token 预算</b>累计上下文逼近窗口上限时强制收尾。</div>
<div class="gridcard"><b><span class="gc-ico">🔁</span>重复检测</b>连续 N 步动作 / 参数完全相同 → 判定卡死，提前退出或注入提示。</div>
</div>

### 3.3 Observation 截断（observation engineering）

工具返回的网页 / 检索结果可能上万 token，直接塞回去会迅速撑爆上下文。原论文本身就做了观测工程：`search` 只返回相关页面的前若干句，`lookup` 只返回包含目标串的句子。常用手段：

<div class="gridcards">
<div class="gridcard"><b><span class="gc-ico">✂️</span>截断</b>只取前 N token / 前 K 句。</div>
<div class="gridcard"><b><span class="gc-ico">📝</span>摘要</b>对长 observation 先用一个轻量 LLM 调用压缩再回填。</div>
<div class="gridcard"><b><span class="gc-ico">🪟</span>滑动窗口 / 外部记忆</b>旧 observation 退出上下文，需要时再检索。</div>
<div class="gridcard"><b><span class="gc-ico">🗂️</span>结构化抽取</b>只保留字段（标题、score、URL），丢弃正文噪声。</div>
</div>

<aside class="callout">
<p><strong>💡 经验法则</strong>：observation 越短越好，只要不丢关键信息。上下文里每一个 token 都在和「模型注意力」抢资源。</p>
</aside>

## 4. 失败模式与局限

把这两件事放一起讲，因为它们是两个层次的问题：**失败模式**是你真把 ReAct 跑起来时会踩的工程坑（可以靠技巧修）；**局限**则是这套「反应式」范式天生的天花板（修不掉，只能换更强的范式）——后者正好引出后面两篇：planning 和 reflection。

### 4.1 失败模式（踩坑 → 修法）

| 失败模式 | 表现 | 解决方案 |
|---|---|---|
| **陷入循环** | 反复 Search 同一个词，或在两个状态间来回震荡——observation 不带来新信息，模型却察觉不到自己在原地打转 | 加重复检测（连续 N 步动作 / 参数相同就判定卡死），配步数预算硬上限；并把「你已经试过 X、没用」显式写回上下文打破循环 |
| **上下文爆了** | 累计 observation 超出窗口，早期关键信息被挤出去，模型「忘了」最初的问题 | 截断 / 摘要 observation（只回填前 K 句或先压缩）；滑动窗口 + 外部记忆让旧观测可检索而非常驻；根因解法见 §4.2 的 ReWOO（把观测从推理里解耦） |
| **幻觉 action / 参数** | 调用不存在的工具、参数 schema 写错、Search 一个编造的实体 | function calling 强 schema + action 校验：非法动作不报错崩溃，而是当成一条 error observation 回喂（「该工具不存在 / 参数缺失」），让模型自己改；必要时上约束解码 |

<aside class="callout">
<p><strong>💡 共性</strong>：这三类坑的修法都绕不开一句话——<strong>把失败信号显式写回上下文</strong>。模型看不见自己的错，你就得替它把错「说出来」放进 prompt。这其实已经是 reflection 的雏形了（§4.2）。</p>
</aside>

### 4.2 局限：ReAct 范式的两个天花板

失败模式能靠工程技巧缓解，但有两件事 ReAct 结构上做不到——而这正是后面两篇要补的。

**① 它「走一步看一步」，没有全局规划 → 引出 plan-and-execute**

ReAct 每一步只盯着当前 observation 做局部最优决策，贪心、无前瞻、不回溯。任务一长，它要么迷失方向，要么因为「每步都回填 observation」而 token 爆炸、效率低下。

<div class="compare">
<div class="compare-col is-workflow">
<span class="compare-head">ReAct（反应式）</span>
<p>走一步看一步，无全局蓝图；长任务里方向易飘、token 开销大。</p>
</div>
<div class="compare-col is-agent">
<span class="compare-head">Plan-and-Execute</span>
<p>先一次性产出完整计划，再去执行——长任务方向更稳，还能省掉中间反复回填的开销（ReWOO 正是把观测从规划里解耦来省 token）。<strong>下一篇主题。</strong></p>
</div>
</div>

**② 它不会从失败里长记性，没有自我批判 → 引出 reflection & self-critique**

ReAct 一轮失败了就是失败了——它不会停下来问「我刚才哪一步错了、下次该怎么改」，更不会把这个教训留到下一次尝试。它只有「行动的闭环」，没有「评估的闭环」。

<div class="compare">
<div class="compare-col is-workflow">
<span class="compare-head">ReAct（行动闭环）</span>
<p>失败即结束，不复盘、不跨尝试积累经验。</p>
</div>
<div class="compare-col is-agent">
<span class="compare-head">Reflection / Self-Critique</span>
<p>一次尝试后让模型显式自我批判、生成「复盘」再重做。Reflexion 把反思存进 episodic memory 跨 trial 复用，Self-Refine / CRITIC 则在单轮内迭代改写。<strong>再下一篇主题。</strong></p>
</div>
</div>

<aside class="callout">
<p><strong>🧭 三者关系一句话</strong>：ReAct 解决「推理要不要落地」（要，靠 observation）；planning 解决「要不要先有蓝图」（长任务要）；reflection 解决「会不会从失败中长记性」（要，靠评估闭环）。后两者都是站在 ReAct 的肩膀上，补它结构上缺的那一块。</p>
</aside>

## 5. 工程实践与面试要点

前面几节把「ReAct 是什么、为什么」讲清了。但要把它搬上生产、或拿去面试，绕不开下面这些工程问题。先用一张速查表锁定高频七问，再逐个过要点：

<div class="gridcards">
<div class="gridcard"><b>Q1　死循环</b>如何发现并打破原地打转？</div>
<div class="gridcard"><b>Q2　Few-shot</b>次优轨迹的影响、如何构建与评估？</div>
<div class="gridcard"><b>Q3　延迟与成本</b>至少三种优化策略。</div>
<div class="gridcard"><b>Q4　评测</b>结果 / 过程 / 开销三类核心指标。</div>
<div class="gridcard"><b>Q5　定位瓶颈</b>「太慢了」从哪查起？</div>
<div class="gridcard"><b>Q6　状态管理</b>好处？LangGraph 如何体现？</div>
<div class="gridcard"><b>Q7　安全 Agent</b>自动比价下单的安全性与鲁棒性。</div>
</div>

### Q1 · 陷入无限循环：如何发现并打破？

**发现**——光靠步数上限不够，要主动检测「在原地打转」：

- **动作指纹**：对 `(tool_name, args)` 做哈希，连续或高频重复同一指纹 → 报警。
- **no-progress 检测**：observation 与上一轮相同（或相似度过高）却仍在继续。
- **振荡检测**：识别 `A→B→A→B` 这类来回跳的模式。
- **硬熔断**：步数 / token / 墙钟时间任一超预算就强制停。

**打破**——

- 把「你已经试过 X 且无效」显式写回上下文，直接掐断重复。
- 提高 temperature 强制探索，或屏蔽刚用过的 action。
- 触发一次 reflection，让模型复盘「为什么没进展」再续。
- 升级策略：超阈值就停 → 回退 → 转人工（human-in-the-loop）。

### Q2 · Few-shot 示例：次优轨迹的影响、如何构建与评估？

次优轨迹的危害——**模型会照着模仿**：

- 模仿冗余搜索 → 步数变多，更慢更贵；
- 模仿过早 / 过晚停机 → 准确率下降；
- 格式不一致 → 格式漂移，导致 action 解析失败。

<aside class="callout">
<p><strong>⚠️ Few-shot 是「行为示范」，不是「知识注入」</strong>。示例里有的坏习惯，模型会原样学走。</p>
</aside>

<div class="compare">
<div class="compare-col is-agent">
<span class="compare-head">如何构建</span>
<ul>
<li>覆盖每种 action 类型至少一次；</li>
<li>故意包含<strong>纠错轨迹</strong>（走错一步再恢复），教模型从失败里爬出来；</li>
<li>轨迹尽量短而最优（模型会模仿长度）；</li>
<li>保证多样性（不同难度 / 领域），避免同质化。</li>
</ul>
</div>
<div class="compare-col is-workflow">
<span class="compare-head">如何评估</span>
<ul>
<li>消融 / A-B：换不同示例集，在 held-out 测试集上比<strong>成功率、平均步数、解析失败率</strong>；</li>
<li>leave-one-out：看单条示例的边际贡献。</li>
</ul>
</div>
</div>

### Q3 · 延迟与成本：至少三种优化策略

| 策略 | 做法 | 主要省什么 |
|---|---|---|
| **减少调用次数** | plan-and-execute / ReWOO 一次出计划，少回填观测 | 步数（轮数） |
| **模型分级路由** | 简单步用便宜小模型，难步才上大模型 | 单位 token 成本 |
| **缓存** | prompt caching（固定的 instruction + few-shot 前缀）+ 工具结果缓存 | prefill + 重复调用 |
| **并行 tool calls** | 互不依赖的动作并发执行 | 墙钟延迟 |
| **缩短上下文** | observation 截断 / 摘要、丢弃旧轮 | 每步 prefill token |
| **蒸馏 / 微调** | 把 few-shot 知识固化进权重，省掉长前缀 | 固定前缀开销 |

<aside class="callout">
<p>任答三种即可，推荐 <strong>「减少调用次数 + 缓存 + 模型路由」</strong> 这组，覆盖面最广。</p>
</aside>

### Q4 · 如何有效评测？核心指标

| 维度 | 指标 |
|---|---|
| **结果质量** | 成功率 / EM / F1 / pass@k |
| **过程质量** | 平均步数、工具调用准确率、循环率、幻觉率、失败恢复率、groundedness（答案是否被 observation 支持） |
| **系统开销** | 延迟 P50 / P95、token 数、\$/task |

<aside class="callout">
<p><strong>方法上</strong>：除了固定测试集做回归，轨迹级质量可以用 <strong>LLM-as-judge</strong> 对每一步的合理性打分——光看最终答案对错，会漏掉「蒙对」和「绕远路」。</p>
</aside>

### Q5 · 用户说「太慢了」：如何定位瓶颈？

先抓住一个公式，把延迟拆成「步数」和「单步耗时」两个乘子：

$$T_{\text{total}} \approx N_{\text{steps}} \times \left(T_{\text{LLM}} + T_{\text{tool}} + T_{\text{net}}\right)$$

然后按这个顺序拆：

- **先分大类**：是步数太多，还是单步太慢？
- **单步 LLM**：看 TTFT、输出 token 数、上下文长度（长 prefill 很贵）、模型大小；
- **工具侧**：外部 API 自身延迟、超时重试在拖后腿；
- **串行 vs 并行**：本可并发的步骤是不是被串起来了；
- **上 tracing**（LangSmith 之类）按 span 看每段耗时，别靠猜。

### Q6 · 引入状态管理（State Management）的好处？LangGraph 如何体现？

**好处**——把 state 从裸 context 里抽成一个显式对象之后：

- 可持久化 / checkpoint，断点续跑、崩溃恢复；
- 支持 human-in-the-loop：中断、审批、人工改写中间状态；
- 可分支 / 回溯 / 时间旅行调试；
- 多 agent 之间可共享同一份状态。

**LangGraph 怎么体现**：

- 把 agent 建成一张**有向图**：节点 = 步骤，边 = 转移；ReAct 循环就是图里的一个 cycle；
- 用显式 `State`（通常 `TypedDict`）在节点间流转，而不是把一切塞进 prompt 字符串；
- `checkpointer` 持久化每一步状态，天然支持 interrupt 与恢复；
- 条件边决定「继续循环还是走向 finish」。

### Q7 · 电商「自动比价并下单」Agent：安全性与鲁棒性

<div class="compare">
<div class="compare-col is-workflow">
<span class="compare-head">🛡️ 安全性（不可逆动作要设防）</span>
<ul>
<li>下单 / 付款这类<strong>不可逆动作强制人工确认</strong>（human-in-the-loop）；</li>
<li>消费上限：单笔 + 日累计额度；</li>
<li>站点 / 卖家白名单；</li>
<li>防 prompt injection：商品页、评论里可能藏恶意指令，observation 要隔离、<strong>绝不当 instruction 执行</strong>；</li>
<li>幂等：用 idempotency key 防重复下单；</li>
<li>最小权限 + 审计日志 + dry-run 模式（先空跑验证再真下单）。</li>
</ul>
</div>
<div class="compare-col is-agent">
<span class="compare-head">💪 鲁棒性</span>
<ul>
<li>超时 + 指数退避重试；</li>
<li>下单前做<strong>价格时效复核</strong>（比价结果可能已过期）；</li>
<li>observation 校验（schema / 合理性，挡住异常返回）；</li>
<li>失败回滚 / 补偿事务，避免半成品订单。</li>
</ul>
</div>
</div>
