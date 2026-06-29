---
title: 'Plan and Execute & REWOO'
description: '从 Planner-Executor 分离出发，理解 Plan-and-Execute 如何将复杂任务拆解为可执行步骤，以及 REWOO 如何通过「无观察推理」降低 Agent 的 token 开销。'
pubDate: 'Jun 29 2026'
---

<p class="lead"><strong>ReWOO = Reasoning WithOut Observation</strong>。把 ReAct 里推理和观察绑死的循环拆开：Planner 先一次性出计划（不看任何工具返回），Worker 批量执行，Solver 汇总答案。</p>

<aside class="callout">
<p><strong>出处</strong>：Xu et al., <em>ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models</em>, 2023（arXiv:2305.18323）。</p>
</aside>

## Part 1：Introduction to ReWOO

### 1.1 三个角色 + 两次 LLM

<div class="cycle">
<span class="cycle-node">Planner 出计划</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">Worker 调工具</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">Solver 出答案</span>
<span class="cycle-note">只有 Planner 和 Solver 调 LLM；Worker 只跑工具</span>
</div>

<div class="steps">
<div class="step">
<div class="step-num">P</div>
<div class="step-body"><p><strong>Planner</strong>　接收问题，输出带占位符的完整计划，零 observation</p></div>
</div>
<div class="step">
<div class="step-num">W</div>
<div class="step-body"><p><strong>Worker</strong>　按计划调工具，把真实结果填回占位符</p></div>
</div>
<div class="step">
<div class="step-num">S</div>
<div class="step-body"><p><strong>Solver</strong>　拿「计划 + 证据」做最后一轮推理，输出答案</p></div>
</div>
</div>

对比 ReAct 每步「想→做→看」循环调一次 LLM，ReWOO 固定 **2 次 LLM + N 次工具**，中间 observation 不进 Planner 上下文。

### 1.2 占位符：#E1、#E2……

Planner 输出的计划用 `#Ei` 给每步证据命名：

<div class="trace">
<div class="trace-row"><p><strong>Plan</strong></p><p><code>#E1 = search[Apple Remote]</code> → 答案在 #E1 里<br><code>#E2 = search[Front Row media program]</code> → 什么设备能控制它？<br><code>#E3 = lookup[Front Row, devices]</code></p></div>
</div>

- `#Ei = tool[...]`：声明这一步去拿什么证据
- 后步参数可引用前步占位符，串起依赖链
- Worker 跑完后 `#E1` → 真实文本，再交给 Solver

本质：Planner 在信息不全时，用变量名把步骤串起来，不必等结果回来再规划。

### 1.3 优劣一览

<div class="compare">
<div class="compare-col is-agent">
<span class="compare-head">优势</span>
<ul>
<li><strong>可并行</strong>：无依赖的 #Ei 同时调工具</li>
<li><strong>全局规划</strong>：一次性看到任务全貌，非贪心逐步决策</li>
<li><strong>Planner 可用小模型</strong>：规划偏结构化，Solver 才需强推理</li>
<li><strong>推理/观察解耦</strong>：省 token，上下文不被 observation 滚雪球撑爆</li>
</ul>
</div>
<div class="compare-col is-workflow">
<span class="compare-head">劣势</span>
<ul>
<li><strong>无法适应突变</strong>：#E1 返回空或意外内容，后续全建立在错假设上</li>
<li><strong>错误级联</strong>：没有 ReAct 那种「看到意外立刻改道」的回路</li>
<li><strong>Planner 负担重</strong>：零外部信息就要猜对整条路径</li>
</ul>
</div>
</div>

<aside class="callout">
<p><strong>⚖️ 一句话</strong>：ReWOO 用<strong>适应性</strong>换<strong>效率</strong>——适合步骤可预期、工具返回稳定的任务；环境不确定时 ReAct 仍更稳。Part 2 看 Plan-and-Execute 如何折中。</p>
</aside>

## Part 2：Plan-and-Execute

Plan-and-Execute 是 ReAct 和 ReWOO 之间的折中：**先有计划的全局感，又能在执行中看结果、必要时改计划**。

### 2.1 链路：Planner → Executor → Replanner

<div class="cycle">
<span class="cycle-node">Planner 出计划</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">Executor 执行一步</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">Replanner 评估</span>
<span class="cycle-arrow">↻</span>
<span class="cycle-node">结束 / 下一步 / 重规划</span>
</div>

<div class="steps">
<div class="step">
<div class="step-num">P</div>
<div class="step-body"><p><strong>Planner</strong><span class="step-tag">强模型</span>　把用户目标拆成有序步骤列表，定好整体方向</p></div>
</div>
<div class="step">
<div class="step-num">E</div>
<div class="step-body"><p><strong>Executor</strong><span class="step-tag">较弱模型</span>　只盯<strong>当前这一步</strong>：选工具、调工具、拿 observation</p></div>
</div>
<div class="step">
<div class="step-num">R</div>
<div class="step-body"><p><strong>Replanner</strong>　对照原计划目标 + 已执行结果，判断：结束了？继续下一步？还是计划已经不对、要重规划？</p></div>
</div>
</div>

Executor 每跑完一步，Replanner 介入决策：

<div class="steps">
<div class="step">
<div class="step-num">✓</div>
<div class="step-body"><p><strong>结束</strong>　目标已达成，输出最终答案</p></div>
</div>
<div class="step">
<div class="step-num">→</div>
<div class="step-body"><p><strong>继续</strong>　当前步骤 OK，按计划执行下一步</p></div>
</div>
<div class="step">
<div class="step-num">↻</div>
<div class="step-body"><p><strong>重规划</strong>　结果与预期不符（工具失败、返回意外、路径走不通），带着已有 observation 重新出计划，再交给 Executor</p></div>
</div>
</div>

### 2.2 为什么是 ReAct 和 ReWOO 的中间版

| | 全局计划 | 执行中看 observation | 可重规划 |
|---|---|---|---|
| ReAct | ✗ | 每步都看 | 隐式（下一步 thought 改道） |
| **Plan-and-Execute** | ✓ | 每步都看 | ✓ 显式 Replanner |
| ReWOO | ✓ | Planner 不看 | ✗ |

- **ReAct**：无全局计划，每步贪心决策；适应性强，但方向易飘、token 贵
- **ReWOO**：计划一次定死、推理与观察完全解耦；效率高，但无法中途改道
- **Plan-and-Execute**：先有计划的全局感，又在每步执行后留一个「要不要改计划」的口子

模型分工也体现了这种折中：Planner 用强模型扛「拆任务、定方向」；Executor 用弱模型专注「当前这一步怎么调工具」——把贵的推理留给规划，把便宜的执行留给逐步落地。Replanner 补上 ReWOO 缺的环节：计划不是一次性的，环境变了可以重来。

<aside class="callout">
<p><strong>⚖️ 一句话</strong>：ReWOO 砍掉 observation 换效率，ReAct 拥抱 observation 换灵活；Plan-and-Execute 保留计划骨架，又在每步执行后留一个「要不要改计划」的口子。</p>
</aside>

## Part 3：踩坑、解法与选型

### 3.1 ReWOO 踩坑

<div class="qbox">
<div class="qbox-q">①</div>
<div class="qbox-body"><strong>无 replan，E1 出错 → E2、E3 全废</strong><p>ReWOO 结构上不能改计划。<code>#E1</code> 返回空、超时或内容离谱时，后面引用 <code>#E1</code> 的步骤都在错地基上盖楼。</p></div>
</div>

<div class="compare">
<div class="compare-col is-workflow">
<span class="compare-head">坑</span>
<p>错误级联，Solver 仍按「#E1 有效」做汇总，幻觉答案。</p>
</div>
<div class="compare-col is-agent">
<span class="compare-head">解法</span>
<ul>
<li><strong>Worker 守卫</strong>：空结果 / 超时 / 格式异常 → 标记 <code>#E1 = FAILED</code>，不填假内容</li>
<li><strong>Solver 容错 prompt</strong>：明确「某 #Ei 为 FAILED 时，禁止引用该证据；在缺失信息下推理，或诚实声明无法回答」</li>
</ul>
</div>
</div>

<div class="qbox">
<div class="qbox-q">②</div>
<div class="qbox-body"><strong>占位符解析脆弱</strong><p>自由文本里的 <code>#E1</code>、<code>#e1</code>、漏写 <code>#</code>、循环引用（#E2 依赖 #E3，#E3 又依赖 #E2）——Worker 一解析就崩。</p></div>
</div>

<div class="compare">
<div class="compare-col is-workflow">
<span class="compare-head">坑</span>
<p>变量名拼错、循环依赖、正则解析不稳定。</p>
</div>
<div class="compare-col is-agent">
<span class="compare-head">解法</span>
<ul>
<li>Planner 改用<strong>结构化输出</strong>（JSON / function calling），字段固定为 <code>{id, tool, args, depends_on}</code></li>
<li>执行前做依赖图校验：检测环、未定义引用，不合法直接拒跑</li>
</ul>
</div>
</div>

### 3.2 Plan-and-Execute 踩坑

<div class="qbox">
<div class="qbox-q">①</div>
<div class="qbox-body"><strong>何时重规划：不重则钝，太勤则退化</strong><p>不重规划 → 和 ReWOO 一样，错了只能硬扛，精度掉；每步都重规划 → 全局计划名存实亡，退化成 ReAct，token 也回来了。</p></div>
</div>

<div class="compare">
<div class="compare-col is-workflow">
<span class="compare-head">坑</span>
<p>重规划阈值难定：太松漏错，太紧白规划。</p>
</div>
<div class="compare-col is-agent">
<span class="compare-head">解法</span>
<p>每步 Executor 完成后，Replanner 看一组<strong>可量化信号</strong>再决策：</p>
<ul>
<li>工具调用是否成功（HTTP 码、空结果、异常）</li>
<li>observation 与当前步预期是否匹配（可用小模型或规则打分）</li>
<li>Executor 自报置信度 / 连续失败次数</li>
</ul>
<p>仅当信号超阈值才触发 replan；否则继续下一步。</p>
</div>
</div>

<div class="qbox">
<div class="qbox-q">②</div>
<div class="qbox-body"><strong>Planner 步子太大，Executor 一步搞不定</strong><p>Planner 写出「分析竞品并生成完整报告」，Executor 只负责当前一步——粒度对不上，执行失败或敷衍带过。</p></div>
</div>

<div class="compare">
<div class="compare-col is-workflow">
<span class="compare-head">坑</span>
<p>计划步太粗，弱模型 Executor 无法单步闭环。</p>
</div>
<div class="compare-col is-agent">
<span class="compare-head">解法</span>
<ul>
<li><strong>Few-shot 约束 Planner</strong>：示例里每步都是「一次工具调用可完成」的原子动作</li>
<li><strong>Executor 反馈 replan</strong>：检测「本步无法完成」时，不硬做，显式请求 Planner 把当前步拆细</li>
</ul>
</div>
</div>

<div class="qbox">
<div class="qbox-q">③</div>
<div class="qbox-body"><strong>重规划振荡 / 状态膨胀</strong><p>反复「plan → fail → replan → fail」空转；或每轮 replan 把全部历史 observation 塞回 Planner，上下文滚雪球。</p></div>
</div>

- 设<strong>重规划次数上限</strong>，超限降级 ReAct 或人工介入
- Replanner 只传<strong>摘要后的关键 observation</strong>，不全量回填

### 3.3 选型对比

| 维度 | ReWOO | Plan-and-Execute |
|---|---|---|
| **适合的任务** | 步骤依赖能在开头想清楚；流程相对固定 | 长程多步；中间结果会改变后续决策 |
| **核心收益** | 仅 2 次 LLM；省 token；工具可并行；易蒸馏小模型 | 全局方向 + 适应性兼顾；「大模型规划 + 小模型执行」控成本 |
| **最该警惕的坑** | 出错不能 replan；变量解析脆弱 | 重规划振荡或死循环；状态膨胀 |
| **不适合** | 需探索试错、中途改道的任务 | 极简单步任务；对延迟极度敏感的场景 |
| **典型应用** | 多跳 QA；固定「检索 + 计算」流水线；确定性 ETL | 研究型任务；多阶段数据分析；复杂工具编排 |

<aside class="callout">
<p><strong>🧭 选型直觉</strong>：能在开工前画出<strong>确定性依赖图</strong> → ReWOO；需要<strong>边走边看</strong>、中途发现会改写后续步骤 → Plan-and-Execute；一两步能搞定 → 单次 LLM 或轻量 ReAct 就够，别上大炮。</p>
</aside>
