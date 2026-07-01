---
title: '什么是 Agent'
description: '从概念定义到核心能力，理解 Agent 为何成为大模型应用的关键范式。'
series: 'agent'
pubDate: 'Jun 25 2026'
---

## Part 1：Agent 的定义

<p class="lead"><code>Agent</code> 不是一次性的问答工具，而是一个<strong>持续运转的闭环系统</strong>。给定一个任务，它自己决定如何搜索信息、调用工具、生成输出、读写记忆、进行反思——每一步都不是预设脚本，而是模型根据当前状态当场判断的。</p>

### 闭环框架

Agent 的核心循环不是一条直线，而是一个不断转动的环：

<figure class="loop">
<svg viewBox="0 0 420 340" role="img" aria-label="Agent 闭环：感知、规划、行动、观察、记忆，循环往复">
<defs>
<marker id="loopArrow" viewBox="0 0 6 6" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
<path d="M0,0 L6,3 L0,6 Z" fill="#14b8a6"></path>
</marker>
</defs>
<g fill="none" stroke="#5eead4" stroke-width="2.5" marker-end="url(#loopArrow)">
<path d="M239.0,53.6 A120,120 0 0 1 311.8,106.4"></path>
<path d="M329.7,161.6 A120,120 0 0 1 301.9,247.2"></path>
<path d="M254.9,281.3 A120,120 0 0 1 165.1,281.3"></path>
<path d="M118.1,247.2 A120,120 0 0 1 90.3,161.6"></path>
<path d="M108.2,106.4 A120,120 0 0 1 181.0,53.6"></path>
</g>
<g font-size="22" fill="#99f6e4" font-weight="700" text-anchor="middle">
<text x="210" y="164">↻</text>
</g>
<g font-size="12" fill="#5eead4" text-anchor="middle">
<text x="210" y="186">闭环</text>
</g>
<g font-size="15" font-weight="600" fill="#115e59" text-anchor="middle" dominant-baseline="central">
<g><rect x="178" y="34" width="64" height="32" rx="16" fill="#fff" stroke="#5eead4" stroke-width="1.5"></rect><text x="210" y="51">感知</text></g>
<g><rect x="292" y="117" width="64" height="32" rx="16" fill="#fff" stroke="#5eead4" stroke-width="1.5"></rect><text x="324" y="134">规划</text></g>
<g><rect x="249" y="251" width="64" height="32" rx="16" fill="#fff" stroke="#5eead4" stroke-width="1.5"></rect><text x="281" y="268">行动</text></g>
<g><rect x="107" y="251" width="64" height="32" rx="16" fill="#fff" stroke="#5eead4" stroke-width="1.5"></rect><text x="139" y="268">观察</text></g>
<g><rect x="64" y="117" width="64" height="32" rx="16" fill="#fff" stroke="#5eead4" stroke-width="1.5"></rect><text x="96" y="134">记忆</text></g>
</g>
</svg>
<figcaption>观察与记忆的结果回流到「感知」，循环往复，直到任务完成</figcaption>
</figure>

### 五个环节

<div class="steps">
<div class="step">
<div class="step-num">1</div>
<div class="step-body"><p><strong>感知（Perception）</strong>——把环境状态读入上下文。在 LLM Agent 中，这通常意味着：用户输入、上一步工具的返回值、代码执行结果等，被<strong>拼接进当前 context</strong>，成为模型下一步决策的依据。</p></div>
</div>
<div class="step">
<div class="step-num">2</div>
<div class="step-body"><p><strong>规划（Planning）</strong><span class="step-tag">灵魂</span>——基于当前状态，决定「接下来做什么」。典型流程是 <code>提出候选 → 评估 → 选择</code>，从多个可能的动作中挑出一个执行。</p></div>
</div>
<div class="step">
<div class="step-num">3</div>
<div class="step-body"><p><strong>行动（Action）</strong>——执行被选中的动作，分两类：<strong>外部行动</strong>（调用工具 / API、操作网页、控制物理设备）与<strong>内部行动</strong>（推理、读写记忆）。注意，推理本身也是一种行动。</p></div>
</div>
<div class="step">
<div class="step-num">4</div>
<div class="step-body"><p><strong>观察（Observation）</strong>——行动改变了环境，Agent 收到新的反馈（ground truth），据此判断任务进展，决定是否需要纠错、重试或调整计划。</p></div>
</div>
<div class="step">
<div class="step-num">5</div>
<div class="step-body"><p><strong>记忆（Memory）</strong>——跨步骤、跨会话保留信息：<strong>工作记忆</strong>是当前上下文；<strong>长期记忆</strong>则沉淀经验（episodic）、知识（semantic）与技能（procedural）。</p></div>
</div>
</div>

### Agent 与 Pipeline 的本质区别

<aside class="callout">
<p><strong>关键点</strong>：在这个循环里，「选择下一步动作」是<strong>模型当场做出的判断</strong>，而不是写死的 <code>if/else</code> 分支。正是这种动态决策，构成了 Agent 与传统 Pipeline 的根本差异。</p>
</aside>

Pipeline 的路径在部署时就固定了：先检索、再生成、再格式化——无论输入是什么，走同一条路。Agent 则面对不同任务会走出**完全不同的执行路径**：有的任务需要先搜资料，有的直接写代码，有的要反复试错。同一个 Agent，循环可以跑很多轮，直到任务完成或达到终止条件。

## Part 2：本质区别——谁决定下一步

很多人把「用了大模型 + 有循环」就叫 Agent，这是误解。真正的分水岭是**控制流（control flow）归谁所有**。

沿自主程度从低到高，有三档：

| 类型 | 控制流由谁决定 | 典型例子 | 成本 / 可预测性 |
|---|---|---|---|
| 单次问答（Task） | 无流程，一锤子买卖 | 「把这段话总结一下」 | 最低 / 完全可预测 |
| 固定 Workflow | 工程师预先编排，模型只填空 | 「先翻译 → 再校对 → 再排版」 | 中 / 可预测、有界 |
| Autonomous Agent | 模型运行时自主决定 | 「整理一份竞品定价对比报告」 | 高 / 不可预测 |

<blockquote class="pullquote">
Workflow 是你拥有管道（you own the plumbing），Agent 是模型拥有管道。其余所有 tradeoff 都派生自这一个结构性选择。
<cite>—— Anthropic, Barry Zhang</cite>
</blockquote>

### 具体对比

同一个任务：「查竞品最近的定价，然后写份对比。」两种做法走出完全不同的路径——

<div class="compare">
<div class="compare-col is-workflow">
<span class="compare-head">Workflow 版</span>
<p>代码写死三步——调搜索 API 拿前 5 条 → 喂给模型总结 → 套模板输出。永远这三步，无论中间结果好不好。</p>
</div>
<div class="compare-col is-agent">
<span class="compare-head">Agent 版</span>
<p>先搜，发现某家信息不全，自己决定换关键词再搜一次；发现有套餐差异，又去抓官网详情页；最后才写。搜几次、搜什么，<strong>临场决定</strong>。</p>
</div>
</div>

同一个任务，Workflow 的路径在写代码时就定死了；Agent 的路径在运行时才浮现出来。这就是 Part 1 里「模型拥有管道」在工程上的含义——不是多调几次 API，而是**下一步走哪条路，由模型说了算**。

## Part 3：Anthropic 的五种 Workflow 模式

Anthropic 在 [*Building Effective Agents*](https://www.anthropic.com/research/building-effective-agents) 里提炼了五种生产中常见的 Workflow 模式。它们的共同点是：**控制流仍由人编排**，但通过组合多次 LLM 调用来提升能力。

在讲五种模式之前，先记住它们共同的基础积木——**增强型 LLM（Augmented LLM）**：一个 LLM，加上检索、工具、记忆三种增强能力。五种模式都是在回答：拿到这块积木之后，**怎么编排多次调用**。

### 1. Prompt Chaining（提示链）

把一个任务拆成固定的若干步，每一步的 LLM 调用处理上一步的输出。可以在步骤之间插入程序化的「门」（gate）做检查。

- **何时用**：任务能干净地拆成固定的子步骤，用准确率换延迟划算
- **例子**：先生成文档大纲 → 校验大纲是否符合要求 → 再写正文；先写营销文案 → 再翻译成另一种语言

### 2. Routing（路由）

先用一次 LLM 调用对输入做分类，再把它分发到专门的后续处理。实现了「关注点分离」。

- **何时用**：输入有清晰的类别，且分类本身能做得准
- **例子**：客服问题分流（退款 / 技术支持 / 一般咨询）走不同的 prompt 和工具；把简单问题路由到小模型、难问题路由到大模型，省成本

### 3. Parallelization（并行化）

多个 LLM 同时处理任务，结果用程序聚合。有两个变种：

**Sectioning（切分）**——把任务拆成互相独立的子任务并行跑。

- 例：一个模型回答用户问题，另一个模型同时做内容安全审查（guardrail）

**Voting（投票）**——同一个任务跑多次，拿多样化结果再聚合。

- 例：用多个 prompt 审查同一段代码的漏洞，多数票决定是否有问题

- **何时用**：子任务可拆分并行以提速，或需要多视角 / 多次采样提升置信度

### 4. Orchestrator-Workers（编排者-工人）

一个中心 LLM（orchestrator）动态拆解任务、分派给多个 worker LLM，最后综合结果。

与并行化的区别：子任务**不是预先定义好的**，而是由 orchestrator 根据具体输入临场决定。

- **何时用**：复杂任务，但你无法预测要拆成哪些子任务
- **例子**：编码工具同时改动多个文件；复杂检索任务从多个来源汇总信息

<aside class="callout">
<p><strong>注意</strong>：这是最接近 Agent 的一种 Workflow——orchestrator 已经在「动态决定子任务」了。它和 Agent 的边界很模糊，差别在于整体结构仍是固定的「拆解 → 分派 → 综合」，而非完全开放的自由循环。</p>
</aside>

### 5. Evaluator-Optimizer（评估者-优化者）

一个 LLM 生成结果，另一个 LLM 给出评价和反馈，在循环里迭代改进。

- **何时用**：存在清晰的评估标准，且迭代精修能带来可衡量的提升
- **两个适配信号**：① 人能明确说出反馈，且这反馈能让结果变好；② LLM 自己也能给出这种反馈
- **例子**：文学翻译（译者 LLM 初译，评估 LLM 指出未捕捉的语义细节）；需要多轮检索的复杂搜索（评估者决定是否还要继续搜）

### 五种模式一览

| 模式 | 核心思路 | 控制流归属 |
|---|---|---|
| Prompt Chaining | 固定步骤串联 | 人定义步骤顺序 |
| Routing | 分类后分发 | 人定义路由表 |
| Parallelization | 并行执行再聚合 | 人定义拆分 / 投票策略 |
| Orchestrator-Workers | 中心动态拆解分派 | 人定义「拆解-分派-综合」框架 |
| Evaluator-Optimizer | 生成-评估循环 | 人定义评估标准和终止条件 |

五种模式从简单到复杂，自主程度逐步升高，但**管道的骨架始终握在工程师手里**。那反过来问：什么时候**不该**上 Agent？

## Part 4：何时不该用 Agent

<aside class="callout">
<p><strong>总原则</strong>（Anthropic）：先找最简单的方案，只在必要时才增加复杂度。很多场景甚至不需要 agentic system——单次 LLM 调用 + 检索 + few-shot 就够了。</p>
</aside>

### 三个实操判断

下面三问来自 Barry Zhang——任何一问指向「能预先编排」，就别上 Agent：

<div class="qbox">
<div class="qbox-q">01</div>
<div class="qbox-body"><strong>任务是否模糊到无法预先画决策树？</strong><p>能画出来就别上 Agent——固定 Workflow 在准确率、可控性、成本上都更优。</p></div>
</div>
<div class="qbox">
<div class="qbox-q">02</div>
<div class="qbox-body"><strong>任务价值是否高到值得 token 开销？</strong><p>Agent 要探索，探索烧钱。粗算 10 美分 / 任务 ≈ 3–5 万 token——这是 Workflow 的地盘。</p></div>
</div>
<div class="qbox">
<div class="qbox-q">03</div>
<div class="qbox-body"><strong>错误成本有多高？</strong><p>错误高危且难检测时，自主性就变成负债。read-only 权限和 human-in-the-loop 是真实的缓解手段，但也会限制能 scale 的上限。</p></div>
</div>

### Agent 的代价

任务确定、可枚举的场景，用 Workflow 或纯 RAG 更可控也更省钱。Agent 的代价是四样东西：

| 代价 | 含义 |
|---|---|
| 成本 | 多轮调用、探索性搜索，token 开销远高于固定流程 |
| 延迟 | 每多一轮 loop，用户多等一次 |
| 不可预测性 | 同任务不同次可能走完全不同的路径 |
| 错误累积 | 每轮 loop 都乘上最弱环节的失败率——固定 Workflow 不会有这种复合风险 |

### 编码：少数成立的 Agent 场景

为什么编码是少数真正适合 Agent 的场景？它同时满足几个硬条件：

- **任务模糊**：无法预先枚举所有子步骤
- **输出价值明确**：写对代码的收益足够覆盖探索成本
- **前沿模型擅长**：当前最强模型在代码任务上表现突出
- **可自动验证**：单元测试能判断对错——满足「模型能从自己的错误中恢复」这个硬前提

反过来想：如果任务没有清晰的验证信号，Agent 跑了几轮 loop，模型却不知道自己错了，错误会在每一步**复合放大**——这是选架构时最该警惕的风险。

<aside class="callout">
<p><strong>收束</strong>：Agent 不是更高级的默认选项，而是<strong>在控制流无法预先编排时的最后手段</strong>。先问能不能用 Workflow 解决；能解决，就别让模型拥有管道。</p>
</aside>
