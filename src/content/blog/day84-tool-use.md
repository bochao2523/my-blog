---
title: 'Tool Use'
description: 'Agent 中关于 Tool Use 的完整知识体系——从工具定义、调用协议到编排策略与工程实践。'
series: 'agent'
pubDate: 'Jul 1 2026'
---

<p class="lead">Tool Use 的本质：<strong>模型选动作，运行时执行，结果回填上下文</strong>。LLM 碰不到外部世界，工具是它伸出去的手；本篇从调用原理、两种协议、串并行编排到 observation 工程，把这条链路讲清楚。</p>

## Part 1：工具调用的原理

### 1.1 一条不变的循环

无论用 prompt 还是 function calling，底层都是同一件事：

<div class="cycle">
<span class="cycle-node">模型输出动作</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">运行时执行</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">回填 observation</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">模型续写</span>
<span class="cycle-note">控制权在「模型」和「你的代码」之间来回切换——模型永远不直接调 API</span>
</div>

职责切分很硬：**模型**只决定调哪个工具、传什么参数（或不调、直接回答）；**运行时**负责解析、校验、执行、回填；**工具**是纯函数 / API 封装。即模型只出「意图」，副作用、鉴权、超时、重试全在运行时一侧。

### 1.2 两种基本用法

同一循环，两种「接线方式」。最大区别是：**动作是普通文本，还是协议级的结构化字段。**

**① Prompt 协议**（ReAct 式）——在 prompt 里用文字约定格式，模型把动作当正文写出来：

```text
# 你在 system / few-shot 里约定好格式，模型生成：
Thought: 我需要查一下 Apple Remote
Action: Search[Apple Remote]
# ↑ 运行时设 stop="\nObservation"，模型停在这，你的代码执行后拼回：
Observation: Apple Remote 是一款遥控器……
```

**② Function Calling**——把工具 schema 随请求传给模型，模型在<strong>专用通道</strong>返回结构化调用，运行时按 `tool_call_id` 回填：

```jsonc
// 模型返回（不是正文，而是独立字段）：
{ "tool_calls": [{ "id": "call_1",
    "function": { "name": "search", "arguments": "{\"q\":\"Apple Remote\"}" } }] }
// 运行时执行后追加一条消息：
{ "role": "tool", "tool_call_id": "call_1", "content": "Apple Remote 是……" }
```

两者优缺点正好互补：

| | **Prompt 协议** | **Function Calling** |
|---|---|---|
| 优点 | 任何模型都能用、零依赖、格式可随意定制 | schema 强约束、参数自动校验、原生支持并行调用、解析稳 |
| 缺点 | 靠正则解析易碎、模型可能跑偏格式、要自己防注入 | 仅限支持该 API 的模型、格式被厂商锁死 |
| 用在 | 论文复现、快速原型、无 tool API 的模型 | 生产 Agent、多工具编排 |

<aside class="callout">
<p><strong>外壳不同，逻辑相同</strong>：都是「生成动作 → 停下交出控制权 → 执行 → 回填 → 续写」。Function calling 只是把 prompt 协议里靠约定和正则维持的格式，升级成了协议级的结构化字段。</p>
</aside>

### 1.3 串行与并行

一轮调一个还是多个工具，唯一的判断标准：<strong>没有上一步的 observation 就写不出下一步参数 → 串行；参数事先能定死 → 可并行。</strong>

<div class="cycle">
<span class="cycle-node">串行：调 A → 看结果 → 再决定 B</span>
<span class="cycle-note">B 的参数依赖 A 的返回（搜实体 → 再 lookup），必须等</span>
</div>

<div class="cycle">
<span class="cycle-node">并行：调 A ∥ B ∥ C → observation 一起回填</span>
<span class="cycle-note">A/B/C 互不依赖（多路检索、批量读文件），同轮发出、批量执行，省 latency</span>
</div>

实际任务常是**混合**：Plan 阶段标好 `depends_on`，无依赖的并行、有依赖的串行——即 Plan-and-Execute / DAG 编排的做法。

### 1.4 Observation 的逻辑

Observation 是工具执行的<strong>真实返回值</strong>，由运行时回填——绝不能让模型自己「想象」，否则 grounding 失效、退化成纯 CoT。它给下一步推理提供外部事实（搜到了什么、代码跑没跑通、API 报什么错）。

三类 observation，处理方式不同：

| 类型 | 示例 | 要点 |
|---|---|---|
| **成功结果** | 检索摘要、查询行、命令 stdout | 往往要截断 / 摘要——原始返回可能上万 token |
| **空 / 无信息** | 搜不到、0 条命中 | 也是有效信号，应原样回填，让模型换策略 |
| **错误** | 参数非法、超时、权限拒绝 | 当成 observation 回喂（不是抛异常中断），模型才有机会自我纠正 |

<aside class="callout">
<p><strong>⚠️ 两个硬规则</strong>：① 每一轮停在 Action / tool_call 之后，控制权交给运行时；② observation 越短越好，只要不丢决策所需的关键字段。</p>
</aside>

## Part 2：工程上的考量

原理简单，但工程上每个环节都有坑。按调用生命周期走一遍：<strong>定义 → 调用 → 返回 → 出错 → 幻觉</strong>。

### 2.1 定义工具时：写给模型看，不是给人看

工具的 name + description + schema 就是模型唯一的「说明书」——它选错工具、传错参数，十有八九是定义没写清。

<div class="gridcards">
<div class="gridcard"><b><span class="gc-ico">🏷️</span>命名即语义</b>名字要动词化、无歧义（<code>get_weather</code> 而非 <code>weather</code>）。功能重叠的工具会让模型反复纠结选哪个。</div>
<div class="gridcard"><b><span class="gc-ico">📝</span>描述写边界</b>不只说「做什么」，更要说<strong>何时该用、何时不该用</strong>、有什么前置条件，把容易混淆的工具区分开。</div>
<div class="gridcard"><b><span class="gc-ico">📐</span>schema 收紧</b>能用 enum 就别用自由字符串，必填项标 <code>required</code>，加默认值。约束越强，模型越难传错。</div>
<div class="gridcard"><b><span class="gc-ico">✂️</span>控制数量</b>工具一多，选择就退化。超过十几个考虑分组、按场景动态裁剪可见工具集。</div>
</div>

<aside class="callout">
<p><strong>💡 一句话</strong>：参数校验靠 schema，工具选型靠 description。两者都是 prompt 的一部分，要像调 prompt 一样反复打磨。</p>
</aside>

### 2.2 调用时：模型给的参数不可信

模型产出的 tool call 是「意图」，执行前必须当成<strong>不可信输入</strong>处理：

<div class="gridcards">
<div class="gridcard"><b><span class="gc-ico">✅</span>先校验再执行</b>按 schema 校验类型、范围、必填；不合法直接拦下，别带着脏参数去执行。</div>
<div class="gridcard"><b><span class="gc-ico">🔐</span>权限与副作用</b>写操作、删除、转账这类带副作用的工具，要鉴权、限流，必要时加人工确认（human-in-the-loop）。</div>
<div class="gridcard"><b><span class="gc-ico">🧨</span>防注入</b>参数可能被 prompt injection 污染。绝不直接拼进 SQL / shell / 文件路径，做参数化和白名单。</div>
<div class="gridcard"><b><span class="gc-ico">⏱️</span>超时与幂等</b>每个工具设超时；带副作用的调用尽量幂等，配合 <code>tool_call_id</code> 去重，防重试重复执行。</div>
</div>

### 2.3 返回 result 后：管好上下文

工具返回的原始数据往往又大又脏，直接回填会撑爆上下文、淹没关键信息：

| 手段 | 做什么 |
|---|---|
| **截断 / 摘要** | 只取前 K 条或先压缩，原始返回可能上万 token |
| **结构化抽取** | 只留模型决策需要的字段（标题、id、score），丢正文噪声 |
| **稳定格式** | 返回格式固定（JSON / 表格），模型更容易解析、不易误读 |
| **外部存储** | 大结果存盘，上下文里只放引用 / 摘要，需要时再取 |

<aside class="callout">
<p><strong>💡 经验法则</strong>：上下文里每个 token 都在和模型注意力抢资源。observation 越短越好，只要不丢决策所需信息。</p>
</aside>

### 2.4 出错时：把错误变成可纠正的信号

工具失败是常态（超时、限流、参数非法、服务挂了）。关键原则：<strong>别让异常崩掉整个 loop，把错误当成一条 observation 回喂，给模型自我纠正的机会。</strong>

<div class="cycle">
<span class="cycle-node">工具报错</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">包成 error observation</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">模型改参数 / 换工具重试</span>
<span class="cycle-note">错误信息要具体（「缺少 required 字段 date」），模型才知道怎么改</span>
</div>

但不能无限重试——配合 Part 1 的停止条件：**重试上限**、**重复检测**（连续几次同样的错就别再硬试）、对瞬时错误（限流、网络）做**退避重试**，对确定性错误（参数非法）直接回喂让模型改。

### 2.5 幻觉了怎么办

工具场景下的幻觉主要两类，对策不同：

| 幻觉类型 | 表现 | 对策 |
|---|---|---|
| **调用幻觉** | 调不存在的工具、编造参数、瞎填字段 | function calling 强 schema + 工具白名单；非法调用当 error observation 回喂，而非崩溃 |
| **结果幻觉** | 无视 observation，凭空编造工具「应该」返回什么 | 强制答案<strong>引用真实 observation</strong>；交付前做 grounding 校验，对不上就重来 |

<aside class="callout">
<p><strong>🧭 根因</strong>：幻觉往往不是模型「坏」，而是<strong>信息不足或定义不清</strong>——工具描述含糊、observation 缺失或太长被忽略。先把 2.1～2.3 做扎实，幻觉会少一大半；剩下的再靠校验和 reflection（见 day83）兜底。</p>
</aside>

## Part 3：工具数量太多怎么处理

### 3.1 为什么「多」会出问题

工具不是越多越好。全量塞进 prompt 会同时撞上两堵墙：

<div class="gridcards">
<div class="gridcard"><b><span class="gc-ico">🪙</span>占上下文</b>每个工具的 schema 都常驻 prompt，几十上百个工具能吃掉成千上万 token，挤压真正的任务空间，还每轮都付费。</div>
<div class="gridcard"><b><span class="gc-ico">😵</span>选择退化</b>候选越多、功能越重叠，模型越容易选错、来回纠结，准确率随工具数上升而下降。</div>
</div>

一句话：**模型每一轮真正需要的，往往只是全部工具里的一小撮。** 思路就是——别让它每次都面对全集。

### 3.2 核心解法：按需暴露

<div class="cycle">
<span class="cycle-node">全部工具</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">按当前任务筛选</span>
<span class="cycle-arrow">→</span>
<span class="cycle-node">只把 Top-K 暴露给模型</span>
<span class="cycle-note">从「让模型在 100 个里选」变成「先缩到 5 个，再让它选」</span>
</div>

| 策略 | 做法 | 适合 |
|---|---|---|
| **静态分组** | 按业务域分组（文件类 / 网络类 / DB 类），按场景预先决定加载哪组 | 工具集稳定、场景边界清晰 |
| **语义检索（RAG）** | 把工具描述向量化，用 query 检索出 Top-K 相关工具再注入 prompt | 工具量大、动态、超过几十个 |
| **分层 / 路由** | 上层 router 先选「类别」或子 agent，下层只暴露该类下的工具 | 工具天然有层级、可拆多 agent |
| **渐进暴露** | 默认给少量常用工具，模型显式请求时再加载更多 | 长尾工具多、多数用不到 |

### 3.3 配套手段

光筛选还不够，再叠几层：

<div class="gridcards">
<div class="gridcard"><b><span class="gc-ico">🧰</span>合并冗余</b>把功能相近的小工具合并成一个带参数的大工具（如多个查询合成一个 <code>query(type, ...)</code>），减少候选数。</div>
<div class="gridcard"><b><span class="gc-ico">🧱</span>命名空间</b>用前缀 / 分组让名字不冲突、语义清晰，降低模型在相似工具间的混淆。</div>
<div class="gridcard"><b><span class="gc-ico">🤖</span>多 Agent 拆分</b>工具实在太多，就按职责拆成多个子 agent，每个只管自己那一小套工具。</div>
<div class="gridcard"><b><span class="gc-ico">📊</span>用量驱动</b>按调用频率排序 / 裁剪，常用的常驻、长尾的走检索。</div>
</div>

<aside class="callout">
<p><strong>🧭 一句话</strong>：解决工具过多的本质，是<strong>把「选择」的难题从模型转移到检索 / 路由系统</strong>——让模型每轮只面对一个小而精准的工具集。这也正是 MCP 等协议要标准化的能力。</p>
</aside>

