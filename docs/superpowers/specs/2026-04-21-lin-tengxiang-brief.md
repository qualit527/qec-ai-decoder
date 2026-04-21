# 林腾祥任务说明书

你对应的模型是 `Codex`。

你的角色可以简单理解成：**把 AutoQEC 里最核心的 AI predecoder 真正实现出来，并让它能接到经典 QEC decoder 上工作。**

你不是只负责“写几个模型模板”。你的核心价值在于：
- 实现 predecoder 的 DSL 和编译路径。
- 搭好训练与评估 Runner。
- 把神经网络输出稳定地接到 MWPM / OSD 这些经典后端上。

## 1. 这三天你具体要做什么

### Day 1: 把 predecoder 的表达能力搭起来

你先不要追求 fancy 模型，而是先把“能表达、能编译、能实例化”这件事做好。

你要做的事：
- 实现 `dsl_schema.py`，把 predecoder 的可配结构定义清楚。
- 实现 `dsl_compiler.py`，把 YAML config 编译成可运行的 `nn.Module`。
- 加入 3 个 GNN seed template 和 3 个 Neural-BP seed template。
- 定义 predecoder 的输入输出契约，至少要明确：
  - 输入张量是什么
  - 输出是 `hard_flip` 还是 `soft_priors`
  - 输出如何传给 classical backend

Day 1 的最低交付物：
- DSL 能描述模型。
- compiler 能把配置变成模型。
- 至少有若干可实例化的种子结构。

### Day 2: 把训练 Runner 和后端连接起来

这一天你最关键的目标，是让模型不是“摆设”，而是真的能训练、能评估、能接到 decoder 后面。

你要做的事：
- 实现 Runner 的 train / eval 主流程。
- 加入 FLOPs 统计。
- 加入 `RunnerSafety`，至少包括 wall-clock cutoff、VRAM 预检查、NaN 防护。
- 把 predecoder 输出接到 MWPM 或 OSD 上：
  - `hard_flip` 模式下怎么修改 syndrome
  - `soft_priors` 模式下怎么生成 priors
- 解决第一轮 end-to-end 跑起来时出现的 compile/runtime 问题。

Day 2 的最低交付物：
- 一条能真正训练模型并产出评估结果的 Runner 路径。
- predecoder + classical backend 能联通。
- 至少一次真实的 `Δ_LER` 输出。

### Day 3: 支撑 Demo 1 和 Demo 2 的模型主线

这一天你的重点是稳定性和交付，不是继续大改架构。

你要做的事：
- 保障 Demo 1 的 predecoder 训练与推理链路稳定。
- 参与 Demo 2 的 `bb72` 尝试，确保 predecoder 和 qLDPC 后端接口不崩。
- 收尾 Runner、DSL、CLI 和 fallback config。
- 当某个模型结构跑不动时，快速给出能继续推进演示的替代配置。

Day 3 的最低交付物：
- Demo 1 能依赖你的 Runner 和 predecoder 路径稳定运行。
- Demo 2 至少有 dev-profile 尝试路径。
- 别人能够基于你的 DSL 配出一个模型并跑起来。

## 2. 你和另外两个人怎么配合

- 你要和陈嘉汉对齐 orchestrator 调用 Runner 的接口，因为你的代码最终要被主流程驱动。
- 你要和谢金谷对齐 checkpoint、metrics、config 落盘格式，因为验证模块要读这些输出。
- 你是模型主实现负责人，所以当“模型为什么接不上 classical backend”时，这个问题主要是你来收口。

最需要你盯住的协作点：
- DSL 字段和实际模型实现是否一一对应
- predecoder 输出和 backend 输入是否定义清楚
- 训练产物是否足够让 verify 复现
- profile 切换时，dev / prod 行为是否一致

## 3. 你会碰到的 QEC 概念解释

下面这些概念，是你这条线最核心的。你做的是整个项目里最贴近“AI decoder 本体”的部分，所以这些术语你要真正吃透到工程层面。

### Predecoder

`predecoder` 是这个项目最核心的对象。

它不是最终单独完成全部纠错的 decoder，而是一个放在 classical backend 前面的神经网络模块。它的作用是先帮后端“清洗输入”或者“提供更好的先验信息”。

### Classical backend

`classical backend` 是 predecoder 后面的传统解码器。

在这个项目里：
- surface code 方向通常接 `MWPM`
- qLDPC 方向通常接 `OSD`

这样设计的好处是：哪怕神经网络效果一般，后面的经典解码器仍然能保证整个流程有一个可靠下限。

### GNN

`GNN` 是 graph neural network。

在 QEC 场景里，很多信息天然是图结构，比如 stabilizer、variable node、检测事件之间的关系，所以用图网络来做 predecoder 是很自然的一条路。

### Neural-BP

`Neural-BP` 可以理解成“带可学习参数的 belief propagation 变体”。

它保留了 BP 的结构化推断味道，但把一些规则变成可以学的参数。对你来说，它是另一条很重要的 predecoder family。

### DSL

`DSL` 是 domain-specific language，也就是领域专用配置语言。

在这里，它的作用不是炫技，而是把“哪些模型结构允许 agent 搜索”这件事固定下来。你负责的 DSL，本质上是在定义 agent 可以探索的 predecoder 设计空间。

### `dsl_compiler`

`dsl_compiler` 的作用是把 DSL 配置真正落成一个神经网络实例。

换句话说，DSL 是“设计图”，compiler 是“把设计图变成机器”的工序。

### `hard_flip`

`hard_flip` 模式表示 predecoder 直接对 syndrome 或相关离散状态做硬性的翻转或修正。

你可以把它理解成：模型直接说“这里改掉”。这种方式更直接，但自由度也更硬。

### `soft_priors`

`soft_priors` 模式表示 predecoder 不直接改最终输入，而是给后端一个“哪些位置更可能出错”的软信息。

你可以把它理解成：模型不替后端做决定，而是给后端更好的参考。

### Prior

`prior` 是先验信息。

在解码里，它表示“还没真正推断之前，我们先认为哪些错误更可能发生”。如果你的 predecoder 能把 prior 估得更准，后面的经典 decoder 通常就能做得更好。

### MWPM

`MWPM` 是 minimum-weight perfect matching。

它是 surface code 方向很经典的后端解码方法。你不一定现在就要吃透算法细节，但必须知道：如果你输出的是 `hard_flip` 或 `soft_priors`，最后都要能合理地被这个后端消费。

### OSD

`OSD` 是 ordered statistics decoding。

在 qLDPC 方向，它扮演 classical backend 的角色之一。你要保证自己的 predecoder 输出形式，能和这类后端配合起来。

### Syndrome cleaning

所谓 `syndrome cleaning`，就是让 predecoder 先把 syndrome 中不干净、不稳定、容易误导后端的部分处理一下。

如果 `hard_flip` 路线成立，本质上就是在做这件事。

### Parameter count

`params` 是模型参数量。

在你这里，它不只是一个工程数字，还和 Pareto 分析直接相关。因为一个很准但非常重的模型，不一定比一个稍弱但非常轻的模型更有价值。

### FLOPs

`FLOPs` 是浮点运算量，可以粗略理解成推理成本指标。

这个项目把它当 latency proxy，所以你不能只让模型变准，还要知道它贵不贵。

### `Δ_LER`

`Δ_LER` 表示 predecoder 加进去之后，逻辑错误率相对 baseline 改善了多少。

它是判断“你做的神经网络到底有没有帮助”的核心指标。

### Runner

`Runner` 是真正执行训练和评估的非 LLM 部分。

它是你的主战场之一。因为再好的 DSL，如果不能稳定执行训练、评估、统计 FLOPs、保存 checkpoint，就没有研究价值。

### `RunnerSafety`

`RunnerSafety` 是保护机制。

它的意义不是限制创新，而是避免 agent 配出一个把机器跑挂、把训练跑炸的配置。比如 OOM、训练 NaN、wall-clock 超时，都需要这里兜住。

## 4. 你这几天最容易踩的坑

- DSL 写得很漂亮，但 compiler 和实际模型实现对不上。
- 模型能训练，但输出定义不清，接不上 MWPM / OSD。
- 只顾着做更复杂的模型，没有优先保证第一条可跑通路径。
- Runner 没有把 checkpoint、metrics、FLOPs 这些基础产物存好，导致后面无法验证和复盘。

## 5. 你的完成标准

如果到第 3 天结束，你能做到下面这些，就算你这条线完成得不错：
- 别人能根据 DSL 配出并实例化一个 predecoder。
- 至少有一条 predecoder + classical backend 的完整训练评估链路。
- Demo 1 的模型路径稳定可用。
- 当别人问“AI 部分到底做了什么”时，你能明确回答输入、输出、训练、评估、后端耦合分别是什么。
