# Lin Tengxiang任务说明书

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

## 2. 你实际怎么和 Agent 配合推进

你的工作很适合用 Agent 提速，但前提是你不能把“关键设计决策”一股脑外包出去。

正确方式是：
- 先让 Agent 实现一小段清晰边界的功能。
- 你人工确认输入输出、接口和运行结果。
- 再让 Agent 继续下一段。

你要把自己理解成“模型主设计者 + 代码审核者”，不是“让 Agent 自己发明一套 decoder”。

### 阶段 A: Day 1 先搭 DSL 和 compiler

第一轮先发给 Agent 的 prompt：

```text
请基于 docs/specs/2026-04-20-autoqec-design.md 和当前仓库，实现 AutoQEC predecoder 的最小 DSL + compiler。

目标：
1. 支持 gnn 和 neural_bp 两个 family 的最小可用配置
2. 明确 hard_flip 和 soft_priors 的输出契约
3. 先保证可编译、可实例化，不要先追求最全功能
4. 改完后告诉我：
   - 改了哪些文件
   - 目前支持哪些字段
   - 哪些地方需要我人工拍板
```

Agent 做完后，你人工要做的事：
- 检查 DSL 字段名是否和设计稿一致。
- 检查 compiler 生成的模块是不是你真的愿意让后续 agent 搜索的结构。
- 对它列出来的“需要人工拍板”项做决定，比如输出形式、最小字段集合。

第二轮再发给 Agent 的 prompt：

```text
继续在刚才的实现上，补 3 个 GNN seed template 和 3 个 Neural-BP seed template，并整理一份最小输入输出契约。

要求：
1. 说明模型输入张量含义
2. 说明 hard_flip / soft_priors 分别输出什么
3. 说明这些输出之后怎么接到 classical backend
4. 如果还接不动 backend，也要先把接口定义清楚
```

这轮做完后，你人工要做的事：
- 把输入输出契约同步给Chen Jiahan，确保 orchestrator 调 Runner 时不会猜字段。
- 把 checkpoint / config / metrics 需求同步给Xie Jingu，确保后面 verify 能复现。

### 阶段 B: Day 2 打通 Runner 和 classical backend

下一轮发给 Agent 的 prompt：

```text
请在现有 DSL + compiler 基础上，实现 Runner 的最小 train/eval 流程，并打通 predecoder 到 classical backend 的接口。

必须包含：
1. train / eval 主流程
2. params 和 FLOPs 统计
3. checkpoint.pt、metrics.json、config.yaml 的标准输出
4. RunnerSafety 的最小保护
5. 优先接通 MWPM；如果 OSD 暂时不完整，也请把接口占位清楚
```

Agent 做完后，你人工要做的事：
- 亲自检查 Runner 产物是否完整，而不是只看它说“成功了”。
- 看 `hard_flip` 和 `soft_priors` 有没有被清楚地区分。
- 和Chen Jiahan对齐 Runner 调用接口，和Xie Jingu对齐 verify 所需输出。

如果第一轮 end-to-end 出现错误，再发下一轮 prompt：

```text
下面是当前 end-to-end 暴露的问题，请只做收敛式修复，不做大重构：

[把错误日志、报错栈、字段不一致情况贴进来]

修复优先级：
1. 保证 demo 可跑
2. 保证输出文件齐全
3. 保证 backend 接口清楚
4. 如果有多种修法，优先最保守方案
```

### 阶段 C: Day 3 支撑 Demo 1 / Demo 2

发给 Agent 的 prompt：

```text
请把当前 predecoder 路径整理成最适合 Demo 1 / Demo 2 使用的版本。

输出要求：
1. 推荐的模型配置
2. 推荐的运行命令
3. 如果训练不稳定，fallback config 是什么
4. 我人工演示时应该怎么解释 hard_flip / soft_priors 和 classical backend 的关系
5. 给我一段 1 分钟口头说明
```

Agent 做完后，你人工要做的事：
- 真正跑一遍它推荐的配置，而不是直接拿去演示。
- 选出一个“最稳版本”作为保底方案。
- 把 1 分钟口头说明改成你自己的表达，确保你能解释模型到底做了什么。

如果 Day 3 时间很紧，再发最后一轮 prompt：

```text
请把 predecoder 演示路径压缩成最稳妥版本。

要求：
1. 优先保成功率，不追求最复杂模型
2. 给我一个最小可讲清楚的 config
3. 列出 demo 现场最容易失败的两个点和对应 fallback
4. 不新增大功能
```

## 3. 你和另外两个人怎么配合

- 你要和Chen Jiahan对齐 orchestrator 调用 Runner 的接口，因为你的代码最终要被主流程驱动。
- 你要和Xie Jingu对齐 checkpoint、metrics、config 落盘格式，因为验证模块要读这些输出。
- 你是模型主实现负责人，所以当“模型为什么接不上 classical backend”时，这个问题主要是你来收口。

最需要你盯住的协作点：
- DSL 字段和实际模型实现是否一一对应
- predecoder 输出和 backend 输入是否定义清楚
- 训练产物是否足够让 verify 复现
- profile 切换时，dev / prod 行为是否一致

## 4. 你会碰到的 QEC 概念解释

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

## 5. 你这几天最容易踩的坑

- DSL 写得很漂亮，但 compiler 和实际模型实现对不上。
- 模型能训练，但输出定义不清，接不上 MWPM / OSD。
- 只顾着做更复杂的模型，没有优先保证第一条可跑通路径。
- Runner 没有把 checkpoint、metrics、FLOPs 这些基础产物存好，导致后面无法验证和复盘。

## 6. 你的完成标准

如果到第 3 天结束，你能做到下面这些，就算你这条线完成得不错：
- 别人能根据 DSL 配出并实例化一个 predecoder。
- 至少有一条 predecoder + classical backend 的完整训练评估链路。
- Demo 1 的模型路径稳定可用。
- 当别人问“AI 部分到底做了什么”时，你能明确回答输入、输出、训练、评估、后端耦合分别是什么。
