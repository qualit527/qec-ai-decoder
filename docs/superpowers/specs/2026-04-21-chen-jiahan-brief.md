# 陈嘉汉任务说明书

你对应的模型是 `Claude Code`。

你的角色可以简单理解成：**把 AutoQEC 的实验主流程和 `surface_d5` 这条基础 QEC 链路搭起来，让系统先真正跑通。**

你不是只负责“写提示词”或“做文档”。你的核心价值在于：
- 把 `surface_d5_depol` 这个参考环境准备好。
- 把多代理编排流程和 Runner 串起来。
- 确保 Demo 1 能稳定地展示一条完整的 QEC 自动研究链路。

## 1. 这三天你具体要做什么

### Day 1: 先把 `surface_d5` 场景跑通

你当天最重要的目标，不是做很多功能，而是把最基础的 QEC 实验环境搭出来。

你要做的事：
- 确认 `surface_d5_depol` 的环境配置，包括 code、noise、constraints、baseline decoder。
- 用 `stim.Circuit.generated(...)` 把 `surface_d5` 电路生成出来。
- 把 syndrome 数据流梳理清楚，确认从电路到检测事件、再到 decoder 输入的路径是通的。
- 接上 `PyMatching` 基线，拿到 bare classical baseline 的结果。
- 跑一次 1M-shot benchmark，估一个后面大家都能参考的 wall-clock 数字。
- 起草多代理编排侧需要的输入输出契约，尤其是：
  - orchestrator 给 subagent 什么输入
  - Runner 返回什么结构
  - `metrics.json` 至少有哪些字段

Day 1 的最低交付物：
- 一个可工作的 `surface_d5_depol` 环境。
- 一条可以复现实验的 PyMatching baseline 路径。
- 一份简洁的 orchestrator / Runner / metrics 接口约定。

### Day 2: 把主流程串起来

这一天你最重要的事，是让系统完成“一轮完整闭环”。

你要做的事：
- 把 orchestrator 和 Ideator / Coder / Analyst 三个子代理的调用关系接起来。
- 让 orchestrator 能读环境、发起一轮实验、接收 Runner 结果、再把结果交给 Analyst。
- 在 `surface_d5` 上跑通至少一轮 dev-profile 的 end-to-end 流程。
- 重点处理“流程没接上”的问题，比如：
  - env spec 字段对不上
  - Runner 返回格式和 orchestrator 预期不一致
  - round 目录落盘结构和日志结构不一致

Day 2 的最低交付物：
- 一轮完整的 hypothesis -> config -> train/eval -> report 流程。
- 一份能让其他两位继续开发而不反复猜接口的编排边界说明。

### Day 3: 交付 Demo 1，并把入口做得可用

这一天的重点不是继续扩功能，而是把最核心的演示交付好。

你要做的事：
- 在 `surface_d5` 上跑 Demo 1 的完整版本。
- 整理 `/autoqec-run` 和 `/add-env` 两个用户入口的包装逻辑。
- 产出 Demo 1 的 walkthrough，说明用户怎么触发、系统怎么工作、最终结果怎么看。
- 记录 lab notebook 风格的过程说明，让最终展示不是“只有结果”，而是“有过程、有判断、有证据”。

Day 3 的最低交付物：
- Demo 1 可演示。
- `/autoqec-run` 可作为主入口使用。
- 至少有一份人人看得懂的运行说明。

## 2. 你和另外两个人怎么配合

- 你要和林腾祥对齐 Runner 接口，因为你这边的 orchestrator 最终要调用他负责的训练与评估主流程。
- 你要和谢金谷对齐 `metrics.json` 和 verify 所需字段，因为后续验证模块会直接吃你这里落盘的数据。
- 你是主流程负责人，所以接口不清时，应该由你推动“先冻结最小版本”，而不是让三个人各自猜。

最需要你盯住的协作点：
- `env_spec` 的字段命名
- `round_<N>/` 目录结构
- `metrics.json` 的标准字段
- demo 脚本的调用入口

## 3. 你会碰到的 QEC 概念解释

下面这些概念，不需要你一开始就学到论文级别，但你必须知道它们在工程里分别意味着什么。

### QEC

`QEC` 是 quantum error correction，也就是量子纠错。

它要解决的问题很直接：量子比特很脆弱，会出错，所以我们不能只看一次测量结果就相信它，而是要设计一套编码和解码流程，把物理错误尽量变成可检测、可纠正的东西。

在这个项目里，QEC 不是抽象背景，而是你整个实验环境存在的原因。

### Decoder

`decoder` 是解码器。

它的工作是：根据 syndrome 或相关观测信息，猜测系统里最可能发生了什么错误，然后给出一个 correction。

你负责的是让系统能稳定跑 baseline decoder 和后续的 AI-assisted decoder，所以 decoder 是你每天都会碰到的核心对象。

### Surface code

`surface code` 是最经典的一类量子纠错码之一。

你可以先把它理解成：一种结构规整、社区里非常常用、很适合拿来做 baseline 的 QEC code family。这个项目里 `surface_d5` 是最重要的第一个参考环境。

### `d=5`

这里的 `d` 是 code distance。

直观上，它表示这个码抵抗错误的“厚度”或者“安全边际”。数字越大，通常越强，但也越贵。`surface_d5` 就是 distance 为 5 的 surface code。

### Depolarizing noise

`depolarizing noise` 是一种常见噪声模型。

简单理解，就是量子比特可能以某个概率发生随机错误。它是一个标准、常见、容易做基准对比的噪声模型，所以很适合做 MVP 环境。

### Circuit-level noise

`circuit-level noise` 表示噪声不是只加在某一个静态比特上，而是发生在整个量子电路执行过程中。

这比简单的 toy noise 更真实，也意味着你得到的数据流会更接近真正的 syndrome extraction 场景。

### Syndrome

`syndrome` 可以理解成“错误留下来的间接痕迹”。

它不是直接告诉你“哪一个物理比特错了”，而是告诉你“哪里看起来不对劲”。decoder 就是要根据这些痕迹去反推可能的错误。

你负责把 syndrome 数据流接出来，所以这个概念是你必须真正理解的。

### Stim

`Stim` 是一个常用的量子纠错模拟工具。

在这个项目里，它主要负责帮助你生成电路、模拟噪声、产生检测事件或相关样本。对你来说，它更像“实验数据来源”和“标准化电路工具”。

### Sinter

`sinter` 可以理解成围绕 Stim 的批量采样和实验统计工具。

它的作用是让你更方便地做很多 shot 的实验、收集统计量、评估逻辑错误率。你跑 benchmark 或 baseline 的时候会用到它。

### DEM

`DEM` 是 detector error model。

你可以把它理解成“Stim 帮你整理出来的一份错误传播说明书”，它描述哪些错误会导致哪些检测事件。很多经典 decoder 会直接或间接依赖这类结构化信息。

### PyMatching

`PyMatching` 是一个常用的经典 decoder 实现库。

在你这条线里，它承担的是 baseline 角色。也就是说，先不用 AI，只用一个公认可靠的经典解码器，先看看系统本来能做到什么水平。

### MWPM

`MWPM` 是 minimum-weight perfect matching。

不用记算法细节，先记住一件事：`PyMatching` 背后的代表性思路就是 matching-based decoding。你的任务里，MWPM 是 surface code 这条线的固定 classical backend。

### Baseline

`baseline` 是基线。

意思是：先拿一个大家都认可的标准方法做参照，后面你所有的改进都要和它比。没有 baseline，后面的“改进”就没有意义。

### LER

`LER` 是 logical error rate，也就是逻辑错误率。

它不是物理层上某个比特错没错，而是最终这个纠错系统有没有把逻辑信息保护住。它是你们最关心的核心效果指标之一。

### `Δ_LER`

`Δ_LER` 表示引入 predecoder 之后，相比纯 classical baseline，逻辑错误率改善了多少。

它是这个项目判断“AI predecoder 有没有价值”的最直接指标。

### Shot

一个 `shot` 可以粗略理解成“一次完整采样或实验尝试”。

你做 1M-shot benchmark，本质上就是做很多次重复实验，看看统计结果稳不稳。

## 4. 你这几天最容易踩的坑

- 只顾着搭 agent 流程，但 `surface_d5` baseline 还没真正跑通。
- 接口没有冻结，导致 Runner、verify、orchestrator 三边字段不一致。
- Demo 前才发现日志结构或目录结构不能支持后处理。
- 把自己做成“流程协调员”，却没有亲手跑过一次真正的 `surface_d5` baseline。

## 5. 你的完成标准

如果到第 3 天结束，你能做到下面这些，就算你这条线完成得不错：
- `surface_d5` baseline 能稳定复现。
- orchestrator 能完整地驱动至少一轮 end-to-end 流程。
- Demo 1 可以向别人解释清楚“输入是什么、系统做了什么、结果怎么看”。
- 其他两个人不需要反复来问你“字段名到底是什么、目录该怎么放、主入口怎么调用”。
