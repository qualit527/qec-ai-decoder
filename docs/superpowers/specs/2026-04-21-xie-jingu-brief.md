# 谢金谷任务说明书

你对应的模型是 `GLM`。

你的角色可以简单理解成：**证明这套系统的结果是可信的，同时把 `bb72` / qLDPC 这条第二条 QEC 赛道搭起来。**

你不是只负责“看结果”。你的核心价值在于：
- 建立独立验证机制，防止系统自我欺骗。
- 给出 `VERIFIED / SUSPICIOUS / FAILED` 这套判断依据。
- 让项目不只停留在 `surface_d5` 一个例子上，而是能往 `bb72` / qLDPC 推进。

## 1. 这三天你具体要做什么

### Day 1: 先把验证对象和第二条赛道摸清楚

你要先搞清楚两件事：你要验证什么，以及 `bb72` 这条线怎么进入系统。

你要做的事：
- 调研并确认 `bb72_depol` 环境需要的输入来源和可用工具链。
- 尝试把 BP+OSD baseline 接起来。
- 如果 Relay-BP 能接上，也一起梳理接口；如果短期不行，要尽早确认 fallback。
- 设计 `independent_eval.py` 需要读取的输入字段和输出结构。
- 明确 holdout seed 的使用规则，避免训练和验证数据混掉。
- 设计 reward-hacking 检查用的测试样例，比如“背训练样本”的作弊 predecoder。

Day 1 的最低交付物：
- 一份 `bb72` / qLDPC 方向的最小落地路径。
- 一份验证模块的数据输入输出契约。
- 一份“什么叫作弊、怎么抓作弊”的样例设计。

### Day 2: 把独立验证模块做出来

这一天你最关键的目标，是让“验证”不只是口头说法，而是一段能独立执行的流程。

你要做的事：
- 实现 `independent_eval.py`。
- 接上 `autoqec verify` 这一层调用。
- 加入三项 MVP 级验证：
  - holdout seed isolation
  - bootstrap 95% CI
  - ablation sanity check
- 明确 `VERIFIED / SUSPICIOUS / FAILED` 的判断逻辑。
- 如果时间允许，把 Pareto admission 所需的验证结果也串起来。

Day 2 的最低交付物：
- 一个可以独立运行的 verify 流程。
- 一份别人能读懂的验证报告结构。
- 至少一个能够被正确识别的异常样例。

### Day 3: 交付 Demo 4，并把诊断能力包装出来

这一天的重点是把“系统不会自己骗自己”这件事讲清楚。

你要做的事：
- 跑 Demo 4，也就是 cheating-predecoder 检测。
- 完成 `/verify-decoder` 的包装。
- 整理 `/review-log` 和 `/diagnose-failure` 所需要的诊断逻辑或提示框架。
- 为最终展示准备结果解读文本，告诉别人为什么某个 candidate 是可信的，为什么另一个是可疑的。
- 如果 `bb72` 有进展，补充其 baseline 对比说明。

Day 3 的最低交付物：
- Demo 4 可演示。
- `/verify-decoder` 可用。
- 最终结果里有一部分是你来负责“可信度解释”的。

## 2. 你和另外两个人怎么配合

- 你要和陈嘉汉对齐 `metrics.json`、日志、目录落盘格式，因为你的验证模块要直接读这些东西。
- 你要和林腾祥对齐 checkpoint、模型输出、训练配置格式，因为 ablation 和 holdout verify 会用到这些内容。
- 你不应该等所有实验做完再介入。验证逻辑要尽早进系统，否则后面很容易出现“有结果，但证据链不完整”。

最需要你盯住的协作点：
- holdout seeds 是否真的隔离
- verify 是否物理上与 Runner 解耦
- baseline 对比是否公平
- 最终 verdict 是否有证据支持

## 3. 你会碰到的 QEC 概念解释

下面这些概念，是你这条线最需要真的理解的。你不一定要会推公式，但必须知道它们在“结果是否可信”这件事里分别扮演什么角色。

### QEC

`QEC` 是 quantum error correction。

它的核心不是“模型分数高”，而是“在有噪声的量子系统里，最终逻辑信息有没有被保护住”。所以你做验证时，不能只看训练损失，要看真正的解码效果和泛化表现。

### qLDPC

`qLDPC` 是 quantum low-density parity-check code。

可以先把它理解成另一类量子纠错码，结构和 surface code 不一样，解码方式和 baseline 也常常不同。`bb72` 就属于这条线的代表环境。

### `bb72`

这里的 `bb72` 指的是一个具体的 qLDPC 参考码环境。

你负责的是让项目不只在 `surface_d5` 这个较经典场景下有故事，也能在 `bb72` 这类场景下展示“这个系统是通用的”。

### Phenomenological noise

`phenomenological noise` 是一种相对抽象一些的噪声模型。

相比更细的 circuit-level 描述，它更强调 syndrome 或测量层面的错误行为。你在 `bb72` 这条线可能会碰到它。

### BP

`BP` 是 belief propagation。

你可以把它理解成一种基于图结构传递“相信谁更像错误”的经典推断方法。它在 LDPC / qLDPC 相关解码里很常见。

### OSD

`OSD` 是 ordered statistics decoding。

它经常和 BP 搭配使用。直观上，你可以把 BP 理解为先给出一些“哪些位置更可疑”的软信息，而 OSD 再利用这些信息做更强的后处理。

### BP+OSD

`BP+OSD` 就是先跑 BP，再跑 OSD。

在 `bb72` 这条线里，它是非常关键的 baseline。你后续所有“AI predecoder 带来改进”的说法，都要先和它比。

### Relay-BP

`Relay-BP` 是更强的一类 BP 变体或相关 baseline。

你不用先深究论文细节，但要知道它为什么重要：如果社区里更强的 baseline 已经存在，而你没有对齐它，你的结果说服力就会变弱。

### Holdout seed

`holdout seed` 是保留给最终验证的一组随机种子。

它的意义是：训练、调参、开发过程中都不能碰这批数据，否则最后的“验证集表现”就不再可信。

### Seed isolation

`seed isolation` 就是随机种子的严格隔离。

你负责守住这条线。如果训练、验证、分析共享了不该共享的种子，那么模型可能只是“见过题目”，不是“真的会做”。

### Bootstrap confidence interval

`bootstrap CI` 是一种统计置信区间估计方法。

简单说，就是你不是只报一个单点数字，而是给出“这个结果大概落在什么范围内”。这样别人才能判断改进是不是统计波动。

### Ablation sanity check

`ablation sanity check` 在这里的作用是判断模型到底学到了东西，还是只是流程偶然凑巧。

比如把 predecoder 权重打乱后，如果效果还和原来差不多，那就说明这个模型很可能没有真正提供有用信息。

### Reward hacking

`reward hacking` 可以理解成系统找到了一个“看起来指标不错，但本质上是在钻空子”的办法。

在自动研究系统里，这个问题很危险。因为 agent 可能不是故意作弊，但它会倾向于朝着容易提升表面指标的方向走。

### `VERIFIED / SUSPICIOUS / FAILED`

这是你负责定义和解释的三种验证结论。

可以这样理解：
- `VERIFIED`: 证据链完整，结果可信。
- `SUSPICIOUS`: 有提升，但证据不足或存在异常，需要人工复核。
- `FAILED`: 明确不可信，或者违反了验证规则。

### Pareto front

`Pareto front` 表示一组“互相之间没有谁在所有指标上都更好”的候选点。

你的工作不是只看一个点好不好，还要判断一个 candidate 有没有资格进最终的 Pareto 集合。

## 4. 你这几天最容易踩的坑

- 等实验都跑完了才开始想验证标准，结果前面的数据结构根本不够用。
- 验证逻辑和训练逻辑混在一起，导致“独立验证”名义上存在，实际上不独立。
- 只看最优数字，不看置信区间和 baseline fairness。
- 口头上说“可疑”，但没有清晰的证据链和判据。

## 5. 你的完成标准

如果到第 3 天结束，你能做到下面这些，就算你这条线完成得不错：
- `autoqec verify` 能独立跑起来。
- 至少一个异常或作弊 case 能被正确识别。
- 对 `bb72` / qLDPC 这条线有清晰的 baseline 和推进路径。
- 最终展示时，别人问“你怎么证明这不是巧合”，你能拿出结构化证据回答。
