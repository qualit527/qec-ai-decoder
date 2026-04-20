# AutoQEC — AI-Driven Quantum Error Correction Decoder

## System Design Document

**Version:** v0.1.0
**Date:** 2026-04-20
**Status:** Draft

---

## 1. Overview

AutoQEC 是一个面向 QEC 研究者的自动化解码引擎。系统以 Surface Code 的噪声参数为输入，利用 Stim 生成带噪 syndrome 数据，训练 MLP 解码器，并与 MWPM（PyMatching）、BP-OSD（ldpc）baseline 进行对比评估，最终输出包含逻辑错误率曲线、训练曲线、解码延迟、错误分析的 HTML 可视化报告。

**核心 Pipeline：**
```
Stim 数据生成 → 数据预处理 → MLP 训练 → Baseline 对比 → HTML 报告输出
```

---

## 2. Motivation

- 传统解码器（MWPM、BP-OSD）对复杂噪声模型的适应能力有限
- 近期研究表明 AI 解码器可实现最多 17x 的逻辑错误率降低（Harvard, 2026）
- Google AlphaQubit、NVIDIA/QuEra 等均在探索 AI 解码路线
- Surface Code 是当前容错量子计算的主流码字，工具链成熟（Stim、PyMatching）
- 本项目作为 MVP 验证 AI 解码 pipeline 的可行性，后续扩展至 qLDPC 码

**关键参考文献：**
- Bravyi et al., "High-threshold and low-overhead fault-tolerant quantum memory" (arXiv:2308.07915)
- Google DeepMind, "AlphaQubit" (2024)
- Harvard, "AI Decoder Could Cut Quantum Errors by Up to 17x" (2026)
- Blue et al., ML decoding on [[72,12,6]] BB code, ~5x improvement (2025)
- Craig Gidney, Stim (https://github.com/quantumlib/Stim)
- Oscar Higgott, PyMatching (https://github.com/oscarhiggott/PyMatching)
- quantumgizmos, ldpc (https://github.com/quantumgizmos/ldpc)

---

## 3. Success Metrics

| 指标 | 目标 |
|------|------|
| 逻辑错误率曲线 | MLP 解码器在部分物理错误率区间优于或持平 MWPM |
| 训练收敛 | Loss 在 100 epochs 内收敛，验证准确率 > 99% |
| 解码延迟 | MLP 单次推理 < 1ms（CPU） |
| 对比完整性 | MWPM + BP-OSD 双 baseline 对比 |
| 可复现性 | 一条命令跑完全流程，输出 HTML 报告 |

---

## 4. Requirements & Constraints

### 4.1 功能需求

- 支持 Surface Code（Rotated Memory Z），距离 d=3 和 d=5（d=7 可选）
- 支持 Bit-flip 噪声和 Depolarizing 噪声
- 自定义噪声模型留接口占位
- 多错误率混合训练（p ∈ [0.001, 0.02]）
- 可配置的 MLP 解码器（层数、宽度、激活函数、dropout 等）
- 可配置的输出预测模式：
  - 逻辑 observable 翻转预测（默认）
  - 逐 qubit Pauli 错误预测
  - 完整错误模式预测
- MWPM + BP-OSD 双 baseline 对比
- 独立测试集评估
- 生成 HTML 可视化报告

### 4.2 非功能需求

- CPU 可运行（不强制要求 GPU）
- Python 3.9+
- 主要依赖：Stim、PyMatching、ldpc、PyTorch、numpy
- 一周交付周期

### 4.3 Scope

**In-scope（MVP）：**
- Surface Code (d=3, d=5) 的 MLP 解码
- Bit-flip + Depolarizing 噪声
- MWPM + BP-OSD baseline
- CLI 工具 + HTML 报告
- 文件系统存储

**Out-of-scope（后续迭代）：**
- NAS 架构搜索
- Neural-BP 混合解码
- qLDPC 码（Bivariate Bicycle 等）
- 真实芯片噪声数据
- 前后端分离的 Web 应用
- FPGA 部署

---

## 5. Methodology

### 5.1 问题建模

将 QEC 解码建模为**分类问题**：

- **输入**：d 轮 syndrome 测量结果（二值向量）
- **输出**（默认模式）：逻辑 observable 是否翻转（k 个二值标签）

输入维度 = `syndrome_bits × d_rounds`，例如 d=3 时为 `24 × 3 = 72`

### 5.2 数据管线

#### 5.2.1 数据生成（Stim）

使用 Stim 的内置 Surface Code circuit 生成器：

```python
import stim

circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z",
    rounds=d,          # d 轮测量
    distance=d,        # 码距
    after_clifford_depolarization=p,  # Clifford 门后去极化噪声
    after_reset_flip_probability=p,   # Reset 翻转概率
    before_measure_flip_probability=p # 测量前翻转概率
)
```

从 circuit 中采样：
- `detection_events`：syndrome 测量结果（每轮的翻转指示器）
- `observable_flips`：逻辑 observable 是否翻转（标签）

#### 5.2.2 数据格式

每个样本存储为：
```python
{
    "syndrome": np.ndarray,    # shape: (syndrome_bits * rounds,), dtype: float32
    "label": np.ndarray,       # shape: (k,) 或 (2n,) 或 (4n,), 取决于输出模式
}
```

#### 5.2.3 训练/测试分割

- **训练集**：从混合 p ∈ [0.001, 0.002, 0.005, 0.008, 0.01, 0.012, 0.015, 0.02] 采样，每个 p 生成 10,000 样本，共 80,000 样本
- **测试集**：独立生成，每个 p 生成 50,000 样本

### 5.3 模型架构

#### 5.3.1 MLP 解码器（默认配置）

```python
# 默认架构（在 yaml 中可配置）
MLPDecoder(
    input_dim = syndrome_bits * rounds,  # d=3 → 72, d=5 → 400
    hidden_dims = [256, 128, 64],         # 可配置
    output_dim = k,                        # 取决于输出模式
    activation = 'relu',                   # 可配置: relu, tanh, gelu
    dropout = 0.0,                         # 可配置
    use_batch_norm = False,                # 可配置
)
```

损失函数：
- 逻辑 observable 翻转预测 → BCE (Binary Cross Entropy)
- 逐 qubit Pauli 预测 → BCE (多标签)
- 完整错误模式预测 → CrossEntropy (4类 per qubit)

#### 5.3.2 配置文件格式

```yaml
# configs/default.yaml
code:
  type: "surface_code:rotated_memory_z"
  distances: [3, 5]          # d=7 可选

noise:
  type: "depolarizing"        # "bitflip" | "depolarizing" | "custom"
  error_rates: [0.001, 0.002, 0.005, 0.008, 0.01, 0.012, 0.015, 0.02]

data:
  train_samples_per_p: 10000
  test_samples_per_p: 50000
  rounds: "d"                 # "d" 表示与码距相同

model:
  type: "mlp"
  hidden_dims: [256, 128, 64]
  activation: "relu"
  dropout: 0.0
  batch_norm: false
  output_mode: "observable_flip"  # "observable_flip" | "pauli_per_qubit" | "full_error"

training:
  optimizer: "adam"
  learning_rate: 0.001
  batch_size: 256
  epochs: 100
  train_split: 0.8
  seed: 42

evaluation:
  baselines: ["mwpm", "bp_osd"]
  compare_on_same_testset: true
  run_pymatching_builtin_benchmark: true

output:
  dir: "outputs/"
  save_model: true
  generate_html_report: true
```

### 5.4 Baseline 解码器

#### 5.4.1 MWPM（PyMatching）

```python
import pymatching

# 从 Stim circuit 构建 Matching 对象
matching = pymatching.Matching.from_detector_error_model(
    circuit.detector_error_model(decompose_errors=True)
)

# 解码
prediction = matching.decode(syndrome_vector)
```

**注意：**
- PyMatching 输入是 detection events（syndrome 差分），不是原始 syndrome
- 需要从 Stim 采样结果中正确提取 detection events
- 内置 benchmark 可通过 `stim.Circuit.compile_detector_sampler()` + 统计实现

#### 5.4.2 BP-OSD（ldpc 库）

```python
from ldpc import BpOsdDecoder

# 从校验矩阵构建解码器
# H 为 Surface Code 的 X 或 Z 校验矩阵
decoder = BpOsdDecoder(
    parity_check_matrix=H,
    error_rate=p,
    max_iter=50,        # BP 迭代次数
    bp_method="minimum_sum",
    osd_method="osd_0"  # OSD 后处理方法
)

# 解码
prediction = decoder.decode(syndrome)
```

**数据格式转换：**
- ldpc 库需要校验矩阵 H（numpy sparse matrix）
- Syndrome 需要从 Stim 的 detection events 格式转换为标准二值向量
- 需要分别对 X 和 Z stabilizer 构建校验矩阵（CSS 码结构）

**安装：**
```bash
pip install ldpc
```

**关键接口文档：**
- `BpOsdDecoder(H, error_rate, max_iter, bp_method, osd_method)` — 初始化
- `decoder.decode(syndrome)` — 输入 syndrome，输出错误预测
- `decoder.log_prob_ratio` — BP 的对数似然比（可用于分析）

### 5.5 评估协议

1. 对每个物理错误率 p，独立生成 50,000 个测试样本
2. 分别用 MLP、MWPM、BP-OSD 解码
3. 对比解码结果与真实 observable flip，统计逻辑错误率
4. 同时运行 PyMatching 内置 benchmark 作为参考

**评估指标：**
- 逻辑错误率 vs 物理错误率（主图）
- 训练 loss 曲线
- 验证准确率曲线
- 解码延迟（每个 syndrome 的平均推理时间）
- 混淆矩阵（p=1% 处）
- 错误类型分布饼图

---

## 6. System Architecture

### 6.1 目录结构

```
qec-ai-decoder/
├── autoqec/                     # 核心包
│   ├── __init__.py
│   ├── codes/                   # 码字定义
│   │   ├── __init__.py
│   │   ├── base.py              # Code 基类接口
│   │   └── surface_code.py      # Surface Code 封装
│   ├── data/                    # 数据生成模块
│   │   ├── __init__.py
│   │   ├── generator.py         # Stim 数据生成
│   │   └── dataset.py           # PyTorch Dataset 封装
│   ├── models/                  # 模型定义
│   │   ├── __init__.py
│   │   ├── mlp.py               # MLP 解码器
│   │   └── registry.py          # 模型注册（后续扩展 GNN 等）
│   ├── baselines/               # Baseline 解码器
│   │   ├── __init__.py
│   │   ├── mwpm.py              # PyMatching 封装
│   │   └── bp_osd.py            # ldpc BP-OSD 封装
│   ├── train/                   # 训练逻辑
│   │   ├── __init__.py
│   │   └── trainer.py           # 训练循环
│   ├── eval/                    # 评估模块
│   │   ├── __init__.py
│   │   ├── evaluator.py         # 统一评估入口
│   │   └── metrics.py           # 指标计算（逻辑错误率、延迟等）
│   └── viz/                     # 可视化
│       ├── __init__.py
│       └── report.py            # HTML 报告生成（ECharts）
├── configs/                     # 实验配置
│   └── default.yaml             # 默认配置
├── outputs/                     # 输出目录
│   ├── models/                  # 保存的模型权重
│   ├── results/                 # 评估结果（JSON/CSV）
│   └── reports/                 # HTML 报告
├── main.py                      # CLI 入口
├── design.md                    # 本文件
├── requirements.txt             # 依赖
└── README.md
```

### 6.2 模块接口定义

#### `autoqec/codes/base.py`

```python
class Code(ABC):
    @abstractmethod
    def get_hx(self) -> scipy.sparse.spmatrix: ...

    @abstractmethod
    def get_hz(self) -> scipy.sparse.spmatrix: ...

    @property
    @abstractmethod
    def n(self) -> int: ...      # 物理比特数

    @property
    @abstractmethod
    def k(self) -> int: ...      # 逻辑比特数

    @property
    @abstractmethod
    def d(self) -> int: ...      # 码距

    def get_stim_circuit(self, rounds: int, noise_model: dict) -> stim.Circuit: ...
```

#### `autoqec/codes/surface_code.py`

```python
class SurfaceCode(Code):
    def __init__(self, distance: int):
        ...

    # 从 Stim 生成的 circuit 中提取 Hx, Hz
```

#### `autoqec/data/generator.py`

```python
def generate_data(
    circuit: stim.Circuit,
    n_samples: int,
    output_mode: str = "observable_flip"  # "observable_flip" | "pauli_per_qubit" | "full_error"
) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (syndromes, labels)"""
```

#### `autoqec/data/dataset.py`

```python
class QCEDataset(torch.utils.data.Dataset):
    def __init__(self, syndromes: np.ndarray, labels: np.ndarray): ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]: ...
```

#### `autoqec/models/mlp.py`

```python
class MLPDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
    ): ...

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

#### `autoqec/baselines/mwpm.py`

```python
class MWPMDecoder:
    def __init__(self, circuit: stim.Circuit): ...
    def decode(self, syndrome: np.ndarray) -> np.ndarray: ...
    def benchmark(self, n_samples: int, error_rate: float) -> dict: ...
```

#### `autoqec/baselines/bp_osd.py`

```python
class BPOSDDecoder:
    def __init__(self, hx: spmatrix, hz: spmatrix, error_rate: float): ...
    def decode(self, syndrome: np.ndarray) -> np.ndarray: ...
```

#### `autoqec/train/trainer.py`

```python
class Trainer:
    def __init__(self, model: nn.Module, config: dict): ...
    def train(self, train_dataset: QCEDataset, val_dataset: QCEDataset) -> dict:
        """返回 {"loss_history": [...], "acc_history": [...]}"""
```

#### `autoqec/eval/evaluator.py`

```python
class Evaluator:
    def __init__(self, code: Code, config: dict): ...
    def evaluate(self, model: nn.Module, test_data: dict) -> dict:
        """返回 {
            "logical_error_rate": {p: rate, ...},
            "latency_ms": {method: ms, ...},
            "confusion_matrix": np.ndarray,
            "error_breakdown": dict
        }"""
```

#### `autoqec/viz/report.py`

```python
def generate_report(
    results: dict,
    config: dict,
    output_path: str = "outputs/reports/report.html"
) -> str:
    """生成 HTML 报告，返回文件路径"""
```

### 6.3 数据流

```
                      configs/default.yaml
                              |
                              v
                    ┌─────────────────┐
                    │    main.py      │
                    │  (CLI 入口)     │
                    └────────┬────────┘
                             |
              ┌──────────────┼──────────────┐
              v              v              v
     ┌────────────┐  ┌────────────┐  ┌────────────┐
     │ SurfaceCode│  │   Stim     │  │  配置参数  │
     │  (d, Hx,Hz)│  │  Circuit   │  │            │
     └─────┬──────┘  └─────┬──────┘  └────────────┘
           |               |
           v               v
     ┌──────────────────────────┐
     │     DataGenerator        │
     │  generate_data()         │
     │  → (syndromes, labels)   │
     └────────────┬─────────────┘
                  |
           ┌──────┴──────┐
           v             v
     ┌──────────┐  ┌──────────┐
     │Train Split│  │Test Split │
     │  (80%)    │  │ (独立生成) │
     └─────┬────┘  └─────┬────┘
           |              |
           v              |
     ┌──────────┐         |
     │  Trainer │         |
     │ MLP 训练 │         |
     │ 100 ep   │         |
     └─────┬────┘         |
           |              |
           v              v
     ┌──────────────────────────┐
     │      Evaluator           │
     │  ┌──────┐ ┌──────┐      │
     │  │ ML   │ │ MWPM │      │
     │  │Model │ │      │      │
     │  └──┬───┘ └──┬───┘      │
     │     └───┬────┘          │
     │   ┌─────┴─────┐         │
     │   │  BP-OSD   │         │
     │   └─────┬─────┘         │
     │         |               │
     │  统一测试集对比          │
     └────────┬────────────────┘
              |
              v
     ┌──────────────────────────┐
     │   generate_report()      │
     │   → HTML (ECharts)       │
     │                          │
     │  [Overview] [Logical Err]│
     │  [Training] [Latency]    │
     │  [Error Analysis]        │
     └──────────────────────────┘
```

---

## 7. Implementation Details

### 7.1 依赖库

```
stim>=1.13.0
pymatching>=2.2.0
ldpc>=0.1.6
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
pyyaml>=6.0
tqdm>=4.65.0
```

### 7.2 关键实现注意事项

1. **Stim detection events vs raw syndrome**：Stim 输出的 detection events 是相邻轮的差分，需要正确处理。训练数据可以直接用 detection events 作为输入。

2. **BP-OSD 数据格式转换**：ldpc 库需要 sparse 格式的校验矩阵。从 Stim circuit 提取 Hx、Hz 需要解析 circuit 中的 stabilizer 信息。

3. **多错误率混合训练**：训练时从不同 p 的数据中均匀采样，避免低 p 样本占比过大。

4. **HTML 报告**：使用 ECharts（CDN 加载），所有数据内嵌到 HTML 中，无需额外服务器。报告支持 Tab 切换、鼠标悬停、缩放交互。

5. **随机种子**：全局设置 `torch.manual_seed(42)` 和 `np.random.seed(42)`，保证可复现。

---

## 8. Experiment Plan

### 8.1 实验矩阵

| 实验 | 码距 | 噪声模型 | 错误率范围 | 训练样本 | 测试样本 |
|------|------|----------|-----------|----------|----------|
| E1 | d=3 | bit-flip | p ∈ [0.001, 0.02] | 80,000 | 400,000 |
| E2 | d=3 | depolarizing | p ∈ [0.001, 0.02] | 80,000 | 400,000 |
| E3 | d=5 | bit-flip | p ∈ [0.001, 0.02] | 80,000 | 400,000 |
| E4 | d=5 | depolarizing | p ∈ [0.001, 0.02] | 80,000 | 400,000 |

### 8.2 预期结果

- d=3 bit-flip：MLP 应接近 MWPM（简单噪声下 MWPM 近乎最优）
- d=3 depolarizing：MLP 有可能略优于 MWPM
- d=5：差距可能更明显，AI 解码器优势随码距增大
- 解码延迟：MLP < 1ms，MWPM < 1ms，BP-OSD 几 ms ~ 几十 ms

---

## 9. Risks & Mitigations

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| MLP 在简单噪声下不如 MWPM | 高 | 低 | 这是预期内的，重点展示复杂噪声场景 |
| BP-OSD 接口对接困难 | 中 | 中 | 如果来不及，可以只保留 MWPM baseline |
| Stim detection events 格式理解错误 | 中 | 高 | 先用小数据手动验证，对比 PyMatching 结果 |
| 训练不收敛 | 低 | 高 | 调整 lr、增加 epoch、检查数据标签 |
| d=5 训练太慢（CPU） | 中 | 中 | 减少 epochs 或用更小 batch size |

---

## 10. References

### 论文
- Bravyi et al., "High-threshold and low-overhead fault-tolerant quantum memory", arXiv:2308.07915
- Google DeepMind, "AlphaQubit: Accurate decoding for quantum error correction", Nature (2024)
- Harvard, "Learning high-accuracy error decoding for quantum processors", Nature s41586-024-08148-8
- Blue et al., ML decoding on BB codes, ~5x improvement (2025)
- Nachmani et al., "Neural Belief Propagation Decoding"

### 工具
- **Stim**: https://github.com/quantumlib/Stim — QEC 模拟器
- **PyMatching**: https://github.com/oscarhiggott/PyMatching — MWPM 解码器
- **ldpc**: https://github.com/quantumgizmos/ldpc — BP-OSD 解码器
- **PyTorch**: https://pytorch.org — 深度学习框架
- **ECharts**: https://echarts.apache.org — HTML 图表库

### 设计模板参考
- Eugene Yan, ML Design Docs: https://github.com/eugeneyan/ml-design-docs
