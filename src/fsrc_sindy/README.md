# fsrc_sindy

这是项目的核心 Python 包。

它包含四条主线：

1. benchmark 基础设施
2. 模型实现
3. 因子挖掘
4. 研究分析与总控 loop

## 目录说明

### 顶层模块

- `benchmarks.py`
  - 定义 suite 和任务集合
- `systems.py`
  - 定义动力系统、观测方式、仿真入口
- `experiment.py`
  - benchmark suite 级运行器
- `selection.py`
  - 模型注册、搜索空间和模型选择逻辑
- `metrics.py`
  - 误差、频谱、自相关等基础指标
- `fastslow.py`
  - causal fast/slow encoder

### `models/`

这里放模型实现，包括：

- RC
- NGRC
- hybrid RC+NGRC
- SINDy 及其 structured residual 变体

### `factors/`

这里放因子挖掘主逻辑，包括：

- `feature_engine.py`
  - 因果特征构造
- `factor_bank.py`
  - 因子定义与公式
- `identifiers.py`
  - 物理识别后端
- `rc_proxy.py`
  - RC 快速筛选代理
- `miner.py`
  - 候选筛选与前向选择
- `property_analyzer.py`
  - 信号性质分析与 Koopman 风格单因子打分
- `review.py`
  - 人工审核队列生成
- `archive.py`
  - 运行产物落盘

### `pipeline/`

- `factor_mining.py`
  - suite 级因子挖掘 orchestration

### `research/`

这里放更偏“研究工作流”的分析模块：

- `coordinate_analysis.py`
  - 坐标层证据面板
- `loop.py`
  - 总控闭环

## 你最可能会改的地方

### 如果你在加新因子

优先看：

- `factors/factor_bank.py`
- `factors/feature_engine.py`
- `factors/miner.py`

### 如果你在加新 identifier

优先看：

- `factors/identifiers.py`
- `factors/miner.py`

### 如果你在加新坐标诊断

优先看：

- `research/coordinate_analysis.py`
- `research/loop.py`

### 如果你在加新模型

优先看：

- `models/`
- `selection.py`
- `experiment.py`

## 修改时的建议

- 不要只改模型而不接入 `selection.py`
- 不要只改 factor scoring 而不更新 review / archive 产物
- 不要绕开 `research/loop.py` 再平行造一个总控入口
## Recent Package Updates

- `experiment.py` now supports task-specific model lists and per-task model context, which is used by gated benchmark execution.
- `models/rc.py` and `models/ngrc.py` now reuse a shared causal readout layer instead of maintaining separate fast/slow feature plumbing.
- `factors/readout.py` provides the shared causal readout encoder for mined factors and legacy fast/slow readout features.
- `factors/repository.py` provides reusable factor presets, a central registry, and selected-library loading helpers.
- `research/loop.py` now includes identify-mode preanalysis, validation gating, and theory-aware evidence synthesis outputs.

