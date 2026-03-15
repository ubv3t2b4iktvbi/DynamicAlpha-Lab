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
- `research/loop.py` now includes identify-mode preanalysis, validation gating, theory-aware evidence synthesis, autonomous theory-research summaries, and source-guided quant-factor update plans.

## Fast-Slow Validation Additions

- `benchmarks.py` now contains `fastslow_smoke` and `fastslow_theory` suites that focus on multiscale systems instead of mixing in non-multiscale tasks.
- `benchmarks.py` also contains `fastslow_finance_theory`, which stress-tests slow-fast coordinates under volatility-clustered process and observation noise.
- `benchmarks.py` now also exposes mechanism-isolation suites for `fastslow_gating_sweep`, `fastslow_observability_sweep`, `fastslow_hetero_sweep`, and the combined `fastslow_mechanism_sweeps`.
- `systems.py` now supports asymmetric Lorenz-96 two-scale coupling via `slow_to_fast_h` and `fast_to_slow_h`, plus matched-energy observation noise for heteroscedastic controls.
- `factors/feature_engine.py` and `factors/factor_bank.py` now expose slow-manifold and adiabatic observables for factor mining and readout validation.
- `research/coordinate_analysis.py` now accepts the `theory_fastslow` coordinate family.
- `research/fastslow_validation.py` now writes both the overall `fastslow_validation_report.md` and a sweep-specific `fastslow_mechanism_report.md`, while propagating task metadata such as `sweep_group`, `sweep_value`, `observability_profile`, and `noise_profile` into downstream summaries.

## Notebook Demo Helpers

- `research/demo.py` adds reusable helpers for notebook orchestration, artifact reuse, factor-frequency summaries, benchmark tables, and cautious theory-evidence grading.
- The paired notebook path is `notebooks/sf/classic_sparse_sf_demo.ipynb`, with notebook-owned reruns grouped under `runs/demo_notebook/sf/`.

## Attractor Prior Additions

- `attractor_prior.py` now contains a pure-numpy WSGA implementation, Gaussian fixed-point priors, label assignment, and EPR-style coordinate diagnostics.
- `research/coordinate_analysis.py` can optionally score coordinates against this prior via `wsga_epr_score`, `wsga_basin_sep_gap`, and related attractor-aware metrics.
- `factors/miner.py` can optionally consume the same prior as a screening signal and as part of forward selection / final scoring, controlled by `FactorMiningConfig.use_wsga_prior` and `FactorMiningConfig.epr_weight_strength`.
- `benchmarks.py` now exposes `gaepr_smoke` with `bistable_wsga_smoke` and `bistable_wsga_noisy` for multistable attractor-prior experiments.

## Factor Curation Additions

- `factors/curation.py` now computes factor-layer evidence from effectiveness, target relevance, and redundancy.
- `factors/miner.py` writes `core / extended / experimental / holding` assignments back into `candidate_scores`, selected libraries, and archive artifacts.
- `factors/archive.py` now preserves `layered_factor_library.json` and `future_factor_queue.json` so later mined factors can stay in a promotion queue instead of entering the default library immediately.

## Manifold Role Taxonomy

- `factors/manifold_roles.py` now reorganizes the factor library around broad manifold representations rather than only finance-style families.
- `references/manifold-factor-theory.md` records the shared geometric structures, primary-source pointers, and the current theory-to-factor translation.
- The main role set is:
  - `chart_position`
  - `tangent_flow`
  - `normal_amplitude`
  - `closure_memory`
  - `coarse_geometry`
  - `control_drive`
  - `regime_boundary`
  - `surprise_alignment`
- `factors/repository.py` now exposes role-aware presets such as `manifold_chart_position`, `manifold_tangent_flow`, and the broader `broad_manifold_core` preset.
- The intended default representation story is:
  - use `broad_manifold_core` to cover generally reusable low-dimensional geometry
  - keep `control_drive`, `regime_boundary`, and `surprise_alignment` as task-dependent extensions
