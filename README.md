# DynamicAlpha Lab

## Abstract | 摘要

**English.** This project studies interpretable dynamic factor discovery for nonlinear dynamical systems under sparse observation. It treats good factors as approximate Koopman coordinates, and combines benchmark ablations, coordinate diagnostics, factor mining, and a closed-loop research workflow. The system supports `accumulate` mode for factor-library growth and `identify` mode for property-guided screening of unknown systems, with confidence scoring and mandatory dynamics-expert review.

**中文。** 本项目研究稀疏观测条件下的可解释动力学因子发现问题，将好因子视为 Koopman 坐标的近似，并把基准消融、坐标诊断、因子挖掘和研究闭环整合到同一套流程中。系统支持用于扩展因子库的 `accumulate` 模式，以及面向未知系统、带性质引导筛选的 `identify` 模式，同时输出置信度并强制经过动力学专家审核。

这是一个面向非线性动力系统研究的实验仓库。它把原先的 `fast/slow + SINDy + RC + NGRC` benchmark 扩展成了一个更完整的研究工作流：

- 基准模型消融实验
- 坐标/状态表示分析
- 动力学因子挖掘
- `accumulate` / `identify` 两种主流程模式
- 研究总控 loop
- LLM 辅助解释 + 置信度输出 + 动力学专家审核门禁

项目的核心目标不是做金融交易系统，而是借用“金融因子”的表达方式，去发现适用于复杂动力系统的、可解释的、因果的动态坐标与特征。

## 1. 项目定位

这个仓库围绕一个核心问题展开：

> 我们能否从稀疏观测中自动发现一组“动态因子”，使它们既有预测力，又更接近真实动力系统的闭合坐标、Koopman 风格坐标或慢流形坐标？

当前仓库已经实现的不是单一模型，而是一套研究基础设施：

- benchmark 层：比较 RC、NGRC、fast/slow readout、structured SINDy residual 等模型
- coordinate analysis 层：比较 `raw / delay / fastslow / factor` 坐标，并检测 Markov 性、局部谱保持、可分离性、Koopman 线性不变子空间近似
- factor mining 层：在可解释因子库中做快速筛选、前向选择、人工审核排队
- research loop 层：把消融实验、坐标分析、因子挖掘串起来，形成一键式研究闭环

## 2. 核心理论介绍

### 2.1 因子挖掘与 Koopman 特征函数的联系

这个项目有一个很重要的理论视角：

> 好的动力学因子，可以被看成系统 Koopman eigenfunctions 的近似。

对动力系统

`x_{t+1} = F(x_t)`

Koopman operator 作用在函数空间上：

`K g(x) = g(F(x))`

如果某个函数 `psi(x)` 满足：

`psi(x_{t+1}) = lambda psi(x_t)`

那么它就是 Koopman eigenfunction。

而在因子模型里，我们常常也在寻找满足近似线性动力学的因子：

`f_{t+1} ~= A f_t`

所以从研究视角看：

- `psi(x)` 对应动态因子 `f(x)`
- Koopman eigenfunction 对应“稳定、可预测、结构不易漂移”的好因子
- 一组好的因子对应一个近似的 Koopman invariant subspace

这也是为什么项目里会同时关心：

- 因子预测力
- Markov 闭合性
- 谱结构保持
- Koopman 风格线性不变性

### 2.2 为什么这对当前项目重要

当前项目不是单纯把更多模型叠在一起，而是在问：

> 我们构造或发现的坐标，是否更接近真实动力系统的自然表示？

这正对应三个层次的问题：

1. 这个坐标是否更接近 Markov？
2. 这个坐标是否更接近线性不变子空间？
3. 这个坐标是否能保留吸引子的局部几何和谱结构？

因此，项目里的 `raw / delay / fastslow / factor` 比较，不只是工程 ablation，而是坐标学习问题。

### 2.3 项目里的实际落地

这个理论联系已经被接进代码：

- `coordinate_analysis`
  - 输出 `koopman_invariance_score`
  - 输出 `koopman_linear_r2`
  - 比较哪种坐标最接近 Koopman 风格线性子空间
- `factor_mining`
  - 对单个因子计算近似 `f_{t+1} ~= lambda f_t` 的标量 Koopman 诊断
  - 把这类分数作为 `identify` 模式下的筛选参考
- `research_loop`
  - 在总控总结里单列 “Best Koopman-like coordinates”
  - 把 Koopman 指标纳入置信度评估

如果你想更集中地看这部分内容，可以继续读：

- [docs/theory/koopman_factor_introduction.md](/C:/Users/12345/Desktop/DynamicAlpha-Lab/docs/theory/koopman_factor_introduction.md)

## 3. 更适合当前目标的推荐流程

如果按照 README 的研究目标和 `src/fsrc_sindy/research/loop.py` 当前代码来理解，这个仓库更合理的默认主线不是“先跑 benchmark 再解释”，而是先判断什么坐标和什么因子值得进入验证，再用 benchmark 做受控验证。

### 3.1 先选模式，而不是先选模型

- `identify`
  - 当前推荐默认主线
  - 面向新任务、未知系统、或“先理解表示再验证模型”的场景
  - 会先做原始信号性质分析，再决定是否值得继续探测 `fastslow`
- `accumulate`
  - 面向扩库、压力测试、跨任务沉淀因子
  - 适合当你已经知道当前因子库不够，想先增加候选因子再回到验证闭环

### 3.2 推荐先跑轻量闭环

入口：

```bash
python scripts/run_research_loop.py --suite smoke --tasks vanderpol_smoke --out_dir runs/research_loop/demo --model_groups fastslow_ablation --mining_mode identify
```

当前代码里的实际顺序是：

1. `preanalysis`
2. `coordinate_analysis`
3. `factor_mining`
4. `validation_gate`
5. `benchmarks`
6. `theory_evidence + confidence_report + expert_review_template`

这样排的原因是：benchmark 阶段现在依赖前面的证据。`identify` 模式会先根据原始信号判断 `fastslow` 是否值得作为假设坐标，再根据坐标分析结果决定哪些 fast/slow 模型允许进入验证，同时把 factor mining 选出来的因子 readout 变体一并接入 benchmark。

### 3.3 每一步应该回答什么问题

#### `preanalysis`

- 原始观测更像 oscillatory / multiscale / trend / bursty 的哪种组合
- `fastslow` 该被当成默认结构，还是仅作为待证伪假设

#### `coordinate_analysis`

- `raw / delay / fastslow / factor` 哪个更接近 Markov
- 哪个更保留局部谱结构
- 哪个更接近 Koopman 风格线性不变子空间

#### `factor_mining`

- 在当前任务上，哪个 identifier 和哪些因子能通过性质引导预筛、前向选择和外推验证
- 这些因子能否作为 readout 特征重新回接到 RC / NGRC 验证链路

#### `benchmarks`

- 在坐标门禁和因子 readout 变体都准备好之后，哪些模型在受控比较下真正胜出
- 模型胜出是否来自更好的表示，而不是盲目增加结构先验

### 3.4 看哪些产物来决定下一步

- `preanalysis/preanalysis_summary.*`
  - 先看任务是否真的支持 fast/slow 假设
- `coordinate_analysis/<task>/coordinate_summary.md`
  - 决定后续更该押注 `delay`、`factor` 还是 `fastslow`
- `factor_mining/<task>/<identifier>/selected_factor_library.json`
  - 看哪些因子真的进入 readout 候选
- `validation_gate.json`
  - 看哪些模型被允许进入最终验证
- `theory_evidence.md` 和 `loop_summary.md`
  - 汇总“坐标 -> 因子 -> 验证模型”的证据链

### 3.5 子入口更适合什么时候单独用

#### 只跑 benchmark

```bash
python scripts/run_benchmarks.py --suite smoke --model_groups fastslow_ablation --out_dir runs/benchmarks/smoke
```

适合在坐标假设已经比较稳定以后，专门比较模型族。

#### 只跑 coordinate analysis

```bash
python scripts/run_coordinate_analysis.py --suite smoke --tasks vanderpol_smoke --out_dir runs/coordinate_analysis/demo
```

适合在“应该用什么表示”仍不清楚的时候先做表示诊断。

#### 只跑 factor mining

```bash
python scripts/run_factor_mining.py --suite smoke --tasks vanderpol_smoke --mode identify --out_dir runs/factor_mining/identify_demo
```

适合在你已经确认任务值得挖因子，但还不想马上跑完整闭环的时候使用。

## 4. 重要设计理念

### 3.1 不把 AI 当成自动结论机器

项目明确区分：

- 代码负责跑实验、整理证据、生成报告
- LLM 负责读取结果、做理论映射、提出假设
- 人类动力学专家负责最终审核

总控 loop 会自动生成：

- `confidence_report.json`
- `expert_review_template.md`

所有 LLM 解释都应该被视为：

- `PROVISIONAL`
- `PENDING_DYNAMICS_EXPERT_REVIEW`

### 3.2 因子不是黑箱特征

每个因子都应当尽量满足：

- 因果
- 可解释
- 能说清 finance origin
- 能说清 dynamics meaning
- 不只是一阶预测捷径

### 3.3 因子挖掘与 Koopman 学习有关

当前实现已经把这层思想接进工程：

- 坐标分析里会输出 Koopman 风格线性不变子空间近似分数
- 因子挖掘里会输出单因子 `f_{t+1} ~= lambda f_t` 近似打分
- 这些指标会影响 `identify` 模式下的筛选优先级和闭环置信度

## 5. 目录结构

```text
.
|-- .claude/
|   `-- skills/
|-- archive/
|-- configs/
|   |-- README.md
|   `-- factor_mining.yaml
|-- data/
|   |-- README.md
|   `-- factorlib/
|       |-- raw/
|       `-- selected/
|-- docs/
|   |-- README.md
|   |-- architecture/
|   `-- plans/
|-- runs/
|   |-- README.md
|   |-- coordinate_analysis/
|   |-- factor_mining/
|   `-- research_loop/
|-- scripts/
|   |-- README.md
|   |-- run_benchmarks.py
|   |-- run_coordinate_analysis.py
|   |-- run_factor_mining.py
|   `-- run_research_loop.py
|-- src/
|   |-- README.md
|   `-- fsrc_sindy/
|       |-- README.md
|       |-- factors/
|       |-- models/
|       |-- pipeline/
|       `-- research/
|-- AGENTS.md
|-- CLAUDE.md
|-- MODEL_AND_EXPERIMENT_FORMULAS.md
|-- README.md
`-- requirements.txt
```

## 6. 关键脚本速查

### benchmark

```bash
python scripts/run_benchmarks.py --suite smoke --out_dir runs/benchmarks/smoke
```

### coordinate analysis

```bash
python scripts/run_coordinate_analysis.py --suite smoke --tasks vanderpol_smoke --out_dir runs/coordinate_analysis/demo
```

### factor mining

```bash
python scripts/run_factor_mining.py --suite smoke --tasks vanderpol_smoke --mode identify --out_dir runs/factor_mining/demo
```

### research loop

```bash
python scripts/run_research_loop.py --suite smoke --tasks vanderpol_smoke --out_dir runs/research_loop/demo --model_groups fastslow_ablation --mining_mode identify --full_library_search
```

## 7. 当前主要输出文件

### factor mining

单个 task / identifier 通常会包含：

- `candidate_scores.csv`
- `selected_factor_library.json`
- `metrics.json`
- `manual_review.md`
- `run_summary.md`
- `finance_to_dynamics_translation.md`
- `manifest.txt`

### coordinate analysis

单个 task 通常会包含：

- `coordinate_summary.csv`
- `coordinate_summary.md`
- `coordinate_details.json`

### research loop

总控运行会额外包含：

- `preanalysis/`
- `validation_gate.json`
- `theory_evidence.md`
- `theory_research.md`
- `quant_factor_update_plan.md`
- `loop_summary.md`
- `loop_manifest.json`
- `confidence_report.json`
- `expert_review_template.md`

## 8. 项目 skill

项目内有一组专用 skill，源码放在 `.agents/skills/project/`，需要跨工具链复用时会镜像到 `.claude/skills/project/`：

- `dynamics-factor-miner`
- `dynamics-literature-factor-updater`
- `factor-suite-orchestrator`
- `factor-review-archivist`
- `physics-identifier-swapper`
- `dynamical-theory-reasoner`
- `closed-loop-factor-orchestrator`
- `parallel-workflow-planner`

这些 skill 不是用户文档，而是给 AI 代理用的工作流程提示。详细触发方式见顶层 [AGENTS.md](/C:/Users/12345/Desktop/DynamicAlpha-Lab/AGENTS.md)。

## 9. 环境安装

建议使用 Python 3.11。

安装依赖：

```bash
pip install -r requirements.txt
```

当前核心依赖包括：

- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `tqdm`
- `PyYAML`

## 10. 阅读顺序建议

如果你第一次进入这个项目，推荐按下面顺序看：

1. 本 README
2. [scripts/README.md](/C:/Users/12345/Desktop/DynamicAlpha-Lab/scripts/README.md)
3. [src/README.md](/C:/Users/12345/Desktop/DynamicAlpha-Lab/src/README.md)
4. [src/fsrc_sindy/README.md](/C:/Users/12345/Desktop/DynamicAlpha-Lab/src/fsrc_sindy/README.md)
5. [docs/README.md](/C:/Users/12345/Desktop/DynamicAlpha-Lab/docs/README.md)
6. [docs/theory/koopman_factor_introduction.md](/C:/Users/12345/Desktop/DynamicAlpha-Lab/docs/theory/koopman_factor_introduction.md)
6. 再去看最近一次 `runs/research_loop/.../loop_summary.md`

## 11. 当前边界

这个仓库已经具备研究闭环雏形，但还不是全自动科研系统。

当前明确边界是：

- 可以自动运行实验面板
- 可以自动整理候选解释与下一步实验提示
- 可以自动给出置信度
- 不能跳过动力学专家审核
- 不应把 LLM 输出直接视为论文级结论

## 12. 后续维护建议

- 新增模型时，优先接入 `selection.py` 的 model group，而不是只加独立脚本
- 新增因子时，务必同步更新代码、归档输出和人工审核语义
- 新增研究套路时，优先接到 `run_research_loop.py`，避免再形成平行入口
## Recent Maintenance Updates

- The factor and readout stack now has a reusable causal readout layer in `src/fsrc_sindy/factors/readout.py` plus a reusable factor registry in `src/fsrc_sindy/factors/repository.py`. This lets RC and NGRC-family models share the same fast/slow or factor-based readout features instead of duplicating bespoke logic.
- `src/fsrc_sindy/research/loop.py` now adds an identify-mode preanalysis gate before validation, writes gate decisions into `validation_gate.json`, and produces a theory-oriented evidence report in `theory_evidence.md`.
- `src/fsrc_sindy/experiment.py` now supports task-specific model lists and per-task model context so gated benchmark runs can vary by task without forking the benchmark runner.
- `scripts/skill_inventory_report.py` and the project-local maintenance skills under `.agents/skills/project/` support ongoing repository upkeep, including diff summarization, README syncing, safer GitHub publishing, repo-aware vibe coding, and human-gated parallel workflow planning before execution.
- `dynamics-literature-factor-updater` now gives the repo a reusable workflow for mining factor ideas from primary dynamics papers or canonical models and translating them into local fast-slow, RG, and closure-aware factor candidates.

## Fast-Slow Validation Entry Point

For theory-focused fast/slow experiments, use:

```bash
python scripts/run_fastslow_validation.py --suite fastslow_smoke --out_dir runs/fastslow_validation/fastslow_smoke
```

This entrypoint adds:

- curated `fastslow_smoke` and `fastslow_theory` suites
- mechanism-isolation suites for `fastslow_gating_sweep`, `fastslow_observability_sweep`, `fastslow_hetero_sweep`, and the combined `fastslow_mechanism_sweeps`
- a `theory_fastslow` coordinate family for Markov / spectral / Koopman checks
- theory-oriented factor presets such as `slow_manifold_alignment`, `adiabatic_coherence`, `slow_level_norm`, and `closure_stress`
- compact outputs in `fastslow_validation_report.md`, `fastslow_mechanism_report.md`, and `fastslow_validation_summary.json`

Mechanism-sweep examples:

```bash
python scripts/run_fastslow_validation.py --suite fastslow_gating_sweep --out_dir runs/fastslow_validation/fastslow_gating_sweep --grid_mode quick --identifier_kinds sindy_slow
python scripts/run_fastslow_validation.py --suite fastslow_observability_sweep --out_dir runs/fastslow_validation/fastslow_observability_sweep --grid_mode quick --identifier_kinds sindy_slow
python scripts/run_fastslow_validation.py --suite fastslow_hetero_sweep --out_dir runs/fastslow_validation/fastslow_hetero_sweep_accumulate --grid_mode quick --identifier_kinds sindy_slow --mining_mode accumulate
```

The heteroscedasticity control sometimes needs `--mining_mode accumulate` when you want an ungated fast/slow-vs-raw benchmark, because `identify` mode may suppress fast/slow validation once the coordinate gate decides the hypothesis is not justified.

## Notebook Demo Update

- Slow-fast notebooks now live under `notebooks/sf/` and follow the `<scope>_sf_demo.ipynb` naming rule.
- `notebooks/sf/classic_sparse_sf_demo.ipynb` is the one-click review notebook for classic noisy and sparse-observation slow-fast validation.
- Notebook reruns should write to `runs/demo_notebook/sf/<scope>/`, so cached review artifacts stay grouped by family.
- `requirements.txt` now includes `matplotlib` because expert-facing review notebooks depend on plots.

## GA-EPR Prior Integration

The repository now has a pure-numpy WSGA / EPR prior path that is meant to support the question “does this coordinate preserve attractor-local geometry, basin structure, and an EPR-style quasi-potential residual?” without introducing a separate GA-EPR neural network stack.

What is integrated:

- `src/fsrc_sindy/attractor_prior.py` implements WSGA-style fixed-point discovery and Gaussian basin priors.
- `scripts/run_coordinate_analysis.py --wsga_prior` can score coordinates with attractor-aware metrics such as `wsga_epr_score` and `wsga_basin_sep_gap`.
- `scripts/run_factor_mining.py --wsga_prior` can optionally use the same score inside factor screening.
- `gaepr_smoke` adds a multistable `bistable` experiment surface for running these checks with the existing project workflow.
