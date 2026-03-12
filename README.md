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

- [docs/theory/koopman_factor_introduction.md](/C:/Users/12345/Desktop/rc_dynamics_factor_mining_project/docs/theory/koopman_factor_introduction.md)

## 3. 当前主流程

### 2.1 基准实验

入口：

```bash
python scripts/run_benchmarks.py --suite smoke --out_dir runs/benchmarks/smoke
```

用途：

- 运行现有模型族在标准任务套件上的对比
- 分析 `fastslow_ablation`、`memory_ablation`、`structured_ablation` 等模型组
- 产出逐任务指标，用于后续理论解释

### 2.2 坐标分析

入口：

```bash
python scripts/run_coordinate_analysis.py --suite smoke --tasks vanderpol_smoke --out_dir runs/coordinate_analysis/demo
```

用途：

- 检测某种坐标是否更接近 Markov
- 比较坐标是否保持局部 Jacobian / 谱结构
- 分析坐标间弱耦合程度
- 评估它是否更接近 Koopman 线性不变子空间

默认比较的坐标：

- `raw`
- `delay`
- `fastslow`
- `factor`

### 2.3 因子挖掘

入口：

```bash
python scripts/run_factor_mining.py --suite smoke --tasks vanderpol_smoke --out_dir runs/factor_mining/demo
```

当前支持两种模式：

- `accumulate`
  - 面向“因子积累/训练”
  - 更偏研究式扩库、压力测试、跨任务筛选
  - 保留全库视角，强调人工审查与后续归档
- `identify`
  - 面向“未知系统识别/推理”
  - 先做信号性质分析，再对因子库加权预筛
  - 性质分析结果会给因子权重一个初始化
  - 也支持 `--full_library_search` 强制全库搜索

示例：

```bash
python scripts/run_factor_mining.py --suite smoke --tasks vanderpol_smoke --mode identify --full_library_search --out_dir runs/factor_mining/identify_demo
```

### 2.4 研究总控闭环

入口：

```bash
python scripts/run_research_loop.py --suite smoke --tasks vanderpol_smoke --out_dir runs/research_loop/demo --model_groups fastslow_ablation --mining_mode identify --full_library_search
```

这个脚本会自动串起来：

1. benchmark 消融实验
2. coordinate analysis
3. factor mining
4. 置信度报告
5. 专家审核模板

它是当前仓库最接近“一键式研究闭环”的入口。

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

- `loop_summary.md`
- `loop_manifest.json`
- `confidence_report.json`
- `expert_review_template.md`

## 8. 项目 skill

项目内有一组专用 skill，主要放在 `.claude/skills/project/`：

- `dynamics-factor-miner`
- `factor-suite-orchestrator`
- `factor-review-archivist`
- `physics-identifier-swapper`
- `dynamical-theory-reasoner`
- `closed-loop-factor-orchestrator`

这些 skill 不是用户文档，而是给 AI 代理用的工作流程提示。详细触发方式见顶层 [AGENTS.md](/C:/Users/12345/Desktop/rc_dynamics_factor_mining_project/AGENTS.md)。

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
2. [scripts/README.md](/C:/Users/12345/Desktop/rc_dynamics_factor_mining_project/scripts/README.md)
3. [src/README.md](/C:/Users/12345/Desktop/rc_dynamics_factor_mining_project/src/README.md)
4. [src/fsrc_sindy/README.md](/C:/Users/12345/Desktop/rc_dynamics_factor_mining_project/src/fsrc_sindy/README.md)
5. [docs/README.md](/C:/Users/12345/Desktop/rc_dynamics_factor_mining_project/docs/README.md)
6. [docs/theory/koopman_factor_introduction.md](/C:/Users/12345/Desktop/rc_dynamics_factor_mining_project/docs/theory/koopman_factor_introduction.md)
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
