# scripts

这个目录是项目的命令行入口层。

如果你只想“运行项目”，通常应该先看这个目录。

## 入口脚本

### `run_benchmarks.py`

用途：

- 运行 benchmark suite
- 比较不同模型族或 model group

示例：

```bash
python scripts/run_benchmarks.py --suite smoke --out_dir runs/benchmarks/smoke
```

### `run_coordinate_analysis.py`

用途：

- 比较不同坐标表示
- 输出 Markov closure、谱保持、可分离性、Koopman 近似指标

示例：

```bash
python scripts/run_coordinate_analysis.py --suite smoke --tasks vanderpol_smoke --out_dir runs/coordinate_analysis/demo
```

### `run_factor_mining.py`

用途：

- 运行动力学因子挖掘
- 支持 `accumulate` / `identify` 两种模式

示例：

```bash
python scripts/run_factor_mining.py --suite smoke --tasks vanderpol_smoke --mode identify --out_dir runs/factor_mining/demo
```

### `run_research_loop.py`

用途：

- 一键执行研究闭环
- 串联 benchmark、coordinate analysis、factor mining
- 自动生成理论证据、theory research、外部量化因子更新计划、置信度报告和专家审核模板

示例：

```bash
python scripts/run_research_loop.py --suite smoke --tasks vanderpol_smoke --out_dir runs/research_loop/demo --model_groups fastslow_ablation --mining_mode identify --full_library_search
```

### 其它辅助脚本

- `analyze_results.py`
- `merge_results.py`
- `skill_inventory_report.py`

这两个更偏结果汇总或后处理，不是主入口。

## 推荐使用顺序

1. 想看模型对比：`run_benchmarks.py`
2. 想看坐标好不好：`run_coordinate_analysis.py`
3. 想挖新因子：`run_factor_mining.py`
4. 想跑完整研究面板：`run_research_loop.py`
## Maintenance Notes

### `skill_inventory_report.py`

Use this utility to compare `.agents/skills/project/` and `.claude/skills/project/`, detect missing mirrors, and spot content or metadata drift before promoting or publishing project-local skills.

Example:
```bash
python scripts/skill_inventory_report.py --json-out runs/skill_inventory/report.json --md-out runs/skill_inventory/report.md
```

This script is intended for skill maintenance and repository hygiene, not for the main dynamics research loop.

## Fast-Slow Validation Update

Use `run_fastslow_validation.py` for a theory-focused fast/slow experiment loop that bundles:

- curated `fastslow_smoke` / `fastslow_theory` suites
- targeted mechanism suites: `fastslow_gating_sweep`, `fastslow_observability_sweep`, `fastslow_hetero_sweep`, and the combined `fastslow_mechanism_sweeps`
- `raw / delay / fastslow / theory_fastslow / factor` coordinate diagnostics
- factor mining with the dedicated `configs/fastslow_theory_factor_mining.yaml` preset
- a compact `fastslow_validation_report.md`, a sweep-focused `fastslow_mechanism_report.md`, plus `fastslow_validation_summary.json`

Example:

```bash
python scripts/run_fastslow_validation.py --suite fastslow_smoke --out_dir runs/fastslow_validation/fastslow_smoke
```

Mechanism-sweep examples:

```bash
python scripts/run_fastslow_validation.py --suite fastslow_gating_sweep --out_dir runs/fastslow_validation/fastslow_gating_sweep --grid_mode quick --identifier_kinds sindy_slow
python scripts/run_fastslow_validation.py --suite fastslow_observability_sweep --out_dir runs/fastslow_validation/fastslow_observability_sweep --grid_mode quick --identifier_kinds sindy_slow
python scripts/run_fastslow_validation.py --suite fastslow_hetero_sweep --out_dir runs/fastslow_validation/fastslow_hetero_sweep --grid_mode quick --identifier_kinds sindy_slow --mining_mode accumulate
```

Use `--mining_mode accumulate` on the heteroscedasticity control when you want an ungated head-to-head fast/slow benchmark, because `identify` mode may legitimately suppress fast/slow validation if the coordinate gate rejects it.

## RG Readout Update

Use `--model_groups rg_ablation` to compare the new RG-style readout models against raw and fast/slow baselines. The reusable preset name is `rg_readout`, and it exposes coarse-grained order, control, noise, and beta-flow observables without touching the private `finance/` directory.

Example:

```bash
python scripts/run_benchmarks.py --suite smoke --tasks vanderpol_smoke --model_groups rg_ablation --out_dir runs/benchmarks/rg_smoke
```

## SF-RG Interaction Update

Use `--model_groups sf_rg_ablation` to compare the new hierarchical fast/slow plus RG interaction readouts against raw, fast/slow-only, and RG-only baselines. The new variants are `rc_sf_rg_interaction` and `ngrc_sf_rg_interaction`.

Example:

```bash
python scripts/run_benchmarks.py --suite smoke --tasks vanderpol_smoke --model_groups sf_rg_ablation --out_dir runs/benchmarks/sf_rg_smoke
```

For the sparser gated variant, use `--model_groups sf_rg_gated_ablation` or `--model_groups sf_rg_gate_compare`. The new variants are `rc_sf_rg_gated` and `ngrc_sf_rg_gated`, and they keep only a few mechanistically selected `SF x RG` terms rather than the full pairwise interaction surface.

Example:

```bash
python scripts/run_benchmarks.py --suite fastslow_mechanism_sweeps --tasks lorenz96_twoscale_obs_slow0 lorenz96_twoscale_obs_mixed_projection --model_groups sf_rg_gate_compare --out_dir runs/benchmarks/sf_rg_gated_demo
```

## Takens-RG Residual NGRC Update

Use `--model_groups ngrc_takens_rg_ablation` or `--model_groups ngrc_takens_rg_boundary` to compare the new `ngrc_takens_rg_residual` model against raw NGRC, fast/slow readout, RG readout, and sparse SF-RG gating. This variant keeps the Takens delay backbone as the main state representation and uses sparse RG factors only for a residual regime-conditioned correction.

Example:

```bash
python scripts/run_benchmarks.py --suite fastslow_mechanism_sweeps --tasks lorenz96_twoscale_obs_slow0 lorenz96_twoscale_gate_s2f_0p8 --model_groups ngrc_takens_rg_boundary --seeds 123 231 341 --out_dir runs/benchmarks/ngrc_takens_rg_boundary
```

For representation-level checks, `run_coordinate_analysis.py` now accepts `delay_rg_joint`, which augments the Takens delay coordinate with the same sparse RG macro variables used by the residual NGRC correction.

## Takens-RG Validation Study

Use `run_takens_rg_validation.py` when you want the full multi-seed validation package for the Takens-RG story instead of a single benchmark table. The workflow bundles:

- RG specificity controls against lagged-RG and random-summary matched controls
- conditioning-form ablations comparing additive and interaction-style residual corrections
- mechanism-boundary sweeps across gating and observability structure
- delay-sufficiency sweeps for raw NGRC, RG readout, and Takens-RG residual NGRC
- coordinate diagnostics comparing `raw`, `delay`, `delay_rg_joint`, `rg`, and `fastslow`

Example:

```bash
python scripts/run_takens_rg_validation.py --out_dir runs/benchmarks/ngrc_takens_rg_validation --seeds 123 231 341 451 561
```

## Factor Changepoint Workflow

Use `run_factor_changepoint.py` when you want an offline-identification plus online-detection workflow rather than a long-horizon forecast benchmark.

The current default experiment:

- fits `rc_rg_readout` and `ngrc_rg_readout` on the pre-change regime
- builds detector features from one-step residuals plus causal readout factors
- trains an online `FTRL` head for each predictor and a joint `RC+NGRC` detector
- reports pointwise post-change probabilities and episode-level detection delays / false alarms

Example:

```bash
python scripts/run_factor_changepoint.py --out_dir runs/factor_changepoint/vanderpol_rg_switch
```

## GA-EPR Prior Update

Use `gaepr_smoke` when you want a multistable smoke task that is friendly to WSGA-style attractor priors.

Coordinate-analysis example:

```bash
python scripts/run_coordinate_analysis.py --suite gaepr_smoke --tasks bistable_wsga_smoke --coordinates raw factor --wsga_prior --wsga_noise_strength 0.1 --out_dir runs/coordinate_analysis/gaepr_smoke
```

Factor-mining example:

```bash
python scripts/run_factor_mining.py --suite gaepr_smoke --tasks bistable_wsga_smoke --identifier_kinds sindy_slow --wsga_prior --wsga_noise_strength 0.1 --epr_weight_strength 0.1 --out_dir runs/factor_mining/gaepr_smoke
```

What the new flags do:

- `--wsga_prior` builds a pure-numpy fixed-point attractor prior from the true benchmark drift.
- `run_coordinate_analysis.py` now writes attractor-aware columns such as `wsga_epr_score` and `wsga_basin_sep_gap`.
- `run_factor_mining.py` can optionally fold `wsga_epr_score` into candidate screening, forward selection, and the final combined `validation_score` when `--epr_weight_strength` is positive.

## Factor Layering Update

`run_factor_mining.py` now also writes:

- `layered_factor_library.json`
- `future_factor_queue.json`

The layering step combines three evidence channels:

- effectiveness: RC gain, Koopman score, and optional attractor-prior support
- target relevance: correlation plus normalized mutual information to the next-step target
- redundancy: correlation plus mutual information against higher-ranked factors

This keeps the default library compact while storing promising newly mined or composite factors in a later-promotion queue.

The same run summaries now also expose a manifold-role view of the factor library, so the selected and screened factors can be read as broad geometric coordinates such as `chart_position`, `tangent_flow`, `normal_amplitude`, `closure_memory`, and `coarse_geometry` instead of only finance-derived families.
