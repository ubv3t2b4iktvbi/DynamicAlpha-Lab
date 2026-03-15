# src

这个目录存放项目源代码。

当前主要包是：

- `fsrc_sindy/`

建议理解方式是把 `src` 看成“实现层”，把 `scripts` 看成“入口层”。

## 推荐阅读顺序

1. `src/fsrc_sindy/README.md`
2. `src/fsrc_sindy/benchmarks.py`
3. `src/fsrc_sindy/selection.py`
4. `src/fsrc_sindy/factors/`
5. `src/fsrc_sindy/research/`

## 模块边界

- `scripts/` 负责命令行参数和运行入口
- `src/` 负责真正的算法实现、调度和结果组织
## Recent Implementation Updates

- Shared causal readout logic now lives in `src/fsrc_sindy/factors/readout.py`, so RC and NGRC-family models can consume the same factor-driven or fast/slow readout features.
- Reusable factor presets and library loading now live in `src/fsrc_sindy/factors/repository.py`.
- The research loop now contains identify-mode preanalysis, task-level validation gating, and theory-aware evidence synthesis rather than only flat benchmark aggregation.

## Fast-Slow Theory Update

- `src/fsrc_sindy/benchmarks.py` now exposes `fastslow_smoke` and `fastslow_theory` suites for multiscale validation.
- `src/fsrc_sindy/factors/feature_engine.py` now computes theory-grounded fast/slow observables such as `slow_level_norm`, `timescale_separation`, `slow_manifold_alignment`, `adiabatic_coherence`, and `closure_stress`.
- `src/fsrc_sindy/research/fastslow_validation.py` now wraps the general research loop with a dedicated fast/slow evidence report.

## Notebook Demo Support

- `src/fsrc_sindy/research/demo.py` provides notebook-facing helpers that reuse validation runs, build dashboard tables, and summarize theory evidence conservatively.
- The current slow-fast review notebook is `notebooks/sf/classic_sparse_sf_demo.ipynb`, which can reuse `fastslow_theory` and `fastslow_sparse_theory` artifacts when they already exist.

## Attractor-Prior Update

- `src/fsrc_sindy/attractor_prior.py` adds a pure-numpy WSGA implementation plus attractor-aware EPR-style diagnostics.
- The implementation is reusable across research and factor-mining paths, instead of being tied to a separate neural landscape trainer.
- The new `gaepr_smoke` suite is intended as a low-friction entry point for these attractor-prior experiments.
