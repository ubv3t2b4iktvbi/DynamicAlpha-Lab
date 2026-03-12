# DynamicAlpha Lab Repo Map

This file is the code-authoritative quick map for maintenance, validation, and workflow decisions.

## Research Goal

The repository is not primarily a model zoo. Its main goal is to discover interpretable dynamical factors and coordinates that are:

- predictive
- closer to Markov closure
- closer to Koopman-style linear invariance
- compatible with human expert review

Because of that goal, the best default workflow is evidence-first, then validation.

## Default Workflow

### Recommended default: `identify`

Use `identify` when you are studying a new task or trying to understand what representation should be trusted.

Current code path in `src/fsrc_sindy/research/loop.py`:

1. `preanalysis`
2. `coordinate_analysis`
3. `factor_mining`
4. `validation_gate`
5. `benchmarks`
6. `theory_evidence`
7. `confidence_report`
8. `expert_review_template`

Why this order:

- preanalysis decides whether `fastslow` is even worth probing
- coordinate analysis decides which representation is closest to closure / Koopman / spectral preservation
- factor mining produces factor readout variants that are later benchmarked
- benchmarks are therefore downstream validation, not the first research lens

### Secondary workflow: `accumulate`

Use `accumulate` when the factor bank is too weak and the next bottleneck is library growth, not model comparison.

## Entrypoints

### Closed loop

```bash
python scripts/run_research_loop.py --suite smoke --tasks vanderpol_smoke --out_dir runs/research_loop/demo --model_groups fastslow_ablation --mining_mode identify
```

### Coordinate diagnostics only

```bash
python scripts/run_coordinate_analysis.py --suite smoke --tasks vanderpol_smoke --out_dir runs/coordinate_analysis/demo
```

### Factor mining only

```bash
python scripts/run_factor_mining.py --suite smoke --tasks vanderpol_smoke --mode identify --out_dir runs/factor_mining/demo
```

### Benchmarks only

```bash
python scripts/run_benchmarks.py --suite smoke --model_groups fastslow_ablation --out_dir runs/benchmarks/smoke
```

## Where To Read Or Edit

### CLI entry layer

- `scripts/run_benchmarks.py`
- `scripts/run_coordinate_analysis.py`
- `scripts/run_factor_mining.py`
- `scripts/run_research_loop.py`

### Core implementation

- `src/fsrc_sindy/benchmarks.py`: task and suite definitions
- `src/fsrc_sindy/experiment.py`: benchmark execution
- `src/fsrc_sindy/selection.py`: model registry and search spaces
- `src/fsrc_sindy/research/coordinate_analysis.py`: representation diagnostics
- `src/fsrc_sindy/pipeline/factor_mining.py`: suite-level factor mining orchestration
- `src/fsrc_sindy/research/loop.py`: code-authoritative closed loop

### Factor stack

- `src/fsrc_sindy/factors/factor_bank.py`
- `src/fsrc_sindy/factors/identifiers.py`
- `src/fsrc_sindy/factors/miner.py`
- `src/fsrc_sindy/factors/repository.py`
- `src/fsrc_sindy/factors/readout.py`

## Main Artifacts To Inspect

- `preanalysis/preanalysis_summary.*`
- `coordinate_analysis/<task>/coordinate_summary.md`
- `factor_mining/<task>/<identifier>/selected_factor_library.json`
- `validation_gate.json`
- `theory_evidence.md`
- `loop_summary.md`
- `confidence_report.json`
- `expert_review_template.md`

## Minimal Validation Checks

Use the smallest check that matches the change.

### Cheap checks

```bash
python scripts/run_research_loop.py --help
python -m compileall src scripts
```

### Smoke closed loop

```bash
python scripts/run_research_loop.py --suite smoke --tasks vanderpol_smoke --out_dir runs/research_loop/smoke_check --model_groups fastslow_ablation --mining_mode identify --identifier_kinds sindy_slow
```

### When to avoid `--full_library_search`

In `identify` mode, prefer leaving `--full_library_search` off unless you explicitly want to bypass property-guided prescreening for debugging or stress tests.
