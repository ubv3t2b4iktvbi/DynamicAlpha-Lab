This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This repository extends the original fast-slow + SINDy + RC benchmark into an RC-based automatic dynamical factor mining workflow.

The core idea is:
1. build causal slow-fast / phase / energy / multiscale features from sparse observations;
2. optionally attach a physics-identification backbone such as slow SINDy or a spline-based KAN-like surrogate;
3. screen many candidate factors quickly with a reservoir-computing proxy;
4. forward-select a small factor set;
5. archive all candidate tables and expose a manual review queue.

## High-level architecture

### Existing benchmark core
- `src/fsrc_sindy/systems.py`: synthetic dynamical systems and observation maps.
- `src/fsrc_sindy/benchmarks.py`: benchmark suites and task definitions.
- `src/fsrc_sindy/models/`: RC, NGRC, SINDy, and hybrid forecasting models.
- `scripts/run_benchmarks.py`: original benchmark entry point.

### New factor-mining stack
- `src/fsrc_sindy/factors/feature_engine.py`: causal feature engine. Produces order-parameter, phase, energy, and multiscale features.
- `src/fsrc_sindy/factors/factor_bank.py`: factor definitions and factor-expression evaluation.
- `src/fsrc_sindy/factors/identifiers.py`: pluggable physics-identification backbones.
- `src/fsrc_sindy/factors/rc_proxy.py`: RC screening cache and factor-augmented RC forecaster.
- `src/fsrc_sindy/factors/miner.py`: candidate screening + forward selection.
- `src/fsrc_sindy/factors/review.py`: manual review queue generation.
- `src/fsrc_sindy/factors/archive.py`: run artifact writing.
- `src/fsrc_sindy/pipeline/factor_mining.py`: suite-level orchestration.
- `scripts/run_factor_mining.py`: CLI entry point for the new pipeline.

### Skills layout
- `.claude/skills/project/`: project-specific skills for factor mining, backbone swapping, review, and suite orchestration.
- `.claude/skills/project/math-implementation-validator/`: math-audit workflow for code-to-formula translation, operator checks, and numerical test design.
- `.claude/skills/upstream/dynamics-research-skills/`: vendored upstream research skills bundle.

## Commands

### Environment
```bash
pip install -r requirements.txt
```

### Original benchmark smoke test
```bash
python scripts/run_benchmarks.py --suite smoke --out_dir runs/smoke
```

### New factor-mining smoke test
```bash
python scripts/run_factor_mining.py --suite smoke --out_dir runs/factor_mining/smoke
```

### Targeted factor mining on a specific task
```bash
python scripts/run_factor_mining.py --suite smoke --tasks vanderpol_smoke --out_dir runs/factor_mining/vanderpol_smoke
```

### Use explicit config
```bash
python scripts/run_factor_mining.py --suite common --config configs/factor_mining.yaml --out_dir runs/factor_mining/common
```

## Working rules

1. Preserve causal feature computation. Do not introduce non-causal rolling features.
2. If replacing the physics identifier, keep the identifier interface stable.
3. Any change that affects factor semantics must update both the code and the review/archive outputs.
4. Do not delete archived run artifacts without replacing them with a new manifest-complete run.
5. When adding a factor, write down its finance origin and its dynamics meaning.
6. For tasks touching 3+ files, split the work into smaller architecture-preserving changes.

## Design notes

- `spline_kan_like` is intentionally labeled KAN-like, not a full external KAN dependency.
- RC is the fast screening layer. Slower model families should be used only after candidate compression.
- The factor bank is intentionally interpretable. Avoid opaque latent factors unless a separate validation path exists.
- Manual review is not optional. A factor that only improves one-step error but damages rollout should not be promoted.
