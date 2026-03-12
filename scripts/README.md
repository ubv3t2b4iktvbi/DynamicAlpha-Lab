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
- `raw / delay / fastslow / theory_fastslow / factor` coordinate diagnostics
- factor mining with the dedicated `configs/fastslow_theory_factor_mining.yaml` preset
- a compact `fastslow_validation_report.md` plus `fastslow_validation_summary.json`

Example:

```bash
python scripts/run_fastslow_validation.py --suite fastslow_smoke --out_dir runs/fastslow_validation/fastslow_smoke
```
