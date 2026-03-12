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
- 自动生成置信度报告和专家审核模板

示例：

```bash
python scripts/run_research_loop.py --suite smoke --tasks vanderpol_smoke --out_dir runs/research_loop/demo --model_groups fastslow_ablation --mining_mode identify --full_library_search
```

### 其它辅助脚本

- `analyze_results.py`
- `merge_results.py`

这两个更偏结果汇总或后处理，不是主入口。

## 推荐使用顺序

1. 想看模型对比：`run_benchmarks.py`
2. 想看坐标好不好：`run_coordinate_analysis.py`
3. 想挖新因子：`run_factor_mining.py`
4. 想跑完整研究面板：`run_research_loop.py`

