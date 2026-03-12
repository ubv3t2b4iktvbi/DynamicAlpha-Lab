# 2026-03-11 RC 基础上的动力学因子自动挖掘设计记录

## 已落实的决策

1. **保留原 benchmark 主干**，避免破坏现有 RC / NGRC / SINDy 对照体系。
2. **新增 `factors/` + `pipeline/`**，把因子挖掘做成并行主线，而不是散落到旧模型文件里。
3. **物理识别与 RC 解耦**：
   - 识别层输出可解释中间量；
   - RC 负责快速实验筛选。
4. **全部输出进入可归档目录**，并自动生成人工审核材料。
5. **skills 双层结构**：项目专用 + 导入上游 bundle。

## 目录定案

```text
src/fsrc_sindy/factors/
src/fsrc_sindy/pipeline/
configs/
docs/
runs/factor_mining/
.claude/skills/project/
.claude/skills/upstream/
```

## 首版 identifier

- `sindy_slow`
- `spline_kan_like`
- `none`

## 首版 factor family

- order parameter
- phase
- energy
- multiscale
- physics identifier
- composite mutations

## 首版运行验证

已跑通 smoke 任务：

- `lorenz63_smoke`
- `vanderpol_smoke`

并成功产出：

- 候选因子评分表
- 入选因子库
- 人工审核清单
- 汇总报告

## 后续待办

1. 在 `highdim` / `highdim_theory` 上做系统筛选。
2. 增加多 seed 聚合与显著性报告。
3. 将 factor-mining 结果反向注入原 benchmark 的 model registry。
4. 若外部环境允许，补真 KAN backend。
