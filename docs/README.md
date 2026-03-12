# docs

这个目录放项目的说明性文档，偏“设计背景”和“架构决策”，不放运行产物。

当前包含：

- `architecture/`
  - 系统架构说明
- `theory/`
  - 核心理论联系与研究解释入口
- `plans/`
  - 设计计划和演进思路

## 推荐阅读顺序

1. 顶层 `README.md`
2. `theory/koopman_factor_introduction.md`
3. `architecture/2026-03-11-rc-dynamics-factor-mining-architecture.md`
4. `plans/2026-03-11-rc-dynamics-factor-mining-design.md`

## 这个目录适合放什么

- 架构说明
- 研究计划
- 模块间职责边界
- 重要设计决策记录

## 不建议放什么

- 临时运行日志
- 自动生成结果
- 大量重复的脚本用法说明

这些内容更适合放到 `runs/`、顶层 `README.md` 或各目录自己的 README 里。
