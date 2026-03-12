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

