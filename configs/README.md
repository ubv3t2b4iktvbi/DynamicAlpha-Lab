# configs

这个目录存放项目主流程使用的配置文件。

当前最重要的文件是：

- `factor_mining.yaml`

它控制：

- factor mining 模式默认值
- identifier 列表
- RC screening 参数
- 性质分析和 Koopman 权重
- feature engine 参数

## 主要字段

### `factor_mining`

- `mode`
  - `accumulate`：偏扩库和积累
  - `identify`：偏未知系统识别
- `identifier_kinds`
- `screening_ridge`
- `property_weight_strength`
- `koopman_weight_strength`
- `property_prescreen_top_k`
- `full_library_search`
- `screen_top_m`
- `max_selected_factors`

### `rc`

控制 RC screening 和因子增强 RC 的默认超参数。

### `features`

控制 `DynamicsFeatureEngine` 的因果特征窗口与尺度参数。

## 修改建议

- 修改 `mode` 前，先明确你是在做“因子积累”还是“未知系统识别”
- 修改窗口参数时，尽量配合 `coordinate_analysis` 一起看，不要只看最终 RMSE
- 如果你调高 `property_weight_strength` 或 `koopman_weight_strength`，要检查筛选是否被先验过度主导

