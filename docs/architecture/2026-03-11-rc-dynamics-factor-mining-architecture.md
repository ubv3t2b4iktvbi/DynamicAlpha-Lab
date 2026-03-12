# RC 基础上的动力学因子自动挖掘架构设计

## 1. 设计目标

在原 `fsrc_sindy` benchmark 基础上增加一条新的研究主线：

1. 从稀疏观测中提取有动力学意义的候选因子；
2. 将“金融中有效的因子”翻译成复杂系统中的 slow-fast / phase / energy / multiscale 证据；
3. 用物理识别模块给出可解释的中间状态；
4. 用 RC 做大规模快速筛选；
5. 把结果自动归档，并进入人工审核队列；
6. 用 skill 形式把常见工作流封装起来。

这条链路的关键不是追求一次性最强模型，而是构建 **可持续扩展的研究基础设施**。

## 2. 参考思路抽象

### 2.1 从 QuantaAlpha 借鉴的部分

借鉴的是“顶层目录 + pipeline/factors/配置中心 + 因子库 + 独立分析”的组织方式，而不是照搬量化金融实现。

在本项目中的对应关系：

- `configs/`：实验参数的单一入口
- `src/fsrc_sindy/pipeline/`：批量运行与汇总
- `src/fsrc_sindy/factors/`：因子定义、自动筛选、归档、审核
- `runs/factor_mining/`：每次运行的完整产物
- `data/factorlib/`：沉淀后的因子库

### 2.2 从 AI-research-SKILLs 借鉴的部分

借鉴的是“工作流能力封装成 skills，而不是把所有知识写进一个超长 prompt”。

本项目分为两类 skills：

- `project/`：项目专用 skill
- `upstream/`：导入的通用 research skills

这样后续做：
- 结果审核
- 证据分级
- 理论扩展
- 目标实验设计

时，不必重复造轮子。

## 3. 功能分层

### 3.1 观测层
输入为标量或稀疏观测序列 `y_t`。

### 3.2 特征层
由 `DynamicsFeatureEngine` 计算 causal features：

- fast / slow manifold
- order parameter `m`
- `dm`, `d2m`
- phase bottom score
- energy ratio
- critical window
- collapse quality
- compression ratio
- support recovery

### 3.3 物理识别层
由 `identifiers.py` 定义统一接口：

- `sindy_slow`
- `spline_kan_like`
- `none`

所有识别器都输出统一的物理中间量：

- `id_drift_pred`
- `id_drift_pred_norm`
- `id_drift_surprise`
- `id_drift_surprise_norm`
- `id_drift_alignment`

### 3.4 因子层
由 `FactorSpec` 与 `factor_bank.py` 定义。

因子不是任意黑盒特征，而是：
- 有公式
- 有 finance origin
- 有 dynamics meaning
- 有 theory tags

### 3.5 快速筛选层
由 `ReservoirTeacherForcedScreen` 完成：

- 固定 RC reservoir state
- 快速比较候选因子的 one-step 贡献
- 输出候选因子排名

### 3.6 严格验证层
由 `FactorAugmentedRCModel` + `forward selection` 完成：

- 对 top 候选进行 rollout 验证
- 只接受对验证分数真正有提升的因子
- 防止“只改善 one-step 但伤害长程 rollout”

### 3.7 归档与审核层
每个 task / identifier 组合都会产出：

- `candidate_scores.csv`
- `selected_factor_library.json`
- `metrics.json`
- `manual_review.md`
- `run_summary.md`
- `manifest.txt`

## 4. 数据流

```text
y(t)
  -> DynamicsFeatureEngine
  -> Physics Identifier
  -> Factor Bank evaluation
  -> RC one-step screen
  -> Top-M factors
  -> FactorAugmentedRC forward selection
  -> Selected factor library
  -> Archive + manual review queue
```

## 5. 为什么 RC 适合放在因子挖掘层

因为这里的核心任务不是“用最重模型一把做完”，而是：

1. 保持高速；
2. 可以跑很多候选实验；
3. 在相同 reservoir 下公平比较不同因子的增量贡献；
4. 将复杂理论搜索压缩为较小的候选集合，再交给更慢、更强的模型做后验验证。

所以 RC 在这里是 **实验加速器**，不是最终唯一理论模型。

## 6. 为什么还保留 SINDy / KAN-like 识别层

因为纯 RC 虽快，但不天然提供“慢漂移”“预期差”“机制切换证据”等可解释中间量。

物理识别层的作用是：

1. 给出慢流形漂移预测；
2. 给出 drift surprise；
3. 给出“当前是否偏离可解释规律”的证据；
4. 为后续因子提供理论 grounding。

## 7. skill 设计

### 项目专用 skill
- `dynamics-factor-miner`
- `physics-identifier-swapper`
- `factor-review-archivist`
- `factor-suite-orchestrator`

### 导入的上游 skill bundle
- `ablation-results-auditor`
- `insight-evidence-grader`
- `targeted-experiment-designer`
- `theory-expansion-engine`
- 其他 dynamics research skills

## 8. 当前实现边界

当前版本已实现：

- factor bank
- identifier registry
- RC screening
- forward selection
- suite pipeline
- archive + review
- skill layout

当前尚未实现但留好接口：

- 真正外部 KAN backend
- 多 seed 统计
- 原 benchmark registry 中的一体化 model entry
- UI / dashboard

## 9. 后续扩展建议

1. 新 identifier 只需复用统一接口。
2. 新 factor family 只需增加 `FactorSpec` 与公式实现。
3. 新人工审核规则可在 `review.py` 中扩展。
4. 如需更复杂的进化式 factor mining，可在 `miner.py` 中加入 mutation / crossover，而无需动 archive 与 pipeline 层。
