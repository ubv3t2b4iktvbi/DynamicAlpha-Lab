# 因子挖掘与 Koopman 特征函数

这份文档是项目里的核心理论介绍，目的不是完整讲授 Koopman 理论，而是说明它和本仓库当前实现之间的直接关系。

## 1. 核心观点

本项目采用的一个重要研究视角是：

> 好的动力学因子，本质上可以看作系统 Koopman eigenfunctions 的近似。

也就是说，因子挖掘不只是“找预测特征”，更是在寻找更自然、更稳定的动力学坐标。

## 2. Koopman 视角

考虑离散动力系统：

`x_{t+1} = F(x_t)`

Koopman operator 定义为：

`K g(x) = g(F(x))`

它不是直接作用在状态 `x` 上，而是作用在观测函数 `g(x)` 上。

如果存在某个函数 `psi(x)` 满足：

`psi(x_{t+1}) = lambda psi(x_t)`

那么 `psi` 就是 Koopman eigenfunction。

在这些坐标下，原本非线性的动力系统会变成更接近线性的演化：

`psi_{t+1} = Lambda psi_t`

## 3. 因子模型视角

在金融或统计建模里，常见的因子写法是：

`r_t = B f_t + epsilon_t`

其中 `f_t` 是因子。

很多好的因子往往也有自己的动力学结构：

`f_{t+1} ~= A f_t`

这和 Koopman 的形式是高度一致的。

## 4. 两者为什么可以统一理解

可以把两边对应起来：

- `psi(x)` -> `f(x)`
- eigenfunction -> factor
- invariant coordinate -> predictive feature
- `psi(x_{t+1}) = lambda psi(x_t)` -> `f_{t+1} ~= A f_t`

所以在这个项目中，“找好因子”和“找更好的动力学坐标”并不是两件事。

## 5. 对本项目的直接意义

### 5.1 为什么不是只看 RMSE

一个因子哪怕短期预测很好，也可能：

- 不是 Markov 闭合的
- 会扭曲 attractor
- 不能保持局部谱结构
- 只是记住了局部噪声或一阶捷径

因此项目里除了预测误差，还会看：

- Markov closure
- spectral preservation
- dynamical separability
- Koopman invariance

### 5.2 为什么 `fast/slow` 只是手工近似

`fastslow` 坐标可以理解为人工构造的一组候选动力学观测函数。

它们有时有帮助，但不等于它们一定就是 Koopman eigenfunctions。

如果系统本身并没有自然 slow variable，那么它们可能：

- 改善局部拟合
- 但扭曲整体几何
- 或者破坏长期 rollout

### 5.3 为什么自动因子发现更重要

如果一组因子满足近似：

`f_{t+1} ~= A f_t`

那它们就更接近一个 Koopman invariant subspace。

这意味着：

- 更容易解释
- 更稳定
- 更适合作为 structured model 的坐标层

## 6. 当前代码里的落地

### coordinate analysis

在 [coordinate_analysis.py](/C:/Users/12345/Desktop/DynamicAlpha-Lab/src/fsrc_sindy/research/coordinate_analysis.py) 中，当前已经加入：

- `koopman_invariance_score`
- `koopman_linear_r2`
- “Best Koopman-like coordinates” 汇总

它们用于比较：

- `raw`
- `delay`
- `fastslow`
- `factor`

### factor mining

在 [property_analyzer.py](/C:/Users/12345/Desktop/DynamicAlpha-Lab/src/fsrc_sindy/factors/property_analyzer.py) 和 [miner.py](/C:/Users/12345/Desktop/DynamicAlpha-Lab/src/fsrc_sindy/factors/miner.py) 中，当前已经加入：

- 单因子 Koopman 标量诊断
- 近似 `f_{t+1} ~= lambda f_t` 评分
- 在 `identify` 模式中的权重化筛选

### research loop

在 [loop.py](/C:/Users/12345/Desktop/DynamicAlpha-Lab/src/fsrc_sindy/research/loop.py) 中，当前会：

- 汇总最优 Koopman-like 坐标
- 将 Koopman 指标纳入闭环置信度
- 和专家审核门禁一起输出

## 7. 这个理论不会自动替代专家判断

本项目不会把 “Koopman 分数高” 自动当作最终真理。

原因是：

- 代理指标仍然是近似
- 局部线性诊断不等于全局正确
- 任务不同，真正重要的结构也不同

所以项目采用的原则是：

- 先把 Koopman 视角变成可计算证据
- 再由 LLM 做候选解释
- 最后由动力学专家做审核

## 8. 在项目里怎么用这份视角

推荐做法：

1. 先跑 `run_coordinate_analysis.py`
2. 看哪种坐标 closure 更好、谱保持更好、Koopman 分数更好
3. 再跑 `run_factor_mining.py --mode identify`
4. 看候选因子的 Koopman 分数是否支持它们进入人工审核
5. 最后用 `run_research_loop.py` 统一整理证据和 review gate

## 9. 一句话总结

这个项目把“动力学因子挖掘”视为“寻找更接近 Koopman 坐标的可解释特征”的过程，而不是单纯追求更低误差的黑箱特征工程。
