# Models and Experiment Formulas

本文档根据当前仓库实现整理，目标是把“已有模型”和“已有实验流程”的数学公式统一到一处，便于后续写论文、做汇报或继续改模型时直接引用。内容主要对应以下源码：

- `src/fsrc_sindy/systems.py`
- `src/fsrc_sindy/fastslow.py`
- `src/fsrc_sindy/library.py`
- `src/fsrc_sindy/metrics.py`
- `src/fsrc_sindy/models/rc.py`
- `src/fsrc_sindy/models/ngrc.py`
- `src/fsrc_sindy/models/sindy.py`
- `src/fsrc_sindy/models/hybrid.py`
- `src/fsrc_sindy/selection.py`
- `src/fsrc_sindy/experiment.py`
- `src/fsrc_sindy/benchmarks.py`

## 1. 统一记号

设：

- 潜在真实状态为 $x(t) \in \mathbb{R}^d$
- 标量观测序列为 $y_t \in \mathbb{R}$
- 时间步长为 $\Delta t$
- 训练集均值和标准差分别为
  $$
  \mu = \frac{1}{N}\sum_{t=1}^{N} y_t,\qquad
  \sigma = \sqrt{\frac{1}{N}\sum_{t=1}^{N}(y_t-\mu)^2} + 10^{-12}
  $$
- 标准化后的序列为
  $$
  \tilde y_t = \frac{y_t - \mu}{\sigma}
  $$

代码里许多模型先在标准化域中训练与滚动预测，最后再反标准化：

$$
\hat y_t = \sigma \hat{\tilde y}_t + \mu
$$

定义截断算子：

$$
\operatorname{clip}_c(z)=\min(\max(z,-c),c)
$$

## 2. 数据生成与观测公式

### 2.1 四阶 Runge-Kutta 离散化

仓库用 RK4 从连续系统生成离散轨迹。对

$$
\dot x = f(x,t;\theta)
$$

一步积分为

$$
\begin{aligned}
k_1 &= f(x_t,t_t;\theta), \\
k_2 &= f\!\left(x_t+\frac{\Delta t}{2}k_1,\; t_t+\frac{\Delta t}{2};\theta\right), \\
k_3 &= f\!\left(x_t+\frac{\Delta t}{2}k_2,\; t_t+\frac{\Delta t}{2};\theta\right), \\
k_4 &= f(x_t+\Delta t\,k_3,\; t_t+\Delta t;\theta), \\
x_{t+1} &= x_t + \frac{\Delta t}{6}(k_1+2k_2+2k_3+k_4).
\end{aligned}
$$

如果开启过程噪声，代码随后做

$$
x_{t+1} \leftarrow x_{t+1} + \sqrt{\Delta t}\,\sigma_p \varepsilon_t,
\qquad \varepsilon_t \sim \mathcal{N}(0,I)
$$

观测噪声为

$$
y_t = h(x_t) + \sigma_o \eta_t,
\qquad \eta_t \sim \mathcal{N}(0,1).
$$

### 2.2 观测映射

代码中的观测方式为：

- `x0`:
  $$
  y_t = x_t^{(0)}
  $$
- `x1`:
  $$
  y_t = x_t^{(1)}
  $$
- `mean`:
  $$
  y_t = \frac{1}{d}\sum_{i=0}^{d-1} x_t^{(i)}
  $$
- `slow0` (仅用于双尺度 Lorenz-96):
  $$
  y_t = X_{0,t}
  $$
- `slow_mean` (代码已支持，但当前任务中未使用):
  $$
  y_t = \frac{1}{K}\sum_{k=0}^{K-1} X_{k,t}
  $$

这里上标/下标中的 `0` 对应代码里的第一个分量。

### 2.3 当前实验所用真实动力系统

#### Lorenz-63

$$
\begin{aligned}
\dot x &= \sigma(y-x), \\
\dot y &= x(\rho-z)-y, \\
\dot z &= xy-\beta z.
\end{aligned}
$$

默认参数：

$$
\sigma=10,\qquad \rho=28,\qquad \beta=\frac{8}{3}.
$$

#### Rossler

$$
\begin{aligned}
\dot x &= -y-z, \\
\dot y &= x+a y, \\
\dot z &= b + z(x-c).
\end{aligned}
$$

默认参数：

$$
a=0.2,\qquad b=0.2,\qquad c=5.7.
$$

#### Duffing

设状态为 $(x,v)$，则

$$
\begin{aligned}
\dot x &= v, \\
\dot v &= x - x^3 - \delta v + \gamma \cos(\omega t).
\end{aligned}
$$

默认参数：

$$
\delta=0.2,\qquad \gamma=0.3,\qquad \omega=1.2.
$$

#### Van der Pol

设状态为 $(x,v)$，则

$$
\begin{aligned}
\dot x &= v, \\
\dot v &= \mu(1-x^2)v - x.
\end{aligned}
$$

默认实验里常用 $\mu=8$ 或 $\mu=12$。

#### FitzHugh-Nagumo

设状态为 $(v,w)$，则

$$
\begin{aligned}
\dot v &= v - \frac{v^3}{3} - w + I, \\
\dot w &= \varepsilon (v + a - b w).
\end{aligned}
$$

默认参数：

$$
a=0.7,\qquad b=0.8,\qquad \varepsilon=0.08,\qquad I=0.5.
$$

#### Lorenz-96

对 $k=0,\dots,K-1$，

$$
\dot X_k = (X_{k+1}-X_{k-2})X_{k-1} - X_k + F.
$$

当前实验常用：

$$
K\in\{8,16,32\},\qquad F=8.
$$

#### 双尺度 Lorenz-96

慢变量 $X_k$ 与快变量 $Y_{k,j}$ 满足：

$$
\dot X_k = X_{k-1}(X_{k+1}-X_{k-2}) - X_k + F - \frac{hc}{b}\sum_{j=0}^{J-1} Y_{k,j}
$$

$$
\dot Y_{k,j} = -cb\,Y_{k,j+1}(Y_{k,j+2}-Y_{k,j-1}) - cY_{k,j} + \frac{hc}{b}X_k
$$

代码常用参数：

$$
F=10,\qquad h=1,\qquad c=10,\qquad b=10
$$

并在不同任务中改变 $K,J$。

## 3. 快慢特征编码器

所有 fast-slow 相关模型共用 `CausalFastSlowEncoder`。

### 3.1 双 EMA 快变量

令

$$
\alpha_n = \frac{2}{n+1}
$$

其中 $n=t_0$。代码先做一级 EMA：

$$
f^{(1)}_t = (1-\alpha_{t_0}) f^{(1)}_{t-1} + \alpha_{t_0}\tilde y_t
$$

再做二级 EMA：

$$
f^{(2)}_t = (1-\alpha_{t_0}) f^{(2)}_{t-1} + \alpha_{t_0} f^{(1)}_t
$$

并定义

$$
\text{fast}_t = f^{(2)}_t.
$$

### 3.2 多尺度慢变量

对每个慢时间尺度 $s_j \in \texttt{slow\_scales}$，定义

$$
\ell^{(j)}_t = (1-\alpha_{s_j}) \ell^{(j)}_{t-1} + \alpha_{s_j}\tilde y_t,
\qquad
\alpha_{s_j}=\frac{2}{s_j+1}.
$$

代码把多个慢尺度平均为

$$
\text{slow}_t = \frac{1}{J}\sum_{j=1}^{J} \ell^{(j)}_t.
$$

### 3.3 派生特征

$$
\begin{aligned}
m_t &= \text{fast}_t - \text{slow}_t, \\
\text{resid}_t &= \tilde y_t - \text{slow}_t, \\
d s_t &= \text{slow}_t - \text{slow}_{t-1}, \\
d\text{fast}_t &= \text{fast}_t - \text{fast}_{t-1}.
\end{aligned}
$$

代码里对首项使用

$$
ds_0 = 0,\qquad d\text{fast}_0 = 0.
$$

## 4. 公共回归与稀疏库公式

### 4.1 二阶多项式库

给定特征向量 $z=(z_1,\dots,z_d)$，代码构造二阶库

$$
\Theta(z) =
\big[
1,\;
z_1,\dots,z_d,\;
z_1^2,\dots,z_d^2,\;
z_1z_2,\dots,z_{d-1}z_d
\big].
$$

若使用二阶库，则特征维数为

$$
\dim(\Theta) = 1 + 2d + \frac{d(d-1)}{2}.
$$

### 4.2 Ridge 回归

仓库的线性读出统一求解

$$
w^\star = \arg\min_w \|Xw-y\|_2^2 + \lambda \|w\|_2^2
$$

闭式解为

$$
w^\star = (X^\top X + \lambda I)^{-1}X^\top y.
$$

### 4.3 STLSQ 稀疏回归

SINDy 使用顺序阈值最小二乘：

1. 先解 ridge：
   $$
   \xi^{(0)} = \arg\min_\xi \|\Theta \xi - \dot y\|_2^2 + \lambda \|\xi\|_2^2
   $$
2. 迭代执行阈值化：
   $$
   \xi_i^{(k)} = 0 \quad \text{if } |\xi_i^{(k)}| < \tau
   $$
3. 在保留下来的支撑集上重新做 ridge 拟合，重复若干轮。

## 5. 模型公式整理

### 5.1 RC / ESN 基线

### 状态更新

对 reservoir 状态 $r_t \in \mathbb{R}^{N_r}$，

$$
r_t = (1-\ell)r_{t-1} + \ell \tanh(W r_{t-1} + W_{\text{in}}\tilde y_t + b),
$$

其中：

- $\ell$ 为 `leak_rate`
- $W$ 为随机稀疏 reservoir 矩阵，并在生成后做谱半径归一化：
  $$
  W \leftarrow \rho \frac{W_0}{\rho(W_0)}
  $$
- $W_{\text{in}}$ 和 $b$ 为固定随机输入权重和偏置

### 读出层

#### `rc_raw`

$$
\phi_t^{\text{RC}} = [\,r_t,\; \tilde y_t,\; 1\,]
$$

$$
\hat{\tilde y}_{t+1} = (\phi_t^{\text{RC}})^\top w.
$$

#### `rc_fastslow_readout`

在读出端额外拼接快慢特征：

$$
\phi_t^{\text{RC-FS}} = [\,r_t,\; \tilde y_t,\; \text{fast}_t,\; \text{slow}_t,\; m_t,\; 1\,]
$$

$$
\hat{\tilde y}_{t+1} = (\phi_t^{\text{RC-FS}})^\top w.
$$

### 5.2 NGRC / NVAR 基线

### 延迟嵌入

给定延迟数 $D$ 和步长 `stride = s`，构造

$$
z_t = [\,\tilde y_t,\; \tilde y_{t-s},\; \tilde y_{t-2s},\; \dots,\; \tilde y_{t-(D-1)s}\,].
$$

### 特征与预测

#### `ngrc_raw`

$$
\phi_t^{\text{NGRC}} = \Theta\!\big(\operatorname{clip}_{c_f}(z_t)\big)
$$

$$
\hat{\tilde y}_{t+1} = \operatorname{clip}_{c_y}\!\big((\phi_t^{\text{NGRC}})^\top w\big).
$$

#### `ngrc_fastslow_readout`

$$
\phi_t^{\text{NGRC-FS}} =
\big[
\Theta\!\big(\operatorname{clip}_{c_f}(z_t)\big),\;
\text{fast}_t,\;
\text{slow}_t,\;
m_t
\big]
$$

$$
\hat{\tilde y}_{t+1} = \operatorname{clip}_{c_y}\!\big((\phi_t^{\text{NGRC-FS}})^\top w\big).
$$

其中 $c_f$ 对应 `feature_clip`，$c_y$ 对应 `y_clip`。

### 5.3 RC + NGRC 混合模型

模型名：`hybrid_rc_ngrc_fastslow`

其 reservoir 更新与 RC 相同，延迟嵌入与 NGRC 相同，最终读出做拼接：

$$
\phi_t^{\text{Hybrid}} = [\, r_t,\; \phi_t^{\text{NGRC-FS}} \,]
$$

$$
\hat{\tilde y}_{t+1} = \operatorname{clip}_{c_y}\!\big((\phi_t^{\text{Hybrid}})^\top w\big).
$$

这个模型本质上把“递归记忆”与“延迟多项式记忆”放到同一个线性读出里联合学习。

### 5.4 全可观测代理 SINDy

模型名：`sindy_full`

虽然只有标量观测，但代码先从标量序列构造代理状态：

$$
\begin{aligned}
b_t &= \text{fast}_t, \\
db_t &= b_t - b_{t-1}, \\
m_t &= \text{fast}_t - \text{slow}_t, \\
v_t &= m_t - m_{t-1}, \\
a_t &= |v_t|, \\
q_t &= \log(a_t + 10^{-6}).
\end{aligned}
$$

于是代理状态向量为

$$
s_t = [\,\tilde y_t,\; b_t,\; db_t,\; m_t,\; v_t,\; a_t,\; q_t\,].
$$

SINDy 拟合的是增量

$$
\Delta \tilde y_t = \tilde y_{t+1} - \tilde y_t
$$

并使用稀疏多项式模型

$$
\widehat{\Delta \tilde y_t} = \Theta(s_t)^\top \xi.
$$

训练时目标会被截断：

$$
\Delta \tilde y_t \leftarrow \operatorname{clip}_{c_\Delta}(\Delta \tilde y_t)
$$

预测时为

$$
\hat{\tilde y}_{t+1}
= \operatorname{clip}_{c_y}\!\left(
\tilde y_t + \operatorname{clip}_{c_\Delta}(\Theta(s_t)^\top \xi)
\right).
$$

### 5.5 慢流形 SINDy 主干

### `SlowBackboneSINDy`

首先从标准化序列提取：

$$
\text{slow}_t,\qquad ds_t=\text{slow}_t-\text{slow}_{t-1}.
$$

再定义慢变量增量目标：

$$
\Delta \text{slow}_t = \text{slow}_{t+1} - \text{slow}_t.
$$

主干模型为

$$
\widehat{\Delta \text{slow}_t}
= \Theta([\text{slow}_t, ds_t])^\top \xi_{\text{slow}}.
$$

因此

$$
\widehat{\text{slow}}_{t+1}
= \operatorname{clip}_{10}\!\left(
\text{slow}_t + \Theta([\text{slow}_t, ds_t])^\top \xi_{\text{slow}}
\right).
$$

### `slow_sindy_only`

该模型直接把慢主干预测当作输出：

$$
\hat{\tilde y}_{t+1} = \widehat{\text{slow}}_{t+1}.
$$

滚动时还使用

$$
ds_{t+1} = \widehat{\text{slow}}_{t+1} - \text{slow}_t.
$$

### 5.6 基于慢主干的残差建模

公共定义：

$$
e_t = \tilde y_t - \text{slow}_t
$$

并令慢主干给出的下一步慢变量预测为

$$
\widehat{\text{slow}}_{t+1}
$$

对应预测慢增量

$$
d\hat s_t = \widehat{\text{slow}}_{t+1} - \text{slow}_t.
$$

### A. 线性 level 残差

模型名：`slow_sindy_level_linear`，仅保留作诊断。

特征向量：

$$
\phi_t^{\text{level-lin}}
= [\,e_t,\; \text{slow}_t,\; \text{fast}_t,\; m_t,\; d\hat s_t,\; 1\,]
$$

直接回归下一步残差水平：

$$
\hat e_{t+1} = (\phi_t^{\text{level-lin}})^\top w
$$

$$
\hat{\tilde y}_{t+1} = \widehat{\text{slow}}_{t+1} + \hat e_{t+1}.
$$

### B. 线性 delta 残差

模型名：`slow_sindy_delta_linear`

训练目标先定义真实残差增量：

$$
\Delta e_t = e_{t+1}^{\text{true}} - e_t
$$

其中

$$
e_{t+1}^{\text{true}} = \tilde y_{t+1} - \widehat{\text{slow}}_{t+1}.
$$

代码训练的是截断后的目标：

$$
\Delta e_t \leftarrow \operatorname{clip}_{c_\delta}(\Delta e_t).
$$

预测时：

$$
\widehat{\Delta e_t} = (\phi_t^{\text{level-lin}})^\top w
$$

$$
\hat e_{t+1}
= \operatorname{clip}_{c_r}\!\left(
e_t + \gamma \operatorname{clip}_{c_\delta}(\widehat{\Delta e_t})
\right)
$$

$$
\hat{\tilde y}_{t+1} = \widehat{\text{slow}}_{t+1} + \hat e_{t+1}.
$$

其中 $\gamma$ 对应 `damp`。

### C. RC level 残差

模型名：`slow_sindy_level_rc`，仅保留作诊断。

reservoir 由残差驱动：

$$
r_t = (1-\ell)r_{t-1} + \ell \tanh(Wr_{t-1} + W_{\text{in}} e_t + b)
$$

特征为

$$
\phi_t^{\text{level-rc}}
= [\,r_t,\; e_t,\; \text{slow}_t,\; \text{fast}_t,\; m_t,\; d\hat s_t,\; 1\,]
$$

并回归

$$
\hat e_{t+1} = (\phi_t^{\text{level-rc}})^\top w.
$$

### D. RC delta 残差

模型名：`slow_sindy_delta_rc`

其输入特征与 `slow_sindy_level_rc` 相同，但训练目标改为残差增量：

$$
\Delta e_t = \operatorname{clip}_{c_\delta}(e_{t+1}^{\text{true}} - e_t).
$$

预测时：

$$
\widehat{\Delta e_t} = (\phi_t^{\text{delta-rc}})^\top w
$$

$$
\hat e_{t+1}
= \operatorname{clip}_{c_r}\!\left(
e_t + \gamma \operatorname{clip}_{c_\delta}(\widehat{\Delta e_t})
\right)
$$

$$
\hat{\tilde y}_{t+1}
= \widehat{\text{slow}}_{t+1} + \hat e_{t+1}.
$$

### E. NGRC delta 残差

模型名：`slow_sindy_delta_ngrc`

对残差历史做延迟嵌入：

$$
z_t^{(e)} = [\,e_t,\; e_{t-s},\; \dots,\; e_{t-(D-1)s}\,].
$$

再构造特征

$$
\phi_t^{\text{delta-ngrc}}
= \Big[
\Theta(\operatorname{clip}_{c_f}(z_t^{(e)})),
\text{slow}_t,\;
\text{fast}_t,\;
m_t,\;
d\hat s_t
\Big].
$$

训练目标：

$$
\Delta e_t = \operatorname{clip}_{c_\delta}(e_{t+1}^{\text{true}} - e_t).
$$

预测更新：

$$
\widehat{\Delta e_t} = (\phi_t^{\text{delta-ngrc}})^\top w
$$

$$
\hat e_{t+1}
= \operatorname{clip}_{c_r}\!\left(
e_t + \gamma \operatorname{clip}_{c_\delta}(\widehat{\Delta e_t})
\right)
$$

$$
\hat{\tilde y}_{t+1}
= \operatorname{clip}_{c_y}\!\left(
\widehat{\text{slow}}_{t+1} + \hat e_{t+1}
\right).
$$

### F. RC + NGRC delta 混合残差

模型名：`slow_sindy_delta_hybrid`

其 reservoir 同样由残差驱动：

$$
r_t = (1-\ell)r_{t-1} + \ell \tanh(Wr_{t-1} + W_{\text{in}} e_t + b)
$$

最终特征：

$$
\phi_t^{\text{delta-hybrid}}
= [\, r_t,\; \phi_t^{\text{delta-ngrc}} \,]
$$

训练目标与更新公式与 `slow_sindy_delta_ngrc` 相同，只是把 NGRC 残差特征替换为 RC+NGRC 联合特征。

## 6. 实验流程公式

### 6.1 数据切分

仿真得到的观测序列记为 $\{y_t\}_{t=1}^{T}$，其中

$$
T = n_{\text{train}} + n_{\text{val}} + n_{\text{test}}.
$$

代码切分为

$$
\begin{aligned}
\mathcal{D}_{\text{train}} &= \{y_1,\dots,y_{n_{\text{train}}}\}, \\
\mathcal{D}_{\text{val}} &= \{y_{n_{\text{train}}+1},\dots,y_{n_{\text{train}}+n_{\text{val}}}\}, \\
\mathcal{D}_{\text{test}} &= \{y_{n_{\text{train}}+n_{\text{val}}+1},\dots,y_T\}.
\end{aligned}
$$

### 6.2 上下文长度

模型选择时使用

$$
L_{\text{ctx}} = \max\big(200,\; 4 \max(\mathcal{H}_{\text{sel}})\big)
$$

其中默认选择 horizon 集合是

$$
\mathcal{H}_{\text{sel}} = \{10, 50\}.
$$

因此默认

$$
L_{\text{ctx}} = 200.
$$

### 6.3 验证集模型选择

对每个候选配置，验证时构造

$$
y^{\text{ctx}}_{\text{val}} =
\big[
y^{\text{train}}_{n_{\text{train}}-L_{\text{ctx}}+1 : n_{\text{train}}},
\;
y^{\text{val}}_1
\big]
$$

未来目标为

$$
y^{\text{future}}_{\text{val}} =
\big[
y^{\text{val}}_2,\dots,y^{\text{val}}_{1+\max(\mathcal{H}_{\text{sel}})}
\big].
$$

记验证尺度

$$
s_y = \operatorname{std}(y_{\text{train}}) + 10^{-12}.
$$

验证分数为

$$
\text{Score}
= \sum_{H \in \mathcal{H}_{\text{sel}}} \operatorname{NRMSE}@H
+ \text{Penalty}_{\text{diverge}}.
$$

其中

$$
\operatorname{NRMSE}@H
= \frac{\operatorname{RMSE}(\hat y_{1:H}, y_{1:H})}{\operatorname{std}(y_{1:H}) + 10^{-12}}.
$$

发散惩罚按代码可写为：

$$
\text{Penalty}_{\text{diverge}}=
\begin{cases}
10^6, & \text{若 rollout 非有限值} \\
10^6, & \text{若最大绝对值非有限} \\
1000 + 0.1 M, & \text{若 } M > 10 s_y \\
0, & \text{其他情况}
\end{cases}
$$

其中

$$
M = \max_t |\hat y_t|.
$$

### 6.4 测试阶段开放滚动

测试时上下文为

$$
y^{\text{ctx}}_{\text{test}} =
\big[
y^{\text{val}}_{n_{\text{val}}-L_{\text{ctx}}+1:n_{\text{val}}},
\;
y^{\text{test}}_1
\big]
$$

未来目标为

$$
y^{\text{future}}_{\text{test}} =
\big[
y^{\text{test}}_2,\dots,y^{\text{test}}_{1+\max(\max(\mathcal{H}_{\text{eval}}), H_{\text{stat}})}
\big].
$$

默认评估 horizon 为

$$
\mathcal{H}_{\text{eval}} = \{1, 5, 10, 20, 50, 100\}.
$$

### 6.5 一步预测与多步滚动

### 一步预测

一步预测使用 teacher forcing，即每一步都喂入真实历史，计算

$$
\hat y_{t+1|t}.
$$

### 多步滚动

多步滚动使用模型自己的输出递推：

$$
\hat y_{t+1}, \hat y_{t+2}, \dots, \hat y_{t+H}.
$$

这也是 `rollout()` 的含义。

## 7. 评估指标公式

### 7.1 点预测误差

对真实序列 $y$ 和预测序列 $\hat y$：

$$
\operatorname{MSE}(y,\hat y) = \frac{1}{T}\sum_{t=1}^{T}(y_t-\hat y_t)^2
$$

$$
\operatorname{RMSE}(y,\hat y)=\sqrt{\operatorname{MSE}(y,\hat y)}
$$

$$
\operatorname{MAE}(y,\hat y)=\frac{1}{T}\sum_{t=1}^{T}|y_t-\hat y_t|
$$

$$
R^2 = 1 - \frac{\sum_t (y_t-\hat y_t)^2}{\sum_t (y_t-\bar y)^2}
$$

标准化 RMSE 为

$$
\operatorname{NRMSE}(y,\hat y)=\frac{\operatorname{RMSE}(y,\hat y)}{\operatorname{std}(y)+10^{-12}}.
$$

### 7.2 自相关误差

对中心化序列 $\bar x_t = x_t - \frac{1}{T}\sum_s x_s$，代码采用

$$
\operatorname{ACF}_x(k) =
\frac{\sum_{t=1}^{T-k}\bar x_t \bar x_{t+k}}
{\sum_{t=1}^{T}\bar x_t^2 + 10^{-12}},
\qquad k=0,1,\dots,K.
$$

然后比较预测与真实 ACF 的 RMSE：

$$
\operatorname{acf\_rmse}
= \operatorname{RMSE}\big(\operatorname{ACF}_{y^{\text{true}}}, \operatorname{ACF}_{y^{\text{pred}}}\big).
$$

### 7.3 功率谱误差

代码使用归一化功率谱：

$$
\operatorname{PSD}_x(\omega)
=
\frac{|\operatorname{rFFT}(\bar x)(\omega)|^2}
{\sum_{\omega'} |\operatorname{rFFT}(\bar x)(\omega')|^2 + 10^{-12}}.
$$

再计算

$$
\operatorname{psd\_rmse}
= \operatorname{RMSE}\big(\operatorname{PSD}_{y^{\text{true}}}, \operatorname{PSD}_{y^{\text{pred}}}\big).
$$

### 7.4 分布统计偏差

还会报告

$$
\text{mean\_gap} = |\mathbb{E}[y^{\text{true}}] - \mathbb{E}[y^{\text{pred}}]|
$$

$$
\text{std\_gap} = |\operatorname{std}(y^{\text{true}}) - \operatorname{std}(y^{\text{pred}})|.
$$

## 8. 参数规模定义

仓库中用了三种“规模”概念：

- `effective_dim`：模型动态特征维度
- `trained_params`：真正通过回归或稀疏回归拟合的参数数
- `total_params`：包含固定 reservoir 权重在内的总参数量

以主要模型为例：

- `rc_raw`
  $$
  \text{trained\_params} = N_r + 2
  $$
  $$
  \text{total\_params} = N_r^2 + 3N_r + 2
  $$
- `rc_fastslow_readout`
  $$
  \text{trained\_params} = N_r + 5
  $$
  $$
  \text{total\_params} = N_r^2 + 3N_r + 5
  $$
- `ngrc_*`
  $$
  \text{trained\_params} = \text{total\_params} = \dim(\phi)
  $$
- `hybrid_rc_ngrc_fastslow`
  $$
  \text{trained\_params} = N_r + \dim(\phi^{\text{NGRC-FS}})
  $$

结构化残差模型的参数量在上述基础上再加上慢主干 SINDy 的稀疏系数数目。

## 9. 当前 benchmark 套件总览

### 9.1 smoke

| task | system | dt | train/val/test | noise $(\sigma_p,\sigma_o)$ | obs |
| --- | --- | ---: | --- | --- | --- |
| `lorenz63_smoke` | Lorenz-63 | 0.01 | 1200 / 500 / 500 | (0, 0) | `x0` |
| `vanderpol_smoke` | Van der Pol, $\mu=8$ | 0.01 | 1200 / 500 / 500 | (0, 0) | `x0` |

### 9.2 common

| task | system | dt | train/val/test | noise $(\sigma_p,\sigma_o)$ | obs |
| --- | --- | ---: | --- | --- | --- |
| `lorenz63_clean` | Lorenz-63 | 0.01 | 4000 / 1500 / 1500 | (0, 0) | `x0` |
| `lorenz63_noisy` | Lorenz-63 | 0.01 | 4000 / 1500 / 1500 | (0.15, 0.02) | `x0` |
| `rossler_clean` | Rossler | 0.05 | 5000 / 2000 / 2000 | (0, 0) | `x0` |
| `duffing_noisy` | Duffing | 0.02 | 5000 / 2000 / 2000 | (0.02, 0.01) | `x0` |
| `vanderpol_stiff` | Van der Pol, $\mu=8$ | 0.01 | 4000 / 1500 / 1500 | (0, 0) | `x0` |

### 9.3 hard

| task | system | dt | train/val/test | noise $(\sigma_p,\sigma_o)$ | obs |
| --- | --- | ---: | --- | --- | --- |
| `fitzhugh_nagumo_noisy` | FitzHugh-Nagumo | 0.05 | 5000 / 2000 / 2000 | (0.02, 0.01) | `x0` |
| `lorenz96_partial` | Lorenz-96, $K=8,F=8$ | 0.01 | 5000 / 2000 / 2000 | (0, 0) | `x0` |
| `lorenz96_partial_noisy` | Lorenz-96, $K=8,F=8$ | 0.01 | 5000 / 2000 / 2000 | (0.05, 0.02) | `x0` |
| `lorenz96_twoscale_partial` | Two-scale Lorenz-96, $K=8,J=4$ | 0.005 | 6000 / 2500 / 2500 | (0, 0) | `slow0` |
| `vanderpol_shorttrain` | Van der Pol, $\mu=12$ | 0.01 | 1200 / 1200 / 1200 | (0.01, 0.01) | `x0` |

### 9.4 highdim

| task | system | dt | train/val/test | noise $(\sigma_p,\sigma_o)$ | obs |
| --- | --- | ---: | --- | --- | --- |
| `lorenz96_k16_partial` | Lorenz-96, $K=16,F=8$ | 0.01 | 5000 / 2000 / 2000 | (0, 0) | `x0` |
| `lorenz96_k32_partial` | Lorenz-96, $K=32,F=8$ | 0.01 | 5000 / 2000 / 2000 | (0, 0) | `x0` |
| `lorenz96_k32_partial_noisy` | Lorenz-96, $K=32,F=8$ | 0.01 | 5000 / 2000 / 2000 | (0.04, 0.02) | `x0` |
| `lorenz96_twoscale_k12_j6` | Two-scale Lorenz-96, $K=12,J=6$ | 0.005 | 6000 / 2500 / 2500 | (0, 0) | `slow0` |
| `lorenz96_twoscale_k16_j4_noisy` | Two-scale Lorenz-96, $K=16,J=4$ | 0.005 | 6000 / 2500 / 2500 | (0.03, 0.02) | `slow0` |

## 10. 当前默认参赛模型名单

默认 benchmark 会比较：

- `rc_raw`
- `rc_fastslow_readout`
- `ngrc_raw`
- `ngrc_fastslow_readout`
- `hybrid_rc_ngrc_fastslow`
- `sindy_full`
- `slow_sindy_only`
- `slow_sindy_delta_rc`
- `slow_sindy_delta_ngrc`

此外仓库还保留以下模型用于扩展或诊断：

- `slow_sindy_delta_linear`
- `slow_sindy_level_linear`
- `slow_sindy_level_rc`
- `slow_sindy_delta_hybrid`

## 11. 一句话总结

这个仓库的核心思路可以概括为：

$$
\text{标量观测} \longrightarrow \text{快慢分解} \longrightarrow
\begin{cases}
\text{RC 记忆} \\
\text{NGRC 延迟多项式} \\
\text{SINDy 慢流形先验}
\end{cases}
\longrightarrow \text{一步预测或多步滚动}
$$

其中最有代表性的结构化模型是：

$$
\hat y_{t+1}
=
\underbrace{\widehat{\text{slow}}_{t+1}}_{\text{SINDy backbone}}
+
\underbrace{\hat e_{t+1}}_{\text{RC / NGRC / Hybrid residual closure}}.
$$
