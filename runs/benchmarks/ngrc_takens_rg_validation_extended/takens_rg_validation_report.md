# Takens-RG Validation Report

This report validates the Takens-RG NGRC story with multi-seed matched controls, conditioning-form ablations, regime-boundary sweeps, delay-sufficiency sweeps, and coordinate diagnostics.

## Ablation Equations

The compared RC/NGRC ablations can be read as follows:

| variant                 | family   | state_backbone                                                                      | readout_term                                                                                                      | extra_features                                                                                  | mechanistic_role                                                                                         |
|:------------------------|:---------|:------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------|
| ngrc_fastslow_readout   | ngrc     | $z_t=[y_t,y_{t-\tau},\ldots,y_{t-(d-1)\tau}]^\top,\ \phi(z_t)=\mathrm{Poly}_2(z_t)$ | $\hat y_{t+1}=W_\phi^\top \phi(z_t)+u^\top sf_t$                                                                  | $sf_t=[f_t,s_t,m_t]^\top$                                                                       | Delay backbone plus explicit fast/slow readout features.                                                 |
| ngrc_raw                | ngrc     | $z_t=[y_t,y_{t-\tau},\ldots,y_{t-(d-1)\tau}]^\top,\ \phi(z_t)=\mathrm{Poly}_2(z_t)$ | $\hat y_{t+1}=W_\phi^\top \phi(z_t)$                                                                              | none                                                                                            | Pure Takens/NGRC baseline; all predictive structure must be absorbed by the delay library.               |
| ngrc_rg_readout         | ngrc     | $z_t=[y_t,y_{t-\tau},\ldots,y_{t-(d-1)\tau}]^\top,\ \phi(z_t)=\mathrm{Poly}_2(z_t)$ | $\hat y_{t+1}=W_\phi^\top \phi(z_t)+b^\top \rho_t$                                                                | $\rho_t=[\rho_t^{op},\rho_t^{ctrl},\rho_t^{noise},\rho_t^{\beta},\rho_t^{cg},\rho_t^{cb}]^\top$ | Treats RG as an additive macro readout bias on top of delay features.                                    |
| ngrc_sf_rg_gated        | ngrc     | $z_t=[y_t,y_{t-\tau},\ldots,y_{t-(d-1)\tau}]^\top,\ \phi(z_t)=\mathrm{Poly}_2(z_t)$ | $\hat y_{t+1}=W_\phi^\top \phi(z_t)+u^\top sf_t+b^\top \rho_t+c^\top g(sf_t,\rho_t)$                              | $g(sf_t,\rho_t)=[m_t\rho_t^{op},\,m_t\rho_t^{\beta},\,s_t\rho_t^{cb}]^\top$                     | Lets RG modulate explicit fast/slow closure channels through a sparse mechanistic gate.                  |
| ngrc_takens_rg_additive | ngrc     | $z_t=[y_t,y_{t-\tau},\ldots,y_{t-(d-1)\tau}]^\top,\ \phi(z_t)=\mathrm{Poly}_2(z_t)$ | $\widehat{\Delta y}_t=w^\top \phi(z_t)+b^\top \rho_t,\ \hat y_{t+1}=y_t+\widehat{\Delta y}_t$                     | $\rho_t=[\rho_t^{op},\rho_t^{\beta},\rho_t^{cb}]^\top$                                          | Tests whether RG is only a macro correction term, without operator modulation.                           |
| ngrc_takens_rg_true     | ngrc     | $z_t=[y_t,y_{t-\tau},\ldots,y_{t-(d-1)\tau}]^\top,\ \phi(z_t)=\mathrm{Poly}_2(z_t)$ | $\widehat{\Delta y}_t=w^\top \phi(z_t)+b^\top \rho_t+s(z_t)^\top A\rho_t,\ \hat y_{t+1}=y_t+\widehat{\Delta y}_t$ | $s(z_t)=[\bar z_t,\ y_t-y_{t-\tau},\ y_t-2y_{t-\tau}+y_{t-2\tau},\ \mathrm{Var}(z_t)]^\top$     | Implements the main theory: RG conditions the local delay-space operator instead of replacing the state. |
| rc_fastslow_readout     | rc       | $r_t=(1-\lambda)r_{t-1}+\lambda\tanh(Wr_{t-1}+W_{in}y_t+b)$                         | $\hat y_{t+1}=w_r^\top r_t+w_y y_t+u^\top sf_t+c$                                                                 | $sf_t=[f_t,s_t,m_t]^\top$                                                                       | Adds explicit fast/slow closure variables to the RC readout.                                             |
| rc_raw                  | rc       | $r_t=(1-\lambda)r_{t-1}+\lambda\tanh(Wr_{t-1}+W_{in}y_t+b)$                         | $\hat y_{t+1}=w_r^\top r_t+w_y y_t+c$                                                                             | none                                                                                            | Pure reservoir baseline; all latent structure must be absorbed by the reservoir state.                   |
| rc_rg_readout           | rc       | $r_t=(1-\lambda)r_{t-1}+\lambda\tanh(Wr_{t-1}+W_{in}y_t+b)$                         | $\hat y_{t+1}=w_r^\top r_t+w_y y_t+b_\rho^\top \rho_t+c$                                                          | $\rho_t=[\rho_t^{op},\rho_t^{ctrl},\rho_t^{noise},\rho_t^{\beta},\rho_t^{cg},\rho_t^{cb}]^\top$ | Treats RG as a direct macro readout bias on top of the reservoir memory.                                 |
| rc_sf_rg_gated          | rc       | $r_t=(1-\lambda)r_{t-1}+\lambda\tanh(Wr_{t-1}+W_{in}y_t+b)$                         | $\hat y_{t+1}=w_r^\top r_t+w_y y_t+u^\top sf_t+b_\rho^\top \rho_t+c_g^\top g(sf_t,\rho_t)+c$                      | $g(sf_t,\rho_t)=[m_t\rho_t^{op},\,m_t\rho_t^{\beta},\,s_t\rho_t^{cb}]^\top$                     | Sparse SF-RG gate: RG modulates only a few mechanistically chosen fast/slow channels.                    |

## Experimental System Equations

These are the true benchmark dynamics and observation equations used in the current validation tasks:

| system            | state_equation                                                                                                                                            | observation_equation                                                                        | task_variants                                                             | mechanistic_axis                                                                  |
|:------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
| fitzhugh_nagumo   | $\dot v=v-\frac{v^3}{3}-h_{sf}w+I,\ \dot w=\epsilon(h_{fs}v+a-bw)$                                                                                        | $y_t=v(t)+\eta_t$                                                                           | classic_clean, classic_noisy, classic_volclustered                        | excitable fast-slow dynamics with explicit slow-to-fast and fast-to-slow coupling |
| hindmarsh_rose    | $\dot v=y-av^3+bv^2-h_{sf}z+I,\ \dot y=c-dv^2-y,\ \dot z=r(h_{fs}s(v-x_r)-z)$                                                                             | $y^{obs}_t=v(t)+\eta_t$                                                                     | bursting_clean, bursting_noisy, bursting_volclustered                     | bursting with slow adaptation controlling fast spiking episodes                   |
| lorenz96_twoscale | $\dot X_k=-X_{k-1}(X_{k-2}-X_{k+1})-X_k+F-\frac{h_{fs}c}{b}\sum_j Y_{k,j},\ \dot Y_{k,j}=-cbY_{k,j+1}(Y_{k,j+2}-Y_{k,j-1})-cY_{k,j}+\frac{h_{sf}c}{b}X_k$ | $y_t \in \{X_0,\ \sum_i \alpha_i X_i,\ \sum_i \alpha_i X_i+\sum_j \beta_j Y_{0,j}\}+\eta_t$ | obs_slow0, obs_sparse_slowproj, obs_mixed_projection, gate_s2f_*, noise_* | multiscale closure, observability geometry, and coupling-strength sweeps          |
| vanderpol         | $\dot x_1=x_2,\ \dot x_2=\mu(1-x_1^2)x_2-x_1$                                                                                                             | $y_t=x_1(t)+\eta_t$                                                                         | relaxation_clean, relaxation_noisy, relaxation_volclustered               | relaxation oscillation with a slow manifold and fast jump segments                |

## Evaluation Metrics

The validation does not only use `rmse@50`; the experiment logs predictive, distributional, and coordinate-dynamical metrics:

| metric_family       | metric                     | formula_or_definition                                                      | why_it_matters                                                                    |
|:--------------------|:---------------------------|:---------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
| coordinate_dynamics | koopman_invariance_score   | normalized error of the best linear one-step operator on the coordinate    | Larger is better; indicates better approximate Koopman invariance.                |
| coordinate_dynamics | markov_gain_ratio          | improvement from adding one more lag to the coordinate transition model    | Smaller is better; near-zero means the coordinate is closer to Markovian closure. |
| coordinate_dynamics | spectral_radius_rmse       | $\mathrm{RMSE}$ between local true and coordinate-implied spectral radii   | Smaller is better; checks whether local growth/decay structure is preserved.      |
| distributional      | acf_rmse                   | $\mathrm{RMSE}(\mathrm{ACF}(y),\mathrm{ACF}(\hat y))$                      | Checks whether temporal dependence is reproduced, not just point accuracy.        |
| distributional      | psd_rmse                   | $\mathrm{RMSE}(\mathrm{PSD}(y),\mathrm{PSD}(\hat y))$                      | Checks whether oscillatory content and spectral energy are preserved.             |
| predictive_local    | one_step_rmse              | $\sqrt{\frac{1}{N}\sum_t (y_{t+1}-\hat y_{t+1|t})^2}$                      | Tests local one-step fit quality.                                                 |
| predictive_rollout  | rmse@10, rmse@50, rmse@100 | $\mathrm{RMSE}@H=\sqrt{\frac{1}{H}\sum_{h=1}^{H}(y_{t+h}-\hat y_{t+h})^2}$ | Separates short-, mid-, and long-horizon rollout quality.                         |

## Overall Ablation Overview

The table below merges the main conditioning study, the boundary-only extra tasks, and the robustness validation tasks into one task-by-task view using `rmse@50_mean` as the primary sort metric.

| task                                      | task_group                   |   ngrc_raw |   ngrc_rg_readout |   ngrc_takens_rg_additive |   ngrc_takens_rg_true |   ngrc_sf_rg_gated | winner                  |   winner_rmse@50_mean |
|:------------------------------------------|:-----------------------------|-----------:|------------------:|--------------------------:|----------------------:|-------------------:|:------------------------|----------------------:|
| fitzhugh_nagumo_classic_clean             | classic_fastslow             | 0.00500111 |         0.0660049 |                 0.0174545 |             0.0526482 |         0.00573935 | ngrc_raw                |            0.00500111 |
| fitzhugh_nagumo_classic_noisy             | classic_fastslow             | 0.0962377  |         0.0910248 |                 0.0946832 |             0.104601  |         0.0763619  | ngrc_sf_rg_gated        |            0.0763619  |
| hindmarsh_rose_bursting_clean             | classic_fastslow             | 0.232695   |         0.107908  |                 0.133886  |             0.0136519 |         0.0663336  | ngrc_takens_rg_true     |            0.0136519  |
| hindmarsh_rose_bursting_noisy             | classic_fastslow             | 0.174603   |         0.17346   |                 0.159417  |             0.112355  |         0.158509   | ngrc_takens_rg_true     |            0.112355   |
| vanderpol_relaxation_clean                | classic_fastslow             | 0.276932   |         0.381185  |                 1.50665   |             0.108042  |         0.60152    | ngrc_takens_rg_true     |            0.108042   |
| vanderpol_relaxation_noisy                | classic_fastslow             | 0.0481144  |         0.0262099 |                 0.0216488 |             0.0215328 |         0.0278728  | ngrc_takens_rg_true     |            0.0215328  |
| fitzhugh_nagumo_classic_volclustered      | classic_fastslow_finance     | 0.137225   |         0.201308  |                 0.135712  |             0.200397  |         0.215287   | ngrc_takens_rg_additive |            0.135712   |
| hindmarsh_rose_bursting_volclustered      | classic_fastslow_finance     | 0.0771236  |         0.11856   |                 0.080611  |             0.0511485 |         0.052024   | ngrc_takens_rg_true     |            0.0511485  |
| vanderpol_relaxation_volclustered         | classic_fastslow_finance     | 0.532564   |         0.505159  |                 0.521406  |             0.556491  |         0.456947   | ngrc_sf_rg_gated        |            0.456947   |
| lorenz96_twoscale_gate_s2f_0p0            | highdim_multiscale_mechanism | 1.46848    |         1.6346    |                 1.06159   |             0.975204  |         0.459837   | ngrc_sf_rg_gated        |            0.459837   |
| lorenz96_twoscale_gate_s2f_0p4            | highdim_multiscale_mechanism | 2.79017    |         1.89573   |               nan         |             2.11148   |         2.38331    | ngrc_rg_readout         |            1.89573    |
| lorenz96_twoscale_gate_s2f_0p8            | highdim_multiscale_mechanism | 0.636198   |         0.691981  |                 0.602873  |             0.649348  |         0.0324106  | ngrc_sf_rg_gated        |            0.0324106  |
| lorenz96_twoscale_gate_s2f_1p2            | highdim_multiscale_mechanism | 0.859456   |         0.289167  |               nan         |             0.650472  |         0.475749   | ngrc_rg_readout         |            0.289167   |
| lorenz96_twoscale_gate_s2f_1p6            | highdim_multiscale_mechanism | 0.0251629  |         0.0798148 |               nan         |             0.0986245 |         0.0970608  | ngrc_raw                |            0.0251629  |
| lorenz96_twoscale_noise_homoskedastic     | highdim_multiscale_mechanism | 6.91632    |         6.92006   |                 5.22185   |             5.76153   |         6.83785    | ngrc_takens_rg_additive |            5.22185    |
| lorenz96_twoscale_noise_matched_clustered | highdim_multiscale_mechanism | 5.53372    |         7.71595   |                 6.92851   |            12.3693    |         6.94185    | ngrc_raw                |            5.53372    |
| lorenz96_twoscale_obs_mixed_projection    | highdim_multiscale_mechanism | 0.73855    |         0.823314  |                 0.707837  |             0.683103  |         0.594634   | ngrc_sf_rg_gated        |            0.594634   |
| lorenz96_twoscale_obs_slow0               | highdim_multiscale_mechanism | 0.859456   |         0.289167  |                 0.701668  |             0.650472  |         0.475749   | ngrc_rg_readout         |            0.289167   |
| lorenz96_twoscale_obs_sparse_slowproj     | highdim_multiscale_mechanism | 1.45541    |         1.39257   |               nan         |             0.835671  |         0.896542   | ngrc_takens_rg_true     |            0.835671   |

Per-task multi-metric view for the same merged ablation set:

| task                                      | task_group                   | variant                 |   one_step_rmse_mean |   rmse@10_mean |   rmse@50_mean |   rmse@100_mean |   acf_rmse_mean |   psd_rmse_mean |
|:------------------------------------------|:-----------------------------|:------------------------|---------------------:|---------------:|---------------:|----------------:|----------------:|----------------:|
| fitzhugh_nagumo_classic_clean             | classic_fastslow             | ngrc_raw                |          5.61833e-05 |    0.000169462 |     0.00500111 |       2.03483   |      0.937189   |      0.0712191  |
| fitzhugh_nagumo_classic_clean             | classic_fastslow             | ngrc_rg_readout         |          1.81861e-05 |    0.000163707 |     0.0660049  |       5.51162   |      0.929536   |      0.0695115  |
| fitzhugh_nagumo_classic_clean             | classic_fastslow             | ngrc_sf_rg_gated        |          1.54682e-05 |    1.29625e-05 |     0.00573935 |       2.22584   |      0.962851   |      0.0715106  |
| fitzhugh_nagumo_classic_clean             | classic_fastslow             | ngrc_takens_rg_additive |          4.52171e-05 |    0.000356192 |     0.0174545  |       2.06437   |      0.9365     |      0.0722367  |
| fitzhugh_nagumo_classic_clean             | classic_fastslow             | ngrc_takens_rg_true     |          3.35871e-05 |    0.000561137 |     0.0526482  |       3.90058   |      0.941606   |      0.0732543  |
| fitzhugh_nagumo_classic_noisy             | classic_fastslow             | ngrc_raw                |          0.0149376   |    0.0232298   |     0.0962377  |       0.447278  |      0.325264   |      0.0329528  |
| fitzhugh_nagumo_classic_noisy             | classic_fastslow             | ngrc_rg_readout         |          0.0148978   |    0.0215982   |     0.0910248  |       0.534445  |      0.0642308  |      0.0161231  |
| fitzhugh_nagumo_classic_noisy             | classic_fastslow             | ngrc_sf_rg_gated        |          0.0145402   |    0.0240092   |     0.0763619  |       0.301699  |      0.437125   |      0.0341365  |
| fitzhugh_nagumo_classic_noisy             | classic_fastslow             | ngrc_takens_rg_additive |          0.0149402   |    0.0241128   |     0.0946832  |       0.501062  |      0.0655082  |      0.0185415  |
| fitzhugh_nagumo_classic_noisy             | classic_fastslow             | ngrc_takens_rg_true     |          0.0145391   |    0.0228535   |     0.104601   |       0.568517  |      0.112299   |      0.0228104  |
| hindmarsh_rose_bursting_clean             | classic_fastslow             | ngrc_raw                |          5.02807e-05 |    0.00218866  |     0.232695   |       3.5769    |      1.09565    |      0.0726855  |
| hindmarsh_rose_bursting_clean             | classic_fastslow             | ngrc_rg_readout         |          2.59857e-05 |    0.00423978  |     0.107908   |       0.489021  |      0.0707107  |      0.0536232  |
| hindmarsh_rose_bursting_clean             | classic_fastslow             | ngrc_sf_rg_gated        |          0.000110573 |    0.00370627  |     0.0663336  |       0.209216  |      1.05477    |      0.0559254  |
| hindmarsh_rose_bursting_clean             | classic_fastslow             | ngrc_takens_rg_additive |          4.08036e-05 |    0.00277295  |     0.133886   |       1.26731   |      1.07785    |      0.0621471  |
| hindmarsh_rose_bursting_clean             | classic_fastslow             | ngrc_takens_rg_true     |          4.258e-05   |    0.000651402 |     0.0136519  |       0.0334858 |      0.113314   |      0.00919787 |
| hindmarsh_rose_bursting_noisy             | classic_fastslow             | ngrc_raw                |          0.0129856   |    0.0256607   |     0.174603   |       0.50877   |      0.252458   |      0.0292286  |
| hindmarsh_rose_bursting_noisy             | classic_fastslow             | ngrc_rg_readout         |          0.0135573   |    0.0320049   |     0.17346    |       0.572885  |      0.200133   |      0.0262363  |
| hindmarsh_rose_bursting_noisy             | classic_fastslow             | ngrc_sf_rg_gated        |          0.0127586   |    0.0231503   |     0.158509   |       0.576106  |      0.186741   |      0.020422   |
| hindmarsh_rose_bursting_noisy             | classic_fastslow             | ngrc_takens_rg_additive |          0.0127464   |    0.0259251   |     0.159417   |       0.485968  |      0.248752   |      0.0430841  |
| hindmarsh_rose_bursting_noisy             | classic_fastslow             | ngrc_takens_rg_true     |          0.0167162   |    0.0244559   |     0.112355   |       0.345773  |      0.234267   |      0.0227385  |
| vanderpol_relaxation_clean                | classic_fastslow             | ngrc_raw                |          0.00024932  |    0.00029444  |     0.276932   |       8.83144   |      1.07229    |      0.0754198  |
| vanderpol_relaxation_clean                | classic_fastslow             | ngrc_rg_readout         |          0.000249011 |    0.00128458  |     0.381185   |       9.02877   |      1.0755     |      0.0757613  |
| vanderpol_relaxation_clean                | classic_fastslow             | ngrc_sf_rg_gated        |          0.000243285 |    0.00052613  |     0.60152    |       9.52837   |      1.07895    |      0.0821502  |
| vanderpol_relaxation_clean                | classic_fastslow             | ngrc_takens_rg_additive |          0.000119988 |    0.0008298   |     1.50665    |       9.76242   |      1.06382    |      0.0761276  |
| vanderpol_relaxation_clean                | classic_fastslow             | ngrc_takens_rg_true     |          0.00023042  |    9.03555e-05 |     0.108042   |       7.22613   |      1.04827    |      0.0778859  |
| vanderpol_relaxation_noisy                | classic_fastslow             | ngrc_raw                |          0.0154784   |    0.0164597   |     0.0481144  |       0.121356  |      0.0292482  |      0.00618834 |
| vanderpol_relaxation_noisy                | classic_fastslow             | ngrc_rg_readout         |          0.0140671   |    0.0111121   |     0.0262099  |       0.046542  |      0.0274486  |      0.00711544 |
| vanderpol_relaxation_noisy                | classic_fastslow             | ngrc_sf_rg_gated        |          0.013868    |    0.0118343   |     0.0278728  |       0.0506549 |      0.211069   |      0.0258077  |
| vanderpol_relaxation_noisy                | classic_fastslow             | ngrc_takens_rg_additive |          0.0147855   |    0.0118581   |     0.0216488  |       0.0324906 |      0.0206529  |      0.00419894 |
| vanderpol_relaxation_noisy                | classic_fastslow             | ngrc_takens_rg_true     |          0.0140651   |    0.0111724   |     0.0215328  |       0.0332546 |      0.0205817  |      0.00424696 |
| fitzhugh_nagumo_classic_volclustered      | classic_fastslow_finance     | ngrc_raw                |          0.0454195   |    0.041187    |     0.137225   |       0.483331  |      0.0376923  |      0.00474414 |
| fitzhugh_nagumo_classic_volclustered      | classic_fastslow_finance     | ngrc_rg_readout         |          0.0457829   |    0.0447013   |     0.201308   |       0.667302  |      0.0480095  |      0.0122253  |
| fitzhugh_nagumo_classic_volclustered      | classic_fastslow_finance     | ngrc_sf_rg_gated        |          0.04499     |    0.0433844   |     0.215287   |       0.684682  |      0.57808    |      0.0470108  |
| fitzhugh_nagumo_classic_volclustered      | classic_fastslow_finance     | ngrc_takens_rg_additive |          0.0456965   |    0.0431002   |     0.135712   |       0.485863  |      0.0279436  |      0.00954576 |
| fitzhugh_nagumo_classic_volclustered      | classic_fastslow_finance     | ngrc_takens_rg_true     |          0.0449738   |    0.0487453   |     0.200397   |       0.641251  |      0.223066   |      0.0253886  |
| hindmarsh_rose_bursting_volclustered      | classic_fastslow_finance     | ngrc_raw                |          0.0536314   |    0.0320227   |     0.0771236  |       0.225996  |      0.271134   |      0.0382014  |
| hindmarsh_rose_bursting_volclustered      | classic_fastslow_finance     | ngrc_rg_readout         |          0.054041    |    0.0418415   |     0.11856    |       0.45417   |      0.296172   |      0.0233715  |
| hindmarsh_rose_bursting_volclustered      | classic_fastslow_finance     | ngrc_sf_rg_gated        |          0.0537616   |    0.0346456   |     0.052024   |       0.0940554 |      0.287083   |      0.023727   |
| hindmarsh_rose_bursting_volclustered      | classic_fastslow_finance     | ngrc_takens_rg_additive |          0.0535103   |    0.0367034   |     0.080611   |       0.228017  |      0.280814   |      0.029047   |
| hindmarsh_rose_bursting_volclustered      | classic_fastslow_finance     | ngrc_takens_rg_true     |          0.0543923   |    0.0347273   |     0.0511485  |       0.105499  |      0.259046   |      0.019698   |
| vanderpol_relaxation_volclustered         | classic_fastslow_finance     | ngrc_raw                |          0.0653366   |    0.0635975   |     0.532564   |       0.865406  |      0.171541   |      0.01457    |
| vanderpol_relaxation_volclustered         | classic_fastslow_finance     | ngrc_rg_readout         |          0.0625749   |    0.0658636   |     0.505159   |       0.727023  |      0.29205    |      0.026436   |
| vanderpol_relaxation_volclustered         | classic_fastslow_finance     | ngrc_sf_rg_gated        |          0.0641326   |    0.0560928   |     0.456947   |       0.675485  |      0.175156   |      0.0187549  |
| vanderpol_relaxation_volclustered         | classic_fastslow_finance     | ngrc_takens_rg_additive |          0.0682239   |    0.0626406   |     0.521406   |       0.799209  |      0.202756   |      0.0141653  |
| vanderpol_relaxation_volclustered         | classic_fastslow_finance     | ngrc_takens_rg_true     |          0.0659318   |    0.0666363   |     0.556491   |       0.857983  |      0.17186    |      0.0199282  |
| lorenz96_twoscale_gate_s2f_0p0            | highdim_multiscale_mechanism | ngrc_raw                |          0.000320336 |    0.0151841   |     1.46848    |       4.4385    |      0.0386328  |      0.0252107  |
| lorenz96_twoscale_gate_s2f_0p0            | highdim_multiscale_mechanism | ngrc_rg_readout         |          0.000335965 |    0.0174321   |     1.6346     |       5.10496   |      0.0281866  |      0.0370641  |
| lorenz96_twoscale_gate_s2f_0p0            | highdim_multiscale_mechanism | ngrc_sf_rg_gated        |          0.000673511 |    0.0236397   |     0.459837   |       2.09985   |      0.0391164  |      0.0704636  |
| lorenz96_twoscale_gate_s2f_0p0            | highdim_multiscale_mechanism | ngrc_takens_rg_additive |          0.000418262 |    0.0339778   |     1.06159    |       5.54671   |      1.0269     |      0.0647551  |
| lorenz96_twoscale_gate_s2f_0p0            | highdim_multiscale_mechanism | ngrc_takens_rg_true     |          0.000417832 |    0.0324921   |     0.975204   |       4.7499    |      0.0117347  |      0.0450255  |
| lorenz96_twoscale_gate_s2f_0p4            | highdim_multiscale_mechanism | ngrc_raw                |          0.000322748 |    0.0241647   |     2.79017    |       6.04969   |      0.0695251  |      0.0254741  |
| lorenz96_twoscale_gate_s2f_0p4            | highdim_multiscale_mechanism | ngrc_rg_readout         |          0.00042794  |    0.0113372   |     1.89573    |       5.12947   |      0.111982   |      0.0181395  |
| lorenz96_twoscale_gate_s2f_0p4            | highdim_multiscale_mechanism | ngrc_sf_rg_gated        |          0.000328201 |    0.0226765   |     2.38331    |       5.18752   |      0.0592548  |      0.0142656  |
| lorenz96_twoscale_gate_s2f_0p4            | highdim_multiscale_mechanism | ngrc_takens_rg_true     |          0.000357984 |    0.0183046   |     2.11148    |       5.45512   |      0.117237   |      0.0225473  |
| lorenz96_twoscale_gate_s2f_0p8            | highdim_multiscale_mechanism | ngrc_raw                |          0.000346812 |    0.00771734  |     0.636198   |       1.25328   |      0.0854081  |      0.01826    |
| lorenz96_twoscale_gate_s2f_0p8            | highdim_multiscale_mechanism | ngrc_rg_readout         |          0.000353108 |    0.00847257  |     0.691981   |       1.20399   |      0.058545   |      0.00789163 |
| lorenz96_twoscale_gate_s2f_0p8            | highdim_multiscale_mechanism | ngrc_sf_rg_gated        |          0.000225135 |    0.000255969 |     0.0324106  |       0.507283  |      0.038999   |      0.017947   |
| lorenz96_twoscale_gate_s2f_0p8            | highdim_multiscale_mechanism | ngrc_takens_rg_additive |          0.000283112 |    0.00706004  |     0.602873   |       1.14804   |      0.0794596  |      0.0166527  |
| lorenz96_twoscale_gate_s2f_0p8            | highdim_multiscale_mechanism | ngrc_takens_rg_true     |          0.000282073 |    0.00747226  |     0.649348   |       1.43709   |      0.06604    |      0.0142075  |
| lorenz96_twoscale_gate_s2f_1p2            | highdim_multiscale_mechanism | ngrc_raw                |          0.000107764 |    0.0061738   |     0.859456   |       2.40642   |      0.148143   |      0.0316711  |
| lorenz96_twoscale_gate_s2f_1p2            | highdim_multiscale_mechanism | ngrc_rg_readout         |          0.000107735 |    0.0019802   |     0.289167   |       0.551939  |      0.069113   |      0.0238296  |
| lorenz96_twoscale_gate_s2f_1p2            | highdim_multiscale_mechanism | ngrc_sf_rg_gated        |          0.000104033 |    0.00385684  |     0.475749   |       0.601971  |      0.0356941  |      0.0318697  |
| lorenz96_twoscale_gate_s2f_1p2            | highdim_multiscale_mechanism | ngrc_takens_rg_true     |          5.12225e-05 |    0.00367283  |     0.650472   |       1.66674   |      0.00830401 |      0.00847939 |
| lorenz96_twoscale_gate_s2f_1p6            | highdim_multiscale_mechanism | ngrc_raw                |          4.19509e-05 |    0.000436691 |     0.0251629  |       0.351479  |      0.113627   |      0.0075264  |
| lorenz96_twoscale_gate_s2f_1p6            | highdim_multiscale_mechanism | ngrc_rg_readout         |          4.35629e-05 |    6.97864e-05 |     0.0798148  |       0.3941    |      0.00907136 |      0.00795891 |
| lorenz96_twoscale_gate_s2f_1p6            | highdim_multiscale_mechanism | ngrc_sf_rg_gated        |          0.000106141 |    0.00193525  |     0.0970608  |       0.835505  |      0.0125924  |      0.00800554 |
| lorenz96_twoscale_gate_s2f_1p6            | highdim_multiscale_mechanism | ngrc_takens_rg_true     |          3.51276e-05 |    0.000942068 |     0.0986245  |       0.759214  |      1.02742    |      0.0709823  |
| lorenz96_twoscale_noise_homoskedastic     | highdim_multiscale_mechanism | ngrc_raw                |          0.082419    |    0.100442    |     6.91632    |       8.67227   |      0.308793   |      0.0503784  |
| lorenz96_twoscale_noise_homoskedastic     | highdim_multiscale_mechanism | ngrc_rg_readout         |          0.0832514   |    0.100998    |     6.92006    |       8.2789    |      0.278688   |      0.0410389  |
| lorenz96_twoscale_noise_homoskedastic     | highdim_multiscale_mechanism | ngrc_sf_rg_gated        |          0.072393    |    0.0808256   |     6.83785    |       8.30483   |      0.072432   |      0.0279287  |
| lorenz96_twoscale_noise_homoskedastic     | highdim_multiscale_mechanism | ngrc_takens_rg_additive |          0.082398    |    0.0968164   |     5.22185    |       6.91953   |      0.158261   |      0.0447344  |
| lorenz96_twoscale_noise_homoskedastic     | highdim_multiscale_mechanism | ngrc_takens_rg_true     |          0.0845546   |    0.0829563   |     5.76153    |       7.16567   |      0.346085   |      0.0658386  |
| lorenz96_twoscale_noise_matched_clustered | highdim_multiscale_mechanism | ngrc_raw                |          0.0766888   |    0.104881    |     5.53372    |       8.20871   |      0.28219    |      0.0406361  |
| lorenz96_twoscale_noise_matched_clustered | highdim_multiscale_mechanism | ngrc_rg_readout         |          0.076444    |    0.103502    |     7.71595    |       9.14589   |      0.285251   |      0.0363109  |
| lorenz96_twoscale_noise_matched_clustered | highdim_multiscale_mechanism | ngrc_sf_rg_gated        |          0.0714292   |    0.0940534   |     6.94185    |       9.12407   |      0.295361   |      0.0304111  |
| lorenz96_twoscale_noise_matched_clustered | highdim_multiscale_mechanism | ngrc_takens_rg_additive |          0.0766768   |    0.100329    |     6.92851    |       7.9294    |      0.302624   |      0.0453773  |
| lorenz96_twoscale_noise_matched_clustered | highdim_multiscale_mechanism | ngrc_takens_rg_true     |          0.0781143   |    0.0933563   |    12.3693     |      13.7111    |      0.523747   |      0.0584539  |
| lorenz96_twoscale_obs_mixed_projection    | highdim_multiscale_mechanism | ngrc_raw                |          7.74011e-05 |    0.00436431  |     0.73855    |       1.88663   |      0.0505407  |      0.012058   |
| lorenz96_twoscale_obs_mixed_projection    | highdim_multiscale_mechanism | ngrc_rg_readout         |          0.00012754  |    0.00683595  |     0.823314   |       2.22418   |      0.132349   |      0.017506   |
| lorenz96_twoscale_obs_mixed_projection    | highdim_multiscale_mechanism | ngrc_sf_rg_gated        |          0.000146382 |    0.0043946   |     0.594634   |       1.16666   |      0.0248526  |      0.0165804  |
| lorenz96_twoscale_obs_mixed_projection    | highdim_multiscale_mechanism | ngrc_takens_rg_additive |          6.1905e-05  |    0.00402506  |     0.707837   |       1.76774   |      0.0299276  |      0.0155448  |
| lorenz96_twoscale_obs_mixed_projection    | highdim_multiscale_mechanism | ngrc_takens_rg_true     |          6.57405e-05 |    0.00391773  |     0.683103   |       1.71507   |      0.0107602  |      0.0102274  |
| lorenz96_twoscale_obs_slow0               | highdim_multiscale_mechanism | ngrc_raw                |          0.000107764 |    0.0061738   |     0.859456   |       2.40642   |      0.148143   |      0.0316711  |
| lorenz96_twoscale_obs_slow0               | highdim_multiscale_mechanism | ngrc_rg_readout         |          0.000107735 |    0.0019802   |     0.289167   |       0.551939  |      0.069113   |      0.0238296  |
| lorenz96_twoscale_obs_slow0               | highdim_multiscale_mechanism | ngrc_sf_rg_gated        |          0.000104033 |    0.00385684  |     0.475749   |       0.601971  |      0.0356941  |      0.0318697  |
| lorenz96_twoscale_obs_slow0               | highdim_multiscale_mechanism | ngrc_takens_rg_additive |          5.22371e-05 |    0.00390081  |     0.701668   |       1.9024    |      0.00748375 |      0.012657   |
| lorenz96_twoscale_obs_slow0               | highdim_multiscale_mechanism | ngrc_takens_rg_true     |          5.12225e-05 |    0.00367283  |     0.650472   |       1.66674   |      0.00830401 |      0.00847939 |
| lorenz96_twoscale_obs_sparse_slowproj     | highdim_multiscale_mechanism | ngrc_raw                |          0.000166431 |    0.00204651  |     1.45541    |       4.02089   |      0.104887   |      0.0838465  |
| lorenz96_twoscale_obs_sparse_slowproj     | highdim_multiscale_mechanism | ngrc_rg_readout         |          0.000310097 |    0.00244031  |     1.39257    |       5.43188   |      0.0946778  |      0.0845034  |
| lorenz96_twoscale_obs_sparse_slowproj     | highdim_multiscale_mechanism | ngrc_sf_rg_gated        |          0.000388252 |    0.00232668  |     0.896542   |       3.89752   |      0.0433723  |      0.0615146  |
| lorenz96_twoscale_obs_sparse_slowproj     | highdim_multiscale_mechanism | ngrc_takens_rg_true     |          0.000236827 |    0.00149478  |     0.835671   |       2.24809   |      0.0947146  |      0.0892554  |

Winner by metric on the merged ablation set:

| task                                      | task_group                   | metric             | winner                  |   winner_value |
|:------------------------------------------|:-----------------------------|:-------------------|:------------------------|---------------:|
| fitzhugh_nagumo_classic_clean             | classic_fastslow             | acf_rmse_mean      | ngrc_rg_readout         |    0.929536    |
| fitzhugh_nagumo_classic_clean             | classic_fastslow             | one_step_rmse_mean | ngrc_sf_rg_gated        |    1.54682e-05 |
| fitzhugh_nagumo_classic_clean             | classic_fastslow             | psd_rmse_mean      | ngrc_rg_readout         |    0.0695115   |
| fitzhugh_nagumo_classic_clean             | classic_fastslow             | rmse@100_mean      | ngrc_raw                |    2.03483     |
| fitzhugh_nagumo_classic_clean             | classic_fastslow             | rmse@10_mean       | ngrc_sf_rg_gated        |    1.29625e-05 |
| fitzhugh_nagumo_classic_clean             | classic_fastslow             | rmse@50_mean       | ngrc_raw                |    0.00500111  |
| fitzhugh_nagumo_classic_noisy             | classic_fastslow             | acf_rmse_mean      | ngrc_rg_readout         |    0.0642308   |
| fitzhugh_nagumo_classic_noisy             | classic_fastslow             | one_step_rmse_mean | ngrc_takens_rg_true     |    0.0145391   |
| fitzhugh_nagumo_classic_noisy             | classic_fastslow             | psd_rmse_mean      | ngrc_rg_readout         |    0.0161231   |
| fitzhugh_nagumo_classic_noisy             | classic_fastslow             | rmse@100_mean      | ngrc_sf_rg_gated        |    0.301699    |
| fitzhugh_nagumo_classic_noisy             | classic_fastslow             | rmse@10_mean       | ngrc_rg_readout         |    0.0215982   |
| fitzhugh_nagumo_classic_noisy             | classic_fastslow             | rmse@50_mean       | ngrc_sf_rg_gated        |    0.0763619   |
| hindmarsh_rose_bursting_clean             | classic_fastslow             | acf_rmse_mean      | ngrc_rg_readout         |    0.0707107   |
| hindmarsh_rose_bursting_clean             | classic_fastslow             | one_step_rmse_mean | ngrc_rg_readout         |    2.59857e-05 |
| hindmarsh_rose_bursting_clean             | classic_fastslow             | psd_rmse_mean      | ngrc_takens_rg_true     |    0.00919787  |
| hindmarsh_rose_bursting_clean             | classic_fastslow             | rmse@100_mean      | ngrc_takens_rg_true     |    0.0334858   |
| hindmarsh_rose_bursting_clean             | classic_fastslow             | rmse@10_mean       | ngrc_takens_rg_true     |    0.000651402 |
| hindmarsh_rose_bursting_clean             | classic_fastslow             | rmse@50_mean       | ngrc_takens_rg_true     |    0.0136519   |
| hindmarsh_rose_bursting_noisy             | classic_fastslow             | acf_rmse_mean      | ngrc_sf_rg_gated        |    0.186741    |
| hindmarsh_rose_bursting_noisy             | classic_fastslow             | one_step_rmse_mean | ngrc_takens_rg_additive |    0.0127464   |
| hindmarsh_rose_bursting_noisy             | classic_fastslow             | psd_rmse_mean      | ngrc_sf_rg_gated        |    0.020422    |
| hindmarsh_rose_bursting_noisy             | classic_fastslow             | rmse@100_mean      | ngrc_takens_rg_true     |    0.345773    |
| hindmarsh_rose_bursting_noisy             | classic_fastslow             | rmse@10_mean       | ngrc_sf_rg_gated        |    0.0231503   |
| hindmarsh_rose_bursting_noisy             | classic_fastslow             | rmse@50_mean       | ngrc_takens_rg_true     |    0.112355    |
| vanderpol_relaxation_clean                | classic_fastslow             | acf_rmse_mean      | ngrc_takens_rg_true     |    1.04827     |
| vanderpol_relaxation_clean                | classic_fastslow             | one_step_rmse_mean | ngrc_takens_rg_additive |    0.000119988 |
| vanderpol_relaxation_clean                | classic_fastslow             | psd_rmse_mean      | ngrc_raw                |    0.0754198   |
| vanderpol_relaxation_clean                | classic_fastslow             | rmse@100_mean      | ngrc_takens_rg_true     |    7.22613     |
| vanderpol_relaxation_clean                | classic_fastslow             | rmse@10_mean       | ngrc_takens_rg_true     |    9.03555e-05 |
| vanderpol_relaxation_clean                | classic_fastslow             | rmse@50_mean       | ngrc_takens_rg_true     |    0.108042    |
| vanderpol_relaxation_noisy                | classic_fastslow             | acf_rmse_mean      | ngrc_takens_rg_true     |    0.0205817   |
| vanderpol_relaxation_noisy                | classic_fastslow             | one_step_rmse_mean | ngrc_sf_rg_gated        |    0.013868    |
| vanderpol_relaxation_noisy                | classic_fastslow             | psd_rmse_mean      | ngrc_takens_rg_additive |    0.00419894  |
| vanderpol_relaxation_noisy                | classic_fastslow             | rmse@100_mean      | ngrc_takens_rg_additive |    0.0324906   |
| vanderpol_relaxation_noisy                | classic_fastslow             | rmse@10_mean       | ngrc_rg_readout         |    0.0111121   |
| vanderpol_relaxation_noisy                | classic_fastslow             | rmse@50_mean       | ngrc_takens_rg_true     |    0.0215328   |
| fitzhugh_nagumo_classic_volclustered      | classic_fastslow_finance     | acf_rmse_mean      | ngrc_takens_rg_additive |    0.0279436   |
| fitzhugh_nagumo_classic_volclustered      | classic_fastslow_finance     | one_step_rmse_mean | ngrc_takens_rg_true     |    0.0449738   |
| fitzhugh_nagumo_classic_volclustered      | classic_fastslow_finance     | psd_rmse_mean      | ngrc_raw                |    0.00474414  |
| fitzhugh_nagumo_classic_volclustered      | classic_fastslow_finance     | rmse@100_mean      | ngrc_raw                |    0.483331    |
| fitzhugh_nagumo_classic_volclustered      | classic_fastslow_finance     | rmse@10_mean       | ngrc_raw                |    0.041187    |
| fitzhugh_nagumo_classic_volclustered      | classic_fastslow_finance     | rmse@50_mean       | ngrc_takens_rg_additive |    0.135712    |
| hindmarsh_rose_bursting_volclustered      | classic_fastslow_finance     | acf_rmse_mean      | ngrc_takens_rg_true     |    0.259046    |
| hindmarsh_rose_bursting_volclustered      | classic_fastslow_finance     | one_step_rmse_mean | ngrc_takens_rg_additive |    0.0535103   |
| hindmarsh_rose_bursting_volclustered      | classic_fastslow_finance     | psd_rmse_mean      | ngrc_takens_rg_true     |    0.019698    |
| hindmarsh_rose_bursting_volclustered      | classic_fastslow_finance     | rmse@100_mean      | ngrc_sf_rg_gated        |    0.0940554   |
| hindmarsh_rose_bursting_volclustered      | classic_fastslow_finance     | rmse@10_mean       | ngrc_raw                |    0.0320227   |
| hindmarsh_rose_bursting_volclustered      | classic_fastslow_finance     | rmse@50_mean       | ngrc_takens_rg_true     |    0.0511485   |
| vanderpol_relaxation_volclustered         | classic_fastslow_finance     | acf_rmse_mean      | ngrc_raw                |    0.171541    |
| vanderpol_relaxation_volclustered         | classic_fastslow_finance     | one_step_rmse_mean | ngrc_rg_readout         |    0.0625749   |
| vanderpol_relaxation_volclustered         | classic_fastslow_finance     | psd_rmse_mean      | ngrc_takens_rg_additive |    0.0141653   |
| vanderpol_relaxation_volclustered         | classic_fastslow_finance     | rmse@100_mean      | ngrc_sf_rg_gated        |    0.675485    |
| vanderpol_relaxation_volclustered         | classic_fastslow_finance     | rmse@10_mean       | ngrc_sf_rg_gated        |    0.0560928   |
| vanderpol_relaxation_volclustered         | classic_fastslow_finance     | rmse@50_mean       | ngrc_sf_rg_gated        |    0.456947    |
| lorenz96_twoscale_gate_s2f_0p0            | highdim_multiscale_mechanism | acf_rmse_mean      | ngrc_takens_rg_true     |    0.0117347   |
| lorenz96_twoscale_gate_s2f_0p0            | highdim_multiscale_mechanism | one_step_rmse_mean | ngrc_raw                |    0.000320336 |
| lorenz96_twoscale_gate_s2f_0p0            | highdim_multiscale_mechanism | psd_rmse_mean      | ngrc_raw                |    0.0252107   |
| lorenz96_twoscale_gate_s2f_0p0            | highdim_multiscale_mechanism | rmse@100_mean      | ngrc_sf_rg_gated        |    2.09985     |
| lorenz96_twoscale_gate_s2f_0p0            | highdim_multiscale_mechanism | rmse@10_mean       | ngrc_raw                |    0.0151841   |
| lorenz96_twoscale_gate_s2f_0p0            | highdim_multiscale_mechanism | rmse@50_mean       | ngrc_sf_rg_gated        |    0.459837    |
| lorenz96_twoscale_gate_s2f_0p4            | highdim_multiscale_mechanism | acf_rmse_mean      | ngrc_sf_rg_gated        |    0.0592548   |
| lorenz96_twoscale_gate_s2f_0p4            | highdim_multiscale_mechanism | one_step_rmse_mean | ngrc_raw                |    0.000322748 |
| lorenz96_twoscale_gate_s2f_0p4            | highdim_multiscale_mechanism | psd_rmse_mean      | ngrc_sf_rg_gated        |    0.0142656   |
| lorenz96_twoscale_gate_s2f_0p4            | highdim_multiscale_mechanism | rmse@100_mean      | ngrc_rg_readout         |    5.12947     |
| lorenz96_twoscale_gate_s2f_0p4            | highdim_multiscale_mechanism | rmse@10_mean       | ngrc_rg_readout         |    0.0113372   |
| lorenz96_twoscale_gate_s2f_0p4            | highdim_multiscale_mechanism | rmse@50_mean       | ngrc_rg_readout         |    1.89573     |
| lorenz96_twoscale_gate_s2f_0p8            | highdim_multiscale_mechanism | acf_rmse_mean      | ngrc_sf_rg_gated        |    0.038999    |
| lorenz96_twoscale_gate_s2f_0p8            | highdim_multiscale_mechanism | one_step_rmse_mean | ngrc_sf_rg_gated        |    0.000225135 |
| lorenz96_twoscale_gate_s2f_0p8            | highdim_multiscale_mechanism | psd_rmse_mean      | ngrc_rg_readout         |    0.00789163  |
| lorenz96_twoscale_gate_s2f_0p8            | highdim_multiscale_mechanism | rmse@100_mean      | ngrc_sf_rg_gated        |    0.507283    |
| lorenz96_twoscale_gate_s2f_0p8            | highdim_multiscale_mechanism | rmse@10_mean       | ngrc_sf_rg_gated        |    0.000255969 |
| lorenz96_twoscale_gate_s2f_0p8            | highdim_multiscale_mechanism | rmse@50_mean       | ngrc_sf_rg_gated        |    0.0324106   |
| lorenz96_twoscale_gate_s2f_1p2            | highdim_multiscale_mechanism | acf_rmse_mean      | ngrc_takens_rg_true     |    0.00830401  |
| lorenz96_twoscale_gate_s2f_1p2            | highdim_multiscale_mechanism | one_step_rmse_mean | ngrc_takens_rg_true     |    5.12225e-05 |
| lorenz96_twoscale_gate_s2f_1p2            | highdim_multiscale_mechanism | psd_rmse_mean      | ngrc_takens_rg_true     |    0.00847939  |
| lorenz96_twoscale_gate_s2f_1p2            | highdim_multiscale_mechanism | rmse@100_mean      | ngrc_rg_readout         |    0.551939    |
| lorenz96_twoscale_gate_s2f_1p2            | highdim_multiscale_mechanism | rmse@10_mean       | ngrc_rg_readout         |    0.0019802   |
| lorenz96_twoscale_gate_s2f_1p2            | highdim_multiscale_mechanism | rmse@50_mean       | ngrc_rg_readout         |    0.289167    |
| lorenz96_twoscale_gate_s2f_1p6            | highdim_multiscale_mechanism | acf_rmse_mean      | ngrc_rg_readout         |    0.00907136  |
| lorenz96_twoscale_gate_s2f_1p6            | highdim_multiscale_mechanism | one_step_rmse_mean | ngrc_takens_rg_true     |    3.51276e-05 |
| lorenz96_twoscale_gate_s2f_1p6            | highdim_multiscale_mechanism | psd_rmse_mean      | ngrc_raw                |    0.0075264   |
| lorenz96_twoscale_gate_s2f_1p6            | highdim_multiscale_mechanism | rmse@100_mean      | ngrc_raw                |    0.351479    |
| lorenz96_twoscale_gate_s2f_1p6            | highdim_multiscale_mechanism | rmse@10_mean       | ngrc_rg_readout         |    6.97864e-05 |
| lorenz96_twoscale_gate_s2f_1p6            | highdim_multiscale_mechanism | rmse@50_mean       | ngrc_raw                |    0.0251629   |
| lorenz96_twoscale_noise_homoskedastic     | highdim_multiscale_mechanism | acf_rmse_mean      | ngrc_sf_rg_gated        |    0.072432    |
| lorenz96_twoscale_noise_homoskedastic     | highdim_multiscale_mechanism | one_step_rmse_mean | ngrc_sf_rg_gated        |    0.072393    |
| lorenz96_twoscale_noise_homoskedastic     | highdim_multiscale_mechanism | psd_rmse_mean      | ngrc_sf_rg_gated        |    0.0279287   |
| lorenz96_twoscale_noise_homoskedastic     | highdim_multiscale_mechanism | rmse@100_mean      | ngrc_takens_rg_additive |    6.91953     |
| lorenz96_twoscale_noise_homoskedastic     | highdim_multiscale_mechanism | rmse@10_mean       | ngrc_sf_rg_gated        |    0.0808256   |
| lorenz96_twoscale_noise_homoskedastic     | highdim_multiscale_mechanism | rmse@50_mean       | ngrc_takens_rg_additive |    5.22185     |
| lorenz96_twoscale_noise_matched_clustered | highdim_multiscale_mechanism | acf_rmse_mean      | ngrc_raw                |    0.28219     |
| lorenz96_twoscale_noise_matched_clustered | highdim_multiscale_mechanism | one_step_rmse_mean | ngrc_sf_rg_gated        |    0.0714292   |
| lorenz96_twoscale_noise_matched_clustered | highdim_multiscale_mechanism | psd_rmse_mean      | ngrc_sf_rg_gated        |    0.0304111   |
| lorenz96_twoscale_noise_matched_clustered | highdim_multiscale_mechanism | rmse@100_mean      | ngrc_takens_rg_additive |    7.9294      |
| lorenz96_twoscale_noise_matched_clustered | highdim_multiscale_mechanism | rmse@10_mean       | ngrc_takens_rg_true     |    0.0933563   |
| lorenz96_twoscale_noise_matched_clustered | highdim_multiscale_mechanism | rmse@50_mean       | ngrc_raw                |    5.53372     |
| lorenz96_twoscale_obs_mixed_projection    | highdim_multiscale_mechanism | acf_rmse_mean      | ngrc_takens_rg_true     |    0.0107602   |
| lorenz96_twoscale_obs_mixed_projection    | highdim_multiscale_mechanism | one_step_rmse_mean | ngrc_takens_rg_additive |    6.1905e-05  |
| lorenz96_twoscale_obs_mixed_projection    | highdim_multiscale_mechanism | psd_rmse_mean      | ngrc_takens_rg_true     |    0.0102274   |
| lorenz96_twoscale_obs_mixed_projection    | highdim_multiscale_mechanism | rmse@100_mean      | ngrc_sf_rg_gated        |    1.16666     |
| lorenz96_twoscale_obs_mixed_projection    | highdim_multiscale_mechanism | rmse@10_mean       | ngrc_takens_rg_true     |    0.00391773  |
| lorenz96_twoscale_obs_mixed_projection    | highdim_multiscale_mechanism | rmse@50_mean       | ngrc_sf_rg_gated        |    0.594634    |
| lorenz96_twoscale_obs_slow0               | highdim_multiscale_mechanism | acf_rmse_mean      | ngrc_takens_rg_additive |    0.00748375  |
| lorenz96_twoscale_obs_slow0               | highdim_multiscale_mechanism | one_step_rmse_mean | ngrc_takens_rg_true     |    5.12225e-05 |
| lorenz96_twoscale_obs_slow0               | highdim_multiscale_mechanism | psd_rmse_mean      | ngrc_takens_rg_true     |    0.00847939  |
| lorenz96_twoscale_obs_slow0               | highdim_multiscale_mechanism | rmse@100_mean      | ngrc_rg_readout         |    0.551939    |
| lorenz96_twoscale_obs_slow0               | highdim_multiscale_mechanism | rmse@10_mean       | ngrc_rg_readout         |    0.0019802   |
| lorenz96_twoscale_obs_slow0               | highdim_multiscale_mechanism | rmse@50_mean       | ngrc_rg_readout         |    0.289167    |
| lorenz96_twoscale_obs_sparse_slowproj     | highdim_multiscale_mechanism | acf_rmse_mean      | ngrc_sf_rg_gated        |    0.0433723   |
| lorenz96_twoscale_obs_sparse_slowproj     | highdim_multiscale_mechanism | one_step_rmse_mean | ngrc_raw                |    0.000166431 |
| lorenz96_twoscale_obs_sparse_slowproj     | highdim_multiscale_mechanism | psd_rmse_mean      | ngrc_sf_rg_gated        |    0.0615146   |
| lorenz96_twoscale_obs_sparse_slowproj     | highdim_multiscale_mechanism | rmse@100_mean      | ngrc_takens_rg_true     |    2.24809     |
| lorenz96_twoscale_obs_sparse_slowproj     | highdim_multiscale_mechanism | rmse@10_mean       | ngrc_takens_rg_true     |    0.00149478  |
| lorenz96_twoscale_obs_sparse_slowproj     | highdim_multiscale_mechanism | rmse@50_mean       | ngrc_takens_rg_true     |    0.835671    |

## Paper-Style Task Summary

A compact paper-style synthesis that combines predictive winners, coordinate diagnostics, and a conservative mechanistic reading:

| task_group                   | task                                      | context                                    | best_local                | best_mid_rollout       | best_long_rollout       | best_distribution                                       | best_coordinate                                                         | paper_takeaway                                                                                   | confidence   |
|:-----------------------------|:------------------------------------------|:-------------------------------------------|:--------------------------|:-----------------------|:------------------------|:--------------------------------------------------------|:------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------|:-------------|
| classic_fastslow             | fitzhugh_nagumo_classic_clean             | excitable_clean_partial                    | SF+RG-gate (1.547e-05)    | Raw (0.005001)         | Raw (2.035)             | ACF:RG-readout (0.93); PSD:RG-readout (0.0695)          |                                                                         | Delay backbone is sufficient; extra macro conditioning does not help here.                       | high         |
| classic_fastslow             | fitzhugh_nagumo_classic_noisy             | excitable_noisy_partial                    | Takens+RG-op (0.01454)    | SF+RG-gate (0.07636)   | SF+RG-gate (0.3017)     | ACF:RG-readout (0.0642); PSD:RG-readout (0.0161)        | closure:delay (0.00261); koopman:delay (0.997); spectral:delay (0.0627) | Explicit fast-slow closure with sparse RG gating is required to capture the dominant mechanism.  | high         |
| classic_fastslow             | hindmarsh_rose_bursting_clean             | bursting_clean_partial                     | RG-readout (2.599e-05)    | Takens+RG-op (0.01365) | Takens+RG-op (0.03349)  | ACF:RG-readout (0.0707); PSD:Takens+RG-op (0.0092)      |                                                                         | Delay backbone plus RG-conditioned local operator is the strongest predictive mechanism.         | high         |
| classic_fastslow             | hindmarsh_rose_bursting_noisy             | bursting_noisy_partial                     | Takens+RG-add (0.01275)   | Takens+RG-op (0.1124)  | Takens+RG-op (0.3458)   | ACF:SF+RG-gate (0.187); PSD:SF+RG-gate (0.0204)         |                                                                         | Delay backbone plus RG-conditioned local operator is the strongest predictive mechanism.         | high         |
| classic_fastslow             | vanderpol_relaxation_clean                | relaxation_clean_partial                   | Takens+RG-add (0.00012)   | Takens+RG-op (0.108)   | Takens+RG-op (7.226)    | ACF:Takens+RG-op (1.05); PSD:Raw (0.0754)               |                                                                         | Delay backbone plus RG-conditioned local operator is the strongest predictive mechanism.         | high         |
| classic_fastslow             | vanderpol_relaxation_noisy                | relaxation_noisy_partial                   | SF+RG-gate (0.01387)      | Takens+RG-op (0.02153) | Takens+RG-add (0.03249) | ACF:Takens+RG-op (0.0206); PSD:Takens+RG-add (0.0042)   | closure:delay (0.00171); koopman:delay (0.997); spectral:delay (0.0399) | Delay backbone plus RG-conditioned local operator is the strongest predictive mechanism.         | low          |
| classic_fastslow_finance     | fitzhugh_nagumo_classic_volclustered      | excitable_volclustered_partial             | Takens+RG-op (0.04497)    | Takens+RG-add (0.1357) | Raw (0.4833)            | ACF:Takens+RG-add (0.0279); PSD:Raw (0.00474)           |                                                                         | A simple macro residual is more robust than multiplicative operator conditioning in this regime. | low          |
| classic_fastslow_finance     | hindmarsh_rose_bursting_volclustered      | bursting_volclustered_partial              | Takens+RG-add (0.05351)   | Takens+RG-op (0.05115) | SF+RG-gate (0.09406)    | ACF:Takens+RG-op (0.259); PSD:Takens+RG-op (0.0197)     |                                                                         | Delay backbone plus RG-conditioned local operator is the strongest predictive mechanism.         | low          |
| classic_fastslow_finance     | vanderpol_relaxation_volclustered         | relaxation_volclustered_partial            | RG-readout (0.06257)      | SF+RG-gate (0.4569)    | SF+RG-gate (0.6755)     | ACF:Raw (0.172); PSD:Takens+RG-add (0.0142)             |                                                                         | Explicit fast-slow closure with sparse RG gating is required to capture the dominant mechanism.  | medium       |
| highdim_multiscale_mechanism | lorenz96_twoscale_gate_s2f_0p0            | slow0 / clean                              | Raw (0.0003203)           | SF+RG-gate (0.4598)    | SF+RG-gate (2.1)        | ACF:Takens+RG-op (0.0117); PSD:Raw (0.0252)             | closure:delay (0.166); koopman:delay (1); spectral:delay (0.042)        | Explicit fast-slow closure with sparse RG gating is required to capture the dominant mechanism.  | high         |
| highdim_multiscale_mechanism | lorenz96_twoscale_gate_s2f_0p4            | slow0 / clean                              | Raw (0.0003227)           | RG-readout (1.896)     | RG-readout (5.129)      | ACF:SF+RG-gate (0.0593); PSD:SF+RG-gate (0.0143)        |                                                                         | Observation is already aligned with the macro slow state; additive RG readout is sufficient.     | high         |
| highdim_multiscale_mechanism | lorenz96_twoscale_gate_s2f_0p8            | slow0 / clean                              | SF+RG-gate (0.0002251)    | SF+RG-gate (0.03241)   | SF+RG-gate (0.5073)     | ACF:SF+RG-gate (0.039); PSD:RG-readout (0.00789)        | closure:delay (-0.224); koopman:delay (1); spectral:rg (0.241)          | Explicit fast-slow closure with sparse RG gating is required to capture the dominant mechanism.  | high         |
| highdim_multiscale_mechanism | lorenz96_twoscale_gate_s2f_1p2            | slow0 / clean                              | Takens+RG-op (5.122e-05)  | RG-readout (0.2892)    | RG-readout (0.5519)     | ACF:Takens+RG-op (0.0083); PSD:Takens+RG-op (0.00848)   |                                                                         | Observation is already aligned with the macro slow state; additive RG readout is sufficient.     | high         |
| highdim_multiscale_mechanism | lorenz96_twoscale_gate_s2f_1p6            | slow0 / clean                              | Takens+RG-op (3.513e-05)  | Raw (0.02516)          | Raw (0.3515)            | ACF:RG-readout (0.00907); PSD:Raw (0.00753)             |                                                                         | Delay backbone is sufficient; extra macro conditioning does not help here.                       | high         |
| highdim_multiscale_mechanism | lorenz96_twoscale_noise_homoskedastic     | sparse_slow_projection / homoskedastic     | SF+RG-gate (0.07239)      | Takens+RG-add (5.222)  | Takens+RG-add (6.92)    | ACF:SF+RG-gate (0.0724); PSD:SF+RG-gate (0.0279)        |                                                                         | A simple macro residual is more robust than multiplicative operator conditioning in this regime. | medium       |
| highdim_multiscale_mechanism | lorenz96_twoscale_noise_matched_clustered | sparse_slow_projection / matched_clustered | SF+RG-gate (0.07143)      | Raw (5.534)            | Takens+RG-add (7.929)   | ACF:Raw (0.282); PSD:SF+RG-gate (0.0304)                |                                                                         | Delay backbone is sufficient; extra macro conditioning does not help here.                       | medium       |
| highdim_multiscale_mechanism | lorenz96_twoscale_obs_mixed_projection    | slow_fast_mixed_projection / clean         | Takens+RG-add (6.191e-05) | SF+RG-gate (0.5946)    | SF+RG-gate (1.167)      | ACF:Takens+RG-op (0.0108); PSD:Takens+RG-op (0.0102)    | closure:delay (0.0768); koopman:delay (1); spectral:delay+rg (0.444)    | Explicit fast-slow closure with sparse RG gating is required to capture the dominant mechanism.  | high         |
| highdim_multiscale_mechanism | lorenz96_twoscale_obs_slow0               | slow0 / clean                              | Takens+RG-op (5.122e-05)  | RG-readout (0.2892)    | RG-readout (0.5519)     | ACF:Takens+RG-add (0.00748); PSD:Takens+RG-op (0.00848) | closure:delay (0.0827); koopman:delay (1); spectral:rg (0.409)          | Observation is already aligned with the macro slow state; additive RG readout is sufficient.     | high         |
| highdim_multiscale_mechanism | lorenz96_twoscale_obs_sparse_slowproj     | sparse_slow_projection / clean             | Raw (0.0001664)           | Takens+RG-op (0.8357)  | Takens+RG-op (2.248)    | ACF:SF+RG-gate (0.0434); PSD:SF+RG-gate (0.0615)        |                                                                         | Delay backbone plus RG-conditioned local operator is the strongest predictive mechanism.         | medium       |

## Specificity Controls

Matched-control benchmark summary (`rmse@50_mean` and `rmse@100_mean`):

| task                                   | variant                       |   seed_count |   rmse@50_mean |   rmse@50_std |   rmse@100_mean |   rmse@100_std |   acf_rmse_mean |   psd_rmse_mean |
|:---------------------------------------|:------------------------------|-------------:|---------------:|--------------:|----------------:|---------------:|----------------:|----------------:|
| fitzhugh_nagumo_classic_noisy          | ngrc_raw                      |            3 |      0.0962377 |    0.0655392  |       0.447278  |      0.462503  |      0.325264   |      0.0329528  |
| fitzhugh_nagumo_classic_noisy          | ngrc_rg_readout               |            3 |      0.0910248 |    0.0802339  |       0.534445  |      0.502277  |      0.0642308  |      0.0161231  |
| fitzhugh_nagumo_classic_noisy          | ngrc_takens_rg_lagged_control |            3 |      0.0907829 |    0.0816959  |       0.348944  |      0.449222  |      0.601706   |      0.0454621  |
| fitzhugh_nagumo_classic_noisy          | ngrc_takens_rg_random_control |            3 |      0.104862  |    0.0310668  |       0.402034  |      0.374991  |      0.343177   |      0.0274514  |
| fitzhugh_nagumo_classic_noisy          | ngrc_takens_rg_true           |            3 |      0.104601  |    0.122049   |       0.568517  |      0.591877  |      0.112299   |      0.0228104  |
| lorenz96_twoscale_gate_s2f_0p0         | ngrc_raw                      |            3 |      1.46848   |    0          |       4.4385    |      0         |      0.0386328  |      0.0252107  |
| lorenz96_twoscale_gate_s2f_0p0         | ngrc_rg_readout               |            3 |      1.6346    |    0          |       5.10496   |      0         |      0.0281866  |      0.0370641  |
| lorenz96_twoscale_gate_s2f_0p0         | ngrc_takens_rg_lagged_control |            3 |      1.156     |    0          |       2.5158    |      0         |      0.0224464  |      0.0289273  |
| lorenz96_twoscale_gate_s2f_0p0         | ngrc_takens_rg_random_control |            3 |      0.847444  |    0          |       5.01655   |      0         |      0.012217   |      0.0308438  |
| lorenz96_twoscale_gate_s2f_0p0         | ngrc_takens_rg_true           |            3 |      0.975204  |    0          |       4.7499    |      0         |      0.0117347  |      0.0450255  |
| lorenz96_twoscale_gate_s2f_0p8         | ngrc_raw                      |            3 |      0.636198  |    0          |       1.25328   |      0         |      0.0854081  |      0.01826    |
| lorenz96_twoscale_gate_s2f_0p8         | ngrc_rg_readout               |            3 |      0.691981  |    0          |       1.20399   |      0         |      0.058545   |      0.00789163 |
| lorenz96_twoscale_gate_s2f_0p8         | ngrc_takens_rg_lagged_control |            3 |      0.0621328 |    0          |       0.561079  |      0         |      0.00740105 |      0.00725478 |
| lorenz96_twoscale_gate_s2f_0p8         | ngrc_takens_rg_random_control |            3 |      0.689154  |    0          |       1.45994   |      0         |      0.0699524  |      0.00984971 |
| lorenz96_twoscale_gate_s2f_0p8         | ngrc_takens_rg_true           |            3 |      0.649348  |    0          |       1.43709   |      0         |      0.06604    |      0.0142075  |
| lorenz96_twoscale_obs_mixed_projection | ngrc_raw                      |            3 |      0.73855   |    0          |       1.88663   |      0         |      0.0505407  |      0.012058   |
| lorenz96_twoscale_obs_mixed_projection | ngrc_rg_readout               |            3 |      0.823314  |    0          |       2.22418   |      0         |      0.132349   |      0.017506   |
| lorenz96_twoscale_obs_mixed_projection | ngrc_takens_rg_lagged_control |            3 |      0.679465  |    0          |       1.73034   |      0         |      0.0106398  |      0.0168647  |
| lorenz96_twoscale_obs_mixed_projection | ngrc_takens_rg_random_control |            3 |      0.553061  |    0          |       1.08607   |      0         |      0.0129669  |      0.0169775  |
| lorenz96_twoscale_obs_mixed_projection | ngrc_takens_rg_true           |            3 |      0.683103  |    0          |       1.71507   |      0         |      0.0107602  |      0.0102274  |
| lorenz96_twoscale_obs_slow0            | ngrc_raw                      |            3 |      0.859456  |    0          |       2.40642   |      0         |      0.148143   |      0.0316711  |
| lorenz96_twoscale_obs_slow0            | ngrc_rg_readout               |            3 |      0.289167  |    0          |       0.551939  |      0         |      0.069113   |      0.0238296  |
| lorenz96_twoscale_obs_slow0            | ngrc_takens_rg_lagged_control |            3 |      0.753076  |    0          |       2.22493   |      0         |      0.00797014 |      0.02839    |
| lorenz96_twoscale_obs_slow0            | ngrc_takens_rg_random_control |            3 |      0.664795  |    0          |       1.76125   |      0         |      0.0287887  |      0.01115    |
| lorenz96_twoscale_obs_slow0            | ngrc_takens_rg_true           |            3 |      0.650472  |    0          |       1.66674   |      0         |      0.00830401 |      0.00847939 |
| vanderpol_relaxation_noisy             | ngrc_raw                      |            3 |      0.0481144 |    0.031054   |       0.121356  |      0.0885164 |      0.0292482  |      0.00618834 |
| vanderpol_relaxation_noisy             | ngrc_rg_readout               |            3 |      0.0262099 |    0.010723   |       0.046542  |      0.0182809 |      0.0274486  |      0.00711544 |
| vanderpol_relaxation_noisy             | ngrc_takens_rg_lagged_control |            3 |      0.0767211 |    0.0360707  |       0.17444   |      0.0701406 |      0.152804   |      0.0144436  |
| vanderpol_relaxation_noisy             | ngrc_takens_rg_random_control |            3 |      0.0627217 |    0.0202705  |       0.155105  |      0.02442   |      0.0248977  |      0.00551437 |
| vanderpol_relaxation_noisy             | ngrc_takens_rg_true           |            3 |      0.0215328 |    0.00727087 |       0.0332546 |      0.0186209 |      0.0205817  |      0.00424696 |

True-vs-control gaps (`true - control`, lower is better so negative values favor true RG conditioning):

| task                                   |   ngrc_takens_rg_true |   ngrc_takens_rg_lagged_control |   ngrc_takens_rg_random_control |   true_minus_lagged |   true_minus_random |
|:---------------------------------------|----------------------:|--------------------------------:|--------------------------------:|--------------------:|--------------------:|
| fitzhugh_nagumo_classic_noisy          |             0.104601  |                       0.0907829 |                       0.104862  |          0.0138176  |         -0.00026198 |
| lorenz96_twoscale_gate_s2f_0p0         |             0.975204  |                       1.156     |                       0.847444  |         -0.180792   |          0.12776    |
| lorenz96_twoscale_gate_s2f_0p8         |             0.649348  |                       0.0621328 |                       0.689154  |          0.587215   |         -0.0398059  |
| lorenz96_twoscale_obs_mixed_projection |             0.683103  |                       0.679465  |                       0.553061  |          0.00363749 |          0.130042   |
| lorenz96_twoscale_obs_slow0            |             0.650472  |                       0.753076  |                       0.664795  |         -0.102605   |         -0.0143231  |
| vanderpol_relaxation_noisy             |             0.0215328 |                       0.0767211 |                       0.0627217 |         -0.0551883  |         -0.0411889  |

## Conditioning-Form Ablation

| task                                   | variant                 |   seed_count |   rmse@50_mean |   rmse@50_std |   rmse@100_mean |   acf_rmse_mean |   psd_rmse_mean |
|:---------------------------------------|:------------------------|-------------:|---------------:|--------------:|----------------:|----------------:|----------------:|
| lorenz96_twoscale_gate_s2f_0p0         | ngrc_raw                |            3 |      1.46848   |             0 |        4.4385   |      0.0386328  |      0.0252107  |
| lorenz96_twoscale_gate_s2f_0p0         | ngrc_rg_readout         |            3 |      1.6346    |             0 |        5.10496  |      0.0281866  |      0.0370641  |
| lorenz96_twoscale_gate_s2f_0p0         | ngrc_sf_rg_gated        |            3 |      0.459837  |             0 |        2.09985  |      0.0391164  |      0.0704636  |
| lorenz96_twoscale_gate_s2f_0p0         | ngrc_takens_rg_additive |            3 |      1.06159   |             0 |        5.54671  |      1.0269     |      0.0647551  |
| lorenz96_twoscale_gate_s2f_0p0         | ngrc_takens_rg_true     |            3 |      0.975204  |             0 |        4.7499   |      0.0117347  |      0.0450255  |
| lorenz96_twoscale_gate_s2f_0p8         | ngrc_raw                |            3 |      0.636198  |             0 |        1.25328  |      0.0854081  |      0.01826    |
| lorenz96_twoscale_gate_s2f_0p8         | ngrc_rg_readout         |            3 |      0.691981  |             0 |        1.20399  |      0.058545   |      0.00789163 |
| lorenz96_twoscale_gate_s2f_0p8         | ngrc_sf_rg_gated        |            3 |      0.0324106 |             0 |        0.507283 |      0.038999   |      0.017947   |
| lorenz96_twoscale_gate_s2f_0p8         | ngrc_takens_rg_additive |            3 |      0.602873  |             0 |        1.14804  |      0.0794596  |      0.0166527  |
| lorenz96_twoscale_gate_s2f_0p8         | ngrc_takens_rg_true     |            3 |      0.649348  |             0 |        1.43709  |      0.06604    |      0.0142075  |
| lorenz96_twoscale_obs_mixed_projection | ngrc_raw                |            3 |      0.73855   |             0 |        1.88663  |      0.0505407  |      0.012058   |
| lorenz96_twoscale_obs_mixed_projection | ngrc_rg_readout         |            3 |      0.823314  |             0 |        2.22418  |      0.132349   |      0.017506   |
| lorenz96_twoscale_obs_mixed_projection | ngrc_sf_rg_gated        |            3 |      0.594634  |             0 |        1.16666  |      0.0248526  |      0.0165804  |
| lorenz96_twoscale_obs_mixed_projection | ngrc_takens_rg_additive |            3 |      0.707837  |             0 |        1.76774  |      0.0299276  |      0.0155448  |
| lorenz96_twoscale_obs_mixed_projection | ngrc_takens_rg_true     |            3 |      0.683103  |             0 |        1.71507  |      0.0107602  |      0.0102274  |
| lorenz96_twoscale_obs_slow0            | ngrc_raw                |            3 |      0.859456  |             0 |        2.40642  |      0.148143   |      0.0316711  |
| lorenz96_twoscale_obs_slow0            | ngrc_rg_readout         |            3 |      0.289167  |             0 |        0.551939 |      0.069113   |      0.0238296  |
| lorenz96_twoscale_obs_slow0            | ngrc_sf_rg_gated        |            3 |      0.475749  |             0 |        0.601971 |      0.0356941  |      0.0318697  |
| lorenz96_twoscale_obs_slow0            | ngrc_takens_rg_additive |            3 |      0.701668  |             0 |        1.9024   |      0.00748375 |      0.012657   |
| lorenz96_twoscale_obs_slow0            | ngrc_takens_rg_true     |            3 |      0.650472  |             0 |        1.66674  |      0.00830401 |      0.00847939 |

Best variant per task for the conditioning-form study:

| task                                   | variant          |   rmse@50_mean |   rmse@100_mean |   acf_rmse_mean |   psd_rmse_mean |
|:---------------------------------------|:-----------------|---------------:|----------------:|----------------:|----------------:|
| lorenz96_twoscale_gate_s2f_0p0         | ngrc_sf_rg_gated |      0.459837  |        2.09985  |       0.0391164 |       0.0704636 |
| lorenz96_twoscale_gate_s2f_0p8         | ngrc_sf_rg_gated |      0.0324106 |        0.507283 |       0.038999  |       0.017947  |
| lorenz96_twoscale_obs_mixed_projection | ngrc_sf_rg_gated |      0.594634  |        1.16666  |       0.0248526 |       0.0165804 |
| lorenz96_twoscale_obs_slow0            | ngrc_rg_readout  |      0.289167  |        0.551939 |       0.069113  |       0.0238296 |

## Boundary Sweep

| task                                   | sweep_group          |   sweep_value | observability_profile      | variant             |   rmse@50_mean |   rmse@50_std |   rmse@100_mean |
|:---------------------------------------|:---------------------|--------------:|:---------------------------|:--------------------|---------------:|--------------:|----------------:|
| lorenz96_twoscale_obs_slow0            | observability        |           0   | slow0                      | ngrc_raw            |      0.859456  |             0 |        2.40642  |
| lorenz96_twoscale_obs_slow0            | observability        |           0   | slow0                      | ngrc_rg_readout     |      0.289167  |             0 |        0.551939 |
| lorenz96_twoscale_obs_slow0            | observability        |           0   | slow0                      | ngrc_sf_rg_gated    |      0.475749  |             0 |        0.601971 |
| lorenz96_twoscale_obs_slow0            | observability        |           0   | slow0                      | ngrc_takens_rg_true |      0.650472  |             0 |        1.66674  |
| lorenz96_twoscale_obs_sparse_slowproj  | observability        |           1   | sparse_slow_projection     | ngrc_raw            |      1.45541   |             0 |        4.02089  |
| lorenz96_twoscale_obs_sparse_slowproj  | observability        |           1   | sparse_slow_projection     | ngrc_rg_readout     |      1.39257   |             0 |        5.43188  |
| lorenz96_twoscale_obs_sparse_slowproj  | observability        |           1   | sparse_slow_projection     | ngrc_sf_rg_gated    |      0.896542  |             0 |        3.89752  |
| lorenz96_twoscale_obs_sparse_slowproj  | observability        |           1   | sparse_slow_projection     | ngrc_takens_rg_true |      0.835671  |             0 |        2.24809  |
| lorenz96_twoscale_obs_mixed_projection | observability        |           2   | slow_fast_mixed_projection | ngrc_raw            |      0.73855   |             0 |        1.88663  |
| lorenz96_twoscale_obs_mixed_projection | observability        |           2   | slow_fast_mixed_projection | ngrc_rg_readout     |      0.823314  |             0 |        2.22418  |
| lorenz96_twoscale_obs_mixed_projection | observability        |           2   | slow_fast_mixed_projection | ngrc_sf_rg_gated    |      0.594634  |             0 |        1.16666  |
| lorenz96_twoscale_obs_mixed_projection | observability        |           2   | slow_fast_mixed_projection | ngrc_takens_rg_true |      0.683103  |             0 |        1.71507  |
| lorenz96_twoscale_gate_s2f_0p0         | slow_gating_strength |           0   | slow0                      | ngrc_raw            |      1.46848   |             0 |        4.4385   |
| lorenz96_twoscale_gate_s2f_0p0         | slow_gating_strength |           0   | slow0                      | ngrc_rg_readout     |      1.6346    |             0 |        5.10496  |
| lorenz96_twoscale_gate_s2f_0p0         | slow_gating_strength |           0   | slow0                      | ngrc_sf_rg_gated    |      0.459837  |             0 |        2.09985  |
| lorenz96_twoscale_gate_s2f_0p0         | slow_gating_strength |           0   | slow0                      | ngrc_takens_rg_true |      0.975204  |             0 |        4.7499   |
| lorenz96_twoscale_gate_s2f_0p4         | slow_gating_strength |           0.4 | slow0                      | ngrc_raw            |      2.79017   |             0 |        6.04969  |
| lorenz96_twoscale_gate_s2f_0p4         | slow_gating_strength |           0.4 | slow0                      | ngrc_rg_readout     |      1.89573   |             0 |        5.12947  |
| lorenz96_twoscale_gate_s2f_0p4         | slow_gating_strength |           0.4 | slow0                      | ngrc_sf_rg_gated    |      2.38331   |             0 |        5.18752  |
| lorenz96_twoscale_gate_s2f_0p4         | slow_gating_strength |           0.4 | slow0                      | ngrc_takens_rg_true |      2.11148   |             0 |        5.45512  |
| lorenz96_twoscale_gate_s2f_0p8         | slow_gating_strength |           0.8 | slow0                      | ngrc_raw            |      0.636198  |             0 |        1.25328  |
| lorenz96_twoscale_gate_s2f_0p8         | slow_gating_strength |           0.8 | slow0                      | ngrc_rg_readout     |      0.691981  |             0 |        1.20399  |
| lorenz96_twoscale_gate_s2f_0p8         | slow_gating_strength |           0.8 | slow0                      | ngrc_sf_rg_gated    |      0.0324106 |             0 |        0.507283 |
| lorenz96_twoscale_gate_s2f_0p8         | slow_gating_strength |           0.8 | slow0                      | ngrc_takens_rg_true |      0.649348  |             0 |        1.43709  |
| lorenz96_twoscale_gate_s2f_1p2         | slow_gating_strength |           1.2 | slow0                      | ngrc_raw            |      0.859456  |             0 |        2.40642  |
| lorenz96_twoscale_gate_s2f_1p2         | slow_gating_strength |           1.2 | slow0                      | ngrc_rg_readout     |      0.289167  |             0 |        0.551939 |
| lorenz96_twoscale_gate_s2f_1p2         | slow_gating_strength |           1.2 | slow0                      | ngrc_sf_rg_gated    |      0.475749  |             0 |        0.601971 |
| lorenz96_twoscale_gate_s2f_1p2         | slow_gating_strength |           1.2 | slow0                      | ngrc_takens_rg_true |      0.650472  |             0 |        1.66674  |
| lorenz96_twoscale_gate_s2f_1p6         | slow_gating_strength |           1.6 | slow0                      | ngrc_raw            |      0.0251629 |             0 |        0.351479 |
| lorenz96_twoscale_gate_s2f_1p6         | slow_gating_strength |           1.6 | slow0                      | ngrc_rg_readout     |      0.0798148 |             0 |        0.3941   |
| lorenz96_twoscale_gate_s2f_1p6         | slow_gating_strength |           1.6 | slow0                      | ngrc_sf_rg_gated    |      0.0970608 |             0 |        0.835505 |
| lorenz96_twoscale_gate_s2f_1p6         | slow_gating_strength |           1.6 | slow0                      | ngrc_takens_rg_true |      0.0986245 |             0 |        0.759214 |

Task-wise winners on the boundary sweep:

| task                                   | variant             |   rmse@50_mean |   rmse@100_mean |
|:---------------------------------------|:--------------------|---------------:|----------------:|
| lorenz96_twoscale_gate_s2f_0p0         | ngrc_sf_rg_gated    |      0.459837  |        2.09985  |
| lorenz96_twoscale_gate_s2f_0p4         | ngrc_rg_readout     |      1.89573   |        5.12947  |
| lorenz96_twoscale_gate_s2f_0p8         | ngrc_sf_rg_gated    |      0.0324106 |        0.507283 |
| lorenz96_twoscale_gate_s2f_1p2         | ngrc_rg_readout     |      0.289167  |        0.551939 |
| lorenz96_twoscale_gate_s2f_1p6         | ngrc_raw            |      0.0251629 |        0.351479 |
| lorenz96_twoscale_obs_mixed_projection | ngrc_sf_rg_gated    |      0.594634  |        1.16666  |
| lorenz96_twoscale_obs_slow0            | ngrc_rg_readout     |      0.289167  |        0.551939 |
| lorenz96_twoscale_obs_sparse_slowproj  | ngrc_takens_rg_true |      0.835671  |        2.24809  |

## Delay Sufficiency Sweep

| task                                   | variant                    |   seed_count |   rmse@50_mean |   rmse@50_std |   rmse@100_mean |
|:---------------------------------------|:---------------------------|-------------:|---------------:|--------------:|----------------:|
| lorenz96_twoscale_gate_s2f_0p8         | ngrc_raw_d12_s1            |            3 |      0.636198  |             0 |        1.25328  |
| lorenz96_twoscale_gate_s2f_0p8         | ngrc_raw_d16_s1            |            3 |      0.572698  |             0 |        0.985515 |
| lorenz96_twoscale_gate_s2f_0p8         | ngrc_raw_d24_s1            |            3 |      0.0300044 |             0 |        0.452327 |
| lorenz96_twoscale_gate_s2f_0p8         | ngrc_raw_d4_s1             |            3 |      0.805332  |             0 |        1.96461  |
| lorenz96_twoscale_gate_s2f_0p8         | ngrc_raw_d8_s1             |            3 |      0.583039  |             0 |        1.1597   |
| lorenz96_twoscale_gate_s2f_0p8         | ngrc_rg_readout_d12_s1     |            3 |      0.691981  |             0 |        1.20399  |
| lorenz96_twoscale_gate_s2f_0p8         | ngrc_rg_readout_d16_s1     |            3 |      0.634248  |             0 |        1.00384  |
| lorenz96_twoscale_gate_s2f_0p8         | ngrc_rg_readout_d24_s1     |            3 |      0.057797  |             0 |        0.787238 |
| lorenz96_twoscale_gate_s2f_0p8         | ngrc_rg_readout_d4_s1      |            3 |      0.713295  |             0 |        1.39455  |
| lorenz96_twoscale_gate_s2f_0p8         | ngrc_rg_readout_d8_s1      |            3 |      0.619236  |             0 |        1.06626  |
| lorenz96_twoscale_gate_s2f_0p8         | ngrc_takens_rg_true_d12_s1 |            3 |      0.454234  |             0 |        0.837235 |
| lorenz96_twoscale_gate_s2f_0p8         | ngrc_takens_rg_true_d16_s1 |            3 |      0.626587  |             0 |        1.2731   |
| lorenz96_twoscale_gate_s2f_0p8         | ngrc_takens_rg_true_d24_s1 |            3 |      0.100438  |             0 |        0.850068 |
| lorenz96_twoscale_gate_s2f_0p8         | ngrc_takens_rg_true_d4_s1  |            3 |      0.397064  |             0 |        0.828624 |
| lorenz96_twoscale_gate_s2f_0p8         | ngrc_takens_rg_true_d8_s1  |            3 |      0.384277  |             0 |        0.829409 |
| lorenz96_twoscale_obs_mixed_projection | ngrc_raw_d12_s1            |            3 |      0.927212  |             0 |        2.78268  |
| lorenz96_twoscale_obs_mixed_projection | ngrc_raw_d16_s1            |            3 |      0.73855   |             0 |        1.88663  |
| lorenz96_twoscale_obs_mixed_projection | ngrc_raw_d24_s1            |            3 |      0.73056   |             0 |        1.63768  |
| lorenz96_twoscale_obs_mixed_projection | ngrc_raw_d4_s1             |            3 |      1.5127    |             0 |        3.36636  |
| lorenz96_twoscale_obs_mixed_projection | ngrc_raw_d8_s1             |            3 |      0.976334  |             0 |        2.73757  |
| lorenz96_twoscale_obs_mixed_projection | ngrc_rg_readout_d12_s1     |            3 |      0.823314  |             0 |        2.22418  |
| lorenz96_twoscale_obs_mixed_projection | ngrc_rg_readout_d16_s1     |            3 |      0.901444  |             0 |        2.33034  |
| lorenz96_twoscale_obs_mixed_projection | ngrc_rg_readout_d24_s1     |            3 |      0.898548  |             0 |        2.05379  |
| lorenz96_twoscale_obs_mixed_projection | ngrc_rg_readout_d4_s1      |            3 |      1.29966   |             0 |        3.12384  |
| lorenz96_twoscale_obs_mixed_projection | ngrc_rg_readout_d8_s1      |            3 |      0.849803  |             0 |        2.33314  |
| lorenz96_twoscale_obs_mixed_projection | ngrc_takens_rg_true_d12_s1 |            3 |      0.899382  |             0 |        2.86675  |
| lorenz96_twoscale_obs_mixed_projection | ngrc_takens_rg_true_d16_s1 |            3 |      0.683103  |             0 |        1.71507  |
| lorenz96_twoscale_obs_mixed_projection | ngrc_takens_rg_true_d24_s1 |            3 |      0.681261  |             0 |        1.48147  |
| lorenz96_twoscale_obs_mixed_projection | ngrc_takens_rg_true_d4_s1  |            3 |      1.82553   |             0 |        4.55263  |
| lorenz96_twoscale_obs_mixed_projection | ngrc_takens_rg_true_d8_s1  |            3 |      0.928995  |             0 |        2.67179  |
| lorenz96_twoscale_obs_slow0            | ngrc_raw_d12_s1            |            3 |      0.859456  |             0 |        2.40642  |
| lorenz96_twoscale_obs_slow0            | ngrc_raw_d16_s1            |            3 |      0.742137  |             0 |        2.14128  |
| lorenz96_twoscale_obs_slow0            | ngrc_raw_d24_s1            |            3 |      0.809558  |             0 |        2.36913  |
| lorenz96_twoscale_obs_slow0            | ngrc_raw_d4_s1             |            3 |      1.4796    |             0 |        3.76723  |
| lorenz96_twoscale_obs_slow0            | ngrc_raw_d8_s1             |            3 |      0.94203   |             0 |        2.54588  |
| lorenz96_twoscale_obs_slow0            | ngrc_rg_readout_d12_s1     |            3 |      0.289167  |             0 |        0.551939 |
| lorenz96_twoscale_obs_slow0            | ngrc_rg_readout_d16_s1     |            3 |      0.320786  |             0 |        0.709846 |
| lorenz96_twoscale_obs_slow0            | ngrc_rg_readout_d24_s1     |            3 |      0.433496  |             0 |        0.590337 |
| lorenz96_twoscale_obs_slow0            | ngrc_rg_readout_d4_s1      |            3 |      0.7831    |             0 |        2.41312  |
| lorenz96_twoscale_obs_slow0            | ngrc_rg_readout_d8_s1      |            3 |      0.382508  |             0 |        0.742737 |
| lorenz96_twoscale_obs_slow0            | ngrc_takens_rg_true_d12_s1 |            3 |      0.482535  |             0 |        1.18751  |
| lorenz96_twoscale_obs_slow0            | ngrc_takens_rg_true_d16_s1 |            3 |      0.650472  |             0 |        1.66674  |
| lorenz96_twoscale_obs_slow0            | ngrc_takens_rg_true_d24_s1 |            3 |      0.74934   |             0 |        2.03827  |
| lorenz96_twoscale_obs_slow0            | ngrc_takens_rg_true_d4_s1  |            3 |      1.5515    |             0 |        3.90232  |
| lorenz96_twoscale_obs_slow0            | ngrc_takens_rg_true_d8_s1  |            3 |      0.622107  |             0 |        1.74933  |

Best delay setting per model family:

| task                                   | ngrc_raw_best_variant   |   ngrc_raw_best_rmse@50_mean | ngrc_rg_readout_best_variant   |   ngrc_rg_readout_best_rmse@50_mean | ngrc_takens_rg_true_best_variant   |   ngrc_takens_rg_true_best_rmse@50_mean |   takens_minus_raw_best |
|:---------------------------------------|:------------------------|-----------------------------:|:-------------------------------|------------------------------------:|:-----------------------------------|----------------------------------------:|------------------------:|
| lorenz96_twoscale_gate_s2f_0p8         | ngrc_raw_d24_s1         |                    0.0300044 | ngrc_rg_readout_d24_s1         |                            0.057797 | ngrc_takens_rg_true_d24_s1         |                                0.100438 |               0.0704337 |
| lorenz96_twoscale_obs_mixed_projection | ngrc_raw_d24_s1         |                    0.73056   | ngrc_rg_readout_d12_s1         |                            0.823314 | ngrc_takens_rg_true_d24_s1         |                                0.681261 |              -0.0492989 |
| lorenz96_twoscale_obs_slow0            | ngrc_raw_d16_s1         |                    0.742137  | ngrc_rg_readout_d12_s1         |                            0.289167 | ngrc_takens_rg_true_d12_s1         |                                0.482535 |              -0.259602  |

## Robustness Validation

Additional validation on clean/noisy/volatility-clustered classic tasks and multiscale noise-profile tasks:

| task                                      | task_family                  | task_regime                      | variant                 |   seed_count |   rmse@50_mean |   rmse@50_std |   rmse@100_mean |   acf_rmse_mean |   psd_rmse_mean |
|:------------------------------------------|:-----------------------------|:---------------------------------|:------------------------|-------------:|---------------:|--------------:|----------------:|----------------:|----------------:|
| fitzhugh_nagumo_classic_clean             | classic_fastslow             | excitable_clean_partial          | ngrc_raw                |            3 |     0.00500111 |    0          |       2.03483   |       0.937189  |      0.0712191  |
| fitzhugh_nagumo_classic_clean             | classic_fastslow             | excitable_clean_partial          | ngrc_rg_readout         |            3 |     0.0660049  |    0          |       5.51162   |       0.929536  |      0.0695115  |
| fitzhugh_nagumo_classic_clean             | classic_fastslow             | excitable_clean_partial          | ngrc_sf_rg_gated        |            3 |     0.00573935 |    0          |       2.22584   |       0.962851  |      0.0715106  |
| fitzhugh_nagumo_classic_clean             | classic_fastslow             | excitable_clean_partial          | ngrc_takens_rg_additive |            3 |     0.0174545  |    0          |       2.06437   |       0.9365    |      0.0722367  |
| fitzhugh_nagumo_classic_clean             | classic_fastslow             | excitable_clean_partial          | ngrc_takens_rg_true     |            3 |     0.0526482  |    0          |       3.90058   |       0.941606  |      0.0732543  |
| fitzhugh_nagumo_classic_noisy             | classic_fastslow             | excitable_noisy_partial          | ngrc_raw                |            3 |     0.0962377  |    0.0655392  |       0.447278  |       0.325264  |      0.0329528  |
| fitzhugh_nagumo_classic_noisy             | classic_fastslow             | excitable_noisy_partial          | ngrc_rg_readout         |            3 |     0.0910248  |    0.0802339  |       0.534445  |       0.0642308 |      0.0161231  |
| fitzhugh_nagumo_classic_noisy             | classic_fastslow             | excitable_noisy_partial          | ngrc_sf_rg_gated        |            3 |     0.0763619  |    0.0772912  |       0.301699  |       0.437125  |      0.0341365  |
| fitzhugh_nagumo_classic_noisy             | classic_fastslow             | excitable_noisy_partial          | ngrc_takens_rg_additive |            3 |     0.0946832  |    0.0824077  |       0.501062  |       0.0655082 |      0.0185415  |
| fitzhugh_nagumo_classic_noisy             | classic_fastslow             | excitable_noisy_partial          | ngrc_takens_rg_true     |            3 |     0.104601   |    0.122049   |       0.568517  |       0.112299  |      0.0228104  |
| hindmarsh_rose_bursting_clean             | classic_fastslow             | bursting_clean_partial           | ngrc_raw                |            3 |     0.232695   |    0          |       3.5769    |       1.09565   |      0.0726855  |
| hindmarsh_rose_bursting_clean             | classic_fastslow             | bursting_clean_partial           | ngrc_rg_readout         |            3 |     0.107908   |    0          |       0.489021  |       0.0707107 |      0.0536232  |
| hindmarsh_rose_bursting_clean             | classic_fastslow             | bursting_clean_partial           | ngrc_sf_rg_gated        |            3 |     0.0663336  |    0          |       0.209216  |       1.05477   |      0.0559254  |
| hindmarsh_rose_bursting_clean             | classic_fastslow             | bursting_clean_partial           | ngrc_takens_rg_additive |            3 |     0.133886   |    0          |       1.26731   |       1.07785   |      0.0621471  |
| hindmarsh_rose_bursting_clean             | classic_fastslow             | bursting_clean_partial           | ngrc_takens_rg_true     |            3 |     0.0136519  |    0          |       0.0334858 |       0.113314  |      0.00919787 |
| hindmarsh_rose_bursting_noisy             | classic_fastslow             | bursting_noisy_partial           | ngrc_raw                |            3 |     0.174603   |    0.0917821  |       0.50877   |       0.252458  |      0.0292286  |
| hindmarsh_rose_bursting_noisy             | classic_fastslow             | bursting_noisy_partial           | ngrc_rg_readout         |            3 |     0.17346    |    0.203924   |       0.572885  |       0.200133  |      0.0262363  |
| hindmarsh_rose_bursting_noisy             | classic_fastslow             | bursting_noisy_partial           | ngrc_sf_rg_gated        |            3 |     0.158509   |    0.135065   |       0.576106  |       0.186741  |      0.020422   |
| hindmarsh_rose_bursting_noisy             | classic_fastslow             | bursting_noisy_partial           | ngrc_takens_rg_additive |            3 |     0.159417   |    0.0858388  |       0.485968  |       0.248752  |      0.0430841  |
| hindmarsh_rose_bursting_noisy             | classic_fastslow             | bursting_noisy_partial           | ngrc_takens_rg_true     |            3 |     0.112355   |    0.127534   |       0.345773  |       0.234267  |      0.0227385  |
| vanderpol_relaxation_clean                | classic_fastslow             | relaxation_clean_partial         | ngrc_raw                |            3 |     0.276932   |    0          |       8.83144   |       1.07229   |      0.0754198  |
| vanderpol_relaxation_clean                | classic_fastslow             | relaxation_clean_partial         | ngrc_rg_readout         |            3 |     0.381185   |    0          |       9.02877   |       1.0755    |      0.0757613  |
| vanderpol_relaxation_clean                | classic_fastslow             | relaxation_clean_partial         | ngrc_sf_rg_gated        |            3 |     0.60152    |    0          |       9.52837   |       1.07895   |      0.0821502  |
| vanderpol_relaxation_clean                | classic_fastslow             | relaxation_clean_partial         | ngrc_takens_rg_additive |            3 |     1.50665    |    0          |       9.76242   |       1.06382   |      0.0761276  |
| vanderpol_relaxation_clean                | classic_fastslow             | relaxation_clean_partial         | ngrc_takens_rg_true     |            3 |     0.108042   |    0          |       7.22613   |       1.04827   |      0.0778859  |
| vanderpol_relaxation_noisy                | classic_fastslow             | relaxation_noisy_partial         | ngrc_raw                |            3 |     0.0481144  |    0.031054   |       0.121356  |       0.0292482 |      0.00618834 |
| vanderpol_relaxation_noisy                | classic_fastslow             | relaxation_noisy_partial         | ngrc_rg_readout         |            3 |     0.0262099  |    0.010723   |       0.046542  |       0.0274486 |      0.00711544 |
| vanderpol_relaxation_noisy                | classic_fastslow             | relaxation_noisy_partial         | ngrc_sf_rg_gated        |            3 |     0.0278728  |    0.00763416 |       0.0506549 |       0.211069  |      0.0258077  |
| vanderpol_relaxation_noisy                | classic_fastslow             | relaxation_noisy_partial         | ngrc_takens_rg_additive |            3 |     0.0216488  |    0.00545533 |       0.0324906 |       0.0206529 |      0.00419894 |
| vanderpol_relaxation_noisy                | classic_fastslow             | relaxation_noisy_partial         | ngrc_takens_rg_true     |            3 |     0.0215328  |    0.00727087 |       0.0332546 |       0.0205817 |      0.00424696 |
| fitzhugh_nagumo_classic_volclustered      | classic_fastslow_finance     | excitable_volclustered_partial   | ngrc_raw                |            3 |     0.137225   |    0.10481    |       0.483331  |       0.0376923 |      0.00474414 |
| fitzhugh_nagumo_classic_volclustered      | classic_fastslow_finance     | excitable_volclustered_partial   | ngrc_rg_readout         |            3 |     0.201308   |    0.0786729  |       0.667302  |       0.0480095 |      0.0122253  |
| fitzhugh_nagumo_classic_volclustered      | classic_fastslow_finance     | excitable_volclustered_partial   | ngrc_sf_rg_gated        |            3 |     0.215287   |    0.17569    |       0.684682  |       0.57808   |      0.0470108  |
| fitzhugh_nagumo_classic_volclustered      | classic_fastslow_finance     | excitable_volclustered_partial   | ngrc_takens_rg_additive |            3 |     0.135712   |    0.129624   |       0.485863  |       0.0279436 |      0.00954576 |
| fitzhugh_nagumo_classic_volclustered      | classic_fastslow_finance     | excitable_volclustered_partial   | ngrc_takens_rg_true     |            3 |     0.200397   |    0.166078   |       0.641251  |       0.223066  |      0.0253886  |
| hindmarsh_rose_bursting_volclustered      | classic_fastslow_finance     | bursting_volclustered_partial    | ngrc_raw                |            3 |     0.0771236  |    0.0439074  |       0.225996  |       0.271134  |      0.0382014  |
| hindmarsh_rose_bursting_volclustered      | classic_fastslow_finance     | bursting_volclustered_partial    | ngrc_rg_readout         |            3 |     0.11856    |    0.114689   |       0.45417   |       0.296172  |      0.0233715  |
| hindmarsh_rose_bursting_volclustered      | classic_fastslow_finance     | bursting_volclustered_partial    | ngrc_sf_rg_gated        |            3 |     0.052024   |    0.0178802  |       0.0940554 |       0.287083  |      0.023727   |
| hindmarsh_rose_bursting_volclustered      | classic_fastslow_finance     | bursting_volclustered_partial    | ngrc_takens_rg_additive |            3 |     0.080611   |    0.0550225  |       0.228017  |       0.280814  |      0.029047   |
| hindmarsh_rose_bursting_volclustered      | classic_fastslow_finance     | bursting_volclustered_partial    | ngrc_takens_rg_true     |            3 |     0.0511485  |    0.019799   |       0.105499  |       0.259046  |      0.019698   |
| vanderpol_relaxation_volclustered         | classic_fastslow_finance     | relaxation_volclustered_partial  | ngrc_raw                |            3 |     0.532564   |    0.825312   |       0.865406  |       0.171541  |      0.01457    |
| vanderpol_relaxation_volclustered         | classic_fastslow_finance     | relaxation_volclustered_partial  | ngrc_rg_readout         |            3 |     0.505159   |    0.816402   |       0.727023  |       0.29205   |      0.026436   |
| vanderpol_relaxation_volclustered         | classic_fastslow_finance     | relaxation_volclustered_partial  | ngrc_sf_rg_gated        |            3 |     0.456947   |    0.723042   |       0.675485  |       0.175156  |      0.0187549  |
| vanderpol_relaxation_volclustered         | classic_fastslow_finance     | relaxation_volclustered_partial  | ngrc_takens_rg_additive |            3 |     0.521406   |    0.848197   |       0.799209  |       0.202756  |      0.0141653  |
| vanderpol_relaxation_volclustered         | classic_fastslow_finance     | relaxation_volclustered_partial  | ngrc_takens_rg_true     |            3 |     0.556491   |    0.871328   |       0.857983  |       0.17186   |      0.0199282  |
| lorenz96_twoscale_noise_homoskedastic     | highdim_multiscale_mechanism | hetero_control_homoskedastic     | ngrc_raw                |            3 |     6.91632    |   11.475      |       8.67227   |       0.308793  |      0.0503784  |
| lorenz96_twoscale_noise_homoskedastic     | highdim_multiscale_mechanism | hetero_control_homoskedastic     | ngrc_rg_readout         |            3 |     6.92006    |   11.4422     |       8.2789    |       0.278688  |      0.0410389  |
| lorenz96_twoscale_noise_homoskedastic     | highdim_multiscale_mechanism | hetero_control_homoskedastic     | ngrc_sf_rg_gated        |            3 |     6.83785    |   10.1547     |       8.30483   |       0.072432  |      0.0279287  |
| lorenz96_twoscale_noise_homoskedastic     | highdim_multiscale_mechanism | hetero_control_homoskedastic     | ngrc_takens_rg_additive |            3 |     5.22185    |    8.50751    |       6.91953   |       0.158261  |      0.0447344  |
| lorenz96_twoscale_noise_homoskedastic     | highdim_multiscale_mechanism | hetero_control_homoskedastic     | ngrc_takens_rg_true     |            3 |     5.76153    |    9.51473    |       7.16567   |       0.346085  |      0.0658386  |
| lorenz96_twoscale_noise_matched_clustered | highdim_multiscale_mechanism | hetero_control_matched_clustered | ngrc_raw                |            3 |     5.53372    |    9.01234    |       8.20871   |       0.28219   |      0.0406361  |
| lorenz96_twoscale_noise_matched_clustered | highdim_multiscale_mechanism | hetero_control_matched_clustered | ngrc_rg_readout         |            3 |     7.71595    |   12.747      |       9.14589   |       0.285251  |      0.0363109  |
| lorenz96_twoscale_noise_matched_clustered | highdim_multiscale_mechanism | hetero_control_matched_clustered | ngrc_sf_rg_gated        |            3 |     6.94185    |   11.12       |       9.12407   |       0.295361  |      0.0304111  |
| lorenz96_twoscale_noise_matched_clustered | highdim_multiscale_mechanism | hetero_control_matched_clustered | ngrc_takens_rg_additive |            3 |     6.92851    |   11.3676     |       7.9294    |       0.302624  |      0.0453773  |
| lorenz96_twoscale_noise_matched_clustered | highdim_multiscale_mechanism | hetero_control_matched_clustered | ngrc_takens_rg_true     |            3 |    12.3693     |   10.4615     |      13.7111    |       0.523747  |      0.0584539  |

Best variant per robustness task:

| task                                      | variant                 |   rmse@50_mean |   rmse@100_mean |   acf_rmse_mean |   psd_rmse_mean |
|:------------------------------------------|:------------------------|---------------:|----------------:|----------------:|----------------:|
| fitzhugh_nagumo_classic_clean             | ngrc_raw                |     0.00500111 |       2.03483   |       0.937189  |      0.0712191  |
| fitzhugh_nagumo_classic_noisy             | ngrc_sf_rg_gated        |     0.0763619  |       0.301699  |       0.437125  |      0.0341365  |
| fitzhugh_nagumo_classic_volclustered      | ngrc_takens_rg_additive |     0.135712   |       0.485863  |       0.0279436 |      0.00954576 |
| hindmarsh_rose_bursting_clean             | ngrc_takens_rg_true     |     0.0136519  |       0.0334858 |       0.113314  |      0.00919787 |
| hindmarsh_rose_bursting_noisy             | ngrc_takens_rg_true     |     0.112355   |       0.345773  |       0.234267  |      0.0227385  |
| hindmarsh_rose_bursting_volclustered      | ngrc_takens_rg_true     |     0.0511485  |       0.105499  |       0.259046  |      0.019698   |
| lorenz96_twoscale_noise_homoskedastic     | ngrc_takens_rg_additive |     5.22185    |       6.91953   |       0.158261  |      0.0447344  |
| lorenz96_twoscale_noise_matched_clustered | ngrc_raw                |     5.53372    |       8.20871   |       0.28219   |      0.0406361  |
| vanderpol_relaxation_clean                | ngrc_takens_rg_true     |     0.108042   |       7.22613   |       1.04827   |      0.0778859  |
| vanderpol_relaxation_noisy                | ngrc_takens_rg_true     |     0.0215328  |       0.0332546 |       0.0205817 |      0.00424696 |
| vanderpol_relaxation_volclustered         | ngrc_sf_rg_gated        |     0.456947   |       0.675485  |       0.175156  |      0.0187549  |

## Coordinate Diagnostics

| task                                   | coordinate     |   seed_count |   markov_gain_ratio_mean |   koopman_invariance_score_mean |   spectral_radius_rmse_mean |   spectral_radius_corr_mean |
|:---------------------------------------|:---------------|-------------:|-------------------------:|--------------------------------:|----------------------------:|----------------------------:|
| fitzhugh_nagumo_classic_noisy          | delay          |            3 |               0.00261333 |                        0.996778 |                   0.0627397 |                   0.148056  |
| fitzhugh_nagumo_classic_noisy          | delay_rg_joint |            3 |               0.116215   |                        0.987504 |                   0.068486  |                   0.0586399 |
| fitzhugh_nagumo_classic_noisy          | fastslow       |            3 |               0.122319   |                        0.99199  |                   0.0971497 |                   0.173854  |
| fitzhugh_nagumo_classic_noisy          | raw            |            3 |               0.0977072  |                        0.982598 |                   0.4667    |                   0.039235  |
| fitzhugh_nagumo_classic_noisy          | rg             |            3 |               0.09548    |                        0.973325 |                   0.607512  |                   0.135919  |
| lorenz96_twoscale_gate_s2f_0p0         | delay          |            3 |               0.165617   |                        0.999942 |                   0.0419653 |                  -0.0486515 |
| lorenz96_twoscale_gate_s2f_0p0         | delay_rg_joint |            3 |               0.820742   |                        0.979636 |                   0.103076  |                   0.185669  |
| lorenz96_twoscale_gate_s2f_0p0         | fastslow       |            3 |               0.90645    |                        0.997597 |                   0.0560157 |                   0.118088  |
| lorenz96_twoscale_gate_s2f_0p0         | raw            |            3 |               0.938576   |                        0.968974 |                   0.400591  |                  -0.170679  |
| lorenz96_twoscale_gate_s2f_0p0         | rg             |            3 |               0.892195   |                        0.969775 |                   0.0853857 |                  -0.198622  |
| lorenz96_twoscale_gate_s2f_0p8         | delay          |            3 |              -0.224404   |                        0.999961 |                   0.2786    |                   0.111548  |
| lorenz96_twoscale_gate_s2f_0p8         | delay_rg_joint |            3 |               0.714543   |                        0.982456 |                   0.257146  |                  -0.0489715 |
| lorenz96_twoscale_gate_s2f_0p8         | fastslow       |            3 |               0.827692   |                        0.998748 |                   0.284908  |                   0.131311  |
| lorenz96_twoscale_gate_s2f_0p8         | raw            |            3 |               0.973678   |                        0.969834 |                   0.475199  |                   0.0466403 |
| lorenz96_twoscale_gate_s2f_0p8         | rg             |            3 |               0.909125   |                        0.956382 |                   0.241424  |                  -0.179143  |
| lorenz96_twoscale_obs_mixed_projection | delay          |            3 |               0.0768067  |                        0.999977 |                   0.499083  |                  -0.0802403 |
| lorenz96_twoscale_obs_mixed_projection | delay_rg_joint |            3 |               0.708126   |                        0.988262 |                   0.444362  |                   0.0224823 |
| lorenz96_twoscale_obs_mixed_projection | fastslow       |            3 |               0.81294    |                        0.99938  |                   0.503056  |                  -0.146368  |
| lorenz96_twoscale_obs_mixed_projection | raw            |            3 |               0.977221   |                        0.980696 |                   0.621898  |                  -0.16727   |
| lorenz96_twoscale_obs_mixed_projection | rg             |            3 |               0.796398   |                        0.971418 |                   0.483961  |                  -0.105289  |
| lorenz96_twoscale_obs_slow0            | delay          |            3 |               0.0827199  |                        0.999977 |                   0.488601  |                  -0.359176  |
| lorenz96_twoscale_obs_slow0            | delay_rg_joint |            3 |               0.793775   |                        0.985752 |                   0.450457  |                   0.0553157 |
| lorenz96_twoscale_obs_slow0            | fastslow       |            3 |               0.83139    |                        0.99919  |                   0.486846  |                  -0.0491187 |
| lorenz96_twoscale_obs_slow0            | raw            |            3 |               0.96967    |                        0.979783 |                   0.664153  |                   0.109789  |
| lorenz96_twoscale_obs_slow0            | rg             |            3 |               0.890438   |                        0.972065 |                   0.409443  |                  -0.0580249 |
| vanderpol_relaxation_noisy             | delay          |            3 |               0.00171468 |                        0.997395 |                   0.0399378 |                   0.225299  |
| vanderpol_relaxation_noisy             | delay_rg_joint |            3 |               0.0130718  |                        0.990773 |                   0.0971069 |                   0.0222315 |
| vanderpol_relaxation_noisy             | fastslow       |            3 |               0.0900278  |                        0.993652 |                   0.0782969 |                   0.265981  |
| vanderpol_relaxation_noisy             | raw            |            3 |               0.0191555  |                        0.98793  |                   0.485946  |                  -0.0770664 |
| vanderpol_relaxation_noisy             | rg             |            3 |               0.0297184  |                        0.98851  |                   0.131364  |                  -0.0646761 |

Best coordinate by each dynamical metric:

| task                                   | metric                   | winner         |   winner_value |
|:---------------------------------------|:-------------------------|:---------------|---------------:|
| fitzhugh_nagumo_classic_noisy          | best_closure_coordinate  | delay          |     0.00261333 |
| fitzhugh_nagumo_classic_noisy          | best_koopman_coordinate  | delay          |     0.996778   |
| fitzhugh_nagumo_classic_noisy          | best_spectral_coordinate | delay          |     0.0627397  |
| lorenz96_twoscale_gate_s2f_0p0         | best_closure_coordinate  | delay          |     0.165617   |
| lorenz96_twoscale_gate_s2f_0p0         | best_koopman_coordinate  | delay          |     0.999942   |
| lorenz96_twoscale_gate_s2f_0p0         | best_spectral_coordinate | delay          |     0.0419653  |
| lorenz96_twoscale_gate_s2f_0p8         | best_closure_coordinate  | delay          |    -0.224404   |
| lorenz96_twoscale_gate_s2f_0p8         | best_koopman_coordinate  | delay          |     0.999961   |
| lorenz96_twoscale_gate_s2f_0p8         | best_spectral_coordinate | rg             |     0.241424   |
| lorenz96_twoscale_obs_mixed_projection | best_closure_coordinate  | delay          |     0.0768067  |
| lorenz96_twoscale_obs_mixed_projection | best_koopman_coordinate  | delay          |     0.999977   |
| lorenz96_twoscale_obs_mixed_projection | best_spectral_coordinate | delay_rg_joint |     0.444362   |
| lorenz96_twoscale_obs_slow0            | best_closure_coordinate  | delay          |     0.0827199  |
| lorenz96_twoscale_obs_slow0            | best_koopman_coordinate  | delay          |     0.999977   |
| lorenz96_twoscale_obs_slow0            | best_spectral_coordinate | rg             |     0.409443   |
| vanderpol_relaxation_noisy             | best_closure_coordinate  | delay          |     0.00171468 |
| vanderpol_relaxation_noisy             | best_koopman_coordinate  | delay          |     0.997395   |
| vanderpol_relaxation_noisy             | best_spectral_coordinate | delay          |     0.0399378  |

## Interpretation Notes

- If `delay` keeps the best Markov/Koopman scores while `ngrc_takens_rg_true` wins some prediction tasks, that supports the conditioner view rather than the state-replacement view.
- If `ngrc_takens_rg_true` consistently beats `lagged` and `random` controls, the gain is RG-specific rather than a generic extra-feature effect.
- If the interaction variant beats the additive residual on mixed or intermediate-coupling tasks, that supports the operator-conditioning hypothesis.
- If the Takens-RG gain shrinks as delay grows, RG is acting partly as a finite-delay regularizer; if it survives at large delay, it is acting more like a regime-conditioned operator.
