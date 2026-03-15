# Manifold Factor Theory Map

This note records the geometric structures that recur across many nonlinear systems and the causal factor translations used in `src/fsrc_sindy/factors/factor_bank.py`.

## Shared geometric structures

### 1. Delay-history manifolds

- Core idea: when only a scalar observable is available, the natural state for prediction is a delay or history manifold rather than the hidden physical state itself.
- Source: Takens, `Detecting strange attractors in turbulence` (1981), [DOI](https://doi.org/10.1007/BFb0091924)
- Factor translation:
  - `chart_position_confidence`
  - `phase_chart_consistency`
  - `closure_margin`

### 2. Normally hyperbolic slow manifolds

- Core idea: many slow-fast systems admit persistent invariant manifolds with a tangent / normal splitting; motion along the manifold and escape away from it should be measured separately.
- Source: Fenichel, `Geometric singular perturbation theory for ordinary differential equations` (1979), [DOI](https://doi.org/10.1016/0022-0396(79)90152-9)
- Factor translation:
  - `tangent_flow_confidence`
  - `normal_escape_pressure`
  - `isostable_return_margin`

### 3. Phase-isostable / Koopman chart coordinates

- Core idea: near attractors, phase and isostable coordinates provide a geometry-aware representation in which return-to-attractor and chart reliability can be measured separately.
- Sources:
  - Mauroy and Mezić, `Global stability analysis using the eigenfunctions of the Koopman operator` (2016), [arXiv](https://arxiv.org/abs/1408.1379)
  - Mezić, `Spectrum of the Koopman operator, spectral expansions in functional spaces, and state space geometry` (2017), [arXiv](https://arxiv.org/abs/1702.07597)
  - Wilson and Moehlis, `Isostable reduction of periodic orbits` (2016), [PDF](https://sites.me.ucsb.edu/~moehlis/moehlis_papers/isostable_po.pdf)
- Factor translation:
  - `phase_chart_consistency`
  - `isostable_return_margin`
  - `chart_stability_margin`

### 4. Local tangent-space and principal-manifold geometry

- Core idea: across many datasets, the most reusable local geometry comes from tangent-space quality, chart thinness, and low intrinsic dimensionality.
- Source: Zhang and Zha, `Principal Manifolds and Nonlinear Dimension Reduction via Local Tangent Space Alignment` (2002), [arXiv](https://arxiv.org/abs/cs/0212008)
- Factor translation:
  - `tangent_flow_confidence`
  - `coarse_chart_integrity`
  - `chart_position_confidence`

### 5. Critical transitions and loss of normal hyperbolicity

- Core idea: near tipping points, recovery weakens and noise pushes trajectories more easily across regime boundaries.
- Sources:
  - Scheffer et al., `Early-warning signals for critical transitions` (2009), [Nature](https://www.nature.com/articles/nature08227)
  - Kuehn, `A mathematical framework for critical transitions: Bifurcations, fast-slow systems and stochastic dynamics` (2011), [arXiv](https://arxiv.org/abs/1101.2900)
- Factor translation:
  - `critical_softening_load`
  - `critical_escape_pressure`
  - `drive_off_manifold_pressure`

### 6. Memory closure and unresolved fibers

- Core idea: after reduction, missing variables appear as memory and noise; this is especially important when a coordinate is not yet Markov enough.
- Source: Chorin, Hald, and Kupferman, `Optimal prediction and the Mori-Zwanzig representation of irreversible processes` (2000), [arXiv](https://arxiv.org/abs/physics/0002067)
- Factor translation:
  - `closure_margin`
  - `memory_fiber_load`
  - `normal_escape_pressure`

## Added theory-manifold factors

The current theory pass adds these directly compositional factors without changing `DynamicsFeatureEngine`:

- `chart_position_confidence = slow_level_norm * rg_coarse_grain_score`
- `phase_chart_consistency = band_position * collapse_quality`
- `tangent_flow_confidence = rg_beta_flow * rg_coarse_grain_score`
- `normal_escape_pressure = rg_noise_scale / slow_manifold_alignment`
- `isostable_return_margin = isostable_relaxation * slow_manifold_alignment`
- `closure_margin = adiabatic_coherence / closure_stress`
- `memory_fiber_load = lag1_autocorr * rg_noise_scale`
- `coarse_chart_integrity = collapse_quality * timescale_separation`
- `critical_softening_load = critical_window * lag1_autocorr`
- `critical_escape_pressure = critical_window * rg_noise_scale`
- `drive_off_manifold_pressure = rg_control_parameter * rg_noise_scale`
- `chart_stability_margin = adiabatic_coherence * rg_coarse_grain_score`

## Design rule

Prefer factors that approximate one of the following broad geometric roles:

- `chart_position`
- `tangent_flow`
- `normal_amplitude`
- `closure_memory`
- `coarse_geometry`

Treat the following as important but more task-specific extensions:

- `control_drive`
- `regime_boundary`
- `surprise_alignment`
