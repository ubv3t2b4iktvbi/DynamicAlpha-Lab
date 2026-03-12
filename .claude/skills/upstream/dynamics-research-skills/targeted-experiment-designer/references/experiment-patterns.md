# High-Value Experiment Patterns

Use the smallest experiment that can separate competing explanations.

Useful patterns:
- add-back study: reintroduce a removed module after an ablation
- partial-strength sweep: vary the degree of a component instead of only on/off
- capacity-matched control: compare with a control that preserves parameter count or compute
- regime sweep: vary dimension, complexity, horizon, noise, or data sparsity
- interaction test: 2x2 or small factorial to test synergy or redundancy
- perturbation test: probe stability and robustness with controlled disturbances
- constraint test: enforce or relax a theoretical constraint to see whether the prediction flips

Each experiment should specify:
- manipulated factors
- fixed factors
- metrics
- expected signature under each hypothesis
- decision rule
