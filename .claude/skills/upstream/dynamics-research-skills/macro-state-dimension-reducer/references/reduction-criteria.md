# Reduction Criteria

A good reduction should balance:
- **minimality**: no unnecessary dimensions
- **sufficiency**: enough information for the task
- **closure**: the reduced state should evolve with minimal dependence on hidden leftovers
- **stability**: similar meaning across regimes and retraining runs
- **decodability**: reconstruct what matters, if reconstruction is required
- **control usefulness**: interventions in reduced space should map to meaningful outcome changes
- **interpretability**: the coordinates should admit physical or task semantics when possible

Information-theoretic reading:
- preserve task-relevant information
- compress nuisance or irrelevant detail
- prefer reductions whose coordinates line up with persistent, controllable, or slow structure

Warning signs of a bad reduction:
- excellent reconstruction but poor macro prediction
- task performance depends on hidden decoder hacks
- latent coordinates change meaning across regimes
- no clear closure for the reduced dynamics
