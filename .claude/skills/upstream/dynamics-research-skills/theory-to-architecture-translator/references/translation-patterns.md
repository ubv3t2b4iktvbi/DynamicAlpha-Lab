# Translation Patterns

Useful mappings:
- multiscale architecture <-> slow-fast or coarse-fine decomposition
- encoder macro branch <-> order parameter or reduced state
- decoder refinement <-> reconstruction of fine detail conditioned on coarse latent state
- control injection at each scale <-> conditioned forcing or correction field
- RC / reservoir path <-> fast approximate state evolution under latency constraints
- renormalization-style coarse-graining <-> scale-wise aggregation with parameter flow across resolutions
- SINDy / sparse identification <-> explicit law discovery for latent or macro dynamics
- trigger-carry logic <-> control policy acting on state transitions or regime switches

Every mapping needs:
1. a concrete module or loss
2. an expected empirical signature
3. a discriminative validation experiment
