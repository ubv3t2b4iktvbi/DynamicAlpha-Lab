# Ordering Policy for Multi-Experiment Ablations

Preferred ordering:
1. lower-dimensional / lower-order / simpler cases
2. intermediate cases
3. higher-dimensional / higher-order / more complex cases

If several orderings are possible, prefer the one that best tracks:
- state-space dimension
- dynamical complexity
- number of active modules
- nonlinearity level
- memory depth
- noise level
- interaction richness

If filenames encode settings, build a table with:
- experiment id
- parsed tags
- inferred complexity level
- notes about ambiguity

If the order cannot be inferred reliably, say so explicitly and present two candidate orderings rather than silently inventing one.
