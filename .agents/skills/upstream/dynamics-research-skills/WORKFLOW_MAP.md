# Workflow Map

This repository abstracts a scientific workflow rather than a single model family.

## Stage 1: Evidence compression
Use `ablation-results-auditor` to read all experiments, order them, and separate direct observations from interpretations.

## Stage 2: Claim control
Use `insight-evidence-grader` so that a claim becomes an insight only when it is supported by both theory and targeted validation.

## Stage 3: Essence extraction
Use `model-essence-decomposer` to identify the architecture-independent function of the current model.

## Stage 4: State reduction
Use `macro-state-dimension-reducer` when macro equations, order parameters, slow-fast separation, or encoder-decoder decomposition become central.

## Stage 5: Mechanism expansion
Use `theory-expansion-engine` to widen the search space with falsifiable, theory-grounded hypotheses.

## Stage 6: Redesign translation
Use `theory-to-architecture-translator` to map theory objects into modules, losses, latent states, control interfaces, or identification heads.

## Stage 7: Validation design
Use `targeted-experiment-designer` to return to modular validation.

## Stage 8: Loop management
Use `research-loop-orchestrator` to decide when to converge, diverge, reduce, translate, or validate.

## Stage 9: Writing
Use `paper-ready-synthesis` after the evidence classes are already separated.

## Role of case studies
Case studies such as “multiscale model -> coarse latent dynamics -> validation loop,” or “slow-fast finance -> order parameter -> RG -> factors -> online identification” are examples that exercise this workflow. They should not become hidden assumptions of the skills themselves.
