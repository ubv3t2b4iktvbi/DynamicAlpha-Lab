#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-dynamics-research-skills}"
mkdir -p "$OUT_DIR"

write_file() {
  local path="$1"
  mkdir -p "$(dirname "$path")"
  cat > "$path"
}

write_file "$OUT_DIR/LICENSE.txt" <<'EOF'
MIT License

Copyright (c) 2026 OpenAI ChatGPT

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

write_file "$OUT_DIR/README.md" <<'EOF'
# Dynamics Research Skills

A portable workflow-oriented skill pack for dynamical-systems-flavored AI research.

This pack is not centered on any single architecture or application domain. The goal here is to abstract the **research process** itself into reusable skills, in the spirit of an AI-research skill library but specialized for:
- dynamical systems
- multiscale modeling
- macro-state discovery
- theory-guided model redesign
- ablation-driven validation

## Included skills
- `research-loop-orchestrator`
- `ablation-results-auditor`
- `insight-evidence-grader`
- `theory-expansion-engine`
- `targeted-experiment-designer`
- `paper-ready-synthesis`
- `model-essence-decomposer`
- `theory-to-architecture-translator`
- `macro-state-dimension-reducer`

## Workflow view
1. audit all experiments and extract observations
2. grade claims so that “insight” is reserved for theory-backed and validated claims
3. extract the architecture-independent essence of the current model
4. design a better macro-state / latent reduction if needed
5. expand the mechanism search space with theory
6. translate theory objects into concrete architecture, loss, control, or identification modules
7. design targeted experiments to validate the redesign
8. orchestrate the loop across multiple cycles
9. synthesize results in paper-ready prose

## Design principles
- Analyze all experiments, not only pairwise comparisons.
- Prefer low-dimensional to high-dimensional and simple-to-complex ordering.
- Distinguish observation, hypothesis, candidate insight, and insight.
- Alternate divergent theory expansion with convergent validation.
- Extract model essence before redesigning surface architecture.
- Use explicit macro-state or latent reduction when macro equations, physics-informed training, or information-theoretic design become central.
- Return to module-level validation whenever a mechanism claim is made.

## Install notes
Each top-level skill directory is already a valid Agent Skills skill.
You can copy individual skill folders into a skills directory for your agent, or upload them where supported.

Common locations used by skills-compatible tools:
- project scope: `.github/skills/` or `.claude/skills/`
- user scope: `~/.copilot/skills/` or `~/.claude/skills/`

## Validate locally
```bash
python3 validate_skills.py .
```

## Regenerate the whole pack
```bash
bash generate_skills.sh
```
EOF

write_file "$OUT_DIR/WORKFLOW_MAP.md" <<'EOF'
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
EOF

write_file "$OUT_DIR/ablation-results-auditor/SKILL.md" <<'EOF'
---
name: ablation-results-auditor
description: Use this skill when the user wants a full analysis of an ablation study with many experiments, especially when experiment settings are encoded in filenames or .npz names, and the analysis must cover all experiments from low-dimensional to high-dimensional or simple to complex rather than only comparing two rows.
license: MIT
compatibility: Designed for skills-compatible agents that can read experiment tables, filenames, notes, and optional result artifacts. No network access is required. Optional Python or shell access helps inspect files such as .npz outputs.
metadata:
  author: OpenAI ChatGPT
  version: "1.1.0"
  pack: dynamics-research-skills
---

# Ablation Results Auditor

This skill analyzes a complete ablation study without collapsing it into a superficial baseline-vs-one-ablation comparison.

Use [the ordering policy](references/analysis-order.md) when the experiment names or `.npz` files encode the settings.

## Goal
Produce a mathematically grounded and dynamics-aware reading of **all** experiments.

## Required inputs
Prefer to have:
- the full-model formulation or equation
- an explanation of each module
- metric definitions
- a result table or parsed outputs
- experiment names or `.npz` filenames
- any notes about dimension, regime, or complexity

If some inputs are missing, proceed with placeholders instead of inventing details.

## Mandatory workflow
### Step 1: Inventory every experiment
Build a table with:
- experiment id / filename
- changed component
- parsed tags from the name
- inferred dimension or complexity level
- metrics
- comments

Do not skip experiments. Do not focus only on the best and worst rows.

### Step 2: Order the study
Organize the analysis from:
- low dimension -> high dimension
- simple regime -> complex regime

If the ordering is ambiguous, say so and provide the rationale for the chosen ordering.

### Step 3: Restate the full model
Write the full model mathematically.
For each module, explain:
- where it enters the formula
- what role it plays
- which inductive bias it introduces

### Step 4: Analyze each ablation mechanistically
For every experiment:
1. state what changed
2. show how the mathematical object changed
3. explain the expected effect on expressivity, stability, memory, robustness, identifiability, or optimization
4. connect the expected effect to the observed metrics

### Step 5: Read the results through dynamics
When the task involves dynamical systems, explicitly discuss:
- stability
- attractor or trajectory structure
- memory depth
- nonlinearity handling
- noise sensitivity
- regime dependence

When the task is not a dynamical system, map these ideas to the nearest valid analogues.

### Step 6: Separate evidence from interpretation
Produce:
- an **observations** section: only what is directly supported by the results
- a **hypotheses** section: interpretations that explain the observations

Do not upgrade to insight here.

## Output template
1. Experiment inventory
2. Ordering rationale
3. Full-model formulation
4. Per-experiment analysis
5. Cross-experiment patterns
6. Observations
7. Hypotheses suggested by the observations
8. Missing evidence

## Guardrails
- Analyze every experiment in the comparison set.
- Use equations and mechanism, not only metric narration.
- Treat unexpected reversals and weak effects as important data.
- Small effect size does not imply a useless module; it may indicate redundancy, saturation, or regime-specific relevance.
EOF

write_file "$OUT_DIR/ablation-results-auditor/references/analysis-order.md" <<'EOF'
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
EOF

write_file "$OUT_DIR/insight-evidence-grader/SKILL.md" <<'EOF'
---
name: insight-evidence-grader
description: Use this skill when the user wants to decide whether a claim from an ablation or modeling study is merely an observation, a hypothesis, or a true insight backed by both theory and targeted experiments. It grades claims, downgrades overstatements, and lists the missing evidence required for promotion.
license: MIT
compatibility: Designed for skills-compatible agents that can read experiment tables, filenames, notes, and optional result artifacts. No network access is required. Optional Python or shell access helps inspect files such as .npz outputs.
metadata:
  author: OpenAI ChatGPT
  version: "1.1.0"
  pack: dynamics-research-skills
---

# Insight Evidence Grader

This skill enforces a strict distinction between observations, hypotheses, candidate insights, and insights.

See [the claim taxonomy](references/claim-taxonomy.md).

## Goal
Prevent the agent from calling something an “insight” unless it is supported by both:
- a theory-linked explanation
- targeted empirical validation

## Scoring rubric
Score each claim on three axes from 0 to 2.

### A. Theory support
- 0 = no theory-level mechanism
- 1 = plausible explanation, but not tightly linked to the formulation or dynamics
- 2 = explicit mechanism connected to equations, modules, or dynamical properties

### B. Experimental support
- 0 = only anecdotal or indirect support
- 1 = supportive empirical pattern, but not from a discriminative experiment
- 2 = targeted experiment with a prediction that clearly bears on the claim

### C. Alternative-explanation control
- 0 = alternatives not considered
- 1 = alternatives mentioned but not stress-tested
- 2 = alternatives explicitly checked or strongly constrained

## Promotion rules
- observation: direct pattern, even with no explanation
- hypothesis: mechanism proposed but not sufficiently validated
- candidate insight: theory support is strong, empirical support is suggestive, but at least one decisive check is missing
- insight: scores of 2/2/2 or equivalent qualitative evidence
- rejected: the claim fails its key prediction or is superseded by a better explanation

## Required workflow
1. List each candidate claim separately.
2. Restate the exact evidence supporting it.
3. Score the claim on the three axes.
4. Assign the correct status.
5. State the minimal next experiment needed to promote or reject it.

## Output template
| Claim | Theory support | Experimental support | Alternative control | Status | Why | Missing evidence |
|---|---:|---:|---:|---|---|---|

After the table, produce:
1. claims that are safe to write as observations
2. claims that are promising hypotheses
3. claims that are genuine insights
4. claims that should be removed from the narrative

## Guardrails
- “Interesting” does not mean “insight.”
- A theory story without validation is a hypothesis.
- A table pattern without a mechanism is an observation.
- Never hide uncertainty by rewriting a weak claim in more polished prose.
EOF

write_file "$OUT_DIR/insight-evidence-grader/references/claim-taxonomy.md" <<'EOF'
# Claim Taxonomy

## Observation
A statement directly supported by the current measurements.
Example:
- "Removing nonlinear terms increases error on the high-complexity regime."

## Hypothesis
A candidate explanation of an observation.
Example:
- "The nonlinear terms matter because the target regime requires higher-order interactions."

## Candidate insight
A hypothesis that has a coherent theory-level explanation and some supportive empirical evidence, but still lacks a sharp discriminative validation step or a clean alternative-explanation check.

## Insight
A claim that satisfies all three:
1. a mechanism-level explanation linked to the model formulation
2. targeted experimental support, not only post-hoc reading of the same ablation table
3. explicit consideration of plausible alternative explanations

## Rejected claim
A claim whose key prediction fails or whose evidence is better explained by an alternative account.
EOF

write_file "$OUT_DIR/macro-state-dimension-reducer/SKILL.md" <<'EOF'
---
name: macro-state-dimension-reducer
description: Use this skill when the user wants a better dimensionality reduction from high-dimensional states or model features to macro variables, order parameters, or latent coordinates that can support macro equations, physics-informed training, or information-theoretic objectives. It covers encoder-decoder setups, slow-fast separation, macro-state discovery, and physically meaningful latent design.
license: MIT
compatibility: Designed for skills-compatible agents that can read experiment tables, filenames, notes, and optional result artifacts. No network access is required. Optional Python or shell access helps inspect files such as .npz outputs.
metadata:
  author: OpenAI ChatGPT
  version: "1.1.0"
  pack: dynamics-research-skills
---

# Macro State Dimension Reducer

This skill designs a principled reduction from a high-dimensional state to a smaller latent or macro representation.

See [the reduction criteria](references/reduction-criteria.md).

## Goal
Find a reduced representation that is not only compact, but also useful for dynamics, control, reconstruction, and interpretation.

## Required workflow
### Step 1: Define the purpose of reduction
Say which of these matter most:
- prediction
- control
- macro-equation learning
- interpretability
- compression
- sparse-observation robustness
- latency

### Step 2: Define what the reduced state must preserve
Specify:
- invariances or symmetries
- slow variables or order parameters
- closure requirements for the macro dynamics
- what the decoder must reconstruct
- which nuisance information should be discarded

### Step 3: Propose candidate reductions
Consider a small set of candidates, such as:
- linear projection or PCA-style baseline
- supervised encoder with task-sufficient latent state
- autoencoder or variational autoencoder
- information bottleneck encoder
- slow-feature or timescale-separation encoder
- Koopman / observable-based latent state
- RG-style coarse-grained state across scales

### Step 4: Attach the macro dynamics
For each candidate, say how the macro state will evolve:
- explicit ODE/PDE or closure law
- neural state transition
- sparse identified dynamics
- hybrid equation + learned residual

### Step 5: Score candidates
Evaluate each candidate on:
- sufficiency for the downstream task
- compression / minimality
- predictive closure of the macro dynamics
- stability across regimes
- decoder fidelity if reconstruction matters
- control usefulness if intervention matters
- scientific interpretability

## Output template
| Candidate reduction | Latent meaning | Encoder objective | Macro dynamics model | Decoder role | Strengths | Risks | Validation |
|---|---|---|---|---|---|---|---|

Then provide:
1. the recommended reduced state
2. the recommended encoder / decoder split
3. the minimal ablation suite needed to verify that the reduction is meaningful
4. what would prove the latent is merely compressive but not mechanistic

## Guardrails
- The best low-dimensional latent is not always the best reconstruction latent.
- A compact latent is not enough; it should support useful macro dynamics or control.
- Do not pick a reduction family before defining the preservation targets.
- Treat “order parameter” as an earned status, not a name for any low-dimensional feature.
EOF

write_file "$OUT_DIR/macro-state-dimension-reducer/references/reduction-criteria.md" <<'EOF'
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
EOF

write_file "$OUT_DIR/manifest.txt" <<'EOF'
LICENSE.txt
README.md
WORKFLOW_MAP.md
ablation-results-auditor/SKILL.md
ablation-results-auditor/references/analysis-order.md
generate_skills.sh
insight-evidence-grader/SKILL.md
insight-evidence-grader/references/claim-taxonomy.md
macro-state-dimension-reducer/SKILL.md
macro-state-dimension-reducer/references/reduction-criteria.md
manifest.txt
model-essence-decomposer/SKILL.md
model-essence-decomposer/references/essence-schema.md
paper-ready-synthesis/SKILL.md
paper-ready-synthesis/references/writing-templates.md
research-loop-orchestrator/SKILL.md
research-loop-orchestrator/references/loop-policy.md
targeted-experiment-designer/SKILL.md
targeted-experiment-designer/references/experiment-patterns.md
theory-expansion-engine/SKILL.md
theory-expansion-engine/references/theory-lenses.md
theory-to-architecture-translator/SKILL.md
theory-to-architecture-translator/references/translation-patterns.md
validate_skills.py
EOF

write_file "$OUT_DIR/model-essence-decomposer/SKILL.md" <<'EOF'
---
name: model-essence-decomposer
description: Use this skill when the user wants the AI to extract the architecture-independent essence of a model, separate essential mechanisms from accidental implementation details, and identify redesign levers. It is useful for moving from implementation details toward scale separation, latent evolution, control injection, or other deeper functional abstractions.
license: MIT
compatibility: Designed for skills-compatible agents that can read experiment tables, filenames, notes, and optional result artifacts. No network access is required. Optional Python or shell access helps inspect files such as .npz outputs.
metadata:
  author: OpenAI ChatGPT
  version: "1.1.0"
  pack: dynamics-research-skills
---

# Model Essence Decomposer

This skill extracts what the model fundamentally needs to do before proposing new architecture.

See [the essence schema](references/essence-schema.md).

## Goal
Convert a concrete model description into a smaller set of architecture-independent functions, constraints, and mechanisms.

## Required workflow
### Step 1: Separate objective from implementation
Write down:
- task objective
- observables and hidden state
- what must be predicted, controlled, or reconstructed
- hard constraints: physics, causality, latency, sparsity, memory, interpretability

### Step 2: Extract the core functions
Identify the minimum set of functions the model appears to implement, such as:
- multiscale aggregation
- slow-fast separation
- macro-state extraction
- local-to-global message passing
- control-conditioned correction
- online law identification
- robustness to sparse or noisy observations

### Step 3: Separate essence from accident
For each observed module, classify it as one of:
- essential mechanism
- implementation choice
- optimization convenience
- legacy carry-over
- unexplained component

### Step 4: Build a redesign map
For each essential mechanism, propose 2-3 alternative realizations.
These alternatives can come from:
- architecture changes
- objective changes
- macro-variable design
- control interfaces
- online identification modules

### Step 5: State what must be preserved
Explicitly say which learned capability or evidence-backed effect cannot be lost during redesign.

## Output template
1. Objective / constraints
2. Essential mechanisms
3. Non-essential implementation details
4. Candidate redesign levers
5. Evidence that each essential mechanism is real
6. Missing experiments needed to confirm the decomposition

## Guardrails
- Do not confuse a familiar architecture block with a fundamental mechanism.
- Do not rename the same idea three times and call that abstraction.
- Do not propose redesign until the essential function is stated clearly.
EOF

write_file "$OUT_DIR/model-essence-decomposer/references/essence-schema.md" <<'EOF'
# Essence Extraction Schema

Ask these questions:
1. What information must flow from input to output?
2. What information can be compressed into a macro-state?
3. Which scales must interact?
4. Which components carry memory?
5. Where is control or conditioning injected?
6. Which constraints are task-defining versus implementation-specific?
7. Which observed gains are robust across regimes?

Strong essence statements are short and architecture-independent.
Example pattern:
- "The model needs a coarse macro-state with slow dynamics, plus scale-local corrective injections that restore fine detail under fast disturbances."
EOF

write_file "$OUT_DIR/paper-ready-synthesis/SKILL.md" <<'EOF'
---
name: paper-ready-synthesis
description: Use this skill when the user wants to turn the outcome of the ablation-analysis loop into a paper-ready narrative that clearly separates observations, hypotheses, and validated insights. It writes conservative scientific prose without collapsing uncertainty.
license: MIT
compatibility: Designed for skills-compatible agents that can read experiment tables, filenames, notes, and optional result artifacts. No network access is required. Optional Python or shell access helps inspect files such as .npz outputs.
metadata:
  author: OpenAI ChatGPT
  version: "1.1.0"
  pack: dynamics-research-skills
---

# Paper Ready Synthesis

This skill writes results and discussion text after the research loop has already separated evidence classes.

See [the writing templates](references/writing-templates.md).

## Goal
Produce scientific prose that is clear, mechanistic, and honest about evidence.

## Required workflow
1. Gather the claim ledger.
2. Separate observations, hypotheses, and validated insights.
3. Write each class with the appropriate level of certainty.
4. Summarize the new research directions, but mark them as forward-looking rather than validated.

## Required sections
### Results
- summarize the ordered ablation findings
- mention all relevant groups, not just a single pairwise comparison
- keep direct observations separate from explanations

### Mechanistic interpretation
- connect validated claims to equations, modules, or dynamics
- mention unresolved alternatives when they still matter

### Follow-up directions
- list the most credible new directions derived from the validated insights and strongest hypotheses
- state what experiment would be needed next

## Output formats you may provide
- paper-style paragraph
- discussion subsection
- rebuttal-ready summary
- concise bullet-free executive synthesis

## Guardrails
- Never write a hypothesis as if it were already validated.
- Never suppress uncertainty for stylistic smoothness.
- Prefer exact causal language only when the evidence supports it.
- Keep terminology consistent across observation, hypothesis, and insight.
EOF

write_file "$OUT_DIR/paper-ready-synthesis/references/writing-templates.md" <<'EOF'
# Writing Templates

Recommended claim ladder in writing:
- "We observe that ..."
- "One plausible explanation is ..."
- "This hypothesis predicts ..."
- "Targeted validation shows ..."
- "Taken together, these results support the insight that ..."

Avoid:
- "These experiments prove ..."
- "This definitively shows ..." unless the evidence is unusually strong
- mixing observations and interpretations in the same sentence when status matters

Good discussion structure:
1. what was observed
2. what mechanism might explain it
3. what targeted experiment tested that mechanism
4. what can now be stated as an insight
5. what remains open
EOF

write_file "$OUT_DIR/research-loop-orchestrator/SKILL.md" <<'EOF'
---
name: research-loop-orchestrator
description: Use this skill when the user wants an iterative research workflow for ablation studies, insight generation, model-essence extraction, theory expansion, redesign, and follow-up experiments. It alternates convergent analysis and divergent exploration, keeps claim status explicit, and decides when to summarize observations, when to widen the search space, and when to return to modular validation.
license: MIT
compatibility: Designed for skills-compatible agents that can read experiment tables, filenames, notes, and optional result artifacts. No network access is required. Optional Python or shell access helps inspect files such as .npz outputs.
metadata:
  author: OpenAI ChatGPT
  version: "1.1.0"
  pack: dynamics-research-skills
---

# Research Loop Orchestrator

This skill controls a full research loop:

Observation -> Hypothesis -> Targeted Experiment -> Insight -> Redesign -> Revalidation

The main job is not to produce clever prose. The main job is to decide **which mode should run now** and to keep claims from being upgraded too early.

See [the loop policy](references/loop-policy.md) for mode-switch rules.

## When this skill should dominate
Use it when the user asks for:
- a complete ablation-study workflow
- theory-guided next steps after experiments
- a decision about when to brainstorm and when to validate
- an iteration policy for moving between analysis and experiment design

## Core rules
1. Never convert a post-hoc pattern directly into an insight.
2. Keep every claim in one of these states: observation, hypothesis, candidate insight, insight, rejected.
3. Separate **divergent work** from **convergent work**.
4. Prefer a small number of falsifiable hypotheses over a long list of vague ideas.
5. When evidence is weak, compress first and branch later.
6. When the current search space is too narrow, abstract upward to mechanism, then come back down to module-level tests.

## Required workflow
### Step 1: Build the current state
Create a compact research state with:
- model name and task
- full-model equation or symbolic formulation
- module inventory
- experiment inventory
- existing claims
- unresolved contradictions
- current bottleneck

### Step 2: Decide the mode
Choose one of:
- **Convergent mode**: clean up results, compress claims, remove overreach
- **Divergent mode**: expand the mechanism search space using theory
- **Validation mode**: design targeted experiments that discriminate among competing hypotheses
- **Essence mode**: extract architecture-independent model essence before redesign
- **Translation mode**: translate theory objects into architecture, objective, or control changes
- **Reduction mode**: design a macro-state or latent reduction before writing a macro equation

Explain the choice in 2-4 sentences.

### Step 3: Execute the chosen mode
#### Convergent mode
- restate the mathematical object being studied
- summarize all experiments, not just the best two rows
- extract observations only
- separate evidence from interpretation
- downgrade weak claims

#### Divergent mode
- generate 3-7 hypotheses only
- connect each hypothesis to a theory lens
- give each hypothesis a prediction signature
- give each hypothesis a falsifier
- do not declare any of them true

#### Validation mode
- design minimal discriminative experiments
- specify manipulated factors, controls, metrics, and expected signatures
- tie every experiment back to a concrete module or mechanism

#### Essence / Translation / Reduction modes
- identify what the model fundamentally needs to do
- identify which theoretical objects or macro variables could implement that need
- propose only redesigns that preserve the evidence already earned

### Step 4: Update the claim ledger
For each claim, provide:
- claim text
- current status
- evidence for
- evidence missing
- next experiment needed for promotion or rejection

### Step 5: End the cycle with a priority list
Rank the next 3 actions by expected information gain, not by convenience alone.

## Output template
1. Current research state
2. Chosen mode and why
3. Execution output
4. Updated claim ledger
5. Top next actions

## Guardrails
- Do not let “related theory” become decorative name-dropping.
- Do not let “insight” become a synonym for “interesting observation.”
- Do not keep brainstorming once a sharp validation experiment is obvious.
- Do not keep validating a hypothesis that no longer has a distinctive prediction.
EOF

write_file "$OUT_DIR/research-loop-orchestrator/references/loop-policy.md" <<'EOF'
# Divergence-Convergence Policy

Use **convergent mode** when:
- the result table is messy or incomplete
- multiple claims are being mixed together
- the team is overclaiming from one ablation table
- the next step should be a discriminative experiment rather than more brainstorming

Use **divergent mode** when:
- current explanations are narrow or all variants of the same story
- the current module decomposition may be hiding a deeper mechanism
- the user explicitly asks for related theory, deeper mechanism, or new directions
- the current evidence supports several non-equivalent explanations

Recommended cycle:
1. Inventory inputs and current claims.
2. Run convergent analysis to extract observations only.
3. Run divergent analysis to generate 3-7 theory-grounded hypotheses.
4. Return to convergent mode to design targeted experiments.
5. Update the claim ledger and prioritize the next cycle.

Promotion rule:
- observation -> hypothesis: allowed when a mechanism is proposed
- hypothesis -> candidate insight: allowed when theory and supportive evidence align
- candidate insight -> insight: allowed only after targeted validation and alternative-explanation checks

Stop a divergence round when:
- three distinct mechanisms have already been proposed
- new hypotheses are only paraphrases of older ones
- no new falsifiable prediction is being produced
EOF

write_file "$OUT_DIR/targeted-experiment-designer/SKILL.md" <<'EOF'
---
name: targeted-experiment-designer
description: Use this skill when the user wants follow-up experiments that discriminate between competing hypotheses, validate a proposed insight modularly, or return from theory to module-level testing. It designs minimal, controlled, and information-rich experiments rather than generic ablations.
license: MIT
compatibility: Designed for skills-compatible agents that can read experiment tables, filenames, notes, and optional result artifacts. No network access is required. Optional Python or shell access helps inspect files such as .npz outputs.
metadata:
  author: OpenAI ChatGPT
  version: "1.1.0"
  pack: dynamics-research-skills
---

# Targeted Experiment Designer

This skill converts hypotheses into validation experiments.

Use [the experiment patterns](references/experiment-patterns.md) as defaults.

## Goal
Design experiments that tell the team **which explanation is correct**, not merely whether performance can move up or down.

## Core rules
1. One experiment should answer one sharp question whenever possible.
2. Control obvious confounders: parameter count, compute, data split, optimizer, and training budget.
3. Prefer minimal experiments with strong discriminative power.
4. Tie every experiment back to a module, mechanism, or theoretical prediction.
5. State what result would count against the favored hypothesis.

## Required workflow
### Step 1: State the hypotheses to separate
Name the competing explanations explicitly.

### Step 2: Choose the experiment type
Select the smallest pattern that can separate them:
- add-back
- sweep
- matched control
- regime stress test
- interaction test
- perturbation test

### Step 3: Specify the protocol
For each experiment provide:
- objective
- manipulated factors
- fixed factors
- datasets or regimes
- metrics
- predicted result if H1 is correct
- predicted result if H2 or null is correct
- failure mode or ambiguity risk
- success / stop rule

### Step 4: Return to modular validation
Map the experiment back to the module inventory:
- which module is being validated
- which mechanism it is supposed to implement
- whether the result would support necessity, sufficiency, synergy, redundancy, or regime dependence

## Output template
| Experiment | Question | Manipulation | Controls | Metrics | Prediction if H1 | Prediction if H2 / null | What it validates | Priority |
|---|---|---|---|---|---|---|---|---:|

After the table, add:
1. the smallest must-run experiment
2. the highest-information regime sweep
3. the experiment most likely to falsify the current favorite story

## Guardrails
- Avoid “more ablations” as an answer. Design a discriminative test.
- Do not change multiple uncontrolled factors at once.
- Do not confuse a performance gain with mechanism validation.
- Prefer experiments that reveal whether an effect is necessary, sufficient, synergistic, or regime-specific.
EOF

write_file "$OUT_DIR/targeted-experiment-designer/references/experiment-patterns.md" <<'EOF'
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
EOF

write_file "$OUT_DIR/theory-expansion-engine/SKILL.md" <<'EOF'
---
name: theory-expansion-engine
description: Use this skill when the user wants to expand beyond the current ablation table, connect the results to related theory, search for deeper mechanisms, and propose new research directions without being trapped by the current module decomposition. It is the divergent phase of the loop.
license: MIT
compatibility: Designed for skills-compatible agents that can read experiment tables, filenames, notes, and optional result artifacts. No network access is required. Optional Python or shell access helps inspect files such as .npz outputs.
metadata:
  author: OpenAI ChatGPT
  version: "1.1.0"
  pack: dynamics-research-skills
---

# Theory Expansion Engine

This skill is the **divergent** stage. Its job is to widen the mechanism search space without losing rigor.

See [the theory lenses reference](references/theory-lenses.md).

## Goal
Move from “what happened in this ablation table?” to “what deeper mechanism might explain the pattern, and what new direction follows from that mechanism?”

## Core rules
1. Use theory to generate mechanisms, not decoration.
2. Propose a small set of distinct hypotheses, not a long speculative list.
3. Every hypothesis must produce a falsifiable prediction.
4. Every new direction must be traceable back to a mechanism.
5. Do not declare any expanded idea to be true.

## Required workflow
### Step 1: Start from constrained evidence
Begin with the observations already supported by experiments.

### Step 2: Choose theory lenses
Select 2-4 relevant theory lenses.
Do not apply all possible lenses blindly.

### Step 3: Generate 3-7 hypotheses
For each hypothesis, provide:
- the mechanism statement
- the theory lens
- the architecture -> formula -> dynamics -> observable chain
- the falsifier
- the most informative next experiment
- the new research direction that would follow if the hypothesis survives

### Step 4: Force distinctness
Remove hypotheses that are mere paraphrases of each other.
Keep only hypotheses that differ in mechanism or prediction.

## Output template
| Hypothesis | Theory lens | Mechanism chain | Predicted signature | Falsifier | New direction if supported |
|---|---|---|---|---|---|

Then add:
1. a short section called **What this expands beyond the current ablation space**
2. a short section called **What would make the search too unconstrained**
3. a prioritized list of which hypotheses deserve immediate validation

## Guardrails
- Do not produce generic “future work.”
- Do not import a theory lens unless it changes the prediction structure.
- Do not let related theory substitute for experimental design.
- Keep the hypothesis count limited and high-value.
EOF

write_file "$OUT_DIR/theory-expansion-engine/references/theory-lenses.md" <<'EOF'
# Theory Lenses for Expansion

Use only the lenses that genuinely fit the task.

Candidate lenses:
- approximation / expressivity
- dynamical systems: stability, attractors, bifurcations, memory
- optimization and implicit bias
- representation learning and invariance
- information bottlenecks and compression
- sparsity, identifiability, and feature library design
- scale separation, slow-fast structure, and multiscale effects
- symmetry, conservation, and physical constraints
- robustness, perturbation response, and noise sensitivity

For each chosen lens, convert theory into:
1. a mechanism statement
2. a predicted observable signature
3. a falsifier
EOF

write_file "$OUT_DIR/theory-to-architecture-translator/SKILL.md" <<'EOF'
---
name: theory-to-architecture-translator
description: Use this skill when the user wants to convert theory objects such as slow-fast structure, order parameters, renormalization-group ideas, macro equations, control signals, or online system identification into concrete modules, losses, conditioning paths, or architecture changes. It also works in the reverse direction: from architecture back to theory.
license: MIT
compatibility: Designed for skills-compatible agents that can read experiment tables, filenames, notes, and optional result artifacts. No network access is required. Optional Python or shell access helps inspect files such as .npz outputs.
metadata:
  author: OpenAI ChatGPT
  version: "1.1.0"
  pack: dynamics-research-skills
---

# Theory to Architecture Translator

This skill translates between abstract theoretical objects and concrete model design.

See [the translation patterns](references/translation-patterns.md).

## Goal
Move cleanly between:
- theory objects
- macro variables / equations
- architecture modules
- objectives and losses
- experiments that validate the translation

## Required workflow
### Step 1: Name the theory object clearly
Examples:
- slow-fast decomposition
- order parameter
- renormalization-group flow
- macro equation / closure law
- control signal / conditional forcing
- online sparse law identification

### Step 2: Map it to design roles
For each theory object, state whether it should appear as:
- latent state
- encoder or decoder target
- explicit equation module
- multiscale block
- per-scale conditioning injection
- loss term or regularizer
- online adaptation module

### Step 3: Generate architecture candidates
Produce a small set of candidates with exact roles for each module.
Example directions:
- multiscale model -> generator with explicit coarse-to-fine decomposition
- encoder-decoder -> encoder learns macro variables or order parameters, decoder reconstructs fine detail
- RC-like fast path -> low-latency state update for the fast component
- ControlNet-style injection -> conditional corrections injected at each scale
- SINDy-like head -> online identification of local laws or regime-specific drift terms

### Step 4: State the validation obligations
For each candidate, specify what evidence would show that the translation is real rather than just an analogy.

## Output template
| Theory object | Design role | Candidate module / loss | What it should improve | What would falsify the mapping |
|---|---|---|---|---|

Then add:
1. a compact architecture sketch in words
2. the minimal ablation or add-back needed to validate the translation
3. which parts are essential and which are optional

## Guardrails
- Do not import a theory object unless it changes a concrete design choice.
- Do not treat analogy as validation.
- Keep the mapping reversible: architecture -> theory and theory -> architecture should both be explainable.
EOF

write_file "$OUT_DIR/theory-to-architecture-translator/references/translation-patterns.md" <<'EOF'
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
EOF

write_file "$OUT_DIR/validate_skills.py" <<'EOF'
#!/usr/bin/env python3
import re
import sys
from pathlib import Path

NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def parse_frontmatter(text: str):
    lines = text.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        raise ValueError("SKILL.md must start with YAML frontmatter delimited by ---")
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        raise ValueError("Missing closing --- for YAML frontmatter")
    fm_lines = lines[1:end]
    data = {}
    current_key = None
    for line in fm_lines:
        if not line.strip():
            continue
        if not line.startswith(" ") and ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                data[key] = value.strip('"').strip("'")
                current_key = None
            else:
                current_key = key
                data[current_key] = {}
        elif current_key and line.startswith("  ") and ":" in line:
            key, value = line.strip().split(":", 1)
            data[current_key][key.strip()] = value.strip().strip('"').strip("'")
        else:
            pass
    return data, end + 1


def validate_skill(skill_dir: Path):
    errors = []
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return [f"{skill_dir}: missing SKILL.md"]
    text = skill_md.read_text(encoding="utf-8")
    try:
        data, body_start = parse_frontmatter(text)
    except Exception as e:
        return [f"{skill_dir.name}: invalid frontmatter: {e}"]

    name = data.get("name", "")
    desc = data.get("description", "")

    if not name:
        errors.append(f"{skill_dir.name}: missing required frontmatter field 'name'")
    if not desc:
        errors.append(f"{skill_dir.name}: missing required frontmatter field 'description'")

    if name:
        if len(name) > 64:
            errors.append(f"{skill_dir.name}: name longer than 64 chars")
        if not NAME_RE.fullmatch(name):
            errors.append(f"{skill_dir.name}: name must match ^[a-z0-9]+(?:-[a-z0-9]+)*$")
        if name != skill_dir.name:
            errors.append(f"{skill_dir.name}: frontmatter name '{name}' does not match directory name")

    if desc and len(desc) > 1024:
        errors.append(f"{skill_dir.name}: description longer than 1024 chars")

    line_count = len(text.splitlines())
    if line_count > 500:
        errors.append(f"{skill_dir.name}: SKILL.md has {line_count} lines; keep under ~500 recommended limit")

    body = text.splitlines()[body_start:]
    if not any(line.strip() for line in body):
        errors.append(f"{skill_dir.name}: SKILL.md body is empty")

    refs = skill_dir / "references"
    if refs.exists() and not refs.is_dir():
        errors.append(f"{skill_dir.name}: references exists but is not a directory")

    return errors


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    skill_dirs = sorted([p for p in root.iterdir() if p.is_dir() and (p / "SKILL.md").exists()])
    if not skill_dirs:
        print(f"No skill directories with SKILL.md found under {root}")
        sys.exit(1)

    all_errors = []
    for skill_dir in skill_dirs:
        all_errors.extend(validate_skill(skill_dir))

    if all_errors:
        print("Validation failed:")
        for err in all_errors:
            print(f" - {err}")
        sys.exit(1)

    print(f"Validated {len(skill_dirs)} skills successfully.")
    for skill_dir in skill_dirs:
        print(f" - {skill_dir.name}")


if __name__ == "__main__":
    main()
EOF

chmod +x "$OUT_DIR/validate_skills.py" || true
if command -v python3 >/dev/null 2>&1; then
  python3 "$OUT_DIR/validate_skills.py" "$OUT_DIR" || exit 1
fi
if command -v find >/dev/null 2>&1 && command -v sort >/dev/null 2>&1; then
  find "$OUT_DIR" -type f | sed "s#^$OUT_DIR/##" | sort > "$OUT_DIR/manifest.txt"
fi
echo "Generated skill pack at: $OUT_DIR"
