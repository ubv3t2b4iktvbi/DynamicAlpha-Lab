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
