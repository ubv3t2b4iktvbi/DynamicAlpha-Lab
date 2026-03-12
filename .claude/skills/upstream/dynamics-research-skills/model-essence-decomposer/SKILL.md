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
