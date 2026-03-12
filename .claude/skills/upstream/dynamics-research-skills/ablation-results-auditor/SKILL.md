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
