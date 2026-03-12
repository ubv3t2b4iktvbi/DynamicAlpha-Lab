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
