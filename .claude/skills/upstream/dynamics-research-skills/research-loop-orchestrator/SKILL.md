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
