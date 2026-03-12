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
