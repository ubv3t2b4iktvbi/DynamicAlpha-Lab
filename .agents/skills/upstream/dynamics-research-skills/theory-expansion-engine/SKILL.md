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
