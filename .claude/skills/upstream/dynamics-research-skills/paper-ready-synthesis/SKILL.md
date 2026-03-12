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
