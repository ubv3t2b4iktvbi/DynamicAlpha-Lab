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
