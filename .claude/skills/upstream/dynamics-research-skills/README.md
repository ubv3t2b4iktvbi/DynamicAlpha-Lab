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
