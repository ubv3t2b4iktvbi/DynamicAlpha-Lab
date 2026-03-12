# Divergence-Convergence Policy

Use **convergent mode** when:
- the result table is messy or incomplete
- multiple claims are being mixed together
- the team is overclaiming from one ablation table
- the next step should be a discriminative experiment rather than more brainstorming

Use **divergent mode** when:
- current explanations are narrow or all variants of the same story
- the current module decomposition may be hiding a deeper mechanism
- the user explicitly asks for related theory, deeper mechanism, or new directions
- the current evidence supports several non-equivalent explanations

Recommended cycle:
1. Inventory inputs and current claims.
2. Run convergent analysis to extract observations only.
3. Run divergent analysis to generate 3-7 theory-grounded hypotheses.
4. Return to convergent mode to design targeted experiments.
5. Update the claim ledger and prioritize the next cycle.

Promotion rule:
- observation -> hypothesis: allowed when a mechanism is proposed
- hypothesis -> candidate insight: allowed when theory and supportive evidence align
- candidate insight -> insight: allowed only after targeted validation and alternative-explanation checks

Stop a divergence round when:
- three distinct mechanisms have already been proposed
- new hypotheses are only paraphrases of older ones
- no new falsifiable prediction is being produced
