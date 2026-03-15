from __future__ import annotations

from typing import Sequence

import pandas as pd

from .base import CandidateScore, SelectedFactorLibrary
from .factor_bank import factor_formula


def candidate_scores_frame(rows: Sequence[CandidateScore]) -> pd.DataFrame:
    return pd.DataFrame([row.to_dict() for row in rows])


def manual_review_markdown(
    task_name: str,
    identifier_kind: str,
    rows: Sequence[CandidateScore],
    selected_library: SelectedFactorLibrary,
) -> str:
    lines = [
        f"# Manual review queue: {task_name} / {identifier_kind}",
        "",
        "## 1. Selected factor set",
        "",
        f"- baseline rmse@50: {selected_library.baseline_rmse50:.6g}",
        f"- final validation rmse@10: {selected_library.final_rmse10:.6g}",
        f"- final validation rmse@50: {selected_library.final_rmse50:.6g}",
        f"- rollout validation score: {selected_library.rollout_validation_score:.6g}",
        f"- combined validation score: {selected_library.validation_score:.6g}",
        f"- baseline wsga epr score: {selected_library.baseline_wsga_epr_score:.6g}",
        f"- final wsga epr score: {selected_library.final_wsga_epr_score:.6g}",
        f"- library layers: {', '.join(f'{name}={len(items)}' for name, items in selected_library.library_layers.items())}",
        f"- future factor queue: {', '.join(selected_library.future_factor_queue) if selected_library.future_factor_queue else 'none'}",
        f"- curation notes: {selected_library.curation_notes}",
        f"- notes: {selected_library.notes}",
        "",
        "## 2. Factors requiring human review",
        "",
        "Only a small subset is queued for manual inspection: selected factors, top-ranked alternatives, and any factor with physics-identifier dependence.",
        "",
    ]
    review_rows = []
    selected_names = {spec.name for spec in selected_library.selected_factors}
    for row in rows:
        needs_review = row.selected or row.rank <= 8 or row.family == "physics_id" or "physics_identifier" in row.theory_tags
        if needs_review:
            review_rows.append(
                {
                    "rank": row.rank,
                    "selected": row.selected,
                    "factor": row.factor_name,
                    "family": row.family,
                    "role": row.manifold_role,
                    "rmse": row.one_step_rmse,
                    "gain_vs_baseline": row.gain_vs_baseline,
                    "koopman_score": row.koopman_score,
                    "wsga_epr_score": row.wsga_epr_score,
                    "target_corr": row.target_corr,
                    "target_mi": row.target_mutual_info,
                    "redundancy_corr": row.max_redundancy_corr,
                    "redundancy_mi": row.max_redundancy_mutual_info,
                    "effectiveness": row.effectiveness_score,
                    "curation_score": row.curation_score,
                    "tier": row.curation_tier,
                    "formula": row.formula,
                    "finance_origin": row.finance_origin,
                    "dynamics_meaning": row.dynamics_meaning,
                    "notes": row.notes,
                }
            )
    if review_rows:
        lines.append(pd.DataFrame(review_rows).to_markdown(index=False))
    else:
        lines.append("No manual-review rows were generated.")
    lines.append("")
    lines.append("## 3. Property-guided context")
    lines.append("")
    lines.append(f"- property summary: {selected_library.notes}")
    lines.append(f"- layering summary: {selected_library.curation_notes}")
    lines.append("")
    lines.append("## 4. Review checklist")
    lines.append("")
    lines.append("- Check whether the selected factors are mechanistically distinct rather than duplicate proxies.")
    lines.append("- Check whether the selected factors behave like approximate Koopman coordinates rather than only fitting one-step noise.")
    lines.append("- Confirm that any physics-identifier factor is interpretable under the chosen identifier family.")
    lines.append("- Reject factors whose gain comes only from one-step fitting but harms rollout stability.")
    lines.append("- Promote factors that remain stable across additional suites or seeds into the curated library.")
    return "\n".join(lines)
