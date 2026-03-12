from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..utils import ensure_dir
from .miner import FactorMiningRunResult
from .review import candidate_scores_frame, manual_review_markdown


def save_run_artifacts(
    run_dir: Path,
    result: FactorMiningRunResult,
    translation_table_markdown: str | None = None,
) -> list[str]:
    ensure_dir(run_dir)
    manifest: list[str] = []

    candidate_csv = run_dir / "candidate_scores.csv"
    candidate_scores_frame(result.candidate_scores).to_csv(candidate_csv, index=False)
    manifest.append(candidate_csv.name)

    selected_json = run_dir / "selected_factor_library.json"
    with selected_json.open("w", encoding="utf-8") as f:
        json.dump(result.selected_library.to_dict(), f, ensure_ascii=False, indent=2)
    manifest.append(selected_json.name)

    metrics_json = run_dir / "metrics.json"
    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "property_profile": result.property_profile,
                "property_summary": result.property_summary,
                "baseline_metrics": result.baseline_metrics,
                "final_metrics": result.final_metrics,
                "test_metrics": result.test_metrics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    manifest.append(metrics_json.name)

    review_md = run_dir / "manual_review.md"
    review_md.write_text(
        manual_review_markdown(
            task_name=result.task_name,
            identifier_kind=result.identifier_kind,
            rows=result.candidate_scores,
            selected_library=result.selected_library,
        ),
        encoding="utf-8",
    )
    manifest.append(review_md.name)

    summary_md = run_dir / "run_summary.md"
    summary_lines = [
        f"# Factor mining summary: {result.task_name} / {result.identifier_kind}",
        "",
        f"- mode summary: {result.property_summary}",
        f"- selected factors: {', '.join(spec.name for spec in result.selected_specs) if result.selected_specs else 'none'}",
        f"- validation rmse@10: {result.final_metrics.get('rmse@10', float('nan')):.6g}",
        f"- validation rmse@50: {result.final_metrics.get('rmse@50', float('nan')):.6g}",
        f"- test rmse@10: {result.test_metrics.get('rmse@10', float('nan')):.6g}",
        f"- test rmse@50: {result.test_metrics.get('rmse@50', float('nan')):.6g}",
        "",
        "## Property profile",
        "",
        pd.DataFrame([result.property_profile]).to_markdown(index=False),
        "",
        "## Selected factors",
        "",
    ]
    if result.selected_specs:
        rows = []
        for spec in result.selected_specs:
            candidate_row = next((row for row in result.candidate_scores if row.factor_name == spec.name), None)
            rows.append(
                {
                    "factor": spec.name,
                    "family": spec.family,
                    "formula": candidate_row.formula if candidate_row is not None else "",
                    "koopman_score": candidate_row.koopman_score if candidate_row is not None else float("nan"),
                    "finance_origin": spec.finance_origin,
                    "dynamics_meaning": spec.dynamics_meaning,
                }
            )
        summary_lines.append(pd.DataFrame(rows).to_markdown(index=False))
    else:
        summary_lines.append("No factor passed the forward-selection gate; the RC baseline was retained.")
    summary_md.write_text("\n".join(summary_lines), encoding="utf-8")
    manifest.append(summary_md.name)

    if translation_table_markdown is not None:
        translation_md = run_dir / "finance_to_dynamics_translation.md"
        translation_md.write_text(translation_table_markdown, encoding="utf-8")
        manifest.append(translation_md.name)

    manifest_path = run_dir / "manifest.txt"
    manifest_path.write_text("\n".join(manifest) + "\n", encoding="utf-8")
    manifest.append(manifest_path.name)
    return manifest
