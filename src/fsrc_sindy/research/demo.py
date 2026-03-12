from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .fastslow_validation import run_fastslow_validation

REPO_ROOT = Path(__file__).resolve().parents[3]
FASTSLOW_COORDINATE_NAMES = frozenset({"fastslow", "theory_fastslow"})


@dataclass(frozen=True)
class DemoRunSpec:
    label: str
    suite: str
    tasks: tuple[str, ...]
    description: str
    fresh_out_dir: str
    preferred_existing_dirs: tuple[str, ...] = ()


def build_demo_specs(quick: bool = True) -> list[DemoRunSpec]:
    classic_tasks: tuple[str, ...] = (
        "hindmarsh_rose_bursting_noisy",
        "fitzhugh_nagumo_classic_noisy",
    )
    sparse_tasks: tuple[str, ...] = ("lorenz96_twoscale_sparse_triplet_noisy",)
    if not quick:
        classic_tasks = classic_tasks + ("vanderpol_relaxation_noisy",)
        sparse_tasks = sparse_tasks + ("lorenz96_twoscale_sparse_mixed_noisy",)
    return [
        DemoRunSpec(
            label="classic_noisy",
            suite="fastslow_theory",
            tasks=classic_tasks,
            description="Classic slow-fast systems under noise, used as the baseline multiscale benchmark.",
            fresh_out_dir="runs/demo_notebook/sf/classic_noisy",
            preferred_existing_dirs=("runs/fastslow_validation/fastslow_theory_noisy",),
        ),
        DemoRunSpec(
            label="sparse_observation",
            suite="fastslow_sparse_theory",
            tasks=sparse_tasks,
            description="High-dimensional slow-fast system under sparse observation, where only a few coordinates are measured.",
            fresh_out_dir="runs/demo_notebook/sf/sparse_observation",
            preferred_existing_dirs=(
                "runs/fastslow_validation/fastslow_sparse_theory",
                "runs/fastslow_validation/fastslow_sparse_triplet_noisy",
            ),
        ),
    ]


def _summary_path(run_dir: Path) -> Path:
    return run_dir / "fastslow_validation_summary.json"


def _repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _task_filter(df: pd.DataFrame, task_names: Sequence[str] | None) -> pd.DataFrame:
    if df.empty or not task_names or "task" not in df.columns:
        return df
    return df[df["task"].astype(str).isin(list(task_names))].reset_index(drop=True)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_demo_runs(
    *,
    quick: bool = True,
    reuse_existing: bool = True,
    rerun: bool = False,
    out_root: str | Path | None = None,
    seed: int = 123,
    model_groups: Sequence[str] = ("fastslow_ablation",),
    grid_mode: str = "quick",
    full_library_search: bool = True,
) -> list[dict[str, Any]]:
    run_records: list[dict[str, Any]] = []
    out_root = _repo_path(out_root) if out_root is not None else None
    for spec in build_demo_specs(quick=quick):
        selected_run_dir: Path | None = None
        source = "fresh_run"
        if reuse_existing and not rerun:
            for existing_dir in spec.preferred_existing_dirs:
                candidate = _repo_path(existing_dir)
                if _summary_path(candidate).exists():
                    selected_run_dir = candidate
                    source = "existing"
                    break
        if selected_run_dir is None:
            selected_run_dir = _repo_path(spec.fresh_out_dir) if out_root is None else out_root / spec.label
            had_summary = _summary_path(selected_run_dir).exists()
            if rerun or not had_summary:
                run_fastslow_validation(
                    suite=spec.suite,
                    out_dir=str(selected_run_dir),
                    seed=seed,
                    task_names=spec.tasks,
                    model_groups=model_groups,
                    grid_mode=grid_mode,
                    full_library_search=full_library_search,
                )
                source = "rerun" if had_summary and rerun else "fresh_run"
            else:
                source = "cached_demo"
        run_records.append(
            {
                "label": spec.label,
                "suite": spec.suite,
                "tasks": list(spec.tasks),
                "description": spec.description,
                "run_dir": str(selected_run_dir),
                "summary_path": str(_summary_path(selected_run_dir)),
                "source": source,
            }
        )
    return run_records


def load_demo_bundle(run_dir: str | Path, *, task_filter: Sequence[str] | None = None) -> dict[str, Any]:
    run_dir = _repo_path(run_dir)
    summary = _read_json(_summary_path(run_dir), default={"benchmarks": [], "coordinates": [], "factors": []})
    benchmark_summary = _task_filter(pd.DataFrame(summary.get("benchmarks", [])), task_filter)
    coordinate_summary = _task_filter(pd.DataFrame(summary.get("coordinates", [])), task_filter)
    factor_summary = _task_filter(pd.DataFrame(summary.get("factors", [])), task_filter)
    benchmark_results = _task_filter(_read_csv(run_dir / "benchmarks" / "benchmark_results.csv"), task_filter)
    coordinate_results = _task_filter(_read_csv(run_dir / "coordinate_analysis" / "coordinate_analysis_summary.csv"), task_filter)
    factor_results = _task_filter(_read_csv(run_dir / "factor_mining" / "factor_mining_summary.csv"), task_filter)
    validation_gate = _task_filter(pd.DataFrame(_read_json(run_dir / "validation_gate.json", default=[])), task_filter)
    loop_manifest = _read_json(run_dir / "loop_manifest.json", default={})
    return {
        "run_dir": str(run_dir),
        "benchmark_summary": benchmark_summary,
        "coordinate_summary": coordinate_summary,
        "factor_summary": factor_summary,
        "benchmark_results": benchmark_results,
        "coordinate_results": coordinate_results,
        "factor_results": factor_results,
        "validation_gate": validation_gate,
        "loop_manifest": loop_manifest,
        "theory_evidence": (run_dir / "theory_evidence.md").read_text(encoding="utf-8") if (run_dir / "theory_evidence.md").exists() else "",
        "loop_summary": (run_dir / "loop_summary.md").read_text(encoding="utf-8") if (run_dir / "loop_summary.md").exists() else "",
        "fastslow_report": (run_dir / "fastslow_validation_report.md").read_text(encoding="utf-8") if (run_dir / "fastslow_validation_report.md").exists() else "",
    }


def collect_demo_tables(run_records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    table_keys = [
        "benchmark_summary",
        "coordinate_summary",
        "factor_summary",
        "benchmark_results",
        "coordinate_results",
        "factor_results",
        "validation_gate",
    ]
    collected: dict[str, list[pd.DataFrame]] = {key: [] for key in table_keys}
    text_artifacts: dict[str, dict[str, str]] = {}
    for record in run_records:
        bundle = load_demo_bundle(record["run_dir"], task_filter=record.get("tasks"))
        text_artifacts[record["label"]] = {
            "run_dir": bundle["run_dir"],
            "theory_evidence": bundle["theory_evidence"],
            "loop_summary": bundle["loop_summary"],
            "fastslow_report": bundle["fastslow_report"],
        }
        for key in table_keys:
            df = bundle[key]
            if df.empty:
                continue
            tagged = df.copy()
            tagged["demo_label"] = record["label"]
            tagged["demo_suite"] = record["suite"]
            collected[key].append(tagged)
    tables: dict[str, Any] = {
        key: (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())
        for key, frames in collected.items()
    }
    tables["artifacts"] = text_artifacts
    tables["run_records"] = list(run_records)
    return tables


def _split_factors(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(";") if item.strip()]


def build_factor_frequency_table(factor_summary: pd.DataFrame) -> pd.DataFrame:
    if factor_summary.empty:
        return pd.DataFrame(columns=["factor", "task_count", "tasks", "mean_koopman_score"])
    rows: list[dict[str, Any]] = []
    for _, row in factor_summary.iterrows():
        for factor in _split_factors(row.get("selected_factors", "")):
            rows.append(
                {
                    "factor": factor,
                    "task": row.get("task", ""),
                    "selected_koopman_score": row.get("selected_koopman_score", np.nan),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["factor", "task_count", "tasks", "mean_koopman_score"])
    exploded = pd.DataFrame(rows)
    grouped = (
        exploded.groupby("factor")
        .agg(
            task_count=("task", "nunique"),
            tasks=("task", lambda s: "; ".join(sorted(set(str(x) for x in s if str(x))))),
            mean_koopman_score=("selected_koopman_score", "mean"),
        )
        .reset_index()
        .sort_values(["task_count", "mean_koopman_score", "factor"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    return grouped


def _split_wins(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(";") if item.strip()]


def _best_gain_pct(row: pd.Series) -> float:
    gains = [row.get("rc_fastslow_gain_pct", np.nan), row.get("ngrc_fastslow_gain_pct", np.nan)]
    finite = [float(gain) for gain in gains if pd.notna(gain)]
    if not finite:
        return float("nan")
    return float(max(finite))


def build_task_evidence_table(
    benchmark_summary: pd.DataFrame,
    coordinate_summary: pd.DataFrame,
    factor_summary: pd.DataFrame,
    validation_gate: pd.DataFrame,
) -> pd.DataFrame:
    if benchmark_summary.empty and coordinate_summary.empty:
        return pd.DataFrame()
    if benchmark_summary.empty:
        merged = coordinate_summary[["task", "demo_label"]].drop_duplicates().reset_index(drop=True)
    else:
        merged = benchmark_summary.copy()
    coord_cols = [
        "task",
        "demo_label",
        "best_closure_coordinate",
        "best_spectral_coordinate",
        "best_koopman_coordinate",
        "best_fastslow_coordinate",
        "fastslow_wins",
        "fastslow_markov_gain_ratio",
        "fastslow_spectral_corr",
        "fastslow_koopman_score",
    ]
    gate_cols = [
        "task",
        "demo_label",
        "fastslow_validation_allowed",
        "reason",
        "dominant_axes",
    ]
    factor_cols = ["task", "demo_label", "selected_factors", "selected_koopman_score", "test_rmse50"]
    if not coordinate_summary.empty:
        merged = merged.merge(coordinate_summary[coord_cols], on=["task", "demo_label"], how="outer")
    if not factor_summary.empty:
        available_factor_cols = [col for col in factor_cols if col in factor_summary.columns]
        merged = merged.merge(factor_summary[available_factor_cols], on=["task", "demo_label"], how="left")
    if not validation_gate.empty:
        available_gate_cols = [col for col in gate_cols if col in validation_gate.columns]
        merged = merged.merge(validation_gate[available_gate_cols], on=["task", "demo_label"], how="left")
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        wins = _split_wins(row.get("fastslow_wins", ""))
        lens_count = len(wins)
        best_gain = _best_gain_pct(row)
        factor_best = "factor_readout" in str(row.get("best_overall_variant", ""))
        allowed = bool(row.get("fastslow_validation_allowed")) if pd.notna(row.get("fastslow_validation_allowed", np.nan)) else False
        if pd.notna(best_gain) and best_gain > 5.0 and lens_count >= 2 and allowed:
            claim_status = "supported"
            confidence = "high"
            takeaway = "Predictive lift and coordinate diagnostics align for a fast/slow explanation."
        elif pd.notna(best_gain) and best_gain > 5.0 and lens_count >= 1 and allowed:
            claim_status = "supported"
            confidence = "medium"
            takeaway = "Predictive lift is present, with partial support from the theory lenses."
        elif lens_count >= 1 and allowed:
            claim_status = "partial"
            confidence = "low"
            takeaway = "Fast/slow keeps at least one theory lens, but prediction does not clearly dominate."
        elif factor_best:
            claim_status = "factor_refinement"
            confidence = "low"
            takeaway = "The task favors a mined factor readout more than a generic fast/slow coordinate."
        else:
            claim_status = "not_supported"
            confidence = "low"
            takeaway = "Current evidence does not support fast/slow as the dominant coordinate."
        rows.append(
            {
                "demo_label": row.get("demo_label", ""),
                "task": row.get("task", ""),
                "best_gain_pct": best_gain,
                "best_overall_variant": row.get("best_overall_variant", ""),
                "fastslow_validation_allowed": row.get("fastslow_validation_allowed", np.nan),
                "fastslow_wins": row.get("fastslow_wins", ""),
                "best_closure_coordinate": row.get("best_closure_coordinate", ""),
                "best_spectral_coordinate": row.get("best_spectral_coordinate", ""),
                "best_koopman_coordinate": row.get("best_koopman_coordinate", ""),
                "selected_factors": row.get("selected_factors", ""),
                "dominant_axes": row.get("dominant_axes", ""),
                "claim_status": claim_status,
                "confidence": confidence,
                "takeaway": takeaway,
            }
        )
    return pd.DataFrame(rows).sort_values(["demo_label", "task"]).reset_index(drop=True)


def load_benchmark_series(run_dir: str | Path, task: str) -> dict[str, np.ndarray]:
    run_dir = _repo_path(run_dir)
    series_path = run_dir / "benchmarks" / f"{task}_series.npz"
    if not series_path.exists():
        raise FileNotFoundError(f"Missing benchmark series artifact: {series_path}")
    with np.load(series_path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def read_artifact_excerpt(run_dir: str | Path, artifact_name: str, *, max_lines: int = 24) -> str:
    run_dir = _repo_path(run_dir)
    path = run_dir / artifact_name
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[:max_lines])


__all__ = [
    "DemoRunSpec",
    "build_demo_specs",
    "prepare_demo_runs",
    "load_demo_bundle",
    "collect_demo_tables",
    "build_factor_frequency_table",
    "build_task_evidence_table",
    "load_benchmark_series",
    "read_artifact_excerpt",
]
