from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..utils import ensure_dir
from .loop import run_research_loop


FASTSLOW_COORDINATE_NAMES = frozenset({"fastslow", "theory_fastslow"})
DEFAULT_FASTSLOW_COORDINATES = ("raw", "delay", "fastslow", "theory_fastslow", "factor")


def _clip01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def _relative_gain_pct(baseline: float, candidate: float) -> float:
    baseline = float(baseline)
    candidate = float(candidate)
    if not np.isfinite(baseline) or not np.isfinite(candidate):
        return float("nan")
    return float(100.0 * (baseline - candidate) / (abs(baseline) + 1e-12))


def _best_row(df: pd.DataFrame, *, sort_by: str, ascending: bool = True) -> pd.Series | None:
    if df.empty or sort_by not in df.columns:
        return None
    ordered = df.sort_values(sort_by, ascending=ascending, na_position="last").reset_index(drop=True)
    return ordered.iloc[0] if not ordered.empty else None


def _fastslow_coordinate_score(row: pd.Series) -> float:
    markov_quality = _clip01(1.0 - max(float(row.get("markov_gain_ratio", np.inf)), 0.0))
    spectral_quality = (
        0.4 * _clip01(float(row.get("spectral_radius_corr", 0.0)))
        + 0.35 * _clip01(1.0 - float(row.get("spectral_radius_rmse", 1.0)))
        + 0.25 * _clip01(float(row.get("koopman_invariance_score", 0.0)))
    )
    koopman_quality = _clip01(float(row.get("koopman_invariance_score", 0.0)))
    return float(0.4 * markov_quality + 0.35 * spectral_quality + 0.25 * koopman_quality)


def summarize_fastslow_benchmarks(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    if benchmark_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for task, task_df in benchmark_df.groupby("task"):
        task_df = task_df.reset_index(drop=True)
        rc_raw = _best_row(task_df[task_df["base_model_name"] == "rc_raw"], sort_by="rmse@50")
        rc_fastslow = _best_row(task_df[task_df["base_model_name"] == "rc_fastslow_readout"], sort_by="rmse@50")
        ngrc_raw = _best_row(task_df[task_df["base_model_name"] == "ngrc_raw"], sort_by="rmse@50")
        ngrc_fastslow = _best_row(task_df[task_df["base_model_name"] == "ngrc_fastslow_readout"], sort_by="rmse@50")
        best_fastslow = _best_row(
            task_df[
                task_df["uses_fastslow"].astype(int).eq(1)
                & ~task_df["variant"].astype(str).str.contains("factor_readout", regex=False)
            ],
            sort_by="rmse@50",
        )
        best_factor = _best_row(
            task_df[task_df["variant"].astype(str).str.contains("factor_readout", regex=False)],
            sort_by="rmse@50",
        )
        best_overall = _best_row(task_df, sort_by="rmse@50")
        rows.append(
            {
                "task": task,
                "rc_raw_rmse50": float(rc_raw["rmse@50"]) if rc_raw is not None else float("nan"),
                "rc_fastslow_rmse50": float(rc_fastslow["rmse@50"]) if rc_fastslow is not None else float("nan"),
                "rc_fastslow_gain_pct": _relative_gain_pct(
                    float(rc_raw["rmse@50"]) if rc_raw is not None else float("nan"),
                    float(rc_fastslow["rmse@50"]) if rc_fastslow is not None else float("nan"),
                ),
                "ngrc_raw_rmse50": float(ngrc_raw["rmse@50"]) if ngrc_raw is not None else float("nan"),
                "ngrc_fastslow_rmse50": float(ngrc_fastslow["rmse@50"]) if ngrc_fastslow is not None else float("nan"),
                "ngrc_fastslow_gain_pct": _relative_gain_pct(
                    float(ngrc_raw["rmse@50"]) if ngrc_raw is not None else float("nan"),
                    float(ngrc_fastslow["rmse@50"]) if ngrc_fastslow is not None else float("nan"),
                ),
                "best_fastslow_variant": str(best_fastslow["variant"]) if best_fastslow is not None else "",
                "best_fastslow_rmse50": float(best_fastslow["rmse@50"]) if best_fastslow is not None else float("nan"),
                "best_factor_variant": str(best_factor["variant"]) if best_factor is not None else "",
                "best_factor_rmse50": float(best_factor["rmse@50"]) if best_factor is not None else float("nan"),
                "factor_vs_best_fastslow_gain_pct": _relative_gain_pct(
                    float(best_fastslow["rmse@50"]) if best_fastslow is not None else float("nan"),
                    float(best_factor["rmse@50"]) if best_factor is not None else float("nan"),
                ),
                "best_factor_names": str(best_factor.get("readout_factor_names", "")) if best_factor is not None else "",
                "best_overall_variant": str(best_overall["variant"]) if best_overall is not None else "",
                "best_overall_rmse50": float(best_overall["rmse@50"]) if best_overall is not None else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("task").reset_index(drop=True)


def summarize_fastslow_coordinates(coordinate_df: pd.DataFrame) -> pd.DataFrame:
    if coordinate_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for task, task_df in coordinate_df.groupby("task"):
        task_df = task_df.reset_index(drop=True)
        best_closure = _best_row(task_df, sort_by="markov_gain_ratio", ascending=True)
        best_spectral = _best_row(task_df, sort_by="spectral_radius_rmse", ascending=True)
        best_koopman = _best_row(task_df, sort_by="koopman_invariance_score", ascending=False)
        fastslow_family = task_df[task_df["coordinate"].isin(FASTSLOW_COORDINATE_NAMES)].copy()
        if fastslow_family.empty:
            best_fastslow = None
        else:
            fastslow_family["fastslow_score"] = fastslow_family.apply(_fastslow_coordinate_score, axis=1)
            best_fastslow = fastslow_family.sort_values(
                ["fastslow_score", "markov_gain_ratio", "spectral_radius_rmse", "koopman_invariance_score"],
                ascending=[False, True, True, False],
                na_position="last",
            ).iloc[0]
        wins: list[str] = []
        if best_closure is not None and str(best_closure["coordinate"]) in FASTSLOW_COORDINATE_NAMES:
            wins.append("Markov")
        if best_spectral is not None and str(best_spectral["coordinate"]) in FASTSLOW_COORDINATE_NAMES:
            wins.append("Spectral")
        if best_koopman is not None and str(best_koopman["coordinate"]) in FASTSLOW_COORDINATE_NAMES:
            wins.append("Koopman")
        rows.append(
            {
                "task": task,
                "best_closure_coordinate": str(best_closure["coordinate"]) if best_closure is not None else "",
                "best_spectral_coordinate": str(best_spectral["coordinate"]) if best_spectral is not None else "",
                "best_koopman_coordinate": str(best_koopman["coordinate"]) if best_koopman is not None else "",
                "best_fastslow_coordinate": str(best_fastslow["coordinate"]) if best_fastslow is not None else "",
                "fastslow_wins": "; ".join(wins),
                "fastslow_markov_gain_ratio": float(best_fastslow["markov_gain_ratio"]) if best_fastslow is not None else float("nan"),
                "fastslow_spectral_corr": float(best_fastslow["spectral_radius_corr"]) if best_fastslow is not None else float("nan"),
                "fastslow_spectral_rmse": float(best_fastslow["spectral_radius_rmse"]) if best_fastslow is not None else float("nan"),
                "fastslow_koopman_score": float(best_fastslow["koopman_invariance_score"]) if best_fastslow is not None else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("task").reset_index(drop=True)


def summarize_fastslow_factors(factor_df: pd.DataFrame) -> pd.DataFrame:
    if factor_df.empty:
        return pd.DataFrame()
    ordered = factor_df.sort_values("validation_score", na_position="last").groupby("task").first().reset_index()
    cols = [
        "task",
        "identifier_kind",
        "selected_factors",
        "selected_koopman_score",
        "baseline_rmse50",
        "final_rmse50",
        "test_rmse50",
    ]
    existing = [col for col in cols if col in ordered.columns]
    return ordered[existing].sort_values("task").reset_index(drop=True)


def _records(df: pd.DataFrame) -> list[dict[str, object]]:
    if df.empty:
        return []
    return json.loads(df.to_json(orient="records"))


def render_fastslow_validation_report(
    *,
    suite: str,
    benchmark_summary: pd.DataFrame,
    coordinate_summary: pd.DataFrame,
    factor_summary: pd.DataFrame,
    manifest: dict[str, str],
) -> str:
    task_count = len({*benchmark_summary.get("task", pd.Series(dtype=str)).tolist(), *coordinate_summary.get("task", pd.Series(dtype=str)).tolist()})
    rc_lift_count = int(benchmark_summary["rc_fastslow_gain_pct"].gt(0).sum()) if "rc_fastslow_gain_pct" in benchmark_summary else 0
    ngrc_lift_count = int(benchmark_summary["ngrc_fastslow_gain_pct"].gt(0).sum()) if "ngrc_fastslow_gain_pct" in benchmark_summary else 0
    factor_lift_count = int(benchmark_summary["factor_vs_best_fastslow_gain_pct"].gt(0).sum()) if "factor_vs_best_fastslow_gain_pct" in benchmark_summary else 0
    coordinate_win_count = int(coordinate_summary["fastslow_wins"].astype(str).str.len().gt(0).sum()) if "fastslow_wins" in coordinate_summary else 0
    lines = [
        "# Fast-Slow Validation Report",
        "",
        f"- suite: {suite}",
        f"- tasks analyzed: {task_count}",
        f"- RC tasks with fast/slow lift: {rc_lift_count}",
        f"- NGRC tasks with fast/slow lift: {ngrc_lift_count}",
        f"- tasks where factor readout beats best fast/slow baseline: {factor_lift_count}",
        f"- tasks where a fast/slow-family coordinate wins at least one theory lens: {coordinate_win_count}",
        "",
        "## Benchmark lift",
        "",
    ]
    if not benchmark_summary.empty:
        cols = [
            "task",
            "rc_fastslow_gain_pct",
            "ngrc_fastslow_gain_pct",
            "best_fastslow_variant",
            "best_factor_variant",
            "factor_vs_best_fastslow_gain_pct",
            "best_overall_variant",
        ]
        lines.append(benchmark_summary[cols].to_markdown(index=False))
    else:
        lines.append("No benchmark rows were generated.")
    lines.extend(["", "## Coordinate evidence", ""])
    if not coordinate_summary.empty:
        lines.append(
            coordinate_summary[
                [
                    "task",
                    "best_closure_coordinate",
                    "best_spectral_coordinate",
                    "best_koopman_coordinate",
                    "best_fastslow_coordinate",
                    "fastslow_wins",
                    "fastslow_markov_gain_ratio",
                    "fastslow_spectral_corr",
                    "fastslow_koopman_score",
                ]
            ].to_markdown(index=False)
        )
    else:
        lines.append("No coordinate rows were generated.")
    lines.extend(["", "## Factor winners", ""])
    if not factor_summary.empty:
        lines.append(factor_summary.to_markdown(index=False))
    else:
        lines.append("No factor-mining rows were generated.")
    lines.extend(["", "## Artifact map", ""])
    lines.extend(f"- {name}: {path}" for name, path in manifest.items())
    return "\n".join(lines)


def run_fastslow_validation(
    *,
    suite: str = "fastslow_theory",
    out_dir: str = "runs/fastslow_validation/fastslow_theory",
    seed: int = 123,
    task_names: Sequence[str] | None = None,
    model_groups: Sequence[str] | None = ("fastslow_ablation",),
    grid_mode: str = "quick",
    coordinate_kinds: Sequence[str] = DEFAULT_FASTSLOW_COORDINATES,
    delay_dim: int = 8,
    mining_mode: str = "identify",
    full_library_search: bool = True,
    factor_config_path: str | None = "configs/fastslow_theory_factor_mining.yaml",
    identifier_kinds: Sequence[str] | None = None,
) -> dict[str, object]:
    result = run_research_loop(
        suite=suite,
        out_dir=out_dir,
        seed=seed,
        task_names=task_names,
        model_groups=model_groups,
        grid_mode=grid_mode,
        coordinate_kinds=coordinate_kinds,
        delay_dim=delay_dim,
        mining_mode=mining_mode,
        full_library_search=full_library_search,
        factor_config_path=factor_config_path,
        identifier_kinds=identifier_kinds,
    )
    out_path = Path(out_dir)
    ensure_dir(out_path)
    benchmark_summary = summarize_fastslow_benchmarks(result["benchmarks"])
    coordinate_summary = summarize_fastslow_coordinates(result["coordinate_analysis"])
    factor_summary = summarize_fastslow_factors(result["factor_mining"])
    report_path = out_path / "fastslow_validation_report.md"
    report_path.write_text(
        render_fastslow_validation_report(
            suite=suite,
            benchmark_summary=benchmark_summary,
            coordinate_summary=coordinate_summary,
            factor_summary=factor_summary,
            manifest=result["manifest"],
        ),
        encoding="utf-8",
    )
    summary_path = out_path / "fastslow_validation_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "suite": suite,
                "benchmarks": _records(benchmark_summary),
                "coordinates": _records(coordinate_summary),
                "factors": _records(factor_summary),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    result["manifest"]["fastslow_validation_report"] = str(report_path)
    result["manifest"]["fastslow_validation_summary"] = str(summary_path)
    return result
