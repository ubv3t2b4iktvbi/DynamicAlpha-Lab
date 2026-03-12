from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yaml

from ..experiment import run_benchmark_suite
from ..factors import DynamicsFeatureConfig, FactorMiningConfig
from ..models.rc import RCConfig
from ..pipeline import run_factor_mining_suite
from ..selection import expand_model_group_names
from ..utils import ensure_dir
from .coordinate_analysis import run_coordinate_analysis_suite


def _load_factor_config(path: str | None) -> tuple[dict, dict, dict]:
    if path is None:
        return {}, {}, {}
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    return (
        dict(payload.get("factor_mining", {})),
        dict(payload.get("rc", {})),
        dict(payload.get("features", {})),
    )


def _best_benchmark_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    sort_key = "rmse@50" if "rmse@50" in df.columns else df.columns[0]
    return df.sort_values(sort_key).groupby("task").first().reset_index()


def _best_coordinate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df, df, df
    best_closure = df.sort_values("markov_gain_ratio", na_position="last").groupby("task").first().reset_index()
    best_spectral = df.sort_values("spectral_radius_rmse", na_position="last").groupby("task").first().reset_index()
    best_koopman = df.sort_values("koopman_invariance_score", ascending=False, na_position="last").groupby("task").first().reset_index()
    return best_closure, best_spectral, best_koopman


def _best_factor_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.sort_values("validation_score", na_position="last").groupby("task").first().reset_index()


def _clip01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def _confidence_tier(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def _benchmark_confidence(df: pd.DataFrame) -> list[dict[str, object]]:
    if df.empty:
        return []
    rows: list[dict[str, object]] = []
    for task, task_df in df.groupby("task"):
        sorted_df = task_df.sort_values("rmse@50", na_position="last").reset_index(drop=True)
        best = sorted_df.iloc[0]
        runner = sorted_df.iloc[1] if len(sorted_df) > 1 else None
        margin = float(((runner["rmse@50"] - best["rmse@50"]) / (abs(runner["rmse@50"]) + 1e-12)) if runner is not None else 0.1)
        stability = float(1.0 / (1.0 + best.get("acf_rmse", 0.0) + best.get("psd_rmse", 0.0)))
        score = _clip01(0.6 * margin * 5.0 + 0.4 * stability)
        rows.append(
            {
                "task": task,
                "section": "benchmarks",
                "winner": best["variant"],
                "score": score,
                "tier": _confidence_tier(score),
                "evidence": f"rmse@50 margin={margin:.3f}, acf_rmse={best.get('acf_rmse', float('nan')):.4g}, psd_rmse={best.get('psd_rmse', float('nan')):.4g}",
            }
        )
    return rows


def _coordinate_confidence(df: pd.DataFrame) -> list[dict[str, object]]:
    if df.empty:
        return []
    rows: list[dict[str, object]] = []
    for task, task_df in df.groupby("task"):
        closure_df = task_df.sort_values("markov_gain_ratio", na_position="last").reset_index(drop=True)
        spectral_df = task_df.sort_values("spectral_radius_rmse", na_position="last").reset_index(drop=True)
        best_closure = closure_df.iloc[0]
        best_spectral = spectral_df.iloc[0]
        closure_quality = 1.0 - max(float(best_closure.get("markov_gain_ratio", 1.0)), 0.0)
        spectral_quality = 0.35 * _clip01(float(best_spectral.get("spectral_radius_corr", 0.0))) + 0.35 * _clip01(1.0 - float(best_spectral.get("spectral_radius_rmse", 1.0))) + 0.30 * _clip01(float(best_spectral.get("koopman_invariance_score", 0.0)))
        agreement_bonus = 0.15 if best_closure["coordinate"] == best_spectral["coordinate"] else 0.0
        score = _clip01(0.45 * closure_quality + 0.4 * spectral_quality + agreement_bonus)
        rows.append(
            {
                "task": task,
                "section": "coordinates",
                "winner": f"closure={best_closure['coordinate']}, spectral={best_spectral['coordinate']}",
                "score": score,
                "tier": _confidence_tier(score),
                "evidence": f"markov_gain_ratio={best_closure.get('markov_gain_ratio', float('nan')):.4g}, spectral_corr={best_spectral.get('spectral_radius_corr', float('nan')):.4g}, koopman_score={best_spectral.get('koopman_invariance_score', float('nan')):.4g}",
            }
        )
    return rows


def _factor_confidence(df: pd.DataFrame) -> list[dict[str, object]]:
    if df.empty:
        return []
    rows: list[dict[str, object]] = []
    for task, task_df in df.groupby("task"):
        sorted_df = task_df.sort_values("validation_score", na_position="last").reset_index(drop=True)
        best = sorted_df.iloc[0]
        runner = sorted_df.iloc[1] if len(sorted_df) > 1 else None
        margin = float(((runner["validation_score"] - best["validation_score"]) / (abs(runner["validation_score"]) + 1e-12)) if runner is not None else 0.1)
        generalization = _clip01(1.0 - abs(float(best.get("test_rmse50", np.inf)) - float(best.get("final_rmse50", np.inf))) / (abs(float(best.get("final_rmse50", 1.0))) + 1e-12))
        koopman_quality = _clip01(float(best.get("selected_koopman_score", 0.0))) if np.isfinite(float(best.get("selected_koopman_score", np.nan))) else 0.0
        score = _clip01(0.4 * margin * 5.0 + 0.35 * generalization + 0.25 * koopman_quality)
        rows.append(
            {
                "task": task,
                "section": "factor_mining",
                "winner": f"{best['identifier_kind']} -> {best.get('selected_factors', '')}",
                "score": score,
                "tier": _confidence_tier(score),
                "evidence": f"validation margin={margin:.3f}, final_rmse50={best.get('final_rmse50', float('nan')):.4g}, test_rmse50={best.get('test_rmse50', float('nan')):.4g}, selected_koopman_score={best.get('selected_koopman_score', float('nan')):.4g}",
            }
        )
    return rows


def _build_confidence_report(benchmark_df: pd.DataFrame, coordinate_df: pd.DataFrame, factor_df: pd.DataFrame) -> list[dict[str, object]]:
    report = _benchmark_confidence(benchmark_df) + _coordinate_confidence(coordinate_df) + _factor_confidence(factor_df)
    for row in report:
        row["review_required"] = True
        row["approval_status"] = "PENDING_DYNAMICS_EXPERT_REVIEW"
    return report


def _render_expert_review_template(confidence_rows: list[dict[str, object]], manifest: dict[str, str]) -> str:
    lines = [
        "# Dynamics expert review gate",
        "",
        "Status: PENDING_DYNAMICS_EXPERT_REVIEW",
        "",
        "This loop is not self-approving. LLM-generated explanations, next experiments, and factor edits remain provisional until a human dynamics expert signs off.",
        "",
        "## Confidence report",
        "",
    ]
    if confidence_rows:
        lines.append(pd.DataFrame(confidence_rows)[["task", "section", "winner", "tier", "score", "evidence"]].to_markdown(index=False))
    else:
        lines.append("No confidence rows were generated.")
    lines.extend(
        [
            "",
            "## Required expert checks",
            "",
            "- Verify that the winning ablation is not an artifact of short-horizon metrics alone.",
            "- Verify that coordinate conclusions are consistent with known attractor geometry and not caused by proxy mismatch.",
            "- Verify that selected or proposed factors have valid mechanistic meaning and remain causal.",
            "- Approve or reject the proposed next experiment.",
            "- Approve or reject any factor-bank modification.",
            "",
            "## Sign-off",
            "",
            "- Reviewer:",
            "- Date:",
            "- Verdict: APPROVED / REJECTED / NEEDS MORE EVIDENCE",
            "- Notes:",
            "",
            "## Artifact map",
            "",
            *(f"- {name}: {path}" for name, path in manifest.items()),
        ]
    )
    return "\n".join(lines)


def _render_loop_summary(
    suite: str,
    loop_mode: str,
    benchmark_df: pd.DataFrame,
    coordinate_df: pd.DataFrame,
    factor_df: pd.DataFrame,
    manifest: dict[str, str],
    confidence_rows: list[dict[str, object]],
) -> str:
    lines = [
        "# Research loop summary",
        "",
        f"- suite: {suite}",
        f"- loop mode: {loop_mode}",
        "- llm_interpretation_status: PROVISIONAL",
        "- expert_review_gate: REQUIRED",
        "",
    ]
    if confidence_rows:
        lines.extend(
            [
                "## Confidence report",
                "",
                pd.DataFrame(confidence_rows)[["task", "section", "winner", "tier", "score"]].to_markdown(index=False),
                "",
            ]
        )
    if not benchmark_df.empty:
        lines.extend(
            [
                "## Best benchmark variants",
                "",
                _best_benchmark_rows(benchmark_df)[["task", "variant", "rmse@50", "acf_rmse", "psd_rmse"]].to_markdown(index=False),
                "",
            ]
        )
    if not coordinate_df.empty:
        best_closure, best_spectral, best_koopman = _best_coordinate_rows(coordinate_df)
        lines.extend(
            [
                "## Coordinate diagnostics",
                "",
                "### Best closure coordinates",
                "",
                best_closure[["task", "coordinate", "markov_gain_ratio", "lagged_rmse"]].to_markdown(index=False),
                "",
                "### Best spectral-preservation coordinates",
                "",
                best_spectral[["task", "coordinate", "spectral_radius_rmse", "spectral_radius_corr"]].to_markdown(index=False),
                "",
                "### Best Koopman-like coordinates",
                "",
                best_koopman[["task", "coordinate", "koopman_invariance_score", "koopman_linear_r2"]].to_markdown(index=False),
                "",
            ]
        )
    if not factor_df.empty:
        lines.extend(
            [
                "## Factor-mining winners",
                "",
                _best_factor_rows(factor_df)[["task", "identifier_kind", "selected_factors", "final_rmse50", "test_rmse50"]].to_markdown(index=False),
                "",
            ]
        )
    lines.extend(
        [
            "## Artifact map",
            "",
            *(f"- {name}: {path}" for name, path in manifest.items()),
            "",
            "## Review gate",
            "",
            "All interpretations and proposed edits must be reviewed in `expert_review_template.md` before they are treated as accepted research conclusions.",
            "",
            "## Next-step prompt",
            "",
            "Read the detailed reports under the artifact map, explain the dominant mechanism behind the best and worst ablations, attach a confidence tier to every claim, then propose one concrete next experiment and one factor or coordinate change. Mark every recommendation as pending dynamics-expert review.",
        ]
    )
    return "\n".join(lines)


def run_research_loop(
    suite: str,
    out_dir: str,
    seed: int = 123,
    task_names: Sequence[str] | None = None,
    model_groups: Sequence[str] | None = None,
    grid_mode: str = "quick",
    coordinate_kinds: Sequence[str] = ("raw", "delay", "fastslow", "factor"),
    delay_dim: int = 8,
    mining_mode: str = "identify",
    full_library_search: bool = True,
    factor_config_path: str | None = "configs/factor_mining.yaml",
    identifier_kinds: Sequence[str] | None = None,
    skip_benchmarks: bool = False,
    skip_coordinate_analysis: bool = False,
    skip_factor_mining: bool = False,
) -> dict[str, object]:
    out_path = Path(out_dir)
    ensure_dir(out_path)

    benchmark_df = pd.DataFrame()
    coordinate_df = pd.DataFrame()
    factor_df = pd.DataFrame()
    manifest: dict[str, str] = {}

    if not skip_benchmarks:
        benchmark_out = out_path / "benchmarks"
        ensure_dir(benchmark_out)
        model_names = expand_model_group_names(model_groups or ["fastslow_ablation"])
        benchmark_df = run_benchmark_suite(
            suite=suite,
            seed=seed,
            out_dir=str(benchmark_out),
            model_names=model_names,
            grid_mode=grid_mode,
            task_names=task_names,
        )
        manifest["benchmarks"] = str(benchmark_out)

    if not skip_coordinate_analysis:
        coordinate_out = out_path / "coordinate_analysis"
        ensure_dir(coordinate_out)
        coordinate_df = run_coordinate_analysis_suite(
            suite=suite,
            out_dir=str(coordinate_out),
            seed=seed,
            task_names=task_names,
            coordinate_kinds=coordinate_kinds,
            delay_dim=delay_dim,
        )
        manifest["coordinate_analysis"] = str(coordinate_out)

    if not skip_factor_mining:
        mining_out = out_path / "factor_mining"
        ensure_dir(mining_out)
        mining_kwargs, rc_kwargs, feature_kwargs = _load_factor_config(factor_config_path)
        mining_kwargs["mode"] = mining_mode
        if identifier_kinds is not None:
            mining_kwargs["identifier_kinds"] = tuple(identifier_kinds)
        if mining_mode == "identify":
            mining_kwargs["full_library_search"] = bool(full_library_search)
        mining_cfg = FactorMiningConfig(**mining_kwargs)
        rc_cfg = RCConfig(**rc_kwargs)
        feature_cfg = DynamicsFeatureConfig(**feature_kwargs)
        factor_df = run_factor_mining_suite(
            suite=suite,
            out_dir=str(mining_out),
            seed=seed,
            task_names=task_names,
            identifier_kinds=identifier_kinds,
            mining_cfg=mining_cfg,
            rc_cfg=rc_cfg,
            feature_cfg=feature_cfg,
        )
        manifest["factor_mining"] = str(mining_out)

    confidence_rows = _build_confidence_report(
        benchmark_df=benchmark_df,
        coordinate_df=coordinate_df,
        factor_df=factor_df,
    )
    loop_summary = _render_loop_summary(
        suite=suite,
        loop_mode=mining_mode,
        benchmark_df=benchmark_df,
        coordinate_df=coordinate_df,
        factor_df=factor_df,
        manifest=manifest,
        confidence_rows=confidence_rows,
    )
    (out_path / "loop_summary.md").write_text(loop_summary, encoding="utf-8")
    manifest["loop_summary"] = str(out_path / "loop_summary.md")
    confidence_path = out_path / "confidence_report.json"
    confidence_path.write_text(json.dumps(confidence_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["confidence_report"] = str(confidence_path)
    expert_review_path = out_path / "expert_review_template.md"
    expert_review_path.write_text(_render_expert_review_template(confidence_rows, manifest), encoding="utf-8")
    manifest["expert_review_template"] = str(expert_review_path)
    (out_path / "loop_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "benchmarks": benchmark_df,
        "coordinate_analysis": coordinate_df,
        "factor_mining": factor_df,
        "manifest": manifest,
        "confidence_report": confidence_rows,
    }
