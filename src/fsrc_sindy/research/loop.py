from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yaml

from ..benchmarks import build_suite
from ..experiment import run_benchmark_suite
from ..factors import DynamicsFeatureConfig, FactorMiningConfig, analyze_signal_properties, load_selected_factor_library
from ..models.rc import RCConfig
from ..pipeline import run_factor_mining_suite
from ..selection import expand_model_group_names, get_model_spec
from ..systems import simulate_task, split_series
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


@dataclass(frozen=True)
class TaskGateDecision:
    task: str
    fastslow_coordinate_hypothesis: bool
    fastslow_validation_allowed: bool
    recommended_coordinates: tuple[str, ...]
    recommended_models: tuple[str, ...]
    dominant_axes: str
    reason: str
    evidence: str


def _build_tasks(suite: str, task_names: Sequence[str] | None = None):
    tasks = build_suite(suite)
    if task_names:
        wanted = set(task_names)
        tasks = [task for task in tasks if task.name in wanted]
        missing = sorted(wanted - {task.name for task in tasks})
        if missing:
            raise ValueError(f"Unknown task names for suite={suite}: {missing}")
    return tasks


def _top_axes_text(profile: dict[str, float]) -> str:
    ordered = sorted(
        [
            ("oscillatory", float(profile.get("oscillatory_score", 0.0))),
            ("multiscale", float(profile.get("multiscale_score", 0.0))),
            ("trend", float(profile.get("trend_score", 0.0))),
            ("burstiness", float(profile.get("burstiness_score", 0.0))),
            ("unpredictability", float(profile.get("unpredictability_score", 0.0))),
            ("closure_need", float(profile.get("closure_need_score", 0.0))),
        ],
        key=lambda item: item[1],
        reverse=True,
    )
    return ", ".join(f"{name}={value:.3f}" for name, value in ordered[:3])


def _should_probe_fastslow(profile: dict[str, float]) -> tuple[bool, str]:
    multiscale = float(profile.get("multiscale_score", 0.0))
    trend = float(profile.get("trend_score", 0.0))
    oscillatory = float(profile.get("oscillatory_score", 0.0))
    closure_need = float(profile.get("closure_need_score", 0.0))
    allowed = multiscale >= 0.55 and (trend >= 0.45 or oscillatory >= 0.55)
    if allowed:
        return True, (
            f"raw preanalysis suggests a structured slow context "
            f"(multiscale={multiscale:.3f}, trend={trend:.3f}, oscillatory={oscillatory:.3f}); "
            f"probe fast/slow as a hypothesis, not as a default."
        )
    return False, (
        f"raw preanalysis does not justify a fast/slow prior "
        f"(multiscale={multiscale:.3f}, trend={trend:.3f}, closure_need={closure_need:.3f}); "
        f"skip fast/slow-specific experiments."
    )


def _preanalysis_for_tasks(
    tasks,
    *,
    seed: int,
    coordinate_kinds: Sequence[str],
    out_dir: Path,
) -> tuple[pd.DataFrame, dict[str, tuple[str, ...]], dict[str, str]]:
    rows: list[dict[str, object]] = []
    task_coordinate_kinds: dict[str, tuple[str, ...]] = {}
    task_reasons: dict[str, str] = {}
    for task in tasks:
        sim = simulate_task(task, seed=seed)
        split = split_series(sim.obs, n_train=task.n_train, n_val=task.n_val, n_test=task.n_test)
        profile_obj = analyze_signal_properties(split["train"])
        profile = profile_obj.to_dict()
        allow_fastslow, reason = _should_probe_fastslow(profile)
        recommended_coordinates = tuple(
            kind for kind in coordinate_kinds if kind != "fastslow" or allow_fastslow
        )
        task_coordinate_kinds[task.name] = recommended_coordinates
        task_reasons[task.name] = reason
        rows.append(
            {
                "task": task.name,
                "system": task.system,
                "task_family": task.family,
                "task_regime": task.regime,
                **profile,
                "dominant_axes": _top_axes_text(profile),
                "fastslow_coordinate_hypothesis": bool(allow_fastslow),
                "recommended_coordinates": "; ".join(recommended_coordinates),
                "reason": reason,
            }
        )
    df = pd.DataFrame(rows).sort_values("task").reset_index(drop=True) if rows else pd.DataFrame()
    ensure_dir(out_dir)
    (out_dir / "preanalysis_summary.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    lines = [
        "# Identify-Mode Preanalysis",
        "",
        "This stage inspects raw training observations before factor screening or architecture validation.",
        "",
    ]
    if not df.empty:
        cols = [
            "task",
            "dominant_axes",
            "closure_need_score",
            "fastslow_coordinate_hypothesis",
            "recommended_coordinates",
            "reason",
        ]
        lines.append(df[cols].to_markdown(index=False))
    else:
        lines.append("No tasks were analyzed.")
    (out_dir / "preanalysis_summary.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "preanalysis_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return df, task_coordinate_kinds, task_reasons


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


def _build_factor_readout_variants(
    factor_df: pd.DataFrame,
    *,
    mining_out: Path | None,
    feature_cfg: DynamicsFeatureConfig | None,
) -> tuple[dict[str, dict[str, dict[str, object]]], dict[str, list[str]]]:
    if factor_df.empty or mining_out is None:
        return {}, {}
    task_model_contexts: dict[str, dict[str, dict[str, object]]] = {}
    task_model_names: dict[str, list[str]] = {}
    for _, row in _best_factor_rows(factor_df).iterrows():
        task = str(row.get("task", "")).strip()
        identifier_kind = str(row.get("identifier_kind", "")).strip()
        num_selected = int(row.get("num_selected", 0) or 0)
        if not task or not identifier_kind or num_selected <= 0:
            continue
        library_path = mining_out / task / identifier_kind / "selected_factor_library.json"
        if not library_path.exists():
            raise FileNotFoundError(f"Expected selected factor library not found: {library_path}")
        selected_library = load_selected_factor_library(library_path)
        if not selected_library.selected_factors:
            continue
        base_context = {
            "readout_factor_specs": list(selected_library.selected_factors),
            "readout_identifier_kind": selected_library.identifier_kind,
            "readout_feature_cfg": feature_cfg,
            "readout_source": str(library_path),
            "readout_factor_names": [spec.name for spec in selected_library.selected_factors],
        }
        task_model_contexts[task] = {
            "rc_factor_readout": {
                **base_context,
                "variant_label": f"rc_factor_readout[{selected_library.identifier_kind}]",
            },
            "ngrc_factor_readout": {
                **base_context,
                "variant_label": f"ngrc_factor_readout[{selected_library.identifier_kind}]",
            },
        }
        task_model_names[task] = ["rc_factor_readout", "ngrc_factor_readout"]
    return task_model_contexts, task_model_names


def _fastslow_validation_gate(
    profile_row: pd.Series,
    coordinate_task_df: pd.DataFrame,
    requested_models: Sequence[str],
    preanalysis_reason: str,
) -> TaskGateDecision:
    recommended_coordinates = tuple(str(x) for x in str(profile_row.get("recommended_coordinates", "")).split("; ") if x)
    if not recommended_coordinates:
        recommended_coordinates = ("raw", "delay", "factor")
    dominant_axes = str(profile_row.get("dominant_axes", ""))
    non_fastslow_models = tuple(
        model_name for model_name in requested_models if not get_model_spec(model_name).uses_fastslow
    )
    fastslow_row_df = coordinate_task_df[coordinate_task_df["coordinate"] == "fastslow"]
    if fastslow_row_df.empty:
        reason = preanalysis_reason
        evidence = "fast/slow was not even promoted to coordinate analysis, so fast/slow validation models are skipped."
        return TaskGateDecision(
            task=str(profile_row["task"]),
            fastslow_coordinate_hypothesis=bool(profile_row.get("fastslow_coordinate_hypothesis", False)),
            fastslow_validation_allowed=False,
            recommended_coordinates=recommended_coordinates,
            recommended_models=non_fastslow_models or tuple(requested_models),
            dominant_axes=dominant_axes,
            reason=reason,
            evidence=evidence,
        )

    ordered_closure = coordinate_task_df.sort_values("markov_gain_ratio", na_position="last").reset_index(drop=True)
    ordered_spectral = coordinate_task_df.sort_values("spectral_radius_rmse", na_position="last").reset_index(drop=True)
    ordered_koopman = coordinate_task_df.sort_values("koopman_invariance_score", ascending=False, na_position="last").reset_index(drop=True)
    fastslow_row = fastslow_row_df.iloc[0]
    best_closure = ordered_closure.iloc[0]
    best_spectral = ordered_spectral.iloc[0]
    best_koopman = ordered_koopman.iloc[0]
    wins_any = any(
        row["coordinate"] == "fastslow"
        for row in (best_closure, best_spectral, best_koopman)
    )
    near_best_closure = float(fastslow_row.get("markov_gain_ratio", np.inf)) <= float(best_closure.get("markov_gain_ratio", np.inf)) + 0.05
    near_best_spectral = (
        float(fastslow_row.get("spectral_radius_rmse", np.inf)) <= float(best_spectral.get("spectral_radius_rmse", np.inf)) * 1.25 + 1e-12
        and float(fastslow_row.get("spectral_radius_corr", -np.inf)) >= float(best_spectral.get("spectral_radius_corr", -np.inf)) - 0.05
    )
    near_best_koopman = float(fastslow_row.get("koopman_invariance_score", -np.inf)) >= float(best_koopman.get("koopman_invariance_score", -np.inf)) - 0.01
    multiscale = float(profile_row.get("multiscale_score", 0.0))
    trend = float(profile_row.get("trend_score", 0.0))
    allowed = bool(
        profile_row.get("fastslow_coordinate_hypothesis", False)
        and (
            wins_any
            or (multiscale >= 0.85 and trend >= 0.60 and near_best_closure and near_best_spectral and near_best_koopman)
        )
    )
    if allowed:
        recommended_models = tuple(requested_models)
        reason = (
            "fast/slow remains admissible because it either wins a theory lens or stays close to the best "
            "Markov, Koopman, and spectral candidates under a strong multiscale prior."
        )
    else:
        recommended_models = non_fastslow_models or tuple(requested_models)
        reason = (
            "fast/slow is rejected for validation because the coordinate-level evidence does not support it as the "
            "dominant state representation, even if raw preanalysis saw multiscale structure."
        )
    evidence = (
        f"best_closure={best_closure['coordinate']} (markov_gain_ratio={best_closure.get('markov_gain_ratio', float('nan')):.4g}); "
        f"best_spectral={best_spectral['coordinate']} (spectral_rmse={best_spectral.get('spectral_radius_rmse', float('nan')):.4g}, "
        f"corr={best_spectral.get('spectral_radius_corr', float('nan')):.4g}); "
        f"best_koopman={best_koopman['coordinate']} (koopman_score={best_koopman.get('koopman_invariance_score', float('nan')):.4g}); "
        f"fastslow=(markov_gain_ratio={fastslow_row.get('markov_gain_ratio', float('nan')):.4g}, "
        f"spectral_rmse={fastslow_row.get('spectral_radius_rmse', float('nan')):.4g}, "
        f"spectral_corr={fastslow_row.get('spectral_radius_corr', float('nan')):.4g}, "
        f"koopman_score={fastslow_row.get('koopman_invariance_score', float('nan')):.4g})"
    )
    return TaskGateDecision(
        task=str(profile_row["task"]),
        fastslow_coordinate_hypothesis=bool(profile_row.get("fastslow_coordinate_hypothesis", False)),
        fastslow_validation_allowed=allowed,
        recommended_coordinates=recommended_coordinates,
        recommended_models=recommended_models,
        dominant_axes=dominant_axes,
        reason=reason,
        evidence=evidence,
    )


def _gate_benchmark_models(
    requested_models: Sequence[str],
    preanalysis_df: pd.DataFrame,
    coordinate_df: pd.DataFrame,
    task_reasons: dict[str, str],
) -> tuple[dict[str, list[str]], list[TaskGateDecision]]:
    if preanalysis_df.empty:
        fallback = {task: list(requested_models) for task in sorted(coordinate_df["task"].unique())}
        return fallback, []
    task_model_names: dict[str, list[str]] = {}
    decisions: list[TaskGateDecision] = []
    for _, profile_row in preanalysis_df.iterrows():
        task = str(profile_row["task"])
        task_df = coordinate_df[coordinate_df["task"] == task].reset_index(drop=True)
        decision = _fastslow_validation_gate(
            profile_row=profile_row,
            coordinate_task_df=task_df,
            requested_models=requested_models,
            preanalysis_reason=task_reasons.get(task, ""),
        )
        task_model_names[task] = list(decision.recommended_models)
        decisions.append(decision)
    return task_model_names, decisions


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
                "analysis": (
                    "factors were screened after raw dynamics preanalysis and should be judged by rollout "
                    "generalization plus Koopman-style stability, not one-step fit alone"
                ),
            }
        )
    return rows


def _theory_tier_from_markov(value: float) -> str:
    if not np.isfinite(value):
        return "low"
    if value < 0.05:
        return "high"
    if value < 0.20:
        return "medium"
    return "low"


def _theory_tier_from_koopman(score: float, linear_r2: float) -> str:
    if not np.isfinite(score) or not np.isfinite(linear_r2):
        return "low"
    if score >= 0.95 and linear_r2 >= 0.999:
        return "high"
    if score >= 0.80 and linear_r2 >= 0.99:
        return "medium"
    return "low"


def _theory_tier_from_spectral(corr: float, rmse_value: float) -> str:
    if not np.isfinite(corr) or not np.isfinite(rmse_value):
        return "low"
    if corr >= 0.60 and rmse_value <= 0.08:
        return "high"
    if corr >= 0.35 and rmse_value <= 0.18:
        return "medium"
    return "low"


def _task_theory_evidence(
    task: str,
    coordinate_task_df: pd.DataFrame,
    factor_task_df: pd.DataFrame,
    benchmark_task_df: pd.DataFrame,
    gate_decision: TaskGateDecision | None,
) -> list[dict[str, object]]:
    if coordinate_task_df.empty:
        return []
    closure_row = coordinate_task_df.sort_values("markov_gain_ratio", na_position="last").iloc[0]
    spectral_row = coordinate_task_df.sort_values("spectral_radius_rmse", na_position="last").iloc[0]
    koopman_row = coordinate_task_df.sort_values("koopman_invariance_score", ascending=False, na_position="last").iloc[0]
    rows: list[dict[str, object]] = [
        {
            "task": task,
            "question": "Is this coordinate closer to Markov?",
            "winner": closure_row["coordinate"],
            "tier": _theory_tier_from_markov(float(closure_row.get("markov_gain_ratio", np.inf))),
            "answer": (
                f"Yes, `{closure_row['coordinate']}` is the closest Markov candidate here; "
                f"adding lagged state changes the error ratio by {closure_row.get('markov_gain_ratio', float('nan')):.4g}."
            ),
            "evidence": (
                f"markov_rmse={closure_row.get('markov_rmse', float('nan')):.4g}, "
                f"lagged_rmse={closure_row.get('lagged_rmse', float('nan')):.4g}, "
                f"markov_r2={closure_row.get('markov_r2', float('nan')):.4g}"
            ),
        },
        {
            "task": task,
            "question": "Is this coordinate closer to a linear invariant subspace?",
            "winner": koopman_row["coordinate"],
            "tier": _theory_tier_from_koopman(
                float(koopman_row.get("koopman_invariance_score", np.nan)),
                float(koopman_row.get("koopman_linear_r2", np.nan)),
            ),
            "answer": (
                f"`{koopman_row['coordinate']}` is the closest Koopman-like candidate in this run."
            ),
            "evidence": (
                f"koopman_invariance_score={koopman_row.get('koopman_invariance_score', float('nan')):.4g}, "
                f"koopman_linear_r2={koopman_row.get('koopman_linear_r2', float('nan')):.4g}"
            ),
        },
        {
            "task": task,
            "question": "Does this coordinate preserve local attractor geometry and spectral structure?",
            "winner": spectral_row["coordinate"],
            "tier": _theory_tier_from_spectral(
                float(spectral_row.get("spectral_radius_corr", np.nan)),
                float(spectral_row.get("spectral_radius_rmse", np.nan)),
            ),
            "answer": (
                f"`{spectral_row['coordinate']}` is the best available local spectral-preservation proxy."
            ),
            "evidence": (
                f"spectral_radius_corr={spectral_row.get('spectral_radius_corr', float('nan')):.4g}, "
                f"spectral_radius_rmse={spectral_row.get('spectral_radius_rmse', float('nan')):.4g}, "
                f"max_real_eig_rmse={spectral_row.get('max_real_eig_rmse', float('nan')):.4g}"
            ),
        },
    ]
    if gate_decision is not None:
        rows.append(
            {
                "task": task,
                "question": "Should fast/slow-specific validation models be run?",
                "winner": "allowed" if gate_decision.fastslow_validation_allowed else "rejected",
                "tier": "high",
                "answer": gate_decision.reason,
                "evidence": gate_decision.evidence,
            }
        )
    if not factor_task_df.empty:
        best_factor = factor_task_df.sort_values("validation_score", na_position="last").iloc[0]
        rows.append(
            {
                "task": task,
                "question": "Which factor family survived property-guided prescreen plus validation?",
                "winner": best_factor["selected_factors"],
                "tier": _confidence_tier(_clip01(float(best_factor.get("selected_koopman_score", 0.0)))),
                "answer": (
                    f"Property-guided screening promoted `{best_factor['selected_factors']}` for `{best_factor['identifier_kind']}`."
                ),
                "evidence": (
                    f"final_rmse50={best_factor.get('final_rmse50', float('nan')):.4g}, "
                    f"test_rmse50={best_factor.get('test_rmse50', float('nan')):.4g}, "
                    f"selected_koopman_score={best_factor.get('selected_koopman_score', float('nan')):.4g}"
                ),
            }
        )
    if not benchmark_task_df.empty:
        best_benchmark = benchmark_task_df.sort_values("rmse@50", na_position="last").iloc[0]
        rows.append(
            {
                "task": task,
                "question": "What validation model survived the gated benchmark stage?",
                "winner": best_benchmark["variant"],
                "tier": _confidence_tier(1.0 / (1.0 + float(best_benchmark.get("acf_rmse", 0.0)) + float(best_benchmark.get("psd_rmse", 0.0)))),
                "answer": (
                    f"`{best_benchmark['variant']}` is the best model among the theory-gated validation candidates."
                ),
                "evidence": (
                    f"rmse@50={best_benchmark.get('rmse@50', float('nan')):.4g}, "
                    f"acf_rmse={best_benchmark.get('acf_rmse', float('nan')):.4g}, "
                    f"psd_rmse={best_benchmark.get('psd_rmse', float('nan')):.4g}"
                ),
            }
        )
    return rows


def _render_theory_evidence_report(evidence_rows: list[dict[str, object]], include_header: bool = True) -> str:
    lines = []
    if include_header:
        lines.extend(
            [
                "# Theory-Aware Evidence",
                "",
                "These answers are provisional and must still pass the expert-review gate.",
                "",
            ]
        )
    if not evidence_rows:
        lines.append("No theory-aware evidence rows were generated.")
        return "\n".join(lines)
    evidence_df = pd.DataFrame(evidence_rows)
    for task, task_df in evidence_df.groupby("task"):
        lines.extend([f"## {task}", ""])
        for _, row in task_df.iterrows():
            lines.append(f"- {row['question']}")
            lines.append(f"  Answer: {row['answer']}")
            lines.append(f"  Confidence: {row['tier']}")
            lines.append(f"  Evidence: {row['evidence']}")
        lines.append("")
    return "\n".join(lines)


def _build_confidence_report(
    benchmark_df: pd.DataFrame,
    coordinate_df: pd.DataFrame,
    factor_df: pd.DataFrame,
    theory_evidence_rows: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    report = _benchmark_confidence(benchmark_df) + _coordinate_confidence(coordinate_df) + _factor_confidence(factor_df)
    if theory_evidence_rows:
        for row in theory_evidence_rows:
            report.append(
                {
                    "task": row["task"],
                    "section": "theory_evidence",
                    "winner": row["winner"],
                    "score": {"high": 0.9, "medium": 0.6, "low": 0.3}.get(str(row["tier"]), 0.3),
                    "tier": row["tier"],
                    "evidence": row["evidence"],
                    "analysis": row["answer"],
                }
            )
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
    preanalysis_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    coordinate_df: pd.DataFrame,
    factor_df: pd.DataFrame,
    manifest: dict[str, str],
    confidence_rows: list[dict[str, object]],
    gate_decisions: list[TaskGateDecision],
    theory_evidence_rows: list[dict[str, object]],
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
    if not preanalysis_df.empty:
        lines.extend(
            [
                "## Preanalysis Gate",
                "",
                preanalysis_df[
                    [
                        "task",
                        "dominant_axes",
                        "closure_need_score",
                        "fastslow_coordinate_hypothesis",
                        "recommended_coordinates",
                    ]
                ].to_markdown(index=False),
                "",
            ]
        )
    if not benchmark_df.empty:
        best_bench = _best_benchmark_rows(benchmark_df)
        bench_cols = ["task", "variant", "rmse@50", "acf_rmse", "psd_rmse"]
        if "readout_identifier_kind" in best_bench.columns:
            has_factor_readout = best_bench["readout_identifier_kind"].fillna("").astype(str).str.len().gt(0).any()
            if has_factor_readout:
                bench_cols = ["task", "variant", "readout_identifier_kind", "readout_factor_names", "rmse@50", "acf_rmse", "psd_rmse"]
        lines.extend(
            [
                "## Best benchmark variants",
                "",
                best_bench[bench_cols].to_markdown(index=False),
                "",
            ]
        )
    if gate_decisions:
        gate_df = pd.DataFrame(
            [
                {
                    "task": decision.task,
                    "fastslow_validation_allowed": decision.fastslow_validation_allowed,
                    "recommended_models": "; ".join(decision.recommended_models),
                    "reason": decision.reason,
                }
                for decision in gate_decisions
            ]
        )
        lines.extend(
            [
                "## Validation Gate",
                "",
                gate_df.to_markdown(index=False),
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
    if theory_evidence_rows:
        lines.extend(
            [
                "## Theory-aware evidence",
                "",
                _render_theory_evidence_report(theory_evidence_rows, include_header=False),
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
    tasks = _build_tasks(suite, task_names=task_names)

    preanalysis_df = pd.DataFrame()
    benchmark_df = pd.DataFrame()
    coordinate_df = pd.DataFrame()
    factor_df = pd.DataFrame()
    manifest: dict[str, str] = {}
    task_coordinate_kinds: dict[str, tuple[str, ...]] | None = None
    task_reasons: dict[str, str] = {}
    gate_decisions: list[TaskGateDecision] = []
    theory_evidence_rows: list[dict[str, object]] = []
    feature_cfg: DynamicsFeatureConfig | None = None
    mining_out: Path | None = None

    if mining_mode == "identify" and tasks:
        preanalysis_out = out_path / "preanalysis"
        preanalysis_df, task_coordinate_kinds, task_reasons = _preanalysis_for_tasks(
            tasks,
            seed=seed,
            coordinate_kinds=coordinate_kinds,
            out_dir=preanalysis_out,
        )
        manifest["preanalysis"] = str(preanalysis_out)

    if not skip_coordinate_analysis:
        coordinate_out = out_path / "coordinate_analysis"
        ensure_dir(coordinate_out)
        coordinate_df = run_coordinate_analysis_suite(
            suite=suite,
            out_dir=str(coordinate_out),
            seed=seed,
            task_names=task_names,
            coordinate_kinds=coordinate_kinds,
            task_coordinate_kinds=task_coordinate_kinds,
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
            # In identify mode, the CLI flag must override config so that property-guided prescreen
            # remains the default unless full-library search is explicitly requested.
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

    if not skip_benchmarks:
        benchmark_out = out_path / "benchmarks"
        ensure_dir(benchmark_out)
        model_names = expand_model_group_names(model_groups or ["fastslow_ablation"])
        task_model_contexts, task_factor_models = _build_factor_readout_variants(
            factor_df,
            mining_out=mining_out,
            feature_cfg=feature_cfg,
        )
        task_model_names = None
        if mining_mode == "identify" and not coordinate_df.empty and not preanalysis_df.empty:
            task_model_names, gate_decisions = _gate_benchmark_models(
                requested_models=model_names,
                preanalysis_df=preanalysis_df,
                coordinate_df=coordinate_df,
                task_reasons=task_reasons,
            )
        for task in task_factor_models:
            base_models = list(task_model_names.get(task, model_names)) if task_model_names is not None else list(model_names)
            for factor_model in task_factor_models[task]:
                if factor_model not in base_models:
                    base_models.append(factor_model)
            if task_model_names is None:
                task_model_names = {}
            task_model_names[task] = base_models
        if gate_decisions:
            gate_payload = [
                {
                    "task": decision.task,
                    "fastslow_coordinate_hypothesis": decision.fastslow_coordinate_hypothesis,
                    "fastslow_validation_allowed": decision.fastslow_validation_allowed,
                    "recommended_coordinates": list(decision.recommended_coordinates),
                    "recommended_models": list(decision.recommended_models),
                    "factor_readout_variants": list(task_factor_models.get(decision.task, [])),
                    "final_benchmark_models": list(task_model_names.get(decision.task, decision.recommended_models)) if task_model_names is not None else list(decision.recommended_models),
                    "dominant_axes": decision.dominant_axes,
                    "reason": decision.reason,
                    "evidence": decision.evidence,
                }
                for decision in gate_decisions
            ]
            (out_path / "validation_gate.json").write_text(
                json.dumps(gate_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            manifest["validation_gate"] = str(out_path / "validation_gate.json")
        benchmark_df = run_benchmark_suite(
            suite=suite,
            seed=seed,
            out_dir=str(benchmark_out),
            model_names=model_names,
            task_model_names=task_model_names,
            task_model_contexts=task_model_contexts or None,
            grid_mode=grid_mode,
            task_names=task_names,
        )
        manifest["benchmarks"] = str(benchmark_out)

    gate_by_task = {decision.task: decision for decision in gate_decisions}
    task_order = [task.name for task in tasks]
    for task in task_order:
        theory_evidence_rows.extend(
            _task_theory_evidence(
                task=task,
                coordinate_task_df=coordinate_df[coordinate_df["task"] == task].reset_index(drop=True),
                factor_task_df=factor_df[factor_df["task"] == task].reset_index(drop=True),
                benchmark_task_df=benchmark_df[benchmark_df["task"] == task].reset_index(drop=True),
                gate_decision=gate_by_task.get(task),
            )
        )
    theory_evidence_path = out_path / "theory_evidence.md"
    theory_evidence_path.write_text(
        _render_theory_evidence_report(theory_evidence_rows),
        encoding="utf-8",
    )
    manifest["theory_evidence"] = str(theory_evidence_path)

    confidence_rows = _build_confidence_report(
        benchmark_df=benchmark_df,
        coordinate_df=coordinate_df,
        factor_df=factor_df,
        theory_evidence_rows=theory_evidence_rows,
    )
    loop_summary = _render_loop_summary(
        suite=suite,
        loop_mode=mining_mode,
        preanalysis_df=preanalysis_df,
        benchmark_df=benchmark_df,
        coordinate_df=coordinate_df,
        factor_df=factor_df,
        manifest=manifest,
        confidence_rows=confidence_rows,
        gate_decisions=gate_decisions,
        theory_evidence_rows=theory_evidence_rows,
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
        "preanalysis": preanalysis_df,
        "benchmarks": benchmark_df,
        "coordinate_analysis": coordinate_df,
        "factor_mining": factor_df,
        "manifest": manifest,
        "confidence_report": confidence_rows,
        "gate_decisions": gate_decisions,
        "theory_evidence": theory_evidence_rows,
    }
