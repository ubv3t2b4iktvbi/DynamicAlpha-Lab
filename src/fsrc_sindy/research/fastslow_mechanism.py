from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

from ..benchmarks import build_suite
from ..factors import analyze_signal_properties
from ..fastslow import CausalFastSlowEncoder, FastSlowConfig
from ..models import NGRCConfig, PureNGRCModel, PureRCModel, RCConfig, ReservoirTemplateFactory
from ..selection import instantiate_model
from ..systems import BenchmarkTask, simulate_task, split_series, task_metadata_columns
from ..utils import ensure_dir
from .fastslow_validation import (
    summarize_fastslow_benchmarks,
    summarize_fastslow_coordinates,
    summarize_fastslow_mechanism_sweeps,
)
from .loop import run_research_loop


RAW_FEATURE_COLUMNS = (
    "oscillatory_score",
    "multiscale_score",
    "trend_score",
    "burstiness_score",
    "unpredictability_score",
    "closure_need_score",
    "dominant_frequency_power",
    "lag1_autocorr",
    "train_obs_std",
    "train_diff_std",
    "train_abs_diff_mean",
    "train_range",
)

MECHANISM_METRIC_COLUMNS = (
    "rc_fastslow_gain_pct",
    "ngrc_fastslow_gain_pct",
    "mean_fastslow_gain_pct",
    "fastslow_markov_gain_ratio",
    "fastslow_spectral_corr",
    "fastslow_spectral_rmse",
    "fastslow_koopman_score",
)

GATE_STRATEGY_ORDER = (
    "oracle_single_probe",
    "estimated_single_probe",
    "estimated_probe_grid",
    "exhaustive_grid",
)

BETA_GAMMA_FEATURE_NAMES = (
    "fast",
    "slow",
    "m",
    "resid",
    "ds",
    "dfast",
)

BETA_GAMMA_SCORE_COLUMNS = (
    "beta_gamma_viability_score",
    "beta_gamma_strength_proxy",
)

ROW_GATE_STRATEGY_ORDER = (
    "oracle_viable_only",
    "beta_gamma_viability_only",
    "beta_gamma_two_stage",
    "effective_gain_two_stage",
    "always_use_fastslow",
    "never_use_fastslow",
)


def _benchmark_csv_path(seed_dir: Path) -> Path:
    return seed_dir / "benchmarks" / "benchmark_results.csv"


def _context_len_for_task(task: BenchmarkTask) -> int:
    return max(200, 4 * max(task.selection_horizons))


def _canonical_encoder_grid(task: BenchmarkTask) -> list[FastSlowConfig]:
    fast_steps = (4.0, 6.0, 8.0)
    slow_bases = (12.0, 16.0, 24.0)
    return [
        FastSlowConfig(t0=fast_step, slow_scales=(slow_base, 2.0 * slow_base, 4.0 * slow_base), dt=float(task.dt))
        for fast_step in fast_steps
        for slow_base in slow_bases
    ]


def _load_best_config(payload: object) -> dict[str, Any]:
    if isinstance(payload, dict):
        return dict(payload)
    text = str(payload).strip()
    if not text:
        return {}
    loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected best_config payload to decode to dict, got {type(loaded)!r}")
    return loaded


def _raw_cfg_from_payload(model_name: str, payload: object) -> Any:
    cfg_dict = _load_best_config(payload)
    cfg_dict = {key: value for key, value in cfg_dict.items() if value is not None}
    if model_name == "rc_raw":
        return RCConfig(**cfg_dict)
    if model_name == "ngrc_raw":
        return NGRCConfig(**cfg_dict)
    raise ValueError(f"Unsupported raw model_name={model_name}")


def _instantiate_fitted_raw_model(
    *,
    model_name: str,
    cfg_payload: object,
    y_train: np.ndarray,
    seed: int,
    short_train: bool,
) -> PureRCModel | PureNGRCModel:
    cfg = _raw_cfg_from_payload(model_name, cfg_payload)
    template_factory = ReservoirTemplateFactory(seed=int(seed))
    model = instantiate_model(
        model_name,
        cfg,
        template_factory,
        short_train=short_train,
        model_context=None,
    )
    fitted = model.fit(y_train)
    if not isinstance(fitted, (PureRCModel, PureNGRCModel)):
        raise TypeError(f"Unexpected fitted model type for {model_name}: {type(fitted)!r}")
    return fitted


def _rc_one_step_trace(model: PureRCModel, series: np.ndarray, burn_in: int) -> dict[str, np.ndarray]:
    ys = ((np.asarray(series, dtype=float).reshape(-1) - model.mu_) / model.std_).astype(float)
    _, factor_mat = model.readout.transform(ys)
    r = np.zeros(model.cfg.n_reservoir, dtype=float)
    preds: list[float] = []
    truth: list[float] = []
    eval_indices: list[int] = []
    start_eval = min(max(int(burn_in), model.cfg.washout), len(ys) - 2)
    for t in range(len(ys) - 1):
        r = model._step(r, ys[t])
        if t >= start_eval:
            pred_std = float(model._readout_aug(r, ys[t], factor_mat=factor_mat, t=t) @ model.Wout)
            preds.append(pred_std)
            truth.append(float(ys[t + 1]))
            eval_indices.append(int(t))
    preds_arr = np.asarray(preds, dtype=float)
    truth_arr = np.asarray(truth, dtype=float)
    return {
        "series_std": ys,
        "eval_indices": np.asarray(eval_indices, dtype=int),
        "pred_std": preds_arr,
        "truth_std": truth_arr,
        "residual_std": truth_arr - preds_arr,
    }


def _ngrc_one_step_trace(model: PureNGRCModel, series: np.ndarray, burn_in: int) -> dict[str, np.ndarray]:
    ys = ((np.asarray(series, dtype=float).reshape(-1) - model.mu_) / model.std_).astype(float)
    _, factor_mat = model.readout.transform(ys)
    preds: list[float] = []
    truth: list[float] = []
    eval_indices: list[int] = []
    start_eval = max(int(burn_in), model.cfg.washout, model.delay.max_lag)
    for t in range(start_eval, len(ys) - 1):
        delay_row = model.delay.row_from_series(ys, t)
        x = model._feature_row(delay_row, factor_mat=factor_mat, t=t)
        pred_std = float(np.clip(x @ model.coef_, -model.cfg.y_clip, model.cfg.y_clip))
        preds.append(pred_std)
        truth.append(float(ys[t + 1]))
        eval_indices.append(int(t))
    preds_arr = np.asarray(preds, dtype=float)
    truth_arr = np.asarray(truth, dtype=float)
    return {
        "series_std": ys,
        "eval_indices": np.asarray(eval_indices, dtype=int),
        "pred_std": preds_arr,
        "truth_std": truth_arr,
        "residual_std": truth_arr - preds_arr,
    }


def _one_step_trace(model: PureRCModel | PureNGRCModel, series: np.ndarray, burn_in: int) -> dict[str, np.ndarray]:
    if isinstance(model, PureRCModel):
        return _rc_one_step_trace(model, series, burn_in)
    if isinstance(model, PureNGRCModel):
        return _ngrc_one_step_trace(model, series, burn_in)
    raise TypeError(f"Unsupported model type for one-step trace: {type(model)!r}")


def _feature_matrix_from_trace(series_std: np.ndarray, eval_indices: np.ndarray, fs_cfg: FastSlowConfig) -> np.ndarray:
    encoder = CausalFastSlowEncoder(fs_cfg)
    feats = encoder.build_feature_sequence(series_std)
    cols = [np.asarray(feats[name], dtype=float)[eval_indices] for name in BETA_GAMMA_FEATURE_NAMES]
    return np.column_stack(cols)


def _beta_gamma_from_trace(trace: dict[str, np.ndarray], fs_cfg: FastSlowConfig) -> dict[str, float]:
    eval_indices = np.asarray(trace["eval_indices"], dtype=int)
    residual = np.asarray(trace["residual_std"], dtype=float).reshape(-1)
    if len(eval_indices) == 0 or len(residual) == 0:
        return {}
    feature_mat = _feature_matrix_from_trace(np.asarray(trace["series_std"], dtype=float), eval_indices, fs_cfg)
    mask = np.isfinite(residual)
    if feature_mat.ndim != 2:
        return {}
    mask &= np.all(np.isfinite(feature_mat), axis=1)
    U = feature_mat[mask]
    e = residual[mask]
    if len(e) < max(16, U.shape[1] + 2):
        return {}
    e_center = e - float(np.mean(e))
    U_mean = np.mean(U, axis=0)
    U_std = np.std(U, axis=0)
    U_std = np.where(U_std > 1.0e-12, U_std, 1.0)
    U_norm = (U - U_mean) / U_std
    c = np.mean(U_norm * e_center[:, None], axis=0)
    S = (U_norm.T @ U_norm) / float(len(U_norm))
    beta = float(np.dot(c, c))
    gamma = float(c @ S @ c)
    beta = max(beta, 0.0)
    gamma = max(gamma, 1.0e-12)
    viability = float((beta * beta) / (4.0 * gamma))
    strength = float(np.sqrt(beta / (2.0 * gamma))) if beta > 0.0 else 0.0
    return {
        "beta": beta,
        "gamma": gamma,
        "viability_score": viability,
        "strength_proxy": strength,
        "num_points": float(len(U_norm)),
        "residual_std": float(np.std(e_center)),
    }


def _aggregate_beta_gamma_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    weights = np.asarray([max(float(row.get("num_points", 0.0)), 1.0) for row in rows], dtype=float)
    beta = np.asarray([float(row.get("beta", np.nan)) for row in rows], dtype=float)
    gamma = np.asarray([float(row.get("gamma", np.nan)) for row in rows], dtype=float)
    resid_std = np.asarray([float(row.get("residual_std", np.nan)) for row in rows], dtype=float)
    mask = np.isfinite(beta) & np.isfinite(gamma) & (gamma > 0.0)
    if not mask.any():
        return {}
    weights = weights[mask]
    weights = weights / np.sum(weights)
    beta_mean = float(np.sum(weights * beta[mask]))
    gamma_mean = float(np.sum(weights * gamma[mask]))
    viability = float((beta_mean * beta_mean) / (4.0 * max(gamma_mean, 1.0e-12)))
    strength = float(np.sqrt(beta_mean / (2.0 * max(gamma_mean, 1.0e-12)))) if beta_mean > 0.0 else 0.0
    return {
        "beta": beta_mean,
        "gamma": gamma_mean,
        "viability_score": viability,
        "strength_proxy": strength,
        "residual_std": float(np.sum(weights * resid_std[mask])) if np.isfinite(resid_std[mask]).any() else float("nan"),
        "num_families": float(mask.sum()),
    }


def _compute_row_beta_gamma_stats(
    *,
    benchmark_task_df: pd.DataFrame,
    task: BenchmarkTask,
    seed: int,
) -> dict[str, object]:
    if benchmark_task_df.empty:
        return {}
    sim = simulate_task(task, seed=seed)
    split = split_series(sim.obs, n_train=task.n_train, n_val=task.n_val, n_test=task.n_test)
    y_train = np.asarray(split["train"], dtype=float)
    y_val = np.asarray(split["val"], dtype=float)
    context_len = _context_len_for_task(task)
    val_series = np.concatenate([y_train[-context_len:], y_val])
    short_train = task.n_train < 2000
    family_rows: list[dict[str, float]] = []
    best_row: dict[str, object] | None = None
    best_viability = float("-inf")

    for model_name in ("rc_raw", "ngrc_raw"):
        model_row = benchmark_task_df[benchmark_task_df["base_model_name"] == model_name]
        if model_row.empty:
            continue
        fitted = _instantiate_fitted_raw_model(
            model_name=model_name,
            cfg_payload=model_row.iloc[0]["best_config"],
            y_train=y_train,
            seed=int(seed),
            short_train=short_train,
        )
        trace = _one_step_trace(fitted, val_series, context_len)
        if len(trace["residual_std"]) == 0:
            continue
        per_encoder: list[dict[str, object]] = []
        for fs_cfg in _canonical_encoder_grid(task):
            stats = _beta_gamma_from_trace(trace, fs_cfg)
            if not stats:
                continue
            row = {
                "model_name": model_name,
                "encoder_label": (
                    f"t0={fs_cfg.fast_step_equivalent:.1f}|slow="
                    + ",".join(f"{step:.1f}" for step in fs_cfg.slow_step_equivalents)
                ),
                **stats,
            }
            per_encoder.append(row)
        if not per_encoder:
            continue
        family_best = max(per_encoder, key=lambda row: float(row["viability_score"]))
        family_rows.append(dict(family_best))
        if float(family_best["viability_score"]) > best_viability:
            best_viability = float(family_best["viability_score"])
            best_row = dict(family_best)

    aggregate = _aggregate_beta_gamma_rows(family_rows)
    if not aggregate:
        return {}
    row: dict[str, object] = {
        "beta_gamma_model_count": int(len(family_rows)),
        "beta_gamma_best_encoder": str(best_row["encoder_label"]) if best_row is not None else "",
        "beta_gamma_best_model_family": str(best_row["model_name"]) if best_row is not None else "",
    }
    for family_row in family_rows:
        family_name = str(family_row["model_name"]).replace("_raw", "")
        row[f"{family_name}_beta"] = float(family_row["beta"])
        row[f"{family_name}_gamma"] = float(family_row["gamma"])
        row[f"{family_name}_viability_score"] = float(family_row["viability_score"])
        row[f"{family_name}_strength_proxy"] = float(family_row["strength_proxy"])
        row[f"{family_name}_beta_gamma_encoder"] = str(family_row["encoder_label"])
    row["beta_gamma_beta"] = float(aggregate["beta"])
    row["beta_gamma_gamma"] = float(aggregate["gamma"])
    row["beta_gamma_viability_score"] = float(aggregate["viability_score"])
    row["beta_gamma_strength_proxy"] = float(aggregate["strength_proxy"])
    row["beta_gamma_residual_std"] = float(aggregate["residual_std"])
    row["beta_gamma_num_families"] = int(aggregate["num_families"])
    row["beta_gamma_log_beta"] = float(np.log10(max(float(aggregate["beta"]), 1.0e-12)))
    row["beta_gamma_log_gamma"] = float(np.log10(max(float(aggregate["gamma"]), 1.0e-12)))
    row["beta_gamma_log_viability"] = float(np.log10(max(float(aggregate["viability_score"]), 1.0e-12)))
    row["beta_gamma_log_strength"] = float(np.log10(max(float(aggregate["strength_proxy"]), 1.0e-12)))
    return row


def _attach_beta_gamma_stats(
    seed_df: pd.DataFrame,
    *,
    suite: str,
    out_path: Path,
    seeds: Sequence[int],
    task_names: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if seed_df.empty:
        return seed_df.copy(), pd.DataFrame()
    tasks = build_suite(suite)
    if task_names is not None:
        selected = set(task_names)
        tasks = [task for task in tasks if task.name in selected]
    task_lookup = {task.name: task for task in tasks}
    stats_rows: list[dict[str, object]] = []
    benchmark_cache: dict[int, pd.DataFrame] = {}
    for seed in seeds:
        csv_path = _benchmark_csv_path(out_path / f"seed_{seed}")
        if not csv_path.exists():
            continue
        benchmark_cache[int(seed)] = pd.read_csv(csv_path)
    for row in seed_df.itertuples(index=False):
        task_name = str(getattr(row, "task"))
        seed_value = int(getattr(row, "seed"))
        task = task_lookup.get(task_name)
        benchmark_df = benchmark_cache.get(seed_value)
        if task is None or benchmark_df is None:
            continue
        benchmark_task_df = benchmark_df[benchmark_df["task"] == task_name].reset_index(drop=True)
        stats = _compute_row_beta_gamma_stats(
            benchmark_task_df=benchmark_task_df,
            task=task,
            seed=seed_value,
        )
        if not stats:
            continue
        stats_rows.append(
            {
                "task": task_name,
                "seed": seed_value,
                **stats,
            }
        )
    stats_df = pd.DataFrame(stats_rows)
    if stats_df.empty:
        return seed_df.copy(), stats_df
    merged = seed_df.merge(stats_df, on=["task", "seed"], how="left")
    return merged, stats_df


def _finite_or_nan(value: object) -> float:
    try:
        cast = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return cast if np.isfinite(cast) else float("nan")


def _task_observable_features(task: BenchmarkTask, seed: int) -> dict[str, object]:
    sim = simulate_task(task, seed=seed)
    split = split_series(sim.obs, n_train=task.n_train, n_val=task.n_val, n_test=task.n_test)
    y_train = np.asarray(split["train"], dtype=float)
    profile = analyze_signal_properties(y_train).to_dict()
    dy = np.diff(y_train)
    row = {
        "task": task.name,
        "system": task.system,
        "task_family": task.family,
        "task_regime": task.regime,
        "seed": int(seed),
        "train_obs_std": float(np.std(y_train)),
        "train_diff_std": float(np.std(dy)) if len(dy) else 0.0,
        "train_abs_diff_mean": float(np.mean(np.abs(dy))) if len(dy) else 0.0,
        "train_range": float(np.max(y_train) - np.min(y_train)),
    }
    row.update(task_metadata_columns(task))
    row.update(profile)
    return row


def _fit_poly_ridge(
    df: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_column: str,
    alpha: float = 1.0e-3,
    degree: int = 2,
) -> tuple[PolynomialFeatures, Ridge, np.ndarray, np.ndarray]:
    train_df = df[np.isfinite(df[target_column].to_numpy(dtype=float))].reset_index(drop=True)
    if train_df.empty:
        raise ValueError(f"No finite rows available for target_column={target_column}.")
    X = train_df.loc[:, feature_columns].to_numpy(dtype=float)
    y = train_df.loc[:, target_column].to_numpy(dtype=float)
    mean = np.nanmean(X, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    X_imputed = np.where(np.isfinite(X), X, mean)
    std = np.nanstd(X_imputed, axis=0)
    std = np.where(np.isfinite(std) & (std > 0.0), std, 1.0)
    X_norm = (X_imputed - mean) / std
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_norm)
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_poly, y)
    return poly, model, mean, std


def _predict_poly_ridge(
    df: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    poly: PolynomialFeatures,
    model: Ridge,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    X = df.loc[:, feature_columns].to_numpy(dtype=float)
    X_imputed = np.where(np.isfinite(X), X, mean)
    X_norm = (X_imputed - mean) / std
    return np.asarray(model.predict(poly.transform(X_norm)), dtype=float)


def _fit_leave_one_group_out(
    df: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_column: str,
    group_column: str,
    alpha: float = 1.0e-3,
    degree: int = 2,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    predictions = np.full(len(df), np.nan, dtype=float)
    rows: list[dict[str, object]] = []
    for group_name in sorted(df[group_column].dropna().unique()):
        train_df = df[df[group_column] != group_name].reset_index(drop=True)
        test_df = df[df[group_column] == group_name].reset_index(drop=True)
        if train_df.empty or test_df.empty:
            continue
        try:
            poly, model, mean, std = _fit_poly_ridge(
                train_df,
                feature_columns=feature_columns,
                target_column=target_column,
                alpha=alpha,
                degree=degree,
            )
        except ValueError:
            continue
        pred = _predict_poly_ridge(
            test_df,
            feature_columns=feature_columns,
            poly=poly,
            model=model,
            mean=mean,
            std=std,
        )
        predictions[df[group_column] == group_name] = pred
        truth = test_df[target_column].to_numpy(dtype=float)
        mask = np.isfinite(pred) & np.isfinite(truth)
        if mask.sum() == 0:
            rmse = float("nan")
            corr = float("nan")
        else:
            rmse = float(np.sqrt(np.mean(np.square(pred[mask] - truth[mask]))))
            corr = (
                float(np.corrcoef(pred[mask], truth[mask])[0, 1])
                if mask.sum() > 1 and np.std(pred[mask]) > 1e-12 and np.std(truth[mask]) > 1e-12
                else float("nan")
            )
        rows.append(
            {
                group_column: group_name,
                f"{target_column}_rmse": rmse,
                f"{target_column}_corr": corr,
                "num_rows": int(len(test_df)),
            }
        )
    return predictions, rows


def _fit_leave_one_seed_out(
    df: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_column: str,
    seed_column: str = "seed",
    alpha: float = 1.0e-3,
    degree: int = 2,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    predictions = np.full(len(df), np.nan, dtype=float)
    rows: list[dict[str, object]] = []
    if seed_column not in df.columns:
        return predictions, rows
    for seed_value in sorted(df[seed_column].dropna().unique()):
        test_mask = df[seed_column] == seed_value
        train_df = df[~test_mask].reset_index(drop=True)
        test_df = df[test_mask].reset_index(drop=True)
        if train_df.empty or test_df.empty:
            continue
        try:
            poly, model, mean, std = _fit_poly_ridge(
                train_df,
                feature_columns=feature_columns,
                target_column=target_column,
                alpha=alpha,
                degree=degree,
            )
        except ValueError:
            continue
        pred = _predict_poly_ridge(
            test_df,
            feature_columns=feature_columns,
            poly=poly,
            model=model,
            mean=mean,
            std=std,
        )
        predictions[np.flatnonzero(test_mask.to_numpy())] = pred
        truth = test_df[target_column].to_numpy(dtype=float)
        mask = np.isfinite(pred) & np.isfinite(truth)
        rmse = float(np.sqrt(np.mean(np.square(pred[mask] - truth[mask])))) if mask.any() else float("nan")
        corr = (
            float(np.corrcoef(pred[mask], truth[mask])[0, 1])
            if mask.sum() > 1 and np.std(pred[mask]) > 1e-12 and np.std(truth[mask]) > 1e-12
            else float("nan")
        )
        rows.append(
            {
                seed_column: int(seed_value),
                f"{target_column}_rmse": rmse,
                f"{target_column}_corr": corr,
                "num_rows": int(len(test_df)),
            }
        )
    return predictions, rows


def _suggest_probe_grid(estimate: float, observed_levels: Sequence[float]) -> list[float]:
    levels = np.asarray(sorted({float(level) for level in observed_levels if np.isfinite(level)}), dtype=float)
    if len(levels) == 0 or not np.isfinite(estimate):
        return []
    if len(levels) == 1:
        return [float(levels[0])]
    step = float(np.median(np.diff(levels)))
    lo = float(np.clip(estimate - 0.5 * step, levels.min(), levels.max()))
    mid = float(np.clip(estimate, levels.min(), levels.max()))
    hi = float(np.clip(estimate + 0.5 * step, levels.min(), levels.max()))
    return [round(lo, 4), round(mid, 4), round(hi, 4)]


def _gate_policy(estimated_hsf: float, predicted_gain_pct: float) -> str:
    if not np.isfinite(predicted_gain_pct):
        return "insufficient_evidence"
    if predicted_gain_pct <= 0.0:
        return "prefer_raw_or_delay"
    if np.isfinite(estimated_hsf) and estimated_hsf < 0.6:
        return "weak_gate_probe_only"
    if np.isfinite(estimated_hsf) and estimated_hsf <= 1.25:
        return "moderate_fastslow_candidate"
    return "strong_gate_use_memory_controls"


def _nearest_observed_level(estimate: float, observed_levels: Sequence[float]) -> float:
    levels = np.asarray(sorted({float(level) for level in observed_levels if np.isfinite(level)}), dtype=float)
    if len(levels) == 0 or not np.isfinite(estimate):
        return float("nan")
    idx = int(np.argmin(np.abs(levels - float(estimate))))
    return float(levels[idx])


def _select_row_by_level(group_df: pd.DataFrame, coupling: float) -> pd.Series | None:
    if not np.isfinite(coupling) or "slow_to_fast_coupling" not in group_df.columns:
        return None
    diffs = np.abs(group_df["slow_to_fast_coupling"].to_numpy(dtype=float) - float(coupling))
    if not np.isfinite(diffs).any():
        return None
    idx = int(np.nanargmin(diffs))
    return group_df.iloc[idx]


def _select_best_row(group_df: pd.DataFrame, score_column: str) -> pd.Series | None:
    if group_df.empty or score_column not in group_df.columns:
        return None
    valid = group_df[np.isfinite(group_df[score_column].to_numpy(dtype=float))].copy()
    if valid.empty:
        return None
    return valid.sort_values(score_column, ascending=False, na_position="last").iloc[0]


def _evaluate_gate_strategies(estimator_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if estimator_df.empty or "mechanism_system" not in estimator_df.columns:
        return pd.DataFrame(), pd.DataFrame()
    decision_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for system_name, system_df in estimator_df.groupby("mechanism_system"):
        observed_levels = sorted(system_df["slow_to_fast_coupling"].dropna().astype(float).unique())
        feature_columns = [col for col in RAW_FEATURE_COLUMNS if col in system_df.columns]
        if not feature_columns or "seed" not in system_df.columns:
            continue
        h_pred, _ = _fit_leave_one_seed_out(
            system_df,
            feature_columns=feature_columns,
            target_column="slow_to_fast_coupling",
        )
        gain_pred, _ = _fit_leave_one_seed_out(
            system_df,
            feature_columns=feature_columns,
            target_column="mean_fastslow_gain_pct",
        )
        system_eval = system_df.reset_index(drop=True).copy()
        system_eval["within_system_estimated_hsf"] = h_pred
        system_eval["within_system_predicted_gain_pct"] = gain_pred
        group_cols = [
            col
            for col in ("seed", "mechanism_system", "observation_family", "observability_profile", "noise_profile")
            if col in system_eval.columns
        ]
        if len(group_cols) <= 1:
            group_cols = [col for col in ("seed", "system") if col in system_eval.columns]
        for _, group_df in system_eval.groupby(group_cols, dropna=False):
            oracle_row = _select_best_row(group_df, "mean_fastslow_gain_pct")
            if oracle_row is None:
                continue
            exhaustive_probe_count = int(group_df["slow_to_fast_coupling"].nunique()) if "slow_to_fast_coupling" in group_df.columns else len(group_df)
            estimated_hsf = float(group_df["within_system_estimated_hsf"].dropna().iloc[0]) if group_df["within_system_estimated_hsf"].notna().any() else float("nan")
            nearest_level = _nearest_observed_level(estimated_hsf, observed_levels)
            single_probe_row = _select_row_by_level(group_df, nearest_level)
            suggested_levels = _suggest_probe_grid(estimated_hsf, observed_levels)
            probe_rows = pd.concat(
                [
                    group_df[np.isclose(group_df["slow_to_fast_coupling"].astype(float), level, atol=1.0e-9)]
                    for level in suggested_levels
                ],
                axis=0,
            ).drop_duplicates() if suggested_levels else pd.DataFrame()
            probe_best_row = _select_best_row(probe_rows, "mean_fastslow_gain_pct") if not probe_rows.empty else None
            strategy_rows = [
                ("oracle_single_probe", oracle_row, 1, float(oracle_row["slow_to_fast_coupling"]), [float(oracle_row["slow_to_fast_coupling"])]),
                ("estimated_single_probe", single_probe_row, 1, float(nearest_level), [float(nearest_level)] if np.isfinite(nearest_level) else []),
                ("estimated_probe_grid", probe_best_row, len(suggested_levels), float(estimated_hsf), [float(level) for level in suggested_levels]),
                ("exhaustive_grid", oracle_row, exhaustive_probe_count, float(estimated_hsf), [float(level) for level in observed_levels]),
            ]
            oracle_gain = float(oracle_row["mean_fastslow_gain_pct"])
            positive_available = bool(oracle_gain > 0.0)
            for strategy_name, picked_row, num_probes, strategy_signal, probed_levels in strategy_rows:
                realized_gain = float(picked_row["mean_fastslow_gain_pct"]) if picked_row is not None else float("nan")
                picked_level = float(picked_row["slow_to_fast_coupling"]) if picked_row is not None else float("nan")
                decision_rows.append(
                    {
                        "mechanism_system": system_name,
                        "seed": int(group_df["seed"].iloc[0]),
                        "task": str(group_df["task"].iloc[0]),
                        "decision_group": json.dumps(
                            {col: _json_safe(group_df.iloc[0][col]) for col in group_cols},
                            ensure_ascii=False,
                        ),
                        "strategy": strategy_name,
                        "oracle_best_coupling": float(oracle_row["slow_to_fast_coupling"]),
                        "oracle_best_gain_pct": oracle_gain,
                        "picked_coupling": picked_level,
                        "realized_gain_pct": realized_gain,
                        "regret_pct": oracle_gain - realized_gain if np.isfinite(realized_gain) else float("nan"),
                        "num_probes": int(max(num_probes, 0)),
                        "positive_gain_found": bool(np.isfinite(realized_gain) and realized_gain > 0.0),
                        "missed_positive_gain": bool(positive_available and (not np.isfinite(realized_gain) or realized_gain <= 0.0)),
                        "strategy_signal": strategy_signal,
                        "probed_levels": json.dumps(probed_levels, ensure_ascii=False),
                    }
                )
    decision_df = pd.DataFrame(decision_rows)
    if not decision_df.empty:
        strategy_rank = {name: idx for idx, name in enumerate(GATE_STRATEGY_ORDER)}
        grouped = decision_df.groupby(["mechanism_system", "strategy"], dropna=False)
        summary_df = grouped.agg(
            mean_realized_gain_pct=("realized_gain_pct", "mean"),
            std_realized_gain_pct=("realized_gain_pct", "std"),
            mean_regret_pct=("regret_pct", "mean"),
            positive_gain_hit_rate=("positive_gain_found", "mean"),
            missed_positive_rate=("missed_positive_gain", "mean"),
            mean_num_probes=("num_probes", "mean"),
            num_tasks=("task", "count"),
        ).reset_index()
        summary_df["strategy_order"] = summary_df["strategy"].map(strategy_rank).fillna(len(strategy_rank))
        summary_df = summary_df.sort_values(["mechanism_system", "strategy_order", "strategy"]).drop(columns=["strategy_order"]).reset_index(drop=True)
    else:
        summary_df = pd.DataFrame()
    return decision_df, summary_df


def _threshold_candidates(scores: np.ndarray) -> list[float]:
    finite = np.asarray(scores, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return [float("inf")]
    unique = np.unique(np.sort(finite))
    candidates = [float("inf"), float(unique[0] - 1.0e-6)]
    if unique.size == 1:
        candidates.append(float(unique[0]))
        return candidates
    mids = 0.5 * (unique[:-1] + unique[1:])
    candidates.extend(float(mid) for mid in mids)
    candidates.append(float(unique[-1] + 1.0e-6))
    return candidates


def _best_utility_threshold(scores: np.ndarray, gains: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    gains = np.asarray(gains, dtype=float)
    if scores.size == 0 or gains.size == 0:
        return float("inf")
    best_tau = float("inf")
    best_utility = float("-inf")
    for tau in _threshold_candidates(scores):
        use = np.isfinite(scores) & (scores >= tau)
        realized = np.where(use, gains, 0.0)
        utility = float(np.nanmean(realized)) if realized.size else float("-inf")
        if utility > best_utility + 1.0e-12:
            best_utility = utility
            best_tau = float(tau)
    return best_tau


def _append_row_gate_decisions(
    *,
    rows: list[dict[str, object]],
    strategy: str,
    system_name: str,
    test_df: pd.DataFrame,
    use_mask: np.ndarray,
    score: np.ndarray,
    threshold: float,
    estimated_coupling: np.ndarray | None = None,
) -> None:
    gains = test_df["mean_fastslow_gain_pct"].to_numpy(dtype=float)
    oracle = np.maximum(gains, 0.0)
    predicted_use = np.asarray(use_mask, dtype=bool)
    realized = np.where(predicted_use, gains, 0.0)
    est_coupling = (
        np.asarray(estimated_coupling, dtype=float)
        if estimated_coupling is not None
        else np.full(len(test_df), np.nan, dtype=float)
    )
    score_arr = np.asarray(score, dtype=float)
    for idx, (_, row) in enumerate(test_df.iterrows()):
        gain = float(gains[idx])
        rows.append(
            {
                "mechanism_system": system_name,
                "seed": int(row["seed"]),
                "task": str(row["task"]),
                "strategy": strategy,
                "score": float(score_arr[idx]) if idx < len(score_arr) and np.isfinite(score_arr[idx]) else float("nan"),
                "threshold": float(threshold) if np.isfinite(threshold) else float("nan"),
                "predicted_use_fastslow": bool(predicted_use[idx]),
                "true_positive_gain": bool(gain > 0.0),
                "positive_realized_gain": bool(realized[idx] > 0.0),
                "realized_gain_pct": float(realized[idx]),
                "oracle_gain_pct": float(oracle[idx]),
                "regret_pct": float(oracle[idx] - realized[idx]),
                "missed_positive_gain": bool((gain > 0.0) and (not predicted_use[idx])),
                "false_positive_use": bool((gain <= 0.0) and predicted_use[idx]),
                "estimated_coupling": float(est_coupling[idx]) if np.isfinite(est_coupling[idx]) else float("nan"),
                "true_coupling": float(row["slow_to_fast_coupling"]) if "slow_to_fast_coupling" in row else float("nan"),
            }
        )


def _fit_predict_on_split(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    poly, model, mean, std = _fit_poly_ridge(
        train_df,
        feature_columns=feature_columns,
        target_column=target_column,
    )
    train_pred = _predict_poly_ridge(
        train_df,
        feature_columns=feature_columns,
        poly=poly,
        model=model,
        mean=mean,
        std=std,
    )
    test_pred = _predict_poly_ridge(
        test_df,
        feature_columns=feature_columns,
        poly=poly,
        model=model,
        mean=mean,
        std=std,
    )
    return train_pred, test_pred


def _evaluate_beta_gamma_two_stage(
    estimator_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required_cols = {"mechanism_system", "seed", "mean_fastslow_gain_pct", "slow_to_fast_coupling"}
    if estimator_df.empty or not required_cols.issubset(estimator_df.columns):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    decision_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    beta_feature_columns = [
        col
        for col in (
            "beta_gamma_log_beta",
            "beta_gamma_log_gamma",
            "beta_gamma_log_viability",
            "beta_gamma_log_strength",
        )
        if col in estimator_df.columns
    ]
    raw_feature_columns = [col for col in RAW_FEATURE_COLUMNS if col in estimator_df.columns]

    for system_name, system_df in estimator_df.groupby("mechanism_system"):
        system_df = system_df.reset_index(drop=True).copy()
        observed_levels = sorted(system_df["slow_to_fast_coupling"].dropna().astype(float).unique())
        for seed_value in sorted(system_df["seed"].dropna().unique()):
            test_mask = system_df["seed"] == seed_value
            train_df = system_df[~test_mask].reset_index(drop=True)
            test_df = system_df[test_mask].reset_index(drop=True)
            if train_df.empty or test_df.empty:
                continue

            gains_train = train_df["mean_fastslow_gain_pct"].to_numpy(dtype=float)
            beta_viab_train = train_df["beta_gamma_viability_score"].to_numpy(dtype=float) if "beta_gamma_viability_score" in train_df.columns else np.full(len(train_df), np.nan, dtype=float)
            beta_viab_test = test_df["beta_gamma_viability_score"].to_numpy(dtype=float) if "beta_gamma_viability_score" in test_df.columns else np.full(len(test_df), np.nan, dtype=float)
            beta_viab_tau = _best_utility_threshold(beta_viab_train, gains_train)
            beta_viab_use = np.isfinite(beta_viab_test) & (beta_viab_test >= beta_viab_tau)

            beta_scores_train = np.full(len(train_df), np.nan, dtype=float)
            beta_scores_test = np.full(len(test_df), np.nan, dtype=float)
            beta_tau = float("inf")
            if beta_feature_columns:
                try:
                    beta_scores_train, beta_scores_test = _fit_predict_on_split(
                        train_df=train_df,
                        test_df=test_df,
                        feature_columns=beta_feature_columns,
                        target_column="mean_fastslow_gain_pct",
                    )
                    beta_tau = _best_utility_threshold(beta_scores_train, gains_train)
                except ValueError:
                    beta_scores_train = np.full(len(train_df), np.nan, dtype=float)
                    beta_scores_test = np.full(len(test_df), np.nan, dtype=float)
                    beta_tau = float("inf")
            beta_use = np.isfinite(beta_scores_test) & (beta_scores_test >= beta_tau)

            eff_scores_train = np.full(len(train_df), np.nan, dtype=float)
            eff_scores_test = np.full(len(test_df), np.nan, dtype=float)
            eff_tau = float("inf")
            if raw_feature_columns:
                try:
                    eff_scores_train, eff_scores_test = _fit_predict_on_split(
                        train_df=train_df,
                        test_df=test_df,
                        feature_columns=raw_feature_columns,
                        target_column="mean_fastslow_gain_pct",
                    )
                    eff_tau = _best_utility_threshold(eff_scores_train, gains_train)
                except ValueError:
                    eff_scores_train = np.full(len(train_df), np.nan, dtype=float)
                    eff_scores_test = np.full(len(test_df), np.nan, dtype=float)
                    eff_tau = float("inf")
            eff_use = np.isfinite(eff_scores_test) & (eff_scores_test >= eff_tau)

            beta_coupling_pred = np.full(len(test_df), np.nan, dtype=float)
            if beta_feature_columns:
                positive_train = train_df[train_df["mean_fastslow_gain_pct"] > 0.0].reset_index(drop=True)
                positive_test = test_df[test_df["mean_fastslow_gain_pct"] > 0.0].reset_index(drop=True)
                if len(positive_train) >= max(2, len(beta_feature_columns)) and not positive_test.empty:
                    try:
                        _, coupling_pred = _fit_predict_on_split(
                            train_df=positive_train,
                            test_df=positive_test,
                            feature_columns=beta_feature_columns,
                            target_column="slow_to_fast_coupling",
                        )
                        coupling_pred = np.asarray(
                            [_nearest_observed_level(pred, observed_levels) for pred in coupling_pred],
                            dtype=float,
                        )
                        pos_indices = np.flatnonzero(test_df["mean_fastslow_gain_pct"].to_numpy(dtype=float) > 0.0)
                        beta_coupling_pred[pos_indices] = coupling_pred
                        truth = positive_test["slow_to_fast_coupling"].to_numpy(dtype=float)
                        mask = np.isfinite(coupling_pred) & np.isfinite(truth)
                        calibration_rows.append(
                            {
                                "mechanism_system": system_name,
                                "seed": int(seed_value),
                                "method": "beta_gamma_strength",
                                "num_rows": int(mask.sum()),
                                "coupling_rmse": float(np.sqrt(np.mean(np.square(coupling_pred[mask] - truth[mask])))) if mask.any() else float("nan"),
                                "coupling_corr": (
                                    float(np.corrcoef(coupling_pred[mask], truth[mask])[0, 1])
                                    if mask.sum() > 1 and np.std(coupling_pred[mask]) > 1.0e-12 and np.std(truth[mask]) > 1.0e-12
                                    else float("nan")
                                ),
                                "nearest_level_acc": float(np.mean(np.isclose(coupling_pred[mask], truth[mask], atol=1.0e-9))) if mask.any() else float("nan"),
                            }
                        )
                    except ValueError:
                        pass
            if raw_feature_columns:
                positive_train = train_df[train_df["mean_fastslow_gain_pct"] > 0.0].reset_index(drop=True)
                positive_test = test_df[test_df["mean_fastslow_gain_pct"] > 0.0].reset_index(drop=True)
                if len(positive_train) >= 2 and not positive_test.empty:
                    try:
                        _, eff_coupling_pred = _fit_predict_on_split(
                            train_df=positive_train,
                            test_df=positive_test,
                            feature_columns=raw_feature_columns,
                            target_column="slow_to_fast_coupling",
                        )
                        eff_coupling_pred = np.asarray(
                            [_nearest_observed_level(pred, observed_levels) for pred in eff_coupling_pred],
                            dtype=float,
                        )
                        truth = positive_test["slow_to_fast_coupling"].to_numpy(dtype=float)
                        mask = np.isfinite(eff_coupling_pred) & np.isfinite(truth)
                        calibration_rows.append(
                            {
                                "mechanism_system": system_name,
                                "seed": int(seed_value),
                                "method": "effective_hsf_raw_features",
                                "num_rows": int(mask.sum()),
                                "coupling_rmse": float(np.sqrt(np.mean(np.square(eff_coupling_pred[mask] - truth[mask])))) if mask.any() else float("nan"),
                                "coupling_corr": (
                                    float(np.corrcoef(eff_coupling_pred[mask], truth[mask])[0, 1])
                                    if mask.sum() > 1 and np.std(eff_coupling_pred[mask]) > 1.0e-12 and np.std(truth[mask]) > 1.0e-12
                                    else float("nan")
                                ),
                                "nearest_level_acc": float(np.mean(np.isclose(eff_coupling_pred[mask], truth[mask], atol=1.0e-9))) if mask.any() else float("nan"),
                            }
                        )
                    except ValueError:
                        pass

            gains_test = test_df["mean_fastslow_gain_pct"].to_numpy(dtype=float)
            _append_row_gate_decisions(
                rows=decision_rows,
                strategy="oracle_viable_only",
                system_name=system_name,
                test_df=test_df,
                use_mask=gains_test > 0.0,
                score=gains_test,
                threshold=0.0,
            )
            _append_row_gate_decisions(
                rows=decision_rows,
                strategy="beta_gamma_viability_only",
                system_name=system_name,
                test_df=test_df,
                use_mask=beta_viab_use,
                score=beta_viab_test,
                threshold=beta_viab_tau,
            )
            _append_row_gate_decisions(
                rows=decision_rows,
                strategy="beta_gamma_two_stage",
                system_name=system_name,
                test_df=test_df,
                use_mask=beta_use,
                score=beta_scores_test,
                threshold=beta_tau,
                estimated_coupling=beta_coupling_pred,
            )
            _append_row_gate_decisions(
                rows=decision_rows,
                strategy="effective_gain_two_stage",
                system_name=system_name,
                test_df=test_df,
                use_mask=eff_use,
                score=eff_scores_test,
                threshold=eff_tau,
            )
            _append_row_gate_decisions(
                rows=decision_rows,
                strategy="always_use_fastslow",
                system_name=system_name,
                test_df=test_df,
                use_mask=np.ones(len(test_df), dtype=bool),
                score=np.ones(len(test_df), dtype=float),
                threshold=1.0,
            )
            _append_row_gate_decisions(
                rows=decision_rows,
                strategy="never_use_fastslow",
                system_name=system_name,
                test_df=test_df,
                use_mask=np.zeros(len(test_df), dtype=bool),
                score=np.zeros(len(test_df), dtype=float),
                threshold=float("inf"),
            )

    decision_df = pd.DataFrame(decision_rows)
    calibration_df = pd.DataFrame(calibration_rows)
    if decision_df.empty:
        return decision_df, pd.DataFrame(), calibration_df

    strategy_rank = {name: idx for idx, name in enumerate(ROW_GATE_STRATEGY_ORDER)}
    summary_df = (
        decision_df.groupby(["mechanism_system", "strategy"], dropna=False)
        .agg(
            mean_realized_gain_pct=("realized_gain_pct", "mean"),
            std_realized_gain_pct=("realized_gain_pct", "std"),
            mean_oracle_gain_pct=("oracle_gain_pct", "mean"),
            mean_regret_pct=("regret_pct", "mean"),
            use_rate=("predicted_use_fastslow", "mean"),
            positive_gain_hit_rate=("positive_realized_gain", "mean"),
            missed_positive_rate=("missed_positive_gain", "mean"),
            false_positive_rate=("false_positive_use", "mean"),
            num_tasks=("task", "count"),
        )
        .reset_index()
    )
    summary_df["strategy_order"] = summary_df["strategy"].map(strategy_rank).fillna(len(strategy_rank))
    summary_df = summary_df.sort_values(["mechanism_system", "strategy_order", "strategy"]).drop(columns=["strategy_order"]).reset_index(drop=True)

    if not calibration_df.empty:
        calibration_summary = (
            calibration_df.groupby(["mechanism_system", "method"], dropna=False)
            .agg(
                mean_coupling_rmse=("coupling_rmse", "mean"),
                mean_coupling_corr=("coupling_corr", "mean"),
                mean_nearest_level_acc=("nearest_level_acc", "mean"),
                num_holds=("seed", "count"),
                num_rows=("num_rows", "sum"),
            )
            .reset_index()
            .sort_values(["mechanism_system", "method"])
            .reset_index(drop=True)
        )
    else:
        calibration_summary = pd.DataFrame()

    return decision_df, summary_df, calibration_summary


def _format_pct(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.2f}%"


def _json_safe(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        cast = float(value)
        return cast if np.isfinite(cast) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _trend_label(values: Sequence[float]) -> str:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return "insufficient data"
    diffs = np.diff(arr)
    if np.all(diffs >= -1.0e-6):
        return "non-decreasing"
    if np.all(diffs <= 1.0e-6):
        return "non-increasing"
    return "non-monotone"


def _render_study_report(
    *,
    suite: str,
    seeds: Sequence[int],
    system_summary: pd.DataFrame,
    estimator_df: pd.DataFrame,
    estimator_loso_df: pd.DataFrame,
    gate_summary_df: pd.DataFrame,
    beta_gamma_summary_df: pd.DataFrame,
    beta_gamma_calibration_df: pd.DataFrame,
) -> str:
    def cols(df: pd.DataFrame, names: Sequence[str]) -> list[str]:
        return [name for name in names if name in df.columns]

    lines = [
        "# Fast-Slow Mechanism Study",
        "",
        f"- suite: {suite}",
        f"- seeds: {', '.join(str(seed) for seed in seeds)}",
        f"- systems covered: {system_summary['mechanism_system'].nunique() if 'mechanism_system' in system_summary else 0}",
        "",
        "## Cross-Seed System Summary",
        "",
    ]
    if not system_summary.empty:
        display = system_summary.copy()
        for col in (
            "mean_fastslow_gain_pct_mean",
            "mean_fastslow_gain_pct_std",
            "rc_fastslow_gain_pct_mean",
            "ngrc_fastslow_gain_pct_mean",
        ):
            if col in display.columns:
                display[col] = display[col].map(_format_pct)
        lines.append(
            display[
                cols(
                    display,
                    [
                    "mechanism_system",
                    "slow_to_fast_coupling",
                    "num_seeds",
                    "mean_fastslow_gain_pct_mean",
                    "mean_fastslow_gain_pct_std",
                    "rc_fastslow_gain_pct_mean",
                    "ngrc_fastslow_gain_pct_mean",
                    "fastslow_markov_gain_ratio_mean",
                    "fastslow_koopman_score_mean",
                    ],
                )
            ].to_markdown(index=False)
        )
        lines.extend(["", "### Trend Notes", ""])
        group_key = "mechanism_system" if "mechanism_system" in system_summary.columns else "system"
        for system_name, group_df in system_summary.groupby(group_key):
            ordered = group_df.sort_values("slow_to_fast_coupling")
            trend = _trend_label(ordered["mean_fastslow_gain_pct_mean"])
            best_idx = ordered["mean_fastslow_gain_pct_mean"].astype(float).idxmax()
            best_row = ordered.loc[best_idx]
            lines.append(
                f"- `{system_name}`: mean lift is {trend}; best observed coupling is {best_row['slow_to_fast_coupling']:.2f} "
                f"with cross-seed mean {_format_pct(float(best_row['mean_fastslow_gain_pct_mean']))}."
            )
    else:
        lines.append("No cross-seed summaries were generated.")
    lines.extend(["", "## Effective h_sf Estimator", ""])
    if not estimator_df.empty:
        display = estimator_df.copy()
        if "predicted_mean_fastslow_gain_pct" in display.columns:
            display["predicted_mean_fastslow_gain_pct"] = display["predicted_mean_fastslow_gain_pct"].map(_format_pct)
        lines.append(
            display[
                cols(
                    display,
                    [
                    "mechanism_system",
                    "task",
                    "seed",
                    "slow_to_fast_coupling",
                    "estimated_effective_hsf",
                    "predicted_mean_fastslow_gain_pct",
                    "recommended_gate_policy",
                    "suggested_probe_grid",
                    ],
                )
            ].to_markdown(index=False)
        )
    else:
        lines.append("No estimator rows were generated.")
    lines.extend(["", "### Leave-One-System-Out", ""])
    if not estimator_loso_df.empty:
        lines.append(estimator_loso_df.to_markdown(index=False))
    else:
        lines.append("Leave-one-system-out evaluation was not available.")
    lines.extend(["", "## Gate Design Evaluation", ""])
    if not gate_summary_df.empty:
        display = gate_summary_df.copy()
        for col in ("mean_realized_gain_pct", "std_realized_gain_pct", "mean_regret_pct"):
            if col in display.columns:
                display[col] = display[col].map(_format_pct)
        if "positive_gain_hit_rate" in display.columns:
            display["positive_gain_hit_rate"] = display["positive_gain_hit_rate"].map(lambda v: "nan" if not np.isfinite(v) else f"{100.0 * v:.1f}%")
        if "missed_positive_rate" in display.columns:
            display["missed_positive_rate"] = display["missed_positive_rate"].map(lambda v: "nan" if not np.isfinite(v) else f"{100.0 * v:.1f}%")
        lines.append(display.to_markdown(index=False))
        lines.extend(["", "### Gate Readout", ""])
        for system_name, group_df in gate_summary_df.groupby("mechanism_system"):
            oracle_gain = group_df.loc[group_df["strategy"] == "oracle_single_probe", "mean_realized_gain_pct"]
            est_gain = group_df.loc[group_df["strategy"] == "estimated_single_probe", "mean_realized_gain_pct"]
            est_grid_gain = group_df.loc[group_df["strategy"] == "estimated_probe_grid", "mean_realized_gain_pct"]
            if oracle_gain.empty or est_gain.empty:
                continue
            lines.append(
                f"- `{system_name}`: estimated single-probe realized {_format_pct(float(est_gain.iloc[0]))}"
                f" vs oracle {_format_pct(float(oracle_gain.iloc[0]))}"
                + (
                    f"; 3-probe grid realizes {_format_pct(float(est_grid_gain.iloc[0]))}."
                    if not est_grid_gain.empty
                    else "."
                )
            )
    else:
        lines.append("Gate-design evaluation was not available.")
    lines.extend(["", "## Beta/Gamma Two-Stage Gate", ""])
    if not beta_gamma_summary_df.empty:
        display = beta_gamma_summary_df.copy()
        for col in ("mean_realized_gain_pct", "std_realized_gain_pct", "mean_oracle_gain_pct", "mean_regret_pct"):
            if col in display.columns:
                display[col] = display[col].map(_format_pct)
        for col in ("use_rate", "positive_gain_hit_rate", "missed_positive_rate", "false_positive_rate"):
            if col in display.columns:
                display[col] = display[col].map(lambda v: "nan" if not np.isfinite(v) else f"{100.0 * v:.1f}%")
        lines.append(display.to_markdown(index=False))
    else:
        lines.append("Beta/gamma gate evaluation was not available.")
    lines.extend(["", "### Coupling Calibration", ""])
    if not beta_gamma_calibration_df.empty:
        display = beta_gamma_calibration_df.copy()
        if "mean_nearest_level_acc" in display.columns:
            display["mean_nearest_level_acc"] = display["mean_nearest_level_acc"].map(lambda v: "nan" if not np.isfinite(v) else f"{100.0 * v:.1f}%")
        lines.append(display.to_markdown(index=False))
    else:
        lines.append("No coupling-calibration summary was available.")
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "The estimator should be treated as an effective-coupling proxy learned from this synthetic family, not as a proof that scalar observations uniquely identify the true physical coupling parameter in arbitrary systems.",
        ]
    )
    return "\n".join(lines)


def _strip_beta_gamma_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = []
    for col in df.columns:
        if col.startswith("beta_gamma_"):
            drop_cols.append(col)
        if col in {
            "rc_beta",
            "rc_gamma",
            "rc_viability_score",
            "rc_strength_proxy",
            "rc_beta_gamma_encoder",
            "ngrc_beta",
            "ngrc_gamma",
            "ngrc_viability_score",
            "ngrc_strength_proxy",
            "ngrc_beta_gamma_encoder",
        }:
            drop_cols.append(col)
    if not drop_cols:
        return df.copy()
    return df.drop(columns=sorted(set(drop_cols))).copy()


def _postprocess_fastslow_mechanism_study(
    *,
    suite: str,
    out_path: Path,
    seeds: Sequence[int],
    seed_df: pd.DataFrame,
    task_names: Sequence[str] | None = None,
) -> dict[str, object]:
    manifest: dict[str, str] = {}
    clean_seed_df = _strip_beta_gamma_columns(seed_df)
    seed_df_with_stats, beta_gamma_stats_df = _attach_beta_gamma_stats(
        clean_seed_df,
        suite=suite,
        out_path=out_path,
        seeds=seeds,
        task_names=task_names,
    )
    seed_summary_path = out_path / "mechanism_seed_summary.csv"
    seed_df_with_stats.to_csv(seed_summary_path, index=False)
    manifest["mechanism_seed_summary"] = str(seed_summary_path)
    beta_gamma_stats_path = out_path / "beta_gamma_row_stats.csv"
    beta_gamma_stats_df.to_csv(beta_gamma_stats_path, index=False)
    manifest["beta_gamma_row_stats"] = str(beta_gamma_stats_path)

    agg_cols = {col: ["mean", "std"] for col in MECHANISM_METRIC_COLUMNS if col in seed_df_with_stats.columns}
    group_cols = [
        col
        for col in ("mechanism_system", "slow_to_fast_coupling", "task", "system", "sweep_label")
        if col in seed_df_with_stats.columns
    ]
    system_summary = pd.DataFrame()
    if not seed_df_with_stats.empty and group_cols:
        system_summary = seed_df_with_stats.groupby(group_cols, dropna=False).agg(agg_cols)
        system_summary.columns = ["_".join(part for part in col if part) for col in system_summary.columns.to_flat_index()]
        system_summary = system_summary.reset_index()
        if "seed" in seed_df_with_stats.columns:
            counts = seed_df_with_stats.groupby(group_cols, dropna=False)["seed"].nunique().reset_index(name="num_seeds")
            system_summary = system_summary.merge(counts, on=group_cols, how="left")
        system_summary = system_summary.sort_values(
            [col for col in ("mechanism_system", "slow_to_fast_coupling", "task") if col in system_summary.columns]
        ).reset_index(drop=True)
    system_summary_path = out_path / "mechanism_crossseed_summary.csv"
    system_summary.to_csv(system_summary_path, index=False)
    manifest["mechanism_crossseed_summary"] = str(system_summary_path)

    estimator_df = pd.DataFrame()
    estimator_loso_df = pd.DataFrame()
    gate_decision_df = pd.DataFrame()
    gate_summary_df = pd.DataFrame()
    beta_gamma_decision_df = pd.DataFrame()
    beta_gamma_summary_df = pd.DataFrame()
    beta_gamma_calibration_df = pd.DataFrame()
    if not seed_df_with_stats.empty and "slow_to_fast_coupling" in seed_df_with_stats.columns:
        estimator_df = seed_df_with_stats.copy()
        feature_columns = [col for col in RAW_FEATURE_COLUMNS if col in estimator_df.columns]
        observed_levels = sorted(estimator_df["slow_to_fast_coupling"].dropna().astype(float).unique())
        if feature_columns:
            try:
                h_poly, h_model, h_mean, h_std = _fit_poly_ridge(
                    estimator_df,
                    feature_columns=feature_columns,
                    target_column="slow_to_fast_coupling",
                )
                estimator_df["estimated_effective_hsf"] = _predict_poly_ridge(
                    estimator_df,
                    feature_columns=feature_columns,
                    poly=h_poly,
                    model=h_model,
                    mean=h_mean,
                    std=h_std,
                )
            except ValueError:
                estimator_df["estimated_effective_hsf"] = float("nan")
            if "mean_fastslow_gain_pct" in estimator_df.columns:
                try:
                    gain_poly, gain_model, gain_mean, gain_std = _fit_poly_ridge(
                        estimator_df,
                        feature_columns=feature_columns,
                        target_column="mean_fastslow_gain_pct",
                    )
                    estimator_df["predicted_mean_fastslow_gain_pct"] = _predict_poly_ridge(
                        estimator_df,
                        feature_columns=feature_columns,
                        poly=gain_poly,
                        model=gain_model,
                        mean=gain_mean,
                        std=gain_std,
                    )
                except ValueError:
                    estimator_df["predicted_mean_fastslow_gain_pct"] = float("nan")
            else:
                estimator_df["predicted_mean_fastslow_gain_pct"] = float("nan")
            estimator_df["suggested_probe_grid"] = estimator_df["estimated_effective_hsf"].map(
                lambda value: json.dumps(_suggest_probe_grid(float(value), observed_levels), ensure_ascii=False)
            )
            estimator_df["recommended_gate_policy"] = estimator_df.apply(
                lambda row: _gate_policy(
                    _finite_or_nan(row.get("estimated_effective_hsf")),
                    _finite_or_nan(row.get("predicted_mean_fastslow_gain_pct")),
                ),
                axis=1,
            )
            if "mechanism_system" in estimator_df.columns:
                h_pred_loso, h_rows = _fit_leave_one_group_out(
                    estimator_df,
                    feature_columns=feature_columns,
                    target_column="slow_to_fast_coupling",
                    group_column="mechanism_system",
                )
                estimator_df["loso_estimated_effective_hsf"] = h_pred_loso
                loso_rows = h_rows
                if "mean_fastslow_gain_pct" in estimator_df.columns:
                    gain_pred_loso, gain_rows = _fit_leave_one_group_out(
                        estimator_df,
                        feature_columns=feature_columns,
                        target_column="mean_fastslow_gain_pct",
                        group_column="mechanism_system",
                    )
                    estimator_df["loso_predicted_mean_fastslow_gain_pct"] = gain_pred_loso
                    gain_map = {row["mechanism_system"]: row for row in gain_rows}
                    for row in loso_rows:
                        gain_row = gain_map.get(row["mechanism_system"])
                        if gain_row is not None:
                            row.update(gain_row)
                estimator_loso_df = pd.DataFrame(loso_rows)
            gate_decision_df, gate_summary_df = _evaluate_gate_strategies(estimator_df)
            beta_gamma_decision_df, beta_gamma_summary_df, beta_gamma_calibration_df = _evaluate_beta_gamma_two_stage(estimator_df)

    estimator_path = out_path / "effective_hsf_estimates.csv"
    estimator_df.to_csv(estimator_path, index=False)
    manifest["effective_hsf_estimates"] = str(estimator_path)
    estimator_loso_path = out_path / "effective_hsf_loso.csv"
    estimator_loso_df.to_csv(estimator_loso_path, index=False)
    manifest["effective_hsf_loso"] = str(estimator_loso_path)
    gate_decision_path = out_path / "effective_hsf_gate_decisions.csv"
    gate_decision_df.to_csv(gate_decision_path, index=False)
    manifest["effective_hsf_gate_decisions"] = str(gate_decision_path)
    gate_summary_path = out_path / "effective_hsf_gate_summary.csv"
    gate_summary_df.to_csv(gate_summary_path, index=False)
    manifest["effective_hsf_gate_summary"] = str(gate_summary_path)
    beta_gamma_decision_path = out_path / "beta_gamma_gate_decisions.csv"
    beta_gamma_decision_df.to_csv(beta_gamma_decision_path, index=False)
    manifest["beta_gamma_gate_decisions"] = str(beta_gamma_decision_path)
    beta_gamma_summary_path = out_path / "beta_gamma_gate_summary.csv"
    beta_gamma_summary_df.to_csv(beta_gamma_summary_path, index=False)
    manifest["beta_gamma_gate_summary"] = str(beta_gamma_summary_path)
    beta_gamma_calibration_path = out_path / "beta_gamma_coupling_calibration.csv"
    beta_gamma_calibration_df.to_csv(beta_gamma_calibration_path, index=False)
    manifest["beta_gamma_coupling_calibration"] = str(beta_gamma_calibration_path)

    report_path = out_path / "fastslow_mechanism_study_report.md"
    report_path.write_text(
        _render_study_report(
            suite=suite,
            seeds=seeds,
            system_summary=system_summary,
            estimator_df=estimator_df,
            estimator_loso_df=estimator_loso_df,
            gate_summary_df=gate_summary_df,
            beta_gamma_summary_df=beta_gamma_summary_df,
            beta_gamma_calibration_df=beta_gamma_calibration_df,
        ),
        encoding="utf-8",
    )
    manifest["fastslow_mechanism_study_report"] = str(report_path)
    (out_path / "mechanism_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "mechanism_seed_summary": seed_df_with_stats,
        "mechanism_crossseed_summary": system_summary,
        "beta_gamma_row_stats": beta_gamma_stats_df,
        "effective_hsf_estimates": estimator_df,
        "effective_hsf_loso": estimator_loso_df,
        "effective_hsf_gate_decisions": gate_decision_df,
        "effective_hsf_gate_summary": gate_summary_df,
        "beta_gamma_gate_decisions": beta_gamma_decision_df,
        "beta_gamma_gate_summary": beta_gamma_summary_df,
        "beta_gamma_coupling_calibration": beta_gamma_calibration_df,
        "manifest": manifest,
    }


def reanalyze_fastslow_mechanism_study(
    *,
    suite: str = "fastslow_crosssystem_gating_smoke",
    out_dir: str = "runs/fastslow_mechanism_study/fastslow_crosssystem_gating_smoke",
    seeds: Sequence[int] = (101, 202, 303),
    task_names: Sequence[str] | None = None,
) -> dict[str, object]:
    out_path = Path(out_dir)
    seed_summary_path = out_path / "mechanism_seed_summary.csv"
    if not seed_summary_path.exists():
        raise FileNotFoundError(f"Mechanism seed summary not found: {seed_summary_path}")
    seed_df = pd.read_csv(seed_summary_path)
    return _postprocess_fastslow_mechanism_study(
        suite=suite,
        out_path=out_path,
        seeds=seeds,
        seed_df=seed_df,
        task_names=task_names,
    )


def run_fastslow_mechanism_study(
    *,
    suite: str = "fastslow_crosssystem_gating_smoke",
    out_dir: str = "runs/fastslow_mechanism_study/fastslow_crosssystem_gating_smoke",
    seeds: Sequence[int] = (101, 202, 303),
    task_names: Sequence[str] | None = None,
    model_groups: Sequence[str] | None = ("fastslow_ablation",),
    grid_mode: str = "quick",
    coordinate_kinds: Sequence[str] = ("raw", "delay", "fastslow", "theory_fastslow", "factor"),
    delay_dim: int = 8,
    mining_mode: str = "accumulate",
    full_library_search: bool = False,
    factor_config_path: str | None = "configs/fastslow_theory_factor_mining.yaml",
    identifier_kinds: Sequence[str] | None = None,
    skip_factor_mining: bool = True,
) -> dict[str, object]:
    out_path = Path(out_dir)
    ensure_dir(out_path)
    tasks = build_suite(suite)
    if task_names is not None:
        selected = set(task_names)
        tasks = [task for task in tasks if task.name in selected]
    seed_rows: list[pd.DataFrame] = []
    feature_rows: list[dict[str, object]] = []
    manifest: dict[str, str] = {}

    for seed in seeds:
        seed_dir = out_path / f"seed_{seed}"
        result = run_research_loop(
            suite=suite,
            out_dir=str(seed_dir),
            seed=int(seed),
            task_names=task_names,
            model_groups=model_groups,
            grid_mode=grid_mode,
            coordinate_kinds=coordinate_kinds,
            delay_dim=delay_dim,
            mining_mode=mining_mode,
            full_library_search=full_library_search,
            factor_config_path=factor_config_path,
            identifier_kinds=identifier_kinds,
            skip_factor_mining=skip_factor_mining,
        )
        benchmark_summary = summarize_fastslow_benchmarks(result["benchmarks"])
        coordinate_summary = summarize_fastslow_coordinates(result["coordinate_analysis"])
        mechanism_summary = summarize_fastslow_mechanism_sweeps(benchmark_summary, coordinate_summary)
        if not mechanism_summary.empty:
            mechanism_summary["seed"] = int(seed)
            seed_rows.append(mechanism_summary)
        for task in tasks:
            feature_rows.append(_task_observable_features(task, seed=int(seed)))
        manifest[f"seed_{seed}"] = str(seed_dir)

    seed_df = pd.concat(seed_rows, axis=0, ignore_index=True) if seed_rows else pd.DataFrame()
    feature_df = pd.DataFrame(feature_rows)
    if not seed_df.empty and not feature_df.empty:
        merge_keys = [
            key
            for key in ("task", "seed", "task_metadata", "sweep_suite", "sweep_group", "sweep_value")
            if key in seed_df.columns and key in feature_df.columns
        ]
        if not merge_keys:
            merge_keys = [key for key in ("task", "seed") if key in seed_df.columns and key in feature_df.columns]
        seed_df = seed_df.merge(
            feature_df,
            on=merge_keys,
            how="left",
            suffixes=("", "_feature"),
        )
    result = _postprocess_fastslow_mechanism_study(
        suite=suite,
        out_path=out_path,
        seeds=seeds,
        seed_df=seed_df,
        task_names=task_names,
    )
    for seed in seeds:
        result["manifest"][f"seed_{seed}"] = str(out_path / f"seed_{seed}")
    (out_path / "mechanism_manifest.json").write_text(
        json.dumps(result["manifest"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result
