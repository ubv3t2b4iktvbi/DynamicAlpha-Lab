from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..benchmarks import build_suite
from ..models import NGRCConfig, PureNGRCModel, PureRCModel, RCConfig, ReservoirTemplateFactory
from ..online_ftrl import FTRLConfig, OnlineFTRLBinary
from ..selection import instantiate_model, select_best_model
from ..systems import SYSTEMS, BenchmarkTask, observe, rk4_step
from ..utils import ensure_dir, to_jsonable


DEFAULT_TRAIN_SEEDS: tuple[int, ...] = (101, 211, 307, 401, 503, 601)
DEFAULT_TEST_SEEDS: tuple[int, ...] = (701, 809, 907)
DEFAULT_MODEL_NAMES: tuple[str, ...] = ("rc_rg_readout", "ngrc_rg_readout")


@dataclass(frozen=True)
class PiecewiseChangeConfig:
    suite: str = "fastslow_smoke"
    task_name: str = "vanderpol_relaxation_smoke"
    pre_params: dict[str, Any] = field(default_factory=lambda: {"mu": 8.0})
    post_params: dict[str, Any] = field(default_factory=lambda: {"mu": 14.0})
    fit_train_len: int = 700
    fit_val_len: int = 250
    change_index: int = 1250
    total_length: int = 2200
    burn_in: int | None = None
    process_noise_std: float = 0.005
    obs_noise_std: float = 0.01


@dataclass(frozen=True)
class DetectorTrainConfig:
    ftrl: FTRLConfig = field(default_factory=FTRLConfig)
    epochs: int = 3
    threshold_grid: tuple[float, ...] = tuple(np.linspace(0.35, 0.85, 21))
    min_consecutive: int = 3


@dataclass
class PiecewiseEpisode:
    seed: int
    obs: np.ndarray
    states: np.ndarray
    change_index: int


@dataclass
class PredictorSelection:
    model_name: str
    best_config: RCConfig | NGRCConfig | Any
    selection_metrics: dict[str, float]


def _lookup_task(suite: str, task_name: str) -> BenchmarkTask:
    for task in build_suite(suite):
        if task.name == task_name:
            return task
    raise ValueError(f"Unknown task_name={task_name} for suite={suite}")


def _resolve_initial_state(task: BenchmarkTask) -> np.ndarray:
    meta = SYSTEMS[task.system]
    default_x0 = meta["default_x0"]
    if task.x0 is not None:
        return np.asarray(task.x0, dtype=float).copy()
    if callable(default_x0):
        return np.asarray(default_x0(task.params), dtype=float).copy()
    return np.asarray(default_x0, dtype=float).copy()


def simulate_piecewise_episode(
    task: BenchmarkTask,
    cfg: PiecewiseChangeConfig,
    *,
    seed: int,
) -> PiecewiseEpisode:
    rng = np.random.default_rng(int(seed))
    meta = SYSTEMS[task.system]
    rhs = meta["rhs"]
    x = _resolve_initial_state(task)
    burn_in = int(task.burn_in if cfg.burn_in is None else cfg.burn_in)
    total_steps = int(cfg.total_length + burn_in)
    change_step = int(burn_in + cfg.change_index)
    states = np.zeros((total_steps, len(x)), dtype=float)
    t = 0.0
    pre_params = dict(task.params)
    pre_params.update(cfg.pre_params or {})
    post_params = dict(task.params)
    post_params.update(cfg.post_params or {})
    for step in range(total_steps):
        params = pre_params if step < change_step else post_params
        x_next = rk4_step(rhs, x, t, task.dt, params)
        if cfg.process_noise_std > 0.0:
            x_next = x_next + np.sqrt(task.dt) * float(cfg.process_noise_std) * rng.normal(size=x.shape)
        x = x_next
        states[step] = x_next
        t += task.dt
    states = states[burn_in:]
    obs = observe(states, task.system, task.obs_mode, params=post_params, obs_params=task.obs_params).astype(float)
    if cfg.obs_noise_std > 0.0:
        obs = obs + float(cfg.obs_noise_std) * rng.normal(size=obs.shape)
    return PiecewiseEpisode(seed=int(seed), obs=obs, states=states, change_index=int(cfg.change_index))


def _fit_model(
    model_name: str,
    cfg: Any,
    *,
    fit_series: np.ndarray,
    template_factory: ReservoirTemplateFactory,
) -> PureRCModel | PureNGRCModel:
    model = instantiate_model(
        model_name=model_name,
        cfg=cfg,
        template_factory=template_factory,
        short_train=len(fit_series) < 2000,
        model_context=None,
    ).fit(fit_series)
    if not isinstance(model, (PureRCModel, PureNGRCModel)):
        raise TypeError(f"Unsupported fitted model type for changepoint detector: {type(model)!r}")
    return model


def _select_predictor(
    task: BenchmarkTask,
    episode: PiecewiseEpisode,
    *,
    model_name: str,
    cfg: PiecewiseChangeConfig,
    grid_mode: str,
    template_factory: ReservoirTemplateFactory,
) -> PredictorSelection:
    fit_end = int(cfg.fit_train_len + cfg.fit_val_len)
    if fit_end >= cfg.change_index:
        raise ValueError("fit_train_len + fit_val_len must be strictly smaller than change_index")
    y_train = np.asarray(episode.obs[: cfg.fit_train_len], dtype=float)
    y_val = np.asarray(episode.obs[cfg.fit_train_len:fit_end], dtype=float)
    context_len = min(max(200, 4 * max(task.selection_horizons)), max(32, len(y_train) // 2))
    model, val_metrics, best_cfg = select_best_model(
        model_name=model_name,
        y_train=y_train,
        y_val=y_val,
        context_len=context_len,
        score_horizons=task.selection_horizons,
        grid_mode=grid_mode,
        template_factory=template_factory,
        short_train=len(y_train) < 2000,
        progress_desc=f"{task.name}-{model_name}-cp",
        data_dt=task.dt,
        model_context=None,
    )
    if model is None or best_cfg is None or val_metrics is None:
        raise RuntimeError(f"Model selection failed for {model_name}")
    return PredictorSelection(model_name=model_name, best_config=best_cfg, selection_metrics=val_metrics)


def _rc_trace(model: PureRCModel, series: np.ndarray, *, start_eval: int) -> dict[str, np.ndarray]:
    ys = ((np.asarray(series, dtype=float).reshape(-1) - model.mu_) / model.std_).astype(float)
    _, factor_mat = model.readout.transform(ys)
    r = np.zeros(model.cfg.n_reservoir, dtype=float)
    eval_indices: list[int] = []
    preds: list[float] = []
    truth: list[float] = []
    y_vals: list[float] = []
    dy_vals: list[float] = []
    factors: list[np.ndarray] = []
    begin = max(int(start_eval), int(model.cfg.washout))
    for t in range(len(ys) - 1):
        r = model._step(r, ys[t])
        if t < begin:
            continue
        pred_std = float(model._readout_aug(r, ys[t], factor_mat=factor_mat, t=t) @ model.Wout)
        eval_indices.append(int(t))
        preds.append(pred_std)
        truth.append(float(ys[t + 1]))
        y_vals.append(float(ys[t]))
        dy_vals.append(float(ys[t] - ys[t - 1]) if t > 0 else 0.0)
        if factor_mat is not None and factor_mat.shape[1] > 0:
            factors.append(np.asarray(factor_mat[t], dtype=float))
    factor_rows = np.vstack(factors) if factors else np.zeros((len(eval_indices), 0), dtype=float)
    truth_arr = np.asarray(truth, dtype=float)
    pred_arr = np.asarray(preds, dtype=float)
    return {
        "eval_indices": np.asarray(eval_indices, dtype=int),
        "pred_std": pred_arr,
        "truth_std": truth_arr,
        "residual_std": truth_arr - pred_arr,
        "y_std": np.asarray(y_vals, dtype=float),
        "dy_std": np.asarray(dy_vals, dtype=float),
        "factor_rows": factor_rows,
    }


def _ngrc_trace(model: PureNGRCModel, series: np.ndarray, *, start_eval: int) -> dict[str, np.ndarray]:
    ys = ((np.asarray(series, dtype=float).reshape(-1) - model.mu_) / model.std_).astype(float)
    _, factor_mat = model.readout.transform(ys)
    eval_indices: list[int] = []
    preds: list[float] = []
    truth: list[float] = []
    y_vals: list[float] = []
    dy_vals: list[float] = []
    factors: list[np.ndarray] = []
    begin = max(int(start_eval), int(model.cfg.washout), int(model.delay.max_lag))
    for t in range(begin, len(ys) - 1):
        delay_row = model.delay.row_from_series(ys, t)
        pred_std = float(model._feature_row(delay_row, factor_mat=factor_mat, t=t) @ model.coef_)
        eval_indices.append(int(t))
        preds.append(pred_std)
        truth.append(float(ys[t + 1]))
        y_vals.append(float(ys[t]))
        dy_vals.append(float(ys[t] - ys[t - 1]) if t > 0 else 0.0)
        if factor_mat is not None and factor_mat.shape[1] > 0:
            factors.append(np.asarray(factor_mat[t], dtype=float))
    factor_rows = np.vstack(factors) if factors else np.zeros((len(eval_indices), 0), dtype=float)
    truth_arr = np.asarray(truth, dtype=float)
    pred_arr = np.asarray(preds, dtype=float)
    return {
        "eval_indices": np.asarray(eval_indices, dtype=int),
        "pred_std": pred_arr,
        "truth_std": truth_arr,
        "residual_std": truth_arr - pred_arr,
        "y_std": np.asarray(y_vals, dtype=float),
        "dy_std": np.asarray(dy_vals, dtype=float),
        "factor_rows": factor_rows,
    }


def _predictor_trace(model: PureRCModel | PureNGRCModel, series: np.ndarray, *, start_eval: int) -> dict[str, np.ndarray]:
    if isinstance(model, PureRCModel):
        return _rc_trace(model, series, start_eval=start_eval)
    if isinstance(model, PureNGRCModel):
        return _ngrc_trace(model, series, start_eval=start_eval)
    raise TypeError(f"Unsupported model type: {type(model)!r}")


def _feature_block(trace: dict[str, np.ndarray], *, prefix: str) -> tuple[np.ndarray, list[str]]:
    residual = np.asarray(trace["residual_std"], dtype=float).reshape(-1, 1)
    y_std = np.asarray(trace["y_std"], dtype=float).reshape(-1, 1)
    dy_std = np.asarray(trace["dy_std"], dtype=float).reshape(-1, 1)
    factor_rows = np.asarray(trace["factor_rows"], dtype=float)
    cols = [
        residual,
        np.abs(residual),
        residual ** 2,
        y_std,
        dy_std,
    ]
    names = [
        f"{prefix}residual_std",
        f"{prefix}abs_residual_std",
        f"{prefix}sq_residual_std",
        f"{prefix}y_std",
        f"{prefix}dy_std",
    ]
    if factor_rows.size > 0:
        factor_delta = factor_rows - np.vstack([factor_rows[:1], factor_rows[:-1]])
        cols.extend([factor_rows, factor_delta])
        for idx in range(factor_rows.shape[1]):
            names.append(f"{prefix}factor_{idx}")
        for idx in range(factor_rows.shape[1]):
            names.append(f"{prefix}factor_delta_{idx}")
    X = np.hstack(cols) if cols else np.zeros((residual.shape[0], 0), dtype=float)
    return np.asarray(X, dtype=float), names


def _pooled_feature_variants(
    traces: dict[str, dict[str, np.ndarray]],
) -> dict[str, tuple[np.ndarray, list[str], np.ndarray]]:
    names = sorted(traces)
    if not names:
        return {}
    variants: dict[str, tuple[np.ndarray, list[str], np.ndarray]] = {}
    for model_name in names:
        X, feature_names = _feature_block(traces[model_name], prefix=f"{model_name}::")
        variants[model_name] = (X, feature_names, np.asarray(traces[model_name]["eval_indices"], dtype=int))
    if len(names) >= 2:
        common = np.asarray(traces[names[0]]["eval_indices"], dtype=int)
        for model_name in names[1:]:
            common = np.intersect1d(common, np.asarray(traces[model_name]["eval_indices"], dtype=int))
        blocks = []
        feature_names: list[str] = []
        for model_name in names:
            X, local_names = _feature_block(traces[model_name], prefix=f"{model_name}::")
            model_eval = np.asarray(traces[model_name]["eval_indices"], dtype=int)
            keep = np.isin(model_eval, common)
            blocks.append(X[keep])
            feature_names.extend(local_names)
        variants["joint__" + "__".join(names)] = (np.hstack(blocks), feature_names, common)
    return variants


def _standardize_features(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (np.asarray(X, dtype=float) - mu) / sigma


def _append_bias(X: np.ndarray) -> np.ndarray:
    X_use = np.asarray(X, dtype=float)
    return np.hstack([X_use, np.ones((X_use.shape[0], 1), dtype=float)])


def _binary_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    s = np.asarray(score, dtype=float).reshape(-1)
    pos = int(np.sum(y > 0.5))
    neg = int(np.sum(y <= 0.5))
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy(dtype=float)
    sum_pos = float(np.sum(ranks[y > 0.5]))
    auc = (sum_pos - pos * (pos + 1) / 2.0) / float(pos * neg)
    return float(auc)


def _fit_ftrl_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    cfg: DetectorTrainConfig,
) -> OnlineFTRLBinary:
    X_use = _append_bias(X)
    y_use = np.asarray(y, dtype=float).reshape(-1)
    model = OnlineFTRLBinary(n_features=X_use.shape[1], cfg=cfg.ftrl)
    pos = float(np.sum(y_use > 0.5))
    neg = float(np.sum(y_use <= 0.5))
    pos_weight = 1.0 if pos <= 0.0 else max(1.0, neg / max(pos, 1.0))
    for _ in range(max(1, int(cfg.epochs))):
        for row, label in zip(X_use, y_use):
            sample_weight = pos_weight if label > 0.5 else 1.0
            model.update(row, label, sample_weight=sample_weight)
    return model


def _predict_probabilities(model: OnlineFTRLBinary, X: np.ndarray) -> np.ndarray:
    X_use = _append_bias(X)
    return np.asarray([model.predict_proba(row) for row in X_use], dtype=float)


def _first_alarm_time(
    eval_indices: np.ndarray,
    probs: np.ndarray,
    *,
    threshold: float,
    min_consecutive: int,
) -> int | None:
    streak = 0
    idxs = np.asarray(eval_indices, dtype=int).reshape(-1)
    ps = np.asarray(probs, dtype=float).reshape(-1)
    for local_idx, prob in enumerate(ps):
        streak = streak + 1 if prob >= threshold else 0
        if streak >= int(min_consecutive):
            return int(idxs[local_idx] + 1)
    return None


def _choose_threshold_from_episodes(
    rows: Sequence[tuple[int, np.ndarray, np.ndarray, np.ndarray, int]],
    probs: np.ndarray,
    *,
    grid: Sequence[float],
    min_consecutive: int,
) -> float:
    best_threshold = 0.5
    best_score = float("-inf")
    probs_use = np.asarray(probs, dtype=float).reshape(-1)
    for thr in grid:
        offset = 0
        false_alarm_rate = 0.0
        detection_rate = 0.0
        delay_penalty = 0.0
        for _, _, labels, eval_indices, change_index in rows:
            length = len(labels)
            part = probs_use[offset:offset + length]
            offset += length
            first_alarm = _first_alarm_time(
                eval_indices,
                part,
                threshold=float(thr),
                min_consecutive=min_consecutive,
            )
            if first_alarm is not None and int(first_alarm) < int(change_index):
                false_alarm_rate += 1.0
            elif first_alarm is not None:
                detection_rate += 1.0
                delay_penalty += max(0.0, float(first_alarm - int(change_index)))
            else:
                delay_penalty += 0.5 * float(len(eval_indices))
        n_rows = max(len(rows), 1)
        score = (
            detection_rate / n_rows
            - 2.0 * false_alarm_rate / n_rows
            - 0.002 * delay_penalty / n_rows
        )
        if score > best_score or (score == best_score and float(thr) > best_threshold):
            best_score = score
            best_threshold = float(thr)
    return float(best_threshold)


def _episode_pointwise_rows(
    *,
    split: str,
    variant: str,
    seed: int,
    eval_indices: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    change_index: int,
) -> pd.DataFrame:
    target_time = np.asarray(eval_indices, dtype=int) + 1
    return pd.DataFrame(
        {
            "split": split,
            "variant": variant,
            "seed": int(seed),
            "target_time": target_time,
            "label": np.asarray(labels, dtype=float),
            "prob_post_change": np.asarray(probs, dtype=float),
            "change_index": int(change_index),
        }
    )


def _episode_detection_row(
    *,
    split: str,
    variant: str,
    seed: int,
    eval_indices: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    change_index: int,
    threshold: float,
    min_consecutive: int,
) -> dict[str, Any]:
    first_alarm = _first_alarm_time(
        eval_indices,
        probs,
        threshold=threshold,
        min_consecutive=min_consecutive,
    )
    labels_use = np.asarray(labels, dtype=float)
    probs_use = np.asarray(probs, dtype=float)
    pre_mask = labels_use <= 0.5
    post_mask = labels_use > 0.5
    detected = first_alarm is not None and int(first_alarm) >= int(change_index)
    false_alarm = first_alarm is not None and int(first_alarm) < int(change_index)
    delay = float(first_alarm - int(change_index)) if detected else float("nan")
    return {
        "split": split,
        "variant": variant,
        "seed": int(seed),
        "change_index": int(change_index),
        "threshold": float(threshold),
        "min_consecutive": int(min_consecutive),
        "first_alarm_time": int(first_alarm) if first_alarm is not None else np.nan,
        "detected_post_change": int(detected),
        "false_alarm": int(false_alarm),
        "detection_delay": delay,
        "brier": float(np.mean((probs_use - labels_use) ** 2)),
        "auc": _binary_auc(labels_use, probs_use),
        "mean_prob_pre": float(np.mean(probs_use[pre_mask])) if np.any(pre_mask) else np.nan,
        "mean_prob_post": float(np.mean(probs_use[post_mask])) if np.any(post_mask) else np.nan,
    }


def _render_report(
    *,
    out_dir: Path,
    exp_cfg: PiecewiseChangeConfig,
    selections_df: pd.DataFrame,
    detector_summary_df: pd.DataFrame,
    episode_results_df: pd.DataFrame,
) -> None:
    lines = [
        "# Factor-Based Changepoint Report",
        "",
        "This workflow fits stable-regime predictors offline, then uses an online FTRL head to score post-change probability from residual and readout-factor features.",
        "",
        "## Episode Setup",
        "",
        f"- base task: `{exp_cfg.task_name}` from suite `{exp_cfg.suite}`",
        f"- pre params: `{json.dumps(exp_cfg.pre_params, ensure_ascii=False, sort_keys=True)}`",
        f"- post params: `{json.dumps(exp_cfg.post_params, ensure_ascii=False, sort_keys=True)}`",
        f"- fit window: `{exp_cfg.fit_train_len}+{exp_cfg.fit_val_len}`",
        f"- changepoint index: `{exp_cfg.change_index}`",
        f"- total length: `{exp_cfg.total_length}`",
        "",
        "## Offline Predictor Selection",
        "",
        selections_df.to_markdown(index=False) if not selections_df.empty else "No predictor rows were produced.",
        "",
        "## Detector Summary",
        "",
        detector_summary_df.to_markdown(index=False) if not detector_summary_df.empty else "No detector rows were produced.",
        "",
        "## Test Episode Results",
        "",
        episode_results_df[episode_results_df["split"] == "test"].to_markdown(index=False)
        if not episode_results_df.empty
        else "No episode rows were produced.",
        "",
    ]
    (out_dir / "factor_changepoint_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_factor_changepoint_experiment(
    *,
    out_dir: str,
    experiment_cfg: PiecewiseChangeConfig | None = None,
    detector_cfg: DetectorTrainConfig | None = None,
    model_names: Sequence[str] | None = None,
    train_seeds: Sequence[int] | None = None,
    test_seeds: Sequence[int] | None = None,
    grid_mode: str = "quick",
) -> dict[str, pd.DataFrame]:
    exp_cfg = experiment_cfg or PiecewiseChangeConfig()
    det_cfg = detector_cfg or DetectorTrainConfig()
    predictor_names = tuple(model_names or DEFAULT_MODEL_NAMES)
    if not predictor_names:
        raise ValueError("At least one predictor model_name is required")
    train_seed_list = tuple(int(seed) for seed in (train_seeds or DEFAULT_TRAIN_SEEDS))
    test_seed_list = tuple(int(seed) for seed in (test_seeds or DEFAULT_TEST_SEEDS))
    out_path = Path(out_dir)
    ensure_dir(out_path)

    task = _lookup_task(exp_cfg.suite, exp_cfg.task_name)
    support_episode = simulate_piecewise_episode(task, exp_cfg, seed=int(train_seed_list[0]))

    selections: list[PredictorSelection] = []
    template_seed = int(train_seed_list[0])
    selection_factory = ReservoirTemplateFactory(seed=template_seed)
    for model_name in predictor_names:
        selections.append(
            _select_predictor(
                task,
                support_episode,
                model_name=model_name,
                cfg=exp_cfg,
                grid_mode=grid_mode,
                template_factory=selection_factory,
            )
        )
    selection_map = {row.model_name: row for row in selections}

    fit_end = int(exp_cfg.fit_train_len + exp_cfg.fit_val_len)
    train_variant_rows: dict[str, list[tuple[int, np.ndarray, np.ndarray, np.ndarray, int]]] = {}
    test_variant_rows: dict[str, list[tuple[int, np.ndarray, np.ndarray, np.ndarray, int]]] = {}

    for split, seeds, holder in (
        ("train", train_seed_list, train_variant_rows),
        ("test", test_seed_list, test_variant_rows),
    ):
        for seed in seeds:
            episode = simulate_piecewise_episode(task, exp_cfg, seed=int(seed))
            episode_dir = out_path / split / f"seed_{seed}"
            ensure_dir(episode_dir)
            np.savez(
                episode_dir / "episode.npz",
                obs=episode.obs,
                states=episode.states,
                change_index=np.array([episode.change_index], dtype=int),
            )
            traces: dict[str, dict[str, np.ndarray]] = {}
            for model_name in predictor_names:
                template_factory = ReservoirTemplateFactory(seed=template_seed)
                predictor = _fit_model(
                    model_name,
                    selection_map[model_name].best_config,
                    fit_series=np.asarray(episode.obs[:fit_end], dtype=float),
                    template_factory=template_factory,
                )
                traces[model_name] = _predictor_trace(
                    predictor,
                    episode.obs,
                    start_eval=fit_end,
                )
            feature_variants = _pooled_feature_variants(traces)
            for variant, (X, _, eval_indices) in feature_variants.items():
                labels = (np.asarray(eval_indices, dtype=int) + 1 >= int(episode.change_index)).astype(float)
                holder.setdefault(variant, []).append(
                    (
                        int(seed),
                        np.asarray(X, dtype=float),
                        np.asarray(labels, dtype=float),
                        np.asarray(eval_indices, dtype=int),
                        int(episode.change_index),
                    )
                )

    detector_rows: list[dict[str, Any]] = []
    pointwise_frames: list[pd.DataFrame] = []
    episode_rows: list[dict[str, Any]] = []

    for variant, rows in train_variant_rows.items():
        X_train = np.vstack([row[1] for row in rows])
        y_train = np.concatenate([row[2] for row in rows])
        mu = np.mean(X_train, axis=0)
        sigma = np.std(X_train, axis=0) + 1e-8
        X_train_std = _standardize_features(X_train, mu, sigma)
        detector = _fit_ftrl_classifier(X_train_std, y_train, cfg=det_cfg)
        train_probs = _predict_probabilities(detector, X_train_std)
        threshold = _choose_threshold_from_episodes(
            rows,
            train_probs,
            grid=det_cfg.threshold_grid,
            min_consecutive=det_cfg.min_consecutive,
        )
        detector_rows.append(
            {
                "variant": variant,
                "threshold": float(threshold),
                "min_consecutive": int(det_cfg.min_consecutive),
                "train_auc": _binary_auc(y_train, train_probs),
                "train_brier": float(np.mean((train_probs - y_train) ** 2)),
                "n_train_rows": int(len(y_train)),
                "n_features": int(X_train.shape[1]),
                "detector_weight_norm": float(np.linalg.norm(detector.weights())),
            }
        )

        start = 0
        for seed, X_part, y_part, eval_indices, change_index in rows:
            stop = start + len(y_part)
            probs = train_probs[start:stop]
            pointwise_frames.append(
                _episode_pointwise_rows(
                    split="train",
                    variant=variant,
                    seed=seed,
                    eval_indices=eval_indices,
                    labels=y_part,
                    probs=probs,
                    change_index=change_index,
                )
            )
            episode_rows.append(
                _episode_detection_row(
                    split="train",
                    variant=variant,
                    seed=seed,
                    eval_indices=eval_indices,
                    labels=y_part,
                    probs=probs,
                    change_index=change_index,
                    threshold=threshold,
                    min_consecutive=det_cfg.min_consecutive,
                )
            )
            start = stop

        for seed, X_part, y_part, eval_indices, change_index in test_variant_rows.get(variant, []):
            probs = _predict_probabilities(detector, _standardize_features(X_part, mu, sigma))
            pointwise_frames.append(
                _episode_pointwise_rows(
                    split="test",
                    variant=variant,
                    seed=seed,
                    eval_indices=eval_indices,
                    labels=y_part,
                    probs=probs,
                    change_index=change_index,
                )
            )
            episode_rows.append(
                _episode_detection_row(
                    split="test",
                    variant=variant,
                    seed=seed,
                    eval_indices=eval_indices,
                    labels=y_part,
                    probs=probs,
                    change_index=change_index,
                    threshold=threshold,
                    min_consecutive=det_cfg.min_consecutive,
                )
            )

    selections_df = pd.DataFrame(
        [
            {
                "model_name": row.model_name,
                "best_config": to_jsonable(row.best_config),
                **{f"select_{key}": value for key, value in row.selection_metrics.items()},
            }
            for row in selections
        ]
    )
    detector_summary_df = pd.DataFrame(detector_rows)
    episode_results_df = pd.DataFrame(episode_rows)
    pointwise_df = pd.concat(pointwise_frames, axis=0, ignore_index=True) if pointwise_frames else pd.DataFrame()

    if not selections_df.empty:
        selections_df.to_csv(out_path / "predictor_selection.csv", index=False)
    if not detector_summary_df.empty:
        detector_summary_df.to_csv(out_path / "detector_summary.csv", index=False)
    if not pointwise_df.empty:
        pointwise_df.to_csv(out_path / "pointwise_probabilities.csv", index=False)
    if not episode_results_df.empty:
        episode_results_df.to_csv(out_path / "episode_detection_results.csv", index=False)

    manifest = {
        "out_dir": str(out_path),
        "predictor_selection": str(out_path / "predictor_selection.csv"),
        "detector_summary": str(out_path / "detector_summary.csv"),
        "pointwise_probabilities": str(out_path / "pointwise_probabilities.csv"),
        "episode_detection_results": str(out_path / "episode_detection_results.csv"),
    }
    (out_path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _render_report(
        out_dir=out_path,
        exp_cfg=exp_cfg,
        selections_df=selections_df,
        detector_summary_df=detector_summary_df,
        episode_results_df=episode_results_df,
    )
    return {
        "predictor_selection": selections_df,
        "detector_summary": detector_summary_df,
        "episode_detection_results": episode_results_df,
        "pointwise_probabilities": pointwise_df,
    }
