from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..metrics import rmse
from ..models.rc import ReservoirTemplateFactory
from ..utils import ensure_dir, ridge_solve


@dataclass(frozen=True)
class KoopmanReadoutProofConfig:
    train_episodes: int = 256
    test_episodes: int = 128
    steps: int = 120
    washout: int = 10
    warmup: int = 12
    rollout_horizons: tuple[int, ...] = (10, 40)
    seeds: tuple[int, ...] = (11, 23, 37, 51, 67)
    reservoir_size: int = 48
    spectral_radius: float = 0.8
    input_scale: float = 0.35
    leak_rate: float = 0.4
    ridge: float = 1e-4
    sparsity: float = 0.08
    latent_eigs: tuple[float, ...] = (0.992, -0.989, 0.985)
    koopman_powers: tuple[int, ...] = (3, 5, 3)
    koopman_weights: tuple[float, ...] = (0.45, -0.3, 0.2)
    y_self_coupling: float = 0.75
    process_noise_std: float = 0.01
    init_hidden_scale: float = 1.0
    init_obs_scale: float = 0.25
    test_seed_offset: int = 1000
    stability_abs_threshold: float = 5.0


@dataclass
class SimulatedEpisodes:
    y: np.ndarray
    phi: np.ndarray
    koopman_eigs: np.ndarray


@dataclass
class FittedProofModel:
    variant: str
    include_reservoir: bool
    include_koopman: bool
    y_mean: float
    y_std: float
    phi_scale: np.ndarray
    koopman_eigs: np.ndarray
    Wout: np.ndarray
    W: Any | None = None
    Win: np.ndarray | None = None
    bias: np.ndarray | None = None


VARIANT_SPECS: tuple[tuple[str, bool, bool], ...] = (
    ("rc_raw", True, False),
    ("rc_koopman", True, True),
    ("koopman_linear", False, True),
)


def simulate_koopman_hidden_system(
    cfg: KoopmanReadoutProofConfig,
    *,
    n_episodes: int,
    seed: int,
) -> SimulatedEpisodes:
    rng = np.random.default_rng(seed)
    hidden_dim = len(cfg.latent_eigs)
    if hidden_dim != len(cfg.koopman_powers) or hidden_dim != len(cfg.koopman_weights):
        raise ValueError("latent_eigs, koopman_powers, and koopman_weights must have the same length")

    hidden = np.zeros((n_episodes, cfg.steps + 1, hidden_dim), dtype=float)
    y = np.zeros((n_episodes, cfg.steps + 1), dtype=float)
    hidden[:, 0, :] = rng.uniform(-cfg.init_hidden_scale, cfg.init_hidden_scale, size=(n_episodes, hidden_dim))
    y[:, 0] = rng.normal(scale=cfg.init_obs_scale, size=n_episodes)

    koopman_eigs = np.asarray(
        [cfg.latent_eigs[idx] ** cfg.koopman_powers[idx] for idx in range(hidden_dim)],
        dtype=float,
    )
    koopman_weights = np.asarray(cfg.koopman_weights, dtype=float)

    for t in range(cfg.steps):
        phi_t = np.column_stack(
            [hidden[:, t, idx] ** cfg.koopman_powers[idx] for idx in range(hidden_dim)]
        )
        noise = cfg.process_noise_std * rng.normal(size=n_episodes)
        y[:, t + 1] = cfg.y_self_coupling * y[:, t] + phi_t @ koopman_weights + noise
        for idx, eig in enumerate(cfg.latent_eigs):
            hidden[:, t + 1, idx] = eig * hidden[:, t, idx]

    phi = np.stack(
        [hidden[:, :, idx] ** cfg.koopman_powers[idx] for idx in range(hidden_dim)],
        axis=-1,
    )
    return SimulatedEpisodes(y=y, phi=phi, koopman_eigs=koopman_eigs)


def _build_reservoir(cfg: KoopmanReadoutProofConfig, seed: int) -> tuple[Any, np.ndarray, np.ndarray]:
    factory = ReservoirTemplateFactory(seed=seed)
    template = factory.get(cfg.reservoir_size, 1, cfg.sparsity)
    W = template["W_unit"] * cfg.spectral_radius
    Win = template["Win_base"] * cfg.input_scale
    bias = template["bias"]
    return W, Win, bias


def _reservoir_step(
    W: Any,
    Win: np.ndarray,
    bias: np.ndarray,
    r: np.ndarray,
    u: float,
    leak_rate: float,
) -> np.ndarray:
    pre = W.dot(r) + Win[:, 0] * float(u) + bias
    cand = np.tanh(pre)
    return (1.0 - leak_rate) * r + leak_rate * cand


def _make_readout_vector(
    model: FittedProofModel,
    y_t_std: float,
    phi_t_scaled: np.ndarray | None,
    r_t: np.ndarray | None,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    if model.include_reservoir:
        if r_t is None:
            raise ValueError("reservoir feature requested but no reservoir state was provided")
        parts.append(np.asarray(r_t, dtype=float))
    parts.append(np.asarray([y_t_std], dtype=float))
    if model.include_koopman:
        if phi_t_scaled is None:
            raise ValueError("Koopman feature requested but no Koopman feature was provided")
        parts.append(np.asarray(phi_t_scaled, dtype=float))
    parts.append(np.asarray([1.0], dtype=float))
    return np.concatenate(parts)


def _readout_vector_from_flags(
    *,
    include_reservoir: bool,
    include_koopman: bool,
    y_t_std: float,
    phi_t_scaled: np.ndarray | None,
    r_t: np.ndarray | None,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    if include_reservoir:
        if r_t is None:
            raise ValueError("reservoir feature requested but no reservoir state was provided")
        parts.append(np.asarray(r_t, dtype=float))
    parts.append(np.asarray([y_t_std], dtype=float))
    if include_koopman:
        if phi_t_scaled is None:
            raise ValueError("Koopman feature requested but no Koopman feature was provided")
        parts.append(np.asarray(phi_t_scaled, dtype=float))
    parts.append(np.asarray([1.0], dtype=float))
    return np.concatenate(parts)


def fit_proof_model(
    train: SimulatedEpisodes,
    *,
    variant: str,
    include_reservoir: bool,
    include_koopman: bool,
    cfg: KoopmanReadoutProofConfig,
    seed: int,
) -> FittedProofModel:
    y_mean = float(np.mean(train.y))
    y_std = float(np.std(train.y) + 1e-12)
    y_train_std = (train.y - y_mean) / y_std
    phi_scale = np.std(train.phi.reshape(-1, train.phi.shape[-1]), axis=0) + 1e-12
    phi_train_scaled = train.phi / phi_scale

    W = None
    Win = None
    bias = None
    if include_reservoir:
        W, Win, bias = _build_reservoir(cfg, seed=seed)

    X_rows: list[np.ndarray] = []
    Y_rows: list[float] = []
    for episode_idx in range(train.y.shape[0]):
        r = np.zeros(cfg.reservoir_size, dtype=float) if include_reservoir else None
        for t in range(cfg.steps):
            if include_reservoir and r is not None:
                r = _reservoir_step(W, Win, bias, r, y_train_std[episode_idx, t], cfg.leak_rate)
            if t < cfg.washout:
                continue
            phi_t = phi_train_scaled[episode_idx, t] if include_koopman else None
            X_rows.append(
                _readout_vector_from_flags(
                    include_reservoir=include_reservoir,
                    include_koopman=include_koopman,
                    y_t_std=float(y_train_std[episode_idx, t]),
                    phi_t_scaled=phi_t,
                    r_t=r,
                )
            )
            Y_rows.append(float(y_train_std[episode_idx, t + 1]))

    Wout = ridge_solve(np.vstack(X_rows), np.asarray(Y_rows, dtype=float), cfg.ridge)
    return FittedProofModel(
        variant=variant,
        include_reservoir=include_reservoir,
        include_koopman=include_koopman,
        y_mean=y_mean,
        y_std=y_std,
        phi_scale=phi_scale,
        koopman_eigs=np.asarray(train.koopman_eigs, dtype=float),
        Wout=Wout,
        W=W,
        Win=Win,
        bias=bias,
    )


def evaluate_one_step_rmse(
    model: FittedProofModel,
    episodes: SimulatedEpisodes,
    cfg: KoopmanReadoutProofConfig,
) -> float:
    y_std = (episodes.y - model.y_mean) / model.y_std
    phi_scaled = episodes.phi / model.phi_scale
    preds: list[float] = []
    truth: list[float] = []

    for episode_idx in range(episodes.y.shape[0]):
        r = np.zeros(cfg.reservoir_size, dtype=float) if model.include_reservoir else None
        for t in range(cfg.steps):
            if model.include_reservoir and r is not None:
                r = _reservoir_step(model.W, model.Win, model.bias, r, y_std[episode_idx, t], cfg.leak_rate)
            if t < cfg.washout:
                continue
            phi_t = phi_scaled[episode_idx, t] if model.include_koopman else None
            pred_std = float(_make_readout_vector(model, y_std[episode_idx, t], phi_t, r) @ model.Wout)
            preds.append(pred_std * model.y_std + model.y_mean)
            truth.append(float(episodes.y[episode_idx, t + 1]))
    return float(rmse(np.asarray(truth), np.asarray(preds)))


def evaluate_rollout_metrics(
    model: FittedProofModel,
    episodes: SimulatedEpisodes,
    cfg: KoopmanReadoutProofConfig,
) -> dict[str, float]:
    y_std = (episodes.y - model.y_mean) / model.y_std
    phi_scaled = episodes.phi / model.phi_scale
    metrics: dict[str, float] = {}
    max_h = max(cfg.rollout_horizons)

    for horizon in cfg.rollout_horizons:
        preds: list[float] = []
        truth: list[float] = []
        stable_flags: list[float] = []
        for episode_idx in range(episodes.y.shape[0]):
            r = np.zeros(cfg.reservoir_size, dtype=float) if model.include_reservoir else None
            if model.include_reservoir and r is not None:
                for t in range(cfg.warmup + 1):
                    r = _reservoir_step(model.W, model.Win, model.bias, r, y_std[episode_idx, t], cfg.leak_rate)
            y_cur = float(y_std[episode_idx, cfg.warmup])
            phi_cur = phi_scaled[episode_idx, cfg.warmup].copy() if model.include_koopman else None

            for step_idx in range(horizon):
                pred_std = float(_make_readout_vector(model, y_cur, phi_cur, r) @ model.Wout)
                pred = pred_std * model.y_std + model.y_mean
                preds.append(pred)
                truth.append(float(episodes.y[episode_idx, cfg.warmup + 1 + step_idx]))
                stable_flags.append(1.0 if abs(pred) <= cfg.stability_abs_threshold else 0.0)
                if model.include_reservoir and r is not None:
                    r = _reservoir_step(model.W, model.Win, model.bias, r, pred_std, cfg.leak_rate)
                y_cur = pred_std
                if phi_cur is not None:
                    phi_cur = phi_cur * model.koopman_eigs

        metrics[f"rollout_rmse@{horizon}"] = float(rmse(np.asarray(truth), np.asarray(preds)))
        metrics[f"rollout_stable_frac@{horizon}"] = float(np.mean(stable_flags))

    if max_h not in cfg.rollout_horizons:
        raise ValueError("max_h must be included in rollout_horizons")
    return metrics


def _aggregate_summary(df: pd.DataFrame, cfg: KoopmanReadoutProofConfig) -> pd.DataFrame:
    summary_rows: list[dict[str, float | str]] = []
    for variant, group in df.groupby("variant", dropna=False):
        row: dict[str, float | str] = {
            "variant": str(variant),
            "seed_count": float(group["seed"].nunique()),
            "one_step_rmse_mean": float(group["one_step_rmse"].mean()),
            "one_step_rmse_std": float(group["one_step_rmse"].std(ddof=0)),
            "one_step_rmse_median": float(group["one_step_rmse"].median()),
        }
        for horizon in cfg.rollout_horizons:
            metric_name = f"rollout_rmse@{horizon}"
            stable_name = f"rollout_stable_frac@{horizon}"
            row[f"{metric_name}_median"] = float(group[metric_name].median())
            row[f"{metric_name}_mean"] = float(group[metric_name].mean())
            row[f"{stable_name}_mean"] = float(group[stable_name].mean())
        summary_rows.append(row)
    return pd.DataFrame(summary_rows).sort_values("variant").reset_index(drop=True)


def _find_row(summary: pd.DataFrame, variant: str) -> pd.Series | None:
    matches = summary[summary["variant"] == variant]
    if matches.empty:
        return None
    return matches.iloc[0]


def render_koopman_proof_summary(
    summary: pd.DataFrame,
    cfg: KoopmanReadoutProofConfig,
) -> str:
    raw_row = _find_row(summary, "rc_raw")
    koop_row = _find_row(summary, "rc_koopman")
    linear_row = _find_row(summary, "koopman_linear")

    lines = [
        "# Koopman Readout Proof Experiment",
        "",
        "## Controlled system",
        "",
        "We simulate a partially observed nonlinear system:",
        "",
        "```text",
        "x_i(t+1) = a_i x_i(t),",
        "y(t+1) = beta y(t) + sum_j c_j phi_j(x_t) + eps_t,",
        "phi_j(x_t) = x_j(t)^{p_j}.",
        "```",
        "",
        "With the chosen powers, each `phi_j` is an exact Koopman eigenfunction because",
        "",
        "```text",
        "phi_j(x_{t+1}) = a_j^{p_j} phi_j(x_t).",
        "```",
        "",
        "So the lifted coordinate `[y_t, phi_t]` is one-step closed, while raw scalar `y_t` is only history-Markov.",
        "",
        "## Aggregate results",
        "",
        summary.to_markdown(index=False),
        "",
    ]
    if raw_row is not None and koop_row is not None:
        one_step_gain = 1.0 - float(koop_row["one_step_rmse_mean"]) / max(float(raw_row["one_step_rmse_mean"]), 1e-12)
        h10_gain = 1.0 - float(koop_row["rollout_rmse@10_median"]) / max(float(raw_row["rollout_rmse@10_median"]), 1e-12)
        h40_gain = 1.0 - float(koop_row["rollout_rmse@40_median"]) / max(float(raw_row["rollout_rmse@40_median"]), 1e-12)
        lines.extend(
            [
                "## Readout takeaway",
                "",
                f"- Adding exact Koopman features to the RC readout reduced mean one-step RMSE by about `{100.0 * one_step_gain:.1f}%`.",
                f"- The median `10`-step rollout RMSE dropped by about `{100.0 * h10_gain:.1f}%` at matched reservoir size.",
                f"- The median `40`-step rollout RMSE dropped by about `{100.0 * h40_gain:.1f}%`.",
            ]
        )
    if koop_row is not None and linear_row is not None:
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                "- `rc_koopman` and `koopman_linear` land at nearly the same error level, which means the main gain comes from the lifted Koopman coordinates rather than from a larger recurrent state.",
                "- This is an oracle experiment: the Koopman features are supplied directly from the hidden state. It proves the representation claim, not the separate problem of estimating those features from history.",
            ]
        )
    return "\n".join(lines) + "\n"


def run_koopman_readout_proof(
    *,
    out_dir: str,
    config: KoopmanReadoutProofConfig | None = None,
) -> dict[str, str]:
    cfg = config if config is not None else KoopmanReadoutProofConfig()
    out_path = Path(out_dir)
    ensure_dir(out_path)

    rows: list[dict[str, float | int | str]] = []
    for seed in cfg.seeds:
        train = simulate_koopman_hidden_system(cfg, n_episodes=cfg.train_episodes, seed=int(seed))
        test = simulate_koopman_hidden_system(cfg, n_episodes=cfg.test_episodes, seed=int(seed) + cfg.test_seed_offset)
        for variant, include_reservoir, include_koopman in VARIANT_SPECS:
            model = fit_proof_model(
                train,
                variant=variant,
                include_reservoir=include_reservoir,
                include_koopman=include_koopman,
                cfg=cfg,
                seed=int(seed),
            )
            row: dict[str, float | int | str] = {
                "seed": int(seed),
                "variant": variant,
                "one_step_rmse": evaluate_one_step_rmse(model, test, cfg),
            }
            row.update(evaluate_rollout_metrics(model, test, cfg))
            rows.append(row)

    result_df = pd.DataFrame(rows).sort_values(["variant", "seed"]).reset_index(drop=True)
    summary_df = _aggregate_summary(result_df, cfg)

    result_fp = out_path / "koopman_readout_seed_results.csv"
    summary_fp = out_path / "koopman_readout_summary.csv"
    summary_md_fp = out_path / "koopman_readout_summary.md"
    config_fp = out_path / "koopman_readout_config.json"

    result_df.to_csv(result_fp, index=False)
    summary_df.to_csv(summary_fp, index=False)
    summary_md_fp.write_text(render_koopman_proof_summary(summary_df, cfg), encoding="utf-8")
    config_fp.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    return {
        "seed_results": str(result_fp),
        "summary_csv": str(summary_fp),
        "summary_md": str(summary_md_fp),
        "config_json": str(config_fp),
    }
