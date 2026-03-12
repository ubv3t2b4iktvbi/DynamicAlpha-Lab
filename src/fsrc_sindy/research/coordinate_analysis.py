from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PolynomialFeatures
from tqdm.auto import tqdm

from ..benchmarks import build_suite
from ..factors import DynamicsFeatureConfig, DynamicsFeatureEngine
from ..fastslow import CausalFastSlowEncoder, FastSlowConfig
from ..metrics import r2_score, rmse
from ..systems import SYSTEMS, BenchmarkTask, rk4_step, simulate_task
from ..utils import ensure_dir, set_seed


@dataclass(frozen=True)
class CoordinateSequence:
    name: str
    values: np.ndarray
    columns: tuple[str, ...]
    offset: int
    notes: str = ""


def _standardize_by_train(y: np.ndarray, n_train: int) -> np.ndarray:
    y = np.asarray(y, dtype=float).reshape(-1)
    mu = float(np.mean(y[:n_train]))
    std = float(np.std(y[:n_train]) + 1e-12)
    return (y - mu) / std


def _build_delay_embedding(y: np.ndarray, delay_dim: int) -> CoordinateSequence:
    y = np.asarray(y, dtype=float).reshape(-1)
    if delay_dim < 2:
        raise ValueError("delay_dim must be >= 2")
    if len(y) <= delay_dim:
        raise ValueError("series too short for requested delay embedding")
    rows = []
    for t in range(delay_dim - 1, len(y)):
        rows.append([float(y[t - lag]) for lag in range(delay_dim)])
    columns = tuple("delay_lag_" + str(lag) for lag in range(delay_dim))
    return CoordinateSequence(
        name="delay",
        values=np.asarray(rows, dtype=float),
        columns=columns,
        offset=delay_dim - 1,
        notes=f"Takens-style delay embedding with {delay_dim} lags.",
    )


def _build_fastslow_coordinate(y: np.ndarray, dt: float) -> CoordinateSequence:
    encoder = CausalFastSlowEncoder(FastSlowConfig(t0=4.0, slow_scales=(8.0, 16.0, 32.0), dt=dt))
    seq = encoder.build_feature_sequence(y)
    values = np.column_stack([seq["fast"], seq["slow"], seq["m"], seq["resid"]])
    return CoordinateSequence(
        name="fastslow",
        values=np.asarray(values, dtype=float),
        columns=("fast", "slow", "m", "resid"),
        offset=0,
        notes="Causal fast/slow coordinate with order-parameter and residual components.",
    )


def _build_factor_coordinate(y: np.ndarray) -> CoordinateSequence:
    engine = DynamicsFeatureEngine(DynamicsFeatureConfig())
    ctx = engine.build_base_sequence(y)
    columns = ("m_norm", "phase_bottom_score", "energy_ratio", "collapse_quality")
    values = np.column_stack([ctx[name] for name in columns])
    return CoordinateSequence(
        name="factor",
        values=np.asarray(values, dtype=float),
        columns=columns,
        offset=0,
        notes="Hand-crafted dynamical factor coordinate built from causal feature-engine outputs.",
    )


def _build_theory_fastslow_coordinate(y: np.ndarray) -> CoordinateSequence:
    engine = DynamicsFeatureEngine(DynamicsFeatureConfig())
    ctx = engine.build_base_sequence(y)
    columns = (
        "slow_level_norm",
        "slow_drift_norm",
        "timescale_separation",
        "slow_manifold_alignment",
        "adiabatic_coherence",
    )
    values = np.column_stack([ctx[name] for name in columns])
    return CoordinateSequence(
        name="theory_fastslow",
        values=np.asarray(values, dtype=float),
        columns=columns,
        offset=0,
        notes=(
            "Theory-grounded fast/slow coordinate built from a slow-mode proxy, "
            "local timescale separation, slow-manifold alignment, and adiabatic coherence."
        ),
    )


def build_coordinate_sequences(y: np.ndarray, n_train: int, dt: float, coordinate_kinds: Sequence[str], delay_dim: int) -> list[CoordinateSequence]:
    y_std = _standardize_by_train(y, n_train=n_train)
    builders = {
        "raw": lambda: CoordinateSequence(
            name="raw",
            values=y_std.reshape(-1, 1),
            columns=("y",),
            offset=0,
            notes="Standardized scalar observation.",
        ),
        "delay": lambda: _build_delay_embedding(y_std, delay_dim=delay_dim),
        "fastslow": lambda: _build_fastslow_coordinate(y_std, dt=dt),
        "theory_fastslow": lambda: _build_theory_fastslow_coordinate(y_std),
        "factor": lambda: _build_factor_coordinate(y_std),
    }
    sequences: list[CoordinateSequence] = []
    for kind in coordinate_kinds:
        try:
            sequences.append(builders[kind]())
        except KeyError as exc:
            raise ValueError(f"Unknown coordinate kind: {kind}") from exc
    return sequences


def _aligned_lengths(task: BenchmarkTask, offset: int) -> tuple[int, int, int]:
    n_train = max(task.n_train - offset, 0)
    n_val = max(task.n_val, 0)
    n_test = max(task.n_test, 0)
    return n_train, n_val, n_test


def _split_aligned(values: np.ndarray, task: BenchmarkTask, offset: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_train, n_val, n_test = _aligned_lengths(task, offset)
    train = values[:n_train]
    val = values[n_train:n_train + n_val]
    test = values[n_train + n_val:n_train + n_val + n_test]
    return train, val, test


def _fit_polynomial_ridge(inputs: np.ndarray, targets: np.ndarray, alpha: float = 1e-4, degree: int = 2) -> tuple[PolynomialFeatures, Ridge]:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X = poly.fit_transform(np.asarray(inputs, dtype=float))
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X, np.asarray(targets, dtype=float))
    return poly, model


def _predict_transition(poly: PolynomialFeatures, model: Ridge, inputs: np.ndarray) -> np.ndarray:
    X = poly.transform(np.asarray(inputs, dtype=float))
    return np.asarray(model.predict(X), dtype=float)


def _flattened_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(np.asarray(y_true, dtype=float).reshape(-1), np.asarray(y_pred, dtype=float).reshape(-1)))


def _clip01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def evaluate_koopman_invariance(z_train: np.ndarray, z_val: np.ndarray, alpha: float = 1e-4) -> dict[str, float]:
    z_train = np.asarray(z_train, dtype=float)
    z_val = np.asarray(z_val, dtype=float)
    if len(z_train) < 8 or len(z_val) < 8:
        return {
            "koopman_linear_rmse": float("nan"),
            "koopman_linear_r2": float("nan"),
            "koopman_invariance_score": float("nan"),
            "koopman_spectral_radius": float("nan"),
        }
    X_train = z_train[:-1]
    Y_train = z_train[1:]
    A = np.linalg.solve(
        X_train.T @ X_train + alpha * np.eye(X_train.shape[1], dtype=float),
        X_train.T @ Y_train,
    )
    pred = z_val[:-1] @ A
    truth = z_val[1:]
    err = float(rmse(truth.reshape(-1), pred.reshape(-1)))
    scale = float(np.std(truth.reshape(-1)) + 1e-12)
    eigvals = np.linalg.eigvals(A)
    return {
        "koopman_linear_rmse": err,
        "koopman_linear_r2": _flattened_r2(truth, pred),
        "koopman_invariance_score": _clip01(1.0 - err / scale),
        "koopman_spectral_radius": float(np.max(np.abs(eigvals))),
    }


def evaluate_markov_closure(z_train: np.ndarray, z_val: np.ndarray) -> dict[str, float]:
    if len(z_train) < 8 or len(z_val) < 8:
        raise ValueError("Not enough samples to run the Markov closure test")

    markov_poly, markov_model = _fit_polynomial_ridge(z_train[:-1], z_train[1:])
    markov_pred = _predict_transition(markov_poly, markov_model, z_val[:-1])
    markov_true = z_val[1:]

    lag_inputs_train = np.hstack([z_train[1:-1], z_train[:-2]])
    lag_targets_train = z_train[2:]
    lag_inputs_val = np.hstack([z_val[1:-1], z_val[:-2]])
    lag_targets_val = z_val[2:]
    lag_poly, lag_model = _fit_polynomial_ridge(lag_inputs_train, lag_targets_train)
    lag_pred = _predict_transition(lag_poly, lag_model, lag_inputs_val)

    markov_rmse = float(rmse(markov_true.reshape(-1), markov_pred.reshape(-1)))
    lagged_rmse = float(rmse(lag_targets_val.reshape(-1), lag_pred.reshape(-1)))
    markov_gain = float(markov_rmse - lagged_rmse)
    markov_gain_ratio = float(markov_gain / (markov_rmse + 1e-12))
    return {
        "markov_rmse": markov_rmse,
        "lagged_rmse": lagged_rmse,
        "markov_gain": markov_gain,
        "markov_gain_ratio": markov_gain_ratio,
        "markov_r2": _flattened_r2(markov_true, markov_pred),
        "lagged_r2": _flattened_r2(lag_targets_val, lag_pred),
    }


def evaluate_dynamical_separability(z_train: np.ndarray) -> dict[str, float]:
    z_train = np.asarray(z_train, dtype=float)
    if z_train.ndim != 2 or z_train.shape[1] <= 1 or len(z_train) < 16:
        return {
            "offdiag_mi_mean": float("nan"),
            "offdiag_mi_max": float("nan"),
            "offdiag_mi_min": float("nan"),
        }

    dz = z_train[1:] - z_train[:-1]
    z_prev = z_train[:-1]
    scores: list[float] = []
    for target_idx in range(z_train.shape[1]):
        for source_idx in range(z_train.shape[1]):
            if target_idx == source_idx:
                continue
            mi = mutual_info_regression(
                z_prev[:, [source_idx]],
                dz[:, target_idx],
                discrete_features=False,
                random_state=0,
            )
            scores.append(float(mi[0]))
    return {
        "offdiag_mi_mean": float(np.mean(scores)),
        "offdiag_mi_max": float(np.max(scores)),
        "offdiag_mi_min": float(np.min(scores)),
    }


def _true_step_jacobian(task: BenchmarkTask, x: np.ndarray, step_index: int, offset: int) -> np.ndarray:
    rhs = SYSTEMS[task.system]["rhs"]
    x = np.asarray(x, dtype=float)
    dim = x.size
    jac = np.zeros((dim, dim), dtype=float)
    absolute_index = task.burn_in + offset + step_index
    t = float(absolute_index * task.dt)
    for j in range(dim):
        h = 1e-6 * max(1.0, abs(float(x[j])))
        delta = np.zeros(dim, dtype=float)
        delta[j] = h
        plus = rk4_step(rhs, x + delta, t, task.dt, task.params)
        minus = rk4_step(rhs, x - delta, t, task.dt, task.params)
        jac[:, j] = (plus - minus) / (2.0 * h)
    return jac


def _spectral_summary(matrix: np.ndarray) -> dict[str, float]:
    eigvals = np.linalg.eigvals(np.asarray(matrix, dtype=float))
    return {
        "spectral_radius": float(np.max(np.abs(eigvals))),
        "max_real_eig": float(np.max(np.real(eigvals))),
        "fro_norm": float(np.linalg.norm(matrix)),
    }


def _fit_local_linear_map(curr: np.ndarray, nxt: np.ndarray, query_idx: int, neighbor_indices: np.ndarray, ridge: float) -> np.ndarray | None:
    center_x = curr[query_idx]
    center_y = nxt[query_idx]
    X = curr[neighbor_indices] - center_x
    Y = nxt[neighbor_indices] - center_y
    if len(X) < 2 or np.allclose(X, 0.0):
        return None
    p = X.shape[1]
    lhs = X.T @ X + ridge * np.eye(p, dtype=float)
    rhs = X.T @ Y
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return None


def evaluate_spectral_preservation(
    task: BenchmarkTask,
    states_aligned: np.ndarray,
    z_aligned: np.ndarray,
    offset: int,
    val_start: int,
    val_len: int,
    sample_count: int,
    local_k: int,
    ridge: float,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    curr = np.asarray(z_aligned[:-1], dtype=float)
    nxt = np.asarray(z_aligned[1:], dtype=float)
    if len(curr) < 8 or val_len < 4:
        metrics = {
            "spectral_radius_true_mean": float("nan"),
            "spectral_radius_coord_mean": float("nan"),
            "spectral_radius_rmse": float("nan"),
            "spectral_radius_corr": float("nan"),
            "max_real_eig_rmse": float("nan"),
            "jacobian_samples": 0.0,
        }
        return metrics, []

    end_index = min(val_start + val_len - 1, len(curr) - 1)
    start_index = min(val_start, end_index)
    sample_indices = np.linspace(start_index, end_index, num=max(1, min(sample_count, end_index - start_index + 1)), dtype=int)
    nn = NearestNeighbors(n_neighbors=min(local_k, len(curr)))
    nn.fit(curr)

    details: list[dict[str, float]] = []
    true_radii: list[float] = []
    coord_radii: list[float] = []
    true_real: list[float] = []
    coord_real: list[float] = []

    for idx in sample_indices:
        neighbor_indices = nn.kneighbors(curr[idx : idx + 1], return_distance=False)[0]
        local_map = _fit_local_linear_map(curr, nxt, idx, neighbor_indices, ridge=ridge)
        if local_map is None:
            continue
        coord_spec = _spectral_summary(local_map)
        true_spec = _spectral_summary(_true_step_jacobian(task, states_aligned[idx], step_index=idx, offset=offset))
        true_radii.append(true_spec["spectral_radius"])
        coord_radii.append(coord_spec["spectral_radius"])
        true_real.append(true_spec["max_real_eig"])
        coord_real.append(coord_spec["max_real_eig"])
        details.append(
            {
                "sample_index": float(idx),
                "true_spectral_radius": true_spec["spectral_radius"],
                "coord_spectral_radius": coord_spec["spectral_radius"],
                "true_max_real_eig": true_spec["max_real_eig"],
                "coord_max_real_eig": coord_spec["max_real_eig"],
            }
        )

    if not details:
        metrics = {
            "spectral_radius_true_mean": float("nan"),
            "spectral_radius_coord_mean": float("nan"),
            "spectral_radius_rmse": float("nan"),
            "spectral_radius_corr": float("nan"),
            "max_real_eig_rmse": float("nan"),
            "jacobian_samples": 0.0,
        }
        return metrics, []

    if len(details) > 1:
        corr = float(np.corrcoef(true_radii, coord_radii)[0, 1])
    else:
        corr = float("nan")
    metrics = {
        "spectral_radius_true_mean": float(np.mean(true_radii)),
        "spectral_radius_coord_mean": float(np.mean(coord_radii)),
        "spectral_radius_rmse": float(rmse(np.asarray(true_radii), np.asarray(coord_radii))),
        "spectral_radius_corr": corr,
        "max_real_eig_rmse": float(rmse(np.asarray(true_real), np.asarray(coord_real))),
        "jacobian_samples": float(len(details)),
    }
    return metrics, details


def _interpret_coordinate(row: pd.Series) -> str:
    notes: list[str] = []
    gain_ratio = float(row.get("markov_gain_ratio", np.nan))
    if np.isfinite(gain_ratio):
        if gain_ratio > 0.15:
            notes.append("lagged closure helps materially, so the coordinate is not close to Markov")
        elif gain_ratio < 0.03:
            notes.append("single-step closure is already strong, so the coordinate is closer to a Markov state")

    spectral_corr = float(row.get("spectral_radius_corr", np.nan))
    spectral_rmse = float(row.get("spectral_radius_rmse", np.nan))
    if np.isfinite(spectral_corr) and np.isfinite(spectral_rmse):
        if spectral_corr > 0.4 and spectral_rmse < 1.0:
            notes.append("local expansion and contraction are being preserved reasonably well")
        elif spectral_corr < 0.2:
            notes.append("local spectral structure is poorly preserved, so long-rollout stability is suspect")

    koopman_score = float(row.get("koopman_invariance_score", np.nan))
    if np.isfinite(koopman_score):
        if koopman_score > 0.8:
            notes.append("the coordinate is close to a linear invariant subspace proxy")
        elif koopman_score < 0.4:
            notes.append("the coordinate is a poor Koopman-style linearization")

    offdiag_mi = float(row.get("offdiag_mi_mean", np.nan))
    if np.isfinite(offdiag_mi):
        if offdiag_mi < 0.05:
            notes.append("coordinate updates are relatively weakly coupled")
        elif offdiag_mi > 0.15:
            notes.append("coordinate components remain strongly entangled")

    if not notes:
        notes.append("no single mechanism dominates; inspect the raw tables before committing to an interpretation")
    return "; ".join(notes)


def render_coordinate_report(task: BenchmarkTask, df: pd.DataFrame) -> str:
    ordered = df.sort_values(["markov_gain_ratio", "spectral_radius_rmse"], na_position="last").reset_index(drop=True)
    lines = [
        f"# Coordinate dynamics report: {task.name}",
        "",
        f"- system: {task.system}",
        f"- regime: {task.regime}",
        f"- family: {task.family}",
        f"- observation: {task.obs_mode}",
        "",
        "## Summary table",
        "",
        ordered[
            [
                "coordinate",
                "coord_dim",
                "markov_rmse",
                "lagged_rmse",
                "markov_gain_ratio",
                "koopman_invariance_score",
                "koopman_linear_r2",
                "spectral_radius_rmse",
                "spectral_radius_corr",
                "offdiag_mi_mean",
            ]
        ].to_markdown(index=False),
        "",
        "## Interpretations",
        "",
    ]
    for _, row in ordered.iterrows():
        lines.append(f"- `{row['coordinate']}`: {_interpret_coordinate(row)}")
    lines.extend(
        [
            "",
            "## Follow-up suggestions",
            "",
        ]
    )
    best_markov = ordered.iloc[0]["coordinate"] if not ordered.empty else "n/a"
    best_spectral = ordered.sort_values("spectral_radius_rmse", na_position="last").iloc[0]["coordinate"] if not ordered.empty else "n/a"
    best_koopman = ordered.sort_values("koopman_invariance_score", ascending=False, na_position="last").iloc[0]["coordinate"] if not ordered.empty else "n/a"
    lines.append(f"- Best closure candidate: `{best_markov}`")
    lines.append(f"- Best spectral-preservation candidate: `{best_spectral}`")
    lines.append(f"- Best Koopman-like coordinate: `{best_koopman}`")
    if best_markov == "delay":
        lines.append("- Delay coordinates leading the closure test suggests unresolved memory; prioritize closure terms or memory encoders.")
    if best_koopman == "factor":
        lines.append("- Factor coordinates are the closest linear invariant-subspace proxy here; treat the top factors as approximate Koopman coordinates.")
    if "factor" in ordered["coordinate"].tolist():
        factor_row = ordered[ordered["coordinate"] == "factor"].iloc[0]
        if float(factor_row.get("markov_gain_ratio", np.inf)) < 0.1:
            lines.append("- Factor coordinates are close to Markov on this task; they are good candidates for structured residual or Koopman-style follow-up work.")
    if "fastslow" in ordered["coordinate"].tolist():
        fs_row = ordered[ordered["coordinate"] == "fastslow"].iloc[0]
        if float(fs_row.get("offdiag_mi_mean", np.inf)) < 0.1 and float(fs_row.get("spectral_radius_corr", -np.inf)) < 0.2:
            lines.append("- Fast-slow coordinates appear to decouple dynamics while distorting local geometry; retune timescales or learn slow coordinates rather than fixing them.")
    if "theory_fastslow" in ordered["coordinate"].tolist():
        theory_row = ordered[ordered["coordinate"] == "theory_fastslow"].iloc[0]
        if float(theory_row.get("koopman_invariance_score", -np.inf)) >= 0.7 and float(theory_row.get("markov_gain_ratio", np.inf)) <= 0.1:
            lines.append("- Theory fast-slow coordinates are simultaneously close to Markov and Koopman-like; use them to guide factor selection and structured residual follow-up.")
    return "\n".join(lines)


def run_coordinate_analysis_for_task(
    task: BenchmarkTask,
    out_dir: Path,
    coordinate_kinds: Sequence[str],
    delay_dim: int,
    seed: int,
    sample_count: int,
    local_k: int,
    ridge: float,
) -> pd.DataFrame:
    sim = simulate_task(task, seed=seed)
    sequences = build_coordinate_sequences(
        y=sim.obs,
        n_train=task.n_train,
        dt=task.dt,
        coordinate_kinds=coordinate_kinds,
        delay_dim=delay_dim,
    )
    rows: list[dict[str, object]] = []
    detail_payload: dict[str, object] = {
        "task": task.name,
        "system": task.system,
        "coordinates": {},
    }

    for seq in sequences:
        states_aligned = sim.states[seq.offset : seq.offset + len(seq.values)]
        z_train, z_val, z_test = _split_aligned(seq.values, task=task, offset=seq.offset)
        val_start = max(task.n_train - seq.offset, 0)
        val_len = len(z_val)
        markov = evaluate_markov_closure(z_train, z_val)
        separability = evaluate_dynamical_separability(z_train)
        spectral, details = evaluate_spectral_preservation(
            task=task,
            states_aligned=states_aligned,
            z_aligned=seq.values,
            offset=seq.offset,
            val_start=val_start,
            val_len=val_len,
            sample_count=sample_count,
            local_k=local_k,
            ridge=ridge,
        )
        row = {
            "task": task.name,
            "coordinate": seq.name,
            "coord_dim": int(seq.values.shape[1]),
            "offset": int(seq.offset),
            "train_len": int(len(z_train)),
            "val_len": int(len(z_val)),
            "test_len": int(len(z_test)),
            "notes": seq.notes,
        }
        row.update(markov)
        row.update(evaluate_koopman_invariance(z_train, z_val))
        row.update(separability)
        row.update(spectral)
        rows.append(row)
        detail_payload["coordinates"][seq.name] = {
            "columns": list(seq.columns),
            "offset": seq.offset,
            "notes": seq.notes,
            "spectral_samples": details,
        }

    df = pd.DataFrame(rows).sort_values("coordinate").reset_index(drop=True)
    ensure_dir(out_dir)
    (out_dir / "coordinate_summary.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    (out_dir / "coordinate_summary.md").write_text(render_coordinate_report(task, df), encoding="utf-8")
    (out_dir / "coordinate_details.json").write_text(json.dumps(detail_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return df


def run_coordinate_analysis_suite(
    suite: str,
    out_dir: str,
    seed: int = 123,
    task_names: Sequence[str] | None = None,
    coordinate_kinds: Sequence[str] = ("raw", "delay", "fastslow", "factor"),
    task_coordinate_kinds: Mapping[str, Sequence[str]] | None = None,
    delay_dim: int = 8,
    sample_count: int = 24,
    local_k: int = 64,
    ridge: float = 1e-4,
) -> pd.DataFrame:
    set_seed(seed)
    out_path = Path(out_dir)
    ensure_dir(out_path)
    tasks = build_suite(suite)
    if task_names:
        wanted = set(task_names)
        tasks = [task for task in tasks if task.name in wanted]
        missing = sorted(wanted - {task.name for task in tasks})
        if missing:
            raise ValueError(f"Unknown task names for suite={suite}: {missing}")

    all_rows: list[pd.DataFrame] = []
    for task in tqdm(tasks, desc=f"coordinate_analysis[{suite}]"):
        task_dir = out_path / task.name
        ensure_dir(task_dir)
        task_kinds = tuple(task_coordinate_kinds.get(task.name, coordinate_kinds)) if task_coordinate_kinds is not None else tuple(coordinate_kinds)
        all_rows.append(
            run_coordinate_analysis_for_task(
                task=task,
                out_dir=task_dir,
                coordinate_kinds=task_kinds,
                delay_dim=delay_dim,
                seed=seed,
                sample_count=sample_count,
                local_k=local_k,
                ridge=ridge,
            )
        )
    result = pd.concat(all_rows, axis=0, ignore_index=True) if all_rows else pd.DataFrame()
    if not result.empty:
        result = result.sort_values(["task", "markov_gain_ratio", "spectral_radius_rmse"], na_position="last").reset_index(drop=True)
    (out_path / "coordinate_analysis_summary.csv").write_text(result.to_csv(index=False), encoding="utf-8")
    return result
