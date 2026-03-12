from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom <= 1e-12:
        return 0.0
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    scale = float(np.std(y_true) + 1e-12)
    out = {
        f"{prefix}rmse": rmse(y_true, y_pred),
        f"{prefix}nrmse": rmse(y_true, y_pred) / scale,
        f"{prefix}r2": r2_score(y_true, y_pred),
        f"{prefix}mae": mae(y_true, y_pred),
    }
    return out


def autocorr(x: np.ndarray, max_lag: int = 20) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x - np.mean(x)
    denom = np.dot(x, x) + 1e-12
    ac = []
    for lag in range(max_lag + 1):
        if lag == 0:
            ac.append(1.0)
        else:
            ac.append(float(np.dot(x[:-lag], x[lag:]) / denom))
    return np.asarray(ac, dtype=float)


def psd_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x - np.mean(x)
    spec = np.abs(np.fft.rfft(x)) ** 2
    spec = spec / (np.sum(spec) + 1e-12)
    return spec


def distribution_metrics(y_true: np.ndarray, y_pred: np.ndarray, acf_lag: int = 20) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    acf_true = autocorr(y_true, max_lag=min(acf_lag, len(y_true) - 2))
    acf_pred = autocorr(y_pred, max_lag=min(acf_lag, len(y_pred) - 2))
    psd_true = psd_norm(y_true)
    psd_pred = psd_norm(y_pred)
    m = min(len(psd_true), len(psd_pred))
    return {
        "mean_gap": float(abs(np.mean(y_true) - np.mean(y_pred))),
        "std_gap": float(abs(np.std(y_true) - np.std(y_pred))),
        "acf_rmse": rmse(acf_true, acf_pred),
        "psd_rmse": rmse(psd_true[:m], psd_pred[:m]),
    }


def evaluate_horizons(model, y_context: np.ndarray, y_future: np.ndarray, horizons: Sequence[int]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    y_future = np.asarray(y_future, dtype=float).reshape(-1)
    scale_ref = float(np.std(y_future) + 1e-12)
    max_h = int(max(horizons))
    full_pred = np.asarray(model.rollout(y_context, horizon=max_h), dtype=float)
    finite_flag = bool(np.isfinite(full_pred).all())
    max_abs = float(np.max(np.abs(full_pred[np.isfinite(full_pred)]))) if finite_flag and len(full_pred) else float("inf")
    out["rollout_finite"] = 1.0 if finite_flag else 0.0
    out["rollout_max_abs"] = max_abs
    for H in horizons:
        pred = np.asarray(full_pred[:H], dtype=float)
        truth = np.asarray(y_future[:H], dtype=float)
        if not np.isfinite(pred).all():
            out[f"rmse@{H}"] = float("inf")
            out[f"nrmse@{H}"] = float("inf")
            out[f"r2@{H}"] = float("-inf")
            out[f"mae@{H}"] = float("inf")
            continue
        out[f"rmse@{H}"] = rmse(truth, pred)
        out[f"nrmse@{H}"] = float(out[f"rmse@{H}"] / scale_ref)
        out[f"r2@{H}"] = r2_score(truth, pred)
        out[f"mae@{H}"] = mae(truth, pred)
    return out


def evaluate_distribution(model, y_context: np.ndarray, y_future: np.ndarray, stat_horizon: int) -> Dict[str, float]:
    H = min(int(stat_horizon), len(y_future))
    pred = np.asarray(model.rollout(y_context, horizon=H), dtype=float)
    truth = np.asarray(y_future[:H], dtype=float)
    if not np.isfinite(pred).all():
        return {"mean_gap": float("inf"), "std_gap": float("inf"), "acf_rmse": float("inf"), "psd_rmse": float("inf")}
    return distribution_metrics(truth, pred)
