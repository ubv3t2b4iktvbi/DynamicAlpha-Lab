from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

from ..metrics import autocorr, psd_norm, rmse
from .base import FactorSpec


def _clip01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def _flattened_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(rmse(np.asarray(y_true, dtype=float).reshape(-1), np.asarray(y_pred, dtype=float).reshape(-1)))


def _fit_transition(inputs: np.ndarray, targets: np.ndarray, degree: int = 2, alpha: float = 1e-4) -> tuple[PolynomialFeatures, Ridge]:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X = poly.fit_transform(np.asarray(inputs, dtype=float))
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X, np.asarray(targets, dtype=float))
    return poly, model


def _predict_transition(poly: PolynomialFeatures, model: Ridge, inputs: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(poly.transform(np.asarray(inputs, dtype=float))), dtype=float)


@dataclass(frozen=True)
class SignalPropertyProfile:
    oscillatory_score: float
    multiscale_score: float
    trend_score: float
    burstiness_score: float
    unpredictability_score: float
    closure_need_score: float
    dominant_frequency_power: float
    lag1_autocorr: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def analyze_signal_properties(y: np.ndarray) -> SignalPropertyProfile:
    y = np.asarray(y, dtype=float).reshape(-1)
    if len(y) < 16:
        raise ValueError("Need at least 16 samples to analyze signal properties")
    y_std = (y - np.mean(y)) / (np.std(y) + 1e-12)
    dy = np.diff(y_std)

    ac = autocorr(y_std, max_lag=min(16, len(y_std) - 2))
    lag1 = float(ac[1]) if len(ac) > 1 else 0.0
    spec = psd_norm(y_std)
    dominant_power = float(np.max(spec[1:])) if len(spec) > 1 else 0.0

    centered = y_std - np.mean(y_std)
    sign_changes = np.mean(np.sign(centered[1:]) != np.sign(centered[:-1])) if len(centered) > 1 else 0.0
    oscillatory_score = _clip01(0.65 * dominant_power * max(len(spec) - 1, 1) + 0.35 * sign_changes * 4.0)

    short_scale = float(np.std(dy[: max(8, len(dy) // 8)])) if len(dy) > 8 else float(np.std(dy))
    long_window = max(8, len(y_std) // 12)
    kernel = np.ones(long_window, dtype=float) / long_window
    smooth = np.convolve(y_std, kernel, mode="same")
    residual = y_std - smooth
    multiscale_score = _clip01(np.std(smooth) / (np.std(residual) + np.std(smooth) + 1e-12) + np.std(residual) / (short_scale + np.std(residual) + 1e-12))

    trend_score = _clip01(abs(np.mean(ac[1 : min(len(ac), 5)])) if len(ac) > 2 else abs(lag1))

    burstiness = float(np.mean(np.abs(dy) > 2.0 * (np.std(dy) + 1e-12))) if len(dy) else 0.0
    energy_skew = float(np.mean(np.maximum(np.abs(dy) - np.std(dy), 0.0))) if len(dy) else 0.0
    burstiness_score = _clip01(0.7 * burstiness * 4.0 + 0.3 * energy_skew)

    raw_inputs = y_std[:-1].reshape(-1, 1)
    raw_targets = y_std[1:].reshape(-1, 1)
    raw_poly, raw_model = _fit_transition(raw_inputs[:-1], raw_targets[:-1])
    raw_pred = _predict_transition(raw_poly, raw_model, raw_inputs[-min(len(raw_inputs) - 1, 256) :])
    raw_rmse = _flattened_rmse(raw_targets[-len(raw_pred) :], raw_pred)
    unpredictability_score = _clip01(raw_rmse / (np.std(y_std) + 1e-12))

    lag_inputs = np.column_stack([y_std[1:-1], y_std[:-2]])
    lag_targets = y_std[2:].reshape(-1, 1)
    lag_poly, lag_model = _fit_transition(lag_inputs[:-1], lag_targets[:-1])
    lag_pred = _predict_transition(lag_poly, lag_model, lag_inputs[-min(len(lag_inputs) - 1, 256) :])
    lag_rmse = _flattened_rmse(lag_targets[-len(lag_pred) :], lag_pred)
    closure_need_score = _clip01(max(raw_rmse - lag_rmse, 0.0) / (raw_rmse + 1e-12))

    return SignalPropertyProfile(
        oscillatory_score=oscillatory_score,
        multiscale_score=multiscale_score,
        trend_score=trend_score,
        burstiness_score=burstiness_score,
        unpredictability_score=unpredictability_score,
        closure_need_score=closure_need_score,
        dominant_frequency_power=dominant_power,
        lag1_autocorr=lag1,
    )


def _tag_weight(tag: str, profile: SignalPropertyProfile) -> float:
    mapping = {
        "slow_fast": 0.6 * profile.multiscale_score + 0.4 * profile.trend_score,
        "order_parameter": 0.7 * profile.multiscale_score + 0.3 * profile.trend_score,
        "phase": profile.oscillatory_score,
        "oscillation": profile.oscillatory_score,
        "energy": profile.burstiness_score,
        "control_parameter": 0.7 * profile.burstiness_score + 0.3 * profile.unpredictability_score,
        "criticality": 0.5 * profile.multiscale_score + 0.5 * profile.unpredictability_score,
        "susceptibility": 0.4 * profile.multiscale_score + 0.6 * profile.burstiness_score,
        "multiscale": profile.multiscale_score,
        "rg": profile.multiscale_score,
        "compression": profile.multiscale_score,
        "support": 0.5 * profile.oscillatory_score + 0.5 * profile.trend_score,
        "breakout": 0.4 * profile.trend_score + 0.6 * profile.burstiness_score,
        "trend": profile.trend_score,
        "slow_manifold": 0.5 * profile.multiscale_score + 0.5 * profile.trend_score,
        "physics_identifier": 0.6 * profile.unpredictability_score + 0.4 * profile.closure_need_score,
        "drift": 0.7 * profile.trend_score + 0.3 * profile.multiscale_score,
        "changepoint": 0.5 * profile.burstiness_score + 0.5 * profile.unpredictability_score,
        "consistency": 0.5 * profile.trend_score + 0.5 * (1.0 - profile.unpredictability_score),
        "activation": 0.6 * profile.burstiness_score + 0.4 * profile.oscillatory_score,
        "recovery": 0.6 * profile.oscillatory_score + 0.4 * profile.closure_need_score,
        "curvature": 0.5 * profile.oscillatory_score + 0.5 * profile.multiscale_score,
    }
    return float(mapping.get(tag, 0.5 * profile.trend_score + 0.5 * profile.unpredictability_score))


def _family_weight(family: str, profile: SignalPropertyProfile) -> float:
    mapping = {
        "order_parameter": 0.65 * profile.multiscale_score + 0.35 * profile.trend_score,
        "phase": profile.oscillatory_score,
        "energy": profile.burstiness_score,
        "criticality": 0.5 * profile.multiscale_score + 0.5 * profile.unpredictability_score,
        "multiscale": profile.multiscale_score,
        "event": 0.5 * profile.burstiness_score + 0.5 * profile.oscillatory_score,
        "slow_manifold": 0.5 * profile.multiscale_score + 0.5 * profile.trend_score,
        "physics_id": 0.6 * profile.unpredictability_score + 0.4 * profile.closure_need_score,
        "composite": 0.4 * profile.multiscale_score + 0.3 * profile.burstiness_score + 0.3 * profile.unpredictability_score,
    }
    return float(mapping.get(family, 0.5))


def factor_prior_weight(spec: FactorSpec, profile: SignalPropertyProfile) -> float:
    tag_weights = [_tag_weight(tag, profile) for tag in spec.theory_tags] or [0.5]
    family_weight = _family_weight(spec.family, profile)
    complexity_penalty = 0.06 * max(spec.complexity - 1, 0)
    score = 0.55 * float(np.mean(tag_weights)) + 0.45 * family_weight - complexity_penalty
    return _clip01(score)


def scalar_koopman_diagnostic(train_values: np.ndarray, val_values: np.ndarray) -> dict[str, float]:
    train_values = np.asarray(train_values, dtype=float).reshape(-1)
    val_values = np.asarray(val_values, dtype=float).reshape(-1)
    if len(train_values) < 4 or len(val_values) < 4:
        return {
            "koopman_lambda": 0.0,
            "koopman_rmse": float("inf"),
            "koopman_score": 0.0,
        }
    x_train = train_values[:-1]
    y_train = train_values[1:]
    lam = float(np.dot(x_train, y_train) / (np.dot(x_train, x_train) + 1e-12))
    pred = lam * val_values[:-1]
    truth = val_values[1:]
    err = float(rmse(truth, pred))
    scale = float(np.std(truth) + 1e-12)
    corr = float(np.corrcoef(pred, truth)[0, 1]) if len(pred) > 1 and np.std(pred) > 1e-12 and np.std(truth) > 1e-12 else 0.0
    score = _clip01(0.7 * (1.0 - err / scale) + 0.3 * abs(corr))
    return {
        "koopman_lambda": lam,
        "koopman_rmse": err,
        "koopman_score": score,
    }


def prioritize_factor_bank(
    specs: Sequence[FactorSpec],
    profile: SignalPropertyProfile,
    mode: str,
    full_library_search: bool,
    prescreen_top_k: int,
) -> tuple[list[FactorSpec], dict[str, float]]:
    if mode not in {"accumulate", "identify"}:
        raise ValueError(f"Unsupported factor-mining mode: {mode}")
    prior_map = {spec.name: factor_prior_weight(spec, profile) for spec in specs}
    ordered = sorted(specs, key=lambda spec: (-prior_map[spec.name], spec.complexity, spec.name))
    if mode == "identify" and not full_library_search:
        keep = max(1, min(int(prescreen_top_k), len(ordered)))
        return ordered[:keep], prior_map
    return ordered, prior_map


def render_property_summary(profile: SignalPropertyProfile, mode: str, screened_count: int, total_count: int) -> str:
    top_axes = sorted(
        [
            ("oscillatory", profile.oscillatory_score),
            ("multiscale", profile.multiscale_score),
            ("trend", profile.trend_score),
            ("burstiness", profile.burstiness_score),
            ("unpredictability", profile.unpredictability_score),
            ("closure_need", profile.closure_need_score),
        ],
        key=lambda item: item[1],
        reverse=True,
    )
    axes_text = ", ".join(f"{name}={value:.3f}" for name, value in top_axes[:3])
    if mode == "identify":
        mode_text = "unknown-system identification mode"
    else:
        mode_text = "factor accumulation mode"
    return f"{mode_text}; dominant properties: {axes_text}; screened {screened_count} of {total_count} factors"
