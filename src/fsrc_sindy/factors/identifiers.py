from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Protocol

import numpy as np

from ..fastslow import FastSlowConfig
from ..library import build_poly_library, fit_stlsq
from .base import DynamicsFeatureConfig

try:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer, StandardScaler
except Exception:  # pragma: no cover - optional dependency fallback
    Ridge = None
    make_pipeline = None
    SplineTransformer = None
    StandardScaler = None


class PhysicsIdentifier(Protocol):
    kind: str

    def fit(self, features: Mapping[str, np.ndarray]) -> "PhysicsIdentifier":
        ...

    def batch_outputs(self, features: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        ...

    def step_outputs(self, ctx: Mapping[str, float]) -> Dict[str, float]:
        ...


@dataclass(frozen=True)
class IdentifierConfig:
    poly_order: int = 2
    threshold: float = 1e-4
    ridge: float = 1e-6
    spline_knots: int = 5
    spline_alpha: float = 1e-3


class BaseIdentifier:
    kind = "base"

    def fit(self, features: Mapping[str, np.ndarray]) -> "BaseIdentifier":
        return self

    def batch_outputs(self, features: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        zeros = np.zeros_like(np.asarray(features["slow"], dtype=float))
        return self._pack(zeros, np.asarray(features["slow_drift"], dtype=float), np.asarray(features["scale"], dtype=float))

    def step_outputs(self, ctx: Mapping[str, float]) -> Dict[str, float]:
        return self._pack_step(0.0, float(ctx["slow_drift"]), float(ctx["scale"]))

    @staticmethod
    def _pack(pred_delta: np.ndarray, slow_drift: np.ndarray, scale: np.ndarray) -> Dict[str, np.ndarray]:
        pred_delta = np.asarray(pred_delta, dtype=float)
        slow_drift = np.asarray(slow_drift, dtype=float)
        scale = np.asarray(scale, dtype=float)
        scale = np.maximum(scale, 1e-6)
        surprise = slow_drift - pred_delta
        alignment = np.sign(pred_delta * slow_drift)
        return {
            "id_drift_pred": pred_delta,
            "id_drift_pred_norm": pred_delta / scale,
            "id_drift_surprise": surprise,
            "id_drift_surprise_norm": surprise / scale,
            "id_drift_alignment": alignment,
        }

    @staticmethod
    def _pack_step(pred_delta: float, slow_drift: float, scale: float) -> Dict[str, float]:
        scale = max(float(scale), 1e-6)
        surprise = float(slow_drift - pred_delta)
        alignment = float(np.sign(pred_delta * slow_drift))
        return {
            "id_drift_pred": float(pred_delta),
            "id_drift_pred_norm": float(pred_delta / scale),
            "id_drift_surprise": surprise,
            "id_drift_surprise_norm": float(surprise / scale),
            "id_drift_alignment": alignment,
        }


class NoPhysicsIdentifier(BaseIdentifier):
    kind = "none"


class SlowSINDyIdentifier(BaseIdentifier):
    kind = "sindy_slow"

    def __init__(self, feature_cfg: DynamicsFeatureConfig, cfg: IdentifierConfig | None = None):
        self.feature_cfg = feature_cfg
        self.cfg = cfg if cfg is not None else IdentifierConfig()
        self.coef_: np.ndarray | None = None
        self.lib_names_: list[str] = []
        self.fs_cfg = FastSlowConfig(t0=feature_cfg.fast_window, slow_scales=feature_cfg.slow_windows)

    def fit(self, features: Mapping[str, np.ndarray]) -> "SlowSINDyIdentifier":
        slow = np.asarray(features["slow"], dtype=float)
        ds = np.asarray(features["slow_drift"], dtype=float)
        m = np.asarray(features["m_norm"], dtype=float)
        X = np.column_stack([slow[:-1], ds[:-1], m[:-1]])
        target = slow[1:] - slow[:-1]
        Theta, names = build_poly_library(X, ["slow", "ds", "m_norm"], poly_order=self.cfg.poly_order)
        self.coef_ = fit_stlsq(Theta, target, ridge=self.cfg.ridge, threshold=self.cfg.threshold)
        self.lib_names_ = names
        return self

    def _predict_delta_array(self, X: np.ndarray) -> np.ndarray:
        Theta, _ = build_poly_library(X, ["slow", "ds", "m_norm"], poly_order=self.cfg.poly_order)
        pred = Theta @ self.coef_
        return np.asarray(pred, dtype=float)

    def _predict_delta_step(self, slow: float, slow_drift: float, m_norm: float) -> float:
        X = np.array([[slow, slow_drift, m_norm]], dtype=float)
        return float(self._predict_delta_array(X)[0])

    def batch_outputs(self, features: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        X = np.column_stack([
            np.asarray(features["slow"], dtype=float),
            np.asarray(features["slow_drift"], dtype=float),
            np.asarray(features["m_norm"], dtype=float),
        ])
        pred = self._predict_delta_array(X)
        return self._pack(pred, np.asarray(features["slow_drift"], dtype=float), np.asarray(features["scale"], dtype=float))

    def step_outputs(self, ctx: Mapping[str, float]) -> Dict[str, float]:
        pred = self._predict_delta_step(float(ctx["slow"]), float(ctx["slow_drift"]), float(ctx["m_norm"]))
        return self._pack_step(pred, float(ctx["slow_drift"]), float(ctx["scale"]))


class SplineKANLikeIdentifier(BaseIdentifier):
    """
    A light-weight, dependency-friendly KAN-like identifier.

    It is not a full pykan implementation. Instead it uses spline basis functions
    with a linear head, which preserves the “edge function / learned spline” flavor
    while remaining fast enough for broad RC screening.
    """

    kind = "spline_kan_like"

    def __init__(self, cfg: IdentifierConfig | None = None):
        self.cfg = cfg if cfg is not None else IdentifierConfig()
        self.model = None
        if SplineTransformer is not None and make_pipeline is not None and Ridge is not None:
            self.model = make_pipeline(
                StandardScaler(with_mean=True, with_std=True),
                SplineTransformer(n_knots=self.cfg.spline_knots, degree=3, include_bias=False),
                Ridge(alpha=self.cfg.spline_alpha),
            )
        self.linear_coef_: np.ndarray | None = None

    @staticmethod
    def _design(features: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.column_stack([
            np.asarray(features["slow"], dtype=float),
            np.asarray(features["slow_drift"], dtype=float),
            np.asarray(features["m_norm"], dtype=float),
            np.asarray(features["dm_norm"], dtype=float),
            np.asarray(features["critical_window"], dtype=float),
        ])

    def fit(self, features: Mapping[str, np.ndarray]) -> "SplineKANLikeIdentifier":
        X = self._design(features)
        target = np.asarray(features["slow_drift"], dtype=float)
        target = np.concatenate([target[1:], target[-1:]], axis=0)
        if self.model is not None:
            self.model.fit(X, target)
            return self
        # fallback: simple linear projection if sklearn is unavailable
        X_aug = np.column_stack([X, np.ones(len(X), dtype=float)])
        lam = max(self.cfg.spline_alpha, 1e-8)
        A = X_aug.T @ X_aug + lam * np.eye(X_aug.shape[1])
        b = X_aug.T @ target
        self.linear_coef_ = np.linalg.solve(A, b)
        return self

    def _predict_array(self, X: np.ndarray) -> np.ndarray:
        if self.model is not None:
            return np.asarray(self.model.predict(X), dtype=float)
        X_aug = np.column_stack([X, np.ones(len(X), dtype=float)])
        return np.asarray(X_aug @ self.linear_coef_, dtype=float)

    def batch_outputs(self, features: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        X = self._design(features)
        pred = self._predict_array(X)
        return self._pack(pred, np.asarray(features["slow_drift"], dtype=float), np.asarray(features["scale"], dtype=float))

    def step_outputs(self, ctx: Mapping[str, float]) -> Dict[str, float]:
        X = np.array([[ctx["slow"], ctx["slow_drift"], ctx["m_norm"], ctx["dm_norm"], ctx["critical_window"]]], dtype=float)
        pred = float(self._predict_array(X)[0])
        return self._pack_step(pred, float(ctx["slow_drift"]), float(ctx["scale"]))


IDENTIFIER_REGISTRY = {
    "none": NoPhysicsIdentifier,
    "sindy_slow": SlowSINDyIdentifier,
    "spline_kan_like": SplineKANLikeIdentifier,
}


def make_identifier(kind: str, feature_cfg: DynamicsFeatureConfig) -> BaseIdentifier:
    kind = str(kind)
    if kind not in IDENTIFIER_REGISTRY:
        raise ValueError(f"Unknown identifier kind: {kind}")
    cls = IDENTIFIER_REGISTRY[kind]
    if kind == "sindy_slow":
        return cls(feature_cfg=feature_cfg)
    return cls()
