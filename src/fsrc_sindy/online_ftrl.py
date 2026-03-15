from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def safe_sigmoid(value: float | np.ndarray) -> float | np.ndarray:
    arr = np.clip(value, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-arr))


def safe_logit(prob: float | np.ndarray) -> float | np.ndarray:
    arr = np.clip(prob, 1e-6, 1.0 - 1e-6)
    return np.log(arr / (1.0 - arr))


@dataclass(frozen=True)
class FTRLConfig:
    alpha: float = 0.12
    beta: float = 1.0
    l1: float = 0.0
    l2: float = 0.20


class OnlineFTRLBinary:
    """Dense FTRL-Proximal binary learner for causal walk-forward updates."""

    def __init__(self, n_features: int, cfg: FTRLConfig | None = None):
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        self.n_features = int(n_features)
        self.cfg = cfg or FTRLConfig()
        self.z = np.zeros(self.n_features, dtype=float)
        self.n = np.zeros(self.n_features, dtype=float)

    def _weights(self) -> np.ndarray:
        z = self.z
        n = self.n
        cfg = self.cfg
        w = np.zeros_like(z)
        mask = np.abs(z) > cfg.l1
        if np.any(mask):
            w[mask] = -(
                z[mask] - np.sign(z[mask]) * cfg.l1
            ) / (((cfg.beta + np.sqrt(n[mask])) / cfg.alpha) + cfg.l2)
        return w

    def weights(self) -> np.ndarray:
        return np.asarray(self._weights(), dtype=float)

    def predict_raw(self, x: np.ndarray) -> float:
        x_use = np.asarray(x, dtype=float).reshape(-1)
        if x_use.shape[0] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {x_use.shape[0]}")
        return float(np.dot(self._weights(), x_use))

    def predict_proba(self, x: np.ndarray) -> float:
        return float(safe_sigmoid(self.predict_raw(x)))

    def update(self, x: np.ndarray, y: float, sample_weight: float = 1.0) -> float:
        x_use = np.asarray(x, dtype=float).reshape(-1)
        if x_use.shape[0] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {x_use.shape[0]}")
        y_use = float(np.clip(y, 0.0, 1.0))
        weight = float(max(sample_weight, 0.0))
        p = self.predict_proba(x_use)
        g = (p - y_use) * x_use * weight
        sigma = (np.sqrt(self.n + g * g) - np.sqrt(self.n)) / self.cfg.alpha
        w = self._weights()
        self.z += g - sigma * w
        self.n += g * g
        return p


def blend_probabilities(*terms: tuple[float, float]) -> float:
    if not terms:
        return 0.5
    total_weight = 0.0
    logit_sum = 0.0
    for prob, weight in terms:
        w = float(max(weight, 0.0))
        if w <= 0.0:
            continue
        logit_sum += w * float(safe_logit(float(prob)))
        total_weight += w
    if total_weight <= 0.0:
        return 0.5
    return float(safe_sigmoid(logit_sum / total_weight))
