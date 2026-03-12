from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class FastSlowConfig:
    """
    Continuous-time parameterization for the causal fast/slow encoder.

    Backward compatibility:
    - legacy integer EMA windows can still be passed via ``t0`` and ``slow_scales``;
      they are converted to continuous-time constants using the exact EMA-alpha match.
    - the slow branch is parameterized by a single base scale ``n`` and expanded as
      ``(n, 2n, 4n, ...)``, matching the requested geometric design.
    """

    t0: float | None = None
    slow_scales: Tuple[float, ...] | None = None
    dt: float = 1.0
    fast_tau: float | None = None
    slow_base_tau: float | None = None
    slow_levels: int = 3
    slow_ratio: float = 2.0

    @staticmethod
    def window_to_tau(window: float, dt: float = 1.0) -> float:
        window = float(window)
        dt = float(dt)
        if window <= 1.0:
            return max(0.5 * dt, 1e-8)
        alpha = 2.0 / (window + 1.0)
        alpha = min(max(alpha, 1e-8), 1.0 - 1e-8)
        return float(-dt / np.log(1.0 - alpha))

    @staticmethod
    def tau_to_alpha(tau: float, dt: float = 1.0) -> float:
        tau = max(float(tau), 1e-8)
        dt = float(dt)
        alpha = 1.0 - np.exp(-dt / tau)
        return float(min(max(alpha, 1e-8), 1.0 - 1e-8))

    @staticmethod
    def _infer_ratio(scales: Tuple[float, ...] | None, default: float) -> float:
        if scales is None or len(scales) < 2:
            return float(default)
        base = float(scales[0])
        nxt = float(scales[1])
        if abs(base) < 1e-12:
            return float(default)
        ratio = nxt / base
        return float(ratio) if ratio > 1.0 else float(default)

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.slow_levels < 1:
            raise ValueError("slow_levels must be >= 1")
        if self.slow_ratio <= 1.0:
            raise ValueError("slow_ratio must be > 1")

        if self.fast_tau is None:
            base_fast = 4.0 if self.t0 is None else float(self.t0)
            object.__setattr__(self, "fast_tau", float(self.window_to_tau(base_fast, dt=self.dt)))

        if self.slow_base_tau is None:
            legacy_scales = self.slow_scales if self.slow_scales is not None else (8.0, 16.0, 32.0)
            object.__setattr__(self, "slow_base_tau", float(self.window_to_tau(float(legacy_scales[0]), dt=self.dt)))
            object.__setattr__(self, "slow_levels", int(len(legacy_scales)))
            object.__setattr__(self, "slow_ratio", self._infer_ratio(tuple(float(v) for v in legacy_scales), self.slow_ratio))

    @property
    def slow_taus(self) -> Tuple[float, ...]:
        return tuple(float(self.slow_base_tau) * (float(self.slow_ratio) ** i) for i in range(int(self.slow_levels)))

    @property
    def fast_alpha(self) -> float:
        return self.tau_to_alpha(float(self.fast_tau), dt=self.dt)

    @property
    def slow_alphas(self) -> Tuple[float, ...]:
        return tuple(self.tau_to_alpha(tau, dt=self.dt) for tau in self.slow_taus)

    @property
    def fast_step_equivalent(self) -> float:
        return float(self.fast_tau / self.dt)

    @property
    def slow_step_equivalents(self) -> Tuple[float, ...]:
        return tuple(float(tau / self.dt) for tau in self.slow_taus)


class CausalFastSlowEncoder:
    def __init__(self, cfg: FastSlowConfig):
        self.cfg = cfg

    @staticmethod
    def _ema_update(prev: float, x: float, alpha: float) -> float:
        return (1.0 - alpha) * prev + alpha * x

    def build_feature_sequence(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        y = np.asarray(y, dtype=float).reshape(-1)
        n = len(y)
        fast_alpha = self.cfg.fast_alpha
        slow_alphas = self.cfg.slow_alphas
        f1 = np.zeros(n, dtype=float)
        f2 = np.zeros(n, dtype=float)
        slows = np.zeros((n, len(slow_alphas)), dtype=float)
        f1[0] = y[0]
        f2[0] = y[0]
        slows[0, :] = y[0]
        for t in range(1, n):
            f1[t] = self._ema_update(f1[t - 1], y[t], fast_alpha)
            f2[t] = self._ema_update(f2[t - 1], f1[t], fast_alpha)
            for j, alpha in enumerate(slow_alphas):
                slows[t, j] = self._ema_update(slows[t - 1, j], y[t], alpha)
        slow = np.mean(slows, axis=1)
        fast = f2
        m = fast - slow
        resid = y - slow
        ds = np.concatenate([[0.0], np.diff(slow)])
        dfast = np.concatenate([[0.0], np.diff(fast)])
        return {"y": y, "fast": fast, "slow": slow, "m": m, "resid": resid, "ds": ds, "dfast": dfast}

    def init_full_state(self, y_hist: np.ndarray) -> Dict[str, Any]:
        y_hist = np.asarray(y_hist, dtype=float).reshape(-1)
        fast_alpha = self.cfg.fast_alpha
        slow_alphas = self.cfg.slow_alphas
        f1 = float(y_hist[0])
        f2 = float(y_hist[0])
        slows = np.array([y_hist[0]] * len(slow_alphas), dtype=float)
        for val in y_hist[1:]:
            f1 = self._ema_update(f1, float(val), fast_alpha)
            f2 = self._ema_update(f2, f1, fast_alpha)
            for j, alpha in enumerate(slow_alphas):
                slows[j] = self._ema_update(float(slows[j]), float(val), alpha)
        return {"f1": f1, "f2": f2, "slows": slows}

    def init_fast_state(self, y_hist: np.ndarray) -> Dict[str, float]:
        y_hist = np.asarray(y_hist, dtype=float).reshape(-1)
        fast_alpha = self.cfg.fast_alpha
        f1 = float(y_hist[0])
        f2 = float(y_hist[0])
        for val in y_hist[1:]:
            f1 = self._ema_update(f1, float(val), fast_alpha)
            f2 = self._ema_update(f2, f1, fast_alpha)
        return {"f1": f1, "f2": f2}

    def advance_full(self, state: Dict[str, Any], y_new: float) -> Dict[str, Any]:
        fast_alpha = self.cfg.fast_alpha
        f1_new = self._ema_update(float(state["f1"]), float(y_new), fast_alpha)
        f2_new = self._ema_update(float(state["f2"]), float(f1_new), fast_alpha)
        slows_new = np.asarray(state["slows"], dtype=float).copy()
        for j, alpha in enumerate(self.cfg.slow_alphas):
            slows_new[j] = self._ema_update(float(slows_new[j]), float(y_new), alpha)
        state["f1"] = float(f1_new)
        state["f2"] = float(f2_new)
        state["slows"] = slows_new
        return state

    def advance_fast(self, state: Dict[str, float], y_new: float) -> Dict[str, float]:
        fast_alpha = self.cfg.fast_alpha
        f1_new = self._ema_update(float(state["f1"]), float(y_new), fast_alpha)
        f2_new = self._ema_update(float(state["f2"]), float(f1_new), fast_alpha)
        state["f1"] = float(f1_new)
        state["f2"] = float(f2_new)
        return state
