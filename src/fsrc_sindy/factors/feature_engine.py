from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

import numpy as np

from .base import DynamicsFeatureConfig


@dataclass
class _State:
    f1: float
    fast: float
    slows: np.ndarray
    scale: float
    scale_long: float
    energy: float
    energy_long: float
    susceptibility: float
    susceptibility_long: float
    positive_impulse: float
    negative_impulse: float
    band_high: float
    band_low: float
    prev_y: float
    prev2_y: float
    prev_fast: float
    prev_slow: float
    prev_m: float
    prev_dm: float
    time_index: int = 0


class DynamicsFeatureEngine:
    """
    Causal feature engine that translates finance-inspired moving-average logic
    into dynamical quantities: order parameter, phase evidence, energy injection,
    and multiscale collapse quality.

    All quantities are computed causally so that the same formulas can be used
    during teacher-forced scoring and autoregressive rollout.
    """

    def __init__(self, cfg: DynamicsFeatureConfig):
        self.cfg = cfg
        self.fast_alpha = self._window_to_alpha(cfg.fast_window)
        self.slow_alphas = np.asarray([self._window_to_alpha(win) for win in cfg.slow_windows], dtype=float)
        self.scale_alpha = self._window_to_alpha(cfg.scale_window)
        self.scale_long_alpha = self._window_to_alpha(cfg.scale_long_window)
        self.energy_alpha = self._window_to_alpha(cfg.energy_window)
        self.energy_long_alpha = self._window_to_alpha(cfg.energy_long_window)
        self.sus_alpha = self._window_to_alpha(cfg.susceptibility_window)
        self.sus_long_alpha = self._window_to_alpha(cfg.susceptibility_long_window)

    @staticmethod
    def _window_to_alpha(window: float) -> float:
        window = float(window)
        if window <= 1.0:
            return 1.0
        return float(min(max(2.0 / (window + 1.0), 1e-6), 1.0))

    @staticmethod
    def _ema(prev: float, x: float, alpha: float) -> float:
        return float((1.0 - alpha) * prev + alpha * x)

    @staticmethod
    def _pairwise_mean_abs(values: np.ndarray) -> float:
        if len(values) <= 1:
            return 0.0
        diffs = []
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                diffs.append(abs(float(values[i] - values[j])))
        return float(np.mean(diffs)) if diffs else 0.0

    @staticmethod
    def _relu(x: float) -> float:
        return float(max(0.0, x))

    def init_state(self, y0: float) -> _State:
        y0 = float(np.clip(y0, -self.cfg.clip, self.cfg.clip))
        slows = np.full(len(self.slow_alphas), y0, dtype=float)
        return _State(
            f1=y0,
            fast=y0,
            slows=slows,
            scale=abs(y0) + 1e-3,
            scale_long=abs(y0) + 1e-3,
            energy=0.0,
            energy_long=0.0,
            susceptibility=0.0,
            susceptibility_long=0.0,
            positive_impulse=0.0,
            negative_impulse=0.0,
            band_high=y0,
            band_low=y0,
            prev_y=y0,
            prev2_y=y0,
            prev_fast=y0,
            prev_slow=y0,
            prev_m=0.0,
            prev_dm=0.0,
            time_index=0,
        )

    def _context_from_state(self, state: _State) -> dict[str, float]:
        slow = float(np.mean(state.slows))
        fast = float(state.fast)
        y = float(state.prev_y)
        dy = float(y - state.prev2_y)
        slow_drift = float(slow - state.prev_slow)
        fast_drift = float(fast - state.prev_fast)
        m = float(fast - slow)
        dm = float(m - state.prev_m)
        d2m = float(dm - state.prev_dm)
        resid = float(y - slow)
        scale = float(max(state.scale, self.cfg.eps))
        scale_long = float(max(state.scale_long, self.cfg.eps))
        m_norm = float(m / scale)
        dm_norm = float(dm / scale)
        d2m_norm = float(d2m / scale)
        resid_norm = float(resid / scale)
        slow_drift_norm = float(slow_drift / scale)
        fast_drift_norm = float(fast_drift / scale)
        energy_ratio = float(state.energy / max(state.energy_long, self.cfg.eps)) if state.energy_long > 0 else 1.0
        susceptibility_ratio = float(state.susceptibility / max(state.susceptibility_long, self.cfg.eps)) if state.susceptibility_long > 0 else 1.0
        critical_proximity = float(1.0 / (1.0 + abs(m_norm)))
        critical_window = float(np.clip(critical_proximity * susceptibility_ratio * energy_ratio / self.cfg.critical_shrink, 0.0, 1.0))
        M_vals = np.asarray([(y - slow_i) / scale for slow_i in state.slows], dtype=float)
        collapse_error = self._pairwise_mean_abs(M_vals)
        collapse_quality = float(1.0 / (1.0 + collapse_error))
        phase = float(np.arctan2(dm_norm, resid_norm + self.cfg.eps))
        phase_bottom_score = self._relu(-resid_norm) * self._relu(dm_norm)
        breakout_strength = self._relu(m_norm) * energy_ratio
        support_recovery = self._relu(m_norm) * phase_bottom_score
        compression_ratio = float(scale / max(scale_long, self.cfg.eps))
        energy_release = energy_ratio * self._relu(dm_norm)
        shock_recovery = self._relu(-dy / scale) * self._relu(dm_norm)
        trend_persistence = float(abs(slow_drift_norm) / max(abs(dm_norm) + self.cfg.eps, self.cfg.eps))
        slow_level_norm = float(slow / max(scale_long, self.cfg.eps))
        fast_level_norm = float(fast / max(scale, self.cfg.eps))
        timescale_ratio = float(abs(fast_drift_norm) / max(abs(slow_drift_norm), self.cfg.eps))
        timescale_separation = float(np.log1p(timescale_ratio))
        slow_manifold_alignment = float(1.0 / (1.0 + abs(resid_norm) + abs(dm_norm)))
        adiabatic_coherence = float(collapse_quality * slow_manifold_alignment)
        closure_stress = float(abs(dm_norm) * energy_ratio / max(collapse_quality, self.cfg.eps))
        total_impulse = float(max(state.positive_impulse + state.negative_impulse, self.cfg.eps))
        positive_impulse_share = float(state.positive_impulse / total_impulse)
        impulse_balance = float((state.positive_impulse - state.negative_impulse) / total_impulse)
        band_width = float(max(state.band_high - state.band_low, self.cfg.eps))
        band_position = float(np.clip((y - state.band_low) / band_width, 0.0, 1.0))
        trend_regression_quality = float(
            collapse_quality / (1.0 + abs(resid_norm) + abs(fast_drift_norm - slow_drift_norm))
        )
        return {
            "y": y,
            "dy": dy,
            "fast": fast,
            "slow": slow,
            "fast_drift": fast_drift,
            "slow_drift": slow_drift,
            "resid": resid,
            "m": m,
            "dm": dm,
            "d2m": d2m,
            "scale": scale,
            "scale_long": scale_long,
            "m_norm": m_norm,
            "dm_norm": dm_norm,
            "d2m_norm": d2m_norm,
            "resid_norm": resid_norm,
            "slow_drift_norm": slow_drift_norm,
            "fast_drift_norm": fast_drift_norm,
            "energy": float(state.energy),
            "energy_ratio": energy_ratio,
            "susceptibility": float(state.susceptibility),
            "susceptibility_ratio": susceptibility_ratio,
            "critical_proximity": critical_proximity,
            "critical_window": critical_window,
            "collapse_error": collapse_error,
            "collapse_quality": collapse_quality,
            "phase": phase,
            "phase_sin": float(np.sin(phase)),
            "phase_cos": float(np.cos(phase)),
            "phase_bottom_score": phase_bottom_score,
            "breakout_strength": breakout_strength,
            "support_recovery": support_recovery,
            "compression_ratio": compression_ratio,
            "energy_release": energy_release,
            "shock_recovery": shock_recovery,
            "trend_persistence": trend_persistence,
            "slow_level_norm": slow_level_norm,
            "fast_level_norm": fast_level_norm,
            "timescale_separation": timescale_separation,
            "slow_manifold_alignment": slow_manifold_alignment,
            "adiabatic_coherence": adiabatic_coherence,
            "closure_stress": closure_stress,
            "positive_impulse_share": positive_impulse_share,
            "impulse_balance": impulse_balance,
            "band_position": band_position,
            "trend_regression_quality": trend_regression_quality,
            "time_index": float(state.time_index),
        }

    def step(self, state: _State, y_new: float) -> dict[str, float]:
        y_new = float(np.clip(y_new, -self.cfg.clip, self.cfg.clip))
        prev_fast = float(state.fast)
        prev_slow = float(np.mean(state.slows))
        prev_m = float(prev_fast - prev_slow)
        prev_dm = float(prev_m - state.prev_m)

        f1 = self._ema(state.f1, y_new, self.fast_alpha)
        fast = self._ema(state.fast, f1, self.fast_alpha)
        slows = state.slows.copy()
        for j, alpha in enumerate(self.slow_alphas):
            slows[j] = self._ema(float(slows[j]), y_new, float(alpha))
        slow = float(np.mean(slows))
        resid = float(y_new - slow)
        dy = float(y_new - state.prev_y)
        scale = self._ema(state.scale, abs(resid) + self.cfg.eps, self.scale_alpha)
        scale_long = self._ema(state.scale_long, abs(resid) + self.cfg.eps, self.scale_long_alpha)
        energy = self._ema(state.energy, dy * dy, self.energy_alpha)
        energy_long = self._ema(state.energy_long, dy * dy, self.energy_long_alpha)
        positive_impulse = self._ema(state.positive_impulse, self._relu(dy), self.energy_alpha)
        negative_impulse = self._ema(state.negative_impulse, self._relu(-dy), self.energy_alpha)
        band_high = max(y_new, self._ema(state.band_high, y_new, self.fast_alpha))
        band_low = min(y_new, self._ema(state.band_low, y_new, self.fast_alpha))
        m = float(fast - slow)
        dm = float(m - prev_m)
        susceptibility = self._ema(state.susceptibility, abs(dm), self.sus_alpha)
        susceptibility_long = self._ema(state.susceptibility_long, abs(dm), self.sus_long_alpha)

        state.prev2_y = float(state.prev_y)
        state.prev_y = y_new
        state.prev_fast = prev_fast
        state.prev_slow = prev_slow
        state.prev_m = prev_m
        state.prev_dm = prev_dm
        state.f1 = f1
        state.fast = fast
        state.slows = slows
        state.scale = scale
        state.scale_long = scale_long
        state.energy = energy
        state.energy_long = energy_long
        state.susceptibility = susceptibility
        state.susceptibility_long = susceptibility_long
        state.positive_impulse = positive_impulse
        state.negative_impulse = negative_impulse
        state.band_high = band_high
        state.band_low = band_low
        state.time_index += 1
        return self._context_from_state(state)

    def build_base_sequence(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        y = np.asarray(y, dtype=float).reshape(-1)
        if len(y) == 0:
            raise ValueError("y must have at least one element")
        state = self.init_state(float(y[0]))
        contexts = [self._context_from_state(state)]
        for value in y[1:]:
            contexts.append(self.step(state, float(value)))
        keys = list(contexts[0].keys())
        out: Dict[str, np.ndarray] = {}
        for key in keys:
            out[key] = np.asarray([ctx[key] for ctx in contexts], dtype=float)
        return out

    def warmup_state(self, y_hist: Iterable[float]) -> tuple[_State, dict[str, float]]:
        y_hist = list(float(v) for v in y_hist)
        if not y_hist:
            raise ValueError("y_hist must not be empty")
        state = self.init_state(y_hist[0])
        ctx = self._context_from_state(state)
        for value in y_hist[1:]:
            ctx = self.step(state, value)
        return state, ctx

    def augment_with_identifier(
        self,
        base_context: Mapping[str, np.ndarray],
        identifier_outputs: Mapping[str, np.ndarray] | None,
    ) -> Dict[str, np.ndarray]:
        out = {key: np.asarray(val, dtype=float) for key, val in base_context.items()}
        if identifier_outputs is None:
            return out
        for key, val in identifier_outputs.items():
            out[key] = np.asarray(val, dtype=float)
        return out
