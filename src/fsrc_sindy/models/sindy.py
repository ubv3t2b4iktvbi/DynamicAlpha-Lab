from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from ..fastslow import CausalFastSlowEncoder, FastSlowConfig
from ..library import build_poly_library, fit_stlsq
from ..metrics import metric_dict
from .base import BaseForecastModel


@dataclass
class FullSINDyConfig:
    fs_cfg: Optional[FastSlowConfig] = None
    t0: int | None = None
    slow_scales: tuple[float, ...] | None = None
    poly_order: int = 2
    threshold: float = 1e-3
    ridge: float = 1e-6
    dy_clip: float = 2.0
    y_clip: float = 25.0


@dataclass
class SlowSINDyConfig:
    fs_cfg: FastSlowConfig
    poly_order: int = 2
    threshold: float = 1e-4
    ridge: float = 1e-6


class FullObservableSINDy(BaseForecastModel):
    def __init__(self, cfg: FullSINDyConfig):
        self.cfg = cfg
        self.mu_ = 0.0
        self.std_ = 1.0
        self.encoder = CausalFastSlowEncoder(cfg.fs_cfg if cfg.fs_cfg is not None else FastSlowConfig(t0=cfg.t0, slow_scales=cfg.slow_scales))
        self.coef_: Optional[np.ndarray] = None
        self.lib_names_: List[str] = []

    def _build_state_sequence(self, y_std: np.ndarray) -> dict[str, np.ndarray]:
        feats = self.encoder.build_feature_sequence(y_std)
        v = np.concatenate([[0.0], np.diff(feats["m"])])
        a = np.abs(v)
        q = np.log(a + 1e-6)
        return {"y": feats["y"], "b": feats["fast"], "db": feats["dfast"], "m": feats["m"], "v": v, "a": a, "q": q}

    def _base_matrix(self, state: dict[str, np.ndarray]) -> np.ndarray:
        return np.column_stack([state["y"], state["b"], state["db"], state["m"], state["v"], state["a"], state["q"]])

    def fit(self, y_train: np.ndarray) -> "FullObservableSINDy":
        self.mu_ = float(np.mean(y_train))
        self.std_ = float(np.std(y_train) + 1e-12)
        ys = np.clip((y_train - self.mu_) / self.std_, -self.cfg.y_clip, self.cfg.y_clip)
        state = self._build_state_sequence(ys)
        X = self._base_matrix(state)
        Theta, names = build_poly_library(X[:-1], ["y", "b", "db", "m", "v", "a", "q"], poly_order=self.cfg.poly_order)
        target = np.clip(ys[1:] - ys[:-1], -self.cfg.dy_clip, self.cfg.dy_clip)
        self.coef_ = fit_stlsq(Theta, target, ridge=self.cfg.ridge, threshold=self.cfg.threshold)
        self.lib_names_ = names
        return self

    def _predict_delta(self, x: np.ndarray) -> float:
        Theta, _ = build_poly_library(x.reshape(1, -1), ["y", "b", "db", "m", "v", "a", "q"], poly_order=self.cfg.poly_order)
        dy = float((Theta @ self.coef_).item())
        return float(np.clip(dy, -self.cfg.dy_clip, self.cfg.dy_clip))

    def rollout(self, y_hist: np.ndarray, horizon: int) -> np.ndarray:
        y_hist_std = np.clip((np.asarray(y_hist, dtype=float).reshape(-1) - self.mu_) / self.std_, -self.cfg.y_clip, self.cfg.y_clip)
        feats = self.encoder.build_feature_sequence(y_hist_std)
        y_cur = float(y_hist_std[-1])
        b_cur = float(feats["fast"][-1])
        db_cur = float(feats["dfast"][-1])
        m_cur = float(feats["m"][-1])
        v_cur = 0.0 if len(feats["m"]) < 2 else float(feats["m"][-1] - feats["m"][-2])
        full_state = self.encoder.init_full_state(y_hist_std)
        preds = []
        for _ in range(horizon):
            a_cur = abs(v_cur)
            q_cur = float(np.clip(np.log(a_cur + 1e-6), -10.0, 10.0))
            x = np.array([y_cur, b_cur, db_cur, m_cur, v_cur, a_cur, q_cur], dtype=float)
            dy = self._predict_delta(x)
            y_next = float(np.clip(y_cur + dy, -self.cfg.y_clip, self.cfg.y_clip))
            preds.append(y_next)
            self.encoder.advance_full(full_state, y_next)
            b_next = float(full_state["f2"])
            slow_next = float(np.mean(full_state["slows"]))
            m_next = float(b_next - slow_next)
            db_next = float(b_next - b_cur)
            v_next = float(m_next - m_cur)
            y_cur, b_cur, db_cur, m_cur, v_cur = y_next, b_next, db_next, m_next, v_next
        preds = np.asarray(preds, dtype=float)
        return preds * self.std_ + self.mu_

    def one_step_metrics(self, series: np.ndarray, burn_in: int):
        ys = np.clip((np.asarray(series, dtype=float).reshape(-1) - self.mu_) / self.std_, -self.cfg.y_clip, self.cfg.y_clip)
        state = self._build_state_sequence(ys)
        X = self._base_matrix(state)
        Theta, _ = build_poly_library(X[:-1], ["y", "b", "db", "m", "v", "a", "q"], poly_order=self.cfg.poly_order)
        dy_pred = Theta @ self.coef_
        pred_std = ys[:-1] + np.clip(dy_pred, -self.cfg.dy_clip, self.cfg.dy_clip)
        truth_std = ys[1:]
        start = min(max(int(burn_in), 1), len(ys) - 2)
        pred = pred_std[start:] * self.std_ + self.mu_
        truth = truth_std[start:] * self.std_ + self.mu_
        return {f"one_step_{k}": v for k, v in metric_dict(truth, pred).items()}

    def count_total_params(self) -> int:
        return 0 if self.coef_ is None else int(len(self.coef_))

    def count_trained_params(self) -> int:
        return self.count_total_params()

    def effective_dim(self) -> int:
        return self.count_total_params()


class SlowBackboneSINDy:
    def __init__(self, cfg: SlowSINDyConfig):
        self.cfg = cfg
        self.encoder = CausalFastSlowEncoder(cfg.fs_cfg)
        self.coef_: Optional[np.ndarray] = None
        self.lib_names_: List[str] = []

    def fit_on_standardized(self, y_std: np.ndarray) -> "SlowBackboneSINDy":
        feats = self.encoder.build_feature_sequence(y_std)
        slow = feats["slow"]
        ds = feats["ds"]
        X = np.column_stack([slow[:-1], ds[:-1]])
        target = slow[1:] - slow[:-1]
        Theta, names = build_poly_library(X, ["slow", "ds"], poly_order=self.cfg.poly_order)
        self.coef_ = fit_stlsq(Theta, target, ridge=self.cfg.ridge, threshold=self.cfg.threshold)
        self.lib_names_ = names
        return self

    def predict_delta(self, slow_t: float, ds_t: float) -> float:
        x = np.array([[slow_t, ds_t]], dtype=float)
        Theta, _ = build_poly_library(x, ["slow", "ds"], poly_order=self.cfg.poly_order)
        return float((Theta @ self.coef_).item())

    def predict_next(self, slow_t: float, ds_t: float) -> float:
        return float(np.clip(slow_t + self.predict_delta(slow_t, ds_t), -10.0, 10.0))

    def count_params(self) -> int:
        return 0 if self.coef_ is None else int(len(self.coef_))

    def effective_dim(self) -> int:
        return self.count_params()


class SlowSINDyOnlyModel(BaseForecastModel):
    def __init__(self, slow_cfg: SlowSINDyConfig):
        self.slow_cfg = slow_cfg
        self.encoder = CausalFastSlowEncoder(slow_cfg.fs_cfg)
        self.backbone = SlowBackboneSINDy(slow_cfg)
        self.mu_ = 0.0
        self.std_ = 1.0

    def fit(self, y_train: np.ndarray) -> "SlowSINDyOnlyModel":
        self.mu_ = float(np.mean(y_train))
        self.std_ = float(np.std(y_train) + 1e-12)
        ys = ((y_train - self.mu_) / self.std_).astype(float)
        self.backbone.fit_on_standardized(ys)
        return self

    def rollout(self, y_hist: np.ndarray, horizon: int) -> np.ndarray:
        y_hist_std = ((np.asarray(y_hist, dtype=float).reshape(-1) - self.mu_) / self.std_).astype(float)
        feats = self.encoder.build_feature_sequence(y_hist_std)
        slow_cur = float(feats["slow"][-1])
        ds_cur = float(feats["ds"][-1])
        preds = []
        for _ in range(horizon):
            slow_next = self.backbone.predict_next(slow_cur, ds_cur)
            preds.append(slow_next)
            ds_cur = slow_next - slow_cur
            slow_cur = slow_next
        return np.asarray(preds, dtype=float) * self.std_ + self.mu_

    def one_step_metrics(self, series: np.ndarray, burn_in: int):
        ys = ((np.asarray(series, dtype=float).reshape(-1) - self.mu_) / self.std_).astype(float)
        feats = self.encoder.build_feature_sequence(ys)
        slow = feats["slow"]
        ds = feats["ds"]
        preds_std = np.array([self.backbone.predict_next(slow[t], ds[t]) for t in range(len(ys) - 1)], dtype=float)
        truth = ys[1:]
        start = min(max(int(burn_in), 1), len(ys) - 2)
        pred = preds_std[start:] * self.std_ + self.mu_
        truth = truth[start:] * self.std_ + self.mu_
        return {f"one_step_{k}": v for k, v in metric_dict(truth, pred).items()}

    def count_total_params(self) -> int:
        return self.backbone.count_params()

    def count_trained_params(self) -> int:
        return self.backbone.count_params()

    def effective_dim(self) -> int:
        return self.backbone.count_params()