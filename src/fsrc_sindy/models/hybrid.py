from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..fastslow import CausalFastSlowEncoder
from ..metrics import metric_dict
from ..utils import ridge_solve, safe_clip
from .base import BaseForecastModel
from .rc import RCConfig, ReservoirTemplateFactory
from .sindy import SlowBackboneSINDy, SlowSINDyConfig


@dataclass
class ResidualLinearConfig:
    ridge: float = 1e-4
    washout: int = 50
    delta_clip: float = 1.0
    resid_clip: float = 5.0
    damp: float = 1.0


@dataclass
class ResidualRCConfig(RCConfig):
    delta_clip: float = 1.0
    resid_clip: float = 5.0
    damp: float = 0.7


class _SlowResidualBase(BaseForecastModel):
    def __init__(self, slow_cfg: SlowSINDyConfig):
        self.slow_cfg = slow_cfg
        self.encoder = CausalFastSlowEncoder(slow_cfg.fs_cfg)
        self.backbone = SlowBackboneSINDy(slow_cfg)
        self.mu_ = 0.0
        self.std_ = 1.0

    def _standardize(self, y: np.ndarray) -> np.ndarray:
        return ((np.asarray(y, dtype=float).reshape(-1) - self.mu_) / self.std_).astype(float)


class SlowSINDyLevelLinearModel(_SlowResidualBase):
    """Legacy unstable model kept for diagnosis."""
    def __init__(self, slow_cfg: SlowSINDyConfig, cfg: ResidualLinearConfig):
        super().__init__(slow_cfg)
        self.cfg = cfg
        self.coef_: Optional[np.ndarray] = None

    @staticmethod
    def _feature_vec(resid_t: float, slow_t: float, fast_t: float, m_t: float, ds_pred: float) -> np.ndarray:
        return np.array([resid_t, slow_t, fast_t, m_t, ds_pred, 1.0], dtype=float)

    def fit(self, y_train: np.ndarray):
        self.mu_ = float(np.mean(y_train))
        self.std_ = float(np.std(y_train) + 1e-12)
        ys = self._standardize(y_train)
        feats = self.encoder.build_feature_sequence(ys)
        self.backbone.fit_on_standardized(ys)
        X_rows, Y = [], []
        start = max(self.cfg.washout, 1)
        for t in range(start, len(ys) - 1):
            slow_t = float(feats["slow"][t])
            ds_t = float(feats["ds"][t])
            slow_next = self.backbone.predict_next(slow_t, ds_t)
            ds_pred = slow_next - slow_t
            x = self._feature_vec(float(ys[t] - slow_t), slow_t, float(feats["fast"][t]), float(feats["m"][t]), ds_pred)
            y_target = float(ys[t + 1] - slow_next)
            X_rows.append(x)
            Y.append(y_target)
        self.coef_ = ridge_solve(np.vstack(X_rows), np.asarray(Y), self.cfg.ridge)
        return self

    def rollout(self, y_hist: np.ndarray, horizon: int) -> np.ndarray:
        ys = self._standardize(y_hist)
        feats = self.encoder.build_feature_sequence(ys)
        fast_state = self.encoder.init_fast_state(ys)
        slow_cur = float(feats["slow"][-1])
        ds_cur = float(feats["ds"][-1])
        resid_cur = float(ys[-1] - slow_cur)
        preds = []
        for _ in range(horizon):
            slow_next = self.backbone.predict_next(slow_cur, ds_cur)
            ds_pred = slow_next - slow_cur
            fast_cur = float(fast_state["f2"])
            m_cur = float(fast_cur - slow_cur)
            x = self._feature_vec(resid_cur, slow_cur, fast_cur, m_cur, ds_pred)
            resid_next = float(x @ self.coef_)
            y_next = float(slow_next + resid_next)
            preds.append(y_next)
            self.encoder.advance_fast(fast_state, y_next)
            resid_cur, ds_cur, slow_cur = resid_next, ds_pred, slow_next
        return np.asarray(preds) * self.std_ + self.mu_

    def one_step_metrics(self, series: np.ndarray, burn_in: int):
        ys = self._standardize(series)
        feats = self.encoder.build_feature_sequence(ys)
        preds, truth = [], []
        start = min(max(int(burn_in), self.cfg.washout), len(ys) - 2)
        for t in range(start, len(ys) - 1):
            slow_t = float(feats["slow"][t])
            ds_t = float(feats["ds"][t])
            slow_next = self.backbone.predict_next(slow_t, ds_t)
            ds_pred = slow_next - slow_t
            x = self._feature_vec(float(ys[t] - slow_t), slow_t, float(feats["fast"][t]), float(feats["m"][t]), ds_pred)
            resid_next = float(x @ self.coef_)
            pred_std = slow_next + resid_next
            preds.append(pred_std * self.std_ + self.mu_)
            truth.append(ys[t + 1] * self.std_ + self.mu_)
        return {f"one_step_{k}": v for k, v in metric_dict(np.asarray(truth), np.asarray(preds)).items()}

    def count_total_params(self) -> int:
        return self.backbone.count_params() + (0 if self.coef_ is None else int(len(self.coef_)))

    def count_trained_params(self) -> int:
        return self.count_total_params()

    def effective_dim(self) -> int:
        return self.count_total_params()


class SlowSINDyDeltaLinearModel(_SlowResidualBase):
    def __init__(self, slow_cfg: SlowSINDyConfig, cfg: ResidualLinearConfig):
        super().__init__(slow_cfg)
        self.cfg = cfg
        self.coef_: Optional[np.ndarray] = None

    @staticmethod
    def _feature_vec(resid_t: float, slow_t: float, fast_t: float, m_t: float, ds_pred: float) -> np.ndarray:
        return np.array([resid_t, slow_t, fast_t, m_t, ds_pred, 1.0], dtype=float)

    def fit(self, y_train: np.ndarray):
        self.mu_ = float(np.mean(y_train))
        self.std_ = float(np.std(y_train) + 1e-12)
        ys = self._standardize(y_train)
        feats = self.encoder.build_feature_sequence(ys)
        self.backbone.fit_on_standardized(ys)
        X_rows, Y = [], []
        start = max(self.cfg.washout, 1)
        for t in range(start, len(ys) - 1):
            slow_t = float(feats["slow"][t])
            ds_t = float(feats["ds"][t])
            slow_next = self.backbone.predict_next(slow_t, ds_t)
            ds_pred = slow_next - slow_t
            resid_t = float(ys[t] - slow_t)
            resid_next_true = float(ys[t + 1] - slow_next)
            delta = float(safe_clip(resid_next_true - resid_t, self.cfg.delta_clip))
            x = self._feature_vec(resid_t, slow_t, float(feats["fast"][t]), float(feats["m"][t]), ds_pred)
            X_rows.append(x)
            Y.append(delta)
        self.coef_ = ridge_solve(np.vstack(X_rows), np.asarray(Y), self.cfg.ridge)
        return self

    def rollout(self, y_hist: np.ndarray, horizon: int) -> np.ndarray:
        ys = self._standardize(y_hist)
        feats = self.encoder.build_feature_sequence(ys)
        fast_state = self.encoder.init_fast_state(ys)
        slow_cur = float(feats["slow"][-1])
        ds_cur = float(feats["ds"][-1])
        resid_cur = float(ys[-1] - slow_cur)
        preds = []
        for _ in range(horizon):
            slow_next = self.backbone.predict_next(slow_cur, ds_cur)
            ds_pred = slow_next - slow_cur
            fast_cur = float(fast_state["f2"])
            m_cur = float(fast_cur - slow_cur)
            x = self._feature_vec(resid_cur, slow_cur, fast_cur, m_cur, ds_pred)
            delta = float(safe_clip(x @ self.coef_, self.cfg.delta_clip))
            resid_next = float(safe_clip(resid_cur + self.cfg.damp * delta, self.cfg.resid_clip))
            y_next = float(slow_next + resid_next)
            preds.append(y_next)
            self.encoder.advance_fast(fast_state, y_next)
            resid_cur, ds_cur, slow_cur = resid_next, ds_pred, slow_next
        return np.asarray(preds) * self.std_ + self.mu_

    def one_step_metrics(self, series: np.ndarray, burn_in: int):
        ys = self._standardize(series)
        feats = self.encoder.build_feature_sequence(ys)
        preds, truth = [], []
        start = min(max(int(burn_in), self.cfg.washout), len(ys) - 2)
        for t in range(start, len(ys) - 1):
            slow_t = float(feats["slow"][t])
            ds_t = float(feats["ds"][t])
            slow_next = self.backbone.predict_next(slow_t, ds_t)
            ds_pred = slow_next - slow_t
            resid_t = float(ys[t] - slow_t)
            x = self._feature_vec(resid_t, slow_t, float(feats["fast"][t]), float(feats["m"][t]), ds_pred)
            delta = float(safe_clip(x @ self.coef_, self.cfg.delta_clip))
            resid_next = float(safe_clip(resid_t + self.cfg.damp * delta, self.cfg.resid_clip))
            pred_std = slow_next + resid_next
            preds.append(pred_std * self.std_ + self.mu_)
            truth.append(ys[t + 1] * self.std_ + self.mu_)
        return {f"one_step_{k}": v for k, v in metric_dict(np.asarray(truth), np.asarray(preds)).items()}

    def count_total_params(self) -> int:
        return self.backbone.count_params() + (0 if self.coef_ is None else int(len(self.coef_)))

    def count_trained_params(self) -> int:
        return self.count_total_params()

    def effective_dim(self) -> int:
        return self.count_total_params()


class SlowSINDyLevelRCModel(_SlowResidualBase):
    """Legacy unstable model kept for diagnosis."""
    def __init__(self, slow_cfg: SlowSINDyConfig, cfg: ResidualRCConfig, template_factory: ReservoirTemplateFactory):
        super().__init__(slow_cfg)
        self.cfg = cfg
        self.template_factory = template_factory
        self.W = None
        self.Win = None
        self.bias = None
        self.Wout = None

    def _setup(self) -> None:
        tpl = self.template_factory.get(self.cfg.n_reservoir, 1, self.cfg.sparsity)
        self.W = tpl["W_unit"] * self.cfg.spectral_radius
        self.Win = tpl["Win_base"] * self.cfg.input_scale
        self.bias = tpl["bias"]

    def _step(self, r: np.ndarray, resid_t: float) -> np.ndarray:
        pre = self.W.dot(r) + self.Win[:, 0] * float(resid_t) + self.bias
        cand = np.tanh(pre)
        return (1.0 - self.cfg.leak_rate) * r + self.cfg.leak_rate * cand

    @staticmethod
    def _aug(r: np.ndarray, resid_t: float, slow_t: float, fast_t: float, m_t: float, ds_pred: float) -> np.ndarray:
        return np.concatenate([r, np.array([resid_t, slow_t, fast_t, m_t, ds_pred, 1.0], dtype=float)])

    def fit(self, y_train: np.ndarray):
        self.mu_ = float(np.mean(y_train))
        self.std_ = float(np.std(y_train) + 1e-12)
        ys = self._standardize(y_train)
        feats = self.encoder.build_feature_sequence(ys)
        self.backbone.fit_on_standardized(ys)
        self._setup()
        r = np.zeros(self.cfg.n_reservoir, dtype=float)
        X_rows, Y = [], []
        for t in range(len(ys) - 1):
            resid_t = float(ys[t] - feats["slow"][t])
            r = self._step(r, resid_t)
            if t >= self.cfg.washout:
                slow_t = float(feats["slow"][t])
                ds_t = float(feats["ds"][t])
                slow_next = self.backbone.predict_next(slow_t, ds_t)
                ds_pred = slow_next - slow_t
                X_rows.append(self._aug(r, resid_t, slow_t, float(feats["fast"][t]), float(feats["m"][t]), ds_pred))
                Y.append(float(ys[t + 1] - slow_next))
        self.Wout = ridge_solve(np.vstack(X_rows), np.asarray(Y), self.cfg.ridge)
        return self

    def rollout(self, y_hist: np.ndarray, horizon: int) -> np.ndarray:
        ys = self._standardize(y_hist)
        feats = self.encoder.build_feature_sequence(ys)
        fast_state = self.encoder.init_fast_state(ys)
        r = np.zeros(self.cfg.n_reservoir, dtype=float)
        resid_hist = ys - feats["slow"]
        for val in resid_hist:
            r = self._step(r, float(val))
        slow_cur = float(feats["slow"][-1])
        ds_cur = float(feats["ds"][-1])
        resid_cur = float(ys[-1] - slow_cur)
        preds = []
        for _ in range(horizon):
            slow_next = self.backbone.predict_next(slow_cur, ds_cur)
            ds_pred = slow_next - slow_cur
            fast_cur = float(fast_state["f2"])
            m_cur = float(fast_cur - slow_cur)
            aug = self._aug(r, resid_cur, slow_cur, fast_cur, m_cur, ds_pred)
            resid_next = float(aug @ self.Wout)
            y_next = float(slow_next + resid_next)
            preds.append(y_next)
            self.encoder.advance_fast(fast_state, y_next)
            r = self._step(r, resid_next)
            slow_cur, ds_cur, resid_cur = slow_next, ds_pred, resid_next
        return np.asarray(preds) * self.std_ + self.mu_

    def one_step_metrics(self, series: np.ndarray, burn_in: int):
        ys = self._standardize(series)
        feats = self.encoder.build_feature_sequence(ys)
        r = np.zeros(self.cfg.n_reservoir, dtype=float)
        preds, truth = [], []
        start = min(max(int(burn_in), self.cfg.washout), len(ys) - 2)
        for t in range(len(ys) - 1):
            resid_t = float(ys[t] - feats["slow"][t])
            r = self._step(r, resid_t)
            if t >= start:
                slow_t = float(feats["slow"][t])
                ds_t = float(feats["ds"][t])
                slow_next = self.backbone.predict_next(slow_t, ds_t)
                ds_pred = slow_next - slow_t
                aug = self._aug(r, resid_t, slow_t, float(feats["fast"][t]), float(feats["m"][t]), ds_pred)
                resid_next = float(aug @ self.Wout)
                pred_std = slow_next + resid_next
                preds.append(pred_std * self.std_ + self.mu_)
                truth.append(ys[t + 1] * self.std_ + self.mu_)
        return {f"one_step_{k}": v for k, v in metric_dict(np.asarray(truth), np.asarray(preds)).items()}

    def count_total_params(self) -> int:
        extra = 6
        return self.backbone.count_params() + int(self.cfg.n_reservoir + self.cfg.n_reservoir * self.cfg.n_reservoir + self.cfg.n_reservoir + (self.cfg.n_reservoir + extra))

    def count_trained_params(self) -> int:
        extra = 6
        return self.backbone.count_params() + int(self.cfg.n_reservoir + extra)

    def effective_dim(self) -> int:
        return self.backbone.count_params() + int(self.cfg.n_reservoir)


class SlowSINDyDeltaRCModel(_SlowResidualBase):
    def __init__(self, slow_cfg: SlowSINDyConfig, cfg: ResidualRCConfig, template_factory: ReservoirTemplateFactory):
        super().__init__(slow_cfg)
        self.cfg = cfg
        self.template_factory = template_factory
        self.W = None
        self.Win = None
        self.bias = None
        self.Wout = None

    def _setup(self) -> None:
        tpl = self.template_factory.get(self.cfg.n_reservoir, 1, self.cfg.sparsity)
        self.W = tpl["W_unit"] * self.cfg.spectral_radius
        self.Win = tpl["Win_base"] * self.cfg.input_scale
        self.bias = tpl["bias"]

    def _step(self, r: np.ndarray, resid_t: float) -> np.ndarray:
        pre = self.W.dot(r) + self.Win[:, 0] * float(resid_t) + self.bias
        cand = np.tanh(pre)
        return (1.0 - self.cfg.leak_rate) * r + self.cfg.leak_rate * cand

    @staticmethod
    def _aug(r: np.ndarray, resid_t: float, slow_t: float, fast_t: float, m_t: float, ds_pred: float) -> np.ndarray:
        return np.concatenate([r, np.array([resid_t, slow_t, fast_t, m_t, ds_pred, 1.0], dtype=float)])

    def fit(self, y_train: np.ndarray):
        self.mu_ = float(np.mean(y_train))
        self.std_ = float(np.std(y_train) + 1e-12)
        ys = self._standardize(y_train)
        feats = self.encoder.build_feature_sequence(ys)
        self.backbone.fit_on_standardized(ys)
        self._setup()
        r = np.zeros(self.cfg.n_reservoir, dtype=float)
        X_rows, Y = [], []
        for t in range(len(ys) - 1):
            resid_t = float(ys[t] - feats["slow"][t])
            r = self._step(r, resid_t)
            if t >= self.cfg.washout:
                slow_t = float(feats["slow"][t])
                ds_t = float(feats["ds"][t])
                slow_next = self.backbone.predict_next(slow_t, ds_t)
                ds_pred = slow_next - slow_t
                resid_next_true = float(ys[t + 1] - slow_next)
                delta = float(safe_clip(resid_next_true - resid_t, self.cfg.delta_clip))
                X_rows.append(self._aug(r, resid_t, slow_t, float(feats["fast"][t]), float(feats["m"][t]), ds_pred))
                Y.append(delta)
        self.Wout = ridge_solve(np.vstack(X_rows), np.asarray(Y), self.cfg.ridge)
        return self

    def rollout(self, y_hist: np.ndarray, horizon: int) -> np.ndarray:
        ys = self._standardize(y_hist)
        feats = self.encoder.build_feature_sequence(ys)
        fast_state = self.encoder.init_fast_state(ys)
        r = np.zeros(self.cfg.n_reservoir, dtype=float)
        resid_hist = ys - feats["slow"]
        for val in resid_hist:
            r = self._step(r, float(val))
        slow_cur = float(feats["slow"][-1])
        ds_cur = float(feats["ds"][-1])
        resid_cur = float(ys[-1] - slow_cur)
        preds = []
        for _ in range(horizon):
            slow_next = self.backbone.predict_next(slow_cur, ds_cur)
            ds_pred = slow_next - slow_cur
            fast_cur = float(fast_state["f2"])
            m_cur = float(fast_cur - slow_cur)
            aug = self._aug(r, resid_cur, slow_cur, fast_cur, m_cur, ds_pred)
            delta = float(safe_clip(aug @ self.Wout, self.cfg.delta_clip))
            resid_next = float(safe_clip(resid_cur + self.cfg.damp * delta, self.cfg.resid_clip))
            y_next = float(slow_next + resid_next)
            preds.append(y_next)
            self.encoder.advance_fast(fast_state, y_next)
            r = self._step(r, resid_next)
            slow_cur, ds_cur, resid_cur = slow_next, ds_pred, resid_next
        return np.asarray(preds) * self.std_ + self.mu_

    def one_step_metrics(self, series: np.ndarray, burn_in: int):
        ys = self._standardize(series)
        feats = self.encoder.build_feature_sequence(ys)
        r = np.zeros(self.cfg.n_reservoir, dtype=float)
        preds, truth = [], []
        start = min(max(int(burn_in), self.cfg.washout), len(ys) - 2)
        for t in range(len(ys) - 1):
            resid_t = float(ys[t] - feats["slow"][t])
            r = self._step(r, resid_t)
            if t >= start:
                slow_t = float(feats["slow"][t])
                ds_t = float(feats["ds"][t])
                slow_next = self.backbone.predict_next(slow_t, ds_t)
                ds_pred = slow_next - slow_t
                aug = self._aug(r, resid_t, slow_t, float(feats["fast"][t]), float(feats["m"][t]), ds_pred)
                delta = float(safe_clip(aug @ self.Wout, self.cfg.delta_clip))
                resid_next = float(safe_clip(resid_t + self.cfg.damp * delta, self.cfg.resid_clip))
                pred_std = slow_next + resid_next
                preds.append(pred_std * self.std_ + self.mu_)
                truth.append(ys[t + 1] * self.std_ + self.mu_)
        return {f"one_step_{k}": v for k, v in metric_dict(np.asarray(truth), np.asarray(preds)).items()}

    def count_total_params(self) -> int:
        extra = 6
        return self.backbone.count_params() + int(self.cfg.n_reservoir + self.cfg.n_reservoir * self.cfg.n_reservoir + self.cfg.n_reservoir + (self.cfg.n_reservoir + extra))

    def count_trained_params(self) -> int:
        extra = 6
        return self.backbone.count_params() + int(self.cfg.n_reservoir + extra)

    def effective_dim(self) -> int:
        return self.backbone.count_params() + int(self.cfg.n_reservoir)
