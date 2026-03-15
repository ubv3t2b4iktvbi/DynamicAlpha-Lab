from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..factors.base import DynamicsFeatureConfig, FactorSpec
from ..factors.readout import CausalFactorReadout, ReadoutState
from ..factors.repository import fastslow_readout_specs, sparse_rg_gate_specs
from ..fastslow import CausalFastSlowEncoder, FastSlowConfig
from ..library import build_poly_library
from ..metrics import metric_dict
from ..utils import ridge_solve, safe_clip
from .base import BaseForecastModel
from .rc import RCConfig, ReservoirTemplateFactory
from .sindy import SlowBackboneSINDy, SlowSINDyConfig


def quadratic_feature_dim(base_dim: int) -> int:
    return int(1 + 2 * base_dim + (base_dim * (base_dim - 1)) // 2)


@dataclass
class NGRCConfig:
    n_delays: int = 14
    stride: int = 1
    poly_order: int = 2
    ridge: float = 1e-5
    washout: int = 50
    feature_clip: float = 5.0
    y_clip: float = 10.0
    fs_cfg: Optional[FastSlowConfig] = None


@dataclass
class RCNGRCConfig(RCConfig):
    n_delays: int = 10
    stride: int = 1
    poly_order: int = 2
    feature_clip: float = 5.0
    y_clip: float = 10.0


@dataclass
class ResidualNGRCConfig(NGRCConfig):
    delta_clip: float = 1.0
    resid_clip: float = 5.0
    damp: float = 0.7


@dataclass
class ResidualRCNGRCConfig(RCNGRCConfig):
    delta_clip: float = 1.0
    resid_clip: float = 5.0
    damp: float = 0.7


class _DelayBuilder:
    def __init__(self, n_delays: int, stride: int):
        assert n_delays >= 1
        assert stride >= 1
        self.n_delays = int(n_delays)
        self.stride = int(stride)
        self.max_lag = (self.n_delays - 1) * self.stride

    def row_from_series(self, series: np.ndarray, t: int) -> np.ndarray:
        return np.asarray([series[t - i * self.stride] for i in range(self.n_delays)], dtype=float)

    def row_from_deque(self, hist: deque[float]) -> np.ndarray:
        arr = np.asarray(hist, dtype=float)
        return np.asarray([arr[-1 - i * self.stride] for i in range(self.n_delays)], dtype=float)


class PureNGRCModel(BaseForecastModel):
    def __init__(
        self,
        cfg: NGRCConfig,
        fs_cfg: Optional[FastSlowConfig] = None,
        use_fastslow_readout: bool = False,
        readout_factor_specs: Sequence[FactorSpec] | None = None,
        readout_identifier_kind: str | None = None,
        readout_feature_cfg: DynamicsFeatureConfig | None = None,
    ):
        self.cfg = cfg
        self.use_fastslow_readout = use_fastslow_readout
        self.fs_cfg = fs_cfg if fs_cfg is not None else (cfg.fs_cfg if cfg.fs_cfg is not None else FastSlowConfig(t0=4, slow_scales=(8, 16, 32)))
        factor_specs = list(readout_factor_specs) if readout_factor_specs is not None else (fastslow_readout_specs() if use_fastslow_readout else [])
        self.readout = CausalFactorReadout(
            factor_specs=factor_specs,
            feature_cfg=readout_feature_cfg,
            identifier_kind=readout_identifier_kind,
            fastslow_cfg=self.fs_cfg,
        )
        self.delay = _DelayBuilder(cfg.n_delays, cfg.stride)

        self.mu_ = 0.0
        self.std_ = 1.0
        self.coef_: Optional[np.ndarray] = None
        self.n_features_: int = 0

    def _build_design(self, ys: np.ndarray, factor_mat: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        base_rows = []
        targets = []
        start = max(self.cfg.washout, self.delay.max_lag)
        for t in range(start, len(ys) - 1):
            base_rows.append(self.delay.row_from_series(ys, t))
            targets.append(float(ys[t + 1]))
        if not base_rows:
            raise ValueError('Not enough data for NGRC design matrix')
        base = np.vstack(base_rows)
        base = np.asarray(safe_clip(base, self.cfg.feature_clip), dtype=float)
        Phi, _ = build_poly_library(base, [f'z{i}' for i in range(base.shape[1])], poly_order=self.cfg.poly_order)
        if factor_mat is not None and factor_mat.shape[1] > 0:
            extra = np.asarray(factor_mat[start:len(ys) - 1], dtype=float)
            X = np.hstack([Phi, extra])
        else:
            X = Phi
        X = np.asarray(safe_clip(X, self.cfg.feature_clip), dtype=float)
        Y = np.asarray(targets, dtype=float)
        return X, Y

    def _feature_row(
        self,
        delay_row: np.ndarray,
        readout_state: Optional[ReadoutState] = None,
        factor_mat: Optional[np.ndarray] = None,
        t: Optional[int] = None,
    ) -> np.ndarray:
        delay_row = np.asarray(safe_clip(delay_row, self.cfg.feature_clip), dtype=float).reshape(1, -1)
        Phi, _ = build_poly_library(delay_row, [f'z{i}' for i in range(delay_row.shape[1])], poly_order=self.cfg.poly_order)
        row = Phi.reshape(-1)
        if self.readout.dim > 0:
            if factor_mat is not None and t is not None:
                extra = np.asarray(factor_mat[t], dtype=float)
            elif readout_state is not None:
                extra = self.readout.factor_step(readout_state.context)
            else:
                raise ValueError('factor readout requested but no state provided')
            row = np.concatenate([row, extra])
        row = np.asarray(safe_clip(row, self.cfg.feature_clip), dtype=float)
        return row

    def fit(self, y_train: np.ndarray) -> 'PureNGRCModel':
        self.mu_ = float(np.mean(y_train))
        self.std_ = float(np.std(y_train) + 1e-12)
        ys = ((np.asarray(y_train, dtype=float).reshape(-1) - self.mu_) / self.std_).astype(float)
        _, factor_mat = self.readout.fit_transform(ys)
        X, Y = self._build_design(ys, factor_mat)
        self.coef_ = ridge_solve(X, Y, self.cfg.ridge)
        self.n_features_ = int(X.shape[1])
        return self

    def rollout(self, y_hist: np.ndarray, horizon: int) -> np.ndarray:
        ys = ((np.asarray(y_hist, dtype=float).reshape(-1) - self.mu_) / self.std_).astype(float)
        if len(ys) <= self.delay.max_lag:
            raise ValueError('history too short for NGRC rollout')
        hist = deque([float(v) for v in ys], maxlen=max(len(ys), self.delay.max_lag + 1))
        readout_state = self.readout.warmup(ys) if self.readout.dim > 0 else None
        preds = []
        for _ in range(horizon):
            delay_row = self.delay.row_from_deque(hist)
            x = self._feature_row(delay_row, readout_state=readout_state)
            y_next = float(safe_clip(x @ self.coef_, self.cfg.y_clip))
            preds.append(y_next)
            hist.append(y_next)
            if readout_state is not None:
                self.readout.advance(readout_state, y_next)
        return np.asarray(preds, dtype=float) * self.std_ + self.mu_

    def one_step_metrics(self, series: np.ndarray, burn_in: int):
        ys = ((np.asarray(series, dtype=float).reshape(-1) - self.mu_) / self.std_).astype(float)
        _, factor_mat = self.readout.transform(ys)
        start = max(int(burn_in), self.cfg.washout, self.delay.max_lag)
        preds, truth = [], []
        for t in range(start, len(ys) - 1):
            delay_row = self.delay.row_from_series(ys, t)
            x = self._feature_row(delay_row, factor_mat=factor_mat, t=t)
            pred_std = float(safe_clip(x @ self.coef_, self.cfg.y_clip))
            preds.append(pred_std * self.std_ + self.mu_)
            truth.append(ys[t + 1] * self.std_ + self.mu_)
        return {f'one_step_{k}': v for k, v in metric_dict(np.asarray(truth), np.asarray(preds)).items()}

    def count_total_params(self) -> int:
        return int(self.n_features_)

    def count_trained_params(self) -> int:
        return int(self.n_features_)

    def effective_dim(self) -> int:
        return int(self.n_features_)


def _takens_delay_summaries(delay_rows: np.ndarray) -> np.ndarray:
    delay_rows = np.asarray(delay_rows, dtype=float)
    if delay_rows.ndim == 1:
        delay_rows = delay_rows.reshape(1, -1)
    mean = np.mean(delay_rows, axis=1, keepdims=True)
    centered = delay_rows - mean
    variance = np.mean(centered ** 2, axis=1, keepdims=True)
    if delay_rows.shape[1] >= 2:
        tangent = (delay_rows[:, 0] - delay_rows[:, 1]).reshape(-1, 1)
    else:
        tangent = np.zeros((delay_rows.shape[0], 1), dtype=float)
    if delay_rows.shape[1] >= 3:
        curvature = (delay_rows[:, 0] - 2.0 * delay_rows[:, 1] + delay_rows[:, 2]).reshape(-1, 1)
    else:
        curvature = np.zeros((delay_rows.shape[0], 1), dtype=float)
    return np.hstack([mean, tangent, curvature, variance])


class TakensRGResidualNGRCModel(BaseForecastModel):
    """Delay-backbone NGRC plus a sparse RG-conditioned residual correction."""

    def __init__(
        self,
        cfg: NGRCConfig,
        fs_cfg: Optional[FastSlowConfig] = None,
        readout_factor_specs: Sequence[FactorSpec] | None = None,
        readout_identifier_kind: str | None = None,
        readout_feature_cfg: DynamicsFeatureConfig | None = None,
        correction_mode: str = "interaction",
        rg_control_mode: str = "none",
        rg_lag: int = 1,
        random_feature_seed: int = 0,
    ):
        self.cfg = cfg
        self.fs_cfg = fs_cfg if fs_cfg is not None else (cfg.fs_cfg if cfg.fs_cfg is not None else FastSlowConfig(t0=4, slow_scales=(8, 16, 32)))
        factor_specs = list(readout_factor_specs) if readout_factor_specs is not None else sparse_rg_gate_specs()
        self.readout = CausalFactorReadout(
            factor_specs=factor_specs,
            feature_cfg=readout_feature_cfg,
            identifier_kind=readout_identifier_kind,
            fastslow_cfg=self.fs_cfg,
        )
        self.delay = _DelayBuilder(cfg.n_delays, cfg.stride)
        self.correction_mode = str(correction_mode).strip().lower() or "interaction"
        self.rg_control_mode = str(rg_control_mode).strip().lower() or "none"
        self.rg_lag = max(1, int(rg_lag))
        self.random_feature_seed = int(random_feature_seed)
        if self.correction_mode not in {"interaction", "additive"}:
            raise ValueError(f"Unsupported correction_mode={correction_mode}")
        if self.rg_control_mode not in {"none", "lagged_rg", "random_summary"}:
            raise ValueError(f"Unsupported rg_control_mode={rg_control_mode}")

        self.mu_ = 0.0
        self.std_ = 1.0
        self.base_coef_: Optional[np.ndarray] = None
        self.corr_coef_: Optional[np.ndarray] = None
        self.base_dim_: int = 0
        self.corr_dim_: int = 0
        self.n_features_: int = 0
        self.summary_mu_ = np.zeros(4, dtype=float)
        self.summary_std_ = np.ones(4, dtype=float)
        self.random_W_: Optional[np.ndarray] = None
        self.random_b_: Optional[np.ndarray] = None

    def _standardize(self, y: np.ndarray) -> np.ndarray:
        return ((np.asarray(y, dtype=float).reshape(-1) - self.mu_) / self.std_).astype(float)

    def _delay_basis(self, delay_rows: np.ndarray) -> np.ndarray:
        delay_rows = np.asarray(safe_clip(delay_rows, self.cfg.feature_clip), dtype=float)
        Phi, _ = build_poly_library(delay_rows, [f"z{i}" for i in range(delay_rows.shape[1])], poly_order=self.cfg.poly_order)
        return np.asarray(safe_clip(Phi, self.cfg.feature_clip), dtype=float)

    def _random_control_rows(self, delay_rows: np.ndarray, fit: bool) -> np.ndarray:
        delay_rows = np.asarray(delay_rows, dtype=float)
        if delay_rows.ndim == 1:
            delay_rows = delay_rows.reshape(1, -1)
        if self.readout.dim <= 0:
            return np.zeros((delay_rows.shape[0], 0), dtype=float)
        summaries = _takens_delay_summaries(delay_rows)
        if fit or self.random_W_ is None or self.random_b_ is None:
            self.summary_mu_ = np.mean(summaries, axis=0)
            self.summary_std_ = np.std(summaries, axis=0) + 1e-8
            rng = np.random.default_rng(self.random_feature_seed)
            self.random_W_ = rng.normal(
                loc=0.0,
                scale=1.0 / np.sqrt(max(1, summaries.shape[1])),
                size=(summaries.shape[1], self.readout.dim),
            )
            self.random_b_ = rng.normal(loc=0.0, scale=0.1, size=self.readout.dim)
        normed = (summaries - self.summary_mu_) / self.summary_std_
        proj = normed @ self.random_W_ + self.random_b_
        return np.asarray(np.tanh(proj), dtype=float)

    def _control_rows_from_matrix(
        self,
        delay_rows: np.ndarray,
        factor_mat: Optional[np.ndarray],
        indices: Sequence[int],
        fit: bool,
    ) -> np.ndarray:
        if self.readout.dim <= 0:
            return np.zeros((len(indices), 0), dtype=float)
        if self.rg_control_mode == "random_summary":
            return self._random_control_rows(delay_rows, fit=fit)
        raw = np.asarray(factor_mat, dtype=float) if factor_mat is not None else np.zeros((0, self.readout.dim), dtype=float)
        rows: list[np.ndarray] = []
        for t in indices:
            src = int(t)
            if self.rg_control_mode == "lagged_rg":
                src = max(0, src - self.rg_lag)
            if raw.shape[0] == 0:
                rows.append(np.zeros(self.readout.dim, dtype=float))
            else:
                src = min(src, raw.shape[0] - 1)
                rows.append(np.asarray(raw[src], dtype=float))
        return np.vstack(rows) if rows else np.zeros((0, self.readout.dim), dtype=float)

    def _correction_basis(self, delay_rows: np.ndarray, control_rows: np.ndarray) -> np.ndarray:
        control_rows = np.asarray(control_rows, dtype=float)
        if control_rows.ndim == 1:
            control_rows = control_rows.reshape(1, -1)
        if control_rows.size == 0 or control_rows.shape[1] == 0:
            return np.zeros((control_rows.shape[0], 0), dtype=float)
        if self.correction_mode == "additive":
            return np.asarray(safe_clip(control_rows, self.cfg.feature_clip), dtype=float)
        summaries = _takens_delay_summaries(delay_rows)
        cols = [control_rows]
        for idx in range(summaries.shape[1]):
            cols.append(summaries[:, idx:idx + 1] * control_rows)
        corr = np.hstack(cols)
        return np.asarray(safe_clip(corr, self.cfg.feature_clip), dtype=float)

    def _feature_blocks(
        self,
        delay_row: np.ndarray,
        readout_state: Optional[ReadoutState] = None,
        factor_mat: Optional[np.ndarray] = None,
        t: Optional[int] = None,
        rg_queue: deque[np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        delay_arr = np.asarray(delay_row, dtype=float).reshape(1, -1)
        base = self._delay_basis(delay_arr).reshape(-1)
        if self.readout.dim > 0:
            if factor_mat is not None and t is not None:
                control = self._control_rows_from_matrix(delay_arr, factor_mat=factor_mat, indices=[int(t)], fit=False)
            elif self.rg_control_mode == "random_summary":
                control = self._random_control_rows(delay_arr, fit=False)
            elif readout_state is not None:
                if self.rg_control_mode == "lagged_rg":
                    if rg_queue is None or not rg_queue:
                        raise ValueError("lagged_rg control requires an initialized RG queue during rollout")
                    control = np.asarray(rg_queue[0], dtype=float).reshape(1, -1)
                else:
                    control = np.asarray(self.readout.factor_step(readout_state.context), dtype=float).reshape(1, -1)
            else:
                raise ValueError("factor readout requested but no RG state provided")
            corr = self._correction_basis(delay_arr, control).reshape(-1)
        else:
            corr = np.zeros(0, dtype=float)
        return base, corr

    def _init_rollout_control_state(self, ys: np.ndarray) -> tuple[Optional[ReadoutState], deque[np.ndarray] | None]:
        if self.readout.dim <= 0 or self.rg_control_mode == "random_summary":
            return None, None
        readout_state = self.readout.warmup(ys)
        if self.rg_control_mode != "lagged_rg":
            return readout_state, None
        _, factor_mat = self.readout.transform(ys)
        queue = deque(maxlen=self.rg_lag + 1)
        if factor_mat is None or factor_mat.shape[1] == 0:
            for _ in range(self.rg_lag + 1):
                queue.append(np.zeros(self.readout.dim, dtype=float))
            return readout_state, queue
        start = max(0, factor_mat.shape[0] - (self.rg_lag + 1))
        init_rows = [np.asarray(row, dtype=float) for row in factor_mat[start:]]
        if not init_rows:
            init_rows = [np.zeros(self.readout.dim, dtype=float)]
        while len(init_rows) < self.rg_lag + 1:
            init_rows.insert(0, init_rows[0].copy())
        for row in init_rows[-(self.rg_lag + 1):]:
            queue.append(row)
        return readout_state, queue

    def fit(self, y_train: np.ndarray) -> "TakensRGResidualNGRCModel":
        self.mu_ = float(np.mean(y_train))
        self.std_ = float(np.std(y_train) + 1e-12)
        ys = self._standardize(y_train)
        _, factor_mat = self.readout.fit_transform(ys)
        start = max(self.cfg.washout, self.delay.max_lag)
        delay_rows = []
        deltas = []
        indices = []
        for t in range(start, len(ys) - 1):
            delay_rows.append(self.delay.row_from_series(ys, t))
            deltas.append(float(ys[t + 1] - ys[t]))
            indices.append(int(t))
        if not delay_rows:
            raise ValueError("Not enough data for Takens RG residual NGRC design matrix")
        delay_block = np.vstack(delay_rows)
        delta_target = np.asarray(deltas, dtype=float)
        Phi = self._delay_basis(delay_block)
        self.base_coef_ = ridge_solve(Phi, delta_target, self.cfg.ridge)
        self.base_dim_ = int(Phi.shape[1])
        resid = delta_target - Phi @ self.base_coef_
        if self.readout.dim > 0:
            control_block = self._control_rows_from_matrix(delay_block, factor_mat=factor_mat, indices=indices, fit=True)
            Corr = self._correction_basis(delay_block, control_block)
            self.corr_coef_ = ridge_solve(Corr, resid, self.cfg.ridge)
            self.corr_dim_ = int(Corr.shape[1])
        else:
            self.corr_coef_ = np.zeros(0, dtype=float)
            self.corr_dim_ = 0
        self.n_features_ = int(self.base_dim_ + self.corr_dim_)
        return self

    def rollout(self, y_hist: np.ndarray, horizon: int) -> np.ndarray:
        ys = self._standardize(y_hist)
        if len(ys) <= self.delay.max_lag:
            raise ValueError("history too short for Takens RG residual NGRC rollout")
        hist = deque([float(v) for v in ys], maxlen=max(len(ys), self.delay.max_lag + 1))
        readout_state, rg_queue = self._init_rollout_control_state(ys)
        preds = []
        for _ in range(horizon):
            delay_row = self.delay.row_from_deque(hist)
            base, corr = self._feature_blocks(delay_row, readout_state=readout_state, rg_queue=rg_queue)
            delta = float(base @ self.base_coef_)
            if corr.size > 0:
                delta += float(corr @ self.corr_coef_)
            delta = float(safe_clip(delta, self.cfg.y_clip))
            y_next = float(safe_clip(hist[-1] + delta, self.cfg.y_clip))
            preds.append(y_next)
            hist.append(y_next)
            if readout_state is not None:
                self.readout.advance(readout_state, y_next)
                if self.rg_control_mode == "lagged_rg":
                    rg_queue.append(np.asarray(self.readout.factor_step(readout_state.context), dtype=float))
        return np.asarray(preds, dtype=float) * self.std_ + self.mu_

    def one_step_metrics(self, series: np.ndarray, burn_in: int):
        ys = self._standardize(series)
        _, factor_mat = self.readout.transform(ys)
        start = max(int(burn_in), self.cfg.washout, self.delay.max_lag)
        preds, truth = [], []
        for t in range(start, len(ys) - 1):
            delay_row = self.delay.row_from_series(ys, t)
            base, corr = self._feature_blocks(delay_row, factor_mat=factor_mat, t=t)
            delta = float(base @ self.base_coef_)
            if corr.size > 0:
                delta += float(corr @ self.corr_coef_)
            delta = float(safe_clip(delta, self.cfg.y_clip))
            pred_std = float(safe_clip(ys[t] + delta, self.cfg.y_clip))
            preds.append(pred_std * self.std_ + self.mu_)
            truth.append(ys[t + 1] * self.std_ + self.mu_)
        return {f"one_step_{k}": v for k, v in metric_dict(np.asarray(truth), np.asarray(preds)).items()}

    def count_total_params(self) -> int:
        return int(self.n_features_)

    def count_trained_params(self) -> int:
        return int(self.n_features_)

    def effective_dim(self) -> int:
        return int(self.n_features_)


class HybridRCNGRCModel(BaseForecastModel):
    def __init__(
        self,
        cfg: RCNGRCConfig,
        template_factory: ReservoirTemplateFactory,
        fs_cfg: Optional[FastSlowConfig] = None,
        use_fastslow_readout: bool = True,
        readout_factor_specs: Sequence[FactorSpec] | None = None,
        readout_identifier_kind: str | None = None,
        readout_feature_cfg: DynamicsFeatureConfig | None = None,
    ):
        self.cfg = cfg
        self.template_factory = template_factory
        self.use_fastslow_readout = use_fastslow_readout
        self.fs_cfg = fs_cfg if fs_cfg is not None else (cfg.fs_cfg if cfg.fs_cfg is not None else FastSlowConfig(t0=4, slow_scales=(8, 16, 32)))
        factor_specs = list(readout_factor_specs) if readout_factor_specs is not None else (fastslow_readout_specs() if use_fastslow_readout else [])
        self.readout = CausalFactorReadout(
            factor_specs=factor_specs,
            feature_cfg=readout_feature_cfg,
            identifier_kind=readout_identifier_kind,
            fastslow_cfg=self.fs_cfg,
        )
        self.delay = _DelayBuilder(cfg.n_delays, cfg.stride)

        self.mu_ = 0.0
        self.std_ = 1.0
        self.W = None
        self.Win = None
        self.bias = None
        self.Wout = None
        self.ngrc_dim_: int = 0

    def _setup(self) -> None:
        tpl = self.template_factory.get(self.cfg.n_reservoir, 1, self.cfg.sparsity)
        self.W = tpl['W_unit'] * self.cfg.spectral_radius
        self.Win = tpl['Win_base'] * self.cfg.input_scale
        self.bias = tpl['bias']

    def _step(self, r: np.ndarray, u: float) -> np.ndarray:
        pre = self.W.dot(r) + self.Win[:, 0] * float(u) + self.bias
        cand = np.tanh(pre)
        return (1.0 - self.cfg.leak_rate) * r + self.cfg.leak_rate * cand

    def _ngrc_row(
        self,
        delay_row: np.ndarray,
        readout_state: Optional[ReadoutState] = None,
        factor_mat: Optional[np.ndarray] = None,
        t: Optional[int] = None,
    ) -> np.ndarray:
        delay_row = np.asarray(safe_clip(delay_row, self.cfg.feature_clip), dtype=float).reshape(1, -1)
        Phi, _ = build_poly_library(delay_row, [f'z{i}' for i in range(delay_row.shape[1])], poly_order=self.cfg.poly_order)
        row = Phi.reshape(-1)
        if self.readout.dim > 0:
            if factor_mat is not None and t is not None:
                extra = np.asarray(factor_mat[t], dtype=float)
            elif readout_state is not None:
                extra = self.readout.factor_step(readout_state.context)
            else:
                raise ValueError('factor readout requested but no state provided')
            row = np.concatenate([row, extra])
        row = np.asarray(safe_clip(row, self.cfg.feature_clip), dtype=float)
        return row

    def _aug(
        self,
        r: np.ndarray,
        delay_row: np.ndarray,
        readout_state: Optional[ReadoutState] = None,
        factor_mat: Optional[np.ndarray] = None,
        t: Optional[int] = None,
    ) -> np.ndarray:
        ngrc_row = self._ngrc_row(delay_row, readout_state=readout_state, factor_mat=factor_mat, t=t)
        return np.concatenate([r, ngrc_row])

    def fit(self, y_train: np.ndarray) -> 'HybridRCNGRCModel':
        self.mu_ = float(np.mean(y_train))
        self.std_ = float(np.std(y_train) + 1e-12)
        ys = ((np.asarray(y_train, dtype=float).reshape(-1) - self.mu_) / self.std_).astype(float)
        self._setup()
        _, factor_mat = self.readout.fit_transform(ys)
        r = np.zeros(self.cfg.n_reservoir, dtype=float)
        X_rows, Y = [], []
        start = max(self.cfg.washout, self.delay.max_lag)
        for t in range(len(ys) - 1):
            r = self._step(r, ys[t])
            if t >= start:
                delay_row = self.delay.row_from_series(ys, t)
                aug = self._aug(r, delay_row, factor_mat=factor_mat, t=t)
                X_rows.append(aug)
                Y.append(float(ys[t + 1]))
        X = np.vstack(X_rows)
        Y = np.asarray(Y, dtype=float)
        self.Wout = ridge_solve(X, Y, self.cfg.ridge)
        self.ngrc_dim_ = int(X.shape[1] - self.cfg.n_reservoir)
        return self

    def rollout(self, y_hist: np.ndarray, horizon: int) -> np.ndarray:
        ys = ((np.asarray(y_hist, dtype=float).reshape(-1) - self.mu_) / self.std_).astype(float)
        r = np.zeros(self.cfg.n_reservoir, dtype=float)
        for v in ys:
            r = self._step(r, float(v))
        hist = deque([float(v) for v in ys], maxlen=max(len(ys), self.delay.max_lag + 1))
        readout_state = self.readout.warmup(ys) if self.readout.dim > 0 else None
        preds = []
        for _ in range(horizon):
            delay_row = self.delay.row_from_deque(hist)
            aug = self._aug(r, delay_row, readout_state=readout_state)
            y_next = float(safe_clip(aug @ self.Wout, self.cfg.y_clip))
            preds.append(y_next)
            r = self._step(r, y_next)
            hist.append(y_next)
            if readout_state is not None:
                self.readout.advance(readout_state, y_next)
        return np.asarray(preds, dtype=float) * self.std_ + self.mu_

    def one_step_metrics(self, series: np.ndarray, burn_in: int):
        ys = ((np.asarray(series, dtype=float).reshape(-1) - self.mu_) / self.std_).astype(float)
        _, factor_mat = self.readout.transform(ys)
        r = np.zeros(self.cfg.n_reservoir, dtype=float)
        preds, truth = [], []
        start = max(int(burn_in), self.cfg.washout, self.delay.max_lag)
        for t in range(len(ys) - 1):
            r = self._step(r, ys[t])
            if t >= start:
                delay_row = self.delay.row_from_series(ys, t)
                aug = self._aug(r, delay_row, factor_mat=factor_mat, t=t)
                pred_std = float(safe_clip(aug @ self.Wout, self.cfg.y_clip))
                preds.append(pred_std * self.std_ + self.mu_)
                truth.append(ys[t + 1] * self.std_ + self.mu_)
        return {f'one_step_{k}': v for k, v in metric_dict(np.asarray(truth), np.asarray(preds)).items()}

    def count_total_params(self) -> int:
        return int(self.cfg.n_reservoir + self.cfg.n_reservoir * self.cfg.n_reservoir + self.cfg.n_reservoir + self.ngrc_dim_ + self.cfg.n_reservoir)

    def count_trained_params(self) -> int:
        return int(self.cfg.n_reservoir + self.ngrc_dim_)

    def effective_dim(self) -> int:
        return int(self.cfg.n_reservoir + self.ngrc_dim_)


class SlowSINDyDeltaNGRCModel(BaseForecastModel):
    def __init__(self, slow_cfg: SlowSINDyConfig, cfg: ResidualNGRCConfig):
        self.slow_cfg = slow_cfg
        self.cfg = cfg
        self.encoder = CausalFastSlowEncoder(slow_cfg.fs_cfg)
        self.backbone = SlowBackboneSINDy(slow_cfg)
        self.delay = _DelayBuilder(cfg.n_delays, cfg.stride)
        self.mu_ = 0.0
        self.std_ = 1.0
        self.coef_: Optional[np.ndarray] = None
        self.n_features_: int = 0

    def _standardize(self, y: np.ndarray) -> np.ndarray:
        return ((np.asarray(y, dtype=float).reshape(-1) - self.mu_) / self.std_).astype(float)

    def _feature_row(self, resid_delay: np.ndarray, slow_t: float, fast_t: float, m_t: float, ds_pred: float) -> np.ndarray:
        base = np.asarray(safe_clip(resid_delay, self.cfg.feature_clip), dtype=float).reshape(1, -1)
        Phi, _ = build_poly_library(base, [f'r{i}' for i in range(base.shape[1])], poly_order=self.cfg.poly_order)
        row = np.concatenate([Phi.reshape(-1), np.array([slow_t, fast_t, m_t, ds_pred], dtype=float)])
        return np.asarray(safe_clip(row, self.cfg.feature_clip), dtype=float)

    def fit(self, y_train: np.ndarray) -> 'SlowSINDyDeltaNGRCModel':
        self.mu_ = float(np.mean(y_train))
        self.std_ = float(np.std(y_train) + 1e-12)
        ys = self._standardize(y_train)
        feats = self.encoder.build_feature_sequence(ys)
        self.backbone.fit_on_standardized(ys)
        resid = ys - feats['slow']
        X_rows, Y = [], []
        start = max(self.cfg.washout, self.delay.max_lag)
        for t in range(start, len(ys) - 1):
            slow_t = float(feats['slow'][t])
            ds_t = float(feats['ds'][t])
            slow_next = self.backbone.predict_next(slow_t, ds_t)
            ds_pred = slow_next - slow_t
            resid_t = float(resid[t])
            resid_next_true = float(ys[t + 1] - slow_next)
            delta = float(safe_clip(resid_next_true - resid_t, self.cfg.delta_clip))
            resid_delay = self.delay.row_from_series(resid, t)
            x = self._feature_row(resid_delay, slow_t, float(feats['fast'][t]), float(feats['m'][t]), ds_pred)
            X_rows.append(x)
            Y.append(delta)
        X = np.vstack(X_rows)
        self.coef_ = ridge_solve(X, np.asarray(Y, dtype=float), self.cfg.ridge)
        self.n_features_ = int(X.shape[1])
        return self

    def rollout(self, y_hist: np.ndarray, horizon: int) -> np.ndarray:
        ys = self._standardize(y_hist)
        feats = self.encoder.build_feature_sequence(ys)
        fast_state = self.encoder.init_fast_state(ys)
        slow_cur = float(feats['slow'][-1])
        ds_cur = float(feats['ds'][-1])
        resid_hist = ys - feats['slow']
        resid_cur = float(resid_hist[-1])
        resid_deque = deque([float(v) for v in resid_hist], maxlen=max(len(resid_hist), self.delay.max_lag + 1))
        preds = []
        for _ in range(horizon):
            slow_next = self.backbone.predict_next(slow_cur, ds_cur)
            ds_pred = slow_next - slow_cur
            fast_cur = float(fast_state['f2'])
            m_cur = float(fast_cur - slow_cur)
            resid_delay = self.delay.row_from_deque(resid_deque)
            x = self._feature_row(resid_delay, slow_cur, fast_cur, m_cur, ds_pred)
            delta = float(safe_clip(x @ self.coef_, self.cfg.delta_clip))
            resid_next = float(safe_clip(resid_cur + self.cfg.damp * delta, self.cfg.resid_clip))
            y_next = float(safe_clip(slow_next + resid_next, self.cfg.y_clip))
            preds.append(y_next)
            resid_deque.append(resid_next)
            self.encoder.advance_fast(fast_state, y_next)
            slow_cur, ds_cur, resid_cur = slow_next, ds_pred, resid_next
        return np.asarray(preds, dtype=float) * self.std_ + self.mu_

    def one_step_metrics(self, series: np.ndarray, burn_in: int):
        ys = self._standardize(series)
        feats = self.encoder.build_feature_sequence(ys)
        resid = ys - feats['slow']
        preds, truth = [], []
        start = max(int(burn_in), self.cfg.washout, self.delay.max_lag)
        for t in range(start, len(ys) - 1):
            slow_t = float(feats['slow'][t])
            ds_t = float(feats['ds'][t])
            slow_next = self.backbone.predict_next(slow_t, ds_t)
            ds_pred = slow_next - slow_t
            resid_t = float(resid[t])
            resid_delay = self.delay.row_from_series(resid, t)
            x = self._feature_row(resid_delay, slow_t, float(feats['fast'][t]), float(feats['m'][t]), ds_pred)
            delta = float(safe_clip(x @ self.coef_, self.cfg.delta_clip))
            resid_next = float(safe_clip(resid_t + self.cfg.damp * delta, self.cfg.resid_clip))
            pred_std = float(safe_clip(slow_next + resid_next, self.cfg.y_clip))
            preds.append(pred_std * self.std_ + self.mu_)
            truth.append(ys[t + 1] * self.std_ + self.mu_)
        return {f'one_step_{k}': v for k, v in metric_dict(np.asarray(truth), np.asarray(preds)).items()}

    def count_total_params(self) -> int:
        return self.backbone.count_params() + int(self.n_features_)

    def count_trained_params(self) -> int:
        return self.backbone.count_params() + int(self.n_features_)

    def effective_dim(self) -> int:
        return self.backbone.count_params() + int(self.n_features_)


class SlowSINDyDeltaHybridModel(BaseForecastModel):
    def __init__(self, slow_cfg: SlowSINDyConfig, cfg: ResidualRCNGRCConfig, template_factory: ReservoirTemplateFactory):
        self.slow_cfg = slow_cfg
        self.cfg = cfg
        self.template_factory = template_factory
        self.encoder = CausalFastSlowEncoder(slow_cfg.fs_cfg)
        self.backbone = SlowBackboneSINDy(slow_cfg)
        self.delay = _DelayBuilder(cfg.n_delays, cfg.stride)

        self.mu_ = 0.0
        self.std_ = 1.0
        self.W = None
        self.Win = None
        self.bias = None
        self.Wout = None
        self.ngrc_dim_ = 0

    def _setup(self) -> None:
        tpl = self.template_factory.get(self.cfg.n_reservoir, 1, self.cfg.sparsity)
        self.W = tpl["W_unit"] * self.cfg.spectral_radius
        self.Win = tpl["Win_base"] * self.cfg.input_scale
        self.bias = tpl["bias"]

    def _step(self, r: np.ndarray, resid_t: float) -> np.ndarray:
        pre = self.W.dot(r) + self.Win[:, 0] * float(resid_t) + self.bias
        cand = np.tanh(pre)
        return (1.0 - self.cfg.leak_rate) * r + self.cfg.leak_rate * cand

    def _standardize(self, y: np.ndarray) -> np.ndarray:
        return ((np.asarray(y, dtype=float).reshape(-1) - self.mu_) / self.std_).astype(float)

    def _ngrc_row(self, resid_delay: np.ndarray, slow_t: float, fast_t: float, m_t: float, ds_pred: float) -> np.ndarray:
        base = np.asarray(safe_clip(resid_delay, self.cfg.feature_clip), dtype=float).reshape(1, -1)
        Phi, _ = build_poly_library(base, [f"r{i}" for i in range(base.shape[1])], poly_order=self.cfg.poly_order)
        row = np.concatenate([Phi.reshape(-1), np.array([slow_t, fast_t, m_t, ds_pred], dtype=float)])
        return np.asarray(safe_clip(row, self.cfg.feature_clip), dtype=float)

    def _aug(self, r: np.ndarray, resid_delay: np.ndarray, slow_t: float, fast_t: float, m_t: float, ds_pred: float) -> np.ndarray:
        ngrc_row = self._ngrc_row(resid_delay, slow_t, fast_t, m_t, ds_pred)
        return np.concatenate([r, ngrc_row])

    def fit(self, y_train: np.ndarray) -> "SlowSINDyDeltaHybridModel":
        self.mu_ = float(np.mean(y_train))
        self.std_ = float(np.std(y_train) + 1e-12)
        ys = self._standardize(y_train)
        feats = self.encoder.build_feature_sequence(ys)
        self.backbone.fit_on_standardized(ys)
        resid = ys - feats["slow"]
        self._setup()
        r = np.zeros(self.cfg.n_reservoir, dtype=float)
        X_rows, Y = [], []
        start = max(self.cfg.washout, self.delay.max_lag)
        for t in range(len(ys) - 1):
            resid_t = float(resid[t])
            r = self._step(r, resid_t)
            if t >= start:
                slow_t = float(feats["slow"][t])
                ds_t = float(feats["ds"][t])
                slow_next = self.backbone.predict_next(slow_t, ds_t)
                ds_pred = slow_next - slow_t
                resid_next_true = float(ys[t + 1] - slow_next)
                delta = float(safe_clip(resid_next_true - resid_t, self.cfg.delta_clip))
                resid_delay = self.delay.row_from_series(resid, t)
                aug = self._aug(r, resid_delay, slow_t, float(feats["fast"][t]), float(feats["m"][t]), ds_pred)
                X_rows.append(aug)
                Y.append(delta)
        X = np.vstack(X_rows)
        self.Wout = ridge_solve(X, np.asarray(Y, dtype=float), self.cfg.ridge)
        self.ngrc_dim_ = int(X.shape[1] - self.cfg.n_reservoir)
        return self

    def rollout(self, y_hist: np.ndarray, horizon: int) -> np.ndarray:
        ys = self._standardize(y_hist)
        feats = self.encoder.build_feature_sequence(ys)
        fast_state = self.encoder.init_fast_state(ys)
        resid_hist = ys - feats["slow"]
        resid_deque = deque([float(v) for v in resid_hist], maxlen=max(len(resid_hist), self.delay.max_lag + 1))
        r = np.zeros(self.cfg.n_reservoir, dtype=float)
        for val in resid_hist:
            r = self._step(r, float(val))
        slow_cur = float(feats["slow"][-1])
        ds_cur = float(feats["ds"][-1])
        resid_cur = float(resid_hist[-1])
        preds = []
        for _ in range(horizon):
            slow_next = self.backbone.predict_next(slow_cur, ds_cur)
            ds_pred = slow_next - slow_cur
            fast_cur = float(fast_state["f2"])
            m_cur = float(fast_cur - slow_cur)
            resid_delay = self.delay.row_from_deque(resid_deque)
            aug = self._aug(r, resid_delay, slow_cur, fast_cur, m_cur, ds_pred)
            delta = float(safe_clip(aug @ self.Wout, self.cfg.delta_clip))
            resid_next = float(safe_clip(resid_cur + self.cfg.damp * delta, self.cfg.resid_clip))
            y_next = float(safe_clip(slow_next + resid_next, self.cfg.y_clip))
            preds.append(y_next)
            resid_deque.append(resid_next)
            self.encoder.advance_fast(fast_state, y_next)
            r = self._step(r, resid_next)
            slow_cur, ds_cur, resid_cur = slow_next, ds_pred, resid_next
        return np.asarray(preds, dtype=float) * self.std_ + self.mu_

    def one_step_metrics(self, series: np.ndarray, burn_in: int):
        ys = self._standardize(series)
        feats = self.encoder.build_feature_sequence(ys)
        resid = ys - feats["slow"]
        r = np.zeros(self.cfg.n_reservoir, dtype=float)
        preds, truth = [], []
        start = max(int(burn_in), self.cfg.washout, self.delay.max_lag)
        for t in range(len(ys) - 1):
            resid_t = float(resid[t])
            r = self._step(r, resid_t)
            if t >= start:
                slow_t = float(feats["slow"][t])
                ds_t = float(feats["ds"][t])
                slow_next = self.backbone.predict_next(slow_t, ds_t)
                ds_pred = slow_next - slow_t
                resid_delay = self.delay.row_from_series(resid, t)
                aug = self._aug(r, resid_delay, slow_t, float(feats["fast"][t]), float(feats["m"][t]), ds_pred)
                delta = float(safe_clip(aug @ self.Wout, self.cfg.delta_clip))
                resid_next = float(safe_clip(resid_t + self.cfg.damp * delta, self.cfg.resid_clip))
                pred_std = float(safe_clip(slow_next + resid_next, self.cfg.y_clip))
                preds.append(pred_std * self.std_ + self.mu_)
                truth.append(ys[t + 1] * self.std_ + self.mu_)
        return {f"one_step_{k}": v for k, v in metric_dict(np.asarray(truth), np.asarray(preds)).items()}

    def count_total_params(self) -> int:
        return self.backbone.count_params() + int(
            self.cfg.n_reservoir + self.cfg.n_reservoir * self.cfg.n_reservoir + self.cfg.n_reservoir + self.ngrc_dim_ + self.cfg.n_reservoir
        )

    def count_trained_params(self) -> int:
        return self.backbone.count_params() + int(self.cfg.n_reservoir + self.ngrc_dim_)

    def effective_dim(self) -> int:
        return self.backbone.count_params() + int(self.cfg.n_reservoir + self.ngrc_dim_)
