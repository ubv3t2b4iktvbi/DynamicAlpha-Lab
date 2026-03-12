from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from ..factors.base import DynamicsFeatureConfig, FactorSpec
from ..factors.readout import CausalFactorReadout, ReadoutState
from ..factors.repository import fastslow_readout_specs
from ..fastslow import FastSlowConfig
from ..metrics import metric_dict
from ..utils import ridge_solve
from .base import BaseForecastModel


@dataclass
class RCConfig:
    n_reservoir: int = 200
    spectral_radius: float = 1.0
    input_scale: float = 0.5
    leak_rate: float = 1.0
    ridge: float = 1e-5
    sparsity: float = 0.05
    washout: int = 100
    fs_cfg: Optional[FastSlowConfig] = None


class ReservoirTemplateFactory:
    def __init__(self, seed: int):
        self.seed = seed
        self.cache: Dict[tuple[int, int, float], Dict[str, Any]] = {}

    def _estimate_radius(self, W: sparse.csr_matrix) -> float:
        try:
            eig = spla.eigs(W.astype(np.float64), k=1, which="LM", return_eigenvectors=False, maxiter=2000, tol=1e-2)
            rad = float(np.abs(eig[0]))
            if np.isfinite(rad) and rad > 1e-8:
                return rad
        except Exception:
            pass
        rng = np.random.default_rng(self.seed)
        x = rng.normal(size=W.shape[0])
        x /= np.linalg.norm(x) + 1e-12
        for _ in range(40):
            x = W.dot(x)
            nrm = np.linalg.norm(x) + 1e-12
            x = x / nrm
        Wx = W.dot(x)
        rad = float(np.linalg.norm(Wx) / (np.linalg.norm(x) + 1e-12))
        return max(rad, 1e-8)

    def get(self, n_reservoir: int, input_dim: int, sparsity: float) -> Dict[str, Any]:
        key = (n_reservoir, input_dim, float(sparsity))
        if key in self.cache:
            return self.cache[key]
        rng = np.random.default_rng(self.seed + 1000 * n_reservoir + 31 * input_dim)
        nnz = max(int(n_reservoir * n_reservoir * sparsity), n_reservoir)
        rows = rng.integers(0, n_reservoir, size=nnz)
        cols = rng.integers(0, n_reservoir, size=nnz)
        data = rng.uniform(-1.0, 1.0, size=nnz)
        W = sparse.coo_matrix((data, (rows, cols)), shape=(n_reservoir, n_reservoir)).tocsr()
        W = W + sparse.eye(n_reservoir, format="csr") * 0.01
        rad = self._estimate_radius(W)
        W_unit = W * (1.0 / rad)
        Win = rng.uniform(-1.0, 1.0, size=(n_reservoir, input_dim))
        bias = rng.uniform(-0.1, 0.1, size=n_reservoir)
        out = {"W_unit": W_unit.tocsr(), "Win_base": Win, "bias": bias}
        self.cache[key] = out
        return out


class PureRCModel(BaseForecastModel):
    def __init__(
        self,
        cfg: RCConfig,
        template_factory: ReservoirTemplateFactory,
        fs_cfg: Optional[FastSlowConfig] = None,
        use_fastslow_readout: bool = False,
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

        self.mu_ = 0.0
        self.std_ = 1.0
        self.W: Optional[sparse.csr_matrix] = None
        self.Win: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None
        self.Wout: Optional[np.ndarray] = None

    def _setup(self) -> None:
        tpl = self.template_factory.get(self.cfg.n_reservoir, 1, self.cfg.sparsity)
        self.W = tpl["W_unit"] * self.cfg.spectral_radius
        self.Win = tpl["Win_base"] * self.cfg.input_scale
        self.bias = tpl["bias"]

    def _step(self, r: np.ndarray, u: float) -> np.ndarray:
        pre = self.W.dot(r) + self.Win[:, 0] * float(u) + self.bias
        cand = np.tanh(pre)
        return (1.0 - self.cfg.leak_rate) * r + self.cfg.leak_rate * cand

    def _readout_aug(
        self,
        r: np.ndarray,
        y_t: float,
        factor_mat: Optional[np.ndarray] = None,
        t: Optional[int] = None,
        readout_state: Optional[ReadoutState] = None,
    ) -> np.ndarray:
        parts = [r, np.array([y_t], dtype=float)]
        if self.readout.dim > 0:
            if factor_mat is not None and t is not None:
                parts.append(np.asarray(factor_mat[t], dtype=float))
            elif readout_state is not None:
                parts.append(self.readout.factor_step(readout_state.context))
            else:
                raise ValueError("factor readout requires either sequence features or causal state")
        parts.append(np.array([1.0], dtype=float))
        return np.concatenate(parts)

    def fit(self, y_train: np.ndarray) -> "PureRCModel":
        self.mu_ = float(np.mean(y_train))
        self.std_ = float(np.std(y_train) + 1e-12)
        ys = ((y_train - self.mu_) / self.std_).reshape(-1)
        self._setup()
        _, factor_mat = self.readout.fit_transform(ys)
        r = np.zeros(self.cfg.n_reservoir, dtype=float)
        X_rows = []
        Y = []
        for t in range(len(ys) - 1):
            r = self._step(r, ys[t])
            if t >= self.cfg.washout:
                X_rows.append(self._readout_aug(r, ys[t], factor_mat=factor_mat, t=t))
                Y.append(ys[t + 1])
        X = np.vstack(X_rows)
        Y = np.asarray(Y, dtype=float)
        self.Wout = ridge_solve(X, Y, self.cfg.ridge)
        return self

    def rollout(self, y_hist: np.ndarray, horizon: int) -> np.ndarray:
        y_hist_std = ((np.asarray(y_hist, dtype=float).reshape(-1) - self.mu_) / self.std_).astype(float)
        r = np.zeros(self.cfg.n_reservoir, dtype=float)
        readout_state = self.readout.warmup(y_hist_std) if self.readout.dim > 0 else None
        for v in y_hist_std:
            r = self._step(r, float(v))
        y_cur = float(y_hist_std[-1])
        preds = []
        for _ in range(horizon):
            aug = self._readout_aug(r, y_cur, readout_state=readout_state)
            y_next = float(aug @ self.Wout)
            preds.append(y_next)
            r = self._step(r, y_next)
            y_cur = y_next
            if readout_state is not None:
                self.readout.advance(readout_state, y_next)
        preds = np.asarray(preds, dtype=float)
        return preds * self.std_ + self.mu_

    def one_step_metrics(self, series: np.ndarray, burn_in: int):
        ys = ((np.asarray(series, dtype=float).reshape(-1) - self.mu_) / self.std_).astype(float)
        _, factor_mat = self.readout.transform(ys)
        r = np.zeros(self.cfg.n_reservoir, dtype=float)
        preds = []
        truth = []
        start_eval = min(max(int(burn_in), self.cfg.washout), len(ys) - 2)
        for t in range(len(ys) - 1):
            r = self._step(r, ys[t])
            if t >= start_eval:
                pred_std = float(self._readout_aug(r, ys[t], factor_mat=factor_mat, t=t) @ self.Wout)
                preds.append(pred_std * self.std_ + self.mu_)
                truth.append(ys[t + 1] * self.std_ + self.mu_)
        return {f"one_step_{k}": v for k, v in metric_dict(np.asarray(truth), np.asarray(preds)).items()}

    def count_total_params(self) -> int:
        extra = 2 + self.readout.dim
        return int(self.cfg.n_reservoir + self.cfg.n_reservoir * self.cfg.n_reservoir + self.cfg.n_reservoir + (self.cfg.n_reservoir + extra))

    def count_trained_params(self) -> int:
        extra = 2 + self.readout.dim
        return int(self.cfg.n_reservoir + extra)

    def effective_dim(self) -> int:
        return int(self.cfg.n_reservoir)
