from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from ..metrics import metric_dict
from ..models.base import BaseForecastModel
from ..models.rc import RCConfig, ReservoirTemplateFactory
from ..utils import ridge_solve
from .base import DynamicsFeatureConfig, FactorSpec
from .factor_bank import evaluate_factor_array, evaluate_factor_step
from .feature_engine import DynamicsFeatureEngine
from .identifiers import BaseIdentifier, make_identifier


@dataclass(frozen=True)
class ScreenScore:
    rmse: float
    nrmse: float


class ReservoirTeacherForcedScreen:
    """Fast one-step screen for candidate factors using a fixed RC state cache."""

    def __init__(
        self,
        cfg: RCConfig,
        template_factory: ReservoirTemplateFactory,
        ridge: float,
        train_len: int,
    ):
        self.cfg = cfg
        self.template_factory = template_factory
        self.ridge = ridge
        self.train_len = int(train_len)
        self.W = None
        self.Win = None
        self.bias = None

    def _setup(self) -> None:
        tpl = self.template_factory.get(self.cfg.n_reservoir, 1, self.cfg.sparsity)
        self.W = tpl["W_unit"] * self.cfg.spectral_radius
        self.Win = tpl["Win_base"] * self.cfg.input_scale
        self.bias = tpl["bias"]

    def _step(self, r: np.ndarray, u: float) -> np.ndarray:
        pre = self.W.dot(r) + self.Win[:, 0] * float(u) + self.bias
        cand = np.tanh(pre)
        return (1.0 - self.cfg.leak_rate) * r + self.cfg.leak_rate * cand

    def build_state_rows(self, y_std: np.ndarray) -> np.ndarray:
        if self.W is None:
            self._setup()
        r = np.zeros(self.cfg.n_reservoir, dtype=float)
        rows = []
        for t in range(len(y_std) - 1):
            r = self._step(r, float(y_std[t]))
            rows.append(r.copy())
        return np.vstack(rows)

    def fit_and_score(
        self,
        y_std: np.ndarray,
        states: np.ndarray,
        factor_columns: Sequence[np.ndarray] | None = None,
    ) -> ScreenScore:
        if factor_columns is None:
            factor_columns = []
        factor_columns = [np.asarray(col, dtype=float).reshape(-1) for col in factor_columns]
        n = len(y_std)
        if states.shape[0] != n - 1:
            raise ValueError("states rows must match len(y_std) - 1")
        all_indices = np.arange(n - 1)
        train_idx = all_indices[(all_indices >= self.cfg.washout) & (all_indices <= self.train_len - 2)]
        val_idx = all_indices[all_indices >= self.train_len - 1]
        y_feat = y_std[:-1]
        target = y_std[1:]
        ones = np.ones(n - 1, dtype=float)
        X_parts = [states, y_feat[:, None]]
        for col in factor_columns:
            X_parts.append(col[:-1, None] if len(col) == n else col[:, None])
        X_parts.append(ones[:, None])
        X = np.hstack(X_parts)
        coef = ridge_solve(X[train_idx], target[train_idx], self.ridge)
        pred_val = X[val_idx] @ coef
        truth_val = target[val_idx]
        rmse = float(np.sqrt(np.mean((truth_val - pred_val) ** 2)))
        nrmse = float(rmse / (np.std(truth_val) + 1e-12))
        return ScreenScore(rmse=rmse, nrmse=nrmse)


class FactorAugmentedRCModel(BaseForecastModel):
    def __init__(
        self,
        rc_cfg: RCConfig,
        factor_specs: Sequence[FactorSpec],
        identifier_kind: str,
        template_factory: ReservoirTemplateFactory,
        feature_cfg: DynamicsFeatureConfig,
    ):
        self.rc_cfg = rc_cfg
        self.factor_specs = list(factor_specs)
        self.identifier_kind = identifier_kind
        self.template_factory = template_factory
        self.feature_cfg = feature_cfg
        self.engine = DynamicsFeatureEngine(feature_cfg)
        self.identifier: BaseIdentifier = make_identifier(identifier_kind, feature_cfg)
        self.mu_ = 0.0
        self.std_ = 1.0
        self.W = None
        self.Win = None
        self.bias = None
        self.Wout = None

    def _setup(self) -> None:
        tpl = self.template_factory.get(self.rc_cfg.n_reservoir, 1, self.rc_cfg.sparsity)
        self.W = tpl["W_unit"] * self.rc_cfg.spectral_radius
        self.Win = tpl["Win_base"] * self.rc_cfg.input_scale
        self.bias = tpl["bias"]

    def _step(self, r: np.ndarray, u: float) -> np.ndarray:
        pre = self.W.dot(r) + self.Win[:, 0] * float(u) + self.bias
        cand = np.tanh(pre)
        return (1.0 - self.rc_cfg.leak_rate) * r + self.rc_cfg.leak_rate * cand

    def _factor_matrix(self, context: Mapping[str, np.ndarray]) -> np.ndarray:
        if not self.factor_specs:
            return np.zeros((len(next(iter(context.values()))), 0), dtype=float)
        cols = [evaluate_factor_array(spec, context).reshape(-1, 1) for spec in self.factor_specs]
        return np.hstack(cols)

    def _factor_step(self, ctx: Mapping[str, float]) -> np.ndarray:
        if not self.factor_specs:
            return np.zeros(0, dtype=float)
        return np.asarray([evaluate_factor_step(spec, ctx) for spec in self.factor_specs], dtype=float)

    def _fit_identifier_and_context(self, y_std: np.ndarray) -> dict[str, np.ndarray]:
        base = self.engine.build_base_sequence(y_std)
        self.identifier.fit(base)
        identifier_outputs = self.identifier.batch_outputs(base)
        return self.engine.augment_with_identifier(base, identifier_outputs)

    def _transform_context(self, y_std: np.ndarray) -> dict[str, np.ndarray]:
        base = self.engine.build_base_sequence(y_std)
        identifier_outputs = self.identifier.batch_outputs(base)
        return self.engine.augment_with_identifier(base, identifier_outputs)

    def fit(self, y_train: np.ndarray):
        self.mu_ = float(np.mean(y_train))
        self.std_ = float(np.std(y_train) + 1e-12)
        y_std = ((np.asarray(y_train, dtype=float).reshape(-1) - self.mu_) / self.std_).astype(float)
        context = self._fit_identifier_and_context(y_std)
        factor_mat = self._factor_matrix(context)
        self._setup()
        r = np.zeros(self.rc_cfg.n_reservoir, dtype=float)
        X_rows = []
        Y = []
        for t in range(len(y_std) - 1):
            r = self._step(r, y_std[t])
            if t >= self.rc_cfg.washout:
                parts = [r, np.array([y_std[t]], dtype=float)]
                if factor_mat.shape[1] > 0:
                    parts.append(factor_mat[t])
                parts.append(np.array([1.0], dtype=float))
                X_rows.append(np.concatenate(parts))
                Y.append(y_std[t + 1])
        X = np.vstack(X_rows)
        Y = np.asarray(Y, dtype=float)
        self.Wout = ridge_solve(X, Y, self.rc_cfg.ridge)
        return self

    def _warmup_context(self, y_hist_std: Iterable[float]) -> tuple[object, dict[str, float]]:
        state, ctx = self.engine.warmup_state(y_hist_std)
        identifier_ctx = self.identifier.step_outputs(ctx)
        ctx = {**ctx, **identifier_ctx}
        return state, ctx

    def rollout(self, y_hist: np.ndarray, horizon: int) -> np.ndarray:
        y_hist_std = ((np.asarray(y_hist, dtype=float).reshape(-1) - self.mu_) / self.std_).astype(float)
        self._setup() if self.W is None else None
        r = np.zeros(self.rc_cfg.n_reservoir, dtype=float)
        for v in y_hist_std:
            r = self._step(r, float(v))
        state, ctx = self._warmup_context(y_hist_std)
        y_cur = float(y_hist_std[-1])
        preds = []
        for _ in range(horizon):
            factor_vec = self._factor_step(ctx)
            parts = [r, np.array([y_cur], dtype=float)]
            if len(factor_vec) > 0:
                parts.append(factor_vec)
            parts.append(np.array([1.0], dtype=float))
            aug = np.concatenate(parts)
            y_next = float(aug @ self.Wout)
            preds.append(y_next)
            r = self._step(r, y_next)
            ctx = self.engine.step(state, y_next)
            ctx = {**ctx, **self.identifier.step_outputs(ctx)}
            y_cur = y_next
        return np.asarray(preds, dtype=float) * self.std_ + self.mu_

    def one_step_metrics(self, series: np.ndarray, burn_in: int):
        y_std = ((np.asarray(series, dtype=float).reshape(-1) - self.mu_) / self.std_).astype(float)
        context = self._transform_context(y_std)
        factor_mat = self._factor_matrix(context)
        self._setup() if self.W is None else None
        r = np.zeros(self.rc_cfg.n_reservoir, dtype=float)
        preds = []
        truth = []
        start = min(max(int(burn_in), self.rc_cfg.washout), len(y_std) - 2)
        for t in range(len(y_std) - 1):
            r = self._step(r, y_std[t])
            if t >= start:
                parts = [r, np.array([y_std[t]], dtype=float)]
                if factor_mat.shape[1] > 0:
                    parts.append(factor_mat[t])
                parts.append(np.array([1.0], dtype=float))
                pred_std = float(np.concatenate(parts) @ self.Wout)
                preds.append(pred_std * self.std_ + self.mu_)
                truth.append(y_std[t + 1] * self.std_ + self.mu_)
        return {f"one_step_{k}": v for k, v in metric_dict(np.asarray(truth), np.asarray(preds)).items()}

    def count_total_params(self) -> int:
        extra = 2 + len(self.factor_specs)
        return int(self.rc_cfg.n_reservoir + self.rc_cfg.n_reservoir * self.rc_cfg.n_reservoir + self.rc_cfg.n_reservoir + (self.rc_cfg.n_reservoir + extra))

    def count_trained_params(self) -> int:
        extra = 2 + len(self.factor_specs)
        return int(self.rc_cfg.n_reservoir + extra)

    def effective_dim(self) -> int:
        return int(self.rc_cfg.n_reservoir + len(self.factor_specs))
