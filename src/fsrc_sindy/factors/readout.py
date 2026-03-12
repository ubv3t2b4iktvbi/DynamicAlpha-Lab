from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..fastslow import CausalFastSlowEncoder, FastSlowConfig
from .base import DynamicsFeatureConfig, FactorSpec
from .factor_bank import evaluate_factor_array, evaluate_factor_step
from .feature_engine import DynamicsFeatureEngine
from .identifiers import BaseIdentifier, make_identifier


@dataclass
class ReadoutState:
    dynamics_state: Any | None
    fastslow_state: dict[str, Any] | None
    context: dict[str, float]


class CausalFactorReadout:
    """Unified causal encoder for both legacy fast/slow readout factors and mined factors."""

    def __init__(
        self,
        factor_specs: Sequence[FactorSpec] | None,
        *,
        feature_cfg: DynamicsFeatureConfig | None = None,
        identifier_kind: str | None = None,
        fastslow_cfg: FastSlowConfig | None = None,
    ):
        self.factor_specs = list(factor_specs or [])
        self.feature_cfg = feature_cfg if feature_cfg is not None else DynamicsFeatureConfig()
        self.identifier_kind = identifier_kind
        self.fastslow_cfg = fastslow_cfg if fastslow_cfg is not None else FastSlowConfig(t0=4, slow_scales=(8, 16, 32))
        self.engine = DynamicsFeatureEngine(self.feature_cfg)
        self.fastslow_encoder = CausalFastSlowEncoder(self.fastslow_cfg)
        self.identifier: BaseIdentifier | None = make_identifier(identifier_kind, self.feature_cfg) if identifier_kind else None

        required_inputs = {name for spec in self.factor_specs for name in spec.inputs}
        self._needs_fastslow = any(name.startswith("fs_") for name in required_inputs)
        self._needs_dynamics = bool(required_inputs - {"fs_fast", "fs_slow", "fs_m"}) or self.identifier is not None

    @property
    def dim(self) -> int:
        return len(self.factor_specs)

    def _fastslow_sequence_context(self, y_std: np.ndarray) -> dict[str, np.ndarray]:
        feats = self.fastslow_encoder.build_feature_sequence(y_std)
        return {
            "fs_fast": np.asarray(feats["fast"], dtype=float),
            "fs_slow": np.asarray(feats["slow"], dtype=float),
            "fs_m": np.asarray(feats["m"], dtype=float),
        }

    @staticmethod
    def _fastslow_step_context(state: Mapping[str, Any]) -> dict[str, float]:
        fast = float(state["f2"])
        slow = float(np.mean(np.asarray(state["slows"], dtype=float)))
        return {
            "fs_fast": fast,
            "fs_slow": slow,
            "fs_m": float(fast - slow),
        }

    def _merged_sequence_context(self, y_std: np.ndarray, *, fit_identifier: bool) -> dict[str, np.ndarray]:
        context: dict[str, np.ndarray] = {}
        if self._needs_dynamics:
            dyn_context = self.engine.build_base_sequence(y_std)
            if self.identifier is not None:
                if fit_identifier:
                    self.identifier.fit(dyn_context)
                dyn_context = self.engine.augment_with_identifier(dyn_context, self.identifier.batch_outputs(dyn_context))
            context.update({key: np.asarray(val, dtype=float) for key, val in dyn_context.items()})
        if self._needs_fastslow:
            context.update(self._fastslow_sequence_context(y_std))
        return context

    def fit_transform(self, y_std: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
        context = self._merged_sequence_context(y_std, fit_identifier=True)
        return context, self.factor_matrix(context)

    def transform(self, y_std: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
        context = self._merged_sequence_context(y_std, fit_identifier=False)
        return context, self.factor_matrix(context)

    def factor_matrix(self, context: Mapping[str, np.ndarray]) -> np.ndarray:
        if not self.factor_specs:
            length = len(next(iter(context.values()))) if context else 0
            return np.zeros((length, 0), dtype=float)
        cols = [evaluate_factor_array(spec, context).reshape(-1, 1) for spec in self.factor_specs]
        return np.hstack(cols) if cols else np.zeros((0, 0), dtype=float)

    def warmup(self, y_hist_std: np.ndarray) -> ReadoutState:
        context: dict[str, float] = {}
        dynamics_state = None
        fastslow_state = None
        if self._needs_dynamics:
            dynamics_state, dyn_ctx = self.engine.warmup_state(y_hist_std)
            if self.identifier is not None:
                dyn_ctx = {**dyn_ctx, **self.identifier.step_outputs(dyn_ctx)}
            context.update({key: float(val) for key, val in dyn_ctx.items()})
        if self._needs_fastslow:
            fastslow_state = self.fastslow_encoder.init_full_state(y_hist_std)
            context.update(self._fastslow_step_context(fastslow_state))
        return ReadoutState(
            dynamics_state=dynamics_state,
            fastslow_state=fastslow_state,
            context=context,
        )

    def advance(self, state: ReadoutState, y_new: float) -> dict[str, float]:
        context: dict[str, float] = {}
        if state.dynamics_state is not None:
            dyn_ctx = self.engine.step(state.dynamics_state, float(y_new))
            if self.identifier is not None:
                dyn_ctx = {**dyn_ctx, **self.identifier.step_outputs(dyn_ctx)}
            context.update({key: float(val) for key, val in dyn_ctx.items()})
        if state.fastslow_state is not None:
            self.fastslow_encoder.advance_full(state.fastslow_state, float(y_new))
            context.update(self._fastslow_step_context(state.fastslow_state))
        state.context = context
        return context

    def factor_step(self, context: Mapping[str, float]) -> np.ndarray:
        if not self.factor_specs:
            return np.zeros(0, dtype=float)
        return np.asarray([evaluate_factor_step(spec, context) for spec in self.factor_specs], dtype=float)
