from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ..attractor_prior import AttractorPrior, evaluate_coordinate_attractor_prior
from ..metrics import evaluate_horizons
from ..models.rc import RCConfig, ReservoirTemplateFactory
from ..selection import validation_score
from .base import CandidateScore, DynamicsFeatureConfig, FactorMiningConfig, FactorSpec, SelectedFactorLibrary
from .curation import curate_candidate_scores
from .factor_bank import build_factor_bank, evaluate_factor_array, factor_formula
from .feature_engine import DynamicsFeatureEngine
from .identifiers import make_identifier
from .property_analyzer import analyze_signal_properties, prioritize_factor_bank, render_property_summary, scalar_koopman_diagnostic
from .rc_proxy import FactorAugmentedRCModel, ReservoirTeacherForcedScreen
from .readout import CausalFactorReadout


@dataclass
class FactorMiningRunResult:
    task_name: str
    identifier_kind: str
    candidate_scores: list[CandidateScore]
    selected_library: SelectedFactorLibrary
    property_profile: dict[str, float]
    property_summary: str
    baseline_metrics: dict[str, float]
    final_metrics: dict[str, float]
    test_metrics: dict[str, float]
    selected_specs: list[FactorSpec]
    baseline_prior_metrics: dict[str, float]
    final_prior_metrics: dict[str, float]
    layered_library: dict[str, object]

    def summary_row(self) -> dict[str, object]:
        selected_names = {spec.name for spec in self.selected_specs}
        selected_koopman_scores = [row.koopman_score for row in self.candidate_scores if row.factor_name in selected_names]
        selected_wsga_scores = [row.wsga_epr_score for row in self.candidate_scores if row.factor_name in selected_names and np.isfinite(row.wsga_epr_score)]
        return {
            "task": self.task_name,
            "identifier_kind": self.identifier_kind,
            "mode": self.property_summary.split(";")[0],
            "num_candidates": len(self.candidate_scores),
            "num_selected": len(self.selected_specs),
            "selected_factors": "; ".join(spec.name for spec in self.selected_specs),
            "selected_koopman_score": float(np.mean(selected_koopman_scores)) if selected_koopman_scores else np.nan,
            "selected_wsga_epr_score": float(np.mean(selected_wsga_scores)) if selected_wsga_scores else np.nan,
            "baseline_rmse10": self.baseline_metrics.get("rmse@10", np.nan),
            "baseline_rmse50": self.baseline_metrics.get("rmse@50", np.nan),
            "final_rmse10": self.final_metrics.get("rmse@10", np.nan),
            "final_rmse50": self.final_metrics.get("rmse@50", np.nan),
            "test_rmse10": self.test_metrics.get("rmse@10", np.nan),
            "test_rmse50": self.test_metrics.get("rmse@50", np.nan),
            "baseline_wsga_epr_score": self.baseline_prior_metrics.get("wsga_epr_score", np.nan),
            "final_wsga_epr_score": self.final_prior_metrics.get("wsga_epr_score", np.nan),
            "validation_score": self.selected_library.validation_score,
            "rollout_validation_score": self.selected_library.rollout_validation_score,
            "core_factor_count": len(self.selected_library.library_layers.get("core", [])),
            "extended_factor_count": len(self.selected_library.library_layers.get("extended", [])),
            "holding_factor_count": len(self.selected_library.library_layers.get("holding", [])),
        }


class DynamicsFactorMiner:
    def __init__(
        self,
        mining_cfg: FactorMiningConfig,
        rc_cfg: RCConfig,
        feature_cfg: DynamicsFeatureConfig,
        template_factory: ReservoirTemplateFactory,
    ):
        self.mining_cfg = mining_cfg
        self.rc_cfg = rc_cfg
        self.feature_cfg = feature_cfg
        self.template_factory = template_factory
        self.engine = DynamicsFeatureEngine(feature_cfg)

    def _context_splits(
        self,
        y_train: np.ndarray,
        y_val: np.ndarray,
        identifier_kind: str,
    ) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], object]:
        mu = float(np.mean(y_train))
        std = float(np.std(y_train) + 1e-12)
        train_std = ((np.asarray(y_train, dtype=float).reshape(-1) - mu) / std).astype(float)
        full = np.concatenate([y_train, y_val])
        full_std = ((np.asarray(full, dtype=float).reshape(-1) - mu) / std).astype(float)
        train_context = self.engine.build_base_sequence(train_std)
        identifier = make_identifier(identifier_kind, self.feature_cfg)
        identifier.fit(train_context)
        full_context = self.engine.build_base_sequence(full_std)
        full_context = self.engine.augment_with_identifier(full_context, identifier.batch_outputs(full_context))
        train_context = self.engine.augment_with_identifier(train_context, identifier.batch_outputs(train_context))
        return train_std, train_context, full_std, full_context, identifier

    def _screen_candidates(
        self,
        y_train: np.ndarray,
        y_val: np.ndarray,
        full_std: np.ndarray,
        full_context: dict[str, np.ndarray],
        train_context: dict[str, np.ndarray],
        val_context: dict[str, np.ndarray],
        factor_bank: Sequence[FactorSpec],
        prior_weights: dict[str, float],
        attractor_prior: AttractorPrior | None,
        attractor_labels: np.ndarray | None,
        dt: float,
    ) -> tuple[ScreenedCandidates, float, dict[str, np.ndarray]]:
        screen = ReservoirTeacherForcedScreen(
            cfg=self.rc_cfg,
            template_factory=self.template_factory,
            ridge=self.mining_cfg.screening_ridge,
            train_len=len(y_train),
        )
        states = screen.build_state_rows(full_std)
        baseline_score = screen.fit_and_score(full_std, states, [])
        scored: list[CandidateScore] = []
        factor_columns: dict[str, np.ndarray] = {}
        for spec in factor_bank:
            factor_col = evaluate_factor_array(spec, full_context)
            factor_columns[spec.name] = factor_col
            metrics = screen.fit_and_score(full_std, states, [factor_col])
            koopman = scalar_koopman_diagnostic(
                evaluate_factor_array(spec, train_context),
                evaluate_factor_array(spec, val_context),
            )
            wsga_diag = evaluate_coordinate_attractor_prior(
                factor_col.reshape(-1, 1),
                attractor_labels[: len(factor_col)] if attractor_labels is not None else np.zeros(len(factor_col), dtype=int),
                attractor_prior,
                dt=dt,
            )
            prior_weight = float(prior_weights.get(spec.name, 0.0))
            screening_score = float(
                metrics.rmse
                - self.mining_cfg.property_weight_strength * prior_weight * baseline_score.rmse
                - self.mining_cfg.koopman_weight_strength * koopman["koopman_score"] * baseline_score.rmse
                - self.mining_cfg.epr_weight_strength * float(wsga_diag.get("wsga_epr_score", 0.0)) * baseline_score.rmse
            )
            scored.append(
                CandidateScore(
                    identifier_kind="",
                    factor_name=spec.name,
                    family=spec.family,
                    one_step_rmse=metrics.rmse,
                    one_step_nrmse=metrics.nrmse,
                    gain_vs_baseline=baseline_score.rmse - metrics.rmse,
                    prior_weight=prior_weight,
                    koopman_lambda=koopman["koopman_lambda"],
                    koopman_rmse=koopman["koopman_rmse"],
                    koopman_score=koopman["koopman_score"],
                    screening_score=screening_score,
                    formula=factor_formula(spec),
                    finance_origin=spec.finance_origin,
                    dynamics_meaning=spec.dynamics_meaning,
                    wsga_epr_loss=float(wsga_diag.get("wsga_epr_loss", np.nan)),
                    wsga_epr_score=float(wsga_diag.get("wsga_epr_score", np.nan)),
                    wsga_basin_sep_gap=float(wsga_diag.get("wsga_basin_sep_gap", np.nan)),
                    theory_tags=spec.theory_tags,
                    source=spec.source,
                    default_tier=spec.default_tier,
                    manifold_role=spec.manifold_role,
                )
            )
        scored.sort(key=lambda row: (row.screening_score, row.one_step_rmse, -row.gain_vs_baseline))
        for rank, row in enumerate(scored, start=1):
            row.rank = rank
        return ScreenedCandidates(scored), baseline_score.rmse, factor_columns

    def _validation_context(self, y_train: np.ndarray, y_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        max_h = int(max(self.mining_cfg.score_horizons))
        context_len = min(self.mining_cfg.context_len, len(y_train) - 1, max(50, 4 * max_h))
        context = np.concatenate([y_train[-context_len:], y_val[:1]])
        future = y_val[1:1 + max_h]
        if len(future) < max_h:
            raise ValueError("validation sequence is too short for requested score horizons")
        return context, future

    def _test_context(self, y_val: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        max_h = int(max(self.mining_cfg.score_horizons))
        context_len = min(self.mining_cfg.context_len, len(y_val) - 1, max(50, 4 * max_h))
        context = np.concatenate([y_val[-context_len:], y_test[:1]])
        future = y_test[1:1 + max_h]
        if len(future) < max_h:
            raise ValueError("test sequence is too short for requested score horizons")
        return context, future

    def _score_model(self, model: FactorAugmentedRCModel, y_train: np.ndarray, y_val: np.ndarray) -> dict[str, float]:
        context, future = self._validation_context(y_train, y_val)
        metrics = evaluate_horizons(model, context, future, self.mining_cfg.score_horizons)
        metrics.update(model.one_step_metrics(np.concatenate([y_train[-len(context):], y_val]), burn_in=len(context)))
        return metrics

    def _prior_metrics_for_specs(
        self,
        *,
        y_train: np.ndarray,
        y_val: np.ndarray,
        identifier_kind: str,
        factor_specs: Sequence[FactorSpec],
        attractor_prior: AttractorPrior | None,
        attractor_labels: np.ndarray | None,
        dt: float,
        cache: dict[tuple[str, ...], dict[str, float]],
    ) -> dict[str, float]:
        key = tuple(spec.name for spec in factor_specs)
        if key in cache:
            return cache[key]
        if attractor_prior is None or attractor_labels is None:
            metrics = {
                "wsga_epr_loss": float("nan"),
                "wsga_epr_score": float("nan"),
                "wsga_basin_sep_gap": float("nan"),
            }
            cache[key] = metrics
            return metrics
        mu = float(np.mean(y_train))
        std = float(np.std(y_train) + 1e-12)
        full_std = ((np.asarray(np.concatenate([y_train, y_val]), dtype=float).reshape(-1) - mu) / std).astype(float)
        if factor_specs:
            readout = CausalFactorReadout(
                factor_specs=factor_specs,
                feature_cfg=self.feature_cfg,
                identifier_kind=identifier_kind,
            )
            _, coord = readout.fit_transform(full_std)
        else:
            coord = full_std.reshape(-1, 1)
        metrics = evaluate_coordinate_attractor_prior(
            coord,
            attractor_labels[: len(coord)],
            attractor_prior,
            dt=dt,
        )
        cache[key] = metrics
        return metrics

    def _selection_objective(
        self,
        *,
        rollout_metrics: dict[str, float],
        prior_metrics: dict[str, float],
        y_scale: float,
    ) -> tuple[float, float]:
        rollout_score = validation_score(
            rollout_metrics,
            score_horizons=self.mining_cfg.score_horizons,
            y_scale=y_scale,
        )
        total_score = float(rollout_score)
        if self.mining_cfg.use_wsga_prior and self.mining_cfg.epr_weight_strength > 0.0:
            epr_score = float(prior_metrics.get("wsga_epr_score", np.nan))
            if np.isfinite(epr_score):
                total_score += self.mining_cfg.epr_weight_strength * (1.0 - epr_score)
            else:
                total_score += self.mining_cfg.epr_weight_strength
        return total_score, float(rollout_score)

    def _baseline_and_selection(
        self,
        identifier_kind: str,
        candidate_specs: Sequence[FactorSpec],
        y_train: np.ndarray,
        y_val: np.ndarray,
        task_name: str,
        attractor_prior: AttractorPrior | None,
        attractor_labels: np.ndarray | None,
        dt: float,
    ) -> tuple[list[FactorSpec], dict[str, float], dict[str, float], float, float, dict[str, float], dict[str, float]]:
        baseline_model = FactorAugmentedRCModel(
            rc_cfg=self.rc_cfg,
            factor_specs=[],
            identifier_kind=identifier_kind,
            template_factory=self.template_factory,
            feature_cfg=self.feature_cfg,
        ).fit(y_train)
        baseline_metrics = self._score_model(baseline_model, y_train, y_val)
        y_scale = float(np.std(y_train) + 1e-12)
        prior_cache: dict[tuple[str, ...], dict[str, float]] = {}
        baseline_prior_metrics = self._prior_metrics_for_specs(
            y_train=y_train,
            y_val=y_val,
            identifier_kind=identifier_kind,
            factor_specs=[],
            attractor_prior=attractor_prior,
            attractor_labels=attractor_labels,
            dt=dt,
            cache=prior_cache,
        )
        current_score, current_rollout_score = self._selection_objective(
            rollout_metrics=baseline_metrics,
            prior_metrics=baseline_prior_metrics,
            y_scale=y_scale,
        )
        current_metrics = baseline_metrics
        current_prior_metrics = baseline_prior_metrics
        selected: list[FactorSpec] = []
        remaining = list(candidate_specs)
        while remaining and len(selected) < self.mining_cfg.max_selected_factors:
            best_spec = None
            best_metrics = None
            best_prior_metrics = None
            best_score = current_score
            best_rollout_score = current_rollout_score
            for spec in remaining:
                trial_specs = selected + [spec]
                model = FactorAugmentedRCModel(
                    rc_cfg=self.rc_cfg,
                    factor_specs=trial_specs,
                    identifier_kind=identifier_kind,
                    template_factory=self.template_factory,
                    feature_cfg=self.feature_cfg,
                ).fit(y_train)
                metrics = self._score_model(model, y_train, y_val)
                prior_metrics = self._prior_metrics_for_specs(
                    y_train=y_train,
                    y_val=y_val,
                    identifier_kind=identifier_kind,
                    factor_specs=trial_specs,
                    attractor_prior=attractor_prior,
                    attractor_labels=attractor_labels,
                    dt=dt,
                    cache=prior_cache,
                )
                score, rollout_score = self._selection_objective(
                    rollout_metrics=metrics,
                    prior_metrics=prior_metrics,
                    y_scale=y_scale,
                )
                if score < best_score:
                    best_score = score
                    best_spec = spec
                    best_metrics = metrics
                    best_prior_metrics = prior_metrics
                    best_rollout_score = rollout_score
            improvement = current_score - best_score
            if best_spec is None or improvement < self.mining_cfg.min_score_improvement:
                break
            selected.append(best_spec)
            remaining = [spec for spec in remaining if spec.name != best_spec.name]
            current_score = best_score
            current_rollout_score = best_rollout_score
            current_metrics = best_metrics
            current_prior_metrics = best_prior_metrics
        return selected, baseline_metrics, current_metrics, current_score, current_rollout_score, baseline_prior_metrics, current_prior_metrics

    def run_for_identifier(
        self,
        task_name: str,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        identifier_kind: str,
        attractor_prior: AttractorPrior | None = None,
        attractor_labels: np.ndarray | None = None,
        dt: float = 1.0,
    ) -> FactorMiningRunResult:
        factor_bank = build_factor_bank(
            include_pairwise_mutations=self.mining_cfg.include_pairwise_mutations,
            max_pairwise_mutations=self.mining_cfg.max_pairwise_mutations,
        )
        property_profile = analyze_signal_properties(y_train)
        prioritized_bank, prior_weights = prioritize_factor_bank(
            factor_bank,
            profile=property_profile,
            mode=self.mining_cfg.mode,
            full_library_search=self.mining_cfg.full_library_search,
            prescreen_top_k=max(self.mining_cfg.property_prescreen_top_k, self.mining_cfg.screen_top_m),
        )
        property_summary = render_property_summary(
            property_profile,
            mode=self.mining_cfg.mode,
            screened_count=len(prioritized_bank),
            total_count=len(factor_bank),
        )
        train_std, train_context, full_std, full_context, identifier = self._context_splits(y_train=y_train, y_val=y_val, identifier_kind=identifier_kind)
        train_mu = float(np.mean(y_train))
        train_sigma = float(np.std(y_train) + 1e-12)
        val_std = ((np.asarray(y_val, dtype=float).reshape(-1) - train_mu) / train_sigma).astype(float)
        val_context = self.engine.build_base_sequence(val_std)
        val_context = self.engine.augment_with_identifier(val_context, identifier.batch_outputs(val_context))
        screened, baseline_screen_rmse, factor_columns = self._screen_candidates(
            y_train=y_train,
            y_val=y_val,
            full_std=full_std,
            full_context=full_context,
            train_context=train_context,
            val_context=val_context,
            factor_bank=prioritized_bank,
            prior_weights=prior_weights,
            attractor_prior=attractor_prior,
            attractor_labels=attractor_labels,
            dt=dt,
        )
        top_specs = [spec for spec in prioritized_bank if spec.name in {row.factor_name for row in screened.top(self.mining_cfg.screen_top_m)}]
        top_specs = sorted(top_specs, key=lambda spec: next(row.rank for row in screened.rows if row.factor_name == spec.name))
        selected_specs, baseline_metrics, final_metrics, final_score, rollout_validation_score, baseline_prior_metrics, final_prior_metrics = self._baseline_and_selection(
            identifier_kind=identifier_kind,
            candidate_specs=top_specs,
            y_train=y_train,
            y_val=y_val,
            task_name=task_name,
            attractor_prior=attractor_prior,
            attractor_labels=attractor_labels,
            dt=dt,
        )
        final_model = FactorAugmentedRCModel(
            rc_cfg=self.rc_cfg,
            factor_specs=selected_specs,
            identifier_kind=identifier_kind,
            template_factory=self.template_factory,
            feature_cfg=self.feature_cfg,
        ).fit(np.concatenate([y_train, y_val]))
        test_context, test_future = self._test_context(y_val, y_test)
        test_metrics = evaluate_horizons(final_model, test_context, test_future, self.mining_cfg.score_horizons)
        test_metrics.update(final_model.one_step_metrics(np.concatenate([y_val[-len(test_context):], y_test]), burn_in=len(test_context)))
        selected_names = {spec.name for spec in selected_specs}
        for row in screened.rows:
            row.identifier_kind = identifier_kind
            row.selected = row.factor_name in selected_names
            if row.selected:
                row.notes = "selected for rollout validation"
        layer_result = curate_candidate_scores(
            screened.rows,
            factor_columns=factor_columns,
            target=full_std,
            specs_by_name={spec.name: spec for spec in prioritized_bank},
            selected_names=selected_names,
        )
        spec_map = {spec.name: spec for spec in prioritized_bank}
        layer_entries = {
            tier: [
                {
                    "score": row.to_dict(),
                    "spec": spec_map[row.factor_name].to_dict() if row.factor_name in spec_map else None,
                }
                for row in screened.rows
                if row.curation_tier == tier
            ]
            for tier in layer_result.layers
        }
        role_groups: dict[str, list[str]] = {}
        for spec in prioritized_bank:
            role_groups.setdefault(spec.manifold_role, []).append(spec.name)
        selected_library = SelectedFactorLibrary(
            task_name=task_name,
            identifier_kind=identifier_kind,
            selected_factors=selected_specs,
            baseline_rmse50=baseline_metrics.get("rmse@50", np.nan),
            final_rmse50=final_metrics.get("rmse@50", np.nan),
            final_rmse10=final_metrics.get("rmse@10", np.nan),
            validation_score=final_score,
            rollout_validation_score=rollout_validation_score,
            baseline_wsga_epr_score=baseline_prior_metrics.get("wsga_epr_score", np.nan),
            final_wsga_epr_score=final_prior_metrics.get("wsga_epr_score", np.nan),
            baseline_wsga_epr_loss=baseline_prior_metrics.get("wsga_epr_loss", np.nan),
            final_wsga_epr_loss=final_prior_metrics.get("wsga_epr_loss", np.nan),
            library_layers=layer_result.layers,
            future_factor_queue=layer_result.promotion_queue,
            curation_notes=layer_result.notes,
            notes=(
                f"{property_summary}; "
                f"one-step baseline screening rmse={baseline_screen_rmse:.4g}; "
                f"selected {len(selected_specs)} factors via forward selection; "
                f"final rollout score={rollout_validation_score:.4g}; "
                f"final wsga_epr_score={final_prior_metrics.get('wsga_epr_score', np.nan):.4g}"
            ),
        )
        return FactorMiningRunResult(
            task_name=task_name,
            identifier_kind=identifier_kind,
            candidate_scores=screened.rows,
            selected_library=selected_library,
            property_profile=property_profile.to_dict(),
            property_summary=property_summary,
            baseline_metrics=baseline_metrics,
            final_metrics=final_metrics,
            test_metrics=test_metrics,
            selected_specs=selected_specs,
            baseline_prior_metrics=baseline_prior_metrics,
            final_prior_metrics=final_prior_metrics,
            layered_library={
                "task_name": task_name,
                "identifier_kind": identifier_kind,
                "layers": layer_result.layers,
                "role_groups": role_groups,
                "layer_entries": layer_entries,
                "promotion_queue": layer_entries.get("holding", []),
                "notes": layer_result.notes,
            },
        )


class ScreenedCandidates:
    def __init__(self, rows: Iterable[CandidateScore]):
        self.rows = list(rows)

    def top(self, n: int) -> list[CandidateScore]:
        return list(self.rows[: int(n)])

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([row.to_dict() for row in self.rows])
