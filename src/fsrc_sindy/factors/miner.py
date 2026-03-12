from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ..metrics import evaluate_horizons
from ..models.rc import RCConfig, ReservoirTemplateFactory
from ..selection import validation_score
from .base import CandidateScore, DynamicsFeatureConfig, FactorMiningConfig, FactorSpec, SelectedFactorLibrary
from .factor_bank import build_factor_bank, evaluate_factor_array, factor_formula
from .feature_engine import DynamicsFeatureEngine
from .identifiers import make_identifier
from .property_analyzer import analyze_signal_properties, prioritize_factor_bank, render_property_summary, scalar_koopman_diagnostic
from .rc_proxy import FactorAugmentedRCModel, ReservoirTeacherForcedScreen


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

    def summary_row(self) -> dict[str, object]:
        selected_names = {spec.name for spec in self.selected_specs}
        selected_koopman_scores = [row.koopman_score for row in self.candidate_scores if row.factor_name in selected_names]
        return {
            "task": self.task_name,
            "identifier_kind": self.identifier_kind,
            "mode": self.property_summary.split(";")[0],
            "num_candidates": len(self.candidate_scores),
            "num_selected": len(self.selected_specs),
            "selected_factors": "; ".join(spec.name for spec in self.selected_specs),
            "selected_koopman_score": float(np.mean(selected_koopman_scores)) if selected_koopman_scores else np.nan,
            "baseline_rmse10": self.baseline_metrics.get("rmse@10", np.nan),
            "baseline_rmse50": self.baseline_metrics.get("rmse@50", np.nan),
            "final_rmse10": self.final_metrics.get("rmse@10", np.nan),
            "final_rmse50": self.final_metrics.get("rmse@50", np.nan),
            "test_rmse10": self.test_metrics.get("rmse@10", np.nan),
            "test_rmse50": self.test_metrics.get("rmse@50", np.nan),
            "validation_score": self.selected_library.validation_score,
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
    ) -> tuple[ScreenedCandidates, float]:
        screen = ReservoirTeacherForcedScreen(
            cfg=self.rc_cfg,
            template_factory=self.template_factory,
            ridge=self.mining_cfg.screening_ridge,
            train_len=len(y_train),
        )
        states = screen.build_state_rows(full_std)
        baseline_score = screen.fit_and_score(full_std, states, [])
        scored: list[CandidateScore] = []
        for spec in factor_bank:
            factor_col = evaluate_factor_array(spec, full_context)
            metrics = screen.fit_and_score(full_std, states, [factor_col])
            koopman = scalar_koopman_diagnostic(
                evaluate_factor_array(spec, train_context),
                evaluate_factor_array(spec, val_context),
            )
            prior_weight = float(prior_weights.get(spec.name, 0.0))
            screening_score = float(
                metrics.rmse
                - self.mining_cfg.property_weight_strength * prior_weight * baseline_score.rmse
                - self.mining_cfg.koopman_weight_strength * koopman["koopman_score"] * baseline_score.rmse
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
                    theory_tags=spec.theory_tags,
                )
            )
        scored.sort(key=lambda row: (row.screening_score, row.one_step_rmse, -row.gain_vs_baseline))
        for rank, row in enumerate(scored, start=1):
            row.rank = rank
        return ScreenedCandidates(scored), baseline_score.rmse

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

    def _baseline_and_selection(
        self,
        identifier_kind: str,
        candidate_specs: Sequence[FactorSpec],
        y_train: np.ndarray,
        y_val: np.ndarray,
        task_name: str,
    ) -> tuple[list[FactorSpec], dict[str, float], dict[str, float], float]:
        baseline_model = FactorAugmentedRCModel(
            rc_cfg=self.rc_cfg,
            factor_specs=[],
            identifier_kind=identifier_kind,
            template_factory=self.template_factory,
            feature_cfg=self.feature_cfg,
        ).fit(y_train)
        baseline_metrics = self._score_model(baseline_model, y_train, y_val)
        current_score = validation_score(
            baseline_metrics,
            score_horizons=self.mining_cfg.score_horizons,
            y_scale=float(np.std(y_train) + 1e-12),
        )
        current_metrics = baseline_metrics
        selected: list[FactorSpec] = []
        remaining = list(candidate_specs)
        while remaining and len(selected) < self.mining_cfg.max_selected_factors:
            best_spec = None
            best_metrics = None
            best_score = current_score
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
                score = validation_score(
                    metrics,
                    score_horizons=self.mining_cfg.score_horizons,
                    y_scale=float(np.std(y_train) + 1e-12),
                )
                if score < best_score:
                    best_score = score
                    best_spec = spec
                    best_metrics = metrics
            improvement = current_score - best_score
            if best_spec is None or improvement < self.mining_cfg.min_score_improvement:
                break
            selected.append(best_spec)
            remaining = [spec for spec in remaining if spec.name != best_spec.name]
            current_score = best_score
            current_metrics = best_metrics
        return selected, baseline_metrics, current_metrics, current_score

    def run_for_identifier(
        self,
        task_name: str,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        identifier_kind: str,
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
        screened, baseline_screen_rmse = self._screen_candidates(
            y_train=y_train,
            y_val=y_val,
            full_std=full_std,
            full_context=full_context,
            train_context=train_context,
            val_context=val_context,
            factor_bank=prioritized_bank,
            prior_weights=prior_weights,
        )
        top_specs = [spec for spec in prioritized_bank if spec.name in {row.factor_name for row in screened.top(self.mining_cfg.screen_top_m)}]
        top_specs = sorted(top_specs, key=lambda spec: next(row.rank for row in screened.rows if row.factor_name == spec.name))
        selected_specs, baseline_metrics, final_metrics, final_score = self._baseline_and_selection(
            identifier_kind=identifier_kind,
            candidate_specs=top_specs,
            y_train=y_train,
            y_val=y_val,
            task_name=task_name,
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
        rows_by_name = {row.factor_name: row for row in screened.rows}
        for row in screened.rows:
            row.identifier_kind = identifier_kind
            row.selected = row.factor_name in {spec.name for spec in selected_specs}
            if row.selected:
                row.notes = "selected for rollout validation"
        selected_library = SelectedFactorLibrary(
            task_name=task_name,
            identifier_kind=identifier_kind,
            selected_factors=selected_specs,
            baseline_rmse50=baseline_metrics.get("rmse@50", np.nan),
            final_rmse50=final_metrics.get("rmse@50", np.nan),
            final_rmse10=final_metrics.get("rmse@10", np.nan),
            validation_score=final_score,
            notes=(
                f"{property_summary}; "
                f"one-step baseline screening rmse={baseline_screen_rmse:.4g}; "
                f"selected {len(selected_specs)} factors via forward selection"
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
        )


class ScreenedCandidates:
    def __init__(self, rows: Iterable[CandidateScore]):
        self.rows = list(rows)

    def top(self, n: int) -> list[CandidateScore]:
        return list(self.rows[: int(n)])

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([row.to_dict() for row in self.rows])
