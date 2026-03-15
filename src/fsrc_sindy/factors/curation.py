from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .base import CandidateScore, FactorSpec


def _clip01(value: float) -> float:
    return float(min(max(value, 0.0), 1.0))


def _safe_series(values: np.ndarray | Sequence[float]) -> np.ndarray:
    return np.nan_to_num(np.asarray(values, dtype=float).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = _safe_series(x)
    y = _safe_series(y)
    if x.size != y.size or x.size < 4:
        return 0.0
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return 0.0
    return float(abs(np.corrcoef(x, y)[0, 1]))


def _digitize_rank(values: np.ndarray, bins: int = 12) -> np.ndarray:
    values = _safe_series(values)
    if values.size == 0:
        return np.zeros(0, dtype=int)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, num=values.size, endpoint=False)
    edges = np.linspace(0.0, 1.0, num=max(int(bins), 2) + 1)
    codes = np.digitize(ranks, edges[1:-1], right=False)
    return np.asarray(codes, dtype=int)


def _normalized_mutual_info(x: np.ndarray, y: np.ndarray, bins: int = 12) -> float:
    x_codes = _digitize_rank(x, bins=bins)
    y_codes = _digitize_rank(y, bins=bins)
    if x_codes.size != y_codes.size or x_codes.size < 4:
        return 0.0
    x_bins = int(np.max(x_codes)) + 1
    y_bins = int(np.max(y_codes)) + 1
    joint = np.zeros((x_bins, y_bins), dtype=float)
    np.add.at(joint, (x_codes, y_codes), 1.0)
    joint /= float(np.sum(joint) + 1e-12)
    px = np.sum(joint, axis=1)
    py = np.sum(joint, axis=0)
    nz = joint > 0.0
    mi = float(np.sum(joint[nz] * np.log((joint[nz] + 1e-12) / ((px[:, None] * py[None, :])[nz] + 1e-12))))
    hx = float(-np.sum(px[px > 0.0] * np.log(px[px > 0.0] + 1e-12)))
    hy = float(-np.sum(py[py > 0.0] * np.log(py[py > 0.0] + 1e-12)))
    normalizer = max(hx, hy, 1e-12)
    return _clip01(mi / normalizer)


def _effectiveness_score(row: CandidateScore, gain_scale: float) -> float:
    gain_component = _clip01(max(float(row.gain_vs_baseline), 0.0) / (gain_scale + 1e-12))
    koopman_component = _clip01(float(row.koopman_score))
    wsga_component = _clip01(float(row.wsga_epr_score)) if np.isfinite(row.wsga_epr_score) else 0.0
    return _clip01(0.55 * gain_component + 0.30 * koopman_component + 0.15 * wsga_component)


def _assign_tier(
    *,
    row: CandidateScore,
    selected_names: set[str],
) -> str:
    redundancy = float(max(
        0.0 if not np.isfinite(row.max_redundancy_corr) else row.max_redundancy_corr,
        0.0 if not np.isfinite(row.max_redundancy_mutual_info) else row.max_redundancy_mutual_info,
    ))
    selected = row.factor_name in selected_names
    is_holding_source = row.source in {"pairwise_mutation", "mined"} or row.family in {"composite", "physics_id", "readout_interaction"}
    if is_holding_source and (selected or row.curation_score >= 0.45 or row.effectiveness_score >= 0.45):
        return "holding"
    if (
        row.default_tier == "core"
        and row.curation_score >= 0.58
        and row.novelty_score >= 0.25
        and redundancy <= 0.92
    ):
        return "core"
    if selected or row.curation_score >= 0.42 or row.default_tier in {"core", "extended"}:
        return "extended"
    return "experimental"


@dataclass(frozen=True)
class FactorLayeringResult:
    layers: dict[str, list[str]]
    promotion_queue: list[str]
    notes: str


def curate_candidate_scores(
    rows: Sequence[CandidateScore],
    *,
    factor_columns: Mapping[str, np.ndarray],
    target: np.ndarray,
    specs_by_name: Mapping[str, FactorSpec],
    selected_names: set[str],
) -> FactorLayeringResult:
    ordered_rows = list(rows)
    if not ordered_rows:
        return FactorLayeringResult(
            layers={"core": [], "extended": [], "experimental": [], "holding": []},
            promotion_queue=[],
            notes="no candidate factors were available for curation",
        )

    target_next = _safe_series(target)[1:]
    feature_cache = {
        name: _safe_series(values)[:-1]
        for name, values in factor_columns.items()
        if _safe_series(values).size >= 2
    }
    gain_scale = max(max(max(float(row.gain_vs_baseline), 0.0) for row in ordered_rows), 1e-12)
    seen_features: list[tuple[str, np.ndarray]] = []
    layers = {"core": [], "extended": [], "experimental": [], "holding": []}

    for row in ordered_rows:
        spec = specs_by_name.get(row.factor_name)
        if spec is not None:
            row.source = spec.source
            row.default_tier = spec.default_tier
        feature = feature_cache.get(row.factor_name, np.zeros_like(target_next))
        row.target_corr = _safe_corr(feature, target_next)
        row.target_mutual_info = _normalized_mutual_info(feature, target_next)
        if seen_features:
            row.max_redundancy_corr = max(_safe_corr(feature, prev) for _, prev in seen_features)
            row.max_redundancy_mutual_info = max(_normalized_mutual_info(feature, prev) for _, prev in seen_features)
        else:
            row.max_redundancy_corr = 0.0
            row.max_redundancy_mutual_info = 0.0
        row.effectiveness_score = _effectiveness_score(row, gain_scale=gain_scale)
        row.novelty_score = _clip01(1.0 - 0.5 * (row.max_redundancy_corr + row.max_redundancy_mutual_info))
        row.curation_score = _clip01(
            0.45 * row.effectiveness_score
            + 0.25 * row.target_mutual_info
            + 0.15 * row.target_corr
            + 0.15 * row.novelty_score
        )
        row.curation_tier = _assign_tier(row=row, selected_names=selected_names)
        layers[row.curation_tier].append(row.factor_name)
        seen_features.append((row.factor_name, feature))

    promotion_queue = list(layers["holding"])
    notes = (
        "tiering uses predictive effectiveness (gain + Koopman + optional EPR), "
        "target relevance (correlation + normalized mutual information), and "
        "novelty against higher-ranked factors; pairwise or newly mined factors "
        "are held in a promotion queue before entering the default core library"
    )
    return FactorLayeringResult(layers=layers, promotion_queue=promotion_queue, notes=notes)
