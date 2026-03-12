from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class DynamicsFeatureConfig:
    fast_window: float = 10.0
    slow_windows: tuple[float, ...] = (14.0, 28.0, 57.0, 114.0)
    scale_window: float = 28.0
    scale_long_window: float = 114.0
    energy_window: float = 20.0
    energy_long_window: float = 80.0
    susceptibility_window: float = 28.0
    susceptibility_long_window: float = 114.0
    clip: float = 25.0
    critical_shrink: float = 1.5
    eps: float = 1e-6


@dataclass(frozen=True)
class FactorSpec:
    name: str
    op: str
    inputs: tuple[str, ...]
    params: dict[str, float] = field(default_factory=dict)
    family: str = "generic"
    finance_origin: str = ""
    dynamics_meaning: str = ""
    theory_tags: tuple[str, ...] = field(default_factory=tuple)
    complexity: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateScore:
    identifier_kind: str
    factor_name: str
    family: str
    one_step_rmse: float
    one_step_nrmse: float
    gain_vs_baseline: float
    prior_weight: float = 0.0
    koopman_lambda: float = 0.0
    koopman_rmse: float = 0.0
    koopman_score: float = 0.0
    screening_score: float = 0.0
    rank: int = 0
    selected: bool = False
    formula: str = ""
    finance_origin: str = ""
    dynamics_meaning: str = ""
    theory_tags: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["theory_tags"] = list(self.theory_tags)
        return payload


@dataclass
class SelectedFactorLibrary:
    task_name: str
    identifier_kind: str
    selected_factors: list[FactorSpec]
    baseline_rmse50: float
    final_rmse50: float
    final_rmse10: float
    validation_score: float
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "identifier_kind": self.identifier_kind,
            "selected_factors": [spec.to_dict() for spec in self.selected_factors],
            "baseline_rmse50": self.baseline_rmse50,
            "final_rmse50": self.final_rmse50,
            "final_rmse10": self.final_rmse10,
            "validation_score": self.validation_score,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class FactorMiningConfig:
    mode: str = "accumulate"
    identifier_kinds: tuple[str, ...] = ("sindy_slow", "spline_kan_like")
    screening_ridge: float = 1e-5
    property_weight_strength: float = 0.2
    koopman_weight_strength: float = 0.15
    property_prescreen_top_k: int = 16
    full_library_search: bool = True
    screen_top_m: int = 12
    max_selected_factors: int = 4
    min_score_improvement: float = 1e-3
    score_horizons: tuple[int, ...] = (10, 50)
    context_len: int = 200
    random_seed: int = 123
    include_pairwise_mutations: bool = True
    max_pairwise_mutations: int = 12


@dataclass
class IdentifierOutputs:
    values: dict[str, float]


ContextLike = Mapping[str, float]
