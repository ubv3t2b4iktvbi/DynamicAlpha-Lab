from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .base import FactorSpec, SelectedFactorLibrary
from .factor_bank import build_factor_bank
from .manifold_roles import BROAD_MANIFOLD_CORE_ROLES, MANIFOLD_ROLE_ORDER

FASTSLOW_READOUT_PRESET = "fastslow_readout"
RG_READOUT_PRESET = "rg_readout"
FASTSLOW_RG_INTERACTION_PRESET = "sf_rg_interaction_readout"
FASTSLOW_RG_GATED_PRESET = "sf_rg_gated_readout"
CORE_LIBRARY_PRESET = "core_library"
EXTENDED_LIBRARY_PRESET = "extended_library"
EXPERIMENTAL_LIBRARY_PRESET = "experimental_library"
BROAD_MANIFOLD_CORE_PRESET = "broad_manifold_core"
RG_READOUT_FACTOR_NAMES: tuple[str, ...] = (
    "rg_order_parameter",
    "rg_control_parameter",
    "rg_noise_scale",
    "rg_beta_flow",
    "rg_coarse_grain_score",
    "rg_critical_balance",
)
FASTSLOW_READOUT_INPUTS: tuple[tuple[str, str], ...] = (
    ("fastslow_fast", "fs_fast"),
    ("fastslow_slow", "fs_slow"),
    ("fastslow_gap", "fs_m"),
)
SPARSE_RG_GATE_FACTOR_NAMES: tuple[str, ...] = (
    "rg_order_parameter",
    "rg_beta_flow",
    "rg_critical_balance",
)
SPARSE_FASTSLOW_RG_GATED_FACTOR_NAMES: tuple[str, ...] = (
    "fastslow_fast",
    "fastslow_slow",
    "fastslow_gap",
    *SPARSE_RG_GATE_FACTOR_NAMES,
    "fastslow_gap_x_rg_order_parameter",
    "fastslow_gap_x_rg_beta_flow",
    "fastslow_slow_x_rg_critical_balance",
)

FASTSLOW_READOUT_SPECS: tuple[FactorSpec, ...] = (
    FactorSpec(
        name="fastslow_fast",
        op="identity",
        inputs=("fs_fast",),
        family="readout_base",
        finance_origin="legacy fast branch",
        dynamics_meaning="Legacy causal fast branch kept as a first-class readout factor.",
        theory_tags=("fastslow", "readout"),
        source="readout_preset",
        default_tier="extended",
        manifold_role="normal_amplitude",
    ),
    FactorSpec(
        name="fastslow_slow",
        op="identity",
        inputs=("fs_slow",),
        family="readout_base",
        finance_origin="legacy slow branch",
        dynamics_meaning="Legacy causal slow branch kept as a first-class readout factor.",
        theory_tags=("fastslow", "readout"),
        source="readout_preset",
        default_tier="extended",
        manifold_role="chart_position",
    ),
    FactorSpec(
        name="fastslow_gap",
        op="identity",
        inputs=("fs_m",),
        family="readout_base",
        finance_origin="legacy fast-slow gap",
        dynamics_meaning="Legacy fast-minus-slow readout gap kept as a first-class readout factor.",
        theory_tags=("fastslow", "order_parameter", "readout"),
        source="readout_preset",
        default_tier="extended",
        manifold_role="normal_amplitude",
    ),
)

FASTSLOW_RG_INTERACTION_SPECS: tuple[FactorSpec, ...] = tuple(
    FactorSpec(
        name=f"{fastslow_name}_x_{rg_name}",
        op="product",
        inputs=(fastslow_input, rg_name),
        family="readout_interaction",
        finance_origin="hierarchical sf-rg regime conditioning",
        dynamics_meaning=(
            f"Interaction term between local {fastslow_name} readout state and macro "
            f"RG regime signal {rg_name}, approximating regime-conditioned effective dynamics."
        ),
        theory_tags=("fastslow", "rg", "interaction", "readout"),
        complexity=2,
        source="readout_interaction",
        default_tier="experimental",
        manifold_role="regime_boundary",
    )
    for fastslow_name, fastslow_input in FASTSLOW_READOUT_INPUTS
    for rg_name in RG_READOUT_FACTOR_NAMES
)


class FactorRepository:
    """Central registry for reusable factor definitions and named presets."""

    def __init__(
        self,
        specs: Iterable[FactorSpec] = (),
        presets: Mapping[str, Sequence[str]] | None = None,
    ):
        self._specs: dict[str, FactorSpec] = {}
        for spec in specs:
            self.add(spec)
        self._presets: dict[str, tuple[str, ...]] = {
            str(name): tuple(str(spec_name) for spec_name in names)
            for name, names in (presets or {}).items()
        }

    @classmethod
    def default(
        cls,
        *,
        include_pairwise_mutations: bool = True,
        max_pairwise_mutations: int = 12,
    ) -> "FactorRepository":
        specs = list(build_factor_bank(
            include_pairwise_mutations=include_pairwise_mutations,
            max_pairwise_mutations=max_pairwise_mutations,
        ))
        specs.extend(FASTSLOW_READOUT_SPECS)
        specs.extend(FASTSLOW_RG_INTERACTION_SPECS)
        core_names = tuple(spec.name for spec in specs if spec.default_tier == "core")
        extended_names = tuple(spec.name for spec in specs if spec.default_tier in {"core", "extended"})
        experimental_names = tuple(spec.name for spec in specs if spec.default_tier == "experimental")
        role_presets = {
            f"manifold_{role}": tuple(spec.name for spec in specs if spec.manifold_role == role)
            for role in MANIFOLD_ROLE_ORDER
        }
        broad_core_names = tuple(
            spec.name
            for spec in specs
            if spec.manifold_role in BROAD_MANIFOLD_CORE_ROLES and spec.default_tier in {"core", "extended"}
        )
        presets = {
            CORE_LIBRARY_PRESET: core_names,
            EXTENDED_LIBRARY_PRESET: extended_names,
            EXPERIMENTAL_LIBRARY_PRESET: experimental_names,
            BROAD_MANIFOLD_CORE_PRESET: broad_core_names,
            FASTSLOW_READOUT_PRESET: tuple(spec.name for spec in FASTSLOW_READOUT_SPECS),
            RG_READOUT_PRESET: RG_READOUT_FACTOR_NAMES,
            FASTSLOW_RG_INTERACTION_PRESET: (
                tuple(spec.name for spec in FASTSLOW_READOUT_SPECS)
                + RG_READOUT_FACTOR_NAMES
                + tuple(spec.name for spec in FASTSLOW_RG_INTERACTION_SPECS)
            ),
            FASTSLOW_RG_GATED_PRESET: SPARSE_FASTSLOW_RG_GATED_FACTOR_NAMES,
            **role_presets,
        }
        return cls(specs=specs, presets=presets)

    def add(self, spec: FactorSpec) -> None:
        self._specs[spec.name] = spec

    def get(self, name: str) -> FactorSpec:
        if name not in self._specs:
            raise KeyError(f"Unknown factor: {name}")
        return self._specs[name]

    def select(self, names: Sequence[str]) -> list[FactorSpec]:
        return [self.get(name) for name in names]

    def preset(self, name: str) -> list[FactorSpec]:
        if name not in self._presets:
            raise KeyError(f"Unknown factor preset: {name}")
        return self.select(self._presets[name])

    def all(self) -> list[FactorSpec]:
        return list(self._specs.values())


_DEFAULT_REPOSITORY = FactorRepository.default()


def default_factor_repository() -> FactorRepository:
    return _DEFAULT_REPOSITORY


def fastslow_readout_specs() -> list[FactorSpec]:
    return default_factor_repository().preset(FASTSLOW_READOUT_PRESET)


def broad_manifold_core_specs() -> list[FactorSpec]:
    return default_factor_repository().preset(BROAD_MANIFOLD_CORE_PRESET)


def manifold_role_specs(role: str) -> list[FactorSpec]:
    return default_factor_repository().preset(f"manifold_{role}")


def core_library_specs() -> list[FactorSpec]:
    return default_factor_repository().preset(CORE_LIBRARY_PRESET)


def extended_library_specs() -> list[FactorSpec]:
    return default_factor_repository().preset(EXTENDED_LIBRARY_PRESET)


def experimental_library_specs() -> list[FactorSpec]:
    return default_factor_repository().preset(EXPERIMENTAL_LIBRARY_PRESET)


def rg_readout_specs() -> list[FactorSpec]:
    return default_factor_repository().preset(RG_READOUT_PRESET)


def sparse_rg_gate_specs() -> list[FactorSpec]:
    return default_factor_repository().select(SPARSE_RG_GATE_FACTOR_NAMES)


def sf_rg_interaction_readout_specs() -> list[FactorSpec]:
    return default_factor_repository().preset(FASTSLOW_RG_INTERACTION_PRESET)


def sf_rg_gated_readout_specs() -> list[FactorSpec]:
    return default_factor_repository().preset(FASTSLOW_RG_GATED_PRESET)


def load_selected_factor_library(path: str | Path) -> SelectedFactorLibrary:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    selected_specs = [FactorSpec(**spec_payload) for spec_payload in payload.get("selected_factors", [])]
    return SelectedFactorLibrary(
        task_name=str(payload.get("task_name", "")),
        identifier_kind=str(payload.get("identifier_kind", "none")),
        selected_factors=selected_specs,
        baseline_rmse50=float(payload.get("baseline_rmse50", float("nan"))),
        final_rmse50=float(payload.get("final_rmse50", float("nan"))),
        final_rmse10=float(payload.get("final_rmse10", float("nan"))),
        validation_score=float(payload.get("validation_score", float("nan"))),
        rollout_validation_score=float(payload.get("rollout_validation_score", payload.get("validation_score", float("nan")))),
        baseline_wsga_epr_score=float(payload.get("baseline_wsga_epr_score", float("nan"))),
        final_wsga_epr_score=float(payload.get("final_wsga_epr_score", float("nan"))),
        baseline_wsga_epr_loss=float(payload.get("baseline_wsga_epr_loss", float("nan"))),
        final_wsga_epr_loss=float(payload.get("final_wsga_epr_loss", float("nan"))),
        library_layers={
            str(name): [str(item) for item in names]
            for name, names in dict(payload.get("library_layers", {})).items()
        },
        future_factor_queue=[str(name) for name in payload.get("future_factor_queue", [])],
        curation_notes=str(payload.get("curation_notes", "")),
        notes=str(payload.get("notes", "")),
    )
