from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .base import FactorSpec, SelectedFactorLibrary
from .factor_bank import build_factor_bank

FASTSLOW_READOUT_PRESET = "fastslow_readout"

FASTSLOW_READOUT_SPECS: tuple[FactorSpec, ...] = (
    FactorSpec(
        name="fastslow_fast",
        op="identity",
        inputs=("fs_fast",),
        family="readout_base",
        finance_origin="legacy fast branch",
        dynamics_meaning="Legacy causal fast branch kept as a first-class readout factor.",
        theory_tags=("fastslow", "readout"),
    ),
    FactorSpec(
        name="fastslow_slow",
        op="identity",
        inputs=("fs_slow",),
        family="readout_base",
        finance_origin="legacy slow branch",
        dynamics_meaning="Legacy causal slow branch kept as a first-class readout factor.",
        theory_tags=("fastslow", "readout"),
    ),
    FactorSpec(
        name="fastslow_gap",
        op="identity",
        inputs=("fs_m",),
        family="readout_base",
        finance_origin="legacy fast-slow gap",
        dynamics_meaning="Legacy fast-minus-slow readout gap kept as a first-class readout factor.",
        theory_tags=("fastslow", "order_parameter", "readout"),
    ),
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
        presets = {
            FASTSLOW_READOUT_PRESET: tuple(spec.name for spec in FASTSLOW_READOUT_SPECS),
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
        notes=str(payload.get("notes", "")),
    )
