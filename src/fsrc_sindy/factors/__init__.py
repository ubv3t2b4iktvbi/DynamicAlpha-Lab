from .base import CandidateScore, DynamicsFeatureConfig, FactorMiningConfig, FactorSpec, SelectedFactorLibrary
from .curation import FactorLayeringResult, curate_candidate_scores
from .factor_bank import BASE_FACTOR_SPECS, build_factor_bank, evaluate_factor_array, evaluate_factor_step, factor_formula
from .feature_engine import DynamicsFeatureEngine
from .identifiers import IDENTIFIER_REGISTRY, make_identifier
from .manifold_roles import (
    BROAD_MANIFOLD_CORE_ROLES,
    MANIFOLD_ROLE_DESCRIPTIONS,
    MANIFOLD_ROLE_ORDER,
    infer_manifold_role,
    manifold_role_description,
)
from .property_analyzer import SignalPropertyProfile, analyze_signal_properties, factor_prior_weight, prioritize_factor_bank, render_property_summary, scalar_koopman_diagnostic
from .readout import CausalFactorReadout, ReadoutState
from .repository import (
    BROAD_MANIFOLD_CORE_PRESET,
    CORE_LIBRARY_PRESET,
    EXPERIMENTAL_LIBRARY_PRESET,
    EXTENDED_LIBRARY_PRESET,
    FASTSLOW_READOUT_PRESET,
    FASTSLOW_RG_GATED_PRESET,
    FASTSLOW_RG_INTERACTION_PRESET,
    RG_READOUT_PRESET,
    SPARSE_RG_GATE_FACTOR_NAMES,
    FactorRepository,
    broad_manifold_core_specs,
    core_library_specs,
    default_factor_repository,
    experimental_library_specs,
    extended_library_specs,
    fastslow_readout_specs,
    load_selected_factor_library,
    manifold_role_specs,
    rg_readout_specs,
    sparse_rg_gate_specs,
    sf_rg_gated_readout_specs,
    sf_rg_interaction_readout_specs,
)

__all__ = [
    "CandidateScore",
    "DynamicsFeatureConfig",
    "FactorMiningConfig",
    "FactorSpec",
    "SelectedFactorLibrary",
    "FactorLayeringResult",
    "BASE_FACTOR_SPECS",
    "build_factor_bank",
    "evaluate_factor_array",
    "evaluate_factor_step",
    "factor_formula",
    "curate_candidate_scores",
    "MANIFOLD_ROLE_ORDER",
    "MANIFOLD_ROLE_DESCRIPTIONS",
    "BROAD_MANIFOLD_CORE_ROLES",
    "infer_manifold_role",
    "manifold_role_description",
    "DynamicsFeatureEngine",
    "IDENTIFIER_REGISTRY",
    "make_identifier",
    "DynamicsFactorMiner",
    "FactorMiningRunResult",
    "SignalPropertyProfile",
    "analyze_signal_properties",
    "factor_prior_weight",
    "prioritize_factor_bank",
    "render_property_summary",
    "scalar_koopman_diagnostic",
    "CausalFactorReadout",
    "ReadoutState",
    "BROAD_MANIFOLD_CORE_PRESET",
    "CORE_LIBRARY_PRESET",
    "EXTENDED_LIBRARY_PRESET",
    "EXPERIMENTAL_LIBRARY_PRESET",
    "FASTSLOW_READOUT_PRESET",
    "FASTSLOW_RG_GATED_PRESET",
    "FASTSLOW_RG_INTERACTION_PRESET",
    "RG_READOUT_PRESET",
    "SPARSE_RG_GATE_FACTOR_NAMES",
    "FactorRepository",
    "broad_manifold_core_specs",
    "core_library_specs",
    "default_factor_repository",
    "experimental_library_specs",
    "extended_library_specs",
    "fastslow_readout_specs",
    "rg_readout_specs",
    "sparse_rg_gate_specs",
    "sf_rg_gated_readout_specs",
    "sf_rg_interaction_readout_specs",
    "load_selected_factor_library",
    "manifold_role_specs",
    "FactorAugmentedRCModel",
    "ReservoirTeacherForcedScreen",
]


def __getattr__(name: str):
    if name in {"DynamicsFactorMiner", "FactorMiningRunResult"}:
        from .miner import DynamicsFactorMiner, FactorMiningRunResult

        exports = {
            "DynamicsFactorMiner": DynamicsFactorMiner,
            "FactorMiningRunResult": FactorMiningRunResult,
        }
        return exports[name]
    if name in {"FactorAugmentedRCModel", "ReservoirTeacherForcedScreen"}:
        from .rc_proxy import FactorAugmentedRCModel, ReservoirTeacherForcedScreen

        exports = {
            "FactorAugmentedRCModel": FactorAugmentedRCModel,
            "ReservoirTeacherForcedScreen": ReservoirTeacherForcedScreen,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
