from .base import CandidateScore, DynamicsFeatureConfig, FactorMiningConfig, FactorSpec, SelectedFactorLibrary
from .factor_bank import BASE_FACTOR_SPECS, build_factor_bank, evaluate_factor_array, evaluate_factor_step, factor_formula
from .feature_engine import DynamicsFeatureEngine
from .identifiers import IDENTIFIER_REGISTRY, make_identifier
from .property_analyzer import SignalPropertyProfile, analyze_signal_properties, factor_prior_weight, prioritize_factor_bank, render_property_summary, scalar_koopman_diagnostic
from .readout import CausalFactorReadout, ReadoutState
from .repository import FASTSLOW_READOUT_PRESET, FactorRepository, default_factor_repository, fastslow_readout_specs, load_selected_factor_library

__all__ = [
    "CandidateScore",
    "DynamicsFeatureConfig",
    "FactorMiningConfig",
    "FactorSpec",
    "SelectedFactorLibrary",
    "BASE_FACTOR_SPECS",
    "build_factor_bank",
    "evaluate_factor_array",
    "evaluate_factor_step",
    "factor_formula",
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
    "FASTSLOW_READOUT_PRESET",
    "FactorRepository",
    "default_factor_repository",
    "fastslow_readout_specs",
    "load_selected_factor_library",
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
