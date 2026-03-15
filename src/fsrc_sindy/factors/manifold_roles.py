from __future__ import annotations

from typing import Mapping

from .base import FactorSpec

MANIFOLD_ROLE_ORDER: tuple[str, ...] = (
    "chart_position",
    "tangent_flow",
    "normal_amplitude",
    "closure_memory",
    "coarse_geometry",
    "control_drive",
    "regime_boundary",
    "surprise_alignment",
)

MANIFOLD_ROLE_DESCRIPTIONS: Mapping[str, str] = {
    "chart_position": "Coordinates that locate the system on a slow manifold or phase chart.",
    "tangent_flow": "Observables that describe motion along the local tangent direction of the manifold.",
    "normal_amplitude": "Observables that measure deviation from the manifold and amplitude-style relaxation.",
    "closure_memory": "Observables that diagnose unresolved memory, adiabatic validity, or closure pressure.",
    "coarse_geometry": "Observables that measure multiscale collapse quality and low-dimensional geometric validity.",
    "control_drive": "Observables that measure forcing, energy injection, or control pressure acting on the manifold.",
    "regime_boundary": "Observables that indicate boundary crossing, critical windows, escape, or chart switching.",
    "surprise_alignment": "Observables that compare inferred physics against data and capture model disagreement.",
}

BROAD_MANIFOLD_CORE_ROLES: tuple[str, ...] = (
    "chart_position",
    "tangent_flow",
    "normal_amplitude",
    "closure_memory",
    "coarse_geometry",
)

ROLE_TO_FACTOR_NAMES: Mapping[str, tuple[str, ...]] = {
    "chart_position": (
        "phase_bottom_score",
        "support_recovery",
        "slow_level_norm",
        "rg_order_parameter",
        "positive_gap",
        "negative_retracement",
        "band_position",
        "chart_position_confidence",
        "phase_chart_consistency",
    ),
    "tangent_flow": (
        "gap_velocity",
        "gap_acceleration",
        "trend_persistence",
        "rg_beta_flow",
        "abs_gap_velocity",
        "trend_regression_quality",
        "impulse_balance",
        "drift_energy_alignment",
        "tangent_flow_confidence",
    ),
    "normal_amplitude": (
        "slow_fast_gap",
        "shock_recovery",
        "slow_manifold_alignment",
        "isostable_relaxation",
        "rg_fast_mode",
        "isostable_bottom_recovery",
        "isostable_adiabatic_support",
        "retracement_support_gate",
        "normal_escape_pressure",
        "isostable_return_margin",
    ),
    "closure_memory": (
        "adiabatic_coherence",
        "closure_stress",
        "lag1_autocorr",
        "rg_noise_scale",
        "memory_closure_load",
        "adiabatic_noise_shield",
        "rg_flow_vs_noise",
        "closure_margin",
        "memory_fiber_load",
        "chart_stability_margin",
    ),
    "coarse_geometry": (
        "collapse_quality",
        "compression_ratio",
        "timescale_separation",
        "rg_coarse_grain_score",
        "critical_collapse_gate",
        "breakout_multiscale_gate",
        "slowbreak_ratio",
        "energy_over_compression",
        "coarse_chart_integrity",
    ),
    "control_drive": (
        "energy_ratio",
        "energy_release",
        "rg_control_parameter",
        "positive_impulse_share",
        "rg_order_control_coupling",
        "rg_relevant_drive",
        "trend_energy_resonance",
        "gap_energy_coupling",
        "phase_energy_coupling",
        "support_energy_gate",
        "alignment_energy_gate",
        "drive_off_manifold_pressure",
    ),
    "regime_boundary": (
        "critical_window",
        "breakout_strength",
        "rg_critical_balance",
        "critical_memory_gate",
        "critical_slowing_pressure",
        "kramers_escape_pressure",
        "flow_supported_breakout",
        "imbalance_recovery_gate",
        "physics_critical_gate",
        "critical_softening_load",
        "critical_escape_pressure",
    ),
    "surprise_alignment": (
        "physics_drift_pred",
        "physics_drift_surprise",
        "physics_alignment",
        "physics_phase_gate",
        "surprise_to_gap_ratio",
    ),
}

_NAME_TO_ROLE = {
    factor_name: role
    for role, factor_names in ROLE_TO_FACTOR_NAMES.items()
    for factor_name in factor_names
}


def infer_manifold_role(spec: FactorSpec) -> str:
    if spec.name in _NAME_TO_ROLE:
        return _NAME_TO_ROLE[spec.name]
    tags = set(spec.theory_tags)
    if "physics_identifier" in tags or spec.family == "physics_id":
        return "surprise_alignment"
    if tags & {"criticality", "breakout", "susceptibility", "changepoint"}:
        return "regime_boundary"
    if tags & {"energy", "control_parameter", "activation"}:
        return "control_drive"
    if tags & {"markov", "memory", "adiabatic", "noise"} or spec.family == "markov":
        return "closure_memory"
    if tags & {"multiscale", "rg", "compression", "spectral"} or spec.family == "multiscale":
        return "coarse_geometry"
    if tags & {"phase", "oscillation", "support"} or spec.family == "phase":
        return "chart_position"
    if tags & {"isostable", "phase_amplitude", "recovery"}:
        return "normal_amplitude"
    if tags & {"trend", "drift", "beta_function", "order_parameter", "slow_fast", "slow_manifold", "curvature"}:
        return "tangent_flow"
    return "coarse_geometry"


def manifold_role_description(role: str) -> str:
    return MANIFOLD_ROLE_DESCRIPTIONS.get(role, MANIFOLD_ROLE_DESCRIPTIONS["coarse_geometry"])
