from __future__ import annotations

TRANSLATION_TABLE = [
    {
        "finance_pattern": "Moving-average crosses / trend turn",
        "dynamics_object": "Fast-slow order parameter and sign flips",
        "project_feature": "slow_fast_gap, positive_gap",
        "mechanism_note": "A fast branch leaving the slow manifold is treated as a phase or attractor departure rather than a literal MA crossover.",
    },
    {
        "finance_pattern": "KDJ low zone / last pullback",
        "dynamics_object": "Phase-bottom recovery evidence",
        "project_feature": "phase_bottom_score, support_recovery",
        "mechanism_note": "Negative residual plus positive phase advance becomes causal evidence for a recovery from the lower side of a cycle.",
    },
    {
        "finance_pattern": "Volume burst / capital entry",
        "dynamics_object": "Control-energy injection",
        "project_feature": "energy_ratio, energy_release",
        "mechanism_note": "Abnormal participation is translated into elevated local energy or control pressure in the observed dynamics.",
    },
    {
        "finance_pattern": "Cost zone / dominant trend track",
        "dynamics_object": "Slow manifold or slow reference orbit",
        "project_feature": "slow_level_norm, trend_persistence",
        "mechanism_note": "What finance calls a cost zone becomes a slow coordinate that constrains long-horizon drift.",
    },
    {
        "finance_pattern": "Multi-window resonance / curve quality",
        "dynamics_object": "Multiscale collapse consistency",
        "project_feature": "collapse_quality, critical_collapse_gate",
        "mechanism_note": "Agreement across windows is restated as stable coarse-graining and better collapsed coordinates.",
    },
    {
        "finance_pattern": "Model surprise / key management zone",
        "dynamics_object": "Physics-identifier mismatch",
        "project_feature": "physics_drift_surprise, physics_phase_gate",
        "mechanism_note": "Forecast mismatch is treated as mechanism-switch evidence instead of a market-only expectation gap.",
    },
    {
        "finance_pattern": "QuantaAlpha RSQR / MA / VMA seeds",
        "dynamics_object": "Locally coherent drift quality",
        "project_feature": "trend_regression_quality, trend_energy_resonance",
        "mechanism_note": "Trend-stability seeds are translated into causal trend-fit quality and then gated by control energy when persistence needs support.",
    },
    {
        "finance_pattern": "QuantaAlpha CORR / CORD / WVMA price-volume resonance",
        "dynamics_object": "Drift-energy alignment",
        "project_feature": "drift_energy_alignment, trend_energy_resonance",
        "mechanism_note": "Price-volume resonance becomes alignment between directional drift and the signed impulse or energy carried by the local dynamics.",
    },
    {
        "finance_pattern": "QuantaAlpha VSUMP / VSUMD directional flow imbalance",
        "dynamics_object": "Signed impulse imbalance",
        "project_feature": "positive_impulse_share, impulse_balance, flow_supported_breakout",
        "mechanism_note": "Up-minus-down flow is restated as the share and balance of positive versus negative signed impulse in a causal window.",
    },
    {
        "finance_pattern": "QuantaAlpha RSV / MAX / MIN window position",
        "dynamics_object": "Relative band position",
        "project_feature": "band_position, imbalance_recovery_gate",
        "mechanism_note": "Relative location inside a recent price range becomes a causal local-band coordinate for recovery or breakout timing.",
    },
    {
        "finance_pattern": "Critical slowing down / early-warning signals",
        "dynamics_object": "Recovery-rate weakening expressed through positive lag-1 memory inside a critical window",
        "project_feature": "lag1_autocorr, critical_memory_gate, critical_slowing_pressure",
        "mechanism_note": "Rising short-lag memory is treated as evidence that local recovery is slowing, especially when the critical window and short-horizon energy are both elevated.",
    },
    {
        "finance_pattern": "Phase-amplitude reduction / isostable coordinates",
        "dynamics_object": "Amplitude deviation relaxing back toward a slow manifold or cycle",
        "project_feature": "isostable_relaxation, isostable_bottom_recovery, isostable_adiabatic_support",
        "mechanism_note": "The literature's amplitude or isostable coordinate is restated as a causal restoring-force proxy that is strongest when deviations already decay back toward the low-dimensional structure.",
    },
    {
        "finance_pattern": "Kramers escape / metastable barrier crossing",
        "dynamics_object": "Noise-to-barrier pressure balance",
        "project_feature": "kramers_escape_pressure",
        "mechanism_note": "Metastable escape is mapped to fast agitation normalized by the current coarse-grained barrier proxy instead of a literal thermodynamic barrier.",
    },
    {
        "finance_pattern": "Mori-Zwanzig / generalized Langevin memory",
        "dynamics_object": "Short-lag memory carried by unresolved fast modes",
        "project_feature": "memory_closure_load",
        "mechanism_note": "Memory-kernel pressure is represented by positive lag-1 memory interacting with closure stress, flagging when the current coordinates are not Markov enough.",
    },
    {
        "finance_pattern": "Takens delay manifold / reconstructed observation chart",
        "dynamics_object": "History-space chart quality and reduced-coordinate closure margin",
        "project_feature": "chart_position_confidence, phase_chart_consistency, closure_margin",
        "mechanism_note": "Delay-style reconstructed coordinates work best when the local chart stays geometrically coherent and the reduced dynamics remains close to a Markov closure.",
    },
    {
        "finance_pattern": "Fenichel slow manifold / normal hyperbolicity",
        "dynamics_object": "Tangential flow confidence versus normal-fiber escape and return",
        "project_feature": "tangent_flow_confidence, normal_escape_pressure, isostable_return_margin",
        "mechanism_note": "Shared slow-fast geometry is tracked through tangential drift, normal escape pressure, and normal relaxation back toward a persistent slow manifold.",
    },
    {
        "finance_pattern": "Phase-isostable Koopman coordinates",
        "dynamics_object": "Phase chart stability and amplitude-return structure",
        "project_feature": "phase_chart_consistency, chart_stability_margin, isostable_return_margin",
        "mechanism_note": "Phase location is promoted only when the local chart is coherent, while amplitude-like return strength captures relaxation along isostable fibers.",
    },
    {
        "finance_pattern": "Local tangent space / principal manifold geometry",
        "dynamics_object": "Thin chart integrity and tangent-bundle reliability",
        "project_feature": "coarse_chart_integrity, tangent_flow_confidence",
        "mechanism_note": "Local manifold methods motivate factors that are strong only when the reduced chart remains thin and the tangent dynamics remains trustworthy.",
    },
    {
        "finance_pattern": "Critical transitions / loss of normal hyperbolicity",
        "dynamics_object": "Critical softening and noise-assisted escape pressure",
        "project_feature": "critical_softening_load, critical_escape_pressure, drive_off_manifold_pressure",
        "mechanism_note": "Near regime boundaries, rising memory and fast agitation indicate softening recovery and off-manifold forcing instead of benign drift.",
    },
    {
        "finance_pattern": "Mori-Zwanzig unresolved fibers",
        "dynamics_object": "Memory carried by unresolved fast fibers rather than explicit state coordinates",
        "project_feature": "memory_fiber_load",
        "mechanism_note": "Short-lag memory multiplied by fast agitation approximates unresolved fiber dynamics that should not be mistaken for a clean low-dimensional Markov state.",
    },
]


def translation_markdown() -> str:
    lines = [
        "# Finance-to-Dynamics Translation Table",
        "",
        "| Finance motif | Dynamics object | Project factor(s) | Mechanism note |",
        "|---|---|---|---|",
    ]
    for row in TRANSLATION_TABLE:
        lines.append(
            f"| {row['finance_pattern']} | {row['dynamics_object']} | {row['project_feature']} | {row['mechanism_note']} |"
        )
    return "\n".join(lines)
