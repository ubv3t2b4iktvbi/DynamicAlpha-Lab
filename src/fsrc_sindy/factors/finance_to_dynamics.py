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
