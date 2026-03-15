from __future__ import annotations

from dataclasses import replace
from itertools import islice
from typing import Iterable, Mapping

import numpy as np

from .base import FactorSpec
from .manifold_roles import infer_manifold_role

CORE_BASE_FAMILIES: tuple[str, ...] = (
    "order_parameter",
    "phase",
    "energy",
    "criticality",
    "multiscale",
    "slow_manifold",
    "markov",
    "koopman",
)


def _safe_div(num: np.ndarray | float, den: np.ndarray | float, eps: float = 1e-6):
    return np.asarray(num, dtype=float) / (np.abs(np.asarray(den, dtype=float)) + eps)


def factor_formula(spec: FactorSpec) -> str:
    inputs = list(spec.inputs)
    if spec.op == "identity":
        return inputs[0]
    if spec.op == "abs":
        return f"abs({inputs[0]})"
    if spec.op == "square":
        return f"({inputs[0]})^2"
    if spec.op == "relu":
        return f"relu({inputs[0]})"
    if spec.op == "neg_relu":
        return f"relu(-{inputs[0]})"
    if spec.op == "tanh":
        return f"tanh({inputs[0]})"
    if spec.op == "product":
        return f"{inputs[0]} * {inputs[1]}"
    if spec.op == "ratio":
        return f"{inputs[0]} / (|{inputs[1]}| + eps)"
    if spec.op == "sum":
        return f"{inputs[0]} + {inputs[1]}"
    if spec.op == "diff":
        return f"{inputs[0]} - {inputs[1]}"
    if spec.op == "pos_gate_product":
        return f"relu({inputs[0]}) * {inputs[1]}"
    if spec.op == "neg_gate_product":
        return f"relu(-{inputs[0]}) * {inputs[1]}"
    raise ValueError(f"Unknown op: {spec.op}")


def evaluate_factor_array(spec: FactorSpec, context: Mapping[str, np.ndarray]) -> np.ndarray:
    vals = [np.asarray(context[name], dtype=float) for name in spec.inputs]
    if spec.op == "identity":
        out = vals[0]
    elif spec.op == "abs":
        out = np.abs(vals[0])
    elif spec.op == "square":
        out = vals[0] ** 2
    elif spec.op == "relu":
        out = np.maximum(vals[0], 0.0)
    elif spec.op == "neg_relu":
        out = np.maximum(-vals[0], 0.0)
    elif spec.op == "tanh":
        out = np.tanh(vals[0])
    elif spec.op == "product":
        out = vals[0] * vals[1]
    elif spec.op == "ratio":
        out = _safe_div(vals[0], vals[1])
    elif spec.op == "sum":
        out = vals[0] + vals[1]
    elif spec.op == "diff":
        out = vals[0] - vals[1]
    elif spec.op == "pos_gate_product":
        out = np.maximum(vals[0], 0.0) * vals[1]
    elif spec.op == "neg_gate_product":
        out = np.maximum(-vals[0], 0.0) * vals[1]
    else:
        raise ValueError(f"Unknown op: {spec.op}")
    return np.nan_to_num(np.asarray(out, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def evaluate_factor_step(spec: FactorSpec, ctx: Mapping[str, float]) -> float:
    vals = [float(ctx[name]) for name in spec.inputs]
    if spec.op == "identity":
        out = vals[0]
    elif spec.op == "abs":
        out = abs(vals[0])
    elif spec.op == "square":
        out = vals[0] ** 2
    elif spec.op == "relu":
        out = max(vals[0], 0.0)
    elif spec.op == "neg_relu":
        out = max(-vals[0], 0.0)
    elif spec.op == "tanh":
        out = float(np.tanh(vals[0]))
    elif spec.op == "product":
        out = vals[0] * vals[1]
    elif spec.op == "ratio":
        out = float(vals[0] / (abs(vals[1]) + 1e-6))
    elif spec.op == "sum":
        out = vals[0] + vals[1]
    elif spec.op == "diff":
        out = vals[0] - vals[1]
    elif spec.op == "pos_gate_product":
        out = max(vals[0], 0.0) * vals[1]
    elif spec.op == "neg_gate_product":
        out = max(-vals[0], 0.0) * vals[1]
    else:
        raise ValueError(f"Unknown op: {spec.op}")
    return float(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0))


BASE_FACTOR_SPECS: list[FactorSpec] = [
    FactorSpec(
        name="slow_fast_gap",
        op="identity",
        inputs=("m_norm",),
        family="order_parameter",
        finance_origin="均线金叉/白黄线间距",
        dynamics_meaning="快慢流形间的无量纲序参量，衡量系统是否脱离慢流形。",
        theory_tags=("slow_fast", "order_parameter"),
    ),
    FactorSpec(
        name="gap_velocity",
        op="identity",
        inputs=("dm_norm",),
        family="order_parameter",
        finance_origin="趋势加速度/拐点确认",
        dynamics_meaning="序参量的一阶变化率，反映接近相变时的推进速度。",
        theory_tags=("slow_fast", "criticality"),
    ),
    FactorSpec(
        name="gap_acceleration",
        op="identity",
        inputs=("d2m_norm",),
        family="order_parameter",
        finance_origin="二次拐点/动量再加速",
        dynamics_meaning="序参量的二阶变化率，捕捉曲率变化与二次启动。",
        theory_tags=("slow_fast", "curvature"),
    ),
    FactorSpec(
        name="phase_bottom_score",
        op="identity",
        inputs=("phase_bottom_score",),
        family="phase",
        finance_origin="KDJ低位/最后一次回调",
        dynamics_meaning="负残差与正相位推进的乘积，近似“谷底上拐”证据。",
        theory_tags=("phase", "oscillation"),
    ),
    FactorSpec(
        name="energy_ratio",
        op="identity",
        inputs=("energy_ratio",),
        family="energy",
        finance_origin="放量异动/能量注入",
        dynamics_meaning="短能量相对长期能量的比值，刻画控制参量增强。",
        theory_tags=("energy", "control_parameter"),
    ),
    FactorSpec(
        name="critical_window",
        op="identity",
        inputs=("critical_window",),
        family="criticality",
        finance_origin="临界窗口/突破前夕",
        dynamics_meaning="临界接近度、涨落响应与能量注入的联合分数。",
        theory_tags=("criticality", "susceptibility"),
    ),
    FactorSpec(
        name="collapse_quality",
        op="identity",
        inputs=("collapse_quality",),
        family="multiscale",
        finance_origin="多周期共振/主曲线质量",
        dynamics_meaning="多尺度归一化偏离的一致性，越高表示多尺度塌缩越好。",
        theory_tags=("multiscale", "rg"),
    ),
    FactorSpec(
        name="breakout_strength",
        op="identity",
        inputs=("breakout_strength",),
        family="event",
        finance_origin="强势突破成本区",
        dynamics_meaning="正序参量与能量比的联合强度，代表离开慢流形的突破。",
        theory_tags=("breakout", "energy"),
    ),
    FactorSpec(
        name="support_recovery",
        op="identity",
        inputs=("support_recovery",),
        family="event",
        finance_origin="回踩白线后再起",
        dynamics_meaning="多头背景下的谷底上拐恢复强度。",
        theory_tags=("support", "phase"),
    ),
    FactorSpec(
        name="compression_ratio",
        op="identity",
        inputs=("compression_ratio",),
        family="multiscale",
        finance_origin="缩量压缩/波动收敛",
        dynamics_meaning="局部尺度相对长期尺度的压缩比，低值常见于蓄势阶段。",
        theory_tags=("compression", "multiscale"),
    ),
    FactorSpec(
        name="energy_release",
        op="identity",
        inputs=("energy_release",),
        family="energy",
        finance_origin="放量加速",
        dynamics_meaning="能量比与正速度的乘积，近似突发释放。",
        theory_tags=("energy", "activation"),
    ),
    FactorSpec(
        name="shock_recovery",
        op="identity",
        inputs=("shock_recovery",),
        family="phase",
        finance_origin="下杀后快速修复",
        dynamics_meaning="负冲击后序参量回升的恢复评分。",
        theory_tags=("recovery", "phase"),
    ),
    FactorSpec(
        name="trend_persistence",
        op="identity",
        inputs=("trend_persistence",),
        family="slow_manifold",
        finance_origin="趋势流畅度",
        dynamics_meaning="慢流形漂移相对快变量扰动的持续性。",
        theory_tags=("trend", "slow_manifold"),
    ),
    FactorSpec(
        name="slow_level_norm",
        op="identity",
        inputs=("slow_level_norm",),
        family="slow_manifold",
        finance_origin="慢均值主趋势",
        dynamics_meaning="归一化慢模态幅值，近似被部分观测折叠后的慢变量坐标。",
        theory_tags=("slow_manifold", "trend", "koopman"),
    ),
    FactorSpec(
        name="timescale_separation",
        op="identity",
        inputs=("timescale_separation",),
        family="multiscale",
        finance_origin="快慢线速度比",
        dynamics_meaning="快漂移相对慢漂移的局部时间尺度分离强度。",
        theory_tags=("slow_fast", "multiscale", "spectral"),
    ),
    FactorSpec(
        name="slow_manifold_alignment",
        op="identity",
        inputs=("slow_manifold_alignment",),
        family="slow_manifold",
        finance_origin="贴近主趋势且动量平缓",
        dynamics_meaning="残差与序参量速度同时较小时，对慢流形邻域的因果接近度估计。",
        theory_tags=("slow_manifold", "markov", "koopman"),
    ),
    FactorSpec(
        name="adiabatic_coherence",
        op="identity",
        inputs=("adiabatic_coherence",),
        family="koopman",
        finance_origin="多周期同向且扰动受控",
        dynamics_meaning="多尺度塌缩质量与慢流形接近度的乘积，近似绝热一致性的可观测量。",
        theory_tags=("koopman", "adiabatic", "multiscale"),
    ),
    FactorSpec(
        name="closure_stress",
        op="identity",
        inputs=("closure_stress",),
        family="markov",
        finance_origin="加速冲击下的失配压力",
        dynamics_meaning="序参量速度、能量注入与塌缩误差共同诱发的 Markov 闭合压力。",
        theory_tags=("markov", "criticality", "energy"),
    ),
    FactorSpec(
        name="lag1_autocorr",
        op="identity",
        inputs=("lag1_autocorr",),
        family="markov",
        finance_origin="Critical-transition lag-1 autocorrelation",
        dynamics_meaning="Short-lag memory proxy that rises when local recovery weakens and unresolved memory persists.",
        theory_tags=("memory", "criticality", "markov"),
    ),
    FactorSpec(
        name="isostable_relaxation",
        op="identity",
        inputs=("isostable_relaxation",),
        family="phase",
        finance_origin="Phase-amplitude reduction / isostable decay",
        dynamics_meaning="Positive restoring-force proxy that is high when deviation from the slow manifold or cycle is actively relaxing.",
        theory_tags=("phase_amplitude", "isostable", "slow_manifold"),
    ),
    FactorSpec(
        name="rg_order_parameter",
        op="identity",
        inputs=("rg_order_parameter",),
        family="order_parameter",
        finance_origin="RG macro order parameter",
        dynamics_meaning="Coarse-grained slow coordinate used as the low-dimensional order parameter m.",
        theory_tags=("rg", "order_parameter", "slow_manifold"),
    ),
    FactorSpec(
        name="rg_control_parameter",
        op="identity",
        inputs=("rg_control_parameter",),
        family="energy",
        finance_origin="RG control parameter",
        dynamics_meaning="Effective control pressure that combines local energy injection with susceptibility.",
        theory_tags=("rg", "control_parameter", "criticality"),
    ),
    FactorSpec(
        name="rg_fast_mode",
        op="identity",
        inputs=("rg_fast_mode",),
        family="multiscale",
        finance_origin="RG microscopic fast mode",
        dynamics_meaning="Signed microscopic fluctuation around the slow manifold, kept causal at the current step.",
        theory_tags=("rg", "slow_fast", "multiscale"),
    ),
    FactorSpec(
        name="rg_noise_scale",
        op="identity",
        inputs=("rg_noise_scale",),
        family="markov",
        finance_origin="RG noise scale",
        dynamics_meaning="Magnitude of fast agitation that plays the role of eta in a coarse-grained macro equation.",
        theory_tags=("rg", "noise", "markov"),
    ),
    FactorSpec(
        name="rg_coarse_grain_score",
        op="identity",
        inputs=("rg_coarse_grain_score",),
        family="multiscale",
        finance_origin="RG coarse-graining quality",
        dynamics_meaning="Quality of the low-dimensional collapse, high only when multiscale collapse and slow-manifold alignment agree.",
        theory_tags=("rg", "multiscale", "koopman"),
    ),
    FactorSpec(
        name="rg_beta_flow",
        op="identity",
        inputs=("rg_beta_flow",),
        family="order_parameter",
        finance_origin="RG beta flow",
        dynamics_meaning="Renormalized macro drift of the order parameter after weighting by coarse-graining quality.",
        theory_tags=("rg", "beta_function", "order_parameter"),
    ),
    FactorSpec(
        name="rg_critical_balance",
        op="identity",
        inputs=("rg_critical_balance",),
        family="criticality",
        finance_origin="RG critical balance",
        dynamics_meaning="Control pressure projected into the current critical window, highlighting likely regime transitions.",
        theory_tags=("rg", "criticality", "susceptibility"),
    ),
    FactorSpec(
        name="physics_drift_pred",
        op="identity",
        inputs=("id_drift_pred_norm",),
        family="physics_id",
        finance_origin="主力趋势预判/模型漂移预测",
        dynamics_meaning="物理识别器对慢流形下一步漂移的归一化预测。",
        theory_tags=("physics_identifier", "drift"),
    ),
    FactorSpec(
        name="physics_drift_surprise",
        op="identity",
        inputs=("id_drift_surprise_norm",),
        family="physics_id",
        finance_origin="预期差/超预期",
        dynamics_meaning="真实慢漂移相对识别器预测的偏差，用于捕捉机制切换。",
        theory_tags=("physics_identifier", "changepoint"),
    ),
    FactorSpec(
        name="physics_alignment",
        op="identity",
        inputs=("id_drift_alignment",),
        family="physics_id",
        finance_origin="趋势与成本同向",
        dynamics_meaning="识别器漂移与实际漂移的一致性符号。",
        theory_tags=("physics_identifier", "consistency"),
    ),
    FactorSpec(
        name="abs_gap_velocity",
        op="abs",
        inputs=("dm_norm",),
        family="order_parameter",
        finance_origin="速度绝对值",
        dynamics_meaning="序参量速度幅值，关注切换强度而非方向。",
        theory_tags=("criticality",),
        complexity=2,
    ),
    FactorSpec(
        name="positive_gap",
        op="relu",
        inputs=("m_norm",),
        family="order_parameter",
        finance_origin="只看金叉上方",
        dynamics_meaning="只保留正序参量部分，强调慢流形上方的有效推进。",
        theory_tags=("slow_fast",),
        complexity=2,
    ),
    FactorSpec(
        name="negative_retracement",
        op="neg_relu",
        inputs=("resid_norm",),
        family="phase",
        finance_origin="低位回撤幅度",
        dynamics_meaning="负残差的正部分，表征向慢流形回撤的深度。",
        theory_tags=("phase",),
        complexity=2,
    ),
    FactorSpec(
        name="trend_regression_quality",
        op="identity",
        inputs=("trend_regression_quality",),
        family="slow_manifold",
        finance_origin="QuantaAlpha RSQR / MA trend stability",
        dynamics_meaning="Causal trend-fit quality that stays high when drift remains coherent and residual noise stays low.",
        theory_tags=("trend", "slow_manifold", "koopman"),
    ),
    FactorSpec(
        name="positive_impulse_share",
        op="identity",
        inputs=("positive_impulse_share",),
        family="energy",
        finance_origin="QuantaAlpha VSUMP directional flow share",
        dynamics_meaning="Share of recent signed impulse carried by positive moves, analogous to persistent inflow support.",
        theory_tags=("energy", "activation", "trend"),
    ),
    FactorSpec(
        name="impulse_balance",
        op="identity",
        inputs=("impulse_balance",),
        family="energy",
        finance_origin="QuantaAlpha VSUMD up-minus-down flow imbalance",
        dynamics_meaning="Net signed impulse imbalance that distinguishes sustained push from symmetric churning.",
        theory_tags=("energy", "changepoint", "activation"),
    ),
    FactorSpec(
        name="band_position",
        op="identity",
        inputs=("band_position",),
        family="phase",
        finance_origin="QuantaAlpha RSV / MAX / MIN window position",
        dynamics_meaning="Relative location of the state inside its causal local range, useful for recovery and breakout timing.",
        theory_tags=("phase", "support", "oscillation"),
    ),
    FactorSpec(
        name="critical_memory_gate",
        op="pos_gate_product",
        inputs=("lag1_autocorr", "critical_window"),
        family="criticality",
        finance_origin="Critical slowing down early-warning signal",
        dynamics_meaning="Positive short-lag memory retained only when the system already sits inside a critical window, approximating local recovery-rate collapse.",
        theory_tags=("criticality", "memory", "slowing_down"),
        complexity=3,
    ),
    FactorSpec(
        name="critical_slowing_pressure",
        op="pos_gate_product",
        inputs=("lag1_autocorr", "energy_ratio"),
        family="criticality",
        finance_origin="Critical slowing down variance-memory coupling",
        dynamics_meaning="Variance amplification counted only when lag-1 memory has turned positive, matching classic early-warning behavior near transitions.",
        theory_tags=("criticality", "memory", "energy"),
        complexity=3,
    ),
    FactorSpec(
        name="memory_closure_load",
        op="pos_gate_product",
        inputs=("lag1_autocorr", "closure_stress"),
        family="markov",
        finance_origin="Mori-Zwanzig memory kernel load",
        dynamics_meaning="Closure stress that is kept only when short-lag memory is positive, highlighting unresolved fast-mode feedback.",
        theory_tags=("memory", "closure", "markov"),
        complexity=3,
    ),
    FactorSpec(
        name="isostable_bottom_recovery",
        op="product",
        inputs=("isostable_relaxation", "phase_bottom_score"),
        family="phase",
        finance_origin="Isostable relaxation plus bottom-turn evidence",
        dynamics_meaning="Lower-side recovery evidence that is strongest when amplitude deviation is already relaxing back toward the attractor.",
        theory_tags=("phase_amplitude", "recovery", "support"),
        complexity=3,
    ),
    FactorSpec(
        name="isostable_adiabatic_support",
        op="product",
        inputs=("isostable_relaxation", "adiabatic_coherence"),
        family="koopman",
        finance_origin="Adiabatic isostable support",
        dynamics_meaning="Fast deviation relaxes while the slow manifold remains coherent, approximating adiabatic return to a low-dimensional invariant structure.",
        theory_tags=("isostable", "adiabatic", "koopman"),
        complexity=3,
    ),
    FactorSpec(
        name="kramers_escape_pressure",
        op="ratio",
        inputs=("rg_noise_scale", "rg_critical_balance"),
        family="criticality",
        finance_origin="Kramers noise-assisted escape",
        dynamics_meaning="Fast agitation normalized by the current coarse-grained barrier pressure, highlighting likely regime escape under noise.",
        theory_tags=("kramers", "noise", "criticality"),
        complexity=3,
    ),
    FactorSpec(
        name="adiabatic_noise_shield",
        op="ratio",
        inputs=("adiabatic_coherence", "rg_noise_scale"),
        family="koopman",
        finance_origin="Adiabatic invariance under fast agitation",
        dynamics_meaning="Adiabatic coherence per unit fast agitation, favoring slow-manifold persistence over noise-driven pseudo-structure.",
        theory_tags=("adiabatic", "slow_fast", "noise"),
        complexity=3,
    ),
    FactorSpec(
        name="rg_order_control_coupling",
        op="product",
        inputs=("rg_order_parameter", "rg_control_parameter"),
        family="composite",
        finance_origin="RG order-control coupling",
        dynamics_meaning="Interaction between coarse-grained order and control pressure, useful when phase changes are control-driven.",
        theory_tags=("rg", "order_parameter", "control_parameter"),
        complexity=3,
    ),
    FactorSpec(
        name="rg_relevant_drive",
        op="product",
        inputs=("rg_control_parameter", "rg_coarse_grain_score"),
        family="composite",
        finance_origin="RG relevant drive",
        dynamics_meaning="Control pressure retained only when the coarse-grained description remains trustworthy.",
        theory_tags=("rg", "control_parameter", "multiscale"),
        complexity=3,
    ),
    FactorSpec(
        name="rg_flow_vs_noise",
        op="ratio",
        inputs=("rg_beta_flow", "rg_noise_scale"),
        family="composite",
        finance_origin="RG flow-versus-noise ratio",
        dynamics_meaning="Macro flow normalized by fast agitation, favoring stable coarse-grained motion over noisy one-step gain.",
        theory_tags=("rg", "beta_function", "markov"),
        complexity=3,
    ),
    FactorSpec(
        name="trend_energy_resonance",
        op="product",
        inputs=("trend_regression_quality", "energy_ratio"),
        family="composite",
        finance_origin="QuantaAlpha RSQR x WVMA/CORR resonance motifs",
        dynamics_meaning="Stable drift that is reinforced by elevated control energy instead of isolated noise.",
        theory_tags=("trend", "energy", "koopman"),
        complexity=3,
    ),
    FactorSpec(
        name="flow_supported_breakout",
        op="product",
        inputs=("positive_impulse_share", "breakout_strength"),
        family="composite",
        finance_origin="QuantaAlpha VSUMP + breakout seeds",
        dynamics_meaning="Order-parameter breakout supported by a dominant share of positive signed impulse.",
        theory_tags=("energy", "breakout", "trend"),
        complexity=3,
    ),
    FactorSpec(
        name="imbalance_recovery_gate",
        op="pos_gate_product",
        inputs=("impulse_balance", "phase_bottom_score"),
        family="composite",
        finance_origin="QuantaAlpha VSUMD + RSV/KDJ recovery motifs",
        dynamics_meaning="Bottom-recovery evidence kept only when directional impulse imbalance has already turned positive.",
        theory_tags=("phase", "support", "energy"),
        complexity=3,
    ),
    FactorSpec(
        name="drift_energy_alignment",
        op="product",
        inputs=("dm_norm", "impulse_balance"),
        family="composite",
        finance_origin="QuantaAlpha CORD price-volume alignment",
        dynamics_meaning="Directional drift aligned with signed impulse imbalance, highlighting resonant propagation rather than passive motion.",
        theory_tags=("trend", "energy", "activation"),
        complexity=3,
    ),
    FactorSpec(
        name="chart_position_confidence",
        op="product",
        inputs=("slow_level_norm", "rg_coarse_grain_score"),
        family="composite",
        finance_origin="Differential-geometric manifold chart confidence",
        dynamics_meaning="Slow-chart position retained only when the coarse-grained manifold remains geometrically trustworthy.",
        theory_tags=("slow_manifold", "chart", "multiscale"),
        complexity=3,
        source="theory_manifold",
        default_tier="experimental",
    ),
    FactorSpec(
        name="phase_chart_consistency",
        op="product",
        inputs=("band_position", "collapse_quality"),
        family="composite",
        finance_origin="Phase-isostable chart consistency",
        dynamics_meaning="Relative phase position reinforced only when the local manifold chart remains thin and coherent.",
        theory_tags=("phase", "chart", "multiscale"),
        complexity=3,
        source="theory_manifold",
        default_tier="experimental",
    ),
    FactorSpec(
        name="tangent_flow_confidence",
        op="product",
        inputs=("rg_beta_flow", "rg_coarse_grain_score"),
        family="composite",
        finance_origin="Tangent bundle flow confidence",
        dynamics_meaning="Tangential drift is trusted only when the local coarse-grained geometry remains well aligned.",
        theory_tags=("tangent", "rg", "multiscale", "drift"),
        complexity=3,
        source="theory_manifold",
        default_tier="experimental",
    ),
    FactorSpec(
        name="normal_escape_pressure",
        op="ratio",
        inputs=("rg_noise_scale", "slow_manifold_alignment"),
        family="composite",
        finance_origin="Normal-fiber escape pressure",
        dynamics_meaning="Fast agitation normalized by manifold alignment, highlighting when trajectories are pushed away from the slow geometric skeleton.",
        theory_tags=("normal", "slow_manifold", "noise", "closure"),
        complexity=3,
        source="theory_manifold",
        default_tier="experimental",
    ),
    FactorSpec(
        name="isostable_return_margin",
        op="product",
        inputs=("isostable_relaxation", "slow_manifold_alignment"),
        family="composite",
        finance_origin="Isostable return margin",
        dynamics_meaning="Amplitude relaxation retained only when the current state already lies near the slow manifold, approximating normal-fiber return strength.",
        theory_tags=("isostable", "slow_manifold", "recovery"),
        complexity=3,
        source="theory_manifold",
        default_tier="experimental",
    ),
    FactorSpec(
        name="closure_margin",
        op="ratio",
        inputs=("adiabatic_coherence", "closure_stress"),
        family="composite",
        finance_origin="Manifold closure margin",
        dynamics_meaning="Adiabatic coherence per unit closure burden, measuring how safely the current reduced coordinates stay inside a near-Markov regime.",
        theory_tags=("adiabatic", "markov", "closure", "koopman"),
        complexity=3,
        source="theory_manifold",
        default_tier="experimental",
    ),
    FactorSpec(
        name="memory_fiber_load",
        op="product",
        inputs=("lag1_autocorr", "rg_noise_scale"),
        family="composite",
        finance_origin="Unresolved memory-fiber load",
        dynamics_meaning="Short-lag memory amplified by fast agitation, indicating unresolved fiber dynamics beyond the current reduced coordinates.",
        theory_tags=("memory", "noise", "markov"),
        complexity=3,
        source="theory_manifold",
        default_tier="experimental",
    ),
    FactorSpec(
        name="coarse_chart_integrity",
        op="product",
        inputs=("collapse_quality", "timescale_separation"),
        family="composite",
        finance_origin="Principal-manifold chart integrity",
        dynamics_meaning="A low-dimensional chart is considered reliable only when the manifold remains thin and the slow-fast separation remains visible.",
        theory_tags=("multiscale", "spectral", "chart", "slow_fast"),
        complexity=3,
        source="theory_manifold",
        default_tier="experimental",
    ),
    FactorSpec(
        name="critical_softening_load",
        op="product",
        inputs=("critical_window", "lag1_autocorr"),
        family="composite",
        finance_origin="Critical softening load",
        dynamics_meaning="Critical-window exposure multiplied by short-lag memory, approximating local softening and slower recovery near a transition.",
        theory_tags=("criticality", "memory", "slow_fast"),
        complexity=3,
        source="theory_manifold",
        default_tier="experimental",
    ),
    FactorSpec(
        name="critical_escape_pressure",
        op="product",
        inputs=("critical_window", "rg_noise_scale"),
        family="composite",
        finance_origin="Critical escape pressure",
        dynamics_meaning="Noise pressure accumulated inside a critical window, highlighting likely boundary crossing or loss of normal hyperbolicity.",
        theory_tags=("criticality", "noise", "boundary"),
        complexity=3,
        source="theory_manifold",
        default_tier="experimental",
    ),
    FactorSpec(
        name="drive_off_manifold_pressure",
        op="product",
        inputs=("rg_control_parameter", "rg_noise_scale"),
        family="composite",
        finance_origin="Off-manifold control pressure",
        dynamics_meaning="Control or forcing pressure that acts together with fast agitation, favoring motion away from a stable coarse manifold instead of along it.",
        theory_tags=("control_parameter", "noise", "slow_manifold"),
        complexity=3,
        source="theory_manifold",
        default_tier="experimental",
    ),
    FactorSpec(
        name="chart_stability_margin",
        op="product",
        inputs=("adiabatic_coherence", "rg_coarse_grain_score"),
        family="composite",
        finance_origin="Koopman chart stability margin",
        dynamics_meaning="A phase-amplitude chart is most reliable when adiabatic persistence and coarse-grained geometry stay strong at the same time.",
        theory_tags=("adiabatic", "koopman", "chart", "multiscale"),
        complexity=3,
        source="theory_manifold",
        default_tier="experimental",
    ),
]


PAIRWISE_MUTATIONS: list[tuple[str, str, str, str, str, tuple[str, ...]]] = [
    ("gap_energy_coupling", "product", "m_norm", "energy_ratio", "均线金叉 + 放量", ("slow_fast", "energy")),
    ("phase_energy_coupling", "product", "phase_bottom_score", "energy_ratio", "低位回调 + 放量", ("phase", "energy")),
    ("critical_collapse_gate", "product", "critical_window", "collapse_quality", "临界窗口 + 多周期共振", ("criticality", "multiscale")),
    ("support_energy_gate", "product", "support_recovery", "energy_release", "回踩修复 + 能量释放", ("support", "energy")),
    ("breakout_multiscale_gate", "product", "breakout_strength", "collapse_quality", "突破 + 多尺度一致性", ("breakout", "multiscale")),
    ("physics_critical_gate", "product", "id_drift_pred_norm", "critical_window", "物理漂移预测 + 临界窗口", ("physics_identifier", "criticality")),
    ("physics_phase_gate", "product", "id_drift_surprise_norm", "phase_bottom_score", "预期差 + 回调相位", ("physics_identifier", "phase")),
    ("alignment_energy_gate", "product", "id_drift_alignment", "energy_ratio", "趋势一致 + 能量增强", ("physics_identifier", "energy")),
    ("retracement_support_gate", "pos_gate_product", "m_norm", "phase_bottom_score", "正趋势背景下的低位回调", ("support", "phase")),
    ("slowbreak_ratio", "ratio", "breakout_strength", "collapse_error", "突破强度相对塌缩误差", ("breakout", "rg")),
    ("surprise_to_gap_ratio", "ratio", "id_drift_surprise_norm", "m_norm", "物理预期差相对序参量", ("physics_identifier", "order_parameter")),
    ("energy_over_compression", "ratio", "energy_ratio", "compression_ratio", "能量释放相对压缩程度", ("energy", "compression")),
]


def _default_tier_for_spec(spec: FactorSpec) -> str:
    if spec.family in {"readout_interaction", "physics_id"}:
        return "experimental"
    if spec.complexity == 1 and spec.family in CORE_BASE_FAMILIES and "physics_identifier" not in spec.theory_tags:
        return "core"
    if spec.complexity >= 3 or "physics_identifier" in spec.theory_tags:
        return "experimental"
    return "extended"


def _decorate_spec(
    spec: FactorSpec,
    *,
    source: str | None = None,
    default_tier: str | None = None,
) -> FactorSpec:
    return replace(
        spec,
        source=source if source is not None else spec.source,
        default_tier=default_tier if default_tier is not None else _default_tier_for_spec(spec),
        manifold_role=infer_manifold_role(spec),
    )


def build_factor_bank(
    include_pairwise_mutations: bool = True,
    max_pairwise_mutations: int = 12,
) -> list[FactorSpec]:
    specs = [_decorate_spec(spec) for spec in BASE_FACTOR_SPECS]
    if include_pairwise_mutations:
        for name, op, a, b, origin, theory_tags in islice(PAIRWISE_MUTATIONS, 0, max_pairwise_mutations):
            specs.append(_decorate_spec(
                FactorSpec(
                    name=name,
                    op=op,
                    inputs=(a, b),
                    family="composite",
                    finance_origin=origin,
                    dynamics_meaning=f"组合因子：{a} 与 {b} 的 {op} 组合。",
                    theory_tags=theory_tags,
                    complexity=3,
                ),
                source="pairwise_mutation",
                default_tier="experimental",
            ))
    return specs


def attach_formula_metadata(specs: Iterable[FactorSpec]) -> list[dict[str, object]]:
    rows = []
    for spec in specs:
        rows.append(
            {
                **spec.to_dict(),
                "formula": factor_formula(spec),
            }
        )
    return rows
