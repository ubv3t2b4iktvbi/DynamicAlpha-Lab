from __future__ import annotations

from itertools import islice
from typing import Iterable, Mapping

import numpy as np

from .base import FactorSpec


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


def build_factor_bank(
    include_pairwise_mutations: bool = True,
    max_pairwise_mutations: int = 12,
) -> list[FactorSpec]:
    specs = list(BASE_FACTOR_SPECS)
    if include_pairwise_mutations:
        for name, op, a, b, origin, theory_tags in islice(PAIRWISE_MUTATIONS, 0, max_pairwise_mutations):
            specs.append(
                FactorSpec(
                    name=name,
                    op=op,
                    inputs=(a, b),
                    family="composite",
                    finance_origin=origin,
                    dynamics_meaning=f"组合因子：{a} 与 {b} 的 {op} 组合。",
                    theory_tags=theory_tags,
                    complexity=3,
                )
            )
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
