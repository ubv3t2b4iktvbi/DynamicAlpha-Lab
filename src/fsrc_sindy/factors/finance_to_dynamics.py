from __future__ import annotations

TRANSLATION_TABLE = [
    {
        "finance_pattern": "均线金叉 / 白线上穿黄线",
        "dynamics_object": "快慢流形序参量 m 与其符号翻转",
        "project_feature": "slow_fast_gap, positive_gap",
        "mechanism_note": "快流形脱离慢流形，说明系统可能进入新相或新吸引子邻域。",
    },
    {
        "finance_pattern": "KDJ 低位 / 最后一次回调",
        "dynamics_object": "相位谷底上拐证据",
        "project_feature": "phase_bottom_score, support_recovery",
        "mechanism_note": "在负残差区出现正相位推进，常对应回调末端与相位提前。",
    },
    {
        "finance_pattern": "放量异动 / 主力入场",
        "dynamics_object": "控制参量增强与能量注入",
        "project_feature": "energy_ratio, energy_release",
        "mechanism_note": "创新功率或局部动能相对背景显著抬升。",
    },
    {
        "finance_pattern": "主力成本区 / 多空线",
        "dynamics_object": "慢流形或慢参考轨道",
        "project_feature": "slow, trend_persistence",
        "mechanism_note": "慢变量刻画长期约束或低频支撑/阻尼。",
    },
    {
        "finance_pattern": "多周期共振 / 主曲线质量",
        "dynamics_object": "多尺度塌缩一致性",
        "project_feature": "collapse_quality, critical_collapse_gate",
        "mechanism_note": "不同尺度的无量纲变量在同一主曲线附近，说明 coarse-graining 更稳定。",
    },
    {
        "finance_pattern": "预期差 / 关键K 管理区间",
        "dynamics_object": "物理识别残差与机制切换证据",
        "project_feature": "physics_drift_surprise, physics_phase_gate",
        "mechanism_note": "当识别器预测与真实漂移偏离时，可能发生机制切换或未知驱动注入。",
    },
]


def translation_markdown() -> str:
    lines = [
        "# 金融因子到动力学因子的翻译表",
        "",
        "| 金融语义 | 动力学对象 | 项目内因子 | 机制说明 |",
        "|---|---|---|---|",
    ]
    for row in TRANSLATION_TABLE:
        lines.append(
            f"| {row['finance_pattern']} | {row['dynamics_object']} | {row['project_feature']} | {row['mechanism_note']} |"
        )
    return "\n".join(lines)
