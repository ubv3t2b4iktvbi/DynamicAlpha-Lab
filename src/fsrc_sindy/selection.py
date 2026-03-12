from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Sequence

import numpy as np
from tqdm.auto import tqdm

from .factors.base import DynamicsFeatureConfig, FactorSpec
from .fastslow import FastSlowConfig
from .metrics import evaluate_horizons
from .models import (
    FullObservableSINDy,
    FullSINDyConfig,
    HybridRCNGRCModel,
    NGRCConfig,
    PureNGRCModel,
    PureRCModel,
    RCConfig,
    RCNGRCConfig,
    ReservoirTemplateFactory,
    ResidualLinearConfig,
    ResidualNGRCConfig,
    ResidualRCConfig,
    ResidualRCNGRCConfig,
    SlowSINDyConfig,
    SlowSINDyDeltaHybridModel,
    SlowSINDyDeltaLinearModel,
    SlowSINDyDeltaNGRCModel,
    SlowSINDyDeltaRCModel,
    SlowSINDyLevelLinearModel,
    SlowSINDyLevelRCModel,
    SlowSINDyOnlyModel,
)


@dataclass(frozen=True)
class ModelSpec:
    family: str
    uses_fastslow: bool
    uses_sindy_backbone: bool
    uses_reservoir: bool
    uses_ngrc: bool
    residual_mode: str
    description: str


@dataclass(frozen=True)
class SlowResidualSearchConfig:
    slow_cfg: SlowSINDyConfig
    residual_cfg: Any


MODEL_SPECS: dict[str, ModelSpec] = {
    "rc_raw": ModelSpec("reservoir", False, False, True, False, "none", "RC baseline with direct readout."),
    "rc_fastslow_readout": ModelSpec("reservoir", True, False, True, False, "none", "RC baseline with fast/slow readout features."),
    "rc_factor_readout": ModelSpec("reservoir", False, False, True, False, "none", "RC baseline with task-selected factor readout."),
    "ngrc_raw": ModelSpec("ngrc", False, False, False, True, "none", "NGRC/NVAR baseline on delay coordinates."),
    "ngrc_fastslow_readout": ModelSpec("ngrc", True, False, False, True, "none", "NGRC augmented with fast/slow readout features."),
    "ngrc_factor_readout": ModelSpec("ngrc", False, False, False, True, "none", "NGRC baseline with task-selected factor readout."),
    "hybrid_rc_ngrc_fastslow": ModelSpec("hybrid_memory", True, False, True, True, "none", "Joint RC + NGRC readout with fast/slow features."),
    "sindy_full": ModelSpec("sindy", True, False, False, False, "none", "Full observable SINDy proxy built from scalar fast/slow features."),
    "slow_sindy_only": ModelSpec("structured_sindy", True, True, False, False, "none", "Slow manifold SINDy backbone without residual closure."),
    "slow_sindy_delta_linear": ModelSpec("structured_residual", True, True, False, False, "delta_linear", "Slow SINDy backbone plus linear delta residual."),
    "slow_sindy_delta_rc": ModelSpec("structured_residual", True, True, True, False, "delta_rc", "Slow SINDy backbone plus RC residual delta closure."),
    "slow_sindy_delta_ngrc": ModelSpec("structured_residual", True, True, False, True, "delta_ngrc", "Slow SINDy backbone plus NGRC residual delta closure."),
    "slow_sindy_delta_hybrid": ModelSpec("structured_residual", True, True, True, True, "delta_hybrid", "Slow SINDy backbone plus hybrid RC+NGRC residual delta closure."),
    "slow_sindy_level_linear": ModelSpec("legacy", True, True, False, False, "level_linear", "Legacy level residual linear model."),
    "slow_sindy_level_rc": ModelSpec("legacy", True, True, True, False, "level_rc", "Legacy level residual RC model."),
}

DEFAULT_MODEL_NAMES = [
    "rc_raw",
    "rc_fastslow_readout",
    "ngrc_raw",
    "ngrc_fastslow_readout",
    "hybrid_rc_ngrc_fastslow",
    "sindy_full",
    "slow_sindy_only",
    "slow_sindy_delta_rc",
    "slow_sindy_delta_ngrc",
]

RESEARCH_MODEL_NAMES = [
    "rc_fastslow_readout",
    "ngrc_fastslow_readout",
    "hybrid_rc_ngrc_fastslow",
    "slow_sindy_only",
    "slow_sindy_delta_rc",
    "slow_sindy_delta_ngrc",
    "slow_sindy_delta_hybrid",
]

LEGACY_MODEL_NAMES = ["slow_sindy_delta_linear", "slow_sindy_level_linear", "slow_sindy_level_rc"]

MODEL_GROUPS: dict[str, list[str]] = {
    "default": list(DEFAULT_MODEL_NAMES),
    "general": ["rc_raw", "rc_fastslow_readout", "ngrc_raw", "ngrc_fastslow_readout", "hybrid_rc_ngrc_fastslow", "sindy_full"],
    "base":["rc_raw", "ngrc_raw"],
    "fastslow_ablation": ["rc_raw", "rc_fastslow_readout", "ngrc_raw", "ngrc_fastslow_readout"],
    "memory_ablation": ["rc_fastslow_readout", "ngrc_fastslow_readout", "hybrid_rc_ngrc_fastslow"],
    "structured_ablation": ["slow_sindy_only", "slow_sindy_delta_linear", "slow_sindy_delta_rc", "slow_sindy_delta_ngrc", "slow_sindy_delta_hybrid"],
    "research_core": list(RESEARCH_MODEL_NAMES),
    "legacy": list(LEGACY_MODEL_NAMES),
}

ABLATION_COMPARISONS = [
    {
        "name": "fastslow_on_rc",
        "baseline": "rc_raw",
        "candidate": "rc_fastslow_readout",
        "hypothesis": "Fast/slow readout should help under scalar partial observation at matched RC state size.",
    },
    {
        "name": "fastslow_on_ngrc",
        "baseline": "ngrc_raw",
        "candidate": "ngrc_fastslow_readout",
        "hypothesis": "Fast/slow readout should help NGRC resolve latent slow context.",
    },
    {
        "name": "dual_memory_vs_single_memory",
        "baseline": "rc_fastslow_readout",
        "candidate": "hybrid_rc_ngrc_fastslow",
        "hypothesis": "Combining recurrent and delay memories should improve horizon balance.",
    },
    {
        "name": "slow_backbone_vs_none",
        "baseline": "hybrid_rc_ngrc_fastslow",
        "candidate": "slow_sindy_delta_hybrid",
        "hypothesis": "A slow-manifold prior should help on multiscale high-dimensional tasks.",
    },
    {
        "name": "hybrid_residual_vs_ngrc_residual",
        "baseline": "slow_sindy_delta_ngrc",
        "candidate": "slow_sindy_delta_hybrid",
        "hypothesis": "Adding reservoir memory should stabilize residual NGRC rollouts.",
    },
]


def expand_model_group_names(group_names: Sequence[str] | None) -> list[str]:
    if not group_names:
        return []
    models: list[str] = []
    for group_name in group_names:
        if group_name not in MODEL_GROUPS:
            raise ValueError(f"Unknown model_group={group_name}")
        for model_name in MODEL_GROUPS[group_name]:
            if model_name not in models:
                models.append(model_name)
    return models


def get_model_spec(model_name: str) -> ModelSpec:
    try:
        return MODEL_SPECS[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown model_name={model_name}") from exc


def build_rc_grid(mode: str = "quick") -> list[RCConfig]:
    if mode == "quick":
        sizes = [120, 240]
        radii = [0.9, 1.1]
        leaks = [0.2, 0.6, 1.0]
    else:
        sizes = [96, 160, 240, 320]
        radii = [0.8, 1.0, 1.2]
        leaks = [0.2, 0.5, 0.8, 1.0]
    return [
        RCConfig(n_reservoir=n, spectral_radius=sr, input_scale=0.5, leak_rate=lk, ridge=1e-5, sparsity=0.05, washout=100)
        for n in sizes for sr in radii for lk in leaks
    ]


def build_ngrc_grid(mode: str = "quick", short_train: bool = False) -> list[NGRCConfig]:
    if mode == "quick":
        delays = [10, 14] if short_train else [14, 20]
        ridges = [1e-6, 1e-5, 1e-4]
        clips = [5.0]
    else:
        delays = [8, 10, 14, 18, 20]
        ridges = [1e-6, 1e-5, 1e-4, 1e-3]
        clips = [3.0, 5.0, 8.0]
    return [
        NGRCConfig(n_delays=d, stride=1, poly_order=2, ridge=r, washout=max(25, d), feature_clip=fc, y_clip=12.0)
        for d in delays for r in ridges for fc in clips
    ]


def build_rc_ngrc_grid(mode: str = "quick", short_train: bool = False) -> list[RCNGRCConfig]:
    if mode == "quick":
        pairs = [(48, 8), (64, 10)] if short_train else [(64, 10), (120, 14)]
        radii = [0.9, 1.1]
        leaks = [0.4, 0.8]
    else:
        pairs = [(48, 8), (64, 10), (96, 12), (120, 14)]
        radii = [0.8, 1.0, 1.2]
        leaks = [0.3, 0.6, 0.9]
    out = []
    for n_reservoir, n_delays in pairs:
        for sr in radii:
            for lk in leaks:
                out.append(
                    RCNGRCConfig(
                        n_reservoir=n_reservoir,
                        spectral_radius=sr,
                        input_scale=0.5,
                        leak_rate=lk,
                        ridge=1e-5,
                        sparsity=0.05,
                        washout=max(50, n_delays),
                        n_delays=n_delays,
                        stride=1,
                        poly_order=2,
                        feature_clip=5.0,
                        y_clip=12.0,
                    )
                )
    return out


def build_residual_ngrc_grid(mode: str = "quick", short_train: bool = False) -> list[ResidualNGRCConfig]:
    if mode == "quick":
        delays = [8, 10] if short_train else [10, 14]
        ridges = [1e-5, 1e-4]
        delta_clips = [0.5, 1.0]
        damps = [0.5, 0.8]
    else:
        delays = [8, 10, 12, 14]
        ridges = [1e-6, 1e-5, 1e-4]
        delta_clips = [0.25, 0.5, 1.0]
        damps = [0.4, 0.6, 0.8]
    return [
        ResidualNGRCConfig(
            n_delays=d,
            stride=1,
            poly_order=2,
            ridge=r,
            washout=max(25, d),
            feature_clip=5.0,
            y_clip=12.0,
            delta_clip=dc,
            resid_clip=5.0,
            damp=dm,
        )
        for d in delays for r in ridges for dc in delta_clips for dm in damps
    ]


def build_residual_rc_ngrc_grid(mode: str = "quick", short_train: bool = False) -> list[ResidualRCNGRCConfig]:
    base = build_rc_ngrc_grid(mode, short_train=short_train)
    out = []
    for cfg in base:
        delta_clips = [0.5, 1.0] if mode == "quick" else [0.25, 0.5, 1.0]
        damps = [0.5, 0.8] if mode == "quick" else [0.4, 0.6, 0.8]
        resid_clips = [5.0] if mode == "quick" else [4.0, 5.0, 6.0]
        for dc in delta_clips:
            for damp in damps:
                for resid_clip in resid_clips:
                    out.append(
                        ResidualRCNGRCConfig(
                            **asdict(cfg),
                            delta_clip=dc,
                            resid_clip=resid_clip,
                            damp=damp,
                        )
                    )
    return out


def build_fastslow_grid(data_dt: float, mode: str = "quick", short_train: bool = False) -> list[FastSlowConfig]:
    # continuous step-equivalent parameterization; slow scales are constrained as (n, 2n, 4n)
    if mode == "quick":
        fast_steps = [3.5, 5.0] if short_train else [4.0, 6.0, 8.0]
        slow_bases = [8.0, 12.0] if short_train else [12.0, 16.0, 24.0]
    else:
        fast_steps = [3.0, 4.0, 5.5, 7.0] if short_train else [3.5, 4.5, 6.0, 8.0, 10.0]
        slow_bases = [6.0, 8.0, 12.0] if short_train else [8.0, 12.0, 16.0, 24.0, 32.0]
    return [
        FastSlowConfig(t0=fast_n, slow_scales=(slow_n, 2.0 * slow_n, 4.0 * slow_n), dt=data_dt)
        for fast_n in fast_steps for slow_n in slow_bases
    ]


def build_structured_fastslow_grid(data_dt: float, mode: str = "quick", short_train: bool = False) -> list[FastSlowConfig]:
    if mode == "quick":
        fast_steps = [4.0] if short_train else [4.0, 6.0]
        slow_bases = [8.0, 12.0] if short_train else [12.0, 16.0]
    else:
        fast_steps = [3.5, 5.0] if short_train else [4.0, 6.0, 8.0]
        slow_bases = [8.0, 12.0, 16.0] if short_train else [12.0, 16.0, 24.0]
    return [
        FastSlowConfig(t0=fast_n, slow_scales=(slow_n, 2.0 * slow_n, 4.0 * slow_n), dt=data_dt)
        for fast_n in fast_steps for slow_n in slow_bases
    ]


def with_fastslow_cfgs(configs: list[Any], fs_cfgs: list[FastSlowConfig]) -> list[Any]:
    return [replace(cfg, fs_cfg=fs_cfg) for cfg in configs for fs_cfg in fs_cfgs]


def build_full_sindy_grid(mode: str = "quick", data_dt: float = 1.0) -> list[FullSINDyConfig]:
    fs_cfgs = build_fastslow_grid(data_dt=data_dt, mode=mode, short_train=False)
    thresholds = [1e-4, 1e-3] if mode == "quick" else [1e-4, 5e-4, 1e-3]
    return [
        FullSINDyConfig(fs_cfg=fs_cfg, poly_order=2, threshold=thr, ridge=1e-6)
        for fs_cfg in fs_cfgs for thr in thresholds
    ]


def build_slow_cfg_grid(data_dt: float, short_train: bool = False, mode: str = "quick") -> list[SlowSINDyConfig]:
    return [
        SlowSINDyConfig(fs_cfg=fs_cfg, poly_order=2, threshold=1e-4, ridge=1e-6)
        for fs_cfg in build_structured_fastslow_grid(data_dt=data_dt, mode=mode, short_train=short_train)
    ]


def build_residual_linear_grid(mode: str = "quick") -> list[ResidualLinearConfig]:
    if mode == "quick":
        ridges = [1e-3, 1e-4]
        delta_clips = [0.5, 1.0]
        damps = [0.7, 1.0]
    else:
        ridges = [1e-2, 1e-3, 1e-4, 1e-5]
        delta_clips = [0.25, 0.5, 1.0]
        damps = [0.5, 0.7, 1.0]
    return [
        ResidualLinearConfig(ridge=r, washout=50, delta_clip=dc, resid_clip=5.0, damp=d)
        for r in ridges for dc in delta_clips for d in damps
    ]


def build_residual_rc_grid(mode: str = "quick") -> list[ResidualRCConfig]:
    base = build_rc_grid(mode)
    out = []
    for cfg in base:
        delta_clips = [0.5, 1.0] if mode == "quick" else [0.25, 0.5, 1.0]
        damps = [0.5, 0.7] if mode == "quick" else [0.4, 0.6, 0.8]
        for dc in delta_clips:
            for damp in damps:
                out.append(ResidualRCConfig(**asdict(cfg), delta_clip=dc, resid_clip=5.0, damp=damp))
    return out


def _coerce_factor_specs(payload: Any) -> list[FactorSpec]:
    if payload is None:
        return []
    specs: list[FactorSpec] = []
    for item in payload:
        if isinstance(item, FactorSpec):
            specs.append(item)
        elif isinstance(item, dict):
            specs.append(FactorSpec(**item))
        else:
            raise TypeError(f"Unsupported factor spec payload: {type(item)!r}")
    return specs


def _coerce_feature_cfg(payload: Any) -> DynamicsFeatureConfig | None:
    if payload is None or isinstance(payload, DynamicsFeatureConfig):
        return payload
    if isinstance(payload, dict):
        return DynamicsFeatureConfig(**payload)
    raise TypeError(f"Unsupported feature config payload: {type(payload)!r}")


def _factor_readout_context(
    model_name: str,
    model_context: dict[str, Any] | None,
) -> tuple[list[FactorSpec], str, DynamicsFeatureConfig | None]:
    if model_context is None:
        raise ValueError(f"{model_name} requires model_context with task-selected readout factors")
    factor_specs = _coerce_factor_specs(model_context.get("readout_factor_specs"))
    identifier_kind = str(model_context.get("readout_identifier_kind", "")).strip()
    feature_cfg = _coerce_feature_cfg(model_context.get("readout_feature_cfg"))
    if not factor_specs:
        raise ValueError(f"{model_name} requires at least one selected readout factor")
    if not identifier_kind:
        raise ValueError(f"{model_name} requires readout_identifier_kind in model_context")
    return factor_specs, identifier_kind, feature_cfg


def get_search_space(model_name: str, grid_mode: str, short_train: bool, data_dt: float) -> list[Any]:
    if model_name == "rc_raw":
        return build_rc_grid(grid_mode)
    if model_name == "rc_fastslow_readout":
        return with_fastslow_cfgs(build_rc_grid(grid_mode), build_fastslow_grid(data_dt=data_dt, mode=grid_mode, short_train=short_train))
    if model_name == "rc_factor_readout":
        return build_rc_grid(grid_mode)
    if model_name == "ngrc_raw":
        return build_ngrc_grid(grid_mode, short_train=short_train)
    if model_name == "ngrc_fastslow_readout":
        return with_fastslow_cfgs(build_ngrc_grid(grid_mode, short_train=short_train), build_fastslow_grid(data_dt=data_dt, mode=grid_mode, short_train=short_train))
    if model_name == "ngrc_factor_readout":
        return build_ngrc_grid(grid_mode, short_train=short_train)
    if model_name == "hybrid_rc_ngrc_fastslow":
        return with_fastslow_cfgs(build_rc_ngrc_grid(grid_mode, short_train=short_train), build_fastslow_grid(data_dt=data_dt, mode=grid_mode, short_train=short_train))
    if model_name == "sindy_full":
        return build_full_sindy_grid(grid_mode, data_dt=data_dt)
    if model_name == "slow_sindy_only":
        return build_slow_cfg_grid(data_dt=data_dt, short_train=short_train, mode=grid_mode)
    if model_name in {"slow_sindy_delta_linear", "slow_sindy_level_linear"}:
        return [
            SlowResidualSearchConfig(slow_cfg=slow_cfg, residual_cfg=resid_cfg)
            for slow_cfg in build_slow_cfg_grid(data_dt=data_dt, short_train=short_train, mode=grid_mode)
            for resid_cfg in build_residual_linear_grid(grid_mode)
        ]
    if model_name in {"slow_sindy_delta_rc", "slow_sindy_level_rc"}:
        return [
            SlowResidualSearchConfig(slow_cfg=slow_cfg, residual_cfg=resid_cfg)
            for slow_cfg in build_slow_cfg_grid(data_dt=data_dt, short_train=short_train, mode=grid_mode)
            for resid_cfg in build_residual_rc_grid(grid_mode)
        ]
    if model_name == "slow_sindy_delta_ngrc":
        return [
            SlowResidualSearchConfig(slow_cfg=slow_cfg, residual_cfg=resid_cfg)
            for slow_cfg in build_slow_cfg_grid(data_dt=data_dt, short_train=short_train, mode=grid_mode)
            for resid_cfg in build_residual_ngrc_grid(grid_mode, short_train=short_train)
        ]
    if model_name == "slow_sindy_delta_hybrid":
        return [
            SlowResidualSearchConfig(slow_cfg=slow_cfg, residual_cfg=resid_cfg)
            for slow_cfg in build_slow_cfg_grid(data_dt=data_dt, short_train=short_train, mode=grid_mode)
            for resid_cfg in build_residual_rc_ngrc_grid(grid_mode, short_train=short_train)
        ]
    raise ValueError(f"Unknown model_name={model_name}")


def instantiate_model(
    model_name: str,
    cfg: Any,
    template_factory: ReservoirTemplateFactory,
    short_train: bool,
    model_context: dict[str, Any] | None = None,
):
    default_slow_cfg = build_slow_cfg_grid(data_dt=1.0, short_train=short_train, mode="quick")[0]
    if model_name == "rc_raw":
        return PureRCModel(cfg=cfg, template_factory=template_factory, fs_cfg=default_slow_cfg.fs_cfg, use_fastslow_readout=False)
    if model_name == "rc_fastslow_readout":
        return PureRCModel(cfg=cfg, template_factory=template_factory, fs_cfg=getattr(cfg, "fs_cfg", None), use_fastslow_readout=True)
    if model_name == "rc_factor_readout":
        factor_specs, identifier_kind, feature_cfg = _factor_readout_context(model_name, model_context)
        return PureRCModel(
            cfg=cfg,
            template_factory=template_factory,
            fs_cfg=getattr(cfg, "fs_cfg", None) or default_slow_cfg.fs_cfg,
            use_fastslow_readout=False,
            readout_factor_specs=factor_specs,
            readout_identifier_kind=identifier_kind,
            readout_feature_cfg=feature_cfg,
        )
    if model_name == "ngrc_raw":
        return PureNGRCModel(cfg=cfg, fs_cfg=default_slow_cfg.fs_cfg, use_fastslow_readout=False)
    if model_name == "ngrc_fastslow_readout":
        return PureNGRCModel(cfg=cfg, fs_cfg=getattr(cfg, "fs_cfg", None), use_fastslow_readout=True)
    if model_name == "ngrc_factor_readout":
        factor_specs, identifier_kind, feature_cfg = _factor_readout_context(model_name, model_context)
        return PureNGRCModel(
            cfg=cfg,
            fs_cfg=getattr(cfg, "fs_cfg", None) or default_slow_cfg.fs_cfg,
            use_fastslow_readout=False,
            readout_factor_specs=factor_specs,
            readout_identifier_kind=identifier_kind,
            readout_feature_cfg=feature_cfg,
        )
    if model_name == "hybrid_rc_ngrc_fastslow":
        return HybridRCNGRCModel(cfg=cfg, template_factory=template_factory, fs_cfg=getattr(cfg, "fs_cfg", None), use_fastslow_readout=True)
    if model_name == "sindy_full":
        return FullObservableSINDy(cfg)
    if model_name == "slow_sindy_only":
        return SlowSINDyOnlyModel(cfg)
    if model_name == "slow_sindy_delta_linear":
        return SlowSINDyDeltaLinearModel(cfg.slow_cfg, cfg.residual_cfg)
    if model_name == "slow_sindy_delta_rc":
        return SlowSINDyDeltaRCModel(cfg.slow_cfg, cfg.residual_cfg, template_factory=template_factory)
    if model_name == "slow_sindy_delta_ngrc":
        return SlowSINDyDeltaNGRCModel(cfg.slow_cfg, cfg.residual_cfg)
    if model_name == "slow_sindy_delta_hybrid":
        return SlowSINDyDeltaHybridModel(cfg.slow_cfg, cfg.residual_cfg, template_factory=template_factory)
    if model_name == "slow_sindy_level_linear":
        return SlowSINDyLevelLinearModel(cfg.slow_cfg, cfg.residual_cfg)
    if model_name == "slow_sindy_level_rc":
        return SlowSINDyLevelRCModel(cfg.slow_cfg, cfg.residual_cfg, template_factory=template_factory)
    raise ValueError(f"Unknown model_name={model_name}")


def validation_score(metrics: dict[str, float], score_horizons: Sequence[int], y_scale: float, max_abs_threshold: float = 10.0) -> float:
    score = 0.0
    for H in score_horizons:
        v = float(metrics.get(f"nrmse@{H}", np.inf))
        if np.isfinite(v):
            score += min(v, 1e6)
        else:
            score += 1e6
    rollout_finite = metrics.get("rollout_finite", 0.0)
    rollout_max_abs = metrics.get("rollout_max_abs", np.inf)
    if rollout_finite < 0.5:
        score += 1e6
    if not np.isfinite(rollout_max_abs):
        score += 1e6
    elif rollout_max_abs > max_abs_threshold * max(y_scale, 1e-6):
        score += 1000.0 + 0.1 * rollout_max_abs
    return float(score)


def select_best_model(
    model_name: str,
    y_train: np.ndarray,
    y_val: np.ndarray,
    context_len: int,
    score_horizons: Sequence[int],
    grid_mode: str,
    template_factory: ReservoirTemplateFactory,
    short_train: bool,
    progress_desc: str,
    data_dt: float,
    model_context: dict[str, Any] | None = None,
):
    search_space = get_search_space(model_name=model_name, grid_mode=grid_mode, short_train=short_train, data_dt=data_dt)
    best_model = None
    best_cfg = None
    best_metrics = None
    best_score = float("inf")
    context = np.concatenate([y_train[-context_len:], y_val[:1]])
    future = y_val[1:1 + max(score_horizons)]
    y_scale = float(np.std(y_train) + 1e-12)
    for cfg in tqdm(search_space, desc=progress_desc, leave=False):
        model = instantiate_model(
            model_name,
            cfg,
            template_factory,
            short_train=short_train,
            model_context=model_context,
        ).fit(y_train)
        metrics = evaluate_horizons(model, context, future, score_horizons)
        score = validation_score(metrics, score_horizons=score_horizons, y_scale=y_scale)
        if best_model is None or score < best_score:
            best_score = score
            best_model = model
            best_cfg = cfg
            best_metrics = metrics
    return best_model, best_metrics, best_cfg
