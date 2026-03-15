from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from ..experiment import run_benchmark_seed_sweep
from ..utils import ensure_dir
from .coordinate_analysis import run_coordinate_analysis_seed_sweep


DEFAULT_VALIDATION_SEEDS: tuple[int, ...] = (123, 231, 341, 451, 561)

MECHANISM_CORE_TASKS: tuple[str, ...] = (
    "lorenz96_twoscale_obs_slow0",
    "lorenz96_twoscale_obs_mixed_projection",
    "lorenz96_twoscale_gate_s2f_0p0",
    "lorenz96_twoscale_gate_s2f_0p8",
)

CLASSIC_CORE_TASKS: tuple[str, ...] = (
    "vanderpol_relaxation_noisy",
    "fitzhugh_nagumo_classic_noisy",
)

BOUNDARY_TASKS: tuple[str, ...] = (
    "lorenz96_twoscale_gate_s2f_0p0",
    "lorenz96_twoscale_gate_s2f_0p4",
    "lorenz96_twoscale_gate_s2f_0p8",
    "lorenz96_twoscale_gate_s2f_1p2",
    "lorenz96_twoscale_gate_s2f_1p6",
    "lorenz96_twoscale_obs_slow0",
    "lorenz96_twoscale_obs_sparse_slowproj",
    "lorenz96_twoscale_obs_mixed_projection",
)

DELAY_SWEEP_TASKS: tuple[str, ...] = (
    "lorenz96_twoscale_obs_slow0",
    "lorenz96_twoscale_obs_mixed_projection",
    "lorenz96_twoscale_gate_s2f_0p8",
)

ROBUSTNESS_CLASSIC_TASKS: tuple[str, ...] = (
    "vanderpol_relaxation_clean",
    "vanderpol_relaxation_noisy",
    "fitzhugh_nagumo_classic_clean",
    "fitzhugh_nagumo_classic_noisy",
    "hindmarsh_rose_bursting_clean",
    "hindmarsh_rose_bursting_noisy",
)

ROBUSTNESS_FINANCE_TASKS: tuple[str, ...] = (
    "vanderpol_relaxation_volclustered",
    "fitzhugh_nagumo_classic_volclustered",
    "hindmarsh_rose_bursting_volclustered",
)

ROBUSTNESS_MECHANISM_NOISE_TASKS: tuple[str, ...] = (
    "lorenz96_twoscale_noise_homoskedastic",
    "lorenz96_twoscale_noise_matched_clustered",
)

COORDINATE_TASKS: tuple[str, ...] = MECHANISM_CORE_TASKS + CLASSIC_CORE_TASKS

CORE_VARIANT_ORDER: tuple[str, ...] = (
    "ngrc_raw",
    "ngrc_rg_readout",
    "ngrc_takens_rg_additive",
    "ngrc_takens_rg_true",
    "ngrc_sf_rg_gated",
)


@dataclass(frozen=True)
class BenchmarkVariant:
    base_model_name: str
    variant_label: str
    model_context: dict[str, Any] | None = None


def _task_contexts(task_names: Sequence[str], variant: BenchmarkVariant) -> dict[str, dict[str, dict[str, object]]]:
    context = dict(variant.model_context or {})
    context["variant_label"] = variant.variant_label
    return {str(task): {variant.base_model_name: context} for task in task_names}


def _run_variant_collection(
    suite: str,
    task_names: Sequence[str],
    seeds: Sequence[int],
    variants: Sequence[BenchmarkVariant],
    out_dir: Path,
    grid_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dir(out_dir)
    combined_runs: list[pd.DataFrame] = []
    combined_summaries: list[pd.DataFrame] = []
    for variant in variants:
        variant_dir = out_dir / variant.variant_label
        _, summary = run_benchmark_seed_sweep(
            suite=suite,
            seeds=seeds,
            out_dir=str(variant_dir),
            model_names=[variant.base_model_name],
            task_model_contexts=_task_contexts(task_names, variant),
            grid_mode=grid_mode,
            task_names=task_names,
        )
        runs = pd.read_csv(variant_dir / "benchmark_seed_results.csv")
        runs["experiment_variant"] = variant.variant_label
        combined_runs.append(runs)
        if not summary.empty:
            summary = summary.copy()
            summary["experiment_variant"] = variant.variant_label
            combined_summaries.append(summary)
    all_runs = pd.concat(combined_runs, axis=0, ignore_index=True) if combined_runs else pd.DataFrame()
    all_summary = pd.concat(combined_summaries, axis=0, ignore_index=True) if combined_summaries else pd.DataFrame()
    all_runs.to_csv(out_dir / "benchmark_seed_results_all.csv", index=False)
    all_summary.to_csv(out_dir / "benchmark_seed_summary_all.csv", index=False)
    return all_runs, all_summary


def _delay_search_space(delays: Sequence[int], strides: Sequence[int], ridges: Sequence[float]) -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []
    for stride in strides:
        for delay in delays:
            for ridge in ridges:
                configs.append(
                    {
                        "n_delays": int(delay),
                        "stride": int(stride),
                        "poly_order": 2,
                        "ridge": float(ridge),
                        "washout": max(25, int(delay) * int(stride)),
                        "feature_clip": 5.0,
                        "y_clip": 12.0,
                    }
                )
    return configs


def _validation_search_space() -> list[dict[str, object]]:
    return _delay_search_space(delays=(12, 16), strides=(1,), ridges=(1e-6, 1e-5))


def _render_markdown_table(df: pd.DataFrame, cols: Sequence[str], sort_cols: Sequence[str] | None = None) -> str:
    if df.empty:
        return "No rows were produced."
    keep = [col for col in cols if col in df.columns]
    table = df[keep]
    if sort_cols:
        actual = [col for col in sort_cols if col in table.columns]
        if actual:
            table = table.sort_values(actual)
    return table.to_markdown(index=False)


def _best_variant_table(summary_df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    if summary_df.empty or metric_col not in summary_df.columns:
        return pd.DataFrame()
    ordered = summary_df.sort_values(["task", metric_col, "variant"]).reset_index(drop=True)
    return ordered.groupby("task", as_index=False).first()


def _control_gap_table(summary_df: pd.DataFrame, metric_col: str = "rmse@50_mean") -> pd.DataFrame:
    if summary_df.empty or metric_col not in summary_df.columns:
        return pd.DataFrame()
    pivot = summary_df.pivot_table(index="task", columns="variant", values=metric_col, aggfunc="first")
    required = [
        "ngrc_takens_rg_true",
        "ngrc_takens_rg_lagged_control",
        "ngrc_takens_rg_random_control",
    ]
    if any(col not in pivot.columns for col in required):
        return pd.DataFrame()
    out = pivot.reset_index()
    out["true_minus_lagged"] = out["ngrc_takens_rg_true"] - out["ngrc_takens_rg_lagged_control"]
    out["true_minus_random"] = out["ngrc_takens_rg_true"] - out["ngrc_takens_rg_random_control"]
    return out


def _delay_best_table(summary_df: pd.DataFrame, metric_col: str = "rmse@50_mean") -> pd.DataFrame:
    if summary_df.empty or metric_col not in summary_df.columns:
        return pd.DataFrame()
    parts: list[pd.DataFrame] = []
    for prefix in ("ngrc_raw", "ngrc_rg_readout", "ngrc_takens_rg_true"):
        subset = summary_df[summary_df["variant"].str.startswith(prefix)].copy()
        if subset.empty:
            continue
        best = subset.sort_values(["task", metric_col]).groupby("task", as_index=False).first()
        rename = {
            "variant": f"{prefix}_best_variant",
            metric_col: f"{prefix}_best_{metric_col}",
        }
        parts.append(best[["task", "variant", metric_col]].rename(columns=rename))
    if not parts:
        return pd.DataFrame()
    merged = parts[0]
    for part in parts[1:]:
        merged = merged.merge(part, on="task", how="outer")
    if {
        "ngrc_raw_best_rmse@50_mean",
        "ngrc_takens_rg_true_best_rmse@50_mean",
    }.issubset(merged.columns):
        merged["takens_minus_raw_best"] = (
            merged["ngrc_takens_rg_true_best_rmse@50_mean"] - merged["ngrc_raw_best_rmse@50_mean"]
        )
    return merged


def _ablation_equation_table() -> pd.DataFrame:
    rows = [
        {
            "variant": "rc_raw",
            "family": "rc",
            "state_backbone": r"$r_t=(1-\lambda)r_{t-1}+\lambda\tanh(Wr_{t-1}+W_{in}y_t+b)$",
            "readout_term": r"$\hat y_{t+1}=w_r^\top r_t+w_y y_t+c$",
            "extra_features": "none",
            "mechanistic_role": "Pure reservoir baseline; all latent structure must be absorbed by the reservoir state.",
        },
        {
            "variant": "rc_fastslow_readout",
            "family": "rc",
            "state_backbone": r"$r_t=(1-\lambda)r_{t-1}+\lambda\tanh(Wr_{t-1}+W_{in}y_t+b)$",
            "readout_term": r"$\hat y_{t+1}=w_r^\top r_t+w_y y_t+u^\top sf_t+c$",
            "extra_features": r"$sf_t=[f_t,s_t,m_t]^\top$",
            "mechanistic_role": "Adds explicit fast/slow closure variables to the RC readout.",
        },
        {
            "variant": "rc_rg_readout",
            "family": "rc",
            "state_backbone": r"$r_t=(1-\lambda)r_{t-1}+\lambda\tanh(Wr_{t-1}+W_{in}y_t+b)$",
            "readout_term": r"$\hat y_{t+1}=w_r^\top r_t+w_y y_t+b_\rho^\top \rho_t+c$",
            "extra_features": r"$\rho_t=[\rho_t^{op},\rho_t^{ctrl},\rho_t^{noise},\rho_t^{\beta},\rho_t^{cg},\rho_t^{cb}]^\top$",
            "mechanistic_role": "Treats RG as a direct macro readout bias on top of the reservoir memory.",
        },
        {
            "variant": "rc_sf_rg_gated",
            "family": "rc",
            "state_backbone": r"$r_t=(1-\lambda)r_{t-1}+\lambda\tanh(Wr_{t-1}+W_{in}y_t+b)$",
            "readout_term": r"$\hat y_{t+1}=w_r^\top r_t+w_y y_t+u^\top sf_t+b_\rho^\top \rho_t+c_g^\top g(sf_t,\rho_t)+c$",
            "extra_features": r"$g(sf_t,\rho_t)=[m_t\rho_t^{op},\,m_t\rho_t^{\beta},\,s_t\rho_t^{cb}]^\top$",
            "mechanistic_role": "Sparse SF-RG gate: RG modulates only a few mechanistically chosen fast/slow channels.",
        },
        {
            "variant": "ngrc_raw",
            "family": "ngrc",
            "state_backbone": r"$z_t=[y_t,y_{t-\tau},\ldots,y_{t-(d-1)\tau}]^\top,\ \phi(z_t)=\mathrm{Poly}_2(z_t)$",
            "readout_term": r"$\hat y_{t+1}=W_\phi^\top \phi(z_t)$",
            "extra_features": "none",
            "mechanistic_role": "Pure Takens/NGRC baseline; all predictive structure must be absorbed by the delay library.",
        },
        {
            "variant": "ngrc_fastslow_readout",
            "family": "ngrc",
            "state_backbone": r"$z_t=[y_t,y_{t-\tau},\ldots,y_{t-(d-1)\tau}]^\top,\ \phi(z_t)=\mathrm{Poly}_2(z_t)$",
            "readout_term": r"$\hat y_{t+1}=W_\phi^\top \phi(z_t)+u^\top sf_t$",
            "extra_features": r"$sf_t=[f_t,s_t,m_t]^\top$",
            "mechanistic_role": "Delay backbone plus explicit fast/slow readout features.",
        },
        {
            "variant": "ngrc_rg_readout",
            "family": "ngrc",
            "state_backbone": r"$z_t=[y_t,y_{t-\tau},\ldots,y_{t-(d-1)\tau}]^\top,\ \phi(z_t)=\mathrm{Poly}_2(z_t)$",
            "readout_term": r"$\hat y_{t+1}=W_\phi^\top \phi(z_t)+b^\top \rho_t$",
            "extra_features": r"$\rho_t=[\rho_t^{op},\rho_t^{ctrl},\rho_t^{noise},\rho_t^{\beta},\rho_t^{cg},\rho_t^{cb}]^\top$",
            "mechanistic_role": "Treats RG as an additive macro readout bias on top of delay features.",
        },
        {
            "variant": "ngrc_takens_rg_additive",
            "family": "ngrc",
            "state_backbone": r"$z_t=[y_t,y_{t-\tau},\ldots,y_{t-(d-1)\tau}]^\top,\ \phi(z_t)=\mathrm{Poly}_2(z_t)$",
            "readout_term": r"$\widehat{\Delta y}_t=w^\top \phi(z_t)+b^\top \rho_t,\ \hat y_{t+1}=y_t+\widehat{\Delta y}_t$",
            "extra_features": r"$\rho_t=[\rho_t^{op},\rho_t^{\beta},\rho_t^{cb}]^\top$",
            "mechanistic_role": "Tests whether RG is only a macro correction term, without operator modulation.",
        },
        {
            "variant": "ngrc_takens_rg_true",
            "family": "ngrc",
            "state_backbone": r"$z_t=[y_t,y_{t-\tau},\ldots,y_{t-(d-1)\tau}]^\top,\ \phi(z_t)=\mathrm{Poly}_2(z_t)$",
            "readout_term": r"$\widehat{\Delta y}_t=w^\top \phi(z_t)+b^\top \rho_t+s(z_t)^\top A\rho_t,\ \hat y_{t+1}=y_t+\widehat{\Delta y}_t$",
            "extra_features": r"$s(z_t)=[\bar z_t,\ y_t-y_{t-\tau},\ y_t-2y_{t-\tau}+y_{t-2\tau},\ \mathrm{Var}(z_t)]^\top$",
            "mechanistic_role": "Implements the main theory: RG conditions the local delay-space operator instead of replacing the state.",
        },
        {
            "variant": "ngrc_sf_rg_gated",
            "family": "ngrc",
            "state_backbone": r"$z_t=[y_t,y_{t-\tau},\ldots,y_{t-(d-1)\tau}]^\top,\ \phi(z_t)=\mathrm{Poly}_2(z_t)$",
            "readout_term": r"$\hat y_{t+1}=W_\phi^\top \phi(z_t)+u^\top sf_t+b^\top \rho_t+c^\top g(sf_t,\rho_t)$",
            "extra_features": r"$g(sf_t,\rho_t)=[m_t\rho_t^{op},\,m_t\rho_t^{\beta},\,s_t\rho_t^{cb}]^\top$",
            "mechanistic_role": "Lets RG modulate explicit fast/slow closure channels through a sparse mechanistic gate.",
        },
    ]
    return pd.DataFrame(rows)


def _experimental_system_table() -> pd.DataFrame:
    rows = [
        {
            "system": "vanderpol",
            "state_equation": r"$\dot x_1=x_2,\ \dot x_2=\mu(1-x_1^2)x_2-x_1$",
            "observation_equation": r"$y_t=x_1(t)+\eta_t$",
            "task_variants": "relaxation_clean, relaxation_noisy, relaxation_volclustered",
            "mechanistic_axis": "relaxation oscillation with a slow manifold and fast jump segments",
        },
        {
            "system": "fitzhugh_nagumo",
            "state_equation": r"$\dot v=v-\frac{v^3}{3}-h_{sf}w+I,\ \dot w=\epsilon(h_{fs}v+a-bw)$",
            "observation_equation": r"$y_t=v(t)+\eta_t$",
            "task_variants": "classic_clean, classic_noisy, classic_volclustered",
            "mechanistic_axis": "excitable fast-slow dynamics with explicit slow-to-fast and fast-to-slow coupling",
        },
        {
            "system": "hindmarsh_rose",
            "state_equation": r"$\dot v=y-av^3+bv^2-h_{sf}z+I,\ \dot y=c-dv^2-y,\ \dot z=r(h_{fs}s(v-x_r)-z)$",
            "observation_equation": r"$y^{obs}_t=v(t)+\eta_t$",
            "task_variants": "bursting_clean, bursting_noisy, bursting_volclustered",
            "mechanistic_axis": "bursting with slow adaptation controlling fast spiking episodes",
        },
        {
            "system": "lorenz96_twoscale",
            "state_equation": r"$\dot X_k=-X_{k-1}(X_{k-2}-X_{k+1})-X_k+F-\frac{h_{fs}c}{b}\sum_j Y_{k,j},\ \dot Y_{k,j}=-cbY_{k,j+1}(Y_{k,j+2}-Y_{k,j-1})-cY_{k,j}+\frac{h_{sf}c}{b}X_k$",
            "observation_equation": r"$y_t \in \{X_0,\ \sum_i \alpha_i X_i,\ \sum_i \alpha_i X_i+\sum_j \beta_j Y_{0,j}\}+\eta_t$",
            "task_variants": "obs_slow0, obs_sparse_slowproj, obs_mixed_projection, gate_s2f_*, noise_*",
            "mechanistic_axis": "multiscale closure, observability geometry, and coupling-strength sweeps",
        },
    ]
    return pd.DataFrame(rows)


def _metric_scope_table() -> pd.DataFrame:
    rows = [
        {
            "metric_family": "predictive_local",
            "metric": "one_step_rmse",
            "formula_or_definition": r"$\sqrt{\frac{1}{N}\sum_t (y_{t+1}-\hat y_{t+1|t})^2}$",
            "why_it_matters": "Tests local one-step fit quality.",
        },
        {
            "metric_family": "predictive_rollout",
            "metric": "rmse@10, rmse@50, rmse@100",
            "formula_or_definition": r"$\mathrm{RMSE}@H=\sqrt{\frac{1}{H}\sum_{h=1}^{H}(y_{t+h}-\hat y_{t+h})^2}$",
            "why_it_matters": "Separates short-, mid-, and long-horizon rollout quality.",
        },
        {
            "metric_family": "distributional",
            "metric": "acf_rmse",
            "formula_or_definition": r"$\mathrm{RMSE}(\mathrm{ACF}(y),\mathrm{ACF}(\hat y))$",
            "why_it_matters": "Checks whether temporal dependence is reproduced, not just point accuracy.",
        },
        {
            "metric_family": "distributional",
            "metric": "psd_rmse",
            "formula_or_definition": r"$\mathrm{RMSE}(\mathrm{PSD}(y),\mathrm{PSD}(\hat y))$",
            "why_it_matters": "Checks whether oscillatory content and spectral energy are preserved.",
        },
        {
            "metric_family": "coordinate_dynamics",
            "metric": "markov_gain_ratio",
            "formula_or_definition": "improvement from adding one more lag to the coordinate transition model",
            "why_it_matters": "Smaller is better; near-zero means the coordinate is closer to Markovian closure.",
        },
        {
            "metric_family": "coordinate_dynamics",
            "metric": "koopman_invariance_score",
            "formula_or_definition": "normalized error of the best linear one-step operator on the coordinate",
            "why_it_matters": "Larger is better; indicates better approximate Koopman invariance.",
        },
        {
            "metric_family": "coordinate_dynamics",
            "metric": "spectral_radius_rmse",
            "formula_or_definition": r"$\mathrm{RMSE}$ between local true and coordinate-implied spectral radii",
            "why_it_matters": "Smaller is better; checks whether local growth/decay structure is preserved.",
        },
    ]
    return pd.DataFrame(rows)


def _overall_ablation_frame(
    conditioning_summary: pd.DataFrame,
    boundary_summary: pd.DataFrame,
    robustness_summary: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    conditioning_tasks = set(conditioning_summary["task"].unique()) if not conditioning_summary.empty else set()
    boundary_extra = boundary_summary[~boundary_summary["task"].isin(conditioning_tasks)].copy() if not boundary_summary.empty else pd.DataFrame()
    if not conditioning_summary.empty:
        cond = conditioning_summary.copy()
        cond["ablation_source"] = "conditioning"
        frames.append(cond)
    if not boundary_extra.empty:
        boundary_extra["ablation_source"] = "boundary"
        frames.append(boundary_extra)
    if not robustness_summary.empty:
        robust = robustness_summary.copy()
        robust["ablation_source"] = "robustness"
        frames.append(robust)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, axis=0, ignore_index=True)
    combined = combined.sort_values(["task", "variant"]).reset_index(drop=True)
    return combined


def _overall_ablation_table(summary_df: pd.DataFrame, metric_col: str = "rmse@50_mean") -> pd.DataFrame:
    if summary_df.empty or metric_col not in summary_df.columns:
        return pd.DataFrame()
    keep = summary_df[summary_df["variant"].isin(CORE_VARIANT_ORDER)].copy()
    if keep.empty:
        return pd.DataFrame()
    keep["task_group"] = keep.get("task_family", keep.get("ablation_source", ""))
    pivot = keep.pivot_table(index=["task", "task_group"], columns="variant", values=metric_col, aggfunc="first").reset_index()
    ordered_cols = ["task", "task_group"] + [col for col in CORE_VARIANT_ORDER if col in pivot.columns]
    pivot = pivot[ordered_cols]
    winners = keep.sort_values(["task", metric_col, "variant"]).groupby("task", as_index=False).first()[["task", "variant", metric_col]]
    winners = winners.rename(columns={"variant": "winner", metric_col: "winner_rmse@50_mean"})
    return pivot.merge(winners, on="task", how="left").sort_values(["task_group", "task"]).reset_index(drop=True)


def _robustness_winner_table(summary_df: pd.DataFrame, metric_col: str = "rmse@50_mean") -> pd.DataFrame:
    if summary_df.empty or metric_col not in summary_df.columns:
        return pd.DataFrame()
    return (
        summary_df.sort_values(["task", metric_col, "variant"])
        .groupby("task", as_index=False)
        .first()[["task", "variant", metric_col, "rmse@100_mean", "acf_rmse_mean", "psd_rmse_mean"]]
    )


def _overall_multimetric_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    keep = summary_df[summary_df["variant"].isin(CORE_VARIANT_ORDER)].copy()
    if keep.empty:
        return pd.DataFrame()
    keep["task_group"] = keep.get("task_family", keep.get("ablation_source", ""))
    cols = [
        "task",
        "task_group",
        "variant",
        "one_step_rmse_mean",
        "rmse@10_mean",
        "rmse@50_mean",
        "rmse@100_mean",
        "acf_rmse_mean",
        "psd_rmse_mean",
    ]
    return keep[[col for col in cols if col in keep.columns]].sort_values(["task_group", "task", "variant"]).reset_index(drop=True)


def _winner_by_metric_table(summary_df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    keep = summary_df[summary_df["variant"].isin(CORE_VARIANT_ORDER)].copy()
    if keep.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for metric in metrics:
        if metric not in keep.columns:
            continue
        ordered = keep.sort_values(["task", metric, "variant"]).groupby("task", as_index=False).first()
        for _, row in ordered.iterrows():
            rows.append(
                {
                    "task": row["task"],
                    "task_group": row.get("task_family", row.get("ablation_source", "")),
                    "metric": metric,
                    "winner": row["variant"],
                    "winner_value": row[metric],
                }
            )
    return pd.DataFrame(rows).sort_values(["task_group", "task", "metric"]).reset_index(drop=True) if rows else pd.DataFrame()


def _coordinate_winner_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    metric_specs = [
        ("markov_gain_ratio_mean", True, "best_closure_coordinate"),
        ("koopman_invariance_score_mean", False, "best_koopman_coordinate"),
        ("spectral_radius_rmse_mean", True, "best_spectral_coordinate"),
    ]
    for metric, ascending, label in metric_specs:
        if metric not in summary_df.columns:
            continue
        ordered = summary_df.sort_values(["task", metric], ascending=[True, ascending]).groupby("task", as_index=False).first()
        for _, row in ordered.iterrows():
            rows.append(
                {
                    "task": row["task"],
                    "metric": label,
                    "winner": row["coordinate"],
                    "winner_value": row[metric],
                }
            )
    return pd.DataFrame(rows).sort_values(["task", "metric"]).reset_index(drop=True) if rows else pd.DataFrame()


def _variant_short_label(name: str) -> str:
    mapping = {
        "ngrc_raw": "Raw",
        "ngrc_rg_readout": "RG-readout",
        "ngrc_takens_rg_additive": "Takens+RG-add",
        "ngrc_takens_rg_true": "Takens+RG-op",
        "ngrc_sf_rg_gated": "SF+RG-gate",
        "delay": "delay",
        "delay_rg_joint": "delay+rg",
        "fastslow": "fastslow",
        "rg": "rg",
        "raw": "raw",
    }
    return mapping.get(str(name), str(name))


def _format_entry(label: str, value: object, precision: int = 4) -> str:
    try:
        value_f = float(value)
    except Exception:
        return str(label)
    return f"{_variant_short_label(label)} ({value_f:.{precision}g})"


def _task_context_label(row: pd.Series) -> str:
    parts: list[str] = []
    for key in ("observability_profile", "noise_profile"):
        value = row.get(key, "")
        if pd.notna(value) and str(value).strip() and str(value).strip().lower() != "nan":
            parts.append(str(value).strip())
    if not parts:
        regime = row.get("task_regime", "")
        if pd.notna(regime) and str(regime).strip():
            parts.append(str(regime).strip())
    return " / ".join(parts)


def _downgrade_tier(level: str) -> str:
    order = ["low", "medium", "high"]
    try:
        idx = order.index(level)
    except ValueError:
        return level
    return order[max(0, idx - 1)]


def _upgrade_tier(level: str) -> str:
    order = ["low", "medium", "high"]
    try:
        idx = order.index(level)
    except ValueError:
        return level
    return order[min(len(order) - 1, idx + 1)]


def _paper_takeaway(best_variant: str) -> str:
    mapping = {
        "ngrc_raw": "Delay backbone is sufficient; extra macro conditioning does not help here.",
        "ngrc_rg_readout": "Observation is already aligned with the macro slow state; additive RG readout is sufficient.",
        "ngrc_takens_rg_additive": "A simple macro residual is more robust than multiplicative operator conditioning in this regime.",
        "ngrc_takens_rg_true": "Delay backbone plus RG-conditioned local operator is the strongest predictive mechanism.",
        "ngrc_sf_rg_gated": "Explicit fast-slow closure with sparse RG gating is required to capture the dominant mechanism.",
    }
    return mapping.get(str(best_variant), "No stable paper-style interpretation was assigned.")


def _paper_style_summary_table(
    benchmark_summary: pd.DataFrame,
    coordinate_summary: pd.DataFrame,
    specificity_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if benchmark_summary.empty:
        return pd.DataFrame()
    keep = benchmark_summary[benchmark_summary["variant"].isin(CORE_VARIANT_ORDER)].copy()
    if keep.empty:
        return pd.DataFrame()
    keep["task_group"] = keep.get("task_family", keep.get("ablation_source", ""))
    control_gaps = _control_gap_table(specificity_summary) if specificity_summary is not None and not specificity_summary.empty else pd.DataFrame()
    control_map = {
        str(row["task"]): row
        for _, row in control_gaps.iterrows()
    } if not control_gaps.empty else {}
    rows: list[dict[str, object]] = []
    for task, task_df in keep.groupby("task", sort=True):
        task_df = task_df.sort_values("variant").reset_index(drop=True)
        meta = task_df.iloc[0]
        best_local = task_df.sort_values(["one_step_rmse_mean", "variant"]).iloc[0] if "one_step_rmse_mean" in task_df.columns else None
        best_mid = task_df.sort_values(["rmse@50_mean", "variant"]).iloc[0] if "rmse@50_mean" in task_df.columns else None
        best_long = task_df.sort_values(["rmse@100_mean", "variant"]).iloc[0] if "rmse@100_mean" in task_df.columns else None
        best_acf = task_df.sort_values(["acf_rmse_mean", "variant"]).iloc[0] if "acf_rmse_mean" in task_df.columns else None
        best_psd = task_df.sort_values(["psd_rmse_mean", "variant"]).iloc[0] if "psd_rmse_mean" in task_df.columns else None
        coord_df = coordinate_summary[coordinate_summary["task"] == task].copy() if not coordinate_summary.empty else pd.DataFrame()
        best_closure = coord_df.sort_values(["markov_gain_ratio_mean", "coordinate"]).iloc[0] if "markov_gain_ratio_mean" in coord_df.columns and not coord_df.empty else None
        best_koopman = coord_df.sort_values(["koopman_invariance_score_mean", "coordinate"], ascending=[False, True]).iloc[0] if "koopman_invariance_score_mean" in coord_df.columns and not coord_df.empty else None
        best_spectral = coord_df.sort_values(["spectral_radius_rmse_mean", "coordinate"]).iloc[0] if "spectral_radius_rmse_mean" in coord_df.columns and not coord_df.empty else None

        confidence = "low"
        if best_mid is not None and "rmse@50_mean" in task_df.columns:
            ordered_mid = task_df.sort_values(["rmse@50_mean", "variant"]).reset_index(drop=True)
            if len(ordered_mid) >= 2:
                best_val = float(ordered_mid.loc[0, "rmse@50_mean"])
                runner_val = float(ordered_mid.loc[1, "rmse@50_mean"])
                margin = (runner_val - best_val) / (abs(runner_val) + 1e-12)
            else:
                margin = 0.0
            if margin >= 0.25:
                confidence = "high"
            elif margin >= 0.10:
                confidence = "medium"
            if best_long is not None and str(best_long["variant"]) == str(best_mid["variant"]):
                confidence = _upgrade_tier(confidence)
            control_row = control_map.get(str(task))
            if control_row is not None and str(best_mid["variant"]) == "ngrc_takens_rg_true":
                true_minus_lagged = float(control_row.get("true_minus_lagged", float("nan")))
                true_minus_random = float(control_row.get("true_minus_random", float("nan")))
                if true_minus_lagged > 0.0 or true_minus_random > 0.0:
                    confidence = _downgrade_tier(confidence)

        best_mid_variant = str(best_mid["variant"]) if best_mid is not None else ""
        takeaway = _paper_takeaway(best_mid_variant)
        control_row = control_map.get(str(task))
        if control_row is not None and best_mid_variant == "ngrc_takens_rg_true":
            true_minus_lagged = float(control_row.get("true_minus_lagged", float("nan")))
            true_minus_random = float(control_row.get("true_minus_random", float("nan")))
            if true_minus_lagged > 0.0 or true_minus_random > 0.0:
                takeaway = takeaway.rstrip(".") + "; however, matched controls do not fully support RG-specificity."

        rows.append(
            {
                "task_group": meta.get("task_group", ""),
                "task": task,
                "context": _task_context_label(meta),
                "best_local": _format_entry(str(best_local["variant"]), best_local["one_step_rmse_mean"]) if best_local is not None else "",
                "best_mid_rollout": _format_entry(str(best_mid["variant"]), best_mid["rmse@50_mean"]) if best_mid is not None else "",
                "best_long_rollout": _format_entry(str(best_long["variant"]), best_long["rmse@100_mean"]) if best_long is not None else "",
                "best_distribution": (
                    f"ACF:{_format_entry(str(best_acf['variant']), best_acf['acf_rmse_mean'], precision=3)}; "
                    f"PSD:{_format_entry(str(best_psd['variant']), best_psd['psd_rmse_mean'], precision=3)}"
                    if best_acf is not None and best_psd is not None
                    else ""
                ),
                "best_coordinate": (
                    f"closure:{_format_entry(str(best_closure['coordinate']), best_closure['markov_gain_ratio_mean'], precision=3)}; "
                    f"koopman:{_format_entry(str(best_koopman['coordinate']), best_koopman['koopman_invariance_score_mean'], precision=3)}; "
                    f"spectral:{_format_entry(str(best_spectral['coordinate']), best_spectral['spectral_radius_rmse_mean'], precision=3)}"
                    if best_closure is not None and best_koopman is not None and best_spectral is not None
                    else ""
                ),
                "paper_takeaway": takeaway,
                "confidence": confidence,
            }
        )
    return pd.DataFrame(rows).sort_values(["task_group", "task"]).reset_index(drop=True)


def _render_paper_style_summary_md(summary_df: pd.DataFrame) -> str:
    if summary_df.empty:
        return "# Paper-Style Summary\n\nNo rows were produced.\n"
    cols = [
        "task_group",
        "task",
        "context",
        "best_local",
        "best_mid_rollout",
        "best_long_rollout",
        "best_distribution",
        "best_coordinate",
        "paper_takeaway",
        "confidence",
    ]
    return "\n".join(
        [
            "# Paper-Style Summary",
            "",
            summary_df[cols].to_markdown(index=False),
            "",
        ]
    )


def _write_report(
    out_dir: Path,
    specificity_summary: pd.DataFrame,
    conditioning_summary: pd.DataFrame,
    boundary_summary: pd.DataFrame,
    delay_summary: pd.DataFrame,
    coordinate_summary: pd.DataFrame,
    robustness_summary: pd.DataFrame,
) -> None:
    merged_ablation = _overall_ablation_frame(
        conditioning_summary=conditioning_summary,
        boundary_summary=boundary_summary,
        robustness_summary=robustness_summary,
    )
    equation_table = _ablation_equation_table()
    system_table = _experimental_system_table()
    metric_scope_table = _metric_scope_table()
    overall_ablation = _overall_ablation_table(merged_ablation)
    multimetric_ablation = _overall_multimetric_table(merged_ablation)
    winner_by_metric = _winner_by_metric_table(
        merged_ablation,
        metrics=(
            "one_step_rmse_mean",
            "rmse@10_mean",
            "rmse@50_mean",
            "rmse@100_mean",
            "acf_rmse_mean",
            "psd_rmse_mean",
        ),
    )
    paper_style_summary = _paper_style_summary_table(
        benchmark_summary=merged_ablation,
        coordinate_summary=coordinate_summary,
        specificity_summary=specificity_summary,
    )
    lines: list[str] = [
        "# Takens-RG Validation Report",
        "",
        "This report validates the Takens-RG NGRC story with multi-seed matched controls, conditioning-form ablations, regime-boundary sweeps, delay-sufficiency sweeps, and coordinate diagnostics.",
        "",
        "## Ablation Equations",
        "",
        "The compared RC/NGRC ablations can be read as follows:",
        "",
        _render_markdown_table(
            equation_table,
            cols=["variant", "family", "state_backbone", "readout_term", "extra_features", "mechanistic_role"],
            sort_cols=["family", "variant"],
        ),
        "",
        "## Experimental System Equations",
        "",
        "These are the true benchmark dynamics and observation equations used in the current validation tasks:",
        "",
        _render_markdown_table(
            system_table,
            cols=["system", "state_equation", "observation_equation", "task_variants", "mechanistic_axis"],
            sort_cols=["system"],
        ),
        "",
        "## Evaluation Metrics",
        "",
        "The validation does not only use `rmse@50`; the experiment logs predictive, distributional, and coordinate-dynamical metrics:",
        "",
        _render_markdown_table(
            metric_scope_table,
            cols=["metric_family", "metric", "formula_or_definition", "why_it_matters"],
            sort_cols=["metric_family", "metric"],
        ),
        "",
        "## Overall Ablation Overview",
        "",
        "The table below merges the main conditioning study, the boundary-only extra tasks, and the robustness validation tasks into one task-by-task view using `rmse@50_mean` as the primary sort metric.",
        "",
        _render_markdown_table(
            overall_ablation,
            cols=["task", "task_group", *CORE_VARIANT_ORDER, "winner", "winner_rmse@50_mean"],
            sort_cols=["task_group", "task"],
        ),
        "",
        "Per-task multi-metric view for the same merged ablation set:",
        "",
        _render_markdown_table(
            multimetric_ablation,
            cols=[
                "task",
                "task_group",
                "variant",
                "one_step_rmse_mean",
                "rmse@10_mean",
                "rmse@50_mean",
                "rmse@100_mean",
                "acf_rmse_mean",
                "psd_rmse_mean",
            ],
            sort_cols=["task_group", "task", "variant"],
        ),
        "",
        "Winner by metric on the merged ablation set:",
        "",
        _render_markdown_table(
            winner_by_metric,
            cols=["task", "task_group", "metric", "winner", "winner_value"],
            sort_cols=["task_group", "task", "metric"],
        ),
        "",
        "## Paper-Style Task Summary",
        "",
        "A compact paper-style synthesis that combines predictive winners, coordinate diagnostics, and a conservative mechanistic reading:",
        "",
        _render_markdown_table(
            paper_style_summary,
            cols=[
                "task_group",
                "task",
                "context",
                "best_local",
                "best_mid_rollout",
                "best_long_rollout",
                "best_distribution",
                "best_coordinate",
                "paper_takeaway",
                "confidence",
            ],
            sort_cols=["task_group", "task"],
        ),
        "",
        "## Specificity Controls",
        "",
        "Matched-control benchmark summary (`rmse@50_mean` and `rmse@100_mean`):",
        "",
        _render_markdown_table(
            specificity_summary,
            cols=[
                "task",
                "variant",
                "seed_count",
                "rmse@50_mean",
                "rmse@50_std",
                "rmse@100_mean",
                "rmse@100_std",
                "acf_rmse_mean",
                "psd_rmse_mean",
            ],
            sort_cols=["task", "variant"],
        ),
        "",
        "True-vs-control gaps (`true - control`, lower is better so negative values favor true RG conditioning):",
        "",
        _render_markdown_table(
            _control_gap_table(specificity_summary),
            cols=[
                "task",
                "ngrc_takens_rg_true",
                "ngrc_takens_rg_lagged_control",
                "ngrc_takens_rg_random_control",
                "true_minus_lagged",
                "true_minus_random",
            ],
            sort_cols=["task"],
        ),
        "",
        "## Conditioning-Form Ablation",
        "",
        _render_markdown_table(
            conditioning_summary,
            cols=[
                "task",
                "variant",
                "seed_count",
                "rmse@50_mean",
                "rmse@50_std",
                "rmse@100_mean",
                "acf_rmse_mean",
                "psd_rmse_mean",
            ],
            sort_cols=["task", "variant"],
        ),
        "",
        "Best variant per task for the conditioning-form study:",
        "",
        _render_markdown_table(
            _best_variant_table(conditioning_summary, metric_col="rmse@50_mean"),
            cols=["task", "variant", "rmse@50_mean", "rmse@100_mean", "acf_rmse_mean", "psd_rmse_mean"],
            sort_cols=["task"],
        ),
        "",
        "## Boundary Sweep",
        "",
        _render_markdown_table(
            boundary_summary,
            cols=[
                "task",
                "sweep_group",
                "sweep_value",
                "observability_profile",
                "variant",
                "rmse@50_mean",
                "rmse@50_std",
                "rmse@100_mean",
            ],
            sort_cols=["sweep_group", "sweep_value", "task", "variant"],
        ),
        "",
        "Task-wise winners on the boundary sweep:",
        "",
        _render_markdown_table(
            _best_variant_table(boundary_summary, metric_col="rmse@50_mean"),
            cols=["task", "variant", "rmse@50_mean", "rmse@100_mean"],
            sort_cols=["task"],
        ),
        "",
        "## Delay Sufficiency Sweep",
        "",
        _render_markdown_table(
            delay_summary,
            cols=[
                "task",
                "variant",
                "seed_count",
                "rmse@50_mean",
                "rmse@50_std",
                "rmse@100_mean",
            ],
            sort_cols=["task", "variant"],
        ),
        "",
        "Best delay setting per model family:",
        "",
        _render_markdown_table(
            _delay_best_table(delay_summary),
            cols=[
                "task",
                "ngrc_raw_best_variant",
                "ngrc_raw_best_rmse@50_mean",
                "ngrc_rg_readout_best_variant",
                "ngrc_rg_readout_best_rmse@50_mean",
                "ngrc_takens_rg_true_best_variant",
                "ngrc_takens_rg_true_best_rmse@50_mean",
                "takens_minus_raw_best",
            ],
            sort_cols=["task"],
        ),
        "",
        "## Robustness Validation",
        "",
        "Additional validation on clean/noisy/volatility-clustered classic tasks and multiscale noise-profile tasks:",
        "",
        _render_markdown_table(
            robustness_summary,
            cols=[
                "task",
                "task_family",
                "task_regime",
                "variant",
                "seed_count",
                "rmse@50_mean",
                "rmse@50_std",
                "rmse@100_mean",
                "acf_rmse_mean",
                "psd_rmse_mean",
            ],
            sort_cols=["task_family", "task", "variant"],
        ),
        "",
        "Best variant per robustness task:",
        "",
        _render_markdown_table(
            _robustness_winner_table(robustness_summary),
            cols=["task", "variant", "rmse@50_mean", "rmse@100_mean", "acf_rmse_mean", "psd_rmse_mean"],
            sort_cols=["task"],
        ),
        "",
        "## Coordinate Diagnostics",
        "",
        _render_markdown_table(
            coordinate_summary,
            cols=[
                "task",
                "coordinate",
                "seed_count",
                "markov_gain_ratio_mean",
                "koopman_invariance_score_mean",
                "spectral_radius_rmse_mean",
                "spectral_radius_corr_mean",
            ],
            sort_cols=["task", "coordinate"],
        ),
        "",
        "Best coordinate by each dynamical metric:",
        "",
        _render_markdown_table(
            _coordinate_winner_table(coordinate_summary),
            cols=["task", "metric", "winner", "winner_value"],
            sort_cols=["task", "metric"],
        ),
        "",
        "## Interpretation Notes",
        "",
        "- If `delay` keeps the best Markov/Koopman scores while `ngrc_takens_rg_true` wins some prediction tasks, that supports the conditioner view rather than the state-replacement view.",
        "- If `ngrc_takens_rg_true` consistently beats `lagged` and `random` controls, the gain is RG-specific rather than a generic extra-feature effect.",
        "- If the interaction variant beats the additive residual on mixed or intermediate-coupling tasks, that supports the operator-conditioning hypothesis.",
        "- If the Takens-RG gain shrinks as delay grows, RG is acting partly as a finite-delay regularizer; if it survives at large delay, it is acting more like a regime-conditioned operator.",
        "",
    ]
    (out_dir / "takens_rg_validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_takens_rg_validation(
    out_dir: str,
    seeds: Sequence[int] = DEFAULT_VALIDATION_SEEDS,
    grid_mode: str = "quick",
) -> dict[str, pd.DataFrame]:
    root = Path(out_dir)
    ensure_dir(root)
    common_validation_context = {"search_space_override": _validation_search_space()}

    specificity_variants = [
        BenchmarkVariant("ngrc_raw", "ngrc_raw", model_context=dict(common_validation_context)),
        BenchmarkVariant("ngrc_rg_readout", "ngrc_rg_readout", model_context=dict(common_validation_context)),
        BenchmarkVariant("ngrc_takens_rg_residual", "ngrc_takens_rg_true", model_context=dict(common_validation_context)),
        BenchmarkVariant(
            "ngrc_takens_rg_residual",
            "ngrc_takens_rg_lagged_control",
            model_context=dict(common_validation_context, rg_control_mode="lagged_rg", rg_lag=12),
        ),
        BenchmarkVariant(
            "ngrc_takens_rg_residual",
            "ngrc_takens_rg_random_control",
            model_context=dict(common_validation_context, rg_control_mode="random_summary", random_feature_seed=17),
        ),
    ]
    _, specificity_mech = _run_variant_collection(
        suite="fastslow_mechanism_sweeps",
        task_names=MECHANISM_CORE_TASKS,
        seeds=seeds,
        variants=specificity_variants,
        out_dir=root / "specificity_mechanism",
        grid_mode=grid_mode,
    )
    _, specificity_classic = _run_variant_collection(
        suite="fastslow_theory",
        task_names=CLASSIC_CORE_TASKS,
        seeds=seeds,
        variants=specificity_variants,
        out_dir=root / "specificity_classic",
        grid_mode=grid_mode,
    )
    specificity_summary = pd.concat([specificity_mech, specificity_classic], axis=0, ignore_index=True)
    specificity_summary.to_csv(root / "specificity_summary_all.csv", index=False)

    conditioning_variants = [
        BenchmarkVariant("ngrc_raw", "ngrc_raw", model_context=dict(common_validation_context)),
        BenchmarkVariant("ngrc_rg_readout", "ngrc_rg_readout", model_context=dict(common_validation_context)),
        BenchmarkVariant("ngrc_sf_rg_gated", "ngrc_sf_rg_gated", model_context=dict(common_validation_context)),
        BenchmarkVariant(
            "ngrc_takens_rg_residual",
            "ngrc_takens_rg_additive",
            model_context=dict(common_validation_context, correction_mode="additive"),
        ),
        BenchmarkVariant("ngrc_takens_rg_residual", "ngrc_takens_rg_true", model_context=dict(common_validation_context)),
    ]
    _, conditioning_mech = _run_variant_collection(
        suite="fastslow_mechanism_sweeps",
        task_names=MECHANISM_CORE_TASKS,
        seeds=seeds,
        variants=conditioning_variants,
        out_dir=root / "conditioning_mechanism",
        grid_mode=grid_mode,
    )
    conditioning_summary = conditioning_mech.copy()
    conditioning_summary.to_csv(root / "conditioning_summary_all.csv", index=False)

    boundary_variants = [
        BenchmarkVariant("ngrc_raw", "ngrc_raw", model_context=dict(common_validation_context)),
        BenchmarkVariant("ngrc_rg_readout", "ngrc_rg_readout", model_context=dict(common_validation_context)),
        BenchmarkVariant("ngrc_sf_rg_gated", "ngrc_sf_rg_gated", model_context=dict(common_validation_context)),
        BenchmarkVariant("ngrc_takens_rg_residual", "ngrc_takens_rg_true", model_context=dict(common_validation_context)),
    ]
    _, boundary_summary = _run_variant_collection(
        suite="fastslow_mechanism_sweeps",
        task_names=BOUNDARY_TASKS,
        seeds=seeds,
        variants=boundary_variants,
        out_dir=root / "boundary_sweep",
        grid_mode=grid_mode,
    )

    delay_search_space = _delay_search_space(delays=(4, 8, 12, 16, 24), strides=(1, 2), ridges=(1e-6, 1e-5, 1e-4))
    delay_variants: list[BenchmarkVariant] = []
    for base_name, prefix, extra_context in [
        ("ngrc_raw", "ngrc_raw", {}),
        ("ngrc_rg_readout", "ngrc_rg_readout", {}),
        ("ngrc_takens_rg_residual", "ngrc_takens_rg_true", {}),
    ]:
        for stride in (1, 2):
            for delay in (4, 8, 12, 16, 24):
                label = f"{prefix}_d{delay}_s{stride}"
                contexts = {
                    "search_space_override": [
                        cfg for cfg in delay_search_space if int(cfg["n_delays"]) == delay and int(cfg["stride"]) == stride
                    ],
                }
                contexts.update(extra_context)
                delay_variants.append(BenchmarkVariant(base_name, label, model_context=contexts))
    _, delay_summary = _run_variant_collection(
        suite="fastslow_mechanism_sweeps",
        task_names=DELAY_SWEEP_TASKS,
        seeds=seeds,
        variants=delay_variants,
        out_dir=root / "delay_sufficiency",
        grid_mode=grid_mode,
    )

    _, coordinate_summary = run_coordinate_analysis_seed_sweep(
        suite="fastslow_mechanism_sweeps",
        out_dir=str(root / "coordinate_mechanism"),
        seeds=seeds,
        task_names=MECHANISM_CORE_TASKS,
        coordinate_kinds=("raw", "delay", "delay_rg_joint", "rg", "fastslow"),
        delay_dim=12,
        sample_count=32,
        local_k=64,
        ridge=1e-4,
    )
    _, coordinate_classic = run_coordinate_analysis_seed_sweep(
        suite="fastslow_theory",
        out_dir=str(root / "coordinate_classic"),
        seeds=seeds,
        task_names=CLASSIC_CORE_TASKS,
        coordinate_kinds=("raw", "delay", "delay_rg_joint", "rg", "fastslow"),
        delay_dim=12,
        sample_count=32,
        local_k=64,
        ridge=1e-4,
    )
    coordinate_summary = pd.concat([coordinate_summary, coordinate_classic], axis=0, ignore_index=True)
    coordinate_summary.to_csv(root / "coordinate_summary_all.csv", index=False)

    robustness_variants = [
        BenchmarkVariant("ngrc_raw", "ngrc_raw", model_context=dict(common_validation_context)),
        BenchmarkVariant("ngrc_rg_readout", "ngrc_rg_readout", model_context=dict(common_validation_context)),
        BenchmarkVariant(
            "ngrc_takens_rg_residual",
            "ngrc_takens_rg_additive",
            model_context=dict(common_validation_context, correction_mode="additive"),
        ),
        BenchmarkVariant("ngrc_takens_rg_residual", "ngrc_takens_rg_true", model_context=dict(common_validation_context)),
        BenchmarkVariant("ngrc_sf_rg_gated", "ngrc_sf_rg_gated", model_context=dict(common_validation_context)),
    ]
    _, robustness_classic = _run_variant_collection(
        suite="fastslow_theory",
        task_names=ROBUSTNESS_CLASSIC_TASKS,
        seeds=seeds,
        variants=robustness_variants,
        out_dir=root / "robustness_classic",
        grid_mode=grid_mode,
    )
    _, robustness_finance = _run_variant_collection(
        suite="fastslow_finance_theory",
        task_names=ROBUSTNESS_FINANCE_TASKS,
        seeds=seeds,
        variants=robustness_variants,
        out_dir=root / "robustness_finance",
        grid_mode=grid_mode,
    )
    _, robustness_mechanism_noise = _run_variant_collection(
        suite="fastslow_mechanism_sweeps",
        task_names=ROBUSTNESS_MECHANISM_NOISE_TASKS,
        seeds=seeds,
        variants=robustness_variants,
        out_dir=root / "robustness_mechanism_noise",
        grid_mode=grid_mode,
    )
    robustness_summary = pd.concat(
        [robustness_classic, robustness_finance, robustness_mechanism_noise],
        axis=0,
        ignore_index=True,
    )
    robustness_summary.to_csv(root / "robustness_summary_all.csv", index=False)

    merged_ablation = _overall_ablation_frame(
        conditioning_summary=conditioning_summary,
        boundary_summary=boundary_summary,
        robustness_summary=robustness_summary,
    )
    _ablation_equation_table().to_csv(root / "model_equations.csv", index=False)
    _experimental_system_table().to_csv(root / "experimental_system_equations.csv", index=False)
    _metric_scope_table().to_csv(root / "evaluation_metrics.csv", index=False)
    _overall_ablation_table(merged_ablation).to_csv(root / "overall_ablation_rmse50_mean.csv", index=False)
    _overall_multimetric_table(merged_ablation).to_csv(root / "overall_ablation_multimetric.csv", index=False)
    _winner_by_metric_table(
        merged_ablation,
        metrics=(
            "one_step_rmse_mean",
            "rmse@10_mean",
            "rmse@50_mean",
            "rmse@100_mean",
            "acf_rmse_mean",
            "psd_rmse_mean",
        ),
    ).to_csv(root / "overall_metric_winners.csv", index=False)
    _coordinate_winner_table(coordinate_summary).to_csv(root / "coordinate_metric_winners.csv", index=False)
    paper_style_summary = _paper_style_summary_table(
        benchmark_summary=merged_ablation,
        coordinate_summary=coordinate_summary,
        specificity_summary=specificity_summary,
    )
    paper_style_summary.to_csv(root / "paper_style_summary.csv", index=False)
    (root / "paper_style_summary.md").write_text(_render_paper_style_summary_md(paper_style_summary), encoding="utf-8")

    _write_report(
        out_dir=root,
        specificity_summary=specificity_summary,
        conditioning_summary=conditioning_summary,
        boundary_summary=boundary_summary,
        delay_summary=delay_summary,
        coordinate_summary=coordinate_summary,
        robustness_summary=robustness_summary,
    )
    return {
        "specificity_summary": specificity_summary,
        "conditioning_summary": conditioning_summary,
        "boundary_summary": boundary_summary,
        "delay_summary": delay_summary,
        "coordinate_summary": coordinate_summary,
        "robustness_summary": robustness_summary,
    }
