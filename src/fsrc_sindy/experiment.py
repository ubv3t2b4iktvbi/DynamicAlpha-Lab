from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .benchmarks import build_suite
from .factors.repository import (
    fastslow_readout_specs,
    rg_readout_specs,
    sparse_rg_gate_specs,
    sf_rg_gated_readout_specs,
    sf_rg_interaction_readout_specs,
)
from .metrics import evaluate_distribution, evaluate_horizons
from .models import ReservoirTemplateFactory
from .selection import DEFAULT_MODEL_NAMES, get_model_spec, instantiate_model, select_best_model
from .systems import BenchmarkTask, simulate_task, split_series, task_metadata_columns
from .tracking import ProgressTracker
from .utils import ensure_dir, set_seed, to_jsonable


PRESET_READOUT_FACTORIES = {
    "rc_fastslow_readout": fastslow_readout_specs,
    "ngrc_fastslow_readout": fastslow_readout_specs,
    "hybrid_rc_ngrc_fastslow": fastslow_readout_specs,
    "rc_rg_readout": rg_readout_specs,
    "ngrc_rg_readout": rg_readout_specs,
    "hybrid_rc_ngrc_rg": rg_readout_specs,
    "ngrc_takens_rg_residual": sparse_rg_gate_specs,
    "rc_sf_rg_gated": sf_rg_gated_readout_specs,
    "ngrc_sf_rg_gated": sf_rg_gated_readout_specs,
    "rc_sf_rg_interaction": sf_rg_interaction_readout_specs,
    "ngrc_sf_rg_interaction": sf_rg_interaction_readout_specs,
}


def run_task(
    task: BenchmarkTask,
    seed: int,
    out_dir: Path,
    model_names: Sequence[str],
    model_contexts: dict[str, dict[str, object]] | None,
    grid_mode: str,
    tracker: ProgressTracker,
    template_factory: ReservoirTemplateFactory,
) -> pd.DataFrame:
    set_seed(seed)
    ensure_dir(out_dir)
    sim = simulate_task(task, seed=seed)
    split = split_series(sim.obs, n_train=task.n_train, n_val=task.n_val, n_test=task.n_test)
    y_train, y_val, y_test = split['train'], split['val'], split['test']
    np.savez(out_dir / f'{task.name}_series.npz', y=sim.obs, train=y_train, val=y_val, test=y_test)

    short_train = task.n_train < 2000
    context_len = max(200, 4 * max(task.selection_horizons))
    rows = []

    for model_name in model_names:
        model_context = dict(model_contexts.get(model_name, {})) if model_contexts is not None and model_name in model_contexts else None
        variant_label = str(model_context.get("variant_label", model_name)) if model_context is not None else model_name
        readout_factor_names: list[str] = []
        if model_context is not None:
            for spec in model_context.get("readout_factor_specs", []):
                if isinstance(spec, dict):
                    readout_factor_names.append(str(spec.get("name", "")))
                else:
                    readout_factor_names.append(str(getattr(spec, "name", spec)))
        elif model_name in PRESET_READOUT_FACTORIES:
            readout_factor_names = [spec.name for spec in PRESET_READOUT_FACTORIES[model_name]()]
        model_spec = get_model_spec(model_name)
        t0 = time.perf_counter()
        model, val_metrics, best_cfg = select_best_model(
            model_name=model_name,
            y_train=y_train,
            y_val=y_val,
            context_len=context_len,
            score_horizons=task.selection_horizons,
            grid_mode=grid_mode,
            template_factory=template_factory,
            short_train=short_train,
            progress_desc=f'{task.name}-{model_name}',
            data_dt=task.dt,
            model_context=model_context,
        )
        train_time = time.perf_counter() - t0

        test_context = np.concatenate([y_val[-context_len:], y_test[:1]])
        test_future = y_test[1:1 + max(max(task.eval_horizons), task.stat_horizon)]
        t1 = time.perf_counter()
        roll_metrics = evaluate_horizons(model, test_context, test_future, task.eval_horizons)
        dist_metrics = evaluate_distribution(model, test_context, test_future, stat_horizon=task.stat_horizon)
        roll_time = time.perf_counter() - t1
        one_metrics = model.one_step_metrics(np.concatenate([y_val[-context_len:], y_test]), burn_in=context_len)

        row = {
            'task': task.name,
            'system': task.system,
            'task_family': task.family or task.system,
            'task_regime': task.regime,
            'task_tags': '|'.join(task.tags),
            'state_dim': int(sim.states.shape[1]),
            'observed_dim': 1,
            'variant': variant_label,
            'base_model_name': model_name,
            'model_family': model_spec.family,
            'uses_fastslow': int(model_spec.uses_fastslow),
            'uses_sindy_backbone': int(model_spec.uses_sindy_backbone),
            'uses_reservoir': int(model_spec.uses_reservoir),
            'uses_ngrc': int(model_spec.uses_ngrc),
            'residual_mode': model_spec.residual_mode,
            'dt': task.dt,
            'process_noise_std': task.process_noise_std,
            'obs_noise_std': task.obs_noise_std,
            'process_noise_volatility': task.process_noise_volatility,
            'obs_noise_volatility': task.obs_noise_volatility,
            'noise_ema_span': task.noise_ema_span,
            'match_obs_noise_energy': int(task.match_obs_noise_energy),
            'obs_mode': task.obs_mode,
            'obs_params': to_jsonable(task.obs_params),
            'selection_horizons': list(task.selection_horizons),
            'eval_horizons': list(task.eval_horizons),
            'stat_horizon': task.stat_horizon,
            'train_time_sec': train_time,
            'rollout_eval_time_sec': roll_time,
            'speed_us_per_step': 1e6 * roll_time / max(task.eval_horizons),
            'effective_dim': model.effective_dim(),
            'total_params': model.count_total_params(),
            'trained_params': model.count_trained_params(),
            'best_config': to_jsonable(best_cfg),
            'readout_identifier_kind': str(model_context.get("readout_identifier_kind", "")) if model_context is not None else "",
            'readout_factor_count': len(readout_factor_names),
            'readout_factor_names': "; ".join(name for name in readout_factor_names if name),
        }
        row.update(task_metadata_columns(task))
        row.update(val_metrics)
        row.update(one_metrics)
        row.update(roll_metrics)
        row.update(dist_metrics)
        tracker.add_row(row)
        rows.append(row)

    return pd.DataFrame(rows)


def run_benchmark_suite(
    suite: str,
    seed: int,
    out_dir: str,
    model_names: Sequence[str] | None = None,
    task_model_names: dict[str, Sequence[str]] | None = None,
    task_model_contexts: dict[str, dict[str, dict[str, object]]] | None = None,
    grid_mode: str = 'quick',
    task_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    out_path = Path(out_dir)
    ensure_dir(out_path)
    tasks = build_suite(suite)
    if task_names:
        wanted = set(task_names)
        tasks = [task for task in tasks if task.name in wanted]
        missing = sorted(wanted - {task.name for task in tasks})
        if missing:
            raise ValueError(f'Unknown task names for suite={suite}: {missing}')
    tracker = ProgressTracker(out_path)
    template_factory = ReservoirTemplateFactory(seed=seed)
    model_names = list(model_names) if model_names is not None else list(DEFAULT_MODEL_NAMES)
    all_rows = []
    for task in tqdm(tasks, desc=f'suite={suite}'):
        model_names_for_task = list(task_model_names.get(task.name, model_names)) if task_model_names is not None else list(model_names)
        if not model_names_for_task:
            continue
        model_contexts_for_task = task_model_contexts.get(task.name, {}) if task_model_contexts is not None else None
        df_task = run_task(
            task=task,
            seed=seed,
            out_dir=out_path,
            model_names=model_names_for_task,
            model_contexts=model_contexts_for_task,
            grid_mode=grid_mode,
            tracker=tracker,
            template_factory=template_factory,
        )
        all_rows.append(df_task)
    result_df = pd.concat(all_rows, axis=0, ignore_index=True) if all_rows else pd.DataFrame()
    return result_df


def _flatten_multiindex_columns(columns: pd.Index) -> list[str]:
    out: list[str] = []
    for col in columns:
        if isinstance(col, tuple):
            head, tail = col
            out.append(str(head) if not tail else f"{head}_{tail}")
        else:
            out.append(str(col))
    return out


def _render_benchmark_seed_summary(summary_df: pd.DataFrame) -> str:
    if summary_df.empty:
        return "# Benchmark Seed Summary\n\nNo rows were produced.\n"
    preferred = [
        "task",
        "variant",
        "seed_count",
        "rmse@50_mean",
        "rmse@50_std",
        "rmse@100_mean",
        "rmse@100_std",
        "acf_rmse_mean",
        "acf_rmse_std",
        "psd_rmse_mean",
        "psd_rmse_std",
    ]
    cols = [c for c in preferred if c in summary_df.columns]
    return "\n".join(
        [
            "# Benchmark Seed Summary",
            "",
            summary_df[cols].sort_values(["task", "variant"]).to_markdown(index=False),
            "",
        ]
    )


def run_benchmark_seed_sweep(
    suite: str,
    seeds: Sequence[int],
    out_dir: str,
    model_names: Sequence[str] | None = None,
    task_model_names: dict[str, Sequence[str]] | None = None,
    task_model_contexts: dict[str, dict[str, dict[str, object]]] | None = None,
    grid_mode: str = "quick",
    task_names: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_path = Path(out_dir)
    ensure_dir(out_path)
    all_runs: list[pd.DataFrame] = []
    for seed in seeds:
        seed_dir = out_path / f"seed_{seed}"
        df_seed = run_benchmark_suite(
            suite=suite,
            seed=int(seed),
            out_dir=str(seed_dir),
            model_names=model_names,
            task_model_names=task_model_names,
            task_model_contexts=task_model_contexts,
            grid_mode=grid_mode,
            task_names=task_names,
        ).copy()
        if not df_seed.empty:
            df_seed["seed"] = int(seed)
            all_runs.append(df_seed)

    combined = pd.concat(all_runs, axis=0, ignore_index=True) if all_runs else pd.DataFrame()
    combined_fp = out_path / "benchmark_seed_results.csv"
    summary_fp = out_path / "benchmark_seed_summary.csv"
    summary_md_fp = out_path / "benchmark_seed_summary.md"
    pivot_fp = out_path / "benchmark_seed_pivot_rmse50_mean.csv"
    combined.to_csv(combined_fp, index=False)

    if combined.empty:
        summary = pd.DataFrame()
    else:
        group_cols = [
            c
            for c in [
                "task",
                "system",
                "task_family",
                "task_regime",
                "variant",
                "base_model_name",
                "model_family",
                "sweep_group",
                "sweep_axis",
                "sweep_value",
                "sweep_label",
                "observability_profile",
                "noise_profile",
                "readout_identifier_kind",
                "readout_factor_count",
                "readout_factor_names",
            ]
            if c in combined.columns
        ]
        metric_cols = [
            c
            for c in [
                "one_step_rmse",
                "rmse@10",
                "rmse@50",
                "rmse@100",
                "acf_rmse",
                "psd_rmse",
                "train_time_sec",
                "rollout_eval_time_sec",
            ]
            if c in combined.columns
        ]
        summary = (
            combined.groupby(group_cols, dropna=False)[metric_cols]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
        )
        summary.columns = _flatten_multiindex_columns(summary.columns)
        seed_counts = (
            combined.groupby(group_cols, dropna=False)["seed"]
            .nunique()
            .reset_index(name="seed_count")
        )
        summary = seed_counts.merge(summary, on=group_cols, how="left")
        summary = summary.sort_values(["task", "variant"]).reset_index(drop=True)
        if "rmse@50_mean" in summary.columns:
            pivot = summary.pivot_table(index="task", columns="variant", values="rmse@50_mean", aggfunc="first")
            pivot.to_csv(pivot_fp)
    summary.to_csv(summary_fp, index=False)
    summary_md_fp.write_text(_render_benchmark_seed_summary(summary), encoding="utf-8")
    return combined, summary
