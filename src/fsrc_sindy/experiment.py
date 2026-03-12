from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .benchmarks import build_suite
from .metrics import evaluate_distribution, evaluate_horizons
from .models import ReservoirTemplateFactory
from .selection import DEFAULT_MODEL_NAMES, get_model_spec, instantiate_model, select_best_model
from .systems import BenchmarkTask, simulate_task, split_series
from .tracking import ProgressTracker
from .utils import ensure_dir, set_seed, to_jsonable


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
