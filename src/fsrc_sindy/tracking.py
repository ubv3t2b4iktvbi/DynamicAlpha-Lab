from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .utils import ensure_dir


class ProgressTracker:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        ensure_dir(out_dir)
        self.rows: list[dict[str, Any]] = []
        self.results_fp = out_dir / 'benchmark_results.csv'
        self.summary_fp = out_dir / 'benchmark_summary.csv'
        self.pivot_fp = out_dir / 'benchmark_pivot_rmse50.csv'

    def add_row(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        df = pd.DataFrame(self.rows)
        df.to_csv(self.results_fp, index=False)
        summary_cols = [c for c in [
            'task', 'system', 'task_family', 'task_regime', 'state_dim', 'variant', 'base_model_name', 'model_family',
            'sweep_group', 'sweep_axis', 'sweep_value', 'sweep_label', 'observability_profile', 'noise_profile',
            'readout_identifier_kind', 'readout_factor_count', 'readout_factor_names',
            'train_time_sec', 'rollout_eval_time_sec',
            'speed_us_per_step', 'effective_dim', 'trained_params', 'total_params',
            'one_step_rmse', 'rmse@10', 'rmse@50', 'rmse@100', 'acf_rmse', 'psd_rmse'
        ] if c in df.columns]
        df[summary_cols].to_csv(self.summary_fp, index=False)
        pivot_metric = 'rmse@50' if 'rmse@50' in df.columns else summary_cols[-1]
        pivot_df = df.pivot_table(index='task', columns='variant', values=pivot_metric, aggfunc='first')
        pivot_df.to_csv(self.pivot_fp)
        print(
            f"[saved] {row['task']} | {row['variant']} | "
            f"dim={row.get('effective_dim', float('nan'))} | "
            f"one_step_rmse={row.get('one_step_rmse', float('nan')):.4f} | "
            f"rmse@10={row.get('rmse@10', float('nan')):.4f} | "
            f"rmse@50={row.get('rmse@50', float('nan')):.4f} | "
            f"acf_rmse={row.get('acf_rmse', float('nan')):.4f} | "
            f"psd_rmse={row.get('psd_rmse', float('nan')):.4f}"
        )
