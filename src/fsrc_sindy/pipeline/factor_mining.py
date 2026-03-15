from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
from tqdm.auto import tqdm

from ..attractor_prior import WSGAConfig, assign_attractor_labels, build_task_attractor_prior
from ..benchmarks import build_suite
from ..models.rc import RCConfig, ReservoirTemplateFactory
from ..systems import split_series, simulate_task, task_metadata_columns
from ..utils import ensure_dir, set_seed
from ..factors.archive import save_run_artifacts
from ..factors.base import DynamicsFeatureConfig, FactorMiningConfig
from ..factors.finance_to_dynamics import translation_markdown
from ..factors.miner import DynamicsFactorMiner


def build_suite_tasks(suite: str, task_names: Sequence[str] | None = None):
    tasks = build_suite(suite)
    if task_names:
        wanted = set(task_names)
        tasks = [task for task in tasks if task.name in wanted]
        missing = sorted(wanted - {task.name for task in tasks})
        if missing:
            raise ValueError(f"Unknown task names for suite={suite}: {missing}")
    return tasks


def run_factor_mining_suite(
    suite: str,
    out_dir: str,
    seed: int = 123,
    task_names: Sequence[str] | None = None,
    identifier_kinds: Sequence[str] | None = None,
    mining_cfg: FactorMiningConfig | None = None,
    rc_cfg: RCConfig | None = None,
    feature_cfg: DynamicsFeatureConfig | None = None,
) -> pd.DataFrame:
    set_seed(seed)
    out_path = Path(out_dir)
    ensure_dir(out_path)
    tasks = build_suite_tasks(suite, task_names=task_names)
    translation_text = translation_markdown()
    mining_cfg = mining_cfg if mining_cfg is not None else FactorMiningConfig(
        identifier_kinds=tuple(identifier_kinds) if identifier_kinds is not None else ("sindy_slow", "spline_kan_like")
    )
    rc_cfg = rc_cfg if rc_cfg is not None else RCConfig(
        n_reservoir=200,
        spectral_radius=0.95,
        input_scale=0.5,
        leak_rate=0.7,
        ridge=1e-5,
        sparsity=0.05,
        washout=100,
    )
    feature_cfg = feature_cfg if feature_cfg is not None else DynamicsFeatureConfig()
    template_factory = ReservoirTemplateFactory(seed=seed)
    miner = DynamicsFactorMiner(
        mining_cfg=mining_cfg,
        rc_cfg=rc_cfg,
        feature_cfg=feature_cfg,
        template_factory=template_factory,
    )
    rows = []
    for task in tqdm(tasks, desc=f"factor_mining[{suite}]"):
        sim = simulate_task(task, seed=seed)
        split = split_series(sim.obs, n_train=task.n_train, n_val=task.n_val, n_test=task.n_test)
        y_train, y_val, y_test = split["train"], split["val"], split["test"]
        attractor_prior = None
        attractor_labels = None
        if mining_cfg.use_wsga_prior:
            try:
                attractor_prior = build_task_attractor_prior(
                    task,
                    seed=seed,
                    config=WSGAConfig(
                        noise_strength=mining_cfg.wsga_noise_strength,
                        dt=mining_cfg.wsga_dt,
                        steps=mining_cfg.wsga_steps,
                        rand_num=mining_cfg.wsga_rand_num,
                    ),
                    reference_states=sim.states,
                )
                attractor_labels = assign_attractor_labels(sim.states[: task.n_train + task.n_val], attractor_prior)
            except ValueError:
                attractor_prior = None
                attractor_labels = None
        task_dir = out_path / task.name
        ensure_dir(task_dir)
        for identifier_kind in mining_cfg.identifier_kinds:
            result = miner.run_for_identifier(
                task_name=task.name,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
                identifier_kind=identifier_kind,
                attractor_prior=attractor_prior,
                attractor_labels=attractor_labels,
                dt=task.dt,
            )
            run_dir = task_dir / identifier_kind
            save_run_artifacts(run_dir, result, translation_table_markdown=translation_text)
            row = result.summary_row()
            row.update(
                {
                    "suite": suite,
                    "system": task.system,
                    "task_family": task.family,
                    "task_regime": task.regime,
                    "state_dim": int(sim.states.shape[1]),
                    "dt": task.dt,
                }
            )
            row.update(task_metadata_columns(task))
            rows.append(row)
        translation_doc = task_dir / "finance_to_dynamics_translation.md"
        if not translation_doc.exists():
            translation_doc.write_text(translation_text, encoding="utf-8")
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["task", "validation_score"]).reset_index(drop=True)
    summary_csv = out_path / "factor_mining_summary.csv"
    df.to_csv(summary_csv, index=False)
    summary_md = out_path / "factor_mining_summary.md"
    lines = ["# Dynamics factor mining summary", ""]
    if not df.empty:
        lines.append(df.to_markdown(index=False))
        lines.append("")
        best = df.sort_values("validation_score").groupby("task").first().reset_index()
        lines.append("## Best identifier per task")
        lines.append("")
        best_cols = [c for c in ["task", "identifier_kind", "selected_factors", "validation_score", "rollout_validation_score", "final_rmse50", "test_rmse50", "selected_wsga_epr_score"] if c in best.columns]
        lines.append(best[best_cols].to_markdown(index=False))
    else:
        lines.append("No runs were generated.")
    summary_md.write_text("\n".join(lines), encoding="utf-8")
    return df
