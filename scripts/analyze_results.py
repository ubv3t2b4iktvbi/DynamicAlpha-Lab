import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fsrc_sindy.selection import ABLATION_COMPARISONS, MODEL_SPECS


def metric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in ["one_step_rmse", "rmse@10", "rmse@50", "rmse@100", "acf_rmse", "psd_rmse"] if c in df.columns]


def mean_table(df: pd.DataFrame, by: list[str], cols: list[str], sort_col: str | None = None) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame()
    out = df.groupby(by)[cols].mean(numeric_only=True)
    if sort_col is not None and sort_col in out.columns:
        out = out.sort_values(sort_col)
    return out


def ablation_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for comp in ABLATION_COMPARISONS:
        base = comp["baseline"]
        cand = comp["candidate"]
        common_tasks = sorted(set(df.loc[df["variant"] == base, "task"]) & set(df.loc[df["variant"] == cand, "task"]))
        if not common_tasks:
            continue
        per_task = []
        for task in common_tasks:
            base_row = df[(df["task"] == task) & (df["variant"] == base)].iloc[0]
            cand_row = df[(df["task"] == task) & (df["variant"] == cand)].iloc[0]
            gain50 = float(base_row["rmse@50"] - cand_row["rmse@50"]) if {"rmse@50"}.issubset(df.columns) else float("nan")
            gain100 = float(base_row["rmse@100"] - cand_row["rmse@100"]) if {"rmse@100"}.issubset(df.columns) else float("nan")
            mismatch_base = float(base_row["rmse@50"] / (base_row["one_step_rmse"] + 1e-12)) if {"rmse@50", "one_step_rmse"}.issubset(df.columns) else float("nan")
            mismatch_cand = float(cand_row["rmse@50"] / (cand_row["one_step_rmse"] + 1e-12)) if {"rmse@50", "one_step_rmse"}.issubset(df.columns) else float("nan")
            per_task.append(
                {
                    "gain50": gain50,
                    "gain100": gain100,
                    "mismatch_delta": mismatch_base - mismatch_cand,
                }
            )
        sub = pd.DataFrame(per_task)
        rows.append(
            {
                "comparison": comp["name"],
                "baseline": base,
                "candidate": cand,
                "tasks": len(common_tasks),
                "avg_rmse50_gain": sub["gain50"].mean(),
                "avg_rmse100_gain": sub["gain100"].mean(),
                "avg_mismatch_reduction": sub["mismatch_delta"].mean(),
                "hypothesis": comp["hypothesis"],
            }
        )
    return pd.DataFrame(rows).sort_values("avg_rmse50_gain", ascending=False) if rows else pd.DataFrame()


def dimension_table(df: pd.DataFrame) -> pd.DataFrame:
    if not {"state_dim", "variant", "rmse@50"}.issubset(df.columns):
        return pd.DataFrame()
    keep = ["rmse@50"]
    if "rmse@100" in df.columns:
        keep.append("rmse@100")
    if "one_step_rmse" in df.columns:
        keep.append("one_step_rmse")
    out = mean_table(df, ["state_dim", "variant"], keep)
    return out.reset_index().sort_values(["state_dim", "rmse@50"])


def top_by_dimension(df: pd.DataFrame) -> pd.DataFrame:
    if not {"state_dim", "variant", "rmse@50"}.issubset(df.columns):
        return pd.DataFrame()
    rows = []
    for state_dim, sub in df.groupby("state_dim"):
        best = sub.sort_values("rmse@50").iloc[0]
        rows.append(
            {
                "state_dim": state_dim,
                "best_variant": best["variant"],
                "task_count": sub["task"].nunique(),
                "avg_rmse50": sub[sub["variant"] == best["variant"]]["rmse@50"].mean(),
                "avg_rmse100": sub[sub["variant"] == best["variant"]]["rmse@100"].mean() if "rmse@100" in sub.columns else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("state_dim")


def derive_insights(df: pd.DataFrame, abl: pd.DataFrame) -> list[str]:
    insights: list[str] = []

    def add(text: str) -> None:
        if text not in insights:
            insights.append(text)

    if not abl.empty:
        row = abl[abl["comparison"] == "fastslow_on_rc"]
        if not row.empty and float(row.iloc[0]["avg_rmse50_gain"]) > 0:
            add("Fast/slow readout consistently helps RC at matched memory size, which suggests the bottleneck is latent state reconstruction rather than raw capacity.")
        row = abl[abl["comparison"] == "fastslow_on_ngrc"]
        if not row.empty and float(row.iloc[0]["avg_rmse50_gain"]) <= 0:
            add("Fast/slow features do not reliably fix NGRC long-rollout instability, so the issue is not only missing coarse variables but also accumulated autoregressive error.")
        row = abl[abl["comparison"] == "dual_memory_vs_single_memory"]
        if not row.empty and float(row.iloc[0]["avg_mismatch_reduction"]) > 0:
            add("The RC+NGRC hybrid improves the one-step to long-horizon mismatch relative to single-memory RC, indicating that recurrent state and explicit delay coordinates contribute complementary memory.")
        row = abl[abl["comparison"] == "slow_backbone_vs_none"]
        if not row.empty and float(row.iloc[0]["avg_rmse50_gain"]) > 0:
            add("Injecting a slow SINDy backbone helps when the task family is genuinely multiscale or high-dimensional, which matches the idea that an explicit resolved slow manifold stabilizes forecasting.")
        row = abl[abl["comparison"] == "hybrid_residual_vs_ngrc_residual"]
        if not row.empty and float(row.iloc[0]["avg_mismatch_reduction"]) > 0:
            add("The new slow-SINDy delta hybrid reduces residual-model mismatch relative to pure NGRC closure, suggesting that residual dynamics also benefit from recurrent state regularization.")

    if {"variant", "task_family", "rmse@50"}.issubset(df.columns):
        highdim = df[df["task_family"].str.contains("highdim", na=False)]
        if not highdim.empty:
            winners = highdim.sort_values("rmse@50").groupby("task").first()["variant"].value_counts()
            if not winners.empty:
                leader = winners.index[0]
                add(f"On the current high-dimensional tasks, `{leader}` wins most often, so claims should be framed around partial-observation structure rather than universal dominance.")

    if {"variant", "one_step_rmse", "rmse@50"}.issubset(df.columns):
        mismatch = df.assign(mismatch_ratio=df["rmse@50"] / (df["one_step_rmse"] + 1e-12))
        ngrc_rows = mismatch[mismatch["variant"].str.contains("ngrc", na=False)]
        rc_rows = mismatch[mismatch["variant"].str.contains("rc", na=False)]
        if not ngrc_rows.empty and not rc_rows.empty:
            if ngrc_rows["mismatch_ratio"].median() > 5.0 * rc_rows["mismatch_ratio"].median():
                add("NGRC variants show much stronger teacher-forcing to rollout mismatch than RC-style models, so stability control should be a primary design axis, not an afterthought.")

    return insights


def reviewer_notes(df: pd.DataFrame, insights: list[str]) -> list[str]:
    notes: list[str] = []
    if {"task", "variant", "rmse@50"}.issubset(df.columns):
        wins = df.sort_values("rmse@50").groupby("task").first()["variant"].value_counts()
        if not wins.empty:
            notes.append(f"Strength: the benchmark exposes that no single baseline dominates all regimes; `{wins.index[0]}` is only the most frequent winner, not a universal winner.")
    if any("Fast/slow readout" in item for item in insights):
        notes.append("Strength: the paper can make a credible claim that coarse fast/slow features are a real source of gain under scalar observation because the ablation is budget-matched.")
    if any("teacher-forcing" in item for item in insights):
        notes.append("Weakness: several models are selected on short validation horizons but fail badly in free rollout, so a reviewer will ask for stronger stability-aware model selection and reporting.")
    if "state_dim" in df.columns and int(df["state_dim"].max()) >= 32:
        notes.append("Strength: the updated suite now includes higher-dimensional Lorenz-96 scaling tasks, which makes the evaluation less vulnerable to the criticism that the benchmark is only low-dimensional.")
    notes.append("Weakness: all current comparisons stay within RC/NGRC/SINDy-style families; a reviewer would still want at least one modern sequence-model baseline or a clear justification for excluding it.")
    notes.append("Weakness: results appear to rely on a single random seed, so variance bars and multi-seed significance should be added before making strong superiority claims.")
    return notes


def next_direction_lines(df: pd.DataFrame) -> list[str]:
    lines = []
    lines.append("The most natural next direction is `slow_sindy_delta_hybrid`: a slow-manifold backbone plus a dual-memory residual closure.")
    lines.append("Theory link: under slow-fast decomposition and Mori-Zwanzig style reasoning, the resolved slow dynamics should be modeled explicitly, while unresolved memory effects are better captured by a closure term with both recurrent state and delay coordinates.")
    lines.append("Targeted experiment: compare `hybrid_rc_ngrc_fastslow`, `slow_sindy_delta_ngrc`, `slow_sindy_delta_rc`, and `slow_sindy_delta_hybrid` on the high-dimensional multiscale tasks only, then measure `rmse@50`, `rmse@100`, `acf_rmse`, and the one-step/rollout mismatch ratio.")
    if {"variant", "task_family", "rmse@50"}.issubset(df.columns):
        highdim = df[df["task_family"].str.contains("highdim", na=False)]
        if not highdim.empty and "slow_sindy_delta_hybrid" in set(highdim["variant"]):
            hybrid = highdim[highdim["variant"] == "slow_sindy_delta_hybrid"]["rmse@50"].mean()
            lines.append(f"Current signal: `slow_sindy_delta_hybrid` is already implemented in this revision, so the new hypothesis is directly testable; its current average high-dimensional `rmse@50` is {hybrid:.4g}.")
    lines.append("Modular validation loop: keep the suite fixed, swap only one component family at a time, and use the ablation table to verify whether each theoretical ingredient improves the intended failure mode.")
    return lines


def build_report(df: pd.DataFrame) -> str:
    lines = []
    lines.append("# Result analysis")
    lines.append("")
    if df.empty:
        lines.append("No rows found.")
        return "\n".join(lines)

    metrics = metric_columns(df)

    lines.append("## Best model by task")
    lines.append("")
    for task, sub in df.groupby("task"):
        lines.append(f"### {task}")
        for metric in metrics:
            best = sub.sort_values(metric).iloc[0]
            lines.append(f"- best {metric}: `{best['variant']}` = {best[metric]:.6g}")
        if {"state_dim", "task_family", "task_regime"}.issubset(sub.columns):
            meta = sub.iloc[0]
            lines.append(f"- metadata: state_dim={int(meta['state_dim'])}, family=`{meta['task_family']}`, regime=`{meta['task_regime']}`")
        lines.append("")

    lines.append("## Cross-task averages")
    lines.append("")
    avg_cols = metrics + [c for c in ["effective_dim", "trained_params", "total_params", "train_time_sec", "speed_us_per_step"] if c in df.columns]
    agg = mean_table(df, ["variant"], avg_cols, sort_col="rmse@50" if "rmse@50" in avg_cols else metrics[0])
    lines.append(agg.to_markdown())
    lines.append("")

    if {"task_family", "variant", "rmse@50"}.issubset(df.columns):
        lines.append("## Family-wise averages")
        lines.append("")
        fam_cols = [c for c in ["rmse@10", "rmse@50", "rmse@100", "acf_rmse", "psd_rmse"] if c in df.columns]
        fam = mean_table(df, ["task_family", "variant"], fam_cols, sort_col="rmse@50")
        lines.append(fam.to_markdown())
        lines.append("")

    dim = dimension_table(df)
    if not dim.empty:
        lines.append("## Dimension scaling")
        lines.append("")
        lines.append(dim.to_markdown(index=False))
        lines.append("")
        top = top_by_dimension(df)
        if not top.empty:
            lines.append("### Best variant per state dimension")
            lines.append(top.to_markdown(index=False))
            lines.append("")

    abl = ablation_table(df)
    if not abl.empty:
        lines.append("## Ablation summary")
        lines.append("")
        lines.append(abl.to_markdown(index=False))
        lines.append("")

    if {"variant", "one_step_rmse", "rmse@50"}.issubset(df.columns):
        lines.append("## One-step vs long-horizon mismatch")
        lines.append("")
        mismatch = df.assign(mismatch_ratio=df["rmse@50"] / (df["one_step_rmse"] + 1e-12)).sort_values("mismatch_ratio", ascending=False)
        cols = ["task", "variant", "one_step_rmse", "rmse@50", "mismatch_ratio"]
        if "state_dim" in mismatch.columns:
            cols.append("state_dim")
        lines.append(mismatch[cols].head(20).to_markdown(index=False))
        lines.append("")

    insights = derive_insights(df, abl)
    if insights:
        lines.append("## Insights")
        lines.append("")
        for item in insights:
            lines.append(f"- {item}")
        lines.append("")

    lines.append("## Theory-motivated next direction")
    lines.append("")
    for item in next_direction_lines(df):
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## Reviewer view")
    lines.append("")
    for item in reviewer_notes(df, insights):
        lines.append(f"- {item}")
    lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze benchmark result csv.")
    parser.add_argument("--results_csv", type=str, required=True)
    parser.add_argument("--out_md", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.results_csv)
    report = build_report(df)
    if args.out_md:
        out_md = Path(args.out_md)
    else:
        out_md = Path(args.results_csv).with_name("analysis_report.md")
    out_md.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved report to: {out_md}")


if __name__ == "__main__":
    main()
