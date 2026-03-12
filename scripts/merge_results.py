import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge benchmark result directories into one combined table.")
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="Run directories that contain benchmark_results.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for merged benchmark_results.csv and benchmark_summary.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for item in args.inputs:
        run_dir = Path(item)
        csv_path = run_dir / "benchmark_results.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing benchmark_results.csv under: {run_dir}")
        df = pd.read_csv(csv_path)
        df["source_run"] = run_dir.name
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged_csv = out_dir / "benchmark_results.csv"
    merged.to_csv(merged_csv, index=False)

    metric_cols = [
        c
        for c in [
            "one_step_rmse",
            "rmse@10",
            "rmse@50",
            "rmse@100",
            "acf_rmse",
            "psd_rmse",
            "effective_dim",
            "trained_params",
            "total_params",
            "train_time_sec",
            "speed_us_per_step",
        ]
        if c in merged.columns
    ]
    summary = merged.groupby("variant")[metric_cols].mean(numeric_only=True).sort_values("rmse@50")
    summary_csv = out_dir / "benchmark_summary.csv"
    summary.to_csv(summary_csv)

    print(f"Merged {len(frames)} runs into: {merged_csv}")
    print(f"Saved summary to: {summary_csv}")
    print(f"Rows: {len(merged)}")
    print(f"Tasks: {merged['task'].nunique()}")
    print(f"Variants: {merged['variant'].nunique()}")


if __name__ == "__main__":
    main()
