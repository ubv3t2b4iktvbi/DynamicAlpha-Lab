import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fsrc_sindy.experiment import run_benchmark_seed_sweep, run_benchmark_suite
from fsrc_sindy.selection import DEFAULT_MODEL_NAMES, LEGACY_MODEL_NAMES, MODEL_GROUPS, expand_model_group_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fast-slow + RC + SINDy benchmark suites.")
    parser.add_argument("--suite", type=str, default="smoke", choices=["smoke", "common", "hard", "fastslow_smoke", "fastslow_theory", "fastslow_sparse_theory", "fastslow_finance_theory", "fastslow_gating_sweep", "fastslow_observability_sweep", "fastslow_hetero_sweep", "fastslow_mechanism_sweeps", "highdim", "highdim_theory", "gaepr_smoke", "all", "research"])
    parser.add_argument("--out_dir", type=str, default="./runs/smoke")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--grid_mode", type=str, default="quick", choices=["quick", "full"])
    parser.add_argument("--models", type=str, nargs="+", default=None)
    parser.add_argument("--model_groups", type=str, nargs="+", default=[])
    parser.add_argument("--tasks", type=str, nargs="+", default=None)
    parser.add_argument("--include_legacy_hybrids", action="store_true")
    parser.add_argument("--list_model_groups", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_model_groups:
        print("Available model groups:")
        for name, members in MODEL_GROUPS.items():
            print(f"- {name}: {' '.join(members)}")
        return
    explicit_models = list(args.models) if args.models is not None else []
    grouped_models = expand_model_group_names(args.model_groups)
    models = grouped_models + explicit_models
    if not models:
        models = list(DEFAULT_MODEL_NAMES)
    models = list(dict.fromkeys(models))
    if args.include_legacy_hybrids:
        models.extend([m for m in LEGACY_MODEL_NAMES if m not in models])
    seeds = list(args.seeds) if args.seeds else [args.seed]
    if len(seeds) == 1:
        df = run_benchmark_suite(
            suite=args.suite,
            seed=seeds[0],
            out_dir=args.out_dir,
            model_names=models,
            grid_mode=args.grid_mode,
            task_names=args.tasks,
        )
    else:
        df, summary = run_benchmark_seed_sweep(
            suite=args.suite,
            seeds=seeds,
            out_dir=args.out_dir,
            model_names=models,
            grid_mode=args.grid_mode,
            task_names=args.tasks,
        )
    cols = [c for c in [
        "task", "variant", "one_step_rmse", "rmse@10", "rmse@50", "rmse@100", "acf_rmse", "psd_rmse"
    ] if c in df.columns]
    print("\n=== finished ===")
    print(df[cols].to_string(index=False))
    if len(seeds) > 1 and not summary.empty:
        summary_cols = [
            c
            for c in [
                "task",
                "variant",
                "seed_count",
                "rmse@50_mean",
                "rmse@50_std",
                "rmse@100_mean",
                "rmse@100_std",
            ]
            if c in summary.columns
        ]
        print("\n=== seed summary ===")
        print(summary[summary_cols].to_string(index=False))
    print(f"\nResults saved under: {args.out_dir}")


if __name__ == "__main__":
    main()
