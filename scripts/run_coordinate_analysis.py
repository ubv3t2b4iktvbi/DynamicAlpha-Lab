import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fsrc_sindy.research import run_coordinate_analysis_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run coordinate-closure and theory-aware diagnostics on benchmark tasks.")
    parser.add_argument("--suite", type=str, default="smoke", choices=["smoke", "common", "hard", "fastslow_smoke", "fastslow_theory", "fastslow_sparse_theory", "highdim", "highdim_theory", "all", "research"])
    parser.add_argument("--out_dir", type=str, default="runs/coordinate_analysis/smoke")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--coordinates", nargs="+", default=["raw", "delay", "fastslow", "factor"])
    parser.add_argument("--delay_dim", type=int, default=8)
    parser.add_argument("--sample_count", type=int, default=24)
    parser.add_argument("--local_k", type=int, default=64)
    parser.add_argument("--ridge", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = run_coordinate_analysis_suite(
        suite=args.suite,
        out_dir=args.out_dir,
        seed=args.seed,
        task_names=args.tasks,
        coordinate_kinds=args.coordinates,
        delay_dim=args.delay_dim,
        sample_count=args.sample_count,
        local_k=args.local_k,
        ridge=args.ridge,
    )
    print("\n=== coordinate analysis finished ===")
    if not df.empty:
        cols = [
            c
            for c in [
                "task",
                "coordinate",
                "coord_dim",
                "markov_gain_ratio",
                "spectral_radius_rmse",
                "spectral_radius_corr",
                "offdiag_mi_mean",
            ]
            if c in df.columns
        ]
        print(df[cols].to_string(index=False))
    print(f"\nResults saved under: {args.out_dir}")


if __name__ == "__main__":
    main()
