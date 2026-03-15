import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fsrc_sindy.attractor_prior import WSGAConfig
from fsrc_sindy.research import run_coordinate_analysis_seed_sweep, run_coordinate_analysis_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run coordinate-closure and theory-aware diagnostics on benchmark tasks.")
    parser.add_argument("--suite", type=str, default="smoke", choices=["smoke", "common", "hard", "fastslow_smoke", "fastslow_theory", "fastslow_sparse_theory", "fastslow_finance_theory", "fastslow_gating_sweep", "fastslow_observability_sweep", "fastslow_hetero_sweep", "fastslow_mechanism_sweeps", "highdim", "highdim_theory", "gaepr_smoke", "all", "research"])
    parser.add_argument("--out_dir", type=str, default="runs/coordinate_analysis/smoke")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--coordinates", nargs="+", default=["raw", "delay", "fastslow", "factor"])
    parser.add_argument("--delay_dim", type=int, default=8)
    parser.add_argument("--sample_count", type=int, default=24)
    parser.add_argument("--local_k", type=int, default=64)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--wsga_prior", action="store_true", help="Enable WSGA-based attractor-prior diagnostics.")
    parser.add_argument("--wsga_noise_strength", type=float, default=0.01)
    parser.add_argument("--wsga_dt", type=float, default=0.01)
    parser.add_argument("--wsga_steps", type=int, default=2000)
    parser.add_argument("--wsga_rand_num", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = list(args.seeds) if args.seeds else [args.seed]
    wsga_config = None
    if args.wsga_prior:
        wsga_config = WSGAConfig(
            noise_strength=args.wsga_noise_strength,
            dt=args.wsga_dt,
            steps=args.wsga_steps,
            rand_num=args.wsga_rand_num,
        )
    if len(seeds) == 1:
        df = run_coordinate_analysis_suite(
            suite=args.suite,
            out_dir=args.out_dir,
            seed=seeds[0],
            task_names=args.tasks,
            coordinate_kinds=args.coordinates,
            delay_dim=args.delay_dim,
            sample_count=args.sample_count,
            local_k=args.local_k,
            ridge=args.ridge,
            wsga_config=wsga_config,
        )
    else:
        df, summary = run_coordinate_analysis_seed_sweep(
            suite=args.suite,
            out_dir=args.out_dir,
            seeds=seeds,
            task_names=args.tasks,
            coordinate_kinds=args.coordinates,
            delay_dim=args.delay_dim,
            sample_count=args.sample_count,
            local_k=args.local_k,
            ridge=args.ridge,
            wsga_config=wsga_config,
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
                "wsga_epr_score",
                "wsga_basin_sep_gap",
                "wsga_centroid_dist_corr",
            ]
            if c in df.columns
        ]
        print(df[cols].to_string(index=False))
    if len(seeds) > 1 and not summary.empty:
        summary_cols = [
            c
            for c in [
                "task",
                "coordinate",
                "seed_count",
                "markov_gain_ratio_mean",
                "koopman_invariance_score_mean",
                "spectral_radius_rmse_mean",
            ]
            if c in summary.columns
        ]
        print("\n=== coordinate seed summary ===")
        print(summary[summary_cols].to_string(index=False))
    print(f"\nResults saved under: {args.out_dir}")


if __name__ == "__main__":
    main()
