import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fsrc_sindy.attractor_prior import WSGAConfig
from fsrc_sindy.research.loop import run_research_loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the closed-loop research workflow across ablations, coordinates, and factor mining.")
    parser.add_argument("--suite", type=str, default="smoke", choices=["smoke", "common", "hard", "fastslow_smoke", "fastslow_theory", "fastslow_sparse_theory", "fastslow_finance_theory", "fastslow_gating_sweep", "fastslow_observability_sweep", "fastslow_hetero_sweep", "fastslow_mechanism_sweeps", "fastslow_crosssystem_gating_smoke", "highdim", "highdim_theory", "gaepr_smoke", "all", "research"])
    parser.add_argument("--out_dir", type=str, default="runs/research_loop/smoke")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--model_groups", nargs="+", default=["fastslow_ablation"])
    parser.add_argument("--grid_mode", type=str, default="quick", choices=["quick", "full"])
    parser.add_argument("--coordinates", nargs="+", default=["raw", "delay", "fastslow", "factor"])
    parser.add_argument("--delay_dim", type=int, default=8)
    parser.add_argument("--mining_mode", type=str, default="identify", choices=["accumulate", "identify"])
    parser.add_argument("--full_library_search", action="store_true")
    parser.add_argument("--factor_config", type=str, default="configs/factor_mining.yaml")
    parser.add_argument("--identifier_kinds", nargs="+", default=None)
    parser.add_argument("--wsga_prior", action="store_true")
    parser.add_argument("--wsga_noise_strength", type=float, default=0.01)
    parser.add_argument("--wsga_steps", type=int, default=2000)
    parser.add_argument("--wsga_rand_num", type=int, default=128)
    parser.add_argument("--skip_benchmarks", action="store_true")
    parser.add_argument("--skip_coordinate_analysis", action="store_true")
    parser.add_argument("--skip_factor_mining", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wsga_config = None
    if args.wsga_prior:
        wsga_config = WSGAConfig(
            noise_strength=args.wsga_noise_strength,
            steps=args.wsga_steps,
            rand_num=args.wsga_rand_num,
        )
    result = run_research_loop(
        suite=args.suite,
        out_dir=args.out_dir,
        seed=args.seed,
        task_names=args.tasks,
        model_groups=args.model_groups,
        grid_mode=args.grid_mode,
        coordinate_kinds=args.coordinates,
        delay_dim=args.delay_dim,
        mining_mode=args.mining_mode,
        full_library_search=args.full_library_search,
        factor_config_path=args.factor_config,
        identifier_kinds=args.identifier_kinds,
        coordinate_wsga_config=wsga_config,
        skip_benchmarks=args.skip_benchmarks,
        skip_coordinate_analysis=args.skip_coordinate_analysis,
        skip_factor_mining=args.skip_factor_mining,
    )
    print("\n=== research loop finished ===")
    for name, path in result["manifest"].items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
