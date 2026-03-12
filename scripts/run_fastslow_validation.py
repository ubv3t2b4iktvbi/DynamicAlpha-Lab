import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fsrc_sindy.research import run_fastslow_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a theory-focused fast/slow validation loop and write a compact evidence report.")
    parser.add_argument("--suite", type=str, default="fastslow_theory", choices=["fastslow_smoke", "fastslow_theory", "fastslow_sparse_theory", "smoke", "common", "hard", "highdim", "highdim_theory", "all", "research"])
    parser.add_argument("--out_dir", type=str, default="runs/fastslow_validation/fastslow_theory")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--model_groups", nargs="+", default=["fastslow_ablation"])
    parser.add_argument("--grid_mode", type=str, default="quick", choices=["quick", "full"])
    parser.add_argument("--coordinates", nargs="+", default=["raw", "delay", "fastslow", "theory_fastslow", "factor"])
    parser.add_argument("--delay_dim", type=int, default=8)
    parser.add_argument("--mining_mode", type=str, default="identify", choices=["accumulate", "identify"])
    parser.add_argument("--full_library_search", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--factor_config", type=str, default="configs/fastslow_theory_factor_mining.yaml")
    parser.add_argument("--identifier_kinds", nargs="+", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_fastslow_validation(
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
    )
    print("\n=== fast-slow validation finished ===")
    for name, path in result["manifest"].items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
