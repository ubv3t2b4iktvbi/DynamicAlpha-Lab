import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fsrc_sindy.research import reanalyze_fastslow_mechanism_study, run_fastslow_mechanism_study


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a cross-system fast/slow mechanism study with multi-seed summaries and effective h_sf estimates."
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="fastslow_crosssystem_gating_smoke",
        choices=[
            "fastslow_crosssystem_gating_smoke",
            "fastslow_gating_sweep",
            "fastslow_mechanism_sweeps",
        ],
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="runs/fastslow_mechanism_study/fastslow_crosssystem_gating_smoke",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303])
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--model_groups", nargs="+", default=["fastslow_ablation"])
    parser.add_argument("--grid_mode", type=str, default="quick", choices=["quick", "full"])
    parser.add_argument(
        "--coordinates",
        nargs="+",
        default=["raw", "delay", "fastslow", "theory_fastslow", "factor"],
    )
    parser.add_argument("--delay_dim", type=int, default=8)
    parser.add_argument("--mining_mode", type=str, default="accumulate", choices=["accumulate", "identify"])
    parser.add_argument("--full_library_search", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--factor_config", type=str, default="configs/fastslow_theory_factor_mining.yaml")
    parser.add_argument("--identifier_kinds", nargs="+", default=None)
    parser.add_argument("--skip_factor_mining", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--reuse_existing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse existing seed-level outputs under out_dir and only recompute mechanism-study summaries/gates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reuse_existing:
        result = reanalyze_fastslow_mechanism_study(
            suite=args.suite,
            out_dir=args.out_dir,
            seeds=args.seeds,
            task_names=args.tasks,
        )
    else:
        result = run_fastslow_mechanism_study(
            suite=args.suite,
            out_dir=args.out_dir,
            seeds=args.seeds,
            task_names=args.tasks,
            model_groups=args.model_groups,
            grid_mode=args.grid_mode,
            coordinate_kinds=args.coordinates,
            delay_dim=args.delay_dim,
            mining_mode=args.mining_mode,
            full_library_search=args.full_library_search,
            factor_config_path=args.factor_config,
            identifier_kinds=args.identifier_kinds,
            skip_factor_mining=args.skip_factor_mining,
        )
    print("\n=== fast-slow mechanism study finished ===")
    for name, path in result["manifest"].items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
