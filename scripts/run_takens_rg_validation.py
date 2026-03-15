import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fsrc_sindy.research import DEFAULT_VALIDATION_SEEDS, run_takens_rg_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the multi-seed Takens-RG validation suite.")
    parser.add_argument("--out_dir", type=str, default="runs/benchmarks/ngrc_takens_rg_validation")
    parser.add_argument("--grid_mode", type=str, default="quick", choices=["quick", "full"])
    parser.add_argument("--seed", type=int, default=int(DEFAULT_VALIDATION_SEEDS[0]))
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(args.seeds) if args.seeds else (int(args.seed),)
    outputs = run_takens_rg_validation(
        out_dir=args.out_dir,
        seeds=seeds,
        grid_mode=args.grid_mode,
    )
    print("\n=== takens-rg validation finished ===")
    for name, df in outputs.items():
        print(f"- {name}: {len(df)} rows")
    print(f"\nResults saved under: {args.out_dir}")


if __name__ == "__main__":
    main()
