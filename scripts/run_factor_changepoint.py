import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fsrc_sindy.research import PiecewiseChangeConfig, run_factor_changepoint_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run factor-based changepoint detection with offline RC/NGRC identification and an online FTRL detector."
    )
    parser.add_argument("--out_dir", type=str, default="runs/factor_changepoint/vanderpol_rg_switch")
    parser.add_argument("--suite", type=str, default="fastslow_smoke")
    parser.add_argument("--task", type=str, default="vanderpol_relaxation_smoke")
    parser.add_argument("--pre_mu", type=float, default=8.0)
    parser.add_argument("--post_mu", type=float, default=14.0)
    parser.add_argument("--fit_train_len", type=int, default=700)
    parser.add_argument("--fit_val_len", type=int, default=250)
    parser.add_argument("--change_index", type=int, default=1250)
    parser.add_argument("--total_length", type=int, default=2200)
    parser.add_argument("--process_noise_std", type=float, default=0.005)
    parser.add_argument("--obs_noise_std", type=float, default=0.01)
    parser.add_argument("--grid_mode", type=str, default="quick", choices=["quick", "full"])
    parser.add_argument("--model_names", nargs="+", default=["rc_rg_readout", "ngrc_rg_readout"])
    parser.add_argument("--train_seeds", nargs="+", type=int, default=None)
    parser.add_argument("--test_seeds", nargs="+", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_cfg = PiecewiseChangeConfig(
        suite=args.suite,
        task_name=args.task,
        pre_params={"mu": float(args.pre_mu)},
        post_params={"mu": float(args.post_mu)},
        fit_train_len=int(args.fit_train_len),
        fit_val_len=int(args.fit_val_len),
        change_index=int(args.change_index),
        total_length=int(args.total_length),
        process_noise_std=float(args.process_noise_std),
        obs_noise_std=float(args.obs_noise_std),
    )
    outputs = run_factor_changepoint_experiment(
        out_dir=args.out_dir,
        experiment_cfg=exp_cfg,
        model_names=args.model_names,
        train_seeds=args.train_seeds,
        test_seeds=args.test_seeds,
        grid_mode=args.grid_mode,
    )
    print("\n=== factor changepoint experiment finished ===")
    for name, df in outputs.items():
        print(f"- {name}: {len(df)} rows")
    print(f"\nResults saved under: {args.out_dir}")


if __name__ == "__main__":
    main()
