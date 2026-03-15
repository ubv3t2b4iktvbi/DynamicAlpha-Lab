import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fsrc_sindy.research.koopman_readout_proof import KoopmanReadoutProofConfig, run_koopman_readout_proof


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a controlled experiment showing how direct Koopman readout features change RC forecasting."
    )
    parser.add_argument("--out_dir", type=str, default="runs/koopman_readout_proof/default")
    parser.add_argument("--train_episodes", type=int, default=256)
    parser.add_argument("--test_episodes", type=int, default=128)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--washout", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37, 51, 67])
    parser.add_argument("--reservoir_size", type=int, default=48)
    parser.add_argument("--spectral_radius", type=float, default=0.8)
    parser.add_argument("--input_scale", type=float, default=0.35)
    parser.add_argument("--leak_rate", type=float, default=0.4)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--sparsity", type=float, default=0.08)
    parser.add_argument("--rollout_horizons", nargs="+", type=int, default=[10, 40])
    parser.add_argument("--latent_eigs", nargs="+", type=float, default=[0.992, -0.989, 0.985])
    parser.add_argument("--koopman_powers", nargs="+", type=int, default=[3, 5, 3])
    parser.add_argument("--koopman_weights", nargs="+", type=float, default=[0.45, -0.3, 0.2])
    parser.add_argument("--y_self_coupling", type=float, default=0.75)
    parser.add_argument("--process_noise_std", type=float, default=0.01)
    parser.add_argument("--init_hidden_scale", type=float, default=1.0)
    parser.add_argument("--init_obs_scale", type=float, default=0.25)
    parser.add_argument("--test_seed_offset", type=int, default=1000)
    parser.add_argument("--stability_abs_threshold", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = KoopmanReadoutProofConfig(
        train_episodes=args.train_episodes,
        test_episodes=args.test_episodes,
        steps=args.steps,
        washout=args.washout,
        warmup=args.warmup,
        rollout_horizons=tuple(args.rollout_horizons),
        seeds=tuple(args.seeds),
        reservoir_size=args.reservoir_size,
        spectral_radius=args.spectral_radius,
        input_scale=args.input_scale,
        leak_rate=args.leak_rate,
        ridge=args.ridge,
        sparsity=args.sparsity,
        latent_eigs=tuple(args.latent_eigs),
        koopman_powers=tuple(args.koopman_powers),
        koopman_weights=tuple(args.koopman_weights),
        y_self_coupling=args.y_self_coupling,
        process_noise_std=args.process_noise_std,
        init_hidden_scale=args.init_hidden_scale,
        init_obs_scale=args.init_obs_scale,
        test_seed_offset=args.test_seed_offset,
        stability_abs_threshold=args.stability_abs_threshold,
    )
    manifest = run_koopman_readout_proof(out_dir=args.out_dir, config=cfg)
    print("\n=== koopman readout proof finished ===")
    for name, path in manifest.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
