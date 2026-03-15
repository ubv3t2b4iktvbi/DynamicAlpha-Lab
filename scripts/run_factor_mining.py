import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fsrc_sindy.factors import DynamicsFeatureConfig, FactorMiningConfig
from fsrc_sindy.models.rc import RCConfig
from fsrc_sindy.pipeline import run_factor_mining_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RC-based automatic dynamical factor mining.")
    parser.add_argument("--suite", type=str, default="smoke", choices=["smoke", "common", "hard", "fastslow_smoke", "fastslow_theory", "fastslow_sparse_theory", "fastslow_finance_theory", "highdim", "highdim_theory", "gaepr_smoke", "all", "research"])
    parser.add_argument("--out_dir", type=str, default="runs/factor_mining/smoke")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override. Defaults to factor_mining.random_seed from config, else 123.")
    parser.add_argument("--mode", type=str, default=None, choices=["accumulate", "identify"], help="Optional mining mode override.")
    parser.add_argument("--full_library_search", action="store_true", help="Force full-library screening even in identify mode.")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--identifier_kinds", nargs="+", default=None)
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config for RC / factor mining / feature settings.")
    parser.add_argument("--wsga_prior", action="store_true", help="Use WSGA attractor prior as an optional EPR-style screening signal.")
    parser.add_argument("--wsga_noise_strength", type=float, default=0.01)
    parser.add_argument("--wsga_steps", type=int, default=2000)
    parser.add_argument("--wsga_rand_num", type=int, default=128)
    parser.add_argument("--epr_weight_strength", type=float, default=None)
    return parser.parse_args()


def load_config(path: str | None):
    if path is None:
        return {}, {}, {}
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    unknown_sections = sorted(set(payload) - {"factor_mining", "rc", "features"})
    if unknown_sections:
        raise ValueError(f"Unknown config section(s): {unknown_sections}. Expected only factor_mining, rc, and features.")
    mining = dict(payload.get("factor_mining", {}))
    rc = dict(payload.get("rc", {}))
    features = dict(payload.get("features", {}))
    if "identifier_kinds" in mining:
        mining["identifier_kinds"] = tuple(mining["identifier_kinds"])
    if "score_horizons" in mining:
        mining["score_horizons"] = tuple(mining["score_horizons"])
    if "slow_windows" in features:
        features["slow_windows"] = tuple(features["slow_windows"])
    return mining, rc, features


def main() -> None:
    args = parse_args()
    mining_kwargs, rc_kwargs, feature_kwargs = load_config(args.config)
    if args.identifier_kinds is not None:
        mining_kwargs["identifier_kinds"] = tuple(args.identifier_kinds)
    if args.mode is not None:
        mining_kwargs["mode"] = args.mode
    if args.full_library_search:
        mining_kwargs["full_library_search"] = True
    if args.wsga_prior:
        mining_kwargs["use_wsga_prior"] = True
        mining_kwargs["wsga_noise_strength"] = args.wsga_noise_strength
        mining_kwargs["wsga_steps"] = args.wsga_steps
        mining_kwargs["wsga_rand_num"] = args.wsga_rand_num
    if args.epr_weight_strength is not None:
        mining_kwargs["epr_weight_strength"] = args.epr_weight_strength
    seed = args.seed if args.seed is not None else int(mining_kwargs.get("random_seed", 123))
    mining_cfg = FactorMiningConfig(**mining_kwargs)
    rc_cfg = RCConfig(**rc_kwargs)
    feature_cfg = DynamicsFeatureConfig(**feature_kwargs)
    df = run_factor_mining_suite(
        suite=args.suite,
        out_dir=args.out_dir,
        seed=seed,
        task_names=args.tasks,
        identifier_kinds=args.identifier_kinds,
        mining_cfg=mining_cfg,
        rc_cfg=rc_cfg,
        feature_cfg=feature_cfg,
    )
    cols = [c for c in ["task", "identifier_kind", "mode", "num_selected", "validation_score", "rollout_validation_score", "final_rmse10", "final_rmse50", "test_rmse50", "selected_factors", "selected_wsga_epr_score"] if c in df.columns]
    print("\n=== factor mining finished ===")
    if not df.empty:
        print(df[cols].to_string(index=False))
    print(f"\nResults saved under: {args.out_dir}")
    print(f"Seed used: {seed}")


if __name__ == "__main__":
    main()
