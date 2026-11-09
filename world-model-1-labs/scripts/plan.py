"""Run the random-shooting planner in the real environment.

Compares the planner's real return against a random-action baseline. Planning
uses only the learned world model; the simulator is stepped just to execute the
chosen first action each step.

Usage::

    python scripts/plan.py --config configs/lab.yaml --episodes 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from _cli import build_parser, load_trained_model

from lab1.config import Config
from lab1.planning import evaluate_planner_vs_random
from lab1.training import resolve_device


def resolve_checkpoint(cfg: Config, checkpoint: str | None = None) -> Path:
    """Prefer an explicit checkpoint; otherwise use the loaded training config."""
    if checkpoint is not None:
        return Path(checkpoint)
    return Path(cfg.training.checkpoint_dir) / "world_model_best.pt"


def main() -> None:
    parser = build_parser("Run the random-shooting planner.")
    parser.add_argument(
        "--checkpoint",
        help="Override the default <training.checkpoint_dir>/world_model_best.pt.",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=30)
    args: argparse.Namespace = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1:
        parser.error("--episodes and --max-steps must be positive")

    cfg = Config.load(args.config, args.overrides)
    checkpoint = resolve_checkpoint(cfg, args.checkpoint)
    model = load_trained_model(cfg, str(checkpoint))
    device = resolve_device(cfg.training.device)
    print(f"checkpoint: {checkpoint} | device: {device}")
    comparison = evaluate_planner_vs_random(
        model,
        cfg,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed0=cfg.seed,
        device=device,
    )
    print("seed | planner return | random return | planner crash | random crash")
    for ep, (plan, rand) in enumerate(
        zip(comparison["planner"], comparison["random"], strict=True)
    ):
        print(
            f"{cfg.seed + ep:4d} | {plan['total_reward']:14.3f} | "
            f"{rand['total_reward']:13.3f} | {str(plan['crashed']):13s} | "
            f"{rand['crashed']}"
        )

    plan_rewards = comparison["planner_returns"]
    rand_rewards = comparison["random_returns"]
    print(
        f"planner: {np.mean(plan_rewards):.3f} +/- {np.std(plan_rewards):.3f} | "
        f"random:  {np.mean(rand_rewards):.3f} +/- {np.std(rand_rewards):.3f}"
    )
    print(
        f"crashes: planner {comparison['planner_crashes']}/{args.episodes} | "
        f"random {comparison['random_crashes']}/{args.episodes}"
    )
    print("+/- is episode SD (ddof=0), not a confidence interval.")


if __name__ == "__main__":
    main()
