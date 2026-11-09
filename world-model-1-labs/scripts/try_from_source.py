"""Tiny collect/train/imagine/plan check, not a classroom-result reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from _cli import load_trained_model

from lab1.config import Config
from lab1.data import collect_dataset, make_dataloaders
from lab1.planning import evaluate_planner_vs_random
from lab1.training import resolve_device, train
from lab1.visualization import plot_error_vs_horizon, plot_imagination_grid


def run_source_check(output_dir: Path, device: str = "cpu") -> dict:
    """Create all smoke-check artifacts in a new directory; never reuse one."""
    cfg = Config.load("configs/lab.yaml")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    cfg.data.dataset_path = str(output_dir / "trajectories.npz")
    cfg.data.num_transitions = 128
    cfg.data.batch_size = 32
    cfg.training.epochs = 1
    cfg.training.device = device
    cfg.training.checkpoint_dir = str(output_dir / "checkpoints")
    cfg.training.log_dir = str(output_dir / "runs")
    cfg.planning.num_sequences = 8
    cfg.planning.horizon = 3

    print("PIPELINE CHECK ONLY: 128 transitions, 1 epoch; not a performance result.")
    print("All generated files:", output_dir)
    collect_dataset(cfg)
    checkpoint = train(cfg)
    model = load_trained_model(cfg, str(checkpoint))
    resolved_device = resolve_device(device)
    _, _, _, val_eps = make_dataloaders(cfg)
    episode = max(val_eps, key=len)
    horizon = min(3, len(episode) - 1)
    plot_imagination_grid(
        model,
        episode,
        horizon=horizon,
        device=resolved_device,
        path=output_dir / "imagination.png",
    )
    plot_error_vs_horizon(
        model,
        val_eps,
        horizon=horizon,
        num_samples=8,
        device=resolved_device,
        path=output_dir / "error_vs_horizon.png",
    )
    plt.close("all")
    comparison = evaluate_planner_vs_random(
        model,
        cfg,
        episodes=2,
        max_steps=5,
        seed0=cfg.seed,
        device=resolved_device,
    )
    report = {
        "purpose": "pipeline check only; not the full lab experiment",
        "device": str(resolved_device),
        "config": cfg.to_dict(),
        "checkpoint": str(checkpoint),
        "planner_returns": comparison["planner_returns"].tolist(),
        "random_returns": comparison["random_returns"].tolist(),
        "planner_crashes": comparison["planner_crashes"],
        "random_crashes": comparison["random_crashes"],
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Finished: dataset, checkpoint, history, two figures, and summary.json.")
    print("This tiny model is not used by the required notebooks.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/source_check"))
    parser.add_argument(
        "--device", default="cpu", choices=("cpu", "auto", "mps", "cuda")
    )
    args = parser.parse_args()
    try:
        run_source_check(args.output_dir, args.device)
    except FileExistsError:
        parser.error("--output-dir already exists; choose a new directory")


if __name__ == "__main__":
    main()
