"""Shared argument parsing for the CLI scripts."""

from __future__ import annotations

import argparse

from lab1.config import Config


def parse_config(description: str) -> tuple[Config, argparse.Namespace]:
    """Parse ``--config`` plus ``dotted.key=value`` overrides.

    Returns the loaded :class:`~lab1.config.Config` and the full namespace so
    individual scripts can read their own extra arguments.
    """
    parser = build_parser(description)
    args = parser.parse_args()
    cfg = Config.load(args.config, args.overrides)
    return cfg, args


def build_parser(description: str) -> argparse.ArgumentParser:
    """Build a parser with the common ``--config`` and overrides arguments."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to a YAML config file.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides as dotted.key=value (e.g. training.epochs=10).",
    )
    return parser


def load_trained_model(cfg: Config, checkpoint: str):
    """Load a trained :class:`~lab1.models.WorldModel` from a checkpoint."""
    from lab1.data import load_num_actions
    from lab1.models import WorldModel
    from lab1.utils.checkpoint import load_checkpoint

    num_actions = load_num_actions(cfg.data.dataset_path)
    model = WorldModel.from_config(cfg.model, cfg.env, num_actions)
    load_checkpoint(checkpoint, model)
    model.eval()
    return model
