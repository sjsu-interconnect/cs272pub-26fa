"""Train the world model.

Usage::

    python scripts/train.py --config configs/lab.yaml
    python scripts/train.py --config configs/default.yaml training.epochs=30
"""

from __future__ import annotations

from _cli import parse_config

from lab1.training import train


def main() -> None:
    cfg, _ = parse_config("Train the latent world model.")
    best = train(cfg)
    print(f"Best checkpoint: {best}")


if __name__ == "__main__":
    main()
