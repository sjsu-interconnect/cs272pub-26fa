"""Collect a trajectory dataset from HighwayEnv.

Usage::

    python scripts/collect.py --config configs/lab.yaml
"""

from __future__ import annotations

from pathlib import Path

from _cli import parse_config

from lab1.data import collect_dataset, load_episodes


def main() -> None:
    cfg, _ = parse_config("Collect a HighwayEnv trajectory dataset.")
    destination = Path(cfg.data.dataset_path)
    if destination.exists():
        try:
            existing = sum(len(episode) for episode in load_episodes(destination))
        except Exception:
            raise SystemExit(
                f"The dataset already exists at {destination}, could not be "
                "verified, and was not overwritten. Move or delete it before "
                "collecting again."
            ) from None
        requested = int(cfg.data.num_transitions)
        if existing == requested:
            print(
                f"Dataset already ready: {destination} "
                f"({existing:,} transitions). Continue to Lab 1."
            )
            return
        suggested = destination.with_name(
            f"{destination.stem}_{requested}{destination.suffix}"
        )
        raise SystemExit(
            f"The existing dataset has {existing:,} transitions, but "
            f"configs/lab.yaml requests {requested:,}. It was not overwritten.\n"
            "Choose one option, then run this command again:\n"
            f"1. Keep the existing data: edit data.dataset_path in configs/lab.yaml "
            f"to {suggested}.\n"
            f"2. Replace the existing data: move or delete {destination}."
        )
    path = collect_dataset(cfg)
    print(f"Saved dataset to {path}")


if __name__ == "__main__":
    main()
