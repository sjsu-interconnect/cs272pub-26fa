"""Checkpoint saving/loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save model (and optionally optimizer) state to ``path``.

    Parameters
    ----------
    path:
        Destination file. Parent directories are created if needed.
    model:
        The module whose ``state_dict`` is saved.
    optimizer:
        Optional optimizer whose ``state_dict`` is also saved.
    epoch:
        Epoch index stored alongside the weights.
    extra:
        Optional extra metadata (e.g. the config dict) to store.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if extra is not None:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load weights from ``path`` into ``model`` (and optionally optimizer).

    Returns
    -------
    dict
        The full checkpoint payload (including ``epoch`` and any ``extra``).
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload
