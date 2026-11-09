"""Model-only checkpoints for optional student-to-student model comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from lab1.config import Config
from lab1.models import WorldModel

FORMAT_ID = "lab1-world-model-personal"
FORMAT_VERSION = 1


def _architecture(model: WorldModel, cfg: Config) -> dict[str, Any]:
    return {
        "image_size": cfg.env.image_size,
        "frame_stack_size": cfg.env.stack_size,
        "latent_dimension": model.latent_dim,
        "number_of_actions": model.num_actions,
        "encoder_channels": list(cfg.model.encoder_channels),
        "dynamics_hidden": list(cfg.model.dynamics_hidden),
        "decoder": model.decoder is not None,
        "reward_head": model.reward_head is not None,
    }


def _signature(architecture: dict[str, Any]) -> str:
    encoded = json.dumps(architecture, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def save_personal_checkpoint(
    path: str | Path,
    model: WorldModel,
    cfg: Config,
    *,
    training_transitions: int,
    epochs: int,
    seed: int,
) -> Path:
    """Write a versioned model-only checkpoint, without optimizer state."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    architecture = _architecture(model, cfg)
    payload = {
        "format": FORMAT_ID,
        "version": FORMAT_VERSION,
        "architecture": architecture,
        "architecture_signature": _signature(architecture),
        "image_size": architecture["image_size"],
        "frame_stack_size": architecture["frame_stack_size"],
        "latent_dimension": architecture["latent_dimension"],
        "number_of_actions": architecture["number_of_actions"],
        "training_transitions": int(training_transitions),
        "epochs": int(epochs),
        "seed": int(seed),
        "model_state": {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        },
    }
    torch.save(payload, destination)
    return destination


def load_personal_checkpoint(
    path: str | Path,
    model: WorldModel,
    cfg: Config,
) -> dict[str, Any]:
    """Validate a personal checkpoint's format and architecture before loading."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT_ID:
        raise ValueError("This is not a supported Lab 1 model-only checkpoint.")
    if payload.get("version") != FORMAT_VERSION:
        raise ValueError("This Lab 1 model checkpoint uses an unsupported version.")

    expected = _architecture(model, cfg)
    if payload.get("architecture") != expected or payload.get(
        "architecture_signature"
    ) != _signature(expected):
        raise ValueError(
            "This Lab 1 model has an incompatible architecture for the supplied "
            "Lab 2 comparison model."
        )
    for key in (
        "image_size",
        "frame_stack_size",
        "latent_dimension",
        "number_of_actions",
    ):
        if payload.get(key) != expected[key]:
            raise ValueError(f"This Lab 1 model checkpoint has incompatible {key}.")
    if "optimizer_state" in payload:
        raise ValueError("Personal model checkpoints must not contain optimizer state.")

    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return payload
