"""Configuration system for Lab 1.

Configuration is expressed as nested dataclasses and loaded from YAML files.
Command-line overrides use dotted keys, e.g. ``training.epochs=10`` or
``env.stack_size=1``. There are no hardcoded hyperparameters in the training
code; everything flows through :class:`Config`.

Example
-------
>>> cfg = Config.load("configs/default.yaml", overrides=["training.epochs=5"])
>>> cfg.env.stack_size
4
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class EnvConfig:
    """HighwayEnv observation and action settings."""

    id: str = "highway-fast-v0"
    image_size: int = 64
    stack_size: int = 4  # frame stacking; 1 is simpler but loses motion info
    policy_frequency: int = 1
    simulation_frequency: int = 15
    lanes_count: int = 4
    vehicles_count: int = 25
    duration: int = 30
    offscreen_rendering: bool = True
    scaling: float = 1.75


@dataclass
class DataConfig:
    """Dataset collection and loading settings."""

    num_transitions: int = 20000
    dataset_path: str = "datasets/highway.npz"
    val_fraction: float = 0.1
    batch_size: int = 128
    num_workers: int = 0


@dataclass
class ModelConfig:
    """World-model architecture settings."""

    latent_dim: int = 64
    use_decoder: bool = True
    use_reward_head: bool = True
    encoder_channels: tuple[int, ...] = (32, 64, 128)
    dynamics_hidden: tuple[int, ...] = (256, 256)


@dataclass
class TrainingConfig:
    """Training-loop settings."""

    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0  # max global grad norm; <= 0 disables clipping
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    recon_loss_weight: float = 1.0
    reward_loss_weight: float = 1.0
    dynamics_loss_weight: float = 1.0
    log_dir: str = "runs"
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 5


@dataclass
class ImaginationConfig:
    """Imagined-rollout settings."""

    horizon: int = 10  # rollout length (5-20)


@dataclass
class PlanningConfig:
    """Random-shooting planner settings.

    ``penalty_weight`` (Lab 2) is the pessimism coefficient ``lambda``: the
    uncertainty-aware planner scores a plan by ``sum_t (mean_reward -
    lambda * disagreement)``. With an ensemble, ``0.0`` removes the uncertainty
    penalty but still scores with the ensemble mean; it is therefore not
    identical to Lab 1's single reward head.
    """

    enabled: bool = True
    num_sequences: int = 200
    horizon: int = 10
    penalty_weight: float = 0.0


@dataclass
class EnsembleConfig:
    """Reward-ensemble settings (Lab 2).

    The ensemble is trained on top of a *frozen* Lab 1 backbone, so only these
    small heads learn. Members differ by initialization and by a per-head
    bootstrap resample of the transitions, which is what makes them disagree on
    out-of-distribution latents.
    """

    num_heads: int = 5
    epochs: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    bootstrap_fraction: float = 0.8
    checkpoint_path: str = "checkpoints/strong/reward_ensemble.pt"


@dataclass
class Config:
    """Top-level configuration container."""

    seed: int = 0
    env: EnvConfig = field(default_factory=EnvConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    imagination: ImaginationConfig = field(default_factory=ImaginationConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)

    # -- construction ----------------------------------------------------

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        overrides: list[str] | None = None,
    ) -> Config:
        """Load a config from YAML and apply optional CLI overrides.

        Parameters
        ----------
        path:
            Path to a YAML file. If ``None``, all defaults are used.
        overrides:
            List of ``dotted.key=value`` strings (as produced by argparse
            ``nargs="*"``). Values are parsed as YAML scalars, so ``true``,
            ``10`` and ``3e-4`` are typed correctly.
        """
        data: dict[str, Any] = {}
        if path is not None:
            with open(path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        cfg = cls.from_dict(data)
        if overrides:
            cfg.apply_overrides(overrides)
        return cfg

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        """Build a :class:`Config` from a (possibly partial) nested dict."""
        return _from_dict(cls, data)

    # -- overrides -------------------------------------------------------

    def apply_overrides(self, overrides: list[str]) -> None:
        """Apply ``dotted.key=value`` overrides in place."""
        for item in overrides:
            if "=" not in item:
                raise ValueError(
                    f"Invalid override '{item}'; expected 'dotted.key=value'."
                )
            key, raw_value = item.split("=", 1)
            value = yaml.safe_load(raw_value)
            self._set_dotted(key.strip(), value)

    def _set_dotted(self, dotted_key: str, value: Any) -> None:
        parts = dotted_key.split(".")
        target: Any = self
        for part in parts[:-1]:
            if not hasattr(target, part):
                raise KeyError(f"Unknown config section '{part}' in '{dotted_key}'.")
            target = getattr(target, part)
        leaf = parts[-1]
        if not hasattr(target, leaf):
            raise KeyError(f"Unknown config key '{leaf}' in '{dotted_key}'.")
        setattr(target, leaf, value)

    # -- serialization ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a nested dict representation of this config."""
        return _to_dict(self)


def _from_dict(dc_type: type, data: dict[str, Any]) -> Any:
    """Recursively build a dataclass instance from a nested dict.

    ``from __future__ import annotations`` turns field annotations into
    strings, so we resolve them with :func:`typing.get_type_hints` before
    checking whether a field is itself a dataclass.
    """
    hints = get_type_hints(dc_type)
    kwargs: dict[str, Any] = {}
    for f in fields(dc_type):
        if f.name not in data:
            continue
        value = data[f.name]
        field_type = hints.get(f.name, f.type)
        if is_dataclass(field_type) and isinstance(value, dict):
            kwargs[f.name] = _from_dict(field_type, value)
        else:
            kwargs[f.name] = value
    return dc_type(**kwargs)


def _to_dict(obj: Any) -> Any:
    """Recursively convert a dataclass instance into a plain dict."""
    if is_dataclass(obj):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(v) for v in obj]
    return obj
