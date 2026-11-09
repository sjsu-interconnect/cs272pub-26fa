"""Load trajectory datasets and expose them as PyTorch datasets.

The on-disk format is a compressed ``.npz`` written by
:func:`lab1.data.collect.collect_dataset`. Here we load it into a list of
:class:`Episode` objects and provide:

* :class:`TransitionDataset` -- a flat view over single-step transitions
  ``(obs, action, reward, next_obs, done)`` for training the world model.
* :func:`make_dataloaders` -- builds train/validation loaders, splitting **by
  episode** so that temporally adjacent (near-duplicate) frames never leak
  across the split.

Observations are stored as ``uint8`` and normalized to ``float32`` in
``[0, 1]`` inside :meth:`TransitionDataset.__getitem__` -- the single place
where normalization happens.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from lab1.config import Config


@dataclass
class Episode:
    """A single contiguous trajectory.

    Attributes
    ----------
    obs:
        Stacked observations ``(T+1, C, H, W)`` as ``uint8``.
    actions:
        Discrete actions ``(T,)`` as ``int64``.
    rewards:
        Rewards ``(T,)`` as ``float32``.
    dones:
        Episode-end flags ``(T,)`` as ``bool``.
    """

    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray

    def __len__(self) -> int:
        """Number of transitions in the episode (``T``)."""
        return int(self.actions.shape[0])


def load_episodes(path: str | Path) -> list[Episode]:
    """Load all episodes from a ``.npz`` dataset file."""
    path = Path(path)
    with np.load(path) as data:
        num_episodes = int(data["num_episodes"])
        episodes = [
            Episode(
                obs=data[f"obs_{i}"],
                actions=data[f"actions_{i}"],
                rewards=data[f"rewards_{i}"],
                dones=data[f"dones_{i}"],
            )
            for i in range(num_episodes)
        ]
    return episodes


def load_num_actions(path: str | Path) -> int:
    """Return the size of the discrete action space stored in the dataset."""
    with np.load(Path(path)) as data:
        return int(data["num_actions"])


class TransitionDataset(Dataset):
    """Flat single-step view over a list of episodes.

    Each item is a dict of tensors:

    * ``obs``      -- ``float32`` ``(C, H, W)`` in ``[0, 1]``
    * ``action``   -- ``int64`` scalar
    * ``reward``   -- ``float32`` scalar
    * ``next_obs`` -- ``float32`` ``(C, H, W)`` in ``[0, 1]``
    * ``done``     -- ``float32`` scalar (0.0 or 1.0)
    """

    def __init__(self, episodes: list[Episode]) -> None:
        self.episodes = episodes
        # Precompute a flat index of (episode, timestep) pairs.
        self._index: list[tuple[int, int]] = [
            (ep_i, t) for ep_i, ep in enumerate(episodes) for t in range(len(ep))
        ]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_i, t = self._index[idx]
        ep = self.episodes[ep_i]
        obs = _to_float_image(ep.obs[t])
        next_obs = _to_float_image(ep.obs[t + 1])
        return {
            "obs": obs,
            "action": torch.tensor(ep.actions[t], dtype=torch.long),
            "reward": torch.tensor(ep.rewards[t], dtype=torch.float32),
            "next_obs": next_obs,
            "done": torch.tensor(float(ep.dones[t]), dtype=torch.float32),
        }


def _to_float_image(arr: np.ndarray) -> torch.Tensor:
    """Convert a ``uint8`` image to a ``float32`` tensor in ``[0, 1]``."""
    return torch.from_numpy(arr.astype(np.float32) / 255.0)


def split_episodes(
    episodes: list[Episode],
    val_fraction: float,
    seed: int,
) -> tuple[list[Episode], list[Episode]]:
    """Split episodes into train/val lists by whole episode.

    Splitting at the episode level (not the transition level) prevents
    temporally adjacent, near-duplicate frames from leaking across the split.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(episodes))
    n_val = max(1, int(round(len(episodes) * val_fraction)))
    val_idx = set(order[:n_val].tolist())
    train = [ep for i, ep in enumerate(episodes) if i not in val_idx]
    val = [ep for i, ep in enumerate(episodes) if i in val_idx]
    return train, val


def make_dataloaders(
    cfg: Config,
    path: str | Path | None = None,
    seed: int | None = None,
) -> tuple[DataLoader, DataLoader, list[Episode], list[Episode]]:
    """Build train/validation dataloaders from a saved dataset.

    Returns
    -------
    tuple
        ``(train_loader, val_loader, train_episodes, val_episodes)``. The raw
        episode lists are returned as well because imagination and planning
        operate on contiguous trajectories rather than shuffled transitions.
    """
    if path is None:
        path = cfg.data.dataset_path
    if seed is None:
        seed = cfg.seed

    episodes = load_episodes(path)
    train_eps, val_eps = split_episodes(episodes, cfg.data.val_fraction, seed)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TransitionDataset(train_eps),
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        generator=generator,
        drop_last=False,
    )
    val_loader = DataLoader(
        TransitionDataset(val_eps),
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        drop_last=False,
    )
    return train_loader, val_loader, train_eps, val_eps
