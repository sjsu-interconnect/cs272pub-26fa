"""Collect driving experience from HighwayEnv into a trajectory dataset.

The collector runs episodes with a random (or user-supplied) policy and stores
each episode as a contiguous trajectory. Storing whole episodes -- rather than a
shuffled pool of independent transitions -- is what later lets us evaluate
multi-step imagination (prediction error vs. horizon, real vs. imagined
trajectories) without re-collecting data.

Each episode stores:

* ``obs``     -- stacked observations ``o_0 .. o_T`` of shape ``(T+1, C, H, W)``
* ``actions`` -- discrete actions ``a_0 .. a_{T-1}`` of shape ``(T,)``
* ``rewards`` -- scalar rewards ``r_0 .. r_{T-1}`` of shape ``(T,)``
* ``dones``   -- episode-end flags ``d_0 .. d_{T-1}`` of shape ``(T,)``

so that transition ``t`` is ``(obs[t], actions[t], rewards[t], obs[t+1],
dones[t])``. Observations are kept as ``uint8`` on disk; normalization to
``float32`` happens in the :class:`~lab1.data.dataset.TransitionDataset`.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from tqdm import tqdm

from lab1.config import Config
from lab1.envs.highway import make_env

# A policy maps a stacked observation to a discrete action index.
Policy = Callable[[np.ndarray], int]


def collect_dataset(
    cfg: Config,
    out_path: str | Path | None = None,
    seed: int | None = None,
    policy: Policy | None = None,
    progress: bool = True,
) -> Path:
    """Collect transitions from HighwayEnv and save them as episodes.

    Parameters
    ----------
    cfg:
        Full configuration; ``cfg.data.num_transitions`` sets the target size
        and ``cfg.data.dataset_path`` the default output location.
    out_path:
        Where to write the compressed ``.npz`` file. Defaults to
        ``cfg.data.dataset_path``.
    seed:
        Base seed. Defaults to ``cfg.seed``. Each episode uses ``seed + i``.
    policy:
        Optional ``obs -> action`` callable. Defaults to a uniform random
        policy over the discrete action space.
    progress:
        Show a tqdm progress bar over collected transitions.

    Returns
    -------
    pathlib.Path
        The path to the saved dataset.
    """
    if seed is None:
        seed = cfg.seed
    if out_path is None:
        out_path = cfg.data.dataset_path
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = make_env(cfg.env)
    env.action_space.seed(seed)
    num_actions = int(env.action_space.n)

    target = cfg.data.num_transitions
    episodes: list[dict[str, np.ndarray]] = []
    collected = 0
    episode_idx = 0

    bar = tqdm(total=target, disable=not progress, desc="collecting", unit="step")
    while collected < target:
        obs, _ = env.reset(seed=seed + episode_idx)
        obs_seq: list[np.ndarray] = [np.asarray(obs, dtype=np.uint8)]
        actions: list[int] = []
        rewards: list[float] = []
        dones: list[bool] = []

        done = False
        while not done:
            action = (
                int(env.action_space.sample()) if policy is None else int(policy(obs))
            )
            obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)

            obs_seq.append(np.asarray(obs, dtype=np.uint8))
            actions.append(action)
            rewards.append(float(reward))
            dones.append(done)

            collected += 1
            bar.update(1)
            if collected >= target:
                break

        episodes.append(
            {
                "obs": np.stack(obs_seq).astype(np.uint8),
                "actions": np.asarray(actions, dtype=np.int64),
                "rewards": np.asarray(rewards, dtype=np.float32),
                "dones": np.asarray(dones, dtype=bool),
            }
        )
        episode_idx += 1
    bar.close()
    env.close()

    _save_episodes(out_path, episodes, num_actions=num_actions)
    return out_path


def _save_episodes(
    path: Path,
    episodes: list[dict[str, np.ndarray]],
    num_actions: int,
) -> None:
    """Write episodes to a compressed ``.npz`` file with per-episode keys."""
    payload: dict[str, np.ndarray] = {
        "num_episodes": np.asarray(len(episodes)),
        "num_actions": np.asarray(num_actions),
    }
    for i, ep in enumerate(episodes):
        payload[f"obs_{i}"] = ep["obs"]
        payload[f"actions_{i}"] = ep["actions"]
        payload[f"rewards_{i}"] = ep["rewards"]
        payload[f"dones_{i}"] = ep["dones"]
    np.savez_compressed(path, **payload)
