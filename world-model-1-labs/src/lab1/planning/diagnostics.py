"""Concise diagnostics used by the planning notebook.

These functions expose the quantities students should inspect while keeping
tensor batching and repeated evaluation loops out of the notebook itself.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from lab1.models import WorldModel
from lab1.planning.random_shooting import RandomShootingPlanner, rollout_with_policy


@torch.no_grad()
def sample_candidate_plans(
    model: WorldModel,
    obs: np.ndarray,
    *,
    num_sequences: int = 64,
    horizon: int = 6,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> dict[str, object]:
    """Sample and score one batch of random-shooting candidate plans."""
    device = torch.device(device)
    model = model.to(device).eval()
    rng = np.random.default_rng(seed)
    actions = rng.integers(
        0, model.num_actions, size=(num_sequences, horizon), dtype=np.int64
    )
    obs_t = (
        torch.from_numpy(obs.astype(np.float32) / 255.0)
        .unsqueeze(0)
        .expand(num_sequences, -1, -1, -1)
        .contiguous()
        .to(device)
    )
    rewards = (
        model.imagine(obs_t, torch.from_numpy(actions).to(device))["rewards"]
        .cpu()
        .numpy()
    )
    returns = rewards.sum(axis=1)
    best = int(np.argmax(returns))
    return {
        "actions": actions,
        "rewards": rewards,
        "returns": returns,
        "best_index": best,
        "num_sequences": num_sequences,
        "horizon": horizon,
    }


def probe_planning_decision(
    model: WorldModel,
    obs: np.ndarray,
    *,
    num_sequences: int = 64,
    horizon: int = 6,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> dict[str, object]:
    """Probe one internal planning decision from one fixed observation.

    Reuses :func:`sample_candidate_plans` (sample, imagine, score, select). It
    never creates or steps an environment, so it says nothing about episode
    return or safety. ``seconds`` times a second, identical call so that
    one-off device warm-up for a new batch shape is not counted.
    """
    kwargs = {
        "num_sequences": num_sequences,
        "horizon": horizon,
        "seed": seed,
        "device": device,
    }
    sample_candidate_plans(model, obs, **kwargs)  # warm-up, not timed
    start = time.perf_counter()
    candidates = sample_candidate_plans(model, obs, **kwargs)
    seconds = time.perf_counter() - start
    sequence = [int(a) for a in candidates["actions"][candidates["best_index"]]]
    best_return = float(candidates["returns"][candidates["best_index"]])
    return {
        "num_sequences": num_sequences,
        "horizon": horizon,
        "imagined_transitions": num_sequences * horizon,
        "seed": seed,
        "first_action": sequence[0],
        "sequence": sequence,
        "best_J": best_return,
        "J_per_step": best_return / horizon,
        "seconds": seconds,
    }


def evaluate_planner_vs_random(
    model: WorldModel,
    cfg,
    *,
    episodes: int = 5,
    max_steps: int = 30,
    seed0: int = 0,
    device: torch.device | str = "cpu",
) -> dict[str, object]:
    """Evaluate receding-horizon planning and a random policy on shared seeds."""
    planner_runs, random_runs = [], []
    for episode_index in range(episodes):
        seed = seed0 + episode_index
        planner = RandomShootingPlanner(
            model,
            num_sequences=cfg.planning.num_sequences,
            horizon=cfg.planning.horizon,
            device=device,
            seed=seed,
        )
        planner_runs.append(
            rollout_with_policy(cfg, planner.act, max_steps=max_steps, seed=seed)
        )

        rng = np.random.default_rng(seed)
        random_runs.append(
            rollout_with_policy(
                cfg,
                lambda obs, rng=rng: int(rng.integers(0, model.num_actions)),
                max_steps=max_steps,
                seed=seed,
            )
        )

    planner_returns = np.asarray(
        [run["total_reward"] for run in planner_runs], dtype=float
    )
    random_returns = np.asarray(
        [run["total_reward"] for run in random_runs], dtype=float
    )
    return {
        "planner": planner_runs,
        "random": random_runs,
        "planner_returns": planner_returns,
        "random_returns": random_returns,
        "planner_crashes": int(sum(bool(run["crashed"]) for run in planner_runs)),
        "random_crashes": int(sum(bool(run["crashed"]) for run in random_runs)),
    }
