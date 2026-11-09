"""Reusable Lab 2 experiments: the planning-paradox investigation.

All the heavy logic for Lab 2 lives here so the notebooks and the
``evaluate_planning.py`` script stay thin. Each function answers one question in
the student storyline:

* :func:`return_vs_budget` -- *Does more search help or hurt?* (M0)
* :func:`overoptimism_vs_budget` -- *Is the model over-optimistic, and does that
  grow with search?* (M1)
* :func:`error_vs_disagreement` -- *Can the model tell when it is wrong?* (M2)
* :func:`lambda_sweep` -- *Does pessimism restore performance?* (M3)

Everything runs against a **frozen** world model; the only trained addition is
the reward ensemble.
"""

from __future__ import annotations

import numpy as np
import torch

from lab1.config import Config
from lab1.envs.highway import make_env
from lab1.models import RewardEnsemble, WorldModel
from lab1.planning.random_shooting import RandomShootingPlanner, rollout_with_policy


def _random_policy(num_actions: int, seed: int):
    rng = np.random.default_rng(seed)
    return lambda obs: int(rng.integers(0, num_actions))


def _episode_stats(returns: np.ndarray, crashes: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(returns.mean()),
        "std": float(returns.std()),
        "crash_rate": float(crashes.mean()),
    }


def random_baseline(
    model: WorldModel,
    cfg: Config,
    episodes: int,
    max_steps: int,
    seed0: int = 0,
) -> dict[str, float]:
    """Mean return / crash rate of a uniform-random policy (shared reference)."""
    returns, crashes = [], []
    for ep in range(episodes):
        seed = seed0 + ep
        out = rollout_with_policy(
            cfg, _random_policy(model.num_actions, seed), max_steps, seed=seed
        )
        returns.append(out["total_reward"])
        crashes.append(out["crashed"])
    return _episode_stats(np.array(returns), np.array(crashes))


def _run_planner_episodes(
    model: WorldModel,
    cfg: Config,
    num_sequences: int,
    horizon: int,
    episodes: int,
    max_steps: int,
    device: torch.device | str,
    reward_ensemble: RewardEnsemble | None,
    penalty_weight: float,
    seed0: int,
) -> dict[str, float]:
    returns, crashes = [], []
    for ep in range(episodes):
        seed = seed0 + ep
        planner = RandomShootingPlanner(
            model,
            num_sequences=num_sequences,
            horizon=horizon,
            device=device,
            seed=seed,
            reward_ensemble=reward_ensemble,
            penalty_weight=penalty_weight,
        )
        out = rollout_with_policy(cfg, planner.act, max_steps, seed=seed)
        returns.append(out["total_reward"])
        crashes.append(out["crashed"])
    return _episode_stats(np.array(returns), np.array(crashes))


def return_vs_budget(
    model: WorldModel,
    cfg: Config,
    budgets: list[int],
    horizon: int = 6,
    episodes: int = 30,
    max_steps: int = 30,
    device: torch.device | str = "cpu",
    reward_ensemble: RewardEnsemble | None = None,
    penalty_weight: float = 0.0,
    seed0: int = 0,
) -> dict[str, object]:
    """M0/M3: real MPC return as the candidate budget grows.

    With ``reward_ensemble=None`` this tests the Lab 1 planning paradox: more
    search may improve the internal optimum without reliably improving real
    return. With an ensemble and ``penalty_weight>0`` it evaluates whether an
    uncertainty-aware score transfers better. Finite-sample curves can be
    noisy and non-monotonic.
    """
    rows = []
    for n in budgets:
        stats = _run_planner_episodes(
            model,
            cfg,
            num_sequences=n,
            horizon=horizon,
            episodes=episodes,
            max_steps=max_steps,
            device=device,
            reward_ensemble=reward_ensemble,
            penalty_weight=penalty_weight,
            seed0=seed0,
        )
        stats["budget"] = n
        rows.append(stats)
    return {
        "budgets": list(budgets),
        "mean": np.array([r["mean"] for r in rows]),
        "std": np.array([r["std"] for r in rows]),
        "crash_rate": np.array([r["crash_rate"] for r in rows]),
        "rows": rows,
    }


def overoptimism_vs_budget(
    model: WorldModel,
    cfg: Config,
    budgets: list[int],
    horizon: int = 6,
    num_plans: int = 60,
    warmup: int = 5,
    device: torch.device | str = "cpu",
    reward_ensemble: RewardEnsemble | None = None,
    penalty_weight: float = 0.0,
    seed0: int = 0,
) -> dict[str, object]:
    """M1: predicted vs realized return of the *selected* plan, per budget.

    For each budget the planner picks its single best plan from a warmed-up
    state; we then execute that whole plan **open-loop** and compare the
    realized cumulative reward against the return the planner predicted. The
    gap (predicted - realized) is the "winner's curse": the argmax over more
    candidates increasingly selects plans the reward model is over-optimistic
    about, so the gap grows with the budget.
    """
    num_actions = model.num_actions
    rows = []
    for n in budgets:
        preds, reals = [], []
        for i in range(num_plans):
            seed = seed0 + i
            env = make_env(cfg.env)
            obs, _ = env.reset(seed=seed)
            warm_rng = np.random.default_rng(seed)
            for _ in range(warmup):
                obs, _, term, trunc, _ = env.step(
                    int(warm_rng.integers(0, num_actions))
                )
                if term or trunc:
                    obs, _ = env.reset(seed=seed + 9973)
            planner = RandomShootingPlanner(
                model,
                num_sequences=n,
                horizon=horizon,
                device=device,
                seed=seed,
                reward_ensemble=reward_ensemble,
                penalty_weight=penalty_weight,
            )
            best_actions, predicted = planner.plan(obs)
            realized = 0.0
            for h in range(horizon):
                obs, r, term, trunc, _ = env.step(int(best_actions[h]))
                realized += float(r)
                if term or trunc:
                    break
            env.close()
            preds.append(predicted)
            reals.append(realized)
        preds = np.array(preds)
        reals = np.array(reals)
        gap = preds - reals
        rows.append(
            {
                "budget": n,
                "predicted_mean": float(preds.mean()),
                "realized_mean": float(reals.mean()),
                "gap_mean": float(gap.mean()),
                "gap_std": float(gap.std()),
            }
        )
    return {
        "budgets": list(budgets),
        "predicted_mean": np.array([r["predicted_mean"] for r in rows]),
        "realized_mean": np.array([r["realized_mean"] for r in rows]),
        "gap_mean": np.array([r["gap_mean"] for r in rows]),
        "gap_std": np.array([r["gap_std"] for r in rows]),
        "rows": rows,
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks starting at 0, with tied values sharing their average rank."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and sorted_values[end + 1] == sorted_values[start]:
            end += 1
        ranks[order[start : end + 1]] = (start + end) / 2.0
        start = end + 1
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation using only NumPy.

    Ties receive average ranks. A constant input has no rank order, so the
    correlation is undefined and ``nan`` is returned.
    """
    xr, yr = _average_ranks(x), _average_ranks(y)
    if xr.std() == 0 or yr.std() == 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


@torch.no_grad()
def reward_overoptimism(
    model: WorldModel,
    ensemble: RewardEnsemble | None,
    episodes: list,
    device: torch.device | str = "cpu",
    bins: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0001),
) -> dict[str, object]:
    """M1: where does the reward model err? Bin transitions by *true* reward.

    This is the crux of the paradox and it needs no environment rollouts. We
    predict the reward for every (encoded) transition and group by the true
    reward. The supplied checkpoint overpredicts rare low-reward transitions.
    This makes reward prediction an actionable suspect, not a unique causal
    diagnosis of crashes. Reward bins are not safety labels. Where an ensemble
    is supplied, its disagreement is also reported per bin; inspect whether it
    is associated with error rather than assuming it detects failures.
    """
    from lab1.training.reward_ensemble_trainer import encode_transitions

    device = torch.device(device)
    z, a, r = encode_transitions(model, episodes, device)
    pred = model.predict_reward(z, a)
    true = r.cpu().numpy()
    pred = pred.cpu().numpy()
    disagreement = None
    if ensemble is not None:
        _, dis = ensemble.to(device).eval().predict(z, a)
        disagreement = dis.cpu().numpy()

    edges = np.asarray(bins)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        mask = (true >= lo) & (true < hi)
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "lo": float(lo),
                "hi": float(min(hi, 1.0)),
                "count": int(mask.sum()),
                "true_mean": float(true[mask].mean()),
                "pred_mean": float(pred[mask].mean()),
                "bias": float((pred[mask] - true[mask]).mean()),
                "mae": float(np.abs(pred[mask] - true[mask]).mean()),
                "disagreement": (
                    float(disagreement[mask].mean())
                    if disagreement is not None
                    else float("nan")
                ),
            }
        )
    return {
        "rows": rows,
        "overall_mae": float(np.abs(pred - true).mean()),
        "true_mean": np.array([r_["true_mean"] for r_ in rows]),
        "pred_mean": np.array([r_["pred_mean"] for r_ in rows]),
        "bias": np.array([r_["bias"] for r_ in rows]),
        "disagreement": np.array([r_["disagreement"] for r_ in rows]),
        "labels": [f"[{r_['lo']:.1f},{r_['hi']:.1f}]" for r_ in rows],
        "counts": np.array([r_["count"] for r_ in rows]),
    }


@torch.no_grad()
def error_vs_disagreement(
    model: WorldModel,
    ensemble: RewardEnsemble,
    episodes: list,
    horizon: int = 6,
    num_starts: int = 200,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> dict[str, object]:
    """M2: does ensemble disagreement predict the ensemble's reward error?

    We roll the frozen dynamics forward ``horizon`` steps from many start
    states (where predicted inputs may drift off the training manifold), and at
    every imagined step record the ensemble's reward error ``|mean - true|`` and
    its disagreement (member std). If disagreement is informative, the two are
    positively correlated on these samples. This is not calibrated uncertainty,
    a failure probability, or a guarantee of detecting shared model errors.
    """
    device = torch.device(device)
    model = model.to(device).eval()
    ensemble = ensemble.to(device).eval()
    rng = np.random.default_rng(seed)

    candidates = [
        (i, s) for i, ep in enumerate(episodes) for s in range(len(ep) - horizon)
    ]
    if not candidates:
        raise ValueError(f"No episode is long enough for horizon={horizon}.")
    picks = rng.choice(
        len(candidates), size=min(num_starts, len(candidates)), replace=False
    )

    errors, disagreements, steps = [], [], []
    for idx in picks:
        ep_i, start = candidates[idx]
        ep = episodes[ep_i]
        obs0 = (
            torch.from_numpy(ep.obs[start].astype(np.float32) / 255.0)
            .unsqueeze(0)
            .to(device)
        )
        z = model.encode(obs0)
        for h in range(horizon):
            a = torch.tensor([ep.actions[start + h]], device=device)
            mean, dis = ensemble.predict(z, a)
            true_r = float(ep.rewards[start + h])
            errors.append(abs(float(mean.item()) - true_r))
            disagreements.append(float(dis.item()))
            steps.append(h)
            z = model.predict_next(z, a)

    errors = np.array(errors)
    disagreements = np.array(disagreements)
    return {
        "errors": errors,
        "disagreements": disagreements,
        "steps": np.array(steps),
        "pearson": float(np.corrcoef(errors, disagreements)[0, 1]),
        "spearman": _spearman(errors, disagreements),
    }


def lambda_sweep(
    model: WorldModel,
    ensemble: RewardEnsemble,
    cfg: Config,
    lambdas: list[float],
    num_sequences: int = 512,
    horizon: int = 6,
    episodes: int = 30,
    max_steps: int = 30,
    device: torch.device | str = "cpu",
    seed0: int = 0,
) -> dict[str, object]:
    """M3: sweep the pessimism weight at a *large* (exploitation-prone) budget.

    At ``lambda=0`` the ensemble planner is still greedy and exploits errors;
    increasing ``lambda`` trades a little greed for robustness. The sweet spot
    recovers return and cuts the crash rate.
    """
    rows = []
    for lam in lambdas:
        stats = _run_planner_episodes(
            model,
            cfg,
            num_sequences=num_sequences,
            horizon=horizon,
            episodes=episodes,
            max_steps=max_steps,
            device=device,
            reward_ensemble=ensemble,
            penalty_weight=lam,
            seed0=seed0,
        )
        stats["lambda"] = lam
        rows.append(stats)
    return {
        "lambdas": list(lambdas),
        "mean": np.array([r["mean"] for r in rows]),
        "std": np.array([r["std"] for r in rows]),
        "crash_rate": np.array([r["crash_rate"] for r in rows]),
        "rows": rows,
    }
