"""Random-shooting planner (a.k.a. random-sampling MPC).

This is the payoff of Lab 1: planning **inside the learned world model**, with
no reinforcement learning and no calls to the simulator during planning.

The recipe (only a few lines of real logic):

1. Encode the current observation once.
2. Sample many random action sequences.
3. Roll each sequence through the dynamics model, scoring predicted rewards.
4. Keep the sequence with the highest predicted return.
5. Execute only its first action, then re-plan next step (model-predictive).

Lab 2 extends this **same** planner with an optional pessimism term. If a
:class:`~lab1.models.RewardEnsemble` is supplied, a plan is scored by

    sum_t (mean_reward_t - penalty_weight * disagreement_t)

so that plans relying on latents the model is unsure about are discounted. With
``penalty_weight=0`` (and no ensemble) the behaviour is *identical* to Lab 1.

A reward signal is required: either the world model's reward head
(``model.use_reward_head=True``) or a reward ensemble.
"""

from __future__ import annotations

import numpy as np
import torch

from lab1.envs.display import DisplayCamera
from lab1.envs.highway import make_env
from lab1.models import RewardEnsemble, WorldModel


class RandomShootingPlanner:
    """Pick actions by shooting random plans through the world model.

    Parameters
    ----------
    model:
        A trained world model. Used for the dynamics rollout and, when no
        ensemble is given, for the reward score.
    num_sequences:
        Number of random action sequences sampled per decision. The lab's
        recommended *naive* default is small (64): a larger pool lets the search
        expose model errors, but need not reliably improve or lower real return.
    horizon:
        Planning horizon (length of each sampled sequence). Recommended 6.
    device:
        Device on which to run the imagined rollouts.
    seed:
        Seed for the action-sampling RNG.
    reward_ensemble:
        Optional :class:`~lab1.models.RewardEnsemble`. When supplied, the plan
        score becomes ``mean - penalty_weight * disagreement`` summed over the
        horizon. This is the Lab 2 "planning under uncertainty" mechanism.
    penalty_weight:
        Pessimism coefficient ``lambda``. Only used when an ensemble is given.
        ``0.0`` scores by the ensemble mean alone (no pessimism).
    """

    def __init__(
        self,
        model: WorldModel,
        num_sequences: int = 64,
        horizon: int = 6,
        device: torch.device | str = "cpu",
        seed: int = 0,
        reward_ensemble: RewardEnsemble | None = None,
        penalty_weight: float = 0.0,
    ) -> None:
        if model.reward_head is None and reward_ensemble is None:
            raise RuntimeError(
                "Random-shooting planning requires a reward signal: either a "
                "world model with a reward head (model.use_reward_head=True) or "
                "a reward_ensemble."
            )
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.num_sequences = num_sequences
        self.horizon = horizon
        self.num_actions = model.num_actions
        self.rng = np.random.default_rng(seed)
        self.reward_ensemble = (
            reward_ensemble.to(self.device).eval()
            if reward_ensemble is not None
            else None
        )
        self.penalty_weight = penalty_weight
        # Diagnostic: the planner's own predicted (mean) return for the plan it
        # just chose. Compared against the *realized* return, this exposes
        # reward-model over-optimism (see scripts/evaluate_planning.py).
        self.last_predicted_return: float | None = None

    @torch.no_grad()
    def plan(self, obs: np.ndarray) -> tuple[np.ndarray, float]:
        """Return the best imagined ``(action_sequence, predicted_return)``.

        ``action_sequence`` has length ``horizon``; ``predicted_return`` is the
        planner's own (mean) estimate of that plan's cumulative reward. Exposing
        the whole plan lets us execute it open-loop and measure how optimistic
        the estimate was (the reward-model exploitation diagnostic).
        """
        obs_t = (
            torch.from_numpy(obs.astype(np.float32) / 255.0)
            .unsqueeze(0)
            .to(self.device)
        )
        z = self.model.encode(obs_t).expand(self.num_sequences, -1).contiguous()

        actions = self.rng.integers(
            0, self.num_actions, size=(self.num_sequences, self.horizon)
        )
        actions_t = torch.from_numpy(actions).to(self.device)

        score = torch.zeros(self.num_sequences, device=self.device)
        predicted_return = torch.zeros(self.num_sequences, device=self.device)
        for h in range(self.horizon):
            a = actions_t[:, h]
            if self.reward_ensemble is not None:
                mean, disagreement = self.reward_ensemble.predict(z, a)
                score += mean - self.penalty_weight * disagreement
                predicted_return += mean
            else:
                reward = self.model.predict_reward(z, a)
                score += reward
                predicted_return += reward
            z = self.model.predict_next(z, a)

        best = int(torch.argmax(score).item())
        # The predicted return the planner believed the *selected* plan would
        # earn (ensemble mean, or reward head) -- not the penalized score.
        self.last_predicted_return = float(predicted_return[best].item())
        return actions[best].copy(), self.last_predicted_return

    def act(self, obs: np.ndarray) -> int:
        """Return the first action of the best imagined plan for ``obs``."""
        best_actions, _ = self.plan(obs)
        return int(best_actions[0])


def rollout_with_policy(
    cfg,
    policy,
    max_steps: int = 30,
    seed: int = 0,
    record_display: bool = False,
) -> dict[str, object]:
    """Run one real episode driving the env with ``policy`` (obs -> action).

    Used to compare the planner against a random baseline in the notebook. The
    simulator *is* stepped here -- this measures real return of the actions the
    planner chose (each action was itself selected by imagining, not simulating).

    The policy always receives the original stacked grayscale observation, the
    same input the world model was trained on. With ``record_display=True`` a
    separate, display-only camera (:class:`lab1.envs.display.DisplayCamera`)
    also records color frames for figures; they are never passed to the policy
    and do not change the simulation.

    Returns
    -------
    dict
        ``total_reward``, ``steps``, ``crashed`` and aligned lists of stacked
        observations, actions, rewards, and done flags.  Observations have one
        more element than transition-level metadata. ``frames`` and its alias
        ``model_frames`` are the model observations; ``lane_indices`` is the
        ego lane (0 = leftmost) at each observation. With ``record_display``,
        ``display_frames`` and ``display_ego_boxes`` (one per observation) hold
        the display-only view and the ego's pixel box in it.
    """
    env = make_env(cfg.env)
    obs, _ = env.reset(seed=seed)
    camera = DisplayCamera(env) if record_display else None
    display_frames, display_ego_boxes = [], []

    def capture_display() -> None:
        if camera is not None:
            display_frame, ego_box = camera.render()
            display_frames.append(display_frame)
            display_ego_boxes.append(ego_box)

    capture_display()
    total_reward = 0.0
    crashed = False
    frames = [np.asarray(obs, dtype=np.uint8)]
    lane_indices = [int(env.unwrapped.vehicle.lane_index[2])]
    actions = []
    rewards = []
    dones = []
    steps = 0
    for _ in range(max_steps):
        action = int(policy(obs))  # the policy sees only the model observation
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        total_reward += float(reward)
        frames.append(np.asarray(obs, dtype=np.uint8))
        lane_indices.append(int(env.unwrapped.vehicle.lane_index[2]))
        capture_display()
        actions.append(action)
        rewards.append(float(reward))
        dones.append(done)
        steps += 1
        if done:
            crashed = bool(info.get("crashed", terminated))
            break
    env.close()
    result = {
        "total_reward": total_reward,
        "steps": steps,
        "crashed": crashed,
        "frames": frames,
        "model_frames": frames,
        "lane_indices": lane_indices,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
    }
    if record_display:
        result["display_frames"] = display_frames
        result["display_ego_boxes"] = display_ego_boxes
    return result


def imagination_budget(
    real_steps: int,
    num_candidates: int,
    horizon: int,
) -> dict[str, int]:
    """Count imagined vs real transitions for a planning run.

    A world model lets the agent *think* far more than it *acts*: for every
    real environment step, the planner rolls ``num_candidates`` action
    sequences of length ``horizon`` entirely inside the model.

    Returns a dict with ``real_steps``, ``imagined_steps`` and the integer
    ``ratio`` of imagined transitions per real step.
    """
    imagined_steps = real_steps * num_candidates * horizon
    ratio = num_candidates * horizon  # imagined steps per real step
    return {
        "real_steps": real_steps,
        "num_candidates": num_candidates,
        "horizon": horizon,
        "imagined_steps": imagined_steps,
        "ratio": ratio,
    }
