"""Train a reward ensemble on top of a frozen Lab 1 backbone (Lab 2).

The whole point of Lab 2 is to add an *uncertainty estimate* without disturbing
the world model students already understand. So this trainer:

1. **Freezes** the Lab 1 encoder/dynamics/decoder entirely.
2. **Encodes every transition once** into a latent (the expensive part, done a
   single time because the encoder never changes).
3. Trains ``K`` small reward heads on those cached latents. Each head sees a
   different bootstrap subsample and starts from a different initialization, so
   the heads agree on familiar latents and disagree on unfamiliar ones.

Because only tiny MLPs train on cached 64-d vectors, the whole thing runs in
well under a minute on CPU/MPS.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from lab1.config import Config
from lab1.data.dataset import Episode
from lab1.models import RewardEnsemble, WorldModel
from lab1.utils.checkpoint import load_checkpoint, save_checkpoint


@torch.no_grad()
def encode_transitions(
    model: WorldModel,
    episodes: list[Episode],
    device: torch.device,
    batch_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode every transition's observation into a cached latent.

    Returns ``(latents (N, D), actions (N,), rewards (N,))`` on ``device``. The
    frozen encoder is applied exactly once, which is what makes ensemble
    training cheap.
    """
    model = model.to(device).eval()
    obs_list, act_list, rew_list = [], [], []
    for ep in episodes:
        obs_list.append(ep.obs[: len(ep)])  # align with actions/rewards
        act_list.append(ep.actions)
        rew_list.append(ep.rewards)
    obs = np.concatenate(obs_list, axis=0)
    actions = np.concatenate(act_list, axis=0)
    rewards = np.concatenate(rew_list, axis=0)

    latents = []
    for start in range(0, len(obs), batch_size):
        chunk = obs[start : start + batch_size].astype(np.float32) / 255.0
        latents.append(model.encode(torch.from_numpy(chunk).to(device)))
    latents_t = torch.cat(latents, dim=0)
    actions_t = torch.from_numpy(actions.astype(np.int64)).to(device)
    rewards_t = torch.from_numpy(rewards.astype(np.float32)).to(device)
    return latents_t, actions_t, rewards_t


def _train_one_head(
    head: nn.Module,
    num_actions: int,
    latents: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    subset: np.ndarray,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    rng: np.random.Generator,
) -> None:
    """Train a single reward head on its bootstrap ``subset`` of the data."""
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    n = len(subset)
    for _ in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = subset[perm[start : start + batch_size]]
            z = latents[idx]
            a = F.one_hot(actions[idx], num_classes=num_actions).float()
            pred = head(z, a)
            loss = F.mse_loss(pred, rewards[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()


@torch.no_grad()
def _mae(head_or_ensemble, num_actions, latents, actions, rewards) -> float:
    a = F.one_hot(actions, num_classes=num_actions).float()
    if isinstance(head_or_ensemble, RewardEnsemble):
        pred = head_or_ensemble.forward(latents, actions).mean(dim=0)
    else:
        pred = head_or_ensemble(latents, a)
    return float((pred - rewards).abs().mean().item())


def train_reward_ensemble(
    cfg: Config,
    model: WorldModel,
    train_eps: list[Episode],
    val_eps: list[Episode],
    device: torch.device,
    batch_size: int = 256,
    verbose: bool = True,
) -> tuple[RewardEnsemble, dict[str, object]]:
    """Train a :class:`RewardEnsemble` with the world-model backbone frozen.

    Returns the trained ensemble and a small history dict (per-head and
    ensemble validation MAE) suitable for logging or plotting.
    """
    num_actions = model.num_actions
    ens_cfg = cfg.ensemble

    z_tr, a_tr, r_tr = encode_transitions(model, train_eps, device)
    z_val, a_val, r_val = encode_transitions(model, val_eps, device)

    ensemble = RewardEnsemble.from_config(
        cfg.model, cfg.env, num_actions, num_heads=ens_cfg.num_heads
    ).to(device)

    n = z_tr.shape[0]
    subset_size = max(1, int(round(ens_cfg.bootstrap_fraction * n)))
    per_head_mae: list[float] = []
    for k, head in enumerate(ensemble.heads):
        rng = np.random.default_rng(cfg.seed + 1000 + k)
        # Bootstrap: each head trains on a different random subsample so the
        # members genuinely diverge (bagging).
        subset = rng.choice(n, size=subset_size, replace=False)
        _train_one_head(
            head,
            num_actions,
            z_tr,
            a_tr,
            r_tr,
            subset,
            epochs=ens_cfg.epochs,
            lr=ens_cfg.learning_rate,
            weight_decay=ens_cfg.weight_decay,
            batch_size=batch_size,
            rng=rng,
        )
        mae_k = _mae(head, num_actions, z_val, a_val, r_val)
        per_head_mae.append(mae_k)
        if verbose:
            print(f"  head {k}: val reward MAE = {mae_k:.4f}")

    ens_mae = _mae(ensemble, num_actions, z_val, a_val, r_val)
    # Mean disagreement on held-out (in-distribution) data -- a sanity check
    # that the heads have not collapsed to identical functions.
    _, disagreement = ensemble.predict(z_val, a_val)
    mean_disagreement = float(disagreement.mean().item())
    if verbose:
        print(
            f"  ensemble: val reward MAE = {ens_mae:.4f} | "
            f"mean in-dist disagreement = {mean_disagreement:.4f}"
        )

    history = {
        "per_head_val_mae": per_head_mae,
        "ensemble_val_mae": ens_mae,
        "mean_indist_disagreement": mean_disagreement,
        "num_heads": ens_cfg.num_heads,
    }
    return ensemble, history


def save_reward_ensemble(
    path: str,
    ensemble: RewardEnsemble,
    history: dict[str, object] | None = None,
) -> None:
    """Persist a trained ensemble (and optional history) to ``path``."""
    save_checkpoint(path, ensemble, extra={"history": history} if history else None)


def load_reward_ensemble(
    cfg: Config,
    path: str,
    num_actions: int,
    map_location: str | torch.device = "cpu",
) -> RewardEnsemble:
    """Rebuild and load a trained ensemble matching ``cfg``."""
    ensemble = RewardEnsemble.from_config(
        cfg.model, cfg.env, num_actions, num_heads=cfg.ensemble.num_heads
    )
    load_checkpoint(path, ensemble, map_location=map_location)
    ensemble.eval()
    return ensemble
