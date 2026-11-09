"""Reward head: (latent, action) -> predicted scalar reward.

This small MLP is included **only** so the random-shooting planner can score
imagined action sequences without touching the simulator. It is not part of the
core "learn a latent transition model" story and can be disabled via
``model.use_reward_head=false`` (in which case planning is unavailable).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class RewardHead(nn.Module):
    """Predict the scalar reward obtained from a latent and action.

    Parameters
    ----------
    latent_dim:
        Dimensionality of the latent vector.
    num_actions:
        Size of the discrete action space (actions are passed as one-hot).
    hidden:
        Hidden layer widths of the MLP.
    """

    def __init__(
        self,
        latent_dim: int,
        num_actions: int,
        hidden: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        layers: list[nn.Module] = []
        prev = latent_dim + num_actions
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.ReLU(inplace=True)]
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self, latent: torch.Tensor, action_onehot: torch.Tensor
    ) -> torch.Tensor:
        """Predict reward ``(B,)`` from latent + one-hot action."""
        x = torch.cat([latent, action_onehot], dim=-1)
        return self.net(x).squeeze(-1)
