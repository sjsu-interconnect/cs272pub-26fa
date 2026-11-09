"""Dynamics model: (latent, action) -> next latent.

A simple MLP -- no recurrence, no transformer, no stochastic latent. Given the
current latent and a one-hot action it predicts the next latent vector. This is
the "transition model" at the heart of model-based RL, lifted into latent space.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class DynamicsModel(nn.Module):
    """Predict the next latent from the current latent and action.

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
        layers.append(nn.Linear(prev, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self, latent: torch.Tensor, action_onehot: torch.Tensor
    ) -> torch.Tensor:
        """Predict the next latent ``(B, latent)`` from latent + one-hot action."""
        x = torch.cat([latent, action_onehot], dim=-1)
        return self.net(x)
