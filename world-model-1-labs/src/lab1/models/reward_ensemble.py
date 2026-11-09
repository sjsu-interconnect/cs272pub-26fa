"""Reward ensemble: an imperfect error-warning proxy for optional investigation.

More candidate search need not reliably improve actual return. Reward-model
exploitation is one diagnostic hypothesis, not a unique causal explanation.
Different reward-head initializations and data subsamples provide a measurable
disagreement signal whose association with error must be inspected.

Ensemble disagreement is an epistemic error-warning proxy, not calibrated
uncertainty, failure probability, or a safety guarantee. Shared errors can
produce confident agreement; disagreement is not a definitive OOD detector.

Crucially, the ensemble operates on **latents**, so the Lab 1 encoder and
dynamics stay completely frozen -- only these small heads are trained.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from lab1.config import EnvConfig, ModelConfig
from lab1.models.reward import RewardHead


class RewardEnsemble(nn.Module):
    """A bag of independent reward heads sharing a frozen latent space.

    Each member is an ordinary :class:`~lab1.models.reward.RewardHead`. Because
    the members differ only in initialization and in the bootstrap sample they
    were trained on, their predictions can differ. The per-input standard
    deviation across members is an epistemic error-warning proxy, not calibrated
    uncertainty. Its usefulness on unfamiliar latents is an empirical question.

    Parameters
    ----------
    latent_dim:
        Dimensionality of the latent vector (matches the Lab 1 encoder).
    num_actions:
        Size of the discrete action space (actions are passed as one-hot).
    num_heads:
        Number of ensemble members ``K``.
    hidden:
        Hidden layer widths of each head's MLP.
    """

    def __init__(
        self,
        latent_dim: int,
        num_actions: int,
        num_heads: int = 5,
        hidden: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        if num_heads < 2:
            raise ValueError(
                f"An ensemble needs at least 2 heads to express disagreement "
                f"(got num_heads={num_heads})."
            )
        self.num_actions = num_actions
        self.num_heads = num_heads
        self.heads = nn.ModuleList(
            RewardHead(latent_dim, num_actions, hidden) for _ in range(num_heads)
        )

    @classmethod
    def from_config(
        cls,
        model_cfg: ModelConfig,
        env_cfg: EnvConfig,
        num_actions: int,
        num_heads: int = 5,
    ) -> RewardEnsemble:
        """Build an ensemble whose geometry matches a Lab 1 world model."""
        return cls(
            latent_dim=model_cfg.latent_dim,
            num_actions=num_actions,
            num_heads=num_heads,
            hidden=tuple(model_cfg.dynamics_hidden),
        )

    def _onehot(self, action: torch.Tensor) -> torch.Tensor:
        return F.one_hot(action.long(), num_classes=self.num_actions).float()

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return every member's reward prediction, shape ``(K, B)``."""
        onehot = self._onehot(action)
        return torch.stack([head(latent, onehot) for head in self.heads], dim=0)

    def predict(
        self, latent: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mean, disagreement)`` over the ensemble, each shape ``(B,)``.

        ``mean`` is the ensemble's best guess at the reward; ``disagreement`` is
        the standard deviation across members, an uncalibrated warning proxy.
        Shared errors can produce agreement even on unfamiliar inputs.
        """
        preds = self.forward(latent, action)  # (K, B)
        mean = preds.mean(dim=0)
        disagreement = preds.std(dim=0)
        return mean, disagreement
