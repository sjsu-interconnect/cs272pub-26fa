"""The world model: encoder + dynamics (+ optional decoder, reward head).

This module ties the components together and defines the training objective.

Training signals
----------------
* **dynamics loss** -- MSE between the predicted next latent and a *stop-gradient*
  target ``encoder(next_obs).detach()``. The stop-gradient stops the encoder
  target branch from immediately moving to match the prediction; it does not
  by itself prevent collapse. A constant encoder remains a degenerate solution.
* **reconstruction loss** (optional) -- MSE between ``decoder(encode(obs))`` and
  ``obs``. Helps anchor the representation, alongside reward supervision.
* **reward loss** (optional) -- MSE between the predicted and observed reward.

The encoder receives gradients from the reconstruction loss, the *prediction*
side of the dynamics loss, and the reward loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from lab1.config import EnvConfig, ModelConfig
from lab1.models.decoder import Decoder
from lab1.models.dynamics import DynamicsModel
from lab1.models.encoder import Encoder
from lab1.models.reward import RewardHead


class WorldModel(nn.Module):
    """Deterministic latent world model.

    Parameters
    ----------
    in_channels:
        Number of observation channels (frame ``stack_size``).
    num_actions:
        Size of the discrete action space.
    latent_dim:
        Latent dimensionality.
    channels:
        Encoder/decoder convolution channels.
    dynamics_hidden:
        Hidden widths for the dynamics (and reward) MLPs.
    image_size:
        Spatial size of the (square) observation.
    use_decoder:
        If ``True``, build a decoder and add a reconstruction loss.
    use_reward_head:
        If ``True``, build a reward head and add a reward loss.
    """

    def __init__(
        self,
        in_channels: int,
        num_actions: int,
        latent_dim: int = 64,
        channels: tuple[int, ...] = (32, 64, 128),
        dynamics_hidden: tuple[int, ...] = (256, 256),
        image_size: int = 64,
        use_decoder: bool = True,
        use_reward_head: bool = True,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.latent_dim = latent_dim
        self.use_decoder = use_decoder
        self.use_reward_head = use_reward_head

        self.encoder = Encoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            channels=channels,
            image_size=image_size,
        )
        self.dynamics = DynamicsModel(
            latent_dim=latent_dim,
            num_actions=num_actions,
            hidden=dynamics_hidden,
        )
        self.decoder = (
            Decoder(
                out_channels=in_channels,
                latent_dim=latent_dim,
                channels=channels,
                image_size=image_size,
            )
            if use_decoder
            else None
        )
        self.reward_head = (
            RewardHead(
                latent_dim=latent_dim,
                num_actions=num_actions,
                hidden=dynamics_hidden,
            )
            if use_reward_head
            else None
        )

    # -- construction ----------------------------------------------------

    @classmethod
    def from_config(
        cls,
        model_cfg: ModelConfig,
        env_cfg: EnvConfig,
        num_actions: int,
    ) -> WorldModel:
        """Build a :class:`WorldModel` from config objects."""
        return cls(
            in_channels=env_cfg.stack_size,
            num_actions=num_actions,
            latent_dim=model_cfg.latent_dim,
            channels=tuple(model_cfg.encoder_channels),
            dynamics_hidden=tuple(model_cfg.dynamics_hidden),
            image_size=env_cfg.image_size,
            use_decoder=model_cfg.use_decoder,
            use_reward_head=model_cfg.use_reward_head,
        )

    # -- primitives ------------------------------------------------------

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observations ``(B, C, H, W)`` to latents ``(B, latent)``."""
        return self.encoder(obs)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latents to images. Requires ``use_decoder=True``."""
        if self.decoder is None:
            raise RuntimeError("Decoder is disabled (use_decoder=False).")
        return self.decoder(latent)

    def _onehot(self, action: torch.Tensor) -> torch.Tensor:
        return F.one_hot(action.long(), num_classes=self.num_actions).float()

    def predict_next(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict the next latent given a latent and integer action(s)."""
        return self.dynamics(latent, self._onehot(action))

    def predict_reward(
        self, latent: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Predict reward given a latent and integer action(s)."""
        if self.reward_head is None:
            raise RuntimeError("Reward head is disabled (use_reward_head=False).")
        return self.reward_head(latent, self._onehot(action))

    # -- rollout ---------------------------------------------------------

    @torch.no_grad()
    def imagine(
        self, obs0: torch.Tensor, actions: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Roll the dynamics forward from an initial observation.

        The simulator is queried only to provide ``obs0``; every subsequent
        latent is predicted by the model.

        Parameters
        ----------
        obs0:
            Initial observations ``(B, C, H, W)``.
        actions:
            Integer actions ``(B, T)`` to apply.

        Returns
        -------
        dict
            ``latents`` ``(B, T+1, latent)`` and, if a reward head exists,
            ``rewards`` ``(B, T)``.
        """
        z = self.encode(obs0)
        latents = [z]
        rewards = []
        horizon = actions.shape[1]
        for t in range(horizon):
            a = actions[:, t]
            if self.reward_head is not None:
                rewards.append(self.predict_reward(z, a))
            z = self.predict_next(z, a)
            latents.append(z)
        out = {"latents": torch.stack(latents, dim=1)}
        if rewards:
            out["rewards"] = torch.stack(rewards, dim=1)
        return out

    # -- training objective ---------------------------------------------

    def compute_losses(
        self,
        batch: dict[str, torch.Tensor],
        recon_weight: float = 1.0,
        reward_weight: float = 1.0,
        dynamics_weight: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Compute the (weighted) training losses for one batch.

        Returns a dict with ``total`` and each individual loss term (for
        logging). Disabled components contribute zero.
        """
        obs = batch["obs"]
        next_obs = batch["next_obs"]
        action = batch["action"]

        z = self.encode(obs)
        z_next_pred = self.predict_next(z, action)
        # Stop-gradient target: the encoder must not move the target to meet
        # its own prediction.
        z_next_target = self.encode(next_obs).detach()
        dynamics_loss = F.mse_loss(z_next_pred, z_next_target)

        zero = obs.new_zeros(())
        recon_loss = zero
        if self.decoder is not None:
            recon = self.decoder(z)
            recon_loss = F.mse_loss(recon, obs)

        reward_loss = zero
        if self.reward_head is not None:
            reward_pred = self.predict_reward(z, action)
            reward_loss = F.mse_loss(reward_pred, batch["reward"])

        total = (
            dynamics_weight * dynamics_loss
            + recon_weight * recon_loss
            + reward_weight * reward_loss
        )
        return {
            "total": total,
            "dynamics": dynamics_loss.detach(),
            "recon": recon_loss.detach(),
            "reward": reward_loss.detach(),
        }
