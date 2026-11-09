"""CNN encoder: image observation -> latent vector.

A deliberately small convolutional network that maps a stacked grayscale
observation ``(C, H, W)`` to a ``latent_dim``-dimensional vector. Three
stride-2 convolutions reduce a 64x64 input to 8x8, which is then flattened and
projected to the latent. Students should be able to read this in a minute.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class Encoder(nn.Module):
    """Map an image observation to a latent vector.

    Parameters
    ----------
    in_channels:
        Number of input channels (equals the frame ``stack_size``).
    latent_dim:
        Dimensionality of the output latent vector.
    channels:
        Output channels of each stride-2 convolution.
    image_size:
        Spatial size of the (square) input image.
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int = 64,
        channels: Sequence[int] = (32, 64, 128),
        image_size: int = 64,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_channels
        for out in channels:
            layers += [
                nn.Conv2d(prev, out, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ]
            prev = out
        self.conv = nn.Sequential(*layers)

        reduced = image_size // (2 ** len(channels))
        self.flat_dim = prev * reduced * reduced
        self.fc = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode a batch of observations ``(B, C, H, W)`` to ``(B, latent)``."""
        h = self.conv(obs)
        h = torch.flatten(h, start_dim=1)
        return self.fc(h)
