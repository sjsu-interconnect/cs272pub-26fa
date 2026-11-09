"""CNN decoder: latent vector -> reconstructed image.

The decoder supports **training and visualization**, not action selection.
It lets students compare a real observation with a reconstruction. Alongside
reward supervision, reconstruction helps anchor the representation; a
stop-gradient dynamics target alone does not prevent collapse.
The decoder can be disabled
via ``model.use_decoder=false``; see :class:`~lab1.models.world_model.WorldModel`.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class Decoder(nn.Module):
    """Map a latent vector back to an image observation.

    Mirrors :class:`~lab1.models.encoder.Encoder`: a linear projection to an
    8x8 feature map followed by stride-2 transposed convolutions up to the
    original ``image_size``. A final sigmoid keeps outputs in ``[0, 1]`` to
    match normalized observations.
    """

    def __init__(
        self,
        out_channels: int,
        latent_dim: int = 64,
        channels: Sequence[int] = (32, 64, 128),
        image_size: int = 64,
    ) -> None:
        super().__init__()
        rev = list(reversed(channels))
        self.start_size = image_size // (2 ** len(channels))
        self.start_channels = rev[0]
        self.fc = nn.Linear(
            latent_dim, self.start_channels * self.start_size * self.start_size
        )

        layers: list[nn.Module] = []
        prev = rev[0]
        for out in rev[1:]:
            layers += [
                nn.ConvTranspose2d(prev, out, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ]
            prev = out
        layers += [
            nn.ConvTranspose2d(prev, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        ]
        self.deconv = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode a batch of latents ``(B, latent)`` to ``(B, C, H, W)``."""
        h = self.fc(latent)
        h = h.view(-1, self.start_channels, self.start_size, self.start_size)
        return self.deconv(h)
