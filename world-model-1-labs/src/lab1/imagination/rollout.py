"""Roll the world model forward without querying the simulator.

Given a real episode we encode a single starting observation and then predict
every subsequent latent using only the dynamics model and the episode's action
sequence. Comparing these *imagined* latents to the latents obtained by
encoding the *real* future observations shows how prediction error changes with
the rollout horizon -- the central lesson of the lab.
"""

from __future__ import annotations

import numpy as np
import torch

from lab1.data.dataset import Episode
from lab1.models import WorldModel


def _obs_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert a ``uint8`` observation array to a normalized float tensor."""
    return torch.from_numpy(arr.astype(np.float32) / 255.0).to(device)


@torch.no_grad()
def imagine_from_episode(
    model: WorldModel,
    episode: Episode,
    start: int,
    horizon: int,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Imagine ``horizon`` steps from ``episode`` starting at index ``start``.

    Returns a dict with:

    * ``imagined_latents`` ``(horizon+1, latent)`` -- model rollout
    * ``real_latents``     ``(horizon+1, latent)`` -- ``encode`` of real frames
    * ``latent_error``     ``(horizon+1,)`` -- per-step MSE between the two
    * ``imagined_obs`` / ``real_obs`` ``(horizon+1, C, H, W)`` -- decoded vs
      real frames (only if the model has a decoder)
    """
    device = torch.device(device)
    model = model.to(device).eval()

    actions = torch.from_numpy(
        episode.actions[start : start + horizon].astype(np.int64)
    ).to(device)
    obs0 = _obs_tensor(episode.obs[start], device).unsqueeze(0)

    out = model.imagine(obs0, actions.unsqueeze(0))
    imagined = out["latents"][0]  # (horizon+1, latent)

    real_frames = _obs_tensor(
        episode.obs[start : start + horizon + 1], device
    )  # (horizon+1, C, H, W)
    real_latents = model.encode(real_frames)

    latent_error = ((imagined - real_latents) ** 2).mean(dim=1)

    result = {
        "imagined_latents": imagined,
        "real_latents": real_latents,
        "latent_error": latent_error,
        "real_obs": real_frames,
    }
    if model.decoder is not None:
        result["imagined_obs"] = model.decode(imagined)
    return result


@torch.no_grad()
def prediction_error_vs_horizon(
    model: WorldModel,
    episodes: list[Episode],
    horizon: int,
    num_samples: int = 64,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> dict[str, np.ndarray]:
    """Average latent prediction error at each rollout step.

    Samples ``num_samples`` start points (across episodes long enough to
    support ``horizon``) and averages the per-step latent MSE.

    Returns
    -------
    dict
        ``mean`` and ``std`` arrays of shape ``(horizon+1,)`` indexed by step
        (step 0 is the encoding error, which is ~0).
    """
    device = torch.device(device)
    rng = np.random.default_rng(seed)

    candidates = [
        (ep_i, s) for ep_i, ep in enumerate(episodes) for s in range(len(ep) - horizon)
    ]
    if not candidates:
        raise ValueError(f"No episode is long enough for horizon={horizon}.")
    picks = rng.choice(
        len(candidates), size=min(num_samples, len(candidates)), replace=False
    )

    errors = []
    for idx in picks:
        ep_i, start = candidates[idx]
        out = imagine_from_episode(model, episodes[ep_i], start, horizon, device=device)
        errors.append(out["latent_error"].cpu().numpy())
    stacked = np.stack(errors)  # (num, horizon+1)
    return {"mean": stacked.mean(axis=0), "std": stacked.std(axis=0)}
