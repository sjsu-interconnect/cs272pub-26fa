"""HighwayEnv factory returning stacked grayscale observations.

The factory produces observations of shape ``(stack_size, image_size,
image_size)`` — by default ``(4, 64, 64)`` — using HighwayEnv's built-in
``GrayscaleObservation`` with frame stacking. A stack of four frames gives the
world model motion information (velocity is not recoverable from a single
frame); ``stack_size=1`` is simpler but loses that motion cue.

HighwayEnv itself is never modified: all behaviour is set through
``env.configure``.
"""

from __future__ import annotations

import warnings

import gymnasium as gym

# Importing highway_env registers the environments with gymnasium.
with warnings.catch_warnings():
    # pygame 2.6 currently emits this third-party deprecation warning at import
    # time. It is not actionable for students and otherwise appears in every
    # freshly executed notebook.
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API.*",
        category=UserWarning,
    )
    import highway_env  # noqa: F401

from lab1.config import EnvConfig

# Standard luminosity weights for RGB -> grayscale conversion.
_GRAYSCALE_WEIGHTS = [0.2989, 0.5870, 0.1140]


def make_env(cfg: EnvConfig, render_mode: str = "rgb_array") -> gym.Env:
    """Create a configured HighwayEnv instance.

    Observations come from pygame's renderer. The ``offscreen_rendering: True``
    setting (from :class:`~lab1.config.EnvConfig`) renders without opening a
    window while still producing real frames -- this is what you want on both
    desktops and servers with a display.

    On a *truly* headless machine (CI, no display at all), do **not** set
    ``SDL_VIDEODRIVER=dummy`` -- empirically that yields all-black frames.
    Instead run the process under a virtual framebuffer, e.g.::

        xvfb-run -a python scripts/collect.py --config configs/lab.yaml

    Parameters
    ----------
    cfg:
        Environment configuration (observation size, frame stack, dynamics).
    render_mode:
        Passed to :func:`gymnasium.make`. Must be ``"rgb_array"`` for frame
        rendering to work; this is required by ``GrayscaleObservation``.

    Returns
    -------
    gymnasium.Env
        An environment whose ``reset``/``step`` return observations of shape
        ``(cfg.stack_size, cfg.image_size, cfg.image_size)``. The caller is
        responsible for calling ``env.reset(seed=...)`` before use.
    """
    env = gym.make(cfg.id, render_mode=render_mode)
    env.unwrapped.configure(
        {
            "offscreen_rendering": cfg.offscreen_rendering,
            "lanes_count": cfg.lanes_count,
            "vehicles_count": cfg.vehicles_count,
            "duration": cfg.duration,
            "simulation_frequency": cfg.simulation_frequency,
            "policy_frequency": cfg.policy_frequency,
            "action": {"type": "DiscreteMetaAction"},
            "observation": {
                "type": "GrayscaleObservation",
                "observation_shape": (cfg.image_size, cfg.image_size),
                "stack_size": cfg.stack_size,
                "weights": _GRAYSCALE_WEIGHTS,
                "scaling": cfg.scaling,
            },
        }
    )
    return env


def observation_shape(cfg: EnvConfig) -> tuple[int, int, int]:
    """Return the expected observation shape ``(stack, H, W)`` for a config."""
    return (cfg.stack_size, cfg.image_size, cfg.image_size)
