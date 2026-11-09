"""Display-only camera for student-facing figures and animations.

The world model never sees these frames. Its input is the 64x64 grayscale
observation produced by HighwayEnv's ``GrayscaleObservation``, which owns a
separate viewer whose settings were copied from ``env.config`` when the
observation was created. This module adds a *second* pygame viewer with its own
config copy, so rendering it changes neither ``env.config`` nor the model
observation, and it does not step or reseed the simulator.

The display camera follows the ego vehicle along the road (the ego stays at the
horizontal center) and keeps the road vertically centered, so a lane change
shows up as the ego moving between lanes instead of the road sliding.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from highway_env.envs.common.graphics import EnvViewer

DISPLAY_WIDTH = 600  # px, HighwayEnv's default render size
DISPLAY_HEIGHT = 110  # px: the four-lane road plus a margin
DISPLAY_SCALING = 5.5  # px per metre, HighwayEnv's default render scale


class _RoadCenteredViewer(EnvViewer):
    """An EnvViewer centered on (ego x, middle of the road)."""

    lateral_center: float = 0.0

    def window_position(self) -> np.ndarray:
        return np.array([self.env.vehicle.position[0], self.lateral_center])


class DisplayCamera:
    """Render color display frames and the ego vehicle's pixel bounding box.

    Create it after ``env.reset`` and call :meth:`render` after each reset or
    step. Do not close it: closing a pygame viewer quits pygame for the model's
    observation viewer too. ``env.close()`` cleans up both.
    """

    def __init__(
        self,
        env: gym.Env,
        width: int = DISPLAY_WIDTH,
        height: int = DISPLAY_HEIGHT,
        scaling: float = DISPLAY_SCALING,
    ) -> None:
        self._env = env.unwrapped
        config = dict(self._env.config)  # a copy: env.config is left untouched
        config.update(
            {
                "offscreen_rendering": True,
                "screen_width": width,
                "screen_height": height,
                "scaling": scaling,
                "centering_position": [0.5, 0.5],
                "render_agent": False,
            }
        )
        self._viewer = _RoadCenteredViewer(self._env, config=config)
        lanes = self._env.road.network.lanes_list()
        self._viewer.lateral_center = float(
            np.mean([lane.position(0.0, 0.0)[1] for lane in lanes])
        )

    def render(self) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Return ``(display_frame, ego_box)`` for the current simulator state.

        ``display_frame`` is an ``(height, width, 3)`` uint8 RGB image.
        ``ego_box`` is ``(x0, y0, x1, y1)`` in display pixels: the axis-aligned
        bounds of the ego vehicle's rotated footprint.
        """
        self._viewer.display()
        frame = np.array(self._viewer.get_image(), dtype=np.uint8)
        ego = self._env.vehicle
        cos, sin = np.cos(ego.heading), np.sin(ego.heading)
        corners = [
            ego.position + np.array([cos * dx - sin * dy, sin * dx + cos * dy])
            for dx in (-ego.LENGTH / 2, ego.LENGTH / 2)
            for dy in (-ego.WIDTH / 2, ego.WIDTH / 2)
        ]
        pixels = np.array([self._viewer.sim_surface.vec2pix(c) for c in corners])
        x0, y0 = pixels.min(axis=0)
        x1, y1 = pixels.max(axis=0)
        return frame, (int(x0), int(y0), int(x1), int(y1))
