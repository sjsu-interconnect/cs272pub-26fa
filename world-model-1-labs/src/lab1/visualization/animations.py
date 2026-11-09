"""Small notebook-friendly animations for recorded driving trajectories.

The notebooks should explain *why* a trajectory is useful, not spend a full
screen on Matplotlib animation plumbing.  These helpers keep that plumbing in
one tested place and return an ``IPython.display.HTML`` object that renders
inline without ffmpeg.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

ACTION_NAMES = {
    0: "lane left",
    1: "idle",
    2: "lane right",
    3: "faster",
    4: "slower",
}


def frame_captions(
    num_observations: int,
    actions: Sequence[int] | np.ndarray | None = None,
    rewards: Sequence[float] | np.ndarray | None = None,
    dones: Sequence[bool] | np.ndarray | None = None,
) -> list[str]:
    """Caption for each animation frame, aligned with the transitions.

    A trajectory with ``N`` transitions has ``N + 1`` observations: observation
    0 comes *before* any action, and observation ``k`` (``k >= 1``) is the
    result of transition ``k - 1``. Each frame is therefore labeled with the
    action, reward and done flag that *produced* it. If the metadata has the
    same length as the observations, frame ``k`` shows metadata ``k``.
    """
    metadata = [m for m in (actions, rewards, dones) if m is not None]
    lengths = {len(m) for m in metadata}
    if len(lengths) > 1:
        raise ValueError("actions, rewards and dones must have equal lengths.")
    transitions = lengths.pop() if lengths else 0
    after_action = bool(metadata) and num_observations == transitions + 1
    if metadata and not after_action and num_observations != transitions:
        raise ValueError(
            "Expected one observation per transition, or one extra initial "
            f"observation; got {num_observations} observations and "
            f"{transitions} transitions."
        )

    def describe(index: int) -> str:
        parts = []
        if actions is not None:
            action_id = int(actions[index])
            parts.append(f"executed action = {ACTION_NAMES.get(action_id, action_id)}")
        if rewards is not None:
            parts.append(f"received reward = {float(rewards[index]):.3f}")
        if dones is not None:
            parts.append(f"done = {bool(dones[index])}")
        return "\n".join(parts)

    captions = []
    for frame in range(num_observations):
        if after_action and frame == 0:
            captions.append("Initial observation\nbefore any action")
        elif after_action:
            captions.append(
                f"After real step {frame} / {transitions}\n{describe(frame - 1)}"
            )
        elif metadata:
            captions.append(f"Real step {frame}\n{describe(frame)}")
        else:
            captions.append(f"Observation {frame} / {num_observations - 1}")
    return captions


def _inline_animation(
    observations: Sequence[np.ndarray] | np.ndarray,
    *,
    actions: Sequence[int] | np.ndarray | None = None,
    rewards: Sequence[float] | np.ndarray | None = None,
    dones: Sequence[bool] | np.ndarray | None = None,
    title: str = "Trajectory",
    interval: int = 450,
) -> Any:
    """Render stacked grayscale observations as a compact inline animation."""
    from IPython.display import HTML

    observations = np.asarray(observations)
    if len(observations) == 0:
        raise ValueError("Animation requires at least one observation.")
    captions = frame_captions(len(observations), actions, rewards, dones)

    # The source images are only 64x64, so a lower-DPI canvas keeps embedded
    # JSHTML outputs substantially smaller while remaining clear in Jupyter.
    # Keep the player visually subordinate to the six-frame overview above it.
    fig, ax = plt.subplots(figsize=(4.6, 4.8), dpi=80, layout="constrained")
    image = ax.imshow(
        observations[0, -1],
        cmap="gray",
        vmin=0,
        vmax=255,
        interpolation="nearest",
    )
    heading = ax.set_title(f"{title}\n{captions[0]}", fontsize=9, loc="left")
    ax.axis("off")

    def update(frame_index: int):
        image.set_data(observations[frame_index, -1])
        heading.set_text(f"{title}\n{captions[frame_index]}")
        return image, heading

    animation = FuncAnimation(
        fig, update, frames=len(observations), interval=interval, blit=False
    )
    plt.close(fig)
    return HTML(animation.to_jshtml(default_mode="once"))


def animate_episode(episode, *, title: str = "Collected episode", interval: int = 450):
    """Animate a dataset ``Episode`` with action, reward, and done metadata."""
    return _inline_animation(
        episode.obs,
        actions=episode.actions,
        rewards=episode.rewards,
        dones=episode.dones,
        title=title,
        interval=interval,
    )


def animate_policy_run(
    run: dict[str, object], *, title: str = "Planner episode", interval: int = 450
):
    """Animate the output of :func:`lab1.planning.rollout_with_policy`."""
    return _inline_animation(
        run["frames"],
        actions=run.get("actions"),
        rewards=run.get("rewards"),
        dones=run.get("dones"),
        title=title,
        interval=interval,
    )
