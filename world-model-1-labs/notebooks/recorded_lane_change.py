"""Show a real lane-right transition from saved experience, never a prediction."""

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt

from lab1.data import load_episodes

DATASET_SHA256 = "270d59559ede9702846393b543d03e04ad7c6256b872458669afa276ff68beb8"
EPISODE_INDEX = 3
ENVIRONMENT_SEED = 3
STEP = 6


def _select_lane_right_transition(dataset_path):
    """Keep the verified classroom example, or find one in regenerated data."""
    path = Path(dataset_path)
    episodes = load_episodes(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() == DATASET_SHA256:
        episode = episodes[EPISODE_INDEX]
        if int(episode.actions[STEP]) != 2:
            raise ValueError("The saved action does not match the verified example.")
        return episode, EPISODE_INDEX, STEP, True

    for episode_index, episode in enumerate(episodes):
        for step, action in enumerate(episode.actions):
            if int(action) == 2 and step + 1 < len(episode.obs):
                return episode, episode_index, step, False
    raise ValueError("This dataset contains no saved lane-right transition to display.")


def plot_recorded_lane_change(dataset_path):
    episode, episode_index, step, verified_classroom_example = (
        _select_lane_right_transition(dataset_path)
    )
    if verified_classroom_example:
        titles = (
            "Before action: $o_6$ — observed lane 2",
            "Immediately after: $o_7$ — observed lane 3",
        )
        provenance = "trajectory 3 · replay seed 3 · step 6 → 7"
    else:
        titles = (
            f"Before action: $o_{{{step}}}$",
            f"Immediately after: $o_{{{step + 1}}}$",
        )
        provenance = (
            f"generated dataset · trajectory {episode_index} · "
            f"step {step} → {step + 1}"
        )
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.6))
    for ax, index, title in [
        (axes[0], step, titles[0]),
        (axes[1], step + 1, titles[1]),
    ]:
        ax.imshow(
            episode.obs[index, -1][::-1],  # display only: the ego car drives up
            cmap="gray",
            vmin=0,
            vmax=255,
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Real saved observation (newest frame)", fontsize=10)
    fig.suptitle(
        "Observed Effect of a Lane Change\nRecorded action: lane right",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.02,
        f"{Path(dataset_path).name} · {provenance}\n"
        "One real simulator step (1 s); a requested lane change need not succeed.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.92))
    return fig
