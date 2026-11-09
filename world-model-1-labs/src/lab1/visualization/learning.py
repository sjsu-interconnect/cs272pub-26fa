"""Figures and metrics for Lab 1: how the world model is learned.

The story is one transition, one diagram, one update, and a before/after
comparison of the world model's own predictions on held-out transitions.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from lab1.data.dataset import Episode, load_episodes
from lab1.models import WorldModel
from lab1.visualization.animations import ACTION_NAMES
from lab1.visualization.story import (
    ERROR,
    EXECUTED,
    PREDICTED,
    PREDICTED_LIGHT,
    RECORDED,
    RECORDED_LIGHT,
    REFERENCE,
    TAIL_TEXT,
    _arrow,
    _box,
    _save,
)

LOW_REWARD = 0.3  # below this, a transition usually involved a collision
QUIET = {"fc": "#f4f4f4", "ec": "#a0a0a0", "color": TAIL_TEXT}


def plot_experience_to_world_model(
    observation: np.ndarray, path: str | Path | None = None
) -> Figure:
    """Opening overview: experience, learning E, f and R, then frozen inference.

    ``observation`` is one real grayscale frame (64 x 64) from the dataset.
    """
    fig, ax = plt.subplots(figsize=(13, 5.0))
    ax.set_xlim(0, 17)
    ax.set_ylim(-0.1, 6.4)
    ax.set_aspect("equal")
    ax.axis("off")
    blue = {"fc": RECORDED_LIGHT, "ec": RECORDED}
    orange = {"fc": PREDICTED_LIGHT, "ec": PREDICTED}
    plain = {"fc": "white", "ec": "#444444"}

    for x, text, color in [
        (1.9, "Experience", RECORDED),
        (8.25, "Learning from saved transitions", "#444444"),
        (14.95, "Inference", EXECUTED),
    ]:
        ax.text(x, 5.95, text, ha="center", fontsize=12, fontweight="bold", color=color)

    # 1. Experience: one real observation and the transition tuple.
    ax.imshow(
        observation,
        cmap="gray",
        vmin=0,
        vmax=255,
        interpolation="nearest",
        extent=(0.4, 3.4, 1.9, 4.9),
        zorder=2,
    )
    ax.add_patch(
        plt.Rectangle((0.4, 1.9), 3.0, 3.0, fill=False, ec=RECORDED, lw=2.5, zorder=3)
    )
    ax.text(1.9, 5.2, "real HighwayEnv observation $o_t$", ha="center", fontsize=10)
    ax.text(1.9, 1.35, "$(o_t,\\ a_t,\\ r_{t+1},\\ o_{t+1})$", ha="center", fontsize=12)
    ax.text(
        1.9,
        0.7,
        "HighwayEnv produces\nexperience",
        ha="center",
        va="center",
        fontsize=10,
    )
    _arrow(ax, (3.6, 3.4), (4.75, 3.4), lw=2)
    ax.text(4.17, 3.6, "learn", ha="center", fontsize=10)

    # 2. Learning: E, f and R, with predictions of the next latent and reward.
    _box(ax, 4.8, 2.9, 1.0, 1.0, "$E$", weight="bold", **plain)
    _box(ax, 6.2, 2.9, 1.8, 1.0, "learned state\n$z_t=E(o_t)$", **blue)
    _box(ax, 6.2, 1.15, 1.8, 0.85, "action $a_t$", **blue)
    _box(ax, 8.6, 3.3, 1.5, 0.9, "$f(z_t, a_t)$", **plain)
    _box(ax, 8.6, 1.9, 1.5, 0.9, "$R(z_t, a_t)$", **plain)
    _box(ax, 10.6, 3.3, 1.2, 0.9, "$\\hat z_{t+1}$", **orange)
    _box(ax, 10.6, 1.9, 1.2, 0.9, "$\\hat r_{t+1}$", **orange)
    _arrow(ax, (5.82, 3.4), (6.18, 3.4), lw=1.4)
    # z_t and a_t join one input line that feeds both heads.
    ax.plot([8.02, 8.3], [3.4, 3.4], color="#444444", lw=1.4)
    ax.plot([8.02, 8.3], [1.575, 1.575], color="#444444", lw=1.4)
    ax.plot([8.3, 8.3], [1.575, 3.75], color="#444444", lw=1.4)
    for start, end in [
        ((8.3, 3.75), (8.58, 3.75)),
        ((8.3, 2.35), (8.58, 2.35)),
        ((10.12, 3.75), (10.58, 3.75)),
        ((10.12, 2.35), (10.58, 2.35)),
    ]:
        _arrow(ax, start, end, lw=1.4)
    ax.plot([7.1, 7.1], [3.92, 4.45], ls="--", color="#a0a0a0", lw=1.3)
    _box(ax, 6.2, 4.45, 1.8, 0.7, "decoder $D$", **QUIET)
    ax.text(
        8.15,
        4.8,
        "auxiliary image loss,\ntraining only",
        fontsize=9,
        va="center",
        color=TAIL_TEXT,
    )
    ax.text(
        8.3,
        0.55,
        "predict the next latent state and reward;\n"
        "real next observations and rewards are the targets",
        ha="center",
        va="center",
        fontsize=10,
    )

    # 3. Inference: the frozen model is used for imagination and MPC.
    _arrow(ax, (12.0, 3.05), (12.95, 3.05), lw=2, color=EXECUTED)
    ax.text(12.47, 3.25, "freeze", ha="center", fontsize=10, color=EXECUTED)
    _box(
        ax,
        13.0,
        1.4,
        3.9,
        3.3,
        "Lab 2\n\nfrozen $E$, $f$, $R$\nimagine latent futures\n"
        "and rewards\n\nMPC scores candidate\naction sequences",
        fc="white",
        ec=EXECUTED,
        size=10,
    )
    ax.set_title(
        "From Experience to a Learned World Model", fontsize=14, fontweight="bold"
    )
    fig.tight_layout()
    return _save(fig, path)


def plot_learning_to_imagination(path: str | Path | None = None) -> Figure:
    """Closing bridge: the trained model is frozen and used for imagination."""
    steps = [
        ("trained world model: $E$, $f$, $R$", "white", "#444444", "black"),
        (
            "imagined latent futures $\\hat z$ and rewards $\\hat r$",
            PREDICTED_LIGHT,
            PREDICTED,
            "black",
        ),
        ("Lab 2: MPC", "white", EXECUTED, EXECUTED),
    ]
    labels = ["freeze parameters", "score candidate action sequences"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.5, 6.0)
    ax.axis("off")
    tops = [5.0, 2.85, 0.7]
    for (text, fc, ec, color), y in zip(steps, tops, strict=True):
        _box(ax, 1.5, y, 7.0, 0.85, text, fc=fc, ec=ec, color=color, size=11)
    for i, label in enumerate(labels):
        _arrow(ax, (5.0, tops[i] - 0.02), (5.0, tops[i + 1] + 0.87), lw=1.8)
        ax.text(
            5.25, (tops[i] + tops[i + 1] + 0.85) / 2, label, va="center", fontsize=10
        )
    ax.set_title("From learning to imagination", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def _gray(ax, image: np.ndarray, title: str, color: str, width: float = 2.5) -> None:
    ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(width)
    ax.set_title(title, fontsize=10, color=color)


def plot_saved_transition(
    episode: Episode, t: int, path: str | Path | None = None
) -> Figure:
    """One saved transition: the model input and the two real targets."""
    action = ACTION_NAMES.get(int(episode.actions[t]), str(episode.actions[t]))
    reward = float(episode.rewards[t])
    fig = plt.figure(figsize=(10, 4.4), layout="constrained")
    grid = fig.add_gridspec(2, 7, height_ratios=[3.2, 1.0])
    _gray(
        fig.add_subplot(grid[0, 0:2]),
        episode.obs[t, -1],
        "$o_t$: current observation\n(newest of four frames)",
        RECORDED,
    )
    middle = fig.add_subplot(grid[0, 2:5])
    middle.set_xlim(0, 3)
    middle.set_ylim(0, 3)
    middle.axis("off")
    _box(
        middle,
        0.25,
        1.75,
        2.5,
        0.85,
        f"$a_t$ = {action}\nsaved action (random data policy)",
        fc=RECORDED_LIGHT,
        ec=RECORDED,
        size=10,
    )
    _arrow(middle, (0.1, 1.45), (2.9, 1.45), color="#444444", lw=2)
    middle.text(1.5, 1.15, "HighwayEnv steps once", ha="center", fontsize=10)
    _box(
        middle,
        0.25,
        0.1,
        2.5,
        0.75,
        f"$r_{{t+1}}$ = {reward:.3f}\nreward from HighwayEnv",
        fc="white",
        ec=RECORDED,
        size=10,
    )
    _gray(
        fig.add_subplot(grid[0, 5:7]),
        episode.obs[t + 1, -1],
        "$o_{t+1}$: next observation\n(newest of four frames)",
        RECORDED,
    )
    for k in range(4):
        ax = fig.add_subplot(grid[1, k])
        _gray(ax, episode.obs[t, k], f"frame {k + 1} of $o_t$", REFERENCE, width=1)
    note = fig.add_subplot(grid[1, 4:7])
    note.axis("off")
    note.text(
        0.02,
        0.5,
        "Model input: $o_t$ and $a_t$\nReal targets: $r_{t+1}$ and $o_{t+1}$",
        fontsize=11,
        va="center",
    )
    fig.suptitle(
        f"One saved transition (step {t} of a saved trajectory)",
        fontsize=12,
        fontweight="bold",
    )
    return _save(fig, path)


def plot_learning_diagram(path: str | Path | None = None) -> Figure:
    """What is predicted, what supervises it, and the auxiliary decoder."""
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 16)
    ax.set_ylim(-0.6, 6.7)
    ax.axis("off")
    blue = {"fc": RECORDED_LIGHT, "ec": RECORDED}
    orange = {"fc": PREDICTED_LIGHT, "ec": PREDICTED}
    plain = {"fc": "white", "ec": "#444444"}
    quiet = {"fc": "#f4f4f4", "ec": "#a0a0a0", "color": TAIL_TEXT}

    _box(ax, 0.2, 3.6, 2.4, 1.0, "$o_t$\nobservation", **blue)
    _box(ax, 3.1, 3.6, 1.0, 1.0, "$E$", weight="bold", **plain)
    _box(ax, 4.6, 3.6, 1.6, 1.0, "$z_t$\nlearned state", **blue)
    _box(ax, 4.6, 1.4, 1.6, 1.0, "$a_t$\naction", **blue)
    _box(
        ax,
        6.9,
        1.4,
        2.4,
        3.2,
        "$M_\\phi$\n\ndynamics $f$\n\nreward head $R$",
        weight="bold",
        **plain,
    )
    _box(ax, 10.0, 3.6, 1.8, 1.0, "$\\hat z_{t+1}$\npredicted", **orange)
    _box(ax, 10.0, 1.4, 1.8, 1.0, "$\\hat r_{t+1}$\npredicted", **orange)
    _box(ax, 13.4, 3.6, 2.4, 1.0, "$z_{t+1}=E(o_{t+1})$\nfixed target", **blue)
    _box(ax, 13.4, 1.4, 2.4, 1.0, "$r_{t+1}$\nreceived reward", **blue)
    _box(ax, 13.4, 5.5, 2.4, 1.0, "$o_{t+1}$\nnext observation", **blue)
    for start, end in [
        ((2.62, 4.1), (3.08, 4.1)),
        ((4.12, 4.1), (4.58, 4.1)),
        ((6.22, 4.1), (6.88, 4.1)),
        ((6.22, 1.9), (6.88, 1.9)),
        ((9.32, 4.1), (9.98, 4.1)),
        ((9.32, 1.9), (9.98, 1.9)),
        ((14.6, 5.48), (14.6, 4.62)),
    ]:
        _arrow(ax, start, end)
    ax.text(14.75, 5.0, "$E$, held fixed", fontsize=10)
    for y, label in [
        (4.1, "$\\mathcal{L}_{\\mathrm{next\\ latent}}$"),
        (1.9, "$\\mathcal{L}_{\\mathrm{reward}}$"),
    ]:
        _arrow(ax, (11.85, y), (13.35, y), color=ERROR, style="<|-|>", lw=1.6)
        ax.text(12.6, y + 0.18, label, ha="center", fontsize=10, color=ERROR)

    # Auxiliary decoder: helps train E, never used by imagination or MPC.
    ax.plot([5.4, 5.4], [4.62, 5.5], ls="--", color="#a0a0a0", lw=1.4)
    _box(ax, 4.75, 5.5, 1.3, 0.8, "$D$", **quiet)
    ax.plot([4.73, 2.62], [5.9, 5.9], ls="--", color="#a0a0a0", lw=1.4)
    _box(ax, 0.2, 5.5, 2.4, 0.8, "$\\hat o_t$ reconstruction", **quiet)
    ax.plot([1.4, 1.4], [5.48, 4.62], ls="--", color="#a0a0a0", lw=1.4)
    ax.text(
        1.55,
        5.02,
        "$\\mathcal{L}_{\\mathrm{aux\\ image}}$",
        fontsize=10,
        color=TAIL_TEXT,
    )
    ax.text(
        6.3,
        6.0,
        "auxiliary, training only",
        fontsize=10,
        color=TAIL_TEXT,
        va="center",
    )
    ax.text(
        0.2,
        0.55,
        "Orange: predictions.  Blue: real data from the saved transition.  "
        "Red: losses compare prediction with target.",
        fontsize=10,
    )
    ax.text(
        0.2,
        -0.15,
        "Imagination and MPC use $E$, $f$ and $R$. "
        "They never use reconstructed images.",
        fontsize=10,
        color=TAIL_TEXT,
    )
    ax.set_title("How the world model learns from one transition", fontsize=12)
    fig.tight_layout()
    return _save(fig, path)


def _transition_keys(episodes: Sequence[Episode]) -> list[tuple[str, int]]:
    return [
        (hashlib.sha1(ep.obs[t].tobytes()).hexdigest(), int(ep.actions[t]))
        for ep in episodes
        for t in range(len(ep))
    ]


@torch.no_grad()
def prediction_metrics(
    model: WorldModel,
    episodes: Sequence[Episode],
    device: torch.device | str = "cpu",
    exclude_dataset: str | Path | None = None,
) -> dict[str, Any]:
    """One-step predictions of a world model on fixed held-out transitions.

    ``latent_mse`` is the mean squared error between the predicted next latent
    and the encoded real next observation, in the model's own latent space.
    ``latent_error`` divides it by the variance of those target latents, so 1.0
    means no better than always predicting their average latent. This makes
    models with different latent scales comparable.

    ``exclude_dataset`` drops transitions that also occur in another dataset,
    for example the training data of a reference model.
    """
    model = model.to(device).eval()
    keep = np.ones(sum(len(ep) for ep in episodes), dtype=bool)
    if exclude_dataset is not None:
        seen = set(_transition_keys(load_episodes(exclude_dataset)))
        keep = np.array([key not in seen for key in _transition_keys(episodes)])
    predicted, targets, rewards, predicted_rewards = [], [], [], []
    for ep in episodes:
        obs = torch.from_numpy(ep.obs.astype(np.float32) / 255.0).to(device)
        action = torch.from_numpy(ep.actions.astype(np.int64)).to(device)
        z = model.encode(obs[:-1])
        predicted.append(model.predict_next(z, action))
        targets.append(model.encode(obs[1:]))
        predicted_rewards.append(model.predict_reward(z, action))
        rewards.append(torch.from_numpy(ep.rewards.astype(np.float32)))
    mask = torch.from_numpy(keep)
    z_pred = torch.cat(predicted).cpu()[mask]
    z_true = torch.cat(targets).cpu()[mask]
    mse = float(((z_pred - z_true) ** 2).mean())
    variance = float(z_true.var(dim=0).mean())
    reward = torch.cat(rewards).numpy()[keep]
    reward_pred = torch.cat(predicted_rewards).cpu().numpy()[keep]
    return {
        "transitions": int(keep.sum()),
        "latent_mse": mse,
        "latent_error": mse / variance,
        "reward": reward,
        "reward_pred": reward_pred,
        "reward_mae": float(np.abs(reward_pred - reward).mean()),
    }


def plot_latent_before_after(
    before: dict[str, Any], after: dict[str, Any], path: str | Path | None = None
) -> Figure:
    """Normalized one-step next-latent error before and after training."""
    values = [before["latent_error"], after["latent_error"]]
    labels = ["random weights\n(before training)", "after training"]
    fig, ax = plt.subplots(figsize=(8, 3.3), layout="constrained")
    ax.barh([1, 0], values, color=[REFERENCE, PREDICTED], height=0.55)
    ax.set_xscale("log")
    ax.set_xlim(min(values) / 4, max(values) * 6)
    ax.set_yticks([1, 0], labels)
    for y, value in zip([1, 0], values, strict=True):
        ax.text(value * 1.15, y, f"{value:.3g}", va="center", fontsize=11)
    ax.axvline(1.0, color="#444444", ls="--", lw=1.2)
    ax.text(
        1.08,
        0.5,
        "1 = no better than\npredicting the average",
        fontsize=9,
        va="center",
        color="#444444",
    )
    ax.set_xlabel("one-step next-latent error (MSE / target variance), log scale")
    ax.set_title(
        f"Next-latent prediction on {after['transitions']} held-out transitions",
        fontsize=11,
    )
    return _save(fig, path)


def plot_reward_before_after(
    panels: Sequence[tuple[str, dict[str, Any]]], path: str | Path | None = None
) -> Figure:
    """Predicted versus observed reward, one panel per model."""
    fig, axes = plt.subplots(
        1, len(panels), figsize=(3.6 * len(panels), 3.9), layout="constrained"
    )
    axes = np.atleast_1d(axes)
    for ax, (title, metrics) in zip(axes, panels, strict=True):
        reward, pred = metrics["reward"], metrics["reward_pred"]
        low = reward < LOW_REWARD
        ax.plot([0, 1], [0, 1], ls="--", color="#888888", lw=1)
        ax.scatter(reward[~low], pred[~low], s=12, alpha=0.45, color=PREDICTED)
        ax.scatter(reward[low], pred[low], s=22, alpha=0.9, color=ERROR, marker="x")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.15, 1.15)
        ax.set_xlabel("observed reward $r_{t+1}$")
        ax.set_title(
            f"{title}\nmean abs. error {metrics['reward_mae']:.3f}", fontsize=10
        )
    axes[0].set_ylabel("predicted reward $\\hat r_{t+1}$")
    axes[-1].legend(
        handles=[
            plt.Line2D([], [], marker="o", ls="", color=PREDICTED, label="transition"),
            plt.Line2D(
                [], [], marker="x", ls="", color=ERROR, label=f"reward < {LOW_REWARD}"
            ),
            plt.Line2D([], [], ls="--", color="#888888", label="perfect prediction"),
        ],
        loc="lower right",
        fontsize=8,
    )
    fig.suptitle("Reward prediction on held-out transitions", fontsize=12)
    return _save(fig, path)
