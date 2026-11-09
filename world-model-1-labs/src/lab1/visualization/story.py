"""Story figures for the required Lab 1 notebook, in one visual language.

Every figure answers one question and uses the same colors:

* recorded (real) observations and actions: blue
* predicted or imagined quantities: orange
* the action actually executed: dark navy
* hypothetical plan tail that will be discarded: light gray
* error or failure: red
* reference only (not a model input): muted gray-blue
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patheffects
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle

from lab1.data.dataset import Episode
from lab1.imagination.rollout import imagine_from_episode
from lab1.models import WorldModel
from lab1.visualization.animations import ACTION_NAMES

RECORDED = "#1f5fa8"
RECORDED_LIGHT = "#dbe7f5"
PREDICTED = "#e6862e"
PREDICTED_LIGHT = "#fbe5cf"
EXECUTED = "#0b2a5b"
TAIL = "#dddddd"
TAIL_TEXT = "#6f6f6f"
ERROR = "#c62828"
REFERENCE = "#7d93ab"
EGO = "#39e75f"  # display-only ego outline; never part of the model input

# The camera follows the ego vehicle, which stays near the top center of each
# 64x64 frame; the driver's right is the image's right (checked in HighwayEnv).
_EGO_BOX = (28.5, 12.5, 5.0, 11.0)  # x, y, width, height in pixels


def _save(fig: Figure, path: str | Path | None) -> Figure:
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    return fig


def _name(action: int) -> str:
    return ACTION_NAMES.get(int(action), str(action))


def _frame(ax, image: np.ndarray, *, color: str, width: float, title: str) -> None:
    ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(width)
    ax.set_title(title, fontsize=9, color=color)


def _box(ax, x, y, w, h, text, *, fc, ec, color="black", weight="normal", size=10):
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=1.5)
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        color=color,
        fontsize=size,
        fontweight=weight,
    )


def _sequence(ax, names: Sequence[str], x0: float, y: float, *, w=1.0, h=0.5):
    """Draw a plan: the first action executed (navy), the tail hypothetical (gray)."""
    for k, name in enumerate(names):
        first = k == 0
        _box(
            ax,
            x0 + k * w + 0.04,
            y,
            w - 0.08,
            h,
            name,
            fc=EXECUTED if first else TAIL,
            ec=EXECUTED if first else "#b5b5b5",
            color="white" if first else TAIL_TEXT,
            weight="bold" if first else "normal",
            size=9,
        )


def _arrow(ax, start, end, *, color="#444444", style="-|>", lw=1.6, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=14,
            color=color,
            lw=lw,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


# --------------------------------------------------------------------------
# Legend
# --------------------------------------------------------------------------
def plot_visual_legend(path: str | Path | None = None) -> Figure:
    """One compact legend for the colors used in every figure of the lab."""
    items = [
        (RECORDED, "recorded (real) observation or action"),
        (PREDICTED, "predicted or imagined quantity"),
        (EXECUTED, "first action actually executed"),
        (TAIL, "hypothetical tail, discarded"),
        (ERROR, "error or failure"),
        (REFERENCE, "reference only, not model input"),
        (EGO, "ego vehicle outline (display only)"),
    ]
    fig, ax = plt.subplots(figsize=(9.0, 1.5))
    ax.legend(
        handles=[Patch(fc=color, ec="#888888", label=label) for color, label in items],
        loc="center",
        ncol=2,
        frameon=False,
        fontsize=10,
        handlelength=1.6,
    )
    ax.axis("off")
    ax.set_title("Colors used in every figure of this lab", fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


def _line(ax, points, *, color="#444444", lw=1.6) -> None:
    xs, ys = zip(*points, strict=True)
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="butt")


# --------------------------------------------------------------------------
# The world model
# --------------------------------------------------------------------------
def plot_world_model_architecture(path: str | Path | None = None) -> Figure:
    """Block diagram: encoder, then one world-model step with two outputs."""
    fig, ax = plt.subplots(figsize=(9.5, 3.3))
    ax.set_xlim(0, 13.4)
    ax.set_ylim(0.2, 4.7)
    ax.axis("off")
    blue = {"fc": RECORDED_LIGHT, "ec": RECORDED}
    orange = {"fc": PREDICTED_LIGHT, "ec": PREDICTED}
    plain = {"fc": "white", "ec": "#444444", "weight": "bold"}
    _box(ax, 0.1, 2.9, 2.5, 1.1, "Observation stack\n$o_t$", **blue)
    _box(ax, 3.2, 2.9, 1.6, 1.1, "Encoder\n$E$", **plain)
    _box(ax, 5.4, 2.9, 2.0, 1.1, "Latent state\n$z_t$", **blue)
    _box(ax, 5.4, 0.5, 2.0, 1.1, "Action\n$a_t$", **blue)
    _box(
        ax, 8.1, 0.5, 2.0, 3.5, "World model\n\ndynamics $f$\nreward head $R$", **plain
    )
    _box(ax, 10.8, 2.9, 2.5, 1.1, "Predicted next\nlatent $\\hat z_{t+1}$", **orange)
    _box(ax, 10.8, 0.5, 2.5, 1.1, "Predicted reward\n$\\hat r_{t+1}$", **orange)
    for (x0, x1), y in [((2.62, 3.18), 3.45), ((4.82, 5.38), 3.45)]:
        _arrow(ax, (x0, y), (x1, y))
    for y in (3.45, 1.05):
        _arrow(ax, (7.42, y), (8.08, y))
        _arrow(ax, (10.12, y), (10.78, y))
    ax.text(0.1, 4.4, "Encode what the car sees", fontsize=10, color=RECORDED)
    ax.text(5.4, 4.4, "Predict one step ahead", fontsize=10, color=PREDICTED)
    ax.set_title("The Lab 1 world model", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------
# Big question 1: predict a saved future
# --------------------------------------------------------------------------
def plot_latent_error_paths(path: str | Path | None = None) -> Figure:
    """What latent prediction error compares: a reference path and an imagined path."""
    fig, ax = plt.subplots(figsize=(9.5, 3.9))
    ax.set_xlim(0, 14.5)
    ax.set_ylim(-0.05, 4.85)
    ax.axis("off")
    ref = {"fc": "#e9eef4", "ec": REFERENCE}
    blue = {"fc": RECORDED_LIGHT, "ec": RECORDED}
    orange = {"fc": PREDICTED_LIGHT, "ec": PREDICTED}
    ax.text(0.1, 4.45, "REFERENCE", fontsize=10, fontweight="bold", color=REFERENCE)
    _box(ax, 0.1, 3.2, 2.8, 1.0, "Saved future\nobservation $o_{t+k}$", **ref)
    _arrow(ax, (2.92, 3.7), (8.48, 3.7), color=REFERENCE)
    ax.text(5.7, 3.82, "encoder $E$", ha="center", fontsize=10, color=REFERENCE)
    _box(ax, 8.5, 3.2, 2.2, 1.0, "Reference latent\n$z^{\\mathrm{ref}}_{t+k}$", **ref)

    ax.text(0.1, 1.9, "IMAGINATION", fontsize=10, fontweight="bold", color=PREDICTED)
    _box(ax, 0.1, 0.6, 2.8, 1.0, "Initial saved\nobservation $o_t$", **blue)
    _arrow(ax, (2.92, 1.1), (3.48, 1.1))
    ax.text(3.2, 1.22, "$E$", ha="center", fontsize=10)
    _box(ax, 3.5, 0.6, 1.0, 1.0, "$z_t$", **blue)
    _arrow(ax, (4.52, 1.1), (5.08, 1.1))
    _box(ax, 5.1, 0.6, 2.8, 1.0, "World model, $k$ steps\nsame saved actions", **orange)
    _arrow(ax, (7.92, 1.1), (8.48, 1.1))
    _box(ax, 8.5, 0.6, 2.2, 1.0, "Imagined latent\n$\\hat z_{t+k}$", **orange)

    _box(
        ax,
        11.5,
        1.55,
        2.9,
        1.6,
        "COMPARE\n$\\mathrm{MSE}(\\hat z_{t+k},\\ z^{\\mathrm{ref}}_{t+k})$",
        fc="white",
        ec=ERROR,
        color=ERROR,
    )
    _arrow(ax, (10.72, 3.7), (11.48, 2.85), color=REFERENCE)
    _arrow(ax, (10.72, 1.1), (11.48, 1.85), color=PREDICTED)
    ax.text(
        0.1,
        0.1,
        "Future saved observations are evaluation references only. "
        "They are never fed into the imagined rollout.",
        fontsize=10,
        color=TAIL_TEXT,
    )
    ax.set_title(
        "What latent prediction error compares", fontsize=12, fontweight="bold"
    )
    fig.tight_layout()
    return _save(fig, path)


def plot_latent_rollout(
    model: WorldModel,
    episode: Episode,
    start: int = 0,
    horizon: int = 10,
    planner_horizon: int | None = None,
    path: str | Path | None = None,
    device: torch.device | str = "cpu",
) -> Figure:
    """Show one latent rollout: one saved input, saved actions, latent error.

    Top: the saved start observation (the only model input) and the saved
    future observations (evaluation reference only). Middle: the latent chain
    with the saved trajectory action on each arrow. Bottom: latent MSE between
    each imagined latent and the encoding of the saved future observation.
    No image is decoded.
    """
    horizon = min(horizon, len(episode) - start)
    out = imagine_from_episode(model, episode, start, horizon, device=device)
    errors = out["latent_error"].cpu().numpy()
    steps = horizon + 1

    fig = plt.figure(figsize=(9.5, 5.5), layout="constrained")
    grid = fig.add_gridspec(4, steps, height_ratios=[0.32, 0.95, 0.8, 1.9])
    header = fig.add_subplot(grid[0, 1:])
    header.axis("off")
    header.annotate(
        "",
        xy=(0.0, 0.1),
        xytext=(1.0, 0.1),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "|-|", "color": REFERENCE, "lw": 1.2},
    )
    header.text(
        0.5,
        0.55,
        "Saved future observations: evaluation reference only, never fed back",
        ha="center",
        fontsize=10,
        color=REFERENCE,
        transform=header.transAxes,
    )
    start_header = fig.add_subplot(grid[0, 0])
    start_header.axis("off")
    start_header.text(
        0.5,
        0.3,
        "model input",
        ha="center",
        fontsize=9,
        color=RECORDED,
        fontweight="bold",
        transform=start_header.transAxes,
    )
    for k in range(steps):
        ax = fig.add_subplot(grid[1, k])
        if k == 0:
            _frame(ax, episode.obs[start, -1], color=RECORDED, width=3.5, title="t+0")
        else:
            _frame(
                ax,
                episode.obs[start + k, -1],
                color=REFERENCE,
                width=1.0,
                title=f"t+{k}",
            )
            ax.images[0].set_alpha(0.6)

    chain = fig.add_subplot(grid[2, :])
    chain.set_xlim(-0.5, steps - 0.5)
    chain.set_ylim(0, 1)
    chain.axis("off")
    for k in range(steps):
        if k == 0:
            _box(
                chain,
                -0.3,
                0.3,
                0.6,
                0.38,
                "$z_t$",
                fc=RECORDED_LIGHT,
                ec=RECORDED,
                size=9,
            )
            chain.text(0, 0.82, "encode once", ha="center", fontsize=9, color=RECORDED)
        else:
            _box(
                chain,
                k - 0.3,
                0.3,
                0.6,
                0.38,
                rf"$\hat z_{{t+{k}}}$",
                fc=PREDICTED_LIGHT,
                ec=PREDICTED,
                size=9,
            )
            _arrow(chain, (k - 0.68, 0.49), (k - 0.32, 0.49), color=RECORDED)
            chain.text(
                k - 0.5,
                0.08,
                _name(episode.actions[start + k - 1]),
                ha="center",
                fontsize=8,
                color=RECORDED,
            )
    chain.text(
        -0.5,
        -0.12,
        "saved trajectory action on each arrow; no observation is fed back after t+0",
        fontsize=9,
        color="#444444",
        transform=chain.transData,
    )

    ax_err = fig.add_subplot(grid[3, :])
    ax_err.bar(np.arange(steps), errors, color=ERROR, alpha=0.8, width=0.6)
    for k in range(1, steps):
        ax_err.text(
            k, errors[k], f"{errors[k]:.4f}", ha="center", va="bottom", fontsize=8
        )
    ax_err.set_ylim(0, errors.max() * 1.65 if errors.max() > 0 else 1.0)
    ax_err.set_xlim(-0.5, steps - 0.5)
    ax_err.set_xticks(np.arange(steps), [f"t+{k}" for k in range(steps)])
    ax_err.set_ylabel("latent MSE")
    ax_err.set_title(
        "Latent MSE: imagined latent vs encoding of the saved future observation",
        fontsize=10,
    )
    if planner_horizon is not None and planner_horizon < steps:
        ax_err.axvline(planner_horizon, color=EXECUTED, linestyle="--", lw=1.5)
        ax_err.text(
            planner_horizon - 0.08,
            ax_err.get_ylim()[1] * 0.95,
            f"H = {planner_horizon}: planning horizon",
            ha="right",
            fontsize=9,
            color=EXECUTED,
            va="top",
        )
    fig.suptitle(
        "One saved trajectory: imagined latents versus the saved future",
        fontsize=12,
        fontweight="bold",
    )
    return _save(fig, path)


def plot_horizon_error(
    stats: dict[str, np.ndarray],
    planner_horizon: int,
    highlight: Sequence[int] = (1,),
    num_starts: int = 64,
    path: str | Path | None = None,
    max_horizon: int | None = None,
) -> Figure:
    """Mean latent MSE versus horizon (from step 1).

    With ``max_horizon`` equal to the planning horizon the plot stops at H.
    Steps beyond H are shaded as a diagnostic beyond the MPC horizon.
    """
    mean, std = np.asarray(stats["mean"]), np.asarray(stats["std"])
    last = len(mean) - 1 if max_horizon is None else min(max_horizon, len(mean) - 1)
    steps = np.arange(1, last + 1)
    m, s = mean[steps], std[steps]
    fig, ax = plt.subplots(figsize=(8, 4.4), layout="constrained")
    ax.fill_between(
        steps, m - s, m + s, color=ERROR, alpha=0.15, label="± 1 SD across starts"
    )
    ax.plot(steps, m, "-o", color=ERROR, label="mean latent MSE")
    top = float((m + s).max()) * 1.3
    ax.set_ylim(0, top)
    ax.set_xlim(0.5, last + 0.5)
    if planner_horizon < last:
        ax.axvspan(planner_horizon + 0.5, last + 0.5, color=TAIL, alpha=0.6, lw=0)
        ax.text(
            (planner_horizon + last + 1) / 2,
            top * 0.97,
            "diagnostic beyond\nthe MPC horizon",
            ha="center",
            va="top",
            fontsize=10,
            color=TAIL_TEXT,
        )
        ax.axvline(planner_horizon, color=EXECUTED, linestyle="--", lw=1.5)
        ax.text(
            planner_horizon - 0.1,
            top * 0.97,
            f"H = {planner_horizon}",
            ha="right",
            va="top",
            fontsize=10,
            color=EXECUTED,
            fontweight="bold",
        )
    for h in sorted({*highlight, planner_horizon, last}):
        if 1 <= h <= last:
            beside_line = h == planner_horizon < last
            ax.annotate(
                f"{mean[h]:.5f}",
                (h, mean[h]),
                textcoords="offset points",
                xytext=(-6, -16) if beside_line else (0, -16),
                ha="right" if beside_line else "center",
                fontsize=9,
            )
    scope = (
        f"up to the planning horizon H = {planner_horizon}"
        if last == planner_horizon
        else f"horizons 1 to {last}"
    )
    ax.set(
        xlabel="imagined steps ahead, k (same saved trajectory actions)",
        ylabel="latent MSE",
        title=f"Latent error versus horizon, {scope}\n"
        f"mean over {num_starts} starting points",
    )
    ax.set_xticks(steps)
    ax.legend(loc="upper left", fontsize=9)
    return _save(fig, path)


# --------------------------------------------------------------------------
# Big question 2: can imagination help choose an action?
# --------------------------------------------------------------------------
def plot_mpc_cycle(num_candidates: int = 64, horizon: int = 6, path=None) -> Figure:
    """The MPC loop as a straight top-to-bottom flow with one return path."""
    steps = [
        ("Current observation", RECORDED_LIGHT, RECORDED, "black"),
        (
            f"Generate N = {num_candidates} candidate action sequences",
            PREDICTED_LIGHT,
            PREDICTED,
            "black",
        ),
        (
            f"Imagine H = {horizon} steps for each candidate",
            *[PREDICTED_LIGHT, PREDICTED],
            "black",
        ),
        ("Compute predicted return J", PREDICTED_LIGHT, PREDICTED, "black"),
        ("Select the highest-J sequence", PREDICTED_LIGHT, PREDICTED, "black"),
        ("Execute only its first action", EXECUTED, EXECUTED, "white"),
        ("Observe again and replan", RECORDED_LIGHT, RECORDED, "black"),
    ]
    fig, ax = plt.subplots(figsize=(8.0, 6.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.8, 10.3)
    ax.axis("off")
    x0, w, h, gap = 2.9, 4.8, 0.85, 1.32
    tops = [9.2 - i * gap for i in range(len(steps))]
    for i, ((label, fc, ec, color), y) in enumerate(zip(steps, tops, strict=True)):
        _box(ax, x0, y, w, h, label, fc=fc, ec=ec, color=color, size=11)
        if i:
            _arrow(ax, (x0 + w / 2, tops[i - 1] - 0.02), (x0 + w / 2, y + h + 0.02))
    right, loop_x = x0 + w, x0 + w + 0.9
    _line(ax, [(right + 0.02, tops[-1] + h / 2), (loop_x, tops[-1] + h / 2)])
    _line(ax, [(loop_x, tops[-1] + h / 2), (loop_x, tops[0] + h / 2)])
    _arrow(ax, (loop_x, tops[0] + h / 2), (right + 0.02, tops[0] + h / 2))
    ax.text(loop_x + 0.15, (tops[0] + tops[-1] + h) / 2, "next\nreal step", fontsize=10)

    def bracket(first, last, text, color):
        top, bottom = tops[first] + h, tops[last]
        _line(
            ax,
            [
                (x0 - 0.15, top),
                (x0 - 0.3, top),
                (x0 - 0.3, bottom),
                (x0 - 0.15, bottom),
            ],
            color=color,
        )
        ax.text(
            x0 - 0.45,
            (top + bottom) / 2,
            text,
            ha="right",
            va="center",
            fontsize=10,
            color=color,
        )

    bracket(1, 4, "inside the\nlearned model", PREDICTED)
    bracket(5, 5, "one real\nenvironment\nstep", EXECUTED)
    ax.set_title(
        "MPC: from predicted futures to one real action",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return _save(fig, path)


def plot_candidate_decision(
    obs: np.ndarray,
    candidates: dict[str, object],
    path: str | Path | None = None,
) -> Figure:
    """Why did MPC choose this first action now? One decision in one figure."""
    returns = np.asarray(candidates["returns"])
    rewards = np.asarray(candidates["rewards"])
    actions = np.asarray(candidates["actions"])
    best = int(candidates["best_index"])
    horizon = int(candidates["horizon"])
    names = [_name(a) for a in actions[best]]

    fig = plt.figure(figsize=(9.5, 6.2), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 2.7], height_ratios=[1.0, 1.0])
    ax_scene = fig.add_subplot(grid[:, 0])
    ax_scores = fig.add_subplot(grid[0, 1])
    ax_plan = fig.add_subplot(grid[1, 1])

    _frame(
        ax_scene,
        np.asarray(obs)[-1],
        color=RECORDED,
        width=3.0,
        title="Current real observation\n(newest frame of the stack)",
    )
    x, y, w, h = _EGO_BOX
    ax_scene.add_patch(Rectangle((x, y), w, h, fill=False, ec=EXECUTED, lw=2))
    if names[0] in ("lane right", "lane left"):
        sign = 1 if names[0] == "lane right" else -1
        tip = x + w + 9 if sign > 0 else x - 9
        base = x + w + 1 if sign > 0 else x - 1
        _arrow(ax_scene, (base, y + h / 2), (tip, y + h / 2), color=EXECUTED, lw=2)
    label = {"fontsize": 9, "color": EXECUTED, "bbox": {"fc": "white", "ec": "none"}}
    ax_scene.text(x - 1, y + h + 4.5, "ego", **label)

    ax_scores.scatter(
        np.arange(len(returns)),
        returns,
        color=PREDICTED,
        alpha=0.75,
        label="sampled candidate: predicted J",
    )
    ax_scores.scatter(
        best,
        returns[best],
        color=EXECUTED,
        s=170,
        marker="*",
        zorder=3,
        label="selected: highest predicted J",
    )
    ax_scores.annotate(
        f"J = {returns[best]:.3f}",
        (best, returns[best]),
        textcoords="offset points",
        xytext=(10, -14),
        fontsize=10,
        color=EXECUTED,
        fontweight="bold",
    )
    ax_scores.set(
        xlabel="candidate index (not time)",
        ylabel=f"predicted J\n(sum of {horizon} predicted rewards)",
        title=(
            f"{len(returns)} sampled candidate sequences, each scored inside the model"
        ),
    )
    low, high = float(returns.min()), float(returns.max())
    span = max(high - low, 1e-6)
    ax_scores.set_ylim(low - 0.45 * span, high + 0.12 * span)
    ax_scores.legend(loc="lower left", fontsize=9, ncol=2)

    ax_plan.set_xlim(0, horizon)
    ax_plan.set_ylim(-0.62, 1.4)
    ax_plan.axis("off")
    _sequence(ax_plan, names, 0, 0.35, w=1.0, h=0.55)
    for k in range(horizon):
        ax_plan.text(
            k + 0.5,
            1.08,
            "execute now" if k == 0 else "hypothetical",
            ha="center",
            fontsize=9,
            color=EXECUTED if k == 0 else TAIL_TEXT,
            fontweight="bold" if k == 0 else "normal",
        )
        ax_plan.text(
            k + 0.5,
            0.12,
            f"r\u0302 = {rewards[best, k]:.2f}",
            ha="center",
            fontsize=9,
            color=PREDICTED,
        )
    ax_plan.text(
        horizon / 2,
        -0.02,
        f"Execute {names[0]} now. Discard the remaining hypothetical actions\n"
        "after the next observation.",
        ha="center",
        va="top",
        fontsize=10,
        color=EXECUTED,
    )
    ax_plan.set_title(
        f"Selected sequence (J = {returns[best]:.3f}): "
        "only its first action is executed",
        fontsize=11,
    )
    fig.suptitle(f"Why did MPC choose {names[0]} now?", fontsize=13, fontweight="bold")
    return _save(fig, path)


# --------------------------------------------------------------------------
# Big question 3: does the selected action work in the real environment?
# --------------------------------------------------------------------------
def _check_alignment(run: dict[str, Any], decisions: Sequence) -> int:
    steps = len(run["actions"])
    if len(run["frames"]) != steps + 1 or len(decisions) != steps:
        raise ValueError(
            "Expected one decision per executed action and one extra observation: "
            f"{len(run['frames'])} observations, {steps} actions, "
            f"{len(decisions)} decisions."
        )
    for key in ("display_frames", "display_ego_boxes"):
        if key in run and len(run[key]) != steps + 1:
            raise ValueError(f"Expected one {key} entry per observation.")
    for k, (sequence, _) in enumerate(decisions):
        if int(sequence[0]) != int(run["actions"][k]):
            raise ValueError(f"Decision {k} does not start with executed action {k}.")
    return steps


def _annotate_ego(ax, box, action: str | None = None) -> None:
    """Outline and label the ego vehicle in a display frame (never model input).

    ``action`` adds an arrow for a lane change about to be executed. In the
    display view the road runs left to right, so the driver's right is down.
    """
    x0, y0, x1, y1 = box
    pad = 3
    stroke = [patheffects.withStroke(linewidth=4.5, foreground="black")]
    outline = Rectangle(
        (x0 - pad, y0 - pad),
        x1 - x0 + 2 * pad,
        y1 - y0 + 2 * pad,
        fill=False,
        ec=EGO,
        lw=2.2,
    )
    outline.set_path_effects(stroke)
    ax.add_patch(outline)
    ax.text(
        x0 - pad,
        y0 - pad - 2,
        "EGO",
        fontsize=8,
        fontweight="bold",
        color="black",
        va="bottom",
        bbox={"fc": EGO, "ec": "black", "lw": 0.8, "pad": 1.2},
    )
    if action in ("lane right", "lane left"):
        sign = 1 if action == "lane right" else -1
        cx = x1 + pad + 8
        cy = (y0 + y1) / 2
        arrow = FancyArrowPatch(
            (cx, cy - sign * 7),
            (cx, cy + sign * 15),
            arrowstyle="-|>",
            mutation_scale=12,
            color=EGO,
            lw=2.2,
        )
        arrow.set_path_effects(stroke)
        ax.add_patch(arrow)


def _display(ax, frame: np.ndarray, box, action: str | None, title: str) -> None:
    ax.imshow(frame, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(RECORDED)
        spine.set_linewidth(2)
    _annotate_ego(ax, box, action)
    ax.set_title(title, fontsize=10, color=RECORDED, loc="left")


def _model_input(ax, frame: np.ndarray) -> None:
    _frame(ax, frame[-1], color=RECORDED, width=1.5, title="model input\n(grayscale)")


def plot_replanning_storyboard(
    run: dict[str, Any],
    decisions: Sequence,
    count: int = 3,
    path: str | Path | None = None,
) -> Figure:
    """The first ``count`` real decisions: observation, best plan, first action.

    If the run recorded display-only frames, each decision shows the annotated
    display view next to the grayscale model input; otherwise only the model
    input is shown.
    """
    _check_alignment(run, decisions)
    count = min(count, len(decisions))
    display = "display_frames" in run
    if display:
        fig = plt.figure(figsize=(9.5, 2.55 * count + 0.4), layout="constrained")
        grid = fig.add_gridspec(2 * count, 2, width_ratios=[6.0, 1.0])
    else:
        fig = plt.figure(figsize=(9.5, 1.9 * count + 0.5), layout="constrained")
        grid = fig.add_gridspec(count, 2, width_ratios=[1.0, 5.2])
    for k in range(count):
        sequence, predicted = decisions[k]
        names = [_name(a) for a in sequence]
        label = "Real observation 0" if k == 0 else f"New real observation {k}"
        if display:
            lane = run.get("lane_indices", [None] * (k + 1))[k]
            lane_text = "" if lane is None else f", ego in lane {lane}"
            _display(
                fig.add_subplot(grid[2 * k, 0]),
                run["display_frames"][k],
                run["display_ego_boxes"][k],
                names[0],
                f"Decision {k}: {label.lower()}{lane_text} (display view)",
            )
            _model_input(fig.add_subplot(grid[2 * k, 1]), run["frames"][k])
            ax = fig.add_subplot(grid[2 * k + 1, :])
            ax.set_xlim(0, 9.4)
            ax.set_ylim(0, 0.95)
            ax.axis("off")
            _sequence(ax, names, 0, 0.2, w=1.0, h=0.55)
            ax.text(6.25, 0.62, f"Predicted J = {predicted:.2f}", color=PREDICTED)
            ax.text(
                6.25,
                0.12,
                f"Execute: {names[0]}; discard the rest",
                color=EXECUTED,
                fontweight="bold",
            )
            continue
        _frame(
            fig.add_subplot(grid[k, 0]),
            run["frames"][k][-1],
            color=RECORDED,
            width=3.0,
            title="Real\nobservation 0" if k == 0 else f"New real\nobservation {k}",
        )
        ax = fig.add_subplot(grid[k, 1])
        ax.set_xlim(0, 9.4)
        ax.set_ylim(0, 1.6)
        ax.axis("off")
        ax.text(
            0.04,
            1.32,
            f"Decision {k}: new MPC search; best imagined plan",
            fontsize=11,
            fontweight="bold",
        )
        _sequence(ax, names, 0, 0.55, w=1.0, h=0.55)
        ax.text(
            6.25, 1.02, f"Predicted J = {predicted:.2f}", fontsize=11, color=PREDICTED
        )
        ax.text(
            6.25,
            0.62,
            f"Execute: {names[0]}",
            fontsize=11,
            color=EXECUTED,
            fontweight="bold",
        )
        ax.text(6.25, 0.22, "Discard the rest", fontsize=11, color=TAIL_TEXT)
    fig.suptitle(
        "Replanning: a new search after every real observation",
        fontsize=12,
        fontweight="bold",
    )
    return _save(fig, path)


def mpc_frame_records(run: dict[str, Any], decisions: Sequence) -> list[dict]:
    """One record per animation frame, aligned with the MPC decisions.

    Observation ``k`` is shown with the reward of the action that produced it
    (none for the initial observation) and with the new search made from it.
    The last observation has no search: the episode ended or hit the step limit.
    """
    steps = _check_alignment(run, decisions)
    records = []
    for k in range(steps + 1):
        record: dict[str, Any] = {"index": k, "plan": None, "J": None, "execute": None}
        if k == 0:
            record.update(kind="initial", reward=None, executed=None)
            record["header"] = "Initial real observation (before any action)"
        else:
            record.update(
                kind="final" if k == steps else "replan",
                reward=float(run["rewards"][k - 1]),
                executed=_name(run["actions"][k - 1]),
            )
            record["header"] = (
                f"After real step {k} of {steps}"
                if k == steps
                else f"After executing the previous action (real step {k} of {steps})"
            )
        if k < steps:
            sequence, predicted = decisions[k]
            record["plan"] = [_name(a) for a in sequence]
            record["J"] = float(predicted)
            record["execute"] = record["plan"][0]
        else:
            record["crashed"] = bool(run["crashed"])
            record["episode_return"] = float(run["total_reward"])
            record["ended_by"] = "crash" if run["crashed"] else "time or step limit"
        records.append(record)
    return records


def _draw_record_panel(ax, record: dict[str, Any], compact: bool = False) -> None:
    """Text for one animation frame: reward received, new search, first action."""
    size = 10 if compact else 11
    ax.clear()
    ax.set_xlim(0, 6.2)
    ax.set_ylim(0.65 if compact else 0, 5)
    ax.axis("off")
    ax.text(0.05, 4.55, record["header"], fontsize=size + 1, fontweight="bold")
    if record["reward"] is not None:
        ax.text(
            0.05,
            3.95,
            f"Executed {record['executed']}; received actual environment reward "
            f"{record['reward']:.3f}",
            fontsize=size,
            color=RECORDED,
        )
    if record["plan"] is not None:
        ax.text(
            0.05,
            3.25,
            (
                "New MPC search"
                if record["kind"] == "initial"
                else "New real observation, new MPC search"
            )
            + "; best sequence:",
            fontsize=size,
        )
        _sequence(ax, record["plan"], 0, 2.25, w=1.0, h=0.7)
        ax.text(
            0.05,
            1.6,
            f"Predicted J = {record['J']:.2f}",
            fontsize=size,
            color=PREDICTED,
        )
        ax.text(
            0.05,
            1.0,
            f"Execute its first action: {record['execute']}; discard the gray tail",
            fontsize=size,
            color=EXECUTED,
            fontweight="bold",
        )
    else:
        color = ERROR if record["crashed"] else "black"
        ended = "" if record["crashed"] else f" ({record['ended_by']})"
        ax.text(0.05, 3.1, f"Crash: {record['crashed']}", fontsize=13, color=color)
        ax.text(0.05, 2.4, f"Episode ends{ended}", fontsize=13, color=color)
        ax.text(
            0.05,
            1.7,
            f"Actual episode return: {record['episode_return']:.2f}",
            fontsize=13,
            color=color,
            fontweight="bold",
        )


def animate_mpc_run(
    run: dict[str, Any],
    decisions: Sequence,
    title: str = "MPC replanning",
    interval: int = 1200,
    note: str | None = None,
) -> Any:
    """Animate a planner episode together with the search made at each step.

    If the run recorded display-only frames, the annotated display view is
    shown on top, with the grayscale model input beside the search panel.
    """
    from IPython.display import HTML

    records = mpc_frame_records(run, decisions)
    display = "display_frames" in run
    if display:
        fig = plt.figure(figsize=(9.5, 5.2 if note else 4.9), dpi=80)
        fig.set_layout_engine("constrained")
        grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 4.6], height_ratios=[1, 1.7])
        ax_display = fig.add_subplot(grid[0, :])
        ax_obs = fig.add_subplot(grid[1, 0])
        ax_panel = fig.add_subplot(grid[1, 1])
    else:
        fig = plt.figure(figsize=(11.5, 4.9 if note else 4.6), dpi=80)
        fig.set_layout_engine("constrained")
        grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 2.8])
        ax_obs = fig.add_subplot(grid[0, 0])
        ax_panel = fig.add_subplot(grid[0, 1])
    image = ax_obs.imshow(
        run["frames"][0][-1], cmap="gray", vmin=0, vmax=255, interpolation="nearest"
    )
    ax_obs.set_xticks([])
    ax_obs.set_yticks([])
    for spine in ax_obs.spines.values():
        spine.set_edgecolor(RECORDED)
        spine.set_linewidth(3 if not display else 1.5)
    obs_title = ax_obs.set_title(
        "model input\n(grayscale)" if display else "",
        fontsize=9 if display else 10,
        color=RECORDED,
    )
    fig.suptitle(title, fontsize=12, fontweight="bold")
    if note:
        fig.supxlabel(note, fontsize=10, color=TAIL_TEXT)

    def draw(index: int):
        record = records[index]
        image.set_data(run["frames"][index][-1])
        observation = f"Real observation {index}" + (
            " (before any action)" if index == 0 else ""
        )
        if display:
            ax_display.clear()
            lane = run.get("lane_indices", [None] * (index + 1))[index]
            lane_text = "" if lane is None else f", ego in lane {lane}"
            _display(
                ax_display,
                run["display_frames"][index],
                run["display_ego_boxes"][index],
                record["execute"],
                f"{observation}{lane_text} (display view)",
            )
        else:
            obs_title.set_text(observation)
        _draw_record_panel(ax_panel, record, compact=display)
        return (image,)

    draw(0)
    animation = FuncAnimation(
        fig, draw, frames=len(records), interval=interval, blit=False
    )
    plt.close(fig)
    return HTML(animation.to_jshtml(default_mode="once"))


def plot_predicted_vs_actual(
    episodes: Sequence[tuple[str, dict[str, Any], Sequence]],
    path: str | Path | None = None,
) -> Figure:
    """For each episode: final observation, first predicted J, actual outcome.

    ``episodes`` holds ``(label, run, decisions)`` triples from MPC runs.
    """
    count = len(episodes)
    fig = plt.figure(figsize=(9.5, 2.0 * count + 0.6), layout="constrained")
    grid = fig.add_gridspec(count, 2, width_ratios=[2.4, 1.6], hspace=0.15)
    for row, (label, run, decisions) in enumerate(episodes):
        steps = _check_alignment(run, decisions)
        ax_view = fig.add_subplot(grid[row, 0])
        if "display_frames" in run:
            _display(
                ax_view,
                run["display_frames"][steps],
                run["display_ego_boxes"][steps],
                None,
                f"{label}: final real observation (display view)",
            )
        else:
            _frame(
                ax_view,
                run["frames"][steps][-1],
                color=RECORDED,
                width=2.0,
                title=f"{label}: final real observation",
            )
        ax = fig.add_subplot(grid[row, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        sequence, first_j = decisions[0]
        crashed = bool(run["crashed"])
        outcome = "crash" if crashed else "no crash"
        rows = [
            (0.86, "Predicted at the first decision", PREDICTED, "normal"),
            (
                0.66,
                f"J = {float(first_j):.2f} over {len(sequence)} imagined steps",
                PREDICTED,
                "bold",
            ),
            (0.36, "Actual episode outcome", "black", "normal"),
            (
                0.16,
                f"{steps} real steps, return {float(run['total_reward']):.2f}, "
                f"{outcome}",
                ERROR if crashed else EXECUTED,
                "bold",
            ),
        ]
        for y, text, color, weight in rows:
            ax.text(
                0.03, y, text, fontsize=10, color=color, fontweight=weight, va="center"
            )
    fig.suptitle(
        "Predicted J is not the actual episode return", fontsize=12, fontweight="bold"
    )
    return _save(fig, path)


def plot_ten_seed_evidence(
    comparison: dict[str, object], path: str | Path | None = None
) -> Figure:
    """Actual episode returns and crashes for planner and random on fixed seeds."""
    groups = [
        ("world-model planner (MPC)", "planner", "#5b7fb5"),
        ("random policy (baseline)", "random", "#a6a6a6"),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.8), layout="constrained")
    rng = np.random.default_rng(0)
    summaries = []
    for i, (_label, key, color) in enumerate(groups):
        returns = np.asarray(comparison[f"{key}_returns"])
        crashed = np.asarray([bool(run["crashed"]) for run in comparison[key]])
        mean, sd = returns.mean(), returns.std()
        ax.bar(i, mean, width=0.55, color=color, alpha=0.55)
        ax.errorbar(i, mean, yerr=sd, color="black", capsize=8, lw=1.5)
        jitter = rng.uniform(-0.12, 0.12, len(returns))
        ax.scatter(i + jitter[crashed], returns[crashed], marker="x", color=ERROR, s=45)
        ax.scatter(
            i + jitter[~crashed], returns[~crashed], marker="o", color="black", s=30
        )
        summaries.append(
            f"{mean:.2f} ± {sd:.2f} SD, crashes {crashed.sum()}/{len(returns)}"
        )
    ax.set_xticks(
        [0, 1],
        [
            f"{label}\n{text}"
            for (label, _, _), text in zip(groups, summaries, strict=True)
        ],
    )
    ax.set_ylabel("actual episode return\n(sum of actual environment rewards)")
    count = len(comparison["planner_returns"])
    ax.set_title(f"Actual outcomes on {count} fixed seeds (0 to {count - 1})")
    top = max(float(np.max(comparison[f"{k}_returns"])) for _, k, _ in groups)
    ax.set_ylim(0, top * 1.25 + 1)
    ax.legend(
        handles=[
            plt.Line2D(
                [], [], marker="x", color=ERROR, ls="", label="episode, crashed"
            ),
            plt.Line2D(
                [], [], marker="o", color="black", ls="", label="episode, no crash"
            ),
            plt.Line2D([], [], color="black", label="± 1 episode SD (not a CI)"),
        ],
        loc="upper right",
        fontsize=9,
    )
    return _save(fig, path)


def plot_lab_synthesis(path: str | Path | None = None) -> Figure:
    """Three connected cards: imagine, plan, evaluate."""
    cards = [
        (
            "Predict",
            "saved trajectory actions\nlatent error grows\nwith horizon",
            RECORDED,
        ),
        (
            "Plan",
            "sample candidates\nselect by predicted J\n"
            "execute first action\nobserve and replan",
            PREDICTED,
        ),
        (
            "Evaluate",
            "actual return and crashes\nuseful signal,\nnot a safety guarantee",
            ERROR,
        ),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 2.9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")
    for i, (title, body, color) in enumerate(cards):
        x = 0.2 + i * 4.0
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.2),
                3.2,
                2.6,
                boxstyle="round,pad=0.05",
                fc="white",
                ec=color,
                lw=2.5,
            )
        )
        ax.text(
            x + 1.6,
            2.35,
            title,
            ha="center",
            fontsize=14,
            fontweight="bold",
            color=color,
        )
        ax.text(x + 1.6, 1.25, body, ha="center", va="center", fontsize=11)
        if i:
            _arrow(ax, (x - 0.72, 1.5), (x - 0.08, 1.5))
    ax.set_title("Three questions, one chain", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------
# Optional: explore the planner
# --------------------------------------------------------------------------
def planner_probe_table(
    results: Sequence[dict[str, Any]],
    *,
    latent_mse: Sequence[float] | None = None,
    show_sequence: bool = False,
) -> str:
    """Plain-text table of planner probes.

    Raw J sums H predicted rewards, so it is always shown next to J/H, the
    average predicted reward per imagined step. ``latent_mse`` (indexed by
    horizon) adds the recorded-action latent MSE at each probe's horizon.
    """
    header = ["N", "H", "N x H", "seed", "first action", "best J", "J/H"]
    if latent_mse is not None:
        header.append("latent MSE at H")
    header.append("time (ms)")
    rows, sequences = [], []
    for result in results:
        horizon = int(result["horizon"])
        row = [
            str(result["num_sequences"]),
            str(horizon),
            str(result["imagined_transitions"]),
            str(result["seed"]),
            _name(result["first_action"]),
            f"{result['best_J']:.3f}",
            f"{result['J_per_step']:.3f}",
        ]
        if latent_mse is not None:
            row.append(
                f"{latent_mse[horizon]:.5f}" if horizon < len(latent_mse) else "n/a"
            )
        row.append(f"{1000 * result['seconds']:.1f}")
        rows.append(row)
        names = [_name(a) for a in result["sequence"]]
        sequences.append(" -> ".join([f"[{names[0]}]", *names[1:]]))
    widths = [
        max(len(name), *(len(row[i]) for row in rows)) for i, name in enumerate(header)
    ]
    lines = [
        " | ".join(
            name.ljust(width) for name, width in zip(header, widths, strict=True)
        ),
        "-+-".join("-" * width for width in widths),
    ]
    for row, sequence in zip(rows, sequences, strict=True):
        lines.append(
            " | ".join(
                cell.ljust(width) for cell, width in zip(row, widths, strict=True)
            )
        )
        if show_sequence:
            lines.append(f"    selected sequence: {sequence}")
    return "\n".join(lines)
