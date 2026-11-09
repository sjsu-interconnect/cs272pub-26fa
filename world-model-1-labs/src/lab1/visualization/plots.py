"""Matplotlib figures for every milestone of the lab.

Each function returns a :class:`matplotlib.figure.Figure` so the same code
works inline in notebooks and when saved to disk from scripts. Pass ``path`` to
also save the figure.

The six required visualizations:

1/2. training & validation loss -- :func:`plot_losses`
3.   real vs reconstructed image -- :func:`plot_reconstruction`
4.   latent space PCA -- :func:`plot_latent_pca`
5.   prediction error vs horizon -- :func:`plot_error_vs_horizon`
6.   real vs imagined trajectory -- :func:`plot_real_vs_imagined`
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

from lab1.data.dataset import Episode
from lab1.imagination.rollout import (
    imagine_from_episode,
    prediction_error_vs_horizon,
)
from lab1.models import WorldModel
from lab1.visualization.story import PREDICTED, RECORDED

# Teacher-forced reward predictions: a darker shade of the "predicted" orange.
_TEACHER_FORCED = "#8c4a12"

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)


def _save(fig: Figure, path: str | Path | None) -> Figure:
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    return fig


def _last_frame(img: torch.Tensor | np.ndarray) -> np.ndarray:
    """Return the newest grayscale frame ``(H, W)`` from a stacked image."""
    arr = img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else img
    return arr[-1]


def plot_losses(
    history: list[dict],
    path: str | Path | None = None,
    log_scale: bool = False,
    include_total: bool = True,
) -> Figure:
    """Plot train/validation curves for total and all component losses.

    ``log_scale`` puts loss on a log axis, which keeps a large first-epoch
    training average from flattening the remaining curves. With
    ``include_total=False`` only the three component losses are shown.
    """
    epochs = [h["epoch"] for h in history]
    if include_total:
        terms = [
            ("total", "Total loss"),
            ("dynamics", "Next-latent loss"),
            ("recon", "Reconstruction (aux image) loss"),
            ("reward", "Reward loss"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    else:
        terms = [
            ("dynamics", "Next-latent loss"),
            ("reward", "Reward loss"),
            ("recon", "Auxiliary image loss"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
    for ax, (term, title) in zip(axes.flat, terms, strict=True):
        train = [h["train"].get(term, 0.0) for h in history]
        val = [h["val"].get(term, 0.0) for h in history]
        ax.plot(epochs, train, "-o", label="train", color="#1f77b4")
        ax.plot(epochs, val, "-s", label="validation", color="#d62728")
        ax.set(title=title, xlabel="epoch", ylabel="loss")
        if log_scale:
            ax.set_yscale("log")
        ax.set_xticks(epochs)
        ax.legend()

    fig.tight_layout()
    return _save(fig, path)


def plot_training_budget_comparison(
    quick_history: list[dict],
    classroom_history: list[dict],
    path: str | Path | None = None,
) -> Figure:
    """Compare validation loss for the quick and classroom training budgets."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, term, title in [
        (axes[0], "total", "Validation total loss"),
        (axes[1], "reward", "Validation reward loss"),
    ]:
        for history, label, color in [
            (quick_history, "quick: 5k transitions, 5 epochs", "#d62728"),
            (
                classroom_history,
                "classroom: 20k transitions, 20 epochs",
                "#2ca02c",
            ),
        ]:
            epochs = [row["epoch"] for row in history]
            values = [row["val"][term] for row in history]
            ax.plot(epochs, values, "-o", markersize=3, label=label, color=color)
        ax.set(title=title, xlabel="epoch", ylabel="loss")
        ax.legend(fontsize=9)
    fig.suptitle("Same architecture, different training budget")
    fig.tight_layout()
    return _save(fig, path)


def plot_reconstruction(
    model: WorldModel,
    obs: torch.Tensor,
    n: int = 6,
    path: str | Path | None = None,
    device: torch.device | str = "cpu",
) -> Figure:
    """Show ``n`` real observations above their reconstructions."""
    if model.decoder is None:
        raise RuntimeError("Reconstruction requires a decoder (use_decoder=True).")
    device = torch.device(device)
    model = model.to(device).eval()
    obs = obs[:n].to(device)
    with torch.no_grad():
        recon = model.decode(model.encode(obs))

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4.2))
    for i in range(n):
        axes[0, i].imshow(_last_frame(obs[i]), cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_axis_off()
        axes[1, i].imshow(_last_frame(recon[i]), cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_axis_off()
    axes[0, 0].set_title("real", loc="left")
    axes[1, 0].set_title("reconstructed", loc="left")
    fig.suptitle("Original vs reconstructed observations")
    fig.tight_layout()
    return _save(fig, path)


def plot_latent_pca(
    model: WorldModel,
    episodes: list[Episode],
    max_points: int = 2000,
    path: str | Path | None = None,
    device: torch.device | str = "cpu",
) -> Figure:
    """Project encoded latents to 2D with PCA, colored by reward."""
    device = torch.device(device)
    model = model.to(device).eval()

    frames, rewards = [], []
    for ep in episodes:
        frames.append(ep.obs[:-1])  # align with rewards (length T)
        rewards.append(ep.rewards)
    frames = np.concatenate(frames, axis=0)
    rewards = np.concatenate(rewards, axis=0)

    if len(frames) > max_points:
        idx = np.random.default_rng(0).choice(
            len(frames), size=max_points, replace=False
        )
        frames, rewards = frames[idx], rewards[idx]

    with torch.no_grad():
        latents = (
            model.encode(torch.from_numpy(frames.astype(np.float32) / 255.0).to(device))
            .cpu()
            .numpy()
        )
    coords = PCA(n_components=2).fit_transform(latents)

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(
        coords[:, 0], coords[:, 1], c=rewards, cmap="viridis", s=8, alpha=0.7
    )
    fig.colorbar(sc, ax=ax, label="reward")
    ax.set_title("Latent space (PCA)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    fig.tight_layout()
    return _save(fig, path)


def plot_error_vs_horizon(
    model: WorldModel,
    episodes: list[Episode],
    horizon: int = 10,
    num_samples: int = 64,
    path: str | Path | None = None,
    device: torch.device | str = "cpu",
) -> Figure:
    """Plot mean latent prediction error against the rollout horizon."""
    stats = prediction_error_vs_horizon(
        model, episodes, horizon, num_samples=num_samples, device=device
    )
    steps = np.arange(len(stats["mean"]))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(steps, stats["mean"], "-o", color="#d62728", label="mean")
    ax.fill_between(
        steps,
        stats["mean"] - stats["std"],
        stats["mean"] + stats["std"],
        color="#d62728",
        alpha=0.2,
        label="±1 std",
    )
    ax.set_title("Latent prediction error vs. horizon")
    ax.set_xlabel("imagined step")
    ax.set_ylabel("latent MSE")
    ax.legend()
    fig.tight_layout()
    return _save(fig, path)


def plot_real_vs_imagined(
    model: WorldModel,
    episode: Episode,
    start: int = 0,
    horizon: int = 8,
    path: str | Path | None = None,
    device: torch.device | str = "cpu",
) -> Figure:
    """Show real frames above decoder-imagined frames over a rollout."""
    if model.decoder is None:
        raise RuntimeError("This plot requires a decoder (use_decoder=True).")
    out = imagine_from_episode(model, episode, start, horizon, device=device)
    real = out["real_obs"]
    imagined = out["imagined_obs"]
    steps = real.shape[0]

    fig, axes = plt.subplots(2, steps, figsize=(1.8 * steps, 4.2))
    for t in range(steps):
        axes[0, t].imshow(_last_frame(real[t]), cmap="gray", vmin=0, vmax=1)
        axes[0, t].set_axis_off()
        axes[0, t].set_title(f"t+{t}", fontsize=9)
        axes[1, t].imshow(_last_frame(imagined[t]), cmap="gray", vmin=0, vmax=1)
        axes[1, t].set_axis_off()
    axes[0, 0].set_ylabel("real")
    axes[1, 0].set_ylabel("imagined")
    fig.suptitle("Recorded vs imagined trajectory (one recorded start; no live steps)")
    fig.tight_layout()
    return _save(fig, path)


def plot_imagination_grid(
    model: WorldModel,
    episode: Episode,
    start: int = 0,
    horizon: int = 6,
    path: str | Path | None = None,
    device: torch.device | str = "cpu",
) -> Figure:
    """Primary teaching figure: recorded future, imagined frames, and their difference.

    Three rows -- recorded future frames, decoder-imagined frames, and the
    absolute pixel difference between them -- for ``horizon + 1`` steps. The
    imagined branch starts from one recorded observation and recursively
    predicts all later latents. Recorded future observations are comparison
    targets, not rollout inputs. No live simulator is stepped by this diagnostic.
    """
    if model.decoder is None:
        raise RuntimeError("This plot requires a decoder (use_decoder=True).")
    horizon = min(horizon, len(episode) - start)
    out = imagine_from_episode(model, episode, start, horizon, device=device)
    real = out["real_obs"]
    imagined = out["imagined_obs"]
    steps = real.shape[0]

    fig, axes = plt.subplots(3, steps, figsize=(1.9 * steps, 6.0), layout="constrained")
    axes = np.atleast_2d(axes)
    err_im = None
    for t in range(steps):
        rt = _last_frame(real[t])
        it = _last_frame(imagined[t])
        for row, frame in ((0, rt), (1, it)):
            axes[row, t].imshow(frame, cmap="gray", vmin=0, vmax=1)
            axes[row, t].set_xticks([])
            axes[row, t].set_yticks([])
        err_im = axes[2, t].imshow(np.abs(rt - it), cmap="magma", vmin=0, vmax=1)
        axes[2, t].set_xticks([])
        axes[2, t].set_yticks([])
        axes[0, t].set_title(f"t+{t}", fontsize=10)
    axes[0, 0].set_ylabel("recorded future", fontsize=11)
    axes[1, 0].set_ylabel("imagined", fontsize=11)
    axes[2, 0].set_ylabel("absolute pixel\ndifference", fontsize=11)
    fig.colorbar(
        err_im, ax=axes[2, :].tolist(), location="bottom", fraction=0.05, pad=0.02
    )
    fig.suptitle(
        f"Multi-step imagination from one recorded observation (H={steps - 1}): "
        "no live environment steps",
        fontsize=14,
        fontweight="bold",
    )
    return _save(fig, path)


def plot_reward_prediction(
    model: WorldModel,
    episode: Episode,
    max_len: int = 20,
    path: str | Path | None = None,
    device: torch.device | str = "cpu",
) -> Figure:
    """Actual versus predicted reward along one saved trajectory.

    Compares the actual environment reward with the reward head's prediction
    from the observed state (encoded saved frames) and from the imagined state
    (recursively rolled-out latents). Requires ``use_reward_head=True``.
    """
    if model.reward_head is None:
        raise RuntimeError("This plot requires a reward head (use_reward_head=True).")
    device = torch.device(device)
    model = model.to(device).eval()
    horizon = min(max_len, len(episode))
    actions = torch.from_numpy(episode.actions[:horizon].astype(np.int64)).to(device)
    obs0 = (
        torch.from_numpy(episode.obs[0].astype(np.float32) / 255.0)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        imagined = model.imagine(obs0, actions.unsqueeze(0))
        imagined_rewards = imagined["rewards"][0].cpu().numpy()
        real_frames = torch.from_numpy(
            episode.obs[:horizon].astype(np.float32) / 255.0
        ).to(device)
        tf_rewards = (
            model.predict_reward(model.encode(real_frames), actions).cpu().numpy()
        )
    true_rewards = episode.rewards[:horizon]
    steps = np.arange(horizon)

    fig, (ax_line, ax_scatter) = plt.subplots(1, 2, figsize=(10, 4.3))
    ax_line.plot(
        steps, true_rewards, "-o", color=RECORDED, label="actual environment reward"
    )
    ax_line.plot(
        steps,
        tf_rewards,
        "--s",
        color=_TEACHER_FORCED,
        label="prediction from observed state",
    )
    ax_line.plot(
        steps,
        imagined_rewards,
        "-^",
        color=PREDICTED,
        label="prediction from imagined state",
    )
    ax_line.set_title("Reward along the saved trajectory")
    ax_line.set_xlabel("step")
    ax_line.set_ylabel("reward")
    ax_line.legend()

    ax_scatter.scatter(
        true_rewards,
        tf_rewards,
        c=_TEACHER_FORCED,
        label="from observed state",
        alpha=0.7,
    )
    ax_scatter.scatter(
        true_rewards,
        imagined_rewards,
        c=PREDICTED,
        label="from imagined state",
        alpha=0.7,
    )
    lo = float(min(true_rewards.min(), tf_rewards.min(), imagined_rewards.min()))
    hi = float(max(true_rewards.max(), tf_rewards.max(), imagined_rewards.max()))
    ax_scatter.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="perfect")
    ax_scatter.set_title("Predicted vs actual reward")
    ax_scatter.set_xlabel("actual environment reward")
    ax_scatter.set_ylabel("predicted reward")
    ax_scatter.legend()

    mae_tf = float(np.mean(np.abs(tf_rewards - true_rewards)))
    mae_im = float(np.mean(np.abs(imagined_rewards - true_rewards)))
    fig.suptitle(
        "Reward prediction along one saved trajectory\n"
        f"(MAE: observed state {mae_tf:.3f}, imagined state {mae_im:.3f})",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return _save(fig, path)


def plot_cumulative_reward_prediction(
    model: WorldModel,
    episode: Episode,
    max_len: int = 20,
    path: str | Path | None = None,
    device: torch.device | str = "cpu",
) -> Figure:
    """Plot actual and predicted rewards accumulated along one trajectory."""
    if model.reward_head is None:
        raise RuntimeError("This plot requires a reward head (use_reward_head=True).")
    device = torch.device(device)
    model = model.to(device).eval()
    horizon = min(max_len, len(episode))
    actions = torch.from_numpy(episode.actions[:horizon].astype(np.int64)).to(device)
    obs0 = (
        torch.from_numpy(episode.obs[0].astype(np.float32) / 255.0)
        .unsqueeze(0)
        .to(device)
    )
    real_obs = torch.from_numpy(episode.obs[:horizon].astype(np.float32) / 255.0).to(
        device
    )

    with torch.no_grad():
        imagined = model.imagine(obs0, actions.unsqueeze(0))["rewards"][0]
        teacher_forced = model.predict_reward(model.encode(real_obs), actions)

    true = episode.rewards[:horizon]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.cumsum(true), "-o", color=RECORDED, label="actual cumulative reward")
    ax.plot(
        np.cumsum(teacher_forced.cpu().numpy()),
        "--s",
        color=_TEACHER_FORCED,
        label="cumulative prediction from observed state",
    )
    ax.plot(
        np.cumsum(imagined.cpu().numpy()),
        "-^",
        color=PREDICTED,
        label="cumulative prediction from imagined state",
    )
    ax.set(
        xlabel="step along the saved trajectory",
        ylabel="cumulative reward",
        title="Accumulating the per-step reward predictions",
    )
    ax.legend()
    fig.tight_layout()
    return _save(fig, path)


def plot_candidate_plans(
    candidates: dict[str, object],
    path: str | Path | None = None,
) -> Figure:
    """Show candidate returns and the reward accumulation of the winner."""
    returns = np.asarray(candidates["returns"])
    rewards = np.asarray(candidates["rewards"])
    best = int(candidates["best_index"])
    horizon = int(candidates["horizon"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(np.arange(len(returns)), returns, alpha=0.7, label="candidate")
    axes[0].scatter(best, returns[best], color="crimson", s=90, label="selected")
    axes[0].set(
        xlabel="candidate rollout",
        ylabel=f"sum of {horizon} predicted rewards",
        title=f"{len(returns)} candidate plans at one real decision",
    )
    axes[0].legend()

    best_rewards = rewards[best]
    steps = np.arange(1, horizon + 1)
    axes[1].bar(steps, best_rewards, alpha=0.65, label="predicted step reward")
    axes[1].plot(
        steps,
        np.cumsum(best_rewards),
        "-o",
        color="crimson",
        label="cumulative predicted return",
    )
    axes[1].set(
        xlabel="imagined horizon step",
        ylabel="predicted reward",
        title="Why the selected plan wins",
    )
    axes[1].legend()
    fig.tight_layout()
    return _save(fig, path)


def plot_planner_comparison(
    comparison: dict[str, object],
    path: str | Path | None = None,
) -> Figure:
    """Compare actual environment returns from planner and random episodes."""
    planner_returns = np.asarray(comparison["planner_returns"])
    random_returns = np.asarray(comparison["random_returns"])
    fig, ax = plt.subplots(figsize=(7, 4))
    means = [planner_returns.mean(), random_returns.mean()]
    stds = [planner_returns.std(), random_returns.std()]
    ax.bar(
        ["world-model planner", "random policy"],
        means,
        yerr=stds,
        capsize=5,
        color=["#2ca02c", "#7f7f7f"],
        alpha=0.75,
    )
    ax.scatter(np.zeros_like(planner_returns), planner_returns, color="black", zorder=3)
    ax.scatter(np.ones_like(random_returns), random_returns, color="black", zorder=3)
    ax.set(
        ylabel="actual total HighwayEnv reward",
        title="Closed-loop evaluation in HighwayEnv (bars: mean ± episode SD)",
    )
    fig.tight_layout()
    return _save(fig, path)


def plot_imagination_budget(
    real_steps: int,
    imagined_steps: int,
    path: str | Path | None = None,
) -> Figure:
    """Bar chart contrasting real environment steps with imagined model steps.

    Illustrates the core benefit of a world model: the agent *thinks* (imagines
    cheap internal transitions) far more than it *acts* (queries the simulator).
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(
        ["real env steps", "imagined model steps"],
        [real_steps, imagined_steps],
        color=["#1f77b4", "#d62728"],
    )
    ax.set_yscale("log")
    ax.set_ylabel("transitions (log scale)")
    ax.set_title("Think a lot, act a little")
    for rect, value in zip(bars, [real_steps, imagined_steps], strict=True):
        ax.annotate(
            f"{value:,}",
            (rect.get_x() + rect.get_width() / 2, rect.get_height()),
            ha="center",
            va="bottom",
            fontsize=11,
        )
    fig.tight_layout()
    return _save(fig, path)


@torch.no_grad()
def one_step_latent_error(
    model: WorldModel,
    episodes: list[Episode],
    num_samples: int = 256,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> float:
    """Mean one-step prediction error measured in *latent* space.

    A decoder-independent readout of dynamics quality: the MSE between
    ``predict_next(encode(o_t), a_t)`` and the stop-gradient target
    ``encode(o_{t+1})``, averaged over sampled transitions.
    """
    device = torch.device(device)
    model = model.to(device).eval()
    rng = np.random.default_rng(seed)

    pairs = [(ep_i, t) for ep_i, ep in enumerate(episodes) for t in range(len(ep))]
    picks = rng.choice(len(pairs), size=min(num_samples, len(pairs)), replace=False)

    errors = []
    for idx in picks:
        ep_i, t = pairs[idx]
        ep = episodes[ep_i]
        obs = torch.from_numpy(ep.obs[t].astype(np.float32) / 255.0).unsqueeze(0)
        next_obs = torch.from_numpy(ep.obs[t + 1].astype(np.float32) / 255.0).unsqueeze(
            0
        )
        action = torch.tensor([ep.actions[t]])
        z = model.encode(obs.to(device))
        z_pred = model.predict_next(z, action.to(device))
        z_target = model.encode(next_obs.to(device))
        errors.append(float(((z_pred - z_target) ** 2).mean()))
    return float(np.mean(errors))


# ---------------------------------------------------------------------------
# Lab 2: from naive planning to robust planning
# ---------------------------------------------------------------------------


def plot_return_vs_budget(
    naive: dict[str, object],
    random_ref: dict[str, float] | None = None,
    robust: dict[str, object] | None = None,
    path: str | Path | None = None,
    robust_label: str = "uncertainty-aware planner",
) -> Figure:
    """The planning paradox: real MPC return as the search budget grows.

    ``naive`` and ``robust`` are outputs of
    :func:`lab1.planning.return_vs_budget`. The comparison tests whether extra
    search transfers to real return and whether an uncertainty-aware score is
    more stable; neither curve is guaranteed to be monotonic in a finite run.
    """
    budgets = naive["budgets"]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.errorbar(
        budgets,
        naive["mean"],
        yerr=naive["std"],
        marker="o",
        capsize=3,
        color="#d62728",
        label="naive planner",
    )
    if robust is not None:
        ax.errorbar(
            robust["budgets"],
            robust["mean"],
            yerr=robust["std"],
            marker="s",
            capsize=3,
            color="#2ca02c",
            label=robust_label,
        )
    if random_ref is not None:
        ax.axhline(
            random_ref["mean"],
            color="#7f7f7f",
            linestyle="--",
            label="random policy",
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("candidate action sequences (search budget)")
    ax.set_ylabel("real episode return")
    ax.set_title("Does more search help or hurt?")
    ax.legend()
    fig.tight_layout()
    return _save(fig, path)


def plot_reward_overoptimism(
    result: dict[str, object],
    path: str | Path | None = None,
) -> Figure:
    """The core diagnosis: reward predictions binned by *true* reward.

    ``result`` is the output of :func:`lab1.planning.reward_overoptimism`. The
    bars contrast the true reward with the model's prediction in each bin; the
    supplied checkpoint overpredicts rare low-reward transitions. These bins
    are not safety labels or a unique causal diagnosis of crashes. If ensemble
    disagreement is present, it is overlaid as an imperfect error-warning proxy.
    """
    labels = result["labels"]
    true_mean = result["true_mean"]
    pred_mean = result["pred_mean"]
    disagreement = result["disagreement"]
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, true_mean, width, color="#1f77b4", label="true reward")
    ax.bar(
        x + width / 2,
        pred_mean,
        width,
        color="#d62728",
        label="predicted reward",
    )
    for xi, (t, p) in enumerate(zip(true_mean, pred_mean, strict=True)):
        if p - t > 0.05:
            ax.annotate(
                f"+{p - t:.2f}",
                (xi + width / 2, p),
                ha="center",
                va="bottom",
                fontsize=9,
                color="#d62728",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("true reward bin (not a collision or safety label)")
    ax.set_ylabel("reward")
    ax.set_title("Reward prediction bias on recorded validation transitions")
    ax.legend(loc="upper left")

    has_disagreement = np.isfinite(disagreement).any()
    if has_disagreement:
        ax_d = ax.twinx()
        ax_d.plot(
            x,
            disagreement,
            "-o",
            color="#2ca02c",
            label="ensemble disagreement",
        )
        ax_d.set_ylabel("disagreement (uncalibrated proxy)", color="#2ca02c")
        ax_d.tick_params(axis="y", labelcolor="#2ca02c")
        ax_d.grid(False)
        ax_d.legend(loc="upper right")

    fig.tight_layout()
    return _save(fig, path)


def plot_overoptimism_vs_budget(
    gap: dict[str, object],
    path: str | Path | None = None,
) -> Figure:
    """Predicted vs realized return of the selected plan, per search budget.

    ``gap`` is the output of :func:`lab1.planning.overoptimism_vs_budget`. A
    secondary probe comparing the planner's predicted return with realized
    return. A positive gap does not isolate reward error from dynamics,
    termination, horizon or objective effects.
    """
    budgets = gap["budgets"]
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4.6))

    ax_l.plot(budgets, gap["predicted_mean"], "-o", color="#d62728", label="predicted")
    ax_l.plot(budgets, gap["realized_mean"], "-s", color="#1f77b4", label="realized")
    ax_l.set_xscale("log", base=2)
    ax_l.set_xticks(budgets)
    ax_l.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax_l.set_xlabel("candidate action sequences")
    ax_l.set_ylabel("plan return (over horizon)")
    ax_l.set_title("Predicted versus realized plan return")
    ax_l.legend()

    ax_r.errorbar(
        budgets,
        gap["gap_mean"],
        yerr=gap["gap_std"],
        marker="o",
        capsize=3,
        color="#9467bd",
    )
    ax_r.axhline(0.0, color="k", linewidth=0.8)
    ax_r.set_xscale("log", base=2)
    ax_r.set_xticks(budgets)
    ax_r.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax_r.set_xlabel("candidate action sequences")
    ax_r.set_ylabel("over-optimism gap (predicted - realized)")
    ax_r.set_title("Measured prediction gap versus search budget")

    fig.suptitle(
        "Predicted and realized outcomes of selected plans",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, path)


def plot_error_vs_disagreement(
    result: dict[str, object],
    path: str | Path | None = None,
) -> Figure:
    """Scatter of ensemble reward error vs disagreement (does doubt track error?).

    ``result`` is the output of :func:`lab1.planning.error_vs_disagreement`. A
    positive trend is evidence of association on these samples, not calibrated
    uncertainty, a failure probability, or a safety guarantee.
    """
    errors = result["errors"]
    disagreements = result["disagreements"]
    steps = result["steps"]

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        disagreements,
        errors,
        c=steps,
        cmap="viridis",
        s=14,
        alpha=0.6,
    )
    fig.colorbar(sc, ax=ax, label="imagined step")

    # Least-squares trend line for a clear visual read.
    if len(errors) > 1 and np.std(disagreements) > 0:
        slope, intercept = np.polyfit(disagreements, errors, 1)
        xs = np.linspace(disagreements.min(), disagreements.max(), 100)
        ax.plot(xs, slope * xs + intercept, "k--", alpha=0.7, label="trend")
        ax.legend()

    ax.set_xlabel("ensemble disagreement (uncalibrated proxy)")
    ax.set_ylabel("reward error |mean - true|")
    ax.set_title(
        "Is disagreement associated with reward error?\n"
        f"Pearson r = {result['pearson']:.2f}, "
        f"Spearman r = {result['spearman']:.2f}"
    )
    fig.tight_layout()
    return _save(fig, path)


def plot_lambda_sweep(
    sweep: dict[str, object],
    random_ref: dict[str, float] | None = None,
    path: str | Path | None = None,
) -> Figure:
    """Return and crash rate vs the pessimism weight lambda (the trade-off).

    ``sweep`` is the output of :func:`lab1.planning.lambda_sweep`.
    """
    lambdas = sweep["lambdas"]
    fig, ax_ret = plt.subplots(figsize=(7, 4.8))

    color_ret = "#2ca02c"
    ax_ret.errorbar(
        lambdas,
        sweep["mean"],
        yerr=sweep["std"],
        marker="o",
        capsize=3,
        color=color_ret,
        label="return",
    )
    ax_ret.set_xlabel("pessimism weight  lambda")
    ax_ret.set_ylabel("real episode return", color=color_ret)
    ax_ret.tick_params(axis="y", labelcolor=color_ret)
    if random_ref is not None:
        ax_ret.axhline(
            random_ref["mean"], color="#7f7f7f", linestyle="--", label="random"
        )

    ax_crash = ax_ret.twinx()
    color_crash = "#d62728"
    ax_crash.plot(
        lambdas, sweep["crash_rate"], "-s", color=color_crash, label="crash rate"
    )
    ax_crash.set_ylabel("crash rate", color=color_crash)
    ax_crash.tick_params(axis="y", labelcolor=color_crash)
    ax_crash.grid(False)

    ax_ret.set_title("Disagreement penalty: measured return and crash rate")
    fig.tight_layout()
    return _save(fig, path)
