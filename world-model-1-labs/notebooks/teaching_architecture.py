"""Notebook-only teaching visual; generated Colab embeds this helper directly."""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from torch import nn

from lab1.visualization.story import (
    EXECUTED,
    PREDICTED,
    PREDICTED_LIGHT,
    RECORDED,
    RECORDED_LIGHT,
    TAIL_TEXT,
    _arrow,
    _box,
)


def teaching_model_dimensions(model, image_size):
    """Read layer widths, actual feature shapes and parameter counts."""
    convolutions = [m for m in model.encoder.conv if isinstance(m, nn.Conv2d)]
    channels = [convolutions[0].in_channels]
    sizes = [image_size]
    features = torch.zeros(1, channels[0], image_size, image_size)
    with torch.no_grad():
        for layer in model.encoder.conv:
            features = layer(features)
            if isinstance(layer, nn.Conv2d):
                channels.append(layer.out_channels)
                sizes.append(features.shape[-1])

    def widths(head):
        layers = [m for m in head.net if isinstance(m, nn.Linear)]
        return [layers[0].in_features, *(layer.out_features for layer in layers)]

    counts = {
        name: sum(p.numel() for p in getattr(model, name).parameters())
        for name in ("encoder", "dynamics", "reward_head", "decoder")
    }
    return {
        "channels": channels,
        "spatial_sizes": sizes,
        "convolutions": [(layer.kernel_size, layer.stride) for layer in convolutions],
        "flatten_shape": tuple(features.shape[1:]),
        "projection": (
            model.encoder.fc.in_features,
            model.encoder.fc.out_features,
        ),
        "latent": model.latent_dim,
        "actions": model.num_actions,
        "dynamics": widths(model.dynamics),
        "reward": widths(model.reward_head),
        "parameters": counts,
        "total_parameters": sum(counts.values()),
        "planning_parameters": sum(
            counts[name] for name in ("encoder", "dynamics", "reward_head")
        ),
    }


def plot_teaching_architecture(model, image_size):
    """Show the actual CNN, projection and action-conditioned prediction heads."""
    dimensions = teaching_model_dimensions(model, image_size)
    arrow = " → "
    channels = arrow.join(map(str, dimensions["channels"]))
    sizes = arrow.join(map(str, dimensions["spatial_sizes"]))
    flattened = " × ".join(map(str, dimensions["flatten_shape"]))
    latent, actions = dimensions["latent"], dimensions["actions"]
    kernels = set(dimensions["convolutions"])
    if kernels != {((3, 3), (2, 2))}:
        raise ValueError("This teaching figure expects 3×3 stride-2 convolutions.")

    fig, ax = plt.subplots(figsize=(14, 5.4))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 7)
    ax.axis("off")
    blue = {"fc": RECORDED_LIGHT, "ec": RECORDED}
    orange = {"fc": PREDICTED_LIGHT, "ec": PREDICTED}
    plain = {"fc": "white", "ec": "#555555"}
    _box(
        ax,
        0.1,
        3.2,
        2.5,
        1.6,
        f"{dimensions['channels'][0]} stacked\ngrayscale frames\n"
        f"shape: {dimensions['channels'][0]} × {image_size} × {image_size}",
        **blue,
        size=10,
    )
    _box(
        ax,
        3.0,
        2.8,
        4.0,
        2.4,
        f"CNN encoder $E$\n{len(dimensions['convolutions'])} convolutional layers\n"
        f"Conv 3×3, stride 2, ReLU\nchannels: {channels}\nspatial size: {sizes}",
        **blue,
        size=10,
    )
    _box(
        ax,
        7.4,
        3.2,
        2.6,
        1.6,
        f"flatten\n{flattened}\nlinear projection",
        **plain,
        size=10,
    )
    _box(
        ax,
        10.4,
        3.2,
        2.5,
        1.6,
        f"learned state $z_t$\n{latent}-dimensional\nfloat vector",
        **blue,
        size=10,
    )
    _box(
        ax,
        13.3,
        3.0,
        2.6,
        2.0,
        f"$z_t$: {latent} values\n$a_t$: {actions}-value\none-hot action\n"
        f"combined input:\n{latent + actions} values",
        **plain,
        size=10,
    )
    ax.text(
        16.2,
        6.4,
        "MLP (fully connected network)",
        ha="center",
        fontsize=11,
        color=PREDICTED,
    )
    for y, title, widths, output in [
        (4.15, "Dynamics $f$", dimensions["dynamics"], r"next latent $\hat z_{t+1}$"),
        (1.7, "Reward head $R$", dimensions["reward"], r"reward $\hat r_{t+1}$"),
    ]:
        _box(
            ax,
            16.4,
            y,
            3.5,
            1.9,
            f"{title}\nMLP: {arrow.join(map(str, widths))}\n"
            f"ReLU hidden layers\noutput: predicted\n{output}",
            **orange,
            size=10,
        )
    for left, right in [
        (2.6, 3.0),
        (7.0, 7.4),
        (10.0, 10.4),
        (12.9, 13.3),
    ]:
        _arrow(ax, (left + 0.03, 4.0), (right - 0.03, 4.0), lw=1.6)
    _arrow(ax, (15.95, 4.1), (16.37, 5.1), lw=1.6, color=PREDICTED)
    _arrow(ax, (15.95, 3.8), (16.37, 2.65), lw=1.6, color=PREDICTED)
    ax.text(14.6, 5.5, "selected action $a_t$", ha="center", fontsize=10)
    _arrow(ax, (14.6, 5.3), (14.6, 5.03), lw=1.4)
    _box(
        ax,
        7.7,
        0.85,
        7.9,
        1.35,
        "Auxiliary decoder $D$ — training only\n"
        "reconstructs the input observation during training\n"
        "not used by imagination or MPC",
        fc="#f4f4f4",
        ec="#aaaaaa",
        color=TAIL_TEXT,
        size=10,
    )
    ax.patches[-1].set_linestyle("--")
    ax.annotate(
        "",
        xy=(11.65, 2.22),
        xytext=(11.65, 3.17),
        arrowprops={"arrowstyle": "->", "color": "#aaaaaa", "linestyle": "--"},
    )
    ax.text(
        0.1,
        1.35,
        "MPC uses $E$, $f$ and $R$;\nnot the decoder.",
        fontsize=10,
        color=EXECUTED,
    )
    ax.text(
        10,
        0.25,
        "Small deterministic teaching model: "
        "no recurrence, stochastic latent, or Transformer.\n"
        "These are practical teaching choices, not theoretically optimal values.",
        ha="center",
        va="center",
        fontsize=10,
    )
    ax.set_title("Our Small Teaching Model", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig
