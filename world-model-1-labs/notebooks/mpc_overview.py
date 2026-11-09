"""Lab 2's opening MPC overview: plan with the frozen model, act once, replan.

Kept as a notebook helper so the generated Colab inlines the current drawing
code instead of relying on the frozen shared bundle.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from lab1.visualization.story import (
    EXECUTED,
    PREDICTED,
    PREDICTED_LIGHT,
    RECORDED,
    RECORDED_LIGHT,
    _arrow,
    _box,
)


def plot_mpc_overview(observation, use_my_lab1_model=False):
    """Opening overview of Lab 2's MPC loop with the selected frozen model."""
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    ax.imshow(
        observation,
        cmap="gray",
        vmin=0,
        vmax=255,
        interpolation="nearest",
        extent=(0.25, 2.85, 2.0, 4.6),
    )
    ax.add_patch(
        plt.Rectangle((0.25, 2.0), 2.6, 2.6, fill=False, ec=RECORDED, lw=2.2, zorder=3)
    )
    ax.text(1.55, 4.88, "current observation", ha="center", fontsize=10)
    ax.text(
        0.25,
        1.6,
        "real HighwayEnv input",
        ha="left",
        fontsize=9,
        color=RECORDED,
    )

    nodes = [
        (3.45, 3.0, 1.55, 1.15, "encode\nonce", RECORDED_LIGHT, RECORDED),
        (
            5.45,
            3.0,
            2.15,
            1.15,
            "sample candidate\naction sequences",
            "white",
            "#555555",
        ),
        (
            8.05,
            3.0,
            2.4,
            1.15,
            "predict latent futures\nand rewards",
            PREDICTED_LIGHT,
            PREDICTED,
        ),
        (
            10.9,
            3.0,
            2.0,
            1.15,
            "score each sequence\nby predicted return $J$",
            PREDICTED_LIGHT,
            PREDICTED,
        ),
        (
            13.45,
            3.0,
            2.35,
            1.15,
            "execute only\nthe first action",
            "white",
            EXECUTED,
        ),
    ]
    for x, y, width, height, text, face, edge in nodes:
        _box(ax, x, y, width, height, text, fc=face, ec=edge, size=9)
    _arrow(ax, (2.92, 3.58), (3.42, 3.58), lw=1.6)
    for left, right in zip(nodes, nodes[1:], strict=False):
        _arrow(
            ax,
            (left[0] + left[2] + 0.05, 3.58),
            (right[0] - 0.05, 3.58),
            lw=1.6,
        )

    ax.text(
        9.0,
        5.65,
        (
            "Lab 2 uses your Lab 1 model. It does not train a new model."
            if use_my_lab1_model
            else "By default, Lab 2 uses a pretrained world model. "
            "It does not train a new model."
        ),
        ha="center",
        fontsize=10,
        fontweight="bold",
        color=EXECUTED,
    )
    # Feedback: the executed action yields the next real observation, which
    # replaces the current observation before the next search.
    execute_x = nodes[-1][0] + nodes[-1][2] / 2
    loop_y, entry_x = 1.2, 2.5
    ax.plot(
        [execute_x, execute_x, entry_x, entry_x],
        [3.0, loop_y, loop_y, 1.6],
        color=RECORDED,
        lw=1.6,
        solid_joinstyle="miter",
    )
    ax.add_patch(
        FancyArrowPatch(
            (entry_x, 1.55),
            (entry_x, 2.0),
            arrowstyle="-|>",
            mutation_scale=14,
            color=RECORDED,
            lw=1.6,
            shrinkA=0,
            shrinkB=0,
        )
    )
    ax.text(
        8.55,
        0.75,
        "HighwayEnv executes the action and returns the next real observation;\n"
        "MPC plans again from that new observation",
        ha="center",
        va="center",
        fontsize=10,
        color=RECORDED,
    )
    ax.set_title("From a Learned World Model to One Real Action", fontsize=14)
    fig.tight_layout()
    return fig
