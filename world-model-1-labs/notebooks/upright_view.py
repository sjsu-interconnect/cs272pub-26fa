"""Show saved HighwayEnv frames with the ego car driving up the image.

Display only: figures are flipped vertically after drawing. Stored
observations, model inputs and predictions are never changed.
"""

import numpy as np


def upright(frames):
    """A vertically flipped view of a frame or frame stack, for display."""
    return np.asarray(frames)[..., ::-1, :]


def upright_run(run):
    """A copy of an MPC run whose model-input frames are displayed upright."""
    return {**run, "frames": [upright(frame) for frame in run["frames"]]}


def upright_figure(fig):
    """Flip every grayscale observation panel in ``fig``, overlays included."""
    for ax in fig.axes:
        images = ax.get_images()
        if len(images) != 1 or images[0].get_array().ndim != 2:
            continue
        left, right, bottom, top = images[0].get_extent()
        fills_axes = np.allclose(sorted(ax.get_xlim()), sorted((left, right))) and (
            np.allclose(sorted(ax.get_ylim()), sorted((bottom, top)))
        )
        if fills_axes:
            ax.invert_yaxis()
    return fig
