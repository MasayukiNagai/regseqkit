"""Fixed plotting-area geometry, with dimensions in inches."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def subplots_with_plot_size(
    nrows: int = 1,
    ncols: int = 1,
    plot_size: tuple[float, float] = (3.0, 2.5),
    *,
    extra_space: dict[str, float] | None = None,
    wspace: float = 0.4,
    hspace: float = 0.4,
    margins: tuple[float, float, float, float] = (0.9, 0.3, 0.6, 0.3),
    subplot_kwargs: dict[str, Any] | None = None,
    **fig_kwargs: Any,
) -> tuple[Figure, numpy.ndarray]:
    """Return a figure and 2D axes array with exact axes sizes in inches.

    Margins are (left, right, bottom, top); gaps are inches, not fractions.
    Automatic layout and bbox_inches='tight' can change the exported geometry.
    """
    if not isinstance(nrows, int) or not isinstance(ncols, int) or min(nrows, ncols) < 1:
        raise ValueError("nrows and ncols must be positive integers")
    extra = dict(extra_space or {})
    if set(extra) - {"left", "right", "bottom", "top"}:
        raise ValueError("extra_space keys must be left, right, bottom, top")
    if min(plot_size) <= 0 or min(*margins, wspace, hspace, *extra.values()) < 0:
        raise ValueError("plot sizes must be positive and spacing non-negative")
    if any(
        key in fig_kwargs for key in ("figsize", "layout", "tight_layout", "constrained_layout")
    ):
        raise ValueError("fixed plot geometry controls figsize and layout")
    width, height = plot_size
    left, right, bottom, top = (
        value + extra.get(side, 0.0)
        for value, side in zip(margins, ("left", "right", "bottom", "top"))
    )
    fw = left + ncols * width + (ncols - 1) * wspace + right
    fh = bottom + nrows * height + (nrows - 1) * hspace + top
    fig = plt.figure(figsize=(fw, fh), layout="none", **fig_kwargs)
    grid = fig.add_gridspec(
        nrows,
        ncols,
        left=left / fw,
        right=1 - right / fw,
        bottom=bottom / fh,
        top=1 - top / fh,
        wspace=wspace / width,
        hspace=hspace / height,
    )
    axes = numpy.empty((nrows, ncols), dtype=object)
    for row in range(nrows):
        for col in range(ncols):
            axes[row, col] = fig.add_subplot(grid[row, col], **(subplot_kwargs or {}))
    return fig, axes


def figure_with_plot_size(
    plot_size: tuple[float, float] = (3.0, 2.5), **kwargs: Any
) -> tuple[Figure, Axes]:
    """Return one fixed-size plotting area; all dimensions are in inches."""
    fig, axes = subplots_with_plot_size(1, 1, plot_size, **kwargs)
    return fig, axes[0, 0]


def _axes(ax: Axes | None) -> tuple[Figure, Axes]:
    return figure_with_plot_size() if ax is None else (ax.figure, ax)
