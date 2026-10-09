"""Density-colored scatter plots."""

from __future__ import annotations

from typing import Any

import numpy
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .layout import _axes


def plot_density_scatter(
    x: numpy.ndarray,
    y: numpy.ndarray,
    ax: Axes | None = None,
    *,
    bins: int = 100,
    **scatter_kwargs: Any,
) -> tuple[Figure, Axes]:
    """Color points by 2D histogram density, drawing dense points last.

    Histogram lookup avoids the quadratic work of evaluating a KDE at every
    point. Non-finite pairs are omitted. The collection is ax.collections[-1].
    Pass c or color to override density coloring.
    """
    x, y = numpy.asarray(x), numpy.asarray(y)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError("x and y must be matching vectors")
    keep = numpy.isfinite(x) & numpy.isfinite(y)
    original_count = len(x)
    x, y = x[keep], y[keep]
    fig, ax = _axes(ax)
    if len(x):
        density, xe, ye = numpy.histogram2d(x, y, bins=bins)
        xi = numpy.clip(numpy.searchsorted(xe, x, side="right") - 1, 0, len(xe) - 2)
        yi = numpy.clip(numpy.searchsorted(ye, y, side="right") - 1, 0, len(ye) - 2)
        c = density[xi, yi]
        order = numpy.argsort(c, kind="stable")
        density_color = "c" not in scatter_kwargs and "color" not in scatter_kwargs
        if not density_color:
            # Preserve input order for caller-supplied pointwise styling.
            order = numpy.arange(len(x))
        for key in ("s", "alpha", "linewidths", "c"):
            if key not in scatter_kwargs:
                continue
            value = numpy.asarray(scatter_kwargs[key])
            if value.ndim and len(value) == original_count:
                scatter_kwargs[key] = value[keep][order]
        if density_color:
            scatter_kwargs.update(c=c[order], cmap=scatter_kwargs.get("cmap", "viridis"))
        scatter_kwargs.setdefault("s", 4)
        if "linewidth" not in scatter_kwargs and "linewidths" not in scatter_kwargs:
            scatter_kwargs["linewidths"] = 0
        ax.scatter(x[order], y[order], **scatter_kwargs)
    return fig, ax
