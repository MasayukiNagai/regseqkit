"""Density-colored scatter plots and comparable count/contrast frames."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from scipy.stats import pearsonr, spearmanr

from .layout import _axes
from .style import DENSITY_CMAP, GRID, INK, style


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


def plot_contrast_pair(
    first: numpy.ndarray,
    second: numpy.ndarray,
    ax: Axes | None = None,
    limit: float | None = None,
    **scatter_kwargs: Any,
) -> tuple[Figure, Axes]:
    """One signed contrast against another, every point drawn, on a square axis.

    The signed counterpart of :func:`plot_count_pair`: the same density-coloured
    scatter, with the frame centred on zero and both zero lines drawn, since
    the cloud of null loci near the origin is most of the mass and the sign of
    a point is what design reads.

    ``limit`` sets the half-width of the square frame; by default it is the
    99.9th percentile of ``|both|``, so the same pair of vectors always produces
    the same extent whichever stage draws it. Pass an explicit value to share
    one frame across a grid of panels -- necessary when one panel's predictions
    have a heavier tail than another's and would otherwise set their own scale.
    Density is binned within the frame, so a far outlier cannot coarsen the
    colouring of the core. Points outside the frame are not drawn, so a caller
    that narrows the limit should say what fraction it dropped.
    """
    fig, ax = _axes(ax)
    if limit is None:
        limit = float(numpy.percentile(numpy.abs(numpy.concatenate([first, second])), 99.9))
    edges = numpy.linspace(-limit, limit, 101)
    scatter_kwargs.setdefault("bins", [edges, edges])
    scatter_kwargs.setdefault("cmap", DENSITY_CMAP)
    scatter_kwargs.setdefault("norm", LogNorm())
    scatter_kwargs.setdefault("s", 3)
    scatter_kwargs.setdefault("rasterized", True)
    plot_density_scatter(first, second, ax=ax, **scatter_kwargs)
    ax.axhline(0, color=GRID["color"], linewidth=0.8, zorder=1)
    ax.axvline(0, color=GRID["color"], linewidth=0.8, zorder=1)
    ax.plot(
        [-limit, limit],
        [-limit, limit],
        color=INK["secondary"],
        linewidth=1.0,
        linestyle=(0, (4, 3)),
        zorder=2,
    )
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    style(ax)
    return fig, ax


def square_frame(vectors: Sequence[numpy.ndarray], pad: float = 0.03) -> tuple[float, float]:
    """One square frame for every panel of a figure.

    The pooled range of all the vectors, padded so the extreme points do not
    sit on the spine. Passing the result to every panel of a grid is what makes
    a wider cloud mean a noisier pair rather than a different axis.
    """
    low = min(float(numpy.min(vector)) for vector in vectors)
    high = max(float(numpy.max(vector)) for vector in vectors)
    margin = pad * (high - low)
    return low - margin, high + margin


def plot_count_pair(
    x: numpy.ndarray,
    y: numpy.ndarray,
    ax: Axes | None = None,
    *,
    limits: tuple[float, float] | None = None,
    annotate: bool = True,
    **scatter_kwargs: Any,
) -> tuple[Figure, Axes]:
    """One log-count vector against another: every point drawn, coloured by density.

    A hexbin tiles a cloud of this shape into visible cells at any gridsize and
    swallows the outliers a count scatter exists to show, so each point is
    drawn, coloured by the count of its 2D-histogram bin on a log scale. The
    points are rasterized so a PDF of a hundred thousand of them stays small;
    the axes remain vector.

    ``limits`` is the square frame; by default the pair's own padded range.
    Pass one :func:`square_frame` to a grid of panels so spreads compare. With
    ``annotate``, Pearson r on the values drawn and Spearman rho, which the log
    leaves unchanged, are written into the empty upper-left corner.
    """
    fig, ax = _axes(ax)
    if limits is None:
        limits = square_frame([x, y])
    scatter_kwargs.setdefault("cmap", DENSITY_CMAP)
    scatter_kwargs.setdefault("norm", LogNorm())
    scatter_kwargs.setdefault("s", 3)
    scatter_kwargs.setdefault("rasterized", True)
    plot_density_scatter(x, y, ax=ax, **scatter_kwargs)

    ax.plot(
        limits,
        limits,
        color=INK["secondary"],
        linewidth=1.0,
        linestyle=(0, (4, 3)),
        zorder=2,
    )
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_aspect("equal")
    if annotate:
        ax.text(
            0.03,
            0.97,
            f"Pearson r = {pearsonr(x, y).statistic:.3f}\n"
            f"Spearman $\\rho$ = {spearmanr(x, y).statistic:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color=INK["primary"],
            linespacing=1.5,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": GRID["color"],
                "linewidth": 0.6,
            },
        )
    style(ax)
    return fig, ax
