"""Measured/predicted profile plots and display smoothing."""

from __future__ import annotations

from typing import Any

import numpy
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.ndimage import convolve1d

from .layout import _axes


def smooth(values: numpy.ndarray, window: int) -> numpy.ndarray:
    """A centred boxcar mean over the last axis, for display.

    Coverage tracks are piecewise constant over fragment-length runs, and an
    averaged or predicted track steps at every run boundary, so the profile
    figures smooth before drawing. Each position averages only the positions
    the window covers, so a trace does not dip where the kernel runs off the
    end, which zero padding would make it do. ``window <= 1`` returns the
    values unchanged.

    Parameters
    ----------
    values
        ``(..., positions)``.
    window
        Boxcar width in positions.

    Returns
    -------
    ndarray
        Float array of the same shape.
    """
    values = numpy.asarray(values, dtype=float)
    if window <= 1:
        return values
    kernel = numpy.ones(int(window))
    total = convolve1d(values, kernel, axis=-1, mode="constant")
    count = convolve1d(numpy.ones(values.shape[-1]), kernel, mode="constant")
    return total / count


def plot_profiles(
    observed: numpy.ndarray | None = None,
    predicted: numpy.ndarray | None = None,
    ax: Axes | None = None,
    *,
    positions: numpy.ndarray | None = None,
    observed_kwargs: dict[str, Any] | None = None,
    predicted_kwargs: dict[str, Any] | None = None,
    **plot_kwargs: Any,
) -> tuple[Figure, Axes]:
    """Overlay one measured and/or predicted vector in caller-defined units.

    Parameters
    ----------
    observed, predicted
        Matching-length vectors. At least one is required.
    ax
        Draw here when given, otherwise a new fixed-size figure.
    positions
        X coordinates, defaulting to ``arange(len)``.
    observed_kwargs, predicted_kwargs
        Per-series overrides, applied last so they win.
    **plot_kwargs
        Applied to both series.

    Returns
    -------
    (Figure, Axes)

    Raises
    ------
    ValueError
        If neither series is given, or the two lengths differ.
    """
    if observed is None and predicted is None:
        raise ValueError("provide observed or predicted")
    fig, ax = _axes(ax)
    length = None
    for values, defaults, overrides in (
        (observed, {"label": "Observed", "color": "0.4"}, observed_kwargs),
        (predicted, {"label": "Predicted", "color": "#2a78d6"}, predicted_kwargs),
    ):
        if values is None:
            continue
        values = numpy.asarray(values)
        if values.ndim != 1 or (length is not None and len(values) != length):
            raise ValueError("profiles must be matching vectors")
        length = len(values)
        x = numpy.arange(length) if positions is None else positions
        ax.plot(x, values, **(defaults | plot_kwargs | (overrides or {})))
    return fig, ax
