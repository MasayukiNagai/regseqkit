"""Array-only metrics, independent of any prediction model or torch.

Named ``metrics`` rather than ``evaluation`` because a *scorer* in
:mod:`regseqkit.scoring` also produces a number from model outputs, and the two
answer different questions: a scorer says what a sequence is worth, these
functions say how close a prediction came to an observation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d
from scipy.special import gammaln, rel_entr, xlogy
from scipy.stats import rankdata


def pearson_correlation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    axis: int = -1,
    constant: float = np.nan,
) -> np.ndarray:
    """Pearson correlation along one axis, with a chosen value for constants.

    Parameters
    ----------
    x, y
        Matching-shape arrays.
    axis
        Axis to correlate along.
    constant
        Value to report where either input has zero variance, making the
        correlation undefined. Pass ``0.0`` to match reports that treat a flat
        profile as uncorrelated rather than as missing.

    Returns
    -------
    ndarray
        Correlations, with `axis` reduced.

    Raises
    ------
    ValueError
        If the shapes differ.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("correlation arrays must match")
    x = x - x.mean(axis=axis, keepdims=True)
    y = y - y.mean(axis=axis, keepdims=True)
    denominator = np.sqrt((x * x).sum(axis=axis) * (y * y).sum(axis=axis))
    return np.divide(
        (x * y).sum(axis=axis),
        denominator,
        out=np.full_like(denominator, constant),
        where=denominator > 0,
    )


def spearman_correlation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    axis: int = -1,
    constant: float = np.nan,
) -> np.ndarray:
    """Spearman correlation along one axis: Pearson of average ranks.

    Average-rank ties matter here rather than being a detail. Base-resolution
    ATAC profiles are mostly zeros, so ties dominate, and a dense ranking that
    breaks them by position in the array produces a number that is not a rank
    correlation of anything.

    Parameters
    ----------
    x, y
        Matching-shape arrays.
    axis
        Axis to correlate along.
    constant
        Value where either input has zero variance after ranking.

    Returns
    -------
    ndarray
        Correlations, with `axis` reduced.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("correlation arrays must match")
    return pearson_correlation(
        rankdata(x, axis=axis), rankdata(y, axis=axis), axis=axis, constant=constant
    )


def scalar_metrics(
    observed: np.ndarray, predicted: np.ndarray, names: Sequence[str]
) -> list[dict[str, Any]]:
    """Per-output Pearson, Spearman, and MSE over the example axis.

    Non-finite pairs are dropped per output and counted, so one output's missing
    observations do not remove an example from the others.

    Parameters
    ----------
    observed, predicted
        ``(examples, outputs)`` arrays in the same units.
    names
        Output names, one per column.

    Returns
    -------
    list of dict
        One row per output with ``output``, ``n``, ``n_excluded``, ``pearson``,
        ``spearman``, and ``mse``. Correlations are NaN below two examples.

    Raises
    ------
    ValueError
        If the shapes disagree, are not two-dimensional, or do not match `names`.
    """
    observed, predicted = np.asarray(observed), np.asarray(predicted)
    if observed.shape != predicted.shape or observed.ndim != 2 or observed.shape[1] != len(names):
        raise ValueError("expected matching (examples, outputs) arrays and output names")
    rows = []
    for i, name in enumerate(names):
        keep = np.isfinite(observed[:, i]) & np.isfinite(predicted[:, i])
        x, y = observed[keep, i], predicted[keep, i]
        rows.append(
            dict(
                output=name,
                n=int(keep.sum()),
                n_excluded=int((~keep).sum()),
                pearson=float(pearson_correlation(x, y)) if len(x) > 1 else np.nan,
                spearman=float(spearman_correlation(x, y)) if len(x) > 1 else np.nan,
                mse=float(np.mean((x - y) ** 2)) if len(x) else np.nan,
            )
        )
    return rows


def pool_channels(values: np.ndarray, groups: Sequence[int], *, mean: bool = False) -> np.ndarray:
    """Reduce a channel axis into one column per group.

    Parameters
    ----------
    values
        ``(examples, channels)`` array whose channel axis the groups partition.
    groups
        Channel count per output, in output order. Summing to `values`'
        channel axis.
    mean
        Average within each group instead of summing. Use this for metrics,
        where a stranded output's two channels should not double its score, and
        summing for signal, where they should add.

    Returns
    -------
    ndarray
        ``(examples, len(groups))``.

    Raises
    ------
    ValueError
        If `groups` does not partition the channel axis.
    """
    values = np.asarray(values)
    if not groups or min(groups) < 1 or sum(groups) != values.shape[1]:
        raise ValueError("groups must partition the channel axis")
    offset, blocks = 0, []
    for size in groups:
        block = values[:, offset : offset + size]
        blocks.append(block.mean(axis=1) if mean else block.sum(axis=1))
        offset += size
    return np.stack(blocks, axis=1)


def profile_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    smooth_true: bool = False,
    smooth_predictions: bool = False,
    kernel_sigma: float = 7,
    kernel_width: int = 81,
) -> dict[str, np.ndarray]:
    """Per-example, per-channel profile metrics.

    Parameters
    ----------
    observed, predicted
        ``(examples, channels, positions)`` arrays of finite non-negative
        signal. Predictions are normalized to a distribution internally.
    smooth_true, smooth_predictions
        Apply a Gaussian kernel before scoring. MNLL always uses unsmoothed
        observations, and upstream smoothing intentionally does not renormalize
        the MNLL input.
    kernel_sigma
        Gaussian standard deviation in positions.
    kernel_width
        Kernel support, which must be odd and positive.

    Returns
    -------
    dict of ndarray
        ``profile_mnll``, ``profile_jsd``, ``profile_pearson``, and
        ``profile_spearman``, each ``(examples, channels)``. JSD is divergence
        in natural-log units, *not* its square root. Correlations of constant
        profiles are zero, matching existing profile reports. A channel with
        zero observed or predicted mass has undefined JSD, reported as NaN.

    Raises
    ------
    ValueError
        If the shapes disagree, the signal is negative or non-finite, or the
        smoothing parameters are invalid.
    """
    y, p = np.asarray(observed, dtype=float), np.asarray(predicted, dtype=float)
    if y.shape != p.shape or y.ndim != 3 or y.shape[-1] == 0:
        raise ValueError("profiles must match (examples, channels, positions)")
    if not np.isfinite(y).all() or not np.isfinite(p).all() or (y < 0).any() or (p < 0).any():
        raise ValueError("profiles must contain finite non-negative signal")
    mass = p.sum(-1, keepdims=True)
    p = np.divide(p, mass, out=np.zeros_like(p), where=mass > 0)

    def smooth(a: np.ndarray) -> np.ndarray:
        if kernel_width < 1 or kernel_width % 2 != 1 or kernel_sigma <= 0:
            raise ValueError("Gaussian smoothing needs positive sigma and odd positive width")
        offsets = np.arange(kernel_width) - (kernel_width - 1) / 2
        kernel = np.exp(-0.5 * (offsets / kernel_sigma) ** 2)
        return convolve1d(a, kernel / kernel.sum(), axis=-1, mode="constant")

    p_score = smooth(p) if smooth_predictions else p
    y_score = smooth(y) if smooth_true else y
    # Upstream smoothing intentionally does not renormalize the MNLL input.
    mnll = -gammaln(y.sum(-1) + 1) + gammaln(y + 1).sum(-1) - xlogy(y, p_score).sum(-1)
    p_sum, y_sum = p_score.sum(-1, keepdims=True), y_score.sum(-1, keepdims=True)
    pp = np.divide(p_score, p_sum, out=np.zeros_like(p_score), where=p_sum > 0)
    yy = np.divide(y_score, y_sum, out=np.zeros_like(y_score), where=y_sum > 0)
    midpoint = (pp + yy) / 2
    jsd = (rel_entr(pp, midpoint).sum(-1) + rel_entr(yy, midpoint).sum(-1)) / 2
    jsd[(y_sum[..., 0] == 0) | (p_sum[..., 0] == 0)] = np.nan
    return dict(
        profile_mnll=mnll,
        profile_jsd=jsd,
        profile_pearson=pearson_correlation(p_score, y_score, constant=0.0),
        # Ranked on the unsmoothed distributions, matching the existing reports.
        profile_spearman=spearman_correlation(p, y, constant=0.0),
    )


def window_mask(
    coords: pd.DataFrame,
    width: int,
    *,
    central: int | None = None,
    peak_body: bool = False,
) -> np.ndarray:
    """Boolean mask selecting scored positions within a centered output window.

    Parameters
    ----------
    coords
        One row per example. Needs ``start`` and ``end`` columns only when
        `peak_body` is set; otherwise just its length is used.
    width
        Output window width the mask indexes.
    central
        Keep only this many positions at the center. Must fit inside `width`
        with matching parity, so the flanks are equal.
    peak_body
        Keep only positions inside each example's own interval, which varies per
        row. BED coordinates are zero-based and half-open.

    Returns
    -------
    ndarray
        ``(len(coords), width)`` boolean mask.

    Raises
    ------
    ValueError
        If `central` does not fit inside `width` with matching parity.
    """
    n = len(coords)
    mask = np.ones((n, width), dtype=bool)
    positions = np.arange(width)[None, :]
    if central is not None:
        if central < 1 or central > width or (width - central) % 2:
            raise ValueError("central window must fit and have matching parity")
        left = (width - central) // 2
        mask &= (positions >= left) & (positions < left + central)
    if peak_body:
        start, end = coords["start"].to_numpy(), coords["end"].to_numpy()
        window_start = start + (end - start) // 2 - width // 2
        mask &= (positions >= (start - window_start)[:, None]) & (
            positions < (end - window_start)[:, None]
        )
    return mask
