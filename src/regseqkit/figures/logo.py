"""Batch attribution logos with shared, overridable glyph styling."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from fast_logomaker import BatchLogo


LOGOMAKER_STYLE = dict(
    font_weight="bold",
    width=0.95,
    fade_below=0,
    shade_below=0,
    dont_stretch_more_than="E",
)


def stack_attributions(attributions: Sequence[numpy.ndarray]) -> numpy.ndarray:
    """Center-pad ``(4, L)`` arrays of differing length into one ``(N, 4, L)``."""
    arrays = [numpy.asarray(a) for a in attributions]
    if any(a.ndim != 2 or a.shape[0] != 4 for a in arrays):
        raise ValueError("each attribution must be a (4, length) array")
    width = max(a.shape[1] for a in arrays)
    stacked = numpy.zeros((len(arrays), 4, width), dtype=numpy.float32)
    for i, array in enumerate(arrays):
        left = (width - array.shape[1]) // 2
        stacked[i, :, left : left + array.shape[1]] = array
    return stacked


def _pad_sequences(sequences: Sequence[str], width: int, alphabet: Sequence[str]) -> list[str]:
    """Upper-case and center-pad sequences with ``N`` to ``width``, as ``stack_attributions`` pads arrays."""
    letters = {str(char).upper() for char in alphabet} | {"N"}
    padded = []
    for idx, sequence in enumerate(sequences):
        sequence = str(sequence).upper()
        if len(sequence) > width:
            raise ValueError(f"sequence {idx} is {len(sequence)} bp but the logos are {width} wide")
        unknown = set(sequence) - letters
        if unknown:
            raise ValueError(
                f"sequence {idx} has letters {sorted(unknown)} outside the logo alphabet {list(alphabet)}"
            )
        left = (width - len(sequence)) // 2
        padded.append("N" * left + sequence + "N" * (width - len(sequence) - left))
    return padded


def _fade_alternatives(logo: BatchLogo, sequences: Sequence[str], alt_alpha: float) -> None:
    """Scale the opacity of every glyph that is not the sequence's own base by ``alt_alpha``.

    ``sequences`` are already padded to the logo width; ``N`` positions match no glyph,
    so every letter there is faded.
    """
    for idx, sequence in enumerate(sequences):
        for glyph in logo.processed_logos[idx]["glyphs"]:
            if glyph["char"].upper() != sequence[glyph["pos"]]:
                glyph["alpha"] *= alt_alpha


def plot_logos(
    attributions: numpy.ndarray | Sequence[numpy.ndarray],
    titles: Sequence[str] | None = None,
    *,
    sequences: Sequence[str] | None = None,
    alt_alpha: float = 0.45,
    center_values: bool = False,
    figsize: tuple[float, float] = (18, 2.2),
    rows: int | None = None,
    cols: int | None = None,
    **logo_kwargs: Any,
) -> tuple[Figure, list[Axes]]:
    """Draw per-base attribution logos for a batch of ``(N, 4, L)`` scores.

    The one figure function here that does not take an ``ax``: the underlying
    batch renderer builds its own grid of axes, and returns them, so a caller
    annotates the axes it gets back. Accepts a list of ``(4, L)`` arrays, which
    are center-padded to a common width. ``logo_kwargs`` override the shared
    ``LOGOMAKER_STYLE`` defaults and are forwarded to ``BatchLogo``.

    ``sequences`` (one per logo) marks the wild-type base at each position: it is
    drawn at full opacity and the other letters at ``alt_alpha``, which suits a
    hypothetical attribution (for example mean-centred ISM) with all four bases
    shown. Pass ``alt_alpha=1`` to skip the fading. Sequences shorter than the logo
    width are center-padded with ``N`` like the attributions, and every letter at a
    padded or ``N`` position is faded. The padded sequences are also passed to
    ``BatchLogo``, so ``contribution=True`` works through ``logo_kwargs``.
    """
    if isinstance(attributions, (list, tuple)):
        attributions = stack_attributions(attributions)
    attributions = numpy.asarray(attributions)
    if attributions.ndim != 3 or attributions.shape[1] != 4:
        raise ValueError("attributions must have shape (N, 4, length)")
    n = attributions.shape[0]
    if sequences is not None and len(sequences) != n:
        raise ValueError(f"got {len(sequences)} sequences for {n} logos")
    if not 0 <= alt_alpha <= 1:
        raise ValueError("alt_alpha must be in [0, 1]")
    if rows is None and cols is None:
        rows, cols = n, 1
    elif rows is None:
        rows = (n + cols - 1) // cols
    elif cols is None:
        cols = (n + rows - 1) // rows
    options = (
        LOGOMAKER_STYLE
        | dict(
            alphabet=list("ACGT"),
            figsize=list(figsize),
            center_values=center_values,
            show_progress=False,
            batch_size=1,
            color_scheme="classic",
        )
        | logo_kwargs
    )
    if sequences is not None:
        sequences = _pad_sequences(sequences, attributions.shape[2], options["alphabet"])
        options["sequences"] = sequences
    logo = BatchLogo(attributions.transpose(0, 2, 1), **options)
    logo.process_all()
    if sequences is not None and alt_alpha < 1:
        _fade_alternatives(logo, sequences, alt_alpha)
    fig, axes = logo.draw_logos(indices=list(range(n)), rows=rows, cols=cols)
    axes = numpy.asarray(axes, dtype=object).reshape(-1).tolist()
    for ax, title in zip(axes, titles or ()):
        ax.set_title(title, fontsize=10, loc="left")
    return fig, axes
