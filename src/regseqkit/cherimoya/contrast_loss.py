"""Add pairwise log1p-count contrast errors to Cherimoya's count loss.

Pairs address count-output groups, rather than individual profile channels.
Each pair's mean squared contrast error is split equally between its two
outputs before Cherimoya applies its learned count-loss weights. Shared outputs
accumulate contributions from every pair containing them.

Training and validation Count MSE include the extra term; checkpoint selection
by Count Pearson does not. Like the native count loss, the contrast uses all
examples, including negatives.

A fit JSON can specify::

    "contrast": {"weight": 10, "pairs": [[1, 0], [2, 0]]}

The CLI temporarily patches the private loss hook used by Cherimoya 0.2.0.
Its integration must be checked when upgrading Cherimoya.

Usage::

    .venv/bin/python -m regseqkit.cherimoya.contrast_loss -p NAME.fit.json
"""

from __future__ import annotations

import argparse
import json
import math
import operator
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
import cherimoya.cherimoya
from cherimoya.io import normalize_signal_groups
from cherimoya.losses import _mixture_loss


def with_contrast(pairs: Sequence[tuple[int, int]], weight: float) -> Callable:
    """Augment Cherimoya's mixture loss with pairwise count contrasts.

    Parameters
    ----------
    pairs : Sequence[tuple[int, int]]
        Count-output indices. A pair's extra loss is split equally between its
        outputs; shared outputs accumulate contributions from multiple pairs.
    weight : float
        Finite nonnegative multiplier. Zero returns the native losses.

    Returns
    -------
    loss : callable
        Function with the native mixture-loss signature and return shapes.
        Observed counts are summed within each signal group before log1p.
    """
    pairs = [(operator.index(left), operator.index(right)) for left, right in pairs]
    weight = float(weight)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("contrast weight must be finite and nonnegative")
    if any(left < 0 or right < 0 for left, right in pairs):
        raise ValueError("contrast indices must be nonnegative")

    def mixture_loss(y, y_hat_logits, y_hat_logcounts, labels=None, signal_groups=None):
        profile_loss, count_loss = _mixture_loss(
            y, y_hat_logits, y_hat_logcounts, labels=labels, signal_groups=signal_groups
        )
        if weight == 0 or not pairs:
            return profile_loss, count_loss
        groups = signal_groups if signal_groups is not None else [1] * y.shape[1]
        totals = torch.stack(
            [block.sum(dim=(1, 2)) for block in y.split(tuple(groups), dim=1)], dim=1
        )
        residual = totals.log1p() - y_hat_logcounts
        extra = torch.zeros_like(count_loss)
        for left, right in pairs:
            if max(left, right) >= residual.shape[1]:
                raise ValueError("contrast index exceeds the number of count outputs")
            contrast_sq = ((residual[:, left] - residual[:, right]) ** 2).mean()
            extra[left] = extra[left] + weight / 2 * contrast_sq
            extra[right] = extra[right] + weight / 2 * contrast_sq
        return profile_loss, count_loss + extra

    return mixture_loss


def install(pairs: Sequence[tuple[int, int]], weight: float) -> None:
    """Replace the process-wide loss hook called by Cherimoya.fit."""
    cherimoya.cherimoya._mixture_loss = with_contrast(pairs, weight)


def read_block(fit: dict) -> tuple[list[tuple[int, int]], float]:
    """Read count-output pairs and their weight from a fit JSON.

    Parameters
    ----------
    fit : dict
        Fit settings containing signals and a contrast block. Pairs are
        two-element index lists. Older dictionaries with an indices field are
        also accepted; descriptive labels do not determine output identity.

    Returns
    -------
    pairs : list[tuple[int, int]]
        Valid indices into the count-output groups.
    weight : float
        Finite nonnegative multiplier.
    """
    block = fit["contrast"]
    groups = normalize_signal_groups(fit["signals"])[1]
    pairs = []
    for pair in block["pairs"]:
        indices = pair["indices"] if isinstance(pair, dict) else pair
        left, right = (operator.index(index) for index in indices)
        if not (0 <= left < len(groups) and 0 <= right < len(groups)):
            raise ValueError("contrast indices must address existing count-output groups")
        pairs.append((left, right))
    weight = float(block["weight"])
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("contrast weight must be finite and nonnegative")
    return pairs, weight


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-p", "--parameters", required=True, help="Fit JSON with a contrast block.")
    args = parser.parse_args()
    fit = json.loads(Path(args.parameters).read_text())
    pairs, weight = read_block(fit)
    print(f"contrast term: weight {weight:g} on count-output pairs {pairs}", flush=True)

    from cherimoya_cli.commands import fit as fit_command

    original = cherimoya.cherimoya._mixture_loss
    try:
        install(pairs, weight)
        fit_command.run(argparse.Namespace(parameters=args.parameters))
    finally:
        cherimoya.cherimoya._mixture_loss = original


if __name__ == "__main__":
    main()
