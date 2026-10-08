"""A contrast term folded into cherimoya's count loss.

The design objective is a difference of two count outputs within one
experiment, ``logcount_US - logcount_CTRL``. cherimoya's count loss is a mean
squared error per track, and for a US/CONTROL pair the two track terms sum to
half the squared error on the *level* plus half the squared error on the
*contrast*, because the contrast residual is the difference of the two track
residuals. At convergence the level part is about twenty times the contrast
part, so the contrast sees a few percent of the count gradient.

``with_contrast`` adds ``weight`` times the contrast's mean squared error back
into the count loss vector, half onto each track of the pair. cherimoya's fit
loop multiplies that vector by its learned per-track count weights and sums,
so the added term is weighted like a count term: ``weight == 1`` makes one
pair's contrast worth one track's count MSE. Everything else -- the Kendall
weights, logging, checkpoint selection, saving, the post-fit evaluation -- runs
as shipped, because the loop's only contact with the loss is the module-level
name it calls. ``main`` rebinds that name and then runs the stock ``fit``
command in-process on the same fit JSON, whose ``contrast`` block the CLI's
parameter merge carries through untouched.

Two consequences to read the logs with. The training and validation ``Count
MSE`` columns include the term; ``Count Pearson``, the selection criterion,
does not. And the term runs over the whole batch, negatives included, exactly
as the count loss does: the loop passes no labels to the loss.

The fit JSON block::

    "contrast": {
      "weight": 10,
      "pairs": [{"us": "US108.second", "control": "CONTROL.second", "indices": [1, 0]}, ...]
    }

``indices`` address the model's count outputs in ``signals`` order; the track
names beside them are asserted against the bigWig stems before anything runs.

Written against cherimoya 0.2.0, the version pinned in uv.lock.

Usage::

    .venv/bin/python -m regseqkit.cherimoya.contrast_loss -p NAME.fit.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
import cherimoya.cherimoya
from cherimoya.losses import _mixture_loss


def with_contrast(pairs: Sequence[tuple[int, int]], weight: float) -> Callable:
    """cherimoya's mixture loss with ``weight`` x contrast MSE folded into the counts.

    Parameters
    ----------
    pairs
        ``(us, control)`` count-output indices. Each pair's term is split half
        onto each of its two tracks, so a CONTROL shared by several pairs
        collects half of each.
    weight
        Multiplier on the pair's contrast MSE. Zero returns the stock losses.

    Returns
    -------
    callable
        Same signature and return shape as ``cherimoya.losses._mixture_loss``.
    """
    pairs = [(int(us), int(control)) for us, control in pairs]
    weight = float(weight)

    def mixture_loss(y, y_hat_logits, y_hat_logcounts, labels=None, signal_groups=None):
        profile_loss, count_loss = _mixture_loss(
            y, y_hat_logits, y_hat_logcounts, labels=labels, signal_groups=signal_groups
        )
        if y.shape[1] != y_hat_logcounts.shape[1]:
            raise ValueError(
                "the contrast term needs one count output per track, but y has "
                f"{y.shape[1]} tracks and the count head {y_hat_logcounts.shape[1]} outputs"
            )
        # The count target minus the prediction, per example and track; the
        # contrast residual is the difference of two of these columns.
        residual = torch.log(y.sum(dim=-1) + 1) - y_hat_logcounts
        extra = torch.zeros_like(count_loss)
        for us, control in pairs:
            contrast_sq = ((residual[:, us] - residual[:, control]) ** 2).mean()
            extra[us] = extra[us] + weight / 2 * contrast_sq
            extra[control] = extra[control] + weight / 2 * contrast_sq
        return profile_loss, count_loss + extra

    return mixture_loss


def install(pairs: Sequence[tuple[int, int]], weight: float) -> None:
    """Make ``Cherimoya.fit`` call the contrast-augmented loss."""
    cherimoya.cherimoya._mixture_loss = with_contrast(pairs, weight)


def read_block(fit: dict) -> tuple[list[tuple[int, int]], float]:
    """The pairs and weight of a fit JSON's ``contrast`` block, names checked."""
    block = fit["contrast"]
    stems = [Path(signal).stem for signal in fit["signals"]]
    pairs = []
    for pair in block["pairs"]:
        us, control = (int(i) for i in pair["indices"])
        if (stems[us], stems[control]) != (pair["us"], pair["control"]):
            raise ValueError(
                f"contrast pair {pair} does not match the fit JSON's signals: "
                f"index {us} is {stems[us]!r} and index {control} is {stems[control]!r}"
            )
        pairs.append((us, control))
    return pairs, float(block["weight"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-p",
        "--parameters",
        required=True,
        help="A cherimoya fit JSON carrying a `contrast` block.",
    )
    args = parser.parse_args()

    fit = json.loads(Path(args.parameters).read_text())
    pairs, weight = read_block(fit)
    install(pairs, weight)
    names = ", ".join(f"{p['us']} - {p['control']}" for p in fit["contrast"]["pairs"])
    print(f"contrast term: weight {weight:g} on {len(pairs)} pair(s): {names}", flush=True)

    from cherimoya_cli.commands import fit as fit_command

    fit_command.run(argparse.Namespace(parameters=args.parameters))


if __name__ == "__main__":
    main()
