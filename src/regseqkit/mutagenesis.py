"""Single-site mutation predictions and attribution transforms."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from tqdm import trange
from .inference import predict


def _validate_predictions(
    preds: torch.Tensor | list[torch.Tensor],
    n: int,
) -> torch.Tensor:
    """Validate predictions and return detached CPU float32 scores."""
    if (
        not isinstance(preds, torch.Tensor)
        or preds.ndim != 2
        or preds.shape[0] != n
        or preds.shape[1] == 0
    ):
        raise ValueError("model predictions must have shape (N, outputs)")
    preds = preds.detach().cpu().float()
    if not torch.isfinite(preds).all():
        raise ValueError("model predictions must be finite")
    return preds


def _get_positions(
    positions: Sequence[int] | torch.Tensor | None,
    length: int,
) -> torch.Tensor:
    """Resolve and validate sequence positions."""
    positions = (
        torch.arange(length) if positions is None else torch.as_tensor(positions).detach().cpu()
    )
    if positions.ndim != 1 or (
        positions.numel()
        and positions.dtype
        not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        )
    ):
        raise ValueError("positions must be a sorted, unique integer vector")
    positions = positions.long()
    if positions.numel() and (
        positions[0] < 0 or positions[-1] >= length or (positions[1:] <= positions[:-1]).any()
    ):
        raise ValueError("positions must be sorted, unique, and within the input")
    return positions


def single_site_saturation_mutagenesis(
    model: torch.nn.Module,
    X: torch.Tensor,
    *,
    positions: Sequence[int] | torch.Tensor | None = None,
    batch_size: int = 128,
    device: str = "cpu",
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict all single-base substitutions at the requested positions.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model returning a finite tensor of shape (batch, outputs)
        with at least one output. Evaluated through regseqkit.inference.predict.
        To select specific output tracks or reduce model outputs, pass a
        wrapped model whose forward method performs that selection or
        reduction and returns the required (batch, outputs) tensor.
    X : torch.Tensor, shape (N, 4, length)
        Nonempty hard one-hot sequences in ACGT order.
        Each position must contain exactly one 1 and three 0s. The input is
        detached and copied to CPU when needed; its values are not modified.
    positions : Sequence[int] or torch.Tensor or None, default None
        Sorted, unique, zero-based integer coordinates.
        None means all positions in an ascending order.
        Positions are explicit coordinates, not start/end bounds. For a
        contiguous interval, pass a range directly;
            positions=range(start, end)
            positions=list(range(start, end))
    batch_size : int, default 128
        Positive integer limiting the inference batch size. All three
        alternative bases at every requested position are materialized on CPU
        for one reference sequence at a time, then passed to regseqkit.inference.predict.
    device : str, default "cpu"
        Inference device passed to regseqkit.inference.predict.
    verbose : bool, default False
        Show a progress bar over input sequences when True. The bar advances
        after all mutations for each sequence are evaluated.

    Returns
    -------
    ref_preds : torch.Tensor, shape (N, outputs)
        Detached CPU float32 reference predictions.
    mut_preds : torch.Tensor, shape (N, P, 4, outputs)
        Detached CPU float32 absolute predictions. P is the number of requested
        positions (the input length by default), and the base axis is ACGT.
        Entries for the reference base reuse its reference prediction.
        Mutation effects are computed as::

            mut_preds - ref_preds[:, None, None, :]

    Raises
    ------
    ValueError
        If sequences are empty or not hard one-hot ACGT, positions are invalid,
        the batch size is not a positive integer, or reference/mutant
        predictions have invalid shapes or nonfinite values.
    """
    if X.ndim != 3 or X.shape[0] == 0 or X.shape[1] != 4 or X.shape[2] == 0:
        raise ValueError("expected nonempty (N, 4, length) sequences")
    if not (((X == 0) | (X == 1)).all() and (X.sum(1) == 1).all()):
        raise ValueError("single-site mutagenesis requires hard one-hot ACGT sequences")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    positions = _get_positions(positions, X.shape[-1])

    X = X.detach().cpu()
    ref_preds = predict(model, X, batch_size=batch_size, device=device)
    ref_preds = _validate_predictions(ref_preds, len(X))

    mut_preds = ref_preds[:, None, None, :].expand(-1, len(positions), 4, -1).clone()
    if len(positions) == 0:
        return ref_preds, mut_preds

    for index in trange(len(X), disable=not verbose, desc="saturation mutagenesis"):
        ref_onehot = X[index]
        ref_bases = ref_onehot.argmax(0)[positions]
        # Each mutant indexes one entry in positions and one replacement base.
        pos_indices, alt_bases = torch.where(ref_bases[:, None] != torch.arange(4)[None, :])
        mut_onehot = ref_onehot.expand(len(pos_indices), -1, -1).clone()
        copy_ids = torch.arange(len(pos_indices))
        mut_onehot[copy_ids, :, positions[pos_indices]] = 0
        mut_onehot[copy_ids, alt_bases, positions[pos_indices]] = 1
        preds = predict(model, mut_onehot, batch_size=batch_size, device=device)
        preds = _validate_predictions(preds, len(pos_indices))
        mut_preds[index, pos_indices, alt_bases] = preds
        del mut_onehot
    return ref_preds, mut_preds


def ssm_attribution(
    ref_preds: torch.Tensor,
    mut_preds: torch.Tensor,
    *,
    center: bool = True,
    hypothetical: bool = True,
    X: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert single-site mutation predictions into attribution scores.

    Parameters
    ----------
    ref_preds : torch.Tensor, shape (N, outputs)
        Reference predictions from single_site_saturation_mutagenesis.
    mut_preds : torch.Tensor, shape (N, P, 4, outputs)
        Absolute mutant predictions in ACGT order. Inputs are not modified.
    center : bool, default True
        Subtract the mean effect across ACGT at each position. False returns
        mutant-minus-reference effects, with zero at the reference base.
    hypothetical : bool, default True
        Return scores for every possible base. False masks scores to the
        observed bases using X, retaining the same output shape.
    X : torch.Tensor or None, shape (N, 4, P), default None
        One-hot sequences restricted to the mutated positions, in the same
        order as mut_preds. Required when hypothetical is False. For a window,
        pass X[:, :, start:end]; for selected sites, pass X[:, :, positions].

    Returns
    -------
    attributions : torch.Tensor, shape (N, outputs, 4, P)
        Mutation effects, optionally mean-centered and projected onto observed
        bases. This helper performs no model evaluation or precision changes.
    """
    effects = mut_preds - ref_preds[:, None, None, :]
    if center:
        effects = effects - effects.mean(dim=2, keepdim=True)
    attributions = effects.permute(0, 3, 2, 1).contiguous()
    if not hypothetical:
        if X is None:
            raise ValueError("X is required for observed-base attributions")
        attributions = attributions * X.to(attributions)[:, None, :, :]
    return attributions
