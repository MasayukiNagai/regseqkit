"""Batched prediction and model-independent output conversions."""

from collections.abc import Sequence

import torch
from tangermeme.predict import predict

# Keep recovered counts finite in float32, including expected profiles.
MAX_LOG_COUNT = 88.0


def log1p_to_counts(values: torch.Tensor) -> torch.Tensor:
    """Recover nonnegative counts from log1p predictions.

    Parameters
    ----------
    values : torch.Tensor
        Log1p predictions, in any shape. Values above 88 are clipped before
        exponentiation to avoid float32 overflow.

    Returns
    -------
    counts : torch.Tensor
        Recovered counts with the same shape, device and dtype as values.
    """
    return values.clamp(max=MAX_LOG_COUNT).expm1().clamp(min=0)


def compute_signal(
    logits: torch.Tensor,
    scalars: torch.Tensor,
    groups: Sequence[int],
) -> torch.Tensor:
    """Convert profile logits and log1p counts into per-position signal.

    Each count output represents the total signal over a contiguous set of
    profile channels. Softmax normalizes jointly over those channels and their
    positions, then the recovered count is distributed using those probabilities.

    Parameters
    ----------
    logits : torch.Tensor, shape (N, channels, positions)
        Profile logits.
    scalars : torch.Tensor, shape (N, outputs)
        Log1p total counts, one for each channel group described by groups.
    groups : Sequence[int]
        Number of consecutive profile channels sharing each count output,
        in channel order. For example, [1, 2] means scalars[:, 0] supplies
        the total for logits[:, :1], and scalars[:, 1] supplies the combined
        total for logits[:, 1:3]. An unstranded track uses 1; a stranded
        (+, -) pair uses 2 because its strands share one total count. The
        number of entries must equal scalars.shape[1], and their sum must
        equal logits.shape[1]. Usually supplied as model.signal_groups.

    Returns
    -------
    signal : torch.Tensor, shape (N, channels, positions)
        Nonnegative signal. Summing over a group's channels and positions
        recovers its count, with negative recovered counts clipped to zero.
        Float16 and bfloat16 inputs are promoted to float32 before softmax
        and count recovery.
    """
    if logits.ndim != 3 or scalars.ndim != 2:
        raise ValueError("expected 3D profile logits and 2D log counts")
    if logits.shape[0] != scalars.shape[0] or scalars.shape[1] != len(groups):
        raise ValueError("batch and count-output dimensions must match the signal groups")
    if not groups or any(size <= 0 for size in groups):
        raise ValueError("signal groups must contain positive channel counts")
    if sum(groups) != logits.shape[1]:
        raise ValueError("signal groups must cover all profile channels")
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    if scalars.dtype in (torch.float16, torch.bfloat16):
        scalars = scalars.float()
    profiles = []
    for i, block in enumerate(logits.split(tuple(groups), dim=1)):
        probability = block.flatten(1).softmax(-1).reshape_as(block)
        total = log1p_to_counts(scalars[:, i])
        profiles.append(probability * total[:, None, None])
    return torch.cat(profiles, dim=1)
