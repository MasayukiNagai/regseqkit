"""Adapter for gradient-based sequence design with Ledidi."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from tangermeme.predict import predict

from .objectives import ObjectiveProtocol
from ._validation import _validate_design, _validate_loss
from ..mutagenesis import _get_positions


def ledidi_design(
    module: torch.nn.Module,
    template: torch.Tensor,
    objective: ObjectiveProtocol,
    positions: Sequence[int] | torch.Tensor | None = None,
    *,
    max_iter: int = 1000,
    batch_size: int = 16,
    n_samples: int | None = None,
    l: float = 0.1,  # noqa: E741 - Ledidi's own keyword; renaming would hide the mapping
    random_state: int = 0,
    device: str = "cpu",
) -> torch.Tensor:
    """Gradient-based design with Ledidi at the selected positions.

    Parameters
    ----------
    module
        A differentiable ScoreModule returning scores in objective.scorers order.
    template
        ``(1, 4, length)`` starting sequence.
    objective
        Supplies the output loss.
    positions : Sequence[int] or torch.Tensor or None, default None
        Sorted, unique, zero-based full-input positions where substitutions
        are allowed. None selects all positions; an empty sequence allows no
        edits. Use range(start, end) for an interval. Other positions retain
        their template bases.
    max_iter
        Optimization steps.
    batch_size
        Candidate sequences sampled per optimization step.
    n_samples
        Number of fresh sequences to draw after optimization. Samples share
        one learned distribution and may repeat. None preserves Ledidi's
        default of returning the best recorded optimization batch.
    l
        Edit penalty, named for Ledidi's own keyword so the two map one to one.
        It discourages edits but is not a hard edit-count budget.
    random_state
        Seed, making a run reproducible.
    device
        Torch device string.

    Returns
    -------
    torch.Tensor
        ``(N, 4, length)`` hard one-hot designs, asserted to have left every
        locked position untouched.

    Raises
    ------
    ValueError
        If the template, positions or sample count is invalid.
    RuntimeError
        If Ledidi is not installed.
    """
    if n_samples is not None and (
        isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1
    ):
        raise ValueError("n_samples must be a positive integer or None")
    if template.ndim != 3 or template.shape[0] != 1 or template.shape[1] != 4:
        raise ValueError("Ledidi design requires one (1, 4, length) template")
    positions = _get_positions(positions, template.shape[-1])
    editable = torch.zeros(template.shape[-1], dtype=torch.bool)
    editable[positions] = True
    _validate_design(template, template, editable)
    try:
        from ledidi import ledidi
    except ImportError as exc:
        raise RuntimeError(
            "Ledidi is not installed; install the declared design "
            "dependency before running this method"
        ) from exc

    # Ledidi calls output_loss(predicted, desired) and wants a scalar.
    def output_loss(predicted: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
        loss = objective(predicted)
        _validate_loss(loss, n_examples=len(predicted))
        return loss.mean()

    y = predict(module, template, device=device).to(device)
    result = ledidi(
        module,
        template.to(device),
        torch.zeros_like(y),
        output_loss=output_loss,
        input_mask=~editable.to(device),
        max_iter=max_iter,
        batch_size=batch_size,
        n_samples=n_samples,
        l=l,
        random_state=random_state,
        device=device,
        verbose=False,
    )
    result = result.reshape(-1, *template.shape[1:]).detach()
    _validate_design(template, result, editable)
    return result.cpu()
