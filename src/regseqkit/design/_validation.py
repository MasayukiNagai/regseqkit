"""Loss and hard-sequence contracts shared by optimizers."""

import torch


def _validate_loss(loss: torch.Tensor, n_examples: int) -> None:
    """Validate one finite loss per candidate.

    Parameters
    ----------
    loss : torch.Tensor, shape (n_examples,)
        Per-candidate objective losses.
    n_examples : int
        Expected number of candidates.

    Raises
    ------
    ValueError
        If loss is not a tensor of the expected shape or contains non-finite values.
    """
    if not isinstance(loss, torch.Tensor) or loss.shape != (n_examples,):
        raise ValueError("objective must return one loss per example")
    if not torch.isfinite(loss).all():
        raise ValueError("objective returned a non-finite loss")


def _validate_design(template: torch.Tensor, designed: torch.Tensor, editable: torch.Tensor) -> None:
    """Validate that an optimizer returned hard one-hot and touched nothing locked.

    Parameters
    ----------
    template : torch.Tensor, shape (1, 4, length)
        Starting sequence.
    designed : torch.Tensor, shape (N, 4, length)
        Candidate sequences.
    editable : torch.Tensor, shape (length,)
        Boolean mask of positions the optimizer was allowed to edit.

    Raises
    ------
    ValueError
        If the shape is wrong, the output is not hard one-hot, or any locked
        position differs from the template.
    """
    if designed.ndim != 3 or designed.shape[1:] != template.shape[1:]:
        raise ValueError("optimizer returned an unexpected sequence shape")
    if not (((designed == 0) | (designed == 1)).all() and (designed.sum(1) == 1).all()):
        raise ValueError("optimizer must return hard one-hot sequences")
    mask = ~editable.to(designed.device)
    if not torch.equal(
        designed[:, :, mask],
        template.to(designed.device).expand_as(designed)[:, :, mask],
    ):
        raise ValueError("optimizer edited a locked position")
