"""Greedy single-base search and per-round evaluations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math

import torch

from ..mutagenesis import _get_positions, single_site_saturation_mutagenesis
from ._validation import _validate_design, _validate_loss
from .objectives import ObjectiveProtocol


@dataclass(frozen=True)
class GreedyIteration:
    """One evaluation and selection, with detached CPU tensors.

    ``candidate_scores`` is ``(positions, 4, scorers)``; ``candidate_losses``
    is ``(positions, 4)``. Rows follow ascending editable-position order;
    base columns follow ACGT order. Reference entries contain the current
    values, but are excluded from selection. ``mutation`` describes an
    accepted edit, e.g. ``"4A>C"``, using a zero-based full-input coordinate.
    A rejected proposal has ``mutation=None`` and leaves the selected
    sequence, scores, and loss equal to their current values.
    """

    iteration: int
    current_sequence: torch.Tensor
    current_scores: torch.Tensor
    current_loss: float
    candidate_scores: torch.Tensor
    candidate_losses: torch.Tensor
    mutation: str | None
    accepted: bool
    selected_sequence: torch.Tensor
    selected_scores: torch.Tensor
    selected_loss: float
    stop_reason: str | None


def greedy_substitution(
    module: torch.nn.Module,
    template: torch.Tensor,
    objective: ObjectiveProtocol,
    positions: Sequence[int] | torch.Tensor | None = None,
    *,
    max_iter: int = 20,
    batch_size: int = 128,
    tol: float = 1e-3,
    device: str = "cpu",
    on_iteration: Callable[[GreedyIteration], None] | None = None,
) -> torch.Tensor:
    """Accept the best single-base edit when improvement strictly exceeds tol.

    Each round evaluates the three alternative bases at every editable site.
    Ties prefer ascending position, then ACGT order. Previously edited sites
    remain editable. Reference predictions are recomputed each round.

    Parameters
    ----------
    module
        The objective's :class:`~regseqkit.scoring.ScoreModule`, returning
        ``(N, len(objective.scorers))`` scores in objective scorer order.
    template
        ``(1, 4, length)`` hard one-hot ACGT starting sequence.
    objective
        Supplies one finite, lower-is-better loss per candidate. Must already
        have numeric targets for any match objectives before optimization.
    positions : Sequence[int] or torch.Tensor or None, default None
        Sorted, unique, zero-based full-input positions where substitutions
        are allowed. None selects all positions; an empty sequence allows no
        edits. Use range(start, end) for an interval. Other positions retain
        their template bases.
    max_iter
        Maximum number of accepted single-base substitutions. Repeated edits
        at one site count separately. Zero returns the starting sequence
        after evaluating its baseline.
    batch_size
        Maximum number of sequences predicted per inference batch. All mutants
        for the current sequence are allocated on CPU before prediction.
        Must be a positive integer.
    tol
        Finite, nonnegative minimum loss improvement. An edit is accepted
        only when improvement is strictly greater than this value.
    device
        Torch device string used for model predictions.
    on_iteration : Callable[[GreedyIteration], None] or None, default None
        Called after each evaluated round, including a rejected final proposal.
        Receives a GreedyIteration with detached CPU tensors and the selection
        outcome. Empty positions or a zero budget produce no records; no extra
        evaluation follows budget exhaustion. Callback exceptions propagate.
        To collect history::
            history = []
            designed = greedy_substitution(
                module, template, objective, on_iteration=history.append
            )

    Returns
    -------
    torch.Tensor
        ``(1, 4, length)`` detached CPU hard one-hot design, asserted to have
        left every locked position untouched. The template is not modified.

    Raises
    ------
    ValueError
        If the template or positions are invalid, search settings are
        invalid, the module's outputs do not match the objective's scorer
        count, predictions are non-finite, or the objective does not return
        one finite loss per candidate.
    """
    if template.ndim != 3 or template.shape[0] != 1 or template.shape[1] != 4:
        raise ValueError("greedy substitution requires one (1, 4, length) template")
    positions = _get_positions(positions, template.shape[-1])
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 0:
        raise ValueError("max_iter must be a nonnegative integer")
    if not math.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")

    current = template.detach().cpu().clone()
    editable = torch.zeros(template.shape[-1], dtype=torch.bool)
    editable[positions] = True
    _validate_design(current, current, editable)

    scores, _ = single_site_saturation_mutagenesis(
        module,
        current,
        positions=[],
        batch_size=batch_size,
        device=device,
    )
    if scores.shape[1] != len(objective.scorers):
        raise ValueError("module must return only the objective's scorers, in order")

    loss = objective(scores)
    _validate_loss(loss, n_examples=len(scores))
    current_loss = float(loss[0])

    for iteration in range(max_iter):
        if not len(positions):
            break
        scores, candidates = single_site_saturation_mutagenesis(
            module,
            current,
            positions=positions,
            batch_size=batch_size,
            device=device,
        )
        loss = objective(scores)
        _validate_loss(loss, n_examples=len(scores))
        current_loss = float(loss[0])
        candidates = candidates[0]
        candidate_scores = candidates.reshape(-1, scores.shape[1])
        losses = objective(candidate_scores)
        _validate_loss(losses, n_examples=len(candidate_scores))
        losses = losses.detach().cpu()
        losses = losses.reshape(len(positions), 4)
        selectable = losses.clone()
        selectable[torch.arange(len(positions)), current[0].argmax(0)[positions]] = torch.inf
        chosen = int(selectable.flatten().argmin())
        site, base = divmod(chosen, 4)
        position = int(positions[site])
        proposed_loss = float(losses[site, base])
        accepted = current_loss - proposed_loss > tol
        mutation = None
        result = current.clone()
        result_scores, result_loss = scores, current_loss
        if accepted:
            reference = int(current[0, :, position].argmax())
            mutation = f"{position}{'ACGT'[reference]}>{'ACGT'[base]}"
            result[0, :, position] = 0
            result[0, base, position] = 1
            result_scores = candidates[site, base].unsqueeze(0).clone()
            result_loss = proposed_loss
        stop_reason = (
            "insufficient_improvement"
            if not accepted
            else "max_iter"
            if iteration + 1 == max_iter
            else None
        )
        if on_iteration is not None:
            on_iteration(
                GreedyIteration(
                    iteration=iteration,
                    current_sequence=current.clone(),
                    current_scores=scores.clone(),
                    current_loss=current_loss,
                    candidate_scores=candidates.clone(),
                    candidate_losses=losses.clone(),
                    mutation=mutation,
                    accepted=accepted,
                    selected_sequence=result.clone(),
                    selected_scores=result_scores.clone(),
                    selected_loss=result_loss,
                    stop_reason=stop_reason,
                )
            )
        current, scores, current_loss = result, result_scores, result_loss
        if not accepted:
            break
    _validate_design(template, current, editable)
    return current
