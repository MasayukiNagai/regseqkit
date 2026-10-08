"""Greedy single-base search, per-round evaluations, and history aggregation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
import re

import numpy as np
import torch

from ..io import add_extension, load_arrays, save_arrays
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
    on_iteration
        Optional callback receiving a :class:`GreedyIteration` after each
        evaluation, including a rejected proposal that ends the search.
        Records contain detached CPU tensors for the pre-edit sequence,
        current scores/loss, all candidate scores/losses, and the selected
        state, plus the accepted mutation, acceptance decision, and stop
        reason. No extra
        evaluation occurs after the iteration budget is exhausted. An empty
        position selection or zero budget produces no callback records. Callback
        exceptions propagate to the caller.

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


_ROUND_FIELDS = {
    "iteration",
    "baseline_loss",
    "position",
    "base",
    "accepted",
    "result_loss",
    "stop_reason",
}


def aggregate_greedy_history(history_dir: str | Path, destination: str | Path) -> Path:
    """Save one NPZ/JSON pair for a single objective/template search.

    Parameters
    ----------
    history_dir : str or pathlib.Path
        Directory containing zero-based iteration.<round>.npz/JSON pairs.
        Completed rounds must be contiguous. Incomplete trailing pairs from
        an interrupted write are ignored; malformed complete pairs raise.
    destination : str or pathlib.Path
        Output artifact stem, without an extension. Existing aggregates at
        this stem are replaced, so aggregation can be repeated independently
        of optimization. The per-round source files are retained.

    Returns
    -------
    pathlib.Path
        Written NPZ path. Candidate arrays have a leading round axis, followed
        by (positions, 4, scorers) for scores or (positions, 4) for
        losses. Sequence arrays are (rounds, 4, length) and baseline/result
        scores are (rounds, scorers). Positions and shared provenance are
        stored once. Scalar losses, proposed edits, acceptance, and stop
        reasons are stored as round-indexed arrays. Empty stop reasons mean
        the search continued. The JSON marks interrupted trajectories with
        complete=false.

    Raises
    ------
    ValueError
        If there are no complete rounds, a round is missing, or round indices,
        positions, shapes, or shared provenance disagree.
    """
    history_dir = Path(history_dir)
    pairs: dict[int, Path] = {}
    for path in history_dir.iterdir():
        match = re.fullmatch(r"iteration\.(\d+)\.npz", path.name)
        if match:
            iteration = int(match[1])
            if iteration in pairs:
                raise ValueError(f"duplicate iteration {iteration} in {history_dir}")
            pairs[iteration] = path
    complete = sorted(
        iteration
        for iteration, path in pairs.items()
        if add_extension(path.with_suffix(""), ".json").is_file()
    )
    if not complete:
        raise ValueError(f"no complete greedy rounds in {history_dir}")
    if complete != list(range(len(complete))):
        raise ValueError(f"greedy rounds must be contiguous from zero in {history_dir}")

    collected: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "sequence",
            "baseline_scores",
            "candidate_scores",
            "candidate_losses",
            "result",
            "result_scores",
            "baseline_loss",
            "result_loss",
            "edit_positions",
            "edit_bases",
            "accepted",
            "stop_reasons",
        )
    }
    shared = None
    positions = None
    shapes = None
    last_reason = None
    for iteration in complete:
        arrays, metadata = load_arrays(pairs[iteration])
        if metadata["iteration"] != iteration:
            raise ValueError(f"iteration metadata disagrees with filename: {pairs[iteration]}")
        provenance = {key: value for key, value in metadata.items() if key not in _ROUND_FIELDS}
        if shared is None:
            shared = provenance
            positions = arrays["positions"]
            shapes = {
                name: arrays[name].shape
                for name in (
                    "sequence",
                    "baseline_scores",
                    "candidate_scores",
                    "candidate_losses",
                    "result",
                    "result_scores",
                )
            }
        elif provenance != shared or not np.array_equal(arrays["positions"], positions):
            raise ValueError(
                f"greedy rounds have different provenance or positions: {pairs[iteration]}"
            )
        for name in (
            "sequence",
            "baseline_scores",
            "candidate_scores",
            "candidate_losses",
            "result",
            "result_scores",
        ):
            value = arrays[name]
            if value.shape != shapes[name]:
                raise ValueError(f"greedy rounds have different {name} shapes: {pairs[iteration]}")
            if name in {"sequence", "baseline_scores", "result", "result_scores"}:
                if value.shape[0] != 1:
                    raise ValueError(
                        f"greedy history must describe one template: {pairs[iteration]}"
                    )
                value = value[0]
            collected[name].append(value)
        for name in ("baseline_loss", "result_loss", "accepted"):
            collected[name].append(np.asarray(metadata[name]))
        collected["edit_positions"].append(np.asarray(metadata["position"], dtype=np.int64))
        collected["edit_bases"].append(np.asarray(metadata["base"], dtype=np.int64))
        last_reason = metadata["stop_reason"]
        collected["stop_reasons"].append(np.asarray(last_reason or ""))
        if last_reason is not None and iteration != complete[-1]:
            raise ValueError(f"greedy history has rounds after its stop reason: {pairs[iteration]}")

    aggregated = {name: np.stack(values) for name, values in collected.items()}
    aggregated["iterations"] = np.asarray(complete, dtype=np.int64)
    aggregated["positions"] = positions
    summary = dict(
        shared,
        kind="greedy_trajectory",
        n_rounds=len(complete),
        n_accepted=int(aggregated["accepted"].sum()),
        complete=last_reason in {"max_iter", "insufficient_improvement"},
        stop_reason=last_reason,
        source_history=str(history_dir.resolve()),
    )
    return save_arrays(destination, aggregated, summary)
