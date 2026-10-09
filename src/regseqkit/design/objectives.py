"""Objectives that convert scores into per-sequence losses.

Scorers compute scores from model outputs. An Objective applies a goal to
one scorer; a WeightedObjective combines the losses of sub-objectives.
Both declare their required scorers and consume score columns in that order.
ScoreModule evaluates those scorers over shared model predictions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch

from ..scoring import Scorer


class ObjectiveProtocol(Protocol):
    """Typing contract shared by built-in and custom objectives."""

    @property
    def scorers(self) -> tuple[Scorer, ...]:
        """Scorers in the order of the input score columns."""
        ...

    def __call__(self, scores: torch.Tensor) -> torch.Tensor:
        """Return (N,) losses from (N, len(scorers)) scores."""
        ...


@dataclass(frozen=True)
class Objective:
    """Apply an optimization goal to one scorer.

    Parameters
    ----------
    scorer : Scorer
        Scorer whose score is optimized.
    mode : str
        "maximize" minimizes -s, "minimize" minimizes s, and "match"
        minimizes (s - target)**2. "max" and "min" are accepted aliases
        and are stored as "maximize" and "minimize".
    target : float or None, default None
        Finite target required for "match", rejected for other modes.

    Notes
    -----
    Matching is a soft penalty, so optimization can trade target deviations
    against other losses. Targets must be resolved before construction.
    """

    MODES = ("maximize", "minimize", "match")

    scorer: Scorer
    mode: str
    target: float | None = None

    def __post_init__(self) -> None:
        mode = {"max": "maximize", "min": "minimize"}.get(self.mode, self.mode)
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {self.mode!r}")
        object.__setattr__(self, "mode", mode)
        if mode == "match":
            if self.target is None or not np.isfinite(self.target):
                raise ValueError("match objectives need a finite target")
            object.__setattr__(self, "target", float(self.target))
        elif self.target is not None:
            raise ValueError(f"{mode} objectives take no target")

    @property
    def scorers(self) -> tuple[Scorer, ...]:
        """The single scorer required by this objective."""
        return (self.scorer,)

    def __call__(self, scores: torch.Tensor) -> torch.Tensor:
        """Compute one loss per sequence.

        Parameters
        ----------
        scores : torch.Tensor, shape (N, 1)
            Scores from this objective's scorer.

        Returns
        -------
        losses : torch.Tensor, shape (N,)
            Lower-is-better losses, without a batch reduction.
        """
        if scores.ndim != 2 or scores.shape[1] != 1:
            raise ValueError(f"expected (N, 1) scores, got {tuple(scores.shape)}")
        values = scores[:, 0]
        if self.mode == "maximize":
            return -values
        if self.mode == "minimize":
            return values
        return (values - self.target) ** 2

    def __str__(self) -> str:
        target = "" if self.target is None else f", target={self.target:g}"
        return f"{self.mode}({self.scorer}{target})"


@dataclass(frozen=True)
class WeightedObjective:
    """Combine per-sequence losses from sub-objectives.

    Parameters
    ----------
    objectives : Sequence[ObjectiveProtocol]
        Nonempty sequence of built-in or custom objectives. Each sub-objective
        declares its scorers and returns one loss per sequence.
    weights : Sequence[float] or None, default None
        One finite, nonnegative weight per sub-objective, with at least one positive
        weight. None assigns 1 to every sub-objective. Weights are not normalized.

    Notes
    -----
    Required scorers are deduplicated by identity in first-use order.
    Children receive their own score columns in their declared order.
    Weights depend on the scales of the sub-objective losses.
    """

    objectives: Sequence[ObjectiveProtocol]
    weights: Sequence[float] | None = None
    scorers: tuple[Scorer, ...] = field(init=False)
    # Per-sub-objective indices into the shared score columns.
    _scorer_indices: tuple[tuple[int, ...], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        objectives = tuple(self.objectives)
        if not objectives:
            raise ValueError("a weighted objective needs at least one sub-objective")
        weights = (
            (1.0,) * len(objectives)
            if self.weights is None
            else tuple(float(weight) for weight in self.weights)
        )
        if len(weights) != len(objectives):
            raise ValueError("weights must contain one weight per sub-objective")
        if not all(np.isfinite(weight) and weight >= 0 for weight in weights):
            raise ValueError("weights must be finite and nonnegative")
        if not any(weight > 0 for weight in weights):
            raise ValueError("at least one weight must be positive")

        scorers = []
        index = {}
        scorer_indices = []
        for sub_objective in objectives:
            if not sub_objective.scorers:
                raise ValueError("each sub-objective must declare at least one scorer")
            sub_indices = []
            for scorer in sub_objective.scorers:
                key = id(scorer)
                if key not in index:
                    index[key] = len(scorers)
                    scorers.append(scorer)
                sub_indices.append(index[key])
            scorer_indices.append(tuple(sub_indices))
        object.__setattr__(self, "objectives", objectives)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "scorers", tuple(scorers))
        object.__setattr__(self, "_scorer_indices", tuple(scorer_indices))

    def __call__(self, scores: torch.Tensor) -> torch.Tensor:
        """Compute one weighted loss per sequence.

        Parameters
        ----------
        scores : torch.Tensor, shape (N, len(scorers))
            Score columns in this objective's scorer order.

        Returns
        -------
        losses : torch.Tensor, shape (N,)
            Weighted sum of sub-objective losses, without a batch reduction.
        """
        if scores.ndim != 2 or scores.shape[1] != len(self.scorers):
            raise ValueError(f"expected (N, {len(self.scorers)}) scores, got {tuple(scores.shape)}")
        total = scores.new_zeros(len(scores))
        for sub_objective, weight, indices in zip(
            self.objectives, self.weights, self._scorer_indices
        ):
            if weight:
                total = total + weight * sub_objective(scores[:, list(indices)])
        return total

    def __str__(self) -> str:
        return " + ".join(
            str(objective) if weight == 1.0 else f"({objective})*{weight:g}"
            for objective, weight in zip(self.objectives, self.weights)
        )
