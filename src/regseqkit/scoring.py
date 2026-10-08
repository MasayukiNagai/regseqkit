"""Scorers: model outputs reduced to one number per sequence.

Scorers turn model outputs into one score per sequence. Scores can average
replicates, difference conditions, weight a contrast, or apply other
aggregations. An Objective converts scores into losses for maximizing,
minimizing, or matching a target.

One class per model head, because the heads answer different questions.
:class:`ScalarScorer` reads the scalar head, which already emits one number per
output, so it never touches the profile. :class:`ProfileScorer` reads the
profile head, where there is a position axis and therefore something to
collapse. ``reduction`` belongs to the second and nowhere else.

A scorer reads *outputs*, not sequences, which is what lets several of them
share one model forward. :class:`ScoreModule` is that sharing. It serves
attribution, where one mutation pass yields every scorer, and a weighted
objective, which reads several scorers from one forward.
Optimizers receive the stack of scorers required by their objective.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch


def _validate_weights(values: Any) -> torch.Tensor:
    """Convert weights to float32 and validate without rescaling them."""
    weights = torch.as_tensor(values, dtype=torch.float32)
    if weights.ndim != 1 or not weights.numel():
        raise ValueError("weights must be a nonempty vector")
    if not torch.isfinite(weights).all() or not weights.abs().sum():
        raise ValueError("weights must be finite and not all zero")
    return weights


class Scorer(ABC):
    """One number per sequence, computed from prepared model outputs."""

    @abstractmethod
    def __call__(self, outputs: torch.Tensor) -> torch.Tensor:
        """Reduce prepared model outputs to one score per example."""


@dataclass(frozen=True)
class ScalarScorer(Scorer):
    """A weighted sum of the model's scalar outputs.

    The scalar head already emits one number per output, the total over the
    output window, so this never touches the profile. To compute a*y1 - b*y2,
    supply weights [a, -b] in model output order.

    Parameters
    ----------
    weights : torch.Tensor, shape (outputs,)
        One weight per model output, in model output order.
        Predictions are combined in the space the model or its output wrapper
        returns. This scorer performs no output conversion.
    standardize : tuple[torch.Tensor, torch.Tensor] or None, default None
        Optional ``(center, scale)`` per model output, applied before the
        weights, so outputs of different spread contribute on the same footing
        before an objective uses their combined score. Produced by
        :mod:`regseqkit.calibrate`; this class never learns how.

    Raises
    ------
    ValueError
        If the weights are invalid or standardization does not match them.
    """

    weights: torch.Tensor
    standardize: tuple[torch.Tensor, torch.Tensor] | None = None

    def __post_init__(self) -> None:
        weights = _validate_weights(self.weights)
        object.__setattr__(self, "weights", weights)

        if self.standardize is not None:
            center, scale = (torch.as_tensor(v, dtype=torch.float32) for v in self.standardize)
            if center.shape != weights.shape or scale.shape != weights.shape:
                raise ValueError(
                    "standardization must give one center and scale per "
                    f"model output, expected {tuple(weights.shape)}"
                )
            if not (scale > 0).all():
                raise ValueError("standardization scales must be positive")
            object.__setattr__(self, "standardize", (center, scale))

    def __call__(self, outputs: torch.Tensor) -> torch.Tensor:
        """Return the weighted sum of the model's scalar outputs.

        Parameters
        ----------
        outputs : torch.Tensor, shape (N, outputs)
            Prepared scalar predictions in model output order.

        Returns
        -------
        scores : torch.Tensor, shape (N,)
            One weighted score per example.
        """
        if outputs.ndim != 2 or outputs.shape[1] != self.weights.numel():
            raise ValueError(f"expected (N, {self.weights.numel()}) scalar predictions")
        values = outputs
        if self.standardize is not None:
            center, scale = self.standardize
            values = (values - center.to(values)) / scale.to(values)
        return values @ self.weights.to(device=values.device, dtype=values.dtype)


@dataclass(frozen=True)
class ProfileScorer(Scorer):
    """A weighted combination of profile channels, reduced over positions.

    Where :class:`ScalarScorer` combines already-pooled totals, this combines
    the profile head *per position* and only then collapses, so a difference
    that cancels out in the totals can still be measured where it happens. This
    is about *where* signal sits, not how much of it there is.

    Parameters
    ----------
    weights
        One weight per profile channel, in channel order.
    absolute
        Take ``|.|`` of the per-position combination before reducing, so a gain
        at one position and a loss at another add rather than cancel.
    reduction
        How to collapse positions: ``"mean"``, ``"max"`` or ``"sum"``.
    """

    REDUCTIONS = ("mean", "max", "sum")

    weights: torch.Tensor
    absolute: bool = True
    reduction: str = "mean"

    def __post_init__(self) -> None:
        if self.reduction not in self.REDUCTIONS:
            raise ValueError(f"reduction must be one of {self.REDUCTIONS}")
        object.__setattr__(self, "weights", _validate_weights(self.weights))

    def __call__(self, outputs: torch.Tensor) -> torch.Tensor:
        """Combine profile channels and reduce over positions.

        Parameters
        ----------
        outputs : torch.Tensor, shape (N, channels, positions)
            Prepared profile predictions in channel order.

        Returns
        -------
        scores : torch.Tensor, shape (N,)
            One reduced score per example.
        """
        if outputs.ndim != 3 or outputs.shape[1] != self.weights.numel():
            raise ValueError(
                f"expected (N, {self.weights.numel()}, positions) profiles"
            )
        weights = self.weights.to(device=outputs.device, dtype=outputs.dtype)
        per_position = torch.einsum("ncl,c->nl", outputs, weights)
        if self.absolute:
            per_position = per_position.abs()
        if self.reduction == "mean":
            return per_position.mean(-1)
        if self.reduction == "sum":
            return per_position.sum(-1)
        return per_position.max(-1).values


class ScoreModule(torch.nn.Module):
    """Evaluate several scorers over one shared model forward.

    Two consumers, both needing many scores from one pass
    1. ism attribution, where one in-silico mutagenesis pass yields every scorer
    2. a weighted objective, which reads several scorers per forward.
    Optimizers receive the scorers in objective.scorers order.

    Parameters
    ----------
    model : torch.nn.Module
        Returns the prepared prediction tensor consumed by the scorers.
    scorers : Sequence[Scorer]
        Scorers to evaluate, in output column order.

    Raises
    ------
    ValueError
        If no scorers are given. Each scorer validates its input shape when called.
    """

    def __init__(self, model: torch.nn.Module, scorers: Sequence[Scorer]) -> None:
        super().__init__()
        scorers = list(scorers)
        if not scorers:
            raise ValueError("at least one scorer is required")
        self.model = model
        self.scorers = scorers

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Return ``(N, len(scorers))`` in scorer order."""
        outputs = self.model(X)
        return torch.stack([scorer(outputs) for scorer in self.scorers], dim=1)
