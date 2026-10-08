"""Reference distributions, so a design target is a statement about data.

Two independently trained models, or two tracks from different experiments,
predict on whatever scale their assay and depth imply. A weight between two
scorers then means "half as important, times whatever this track's dynamic
range happens to be", which is not what anyone wrote down. Standardizing each
output against a reference distribution fixes that.

**The method is which reference set you point this at, not anything in here.**
Summarizing predictions over GC-matched negatives gives a null where zero means
background; over the peak set it gives a reporting grid; over a held-out split
it gives the model's working range. This module has no preference, and neither
a scorer nor an objective ever learns how its numbers were obtained. That
separation is the point: a sibling project hardcoded one slope and intercept
into five files, and they drifted apart into two optimizers running different
values with the channels swapped in one of them.

Three scales stay distinct:

- **centre and scale** feed :class:`~regseqkit.scoring.ScalarScorer`. The
  transform is affine with constant terms, so it stays differentiable and the
  optimizers are unaffected.
- **the quantile grid** is for reporting and for resolving a ``p90`` target.
  The lookup is monotone and not differentiable, so it never enters a loss.
- **raw model units** are what artifacts record, so nothing is lost.

numpy only. This module reads saved arrays, never a model.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# Coarse enough to stay readable in the JSON, fine enough that interpolating
# between neighbours is harmless for a reporting number.
QUANTILE_LEVELS: tuple[float, ...] = (0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0, 100.0)


@dataclass(frozen=True)
class Distribution:
    """Summary statistics of one output's scores over one reference set."""

    n: int
    mean: float
    sd: float
    grid: dict[str, float]

    @classmethod
    def summarize(cls, values: np.ndarray) -> Distribution:
        """Summarize a vector of scores.

        Raises
        ------
        ValueError
            If fewer than two values are given. ``std(ddof=1)`` on one sample is
            a silent nan, and it would only surface as a nan loss during design.
        """
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size < 2:
            raise ValueError(
                f"need at least 2 finite values to summarize a distribution, got "
                f"{values.size}; check that the locus set survived filtering"
            )
        grid = np.percentile(values, QUANTILE_LEVELS)
        return cls(
            n=int(values.size),
            mean=float(values.mean()),
            # ddof=1: a sample of a population, and the sd goes into a
            # denominator the design objectives depend on.
            sd=float(values.std(ddof=1)),
            grid={f"p{level:g}": float(value) for level, value in zip(QUANTILE_LEVELS, grid)},
        )

    def as_json(self) -> dict[str, Any]:
        return {"n": self.n, "mean": self.mean, "sd": self.sd, "grid": dict(self.grid)}

    @classmethod
    def from_json(cls, blob: Mapping[str, Any]) -> Distribution:
        return cls(
            n=int(blob["n"]),
            mean=float(blob["mean"]),
            sd=float(blob["sd"]),
            grid={str(k): float(v) for k, v in blob["grid"].items()},
        )

    def _sorted_grid(self) -> tuple[np.ndarray, np.ndarray]:
        levels = np.array([float(key[1:]) for key in self.grid], dtype=np.float64)
        values = np.array(list(self.grid.values()), dtype=np.float64)
        order = np.argsort(values)
        return values[order], levels[order]

    def percentile_of(self, values: np.ndarray) -> np.ndarray:
        """Where scores fall in this distribution, clamped to [0, 100].

        A design beating every observed value reports 100 rather than
        extrapolating a percentile the data cannot support.
        """
        grid_values, grid_levels = self._sorted_grid()
        return np.interp(np.asarray(values, dtype=np.float64), grid_values, grid_levels)

    def value_at(self, level: float) -> float:
        """The score at a given percentile of this distribution.

        This is how a target is set from data: "as open as the 90th percentile
        of real peaks" rather than a number read off a loss curve.
        """
        if not 0 <= level <= 100:
            raise ValueError(f"percentile must be in [0, 100], got {level}")
        grid_values, grid_levels = self._sorted_grid()
        return float(np.interp(level, grid_levels, grid_values))


def summarize_predictions(
    predicted: np.ndarray, outputs: Sequence[str]
) -> dict[str, Distribution]:
    """One :class:`Distribution` per output, from an ``(examples, outputs)`` array.

    Raises
    ------
    ValueError
        If the array's output axis does not match `outputs`.
    """
    predicted = np.asarray(predicted)
    if predicted.ndim != 2 or predicted.shape[1] != len(outputs):
        raise ValueError(
            f"expected (examples, {len(outputs)}) predictions, got {predicted.shape}"
        )
    return {
        name: Distribution.summarize(predicted[:, column])
        for column, name in enumerate(outputs)
    }


@dataclass
class Calibration:
    """Named reference sets, each summarized per model output.

    Attributes
    ----------
    references
        Reference set name to output name to :class:`Distribution`. The set
        names are the caller's: ``negatives``, ``peaks``, ``train``, whatever
        was summarized.
    provenance
        Free-form record of how the summaries were produced.
    """

    references: dict[str, dict[str, Distribution]] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def outputs(self, reference: str) -> list[str]:
        """The output names summarized for one reference set."""
        return list(self._reference(reference))

    def _reference(self, reference: str) -> dict[str, Distribution]:
        if reference not in self.references:
            raise KeyError(
                f"no reference set {reference!r} in this calibration; have "
                f"{sorted(self.references)}"
            )
        return self.references[reference]

    def distribution(self, reference: str, output: str) -> Distribution:
        """One output's distribution over one reference set."""
        entry = self._reference(reference)
        if output not in entry:
            raise KeyError(
                f"reference {reference!r} has no output {output!r}; have {sorted(entry)}"
            )
        return entry[output]

    def center_scale(
        self, reference: str, outputs: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-output mean and standard deviation, in `outputs` order.

        This is what a scorer applies before its weights, so every output
        contributes on the same footing and a term weight means what it says.

        Raises
        ------
        ValueError
            If any output's spread is zero, which would divide by zero.
        """
        center = np.array(
            [self.distribution(reference, name).mean for name in outputs], dtype=np.float64
        )
        scale = np.array(
            [self.distribution(reference, name).sd for name in outputs], dtype=np.float64
        )
        if not (scale > 0).all():
            flat = [name for name, value in zip(outputs, scale) if value <= 0]
            raise ValueError(
                f"reference {reference!r} has zero spread for {flat}; it cannot "
                "standardize an output whose scores never vary"
            )
        return center, scale

    def percentile_of(self, reference: str, output: str, values: np.ndarray) -> np.ndarray:
        """Where scores fall in one reference distribution."""
        return self.distribution(reference, output).percentile_of(values)

    def value_at_percentile(self, reference: str, output: str, level: float) -> float:
        """The score at a percentile of one reference distribution."""
        return self.distribution(reference, output).value_at(level)

    def as_json(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance,
            "references": {
                name: {output: d.as_json() for output, d in entry.items()}
                for name, entry in self.references.items()
            },
        }

    def save(self, path: str | Path) -> Path:
        """Write ``calibration.json``, creating parent directories."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_json(), indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> Calibration:
        """Read a calibration written by :meth:`save`."""
        blob = json.loads(Path(path).read_text())
        return cls(
            references={
                name: {output: Distribution.from_json(d) for output, d in entry.items()}
                for name, entry in blob.get("references", {}).items()
            },
            provenance=blob.get("provenance", {}),
        )


def resolve_percentile_targets(
    entries: Sequence[Mapping[str, Any]],
    calibration: Calibration,
    reference: str,
) -> list[dict[str, Any]]:
    """Replace pNN objective targets with the reference set's quantile.

    A percentile target says "as high as the 90th percentile of this reference
    set", which is a statement about data rather than a number chosen to make a
    loss curve look right.

    Parameters
    ----------
    entries : Sequence[Mapping[str, Any]]
        Objective entries with scorer, mode and target, or an objectives list
        for weighted composition. Only targets such as "p90" change.
    calibration : Calibration
        Supplies the distributions.
    reference : str
        Which reference set to resolve against.

    Returns
    -------
    resolved : list[dict[str, Any]]
        Copies of entries with percentile strings replaced, including sub-objectives.

    Raises
    ------
    KeyError
        If reference is not in the calibration.
    ValueError
        If a percentile is malformed, or names a scorer with no distribution.
        A distribution is summarized per model output, so a scorer that
        combines outputs has none; its objective carries a numeric target,
        or uses a match target of "template" in configuration.
    """

    def resolve(entry: Mapping[str, Any]) -> dict[str, Any]:
        entry = dict(entry)
        if "objectives" in entry:
            entry["objectives"] = [resolve(sub_objective) for sub_objective in entry["objectives"]]
        target = entry.get("target")
        if not (isinstance(target, str) and target.startswith("p")):
            return entry
        if entry.get("mode") != "match":
            raise ValueError("percentile targets require mode 'match'")
        try:
            level = float(target[1:])
        except ValueError:
            raise ValueError(f"malformed percentile {target!r}") from None
        name = str(entry.get("scorer", ""))
        if name not in calibration.outputs(reference):
            raise ValueError(
                f"cannot resolve {target!r} for scorer {name!r}: reference "
                f"{reference!r} has no distribution under that name. A scorer "
                "that combines outputs has no single reference distribution; "
                "give a numeric target, or use match with target 'template' to track the "
                "template's own score."
            )
        entry["target"] = calibration.value_at_percentile(reference, name, level)
        return entry

    return [resolve(entry) for entry in entries]
