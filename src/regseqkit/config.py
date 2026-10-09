"""Build scorers and objectives from plain mapping definitions.

This module does not read configuration files or load models and calibration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .design.objectives import Objective, ObjectiveProtocol, WeightedObjective
from .scoring import ScalarScorer, Scorer

__all__ = ["build_scorers", "build_objectives"]

def build_scorers(
    entries: Sequence[Mapping[str, Any]],
    output_names: Sequence[str],
    *,
    standardize: tuple[Any, Any] | None = None,
) -> dict[str, ScalarScorer]:
    """Build scorers from mapping entries.

    Parameters
    ----------
    entries : Sequence[Mapping]
        One mapping per scorer, with name and weights. Weights map
        an output name to its weight; unlisted outputs weigh zero.
        Model wrappers prepare scalar features before scoring.
    output_names : Sequence[str]
        Model output names, giving the weight vector's order.
    standardize : tuple or None, default None
        Optional (center, scale) per model output, applied to every scalar
        scorer that does not opt out with a standardize value of False.

    Returns
    -------
    scorers : dict[str, ScalarScorer]
        Scorer identifiers mapped to numerical scorers, in entry order.

    Raises
    ------
    ValueError
        If a name is repeated, the weights name an unknown
        output, or an entry requests output conversion or profile reduction.
    KeyError
        If an entry has no name.
    """
    index = {name: position for position, name in enumerate(output_names)}
    built: dict[str, ScalarScorer] = {}
    seen: set[str] = set()

    for entry in entries:
        if "units" in entry or "transform" in entry:
            raise ValueError(
                "scorers do not convert outputs; wrap the model before scoring"
            )
        if any(key in entry for key in ("head", "absolute", "reduction")):
            raise ValueError(
                "scorers require prepared scalar outputs; select heads and reduce "
                "profiles in a model wrapper"
            )
        name = str(entry["name"])
        if name in seen:
            raise ValueError(f"duplicate scorer name {name!r}")
        seen.add(name)

        weights = entry.get("weights") or {}
        unknown = [key for key in weights if key not in index]
        if unknown:
            raise ValueError(f"{name}: unknown output(s) {unknown}; have {list(output_names)}")
        vector = torch.zeros(len(output_names), dtype=torch.float32)
        for key, value in weights.items():
            vector[index[key]] = float(value)

        wanted = entry.get("standardize", True)
        built[name] = ScalarScorer(
            vector,
            standardize=standardize if (standardize and wanted) else None,
        )
    return built


def build_objectives(
    entries: Sequence[Mapping[str, Any]],
    scorers: Mapping[str, Scorer],
    *,
    template_scores: Mapping[str, float] | None = None,
) -> dict[str, ObjectiveProtocol]:
    """Build standalone or weighted objectives from mapping entries.

    Parameters
    ----------
    entries : Sequence[Mapping]
        One named objective per mapping. A standalone objective has scorer,
        mode and an optional target. A weighted objective has an objectives
        list; each sub-objective carries an optional weight, defaulting to 1.
        Weighted objectives may contain other weighted objectives.
    scorers : Mapping[str, Scorer]
        The scorers available to be named.
    template_scores : Mapping[str, float] or None, default None
        Starting scores by scorer id. Resolves a configured match target of
        "template" into a numeric target before constructing the objective.

    Returns
    -------
    dict
        Objective id to objective, in entry order.

    Raises
    ------
    ValueError
        If a name is repeated or an objective is invalid.
    KeyError
        If an entry has no name or an objective names a scorer that does not exist.
    """

    def build(entry: Mapping[str, Any]) -> ObjectiveProtocol:
        if "terms" in entry or "value" in entry:
            raise ValueError("use 'objectives' for composition and 'target' for matching")
        if "objectives" in entry:
            if any(key in entry for key in ("scorer", "mode", "target")):
                raise ValueError("a weighted objective cannot also define scorer, mode or target")
            sub_objectives = entry["objectives"]
            # WeightedObjective combines the losses of its child objectives.
            return WeightedObjective(
                objectives=tuple(build(sub_objective) for sub_objective in sub_objectives),
                weights=tuple(
                    float(sub_objective.get("weight", 1.0)) for sub_objective in sub_objectives
                ),
            )
        wanted = str(entry.get("scorer", ""))
        if not wanted:
            raise ValueError("an objective needs 'objectives' or a 'scorer'")
        if wanted not in scorers:
            raise KeyError(f"unknown scorer {wanted!r}; have {sorted(scorers)}")
        target = entry.get("target")
        if target == "template":
            if entry.get("mode") != "match":
                raise ValueError("target 'template' requires mode 'match'")
            if template_scores is None or wanted not in template_scores:
                raise ValueError(f"missing template score for {wanted!r}")
            target = template_scores[wanted]
        # Objective applies one optimization goal to a single scorer.
        return Objective(
            scorer=scorers[wanted],
            mode=str(entry.get("mode", "")),
            target=None if target is None else float(target),
        )

    built: dict[str, ObjectiveProtocol] = {}

    for entry in entries:
        name = str(entry["name"])
        if name in built:
            raise ValueError(f"duplicate objective name {name!r}")

        objective = build(entry)
        if "weight" in entry:
            objective = WeightedObjective((objective,), weights=(entry["weight"],))
        built[name] = objective
    return built
