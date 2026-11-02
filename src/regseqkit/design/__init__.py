"""Objectives, editable geometry, and sequence optimizers."""

from .greedy_substitution import GreedyIteration, greedy_substitution
from .ledidi import ledidi_design
from .objectives import Margin, Objective, WeightedObjective

greedy_design = greedy_substitution

__all__ = [
    "GreedyIteration",
    "Margin",
    "Objective",
    "WeightedObjective",
    "greedy_design",
    "greedy_substitution",
    "ledidi_design",
]
