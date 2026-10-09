# Scorer and objective definitions

Configuration dictionaries describe how model outputs become scores and how those scores become optimization losses. They are converted into callable Python objects that perform those calculations.

## Input dictionaries

Each definition is a dictionary. Pass a list of these dictionaries, plus the model's output order:

```python
output_names = ["y0", "y1"]

scorer_definitions = [
    {"name": "scorer1", "weights": {"y0": -1, "y1": 1}},  # Scalar difference (y1 - y0)
    {"name": "scorer2", "weights": {"y0": 1}},  # Select the first scalar output (y1).
]

objective_definitions = [
    {"name": "objective1", "scorer": "scorer1", "mode": "maximize"},  # Increase scorer1 (= increase the difference)
    {"name": "objective2", "scorer": "scorer2", "mode": "match", "target": 5.0},  # Match scorer2 (= y1 to be 5.0)
]
```

The scorer dictionaries describe weighted combinations of prediction columns. Here, the first score is y1 - y0 and the second is y0. The objective dictionaries describe goals for those scores: increase the first score and bring the second toward 2.0.

## Returned objects

Convert the definitions into dictionaries of scorer and objective objects:

```python
from regseqkit.config import build_scorers, build_objectives

scorers = build_scorers(scorer_definitions, output_names=output_names)
objectives = build_objectives(objective_definitions, scorers)
```

The resulting structure is equivalent to constructing the objects directly:

```python
from regseqkit.scoring import ScalarScorer
from regseqkit.design import Objective

equivalent_scorers = {
    "scorer1": ScalarScorer(weights=[-1, 1]),
    "scorer2": ScalarScorer(weights=[1, 0]),
}

equivalent_objectives = {
    "objective1": Objective(scorer=equivalent_scorers["scorer1"], mode="maximize"),
    "objective2": Objective(scorer=equivalent_scorers["scorer2"], mode="match", target=2.0),
}
```

`scorers` holds callable scorer objects, while `objectives` holds callable objective objects. Each objective references its scorer object directly. The input definitions remain ordinary dictionaries.
