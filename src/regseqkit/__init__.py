"""Tools for regulatory sequence analysis, interpretation, and optimization.

The library works independently of a project's vocabulary or file layout.
Its modules cover sequence data, model evaluation, interpretation, and design:

- :mod:`regseqkit.io` -- artifacts on disk, named arrays beside a JSON sidecar.
- :mod:`regseqkit.sequences` -- reading sequence and signal, from a genome, a
  FASTA, or a prepared NPZ.
- :mod:`regseqkit.metrics` -- prediction-against-observation metrics, arrays only.
- :mod:`regseqkit.calibrate` -- reference distributions, so a scale and a
  target are statements about data rather than hand-picked numbers.
- :mod:`regseqkit.scoring` -- scores computed from model outputs: one class per
  model head, and the stack that evaluates several over one forward.
- :mod:`regseqkit.wrappers` -- output conversion before scoring.
- :mod:`regseqkit.config` -- scorers and objectives built from mapping definitions.
- :mod:`regseqkit.design` -- objectives that convert scores into losses,
  weighted composition, and the two sequence optimizers.
- :mod:`regseqkit.mutagenesis` -- single-site predictions and attribution transforms.
- :mod:`regseqkit.motifs` -- TF-MoDISco export with signs and coordinates.
- :mod:`regseqkit.figures` -- figure primitives and attribution logos.
- :mod:`regseqkit.cherimoya` -- Cherimoya loading and prediction helpers.

Scorers compute scores from model outputs; objectives convert scores into
losses. :mod:`regseqkit.calibrate` supplies reference scales and targets.

This module **re-exports nothing**. Submodules pull in torch, Cherimoya or
matplotlib, and a CPU-only stage should not pay for them, so each stage imports
exactly the submodule it needs. ``io``, ``metrics`` and ``sequences`` import
neither torch nor the optional model integrations.
"""

from __future__ import annotations

__all__ = [
    "calibrate",
    "cherimoya",
    "config",
    "design",
    "figures",
    "greedy_analysis",
    "io",
    "metrics",
    "mutagenesis",
    "motifs",
    "scoring",
    "sequences",
    "wrappers",
]
