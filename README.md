# regseqkit

regseqkit is a Python library for scoring and optimizing regulatory DNA with sequence-to-function models in PyTorch. Its focus is defining activity scores, reference-based output scales, and design objectives across experimental conditions or independent models.

- **Scoring:** combine model outputs into selected activities or contrasts, and evaluate multiple scorers over shared predictions.
- **Reference-based scaling:** standardize outputs against reference datasets and resolve percentile targets before optimization.
- **Design objectives:** combine goals to increase, decrease, or match scores, including penalties for changing a template's activity in selected conditions.
- **Sequence search:** use those objectives with greedy substitution or Ledidi while restricting edits to selected positions.

Supporting modules provide sequence and signal readers, prediction metrics, mutagenesis, motif export, and plotting. Batched prediction uses tangermeme; gradient-based design uses the optional Ledidi integration.

Documentation:

- [Design workflow](docs/design.md): model wrappers, scorers, objectives, and sequence search.
- [Configuration guide](docs/config.md): defining scorers and objectives as dictionaries.

## Installation

Python 3.13 or newer is required. Add regseqkit to your uv project from GitHub over SSH:

```bash
uv add "git+ssh://git@github.com/MasayukiNagai/regseqkit.git"
```

Optional integrations are available as extras:

```bash
uv add "git+ssh://git@github.com/MasayukiNagai/regseqkit.git" --extra ledidi --extra cherimoya --extra motifs
```

## Repository layout

| Module | Purpose |
| --- | --- |
| `sequences` | DNA encoding, reverse complements, FASTA/NPZ reading, and genomic locus extraction |
| `inference` | Batched prediction and conversion of log counts and profile logits |
| `scoring`, `config` | Scalar scorers, shared model evaluation, and dictionary-based definitions |
| `design` | Objectives, weighted composition, greedy substitution, and optional Ledidi integration |
| `calibrate` | Reference prediction distributions, standardization, and percentile targets |
| `metrics` | Scalar and profile evaluation against observations |
| `mutagenesis` | Single-site predictions and attribution transforms |
| `motifs` | TF-MoDISco pattern loading, matrices, and seqlets with genomic coordinates; export requires the motifs extra |
| `figures` | Matplotlib layouts, density scatter plots, profiles, and attribution logos |
| `io` | NPZ array artifacts with JSON metadata sidecars |
| `cherimoya` | Optional model loading, inference, training jobs, and contrast loss; requires the cherimoya extra |
