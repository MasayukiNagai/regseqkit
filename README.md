# regseqkit

Tools for genomic sequence-to-function models.

The core package uses Tangermeme for batched prediction. Seqlet coordinates are converted locally. Ledidi design, Cherimoya integration, and MEME motif export are available through the optional `ledidi`, `cherimoya`, and `motifs` extras.

This project uses the local editable package through the root `pyproject.toml`. Run `uv sync` from the project root to apply dependency metadata changes. The root project retains its own Ledidi, Cherimoya, and Tangermeme dependencies for the project scripts.

## To-Dos
- Update this README.md
- 
