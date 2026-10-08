"""Export both TF-MoDISco pattern signs and zero-based seqlet coordinates.

Two things upstream will not do. ``modiscolite.io``'s MEME exporter visits only
``pos_patterns``, so a negative pattern silently disappears; and its BED
exporter adds one to every start. What is *not* local is the window-to-genome
arithmetic: Tangermeme's ``example_to_fasta_coords`` recenters each locus and
offsets the span vectorized, so this module only enumerates patterns and their
signs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from tangermeme.utils import example_to_fasta_coords


_CATEGORIES = ("pos_patterns", "neg_patterns")


@dataclass(frozen=True)
class Pattern:
    """One TF-MoDISco pattern, with its sign carried in ``name``.

    Attributes
    ----------
    name
        ``<category>.<pattern>``, so ``neg_patterns.pattern_3`` says its own sign.
    ppm
        ``(length, 4)`` base frequencies of the seqlets that formed the pattern.
    cwm
        ``(length, 4)`` signed contributions.
    n_seqlets
        How many seqlets the pattern was built from.
    """

    name: str
    ppm: np.ndarray
    cwm: np.ndarray
    n_seqlets: int


def load_patterns(h5_path: str | Path) -> list[Pattern]:
    """Read both signs of pattern out of a TF-MoDISco discovery HDF5.

    The frequency normalization lives here so that a consumer comparing patterns
    across runs and :func:`export_matrices` cannot disagree about what a PPM is.

    Parameters
    ----------
    h5_path
        TF-MoDISco discovery HDF5, holding ``pos_patterns`` and ``neg_patterns``
        groups.

    Returns
    -------
    list of Pattern
        Positive patterns first, then negative, each in the file's own order.
    """
    patterns = []
    with h5py.File(h5_path) as handle:
        for category in _CATEGORIES:
            for name, pattern in handle.get(category, {}).items():
                sequence = pattern["sequence"][:]
                totals = sequence.sum(1, keepdims=True)
                patterns.append(
                    Pattern(
                        name=f"{category}.{name}",
                        ppm=np.divide(
                            sequence,
                            totals,
                            out=np.full_like(sequence, 0.25),
                            where=totals > 0,
                        ),
                        cwm=pattern["contrib_scores"][:],
                        n_seqlets=int(pattern["seqlets"]["start"].shape[0]),
                    )
                )
    return patterns


def export_matrices(h5_path: str | Path, prefix: str | Path) -> None:
    """Write ``<prefix>.matrices.npz`` and a ``<prefix>.ppm.meme`` with both signs.

    Contribution-weighted matrices (CWMs) are saved beside the probability
    matrices but never written to the MEME file: a negative CWM is not a
    probability matrix, and a MEME reader would treat it as one.

    Parameters
    ----------
    h5_path
        TF-MoDISco discovery HDF5, holding ``pos_patterns`` and
        ``neg_patterns`` groups.
    prefix
        Output stem. Patterns are named ``<category>.<pattern>`` in both files.
    """
    from modiscolite.meme_writer import MEMEWriter, MEMEWriterMotif

    writer = MEMEWriter(
        memesuite_version="5",
        alphabet="ACGT",
        background_frequencies="A 0.25 C 0.25 G 0.25 T 0.25",
    )
    matrices = {}
    for pattern in load_patterns(h5_path):
        matrices[f"{pattern.name}.ppm"] = pattern.ppm
        matrices[f"{pattern.name}.cwm"] = pattern.cwm
        writer.add_motif(
            MEMEWriterMotif(
                name=pattern.name,
                probability_matrix=pattern.ppm,
                source_sites=1,
                alphabet="ACGT",
                alphabet_length=4,
            )
        )
    np.savez_compressed(f"{prefix}.matrices.npz", **matrices)
    writer.write(f"{prefix}.ppm.meme")


def export_seqlets(
    h5_path: str | Path,
    bed_path: str | Path,
    output_path: str | Path,
    window_size: int,
) -> None:
    """Map every seqlet onto the genome as a zero-based, half-open BED row.

    Parameters
    ----------
    h5_path
        TF-MoDISco discovery HDF5.
    bed_path
        The loci the attribution arrays were computed over, in example order; a
        seqlet's ``example_idx`` indexes into it.
    output_path
        BED file to write: ``chrom``, ``start``, ``end``, ``name``, ``score``,
        ``strand``, with no header. ``name`` is ``<category>.<pattern>`` and
        ``strand`` is ``-`` for a reverse-complement seqlet.
    window_size
        Width of the attribution window the seqlet coordinates are relative to.

    Raises
    ------
    ValueError
        If a seqlet's example index or interval falls outside the inputs.
    """
    loci = pd.read_csv(
        bed_path,
        sep="\t",
        header=None,
        usecols=[0, 1, 2],
        names=["chrom", "start", "end"],
    )
    records = []
    with h5py.File(h5_path) as handle:
        for category in _CATEGORIES:
            for name, pattern in handle.get(category, {}).items():
                seqlets = pattern["seqlets"]
                index = np.asarray(seqlets["example_idx"][:], dtype=np.int64)
                start = np.asarray(seqlets["start"][:], dtype=np.int64)
                end = np.asarray(seqlets["end"][:], dtype=np.int64)
                strand = np.where(np.asarray(seqlets["is_revcomp"][:]), "-", "+")
                if len(index) and (index.min() < 0 or index.max() >= len(loci)):
                    raise ValueError("seqlet example index is outside the input windows")
                if len(start) and (
                    start.min() < 0 or end.max() > window_size or (start >= end).any()
                ):
                    raise ValueError("seqlet interval is outside the input windows")
                records.append(
                    pd.DataFrame(
                        dict(
                            example=index,
                            start=start,
                            end=end,
                            name=f"{category}.{name}",
                            score=0,
                            strand=strand,
                        )
                    )
                )
    if not records:
        pd.DataFrame().to_csv(output_path, sep="\t", index=False, header=False)
        return
    spans = pd.concat(records, ignore_index=True)
    coords = example_to_fasta_coords(spans, loci, window=window_size)
    coords[["chrom", "start", "end", "name", "score", "strand"]].to_csv(
        output_path, sep="\t", index=False, header=False
    )
