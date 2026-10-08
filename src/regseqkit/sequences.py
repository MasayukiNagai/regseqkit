"""Reading sequence and signal: from a genome, from a FASTA, or from an NPZ.

Three input sources, three readers, all returning numpy:

- :func:`extract_loci_with_coords` centers a window on each locus of a BED and
  reads one-hot sequence plus one signal track per bigWig.
- :func:`read_fasta` reads sequences already cut to a fixed width, which is how
  design templates arrive.
- :func:`read_npz` reads prepared examples, for a model with no genome behind it.

Nothing here imports torch or tangermeme. The conversion to tensors happens at
the model boundary in :mod:`regseqkit.config`, which is what keeps a stage that
only reads coordinates cheap.

Why this is owned rather than wrapped
-------------------------------------
``tangermeme.io.extract_loci`` returns sequences and a boolean mask but never
coordinates, and the mask indexes the rows it *read* rather than the rows it
returned. Recovering which locus produced example 7 therefore meant rebuilding
the input frame through tangermeme's private ``_interleave_loci``. Returning the
surviving coordinates directly removes that, and encoding the whole batch
through one lookup table rather than one call per locus is faster besides.

**Cherimoya trains through tangermeme's extractor, not this one.** That is why
the name differs: the two are not interchangeable, and the reported evaluation
numbers were computed on the loci tangermeme's filters kept. The drop rules here
reproduce those exactly, and ``tests`` pins that with a parity check.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import pyBigWig
from pyfaidx import Fasta
from tqdm import tqdm


ALPHABET = "ACGT"

# A base outside the alphabet, such as an ambiguous IUPAC code, becomes an
# all-zero column rather than an error, matching what tangermeme does with its
# ``ignore`` list. The locus is kept; `drop_ambiguous` is what removes it,
# because a column with no reference base is one in-silico mutagenesis cannot
# mutate.

BED_COLUMNS = ["chrom", "start", "end"]


class OneHotEncoder:
    """One-hot encoder and decoder for DNA sequences.

    Encoding is table-driven and batched: characters are mapped to indices
    through a 128-entry lookup and the whole batch is built with one fancy
    index, rather than one call per sequence.

    Parameters
    ----------
    alphabet
        Sequence alphabet. Default ``"ACGT"``.
    channel_axis
        Axis the alphabet occupies in the encoding: ``1`` for ``(N, 4, L)``,
        ``-1`` or ``2`` for ``(N, L, 4)``.
    dtype
        Encoding dtype. ``int8`` matches tangermeme and is four times smaller
        than float32, which matters at genome scale.
    """

    def __init__(
        self,
        alphabet: str = ALPHABET,
        channel_axis: int = 1,
        dtype: Any = np.int8,
    ) -> None:
        if channel_axis not in (1, -1, 2):
            raise ValueError("channel_axis must be 1, -1, or 2")

        self.alphabet = alphabet.upper()
        self.vocab_size = len(self.alphabet)
        self.channel_axis = channel_axis
        self.dtype = dtype

        # ASCII to index; 255 marks a character outside the alphabet.
        self._lookup = np.full(256, 255, dtype=np.uint8)
        for index, character in enumerate(self.alphabet):
            self._lookup[ord(character)] = index
        self._reverse = np.frombuffer(self.alphabet.encode("ascii"), dtype="S1")

    def to_onehot(self, seqs: str | Sequence[str]) -> np.ndarray:
        """Encode one sequence or a batch of equal-length sequences.

        Parameters
        ----------
        seqs
            A sequence, or sequences which are padded to the longest with
            all-zero columns.

        Returns
        -------
        ndarray
            ``(N, 4, L)`` or ``(N, L, 4)`` per `channel_axis`. A single input
            sequence drops the batch axis.
        """
        single = isinstance(seqs, str)
        seqs = [seqs] if single else list(seqs)
        if not seqs:
            raise ValueError("no sequences to encode")

        width = max(len(s) for s in seqs)
        packed = np.array([s.upper() for s in seqs], dtype=f"S{width}")
        # Null padding reads as 255 below, so short sequences zero-fill.
        codes = self._lookup[packed.view(np.uint8).reshape(len(seqs), width)]

        unknown = codes == 255
        onehot = np.eye(self.vocab_size, dtype=self.dtype)[np.where(unknown, 0, codes)]
        if unknown.any():
            onehot[unknown] = 0
        if self.channel_axis == 1:
            onehot = onehot.transpose(0, 2, 1)
        return onehot[0] if single else onehot

    def from_onehot(self, onehot: np.ndarray) -> str | list[str]:
        """Decode an encoding back to sequences, with all-zero columns as ``N``."""
        single = onehot.ndim == 2
        if single:
            onehot = onehot[np.newaxis]
        values = onehot.transpose(0, 2, 1) if onehot.shape[1] == self.vocab_size else onehot

        characters = self._reverse[np.argmax(values, axis=-1)]
        undetermined = values.max(axis=-1) != 1
        if undetermined.any():
            characters = characters.copy()
            characters[undetermined] = b"N"

        decoded = [s.decode("ascii") for s in characters.view(f"S{characters.shape[1]}").ravel()]
        return decoded[0] if single else decoded


class Loci(NamedTuple):
    """What one extraction produced, in model row order.

    Attributes
    ----------
    onehot
        ``(N, 4, in_window)`` one-hot ACGT.
    signals
        ``(N, tracks, out_window)`` signal, or None when no bigWigs were given.
    coords
        The surviving loci, with ``chrom``, ``start``, ``end`` and
        ``source_row``: the row of the input frame each example came from.
    """

    onehot: np.ndarray
    signals: np.ndarray | None
    coords: pd.DataFrame


def read_bed(path: str | Path) -> pd.DataFrame:
    """Read the first three columns of a BED as ``chrom``, ``start``, ``end``."""
    frame = pd.read_csv(
        path,
        sep="\t",
        header=None,
        usecols=[0, 1, 2],
        names=BED_COLUMNS,
        dtype={"chrom": str, "start": np.int64, "end": np.int64},
    )
    return frame


def interleave(
    coords: str | Path | pd.DataFrame | Sequence[str | Path | pd.DataFrame],
    chroms: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Combine one or more BEDs into one frame, filtered and round-robin ordered.

    With a single source this is that BED, restricted to `chroms`, in file
    order. With several it interleaves them one row at a time so a truncated
    run stays balanced across sources, which is what tangermeme does.
    """
    sources = (
        [coords]
        if isinstance(coords, (str, Path, pd.DataFrame))
        else list(coords)
    )
    frames = []
    for position, source in enumerate(sources):
        frame = source if isinstance(source, pd.DataFrame) else read_bed(source)
        frame = frame.loc[:, BED_COLUMNS].copy()
        frame["chrom"] = frame["chrom"].astype(str)
        if chroms is not None:
            frame = frame[frame["chrom"].isin(list(chroms))]
        frame["order"] = np.arange(len(frame)) * len(sources) + position
        frames.append(frame)

    combined = pd.concat(frames).sort_values("order")
    return combined.drop(columns="order").reset_index(drop=True)


def _chrom_lengths(fasta: Any) -> dict[str, int]:
    return {str(name): len(record) for name, record in fasta.items()}


def _exclusion_zones(
    chrom_lengths: dict[str, int], exclusion_lists: Sequence[str | Path] | None
) -> dict[str, np.ndarray] | None:
    """Boolean arrays marking excluded 100 bp chunks, one per chromosome.

    The 100 bp resolution is tangermeme's, and it is deliberately coarse: a
    locus is dropped when its window touches any chunk that any excluded
    interval touches. Matching it is what keeps the evaluated locus set equal
    to the one the reported numbers were computed on.
    """
    if not exclusion_lists:
        return None

    zones = {
        chrom: np.zeros(length // 100 + 1, dtype=bool)
        for chrom, length in chrom_lengths.items()
    }
    intervals = pd.concat([read_bed(path) for path in exclusion_lists], ignore_index=True)
    for chrom, group in intervals.groupby("chrom", sort=False):
        zone = zones.get(str(chrom))
        if zone is None:
            continue
        for start, end in zip(group["start"].to_numpy(), group["end"].to_numpy()):
            zone[start // 100 : end // 100 + 1] = True
    return zones


def extract_loci_with_coords(
    coords: str | Path | pd.DataFrame | Sequence[str | Path | pd.DataFrame],
    fasta: str | Path,
    *,
    chroms: Sequence[str] | None = None,
    signals: Sequence[str | Path] | None = None,
    in_window: int = 2114,
    out_window: int = 1000,
    exclusion_lists: Sequence[str | Path] | None = None,
    limit: int | None = None,
    drop_ambiguous: bool = False,
    verbose: bool = False,
) -> Loci:
    """Extract one-hot sequence, signal, and the coordinates that survived.

    Each locus is centered on ``mid = start + (end - start) // 2``; the
    sequence window is `in_window` wide about that midpoint and the signal
    window `out_window` wide.

    Parameters
    ----------
    coords
        BED path, DataFrame, or several of either. Only the first three columns
        are read. Zero-based, half-open.
    fasta
        Reference FASTA. Needs an index beside it.
    chroms
        Restrict to these chromosomes, i.e. one split.
    signals
        bigWig paths, one per output channel, in model output order.
    in_window, out_window
        Sequence and signal widths.
    exclusion_lists
        BED paths whose intervals disqualify an overlapping locus.
    limit
        Stop once this many loci have survived.
    drop_ambiguous
        Also drop a locus whose window contains a base outside the alphabet.
        Off by default, matching tangermeme, which keeps the locus and encodes
        an all-zero column. In-silico mutagenesis cannot mutate such a
        position, so the attribution stage turns this on.
    verbose
        Show a progress bar.

    Returns
    -------
    Loci

    Raises
    ------
    ValueError
        If the windows are not positive, or no locus survived.
    """
    if in_window < 1 or out_window < 1:
        raise ValueError("in_window and out_window must be positive")

    frame = interleave(coords, chroms)
    in_half, out_half = in_window // 2, out_window // 2
    # The bounds check uses the wider of the two windows on both sides, so a
    # locus is kept only if every window it needs fits on the chromosome.
    bound = max(in_half, out_half if signals is not None else 0)

    fasta_handle = Fasta(str(fasta), sequence_always_upper=True)
    signal_handles = [pyBigWig.open(str(path)) for path in (signals or [])]

    try:
        lengths = _chrom_lengths(fasta_handle)
        zones = _exclusion_zones(lengths, exclusion_lists)

        rows = frame.itertuples(index=True)
        if verbose:
            rows = tqdm(rows, total=len(frame), desc="extracting loci")

        sequences: list[str] = []
        signal_blocks: list[np.ndarray] = []
        kept: list[int] = []

        for row in rows:
            chrom = str(row.chrom)
            mid = int(row.start) + (int(row.end) - int(row.start)) // 2

            length = lengths.get(chrom)
            if length is None or mid - bound < 0 or mid + bound > length:
                continue
            if zones is not None:
                zone = zones.get(chrom)
                if zone is None or zone[(mid - bound) // 100 : (mid + bound) // 100 + 1].any():
                    continue

            if signal_handles:
                start, end = mid - out_half, mid + out_half + out_window % 2
                signal_blocks.append(
                    np.stack(
                        [
                            np.nan_to_num(
                                np.asarray(
                                    handle.values(chrom, start, end, numpy=True),
                                    dtype=np.float32,
                                )
                            )
                            for handle in signal_handles
                        ]
                    )
                )

            start, end = mid - in_half, mid + in_half + in_window % 2
            sequences.append(str(fasta_handle[chrom][start:end]))
            kept.append(row.Index)

            if limit is not None and len(sequences) == limit:
                break
    finally:
        fasta_handle.close()
        for handle in signal_handles:
            handle.close()

    if not sequences:
        raise ValueError("no loci survived extraction")

    onehot = OneHotEncoder().to_onehot(sequences)
    keep = np.ones(len(onehot), dtype=bool)
    if drop_ambiguous:
        keep = (onehot.sum(axis=1) == 1).all(axis=-1)

    surviving = frame.loc[np.asarray(kept)[keep]].reset_index(drop=True)
    surviving["source_row"] = np.asarray(kept)[keep]
    return Loci(
        onehot=onehot[keep],
        signals=np.stack(signal_blocks)[keep] if signal_blocks else None,
        coords=surviving,
    )


def read_fasta(path: str | Path, length: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Read a FASTA as one-hot sequences and their record names.

    A base outside the alphabet becomes an all-zero column, as everywhere else
    here, rather than an error. A caller that cannot tolerate one, such as
    design, asserts hard one-hot itself and says so in its own terms.

    Parameters
    ----------
    path
        FASTA to read.
    length
        Required record length, when the caller has one. Design templates must
        carry the complete input window including the model's flanks, because
        no padding is added: a padded flank is context the model would read as
        real sequence.

    Returns
    -------
    onehot : ndarray
        ``(N, 4, length)`` one-hot ACGT.
    names : ndarray
        Record names, in file order.

    Raises
    ------
    ValueError
        If the file is empty, or a record is not `length` long.
    """
    with Fasta(str(path), as_raw=True, sequence_always_upper=True) as fasta:
        names = list(fasta.keys())
        sequences = [fasta[name][:] for name in names]

    if not sequences:
        raise ValueError(f"{path}: no records")
    if length is not None:
        wrong = [n for n, s in zip(names, sequences) if len(s) != length]
        if wrong:
            raise ValueError(
                f"{path}: {len(wrong)} record(s) are not {length} bp, first {wrong[0]!r}"
            )
    return OneHotEncoder().to_onehot(sequences), np.asarray(names)


def read_npz(path: str | Path) -> tuple[np.ndarray, dict[str, np.ndarray], list[str] | None]:
    """Read prepared examples from an NPZ, for a model with no genome.

    The sibling of :func:`read_fasta` and of
    :func:`extract_loci_with_coords`: three input sources, three readers.

    Parameters
    ----------
    path
        NPZ holding ``onehot`` and ``ids``, optionally ``output_names``, plus
        any example-indexed arrays to carry through, typically ``observed``.

    Returns
    -------
    onehot : ndarray
        ``(N, 4, length)`` one-hot ACGT.
    arrays : dict of ndarray
        Every other array in the file, all example-indexed, ``ids`` included.
    outputs : list of str or None
        The file's ``output_names``. Kept out of `arrays` because it indexes
        outputs rather than examples, and everything in `arrays` must share the
        example axis.

    Raises
    ------
    ValueError
        If ``onehot`` or ``ids`` is absent, or the encoding is not
        ``(N, 4, length)``.
    """
    with np.load(path, allow_pickle=False) as handle:
        arrays = dict(handle)

    if "onehot" not in arrays or "ids" not in arrays:
        raise ValueError(f"{path}: prepared examples need 'onehot' and 'ids' arrays")
    onehot = arrays.pop("onehot")
    if onehot.ndim != 3 or onehot.shape[1] != len(ALPHABET):
        raise ValueError("prepared 'onehot' must be (examples, 4, length)")
    names = arrays.pop("output_names", None)
    return onehot, arrays, (None if names is None else [str(n) for n in names])
