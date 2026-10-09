from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import pyBigWig
import torch
from pyfaidx import Fasta
from tqdm import tqdm


ALPHABET = "ACGT"
COMPLEMENT_MAP = {"A": "T", "C": "G", "G": "C", "T": "A"}
BED_COLUMNS = ["chrom", "start", "end"]


class OneHotEncoder:
    """Encode DNA strings as tensors and decode tensors into strings.

    String encoding uses a batched NumPy lookup internally, then returns a
    CPU tensor. Decoding accepts CPU or GPU tensors; only the base indices and
    unknown-position mask are transferred to the CPU to construct strings.
    Bases outside the alphabet become all-zero positions; locus extraction
    can remove sequences containing them with drop_ambiguous=True.

    Parameters
    ----------
    alphabet : str, default "ACGT"
        Channel order of the sequence alphabet.
    channel_axis : int, default 1
        Channel axis for batched encodings: 1 gives (N, channels, length);
        -1 or 2 gives (N, length, channels). Single-sequence encodings drop
        the batch axis. Decoding uses this setting rather than guessing.
    dtype : torch.dtype, default torch.int8
        Output tensor dtype. Int8 stores hard encodings compactly; callers
        can request torch.float32 or convert before model inference.
    """

    def __init__(
        self,
        alphabet: str = ALPHABET,
        channel_axis: int = 1,
        dtype: torch.dtype = torch.int8,
    ) -> None:
        if channel_axis not in (1, -1, 2):
            raise ValueError("channel_axis must be 1, -1, or 2")

        self.alphabet = alphabet.upper()
        self.vocab_size = len(self.alphabet)
        self.channel_axis = channel_axis
        self.dtype = dtype

        # ASCII lookup; 255 marks a character outside the alphabet.
        self._lookup = np.full(256, 255, dtype=np.uint8)
        for index, character in enumerate(self.alphabet):
            self._lookup[ord(character)] = index
        self._reverse = np.frombuffer(self.alphabet.encode("ascii"), dtype="S1")

    def to_onehot(self, seqs: str | Sequence[str]) -> torch.Tensor:
        """Encode strings, padding shorter sequences with all-zero columns.

        Parameters
        ----------
        seqs : str or Sequence[str]
            Single sequence or a nonempty batch. Bases outside the alphabet
            become all-zero positions. Strings are converted to uppercase.

        Returns
        -------
        onehot : torch.Tensor
            CPU tensor in the configured dtype and channel order. Shape is
            (N, channels, length) or (N, length, channels), depending on
            channel_axis. A single input string drops the batch axis.
        """
        single = isinstance(seqs, str)
        seqs = [seqs] if single else list(seqs)
        if not seqs:
            raise ValueError("no sequences to encode")

        width = max(len(s) for s in seqs)
        if width == 0:
            onehot = torch.zeros(len(seqs), 0, self.vocab_size, dtype=self.dtype)
        else:
            packed = np.array([s.upper() for s in seqs], dtype=f"S{width}")
            codes = self._lookup[packed.view(np.uint8).reshape(len(seqs), width)]
            unknown = codes == 255
            encoded = np.eye(self.vocab_size, dtype=np.int8)[np.where(unknown, 0, codes)]
            encoded[unknown] = 0
            onehot = torch.from_numpy(encoded).to(dtype=self.dtype)
        if self.channel_axis == 1:
            onehot = onehot.transpose(1, 2)
        return onehot[0] if single else onehot

    def from_onehot(self, onehot: torch.Tensor) -> str | list[str]:
        """Decode tensors, replacing undetermined positions with N.

        Parameters
        ----------
        onehot : torch.Tensor
            Single or batched encoding in the configured channel layout.
            May be on any device and may require gradients. Positions whose
            maximum channel value is not 1 decode as N.

        Returns
        -------
        sequences : str or list[str]
            String for a single encoding, otherwise one string per example.
            Decoding is nondifferentiable and does not modify the input.
        """
        if onehot.ndim not in (2, 3):
            raise ValueError("onehot must contain a single sequence or a batch")
        single = onehot.ndim == 2
        values = onehot.unsqueeze(0) if single else onehot
        if self.channel_axis == 1:
            values = values.transpose(1, 2)
        if values.shape[-1] != self.vocab_size:
            raise ValueError("the configured channel axis must match the alphabet size")
        if values.shape[1] == 0:
            return "" if single else [""] * len(values)

        values = values.detach()
        indices = values.to(torch.uint8).argmax(-1) if values.dtype == torch.bool else values.argmax(-1)
        characters = self._reverse[indices.cpu().numpy()].copy()
        characters[(values.amax(-1) != 1).cpu().numpy()] = b"N"
        decoded = [
            row.tobytes().decode("ascii") for row in characters
        ]
        return decoded[0] if single else decoded


def reverse_complement(
    seq: str | torch.Tensor,
    complement_map: Mapping[str, str] | None = None,
) -> str | torch.Tensor:
    """Reverse a DNA sequence and replace each base with its complement.

    Parameters
    ----------
    seq : str or torch.Tensor, shape (..., alphabet_size, length)
        String or encoded sequence. Tensors may contain one sequence, batches,
        or additional leading dimensions. Channels occupy the penultimate axis
        and positions the last axis. Soft encodings are also supported.
    complement_map : Mapping[str, str] or None, default None
        Base-to-complement mapping. None uses COMPLEMENT_MAP, whose keys are
        in ACGT order. For tensors, key order specifies the input channel order,
        so an ATCG encoding uses {"A": "T", "T": "A", "C": "G", "G": "C"}.
        Each complementary base must also be a key. Strings use the mapping
        directly; N is preserved without requiring an entry. Case is preserved
        and lowercase bases need corresponding mapping entries.

    Returns
    -------
    complemented : str or torch.Tensor
        Reverse complement with the same type and shape as seq. Tensors retain
        dtype, device, and gradients. The input is not modified.
    """
    mapping = COMPLEMENT_MAP if complement_map is None else complement_map
    if isinstance(seq, str):
        from tangermeme.utils import reverse_complement as reverse_complement_string

        return reverse_complement_string(seq, complement_map=dict(mapping))

    if not isinstance(seq, torch.Tensor):
        raise TypeError("seq must be a string or torch.Tensor")
    if seq.ndim < 2 or seq.shape[-2] != len(mapping):
        raise ValueError("the penultimate tensor axis must match the complement-map channel count")
    alphabet = list(mapping)
    indices = torch.tensor(
        [alphabet.index(mapping[base]) for base in alphabet], device=seq.device,
        dtype=torch.long,
    )
    return seq.index_select(-2, indices).flip(-1)


def read_fasta(path: str | Path, length: int | None = None) -> tuple[torch.Tensor, np.ndarray]:
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
    onehot : torch.Tensor, shape (N, 4, length)
        CPU one-hot sequences in ACGT order.
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


def read_npz(path: str | Path) -> tuple[torch.Tensor, dict[str, np.ndarray], list[str] | None]:
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
    onehot : torch.Tensor, shape (N, 4, length)
        CPU one-hot sequences in ACGT order.
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
    return torch.from_numpy(onehot), arrays, (None if names is None else [str(n) for n in names])


class Loci(NamedTuple):
    """What one extraction produced, in model row order.

    Attributes
    ----------
    onehot : torch.Tensor, shape (N, 4, in_window)
        CPU int8 one-hot sequences in ACGT order.
    signals
        ``(N, tracks, out_window)`` signal, or None when no bigWigs were given.
    coords
        The surviving loci, with ``chrom``, ``start``, ``end`` and
        ``source_row``: the row of the input frame each example came from.
    """

    onehot: torch.Tensor
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
    keep = torch.ones(len(onehot), dtype=torch.bool)
    if drop_ambiguous:
        keep = (onehot.sum(dim=1) == 1).all(dim=-1)
    rows = np.asarray(kept)[keep.numpy()]
    surviving = frame.loc[rows].reset_index(drop=True)
    surviving["source_row"] = rows
    return Loci(
        onehot=onehot[keep],
        signals=np.stack(signal_blocks)[keep.numpy()] if signal_blocks else None,
        coords=surviving,
    )
