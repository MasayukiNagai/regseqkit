"""Artifacts on disk: named arrays beside a JSON sidecar.

A producer writes ``<stem>.npz`` and ``<stem>.json`` together and names the
stem after what it holds, so a directory listing says what is in it. The
sidecar carries the model metadata, which is what lets a scoring stage read a
prediction without loading any model.

``numpy`` is the only import here. Reading an artifact is meant to be cheap.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def add_extension(path: str | Path, extension: str) -> Path:
    """Append an extension to a stem instead of replacing its last one.

    ``Path.with_suffix`` is wrong for these names: a stem like
    ``multitask.seed0.test`` already contains dots, and ``with_suffix(".npz")``
    would rewrite it to ``multitask.seed0.npz``. Every artifact name carries its
    model, seed and split that way, so extensions are always appended.

    Parameters
    ----------
    path
        File stem, which may itself contain dots.
    extension
        Extension to append, leading dot included.

    Returns
    -------
    Path
        `path` with `extension` appended to its final component.
    """
    path = Path(path)
    return path.parent / f"{path.name}{extension}"


def strip_extension(path: str | Path, extensions: Sequence[str] = (".npz", ".json")) -> Path:
    """Recover an artifact's stem, given the stem or either member.

    Parameters
    ----------
    path
        Artifact stem or one of its files.
    extensions
        Extensions to strip, tried in order.

    Returns
    -------
    Path
        `path` without a trailing member extension, unchanged if none matched.
    """
    path = Path(path)
    for extension in extensions:
        if path.name.endswith(extension):
            return path.parent / path.name[: -len(extension)]
    return path


def save_arrays(path: str | Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> Path:
    """Write ``<path>.npz`` and ``<path>.json``, creating parent directories.

    Parameters
    ----------
    path
        File stem, not a directory: the caller chooses a name saying what the
        arrays are, e.g. ``predictions/multitask.seed0.test``.
    arrays
        Named arrays, saved compressed.
    metadata
        JSON-serializable sidecar. Should carry enough model provenance for a
        consumer to score without loading any model.

    Returns
    -------
    Path
        The written ``.npz`` path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(add_extension(path, ".npz"), **arrays)
    add_extension(path, ".json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return add_extension(path, ".npz")


def load_arrays(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load ``<path>.npz`` and its ``.json`` sidecar.

    Parameters
    ----------
    path
        Artifact stem or either member of the pair, so a command-line argument
        can name the file the user actually sees.

    Returns
    -------
    arrays : dict of ndarray
        Contents of the ``.npz``.
    metadata : dict
        Contents of the ``.json`` sidecar.
    """
    path = strip_extension(path)
    with np.load(add_extension(path, ".npz"), allow_pickle=False) as source:
        arrays = dict(source)
    metadata = json.loads(add_extension(path, ".json").read_text())
    return arrays, metadata
