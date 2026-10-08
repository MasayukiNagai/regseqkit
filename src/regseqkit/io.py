"""Artifacts on disk: named arrays beside a JSON sidecar.

A producer writes ``<stem>.npz`` and ``<stem>.json`` together and names the
stem after what it holds, so a directory listing says what is in it. The
sidecar carries the model metadata, which is what lets a scoring stage read a
prediction without loading any model.

``numpy`` is the only import here. Reading an artifact is meant to be cheap.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


# A name that becomes part of a filename: scorer names, objective ids, job
# names. Leading dot, slash and whitespace are what this exists to reject.
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def model_output_space(settings: Mapping[str, Any]) -> str:
    """Resolve the output space, including uniform legacy scorer units.

    Parameters
    ----------
    settings : Mapping
        Configuration settings or artifact definitions. model_output.space
        may be "native" or "counts"; the default is "native". Legacy scalar
        scorers with units "log" or "raw" select native or counts respectively.

    Returns
    -------
    space : str
        One output space shared by all scalar scorers.

    Raises
    ------
    ValueError
        If the space is unknown, legacy units disagree, or old and new settings
        conflict. Mixed spaces require separate model output views.
    """
    configured = settings.get("model_output", {}).get("space")
    entries = [s for s in settings.get("scorers", []) if s.get("head", "scalar") == "scalar"]
    legacy = None
    if any("units" in s for s in entries):
        units = {s.get("units", "raw") for s in entries}
        if not units.issubset({"raw", "log"}):
            raise ValueError("legacy scorer units must be 'raw' or 'log'")
        if len(units) != 1:
            raise ValueError("mixed scorer units require separate model output views")
        legacy = "native" if units == {"log"} else "counts"
    space = configured if configured is not None else (legacy or "native")
    if space not in ("native", "counts"):
        raise ValueError("model_output.space must be 'native' or 'counts'")
    if legacy is not None and space != legacy:
        raise ValueError("model_output.space conflicts with legacy scorer units")
    return space


def scalar_score_transform(metadata: Mapping[str, Any], name: str) -> str:
    """Read a scalar score's encoding from an artifact.

    Parameters
    ----------
    metadata : Mapping
        Artifact sidecar containing model metadata and scorer definitions.
    name : str
        Scorer name. Legacy per-scorer units are read when present.

    Returns
    -------
    transform : str
        "log1p" or "identity" before optional scorer standardization.
    """
    definitions = metadata.get("definitions", {})
    entry = next((s for s in definitions.get("scorers", []) if s["name"] == name), {})
    if "units" in entry:
        return "log1p" if entry["units"] == "log" else "identity"
    space = model_output_space(definitions)
    return "identity" if space == "counts" else metadata.get("model", {}).get(
        "scalar_transform", "identity"
    )


def safe_name(name: str, kind: str = "name") -> str:
    """Return `name` if it is safe to put in a filename, else raise.

    Parameters
    ----------
    name
        The candidate.
    kind
        What the name is, used in the error message.

    Returns
    -------
    str
        `name`, unchanged.

    Raises
    ------
    ValueError
        If `name` is empty or contains anything but letters, digits, and
        ``_.-`` after an alphanumeric first character.
    """
    if not isinstance(name, str) or not _SAFE_NAME.fullmatch(name):
        raise ValueError(
            f"{kind} {name!r} is not filesystem-safe; use letters, digits, "
            "underscore, dot or hyphen, starting with a letter or digit"
        )
    return name


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


def validate_examples(arrays: dict[str, np.ndarray], outputs: Sequence[str] | None = None) -> None:
    """Assert unique example IDs and a shared example axis across every array.

    Parameters
    ----------
    arrays
        Must contain ``ids``; every entry is checked against its length, so
        anything that is not example-indexed belongs outside this dict.
    outputs
        Named model outputs. When given, ``observed`` and ``predicted`` must be
        ``(len(ids), len(outputs))``.

    Raises
    ------
    ValueError
        If IDs are not a unique vector, or any array's axes disagree.
    """
    if "ids" not in arrays:
        raise ValueError("example arrays must contain 'ids'")
    ids = arrays["ids"]
    if ids.ndim != 1 or len(set(ids.tolist())) != len(ids):
        raise ValueError("example IDs must be a unique vector")
    for name, value in arrays.items():
        if value.shape[0] != len(ids):
            raise ValueError(f"{name}: example axis does not match IDs")
    if outputs is not None:
        for key in ("observed", "predicted"):
            if key in arrays and arrays[key].shape != (len(ids), len(outputs)):
                raise ValueError(f"{key}: output axis does not match named outputs")
