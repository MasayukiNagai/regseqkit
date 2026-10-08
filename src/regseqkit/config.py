"""One JSON file, read into resolved arguments for the rest of the package.

Every other module takes plain arguments: absolute paths, dicts, arrays. This
is the only module that knows a configuration file exists, so a project with a
different layout replaces this one and reuses everything else.

The file is a set of independent blocks and a stage reads only what it needs:

``data``
    ``fasta``, ``loci``, ``negatives``, ``exclusion_lists``, or ``examples``
    for a prepared NPZ instead of a genome.
``splits``
    Split name to its chromosomes.
``tracks``
    ``name`` and ``path`` per observed track. ``path`` may be a list when one
    track has several channels, as a stranded pair does.
``models``
    One entry per checkpoint, with the ``outputs`` it emits in order. Several
    entries concatenate, so single-task checkpoints look like one multitask
    model.
``scorers``
    ``name`` and ``weights``, a mapping from output name to weight.
``model_output``
    ``space``, either ``native`` (default) or ``counts``. Conversion is applied
    once to the scalar model output before any scorer reads it.
``calibration``
    ``path`` to a ``calibration.json`` and the ``reference`` set within it, used
    to standardize every scalar scorer. Optional; omitted, scorers read raw
    model units.
``design``
    ``start``, ``end``, ``templates``, ``objectives``, and per-optimizer
    settings. Each objective names a scorer and a mode or combines
    sub-objectives in an objectives list. ``greedy`` holds
    ``max_iter``, ``batch_size``, and ``tol`` for single-site substitutions;
    ``ledidi`` holds ``max_iter``, ``batch_size``, ``n_samples``, ``l`` and ``random_state``.

**The design block can live in its own file, called a recipe**, which is what
:func:`load_recipe` is for. Everything above it describes the model and what
can be measured from it and changes rarely; a recipe describes what you are
trying to make and changes every run. Splitting them means a sweep over an
optimizer parameter is several three-line files rather than several copies of
the whole configuration. A recipe holds those keys at the top level, with no
enclosing ``design``, and the designed sequences it produces are what the word
*design* means everywhere else here.

Paths resolve relative to the file they are written in, so either file can name
inputs beside itself.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .calibrate import Calibration, resolve_percentile_targets
from .design import Objective, WeightedObjective
from .design.objectives import ObjectiveProtocol
from .io import model_output_space, safe_name
from .metrics import pool_channels
from .scoring import ProfileScorer, ScalarScorer, Scorer
from .sequences import Loci, extract_loci_with_coords, read_npz
from .wrappers import Log1pToCounts


_HEADS = ("scalar", "profile")
_SCALAR_TRANSFORMS = ("identity", "log1p")


@dataclass(frozen=True)
class ModelMetadata:
    """Resolved model information for configured input preparation and artifacts.

    This is written into every artifact's sidecar, which is what lets a scoring
    stage read a prediction without importing torch or any model library.

    Attributes
    ----------
    outputs : tuple[str, ...]
        Named scalar outputs, in the order the model emits them. Unique.
    in_window : int
        Input sequence width in base pairs.
    out_window : int or None, default None
        Profile width, or None for a scalar-only model. Must fit inside
        in_window.
    groups : tuple[int, ...], default ()
        Signal channels per output, in output order. Empty for a scalar-only
        model; (2, 2) for two stranded outputs.
    scalar_transform : str, default "identity"
        "identity" or "log1p": what the scalars are already in, so a
        consumer can put observations in the same units before comparing.
        Scoring wrappers can convert predictions without changing this metadata.
    """

    outputs: tuple[str, ...]
    in_window: int
    out_window: int | None = None
    groups: tuple[int, ...] = ()
    scalar_transform: str = "identity"

    def __post_init__(self) -> None:
        """Reject metadata a consumer could silently misinterpret.

        Raises
        ------
        ValueError
            If output names are empty or duplicated, the windows do not nest,
            the groups do not describe every output, or the transform is
            unrecognized.
        """
        if (
            not self.outputs
            or any(not n for n in self.outputs)
            or len(set(self.outputs)) != len(self.outputs)
        ):
            raise ValueError("output names must be nonempty and unique")
        if self.in_window < 1:
            raise ValueError("in_window must be positive")
        if self.out_window is not None and not 0 < self.out_window <= self.in_window:
            raise ValueError("out_window must lie inside in_window")
        if self.groups and (len(self.groups) != len(self.outputs) or min(self.groups) < 1):
            raise ValueError("one positive channel-group size is required per output")
        if self.scalar_transform not in _SCALAR_TRANSFORMS:
            raise ValueError(f"scalar_transform must be one of {_SCALAR_TRANSFORMS}")


@dataclass(frozen=True)
class ConfigFile:
    """Settings, plus the file they came from so paths can resolve.

    One type for both files a design run reads: the project configuration, and
    the recipe, whose keys sit at the top level rather than under ``design``.
    """

    path: Path
    settings: dict[str, Any]

    def resolve_path(self, value: str | Path) -> Path:
        """Resolve a configured path relative to the configuration file."""
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.path.parent / path
        return path.resolve()

    def section(self, name: str) -> dict[str, Any]:
        """A copy of one mapping block, empty when it is absent."""
        return dict(self.settings.get(name, {}))

    def entries(self, name: str) -> list[dict[str, Any]]:
        """A copy of one list block, empty when it is absent."""
        return [dict(entry) for entry in self.settings.get(name, [])]


def load_config(path: str | Path) -> ConfigFile:
    """Load the JSON configuration. Callers apply their own CLI overrides."""
    path = Path(path).resolve()
    return ConfigFile(path=path, settings=json.loads(path.read_text()))


def build_scorers(
    entries: Sequence[Mapping[str, Any]],
    outputs: Sequence[str],
    *,
    standardize: tuple[Any, Any] | None = None,
) -> dict[str, Scorer]:
    """Build scorers from configuration entries.

    Parameters
    ----------
    entries : Sequence[Mapping]
        One mapping per scorer, with name and weights. Weights map
        an output name to its weight; unlisted outputs weigh zero.
        A head value of "profile" builds a ProfileScorer and may carry
        absolute and reduction settings.
    outputs : Sequence[str]
        Model output names, giving the weight vector's order.
    standardize : tuple or None, default None
        Optional (center, scale) per model output, applied to every scalar
        scorer that does not opt out with a standardize value of False.

    Returns
    -------
    scorers : dict[str, Scorer]
        Scorer identifiers mapped to numerical scorers, in entry order.

    Raises
    ------
    ValueError
        If a name is missing, unsafe or repeated, the weights name an unknown
        output, or a field is invalid for the chosen head.
    """
    index = {name: position for position, name in enumerate(outputs)}
    built: dict[str, Scorer] = {}
    seen: set[str] = set()

    for position, entry in enumerate(entries):
        if "units" in entry or "transform" in entry:
            raise ValueError(
                "scorers do not convert outputs; choose model_output.space in the "
                "configuration or wrap the model before scoring"
            )
        name = safe_name(str(entry.get("name", "")), f"scorer {position}'s name")
        if name in seen:
            raise ValueError(f"duplicate scorer name {name!r}")
        seen.add(name)

        weights = entry.get("weights") or {}
        unknown = [key for key in weights if key not in index]
        if unknown:
            raise ValueError(f"{name}: unknown output(s) {unknown}; have {list(outputs)}")
        vector = torch.zeros(len(outputs), dtype=torch.float32)
        for key, value in weights.items():
            vector[index[key]] = float(value)

        head = str(entry.get("head", "scalar"))
        if head not in _HEADS:
            raise ValueError(f"{name}: head must be one of {_HEADS}, got {head!r}")
        if head == "profile":
            built[name] = ProfileScorer(
                vector,
                absolute=bool(entry.get("absolute", True)),
                reduction=str(entry.get("reduction", "mean")),
            )
        else:
            wanted = entry.get("standardize", True)
            built[name] = ScalarScorer(
                vector,
                standardize=standardize if (standardize and wanted) else None,
            )
    return built


def identity_scorers(outputs: Sequence[str]) -> dict[str, ScalarScorer]:
    """One scorer per model output, each selecting only itself.

    The default when a project configures no scorers.

    Parameters
    ----------
    outputs : Sequence[str]
        Model output names.

    Returns
    -------
    scorers : dict[str, ScalarScorer]
        Output names mapped to their selecting scorers.
    """
    eye = torch.eye(len(outputs))
    return {
        name: ScalarScorer(eye[position])
        for position, name in enumerate(outputs)
    }


def build_objectives(
    entries: Sequence[Mapping[str, Any]], scorers: Mapping[str, Scorer],
    *, template_scores: Mapping[str, float] | None = None,
) -> dict[str, ObjectiveProtocol]:
    """Build standalone or weighted objectives from configuration.

    Parameters
    ----------
    entries : Sequence[Mapping]
        One named objective per mapping. A standalone objective has scorer,
        mode and an optional target. A weighted objective has an objectives
        list; each sub-objective carries an optional weight, defaulting to 1.
        Weighted objectives may contain other weighted objectives.
    scorers : Mapping[str, Scorer]
        The scorers available to be named.
    template_scores : Mapping[str, float] or None, default None
        Starting scores by scorer id. Resolves a configured match target of
        "template" into a numeric target before constructing the objective.

    Returns
    -------
    dict
        Objective id to objective, in entry order.

    Raises
    ------
    ValueError
        If a name is missing, unsafe or repeated, or an objective is invalid.
    KeyError
        If an objective names a scorer that does not exist.
    """
    def build(entry: Mapping[str, Any]) -> ObjectiveProtocol:
        if "terms" in entry or "value" in entry:
            raise ValueError("use 'objectives' for composition and 'target' for matching")
        if "objectives" in entry:
            if any(key in entry for key in ("scorer", "mode", "target")):
                raise ValueError("a weighted objective cannot also define scorer, mode or target")
            sub_objectives = entry["objectives"]
            return WeightedObjective(
                objectives=tuple(build(sub_objective) for sub_objective in sub_objectives),
                weights=tuple(
                    float(sub_objective.get("weight", 1.0))
                    for sub_objective in sub_objectives
                ),
            )
        wanted = str(entry.get("scorer", ""))
        if not wanted:
            raise ValueError("an objective needs 'objectives' or a 'scorer'")
        if wanted not in scorers:
            raise KeyError(f"unknown scorer {wanted!r}; have {sorted(scorers)}")
        target = entry.get("target")
        if target == "template":
            if entry.get("mode") != "match":
                raise ValueError("target 'template' requires mode 'match'")
            if template_scores is None or wanted not in template_scores:
                raise ValueError(f"missing template score for {wanted!r}")
            target = template_scores[wanted]
        return Objective(
            scorer=scorers[wanted],
            mode=str(entry.get("mode", "")),
            target=None if target is None else float(target),
        )

    built: dict[str, ObjectiveProtocol] = {}

    for position, entry in enumerate(entries):
        name = safe_name(str(entry.get("name", "")), f"objective {position}'s name")
        if name in built:
            raise ValueError(f"duplicate objective name {name!r}")

        objective = build(entry)
        if "weight" in entry:
            objective = WeightedObjective((objective,), weights=(entry["weight"],))
        built[name] = objective
    return built


def load_calibration(config: ConfigFile) -> tuple[Calibration | None, str | None]:
    """The configured calibration and the reference set to standardize against.

    Returns ``(None, None)`` when no ``calibration`` block is present, which is
    the identity: scorers then read raw model units.

    Raises
    ------
    ValueError
        If the block names a ``path`` but no ``reference``. Which set a scale
        comes from is the method, so it is never defaulted.
    """
    section = config.section("calibration")
    if not section.get("path"):
        return None, None
    if not section.get("reference"):
        raise ValueError(
            f"{config.path}: 'calibration' names a path but no 'reference' set; "
            "which reference set a scale comes from is the method, so say which"
        )
    return Calibration.load(config.resolve_path(section["path"])), str(section["reference"])


def configured_head(config: ConfigFile) -> str:
    """Resolve output selection from configuration, outside numerical scorers."""
    configured = config.section("model_output").get("head")
    heads = {entry.get("head", configured or "scalar") for entry in config.entries("scorers")}
    if len(heads) > 1:
        raise ValueError("configure one model output view per workflow")
    head = configured or next(iter(heads), "scalar")
    if head not in _HEADS or (heads and heads != {head}):
        raise ValueError("model_output.head must be scalar or profile and match scorer definitions")
    return head


def prepare_output_module(
    config: ConfigFile,
    module: torch.nn.Module,
    metadata: ModelMetadata,
    head: str | None = None,
) -> torch.nn.Module:
    """Apply configured output conversion to an already selected model output.

    Parameters
    ----------
    config : ConfigFile
        Output space configuration.
    module : torch.nn.Module
        Prepared module returning the scalar or profile tensor to score.
    metadata : ModelMetadata
        Native scalar encoding for the configured workflow.
    head : str or None, default None
        Selected head. None uses the configuration.

    Returns
    -------
    module : torch.nn.Module
        Supplied module, optionally wrapped to convert log1p scalars to counts.
    """
    space = model_output_space(config.settings)
    head = configured_head(config) if head is None else head
    if head == "scalar" and space == "counts" and metadata.scalar_transform == "log1p":
        return Log1pToCounts(module)
    return module


def load_scorers(config: ConfigFile, metadata: ModelMetadata) -> dict[str, Scorer]:
    """Build the ``scorers`` block, or one scorer per model output if absent.

    When a ``calibration`` block is configured, every scalar scorer is
    standardized against its reference set, so outputs of different spread
    contribute on the same footing and a term weight in an objective means what
    it says. A scorer opts out with ``"standardize": false``.

    Parameters
    ----------
    config
        Read for its ``scorers`` and ``calibration`` blocks.
    metadata
        Supplies output names and the native encoding for calibration checks.

    Returns
    -------
    dict[str, Scorer]
    """
    space = model_output_space(config.settings)
    entries = config.entries("scorers")
    head = configured_head(config)
    if any(e.get("units") == "log" for e in entries) and metadata.scalar_transform != "log1p":
        raise ValueError("legacy log scorers require log1p model predictions")
    for entry in entries:
        entry.pop("units", None)
        entry.setdefault("head", head)
    calibration, reference = load_calibration(config)
    transform = "identity" if space == "counts" else metadata.scalar_transform
    if calibration is not None:
        calibrated_transform = calibration.provenance.get("scalar_transform", metadata.scalar_transform)
        if calibrated_transform != transform:
            raise ValueError(
                "calibration output space differs from model_output.space; "
                "rebuild calibration with 2_calibrate.py --space " + space
            )
    standardize = (
        calibration.center_scale(reference, metadata.outputs) if calibration else None
    )

    if not entries:
        return identity_scorers(metadata.outputs)
    return build_scorers(
        entries,
        metadata.outputs,
        standardize=standardize,
    )


def load_recipe(project: ConfigFile, path: str | Path | None = None) -> ConfigFile:
    """The recipe for one design run, from its own file or the project's block.

    A project file describes the model and what can be measured from it, and
    changes rarely. A recipe describes what you are trying to make: the editable
    interval, the templates, the objectives, and the optimizer settings. Those
    are what vary from run to run, so keeping them in their own small file means
    a sweep is several three-line files rather than several copies of
    everything.

    Paths inside the returned settings resolve relative to whichever file they
    came from, so a recipe can name a template FASTA beside itself.

    Parameters
    ----------
    project
        The project configuration, used as the fallback source.
    path
        A recipe file. When omitted, the project's own ``design`` block is used,
        which is what makes the split optional.

    Returns
    -------
    ConfigFile
        Recipe settings at the top level: ``objectives``, ``greedy``, ``start``
        and so on, with no enclosing ``design`` key.
    """
    if path is not None:
        return load_config(path)
    return ConfigFile(project.path, project.section("design"))


def run_name(project: ConfigFile, recipe: ConfigFile) -> str:
    """A label for one design run, used to tell concatenated runs apart.

    The recipe's own stem, which costs nothing to maintain and cannot drift
    from the settings it names. ``inline`` when the recipe came from the
    project's own block rather than a file of its own.
    """
    return "inline" if recipe.path == project.path else recipe.path.stem


def configured_objectives(
    recipe: ConfigFile,
    scorers: Mapping[str, Scorer],
    *,
    calibration: Calibration | None = None,
    reference: str | None = None,
    template_scores: Mapping[str, float] | None = None,
) -> dict[str, ObjectiveProtocol]:
    """Build standalone and weighted objectives from a recipe.

    Parameters
    ----------
    recipe
        From :func:`load_recipe`, so ``objectives`` sits at the top level.
    scorers
        The scorers available to be named.
    calibration, reference
        From the *project*, since a scale is a property of the model rather
        than of one design run. When given, ``pNN`` values resolve against the
        named reference set first.
    template_scores : Mapping[str, float] or None, default None
        Starting scores by scorer id, used for match targets of "template".
    """
    entries = recipe.settings.get("objectives", [])
    if calibration is not None and reference is not None:
        entries = resolve_percentile_targets(entries, calibration, reference)
    return build_objectives(entries, scorers, template_scores=template_scores)


def artifact_metadata(
    config: ConfigFile, metadata: ModelMetadata, entries: list[dict[str, Any]], **extra: Any
) -> dict[str, Any]:
    """The sidecar fields every artifact records: which model, under which configuration.

    ``definitions`` holds the scorer entries and the calibration block verbatim,
    so a later stage can tell whether a scorer changed between two artifacts
    rather than assuming it did not.

    Parameters
    ----------
    config
        The project configuration the artifact was produced under.
    metadata
        Resolved model information for this workflow.
    entries
        Resolved checkpoint entries recorded by the project loading workflow.
    **extra
        Stage-specific fields, recorded as given.
    """
    return dict(
        model=asdict(metadata),
        models=entries,
        config=str(config.path),
        resolved_config=config.settings,
        definitions={
            "scorers": config.entries("scorers"),
            "calibration": config.section("calibration"),
            "model_output": {
                "space": model_output_space(config.settings), "head": configured_head(config),
            },
        },
        **extra,
    )


def _signal_paths(config: ConfigFile, metadata: ModelMetadata) -> list[str]:
    """Resolved bigWig paths, one per model output channel, in output order.

    Parameters
    ----------
    config
        Read for its ``tracks`` block.
    metadata
        Supplies the output names and the channel count of each.

    Returns
    -------
    list of str

    Raises
    ------
    ValueError
        If a track name repeats, an output has no track, or a track supplies
        the wrong number of channels.
    """
    tracks = {}
    for entry in config.entries("tracks"):
        name = str(entry["name"])
        if name in tracks:
            raise ValueError(f"duplicate track {name!r}")
        paths = entry["path"]
        tracks[name] = [paths] if isinstance(paths, (str, Path)) else list(paths)

    groups = metadata.groups or (1,) * len(metadata.outputs)
    signals: list[str] = []
    for name, size in zip(metadata.outputs, groups):
        if name not in tracks:
            raise ValueError(f"no track named {name!r}; have {sorted(tracks)}")
        if len(tracks[name]) != size:
            raise ValueError(
                f"{name}: the model expects {size} channel(s) but the track "
                f"lists {len(tracks[name])} path(s)"
            )
        signals.extend(str(config.resolve_path(p)) for p in tracks[name])
    return signals


def load_inputs(
    config: ConfigFile,
    metadata: ModelMetadata,
    split: str = "test",
    limit: int | None = None,
    *,
    signal: bool = False,
    drop_ambiguous: bool = False,
    loci: str | Path | None = None,
) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
    """Load model inputs, from a prepared NPZ or from the genome.

    Two sources. ``data.examples`` names a prepared NPZ, one fixed set of
    sequences for a model with no genome behind it; that is what lets the
    stages run on a toy model without a reference FASTA. Otherwise loci are
    extracted from ``data.fasta`` for one split.

    Parameters
    ----------
    config
        Read for ``data`` and ``splits``.
    metadata
        Model metadata, for the windows, groups and scalar transform.
    split
        A key of ``splits``. Not consulted for prepared examples, which are
        already one fixed set.
    limit
        Keep only the first this many surviving examples.
    signal
        Also load observed signal. Genomic extraction needs profile groups for
        this; a scalar-only model must use prepared examples.
    drop_ambiguous
        Drop examples whose window contains a base outside ACGT. Attribution
        needs this, since in-silico mutagenesis cannot mutate an all-zero
        column. Applies to both sources.
    loci
        Extract from this BED instead of ``data.loci``. Pointing inference at
        the GC-matched negatives is how a calibration reference set is built.
        Genomic only: prepared examples already are the sequences.

    Returns
    -------
    X : torch.Tensor
        ``(N, 4, in_window)`` one-hot ACGT.
    arrays : dict of ndarray
        ``ids`` plus, for genomic inputs, ``chrom``/``start``/``end``, and when
        `signal` is set, ``observed`` and ``observed_profiles``.

    Raises
    ------
    ValueError
        If `limit` is not positive, `split` is not a configured split, a
        prepared NPZ disagrees with the model or is combined with `loci`, or
        signal is requested from a scalar-only model.
    """
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    data = config.section("data")

    if data.get("examples"):
        if loci is not None:
            raise ValueError(
                "'data.examples' already supplies the sequences, so there is no "
                "genome to extract `loci` from; point 'data.examples' elsewhere instead"
            )
        onehot, arrays, outputs = read_npz(config.resolve_path(data["examples"]))
        if signal and (outputs is None or list(outputs) != list(metadata.outputs)):
            raise ValueError("prepared 'output_names' must match the model output order")
        if onehot.shape[1:] != (4, metadata.in_window):
            raise ValueError("prepared 'onehot' must be (examples, 4, model input window)")
        rows = np.arange(len(onehot))
        if drop_ambiguous:
            rows = rows[(onehot.sum(axis=1) == 1).all(axis=-1)]
        rows = rows[:limit]
        return torch.from_numpy(onehot[rows]), {k: v[rows] for k, v in arrays.items()}

    if signal and (not metadata.groups or metadata.out_window is None):
        raise ValueError(
            "genomic signal extraction requires profile groups; use prepared "
            "examples for scalar-only models"
        )
    splits = config.section("splits")
    if split not in splits:
        raise ValueError(f"{config.path}: no split {split!r} in 'splits'; have {sorted(splits)}")

    extracted = extract_loci_with_coords(
        config.resolve_path(loci) if loci else config.resolve_path(data["loci"]),
        config.resolve_path(data["fasta"]),
        chroms=splits[split],
        signals=_signal_paths(config, metadata) if signal else None,
        in_window=metadata.in_window,
        out_window=metadata.out_window or metadata.in_window,
        exclusion_lists=[config.resolve_path(p) for p in data.get("exclusion_lists", [])],
        limit=limit,
        drop_ambiguous=drop_ambiguous,
    )

    return torch.from_numpy(extracted.onehot), _genomic_arrays(extracted, metadata)


def _genomic_arrays(extracted: Loci, metadata: ModelMetadata) -> dict[str, np.ndarray]:
    """The example-indexed arrays of one extraction: ids, coordinates, and any observed signal."""
    arrays: dict[str, np.ndarray] = {
        "ids": np.asarray([f"locus_{row}" for row in extracted.coords["source_row"]]),
        "chrom": extracted.coords["chrom"].to_numpy(dtype=str),
        "start": extracted.coords["start"].to_numpy(dtype=np.int64),
        "end": extracted.coords["end"].to_numpy(dtype=np.int64),
    }
    if extracted.signals is not None:
        arrays["observed_profiles"] = extracted.signals.astype(np.float32)
        observed = pool_channels(extracted.signals.sum(-1), metadata.groups)
        arrays["observed"] = (
            np.log1p(observed) if metadata.scalar_transform == "log1p" else observed
        )
    return arrays


def load_inputs_at(
    config: ConfigFile,
    metadata: ModelMetadata,
    coords: pd.DataFrame,
    *,
    signal: bool = True,
) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
    """Model inputs at explicit loci, with no split and no exclusion list.

    The sibling of :func:`load_inputs` for the loci a figure names rather than
    the loci a split contains. A peak chosen for what the experiment measured
    there is drawn even when it overlaps an exclusion list, and it may sit on
    a training chromosome; the caller says so where that matters. The only
    rows dropped are those whose windows run off their chromosome.

    Observed signal is read over the model's own output window, so observed
    and predicted profiles sit on identical positions by construction. A
    separate track read with a flank of its own would reintroduce exactly the
    off-by-one the evaluation stage spends a validation gate ruling out.

    Parameters
    ----------
    config
        Read for ``data.fasta`` and ``tracks``.
    metadata
        Model metadata, for the windows, groups and scalar transform.
    coords
        ``chrom``, ``start`` and ``end`` per locus, zero-based half-open. Other
        columns are ignored, so rows of a peak table pass through as they are.
    signal
        Also read the observed signal, as ``observed`` and ``observed_profiles``.

    Returns
    -------
    X, arrays
        As :func:`load_inputs` returns them for genomic inputs.

    Raises
    ------
    ValueError
        If the configuration supplies prepared examples, which are already one
        fixed set of sequences with no genome to extract from, or if signal is
        requested from a scalar-only model.
    """
    data = config.section("data")
    if data.get("examples"):
        raise ValueError(
            "'data.examples' already supplies the sequences, so there is no genome "
            "to extract loci from"
        )
    if signal and (not metadata.groups or metadata.out_window is None):
        raise ValueError("genomic signal extraction requires profile groups")

    extracted = extract_loci_with_coords(
        coords,
        config.resolve_path(data["fasta"]),
        signals=_signal_paths(config, metadata) if signal else None,
        in_window=metadata.in_window,
        out_window=metadata.out_window or metadata.in_window,
    )
    return torch.from_numpy(extracted.onehot), _genomic_arrays(extracted, metadata)


def configured_job_arguments(config: ConfigFile) -> dict[str, Any]:
    """Resolved keyword arguments for a training-job builder.

    Returns them rather than calling a builder, so this module imports no model
    library and a project can hand them to whichever builder it trains with.

    Returns
    -------
    dict
        ``fasta``, ``loci``, ``negatives``, ``exclusion_lists``, ``tracks``,
        ``splits``, ``parameters``, ``name``, ``mode``, ``seeds``, ``source``.
    """
    data = config.section("data")
    training = config.section("training")
    tracks = []
    for entry in config.entries("tracks"):
        paths = entry["path"]
        paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
        tracks.append(
            dict(entry, track=str(entry["name"]), signal=str(config.resolve_path(paths[0])))
        )
    return dict(
        fasta=config.resolve_path(data["fasta"]),
        loci=config.resolve_path(data["loci"]),
        negatives=config.resolve_path(data["negatives"]) if data.get("negatives") else None,
        exclusion_lists=[config.resolve_path(p) for p in data.get("exclusion_lists", [])],
        tracks=tracks,
        splits=config.section("splits"),
        parameters=training.get("parameters", {}),
        name=training.get("name", "multitask"),
        mode=training.get("mode", "multitask"),
        seeds=training.get("seeds", [0]),
        source=config.path,
    )
