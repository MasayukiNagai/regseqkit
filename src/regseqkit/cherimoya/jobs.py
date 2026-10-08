"""Job specs and the fit JSON a Cherimoya run records.

A job spec says what to train; the fit JSON beside the checkpoint says what the
run actually saw. The **fit JSON is authoritative** for anything the model was
exposed to -- resolved bigWigs, peak BED, exclusion lists, window sizes -- so a
later stage reads it rather than re-deriving those from configuration. That is
what makes a downstream stage's setup match the training setup rather than
merely resemble it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from cherimoya.io import normalize_signal_groups


# Which key of the job JSON each split reads its chromosomes from. The builder
# writes all three, so split names stay meaningful for any spec it made.
SPLIT_CHROM_KEYS = {
    "train": "training_chroms",
    "valid": "validation_chroms",
    "test": "test_chroms",
}


@dataclass
class Job:
    """One trained model, with everything read off disk rather than guessed.

    Attributes
    ----------
    name
        Job name, which is also the run directory and checkpoint stem.
    path
        The ``*.job.json`` this was read from.
    spec
        That file's contents.
    fit
        The run's ``*.fit.json``, authoritative for what the model actually saw.
    run_dir
        Directory holding the checkpoint, fit JSON, and logs.
    model_path
        The selected checkpoint.
    tracks
        Output names, in model output order.
    signals
        Resolved bigWig paths, in the same order.
    signal_groups
        Channels per output.
    mode
        ``"single-task"`` or ``"multitask"``.
    track_records
        The spec's track rows verbatim, so a project can read its own metadata
        columns without this module naming them.
    """

    name: str
    path: Path
    spec: dict
    fit: dict
    run_dir: Path
    model_path: Path
    tracks: list[str]
    signals: list[str]
    signal_groups: list[int]
    mode: str
    # The spec's track rows verbatim, so a project can read its own metadata
    # columns (a condition, a replicate) without this module naming them.
    track_records: list[dict]


def expand_jobs(paths: list[Path]) -> list[Path]:
    """Resolve each argument to job JSONs, expanding any jobs.txt manifest.

    Parameters
    ----------
    paths
        A mix of ``*.job.json`` files and manifests listing them. Manifest
        entries may be relative, in which case they resolve against the
        manifest's own directory.

    Returns
    -------
    list of Path
        Absolute job-spec paths, deduplicated, first occurrence winning.

    Raises
    ------
    FileNotFoundError
        If an argument does not exist.
    ValueError
        If a ``.json`` argument is not a ``*.job.json``, or nothing resolved.
    """
    resolved: list[Path] = []
    for path in paths:
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"no such file: {path}")
        if path.name.endswith(".job.json"):
            resolved.append(path)
            continue
        if path.suffix == ".json":
            raise ValueError(
                f"{path.name} is not a *.job.json; pass a job spec written by "
                "the job builder or a jobs.txt listing them"
            )
        for line in path.read_text().split():
            entry = Path(line)
            resolved.append(entry if entry.is_absolute() else (path.parent / entry))

    seen: dict[Path, None] = {}
    for path in resolved:
        seen.setdefault(path.resolve(), None)
    if not seen:
        raise ValueError("no jobs to load")
    return list(seen)


def load_job(path: Path, run_root: Path | None, checkpoint: str = "best") -> Job:
    """Read one job spec plus the fit JSON its run directory recorded.

    Parameters
    ----------
    path
        A ``*.job.json``.
    run_root
        Directory holding run subdirectories, defaulting to ``../runs`` beside
        the spec.
    checkpoint
        ``"best"`` for ``<name>.torch``, anything else for ``<name>.final.torch``.

    Returns
    -------
    Job

    Raises
    ------
    FileNotFoundError
        If the fit JSON or the checkpoint is absent.
    ValueError
        If the spec's tracks and the fit JSON's signal groups do not correspond
        one to one, in the same order.
    """
    path = Path(path)
    spec = json.loads(path.read_text())
    name = str(spec["name"])
    root = (Path(run_root) if run_root else path.parent.parent / "runs").resolve()
    run_dir = root / name

    fit_path = run_dir / f"{name}.fit.json"
    if not fit_path.is_file():
        raise FileNotFoundError(f"no fit JSON at {fit_path}")
    fit = json.loads(fit_path.read_text())

    # The training CLI writes every key, but a hand-made fit JSON need not, so
    # fall back to Cherimoya's defaults. The model's own trimming turns the
    # fallback into an checkpoint validation rather than a silent guess.
    fit.setdefault("in_window", 2114)
    fit.setdefault("out_window", 1000)

    suffix = ".torch" if checkpoint == "best" else ".final.torch"
    model_path = run_dir / f"{name}{suffix}"
    if not model_path.is_file():
        raise FileNotFoundError(f"no {checkpoint} checkpoint at {model_path}")

    signals, group_sizes = normalize_signal_groups(fit["signals"])
    records = list(spec.get("tracks") or [])
    if len(records) != len(group_sizes):
        raise ValueError(
            f"{name}: the job spec lists {len(records)} track(s) but the fit "
            f"JSON's signals form {len(group_sizes)} signal group(s). A result "
            "is one row per group, named after a track, so the two must "
            "correspond one to one."
        )
    spec_signals = [str(Path(record["signal"]).resolve()) for record in records]
    if spec_signals != [str(Path(s).resolve()) for s in signals]:
        raise ValueError(
            f"{name}: the job spec's track signals are not the fit JSON's "
            f"signals in the same order.\n  spec: {spec_signals}\n  fit:  {signals}"
        )

    return Job(
        name=name,
        path=path,
        spec=spec,
        fit=fit,
        run_dir=run_dir,
        model_path=model_path,
        tracks=[str(record["track"]) for record in records],
        signals=[str(Path(s).resolve()) for s in signals],
        signal_groups=list(group_sizes),
        mode=str(spec.get("mode", "single-task")),
        track_records=records,
    )


def extraction_signature(job: Job, chroms: list[str], loci: list[str], out_window: int) -> tuple:
    """What must match across a collection for one extraction to serve it all.

    Parameters
    ----------
    job
        A loaded job, read for its fit JSON.
    chroms
        Chromosomes of the split being extracted.
    loci
        BED paths.
    out_window
        Signal width, which may be narrower than the model's own.

    Returns
    -------
    tuple
        Hashable signature; two jobs sharing it can share one extraction.
    """
    return (
        str(Path(job.fit["sequences"]).resolve()),
        tuple(loci),
        tuple(sorted(str(Path(p).resolve()) for p in (job.fit.get("exclusion_lists") or []))),
        int(job.fit["in_window"]),
        int(out_window),
        tuple(chroms),
    )


def project_blocks(job: Job) -> dict[str, list[dict]]:
    """The ``tracks`` and ``models`` blocks of a project configuration, for one run.

    A stage-1 run has no project configuration of its own, and writing one per
    model of a sweep is busywork. This reads the two blocks the configuration
    path needs off the job instead: one track per output, named as the spec
    names it and pointing at the bigWig the fit JSON records, and one model
    entry carrying the selected checkpoint, the fit JSON and the outputs in
    order. Merged over a project's settings, as
    ``config.settings | project_blocks(job)``, the run then loads through
    the project's us_responsive.inference.load_models like any configured model. Blocks
    that name outputs, such as ``scorers``, are the caller's to reconcile.

    Parameters
    ----------
    job
        A loaded job. A stranded output lists both of its bigWigs under one track.

    Returns
    -------
    dict
        ``tracks`` and ``models``, in the schema :mod:`regseqkit.config` reads.
    """
    tracks, offset = [], 0
    for name, size in zip(job.tracks, job.signal_groups):
        paths = job.signals[offset : offset + size]
        tracks.append({"name": name, "path": paths[0] if size == 1 else paths})
        offset += size
    return {
        "tracks": tracks,
        "models": [
            {
                "checkpoint": str(job.model_path),
                "fit": str(job.run_dir / f"{job.name}.fit.json"),
                "outputs": list(job.tracks),
            }
        ],
    }
