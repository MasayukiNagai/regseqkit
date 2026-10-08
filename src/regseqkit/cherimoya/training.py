"""Build Cherimoya job specs. Execution stays in the stage's runner.

Every argument is already resolved: absolute paths, a chromosome split mapping,
and a parameter dict. Reading those out of a project's configuration file is
the project's job, so this module has no opinion about its format.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..io import safe_name


def build_jobs(
    output_dir: str | Path,
    *,
    fasta: str | Path,
    loci: str | Path,
    tracks: Sequence[Mapping[str, Any]],
    splits: Mapping[str, Sequence[str]],
    parameters: Mapping[str, Any],
    negatives: str | Path | None = None,
    exclusion_lists: Sequence[str | Path] = (),
    name: str = "multitask",
    mode: str = "multitask",
    seeds: Sequence[int] = (0,),
    source: str | Path | None = None,
) -> list[Path]:
    """Write one ``*.job.json`` per arm/seed/track group plus a ``jobs.txt``.

    Parameters
    ----------
    output_dir
        Collection directory. Specs land in ``configs/`` beside a ``jobs.txt``.
    fasta, loci
        Absolute reference and peak-BED paths.
    tracks
        ``{"track": name, "signal": absolute path}`` rows in output order. Extra
        keys are carried through to the spec, so a project's own metadata
        columns survive.
    splits
        ``train``, ``valid``, and ``test`` chromosome lists, which must not
        overlap.
    parameters
        Cherimoya fit parameters. ``in_window`` and ``out_window`` are required
        and must permit equal flanks.
    negatives
        Background BED. Omitting it sets ``negative_ratio`` to zero.
    exclusion_lists
        BED paths whose intervals disqualify an overlapping locus.
    name
        Collection name, used for multitask job names.
    mode
        ``"single-task"``, ``"multitask"``, or ``"both"``.
    seeds
        Random seeds; one job per seed per arm.
    source
        Configuration file to record in each spec, for provenance only.

    Returns
    -------
    list of Path
        The written spec paths. ``jobs.txt`` lists them relative to its own
        directory, so a collection can be moved.

    Raises
    ------
    ValueError
        If track names are empty or duplicated, the splits overlap, the windows
        are invalid, the mode is unknown, or two jobs would collide on a name.
    """
    names = [r["track"] for r in tracks]
    if not names or len(set(names)) != len(names):
        raise ValueError("track names must be nonempty and unique")
    chroms = [c for split in ("train", "valid", "test") for c in splits[split]]
    if len(chroms) != len(set(chroms)):
        raise ValueError("chromosome splits overlap or contain duplicates")

    params = dict(parameters)
    params.setdefault("early_stopping", None)
    for key in ("in_window", "out_window"):
        if key not in params or int(params[key]) <= 0:
            raise ValueError(f"training parameter {key} must be positive")
    if (
        params["in_window"] < params["out_window"]
        or (params["in_window"] - params["out_window"]) % 2
    ):
        raise ValueError("input/output windows must permit equal flanks")
    if not negatives:
        params["negative_ratio"] = 0.0

    modes = ("single-task", "multitask") if mode == "both" else (mode,)
    root = Path(output_dir).resolve()
    specs = []
    for arm in modes:
        if arm not in ("single-task", "multitask"):
            raise ValueError("mode must be single-task, multitask, or both")
        groups = [[r] for r in tracks] if arm == "single-task" else [list(tracks)]
        for seed in seeds:
            for group in groups:
                label = group[0]["track"] if arm == "single-task" else name
                job_name = safe_name(f"{label}.seed{seed}", "job name")
                specs.append(
                    dict(
                        name=job_name,
                        mode=arm,
                        seed=int(seed),
                        sequences=str(fasta),
                        loci=str(loci),
                        negatives=str(negatives) if negatives else None,
                        exclusion_lists=[str(p) for p in exclusion_lists],
                        training_chroms=splits["train"],
                        validation_chroms=splits["valid"],
                        test_chroms=splits["test"],
                        tracks=group,
                        signal_groups=[1] * len(group),
                        fit_parameters=params,
                        source_config=str(source) if source else None,
                    )
                )

    job_names = [spec["name"] for spec in specs]
    if len(job_names) != len(set(job_names)):
        raise ValueError("job names collide; change the collection name or track names")
    (root / "configs").mkdir(parents=True, exist_ok=True)
    paths = []
    for spec in specs:
        path = root / "configs" / f"{spec['name']}.job.json"
        path.write_text(json.dumps(spec, indent=2) + "\n")
        paths.append(path)
    (root / "jobs.txt").write_text("".join(f"configs/{p.name}\n" for p in paths))
    return paths
