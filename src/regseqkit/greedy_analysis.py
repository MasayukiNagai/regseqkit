"""Stage tables and optional inference caches for greedy trajectory analysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tangermeme.predict import predict

from .mutagenesis import single_site_saturation_mutagenesis, ssm_attribution
from .io import add_extension, load_arrays, save_arrays, scalar_score_transform


def stage_sequences(arrays: dict) -> np.ndarray:
    """Return stage zero followed by accepted results; omit rejected duplicates."""
    return np.concatenate((arrays["sequence"][:1], arrays["result"][arrays["accepted"]]), axis=0)


def stage_table(arrays: dict, metadata: dict, extra_scores=None, targets=None) -> pd.DataFrame:
    """Build one row per accepted stage, with native losses and log2 activities.

    Extra scores must align with ``stage_sequences``. Native objective values
    remain unconverted, since matching/holding terms may be nonlinear.
    Activity columns are present when their scorers are available.
    """
    sequences = stage_sequences(arrays)
    losses = np.r_[arrays["baseline_loss"][0], arrays["result_loss"][arrays["accepted"]]]
    scores = np.concatenate((arrays["baseline_scores"][:1],
                             arrays["result_scores"][arrays["accepted"]]), axis=0)
    frame = pd.DataFrame(dict(
        stage=np.arange(len(sequences)), objective=metadata["objective"],
        template=metadata["template"], template_index=metadata["template_index"],
        loss=losses, loss_improvement=losses[0] - losses,
        n_edits=(sequences.argmax(1) != sequences[0].argmax(0)).sum(1),
    ))
    values = dict(zip(metadata["scorers"], scores.T))
    if extra_scores is not None:
        extra_scores = np.asarray(extra_scores)
        if extra_scores.shape != (len(sequences), len(targets)) or not np.isfinite(extra_scores).all():
            raise ValueError("extra scores must be finite and aligned with stages and targets")
        values.update(zip(targets, extra_scores.T))
    # This analysis uses the project's unstandardized log scorers. The notebook
    # validates their definitions before deriving the missing linear scorer.
    if "CONTROL" in values and "delta_US157" in values and "US157" not in values:
        values["US157"] = values["CONTROL"] + values["delta_US157"]
    for name, label in (("CONTROL", "control"), ("US157", "high"), ("delta_US157", "delta")):
        frame[f"{label}_log2"] = values[name] / np.log(2) if name in values else np.nan
    if "CONTROL" in values and "US157" in values:
        frame["delta_log2"] = frame.high_log2 - frame.control_log2
    for label in ("control", "high", "delta"):
        frame[f"{label}_gain_log2"] = frame[f"{label}_log2"] - frame[f"{label}_log2"].iloc[0]
    return frame


def evaluation_table(arrays: dict, metadata: dict) -> pd.DataFrame:
    """Summarize available edits before each round, including rejected rounds."""
    refs = arrays["sequence"].argmax(1)[:, arrays["positions"]]
    improvements = arrays["baseline_loss"][:, None, None] - arrays["candidate_losses"]
    alternatives = np.ones(improvements.shape, dtype=bool)
    alternatives[np.arange(len(refs))[:, None], np.arange(refs.shape[1])[None, :], refs] = False
    choices = improvements[alternatives].reshape(len(refs), -1)
    return pd.DataFrame(dict(
        round=arrays["iterations"], objective=metadata["objective"],
        template=metadata["template"], template_index=metadata["template_index"],
        accepted=arrays["accepted"], best_improvement=choices.max(1),
        improving_fraction=(choices > metadata["search_settings"]["tol"]).mean(1),
        tol=metadata["search_settings"]["tol"],
        position=arrays["edit_positions"], base=arrays["edit_bases"],
    ))


def select_stages(stages, n_stages: int) -> list[int]:
    """Resolve explicit stage numbers, including -1 for the final stage."""
    result = []
    for value in stages:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError("stages must be integer indices")
        value = int(value) + n_stages if value < 0 else int(value)
        if not 0 <= value < n_stages:
            raise IndexError(f"choose stages from 0 to {n_stages - 1}, or negative indices")
        if value not in result:
            result.append(value)
    if not result:
        raise ValueError("choose at least one stage")
    return result


def saved_stage_attributions(arrays, metadata, stage, targets, window):
    """Center a saved outgoing mutation matrix when it covers requested targets.

    Return None for stages without an outgoing evaluation, unavailable targets,
    or windows outside the saved editable positions. A high-US scorer can be
    reconstructed from CONTROL and delta_US157 only when their recorded weight
    definitions establish that linear relation.
    """
    before = np.r_[0, np.cumsum(arrays["accepted"])[:-1]]
    matches = np.flatnonzero(before == stage)
    positions = arrays["positions"]
    requested = np.arange(*window)
    if not len(matches) or not np.isin(requested, positions).all():
        return None
    r = int(matches[0])
    sites = np.searchsorted(positions, requested)
    predictions = dict(zip(metadata["scorers"], arrays["candidate_scores"][r].transpose(2, 0, 1)))
    baselines = dict(zip(metadata["scorers"], arrays["baseline_scores"][r]))
    definitions = {s["name"]: s for s in metadata["definitions"]["scorers"]}
    names = ("CONTROL", "US157", "delta_US157")
    if "US157" not in predictions and {"CONTROL", "delta_US157"}.issubset(predictions):
        if set(names).issubset(definitions):
            weights = {name: definitions[name]["weights"] for name in names}
            outputs = set().union(*(set(w) for w in weights.values()))
            compatible = all(np.isclose(weights["US157"].get(o, 0),
                                         weights["CONTROL"].get(o, 0) + weights["delta_US157"].get(o, 0))
                             for o in outputs)
            if (compatible and not metadata["definitions"].get("calibration", {}).get("path")
                    and len({scalar_score_transform(metadata, name) for name in names}) == 1):
                predictions["US157"] = predictions["CONTROL"] + predictions["delta_US157"]
                baselines["US157"] = baselines["CONTROL"] + baselines["delta_US157"]
    if not set(targets).issubset(predictions):
        return None
    raw = np.stack([predictions[name][sites].T for name in targets])
    baseline = np.asarray([baselines[name] for name in targets])
    values = raw - baseline[:, None, None]
    values -= values.mean(1, keepdims=True)
    return values, baseline


def _sequence_key(sequence):
    return hashlib.sha256(sequence.astype(np.uint8).tobytes()).hexdigest()


def _fingerprint(metadata, targets, window=None):
    files = []
    for entry in metadata["models"]:
        for field in ("checkpoint", "fit"):
            path = Path(entry[field])
            if not path.is_absolute():
                path = Path(metadata["config"]).parent / path
            stat = path.stat()
            files.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    definition = dict(model=metadata["model"], models=metadata["models"],
                      definitions=metadata["definitions"], files=files,
                      targets=list(targets), window=window, version=1)
    return hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()


def cached_stage_scores(arrays, metadata, cache_dir, targets, *, module=None,
                        batch_size=128, device="cpu"):
    """Load stage scores, or compute/save them only when a module is supplied.

    Return None when no cache and no module are supplied. Cached values are
    checked against sequence hashes, scorer definitions, and checkpoint file
    identity. Re-running reuses completed search caches.
    """
    sequences = stage_sequences(arrays)
    keys = np.asarray([_sequence_key(seq) for seq in sequences])
    fingerprint = _fingerprint(metadata, targets)
    stem = Path(cache_dir) / metadata["objective"] / str(metadata["template_index"])
    if add_extension(stem, ".npz").is_file() and add_extension(stem, ".json").is_file():
        saved, info = load_arrays(stem)
        if info["fingerprint"] != fingerprint or not np.array_equal(saved["sequence_keys"], keys):
            raise ValueError(f"stage score cache differs from this run: {stem}; use another cache directory")
        if saved["scores"].shape != (len(keys), len(targets)) or not np.isfinite(saved["scores"]).all():
            raise ValueError(f"invalid stage score cache: {stem}")
        return saved["scores"]
    if module is None:
        return None
    scores = predict(module, torch.from_numpy(sequences).float(),
                     batch_size=batch_size, device=device).float().numpy()
    if scores.shape != (len(keys), len(targets)) or not np.isfinite(scores).all():
        raise ValueError("stage score module must return finite scores in target order")
    save_arrays(stem, dict(scores=scores, sequence_keys=keys), dict(
        kind="greedy_stage_scores", fingerprint=fingerprint, scorers=list(targets),
        objective=metadata["objective"], template=metadata["template"],
    ))
    return scores


def cached_stage_attributions(sequence, metadata, cache_dir, targets, window, *,
                              module=None, batch_size=64, device="cpu"):
    """Load or explicitly compute centered ISM; identical stages share a cache."""
    start, end = window
    key = _sequence_key(sequence)
    fingerprint = _fingerprint(metadata, targets, window)
    # Settings get their own directory, so changing targets or window preserves
    # earlier caches without needing to delete or overwrite them.
    stem = Path(cache_dir) / fingerprint / key
    if add_extension(stem, ".npz").is_file() and add_extension(stem, ".json").is_file():
        saved, info = load_arrays(stem)
        if info["fingerprint"] != fingerprint or info["sequence_key"] != key:
            raise ValueError(f"incompatible stage attribution cache: {stem}")
        if (saved["attributions"].shape != (len(targets), 4, end - start)
                or saved["baseline"].shape != (len(targets),)
                or not np.isfinite(saved["attributions"]).all()
                or not np.isfinite(saved["baseline"]).all()):
            raise ValueError(f"invalid stage attribution shape: {stem}")
        return saved["attributions"], saved["baseline"]
    if module is None:
        return None
    baseline, mutants = single_site_saturation_mutagenesis(
        module, torch.from_numpy(sequence[None]).float(), positions=range(start, end),
        batch_size=batch_size, device=device,
    )
    values = ssm_attribution(baseline, mutants)
    saved = dict(attributions=values[0].numpy(), baseline=baseline[0].numpy())
    save_arrays(stem, saved, dict(kind="greedy_stage_attributions", fingerprint=fingerprint,
                                sequence_key=key, scorers=list(targets), start=start, end=end))
    return saved["attributions"], saved["baseline"]
