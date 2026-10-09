"""Load Cherimoya checkpoints and predict differentiable sequence outputs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch

try:
    from cherimoya import Cherimoya
    from cherimoya.io import channel_permutation_from_groups, normalize_signal_groups
    from cherimoya.wrappers import ControlWrapper
except ModuleNotFoundError as exc:
    if exc.name == "cherimoya":
        raise ImportError("Cherimoya helpers require regseqkit[cherimoya]") from exc
    raise

from ..inference import compute_signal, predict as batched_predict
from ..sequences import reverse_complement


def load_checkpoint(
    checkpoint: str | Path,
    fit: str | Path,
    outputs: Sequence[str],
    device: str = "cpu",
) -> tuple[Cherimoya, dict]:
    """Load a checkpoint and validate its accompanying fit metadata.

    Parameters
    ----------
    checkpoint : str or Path
        Saved model weights.
    fit : str or Path
        Fit JSON describing input/output windows and training signals.
    outputs : Sequence[str]
        Nonempty output names in checkpoint group order.
    device : str, default "cpu"
        Device passed to Cherimoya.load.

    Returns
    -------
    model : Cherimoya
        Float32 Cherimoya model in evaluation mode.
    information : dict
        Output names, windows, groups, and native scalar encoding.
    """
    settings = json.loads(Path(fit).read_text())
    model = Cherimoya.load(checkpoint, device=device, compile=False).float().eval()
    groups = tuple(model.signal_groups)
    in_window, out_window = int(settings["in_window"]), int(settings["out_window"])
    if in_window - out_window != 2 * model.trimming:
        raise ValueError("fit windows disagree with checkpoint trimming")
    if "signals" in settings and tuple(normalize_signal_groups(settings["signals"])[1]) != groups:
        raise ValueError("fit signal groups disagree with checkpoint")
    if len(outputs) != len(groups) or any(not name for name in outputs):
        raise ValueError("one nonempty output name is required per checkpoint signal group")
    return model, dict(
        outputs=tuple(outputs),
        in_window=in_window,
        out_window=out_window,
        groups=groups,
        scalar_transform="log1p",
    )


def predict_cherimoya(
    model: Cherimoya,
    X: torch.Tensor,
    *,
    device: str = "cpu",
    batch_size: int = 128,
    dtype: str = "float32",
    rc_average: bool = False,
    X_ctl: torch.Tensor | None = None,
    control_groups: Sequence[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict log1p counts and expected profiles through tangermeme.predict.

    Parameters
    ----------
    model : Cherimoya
        Cherimoya model.
    X : torch.Tensor, shape (N, 4, length)
        One-hot sequences in ACGT order.
    device : str, default "cpu"
        Inference device.
    batch_size : int, default 128
        Positive batch size, passed unchanged to tangermeme.predict.
    dtype : str, default "float32"
        Autocast dtype passed to tangermeme.predict.
    rc_average : bool, default False
        Predict both orientations, align reverse-complement profile channels,
        and average logits and log1p counts before computing signal.
    X_ctl : torch.Tensor or None, shape (N, control_channels, length), default None
        Auxiliary input-control coverage tracks, batched alongside sequences.
        When omitted, Cherimoya's ControlWrapper supplies zeros if the model
        needs controls. These inputs are separate from outputs named CONTROL.
    control_groups : Sequence[int] or None, default None
        Number of consecutive input-control channels in each RC group, using
        Cherimoya's terminology. None treats each channel as an independent
        unstranded track. [2] describes a stranded pair whose channels swap
        when positions are reversed. Used only with rc_average=True and
        observed X_ctl. Entries must cover all control channels.

    Returns
    -------
    scalars : torch.Tensor, shape (N, outputs)
        Log1p counts.
    profiles : torch.Tensor, shape (N, channels, positions)
        Expected signal profiles.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    wrapped = ControlWrapper(model)
    wrapped.training = model.training
    options = dict(batch_size=batch_size, dtype=dtype, device=device)
    args = (X_ctl,) if X_ctl is not None else None
    logits, scalars = batched_predict(wrapped, X, args=args, **options)
    logits, scalars = logits.float(), scalars.float()
    if rc_average:
        reverse_controls = None
        if X_ctl is not None:
            groups = control_groups if control_groups is not None else [1] * X_ctl.shape[1]
            permutation = channel_permutation_from_groups(list(groups)).to(X_ctl.device)
            reverse_controls = X_ctl[:, permutation].flip(-1)
        reverse_args = (reverse_controls,) if reverse_controls is not None else None
        reverse_logits, reverse_scalars = batched_predict(
            wrapped, reverse_complement(X), args=reverse_args, **options
        )
        permutation = channel_permutation_from_groups(model.signal_groups)
        logits = (logits + reverse_logits.float()[:, permutation].flip(-1)) / 2
        scalars = (scalars + reverse_scalars.float()) / 2
    return scalars, compute_signal(logits, scalars, model.signal_groups)
