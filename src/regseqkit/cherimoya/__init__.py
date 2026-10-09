"""Cherimoya checkpoint loading, output wrappers, and inference helpers.

These helpers operate on ordinary PyTorch models and return descriptive
fields for the configured workflow. They do not depend on project configuration.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch
try:
    from cherimoya import Cherimoya
    from cherimoya.io import channel_permutation_from_groups, normalize_signal_groups
    from cherimoya.wrappers import ControlWrapper, LogCountWrapper
except ModuleNotFoundError as exc:
    if exc.name == "cherimoya":
        raise ImportError("Cherimoya helpers require the optional regseqkit[cherimoya] dependency") from exc
    raise

from tangermeme.predict import predict
from ..wrappers import log1p_to_counts


def safe_batch_size(in_window: int, n_filters: int, requested: int) -> int:
    """Clamp a forward batch to what Cherimoya's fused kernel can address.

    ``CheriBlock``'s no-grad inference megakernel indexes its ``(N, L, C)``
    buffers with 32-bit offsets, so a forward whose ``N * in_window *
    n_filters`` reaches ``2**31`` fails from inside the Triton kernel with
    ``CUDA error: an illegal memory access was encountered``, which names
    nothing useful. The boundary depends on ``in_window``, which is why this is
    computed rather than written down: a batch that is fine for a 2114 bp model
    would crash a wider one. The 0.9 factor leaves margin for the kernel's own
    padding.

    Measured on an H100: batch 7000 at in_window 2114 works and 8500 crashes;
    5000 at 3114 works and 6000 crashes.

    Parameters
    ----------
    in_window
        Model input width.
    n_filters
        Channel count of the fused block.
    requested
        Desired batch size.

    Returns
    -------
    int
        `requested`, or the largest addressable batch, whichever is smaller.
    """
    ceiling = int(0.9 * 2**31 / (in_window * n_filters))
    return max(1, min(requested, ceiling))


def expected_signal(
    logits: torch.Tensor, scalars: torch.Tensor, groups: Sequence[int]
) -> torch.Tensor:
    """Convert already predicted (possibly RC-averaged) heads to signal.

    Cherimoya's ``ExpectedCountsWrapper`` requires a model forward. Here the
    heads already exist, so apply its group normalization without another
    forward, clipping negative recovered counts to valid signal values.

    Parameters
    ----------
    logits
        ``(N, channels, out_window)`` profile logits.
    scalars
        ``(N, outputs)`` log counts, one per group.
    groups
        Channels per output, partitioning `logits`' channel axis.

    Returns
    -------
    torch.Tensor
        ``(N, channels, out_window)`` non-negative expected signal.
    """
    profiles, offset = [], 0
    for i, size in enumerate(groups):
        block = logits[:, offset : offset + size]
        probability = block.flatten(1).softmax(-1).reshape_as(block)
        total = log1p_to_counts(scalars[:, i])
        profiles.append(probability * total[:, None, None])
        offset += size
    return torch.cat(profiles, dim=1)


class _CherimoyaOutput(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, groups: Sequence[int], head: str) -> None:
        super().__init__()
        controlled = ControlWrapper(model)
        self.model = LogCountWrapper(controlled) if head == "scalar" else controlled
        self.n_filters = model.n_filters
        self.groups = groups
        self.head = head

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        limit = safe_batch_size(X.shape[-1], self.n_filters, len(X))
        if len(X) > limit:
            return torch.cat([self.forward(batch) for batch in X.split(limit)], dim=0)
        if self.head == "scalar":
            return self.model(X)
        logits, scalars = self.model(X)
        return expected_signal(logits, scalars, self.groups)


def load_checkpoint(
    checkpoint: str | Path,
    fit: str | Path,
    outputs: Sequence[str],
    device: str = "cpu",
) -> tuple[torch.nn.Module, dict]:
    """Load a checkpoint and return its model and descriptive fields.

    Parameters
    ----------
    checkpoint : str or Path
        Saved model weights.
    fit : str or Path
        Fit JSON describing input and output windows and training signals.
    outputs : Sequence[str]
        Output names in checkpoint output order.
    device : str, default "cpu"
        Device supplied by the caller.

    Returns
    -------
    model : torch.nn.Module
        Loaded Cherimoya model in evaluation mode.
    information : dict
        Output names, windows, groups and native scalar encoding.

    Raises
    ------
    ValueError
        If fit information disagrees with the checkpoint, output names do not
        match its groups, or the model requires observed control tracks.
    """
    settings = json.loads(Path(fit).read_text())
    model = Cherimoya.load(checkpoint, device=device, compile=False).float().eval()
    groups = tuple(model.signal_groups)
    in_window, out_window = int(settings["in_window"]), int(settings["out_window"])
    if in_window - out_window != 2 * model.trimming:
        raise ValueError("fit windows disagree with checkpoint trimming")
    if "signals" in settings and tuple(normalize_signal_groups(settings["signals"])[1]) != groups:
        raise ValueError("fit signal groups disagree with checkpoint")
    if len(outputs) != len(groups):
        raise ValueError("one output name is required per checkpoint signal group")
    if model.n_control_tracks:
        raise ValueError("control-trained checkpoints require observed controls")
    return model, dict(
        outputs=tuple(outputs), in_window=in_window, out_window=out_window,
        groups=groups, scalar_transform="log1p",
    )


def output_module(model: torch.nn.Module, head: str = "scalar") -> torch.nn.Module:
    """Select a differentiable scalar or expected-profile output module.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded Cherimoya model.
    head : str, default "scalar"
        "scalar" selects log1p counts; "profile" selects expected signal.

    Returns
    -------
    module : torch.nn.Module
        Prepared output module with the existing batch-size limit.
    """
    if head not in ("scalar", "profile"):
        raise ValueError(f"head must be scalar or profile, got {head!r}")
    return _CherimoyaOutput(model, tuple(model.signal_groups), head).eval()


def predict_outputs(
    model: torch.nn.Module,
    X: torch.Tensor,
    *,
    device: str = "cpu",
    batch_size: int = 128,
    dtype: str = "float32",
    rc_average: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict log1p counts and expected profiles.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded Cherimoya model.
    X : torch.Tensor, shape (N, 4, length)
        One-hot sequences in ACGT order.
    device : str, default "cpu"
        Device supplied by the caller.
    batch_size : int, default 128
        Requested inference batch size, clamped to the kernel's limit.
    dtype : str, default "float32"
        Autocast dtype passed to tangermeme.predict.
    rc_average : bool, default False
        Average logits and log counts across orientations before converting
        profiles to expected signal. Strand channels are permuted within groups.

    Returns
    -------
    scalars : torch.Tensor, shape (N, outputs)
        Log1p count predictions.
    profiles : torch.Tensor, shape (N, channels, positions)
        Expected signal profiles.
    """
    groups = tuple(model.signal_groups)
    batch_size = safe_batch_size(X.shape[-1], int(model.n_filters), batch_size)
    options = dict(batch_size=batch_size, dtype=dtype, device=device)
    logits, scalars = predict(ControlWrapper(model), X, **options)
    logits, scalars = logits.float(), scalars.float()
    if rc_average:
        rl, rs = predict(ControlWrapper(model), X.flip((-2, -1)), **options)
        perm = channel_permutation_from_groups(list(groups))
        logits = (logits + rl.float()[:, perm].flip(-1)) / 2
        scalars = (scalars + rs.float()) / 2
    return scalars, expected_signal(logits, scalars, groups)
