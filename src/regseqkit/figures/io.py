"""Figure export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def save(
    fig: Figure,
    outdir: str | Path,
    stem: str,
    format: Literal["png", "pdf", "both"] = "both",
) -> None:
    """Write the selected format(s) to ``outdir`` and close the figure.

    ``format`` accepts ``"png"``, ``"pdf"``, or ``"both"`` (the default).
    PNG output uses 200 dpi; PDF uses the figure's dpi.
    """
    if format not in ("png", "pdf", "both"):
        raise ValueError('format must be "png", "pdf", or "both"')
    extensions = ("png", "pdf") if format == "both" else (format,)
    for extension in extensions:
        path = Path(outdir) / f"{stem}.{extension}"
        fig.savefig(path, dpi=200 if extension == "png" else None)
        print(f"wrote {path}", flush=True)
    plt.close(fig)
