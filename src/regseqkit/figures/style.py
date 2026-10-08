"""Shared neutral axis styling and density colors."""

from __future__ import annotations

import matplotlib
import numpy
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap


# Recessive axis furniture. Not a palette: these are the neutrals every figure
# uses regardless of which project's colours carry its data.
GRID = {"color": "#d8d8d4", "linewidth": 0.6}
INK = {"primary": "#0b0b0b", "secondary": "#52514e"}

# Figures display log2. Models and stored tables stay in natural log, the count
# head's own space, so a displayed log quantity is the stored one divided by
# this. Correlations, ranks and signs are unchanged by the rescale; slopes are
# not, which is why the conversion happens at the axis and nowhere upstream.
LN2 = float(numpy.log(2.0))


def style(ax: Axes) -> None:
    """Recessive axes and grid, so the marks carry the figure."""
    ax.set_axisbelow(True)
    ax.grid(True, **GRID)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID["color"])
    ax.tick_params(colors=INK["secondary"], labelsize=8)


# Blues with its lightest third removed. A lone point at density one would
# otherwise sit near-white on the white surface, and the outliers are what a
# scatter of counts exists to show.
DENSITY_CMAP = ListedColormap(matplotlib.colormaps["Blues"](numpy.linspace(0.3, 1.0, 256)))
