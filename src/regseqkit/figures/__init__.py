"""Figure primitives shared across stages and projects.

Panel composition stays in the stage; project palettes stay in the project.
Plot helpers return their figure and axes without saving or showing them.
``plot_logos`` creates its own batch grid; ``save`` exports and closes a figure.

Public names remain available here for existing callers, and can also be
imported from the individual plotting modules.
"""

from .io import save
from .layout import figure_with_plot_size, subplots_with_plot_size
from .logo import LOGOMAKER_STYLE, plot_logos, stack_attributions
from .profiles import plot_profiles, smooth
from .scatter import plot_contrast_pair, plot_count_pair, plot_density_scatter, square_frame
from .style import DENSITY_CMAP, GRID, INK, LN2, style

__all__ = [
    "DENSITY_CMAP",
    "GRID",
    "INK",
    "LN2",
    "LOGOMAKER_STYLE",
    "figure_with_plot_size",
    "plot_contrast_pair",
    "plot_count_pair",
    "plot_density_scatter",
    "plot_logos",
    "plot_profiles",
    "save",
    "smooth",
    "square_frame",
    "stack_attributions",
    "subplots_with_plot_size",
    "style",
]
