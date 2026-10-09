"""Figure primitives shared across stages and projects.

Panel composition stays in the stage; project palettes stay in the project.
Plot helpers return their figure and axes without saving or showing them.
``plot_logos`` creates its own batch grid; ``save`` exports and closes a figure.

Primitives can also be imported from the individual plotting modules.
"""

from .io import save
from .layout import figure_with_plot_size, subplots_with_plot_size
from .logo import plot_logos, stack_attributions
from .profiles import plot_profiles, smooth
from .scatter import plot_density_scatter

__all__ = [
    "figure_with_plot_size",
    "plot_density_scatter",
    "plot_logos",
    "plot_profiles",
    "save",
    "smooth",
    "stack_attributions",
    "subplots_with_plot_size",
]
