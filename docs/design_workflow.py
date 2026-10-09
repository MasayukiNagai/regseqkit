"""Draw the design workflow figure used in docs/design.md.

Run with the project interpreter; the figure is written next to this file
as design_workflow.svg.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#1f2933"
MUTED = "#52606d"
LOOP = "#2e8b57"

# name: (width, fill, edge, title, role), drawn left to right.
SPECS = {
    "template": (3.0, "#f5f7fa", "#7b8794", "Template", "starting DNA,\neditable positions"),
    "model": (3.2, "#e3f0fb", "#2f80c2", "Model", "predicts from\none-hot DNA"),
    "scorers": (3.7, "#e3f0fb", "#2f80c2", "Scorers", "what to measure,\ne.g. activity, contrast"),
    "objective": (3.6, "#fdf1df", "#d9822b", "Objective", "maximize, minimize,\nor match a target"),
    "search": (3.9, "#e6f4ea", LOOP, "Search algorithm", "proposes edits"),
    "designed": (2.8, "#f5f7fa", "#7b8794", "Designed", "edited DNA"),
}
GAP = 2.4
GAPS: dict[str, float] = {}
Y = 2.3
HEIGHT = 1.5
Y_LOOP = 0.4

# The tensor handed to the next stage, as (label, shape).
FLOW = [
    ("template", "model", None, None),
    ("model", "scorers", "outputs", "(N, # outputs)"),
    ("scorers", "objective", "scores", "(N, # scores)"),
    ("objective", "search", "losses", "(N,)"),
    ("search", "designed", None, None),
]


def _layout(template_in_loop):
    """Place the boxes left to right, returning name to (x, width, ...rest)."""
    boxes = {}
    left = 0.1
    for name, (width, *rest) in SPECS.items():
        boxes[name] = (left + width / 2, width, *rest)
        gap = 3.4 if template_in_loop and name == "template" else GAPS.get(name, GAP)
        left += width + gap
    return boxes, left - GAP + 0.1


def _edge(boxes, name, side):
    """Midpoint of one box edge."""
    x, width = boxes[name][:2]
    return {
        "left": (x - width / 2, Y),
        "right": (x + width / 2, Y),
        "bottom": (x, Y - HEIGHT / 2),
    }[side]


def _arrow(ax, start, end, **arrow_kwargs):
    style = dict(arrowstyle="-|>", mutation_scale=16, lw=1.6, color=INK, shrinkA=0, shrinkB=0)
    style.update(arrow_kwargs)
    ax.add_patch(FancyArrowPatch(start, end, **style))


def plot_design_workflow(ax=None, *, template_in_loop=False, **text_kwargs):
    """Draw the model-guided design workflow.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None, default None
        Axes to draw on. None creates a new figure.
    template_in_loop : bool, default False
        True routes the template into the search loop instead of straight
        into the model, so the template and the returning candidates enter
        at the same point.
    **text_kwargs
        Passed to the box title text, e.g. fontsize or fontfamily.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    boxes, xmax = _layout(template_in_loop)
    if ax is None:
        fig, ax = plt.subplots(figsize=(18, 18 * 4.3 / xmax))
    else:
        fig = ax.figure
    title_kwargs = dict(fontsize=12, fontweight="bold", color=INK)
    title_kwargs.update(text_kwargs)

    for x, width, fill, edge, title, role in boxes.values():
        ax.add_patch(
            FancyBboxPatch(
                (x - width / 2, Y - HEIGHT / 2), width, HEIGHT,
                boxstyle="round,pad=0,rounding_size=0.15", fc=fill, ec=edge, lw=1.6,
            )
        )
        ax.text(x, Y + 0.38, title, ha="center", va="center", **title_kwargs)
        ax.text(x, Y - 0.2, role, ha="center", va="center", fontsize=9.5, color=MUTED, linespacing=1.35)

    # The template's own shape, kept clear of the line leaving its bottom edge.
    shape_y, shape_va = (
        (Y + HEIGHT / 2 + 0.15, "bottom")
        if template_in_loop
        else (Y - HEIGHT / 2 - 0.15, "top")
    )
    ax.text(
        boxes["template"][0], shape_y, "(1, 4, length)",
        ha="center", va=shape_va, fontsize=9, color=MUTED,
    )

    for source, target, name, shape in FLOW:
        if template_in_loop and source == "template":
            continue
        start, end = _edge(boxes, source, "right"), _edge(boxes, target, "left")
        _arrow(ax, start, end)
        if name:
            middle = (start[0] + end[0]) / 2
            ax.text(middle, Y + 0.12, name, ha="center", va="bottom", fontsize=9.5, color=INK)
            ax.text(middle, Y - 0.12, shape, ha="center", va="top", fontsize=8.5, color=MUTED)

    # ScoreModule bundles the model with the scorers it evaluates.
    left = _edge(boxes, "model", "left")[0]
    right = _edge(boxes, "scorers", "right")[0]
    bracket_y = Y + HEIGHT / 2 + 0.3
    ax.plot(
        [left, left, right, right],
        [bracket_y - 0.14, bracket_y, bracket_y, bracket_y - 0.14],
        color=boxes["model"][3], lw=1.4,
    )
    ax.text(
        (left + right) / 2, bracket_y + 0.08, "ScoreModule",
        ha="center", va="bottom", fontsize=11, color=boxes["model"][3], fontweight="bold",
    )

    # The search sends each round's candidates back through the model.
    if template_in_loop:
        join = (left - 1.2, Y_LOOP)
        _arrow(ax, _edge(boxes, "template", "bottom"), join, arrowstyle="-", color=MUTED, lw=1.6,
               connectionstyle="angle,angleA=-90,angleB=0,rad=14")
        _arrow(ax, _edge(boxes, "search", "bottom"), join, arrowstyle="-",
               connectionstyle="angle,angleA=-90,angleB=180,rad=14", color=LOOP, lw=1.6)
        _arrow(ax, join, (left, Y), connectionstyle="angle,angleA=90,angleB=180,rad=14",
               color=LOOP, lw=1.6)
        ax.text(
            left - 1.05, Y + 0.12, "candidate\nsequences", ha="right", va="bottom",
            fontsize=9.5, color=LOOP, linespacing=1.3,
        )
        ax.text(
            (join[0] + boxes["search"][0]) / 2, Y_LOOP - 0.18, "each round's candidates",
            ha="center", va="top", fontsize=10, color=LOOP,
        )
    else:
        _arrow(ax, _edge(boxes, "search", "bottom"), _edge(boxes, "model", "bottom"),
               connectionstyle="arc3,rad=-0.1", color=LOOP, lw=1.6)
        ax.text(
            (boxes["search"][0] + boxes["model"][0]) / 2, Y_LOOP - 0.18, "candidate sequences",
            ha="center", va="top", fontsize=10, color=LOOP,
        )

    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.5, 3.8)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


if __name__ == "__main__":
    fig, ax = plot_design_workflow()
    fig.savefig(
        Path(__file__).resolve().parent / "design_workflow.svg",
        bbox_inches="tight", facecolor="white",
    )
