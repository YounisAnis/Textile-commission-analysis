"""Shared chart styling, so every figure in the project looks like one system."""

import matplotlib as mpl
import matplotlib.pyplot as plt

# Validated categorical palette (light mode). Slots are assigned in fixed order,
# never cycled; the first three clear the all-pairs colour-vision floors.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GOOD, WARN, BAD = "#1baf7a", "#eda100", "#e34948"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#8a8985"


def use_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_MUTED,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": INK_SOFT,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelsize": 10,
        "text.color": INK,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "grid.color": "#e6e5e1",
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2,
        "lines.markersize": 8,
        "font.size": 10,
        "figure.dpi": 110,
    })


def grid(ax, axis="y"):
    ax.grid(axis=axis, alpha=1.0, zorder=0)
    ax.set_axisbelow(True)


def caption(ax, text):
    ax.text(0, -0.20, text, transform=ax.transAxes, fontsize=8.5,
            color=INK_MUTED, va="top", ha="left", wrap=True)


def money(x, _=None):
    if abs(x) >= 1_000_000:
        return f"${x/1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"${x/1_000:.0f}K"
    return f"${x:.0f}"
