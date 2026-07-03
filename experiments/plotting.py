"""Shared matplotlib styling (same validated categorical palette used across
this project's sibling reproductions: fixed slot order, CVD-safe)."""
from __future__ import annotations

import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

CATEGORICAL = {
    "blue": "#2a78d6",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "green": "#008300",
    "violet": "#4a3aa7",
    "red": "#e34948",
    "magenta": "#e87ba4",
    "orange": "#eb6834",
}

METHOD_COLOR = {
    "PoA2": CATEGORICAL["blue"],
    "Vanilla FL": CATEGORICAL["violet"],
    "Centralized": CATEGORICAL["magenta"],
    "Authority-only": CATEGORICAL["aqua"],
    "Association-only": CATEGORICAL["yellow"],
    "Krum": CATEGORICAL["green"],
    "Trimmed Mean": CATEGORICAL["orange"],
    "Median": CATEGORICAL["red"],
    "CNN": CATEGORICAL["orange"],
    "LSTM": CATEGORICAL["yellow"],
    "BiLSTM": CATEGORICAL["blue"],
}

BLUE_RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"]


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_PRIMARY,
        "text.color": INK_PRIMARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "sans-serif",
        "font.size": 10,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
    })


def method_color(name: str) -> str:
    return METHOD_COLOR.get(name, INK_SECONDARY)
