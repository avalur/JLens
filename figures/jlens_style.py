"""Shared plotting theme for the J-lens figures.

Colours and chrome come from the data-viz reference palette (validated for CVD in
both modes). Categorical slots used here:
    J-lens  -> blue   (#2a78d6 light / #3987e5 dark)
    logit   -> orange (#eb6834 light / #d95926 dark)
Each figure is rendered in both a light and a dark theme; dark is a real selected
variant (own surface + re-stepped hues), not an auto-flip.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# role -> (light, dark)
PALETTE = {
    "surface":   ("#fcfcfb", "#1a1a19"),
    "page":      ("#f9f9f7", "#0d0d0d"),
    "ink":       ("#0b0b0b", "#ffffff"),
    "ink2":      ("#52514e", "#c3c2b7"),
    "muted":     ("#898781", "#898781"),
    "grid":      ("#e1e0d9", "#2c2c2a"),
    "axis":      ("#c3c2b7", "#383835"),
    "jlens":     ("#2a78d6", "#3987e5"),   # categorical slot 1 (blue)
    "logit":     ("#eb6834", "#d95926"),   # categorical slot 2 (orange)
    "good":      ("#0ca30c", "#0ca30c"),
}

SANS = ["-apple-system", "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans", "sans-serif"]


def c(role, dark):
    return PALETTE[role][1 if dark else 0]


def apply_theme(dark: bool):
    """Set rcParams for one theme; returns a helper c(role) bound to this mode."""
    surface = c("surface", dark)
    page = c("page", dark)
    ink, ink2, muted = c("ink", dark), c("ink2", dark), c("muted", dark)
    axis = c("axis", dark)
    mpl.rcParams.update({
        "font.family": SANS,
        "font.size": 11,
        "figure.facecolor": page,
        "savefig.facecolor": page,
        "axes.facecolor": surface,
        "axes.edgecolor": axis,
        "axes.linewidth": 1.0,
        "axes.labelcolor": ink2,
        "axes.titlecolor": ink,
        "axes.grid": True,
        "grid.color": c("grid", dark),
        "grid.linewidth": 0.8,
        "xtick.color": muted,
        "ytick.color": muted,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "text.color": ink,
        "legend.frameon": False,
        "svg.fonttype": "none",
    })
    return lambda role: c(role, dark)
