"""Desires / introspection figure (finding #5).

Instruct models, forced one-word answers to introspective questions. For each
question we contrast what the model SAYS out loud (greedy decode) with the top
concept active INSIDE at the answer position (J-lens, read at the deep workspace
layer). The spoken word and the internal concept often differ -- but not always,
which the figure shows honestly with a per-row marker:
    != diverges     ~= aligns     ~ inside is diffuse
Strictly anecdotal: token distributions shaped by training data + persona.

Run:  python figures/plot_desires.py
"""

import json
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch as PatchHandle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jlens_style import apply_theme  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(os.path.join(HERE, "data", "desires.json")))
MODELS = DATA["models"]

CHIP_X = -0.37     # x of the "spoken" chip (gutter, left of the bar baseline)
GLYPH_X = -0.055   # x of the divergence glyph, just left of the bars
BH = 0.46


def fig_desires(dark):
    cc = apply_theme(dark)
    glyph = {"diverges": "≠", "aligns": "≈", "noisy": "~"}   # != ~= ~
    gcolor = {"diverges": "#d03b3b", "aligns": cc("good"), "noisy": cc("muted")}

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.4), sharex=True, sharey=True)

    for ax, m in zip(axes, MODELS):
        rows = m["rows"]
        n = len(rows)
        for i, r in enumerate(rows):
            y = n - 1 - i
            w = r["weight"]
            ax.barh(y, w, height=BH, color=cc("jlens"), zorder=3)

            lbl = f"{r['concept']}{'*' if r.get('star') else ''}  {w:.2f}"
            if w >= 0.55:                                   # label inside long bars
                ax.annotate(lbl, xy=(w, y), xytext=(-7, 0), textcoords="offset points",
                            ha="right", va="center", fontsize=9.6,
                            color=cc("surface"), fontweight="bold")
            else:                                           # label outside short bars
                ax.annotate(lbl, xy=(w, y), xytext=(7, 0), textcoords="offset points",
                            ha="left", va="center", fontsize=9.6,
                            color=cc("ink2"), fontweight="bold")

            ax.annotate(r["spoken"], xy=(CHIP_X, y), ha="center", va="center",
                        fontsize=9.4, color=cc("logit"), fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.34", fc=cc("surface"),
                                  ec=cc("logit"), lw=1.1))

            ax.annotate(glyph[r["align"]], xy=(GLYPH_X, y), ha="center", va="center",
                        fontsize=15, color=gcolor[r["align"]], fontweight="bold")

        ax.set_xlim(-0.56, 1.16)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", ".25", ".5", ".75", "1"])
        ax.axvline(0, color=cc("axis"), lw=1.0, zorder=2)
        ax.grid(axis="x", zorder=0)
        ax.grid(axis="y", visible=False)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("weight of that concept inside  (J-lens)", fontsize=10, color=cc("ink2"))

        ax.set_yticks(range(n))
        if ax is axes[0]:
            ax.set_yticklabels([r["q"] for r in reversed(rows)], fontsize=11, color=cc("ink"))
        else:
            ax.set_yticklabels([])

        ax.set_title(m["name"], fontsize=13, fontweight="bold", loc="left", pad=42,
                     color=cc("ink"))
        # column headers (below the model title)
        ax.annotate("says out loud", xy=(CHIP_X, 1.0), xytext=(0, 20),
                    xycoords=("data", "axes fraction"), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9.5, color=cc("logit"), fontweight="bold")
        ax.annotate("active inside — J-space", xy=(0.52, 1.0), xytext=(0, 20),
                    xycoords=("data", "axes fraction"), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9.5, color=cc("jlens"), fontweight="bold")

    handles = [
        PatchHandle(facecolor="none", edgecolor="none",
                    label="≠  diverges     ≈  aligns     ~  diffuse"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.9),
               fontsize=9.5, labelcolor=cc("ink2"), handlelength=0, handletextpad=0)

    fig.suptitle("What the model says out loud vs what's active inside (J-space)",
                 fontsize=15, fontweight="bold", x=0.012, ha="left", y=0.985, color=cc("ink"))
    fig.text(0.012, 0.93,
             "Instruct models, forced one-word answer. Spoken = greedy decode; inside = J-lens top concept at the "
             "deep workspace read (d≈0.8–0.9) — sometimes they match, sometimes not.",
             fontsize=10.5, color=cc("ink2"), ha="left")
    fig.text(0.012, 0.014,
             "Anecdotal — token distributions shaped by training data + persona, not “real desires”.  Weight sums "
             "case/script variants of one concept.  * surfaced mainly via its Chinese token (the workspace is "
             "cross-lingual).   Source: scripts/desires.py",
             fontsize=8, color=cc("muted"), ha="left")

    fig.subplots_adjust(left=0.135, right=0.99, top=0.70, bottom=0.12, wspace=0.28)
    return fig


def save(fig, stem, dark):
    suffix = "dark" if dark else "light"
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(HERE, f"{stem}_{suffix}.{ext}"),
                    dpi=150 if ext == "png" else None)
    print(f"  wrote {stem}_{suffix}.png / .svg")
    plt.close(fig)


if __name__ == "__main__":
    for dark in (False, True):
        save(fig_desires(dark), "desires", dark)
    print("done")
