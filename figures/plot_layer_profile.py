"""Layer-profile / three-zones figure (finding #3).

Metric: how often each lens's top-1 token equals the MODEL's own final top-1
prediction, layer by layer, on held-out text. The expected shape is three zones:
  sensory   -- first third, agreement ~0 (nothing readable yet)
  workspace -- slow rise through the middle
  motor     -- sharp late convergence to 1.0
Two panels show the reversal with scale: on 1.5B the logit lens keeps pace (or
leads) mid-network; on 7B the J-lens pulls ahead through the middle and late.
Both lenses read 1.000 at the final layer -- a pipeline sanity check.

Run:  python figures/plot_layer_profile.py
"""

import json
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jlens_style import apply_theme  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(os.path.join(HERE, "data", "layer_profile.json")))
MODELS = DATA["models"]
ZONES = DATA["zones"]


def _annot(ax, cc, model_name, xs, jl, ll):
    """Story callout per panel: the mid-network reversal."""
    if model_name.endswith("7B"):
        # L19 (depth 0.70): J-lens 0.107 vs logit 0.027 -- the reversal
        i = 19
        ax.annotate("J-lens  0.11\nvs logit  0.03",
                    xy=(xs[i], jl[i]), xytext=(xs[i] - 0.04, jl[i] + 0.30),
                    ha="right", va="bottom", fontsize=9, color=cc("jlens"),
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=cc("jlens"), lw=1.0,
                                    connectionstyle="arc3,rad=-0.2"))
    else:
        # mid-network: logit lens keeps pace with / edges the J-lens
        i = 18
        ax.annotate("logit lens\nkeeps pace",
                    xy=(xs[i], ll[i]), xytext=(xs[i] - 0.02, ll[i] + 0.26),
                    ha="right", va="bottom", fontsize=9, color=cc("logit"),
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=cc("logit"), lw=1.0,
                                    connectionstyle="arc3,rad=0.2"))


def fig_zones(dark):
    cc = apply_theme(dark)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.2), sharey=True)

    subtitles = {
        "Qwen2.5-1.5B": "textbook zones · logit lens keeps pace mid-network",
        "Qwen2.5-7B":   "the reversal · J-lens leads through the middle",
    }

    for ax, m in zip(axes, MODELS):
        xs, ll, jl = m["depth"], m["logit"], m["jlens"]

        # three zones as increasingly-inked neutral washes (reading gets more decided ->)
        for z, a in zip(ZONES, (0.0, 0.05, 0.10)):
            if a:
                ax.axvspan(z["lo"], z["hi"], color=cc("ink"), alpha=a, lw=0, zorder=0)
        for b in (ZONES[0]["hi"], ZONES[1]["hi"]):
            ax.axvline(b, color=cc("axis"), lw=0.9, ls=(0, (2, 3)), zorder=1)

        ax.plot(xs, ll, color=cc("logit"), lw=2.2, marker="o", ms=3.0,
                solid_capstyle="round", zorder=4)
        ax.plot(xs, jl, color=cc("jlens"), lw=2.4, marker="o", ms=3.2,
                solid_capstyle="round", zorder=5)

        _annot(ax, cc, m["name"], xs, jl, ll)

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.03, 1.06)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["input", "¼", "½", "¾", "output"])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(axis="y", zorder=0)
        ax.grid(axis="x", visible=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlabel("relative depth  (layer / final)", fontsize=10, color=cc("ink2"))

        ax.set_title(m["name"], fontsize=13, fontweight="bold", loc="left", pad=22,
                     color=cc("ink"))
        ax.annotate(subtitles[m["name"]], xy=(0, 1.0), xytext=(0, 6),
                    xycoords="axes fraction", textcoords="offset points",
                    ha="left", va="bottom", fontsize=9.5, color=cc("ink2"))

        # zone labels along the top (left panel only, to avoid clutter)
        if m["name"].endswith("1.5B"):
            for z, ha, xx in ((ZONES[0], "center", 0.17),
                              (ZONES[1], "center", 0.62),
                              (ZONES[2], "right", 1.0)):
                ax.annotate(z["name"], xy=(xx, 0.985), xycoords=("data", "axes fraction"),
                            ha=ha, va="top", fontsize=9, color=cc("muted"),
                            fontstyle="italic")

    axes[0].set_ylabel("top-1 agreement with the model", fontsize=10.5, color=cc("ink2"))

    handles = [
        Line2D([0], [0], color=cc("jlens"), lw=2.6, marker="o", ms=5, label="J-lens"),
        Line2D([0], [0], color=cc("logit"), lw=2.4, marker="o", ms=5, label="logit lens"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.9),
               fontsize=10, handlelength=1.8, columnspacing=1.8, labelcolor=cc("ink2"))

    fig.suptitle("Reading gets easier with depth — and the J-lens wins the middle as models scale",
                 fontsize=15, fontweight="bold", x=0.012, ha="left", y=0.985, color=cc("ink"))
    fig.text(0.012, 0.925,
             "How often each lens's top-1 token matches the model's own final prediction, layer by layer. "
             "Bands: the three zones (sensory → workspace → motor).",
             fontsize=10.5, color=cc("ink2"), ha="left")
    fig.text(0.012, 0.014,
             "Top-1 agreement on held-out text.  Both lenses read 1.000 at the final layer — a sanity check that the "
             "pipeline is honest.   Source: scripts/layer_profile.py",
             fontsize=8, color=cc("muted"), ha="left")

    fig.subplots_adjust(left=0.075, right=0.985, top=0.79, bottom=0.11, wspace=0.07)
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
        save(fig_zones(dark), "layer_profile", dark)
    print("done")
