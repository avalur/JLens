"""Eiffel two-hop figures.

Fig 1 (eiffel_ranks): 2x2 small multiples, one per model. For each layer, the full-
vocab rank of "Paris" read by the logit lens vs the J-lens (lower = closer to the
top of the vocabulary). The J-lens surfaces Paris in the *middle* layers; the logit
lens only in the last few. "France" (the first hop) is shown dashed.

Fig 2 (eiffel_headstart): a dumbbell summarising, per model, the relative depth at
which each lens first ranks Paris in the top-100 -- the J-lens's head start.

Run:  python figures/plot_eiffel.py
"""

import json
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jlens_style import apply_theme  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(os.path.join(HERE, "data", "eiffel_ranks.json")))
MODELS = DATA["models"]
Y_TOP, Y_BOT = 0.8, 3e5           # inverted log range (0.8 at top = best)
TOPK = 100                        # "surfaced" threshold


def first_reaches(ranks, k):
    for i, r in enumerate(ranks):
        if r <= k:
            return i
    return None


def _r(vals):                     # rank -> plot coord (rank 0 -> 1 for log)
    return [v + 1 for v in vals]


def fig_ranks(dark):
    cc = apply_theme(dark)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), sharey=True)
    axes = axes.ravel()
    for ax, m in zip(axes, MODELS):
        L = m["n_layers"]
        x = m["layer"]
        # "surfaced" band (top-K) near the top of the inverted axis
        ax.axhspan(Y_TOP, TOPK, color=cc("good"), alpha=0.06, lw=0, zorder=0)
        ax.axhline(TOPK, color=cc("muted"), lw=0.8, ls=(0, (2, 3)), zorder=1)

        ax.plot(x, _r(m["logit_paris_rank"]), color=cc("logit"), lw=2.2,
                solid_capstyle="round", zorder=4)
        ax.plot(x, _r(m["jlens_france_rank"]), color=cc("jlens"), lw=1.3,
                ls=(0, (4, 2)), alpha=0.75, zorder=3)
        ax.plot(x, _r(m["jlens_paris_rank"]), color=cc("jlens"), lw=2.4,
                solid_capstyle="round", zorder=5)

        # crossover markers: first layer each lens reaches top-K
        jl_i = first_reaches(m["jlens_paris_rank"], TOPK)
        ll_i = first_reaches(m["logit_paris_rank"], TOPK)
        for i, role in ((jl_i, "jlens"), (ll_i, "logit")):
            if i is not None:
                ax.scatter([x[i]], [TOPK], s=42, color=cc(role), zorder=6,
                           edgecolor=cc("surface"), linewidth=1.2, clip_on=False)
        lead = (ll_i - jl_i) if (jl_i is not None and ll_i is not None) else None

        ax.set_yscale("log")
        ax.set_ylim(Y_BOT, Y_TOP)          # inverted: best (top of vocab) at the top
        ax.set_xlim(-0.6, L - 0.4)
        ax.set_yticks([1, 10, 100, 1000, 10000, 100000])
        ax.set_yticklabels(["1", "10", "100", "1k", "10k", "100k"])
        ax.grid(axis="y", zorder=0)
        ax.grid(axis="x", visible=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

        title = f"{m['name']}"
        sub = f"{L} layers"
        if lead is not None:
            sub += f"  ·  J-lens leads by {lead} layers"
        ax.set_title(title, fontsize=12.5, fontweight="bold", loc="left", pad=24)
        ax.annotate(sub, xy=(0, 1.0), xytext=(0, 7), xycoords="axes fraction",
                    textcoords="offset points", ha="left", va="bottom",
                    fontsize=9.5, color=cc("ink2"))

    for ax in (axes[0], axes[2]):
        ax.set_ylabel("vocabulary rank of “Paris”\n(log · top = surfaced)",
                      fontsize=10, color=cc("ink2"))
    for ax in (axes[2], axes[3]):
        ax.set_xlabel("layer", fontsize=10, color=cc("ink2"))

    handles = [
        Line2D([0], [0], color=cc("jlens"), lw=2.6, label="J-lens · Paris"),
        Line2D([0], [0], color=cc("logit"), lw=2.4, label="logit lens · Paris"),
        Line2D([0], [0], color=cc("jlens"), lw=1.4, ls=(0, (4, 2)),
               label="J-lens · France  (1st hop)"),
        Line2D([0], [0], marker="o", color=cc("muted"), lw=0, markersize=7,
               label=f"first reaches top-{TOPK}"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.905),
               fontsize=10, handlelength=1.8, columnspacing=1.8, labelcolor=cc("ink2"))

    fig.suptitle("The J-lens surfaces “Paris” in the middle layers; the logit lens only at the end",
                 fontsize=15, fontweight="bold", x=0.012, ha="left", y=0.985, color=cc("ink"))
    fig.text(0.012, 0.942,
             'Two-hop prompt: “…the capital of the country where the Eiffel Tower is located is the city of ___”  →  Paris',
             fontsize=10.5, color=cc("ink2"), ha="left")
    fig.text(0.012, 0.012,
             "Full-vocab rank (lower = nearer the top of the vocabulary).  J-lens = W_U · norm(J_ℓ · h_ℓ),  "
             "J_ℓ = corpus-averaged Jacobian ∂h_L/∂h_ℓ.   Source: scripts/eiffel_twohop.py",
             fontsize=8, color=cc("muted"), ha="left")

    fig.subplots_adjust(left=0.085, right=0.985, top=0.80, bottom=0.085,
                        wspace=0.09, hspace=0.44)
    return fig


def fig_headstart(dark):
    cc = apply_theme(dark)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    rows = list(reversed(MODELS))            # top model at top
    ys = range(len(rows))
    for y, m in zip(ys, rows):
        L = m["n_layers"]
        jl = first_reaches(m["jlens_paris_rank"], TOPK)
        ll = first_reaches(m["logit_paris_rank"], TOPK)
        jf, lf = jl / (L - 1), ll / (L - 1)
        ax.plot([jf, lf], [y, y], color=cc("axis"), lw=3, solid_capstyle="round", zorder=2)
        ax.scatter([jf], [y], s=150, color=cc("jlens"), zorder=4,
                   edgecolor=cc("surface"), linewidth=1.5)
        ax.scatter([lf], [y], s=150, color=cc("logit"), zorder=4,
                   edgecolor=cc("surface"), linewidth=1.5)
        ax.annotate(f"+{ll - jl} layers earlier", xy=((jf + lf) / 2, y),
                    xytext=(0, 11), textcoords="offset points", ha="center",
                    fontsize=9.5, color=cc("ink2"), fontweight="bold")

    ax.set_yticks(list(ys))
    ax.set_yticklabels([f"{m['name']}\n{m['n_layers']} layers" for m in rows],
                       fontsize=10.5, color=cc("ink"))
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["input", "¼", "½", "¾", "output"])
    ax.set_xlabel(f"relative depth where the lens first ranks “Paris” in the top-{TOPK}",
                  fontsize=10.5, color=cc("ink2"))
    ax.grid(axis="x", zorder=0)
    ax.grid(axis="y", visible=False)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=cc("jlens"),
               markersize=11, label="J-lens"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=cc("logit"),
               markersize=11, label="logit lens"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=10.5, labelcolor=cc("ink2"),
              handletextpad=0.3, borderaxespad=0.6)

    fig.suptitle(f"The J-lens gets there first — head start to top-{TOPK}",
                 fontsize=15, fontweight="bold", x=0.012, ha="left", y=0.98, color=cc("ink"))
    fig.text(0.012, 0.885, "Paris, two-hop Eiffel prompt · full vocabulary",
             fontsize=10.5, color=cc("ink2"), ha="left")
    fig.subplots_adjust(left=0.16, right=0.98, top=0.80, bottom=0.16)
    return fig


def save(fig, stem, dark):
    suffix = "dark" if dark else "light"
    for ext in ("png", "svg"):
        path = os.path.join(HERE, f"{stem}_{suffix}.{ext}")
        fig.savefig(path, dpi=150 if ext == "png" else None)
    print(f"  wrote {stem}_{suffix}.png / .svg")
    plt.close(fig)


if __name__ == "__main__":
    for dark in (False, True):
        save(fig_ranks(dark), "eiffel_ranks", dark)
        save(fig_headstart(dark), "eiffel_headstart", dark)
    print("done")
