"""France -> China broadcast figure (finding #2).

One J-vector swap (France -> China) on the mid-network band redirects many
downstream facts at once. For each fact we show two bars:
  before  = baseline probability of the European (home) answer  (muted)
  after   = probability of the China-consistent answer after patching at alpha=4 (blue)
France facts get redirected; the Germany control is the selectivity check --
it BREAKS on 1.5B (Beijing leaks in) and HOLDS on 7B (Beijing stays out).

Run:  python figures/plot_france_china.py
"""

import json
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch as PatchHandle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jlens_style import apply_theme  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(os.path.join(HERE, "data", "france_china.json")))
MODELS = DATA["models"]
ALPHA = DATA["alpha"]


def fig_broadcast(dark):
    cc = apply_theme(dark)
    before_c = cc("muted")         # neutral reference (mode-invariant grey, reads on both surfaces)
    after_c = cc("jlens")          # blue: China-consistent answer induced
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.9), sharex=True, sharey=True)
    H = 0.36                        # bar half-group height

    for ax, m in zip(axes, MODELS):
        probes = m["probes"]
        n = len(probes)
        for i, p in enumerate(probes):
            y = n - 1 - i           # first probe at top
            ctl = p["is_control"]
            if ctl:
                ax.axhspan(y - 0.5, y + 0.5, color=cc("muted"), alpha=0.10, lw=0, zorder=0)
            ax.barh(y + H / 2, p["p_home_base"], height=H, color=before_c,
                    zorder=3, label="_", alpha=0.9)
            ax.barh(y - H / 2, p["p_target_patched"], height=H, color=after_c, zorder=3)
            # direct label on the "after" bar: what answer + how strong
            v = p["p_target_patched"]
            ax.annotate(f"{p['target']}  {v:.2f}", xy=(v, y - H / 2),
                        xytext=(6, 0), textcoords="offset points", va="center",
                        ha="left", fontsize=8.5, color=cc("ink2"),
                        fontweight="bold" if not ctl else "bold")
            # faint label on the "before" bar
            ax.annotate(f"{p['home']} {p['p_home_base']:.2f}", xy=(p["p_home_base"], y + H / 2),
                        xytext=(6, 0), textcoords="offset points", va="center",
                        ha="left", fontsize=8, color=cc("muted"))

        # divider above the control row
        ax.axhline(0.5, color=cc("axis"), lw=1.0, ls=(0, (3, 3)), zorder=1)

        ax.set_yticks(range(n))
        ax.set_yticklabels([p["probe"] for p in reversed(probes)], fontsize=10.5,
                           color=cc("ink"))
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_xlim(0, 1.06)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(axis="x", zorder=0)
        ax.grid(axis="y", visible=False)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("probability of that answer", fontsize=10, color=cc("ink2"))

        holds = m["control"] == "holds"
        vc = cc("good") if holds else "#d03b3b"
        icon = "✓" if holds else "✗"
        ax.set_title(f"{m['name']}", fontsize=13, fontweight="bold", loc="left", pad=25,
                     color=cc("ink"))
        ax.annotate(f"{icon} control {m['control']}", xy=(0, 1.0), xytext=(0, 7),
                    xycoords="axes fraction", textcoords="offset points",
                    ha="left", va="bottom", fontsize=10.5, color=vc, fontweight="bold")
        ax.annotate(f"patch band L{m['patch_band']} · α={ALPHA}", xy=(1.0, 1.0),
                    xytext=(0, 8), xycoords="axes fraction", textcoords="offset points",
                    ha="right", va="bottom", fontsize=9, color=cc("muted"))

    handles = [
        PatchHandle(facecolor=before_c, label="before — European answer"),
        PatchHandle(facecolor=after_c, label=f"after swap (α={ALPHA}) — China-consistent answer"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.885),
               fontsize=10, labelcolor=cc("ink2"), handlelength=1.4, columnspacing=2.2)

    fig.suptitle("One France→China swap redirects the whole country — and selectivity arrives with scale",
                 fontsize=15, fontweight="bold", x=0.012, ha="left", y=0.985, color=cc("ink"))
    fig.text(0.012, 0.925,
             "Swap the France and China J-vectors on the mid-network band; read four downstream facts + a Germany control.",
             fontsize=10.5, color=cc("ink2"), ha="left")
    fig.text(0.012, 0.015,
             "The shaded row is the selectivity control (Germany’s capital). On 1.5B, “Beijing” leaks into it (0.06) — the patch "
             "is a sledgehammer; on 7B it stays out (0.01).   Source: scripts/france_china.py",
             fontsize=8, color=cc("muted"), ha="left")

    fig.subplots_adjust(left=0.13, right=0.985, top=0.72, bottom=0.115, wspace=0.36)
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
        save(fig_broadcast(dark), "france_china", dark)
    print("done")
