#!/usr/bin/env python3
"""
11_human_reference_figure.py  --  Human-only version of the contrast forest plot.

Draws the intent-vs-outcome contrast axis with ONLY the human developmental
reference ladder (adult, child 8+, child 6-7, child 4-5) — no model rows on the
Y axis. Use this as a clean reference/legend slide that shows where each human
age band sits on the same axis used by agent_contrast_forest.png.

Human values are the intent-vs-outcome contrast = blame(attempted) - blame(accidental),
computed from dataset/human_reference/human_reference.csv (Young et al. 2007 for
adults; Cushman et al. 2013 for children).

Run:
  python code/11_human_reference_figure.py
  python code/11_human_reference_figure.py --out outputs/agents/figures
"""
import os, csv, argparse
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Same palette as 09_agent_figures.py so this reads as the same axis.
HUMAN_COLORS = {"adult": "#1a9850", "child_8plus": "#66bd63",
                "child_6_7": "#fdae61", "child_4_5": "#d73027"}
HUMAN_LABELS = {"adult": "adult", "child_8plus": "age 8+",
                "child_6_7": "age 6–7", "child_4_5": "age 4–5"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
})


def read_csv(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def load_human_ladder(human_csv):
    grp = defaultdict(dict)
    for r in read_csv(human_csv):
        if r.get("norm_blame", "").strip():
            grp[r["group"]][r["condition"]] = float(r["norm_blame"])
    ladder = {}
    for g, p in grp.items():
        if "attempted" in p and "accidental" in p:
            ladder[g] = p["attempted"] - p["accidental"]
    return ladder


def fig_human_only(ladder, out):
    if not ladder:
        print("skip -> no human ladder rows"); return
    groups = sorted(ladder.items(), key=lambda kv: kv[1])   # ascending contrast
    vals = [c for _, c in groups]

    fig, ax = plt.subplots(figsize=(10, 3.4))
    lo, hi = min(vals + [0.0]), max(vals + [0.0])
    pad = 0.12
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(0, 1)

    # Zero reference (pure outcome-vs-intent balance point).
    ax.axvline(0, color="k", lw=0.9)

    for g, c in groups:
        col = HUMAN_COLORS.get(g, "gray")
        ax.axvline(c, ls="--", lw=1.8, color=col, alpha=0.9)
        # marker on a central band
        ax.scatter([c], [0.5], s=90, color=col, zorder=3)
        # group name above, value below the line
        ax.text(c, 0.86, HUMAN_LABELS.get(g, g), rotation=0, ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=col)
        ax.text(c, 0.14, f"{c:+.2f}", ha="center", va="top",
                fontsize=10, color=col)

    ax.set_yticks([])
    ax.set_xlabel("intent-vs-outcome contrast   (blame: attempted − accidental)\n"
                  "← outcome-driven (child-like)          intent-driven (adult-like) →")
    ax.set_title("Human developmental reference ladder\n"
                 "(intent-vs-outcome contrast by age band; no models shown)")
    fig.tight_layout()
    fig.savefig(out); plt.close(fig)
    print("wrote", out)


def main():
    here = os.path.dirname(__file__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", default=os.path.join(here, "..", "dataset",
                                                    "human_reference", "human_reference.csv"))
    ap.add_argument("--out", default=os.path.join(here, "..", "outputs", "agents", "figures"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ladder = load_human_ladder(a.human)
    fig_human_only(ladder, os.path.join(a.out, "agent_contrast_forest_human_only.png"))


if __name__ == "__main__":
    main()
