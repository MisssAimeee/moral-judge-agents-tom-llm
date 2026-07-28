#!/usr/bin/env python3
"""Human-only developmental ladder: the three child measures plus the adult anchor.

No model points. This is the reference figure that establishes what the human
developmental trajectory looks like before any model is placed against it.

The y axis is the intent contrast: normalised blame for ATTEMPTED harm minus
normalised blame for ACCIDENTAL harm. Positive means intent outweighs outcome.
"""
import csv
import os
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HREF = os.path.join(ROOT, "dataset", "human_reference")
OUTDIR = os.path.join(ROOT, "outputs", "updated_figures")

MEASURES = OrderedDict([
    ("digitized", dict(
        file="human_reference_digitized.csv",
        label="Naughty, presented-first (digitized) — pre-specified primary",
        color="#1f6f8b", marker="o", lw=2.4, ls="-")),
    ("punish", dict(
        file="human_reference_punish.csv",
        label="Punish, presented-first (digitized) — construct-matched secondary",
        color="#c45c26", marker="s", lw=2.0, ls="-")),
    ("text_reported", dict(
        file="human_reference.csv",
        label="Text-reported (naughty+punishable pooled) — superseded",
        color="#888888", marker="^", lw=1.6, ls="--")),
])

GROUPS = ["child_4_5", "child_6_7", "child_8plus"]
GROUP_LABEL = {"child_4_5": "ages 4–5", "child_6_7": "ages 6–7",
               "child_8plus": "age 8+", "adult": "adults"}


def contrasts(path):
    """attempted − accidental in normalised blame units, per group."""
    cells = {}
    for r in csv.DictReader(open(path)):
        v = (r.get("norm_blame") or "").strip()
        if v:
            cells[(r["group"], r["condition"])] = float(v)
    out = {}
    for g in GROUPS + ["adult"]:
        a, b = cells.get((g, "attempted")), cells.get((g, "accidental"))
        if a is not None and b is not None:
            out[g] = a - b
    return out


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    data = {k: contrasts(os.path.join(HREF, m["file"])) for k, m in MEASURES.items()}

    xs = list(range(len(GROUPS)))
    x_adult = len(GROUPS) + 0.4

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    ax.axhline(0, color="k", lw=1.0, zorder=1)
    ax.axhspan(-0.25, 0, color="#d9534f", alpha=0.06, zorder=0)

    # Stagger the value labels per measure: the Naughty and text-reported series cross
    # near ages 6-7, so a single offset rule puts two labels in the same spot.
    LABEL_DY = {"digitized": 12, "punish": -15, "text_reported": 13}
    LABEL_DX = {"digitized": 0, "punish": 0, "text_reported": 22}
    for key, m in MEASURES.items():
        ys = [data[key].get(g) for g in GROUPS]
        ax.plot(xs, ys, color=m["color"], marker=m["marker"], ls=m["ls"],
                lw=m["lw"], ms=8, label=m["label"], zorder=3)
        for x, y in zip(xs, ys):
            if y is not None:
                ax.annotate(f"{y:+.2f}", (x, y), textcoords="offset points",
                            xytext=(LABEL_DX[key], LABEL_DY[key]), ha="center",
                            fontsize=8, color=m["color"])

    # All three child measures share one adult anchor (Young et al. 2007).
    adult = data["digitized"]["adult"]
    ax.plot([x_adult], [adult], marker="*", ms=20, color="#2c2c2c", zorder=4,
            label="Adult anchor (Young et al. 2007) — shared by all measures")
    ax.annotate(f"{adult:+.2f}", (x_adult, adult), textcoords="offset points",
                xytext=(0, 14), ha="center", fontsize=9, fontweight="bold")
    for key, m in MEASURES.items():
        ax.plot([xs[-1], x_adult], [data[key][GROUPS[-1]], adult],
                color=m["color"], ls=":", lw=1.0, alpha=0.5, zorder=2)

    ax.axvline((xs[-1] + x_adult) / 2, color="#bbbbbb", lw=0.8, ls="-")
    ax.text((xs[-1] + x_adult) / 2, 0.035, "children  |  adults",
            fontsize=8, color="#999", ha="center")

    ax.set_xticks(xs + [x_adult])
    ax.set_xticklabels([GROUP_LABEL[g] for g in GROUPS] + [GROUP_LABEL["adult"]])
    ax.set_ylabel("intent contrast  (attempted − accidental, normalised blame)")
    ax.set_xlabel("developmental group")
    ax.set_title("Human developmental trajectory of intent-based moral judgment\n"
                 "three published child measures against one shared adult anchor",
                 fontsize=12)
    ax.text(1.0, -0.215,
            "below 0 = outcome-driven (harm counts more than intent)",
            fontsize=8, color="#a33", va="center", ha="center")
    ax.set_ylim(-0.25, 0.80)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0.02, 0.99))
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()

    png = os.path.join(OUTDIR, "human_only_developmental_ladder.png")
    fig.savefig(png, dpi=170, bbox_inches="tight")
    plt.close(fig)

    csv_path = os.path.join(OUTDIR, "human_only_developmental_ladder.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["measure", "group", "intent_contrast_attempted_minus_accidental"])
        for key in MEASURES:
            for g in GROUPS + ["adult"]:
                if g in data[key]:
                    w.writerow([key, g, round(data[key][g], 4)])

    print("wrote", png)
    print("wrote", csv_path)
    for key in MEASURES:
        vals = "  ".join(f"{GROUP_LABEL[g]}={data[key][g]:+.3f}"
                         for g in GROUPS if g in data[key])
        print(f"  {key:14} {vals}   adult={data[key]['adult']:+.3f}")


if __name__ == "__main__":
    main()
