#!/usr/bin/env python3
"""Plot gap-over-surface × pooling × factor for 8 open models (dissociation figure)."""
import os, csv, glob
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROBE = os.path.join(ROOT, "outputs", "probe")
OUT = os.path.join(PROBE, "gap_over_surface_dissociation.png")

POOLS = ["belief_last", "action_last", "mean", "last"]
POOL_LABEL = {
    "belief_last": "belief_last\n(harm not stated)",
    "action_last": "action_last",
    "mean": "mean\n(whole story)",
    "last": "last\n(whole story)",
}

def peaks():
    surf = list(csv.DictReader(open(os.path.join(PROBE, "surface_baseline.csv"))))
    tfidf = {r["target"]: float(r["cv_acc"]) for r in surf
             if r["subset"] == "all" and r["feature_set"] == "tfidf_word_1_2"}
    best = defaultdict(lambda: (-1.0, None))  # (model,pool,tgt)->acc
    for p in glob.glob(os.path.join(PROBE, "*_probe*.csv")):
        base = os.path.basename(p)
        if "surface" in base or "within" in base or "perm" in base or "layer0" in base:
            continue
        if base.endswith("_probe.csv"):
            model, pooling = base[:-len("_probe.csv")], "last"
        elif "_probe_" in base:
            model, pooling = base.split("_probe_", 1)
            pooling = pooling.replace(".csv", "")
        else:
            continue
        for r in csv.DictReader(open(p)):
            if "cv_acc" not in r:
                continue
            k = (model, pooling, r["target"])
            a = float(r["cv_acc"])
            if a > best[k][0]:
                best[k] = (a, r.get("layer"))
    models = sorted({m for m, _, _ in best})
    return tfidf, best, models

def main():
    tfidf, best, models = peaks()
    # within-model counts
    print("within-model intent_gap > outcome_gap:")
    for pool in POOLS:
        n = hold = 0
        for m in models:
            ai = best.get((m, pool, "intent"), (-1,))[0]
            ao = best.get((m, pool, "outcome"), (-1,))[0]
            if ai < 0 or ao < 0:
                continue
            n += 1
            if (ai - tfidf["intent"]) > (ao - tfidf["outcome"]):
                hold += 1
        print(f"  {pool:14} {hold}/{n}")

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(POOLS))
    width = 0.35
    # per-model thin lines + mean bar
    intent_gaps = []
    outcome_gaps = []
    for pool in POOLS:
        ig, og = [], []
        for m in models:
            ai = best.get((m, pool, "intent"), (-1,))[0]
            ao = best.get((m, pool, "outcome"), (-1,))[0]
            if ai < 0 or ao < 0:
                continue
            ig.append(ai - tfidf["intent"])
            og.append(ao - tfidf["outcome"])
        intent_gaps.append(ig)
        outcome_gaps.append(og)

    # jittered model points
    rng = np.random.default_rng(0)
    for i, pool in enumerate(POOLS):
        for g, color, dx in ((intent_gaps[i], "#1f6f8b", -width/2),
                             (outcome_gaps[i], "#c45c26", +width/2)):
            xs = i + dx + rng.uniform(-0.04, 0.04, len(g))
            ax.scatter(xs, g, s=28, color=color, alpha=0.75, zorder=3, edgecolors="none")
        ax.bar(i - width/2, np.mean(intent_gaps[i]), width, color="#1f6f8b",
               alpha=0.35, label="intent gap" if i == 0 else None, zorder=1)
        ax.bar(i + width/2, np.mean(outcome_gaps[i]), width, color="#c45c26",
               alpha=0.35, label="outcome gap" if i == 0 else None, zorder=1)

    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([POOL_LABEL[p] for p in POOLS])
    ax.set_ylabel("gap over TF-IDF surface baseline (probe − TF-IDF)")
    ax.set_title("Representational dissociation: intent gap peaks before harm is stated;\n"
                 "outcome gap peaks at whole-story and collapses at pre-outcome positions")
    ax.legend(frameon=False, loc="upper right")
    # annotate within-model rates
    for i, pool in enumerate(POOLS):
        n = len(intent_gaps[i])
        hold = sum(1 for a, b in zip(intent_gaps[i], outcome_gaps[i]) if a > b)
        ax.text(i, ax.get_ylim()[1] if False else 0.42, f"{hold}/{n}",
                ha="center", fontsize=9, color="#333")
    ax.set_ylim(-0.05, 0.48)
    # fix annotation y after ylim
    ymax = 0.455
    for i, pool in enumerate(POOLS):
        hold = sum(1 for a, b in zip(intent_gaps[i], outcome_gaps[i]) if a > b)
        n = len(intent_gaps[i])
        ax.text(i, ymax, f"within-model\n{hold}/{n}", ha="center", va="top",
                fontsize=8, color="#444")

    fig.tight_layout()
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT)

if __name__ == "__main__":
    main()
