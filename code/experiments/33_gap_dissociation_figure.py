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
PAIRED_CSV = os.path.join(PROBE, "gap_over_surface_within_model_paired.csv")


def sign_test(diffs):
    """Two-sided exact sign test on paired differences; zeros dropped."""
    from math import comb
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n == 0:
        return 0, 0, None
    k = sum(1 for d in nz if d > 0)
    tail = lambda a: sum(comb(n, i) for i in range(a + 1)) / 2 ** n
    p = min(1.0, 2 * min(tail(min(k, n - k)), 1.0))
    return k, n, p

POOLS = ["belief_last", "action_last", "mean", "last"]
POOL_LABEL = {
    "belief_last": "belief_last\n(end of belief clause)",
    "action_last": "action_last\n(end of action clause)",
    "mean": "mean\n(whole story)",
    "last": "last\n(whole story)",
}

# The pre-outcome reading of belief_last/action_last is NOT established. In YS2008
# (192 of 298 items) the sentence that fixes the true state of the world — and therefore
# the outcome — appears BEFORE the belief clause, so a cut at belief_last has already
# seen it. Only YS2009 clearly places the belief before that sentence. Until the
# per-source probe split (02_probe.py --source) is run and shows outcome decoding at
# belief_last differing between sources, this figure is a statement about clause
# POSITION, not about information availability.
CAPTION = ("Probe advantage over a TF-IDF surface baseline, by token position.\n"
           "Positions are clause boundaries; whether they precede the outcome-determining\n"
           "sentence differs by source (YS2008 vs YS2009) and is not yet established.")

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

def write_paired():
    """Per-model paired intent-gap vs outcome-gap, with an exact sign test per pooling.

    The by-pooling table reports the single best model per cell, which cannot support a
    within-model claim. This pairs the two gaps inside each model instead.
    """
    tfidf, best, models = peaks()
    rows, summary = [], []
    for pool in POOLS:
        diffs = []
        for m in models:
            ai = best.get((m, pool, "intent"), (-1,))[0]
            ao = best.get((m, pool, "outcome"), (-1,))[0]
            if ai < 0 or ao < 0:
                continue
            gi, go = ai - tfidf["intent"], ao - tfidf["outcome"]
            diffs.append(gi - go)
            rows.append([pool, m, round(gi, 4), round(go, 4), round(gi - go, 4),
                         "intent" if gi > go else "outcome"])
        k, n, p = sign_test(diffs)
        mean_d = sum(diffs) / len(diffs) if diffs else float("nan")
        summary.append([pool, n, k, round(mean_d, 4),
                        "NA" if p is None else round(p, 4)])

    with open(PAIRED_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pooling", "model", "intent_gap", "outcome_gap",
                    "intent_minus_outcome", "larger"])
        w.writerows(rows)
        w.writerow([])
        w.writerow(["pooling", "n_models", "n_intent_larger", "mean_difference",
                    "sign_test_p_two_sided"])
        w.writerows(summary)
    print("wrote", PAIRED_CSV)
    print("within-model paired intent_gap − outcome_gap (exact sign test):")
    for pool, n, k, md, p in summary:
        print(f"  {pool:14} intent larger in {k}/{n}  mean diff={md:+.4f}  p={p}")


def main():
    tfidf, best, models = peaks()
    write_paired()

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
    ax.set_title("Intent and outcome decodability over a surface baseline, by clause position\n"
                 "intent advantage is largest at clause boundaries; outcome advantage is "
                 "largest at whole-story positions", fontsize=11)
    ax.text(0.5, -0.30, CAPTION, transform=ax.transAxes, ha="center", va="top",
            fontsize=7.5, color="#666")
    # Headroom above the tallest point so the per-pooling annotations and the legend
    # each get their own band and cannot land on top of one another.
    top = max(max(g) for g in intent_gaps + outcome_gaps)
    ax.set_ylim(-0.06, top + 0.20)
    # Annotations occupy the top band; the legend is pinned below it so the two
    # cannot overlap regardless of how tall the bars come out.
    for i, pool in enumerate(POOLS):
        n = len(intent_gaps[i])
        hold = sum(1 for a, b in zip(intent_gaps[i], outcome_gaps[i]) if a > b)
        ax.text(i, top + 0.155, "within-model", ha="center", va="center",
                fontsize=8, color="#444")
        ax.text(i, top + 0.105, f"{hold}/{n}", ha="center", va="center",
                fontsize=9, color="#222", fontweight="bold")
    # Legend below the axes: the annotation band spans the full width, so there is no
    # in-plot corner it can occupy without colliding.
    ax.legend(frameon=False, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, -0.14), handlelength=1.6, columnspacing=2.0)

    fig.tight_layout()
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT)

if __name__ == "__main__":
    main()
