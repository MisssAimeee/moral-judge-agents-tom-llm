#!/usr/bin/env python3
"""Recompute probe−surface gaps using span-matched TF-IDF baselines.

Clause-position probes (belief_last / action_last) see truncated input. The original
gap figure compared them against TF-IDF fit on the FULL story, which inflates the
surface baseline whenever the outcome-determining sentence falls after the cut.
This script:

  1. Reads span-matched rows from surface_baseline.csv (span=belief_last/action_last).
  2. Recomputes per-model peak gaps at each pooling against the matched baseline.
  3. Re-evaluates C2: YS2008 vs YS2009 outcome decoding at belief_last relative to
     the matched surface baseline (and reports absolute probe acc too).

If matched outcome TF-IDF is near chance for YS2009 while the probe reads 0.75–0.88,
the gap is large and the pre-outcome reading is revived — the model represents
outcome before the text states it, or the YS2009 annotation is wrong.
"""
import csv
import glob
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PROBE = os.path.join(ROOT, "outputs", "probe")
OUT_CSV = os.path.join(PROBE, "gap_over_surface_span_matched.csv")
OUT_C2 = os.path.join(ROOT, "outputs", "analysis", "C2_SOURCE_SPLIT_BELIEF_LAST.md")
OUT_FIG = os.path.join(PROBE, "gap_over_surface_dissociation_span_matched.png")

POOLS = ["belief_last", "action_last", "mean", "last"]
POOL_TO_SPAN = {
    "belief_last": "belief_last",
    "action_last": "action_last",
    "mean": "full",
    "last": "full",
}


def load_surface():
    """-> {(subset, span, target): cv_acc} for tfidf_word_1_2."""
    path = os.path.join(PROBE, "surface_baseline.csv")
    out = {}
    for r in csv.DictReader(open(path)):
        if r.get("feature_set") != "tfidf_word_1_2":
            continue
        span = r.get("span") or "full"
        out[(r["subset"], span, r["target"])] = float(r["cv_acc"])
    return out


def peak_probes(pattern="*_probe*.csv", require_src=None):
    """-> {(model, pooling, target): peak_acc}"""
    best = {}
    for p in glob.glob(os.path.join(PROBE, pattern)):
        base = os.path.basename(p)
        if "surface" in base or "within" in base or "perm" in base or "layer0" in base:
            continue
        if require_src and require_src not in base:
            continue
        if require_src is None and "_srcYS" in base:
            continue
        if base.endswith("_probe.csv"):
            model, pooling = base[:-len("_probe.csv")], "last"
        elif "_probe_" in base:
            model, rest = base.split("_probe_", 1)
            pooling = rest.replace(".csv", "")
            for src in ("_srcYS2008", "_srcYS2009", "_srcYS2011"):
                pooling = pooling.replace(src, "")
        else:
            continue
        for r in csv.DictReader(open(p)):
            if "cv_acc" not in r:
                continue
            k = (model, pooling, r["target"])
            a = float(r["cv_acc"])
            if a > best.get(k, (-1,))[0]:
                best[k] = (a, r.get("layer"))
    return {k: v[0] for k, v in best.items()}


def sign_test(diffs):
    from math import comb
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n == 0:
        return 0, 0, None
    k = sum(1 for d in nz if d > 0)
    tail = lambda a: sum(comb(n, i) for i in range(a + 1)) / 2 ** n
    p = min(1.0, 2 * min(tail(min(k, n - k)), 1.0))
    return k, n, p


def write_gaps(surf, peaks):
    rows = []
    summary = []
    for pool in POOLS:
        span = POOL_TO_SPAN[pool]
        tf_i = surf.get(("all", span, "intent"))
        tf_o = surf.get(("all", span, "outcome"))
        if tf_i is None or tf_o is None:
            # fall back to full if span row missing
            tf_i = surf.get(("all", "full", "intent"))
            tf_o = surf.get(("all", "full", "outcome"))
        diffs = []
        models = sorted({m for m, p, _ in peaks if p == pool})
        for m in models:
            ai = peaks.get((m, pool, "intent"))
            ao = peaks.get((m, pool, "outcome"))
            if ai is None or ao is None or tf_i is None or tf_o is None:
                continue
            gi, go = ai - tf_i, ao - tf_o
            diffs.append(gi - go)
            rows.append(dict(pooling=pool, span=span, model=m,
                             intent_probe=round(ai, 4), outcome_probe=round(ao, 4),
                             intent_tfidf=round(tf_i, 4), outcome_tfidf=round(tf_o, 4),
                             intent_gap=round(gi, 4), outcome_gap=round(go, 4),
                             intent_minus_outcome=round(gi - go, 4),
                             larger="intent" if gi > go else "outcome"))
        k, n, p = sign_test(diffs)
        summary.append(dict(pooling=pool, span=span, n_models=n, n_intent_larger=k,
                            mean_diff=round(float(np.mean(diffs)), 4) if diffs else "",
                            sign_test_p=("" if p is None else round(p, 4)),
                            tfidf_intent=tf_i, tfidf_outcome=tf_o))
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["pooling", "span", "model"])
        w.writeheader()
        w.writerows(rows)
        w.writerow({})
        sw = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        sw.writeheader()
        sw.writerows(summary)
    print("wrote", OUT_CSV)
    for s in summary:
        print(f"  {s['pooling']:14} span={s['span']:12} intent>outcome "
              f"{s['n_intent_larger']}/{s['n_models']}  "
              f"tfidf(i/o)={s['tfidf_intent']:.3f}/{s['tfidf_outcome']:.3f}")
    return rows, summary


def plot_gaps(rows, summary, surf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    x = np.arange(len(POOLS))
    width = 0.35
    intent_gaps, outcome_gaps = [], []
    for pool in POOLS:
        ig = [r["intent_gap"] for r in rows if r["pooling"] == pool]
        og = [r["outcome_gap"] for r in rows if r["pooling"] == pool]
        intent_gaps.append(ig)
        outcome_gaps.append(og)
    rng = np.random.default_rng(0)
    for i, pool in enumerate(POOLS):
        for g, color, dx in ((intent_gaps[i], "#1f6f8b", -width / 2),
                             (outcome_gaps[i], "#c45c26", +width / 2)):
            if not g:
                continue
            xs = i + dx + rng.uniform(-0.04, 0.04, len(g))
            ax.scatter(xs, g, s=28, color=color, alpha=0.75, zorder=3, edgecolors="none")
        if intent_gaps[i]:
            ax.bar(i - width / 2, np.mean(intent_gaps[i]), width, color="#1f6f8b",
                   alpha=0.35, label="intent gap" if i == 0 else None)
        if outcome_gaps[i]:
            ax.bar(i + width / 2, np.mean(outcome_gaps[i]), width, color="#c45c26",
                   alpha=0.35, label="outcome gap" if i == 0 else None)
    ax.axhline(0, color="k", lw=0.8)
    labels = []
    for pool in POOLS:
        span = POOL_TO_SPAN[pool]
        labels.append(f"{pool}\n(TF-IDF on {span})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("gap over span-matched TF-IDF (probe − TF-IDF)")
    ax.set_title("Gaps recomputed against span-matched surface baselines\n"
                 "belief_last / action_last TF-IDF uses text up to that clause end")
    # annotate within-model
    top = max((max(g) for g in intent_gaps + outcome_gaps if g), default=0.3)
    ax.set_ylim(-0.08, top + 0.18)
    for i, pool in enumerate(POOLS):
        s = next(s for s in summary if s["pooling"] == pool)
        ax.text(i, top + 0.12, "within-model", ha="center", fontsize=8, color="#444")
        ax.text(i, top + 0.07, f"{s['n_intent_larger']}/{s['n_models']}",
                ha="center", fontsize=9, fontweight="bold")
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT_FIG)


def rewrite_c2(surf, peaks_by_src):
    """Re-evaluate C2 against span-matched baselines."""
    lines = [
        "# C2 — Belief-last probes split by stimulus source (span-matched)",
        "",
        "Job `19025559` produced per-source probes. This revision subtracts the",
        "**span-matched** TF-IDF baseline (`text[:belief_end]`) rather than the",
        "full-story baseline. Absolute probe accuracies are unchanged; only the gap",
        "interpretation can move.",
        "",
        "## Span-matched TF-IDF at belief_last",
        "",
        "| source | target | TF-IDF (belief span) | chance |",
        "| --- | --- | ---: | ---: |",
    ]
    for src in ("YS2008", "YS2009", "all"):
        for tgt in ("intent", "outcome"):
            v = surf.get((src, "belief_last", tgt))
            if v is not None:
                lines.append(f"| {src} | {tgt} | {v:.3f} | ~0.50 |")

    lines += ["", "## Outcome decoding at belief_last", "",
              "| model | YS2008 probe | YS2009 probe | YS2008 gap | YS2009 gap |",
              "| --- | ---: | ---: | ---: | ---: |"]
    tf8 = surf.get(("YS2008", "belief_last", "outcome"))
    tf9 = surf.get(("YS2009", "belief_last", "outcome"))
    # peaks_by_src keys: (model, pooling, target, source)
    models = sorted({m for m, p, t, s in peaks_by_src if p == "belief_last" and t == "outcome"})
    for m in models:
        a8 = peaks_by_src.get((m, "belief_last", "outcome", "YS2008"))
        a9 = peaks_by_src.get((m, "belief_last", "outcome", "YS2009"))
        if a8 is None or a9 is None:
            continue
        g8 = a8 - tf8 if tf8 is not None else float("nan")
        g9 = a9 - tf9 if tf9 is not None else float("nan")
        lines.append(f"| {m} | {a8:.3f} | {a9:.3f} | {g8:+.3f} | {g9:+.3f} |")

    lines += ["", "## Intent decoding at belief_last", "",
              "| model | YS2008 probe | YS2009 probe | YS2008 gap | YS2009 gap |",
              "| --- | ---: | ---: | ---: | ---: |"]
    tf8i = surf.get(("YS2008", "belief_last", "intent"))
    tf9i = surf.get(("YS2009", "belief_last", "intent"))
    for m in models:
        a8 = peaks_by_src.get((m, "belief_last", "intent", "YS2008"))
        a9 = peaks_by_src.get((m, "belief_last", "intent", "YS2009"))
        if a8 is None or a9 is None:
            continue
        g8 = a8 - tf8i if tf8i is not None else float("nan")
        g9 = a9 - tf9i if tf9i is not None else float("nan")
        lines.append(f"| {m} | {a8:.3f} | {a9:.3f} | {g8:+.3f} | {g9:+.3f} |")

    # Verdict logic
    tf9_out = tf9 if tf9 is not None else float("nan")
    ys9_probes = [peaks_by_src[(m, "belief_last", "outcome", "YS2009")]
                  for m in models if (m, "belief_last", "outcome", "YS2009") in peaks_by_src]
    mean_ys9 = float(np.mean(ys9_probes)) if ys9_probes else float("nan")
    mean_gap9 = mean_ys9 - tf9_out if np.isfinite(tf9_out) else float("nan")

    lines += ["", "## Verdict", ""]
    if np.isfinite(tf9_out) and tf9_out < 0.60 and np.isfinite(mean_gap9) and mean_gap9 > 0.20:
        lines += [
            f"Span-matched outcome TF-IDF on YS2009 at belief_last is **{tf9_out:.3f}**",
            f"(near chance), while probes average **{mean_ys9:.3f}** (gap ≈ {mean_gap9:+.3f}).",
            "The absolute probe accuracy is therefore **not** explained by surface lexis",
            "available at the cut. Two readings remain open: (1) the model represents",
            "outcome before the text states it, or (2) the YS2009 clause annotation is",
            "wrong. The neutral caption on the gap figure is **withdrawn** pending",
            "annotation audit; the pre-outcome reading is again a live hypothesis for",
            "YS2009 items.",
        ]
        verdict = "REOPENED"
    elif np.isfinite(tf9_out) and tf9_out >= 0.65:
        lines += [
            f"Span-matched outcome TF-IDF on YS2009 at belief_last is still high",
            f"(**{tf9_out:.3f}**), so surface cues available *before* the outcome clause",
            "already predict the outcome label. The probe reading 0.75–0.88 does not",
            "establish pre-outcome representation. Neutral caption stays; C2 remains a",
            "reported result against the stronger claim.",
        ]
        verdict = "NEUTRAL_STANDS"
    else:
        lines += [
            f"Matched YS2009 outcome TF-IDF = {tf9_out}; mean probe = {mean_ys9:.3f};",
            f"mean gap = {mean_gap9:+.3f}. Intermediate — report the numbers; do not",
            "strengthen either the pre-outcome claim or the permanent-neutral claim",
            "without an annotation audit.",
        ]
        verdict = "INTERMEDIATE"

    lines += ["", f"**Status: {verdict}**", "",
              "Artifacts: `gap_over_surface_span_matched.csv`,",
              "`gap_over_surface_dissociation_span_matched.png`,",
              "`surface_baseline.csv` (rows with `span=belief_last|action_last`).", ""]
    open(OUT_C2, "w").write("\n".join(lines))
    print("wrote", OUT_C2, "verdict=", verdict)
    return verdict


def load_src_peaks():
    out = {}
    for src in ("YS2008", "YS2009"):
        for p in glob.glob(os.path.join(PROBE, f"*_probe_*_src{src}.csv")):
            base = os.path.basename(p)
            model, rest = base.split("_probe_", 1)
            pooling = rest.replace(f"_src{src}.csv", "")
            for r in csv.DictReader(open(p)):
                a = float(r["cv_acc"])
                k = (model, pooling, r["target"], src)
                if a > out.get(k, -1):
                    out[k] = a
    return out


def main():
    surf = load_surface()
    if ("all", "belief_last", "outcome") not in surf:
        raise SystemExit(
            "surface_baseline.csv has no span=belief_last rows — "
            "re-run 21_surface_baseline.py first")
    peaks = peak_probes()
    rows, summary = write_gaps(surf, peaks)
    plot_gaps(rows, summary, surf)
    src_peaks = load_src_peaks()
    rewrite_c2(surf, src_peaks)


if __name__ == "__main__":
    main()
