#!/usr/bin/env python3
"""
07_plots.py  --  Figures for the behavioral / statistical results.

Reads what 03_behavioral.py and 06_stats.py produced and draws the plots a mentor
(or a paper) actually wants to see:

  fig1_contrast_forest.png   THE headline: each model's intent-vs-outcome contrast
                             with its 95% bootstrap CI, against the human ladder
                             (adult / child age bands shown as reference lines).
  fig2_condition_profiles.png  the full 4-cell blame profile (neutral/accidental/
                             attempted/intentional) for every model vs adult + child.
  fig3_prompt_invariance.png the contrast computed under each prompt template per
                             model -> shows whether the result is prompt-stable.
  fig4_base_vs_instruct.png  contrast vs model size, base vs instruct as separate
                             lines -> separates the size effect from instruction tuning.
  fig5_intent_outcome_weights.png  b_intent vs b_outcome per model (regression
                             weights); the diagonal = equal weighting.

Run after 06_stats.py. Saves into outputs/figures/.
"""
import os, csv, glob, argparse, math
from collections import defaultdict

CELLS = ["neutral", "accidental", "attempted", "intentional"]
HUMAN_COLORS = {"adult": "#1a9850", "child_8plus": "#91cf60",
                "child_6_7": "#fee08b", "child_4_5": "#fc8d59"}

def read_csv(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []

def f(x):
    try:    return float(x)
    except (TypeError, ValueError): return float("nan")

def human_profiles(path):
    prof = defaultdict(dict)
    for r in read_csv(path):
        if r.get("norm_blame", "").strip():
            prof[r["group"]][r["condition"]] = float(r["norm_blame"])
    return prof

def human_contrasts(prof):
    out = {}
    for g, p in prof.items():
        if "attempted" in p and "accidental" in p:
            out[g] = p["attempted"] - p["accidental"]
    return out

def model_profiles(behavior_dir, template):
    """-> {model: {cond: mean_norm_blame}} for the given template."""
    out = {}
    for fp in sorted(glob.glob(os.path.join(behavior_dir, "item_means_*.csv"))):
        tag = os.path.basename(fp)[len("item_means_"):-4]
        by = defaultdict(list)
        for r in read_csv(fp):
            if r["template"] == template:
                by[r["condition"]].append(float(r["mean_norm_blame"]))
        out[tag] = {c: (sum(v)/len(v) if v else None) for c, v in by.items()}
    return out

def short(tag):
    return tag.replace("Qwen_Qwen2.5-", "Qwen-").replace("meta-llama_", "")

# ----------------------------------------------------------------- figures ----
def fig_forest(plt, rows, hcon, out):
    rows = [r for r in rows if not math.isnan(f(r["contrast"]))]
    rows.sort(key=lambda r: f(r["contrast"]))
    if not rows:
        return
    y = range(len(rows))
    xs = [f(r["contrast"]) for r in rows]
    lo = [f(r["contrast"]) - f(r["ci_lo"]) for r in rows]
    hi = [f(r["ci_hi"]) - f(r["contrast"]) for r in rows]
    colors = ["#2166ac" if r["type"] == "instruct" else "#b2182b" for r in rows]
    plt.figure(figsize=(9, 0.55*len(rows) + 2))
    for gi, (g, c) in enumerate(sorted(hcon.items(), key=lambda x: -x[1])):
        plt.axvline(c, ls="--", lw=1.2, color=HUMAN_COLORS.get(g, "gray"), alpha=.9)
        plt.text(c, len(rows)-0.3, g.replace("child_", "child ").replace("plus", "+"),
                 rotation=90, va="top", ha="right", fontsize=8,
                 color=HUMAN_COLORS.get(g, "gray"))
    plt.axvline(0, color="k", lw=0.8)
    plt.errorbar(xs, list(y), xerr=[lo, hi], fmt="o", capsize=3,
                 ecolor="gray", mfc="w", zorder=3)
    for yi, (xi, r) in enumerate(zip(xs, rows)):
        plt.plot(xi, yi, "o", color=colors[yi], zorder=4)
    plt.yticks(list(y), [short(r["model"]) for r in rows], fontsize=8)
    plt.xlabel("intent-vs-outcome contrast  (attempted − accidental)\n"
               "← outcome-weighted (child-like)      intent-weighted (adult-like) →")
    plt.title("Model moral-judgment contrast vs human developmental ladder\n"
              "(blue = instruct, red = base; bars = 95% bootstrap CI)")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()

def fig_profiles(plt, mprof, hprof, out):
    if not mprof:
        return
    plt.figure(figsize=(8, 5)); x = range(len(CELLS))
    for tag, p in mprof.items():
        ys = [p.get(c) for c in CELLS]
        if any(v is not None for v in ys):
            plt.plot(x, ys, "o-", lw=1.3, ms=4, label=short(tag), alpha=.85)
    for g, p in hprof.items():
        ys = [p.get(c) for c in CELLS]
        if any(v is not None for v in ys):
            plt.plot(x, ys, "s--", lw=2, label=g, color=HUMAN_COLORS.get(g))
    plt.xticks(list(x), CELLS, rotation=15)
    plt.ylabel("normalized blame  (0 = none, 1 = max)")
    plt.title("Condition profile: models vs humans (human_verbatim prompt)")
    plt.legend(fontsize=7, ncol=2); plt.tight_layout()
    plt.savefig(out, dpi=150); plt.close()

def fig_prompt_invariance(plt, inv_rows, out):
    if not inv_rows:
        return
    tmpls = [k for k in inv_rows[0].keys() if k not in
             ("model", "sd", "range", "sign_flips")]
    if not tmpls:
        return
    plt.figure(figsize=(9, 0.5*len(inv_rows) + 2))
    for yi, r in enumerate(inv_rows):
        vals = [(t, f(r[t])) for t in tmpls if r.get(t) not in ("", "NA", None)]
        vals = [(t, v) for t, v in vals if not math.isnan(v)]
        if not vals:
            continue
        xs = [v for _, v in vals]
        plt.plot(xs, [yi]*len(xs), "o", alpha=.7)
        plt.plot(sum(xs)/len(xs), yi, "D", color="k", ms=7)  # template mean
        if min(xs) < 0 < max(xs):
            plt.plot(max(xs)+.02, yi, "*", color="red", ms=10)  # sign flip flag
    plt.axvline(0, color="k", lw=0.8)
    plt.yticks(range(len(inv_rows)), [short(r["model"]) for r in inv_rows], fontsize=8)
    plt.xlabel("intent-vs-outcome contrast under each prompt template "
               "(◆ = mean, ★ = sign flips across prompts)")
    plt.title("Prompt-invariance: does the contrast hold across wordings?")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()

def fig_base_vs_instruct(plt, rows, hcon, out):
    pts = [r for r in rows if not math.isnan(f(r["size_B"]))
           and not math.isnan(f(r["contrast"]))]
    if not pts:
        return
    plt.figure(figsize=(8, 5))
    for mtype, mk, col in [("instruct", "o-", "#2166ac"), ("base", "s-", "#b2182b")]:
        sub = sorted([r for r in pts if r["type"] == mtype], key=lambda r: f(r["size_B"]))
        if sub:
            plt.errorbar([f(r["size_B"]) for r in sub], [f(r["contrast"]) for r in sub],
                         yerr=[[f(r["contrast"])-f(r["ci_lo"]) for r in sub],
                               [f(r["ci_hi"])-f(r["contrast"]) for r in sub]],
                         fmt=mk, color=col, capsize=3, label=mtype)
    for g, c in hcon.items():
        plt.axhline(c, ls="--", lw=1, color=HUMAN_COLORS.get(g, "gray"), alpha=.8)
        plt.text(plt.xlim()[1], c, " "+g, fontsize=7, va="center",
                 color=HUMAN_COLORS.get(g, "gray"))
    plt.xscale("log"); plt.axhline(0, color="k", lw=0.8)
    plt.xlabel("model size (B params, log scale)")
    plt.ylabel("intent-vs-outcome contrast")
    plt.title("Size vs instruction-tuning: which drives intent-weighting?")
    plt.legend(); plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()

def fig_weights(plt, rows, out):
    pts = [r for r in rows if not math.isnan(f(r["b_intent"]))
           and not math.isnan(f(r["b_outcome"]))]
    if not pts:
        return
    plt.figure(figsize=(6.5, 6))
    for r in pts:
        col = "#2166ac" if r["type"] == "instruct" else "#b2182b"
        plt.scatter(f(r["b_outcome"]), f(r["b_intent"]), color=col, zorder=3)
        plt.annotate(short(r["model"]), (f(r["b_outcome"]), f(r["b_intent"])),
                     fontsize=7, xytext=(4, 2), textcoords="offset points")
    lim = max(0.05, max(max(abs(f(r["b_intent"])), abs(f(r["b_outcome"]))) for r in pts))*1.1
    plt.plot([0, lim], [0, lim], "k--", lw=1, alpha=.6, label="equal weighting")
    plt.xlim(0, lim); plt.ylim(0, lim)
    plt.xlabel("b_outcome  (weight on bad outcome)")
    plt.ylabel("b_intent  (weight on bad intent)")
    plt.title("Regression weights: above diagonal = intent-weighted (adult-like)")
    plt.legend(); plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()

def main():
    here = os.path.dirname(__file__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavior", default=os.path.join(here, "..", "outputs", "behavior"))
    ap.add_argument("--stats", default=os.path.join(here, "..", "outputs", "stats"))
    ap.add_argument("--human", default=os.path.join(here, "..", "dataset",
                                                    "human_reference", "human_reference.csv"))
    ap.add_argument("--out", default=os.path.join(here, "..", "outputs", "figures"))
    ap.add_argument("--template", default="human_verbatim")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    rows = read_csv(os.path.join(a.stats, "contrast_by_model.csv"))
    inv_rows = read_csv(os.path.join(a.stats, "prompt_invariance_contrast.csv"))
    hprof = human_profiles(a.human)
    hcon = human_contrasts(hprof)
    mprof = model_profiles(a.behavior, a.template)

    if not rows:
        print("No outputs/stats/contrast_by_model.csv — run 06_stats.py first.")
    fig_forest(plt, rows, hcon, os.path.join(a.out, "fig1_contrast_forest.png"))
    fig_profiles(plt, mprof, hprof, os.path.join(a.out, "fig2_condition_profiles.png"))
    fig_prompt_invariance(plt, inv_rows, os.path.join(a.out, "fig3_prompt_invariance.png"))
    fig_base_vs_instruct(plt, rows, hcon, os.path.join(a.out, "fig4_base_vs_instruct.png"))
    fig_weights(plt, rows, os.path.join(a.out, "fig5_intent_outcome_weights.png"))
    print(f"Figures written to {a.out}/")
    for p in sorted(glob.glob(os.path.join(a.out, "*.png"))):
        print("  ", os.path.basename(p))

if __name__ == "__main__":
    main()
