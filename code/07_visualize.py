#!/usr/bin/env python3
"""
07_visualize.py  --  Figures for the behavioral + statistical results.

Reads what 03/05/06 produced and writes publication-style PNGs to outputs/figures/:

  1. contrast_forest.png   intent-vs-outcome contrast per model with 95% bootstrap CI,
                           against the human developmental ladder (adult / child bands).
                           THE main inference figure.
  2. profiles.png          each model's blame across the 4 cells vs the adult human profile.
  3. prompt_invariance.png the contrast per prompt template per model (shows phrasing
                           sensitivity; a stable model is a tight vertical cluster).
  4. size_vs_contrast.png  contrast vs model size, base vs instruct (the confound view).
  5. weights_scatter.png   b_intent vs b_outcome per model (intent- vs outcome-weighting).
  6. pairwise_heatmap.png  pairwise contrast differences with significance markers.

Run after 06_stats.py. matplotlib only.
"""
import os, csv, glob, argparse, math
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CELLS = ["neutral", "accidental", "attempted", "intentional"]
COND_MAP = {"neutral": (0, 0), "accidental": (0, 1),
            "attempted": (1, 0), "intentional": (1, 1)}

def fnum(x):
    try: return float(x)
    except (TypeError, ValueError): return float("nan")

def short(tag):
    return tag.replace("Qwen_Qwen2.5-", "Qwen").replace("meta-llama_", "")

def load_contrast(stats_dir):
    p = os.path.join(stats_dir, "contrast_by_model.csv")
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []

def load_human_ladder(human_csv):
    grp = defaultdict(dict)
    if os.path.exists(human_csv):
        for r in csv.DictReader(open(human_csv)):
            if r.get("norm_blame", "").strip():
                grp[r["group"]][r["condition"]] = float(r["norm_blame"])
    ladder, profiles = {}, {}
    for g, p in grp.items():
        profiles[g] = p
        if "attempted" in p and "accidental" in p:
            ladder[g] = p["attempted"] - p["accidental"]
    return ladder, profiles

def pooled_profiles(behavior_dir):
    """model -> {condition: mean over templates & scenarios}."""
    out = {}
    for f in sorted(glob.glob(os.path.join(behavior_dir, "item_means_*.csv"))):
        tag = os.path.basename(f)[len("item_means_"):-4]
        acc = defaultdict(list)
        for r in csv.DictReader(open(f)):
            acc[r["condition"]].append(float(r["mean_norm_blame"]))
        out[tag] = {c: (sum(v)/len(v) if v else None) for c, v in acc.items()}
    return out

GROUP_COLORS = {"adult": "#1a9850", "child_8plus": "#91cf60",
                "child_6_7": "#fee08b", "child_4_5": "#fc8d59"}

def fig_forest(rows, ladder, out):
    rows = [r for r in rows if not math.isnan(fnum(r["contrast"]))]
    rows.sort(key=lambda r: fnum(r["contrast"]))
    if not rows: return
    fig, ax = plt.subplots(figsize=(9, 0.5*len(rows)+2.5))
    for g, c in sorted(ladder.items(), key=lambda x: x[1]):
        ax.axvline(c, ls="--", lw=1, color=GROUP_COLORS.get(g, "gray"), alpha=.9)
        ax.text(c, len(rows)-0.3, g.replace("child_", "age "), rotation=90,
                va="top", ha="right", fontsize=7, color=GROUP_COLORS.get(g, "gray"))
    ax.axvline(0, color="k", lw=0.8)
    for i, r in enumerate(rows):
        c, lo, hi = fnum(r["contrast"]), fnum(r["ci_lo"]), fnum(r["ci_hi"])
        col = "#d73027" if r.get("type") == "instruct" else "#4575b4"
        ax.errorbar(c, i, xerr=[[c-lo], [hi-c]], fmt="o", color=col, capsize=3, ms=6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([short(r["model"]) for r in rows], fontsize=8)
    ax.set_xlabel("intent-vs-outcome contrast  (attempted − accidental)\n"
                  "← outcome-weighted (child-like)      intent-weighted (adult-like) →")
    ax.set_title("Moral-judgment contrast per model (95% bootstrap CI)\n"
                 "blue = base · red = instruct · dashed = human reference")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)

def fig_profiles(profiles, human_profiles, out):
    if not profiles: return
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(CELLS))
    for tag, prof in profiles.items():
        ax.plot(x, [prof.get(c) for c in CELLS], "o-", lw=1, ms=4, alpha=.8, label=short(tag))
    for g in ("adult",):
        if g in human_profiles:
            ax.plot(x, [human_profiles[g].get(c) for c in CELLS], "s--", lw=2.5,
                    color="k", label=f"HUMAN {g}")
    ax.set_xticks(list(x)); ax.set_xticklabels(CELLS, rotation=15)
    ax.set_ylabel("normalized blame (0–1)")
    ax.set_title("Condition profile: model vs adult human")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)

def fig_invariance(stats_dir, ladder, out):
    p = os.path.join(stats_dir, "prompt_invariance_contrast.csv")
    if not os.path.exists(p): return
    rows = list(csv.DictReader(open(p)))
    if not rows: return
    tcols = [c for c in rows[0].keys() if c not in
             ("model", "sd", "range", "sign_flips")]
    fig, ax = plt.subplots(figsize=(9, 0.5*len(rows)+2.5))
    for g, c in ladder.items():
        ax.axvline(c, ls="--", lw=1, color=GROUP_COLORS.get(g, "gray"), alpha=.7)
    ax.axvline(0, color="k", lw=0.8)
    for i, r in enumerate(rows):
        vals = [fnum(r[t]) for t in tcols if r.get(t) not in ("", "NA", None)]
        for t in tcols:
            v = fnum(r.get(t))
            if not math.isnan(v): ax.scatter(v, i, s=30)
        if vals:
            ax.plot([min(vals), max(vals)], [i, i], color="gray", lw=1, zorder=0)
        flip = str(r.get("sign_flips")).lower() in ("true", "1", "yes")
        ax.text(0.99, i, "  SIGN FLIP" if flip else "", color="red", fontsize=7,
                va="center", transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([short(r["model"]) for r in rows], fontsize=8)
    ax.set_xlabel("contrast per prompt template")
    ax.set_title("Prompt-invariance: each dot = one template\n"
                 "(wide spread / sign flip = phrasing-sensitive, not a stable result)")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)

def fig_size(rows, out):
    pts = [(fnum(r["size_B"]), fnum(r["contrast"]), r.get("type")) for r in rows
           if not math.isnan(fnum(r["size_B"])) and not math.isnan(fnum(r["contrast"]))]
    if not pts: return
    fig, ax = plt.subplots(figsize=(7, 5))
    for typ, col in (("base", "#4575b4"), ("instruct", "#d73027")):
        sel = sorted([(s, c) for s, c, t in pts if t == typ])
        if sel:
            ax.plot([s for s, _ in sel], [c for _, c in sel], "o-", color=col, label=typ)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xscale("log"); ax.set_xlabel("model size (B params, log)")
    ax.set_ylabel("intent-vs-outcome contrast")
    ax.set_title("Size vs instruction-tuning (the base-vs-instruct confound)")
    ax.legend(); fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)

def fig_weights(rows, out):
    pts = [(fnum(r["b_outcome"]), fnum(r["b_intent"]), short(r["model"]), r.get("type"))
           for r in rows if not math.isnan(fnum(r.get("b_intent")))]
    if not pts: return
    fig, ax = plt.subplots(figsize=(6.5, 6))
    lim = max([abs(v) for p in pts for v in p[:2]] + [0.1]) * 1.15
    ax.plot([0, lim], [0, lim], ls=":", color="gray")
    for bo, bi, name, typ in pts:
        col = "#d73027" if typ == "instruct" else "#4575b4"
        ax.scatter(bo, bi, color=col, s=40)
        ax.annotate(name, (bo, bi), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("b_outcome (weight on bad outcome)")
    ax.set_ylabel("b_intent (weight on bad intent)")
    ax.set_title("Intent vs outcome regression weights\n(above dotted line = intent-weighted)")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)

def fig_pairwise(stats_dir, out):
    p = os.path.join(stats_dir, "pairwise_model_diffs.csv")
    if not os.path.exists(p): return
    rows = list(csv.DictReader(open(p)))
    if not rows: return
    models = sorted({r["model_a"] for r in rows} | {r["model_b"] for r in rows})
    idx = {m: i for i, m in enumerate(models)}
    n = len(models)
    import numpy as np
    M = np.full((n, n), np.nan)
    sig = {}
    for r in rows:
        i, j = idx[r["model_a"]], idx[r["model_b"]]
        d = fnum(r["contrast_diff"]); M[i, j] = d; M[j, i] = -d
        sig[(i, j)] = r.get("distinguishable") == "yes"
    fig, ax = plt.subplots(figsize=(1.1*n+2, 1.1*n+2))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-np.nanmax(np.abs(M)), vmax=np.nanmax(np.abs(M)))
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([short(m) for m in models], rotation=90, fontsize=7)
    ax.set_yticklabels([short(m) for m in models], fontsize=7)
    for (i, j), s in sig.items():
        if s:
            ax.text(j, i, "*", ha="center", va="center", color="k", fontsize=12)
            ax.text(i, j, "*", ha="center", va="center", color="k", fontsize=12)
    fig.colorbar(im, ax=ax, label="contrast difference (row − col)")
    ax.set_title("Pairwise model differences\n(* = 95% CI excludes 0)")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)

def main():
    here = os.path.dirname(__file__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavior", default=os.path.join(here, "..", "outputs", "behavior"))
    ap.add_argument("--stats", default=os.path.join(here, "..", "outputs", "stats"))
    ap.add_argument("--human", default=os.path.join(here, "..", "dataset",
                                                    "human_reference", "human_reference.csv"))
    ap.add_argument("--out", default=os.path.join(here, "..", "outputs", "figures"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = load_contrast(a.stats)
    ladder, human_profiles = load_human_ladder(a.human)
    profiles = pooled_profiles(a.behavior)
    jobs = [
        ("contrast_forest.png", lambda p: fig_forest(rows, ladder, p)),
        ("profiles.png",        lambda p: fig_profiles(profiles, human_profiles, p)),
        ("prompt_invariance.png", lambda p: fig_invariance(a.stats, ladder, p)),
        ("size_vs_contrast.png", lambda p: fig_size(rows, p)),
        ("weights_scatter.png", lambda p: fig_weights(rows, p)),
        ("pairwise_heatmap.png", lambda p: fig_pairwise(a.stats, p)),
    ]
    for name, fn in jobs:
        try:
            fn(os.path.join(a.out, name)); print("wrote", name)
        except Exception as e:
            print("skip", name, "->", e)
    print("figures in", a.out)

if __name__ == "__main__":
    main()
