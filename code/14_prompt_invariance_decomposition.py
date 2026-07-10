#!/usr/bin/env python3
"""
14_prompt_invariance_decomposition.py  --  Roadmap #3 (analysis half): quantify how
stable each model's intent-vs-outcome contrast is across the 3 prompt wordings
already collected (human_verbatim, para_wrong7, punish7). ANALYSIS ONLY.

The "moral fragility" critique says LLM moral judgments are prompt artifacts.
We pre-empt it by decomposing, per model:
  * contrast per template,
  * mean / SD / range / coefficient of variation,
  * sign stability (do all wordings agree on intent- vs outcome-weighting?),
  * a robustness verdict.

Outputs:
  outputs/analysis/prompt_invariance_decomposition.csv
  outputs/analysis/prompt_invariance_decomposition.png
  console table
"""
import os, csv, math, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tom_common as tc

RANGE_ROBUST = 0.10   # contrast range below this (and sign-stable) == "robust"


def template_contrasts(cells):
    """{template: mean contrast (attempted-accidental)}."""
    out = {}
    for tmpl, scen in cells.items():
        diffs = [c["attempted"] - c["accidental"] for c in scen.values()
                 if "attempted" in c and "accidental" in c]
        if diffs:
            out[tmpl] = float(np.mean(diffs))
    return out


def verdict(vals):
    if len(vals) < 2:
        return "single-prompt"
    # degenerate: model returned no usable signal (e.g. all-identical ratings)
    if all(abs(v) < 1e-6 for v in vals):
        return "degenerate (no signal)"
    signs = {(1 if v > 0 else (-1 if v < 0 else 0)) for v in vals}
    nonzero = {s for s in signs if s != 0}
    flip = len(nonzero) > 1
    rng = max(vals) - min(vals)
    if flip:
        return "FRAGILE (sign flip)"
    if rng <= RANGE_ROBUST:
        return "robust"
    return "sign-stable but variable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(tc.ROOT, "outputs", "analysis"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    all_templates, rows = [], []
    for study, tag, path in tc.iter_item_means():
        tcs = template_contrasts(tc.load_cells(path))
        if not tcs:
            continue
        for t in tcs:
            if t not in all_templates:
                all_templates.append(t)
        vals = list(tcs.values())
        mean = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        rng = (max(vals) - min(vals)) if vals else 0.0
        cv = (sd / abs(mean)) if abs(mean) > 1e-6 else float("inf")
        n_agree = max(sum(1 for v in vals if v > 0), sum(1 for v in vals if v < 0))
        rows.append(dict(model=tc.pretty(tag), study=study, tcs=tcs, n=len(vals),
                         mean=mean, sd=sd, rng=rng, cv=cv,
                         frac_agree=n_agree / len(vals) if vals else 0.0,
                         verdict=verdict(vals)))

    order = {"robust": 0, "sign-stable but variable": 1,
             "single-prompt": 2, "FRAGILE (sign flip)": 3,
             "degenerate (no signal)": 4}
    rows.sort(key=lambda r: (order.get(r["verdict"], 9), r["rng"]))

    # ---- console ----
    print("\n=== PROMPT-INVARIANCE DECOMPOSITION (contrast across wordings) ===")
    print(f"{'model':28} {'mean':>7} {'SD':>6} {'range':>7} {'sign-agree':>11} {'verdict':>26}")
    for r in rows:
        print(f"{r['model'][:28]:28} {r['mean']:+7.3f} {r['sd']:6.3f} {r['rng']:7.3f} "
              f"{r['frac_agree']*100:9.0f}%  {r['verdict']:>26}")

    # ---- csv ----
    out_csv = os.path.join(a.out, "prompt_invariance_decomposition.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "study", "n_templates"] + all_templates +
                   ["contrast_mean", "contrast_sd", "contrast_range",
                    "coef_variation", "fraction_sign_agree", "verdict"])
        for r in rows:
            w.writerow([r["model"], r["study"], r["n"]] +
                       [round(r["tcs"][t], 4) if t in r["tcs"] else "" for t in all_templates] +
                       [round(r["mean"], 4), round(r["sd"], 4), round(r["rng"], 4),
                        ("inf" if math.isinf(r["cv"]) else round(r["cv"], 3)),
                        round(r["frac_agree"], 3), r["verdict"]])

    # ---- figure: per-template dots + mean, sorted by robustness ----
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(rows) + 2))
    ax.axvline(0, color="k", lw=0.9)
    vcolor = {"robust": "#1a9850", "sign-stable but variable": "#fdae61",
              "single-prompt": "#888888", "FRAGILE (sign flip)": "#d73027",
              "degenerate (no signal)": "#555555"}
    for i, r in enumerate(rows):
        col = vcolor.get(r["verdict"], "#888")
        vals = list(r["tcs"].values())
        ax.plot([min(vals), max(vals)], [i, i], color=col, lw=2, alpha=0.35, zorder=1)
        ax.scatter(vals, [i] * len(vals), s=34, color=col, alpha=0.8, zorder=2)
        ax.scatter(r["mean"], i, marker="D", s=52, color=col, edgecolors="k",
                   linewidths=0.7, zorder=3)
        if "FRAGILE" in r["verdict"]:
            ax.text(max(vals) + 0.01, i, "sign flip", color="#d73027", fontsize=7.5,
                    va="center", fontweight="bold")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['model'][:26]} ({'cloud' if r['study']=='cloud API' else 'local'})"
                        for r in rows], fontsize=8)
    ax.set_xlabel("intent-vs-outcome contrast per prompt wording (♦ = mean across the 3 prompts)")
    ax.set_title("Prompt-invariance decomposition\n"
                 "(tight cluster right of 0 = robust · spread across 0 = fragile/prompt-sensitive)",
                 fontsize=12, fontweight="bold")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", ls="", ms=8, color=c, label=v)
               for v, c in vcolor.items()]
    ax.legend(handles=handles, fontsize=8, loc="lower right", framealpha=0.95)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out_png = os.path.join(a.out, "prompt_invariance_decomposition.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nwrote {os.path.relpath(out_csv, tc.ROOT)}")
    print(f"wrote {os.path.relpath(out_png, tc.ROOT)}")


if __name__ == "__main__":
    main()
