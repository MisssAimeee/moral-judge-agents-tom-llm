#!/usr/bin/env python3
"""
13_scale_vs_performance.py  --  Ask 3 / roadmap finding #2: does model SCALE
predict where a model sits on the human developmental (intent-vs-outcome) axis?
ANALYSIS ONLY — plots existing ratings, computes a correlation, no inference.

Combines cloud agents (GPT/Claude/Gemini) with open-weight families
(Qwen, Llama, Mistral, …) for more datapoints.

Outputs:
  outputs/analysis/scale_vs_performance.png          (X=size, Y=contrast)
  outputs/analysis/contrast_vs_scale.png             (X=contrast, Y=size)  ← preferred
  outputs/analysis/scale_vs_performance.csv
"""
import os, csv, math, argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
from scipy import stats
import tom_common as tc

HUMAN_COLORS = {"adult": "#1a9850", "child_8plus": "#66bd63",
                "child_6_7": "#fdae61", "child_4_5": "#d73027"}
FAMILY_COLORS = {"Claude": "#cc785c", "Gemini": "#4285f4", "GPT": "#10a37f",
                 "Qwen": "#00909e", "Llama": "#a259ff", "Mistral": "#ff7000",
                 "OLMo": "#7d3c98", "Gemma": "#e8710a", "Phi": "#c2185b",
                 "other": "#888888"}

# Families to keep when --families is used (default: open + cloud agents)
DEFAULT_FAMILIES = {"Qwen", "Llama", "Mistral", "Claude", "Gemini", "GPT"}


def family_of(name):
    n = name.lower()
    if "gpt" in n or n.startswith("o1") or n.startswith("o3"):
        return "GPT"
    for k in ["claude", "gemini", "qwen", "llama", "mistral", "olmo", "gemma", "phi"]:
        if k in n:
            return k.capitalize() if k != "olmo" else "OLMo"
    return "other"


def contrast_of(pooled):
    diffs = [c["attempted"] - c["accidental"] for c in pooled.values()
             if "attempted" in c and "accidental" in c]
    return float(np.mean(diffs)) if diffs else float("nan")


def corr(sel):
    if len(sel) < 3:
        return None
    x = np.log10([p["size"] for p in sel])
    y = np.array([p["contrast"] for p in sel])
    sr, sp = stats.spearmanr(x, y)
    pr, pp = stats.pearsonr(x, y)
    # OLS fit: contrast ~ log10(size)
    slope, intercept, _, _, _ = stats.linregress(x, y)
    return dict(n=len(sel), sr=sr, sp=sp, pr=pr, pp=pp,
                slope=slope, intercept=intercept)


def draw_fit(ax, pts, orientation="size_x", color="#333", label_prefix=""):
    """Draw OLS fit of contrast ~ log10(size). orientation: 'size_x' or 'contrast_x'."""
    c = corr(pts)
    if not c:
        return None
    sizes = np.array([p["size"] for p in pts])
    log_lo, log_hi = np.log10(sizes.min()), np.log10(sizes.max())
    xs = np.logspace(log_lo, log_hi, 80)
    ys = c["slope"] * np.log10(xs) + c["intercept"]
    if orientation == "size_x":
        ax.plot(xs, ys, color=color, lw=2.0, alpha=0.75, zorder=2,
                label=f"{label_prefix}fit (Pearson r={c['pr']:+.2f})")
    else:
        ax.plot(ys, xs, color=color, lw=2.0, alpha=0.75, zorder=2,
                label=f"{label_prefix}fit (Pearson r={c['pr']:+.2f})")
    return c


def annotate_stats(ax, c_open, c_all, loc=(0.02, 0.02)):
    lines = []
    if c_open:
        lines.append(f"disclosed sizes (n={c_open['n']}): "
                     f"Spearman ρ={c_open['sr']:+.2f} (p={c_open['sp']:.2g}), "
                     f"Pearson r={c_open['pr']:+.2f} (p={c_open['pp']:.2g})")
    if c_all:
        lines.append(f"all points (n={c_all['n']}): "
                     f"Spearman ρ={c_all['sr']:+.2f} (p={c_all['sp']:.2g}), "
                     f"Pearson r={c_all['pr']:+.2f} (p={c_all['pp']:.2g})")
    if lines:
        ax.text(loc[0], loc[1], "\n".join(lines), transform=ax.transAxes, fontsize=9,
                va="bottom", ha="left",
                bbox=dict(boxstyle="round", fc="white", ec="#ccc", alpha=0.95))
    return lines


def fig_size_x(pts, ladder, out, c_open, c_all):
    """X = size (log), Y = contrast. Keeps the original orientation."""
    fig, ax = plt.subplots(figsize=(11, 7))
    for g, c in sorted(ladder.items(), key=lambda kv: kv[1]):
        ax.axhline(c, ls="--", lw=1.1, color=HUMAN_COLORS.get(g, "gray"), alpha=0.8)
        ax.text(0.995, c, " " + g.replace("child_", "age ").replace("plus", "+"),
                transform=ax.get_yaxis_transform(), va="center", ha="right",
                fontsize=8.5, color=HUMAN_COLORS.get(g, "gray"), fontweight="bold")
    ax.axhline(0, color="k", lw=0.8)

    draw_fit(ax, pts, orientation="size_x", color="#444")

    for p in sorted(pts, key=lambda d: d["size"]):
        col = FAMILY_COLORS.get(p["family"], "#888")
        mk = "o" if p["mtype"] == "instruct" else "s"
        if p["estimated"]:
            ax.scatter(p["size"], p["contrast"], s=120, facecolors="white",
                       edgecolors=col, linewidths=2, marker=mk, zorder=3)
        else:
            ax.scatter(p["size"], p["contrast"], s=120, color=col, marker=mk, zorder=3)
        ax.annotate(p["tag"], (p["size"], p["contrast"]), fontsize=7.5,
                    xytext=(6, 4), textcoords="offset points", color="#333")

    ax.set_xscale("log")
    sizes = [p["size"] for p in pts]
    ax.set_xlim(min(sizes) * 0.6, max(sizes) * 4.0)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}B"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("model size — parameters (billions, log scale; hollow = vendor-estimated closed model)")
    ax.set_ylabel("intent-vs-outcome contrast\n← outcome-driven (child-like)    intent-driven (adult-like) →")
    ax.set_title("Does scale predict adult-like moral judgment?")
    annotate_stats(ax, c_open, c_all)

    fams = list(dict.fromkeys(p["family"] for p in pts))
    handles = [Line2D([0], [0], marker="o", ls="", ms=8, color=FAMILY_COLORS.get(f, "#888"),
                      label=f) for f in fams]
    handles += [Line2D([0], [0], marker="s", ls="", ms=8, color="#555", label="base"),
                Line2D([0], [0], marker="o", ls="", ms=8, color="#555", label="instruct"),
                Line2D([0], [0], color="#444", lw=2, label="OLS fit (log size)")]
    ax.legend(handles=handles, fontsize=8, ncol=2, loc="upper left", framealpha=0.95)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.basename(out))


def fig_contrast_x(pts, ladder, out, c_open, c_all):
    """X = contrast, Y = size (log). Preferred orientation."""
    fig, ax = plt.subplots(figsize=(11, 7))
    sizes = [p["size"] for p in pts]
    ymin, ymax = min(sizes) * 0.5, max(sizes) * 3.0
    all_hc = list(ladder.values())
    xvals = [p["contrast"] for p in pts]
    xlo = min(xvals + all_hc) - 0.06
    xhi = max(xvals + all_hc) + 0.08

    ax.set_yscale("log")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ymin, ymax)

    for g, hc in sorted(ladder.items(), key=lambda kv: kv[1]):
        ax.axvline(hc, ls="--", lw=1.1, color=HUMAN_COLORS.get(g, "gray"), alpha=0.75)
        label = g.replace("child_", "age ").replace("_", "–").replace("plus", "+")
        ax.text(hc, ymax * 0.92, f" {label}", fontsize=8, va="top", ha="right",
                rotation=90, color=HUMAN_COLORS.get(g, "gray"), fontweight="bold")
    ax.axvline(0, color="k", lw=0.9)

    draw_fit(ax, pts, orientation="contrast_x", color="#444")

    for i, p in enumerate(sorted(pts, key=lambda d: d["contrast"])):
        col = FAMILY_COLORS.get(p["family"], "#888")
        mk = "o" if p["mtype"] == "instruct" else "s"
        if p["estimated"]:
            ax.scatter(p["contrast"], p["size"], s=120, facecolors="white",
                       edgecolors=col, linewidths=2, marker=mk, zorder=3)
        else:
            ax.scatter(p["contrast"], p["size"], s=120, color=col, marker=mk, zorder=3)
        x_off = 7 if i % 2 == 0 else -7
        ax.annotate(p["tag"], (p["contrast"], p["size"]), fontsize=7.5,
                    xytext=(x_off, 6), textcoords="offset points", color="#333",
                    arrowprops=dict(arrowstyle="-", color="#bbb", lw=0.6))

    tick_vals = [t for t in [0.5, 1, 2, 3, 5, 7, 10, 14, 20, 50, 100, 200, 500, 1000]
                 if ymin <= t <= ymax]
    ax.yaxis.set_major_locator(mticker.FixedLocator(tick_vals))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}B"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    ax.set_xlabel("intent-vs-outcome contrast\n"
                  "← outcome-driven (child-like)          intent-driven (adult-like) →")
    ax.set_ylabel("model size — parameters (B, log scale; hollow = estimated for closed models)")
    ax.set_title("Does scale predict adult-like (intent-based) moral judgment?")
    annotate_stats(ax, c_open, c_all, loc=(0.02, 0.02))

    fams = list(dict.fromkeys(p["family"] for p in pts))
    handles = [Line2D([0], [0], marker="o", ls="", ms=8, color=FAMILY_COLORS.get(f, "#888"),
                      label=f) for f in fams]
    handles += [Line2D([0], [0], marker="s", ls="", ms=8, color="#555", label="base"),
                Line2D([0], [0], marker="o", ls="", ms=8, color="#555", label="instruct"),
                Line2D([0], [0], color="#444", lw=2, label="OLS fit (log size)")]
    ax.legend(handles=handles, fontsize=8, ncol=2, loc="upper right", framealpha=0.95)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.basename(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(tc.ROOT, "outputs", "analysis"))
    ap.add_argument("--families", nargs="+", default=None,
                    help="Restrict to these families (default: Qwen Llama Mistral Claude Gemini GPT). "
                         "Pass 'all' for every family.")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    registry = tc.load_registry()
    ladder = tc.human_ladder()

    keep = None
    if a.families is None:
        keep = DEFAULT_FAMILIES
    elif "all" not in [f.lower() for f in a.families]:
        keep = {f.capitalize() if f.lower() != "olmo" else "OLMo"
                for f in a.families}
        # normalize GPT
        keep = {(f if f != "Gpt" else "GPT") for f in keep}

    pts = []
    for study, tag, path in tc.iter_item_means():
        size, mtype, _ = tc.parse_tag(tag, registry)
        if math.isnan(size):
            continue
        c = contrast_of(tc.pooled_cells(tc.load_cells(path)))
        if math.isnan(c):
            continue
        reg = registry.get(tag, {})
        estimated = (reg.get("params_estimated", "no").lower() == "yes")
        fam = family_of(tag)
        if keep is not None and fam not in keep:
            continue
        pts.append(dict(tag=tc.pretty(tag), study=study, size=size, mtype=mtype,
                        contrast=c, estimated=estimated, family=fam))

    if not pts:
        print("No sized models found."); return

    disclosed = [p for p in pts if not p["estimated"]]
    c_all = corr(pts)
    c_open = corr(disclosed)

    # ---- csv ----
    out_csv = os.path.join(a.out, "scale_vs_performance.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "study", "family", "size_B", "size_estimated",
                    "type", "contrast"])
        for p in sorted(pts, key=lambda d: d["size"]):
            w.writerow([p["tag"], p["study"], p["family"], p["size"],
                        "yes" if p["estimated"] else "no", p["mtype"],
                        round(p["contrast"], 4)])

    fig_size_x(pts, ladder, os.path.join(a.out, "scale_vs_performance.png"), c_open, c_all)
    fig_contrast_x(pts, ladder, os.path.join(a.out, "contrast_vs_scale.png"), c_open, c_all)

    print("\n=== SCALE vs PERFORMANCE (intent-vs-outcome contrast) ===")
    print(f"  families: {sorted({p['family'] for p in pts})}")
    print(f"  n={len(pts)} points ({len(disclosed)} disclosed-size)")
    if c_open:
        print(f"  disclosed: Spearman ρ={c_open['sr']:+.2f} (p={c_open['sp']:.2g}), "
              f"Pearson r={c_open['pr']:+.2f} (p={c_open['pp']:.2g})")
    if c_all:
        print(f"  all:       Spearman ρ={c_all['sr']:+.2f} (p={c_all['sp']:.2g}), "
              f"Pearson r={c_all['pr']:+.2f} (p={c_all['pp']:.2g})")
    print(f"\nwrote {os.path.relpath(out_csv, tc.ROOT)}")


if __name__ == "__main__":
    main()
