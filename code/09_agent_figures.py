#!/usr/bin/env python3
"""
09_agent_figures.py  --  Streamlined figures for the LARGE / DAILY-AGENT LLM
comparison (GPT, Claude, Gemini, Llama).

This is the "cleaner, easier-to-read" companion to 07_visualize.py. It is built for
a general audience comparing the big frontier models people actually use as agents,
not the internal Qwen size-ladder. It reads the SAME pipeline outputs plus a model
registry (dataset/model_registry.csv) that supplies parameter counts, provider, and
context window for closed models whose size isn't in the model name.

Figures written to outputs/agents/figures/:
  1. agent_scale.png            NEW. Y = model, X = parameters (B, log). How big is
                                each model? Filled = disclosed, hollow = estimated
                                (closed-weight vendors don't publish real counts).
  2. agent_contrast_forest.png  Headline: intent-vs-outcome contrast per model with
                                95% CI, colored by provider, vs the human ladder.
  3. agent_profiles.png         4-cell blame profile per model vs the adult human.
  4. agent_prompt_invariance.png  Contrast under each prompt wording (stability).
  5. agent_scale_vs_contrast.png  Does scale predict adult-like (intent) judgment?
  6. agent_weights.png          b_intent vs b_outcome per model.
  7. agent_pairwise.png         Pairwise contrast differences + significance.

Run after 06_stats.py (pointed at the agent outputs). matplotlib + numpy only.
"""
import os, csv, glob, math, argparse
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CELLS = ["neutral", "accidental", "attempted", "intentional"]
CELL_LABELS = ["neutral\n(no intent,\nno harm)", "accidental\n(no intent,\nharm)",
               "attempted\n(intent,\nno harm)", "intentional\n(intent,\nharm)"]

PROVIDER_COLORS = {
    "OpenAI":    "#10a37f",
    "Anthropic": "#cc785c",
    "Google":    "#4285f4",
    "Meta":      "#a259ff",
    "":          "#888888",
}
HUMAN_COLORS = {"adult": "#1a9850", "child_8plus": "#66bd63",
                "child_6_7": "#fdae61", "child_4_5": "#d73027"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
})


def fnum(x):
    try: return float(x)
    except (TypeError, ValueError): return float("nan")


def load_registry(path):
    reg = {}
    if path and os.path.exists(path):
        for r in csv.DictReader(open(path)):
            reg[r["tag"]] = r
    return reg


def disp(tag, registry):
    r = registry.get(tag)
    if r: return r.get("display", tag)
    return tag.replace("meta-llama_", "").replace("Qwen_Qwen2.5-", "Qwen")


def provider_of(tag, registry):
    r = registry.get(tag)
    return r.get("provider", "") if r else ""


def color_of(tag, registry):
    return PROVIDER_COLORS.get(provider_of(tag, registry), PROVIDER_COLORS[""])


def read_csv(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def load_human_ladder(human_csv):
    grp = defaultdict(dict)
    for r in read_csv(human_csv):
        if r.get("norm_blame", "").strip():
            grp[r["group"]][r["condition"]] = float(r["norm_blame"])
    ladder, profiles = {}, {}
    for g, p in grp.items():
        profiles[g] = p
        if "attempted" in p and "accidental" in p:
            ladder[g] = p["attempted"] - p["accidental"]
    return ladder, profiles


def pooled_profiles(behavior_dir, template=None):
    out = {}
    for f in sorted(glob.glob(os.path.join(behavior_dir, "item_means_*.csv"))):
        tag = os.path.basename(f)[len("item_means_"):-4]
        acc = defaultdict(list)
        for r in csv.DictReader(open(f)):
            if template and r["template"] != template:
                continue
            acc[r["condition"]].append(float(r["mean_norm_blame"]))
        out[tag] = {c: (sum(v)/len(v) if v else None) for c, v in acc.items()}
    return out


def provider_legend(ax, providers):
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", ls="", ms=9,
                      mfc=PROVIDER_COLORS.get(p, "#888"),
                      mec=PROVIDER_COLORS.get(p, "#888"), label=p)
               for p in providers]
    ax.legend(handles=handles, title="provider", fontsize=9, title_fontsize=9,
              loc="best", framealpha=0.9)


# ============================================================ 1. SCALE (NEW) ===
def fig_scale(registry, out, only_tags=None):
    rows = list(registry.values())
    if only_tags is not None:
        rows = [r for r in rows if r["tag"] in only_tags]
    rows = [r for r in rows if r.get("params_B", "").strip()]
    if not rows:
        print("skip agent_scale.png -> registry empty"); return
    rows.sort(key=lambda r: float(r["params_B"]))
    names   = [r["display"] for r in rows]
    params  = [float(r["params_B"]) for r in rows]
    provs   = [r["provider"] for r in rows]
    est     = [r.get("params_estimated", "no").lower() == "yes" for r in rows]
    ctx     = [r.get("context_k", "") for r in rows]

    fig, ax = plt.subplots(figsize=(11, 0.55 * len(rows) + 2.2))
    y = range(len(rows))
    for i, (p, prov, e) in enumerate(zip(params, provs, est)):
        col = PROVIDER_COLORS.get(prov, "#888")
        ax.hlines(i, 0.5, p, color=col, lw=2.2, alpha=0.55, zorder=1)
        if e:  # estimated -> hollow marker
            ax.scatter(p, i, s=150, facecolors="white", edgecolors=col,
                       linewidths=2.2, zorder=3)
        else:  # disclosed -> filled
            ax.scatter(p, i, s=150, color=col, zorder=3)
        label = f"~{p:g}B" if e else f"{p:g}B"
        ax.annotate(label, (p, i), xytext=(9, 0), textcoords="offset points",
                    va="center", fontsize=9, fontweight="bold", color=col)

    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{n}   ({c}K ctx)" if c else n
                        for n, c in zip(names, ctx)], fontsize=10)
    ax.set_xscale("log")
    ax.set_xlim(1, max(params) * 3)
    ax.set_xlabel("model size  —  parameters (billions, log scale)")
    ax.set_title("How large is each model?", pad=26)
    ax.text(0.5, 1.012,
            "filled = disclosed (open weights)   ·   hollow = estimate (closed weights)",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10, fontweight="normal", color="#444")
    fig.text(0.01, 0.01,
             "Closed-model parameter counts are NOT published by the vendor; "
             "hollow markers are rough community estimates for scale context only.",
             fontsize=7.5, style="italic", color="#555")
    providers = list(dict.fromkeys(provs))
    provider_legend(ax, providers)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out); plt.close(fig)
    print("wrote", os.path.basename(out))


# ==================================================== 2. CONTRAST FOREST =======
def fig_forest(rows, ladder, registry, out):
    rows = [r for r in rows if not math.isnan(fnum(r["contrast"]))]
    rows.sort(key=lambda r: fnum(r["contrast"]))
    if not rows:
        print("skip agent_contrast_forest.png -> no contrast rows"); return
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(rows) + 2.4))
    for g, c in sorted(ladder.items(), key=lambda x: x[1]):
        ax.axvline(c, ls="--", lw=1.2, color=HUMAN_COLORS.get(g, "gray"), alpha=0.8)
        ax.text(c, len(rows) - 0.2, g.replace("child_", "age ").replace("plus", "+"),
                rotation=90, va="top", ha="right", fontsize=8,
                color=HUMAN_COLORS.get(g, "gray"))
    ax.axvline(0, color="k", lw=0.9)
    provs = []
    for i, r in enumerate(rows):
        c, lo, hi = fnum(r["contrast"]), fnum(r["ci_lo"]), fnum(r["ci_hi"])
        col = color_of(r["model"], registry)
        provs.append(provider_of(r["model"], registry))
        ax.errorbar(c, i, xerr=[[max(0, c - lo)], [max(0, hi - c)]], fmt="o",
                    color=col, ecolor=col, capsize=3, ms=8, elinewidth=2, alpha=0.9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([disp(r["model"], registry) for r in rows], fontsize=10)
    ax.set_xlabel("intent-vs-outcome contrast   (blame: attempted − accidental)\n"
                  "← outcome-driven (child-like)          intent-driven (adult-like) →")
    ax.set_title("Does the model judge by INTENT or by OUTCOME?\n"
                 "(dot = point estimate, bar = 95% bootstrap CI; dashed = human reference)")
    provider_legend(ax, list(dict.fromkeys([p for p in provs if p])))
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    print("wrote", os.path.basename(out))


# ========================================================= 3. PROFILES =========
def fig_profiles(profiles, human_profiles, registry, out):
    profiles = {t: p for t, p in profiles.items()
                if any(v is not None for v in p.values())}
    if not profiles:
        print("skip agent_profiles.png -> no profiles"); return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = list(range(len(CELLS)))

    # Sort tags by provider then by param size so same-provider models plot in order
    def _sort_key(tag):
        r = registry.get(tag, {})
        return (r.get("provider", ""), float(r.get("params_B") or 0))

    LINESTYLES = ["-", "--", ":"]
    MARKERS    = ["o", "s", "^"]
    prov_idx: dict = {}

    for tag in sorted(profiles.keys(), key=_sort_key):
        prof = profiles[tag]
        col  = color_of(tag, registry)
        prov = provider_of(tag, registry)
        idx  = prov_idx.get(prov, 0)
        prov_idx[prov] = idx + 1
        ax.plot(x, [prof.get(c) for c in CELLS],
                marker=MARKERS[idx % len(MARKERS)],
                linestyle=LINESTYLES[idx % len(LINESTYLES)],
                lw=1.8, ms=6, alpha=0.9, color=col, label=disp(tag, registry))

    if "adult" in human_profiles:
        ax.plot(x, [human_profiles["adult"].get(c) for c in CELLS], "s--", lw=3,
                color="k", label="HUMAN adult", zorder=5)
    ax.set_xticks(x); ax.set_xticklabels(CELL_LABELS, fontsize=9)
    ax.set_ylabel("normalized blame   (0 = none, 1 = maximum)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Blame profile across the 2×2 moral scenarios")
    ax.legend(fontsize=8, ncol=2, framealpha=0.9)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    print("wrote", os.path.basename(out))


# =============================================== 4. PROMPT INVARIANCE ==========
def fig_invariance(stats_dir, registry, out):
    p = os.path.join(stats_dir, "prompt_invariance_contrast.csv")
    rows = read_csv(p)
    if not rows:
        print("skip agent_prompt_invariance.png -> no invariance csv"); return
    tcols = [c for c in rows[0].keys() if c not in ("model", "sd", "range", "sign_flips")]
    rows.sort(key=lambda r: np.nanmean([fnum(r[t]) for t in tcols]))
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(rows) + 2.2))
    ax.axvline(0, color="k", lw=0.9)
    provs = []
    for i, r in enumerate(rows):
        col = color_of(r["model"], registry); provs.append(provider_of(r["model"], registry))
        vals = [fnum(r[t]) for t in tcols if not math.isnan(fnum(r.get(t)))]
        if vals:
            ax.plot([min(vals), max(vals)], [i, i], color=col, lw=2, alpha=0.4, zorder=1)
            ax.scatter(vals, [i]*len(vals), s=32, color=col, alpha=0.75, zorder=2)
            ax.scatter(np.mean(vals), i, marker="D", s=55, color=col,
                       edgecolors="k", linewidths=0.8, zorder=3)
        if vals and (min(vals) < 0 < max(vals)):
            ax.text(max(vals) + 0.02, i, "sign flip", color="#d73027",
                    fontsize=8, va="center", fontweight="bold")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([disp(r["model"], registry) for r in rows], fontsize=10)
    ax.set_xlabel("intent-vs-outcome contrast under each prompt wording "
                  "(♦ = mean across prompts)")
    ax.set_title("Is the result stable across prompt wordings?\n"
                 "(tight cluster = robust · wide spread / sign flip = prompt-sensitive)")
    provider_legend(ax, list(dict.fromkeys([p for p in provs if p])))
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    print("wrote", os.path.basename(out))


# ============================================= 5. SCALE vs CONTRAST ============
def _fit_contrast_logsize(pts):
    """pts = [(size, contrast, ...)]. Return (slope, intercept, r, p, n) or None."""
    from scipy import stats as spstats
    if len(pts) < 3:
        return None
    x = np.log10([s for s, *_ in pts])
    y = np.array([c for _, c, *_ in pts])
    slope, intercept, r, p, _ = spstats.linregress(x, y)
    return slope, intercept, r, p, len(pts)


def fig_scale_vs_contrast(rows, ladder, registry, out):
    import matplotlib.ticker as mticker
    pts = []
    for r in rows:
        s, c = fnum(r.get("size_B")), fnum(r.get("contrast"))
        if not math.isnan(s) and not math.isnan(c):
            reg = registry.get(r["model"], {})
            est = reg.get("params_estimated", "no").lower() == "yes"
            pts.append((s, c, r, est))
    if not pts:
        print("skip agent_scale_vs_contrast.png -> no size/contrast pairs"); return

    all_s  = [s for s, *_ in pts]
    xmin, xmax = min(all_s) * 0.5, max(all_s) * 2.2
    # y range: include all human reference lines so the gap is visible
    all_hc = list(ladder.values())
    yvals  = [c for _, c, *_ in pts]
    ylo = min(yvals + all_hc) - 0.06
    yhi = max(yvals + all_hc) + 0.08

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ylo, yhi)

    # Human reference lines — drawn after xlim/ylim are finalised
    for g, hc in sorted(ladder.items(), key=lambda kv: kv[1]):
        ax.axhline(hc, ls="--", lw=1.1, color=HUMAN_COLORS.get(g, "gray"), alpha=0.75)
        label = g.replace("child_", "age ").replace("_", "–").replace("plus", "+")
        ax.text(xmax * 0.99, hc, f" {label}", fontsize=8, va="center", ha="right",
                color=HUMAN_COLORS.get(g, "gray"), fontweight="bold")
    ax.axhline(0, color="k", lw=0.9)

    # Correlation / OLS fit line
    fit = _fit_contrast_logsize(pts)
    if fit:
        slope, intercept, r, p, n = fit
        xs = np.logspace(np.log10(min(all_s)), np.log10(max(all_s)), 80)
        ax.plot(xs, slope * np.log10(xs) + intercept, color="#444", lw=2.0,
                alpha=0.75, zorder=2, label=f"OLS fit (r={r:+.2f}, n={n})")
        ax.text(0.02, 0.02, f"Pearson r={r:+.2f} (p={p:.2g}, n={n})",
                transform=ax.transAxes, fontsize=9, va="bottom",
                bbox=dict(boxstyle="round", fc="white", ec="#ccc", alpha=0.95))

    provs = []
    pts_sorted = sorted(pts, key=lambda p: p[0])   # sort by size for label offset cycling
    for i, (s, c, r, est) in enumerate(pts_sorted):
        col = color_of(r["model"], registry)
        provs.append(provider_of(r["model"], registry))
        if est:
            ax.scatter(s, c, s=130, facecolors="white", edgecolors=col, linewidths=2.2, zorder=3)
        else:
            ax.scatter(s, c, s=130, color=col, zorder=3)
        y_off = 9 if i % 2 == 0 else -17   # alternate up/down to avoid label collision
        ax.annotate(disp(r["model"], registry), (s, c), fontsize=8.5,
                    xytext=(7, y_off), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color="#bbb", lw=0.7))

    # Clean x-axis ticks
    tick_vals = [t for t in [10, 20, 50, 100, 200, 500, 1000] if xmin <= t <= xmax]
    ax.xaxis.set_major_locator(mticker.FixedLocator(tick_vals))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}B"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    ax.set_xlabel("model size — parameters (B, log scale; hollow = estimated for closed models)")
    ax.set_ylabel("intent-vs-outcome contrast")
    ax.set_title("Does scale predict adult-like (intent-based) moral judgment?\n"
                 "← outcome-driven (child-like)                          intent-driven (adult-like) →",
                 fontsize=12)
    provider_legend(ax, list(dict.fromkeys([p for p in provs if p])))
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", os.path.basename(out))


# ======================================== 5b. CONTRAST vs SCALE (flipped) ======
def fig_contrast_vs_scale(rows, ladder, registry, out):
    """Same data as fig_scale_vs_contrast, but axes flipped:
    X = intent-vs-outcome contrast, Y = model size (params, log).
    """
    import matplotlib.ticker as mticker
    pts = []
    for r in rows:
        s, c = fnum(r.get("size_B")), fnum(r.get("contrast"))
        if not math.isnan(s) and not math.isnan(c):
            reg = registry.get(r["model"], {})
            est = reg.get("params_estimated", "no").lower() == "yes"
            pts.append((s, c, r, est))
    if not pts:
        print("skip agent_contrast_vs_scale.png -> no size/contrast pairs"); return

    all_s  = [s for s, *_ in pts]
    ymin, ymax = min(all_s) * 0.5, max(all_s) * 2.2
    all_hc = list(ladder.values())
    xvals  = [c for _, c, *_ in pts]
    xlo = min(xvals + all_hc) - 0.06
    xhi = max(xvals + all_hc) + 0.08

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_yscale("log")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ymin, ymax)

    # Human reference lines — vertical (now on the contrast / X axis)
    for g, hc in sorted(ladder.items(), key=lambda kv: kv[1]):
        ax.axvline(hc, ls="--", lw=1.1, color=HUMAN_COLORS.get(g, "gray"), alpha=0.75)
        label = g.replace("child_", "age ").replace("_", "–").replace("plus", "+")
        ax.text(hc, ymax * 0.92, f" {label}", fontsize=8, va="top", ha="right",
                rotation=90, color=HUMAN_COLORS.get(g, "gray"), fontweight="bold")
    ax.axvline(0, color="k", lw=0.9)

    # Correlation / OLS fit line
    fit = _fit_contrast_logsize(pts)
    if fit:
        slope, intercept, r, p, n = fit
        xs = np.logspace(np.log10(min(all_s)), np.log10(max(all_s)), 80)
        ys = slope * np.log10(xs) + intercept
        ax.plot(ys, xs, color="#444", lw=2.0, alpha=0.75, zorder=2,
                label=f"OLS fit (r={r:+.2f}, n={n})")
        ax.text(0.02, 0.02, f"Pearson r={r:+.2f} (p={p:.2g}, n={n})",
                transform=ax.transAxes, fontsize=9, va="bottom",
                bbox=dict(boxstyle="round", fc="white", ec="#ccc", alpha=0.95))

    provs = []
    pts_sorted = sorted(pts, key=lambda p: p[1])   # sort by contrast for label offset cycling
    for i, (s, c, r, est) in enumerate(pts_sorted):
        col = color_of(r["model"], registry)
        provs.append(provider_of(r["model"], registry))
        if est:
            ax.scatter(c, s, s=130, facecolors="white", edgecolors=col, linewidths=2.2, zorder=3)
        else:
            ax.scatter(c, s, s=130, color=col, zorder=3)
        x_off = 9 if i % 2 == 0 else -9
        ax.annotate(disp(r["model"], registry), (c, s), fontsize=8.5,
                    xytext=(x_off, 7), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color="#bbb", lw=0.7))

    tick_vals = [t for t in [10, 20, 50, 100, 200, 500, 1000] if ymin <= t <= ymax]
    ax.yaxis.set_major_locator(mticker.FixedLocator(tick_vals))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{int(y)}B"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    ax.set_xlabel("intent-vs-outcome contrast\n"
                  "← outcome-driven (child-like)          intent-driven (adult-like) →")
    ax.set_ylabel("model size — parameters (B, log scale; hollow = estimated for closed models)")
    ax.set_title("Does scale predict adult-like (intent-based) moral judgment?",
                 fontsize=12)
    provider_legend(ax, list(dict.fromkeys([p for p in provs if p])))
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", os.path.basename(out))


# ================================================== 6. WEIGHTS SCATTER =========
def fig_weights(rows, registry, out):
    pts = [(fnum(r["b_outcome"]), fnum(r["b_intent"]), r) for r in rows
           if not math.isnan(fnum(r.get("b_intent"))) and not math.isnan(fnum(r.get("b_outcome")))]
    if not pts:
        print("skip agent_weights.png -> no weights"); return
    fig, ax = plt.subplots(figsize=(7, 6.5))
    lim = max([abs(v) for p in pts for v in p[:2]] + [0.1]) * 1.15
    ax.plot([0, lim], [0, lim], ls=":", color="gray", label="equal weighting")
    ax.fill_between([0, lim], [0, lim], lim, color="#1a9850", alpha=0.05)
    ax.text(lim*0.05, lim*0.9, "intent-weighted\n(adult-like)", fontsize=9, color="#1a9850")
    ax.text(lim*0.55, lim*0.05, "outcome-weighted\n(child-like)", fontsize=9, color="#d73027")
    provs = []
    for bo, bi, r in pts:
        col = color_of(r["model"], registry); provs.append(provider_of(r["model"], registry))
        ax.scatter(bo, bi, color=col, s=70, zorder=3)
        ax.annotate(disp(r["model"], registry), (bo, bi), fontsize=8,
                    xytext=(5, 3), textcoords="offset points")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("b_outcome  (weight placed on the bad OUTCOME)")
    ax.set_ylabel("b_intent  (weight placed on the bad INTENT)")
    ax.set_title("What drives each model's blame: intent vs outcome")
    provider_legend(ax, list(dict.fromkeys([p for p in provs if p])))
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    print("wrote", os.path.basename(out))


# =================================================== 7. PAIRWISE HEATMAP =======
def fig_pairwise(stats_dir, registry, out):
    p = os.path.join(stats_dir, "pairwise_model_diffs.csv")
    rows = read_csv(p)
    if not rows:
        print("skip agent_pairwise.png -> no pairwise csv"); return
    models = sorted({r["model_a"] for r in rows} | {r["model_b"] for r in rows})
    idx = {m: i for i, m in enumerate(models)}
    n = len(models)
    M = np.full((n, n), np.nan); sig = {}
    for r in rows:
        i, j = idx[r["model_a"]], idx[r["model_b"]]
        d = fnum(r["contrast_diff"]); M[i, j] = d; M[j, i] = -d
        sig[(i, j)] = r.get("distinguishable") == "yes"
    fig, ax = plt.subplots(figsize=(1.0 * n + 3, 1.0 * n + 3))
    vmax = np.nanmax(np.abs(M)) if np.isfinite(np.nanmax(np.abs(M))) else 1
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([disp(m, registry) for m in models], rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels([disp(m, registry) for m in models], fontsize=9)
    for (i, j), s in sig.items():
        if s:
            ax.text(j, i, "*", ha="center", va="center", fontsize=13)
            ax.text(i, j, "*", ha="center", va="center", fontsize=13)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="contrast difference (row − col)")
    ax.set_title("Which models differ from each other?\n(* = 95% CI excludes 0)")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", os.path.basename(out))


def main():
    here = os.path.dirname(__file__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavior", default=os.path.join(here, "..", "outputs", "agents", "behavior"))
    ap.add_argument("--stats",    default=os.path.join(here, "..", "outputs", "agents", "stats"))
    ap.add_argument("--registry", default=os.path.join(here, "..", "dataset", "model_registry.csv"))
    ap.add_argument("--human",    default=os.path.join(here, "..", "dataset",
                                                       "human_reference", "human_reference.csv"))
    ap.add_argument("--out",      default=os.path.join(here, "..", "outputs", "agents", "figures"))
    ap.add_argument("--template", default="human_verbatim")
    ap.add_argument("--scale_only", action="store_true",
                    help="Only draw the registry-based scale figure (no behavioral data needed).")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    registry = load_registry(a.registry)
    ladder, human_profiles = load_human_ladder(a.human)

    # Scale figure needs only the registry -> can run before any API call.
    fig_scale(registry, os.path.join(a.out, "agent_scale.png"))
    if a.scale_only:
        print("scale_only: done. Figures in", a.out); return

    rows = read_csv(os.path.join(a.stats, "contrast_by_model.csv"))
    profiles = pooled_profiles(a.behavior, a.template)

    jobs = [
        ("agent_contrast_forest.png",   lambda p: fig_forest(rows, ladder, registry, p)),
        ("agent_profiles.png",          lambda p: fig_profiles(profiles, human_profiles, registry, p)),
        ("agent_prompt_invariance.png", lambda p: fig_invariance(a.stats, registry, p)),
        ("agent_scale_vs_contrast.png", lambda p: fig_scale_vs_contrast(rows, ladder, registry, p)),
        ("agent_contrast_vs_scale.png", lambda p: fig_contrast_vs_scale(rows, ladder, registry, p)),
        ("agent_weights.png",           lambda p: fig_weights(rows, registry, p)),
        ("agent_pairwise.png",          lambda p: fig_pairwise(a.stats, registry, p)),
    ]
    for name, fn in jobs:
        try:
            fn(os.path.join(a.out, name))
        except Exception as e:
            print("skip", name, "->", e)
    print("figures in", a.out)


if __name__ == "__main__":
    main()
