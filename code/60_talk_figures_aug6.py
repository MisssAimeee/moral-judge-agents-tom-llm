#!/usr/bin/env python3
"""
60_talk_figures_aug6.py — cosmetic talk re-plots (no new analysis).

1) Dual child-anchor bands on ladder/forest figures
2) Compact talk-safe open ladder
3) RSA heatmap (bigger labels) + rsa_vs_behavior scatter (28 pairs, r≈0.098)

Writes into outputs/... and copies the talk set into presentation_figures/.
"""
from __future__ import annotations

import csv, math, os, shutil
from collections import OrderedDict, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# ---- anchors (attempted − accidental) ---------------------------------------
# Digitized Naughty presented-first = primary (methods_child_measure.md)
# Text-reported naughty+punishable pooled = superseded secondary
DIGITIZED = OrderedDict([
    ("adult", 0.666), ("child_8plus", 0.63),
    ("child_6_7", 0.50), ("child_4_5", 0.24),
])
TEXT_REPORTED = OrderedDict([
    ("adult", 0.666), ("child_8plus", 0.46),
    ("child_6_7", 0.15), ("child_4_5", -0.14),
])
HUMAN_COLORS = {
    "adult": "#1a9850", "child_8plus": "#66bd63",
    "child_6_7": "#fdae61", "child_4_5": "#d73027",
}
HUMAN_LABELS = {
    "adult": "adult", "child_8plus": "age 8+",
    "child_6_7": "age 6–7", "child_4_5": "age 4–5",
}

PROVIDER_COLORS = {
    "OpenAI": "#10a37f", "Anthropic": "#cc785c", "Google": "#4285f4",
    "Meta": "#a259ff", "Alibaba": "#ff6a00", "Mistral": "#5468ff",
    "Moonshot": "#5b5fc7", "AllenAI": "#555555", "HuggingFaceH4": "#777777",
    "": "#888888",
}
FAMILY_COLORS = {
    "Claude": "#cc785c", "Gemini": "#4285f4", "GPT": "#10a37f",
    "Llama": "#a259ff", "Qwen": "#00909e", "Mistral": "#5468ff",
    "OLMo": "#7d3c98", "Gemma": "#e8710a", "other": "#888888",
}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
})


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def read_csv(path):
    path = Path(path)
    return list(csv.DictReader(open(path))) if path.exists() else []


def load_registry(path=ROOT / "dataset" / "model_registry.csv"):
    return {r["tag"]: r for r in read_csv(path)}


def disp(tag, registry):
    r = registry.get(tag)
    if r and r.get("display"):
        return r["display"]
    t = tag or ""
    if t.lower().startswith("gpt-"):
        return "GPT-" + t[4:].replace("_", ".")
    if t.lower().startswith("gemini-"):
        parts = t[7:].replace("_", ".").split("-")
        return "Gemini-" + "-".join(p[:1].upper() + p[1:] for p in parts)
    if t.lower().startswith("claude-"):
        parts = t[7:].replace("_", ".").split("-")
        return "Claude-" + "-".join(p[:1].upper() + p[1:] for p in parts)
    return (t.replace("meta-llama_", "").replace("Qwen_Qwen2.5-", "Qwen")
             .replace("Qwen_Qwen2_5-", "Qwen2.5-").replace("_", "-"))


def provider_of(tag, registry):
    r = registry.get(tag)
    if r and r.get("provider"):
        return r["provider"]
    t = (tag or "").lower()
    if t.startswith("gpt-"):
        return "OpenAI"
    if t.startswith("claude"):
        return "Anthropic"
    if t.startswith("gemini") or "gemma" in t:
        return "Google"
    if "qwen" in t:
        return "Alibaba"
    if "mistral" in t:
        return "Mistral"
    if "llama" in t or t.startswith("meta-"):
        return "Meta"
    if "olmo" in t or "tulu" in t:
        return "AllenAI"
    if "zephyr" in t:
        return "HuggingFaceH4"
    return ""


def color_of(tag, registry):
    return PROVIDER_COLORS.get(provider_of(tag, registry), "#888")


def draw_dual_bands(ax, orientation="v", y_top=None, y_bot=None, x_right=None,
                    label_fs=9, draw_adult=True):
    """Digitized = solid (labels at top); text-reported child = dotted (labels at bottom).
    Adult drawn once as a solid shared line."""
    handles = []
    if draw_adult:
        c = DIGITIZED["adult"]
        col = HUMAN_COLORS["adult"]
        if orientation == "v":
            ax.axvline(c, ls="-", lw=2.0, color=col, alpha=0.95, zorder=1)
            if y_top is not None:
                ax.text(c, y_top, f"adult {c:+.2f}", rotation=90, va="bottom",
                        ha="right", fontsize=label_fs, color=col, fontweight="bold")
        else:
            ax.axhline(c, ls="-", lw=2.0, color=col, alpha=0.95, zorder=1)
            if x_right is not None:
                ax.text(x_right, c, f" adult {c:+.2f}", va="center", ha="left",
                        fontsize=label_fs, color=col, fontweight="bold")
        handles.append(Line2D([0], [0], color=col, lw=2.0, ls="-",
                              label="adult (Young 2007, shared)"))

    for g in ("child_8plus", "child_6_7", "child_4_5"):
        col = HUMAN_COLORS[g]
        lab = HUMAN_LABELS[g]
        c_d = DIGITIZED[g]
        c_t = TEXT_REPORTED[g]
        if orientation == "v":
            ax.axvline(c_d, ls="-", lw=1.5, color=col, alpha=0.9, zorder=1)
            ax.axvline(c_t, ls=":", lw=1.3, color=col, alpha=0.55, zorder=1)
            if y_top is not None:
                ax.text(c_d, y_top, f"{lab} {c_d:+.2f}", rotation=90, va="bottom",
                        ha="right", fontsize=label_fs - 0.5, color=col, fontweight="bold")
            if y_bot is not None:
                ax.text(c_t, y_bot, f"{lab} {c_t:+.2f}", rotation=90, va="top",
                        ha="left", fontsize=label_fs - 1, color=col, alpha=0.75)
        else:
            ax.axhline(c_d, ls="-", lw=1.5, color=col, alpha=0.9, zorder=1)
            ax.axhline(c_t, ls=":", lw=1.3, color=col, alpha=0.55, zorder=1)
            if x_right is not None:
                ax.text(x_right, c_d, f" {lab} {c_d:+.2f}", va="center", ha="left",
                        fontsize=label_fs - 0.5, color=col, fontweight="bold")

    handles.append(Line2D([0], [0], color="#444", lw=1.5, ls="-",
                          label="digitized Naughty (primary)"))
    handles.append(Line2D([0], [0], color="#444", lw=1.3, ls=":",
                          label="text-reported (superseded)"))
    return handles


# =============================================================================
# 1a. Agent contrast forest (dual bands)
# =============================================================================
def fig_forest_dual(rows, registry, out, title=None):
    rows = [r for r in rows if not math.isnan(fnum(r["contrast"]))]
    rows.sort(key=lambda r: fnum(r["contrast"]))
    if not rows:
        print("skip forest", out); return
    n = len(rows)
    fig, ax = plt.subplots(figsize=(10, 0.55 * n + 2.8))
    band_handles = draw_dual_bands(ax, orientation="v", y_top=n + 0.2, y_bot=-0.55,
                                   label_fs=8)
    ax.axvline(0, color="k", lw=0.9)
    provs = []
    for i, r in enumerate(rows):
        c, lo, hi = fnum(r["contrast"]), fnum(r["ci_lo"]), fnum(r["ci_hi"])
        col = color_of(r["model"], registry)
        provs.append(provider_of(r["model"], registry))
        ax.errorbar(c, i, xerr=[[max(0, c - lo)], [max(0, hi - c)]], fmt="o",
                    color=col, ecolor=col, capsize=3, ms=8, elinewidth=2, alpha=0.9)
    ax.set_yticks(range(n))
    ax.set_yticklabels([disp(r["model"], registry) for r in rows], fontsize=11)
    ax.set_ylim(-1.4, n + 1.9)
    ax.set_xlabel("intent-vs-outcome contrast   (blame: attempted − accidental)\n"
                  "← outcome-driven (child-like)          intent-driven (adult-like) →")
    ax.set_title(title or (
        "Does the model judge by INTENT or by OUTCOME?\n"
        "(dot = point estimate, bar = 95% CI; solid = digitized primary, "
        "dotted = text-reported superseded)"))
    # legends
    from matplotlib.patches import Patch
    prov_handles = [Line2D([0], [0], marker="o", ls="", color=PROVIDER_COLORS[p],
                           label=p, ms=8)
                    for p in dict.fromkeys(provs) if p in PROVIDER_COLORS]
    leg1 = ax.legend(handles=prov_handles, title="provider", fontsize=8,
                     loc="lower right", framealpha=0.95)
    ax.add_artist(leg1)
    ax.legend(handles=band_handles, fontsize=8, loc="upper left", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


# =============================================================================
# 1b. Human-only forest with dual bands
# =============================================================================
def fig_human_only_dual(out):
    fig, ax = plt.subplots(figsize=(11, 3.8))
    vals = list(DIGITIZED.values()) + list(TEXT_REPORTED.values()) + [0.0]
    ax.set_xlim(min(vals) - 0.12, max(vals) + 0.12)
    ax.set_ylim(0, 1.15)
    ax.axvline(0, color="k", lw=0.9)
    # Digitized solid markers on upper rail
    for g, c in DIGITIZED.items():
        col = HUMAN_COLORS[g]
        ax.axvline(c, ls="-", lw=2.0 if g == "adult" else 1.6, color=col, alpha=0.95)
        ax.scatter([c], [0.72], s=110, color=col, zorder=3, marker="o")
        ax.text(c, 0.95, f"{HUMAN_LABELS[g]}\n{c:+.2f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=col)
    # Text-reported dotted markers on lower rail (skip adult duplicate)
    for g in ("child_8plus", "child_6_7", "child_4_5"):
        c = TEXT_REPORTED[g]
        col = HUMAN_COLORS[g]
        ax.axvline(c, ls=":", lw=1.4, color=col, alpha=0.6)
        ax.scatter([c], [0.28], s=90, facecolors="white", edgecolors=col,
                   linewidths=1.8, zorder=3)
        ax.text(c, 0.08, f"{HUMAN_LABELS[g]}\n{c:+.2f}", ha="center", va="top",
                fontsize=9, color=col, alpha=0.85)
    ax.set_yticks([])
    ax.set_xlabel("intent-vs-outcome contrast   (blame: attempted − accidental)\n"
                  "← outcome-driven (child-like)          intent-driven (adult-like) →")
    ax.set_title("Human developmental reference — both child measures\n"
                 "filled / solid = digitized Naughty presented-first (primary)   ·   "
                 "hollow / dotted = text-reported naughty+punishable (superseded)")
    handles = [
        Line2D([0], [0], marker="o", ls="-", color="#444", label="digitized (primary)"),
        Line2D([0], [0], marker="o", ls=":", mfc="white", mec="#444",
               color="#444", label="text-reported (superseded)"),
        Line2D([0], [0], color=HUMAN_COLORS["adult"], lw=2.0, label="adult (shared)"),
    ]
    ax.legend(handles=handles, loc="center right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


# =============================================================================
# 1c. Contrast vs scale with dual bands
# =============================================================================
def fig_contrast_vs_scale_dual(rows, registry, out):
    import matplotlib.ticker as mticker
    pts = []
    for r in rows:
        s, c = fnum(r.get("size_B")), fnum(r.get("contrast"))
        if not math.isnan(s) and not math.isnan(c):
            reg = registry.get(r["model"], {})
            est = str(reg.get("params_estimated", "no")).lower() == "yes"
            pts.append((s, c, r, est))
    if not pts:
        print("skip contrast_vs_scale", out); return
    all_s = [s for s, *_ in pts]
    ymin, ymax = min(all_s) * 0.5, max(all_s) * 2.2
    all_hc = list(DIGITIZED.values()) + [TEXT_REPORTED[g] for g in
                                         ("child_8plus", "child_6_7", "child_4_5")]
    xvals = [c for _, c, *_ in pts]
    xlo, xhi = min(xvals + all_hc) - 0.08, max(xvals + all_hc) + 0.10

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_yscale("log")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ymin, ymax)
    band_handles = draw_dual_bands(ax, orientation="v", y_top=ymax * 0.92,
                                   y_bot=ymin * 1.05, label_fs=8)
    ax.axvline(0, color="k", lw=0.9)

    provs = []
    for i, (s, c, r, est) in enumerate(sorted(pts, key=lambda p: p[1])):
        col = color_of(r["model"], registry)
        provs.append(provider_of(r["model"], registry))
        if est:
            ax.scatter(c, s, s=130, facecolors="white", edgecolors=col, linewidths=2.2, zorder=3)
        else:
            ax.scatter(c, s, s=130, color=col, zorder=3)
        ax.annotate(disp(r["model"], registry), (c, s), fontsize=9,
                    xytext=(8 if i % 2 == 0 else -8, 7), textcoords="offset points")

    tick_vals = [t for t in [10, 20, 50, 100, 200, 500, 1000] if ymin <= t <= ymax]
    ax.yaxis.set_major_locator(mticker.FixedLocator(tick_vals))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{int(y)}B"))
    ax.set_xlabel("intent-vs-outcome contrast\n"
                  "← outcome-driven (child-like)          intent-driven (adult-like) →")
    ax.set_ylabel("model size — parameters (B, log scale)")
    ax.set_title("Scale vs intent-based judgment\n"
                 "(solid = digitized primary child bands; dotted = text-reported superseded)")
    prov_handles = [Line2D([0], [0], marker="o", ls="", color=PROVIDER_COLORS[p],
                           label=p, ms=8)
                    for p in dict.fromkeys(provs) if p in PROVIDER_COLORS]
    leg1 = ax.legend(handles=prov_handles, title="provider", fontsize=8,
                     loc="lower right", framealpha=0.95)
    ax.add_artist(leg1)
    ax.legend(handles=band_handles, fontsize=8, loc="upper left", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


# =============================================================================
# 1d / 2. Master open ladder (full + talksafe)
# =============================================================================
def load_open_ladder_rows():
    """Prefer digitized openonly CSV; fall back to local stats."""
    p = ROOT / "outputs" / "master_all_models_digitized_openonly.csv"
    rows = []
    for r in read_csv(p):
        if r["model"].startswith("HUMAN"):
            continue
        rows.append({
            "model": r["model"],
            "family": r.get("family") or "other",
            "type": r.get("type") or "",
            "contrast": fnum(r["contrast"]),
            "ci_lo": fnum(r.get("ci_lo")),
            "ci_hi": fnum(r.get("ci_hi")),
            "degenerate": str(r.get("degenerate", "")).lower() == "true",
        })
    return rows


# Hand-picked talk-safe subset (11): extremes + base/instruct pairs + one/family
TALKSAFE_MODELS = [
    "OLMo-2-1124-7B-Instruct",          # most outcome-driven
    "HuggingFaceH4-zephyr-7b-beta",
    "Mistral-7B-Instruct-v0-3",
    "unsloth-gemma-2-9b-it",
    "Llama-3-1-Tulu-3-8B",
    "Qwen2.5-14B-Instruct",
    "Qwen2.5-7B-Instruct",
    "unsloth-Meta-Llama-3-1-8B-Instruct",
    "Qwen2.5-7B",                         # base pair
    "OLMo-2-1124-7B",                     # base pair
    "unsloth-Meta-Llama-3-1-8B",          # least outcome / near zero
]


def fig_master_ladder(rows, out, talksafe=False, figsize=None, label_fs=13):
    if talksafe:
        want = set(TALKSAFE_MODELS)
        rows = [r for r in rows if r["model"] in want]
        # preserve selection order by contrast sort at end
    rows = [r for r in rows if not r["degenerate"] or talksafe]
    if not talksafe:
        rows = [r for r in rows if not r["degenerate"]]
    rows = sorted(rows, key=lambda r: r["contrast"])
    n = len(rows)
    if figsize is None:
        figsize = (11.5, 6.8) if talksafe else (11, 0.34 * n + 3.4)
    fig, ax = plt.subplots(figsize=figsize)
    band_handles = draw_dual_bands(ax, orientation="v", y_top=n + 0.25, y_bot=-0.6,
                                   label_fs=9 if talksafe else 8)
    ax.axvline(0, color="k", lw=0.8)
    for i, r in enumerate(rows):
        col = FAMILY_COLORS.get(r["family"], "#888")
        lo = max(0.0, r["contrast"] - (r["ci_lo"] if not math.isnan(r["ci_lo"]) else r["contrast"]))
        hi = max(0.0, (r["ci_hi"] if not math.isnan(r["ci_hi"]) else r["contrast"]) - r["contrast"])
        if r["degenerate"]:
            ax.errorbar(r["contrast"], i, xerr=[[lo], [hi]], fmt="s",
                        mfc="none", mec="#999", ecolor="#ccc", capsize=2.5, ms=8,
                        elinewidth=1.2, alpha=0.9, zorder=3)
        else:
            ax.errorbar(r["contrast"], i, xerr=[[lo], [hi]], fmt="s",
                        color=col, ecolor=col, capsize=2.5, ms=8,
                        elinewidth=1.6, alpha=0.9, zorder=3)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r["model"] for r in rows], fontsize=label_fs)
    ax.set_ylim(-1.5, n + 2.0)
    ax.set_xlabel("intent-vs-outcome contrast   =   blame(attempted) − blame(accidental)\n"
                  "← OUTCOME-driven              INTENT-driven →", fontsize=12)
    ttl = ("Open-weight developmental ladder (talk-safe subset)\n"
           if talksafe else
           "Open-weight developmental ladder (full roster)\n")
    ax.set_title(ttl + "solid = digitized Naughty primary · dotted = text-reported superseded",
                 fontsize=13, fontweight="bold")
    fam_present = list(dict.fromkeys(r["family"] for r in rows))
    fam_handles = [Line2D([0], [0], marker="s", ls="", ms=8,
                          color=FAMILY_COLORS.get(f, "#888"), label=f)
                   for f in fam_present]
    leg1 = ax.legend(handles=fam_handles, title="family", fontsize=9,
                     loc="lower right", framealpha=0.95, ncol=2)
    ax.add_artist(leg1)
    ax.legend(handles=band_handles, fontsize=8, loc="upper left", framealpha=0.95)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", out, f"({n} models)")


# =============================================================================
# 3a. RSA heatmap with larger labels
# =============================================================================
def fig_rsa_heatmap(out):
    sim = read_csv(ROOT / "outputs" / "rsa" / "model_similarity.csv")
    # tags from peak rsa rows
    tags = []
    seen = set()
    for r in sim:
        if r.get("mode") != "peak":
            continue
        for k in ("model_a", "model_b"):
            t = r[k]
            if t not in seen:
                seen.add(t); tags.append(t)
    # stable order from convergence pairs if available
    pairs = read_csv(ROOT / "outputs" / "rsa" / "convergence_pairs.csv")
    if pairs:
        order = []
        for r in pairs:
            for k in ("model_a", "model_b"):
                if r[k] not in order:
                    order.append(r[k])
        tags = order
    M = np.full((len(tags), len(tags)), np.nan)
    for i, t1 in enumerate(tags):
        M[i, i] = 1.0
    for r in sim:
        if r.get("mode") != "peak":
            continue
        if r["model_a"] in tags and r["model_b"] in tags:
            i, j = tags.index(r["model_a"]), tags.index(r["model_b"])
            v = float(r["rsa_spearman"])
            M[i, j] = M[j, i] = v
    fig, ax = plt.subplots(figsize=(10.5, 8.8))
    im = ax.imshow(M, cmap="viridis", vmin=np.nanmin(M[np.isfinite(M) & (M < 1)]), vmax=1.0)
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags, rotation=40, ha="right", fontsize=14)
    ax.set_yticks(range(len(tags)))
    ax.set_yticklabels(tags, fontsize=14)
    for i in range(len(tags)):
        for j in range(len(tags)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=11, color="w" if M[i, j] < 0.85 else "#111",
                        fontweight="bold")
    ax.set_title("Representational similarity (RSA Spearman)\nat peak-intent layer",
                 fontsize=15, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("RSA Spearman", fontsize=13)
    cbar.ax.tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


# =============================================================================
# 3b. RSA vs behavior scatter (28 pairs, r≈0.098)
# =============================================================================
def fig_rsa_vs_behavior(out):
    pairs = read_csv(ROOT / "outputs" / "rsa" / "convergence_pairs.csv")
    if len(pairs) < 4:
        print("skip rsa_vs_behavior — need convergence_pairs.csv"); return
    # Published R3: spearman(RSA, |Δcontrast|) = +0.098
    # X = behavioral |Δ| so annotated r matches; left = answer alike
    x = np.array([float(r["abs_behavioral_diff"]) for r in pairs])
    y = np.array([float(r["rsa_spearman"]) for r in pairs])
    from scipy.stats import spearmanr, linregress
    r_obs = float(spearmanr(x, y).statistic)

    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    ax.scatter(x, y, s=70, color="#3b6ea5", edgecolors="white", linewidths=0.8,
               alpha=0.9, zorder=3)
    # faint OLS for visual guide (Spearman is the reported statistic)
    slope, intercept, *_ = linregress(x, y)
    xs = np.linspace(x.min(), x.max(), 80)
    ax.plot(xs, slope * xs + intercept, color="#666", lw=1.6, alpha=0.55, zorder=2)

    # label a few notable pairs
    notables = {
        ("OLMo-2-1124-7B", "OLMo-2-1124-7B-Instruct"),
        ("Qwen2.5-7B", "Qwen2.5-7B-Instruct"),
        ("Qwen2.5-0.5B", "Qwen2.5-0.5B-Instruct"),
        ("OLMo-2-1124-7B-Instruct", "Qwen2.5-0.5B"),
    }
    for r in pairs:
        key = (r["model_a"], r["model_b"])
        key_rev = (r["model_b"], r["model_a"])
        if key in notables or key_rev in notables:
            a_short = r["model_a"].replace("OLMo-2-1124-", "OLMo-").replace("Qwen2.5-", "Q")
            b_short = r["model_b"].replace("OLMo-2-1124-", "OLMo-").replace("Qwen2.5-", "Q")
            ax.annotate(f"{a_short}↔{b_short}",
                        (float(r["abs_behavioral_diff"]), float(r["rsa_spearman"])),
                        fontsize=8.5, xytext=(6, 6), textcoords="offset points",
                        color="#333")

    ax.text(0.03, 0.97,
            f"Spearman r = {r_obs:+.3f}\n"
            f"(published R3: r = +0.098, n = {len(pairs)})\n"
            f"95% CI crosses 0 → null",
            transform=ax.transAxes, va="top", ha="left", fontsize=12,
            bbox=dict(boxstyle="round", fc="white", ec="#ccc", alpha=0.95))
    ax.set_xlabel("behavioral |Δ contrast|   (← answer alike          answer differently →)",
                  fontsize=12)
    ax.set_ylabel("representational similarity (RSA Spearman)", fontsize=12)
    ax.set_title("Answer alike ≠ think alike\n"
                 "model-pair behavioral similarity vs representational similarity",
                 fontsize=14, fontweight="bold")
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", out, f"(r={r_obs:+.3f}, n={len(pairs)})")


# =============================================================================
def copy_to_presentation(src: Path, *dest_rel):
    for rel in dest_rel:
        dest = ROOT / "presentation_figures" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        print("  ->", dest.relative_to(ROOT))


def main():
    registry = load_registry()
    agent_rows = read_csv(ROOT / "outputs" / "agents" / "stats" / "contrast_by_model.csv")
    agents_fig = ROOT / "outputs" / "agents" / "figures"
    agents_fig.mkdir(parents=True, exist_ok=True)
    mech = ROOT / "outputs" / "rsa"
    mech.mkdir(parents=True, exist_ok=True)
    master_out = ROOT / "outputs"
    fig_final = ROOT / "outputs" / "figures_final"

    # --- 1. dual-band agent forests ---
    forest = agents_fig / "agent_contrast_forest.png"
    fig_forest_dual(agent_rows, registry, forest)
    copy_to_presentation(forest, "_headline/agent_contrast_forest.png",
                         "agents/agent_contrast_forest.png")
    # v3 nopro copy
    v3 = fig_final / "agent_contrast_forest_v3_nopro.png"
    shutil.copy2(forest, v3)
    copy_to_presentation(forest, "master_ladders/agent_contrast_forest_v3_nopro.png")

    human_only = agents_fig / "agent_contrast_forest_human_only.png"
    fig_human_only_dual(human_only)
    copy_to_presentation(human_only, "agents/agent_contrast_forest_human_only.png")

    cvs = agents_fig / "agent_contrast_vs_scale.png"
    fig_contrast_vs_scale_dual(agent_rows, registry, cvs)
    copy_to_presentation(cvs, "agents/agent_contrast_vs_scale.png")

    # --- 1d / 2. master ladders ---
    open_rows = load_open_ladder_rows()
    full = master_out / "master_developmental_ladder_digitized_openonly.png"
    fig_master_ladder(open_rows, full, talksafe=False, label_fs=10)
    if fig_final.exists():
        shutil.copy2(full, fig_final / "master_developmental_ladder_digitized_openonly.png")
    copy_to_presentation(full, "master_ladders/master_developmental_ladder_digitized_openonly.png")

    talk = master_out / "master_developmental_ladder_talksafe.png"
    fig_master_ladder(open_rows, talk, talksafe=True, figsize=(12, 7.2), label_fs=13)
    copy_to_presentation(talk, "master_ladders/master_developmental_ladder_talksafe.png")

    # --- 3. RSA ---
    heat = mech / "model_similarity_heatmap_rsa_spearman.png"
    fig_rsa_heatmap(heat)
    # also the figures_final / presentation names
    if fig_final.exists():
        shutil.copy2(heat, fig_final / "rsa_similarity_heatmap.png")
    copy_to_presentation(heat, "mechanistic/model_similarity_heatmap_rsa_spearman.png",
                         "master_ladders/rsa_similarity_heatmap.png")

    scatter = mech / "rsa_vs_behavior_scatter.png"
    fig_rsa_vs_behavior(scatter)
    copy_to_presentation(scatter, "mechanistic/rsa_vs_behavior_scatter.png")

    print("\nDone. Talk figures refreshed under presentation_figures/.")


if __name__ == "__main__":
    main()
