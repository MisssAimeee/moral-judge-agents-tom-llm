#!/usr/bin/env python3
"""
10_master_figure.py -- One shareable "master" figure + combined CSV that places
EVERY tested model (local open-weight study + cloud daily-agent study) on the
human developmental ladder for intent-vs-outcome moral judgment.

Reads the two independent pipelines' stats:
  outputs/stats/contrast_by_model.csv          (open-weight, local GPU)
  outputs/agents/stats/contrast_by_model.csv   (closed APIs: Claude, Gemini)

Writes:
  outputs/master_developmental_ladder.png
  outputs/master_all_models.csv
"""
import os, csv, math
from collections import OrderedDict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")

LOCAL_STATS = os.path.join(ROOT, "outputs", "stats", "contrast_by_model.csv")
AGENT_STATS = os.path.join(ROOT, "outputs", "agents", "stats", "contrast_by_model.csv")
OUT_PNG     = os.path.join(ROOT, "outputs", "master_developmental_ladder.png")
OUT_CSV     = os.path.join(ROOT, "outputs", "master_all_models.csv")

# Human developmental reference (attempted - accidental contrast, 0-1 blame scale)
HUMAN = OrderedDict([
    ("adult",        0.666),
    ("child_8plus",  0.46),
    ("child_6_7",    0.15),
    ("child_4_5",    -0.14),
])
HUMAN_COLORS = {"adult": "#1a9850", "child_8plus": "#66bd63",
                "child_6_7": "#fdae61", "child_4_5": "#d73027"}

# Family -> color for the model dots
FAMILY_COLORS = {
    "Claude": "#cc785c", "Gemini": "#4285f4", "GPT": "#10a37f",
    "Llama": "#a259ff", "Qwen": "#00909e", "Mistral": "#ff7000",
    "OLMo": "#7d3c98", "Gemma": "#e8710a", "Phi": "#c2185b", "other": "#888888",
}


def fnum(x):
    try: return float(x)
    except (TypeError, ValueError): return float("nan")


def family_of(name):
    n = name.lower()
    if "claude" in n: return "Claude"
    if "gemini" in n: return "Gemini"
    if "gpt" in n:    return "GPT"
    if "llama" in n:  return "Llama"
    if "qwen" in n:   return "Qwen"
    if "mistral" in n:return "Mistral"
    if "olmo" in n:   return "OLMo"
    if "gemma" in n:  return "Gemma"
    if "phi" in n:    return "Phi"
    return "other"


def pretty(name):
    return (name.replace("meta-llama_", "").replace("Qwen_", "")
                .replace("mistralai_", "").replace("allenai_", "")
                .replace("google_", "").replace("microsoft_", "")
                .replace("Qwen2_5", "Qwen2.5").replace("-20251001", "")
                .replace("_", "-"))


def norm_key(name):
    """Collapse pipeline naming variants (2_5 vs 2.5) so we don't double-plot."""
    return pretty(name).lower()


def load(path, study):
    rows = []
    if not os.path.exists(path):
        return rows
    for r in csv.DictReader(open(path)):
        c = fnum(r.get("contrast"))
        if math.isnan(c):
            continue
        rows.append({
            "model": pretty(r["model"]),
            "key": norm_key(r["model"]),
            "study": study,
            "type": r.get("type", ""),
            "contrast": c,
            "ci_lo": fnum(r.get("ci_lo")),
            "ci_hi": fnum(r.get("ci_hi")),
            "sig": r.get("sig_vs_0", ""),
            "nearest": r.get("nearest_human_group", ""),
            "family": family_of(r["model"]),
            "degenerate": str(r.get("degenerate", "")).strip().lower() == "true",
        })
    return rows


def main():
    rows = load(AGENT_STATS, "cloud API") + load(LOCAL_STATS, "local open-weight")

    # Dedupe naming variants: keep first occurrence per normalized key.
    seen, uniq = set(), []
    for r in rows:
        if r["key"] in seen:
            continue
        seen.add(r["key"])
        uniq.append(r)
    rows = sorted(uniq, key=lambda r: r["contrast"])

    # ---- combined CSV ----
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "family", "study", "type", "contrast",
                    "ci_lo", "ci_hi", "sig_vs_0", "nearest_human_group", "degenerate"])
        for g, c in HUMAN.items():
            w.writerow([f"HUMAN {g}", "human", "reference", "human",
                        f"{c:+.3f}", "", "", "", g, "False"])
        for r in rows:
            w.writerow([r["model"], r["family"], r["study"], r["type"],
                        f"{r['contrast']:+.3f}", f"{r['ci_lo']:+.3f}",
                        f"{r['ci_hi']:+.3f}", r["sig"], r["nearest"],
                        "True" if r["degenerate"] else "False"])
    print("wrote", os.path.relpath(OUT_CSV, ROOT), f"({len(rows)} models)")

    # ---- master figure ----
    n = len(rows)
    fig, ax = plt.subplots(figsize=(11, 0.34 * n + 3))

    # human reference bands
    for g, c in HUMAN.items():
        col = HUMAN_COLORS[g]
        ax.axvline(c, ls="--", lw=1.4, color=col, alpha=0.85, zorder=1)
        ax.text(c, n + 0.4, g.replace("child_", "age ").replace("plus", "+"),
                rotation=90, va="bottom", ha="center", fontsize=8.5,
                color=col, fontweight="bold")
    ax.axvline(0, color="k", lw=0.8, zorder=1)

    for i, r in enumerate(rows):
        col = FAMILY_COLORS.get(r["family"], "#888")
        lo = max(0, r["contrast"] - r["ci_lo"])
        hi = max(0, r["ci_hi"] - r["contrast"])
        marker = "o" if r["study"] == "cloud API" else "s"
        if r["degenerate"]:
            # degenerate (near-constant ratings): hollow grey marker, no fill,
            # so a QC-failed 0.000 is visibly NOT a real null.
            ax.errorbar(r["contrast"], i, xerr=[[lo], [hi]], fmt=marker,
                        mfc="none", mec="#999999", ecolor="#cccccc", capsize=2.5,
                        ms=7, elinewidth=1.2, alpha=0.9, zorder=3)
        else:
            ax.errorbar(r["contrast"], i, xerr=[[lo], [hi]], fmt=marker,
                        color=col, ecolor=col, capsize=2.5, ms=7,
                        elinewidth=1.6, alpha=0.9, zorder=3)

    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [f"{r['model']}  ({'cloud' if r['study']=='cloud API' else 'local'}/{r['type']})"
         + ("  [degenerate]" if r["degenerate"] else "")
         for r in rows], fontsize=8.5)
    ax.set_ylim(-1, n + 1.5)
    ax.set_xlabel("intent-vs-outcome contrast   =   blame(attempted) − blame(accidental)\n"
                  "← OUTCOME-driven (young-child-like)              INTENT-driven (adult-like) →",
                  fontsize=10)
    ax.set_title("Master ladder: every tested model vs the human developmental curve\n"
                 "(dot = cloud API model · square = local open-weight · bar = 95% bootstrap CI)",
                 fontsize=12, fontweight="bold")

    fam_present = list(dict.fromkeys(r["family"] for r in rows))
    handles = [Line2D([0], [0], marker="o", ls="", ms=8, color=FAMILY_COLORS[f], label=f)
               for f in fam_present]
    if any(r["degenerate"] for r in rows):
        handles.append(Line2D([0], [0], marker="s", ls="", ms=8, mfc="none",
                              mec="#999999", label="degenerate (QC-failed)"))
    ax.legend(handles=handles, title="model family", fontsize=8.5,
              title_fontsize=9, loc="lower right", framealpha=0.95, ncol=2)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(OUT_PNG, ROOT))


if __name__ == "__main__":
    main()
