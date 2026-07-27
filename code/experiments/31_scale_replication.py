#!/usr/bin/env python3
"""
31_scale_replication.py -- YS2008↔YS2009 human_verbatim reprint agreement.

YS2008 human_verbatim uses permissibility 1–3; YS2009 uses blame 1–4. Averaging
reprints within scenario_group (tom_common.load_cells / 06_stats.py) therefore
mixes instruments after 0–1 normalization.

This script:
  - finds reprint pairs via scenario_group (same group, both YS2008 and YS2009)
  - loads item_means (behavior/ and agents/behavior/) for template=human_verbatim
  - for each model with both sources, correlates paired cell means and reports
    Bland–Altman (bias, SD of diffs, 95% LoA)
  - writes outputs/SCALE_REPLICATION.md with per-model + pooled stats and a
    recommendation on whether mixing sources in load_cells is OK

Does NOT change production averaging.

Usage
  python code/experiments/31_scale_replication.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))
import tom_common as tc  # noqa: E402

MASTER = os.path.join(ROOT, "dataset", "master", "moral_2x2_master.csv")
BEH_DIRS = [
    os.path.join(ROOT, "outputs", "behavior"),
    os.path.join(ROOT, "outputs", "agents", "behavior"),
]


def reprint_pairs(master_csv: str) -> list[tuple[str, str, str, str]]:
    """Return list of (group, condition, ys2008_story_id, ys2009_story_id)."""
    by: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for r in csv.DictReader(open(master_csv)):
        g = r.get("scenario_group") or r["scenario_id"]
        if r["source"] in ("YS2008", "YS2009"):
            by[(g, r["condition"])][r["source"]] = r["story_id"]
    pairs = []
    for (g, cond), srcs in sorted(by.items()):
        if "YS2008" in srcs and "YS2009" in srcs:
            pairs.append((g, cond, srcs["YS2008"], srcs["YS2009"]))
    return pairs


def load_hv_means(path: str) -> dict[str, float]:
    """story_id -> mean_norm_blame for human_verbatim only."""
    out = {}
    for r in csv.DictReader(open(path)):
        if r.get("template") != "human_verbatim":
            continue
        out[r["story_id"]] = float(r["mean_norm_blame"])
    return out


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def bland_altman(x: np.ndarray, y: np.ndarray) -> dict:
    """x = YS2008, y = YS2009; diffs = y - x."""
    d = y - x
    bias = float(np.mean(d))
    sd = float(np.std(d, ddof=1)) if len(d) > 1 else float("nan")
    loa_lo = bias - 1.96 * sd if sd == sd else float("nan")
    loa_hi = bias + 1.96 * sd if sd == sd else float("nan")
    return dict(bias=bias, sd_diff=sd, loa_lo=loa_lo, loa_hi=loa_hi, n_ba=len(d))


def paired_xy(means: dict[str, float], pairs: list) -> tuple[np.ndarray, np.ndarray, int]:
    xs, ys = [], []
    for _g, _c, s8, s9 in pairs:
        if s8 in means and s9 in means:
            xs.append(means[s8])
            ys.append(means[s9])
    return np.asarray(xs, float), np.asarray(ys, float), len(xs)


def fmt(x: float, nd: int = 3) -> str:
    if x != x:
        return "nan"
    return f"{x:.{nd}f}"


def recommend(r: float, bias: float, sd: float) -> str:
    """Heuristic: high r + small bias/SD → averaging OK; else keep separate."""
    abs_bias = abs(bias) if bias == bias else 1.0
    sd_ok = (sd == sd) and sd < 0.15
    bias_ok = abs_bias < 0.08
    r_ok = (r == r) and r >= 0.85
    if r_ok and bias_ok and sd_ok:
        return (
            "AGREEMENT_OK — r high and bias/SD small; averaging reprints within "
            "scenario_group for human_verbatim is acceptable (document the mix)."
        )
    if r_ok and (not bias_ok or not sd_ok):
        return (
            "CORRELATED_BUT_BIASED — keep sources separate for human_verbatim, or "
            "average only after noting instrument disagreement (YS2008 1–3 "
            "permissibility vs YS2009 1–4 blame)."
        )
    return (
        "WEAK_AGREEMENT — do NOT average YS2008+YS2009 human_verbatim within "
        "scenario_group; prefer within-template-only within source, or YS2008-only "
        "for the human_verbatim ladder."
    )


def current_averaging_note() -> str:
    return (
        "`tom_common.load_cells` (used by `06_stats.py`) averages all item_means "
        "rows that share `(template, scenario_group, condition)` — including "
        "YS2008 and YS2009 reprints under `human_verbatim`. That mixes the "
        "permissibility (1–3) and blame (1–4) instruments after 0–1 normalization.\n\n"
        "**One-line fix options (not applied here):**\n"
        "1. In `load_cells`, average within `source` first and keep sources separate "
        "for `human_verbatim`; or\n"
        "2. Prefer YS2008-only rows when both exist for `template=='human_verbatim'`; or\n"
        "3. Average reprints only for paraphrase templates (shared instrument), never "
        "for `human_verbatim`."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=MASTER)
    ap.add_argument(
        "--out", default=os.path.join(ROOT, "outputs", "SCALE_REPLICATION.md")
    )
    a = ap.parse_args()

    pairs = reprint_pairs(a.master)
    groups = sorted({g for g, *_ in pairs})
    print(f"Reprint pairs: {len(pairs)} cells across {len(groups)} scenario_groups")

    files = []
    for d in BEH_DIRS:
        files.extend(sorted(glob.glob(os.path.join(d, "item_means_*.csv"))))

    per_model = []
    all_x, all_y = [], []

    for path in files:
        tag = os.path.basename(path)[len("item_means_") : -4]
        means = load_hv_means(path)
        x, y, n = paired_xy(means, pairs)
        if n < 4:
            print(f"  skip {tag}: only {n} paired human_verbatim cells")
            continue
        r = pearson(x, y)
        ba = bland_altman(x, y)
        study = "agents" if "/agents/" in path.replace("\\", "/") else "local"
        per_model.append(
            dict(tag=tag, study=study, r=r, n=n, **ba, path=path)
        )
        all_x.extend(x.tolist())
        all_y.extend(y.tolist())
        print(
            f"  {tag:40s} n={n:3d} r={fmt(r)} bias={fmt(ba['bias'])} "
            f"sd={fmt(ba['sd_diff'])}"
        )

    ax = np.asarray(all_x, float)
    ay = np.asarray(all_y, float)
    if len(ax) >= 4:
        pooled_r = pearson(ax, ay)
        pooled_ba = bland_altman(ax, ay)
    else:
        pooled_r = float("nan")
        pooled_ba = dict(bias=float("nan"), sd_diff=float("nan"),
                         loa_lo=float("nan"), loa_hi=float("nan"), n_ba=0)

    # Recommendation from pooled (primary) + note if models disagree
    rec = recommend(pooled_r, pooled_ba["bias"], pooled_ba["sd_diff"])
    model_recs = [recommend(m["r"], m["bias"], m["sd_diff"]) for m in per_model]
    n_ok = sum(1 for s in model_recs if s.startswith("AGREEMENT_OK"))

    lines = [
        "# Scale replication — YS2008 ↔ YS2009 (`human_verbatim`)",
        "",
        "## Why this check",
        "",
        "YS2008 `human_verbatim` asks **permissibility 1–3**; YS2009 asks **blame 1–4**. "
        "Both are stored as `mean_norm_blame` on [0,1]. Collapsing reprints inside "
        "`scenario_group` therefore averages two different instruments.",
        "",
        f"**Reprint coverage:** {len(pairs)} paired cells "
        f"({len(groups)} scenario_groups × conditions present in both sources).",
        "",
        "## Current production averaging",
        "",
        current_averaging_note(),
        "",
        "## Per-model stats (template=`human_verbatim` only)",
        "",
        "Paired cells: matching `(scenario_group, condition)` with both YS2008 and "
        "YS2009 rows. Pearson *r* on normalized means; Bland–Altman on "
        "(YS2009 − YS2008).",
        "",
        "| model | study | n_pairs | Pearson r | bias (Y09−Y08) | SD(diff) | 95% LoA |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for m in sorted(per_model, key=lambda z: z["tag"]):
        loa = f"[{fmt(m['loa_lo'])}, {fmt(m['loa_hi'])}]"
        lines.append(
            f"| {m['tag']} | {m['study']} | {m['n']} | {fmt(m['r'])} | "
            f"{fmt(m['bias'])} | {fmt(m['sd_diff'])} | {loa} |"
        )

    lines.extend(
        [
            "",
            "## Pooled (all model paired cells stacked)",
            "",
            f"- n = {pooled_ba['n_ba']}",
            f"- Pearson r = **{fmt(pooled_r)}**",
            f"- Bland–Altman bias = **{fmt(pooled_ba['bias'])}**",
            f"- SD(diff) = **{fmt(pooled_ba['sd_diff'])}**",
            f"- 95% LoA = [{fmt(pooled_ba['loa_lo'])}, {fmt(pooled_ba['loa_hi'])}]",
            "",
            f"Models with AGREEMENT_OK heuristic: **{n_ok} / {len(per_model)}**",
            "",
            "## Recommendation",
            "",
            rec,
            "",
            "Production averaging was **not** changed. Prefer report-only until a "
            "fix decides among the one-line fix options above.",
            "",
            "*Generated by `code/experiments/31_scale_replication.py`.*",
            "",
        ]
    )

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {a.out}")
    print("Recommendation:", rec.split("—")[0].strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
