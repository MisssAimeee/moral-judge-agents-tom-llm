#!/usr/bin/env python3
"""
23_build_intent_reliance_summary.py -- Phase 1 / Task C5.

04_link_analysis.py reads outputs/behavior/intent_reliance_summary.csv, which never
existed. What exists is one file per model with one row per prompt template. This builds
the summary, prompt-averaged (NEXT_PHASE_PLAN 2c recommends the prompt-averaged estimate
as the headline) with the SD retained so prompt instability stays visible.

Two silent failure modes are handled explicitly:

1. NAMING. Behavior files are named Qwen_Qwen2.5-0.5B-Instruct (org separated by "_").
   Probe tags come from the .npz filename and are the last path segment only
   (Qwen2.5-0.5B-Instruct). 04 does r["model"].split("/")[-1], which handles a slash but
   not an underscore, so the summary's model column is written as Qwen/Qwen2.5-0.5B-Instruct
   and the existing split then yields a probe-matching tag. Some models also exist under
   both "2.5" and "2_5" spellings; those are de-duplicated.

2. THE INDEX IS UNSTABLE NEAR ZERO. intent_reliance_index = |b_int| / (|b_int| + |b_out|).
   For Qwen2.5-0.5B-Instruct / para_wrong7, b_intent=-0.0015 and b_outcome=0.0019 -- both
   pure noise -- yet the index reads 0.44. Averaging such values produces a number that
   looks like a finding. Templates must clear an effect-size floor to enter the average,
   and a model where no template clears it is marked degenerate and gets NO index rather
   than a misleading one.

Output
  outputs/behavior/intent_reliance_summary.csv
"""
import os, csv, glob, re, argparse
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

# A template counts only if the total moral signal clears this. Tuned against the known
# degenerate cases (Mistral-7B, Zephyr-7B) which sit far below it, while genuinely
# outcome-driven models (OLMo-Instruct, b_outcome ~0.3-0.6) clear it by an order of magnitude.
EFFECT_FLOOR = 0.05


def file_to_model(fname):
    """intent_reliance_Qwen_Qwen2.5-0.5B-Instruct.csv -> Qwen/Qwen2.5-0.5B-Instruct"""
    stem = os.path.basename(fname)[len("intent_reliance_"):-len(".csv")]
    return stem.replace("_", "/", 1)


def norm_key(model):
    """Collapse the 2.5 / 2_5 and 3.1 / 3_1 spelling variants so duplicates merge."""
    return re.sub(r"[._]", "-", model).lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavior", default=os.path.join(ROOT, "outputs", "behavior"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--floor", type=float, default=EFFECT_FLOOR)
    a = ap.parse_args()
    out = a.out or os.path.join(a.behavior, "intent_reliance_summary.csv")

    files = sorted(glob.glob(os.path.join(a.behavior, "intent_reliance_*.csv")))
    files = [f for f in files if not f.endswith("intent_reliance_summary.csv")]

    # merge spelling-variant duplicates, keeping whichever file has more templates
    by_key = {}
    for f in files:
        model = file_to_model(f)
        k = norm_key(model)
        rows = list(csv.DictReader(open(f)))
        prev = by_key.get(k)
        if prev is None or len(rows) > len(prev[1]):
            by_key[k] = (model, rows, f)

    summary = []
    for k, (model, rows, f) in sorted(by_key.items()):
        kept, dropped = [], []
        for r in rows:
            try:
                bi, bo = float(r["b_intent"]), float(r["b_outcome"])
            except (KeyError, ValueError):
                continue
            (kept if abs(bi) + abs(bo) >= a.floor else dropped).append((r["template"], bi, bo))

        degenerate = len(kept) == 0
        if degenerate:
            summary.append({
                "model": model, "intent_reliance_index": "", "intent_reliance_sd": "",
                "n_templates": 0, "n_templates_total": len(rows),
                "b_intent_mean": "", "b_outcome_mean": "", "degenerate": True,
            })
            print(f"  {model:44} DEGENERATE - no template clears |b_int|+|b_out| >= {a.floor}")
            continue

        idx = np.array([abs(bi) / (abs(bi) + abs(bo)) for _, bi, bo in kept])
        summary.append({
            "model": model,
            "intent_reliance_index": round(float(idx.mean()), 4),
            "intent_reliance_sd": round(float(idx.std(ddof=1)) if len(idx) > 1 else 0.0, 4),
            "n_templates": len(kept),
            "n_templates_total": len(rows),
            "b_intent_mean": round(float(np.mean([bi for _, bi, _ in kept])), 4),
            "b_outcome_mean": round(float(np.mean([bo for _, _, bo in kept])), 4),
            "degenerate": False,
        })
        note = f" ({len(dropped)} below floor)" if dropped else ""
        print(f"  {model:44} index={idx.mean():.3f} sd={idx.std(ddof=1) if len(idx)>1 else 0:.3f} "
              f"n={len(kept)}/{len(rows)}{note}")

    cols = ["model", "intent_reliance_index", "intent_reliance_sd", "n_templates",
            "n_templates_total", "b_intent_mean", "b_outcome_mean", "degenerate"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(summary)
    n_ok = sum(1 for s in summary if not s["degenerate"])
    print(f"\n-> {out}  ({n_ok} usable, {len(summary)-n_ok} degenerate)")

    # verify the join 04_link_analysis.py will perform actually resolves
    probe_dir = os.path.join(ROOT, "outputs", "probe")
    probe_tags = {os.path.basename(p)[:-len("_probe.csv")]
                  for p in glob.glob(os.path.join(probe_dir, "*_probe.csv"))}
    joined = [s["model"] for s in summary
              if not s["degenerate"] and s["model"].split("/")[-1] in probe_tags]
    print(f"join check: {len(joined)} of {len(probe_tags)} probed models matched -> {sorted(joined)}")
    if len(joined) < 6:
        print(f"WARNING: only {len(joined)} matched rows; the rep-vs-behavior correlation "
              f"will be underpowered. Probe tags present: {sorted(probe_tags)}")


if __name__ == "__main__":
    main()
