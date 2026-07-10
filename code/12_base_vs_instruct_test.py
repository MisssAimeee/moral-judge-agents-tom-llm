#!/usr/bin/env python3
"""
12_base_vs_instruct_test.py  --  Reinforce finding #3 (precursor to roadmap #4):
a FORMAL paired test that instruction-tuning shifts the intent-vs-outcome
contrast in the outcome-biased (negative) direction. ANALYSIS ONLY.

Finding #3 was stated as an observation ("instruct looks more outcome-biased").
This turns it into statistics by pairing each instruct model with its base
counterpart at the SAME family+size, so model size is controlled:

    delta = contrast(instruct) - contrast(base)      (per matched pair)

We test whether the deltas are systematically negative with:
  * a paired t-test and Wilcoxon signed-rank across the matched pairs,
  * a pooled scenario-level paired bootstrap CI (all Qwen pairs stacked),
reported for the Qwen2.5 ladder specifically and for all matched pairs.

Outputs:
  outputs/analysis/base_vs_instruct_pairs.csv
  console summary
"""
import os, csv, math, argparse
from collections import defaultdict
import numpy as np
from scipy import stats
import tom_common as tc


def contrast_scen(pooled):
    """{scenario: attempted-accidental} for scenarios having both cells."""
    out = {}
    for s, conds in pooled.items():
        if "attempted" in conds and "accidental" in conds:
            out[s] = conds["attempted"] - conds["accidental"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--out", default=os.path.join(tc.ROOT, "outputs", "analysis"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    registry = tc.load_registry()

    # model -> info (only models with a real size + base/instruct label)
    info = {}
    for study, tag, path in tc.iter_item_means():
        size, mtype, fam = tc.parse_tag(tag, registry)
        if math.isnan(size):
            continue
        pooled = tc.pooled_cells(tc.load_cells(path))
        cs = contrast_scen(pooled)
        if not cs:
            continue
        info[(fam, size, mtype)] = dict(
            tag=tc.pretty(tag), study=study, fam=fam, size=size, mtype=mtype,
            contrast=float(np.mean(list(cs.values()))), cs=cs)

    # build matched base/instruct pairs at same (family, size)
    pairs = []
    fams_sizes = {(f, s) for (f, s, _) in info}
    for (f, s) in sorted(fams_sizes):
        b = info.get((f, s, "base"))
        i = info.get((f, s, "instruct"))
        if b and i:
            pairs.append((f, s, b, i, i["contrast"] - b["contrast"]))

    if not pairs:
        print("No matched base/instruct pairs found.")
        return

    # ---- write pairs csv ----
    out_csv = os.path.join(a.out, "base_vs_instruct_pairs.csv")
    with open(out_csv, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["family", "size_B", "base_model", "instruct_model",
                    "contrast_base", "contrast_instruct", "delta_instruct_minus_base"])
        for f, s, b, i, d in pairs:
            w.writerow([f, s, b["tag"], i["tag"], round(b["contrast"], 4),
                        round(i["contrast"], 4), round(d, 4)])

    # ---- console: matched pairs ----
    print("\n=== BASE vs INSTRUCT — matched pairs (same family+size) ===")
    print(f"{'family':10} {'size':>6} {'base':>9} {'instruct':>9} {'delta (I-B)':>12}")
    for f, s, b, i, d in pairs:
        print(f"{f[:10]:10} {s:6.1f} {b['contrast']:+9.3f} {i['contrast']:+9.3f} {d:+12.3f}")

    def paired_report(subset, label):
        if len(subset) < 2:
            print(f"\n[{label}] only {len(subset)} pair(s) — need >=2 for a paired test.")
            return
        base = np.array([b["contrast"] for _, _, b, _, _ in subset])
        inst = np.array([i["contrast"] for _, _, _, i, _ in subset])
        deltas = inst - base
        t_stat, t_p = stats.ttest_rel(inst, base)
        try:
            w_stat, w_p = stats.wilcoxon(inst, base)
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        # pooled scenario-level paired bootstrap across all pairs in the subset
        stacked = []  # per-scenario (instruct - base) contrast deltas, all pairs
        for _, _, b, i, _ in subset:
            common = set(b["cs"]) & set(i["cs"])
            for k in common:
                stacked.append(i["cs"][k] - b["cs"][k])
        stacked = np.array(stacked)
        rng = np.random.default_rng(0)
        boot = np.array([rng.choice(stacked, stacked.size, replace=True).mean()
                         for _ in range(a.boot)]) if stacked.size else np.array([np.nan])
        print(f"\n[{label}]  n_pairs={len(subset)}")
        print(f"  mean delta (instruct - base) = {deltas.mean():+.3f}  "
              f"(all {'negative' if (deltas < 0).all() else 'mixed'})")
        print(f"  paired t-test:        t={t_stat:+.3f}  p={t_p:.4g}")
        print(f"  Wilcoxon signed-rank: W={w_stat}  p={w_p:.4g}")
        print(f"  pooled scenario-level delta = {np.nanmean(boot):+.3f}  "
              f"95% CI [{np.nanpercentile(boot,2.5):+.3f}, {np.nanpercentile(boot,97.5):+.3f}]")

    qwen_pairs = [p for p in pairs if "qwen" in p[0].lower()]
    paired_report(qwen_pairs, "Qwen2.5 ladder")
    paired_report(pairs, "ALL matched pairs")

    print(f"\nwrote {os.path.relpath(out_csv, tc.ROOT)}")
    print("Interpretation: a reliably NEGATIVE delta = instruction-tuning makes models "
          "MORE outcome-biased (less adult-like) at matched size.")


if __name__ == "__main__":
    main()
