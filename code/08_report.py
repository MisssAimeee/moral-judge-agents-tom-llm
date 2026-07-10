#!/usr/bin/env python3
"""
08_report.py  --  One summary table to hand to your mentor.

Pulls together everything 03/05/06 computed into a single table (CSV + Markdown):
each model's moral-judgment contrast and CI, how close it is to the ADULT human
profile (alignment), where it lands on the child→adult developmental ladder, its
prompt-stability, and instruction-tuning status. Human reference rows are included
at the top so the comparison is self-contained.

Columns
  model, params_B, type        what was run
  contrast [95% CI], sig≠0      intent-vs-outcome (attempted − accidental); the headline
  adult_align_corr             profile correlation with adult human (1 = identical shape)
  adult_align_rmse             RMSE to adult profile over shared cells (0 = identical)
  gap_vs_adult                 |contrast − adult contrast|  (0 = adult-like)
  nearest_human                developmental placement (adult / age 8+ / 6–7 / 4–5)
  intent_reliance [CI]         4-cell regression index
  prompt_sd / sign_flip        prompt-invariance (low SD & no flip = stable result)

Run after 06_stats.py. Writes outputs/report/summary_table.csv and .md.
"""
import os, csv, glob, argparse, math
from collections import defaultdict

CELLS = ["neutral", "accidental", "attempted", "intentional"]

def fnum(x):
    try: return float(x)
    except (TypeError, ValueError): return float("nan")

def short(tag):
    return tag.replace("Qwen_Qwen2.5-", "Qwen").replace("meta-llama_", "")

def pooled_profile(item_means_csv):
    acc = defaultdict(list)
    for r in csv.DictReader(open(item_means_csv)):
        acc[r["condition"]].append(float(r["mean_norm_blame"]))
    return {c: (sum(v)/len(v) if v else None) for c, v in acc.items()}

def adult_alignment(prof, adult):
    shared = [c for c in CELLS if prof.get(c) is not None and c in adult]
    if len(shared) < 2: return None, None
    rmse = math.sqrt(sum((prof[c]-adult[c])**2 for c in shared)/len(shared))
    mx = sum(prof[c] for c in shared)/len(shared)
    hx = sum(adult[c] for c in shared)/len(shared)
    cov = sum((prof[c]-mx)*(adult[c]-hx) for c in shared)
    sm = math.sqrt(sum((prof[c]-mx)**2 for c in shared))
    sh = math.sqrt(sum((adult[c]-hx)**2 for c in shared))
    corr = cov/(sm*sh) if sm*sh > 0 else float("nan")
    return corr, rmse

def load_human(human_csv):
    grp = defaultdict(dict)
    if os.path.exists(human_csv):
        for r in csv.DictReader(open(human_csv)):
            if r.get("norm_blame", "").strip():
                grp[r["group"]][r["condition"]] = float(r["norm_blame"])
    ladder = {g: (p["attempted"]-p["accidental"])
              for g, p in grp.items() if "attempted" in p and "accidental" in p}
    return grp, ladder

def main():
    here = os.path.dirname(__file__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavior", default=os.path.join(here, "..", "outputs", "behavior"))
    ap.add_argument("--stats", default=os.path.join(here, "..", "outputs", "stats"))
    ap.add_argument("--human", default=os.path.join(here, "..", "dataset",
                                                    "human_reference", "human_reference.csv"))
    ap.add_argument("--out", default=os.path.join(here, "..", "outputs", "report"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    grp, ladder = load_human(a.human)
    adult = grp.get("adult", {})
    adult_contrast = ladder.get("adult")

    cby = {}
    cpath = os.path.join(a.stats, "contrast_by_model.csv")
    if os.path.exists(cpath):
        for r in csv.DictReader(open(cpath)):
            cby[r["model"]] = r

    # count templates / scenarios for the "what was run" note
    def counts(item_means_csv):
        tmpls, stories = set(), set()
        for r in csv.DictReader(open(item_means_csv)):
            tmpls.add(r["template"]); stories.add(r["story_id"])
        return len(tmpls), len(stories)

    table = []
    for f in sorted(glob.glob(os.path.join(a.behavior, "item_means_*.csv"))):
        tag = os.path.basename(f)[len("item_means_"):-4]
        prof = pooled_profile(f)
        corr, rmse = adult_alignment(prof, adult)
        nt, ns = counts(f)
        c = cby.get(tag, {})
        contrast = fnum(c.get("contrast"))
        gap = abs(contrast-adult_contrast) if (adult_contrast is not None
                                               and not math.isnan(contrast)) else None
        table.append(dict(
            model=short(tag), params_B=c.get("size_B", "?"), type=c.get("type", "?"),
            contrast=contrast, lo=fnum(c.get("ci_lo")), hi=fnum(c.get("ci_hi")),
            sig=c.get("sig_vs_0", "?"),
            corr=corr, rmse=rmse, gap=gap,
            nearest=c.get("nearest_human_group", "?"),
            ir=fnum(c.get("intent_reliance")), ir_lo=fnum(c.get("ir_lo")),
            ir_hi=fnum(c.get("ir_hi")),
            psd=fnum(c.get("contrast_sd_across_templates")),
            flip=str(c.get("sign_flips_across_prompts")).lower() in ("true", "yes", "1"),
            n_templates=nt, n_stories=ns))
    table.sort(key=lambda d: (-d["contrast"] if not math.isnan(d["contrast"]) else 1e9))

    # ---- CSV ----
    cols = ["model", "params_B", "type", "n_stories", "n_templates",
            "contrast", "contrast_CI", "sig_diff_from_0",
            "adult_align_corr", "adult_align_rmse", "gap_vs_adult",
            "nearest_human_group", "intent_reliance", "intent_reliance_CI",
            "prompt_sd", "sign_flips"]
    def ci(lo, hi):
        return f"[{lo:+.2f},{hi:+.2f}]" if not (math.isnan(lo) or math.isnan(hi)) else "NA"
    def f2(x): return "NA" if x is None or (isinstance(x, float) and math.isnan(x)) else round(x, 3)

    with open(os.path.join(a.out, "summary_table.csv"), "w", newline="") as g:
        w = csv.writer(g); w.writerow(cols)
        # human reference rows first
        for grp_name in ("adult", "child_8plus", "child_6_7", "child_4_5"):
            if grp_name in ladder:
                w.writerow([f"HUMAN {grp_name}", "—", "human", "—", "—",
                            round(ladder[grp_name], 3), "—", "—",
                            "1.0" if grp_name == "adult" else "—", "—",
                            0.0 if grp_name == "adult" else round(abs(ladder[grp_name]-adult_contrast), 3)
                            if adult_contrast is not None else "—",
                            grp_name, "—", "—", "—", "—"])
        for d in table:
            w.writerow([d["model"], d["params_B"], d["type"], d["n_stories"], d["n_templates"],
                        f2(d["contrast"]), ci(d["lo"], d["hi"]), d["sig"],
                        f2(d["corr"]), f2(d["rmse"]), f2(d["gap"]), d["nearest"],
                        f2(d["ir"]), ci(d["ir_lo"], d["ir_hi"]),
                        f2(d["psd"]), "YES" if d["flip"] else "no"])

    # ---- Markdown (paste into an email / doc) ----
    md = ["# Moral judgment: model vs human — summary\n",
          f"Adult human reference contrast = **{adult_contrast:+.2f}** "
          f"(intent-weighted). Developmental ladder: " +
          ", ".join(f"{g} {c:+.2f}" for g, c in sorted(ladder.items(), key=lambda x:-x[1])) + ".\n",
          "Contrast = blame(attempted) − blame(accidental). "
          "Positive = judges by **intent** (adult-like); negative = by **outcome** (young-child-like). "
          "CI = 95% bootstrap over scenarios.\n",
          "| model | params | type | contrast [95% CI] | ≠0 | adult corr | adult RMSE | gap vs adult | nearest human | intent-reliance | prompt SD | sign flip |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for grp_name in ("adult", "child_8plus", "child_6_7", "child_4_5"):
        if grp_name in ladder:
            md.append(f"| **HUMAN {grp_name}** | — | human | {ladder[grp_name]:+.2f} | — | "
                      f"{'1.00' if grp_name=='adult' else '—'} | — | "
                      f"{'0.00' if grp_name=='adult' else f'{abs(ladder[grp_name]-adult_contrast):.2f}'} | "
                      f"{grp_name} | — | — | — |")
    for d in table:
        md.append(f"| {d['model']} | {d['params_B']} | {d['type']} | "
                  f"{d['contrast']:+.2f} {ci(d['lo'],d['hi'])} | {d['sig']} | "
                  f"{f2(d['corr'])} | {f2(d['rmse'])} | {f2(d['gap'])} | {d['nearest']} | "
                  f"{d['ir']:.2f} {ci(d['ir_lo'],d['ir_hi'])} | {f2(d['psd'])} | "
                  f"{'YES' if d['flip'] else 'no'} |")
    md += ["\n**Figures** (outputs/figures/): contrast_forest, profiles, prompt_invariance, "
           "size_vs_contrast, weights_scatter, pairwise_heatmap.\n"]
    open(os.path.join(a.out, "summary_table.md"), "w").write("\n".join(md))

    print("\n".join(md))
    print(f"\nWrote {a.out}/summary_table.csv and summary_table.md")

if __name__ == "__main__":
    main()
