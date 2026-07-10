#!/usr/bin/env python3
"""
05_human_comparison.py  --  Compare model judgments to HUMAN ground truth.

Inputs
  - outputs/behavior/item_means_<model>.csv  (from 03_behavioral.py)
  - dataset/human_reference/human_reference.csv  (you fill from published papers)

What it computes, per model:
  1. Model's condition profile: mean normalized blame for the 4 cells
       neutral, accidental, attempted, intentional   (0 = none, 1 = max blame)
  2. Model's intent-reliance index (from the 2x2 cells).
  3. For each human group (adult, child age-bands): correlation + RMSE between the
     model profile and the human profile, and the difference in intent-reliance.
  4. "Developmental placement": which human group the model's intent-reliance is
     closest to  (the "judges like a 5-year-old vs like an adult" headline).

Human reference groups expected (see dataset/human_reference/README):
  adult            (Young, Cushman, Hauser & Saxe 2007, PNAS)
  child_4_5, child_6_7, child_8plus  (Cushman, Sheketoff, Wharton & Carey 2013, Cognition)

Run after 03. Produces outputs/human/model_vs_human_<model>.csv (+ a profile plot).
"""
import os, csv, glob, argparse
from collections import defaultdict

CELLS = ["neutral", "accidental", "attempted", "intentional"]

def model_profile(item_means_csv, template):
    by_cond = defaultdict(list)
    items = []
    for r in csv.DictReader(open(item_means_csv)):
        if r["template"] != template: continue
        by_cond[r["condition"]].append(float(r["mean_norm_blame"]))
        items.append(r)
    prof = {c: (sum(v)/len(v) if v else None) for c, v in by_cond.items()}
    # intent-reliance from cell means: intent effect vs outcome effect
    def mean(c): return prof.get(c)
    have = all(prof.get(c) is not None for c in CELLS)
    idx = None
    if have:
        intent_eff = ((prof["attempted"]+prof["intentional"]) -
                      (prof["neutral"]+prof["accidental"]))/2
        outcome_eff = ((prof["accidental"]+prof["intentional"]) -
                       (prof["neutral"]+prof["attempted"]))/2
        idx = abs(intent_eff)/(abs(intent_eff)+abs(outcome_eff)+1e-9)
    return prof, idx

def all_template_profiles(item_means_csv):
    """Return {template: {condition: mean_norm_blame}} across every template present,
    so prompt-sensitivity can be reported instead of hidden behind one template."""
    by = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(item_means_csv)):
        by[r["template"]][r["condition"]].append(float(r["mean_norm_blame"]))
    out = {}
    for t, cond in by.items():
        out[t] = {c: (sum(v)/len(v) if v else None) for c, v in cond.items()}
    return out

def load_human(path):
    groups = defaultdict(dict)
    if not os.path.exists(path): return groups
    for r in csv.DictReader(open(path)):
        if not r.get("norm_blame","").strip(): continue
        groups[r["group"]][r["condition"]] = float(r["norm_blame"])
    return groups

def human_index(profile):
    if not all(c in profile for c in CELLS): return None
    intent_eff = ((profile["attempted"]+profile["intentional"]) -
                  (profile["neutral"]+profile["accidental"]))/2
    outcome_eff = ((profile["accidental"]+profile["intentional"]) -
                   (profile["neutral"]+profile["attempted"]))/2
    return abs(intent_eff)/(abs(intent_eff)+abs(outcome_eff)+1e-9)

def intent_outcome_contrast(profile):
    """attempted (guilty/no-harm) minus accidental (innocent/harm).
    The canonical single-number intent-vs-outcome index. Works even when only
    these two conditions exist (e.g. children in Cushman 2013).
      > 0  intent-weighted (adult-like):   blames bad intent more than bad outcome
      < 0  outcome-weighted (young-child): blames bad outcome more than bad intent
    """
    if profile.get("attempted") is None or profile.get("accidental") is None:
        return None
    return profile["attempted"] - profile["accidental"]

def main():
    here = os.path.dirname(__file__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavior", default=os.path.join(here,"..","outputs","behavior"))
    ap.add_argument("--human", default=os.path.join(here,"..","dataset","human_reference","human_reference.csv"))
    ap.add_argument("--out", default=os.path.join(here,"..","outputs","human"))
    ap.add_argument("--template", default="human_verbatim")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    human = load_human(a.human)
    human_idx = {g: human_index(p) for g, p in human.items()}
    human_contrast = {g: intent_outcome_contrast(p) for g, p in human.items()}

    for f in sorted(glob.glob(os.path.join(a.behavior, "item_means_*.csv"))):
        tag = os.path.basename(f)[len("item_means_"):-4]
        prof, midx = model_profile(f, a.template)
        mcontrast = intent_outcome_contrast(prof)
        # prompt-invariance: the same contrast computed on EVERY template
        tpl_profiles = all_template_profiles(f)
        tpl_contrasts = {t: intent_outcome_contrast(p) for t, p in tpl_profiles.items()}
        tpl_contrasts = {t: c for t, c in tpl_contrasts.items() if c is not None}
        import statistics as _st
        cvals = list(tpl_contrasts.values())
        contrast_mean = sum(cvals)/len(cvals) if cvals else None
        contrast_sd = _st.pstdev(cvals) if len(cvals) > 1 else 0.0
        outp = os.path.join(a.out, f"model_vs_human_{tag}.csv")
        with open(outp, "w", newline="") as g:
            w = csv.writer(g)
            w.writerow(["model", tag]); w.writerow(["template", a.template])
            w.writerow([]); w.writerow(["MODEL condition profile (0-1 blame)"])
            for c in CELLS: w.writerow([c, round(prof[c],4) if prof.get(c) is not None else "NA"])
            w.writerow(["model_intent_reliance_4cell", round(midx,4) if midx is not None else "NA"])
            w.writerow(["model_intent_vs_outcome_contrast", round(mcontrast,4) if mcontrast is not None else "NA"])
            # ---- prompt-invariance of the contrast (across all templates) ----
            w.writerow([])
            w.writerow([f"PROMPT-INVARIANCE of intent_vs_outcome_contrast (n_templates={len(cvals)})"])
            if contrast_mean is not None:
                w.writerow(["contrast_template_mean", round(contrast_mean,4)])
                w.writerow(["contrast_template_sd", round(contrast_sd,4)])
                w.writerow(["contrast_template_min", round(min(cvals),4)])
                w.writerow(["contrast_template_max", round(max(cvals),4)])
                w.writerow(["contrast_template_range", round(max(cvals)-min(cvals),4)])
                for t in sorted(tpl_contrasts):
                    w.writerow([f"  contrast[{t}]", round(tpl_contrasts[t],4)])
                # crude stability flag: does the sign (intent- vs outcome-weighted) flip?
                flips = (min(cvals) < 0 < max(cvals))
                w.writerow(["contrast_sign_stable_across_prompts", "NO" if flips else "yes"])
            if not human:
                w.writerow([]); w.writerow(["NOTE","no human_reference.csv filled yet — see dataset/human_reference/README"])
            for grp, gp in human.items():
                w.writerow([]); w.writerow([f"vs {grp}"])
                # rmse + corr over shared cells
                shared = [c for c in CELLS if c in gp and prof.get(c) is not None]
                if len(shared) >= 2:
                    import math
                    rmse = math.sqrt(sum((prof[c]-gp[c])**2 for c in shared)/len(shared))
                    mx = sum(prof[c] for c in shared)/len(shared); hx = sum(gp[c] for c in shared)/len(shared)
                    cov = sum((prof[c]-mx)*(gp[c]-hx) for c in shared)
                    sm = math.sqrt(sum((prof[c]-mx)**2 for c in shared)); sh = math.sqrt(sum((gp[c]-hx)**2 for c in shared))
                    corr = cov/(sm*sh) if sm*sh > 0 else float("nan")
                    w.writerow(["rmse", round(rmse,4)]); w.writerow(["profile_corr", round(corr,4)])
                if human_idx.get(grp) is not None and midx is not None:
                    w.writerow(["intent_reliance_4cell_diff", round(midx-human_idx[grp],4)])
                if human_contrast.get(grp) is not None and mcontrast is not None:
                    w.writerow(["intent_vs_outcome_contrast_human", round(human_contrast[grp],4)])
                    w.writerow(["contrast_diff_model_minus_human", round(mcontrast-human_contrast[grp],4)])
        # developmental placement -- use the attempted-vs-accidental contrast, since it
        # is comparable even when humans (children) only have those two conditions.
        valid_c = {g:c for g,c in human_contrast.items() if c is not None}
        if valid_c and mcontrast is not None:
            nearest = min(valid_c, key=lambda g: abs(valid_c[g]-mcontrast))
            extra = ""
            if contrast_mean is not None and len(cvals) > 1:
                near_mean = min(valid_c, key=lambda g: abs(valid_c[g]-contrast_mean))
                flip = "  [SIGN FLIPS across prompts!]" if (min(cvals) < 0 < max(cvals)) else ""
                extra = (f"  |  template-avg={contrast_mean:+.2f}"
                         f" (sd={contrast_sd:.2f}, range {min(cvals):+.2f}..{max(cvals):+.2f})"
                         f" -> {near_mean}{flip}")
            print(f"{tag}[{a.template}]: contrast={mcontrast:+.2f} "
                  f"-> closest human group: {nearest} ({valid_c[nearest]:+.2f}){extra}")
        else:
            print(f"{tag}: model contrast={mcontrast if mcontrast is None else round(mcontrast,2)} "
                  f"(fill human_reference.csv to compare)")

        # optional plot
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            plt.figure(figsize=(6,4))
            x = range(len(CELLS))
            plt.plot(x, [prof.get(c) for c in CELLS], "o-", label=tag)
            for grp, gp in human.items():
                plt.plot(x, [gp.get(c) for c in CELLS], "s--", label=grp)
            plt.xticks(list(x), CELLS, rotation=20); plt.ylabel("normalized blame (0-1)")
            plt.title(f"{tag} vs humans"); plt.legend(fontsize=7); plt.tight_layout()
            plt.savefig(os.path.join(a.out, f"profile_{tag}.png"), dpi=150)
        except Exception as e:
            print("  plot skipped:", e)

if __name__ == "__main__":
    main()
