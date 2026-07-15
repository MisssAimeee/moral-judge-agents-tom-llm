#!/usr/bin/env python3
"""
06_stats.py  --  Inferential statistics on the behavioral results.

03_behavioral.py + 05_human_comparison.py give POINT estimates. This script answers
"is the effect real / are two models actually different / is it prompt-stable?" with
confidence intervals and tests, so a result isn't reported as a bare number.

Everything is bootstrapped over SCENARIOS (the shared story background), not over
samples or items, because the relevant variability for "does model A weight intent
more than model B" is between scenarios. The 4 cells of one scenario are not
independent, so they resample together.

What it produces (console + outputs/stats/):
  1. Per model, template-POOLED intent-vs-outcome contrast (attempted - accidental)
     with a 95% bootstrap CI, and whether it differs from 0 (outcome- vs intent-
     weighted) and from the adult human value (+0.67).
  2. Intent-reliance index (4-cell regression) with a bootstrap CI per model.
  3. PROMPT-INVARIANCE: the contrast per template + SD/range, and (if statsmodels is
     available) a cluster-robust test of the intent x template interaction.
  4. CROSS-MODEL ladder: models sorted by contrast with CIs, plus paired-bootstrap
     differences between models (e.g. is 1.5B really != 3B, or just noise at n=8?).
  5. BASE vs INSTRUCT: contrast grouped by model type, and matched within-size
     base-vs-instruct differences, to separate instruction-tuning from size.
  6. (optional) statsmodels mixed model: norm ~ intent*outcome with a scenario random
     intercept, giving p-values on the intent and outcome main effects.

Run after 03 (needs outputs/behavior/item_means_*.csv). Uses numpy only; statsmodels
is optional (extra p-values when installed).
"""
import os, csv, glob, re, argparse, math
from collections import defaultdict
import numpy as np

CELLS = ["neutral", "accidental", "attempted", "intentional"]
# condition -> (intent present, outcome present)
COND_MAP = {"neutral": (0, 0), "accidental": (0, 1),
            "attempted": (1, 0), "intentional": (1, 1)}

# ---------------------------------------------------------------- loading ----
def scenario_of(story_id):
    """Drop the trailing -<condition> so the 4 cells of a story share a key."""
    return story_id.rsplit("-", 1)[0]

def load_model(item_means_csv):
    """-> cells[template][scenario][condition] = mean_norm_blame"""
    cells = defaultdict(lambda: defaultdict(dict))
    for r in csv.DictReader(open(item_means_csv)):
        cells[r["template"]][scenario_of(r["story_id"])][r["condition"]] = \
            float(r["mean_norm_blame"])
    return cells

def pooled_cells(cells):
    """Average each (scenario, condition) over templates -> {scenario:{cond:val}}."""
    acc = defaultdict(lambda: defaultdict(list))
    for tmpl, scen in cells.items():
        for s, conds in scen.items():
            for c, v in conds.items():
                acc[s][c].append(v)
    return {s: {c: sum(vs)/len(vs) for c, vs in conds.items()}
            for s, conds in acc.items()}

def load_human_adult(path):
    prof = {}
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            if r.get("group") == "adult" and r.get("norm_blame", "").strip():
                prof[r["condition"]] = float(r["norm_blame"])
    if "attempted" in prof and "accidental" in prof:
        return prof["attempted"] - prof["accidental"]
    return None

def load_registry(path):
    """tag -> {params_B, class, provider, display, ...} from model_registry.csv.
    Lets closed API models (no size in the name) still get size/type/provider."""
    reg = {}
    if path and os.path.exists(path):
        for r in csv.DictReader(open(path)):
            reg[r["tag"]] = r
    return reg

def parse_tag(tag, registry=None):
    """'Qwen_Qwen2.5-1.5B-Instruct' -> (size_float, 'instruct'/'base', family, provider).
    Falls back to the model registry for API models whose size isn't in the name."""
    reg = (registry or {}).get(tag)
    if reg:
        try:
            size = float(reg.get("params_B", "") or "nan")
        except ValueError:
            size = float("nan")
        cls = (reg.get("class") or "").lower()
        mtype = "instruct" if cls in ("instruct", "chat", "reasoning") else \
                ("base" if cls == "base" else "instruct")
        provider = reg.get("provider", "")
        return size, mtype, provider, provider
    m = re.search(r"(\d+\.?\d*)\s*[bB]\b", tag) or re.search(r"(\d+\.?\d*)[bB]", tag)
    size = float(m.group(1)) if m else float("nan")
    mtype = "instruct" if re.search(r"instruct|chat|-it\b", tag, re.I) else "base"
    family = re.split(r"[-_]?\d+\.?\d*[bB]", tag)[0].strip("_-")
    return size, mtype, family, ""

# ------------------------------------------------------------- bootstrap ----
def bootstrap(keys, statfn, B=2000, seed=0, alpha=0.05):
    """Resample `keys` (scenarios) with replacement; statfn(keys)->scalar or vector."""
    rng = np.random.default_rng(seed)
    keys = list(keys); n = len(keys)
    point = np.atleast_1d(np.asarray(statfn(keys), float))
    mat = np.empty((B, point.size))
    for b in range(B):
        samp = [keys[i] for i in rng.integers(0, n, n)]
        mat[b] = np.atleast_1d(np.asarray(statfn(samp), float))
    lo = np.percentile(mat, 100*alpha/2, axis=0)
    hi = np.percentile(mat, 100*(1-alpha/2), axis=0)
    return point, lo, hi, mat

def contrast_from(scen_diff):
    """statfn: mean paired (attempted-accidental) over resampled scenarios."""
    def f(keys):
        vals = [scen_diff[k] for k in keys if k in scen_diff]
        return float(np.mean(vals)) if vals else float("nan")
    return f

def scen_paired_diff(cells_scen):
    """{scenario: attempted-accidental} for scenarios having both cells."""
    out = {}
    for s, conds in cells_scen.items():
        if "attempted" in conds and "accidental" in conds:
            out[s] = conds["attempted"] - conds["accidental"]
    return out

def intent_reliance_from(cells_scen):
    """statfn: 4-cell regression intent-reliance index on resampled scenarios."""
    def f(keys):
        X, y = [], []
        for k in keys:
            conds = cells_scen.get(k, {})
            for c, (i_, o_) in COND_MAP.items():
                if c in conds:
                    X.append([1.0, i_, o_]); y.append(conds[c])
        if len(X) < 4:
            return float("nan")
        beta, *_ = np.linalg.lstsq(np.array(X), np.array(y), rcond=None)
        _, bi, bo = beta
        return abs(bi) / (abs(bi) + abs(bo) + 1e-9)
    return f

def weights_point(cells_scen):
    X, y = [], []
    for conds in cells_scen.values():
        for c, (i_, o_) in COND_MAP.items():
            if c in conds:
                X.append([1.0, i_, o_]); y.append(conds[c])
    if len(X) < 4:
        return None, None
    beta, *_ = np.linalg.lstsq(np.array(X), np.array(y), rcond=None)
    return float(beta[1]), float(beta[2])

def nearest_group(contrast, human):
    if contrast is None or not human:
        return "NA"
    return min(human, key=lambda g: abs(human[g] - contrast))

# ------------------------------------------------------------------ main ----
def main():
    here = os.path.dirname(__file__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavior", default=os.path.join(here, "..", "outputs", "behavior"))
    ap.add_argument("--human", default=os.path.join(here, "..", "dataset",
                                                     "human_reference", "human_reference.csv"))
    ap.add_argument("--out", default=os.path.join(here, "..", "outputs", "stats"))
    ap.add_argument("--registry", default=os.path.join(here, "..", "dataset",
                                                       "model_registry.csv"),
                    help="model_registry.csv (params/type/provider for API models)")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    registry = load_registry(a.registry)

    adult_contrast = load_human_adult(a.human)
    # full human ladder (contrast per group) for placement
    human_ladder = {}
    if os.path.exists(a.human):
        grp = defaultdict(dict)
        for r in csv.DictReader(open(a.human)):
            if r.get("norm_blame", "").strip():
                grp[r["group"]][r["condition"]] = float(r["norm_blame"])
        for g, p in grp.items():
            if "attempted" in p and "accidental" in p:
                human_ladder[g] = p["attempted"] - p["accidental"]

    files = sorted(glob.glob(os.path.join(a.behavior, "item_means_*.csv")))
    if not files:
        print("No item_means_*.csv in", a.behavior, "- run 03_behavioral.py first.")
        return

    summary = []         # per-model pooled results
    scen_diff_pooled = {} # model -> {scenario: contrast} for cross-model pairing
    per_template = {}     # model -> {template: contrast}

    for f in files:
        tag = os.path.basename(f)[len("item_means_"):-4]
        cells = load_model(f)
        pooled = pooled_cells(cells)

        # degeneracy QC: near-constant ratings (zero variance) -> no usable signal.
        # Same 0.02 threshold as the checkpoint-dissection entropy filter, so a
        # degenerate model's 0.000 contrast is never read as a real null.
        all_vals = [v for conds in pooled.values() for v in conds.values()]
        rating_std = float(np.std(all_vals)) if all_vals else 0.0
        degenerate = rating_std < 0.02

        # pooled contrast + CI
        sdiff = scen_paired_diff(pooled)
        scen_diff_pooled[tag] = sdiff
        if sdiff:
            pt, lo, hi, _ = bootstrap(list(sdiff), contrast_from(sdiff), B=a.boot)
            c_pt, c_lo, c_hi = float(pt[0]), float(lo[0]), float(hi[0])
        else:
            c_pt = c_lo = c_hi = float("nan")

        # intent-reliance index + CI
        if len(pooled) >= 2:
            ir_pt, ir_lo, ir_hi, _ = bootstrap(list(pooled), intent_reliance_from(pooled), B=a.boot)
            ir_pt, ir_lo, ir_hi = float(ir_pt[0]), float(ir_lo[0]), float(ir_hi[0])
        else:
            ir_pt = ir_lo = ir_hi = float("nan")
        b_int, b_out = weights_point(pooled)

        # per-template contrasts (prompt invariance)
        tcs = {}
        for tmpl, scen in cells.items():
            d = scen_paired_diff(scen)
            if d:
                tcs[tmpl] = float(np.mean(list(d.values())))
        per_template[tag] = tcs
        tvals = list(tcs.values())
        t_sd = float(np.std(tvals)) if len(tvals) > 1 else 0.0
        t_range = (max(tvals) - min(tvals)) if tvals else 0.0
        sign_flip = bool(tvals) and (min(tvals) < 0 < max(tvals))

        size, mtype, family, provider = parse_tag(tag, registry)
        sig0 = "yes" if (not math.isnan(c_lo) and (c_lo > 0 or c_hi < 0)) else "no"
        vs_adult = (c_pt - adult_contrast) if (adult_contrast is not None
                                               and not math.isnan(c_pt)) else None
        summary.append(dict(tag=tag, size=size, mtype=mtype, family=family,
                            provider=provider,
                            contrast=c_pt, lo=c_lo, hi=c_hi, sig_vs0=sig0,
                            ir=ir_pt, ir_lo=ir_lo, ir_hi=ir_hi, b_int=b_int, b_out=b_out,
                            t_sd=t_sd, t_range=t_range, sign_flip=sign_flip,
                            vs_adult=vs_adult, rating_std=rating_std, degenerate=degenerate,
                            nearest=nearest_group(c_pt, human_ladder)))

    summary.sort(key=lambda d: (-d["contrast"] if not math.isnan(d["contrast"]) else 1e9))

    # ----------------------------------------------------------- console ----
    print("\n=== INTENT-vs-OUTCOME CONTRAST (template-pooled, 95% bootstrap CI) ===")
    if adult_contrast is not None:
        print(f"(adult human reference = {adult_contrast:+.2f};  "
              f"human ladder: " + ", ".join(f"{g} {c:+.2f}" for g, c in
              sorted(human_ladder.items(), key=lambda x:-x[1])) + ")")
    print(f"{'model':38} {'contrast':>9} {'95% CI':>17} {'!=0':>4} {'nearest':>10} "
          f"{'pSD':>5} {'prompt':>7}")
    for d in summary:
        ci = f"[{d['lo']:+.2f},{d['hi']:+.2f}]"
        flip = "FLIP" if d["sign_flip"] else "ok"
        print(f"{d['tag']:38} {d['contrast']:+9.3f} {ci:>17} {d['sig_vs0']:>4} "
              f"{d['nearest']:>10} {d['t_sd']:5.2f} {flip:>7}")

    print("\n=== INTENT-RELIANCE INDEX (4-cell regression, 95% CI) ===")
    for d in summary:
        print(f"{d['tag']:38} idx={d['ir']:.3f} [{d['ir_lo']:.3f},{d['ir_hi']:.3f}]  "
              f"b_intent={d['b_int']:+.3f} b_outcome={d['b_out']:+.3f}")

    # ----------------------------------------------- pairwise differences ----
    print("\n=== PAIRWISE MODEL DIFFERENCES in contrast (paired bootstrap over shared scenarios) ===")
    pair_rows = []
    tags = [d["tag"] for d in summary]
    for i in range(len(tags)):
        for j in range(i+1, len(tags)):
            A, Bm = tags[i], tags[j]
            da, db = scen_diff_pooled[A], scen_diff_pooled[Bm]
            common = sorted(set(da) & set(db))
            if len(common) < 5:
                continue
            def diff(keys, da=da, db=db):
                a_ = np.mean([da[k] for k in keys]); b_ = np.mean([db[k] for k in keys])
                return a_ - b_
            pt, lo, hi, _ = bootstrap(common, diff, B=a.boot)
            sig = (lo[0] > 0 or hi[0] < 0)
            pair_rows.append((A, Bm, float(pt[0]), float(lo[0]), float(hi[0]), sig))
    # print only the most relevant (adjacent in the ladder) + significant ones
    for A, Bm, pt, lo, hi, sig in pair_rows:
        mark = "  *" if sig else ""
        print(f"  {A:34} - {Bm:34} d={pt:+.3f} [{lo:+.3f},{hi:+.3f}]{mark}")
    print("  (* = 95% CI excludes 0 -> models are distinguishable; otherwise within noise)")

    # ------------------------------------------------- base vs instruct ----
    print("\n=== BASE vs INSTRUCT ===")
    by_type = defaultdict(list)
    for d in summary:
        if not math.isnan(d["contrast"]):
            by_type[d["mtype"]].append(d["contrast"])
    for t, vals in by_type.items():
        print(f"  {t:9}: mean contrast {np.mean(vals):+.3f}  (n={len(vals)} models, "
              f"range {min(vals):+.2f}..{max(vals):+.2f})")
    # matched within-size base vs instruct
    by_size = defaultdict(dict)
    for d in summary:
        if not math.isnan(d["size"]):
            by_size[(d["family"], d["size"])][d["mtype"]] = d
    print("  matched pairs (same family+size):")
    matched = False
    for (fam, sz), bt in sorted(by_size.items()):
        if "base" in bt and "instruct" in bt:
            matched = True
            db = bt["base"]["contrast"]; di = bt["instruct"]["contrast"]
            print(f"    {fam} {sz}B: base {db:+.3f} vs instruct {di:+.3f}  "
                  f"(instruct - base = {di-db:+.3f})")
    if not matched:
        print("    (none yet — run both base and instruct at the same size to control "
              "the instruction-tuning confound)")

    # --------------------------------------------- optional mixed model ----
    run_statsmodels(files, a.out)

    # ----------------------------------------------------------- write ----
    with open(os.path.join(a.out, "contrast_by_model.csv"), "w", newline="") as g:
        w = csv.writer(g)
        w.writerow(["model", "size_B", "type", "provider", "contrast", "ci_lo", "ci_hi",
                    "sig_vs_0", "intent_reliance", "ir_lo", "ir_hi", "b_intent",
                    "b_outcome", "contrast_sd_across_templates", "contrast_range",
                    "sign_flips_across_prompts", "nearest_human_group",
                    "contrast_minus_adult", "rating_std", "degenerate"])
        for d in summary:
            w.writerow([d["tag"], d["size"], d["mtype"], d["provider"], r4(d["contrast"]), r4(d["lo"]),
                        r4(d["hi"]), d["sig_vs0"], r4(d["ir"]), r4(d["ir_lo"]),
                        r4(d["ir_hi"]), r4(d["b_int"]), r4(d["b_out"]), r4(d["t_sd"]),
                        r4(d["t_range"]), d["sign_flip"], d["nearest"],
                        r4(d["vs_adult"]) if d["vs_adult"] is not None else "NA",
                        r4(d["rating_std"]), d["degenerate"]])

    with open(os.path.join(a.out, "prompt_invariance_contrast.csv"), "w", newline="") as g:
        w = csv.writer(g)
        all_t = sorted({t for tc in per_template.values() for t in tc})
        w.writerow(["model"] + all_t + ["sd", "range", "sign_flips"])
        for d in summary:
            tc = per_template[d["tag"]]
            row = [d["tag"]] + [r4(tc.get(t)) if t in tc else "NA" for t in all_t]
            row += [r4(d["t_sd"]), r4(d["t_range"]), d["sign_flip"]]
            w.writerow(row)

    with open(os.path.join(a.out, "pairwise_model_diffs.csv"), "w", newline="") as g:
        w = csv.writer(g)
        w.writerow(["model_a", "model_b", "contrast_diff", "ci_lo", "ci_hi", "distinguishable"])
        for A, Bm, pt, lo, hi, sig in pair_rows:
            w.writerow([A, Bm, r4(pt), r4(lo), r4(hi), "yes" if sig else "no"])

    print(f"\nWrote CSVs to {a.out}/")

def r4(x):
    try:
        return round(float(x), 4)
    except (TypeError, ValueError):
        return "NA"

def run_statsmodels(files, outdir):
    """Optional: mixed model main effects + cluster-robust intent x template test."""
    try:
        import statsmodels.formula.api as smf
        import pandas as pd
    except Exception:
        print("\n(statsmodels/pandas not installed -> skipping mixed-model p-values; "
              "bootstrap CIs above are the primary inference.)")
        return
    print("\n=== MIXED MODEL  norm ~ intent*outcome  (scenario random intercept) ===")
    rows = []
    for f in files:
        tag = os.path.basename(f)[len("item_means_"):-4]
        for r in csv.DictReader(open(f)):
            cond = r["condition"]
            if cond not in COND_MAP:
                continue
            i_, o_ = COND_MAP[cond]
            rows.append(dict(model=tag, template=r["template"],
                             scenario=scenario_of(r["story_id"]),
                             intent=i_, outcome=o_, norm=float(r["mean_norm_blame"])))
    df = pd.DataFrame(rows)
    for tag, sub in df.groupby("model"):
        try:
            md = smf.mixedlm("norm ~ intent*outcome", sub, groups=sub["scenario"]).fit(reml=False)
            bi = md.params.get("intent", float("nan")); pi = md.pvalues.get("intent", float("nan"))
            bo = md.params.get("outcome", float("nan")); po = md.pvalues.get("outcome", float("nan"))
            print(f"  {tag:38} intent b={bi:+.3f} (p={pi:.1e})  "
                  f"outcome b={bo:+.3f} (p={po:.1e})")
        except Exception as e:
            print(f"  {tag}: mixedlm failed ({e})")
    # prompt sensitivity: intent x template interaction, cluster-robust on scenario
    print("  intent x template interaction (prompt sensitivity, cluster-robust SE):")
    for tag, sub in df.groupby("model"):
        if sub["template"].nunique() < 2:
            continue
        try:
            ols = smf.ols("norm ~ intent*outcome*C(template)", sub).fit(
                cov_type="cluster", cov_kwds={"groups": sub["scenario"]})
            inter = [p for p in ols.pvalues.index if "intent" in p and "template" in p
                     and "outcome" not in p]
            pmin = min((ols.pvalues[p] for p in inter), default=float("nan"))
            verdict = "PROMPT-SENSITIVE" if pmin < 0.05 else "prompt-stable"
            print(f"    {tag:38} min intent:template p={pmin:.1e} -> {verdict}")
        except Exception as e:
            print(f"    {tag}: ols failed ({e})")

if __name__ == "__main__":
    main()
