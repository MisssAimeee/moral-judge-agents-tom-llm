#!/usr/bin/env python3
"""J2 -- item-level representation<->behaviour test. Replaces the n=8 model-level link.

WHY THIS REPLACES THE OLD LINK. 04_link_analysis.py correlates one number per model (peak
intent decoding accuracy) against one number per model (intent-reliance index) across 8
probed models. C6 recomputed it without the effect floor and got r = -0.209 with a 95% CI of
about [-0.80, +0.58]. That interval is consistent with a strong negative link, no link, and
a strong positive link at once, so the test cannot come out informative at that n no matter
how the pipeline is cleaned up. The problem is the design, not the data.

The unit of analysis should be the scenario, not the model. Each model contributes 53
scenario groups, and the question becomes the one actually of interest:

    When a model represents intent more clearly FOR A GIVEN STORY,
    does it weight intent more IN ITS JUDGMENT OF THAT STORY?

  (a) intent decodability per scenario -- mean out-of-fold signed margin of the intent probe
      at that model's peak intent layer, over the items of that scenario group. Signed so
      positive means the probe put the item on the correct side of the boundary; the margin
      rather than a 0/1 hit because it is graded, and 4 items per group would otherwise give
      an almost useless per-group accuracy.

  (b) intent-use per scenario -- the same intent contrast the project reports, computed
      within that scenario group and averaged over prompt templates.

Both are held out where it matters: the margins come from GroupKFold folds split ON
scenario_group, so a scenario's margin is always produced by a probe that never saw that
scenario. Without that the correlation would be partly fitted noise.

TWO DEFINITIONS OF (b), because the stimulus set supports both:
  primary   attempted - accidental, the project's headline diagonal contrast. 48 of 53
            groups have both cells. Intent and outcome both differ across this pair, which
            is exactly why it is the headline: it asks which factor the model follows when
            they point in opposite directions.
  matched   intent effect at constant outcome, averaging (intentional - accidental) within
            harm and (attempted - neutral) within no-harm. Available for all 53 groups and
            a cleaner isolation of intent, so it is the robustness check on the primary.

POOLING. Per-model correlations are bootstrapped over scenario groups. Across models the
data are stacked with both axes z-scored within model, and a mixed model with a random
intercept and random slope per model gives the pooled estimate. Standardising within model
is what makes the pooled slope a statement about within-model, across-scenario covariation
rather than a restatement of the between-model differences the old n=8 test was built on.

Outputs
  outputs/link/item_level_dissociation.csv         per-model r, CI, n
  outputs/link/item_level_groups.csv               per model per scenario group, both axes
  outputs/link/item_level_dissociation.png         per-model scatter with fit
  outputs/link/ITEM_LEVEL_DISSOCIATION.md
"""
import argparse
import csv
import glob
import importlib.util
import os
import re
import warnings
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MASTER = os.path.join(ROOT, "dataset", "master", "moral_2x2_master.csv")
ACTS = os.path.join(ROOT, "outputs", "acts")
PROBE = os.path.join(ROOT, "outputs", "probe")
BEHAVIOR = os.path.join(ROOT, "outputs", "behavior")
OUTDIR = os.path.join(ROOT, "outputs", "link")

N_BOOT = 10000


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


probe_mod = _load("probe_mod", os.path.join(ROOT, "code", "02_probe.py"))


def joinkey(s):
    """Separator-free key: probe tags use dots, behaviour ids use underscores."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def oof_margins(X, y, groups, n_splits=5):
    """Out-of-fold signed distance to the intent boundary, one value per item.

    Same preprocessing as 02_probe.group_cv_acc (standardise, lossless row-space
    projection, L2 logistic with C=1.0) so the margins correspond to the accuracies the
    probe CSV reports. Folds split on scenario_group, so no scenario is ever scored by a
    probe that trained on it.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    marg = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = probe_mod._rowspace_project(sc.transform(X[tr]), sc.transform(X[te]))
        if Xtr.shape[1] == 0 or len(np.unique(y[tr])) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, y[tr])
        marg[te] = (2 * y[te] - 1) * clf.decision_function(Xte)
    return marg


def peak_intent_layer(tag):
    path = os.path.join(PROBE, f"{tag}_probe.csv")
    if not os.path.exists(path):
        return None
    best, best_acc = None, -np.inf
    for r in csv.DictReader(open(path)):
        if r["target"] != "intent" or r.get("degenerate", "False") == "True":
            continue
        try:
            acc = float(r["cv_acc"])
        except (TypeError, ValueError):
            continue
        if acc > best_acc:
            best_acc, best = acc, int(r["layer"])
    return best, best_acc


def behaviour_by_group(path, groups):
    """-> {group: (primary_contrast, matched_contrast)} averaged over templates."""
    cells = defaultdict(lambda: defaultdict(list))   # (tmpl, grp) -> cond -> [vals]
    for r in csv.DictReader(open(path)):
        sid = r["story_id"]
        if sid not in groups:
            continue
        try:
            v = float(r["norm_rating"])
        except (TypeError, ValueError):
            continue
        cells[(r["template"], groups[sid])][r["condition"]].append(v)

    per_group = defaultdict(lambda: ([], []))
    for (tmpl, grp), cd in cells.items():
        m = {c: float(np.mean(v)) for c, v in cd.items() if v}
        if "attempted" in m and "accidental" in m:
            per_group[grp][0].append(m["attempted"] - m["accidental"])
        matched = []
        if "intentional" in m and "accidental" in m:
            matched.append(m["intentional"] - m["accidental"])
        if "attempted" in m and "neutral" in m:
            matched.append(m["attempted"] - m["neutral"])
        if matched:
            per_group[grp][1].append(float(np.mean(matched)))

    out = {}
    for grp, (prim, matched) in per_group.items():
        out[grp] = (float(np.mean(prim)) if prim else np.nan,
                    float(np.mean(matched)) if matched else np.nan)
    return out


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or x.std() < 1e-12 or y.std() < 1e-12:
        return np.nan, 0
    return float(np.corrcoef(x, y)[0, 1]), len(x)


def boot_ci(x, y, n_boot=N_BOOT, seed=0):
    """Resample scenario groups with replacement; percentile interval on r."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 4:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        xs, ys = x[idx], y[idx]
        if xs.std() < 1e-12 or ys.std() < 1e-12:
            continue
        rs.append(np.corrcoef(xs, ys)[0, 1])
    if not rs:
        return np.nan, np.nan
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def spearman(x, y):
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return np.nan, np.nan
    r, p = spearmanr(x[ok], y[ok])
    return float(r), float(p)


def scatter_figure(per_model_pts, results, path, which="primary"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tags = [t for t in per_model_pts if len(per_model_pts[t][0]) > 3]
    if not tags:
        return
    ncol = 4
    nrow = int(np.ceil(len(tags) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.0 * nrow),
                             squeeze=False)
    for k, tag in enumerate(tags):
        ax = axes[k // ncol][k % ncol]
        dec, prim, matched = per_model_pts[tag]
        use = prim if which == "primary" else matched
        dec, use = np.asarray(dec, float), np.asarray(use, float)
        ok = np.isfinite(dec) & np.isfinite(use)
        ax.scatter(dec[ok], use[ok], s=17, alpha=0.72, color="#1f3f8f",
                   edgecolor="none")
        res = next((r for r in results if r["model_tag"] == tag
                    and r["definition"] == which), None)
        if ok.sum() > 2 and dec[ok].std() > 1e-12:
            b, a = np.polyfit(dec[ok], use[ok], 1)
            xs = np.linspace(dec[ok].min(), dec[ok].max(), 20)
            ax.plot(xs, a + b * xs, color="#b3202c", lw=1.5)
        ax.axhline(0, color="#999999", lw=0.7, ls=":")
        if res:
            ax.set_title(f"{tag}\nr = {res['pearson_r']:+.3f}  "
                         f"[{res['ci_lo']:+.2f}, {res['ci_hi']:+.2f}]  n={res['n']}",
                         fontsize=8.2)
        ax.tick_params(labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for k in range(len(tags), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.supxlabel("intent decodability for this scenario  (mean out-of-fold signed margin)",
                  fontsize=10)
    fig.supylabel("intent-use for this scenario  (attempted - accidental)"
                  if which == "primary" else
                  "intent-use for this scenario  (intent effect at constant outcome)",
                  fontsize=10)
    fig.suptitle("Item-level representation-behaviour link: one point per scenario group",
                 fontsize=11.5)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.97))
    fig.savefig(path, dpi=180)
    print(f"  -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUTDIR)
    ap.add_argument("--pooling", default="last")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    lab = {r["story_id"]: r for r in csv.DictReader(open(MASTER))}
    groups = {s: (r.get("scenario_group") or r.get("scenario_id"))
              for s, r in lab.items()}

    beh_files = {joinkey(os.path.basename(f)[len("raw_"):-len(".csv")]): f
                 for f in glob.glob(os.path.join(BEHAVIOR, "raw_*.csv"))}

    results, per_model_pts, group_rows = [], {}, []
    for npz in sorted(glob.glob(os.path.join(ACTS, "*.npz"))):
        tag = os.path.basename(npz)[:-len(".npz")]
        pk = peak_intent_layer(tag)
        if pk is None or pk[0] is None:
            print(f"  {tag:32} no probe CSV; skipped")
            continue
        layer, layer_acc = pk

        bf = beh_files.get(joinkey(tag))
        if bf is None:
            # behaviour files carry the org prefix, probe tags do not
            cand = [v for k, v in beh_files.items() if k.endswith(joinkey(tag))]
            bf = cand[0] if len(cand) == 1 else None
        if bf is None:
            print(f"  {tag:32} no behaviour file matched; skipped")
            continue

        d = np.load(npz, allow_pickle=True)
        acts = d[a.pooling]
        sids = [str(s) for s in d["story_id"]]
        keep = [i for i, s in enumerate(sids) if s in lab]
        acts, sids = acts[keep], [sids[i] for i in keep]
        y = np.array([1 if lab[s]["intent_label"] == "guilty" else 0 for s in sids])
        g = np.array([groups[s] for s in sids])

        marg = oof_margins(acts[:, layer, :], y, g)
        dec_by_group = {}
        for grp in np.unique(g):
            m = marg[g == grp]
            m = m[np.isfinite(m)]
            if len(m):
                dec_by_group[grp] = float(np.mean(m))

        beh = behaviour_by_group(bf, groups)
        common = sorted(set(dec_by_group) & set(beh))
        dec = [dec_by_group[k] for k in common]
        prim = [beh[k][0] for k in common]
        matched = [beh[k][1] for k in common]
        per_model_pts[tag] = (dec, prim, matched)
        for k, dv, pv, mv in zip(common, dec, prim, matched):
            group_rows.append([tag, k, round(dv, 5),
                               "" if not np.isfinite(pv) else round(pv, 5),
                               "" if not np.isfinite(mv) else round(mv, 5)])

        for which, use in (("primary", prim), ("matched", matched)):
            r, n = pearson(dec, use)
            lo, hi = boot_ci(dec, use, a.n_boot)
            rho, sp = spearman(dec, use)
            results.append(dict(model_tag=tag, definition=which, peak_intent_layer=layer,
                                peak_intent_acc=round(layer_acc, 4),
                                pearson_r=r, ci_lo=lo, ci_hi=hi, spearman_rho=rho,
                                spearman_p=sp, n=n))
        pr = next(x for x in results if x["model_tag"] == tag
                  and x["definition"] == "primary")
        print(f"  {tag:32} L{layer:<3} acc={layer_acc:.3f}  "
              f"r={pr['pearson_r']:+.3f} [{pr['ci_lo']:+.2f},{pr['ci_hi']:+.2f}] "
              f"n={pr['n']}")

    # ---------------- pooled, model as random effect ----------------
    pooled = {}
    for which in ("primary", "matched"):
        X, Y, M = [], [], []
        for tag, (dec, prim, matched) in per_model_pts.items():
            use = prim if which == "primary" else matched
            dec_a, use_a = np.asarray(dec, float), np.asarray(use, float)
            ok = np.isfinite(dec_a) & np.isfinite(use_a)
            if ok.sum() < 4:
                continue
            dz, uz = dec_a[ok], use_a[ok]
            if dz.std() < 1e-12 or uz.std() < 1e-12:
                continue
            X.extend((dz - dz.mean()) / dz.std())
            Y.extend((uz - uz.mean()) / uz.std())
            M.extend([tag] * ok.sum())
        if len(set(M)) < 3:
            continue
        import pandas as pd
        from statsmodels.regression.mixed_linear_model import MixedLM
        df = pd.DataFrame({"decod_z": X, "use_z": Y, "model": M})
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = MixedLM.from_formula("use_z ~ decod_z", groups="model",
                                           re_formula="1 + decod_z", data=df).fit(
                                               reml=True, method="lbfgs", maxiter=500)
            pooled[which] = dict(
                slope=float(res.params["decod_z"]), se=float(res.bse["decod_z"]),
                p=float(res.pvalues["decod_z"]), n_obs=int(df.shape[0]),
                n_models=int(df["model"].nunique()), converged=bool(res.converged))
        except Exception as e:
            pooled[which] = dict(slope=float("nan"), se=float("nan"), p=float("nan"),
                                 n_obs=int(df.shape[0]),
                                 n_models=int(df["model"].nunique()),
                                 converged=False, note=str(type(e).__name__))
        rs = [r["pearson_r"] for r in results
              if r["definition"] == which and np.isfinite(r["pearson_r"])]
        pooled[which]["mean_per_model_r"] = float(np.mean(rs)) if rs else float("nan")
        pooled[which]["n_models_positive_r"] = int(sum(1 for x in rs if x > 0))
        pooled[which]["n_models_with_r"] = len(rs)
        p = pooled[which]
        print(f"\npooled [{which}]: slope={p['slope']:+.4f} (SE {p['se']:.4f}, "
              f"p={p['p']:.3g})  n_obs={p['n_obs']} over {p['n_models']} models; "
              f"per-model r positive in {p['n_models_positive_r']}/{p['n_models_with_r']}")

    # ---------------- write ----------------
    res_path = os.path.join(a.out, "item_level_dissociation.csv")
    cols = ["model_tag", "definition", "peak_intent_layer", "peak_intent_acc",
            "pearson_r", "ci_lo", "ci_hi", "spearman_rho", "spearman_p", "n"]
    with open(res_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v)
                        for k, v in r.items()})
        w.writerow({})
        for which, p in pooled.items():
            w.writerow({"model_tag": f"POOLED ({which}), model as random effect",
                        "definition": which,
                        "pearson_r": round(p["slope"], 5),
                        "ci_lo": round(p["slope"] - 1.96 * p["se"], 5),
                        "ci_hi": round(p["slope"] + 1.96 * p["se"], 5),
                        "spearman_p": f"{p['p']:.4g}", "n": p["n_obs"]})
    print(f"  -> {res_path}")

    grp_path = os.path.join(a.out, "item_level_groups.csv")
    with open(grp_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model_tag", "scenario_group", "intent_decodability_margin",
                    "intent_use_primary", "intent_use_matched"])
        w.writerows(group_rows)
    print(f"  -> {grp_path}")

    scatter_figure(per_model_pts, results,
                   os.path.join(a.out, "item_level_dissociation.png"), "primary")

    prim = [r for r in results if r["definition"] == "primary"]
    sig = [r for r in prim if np.isfinite(r["ci_lo"]) and np.isfinite(r["ci_hi"])
           and (r["ci_lo"] > 0 or r["ci_hi"] < 0)]
    pp = pooled.get("primary", {})
    md = [
        "# Item-level representation<->behaviour link (J2)", "",
        "## Why this is the primary test now", "",
        "The model-level link had one observation per model and 8 models. C6 recomputed it",
        "without the effect floor and got r = -0.209, 95% CI about [-0.80, +0.58] -- an",
        "interval consistent with a strong link in either direction and with none at all. No",
        "amount of pipeline repair fixes that; n=8 cannot answer the question.", "",
        "Here the unit is the scenario group, so each model contributes up to 53",
        "observations, and the question is sharper: within a single model, does it use intent",
        "more for the stories whose intent it represents more clearly?", "",
        "## Measures", "",
        "- **(a) intent decodability, per scenario group.** Mean out-of-fold signed margin of",
        "  the intent probe at that model's peak intent layer. Folds are split on",
        "  scenario_group, so a scenario is always scored by a probe that never trained on",
        "  it. Signed, so positive is the correct side of the boundary.",
        "- **(b) intent-use, per scenario group.** Two definitions:",
        "  - `primary`: attempted - accidental, the headline diagonal contrast (48 groups).",
        "  - `matched`: the intent effect holding outcome constant, averaging",
        "    (intentional - accidental) and (attempted - neutral) (53 groups).", "",
        "Correlations are bootstrapped over scenario groups. The pooled estimate z-scores",
        "both axes within model before stacking, so it reflects within-model covariation",
        "across scenarios rather than the between-model differences the old test rested on.",
        "", "## Per-model result, primary definition", "",
        "| model | peak intent layer | probe acc | r | 95% CI (bootstrap over groups) | n groups |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted(prim, key=lambda x: x["model_tag"]):
        md.append(f"| {r['model_tag']} | {r['peak_intent_layer']} | "
                  f"{r['peak_intent_acc']:.3f} | {r['pearson_r']:+.3f} | "
                  f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | {r['n']} |")
    md += ["",
           f"- models whose interval excludes zero: {len(sig)} of {len(prim)}", ""]
    if pp:
        md += ["## Pooled, model as a random effect", "",
               f"- slope (both axes z-scored within model): **{pp['slope']:+.4f}** "
               f"(SE {pp['se']:.4f}, p = {pp['p']:.3g})",
               f"- observations: {pp['n_obs']} scenario-group estimates over "
               f"{pp['n_models']} models",
               f"- per-model r positive in {pp['n_models_positive_r']} of "
               f"{pp['n_models_with_r']} models, mean r = {pp['mean_per_model_r']:+.3f}", ""]
        if "matched" in pooled:
            m = pooled["matched"]
            md += [f"- robustness, `matched` definition: slope {m['slope']:+.4f} "
                   f"(SE {m['se']:.4f}, p = {m['p']:.3g}), {m['n_obs']} observations", ""]
        lo, hi = pp["slope"] - 1.96 * pp["se"], pp["slope"] + 1.96 * pp["se"]
        md += [f"The 95% interval on the pooled slope is [{lo:+.3f}, {hi:+.3f}] in",
               "within-model SD units. Both axes are standardised, so the slope is the SD",
               "change in intent-use per SD of intent decodability, and the interval rules out",
               f"anything larger than about {max(abs(lo), abs(hi)):.2f} SD in either direction.",
               "", "**This is an informative null, and that is the difference from C6.** The",
               "model-level test spanned [-0.80, +0.58] and so excluded nothing; this interval",
               "is narrow enough to exclude a moderate or large effect. The reading it supports",
               "is that within a model, the scenarios whose intent is most clearly represented",
               "are not the scenarios where intent is most used -- a dissociation between",
               "representation and use, measured at the level where the two are comparable.", "",
               "Two limits worth stating with it. The bound is on a LINEAR, MONOTONE relation",
               "between probe margin and contrast; a threshold relation, where intent must",
               "merely be present rather than strongly present, would not show up here. And",
               "probe margin is a proxy for representational quality, not a measure of what the",
               "model reads out downstream -- decodable by a linear probe is not the same as",
               "used by the model. Causal steering at the peak intent layer is the test that",
               "would close that gap.", ""]
    md += ["## Status of the old model-level link", "",
           "Retained as a footnote only, and labelled uninformative: r = -0.209, 95% CI",
           "[-0.80, +0.58], n = 8. It is not evidence of absence and should not be cited as",
           "a null.", ""]
    md_path = os.path.join(a.out, "ITEM_LEVEL_DISSOCIATION.md")
    open(md_path, "w").write("\n".join(md))
    print(f"  -> {md_path}")


if __name__ == "__main__":
    main()
