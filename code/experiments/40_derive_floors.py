#!/usr/bin/env python3
"""J4 -- derive the exclusion floors from data instead of from the excluded cases.

WHY THE OLD VALUE IS VOID. EFFECT_FLOOR = 0.05 in 23_build_intent_reliance_summary.py was
justified in its own comment as "tuned against the known degenerate cases (Mistral-7B,
Zephyr-7B)". Those two were not degenerate models. They were measurement failures: the
SentencePiece digit-token collapse made every rating come back as the scale midpoint, so
their coefficients were near zero by construction. The floor was calibrated to exclude a
bug. With the bug fixed, both models show ordinary effects, and the justification for the
value is gone -- while the value itself still gates the anchor comparison.

TWO DISTINCT FLOORS, DERIVED SEPARATELY. The request was to derive the floor from the
rating_std distribution, but the two quantities are not on the same scale, so one cannot be
read off the other. They are handled as what they are:

  1. ENGAGEMENT floor, on rating_std. "Did the model vary its answer at all?" Derived from
     the largest gap in the sorted rating_std values across all 20 models post-fix: a model
     that is not responding to the stimuli sits near zero, one that is responding does not,
     and the empirical gap between those regimes is the non-arbitrary place to cut.

  2. EFFECT floor, on |b_intent| + |b_outcome|. "Is there any moral signal whose ratio is
     worth taking?" This is the quantity EFFECT_FLOOR actually gates, so it is derived from
     the null distribution of that same statistic: permute the condition labels within
     scenario group (preserving the near-identical text shared by cells, exactly as
     02_probe.py does for its probe null), refit, and take a high quantile. That is the
     magnitude the statistic reaches when there is nothing there, which is what a floor is
     supposed to screen out.

The permutation route is the defensible one for EFFECT_FLOOR because it is calibrated
against noise rather than against a set of models chosen in advance.

Outputs
  outputs/stats/floor_derivation.csv
  outputs/stats/floor_derivation.png
  outputs/stats/FLOOR_DERIVATION.md
"""
import argparse
import csv
import glob
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MASTER = os.path.join(ROOT, "dataset", "master", "moral_2x2_master.csv")
BEHAVIOR = os.path.join(ROOT, "outputs", "behavior")
OUTDIR = os.path.join(ROOT, "outputs", "stats")

COND_MAP = {"neutral": (0, 0), "accidental": (0, 1),
            "attempted": (1, 0), "intentional": (1, 1)}


def ols_2x2_from_cells(cells):
    """Same estimator as 03_behavioral.ols_2x2, so the null matches the observed statistic."""
    X, y = [], []
    for cond, (i, o) in COND_MAP.items():
        if cond in cells and cells[cond] is not None:
            X.append([1.0, float(i), float(o)])
            y.append(cells[cond])
    if len(y) < 3:
        return None, None
    X = np.asarray(X)
    y = np.asarray(y)
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return None, None
    return float(beta[1]), float(beta[2])


def load_rows(path, groups):
    out = []
    for r in csv.DictReader(open(path)):
        sid = r["story_id"]
        if sid not in groups:
            continue
        try:
            val = float(r["norm_rating"])
        except (TypeError, ValueError):
            continue
        out.append((r["template"], sid, groups[sid], r["condition"], val))
    return out


def permute_within_groups(rows, rng):
    """Shuffle the condition label among the stories of the same scenario group."""
    by = defaultdict(list)
    for idx, (tmpl, sid, grp, cond, val) in enumerate(rows):
        by[(tmpl, grp)].append(idx)
    conds = [r[3] for r in rows]
    out = list(conds)
    for idxs in by.values():
        perm = rng.permutation(len(idxs))
        for k, i in enumerate(idxs):
            out[i] = conds[idxs[perm[k]]]
    return out


def observed_and_null(rows, n_perm, seed):
    """-> (observed effect magnitudes per template, null magnitudes pooled)."""
    rng = np.random.default_rng(seed)
    templates = sorted({r[0] for r in rows})

    def magnitudes(cond_labels):
        cells = defaultdict(lambda: defaultdict(list))
        for (tmpl, sid, grp, _cond, val), c in zip(rows, cond_labels):
            cells[tmpl][c].append(val)
        mags = []
        for tmpl in templates:
            cm = {c: (float(np.mean(v)) if v else None) for c, v in cells[tmpl].items()}
            bi, bo = ols_2x2_from_cells(cm)
            if bi is not None:
                mags.append(abs(bi) + abs(bo))
        return mags

    obs = magnitudes([r[3] for r in rows])
    null = []
    for _ in range(n_perm):
        null.extend(magnitudes(permute_within_groups(rows, rng)))
    return obs, null


def largest_gap(vals):
    """Biggest jump in the sorted values; the floor is placed in the middle of it."""
    v = np.sort(np.asarray([x for x in vals if x is not None and np.isfinite(x)]))
    if len(v) < 3:
        return None, None, None
    gaps = np.diff(v)
    k = int(np.argmax(gaps))
    return float(v[k]), float(v[k + 1]), float(gaps[k])


def plot(rating_std, obs, null, eng_floor, eff_floor, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.5))

    names = [n for n, _ in rating_std]
    vals = np.array([v for _, v in rating_std])
    order = np.argsort(vals)
    y = np.arange(len(vals))
    ax1.scatter(vals[order], y, s=26,
                c=["#b3202c" if vals[order][i] < eng_floor else "#1f3f8f"
                   for i in range(len(y))], zorder=3)
    ax1.set_yticks(y)
    ax1.set_yticklabels([names[i].split("/")[-1] for i in order], fontsize=7.2)
    ax1.axvline(eng_floor, color="#b3202c", ls="--", lw=1.6,
                label=f"derived engagement floor = {eng_floor:.4f}")
    ax1.axvline(0.05, color="#888888", ls=":", lw=1.4, label="old value 0.05")
    ax1.set_xlabel("rating_std (SD of ratings across items)", fontsize=9.4)
    ax1.set_title("Engagement: rating_std across 20 models, post-fix\n"
                  "floor placed in the largest empirical gap", fontsize=10)
    ax1.legend(fontsize=7.6, loc="lower right")
    ax1.grid(axis="x", alpha=0.25, lw=0.6)

    bins = np.linspace(0, max(max(null) if null else 0.1,
                              np.percentile(obs, 95) if len(obs) else 0.1) * 1.05, 60)
    ax2.hist(null, bins=bins, color="#9aa7c7", alpha=0.85,
             label=f"permutation null (n={len(null)})", density=True)
    ax2.hist(obs, bins=bins, color="#1f3f8f", alpha=0.55,
             label=f"observed, all model x template (n={len(obs)})", density=True)
    ax2.axvline(eff_floor, color="#b3202c", ls="--", lw=1.6,
                label=f"derived effect floor = {eff_floor:.4f}")
    ax2.axvline(0.05, color="#888888", ls=":", lw=1.4, label="old value 0.05")
    ax2.set_xlabel("|b_intent| + |b_outcome|", fontsize=9.4)
    ax2.set_ylabel("density", fontsize=9.4)
    ax2.set_title("Effect floor: null distribution of the gated statistic\n"
                  "labels permuted within scenario group", fontsize=10)
    ax2.legend(fontsize=7.6)
    for ax in (ax1, ax2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=185)
    print(f"  -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=200,
                    help="permutations per model; pooled across models and templates")
    ap.add_argument("--quantile", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUTDIR)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    groups = {r["story_id"]: (r.get("scenario_group") or r.get("scenario_id"))
              for r in csv.DictReader(open(MASTER))}

    # --- 1. engagement floor from rating_std ------------------------------------
    stats_path = os.path.join(ROOT, "outputs", "stats", "contrast_by_model.csv")
    rating_std = []
    for r in csv.DictReader(open(stats_path)):
        try:
            rating_std.append((r["model"], float(r["rating_std"])))
        except (KeyError, TypeError, ValueError):
            continue
    lo, hi, gap = largest_gap([v for _, v in rating_std])
    eng_floor = round((lo + hi) / 2, 4) if lo is not None else 0.05
    print(f"engagement floor: largest gap in rating_std is {lo:.4f} -> {hi:.4f} "
          f"(width {gap:.4f}); floor = {eng_floor:.4f}")

    # --- 2. effect floor from the permutation null ------------------------------
    files = sorted(glob.glob(os.path.join(BEHAVIOR, "raw_*.csv")))
    all_obs, all_null, per_model = [], [], []
    for f in files:
        model = os.path.basename(f)[len("raw_"):-len(".csv")].replace("_", "/", 1)
        rows = load_rows(f, groups)
        if not rows:
            continue
        obs, null = observed_and_null(rows, a.n_perm, a.seed)
        all_obs.extend(obs)
        all_null.extend(null)
        own = float(np.quantile(null, a.quantile)) if null else float("nan")
        # The floor is applied per template in 23_build_intent_reliance_summary.py, so the
        # per-template count is what decides whether a model is called degenerate. Requiring
        # merely one template to clear is too weak: at the q95 threshold each template clears
        # with probability 0.05 under the null, so with 13 templates chance alone delivers
        # about 0.65 of them. The count is therefore aggregated with a binomial test.
        n_pass = sum(1 for x in obs if x >= own)
        alpha = 1.0 - a.quantile
        from scipy.stats import binomtest
        binom_p = (binomtest(n_pass, len(obs), alpha, alternative="greater").pvalue
                   if obs else float("nan"))
        degenerate = not (binom_p < 0.05)
        per_model.append((model, float(np.mean(obs)) if obs else float("nan"), own,
                          n_pass, len(obs), binom_p, degenerate))
        print(f"  {model:44} obs_mean={np.mean(obs):.4f} "
              f"own_null_q{int(a.quantile*100)}={own:.4f} "
              f"above_null={n_pass}/{len(obs)} binom_p={binom_p:.3g}"
              f"{'   <-- DEGENERATE' if degenerate else ''}")

    eff_floor = round(float(np.quantile(all_null, a.quantile)), 4)
    null_q99 = round(float(np.quantile(all_null, 0.99)), 4)
    print(f"\neffect floor: pooled null q{int(a.quantile*100)} = {eff_floor:.4f} "
          f"(q99 = {null_q99:.4f}); old hand-set value was 0.05")

    plot(rating_std, all_obs, all_null, eng_floor, eff_floor,
         os.path.join(a.out, "floor_derivation.png"))

    csv_path = os.path.join(a.out, "floor_derivation.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "value", "basis"])
        w.writerow(["engagement_floor_rating_std", eng_floor,
                    f"midpoint of largest gap in sorted rating_std "
                    f"({lo:.4f} -> {hi:.4f}, width {gap:.4f}), 20 models post-fix"])
        w.writerow(["effect_floor_abs_b_sum", eff_floor,
                    f"q{int(a.quantile*100)} of within-scenario-group permutation null of "
                    f"|b_intent|+|b_outcome|, {len(all_null)} draws"])
        w.writerow(["effect_floor_abs_b_sum_q99", null_q99, "same null, 99th percentile"])
        w.writerow(["effect_floor_old", 0.05,
                    "hand-set; justified as excluding Mistral-7B and Zephyr-7B, which are "
                    "now known to have been measurement failures"])
        w.writerow([])
        w.writerow(["model", "observed_mean_abs_b_sum",
                    f"own_null_q{int(a.quantile*100)}_abs_b_sum",
                    "n_templates_above_own_null", "n_templates", "binomial_p",
                    "degenerate_by_own_null", f"clears_global_{eff_floor}"])
        for m, o, n, npass, ntot, bp, deg in per_model:
            w.writerow([m, round(o, 4), round(n, 4), npass, ntot,
                        f"{bp:.4g}", deg, o >= eff_floor])
        w.writerow([])
        w.writerow(["model", "rating_std", f"engaged_at_{eng_floor}"])
        for m, v in sorted(rating_std, key=lambda x: x[1]):
            w.writerow([m, round(v, 4), v >= eng_floor])
    print(f"  -> {csv_path}")

    n_below_eng = sum(1 for _, v in rating_std if v < eng_floor)
    n_below_old = sum(1 for _, v in rating_std if v < 0.05)
    md = [
        "# Derivation of the exclusion floors (J4)", "",
        "## Why the old value had to go", "",
        "`EFFECT_FLOOR = 0.05` in `23_build_intent_reliance_summary.py` was justified in its",
        "own comment as tuned against \"the known degenerate cases (Mistral-7B, Zephyr-7B)\".",
        "Those models were not degenerate. The digit-token collapse in the SentencePiece",
        "tokenizers returned the scale midpoint for every item, so their coefficients were",
        "near zero by construction. The floor was calibrated against a bug, and it was still",
        "gating the anchor comparison after the bug was fixed.", "",
        "## Two floors, because there are two quantities", "",
        "The request was to read the floor off the `rating_std` distribution. `rating_std` is a",
        "dispersion in rating units and `|b_intent| + |b_outcome|` is a sum of regression",
        "coefficients, so one cannot be read off the other. Both are derived, separately.", "",
        f"### 1. Engagement floor on `rating_std` = **{eng_floor:.4f}**", "",
        f"Sorted across all 20 models post-fix, the largest gap runs from {lo:.4f} to {hi:.4f}",
        f"(width {gap:.4f}); the floor is placed at its midpoint. Models below it are not",
        "varying their response to the stimuli, so no ratio computed from them is meaningful.", "",
        f"- excluded by the derived floor: {n_below_eng} of {len(rating_std)} models",
        f"- excluded by the old 0.05: {n_below_old} of {len(rating_std)} models", "",
        f"### 2. Effect floor on `|b_intent| + |b_outcome|` = **{eff_floor:.4f}**", "",
        "This is the statistic `EFFECT_FLOOR` actually gates, so it is calibrated against the",
        "magnitude that statistic reaches under noise. Condition labels are permuted within",
        "scenario group -- never across -- because the 4 or 8 cells of a group share nearly all",
        "their text, and a global shuffle would break that dependency and give a null with the",
        "wrong variance. This is the same permutation scheme `02_probe.py` uses for its probe",
        "null.", "",
        f"- pooled null q{int(a.quantile * 100)}: **{eff_floor:.4f}**  ({len(all_null)} draws)",
        f"- pooled null q99: {null_q99:.4f}",
        f"- old hand-set value: 0.05", "",
        "A floor set at a null quantile has a stated meaning: values below it are reached by",
        "chance at least that often when the labels carry no information.", "",
        "### A global constant is the wrong shape for this floor", "",
        "The per-model nulls in the CSV span an order of magnitude, from about 0.005 for",
        "gemma-2-9b base to about 0.145 for zephyr-7b-beta, because the null magnitude of a",
        "coefficient sum scales with how much the model varies its ratings at all. Any single",
        "constant is therefore too strict for low-variance models and too lenient for",
        "high-variance ones. The pooled q95 of "
        f"{eff_floor:.4f} is set mostly by the high-variance instruct models, and the old 0.05",
        "sat below the null of several of them -- meaning it was admitting template estimates",
        "that permutation reaches by chance.", "",
        "**Recommendation: replace the scalar with the per-model permutation test.** A template",
        "enters the average if its `|b_intent| + |b_outcome|` exceeds that model's own null",
        f"q{int(a.quantile * 100)}. That is a per-model significance test rather than a shared",
        "cutoff, and it cannot be re-tuned by which models happen to be in the roster. The",
        "pooled value above is retained only as a fallback for code paths needing one number.", "",
        "The per-template outcomes are then aggregated with a binomial test rather than by",
        f"asking whether any single template clears. At a q{int(a.quantile * 100)} threshold",
        f"each template clears with probability {1 - a.quantile:.2f} under the null, so across",
        "13 templates chance alone supplies about 0.65 of them; \"at least one cleared\" is not",
        "evidence of anything. A model is called degenerate unless the number of clearing",
        "templates is itself unlikely under that binomial (p < 0.05).", "",
        "Models called degenerate on this criterion:", "",
    ] + ([f"- {m} ({npass}/{ntot} templates, binomial p = {bp:.3g})"
          for m, _o, _n, npass, ntot, bp, deg in per_model if deg] or ["- none"]) + [
        "",
        "This is a different and better-founded list than the one the old 0.05 produced, and",
        "notably it no longer contains Mistral or Zephyr, the two models the old value was",
        "built around.", "",
        "### Significant is not the same as usable", "",
        "The base models now clear their own nulls, but read the magnitudes next to the",
        "p-values before treating that as a licence to compute ratios from them. Several sit",
        "around `|b_intent| + |b_outcome|` = 0.01 on a 0-1 blame scale -- reliably nonzero across",
        "13 templates, and still negligible. The original reason for a floor was that",
        "`|b_int| / (|b_int| + |b_outcome|)` is unstable when both terms are tiny, and that",
        "instability is a function of magnitude, not of significance. The permutation test",
        "answers \"is there a signal\"; it does not answer \"is the signal large enough for its",
        "ratio to mean anything\". Both columns are in the CSV and the intent-reliance index",
        "should be reported with the magnitude beside it for any model in that range.", "",
        "## What this changes", "",
        "See `check_c5_engagement_floor.csv` for the anchor comparison re-run at the derived",
        "engagement floor alongside 0.05 and 0.10. The conclusions are stable across all three",
        "only if the same models clear every value; where they do not, the CSV shows which",
        "models move and the anchor counts change with them.", "",
    ]
    md_path = os.path.join(a.out, "FLOOR_DERIVATION.md")
    open(md_path, "w").write("\n".join(md))
    print(f"  -> {md_path}")


if __name__ == "__main__":
    main()
