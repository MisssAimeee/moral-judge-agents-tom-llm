#!/usr/bin/env python3
"""J1 part 2 — BigToM false-belief vs 2×2 contrast.

Primary deliverable is the per-model table and a single scatter (BigToM FB ×
contrast, base/instruct by marker, engaged models labelled, no regression line).
The populated high-FB / negative-contrast region is the finding.

The unrestricted all-model Pearson r is retained only as a confound demonstration
(base-vs-instruct moves both axes). Controlled analyses (a)/(b)/(c) are secondary.

ToMi is not used — see TOMI_SCORING_AUDIT.md. BigToM numbers are init_belief=0.
Floor policy: engaged labels use the rating_std engagement floor; every model is
kept in correlation / OLS analyses (FLOOR_DERIVATION.md).
"""
import argparse
import csv
import math
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
TOMDIR = os.path.join(ROOT, "outputs", "tom_benchmarks")
STATS = os.path.join(ROOT, "outputs", "stats", "contrast_by_model.csv")
FLOOR_MD = os.path.join(ROOT, "outputs", "stats", "FLOOR_DERIVATION.md")
N_BOOT = 10000
ENGAGEMENT_FLOOR = 0.2191  # from 40_derive_floors.py; engagement labels only

FAMILY_STEMS = [
    ("Qwen2.5-0.5B", "Qwen2.5-0.5B", "Qwen2.5-0.5B-Instruct"),
    ("Qwen2.5-1.5B", "Qwen2.5-1.5B", "Qwen2.5-1.5B-Instruct"),
    ("Qwen2.5-3B", "Qwen2.5-3B", "Qwen2.5-3B-Instruct"),
    ("Qwen2.5-7B", "Qwen2.5-7B", "Qwen2.5-7B-Instruct"),
    ("Qwen2.5-14B", "Qwen2.5-14B", "Qwen2.5-14B-Instruct"),
    ("OLMo-2-1124-7B", "OLMo-2-1124-7B", "OLMo-2-1124-7B-Instruct"),
    ("Mistral-7B-v0.3", "Mistral-7B-v0.3", "Mistral-7B-Instruct-v0.3"),
    ("gemma-2-9b", "gemma-2-9b", "gemma-2-9b-it"),
    ("Meta-Llama-3.1-8B", "Meta-Llama-3.1-8B", "Meta-Llama-3.1-8B-Instruct"),
]


def joinkey(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_tom():
    path = os.path.join(TOMDIR, "tom_accuracy_by_model.csv")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}; run 36_tom_benchmarks.py first")
    out = {}
    for r in csv.DictReader(open(path)):
        acc = fnum(r["accuracy"])
        if acc is None:
            continue
        out.setdefault(joinkey(r["model"]), {})[r["subset"]] = acc
    return out


def load_behaviour():
    rows = []
    for r in csv.DictReader(open(STATS)):
        rows.append(dict(
            model=r["model"], contrast=fnum(r.get("contrast")),
            rating_std=fnum(r.get("rating_std")),
            size_B=fnum(r.get("size_B")), mtype=r.get("type", ""),
            key=joinkey(r["model"]),
            short=r["model"].split("/")[-1].replace("_", "-")))
    return rows


def join_rows(tom, beh):
    rows = []
    for b in beh:
        t = tom.get(b["key"])
        if t is None:
            cand = [v for k, v in tom.items()
                    if k.endswith(b["key"]) or b["key"].endswith(k)]
            t = cand[0] if len(cand) == 1 else None
        if t is None or b["contrast"] is None:
            continue
        rs = b["rating_std"]
        rows.append(dict(
            model=b["model"], short=b["short"], mtype=b["mtype"],
            size_B=b["size_B"], rating_std=rs, contrast=b["contrast"],
            engaged=bool(rs is not None and rs >= ENGAGEMENT_FLOOR),
            bigtom_false=t.get("bigtom|false_belief"),
            bigtom_all=t.get("bigtom"), tomi=t.get("tomi"),
            log_size=math.log(b["size_B"]) if b["size_B"] and b["size_B"] > 0 else None,
        ))
    return rows


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def boot_ci(x, y, n_boot=N_BOOT, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 4:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if x[i].std() < 1e-12 or y[i].std() < 1e-12:
            continue
        rs.append(np.corrcoef(x[i], y[i])[0, 1])
    if not rs:
        return np.nan, np.nan
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def label_r(r, lo, hi, n, context=""):
    if not np.isfinite(r) or not np.isfinite(lo):
        return f"NOT ESTIMABLE (n={n})"
    width = hi - lo
    if width > 0.9:
        return ("UNINFORMATIVE — interval too wide to exclude a moderate effect either "
                "way. Not a null.")
    if lo <= 0 <= hi:
        return ("NULL, bounded — ToM accuracy does not predict the contrast once "
                f"{context or 'the stated controls'} are applied.")
    if lo > 0:
        return ("POSITIVE — moral intent use partly tracks ToM competence; "
                "dissociation claim weakens.")
    return ("NEGATIVE — higher ToM tracks more outcome-driven contrast; check residual "
            "confounds before interpreting as a finding.")


def analysis_a(rows, meas_key, meas_label):
    sub = [r for r in rows if r["mtype"] == "instruct" and r.get(meas_key) is not None]
    x = [r[meas_key] for r in sub]
    y = [r["contrast"] for r in sub]
    r = pearson(x, y)
    lo, hi = boot_ci(x, y)
    return dict(analysis="(a) instruct only", measure=meas_label, r=r, lo=lo, hi=hi,
                n=len(sub), label=label_r(r, lo, hi, len(sub), "restricting to instruct"))


def analysis_b(rows, meas_key, meas_label):
    import statsmodels.formula.api as smf
    import pandas as pd
    sub = [r for r in rows if r.get(meas_key) is not None and r["log_size"] is not None
           and r["mtype"] in ("base", "instruct")]
    df = pd.DataFrame(sub).rename(columns={meas_key: "tom"})
    if len(df) < 6:
        return dict(analysis="(b) OLS with covariates", measure=meas_label,
                    r=float("nan"), lo=float("nan"), hi=float("nan"), n=len(df),
                    label=f"NOT ESTIMABLE (n={len(df)})", beta_tom=float("nan"),
                    se_tom=float("nan"), p_tom=float("nan"))
    fit = smf.ols("contrast ~ tom + C(mtype) + log_size", data=df).fit()
    beta = float(fit.params["tom"])
    se = float(fit.bse["tom"])
    p = float(fit.pvalues["tom"])
    r_tom = smf.ols("tom ~ C(mtype) + log_size", data=df).fit()
    r_con = smf.ols("contrast ~ C(mtype) + log_size", data=df).fit()
    pr = pearson(r_tom.resid, r_con.resid)
    plo, phi = boot_ci(r_tom.resid.values, r_con.resid.values)
    return dict(analysis="(b) OLS with covariates", measure=meas_label,
                r=pr, lo=plo, hi=phi, n=len(df),
                beta_tom=beta, se_tom=se, p_tom=p,
                label=label_r(pr, plo, phi, len(df), "controlling for type and log-size"))


def find_row(rows, stem):
    jk = joinkey(stem)
    for r in rows:
        if joinkey(r["short"]) == jk or joinkey(r["model"]).endswith(jk):
            return r
    for r in rows:
        if jk in joinkey(r["short"]) or jk in joinkey(r["model"]):
            return r
    return None


def analysis_c(rows, meas_key, meas_label):
    deltas = []
    for fam, base_stem, inst_stem in FAMILY_STEMS:
        b = find_row(rows, base_stem)
        i = find_row(rows, inst_stem)
        if b is None or i is None:
            continue
        if b.get(meas_key) is None or i.get(meas_key) is None:
            continue
        if b["mtype"] == "instruct" and i["mtype"] == "base":
            b, i = i, b
        deltas.append(dict(
            family=fam,
            d_tom=i[meas_key] - b[meas_key],
            d_contrast=i["contrast"] - b["contrast"],
            base_tom=b[meas_key], inst_tom=i[meas_key],
            base_contrast=b["contrast"], inst_contrast=i["contrast"],
        ))
    if len(deltas) < 3:
        return deltas, dict(analysis="(c) within-family deltas", measure=meas_label,
                            r=float("nan"), lo=float("nan"), hi=float("nan"),
                            n=len(deltas), label=f"NOT ESTIMABLE (n={len(deltas)})")
    x = [d["d_tom"] for d in deltas]
    y = [d["d_contrast"] for d in deltas]
    r = pearson(x, y)
    lo, hi = boot_ci(x, y)
    return deltas, dict(analysis="(c) within-family deltas", measure=meas_label,
                        r=r, lo=lo, hi=hi, n=len(deltas),
                        label=label_r(r, lo, hi, len(deltas),
                                      "looking at within-family tuning deltas"))


def confound_demo(rows, meas_key, meas_label):
    sub = [r for r in rows if r.get(meas_key) is not None]
    x = [r[meas_key] for r in sub]
    y = [r["contrast"] for r in sub]
    r = pearson(x, y)
    lo, hi = boot_ci(x, y)
    return dict(analysis="(confound demo) all models, no controls",
                measure=meas_label, r=r, lo=lo, hi=hi, n=len(sub),
                label=("CONFOUND DEMO — both axes proxy base-vs-instruct (and size). "
                       "Do not cite as a result."))


def scatter_primary(rows, path):
    """Single 2D scatter: BigToM FB × contrast. No regression line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    styles = {
        "instruct": dict(c="#1f3f8f", marker="o", label="instruct"),
        "base": dict(c="#c07a1e", marker="s", label="base"),
    }
    for mtype, sty in styles.items():
        sub = [r for r in rows if r["mtype"] == mtype and r["bigtom_false"] is not None]
        ax.scatter([r["bigtom_false"] for r in sub], [r["contrast"] for r in sub],
                   s=55, alpha=0.9, edgecolor="white", linewidth=0.6, **sty)
    for r in rows:
        if not r.get("engaged") or r["bigtom_false"] is None:
            continue
        ax.annotate(r["short"][:22], (r["bigtom_false"], r["contrast"]),
                    fontsize=7, xytext=(4, 3), textcoords="offset points",
                    color="#222")
    # Finding quadrant guide (not a fit)
    ax.axhline(0, color="#888", lw=0.7, ls=":")
    ax.axvline(0.82, color="#bbb", lw=0.7, ls="--")
    ax.axhline(-0.37, color="#bbb", lw=0.7, ls="--")
    ax.set_xlabel("BigToM false-belief accuracy (init_belief=0)")
    ax.set_ylabel("2×2 contrast (attempted − accidental)")
    ax.set_title("Models that pass hard false belief still show outcome-driven moral contrast")
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    ax.grid(alpha=0.2, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()
    print(f"  -> {path}")


def fmt(v, digits=3):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    return f"{v:.{digits}f}"


def write_md(rows, results, deltas):
    engaged = [r for r in rows if r["engaged"]]
    quad = [r for r in engaged
            if r["bigtom_false"] is not None and r["bigtom_false"] >= 0.82
            and r["contrast"] <= -0.37]
    demo = next(r for r in results if r["analysis"].startswith("(confound")
                and r["measure"] == "bigtom")
    demo_fb = next(r for r in results if r["analysis"].startswith("(confound")
                   and r["measure"] == "bigtom|false_belief")

    md = [
        "# ToM benchmark performance vs intent use in moral judgment (J1)", "",
        "## Question", "",
        "Do models that pass a standard false-belief ToM benchmark still fail to weight",
        "intent in graded moral judgment?",
        "",
        "## Finding (primary)", "",
        "Models that pass BigToM false belief at **0.82–0.99** are the same models with",
        "2×2 contrasts of **−0.37 to −0.65** — outcome-driven, not intent-driven. The",
        "per-model table and the scatter (BigToM FB × contrast, base/instruct by marker,",
        "engaged models labelled, **no regression line**) are the deliverable. The",
        "populated high-FB / negative-contrast region is the dissociation claim.",
        "",
        f"Engagement labels use `rating_std ≥ {ENGAGEMENT_FLOOR}` (engagement floor only;",
        "correlations keep every model — see `FLOOR_DERIVATION.md`).",
        "",
        "## BigToM condition: `init_belief=0`", "",
        "All BigToM numbers here use the **hard** Forward-Belief variant: sentence 4 of",
        "each story (the explicit statement of the agent’s initial belief) is dropped, so",
        "the model must **infer** the belief rather than copy it. That is the",
        "`init_belief=0` setting from the BigToM generator. Reporting a pass under this",
        "condition is stronger than a pass when the belief is written out in the prompt.",
        "",
        "## ToMi", "",
        "ToMi is **not** used in the argument. An audit found the scored 400-item slice is",
        "82% `no_tom` items; aggregate accuracy is not a false-belief measure. See",
        "`TOMI_SCORING_AUDIT.md`. Numbers below retain ToMi only for provenance.",
        "",
        "## Per-model table", "",
        "| model | type | engaged | BigToM FB | contrast |",
        "|---|---|---|---:|---:|",
    ]
    for r in sorted(rows, key=lambda z: (-(z["bigtom_false"] or -1), z["contrast"] or 0)):
        md.append(
            f"| {r['model'].replace('/', '-')} | {r['mtype']} | "
            f"{'yes' if r['engaged'] else 'no'} | {fmt(r['bigtom_false'])} | "
            f"{r['contrast']:+.3f} |"
        )

    md += ["", "### Finding quadrant (FB ≥ 0.82 and contrast ≤ −0.37)", "",
           "| model | BigToM FB | contrast |",
           "|---|---:|---:|"]
    for r in sorted(quad, key=lambda z: z["contrast"]):
        md.append(f"| {r['model'].replace('/', '-')} | {fmt(r['bigtom_false'])} | "
                  f"{r['contrast']:+.3f} |")

    md += [
        "", "## Scatter", "",
        "`tom_vs_contrast.png` — x = BigToM false-belief (`init_belief=0`), y = contrast,",
        "marker = base vs instruct, labels = engaged models only, **no regression line**.",
        "",
        "## Why the all-model correlation is not a result", "",
        "Both axes proxy model type. Base models often struggle with the BigToM QA format",
        "and sit near zero on contrast; instruction tuning moves both. The unrestricted",
        f"Pearson r on BigToM-all over {demo['n']} models is **r = {demo['r']:+.3f}** "
        f"[{demo['lo']:+.3f}, {demo['hi']:+.3f}] — a **confound demonstration**, not a finding.",
        f"On BigToM false belief alone the raw r is {demo_fb['r']:+.3f} "
        f"[{demo_fb['lo']:+.3f}, {demo_fb['hi']:+.3f}]. Do not cite either as the result; "
        "cite the table.",
        "",
        "## Secondary controlled analyses (BigToM FB only)", "",
        "Kept for completeness after the table. These are not the headline.",
        "",
        "| analysis | estimate | 95% CI | n |",
        "|---|---|---|---|",
    ]
    for res in results:
        if res["measure"] != "bigtom|false_belief":
            continue
        if res["analysis"].startswith("(confound"):
            continue
        est = res.get("r")
        lo, hi = res.get("lo"), res.get("hi")
        extra = ""
        if res.get("beta_tom") is not None and isinstance(res["beta_tom"], float) \
                and math.isfinite(res["beta_tom"]):
            extra = f" (β_tom={res['beta_tom']:+.3f}, p={res['p_tom']:.3g})"
        if res["analysis"].startswith("(a)"):
            cell = f"r = {est:+.3f}{extra}"
        elif res["analysis"].startswith("(b)"):
            cell = f"partial r = {est:+.3f}{extra}"
        else:
            cell = f"r = {est:+.3f}{extra}"
        md.append(f"| {res['analysis']} | {cell} | [{lo:+.3f}, {hi:+.3f}] | {res['n']} |")

    md += ["", "## Within-family deltas (BigToM FB)", "",
           "| family | Δ ToM (I−B) | Δ contrast (I−B) |",
           "|---|---|---|"]
    for d in deltas:
        md.append(f"| {d['family']} | {d['d_tom']:+.3f} | {d['d_contrast']:+.3f} |")

    md += [
        "", "## Closed models", "",
        "Closed-API BigToM (generative) is reported in",
        "`tom_accuracy_by_model_generative.csv` / `CLOSED_TOM.md` when available.",
        "**Do not correlate** closed ToM accuracy against their moral contrasts — those",
        "contrasts are still v1-contaminated. Standalone ToM numbers only.",
        "",
        "## Reading", "",
        "Lead with the table and scatter. Models that clear hard false belief still produce",
        "strongly outcome-driven moral contrasts. That is the dissociation measured on our",
        "own open-weight roster under `init_belief=0`.",
        "",
    ]
    path = os.path.join(TOMDIR, "TOM_VS_CONTRAST.md")
    open(path, "w").write("\n".join(md))
    print(f"  -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", default="bigtom_false",
                    help="primary measure: bigtom_false (default) / bigtom_all")
    a = ap.parse_args()
    key_map = {"bigtom": "bigtom_all", "bigtom|false_belief": "bigtom_false",
               "bigtom_all": "bigtom_all", "bigtom_false": "bigtom_false"}
    primary_key = key_map.get(a.primary, a.primary)

    rows = join_rows(load_tom(), load_behaviour())
    print(f"joined {len(rows)} models  (engaged floor={ENGAGEMENT_FLOOR})")

    # Secondary analyses on FB + confound demos on FB and all; ToMi confound only
    measures = [("bigtom_false", "bigtom|false_belief"),
                ("bigtom_all", "bigtom")]
    results = []
    deltas_primary = []
    for mk, label in measures:
        results.append(confound_demo(rows, mk, label))
        if mk == "bigtom_false":
            results.append(analysis_a(rows, mk, label))
            results.append(analysis_b(rows, mk, label))
            deltas, cres = analysis_c(rows, mk, label)
            results.append(cres)
            if mk == primary_key:
                deltas_primary = deltas

    os.makedirs(TOMDIR, exist_ok=True)
    csv_path = os.path.join(TOMDIR, "tom_vs_contrast.csv")
    with open(csv_path, "w", newline="") as fh:
        cols = ["model", "mtype", "size_B", "rating_std", "engaged", "contrast",
                "bigtom_false", "bigtom_all", "tomi"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda z: -(z["bigtom_false"] or -1)):
            w.writerow(r)
        w.writerow({})
        w.writerow({"model": "ANALYSIS", "mtype": "measure", "size_B": "r_or_partial_r",
                    "rating_std": "ci_lo", "engaged": "ci_hi", "contrast": "n",
                    "bigtom_false": "beta_tom", "bigtom_all": "p_tom", "tomi": "label"})
        for res in results:
            w.writerow({"model": res["analysis"], "mtype": res["measure"],
                        "size_B": res.get("r"), "rating_std": res.get("lo"),
                        "engaged": res.get("hi"), "contrast": res.get("n"),
                        "bigtom_false": res.get("beta_tom", ""),
                        "bigtom_all": res.get("p_tom", ""),
                        "tomi": res.get("label")})
        if deltas_primary:
            w.writerow({})
            w.writerow({"model": "WITHIN_FAMILY_DELTAS", "mtype": "family",
                        "size_B": "d_tom", "rating_std": "d_contrast",
                        "engaged": "", "contrast": "base_tom",
                        "bigtom_false": "inst_tom", "bigtom_all": "base_contrast",
                        "tomi": "inst_contrast"})
            for d in deltas_primary:
                w.writerow({"model": d["family"], "mtype": d["family"],
                            "size_B": d["d_tom"], "rating_std": d["d_contrast"],
                            "contrast": d["base_tom"], "bigtom_false": d["inst_tom"],
                            "bigtom_all": d["base_contrast"], "tomi": d["inst_contrast"]})
    print(f"  -> {csv_path}")

    scatter_primary(rows, os.path.join(TOMDIR, "tom_vs_contrast.png"))
    write_md(rows, results, deltas_primary)
    for res in results:
        print(f"  {res['analysis']:40} {res['measure']:22} "
              f"r={res.get('r', float('nan')):+.3f} n={res['n']}")


if __name__ == "__main__":
    main()
