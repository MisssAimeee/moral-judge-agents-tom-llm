#!/usr/bin/env python3
"""J1 part 2 -- ToM accuracy vs 2x2 contrast, with the base/instruct confound controlled.

The raw all-20 correlation is confounded: base models sit near floor on BigToM (they
cannot follow the QA format) and near zero on contrast, while instruct models move both
axes with size. That number is retained only as a demonstration of the confound.

Primary analyses (no engagement floor; every model kept; type/size handled explicitly):

  (a) instruct models only — Pearson r, bootstrap CI over models
  (b) all models, OLS: contrast ~ ToM + C(mtype) + log(size_B)
  (c) within-family base→instruct deltas on both measures — does the tuning step that
      raises ToM also raise intent use?

Interpretation (fixed before the controlled analyses were fit):
  * NULL under (a)/(b)/(c): ToM-benchmark performance does not predict intent use once
    model type is held constant. Dissociation claim supported by our own data.
  * POSITIVE under (a)/(b) or positive delta–delta under (c): the moral task partly
    measures general ToM competence; dissociation claim weakens.
  * The unrestricted all-20 r is NOT a result — it is the confound.

Floor policy (J4 separation): correlation analyses keep every model. The derived
rating_std floor is for engagement / anchor counts only; see FLOOR_DERIVATION.md.
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
N_BOOT = 10000

# Family stems for within-family base→instruct deltas. Zephyr and Tulu have no
# paired base in the behavioural roster under a distinct "base" label.
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
        rows.append(dict(model=r["model"], contrast=fnum(r.get("contrast")),
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
            cand = [v for k, v in tom.items() if k.endswith(b["key"]) or b["key"].endswith(k)]
            t = cand[0] if len(cand) == 1 else None
        if t is None or b["contrast"] is None:
            continue
        rows.append(dict(
            model=b["model"], short=b["short"], mtype=b["mtype"],
            size_B=b["size_B"], rating_std=b["rating_std"], contrast=b["contrast"],
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
    """Instruct models only."""
    sub = [r for r in rows if r["mtype"] == "instruct" and r.get(meas_key) is not None]
    x = [r[meas_key] for r in sub]
    y = [r["contrast"] for r in sub]
    r = pearson(x, y)
    lo, hi = boot_ci(x, y)
    return dict(analysis="(a) instruct only", measure=meas_label, r=r, lo=lo, hi=hi,
                n=len(sub), label=label_r(r, lo, hi, len(sub), "restricting to instruct"))


def analysis_b(rows, meas_key, meas_label):
    """OLS: contrast ~ ToM + C(mtype) + log(size)."""
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
    lo, hi = beta - 1.96 * se, beta + 1.96 * se
    # Partial correlation of residuals after regressing both on covariates
    r_tom = smf.ols("tom ~ C(mtype) + log_size", data=df).fit()
    r_con = smf.ols("contrast ~ C(mtype) + log_size", data=df).fit()
    pr = pearson(r_tom.resid, r_con.resid)
    plo, phi = boot_ci(r_tom.resid.values, r_con.resid.values)
    return dict(analysis="(b) OLS with covariates", measure=meas_label,
                r=pr, lo=plo, hi=phi, n=len(df),
                beta_tom=beta, se_tom=se, p_tom=p, beta_lo=lo, beta_hi=hi,
                label=label_r(pr, plo, phi, len(df), "controlling for type and log-size"))


def find_row(rows, stem):
    jk = joinkey(stem)
    for r in rows:
        if joinkey(r["short"]) == jk or joinkey(r["model"]).endswith(jk):
            return r
    # looser: stem contained
    for r in rows:
        if jk in joinkey(r["short"]) or jk in joinkey(r["model"]):
            return r
    return None


def analysis_c(rows, meas_key, meas_label):
    """Within-family base→instruct deltas."""
    deltas = []
    for fam, base_stem, inst_stem in FAMILY_STEMS:
        b = find_row(rows, base_stem)
        i = find_row(rows, inst_stem)
        if b is None or i is None:
            continue
        if b.get(meas_key) is None or i.get(meas_key) is None:
            continue
        # Prefer typed rows when available
        if b["mtype"] == "instruct" and i["mtype"] == "base":
            b, i = i, b
        deltas.append(dict(
            family=fam,
            d_tom=i[meas_key] - b[meas_key],
            d_contrast=i["contrast"] - b["contrast"],
            base=b["model"], instruct=i["model"],
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
    """All-20 raw r — demonstration of the confound, not a result."""
    sub = [r for r in rows if r.get(meas_key) is not None]
    x = [r[meas_key] for r in sub]
    y = [r["contrast"] for r in sub]
    r = pearson(x, y)
    lo, hi = boot_ci(x, y)
    return dict(analysis="(confound demo) all models, no controls",
                measure=meas_label, r=r, lo=lo, hi=hi, n=len(sub),
                label=("CONFOUND DEMO — both axes proxy base-vs-instruct (and size). "
                       "Do not cite as a result."))


def scatter_panels(rows, deltas, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))

    # (a) instruct only
    ax = axes[0]
    inst = [r for r in rows if r["mtype"] == "instruct" and r["bigtom_all"] is not None]
    ax.scatter([r["bigtom_all"] for r in inst], [r["contrast"] for r in inst],
               s=[30 + 4 * (r["size_B"] or 1) for r in inst],
               c="#1f3f8f", alpha=0.85, edgecolor="white")
    for r in inst:
        ax.annotate(r["short"][:18], (r["bigtom_all"], r["contrast"]),
                    fontsize=6, xytext=(3, 2), textcoords="offset points", color="#444")
    ax.axhline(0, color="#888", lw=0.8, ls=":")
    ax.set_title("(a) instruct only", fontsize=10)
    ax.set_xlabel("BigToM accuracy (all)")
    ax.set_ylabel("2×2 contrast")

    # (b) all, coloured by type
    ax = axes[1]
    for mtype, colour, marker in (("instruct", "#1f3f8f", "o"), ("base", "#c07a1e", "s")):
        sub = [r for r in rows if r["mtype"] == mtype and r["bigtom_all"] is not None]
        ax.scatter([r["bigtom_all"] for r in sub], [r["contrast"] for r in sub],
                   s=40, c=colour, marker=marker, alpha=0.8, label=mtype, edgecolor="white")
    ax.axhline(0, color="#888", lw=0.8, ls=":")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("(b) all models (type shown)", fontsize=10)
    ax.set_xlabel("BigToM accuracy (all)")

    # (c) deltas
    ax = axes[2]
    if deltas:
        ax.scatter([d["d_tom"] for d in deltas], [d["d_contrast"] for d in deltas],
                   s=55, c="#2a6f4e", alpha=0.85, edgecolor="white")
        for d in deltas:
            ax.annotate(d["family"], (d["d_tom"], d["d_contrast"]),
                        fontsize=6.5, xytext=(3, 2), textcoords="offset points")
        ax.axhline(0, color="#888", lw=0.8, ls=":")
        ax.axvline(0, color="#888", lw=0.8, ls=":")
    ax.set_title("(c) within-family Δ (instruct − base)", fontsize=10)
    ax.set_xlabel("Δ BigToM")
    ax.set_ylabel("Δ contrast")

    for ax in axes:
        ax.grid(alpha=0.2, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("ToM vs moral intent use — confound-controlled analyses", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    print(f"  -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", default="bigtom",
                    help="primary ToM measure key in joined rows: bigtom_all / bigtom_false / tomi")
    a = ap.parse_args()
    key_map = {"bigtom": "bigtom_all", "bigtom|false_belief": "bigtom_false",
               "bigtom_all": "bigtom_all", "bigtom_false": "bigtom_false", "tomi": "tomi"}
    primary_key = key_map.get(a.primary, a.primary)

    rows = join_rows(load_tom(), load_behaviour())
    print(f"joined {len(rows)} models")

    measures = [("bigtom_false", "bigtom|false_belief"),
                ("bigtom_all", "bigtom"),
                ("tomi", "tomi")]

    results = []
    deltas_primary = []
    for mk, label in measures:
        results.append(confound_demo(rows, mk, label))
        results.append(analysis_a(rows, mk, label))
        results.append(analysis_b(rows, mk, label))
        deltas, cres = analysis_c(rows, mk, label)
        results.append(cres)
        if mk == primary_key:
            deltas_primary = deltas

    os.makedirs(TOMDIR, exist_ok=True)
    csv_path = os.path.join(TOMDIR, "tom_vs_contrast.csv")
    with open(csv_path, "w", newline="") as fh:
        cols = ["model", "mtype", "size_B", "rating_std", "contrast",
                "bigtom_false", "bigtom_all", "tomi"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda z: -(z["bigtom_all"] or -1)):
            w.writerow(r)
        w.writerow({})
        w.writerow({"model": "ANALYSIS", "mtype": "measure", "size_B": "r_or_partial_r",
                    "rating_std": "ci_lo", "contrast": "ci_hi", "bigtom_false": "n",
                    "bigtom_all": "beta_tom", "tomi": "label"})
        for res in results:
            w.writerow({"model": res["analysis"], "mtype": res["measure"],
                        "size_B": res.get("r"), "rating_std": res.get("lo"),
                        "contrast": res.get("hi"), "bigtom_false": res.get("n"),
                        "bigtom_all": res.get("beta_tom", ""),
                        "tomi": res.get("label")})
        if deltas_primary:
            w.writerow({})
            w.writerow({"model": "WITHIN_FAMILY_DELTAS", "mtype": "family",
                        "size_B": "d_tom", "rating_std": "d_contrast",
                        "contrast": "base_tom", "bigtom_false": "inst_tom",
                        "bigtom_all": "base_contrast", "tomi": "inst_contrast"})
            for d in deltas_primary:
                w.writerow({"model": d["family"], "mtype": d["family"],
                            "size_B": d["d_tom"], "rating_std": d["d_contrast"],
                            "contrast": d["base_tom"], "bigtom_false": d["inst_tom"],
                            "bigtom_all": d["base_contrast"], "tomi": d["inst_contrast"]})
    print(f"  -> {csv_path}")

    scatter_panels(rows, deltas_primary, os.path.join(TOMDIR, "tom_vs_contrast.png"))

    # Markdown report
    md = [
        "# ToM benchmark performance vs intent use in moral judgment (J1)", "",
        "## Question", "",
        "Does standard ToM-benchmark performance predict whether a model weights intent in",
        "graded moral judgment — once the base/instruct (and size) confound is controlled?",
        "",
        "## Why the all-20 correlation is not a result", "",
        "Both axes are proxies for model type. Base models score near floor on BigToM",
        "because they cannot follow the QA format, and sit near zero on the 2×2 contrast.",
        "Instruction tuning and scale move both axes. The unrestricted Pearson r",
        "(e.g. BigToM–contrast ≈ −0.74 over 20 models) demonstrates that confound; it is",
        "**not reported as a finding**. The three analyses below are.",
        "",
        "## Floor policy", "",
        "Correlation analyses keep **every** model. The derived `rating_std` floor",
        "(0.2191) is for engagement / anchor counts only — see",
        "`outputs/stats/FLOOR_DERIVATION.md`. Using it here would select on a variable",
        "adjacent to the outcome. The fix for the confound is controlling for type, not",
        "excluding models.",
        "",
        "## Ceiling gate", "",
        "| benchmark | accuracies (0.5B-I / 14B-I / OLMo-I) | spread | verdict |",
        "|---|---|---|---|",
        "| BigToM | 0.520 / 0.882 / 0.850 | 0.362 | spread, proceed |",
        "| ToMi | 0.482 / 0.512 / 0.818 | 0.335 | spread, proceed |",
        "",
        "## Controlled results", "",
        "| analysis | ToM measure | estimate | 95% CI | n | reading |",
        "|---|---|---|---|---|---|",
    ]
    for res in results:
        if res["analysis"].startswith("(confound"):
            continue
        est = res.get("r")
        lo, hi = res.get("lo"), res.get("hi")
        extra = ""
        if res.get("beta_tom") is not None and not (isinstance(res["beta_tom"], float)
                                                     and math.isnan(res["beta_tom"])):
            extra = f" (β_tom={res['beta_tom']:+.3f}, p={res['p_tom']:.3g})"
        md.append(
            f"| {res['analysis']} | {res['measure']} | "
            f"{est:+.3f}{extra} | [{lo:+.3f}, {hi:+.3f}] | {res['n']} | {res['label']} |"
        )

    md += ["", "### Confound demonstration (not a result)", "",
           "| analysis | ToM measure | r | 95% CI | n |",
           "|---|---|---|---|---|"]
    for res in results:
        if res["analysis"].startswith("(confound"):
            md.append(f"| {res['analysis']} | {res['measure']} | {res['r']:+.3f} | "
                      f"[{res['lo']:+.3f}, {res['hi']:+.3f}] | {res['n']} |")

    md += ["", "## Within-family deltas (primary measure: BigToM all)", "",
           "| family | Δ ToM (I−B) | Δ contrast (I−B) | base contrast | instruct contrast |",
           "|---|---|---|---|---|"]
    for d in deltas_primary:
        md.append(f"| {d['family']} | {d['d_tom']:+.3f} | {d['d_contrast']:+.3f} | "
                  f"{d['base_contrast']:+.3f} | {d['inst_contrast']:+.3f} |")

    md += ["", "## Per-model table", "",
           "| model | type | params | BigToM FB | BigToM all | ToMi | contrast |",
           "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda z: -(z["bigtom_all"] or -1)):
        f = lambda v: "—" if v is None else f"{v:.3f}"
        md.append(f"| {r['model']} | {r['mtype']} | {r['size_B'] or '—'} | "
                  f"{f(r['bigtom_false'])} | {f(r['bigtom_all'])} | {f(r['tomi'])} | "
                  f"{f(r['contrast'])} |")

    md += ["", "## Reading", "",
           "Report (a), (b), and (c). If all three are null or negative, ToM-benchmark",
           "performance does not predict intent-weighting in moral judgment once type is",
           "held constant — the dissociation is measured on our own models. A positive",
           "result under (a)/(b) or a positive delta–delta under (c) would weaken that claim.",
           "Do not cite the all-20 r.", ""]

    md_path = os.path.join(TOMDIR, "TOM_VS_CONTRAST.md")
    open(md_path, "w").write("\n".join(md))
    print(f"  -> {md_path}")
    for res in results:
        print(f"  {res['analysis']:40} {res['measure']:22} "
              f"r={res.get('r', float('nan')):+.3f} n={res['n']}")


if __name__ == "__main__":
    main()
