#!/usr/bin/env python3
"""
33_prompt_factorial_analysis.py — W5 / roadmap #3 analysis half.

After the designed factorial has been scored (see 03_behavioral.FACTORIAL_TEMPLATES):

1. --alias-legacy : copy wrong_w1→para_wrong7 and punish_w1→punish7 rows into each
   model's raw/item_means CSVs so overnight IRI keys stay comparable (identical wording;
   we do not double-score on GPU).
2. Sign-stability table under the pre-registered inclusion rule: a model enters the
   pooled factorial mean only if every factorial paraphrase (the 6 × 1–7 prompts)
   shares the same contrast sign. Sign-flippers are reported separately, never averaged in.
3. Variance decomposition of contrast into wording × construct (+ scale when
   legacy scales are included), with prompt/template as a random intercept
   (statsmodels MixedLM when available; otherwise OLS Type-II SS ANOVA).

Cites (does not recompute) the YS2008↔YS2009 scale-replication result:
  pooled r≈0.71, Bland–Altman bias ≈ −0.06 (outputs/SCALE_REPLICATION.md).

Outputs
  outputs/analysis/prompt_factorial_sign_stability.csv
  outputs/analysis/prompt_factorial_variance.csv
  outputs/analysis/prompt_factorial_report.md
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

import numpy as np

CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CODE)
import tom_common as tc  # noqa: E402
import importlib.util

spec = importlib.util.spec_from_file_location("beh", os.path.join(CODE, "03_behavioral.py"))
beh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(beh)

FACTORIAL_1_7 = [t for t in beh.FACTORIAL_TEMPLATES if t != "human_verbatim"]
OUT = os.path.join(tc.ROOT, "outputs", "analysis")
SCALE_REP_NOTE = (
    "Scale replication (YS2008↔YS2009 human_verbatim): pooled r≈0.71, "
    "Bland–Altman bias ≈ −0.06 — see outputs/SCALE_REPLICATION.md. "
    "Not recomputed here."
)


def _alias_file(path: str, src_tmpl: str, dst_tmpl: str) -> int:
    """Append dst_tmpl rows copied from src_tmpl if dst missing. Returns n added."""
    if not os.path.isfile(path):
        return 0
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return 0
    have_dst = {(r.get("story_id"), r.get("sample", "0"))
                for r in rows if r.get("template") == dst_tmpl}
    src = [r for r in rows if r.get("template") == src_tmpl]
    if not src:
        return 0
    added = []
    for r in src:
        key = (r.get("story_id"), r.get("sample", "0"))
        if key in have_dst:
            continue
        nr = dict(r)
        nr["template"] = dst_tmpl
        added.append(nr)
        have_dst.add(key)
    if not added:
        return 0
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows + added)
    return len(added)


def alias_legacy(behavior_dir: str) -> None:
    n_files = 0
    n_rows = 0
    for src, dst in beh.LEGACY_ALIASES.items():
        for path in glob.glob(os.path.join(behavior_dir, "raw_*.csv")) + \
                    glob.glob(os.path.join(behavior_dir, "item_means_*.csv")):
            k = _alias_file(path, src, dst)
            if k:
                n_files += 1
                n_rows += k
                print(f"  aliased {src}->{dst}: +{k} rows in {os.path.basename(path)}")
    print(f"alias-legacy done: touched {n_files} files, +{n_rows} rows")


def template_contrast(cells, tmpl: str) -> float | None:
    scen = cells.get(tmpl)
    if not scen:
        return None
    diffs = [c["attempted"] - c["accidental"] for c in scen.values()
             if "attempted" in c and "accidental" in c]
    return float(np.mean(diffs)) if diffs else None


def sign_stability_table(studies=None):
    """Per model: contrasts on factorial 1–7 prompts; include only sign-stable."""
    rows = []
    studies = studies or tc.STUDIES

    for study, tag, path in tc.iter_item_means(studies):
        cells = tc.load_cells(path)
        tcs = {t: template_contrast(cells, t) for t in FACTORIAL_1_7}
        tcs = {t: v for t, v in tcs.items() if v is not None}
        if len(tcs) < 2:
            continue
        vals = list(tcs.values())
        signs = {(1 if v > 1e-9 else (-1 if v < -1e-9 else 0)) for v in vals}
        nonzero = {s for s in signs if s != 0}
        flip = len(nonzero) > 1
        # Pre-registered inclusion: all non-zero factorial prompts share a sign.
        include = (not flip) and any(abs(v) > 1e-9 for v in vals)
        rows.append(dict(
            model=tc.pretty(tag), study=study, tag=tag, path=path,
            n_factorial=len(tcs),
            **{f"c_{t}": round(tcs[t], 4) for t in FACTORIAL_1_7 if t in tcs},
            contrast_mean_all=round(float(np.mean(vals)), 4),
            contrast_mean_included=round(float(np.mean(vals)), 4) if include else "",
            sign_stable=not flip,
            include_in_pooled=include,
            verdict=("sign-stable" if include else
                     ("FRAGILE (sign flip)" if flip else "degenerate/zero")),
        ))
    return rows


def variance_decomposition(sign_rows, sign_stable_only=False):
    """
    Long-format observations = model × factorial template contrast.
    Fixed: C(wording) * C(construct); random: template (prompt).

    sign_stable_only=False is the PRIMARY analysis. The pre-registered inclusion
    criterion governs the pooled *mean contrast* (flippers are not averaged in);
    applying it here too would discard exactly the prompt-driven variance this
    model exists to quantify, biasing the decomposition toward stability. The
    filtered version is reported alongside as a sensitivity check.

    Scale has no variance inside the 1–7 factorial — report that explicitly and
    optionally add para_blame10 / para_blame4 as a scale check (separate block).
    """
    long = []
    for r in sign_rows:
        if sign_stable_only and not r["include_in_pooled"]:
            continue
        cells = tc.load_cells(r["path"])
        for tmpl in FACTORIAL_1_7:
            c = template_contrast(cells, tmpl)
            if c is None:
                continue
            meta = beh.TEMPLATE_META[tmpl]
            long.append(dict(
                model=r["model"], template=tmpl, contrast=c,
                wording=int(meta["wording"]), construct=meta["construct"],
                scale=meta["scale"],
            ))
    if len(long) < 6:
        return long, {"error": f"too few observations ({len(long)})",
                      "sign_stable_only": sign_stable_only}

    import pandas as pd
    df = pd.DataFrame(long)
    summary = {
        "sign_stable_only": sign_stable_only,
        "n_obs": len(df),
        "n_models": df["model"].nunique(),
        "n_templates": df["template"].nunique(),
        "note_scale": "All factorial paraphrases are 1–7; scale factor has no within-factorial variance. "
                      "Scale effects are assessed via legacy para_blame10 (1–10) / para_blame4 (1–4) "
                      f"separately. {SCALE_REP_NOTE}",
    }

    # Type II ANOVA via OLS (wording + construct + wording:construct)
    try:
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm
        ols = smf.ols("contrast ~ C(wording) * C(construct)", data=df).fit()
        aov = anova_lm(ols, typ=2)
        summary["anova_typeII"] = aov.round(4).to_dict()
        # Share of total SS per term (terms are the ROWS of the ANOVA table).
        ss = aov["sum_sq"].to_dict()
        total = sum(v for v in ss.values() if v == v)  # skip NaN
        if total > 0:
            summary["variance_share"] = {k: round(v / total, 4)
                                         for k, v in ss.items() if v == v}
        # MixedLM: contrast ~ wording * construct + (1|template)
        try:
            md = smf.mixedlm("contrast ~ C(wording) * C(construct)", df,
                             groups=df["template"])
            mdf = md.fit(reml=True, method="lbfgs")
            summary["mixedlm_converged"] = bool(mdf.converged)
            summary["mixedlm_params"] = {k: round(float(v), 4)
                                         for k, v in mdf.params.items()}
            summary["mixedlm_template_var"] = round(float(mdf.cov_re.iloc[0, 0]), 6)
            summary["mixedlm_resid_var"] = round(float(mdf.scale), 6)
        except Exception as e:
            summary["mixedlm_error"] = str(e)
    except Exception as e:
        summary["anova_error"] = str(e)

    return long, summary


def write_report(sign_rows, var_summary, out_dir, var_summary_filtered=None):
    os.makedirs(out_dir, exist_ok=True)
    # sign stability csv
    keys = ["model", "study", "n_factorial"] + [f"c_{t}" for t in FACTORIAL_1_7] + [
        "contrast_mean_all", "contrast_mean_included", "sign_stable",
        "include_in_pooled", "verdict",
    ]
    sp = os.path.join(out_dir, "prompt_factorial_sign_stability.csv")
    with open(sp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in sign_rows:
            w.writerow(r)

    vp = os.path.join(out_dir, "prompt_factorial_variance.csv")
    with open(vp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["analysis", "key", "value"])
        for k, v in var_summary.items():
            w.writerow(["primary_all_models", k, v])
        for k, v in (var_summary_filtered or {}).items():
            w.writerow(["sensitivity_sign_stable_only", k, v])

    n_in = sum(1 for r in sign_rows if r["include_in_pooled"])
    n_flip = sum(1 for r in sign_rows if r["verdict"].startswith("FRAGILE"))
    md = os.path.join(out_dir, "prompt_factorial_report.md")
    lines = [
        "# Prompt factorial analysis (W5 / roadmap #3)",
        "",
        "## Template set",
        "",
        "Designed 7 = `human_verbatim` + 2 wordings × 3 constructs "
        "(`blame` / `wrongness` / `punishment`) on a common 1–7 scale "
        "(`blame_w1/w2`, `wrong_w1/w2`, `punish_w1/w2`).",
        "",
        "`para_blame10` and other legacy templates remain **additive** "
        "(never replaced). `wrong_w1`/`punish_w1` alias to overnight "
        "`para_wrong7`/`punish7` (identical wording).",
        "",
        f"## Sign stability (pre-registered inclusion)",
        "",
        f"- Models scored on factorial 1–7 prompts: **{len(sign_rows)}**",
        f"- Included in pooled factorial mean (sign-stable): **{n_in}**",
        f"- Sign-flippers (reported separately, not averaged in): **{n_flip}**",
        "",
        "## Variance decomposition",
        "",
        "Fixed effects: `C(wording) * C(construct)` on contrast; "
        "random intercept: template/prompt (MixedLM when available).",
        "",
        "The pre-registered sign-stability rule governs the pooled **mean contrast**, "
        "not this model: filtering flippers out of a variance decomposition would "
        "discard the prompt-driven variance it exists to quantify. Primary = all "
        "models; sign-stable-only is reported as a sensitivity check.",
        "",
        "### Primary (all models)",
        "",
        "```",
        f"{var_summary}",
        "```",
        "",
        "### Sensitivity (sign-stable models only)",
        "",
        "```",
        f"{var_summary_filtered}",
        "```",
        "",
        f"## Scale replication (cited, not recomputed)",
        "",
        SCALE_REP_NOTE,
        "",
        f"Wrote `{os.path.relpath(sp, tc.ROOT)}`, "
        f"`{os.path.relpath(vp, tc.ROOT)}`.",
    ]
    open(md, "w").write("\n".join(lines) + "\n")
    print(f"wrote {sp}")
    print(f"wrote {vp}")
    print(f"wrote {md}")
    print(f"sign-stable included={n_in}  flippers={n_flip}  total={len(sign_rows)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alias-legacy", action="store_true",
                    help="copy wrong_w1→para_wrong7 and punish_w1→punish7 into behavior CSVs")
    ap.add_argument("--behavior", default=os.path.join(tc.ROOT, "outputs", "behavior"))
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--analyze", action="store_true",
                    help="sign-stability + variance decomposition")
    ap.add_argument("--studies-dir", nargs="*", default=None,
                    help="override behavior dirs scanned for item_means (default: tom_common.STUDIES)")
    a = ap.parse_args()
    if not a.alias_legacy and not a.analyze:
        a.alias_legacy = a.analyze = True
    if a.alias_legacy:
        alias_legacy(a.behavior)
    if a.analyze:
        studies = ({os.path.basename(d.rstrip("/")) or d: d for d in a.studies_dir}
                   if a.studies_dir else None)
        rows = sign_stability_table(studies)
        _, summary = variance_decomposition(rows, sign_stable_only=False)
        _, summary_filt = variance_decomposition(rows, sign_stable_only=True)
        write_report(rows, summary, a.out, summary_filt)


if __name__ == "__main__":
    main()
