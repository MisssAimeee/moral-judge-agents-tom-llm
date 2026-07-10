#!/usr/bin/env python3
"""
11_interaction_regression.py  --  Roadmap #2: the full 2x2 intent x outcome
regression (ANALYSIS ONLY — no model inference, no training).

The headline `contrast = blame(attempted) - blame(accidental)` uses only 2 of the
4 conditions. The real human "moral-luck" fingerprint lives in the full 2x2:

    norm_blame = b0 + b_intent*I + b_outcome*O + b_interaction*(I*O)

  b0            = neutral baseline (innocent belief, no harm)
  b_intent      = extra blame for a GUILTY belief   (attempted - neutral)
  b_outcome     = extra blame for a bad OUTCOME      (accidental - neutral)
  b_interaction = intentional - attempted - accidental + neutral

The adult signature is a LARGE intent effect, a SMALL outcome effect, and a
NEGATIVE (sub-additive) interaction: once intent is present, adding a bad outcome
barely raises blame ("attempted" is judged almost as harshly as "intentional").
A young child shows the opposite — outcome dominates.

For every model we report:
  * the four coefficients (pooled over the 3 prompt templates),
  * scenario-bootstrap 95% CIs on each coefficient,
  * mixed-model p-values (norm ~ intent*outcome, scenario random intercept) if
    statsmodels is available,
  * the same decomposition for the human adult / child reference profiles.

Outputs:
  outputs/analysis/interaction_regression.csv
  console table
"""
import os, csv, math, argparse
import numpy as np
import tom_common as tc


def cell_means(cells_scen):
    """Mean per condition across scenarios (pooled)."""
    acc = {c: [] for c in tc.CELLS}
    for conds in cells_scen.values():
        for c in tc.CELLS:
            if c in conds:
                acc[c].append(conds[c])
    return {c: (float(np.mean(v)) if v else float("nan")) for c, v in acc.items()}


def coeffs_from_means(m):
    """Solve the saturated 2x2 for the four regression coefficients."""
    b0 = m["neutral"]
    b_int = m["attempted"] - m["neutral"]
    b_out = m["accidental"] - m["neutral"]
    b_inter = m["intentional"] - m["attempted"] - m["accidental"] + m["neutral"]
    return b0, b_int, b_out, b_inter


def boot_coeff(cells_scen, which, B, seed):
    """Bootstrap CI over scenarios for one coefficient ('b0','int','out','inter')."""
    scen = list(cells_scen)

    def stat(keys):
        acc = {c: [] for c in tc.CELLS}
        for k in keys:
            conds = cells_scen.get(k, {})
            for c in tc.CELLS:
                if c in conds:
                    acc[c].append(conds[c])
        m = {c: (np.mean(v) if v else np.nan) for c, v in acc.items()}
        b0, bi, bo, bx = coeffs_from_means(m)
        return {"b0": b0, "int": bi, "out": bo, "inter": bx}[which]

    return tc.bootstrap(scen, stat, B=B, seed=seed)


def mixed_pvalues(rows):
    """norm ~ intent*outcome with scenario random intercept -> p-values dict."""
    try:
        import statsmodels.formula.api as smf
        import pandas as pd
    except Exception:
        return {}
    df = pd.DataFrame(rows)
    if df["scenario"].nunique() < 3:
        return {}
    try:
        md = smf.mixedlm("norm ~ intent*outcome", df, groups=df["scenario"]).fit(reml=False)
        return {
            "p_intent": float(md.pvalues.get("intent", float("nan"))),
            "p_outcome": float(md.pvalues.get("outcome", float("nan"))),
            "p_interaction": float(md.pvalues.get("intent:outcome", float("nan"))),
        }
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(tc.ROOT, "outputs", "analysis"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    registry = tc.load_registry()

    rows_out = []

    # -- human reference profiles first (adult + child bands) --
    for g, prof in tc.human_profiles().items():
        if all(c in prof for c in tc.CELLS):
            b0, bi, bo, bx = coeffs_from_means(prof)
            rows_out.append(dict(model=f"HUMAN {g}", study="reference", size="",
                                 mtype="human", b0=b0, b_int=bi, b_out=bo, b_inter=bx,
                                 int_lo="", int_hi="", out_lo="", out_hi="",
                                 inter_lo="", inter_hi="",
                                 p_intent="", p_outcome="", p_interaction=""))

    # -- every tested model, both studies --
    for study, tag, path in tc.iter_item_means():
        cells = tc.load_cells(path)
        pooled = tc.pooled_cells(cells)
        if len(pooled) < 3:
            continue
        m = cell_means(pooled)
        if any(math.isnan(m[c]) for c in tc.CELLS):
            continue
        b0, bi, bo, bx = coeffs_from_means(m)
        _, i_lo, i_hi = boot_coeff(pooled, "int", a.boot, 1)
        _, o_lo, o_hi = boot_coeff(pooled, "out", a.boot, 2)
        _, x_lo, x_hi = boot_coeff(pooled, "inter", a.boot, 3)
        size, mtype, _ = tc.parse_tag(tag, registry)
        pv = mixed_pvalues(tc.load_rows(path))
        rows_out.append(dict(
            model=tc.pretty(tag), study=study, size=size, mtype=mtype,
            b0=b0, b_int=bi, b_out=bo, b_inter=bx,
            int_lo=i_lo, int_hi=i_hi, out_lo=o_lo, out_hi=o_hi,
            inter_lo=x_lo, inter_hi=x_hi,
            p_intent=pv.get("p_intent", ""), p_outcome=pv.get("p_outcome", ""),
            p_interaction=pv.get("p_interaction", "")))

    # sort: humans first (by intent effect), then models by intent effect desc
    def key(r):
        return (0 if r["study"] == "reference" else 1, -(r["b_int"] or 0))
    rows_out.sort(key=key)

    # -------- console --------
    print("\n=== 2x2 INTENT x OUTCOME REGRESSION  (norm ~ intent*outcome, pooled prompts) ===")
    print("adult fingerprint = big b_intent, small b_outcome, NEGATIVE (sub-additive) b_interaction\n")
    print(f"{'model':30} {'b_intent':>18} {'b_outcome':>18} {'b_interaction':>20}")
    for r in rows_out:
        def fmt(v, lo, hi):
            if isinstance(lo, float):
                return f"{v:+.3f}[{lo:+.2f},{hi:+.2f}]"
            return f"{v:+.3f}"
        print(f"{r['model'][:30]:30} "
              f"{fmt(r['b_int'], r['int_lo'], r['int_hi']):>18} "
              f"{fmt(r['b_out'], r['out_lo'], r['out_hi']):>18} "
              f"{fmt(r['b_inter'], r['inter_lo'], r['inter_hi']):>20}")

    # -------- csv --------
    out_csv = os.path.join(a.out, "interaction_regression.csv")
    cols = ["model", "study", "size", "mtype", "b0", "b_int", "b_out", "b_inter",
            "int_lo", "int_hi", "out_lo", "out_hi", "inter_lo", "inter_hi",
            "p_intent", "p_outcome", "p_interaction"]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "study", "size_B", "type", "b0_neutral", "b_intent",
                    "b_outcome", "b_interaction", "b_intent_lo", "b_intent_hi",
                    "b_outcome_lo", "b_outcome_hi", "b_interaction_lo",
                    "b_interaction_hi", "p_intent", "p_outcome", "p_interaction"])
        for r in rows_out:
            w.writerow([_r(r[c]) for c in cols])
    print(f"\nwrote {os.path.relpath(out_csv, tc.ROOT)}  ({len(rows_out)} rows incl. humans)")


def _r(x):
    if isinstance(x, float):
        return "" if math.isnan(x) else round(x, 4)
    return x


if __name__ == "__main__":
    main()
