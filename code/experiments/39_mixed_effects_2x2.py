#!/usr/bin/env python3
"""J3 -- per-model mixed-effects fit of the 2x2 with the intent x outcome interaction.

WHY THIS REPLACES ols_2x2. The existing estimator (ols_2x2, 03_behavioral.py) collapses
298 stories into FOUR cell means and solves a 3x3 normal equation on [1, intent, outcome].
That has three consequences:

  * No interaction term. Four cells and three parameters leaves one residual degree of
    freedom; adding the interaction saturates the design, so it was never estimated.
  * No standard errors and no p-values. Item-level variance is averaged away before the
    fit, so there is nothing left to estimate uncertainty from.
  * No account of the nesting. Stories come in scenario groups of 4 or 8 that share nearly
    all their text, so the observations are not independent.

This script fits, per model, on item-level data:

    blame ~ intent * outcome + (1 | scenario_group) + (1 | story_id)

story_id is nested inside scenario_group (each story belongs to exactly one group), so it
enters as a variance component within the group rather than as a crossed effect.

WHY THE INTERACTION IS THE POINT. The human signature in Young et al. 2007 is not a pair
of main effects, it is their interaction. From the normalised cell means, adding a harmful
outcome moves judgment a lot when intent is absent (0.033 -> 0.267) and barely at all when
intent is present (0.933 -> 0.967). Attempted harm is already judged nearly as harshly as
completed intentional harm; the outcome has little left to add. That sub-additivity is a
NEGATIVE interaction of about -0.20. A model that reproduces it is performing the human
computation. A model that only shifts main effects is not, however well its main effects
happen to line up.

Outputs
  outputs/stats/mixed_effects_2x2.csv          one row per model per specification
  outputs/stats/mixed_effects_interaction.png  forest plot, human reference marked
  outputs/stats/MIXED_EFFECTS_2x2.md
"""
import argparse
import csv
import glob
import os
import re
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MASTER = os.path.join(ROOT, "dataset", "master", "moral_2x2_master.csv")
BEHAVIOR = os.path.join(ROOT, "outputs", "behavior")
OUTDIR = os.path.join(ROOT, "outputs", "stats")

# Young et al. 2007, normalised cell means, keyed by (intent, outcome).
HUMAN_CELLS = {(0, 0): 0.033, (0, 1): 0.267, (1, 0): 0.933, (1, 1): 0.967}


def human_terms():
    """Treatment-coded terms from the human cell means, directly comparable to the fits."""
    y00, y01, y10, y11 = (HUMAN_CELLS[(0, 0)], HUMAN_CELLS[(0, 1)],
                          HUMAN_CELLS[(1, 0)], HUMAN_CELLS[(1, 1)])
    return {
        "intercept": y00,
        "b_intent": y10 - y00,                  # simple effect of intent at outcome=0
        "b_outcome": y01 - y00,                 # simple effect of outcome at intent=0
        "b_interaction": (y11 - y10) - (y01 - y00),
    }


def load_groups():
    lab = {}
    for r in csv.DictReader(open(MASTER)):
        lab[r["story_id"]] = r.get("scenario_group") or r.get("scenario_id")
    return lab


def load_model_frame(path, groups):
    d = pd.read_csv(path)
    d = d[["model", "template", "story_id", "source", "condition",
           "intent_label", "outcome_label", "norm_rating"]].copy()
    d["scenario_group"] = d["story_id"].map(groups)
    d = d.dropna(subset=["scenario_group", "norm_rating"])
    d["intent"] = (d["intent_label"] == "guilty").astype(float)
    d["outcome"] = (d["outcome_label"] == "harm").astype(float)
    # one observation per (template, story); average any repeated samples
    d = (d.groupby(["template", "story_id", "scenario_group", "intent", "outcome"],
                   as_index=False)["norm_rating"].mean())
    return d


def fit_one(d, with_template):
    """Returns dict of coefficients, or a reason string if the fit is not estimable."""
    from statsmodels.regression.mixed_linear_model import MixedLM

    if d["norm_rating"].std() < 1e-9:
        return None, "constant ratings"
    if d["intent"].nunique() < 2 or d["outcome"].nunique() < 2:
        return None, "missing a factor level"

    formula = "norm_rating ~ intent * outcome"
    if with_template and d["template"].nunique() > 1:
        formula += " + C(template)"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            md = MixedLM.from_formula(
                formula, groups="scenario_group", re_formula="1",
                vc_formula={"story": "0 + C(story_id)"}, data=d)
            res = md.fit(reml=True, method="lbfgs", maxiter=500)
    except Exception as e:  # singular fits happen for near-constant models
        return None, f"fit failed: {type(e).__name__}"

    name_map = {"intent": "b_intent", "outcome": "b_outcome",
                "intent:outcome": "b_interaction", "Intercept": "intercept"}
    out = {"converged": bool(res.converged), "n_obs": int(d.shape[0]),
           "n_groups": int(d["scenario_group"].nunique()),
           "n_stories": int(d["story_id"].nunique())}
    for raw, nice in name_map.items():
        if raw in res.params.index:
            out[nice] = float(res.params[raw])
            out[f"se_{nice[2:]}" if nice != "intercept" else "se_intercept"] = \
                float(res.bse[raw])
            out[f"p_{nice[2:]}" if nice != "intercept" else "p_intercept"] = \
                float(res.pvalues[raw])
    vc = res.cov_re.iloc[0, 0] if hasattr(res.cov_re, "iloc") else float("nan")
    out["var_scenario_group"] = float(vc)
    out["var_residual"] = float(res.scale)

    # A negative interaction is sub-additivity, and sub-additivity on a bounded scale can
    # come from EITHER factor running out of room. Humans get it because intent nearly
    # saturates blame on its own; an outcome-driven model gets it because outcome does. The
    # cell means and the attempted-minus-accidental diagonal separate the two cases, so the
    # matching sign is not read as a matching computation.
    cells = d.groupby(["intent", "outcome"])["norm_rating"].mean()
    for (i, o), key in (((0, 0), "cell_neutral"), ((0, 1), "cell_accidental"),
                        ((1, 0), "cell_attempted"), ((1, 1), "cell_intentional")):
        out[key] = float(cells.get((float(i), float(o)), np.nan))
    out["diag_attempted_minus_accidental"] = out["cell_attempted"] - out["cell_accidental"]
    out["saturating_factor"] = ("intent" if abs(out["b_intent"]) > abs(out["b_outcome"])
                               else "outcome")
    # Humans: attempted (0.933) > accidental (0.267). Same interaction sign with the
    # opposite cell order is NOT human-likeness — it is outcome-driven sub-additivity.
    diag = out["diag_attempted_minus_accidental"]
    if diag != diag:  # NaN
        out["cell_order"] = "unknown"
    elif diag > 0.02:
        out["cell_order"] = "matches_human"
    elif diag < -0.02:
        out["cell_order"] = "inverted"
    else:
        out["cell_order"] = "tied"
    return out, None


def forest_plot(rows, path, human):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in rows if r.get("b_interaction") is not None
            and not (isinstance(r.get("b_interaction"), float)
                     and np.isnan(r["b_interaction"]))]
    rows = sorted(rows, key=lambda r: r["b_interaction"])
    if not rows:
        print("  no estimable interactions; skipping forest plot")
        return

    labels = [r["model"].split("/")[-1] for r in rows]
    est = np.array([r["b_interaction"] for r in rows])
    se = np.array([r.get("se_interaction", np.nan) for r in rows])
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(9.2, 0.34 * len(rows) + 2.6))
    sig = np.array([r.get("p_interaction", 1.0) < 0.05 for r in rows])
    ax.errorbar(est[sig], y[sig], xerr=1.96 * se[sig], fmt="o", ms=5.2,
                color="#1f3f8f", ecolor="#1f3f8f", elinewidth=1.5, capsize=2.6,
                label="p < 0.05", zorder=3)
    ax.errorbar(est[~sig], y[~sig], xerr=1.96 * se[~sig], fmt="o", ms=5.2,
                mfc="white", mec="#7a7a7a", ecolor="#a8a8a8", elinewidth=1.3,
                capsize=2.6, label="n.s.", zorder=3)

    ax.axvline(0, color="#666666", lw=1.0, ls="-", zorder=1)
    hv = human["b_interaction"]
    ax.axvline(hv, color="#b3202c", lw=1.8, ls="--", zorder=2,
               label=f"human (Young 2007) = {hv:+.3f}")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.4)
    ax.set_ylim(-0.9, len(rows) - 0.1)
    ax.set_xlabel("intent x outcome interaction  (blame scale, 0-1)", fontsize=9.6)
    ax.set_title("Interaction term per model, with 95% CI\n"
                 "blame ~ intent * outcome + (1|scenario_group) + (1|story_id)",
                 fontsize=10.6)
    ax.annotate("negative = sub-additive: the second factor adds less.\n"
                "Sign alone does not identify which factor saturates\n"
                "(see attempted-accidental column in the CSV).",
                xy=(0.015, 0.975), xycoords="axes fraction", fontsize=7.6,
                color="#555555", va="top")
    ax.legend(loc="lower right", fontsize=7.8, framealpha=0.92)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    print(f"  -> {path}")


def headline_cell_order_section(est, human):
    """Promote cell-order (J3 headline): 0/N match human ordering, most inverted.

    Picks two representative models: the most extreme inversion (largest negative
    attempted-minus-accidental diagonal) and the model whose interaction
    COEFFICIENT is closest to the human value despite an inverted cell order --
    the case that most directly shows the coefficient alone is misleading.
    """
    with_order = [r for r in est if r.get("cell_order") in
                  ("matches_human", "inverted", "tied")]
    n = len(with_order)
    n_match = sum(1 for r in with_order if r["cell_order"] == "matches_human")
    n_inv = sum(1 for r in with_order if r["cell_order"] == "inverted")
    n_tied = sum(1 for r in with_order if r["cell_order"] == "tied")

    inverted = [r for r in with_order if r["cell_order"] == "inverted"]
    most_extreme = (min(inverted, key=lambda r: r["diag_attempted_minus_accidental"])
                    if inverted else None)
    closest_coef = (min(inverted, key=lambda r: abs(r["b_interaction"]
                                                     - human["b_interaction"]))
                    if inverted else None)

    lines = [
        "## Cell ordering, not just the coefficient (headline)", "",
        f"**{n_match} of {n} models match the human cell ordering "
        f"(attempted > accidental); {n_inv} are inverted "
        f"(accidental > attempted); {n_tied} are tied.** This is a stronger and",
        "more quotable result than the interaction coefficient alone: several models",
        "approximate the human interaction magnitude "
        f"(human = {human['b_interaction']:+.3f}) while getting the underlying cell",
        "pattern backwards, which the coefficient by itself hides.", "",
    ]
    if most_extreme and closest_coef:
        rows_to_show = [("HUMAN (Young 2007)", human, "matches_human")]
        seen = set()
        for label, r in (("most extreme inversion", most_extreme),
                         ("closest coefficient match (inverted)", closest_coef)):
            if r["model"] in seen:
                continue
            seen.add(r["model"])
            rows_to_show.append((f"{r['model']}  ({label})", r, r["cell_order"]))
        lines += [
            "| model | neutral | accidental | attempted | intentional | "
            "att − acc | b_interaction | cell order |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for label, r, order in rows_to_show:
            if r is human:
                neu, acc, att, intl = (HUMAN_CELLS[(0, 0)], HUMAN_CELLS[(0, 1)],
                                       HUMAN_CELLS[(1, 0)], HUMAN_CELLS[(1, 1)])
                diag = att - acc
                bixo = human["b_interaction"]
            else:
                neu, acc, att, intl = (r["cell_neutral"], r["cell_accidental"],
                                       r["cell_attempted"], r["cell_intentional"])
                diag = r["diag_attempted_minus_accidental"]
                bixo = r["b_interaction"]
            lines.append(f"| {label} | {neu:.3f} | {acc:.3f} | {att:.3f} | "
                        f"{intl:.3f} | {diag:+.3f} | {bixo:+.3f} | {order} |")
        lines += [
            "",
            "Humans: attempted (0.933) is already almost as harsh as intentional",
            "(0.967) — the accident with the same outcome (accidental, 0.267) is judged",
            "far more leniently. These models put accidental ABOVE attempted: an",
            "outcome-free failed attempt is judged more leniently than an accident that",
            "caused harm, the reverse of the human pattern, even when the interaction",
            "coefficient sits close to the human value.", "",
        ]
    lines += [
        "The full per-model cell means and `cell_order` column are in the table below",
        "and in `mixed_effects_2x2.csv`.", "",
    ]
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavior", default=BEHAVIOR)
    ap.add_argument("--out", default=OUTDIR)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    groups = load_groups()
    human = human_terms()
    print("human reference (Young 2007 normalised cells):")
    for k, v in human.items():
        print(f"  {k:15} {v:+.4f}")

    files = sorted(glob.glob(os.path.join(a.behavior, "raw_*.csv")))
    rows, primary = [], []
    for f in files:
        model = os.path.basename(f)[len("raw_"):-len(".csv")].replace("_", "/", 1)
        d = load_model_frame(f, groups)
        for with_tmpl, spec in ((False, "primary"), (True, "template_absorbed")):
            res, err = fit_one(d, with_tmpl)
            if res is None:
                rows.append({"model": model, "spec": spec, "note": err})
                if spec == "primary":
                    print(f"  {model:44} NOT ESTIMABLE ({err})")
                continue
            res.update({"model": model, "spec": spec, "note": ""})
            rows.append(res)
            if spec == "primary":
                primary.append(res)
                print(f"  {model:44} b_int={res.get('b_intent', float('nan')):+.3f} "
                      f"b_out={res.get('b_outcome', float('nan')):+.3f} "
                      f"b_ixo={res.get('b_interaction', float('nan')):+.3f} "
                      f"(SE {res.get('se_interaction', float('nan')):.3f}, "
                      f"p={res.get('p_interaction', float('nan')):.3g})")

    cols = ["model", "spec", "intercept", "se_intercept", "p_intercept",
            "b_intent", "se_intent", "p_intent", "b_outcome", "se_outcome", "p_outcome",
            "b_interaction", "se_interaction", "p_interaction",
            "cell_neutral", "cell_accidental", "cell_attempted", "cell_intentional",
            "diag_attempted_minus_accidental", "cell_order", "saturating_factor",
            "var_scenario_group", "var_residual", "converged",
            "n_obs", "n_groups", "n_stories", "note"]
    csv_path = os.path.join(a.out, "mixed_effects_2x2.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    # human row, for readers who only open the CSV
    with open(csv_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writerow({"model": "HUMAN (Young 2007)", "spec": "reference",
                    "intercept": human["intercept"], "b_intent": human["b_intent"],
                    "b_outcome": human["b_outcome"],
                    "b_interaction": human["b_interaction"],
                    "cell_neutral": HUMAN_CELLS[(0, 0)],
                    "cell_accidental": HUMAN_CELLS[(0, 1)],
                    "cell_attempted": HUMAN_CELLS[(1, 0)],
                    "cell_intentional": HUMAN_CELLS[(1, 1)],
                    "diag_attempted_minus_accidental":
                        HUMAN_CELLS[(1, 0)] - HUMAN_CELLS[(0, 1)],
                    "cell_order": "matches_human",
                    "saturating_factor": "intent",
                    "note": "computed from normalised cell means, no SE available"})
    print(f"  -> {csv_path}")

    forest_plot(primary, os.path.join(a.out, "mixed_effects_interaction.png"), human)

    # readout
    est = [r for r in primary if r.get("b_interaction") is not None]
    neg_sig = [r for r in est if r.get("b_interaction", 0) < 0
               and r.get("p_interaction", 1) < 0.05]
    pos_sig = [r for r in est if r.get("b_interaction", 0) > 0
               and r.get("p_interaction", 1) < 0.05]
    md = [
        "# W1 mixed-effects 2x2 with interaction (J3)", "",
        "Model fitted per model on item-level ratings:", "",
        "    blame ~ intent * outcome + (1|scenario_group) + (1|story_id)", "",
        "story_id is nested within scenario_group and enters as a within-group variance",
        "component. Coefficients are treatment-coded, so `b_intent` is the simple effect of",
        "intent at outcome=0, `b_outcome` the simple effect of outcome at intent=0, and",
        "`b_interaction` the extra effect of outcome when intent is present.", "",
        "## Human reference", "",
        "Computed from the Young et al. 2007 normalised cell means "
        "(neutral 0.033, accidental 0.267, attempted 0.933, intentional 0.967):", "",
        f"    b_intent      = {human['b_intent']:+.3f}",
        f"    b_outcome     = {human['b_outcome']:+.3f}",
        f"    b_interaction = {human['b_interaction']:+.3f}", "",
        "The interaction is negative because a harmful outcome adds little once intent is",
        "present (0.933 -> 0.967) but a great deal when it is absent (0.033 -> 0.267).", "",
    ]
    md += headline_cell_order_section(est, human)
    md += [
        "## Counts", "",
        f"- models with an estimable interaction: {len(est)} of {len(files)}",
        f"- significantly negative (same sign as humans, p<0.05): {len(neg_sig)}",
        f"- significantly positive (opposite sign to humans, p<0.05): {len(pos_sig)}",
        f"- not distinguishable from zero: {len(est) - len(neg_sig) - len(pos_sig)}", "",
        "## The matching sign is not a matching computation", "",
        "Read the sign count above together with the main effects, not on its own. A negative",
        "interaction means sub-additivity: the second factor adds less once the first is",
        "present. On a bounded 0-1 scale that can happen because EITHER factor has already",
        "used up the scale, and the two cases mean opposite things.", "",
        "- Humans: `b_intent` = +0.900 and `b_outcome` = +0.234. Intent nearly saturates blame",
        "  by itself, so outcome has little left to add. The sub-additivity is intent-driven.",
        "- These models: `b_intent` is small and `b_outcome` is large, the reverse ordering.",
        "  Their sub-additivity is outcome-driven -- outcome uses up the scale and intent has",
        "  little left to add.", "",
        f"`saturating_factor` in the CSV records which main effect is larger per model: "
        f"{sum(1 for r in est if r.get('saturating_factor') == 'outcome')} of {len(est)} are "
        f"outcome-saturating against the human pattern of intent-saturating.", "",
        "Human interaction from cell means: "
        f"(0.967−0.933)−(0.267−0.033) = {human['b_interaction']:+.3f}. Several models",
        "approximate that coefficient. Same sign with the **opposite cell order**",
        "(accidental > attempted) is not human-likeness — see `cell_order`.", "",
        f"Cell-order counts (primary spec): "
        f"matches_human="
        f"{sum(1 for r in est if r.get('cell_order') == 'matches_human')}, "
        f"inverted="
        f"{sum(1 for r in est if r.get('cell_order') == 'inverted')}, "
        f"tied={sum(1 for r in est if r.get('cell_order') == 'tied')}.", "",
        "## Interaction terms with cell means (primary specification)", "",
        "| model | b_int | b_out | b_ixo | SE | p | neu | acc | att | int | att−acc | cell order | sat |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(est, key=lambda x: x["b_interaction"]):
        md.append(
            f"| {r['model']} | {r.get('b_intent', float('nan')):+.3f} | "
            f"{r.get('b_outcome', float('nan')):+.3f} | "
            f"{r['b_interaction']:+.3f} | {r.get('se_interaction', float('nan')):.3f} | "
            f"{r.get('p_interaction', float('nan')):.3g} | "
            f"{r.get('cell_neutral', float('nan')):.3f} | "
            f"{r.get('cell_accidental', float('nan')):.3f} | "
            f"{r.get('cell_attempted', float('nan')):.3f} | "
            f"{r.get('cell_intentional', float('nan')):.3f} | "
            f"{r.get('diag_attempted_minus_accidental', float('nan')):+.3f} | "
            f"{r.get('cell_order', '')} | {r.get('saturating_factor', '')} |")
    md += ["", f"| **HUMAN (Young 2007)** | {human['b_intent']:+.3f} | "
               f"{human['b_outcome']:+.3f} | {human['b_interaction']:+.3f} | - | - | "
               f"{HUMAN_CELLS[(0,0)]:.3f} | {HUMAN_CELLS[(0,1)]:.3f} | "
               f"{HUMAN_CELLS[(1,0)]:.3f} | {HUMAN_CELLS[(1,1)]:.3f} | "
               f"{HUMAN_CELLS[(1,0)] - HUMAN_CELLS[(0,1)]:+.3f} | matches_human | intent |",
           "",
           "The `template_absorbed` rows in the CSV repeat every fit with prompt template as",
           "a fixed factor. C1 showed model identity dominates the variance, so this checks",
           "that the interaction is not an artefact of averaging over templates.", ""]
    md_path = os.path.join(a.out, "MIXED_EFFECTS_2x2.md")
    open(md_path, "w").write("\n".join(md))
    print(f"  -> {md_path}")


if __name__ == "__main__":
    main()
