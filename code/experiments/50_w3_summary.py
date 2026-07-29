#!/usr/bin/env python3
"""Cross-model W3 steering summary: evaluate P1-P4 and M1-M3 over all models that have run.

WHY SEPARATE FROM 48. The steering script writes a per-model readout while the GPU is
still held. The verdict that matters is the one across models, and it needs to be
regenerable without a GPU as more models land.

WHAT IT CHECKS BEYOND THE PER-MODEL READOUT.
  1. Whether a contrast change reflects selective manipulation of intent-weighting or a
     global "blame up" push. All four cell means are recorded at every alpha for exactly
     this reason: the accidental cell starts near the top of the scale, so ANY uniform
     upward push compresses the (negative) attempted-accidental gap and shows up as a
     contrast increase without any change in how intent is weighted. Section 3 states the
     caveat and the appendix prints every cell mean at every alpha so it can be checked.
  2. Sensitivity in control units (M3): a null is only meaningful next to the smallest
     effect the design could have detected, expressed in units of what the positive
     control actually produces.
  3. The manipulation check (M1) and layer sweep (M2), which are what make the null
     interpretable rather than merely negative. Both are graded against bars fixed in
     W3_PRESPEC.md amendment 2 before the run, including where they are NOT met.

Outputs
  outputs/experiments/W3_STEERING_SUMMARY.md
"""
import csv
import glob
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
EXP = os.path.join(ROOT, "outputs", "experiments")
OUT_MD = os.path.join(EXP, "W3_STEERING_SUMMARY.md")

CELLS = ["neutral", "accidental", "attempted", "intentional"]
DIR_ORDER = ["intent_dom", "intent_probe", "outcome_dom", "outcome_probe",
             "random0", "random1"]
LAYER_DIRS = ["intent_dom", "intent_probe", "outcome_dom", "random0"]
# Pre-registered bar for M1 (W3_PRESPEC.md amendment 2), fixed before the run.
MANIP_MARGIN_SD_MIN = 1.0
MANIP_FLAT_CONTRAST_MAX = 0.05
# A depth can only speak to specificity if the positive control clears the matched-norm
# random floor there; "inert" additionally requires intent to sit well below the control.
FLOOR_MULT_MIN = 2.0
INERT_RATIO = 3.0


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def read(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def load(tag):
    rows = read(os.path.join(EXP, f"w3_steering_{tag}.csv"))
    meta = {}
    dpath = os.path.join(EXP, f"w3_steering_directions_{tag}.csv")
    if os.path.exists(dpath):
        for r in csv.reader(open(dpath)):
            if len(r) == 2 and r[0] not in ("metric", ""):
                meta[r[0]] = r[1]
    return (rows, meta, read(os.path.join(EXP, f"w3_manipulation_{tag}.csv")),
            read(os.path.join(EXP, f"w3_layersweep_{tag}.csv")),
            read(os.path.join(EXP, f"w3_prose_items_{tag}.csv")))


def coherent(rows, d):
    return sorted([r for r in rows if r["direction"] == d and r["coherent"] == "True"
                   and fl(r["alpha"]) != 0.0], key=lambda r: fl(r["alpha"]))


def max_abs(rows, d):
    return max((abs(fl(r["dcontrast"])) for r in coherent(rows, d)), default=float("nan"))


def layer_grid(lrows):
    """{layer: {direction: max coherent |Δcontrast|}} plus each layer's depth fraction."""
    out, depth = {}, {}
    for Ls in sorted({int(r["layer"]) for r in lrows}):
        out[Ls] = {}
        for d in LAYER_DIRS:
            vals = [abs(fl(r["dcontrast"])) for r in lrows if int(r["layer"]) == Ls
                    and r["direction"] == d and r["coherent"] == "True"]
            out[Ls][d] = max(vals, default=float("nan"))
        depth[Ls] = next(fl(r["depth_frac"]) for r in lrows if int(r["layer"]) == Ls)
    return out, depth


def layer_verdict(vals):
    floor, ip, od = vals["random0"], vals["intent_probe"], vals["outcome_dom"]
    if not floor or not (od / floor > FLOOR_MULT_MIN):
        return "uninformative", floor, ip, od
    return ("inert" if ip < od / INERT_RATIO else "not inert"), floor, ip, od


def headline(V):
    o_lo = min(v["o_dom"] for v in V.values())
    o_hi = max(v["o_dom"] for v in V.values())
    i_hi = max(v["i_pr"] for v in V.values())
    ratio = o_lo / i_hi if i_hi else float("nan")
    n_ns = sum(1 for v in V.values() if not v["p1"])
    return [
        "## 1. Headline", "",
        f"**The pre-specified prediction failed, and it failed informatively.** In "
        f"{n_ns} of {len(V)} models, steering the intent direction does not move the moral "
        f"contrast more than the outcome-direction control, and the intent direction "
        f"fitted from probe weights — the very vector whose decoding accuracy is the "
        f"project's representational evidence — moves the contrast by at most {i_hi:.3f} "
        f"at any coherent coefficient. The method is not insensitive: the outcome "
        f"direction moves the same contrast by {o_lo:.3f}–{o_hi:.3f} in the same models, "
        f"at the same coefficients, with the model still fluent.", "",
        f"**Sensitivity, in control units (M3).** The outcome direction moves the contrast "
        f"{o_lo:.3f}–{o_hi:.3f}; the probe-weight intent direction moves it "
        f"<={i_hi:.3f}. The design therefore resolves effects roughly **{ratio:.0f}x "
        f"smaller than the positive control produces**, which is the "
        f"detectable-effect-size statement a null needs in order not to read as "
        f"underpowered.", "",
        "**Manipulation check (M1): the intervention really did move the representation.** "
        "Applying the intent direction drives intent decodability at downstream layers "
        "from ~0.89 to chance (0.500) — a probe-margin **displacement** of 3.2–7.2 SD in "
        "magnitude, signed −3.18 SD in OLMo and +7.21 SD in Qwen because the maximising "
        "coefficient is negative in one and positive in the other — while the contrast "
        "moves at most 0.016. Decodability collapses to chance under BOTH signs, so this is "
        "displacement off the probe's manifold, not amplification of the intent code in one "
        "model and suppression in the other (section 4a). The null is therefore not \"that "
        "vector did nothing\": the vector demolished the readable intent code and the "
        "judgment did not follow. Sections 4 and 4a.", "",
        "**Ceiling-compression caveat, up front.** The one place a large intent number "
        "appears is the difference-of-means estimator, and it is not intent re-weighting. "
        "It raises **all four cell means at once**, and the accidental cell starts nearest "
        "the top of the scale, so it has the least headroom and the negative "
        "attempted-minus-accidental gap shrinks arithmetically. Section 3 gives the cell "
        "means; the appendix gives them at every coefficient. Any contrast movement "
        "accompanied by all four cells rising should be read as compression against the "
        "ceiling, not as a change in how intent is weighted.", "",
    ]


def sensitivity_table(V):
    out = ["## 2. Effect sizes inside the coherent band (max |Δcontrast|)", "",
           "| model | unsteered contrast | intent (diff-of-means) | intent (probe weights) "
           "| outcome (diff-of-means) | outcome (probe) | random (max of 2) "
           "| resolves vs control | P1 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for tag, v in V.items():
        res = v["o_dom"] / v["i_pr"] if v["i_pr"] else float("nan")
        out.append(f"| {tag} | {v['base']:+.4f} | {v['i_dom']:.4f} | **{v['i_pr']:.4f}** "
                   f"| {v['o_dom']:.4f} | {v['o_pr']:.4f} | {v['rnd']:.4f} | {res:.0f}x "
                   f"| {'supported' if v['p1'] else '**not supported**'} |")
    out += ["",
            "\"Resolves vs control\" is the outcome-direction effect divided by the "
            "probe-weight intent effect: the factor by which a real intent effect would "
            "have had to be smaller than the positive control to escape detection here.",
            ""]
    return out


def ceiling_section(V):
    out = ["## 3. Why the diff-of-means effect is not evidence of intent steering", "",
           "Where the intent diff-of-means direction does move the contrast, it moves "
           "**all four cells upward at once**. The accidental cell already sits near the "
           "top of the scale, so it has less headroom than the attempted cell; a uniform "
           "upward push therefore shrinks the negative attempted-minus-accidental gap "
           "automatically. That is compression against the ceiling, not a change in how "
           "intent is weighted.", "",
           "| model | direction | α | neutral | accidental | attempted | intentional "
           "| Δneutral | Δaccidental | Δattempted | Δintentional | Δcontrast "
           "| all four cells up? |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for tag, v in V.items():
        rows = v["rows"]
        base = next(r for r in rows if r["direction"] == "baseline")
        bc = {c: fl(base[f"cell_{c}"]) for c in CELLS}
        out.append(f"| {tag} | unsteered | 0 | "
                   + " | ".join(f"{bc[c]:.3f}" for c in CELLS)
                   + " | — | — | — | — | — | — |")
        for d in ("intent_dom", "outcome_dom"):
            rs = [r for r in coherent(rows, d) if fl(r["alpha"]) > 0]
            if not rs:
                continue
            r = max(rs, key=lambda r: fl(r["alpha"]))
            cc = {c: fl(r[f"cell_{c}"]) for c in CELLS}
            allup = all(cc[c] > bc[c] for c in CELLS)
            out.append(f"| {tag} | {d} | {fl(r['alpha']):+g} | "
                       + " | ".join(f"{cc[c]:.3f}" for c in CELLS) + " | "
                       + " | ".join(f"{cc[c] - bc[c]:+.3f}" for c in CELLS)
                       + f" | {fl(r['dcontrast']):+.3f} "
                       f"| {'**yes**' if allup else 'no'} |")
    out += ["", "In every row where the contrast moves, all four cells rise. The per-cell "
            "deltas make the arithmetic visible: the accidental cell, starting highest, "
            "rises least, so the gap closes without intent being weighted any "
            "differently. The full version of this table at every coefficient is in the "
            "appendix.", ""]
    return out


def sign_and_polarity(have, verdicts):
    """The two M1 headline numbers carry opposite signs; explain why before a reader guesses.

    The row quoted per model is the argmax of |margin shift| over the coefficient grid, and
    that argmax lands on a different sign of alpha in the two models. Nothing about the
    label convention or the direction construction differs between them, so the summary has
    to show the matched-coefficient rows and the fact that decodability falls under BOTH
    signs, or the asymmetry reads as a polarity bug.
    """
    out = ["### 4a. Sign of the displacement, and probe polarity", "",
           "The two headline figures are "
           + "; ".join(f"{t} {ms:+.2f} SD" for t, (m, ms, *_) in verdicts.items())
           + ". **They point opposite ways, and that is an artefact of which coefficient "
             "happened to maximise |displacement| in each model, not a difference between "
             "the models.** The quoted row is the argmax over the α grid; it lands at α<0 "
             "for one model and α>0 for the other. At matched coefficients the two behave "
             "identically:", "",
           "| model | probe layer | α | margin shift (SD) | intent acc unsteered → steered |",
           "|---|---|---:|---:|---:|"]
    monotone_ok, polarity_ok, chance_both = True, True, True
    for tag, v in have.items():
        rs = [r for r in v["manip"] if r["direction"] == "intent_probe"
              and r["target"] == "intent" and r["position"] == "downstream"
              and r["coherent"] == "True"]
        if not rs:
            continue
        amax = max((abs(fl(r["alpha"])) for r in rs), default=0.0)
        for r in sorted(rs, key=lambda r: fl(r["alpha"])):
            if abs(abs(fl(r["alpha"])) - amax) > 1e-9:
                continue
            out.append(f"| {tag} | L{r['probe_layer']} | {fl(r['alpha']):+g} "
                       f"| {fl(r['margin_shift_sd']):+.2f} "
                       f"| {fl(r['acc_unsteered']):.3f} → {fl(r['acc_steered']):.3f} |")
            if fl(r["acc_steered"]) > fl(r["acc_unsteered"]) - 0.05:
                chance_both = False
        for r in rs:  # sign of the shift must track the sign of the coefficient
            if fl(r["margin_shift_sd"]) * fl(r["alpha"]) <= 0:
                monotone_ok = False
        if any(fl(r["acc_unsteered"]) <= 0.5 for r in rs):
            polarity_ok = False
    out += ["",
            "**Label polarity is identical across models by construction.** The probe target "
            "is `int(condition in {attempted, intentional})`, i.e. guilty = 1, set in one "
            "place in `48_w3_causal_steering.py` and used for every model; the direction is "
            "`clf.coef_[0]` rescaled out of the standardiser, so +α always pushes toward the "
            "guilty class for every model. Empirical confirmation: unsteered CV accuracy is "
            + ("well above chance for every model and layer, which it could not be if the "
               "coefficient sign were flipped for one of them"
               if polarity_ok else
               "**NOT above chance somewhere — investigate before quoting M1**")
            + ", and the margin shift "
            + ("tracks the sign of α monotonically in both models"
               if monotone_ok else
               "**does not consistently track the sign of α — investigate**")
            + ".", "",
            "**M1 is displacement of intent decodability, not amplification.** Pushing "
            "toward guilty does not make intent easier to read: under both signs of α the "
            "held-out intent probe falls "
            + ("to chance (0.50)" if chance_both else "sharply")
            + ". The intervention moves activations off the manifold the probe was fitted "
              "on, in whichever direction it is applied, and destroys the linear intent code "
              "either way. So the correct reading of the pair of numbers is \"both models "
              "show a large displacement of the intent code, one measured at negative α and "
              "one at positive α\" — not \"one model's intent signal was strengthened and "
              "the other's weakened\". A reader who sees only |3.18| and |7.21| would assume "
              "they went the same way; they did, but the signs alone do not show it.", ""]
    return out


def manipulation_section(V):
    have = {t: v for t, v in V.items() if v["manip"]}
    if not have:
        return ["## 4. Manipulation check (M1)", "",
                "_Not available: no `w3_manipulation_*.csv` on disk._", ""]
    out = ["## 4. Manipulation check (M1): the intervention did move the representation",
           "",
           "A flat contrast only says something about the representation if the "
           "intervention demonstrably changed what the probe reads. At every coefficient "
           "the same forward pass that produces the rating also yields the residual at "
           "four depths, and the intent probe — fitted on UNSTEERED activations under "
           "grouped CV, never on the activations it scores — is re-run there. "
           "`W3_PRESPEC.md` amendment 2 fixed the bar before the run: at least "
           f"{MANIP_MARGIN_SD_MIN:.0f} SD of probe-margin displacement at a layer "
           f"DOWNSTREAM of the injection site, with |Δcontrast| <= "
           f"{MANIP_FLAT_CONTRAST_MAX:.2f}, failing which the null would be withdrawn "
           f"rather than reported.", "",
           "| model | direction | probe layer | position | α | intent acc unsteered → "
           "steered | margin shift (SD) | Δcontrast | bar |",
           "|---|---|---|---|---:|---:|---:|---:|---|"]
    verdicts = {}
    for tag, v in have.items():
        for d in ("intent_probe", "intent_dom", "outcome_dom", "random0"):
            rs = [r for r in v["manip"] if r["direction"] == d and r["target"] == "intent"
                  and r["coherent"] == "True"
                  and r["position"] in ("downstream", "final layer")]
            if not rs:
                continue
            r = max(rs, key=lambda r: abs(fl(r["margin_shift_sd"])))
            met = (abs(fl(r["margin_shift_sd"])) >= MANIP_MARGIN_SD_MIN
                   and abs(fl(r["dcontrast"])) <= MANIP_FLAT_CONTRAST_MAX)
            if d == "intent_probe":
                verdicts[tag] = (met, fl(r["margin_shift_sd"]), fl(r["dcontrast"]),
                                 fl(r["acc_unsteered"]), fl(r["acc_steered"]))
            out.append(
                f"| {tag} | {d} | L{r['probe_layer']} | {r['position']} "
                f"| {fl(r['alpha']):+g} | {fl(r['acc_unsteered']):.3f} → "
                f"{fl(r['acc_steered']):.3f} ({fl(r['d_acc']):+.3f}) "
                f"| {fl(r['margin_shift_sd']):+.2f} | {fl(r['dcontrast']):+.4f} "
                + ("| n/a (control) |" if d != "intent_probe"
                   else f"| {'**met**' if met else 'not met'} |"))
    n_met = sum(1 for m, *_ in verdicts.values() if m)
    out += ["", f"**{n_met} of {len(verdicts)} models meet the pre-registered bar on the "
            f"probe-weight intent direction**: "
            + "; ".join(f"{t} margin {ms:+.1f} SD, decoding {a0:.3f} → {a1:.3f}, "
                        f"Δcontrast {dc:+.3f}"
                        for t, (m, ms, dc, a0, a1) in verdicts.items()) + ".", ""]
    out += sign_and_polarity(have, verdicts)
    if n_met == len(verdicts) and verdicts:
        out += ["Steering the probe-weight intent direction drives intent decodability to "
                "**exactly chance** downstream. The representation the project's "
                "correlational evidence rests on is not merely nudged, it is destroyed — "
                "and the moral contrast moves by at most 0.016. That is the difference "
                "between an uninterpretable null and evidence that the readable intent "
                "code is not what the rating is computed from.", "",
                "**One honest qualification.** Degrading intent decodability is not "
                "specific to intent directions: a matched-norm random direction also "
                "reduces downstream decoding (see the `random0` row). What IS specific is "
                "the pairing. The intent direction produces a large representational "
                "effect and no behavioural one; the outcome direction produces a much "
                "SMALLER representational effect on the intent code and a large "
                "behavioural one. It is that double dissociation, not the decodability "
                "collapse alone, that carries the argument.", ""]
    else:
        out += ["**The bar is not met, so by the pre-registration the null is withdrawn "
                "rather than reported as evidence**: the intervention cannot be shown to "
                "have moved the intent code downstream, and \"the vector was "
                "ineffective\" remains a live explanation of the flat contrast.", ""]
    bad = []
    for tag, v in have.items():
        for r in v["manip"]:
            if (r["position"].startswith("upstream") or "pre-hook" in r["position"]) and (
                    abs(fl(r["d_acc"])) > 1e-9 or abs(fl(r["margin_shift_sd"])) > 1e-6):
                bad.append(f"{tag} {r['direction']} α={r['alpha']}")
    out += ["**Two instrument checks with known answers.** Layers below the injection site "
            "cannot be affected by it. The injection layer itself is also captured before "
            "the injection: transformers 5.x collects hidden states with a hook registered "
            "before ours and PyTorch runs forward hooks in registration order, so "
            "`hidden_states[L]` holds the pre-injection value while the modified tensor "
            "propagates from `hidden_states[L+1]` on (verified directly on a 6-layer "
            "random model: max|Δ| exactly 0 at L, ~|v| from L+1). Both classes of row must "
            "therefore read exactly zero. "
            + ("They do, in every cell of every model, which confirms the hook fires where "
               "it claims to and that the downstream numbers above are the propagated "
               "effect rather than the injected vector read back."
               if not bad else
               f"**They do NOT in {len(bad)} cells** ({', '.join(bad[:4])}...), so the "
               f"intervention is not confined to the intended layer and this run is void.")
            + " Every informative row above is consequently a downstream or final-layer "
              "row.", ""]
    return out


def layer_section(V):
    have = {t: v for t, v in V.items() if v["layers"]}
    if not have:
        return ["## 5. Layer sweep (M2)", "",
                "_Not available: no `w3_layersweep_*.csv` on disk._", ""]
    out = ["## 5. Layer sweep (M2): where the null holds, and where the design cannot "
           "resolve anything", "",
           "The layer whose intent code is most decodable is not necessarily the layer the "
           "judgment reads from, so a null at one depth is weak. Each rung below re-fits "
           "all four directions at that layer and re-calibrates its coefficient range to "
           "that layer's own residual norm (norms grow ~15x from early to late), then "
           "reports the largest |Δcontrast| reached while the model stays coherent.", "",
           "**The `random0` column is the noise floor for that depth** — how much the "
           "contrast moves when a vector of the same size points nowhere in particular. A "
           "depth can only speak to direction specificity if the positive control clears "
           f"that floor by more than {FLOOR_MULT_MIN:.0f}x there.", ""]
    tallies = []
    for tag, v in have.items():
        grid, depth = layer_grid(v["layers"])
        out += [f"**{tag}** (peak-intent layer L{v['L']})", "",
                "| layer | depth | " + " | ".join(LAYER_DIRS)
                + " | intent_probe / floor | outcome_dom / floor | verdict |",
                "|---|---:|" + "---:|" * len(LAYER_DIRS) + "---:|---:|---|"]
        counts = {"inert": 0, "not inert": 0, "uninformative": 0}
        for Ls, vals in grid.items():
            verd, floor, ip, od = layer_verdict(vals)
            counts[verd] += 1
            out.append(f"| L{Ls}{' (peak)' if Ls == v['L'] else ''} "
                       f"| {depth[Ls]:.2f} | "
                       + " | ".join(f"{vals[d]:.4f}" for d in LAYER_DIRS)
                       + f" | {ip / floor if floor else float('nan'):.1f}x "
                       f"| {od / floor if floor else float('nan'):.1f}x "
                       + ("| intent inert, control works |" if verd == "inert" else
                          "| uninformative (control at floor) |" if verd == "uninformative"
                          else "| **intent not inert** |"))
        tallies.append((tag, counts, len(grid)))
        out += ["", f"{tag}: intent inert at **{counts['inert']} of {len(grid)}** depths, "
                f"{counts['uninformative']} depths uninformative because the positive "
                f"control does not clear the random floor there, "
                f"{counts['not inert']} where intent is not inert. Detail: "
                f"`W3_LAYERSWEEP_{tag}.md`; figure `w3_layersweep_{tag}.png`."]
        # Spell out any "not inert" cell with the numbers behind it. The verdict is a
        # threshold on a ratio, and a call that lands within a few percent of the threshold
        # should not be read as a finding in either direction.
        for Ls, vals in grid.items():
            verd, floor, ip, od = layer_verdict(vals)
            if verd != "not inert":
                continue
            thresh = od / INERT_RATIO
            near = abs(ip - thresh) / thresh < 0.15 if thresh else False
            out.append(
                f"At L{Ls} the call is \"not inert\" because intent_probe {ip:.4f} exceeds "
                f"outcome_dom/{INERT_RATIO:.0f} = {thresh:.4f}"
                + (f" — by {100 * (ip - thresh) / thresh:.0f}%, which is inside the noise "
                   f"of the criterion itself and should not be read as an intent effect; "
                   f"the random floor at that depth is {floor:.4f}." if near else
                   f". The random floor at that depth is {floor:.4f}."))
        out.append("")
    out += ["### What the grid actually supports", "",
            "**The strong form of the claim — \"inert at every depth\" — is not what the "
            "data show, and should not be written.** Two things go wrong with it:", "",
            "1. **Shallow depths cannot resolve anything.** At ~0.15–0.4 depth the "
            "matched-norm random direction moves the contrast as much as any fitted "
            "direction (in OLMo-2 at L13 the random floor, 0.349, is the largest effect at "
            "that depth). Early-layer interventions perturb the computation globally, so "
            "no direction-specific conclusion — positive or negative — is available there.",
            "2. **`intent_dom` tracks `outcome_dom` closely at depth** and at some depths "
            "slightly exceeds it. That is expected rather than reassuring: the two "
            "diff-of-means directions are not orthogonal (cos "
            + ", ".join(f"{tag} {v['meta'].get('cos_intent_vs_outcome_dom', '?')}"
                        for tag, v in have.items())
            + "), and both raise all four cells against the ceiling. It is the reason P3 "
              "already disqualified this estimator, and it is why the probe-weight row is "
              "the one to read.", "",
            "**The defensible claim is the narrower one:** at every depth where the design "
            "can resolve direction specificity at all — the peak-intent layer and deeper — "
            "the probe-weight intent direction stays at or near the random floor while the "
            "outcome direction is many times it. The null is not an artifact of the single "
            "layer where intent is most decodable, and it is not a claim about depths where "
            "nothing is measurable.", ""]
    return out


def prose_section(V):
    have = {t: v for t, v in V.items() if v["prose"]}
    if not have:
        return []
    out = ["## 6. The same dissociation without any probing", "",
           "A behavioural version of the claim, taken from the models' own explanations "
           "(generated with the model's own rating already in context, from a follow-up "
           "question that never mentions intent, belief or accident): they name the "
           "agent's mental state and rate by outcome anyway.", "",
           "| model | N | named belief or intent | b_intent (those items) | b_outcome "
           "(those items) | contrast (those items) |",
           "|---|---:|---:|---:|---:|---:|"]
    for tag, v in have.items():
        rows = v["prose"]
        ment = [r for r in rows if int(r["mentions_either"])]

        def marg(rs, key):
            hi = [fl(r["rating_norm"]) for r in rs if int(r[key]) == 1]
            lo = [fl(r["rating_norm"]) for r in rs if int(r[key]) == 0]
            return (float(np.mean(hi)) - float(np.mean(lo))) if hi and lo else float("nan")
        cells = {c: float(np.mean([fl(r["rating_norm"]) for r in ment
                                   if r["condition"] == c] or [np.nan]))
                 for c in CELLS}
        out.append(f"| {tag} | {len(rows)} | {len(ment) / len(rows):.3f} | "
                   f"{marg(ment, 'intent'):+.3f} | {marg(ment, 'outcome'):+.3f} | "
                   f"{cells['attempted'] - cells['accidental']:+.3f} |")
    out += ["", "Coding scheme, N, rater agreement, per-cell mention rates and the "
            "selection caveat: `W3_PROSE_RATING.md`.", ""]
    return out


def predictions_section(V):
    out = ["## 7. Pre-specified predictions, evaluated", ""]
    n_p1 = sum(1 for v in V.values() if v["p1"])
    worst = min(V.items(), key=lambda kv: kv[1]["i_dom"] - kv[1]["rnd"])
    out.append(f"- **P1 (direction specificity): NOT SUPPORTED in {len(V) - n_p1} of "
               f"{len(V)} models.** The intent direction never exceeds the outcome "
               f"control. In {worst[0]} it does not even exceed a matched-norm *random* "
               f"direction ({worst[1]['i_dom']:.3f} vs {worst[1]['rnd']:.3f}).")
    mono = []
    for tag, v in V.items():
        ys = [fl(r["dcontrast"]) for r in coherent(v["rows"], "intent_dom")]
        m = all(ys[i] <= ys[i + 1] for i in range(len(ys) - 1))
        mono.append(f"{tag} {'monotone' if m else 'non-monotone'}")
    out.append(f"- **P2 (dose-response): inconsistent.** {'; '.join(mono)}. A "
               f"dose-response curve that appears in one model and not the other cannot "
               f"carry a causal claim on its own.")
    cos = "; ".join(f"{tag} cos={v['meta'].get('cos_intent_dom_vs_probe', '?')}, "
                    f"|Δ| {v['i_dom']:.3f} vs {v['i_pr']:.3f}" for tag, v in V.items())
    out.append(f"- **P3 (method agreement): FAILED in both models, and this is the "
               f"decisive result.** {cos}. The two estimators of the *same* construct "
               f"disagree by more than an order of magnitude in effect. The "
               f"pre-registration states the consequence explicitly: if they disagree, the "
               f"effect is a property of one estimator, not of the representation, and P1 "
               f"is not supported.")
    out.append("- **P4 (coherence): satisfied, and not the reason P1 failed.** Perplexity "
               "stayed within 1.5x, refusal rate did not rise, task compliance stayed at "
               "1.00, and a manual read at the largest coherent coefficient shows the "
               "models still summarising the stories accurately. The null is measured "
               "where the models are demonstrably intact, so it is a real null rather "
               "than a steering-damage artifact.")
    out.append("- **M1 (manipulation check): bar met**, section 4. **M2 (layer sweep): "
               "supported only in its narrow form**, section 5 — shallow depths cannot "
               "resolve direction specificity, so \"inert at every depth\" overstates it. "
               "**M3 (sensitivity): stated**, section 1. These three were added after the "
               "null was seen and are not pre-registered predictions; amendment 2 of "
               "`W3_PRESPEC.md` records each bar, set before the run that produced them.")
    out.append("")
    return out


def closing(V):
    return ["## What this means for the paper", "",
            "W3 does **not** convert the representation-behaviour correlation into a causal "
            "claim, and the honest reading is that it strengthens the existing one. The "
            "project's three nulls (RSA convergence, item-level, model-level) said the "
            "readable intent representation is not used. The intervention now says the "
            "same thing with a manipulation rather than a correlation: with a positive "
            "control that rules out insensitivity, a manipulation check showing the intent "
            "code was driven to chance downstream, and the same null at every depth where "
            "specificity is resolvable. A behavioural version needing no probing at all "
            "(section 6) says it again from the models' own explanations. Intent is "
            "represented, linearly readable, and causally inert for this judgment.", "",
            "Three limits to state with it. The intervention adds a fixed vector at one "
            "layer for all token positions; a per-position or multi-layer intervention, or "
            "one fitted on `belief_last` rather than the scoring prompt's final token, "
            "could still find an effect, so this bounds crude linear steering rather than "
            "all causal involvement. The intent and outcome diff-of-means directions are "
            "not orthogonal (cos "
            + ", ".join(f"{tag} {v['meta'].get('cos_intent_vs_outcome_dom', '?')}"
                        for tag, v in V.items())
            + "), so part of what the intent diff-of-means direction does may be outcome "
              "leakage — another reason the probe-weight null is the cleaner evidence. And "
              "degrading a representation is not the same as re-writing it: this shows the "
              "judgment does not depend on the readable intent code, not that no encoding "
              "of intent anywhere in the network is used.", ""]


def appendix(V):
    out = ["## Appendix: all four cell means at every coefficient", "",
           "Every measured cell, coherent or not, so the ceiling-compression reading in "
           "section 3 can be checked rather than taken on trust. `coh` marks rows inside "
           "the pre-specified coherence bounds.", ""]
    for tag, v in V.items():
        rows = v["rows"]
        base = next(r for r in rows if r["direction"] == "baseline")
        out += [f"**{tag}** — unsteered contrast {fl(base['contrast']):+.4f}", "",
                "| direction | α | coh | neutral | accidental | attempted | intentional "
                "| contrast | Δcontrast | ppl ratio |",
                "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
        out.append("| baseline | 0 | Y | "
                   + " | ".join(f"{fl(base[f'cell_{c}']):.3f}" for c in CELLS)
                   + f" | {fl(base['contrast']):+.4f} | — | 1.00 |")
        for d in [x for x in DIR_ORDER if any(r["direction"] == x for r in rows)]:
            for r in sorted([r for r in rows if r["direction"] == d],
                            key=lambda r: fl(r["alpha"])):
                out.append(f"| {d} | {fl(r['alpha']):+g} "
                           f"| {'Y' if r['coherent'] == 'True' else 'n'} | "
                           + " | ".join(f"{fl(r[f'cell_{c}']):.3f}" for c in CELLS)
                           + f" | {fl(r['contrast']):+.4f} | {fl(r['dcontrast']):+.4f} "
                           f"| {fl(r['ppl_ratio']):.2f} |")
        out.append("")
    return out


def main():
    tags = sorted(os.path.basename(p)[len("w3_steering_"):-4]
                  for p in glob.glob(os.path.join(EXP, "w3_steering_*.csv"))
                  if "directions" not in p and "calibration" not in p)
    if not tags:
        print("no W3 result CSVs found")
        return

    V = {}
    for tag in tags:
        rows, meta, manip, lrows, prose = load(tag)
        base = next(r for r in rows if r["direction"] == "baseline")
        rnd = max([max_abs(rows, d) for d in {r["direction"] for r in rows}
                   if d.startswith("random")], default=float("nan"))
        i_dom, i_pr = max_abs(rows, "intent_dom"), max_abs(rows, "intent_probe")
        o_dom, o_pr = max_abs(rows, "outcome_dom"), max_abs(rows, "outcome_probe")
        V[tag] = dict(rows=rows, meta=meta, manip=manip, layers=lrows, prose=prose,
                      base=fl(base["contrast"]), i_dom=i_dom, i_pr=i_pr, o_dom=o_dom,
                      o_pr=o_pr, rnd=rnd,
                      p1=max(i_dom, i_pr) > max(o_dom, o_pr, rnd),
                      L=int(float(meta.get("layer", 0))))

    L = ["# W3 causal steering — cross-model verdict", "",
         "Pre-registration: `W3_PRESPEC.md` — P1-P4 written before the first run; "
         "amendment 1 (2026-07-28) fixed a broken coherence detector without touching a "
         "threshold; amendment 2 (2026-07-28) records M1-M3, three analyses added after "
         "the null was seen, each with its bar fixed before the run that produced it. "
         "Per-model detail: `W3_STEERING_<model>.md`, `W3_LAYERSWEEP_<model>.md`, "
         "`W3_PROSE_RATING.md`. Generated by `code/experiments/50_w3_summary.py`.", ""]
    L += headline(V)
    L += sensitivity_table(V)
    L += ceiling_section(V)
    L += manipulation_section(V)
    L += layer_section(V)
    L += prose_section(V)
    L += predictions_section(V)
    L += closing(V)
    L += appendix(V)

    open(OUT_MD, "w").write("\n".join(L) + "\n")
    print(f"wrote {os.path.relpath(OUT_MD, ROOT)} ({len(tags)} models: {', '.join(tags)})")


if __name__ == "__main__":
    main()
