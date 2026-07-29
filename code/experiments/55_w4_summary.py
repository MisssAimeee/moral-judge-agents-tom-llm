#!/usr/bin/env python3
"""W4 report: evaluate the pre-registered readings against the curriculum results.

Reads outputs/experiments/w4_prompt_curriculum.csv and writes W4_CURRICULUM.md plus
figures_final/w4_curriculum.png. Every judgment made here is against a bar fixed in
W4_PRESPEC.md before the run: P3 (shift of at least +0.15 with a bootstrap CI on the
paired difference excluding 0, in at least 4 of 6 models) decides "prompting works";
P4 separates full recovery from a partial shift; P5 checks dose-response and flags an
L4-only jump as possible format imitation; P6 prints all four cell means at every level
so a contrast change produced by uniform movement cannot be read as intent re-weighting.
"""
import os, sys, csv
from collections import defaultdict

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)
import tom_common as tc  # noqa: E402
import numpy as np  # noqa: E402

EXP = os.path.join(tc.ROOT, "outputs", "experiments")
CSV_IN = os.path.join(EXP, "w4_prompt_curriculum.csv")
MD_OUT = os.path.join(EXP, "W4_CURRICULUM.md")
FIG = os.path.join(tc.ROOT, "outputs", "figures_final", "w4_curriculum.png")

SHIFT_BAR = 0.15          # P3, in normalised contrast units
MIN_MODELS = 4            # P3, of 6 engaged
ADULT_CONTRAST = 0.933 - 0.267   # Young 2007 digitized adults, attempted - accidental
W3_INTENT_MAX = 0.015     # largest |Δcontrast| from intent steering in W3
W3_OUTCOME_MIN = 0.232    # smallest |Δcontrast| from the W3 outcome positive control
W3_OUTCOME_MAX = 0.259    # largest, i.e. the W3 positive control's range is .232-.259
CUM = (1, 2, 3, 4, 5)     # the pre-registered cumulative curriculum
ABL = (6, 7, 8)           # non-cumulative ablation, added after the fact
ABL_LABEL = {2: "instruction alone", 6: "worked example alone", 7: "few-shot alone",
             8: "principle alone"}


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load():
    if not os.path.exists(CSV_IN):
        return {}
    by = defaultdict(dict)
    for r in csv.DictReader(open(CSV_IN)):
        by[r["model"]][int(r["level"])] = r
    return by


def attribute(r):
    """Where did a contrast change come from? Cell movement, not the contrast alone.

    The contrast is attempted - accidental, so the identical gain arises from adding blame
    to attempted harm (intent re-weighting, the interesting reading) or from removing blame
    from accidental harm (outcome-blame suppression, which is not the same claim), or from
    every cell sliding together (compression, which is neither). W3 already carries this
    caveat for its difference-of-means direction; the same guard belongs here.
    """
    d = {c: fl(r.get(f"d_{c}")) for c in tc.CELLS}
    da, dc = d["attempted"], d["accidental"]
    gain = da - dc
    if any(np.isnan(v) for v in d.values()):
        return "—", ""
    vals = [d[c] for c in tc.CELLS]
    uniform = (all(v > 0 for v in vals) or all(v < 0 for v in vals))
    if gain <= 0.02:
        head = "no gain"
    elif da > 0.02 and dc < -0.02:
        head = "**both** — attempted ↑, accidental ↓ (intent re-weighting)"
    elif dc < -0.02 and da <= 0.02:
        head = ("**accidental ↓ only** — outcome-blame suppressed, no intent-blame added"
                if da < -0.02 else "**accidental ↓** — outcome-blame suppressed")
    elif da > 0.02 and dc >= -0.02:
        head = "**attempted ↑** — intent-blame added"
    else:
        head = "mixed"
    if uniform and gain > 0.02:
        head += "; all four cells move the same way (compression)"
    detail = " ".join(f"{c[:4]}{d[c]:+.3f}" for c in tc.CELLS)
    return head, detail


def plot(by):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (no figure: {e})")
        return
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    cmap = plt.get_cmap("tab10")
    for i, (m, all_lv) in enumerate(sorted(by.items())):
        lv = {k: v for k, v in all_lv.items() if k in CUM}   # ablation is a table, not a line
        ks = sorted(lv)
        y = [fl(lv[k]["contrast"]) for k in ks]
        lo = [fl(lv[k]["contrast_lo"]) for k in ks]
        hi = [fl(lv[k]["contrast_hi"]) for k in ks]
        ax.plot(ks, y, "-o", color=cmap(i % 10), label=tc.pretty(m), lw=1.8, ms=5)
        ax.fill_between(ks, lo, hi, color=cmap(i % 10), alpha=0.10, lw=0)
    ax.axhline(0, color="k", lw=1)
    ax.axhline(ADULT_CONTRAST, color="crimson", ls="--", lw=1.4)
    ax.text(1.02, ADULT_CONTRAST, " adult (Young 2007)", color="crimson",
            va="center", fontsize=8)
    ax.set_xticks([k for k in sorted({k for lv in by.values() for k in lv}) if k in CUM])
    ax.set_xticklabels([f"L{k}\n{n}" for k, n in
                        sorted({int(r["level"]): r["level_name"]
                                for lv in by.values() for r in lv.values()
                                if int(r["level"]) in CUM}.items())],
                       fontsize=8)
    ax.set_ylabel("contrast  (attempted − accidental, normalised)")
    ax.set_title("W4 prompt curriculum: does in-context intervention recover "
                 "intent-weighting?\nnegative = outcome-dominant (inverted vs adults); "
                 "bands are scenario-group bootstrap 95% CIs", fontsize=10)
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=170)
    print(f"  wrote {os.path.relpath(FIG, tc.ROOT)}")


def verdicts(by):
    """Two verdicts per model, because the curriculum turned out to be non-monotone.

    P3 places the bar at the top level (L5), and that is where the pre-registered verdict
    is read. The maximum shift over levels is reported alongside it as descriptive, never
    substituted for it: picking the best level after seeing the results is exactly the
    move a pre-registration exists to prevent.
    """
    out = {}
    for m, all_lv in by.items():
        lv = {k: v for k, v in all_lv.items() if k in CUM}
        ks = sorted(lv)
        if not ks:
            continue
        top = lv[ks[-1]]
        d, dlo, dhi = fl(top["d_contrast"]), fl(top["d_lo"]), fl(top["d_hi"])
        met = (d >= SHIFT_BAR) and (dlo > 0)
        bl = max(ks, key=lambda k: fl(lv[k]["d_contrast"]))
        bd, bdlo = fl(lv[bl]["d_contrast"]), fl(lv[bl]["d_lo"])
        best = max(ks, key=lambda k: fl(lv[k]["contrast"]))
        out[m] = dict(d=d, dlo=dlo, dhi=dhi, met=met, top_level=ks[-1],
                      met_any=(bd >= SHIFT_BAR) and (bdlo > 0),
                      max_level=bl, max_d=bd, max_lo=bdlo, max_hi=fl(lv[bl]["d_hi"]),
                      recovered=fl(lv[best]["contrast"]) > 0,
                      best_level=best, best=fl(lv[best]["contrast"]),
                      base=fl(lv[ks[0]]["contrast"]))
    return out


def main():
    by = load()
    if not by:
        print(f"no {os.path.relpath(CSV_IN, tc.ROOT)} yet — run 54 first")
        return
    V = verdicts(by)
    n_met = sum(1 for v in V.values() if v["met"])
    n_any = sum(1 for v in V.values() if v["met_any"])
    n_rec = sum(1 for v in V.values() if v["recovered"])
    works = n_met >= MIN_MODELS
    level_specific = (not works) and n_any >= MIN_MODELS

    ref = by[list(by)[0]][min(by[list(by)[0]])]
    has_abl = any(k in ABL for lv in by.values() for k in lv)
    L = ["# W4 — prompt curriculum: is the intent representation reachable from the "
         "input?", "",
         f"_{len(by)} engaged open models, {len(CUM)} cumulative levels"
         + (f" plus a {len(ABL)}-cell non-cumulative ablation" if has_abl else "")
         + f", {ref['n_templates']}-template basis, {ref['n_groups']} scenario groups "
         f"({ref.get('n_cells', '?')} template × group cells). "
         "Readings pre-registered in `W4_PRESPEC.md`; prompts verbatim in "
         "`W4_PROMPT_LEVELS.md`._", "",
         "> **Two corrections applied after run 1** (preserved under "
         "`_w4_prefix_fewshot_bug/`). The few-shot block rendered each example under the "
         "question built from the *target* item, and `human_verbatim` interpolates the "
         "agent name — so an example about Nadia was followed by \"How permissible was "
         "Grace's action?\", invalid on 1 of 7 templates and only at the levels containing "
         "the few-shot block. L4/L5 were rescored. Separately, the bootstrap resampled "
         "template × group cells rather than scenario groups, treating one vignette under "
         "seven templates as seven independent observations; all intervals are now over "
         "the scenario group and are correspondingly wider.", "",
         "> **The L4 labels are not inverted.** Checked before interpreting L4, since "
         "Young 2007 is 1–4 permissibility while most templates are 1–7 blame, and this "
         "project has had both a CPR polarity inversion and a permissibility-direction "
         "reversal. The YS2008 anchor is phrased 1 = completely permissible → 3 = "
         "completely impermissible, ascending in condemnation like every other template, "
         "so no reversal is required. The few-shot labels encode attempted > accidental on "
         "every template × scale, implied contrast +0.500 to +0.667 against an adult "
         f"reference of {ADULT_CONTRAST:+.3f}. The table is in `W4_PROMPT_LEVELS.md` and "
         "the check is now a gate that aborts the run.", "",
         "## Verdict", ""]

    if works:
        L += [f"**Prompting works ({n_met}/{len(V)} models meet the pre-registered "
              f"+{SHIFT_BAR:.2f} bar).** By the pre-specified reading P1, the blockage is "
              "downstream of the representation and upstream of the output: the intent "
              "code exists (probe ~0.89), residual-stream intervention on it does not "
              f"reach the judgment (W3, |Δcontrast| ≤ {W3_INTENT_MAX:.3f}), but the "
              "judgment can be re-pointed at it from the input. The failure is in how the "
              "rating computation selects its inputs, not in the absence of a usable "
              "intent signal.",
              "",
              f"{n_rec}/{len(V)} models achieve **full recovery** (P4: contrast crosses "
              "zero, adult ordering restored); the rest show a shift that leaves the "
              "ordering inverted, which is a movement toward adults and not adult-like "
              "judgment."]
    elif level_specific:
        L += [f"**Level-specific, and the pre-registered bar is not met where it was "
              f"placed.** P3 reads the verdict at the top level: there "
              f"{n_met}/{len(V)} models qualify, short of the {MIN_MODELS} required, so by "
              "the pre-registration this is not \"prompting works\". But "
              f"{n_any}/{len(V)} models DO clear +{SHIFT_BAR:.2f} with a CI excluding zero "
              "at some earlier level, which is not nothing and is not what P2 describes "
              "either.", "",
              "The honest statement is the third one the pre-registration did not "
              "anticipate: **in-context intervention can move the contrast substantially, "
              "and the fully escalated prompt is not where it moves most.** Escalation is "
              "not monotone, so \"can prompting fix it\" and \"does more scaffolding fix "
              "it more\" have different answers. The dose-response section below is the "
              "part to read, in particular the L4 column: adding labelled adult ratings "
              "can undo the gain from unlabelled worked reasoning. The labels themselves "
              "are verified non-inverted, so that reversal is not arithmetic; what it is "
              "instead — imitation of the example anchors, the added prompt length, or the "
              "question moving away from the story — this design does not separate, and the "
              "ablation below is the closest available handle. "
              "Reporting the maximum over levels as the headline would be selecting the "
              "level after seeing the data; it is reported as descriptive only."]
    else:
        L += [f"**Prompting also fails ({n_met}/{len(V)} models meet the pre-registered "
              f"+{SHIFT_BAR:.2f} bar; {MIN_MODELS} required).** By the pre-specified "
              "reading P2, the outcome bias is deeper than either intervention reaches. "
              "Two interventions at opposite ends of the pipeline — one on the "
              "representation (W3), one on the input (W4) — both leave the contrast "
              "inverted. Instruction, worked reasoning, adult-consistent few-shot labels "
              "and an explicit statement of the principle are all available to the model "
              "in context, and the rating still tracks the outcome. That makes the bias a "
              "property of the tuned input-output mapping rather than a prompt-level or "
              "read-out-level defect."]
    # The attribution is not a footnote: if no model adds blame to attempted harm, the
    # shift is not the recovery the intervention was designed to produce.
    tops = {m: {k: v for k, v in by[m].items() if k in CUM}[V[m]["top_level"]] for m in V}
    gainers = [m for m in V if fl(tops[m]["d_contrast"]) > 0.02]
    acc_only = [m for m in gainers if fl(tops[m]["d_accidental"]) < -0.02
                and fl(tops[m]["d_attempted"]) <= 0.02]
    att_up = [m for m in gainers if fl(tops[m]["d_attempted"]) > 0.02]
    uniform = [m for m in gainers
               if all(fl(tops[m][f"d_{c}"]) < 0 for c in tc.CELLS)
               or all(fl(tops[m][f"d_{c}"]) > 0 for c in tc.CELLS)]
    if gainers:
        L += ["",
              f"**The mechanism is not the one the intervention was aimed at.** Of the "
              f"{len(gainers)} models whose contrast improves at the top level, "
              f"{len(acc_only)} improve entirely because blame for *accidental* harm falls, "
              f"and {len(att_up)} {'adds' if len(att_up) == 1 else 'add'} blame to "
              "*attempted* harm"
              + (f" ({tc.pretty(att_up[0])}, {fl(tops[att_up[0]]['d_attempted']):+.3f}, "
                 "against a fall in accidental of "
                 f"{fl(tops[att_up[0]]['d_accidental']):+.3f})" if len(att_up) == 1 else "")
              + ". "
              + (f"In {len(uniform)} of them all four cell means move in the same "
                 "direction, which is compression of the rating range rather than a "
                 "re-weighting of either factor. "
                 if uniform else "")
              + "So at the top of the curriculum in-context instruction moves the judgment "
                "by making the model less condemnatory about bad outcomes rather than more "
                "condemnatory about bad intentions — the adult pattern is approached from "
                "the wrong side. This is the same caveat that governs the W3 "
                "difference-of-means direction."]
    # The ablation contradicts that reading for one component, which is worth stating in the
    # verdict rather than leaving in a secondary section.
    # "attempted rises" is only the interesting reading when the rise is also the dominant
    # term. A +0.027 rise beside a -0.224 fall in accidental is outcome suppression with a
    # rounding error attached, and counting it as re-weighting would overstate the result.
    abl_led, abl_any = [], []
    for m, all_lv in by.items():
        r8 = all_lv.get(8)
        if not r8 or fl(r8["d_contrast"]) <= 0.02:
            continue
        da, dc = fl(r8["d_attempted"]), fl(r8["d_accidental"])
        if da > 0.02:
            abl_any.append((m, da, dc))
            if da >= abs(dc):
                abl_led.append((m, da, dc))
    if abl_led:
        L += ["",
              "**But that is a property of the stack, not of instruction as such.** With the "
              "intent principle stated *alone* (L8: no belief cue, no worked example, no "
              f"labelled examples), {len(abl_led)} of {len(by)} models produce a gain in "
              "which the *rise* in blame for attempted harm is the dominant term — "
              + "; ".join(f"{tc.pretty(m)} Δattempted {a:+.3f} against Δaccidental {c:+.3f}"
                          for m, a, c in sorted(abl_led, key=lambda t: -t[1]))
              + (f" ({len(abl_any) - len(abl_led)} further model(s) raise attempted harm by a "
                 "smaller amount than they lower accidental harm, which is still outcome "
                 "suppression.)" if len(abl_any) > len(abl_led) else "")
              + " That is the re-weighting the intervention was designed to produce, and it "
                "appears where the prompt is *least* elaborate. The scaffolding is what "
                "appears to convert intent re-weighting into blanket outcome-blame "
                "suppression, which fits the additivity column: every model is sub-additive, "
                "and in three the single best component beats the full stack. This is the "
                "secondary, post-hoc arm — the pre-registered verdict remains the cumulative "
                "L5 column, and two models are not a result on their own. It is the sharpest "
                "thing to test next."]
    L += ["",
          "Reference points: adults sit at "
          f"{ADULT_CONTRAST:+.3f} on this measure (Young 2007 digitized); the W3 outcome "
          f"positive control moves the contrast {W3_OUTCOME_MIN:.3f}+ at the same models, "
          f"so a shift of {SHIFT_BAR:+.2f} is well inside what this design resolves.", "",
          "## Contrast by level", "",
          "| model | " + " | ".join(f"L{k}" for k in CUM)
          + " | Δ(top−L1) [95% CI] | bar at top level | best level Δ [95% CI] "
            "| crosses 0 | where the gain comes from (at top level) |",
          "|---" * (len(CUM) + 6) + "|"]
    for m, all_lv in sorted(by.items()):
        lv = {k: v for k, v in all_lv.items() if k in CUM}
        v = V[m]
        cells = " | ".join(f"{fl(lv[k]['contrast']):+.3f}" if k in lv else "—"
                           for k in CUM)
        head, _ = attribute(lv[v["top_level"]])
        L.append(f"| {tc.pretty(m)} | {cells} | {v['d']:+.3f} "
                 f"[{v['dlo']:+.3f}, {v['dhi']:+.3f}] "
                 f"| {'**met**' if v['met'] else 'not met'} "
                 f"| L{v['max_level']}: {v['max_d']:+.3f} "
                 f"[{v['max_lo']:+.3f}, {v['max_hi']:+.3f}]"
                 f"{' ✱' if v['met_any'] else ''} "
                 f"| {'yes (L%d)' % v['best_level'] if v['recovered'] else 'no'} "
                 f"| {head} |")
    L += ["", "Δ columns are paired over scenario groups against that model's own L1. "
          "The pre-registered verdict is the `bar at top level` column; `best level Δ` is "
          "descriptive (✱ marks a shift that would have cleared the bar had the "
          "pre-registration placed it at that level, which it does not). The last column "
          "is the ceiling-compression check: a contrast gain built from `accidental` "
          "falling is outcome-blame suppression, not the intent re-weighting the "
          "intervention was aimed at, and a gain in which all four cells slide together is "
          "neither. Per-cell numbers are in the P6 table below."]

    # P5 dose-response, and the L4-vs-L5 test that separates imitation from principle.
    L += ["", "## Dose-response (P5)", "",
          "Levels are cumulative, so under P1 the shift should be monotone or nearly so. "
          "L4 adds labelled adult ratings; L5 adds the principle and no new labels, so an "
          "L4-only jump that does not persist at L5 indicates imitation of the few-shot "
          "format rather than uptake of the principle. The reverse — a gain at L3 that L4 "
          "destroys — indicates the opposite: the labelled examples are being imitated as "
          "anchors and are overriding the reasoning they were meant to illustrate.", "",
          "| model | monotone | L1→L2 | L2→L3 | L3→L4 | L4→L5 | reading |",
          "|---|---|---:|---:|---:|---:|---|"]
    for m, all_lv in sorted(by.items()):
        lv = {k: v for k, v in all_lv.items() if k in CUM}
        ks = sorted(lv)
        ys = [fl(lv[k]["contrast"]) for k in ks]
        mono = all(b >= a - 1e-9 for a, b in zip(ys, ys[1:]))
        step = {}
        for k in (2, 3, 4, 5):
            step[k] = (fl(lv[k]["contrast"]) - fl(lv[k - 1]["contrast"])) \
                if (k in lv and k - 1 in lv) else float("nan")
        gain_l3 = sum(step[k] for k in (2, 3) if not np.isnan(step[k]))
        if any(np.isnan(v) for v in step.values()):
            note = "incomplete"
        elif gain_l3 >= SHIFT_BAR and step[4] <= -SHIFT_BAR:
            # Descriptive only. The labels are verified non-inverted, so this is not an
            # arithmetic artefact, but "anchor imitation" is one hypothesis among several
            # (added prompt length, format shift, position of the question) and the design
            # does not separate them. The ablation's few-shot-alone cell is the closest
            # available handle.
            note = "worked reasoning helps, adding labelled examples reverses it"
        elif step[4] >= SHIFT_BAR and step[5] <= -SHIFT_BAR / 2:
            note = "gain appears only with the labelled examples present"
        elif max(gain_l3, gain_l3 + step[4] + step[5]) >= SHIFT_BAR:
            note = "shift present and survives to the top level"
        else:
            note = "no shift of the pre-registered size at any level"
        L.append(f"| {tc.pretty(m)} | {'yes' if mono else 'no'} | "
                 + " | ".join(f"{step[k]:+.3f}" for k in (2, 3, 4, 5))
                 + f" | {note} |")

    # P6 ceiling guard: all four cell means at every level.
    L += ["", "## All four cell means at every level (P6 ceiling guard)", "",
          "A contrast change produced by all four means moving together is compression, "
          "not intent re-weighting — the same caveat that governs the W3 "
          "difference-of-means estimator. `extreme` is the fraction of ratings at the "
          "top or bottom of the scale; a level that pushes ratings to an endpoint removes "
          "the headroom the contrast needs.", "",
          "| model | level | neutral | accidental | attempted | intentional | contrast "
          "| Δ cells vs L1 | attribution | rating SD | extreme |",
          "|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|"]
    for m, lv in sorted(by.items()):
        for k in sorted(lv):
            r = lv[k]
            head, detail = attribute(r)
            L.append(f"| {tc.pretty(m)} | L{k} {r['level_name']} | "
                     + " | ".join(f"{fl(r['mean_' + c]):.3f}" for c in tc.CELLS)
                     + f" | {fl(r['contrast']):+.3f} | {detail} | {head} "
                     f"| {fl(r['rating_std']):.4f} | {fl(r['extreme_frac']):.2f} |")

    if has_abl:
        L += ["", "## Non-cumulative ablation (secondary)", "",
              "L1–L5 are cumulative, so L5 confounds the explicit principle with repair of "
              "whatever L4 did, and no cumulative level attributes the effect to a "
              "component. Each cell below is that component alone against the same L1 "
              "baseline. L2 already is \"instruction alone\" and is reused rather than "
              "re-run. **This is a secondary attribution analysis**: the pre-registered "
              "verdict stays the cumulative L5 column above, since these cells were "
              "designed after seeing run 1.", "",
              "| model | " + " | ".join(f"{ABL_LABEL[k]} (L{k})" for k in (2, 6, 7, 8))
              + " | Σ parts | L5 cumulative | reading |",
              "|---|---:|---:|---:|---:|---:|---:|---|"]
        for m, lv in sorted(by.items()):
            ds = {k: fl(lv[k]["d_contrast"]) if k in lv else float("nan")
                  for k in (2, 6, 7, 8)}
            if all(np.isnan(v) for v in ds.values()):
                continue
            parts = [v for v in ds.values() if not np.isnan(v)]
            tot, l5 = sum(parts), fl(lv[5]["d_contrast"]) if 5 in lv else float("nan")
            best = max((v, k) for k, v in ds.items() if not np.isnan(v))
            if np.isnan(l5):
                note = "L5 missing"
            elif best[0] >= SHIFT_BAR and l5 < SHIFT_BAR:
                note = f"{ABL_LABEL[best[1]]} beats the full stack — escalation subtracts"
            elif tot > 0 and l5 > tot + 0.05:
                note = "super-additive: the stack does more than its parts"
            elif tot > 0 and l5 < tot - 0.05:
                note = "sub-additive: components interfere"
            else:
                note = "roughly additive"
            L.append(f"| {tc.pretty(m)} | "
                     + " | ".join("—" if np.isnan(ds[k]) else f"{ds[k]:+.3f}"
                                  for k in (2, 6, 7, 8))
                     + f" | {tot:+.3f} | {l5:+.3f} | {note} |")
        L += ["", "Δ vs L1, paired over scenario groups. Σ parts is the arithmetic sum of "
              "the four single-component shifts and is a descriptive additivity reference, "
              "not a prediction any model of the effect entails.", "",
              "### Where the principle-alone gain comes from", "",
              "The attribution differs between the stack and the principle on its own, which "
              "is the most consequential thing the ablation shows. Same columns as the P6 "
              "guard, restricted to L5 (full stack) against L8 (principle alone).", "",
              "| model | L5 Δattempted | L5 Δaccidental | L8 Δattempted | L8 Δaccidental "
              "| L8 mechanism |", "|---|---:|---:|---:|---:|---|"]
        for m, lv in sorted(by.items()):
            if 8 not in lv or 5 not in lv:
                continue
            head8, _ = attribute(lv[8])
            L.append(f"| {tc.pretty(m)} | {fl(lv[5]['d_attempted']):+.3f} "
                     f"| {fl(lv[5]['d_accidental']):+.3f} "
                     f"| {fl(lv[8]['d_attempted']):+.3f} "
                     f"| {fl(lv[8]['d_accidental']):+.3f} | {head8} |")

    # The W3 comparison, deliberately without a ratio.
    best_any = max((V[m]["max_d"] for m in V), default=float("nan"))
    L += ["", "## Relation to W3 (steering)", "",
          "Steering and prompting are different interventions on different quantities and "
          "share no common effect-size scale, so the two are not divided into each other "
          "here. The defensible statement is qualitative: **the intent representation is "
          "inert to residual-stream intervention at the depths where intent is resolvable "
          f"(W3, |Δcontrast| ≤ {W3_INTENT_MAX:.3f}), while the same contrast moves "
          "substantially under in-context instruction. That places the blockage downstream "
          "of the representation and upstream of the output** — the rating computation can "
          "be re-pointed by the input but not by editing the vector the probe reads.", "",
          f"Descriptively, the largest in-context shift observed ({best_any:+.3f}) sits "
          f"close to the range the W3 *outcome* direction produced "
          f"({W3_OUTCOME_MIN:.3f}–{W3_OUTCOME_MAX:.3f}) as a positive control. That is a "
          "coincidence of magnitude worth flagging and nothing is built on it: the two "
          "numbers come from different manipulations, and the outcome-direction figure "
          "carries its own compression caveat. Its only use is as a reminder that shifts "
          "of this size are well inside what the design resolves.", "",
          "## What this does and does not license", "",
          "- The levels differ only in added text. Weights, decoding (logprob-EV over "
          "rating digits, deterministic), items, templates, scale normalisation and the "
          "contrast estimator are identical across levels, so the level effect is not "
          "confounded with the measurement.", "",
          "- The L3/L4 example vignettes are held out: novel content (climbing gym, print "
          "shop) that appears nowhere in the master, so the scaffolding cannot teach a "
          "test item. The L4 labels are the Young 2007 adult profile mapped onto each "
          "template's own scale; they are used only to write the examples and are never a "
          "target in the analysis.", "",
          "- A contrast gain is not automatically intent re-weighting. Read the attribution "
          "column: where the gain is `accidental` falling, the intervention removed blame "
          "from accidental harm rather than adding it to attempted harm, and where all four "
          "cells slide together the movement is compression. Both are real changes to the "
          "judgment and neither is evidence that the model started consulting intent.", "",
          "- This is an in-context result. It says nothing about whether fine-tuning could "
          "move the contrast, and a prompt-level shift does not establish that the model "
          "is using the same intent representation the probe reads — only that the input "
          "can change the weighting. Establishing that it is the same representation "
          "would require the W3 intervention to work, which it does not.", ""]

    with open(MD_OUT, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"  wrote {os.path.relpath(MD_OUT, tc.ROOT)} "
          f"({len(by)} models, verdict: "
          + ('works' if works else 'level-specific' if level_specific else 'also fails')
          + ')')
    plot(by)


if __name__ == "__main__":
    main()
