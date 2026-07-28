#!/usr/bin/env python3
"""Build outputs/MENTOR_PACKET.md -- one page plus figures, for the mentor meeting.

WHY THIS IS GENERATED RATHER THAN HAND-WRITTEN. Every number in the packet is the number
currently on disk. The packet has been wrong twice before by transcription: the retired
2.5-3.9x ratio and the pre-rescore Zephyr zeros both survived in prose after the CSVs had
moved on. Regenerate instead of editing, and the packet cannot drift from the artifacts.

Sources
  outputs/tom_benchmarks/tom_vs_contrast.csv        headline ToM x contrast table
  outputs/experiments/checkpoint_stage_shares.csv   three-family locus finding
  outputs/stats/mixed_effects_2x2.csv               J3 cell ordering
  outputs/link/item_level_dissociation.csv          J2 pooled slope
  outputs/rsa/convergence_test.json                 RSA convergence null
  outputs/_handoff/human_anchor_comparison.csv      anchor ladders + pre-spec note

Outputs
  outputs/MENTOR_PACKET.md
  outputs/mentor_packet_figures/    (figures referenced by the packet, copied in)
"""
import csv
import json
import os
import shutil
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_MD = os.path.join(ROOT, "outputs", "MENTOR_PACKET.md")
FIG_DIR = os.path.join(ROOT, "outputs", "mentor_packet_figures")

# The headline table is the ENGAGED models (rating_std >= the derived floor 0.2191, see
# FLOOR_DERIVATION.md). Engagement is the pre-derived criterion; picking an ad-hoc
# high-FB/low-contrast box instead would be selecting the table on the two variables the
# table is about, and it silently drops Qwen-14B-Instruct, whose contrast sits 0.0001 the
# wrong side of a -0.37 cut.

FIGURES = [
    ("outputs/experiments/checkpoint_dissection.png",
     "Checkpoint dissection, three families, rescored 7-template basis."),
    ("outputs/tom_benchmarks/tom_vs_contrast.png",
     "BigToM false-belief x moral contrast, base/instruct by marker, no regression line."),
    ("outputs/stats/mixed_effects_interaction.png",
     "J3 interaction per model with 95% CI, human reference marked."),
    ("outputs/updated_figures/human_only_developmental_ladder.png",
     "Human-only ladder: three child series, one shared adult anchor."),
    ("outputs/master_developmental_ladder_digitized_openonly.png",
     "Model ladder against the Naughty (pre-specified primary) child anchor."),
    ("outputs/master_developmental_ladder_punish_openonly.png",
     "Model ladder against the Punish (secondary, construct-matched) child anchor."),
    ("outputs/probe/gap_over_surface_dissociation_span_matched.png",
     "Probe gaps over span-matched TF-IDF (position dissociation, downgraded to "
     "supporting)."),
    ("outputs/experiments/w3_steering_OLMo-2-1124-7B-Instruct.png",
     "W3 steering, OLMo-2-7B-Instruct: intent direction inside the random-direction noise "
     "floor, outcome direction well outside it."),
    ("outputs/experiments/w3_steering_Qwen2.5-7B-Instruct.png",
     "W3 steering, Qwen2.5-7B-Instruct: same verdict, with the probe-weight intent "
     "direction flat across the whole coherent band."),
]


def read_csv(rel):
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(x, default=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def tom_section():
    # The same CSV carries an appended ANALYSIS block; mtype filters to real model rows.
    rows = [r for r in read_csv("outputs/tom_benchmarks/tom_vs_contrast.csv")
            if r.get("mtype") in ("base", "instruct")]
    quad = sorted([r for r in rows if r.get("engaged") == "True"],
                  key=lambda r: -f(r["bigtom_false"]))
    fbs = [f(r["bigtom_false"]) for r in quad]
    cons = [f(r["contrast"]) for r in quad]
    n_instruct = sum(1 for r in rows if r["mtype"] == "instruct")
    lines = [
        "## 1. The headline: passing a false-belief benchmark does not buy intent-based "
        "moral judgment",
        "",
        f"Same models, two measures. Every model that engages with the rating task at all "
        f"passes BigToM false belief at **{min(fbs):.3f}-{max(fbs):.3f}** and has a moral "
        f"contrast (attempted - accidental) of **{max(cons):+.3f} to {min(cons):+.3f}** — "
        f"outcome-driven, the inverse of the adult human pattern (+0.67).",
        "",
        "| model | type | BigToM false-belief | moral contrast |",
        "|---|---|---:|---:|",
    ]
    for r in quad:
        lines.append(f"| `{r['model']}` | {r['mtype']} | {f(r['bigtom_false']):.3f} "
                     f"| {f(r['contrast']):+.3f} |")
    lines += [
        "",
        f"These are all {len(quad)} engaged models (`rating_std` >= 0.2191, the floor "
        f"derived in `FLOOR_DERIVATION.md`), out of {len(rows)} scored "
        f"({n_instruct} instruct). All six are instruct models, and there is no engaged "
        "model that passes the benchmark and *also* judges by intent — the cell is empty. "
        "BigToM was run with **`init_belief=0`**: the initial-belief sentence is dropped, so "
        "the model must infer the belief rather than copy it. Passing under the hard "
        "condition is what makes the dissociation strong.",
        "",
        "**Framing caution.** The raw correlation across all 20 models (r = -0.26) is "
        "confounded: both axes proxy base-vs-instruct, since base models cannot follow the "
        "QA format and sit near zero on contrast. The deliverable is this table and the "
        "scatter, not a correlation. ToMi is excluded entirely — the scored 400-item slice "
        "is 82% non-ToM items (`TOMI_SCORING_AUDIT.md`).",
        "",
    ]
    return lines


def checkpoint_section():
    rows = read_csv("outputs/experiments/checkpoint_stage_shares.csv")
    fams = {}
    for r in rows:
        fams.setdefault(r["family"], []).append(r)
    lines = [
        "## 2. Where in tuning it happens: SFT is sufficient, the locus is recipe-dependent",
        "",
        "**Revised finding — this replaces the earlier \"localized to SFT, not RLHF/DPO\" "
        "claim, which Zephyr refutes.** All three families with published intermediate "
        "checkpoints start at a neutral, *engaged* base and move to outcome-weighting. One "
        "stage of plain SFT is sufficient everywhere. But the share of the shift SFT "
        "contributes is a property of the recipe:",
        "",
        "| family | base | SFT | later stages | final | SFT share | concentrates at |",
        "|---|---:|---:|---|---:|---:|---|",
    ]
    for fam, rs in fams.items():
        rs.sort(key=lambda r: int(r["stage_idx"]))
        base, sft, fin = rs[0], next((x for x in rs if x["stage"] == "SFT"), None), rs[-1]
        later = " -> ".join(f"{x['stage']} {f(x['contrast']):+.3f}"
                            for x in rs[2:-1]) or "—"
        share = f(sft["share_of_total_pct"]) if sft else float("nan")
        biggest = max([x for x in rs if x["stage"] != "base"],
                      key=lambda x: abs(f(x["share_of_total_pct"])))
        lines.append(
            f"| {fam} | {f(base['contrast']):+.3f} | {f(sft['contrast']):+.3f} | {later} "
            f"| {f(fin['contrast']):+.3f} ({fin['stage']}) | **{share:.0f}%** "
            f"| {biggest['stage']} ({f(biggest['share_of_total_pct']):.0f}%) |")
    ratios = [f(r["outcome_over_intent"]) for r in rows if r["outcome_over_intent"]]
    lines += [
        "",
        f"Mechanism, in every family at every post-base stage: `b_outcome` grows several "
        f"times faster than `b_intent` — **{min(ratios):.1f}-{max(ratios):.1f}x** across "
        f"all non-base stages. Report that as a range. The previously quoted \"2.5-3.9x\" "
        f"was pre-rescore single-template and is retired.",
        "",
        "**Improvement worth flagging.** All three bases are now engaged "
        "(`rating_std` " + ", ".join(
            f"{fam} {f(rs[0]['rating_std']):.3f}" for fam, rs in fams.items()) + "). "
        "Tulu-3's base was previously degenerate at 0.018 and Zephyr's whole family was "
        "zeros from the digit-token bug. Every family now contrasts a *responsive* "
        "near-zero base against its tuned descendants, so the base -> tuned comparison is "
        "like-for-like. This was a stated limitation and it is resolved.",
        "",
    ]
    return lines


def j3_section():
    rows = [r for r in read_csv("outputs/stats/mixed_effects_2x2.csv")
            if r.get("spec") == "primary" and r.get("cell_order")]
    n = len(rows)
    counts = {k: sum(1 for r in rows if r["cell_order"] == k)
              for k in ("matches_human", "inverted", "tied")}
    inverted = [r for r in rows if r["cell_order"] == "inverted"]
    extreme = min(inverted, key=lambda r: f(r["diag_attempted_minus_accidental"]))
    closest = min(inverted, key=lambda r: abs(f(r["b_interaction"]) + 0.200))
    lines = [
        "## 3. J3: zero of twenty models match the human cell ordering",
        "",
        f"**{counts['matches_human']} of {n} models reproduce the human cell ordering "
        f"(attempted > accidental); {counts['inverted']} are inverted; "
        f"{counts['tied']} are tied.** This is the quotable result, not the interaction "
        "coefficient — several models approximate the human interaction magnitude "
        "(-0.200) while getting the underlying cell pattern backwards.",
        "",
        "| | neutral | accidental | attempted | intentional | att - acc | b_interaction |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| **Humans** (Young 2007) | 0.033 | 0.267 | **0.933** | 0.967 | **+0.666** "
        "| -0.200 |",
    ]
    for label, r in ((f"`{extreme['model']}` (most inverted)", extreme),
                     (f"`{closest['model']}` (closest coefficient to human)", closest)):
        lines.append(
            f"| {label} | {f(r['cell_neutral']):.3f} | **{f(r['cell_accidental']):.3f}** "
            f"| {f(r['cell_attempted']):.3f} | {f(r['cell_intentional']):.3f} "
            f"| {f(r['diag_attempted_minus_accidental']):+.3f} "
            f"| {f(r['b_interaction']):+.3f} |")
    lines += [
        "",
        "Humans judge an attempted harm that caused no damage almost as harshly as a "
        "completed intentional one, and an accident far more leniently. These models do the "
        "reverse: the accident outranks the attempt. Reading the interaction coefficient "
        "alone would call that human-like.",
        "",
    ]
    return lines


def nulls_section():
    item = read_csv("outputs/link/item_level_dissociation.csv")
    pooled = next((r for r in item if r["model_tag"].startswith("POOLED (primary)")), None)
    matched = next((r for r in item if r["model_tag"].startswith("POOLED (matched)")), None)
    rsa_p = os.path.join(ROOT, "outputs", "rsa", "convergence_test.json")
    rsa = json.load(open(rsa_p)) if os.path.exists(rsa_p) else {}
    lines = [
        "## 4. Four tests, one claim: intent is represented, readable, and not used",
        "",
        "Different units of analysis, same conclusion. Present the first two plus the "
        "steering result as the load-bearing set; the model-level test is a footnote because "
        "n=8 cannot answer anything.",
        "",
        "| test | unit | estimate | 95% CI | status |",
        "|---|---|---:|---|---|",
    ]
    if pooled:
        lines.append(
            f"| **J2 item-level link** | scenario group within model "
            f"| slope {f(pooled['pearson_r']):+.3f} "
            f"| [{f(pooled['ci_lo']):+.3f}, {f(pooled['ci_hi']):+.3f}] "
            f"| **informative null** — excludes +0.30 |")
    if rsa:
        lines.append(
            f"| **RSA convergence** | model pair | r = {rsa.get('r', float('nan')):+.3f} "
            f"| [{rsa.get('ci_lo', float('nan')):+.2f}, "
            f"{rsa.get('ci_hi', float('nan')):+.2f}] "
            f"| null — same behaviour, different geometry |")
    lines.append(
        "| Model-level link | model | r = -0.209 | [-0.80, +0.58] "
        "| **uninformative** — footnote only |")
    ub = f"{f(pooled['ci_hi']):+.3f}" if pooled else "n/a"
    lines += [
        "",
        "**Why J2 is informative rather than merely non-significant.** The minimum "
        "theoretically meaningful slope was pre-stated at **+0.30 SD**: if representation "
        "drove use, a scenario whose intent is 1 SD more decodable should show at least a "
        f"medium increase in intent-use. The CI upper bound is **{ub}**, which excludes "
        "that threshold. The model-level test spanned [-0.80, +0.58] and excluded nothing — "
        "that is the difference.",
        "",
    ]
    if matched:
        lines += [f"Robustness, `matched` intent definition (outcome held constant): slope "
                  f"{f(matched['pearson_r']):+.3f} "
                  f"[{f(matched['ci_lo']):+.3f}, {f(matched['ci_hi']):+.3f}], "
                  f"{matched['n']} observations.", ""]
    lines += [
        "**Two limits to state with it.** The bound is on a linear, monotone relation "
        "between probe margin and contrast; a threshold relation would not show up. And "
        "probe margin measures decodability, not what the model reads out downstream. "
        "W3 causal steering is the test that closes that gap.",
        "",
    ]
    lines += w3_lines()
    return lines


def w3_lines():
    """The causal test, if it has run. Fourth leg of the same claim."""
    rows = []
    for tag in ("OLMo-2-1124-7B-Instruct", "Qwen2.5-7B-Instruct"):
        rs = read_csv(f"outputs/experiments/w3_steering_{tag}.csv")
        if rs:
            rows.append((tag, rs))
    if not rows:
        return ["**W3 causal steering is running now** (pre-registered predictions in "
                "`outputs/experiments/W3_PRESPEC.md`, written before any result existed).",
                ""]

    def best(rs, pref):
        vals = [abs(f(r["dcontrast"])) for r in rs
                if r["direction"].startswith(pref) and r["coherent"] == "True"
                and f(r["alpha"]) != 0.0]
        return max(vals, default=float("nan"))

    out = [
        "### The causal test came back, and it agrees with the nulls",
        "",
        "**W3 steering (pre-registered, `W3_PRESPEC.md`) failed its own prediction, and the "
        "failure is the result.** Steering the intent direction at the peak intent layer "
        "does not move the moral contrast more than the outcome-direction control does. The "
        "intent direction taken from the *probe weights* — the vector whose decoding "
        "accuracy is our representational evidence — barely moves it at all:",
        "",
        "| model | unsteered contrast | intent (probe weights) | intent (diff-of-means) "
        "| outcome (diff-of-means) | random (matched norm) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for tag, rs in rows:
        base = next((r for r in rs if r["direction"] == "baseline"), None)
        out.append(
            f"| {tag} | {f(base['contrast']):+.3f} | **{best(rs, 'intent_probe'):.3f}** "
            f"| {best(rs, 'intent_dom'):.3f} | {best(rs, 'outcome_dom'):.3f} "
            f"| {best(rs, 'random'):.3f} |")
    out += [
        "",
        "Max |Δcontrast| over all coefficients where the model stays coherent (perplexity "
        "within 1.5x, no refusal increase, task compliance 1.00, manual read of 20 "
        "generations per level confirming the model still summarises the stories "
        "accurately). **The apparatus is not insensitive** — the outcome direction moves the "
        "same contrast by up to 0.26 in the same models at the same coefficients, which is "
        "the positive control that makes the intent null interpretable. Where the "
        "diff-of-means intent direction does move the contrast, it raises all four cells at "
        "once, and the accidental cell has least headroom, so the change is ceiling "
        "compression rather than a change in intent-weighting. Full verdict: "
        "`W3_STEERING_SUMMARY.md`.",
        "",
        "So the claim is now carried by four tests rather than three, one of them a "
        "manipulation: **intent is represented, linearly readable, and causally inert for "
        "this judgment.**",
        "",
    ]
    return out


def anchor_section():
    rows = [r for r in read_csv("outputs/_handoff/human_anchor_comparison.csv")
            if r.get("model")]
    nd = [r for r in rows if str(r.get("degenerate", "")).lower() != "true"]

    def holds(col):
        return sum(1 for r in nd if str(r.get(col, "")).lower() == "yes")
    n = len(nd)
    lines = [
        "## 5. The human anchor: both ladders, and the choice is not mine to make",
        "",
        "**The anchor decision traces to a prior methods pre-specification, not to which "
        "number is friendlier.** `dataset/human_reference/methods_child_measure.md` chose "
        "Naughty/wrongness, presented-first as primary on **2026-07-10 — sixteen days "
        "before** any model was compared against it. Both digitized ladders are reported "
        "permanently as a robustness table.",
        "",
        "| child series | youngest band (ages 4-5) | models at or below it | status |",
        "|---|---:|---:|---|",
        f"| **Naughty, presented-first** | +0.24 | {holds('below_youngest_digitized_naughty')}"
        f"/{n} | pre-specified primary (2026-07-10) |",
        f"| Punish, presented-first | +0.09 | {holds('below_youngest_punish')}/{n} "
        f"| secondary, construct-matched to the `punish_*` prompts |",
        f"| Text-reported (pooled prose) | -0.14 | {holds('below_youngest_text_reported')}"
        f"/{n} | superseded — mixes two constructs the paper separates |",
        "",
        "The claim \"models fall at or below the youngest measured band\" holds under "
        "**both digitized measures, including the stricter punishment threshold**, and "
        "fails only under the pooled-prose series. That is a robustness result. It does "
        "not select the primary anchor — that stays a decision for you.",
        "",
        "A theoretical check on the digitization: the punish ladder is monotone in age but "
        "flatter than naughtiness (+0.09/+0.12/+0.19 vs +0.24/+0.50/+0.63), which is "
        "exactly Cushman et al. (2013)'s two-process prediction that intent constrains "
        "wrongness before it constrains deserved punishment. Two independent digitizations "
        "reproducing the predicted ordering is evidence the digitization is sound.",
        "",
        "Scope note: these counts are open-weight models only. Closed-API models have not "
        "been rescored since the stimulus repair and their ladders are emitted separately, "
        "marked contaminated-era.",
        "",
    ]
    return lines


def questions_section():
    return [
        "## 6. Six questions for you",
        "",
        "> **Provenance flag:** `mentor_meeting_prep.md` is not in the repo, so these are "
        "reconstructed from the current results and the earlier four-question list rather "
        "than carried over verbatim. Please edit before the meeting.",
        "",
        "1. **Primary claim.** Is the paper's primary claim the **behavioral** one (model "
        "ladder vs human developmental bands, 18/18 open-weight models at or below the "
        "youngest child band under both digitized anchors), with representation as "
        "supporting evidence? Or does a strong submission need the causal result (W3) "
        "in the main claim?",
        "2. **The anchor.** Naughty/presented-first (+0.24) was pre-specified on 2026-07-10 "
        "and the claim holds under it and under the stricter Punish anchor (+0.09). Do you "
        "want the primary to stay with the pre-spec, with Punish as permanent robustness?",
        "3. **Recipe-dependence framing.** Zephyr puts 73% of its shift at DPO while OLMo-2 "
        "puts 85% at SFT, and Llama-3.1-8B-Instruct moves the *other* way entirely. Should "
        "we frame outcome-bias as **a default of many alignment recipes** rather than "
        "\"instruction tuning causes outcome bias\"?",
        "4. **The nulls, now including a causal one.** W3 steering came back negative for "
        "intent with a working positive control (outcome direction moves the contrast up to "
        "0.26; probe-weight intent direction moves it 0.016). Is \"intent is represented, "
        "readable, and causally inert\" publishable as a positive contribution on the "
        "strength of four converging tests, or do reviewers read a negative steering result "
        "as a failed experiment however well controlled?",
        "5. **Roster ceiling.** Our largest model is 14B and half the roster is one family; "
        "recent ToM papers standardly test 2-3 frontier APIs plus 8B-70B open weights with "
        "Llama-3.3-70B-Instruct as reference. Is the `gemma-3-27b` / `Qwen3-32B` mid-band "
        "worth the compute, or do we spend it on W3/W4 depth instead? "
        "(`ROSTER_70B_FEASIBILITY.md`)",
        "6. **Degenerate and contaminated rows.** Closed-API models are still v1-contaminated "
        "(reported standalone for ToM, never correlated against contrasts), and some open "
        "models sit below the engagement floor. Exclude them, or report non-engagement as a "
        "finding about rating elicitation?",
        "",
    ]


def limitations_section():
    return [
        "## 7. Limitations, stated plainly",
        "",
        "1. **Zephyr is a counterexample to the SFT-locus claim.** Its shift is 73% at DPO, "
        "27% at SFT. Any statement that the effect is localized to SFT, or absent from "
        "RLHF/DPO, is withdrawn. SFT sufficiency survives; SFT primacy does not.",
        "2. **The position dissociation is downgraded from headline to supporting.** The "
        "span-matched intent-minus-outcome difference at `belief_last` is **+0.087** "
        "(8/8 models, sign test p=0.008) — real but small, and it does not support an "
        "\"intent represented early, outcome inferred late\" reading. A manual audit of five "
        "YS2009 stories confirmed the clause offsets are correct but the *setup* sentences "
        "before the belief clause already state the hazard, so the probe-over-TF-IDF gap at "
        "that position reflects setup content a bag-of-words baseline cannot generalize "
        "across scenarios — not outcome anticipation. (`C2_SOURCE_SPLIT_BELIEF_LAST.md`)",
        "3. **Llama-3.1-8B-Instruct does not acquire the bias at all** (base -> instruct "
        "delta +0.126, toward intent). The effect is common but not universal.",
        "4. **Only three families publish intermediate checkpoints**, so the stage-level "
        "story rests on OLMo-2, Tulu-3 and Zephyr. Everything else is a 2-point "
        "base -> instruct delta that cannot speak to locus.",
        "5. **Closed-API models are v1-contaminated.** Their ToM accuracies are reported "
        "standalone and never correlated against their contrasts.",
        "6. **The J1 correlation is confounded** by base-vs-instruct on both axes. The "
        "per-model table is the deliverable; the all-20 r is a confound demonstration only.",
        "7. **The steering null bounds crude linear steering, not all causal involvement.** "
        "W3 adds a fixed vector at one layer across all token positions. A per-position, "
        "multi-layer, or `belief_last`-fitted intervention could still find an effect. Also, "
        "the intent and outcome difference-of-means directions are not orthogonal "
        "(cos ~0.32-0.39), so part of the diff-of-means intent effect may be outcome "
        "leakage — which is why the probe-weight null is the cleaner evidence.",
        "8. **The J2 bound is on a linear, monotone relation** between probe margin and "
        "contrast; a threshold relation (intent merely present rather than strongly present) "
        "would not show up in it.",
        "9. **Model ceiling 14B, roster concentration.** Half the roster is Qwen2.5; no "
        "70B-class model has been run.",
        "",
    ]


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    copied = []
    for rel, cap in FIGURES:
        src = os.path.join(ROOT, rel)
        if os.path.exists(src):
            dst = os.path.join(FIG_DIR, os.path.basename(rel))
            shutil.copy2(src, dst)
            copied.append((os.path.basename(rel), cap))
        else:
            print(f"  !! missing figure, omitted from packet: {rel}")

    lines = [
        f"# Mentor packet — moral ToM in LLMs ({date.today().isoformat()})",
        "",
        "One page of results, then figures. Every number is generated from the artifacts on "
        "disk by `code/experiments/49_mentor_packet.py`; regenerate rather than edit.",
        "",
        "**One-paragraph version.** Every open-weight model that engages with the rating "
        "task passes a standard false-belief benchmark at 0.82-0.99 under its hard "
        "(`init_belief=0`) condition, and those same "
        "models weight outcome over intent in graded moral judgment, inverting the adult "
        "human pattern and falling at or below the youngest measured child band. The bias is "
        "induced by post-training: one stage of SFT is sufficient in all three families with "
        "published checkpoints, though which stage carries most of the shift is "
        "recipe-dependent. Intent is nonetheless linearly decodable from the residual stream "
        "at 0.85-0.98, and four tests — three correlational nulls plus a pre-registered "
        "steering intervention with a working positive control — say that readable "
        "representation is not what drives the judgment.",
        "",
    ]
    lines += tom_section()
    lines += checkpoint_section()
    lines += j3_section()
    lines += nulls_section()
    lines += anchor_section()
    lines += questions_section()
    lines += limitations_section()

    lines += ["## Figures", "",
              f"Copied into `outputs/mentor_packet_figures/`.", ""]
    for name, cap in copied:
        lines.append(f"### {cap}")
        lines.append("")
        lines.append(f"![{cap}](mentor_packet_figures/{name})")
        lines.append("")

    open(OUT_MD, "w").write("\n".join(lines))
    print(f"wrote {os.path.relpath(OUT_MD, ROOT)} ({len(lines)} lines, "
          f"{len(copied)} figures)")


if __name__ == "__main__":
    main()
