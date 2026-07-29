#!/usr/bin/env python3
"""W4 -- prompt curriculum: is the intent representation reachable from the input?

Companion result to W3, not a separate item. W3 established that the intent code is
highly decodable (~0.89 held-out) and causally inert to residual-stream intervention:
displacing it by 3.2-7.2 SD of probe margin moves the moral contrast by at most 0.015,
while an outcome direction at the same depths moves it 0.232-0.259. The question W3
leaves open is not whether intent is represented -- it is -- but whether anything
UPSTREAM can get the rating to use it. This experiment escalates in-context
intervention with no weight updates and measures the contrast at each level.

CURRICULUM (cumulative: level k contains every component of level k-1)

  L1 baseline           the unmodified scoring prompt used everywhere else in the project
  L2 + belief cue       one instruction to consider what the character believed
  L3 + worked example    one worked pair, attempted vs accidental, on a held-out vignette,
                         with the reasoning spelled out but NO ratings given
  L4 + few-shot          four labelled examples spanning the 2x2 with adult-consistent
                         ratings (Young 2007 digitized adult profile), held-out vignettes
  L5 + intent principle  explicit statement of the normative principle

Everything after L1 is added text only. Weights, decoding, templates, items, scale
normalisation and the contrast estimator are identical across levels, so the level
effect is not confounded with the measurement.

=============================================================================
PRE-SPECIFIED PREDICTIONS AND BOTH READINGS  (fixed before the first run)
=============================================================================

The two outcomes are both informative and they say different things, so both readings
are committed to here rather than chosen after seeing the numbers.

P1  READING IF PROMPTING WORKS. If the contrast moves toward the adult direction as
    the curriculum escalates, the blockage is DOWNSTREAM of the representation and
    UPSTREAM of the output: the intent code exists, residual-stream intervention on it
    does not reach the judgment, but the judgment can be re-pointed at it from the
    input. That localises the failure to how the rating computation selects its inputs,
    not to the absence of a usable intent signal.

P2  READING IF PROMPTING ALSO FAILS. If escalating instruction, worked reasoning,
    adult-consistent few-shot ratings and an explicit statement of the principle all
    leave the contrast inverted, then the outcome bias is deeper than either
    intervention reaches. Two interventions at opposite ends of the pipeline -- one on
    the representation, one on the input -- both fail to move it, and the bias is a
    property of the tuned mapping rather than a prompt-level or read-out-level defect.

P3  BAR FOR "PROMPTING WORKS", fixed here. Positive shift in contrast (toward the adult
    ordering attempted > accidental) of at least +0.15 at L5 relative to L1, with a
    scenario-group bootstrap CI on the paired difference excluding 0, in at least 4 of
    the 6 engaged models. +0.15 is 10x the largest intent-steering effect in W3 (0.015)
    and about two thirds of what the W3 outcome positive control produces (0.232), so
    it is an effect this design has already been shown to resolve. Anything smaller is
    reported as a shift but not as recovery.

P4  FULL RECOVERY, distinguished from partial. Full recovery = contrast crosses zero
    and becomes positive at some level. Partial = significant positive shift that
    leaves the ordering inverted. These are reported separately; a significant shift
    that still has accidental rated above attempted is not human-like moral judgment.

P5  DOSE-RESPONSE. Under P1 the shift should be monotone or near-monotone in level,
    since the levels are cumulative. A single non-monotone jump at L4 only would be
    consistent with format imitation of the few-shot ratings rather than uptake of the
    principle, and is flagged as such (L5 adds the principle with no new labels, so
    L4->L5 separates imitation from principle).

P6  CEILING-COMPRESSION GUARD, carried over from W3. A contrast change produced by all
    four cell means rising or falling together is compression, not intent
    re-weighting. All four cell means are reported at every level, plus the fraction of
    ratings at the scale extremes, so this is checkable and not inferable only from the
    contrast.

Method: logprob-EV digit scoring (one forward pass per prompt, deterministic, no
sampling variance across levels), scenario-group averaging so YS2009 reprints of YS2008
vignettes do not double-count, and the same 7-template factorial basis as the headline
contrast.
"""
import os, sys, csv, json, argparse, importlib.util
from collections import defaultdict

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)
import tom_common as tc  # noqa: E402
import numpy as np  # noqa: E402

OUT_DIR = os.path.join(tc.ROOT, "outputs", "experiments")
OUT_CSV = os.path.join(OUT_DIR, "w4_prompt_curriculum.csv")
CELL_CSV = os.path.join(OUT_DIR, "w4_curriculum_cells.csv")
PRESPEC = os.path.join(OUT_DIR, "W4_PRESPEC.md")
BASE_LEVEL = 1   # every delta and every ablation contrast references the L1 prompt
LEVELS_MD = os.path.join(OUT_DIR, "W4_PROMPT_LEVELS.md")

# Engaged open models (rating_std above the derived engagement floor; see
# outputs/analysis/DERIVED_FLOORS.md). Ordered so the two W3 models come first and a
# preempted job still yields the W3/W4 comparison.
ENGAGED = [
    "allenai/OLMo-2-1124-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "unsloth/gemma-2-9b-it",
    "allenai/Llama-3.1-Tulu-3-8B",
    "HuggingFaceH4/zephyr-7b-beta",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

# Young 2007 digitized adult profile, normalised 0-1. Used ONLY to write the few-shot
# labels at L4, converted to each template's own scale. Not used as a target anywhere in
# the analysis.
ADULT = {"neutral": 0.033, "accidental": 0.267, "attempted": 0.933, "intentional": 0.967}

# Held-out vignettes for L3/L4. Deliberately novel content: nothing in the master uses a
# climbing gym or a print shop, so the examples cannot teach a test item. Each carries an
# explicit belief clause and an explicit outcome clause in the Young frame.
EX_CLIMB = {
    "attempted": (
        "Nadia and a climber she dislikes are at a climbing gym. Nadia is asked to hand "
        "over a rope from the equipment bin. One rope in the bin is frayed and would snap "
        "under load; the other is new. The frayed rope has been tagged with a red marker. "
        "Nadia sees a red marker on the rope she picks up, so Nadia believes the rope she "
        "is handing over is the frayed one. In fact a staff member had marked the new rope "
        "by mistake, and the rope is sound. Nadia hands over the rope. The climber uses it "
        "and is fine."),
    "accidental": (
        "Nadia and a climber she has just met are at a climbing gym. Nadia is asked to hand "
        "over a rope from the equipment bin. One rope in the bin is frayed and would snap "
        "under load; the other is new. The frayed rope has been tagged with a red marker. "
        "Nadia sees no marker on the rope she picks up, so Nadia believes the rope she is "
        "handing over is the new one. In fact a staff member had forgotten to mark the "
        "frayed rope, and the rope is the frayed one. Nadia hands over the rope. The "
        "climber uses it and falls, breaking an arm."),
}
EX_PRINT = {
    "neutral": (
        "Omar works at a print shop with a colleague. A jug beside the press holds either "
        "water or a solvent that ruins a print run. The jug is labelled \"water\". Omar "
        "reads the label, so Omar believes the jug holds water. The jug does hold water. "
        "Omar tops up the press from the jug. The run comes out clean."),
    "intentional": (
        "Omar works at a print shop with a colleague whose promotion he resents. A jug "
        "beside the press holds either water or a solvent that ruins a print run. The jug "
        "is labelled \"solvent\". Omar reads the label, so Omar believes the jug holds "
        "solvent. The jug does hold solvent. Omar tops up the press from the jug. The "
        "colleague's run is destroyed."),
}

BELIEF_CUE = (
    "Before answering, consider what the character believed about the situation at the "
    "moment of acting.")

WORKED_EXAMPLE = (
    "Here is a worked example of the distinction that matters, on two versions of the "
    "same situation.\n\n"
    "Version A:\n{attempted}\n\n"
    "Version B:\n{accidental}\n\n"
    "Reasoning: in Version A the character believed the rope was unsafe and handed it "
    "over anyway; no one was hurt, but that was luck, and the belief and the choice were "
    "the character's own. In Version B the character believed the rope was safe; someone "
    "was badly hurt, but the harm followed from a mislabelling the character had no way "
    "to know about. The two versions differ in what the character believed and intended, "
    "and they differ in what happened, and those are two separate things."
).format(**EX_CLIMB)

INTENT_PRINCIPLE = (
    "The principle to apply: moral judgment tracks what the character believed and "
    "intended, not whether harm happened to occur. A character who believed they were "
    "causing harm is highly blameworthy even when the harm does not occur. A character "
    "who believed they were doing something harmless is barely blameworthy even when "
    "serious harm does occur. Where the belief and the outcome point in different "
    "directions, weight the belief.")

# Levels 1-5 are the pre-registered cumulative curriculum. Levels 6-8 are the
# non-cumulative ablation added afterwards (each component alone against baseline), because
# L5 otherwise confounds the explicit principle with repair of whatever L4 did. L2 already
# is "instruction only", so it serves as the fourth ablation cell and is not duplicated.
COMPONENTS = {
    1: (),
    2: ("cue",),
    3: ("cue", "worked"),
    4: ("cue", "worked", "fewshot"),
    5: ("cue", "worked", "fewshot", "principle"),
    6: ("worked",),
    7: ("fewshot",),
    8: ("principle",),
}
LEVEL_NAMES = {
    1: "baseline",
    2: "belief_cue",
    3: "worked_example",
    4: "few_shot_adult",
    5: "intent_principle",
    6: "ABL_worked_only",
    7: "ABL_fewshot_only",
    8: "ABL_principle_only",
}
CUMULATIVE = (1, 2, 3, 4, 5)
ABLATION = (6, 7, 8)


def _load(mod_file, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(CODE_DIR, mod_file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def scale_label(norm, s_min, s_max):
    """Adult normalised value -> an integer on this template's own scale."""
    return int(round(s_min + norm * (s_max - s_min)))


EX_ITEMS = [("neutral", EX_PRINT["neutral"]), ("accidental", EX_CLIMB["accidental"]),
            ("attempted", EX_CLIMB["attempted"]), ("intentional", EX_PRINT["intentional"])]


def few_shot_block(beh, template, source, s_min, s_max):
    """Four labelled examples spanning the 2x2, on held-out vignettes.

    Both example scenarios contain an intent contrast (climb: attempted vs accidental;
    print: neutral vs intentional), so the mapping a model could imitate is not
    "scenario -> rating". Ratings are the adult profile expressed on the scale the target
    item will be rated on, since a label on the wrong scale teaches the wrong thing.

    Each example's question is rendered from that example's OWN text. `human_verbatim`
    interpolates the agent name, so reusing the target item's question here asked "How
    permissible was Grace's action?" underneath a story about Nadia — incoherent examples
    on 1 of the 7 templates. Building the question per example fixes it and costs nothing
    on the six templates whose wording has no agent slot.
    """
    out = ["Here are four examples answered the way a thoughtful adult answers them."]
    for cond, text in EX_ITEMS:
        base, lo, hi = beh.build_prompt(text, template, source)
        q = base[len(text.strip()):].strip()
        out.append(f"\nStory: {text}\n{q}\nAnswer: {scale_label(ADULT[cond], lo, hi)}")
    return "\n".join(out)


def check_fewshot_polarity(beh, templates, sources=("YS2008", "YS2009", "YS2011")):
    """Do the L4 labels actually encode attempted > accidental AFTER normalisation?

    The adult profile is stored in blame-normalised units while the templates run on three
    different native scales, one of which (the YS2008 permissibility anchor, 1-3) is the
    kind of scale that has been inverted in this project before. A silent inversion here
    would make the few-shot block teach the opposite of the intended lesson and would
    produce exactly the symptom it would be tempting to read psychologically: L4 more
    negative than baseline. So this runs before every scoring run and raises rather than
    warns.
    """
    rows, bad = [], []
    for tmpl in templates:
        for src in sources:
            _, lo, hi = beh.build_prompt("Grace put it in the coffee.", tmpl, src)
            lab = {c: scale_label(ADULT[c], lo, hi) for c, _ in EX_ITEMS}
            nrm = {c: beh.normalize(v, lo, hi) for c, v in lab.items()}
            implied = nrm["attempted"] - nrm["accidental"]
            rows.append((tmpl, src, lo, hi, lab, nrm, implied))
            if implied <= 0:
                bad.append(f"{tmpl}/{src}: implied contrast {implied:+.3f}")
    if bad:
        raise SystemExit("L4 few-shot labels do not encode attempted > accidental in "
                         "blame terms:\n  " + "\n  ".join(bad))
    return rows


def build_curriculum_prompt(beh, row, template, level):
    """Level-1 prompt from 03_behavioral, with scaffolding prepended for levels 2-5.

    The story and the question keep their original wording and their original adjacency
    (question last, immediately before generation) at every level, so nothing about the
    measurement changes as the scaffolding grows.
    """
    base, s_min, s_max = beh.build_prompt(row["text"], template, row["source"])
    story = row["text"].strip()
    question = base[len(story):].strip() if base.startswith(story) else base
    comps = COMPONENTS[level]
    if not comps:
        return base, s_min, s_max

    pre = []
    if "cue" in comps:
        pre.append(BELIEF_CUE)
    if "worked" in comps:
        pre.append(WORKED_EXAMPLE)
    if "fewshot" in comps:
        pre.append(few_shot_block(beh, template, row["source"], s_min, s_max))
    if "principle" in comps:
        pre.append(INTENT_PRINCIPLE)
    head = "\n\n".join(pre)
    tail = "Now rate this story.\n\n" if "fewshot" in comps else ""
    return f"{head}\n\n{tail}{story}\n\n{question}", s_min, s_max


def write_prespec():
    if os.path.exists(PRESPEC):
        return
    doc = __doc__.split("PRE-SPECIFIED PREDICTIONS AND BOTH READINGS", 1)[1]
    doc = doc.split("=====", 1)[1] if "=====" in doc else doc
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(PRESPEC, "w") as f:
        f.write("# W4 prompt curriculum — pre-specified readings\n\n")
        f.write("Written by `code/experiments/54_w4_prompt_curriculum.py` at the start of "
                "the first `--run`, before any curriculum result existed. Never "
                "overwritten on later runs. Both readings (prompting works / prompting "
                "also fails) are committed here so neither can be adopted after the "
                "fact.\n\n")
        f.write("```\n" + doc.split("Method:", 1)[0].rstrip() + "\n```\n\n")
        f.write("## Relation to W3\n\nW3 intervened on the representation and found it "
                "causally inert: 3.2–7.2 SD of probe-margin displacement of the intent "
                "code, |Δcontrast| ≤ 0.015, against an outcome positive control that "
                "moves the contrast 0.232–0.259 at the same depths. W4 intervenes at the "
                "input instead. The pair localises the failure; neither result alone "
                "does.\n")
    print(f"  wrote pre-spec {os.path.relpath(PRESPEC, tc.ROOT)}")


def write_levels_md(beh, rows, templates):
    """Dump every level's prompt verbatim for one item, so the manipulation is auditable."""
    row = next((r for r in rows if r["condition"] == "attempted"), rows[0])
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(LEVELS_MD, "w") as f:
        f.write("# W4 curriculum levels — verbatim prompts\n\n")
        f.write(f"Shown on item `{row['story_id']}` ({row['condition']}) with template "
                f"`{templates[0]}`. L1–L5 are cumulative; L6–L8 are the non-cumulative "
                "ablation (one component each, against the same L1 baseline).\n")
        f.write("\n## L4 label polarity check\n\n"
                "The adult profile is stored in blame-normalised units and the templates "
                "run on three native scales, one of which (the YS2008 permissibility "
                "anchor) is phrased 1 = completely permissible → 3 = completely "
                "impermissible, i.e. ascending in condemnation like every other template. "
                "This table is the check that the few-shot labels really do encode "
                "attempted > accidental in blame terms on each scale; the run aborts if "
                "any row is non-positive.\n\n"
                "| template | source | scale | neutral | accidental | attempted "
                "| intentional | implied contrast (normalised) |\n"
                "|---|---|---|---:|---:|---:|---:|---:|\n")
        seen_scales = set()
        for t, src, lo, hi, lab, nrm, imp in check_fewshot_polarity(beh, templates):
            if (t, lo, hi) in seen_scales:
                continue
            seen_scales.add((t, lo, hi))
            f.write(f"| `{t}` | {src} | {lo}–{hi} | "
                    + " | ".join(str(lab[c]) for c, _ in EX_ITEMS)
                    + f" | **{imp:+.3f}** |\n")
        f.write(f"\nAdult reference contrast for comparison: "
                f"{ADULT['attempted'] - ADULT['accidental']:+.3f}.\n")
        for lv in sorted(LEVEL_NAMES):
            p, lo, hi = build_curriculum_prompt(beh, row, templates[0], lv)
            f.write(f"\n## L{lv} — {LEVEL_NAMES[lv]}  (scale {lo}–{hi}, "
                    f"{len(p.split())} words)\n\n```\n{p}\n```\n")
    print(f"  wrote {os.path.relpath(LEVELS_MD, tc.ROOT)}")


def score_level(beh, reg11, backend, rows, templates, level):
    """All items x templates at one curriculum level -> per-scenario-group cell means."""
    acc = defaultdict(dict)
    norms, extremes = [], 0
    for tmpl in templates:
        for row in rows:
            prompt, s_min, s_max = build_curriculum_prompt(beh, row, tmpl, level)
            raw, norm = backend.rate(prompt, s_min, s_max, 1, 0.0)
            g = tc.scenario_group_of(row["story_id"])
            acc[f"{tmpl}:{g}"].setdefault(row["condition"], []).append(float(norm))
            norms.append(float(norm))
            if float(norm) <= 0.02 or float(norm) >= 0.98:
                extremes += 1
    pooled = {k: {c: float(np.mean(v)) for c, v in conds.items()}
              for k, conds in acc.items()}
    return pooled, dict(rating_std=float(np.std(norms)) if norms else 0.0,
                        extreme_frac=extremes / max(len(norms), 1),
                        n_ratings=len(norms))


def group_index(pooled):
    """scenario group -> the `template:group` keys it contributes.

    The bootstrap unit is the scenario GROUP, not the template x group cell. Resampling
    cells would treat the same vignette rated under 7 templates as 7 independent
    observations and shrink every interval; the design has 53 groups however many
    templates are run over them.
    """
    idx = defaultdict(list)
    for k in pooled:
        idx[k.split(":", 1)[1]].append(k)
    return idx


def cell_mean_of(pooled, idx, groups, cond):
    """Mean of one cell over a (possibly resampled, so duplicated) list of groups."""
    vals = [pooled[k][cond] for g in groups for k in idx.get(g, [])
            if cond in pooled.get(k, {})]
    return float(np.mean(vals)) if vals else float("nan")


def contrast_of(pooled, idx=None, groups=None):
    idx = group_index(pooled) if idx is None else idx
    groups = list(idx) if groups is None else groups
    return (cell_mean_of(pooled, idx, groups, "attempted")
            - cell_mean_of(pooled, idx, groups, "accidental"))


def summarize(reg11, model_id, lv, pooled, base_pooled, qc, n_templates, B, seed):
    """One output row: point estimates, group-level bootstrap CIs, per-cell deltas.

    The per-cell deltas exist for the ceiling-compression check. A contrast gain produced
    by `accidental` falling is a different finding from one produced by `attempted`
    rising, and a gain produced by all four cells sliding together is compression rather
    than re-weighting of either factor — the same caveat that governs the W3
    difference-of-means direction, so it is computed here and printed in the summary table
    rather than left to be reconstructed from the logs.
    """
    m = reg11.cell_means(pooled)
    b0, b_int, b_out, b_inter = reg11.coeffs_from_means(m)
    idx = group_index(pooled)
    groups = list(idx)
    pt, lo, hi = tc.bootstrap(groups, lambda g: contrast_of(pooled, idx, g),
                              B=B, seed=seed)
    rec = dict(model=model_id, level=lv, level_name=LEVEL_NAMES[lv],
               cumulative=(lv in CUMULATIVE), components="+".join(COMPONENTS[lv]) or "none",
               n_templates=n_templates, n_groups=len(groups),
               n_cells=len(pooled), contrast=pt, contrast_lo=lo, contrast_hi=hi)
    if base_pooled:
        bidx = group_index(base_pooled)
        shared = [g for g in groups if g in bidx]
        # Paired over scenario groups: the same resampled groups enter both levels, so the
        # CI is on the within-group change and not on two independent estimates differenced.
        d_pt, d_lo, d_hi = tc.bootstrap(
            shared,
            lambda g: contrast_of(pooled, idx, g) - contrast_of(base_pooled, bidx, g),
            B=B, seed=seed + 1)
        bm = reg11.cell_means(base_pooled)
        rec.update(d_contrast=d_pt, d_lo=d_lo, d_hi=d_hi,
                   **{f"d_{c}": m.get(c, float("nan")) - bm.get(c, float("nan"))
                      for c in tc.CELLS})
    else:
        rec.update(d_contrast=float("nan"), d_lo=float("nan"), d_hi=float("nan"),
                   **{f"d_{c}": float("nan") for c in tc.CELLS})
    rec.update(b_intent=b_int, b_outcome=b_out, b_interaction=b_inter, b0=b0,
               **{f"mean_{c}": m.get(c, float("nan")) for c in tc.CELLS}, **qc)
    return rec, m


def run_model(beh, reg11, model_id, rows, templates, levels, B, seed, backend_name="hf",
              known=None):
    # mock exists only to exercise the plumbing off-GPU; it has planted effects and its
    # rows must never reach the report (the submit script strips them).
    cls = beh.MockBackend if backend_name == "mock" else beh.HFBackend
    backend = cls(model_id, scoring="logprob")
    recs, cell_recs = [], []
    # Baseline for the paired deltas is always L1, loaded from disk when this run does not
    # rescore it, so an ablation-only or L4-only rerun still reports deltas against the
    # same reference as the original run.
    base_pooled = (known or {}).get((model_id, BASE_LEVEL))
    try:
        for lv in levels:
            print(f"  L{lv} {LEVEL_NAMES[lv]} ...", flush=True)
            pooled, qc = score_level(beh, reg11, backend, rows, templates, lv)
            if lv == BASE_LEVEL:
                base_pooled = pooled
            rec, m = summarize(reg11, model_id, lv, pooled, base_pooled, qc,
                               len(templates), B, seed)
            pt, lo, hi = rec["contrast"], rec["contrast_lo"], rec["contrast_hi"]
            d_pt, d_lo, d_hi = rec["d_contrast"], rec["d_lo"], rec["d_hi"]
            recs.append(rec)
            cells_lv = [dict(model=model_id, level=lv, level_name=LEVEL_NAMES[lv],
                             group=k, condition=c, mean=v)
                        for k, conds in pooled.items() for c, v in conds.items()]
            cell_recs += cells_lv
            # Flush per level, not per model: this runs on a preemptable partition, and a
            # model killed at L4 should not cost the four levels already scored.
            append_csv(OUT_CSV, [rec], ("model", "level"))
            append_csv(CELL_CSV, cells_lv, ("model", "level", "group", "condition"))
            print(f"    contrast {pt:+.4f} [{lo:+.4f},{hi:+.4f}]  "
                  f"Δvs L1 {d_pt:+.4f} [{d_lo:+.4f},{d_hi:+.4f}]  "
                  f"cells " + " ".join(f"{c[:4]}={m.get(c, float('nan')):.3f}"
                                       for c in tc.CELLS)
                  + "  Δcells " + " ".join(f"{c[:4]}={rec[f'd_{c}']:+.3f}"
                                           for c in tc.CELLS)
                  + f"  std={qc['rating_std']:.4f} extreme={qc['extreme_frac']:.2f}",
                  flush=True)
    finally:
        try:
            import torch, gc
            del backend; gc.collect(); torch.cuda.empty_cache()
        except Exception:
            pass
    return recs, cell_recs


def load_cells():
    """(model, level) -> pooled cell means, from the per-cell CSV written during scoring.

    Everything in a summary row except the QC columns is a function of these means, so the
    bootstrap can be corrected and the per-cell deltas added without touching a GPU.
    """
    if not os.path.exists(CELL_CSV):
        return {}
    out = defaultdict(dict)
    for r in csv.DictReader(open(CELL_CSV)):
        out[(r["model"], int(r["level"]))].setdefault(r["group"], {})[r["condition"]] = \
            float(r["mean"])
    return dict(out)


def recompute(reg11, B, seed, models=None, levels=None):
    """Rebuild summary rows from stored cell means. QC columns carry over unchanged."""
    known = load_cells()
    if not known:
        print(f"no {os.path.relpath(CELL_CSV, tc.ROOT)} to recompute from")
        return
    old = {(r["model"], int(r["level"])): r
           for r in csv.DictReader(open(OUT_CSV))} if os.path.exists(OUT_CSV) else {}
    recs = []
    for (mid, lv), pooled in sorted(known.items()):
        if (models and mid not in models) or (levels and lv not in levels):
            continue
        prev = old.get((mid, lv), {})
        qc = {k: float(prev[k]) if prev.get(k) not in (None, "") else float("nan")
              for k in ("rating_std", "extreme_frac", "n_ratings")}
        rec, _ = summarize(reg11, mid, lv, pooled, known.get((mid, BASE_LEVEL)), qc,
                           int(prev.get("n_templates") or 0), B, seed)
        recs.append(rec)
        print(f"  {mid} L{lv}: contrast {rec['contrast']:+.4f} "
              f"[{rec['contrast_lo']:+.4f},{rec['contrast_hi']:+.4f}]  "
              f"Δ {rec['d_contrast']:+.4f} [{rec['d_lo']:+.4f},{rec['d_hi']:+.4f}]")
    if recs:
        append_csv(OUT_CSV, recs, ("model", "level"))
        print(f"  rewrote {len(recs)} row(s) in {os.path.relpath(OUT_CSV, tc.ROOT)}")


def append_csv(path, recs, keyfields):
    """Append-with-replace: a rerun of one (model, level) overwrites just that row."""
    old = []
    if os.path.exists(path):
        old = [r for r in csv.DictReader(open(path))]
    key = lambda r: tuple(str(r[k]) for k in keyfields)
    new_keys = {key(r) for r in recs}
    merged = [r for r in old if key(r) not in new_keys] + [
        {k: v for k, v in r.items()} for r in recs]
    fields = list(recs[0])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in merged:
            w.writerow({k: r.get(k, "") for k in fields})


def print_plan(models, rows, templates, levels, beh):
    print("\n=== W4 PROMPT CURRICULUM PLAN (dry run) ===")
    row = rows[0]
    print(f"{'level':22} {'~prompt words':>14}")
    for lv in levels:
        p, _, _ = build_curriculum_prompt(beh, row, templates[0], lv)
        print(f"L{lv} {LEVEL_NAMES[lv]:19} {len(p.split()):>14}")
    per = len(rows) * len(templates) * len(levels)
    print(f"\nitems={len(rows)} templates={len(templates)} levels={len(levels)}")
    print(f"forward passes per model = {per:,}; {len(models)} models = "
          f"{per * len(models):,}")
    for m in models:
        print(f"  {m}")
    print("\nLaunch: sbatch code/submit_w4_curriculum.sh")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=ENGAGED)
    ap.add_argument("--templates", nargs="+", default=None,
                    help="default: the designed 7-template factorial basis")
    ap.add_argument("--levels", nargs="+", type=int, default=sorted(LEVEL_NAMES))
    ap.add_argument("--limit-groups", type=int, default=0,
                    help="smoke test: use only the first N scenario groups")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", choices=["hf", "mock"], default="hf",
                    help="mock: off-GPU plumbing check only; rows are not reportable")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--recompute", action="store_true",
                    help="rebuild summary rows from the stored cell means (no GPU): "
                         "corrects CIs and adds per-cell deltas without rescoring")
    ap.add_argument("--check-polarity", action="store_true",
                    help="print the L4 label table per template x scale and exit")
    a = ap.parse_args()

    beh = _load("03_behavioral.py", "behavioral")
    reg11 = _load("11_interaction_regression.py", "interaction_reg")
    templates = a.templates or list(beh.FACTORIAL_TEMPLATES)
    rows = beh.load_dataset()

    if a.backend == "mock":
        # The mock backend has planted effects. Its rows are for exercising the plumbing
        # and must never share a file with reportable ones.
        global OUT_CSV, CELL_CSV
        OUT_CSV = OUT_CSV.replace(".csv", "_MOCK.csv")
        CELL_CSV = CELL_CSV.replace(".csv", "_MOCK.csv")
        print(f"[mock] writing to {os.path.basename(OUT_CSV)} — not reportable")

    if a.check_polarity:
        for t, s, lo, hi, lab, nrm, imp in check_fewshot_polarity(beh, templates):
            print(f"{t:16} {s:8} scale {lo}-{hi}  "
                  + " ".join(f"{c[:5]}={lab[c]}" for c, _ in EX_ITEMS)
                  + f"  norm(attempted)={nrm['attempted']:+.3f} "
                  f"norm(accidental)={nrm['accidental']:+.3f}  implied {imp:+.3f}")
        return
    if a.recompute:
        recompute(reg11, a.boot, a.seed,
                  models=set(a.models) if a.templates or a.models != ENGAGED else None,
                  levels=set(a.levels) if a.levels != sorted(LEVEL_NAMES) else None)
        return
    if a.limit_groups:
        keep, seen = [], []
        for r in rows:
            g = tc.scenario_group_of(r["story_id"])
            if g not in seen:
                if len(seen) >= a.limit_groups:
                    continue
                seen.append(g)
            keep.append(r)
        rows = keep

    if not a.run:
        print_plan(a.models, rows, templates, a.levels, beh)
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    write_prespec()
    check_fewshot_polarity(beh, templates)   # raises rather than warns
    write_levels_md(beh, rows, templates)

    done = set()
    if os.path.exists(OUT_CSV) and not a.force:
        done = {(r["model"], int(r["level"])) for r in csv.DictReader(open(OUT_CSV))}
        if done:
            print(f"[resume] {len(done)} (model, level) cell(s) already scored")

    known = load_cells()
    for mid in a.models:
        todo = [lv for lv in a.levels if (mid, lv) not in done]
        if not todo:
            print(f"[skip] {mid}: all levels present")
            continue
        # Every delta references L1. Rescore it only if it is not already on disk, so an
        # L4/L5 or ablation-only rerun does not pay for the baseline again.
        if BASE_LEVEL not in todo and (mid, BASE_LEVEL) not in known:
            todo = [BASE_LEVEL] + todo
        print(f"\n{'='*66}\n {mid}  levels={todo}\n{'='*66}", flush=True)
        try:
            recs, _ = run_model(beh, reg11, mid, rows, templates, todo,
                                a.boot, a.seed, a.backend, known)
        except Exception as e:
            print(f"!! FAILED {mid}: {e}")
            import traceback; traceback.print_exc()
            continue
        print(f"  {len(recs)} level(s) written to "
              f"{os.path.relpath(OUT_CSV, tc.ROOT)}", flush=True)


if __name__ == "__main__":
    main()
