#!/usr/bin/env python3
"""W8 paper assembly: limitations table, data provenance appendix, reproducibility manifest.

Writes three documents into outputs/paper/. Two of them are curated prose and one is
derived from the repository, and the split matters:

  LIMITATIONS.md          curated. Every row names the constraint, what it forbids
                          claiming, where the evidence is, and what would lift it.
  APPENDIX_PROVENANCE.md  curated narrative over CONTAMINATION_REPAIR.md,
                          LABEL_AUDIT_MANUAL.md and the archived pre-repair master,
                          with the archive's checksum verified live.
  REPRODUCIBILITY.md      derived. Commit hash per figure comes from git log on the
                          figure path and on its producing script; every claimed
                          bootstrap seed and B is checked against the script source, and
                          a mismatch is printed in the table rather than silently fixed.

Anything asserted here that can be checked is checked, and the check result is printed in
the document. A manifest that cannot fail is not evidence of reproducibility.
"""
import os, re, sys, csv, hashlib, subprocess, argparse

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)
import tom_common as tc  # noqa: E402

PAPER = os.path.join(tc.ROOT, "outputs", "paper")
FIGDIR = os.path.join(tc.ROOT, "outputs", "figures_final")
PREREPAIR = os.path.join(tc.ROOT, "dataset", "master", "_prerepair_backup",
                         "moral_2x2_master_CONTAMINATED_20260619.csv")
PREREPAIR_MD5 = "5dd904a7609628553319da4acab02f25"

# --------------------------------------------------------------------- limitations
# (short name, what it constrains -- phrased as what may NOT be claimed, evidence, fix)
LIMITATIONS = [
    ("Scale ceiling at 14B, and half the roster is one family",
     "No claim about frontier-scale open models, and no claim that the effect is "
     "family-general. 10 of the 20 open models are Qwen2.5; the largest is 14B. A "
     "family-level idiosyncrasy would look like a general result at this composition.",
     "`outputs/tom_benchmarks/tom_vs_contrast.csv` (roster); "
     "`outputs/analysis/ROSTER_70B_FEASIBILITY.md`",
     "Llama-3.3-70B-Instruct is the field reference (OmniToM 2026, ToMBench); the "
     "gemma-3-27B / Qwen3-32B band is the cheaper middle option costed in the "
     "feasibility note."),
    ("Zephyr's shift is DPO-dominant while OLMo's and Tülu's are SFT-dominant",
     "The locus claim may not be stated as \"SFT, not preference optimization\". SFT "
     "alone is sufficient to induce the bias in all three families (drops of 0.15–0.59 "
     "from a near-zero base), but the SFT share is 85% (OLMo), 65% (Tülu) and 27% "
     "(Zephyr, whose shift is 73% at DPO). The relative contribution is recipe-dependent.",
     "`outputs/experiments/CHECKPOINT_STAGE_SHARES.md`; "
     "`outputs/experiments/checkpoint_dissection_writeup.md`",
     "More families with public intermediate checkpoints, and a recipe-matched pair "
     "(same base, SFT-only vs SFT+DPO) to separate stage from recipe."),
    ("Span-matched baselines downgrade C2, and these stimuli contain no outcome-free "
     "belief position",
     "The position dissociation is supporting, not headline. On span-matched TF-IDF the "
     "intent−outcome difference at the belief position is +0.087, which does not support "
     "\"intent represented early, outcome inferred late\". Outcome decoding of 0.82 at "
     "the belief position sits against a span-matched baseline of 0.58, and the stimuli "
     "give no position at which the text carries belief but no outcome information.",
     "`outputs/probe/gap_over_surface_span_matched.csv`; "
     "`outputs/analysis/C2_SOURCE_SPLIT_BELIEF_LAST.md`",
     "Stimuli written with a belief clause that is uninformative about the outcome — a "
     "new stimulus set, not a reanalysis."),
    ("Human anchors are digitized from published figures",
     "Human cell means are not raw data. They are read off figures in Young et al. "
     "(2007) and the child developmental sources, so anchor comparisons inherit "
     "digitization error and cannot support fine distinctions between adjacent bands.",
     "`outputs/human_anchor_comparison.NOTES.md`; `code/digitize_cushman_calibrated.py`",
     "Original per-subject data from the authors, or a replication with human "
     "participants on these exact stimuli."),
    ("Child developmental data covers 2 of the 4 cells",
     "No claim that models match a child profile across the full 2×2. The child ladders "
     "constrain only the cells they measure; the model-vs-child comparison is a "
     "partial-profile comparison.",
     "`outputs/human_digitized/`; `outputs/human_punish/`",
     "A developmental dataset with all four cells, or restricting the claim to the "
     "measured cells (currently done, and it is why the anchor is not chosen)."),
    ("The two source scales only moderately agree (r = 0.71)",
     "Cross-source averaging of `human_verbatim` mixes instruments. YS2008 is "
     "permissibility 1–3, YS2009 is blame 1–4; after 0–1 normalisation pooled r = 0.71, "
     "Bland–Altman bias −0.06, 95% LoA [−0.44, +0.33], and only 1 of 28 models clears "
     "the agreement criterion.",
     "`outputs/SCALE_REPLICATION.md`",
     "Re-run both source instruments on the full roster, or report per-instrument "
     "results throughout (`tom_common.load_cells` already prefers YS2008-only for "
     "`human_verbatim`)."),
    ("Effective n is 53 CV groups, not 298 items",
     "Item-count-based power claims are not available. All 24 YS2009 scenarios are "
     "reprints of YS2008 scenarios, so 77 scenario ids collapse to 53 CV groups; every "
     "probe, bootstrap and mixed-effects fit is at that resolution.",
     "`dataset/master/CONTAMINATION_REPAIR.md` §6; `scenario_group` column",
     "New vignettes. This is a hard ceiling on the existing stimulus set and it is why "
     "bootstraps resample groups rather than items."),
    ("Closed models are behaviour-only",
     "No representational claim about closed models: no probes, no steering, no layer "
     "analysis, since the APIs do not expose hidden states. Their ToM accuracy and "
     "contrasts are reported standalone and are not correlated against representational "
     "measures.",
     "`outputs/experiments/CLOSED_MODEL_SELECTION.md`; `outputs/closed_reasoning/`",
     "Nothing available without model access. The open roster carries every "
     "representational claim."),
    ("M2 is restricted to depths where specificity is resolvable",
     "The layer-sweep null may not be stated as \"inert at every depth\". At shallow "
     "layers the intent and outcome directions are not separable enough for the "
     "comparison to mean anything, so the claim is: inert at depths where specificity is "
     "resolvable (peak-intent and deeper), with shallow layers uninformative rather than "
     "supporting.",
     "`outputs/experiments/W3_STEERING_SUMMARY.md` §5; `w3_layersweep_*.csv`",
     "A direction-fitting method that separates intent from outcome at shallow depths, "
     "or an intervention that does not require them to be separable."),
]

# ------------------------------------------------------------------ figures / tables
# script that produces each figure; the commit hash is read from git, not asserted here.
FIG_SCRIPTS = {
    "checkpoint_dissection.png": "code/experiments/16_checkpoint_dissection.py",
    "gap_over_surface_span_matched.png": "code/experiments/33_gap_dissociation_figure.py",
    "interaction_forest.png": "code/experiments/39_mixed_effects_2x2.py",
    "item_level_dissociation.png": "code/experiments/41_item_level_dissociation.py",
    "layerwise_curves.png": "code/02_probe.py",
    "rsa_similarity_heatmap.png": "code/experiments/24_rsa_cka.py",
    "rsa_convergence_scatter.png": "code/04_link_analysis.py",
    "tom_vs_contrast.png": "code/experiments/42_tom_vs_contrast.py",
    "master_developmental_ladder_digitized_openonly.png": "code/29_dual_human_ladders.py",
    "master_developmental_ladder_punish_openonly.png": "code/29_dual_human_ladders.py",
    "master_developmental_ladder_text_reported_openonly.png": "code/29_dual_human_ladders.py",
    "w3_steering_dose_OLMo.png": "code/experiments/48_w3_causal_steering.py",
    "w3_steering_dose_Qwen.png": "code/experiments/48_w3_causal_steering.py",
    "w3_layersweep_OLMo.png": "code/experiments/48_w3_causal_steering.py",
    "w3_layersweep_Qwen.png": "code/experiments/48_w3_causal_steering.py",
    "w3_manipulation_OLMo.png": "code/experiments/48_w3_causal_steering.py",
    "w3_manipulation_Qwen.png": "code/experiments/48_w3_causal_steering.py",
    "w3_prose_rating.png": "code/experiments/51_w3_prose_rating.py",
    "w4_curriculum.png": "code/experiments/55_w4_summary.py",
    "w7_bruneau_selectivity.png": "code/experiments/56_w7_bruneau.py",
    "reasoning_dose_response.png": "code/experiments/52_closed_reasoning_dose.py",
}

# (table / document, producing script, source CSV)
TABLES = [
    ("outputs/stats/MIXED_EFFECTS_2x2.md", "code/experiments/39_mixed_effects_2x2.py",
     "outputs/stats/mixed_effects_2x2.csv"),
    ("outputs/experiments/CHECKPOINT_DISSECTION.md",
     "code/experiments/16_checkpoint_dissection.py",
     "outputs/experiments/checkpoint_dissection.csv"),
    ("outputs/experiments/CHECKPOINT_STAGE_SHARES.md",
     "code/experiments/47_checkpoint_stage_shares.py",
     "outputs/experiments/checkpoint_stage_shares.csv"),
    ("outputs/experiments/W3_STEERING_SUMMARY.md", "code/experiments/50_w3_summary.py",
     "outputs/experiments/w3_steering_*.csv"),
    ("outputs/experiments/W3_PROSE_RATING.md", "code/experiments/51_w3_prose_rating.py",
     "outputs/experiments/w3_prose_items_*.csv"),
    ("outputs/experiments/W4_CURRICULUM.md", "code/experiments/55_w4_summary.py",
     "outputs/experiments/w4_prompt_curriculum.csv"),
    ("outputs/experiments/W7_BRUNEAU.md", "code/experiments/56_w7_bruneau.py",
     "outputs/experiments/w7_bruneau_probes.csv"),
    ("outputs/tom_benchmarks/TOM_VS_CONTRAST.md", "code/experiments/42_tom_vs_contrast.py",
     "outputs/tom_benchmarks/tom_vs_contrast.csv"),
    ("outputs/probe/C2_SOURCE_SPLIT.md",
     "code/experiments/43_c2_source_split.py", "outputs/probe/*_probe.csv"),
    ("outputs/analysis/C2_SOURCE_SPLIT_BELIEF_LAST.md",
     "code/experiments/44_span_matched_gaps.py",
     "outputs/probe/gap_over_surface_span_matched.csv"),
    ("outputs/stats/FLOOR_DERIVATION.md", "code/experiments/40_derive_floors.py",
     "outputs/stats/floor_derivation.csv"),
    ("outputs/SCALE_REPLICATION.md", "code/experiments/31_scale_replication.py",
     "outputs/behavior/*item_means*.csv"),
]

# (analysis, script, resampling unit, B, seed, source declaration of B, of the seed).
# The last two are literal substrings that must appear in the script, so a row cannot drift
# away from the code without the manifest saying so.
BOOTSTRAPS = [
    ("Shared resampler used by every scenario-group CI", "code/tom_common.py",
     "scenario group (`keys`)", 2000, 0,
     "def bootstrap(keys, statfn, B=2000, seed=0", "def bootstrap(keys, statfn, B=2000, seed=0"),
    ("Cell means, contrasts, coefficient CIs", "code/06_stats.py",
     "scenario group", 2000, 0, '"--boot", type=int, default=2000',
     "def bootstrap(keys, statfn, B=2000, seed=0"),
    ("Saturated 2x2 coefficient CIs (b0/intent/outcome/interaction)",
     "code/11_interaction_regression.py", "scenario group", 2000, 0,
     '"--boot", type=int, default=2000', "def boot_coeff(cells_scen, which, B, seed)"),
    ("Base vs instruct paired difference", "code/12_base_vs_instruct_test.py",
     "stacked scenario means", 5000, 0, '"--boot", type=int, default=5000',
     "np.random.default_rng(0)"),
    ("Item-level intent-margin slope CI (a reported null)",
     "code/experiments/41_item_level_dissociation.py", "item", 10000, 0,
     "N_BOOT = 10000", "def boot_ci(x, y, n_boot=N_BOOT, seed=0)"),
    ("ToM-vs-contrast correlation CIs", "code/experiments/42_tom_vs_contrast.py",
     "model", 10000, 0, "N_BOOT = 10000", "def boot_ci(x, y, n_boot=N_BOOT, seed=0)"),
    ("RSA convergence CI (a reported null)", "code/experiments/24_rsa_cka.py",
     "model pair", 1000, 0, "def convergence_test(sim_rows, outdir, n_boot=1000)",
     "np.random.default_rng(0)"),
    ("Base→instruct within-family deltas", "code/experiments/18_mini_dissection.py",
     "scenario group", 2000, 0, '"--boot", type=int, default=2000',
     "def paired_delta_ci(base_cells, inst_cells, B=2000, seed=0)"),
    ("Engagement floor derivation", "code/experiments/40_derive_floors.py",
     "model", None, 0, None, '"--seed", type=int, default=0'),
    ("W4 contrast and paired Δ vs L1", "code/experiments/54_w4_prompt_curriculum.py",
     "scenario group", 2000, 0, '"--boot", type=int, default=2000',
     '"--seed", type=int, default=0'),
    ("W7 moral-vs-non-moral interaction", "code/experiments/56_w7_bruneau.py",
     "Bruneau item pair", 1000, 0, '"--boot", type=int, default=1000',
     '"--seed", type=int, default=0'),
]


def git(*args):
    try:
        return subprocess.run(["git"] + list(args), cwd=tc.ROOT, capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def last_commit(path):
    h = git("log", "-1", "--format=%h %ad", "--date=short", "--", path)
    return h or "— (uncommitted)"


def verify_decl(script, decl):
    """Is the declaration this row claims actually present in the script?

    Substring rather than pattern matching, so the manifest quotes real source text. A row
    whose declaration has been edited away reports 'declaration changed' instead of
    silently continuing to assert a value the code no longer uses.
    """
    if decl is None:
        return "n/a"
    p = os.path.join(tc.ROOT, script)
    if not os.path.exists(p):
        return "script missing"
    return "ok" if decl in open(p, errors="ignore").read() else "declaration changed"


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_limitations(head):
    L = ["# Limitations", "",
         f"Repository at `{head}`. Each row states what the limitation forbids claiming, "
         "not merely that it exists — a limitations section that does not constrain any "
         "sentence in the paper is decoration. Where a constraint has already been "
         "applied to the wording of a result, the row says so.", "",
         "| # | limitation | what it forbids claiming | evidence | what would lift it |",
         "|---|---|---|---|---|"]
    for i, (name, forbids, ev, fix) in enumerate(LIMITATIONS, 1):
        L.append(f"| {i} | **{name}** | {forbids} | {ev} | {fix} |")
    L += ["", "## Three that a reviewer will reach for first", "",
          "**The Zephyr counterexample (2) is reported, not buried.** It is the single "
          "clearest constraint on the checkpoint result and it was found by rescoring "
          "rather than by argument. The claim that survives it — SFT alone suffices in all "
          "three families, the SFT-vs-preference-optimization split is recipe-dependent — "
          "is weaker than the original and is what the data support.", "",
          "**The C2 downgrade (3) cost a headline.** The position dissociation was the "
          "second-most quotable result in the project until the surface baselines were "
          "recomputed on span-matched text; at +0.087 it no longer supports the framing it "
          "was carrying, and the figure caption and readout say so in place of the old "
          "wording.", "",
          "**Effective n = 53 (7) is the binding constraint on every interval in the "
          "paper.** It is not a caveat about the stimulus set in the abstract; it is why "
          "bootstraps resample scenario groups, why the mixed-effects model has 53 "
          "grouping levels, and why null results are reported with a minimum meaningful "
          "effect size attached rather than as absence of evidence.", ""]
    p = os.path.join(PAPER, "LIMITATIONS.md")
    open(p, "w").write("\n".join(L) + "\n")
    print(f"  wrote {os.path.relpath(p, tc.ROOT)} ({len(LIMITATIONS)} rows)")


def write_provenance(head):
    ok = os.path.exists(PREREPAIR)
    live = md5(PREREPAIR) if ok else "—"
    match = (live == PREREPAIR_MD5)
    L = ["# Appendix: data provenance and the stimulus integrity audit", "",
         f"Repository at `{head}`.", "",
         "This appendix documents a stimulus defect that invalidated an earlier round of "
         "results, the repair, and the audit that caught a second independent defect while "
         "checking the first. It is written as a contribution because that is what it is: "
         "the artefact is archived, the detector is a script, the repair is in the builder "
         "rather than a post-hoc filter, and every number below is reproducible from the "
         "two files named here. Published moral-ToM stimulus sets are parsed from PDF "
         "appendices by many groups; the failure mode described in §1 is a property of "
         "those appendices, not of this project, and it is silent — the automated quality "
         "gate that was in place stayed green throughout.", "",
         "## 0. Artefacts", "",
         "| artefact | path | status |", "|---|---|---|",
         f"| Pre-repair master (archived, never regenerated) | "
         f"`{os.path.relpath(PREREPAIR, tc.ROOT)}` | "
         + (f"present, md5 `{live}` — **matches** the recorded checksum"
            if ok and match else
            (f"present, md5 `{live}` — **DOES NOT MATCH** recorded "
             f"`{PREREPAIR_MD5}`" if ok else "**MISSING**")) + " |",
         "| Repaired master | `dataset/master/moral_2x2_master.csv` | current, "
         f"last commit {last_commit('dataset/master/moral_2x2_master.csv')} |",
         "| Repair record | `dataset/master/CONTAMINATION_REPAIR.md` | "
         f"{last_commit('dataset/master/CONTAMINATION_REPAIR.md')} |",
         "| Non-circular label audit | `outputs/LABEL_AUDIT_MANUAL.md` | "
         f"{last_commit('outputs/LABEL_AUDIT_MANUAL.md')} |",
         "| Validation gate (10 checks, exits non-zero) | "
         "`code/experiments/28_validate_master.py` | "
         f"{last_commit('code/experiments/28_validate_master.py')} |",
         "| Contaminated-era outputs, quarantined | `outputs/_contaminated_20260726/`, "
         "`outputs/figures_final/_pending_rescore/` | retained, marked, excluded from "
         "figures |",
         "",
         "## 1. Defect 1 — trailing contamination, aligned with the outcome factor", "",
         "The factorial parser merged wrapped PDF lines by appending every following line "
         "that was not itself an item marker. In the source appendices the rating prompt, "
         "the next scenario's ALLCAPS tag and the next scenario's background all follow "
         "the last item with no blank line, so all three were glued onto item 6 — which is "
         "used by exactly the `accidental` and `intentional` cells. Both are harm cells, "
         "so the contamination was nearly collinear with the outcome factor: 96/154 "
         "(62.3%) of harm cells contaminated, 0/144 of no-harm cells.", "",
         "A parameter-free binary flag (\"does this story have an unrelated tail glued "
         "on\") predicted `outcome_label` at **0.966** accuracy before repair and "
         "**0.517** after, the latter being exactly the majority-class rate 154/298 and "
         "not 0.5 by coincidence. The 0.99–1.00 outcome decoding and the size of "
         "`b_outcome` in the pre-repair results were both confounded with this artefact.",
         "",
         "Two things this number does not show, stated because it would be easy to "
         "overclaim: with zero flags the detector is a constant predictor, so the collapse "
         "to 0.517 shows only that the contamination signal is gone, not that harm and "
         "no-harm cells are surface-matched. Residual surface predictability of outcome on "
         "the repaired master is **0.755** (word 1–2gram TF-IDF, "
         "`outputs/probe/surface_baseline.csv`). Both numbers belong in any statement "
         "about what the repair achieved.", "",
         "Three different counters appear in the record and are not interchangeable: "
         "**144** regex-detector hits (the basis of the 0.966 figure), **96/154** harm "
         "cells in the per-condition table, and **99** visible trailing artefacts tallied "
         "on the archived pre-repair CSV (accidental 48, intentional 49, attempted 2). "
         "`CONTAMINATION_REPAIR.md` §1.2 keeps them separate deliberately.", "",
         "## 2. Defect 2 — the following scenario lost its name and background", "",
         "The same bug had a second consequence that truncation-based cleaning cannot fix: "
         "the consumed lines never reached the next scenario. 33 of 48 YS2008 scenarios had "
         "no background at all and fell back to a generic id; `YS2008_02` began "
         "mid-narrative with no lab, no switch and no protagonist introduced. This is why "
         "the repair is three rule changes in `code/build_dataset.py` and the CSV is "
         "regenerated — **no row was hand-edited**. Effect: 260 of 298 rows changed, "
         "median word count 89 → 100.5, YS2008 scenarios with a recovered name 15/48 → "
         "48/48.", "",
         "## 3. The label error was real and was not contamination", "",
         "`YS2008-CPR` and its YS2009 reprint list the harmful item first, against the "
         "appendix convention. Taking the convention on faith had inverted `outcome_label` "
         "in all four cells of both scenarios: the `no_harm` cells ended \"The customer "
         "chokes to death at the table\". This was fixed by deriving act polarity from the "
         "text rather than by relabelling rows, so the fix is auditable and applies to any "
         "future scenario with the same inversion.", "",
         "The first audit of this was circular — it compared each final sentence to "
         "`outcome_label` using the same harm-keyword rule that had assigned the label. "
         "`outputs/LABEL_AUDIT_MANUAL.md` is the replacement: by-eye adjudication of all 8 "
         "corrected cells plus a seeded random sample (seed 42, quota per condition), each "
         "row checked against the 2×2 condition definitions on world / belief / action / "
         "outcome. 24 of 144 no-harm rows do contain harm vocabulary, and in every case it "
         "sits in the belief clause — which is the design of the `attempted` condition, not "
         "an error.", "",
         "## 4. A second, independent defect found while auditing the first", "",
         "All 24 YS2009 scenarios are word-for-word reprints of YS2008 scenarios under "
         "different ids. Since every probe used `GroupKFold(groups=scenario_id)`, a "
         "vignette could sit in train while its identical reprint sat in test — train/test "
         "leakage that inflated every reported CV accuracy, and a second plausible "
         "contributor to the 0.99–1.00 outcome-decoding ceiling alongside the "
         "contamination. The fix is a `scenario_group` column merging reprints, collapsing "
         "77 ids to **53 CV groups**; the duplicated rows are retained (both versions were "
         "run behaviourally and the wordings differ slightly) because grouping is "
         "sufficient to stop the leakage. Effective n = 53 is now the resolution of every "
         "interval in the paper (limitation 7).", "",
         "## 5. What replaced the quality gate that failed", "",
         "The gate in place when the bug survived was the clause annotator's coverage rate "
         "(\"94.6% matched, only 4 fallbacks\"). That measures whether a pattern matched, "
         "not whether the text was correct, and it was green the whole time. Reading three "
         "stories would have exposed the defect immediately. Two hard gates replaced it: "
         "`28_validate_master.py`, ten checks including \"the contamination flag no longer "
         "predicts outcome\", exiting non-zero so it can gate an sbatch chain; and a "
         "20-item manual read (`--sample 20 --seed 0`, transcript in "
         "`outputs/MANUAL_SAMPLE_20.txt`). The generalisable lesson is the one in the "
         "repair note: an automated metric being green is not evidence that the data is "
         "right.", "",
         "## 6. What was thrown away", "",
         "Every behavioural and representational result produced before 2026-07-26 used "
         "the contaminated master and was regenerated. Contaminated-era outputs are "
         "retained under `outputs/_contaminated_20260726/` rather than deleted, so any "
         "pre-repair number that appears in an old note can be traced. Closed-model rows "
         "that have not yet been rescored are marked `PENDING RESCORE — contaminated-era` "
         "in their CSVs and their figures are quarantined in "
         "`outputs/figures_final/_pending_rescore/` under a `STALE_` prefix.", ""]
    p = os.path.join(PAPER, "APPENDIX_PROVENANCE.md")
    open(p, "w").write("\n".join(L) + "\n")
    flag = "" if (ok and match) else "  [WARN] pre-repair archive missing or checksum mismatch"
    print(f"  wrote {os.path.relpath(p, tc.ROOT)}{flag}")


def write_reproducibility(head):
    dirty = git("status", "--porcelain")
    L = ["# Reproducibility manifest", "",
         f"Repository at `{head}`; working tree "
         + ("**has uncommitted changes** at write time, so a hash below may not describe "
            "the file on disk" if dirty else "clean at write time") + ".", "",
         "Generated by `code/experiments/57_w8_paper_assembly.py`. The figure hashes are "
         "read from `git log` on each path, not asserted by hand; the bootstrap rows are "
         "checked against the script source and a disagreement is printed as such rather "
         "than corrected silently.", "",
         "## Commit hash per figure", "",
         "`figure commit` is the last commit that touched the PNG; `script commit` is the "
         "last commit that touched the code that produces it. A script commit LATER than "
         "the figure commit means the figure predates its own generator and should be "
         "regenerated before use.", "",
         "| figure | figure commit | producing script | script commit | stale? |",
         "|---|---|---|---|---|"]
    present = sorted(f for f in os.listdir(FIGDIR)) if os.path.isdir(FIGDIR) else []
    for name in [f for f in present if f.endswith(".png")]:
        script = FIG_SCRIPTS.get(name, "— unmapped —")
        fc = last_commit(os.path.join("outputs", "figures_final", name))
        sc = last_commit(script) if script.startswith("code/") else "—"
        fdate = fc.split()[-1] if fc and fc[0] != "—" else ""
        sdate = sc.split()[-1] if sc and sc[0] != "—" else ""
        stale = "**yes — regenerate**" if (fdate and sdate and sdate > fdate) else "no"
        L.append(f"| `{name}` | {fc} | `{script}` | {sc} | {stale} |")
    quar = os.path.join(FIGDIR, "_pending_rescore")
    if os.path.isdir(quar):
        L += ["", "Quarantined (not talk-safe, excluded above): "
              + ", ".join(f"`_pending_rescore/{f}`" for f in sorted(os.listdir(quar))
                          if f.endswith(".png")) + "."]

    L += ["", "## Script per table", "",
          "| table / document | producing script | source data | present |",
          "|---|---|---|---|"]
    for doc, script, src in TABLES:
        here = os.path.exists(os.path.join(tc.ROOT, doc))
        L.append(f"| `{doc}` | `{script}` | `{src}` | "
                 f"{'yes' if here else '**missing**'} |")

    L += ["", "## Seed per bootstrap", "",
          "The resampling unit is the scenario GROUP wherever the estimate is a "
          "cell mean, contrast or coefficient, because 24 of the 53 groups contain a "
          "reprint pair (provenance appendix §4) and resampling items would treat a "
          "vignette and its reprint as independent. Two analyses legitimately resample "
          "something else, named in the table: the item-level dissociation resamples "
          "items, which is the unit its slope is defined over, and the ToM-vs-contrast "
          "correlation resamples models. `verified against source` re-reads the script and "
          "checks that the declaration quoted for this row is still there; a row whose "
          "code has changed reports `declaration changed` rather than continuing to assert "
          "a stale value.", "",
          "| analysis | script | resampling unit | B | seed | verified against source |",
          "|---|---|---|---:|---:|---|"]
    for name, script, unit, B, seed, bdecl, sdecl in BOOTSTRAPS:
        vb, vs = verify_decl(script, bdecl), verify_decl(script, sdecl)
        v = "ok" if {vb, vs} <= {"ok", "n/a"} else f"B: {vb}; seed: {vs}"
        L.append(f"| {name} | `{script}` | {unit} | "
                 f"{B if B is not None else '—'} | {seed} | {v} |")

    L += ["", "## Environment and data version", "",
          f"- Stimulus master: `dataset/master/moral_2x2_master.csv`, "
          f"{last_commit('dataset/master/moral_2x2_master.csv')}, 298 rows / 53 CV groups, "
          "post-repair (see provenance appendix).",
          f"- Clause offsets: `dataset/master/clause_offsets.csv`, "
          f"{last_commit('dataset/master/clause_offsets.csv')}.",
          f"- Bruneau appendix stimuli: `dataset/bruneau/bruneau_stimuli.csv`, "
          f"{last_commit('dataset/bruneau/bruneau_stimuli.csv')}, 144 items / 72 pairs.",
          "- Behavioural scoring: logprob expected value over rating digits, one forward "
          "pass per prompt, deterministic. Open-weight models only for anything "
          "representational.",
          "- Pre-registrations, written before their runs and never overwritten: "
          "`outputs/experiments/W3_PRESPEC.md`, `outputs/experiments/W4_PRESPEC.md`.", ""]
    p = os.path.join(PAPER, "REPRODUCIBILITY.md")
    open(p, "w").write("\n".join(L) + "\n")
    print(f"  wrote {os.path.relpath(p, tc.ROOT)} "
          f"({len([f for f in present if f.endswith('.png')])} figures, "
          f"{len(TABLES)} tables, {len(BOOTSTRAPS)} bootstraps)")


def write_index(head):
    L = ["# W8 paper assembly", "",
         f"Generated by `code/experiments/57_w8_paper_assembly.py` at `{head}`. "
         "Regenerate after any figure or result changes; nothing here is hand-maintained "
         "except the curated prose in the limitations and provenance documents.", "",
         "| document | what it is |", "|---|---|",
         "| [`LIMITATIONS.md`](LIMITATIONS.md) | Nine limitations, each stating what it "
         "forbids claiming and what would lift it |",
         "| [`APPENDIX_PROVENANCE.md`](APPENDIX_PROVENANCE.md) | The stimulus integrity "
         "audit: two defects, the repair, the archived pre-repair artefact with a verified "
         "checksum |",
         "| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Commit hash per figure, script "
         "per table, seed per bootstrap, each checked against the repository |", "",
         "Related, not generated here: `outputs/MENTOR_PACKET.md` (one-page summary), "
         "`outputs/figures_final/FIGURE_MANIFEST.md` (per-figure source and status), "
         "`outputs/RUN_STATUS.md` (what is running).", ""]
    p = os.path.join(PAPER, "README.md")
    open(p, "w").write("\n".join(L) + "\n")
    print(f"  wrote {os.path.relpath(p, tc.ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["limitations", "provenance", "repro", "index"])
    a = ap.parse_args()
    os.makedirs(PAPER, exist_ok=True)
    head = git("rev-parse", "--short", "HEAD") or "unknown"
    if a.only in (None, "limitations"):
        write_limitations(head)
    if a.only in (None, "provenance"):
        write_provenance(head)
    if a.only in (None, "repro"):
        write_reproducibility(head)
    if a.only in (None, "index"):
        write_index(head)


if __name__ == "__main__":
    main()
