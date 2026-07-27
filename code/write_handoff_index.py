#!/usr/bin/env python3
"""Emit outputs/_handoff/INDEX.md -- one row per bundled file.

Descriptions only: what the file holds, what wrote it, which commit that script is at,
and which stimulus-master state it reflects. No findings, no interpretation.

Provenance rules are matched in order, first match wins, so put specific patterns above
general ones.
"""
import os
import re
import subprocess
import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "outputs", "_handoff")

# --- stimulus-master states -------------------------------------------------
CUR = "current (repaired master, 2026-07-27 10:11)"
MIX = "MIXED — open-weight rows current; closed-API rows STALE (contaminated-era, pre-2026-07-26, never rescored)"
STALE_CLOUD = "STALE — contaminated-era (scored 2026-07-21, before the 07-26 repair)"
ARCHIVE = "pre-repair archive (intentionally the OLD state)"
NA = "n/a — not derived from the stimulus master"
DOC = "documentation of the repair itself"

_commit_cache = {}


def commit_of(script_relpath):
    """Last commit touching the producing script (short hash)."""
    if script_relpath in _commit_cache:
        return _commit_cache[script_relpath]
    out = ""
    if script_relpath:
        p = os.path.join(ROOT, script_relpath)
        if os.path.exists(p):
            out = subprocess.run(
                ["git", "log", "-1", "--format=%h", "--", script_relpath],
                cwd=ROOT, capture_output=True, text=True).stdout.strip()
    _commit_cache[script_relpath] = out or "—"
    return _commit_cache[script_relpath]


# (regex, description, producing script, dataset state)
RULES = [
    # ---- behavioral: per-model ----
    (r"^intent_reliance_summary\.csv$",
     "One row per model: intent-reliance index, its SD across templates, mean intent/outcome "
     "betas, template counts, and a degenerate flag.",
     "code/experiments/23_build_intent_reliance_summary.py", CUR),
    (r"^intent_reliance_.*\.csv$",
     "Per-template intent/outcome regression coefficients and the derived reliance index for one model.",
     "code/03_behavioral.py", CUR),
    (r"^prompt_invariance_.*\.csv$",
     "Per-template attempted−accidental contrast for one model, used for across-prompt spread.",
     "code/03_behavioral.py", CUR),
    (r"^item_means_.*\.csv$",
     "Per-item mean rating for one model, by template and 2×2 cell; the input every behavioral statistic is built from.",
     "code/03_behavioral.py", CUR),

    # ---- behavioral: aggregate ----
    (r"^prompt_factorial_sign_stability\.csv$",
     "Per-model contrast under each of the 6 factorial templates (2 wordings × 3 constructs), "
     "with sign-stability, pooled-inclusion, and verdict columns.",
     "code/experiments/33_prompt_factorial_analysis.py", CUR),
    (r"^prompt_factorial_variance\.csv$",
     "Type-II ANOVA and mixed-model output for contrast ~ wording × construct; primary row is all "
     "models, sensitivity row is sign-stable models only.",
     "code/experiments/33_prompt_factorial_analysis.py", CUR),
    (r"^prompt_factorial_report\.md$",
     "Narrative form of the two files above, including the pre-registered inclusion floor and flip-rate reporting rule.",
     "code/experiments/33_prompt_factorial_analysis.py", CUR),
    (r"^prompt_invariance_decomposition\.csv$",
     "Earlier variance decomposition across the 3-template set that preceded the factorial design.",
     "code/14_prompt_invariance_decomposition.py", STALE_CLOUD),
    (r"^interaction_regression\.csv$",
     "Intent × outcome interaction regression per model.", "code/11_interaction_regression.py", CUR),
    (r"^contrast_by_model\.csv$",
     "Master per-model statistics table: contrast with CI, intent-reliance, template SD/range, "
     "sign-flip flag, nearest human band, degeneracy.",
     "code/06_stats.py", CUR),
    (r"^prompt_invariance_contrast\.csv$",
     "Model × template matrix of contrasts with SD, range, and sign-flip count.",
     "code/06_stats.py", CUR),
    (r"^pairwise_model_diffs\.csv$",
     "All pairwise model contrast differences with CIs and a distinguishability flag.",
     "code/06_stats.py", CUR),
    (r"^checkpoint_dissection\.csv$",
     "Contrast and intent-reliance at each OLMo-2-7B training checkpoint.",
     "code/experiments/16_checkpoint_dissection.py", CUR),
    (r"^checkpoint_dissection_writeup\.md$",
     "Narrative accompanying the checkpoint table.",
     "code/experiments/16_checkpoint_dissection.py", CUR),
    (r"^mini_dissection\.csv$",
     "Earlier reduced-scope checkpoint probe retained for comparison.",
     "code/experiments/18_mini_dissection.py", STALE_CLOUD),

    # ---- ladders ----
    (r"^master_all_models_.*_openonly\.csv$",
     "Ladder table for one human anchor, open-weight models only: contrast, CI, nearest human band, "
     "at/below-youngest flag.", "code/29_dual_human_ladders.py", CUR),
    (r"^master_all_models_.*_all\.csv$",
     "Same ladder table as the matching _openonly file but with closed-API models appended.",
     "code/29_dual_human_ladders.py", MIX),
    (r"^master_all_models\.csv$",
     "Combined ladder table across all studies for the default anchor; closed-API rows are NOT "
     "separated out in this file.", "code/10_master_figure.py", MIX),
    (r"^human_anchor_comparison\.csv$",
     "Side-by-side nearest-band and below-youngest verdict for every model under all three anchors.",
     "code/29_dual_human_ladders.py", CUR),
    (r"^human_anchor_comparison\.NOTES\.md$",
     "Auto-generated notes for the file above: per-anchor counts, scope statement, and the "
     "robustness/theory paragraphs, all computed from the live numbers.",
     "code/29_dual_human_ladders.py", CUR),

    # ---- representation ----
    (r"^.*_probe_mean\.csv$",
     "Layer-wise probe accuracy for intent/outcome, mean-pooled over tokens, for one model.",
     "code/02_probe.py", CUR),
    (r"^.*_probe_belief_last\.csv$",
     "Layer-wise probe accuracy at the last token of the belief clause, for one model.",
     "code/02_probe.py", CUR),
    (r"^.*_probe_action_last\.csv$",
     "Layer-wise probe accuracy at the last token of the action clause, for one model.",
     "code/02_probe.py", CUR),
    (r"^.*_probe\.csv$",
     "Layer-wise probe accuracy at the final token (default pooling), for one model.",
     "code/02_probe.py", CUR),
    (r"^.*_permnull.*\.csv$",
     "Permutation null distribution (N=1000) for probe accuracy at the peak layer and layer 0, for one model/pooling.",
     "code/02_probe.py", CUR),
    (r"^.*_withincell\.csv$",
     "Probe accuracy trained and tested within a single 2×2 cell, holding surface content fixed, for one model.",
     "code/experiments/22_within_cell_probes.py", CUR),
    (r"^surface_baseline\.csv$",
     "TF-IDF bag-of-words baseline accuracy on the same labels the neural probes predict.",
     "code/experiments/21_surface_baseline.py", CUR),
    (r"^layer0_diagnostic\.csv$",
     "Layer-0 (embedding) read-off accuracy, the floor for how much is lexically present before any computation.",
     "code/experiments/20_layer0_diagnostic.py", CUR),
    (r"^layer0_pooling_check\.csv$",
     "Layer-0 read-off repeated under each pooling variant.",
     "code/experiments/20_layer0_diagnostic.py", CUR),
    (r"^gap_over_surface_by_pooling\.csv$",
     "Peak probe accuracy minus the surface baseline, per model and pooling variant. "
     "NOTE: produced by an inline command during the confound chain, not by a checked-in script.",
     "(ad-hoc inline, not checked in)", CUR),

    # ---- RSA ----
    (r"^rsa_.*_model_similarity\.csv$",
     "Pairwise similarity between model RDMs (Spearman and linear CKA) for one pooling variant.",
     "code/experiments/24_rsa_cka.py", CUR),
    (r"^rsa_.*_hypothesis_rdm\.csv$",
     "Correlation of each model RDM against the hypothesis RDMs (intent-structured, outcome-structured), one pooling.",
     "code/experiments/24_rsa_cka.py", CUR),
    (r"^rsa_.*_rsa_permutation_null\.csv$",
     "Permutation null for the hypothesis-RDM correlations, one pooling.", "code/experiments/24_rsa_cka.py", CUR),
    (r"^rsa_.*_convergence_pairs\.csv$",
     "Per model-pair representational similarity alongside behavioral similarity, one pooling.",
     "code/experiments/24_rsa_cka.py", CUR),
    (r"^rsa_.*_convergence_test\.json$",
     "Test statistic for the representation-vs-behavior convergence over model pairs, one pooling.",
     "code/experiments/24_rsa_cka.py", CUR),
    (r"^rsa_.*_base_vs_instruct_geometry\.csv$",
     "Representational geometry change from base to instruct within matched model pairs, one pooling.",
     "code/experiments/24_rsa_cka.py", CUR),
    (r"^representation_vs_behavior\.csv$",
     "Join of peak intent-decoding accuracy against the behavioral intent-reliance index, one row per model that has both.",
     "code/04_link_analysis.py", CUR),

    # ---- human reference ----
    (r"^human_reference\.csv$",
     "Text-reported child/adult contrast bands as transcribed from the paper text.",
     "hand-entered from Cushman et al. (2013)", NA),
    (r"^human_reference_digitized\.csv$",
     "Child/adult bands from digitizing the Naughty presented-first panel.",
     "hand-digitized from the published figure", NA),
    (r"^human_reference_punish\.csv$",
     "Child/adult bands from digitizing the Punish presented-first panel.",
     "hand-digitized from the published figure", NA),
    (r"^cushman_child_bands_PROPOSED\.csv$",
     "Intermediate age-pair cell means for the Naughty series feeding human_reference_digitized.csv.",
     "hand-digitized", NA),
    (r"^cushman_child_bands_PUNISH\.csv$",
     "Intermediate age-pair cell means for the Punish series feeding human_reference_punish.csv.",
     "hand-digitized", NA),
    (r"^cushman_naughty_digitized\.csv$",
     "Raw digitized point reads from the Naughty panel before age-pair averaging.",
     "hand-digitized", NA),
    (r"^methods_child_measure\.md$",
     "Which child measure is used and why, the pre-specification date, and the secondary Punish ladder.",
     "hand-written", NA),
    (r"^QUARANTINE_cushman_calibrated\.md$",
     "Record of a superseded calibrated anchor and why it is quarantined.", "hand-written", NA),
    (r"^cushman_digitized_overlay\.png$",
     "Digitized points overlaid on the source figure, for checking the digitization.",
     "code/digitize_cushman_calibrated.py", NA),

    # ---- audit trail ----
    (r"^CONTAMINATION_REPAIR\.md$",
     "What the story-boundary contamination and CV leakage were, how they were repaired, and the verification performed.",
     "hand-written", DOC),
    (r"^LABEL_AUDIT_MANUAL\.md$",
     "Manual re-read of sampled items against their intent/outcome labels.", "hand-written", DOC),
    (r"^POLARITY_AUDIT\.md$",
     "Check that rating polarity is consistent across templates and scales.",
     "code/experiments/30_polarity_audit.py", DOC),
    (r"^SCALE_REPLICATION\.md$",
     "Agreement between the 1–7 and alternate-scale elicitations (correlation and Bland–Altman).",
     "code/experiments/31_scale_replication.py", DOC),
    (r"^API_COST_ESTIMATE\.md$",
     "Projected token cost of rescoring the closed-API models.", "hand-written", NA),
    (r"^OVERNIGHT_REPORT_20260726\.md$",
     "Status readout written after the 07-26 overnight batch.", "hand-written", "as of 2026-07-26"),
    (r"^MORNING_B3_B9_C3_C5\.md$",
     "Status readout written on the morning of 07-27 covering tasks B3, B9, C3, C5.",
     "hand-written", "as of 2026-07-27 morning"),
    (r"^MASTER_SUMMARY\.md$",
     "Rolling project summary. Predates every run in this bundle.", "hand-written",
     "STALE — written before the 07-26 repair"),
    (r"^moral_2x2_master\.csv$",
     "The repaired stimulus master: 2×2 intent×outcome items with scenario groups and clause annotations.",
     "code/experiments/27_clean_stimuli.py", CUR),
    (r"^moral_2x2_master_CONTAMINATED_20260619\.csv$",
     "The pre-repair stimulus master, kept so the repair diff is reproducible.",
     "code/build_dataset.py", ARCHIVE),
    (r"^prerepair_backup_README\.md$",
     "Explains what the archived contaminated master is and why it is retained.", "hand-written", ARCHIVE),
    (r"^clause_offsets\.csv$",
     "Character offsets of the belief/action/outcome clauses in each item, used for clause-position pooling.",
     "code/experiments/25_annotate_clauses.py", CUR),
    (r"^scoring_parity\.csv$",
     "Agreement between logprob scoring and free-generation scoring on the same items.",
     "code/analysis/15_scoring_parity.py", STALE_CLOUD),
    (r"^scoring_parity_writeup\.md$",
     "Narrative for the scoring-parity comparison.", "code/analysis/15_scoring_parity.py", STALE_CLOUD),

    # ---- figures ----
    (r"^master_developmental_ladder_.*_openonly\.png$",
     "Ladder figure for one anchor, open-weight models only.", "code/29_dual_human_ladders.py", CUR),
    (r"^master_developmental_ladder_.*_all\.png$",
     "Ladder figure for one anchor including closed-API models, which are marked on the figure.",
     "code/29_dual_human_ladders.py", MIX),
    (r"^master_developmental_ladder\.png$",
     "Combined ladder for the default anchor; closed-API models are NOT marked in this version.",
     "code/10_master_figure.py", MIX),
    (r"^checkpoint_dissection\.png$",
     "Contrast and intent-reliance across OLMo-2-7B training checkpoints.",
     "code/experiments/16_checkpoint_dissection.py", CUR),
    (r"^gap_over_surface_dissociation\.png$",
     "Peak probe accuracy against the TF-IDF surface baseline, by pooling.",
     "code/experiments/33_gap_dissociation_figure.py", CUR),
    (r"^layerwise_curves\.png$",
     "Probe accuracy as a function of layer, per model.", "code/02_probe.py", CUR),
    (r"^model_similarity_heatmap_.*\.png$",
     "Heatmap of between-model representational similarity (one panel per similarity metric).",
     "code/experiments/24_rsa_cka.py", CUR),
    (r"^rep_vs_behavior\.png$",
     "Scatter of peak intent-decoding accuracy against behavioral intent-reliance.",
     "code/04_link_analysis.py", CUR),
]


def describe(name):
    for pat, desc, script, state in RULES:
        if re.match(pat, name):
            return desc, script, state
    return "—", "—", "—"


def main():
    files = sorted(f for f in os.listdir(DEST)
                   if os.path.isfile(os.path.join(DEST, f)) and not f.startswith("."))
    files = [f for f in files if f not in ("INDEX.md", "RUN_STATUS.md")]

    head = subprocess.run(["git", "log", "-1", "--format=%h"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# Handoff bundle — file index",
        "",
        f"Assembled {now}. Repository HEAD at assembly: `{head}`.",
        "",
        "Flat bundle of raw artifacts. Descriptions state what each file contains and how it was "
        "produced; they do not interpret the values. Activation tensors (`*.npz`, `*.npy`) are "
        "excluded by design.",
        "",
        "**Commit** is the last commit that touched the producing script, not necessarily the commit "
        "the file was generated under; every file here was regenerated at or after that commit.",
        "",
        "## Stimulus-master states used below",
        "",
        "| State | Meaning |",
        "| --- | --- |",
        f"| current | {CUR}. Reflects the story-boundary/CV-leakage repair (2026-07-26) and the YS2011 title strip (2026-07-27 10:11). |",
        "| MIXED | Open-weight rows are current; closed-API rows were scored 2026-07-21 and have **not** been rescored since the repair. |",
        "| STALE | Produced before the repair and not regenerated since. |",
        "| pre-repair archive | Deliberately the old state, kept for diffing. |",
        "| n/a | Not derived from the stimulus master (human reference data, cost notes). |",
        "",
        "## Files",
        "",
        "| File | Contents | Produced by | Commit | Stimulus-master state |",
        "| --- | --- | --- | --- | --- |",
    ]

    counts = {}
    for f in files:
        desc, script, state = describe(f)
        c = commit_of(script) if script.startswith("code/") and script.endswith(".py") else "—"
        short = state.split(" —")[0].split(" (")[0]
        counts[short] = counts.get(short, 0) + 1
        lines.append(f"| `{f}` | {desc} | `{script}` | `{c}` | {state} |")

    lines += ["", f"**Total files: {len(files)}**", "",
              "Counts by stimulus-master state: "
              + ", ".join(f"{k} = {v}" for k, v in sorted(counts.items())) + ".", ""]

    with open(os.path.join(DEST, "INDEX.md"), "w") as fh:
        fh.write("\n".join(lines))
    print(f"wrote INDEX.md for {len(files)} files")
    unmatched = [f for f in files if describe(f)[0] == "—"]
    if unmatched:
        print("NO RULE MATCHED (fix before shipping):")
        for u in unmatched:
            print("   ", u)


if __name__ == "__main__":
    main()
