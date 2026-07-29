#!/usr/bin/env bash
# build_handoff_bundle.sh -- assemble outputs/_handoff/ as a FLAT bundle of raw artifacts.
#
# Re-runnable: wipes and rebuilds the bundle each time, so it can be run again once a
# mid-flight job lands. Copies only; never regenerates. Activation .npz files are excluded
# by design. RSA outputs exist once per pooling under identically-named files, so those are
# prefixed with their pooling on the way in to keep the bundle flat without collisions.
set -uo pipefail
cd "$(dirname "$0")/.."
DEST="outputs/_handoff"
rm -rf "$DEST"; mkdir -p "$DEST"

MISSING="$DEST/.missing.txt"; : > "$MISSING"

take() {  # take <src> [dest_name]
  local src="$1" name="${2:-}"
  [ -z "$name" ] && name="$(basename "$src")"
  if [ -e "$src" ]; then
    cp -p "$src" "$DEST/$name"
  else
    echo "$src" >> "$MISSING"
  fi
}

take_glob() {  # take_glob <glob> [prefix]
  local pat="$1" prefix="${2:-}" n=0
  for f in $pat; do
    [ -e "$f" ] || continue
    cp -p "$f" "$DEST/${prefix}$(basename "$f")"
    n=$((n+1))
  done
  [ "$n" -eq 0 ] && echo "$pat (no matches)" >> "$MISSING"
  return 0
}

# ---------------- Behavioral ----------------
# Only the scope-split ladder tables travel; the anchor-only files they superseded are
# same-data intermediates from the pre-split run and would just be ambiguous in a flat bundle.
take outputs/master_all_models.csv
take_glob "outputs/master_all_models_*_openonly.csv"
take_glob "outputs/master_all_models_*_all.csv"
take_glob "outputs/stats/*.csv"
take_glob "outputs/behavior/intent_reliance_*.csv"
take_glob "outputs/behavior/prompt_invariance_*.csv"
take_glob "outputs/behavior/item_means_*.csv"
take outputs/behavior/intent_reliance_summary.csv
take outputs/analysis/prompt_factorial_sign_stability.csv
take outputs/analysis/prompt_factorial_variance.csv
take outputs/analysis/prompt_factorial_report.md
take outputs/analysis/prompt_invariance_decomposition.csv
take outputs/analysis/interaction_regression.csv
take outputs/experiments/checkpoint_dissection.csv
take outputs/experiments/checkpoint_dissection_writeup.md
take outputs/experiments/CHECKPOINT_DISSECTION.md
take outputs/experiments/CHECKPOINT_STAGE_SHARES.md
take outputs/experiments/checkpoint_stage_shares.csv

# Hand-maintained status table. It used to live inside $DEST, where the wipe at the top
# deleted it on every rebuild; its source of truth is now outside the bundle.
take outputs/RUN_STATUS.md

# W3 causal steering (+ M1/M2/M3 + prose)
take outputs/experiments/W3_PRESPEC.md
take outputs/experiments/W3_STEERING_SUMMARY.md
take outputs/experiments/W3_PROSE_RATING.md
for m in OLMo-2-1124-7B-Instruct Qwen2.5-7B-Instruct; do
  take "outputs/experiments/W3_STEERING_${m}.md"
  take "outputs/experiments/W3_LAYERSWEEP_${m}.md"
  take "outputs/experiments/w3_steering_${m}.csv"
  take "outputs/experiments/w3_steering_${m}.png"
  take "outputs/experiments/w3_steering_directions_${m}.csv"
  take "outputs/experiments/w3_calibration_${m}.csv"
  take "outputs/experiments/w3_generations_${m}.txt"
  take "outputs/experiments/w3_manipulation_${m}.csv"
  take "outputs/experiments/w3_manipulation_${m}.png"
  take "outputs/experiments/w3_layersweep_${m}.csv"
  take "outputs/experiments/w3_layersweep_${m}.png"
  take "outputs/experiments/w3_prose_items_${m}.csv"
done
take outputs/experiments/w3_prose_rating.png
take outputs/experiments/CLOSED_MODEL_SELECTION.md
take outputs/experiments/CLOSED_MODEL_CATALOG.md

# W4 prompt curriculum (companion to W3: reachable from the input?)
take outputs/experiments/W4_PRESPEC.md
take outputs/experiments/W4_PROMPT_LEVELS.md
take outputs/experiments/W4_CURRICULUM.md
take outputs/experiments/w4_prompt_curriculum.csv
take outputs/experiments/w4_curriculum_cells.csv

# W7 Bruneau selectivity (appendix)
take outputs/experiments/W7_PARSE_REPORT.md
take outputs/experiments/W7_BRUNEAU.md
take outputs/experiments/w7_bruneau_probes.csv
take outputs/experiments/w7_bruneau_transfer.csv
take outputs/experiments/w7_bruneau_interaction.csv
take dataset/bruneau/bruneau_stimuli.csv

# W8 paper assembly
take outputs/paper/README.md
take outputs/paper/LIMITATIONS.md
take outputs/paper/APPENDIX_PROVENANCE.md
take outputs/paper/REPRODUCIBILITY.md

take outputs/MENTOR_PACKET.md
take outputs/figures_final/FIGURE_MANIFEST.md
# Only figures_final/ proper. The _pending_rescore/ subdirectory is deliberately not
# globbed: those figures carry contaminated-era closed rows and must not reach a handoff.
take_glob "outputs/figures_final/*.png" "fig_"
take outputs/experiments/mini_dissection.csv
take outputs/stats/mixed_effects_2x2.csv
take outputs/stats/MIXED_EFFECTS_2x2.md
take outputs/stats/FLOOR_DERIVATION.md
take outputs/stats/floor_derivation.csv
take outputs/tom_benchmarks/tom_accuracy_by_model.csv
take outputs/tom_benchmarks/tom_accuracy_by_model_generative.csv
take outputs/tom_benchmarks/tom_scoring_agreement.csv
take outputs/tom_benchmarks/tom_vs_contrast.csv
take outputs/tom_benchmarks/TOM_VS_CONTRAST.md
take outputs/tom_benchmarks/TOMI_SCORING_AUDIT.md
take outputs/tom_benchmarks/CLOSED_TOM.md
take outputs/link/item_level_dissociation.csv
take outputs/link/item_level_groups.csv
take outputs/link/ITEM_LEVEL_DISSOCIATION.md
take outputs/analysis/C2_SOURCE_SPLIT_BELIEF_LAST.md
take outputs/analysis/SIGN_FLIP_RECONCILIATION.md
take outputs/analysis/ROSTER_70B_FEASIBILITY.md
take outputs/analysis/REVIEW_CHECKS.md
take_glob "outputs/analysis/check_c*.csv"

# ---------------- Representation ----------------
take_glob "outputs/probe/*_probe.csv"
take_glob "outputs/probe/*_probe_mean.csv"
take_glob "outputs/probe/*_probe_belief_last.csv"
take_glob "outputs/probe/*_probe_action_last.csv"
take_glob "outputs/probe/*_permnull*.csv"
take_glob "outputs/probe/*_withincell.csv"
take outputs/probe/surface_baseline.csv
take outputs/probe/layer0_diagnostic.csv
take outputs/probe/layer0_pooling_check.csv
take outputs/probe/gap_over_surface_by_pooling.csv
take outputs/probe/gap_over_surface_span_matched.csv
take outputs/probe/gap_over_surface_within_model_paired.csv
take_glob "outputs/probe/*_probe_belief_last_src*.csv"
take_glob "outputs/probe/*_probe_action_last_src*.csv"

# RSA: outputs/rsa is the mean pooling; the others are per-pooling siblings.
take_glob "outputs/rsa/*.csv"  "rsa_mean_"
take_glob "outputs/rsa/*.json" "rsa_mean_"
for P in last belief_last action_last; do
  take_glob "outputs/rsa_${P}/*.csv"  "rsa_${P}_"
  take_glob "outputs/rsa_${P}/*.json" "rsa_${P}_"
done

take outputs/link/representation_vs_behavior.csv

# ---------------- Human anchors ----------------
take outputs/human_anchor_comparison.csv
take outputs/human_anchor_comparison.NOTES.md
take dataset/human_reference/human_reference.csv
take dataset/human_reference/human_reference_digitized.csv
take dataset/human_reference/human_reference_punish.csv
take dataset/human_reference/cushman_child_bands_PROPOSED.csv
take dataset/human_reference/cushman_child_bands_PUNISH.csv
take dataset/human_reference/cushman_naughty_digitized.csv
take dataset/human_reference/methods_child_measure.md
take dataset/human_reference/QUARANTINE_cushman_calibrated.md

# ---------------- Audit trail ----------------
take dataset/master/CONTAMINATION_REPAIR.md
take outputs/LABEL_AUDIT_MANUAL.md
take outputs/POLARITY_AUDIT.md
take outputs/SCALE_REPLICATION.md
take outputs/API_COST_ESTIMATE.md
take outputs/OVERNIGHT_REPORT_20260726.md
take outputs/MORNING_B3_B9_C3_C5.md
take outputs/MASTER_SUMMARY.md
take dataset/master/moral_2x2_master.csv
take dataset/master/clause_offsets.csv
take dataset/master/_prerepair_backup/moral_2x2_master_CONTAMINATED_20260619.csv
take dataset/master/_prerepair_backup/README.md prerepair_backup_README.md
take outputs/analysis/scoring_parity.csv
take outputs/analysis/scoring_parity_writeup.md

# ---------------- Figures (copied as-is; regenerate upstream before running) ----------
take outputs/master_developmental_ladder.png
take_glob "outputs/master_developmental_ladder_*_openonly.png"
take_glob "outputs/master_developmental_ladder_*_all.png"
take outputs/experiments/checkpoint_dissection.png
take outputs/probe/gap_over_surface_dissociation.png
take outputs/probe/gap_over_surface_dissociation_span_matched.png
take outputs/probe/layerwise_curves.png
take_glob "outputs/rsa/model_similarity_heatmap_*.png"
take outputs/link/rep_vs_behavior.png
take outputs/link/item_level_dissociation.png
take outputs/stats/mixed_effects_interaction.png
take outputs/tom_benchmarks/tom_vs_contrast.png
take outputs/updated_figures/human_only_developmental_ladder.png
take outputs/updated_figures/HUMAN_ANCHOR_CHOICES.md
take dataset/human_reference/cushman_digitized_overlay.png

echo "bundle files: $(find "$DEST" -maxdepth 1 -type f ! -name '.missing.txt' | wc -l)"
if [ -s "$MISSING" ]; then
  echo "MISSING (expected but absent):"
  sed 's/^/  /' "$MISSING"
fi
