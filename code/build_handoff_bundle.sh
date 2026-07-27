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
take outputs/experiments/mini_dissection.csv

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
take outputs/probe/layerwise_curves.png
take_glob "outputs/rsa/model_similarity_heatmap_*.png"
take outputs/link/rep_vs_behavior.png
take dataset/human_reference/cushman_digitized_overlay.png

echo "bundle files: $(find "$DEST" -maxdepth 1 -type f ! -name '.missing.txt' | wc -l)"
if [ -s "$MISSING" ]; then
  echo "MISSING (expected but absent):"
  sed 's/^/  /' "$MISSING"
fi
