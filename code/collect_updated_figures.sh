#!/usr/bin/env bash
# Gather every current figure into outputs/updated_figures/ so there is one place to look.
# Re-runnable. Companion tables and notes travel with the figures they belong to.
set -uo pipefail
cd "$(dirname "$0")/.."
DEST="outputs/updated_figures"
mkdir -p "$DEST"

# Keep files this script generates elsewhere (human-only ladder, its notes).
find "$DEST" -maxdepth 1 -type f \
  ! -name 'human_only_developmental_ladder.*' \
  ! -name 'HUMAN_ANCHOR_CHOICES.md' -delete 2>/dev/null

take() { [ -e "$1" ] && cp -p "$1" "$DEST/${2:-$(basename "$1")}" || echo "  MISSING: $1"; }

# --- developmental ladders (scope-split; degenerate models omitted from plots) ---
for a in text_reported digitized punish; do
  for s in openonly all; do
    take "outputs/master_developmental_ladder_${a}_${s}.png"
    take "outputs/master_all_models_${a}_${s}.csv"
  done
done
take outputs/human_anchor_comparison.csv
take outputs/human_anchor_comparison.NOTES.md

# --- representation ---
take outputs/probe/gap_over_surface_dissociation.png
take outputs/probe/gap_over_surface_by_pooling.csv
take outputs/probe/gap_over_surface_within_model_paired.csv
take outputs/probe/layerwise_curves.png
take outputs/rsa/model_similarity_heatmap_rsa_spearman.png
take outputs/rsa/model_similarity_heatmap_cka_linear.png
take outputs/link/rep_vs_behavior.png
take outputs/link/representation_vs_behavior.csv

# --- behavioral ---
take outputs/experiments/checkpoint_dissection.png
take outputs/experiments/checkpoint_dissection.csv
take outputs/analysis/prompt_factorial_sign_stability.csv
take outputs/analysis/prompt_factorial_variance.csv
take outputs/analysis/prompt_factorial_report.md
take outputs/stats/contrast_by_model.csv
take outputs/behavior/intent_reliance_summary.csv

# --- reviewer checks ---
take outputs/analysis/REVIEW_CHECKS.md
take outputs/analysis/check_c1_within_model_variance.csv
take outputs/analysis/check_c5_engagement_floor.csv
take outputs/analysis/check_c6_link_all_models.csv
take outputs/analysis/check_c7_flip_rate_conditioned.csv

# --- human reference ---
take dataset/human_reference/cushman_digitized_overlay.png

echo "updated_figures: $(find "$DEST" -maxdepth 1 -type f | wc -l) files"
echo "  png: $(find "$DEST" -maxdepth 1 -name '*.png' | wc -l)"
