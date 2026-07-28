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
take outputs/probe/gap_over_surface_dissociation_span_matched.png
take outputs/probe/gap_over_surface_by_pooling.csv
take outputs/probe/gap_over_surface_span_matched.csv
take outputs/probe/gap_over_surface_within_model_paired.csv
take outputs/probe/surface_baseline.csv
take outputs/probe/layerwise_curves.png
take outputs/rsa/model_similarity_heatmap_rsa_spearman.png
take outputs/rsa/model_similarity_heatmap_cka_linear.png
take outputs/link/rep_vs_behavior.png
take outputs/link/representation_vs_behavior.csv
take outputs/link/ITEM_LEVEL_DISSOCIATION.md
take outputs/link/item_level_dissociation.csv
take outputs/link/item_level_dissociation.png

# --- behavioral ---
take outputs/experiments/checkpoint_dissection.png
take outputs/experiments/checkpoint_dissection.csv
take outputs/experiments/CHECKPOINT_DISSECTION.md
take outputs/analysis/prompt_factorial_sign_stability.csv
take outputs/analysis/prompt_factorial_variance.csv
take outputs/analysis/prompt_factorial_report.md
take outputs/analysis/C2_SOURCE_SPLIT_BELIEF_LAST.md
take outputs/stats/contrast_by_model.csv
take outputs/stats/mixed_effects_2x2.csv
take outputs/stats/mixed_effects_interaction.png
take outputs/stats/MIXED_EFFECTS_2x2.md
take outputs/stats/FLOOR_DERIVATION.md
take outputs/behavior/intent_reliance_summary.csv
take outputs/tom_benchmarks/TOM_VS_CONTRAST.md
take outputs/tom_benchmarks/tom_vs_contrast.csv
take outputs/tom_benchmarks/tom_vs_contrast.png
take outputs/tom_benchmarks/tom_accuracy_by_model.csv
take outputs/tom_benchmarks/TOMI_SCORING_AUDIT.md
take outputs/tom_benchmarks/CLOSED_TOM.md
take outputs/tom_benchmarks/tom_accuracy_by_model_generative.csv
take outputs/tom_benchmarks/tom_scoring_agreement.csv
take outputs/analysis/ROSTER_70B_FEASIBILITY.md

# --- reviewer checks ---
take outputs/analysis/REVIEW_CHECKS.md
take outputs/analysis/check_c1_within_model_variance.csv
take outputs/analysis/check_c5_engagement_floor.csv
take outputs/analysis/check_c6_link_all_models.csv
take outputs/analysis/check_c7_flip_rate_conditioned.csv
take outputs/analysis/SIGN_FLIP_RECONCILIATION.md
take outputs/analysis/ROSTER_70B_FEASIBILITY.md

# --- human reference ---
take dataset/human_reference/cushman_digitized_overlay.png

echo "updated_figures: $(find "$DEST" -maxdepth 1 -type f | wc -l) files"
echo "  png: $(find "$DEST" -maxdepth 1 -name '*.png' | wc -l)"
