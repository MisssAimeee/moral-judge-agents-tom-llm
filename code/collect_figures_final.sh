#!/usr/bin/env bash
# Canonical figure location: outputs/figures_final/
# Prior dirs (updated_figures/, loose root PNGs) are deprecated — see DEPRECATED.md there.
set -uo pipefail
cd "$(dirname "$0")/.."
DEST="outputs/figures_final"
# Anything whose closed-model rows are still contaminated-era (v1) lives here, prefixed
# STALE_, so figures_final/ itself contains only figures safe to put in a talk.
QUAR="$DEST/_pending_rescore"
mkdir -p "$DEST" "$QUAR"

take() {
  local src="$1"
  local name="${2:-$(basename "$1")}"
  if [ -e "$src" ]; then
    cp -p "$src" "$DEST/$name"
  else
    echo "  PENDING: $src"
  fi
}

take_stale() {  # quarantined: contaminated-era closed rows
  local src="$1"
  local name="STALE_${2:-$(basename "$1")}"
  if [ -e "$src" ]; then
    cp -p "$src" "$QUAR/$name"
  else
    echo "  PENDING: $src"
  fi
}

# Developmental ladders. The *_openonly variants are clean (open-weight only). The *_all
# variants mix in closed models scored on v1 (gpt-4o, claude-haiku-4-5, gemini-2.5-flash),
# so they are quarantined until the closed rescore in outputs/closed_reasoning/ lands.
for a in text_reported digitized punish; do
  take "outputs/master_developmental_ladder_${a}_openonly.png"
  take_stale "outputs/master_developmental_ladder_${a}_all.png"
done

# Checkpoint
take outputs/experiments/checkpoint_dissection.png

# Gaps / probes / RSA
take outputs/probe/gap_over_surface_dissociation_span_matched.png gap_over_surface_span_matched.png
take outputs/probe/layerwise_curves.png
take outputs/rsa/model_similarity_heatmap_rsa_spearman.png rsa_similarity_heatmap.png
take outputs/link/rep_vs_behavior.png rsa_convergence_scatter.png
take outputs/link/item_level_dissociation.png

# J3 interaction forest
take outputs/stats/mixed_effects_interaction.png interaction_forest.png

# W3
take outputs/experiments/w3_steering_OLMo-2-1124-7B-Instruct.png w3_steering_dose_OLMo.png
take outputs/experiments/w3_steering_Qwen2.5-7B-Instruct.png w3_steering_dose_Qwen.png
take outputs/experiments/w3_layersweep_OLMo-2-1124-7B-Instruct.png w3_layersweep_OLMo.png
take outputs/experiments/w3_layersweep_Qwen2.5-7B-Instruct.png w3_layersweep_Qwen.png
take outputs/experiments/w3_manipulation_OLMo-2-1124-7B-Instruct.png w3_manipulation_OLMo.png
take outputs/experiments/w3_manipulation_Qwen2.5-7B-Instruct.png w3_manipulation_Qwen.png
take outputs/experiments/w3_prose_rating.png
# W4 / W7 write straight into figures_final, so nothing to copy; listed here so the set of
# canonical figures can be read off one file.
#   w4_curriculum.png            code/experiments/55_w4_summary.py
#   w7_bruneau_selectivity.png   code/experiments/56_w7_bruneau.py

# ToM vs contrast
take outputs/tom_benchmarks/tom_vs_contrast.png

# Closed reasoning dose (lands when 52 finishes)
take outputs/closed_reasoning/reasoning_dose_response.png

echo "figures_final: $(find "$DEST" -maxdepth 1 -name '*.png' | wc -l) pngs"
