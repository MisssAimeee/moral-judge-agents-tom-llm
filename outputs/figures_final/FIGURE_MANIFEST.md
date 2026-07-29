# Figure manifest — `outputs/figures_final/` (2026-07-28)

Commit at write: `aff8aae`. No figure ships without a row here. Prior locations (`outputs/updated_figures/`, loose root PNGs) are deprecated.

| file | present | what it shows | source CSV | script | dataset / run |
|---|---|---|---|---|---|
| `master_developmental_ladder_digitized_openonly.png` | yes | Developmental ladder vs naughty/wrongness digitized child bands; open-weight only | `outputs/master_all_models_digitized_openonly.csv` | `code/29_dual_human_ladders.py` | moral_2x2_master.csv |
| `master_developmental_ladder_punish_openonly.png` | yes | Same ladder under punish digitized anchor; open-weight only | `outputs/master_all_models_punish_openonly.csv` | `code/29_dual_human_ladders.py` | moral_2x2_master.csv |
| `master_developmental_ladder_text_reported_openonly.png` | yes | Ladder under the superseded pooled-prose anchor; open-weight only (robustness) | `outputs/master_all_models_text_reported_openonly.csv` | `code/29_dual_human_ladders.py` | moral_2x2_master.csv |
| `checkpoint_dissection.png` | yes | Three-family checkpoint contrast trajectory with revised stage shares | `outputs/experiments/checkpoint_dissection.csv` | `code/experiments/16_checkpoint_dissection.py` | rescored 7-template |
| `gap_over_surface_span_matched.png` | yes | Probe−TF-IDF gaps on span-matched baselines; narrowed caption (C2 supporting) | `outputs/probe/gap_over_surface_span_matched.csv` | `code/experiments/33_gap_dissociation_figure.py` | clause_offsets.csv |
| `layerwise_curves.png` | yes | Layer-wise intent/outcome decoding curves | `outputs/probe/*_probe.csv` | `code/02_probe.py / layerwise plotter` | activations last-token |
| `rsa_similarity_heatmap.png` | yes | RSA model×model representational similarity heatmap | `outputs/rsa/` | `code/ rsa scripts` | probe features |
| `rsa_convergence_scatter.png` | yes | RSA convergence vs behavioral contrast (null) | `outputs/link/representation_vs_behavior.csv` | `code/link scripts` | open roster |
| `item_level_dissociation.png` | yes | Item-level intent margin vs contrast (null) | `outputs/link/item_level_dissociation.csv` | `code/link scripts` | open roster |
| `interaction_forest.png` | yes | Mixed-effects interaction forest with cell-ordering annotation (0/20 human order) | `outputs/stats/mixed_effects_2x2.csv` | `code/experiments/39_mixed_effects_2x2.py` | behavior item means |
| `w3_steering_dose_OLMo.png` | yes | W3 dose–response: contrast vs α; intent/outcome/random; OLMo-2-7B-I | `outputs/experiments/w3_steering_OLMo-2-1124-7B-Instruct.csv` | `code/experiments/48_w3_causal_steering.py` | job 19099255 |
| `w3_steering_dose_Qwen.png` | yes | W3 dose–response: contrast vs α; Qwen2.5-7B-I | `outputs/experiments/w3_steering_Qwen2.5-7B-Instruct.csv` | `code/experiments/48_w3_causal_steering.py` | job 19099255 |
| `w3_layersweep_OLMo.png` | yes | W3 M2 layer×direction grid (narrow claim: peak+deeper) | `outputs/experiments/w3_layersweep_OLMo-2-1124-7B-Instruct.csv` | `code/experiments/48_w3_causal_steering.py` | job 19099255 |
| `w3_layersweep_Qwen.png` | yes | W3 M2 layer×direction grid; Qwen | `outputs/experiments/w3_layersweep_Qwen2.5-7B-Instruct.csv` | `code/experiments/48_w3_causal_steering.py` | job 19099255 |
| `w3_manipulation_OLMo.png` | yes | W3 M1: probe-margin displacement vs Δcontrast; OLMo | `outputs/experiments/w3_manipulation_OLMo-2-1124-7B-Instruct.csv` | `code/experiments/48_w3_causal_steering.py` | job 19099255 |
| `w3_manipulation_Qwen.png` | yes | W3 M1: probe-margin displacement vs Δcontrast; Qwen | `outputs/experiments/w3_manipulation_Qwen2.5-7B-Instruct.csv` | `code/experiments/48_w3_causal_steering.py` | job 19099255 |
| `w3_prose_rating.png` | yes | Prose/rating dissociation: cell means among stories naming intent/belief | `outputs/experiments/w3_prose_items_*.csv` | `code/experiments/51_w3_prose_rating.py` | job 19099255 |
| `tom_vs_contrast.png` | yes | BigToM false-belief vs moral contrast scatter (open; closed pending) | `outputs/tom_benchmarks/tom_vs_contrast.csv` | `code/experiments/42_tom_vs_contrast.py` | init_belief=0 |
| `w4_curriculum.png` | **MISSING** | W4 prompt curriculum: contrast at each of 5 cumulative in-context levels | `outputs/experiments/w4_prompt_curriculum.csv` | `code/experiments/55_w4_summary.py` | job 19130876 |
| `w7_bruneau_selectivity.png` | **MISSING** | W7 (appendix): harm decoding across 3 domains + cross-domain transfer | `outputs/experiments/w7_bruneau_probes.csv` | `code/experiments/56_w7_bruneau.py` | job 19131423, Bruneau 2011 stimuli |
| `reasoning_dose_response.png` | **MISSING** | Roadmap #7: thinking budget vs contrast (closed models) | `outputs/closed_reasoning/closed_reasoning_contrasts.csv` | `code/experiments/52_closed_reasoning_dose.py` | CLOSED_MODEL_SELECTION.md |

## Quarantined — `_pending_rescore/` (NOT talk-safe)

Everything in this table is excluded from `figures_final/` proper. Restore under the original name (drop the `STALE_` prefix) only after the closed rescore in `outputs/closed_reasoning/` replaces the v1-era rows and the ladders are regenerated.

| file | present | status and what it shows | source CSV | script | dataset / run |
|---|---|---|---|---|---|
| `_pending_rescore/STALE_master_developmental_ladder_digitized_all.png` | yes | **STALE — contaminated-era closed rows, do not use.** Closed models here (gpt-4o, gpt-4o-mini, claude-haiku-4-5, gemini-2.5-flash) were scored 2026-07-21 against master **v1** and have not been rescored. Digitized ladder including closed models | `outputs/master_all_models_digitized_all.csv` | `code/29_dual_human_ladders.py` | v1-era closed rows |
| `_pending_rescore/STALE_master_developmental_ladder_punish_all.png` | yes | **STALE — contaminated-era closed rows, do not use.** Closed models here (gpt-4o, gpt-4o-mini, claude-haiku-4-5, gemini-2.5-flash) were scored 2026-07-21 against master **v1** and have not been rescored. Punish ladder including closed models | `outputs/master_all_models_punish_all.csv` | `code/29_dual_human_ladders.py` | v1-era closed rows |
| `_pending_rescore/STALE_master_developmental_ladder_text_reported_all.png` | yes | **STALE — contaminated-era closed rows, do not use.** Closed models here (gpt-4o, gpt-4o-mini, claude-haiku-4-5, gemini-2.5-flash) were scored 2026-07-21 against master **v1** and have not been rescored. Pooled-prose ladder including closed models | `outputs/master_all_models_text_reported_all.csv` | `code/29_dual_human_ladders.py` | v1-era closed rows |
