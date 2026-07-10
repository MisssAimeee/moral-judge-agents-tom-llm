# Cursor prompt — finish #8, stage #1 (scoring parity) and #4 (checkpoint dissection)

Paste into Cursor (Agent mode), "ToM Project" open, `.venv` active. Tasks 1–2 need NO
model runs. Task 3 (#4 runner) and the run-half of #1 are STAGED to run later on a GPU/API
box — write and dry-run them, do not launch full runs unless I say so.

Read first: `NEXT_PHASE_PLAN.md`, `MASTER_SUMMARY.md`, `dataset/human_reference/README.md`.

---

### TASK 1 — Finish Roadmap #8 (human-anchor hardening).  NO model runs.

Cushman 2013 Fig. 3 has been digitized (values read directly from the figure image, not
WebPlotDigitizer — fix that provenance string). The inputs and a helper already exist:
- `dataset/human_reference/cushman_fig3_attempted.csv` / `..._accidental.csv`  (NAUGHTY = wrongness, "presented first")
- `dataset/human_reference/cushman_fig3_*_PUNISH.csv`  (punishment series, robustness)
- `dataset/human_reference/human_reference_child_digitized.csv` (helper output)

IMPORTANT — the child numbers are NOT yet trustworthy. Two facts:
 - A backup of the current file exists: `dataset/human_reference/human_reference_BACKUP_20260710.csv`.
   Do NOT lose it; keep it committed so old-vs-new is diffable.
 - The child values were only EYEBALLED (~+/-0.05); an automated pixel digitizer
   (`code/digitize_cushman_calibrated.py`) has correct y-axis calibration but its marker
   detection is unreliable on this 4-series figure (error bars + legend contaminate it).
   So do NOT overwrite the primary CSV with either the eyeball or the raw auto-detected numbers.

Decision from the PI: **include BOTH child measures on the figure — do not pick one.**
 - **Naughty (wrongness), presented-first** = PRIMARY child ladder (construct-matched to the
   adult permissibility/wrongness scale): child_4_5 +0.24, child_6_7 +0.50, child_8plus +0.63.
 - **Punish, presented-first** = SECOND child ladder shown alongside (more outcome-based —
   Cushman's own two-process point): child_4_5 +0.09, child_6_7 +0.12, child_8plus +0.19.
Both come from the SAME paper (Cushman 2013), same Fig. 3, different series. Rationale +
provenance: `methods_child_measure.md`, `role_of_human_reference_data.md`.
A backup exists: `dataset/human_reference/human_reference_BACKUP_20260710.csv` — keep it committed.

Do this:
1. Put BOTH measures into the human reference so the pipeline can plot both:
   - Write the NAUGHTY child bands into `human_reference.csv` as the primary child_* rows
     (source = "Cushman 2013 Fig.3, Naughty/wrongness, presented-first, calibrated digitization
     from figure, ~±0.05"). Do NOT touch adult rows; preserve neutral/intentional blanks.
   - Write the PUNISH child bands into a parallel file
     `dataset/human_reference/human_reference_punish.csv` (same columns), labeled clearly.
2. Update the ladder / master figure to draw **two human child curves** (Naughty = solid,
   Punish = dashed, distinct colors) plus the single adult line, with a legend naming the
   measure. Models are unchanged (their contrasts don't move). Show me the figure.
3. Re-run the comparison against the PRIMARY (Naughty) anchor and ALSO report each model's
   placement on the Punish ladder as a secondary column:
   `python code/05_human_comparison.py --template human_verbatim`
   (extend it to load both reference files and emit both placements). Show how many models fall
   below the youngest band on EACH measure.
4. Run `python code/06_fetch_osf_developmental.py` (needs network) to add the OSF accuracy/autism
   curve as a separate reference; if network fails, print the manual step.
5. Emit `outputs/analysis/human_anchor_update.md`: old vs new bands, BOTH measures, the
   both-on-figure decision, and the effect on the headline (models below the youngest children
   on the wrongness ladder; closer on punishment).

### TASK 2 — Roadmap #1 scoring-parity (write now; run the small sampled part when possible).

Goal: prove the logprob-EV scores (open models) and the sampling scores (closed APIs) measure
the SAME thing, so cross-model claims aren't a scoring artifact.
Write `code/analysis/15_scoring_parity.py` that:
1. For a few OPEN-weight models already scored by logprob-EV, ALSO computes a sampled score
   (T=1, n=30) on the same items, then reports per-model correlation + Bland–Altman (mean diff,
   limits of agreement) between EV and sampled contrast. (This part needs a GPU run — gate it
   behind `--run`; default is dry-run that just checks the code path on cached logprobs.)
2. Verifies the logprob EV is computed over the RENORMALIZED valid rating tokens only (probs of
   {1..K} sum to 1) and that multi-digit ratings ("10") aren't truncated; print a warning if not.
3. Output `outputs/analysis/scoring_parity.csv` + a scatter (EV vs sampled contrast, y=x line).
Expected result: tight agreement (r>0.95); if so, the cloud-vs-local comparison is defensible.

### TASK 3 — Roadmap #4 checkpoint-dissection runner (write + dry-run; full run later on GPU).

Goal: localize WHERE in the tuning pipeline the outcome-bias appears. Write
`code/experiments/16_checkpoint_dissection.py` that runs the EXISTING behavioral scoring across
release checkpoints of the SAME base along the tuning pipeline:
- **OLMo-2-7B**: base → SFT → DPO → (RLVR/Instruct) — allenai releases all stages.
- **Tülu-3-8B**: base(Llama-3.1-8B) → SFT → DPO → RLVR — allenai.
- **Zephyr-7B**: Mistral-7B base → SFT → DPO (HuggingFaceH4) as a second family.
For each checkpoint: compute the intent-vs-outcome contrast AND the b_intent / b_outcome / b_interaction
from the 2×2 regression (reuse `code/analysis/11_interaction_regression.py` + `tom_common.py`).
Output `outputs/experiments/checkpoint_dissection.csv` and a figure: x = pipeline stage, y = contrast,
one line per family. Hypothesis to state in the header: the outcome-bias shift appears at the
**DPO/preference-optimization** stage, not SFT, and it shows up as b_outcome↑ more than b_intent↓.
Requirements:
- `--models` list + `--dry-run` (default) that prints the checkpoint plan and token/VRAM estimate
  WITHOUT downloading weights. Only download+run when I pass `--run`.
- Reuse the existing elicitation + logprob-EV scoring; add the entropy QC filter so any degenerate
  checkpoint (like the earlier Mistral failure) is flagged, not silently averaged.

Constraints: no training/fine-tuning anywhere — checkpoints are downloaded as-is for inference only.
Show me: the updated human-anchor comparison (Task 1), the scoring-parity dry-run + code checks
(Task 2), and the checkpoint-dissection plan printed by `--dry-run` (Task 3).
