# Cursor prompt — GPU/API next steps (run when compute is available)

Paste into Cursor (Agent mode), "ToM Project" open, `.venv` active, on a box with a GPU
(and API keys set for the closed-model parts). These are the compute-heavy roadmap items.
Read `NEXT_PHASE_PLAN.md` (§3–§6) and `role_of_human_reference_data.md` first. Do NOT train
or fine-tune any model — all steps are forward-pass inference, probing, or steering.

Work in this order; after each item, save outputs under `outputs/` and show me a summary.

### A. Roadmap #1 — scoring-parity (run the sampled half)
Finish `code/analysis/15_scoring_parity.py`: for ~5 open-weight models already scored by
logprob-EV, ALSO sample (T=1, n=30) on the same items; report per-model correlation and
Bland–Altman agreement between EV and sampled contrast; scatter with y=x. PASS if r>0.95.
This validates that cloud-vs-local comparisons aren't a scoring artifact — do it FIRST, since
several later claims depend on it.

### B. Roadmap #3 — finish the prompt factorial (run the new prompts)
Run the same-scale paraphrases already defined in `code/03_behavioral.py`
(`para_blame10`, `para_blame4`, `acceptable7`, `persona_adult7`) across the open-weight set so
you have ≥6 prompts on a common 1–7 scale. Re-run `14_prompt_invariance_decomposition.py`.
Regenerate `outputs/prompts_used.docx`. Deliver a wording-vs-scale-vs-construct variance table.

### C. Roadmap #4 — instruction-tuning checkpoint dissection  (the standout novel result)
Run `code/experiments/16_checkpoint_dissection.py` for real (`--run`) across pipeline stages of
the SAME base:
 - OLMo-2-7B: base → SFT → DPO → Instruct (allenai)
 - Tülu-3-8B: Llama-3.1-8B base → SFT → DPO → RLVR (allenai)
 - Zephyr-7B: Mistral-7B base → SFT → DPO (HuggingFaceH4)
For each checkpoint compute the intent-vs-outcome contrast AND b_intent / b_outcome /
b_interaction (reuse `11_interaction_regression.py`). Figure: x = pipeline stage, y = contrast,
one line per family. Hypothesis: the outcome-bias shift appears at the DPO/preference stage
(not SFT) and shows as b_outcome↑ more than b_intent↓. Apply the entropy QC filter so any
degenerate checkpoint is flagged, not averaged.

### D. Roadmap #5 — representation probes + rep→behavior link
1. `code/01_extract_activations.py`: forward-pass the open-weight set (base+instruct pairs);
   save residual-stream activations at the last vignette token AND at the belief-clause vs
   action-clause offsets, every layer.
2. `code/02_probe.py`: per layer, L2 logistic probe for intent (guilty/innocent) and outcome
   (harm/no-harm), **GroupKFold by scenario_id** (no leakage). Add RSA/CKA with permutation
   null. Run the same probes on a non-moral control set (Bruneau stories) for selectivity.
3. `code/04_link_analysis.py`: across models, correlate peak-layer intent-decoding accuracy
   with behavioral intent-reliance. This is the "inside the model" contribution.

### E. Roadmap #6 — causal steering (the prize)
Fit an intent direction (difference-of-means guilty−innocent, or the probe vector) at the
peak layer; add/subtract it during generation and measure whether the behavioral contrast
moves as predicted. Controls: a random direction and the outcome direction (should NOT raise
intent-reliance). Verify judgments stay coherent (perplexity/refusal check) so effects aren't
artifacts of breaking the model. A clean positive result = a causal representation→behavior claim.

### F. Roadmap #7 — reasoning / test-time compute
Add reasoning models (DeepSeek-R1 + distills for a size ladder; Claude extended-thinking;
Gemini thinking-budget; o-series if billing allows). Three conditions each: direct answer,
CoT on, scaled thinking budget (low/med/high) → a dose–response curve of test-time compute vs
contrast. For open reasoning models, check whether the CoT explicitly mentions intent and
whether that predicts a higher contrast (treat CoT as behavior, not ground truth).

Constraints: inference/probing/steering only — no weight training. Flag any model with
degenerate (near-constant) ratings rather than averaging it. Keep all outputs under `outputs/`.
Report after each of A–F.
