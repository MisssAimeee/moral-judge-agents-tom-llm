# Confound diagnostics — Phase 1 gate report

> ## ⚠ SUPERSEDED IN PART — read this box first
>
> After C1–C5 were complete, a **stimulus contamination bug** was found that invalidates
> parts of the analysis below. 144 of 298 stories carried the study's rating prompt plus the
> opening of the *next* scenario appended to the text, and the contamination was almost
> perfectly confounded with the outcome factor (0/144 no-harm cells vs 144/154 harm cells;
> a binary "has trailing junk" flag predicts outcome at **0.966**).
>
> Everything was rerun on cleaned stimuli. **What changed:**
>
> - **The `intent_harm` collapse was entirely an artefact.** It read 0.565–0.676 on
>   contaminated text and reads **0.823–0.968** on clean text. The "harm suppresses intent
>   representation" hypothesis in the C3 section below is **withdrawn**.
> - **The outcome result shrinks.** The clean surface baseline is higher (0.893 vs 0.835)
>   and the model gap over it falls from +0.13–0.17 to **+0.07–0.09**.
> - **The intent result survives and improves**, gap now **+0.036 → +0.181** with scale.
>
> The clean numbers are in "Clean rerun" at the bottom and are the ones to cite. The
> contaminated sections are retained for provenance.

**Run 2026-07-26.** Tasks C1–C5 of the Confound Resolution + RSA plan.
Models: Qwen2.5-{0.5B, 1.5B, 7B} ± Instruct, OLMo-2-1124-7B ± Instruct (8 total), 298 stimuli.
Protocol throughout: `LogisticRegression(max_iter=2000, C=1.0)` inside `GroupKFold(n_splits=5)`
grouped by `scenario_id`, `StandardScaler` fit per fold.

---

## The question this phase had to answer

> Is the outcome probe lexically explainable? Is the intent probe?

**Answers, up front:**

1. **No, the outcome probe is not merely lexical.** It sits 0.13–0.17 above the strongest
   surface baseline. It *is* partly lexical — surface features alone reach 0.835 — but the
   models clear that by a wide margin.
2. **For intent it depends on scale.** At 0.5B, intent decoding (0.768) is barely above the
   surface baseline (0.745) and should not be claimed. At 7B and above the gap is
   0.14–0.19, and in the tightest available control (harm absent from both cells) it reaches
   0.21. **The intent result survives at 7B+ and should be scoped to that range.**
3. **A methodological finding that changes the headline numbers:** last-token pooling badly
   understates intent. OLMo-2-7B-Instruct reads 0.772 with last-token pooling and **0.936**
   with mean pooling. All previously reported intent numbers were last-token and were too low.

---

## C1 — Layer-0 read-off

Layer 0 is the embedding output, before any attention or MLP. Accuracy there bounds what is
available with zero computation.

An important correction to the plan's assumption: with **last-token** pooling, layer 0 is the
embedding of a *single token* (the story's final token), not a bag of words. Since the stories
end differently ("…dies" vs "…is fine") that one token leaks outcome by itself. The genuine
bag-of-embeddings test is **mean** pooling at layer 0, so both are reported.

| Pooling | intent @ L0 | outcome @ L0 |
|---|---|---|
| last (1 token) | 0.500–0.503 | 0.601–0.604 |
| mean (bag-of-embeddings) | 0.731–0.768 | 0.886–0.926 |

**Read:** the feared failure mode — outcome ≈ 0.99 at layer 0 — did **not** occur under either
pooling. Outcome at mean-pooled L0 is 0.89–0.93 against a peak of 0.96–1.00, so most but not
all of the outcome signal is available pre-computation. Intent at last-token L0 is exactly
chance, which is the shape we wanted; at mean-pooled L0 it is 0.73–0.77, which is essentially
the TF-IDF baseline and confirms that a bag of embeddings already carries substantial intent
lexis.

Files: `layer0_diagnostic.csv`, `layer0_pooling_check.csv`, `layerwise_curves.png`.

## C2 — TF-IDF surface baseline

No activations. Identical classifier and CV protocol, imported from `02_probe.py`.

| Subset | Target | word 1–2gram | char 3–5gram | chance |
|---|---|---|---|---|
| all | intent | 0.745 | 0.752 | 0.500 |
| all | outcome | 0.835 | 0.832 | 0.517 |
| intent_noharm | intent | 0.783 | 0.763 | 0.500 |
| intent_harm | intent | 0.791 | 0.727 | 0.500 |
| outcome_innocent | outcome | 0.798 | 0.825 | 0.517 |
| outcome_guilty | outcome | 0.791 | 0.818 | 0.517 |

**This table is the reason the phase existed.** Surface text alone gets 0.75 on intent and 0.84
on outcome. Any probe number quoted against a 0.5 "chance" line is misleading; the operative
reference is this baseline. Note the restricted subsets have *higher* intent baselines than the
full set, so within-cell probes must clear a higher bar, not a lower one.

## C1+C2 combined — the mean-pooled comparison that matters

| Model | intent | vs TF-IDF | outcome | vs TF-IDF |
|---|---|---|---|---|
| Qwen2.5-0.5B | 0.768 | +0.023 | 0.973 | +0.138 |
| Qwen2.5-0.5B-Instruct | 0.788 | +0.043 | 0.967 | +0.132 |
| Qwen2.5-1.5B | 0.845 | +0.100 | 0.963 | +0.128 |
| Qwen2.5-1.5B-Instruct | 0.856 | +0.111 | 0.973 | +0.138 |
| Qwen2.5-7B | 0.889 | +0.144 | 0.993 | +0.158 |
| Qwen2.5-7B-Instruct | 0.896 | +0.151 | 1.000 | +0.165 |
| OLMo-2-1124-7B | 0.929 | +0.184 | 0.997 | +0.162 |
| OLMo-2-1124-7B-Instruct | 0.936 | +0.191 | 0.993 | +0.158 |

**The intent gap over surface grows monotonically with scale (+0.02 → +0.19).** That is the
signature of genuine computation: a larger model extracts progressively more intent information
than the text affords lexically. The outcome gap is roughly constant (+0.13 to +0.17), i.e. all
models clear the surface baseline on outcome by a similar margin.

## C3 — Within-cell contrasts (the tightest lexical control)

Each factor isolated with the other held constant, compared against that subset's own TF-IDF
baseline. Last-token pooling.

| Model | intent_noharm | intent_harm | outcome_innocent | outcome_guilty |
|---|---|---|---|---|
| Qwen2.5-0.5B | 0.840 (+0.056) | 0.565 (−0.226) | 0.987 (+0.162) | 0.993 (+0.175) |
| Qwen2.5-0.5B-Instruct | 0.840 (+0.056) | 0.565 (−0.226) | 0.987 (+0.162) | 0.993 (+0.175) |
| Qwen2.5-1.5B | 0.916 (+0.133) | 0.604 (−0.187) | 0.993 (+0.168) | 1.000 (+0.182) |
| Qwen2.5-1.5B-Instruct | 0.909 (+0.125) | 0.604 (−0.188) | 0.993 (+0.168) | 1.000 (+0.182) |
| Qwen2.5-7B | 0.966 (+0.182) | 0.670 (−0.121) | 0.993 (+0.168) | 1.000 (+0.182) |
| Qwen2.5-7B-Instruct | 0.993 (+0.210) | 0.662 (−0.130) | 0.993 (+0.168) | 1.000 (+0.182) |
| OLMo-2-1124-7B | 0.973 (+0.190) | 0.668 (−0.123) | 1.000 (+0.175) | 1.000 (+0.182) |
| OLMo-2-1124-7B-Instruct | 0.959 (+0.176) | 0.676 (−0.116) | 0.993 (+0.168) | 1.000 (+0.182) |

**`intent_noharm` is the headline and it passes.** Neither included cell contains a harm event,
so harm-word detection cannot explain the result, and every model beats its own surface
baseline — by +0.06 at 0.5B rising to +0.21 at 7B-Instruct. **The lexical-harm account of the
intent probe is dead.**

**`intent_harm` is an unexpected and substantive finding.** When harm is present in both cells,
intent decoding *collapses* to 0.57–0.68, which is **below** the 0.791 TF-IDF baseline —
the model represents intent worse than the raw text statistics do. The asymmetry is large
(0.96 vs 0.67 for Qwen-7B) and consistent across all 8 models.

The natural reading is that **the presence of harm suppresses the model's representation of
intent** — a representational analogue of the behavioural outcome bias documented in the
checkpoint dissection. This should be treated as a hypothesis, not an established mechanism:
an alternative account is that the accidental/intentional cells are simply harder to separate
lexically in a way the probe cannot recover. Phase 3 (clause offsets) discriminates these,
because it can test whether intent is encoded at the belief clause and then *lost* by the end
of the story.

Note the base/instruct pairs at 0.5B produce identical values to three decimals. This was
checked and is a genuine coincidence, not a data bug: the two `.npz` files differ
(max |Δ| = 37.3, mean |Δ| = 0.16) and the output CSVs have different checksums. With n = 144
accuracy is quantised to ~0.007 steps, so ties are expected.

## C4 — Permutation null (N = 1000, mean pooling)

Labels shuffled **within** scenario so the group dependency structure is preserved.
32 cells (8 models × 2 targets × {layer 0, peak layer}).

- Null mean: 0.500 for intent, 0.507–0.512 for outcome. This confirms the grouped design is
  behaving — an unstructured null would not sit this cleanly on chance.
- Null 95th percentile: 0.544–0.563.
- **Every one of the 32 cells returns p = 0.001**, the floor for N = 1000.

**Interpretation, and its limit:** this establishes that no reported accuracy is a fluke of the
grouping or class balance. It does **not** speak to the confound at all — layer-0 accuracies
(0.73–0.93) are also p = 0.001, and those are precisely the numbers we believe to be largely
surface. The permutation null answers "is it above chance"; only the TF-IDF baseline answers
"is it above lexis". Both must be quoted.

File: `<model>_permnull_mean.csv`.

## C5 — `04_link_analysis.py` unblocked

`intent_reliance_summary.csv` now exists, prompt-averaged across templates with SD retained.

The effect-size floor (|b_intent| + |b_outcome| ≥ 0.05) flags exactly the models already known
to be degenerate — Zephyr-7B, both Mistral-7B variants, gemma-2-9b base, and the Qwen 0.5B/1.5B
bases — and they receive no index rather than a fabricated one. This matters: without the floor,
Qwen2.5-0.5B-Instruct/`para_wrong7` (b_intent = −0.0015, b_outcome = 0.0019, both noise) would
have contributed an index of 0.44 to the average.

The join resolves **6 of 8** probed models, meeting the acceptance bar; the two missing are
degenerate bases, correctly excluded. `04` runs clean and produces
`outputs/link/rep_vs_behavior.png`.

**Result: r = −0.27 over 6 models — not significant.** With n = 6 this is uninformative in
either direction and should not be reported as a finding yet.

---

## Gate decision

| Question | Answer |
|---|---|
| Is the outcome probe lexically explainable? | **No.** +0.13–0.17 over surface, and it clears the harm-free control at 0.99. It is *partly* lexical (surface alone reaches 0.835) and that must be stated. |
| Is the intent probe lexically explainable? | **At 0.5B, essentially yes** (+0.02). **At 7B+, no** (+0.14–0.19 overall; +0.18–0.21 in the harm-free control). |

**Recommendation on Phase 4 (new control stimuli):** intent survived C1 and C3 cleanly at 7B+,
so per the plan's own gate criterion Phase 4 can be **descoped to a robustness appendix**. Two
caveats before finalising that call:

- The `intent_harm` collapse is unexplained and is the most interesting result in this phase.
  Phase 3 (clause offsets) is the right instrument for it, and is already scheduled.
- Claims should be **scoped to 7B+**. The 0.5B and 1.5B models do not clear the surface
  baseline convincingly on the full set, so the scale ladder should be presented as a gradient
  rather than as eight independent confirmations.

## Scripts and artefacts

| Task | Script | Output |
|---|---|---|
| C1 | `code/experiments/20_layer0_diagnostic.py` | `layer0_diagnostic.csv`, `layer0_pooling_check.csv`, `layerwise_curves.png` |
| C2 | `code/experiments/21_surface_baseline.py` | `surface_baseline.csv` |
| C3 | `code/experiments/22_within_cell_probes.py` | `<model>_withincell.csv` |
| C4 | `code/02_probe.py --permute 1000` | `<model>_permnull_mean.csv` |
| C5 | `code/experiments/23_build_intent_reliance_summary.py` | `intent_reliance_summary.csv`, `outputs/link/*` |

### Compute note

`02_probe.py` gained an exact row-space projection. For an L2-penalised linear model the
solution lies in the training row space, so rotating into an orthonormal basis of it leaves
predictions unchanged while cutting the feature dimension from 4096 to ≤238. Predictions are
identical, not approximated. The full sweep went from **1h37m to 2m57s** (26× faster), which is
the only reason a 1000-rep permutation null was affordable — at the original cost it would have
been roughly 35 hours per model.

One side effect: two of the originally published cells shift by 0.0033 (a single test item).
The original 4096-dimensional fits were hitting `max_iter` without converging; the projected
fits converge, so the new values are the more correct ones.

---

# Clean rerun — these are the numbers to cite

Stimuli: `dataset/master/moral_2x2_master_clean.csv` (144 stories truncated at the rating
prompt, mean 180 characters removed; contamination flag now predicts outcome at 0.517 = chance).
Activations re-extracted (`outputs/acts_clean`, job 18955158, 4m57s); probes, within-cell and
RSA rerun. Mean pooling throughout.

## Surface baselines, clean

| Target | word 1–2gram | char 3–5gram | change vs contaminated |
|---|---|---|---|
| intent | 0.745 | 0.745 | unchanged |
| outcome | **0.893** | 0.889 | **+0.058** |

Intent is unchanged exactly as predicted — it was never confounded with the artefact. The
outcome baseline *rises*, because removing ~180 characters of unrelated next-scenario text
makes the genuine harm lexis easier for TF-IDF to isolate. The contamination was adding noise
for a bag-of-words model even while acting as a near-perfect flag.

## Main probes, clean

| Model | intent | vs surface | outcome | vs surface |
|---|---|---|---|---|
| Qwen2.5-0.5B | 0.781 | +0.036 | 0.967 | +0.074 |
| Qwen2.5-0.5B-Instruct | 0.795 | +0.050 | 0.966 | +0.073 |
| Qwen2.5-1.5B | 0.855 | +0.110 | 0.963 | +0.070 |
| Qwen2.5-1.5B-Instruct | 0.872 | +0.127 | 0.963 | +0.070 |
| Qwen2.5-7B | 0.896 | +0.151 | 0.983 | +0.090 |
| Qwen2.5-7B-Instruct | 0.916 | +0.171 | 0.983 | +0.090 |
| OLMo-2-1124-7B | 0.920 | +0.175 | 0.983 | +0.090 |
| OLMo-2-1124-7B-Instruct | 0.926 | +0.181 | 0.987 | +0.094 |

**The headline flips relative to the naive reading.** Outcome has the higher raw accuracy
(0.96–0.99) but adds only +0.07–0.09 over surface, because whether harm occurred is stated in
plain words and TF-IDF already gets 0.893. Intent has the lower raw accuracy but at 7B+ adds
**+0.15 to +0.18** — roughly double outcome's contribution.

> The models contribute more to intent than to outcome. Outcome is mostly *read*; intent is
> mostly *computed*.

The intent gap still grows monotonically with scale (+0.036 at 0.5B → +0.181 at OLMo-7B), and
claims should still be scoped to 7B+ where the margin is comfortable.

## Within-cell, clean

Gap is against each subset's own clean TF-IDF baseline
(intent_noharm 0.783, intent_harm 0.765, outcome_innocent 0.892, outcome_guilty 0.906).

| Model | intent_noharm | intent_harm | outcome_innocent | outcome_guilty |
|---|---|---|---|---|
| Qwen2.5-0.5B | 0.881 (+0.098) | 0.823 (+0.057) | 0.973 (+0.081) | 0.967 (+0.060) |
| Qwen2.5-0.5B-Instruct | 0.875 (+0.092) | 0.837 (+0.072) | 0.973 (+0.081) | 0.960 (+0.054) |
| Qwen2.5-1.5B | 0.917 (+0.133) | 0.923 (+0.158) | 0.980 (+0.088) | 0.980 (+0.074) |
| Qwen2.5-1.5B-Instruct | 0.930 (+0.147) | 0.915 (+0.150) | 0.986 (+0.094) | 0.980 (+0.074) |
| Qwen2.5-7B | 0.938 (+0.155) | 0.941 (+0.175) | 1.000 (+0.108) | 1.000 (+0.094) |
| Qwen2.5-7B-Instruct | 0.972 (+0.189) | 0.941 (+0.176) | 1.000 (+0.108) | 1.000 (+0.094) |
| OLMo-2-1124-7B | 0.966 (+0.182) | 0.935 (+0.170) | 1.000 (+0.108) | 0.986 (+0.080) |
| OLMo-2-1124-7B-Instruct | 0.979 (+0.195) | 0.968 (+0.202) | 1.000 (+0.108) | 0.993 (+0.087)

**`intent_noharm` still passes, and the `intent_harm` asymmetry is gone.** Every cell of every
model now clears its own surface baseline. Intent decodes essentially as well with harm present
as absent, which is the correct null result — and it means the contaminated-data claim that
"harm suppresses intent representation" was measuring the ~180 characters of appended text,
not the model.

This is the cleanest form of the argument: in the harm-free control, where no story contains a
harm event, intent decodes at 0.88–0.98 against a 0.783 lexical ceiling. Harm-word detection
cannot produce that.

## Revised gate decision

| Question | Answer (clean data) |
|---|---|
| Is the outcome probe lexically explainable? | **Largely yes.** Surface alone reaches 0.893 and models add only +0.07–0.09. Outcome should be presented as a mostly-lexical variable, not as evidence of moral representation. |
| Is the intent probe lexically explainable? | **No, at 7B+.** +0.15–0.18 overall and +0.17–0.20 in the harm-free control. At 0.5B (+0.036) it should not be claimed. |

Phase 4 (new control stimuli) can still be **descoped to a robustness appendix** — the harm-free
within-cell control does the essential work. The Bruneau selectivity control remains valuable
for the *outcome* claim specifically, which is now the weaker of the two.

## Artefacts (clean)

`dataset/master/moral_2x2_master_clean.csv`, `outputs/acts_clean/`, `outputs/probe_clean/`,
`outputs/probe/surface_baseline_clean.csv`, `outputs/rsa_clean/`.
Cleaner: `code/experiments/27_clean_stimuli.py`.

## Caveat that still stands

**Behavioural scores were collected on the contaminated text.** Every rating in
`outputs/behavior/`, the developmental ladder, the checkpoint dissection and the scoring-parity
analysis used prompts in which 48% of items had a rating stub and a fragment of the next
scenario appended. The representational track is now clean; the behavioural track is not.
Re-scoring is a substantial inference job and needs a decision on scope before launching.
