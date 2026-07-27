# Overnight report — 2026-07-26

## Morning lead (2026-07-27) — gap over surface, **within-model** first

TF-IDF word 1–2gram: **intent 0.594 · outcome 0.755.**
Full matrix: `outputs/probe/gap_over_surface_by_pooling.csv`.
Figure: `outputs/probe/gap_over_surface_dissociation.png`.

### Headline — within-model paired comparison

Same model, same pooling, gap = peak_probe − TF-IDF:

| pooling | models with intent-gap > outcome-gap |
|---|---|
| **belief_last** | **8/8** |
| **action_last** | **8/8** |
| last | 4/8 |
| mean | 4/8 |

**Lead with this.** Intent outpaces outcome as a representational surplus over surface
features at pre-outcome positions in every model. At whole-story poolings the ordering
is mixed (4/8). Peak-across-models is weaker and must not carry the headline.

### Dissociation (anti-lexical)

Intent gap **peaks at belief_last (+0.392)** where harm is not yet stated; outcome gap
**peaks at whole-story (+0.245, ≥0.995)** and collapses to +0.149/+0.110 at pre-outcome
positions. Outcome ceilings = perfect fold separation.

**B3 skipped cache — see `MORNING_B3_B9_C3_C5.md`.** Llama instruct counterexample
survives on B9. Jobs 18962383–93 = pre-strip; re-extract YS2011 Poison/Parent-accidental.
Clause exclusions: **1/53 groups (LAPTOP)**. Scale replication r=0.71 is roadmap item 1.

---

## Lead questions for the morning (answer these first)

1. **Does the checkpoint-dissection `b_outcome` / `b_intent` ratio survive** repaired text +
   correct `scenario_group` bootstrap + corrected CPR labels? State plainly whether the
   2.5–3.9× finding holds, weakens, or reverses — per checkpoint, old vs new.
2. **What is the probe's gap over the TF-IDF surface baseline?** Absolute accuracy is
   no longer the headline number. Compare **matched poolings** (see morning lead table).
   Peak-across-models: intent-gap > outcome-gap at **all four** poolings. Outcome last/
   mean ceilings are **≥0.995** (perfect fold separation), not plain 1.000.

Phase A is complete on local `main` (push still needs credentials). Phase B was re-queued after
a CPR act-only polarity fix invalidated 16 rows mid-chain. Fill sections 2–6 from the new run.

---

## 0. Correction to the work order's premises

Two things in the work order did not match the state of the node, and both changed what needed
doing.

**Job 18957550 did not need cancelling.** It had already completed (00:09:36) and it ran against
`moral_2x2_master_clean.csv`, not the contaminated master — an earlier truncation-based cleaner
built during the previous session. Its output was superseded by the repair, not thrown away for
being contaminated.

**Contamination counts are not one number.** Detector **144** (STUB+BLAME, drives 0.966) ≠
condition-table **96/154** ≠ visible-artefact **99** (accidental 48 / intentional 49 /
attempted 2) on recovered
`dataset/master/_prerepair_backup/moral_2x2_master_CONTAMINATED_20260619.csv`
(md5 `5dd904a7609628553319da4acab02f25`) — **99 is reproducible**. See
`CONTAMINATION_REPAIR.md` §1.2. The same parser bug also *deleted* text: the swallowed lines
never reached the following scenario, so 33 of 48 YS2008 scenarios lost their entire background
and their name. 260 of 298 rows changed in the repair, and most no-harm rows
*gained* around 160 characters rather than losing any. A truncation-only fix would have left that
half of the bug in place, which is why the repair is in `build_dataset.py`.

---

## 1. Did the dataset repair hold?

Yes. `code/experiments/28_validate_master.py` — 10 checks, all pass. Full detail in
`dataset/master/CONTAMINATION_REPAIR.md`.

| | before | after |
|---|---|---|
| harm cells contaminated | 96 / 154 (62.3%) | 0 |
| no-harm cells contaminated | 0 / 144 (0.0%) | 0 |
| contamination flag predicts `outcome_label` | **0.966** | **0.517** (= chance) |
| median word count | 89 | 100.5 |
| YS2008 scenarios with a real name | 15 / 48 | 48 / 48 |
| rows changed | — | 260 / 298 |

Cell counts are unchanged (neutral 72, accidental 77, attempted 72, intentional 77), so nothing
was dropped.

**Root cause.** `parse_ab_factorial` appended every non-`A.`/`B.` line to the current item, so the
rating prompt, the next scenario's ALLCAPS tag and its background were glued onto item 6 (`act_B`).
`act_B` feeds only the `accidental` and `intentional` cells — both harm cells — which is exactly
why contamination tracked the outcome factor at 62% vs 0%.

**Two further defects found and fixed:**

- **Inverted outcome labels.** `YS2008-CPR` and its reprint `YS2009-YS2009_22` list the harmful
  item first, against the appendix convention. All 8 of their cells had `outcome_label` inverted:
  the `no_harm` cells ended *"The customer chokes to death at the table."* Now derived from the
  text. A full audit of all 298 rows' final sentences against their label reports **0 mismatches**.
- **Duplicate scenarios leaking across CV folds** (see §8). This one is independent of the
  contamination and probably matters as much.

**Manual gate (A7):** 20 stories, stratified, seed 0, printed in `outputs/MANUAL_SAMPLE_20.txt` and
read in full. All 20 are a single coherent story with one protagonist and a condition matching the
text. **Hit rate 20/20.**

---

## 2. Did the 0.99 outcome decode survive cleaning?

*Pending B4.* One number is already in, and it moves a lot.

The **TF-IDF surface baseline** — no model activations at all, same LogisticRegression and
GroupKFold protocol — on the repaired dataset with corrected grouping:

| target | word 1-2gram | char 3-5gram | contaminated run |
|---|---|---|---|
| outcome | **0.748** | 0.755 | 0.893 |
| intent | 0.587 | 0.584 | — |

So the surface-only outcome baseline fell from 0.893 to ~0.75. The question in the morning is
whether the probe's 0.99 falls further, the same, or less — what matters is the *gap* over this
baseline and over layer 0, not the absolute number.

---

## 3. Did the checkpoint-dissection conclusion survive?

*Pending B2 → B3.* This is the result most at risk. `b_outcome` carried the finding that tuning
inflates outcome sensitivity 2.5–3.9× more than intent sensitivity, and `b_outcome` was the
coefficient confounded with "this story has ~200 characters of unrelated text glued on". Whether
it holds, weakens or reverses will be stated plainly, per checkpoint.

## 4. Did the headline contrast move?

*Pending B2.* attempted − accidental, per model, old vs new, with CIs. Note this contrast compared
a 0%-contaminated cell against a 62%-contaminated one, so movement is expected.

## 5. Did the Llama-3.1-8B-Instruct counterexample survive?

*Pending B2.*

---

## 6. Jobs

Submitted as an `afterok` chain so a failure halts its branch instead of cascading.

| stage | job | id | depends on |
|---|---|---|---|
| B1 | activations, 8 models × 4 poolings | 18959834 | — |
| B2 | behavioural rescore, 20 open models | 18959837 | — |
| B6 | TF-IDF surface baseline | 18959838 | — (done) |
| B4 | probe, `last` | 18959974 | B1 |
| B4 | probe, `mean` | 18959976 | B1 |
| B8 | probe, `belief_last` | 18959977 | B1 |
| B8 | probe, `action_last` | 18959978 | B1 |
| B5 | layer-0 diagnostic | 18959979 | B4 ×4 |
| B7 | within-cell probes | 18959980 | B4 ×4 |
| B3 | checkpoint dissection | 18959982 | B2 |
| B9 | master ladder | 18959983 | B2, B3 |

```
B1 acts ──┬─ B4/B8 probe ×4 ──┬─ B5 layer0
          │                   └─ B7 within-cell
B2 rescore ─── B3 ckpt ────────── B9 ladder   (B9 also waits on B2)
B6 surface (independent, complete)
```

Contaminated outputs were moved to `outputs/_contaminated_20260726/` — this both preserves them
for the before/after comparison and stops `--skip_existing` from skipping the entire rerun. The
closed-API results in `outputs/agents/` were left untouched.

Two clause-position probe jobs from the previous session (18958413, 18958414) were cancelled;
they were running against activations from the contaminated text.

**Not queued, by instruction:** permutation null (C4), closed-API rescoring (§7), Phases 5–7.

**Progress at 23:30.** B1 activations completed in 8m10s (all 8 models, all 4 pooling variants,
298 × layers × hidden). B6 surface done. Probes for `last`, `mean` and `belief_last` all completed
in under 3 minutes each — the row-space projection makes these far cheaper than the ~16 min/model
budgeted.

`probe_action_last` (18959978) **failed** after 4s, and the `afterok` gate correctly cancelled
`layer0` and `withincell` rather than running them on three-quarters of the input. Cause:
`LogisticRegression` raised *"Found array with 0 feature(s)"*. OLMo's `action_last` layer 0 has
only **3 unique vectors across 298 stories** — layer 0 is the raw token embedding and the pooled
clause-end position is nearly always the same token, a sentence-final period. A GroupKFold fold
that drew a single unique row had no variance left after standardisation, so the row-space
projection returned 0 columns. Such a layer genuinely carries no information, so it now scores the
majority-class rate instead of crashing. Verified on the failing model: peak intent 0.914 @ layer
16. Resubmitted as 18961409 → layer0 18961410, withincell 18961411.

This is worth noting for §2: it is direct evidence that layer-0 read-off is degenerate for the
clause-pooled variants, which is exactly what the B5 diagnostic is meant to establish.

**Earlier issue:** the submit scripts printed a "Monitor with" footer to stdout, so `PARSABLE=1`
captured it along with the job id and the first attempt at the dependent stages failed with
"Job dependency problem". Footer moved to stderr; B1/B2/B6 had already queued cleanly and were
adopted rather than resubmitted.

---

## 7. Estimated API cost for closed-model rescoring — needs your approval

**Superseded by `outputs/API_COST_ESTIMATE.md` (2026-07-27).** Per-provider with real batching:

| | without batching | with available batching |
|---|---:|---:|
| **total (8 models × 3 tmpl × 20 samp)** | **~$56** | **~$40** |
| Anthropic alone (no `n=`) | ~$36 | ~$36 (unchanged) |

Prior ~$95 used legacy Opus $15/$75 (~$60 of the bill). Prior “batching → $10–15”
assumed **universal** prompt-once batching — **invalid for Anthropic**. Full table with
batchable y/n per model in the estimate file.

---

## 8. Second defect: duplicated scenarios were leaking across CV folds

Found by reading the A7 manual sample — `YS2008-HAM` and `YS2009-YS2009_17` are the same vignette
word for word. Systematically: **all 24 YS2009 scenarios are reprints of YS2008 scenarios** under
different ids. YS2009 is 96 of 298 rows, so about a third of the dataset is duplicated and the
effective item count is 202.

Every probe used `GroupKFold(groups=scenario_id)`, so a reprint carrying a different id could put
the identical story in train and test at once. That is leakage, and it inflated every CV accuracy
reported so far — a second plausible contributor to the 0.99 ceiling alongside the contamination.

Fixed with a `scenario_group` column that merges reprints, collapsing 77 ids to **53 CV groups**.
`02_probe.py`, `20_layer0_diagnostic.py`, `21_surface_baseline.py`, `22_within_cell_probes.py` and
`24_rsa_cka.py` all group on it. The overnight chain runs with the fix in place, so the morning
numbers are free of both defects — but it means clean-vs-contaminated comparisons in §2–§5 differ
by grouping as well as by text, and that will be called out where it matters.

---

## 9. Outstanding

`git push origin main` still fails — no credentials available non-interactively
(`could not read Username for 'https://github.com'`). Six commits are now unpushed, including the
dataset repair. Run it in your terminal with your PAT.
