# Stimulus repair: story-boundary contamination in `moral_2x2_master.csv`

**Date:** 2026-07-26 · **Repaired in:** `code/build_dataset.py` · **Gate:** `code/experiments/28_validate_master.py`

All behavioural and representational results produced before this date used the contaminated
master and must be regenerated. This note is the provenance record for the repair.

---

## 1. What was wrong

`parse_ab_factorial()` merged wrapped PDF lines into each `A.`/`B.` item by appending every
following line that was not itself an `A.`/`B.` marker. In the source appendices the rating
prompt, the next scenario's ALLCAPS tag, and the next scenario's background all sit immediately
after the last item with no blank line, so they were appended to item 6.

Raw source, `YS2008.txt` lines 25–30:

```
B. Grace puts the substance in her friend's coffee. Her friend drinks the coffee and dies.   <- item 6 (act_B)
Putting the substance in was:                                                                <- rating prompt
LAB                                                                                          <- next scenario tag
Dan is giving a visitor a tour of a lab. Before visitors enter the testing room, all test tubes
containing disease antigens must be contained in a chamber by flipping a switch. A
repairman has just come to fix the switch, which had been broken.                            <- next scenario background
```

Everything from the rating prompt down was glued onto `act_B`.

This produced **two** defects from one bug, and the second was the more damaging:

**Defect 1 — trailing contamination, confounded with outcome.** `act_B` is used by exactly the
`accidental` and `intentional` cells. Both are harm cells. So the contamination was almost
perfectly aligned with the outcome factor:

| condition | contaminated (before) | after |
|---|---|---|
| neutral | 0 / 72 (0%) | 0 |
| attempted | 0 / 72 (0%) | 0 |
| accidental | 48 / 77 (62%) | 0 |
| intentional | 48 / 77 (62%) | 0 |
| **harm cells** | **96 / 154 (62.3%)** | **0** |
| **no-harm cells** | **0 / 144 (0.0%)** | **0** |

A "does this story have an unrelated tail glued on" flag predicted `outcome_label` at **0.966**
accuracy. After repair it predicts at **0.517**, which is the majority-class rate (154/298), not
0.5 by coincidence — see §1.1. `b_outcome` and the 0.99–1.00 outcome decoding were therefore
both confounded with this artefact.

### 1.1 Derivation of 0.966 → 0.517 (exact)

This is **accuracy**, not AUC. There is no trained classifier and no cross-validation.

| piece | value |
|---|---|
| **Metric** | Accuracy = `max(agree, 1 − agree)` where `agree = mean(flag == (outcome_label == "harm"))` over all 298 rows. The `max` allows the flag's polarity to be flipped; it does not change the before number. |
| **Predictor** | Binary `contaminated(text)`: true iff a rating-prompt cut point exists before `len(text)`. Cut point = earliest match of either `STUB = r"\b[A-Za-z][^.!?]*?\b(?:was\|were\|is\|are):"` or `BLAME = r"How much (?:blame\|punishment)[^?]*\?"` (same detector as `27_clean_stimuli.py`). |
| **Target** | `outcome_label == "harm"`. |
| **Classifier** | None. The binary flag *is* the prediction. Equivalent to a single-feature threshold rule with no free parameters. |
| **CV scheme** | None. In-sample agreement on the full 298-row master. |

| | n_flag | agree | accuracy | majority-class rate |
|---|---|---|---|---|
| **before** | 144 (all in harm cells; 10/154 harm cells unflagged) | 0.9664 | **0.9664** | 0.5168 (154/298) |
| **after** | 0 | 0.4832 | **0.5168** | 0.5168 (154/298) |

After repair the flag is constant (`False` for every row), so `agree` equals the no-harm rate
(144/298 = 0.483) and `max(agree, 1−agree)` equals the harm rate (154/298 = 0.517). That is
exactly the accuracy of the best constant predictor of `outcome_label`. It is **not** 0.5 by
chance — the class balance is 154:144, and 0.5168 = 154/298. Confirmed: post-repair
`n_flag == 0` and `accuracy == majority-class rate` to machine precision.

**What 0.517 does *not* show.** With zero flags the contamination detector is a constant
predictor, so accuracy collapsing to the majority rate demonstrates only that the
contamination *signal is gone* — not that harm and no-harm cells are surface-matched.
Residual surface predictability of outcome is measured by the TF-IDF baseline on the
repaired master: **word 1–2gram outcome = 0.755** (see `outputs/probe/surface_baseline.csv`;
overnight text sometimes rounds to 0.748). Report both: 0.517 = contamination gone;
0.755 = surface signal that remains.

### 1.2 Reconciling “144 flags” vs “99 visible” vs “96 / 154”

These are **three different counters**, not one bug counted three ways:

| count | what it measures | where |
|---|---|---|
| **144** | `STUB+BLAME` detector (`27_clean_stimuli.contaminated`) on the pre-repair master. Required for 0.966: TP=144, TN=144 → agree=288/298. | §1.1; `27_clean_stimuli.py` |
| **96 / 154 (62%)** | Harm cells in the per-condition repair table (48 accidental + 48 intentional) under the act_B glue pattern. | §1 table above |
| **99** | **Reproducible** visible trailing-artefact count on the recovered pre-repair master. Breakdown: **accidental 48 / intentional 49 / attempted 2**. File: `dataset/master/_prerepair_backup/moral_2x2_master_CONTAMINATED_20260619.csv` (md5 `5dd904a7609628553319da4acab02f25`). | this section; overnight §0 |

So: **144 = regex detector; 96 = condition-table harm rate; 99 = visible-artefact tally on the archived contaminated CSV (reproducible).** The 0.966 confound is tied to the 144-flag detector. The gap between 144 and 99 is detector hits that are not (or not all) “visible” trailing junk under the artefact definition used for the 99 count — do not silently equate them. The overnight report’s earlier wording that 99 was unreproduced is **superseded**; the backup survived locally and is the provenance source of truth for that tally.

**Defect 2 — the following scenario lost its name and background.** Because those lines were
consumed into the previous item, they never reached the next scenario. 33 of 48 YS2008 scenarios
had no background at all and fell back to a generic `YS2008_NN` id. `YS2008_02` began mid-story:

> "The switch has been successfully repaired, so the test tubes are quite safely contained..."

with no lab, no switch, and no protagonist ever introduced. Truncation-based cleaning cannot fix
this; only re-parsing can. This is why the repair is in the builder rather than a post-hoc filter.

---

## 2. The fix

Three rule changes in `code/build_dataset.py`. No row was hand-edited; the CSV is regenerated.

**(a) Terminate the continuation buffer at a prompt or header line.** Fixes both defects at once —
the tail stops being appended, and the freed lines flow to the next scenario as its background.

```python
SCEN_HEADER = re.compile(r"[A-Z][A-Z0-9 &/\-]{2,}")
...
elif is_judgment(s) or SCEN_HEADER.fullmatch(s):
    if buf is not None:
        logical.append(buf); buf = None
    logical.append(("TEXT", s))
```

`is_judgment` (pre-existing) matches lines ending in `?` or `:`, or beginning
`how much` / `putting the` / `doing` / `was:` / `how morally`.

**(b) Derive act polarity from the text instead of assuming item B is harmful.**

CPR (and its YS2009 reprint) invert only the *action* items in the source appendix —
`act_A` ends in death, `act_B` in "is fine" — while `fore_A` remains the safe world
(chilli) and `fore_B` the dangerous one (choking). An earlier draft of this repair swapped
*both* fore and act, which made `outcome_label` agree with the final sentence but left the
vignettes incoherent (choking world + "is fine", or chilli world + death). The correct fix
swaps **act only**:

```python
HARM_OUTCOME = re.compile(
    r"\b(dies?|died|death|dead|kill(?:s|ed)?|chokes? to death|never wakes up|coma|"
    r"emergency room|hospital|third degree|paralys\w*|drowns?|contract diseases)\b", re.I)
...
a_harm, b_harm = bool(HARM_OUTCOME.search(act_A)), bool(HARM_OUTCOME.search(act_B))
if a_harm and not b_harm:
    act_A, act_B = act_B, act_A   # fore polarity is already correct — do not swap it
```

**(c) Strip a trailing bare scenario title in the 2011 parser** (`"...with poison. Incest"`).

```python
body = re.sub(r"(?<=[.!?)])\s+[A-Z][a-z]{2,}\s*$", "", body).strip()
```

Plus: `clean()` restores a final period on the few vignettes whose PDF text layer drops it, and
`word_count` is recomputed from the cleaned text (it was previously computed post-contamination
and was wrong for every affected row).

---

## 3. Effect of the repair

260 of 298 rows changed. Note the sign difference: no-harm cells **gained** restored background,
harm cells net **lost** text after gaining background and shedding the tail.

| condition | rows changed | min Δchars | median Δchars | max Δchars |
|---|---|---|---|---|
| neutral | 57 | +2 | **+160** | +245 |
| attempted | 57 | +2 | **+160** | +245 |
| accidental | 74 | −279 | **−62** | +162 |
| intentional | 72 | −279 | **−63** | +162 |
| all | 260 | −279 | +100 | +245 |

- Median word count: **89 → 100.5**
- YS2008 scenarios with a recovered name: **15/48 → 48/48**
- Contamination/outcome alignment: **0.966 → 0.517** (chance)

### Before / after examples

**`YS2008-COFFEE-accidental`** — 823 → 544 chars (tail removed)

- before, tail: `...Her friend drinks the coffee and dies. Putting the substance in was: LAB Dan is giving a visitor a tour of a lab. Before visitors enter...`
- after, tail: `...Grace puts the substance in her friend's coffee. Her friend drinks the coffee and dies.`

**`YS2008_02` → `YS2008-LAB-*`** — background and name recovered

- before: `The switch has been successfully repaired, so the test tubes are quite safely contained...`
- after: `Dan is giving a visitor a tour of a lab. Before visitors enter the testing room, all test tubes containing disease antigens must be contained in a chamber by flipping a switch. A repairman has just come to fix the switch, which had been broken. The switch has been successfully repaired...`

**`YS2008-YS2008_40-*` (now `FRAT`)** — dangling prompt only, 27 chars

- before, tail: `...where he never wakes up. Not waking his friend was:`
- after, tail: `...where he never wakes up.`

**`YS2011-Poison-accidental`** — bare next-scenario title

- before, tail: `...someone replaced the sugar with poison. Incest`
- after, tail: `...someone replaced the sugar with poison.`

**`YS2008-CPR-*` and `YS2009-YS2009_22-*`** — outcome labels were inverted in all 8 cells; see §4.

---

## 4. Label audit (Task A5)

The suspected error is real and was **not** contamination. Two scenarios — `YS2008-CPR` and its
reprint `YS2009-YS2009_22` — list the harmful item first, opposite to the appendix convention.
Taking the convention on faith inverted `outcome_label` in **all four cells of both scenarios (8
rows)**: the `no_harm` cells ended *"The customer chokes to death at the table."* and the `harm`
cells ended *"The customer has a glass of water and is fine."*

Fixed by rule (b) above, not by relabelling. After repair, an independent audit comparing each
row's final sentence against its `outcome_label` across all 298 rows reports **0 mismatches**.

**Audit of the no-harm cells.** 24 of the 144 `attempted`/`neutral` rows contain harm vocabulary,
but in every case it sits in the *belief* clause, not the outcome — which is the design of the
`attempted` condition (the agent believes harm will follow; it does not). Example,
`YS2008-IRON-attempted`: belief *"...believes that it is still hot and could easily burn her
sister"*, outcome *"They have fun continuing their makeovers."* No no-harm cell has a harmful
outcome.

---

## 5. Clause offsets (Task A6)

`clause_offsets.csv` regenerated against the repaired master; YS2011 hand-annotated
(`method=manual`, `32_ys2011_manual_clauses.py`). Method distribution over 298 rows:

| method | n |
|---|---|
| `belief_verb` (pattern matched) | 280 |
| `manual` (YS2011, eye-verified) | 10 |
| `fallback_position` (guessed; **excluded from clause probes**) | 8 |

**Clause-position exclusions:** **1 of 53 `scenario_group`s — LAPTOP** (YS2008-LAPTOP +
reprint YS2009_18; 8 `fallback_position` cells). Naming both source ids without collapsing
would read as “2 vignettes,” but CV grouping merges them. YS2011 is `manual` (10/10), not
excluded.

`02_probe.py::load_clause_mask` keeps only `belief_verb*` and `manual`. Fallback and
unannotated rows never enter belief_last / action_last.

The 15 non-standard rows, by scenario: `LAPTOP` (4), `YS2009_18` (4), `Parent` (2), `Sibling` (2),
`Allergy` (1), `Poison` (1), `Dog` (1). They do **not** cluster by condition — the two factorial
scenarios contribute all four cells each, and the 7 YS2011 rows are second-person vignettes
("You spoon some powder into your co-worker's coffee") with no `X believes` construction for the
pattern to match. The clustering is by scenario grammar, not by experimental condition, so it does
not bias the contrast.

**These rows are excluded, not defaulted.** `02_probe.py::load_clause_mask` drops any row whose
`method != belief_verb` from the `belief_last` and `action_last` poolings. A position-guessed
belief offset can land on the outcome sentence, which would smuggle end-of-story information into
the belief probe; a defaulted row is indistinguishable from a real one in the output, so it must
not be silently retained.

---

## 6. A second, independent defect: duplicated scenarios leaked across CV folds

Found while reading the 20-item manual sample — `YS2008-HAM` and `YS2009-YS2009_17` are the same
vignette, word for word. Checking systematically: **all 24 YS2009 scenarios are reprints of YS2008
scenarios** under different `scenario_id`s (JELLYFISH/YS2009_24, SESAME/YS2009_21, CPR/YS2009_22,
and 21 more). YS2009 contributes 96 of 298 rows, so roughly a third of the dataset is duplicated
material and the effective item count is 202, not 298.

This is unrelated to the contamination and has its own consequence. Every probe uses
`GroupKFold(groups=scenario_id)` to hold out whole vignettes. Because a reprint carries a
different id, `HAM` could sit in train while `YS2009_17` — the identical story — sat in test.
That is train/test leakage, and it inflates every reported CV accuracy. It is a second plausible
contributor to the 0.99–1.00 outcome-decoding ceiling, alongside the contamination.

**Fix.** `build_dataset.py` now emits a `scenario_group` column that merges reprints, matched on
the opening content words of the neutral cell. This collapses 77 `scenario_id`s to **53 CV
groups**. `02_probe.py`, `20_layer0_diagnostic.py`, `21_surface_baseline.py`,
`22_within_cell_probes.py` and `24_rsa_cka.py` all group on `scenario_group` now, falling back to
`scenario_id` if the column is absent.

The duplicated rows are retained rather than dropped, since the two papers' versions differ
slightly in wording and both were run behaviourally; grouping them is sufficient to stop the
leakage. Whether to down-weight them in the behavioural aggregates is a separate open question.

---

## 7. Validation gate

`code/experiments/28_validate_master.py` — all 10 checks pass. Exits non-zero on failure so it can
gate an sbatch chain.

1. no trailing response-prompt fragment · 2. no ALLCAPS scenario tags · 3. 298 rows, cells
72/77/72/77 · 4. terminal punctuation · 5. `word_count` matches text · 6. word-count distribution
· 7. proper-noun entity count · 8. `outcome_end` within 5 chars of text end · 9. no-harm cell harm
audit · 10. contamination flag no longer predicts outcome

Two warnings, both expected and benign: the 10 YS2011 rows are short (43–65 words) because that
study uses a 2-cell second-person design; and the 24 harm-vocabulary rows in §4.

### Process note

The previous quality gate was the clause annotator's coverage rate ("94.6% matched, only 4
fallbacks"). That measures whether a *pattern matched*, not whether the *text was correct*, and it
was green throughout. Reading three stories would have exposed the bug immediately. Check 9 of the
validator and the 20-item manual sample (`--sample 20 --seed 0`) are now hard gates for this
reason: an automated metric being green is not evidence the data is right.
