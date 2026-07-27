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
accuracy. After repair it predicts at **0.517**, exactly the class base rate. `b_outcome` and the
0.99–1.00 outcome decoding were therefore both confounded with this artefact.

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

**(b) Derive harm polarity from the text instead of assuming item B is harmful.**

```python
HARM_OUTCOME = re.compile(
    r"\b(dies?|died|death|dead|kill(?:s|ed)?|chokes? to death|never wakes up|coma|"
    r"emergency room|hospital|third degree|paralys\w*|drowns?|contract diseases)\b", re.I)
...
a_harm, b_harm = bool(HARM_OUTCOME.search(act_A)), bool(HARM_OUTCOME.search(act_B))
if a_harm and not b_harm:
    fore_A, fore_B = fore_B, fore_A
    act_A, act_B = act_B, act_A
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

`clause_offsets.csv` regenerated against the repaired master. Method distribution over 298 rows:

| method | n |
|---|---|
| `belief_verb` (pattern matched) | 281 |
| `fallback_position` (guessed from sentence position) | 14 |
| `belief_verb+action_eq_outcome` | 1 |

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
