# Appendix: data provenance and the stimulus integrity audit

Repository at `85075ea`.

This appendix documents a stimulus defect that invalidated an earlier round of results, the repair, and the audit that caught a second independent defect while checking the first. It is written as a contribution because that is what it is: the artefact is archived, the detector is a script, the repair is in the builder rather than a post-hoc filter, and every number below is reproducible from the two files named here. Published moral-ToM stimulus sets are parsed from PDF appendices by many groups; the failure mode described in §1 is a property of those appendices, not of this project, and it is silent — the automated quality gate that was in place stayed green throughout.

## 0. Artefacts

| artefact | path | status |
|---|---|---|
| Pre-repair master (archived, never regenerated) | `dataset/master/_prerepair_backup/moral_2x2_master_CONTAMINATED_20260619.csv` | present, md5 `5dd904a7609628553319da4acab02f25` — **matches** the recorded checksum |
| Repaired master | `dataset/master/moral_2x2_master.csv` | current, last commit a875adf 2026-07-27 |
| Repair record | `dataset/master/CONTAMINATION_REPAIR.md` | 5205c92 2026-07-27 |
| Non-circular label audit | `outputs/LABEL_AUDIT_MANUAL.md` | bf4837b 2026-07-26 |
| Validation gate (10 checks, exits non-zero) | `code/experiments/28_validate_master.py` | 5205c92 2026-07-27 |
| Contaminated-era outputs, quarantined | `outputs/_contaminated_20260726/`, `outputs/figures_final/_pending_rescore/` | retained, marked, excluded from figures |

## 1. Defect 1 — trailing contamination, aligned with the outcome factor

The factorial parser merged wrapped PDF lines by appending every following line that was not itself an item marker. In the source appendices the rating prompt, the next scenario's ALLCAPS tag and the next scenario's background all follow the last item with no blank line, so all three were glued onto item 6 — which is used by exactly the `accidental` and `intentional` cells. Both are harm cells, so the contamination was nearly collinear with the outcome factor: 96/154 (62.3%) of harm cells contaminated, 0/144 of no-harm cells.

A parameter-free binary flag ("does this story have an unrelated tail glued on") predicted `outcome_label` at **0.966** accuracy before repair and **0.517** after, the latter being exactly the majority-class rate 154/298 and not 0.5 by coincidence. The 0.99–1.00 outcome decoding and the size of `b_outcome` in the pre-repair results were both confounded with this artefact.

Two things this number does not show, stated because it would be easy to overclaim: with zero flags the detector is a constant predictor, so the collapse to 0.517 shows only that the contamination signal is gone, not that harm and no-harm cells are surface-matched. Residual surface predictability of outcome on the repaired master is **0.755** (word 1–2gram TF-IDF, `outputs/probe/surface_baseline.csv`). Both numbers belong in any statement about what the repair achieved.

Three different counters appear in the record and are not interchangeable: **144** regex-detector hits (the basis of the 0.966 figure), **96/154** harm cells in the per-condition table, and **99** visible trailing artefacts tallied on the archived pre-repair CSV (accidental 48, intentional 49, attempted 2). `CONTAMINATION_REPAIR.md` §1.2 keeps them separate deliberately.

## 2. Defect 2 — the following scenario lost its name and background

The same bug had a second consequence that truncation-based cleaning cannot fix: the consumed lines never reached the next scenario. 33 of 48 YS2008 scenarios had no background at all and fell back to a generic id; `YS2008_02` began mid-narrative with no lab, no switch and no protagonist introduced. This is why the repair is three rule changes in `code/build_dataset.py` and the CSV is regenerated — **no row was hand-edited**. Effect: 260 of 298 rows changed, median word count 89 → 100.5, YS2008 scenarios with a recovered name 15/48 → 48/48.

## 3. The label error was real and was not contamination

`YS2008-CPR` and its YS2009 reprint list the harmful item first, against the appendix convention. Taking the convention on faith had inverted `outcome_label` in all four cells of both scenarios: the `no_harm` cells ended "The customer chokes to death at the table". This was fixed by deriving act polarity from the text rather than by relabelling rows, so the fix is auditable and applies to any future scenario with the same inversion.

The first audit of this was circular — it compared each final sentence to `outcome_label` using the same harm-keyword rule that had assigned the label. `outputs/LABEL_AUDIT_MANUAL.md` is the replacement: by-eye adjudication of all 8 corrected cells plus a seeded random sample (seed 42, quota per condition), each row checked against the 2×2 condition definitions on world / belief / action / outcome. 24 of 144 no-harm rows do contain harm vocabulary, and in every case it sits in the belief clause — which is the design of the `attempted` condition, not an error.

## 4. A second, independent defect found while auditing the first

All 24 YS2009 scenarios are word-for-word reprints of YS2008 scenarios under different ids. Since every probe used `GroupKFold(groups=scenario_id)`, a vignette could sit in train while its identical reprint sat in test — train/test leakage that inflated every reported CV accuracy, and a second plausible contributor to the 0.99–1.00 outcome-decoding ceiling alongside the contamination. The fix is a `scenario_group` column merging reprints, collapsing 77 ids to **53 CV groups**; the duplicated rows are retained (both versions were run behaviourally and the wordings differ slightly) because grouping is sufficient to stop the leakage. Effective n = 53 is now the resolution of every interval in the paper (limitation 7).

## 5. What replaced the quality gate that failed

The gate in place when the bug survived was the clause annotator's coverage rate ("94.6% matched, only 4 fallbacks"). That measures whether a pattern matched, not whether the text was correct, and it was green the whole time. Reading three stories would have exposed the defect immediately. Two hard gates replaced it: `28_validate_master.py`, ten checks including "the contamination flag no longer predicts outcome", exiting non-zero so it can gate an sbatch chain; and a 20-item manual read (`--sample 20 --seed 0`, transcript in `outputs/MANUAL_SAMPLE_20.txt`). The generalisable lesson is the one in the repair note: an automated metric being green is not evidence that the data is right.

## 6. What was thrown away

Every behavioural and representational result produced before 2026-07-26 used the contaminated master and was regenerated. Contaminated-era outputs are retained under `outputs/_contaminated_20260726/` rather than deleted, so any pre-repair number that appears in an old note can be traced. Closed-model rows that have not yet been rescored are marked `PENDING RESCORE — contaminated-era` in their CSVs and their figures are quarantined in `outputs/figures_final/_pending_rescore/` under a `STALE_` prefix.

