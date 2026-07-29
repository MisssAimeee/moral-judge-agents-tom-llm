# W4 run 1 — preserved, superseded (job 19130876, 2026-07-28)

Archived unmodified before the fixes below. Kept because W4 is a load-bearing result and
the pre-fix numbers should be recoverable, not because anything here should be quoted.

## Two defects found while checking whether L4's dip was arithmetic or psychological

**1. The few-shot question named the wrong agent, on `human_verbatim` only.** The L4 block
rendered each example under the question built from the TARGET item, and `human_verbatim`
interpolates the agent name. So a story about Nadia at a climbing gym was followed by "How
permissible was **Grace**'s action?" — Grace being the agent of the item under test. The
six paraphrase templates have no agent slot and were unaffected. This makes every level
that contains the few-shot block (L4, L5) invalid on 1 of the 7 templates, and those levels
are exactly where the anomaly was.

Fixed in `54_w4_prompt_curriculum.py::few_shot_block`, which now builds each example's
question from that example's own text. L4/L5 were rescored.

**2. The bootstrap resampled template × group cells, not scenario groups.** `n_groups`
reads 371 in this archive, which is 7 templates × 53 groups. Resampling those cells treats
one vignette rated under seven templates as seven independent observations, so every
interval here is too narrow. Point estimates are unaffected (the cell means average over
all cells equally). Fixed by resampling the 53 scenario groups and averaging over
templates within a resampled group; the corrected rows were recomputed from the stored cell
means in this archive's `w4_curriculum_cells.csv` — no rescoring was needed for L1–L3.

## What the arithmetic check found

The suspicion that L4's ratings were inverted — plausible, since Young 2007 is 1–4
permissibility and the factorial templates are 1–7 blame, and this project has had both a
CPR polarity inversion and a permissibility-direction reversal before — did **not** hold.
The YS2008 anchor is phrased "1 (completely permissible) to 3 (completely impermissible)",
ascending in condemnation like every other template, so `direction="blame"` in
`TEMPLATE_META` is correct and the linear map needs no reversal. The few-shot labels encode
attempted > accidental on every template × scale, implied contrast +0.500 to +0.667 against
an adult reference of +0.666. That check is now a hard gate that raises before scoring
(`check_fewshot_polarity`) and its table is written into `W4_PROMPT_LEVELS.md`.

So the arithmetic explanation is ruled out for the label VALUES, and a different mechanical
explanation — defect 1 above — was found in the same block.

**And the L4 dip survives the fix.** OLMo-2-7B-Instruct rescored to −0.7061 at L4 against
−0.7065 in this archive, i.e. repairing the incoherent question moved it by 0.0004. The dip
is therefore not an artefact of either the labels or the agent-name defect; whatever the
labelled examples do to these models, they do it on a correctly formed prompt. The report
treats the mechanism as unresolved and points at the L7 few-shot-alone ablation cell rather
than asserting anchor imitation.
