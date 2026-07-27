# Human anchor comparison — methods provenance

METHODS PRE-SPEC (2026-07-10): methods_child_measure.md chose Naughty/wrongness, presented-first as primary — sixteen days before this comparison. human_reference.csv used naughty+punishable text inconsistent with that spec. Anchor decision must trace to that prior methods choice, not to 9/24 vs 24/24. Naughty + Punish ladders both remain as a permanent robustness table; do not choose the anchor here.

## Robustness across measures

SCOPE: computed on open-weight models only (post-repair rescore). Closed-API models have not been rescored since the stimulus repair; their ladders are emitted separately as *_all and marked PENDING RESCORE — contaminated-era.

ROBUSTNESS ACROSS MEASURES (not an anchor choice). The claim 'models fall at or below the youngest measured band' holds under BOTH digitized child ladders — naughtiness (youngest +0.24) and punishment (youngest +0.09) — and fails only under human_reference.csv, which mixed naughty+punishable contrary to the method pre-specified on 2026-07-10. Surviving two independently digitized child measures is a robustness result; it does not select a primary anchor, which remains the user's decision.

THEORETICAL CHECK. The punishment ladder is monotonic in age but flatter than naughtiness (+0.09/+0.12/+0.19 vs +0.24/+0.50/+0.63) — exactly Cushman et al. (2013)'s two-process prediction that intent constrains judgments of wrongness before judgments of deserved punishment. Two independent digitizations reproducing the predicted pattern is evidence the digitization is sound.

Per-ladder outcome (open-weight only):
  - text-reported (human_reference.csv) (youngest band -0.14): does not hold for 8/16 non-degenerate models
  - digitized Naughty presented-first (youngest band +0.24): holds for 16/16 non-degenerate models
  - Punish presented-first (secondary; construct-matched to punish_* prompts) (youngest band +0.09): holds for 16/16 non-degenerate models

Same ladders WITH contaminated-era closed-API models included (marked, not for headline use):
  - text-reported (human_reference.csv) (youngest band -0.14): does not hold for 11/23 non-degenerate models
  - digitized Naughty presented-first (youngest band +0.24): holds for 23/23 non-degenerate models
  - Punish presented-first (secondary; construct-matched to punish_* prompts) (youngest band +0.09): does not hold for 22/23 non-degenerate models
