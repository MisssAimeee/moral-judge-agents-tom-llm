# Polarity audit — all scenario groups

Systematic fore/act / belief×outcome coherence sweep over every `scenario_group` in `dataset/master/moral_2x2_master.csv`.

## Method

- Harm from **last sentence**, using `HARM_WORDS` from `28_validate_master.py` ∪ `HARM_OUTCOME` from `build_dataset.py` ∪ clear outcome-harm terms; strong no-harm phrases (`is fine`, `safely`, …) override. Ambiguous endings do not fail on label alone — structural 2×2 checks still run.
- Full 2×2: accidental↔intentional endings must match; neutral↔attempted endings must match; beliefs (believes/thinks/believing) must form the crossed pattern (innocent pair identical, guilty pair identical, innocent ≠ guilty).
- Partial designs (YS2011 intentional/accidental pairs) skip act/belief 2×2 pattern checks (no shared act_B by design) but still check label↔last-sentence harm and glued-title contamination.
- Reprint groups audited per source; cross-reprint label disagreements fail.

## Summary: **0 fail / 53 groups** (53 pass)

No group failed hard checks. Label↔text harm and within-group act/belief polarity look coherent across the full sweep (including groups beyond CPR).

## Sweep table

| group | n_cells | sources | status | notes |
|---|---:|---|---|---|
| ALARM | 8 | YS2008,YS2009 | PASS | — |
| ASTHMA | 4 | YS2008 | PASS | — |
| Allergy | 2 | YS2011 | PASS | [YS2011] partial design (missing ['attempted', 'neutral']); skip act/belief 2x2 pattern checks |
| BAR | 4 | YS2008 | PASS | — |
| BIKE | 4 | YS2008 | PASS | — |
| BOUNCY BALL | 4 | YS2008 | PASS | — |
| BRIDGE | 4 | YS2008 | PASS | — |
| CAYO | 4 | YS2008 | PASS | — |
| CHAIRLIFT | 4 | YS2008 | PASS | [YS2008] informational: accidental/intentional action spans differ before ending (often foreshadow glued into belief-adjacent text; endings match) |
| COFFEE | 4 | YS2008 | PASS | — |
| CPR | 8 | YS2008,YS2009 | PASS | — |
| Dog | 2 | YS2011 | PASS | [YS2011] partial design (missing ['attempted', 'neutral']); skip act/belief 2x2 pattern checks |
| FRAT | 8 | YS2008,YS2009 | PASS | — |
| FUMIGATION | 4 | YS2008 | PASS | — |
| HAM | 8 | YS2008,YS2009 | PASS | — |
| HARNESS | 4 | YS2008 | PASS | — |
| HUNT | 4 | YS2008 | PASS | [YS2008] informational: accidental/intentional action spans differ before ending (often foreshadow glued into belief-adjacent text; endings match) |
| IGLOO | 8 | YS2008,YS2009 | PASS | — |
| IRON | 8 | YS2008,YS2009 | PASS | — |
| JELLYFISH | 8 | YS2008,YS2009 | PASS | — |
| LAB | 4 | YS2008 | PASS | — |
| LAPTOP | 8 | YS2008,YS2009 | PASS | — |
| LATEX | 4 | YS2008 | PASS | — |
| LOGAN AIRPORT | 4 | YS2008 | PASS | [YS2008] informational: accidental/intentional action spans differ before ending (often foreshadow glued into belief-adjacent text; endings match) |
| MALARIA POND | 8 | YS2008,YS2009 | PASS | — |
| MEATLOAF | 8 | YS2008,YS2009 | PASS | — |
| MOTHER | 4 | YS2008 | PASS | — |
| MOTORBOAT | 4 | YS2008 | PASS | — |
| MUSHROOMS | 4 | YS2008 | PASS | — |
| PARACHUTES | 4 | YS2008 | PASS | — |
| PEANUT ALLERGY | 8 | YS2008,YS2009 | PASS | — |
| POOL | 4 | YS2008 | PASS | — |
| POPCORN | 8 | YS2008,YS2009 | PASS | [YS2008] informational: accidental/intentional action spans differ before ending (often foreshadow glued into belief-adjacent text; endings match); [YS2009] informational: accidental/intentional action spans differ be... |
| PORRIDGE | 8 | YS2008,YS2009 | PASS | — |
| Parent | 2 | YS2011 | PASS | [YS2011] partial design (missing ['attempted', 'neutral']); skip act/belief 2x2 pattern checks |
| Poison | 2 | YS2011 | PASS | [YS2011] partial design (missing ['attempted', 'neutral']); skip act/belief 2x2 pattern checks |
| RABIES | 8 | YS2008,YS2009 | PASS | — |
| RIVER | 4 | YS2008 | PASS | [YS2008] informational: accidental/intentional action spans differ before ending (often foreshadow glued into belief-adjacent text; endings match) |
| SAFETY CORD | 8 | YS2008,YS2009 | PASS | — |
| SAFETY TOWN | 8 | YS2008,YS2009 | PASS | [YS2008] informational: accidental/intentional action spans differ before ending (often foreshadow glued into belief-adjacent text; endings match); [YS2009] informational: accidental/intentional action spans differ be... |
| SEATBELT | 8 | YS2008,YS2009 | PASS | — |
| SESAME | 8 | YS2008,YS2009 | PASS | — |
| SPINACH | 8 | YS2008,YS2009 | PASS | — |
| SPRING BREAK | 4 | YS2008 | PASS | — |
| SUSHI | 8 | YS2008,YS2009 | PASS | [YS2008] informational: accidental/intentional action spans differ before ending (often foreshadow glued into belief-adjacent text; endings match); [YS2009] informational: accidental/intentional action spans differ be... |
| Sibling | 2 | YS2011 | PASS | [YS2011] partial design (missing ['attempted', 'neutral']); skip act/belief 2x2 pattern checks |
| TEENAGERS | 8 | YS2008,YS2009 | PASS | — |
| TRACKS | 4 | YS2008 | PASS | — |
| TREE HOUSE | 8 | YS2008,YS2009 | PASS | — |
| VET | 4 | YS2008 | PASS | — |
| VITAMIN | 4 | YS2008 | PASS | — |
| WET FLOOR | 8 | YS2008,YS2009 | PASS | [YS2008] informational: accidental/intentional action spans differ before ending (often foreshadow glued into belief-adjacent text; endings match); [YS2009] informational: accidental/intentional action spans differ be... |
| ZOO | 8 | YS2008,YS2009 | PASS | [YS2008] informational: accidental/intentional action spans differ before ending (often foreshadow glued into belief-adjacent text; endings match); [YS2009] informational: accidental/intentional action spans differ be... |

*Generated by `code/experiments/30_polarity_audit.py`. This table covers all 53 groups — not only the historically known CPR act-polarity case.*
