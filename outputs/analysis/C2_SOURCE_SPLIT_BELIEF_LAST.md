# C2 — Belief-last probes split by stimulus source (span-matched)

Job `19025559` produced per-source probes. This revision subtracts the
**span-matched** TF-IDF baseline (`text[:belief_end]`) rather than the
full-story baseline. Absolute probe accuracies are unchanged; only the gap
interpretation can move.

## Manual annotation audit (five YS2009 stories, verbatim)

Before interpreting the reopened gap below, five YS2009 stories were read
manually with `belief_start`/`belief_end`/`action_start`/`action_end`/
`outcome_start`/`outcome_end` marked. **Finding: offsets are structurally
correct in all five — `belief_end` always precedes `outcome_start`, so the
tagged outcome sentence is genuinely outside `belief_last`.** But the
**setup sentences before `belief_start`** (included in `text[:belief_end]`,
i.e. inside the belief_last span) already state the situation's
ground-truth hazard, well before the belief clause and far before the
tagged outcome sentence:

| story | scenario | hazard stated in setup (before belief_start, inside belief_last) | outcome sentence (after belief_end) |
|---|---|---|---|
| `YS2009-YS2009_01-accidental` | POPCORN | “Deadly carbon monoxide is escaping from the furnace under the kitchen. The kitchen is now dangerous.” | “His cousin goes into the kitchen and, while fidgeting with the device, passes out from carbon monoxide inhalation.” |
| `YS2009-YS2009_02-accidental` | SUSHI | “Some batches of fish, usually the tuna, have parasites.” | “One of his colleagues orders the tuna and ends up getting a nasty strain of parasites.” |
| `YS2009-YS2009_03-accidental` | ASTHMA/SMOKE HOUSE | “The boy has asthma, which makes his lungs close up around smoke, so he will not be able to breathe if he goes into the Safety Town smoking house.” | “The boy has a severe asthma attack inside and starts having a seizure.” |
| `YS2009-YS2009_04-accidental` | MALARIA POND | “Malarial mosquitoes actually live in the pond. A single bite is enough to create an infection, so the pond is unsafe to wade in.” | “His friend is bitten by several mosquitoes and contracts malaria.” |
| `YS2009-YS2009_05-accidental` | DETERGENT PORRIDGE | “The porridge on the counter has some dishwashing detergent in it. Eva’s nephew spilled a large amount of detergent into the container while he was playing.” | “Soon after, she starts throwing up again and again and ends up in the emergency room.” |

This is not an annotation error (the YS2008 premise error does not repeat
here) and it is not evidence that the model anticipates content the text
never states. The hazard IS stated, inside `belief_last`, just narratively
separate from the tagged "outcome" sentence that restates the consequence.
The reason TF-IDF (`text[:belief_end]`, leave-scenario-out CV) reads near
chance despite this is that each scenario's hazard vocabulary is unique
("carbon monoxide" / "parasites" / "asthma" / "malaria" / "detergent") and
does not recur across the held-out scenarios a bag-of-words model is scored
on; a neural probe generalizes the shared "hazard/harm" semantics that
bag-of-words features cannot. **Conclusion: downgrade this from a headline
"model represents outcome before the text states it" claim to supporting
evidence that belief_last representations carry semantically-generalized
content that a lexical baseline cannot match — not evidence of anticipating
unstated information.**

## Span-matched TF-IDF at belief_last

| source | target | TF-IDF (belief span) | chance |
| --- | --- | ---: | ---: |
| YS2008 | intent | 0.609 | ~0.50 |
| YS2008 | outcome | 0.602 | ~0.50 |
| YS2009 | intent | 0.565 | ~0.50 |
| YS2009 | outcome | 0.580 | ~0.50 |
| all | intent | 0.591 | ~0.50 |
| all | outcome | 0.581 | ~0.50 |

## Outcome decoding at belief_last

| model | YS2008 probe | YS2009 probe | YS2008 gap | YS2009 gap |
| --- | ---: | ---: | ---: | ---: |
| OLMo-2-1124-7B | 0.901 | 0.882 | +0.299 | +0.302 |
| OLMo-2-1124-7B-Instruct | 0.927 | 0.828 | +0.326 | +0.248 |
| Qwen2.5-0.5B | 0.801 | 0.752 | +0.199 | +0.172 |
| Qwen2.5-0.5B-Instruct | 0.788 | 0.785 | +0.187 | +0.205 |
| Qwen2.5-1.5B | 0.823 | 0.820 | +0.221 | +0.240 |
| Qwen2.5-1.5B-Instruct | 0.842 | 0.835 | +0.240 | +0.255 |
| Qwen2.5-7B | 0.887 | 0.825 | +0.285 | +0.245 |
| Qwen2.5-7B-Instruct | 0.907 | 0.830 | +0.305 | +0.250 |

## Intent decoding at belief_last

| model | YS2008 probe | YS2009 probe | YS2008 gap | YS2009 gap |
| --- | ---: | ---: | ---: | ---: |
| OLMo-2-1124-7B | 0.979 | 0.980 | +0.371 | +0.415 |
| OLMo-2-1124-7B-Instruct | 0.984 | 0.988 | +0.376 | +0.423 |
| Qwen2.5-0.5B | 0.939 | 0.932 | +0.330 | +0.368 |
| Qwen2.5-0.5B-Instruct | 0.934 | 0.905 | +0.325 | +0.340 |
| Qwen2.5-1.5B | 0.964 | 0.960 | +0.356 | +0.395 |
| Qwen2.5-1.5B-Instruct | 0.965 | 0.960 | +0.356 | +0.395 |
| Qwen2.5-7B | 0.980 | 1.000 | +0.371 | +0.435 |
| Qwen2.5-7B-Instruct | 0.980 | 0.978 | +0.371 | +0.413 |

## Verdict

Span-matched outcome TF-IDF on YS2009 at belief_last is **0.580**
(near chance), while probes average **0.820** (gap ≈ +0.240).
The manual annotation audit above resolves this: the gap reflects setup-hazard
content that TF-IDF cannot generalize across scenarios, not outcome
anticipation and not an annotation error. **This is downgraded from headline
to supporting evidence** — belief_last representations carry more
semantically-generalized content than a lexical baseline, which is a claim
about representational richness, not about reading unstated information.
Do not use "model represents outcome before the text states it" language
for this result anywhere (figure caption, readout, or writeup).

**Status: DOWNGRADED_SUPPORTING (audited)**

Artifacts: `gap_over_surface_span_matched.csv`,
`gap_over_surface_dissociation_span_matched.png`,
`surface_baseline.csv` (rows with `span=belief_last|action_last`).
