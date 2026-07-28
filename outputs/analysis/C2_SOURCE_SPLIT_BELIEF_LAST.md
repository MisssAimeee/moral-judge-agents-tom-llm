# C2 — Belief-last probes split by stimulus source (span-matched)

Job `19025559` produced per-source probes. This revision subtracts the
**span-matched** TF-IDF baseline (`text[:belief_end]`) rather than the
full-story baseline. Absolute probe accuracies are unchanged; only the gap
interpretation can move.

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
The absolute probe accuracy is therefore **not** explained by surface lexis
available at the cut. Two readings remain open: (1) the model represents
outcome before the text states it, or (2) the YS2009 clause annotation is
wrong. The neutral caption on the gap figure is **withdrawn** pending
annotation audit; the pre-outcome reading is again a live hypothesis for
YS2009 items.

**Status: REOPENED**

Artifacts: `gap_over_surface_span_matched.csv`,
`gap_over_surface_dissociation_span_matched.png`,
`surface_baseline.csv` (rows with `span=belief_last|action_last`).
