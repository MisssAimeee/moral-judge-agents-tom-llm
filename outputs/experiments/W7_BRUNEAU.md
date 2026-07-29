# W7 (appendix) — Bruneau, Pluta & Saxe (2011): is the harm code domain-selective?

**Limitation, first because it governs every number below.** These stimuli are not an intent × outcome factorial. No condition has a character intending harm that does not occur. PP/EP hold belief fixed and vary the outcome; FBP/FBC vary belief content but belief and outcome move together. W7 therefore tests **outcome/harm selectivity only** and says nothing about intent-weighting. It is support for the claim that the harm representation is rich and structured — which sharpens, but does not establish, the asymmetry the main results rest on.

Stimuli: 144 items, 72 matched pairs, parsed from the published stimulus PDF (`W7_PARSE_REPORT.md`). Folds and bootstrap resamples are over item PAIRS, so the harmful and harmless version of one story never land on opposite sides of a split.

## Within-domain harm decoding vs surface baseline

| model | domain | layer | probe | TF-IDF | gap | pairs |
|---|---|---:|---:|---:|---:|---:|
| allenai/OLMo-2-1124-7B-Instruct | PP/PPC (physical) | 1/33 | 0.604 | 0.604 | +0.000 | 24 |
| allenai/OLMo-2-1124-7B-Instruct | PP/PPC (physical) | 8/33 | 0.958 | 0.604 | +0.354 | 24 |
| allenai/OLMo-2-1124-7B-Instruct | PP/PPC (physical) | 16/33 | 0.938 | 0.604 | +0.333 | 24 |
| allenai/OLMo-2-1124-7B-Instruct | PP/PPC (physical) | 22/33 | 0.958 | 0.604 | +0.354 | 24 |
| allenai/OLMo-2-1124-7B-Instruct | PP/PPC (physical) | 24/33 | 1.000 | 0.604 | +0.396 | 24 |
| allenai/OLMo-2-1124-7B-Instruct | PP/PPC (physical) | 32/33 | 0.979 | 0.604 | +0.375 | 24 |
| allenai/OLMo-2-1124-7B-Instruct | EP/EPC (emotional) | 1/33 | 0.604 | 0.750 | -0.146 | 24 |
| allenai/OLMo-2-1124-7B-Instruct | EP/EPC (emotional) | 8/33 | 0.875 | 0.750 | +0.125 | 24 |
| allenai/OLMo-2-1124-7B-Instruct | EP/EPC (emotional) | 16/33 | 0.938 | 0.750 | +0.188 | 24 |
| allenai/OLMo-2-1124-7B-Instruct | EP/EPC (emotional) | 22/33 | 0.979 | 0.750 | +0.229 | 24 |
| allenai/OLMo-2-1124-7B-Instruct | EP/EPC (emotional) | 24/33 | 0.979 | 0.750 | +0.229 | 24 |
| allenai/OLMo-2-1124-7B-Instruct | EP/EPC (emotional) | 32/33 | 1.000 | 0.750 | +0.250 | 24 |
| allenai/OLMo-2-1124-7B-Instruct | FBP/FBC (false belief) | 1/33 | 0.604 | 0.562 | +0.042 | 24 |
| allenai/OLMo-2-1124-7B-Instruct | FBP/FBC (false belief) | 8/33 | 0.854 | 0.562 | +0.292 | 24 |
| allenai/OLMo-2-1124-7B-Instruct | FBP/FBC (false belief) | 16/33 | 0.938 | 0.562 | +0.375 | 24 |
| allenai/OLMo-2-1124-7B-Instruct | FBP/FBC (false belief) | 22/33 | 0.917 | 0.562 | +0.354 | 24 |
| allenai/OLMo-2-1124-7B-Instruct | FBP/FBC (false belief) | 24/33 | 0.917 | 0.562 | +0.354 | 24 |
| allenai/OLMo-2-1124-7B-Instruct | FBP/FBC (false belief) | 32/33 | 0.958 | 0.562 | +0.396 | 24 |
| Qwen/Qwen2.5-7B-Instruct | PP/PPC (physical) | 1/29 | 0.646 | 0.604 | +0.042 | 24 |
| Qwen/Qwen2.5-7B-Instruct | PP/PPC (physical) | 7/29 | 0.708 | 0.604 | +0.104 | 24 |
| Qwen/Qwen2.5-7B-Instruct | PP/PPC (physical) | 14/29 | 0.979 | 0.604 | +0.375 | 24 |
| Qwen/Qwen2.5-7B-Instruct | PP/PPC (physical) | 20/29 | 0.979 | 0.604 | +0.375 | 24 |
| Qwen/Qwen2.5-7B-Instruct | PP/PPC (physical) | 21/29 | 0.979 | 0.604 | +0.375 | 24 |
| Qwen/Qwen2.5-7B-Instruct | PP/PPC (physical) | 28/29 | 0.958 | 0.604 | +0.354 | 24 |
| Qwen/Qwen2.5-7B-Instruct | EP/EPC (emotional) | 1/29 | 0.667 | 0.750 | -0.083 | 24 |
| Qwen/Qwen2.5-7B-Instruct | EP/EPC (emotional) | 7/29 | 0.771 | 0.750 | +0.021 | 24 |
| Qwen/Qwen2.5-7B-Instruct | EP/EPC (emotional) | 14/29 | 0.958 | 0.750 | +0.208 | 24 |
| Qwen/Qwen2.5-7B-Instruct | EP/EPC (emotional) | 20/29 | 0.938 | 0.750 | +0.188 | 24 |
| Qwen/Qwen2.5-7B-Instruct | EP/EPC (emotional) | 21/29 | 0.896 | 0.750 | +0.146 | 24 |
| Qwen/Qwen2.5-7B-Instruct | EP/EPC (emotional) | 28/29 | 0.896 | 0.750 | +0.146 | 24 |
| Qwen/Qwen2.5-7B-Instruct | FBP/FBC (false belief) | 1/29 | 0.708 | 0.562 | +0.146 | 24 |
| Qwen/Qwen2.5-7B-Instruct | FBP/FBC (false belief) | 7/29 | 0.812 | 0.562 | +0.250 | 24 |
| Qwen/Qwen2.5-7B-Instruct | FBP/FBC (false belief) | 14/29 | 0.917 | 0.562 | +0.354 | 24 |
| Qwen/Qwen2.5-7B-Instruct | FBP/FBC (false belief) | 20/29 | 0.917 | 0.562 | +0.354 | 24 |
| Qwen/Qwen2.5-7B-Instruct | FBP/FBC (false belief) | 21/29 | 0.896 | 0.562 | +0.333 | 24 |
| Qwen/Qwen2.5-7B-Instruct | FBP/FBC (false belief) | 28/29 | 0.875 | 0.562 | +0.312 | 24 |

## Moral-vs-non-moral interaction

Difference in harm-decoding accuracy between a mentalizing domain and the physical domain, bootstrapped over item pairs (95% CI). A CI excluding 0 means the harm code is not equally available across domains. The TF-IDF column carries the same difference for the surface baseline: if the probe difference tracks the lexical difference, the effect is in the wording.

| model | domain | layer | acc (domain) | acc (physical) | Δ probe [95% CI] | Δ TF-IDF |
|---|---|---:|---:|---:|---:|---:|
| allenai/OLMo-2-1124-7B-Instruct | EP/EPC (emotional) | 32 | 1.000 | 0.979 | +0.021 [+0.000, +0.021] | +0.146 |
| allenai/OLMo-2-1124-7B-Instruct | FBP/FBC (false belief) | 32 | 0.958 | 0.979 | -0.021 [-0.083, +0.021] | -0.042 |
| Qwen/Qwen2.5-7B-Instruct | EP/EPC (emotional) | 14 | 0.958 | 0.979 | -0.021 [-0.062, +0.042] | +0.146 |
| Qwen/Qwen2.5-7B-Instruct | FBP/FBC (false belief) | 14 | 0.917 | 0.979 | -0.062 [-0.104, +0.021] | -0.042 |

## Cross-domain transfer (the selectivity test)

Fit the harm contrast on one domain, score it on another, at a single layer chosen by mean accuracy across domains (not per contrast, which would inflate the diagonal). High off-diagonal accuracy means one generic "something bad happened" code; near-chance off-diagonal means domain-specific harm codes. One asymmetry to keep in mind: the diagonal is grouped-CV (fit on ~5/6 of the pairs), the off-diagonal is fit on all 48 items of the source domain and tested on a fully disjoint set. The transfer cells therefore have MORE training data, so an off-diagonal that still falls below the diagonal is evidence for selectivity and not an artefact of sample size; an off-diagonal that matches the diagonal is the weaker comparison.

**Qwen/Qwen2.5-7B-Instruct** (layer 14)

| fit on \ test on | physical | emotional | false_belief |
|---|---:|---:|---:|
| physical | 0.979 _(within)_ | 0.604 | 0.604 |
| emotional | 0.792 | 0.958 _(within)_ | 0.875 |
| false_belief | 0.833 | 0.917 | 0.917 _(within)_ |

Mean within-domain 0.951, mean cross-domain 0.771 (chance 0.500). Transfer is substantial: one largely shared harm code.

**allenai/OLMo-2-1124-7B-Instruct** (layer 32)

| fit on \ test on | physical | emotional | false_belief |
|---|---:|---:|---:|
| physical | 0.979 _(within)_ | 0.604 | 0.562 |
| emotional | 0.875 | 1.000 _(within)_ | 0.958 |
| false_belief | 0.896 | 0.958 | 0.958 _(within)_ |

Mean within-domain 0.979, mean cross-domain 0.809 (chance 0.500). Transfer is substantial: one largely shared harm code.


## Reading

Whatever the transfer matrix shows, it is a statement about the harm/outcome representation only. The main results say the outcome code is what the rating uses and the intent code is not; W7 adds resolution to the first half of that sentence. A reviewer should not be allowed to read it as evidence on the second half, and no figure or caption from this script claims otherwise.

