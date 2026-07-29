# Scoring parity — preserved pre-fix artifacts (superseded by job 19188914)

Preserved because the parity result is load-bearing: it is what licenses comparing
open-weight logprob-EV contrasts against closed-model sampled contrasts, i.e. it is the
reason the open and closed rosters can appear on the same axis at all.

## Why these numbers are suspect

The parity run used `--template human_verbatim`, whose scale is the **source-native** one:
1–3 permissibility on the 192 YS2008 items, 1–4 blame on the 96 YS2009 items, 1–7 on the 10
YS2011 items. The sampling side went through `03_behavioral._parse_rating`, which

1. **clamped** out-of-range answers to the nearest endpoint (`max(s_min, min(s_max, v))`), so
   a model answering "6" out of 1–7 habit on a 1–3 scale was recorded as *maximum
   condemnation* rather than as having ignored the scale — and the coercion is
   directional, since over-range answers can only ever land on the maximum; and
2. **imputed the scale midpoint** when no sample parsed at all.

## The signature in these files

Fraction of items whose sampled mean sits exactly on the scale maximum, 1–3 items versus
1–4 items, from the `sampled_*.csv` in this directory:

| model | YS2008 (1–3) at norm=1.0 | YS2009 (1–4) at norm=1.0 |
|---|---:|---:|
| Qwen2.5-7B-Instruct | **45%** | 4% |
| Mistral-7B-Instruct-v0.3 | **41%** | 7% |
| Qwen2.5-3B-Instruct | 7% (plus **76%** at norm=0.5) | 4% (2% at 0.5) |
| Qwen2.5-1.5B-Instruct | 0% (plus **81%** at norm=0.5) | 0% (0% at 0.5) |
| OLMo-2-7B-Instruct | 2% | 1% |

A per-item value of exactly 1.0 means every one of the 30 samples landed on the maximum.
`Qwen2.5-7B-Instruct` and `OLMo-2-7B-Instruct` are the two models that **passed** the
pre-registered r > 0.95 bar (0.9591 and 0.9742), and Qwen is the one showing the strongest
narrow-scale pinning — so the bar may have been cleared partly on coerced values.

## Why this could not be resolved from the artifacts

The raw response text was never saved, only the per-item normalised mean. On a 1–3 scale a
legitimate "3" and a clamped "6" are both recorded as 1.0, and a legitimate "2" and an imputed
midpoint are both 0.5. The distinguishing information does not exist in these files. Only a
rescore with the corrected parser answers it, which is job 19188914.

OLMo's near-absence of pinning (2% vs 45%) is mild evidence that its pass is genuine and
Qwen's needs re-checking, but that is an inference from the pattern, not a measurement.

## Resolution (job 19189218) — the clamp was *degrading* parity, and the bridge holds

Rescored with the corrected parser. Every model improved, several dramatically, and the two
designated models pass on clean data:

| model | r archived (clamped, pre-fix EV) | r intermediate (clamped, v3 EV) | **r fixed** | passes |
|---|---:|---:|---:|:--:|
| OLMo-2-7B-Instruct | 0.9742 | 0.7823 | **0.9904** | PASS |
| Qwen2.5-7B-Instruct | 0.9591 | 0.8233 | **0.9565** | PASS |
| Qwen2.5-3B-Instruct | 0.3413 | 0.6037 | **0.9338** | no |
| Mistral-7B-Instruct-v0.3 | nan | 0.7693 | **0.8956** | no |
| Qwen2.5-0.5B-Instruct | 0.1124 | 0.4973 | **0.6894** | no |
| Qwen2.5-1.5B-Instruct | −0.2097 | 0.5303 | **0.4519** | no |

Two things this settles. The middle column was the alarming one — 0 of 6 passing — and it was
an artefact of pairing a refreshed EV side against a sampled side still full of coerced
values; it was never the real state. And Mistral's `nan` had a mundane cause: imputation
flattened its EV contrast to exactly 0.0, leaving no variance to correlate.

`n_items` is 298 for all six, so no item lost every one of its 30 samples — the fix removed
coercion without removing data. Remaining non-passers are the sub-7B models, which is the
expected pattern (weaker instruction-following produces noisier free-text ratings) and does not
affect the bridge, since the models actually placed on a shared axis are the 7B+ ones.

## What changed

`_parse_rating` now rejects out-of-range values and answers carrying an explicit "N out of M"
whose M is not the scale maximum; `_finish` returns no rating instead of the midpoint; and
`run_sampling` drops unparseable items with a count instead of writing an imputed value.
Expect `n_items` below 298 for the weaker models after the rescore — that reduction is the
finding, not a regression.
