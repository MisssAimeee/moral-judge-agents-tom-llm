# ToMi scoring audit — drop from primary claims

**Verdict: do not report ToMi accuracy as a ToM result.** BigToM (false belief,
`init_belief=0`) carries the argument alone.

## Why we checked

Qwen2.5-14B-Instruct scores **0.985** on BigToM false belief and **0.512** on
ToMi — near ceiling and exact chance on the same model. Most Qwen models sit at
0.48–0.55 on ToMi. That pattern is a scoring / item-mix red flag, not a clean
dissociation.

## Answer-space size

ToMi items are reduced to a two-alternative forced choice over object-container
names extracted with `\b(?:in|to|on) the (\w+)`.

| corpus | n blocks | candidate-set size |
| --- | ---: | --- |
| `fb_all_test.txt` (full) | 5994 | **exactly 2 for every block** |
| scored subset (`--tomi-limit 400`) | 400 | **exactly 2** |

So this is not a silent collapse of a multi-container answer space down to 2AFC —
the extractor never sees more than two object containers. Rooms (`entered the
closet`) are not in the candidate set. Length-normalised logprob argmax over
`[gold, other]` is the intended scoring rule, and the candidate set matches that
rule.

The bug is elsewhere.

## What the scored 400 actually are

Items are sorted with `first_order_*` first, then truncated at 400. That set is:

| coarse type | n | share |
| --- | ---: | ---: |
| `first_order_*_no_tom` (not ToM-critical) | 329 | **82%** |
| `first_order_*_tom` (ToM-critical) | 71 | 18% |

Belief tags on the same 400: true_belief 202 / false_belief 125 /
second_order_false_belief 73. As noted in `36_tom_benchmarks.py`, the trace
belief tag describes story generation, not necessarily the queried agent's
belief state.

**Aggregate `tomi` accuracy is mostly a reading / true-location quiz**, not a
false-belief score.

## Per-question-type breakdown (logprob 2AFC)

| model | ToMi all | no_tom (n=329) | tom-critical (n=71) | BigToM FB |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-14B-Instruct | 0.512 | 0.495 | 0.592 | 0.985 |
| Qwen2.5-7B-Instruct | 0.550 | 0.547 | 0.563 | 0.935 |
| OLMo-2-7B-Instruct | 0.818 | **0.906** | **0.408** | 0.890 |
| gemma-2-9b-it | 0.665 | 0.696 | 0.521 | 0.935 |
| Llama-3.1-8B-Instruct | 0.570 | 0.581 | 0.521 | 0.865 |
| Tulu-3-8B | 0.757 | 0.848 | 0.338 | 0.855 |

OLMo’s high aggregate is easy `no_tom` items; on ToM-critical items it is
below chance. Qwen is near chance on both slices while near ceiling on BigToM
false belief.

## Generative spot-check (n=80: 40 no_tom-TB + 40 tom-critical-FB)

| model | slice | logprob 2AFC | generative match |
| --- | --- | ---: | ---: |
| Qwen2.5-1.5B-Instruct | no_tom TB | 0.525 | 0.741 (13 unparsed) |
| Qwen2.5-1.5B-Instruct | tom-crit FB | 0.600 | 0.429 (12 unparsed) |
| OLMo-2-7B-Instruct | no_tom TB | 0.950 | 1.000 |
| OLMo-2-7B-Instruct | tom-crit FB | 0.475 | 0.526 |

Generative scoring does not rescue a ToM-critical signal for Qwen, and OLMo’s
aggregate lead remains a `no_tom` effect. Free generation also has a high
parse-fail rate for chatty Qwen answers unless the prompt forces a container
name.

## Decision

1. **Drop ToMi from J1 claims, tables used for the argument, and ceiling-gate
   “spread” language.** The gate’s ToMi spread was an artifact of `no_tom`
   variance across families.
2. Keep this audit and the raw `tom_items_*.csv` files for provenance.
3. **BigToM false belief (`init_belief=0`) is the sole ToM measure** for the
   dissociation claim.

Normalization used for the dropped numbers: mean token log-probability of each
option continuation after the chat template, argmax; gold always at index 0.
That procedure is fine for a true 2AFC. It cannot fix an item mix that is 82%
non-ToM.
