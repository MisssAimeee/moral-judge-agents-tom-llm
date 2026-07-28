# C2 — does splitting belief_last by source rescue the pre-outcome reading?

**No. Sentence order has a small, consistent effect in the predicted direction, but
the outcome remains strongly decodable at `belief_last` in BOTH sources, so the
pre-outcome reading cannot be rescued by the split. The neutral caption on the gap
figure stays permanently.**

## The test

The gap figure originally described `belief_last` and `action_last` as positions cut
before the harm is stated. The two stimulus sources order their sentences differently:

- **YS2008** (192 items): the outcome-determining sentence comes
  BEFORE the belief clause.
- **YS2009** (96 items): the belief clause comes first.

So the pre-outcome reading makes a discriminating prediction: outcome decoding at
`belief_last` should be high for YS2008, where the outcome genuinely has been stated,
and at chance for YS2009, where it has not. Probes were refit per source
(02_probe.py --source, job 19025559) across 8 models.

## Result

| measure | YS2008 | YS2009 | difference | sign test |
|---|---|---|---|---|
| outcome decoding at `belief_last` | 0.844 | 0.798 | +0.046 | p = 0.00781 (8/8 models positive) |
| intent decoding at `belief_last` | 0.962 | 0.952 | +0.010 | p = 0.727 (5/8 models positive) |

Accuracies are the mean of each model's top 3 layers rather than its
single best layer, so the numbers are less inflated by selecting over ~33 layers.
The per-layer peaks are in the CSV and tell the same story.

**The prediction fails, and it fails on magnitude rather than on direction.**

Sentence order does matter, slightly and consistently: outcome decoding is higher
for YS2008 than YS2009 by +0.046, in all 8 of 8 models (sign test p = 0.00781). That much of the original
reasoning survives, and it should not be described as a flat null.

But the effect is roughly an order of magnitude too small to carry the claim.
YS2009 outcome decoding is
0.798 against chance 0.500 — above chance in 8 of 8
models, sign test p = 0.00781 — in items where the outcome-determining sentence
has NOT yet appeared. The prediction was chance-level decoding there; what appears
instead is +0.298 above chance, against a source difference
of only +0.046. Ordering shifts outcome
decodability at the margin; it does not create a position where the outcome is
unavailable.

## What this means

`belief_last` is not a pre-outcome position in either source, so outcome decoding
there is not evidence that the model is predicting an outcome it has not been told.
The most likely reason is that the belief clause itself carries the
outcome-relevant fact: a clause like "she believed the powder was sugar" versus
"she believed the powder was poison" differs lexically in exactly the way that
distinguishes the harm conditions, regardless of where the outcome sentence sits.
Cutting the text before the outcome sentence does not cut it before the
outcome-relevant information.

Three consequences:

1. The neutral caption on the gap figure stays permanently. It describes clause
   POSITION and makes no claim about what has been stated, which is the only thing
   the data support.
2. The source split is a reported result, and a real if small one: the ordering
   effect is consistent across all 8 models. It just does not do the work it was
   proposed to do, because both sources leave the outcome strongly decodable.
3. Any argument that depends on `belief_last` being pre-outcome has to be dropped.
   Isolating a genuinely pre-outcome position would require cutting on the
   outcome-relevant CONTENT of the belief clause, not on sentence order — and for
   these stimuli that may not be possible at all, since the belief content is what
   defines the condition.

Note that intent decoding is high and essentially identical across sources
(0.962 vs 0.952), so this is not a
story about one source being harder to probe.
