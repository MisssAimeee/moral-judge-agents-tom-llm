# RSA / CKA — Phase 2 results

**Run 2026-07-26.** Tasks R1–R7. Script: `code/experiments/24_rsa_cka.py`.
8 models, 298 stimuli, mean pooling, 1000 permutations. Job 18953596, elapsed 2m17s.

RDMs are 298 × 298 regardless of hidden size (896 for Qwen-0.5B, 4096 for OLMo-7B). That
dimensional invariance is why RSA can compare architectures whose activations cannot be
compared directly.

---

## Headline

**Intent is decodable but does not organise the representational geometry; outcome does both.**

A trained linear probe recovers intent at up to 0.936. Yet when we ask whether the *geometry*
of the representational space is arranged by intent — with the scenario RDM partialled out to
remove shared background text — the answer is essentially no (partial ρ = 0.00–0.02). Outcome,
by contrast, organises the space robustly (partial ρ = +0.14 to +0.30, every model p = 0.001).

This is not a contradiction. A probe can find a low-variance direction that separates classes
perfectly while that direction accounts for almost none of the variance in the geometry. RSA is
the stricter test precisely because no classifier is trained. The honest statement is:

> Intent information is **present and linearly recoverable**, but it occupies a negligible
> share of representational variance. Outcome information is both recoverable **and
> geometrically dominant.**

That asymmetry is a representational counterpart to the behavioural outcome bias, and it is a
sharper claim than either measure alone.

## R4 — Hypothesis RDMs at the peak-intent layer

Partial Spearman against the intent and outcome hypothesis RDMs, scenario RDM partialled out.
`intent_org_ratio` = |partial ρ_intent| / (|partial ρ_intent| + |partial ρ_outcome|), the
representational analogue of the behavioural b_intent / b_outcome split.

| Model | partial ρ intent | partial ρ outcome | intent share | p (intent) | p (outcome) |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | +0.003 | +0.276 | 0.9% | 0.884 | 0.001 |
| Qwen2.5-0.5B-Instruct | +0.001 | +0.301 | 0.2% | 0.772 | 0.001 |
| Qwen2.5-1.5B | +0.001 | +0.297 | 0.5% | 0.834 | 0.001 |
| Qwen2.5-1.5B-Instruct | +0.001 | +0.288 | 0.3% | 0.791 | 0.001 |
| Qwen2.5-7B | +0.005 | +0.193 | 2.4% | 0.983 | 0.001 |
| Qwen2.5-7B-Instruct | +0.005 | +0.226 | 2.3% | 0.998 | 0.001 |
| OLMo-2-1124-7B | +0.016 | +0.136 | **10.8%** | **0.001** | 0.001 |
| OLMo-2-1124-7B-Instruct | +0.023 | +0.155 | **13.1%** | **0.001** | 0.001 |

**Only the OLMo-2 models show statistically reliable intent organisation**, and even there it
is an order of magnitude weaker than outcome. Across the Qwen ladder intent organisation is
indistinguishable from zero at every scale, including 7B.

This tracks the probe results: OLMo-2 also had the highest intent decoding accuracy (0.929 base,
0.936 instruct). The two independent measures agree that OLMo-2 represents intent more strongly
than Qwen does at matched size — which is worth flagging, because OLMo-2 is the family with
fully published intermediate checkpoints and is therefore the one where the mechanism can
actually be traced.

## R7 — Permutation nulls

1000 permutations, stimulus identity shuffled within scenario. All outcome correlations return
p = 0.001 (the floor). Intent returns p = 0.001 for both OLMo models and p = 0.77–0.998 for all
six Qwen models — i.e. the Qwen intent geometry is *exactly* what you would expect by chance.

## R3 — The convergence test

Does representational similarity track behavioural similarity across model pairs?

- Spearman r = **−0.160**, 95% bootstrap CI **[−0.525, +0.237]**, n = 28 pairs.
- The sign is in the predicted direction (more similar geometry ↔ smaller behavioural
  difference) but the CI comfortably spans zero.

**Verdict: null — "same answer, different route."** Per the plan this is a publishable outcome
and arguably the more interesting one: models converge behaviourally without converging
representationally.

Two honest limits on how hard this can be pushed. First, n = 28 pairs drawn from only 8 models,
and those 8 span just two families, so the CI is wide and the effective sample is smaller than
28 suggests. Second, six of the eight are Qwen variants that are near-identical to each other
(see R5), which compresses the range of the predictor. This should be reported as *"no evidence
for representational convergence in this sample"* rather than as positive evidence of
dissociation. Widening the model set is the fix, and it is cheap — extraction ran in well under
an hour for 8 models.

## R5 — Base vs instruct geometry

| Family | RSA (Spearman) | CKA (linear) | CKA (RBF) |
|---|---|---|---|
| Qwen2.5-0.5B | 0.933 | 0.995 | 0.964 |
| Qwen2.5-1.5B | 0.983 | 1.000 | 0.997 |
| Qwen2.5-7B | 0.985 | 0.997 | 0.993 |
| OLMo-2-1124-7B | 0.941 | 0.957 | 0.986 |

**Instruction tuning barely moves the geometry.** Base and instruct sit at 0.93–0.99 similarity
on both metrics, while the same models differ substantially in behaviour — the checkpoint
dissection found instruction tuning reliably *worsens* outcome bias.

Combined with the probe result that tuning leaves intent decodability roughly unchanged
(Qwen-7B 0.889 → 0.896; OLMo 0.929 → 0.936), this supports a specific and testable claim:

> **Instruction tuning changes the read-out, not the representation.** The information needed
> for intent-based judgment is present before tuning and remains present after; what tuning
> alters is how that information is mapped to an output.

This is a strong standalone figure and it is the natural bridge to Phase 7 (causal steering):
if the representation is intact and only the read-out is biased, steering along the intent
direction should be able to recover intent-based judgment. That is now a motivated prediction
rather than a speculative one.

OLMo-2 is the least similar pair on linear CKA (0.957), consistent with it also being the only
family showing reliable intent organisation.

## R6 — CKA as a second metric

Linear and RBF CKA are reported alongside RSA for every pair
(`model_similarity.csv`, 28 pairs × 2 layer-matching modes). The three metrics agree on the
base-vs-instruct conclusion, which is the point: NEXT_PHASE_PLAN §4 cites a scale-confound
critique of single-metric similarity claims, and the conclusion here does not depend on the
metric chosen.

## Artefacts

| File | Contents |
|---|---|
| `rdm_<model>_L<layer>_mean.npy` | cached 298×298 RDMs, 6 layers per model |
| `hypothesis_rdm.csv` | R4, all cached layers |
| `rsa_permutation_null.csv` | R7 |
| `model_similarity.csv` | R2/R6, peak and matched-depth 0.75 |
| `model_similarity_heatmap_rsa_spearman.png`, `..._cka_linear.png` | R2 heatmaps |
| `base_vs_instruct_geometry.csv` | R5 |
| `convergence_pairs.csv`, `convergence_test.json` | R3 |

## What would strengthen this

1. **More families.** Eight models from two families is the binding constraint on R3. Gemma-2-9B,
   Mistral-7B and the Tülu-3/Zephyr checkpoints are already scored behaviourally; extracting
   their activations is a sub-hour GPU job and would roughly double the pair count while adding
   genuine architectural diversity.
2. **Clause-position RDMs (Phase 3).** If intent organisation is genuinely near zero at the end
   of the story but non-zero at the belief clause, that would explain both this null and the
   `intent_harm` collapse in C3 as the same phenomenon: intent is represented when read and then
   overwritten by outcome.

---

# Clean rerun (supersedes the numbers above)

The stimuli used above were contaminated: 144 of 298 stories carried the rating prompt plus
the opening of the next scenario, almost perfectly confounded with the outcome factor. RSA was
rerun on cleaned stimuli (`outputs/rsa_clean`, job 18956713, 2m07s).

## What changed

**Outcome's geometric dominance was substantially an artefact.** Partial ρ for outcome falls
from +0.14…+0.30 to **+0.034…+0.128** — roughly a threefold reduction. The appended fragments
were making harm stories geometrically distinct as a block.

| Model | partial ρ intent | partial ρ outcome | intent share | p (intent) |
|---|---|---|---|---|
| Qwen2.5-0.5B | +0.005 | +0.043 | 10.4% | 1.000 |
| Qwen2.5-0.5B-Instruct | +0.005 | +0.040 | 10.8% | 1.000 |
| Qwen2.5-1.5B | +0.005 | +0.045 | 10.2% | 1.000 |
| Qwen2.5-1.5B-Instruct | +0.005 | +0.043 | 10.9% | 1.000 |
| Qwen2.5-7B | +0.007 | +0.055 | 11.3% | 1.000 |
| Qwen2.5-7B-Instruct | +0.008 | +0.058 | 11.6% | 1.000 |
| OLMo-2-1124-7B | +0.016 | +0.086 | **16.0%** | **0.001** |
| OLMo-2-1124-7B-Instruct | +0.030 | +0.128 | **19.2%** | **0.001** |

## What survived

**The core dissociation holds, and holds more cleanly.** Intent remains linearly decodable at
0.78–0.93 while contributing only 10–19% of the partialled representational organisation, and
it remains statistically null for every Qwen model (p = 1.000) while being reliable for both
OLMo-2 models (p = 0.001). The gap between "decodable by a trained probe" and "organises the
geometry" is the finding, and cleaning did not touch it.

OLMo-2 is now even more clearly the outlier: it has both the strongest intent decoding and the
only reliable intent geometry, and its intent share rose to 16–19% once the outcome artefact
was removed. That it is also the family with fully published intermediate checkpoints makes it
the obvious target for the mechanistic work.

**R3 convergence: still null**, and slightly stronger in the predicted direction —
r = **−0.249**, 95% CI **[−0.574, +0.175]**, n = 28 pairs. The CI still spans zero. Verdict is
unchanged: no evidence for representational convergence in this sample, with the same power
caveat (8 models, 2 families, six of them near-identical Qwen variants).

**R5 base vs instruct: unchanged conclusion.**

| Family | RSA | CKA linear | CKA RBF |
|---|---|---|---|
| Qwen2.5-0.5B | 0.989 | 0.998 | 0.995 |
| Qwen2.5-1.5B | 0.979 | 1.000 | 0.998 |
| Qwen2.5-7B | 0.989 | 0.994 | 0.988 |
| OLMo-2-1124-7B | 0.866 | 0.955 | 0.988 |

Instruction tuning still barely moves the geometry (0.87–0.99) while behaviour changes
substantially, so **"tuning changes the read-out, not the representation"** stands on clean
data and across all three metrics. OLMo-2 is again the least similar pair (RSA 0.866, down from
0.941 on contaminated data), i.e. OLMo's tuning does move its geometry more than Qwen's does —
worth following up, since OLMo is also the family with published stage checkpoints.
