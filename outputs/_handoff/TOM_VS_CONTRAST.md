# ToM benchmark performance vs intent use in moral judgment (J1)

## Question

Do models that pass a standard false-belief ToM benchmark still fail to weight
intent in graded moral judgment?

## Finding (primary)

Models that pass BigToM false belief at **0.82–0.99** are the same models with
2×2 contrasts of **−0.37 to −0.65** — outcome-driven, not intent-driven. The
per-model table and the scatter (BigToM FB × contrast, base/instruct by marker,
engaged models labelled, **no regression line**) are the deliverable. The
populated high-FB / negative-contrast region is the dissociation claim.

Engagement labels use `rating_std ≥ 0.2191` (engagement floor only;
correlations keep every model — see `FLOOR_DERIVATION.md`).

## BigToM condition: `init_belief=0`

All BigToM numbers here use the **hard** Forward-Belief variant: sentence 4 of
each story (the explicit statement of the agent’s initial belief) is dropped, so
the model must **infer** the belief rather than copy it. That is the
`init_belief=0` setting from the BigToM generator. Reporting a pass under this
condition is stronger than a pass when the belief is written out in the prompt.

## ToMi

ToMi is **not** used in the argument. An audit found the scored 400-item slice is
82% `no_tom` items; aggregate accuracy is not a false-belief measure. See
`TOMI_SCORING_AUDIT.md`. Numbers below retain ToMi only for provenance.

## Per-model table

| model | type | engaged | BigToM FB | contrast |
|---|---|---|---:|---:|
| Qwen-Qwen2-5-14B-Instruct | instruct | yes | 0.985 | -0.370 |
| unsloth-gemma-2-9b-it | instruct | yes | 0.935 | -0.408 |
| allenai-OLMo-2-1124-7B-Instruct | instruct | yes | 0.890 | -0.646 |
| allenai-Llama-3-1-Tulu-3-8B | instruct | yes | 0.855 | -0.401 |
| HuggingFaceH4-zephyr-7b-beta | instruct | yes | 0.835 | -0.551 |
| mistralai-Mistral-7B-Instruct-v0-3 | instruct | yes | 0.815 | -0.473 |
| Qwen-Qwen2-5-7B-Instruct | instruct | no | 0.935 | -0.238 |
| Qwen-Qwen2-5-3B-Instruct | instruct | no | 0.920 | -0.247 |
| unsloth-Meta-Llama-3-1-8B-Instruct | instruct | no | 0.865 | -0.202 |
| Qwen-Qwen2-5-0-5B-Instruct | instruct | no | 0.635 | -0.050 |
| Qwen-Qwen2-5-1-5B-Instruct | instruct | no | 0.545 | -0.167 |
| Qwen-Qwen2-5-14B | base | no | 0.940 | -0.126 |
| Qwen-Qwen2-5-7B | base | no | 0.935 | -0.051 |
| allenai-OLMo-2-1124-7B | base | no | 0.930 | -0.004 |
| unsloth-gemma-2-9b | base | no | 0.925 | -0.000 |
| unsloth-Meta-Llama-3-1-8B | base | no | 0.835 | +0.003 |
| mistralai-Mistral-7B-v0-3 | base | no | 0.800 | -0.003 |
| Qwen-Qwen2-5-3B | base | no | 0.795 | -0.048 |
| Qwen-Qwen2-5-0-5B | base | no | 0.775 | +0.000 |
| Qwen-Qwen2-5-1-5B | base | no | 0.625 | -0.013 |

### Finding quadrant (FB ≥ 0.82 and contrast ≤ −0.37)

| model | BigToM FB | contrast |
|---|---:|---:|
| allenai-OLMo-2-1124-7B-Instruct | 0.890 | -0.646 |
| HuggingFaceH4-zephyr-7b-beta | 0.835 | -0.551 |
| unsloth-gemma-2-9b-it | 0.935 | -0.408 |
| allenai-Llama-3-1-Tulu-3-8B | 0.855 | -0.401 |

## Scatter

`tom_vs_contrast.png` — x = BigToM false-belief (`init_belief=0`), y = contrast,
marker = base vs instruct, labels = engaged models only, **no regression line**.

## Why the all-model correlation is not a result

Both axes proxy model type. Base models often struggle with the BigToM QA format
and sit near zero on contrast; instruction tuning moves both. The unrestricted
Pearson r on BigToM-all over 20 models is **r = -0.738** [-0.841, -0.612] — a **confound demonstration**, not a finding.
On BigToM false belief alone the raw r is -0.261 [-0.526, +0.027]. Do not cite either as the result; cite the table.

## Secondary controlled analyses (BigToM FB only)

Kept for completeness after the table. These are not the headline.

| analysis | estimate | 95% CI | n |
|---|---|---|---|
| (a) instruct only | r = -0.473 | [-0.776, +0.513] | 11 |
| (b) OLS ToM + type + log(size) | partial r = -0.310 (β_tom=-0.379, p=0.193) | [-0.612, +0.158] | 20 |
| (c) within-family Δ | r = -0.160 | [-0.824, +0.807] | 9 |

## Within-family deltas (BigToM FB)

| family | Δ ToM (I−B) | Δ contrast (I−B) |
|---|---|---|
| Qwen2.5-0.5B | -0.140 | -0.051 |
| Qwen2.5-1.5B | -0.080 | -0.154 |
| Qwen2.5-3B | +0.125 | -0.199 |
| Qwen2.5-7B | +0.000 | -0.187 |
| Qwen2.5-14B | +0.045 | -0.244 |
| OLMo-2-1124-7B | -0.040 | -0.642 |
| Mistral-7B-v0.3 | +0.015 | -0.471 |
| gemma-2-9b | +0.010 | -0.408 |
| Meta-Llama-3.1-8B | +0.030 | -0.204 |

## Closed models

Closed-API BigToM (generative) is reported in
`tom_accuracy_by_model_generative.csv` / `CLOSED_TOM.md` when available.
**Do not correlate** closed ToM accuracy against their moral contrasts — those
contrasts are still v1-contaminated. Standalone ToM numbers only.

## Reading

Lead with the table and scatter. Models that clear hard false belief still produce
strongly outcome-driven moral contrasts. That is the dissociation measured on our
own open-weight roster under `init_belief=0`.
