# ToM benchmark performance vs intent use in moral judgment (J1)

## Question

Does standard ToM-benchmark performance predict whether a model weights intent in
graded moral judgment — once the base/instruct (and size) confound is controlled?

## Why the all-20 correlation is not a result

Both axes are proxies for model type. Base models score near floor on BigToM
because they cannot follow the QA format, and sit near zero on the 2×2 contrast.
Instruction tuning and scale move both axes. The unrestricted Pearson r
(e.g. BigToM–contrast ≈ −0.74 over 20 models) demonstrates that confound; it is
**not reported as a finding**. The three analyses below are.

## Floor policy

Correlation analyses keep **every** model. The derived `rating_std` floor
(0.2191) is for engagement / anchor counts only — see
`outputs/stats/FLOOR_DERIVATION.md`. Using it here would select on a variable
adjacent to the outcome. The fix for the confound is controlling for type, not
excluding models.

## Ceiling gate

| benchmark | accuracies (0.5B-I / 14B-I / OLMo-I) | spread | verdict |
|---|---|---|---|
| BigToM | 0.520 / 0.882 / 0.850 | 0.362 | spread, proceed |
| ToMi | 0.482 / 0.512 / 0.818 | 0.335 | spread, proceed |

## Controlled results

| analysis | ToM measure | estimate | 95% CI | n | reading |
|---|---|---|---|---|---|
| (a) instruct only | bigtom|false_belief | -0.473 | [-0.776, +0.513] | 11 | UNINFORMATIVE — interval too wide to exclude a moderate effect either way. Not a null. |
| (b) OLS with covariates | bigtom|false_belief | -0.310 (β_tom=-0.379, p=0.211) | [-0.612, +0.158] | 20 | NULL, bounded — ToM accuracy does not predict the contrast once controlling for type and log-size are applied. |
| (c) within-family deltas | bigtom|false_belief | -0.160 | [-0.824, +0.807] | 9 | UNINFORMATIVE — interval too wide to exclude a moderate effect either way. Not a null. |
| (a) instruct only | bigtom | -0.709 | [-0.931, -0.180] | 11 | NEGATIVE — higher ToM tracks more outcome-driven contrast; check residual confounds before interpreting as a finding. |
| (b) OLS with covariates | bigtom | -0.588 (β_tom=-0.907, p=0.0103) | [-0.825, -0.176] | 20 | NEGATIVE — higher ToM tracks more outcome-driven contrast; check residual confounds before interpreting as a finding. |
| (c) within-family deltas | bigtom | -0.762 | [-0.949, -0.307] | 9 | NEGATIVE — higher ToM tracks more outcome-driven contrast; check residual confounds before interpreting as a finding. |
| (a) instruct only | tomi | -0.650 | [-0.954, -0.058] | 11 | NEGATIVE — higher ToM tracks more outcome-driven contrast; check residual confounds before interpreting as a finding. |
| (b) OLS with covariates | tomi | -0.343 (β_tom=-0.424, p=0.163) | [-0.688, +0.178] | 20 | NULL, bounded — ToM accuracy does not predict the contrast once controlling for type and log-size are applied. |
| (c) within-family deltas | tomi | -0.225 | [-0.742, +0.589] | 9 | UNINFORMATIVE — interval too wide to exclude a moderate effect either way. Not a null. |

### Confound demonstration (not a result)

| analysis | ToM measure | r | 95% CI | n |
|---|---|---|---|---|
| (confound demo) all models, no controls | bigtom|false_belief | -0.261 | [-0.526, +0.027] | 20 |
| (confound demo) all models, no controls | bigtom | -0.738 | [-0.841, -0.612] | 20 |
| (confound demo) all models, no controls | tomi | -0.240 | [-0.710, +0.338] | 20 |

## Within-family deltas (primary measure: BigToM all)

| family | Δ ToM (I−B) | Δ contrast (I−B) | base contrast | instruct contrast |
|---|---|---|---|---|
| Qwen2.5-0.5B | +0.008 | -0.051 | +0.000 | -0.050 |
| Qwen2.5-1.5B | +0.062 | -0.154 | -0.013 | -0.167 |
| Qwen2.5-3B | +0.000 | -0.199 | -0.048 | -0.247 |
| Qwen2.5-7B | +0.148 | -0.187 | -0.051 | -0.238 |
| Qwen2.5-14B | +0.125 | -0.244 | -0.126 | -0.370 |
| OLMo-2-1124-7B | +0.220 | -0.642 | -0.004 | -0.646 |
| Mistral-7B-v0.3 | +0.105 | -0.471 | -0.003 | -0.473 |
| gemma-2-9b | +0.203 | -0.408 | -0.000 | -0.408 |
| Meta-Llama-3.1-8B | +0.085 | -0.204 | +0.003 | -0.202 |

## Per-model table

| model | type | params | BigToM FB | BigToM all | ToMi | contrast |
|---|---|---|---|---|---|---|
| Qwen_Qwen2_5-14B-Instruct | instruct | 14.0 | 0.985 | 0.882 | 0.512 | -0.370 |
| Qwen_Qwen2_5-7B-Instruct | instruct | 7.0 | 0.935 | 0.868 | 0.550 | -0.238 |
| unsloth_gemma-2-9b-it | instruct | 9.0 | 0.935 | 0.858 | 0.665 | -0.408 |
| allenai_OLMo-2-1124-7B-Instruct | instruct | 7.3 | 0.890 | 0.850 | 0.818 | -0.646 |
| mistralai_Mistral-7B-Instruct-v0_3 | instruct | 7.0 | 0.815 | 0.833 | 0.637 | -0.473 |
| HuggingFaceH4_zephyr-7b-beta | instruct | 7.2 | 0.835 | 0.833 | 0.522 | -0.551 |
| unsloth_Meta-Llama-3_1-8B-Instruct | instruct | 8.0 | 0.865 | 0.807 | 0.570 | -0.202 |
| allenai_Llama-3_1-Tulu-3-8B | instruct | 8.0 | 0.855 | 0.805 | 0.757 | -0.401 |
| Qwen_Qwen2_5-14B | base | 14.0 | 0.940 | 0.757 | 0.542 | -0.126 |
| mistralai_Mistral-7B-v0_3 | base | 7.0 | 0.800 | 0.728 | 0.620 | -0.003 |
| unsloth_Meta-Llama-3_1-8B | base | 8.0 | 0.835 | 0.723 | 0.728 | 0.003 |
| Qwen_Qwen2_5-7B | base | 7.0 | 0.935 | 0.720 | 0.520 | -0.051 |
| Qwen_Qwen2_5-3B | base | 3.0 | 0.795 | 0.675 | 0.537 | -0.048 |
| Qwen_Qwen2_5-3B-Instruct | instruct | 3.0 | 0.920 | 0.675 | 0.540 | -0.247 |
| unsloth_gemma-2-9b | base | 9.0 | 0.925 | 0.655 | 0.745 | -0.000 |
| allenai_OLMo-2-1124-7B | base | 7.3 | 0.930 | 0.630 | 0.757 | -0.004 |
| Qwen_Qwen2_5-1_5B-Instruct | instruct | 5.0 | 0.545 | 0.608 | 0.550 | -0.167 |
| Qwen_Qwen2_5-1_5B | base | 5.0 | 0.625 | 0.545 | 0.492 | -0.013 |
| Qwen_Qwen2_5-0_5B-Instruct | instruct | 5.0 | 0.635 | 0.520 | 0.482 | -0.050 |
| Qwen_Qwen2_5-0_5B | base | 5.0 | 0.775 | 0.512 | 0.505 | 0.000 |

## Reading

Report (a), (b), and (c). If all three are null or negative, ToM-benchmark
performance does not predict intent-weighting in moral judgment once type is
held constant — the dissociation is measured on our own models. A positive
result under (a)/(b) or a positive delta–delta under (c) would weaken that claim.
Do not cite the all-20 r.
