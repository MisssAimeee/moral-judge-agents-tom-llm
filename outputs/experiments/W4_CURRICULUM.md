# W4 — prompt curriculum: is the intent representation reachable from the input?

_6 engaged open models, 5 cumulative levels, 7-template basis, 53 scenario groups (371 template × group cells). Readings pre-registered in `W4_PRESPEC.md`; prompts verbatim in `W4_PROMPT_LEVELS.md`._

> **Two corrections applied after run 1** (preserved under `_w4_prefix_fewshot_bug/`). The few-shot block rendered each example under the question built from the *target* item, and `human_verbatim` interpolates the agent name — so an example about Nadia was followed by "How permissible was Grace's action?", invalid on 1 of 7 templates and only at the levels containing the few-shot block. L4/L5 were rescored. Separately, the bootstrap resampled template × group cells rather than scenario groups, treating one vignette under seven templates as seven independent observations; all intervals are now over the scenario group and are correspondingly wider.

> **The L4 labels are not inverted.** Checked before interpreting L4, since Young 2007 is 1–4 permissibility while most templates are 1–7 blame, and this project has had both a CPR polarity inversion and a permissibility-direction reversal. The YS2008 anchor is phrased 1 = completely permissible → 3 = completely impermissible, ascending in condemnation like every other template, so no reversal is required. The few-shot labels encode attempted > accidental on every template × scale, implied contrast +0.500 to +0.667 against an adult reference of +0.666. The table is in `W4_PROMPT_LEVELS.md` and the check is now a gate that aborts the run.

## Verdict

**Level-specific, and the pre-registered bar is not met where it was placed.** P3 reads the verdict at the top level: there 3/6 models qualify, short of the 4 required, so by the pre-registration this is not "prompting works". But 4/6 models DO clear +0.15 with a CI excluding zero at some earlier level, which is not nothing and is not what P2 describes either.

The honest statement is the third one the pre-registration did not anticipate: **in-context intervention can move the contrast substantially, and the fully escalated prompt is not where it moves most.** Escalation is not monotone, so "can prompting fix it" and "does more scaffolding fix it more" have different answers. The dose-response section below is the part to read, in particular the L4 column: adding labelled adult ratings can undo the gain from unlabelled worked reasoning. The labels themselves are verified non-inverted, so that reversal is not arithmetic; what it is instead — imitation of the example anchors, the added prompt length, or the question moving away from the story — this design does not separate, and the ablation below is the closest available handle. Reporting the maximum over levels as the headline would be selecting the level after seeing the data; it is reported as descriptive only.

**The mechanism is not the one the intervention was aimed at.** Of the 6 models whose contrast improves at the top level, 6 improve entirely because blame for *accidental* harm falls, and 0 add blame to *attempted* harm. In 5 of them all four cell means move in the same direction, which is compression of the rating range rather than a re-weighting of either factor. So in-context instruction does move the judgment, but by making the model less condemnatory about bad outcomes rather than more condemnatory about bad intentions — the adult pattern is approached from the wrong side. This is the same caveat that governs the W3 difference-of-means direction and it applies to W4 at least as strongly.

Reference points: adults sit at +0.666 on this measure (Young 2007 digitized); the W3 outcome positive control moves the contrast 0.232+ at the same models, so a shift of +0.15 is well inside what this design resolves.

## Contrast by level

| model | L1 | L2 | L3 | L4 | L5 | Δ(top−L1) [95% CI] | bar at top level | best level Δ [95% CI] | crosses 0 | where the gain comes from (at top level) |
|---|---|---|---|---|---|---|---|---|---|---|
| HuggingFaceH4/zephyr-7b-beta | -0.552 | -0.487 | -0.288 | -0.281 | -0.337 | +0.215 [+0.168, +0.260] | **met** | L4: +0.271 [+0.218, +0.324] ✱ | no | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) |
| Qwen/Qwen2.5-14B-Instruct | -0.338 | -0.138 | -0.129 | -0.209 | -0.010 | +0.328 [+0.266, +0.397] | **met** | L5: +0.328 [+0.266, +0.397] ✱ | no | **accidental ↓** — outcome-blame suppressed |
| allenai/Llama-3.1-Tulu-3-8B | -0.398 | -0.335 | -0.251 | -0.410 | -0.337 | +0.061 [+0.026, +0.095] | not met | L3: +0.147 [+0.112, +0.183] | no | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) |
| allenai/OLMo-2-1124-7B-Instruct | -0.687 | -0.649 | -0.482 | -0.706 | -0.431 | +0.256 [+0.204, +0.303] | **met** | L5: +0.256 [+0.204, +0.303] ✱ | no | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) |
| mistralai/Mistral-7B-Instruct-v0.3 | -0.460 | -0.442 | -0.266 | -0.426 | -0.399 | +0.062 [+0.015, +0.109] | not met | L3: +0.194 [+0.159, +0.230] ✱ | no | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) |
| unsloth/gemma-2-9b-it | -0.375 | -0.234 | -0.236 | -0.333 | -0.271 | +0.104 [+0.042, +0.166] | not met | L2: +0.141 [+0.105, +0.176] | no | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) |

Δ columns are paired over scenario groups against that model's own L1. The pre-registered verdict is the `bar at top level` column; `best level Δ` is descriptive (✱ marks a shift that would have cleared the bar had the pre-registration placed it at that level, which it does not). The last column is the ceiling-compression check: a contrast gain built from `accidental` falling is outcome-blame suppression, not the intent re-weighting the intervention was aimed at, and a gain in which all four cells slide together is neither. Per-cell numbers are in the P6 table below.

## Dose-response (P5)

Levels are cumulative, so under P1 the shift should be monotone or nearly so. L4 adds labelled adult ratings; L5 adds the principle and no new labels, so an L4-only jump that does not persist at L5 indicates imitation of the few-shot format rather than uptake of the principle. The reverse — a gain at L3 that L4 destroys — indicates the opposite: the labelled examples are being imitated as anchors and are overriding the reasoning they were meant to illustrate.

| model | monotone | L1→L2 | L2→L3 | L3→L4 | L4→L5 | reading |
|---|---|---:|---:|---:|---:|---|
| HuggingFaceH4/zephyr-7b-beta | no | +0.064 | +0.199 | +0.007 | -0.056 | shift present and survives to the top level |
| Qwen/Qwen2.5-14B-Instruct | no | +0.201 | +0.008 | -0.079 | +0.199 | shift present and survives to the top level |
| allenai/Llama-3.1-Tulu-3-8B | no | +0.062 | +0.085 | -0.159 | +0.073 | no shift of the pre-registered size at any level |
| allenai/OLMo-2-1124-7B-Instruct | no | +0.038 | +0.166 | -0.224 | +0.276 | worked reasoning helps, adding labelled examples reverses it |
| mistralai/Mistral-7B-Instruct-v0.3 | no | +0.019 | +0.176 | -0.161 | +0.028 | worked reasoning helps, adding labelled examples reverses it |
| unsloth/gemma-2-9b-it | no | +0.141 | -0.002 | -0.097 | +0.062 | no shift of the pre-registered size at any level |

## All four cell means at every level (P6 ceiling guard)

A contrast change produced by all four means moving together is compression, not intent re-weighting — the same caveat that governs the W3 difference-of-means estimator. `extreme` is the fraction of ratings at the top or bottom of the scale; a level that pushes ratings to an endpoint removes the headroom the contrast needs.

| model | level | neutral | accidental | attempted | intentional | contrast | Δ cells vs L1 | attribution | rating SD | extreme |
|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|
| HuggingFaceH4/zephyr-7b-beta | L1 baseline | 0.063 | 0.698 | 0.146 | 0.700 | -0.552 | neut+0.000 acci+0.000 atte+0.000 inte+0.000 | no gain | 0.3878 | 0.39 |
| HuggingFaceH4/zephyr-7b-beta | L2 belief_cue | 0.038 | 0.639 | 0.152 | 0.665 | -0.487 | neut-0.025 acci-0.059 atte+0.006 inte-0.035 | **accidental ↓** — outcome-blame suppressed | 0.3470 | 0.33 |
| HuggingFaceH4/zephyr-7b-beta | L3 worked_example | 0.084 | 0.466 | 0.178 | 0.498 | -0.288 | neut+0.021 acci-0.231 atte+0.033 inte-0.202 | **both** — attempted ↑, accidental ↓ (intent re-weighting) | 0.2678 | 0.24 |
| HuggingFaceH4/zephyr-7b-beta | L4 few_shot_adult | 0.040 | 0.335 | 0.054 | 0.375 | -0.281 | neut-0.024 acci-0.363 atte-0.092 inte-0.325 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) | 0.2160 | 0.15 |
| HuggingFaceH4/zephyr-7b-beta | L5 intent_principle | 0.021 | 0.462 | 0.126 | 0.554 | -0.337 | neut-0.043 acci-0.235 atte-0.020 inte-0.146 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) | 0.2754 | 0.30 |
| Qwen/Qwen2.5-14B-Instruct | L1 baseline | 0.094 | 0.646 | 0.308 | 0.679 | -0.338 | neut+0.000 acci+0.000 atte+0.000 inte+0.000 | no gain | 0.3366 | 0.29 |
| Qwen/Qwen2.5-14B-Instruct | L2 belief_cue | 0.073 | 0.458 | 0.321 | 0.568 | -0.138 | neut-0.022 acci-0.187 atte+0.013 inte-0.111 | **accidental ↓** — outcome-blame suppressed | 0.2787 | 0.22 |
| Qwen/Qwen2.5-14B-Instruct | L3 worked_example | 0.062 | 0.373 | 0.243 | 0.485 | -0.129 | neut-0.033 acci-0.273 atte-0.064 inte-0.194 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) | 0.2157 | 0.20 |
| Qwen/Qwen2.5-14B-Instruct | L4 few_shot_adult | 0.030 | 0.457 | 0.249 | 0.723 | -0.209 | neut-0.064 acci-0.189 atte-0.059 inte+0.044 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added | 0.3395 | 0.39 |
| Qwen/Qwen2.5-14B-Instruct | L5 intent_principle | 0.016 | 0.335 | 0.326 | 0.675 | -0.010 | neut-0.078 acci-0.310 atte+0.018 inte-0.004 | **accidental ↓** — outcome-blame suppressed | 0.3215 | 0.38 |
| allenai/Llama-3.1-Tulu-3-8B | L1 baseline | 0.145 | 0.709 | 0.312 | 0.695 | -0.398 | neut+0.000 acci+0.000 atte+0.000 inte+0.000 | no gain | 0.2894 | 0.03 |
| allenai/Llama-3.1-Tulu-3-8B | L2 belief_cue | 0.113 | 0.625 | 0.289 | 0.638 | -0.335 | neut-0.032 acci-0.084 atte-0.022 inte-0.057 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) | 0.2744 | 0.03 |
| allenai/Llama-3.1-Tulu-3-8B | L3 worked_example | 0.091 | 0.484 | 0.233 | 0.502 | -0.251 | neut-0.054 acci-0.225 atte-0.078 inte-0.193 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) | 0.2251 | 0.04 |
| allenai/Llama-3.1-Tulu-3-8B | L4 few_shot_adult | 0.109 | 0.682 | 0.272 | 0.657 | -0.410 | neut-0.036 acci-0.028 atte-0.040 inte-0.038 | no gain | 0.2860 | 0.02 |
| allenai/Llama-3.1-Tulu-3-8B | L5 intent_principle | 0.092 | 0.597 | 0.260 | 0.614 | -0.337 | neut-0.053 acci-0.113 atte-0.052 inte-0.082 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) | 0.2596 | 0.02 |
| allenai/OLMo-2-1124-7B-Instruct | L1 baseline | 0.094 | 0.904 | 0.217 | 0.888 | -0.687 | neut+0.000 acci+0.000 atte+0.000 inte+0.000 | no gain | 0.4120 | 0.35 |
| allenai/OLMo-2-1124-7B-Instruct | L2 belief_cue | 0.039 | 0.799 | 0.150 | 0.759 | -0.649 | neut-0.055 acci-0.105 atte-0.067 inte-0.128 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) | 0.4030 | 0.34 |
| allenai/OLMo-2-1124-7B-Instruct | L3 worked_example | 0.021 | 0.547 | 0.065 | 0.514 | -0.482 | neut-0.074 acci-0.357 atte-0.153 inte-0.374 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) | 0.3220 | 0.32 |
| allenai/OLMo-2-1124-7B-Instruct | L4 few_shot_adult | 0.054 | 0.853 | 0.147 | 0.820 | -0.706 | neut-0.041 acci-0.051 atte-0.070 inte-0.067 | no gain | 0.4079 | 0.27 |
| allenai/OLMo-2-1124-7B-Instruct | L5 intent_principle | 0.045 | 0.528 | 0.097 | 0.495 | -0.431 | neut-0.049 acci-0.376 atte-0.120 inte-0.393 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) | 0.2740 | 0.11 |
| mistralai/Mistral-7B-Instruct-v0.3 | L1 baseline | 0.175 | 0.762 | 0.302 | 0.755 | -0.460 | neut+0.000 acci+0.000 atte+0.000 inte+0.000 | no gain | 0.3357 | 0.14 |
| mistralai/Mistral-7B-Instruct-v0.3 | L2 belief_cue | 0.110 | 0.722 | 0.281 | 0.757 | -0.442 | neut-0.065 acci-0.040 atte-0.021 inte+0.002 | no gain | 0.3416 | 0.19 |
| mistralai/Mistral-7B-Instruct-v0.3 | L3 worked_example | 0.151 | 0.492 | 0.226 | 0.507 | -0.266 | neut-0.024 acci-0.270 atte-0.076 inte-0.247 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) | 0.2641 | 0.09 |
| mistralai/Mistral-7B-Instruct-v0.3 | L4 few_shot_adult | 0.108 | 0.654 | 0.228 | 0.664 | -0.426 | neut-0.067 acci-0.108 atte-0.074 inte-0.090 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) | 0.2776 | 0.05 |
| mistralai/Mistral-7B-Instruct-v0.3 | L5 intent_principle | 0.066 | 0.634 | 0.236 | 0.683 | -0.399 | neut-0.109 acci-0.128 atte-0.066 inte-0.071 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) | 0.2971 | 0.14 |
| unsloth/gemma-2-9b-it | L1 baseline | 0.173 | 0.814 | 0.439 | 0.807 | -0.375 | neut+0.000 acci+0.000 atte+0.000 inte+0.000 | no gain | 0.3564 | 0.27 |
| unsloth/gemma-2-9b-it | L2 belief_cue | 0.114 | 0.704 | 0.471 | 0.759 | -0.234 | neut-0.059 acci-0.110 atte+0.031 inte-0.048 | **both** — attempted ↑, accidental ↓ (intent re-weighting) | 0.3348 | 0.22 |
| unsloth/gemma-2-9b-it | L3 worked_example | 0.072 | 0.541 | 0.305 | 0.619 | -0.236 | neut-0.100 acci-0.273 atte-0.134 inte-0.188 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) | 0.3055 | 0.21 |
| unsloth/gemma-2-9b-it | L4 few_shot_adult | 0.058 | 0.623 | 0.290 | 0.695 | -0.333 | neut-0.115 acci-0.191 atte-0.150 inte-0.113 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) | 0.3347 | 0.28 |
| unsloth/gemma-2-9b-it | L5 intent_principle | 0.045 | 0.555 | 0.284 | 0.662 | -0.271 | neut-0.128 acci-0.259 atte-0.155 inte-0.145 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) | 0.3540 | 0.36 |

## Relation to W3 (steering)

Steering and prompting are different interventions on different quantities and share no common effect-size scale, so the two are not divided into each other here. The defensible statement is qualitative: **the intent representation is inert to residual-stream intervention at the depths where intent is resolvable (W3, |Δcontrast| ≤ 0.015), while the same contrast moves substantially under in-context instruction. That places the blockage downstream of the representation and upstream of the output** — the rating computation can be re-pointed by the input but not by editing the vector the probe reads.

Descriptively, the largest in-context shift observed (+0.328) sits close to the range the W3 *outcome* direction produced (0.232–0.259) as a positive control. That is a coincidence of magnitude worth flagging and nothing is built on it: the two numbers come from different manipulations, and the outcome-direction figure carries its own compression caveat. Its only use is as a reminder that shifts of this size are well inside what the design resolves.

## What this does and does not license

- The levels differ only in added text. Weights, decoding (logprob-EV over rating digits, deterministic), items, templates, scale normalisation and the contrast estimator are identical across levels, so the level effect is not confounded with the measurement.

- The L3/L4 example vignettes are held out: novel content (climbing gym, print shop) that appears nowhere in the master, so the scaffolding cannot teach a test item. The L4 labels are the Young 2007 adult profile mapped onto each template's own scale; they are used only to write the examples and are never a target in the analysis.

- A contrast gain is not automatically intent re-weighting. Read the attribution column: where the gain is `accidental` falling, the intervention removed blame from accidental harm rather than adding it to attempted harm, and where all four cells slide together the movement is compression. Both are real changes to the judgment and neither is evidence that the model started consulting intent.

- This is an in-context result. It says nothing about whether fine-tuning could move the contrast, and a prompt-level shift does not establish that the model is using the same intent representation the probe reads — only that the input can change the weighting. Establishing that it is the same representation would require the W3 intervention to work, which it does not.

