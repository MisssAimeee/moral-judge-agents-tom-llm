# W4 — prompt curriculum: is the intent representation reachable from the input?

_6 engaged open models, 5 cumulative levels, 7-template basis, 371 scenario groups. Readings pre-registered in `W4_PRESPEC.md`; prompts verbatim in `W4_PROMPT_LEVELS.md`._

## Verdict

**Level-specific, and the pre-registered bar is not met where it was placed.** P3 reads the verdict at the top level: there 3/6 models qualify, short of the 4 required, so by the pre-registration this is not "prompting works". But 4/6 models DO clear +0.15 with a CI excluding zero at some earlier level, which is not nothing and is not what P2 describes either.

The honest statement is the third one the pre-registration did not anticipate: **in-context intervention can move the contrast substantially, and the fully escalated prompt is not where it moves most.** Escalation is not monotone, so "can prompting fix it" and "does more scaffolding fix it more" have different answers. The dose-response section below is the part to read, in particular the L4 column: adding labelled adult ratings can undo the gain from unlabelled worked reasoning, which points at format/anchor imitation competing with the content of the instruction. Reporting the maximum over levels as the headline would be selecting the level after seeing the data; it is reported as descriptive only.

Reference points: adults sit at +0.666 on this measure (Young 2007 digitized); the W3 outcome positive control moves the contrast 0.232+ at the same models, so a shift of +0.15 is well inside what this design resolves.

## Contrast by level

| model | L1 | L2 | L3 | L4 | L5 | Δ(top−L1) [95% CI] | bar at top level | best level Δ [95% CI] | crosses 0 |
|---|---|---|---|---|---|---|---|---|---|
| HuggingFaceH4/zephyr-7b-beta | -0.552 | -0.487 | -0.288 | -0.281 | -0.337 | +0.215 [+0.180, +0.251] | **met** | L4: +0.271 [+0.234, +0.308] ✱ | no |
| Qwen/Qwen2.5-14B-Instruct | -0.338 | -0.138 | -0.129 | -0.209 | -0.010 | +0.328 [+0.296, +0.360] | **met** | L5: +0.328 [+0.296, +0.360] ✱ | no |
| allenai/Llama-3.1-Tulu-3-8B | -0.398 | -0.335 | -0.251 | -0.410 | -0.337 | +0.061 [+0.042, +0.079] | not met | L3: +0.147 [+0.129, +0.165] | no |
| allenai/OLMo-2-1124-7B-Instruct | -0.687 | -0.649 | -0.482 | -0.706 | -0.431 | +0.256 [+0.228, +0.286] | **met** | L5: +0.256 [+0.228, +0.286] ✱ | no |
| mistralai/Mistral-7B-Instruct-v0.3 | -0.460 | -0.442 | -0.266 | -0.426 | -0.399 | +0.062 [+0.038, +0.086] | not met | L3: +0.194 [+0.172, +0.216] ✱ | no |
| unsloth/gemma-2-9b-it | -0.375 | -0.234 | -0.236 | -0.333 | -0.271 | +0.104 [+0.068, +0.138] | not met | L2: +0.141 [+0.122, +0.160] | no |

Δ columns are paired over scenario groups against that model's own L1. The pre-registered verdict is the `bar at top level` column; `best level Δ` is descriptive (✱ marks a shift that would have cleared the bar had the pre-registration placed it at that level, which it does not).

## Dose-response (P5)

Levels are cumulative, so under P1 the shift should be monotone or nearly so. L4 adds labelled adult ratings; L5 adds the principle and no new labels, so an L4-only jump that does not persist at L5 indicates imitation of the few-shot format rather than uptake of the principle. The reverse — a gain at L3 that L4 destroys — indicates the opposite: the labelled examples are being imitated as anchors and are overriding the reasoning they were meant to illustrate.

| model | monotone | L1→L2 | L2→L3 | L3→L4 | L4→L5 | reading |
|---|---|---:|---:|---:|---:|---|
| HuggingFaceH4/zephyr-7b-beta | no | +0.064 | +0.199 | +0.007 | -0.056 | shift present and survives to the top level |
| Qwen/Qwen2.5-14B-Instruct | no | +0.201 | +0.008 | -0.079 | +0.199 | shift present and survives to the top level |
| allenai/Llama-3.1-Tulu-3-8B | no | +0.062 | +0.085 | -0.159 | +0.073 | no shift of the pre-registered size at any level |
| allenai/OLMo-2-1124-7B-Instruct | no | +0.038 | +0.166 | -0.224 | +0.276 | worked reasoning helps, labelled examples undo it — anchor imitation overriding the instruction |
| mistralai/Mistral-7B-Instruct-v0.3 | no | +0.019 | +0.176 | -0.161 | +0.028 | worked reasoning helps, labelled examples undo it — anchor imitation overriding the instruction |
| unsloth/gemma-2-9b-it | no | +0.141 | -0.002 | -0.097 | +0.062 | no shift of the pre-registered size at any level |

## All four cell means at every level (P6 ceiling guard)

A contrast change produced by all four means moving together is compression, not intent re-weighting — the same caveat that governs the W3 difference-of-means estimator. `extreme` is the fraction of ratings at the top or bottom of the scale; a level that pushes ratings to an endpoint removes the headroom the contrast needs.

| model | level | neutral | accidental | attempted | intentional | contrast | b_intent | b_outcome | rating SD | extreme |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HuggingFaceH4/zephyr-7b-beta | L1 baseline | 0.063 | 0.698 | 0.146 | 0.700 | -0.552 | +0.083 | +0.634 | 0.3878 | 0.39 |
| HuggingFaceH4/zephyr-7b-beta | L2 belief_cue | 0.038 | 0.639 | 0.152 | 0.665 | -0.487 | +0.113 | +0.600 | 0.3470 | 0.33 |
| HuggingFaceH4/zephyr-7b-beta | L3 worked_example | 0.084 | 0.466 | 0.178 | 0.498 | -0.288 | +0.094 | +0.382 | 0.2678 | 0.24 |
| HuggingFaceH4/zephyr-7b-beta | L4 few_shot_adult | 0.040 | 0.335 | 0.054 | 0.375 | -0.281 | +0.014 | +0.295 | 0.2160 | 0.15 |
| HuggingFaceH4/zephyr-7b-beta | L5 intent_principle | 0.021 | 0.462 | 0.126 | 0.554 | -0.337 | +0.105 | +0.442 | 0.2754 | 0.30 |
| Qwen/Qwen2.5-14B-Instruct | L1 baseline | 0.094 | 0.646 | 0.308 | 0.679 | -0.338 | +0.214 | +0.552 | 0.3366 | 0.29 |
| Qwen/Qwen2.5-14B-Instruct | L2 belief_cue | 0.073 | 0.458 | 0.321 | 0.568 | -0.138 | +0.248 | +0.386 | 0.2787 | 0.22 |
| Qwen/Qwen2.5-14B-Instruct | L3 worked_example | 0.062 | 0.373 | 0.243 | 0.485 | -0.129 | +0.182 | +0.311 | 0.2157 | 0.20 |
| Qwen/Qwen2.5-14B-Instruct | L4 few_shot_adult | 0.030 | 0.457 | 0.249 | 0.723 | -0.209 | +0.219 | +0.428 | 0.3395 | 0.39 |
| Qwen/Qwen2.5-14B-Instruct | L5 intent_principle | 0.016 | 0.335 | 0.326 | 0.675 | -0.010 | +0.309 | +0.319 | 0.3215 | 0.38 |
| allenai/Llama-3.1-Tulu-3-8B | L1 baseline | 0.145 | 0.709 | 0.312 | 0.695 | -0.398 | +0.167 | +0.564 | 0.2894 | 0.03 |
| allenai/Llama-3.1-Tulu-3-8B | L2 belief_cue | 0.113 | 0.625 | 0.289 | 0.638 | -0.335 | +0.176 | +0.511 | 0.2744 | 0.03 |
| allenai/Llama-3.1-Tulu-3-8B | L3 worked_example | 0.091 | 0.484 | 0.233 | 0.502 | -0.251 | +0.142 | +0.393 | 0.2251 | 0.04 |
| allenai/Llama-3.1-Tulu-3-8B | L4 few_shot_adult | 0.109 | 0.682 | 0.272 | 0.657 | -0.410 | +0.163 | +0.573 | 0.2860 | 0.02 |
| allenai/Llama-3.1-Tulu-3-8B | L5 intent_principle | 0.092 | 0.597 | 0.260 | 0.614 | -0.337 | +0.168 | +0.505 | 0.2596 | 0.02 |
| allenai/OLMo-2-1124-7B-Instruct | L1 baseline | 0.094 | 0.904 | 0.217 | 0.888 | -0.687 | +0.123 | +0.809 | 0.4120 | 0.35 |
| allenai/OLMo-2-1124-7B-Instruct | L2 belief_cue | 0.039 | 0.799 | 0.150 | 0.759 | -0.649 | +0.111 | +0.760 | 0.4030 | 0.34 |
| allenai/OLMo-2-1124-7B-Instruct | L3 worked_example | 0.021 | 0.547 | 0.065 | 0.514 | -0.482 | +0.044 | +0.526 | 0.3220 | 0.32 |
| allenai/OLMo-2-1124-7B-Instruct | L4 few_shot_adult | 0.054 | 0.853 | 0.147 | 0.820 | -0.706 | +0.093 | +0.800 | 0.4079 | 0.27 |
| allenai/OLMo-2-1124-7B-Instruct | L5 intent_principle | 0.045 | 0.528 | 0.097 | 0.495 | -0.431 | +0.052 | +0.482 | 0.2740 | 0.11 |
| mistralai/Mistral-7B-Instruct-v0.3 | L1 baseline | 0.175 | 0.762 | 0.302 | 0.755 | -0.460 | +0.127 | +0.587 | 0.3357 | 0.14 |
| mistralai/Mistral-7B-Instruct-v0.3 | L2 belief_cue | 0.110 | 0.722 | 0.281 | 0.757 | -0.442 | +0.170 | +0.612 | 0.3416 | 0.19 |
| mistralai/Mistral-7B-Instruct-v0.3 | L3 worked_example | 0.151 | 0.492 | 0.226 | 0.507 | -0.266 | +0.075 | +0.341 | 0.2641 | 0.09 |
| mistralai/Mistral-7B-Instruct-v0.3 | L4 few_shot_adult | 0.108 | 0.654 | 0.228 | 0.664 | -0.426 | +0.120 | +0.546 | 0.2776 | 0.05 |
| mistralai/Mistral-7B-Instruct-v0.3 | L5 intent_principle | 0.066 | 0.634 | 0.236 | 0.683 | -0.399 | +0.170 | +0.569 | 0.2971 | 0.14 |
| unsloth/gemma-2-9b-it | L1 baseline | 0.173 | 0.814 | 0.439 | 0.807 | -0.375 | +0.266 | +0.641 | 0.3564 | 0.27 |
| unsloth/gemma-2-9b-it | L2 belief_cue | 0.114 | 0.704 | 0.471 | 0.759 | -0.234 | +0.356 | +0.590 | 0.3348 | 0.22 |
| unsloth/gemma-2-9b-it | L3 worked_example | 0.072 | 0.541 | 0.305 | 0.619 | -0.236 | +0.233 | +0.469 | 0.3055 | 0.21 |
| unsloth/gemma-2-9b-it | L4 few_shot_adult | 0.058 | 0.623 | 0.290 | 0.695 | -0.333 | +0.232 | +0.565 | 0.3347 | 0.28 |
| unsloth/gemma-2-9b-it | L5 intent_principle | 0.045 | 0.555 | 0.284 | 0.662 | -0.271 | +0.239 | +0.510 | 0.3540 | 0.36 |

## What this does and does not license

- The levels differ only in added text. Weights, decoding (logprob-EV over rating digits, deterministic), items, templates, scale normalisation and the contrast estimator are identical across levels, so the level effect is not confounded with the measurement.

- The L3/L4 example vignettes are held out: novel content (climbing gym, print shop) that appears nowhere in the master, so the scaffolding cannot teach a test item. The L4 labels are the Young 2007 adult profile mapped onto each template's own scale; they are used only to write the examples and are never a target in the analysis.

- This is an in-context result. It says nothing about whether fine-tuning could move the contrast, and a prompt-level shift does not establish that the model is using the same intent representation the probe reads — only that the input can change the weighting. Establishing that it is the same representation would require the W3 intervention to work, which it does not.

