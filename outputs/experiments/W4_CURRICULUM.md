# W4 — prompt curriculum: is the intent representation reachable from the input?

_6 engaged open models, 5 cumulative levels plus a 3-cell non-cumulative ablation, 7-template basis, 53 scenario groups (371 template × group cells). Readings pre-registered in `W4_PRESPEC.md`; prompts verbatim in `W4_PROMPT_LEVELS.md`._

> **Two corrections applied after run 1** (preserved under `_w4_prefix_fewshot_bug/`). The few-shot block rendered each example under the question built from the *target* item, and `human_verbatim` interpolates the agent name — so an example about Nadia was followed by "How permissible was Grace's action?", invalid on 1 of 7 templates and only at the levels containing the few-shot block. L4/L5 were rescored. Separately, the bootstrap resampled template × group cells rather than scenario groups, treating one vignette under seven templates as seven independent observations; all intervals are now over the scenario group and are correspondingly wider.

> **The L4 labels are not inverted.** Checked before interpreting L4, since Young 2007 is 1–4 permissibility while most templates are 1–7 blame, and this project has had both a CPR polarity inversion and a permissibility-direction reversal. The YS2008 anchor is phrased 1 = completely permissible → 3 = completely impermissible, ascending in condemnation like every other template, so no reversal is required. The few-shot labels encode attempted > accidental on every template × scale, implied contrast +0.500 to +0.667 against an adult reference of +0.666. The table is in `W4_PROMPT_LEVELS.md` and the check is now a gate that aborts the run.

## Verdict

**Level-specific, and the pre-registered bar is not met where it was placed.** P3 reads the verdict at the top level: there 3/6 models qualify, short of the 4 required, so by the pre-registration this is not "prompting works". But 4/6 models DO clear +0.15 with a CI excluding zero at some earlier level, which is not nothing and is not what P2 describes either.

The honest statement is the third one the pre-registration did not anticipate: **in-context intervention can move the contrast substantially, and the fully escalated prompt is not where it moves most.** Escalation is not monotone, so "can prompting fix it" and "does more scaffolding fix it more" have different answers. The dose-response section below is the part to read, in particular the L4 column: adding labelled adult ratings can undo the gain from unlabelled worked reasoning. The labels themselves are verified non-inverted, so that reversal is not arithmetic; what it is instead — imitation of the example anchors, the added prompt length, or the question moving away from the story — this design does not separate, and the ablation below is the closest available handle. Reporting the maximum over levels as the headline would be selecting the level after seeing the data; it is reported as descriptive only.

**The mechanism is not the one the intervention was aimed at.** Of the 6 models whose contrast improves at the top level, 5 improve entirely because blame for *accidental* harm falls, and 1 add blame to *attempted* harm (Qwen/Qwen2.5-14B-Instruct, +0.027, against a fall in accidental of -0.314). In 5 of them all four cell means move in the same direction, which is compression of the rating range rather than a re-weighting of either factor. So at the top of the curriculum in-context instruction moves the judgment by making the model less condemnatory about bad outcomes rather than more condemnatory about bad intentions — the adult pattern is approached from the wrong side. This is the same caveat that governs the W3 difference-of-means direction.

**But that is a property of the stack, not of instruction as such.** With the intent principle stated *alone* (L8: no belief cue, no worked example, no labelled examples), 2 of 6 models produce a gain in which the *rise* in blame for attempted harm is the dominant term — mistralai/Mistral-7B-Instruct-v0.3 Δattempted +0.241 against Δaccidental -0.008; HuggingFaceH4/zephyr-7b-beta Δattempted +0.107 against Δaccidental -0.063 (2 further model(s) raise attempted harm by a smaller amount than they lower accidental harm, which is still outcome suppression.) That is the re-weighting the intervention was designed to produce, and it appears where the prompt is *least* elaborate. The scaffolding is what appears to convert intent re-weighting into blanket outcome-blame suppression, which fits the additivity column: every model is sub-additive, and in three the single best component beats the full stack. This is the secondary, post-hoc arm — the pre-registered verdict remains the cumulative L5 column, and two models are not a result on their own. It is the sharpest thing to test next.

Reference points: adults sit at +0.666 on this measure (Young 2007 digitized); the W3 outcome positive control moves the contrast 0.232+ at the same models, so a shift of +0.15 is well inside what this design resolves.

## Contrast by level

| model | L1 | L2 | L3 | L4 | L5 | Δ(top−L1) [95% CI] | bar at top level | best level Δ [95% CI] | crosses 0 | where the gain comes from (at top level) |
|---|---|---|---|---|---|---|---|---|---|---|
| HuggingFaceH4/zephyr-7b-beta | -0.552 | -0.487 | -0.288 | -0.280 | -0.337 | +0.215 [+0.169, +0.261] | **met** | L4: +0.272 [+0.221, +0.323] ✱ | no | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) |
| Qwen/Qwen2.5-14B-Instruct | -0.338 | -0.138 | -0.129 | -0.207 | +0.004 | +0.342 [+0.280, +0.410] | **met** | L5: +0.342 [+0.280, +0.410] ✱ | yes (L5) | **both** — attempted ↑, accidental ↓ (intent re-weighting) |
| allenai/Llama-3.1-Tulu-3-8B | -0.398 | -0.335 | -0.251 | -0.397 | -0.325 | +0.073 [+0.039, +0.107] | not met | L3: +0.147 [+0.112, +0.183] | no | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) |
| allenai/OLMo-2-1124-7B-Instruct | -0.687 | -0.649 | -0.482 | -0.706 | -0.434 | +0.253 [+0.201, +0.300] | **met** | L5: +0.253 [+0.201, +0.300] ✱ | no | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) |
| mistralai/Mistral-7B-Instruct-v0.3 | -0.460 | -0.442 | -0.266 | -0.423 | -0.397 | +0.063 [+0.017, +0.111] | not met | L3: +0.194 [+0.159, +0.230] ✱ | no | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) |
| unsloth/gemma-2-9b-it | -0.375 | -0.234 | -0.236 | -0.332 | -0.265 | +0.110 [+0.049, +0.172] | not met | L2: +0.141 [+0.105, +0.176] | no | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) |

Δ columns are paired over scenario groups against that model's own L1. The pre-registered verdict is the `bar at top level` column; `best level Δ` is descriptive (✱ marks a shift that would have cleared the bar had the pre-registration placed it at that level, which it does not). The last column is the ceiling-compression check: a contrast gain built from `accidental` falling is outcome-blame suppression, not the intent re-weighting the intervention was aimed at, and a gain in which all four cells slide together is neither. Per-cell numbers are in the P6 table below.

## Dose-response (P5)

Levels are cumulative, so under P1 the shift should be monotone or nearly so. L4 adds labelled adult ratings; L5 adds the principle and no new labels, so an L4-only jump that does not persist at L5 indicates imitation of the few-shot format rather than uptake of the principle. The reverse — a gain at L3 that L4 destroys — indicates the opposite: the labelled examples are being imitated as anchors and are overriding the reasoning they were meant to illustrate.

| model | monotone | L1→L2 | L2→L3 | L3→L4 | L4→L5 | reading |
|---|---|---:|---:|---:|---:|---|
| HuggingFaceH4/zephyr-7b-beta | no | +0.064 | +0.199 | +0.008 | -0.057 | shift present and survives to the top level |
| Qwen/Qwen2.5-14B-Instruct | no | +0.201 | +0.008 | -0.077 | +0.210 | shift present and survives to the top level |
| allenai/Llama-3.1-Tulu-3-8B | no | +0.062 | +0.085 | -0.146 | +0.072 | no shift of the pre-registered size at any level |
| allenai/OLMo-2-1124-7B-Instruct | no | +0.038 | +0.166 | -0.224 | +0.272 | worked reasoning helps, adding labelled examples reverses it |
| mistralai/Mistral-7B-Instruct-v0.3 | no | +0.019 | +0.176 | -0.157 | +0.026 | worked reasoning helps, adding labelled examples reverses it |
| unsloth/gemma-2-9b-it | no | +0.141 | -0.002 | -0.096 | +0.067 | no shift of the pre-registered size at any level |

## All four cell means at every level (P6 ceiling guard)

A contrast change produced by all four means moving together is compression, not intent re-weighting — the same caveat that governs the W3 difference-of-means estimator. `extreme` is the fraction of ratings at the top or bottom of the scale; a level that pushes ratings to an endpoint removes the headroom the contrast needs.

| model | level | neutral | accidental | attempted | intentional | contrast | Δ cells vs L1 | attribution | rating SD | extreme |
|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|
| HuggingFaceH4/zephyr-7b-beta | L1 baseline | 0.063 | 0.698 | 0.146 | 0.700 | -0.552 | neut+0.000 acci+0.000 atte+0.000 inte+0.000 | no gain | 0.3878 | 0.39 |
| HuggingFaceH4/zephyr-7b-beta | L2 belief_cue | 0.038 | 0.639 | 0.152 | 0.665 | -0.487 | neut-0.025 acci-0.059 atte+0.006 inte-0.035 | **accidental ↓** — outcome-blame suppressed | 0.3470 | 0.33 |
| HuggingFaceH4/zephyr-7b-beta | L3 worked_example | 0.084 | 0.466 | 0.178 | 0.498 | -0.288 | neut+0.021 acci-0.231 atte+0.033 inte-0.202 | **both** — attempted ↑, accidental ↓ (intent re-weighting) | 0.2678 | 0.24 |
| HuggingFaceH4/zephyr-7b-beta | L4 few_shot_adult | 0.041 | 0.336 | 0.056 | 0.371 | -0.280 | neut-0.023 acci-0.361 atte-0.090 inte-0.329 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) | 0.2111 | 0.14 |
| HuggingFaceH4/zephyr-7b-beta | L5 intent_principle | 0.022 | 0.460 | 0.123 | 0.548 | -0.337 | neut-0.041 acci-0.238 atte-0.023 inte-0.152 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) | 0.2721 | 0.30 |
| HuggingFaceH4/zephyr-7b-beta | L6 ABL_worked_only | 0.065 | 0.441 | 0.153 | 0.468 | -0.287 | neut+0.001 acci-0.257 atte+0.007 inte-0.232 | **accidental ↓** — outcome-blame suppressed | 0.2808 | 0.30 |
| HuggingFaceH4/zephyr-7b-beta | L7 ABL_fewshot_only | 0.040 | 0.407 | 0.075 | 0.401 | -0.331 | neut-0.023 acci-0.291 atte-0.071 inte-0.299 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) | 0.2345 | 0.15 |
| HuggingFaceH4/zephyr-7b-beta | L8 ABL_principle_only | 0.065 | 0.634 | 0.253 | 0.686 | -0.382 | neut+0.002 acci-0.063 atte+0.107 inte-0.014 | **both** — attempted ↑, accidental ↓ (intent re-weighting) | 0.3216 | 0.22 |
| Qwen/Qwen2.5-14B-Instruct | L1 baseline | 0.094 | 0.646 | 0.308 | 0.679 | -0.338 | neut+0.000 acci+0.000 atte+0.000 inte+0.000 | no gain | 0.3366 | 0.29 |
| Qwen/Qwen2.5-14B-Instruct | L2 belief_cue | 0.073 | 0.458 | 0.321 | 0.568 | -0.138 | neut-0.022 acci-0.187 atte+0.013 inte-0.111 | **accidental ↓** — outcome-blame suppressed | 0.2787 | 0.22 |
| Qwen/Qwen2.5-14B-Instruct | L3 worked_example | 0.062 | 0.373 | 0.243 | 0.485 | -0.129 | neut-0.033 acci-0.273 atte-0.064 inte-0.194 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) | 0.2157 | 0.20 |
| Qwen/Qwen2.5-14B-Instruct | L4 few_shot_adult | 0.030 | 0.452 | 0.246 | 0.721 | -0.207 | neut-0.064 acci-0.193 atte-0.062 inte+0.042 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added | 0.3357 | 0.38 |
| Qwen/Qwen2.5-14B-Instruct | L5 intent_principle | 0.016 | 0.331 | 0.335 | 0.673 | +0.004 | neut-0.078 acci-0.314 atte+0.027 inte-0.006 | **both** — attempted ↑, accidental ↓ (intent re-weighting) | 0.3215 | 0.38 |
| Qwen/Qwen2.5-14B-Instruct | L6 ABL_worked_only | 0.047 | 0.390 | 0.192 | 0.464 | -0.198 | neut-0.047 acci-0.256 atte-0.116 inte-0.215 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) | 0.2285 | 0.23 |
| Qwen/Qwen2.5-14B-Instruct | L7 ABL_fewshot_only | 0.024 | 0.698 | 0.196 | 0.771 | -0.502 | neut-0.070 acci+0.052 atte-0.112 inte+0.092 | no gain | 0.3889 | 0.48 |
| Qwen/Qwen2.5-14B-Instruct | L8 ABL_principle_only | 0.051 | 0.289 | 0.285 | 0.378 | -0.004 | neut-0.043 acci-0.357 atte-0.023 inte-0.301 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) | 0.1761 | 0.20 |
| allenai/Llama-3.1-Tulu-3-8B | L1 baseline | 0.145 | 0.709 | 0.312 | 0.695 | -0.398 | neut+0.000 acci+0.000 atte+0.000 inte+0.000 | no gain | 0.2894 | 0.03 |
| allenai/Llama-3.1-Tulu-3-8B | L2 belief_cue | 0.113 | 0.625 | 0.289 | 0.638 | -0.335 | neut-0.032 acci-0.084 atte-0.022 inte-0.057 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) | 0.2744 | 0.03 |
| allenai/Llama-3.1-Tulu-3-8B | L3 worked_example | 0.091 | 0.484 | 0.233 | 0.502 | -0.251 | neut-0.054 acci-0.225 atte-0.078 inte-0.193 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) | 0.2251 | 0.04 |
| allenai/Llama-3.1-Tulu-3-8B | L4 few_shot_adult | 0.113 | 0.681 | 0.283 | 0.658 | -0.397 | neut-0.032 acci-0.029 atte-0.028 inte-0.037 | no gain | 0.2843 | 0.02 |
| allenai/Llama-3.1-Tulu-3-8B | L5 intent_principle | 0.095 | 0.594 | 0.269 | 0.613 | -0.325 | neut-0.050 acci-0.115 atte-0.042 inte-0.082 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) | 0.2576 | 0.02 |
| allenai/Llama-3.1-Tulu-3-8B | L6 ABL_worked_only | 0.075 | 0.497 | 0.220 | 0.512 | -0.277 | neut-0.070 acci-0.212 atte-0.091 inte-0.183 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) | 0.2390 | 0.05 |
| allenai/Llama-3.1-Tulu-3-8B | L7 ABL_fewshot_only | 0.125 | 0.748 | 0.309 | 0.721 | -0.439 | neut-0.019 acci+0.039 atte-0.002 inte+0.026 | no gain | 0.3104 | 0.04 |
| allenai/Llama-3.1-Tulu-3-8B | L8 ABL_principle_only | 0.144 | 0.600 | 0.375 | 0.707 | -0.224 | neut-0.001 acci-0.109 atte+0.064 inte+0.011 | **both** — attempted ↑, accidental ↓ (intent re-weighting) | 0.2549 | 0.02 |
| allenai/OLMo-2-1124-7B-Instruct | L1 baseline | 0.094 | 0.904 | 0.217 | 0.888 | -0.687 | neut+0.000 acci+0.000 atte+0.000 inte+0.000 | no gain | 0.4120 | 0.35 |
| allenai/OLMo-2-1124-7B-Instruct | L2 belief_cue | 0.039 | 0.799 | 0.150 | 0.759 | -0.649 | neut-0.055 acci-0.105 atte-0.067 inte-0.128 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) | 0.4030 | 0.34 |
| allenai/OLMo-2-1124-7B-Instruct | L3 worked_example | 0.021 | 0.547 | 0.065 | 0.514 | -0.482 | neut-0.074 acci-0.357 atte-0.153 inte-0.374 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) | 0.3220 | 0.32 |
| allenai/OLMo-2-1124-7B-Instruct | L4 few_shot_adult | 0.053 | 0.851 | 0.145 | 0.818 | -0.706 | neut-0.041 acci-0.052 atte-0.072 inte-0.070 | no gain | 0.4085 | 0.27 |
| allenai/OLMo-2-1124-7B-Instruct | L5 intent_principle | 0.045 | 0.530 | 0.096 | 0.497 | -0.434 | neut-0.050 acci-0.374 atte-0.121 inte-0.391 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) | 0.2745 | 0.11 |
| allenai/OLMo-2-1124-7B-Instruct | L6 ABL_worked_only | 0.035 | 0.682 | 0.089 | 0.643 | -0.593 | neut-0.059 acci-0.222 atte-0.128 inte-0.245 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) | 0.3445 | 0.24 |
| allenai/OLMo-2-1124-7B-Instruct | L7 ABL_fewshot_only | 0.130 | 0.931 | 0.292 | 0.909 | -0.640 | neut+0.035 acci+0.028 atte+0.074 inte+0.021 | **attempted ↑** — intent-blame added; all four cells move the same way (compression) | 0.4027 | 0.31 |
| allenai/OLMo-2-1124-7B-Instruct | L8 ABL_principle_only | 0.075 | 0.680 | 0.244 | 0.692 | -0.436 | neut-0.019 acci-0.224 atte+0.027 inte-0.195 | **both** — attempted ↑, accidental ↓ (intent re-weighting) | 0.3437 | 0.11 |
| mistralai/Mistral-7B-Instruct-v0.3 | L1 baseline | 0.175 | 0.762 | 0.302 | 0.755 | -0.460 | neut+0.000 acci+0.000 atte+0.000 inte+0.000 | no gain | 0.3357 | 0.14 |
| mistralai/Mistral-7B-Instruct-v0.3 | L2 belief_cue | 0.110 | 0.722 | 0.281 | 0.757 | -0.442 | neut-0.065 acci-0.040 atte-0.021 inte+0.002 | no gain | 0.3416 | 0.19 |
| mistralai/Mistral-7B-Instruct-v0.3 | L3 worked_example | 0.151 | 0.492 | 0.226 | 0.507 | -0.266 | neut-0.024 acci-0.270 atte-0.076 inte-0.247 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) | 0.2641 | 0.09 |
| mistralai/Mistral-7B-Instruct-v0.3 | L4 few_shot_adult | 0.107 | 0.659 | 0.236 | 0.671 | -0.423 | neut-0.068 acci-0.104 atte-0.066 inte-0.084 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) | 0.2810 | 0.05 |
| mistralai/Mistral-7B-Instruct-v0.3 | L5 intent_principle | 0.066 | 0.636 | 0.238 | 0.684 | -0.397 | neut-0.110 acci-0.127 atte-0.064 inte-0.071 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) | 0.2973 | 0.14 |
| mistralai/Mistral-7B-Instruct-v0.3 | L6 ABL_worked_only | 0.218 | 0.544 | 0.290 | 0.550 | -0.254 | neut+0.043 acci-0.219 atte-0.013 inte-0.205 | **accidental ↓** — outcome-blame suppressed | 0.2353 | 0.02 |
| mistralai/Mistral-7B-Instruct-v0.3 | L7 ABL_fewshot_only | 0.123 | 0.725 | 0.246 | 0.726 | -0.480 | neut-0.052 acci-0.037 atte-0.056 inte-0.028 | no gain | 0.3105 | 0.05 |
| mistralai/Mistral-7B-Instruct-v0.3 | L8 ABL_principle_only | 0.360 | 0.754 | 0.543 | 0.789 | -0.211 | neut+0.185 acci-0.008 atte+0.241 inte+0.034 | **attempted ↑** — intent-blame added | 0.2385 | 0.04 |
| unsloth/gemma-2-9b-it | L1 baseline | 0.173 | 0.814 | 0.439 | 0.807 | -0.375 | neut+0.000 acci+0.000 atte+0.000 inte+0.000 | no gain | 0.3564 | 0.27 |
| unsloth/gemma-2-9b-it | L2 belief_cue | 0.114 | 0.704 | 0.471 | 0.759 | -0.234 | neut-0.059 acci-0.110 atte+0.031 inte-0.048 | **both** — attempted ↑, accidental ↓ (intent re-weighting) | 0.3348 | 0.22 |
| unsloth/gemma-2-9b-it | L3 worked_example | 0.072 | 0.541 | 0.305 | 0.619 | -0.236 | neut-0.100 acci-0.273 atte-0.134 inte-0.188 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) | 0.3055 | 0.21 |
| unsloth/gemma-2-9b-it | L4 few_shot_adult | 0.060 | 0.624 | 0.293 | 0.697 | -0.332 | neut-0.113 acci-0.190 atte-0.147 inte-0.110 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) | 0.3346 | 0.28 |
| unsloth/gemma-2-9b-it | L5 intent_principle | 0.046 | 0.557 | 0.292 | 0.665 | -0.265 | neut-0.127 acci-0.257 atte-0.147 inte-0.142 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) | 0.3547 | 0.36 |
| unsloth/gemma-2-9b-it | L6 ABL_worked_only | 0.111 | 0.553 | 0.310 | 0.602 | -0.242 | neut-0.062 acci-0.261 atte-0.129 inte-0.205 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) | 0.2783 | 0.14 |
| unsloth/gemma-2-9b-it | L7 ABL_fewshot_only | 0.074 | 0.729 | 0.273 | 0.754 | -0.456 | neut-0.099 acci-0.085 atte-0.166 inte-0.053 | no gain | 0.3598 | 0.32 |
| unsloth/gemma-2-9b-it | L8 ABL_principle_only | 0.070 | 0.460 | 0.265 | 0.574 | -0.196 | neut-0.103 acci-0.354 atte-0.175 inte-0.233 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) | 0.3015 | 0.22 |

## Non-cumulative ablation (secondary)

L1–L5 are cumulative, so L5 confounds the explicit principle with repair of whatever L4 did, and no cumulative level attributes the effect to a component. Each cell below is that component alone against the same L1 baseline. L2 already is "instruction alone" and is reused rather than re-run. **This is a secondary attribution analysis**: the pre-registered verdict stays the cumulative L5 column above, since these cells were designed after seeing run 1.

| model | instruction alone (L2) | worked example alone (L6) | few-shot alone (L7) | principle alone (L8) | Σ parts | L5 cumulative | reading |
|---|---:|---:|---:|---:|---:|---:|---|
| HuggingFaceH4/zephyr-7b-beta | +0.064 | +0.264 | +0.220 | +0.170 | +0.719 | +0.215 | sub-additive: components interfere |
| Qwen/Qwen2.5-14B-Instruct | +0.201 | +0.140 | -0.164 | +0.334 | +0.511 | +0.342 | sub-additive: components interfere |
| allenai/Llama-3.1-Tulu-3-8B | +0.062 | +0.121 | -0.042 | +0.173 | +0.315 | +0.073 | principle alone beats the full stack — escalation subtracts |
| allenai/OLMo-2-1124-7B-Instruct | +0.038 | +0.094 | +0.047 | +0.250 | +0.429 | +0.253 | sub-additive: components interfere |
| mistralai/Mistral-7B-Instruct-v0.3 | +0.019 | +0.206 | -0.019 | +0.249 | +0.455 | +0.063 | principle alone beats the full stack — escalation subtracts |
| unsloth/gemma-2-9b-it | +0.141 | +0.132 | -0.081 | +0.179 | +0.372 | +0.110 | principle alone beats the full stack — escalation subtracts |

Δ vs L1, paired over scenario groups. Σ parts is the arithmetic sum of the four single-component shifts and is a descriptive additivity reference, not a prediction any model of the effect entails.

### Where the principle-alone gain comes from

The attribution differs between the stack and the principle on its own, which is the most consequential thing the ablation shows. Same columns as the P6 guard, restricted to L5 (full stack) against L8 (principle alone).

| model | L5 Δattempted | L5 Δaccidental | L8 Δattempted | L8 Δaccidental | L8 mechanism |
|---|---:|---:|---:|---:|---|
| HuggingFaceH4/zephyr-7b-beta | -0.023 | -0.238 | +0.107 | -0.063 | **both** — attempted ↑, accidental ↓ (intent re-weighting) |
| Qwen/Qwen2.5-14B-Instruct | +0.027 | -0.314 | -0.023 | -0.357 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) |
| allenai/Llama-3.1-Tulu-3-8B | -0.042 | -0.115 | +0.064 | -0.109 | **both** — attempted ↑, accidental ↓ (intent re-weighting) |
| allenai/OLMo-2-1124-7B-Instruct | -0.121 | -0.374 | +0.027 | -0.224 | **both** — attempted ↑, accidental ↓ (intent re-weighting) |
| mistralai/Mistral-7B-Instruct-v0.3 | -0.064 | -0.127 | +0.241 | -0.008 | **attempted ↑** — intent-blame added |
| unsloth/gemma-2-9b-it | -0.147 | -0.257 | -0.175 | -0.354 | **accidental ↓ only** — outcome-blame suppressed, no intent-blame added; all four cells move the same way (compression) |

## Relation to W3 (steering)

Steering and prompting are different interventions on different quantities and share no common effect-size scale, so the two are not divided into each other here. The defensible statement is qualitative: **the intent representation is inert to residual-stream intervention at the depths where intent is resolvable (W3, |Δcontrast| ≤ 0.015), while the same contrast moves substantially under in-context instruction. That places the blockage downstream of the representation and upstream of the output** — the rating computation can be re-pointed by the input but not by editing the vector the probe reads.

Descriptively, the largest in-context shift observed (+0.342) sits close to the range the W3 *outcome* direction produced (0.232–0.259) as a positive control. That is a coincidence of magnitude worth flagging and nothing is built on it: the two numbers come from different manipulations, and the outcome-direction figure carries its own compression caveat. Its only use is as a reminder that shifts of this size are well inside what the design resolves.

## What this does and does not license

- The levels differ only in added text. Weights, decoding (logprob-EV over rating digits, deterministic), items, templates, scale normalisation and the contrast estimator are identical across levels, so the level effect is not confounded with the measurement.

- The L3/L4 example vignettes are held out: novel content (climbing gym, print shop) that appears nowhere in the master, so the scaffolding cannot teach a test item. The L4 labels are the Young 2007 adult profile mapped onto each template's own scale; they are used only to write the examples and are never a target in the analysis.

- A contrast gain is not automatically intent re-weighting. Read the attribution column: where the gain is `accidental` falling, the intervention removed blame from accidental harm rather than adding it to attempted harm, and where all four cells slide together the movement is compression. Both are real changes to the judgment and neither is evidence that the model started consulting intent.

- This is an in-context result. It says nothing about whether fine-tuning could move the contrast, and a prompt-level shift does not establish that the model is using the same intent representation the probe reads — only that the input can change the weighting. Establishing that it is the same representation would require the W3 intervention to work, which it does not.

