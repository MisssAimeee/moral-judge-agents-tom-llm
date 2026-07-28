# W1 mixed-effects 2x2 with interaction (J3)

Model fitted per model on item-level ratings:

    blame ~ intent * outcome + (1|scenario_group) + (1|story_id)

story_id is nested within scenario_group and enters as a within-group variance
component. Coefficients are treatment-coded, so `b_intent` is the simple effect of
intent at outcome=0, `b_outcome` the simple effect of outcome at intent=0, and
`b_interaction` the extra effect of outcome when intent is present.

## Human reference

Computed from the Young et al. 2007 normalised cell means (neutral 0.033, accidental 0.267, attempted 0.933, intentional 0.967):

    b_intent      = +0.900
    b_outcome     = +0.234
    b_interaction = -0.200

The interaction is negative because a harmful outcome adds little once intent is
present (0.933 -> 0.967) but a great deal when it is absent (0.033 -> 0.267).

## Counts

- models with an estimable interaction: 20 of 20
- significantly negative (same sign as humans, p<0.05): 11
- significantly positive (opposite sign to humans, p<0.05): 0
- not distinguishable from zero: 9

## The matching sign is not a matching computation

Read the sign count above together with the main effects, not on its own. A negative
interaction means sub-additivity: the second factor adds less once the first is
present. On a bounded 0-1 scale that can happen because EITHER factor has already
used up the scale, and the two cases mean opposite things.

- Humans: `b_intent` = +0.900 and `b_outcome` = +0.234. Intent nearly saturates blame
  by itself, so outcome has little left to add. The sub-additivity is intent-driven.
- These models: `b_intent` is small and `b_outcome` is large, the reverse ordering.
  Their sub-additivity is outcome-driven -- outcome uses up the scale and intent has
  little left to add.

`saturating_factor` in the CSV records which main effect is larger per model: 18 of 20 are outcome-saturating against the human pattern of intent-saturating.

Human interaction from cell means: (0.967−0.933)−(0.267−0.033) = -0.200. Several models
approximate that coefficient. Same sign with the **opposite cell order**
(accidental > attempted) is not human-likeness — see `cell_order`.

Cell-order counts (primary spec): matches_human=0, inverted=14, tied=6.

## Interaction terms with cell means (primary specification)

| model | b_int | b_out | b_ixo | SE | p | neu | acc | att | int | att−acc | cell order | sat |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| unsloth/gemma-2-9b-it | +0.236 | +0.645 | -0.252 | 0.031 | 3.3e-16 | 0.203 | 0.841 | 0.439 | 0.825 | -0.402 | inverted | outcome |
| unsloth/Meta-Llama-3_1-8B-Instruct | +0.154 | +0.357 | -0.183 | 0.019 | 5.38e-22 | 0.365 | 0.716 | 0.518 | 0.687 | -0.198 | inverted | outcome |
| Qwen/Qwen2_5-14B-Instruct | +0.186 | +0.563 | -0.165 | 0.024 | 3.91e-12 | 0.126 | 0.684 | 0.312 | 0.706 | -0.372 | inverted | outcome |
| allenai/Llama-3_1-Tulu-3-8B | +0.144 | +0.534 | -0.161 | 0.021 | 8.09e-15 | 0.176 | 0.705 | 0.320 | 0.688 | -0.385 | inverted | outcome |
| mistralai/Mistral-7B-Instruct-v0_3 | +0.126 | +0.593 | -0.135 | 0.019 | 1.01e-12 | 0.172 | 0.759 | 0.298 | 0.749 | -0.461 | inverted | outcome |
| Qwen/Qwen2_5-7B-Instruct | +0.104 | +0.347 | -0.126 | 0.017 | 3.21e-14 | 0.304 | 0.653 | 0.408 | 0.630 | -0.245 | inverted | outcome |
| allenai/OLMo-2-1124-7B-Instruct | +0.093 | +0.735 | -0.110 | 0.024 | 3.18e-06 | 0.148 | 0.880 | 0.241 | 0.863 | -0.640 | inverted | outcome |
| HuggingFaceH4/zephyr-7b-beta | +0.091 | +0.630 | -0.087 | 0.026 | 0.000681 | 0.060 | 0.686 | 0.151 | 0.689 | -0.534 | inverted | outcome |
| Qwen/Qwen2_5-3B-Instruct | +0.042 | +0.294 | -0.054 | 0.015 | 0.000379 | 0.378 | 0.671 | 0.420 | 0.659 | -0.251 | inverted | outcome |
| Qwen/Qwen2_5-14B | +0.012 | +0.139 | -0.027 | 0.008 | 0.000935 | 0.217 | 0.355 | 0.229 | 0.340 | -0.126 | inverted | outcome |
| Qwen/Qwen2_5-7B | +0.008 | +0.058 | -0.017 | 0.004 | 6.68e-05 | 0.439 | 0.500 | 0.447 | 0.490 | -0.053 | inverted | outcome |
| Qwen/Qwen2_5-0_5B-Instruct | +0.008 | +0.057 | -0.016 | 0.012 | 0.212 | 0.199 | 0.255 | 0.208 | 0.248 | -0.047 | inverted | outcome |
| Qwen/Qwen2_5-3B | +0.003 | +0.052 | -0.010 | 0.005 | 0.0538 | 0.616 | 0.672 | 0.619 | 0.665 | -0.053 | inverted | outcome |
| allenai/OLMo-2-1124-7B | +0.006 | +0.008 | -0.006 | 0.004 | 0.201 | 0.218 | 0.226 | 0.224 | 0.226 | -0.001 | tied | outcome |
| Qwen/Qwen2_5-1_5B-Instruct | +0.012 | +0.184 | -0.005 | 0.008 | 0.57 | 0.597 | 0.780 | 0.609 | 0.788 | -0.171 | inverted | outcome |
| mistralai/Mistral-7B-v0_3 | +0.003 | +0.005 | -0.004 | 0.003 | 0.1 | 0.241 | 0.246 | 0.244 | 0.245 | -0.002 | tied | outcome |
| unsloth/Meta-Llama-3_1-8B | +0.007 | +0.004 | -0.004 | 0.006 | 0.488 | 0.237 | 0.240 | 0.244 | 0.243 | +0.004 | tied | intent |
| Qwen/Qwen2_5-1_5B | +0.000 | +0.012 | -0.004 | 0.005 | 0.437 | 0.291 | 0.300 | 0.291 | 0.296 | -0.009 | tied | outcome |
| Qwen/Qwen2_5-0_5B | +0.002 | +0.002 | -0.002 | 0.006 | 0.789 | 0.281 | 0.283 | 0.283 | 0.283 | -0.001 | tied | outcome |
| unsloth/gemma-2-9b | +0.002 | +0.001 | -0.001 | 0.008 | 0.864 | 0.105 | 0.106 | 0.107 | 0.107 | +0.001 | tied | intent |

| **HUMAN (Young 2007)** | +0.900 | +0.234 | -0.200 | - | - | 0.033 | 0.267 | 0.933 | 0.967 | +0.666 | matches_human | intent |

The `template_absorbed` rows in the CSV repeat every fit with prompt template as
a fixed factor. C1 showed model identity dominates the variance, so this checks
that the interaction is not an artefact of averaging over templates.
