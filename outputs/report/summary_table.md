# Moral judgment: model vs human — summary

Adult human reference contrast = **+0.67** (intent-weighted). Developmental ladder: adult +0.67, child_8plus +0.46, child_6_7 +0.15, child_4_5 -0.14.

Contrast = blame(attempted) − blame(accidental). Positive = judges by **intent** (adult-like); negative = by **outcome** (young-child-like). CI = 95% bootstrap over scenarios.

| model | params | type | contrast [95% CI] | ≠0 | adult corr | adult RMSE | gap vs adult | nearest human | intent-reliance | prompt SD | sign flip |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **HUMAN adult** | — | human | +0.67 | — | 1.00 | — | 0.00 | adult | — | — | — |
| **HUMAN child_8plus** | — | human | +0.46 | — | — | — | 0.21 | child_8plus | — | — | — |
| **HUMAN child_6_7** | — | human | +0.15 | — | — | — | 0.52 | child_6_7 | — | — | — |
| **HUMAN child_4_5** | — | human | -0.14 | — | — | — | 0.81 | child_4_5 | — | — | — |
| Qwen7B | 7.0 | base | +0.02 [+0.01,+0.03] | yes | 0.135 | 0.431 | 0.644 | child_6_7 | 0.20 [+0.06,+0.47] | 0.072 | YES |
| Qwen3B | 3.0 | base | +0.01 [+0.00,+0.02] | yes | -0.517 | 0.432 | 0.658 | child_6_7 | 0.30 [+0.05,+0.89] | 0.075 | YES |
| Qwen1.5B | 1.5 | base | +0.01 [+0.00,+0.01] | yes | -0.143 | 0.45 | 0.659 | child_6_7 | 0.01 [+0.00,+0.08] | 0.02 | YES |
| mistralai_Mistral-7B-Instruct-v0_3 | 7.0 | instruct | +0.00 [+0.00,+0.00] | no | NA | 0.412 | 0.666 | child_4_5 | 0.00 [+0.00,+0.00] | 0.0 | no |
| mistralai_Mistral-7B-v0_3 | 7.0 | base | +0.00 [+0.00,+0.00] | no | NA | 0.412 | 0.666 | child_4_5 | 0.00 [+0.00,+0.00] | 0.0 | no |
| Qwen0.5B | 0.5 | base | -0.00 [-0.00,+0.00] | no | -0.338 | 0.452 | 0.666 | child_4_5 | 0.13 [+0.01,+0.78] | 0.01 | YES |
| allenai_OLMo-2-1124-7B | 7.0 | base | -0.02 [-0.03,-0.01] | yes | 0.43 | 0.514 | 0.682 | child_4_5 | 0.21 [+0.10,+0.39] | 0.018 | YES |
| Llama-3.1-8B | 8.0 | base | -0.02 [-0.03,-0.02] | yes | 0.37 | 0.432 | 0.688 | child_4_5 | 0.16 [+0.10,+0.22] | 0.034 | YES |
| Qwen_Qwen2_5-14B | 14.0 | base | -0.04 [-0.06,-0.02] | yes | 0.329 | 0.479 | 0.706 | child_4_5 | 0.13 [+0.05,+0.21] | 0.04 | no |
| Qwen0.5B-Instruct | 0.5 | instruct | -0.04 [-0.05,-0.03] | yes | 0.292 | 0.507 | 0.706 | child_4_5 | 0.11 [+0.07,+0.15] | 0.047 | YES |
| Qwen14B | 14.0 | base | -0.09 [-0.11,-0.08] | yes | 0.364 | 0.433 | 0.756 | child_4_5 | 0.16 [+0.11,+0.20] | 0.053 | no |
| Qwen1.5B-Instruct | 1.5 | instruct | -0.12 [-0.14,-0.10] | yes | 0.409 | 0.382 | 0.786 | child_4_5 | 0.19 [+0.16,+0.23] | 0.053 | no |
| Qwen_Qwen2_5-7B-Instruct | 7.0 | instruct | -0.24 [-0.28,-0.21] | yes | 0.307 | 0.392 | 0.908 | child_4_5 | 0.11 [+0.06,+0.15] | 0.08 | no |
| Qwen3B-Instruct | 3.0 | instruct | -0.25 [-0.29,-0.20] | yes | 0.425 | 0.386 | 0.913 | child_4_5 | 0.20 [+0.16,+0.25] | 0.093 | no |
| Qwen_Qwen2_5-14B-Instruct | 14.0 | instruct | -0.28 [-0.32,-0.24] | yes | 0.395 | 0.401 | 0.947 | child_4_5 | 0.18 [+0.14,+0.23] | 0.063 | no |
| Qwen7B-Instruct | 7.0 | instruct | -0.33 [-0.38,-0.28] | yes | 0.363 | 0.388 | 1.0 | child_4_5 | 0.15 [+0.11,+0.19] | 0.083 | no |
| Qwen14B-Instruct | 14.0 | instruct | -0.39 [-0.43,-0.35] | yes | 0.403 | 0.395 | 1.06 | child_4_5 | 0.19 [+0.15,+0.22] | 0.099 | no |

**Figures** (outputs/figures/): contrast_forest, profiles, prompt_invariance, size_vs_contrast, weights_scatter, pairwise_heatmap.
