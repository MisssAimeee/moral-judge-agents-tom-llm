# Moral judgment: model vs human — summary

Adult human reference contrast = **+0.67** (intent-weighted). Developmental ladder: adult +0.67, child_8plus +0.46, child_6_7 +0.15, child_4_5 -0.14.

Contrast = blame(attempted) − blame(accidental). Positive = judges by **intent** (adult-like); negative = by **outcome** (young-child-like). CI = 95% bootstrap over scenarios.

| model | params | type | contrast [95% CI] | ≠0 | adult corr | adult RMSE | gap vs adult | nearest human | intent-reliance | prompt SD | sign flip |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **HUMAN adult** | — | human | +0.67 | — | 1.00 | — | 0.00 | adult | — | — | — |
| **HUMAN child_8plus** | — | human | +0.46 | — | — | — | 0.21 | child_8plus | — | — | — |
| **HUMAN child_6_7** | — | human | +0.15 | — | — | — | 0.52 | child_6_7 | — | — | — |
| **HUMAN child_4_5** | — | human | -0.14 | — | — | — | 0.81 | child_4_5 | — | — | — |
| claude-opus-4-6 | 500.0 | instruct | +0.09 [+0.01,+0.17] | yes | 0.873 | 0.269 | 0.573 | child_6_7 | 0.58 [+0.52,+0.63] | 0.055 | no |
| gemini-2_5-pro | 600.0 | instruct | -0.00 [-0.02,+0.02] | no | 0.791 | 0.352 | 0.666 | child_4_5 | 0.50 [+0.46,+0.55] | 0.029 | YES |
| kimi-k3 | nan | instruct | +0.00 [+0.00,+0.00] | no | NA | 0.412 | 0.666 | child_4_5 | 0.00 [+0.00,+0.00] | 0.0 | no |
| claude-sonnet-4-6 | 175.0 | instruct | -0.01 [-0.09,+0.08] | no | 0.823 | 0.271 | 0.672 | child_4_5 | 0.51 [+0.45,+0.57] | 0.069 | YES |
| gemini-2_5-flash | 32.0 | instruct | -0.06 [-0.14,+0.03] | no | 0.798 | 0.251 | 0.726 | child_4_5 | 0.48 [+0.43,+0.54] | 0.022 | no |
| claude-haiku-4-5-20251001 | 20.0 | instruct | -0.15 [-0.21,-0.09] | yes | 0.621 | 0.342 | 0.817 | child_4_5 | 0.35 [+0.28,+0.41] | 0.052 | no |
| gpt-4o-mini | 8.0 | instruct | -0.28 [-0.32,-0.24] | yes | 0.464 | 0.366 | 0.944 | child_4_5 | 0.23 [+0.20,+0.27] | 0.042 | no |
| gpt-4o | 200.0 | instruct | -0.38 [-0.44,-0.32] | yes | 0.441 | 0.392 | 1.044 | child_4_5 | 0.22 [+0.18,+0.27] | 0.035 | no |

**Figures** (outputs/figures/): contrast_forest, profiles, prompt_invariance, size_vs_contrast, weights_scatter, pairwise_heatmap.
