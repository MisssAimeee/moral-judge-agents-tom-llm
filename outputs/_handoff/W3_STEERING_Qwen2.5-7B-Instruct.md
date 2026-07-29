# W3 causal steering readout — Qwen2.5-7B-Instruct

Steering layer L19; unsteered contrast -0.2667; probe cv_acc at L19: intent 0.869, outcome 0.987; cos(dom, probe) for intent +0.354.

Baseline cells: neutral 0.311, accidental 0.803, attempted 0.539, intentional 0.761

## Effect sizes inside the coherent band

| direction | max |Δcontrast| | alphas tested (coherent) |
|---|---:|---|
| intent_dom | 0.2251 | -0.3, -0.2, -0.15, -0.1, -0.05, -0.025, +0.025, +0.05, +0.1, +0.15, +0.2, +0.3 |
| intent_probe | 0.0155 | -0.3, -0.2, -0.15, -0.1, -0.05, -0.025, +0.025, +0.05, +0.1, +0.15, +0.2, +0.3 |
| outcome_dom | 0.2592 | -0.3, -0.2, -0.15, -0.1, -0.05, -0.025, +0.025, +0.05, +0.1, +0.15, +0.2, +0.3 |
| outcome_probe | 0.2017 | -0.3, -0.2, -0.15, -0.1, -0.05, -0.025, +0.025, +0.05, +0.1, +0.15, +0.2, +0.3 |
| random0 | 0.0207 | -0.3, -0.2, -0.15, -0.1, -0.05, -0.025, +0.025, +0.05, +0.1, +0.15, +0.2, +0.3 |
| random1 | 0.0534 | -0.3, -0.2, -0.15, -0.1, -0.05, -0.025, +0.025, +0.05, +0.1, +0.15, +0.2, +0.3 |

## Pre-specified predictions

- **P1 direction specificity**: intent max |Δ| = 0.2251 vs control max |Δ| = 0.2592 → NOT SUPPORTED
- **P2 dose-response (intent_dom)**: slope +0.6219 per unit α, monotone=no
- **P2 dose-response (intent_probe)**: slope -0.0254 per unit α, monotone=no
- **P3 method agreement**: cos(intent_dom, intent_probe) = +0.354; max |Δ| dom 0.2251 vs probe 0.0155
- **P4 coherence (intent directions)**: coherent band [-0.3000, +0.3000] (α in units of the typical residual norm); all effect sizes above are computed inside it. Manual read: `w3_generations_Qwen2.5-7B-Instruct.txt`.

## Manipulation check: did the intervention move the representation?

A flat contrast is only evidence about the representation if the intervention demonstrably changed what the probe reads. Each cell below re-runs the intent probe — fitted on UNSTEERED activations, grouped CV, never on the activations it scores — on the steered activations, at the largest coherent |α| for that direction.

| probe layer | position | direction | α | intent acc unsteered → steered | margin shift (SD) | Δcontrast |
|---|---|---|---:|---:|---:|---:|
| L15 | upstream (cannot be affected) | intent_dom | -0.3 | 0.658 → 0.658 (+0.000) | +0.00 | -0.1356 |
| L15 | upstream (cannot be affected) | intent_probe | -0.3 | 0.658 → 0.658 (+0.000) | +0.00 | +0.0155 |
| L15 | upstream (cannot be affected) | outcome_dom | -0.3 | 0.658 → 0.658 (+0.000) | +0.00 | +0.1080 |
| L15 | upstream (cannot be affected) | outcome_probe | -0.3 | 0.658 → 0.658 (+0.000) | +0.00 | +0.2017 |
| L15 | upstream (cannot be affected) | random0 | -0.3 | 0.658 → 0.658 (+0.000) | +0.00 | -0.0089 |
| L15 | upstream (cannot be affected) | random1 | -0.3 | 0.658 → 0.658 (+0.000) | +0.00 | +0.0534 |
| L19 | steering site | intent_dom | -0.3 | 0.869 → 0.869 (+0.000) | +0.00 | -0.1356 |
| L19 | steering site | intent_probe | -0.3 | 0.869 → 0.869 (+0.000) | +0.00 | +0.0155 |
| L19 | steering site | outcome_dom | -0.3 | 0.869 → 0.869 (+0.000) | +0.00 | +0.1080 |
| L19 | steering site | outcome_probe | -0.3 | 0.869 → 0.869 (+0.000) | +0.00 | +0.2017 |
| L19 | steering site | random0 | -0.3 | 0.869 → 0.869 (+0.000) | +0.00 | -0.0089 |
| L19 | steering site | random1 | -0.3 | 0.869 → 0.869 (+0.000) | +0.00 | +0.0534 |
| L23 | downstream | intent_dom | -0.3 | 0.899 → 0.500 (-0.399) | -4.46 | -0.1356 |
| L23 | downstream | intent_probe | -0.3 | 0.899 → 0.500 (-0.399) | -6.39 | +0.0155 |
| L23 | downstream | outcome_dom | -0.3 | 0.899 → 0.527 (-0.372) | -1.63 | +0.1080 |
| L23 | downstream | outcome_probe | -0.3 | 0.899 → 0.812 (-0.087) | -0.02 | +0.2017 |
| L23 | downstream | random0 | -0.3 | 0.899 → 0.789 (-0.111) | +0.25 | -0.0089 |
| L23 | downstream | random1 | -0.3 | 0.899 → 0.815 (-0.084) | -0.38 | +0.0534 |
| L28 | final layer | intent_dom | -0.3 | 0.866 → 0.500 (-0.366) | -3.70 | -0.1356 |
| L28 | final layer | intent_probe | -0.3 | 0.866 → 0.500 (-0.366) | -4.48 | +0.0155 |
| L28 | final layer | outcome_dom | -0.3 | 0.866 → 0.510 (-0.356) | -2.42 | +0.1080 |
| L28 | final layer | outcome_probe | -0.3 | 0.866 → 0.644 (-0.222) | -0.91 | +0.2017 |
| L28 | final layer | random0 | -0.3 | 0.866 → 0.752 (-0.114) | -0.40 | -0.0089 |
| L28 | final layer | random1 | -0.3 | 0.866 → 0.846 (-0.020) | -0.17 | +0.0534 |

Two instrument checks with known answers, both of which must read exactly zero. **Layers below the injection site cannot be affected by it.** **The injection layer itself is captured before the injection**: transformers 5.x collects hidden states with a hook registered before ours, and PyTorch runs forward hooks in registration order, so `hidden_states[L]` is the pre-injection value (verified directly: max|Δ| is 0 at `hidden_states[L]` and ~|v| from `hidden_states[L+1]` on). Both read zero here, which confirms the hook fires where it claims to. Every informative row is therefore a downstream or final-layer row, where the injected signal has had to survive the remaining blocks. Full grid: `w3_manipulation_Qwen2.5-7B-Instruct.csv`; figure: `w3_manipulation_Qwen2.5-7B-Instruct.png`.


Generated by `code/experiments/48_w3_causal_steering.py`. Pre-registration: `W3_PRESPEC.md`.
