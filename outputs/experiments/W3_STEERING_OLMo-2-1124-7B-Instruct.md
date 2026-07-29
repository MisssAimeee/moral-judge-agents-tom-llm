# W3 causal steering readout — OLMo-2-1124-7B-Instruct

Steering layer L16; unsteered contrast -0.4452; probe cv_acc at L16: intent 0.846, outcome 0.993; cos(dom, probe) for intent +0.514.

Baseline cells: neutral 0.119, accidental 0.761, attempted 0.310, intentional 0.740

## Effect sizes inside the coherent band

| direction | max |Δcontrast| | alphas tested (coherent) |
|---|---:|---|
| intent_dom | 0.0808 | -0.15, -0.1, -0.05, -0.025, +0.025, +0.05, +0.1, +0.15, +0.2 |
| intent_probe | 0.0128 | -0.3, -0.2, -0.15, -0.1, -0.05, -0.025, +0.025, +0.05, +0.1, +0.15, +0.2, +0.3 |
| outcome_dom | 0.2319 | -0.2, -0.15, -0.1, -0.05, -0.025, +0.025, +0.05, +0.1, +0.15, +0.2 |
| outcome_probe | 0.1612 | -0.2, -0.15, -0.1, -0.05, -0.025, +0.025, +0.05, +0.1, +0.15, +0.2 |
| random0 | 0.0727 | -0.2, -0.15, -0.1, -0.05, -0.025, +0.025, +0.05, +0.1, +0.15, +0.2, +0.3 |
| random1 | 0.1240 | -0.2, -0.15, -0.1, -0.05, -0.025, +0.025, +0.05, +0.1, +0.15, +0.2, +0.3 |

## Pre-specified predictions

- **P1 direction specificity**: intent max |Δ| = 0.0808 vs control max |Δ| = 0.2319 → NOT SUPPORTED
- **P2 dose-response (intent_dom)**: slope +0.3380 per unit α, monotone=yes
- **P2 dose-response (intent_probe)**: slope +0.0244 per unit α, monotone=no
- **P3 method agreement**: cos(intent_dom, intent_probe) = +0.514; max |Δ| dom 0.0808 vs probe 0.0128
- **P4 coherence (intent directions)**: coherent band [-0.3000, +0.3000] (α in units of the typical residual norm); all effect sizes above are computed inside it. Manual read: `w3_generations_OLMo-2-1124-7B-Instruct.txt`.

## Manipulation check: did the intervention move the representation?

A flat contrast is only evidence about the representation if the intervention demonstrably changed what the probe reads. Each cell below re-runs the intent probe — fitted on UNSTEERED activations, grouped CV, never on the activations it scores — on the steered activations, at the largest coherent |α| for that direction.

| probe layer | position | direction | α | intent acc unsteered → steered | margin shift (SD) | Δcontrast |
|---|---|---|---:|---:|---:|---:|
| L12 | upstream (cannot be affected) | intent_dom | +0.2 | 0.685 → 0.685 (+0.000) | +0.00 | +0.0808 |
| L12 | upstream (cannot be affected) | intent_probe | -0.3 | 0.685 → 0.685 (+0.000) | +0.00 | -0.0128 |
| L12 | upstream (cannot be affected) | outcome_dom | -0.2 | 0.685 → 0.685 (+0.000) | +0.00 | +0.1448 |
| L12 | upstream (cannot be affected) | outcome_probe | -0.2 | 0.685 → 0.685 (+0.000) | +0.00 | +0.0806 |
| L12 | upstream (cannot be affected) | random0 | +0.3 | 0.685 → 0.685 (+0.000) | +0.00 | -0.0134 |
| L12 | upstream (cannot be affected) | random1 | +0.3 | 0.685 → 0.685 (+0.000) | +0.00 | +0.1240 |
| L16 | steering site | intent_dom | +0.2 | 0.846 → 0.846 (+0.000) | +0.00 | +0.0808 |
| L16 | steering site | intent_probe | -0.3 | 0.846 → 0.846 (+0.000) | +0.00 | -0.0128 |
| L16 | steering site | outcome_dom | -0.2 | 0.846 → 0.846 (+0.000) | +0.00 | +0.1448 |
| L16 | steering site | outcome_probe | -0.2 | 0.846 → 0.846 (+0.000) | +0.00 | +0.0806 |
| L16 | steering site | random0 | +0.3 | 0.846 → 0.846 (+0.000) | +0.00 | -0.0134 |
| L16 | steering site | random1 | +0.3 | 0.846 → 0.846 (+0.000) | +0.00 | +0.1240 |
| L24 | downstream | intent_dom | +0.2 | 0.893 → 0.517 (-0.376) | +1.48 | +0.0808 |
| L24 | downstream | intent_probe | -0.3 | 0.893 → 0.500 (-0.393) | -3.18 | -0.0128 |
| L24 | downstream | outcome_dom | -0.2 | 0.893 → 0.745 (-0.148) | -0.40 | +0.1448 |
| L24 | downstream | outcome_probe | -0.2 | 0.893 → 0.839 (-0.054) | -0.28 | +0.0806 |
| L24 | downstream | random0 | +0.3 | 0.893 → 0.564 (-0.329) | -1.07 | -0.0134 |
| L24 | downstream | random1 | +0.3 | 0.893 → 0.627 (-0.265) | -1.02 | +0.1240 |
| L32 | final layer | intent_dom | +0.2 | 0.862 → 0.601 (-0.262) | +1.12 | +0.0808 |
| L32 | final layer | intent_probe | -0.3 | 0.862 → 0.500 (-0.362) | -3.02 | -0.0128 |
| L32 | final layer | outcome_dom | -0.2 | 0.862 → 0.691 (-0.171) | -0.43 | +0.1448 |
| L32 | final layer | outcome_probe | -0.2 | 0.862 → 0.775 (-0.087) | -0.41 | +0.0806 |
| L32 | final layer | random0 | +0.3 | 0.862 → 0.571 (-0.292) | -1.20 | -0.0134 |
| L32 | final layer | random1 | +0.3 | 0.862 → 0.654 (-0.208) | -0.93 | +0.1240 |

Two instrument checks with known answers, both of which must read exactly zero. **Layers below the injection site cannot be affected by it.** **The injection layer itself is captured before the injection**: transformers 5.x collects hidden states with a hook registered before ours, and PyTorch runs forward hooks in registration order, so `hidden_states[L]` is the pre-injection value (verified directly: max|Δ| is 0 at `hidden_states[L]` and ~|v| from `hidden_states[L+1]` on). Both read zero here, which confirms the hook fires where it claims to. Every informative row is therefore a downstream or final-layer row, where the injected signal has had to survive the remaining blocks. Full grid: `w3_manipulation_OLMo-2-1124-7B-Instruct.csv`; figure: `w3_manipulation_OLMo-2-1124-7B-Instruct.png`.


Generated by `code/experiments/48_w3_causal_steering.py`. Pre-registration: `W3_PRESPEC.md`.
