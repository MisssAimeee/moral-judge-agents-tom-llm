# Reviewer-requested robustness checks

Each check re-runs a primary claim under a stricter specification. Descriptive output; no claim is restated here that the numbers do not support.

## C7 — Flip rate conditioned on effect size

Pre-registered null threshold: **|mean contrast| <= 0.02** counts as null. A sign flip requires a signal whose sign can flip; models at zero are reported as null, not as fragile.

| null threshold | non-null models | flips among non-null | flip rate | null models | flips among null |
| --- | --- | --- | --- | --- | --- |
| 0.01 | 15 | 0 | 0/15 | 5 | 5 |
| 0.02 **(pre-registered)** | 14 | 0 | 0/14 | 6 | 5 |
| 0.05 | 13 | 0 | 0/13 | 7 | 5 |
| 0.1 | 11 | 0 | 0/11 | 9 | 5 |

Per-model values: `check_c7_flip_rate_conditioned.csv`.
