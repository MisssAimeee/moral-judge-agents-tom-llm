# Reviewer-requested robustness checks

Each check re-runs a primary claim under a stricter specification. Descriptive output; no claim is restated here that the numbers do not support.

## C1 — Variance decomposition with model identity absorbed

Observations: 120 = 20 models x 6 factorial prompts.

| Specification | wording | construct | wording x construct | model | residual |
| --- | --- | --- | --- | --- | --- |
| pooled (as originally specified) | 0.0000 | 0.0005 | 0.0041 | — | 0.9953 |
| model as fixed factor | 0.0000 | 0.0005 | 0.0041 | 0.9428 | 0.0525 |
| model-centered contrasts | 0.0003 | 0.0094 | 0.0723 | — | 0.9180 |

- Within-model SD of the contrast across the 6 prompts: median 0.0287, mean 0.0448.
- Between-model SD of the mean contrast: 0.221.
- Within/between variance ratio: 0.041.

Per-model spread: `check_c1_within_model_variance.csv`.

## C5 — Pre-specified engagement floor on rating_std

Pre-registered floor: **rating_std >= 0.05** on the normalised 0–1 response scale. The floor is a property of the response distribution, not of the effect, so it cannot bias the direction of the contrast. Non-engaged models are excluded, not counted as failures.

| rating_std floor | models engaged | excluded | text-reported | digitized Naughty | punish |
| --- | --- | --- | --- | --- | --- |
| 0.02 | 18 | 2 | 10/18 | 18/18 | 18/18 |
| 0.03 | 15 | 5 | 10/15 | 15/15 | 15/15 |
| 0.05 **(pre-registered)** | 11 | 9 | 10/11 | 11/11 | 11/11 |
| 0.1 | 9 | 11 | 9/9 | 9/9 | 9/9 |

Counts are models at or below the ages 4–5 band of each anchor.
Per-model values: `check_c5_engagement_floor.csv`.

## C6 — Representation-vs-behavior link, all probed models recovered

The effect floor in `23_build_intent_reliance_summary.py` withheld an index from models whose effect was small, which dropped them from the link entirely. Here the index is computed for every probed model with no floor, so nothing is silently missing.

- Probed models: 8; paired with a behavioral index: 8.
- Pearson r = **-0.2085**, 95% CI [-0.7962, 0.5817], n = 8.
- **UNINFORMATIVE — the interval spans strong negative to strong positive, so no effect and a large effect in either direction are all consistent with these data. This is not evidence of absence.**
- The point estimate also changes sign relative to the floored 5-model version (r = +0.561), which is itself a reason to treat neither number as an estimate of anything.

Per-model values: `check_c6_link_all_models.csv`.

## C7 — Flip rate conditioned on effect size

Pre-registered null threshold: **|mean contrast| <= 0.02** counts as null. A sign flip requires a signal whose sign can flip; models at zero are reported as null, not as fragile.

| null threshold | non-null models | flips among non-null | flip rate | null models | flips among null |
| --- | --- | --- | --- | --- | --- |
| 0.01 | 15 | 0 | 0/15 | 5 | 5 |
| 0.02 **(pre-registered)** | 14 | 0 | 0/14 | 6 | 5 |
| 0.05 | 13 | 0 | 0/13 | 7 | 5 |
| 0.1 | 11 | 0 | 0/11 | 9 | 5 |

Per-model values: `check_c7_flip_rate_conditioned.csv`.
