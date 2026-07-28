# Derivation of the exclusion floors (J4)

## Why the old value had to go

`EFFECT_FLOOR = 0.05` in `23_build_intent_reliance_summary.py` was justified in its
own comment as tuned against "the known degenerate cases (Mistral-7B, Zephyr-7B)".
Those models were not degenerate. The digit-token collapse in the SentencePiece
tokenizers returned the scale midpoint for every item, so their coefficients were
near zero by construction. The floor was calibrated against a bug, and it was still
gating the anchor comparison after the bug was fixed.

## Two floors, because there are two quantities

The request was to read the floor off the `rating_std` distribution. `rating_std` is a
dispersion in rating units and `|b_intent| + |b_outcome|` is a sum of regression
coefficients, so one cannot be read off the other. Both are derived, separately.

### 1. Engagement floor on `rating_std` = **0.2191**

Sorted across all 20 models post-fix, the largest gap runs from 0.1777 to 0.2604
(width 0.0827); the floor is placed at its midpoint. Models below it are not
varying their response to the stimuli, so no ratio computed from them is meaningful.

- excluded by the derived floor: 14 of 20 models
- excluded by the old 0.05: 9 of 20 models

### 2. Effect floor on `|b_intent| + |b_outcome|` = **0.0988**

This is the statistic `EFFECT_FLOOR` actually gates, so it is calibrated against the
magnitude that statistic reaches under noise. Condition labels are permuted within
scenario group -- never across -- because the 4 or 8 cells of a group share nearly all
their text, and a global shuffle would break that dependency and give a null with the
wrong variance. This is the same permutation scheme `02_probe.py` uses for its probe
null.

- pooled null q95: **0.0988**  (51400 draws)
- pooled null q99: 0.1439
- old hand-set value: 0.05

A floor set at a null quantile has a stated meaning: values below it are reached by
chance at least that often when the labels carry no information.

### A global constant is the wrong shape for this floor

The per-model nulls in the CSV span an order of magnitude, from about 0.005 for
gemma-2-9b base to about 0.145 for zephyr-7b-beta, because the null magnitude of a
coefficient sum scales with how much the model varies its ratings at all. Any single
constant is therefore too strict for low-variance models and too lenient for
high-variance ones. The pooled q95 of 0.0988 is set mostly by the high-variance instruct models, and the old 0.05
sat below the null of several of them -- meaning it was admitting template estimates
that permutation reaches by chance.

**Recommendation: replace the scalar with the per-model permutation test.** A template
enters the average if its `|b_intent| + |b_outcome|` exceeds that model's own null
q95. That is a per-model significance test rather than a shared
cutoff, and it cannot be re-tuned by which models happen to be in the roster. The
pooled value above is retained only as a fallback for code paths needing one number.

The per-template outcomes are then aggregated with a binomial test rather than by
asking whether any single template clears. At a q95 threshold
each template clears with probability 0.05 under the null, so across
13 templates chance alone supplies about 0.65 of them; "at least one cleared" is not
evidence of anything. A model is called degenerate unless the number of clearing
templates is itself unlikely under that binomial (p < 0.05).

Models called degenerate on this criterion:

- unsloth/gemma-2-9b (1/13 templates, binomial p = 0.487)

This is a different and better-founded list than the one the old 0.05 produced, and
notably it no longer contains Mistral or Zephyr, the two models the old value was
built around.

## What this changes

See `check_c5_engagement_floor.csv` for the anchor comparison re-run at the derived
engagement floor alongside 0.05 and 0.10. The conclusions are stable across all three
only if the same models clear every value; where they do not, the CSV shows which
models move and the anchor counts change with them.

## Two uses of the floor — do not conflate them

`rating_std` correlates with contrast *magnitude*: models that vary their ratings more
also tend to show larger (usually more negative) contrasts. Filtering on `rating_std`
therefore selects on a variable adjacent to the outcome. That is fine for some
questions and wrong for others.

| Use | Floor applied? | Why |
| --- | --- | --- |
| Engagement / degenerate flags | **yes** — derived 0.2191 (with 0.05 / 0.10 sensitivity) | A near-constant rater has no estimable contrast; counting them as "below the youngest band" would treat non-engagement as a developmental claim. |
| Anchor counts ("at or below youngest") | **yes** — same floor | Same reason: the claim is about models that answered the task. |
| Correlations (ToM↔contrast, representation↔behavior, RSA convergence) | **no** — keep every model | Excluding on `rating_std` selects on the outcome's neighbor. Control for model type (and log-size) instead; see `TOM_VS_CONTRAST.md`. |

This separation is fixed here so switching the floor between analyses cannot be read as
floor-shopping. If a correlation is reported on a restricted set, that is a sensitivity
check and must be labelled as such.
