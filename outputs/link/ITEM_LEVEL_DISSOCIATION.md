# Item-level representation<->behaviour link (J2)

## Why this is the primary test now

The model-level link had one observation per model and 8 models. C6 recomputed it
without the effect floor and got r = -0.209, 95% CI about [-0.80, +0.58] -- an
interval consistent with a strong link in either direction and with none at all. No
amount of pipeline repair fixes that; n=8 cannot answer the question.

Here the unit is the scenario group, so each model contributes up to 53
observations, and the question is sharper: within a single model, does it use intent
more for the stories whose intent it represents more clearly?

## Measures

- **(a) intent decodability, per scenario group.** Mean out-of-fold signed margin of
  the intent probe at that model's peak intent layer. Folds are split on
  scenario_group, so a scenario is always scored by a probe that never trained on
  it. Signed, so positive is the correct side of the boundary.
- **(b) intent-use, per scenario group.** Two definitions:
  - `primary`: attempted - accidental, the headline diagonal contrast (48 groups).
  - `matched`: the intent effect holding outcome constant, averaging
    (intentional - accidental) and (attempted - neutral) (53 groups).

Correlations are bootstrapped over scenario groups. The pooled estimate z-scores
both axes within model before stacking, so it reflects within-model covariation
across scenarios rather than the between-model differences the old test rested on.

## Per-model result, primary definition

| model | peak intent layer | probe acc | r | 95% CI (bootstrap over groups) | n groups |
|---|---|---|---|---|---|
| OLMo-2-1124-7B | 16 | 0.849 | +0.065 | [-0.257, +0.362] | 48 |
| OLMo-2-1124-7B-Instruct | 16 | 0.850 | -0.119 | [-0.470, +0.220] | 48 |
| Qwen2.5-0.5B | 12 | 0.671 | +0.108 | [-0.145, +0.365] | 48 |
| Qwen2.5-0.5B-Instruct | 21 | 0.695 | +0.169 | [-0.107, +0.425] | 48 |
| Qwen2.5-1.5B | 13 | 0.739 | +0.299 | [+0.019, +0.516] | 48 |
| Qwen2.5-1.5B-Instruct | 20 | 0.755 | +0.008 | [-0.304, +0.314] | 48 |
| Qwen2.5-7B | 19 | 0.883 | -0.086 | [-0.374, +0.172] | 48 |
| Qwen2.5-7B-Instruct | 19 | 0.879 | +0.051 | [-0.299, +0.367] | 48 |

- models whose interval excludes zero: 1 of 8

## Pooled, model as a random effect

- slope (both axes z-scored within model): **+0.0619** (SE 0.1062, p = 0.56)
- observations: 384 scenario-group estimates over 8 models
- per-model r positive in 6 of 8 models, mean r = +0.062

- robustness, `matched` definition: slope +0.0291 (SE 0.0990, p = 0.769), 424 observations

The 95% interval on the pooled slope is [-0.146, +0.270] in
within-model SD units. Both axes are standardised, so the slope is the SD
change in intent-use per SD of intent decodability, and the interval rules out
anything larger than about 0.27 SD in either direction.

**This is an informative null, and that is the difference from C6.** The
model-level test spanned [-0.80, +0.58] and so excluded nothing; this interval
is narrow enough to exclude a moderate or large effect. The reading it supports
is that within a model, the scenarios whose intent is most clearly represented
are not the scenarios where intent is most used -- a dissociation between
representation and use, measured at the level where the two are comparable.

Two limits worth stating with it. The bound is on a LINEAR, MONOTONE relation
between probe margin and contrast; a threshold relation, where intent must
merely be present rather than strongly present, would not show up here. And
probe margin is a proxy for representational quality, not a measure of what the
model reads out downstream -- decodable by a linear probe is not the same as
used by the model. Causal steering at the peak intent layer is the test that
would close that gap.

## Status of the old model-level link

Retained as a footnote only, and labelled uninformative: r = -0.209, 95% CI
[-0.80, +0.58], n = 8. It is not evidence of absence and should not be cited as
a null.
