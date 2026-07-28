# Reconciliation of the two sign-flip columns

## The apparent contradiction

Two files reported what looked like the same fact and disagreed:

- `outputs/stats/contrast_by_model.csv` said `sign_flips_across_prompts = True` for
  Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct.
- `outputs/analysis/prompt_factorial_sign_stability.csv` said `sign_stable = True` for the
  same two models.

## Cause: different template sets, neither one wrong

The two statistics never covered the same prompts.

- `06_stats.py` ranges over **every template the model was run on** — 13 of them, including
  `human_verbatim` (the original Young 2007 wording) and `para_blame10` (a 1-10 response
  scale). It sets the flag if the contrast is positive for any template and negative for any
  other.
- `33_prompt_factorial_analysis.py` ranges over the **6 factorial templates on the shared
  1-7 scale** only (`blame_w1/w2`, `wrong_w1/w2`, `punish_w1/w2`), because the factorial
  design requires one response scale to be comparable across cells.

A model can therefore hold its sign across the factorial set and still flip on a template
outside it, which is exactly what these two do.

## Fix

Both columns are renamed to carry their scope, so the names can no longer be read as
contradicting each other:

| file | old name | new name |
|---|---|---|
| `outputs/stats/contrast_by_model.csv` | `sign_flips_across_prompts` | `sign_flips_all_templates` |
| `outputs/stats/prompt_invariance_contrast.csv` | `sign_flips` | `sign_flips_all_templates` |
| `outputs/analysis/prompt_factorial_sign_stability.csv` | `sign_stable` | `sign_stable_factorial_1_7` |

Readers of both columns (`08_report.py`, `35_review_checks.py`) accept either spelling, so
older CSVs still parse.

## What the flips actually are

Both files were regenerated after the tokenizer fix. The two columns now agree for 18 of 20
models. The two remaining differences are the ones above, and in both cases the flip comes
from a single non-factorial template:

**Qwen2.5-0.5B-Instruct** — all 6 factorial templates negative (-0.0021 to -0.0865). The
flip is `para_blame10` at **+0.0003**.

**Qwen2.5-1.5B-Instruct** — all 6 factorial templates negative (-0.1824 to -0.2545). The
flip is `human_verbatim` at **+0.0596**.

The 0.5B case is a zero crossing on noise: +0.0003 on a 0-1 scale is not a positive
contrast, it is an absence of one, and calling it a sign flip overstates what happened. This
is the same point C7 makes by conditioning the flip rate on effect size — among models with
a contrast large enough to have a sign, the flip rate is 0 of 13.

The 1.5B case is a genuine single-template reversal, and worth keeping in view: it is the
`human_verbatim` template, the closest to the original human study, that goes the other way.
That is a real caveat about generalising from the factorial set to the original wording, not
a bookkeeping artefact. It is one template out of 13 and the effect is small relative to the
factorial estimates, but it should be reported rather than absorbed into an average.

## Recommended usage

- For prompt-robustness claims about the factorial design, cite `sign_stable_factorial_1_7`.
- For "does this model ever reverse anywhere in the battery", cite
  `sign_flips_all_templates`, and read it next to the magnitudes in
  `prompt_invariance_contrast.csv` so noise-level crossings are not counted as reversals.
- Do not report either as "the" sign-stability number without naming the template set.
