# ToM benchmark performance vs intent use in moral judgment (J1)

## Question

Does standard theory-of-mind benchmark performance predict whether a model weights
intent in graded moral judgment? A null converts "models pass ToM tests but fail
this task" from a literature argument into a result measured on the same models.

## Design

- **ToM axis (primary):** BigToM forward belief, false-belief condition. 200 items,
  two-alternative forced choice scored by length-normalised log-likelihood. The
  explicit statement of the agent's initial belief is removed from the story
  (init_belief=0), so the belief must be inferred rather than copied.
- **ToM axis (secondary):** ToMi first-order belief questions, 400 items, same
  scoring.
- **Behavioural axis:** the 2x2 contrast (attempted - accidental).
- **Restriction:** models clearing the derived engagement floor (rating_std >= 0.2191). A model that does not vary its ratings has no contrast
  to correlate.
- **Uncertainty:** bootstrap over models, 10,000 resamples.

The interpretation of each possible outcome was fixed in the script docstring before
the full roster was scored; only the three gate models had been run.

## Ceiling gate

Run first, on Qwen2.5-0.5B-Instruct, Qwen2.5-14B-Instruct and OLMo-2-7B-Instruct,
because a correlation needs variance on both axes and a ceiling would have killed the
analysis before spending GPU on 20 models:

| benchmark | accuracies | spread | verdict |
|---|---|---|---|
| BigToM | 0.520 / 0.882 / 0.850 | 0.362 | spread, proceed |
| ToMi | 0.482 / 0.512 / 0.818 | 0.335 | spread, proceed |

Neither is near ceiling, so the full roster was worth running.

## Result

| ToM measure | r | 95% CI (bootstrap over models) | n | reading |
|---|---|---|---|---|
| bigtom|false_belief | +0.424 | [-0.711, +0.995] | 6 | UNINFORMATIVE — the interval is too wide to exclude a moderate effect in either direction. Not a null. |
| bigtom | +0.136 | [-0.960, +0.973] | 6 | UNINFORMATIVE — the interval is too wide to exclude a moderate effect in either direction. Not a null. |
| tomi | -0.376 | [-0.999, +0.965] | 6 | UNINFORMATIVE — the interval is too wide to exclude a moderate effect in either direction. Not a null. |

## Per-model table

ToM accuracy is reported as its own column regardless of the correlation result,
as requested, in `tom_vs_contrast.csv` and folded into the master table.

| model | type | params | BigToM false-belief | BigToM all | ToMi | contrast | engaged |
|---|---|---|---|---|---|---|---|
| Qwen_Qwen2_5-14B-Instruct | instruct | 14.0 | 0.985 | 0.882 | 0.512 | -0.370 | yes |
| Qwen_Qwen2_5-14B | base | 14.0 | 0.940 | 0.757 | 0.542 | -0.126 | no |
| Qwen_Qwen2_5-7B | base | 7.0 | 0.935 | 0.720 | 0.520 | -0.051 | no |
| Qwen_Qwen2_5-7B-Instruct | instruct | 7.0 | 0.935 | 0.868 | 0.550 | -0.238 | no |
| unsloth_gemma-2-9b-it | instruct | 9.0 | 0.935 | 0.858 | 0.665 | -0.408 | yes |
| allenai_OLMo-2-1124-7B | base | 7.3 | 0.930 | 0.630 | 0.757 | -0.004 | no |
| unsloth_gemma-2-9b | base | 9.0 | 0.925 | 0.655 | 0.745 | -0.000 | no |
| Qwen_Qwen2_5-3B-Instruct | instruct | 3.0 | 0.920 | 0.675 | 0.540 | -0.247 | no |
| allenai_OLMo-2-1124-7B-Instruct | instruct | 7.3 | 0.890 | 0.850 | 0.818 | -0.646 | yes |
| unsloth_Meta-Llama-3_1-8B-Instruct | instruct | 8.0 | 0.865 | 0.807 | 0.570 | -0.202 | no |
| allenai_Llama-3_1-Tulu-3-8B | instruct | 8.0 | 0.855 | 0.805 | 0.757 | -0.401 | yes |
| unsloth_Meta-Llama-3_1-8B | base | 8.0 | 0.835 | 0.723 | 0.728 | 0.003 | no |
| HuggingFaceH4_zephyr-7b-beta | instruct | 7.2 | 0.835 | 0.833 | 0.522 | -0.551 | yes |
| mistralai_Mistral-7B-Instruct-v0_3 | instruct | 7.0 | 0.815 | 0.833 | 0.637 | -0.473 | yes |
| mistralai_Mistral-7B-v0_3 | base | 7.0 | 0.800 | 0.728 | 0.620 | -0.003 | no |
| Qwen_Qwen2_5-3B | base | 3.0 | 0.795 | 0.675 | 0.537 | -0.048 | no |
| Qwen_Qwen2_5-0_5B | base | 5.0 | 0.775 | 0.512 | 0.505 | 0.000 | no |
| Qwen_Qwen2_5-0_5B-Instruct | instruct | 5.0 | 0.635 | 0.520 | 0.482 | -0.050 | no |
| Qwen_Qwen2_5-1_5B | base | 5.0 | 0.625 | 0.545 | 0.492 | -0.013 | no |
| Qwen_Qwen2_5-1_5B-Instruct | instruct | 5.0 | 0.545 | 0.608 | 0.550 | -0.167 | no |

## Caveats

- n is at most 20 and smaller after the engagement restriction, so the per-model
  table is the deliverable and the correlation is secondary. A wide interval is
  reported as uninformative, not as a null.
- Instruction tuning moves both axes, so it is the obvious confound; base and
  instruct models are marked separately in the scatter.
- ToMi's true_belief / false_belief tags describe the story-generation condition
  rather than the queried agent's belief state, so only the aggregate and the
  question-type breakdown are used.

## Sensitivity to the engagement floor

The primary analysis uses the derived floor (0.2191), which leaves n=6. At that n the
bootstrap intervals span nearly [-1, +1], so the primary correlation is uninformative
by construction. Below are the same correlations at two looser restrictions, so the
result is not an artifact of n=6 alone. These are sensitivity checks; they do not
replace the pre-registered primary.

| floor | ToM measure | r | n | note |
| --- | --- | ---: | ---: | --- |
| 0.05 (old) | bigtom\|false_belief | -0.150 | 11 | point estimate near zero |
| 0.05 (old) | bigtom | -0.555 | 11 | higher ToM → more negative (outcome-driven) contrast |
| 0.05 (old) | tomi | -0.599 | 11 | same direction |
| 0.0 (all 20) | bigtom\|false_belief | -0.261 | 20 | |
| 0.0 (all 20) | bigtom | -0.738 | 20 | |
| 0.0 (all 20) | tomi | -0.240 | 20 | |

None of these is a *positive* correlation. The dissociation claim therefore does not
weaken under the looser floors: higher ToM accuracy does not track greater intent use
(more positive contrast). At n=20 the BigToM–contrast association is large and
negative, which is the wrong sign for "ToM competence explains moral intent use" —
instruction-tuned models tend to score higher on BigToM *and* more outcome-driven on
the 2×2. Treat the unrestricted row as descriptive; base models with near-zero
rating_std pull the association and are excluded by any engagement floor.
