# Prompt factorial analysis (W5 / roadmap #3)

## Headline result — cross-prompt sign stability

**5 of 20 models change sign across construct-matched prompts on a common 1–7 scale** (flip rate 25%).

This is a primary finding, not merely an exclusion filter. The prompts differ only in wording and construct (blame / wrongness / punishment) on one shared response scale, so a sign change means the model does not merely shift magnitude — it reverses which of intent and outcome it weights more. That speaks directly to the prompt-fragility literature (NEXT_PHASE_PLAN §2c) and is reportable whichever way it comes out: a high rate is evidence that single-prompt moral-judgment results are unsafe to generalize, and a low rate is positive evidence that the intent-vs-outcome contrast is a stable property of the model rather than of the prompt.

## Template set

Designed 7 = `human_verbatim` + 2 wordings × 3 constructs (`blame` / `wrongness` / `punishment`) on a common 1–7 scale (`blame_w1/w2`, `wrong_w1/w2`, `punish_w1/w2`).

`para_blame10` and other legacy templates remain **additive** (never replaced). `wrong_w1`/`punish_w1` alias to overnight `para_wrong7`/`punish7` (identical wording).

## Sign stability (pre-registered inclusion)

- Models scored on factorial 1–7 prompts: **20**
- Included in pooled factorial mean (sign-stable): **15**
- Sign-flippers (reported separately, not averaged in): **5**
- Flip rate: **25%**

Pre-registered floor for fitting the variance model: **3 models / 18 observations**. If the sign-stable subset falls below it, the sensitivity fit is reported as not estimable rather than fitted under-powered.

## Variance decomposition

Fixed effects: `C(wording) * C(construct)` on contrast; random intercept: template/prompt (MixedLM when available).

The pre-registered sign-stability rule governs the pooled **mean contrast**, not this model: filtering flippers out of a variance decomposition would discard the prompt-driven variance it exists to quantify. Primary = all models; sign-stable-only is reported as a sensitivity check.

### Primary (all models)

```
{'sign_stable_only': False, 'estimable': True, 'n_obs': 120, 'n_models': 20, 'n_templates': 6, 'note_scale': 'All factorial paraphrases are 1–7; scale factor has no within-factorial variance. Scale effects are assessed via legacy para_blame10 (1–10) / para_blame4 (1–4) separately. Scale replication (YS2008↔YS2009 human_verbatim): pooled r≈0.71, Bland–Altman bias ≈ −0.06 — see outputs/SCALE_REPLICATION.md. Not recomputed here.', 'anova_typeII': {'sum_sq': {'C(wording)': 0.0001, 'C(construct)': 0.0032, 'C(wording):C(construct)': 0.0244, 'Residual': 5.8806}, 'df': {'C(wording)': 1.0, 'C(construct)': 2.0, 'C(wording):C(construct)': 2.0, 'Residual': 114.0}, 'F': {'C(wording)': 0.0017, 'C(construct)': 0.0309, 'C(wording):C(construct)': 0.2369, 'Residual': nan}, 'PR(>F)': {'C(wording)': 0.9673, 'C(construct)': 0.9696, 'C(wording):C(construct)': 0.7894, 'Residual': nan}}, 'variance_share': {'C(wording)': 0.0, 'C(construct)': 0.0005, 'C(wording):C(construct)': 0.0041, 'Residual': 0.9953}, 'mixedlm_converged': True, 'mixedlm_params': {'Intercept': -0.1926, 'C(wording)[T.2]': -0.0312, 'C(construct)[T.punishment]': -0.0052, 'C(construct)[T.wrongness]': -0.0417, 'C(wording)[T.2]:C(construct)[T.punishment]': 0.0203, 'C(wording)[T.2]:C(construct)[T.wrongness]': 0.0681, 'Group Var': 1.0}, 'mixedlm_template_var': 0.051584, 'mixedlm_resid_var': 0.051584}
```

### Sensitivity (sign-stable models only)

```
{'sign_stable_only': True, 'estimable': True, 'n_obs': 90, 'n_models': 15, 'n_templates': 6, 'note_scale': 'All factorial paraphrases are 1–7; scale factor has no within-factorial variance. Scale effects are assessed via legacy para_blame10 (1–10) / para_blame4 (1–4) separately. Scale replication (YS2008↔YS2009 human_verbatim): pooled r≈0.71, Bland–Altman bias ≈ −0.06 — see outputs/SCALE_REPLICATION.md. Not recomputed here.', 'anova_typeII': {'sum_sq': {'C(wording)': 0.0002, 'C(construct)': 0.0049, 'C(wording):C(construct)': 0.0309, 'Residual': 4.1035}, 'df': {'C(wording)': 1.0, 'C(construct)': 2.0, 'C(wording):C(construct)': 2.0, 'Residual': 84.0}, 'F': {'C(wording)': 0.0048, 'C(construct)': 0.05, 'C(wording):C(construct)': 0.3159, 'Residual': nan}, 'PR(>F)': {'C(wording)': 0.9449, 'C(construct)': 0.9513, 'C(wording):C(construct)': 0.73, 'Residual': nan}}, 'variance_share': {'C(wording)': 0.0001, 'C(construct)': 0.0012, 'C(wording):C(construct)': 0.0075, 'Residual': 0.9913}, 'mixedlm_converged': True, 'mixedlm_params': {'Intercept': -0.2572, 'C(wording)[T.2]': -0.0408, 'C(construct)[T.punishment]': -0.0059, 'C(construct)[T.wrongness]': -0.0553, 'C(wording)[T.2]:C(construct)[T.punishment]': 0.0247, 'C(wording)[T.2]:C(construct)[T.wrongness]': 0.0879, 'Group Var': 1.0}, 'mixedlm_template_var': 0.048851, 'mixedlm_resid_var': 0.048851}
```

## Scale replication (cited, not recomputed)

Scale replication (YS2008↔YS2009 human_verbatim): pooled r≈0.71, Bland–Altman bias ≈ −0.06 — see outputs/SCALE_REPLICATION.md. Not recomputed here.

Wrote `outputs/analysis/prompt_factorial_sign_stability.csv`, `outputs/analysis/prompt_factorial_variance.csv`.
