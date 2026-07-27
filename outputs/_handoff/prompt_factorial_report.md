# Prompt factorial analysis (W5 / roadmap #3)

## Headline result — cross-prompt sign stability

**4 of 20 models change sign across construct-matched prompts on a common 1–7 scale** (flip rate 20%).

This is a primary finding, not merely an exclusion filter. The prompts differ only in wording and construct (blame / wrongness / punishment) on one shared response scale, so a sign change means the model does not merely shift magnitude — it reverses which of intent and outcome it weights more. That speaks directly to the prompt-fragility literature (NEXT_PHASE_PLAN §2c) and is reportable whichever way it comes out: a high rate is evidence that single-prompt moral-judgment results are unsafe to generalize, and a low rate is positive evidence that the intent-vs-outcome contrast is a stable property of the model rather than of the prompt.

## Template set

Designed 7 = `human_verbatim` + 2 wordings × 3 constructs (`blame` / `wrongness` / `punishment`) on a common 1–7 scale (`blame_w1/w2`, `wrong_w1/w2`, `punish_w1/w2`).

`para_blame10` and other legacy templates remain **additive** (never replaced). `wrong_w1`/`punish_w1` alias to overnight `para_wrong7`/`punish7` (identical wording).

## Sign stability (pre-registered inclusion)

- Models scored on factorial 1–7 prompts: **20**
- Included in pooled factorial mean (sign-stable): **13**
- Sign-flippers (reported separately, not averaged in): **4**
- Flip rate: **20%**

Pre-registered floor for fitting the variance model: **3 models / 18 observations**. If the sign-stable subset falls below it, the sensitivity fit is reported as not estimable rather than fitted under-powered.

## Variance decomposition

Fixed effects: `C(wording) * C(construct)` on contrast; random intercept: template/prompt (MixedLM when available).

The pre-registered sign-stability rule governs the pooled **mean contrast**, not this model: filtering flippers out of a variance decomposition would discard the prompt-driven variance it exists to quantify. Primary = all models; sign-stable-only is reported as a sensitivity check.

### Primary (all models)

```
{'sign_stable_only': False, 'estimable': True, 'n_obs': 120, 'n_models': 20, 'n_templates': 6, 'note_scale': 'All factorial paraphrases are 1–7; scale factor has no within-factorial variance. Scale effects are assessed via legacy para_blame10 (1–10) / para_blame4 (1–4) separately. Scale replication (YS2008↔YS2009 human_verbatim): pooled r≈0.71, Bland–Altman bias ≈ −0.06 — see outputs/SCALE_REPLICATION.md. Not recomputed here.', 'anova_typeII': {'sum_sq': {'C(wording)': 0.0, 'C(construct)': 0.0061, 'C(wording):C(construct)': 0.0171, 'Residual': 4.9964}, 'df': {'C(wording)': 1.0, 'C(construct)': 2.0, 'C(wording):C(construct)': 2.0, 'Residual': 114.0}, 'F': {'C(wording)': 0.0, 'C(construct)': 0.07, 'C(wording):C(construct)': 0.1955, 'Residual': nan}, 'PR(>F)': {'C(wording)': 0.9986, 'C(construct)': 0.9324, 'C(wording):C(construct)': 0.8227, 'Residual': nan}}, 'variance_share': {'C(wording)': 0.0, 'C(construct)': 0.0012, 'C(wording):C(construct)': 0.0034, 'Residual': 0.9954}, 'mixedlm_converged': True, 'mixedlm_params': {'Intercept': -0.1436, 'C(wording)[T.2]': -0.0296, 'C(construct)[T.punishment]': -0.0068, 'C(construct)[T.wrongness]': -0.0384, 'C(wording)[T.2]:C(construct)[T.punishment]': 0.0304, 'C(wording)[T.2]:C(construct)[T.wrongness]': 0.0585, 'Group Var': 1.0}, 'mixedlm_template_var': 0.043828, 'mixedlm_resid_var': 0.043828}
```

### Sensitivity (sign-stable models only)

```
{'sign_stable_only': True, 'estimable': True, 'n_obs': 78, 'n_models': 13, 'n_templates': 6, 'note_scale': 'All factorial paraphrases are 1–7; scale factor has no within-factorial variance. Scale effects are assessed via legacy para_blame10 (1–10) / para_blame4 (1–4) separately. Scale replication (YS2008↔YS2009 human_verbatim): pooled r≈0.71, Bland–Altman bias ≈ −0.06 — see outputs/SCALE_REPLICATION.md. Not recomputed here.', 'anova_typeII': {'sum_sq': {'C(wording)': 0.0, 'C(construct)': 0.0108, 'C(wording):C(construct)': 0.0246, 'Residual': 3.3348}, 'df': {'C(wording)': 1.0, 'C(construct)': 2.0, 'C(wording):C(construct)': 2.0, 'Residual': 72.0}, 'F': {'C(wording)': 0.0008, 'C(construct)': 0.1166, 'C(wording):C(construct)': 0.2658, 'Residual': nan}, 'PR(>F)': {'C(wording)': 0.9779, 'C(construct)': 0.8901, 'C(wording):C(construct)': 0.7673, 'Residual': nan}}, 'variance_share': {'C(wording)': 0.0, 'C(construct)': 0.0032, 'C(wording):C(construct)': 0.0073, 'Residual': 0.9895}, 'mixedlm_converged': True, 'mixedlm_params': {'Intercept': -0.2216, 'C(wording)[T.2]': -0.0448, 'C(construct)[T.punishment]': -0.0081, 'C(construct)[T.wrongness]': -0.0588, 'C(wording)[T.2]:C(construct)[T.punishment]': 0.0432, 'C(wording)[T.2]:C(construct)[T.wrongness]': 0.087, 'Group Var': 1.0}, 'mixedlm_template_var': 0.046316, 'mixedlm_resid_var': 0.046316}
```

## Scale replication (cited, not recomputed)

Scale replication (YS2008↔YS2009 human_verbatim): pooled r≈0.71, Bland–Altman bias ≈ −0.06 — see outputs/SCALE_REPLICATION.md. Not recomputed here.

Wrote `outputs/analysis/prompt_factorial_sign_stability.csv`, `outputs/analysis/prompt_factorial_variance.csv`.
