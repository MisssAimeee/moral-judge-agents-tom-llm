# ToM behavioral results — summary for mentor

## What this measures (plain words)

Each model reads a short moral story and rates how blameworthy the main character is. The 298 stories come in a 2×2 design crossing the character's **intent** (innocent vs guilty belief) with the **outcome** (no harm vs harm), giving four conditions:

- **neutral** — innocent belief, no harm
- **accidental** — innocent belief, but harm happens ("meant well, bad luck")
- **attempted** — guilty belief, but no harm happens ("tried to harm, failed")
- **intentional** — guilty belief and harm

Every rating is mapped to a common 0–1 blame scale. The key number is the **intent-vs-outcome contrast = blame(attempted) − blame(accidental)**: it is **positive** when a judge blames bad *intent* more than bad *outcome* (the mature, adult pattern) and **negative** when bad *outcome* drives blame (the young-child pattern). Ratings are read directly from the model's token probabilities (deterministic, no sampling noise), under the exact prompt from the source papers plus paraphrases to check the result isn't a wording artifact.

## Human reference (what "adult-like" means)

| group | contrast (attempted − accidental) |
|---|---|
| adult | +0.67 |
| child_8plus | +0.46 |
| child_6_7 | +0.15 |
| child_4_5 | -0.14 |

Adults are strongly intent-weighted (+0.67); 4–5-year-olds are outcome-weighted (−0.14). This is the developmental ladder models are placed on.

## Results table (one row per model)

Columns: **contrast** = behavioral intent-vs-outcome score (CI = 95% bootstrap); **≠0?** = is it reliably intent- or outcome-weighted; **adult-align** = 1 − RMSE of the full 4-condition profile vs adults (1.0 = identical to adults); **Δ vs adult** = gap on the contrast; **placement** = nearest human age group; **prompt SD / flip** = how much the contrast moves across wordings (★ = sign flips → not stable).

| model | params | type | contrast [95% CI] | ≠0? | intent-reliance | adult-align | Δ vs adult | placement | prompt SD |
|---|---|---|---|---|---|---|---|---|---|
| mockBase | nanB | base | -0.01 [-0.06,+0.04] | no | 0.38 | 0.47 | -0.68 | child_4_5 | 0.04 ★ |
| mockA-Instruct | nanB | instruct | -0.02 [-0.08,+0.03] | no | 0.30 | 0.46 | -0.69 | child_4_5 | 0.04 ★ |

## Which models are *statistically* different?

- No model pair has a contrast difference whose 95% CI excludes 0 — the between-model gaps are within noise at the current sample.

## Figures

![fig1_contrast_forest.png](figures/fig1_contrast_forest.png)

![fig2_condition_profiles.png](figures/fig2_condition_profiles.png)

![fig3_prompt_invariance.png](figures/fig3_prompt_invariance.png)

![fig5_intent_outcome_weights.png](figures/fig5_intent_outcome_weights.png)


## What was done / what's next

**Done:**
- 298-item 2×2 (intent×outcome) moral stimulus set built from Saxe-lab papers.
- Deterministic logprob scoring (no sampling noise); exact-paper prompt + 2 paraphrases.
- Adult ground truth (Young et al. 2007) + child developmental ladder (Cushman et al. 2013).
- Bootstrap CIs, cross-model & base-vs-instruct tests, prompt-invariance check, figures.

**Next:**
- Get per-item adult/child ratings on the *exact* stimuli (request from Saxe lab) for a true matched comparison.
- Add same-scale paraphrases to separate wording- from scale-sensitivity.
- Extend the model set (other families/sizes; reasoning models) — see chat for the list.
