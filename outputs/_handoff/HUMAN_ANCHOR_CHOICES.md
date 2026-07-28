# Why there are three different sets of child numbers

Companion to `human_only_developmental_ladder.png`. This explains where each child value
comes from and why they disagree, so the choice of anchor is defensible rather than
convenient.

## The measure being plotted

Every point is the **intent contrast**: normalised blame for *attempted* harm minus
normalised blame for *accidental* harm.

- **Positive** — intent matters more than outcome (the mature pattern).
- **Zero** — intent and outcome weigh equally.
- **Negative** — outcome matters more than intent (the young-child pattern).

Attempted harm means bad intent, no damage. Accidental harm means innocent intent, real
damage. Anyone who blames the attempt more than the accident is reading minds rather than
counting damage.

## Two source papers, two scales

| Group | Source | Original scale | Handling |
| --- | --- | --- | --- |
| Adults | Young, Cushman, Hauser & Saxe (2007) *PNAS*, Exp. 1 Fig. 2 | 1–4 permissibility (1 = forbidden) | reversed and rescaled to 0–1 blame |
| Children | Cushman, Sheketoff, Wharton & Carey (2013) *Cognition*, Fig. 3 | binary yes/no proportions | already 0–1 |

The adult anchor (**+0.67**) is the same in all three child series — there is only one adult
study, and it did not run the child measures. All three child ladders therefore converge on
one shared adult point. The adult study also used a *permissibility* scale, so its values are
reversed before comparison; that reversal is recorded per row in the reference CSVs.

## Why three child series

The 2013 child study asked each child **two separate questions** about the same story —
"Is he naughty?" and "Should he be punished?" — and the paper reports both. That gives two
legitimate construct-matched measures. A third series comes from reading numbers out of the
paper's prose instead of its figure. Hence three.

| Series | Ages 4–5 | Ages 6–7 | Age 8+ | Status |
| --- | --- | --- | --- | --- |
| **Naughty, presented-first** (digitized from Fig. 3) | **+0.24** | +0.50 | +0.63 | **pre-specified primary** (2026-07-10) |
| **Punish, presented-first** (digitized from Fig. 3) | **+0.09** | +0.12 | +0.19 | construct-matched secondary |
| Text-reported (naughty *or* punishable, pooled) | **−0.14** | +0.15 | +0.46 | superseded |

### Naughty, presented-first — the primary
Digitized directly from Fig. 3. This is the wrongness/naughtiness construct, which is what
the `blame_*` and `wrong_*` prompts ask models. Pre-specified as primary in
`methods_child_measure.md` on 2026-07-10, before any model was compared against it.

### Punish, presented-first — the secondary
The same children, the punishment question. Kept because three of the six factorial prompts
(`punish_w1`, `punish_w2`) ask models about deserved punishment, so this series is
construct-matched to those prompts. It is a robustness check, not a replacement.

### Text-reported — superseded, and why
Taken from the paper's prose rather than its figure. The prose reports a single
*condemnation* proportion that pools "naughty" and "punishable" into one number. That mixes
two constructs the paper itself separates, and it is the reason this series alone shows a
**negative** contrast at ages 4–5 (−0.14). It is retained only so the sensitivity of every
conclusion to this choice is visible.

## Two design decisions inside the digitization

**"Presented-first" only.** Each child judged both an accidental and an attempted story.
Whichever story came first is uncontaminated by having just judged the other; the second
judgment carries an order effect. Using presented-first trials only removes that carryover,
at the cost of half the data per child.

**Ages pooled into pairs.** Fig. 3 reports single years of age, with few children per year,
so single-year points are noisy both in the original sampling and in the digitization.
Adjacent years are averaged into 4–5 and 6–7; 8+ is the study's top bin and is left as is.
The paper describes its own developmental shift in these band terms.

## Why naughtiness and punishment disagree so much

This is not digitization error — it is a prediction. Cushman et al. argue for a **two-process**
account in which intent constrains judgments of *wrongness* before it constrains judgments of
*deserved punishment*. Punishment stays outcome-sensitive for longer.

That predicts exactly what the figure shows: the punish series is flatter and lower
(+0.09 → +0.19) than the naughty series (+0.24 → +0.63) across the same children. Two
independent digitizations reproducing the predicted ordering is evidence the digitization is
sound.

## What this means for the model comparison

The headline claim is "models fall at or below the youngest measured band," so the **ages 4–5
value is the threshold**. A *lower* threshold is a *harder* test.

| Threshold | Value | Harder or easier | Open-weight models at/below |
| --- | --- | --- | --- |
| Punish 4–5 | +0.09 | harder | 16/16 |
| Naughty 4–5 | +0.24 | easier | 16/16 |
| Text-reported 4–5 | −0.14 | hardest | 8/16 |

The claim holds under **both digitized measures**, including the stricter punishment
threshold. It fails only under the pooled-prose series — which is also the only series that
mixes two constructs. State it that way rather than picking the friendliest anchor.

## Files

| File | Contents |
| --- | --- |
| `human_only_developmental_ladder.png` | the figure |
| `human_only_developmental_ladder.csv` | the plotted contrasts |
| `dataset/human_reference/human_reference_digitized.csv` | Naughty series, cell means |
| `dataset/human_reference/human_reference_punish.csv` | Punish series, cell means |
| `dataset/human_reference/human_reference.csv` | text-reported series |
| `dataset/human_reference/cushman_child_bands_PROPOSED.csv` | pre-pooling digitized reads, Naughty |
| `dataset/human_reference/cushman_child_bands_PUNISH.csv` | pre-pooling digitized reads, Punish |
| `dataset/human_reference/methods_child_measure.md` | the 2026-07-10 pre-specification |
