# W3 causal steering — pre-specified predictions

Written by `code/experiments/48_w3_causal_steering.py` at the start of the first `--run`, before any steering result existed. This file is never overwritten on later runs.

Coherence bounds (P4), fixed in the script as constants:

- perplexity ratio <= 1.5x baseline
- refusal-rate increase <= 0.10 absolute
- degenerate-generation fraction <= 0.10

```
=========================
Written before any steering run. Recorded verbatim to outputs/experiments/W3_PRESPEC.md on
first --run, and never overwritten, so results cannot be retrofitted to the hypothesis.

  P1 (direction specificity). Adding the INTENT direction raises the contrast
      (attempted - accidental) relative to alpha=0. The OUTCOME direction and the
      matched-norm RANDOM directions do not.
  P2 (dose-response). Within the coherent alpha range, contrast change is monotone in
      alpha for the intent direction, and the sign flips with the sign of alpha.
  P3 (method agreement). The difference-of-means intent direction and the probe-weight
      intent direction produce the same qualitative effect. If they disagree, the effect
      is a property of one estimator, not of the representation, and P1 is not supported.
  P4 (coherence). Any contrast change claimed under P1-P3 occurs at an alpha where the
      model is still coherent, defined in advance as: perplexity ratio <= 1.5x baseline,
      refusal-rate increase <= 0.10 absolute, and degenerate-generation fraction <= 0.10.
      Effects appearing only outside that range are reported as steering damage, not as
      causal evidence about intent.

  FALSIFICATION. P1 fails if the intent direction moves the contrast no more than the
  controls do, or if it moves it only where P4's coherence bound is violated. That outcome
  is reportable: it would mean the decodable intent direction is not causally wired to the
  moral judgment, strengthening the "represented but unused" reading rather than weakening
  the paper.

  NON-TRIVIALITY. A purely uniform additive shift would raise blame in all four cells
  together and leave their DIFFERENCE unchanged. A contrast change therefore requires
  differential movement across cells, so all four cell means are recorded at every alpha.
========================================================================================
```

## Amendment, 2026-07-28 — coherence instrument fix

The first run (job 19092342) produced NO usable verdict on P1-P3 because P4 gated out every
non-zero alpha, including alpha values where the model was demonstrably fine. That was a
bug in the coherence instrument, not a property of the model, and it was fixed before any
P1-P3 result was accepted or reported. Three changes, none of them to a threshold:

1. **Degeneracy detector.** It flagged any generation shorter than three tokens as
   degenerate. The rating prompt correctly answers with a single digit ("3"), so 100% of
   BASELINE generations were labelled degenerate. Degeneracy is now empty output, or heavy
   repetition (unique-token ratio < 0.35) evaluated only on outputs of at least 8 tokens.
2. **Coherence is now measured on prose.** Refusal and repetition cannot be assessed on a
   single digit, and "manual read of 20 outputs per level" is meaningless when every output
   is one character. Each level now also generates a one-to-two sentence explanation per
   story, which is what the refusal, degeneracy and manual-read checks run on.
3. **Symmetric measurement.** The first version generated only for the intent directions at
   mid alphas, so the control directions passed coherence by not being tested -- a weaker
   standard for the controls than for the direction under test, which is backwards for a
   specificity comparison. Every direction is now measured identically at every alpha.

One criterion was ADDED: task compliance (`answer_rate`), the fraction of rating prompts
whose greedy generation still contains a scale digit, may not drop by more than 0.10. A
model that has stopped emitting ratings cannot support a contrast measurement whatever its
perplexity. The three original thresholds (perplexity ratio 1.5x, refusal delta 0.10,
degenerate fraction 0.10) are unchanged.

P1-P4 themselves are unchanged and were not informed by any result.
