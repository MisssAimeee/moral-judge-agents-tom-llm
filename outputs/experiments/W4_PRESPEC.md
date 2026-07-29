# W4 prompt curriculum — pre-specified readings

Written by `code/experiments/54_w4_prompt_curriculum.py` at the start of the first `--run`, before any curriculum result existed. Never overwritten on later runs. Both readings (prompting works / prompting also fails) are committed here so neither can be adopted after the fact.

```
========================================================================

The two outcomes are both informative and they say different things, so both readings
are committed to here rather than chosen after seeing the numbers.

P1  READING IF PROMPTING WORKS. If the contrast moves toward the adult direction as
    the curriculum escalates, the blockage is DOWNSTREAM of the representation and
    UPSTREAM of the output: the intent code exists, residual-stream intervention on it
    does not reach the judgment, but the judgment can be re-pointed at it from the
    input. That localises the failure to how the rating computation selects its inputs,
    not to the absence of a usable intent signal.

P2  READING IF PROMPTING ALSO FAILS. If escalating instruction, worked reasoning,
    adult-consistent few-shot ratings and an explicit statement of the principle all
    leave the contrast inverted, then the outcome bias is deeper than either
    intervention reaches. Two interventions at opposite ends of the pipeline -- one on
    the representation, one on the input -- both fail to move it, and the bias is a
    property of the tuned mapping rather than a prompt-level or read-out-level defect.

P3  BAR FOR "PROMPTING WORKS", fixed here. Positive shift in contrast (toward the adult
    ordering attempted > accidental) of at least +0.15 at L5 relative to L1, with a
    scenario-group bootstrap CI on the paired difference excluding 0, in at least 4 of
    the 6 engaged models. +0.15 is 10x the largest intent-steering effect in W3 (0.015)
    and about two thirds of what the W3 outcome positive control produces (0.232), so
    it is an effect this design has already been shown to resolve. Anything smaller is
    reported as a shift but not as recovery.

P4  FULL RECOVERY, distinguished from partial. Full recovery = contrast crosses zero
    and becomes positive at some level. Partial = significant positive shift that
    leaves the ordering inverted. These are reported separately; a significant shift
    that still has accidental rated above attempted is not human-like moral judgment.

P5  DOSE-RESPONSE. Under P1 the shift should be monotone or near-monotone in level,
    since the levels are cumulative. A single non-monotone jump at L4 only would be
    consistent with format imitation of the few-shot ratings rather than uptake of the
    principle, and is flagged as such (L5 adds the principle with no new labels, so
    L4->L5 separates imitation from principle).

P6  CEILING-COMPRESSION GUARD, carried over from W3. A contrast change produced by all
    four cell means rising or falling together is compression, not intent
    re-weighting. All four cell means are reported at every level, plus the fraction of
    ratings at the scale extremes, so this is checkable and not inferable only from the
    contrast.
```

## Relation to W3

W3 intervened on the representation and found it causally inert: 3.2–7.2 SD of probe-margin displacement of the intent code, |Δcontrast| ≤ 0.015, against an outcome positive control that moves the contrast 0.232–0.259 at the same depths. W4 intervenes at the input instead. The pair localises the failure; neither result alone does.
