# Methods note — choice of child developmental measure (Naughty vs. Punish)

*Ready-to-paste justification for using the "Naughty" (wrongness) series, not the
"Punish" series, from Cushman et al. (2013) Fig. 3 as the child developmental
anchor. Source values are the calibrated reads in `cushman_naughty_digitized.csv`
(verification overlay: `cushman_digitized_overlay.png`).*

---

## Paragraph (paste into the paper)

For the child developmental reference we use the **"Naughty" (moral-wrongness)**
judgment rather than the **"Punish"** judgment from Cushman, Sheketoff, Wharton &
Carey (2013). The load-bearing reason is **construct-matching**: our model and adult
anchors are *moral-wrongness / permissibility* judgments (adults: Young, Cushman,
Hauser & Saxe, 2007, a permissibility rating; models: "how morally wrong…"), so the
child measure must also index wrongness for the developmental ladder to compare like
with like. Using the punishment series would confound the developmental construct of
interest (attribution of wrongness) with a distinct process (deserved sanction), which
follows a different and later developmental trajectory. This choice is reinforced by
Cushman et al.'s own central finding: **intent-based moral judgment emerges earlier in
wrongness/"naughtiness" attributions than in punishment**, where children remain more
outcome-based for longer. Anchoring to the wrongness series therefore gives the most
direct, construct-valid developmental comparison for the intent-vs-outcome contrast
(blame(attempted) − blame(accidental)) that we compute for every model.

## Digitized values used (calibrated reads, ~±0.03–0.05)

Primary variant = **"Presented First"** (first-story-only; avoids within-subject
order/contrast contamination). Ages binned to the reference groups; see
`cushman_child_bands_PROPOSED.csv` for the "first+second average" robustness variant.

| group | accidental | attempted | contrast (att − acc) |
|---|---|---|---|
| child_4_5 | 0.455 | 0.695 | **+0.24** |
| child_6_7 | 0.200 | 0.700 | **+0.50** |
| child_8plus | 0.060 | 0.690 | **+0.63** |
| adult (Young et al. 2007) | 0.267 | 0.933 | +0.67 |

The contrast rises monotonically with age (age-4 ≈ 0 → age-8 ≈ +0.63 → adult +0.67),
i.e. children increasingly discount accidental harm and weight intent — the expected
developmental signature.

## Caveats (state these explicitly)

1. **Approximate reads.** Values are calibrated reads of a published figure (gridline
   y-calibration is automated and exact; marker centers read to ~±0.03–0.05). They are
   *not* the authors' exact per-age means. For a publication lock-down, confirm with
   WebPlotDigitizer or request the raw means from the Cushman lab.
2. **Different scale types.** The child measure is a **proportion** of children judging
   the act naughty (binary → 0–1); the adult anchor is a **mean rating** normalized to
   0–1. Both are directionally comparable on the intent-vs-outcome contrast, but the
   absolute magnitudes are not strictly on the same metric — treat the child bands as an
   ordinal developmental ladder, not an exact numeric scale.
3. **Presentation order / U-shape.** Cushman reports both "first story" and "second
   story"; the accidental-harm series in particular is order-sensitive and the
   developmental curve for accidental judgments is non-monotonic ("U-shaped") in some
   analyses. We use first-story-only as primary and report the first+second average as a
   robustness check.
4. **Revision vs. prior values.** These construct-matched wrongness reads **replace** the
   earlier child bands (which mixed naughty+punishable and were read from text:
   child_4_5 −0.14, child_6_7 +0.15, child_8plus +0.46). The revised bands are higher
   (more intent-based) at every age; under them, most LLMs fall **below the youngest
   children** on intent-weighting — strengthening, not weakening, the headline finding.

## Sources

- Cushman, F., Sheketoff, R., Wharton, S., & Carey, S. (2013). *The development of
  intent-based moral judgment.* Cognition, 127(1), 6–21. — Fig. 3 (child data).
- Young, L., Cushman, F., Hauser, M., & Saxe, R. (2007). *The neural basis of the
  interaction between theory of mind and moral judgment.* PNAS, 104(20). — adult anchor.
