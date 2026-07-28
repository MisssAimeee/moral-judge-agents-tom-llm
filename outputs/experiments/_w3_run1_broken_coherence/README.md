# W3 run 1 (job 19092342) — quarantined, do not cite

The coherence instrument was broken in this run, so it produced no usable verdict on the
pre-specified predictions P1–P3. Kept only for provenance.

What was wrong: the degeneracy detector flagged any generation under three tokens as
degenerate, and the rating prompt correctly answers with a single digit, so 100% of
*baseline* generations were labelled degenerate. P4 then gated out every non-zero alpha,
and the intent directions — the only ones measured with generations at mid alphas — came
out with `nan` effect sizes while the controls passed by not being tested.

The contrast and perplexity numbers here are real; the coherence verdicts are not, and the
intent-vs-control comparison is not valid because the two were held to different standards.

Fixes and the amended pre-registration are in `../W3_PRESPEC.md` (Amendment, 2026-07-28).
Run 2 supersedes this directory.
