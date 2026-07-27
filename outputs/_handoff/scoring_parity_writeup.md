# Scoring parity: logprob-EV vs sampling (Roadmap #1)

**Question.** Open-weight (local) models are scored by **logprob expected value** (EV:
`E[rating] = Σ p(k)·k` in one deterministic forward pass), while closed APIs are scored
by **sampling** (mean of `n=30` draws at `T=1`). Do these two methods measure the *same*
thing, so that cloud-vs-local comparisons aren't a scoring artifact?

**Method.** For instruct models with a logprob-EV score, also sample `T=1, n=30` on the
**same** items (`human_verbatim` scale), then compute per-item Pearson `r`, Bland–Altman
agreement, and the intent-vs-outcome **contrast** (attempted − accidental) under each
method. Criterion: **PASS if r > 0.95.** Scale used is 1–7 (verified single-token, so the
EV path isn't truncated; EV renormalizes over the valid `{1..K}` rating tokens).
Source: `outputs/analysis/scoring_parity.csv` (Slurm job 17627327; code commits
`3c9db2b`, `19d8721`).

## Results (instruct ladder)

| Model | r | EV contrast | Sampled contrast | Verdict |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 0.11 | −0.10 | −0.06 | fail |
| Qwen2.5-1.5B-Instruct | −0.21 | −0.18 | −0.02 | fail |
| Qwen2.5-3B-Instruct | 0.34 | −0.15 | −0.16 | fail |
| **Qwen2.5-7B-Instruct** | **0.96** | −0.28 | −0.30 | **PASS** |
| Mistral-7B-Instruct-v0.3 | nan | 0.00 | −0.35 | degenerate |
| **OLMo-2-7B-Instruct** | **0.97** | −0.27 | −0.28 | **PASS** |

Source: `outputs/analysis/scoring_parity.csv` (`ev_source=computed` for OLMo — EV was
computed in-run, not from a cached logprob pass).

## Interpretation (nuanced — read this carefully before citing)

**What is validated:** the intent/outcome **contrast statistic** (attempted − accidental)
agrees closely across scoring methods for capable instruct models, and this per-item
agreement is not a fluke of one model family — it holds for **two independent 7B-class
families** (Qwen2.5, Alibaba; OLMo-2, AI2). **What is *not* validated:** a universal claim
that "EV and sampling always agree." They agree specifically **at 7B+ scale, in
instruction-tuned chat models** — smaller and base models do not clear the bar (see below).
Keep the claim scoped to that regime; don't generalize past the data.

1. **Two independent families pass cleanly, not just one.** Qwen2.5-7B-Instruct
   (`r = 0.96`, EV −0.28 vs sampled −0.30) and OLMo-2-7B-Instruct (`r = 0.97`, EV −0.27 vs
   sampled −0.28, Bland–Altman mean diff = 0.00) both clear `r > 0.95` with near-identical
   aggregate contrasts. Two unrelated training pipelines converging on the same result is a
   meaningfully stronger basis than one model alone — it supports "EV and sampling are
   interchangeable **for the aggregate contrast** in instruction-following models," which is
   the specific comparison the cross-model (cloud-vs-local) analysis actually needs.

2. **Agreement scales with capability, it isn't binary.** The smaller Qwen instruct models
   (0.5–3B) have weak/negative *item-level* `r` even though their aggregate contrasts land in
   the same ballpark as the 7B models. At small scale the per-item rating is noisier and less
   consistent between a one-shot EV and a 30-sample mean, so item-level correlation is weak
   while the aggregate statistic is comparatively stable. Read this as "scoring-method
   agreement is itself a function of model competence," not as a failure of the EV method —
   and not as license to claim parity holds at all scales.

3. **Mistral is degenerate, not disagreeing.** `r = nan` because its EV contrast is exactly
   0.000 — Mistral returns near-constant ratings (zero variance ⇒ Pearson undefined). This
   matches the earlier prompt-invariance finding that Mistral is degenerate on this task; it
   should be QC-flagged, not counted as a parity pass/fail.

4. **Base models are the wrong test.** An earlier run on Qwen **base** models gave `r ≈ 0`
   (see `outputs/analysis/scoring_parity/sampled_Qwen_Qwen2_5-*B.csv`). Base models don't
   reliably answer the rating question, so EV and sampling pick up different artifacts.
   Parity is only meaningful on instruct/chat models — which is exactly what the closed APIs
   are.

## Bottom line

**Validated, with a clearly scoped claim.** Two 7B-class instruct models from independent
families (Qwen2.5, OLMo-2) pass the `r > 0.95` bar with matching aggregate contrasts
(Bland–Altman mean diff ≈ 0). The defensible statement is: *for capable instruction-tuned
models, EV and sampling are interchangeable for the aggregate intent/outcome contrast* — not
a blanket claim that the two methods always agree. Small models, base models, and
degenerate models (Mistral) do not clear the bar and should be reported as the boundary of
where the equivalence holds, not swept under "capability scaling" as a single monotonic
story.

## Next
- ~~Extend to OLMo-2-7B-Instruct~~ — done, PASS (`r = 0.97`).
- Extend to **Llama-3.1-8B-Instruct** (needs Llama gate access approved on the HF account)
  as a third independent family, and — more importantly — because the same
  `meta-llama/Llama-3.1-8B` base checkpoint is the missing 11th checkpoint blocking the
  Tülu-3-8B base→SFT step in the checkpoint-dissection analysis (see
  `outputs/experiments/checkpoint_dissection_writeup.md`). Command:
  `python code/analysis/15_scoring_parity.py --run`.
- Optionally report parity at the **scenario-contrast** level (correlate per-scenario
  contrasts EV vs sampled) as a second, aggregate-level agreement check.
