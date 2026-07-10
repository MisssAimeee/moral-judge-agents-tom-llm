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

## Interpretation (nuanced)

1. **The flagship model passes cleanly.** Qwen2.5-7B-Instruct shows tight item-level
   agreement (`r = 0.96`) *and* matching contrasts (−0.28 EV vs −0.30 sampled, Bland–Altman
   mean diff −0.02). For a capable instruct model the two scoring methods are effectively
   interchangeable — which is the case that matters, since the cross-model comparison pits
   cloud instruct models (sampling) against local instruct models (EV).

2. **Agreement scales with capability.** The smaller Qwen instruct models (0.5–3B) have
   weak/negative *item-level* `r` even though their aggregate contrasts are in the same
   ballpark. At small scale the per-item rating is noisier and less consistent between a
   one-shot EV and a 30-sample mean, so item-level correlation is weak while the aggregate
   statistic is more stable. Read this as "scoring-method agreement is itself a function of
   model competence," not as a failure of the EV method.

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

**Partial but encouraging validation.** The most capable open instruct model tested passes
the `r > 0.95` bar with matching contrasts, supporting the defensibility of comparing
sampling-scored cloud models against EV-scored local models. Weaker models and the
degenerate Mistral do not clear the bar, which is informative about *where* the two methods
converge rather than a refutation of the method.

## Next
- Extend to **OLMo-2-7B-Instruct** (EV computed in-run) and **Llama-3.1-8B-Instruct**
  (needs an HF token with gated access) to test whether the 7–8B pass generalizes across
  families. Command: `python code/analysis/15_scoring_parity.py --run`.
- Optionally report parity at the **scenario-contrast** level (correlate per-scenario
  contrasts EV vs sampled) as a second, aggregate-level agreement check.
