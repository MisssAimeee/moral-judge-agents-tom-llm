# Closed-API cost estimate — behavioral rescoring

## Workload (overnight §7 / `03_behavioral.py` defaults for closed APIs)

- **Items:** 298 (master)
- **Templates:** 3 (`human_verbatim`, `para_wrong7`, `punish7` — matching `outputs/agents/behavior` / overnight plan; not all 7 `ALL_TEMPLATES`)
- **Samples:** n_samples = **20**
- **Completions per model:** 298 × 3 × 20 = **17,880**
- **Token assumptions:** ~200 input + ~5 output per completion (same as overnight report; vignette ~100 words + short rating question)

## Batching in `code/03_behavioral.py`

| backend | batching |
|---|---|
| OpenAI / Together / Moonshot | `n=n_samples` — one request, prompt billed once |
| Google Gemini | `candidate_count` capped at **8**; n=20 → 3 rounds |
| Anthropic | **no `n=`** — sequential loop over samples |

## Prices used (list rates, mid-2026; confirm before spend)

| model | input $/MTok | output $/MTok | source note |
|---|---:|---:|---|
| Claude-Opus-4.6 | 5.00 | 25.00 | registry closed API |
| Claude-Sonnet-4.6 | 3.00 | 15.00 | registry closed API |
| Claude-Haiku-4.5 | 1.00 | 5.00 | registry closed API |
| GPT-4o | 2.50 | 10.00 | registry closed API |
| GPT-4o-mini | 0.15 | 0.60 | registry closed API |
| Gemini-2.5-Pro | 1.25 | 10.00 | registry closed API |
| Gemini-2.5-Flash | 0.15 | 0.60 | registry closed API |
| Kimi-K3 | 1.00 | 3.00 | registry closed API |

Kimi-K3 Moonshot rates are approximate ($1/$3); verify on the Moonshot console.

## Cost table

| model | provider | batchable | cost WITHOUT batching | cost WITH batching | notes |
|---|---|---|---:|---:|---|
| Claude-Opus-4.6 | Anthropic | no (sequential) | $20.11 | $20.11 | Anthropic Messages API has no n=; cost scales with n_samples |
| Claude-Sonnet-4.6 | Anthropic | no (sequential) | $12.07 | $12.07 | Anthropic Messages API has no n=; cost scales with n_samples |
| Claude-Haiku-4.5 | Anthropic | no (sequential) | $4.02 | $4.02 | Anthropic Messages API has no n=; cost scales with n_samples |
| GPT-4o | OpenAI | yes (n=) | $9.83 | $1.34 | n=20 in one call; prompt billed once |
| GPT-4o-mini | OpenAI | yes (n=) | $0.59 | $0.08 | n=20 in one call; prompt billed once |
| Gemini-2.5-Pro | Google | partial (≤8) | $5.36 | $1.56 | candidate_count≤8 → 3 rounds for n=20 |
| Gemini-2.5-Flash | Google | partial (≤8) | $0.59 | $0.13 | candidate_count≤8 → 3 rounds for n=20 |
| Kimi-K3 | Moonshot | yes (n=) | $3.84 | $0.45 | n=20 in one call; prompt billed once |
| **TOTAL** | — | — | **$56.43** | **$39.77** | — |

## Comparison to overnight ~$95 / ~$10–15

- Prior worst-case **~$95** used legacy Opus list rates (~$15/$75). At those rates Opus alone ≈ **$60** of the bill. Current Opus 4.6 is **$5/$25**, so Opus without batching is ≈ **$20**, not $60.
- Recomputed **without batching** at current prices: **$56** (Anthropic $36 + others $20).
- Recomputed **with available batching**: **$40** (Anthropic still **$36** — unchanged; non-Anthropic drops $20 → $4).
- The overnight claim that batching brings the total to **~$10–15** assumed **universal** prompt-once batching. That is **invalid for Anthropic**: Opus+Sonnet+Haiku alone remain ~$36 even after OpenAI/Gemini/Moonshot batch. A $10–15 figure only works if you drop Anthropic (especially Opus) or cut n_samples / templates.

## Practical options

1. **Full 8-model × 3 tmpl × 20 samp** with real batching: ~$40 (dominated by Anthropic).
2. **Drop Opus** (−$20): total ≈ $20.
3. **Anthropic at n_samples=5** (5× cheaper on Claude line) if variance allows.
4. Do **not** budget $10–15 for the full Anthropic-inclusive plan.

*Generated for deliverable 3; prices dated ~2026-07. Not a commit.*
