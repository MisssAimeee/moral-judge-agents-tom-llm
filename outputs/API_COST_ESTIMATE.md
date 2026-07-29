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

---

# Addendum (2026-07-29) — the reasoning dose–response was never covered by the estimate above

The overnight reasoning-dose run (`52_closed_reasoning_dose.py`, roadmap #7) exhausted Google
prepayment credits and the OpenAI quota and died with 1 of 29 cells usable. That was not a
provider fault and not a token-cap bug. **The run as configured is a ~$23,000 job.** The
estimate above does not cover it, in three independent ways.

**1. It assumed ~5 output tokens per completion.** True for a direct rating. In a thinking
condition the reasoning chain is billed as output, so one completion costs 500–8,000 output
tokens instead of 5 — three orders of magnitude per call, on the most expensive token class.

**2. "Batching" in the table above means `n=` multi-candidate sampling, not the providers'
Batch APIs.** No Batch endpoint is used anywhere in this codebase, so no request gets the
50% batch discount. Worse, reasoning endpoints do not accept `n=` at all, so the
prompt-billed-once saving does not apply either: every one of the 23,840 completions per
cell is a separate request with its own full thinking chain.

**3. `N_SAMPLES = 20` is multiplied by the thinking cost.** 298 items × 4 templates × 20
samples = 23,840 completions per cell, × 29 cells = 691,360 completions, ≈138M input and
≈1,324M output tokens.

## What the configurations cost

**Every figure below is recomputed by `code/experiments/59_dose_cost_model.py`** — run it to
re-derive, sweep the assumptions, or correct the rates. The roster, request counts and token
counts are read from `52_closed_reasoning_dose.py`, so the arithmetic follows the config.

Two inputs are assumptions rather than measurements, and they drive the total:

- **Prices.** The 2026 roster (`claude-opus-5`, `gpt-5.5`, `o3`, `gemini-3.1-pro-preview`,
  `kimi-k2.6`) has no verified rate card in this repository. The table at the top of this file
  prices a *different, older* roster (Claude-Opus-4.6, GPT-4o, Gemini-2.5-Pro). The cost model
  carries those tiers across by role and **guesses the o-series outright** ($10/$40 for o3,
  $1.10/$4.40 for o4-mini), which matters because o3 is the single largest line item. Confirm
  on the consoles before relying on any total.
- **Thinking utilisation.** Cost depends on how much of each thinking budget a model actually
  spends, which is unobservable before running. Default 50%. At 25% the approved config is
  ~$297, at 100% ~$1,152 — so the figure is a band, not a point.

| configuration | calls/cell | input | output | est. cost |
|---|---:|---:|---:|---:|
| as configured (n=20, 4 tmpl, no batch) | 23,840 | 138M | 1,324M | **~$23,300** |
| n=2, 4 tmpl, no batch | 2,384 | 14M | 132M | ~$2,330 |
| n=2, 4 tmpl, Batch API | 2,384 | 14M | 132M | ~$1,160 |
| n=2, 2 tmpl, Batch API — **approved** | 1,192 | 7M | 66M | **~$580** |
| n=2, 2 tmpl, Batch API, minus `claude-opus-5` + `o3` | 1,192 | 7M | 66M | **~$180** |

Where the request count comes from: 298 items × 2 templates × 2 samples = **1,192 calls per
(model, condition) cell**, × **29 cells** over the 9-model roster = **34,568 requests**. The
cells are 5 conditions each for `claude-opus-5`, `claude-sonnet-5`, `gemini-3.1-pro-preview`
and `gemini-3.5-flash`; 3 each for `o3` and `o4-mini`; `direct` only for `gpt-5.5`,
`gpt-5.4-mini` and `kimi-k2.6`.

Per-model at the approved configuration:

| model | cells | est. cost | share |
|---|---:|---:|---:|
| `o3` | 3 | $207 | 36% |
| `claude-opus-5` | 5 | $194 | 33% |
| `claude-sonnet-5` | 5 | $116 | 20% |
| `gemini-3.1-pro-preview` | 5 | $39 | 7% |
| `o4-mini` | 3 | $23 | 4% |
| `gemini-3.5-flash` | 5 | $2 | 0% |
| `gpt-5.5`, `gpt-5.4-mini`, `kimi-k2.6` | 1 each | <$1 | 0% |

The two flagship reasoners are ~69% of the bill: `claude-opus-5` because budget_high is 16,384
thinking tokens at an assumed $25/MTok output, `o3` because high effort is assumed to spend
~12,000 at $40/MTok. **`o3`'s rate is the least reliable input in the whole model and it is
the largest line item** — worth confirming before committing.

## Why `N_SAMPLES = 20` buys almost nothing here

The measured quantity is a cell mean over 1,192 prompts per cell. Per-prompt sampling noise
is averaged away by the 1,192 prompts regardless of how many samples each contributes;
n_samples raises precision on individual items, which nothing in the dose–response analysis
reads. n=20 was inherited from the behavioural sampling-parity config, where per-item
variance was the object of study. Here it multiplies the most expensive token class by 20 for
no gain in the reported statistic.

## Recommended before spending anything further

1. `N_SAMPLES` 20 → 2 (10× on everything).
2. Route through each provider's Batch API (50%, and it removes the rate-limit pressure that
   produced the Anthropic overload 429).
3. Cut templates 4 → 2 (`human_verbatim` + one blame wording) or drop the two flagship
   reasoners, either of which brings the job under $600.
4. Keep the caps raised (o3 4096/8192/16384 by effort; Gemini thinking-budget + 1536;
   Anthropic budget + 1536) — those were a real defect, just not the one that emptied the
   account.

The billing fast-fail added in the same change means a depleted account now stops the
provider immediately instead of spending hours on unwinnable 429 retries, which is how the
failure went unnoticed until morning.
