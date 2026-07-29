# Closed-model selection — queried 2026-07-28T21:53:41

Source catalog: `CLOSED_MODEL_CATALOG.md` (live `models.list` / `list_models` on that timestamp).
Keys present: Anthropic, OpenAI, Google, Moonshot. **DeepSeek: no key — skipped.** Together list endpoint errored; not used as a DeepSeek proxy.

## Selected roster (flagship + mid-tier per provider; reasoning siblings paired)

| provider | role | model string (exact) | reasoning sibling / mode |
|---|---|---|---|
| Anthropic | flagship | `claude-opus-5` | same model, `thinking` on + budget low/med/high |
| Anthropic | mid | `claude-sonnet-5` | same, thinking on + budgets |
| OpenAI | flagship chat | `gpt-5.5` | paired with `o3` |
| OpenAI | mid chat | `gpt-5.4-mini` | paired with `o4-mini` |
| OpenAI | flagship reason | `o3` | `reasoning_effort` low/med/high |
| OpenAI | mid reason | `o4-mini` | `reasoning_effort` low/med/high |
| Google | flagship | `gemini-3.1-pro-preview` | thinking_budget 0 / low / med / high |
| Google | mid | `gemini-3.5-flash` | thinking_budget 0 / low / med / high |
| Moonshot | Kimi | `kimi-k2.6` | direct (no public thinking-budget API on this endpoint) |

## Conditions (roadmap #7)

For each model that exposes a thinking control:

| code | meaning |
|---|---|
| `direct` | thinking off / minimal; answer only |
| `think` | thinking/reasoning enabled at provider default |
| `budget_low` / `budget_med` / `budget_high` | scaled thinking budget |

OpenAI chat models (`gpt-5.5`, `gpt-5.4-mini`) run `direct` only; their dose–response comes from the paired o-series models. Anthropic and Google run all five on the same model string. Kimi runs `direct` only.

## Templates (cost-forced cut)

Full open factorial is 7 (`human_verbatim` + 2×3). **Cut to 4** so contrasts stay construct-matched without blowing the budget:

`human_verbatim`, `blame_w1`, `wrong_w1`, `punish_w1`

(one wording per construct + the human anchor). Documented here so a full 7-template rerun is an explicit follow-up, not a silent gap.

## Scoring parity

Closed APIs: **sampling only** (`T=1`, `n_samples=20`). Claude has no logprobs; OpenAI disables them on reasoning models. Do not mix estimators across the open/closed boundary.

Open-model validation already on disk (`outputs/analysis/scoring_parity.csv`): OLMo-2-7B-I r=0.974, Qwen2.5-7B-I r=0.959 (PASS at r>0.95). Smaller Qwen/Mistral fail the parity bar and are not used as the bridge. Closed contrasts are therefore compared to open models that already pass sample↔logprob-EV agreement.

## Cost estimate (print-before-spend)

Workload per (model, condition): 298 × 4 × 20 = **23,840** completions.

Token assumptions: direct ~200 in / 8 out; thinking ~200 in / (budget) out. Anthropic has no `n=` (sequential). OpenAI/Moonshot batch via `n=`. Gemini `candidate_count`≤8.

| cell | n conditions | est. USD | notes |
|---|---:|---:|---|
| claude-opus-5 | 5 | ~$90 | thinking output dominates; sequential |
| claude-sonnet-5 | 5 | ~$45 | |
| gpt-5.5 | 1 | ~$3 | batched n=20 |
| gpt-5.4-mini | 1 | ~$0.5 | |
| o3 | 3 (L/M/H) | ~$40 | reasoning tokens; Responses API |
| o4-mini | 3 | ~$8 | |
| gemini-3.1-pro-preview | 4 | ~$15 | thinking budgets |
| gemini-3.5-flash | 4 | ~$3 | |
| kimi-k2.6 | 1 | ~$1 | |
| **TOTAL (behavioral)** | | **~$205** | dominated by Anthropic thinking |
| BigToM generative (400 items × roster, once) | | **~$15–25** | no n_samples loop |

If Anthropic thinking proves costlier than estimated mid-run, drop `budget_high` on Opus first, then Opus entirely — Sonnet carries the Anthropic dose–response.

**Proceeding with this plan.** DeepSeek omitted (no key). Full 7-template factorial deferred.
