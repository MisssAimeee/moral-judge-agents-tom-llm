# Closed reasoning dose, run 1 — preserved, unusable (2026-07-28 → died 03:06 2026-07-29)

Do not read any number in this directory. Kept for provenance only. Two independent reasons,
either of which alone disqualifies it.

## 1. API failures were imputed as the scale midpoint

`run_cell` contained:

```python
if not ratings:
    ratings = [(s_min + s_max) / 2.0]
```

So a prompt that returned nothing parseable was written to the raw CSV as a rating at the
exact centre of the scale, with no marker distinguishing it from a real response. Downstream,
`contrast_from_raw` averaged those imputed values into the cell means. **An API failure
became a datum that reads as a deliberately indifferent moral judgment.**

The clearest casualty is `kimi-k2.6/direct`, which the contrast table reported as
contrast = 0.0000 with all four cell means at exactly 0.500 — a model apparently assigning
identical blame to neutral, accidental, attempted and intentional harm. All 56 of its rows
are imputed; it produced zero parseable responses. The number was entirely manufactured by
the fallback.

| raw file | rows | rows at exactly 0.500 |
|---|---:|---:|
| `raw_kimi-k2.6__direct.csv` | 56 | 56 (100%, provably all imputed) |
| `raw_gpt-5.5__direct.csv` | 3028 | 507 (17%) |
| `raw_claude-opus-5__direct.csv` | 2623 | 150 (6%) |
| `raw_gemini-3.5-flash__direct.csv` | 2479 | 147 (6%) |
| `raw_gemini-3.1-pro-preview__direct.csv` | 240 | 0 |
| `raw_o3__budget_low.csv` | 80 | 0 |
| `raw_gpt-5.4-mini__direct.csv` | 80 | 0 |

These files are quarantined rather than filtered because for every model except kimi the
imputed rows **cannot be separated from genuine mid-scale ratings**: a 4 on a 1–7 blame scale
and a 2 on a 1–3 permissibility scale both normalise to exactly 0.500. The information needed
to tell them apart was never written down. Only a rerun fixes it.

Fixed in `52_closed_reasoning_dose.py`: failures now write `status=failed` with an empty
rating, `contrast_from_raw` excludes them, and every contrast is reported beside its
`parse_rate`, `n_ok`/`n_failed` and a `usable` flag (≥40 of 53 scenario groups).

## 2. The run never had the funds to finish

Google returned "prepayment credits are depleted" and OpenAI "exceeded your current quota";
the processes then spent hours on 429s that could not succeed, because the retry guard matched
only "quota"/"per_day". 1 of 29 cells got meaningful coverage before everything died. The
reconstructed cost of the configuration is ~$23,000 — see the addendum in
`outputs/API_COST_ESTIMATE.md`. It was never a token-budget problem.

Run 2 changes the configuration (N_SAMPLES 20 → 2, templates 4 → 2, provider Batch APIs) and
adds a billing fast-fail so a dead account stops the provider immediately.
