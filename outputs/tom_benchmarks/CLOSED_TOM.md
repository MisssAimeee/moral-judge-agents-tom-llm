# Closed-model BigToM (generative, standalone)

Scoring: free generation forced to one of the two Forward-Belief options
(same options as the open-model logprob 2AFC). BigToM uses **init_belief=0**
(initial-belief sentence dropped). ToMi is not scored.

**Do not correlate** these accuracies against closed-model moral contrasts —
those contrasts are still v1-contaminated. Report ToM standalone only.

| model | backend | n | BigToM all | BigToM FB | BigToM TB | parse rate | run |
|---|---|---:|---:|---:|---:|---:|---|
| claude-haiku-4-5-20251001 | anthropic | 400 | 0.953 | 0.975 | 0.930 | 1.000 | complete |
| claude-opus-4-6 | anthropic | 400 | 0.985 | 0.985 | 0.985 | 1.000 | complete |
| claude-sonnet-4-6 | anthropic | 400 | 0.940 | 0.990 | 0.890 | 1.000 | complete |
| gemini-2.5-flash | google | 400 | 0.970 | 0.980 | 0.960 | 1.000 | complete |
| gemini-2.5-pro | google | 234 | 0.974 | 0.992 | 0.957 | 1.000 | **PARTIAL 234/400** |
| gpt-4o-mini | openai | 400 | 0.915 | 0.955 | 0.875 | 1.000 | complete |
| gpt-4o | openai | 400 | 0.968 | 0.970 | 0.965 | 1.000 | complete |

**Partial runs.** `gemini-2.5-pro` scored 234/400 items. These accuracies are computed on the items completed, so they carry wider sampling error than the full runs and the item mix may not be balanced across subsets. Treat them as provisional until the run finishes.

## Open-model logprob vs generative agreement (BigToM)

| model | n | logprob acc | generative acc | pred agreement |
|---|---:|---:|---:|---:|
| Qwen/Qwen2.5-7B-Instruct | 400 | 0.868 | 0.927 | 0.900 |
| allenai/OLMo-2-1124-7B-Instruct | 400 | 0.850 | 0.652 | 0.710 |

Qwen agreement is high; use generative for closed models and treat
open logprob BigToM FB as the open roster measure (parity demonstrated,
not perfect on every family).

## Notes

- Gemini thinking models often return a bare `A`/`B` or put the letter in
  thought parts; the generative scorer accepts bare letters and falls back
  to short thought lines. Report parse rate alongside accuracy.
- Empty responses count as incorrect; re-fetch if parse rate < 0.9.
