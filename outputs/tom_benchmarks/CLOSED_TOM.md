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
| claude-opus-5 | anthropic | 127 | 0.992 | 0.986 | 1.000 | 1.000 | **PARTIAL 127/400** |
| claude-sonnet-4-6 | anthropic | 400 | 0.940 | 0.990 | 0.890 | 1.000 | complete |
| claude-sonnet-5 | anthropic | 397 | 0.972 | 0.985 | 0.960 | 1.000 | **PARTIAL 397/400** |
| gemini-2.5-flash | google | 400 | 0.970 | 0.980 | 0.960 | 1.000 | complete |
| gemini-2.5-pro | google | 234 | 0.974 | 0.992 | 0.957 | 1.000 | **PARTIAL 234/400** |
| gemini-3.5-flash | google | 400 | 0.965 | 0.980 | 0.950 | 1.000 | complete |
| gpt-4o-mini | openai | 400 | 0.915 | 0.955 | 0.875 | 1.000 | complete |
| gpt-4o | openai | 400 | 0.968 | 0.970 | 0.965 | 1.000 | complete |
| gpt-5.4-mini | openai | 400 | 0.950 | 0.990 | 0.910 | 1.000 | complete |
| gpt-5.5 | openai | 393 | 0.990 | 0.990 | 0.990 | 1.000 | **PARTIAL 393/400** |
| o3 | openai | 329 | 0.991 | 0.994 | 0.987 | 1.000 | **PARTIAL 329/400** |
| o4-mini | openai | 373 | 0.984 | 0.995 | 0.972 | 1.000 | **PARTIAL 373/400** |

**Partial runs.** `claude-opus-5` scored 127/400 items; `claude-sonnet-5` scored 397/400 items; `gemini-2.5-pro` scored 234/400 items; `gpt-5.5` scored 393/400 items; `o3` scored 329/400 items; `o4-mini` scored 373/400 items. These accuracies are computed on the items completed, so they carry wider sampling error than the full runs and the item mix may not be balanced across subsets. Treat them as provisional until the run finishes.

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
