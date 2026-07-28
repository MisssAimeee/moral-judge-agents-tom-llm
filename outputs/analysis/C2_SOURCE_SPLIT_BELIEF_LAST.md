# C2 — Belief-last probes split by stimulus source

Job `19025559` completed 2026-07-27. Peak `cv_acc` per model at `belief_last`,
outcome and intent, for YS2008 (n=192) vs YS2009 (n=96).

## Outcome decoding at belief_last

| model | YS2008 | YS2009 | Δ (08−09) |
| --- | ---: | ---: | ---: |
| OLMo-2-1124-7B | 0.901 | 0.882 | +0.018 |
| OLMo-2-1124-7B-Instruct | 0.927 | 0.828 | +0.100 |
| Qwen2.5-0.5B | 0.801 | 0.752 | +0.049 |
| Qwen2.5-0.5B-Instruct | 0.788 | 0.785 | +0.003 |
| Qwen2.5-1.5B | 0.823 | 0.820 | +0.003 |
| Qwen2.5-1.5B-Instruct | 0.842 | 0.835 | +0.007 |
| Qwen2.5-7B | 0.887 | 0.825 | +0.062 |
| Qwen2.5-7B-Instruct | 0.907 | 0.830 | +0.077 |

## Intent decoding at belief_last

| model | YS2008 | YS2009 | Δ (08−09) |
| --- | ---: | ---: | ---: |
| OLMo-2-1124-7B | 0.979 | 0.980 | −0.001 |
| OLMo-2-1124-7B-Instruct | 0.984 | 0.988 | −0.003 |
| Qwen2.5-0.5B | 0.939 | 0.932 | +0.006 |
| Qwen2.5-0.5B-Instruct | 0.934 | 0.905 | +0.029 |
| Qwen2.5-1.5B | 0.964 | 0.960 | +0.004 |
| Qwen2.5-1.5B-Instruct | 0.965 | 0.960 | +0.005 |
| Qwen2.5-7B | 0.980 | 1.000 | −0.020 |
| Qwen2.5-7B-Instruct | 0.980 | 0.978 | +0.002 |

## Verdict

The predicted signature for a true pre-outcome cut — **high outcome decoding on
YS2008, near chance on YS2009** — is **not** present. Outcome decoding at
`belief_last` is high for **both** sources (YS2009 range 0.75–0.88; chance ≈ 0.5).
YS2008 is slightly higher for most models, but YS2009 is nowhere near chance.

So the neutral caption on the gap figure stays permanently. Clause-position
pooling is a statement about **where in the text the probe reads**, not about
whether the outcome-determining fact is unavailable. Even when the story order
puts belief before the state-of-the-world sentence (YS2009), something at
`belief_last` is still linearly predictive of the eventual outcome label —
likely foreshadowing, world knowledge, or residual lexical cues that survive
the surface baseline.

This result is reported, not demoted to a check.

Artifacts: `outputs/probe/*_probe_belief_last_srcYS2008.csv` and `*_srcYS2009.csv`
(and the matching `action_last` files).
