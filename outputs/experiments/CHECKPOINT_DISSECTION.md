# Checkpoint dissection — post digit-token fix (factorial templates)

All 11 checkpoints rescored on `human_verbatim` + 6 factorial templates (job `19030249`).
`n_items=2086` = 298 stories × 7 templates. Same measurement basis as the ladder and
the factorial analysis.

| family | stage | contrast | rating_std |
| --- | --- | ---: | ---: |
| OLMo-2-7B | base | +0.0017 | 0.0575 |
| OLMo-2-7B | SFT | −0.5862 | 0.3513 |
| OLMo-2-7B | DPO | −0.6518 | 0.3962 |
| OLMo-2-7B | Instruct | −0.6868 | 0.4120 |
| Tulu-3-8B | base | +0.0074 | 0.0966 |
| Tulu-3-8B | SFT | −0.2577 | 0.2060 |
| Tulu-3-8B | DPO | −0.3947 | 0.2890 |
| Tulu-3-8B | RLVR | −0.3977 | 0.2894 |
| Zephyr-7B | base | −0.0009 | 0.0285 |
| Zephyr-7B | SFT | −0.1496 | 0.1679 |
| Zephyr-7B | DPO | −0.5516 | 0.3878 |

## SFT is sufficient, but the locus is recipe-dependent — three families

**SFT alone is sufficient to induce outcome bias in all three families**, moving each off a
neutral base by 0.15–0.59 in contrast with no preference optimization involved:

| family | base → SFT | SFT drop |
| --- | --- | ---: |
| OLMo-2-7B | +0.002 → −0.586 | 0.588 |
| Tulu-3-8B | +0.007 → −0.258 | 0.265 |
| Zephyr-7B | −0.001 → −0.150 | 0.149 |

**But the relative contribution of SFT versus preference optimization is recipe-dependent.**
Each stage's step as a share of that family's total base → final shift:

| family | SFT share | later-stage share | concentrates at |
| --- | ---: | ---: | --- |
| OLMo-2-7B | 85.4% | 14.6% | SFT |
| Tulu-3-8B | 65.4% | 34.5% | SFT |
| Zephyr-7B | 27.0% | 73.0% | **DPO** |

Zephyr's shift is concentrated at DPO, where one preference-optimization stage moves the
contrast from −0.150 to −0.552. **Do not describe the effect as localized to SFT, and do not
claim it is "not RLHF/DPO"** — Zephyr refutes both. Zephyr was previously all zeros at every
stage (digit-token collapse on the Mistral tokenizer); it is now a third supporting family
for the sufficiency claim and the decisive counterexample for the locus claim.

`b_outcome`/`b_intent` at final stage: OLMo 6.6×, Tulu 3.4×, Zephyr 7.7× (3.0–7.7× across
all non-base stages). The earlier "2.5–3.9×" figure was pre-rescore and is retired; report
the range, and treat "several-fold faster than `b_intent`, every family, every stage" as the
robust claim.

**Base engagement now resolved.** All three bases are engaged and near-zero on contrast
(`rating_std` OLMo 0.0575, Tulu **0.0966**, Zephyr 0.0285). Tulu's base was previously
degenerate at 0.018 — a stated limitation, now fixed — so the base → tuned comparison is
like-for-like between responsive models, materially strengthening the causal reading.

Full per-stage table: `CHECKPOINT_STAGE_SHARES.md` / `checkpoint_stage_shares.csv`
(`code/experiments/47_checkpoint_stage_shares.py`).
