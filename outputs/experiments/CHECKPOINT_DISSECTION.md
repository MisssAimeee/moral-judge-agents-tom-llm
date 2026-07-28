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

## SFT-locus claim — three families

| family | base → SFT |
| --- | --- |
| OLMo-2-7B | +0.002 → −0.586 |
| Tulu-3-8B | +0.007 → −0.258 |
| Zephyr-7B | −0.001 → −0.150 |

Zephyr was previously all zeros at every stage (digit-token collapse on the Mistral
tokenizer). It is now a third supporting family. The mechanism claim rests on **three**
instruction-tuning pipelines, all scored on the same 7-template basis as the ladder.
