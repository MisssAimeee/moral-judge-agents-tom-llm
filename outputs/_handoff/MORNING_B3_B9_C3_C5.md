# Morning report — B3 / B9 / C3 / C5 (before new jobs)

**Date:** 2026-07-27 · Jobs 18962392 (B3), 18962393 (B9), 18962391 (C3)

---

## B3 checkpoint dissection — **DID NOT RECOMPUTE**

Job `18962392` exited in 6s with:

```
[resume] 11 checkpoint(s) already scored … -> will skip and reuse (pass --force to rescore everything).
```

`outputs/experiments/checkpoint_dissection.csv` is **byte-identical in content to the
pre-repair writeup table** (OLMo Instruct contrast −0.289, b_i 0.104, b_o 0.393, …).
Overnight B2 wrote fresh `item_means_*` at 00:00–00:15, but B3 never read them.

**Therefore: we cannot yet say whether the 2.5–3.9× `b_outcome`/`b_intent` growth
survives repaired text + `scenario_group` + CPR act-only labels.** That requires
`16_checkpoint_dissection.py --run --force` against the new behavior tree (queued next).

> **RESOLVED (job `19030249`, rescore on 7-template basis).** The 2.5–3.9× figure did
> **not** survive and is retired. The ratio is now **3.0–7.7×** across all non-base stages
> (final stage: OLMo 6.6×, Tülu 3.4×, Zephyr 7.7×). What is robust is the qualitative
> claim — `b_outcome` grows several-fold faster than `b_intent` in every family at every
> stage — not any single multiplier. Zephyr is also no longer degenerate. See
> `experiments/CHECKPOINT_STAGE_SHARES.md` and the revised
> `experiments/checkpoint_dissection_writeup.md`.

### Stale CSV ratios (superseded by the rescore — NOT a survival claim)

| transition | Δb_outcome | Δb_intent | ratio |
|---|---:|---:|---:|
| OLMo base→SFT | +0.239 | +0.082 | **2.9×** |
| OLMo DPO→Instruct | +0.152 | +0.039 | **3.9×** |
| Tulu base→SFT | +0.240 | +0.090 | **2.7×** |
| Tulu SFT→DPO | +0.211 | +0.083 | **2.5×** |

No group-level CIs are in this CSV; `16_checkpoint_dissection.py` does not currently emit
bootstrapped CIs on Δb (mini_dissection / ladder do for contrasts). Force-rerun should add
scenario_group bootstrap CIs on coefficients if we extend the script; until then, report
point estimates only and say so.

Zephyr family remains **degenerate** at every stage. *(Superseded: the degeneracy was the
digit-token collapse on the Mistral tokenizer, not model behaviour. Post-fix, Zephyr is
engaged at every stage and is the family whose shift concentrates at DPO rather than SFT.)*

---

## B9 ladder — **DID run on repaired rescore**

Job `18962393` regenerated `outputs/master_all_models.csv` + ladder PNG after B2.
Contrasts use scenario_group bootstrap CIs.

### Llama-3.1-8B-Instruct counterexample — **SURVIVES**

| model | contrast | 95% CI | sig≠0 |
|---|---:|---|---|
| Llama-3.1-8B (base) | −0.022 | [−0.028, −0.016] | yes (tiny) |
| **unsloth-Meta-Llama-3.1-8B-Instruct** | **−0.001** | **[−0.042, +0.043]** | **no** |
| OLMo-2-1124-7B-Instruct (comparison) | −0.463 | [−0.516, −0.410] | yes |

Instruct Llama stays at chance-zero contrast while other instruct models go strongly
outcome-biased (OLMo −0.46, Qwen-14B-Inst −0.28, gpt-4o −0.38). Old writeup had
base→instruct Δcontrast **+0.126** for Llama; on the new ladder Δ ≈ **(+0.021)**
(base −0.022 → instruct −0.001). **Direction of the exception holds; magnitude of the
“improvement” shrank.** Still the clear recipe-dependent counterexample.

Claude Opus (+0.093) remains the only closed model with a reliably positive contrast.

---

## C3 within-cell (18962391) — complete

Intent remains decodable in **no-harm** cells (no harm event in either cell of the pair):

| model | intent_noharm peak | intent_harm peak |
|---|---:|---:|
| OLMo-Instruct | 0.931 | 0.885 |
| Qwen2.5-7B-Instruct | 0.929 | 0.889 |
| Qwen2.5-0.5B | 0.765 | 0.675 |

Outcome within innocent/guilty cells hits **≥0.995** (perfect fold separation) for 7B+.
Lexical-harm account of the intent probe remains dead at 7B+ on repaired data.

---

## C5 intent_reliance_summary / 04_link_analysis — **NOT rebuilt on new behavior**

- `outputs/behavior/intent_reliance_summary.csv` **missing** (only under `_contaminated_…`).
- Floor `|b_intent|+|b_outcome|≥0.05` is coded in `23_build_intent_reliance_summary.py`
  (`EFFECT_FLOOR = 0.05`).
- Status: **must run C5 on post-B2 `outputs/behavior/` then `04_link_analysis.py`**
  (≥6 matched models). Queued in P1.

---

## Contaminated backup

Present at `dataset/master/_prerepair_backup/moral_2x2_master_CONTAMINATED_20260619.csv`.
**md5 `5dd904a7609628553319da4acab02f25` matches.** Committing with §1.2 update.
(PROMPT_TAIL on this file = 96 = 48+48; user’s 99 = 48/49/2 may be a slightly different
visible rule — file is the provenance source either way.)
