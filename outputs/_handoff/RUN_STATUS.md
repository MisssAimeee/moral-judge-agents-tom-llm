# Run status

Status of every analysis named in `confound_and_rsa_plan.md` and `CURSOR_MASTER_SEQUENCE.md`.
Neither plan document is checked into the repository; both were maintained in the working
session, and the item names below are reproduced from them.

Updated 2026-07-28 after jobs `19030249` (checkpoint factorial rescore) and `19027795`
(v3 probe-side refresh) completed.

## Stimulus-master versions referenced

| Tag | Version | Timestamp |
| --- | --- | --- |
| v1 | contaminated (story-boundary leakage, CV leakage) | through 2026-07-26 23:07 |
| v2 | contamination + CV leakage repaired; CPR act-only polarity fixed | 2026-07-26 23:07 / 23:42 |
| v3 | v2 plus the YS2011 title strip — **current** | 2026-07-27 10:11 |

Closed-API (cloud) models were scored 2026-07-21 against **v1** and have not been rescored;
that spend is not yet approved. Every artifact containing cloud rows is marked in `INDEX.md`.

## confound_and_rsa_plan.md

| Analysis | Status | Job ID | Dataset version |
| --- | --- | --- | --- |
| A1 — Fix `build_dataset.py` | complete | — (local) | produces v2 |
| A2 — Rebuild master, recompute derived columns | complete | — (local) | v2 |
| A3 — Validation gate (all checks must pass) | complete | — (local) | v2 |
| A4 — Diff report vs pre-repair master | complete | — (local) | v1 → v2 |
| A5 — Inspect suspected label error | complete | — (local) | v2 |
| C1 — Layer-0 read-off diagnostic | complete | 18962390; refreshed **19027795** | v3 activations |
| C2 — TF-IDF surface baseline (+ span-matched) | complete | 18962385; refreshed **19027795** + local span recompute | v3; spans in `surface_baseline.csv` |
| C3 — Within-cell contrast probes | complete | 18962391; refreshed **19027795** | v3 activations |
| C4 — Permutation null, N=1000, peak layer + layer 0 | complete | 18992763, 18992764, 18992765, 18992767 | v3 activations |
| C5 — Fix `04_link_analysis.py` + build intent-reliance summary | complete | 18992768 **failed**; rerun locally 2026-07-27 | v3 |
| P1 — Annotate belief/action/outcome clause offsets | complete | — (local) | v3 |
| P2 — Re-extract activations with clause offsets | complete | 18992757 | v3 |
| P3 — Clause-position probes (`belief_last`, `action_last`) | complete | 18992760, 18992761 | v3 activations |
| RSA — model similarity, hypothesis RDMs, CKA, nulls (4 poolings) | complete | 18992769, 18992770, 18992771, 18992772 | v3 activations |

## CURSOR_MASTER_SEQUENCE.md

| Analysis | Status | Job ID | Dataset version |
| --- | --- | --- | --- |
| B1 — Activation extraction, 8 models, 4 pooling variants | complete | 18962383, superseded by 18992757 | v3 |
| B2 — Behavioral rescore, all open models | complete | 18962384, superseded by 19000403 | v3 |
| B3 — Checkpoint dissection | complete | 18992762; factorial rescore **19030249** | v3; 7 templates × 11 ckpts; digit-token fix |
| B4 — Layer-wise probes, all 4 pooling variants | complete | 18992758, 18992759, 18992760, 18992761 | v3 activations |
| B5 — Layer-0 diagnostic (= C1) | complete | refreshed **19027795** | v3 activations |
| B6 — TF-IDF surface baseline (= C2) | complete | refreshed **19027795** | v3 (+ span-matched) |
| B7 — Within-cell contrast probes (= C3) | complete | refreshed **19027795** | v3 activations |
| B8 — Clause-position probes (= P3) | complete | 18992760, 18992761 | v3 activations |
| B9 — Master ladder regeneration | complete | 18962393, superseded by local rerun 2026-07-27 14:56 | v3 |
| W0 — Behavioral rescore, 11 templates, 20 open-weight models | complete | 19000403 (18995998 cancelled); digit-token quarantine + rescore 19023272 | v3 |
| W1 — Mixed-effects with interaction (J3) | complete | — (local; `39_mixed_effects_2x2.py`) | v3 |
| W2 — Figures and tables for mentor + talk | in progress | handoff rebuild 2026-07-28 | v3 |
| W3 — Causal steering | not started | — | — |
| W4 — Prompt curriculum | not started | — | — |
| W5 — Prompt factorial (sign stability + variance decomposition) | complete | — (local, 2026-07-27) | v3 |
| W6 — Reasoning dose–response | not started | — | — |
| W7 — Non-moral selectivity control (demoted to appendix) | not started | — | — |
| W8 — Paper assembly | not started | — | — |

## Not covered by either plan

| Item | Status | Job ID | Dataset version |
| --- | --- | --- | --- |
| Closed-API model rescore | not started — pending budget approval | — | would move cloud rows from v1 to v3 |
| Punish-anchor child ladder | complete | — (local, 2026-07-27) | v3 |
| Gap-over-surface by pooling (span-matched) | complete | local after **19027795**; C2 REOPENED | v3 |
| J1 — ToM vs contrast (confound-controlled) | complete | 19027524 (tom_full) + local `42_tom_vs_contrast.py` | v3 |
| J2 — Item-level dissociation null | complete | — (local; `41_item_level_dissociation.py`) | v3 |
| J4 — Floor derivation (dual-use documented) | complete | — (local; `40_derive_floors.py`) | v3 |
| Source-split probes (C2 belief_last YS2008/YS2009) | complete | 19025559 | v3 activations |
| Digit-token collapse quarantine + Zephyr/Mistral rescore | complete | 19023272; ckpt **19030249** | v3 |

## Failed jobs

| Job ID | Name | Outcome |
| --- | --- | --- |
| 18959978 | probe_action_last | failed; resubmitted as 18961409, completed |
| 18959837 | rescore | cancelled; resubmitted as 18962384, completed |
| 18992768 | c5_link | failed (quoting bug in the `bash -c` wrapper); rerun locally, completed |
| 18995998 | w0_rescore | cancelled; resubmitted as 19000403, completed |
| 19025558 | ckpt_factorial | failed (`nvidia-smi \| head` + pipefail); resubmitted as **19030249**, completed |
