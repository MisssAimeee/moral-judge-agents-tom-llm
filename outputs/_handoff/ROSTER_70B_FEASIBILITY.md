# Can this cluster host a 70B — and should the roster?

Flagged before any roster decision. Nothing has been started.

**Short answer: yes on hardware. 70B is the field norm for ToM papers, not an
extravagance; our ceiling at 14B with half the roster in one family is the
anomaly. One 70B still does not by itself break the open/closed × scale confound.**

## Field norm (why 70B is expected)

Recent ToM / social-reasoning LLM evaluations standardly combine **2–3 frontier
APIs** with **open-weight models spanning ~8B–70B**, and treat
**Llama-3.3-70B-Instruct** (or the contemporaneous 70B Llama instruct) as the
open reference point — not a stretch goal. Examples of that pattern:

- Strachan et al. 2024, *Nature Human Behaviour*
- Kosinski 2024, *PNAS*
- OmniToM 2026
- ToMBench
- OpenToM
- “LLMs achieve adult human performance on higher-order ToM”
- Theory of Mind in LLMs overview papers

Against that backdrop:

| our roster | field norm |
| --- | --- |
| open ceiling **14B** | open reference often **70B** |
| **~half the models are Qwen2.5** | multi-family span at each scale band |
| closed APIs present, open scale thin | 2–3 APIs **plus** 8B–70B open ladder |

So a reviewer asking “did you try a 70B?” is asking for the standard control,
not a luxury ablation. Staying at 14B leaves open-vs-closed confounded with
scale in a way the literature has already moved past.

## Hardware (unchanged)

| resource | finding | verdict |
|---|---|---|
| GPU count per node | `mit_preemptable` / `mit_normal_gpu` have 8× H200 or H100 | multi-GPU available |
| GPU memory | H200 = 141 GB, H100 = 80 GB | see below |
| 70B bf16 weights | ~140 GB | needs 2 GPUs |
| host RAM | ~2 TB/node | ample |
| disk | HF cache on scratch, **~75 TB free** | not a constraint |

Allocation for a 70B in bf16:

- **2× H200 = 282 GB** — comfortable
- **2× H100 = 160 GB** — workable but tight
- **1× H200** — do not (weights alone ≈ 140 GB)
- Avoid quantisation for probe comparability with the existing bf16 7B set

Logprob scoring (~4k rated items) and probing (~1.6 GB acts/model; rowspace
projection keeps fits cheap) are both feasible on this cluster. Digit-token
guards from the Mistral/Zephyr collapse must stay on.

## Cheaper middle band (before or instead of a full 70B pair)

If the goal is to leave the 14B ceiling without immediately paying for two 70B
families, the **27–32B** instruct band is the cost-effective step:

| model | params | rough disk (bf16) | GPUs (bf16) | role |
| --- | ---: | ---: | --- | --- |
| **gemma-3-27b-it** | 27B | ~54 GB | 1× H200 (comfortable) or 1× H100 (tight) | non-Qwen mid scale |
| **Qwen3-32B-Instruct** (or Qwen2.5-32B-Instruct) | 32B | ~64 GB | 1× H200 | extends the existing Qwen ladder past 14B |

Cost notes (order-of-magnitude, single-pass behavioral + optional probes):

- **Wall time:** behavioral rescore of 298×7 templates on 1× H200 is typically
  well under a 6–12 h limit; probes are CPU-side minutes after one activation pass.
- **Disk:** ~50–70 GB each in the HF scratch cache — negligible against 75 TB.
- **What it buys:** breaks the “nothing above 14B” objection and adds a second
  family at mid scale (gemma) without the 2-GPU scheduling friction of 70B.
- **What it does not buy:** the field’s Llama-3.3-70B reference point, or a clean
  scale slope — for that you still want ≥1 model at 70B (ideally a base/instruct
  pair in a second family).

Practical sequence: **gemma-3-27b-it + Qwen 32B** as a cheap mid-band, then
**Llama-3.3-70B-Instruct** (and, if claiming scale, Qwen2.5-72B base/instruct)
when ready to match the literature’s reference class.

## The confound caveat (still binding)

Adding **one** 70B answers the narrow reviewer question; it does not separate
scale from family. To support a scale claim:

1. At least **two families** at large scale (e.g. Llama-3.3-70B and Qwen2.5-72B).
2. Prefer **base/instruct pairs** where they exist — J3 already shows instruction
   tuning moves the moral contrast far more than parameter count in the current
   20-model set. Llama-3.3-70B is instruct-only; Qwen2.5-72B + base is the more
   informative pair for that axis.

Probing caveat: move peak-layer analyses to **relative depth** (layer / n_layers)
before comparing 80-layer 70Bs to 33-layer 7Bs; that change touches existing
figures.

## Recommendation

1. Treat **70B as the field-standard open reference**, not optional polish.
2. If budget/scheduling is tight, land **gemma-3-27b / Qwen3-32B** first to exit
   the 14B ceiling and diversify off Qwen-only growth.
3. Decide base/instruct pairing and relative-depth probing **before** the first
   70B run, so the new point does not orphan the existing layer figures.
4. Do not describe a single Llama-3.3-70B point as a completed scale analysis.
