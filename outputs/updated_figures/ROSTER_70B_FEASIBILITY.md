# Can this cluster host a 70B for logprob scoring and probing?

Flagged before any roster decision, as requested. Nothing has been started.

**Short answer: yes to both, without quantisation. The binding constraint is not hardware,
it is that one 70B does not fix the confound it would be added to fix.**

## Hardware

| resource | finding | verdict |
|---|---|---|
| GPU count per node | `mit_preemptable` has `gpu:h200:8` and `gpu:h100:8` nodes; `mit_normal_gpu` has `gpu:h200:8` (6 h limit) | multi-GPU available |
| GPU memory | H200 = 141 GB, H100 = 80 GB, L40S = 48 GB | see below |
| 70B bf16 weights | ~140 GB (measured: Qwen2.5-14B is 28 GB on disk, so 2 GB/B) | needs 2 GPUs |
| host RAM | 2,055 GB per node | ample for offload |
| disk | `~/.cache/huggingface` is a symlink to `/orcd/scratch/bcs/002/aimeeyu/.cache/huggingface`, on a filesystem with **75 TB free** | not a constraint |

The home quota is nearly full (66 GB used of a 195 GB limit, so ~129 GB free) and would NOT
hold a 140 GB checkpoint. It does not have to: the cache resolves to scratch. Worth knowing
before anyone sets `HF_HOME` explicitly and breaks it.

Allocation for a 70B in bf16:
- **2x H200 = 282 GB.** Comfortable; room for KV cache and activations.
- **2x H100 = 160 GB.** Workable but tight with long prompts.
- **1x H200 = 141 GB.** Do not. Weights alone are ~140 GB.
- Quantisation is not needed, and should be avoided for this project specifically: 8-bit or
  4-bit weights change the hidden states, so a quantised model's probe results are not
  comparable to the bf16 7B models already in the set. If quantisation ever becomes
  necessary, the probe comparison has to be re-run quantised across the whole roster.

## Logprob scoring: feasible

298 stories x 13 templates = 3,874 rated items, each one forward pass with digit logprobs.
On 2x H200 this is on the order of 1-2 hours, inside the 6 h `mit_normal_gpu` limit.
`03_behavioral.py` already resumes from a partial `raw_*.csv`, so preemption costs progress
but not the run.

One thing to check per model before trusting the output: the digit-token guard added after
the Mistral/Zephyr collapse. Llama-3.3-70B uses a BPE tokenizer, so it should map digits to
distinct single tokens, but the guard in `_digit_token_ids` will raise rather than fabricate
if it does not. That check is now automatic and must not be bypassed.

## Probing: feasible

- **Storage.** 80 layers + embeddings, hidden 8192, 298 items, 4 pooled variants, float16:
  ~1.6 GB per model. The 7B files are 376 MB, so this is 4x larger and irrelevant against
  75 TB.
- **Extraction.** One forward pass per story with `output_hidden_states=True`, 298 passes.
  Minutes, not hours. Memory during extraction is the concern, not compute: 80 layers of
  hidden states for a long prompt must be moved to CPU per item rather than accumulated on
  device.
- **Probe fitting.** This is the pleasant surprise. `_rowspace_project` in `02_probe.py`
  reduces each layer to the training rank, which is at most 238 with 298 items under
  GroupKFold, so the logistic fit costs the same at 8192 dimensions as at 4096. Cost scales
  with the layer count, not the width: roughly 2.4x the current per-model probe time, still
  CPU-only and still minutes.

## The real caveat, which is not about hardware

The reason to add a 70B is that the open roster tops out at 14B while GPT-4o and Opus sit in
the set, so open-vs-closed is confounded with scale, and "did you try a big one?" is the
first question the work will get. That reasoning is right. But adding **one** 70B does not
resolve it — it gives a single point at 70B, and a single point cannot separate a scale
effect from a model-family effect.

Two things follow:

1. To break the confound, the large models need to span at least two families (for example
   Llama-3.3-70B and Qwen2.5-72B), so that "large" is not synonymous with "Llama". Three
   points would let scale be fitted rather than asserted.
2. J3 gives an independent reason to want this. Across the 20 current models the intent
   contrast tracks instruction tuning far more strongly than parameter count: base models
   sit near zero and instruct models carry the whole effect. If that is the real axis, then
   the scale question is answered by adding large models **in base/instruct pairs**, not by
   adding the largest single model available. Llama-3.3-70B ships instruct-only, which makes
   Qwen2.5-72B and its base the more informative pair.

Also worth noting for the probing side specifically: an 80-layer model cannot be compared
against a 33-layer model by absolute layer index. The peak-layer analyses would need to move
to relative depth (layer / n_layers) before a 70B can be read alongside the 7Bs, and that
change affects the existing figures, not just the new model.

## Recommendation

Hardware is not the reason to wait. The decision to make first is which large models, in
which base/instruct configuration, and whether the peak-layer analyses move to relative
depth — because that choice changes existing figures. A single Llama-3.3-70B run is
affordable and would answer the reviewer question narrowly, but it will not support a scale
claim on its own, and it should not be described as if it does.
