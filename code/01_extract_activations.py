#!/usr/bin/env python3
"""
01_extract_activations.py  --  NO TRAINING. Forward passes only.

For each open-weight model, run every stimulus through a single forward pass and
save the hidden state at every layer. We store two pooling variants:
  - last-token hidden state  (standard for decoder-only probing)
  - mean-pooled over tokens  (robustness comparison)

IMPORTANT: representational analysis (Levels 2-3) requires OPEN-WEIGHT models,
because closed APIs (GPT, Claude, Gemini) do not expose hidden states. Use the
HuggingFace models below. Closed APIs can still be used for the behavioral level
(see 03_behavioral.py).

Output: outputs/acts/<model_tag>.npz  with arrays
  last  : [n_stories, n_layers, hidden]
  mean  : [n_stories, n_layers, hidden]
  story_id : [n_stories]  (aligned index into the master CSV)

Run on a GPU box (lab cluster / Colab). This script does not run in the chat sandbox.
"""
import os, csv, argparse, numpy as np

# Suggested model ladder (scale axis within & across families). Edit freely.
DEFAULT_MODELS = [
    "Qwen/Qwen2.5-0.5B",   "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-7B",
    "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.1-8B",
    # add instruct variants to test tuning effect, e.g. "...-Instruct"
]

def load_stimuli(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    return rows

def load_clause_offsets(path):
    """story_id -> {belief_end, action_end, ...} character offsets, or {} if unavailable."""
    if not path or not os.path.exists(path):
        return {}
    out = {}
    for r in csv.DictReader(open(path)):
        out[r["story_id"]] = {k: int(r[k]) for k in
                              ("belief_start", "belief_end", "action_start",
                               "action_end", "outcome_start", "outcome_end")}
        out[r["story_id"]]["method"] = r["method"]
    return out


def _token_at_char(offsets, char_end):
    """
    Index of the LAST token that ends at or before char_end.

    Character offsets cannot be precomputed as token indices because every tokenizer
    segments differently, so the mapping is resolved here, per model, from
    return_offsets_mapping. Special tokens carry (0, 0) and are skipped.
    """
    best = None
    for t, (s, e) in enumerate(offsets):
        if e == 0 and s == 0:
            continue
        if e <= char_end:
            best = t
        else:
            break
    return best


def extract_for_model(model_name, rows, out_dir, batch_size=8, max_len=512,
                      clause_offsets=None):
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(
        model_name, output_hidden_states=True,
        torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True)
    model.eval()

    want_clause = bool(clause_offsets)
    texts = [r["text"] for r in rows]
    last_all, mean_all = [], []
    belief_all, action_all = [], []
    n_missing = 0
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            brows = rows[i:i+batch_size]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_len,
                      return_offsets_mapping=want_clause)
            offs = enc.pop("offset_mapping").tolist() if want_clause else None
            enc = {k: v.to(model.device) for k, v in enc.items()}
            out = model(**enc)
            hs = torch.stack(out.hidden_states, dim=1)  # [B, L, T, H]
            mask = enc["attention_mask"].unsqueeze(1).unsqueeze(-1)  # [B,1,T,1]
            # last non-pad token index per sequence
            lengths = enc["attention_mask"].sum(1) - 1
            idx = lengths.view(-1,1,1,1).expand(-1, hs.size(1), 1, hs.size(-1))
            last = hs.gather(2, idx).squeeze(2)                       # [B,L,H]
            mean = (hs*mask).sum(2) / mask.sum(2).clamp(min=1)        # [B,L,H]
            last_all.append(last.float().cpu().numpy())
            mean_all.append(mean.float().cpu().numpy())

            if want_clause:
                for key, sink in (("belief_end", belief_all), ("action_end", action_all)):
                    pos = []
                    for b, r in enumerate(brows):
                        co = clause_offsets.get(r["story_id"])
                        t = _token_at_char(offs[b], co[key]) if co else None
                        if t is None:
                            t = int(lengths[b].item())   # fall back to the final token
                            n_missing += 1
                        pos.append(t)
                    p = torch.tensor(pos, device=hs.device)
                    gi = p.view(-1,1,1,1).expand(-1, hs.size(1), 1, hs.size(-1))
                    sink.append(hs.gather(2, gi).squeeze(2).float().cpu().numpy())

    last_all = np.concatenate(last_all); mean_all = np.concatenate(mean_all)
    tag = model_name.split("/")[-1]
    os.makedirs(out_dir, exist_ok=True)
    arrays = dict(last=last_all, mean=mean_all,
                  story_id=np.array([r["story_id"] for r in rows]))
    if want_clause:
        arrays["belief_last"] = np.concatenate(belief_all)
        arrays["action_last"] = np.concatenate(action_all)
    np.savez_compressed(os.path.join(out_dir, f"{tag}.npz"), **arrays)
    extra = f" +belief_last/action_last (fallbacks: {n_missing})" if want_clause else ""
    print(f"{tag}: saved {last_all.shape} (n,layers,hidden){extra}", flush=True)

if __name__ == "__main__":
    here = os.path.dirname(__file__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(here, "..", "dataset", "master", "moral_2x2_master.csv"))
    ap.add_argument("--out", default=os.path.join(here, "..", "outputs", "acts"))
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--clause-offsets", default=None,
                    help="dataset/master/clause_offsets.csv -- additionally save "
                         "belief_last and action_last pooling, so intent can be probed at "
                         "the belief clause, BEFORE the harm is mentioned in the text")
    ap.add_argument("--dry-run", action="store_true",
                    help="print plan only (no weight download / no GPU)")
    a = ap.parse_args()
    rows = load_stimuli(a.csv)
    clause = load_clause_offsets(a.clause_offsets)
    if a.clause_offsets:
        print(f"clause offsets: {len(clause)} stories from {a.clause_offsets}")
    if a.dry_run:
        print(f"=== ACTIVATION EXTRACTION PLAN (dry-run) ===")
        print(f"stimuli={len(rows)}  out={a.out}")
        print(f"{'model':42} {'~VRAM(bf16)':>12}")
        for m in a.models:
            # rough: params from name * 2 GB + overhead
            import re
            mm = re.search(r"(\d+\.?\d*)\s*[bB]\b", m)
            gb = (float(mm.group(1)) * 2 + 2) if mm else float("nan")
            print(f"{m:42} {gb:10.0f}GB" if mm else f"{m:42} {'?':>12}")
        print(f"\nLaunch:\n  JOBNAME=acts PART=mit_preemptable bash engaging/submit_gpu.sh "
              f"\"python code/01_extract_activations.py --models {' '.join(a.models)}\"")
    else:
        for m in a.models:
            extract_for_model(m, rows, a.out, clause_offsets=clause)
