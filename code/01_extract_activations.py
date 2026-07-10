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

def extract_for_model(model_name, rows, out_dir, batch_size=8, max_len=512):
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(
        model_name, output_hidden_states=True,
        torch_dtype=torch.float16, device_map="auto")
    model.eval()

    texts = [r["text"] for r in rows]
    last_all, mean_all = [], []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_len).to(model.device)
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
    last_all = np.concatenate(last_all); mean_all = np.concatenate(mean_all)
    tag = model_name.split("/")[-1]
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"{tag}.npz"),
                        last=last_all, mean=mean_all,
                        story_id=np.array([r["story_id"] for r in rows]))
    print(f"{tag}: saved {last_all.shape} (n,layers,hidden)")

if __name__ == "__main__":
    here = os.path.dirname(__file__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(here, "..", "dataset", "master", "moral_2x2_master.csv"))
    ap.add_argument("--out", default=os.path.join(here, "..", "outputs", "acts"))
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    a = ap.parse_args()
    rows = load_stimuli(a.csv)
    for m in a.models:
        extract_for_model(m, rows, a.out)
