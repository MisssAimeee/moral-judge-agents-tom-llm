#!/usr/bin/env python3
"""
17_steering.py  --  Roadmap #6 (the prize): CAUSAL steering. Test whether adding /
subtracting an INTENT direction in the residual stream causally moves the model's
behavioral intent-vs-outcome contrast in the predicted direction.

Method (inference/steering only -- NO training):
  1. Fit an intent direction at the peak layer: difference-of-means of the last-token
     residual, guilty - innocent (attempted+intentional  vs  neutral+accidental).
     Peak layer is taken from outputs/probe/<tag>_probe.csv if present, else a default.
  2. During generation/scoring, add  alpha * unit(dir) * typical_norm  to that layer's
     output via a forward hook, and re-measure the contrast (attempted - accidental)
     using the same logprob-EV scoring as 03_behavioral.py.
  3. CONTROLS: an OUTCOME direction (harm - no_harm) and a RANDOM direction. Steering
     with intent should raise intent-reliance; the controls should NOT.
  4. COHERENCE: perplexity of a fixed probe sentence at each alpha, so we can tell a
     real behavioral shift from "the steering just broke the model".

A clean, direction-specific, coherence-preserving effect = a causal
representation -> behavior claim.

Modes
  (default) --dry-run : prints the steering plan; downloads nothing.
  --run               : loads the model, fits directions, sweeps alpha, writes outputs.

Outputs (with --run)
  outputs/experiments/steering_<tag>.csv           (direction, alpha, contrast, dppl)
  outputs/experiments/steering_<tag>.png           (contrast vs alpha, line/direction)
"""
import os, sys, csv, argparse, importlib.util
from collections import defaultdict

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CODE_DIR)
import tom_common as tc  # noqa: E402
import numpy as np  # noqa: E402

MASTER_CSV = os.path.join(tc.ROOT, "dataset", "master", "moral_2x2_master.csv")
PROBE_DIR = os.path.join(tc.ROOT, "outputs", "probe")
OUT_DIR = os.path.join(tc.ROOT, "outputs", "experiments")

GUILTY = {"attempted", "intentional"}      # intent present
HARM = {"accidental", "intentional"}       # bad outcome present
PROBE_SENTENCE = ("The teacher explained the lesson to the class and then asked "
                  "the students to work quietly on their assignments.")


def _load_behavioral():
    spec = importlib.util.spec_from_file_location(
        "behavioral", os.path.join(CODE_DIR, "03_behavioral.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def load_dataset():
    with open(MASTER_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def peak_intent_layer(tag, default_frac=0.6, n_layers=None):
    """Read the intent-probe peak layer from 02_probe output, else a depth default."""
    import glob
    cands = glob.glob(os.path.join(PROBE_DIR, f"*{tag}*_probe.csv"))
    if cands:
        best = None
        for r in csv.DictReader(open(cands[0])):
            if r["target"] == "intent":
                acc = float(r["cv_acc"])
                if best is None or acc > best[1]:
                    best = (int(r["layer"]), acc)
        if best:
            return best[0], f"probe peak (acc={best[1]:.3f})"
    if n_layers:
        return int(default_frac * n_layers), f"default {default_frac:.0%} depth"
    return None, "unknown (needs model)"


def digit_token_ids(tok, s_min, s_max):
    ids = {}
    for d in range(int(s_min), int(s_max) + 1):
        t = tok.encode(str(d), add_special_tokens=False)
        if t:
            ids[d] = t[0]
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--template", default="human_verbatim")
    ap.add_argument("--layer", type=int, default=None,
                    help="steering layer (hidden_states index); default = probe peak")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[-1.0, -0.5, 0.0, 0.5, 1.0],
                    help="steering coefficients (x typical residual norm)")
    ap.add_argument("--run", action="store_true",
                    help="load model + steer (default: dry-run plan)")
    a = ap.parse_args()

    tag = a.model.split("/")[-1]

    if not a.run:
        L, how = peak_intent_layer(tag)
        layer_str = (str(a.layer) if a.layer is not None
                     else (str(L) if L is not None
                           else "probe-peak or ~60% depth (resolved at --run)"))
        print("\n=== STEERING PLAN (dry-run: no weights downloaded) ===")
        print(f"  model        : {a.model}")
        print(f"  template     : {a.template}")
        print(f"  steer layer  : {layer_str}  ({how})")
        print(f"  directions   : intent (guilty-innocent), outcome (harm-no_harm) [ctrl],"
              f" random [ctrl]")
        print(f"  alpha sweep  : {a.alphas}  (x mean residual norm at the layer)")
        print(f"  metric       : contrast = blame(attempted) - blame(accidental) per alpha")
        print(f"  coherence    : perplexity of a fixed probe sentence per alpha")
        print(f"  prediction   : intent dir raises contrast; outcome/random do NOT; ppl stable")
        print("\nLaunch for real with:")
        print(f"  JOBNAME=steer bash engaging/submit_gpu.sh "
              f"\"python code/experiments/17_steering.py --run --model {a.model}\"")
        return

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    beh = _load_behavioral()
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_dataset()

    print(f"Loading {a.model} ...")
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.float16, output_hidden_states=True,
        device_map="auto", trust_remote_code=True)
    model.eval()
    n_layers = model.config.num_hidden_layers
    L = a.layer if a.layer is not None else peak_intent_layer(tag, n_layers=n_layers)[0]
    print(f"  {n_layers} layers; steering at hidden_states[{L}] "
          f"(decoder block {L-1} output)")

    # locate the decoder blocks (Llama/Qwen/OLMo/Mistral: model.model.layers)
    blocks = model.model.layers
    steer_block = blocks[max(0, L - 1)]

    def fmt(row, tmpl):
        prompt, s_min, s_max = beh.build_prompt(row["text"], tmpl, row["source"])
        text = (tok.apply_chat_template([{"role": "user", "content": prompt}],
                                        tokenize=False, add_generation_prompt=True)
                if tok.chat_template else prompt)
        return text, s_min, s_max

    # ---- 1. fit directions from last-token residual at layer L ----
    print("  fitting directions (guilty-innocent, harm-no_harm) ...")
    sums = defaultdict(lambda: None); counts = defaultdict(int); norms = []
    with torch.no_grad():
        for row in rows:
            text, _, _ = fmt(row, a.template)
            enc = tok(text, return_tensors="pt").to(model.device)
            hs = model(**enc).hidden_states[L][0, -1, :].float().cpu().numpy()
            norms.append(float(np.linalg.norm(hs)))
            gi = "guilty" if row["condition"] in GUILTY else "innocent"
            ho = "harm" if row["condition"] in HARM else "no_harm"
            for key in (gi, ho):
                sums[key] = hs if sums[key] is None else sums[key] + hs
                counts[key] += 1
    typ_norm = float(np.mean(norms))

    def mean(k):
        return sums[k] / counts[k]
    intent_dir = mean("guilty") - mean("innocent")
    outcome_dir = mean("harm") - mean("no_harm")
    rng = np.random.default_rng(0)
    random_dir = rng.standard_normal(intent_dir.shape)
    dirs = {"intent": intent_dir, "outcome": outcome_dir, "random": random_dir}
    dirs = {k: v / (np.linalg.norm(v) + 1e-8) for k, v in dirs.items()}
    print(f"  typical residual norm @L{L} = {typ_norm:.1f}")

    # ---- steering hook ----
    state = {"vec": None}

    def hook(_m, _inp, out):
        if state["vec"] is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        h = h + state["vec"].to(h.dtype)
        return (h,) + out[1:] if isinstance(out, tuple) else h
    steer_block.register_forward_hook(hook)

    def score_contrast(vec):
        state["vec"] = None if vec is None else torch.tensor(vec, device=model.device)
        by_scen = defaultdict(dict)
        with torch.no_grad():
            for row in rows:
                text, s_min, s_max = fmt(row, a.template)
                enc = tok(text, return_tensors="pt").to(model.device)
                logits = model(**enc).logits[0, -1, :]
                ids = digit_token_ids(tok, s_min, s_max)
                vals = list(ids)
                lp = torch.tensor([logits[ids[d]].item() for d in vals])
                p = torch.softmax(lp, 0).tolist()
                ev = sum(p[i] * vals[i] for i in range(len(vals)))
                norm = (ev - s_min) / (s_max - s_min)
                by_scen[tc.scenario_of(row["story_id"])][row["condition"]] = norm
        diffs = [c["attempted"] - c["accidental"] for c in by_scen.values()
                 if "attempted" in c and "accidental" in c]
        return float(np.mean(diffs)) if diffs else float("nan")

    def perplexity(vec):
        state["vec"] = None if vec is None else torch.tensor(vec, device=model.device)
        enc = tok(PROBE_SENTENCE, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
        return float(torch.exp(out.loss).item())

    # ---- 2-4. sweep ----
    results = []
    base_c = score_contrast(None)
    base_ppl = perplexity(None)
    print(f"  baseline contrast={base_c:+.3f}  ppl={base_ppl:.1f}")
    for dname, d in dirs.items():
        for al in a.alphas:
            vec = (al * typ_norm) * d
            c = score_contrast(vec)
            ppl = perplexity(vec)
            results.append(dict(direction=dname, alpha=al, contrast=round(c, 4),
                                dcontrast=round(c - base_c, 4), ppl=round(ppl, 2),
                                dppl=round(ppl - base_ppl, 2)))
            print(f"    {dname:8} a={al:+.2f}  contrast={c:+.3f} "
                  f"(d={c-base_c:+.3f})  ppl={ppl:.1f}")

    out_csv = os.path.join(OUT_DIR, f"steering_{tag}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["direction", "alpha", "contrast",
                                          "dcontrast", "ppl", "dppl"])
        w.writeheader(); w.writerows(results)
    print(f"\nwrote {os.path.relpath(out_csv, tc.ROOT)}  (baseline contrast={base_c:+.3f})")
    _plot(results, tag, base_c)


def _plot(results, tag, base_c):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (skip figure: {e})"); return
    plt.figure(figsize=(7, 5))
    for dname in ["intent", "outcome", "random"]:
        pts = sorted([r for r in results if r["direction"] == dname],
                     key=lambda r: r["alpha"])
        if pts:
            plt.plot([p["alpha"] for p in pts], [p["contrast"] for p in pts],
                     marker="o", label=dname)
    plt.axhline(base_c, color="gray", ls="--", lw=0.8, label="baseline")
    plt.xlabel("steering coefficient alpha (x residual norm)")
    plt.ylabel("contrast (attempted - accidental)")
    plt.title(f"Causal steering: {tag}\n(intent should move contrast; controls should not)")
    plt.legend(); plt.tight_layout()
    out = os.path.join(OUT_DIR, f"steering_{tag}.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"wrote {os.path.relpath(out, tc.ROOT)}")


if __name__ == "__main__":
    main()
