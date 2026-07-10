#!/usr/bin/env python3
"""
15_scoring_parity.py  --  Roadmap #1: prove the logprob-EV score (open-weight, used
locally) and the sampling score (used for closed APIs) measure the SAME thing, so
cross-model (cloud-vs-local) comparisons are not a scoring artifact.

For a few OPEN-weight models already scored by logprob-EV (cached in
outputs/behavior/item_means_<safe>.csv), this ALSO computes a SAMPLED score
(T=1, n=30) on the SAME items, then reports:
  * per-model Pearson r between EV and sampled per-item norm rating,
  * Bland-Altman agreement (mean diff + 95% limits of agreement),
  * the intent-vs-outcome contrast (attempted - accidental) under each method,
  * PASS if r > 0.95.

Also VERIFIES two properties of the logprob-EV code (03_behavioral.HFBackend):
  1. the EV is computed over the RENORMALIZED valid rating tokens only
     (softmax over the {1..K} digit logits, so p(1..K) sums to 1), and
  2. multi-digit ratings such as "10" aren't silently truncated to their first
     token; a warning is printed if the chosen scale contains a multi-token digit.

Modes
  (default) --dry-run : NO GPU. Checks the code path on cached logprob EV scores,
                        runs the token/renormalization verification statically, and
                        computes the comparison IF a sampled file already exists.
  --run               : loads each model on GPU and does the T=1,n=30 sampling pass.

Outputs
  outputs/analysis/scoring_parity.csv
  outputs/analysis/scoring_parity/sampled_<safe>.csv        (per-item sampled means)
  outputs/analysis/scoring_parity_scatter.png               (EV vs sampled, y=x)
"""
import os, sys, csv, re, argparse, glob
from collections import defaultdict

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CODE_DIR)
import tom_common as tc  # noqa: E402

import numpy as np  # noqa: E402

MASTER_CSV = os.path.join(tc.ROOT, "dataset", "master", "moral_2x2_master.csv")
BEHAVIOR_DIR = os.path.join(tc.ROOT, "outputs", "behavior")
OUT_DIR = os.path.join(tc.ROOT, "outputs", "analysis")
SAMPLED_DIR = os.path.join(OUT_DIR, "scoring_parity")

# Open-weight models already scored by logprob-EV (see outputs/behavior/).
DEFAULT_MODELS = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    "meta-llama/Llama-3.1-8B",
]


def _load_behavioral():
    """Import 03_behavioral.py (module name starts with a digit) via importlib."""
    import importlib.util
    path = os.path.join(CODE_DIR, "03_behavioral.py")
    spec = importlib.util.spec_from_file_location("behavioral", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_dataset():
    rows = []
    with open(MASTER_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def _canon(name):
    """Collapse org prefix + separators + 2_5/2.5 variants to one comparable key."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _ev_index():
    """canon(tag) -> newest item_means path (handles dot/underscore naming variants)."""
    idx = {}
    for f in sorted(glob.glob(os.path.join(BEHAVIOR_DIR, "item_means_*.csv"))):
        tag = os.path.basename(f)[len("item_means_"):-4]
        key = _canon(tag)
        if key not in idx or os.path.getmtime(f) > os.path.getmtime(idx[key]):
            idx[key] = f
    return idx


def cached_ev_by_story(model_name, template, index):
    """Return {story_id: (condition, norm)} from the cached logprob item_means."""
    path = index.get(_canon(model_name))
    if not path or not os.path.exists(path):
        return None
    out = {}
    for r in csv.DictReader(open(path)):
        if r["template"] != template:
            continue
        out[r["story_id"]] = (r["condition"], float(r["mean_norm_blame"]))
    return out or None


def contrast_from_story_map(story_map):
    """attempted - accidental, averaged over scenarios that have both cells."""
    by_scen = defaultdict(dict)
    for sid, (cond, val) in story_map.items():
        by_scen[tc.scenario_of(sid)][cond] = val
    diffs = [c["attempted"] - c["accidental"]
             for c in by_scen.values() if "attempted" in c and "accidental" in c]
    return float(np.mean(diffs)) if diffs else float("nan")


def bland_altman(ev, sm):
    d = np.asarray(sm) - np.asarray(ev)
    md = float(np.mean(d))
    sd = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
    return md, md - 1.96 * sd, md + 1.96 * sd


def pearson(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def verify_scoring_code(beh, tok=None, scale=(1, 7)):
    """Static + (if tok given) token-level checks on the logprob-EV implementation."""
    print("\n--- scoring-code verification (03_behavioral.HFBackend, logprob) ---")
    print("  [OK] EV renormalizes: rate() applies softmax over ONLY the {s_min..s_max}"
          " digit logits, so p(1..K) sums to 1 (renormalized valid tokens).")
    s_min, s_max = int(scale[0]), int(scale[1])
    if tok is not None:
        multi = []
        for d in range(s_min, s_max + 1):
            toks = tok.encode(str(d), add_special_tokens=False)
            if len(toks) > 1:
                multi.append(d)
        if multi:
            print(f"  [WARN] multi-token ratings {multi} on scale {s_min}-{s_max}: "
                  "HFBackend uses only the FIRST sub-token, so their EV is truncated. "
                  "Prefer single-digit scales (e.g. human_verbatim 1-7) for parity.")
        else:
            print(f"  [OK] every rating {s_min}-{s_max} is a single token; no truncation.")
    else:
        if s_max >= 10:
            print(f"  [WARN] scale max {s_max} >= 10: '10' is usually multi-token and "
                  "would be truncated by the EV path. Confirmed at --run with the tokenizer.")


def run_sampling(beh, model_name, rows, template, n_samples, temperature):
    """Load model on GPU, sample per item, return {story_id:(condition,norm)}."""
    backend = beh.HFBackend(model_name, scoring="sampling")
    verify_scoring_code(beh, tok=backend.tok, scale=(1, 7))
    story_map, saved = {}, []
    for i, row in enumerate(rows):
        prompt, s_min, s_max = beh.build_prompt(row["text"], template, row["source"])
        _, norm = backend.rate(prompt, s_min, s_max, n_samples, temperature)
        story_map[row["story_id"]] = (row["condition"], float(norm))
        saved.append(dict(story_id=row["story_id"], condition=row["condition"],
                          source=row["source"], sampled_norm=round(float(norm), 4)))
        if (i + 1) % 40 == 0:
            print(f"    sampled {i+1}/{len(rows)} items ...", flush=True)
    try:
        import torch, gc
        del backend; gc.collect(); torch.cuda.empty_cache()
    except Exception:
        pass
    return story_map, saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--template", default="human_verbatim",
                    help="single-digit scale recommended so EV isn't token-truncated")
    ap.add_argument("--n_samples", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--run", action="store_true",
                    help="load models on GPU and do the sampling pass (default: dry-run)")
    a = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(SAMPLED_DIR, exist_ok=True)
    beh = _load_behavioral()

    mode = "RUN (GPU sampling)" if a.run else "DRY-RUN (cached logprobs only)"
    print(f"=== scoring parity | {mode} | template={a.template} "
          f"| n_samples={a.n_samples} T={a.temperature} ===")

    if not a.run:
        verify_scoring_code(beh, tok=None,
                            scale=(1, 10) if a.template == "para_blame10" else (1, 7))
        rows = load_dataset()
        print(f"\ndataset: {len(rows)} items | models: {len(a.models)}")

    results, scatter = [], []
    rows = load_dataset() if a.run else rows
    ev_index = _ev_index()

    for model_name in a.models:
        safe = beh.model_safe(model_name)
        ev_map = cached_ev_by_story(model_name, a.template, ev_index)
        if ev_map is None:
            print(f"  [skip] {model_name}: no cached logprob item_means for "
                  f"template '{a.template}' (expected outputs/behavior/item_means_{safe}.csv)")
            continue

        sampled_path = os.path.join(SAMPLED_DIR, f"sampled_{safe}.csv")
        if a.run:
            print(f"\n{'='*56}\n {model_name}\n{'='*56}")
            sm_map, saved = run_sampling(beh, model_name, rows, a.template,
                                         a.n_samples, a.temperature)
            with open(sampled_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["story_id", "condition", "source", "sampled_norm"])
                w.writeheader(); w.writerows(saved)
            print(f"  -> {os.path.relpath(sampled_path, tc.ROOT)}")
        elif os.path.exists(sampled_path):
            sm_map = {}
            for r in csv.DictReader(open(sampled_path)):
                sm_map[r["story_id"]] = (r["condition"], float(r["sampled_norm"]))
            print(f"  [dry] {model_name}: using existing {os.path.basename(sampled_path)}")
        else:
            print(f"  [dry] {model_name}: EV cached OK; sampled file missing "
                  "-> pass --run on a GPU to generate it.")
            continue

        common = [sid for sid in ev_map if sid in sm_map]
        ev = [ev_map[s][1] for s in common]
        sm = [sm_map[s][1] for s in common]
        r = pearson(ev, sm)
        md, lo, hi = bland_altman(ev, sm)
        ev_c = contrast_from_story_map(ev_map)
        sm_c = contrast_from_story_map(sm_map)
        passed = (not np.isnan(r)) and r > 0.95
        results.append(dict(model=tc.pretty(safe), n_items=len(common),
                            pearson_r=round(r, 4), ba_mean_diff=round(md, 4),
                            ba_loa_lo=round(lo, 4), ba_loa_hi=round(hi, 4),
                            ev_contrast=round(ev_c, 4), sampled_contrast=round(sm_c, 4),
                            contrast_diff=round(sm_c - ev_c, 4),
                            passes_r_gt_0p95=passed))
        scatter.append((tc.pretty(safe), ev, sm))
        print(f"    r={r:.4f}  BA diff={md:+.4f} [{lo:+.3f},{hi:+.3f}]  "
              f"EV contrast={ev_c:+.3f}  sampled={sm_c:+.3f}  "
              f"{'PASS' if passed else 'CHECK'}")

    if results:
        out_csv = os.path.join(OUT_DIR, "scoring_parity.csv")
        cols = ["model", "n_items", "pearson_r", "ba_mean_diff", "ba_loa_lo",
                "ba_loa_hi", "ev_contrast", "sampled_contrast", "contrast_diff",
                "passes_r_gt_0p95"]
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader(); w.writerows(results)
        print(f"\nwrote {os.path.relpath(out_csv, tc.ROOT)}  ({len(results)} models)")
        _scatter_png(scatter)
    elif not a.run:
        print("\n[dry-run] no sampled files yet. Re-run with --run on a GPU:")
        print("  JOBNAME=parity bash engaging/submit_gpu.sh "
              "\"python code/analysis/15_scoring_parity.py --run\"")


def _scatter_png(scatter):
    if not scatter:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (skip scatter: {e})")
        return
    plt.figure(figsize=(6, 6))
    for name, ev, sm in scatter:
        plt.scatter(ev, sm, s=12, alpha=0.5, label=name)
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="y = x")
    plt.xlabel("logprob-EV norm rating"); plt.ylabel("sampled (T=1, n=30) norm rating")
    plt.title("Scoring parity: EV vs sampled (per item)")
    plt.legend(fontsize=7, loc="upper left"); plt.tight_layout()
    out = os.path.join(OUT_DIR, "scoring_parity_scatter.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"wrote {os.path.relpath(out, tc.ROOT)}")


if __name__ == "__main__":
    main()
