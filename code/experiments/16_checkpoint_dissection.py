#!/usr/bin/env python3
"""
16_checkpoint_dissection.py  --  Roadmap #4 (the standout novel result):
localize WHERE in the instruction-tuning pipeline the outcome-bias appears by
running the SAME behavioral scoring across release checkpoints of the SAME base.

HYPOTHESIS: the outcome-bias shift appears at the DPO / preference-optimization
stage (not SFT), and shows up as b_outcome increasing MORE than b_intent decreasing.

Pipelines (same base, staged tuning):
  OLMo-2-7B  : base -> SFT -> DPO -> Instruct        (allenai)
  Tulu-3-8B  : Llama-3.1-8B base -> SFT -> DPO -> RLVR (allenai)
  Zephyr-7B  : Mistral-7B base -> SFT -> DPO           (HuggingFaceH4 / alignment-handbook)

For each checkpoint we compute (pooled over the chosen prompt templates):
  * contrast = blame(attempted) - blame(accidental)      [the headline luck index]
  * the full 2x2 coefficients b_intent / b_outcome / b_interaction   (reuse 11_...py)
An ENTROPY / variance QC filter flags any degenerate checkpoint (near-constant
ratings, like the earlier Mistral failure) so it is NOT silently averaged in.

Modes
  (default) --dry-run : prints the checkpoint plan + token/VRAM estimate and does
                        NOT download any weights.
  --run               : downloads each checkpoint and runs logprob-EV scoring.

Outputs (only with --run)
  outputs/experiments/checkpoint_dissection.csv
  outputs/experiments/checkpoint_dissection.png   (x = stage, y = contrast, line/family)
"""
import os, sys, csv, argparse, importlib.util
from collections import defaultdict

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CODE_DIR)
import tom_common as tc  # noqa: E402
import numpy as np  # noqa: E402

MASTER_CSV = os.path.join(tc.ROOT, "dataset", "master", "moral_2x2_master.csv")
OUT_DIR = os.path.join(tc.ROOT, "outputs", "experiments")

# stage order matters for the x-axis; params_B for the VRAM estimate.
FAMILIES = {
    "OLMo-2-7B": {"params_B": 7.3, "stages": [
        ("base",     "allenai/OLMo-2-1124-7B"),
        ("SFT",      "allenai/OLMo-2-1124-7B-SFT"),
        ("DPO",      "allenai/OLMo-2-1124-7B-DPO"),
        ("Instruct", "allenai/OLMo-2-1124-7B-Instruct"),
    ]},
    "Tulu-3-8B": {"params_B": 8.0, "stages": [
        ("base",     "meta-llama/Llama-3.1-8B"),
        ("SFT",      "allenai/Llama-3.1-Tulu-3-8B-SFT"),
        ("DPO",      "allenai/Llama-3.1-Tulu-3-8B-DPO"),
        ("RLVR",     "allenai/Llama-3.1-Tulu-3-8B"),
    ]},
    "Zephyr-7B": {"params_B": 7.2, "stages": [
        ("base",     "mistralai/Mistral-7B-v0.1"),
        ("SFT",      "alignment-handbook/zephyr-7b-sft-full"),
        ("DPO",      "HuggingFaceH4/zephyr-7b-beta"),
    ]},
}


def _load(mod_file, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(CODE_DIR, mod_file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_dataset():
    with open(MASTER_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def score_checkpoint(beh, reg11, model_id, rows, templates):
    """Load one checkpoint, logprob-EV score all items, return metrics dict."""
    backend = beh.HFBackend(model_id, scoring="logprob")
    by_scen = defaultdict(dict)
    all_norms = []
    for tmpl in templates:
        for row in rows:
            prompt, s_min, s_max = beh.build_prompt(row["text"], tmpl, row["source"])
            _, norm = backend.rate(prompt, s_min, s_max, 1, 0.0)
            by_scen[(tmpl, tc.scenario_of(row["story_id"]))][row["condition"]] = float(norm)
            all_norms.append(float(norm))
    try:
        import torch, gc
        del backend; gc.collect(); torch.cuda.empty_cache()
    except Exception:
        pass

    # pool over templates+scenarios -> per-condition means (reuse 11's math)
    pooled = defaultdict(dict)
    for (tmpl, scen), conds in by_scen.items():
        for c, v in conds.items():
            pooled[f"{tmpl}:{scen}"][c] = v
    m = reg11.cell_means(pooled)
    b0, b_int, b_out, b_inter = reg11.coeffs_from_means(m)
    contrast = (m.get("attempted", float("nan")) - m.get("accidental", float("nan")))

    std = float(np.std(all_norms)) if all_norms else 0.0
    degenerate = std < 0.02  # near-constant ratings -> no usable signal
    return dict(n_items=len(all_norms), contrast=contrast, b_intent=b_int,
                b_outcome=b_out, b_interaction=b_inter, rating_std=std,
                degenerate=degenerate)


def print_plan(families, templates, n_items):
    print("\n=== CHECKPOINT DISSECTION PLAN (dry-run: no weights downloaded) ===")
    print(f"templates={templates}  items={n_items}  scoring=logprob-EV (1 forward/item)\n")
    print(f"{'family':11} {'stage':9} {'model_id':46} {'~VRAM':>7} {'fwd_passes':>11}")
    total_passes = 0
    for fam, spec in families.items():
        vram = spec["params_B"] * 2 + 2  # bf16 weights + overhead, GB
        for stage, mid in spec["stages"]:
            passes = n_items * len(templates)
            total_passes += passes
            print(f"{fam:11} {stage:9} {mid:46} {vram:5.0f}GB {passes:>11}")
    approx_tokens = total_passes * 220  # ~220 tok/prompt avg
    print(f"\ntotal checkpoints={sum(len(s['stages']) for s in families.values())}  "
          f"total forward-passes={total_passes}  ~input-tokens={approx_tokens:,}")
    print("A single 7-8B checkpoint fits on one 24GB+ GPU; run sequentially "
          "(weights freed between checkpoints).")
    print("\nLaunch for real with:\n  JOBNAME=ckpt bash engaging/submit_gpu.sh "
          "\"python code/experiments/16_checkpoint_dissection.py --run\"")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(FAMILIES),
                    help="which families to run (default: all)")
    ap.add_argument("--templates", nargs="+", default=["human_verbatim"])
    ap.add_argument("--run", action="store_true",
                    help="download + run checkpoints (default: dry-run plan only)")
    a = ap.parse_args()

    families = {k: v for k, v in FAMILIES.items() if k in a.models}
    if not families:
        print(f"no matching families in {list(FAMILIES)}"); return
    rows = load_dataset()

    if not a.run:
        print_plan(families, a.templates, len(rows))
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    beh = _load("03_behavioral.py", "behavioral")
    reg11 = _load("11_interaction_regression.py", "interaction_reg")

    results = []
    for fam, spec in families.items():
        for i, (stage, mid) in enumerate(spec["stages"]):
            print(f"\n{'='*60}\n {fam} / {stage} : {mid}\n{'='*60}")
            try:
                met = score_checkpoint(beh, reg11, mid, rows, a.templates)
            except Exception as e:
                print(f"!! FAILED {mid}: {e}")
                import traceback; traceback.print_exc()
                continue
            met.update(family=fam, stage=stage, stage_idx=i, model_id=mid)
            results.append(met)
            flag = "  [DEGENERATE-flagged]" if met["degenerate"] else ""
            print(f"  contrast={met['contrast']:+.3f}  b_intent={met['b_intent']:+.3f}  "
                  f"b_outcome={met['b_outcome']:+.3f}  b_interaction={met['b_interaction']:+.3f}"
                  f"{flag}")

    if not results:
        print("no checkpoints scored."); return

    cols = ["family", "stage", "stage_idx", "model_id", "n_items", "contrast",
            "b_intent", "b_outcome", "b_interaction", "rating_std", "degenerate"]
    out_csv = os.path.join(OUT_DIR, "checkpoint_dissection.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            w.writerow({c: (round(r[c], 4) if isinstance(r[c], float) else r[c]) for c in cols})
    print(f"\nwrote {os.path.relpath(out_csv, tc.ROOT)}  ({len(results)} checkpoints)")
    _plot(results)


def _plot(results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (skip figure: {e})"); return
    plt.figure(figsize=(7, 5))
    fams = sorted({r["family"] for r in results})
    for fam in fams:
        pts = sorted([r for r in results if r["family"] == fam], key=lambda r: r["stage_idx"])
        xs = [p["stage"] for p in pts]
        ys = [p["contrast"] for p in pts]
        plt.plot(range(len(xs)), ys, marker="o", label=fam)
        for j, p in enumerate(pts):
            if p["degenerate"]:
                plt.scatter([j], [p["contrast"]], marker="x", s=80, color="red", zorder=5)
    # use the longest family's stage names for the x tick labels
    longest = max(results, key=lambda r: r["stage_idx"])
    n = longest["stage_idx"] + 1
    plt.xticks(range(n), ["base", "SFT", "DPO", "RLVR/Instruct"][:n])
    plt.axhline(0, color="gray", lw=0.6)
    plt.ylabel("contrast (attempted - accidental)")
    plt.xlabel("instruction-tuning pipeline stage")
    plt.title("Where does outcome-bias shift? (red x = degenerate/QC-flagged)")
    plt.legend(); plt.tight_layout()
    out = os.path.join(OUT_DIR, "checkpoint_dissection.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"wrote {os.path.relpath(out, tc.ROOT)}")


if __name__ == "__main__":
    main()
