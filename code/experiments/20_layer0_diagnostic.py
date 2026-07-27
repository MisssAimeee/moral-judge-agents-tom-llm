#!/usr/bin/env python3
"""
20_layer0_diagnostic.py -- Phase 1 / Task C1: the layer-0 read-off.

No new compute: reads the per-layer accuracies already in outputs/probe/*_probe.csv.

Layer 0 of a decoder-only model is the token-embedding output, BEFORE any attention or
MLP has run. Nothing has been contextually computed yet. So accuracy at layer 0 is an
upper bound on what a bag-of-embeddings (i.e. surface lexis) affords:

  outcome ~0.99 at layer 0   -> the probe is reading harm WORDS, not a moral representation
  intent  ~chance at layer 0, rising in mid/late layers
                             -> intent is COMPUTED, which is the shape we want

Outputs
  outputs/probe/layer0_diagnostic.csv
  outputs/probe/layerwise_curves.png
"""
import os, csv, glob, argparse
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PROBE_DIR = os.path.join(ROOT, "outputs", "probe")

# plotting order: Qwen ladder base/instruct pairs, then OLMo
ORDER = [
    "Qwen2.5-0.5B", "Qwen2.5-0.5B-Instruct",
    "Qwen2.5-1.5B", "Qwen2.5-1.5B-Instruct",
    "Qwen2.5-7B", "Qwen2.5-7B-Instruct",
    "OLMo-2-1124-7B", "OLMo-2-1124-7B-Instruct",
]


def load_probe(path):
    """-> {target: [(layer, acc, chance), ...]} sorted by layer."""
    by_t = defaultdict(list)
    for r in csv.DictReader(open(path)):
        by_t[r["target"]].append(
            (int(r["layer"]), float(r["cv_acc"]), float(r["chance"]))
        )
    for t in by_t:
        by_t[t].sort()
    return by_t


def layer0_both_poolings(acts_dir, master_csv):
    """
    Direct layer-0 refit from the .npz, for BOTH pooling variants.

    Needed because the cached probe CSVs use last-token pooling, and at layer 0 that is
    the embedding of a SINGLE token (the story's final token) -- not a bag-of-words. Since
    the stories end differently ("...dies" vs "...is fine") that one token can leak outcome
    on its own. Mean pooling at layer 0 IS the bag-of-embeddings test, so it is the honest
    surface bound. Only 1 layer is refit, so this is cheap.
    """
    import numpy as np
    sys_path_hack()
    from importlib import import_module
    probe_mod = import_module("probe_02")
    group_cv_acc = probe_mod.group_cv_acc

    lab = {r["story_id"]: r for r in csv.DictReader(open(master_csv))}
    out = []
    for npz in sorted(glob.glob(os.path.join(acts_dir, "*.npz"))):
        tag = os.path.basename(npz)[:-4]
        d = np.load(npz, allow_pickle=True)
        sids = [str(s) for s in d["story_id"]]
        keep = [i for i, s in enumerate(sids) if s in lab]
        sk = [sids[i] for i in keep]
        intent = np.array([1 if lab[s]["intent_label"] == "guilty" else 0 for s in sk])
        outcome = np.array([1 if lab[s]["outcome_label"] == "harm" else 0 for s in sk])
        groups = np.array([lab[s]["scenario_id"] for s in sk])
        for pooling in ("last", "mean"):
            X = d[pooling][keep][:, 0, :]        # layer 0 == embedding output
            for target, y in (("intent", intent), ("outcome", outcome)):
                acc, sd = group_cv_acc(X, y, groups)
                out.append({"model": tag, "pooling": pooling, "target": target,
                            "acc_layer0": round(acc, 4), "sd": round(sd, 4),
                            "chance": round(max(y.mean(), 1 - y.mean()), 4)})
                print(f"  {tag:26} {pooling:5} {target:8} L0 acc={acc:.3f}")
    return out


def sys_path_hack():
    """Import 02_probe.py under a valid module name (leading digit blocks normal import)."""
    import sys, importlib.util
    if "probe_02" in sys.modules:
        return
    p = os.path.join(ROOT, "code", "02_probe.py")
    spec = importlib.util.spec_from_file_location("probe_02", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["probe_02"] = m
    spec.loader.exec_module(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default=PROBE_DIR)
    ap.add_argument("--acts", default=os.path.join(ROOT, "outputs", "acts"))
    ap.add_argument("--csv", default=os.path.join(ROOT, "dataset", "master", "moral_2x2_master.csv"))
    ap.add_argument("--skip-pooling-check", action="store_true")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.probe, "*_probe.csv")))
    # exclude derived files from other tasks
    files = [f for f in files if "withincell" not in f and "clause" not in f]
    if not files:
        raise SystemExit(f"no probe CSVs in {a.probe} -- run code/02_probe.py first")

    rows, curves = [], {}
    for p in files:
        tag = os.path.basename(p)[: -len("_probe.csv")]
        by_t = load_probe(p)
        curves[tag] = by_t
        for target, series in by_t.items():
            n_layers = len(series)
            acc0 = series[0][1]
            pl, peak, chance = max(series, key=lambda s: s[1])
            rows.append({
                "model": tag,
                "target": target,
                "acc_layer0": round(acc0, 4),
                "acc_peak": round(peak, 4),
                "peak_layer": pl,
                "n_layers": n_layers,
                "peak_layer_frac": round(pl / max(n_layers - 1, 1), 3),
                "chance": round(chance, 4),
                # how much of the peak is already available with zero computation
                "layer0_frac_of_peak": round(acc0 / peak, 3) if peak else float("nan"),
                # headroom above chance that layer 0 already captures
                "layer0_above_chance": round(acc0 - chance, 4),
            })

    os.makedirs(a.probe, exist_ok=True)
    out_csv = os.path.join(a.probe, "layer0_diagnostic.csv")
    cols = ["model", "target", "acc_layer0", "acc_peak", "peak_layer", "n_layers",
            "peak_layer_frac", "chance", "layer0_frac_of_peak", "layer0_above_chance"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"{'model':26} {'target':8} {'L0':>7} {'peak':>7} {'@L':>4} {'L0/peak':>8}")
    print("-" * 66)
    for tag in [t for t in ORDER if t in curves] + [t for t in curves if t not in ORDER]:
        for target in ("intent", "outcome"):
            r = next((x for x in rows if x["model"] == tag and x["target"] == target), None)
            if r:
                print(f"{tag:26} {target:8} {r['acc_layer0']:7.3f} {r['acc_peak']:7.3f} "
                      f"{r['peak_layer']:4d} {r['layer0_frac_of_peak']:8.2f}")
    print(f"\n-> {out_csv}")

    if not a.skip_pooling_check and os.path.isdir(a.acts):
        print("\n=== layer-0 refit, last vs mean pooling (mean = true bag-of-embeddings) ===")
        pooled = layer0_both_poolings(a.acts, a.csv)
        p2 = os.path.join(a.probe, "layer0_pooling_check.csv")
        with open(p2, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["model", "pooling", "target",
                                              "acc_layer0", "sd", "chance"])
            w.writeheader()
            w.writerows(pooled)
        print(f"-> {p2}")

    # ---- accuracy vs relative depth, one panel per target -------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        tags = [t for t in ORDER if t in curves] + [t for t in curves if t not in ORDER]
        cmap = plt.get_cmap("tab10")
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for ax, target in zip(axes, ("intent", "outcome")):
            for i, tag in enumerate(tags):
                series = curves[tag].get(target)
                if not series:
                    continue
                n = len(series)
                x = np.array([s[0] for s in series]) / max(n - 1, 1)
                y = np.array([s[1] for s in series])
                base = "Instruct" in tag
                ax.plot(x, y, color=cmap(i // 2 % 10),
                        ls="-" if base else "--", lw=1.9, marker="o", ms=2.6,
                        label=tag, alpha=0.9)
            ch = np.mean([s[2] for tg in tags if curves[tg].get(target)
                          for s in curves[tg][target]])
            ax.axhline(ch, color="k", ls=":", lw=1.2, label=f"chance ({ch:.2f})")
            ax.axvline(0.0, color="crimson", lw=1.0, alpha=0.35)
            ax.set_title(f"{target} decoding\n(x=0 is the embedding layer: no computation yet)",
                         fontsize=10)
            ax.set_xlabel("relative depth (layer / n_layers)")
            ax.grid(alpha=0.25)
        axes[0].set_ylabel("group-CV accuracy")
        axes[0].set_ylim(0.35, 1.03)
        axes[1].legend(fontsize=6.5, loc="lower right", ncol=1)
        fig.suptitle("Layer-wise decoding: is the signal present before the model computes anything?",
                     fontsize=12)
        fig.tight_layout()
        png = os.path.join(a.probe, "layerwise_curves.png")
        fig.savefig(png, dpi=150)
        print(f"-> {png}")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
