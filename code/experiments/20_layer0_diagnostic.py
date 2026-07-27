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

RESTRICTION — the "at chance at L0, rising later → computed, not lexical" inference is
valid ONLY for last/mean pooling. Clause-position poolings (belief_last / action_last)
often land on the same token across stories (a sentence-final period), so layer 0 has
structurally near-zero variance; the probe CSV marks those cells degenerate=True and
the accuracy is a majority-class fallback, not a fitted chance-level representation.
This script therefore reads only *_probe.csv and *_probe_mean.csv, and refuses to draw
the lexical-vs-computed conclusion from clause-pooled files.

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
    """-> {target: [(layer, acc, chance, degenerate), ...]} sorted by layer."""
    by_t = defaultdict(list)
    for r in csv.DictReader(open(path)):
        deg = str(r.get("degenerate", "")).lower() in ("true", "1")
        by_t[r["target"]].append(
            (int(r["layer"]), float(r["cv_acc"]), float(r["chance"]), deg)
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
        groups = np.array([lab[s].get("scenario_group") or lab[s]["scenario_id"] for s in sk])
        for pooling in ("last", "mean"):
            X = d[pooling][keep][:, 0, :]        # layer 0 == embedding output
            for target, y in (("intent", intent), ("outcome", outcome)):
                acc, sd, deg = group_cv_acc(X, y, groups)
                out.append({"model": tag, "pooling": pooling, "target": target,
                            "acc_layer0": round(acc, 4), "sd": round(sd, 4),
                            "chance": round(max(y.mean(), 1 - y.mean()), 4),
                            "degenerate": bool(deg)})
                print(f"  {tag:26} {pooling:5} {target:8} L0 acc={acc:.3f}"
                      f"{'  DEGENERATE' if deg else ''}")
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

    # Lexical-vs-computed inference is restricted to last/mean pooling. Clause-position
    # files (*_probe_belief_last.csv / *_probe_action_last.csv) are deliberately excluded:
    # their layer 0 is often structurally degenerate (see module docstring).
    candidates = sorted(glob.glob(os.path.join(a.probe, "*_probe.csv"))
                        + glob.glob(os.path.join(a.probe, "*_probe_mean.csv")))
    files = []
    skipped_clause = []
    for f in candidates:
        base = os.path.basename(f)
        if "withincell" in base or "clause" in base:
            continue
        if base.endswith("_probe_belief_last.csv") or base.endswith("_probe_action_last.csv"):
            skipped_clause.append(base)
            continue
        files.append(f)
    if skipped_clause:
        print("RESTRICTED: skipping clause-pooled probe files for L0 inference "
              f"({len(skipped_clause)} files). Reason: layer 0 often has zero variance "
              "under belief_last/action_last; degenerate=True cells are majority-class "
              "fallbacks, not fitted chance-level representations.")
    if not files:
        raise SystemExit(f"no last/mean probe CSVs in {a.probe} -- run code/02_probe.py first")

    rows, curves = [], {}
    for p in files:
        base = os.path.basename(p)
        if base.endswith("_probe_mean.csv"):
            tag = base[: -len("_probe_mean.csv")]
            pooling = "mean"
        else:
            tag = base[: -len("_probe.csv")]
            pooling = "last"
        by_t = load_probe(p)
        curves[(tag, pooling)] = by_t
        for target, series in by_t.items():
            n_layers = len(series)
            acc0, chance0, deg0 = series[0][1], series[0][2], series[0][3]
            pl, peak, chance, _ = max(series, key=lambda s: s[1])
            # "at chance at L0 → computed later" is only valid when L0 actually fitted
            inference_valid = (pooling in ("last", "mean")) and (not deg0)
            rows.append({
                "model": tag,
                "pooling": pooling,
                "target": target,
                "acc_layer0": round(acc0, 4),
                "acc_peak": round(peak, 4),
                "peak_layer": pl,
                "n_layers": n_layers,
                "peak_layer_frac": round(pl / max(n_layers - 1, 1), 3),
                "chance": round(chance, 4),
                "layer0_frac_of_peak": round(acc0 / peak, 3) if peak else float("nan"),
                "layer0_above_chance": round(acc0 - chance0, 4),
                "layer0_degenerate": deg0,
                "inference_valid": inference_valid,
                "inference_note": (
                    "ok: L0 fitted; chance-at-L0 + rise = computed signal"
                    if inference_valid else
                    "INVALID for lexical-vs-computed: L0 degenerate (majority-class fallback)"
                    if deg0 else
                    "INVALID: pooling not last/mean"
                ),
            })

    os.makedirs(a.probe, exist_ok=True)
    out_csv = os.path.join(a.probe, "layer0_diagnostic.csv")
    cols = ["model", "pooling", "target", "acc_layer0", "acc_peak", "peak_layer",
            "n_layers", "peak_layer_frac", "chance", "layer0_frac_of_peak",
            "layer0_above_chance", "layer0_degenerate", "inference_valid",
            "inference_note"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"{'model':26} {'pool':5} {'target':8} {'L0':>7} {'peak':>7} {'@L':>4} "
          f"{'valid':>5} {'note'}")
    print("-" * 100)
    keys = sorted(curves, key=lambda k: (ORDER.index(k[0]) if k[0] in ORDER else 99, k[1]))
    for tag, pooling in keys:
        for target in ("intent", "outcome"):
            r = next((x for x in rows if x["model"] == tag and x["pooling"] == pooling
                      and x["target"] == target), None)
            if r:
                print(f"{tag:26} {pooling:5} {target:8} {r['acc_layer0']:7.3f} "
                      f"{r['acc_peak']:7.3f} {r['peak_layer']:4d} "
                      f"{'yes' if r['inference_valid'] else 'NO':>5}  "
                      f"{r['inference_note'][:56]}")
    print(f"\n-> {out_csv}")
    print("NOTE: lexical-vs-computed inference restricted to last/mean pooling with "
          "layer0_degenerate=False.")

    if not a.skip_pooling_check and os.path.isdir(a.acts):
        print("\n=== layer-0 refit, last vs mean pooling (mean = true bag-of-embeddings) ===")
        pooled = layer0_both_poolings(a.acts, a.csv)
        p2 = os.path.join(a.probe, "layer0_pooling_check.csv")
        with open(p2, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["model", "pooling", "target",
                                              "acc_layer0", "sd", "chance",
                                              "degenerate"])
            w.writeheader()
            w.writerows(pooled)
        print(f"-> {p2}")

    # ---- accuracy vs relative depth: last-token pooling only (inference-valid) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        # plot last-pooling curves only — the pooling for which the L0 inference is defined
        tags = [t for t in ORDER if (t, "last") in curves]
        tags += [t for (t, p) in curves if p == "last" and t not in tags]
        cmap = plt.get_cmap("tab10")
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for ax, target in zip(axes, ("intent", "outcome")):
            for i, tag in enumerate(tags):
                series = curves[(tag, "last")].get(target)
                if not series:
                    continue
                n = len(series)
                x = np.array([s[0] for s in series]) / max(n - 1, 1)
                y = np.array([s[1] for s in series])
                base = "Instruct" in tag
                ax.plot(x, y, color=cmap(i // 2 % 10),
                        ls="-" if base else "--", lw=1.9, marker="o", ms=2.6,
                        label=tag, alpha=0.9)
            ch = np.mean([s[2] for tg in tags if curves[(tg, "last")].get(target)
                          for s in curves[(tg, "last")][target]])
            ax.axhline(ch, color="k", ls=":", lw=1.2, label=f"chance ({ch:.2f})")
            ax.axvline(0.0, color="crimson", lw=1.0, alpha=0.35)
            ax.set_title(f"{target} decoding (last pooling)\n"
                         f"(x=0 = embedding layer; clause poolings excluded)",
                         fontsize=10)
            ax.set_xlabel("relative depth (layer / n_layers)")
            ax.grid(alpha=0.25)
        axes[0].set_ylabel("group-CV accuracy")
        axes[0].set_ylim(0.35, 1.03)
        axes[1].legend(fontsize=6.5, loc="lower right", ncol=1)
        fig.suptitle("Layer-wise decoding (last pooling only; L0 inference restricted to "
                     "non-degenerate last/mean)",
                     fontsize=11)
        fig.tight_layout()
        png = os.path.join(a.probe, "layerwise_curves.png")
        fig.savefig(png, dpi=150)
        print(f"-> {png}")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
