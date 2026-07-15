#!/usr/bin/env python3
"""
18_mini_dissection.py  --  Roadmap #4, extended to EVERY open family at its
available checkpoint resolution (ANALYSIS ONLY -- no inference, no training).

Only OLMo-2, Tulu-3 and Zephyr publish intermediate pipeline checkpoints
(base -> SFT -> DPO -> ...). Qwen, Gemma, Mistral and Llama release only `base`
and `instruct`, so for those families we can measure the tuning effect at a
2-point resolution (base -> instruct) only. This script combines both:

  * FULL pipelines  (OLMo-2 / Tulu-3 / Zephyr): read the per-stage contrasts that
    16_checkpoint_dissection.py already produced (single-template human_verbatim,
    logprob-EV) from outputs/experiments/checkpoint_dissection.csv.
  * 2-POINT families (Qwen ladder, Gemma-2-9B, Mistral-7B-v0.3, Llama-3.1-8B):
    compute the base->instruct delta-contrast from the ratings 03_behavioral.py
    already saved (outputs/behavior/item_means_*.csv), restricted to the SAME
    single template (human_verbatim) so the method is identical to the full
    pipelines, with a paired scenario bootstrap CI on the delta.

An entropy/variance QC filter (same 0.02 threshold as the checkpoint dissection)
flags degenerate families (near-constant ratings, e.g. Mistral) so a 0.000 delta
is never read as a real null.

Outputs (only with --run):
  outputs/experiments/mini_dissection.csv
  outputs/experiments/mini_dissection.png
Default (no --run): a dry-run that lists the inputs and reports which
item_means files are present / missing, downloading nothing.
"""
import os, sys, csv, glob, re, argparse
from collections import defaultdict

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CODE_DIR)
import tom_common as tc            # noqa: E402
import numpy as np                # noqa: E402

# load 11_interaction_regression.py by path (module name starts with a digit)
import importlib.util             # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "interaction_reg", os.path.join(CODE_DIR, "11_interaction_regression.py"))
ireg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ireg)

BEHAVIOR_DIR = os.path.join(tc.ROOT, "outputs", "behavior")
OUT_DIR = os.path.join(tc.ROOT, "outputs", "experiments")
CKPT_CSV = os.path.join(OUT_DIR, "checkpoint_dissection.csv")
TEMPLATE = "human_verbatim"       # single template -> identical method to the C pipeline
DEGEN_STD = 0.02

# ------- 2-point (base->instruct only) families: display -> (base tag, instruct tag) -------
# tags are the model_safe names 03_behavioral.py writes (item_means_<tag>.csv).
TWO_POINT = {
    "Qwen2.5-0.5B":    ("Qwen_Qwen2.5-0.5B",        "Qwen_Qwen2.5-0.5B-Instruct"),
    "Qwen2.5-1.5B":    ("Qwen_Qwen2.5-1.5B",        "Qwen_Qwen2.5-1.5B-Instruct"),
    "Qwen2.5-3B":      ("Qwen_Qwen2.5-3B",          "Qwen_Qwen2.5-3B-Instruct"),
    "Qwen2.5-7B":      ("Qwen_Qwen2.5-7B",          "Qwen_Qwen2.5-7B-Instruct"),
    "Qwen2.5-14B":     ("Qwen_Qwen2.5-14B",         "Qwen_Qwen2.5-14B-Instruct"),
    "Gemma-2-9B":      ("unsloth_gemma-2-9b",       "unsloth_gemma-2-9b-it"),
    "Mistral-7B-v0.3": ("mistralai_Mistral-7B-v0_3","mistralai_Mistral-7B-Instruct-v0_3"),
    "Llama-3.1-8B":    ("meta-llama_Llama-3.1-8B",  "unsloth_Meta-Llama-3.1-8B-Instruct"),
}

# full-pipeline families as named in checkpoint_dissection.csv (final stage last)
FULL_FAMILIES = ["OLMo-2-7B", "Tulu-3-8B", "Zephyr-7B"]

COLORS = {
    "Qwen2.5-0.5B": "#9ecfd6", "Qwen2.5-1.5B": "#5fb3bf", "Qwen2.5-3B": "#2f97a6",
    "Qwen2.5-7B": "#00909e", "Qwen2.5-14B": "#006673",
    "Gemma-2-9B": "#e8710a", "Mistral-7B-v0.3": "#ff7000", "Llama-3.1-8B": "#a259ff",
    "OLMo-2-7B": "#7d3c98", "Tulu-3-8B": "#1f77b4", "Zephyr-7B": "#2ca02c",
}


def _loose(t):
    """Collapse '.'/'_' to '-' so 2_5<->2.5 and 3_1<->3.1 tag variants match."""
    return re.sub(r"[._]", "-", t).lower()


def resolve_path(tag):
    """Find item_means_<tag>.csv, tolerating the 2_5<->2.5 / 3_1<->3.1 variants."""
    cand = os.path.join(BEHAVIOR_DIR, f"item_means_{tag}.csv")
    if os.path.exists(cand):
        return cand
    want = _loose(tag)
    for f in glob.glob(os.path.join(BEHAVIOR_DIR, "item_means_*.csv")):
        t = os.path.basename(f)[len("item_means_"):-4]
        if _loose(t) == want:
            return f
    return None


def load_hv_cells(path):
    """{scenario: {condition: norm}} using ONLY the human_verbatim template."""
    cells = defaultdict(dict)
    allv = []
    for r in csv.DictReader(open(path)):
        if r["template"] != TEMPLATE:
            continue
        v = float(r["mean_norm_blame"])
        cells[tc.scenario_of(r["story_id"])][r["condition"]] = v
        allv.append(v)
    return cells, allv


def contrast_of(cells):
    """Mean paired (attempted - accidental) over scenarios having both cells."""
    diffs = [c["attempted"] - c["accidental"] for c in cells.values()
             if "attempted" in c and "accidental" in c]
    return float(np.mean(diffs)) if diffs else float("nan")


def decomposition(cells):
    m = ireg.cell_means(cells)
    _, b_int, b_out, b_inter = ireg.coeffs_from_means(m)
    return b_int, b_out, b_inter


def paired_delta_ci(base_cells, inst_cells, B=2000, seed=0):
    """Bootstrap CI on (instruct_contrast - base_contrast) over shared scenarios."""
    common = sorted(set(base_cells) & set(inst_cells))
    bd = {s: base_cells[s]["attempted"] - base_cells[s]["accidental"]
          for s in common
          if "attempted" in base_cells[s] and "accidental" in base_cells[s]}
    idd = {s: inst_cells[s]["attempted"] - inst_cells[s]["accidental"]
           for s in common
           if "attempted" in inst_cells[s] and "accidental" in inst_cells[s]}
    keys = sorted(set(bd) & set(idd))

    def stat(ks):
        return float(np.mean([idd[k] for k in ks]) - np.mean([bd[k] for k in ks]))

    if len(keys) < 5:
        return float("nan"), float("nan"), float("nan"), len(keys)
    pt, lo, hi = tc.bootstrap(keys, stat, B=B, seed=seed)
    return pt, lo, hi, len(keys)


def load_full_pipelines():
    """family -> list of stage dicts (sorted) from checkpoint_dissection.csv."""
    out = defaultdict(list)
    if not os.path.exists(CKPT_CSV):
        return out
    for r in csv.DictReader(open(CKPT_CSV)):
        out[r["family"]].append(dict(
            stage=r["stage"], stage_idx=int(r["stage_idx"]),
            contrast=float(r["contrast"]), b_intent=float(r["b_intent"]),
            b_outcome=float(r["b_outcome"]),
            degenerate=(r["degenerate"] == "True")))
    for fam in out:
        out[fam].sort(key=lambda d: d["stage_idx"])
    return out


def compute_two_point(a):
    rows = []
    for fam, (base_tag, inst_tag) in TWO_POINT.items():
        bp, ip = resolve_path(base_tag), resolve_path(inst_tag)
        if not bp or not ip:
            print(f"[skip] {fam}: missing item_means "
                  f"({'base' if not bp else ''}{' & ' if not bp and not ip else ''}"
                  f"{'instruct' if not ip else ''}) -> not scored yet")
            continue
        bcells, bvals = load_hv_cells(bp)
        icells, ivals = load_hv_cells(ip)
        bstd = float(np.std(bvals)) if bvals else 0.0
        istd = float(np.std(ivals)) if ivals else 0.0
        degenerate = (bstd < DEGEN_STD) or (istd < DEGEN_STD)
        bc, ic = contrast_of(bcells), contrast_of(icells)
        b_bi, b_bo, _ = decomposition(bcells)
        i_bi, i_bo, _ = decomposition(icells)
        delta, lo, hi, nsc = paired_delta_ci(bcells, icells, B=a.boot)
        rows.append(dict(
            family=fam, resolution="base_instruct", n_scenarios=nsc,
            base_contrast=bc, final_contrast=ic, delta_contrast=delta,
            delta_lo=lo, delta_hi=hi,
            delta_b_intent=i_bi - b_bi, delta_b_outcome=i_bo - b_bo,
            degenerate=degenerate,
            sig=(not np.isnan(lo) and (lo > 0 or hi < 0))))
        flag = "  [DEGENERATE]" if degenerate else ""
        ci = (f"[{lo:+.3f},{hi:+.3f}]" if not np.isnan(lo) else "[n/a]")
        print(f"  {fam:16} base {bc:+.3f} -> instruct {ic:+.3f}  "
              f"delta {delta:+.3f} {ci}  db_out={i_bo-b_bo:+.3f}{flag}")
    return rows


def summarize_full(full):
    rows = []
    for fam in FULL_FAMILIES:
        stages = full.get(fam, [])
        if len(stages) < 2:
            continue
        base, final = stages[0], stages[-1]
        degenerate = any(s["degenerate"] for s in stages)
        rows.append(dict(
            family=fam, resolution="full_pipeline", n_scenarios="",
            base_contrast=base["contrast"], final_contrast=final["contrast"],
            delta_contrast=final["contrast"] - base["contrast"],
            delta_lo="", delta_hi="",
            delta_b_intent=final["b_intent"] - base["b_intent"],
            delta_b_outcome=final["b_outcome"] - base["b_outcome"],
            degenerate=degenerate, sig=""))
    return rows


def plot(two_point_rows, full, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as e:
        print(f"  (skip figure: {e})"); return
    fig, ax = plt.subplots(figsize=(9, 6))
    XMAX = 3  # base=0 ... RLVR/Instruct=3

    # full pipelines: solid lines at their true stage positions
    for fam in FULL_FAMILIES:
        stages = full.get(fam, [])
        if len(stages) < 2:
            continue
        col = COLORS.get(fam, "#444")
        xs = [s["stage_idx"] for s in stages]
        ys = [s["contrast"] for s in stages]
        ax.plot(xs, ys, "-o", color=col, lw=2.0, ms=6, label=fam, zorder=3)
        for s in stages:
            if s["degenerate"]:
                ax.scatter([s["stage_idx"]], [s["contrast"]], marker="x",
                           s=80, color="red", zorder=5)

    # 2-point families: dashed base(0) -> instruct(XMAX), middle unresolved
    for r in two_point_rows:
        col = COLORS.get(r["family"], "#888")
        ax.plot([0, XMAX], [r["base_contrast"], r["final_contrast"]], "--o",
                color=col, lw=1.6, ms=5, alpha=0.9, label=r["family"], zorder=2)
        if r["degenerate"]:
            for x, y in [(0, r["base_contrast"]), (XMAX, r["final_contrast"])]:
                ax.scatter([x], [y], marker="x", s=80, color="red", zorder=5)

    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xticks(range(XMAX + 1))
    ax.set_xticklabels(["base", "SFT", "DPO", "RLVR/Instruct"])
    ax.set_ylabel("contrast (attempted - accidental), human_verbatim EV")
    ax.set_xlabel("instruction-tuning pipeline stage\n"
                  "(solid = full published pipeline;  dashed = base->instruct only, "
                  "middle stages not released)")
    ax.set_title("Tuning effect across all open families (red x = degenerate/QC-flagged)")
    # two-part legend: families (color) + line-style meaning
    fam_handles = [Line2D([0], [0], color=COLORS.get(f, "#888"), marker="o", lw=2)
                   for f in FULL_FAMILIES if len(full.get(f, [])) >= 2]
    fam_labels = [f for f in FULL_FAMILIES if len(full.get(f, [])) >= 2]
    for r in two_point_rows:
        fam_handles.append(Line2D([0], [0], color=COLORS.get(r["family"], "#888"),
                                  marker="o", ls="--", lw=1.6))
        fam_labels.append(r["family"])
    style_handles = [Line2D([0], [0], color="k", lw=2, label="full pipeline"),
                     Line2D([0], [0], color="k", lw=1.6, ls="--", label="base->instruct only")]
    leg1 = ax.legend(fam_handles, fam_labels, title="family", fontsize=7.5,
                     title_fontsize=8, loc="lower left", ncol=2, framealpha=0.95)
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, fontsize=8, loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.relpath(path, tc.ROOT)}")


def dry_run():
    print("\n=== MINI-DISSECTION DRY-RUN (no inference, reads existing CSVs only) ===")
    print(f"template = {TEMPLATE}  (single-template EV, identical method to 16_checkpoint_dissection)\n")
    print("FULL pipelines (from checkpoint_dissection.csv):")
    full = load_full_pipelines()
    for fam in FULL_FAMILIES:
        n = len(full.get(fam, []))
        print(f"  {fam:12} {n} stage(s) "
              f"{'OK' if n >= 2 else 'MISSING -> run 16_checkpoint_dissection.py --run'}")
    print("\n2-POINT families (need base + instruct item_means, human_verbatim):")
    for fam, (bt, it) in TWO_POINT.items():
        bp, ip = resolve_path(bt), resolve_path(it)
        print(f"  {fam:16} base:{'OK ' if bp else 'MISSING'}  instruct:{'OK ' if ip else 'MISSING'}"
              f"   ({bt}  |  {it})")
    print("\nRun for real with:\n  python code/experiments/18_mini_dissection.py --run")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true",
                    help="compute deltas + write CSV/PNG (default: dry-run inventory)")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()

    if not a.run:
        dry_run()
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    full = load_full_pipelines()
    print("\n=== 2-point base->instruct deltas (human_verbatim EV, paired scenario bootstrap) ===")
    tp_rows = compute_two_point(a)
    full_rows = summarize_full(full)

    out_csv = os.path.join(OUT_DIR, "mini_dissection.csv")
    cols = ["family", "resolution", "n_scenarios", "base_contrast", "final_contrast",
            "delta_contrast", "delta_lo", "delta_hi", "delta_b_intent",
            "delta_b_outcome", "degenerate", "sig"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in full_rows + tp_rows:
            w.writerow({c: (round(r[c], 4) if isinstance(r[c], float) else r[c])
                        for c in cols})
    print(f"\nwrote {os.path.relpath(out_csv, tc.ROOT)}  "
          f"({len(full_rows)} full-pipeline + {len(tp_rows)} 2-point families)")
    plot(tp_rows, full, os.path.join(OUT_DIR, "mini_dissection.png"))


if __name__ == "__main__":
    main()
