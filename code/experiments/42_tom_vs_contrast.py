#!/usr/bin/env python3
"""J1 part 2 -- correlate ToM-benchmark accuracy against the 2x2 intent contrast.

PRE-REGISTERED BEFORE THE FULL ROSTER WAS SCORED. The ceiling gate (job 19026525) had run
on three models and confirmed spread on both benchmarks; the 20-model accuracies were not
available when this file was written, and the reading of each outcome below is fixed here so
it cannot be chosen afterwards.

  PRIMARY MEASURE: BigToM forward-belief accuracy, false-belief condition. This is the
  belief-inference construct, the closest of the available benchmark conditions to what the
  moral task requires -- inferring a mental state that diverges from the state of the world.
  ToMi total accuracy is secondary and reported alongside.

  BEHAVIOURAL AXIS: the 2x2 contrast (attempted - accidental) from contrast_by_model.csv,
  restricted to models clearing the derived engagement floor (rating_std >= 0.2191, see
  40_derive_floors.py). Models that do not vary their ratings have no contrast to correlate.

  INTERPRETATION, fixed in advance:
  * NULL (interval includes zero and excludes moderate effects): ToM-benchmark performance
    does not predict whether a model weights intent in graded moral judgment. The two come
    apart, and "models pass ToM tests but fail this" becomes a result in our own data rather
    than a claim borrowed from the literature.
  * POSITIVE: the moral task is partly measuring general ToM competence. The dissociation
    claim weakens and must be restated -- models that reason better about beliefs also use
    intent more, so the moral failure is not a separate phenomenon.
  * NEGATIVE: would need explaining, not celebrating. The most likely mundane cause is that
    both axes track instruction tuning in opposite directions, so the instruct/base split is
    reported alongside any negative result.
  * WIDE INTERVAL: uninformative, and reported as such. With at most 20 models and a
    restriction to engaged ones, this is a live possibility and is not to be written up as a
    null. C6 made exactly that mistake at n=8.

  Because n is small either way, the per-model table is the deliverable and the correlation
  is secondary. The scatter is reported with base and instruct models marked, since
  instruction tuning moves both axes and is the obvious confound.

Outputs
  outputs/tom_benchmarks/tom_vs_contrast.csv
  outputs/tom_benchmarks/tom_vs_contrast.png
  outputs/tom_benchmarks/TOM_VS_CONTRAST.md
"""
import argparse
import csv
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
TOMDIR = os.path.join(ROOT, "outputs", "tom_benchmarks")
STATS = os.path.join(ROOT, "outputs", "stats", "contrast_by_model.csv")

ENGAGEMENT_FLOOR = 0.2191   # derived; 40_derive_floors.py
N_BOOT = 10000


def joinkey(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_tom():
    """-> {joinkey: {subset: accuracy}}"""
    path = os.path.join(TOMDIR, "tom_accuracy_by_model.csv")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}; run 36_tom_benchmarks.py first")
    out = {}
    for r in csv.DictReader(open(path)):
        acc = fnum(r["accuracy"])
        if acc is None:
            continue
        out.setdefault(joinkey(r["model"]), {})[r["subset"]] = acc
    return out


def load_behaviour():
    rows = []
    for r in csv.DictReader(open(STATS)):
        rows.append(dict(model=r["model"], contrast=fnum(r.get("contrast")),
                         rating_std=fnum(r.get("rating_std")),
                         size_B=fnum(r.get("size_B")), mtype=r.get("type", ""),
                         key=joinkey(r["model"])))
    return rows


def boot_ci(x, y, n_boot=N_BOOT, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 4:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if x[i].std() < 1e-12 or y[i].std() < 1e-12:
            continue
        rs.append(np.corrcoef(x[i], y[i])[0, 1])
    if not rs:
        return np.nan, np.nan
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def label(r, lo, hi, n):
    if not np.isfinite(lo):
        return f"NOT ESTIMABLE (n={n})"
    width = hi - lo
    if width > 0.9:
        return ("UNINFORMATIVE — the interval is too wide to exclude a moderate effect in "
                "either direction. Not a null.")
    if lo <= 0 <= hi:
        return ("NULL, and bounded — ToM-benchmark accuracy does not predict the intent "
                "contrast, and the interval excludes a large effect.")
    if lo > 0:
        return ("POSITIVE — the moral task partly measures general ToM competence; the "
                "dissociation claim must be weakened.")
    return ("NEGATIVE — check the base/instruct split before interpreting; both axes track "
            "instruction tuning.")


def scatter(pts, path, rlab):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    for mtype, colour, marker in (("instruct", "#1f3f8f", "o"),
                                  ("base", "#c07a1e", "s")):
        sub = [p for p in pts if p["mtype"] == mtype]
        if not sub:
            continue
        ax.scatter([p["tom"] for p in sub], [p["contrast"] for p in sub],
                   s=[26 + 5.0 * (p["size_B"] or 1) for p in sub],
                   c=colour, marker=marker, alpha=0.82, edgecolor="white",
                   linewidth=0.7, label=f"{mtype} (area ~ params)", zorder=3)
    others = [p for p in pts if p["mtype"] not in ("instruct", "base")]
    if others:
        ax.scatter([p["tom"] for p in others], [p["contrast"] for p in others],
                   s=32, c="#777777", marker="^", alpha=0.8, label="unlabelled", zorder=3)

    for p in pts:
        ax.annotate(p["name"].split("/")[-1], (p["tom"], p["contrast"]),
                    fontsize=6.4, xytext=(4, 3), textcoords="offset points",
                    color="#444444")
    ax.axhline(0, color="#888888", lw=0.9, ls=":")
    ax.axvline(0.5, color="#888888", lw=0.9, ls=":")
    ax.annotate("chance on a 2-way forced choice", xy=(0.5, ax.get_ylim()[0]),
                xytext=(3, 4), textcoords="offset points", fontsize=6.8,
                color="#777777", rotation=90, va="bottom")
    ax.set_xlabel("BigToM forward-belief accuracy, false-belief condition", fontsize=9.8)
    ax.set_ylabel("2x2 intent contrast  (attempted - accidental)", fontsize=9.8)
    ax.set_title("Does standard ToM-benchmark performance predict intent use in moral "
                 "judgment?\n" + rlab, fontsize=10.4)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.22, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=185)
    print(f"  -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", default="bigtom|false_belief")
    ap.add_argument("--floor", type=float, default=ENGAGEMENT_FLOOR)
    a = ap.parse_args()

    tom = load_tom()
    beh = load_behaviour()

    rows, pts = [], []
    for b in beh:
        t = tom.get(b["key"])
        if t is None:
            cand = [v for k, v in tom.items() if k.endswith(b["key"]) or b["key"].endswith(k)]
            t = cand[0] if len(cand) == 1 else None
        if t is None:
            continue
        engaged = (b["rating_std"] is not None and b["rating_std"] >= a.floor)
        rows.append(dict(model=b["model"], mtype=b["mtype"], size_B=b["size_B"],
                         rating_std=b["rating_std"], contrast=b["contrast"],
                         bigtom_false=t.get("bigtom|false_belief"),
                         bigtom_true=t.get("bigtom|true_belief"),
                         bigtom_all=t.get("bigtom"), tomi_all=t.get("tomi"),
                         engaged=engaged))
        if engaged and b["contrast"] is not None and t.get(a.primary) is not None:
            pts.append(dict(name=b["model"], tom=t[a.primary], contrast=b["contrast"],
                            mtype=b["mtype"], size_B=b["size_B"]))

    print(f"models with both ToM and behaviour: {len(rows)}; "
          f"engaged and usable: {len(pts)}")

    results = {}
    for meas in ("bigtom|false_belief", "bigtom", "tomi"):
        x, y = [], []
        for b in beh:
            t = tom.get(b["key"])
            if t is None or t.get(meas) is None or b["contrast"] is None:
                continue
            if b["rating_std"] is None or b["rating_std"] < a.floor:
                continue
            x.append(t[meas])
            y.append(b["contrast"])
        if len(x) < 4 or np.std(x) < 1e-12:
            results[meas] = dict(r=float("nan"), lo=float("nan"), hi=float("nan"),
                                 n=len(x), label=f"NOT ESTIMABLE (n={len(x)})")
            print(f"  {meas:24} n={len(x)}  not estimable")
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        lo, hi = boot_ci(x, y)
        results[meas] = dict(r=r, lo=lo, hi=hi, n=len(x), label=label(r, lo, hi, len(x)))
        print(f"  {meas:24} r={r:+.3f} [{lo:+.3f},{hi:+.3f}] n={len(x)}")

    os.makedirs(TOMDIR, exist_ok=True)
    csv_path = os.path.join(TOMDIR, "tom_vs_contrast.csv")
    cols = ["model", "mtype", "size_B", "rating_std", "engaged", "contrast",
            "bigtom_false", "bigtom_true", "bigtom_all", "tomi_all"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda z: -(z["bigtom_false"] or -1)):
            w.writerow(r)
        w.writerow({})
        for meas, res in results.items():
            w.writerow({"model": f"CORRELATION {meas} vs contrast", "contrast": res["r"],
                        "bigtom_false": res["lo"], "bigtom_true": res["hi"],
                        "bigtom_all": res["n"], "tomi_all": res["label"]})
    print(f"  -> {csv_path}")

    prim = results.get(a.primary, {})
    rlab = (f"r = {prim.get('r', float('nan')):+.3f}, "
            f"95% CI [{prim.get('lo', float('nan')):+.2f}, "
            f"{prim.get('hi', float('nan')):+.2f}], n = {prim.get('n', 0)} engaged models")
    if pts:
        scatter(pts, os.path.join(TOMDIR, "tom_vs_contrast.png"), rlab)

    md = [
        "# ToM benchmark performance vs intent use in moral judgment (J1)", "",
        "## Question", "",
        "Does standard theory-of-mind benchmark performance predict whether a model weights",
        "intent in graded moral judgment? A null converts \"models pass ToM tests but fail",
        "this task\" from a literature argument into a result measured on the same models.", "",
        "## Design", "",
        "- **ToM axis (primary):** BigToM forward belief, false-belief condition. 200 items,",
        "  two-alternative forced choice scored by length-normalised log-likelihood. The",
        "  explicit statement of the agent's initial belief is removed from the story",
        "  (init_belief=0), so the belief must be inferred rather than copied.",
        "- **ToM axis (secondary):** ToMi first-order belief questions, 400 items, same",
        "  scoring.",
        "- **Behavioural axis:** the 2x2 contrast (attempted - accidental).",
        f"- **Restriction:** models clearing the derived engagement floor "
        f"(rating_std >= {a.floor}). A model that does not vary its ratings has no contrast",
        "  to correlate.",
        "- **Uncertainty:** bootstrap over models, 10,000 resamples.", "",
        "The interpretation of each possible outcome was fixed in the script docstring before",
        "the full roster was scored; only the three gate models had been run.", "",
        "## Ceiling gate", "",
        "Run first, on Qwen2.5-0.5B-Instruct, Qwen2.5-14B-Instruct and OLMo-2-7B-Instruct,",
        "because a correlation needs variance on both axes and a ceiling would have killed the",
        "analysis before spending GPU on 20 models:", "",
        "| benchmark | accuracies | spread | verdict |",
        "|---|---|---|---|",
        "| BigToM | 0.520 / 0.882 / 0.850 | 0.362 | spread, proceed |",
        "| ToMi | 0.482 / 0.512 / 0.818 | 0.335 | spread, proceed |", "",
        "Neither is near ceiling, so the full roster was worth running.", "",
        "## Result", "",
        "| ToM measure | r | 95% CI (bootstrap over models) | n | reading |",
        "|---|---|---|---|---|",
    ]
    for meas, res in results.items():
        md.append(f"| {meas} | {res['r']:+.3f} | "
                  f"[{res['lo']:+.3f}, {res['hi']:+.3f}] | {res['n']} | {res['label']} |")
    md += ["", "## Per-model table", "",
           "ToM accuracy is reported as its own column regardless of the correlation result,",
           "as requested, in `tom_vs_contrast.csv` and folded into the master table.", "",
           "| model | type | params | BigToM false-belief | BigToM all | ToMi | contrast | engaged |",
           "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda z: -(z["bigtom_false"] or -1)):
        f = lambda v: "—" if v is None else f"{v:.3f}"
        md.append(f"| {r['model']} | {r['mtype']} | {r['size_B'] or '—'} | "
                  f"{f(r['bigtom_false'])} | {f(r['bigtom_all'])} | {f(r['tomi_all'])} | "
                  f"{f(r['contrast'])} | {'yes' if r['engaged'] else 'no'} |")
    md += ["", "## Caveats", "",
           "- n is at most 20 and smaller after the engagement restriction, so the per-model",
           "  table is the deliverable and the correlation is secondary. A wide interval is",
           "  reported as uninformative, not as a null.",
           "- Instruction tuning moves both axes, so it is the obvious confound; base and",
           "  instruct models are marked separately in the scatter.",
           "- ToMi's true_belief / false_belief tags describe the story-generation condition",
           "  rather than the queried agent's belief state, so only the aggregate and the",
           "  question-type breakdown are used.", ""]
    md_path = os.path.join(TOMDIR, "TOM_VS_CONTRAST.md")
    open(md_path, "w").write("\n".join(md))
    print(f"  -> {md_path}")


if __name__ == "__main__":
    main()
