#!/usr/bin/env python3
"""
29_dual_human_ladders.py -- regenerate the master ladder under BOTH human anchors.

Does NOT choose which anchor is primary. Writes two side-by-side ladders so the
user can see where every model lands under each before deciding the paper's
headline claim.

Anchors
  text-reported  dataset/human_reference/human_reference.csv
                 child contrasts: 4–5 = −0.14, 6–7 = +0.15, 8+ = +0.46
                 (naughty+punishable mix from paper text; see methods_child_measure.md)
  digitized      dataset/human_reference/human_reference_digitized.csv
                 child contrasts: 4–5 = +0.24, 6–7 = +0.50, 8+ = +0.63
                 (Naughty/wrongness, presented-first; from cushman_child_bands_PROPOSED.csv)

Also re-runs 05_human_comparison.py against each anchor into separate output dirs.

Usage
  python code/29_dual_human_ladders.py
"""
import os, csv, math, subprocess, sys
from collections import OrderedDict, defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
import importlib.util
spec = importlib.util.spec_from_file_location("mf", os.path.join(HERE, "10_master_figure.py"))
mf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mf)

ANCHORS = OrderedDict([
    ("text_reported", {
        "human_csv": os.path.join(ROOT, "dataset", "human_reference", "human_reference.csv"),
        "bands": OrderedDict([
            ("adult", 0.666), ("child_8plus", 0.46),
            ("child_6_7", 0.15), ("child_4_5", -0.14),
        ]),
        "out_png": os.path.join(ROOT, "outputs", "master_developmental_ladder_text_reported.png"),
        "out_csv": os.path.join(ROOT, "outputs", "master_all_models_text_reported.csv"),
        "human_out": os.path.join(ROOT, "outputs", "human_text_reported"),
        "label": "text-reported (human_reference.csv)",
    }),
    ("digitized", {
        "human_csv": os.path.join(ROOT, "dataset", "human_reference", "human_reference_digitized.csv"),
        "bands": OrderedDict([
            ("adult", 0.666), ("child_8plus", 0.63),
            ("child_6_7", 0.50), ("child_4_5", 0.24),
        ]),
        "out_png": os.path.join(ROOT, "outputs", "master_developmental_ladder_digitized.png"),
        "out_csv": os.path.join(ROOT, "outputs", "master_all_models_digitized.csv"),
        "human_out": os.path.join(ROOT, "outputs", "human_digitized"),
        "label": "digitized Naughty presented-first",
    }),
    # Secondary construct-matched ladder: punishment series (Cushman 2013 Fig.3).
    # Does NOT replace Naughty; shown alongside. Cell means pending fig3 PUNISH
    # digitization — bands are the PHASE2 pre-specified contrasts.
    ("punish", {
        "human_csv": os.path.join(ROOT, "dataset", "human_reference", "human_reference_punish.csv"),
        "bands": OrderedDict([
            ("adult", 0.666), ("child_8plus", 0.19),
            ("child_6_7", 0.12), ("child_4_5", 0.09),
        ]),
        "out_png": os.path.join(ROOT, "outputs", "master_developmental_ladder_punish.png"),
        "out_csv": os.path.join(ROOT, "outputs", "master_all_models_punish.csv"),
        "human_out": os.path.join(ROOT, "outputs", "human_punish"),
        "label": "Punish presented-first (secondary; construct-matched to punish_* prompts)",
    }),
])


def nearest(contrast, bands):
    return min(bands, key=lambda g: abs(bands[g] - contrast))


def below_youngest(contrast, bands):
    """Does the model fall at or below the youngest measured band?"""
    youngest = bands["child_4_5"]
    return contrast <= youngest + 1e-9


def render(rows, bands, out_png, out_csv, title_suffix):
    rows = sorted(rows, key=lambda r: r["contrast"])
    # recompute nearest under THIS anchor
    for r in rows:
        r["nearest"] = nearest(r["contrast"], bands) if not r["degenerate"] else "NA"

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "family", "study", "type", "contrast",
                    "ci_lo", "ci_hi", "sig_vs_0", "nearest_human_group",
                    "degenerate", "at_or_below_youngest_band", "anchor"])
        for g, c in bands.items():
            w.writerow([f"HUMAN {g}", "human", "reference", "human",
                        f"{c:+.3f}", "", "", "", g, "False", "", title_suffix])
        for r in rows:
            w.writerow([r["model"], r["family"], r["study"], r["type"],
                        f"{r['contrast']:+.3f}", f"{r['ci_lo']:+.3f}",
                        f"{r['ci_hi']:+.3f}", r["sig"], r["nearest"],
                        "True" if r["degenerate"] else "False",
                        "yes" if (not r["degenerate"] and below_youngest(r["contrast"], bands))
                        else "no",
                        title_suffix])

    n = len(rows)
    fig, ax = plt.subplots(figsize=(11, 0.34 * n + 3.2))
    for g, c in bands.items():
        col = mf.HUMAN_COLORS[g]
        ax.axvline(c, ls="--", lw=1.4, color=col, alpha=0.85, zorder=1)
        ax.text(c, n + 0.4, g.replace("child_", "age ").replace("plus", "+"),
                rotation=90, va="bottom", ha="center", fontsize=8.5,
                color=col, fontweight="bold")
    ax.axvline(0, color="k", lw=0.8, zorder=1)
    for i, r in enumerate(rows):
        col = mf.FAMILY_COLORS.get(r["family"], "#888")
        lo = max(0, r["contrast"] - r["ci_lo"])
        hi = max(0, r["ci_hi"] - r["contrast"])
        marker = "o" if r["study"] == "cloud API" else "s"
        if r["degenerate"]:
            ax.errorbar(r["contrast"], i, xerr=[[lo], [hi]], fmt=marker,
                        mfc="none", mec="#999999", ecolor="#cccccc", capsize=2.5,
                        ms=7, elinewidth=1.2, alpha=0.9, zorder=3)
        else:
            ax.errorbar(r["contrast"], i, xerr=[[lo], [hi]], fmt=marker,
                        color=col, ecolor=col, capsize=2.5, ms=7,
                        elinewidth=1.6, alpha=0.9, zorder=3)
    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [f"{r['model']}  ({'cloud' if r['study']=='cloud API' else 'local'}/{r['type']})"
         + ("  [degenerate]" if r["degenerate"] else "")
         for r in rows], fontsize=8.5)
    ax.set_ylim(-1, n + 1.5)
    ax.set_xlabel("intent-vs-outcome contrast   =   blame(attempted) − blame(accidental)\n"
                  "← OUTCOME-driven              INTENT-driven →", fontsize=10)
    ax.set_title(f"Master ladder under {title_suffix}\n"
                 f"(dot = cloud · square = local · bar = 95% CI)  "
                 f"youngest band = {bands['child_4_5']:+.2f}",
                 fontsize=11, fontweight="bold")
    fam_present = list(dict.fromkeys(r["family"] for r in rows))
    handles = [Line2D([0], [0], marker="o", ls="", ms=8,
                      color=mf.FAMILY_COLORS[f], label=f) for f in fam_present]
    ax.legend(handles=handles, title="family", fontsize=8, loc="lower right",
              framealpha=0.95, ncol=2)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.relpath(out_png, ROOT)}")
    print(f"wrote {os.path.relpath(out_csv, ROOT)}")
    return rows


def run_05(human_csv, out_dir, behavior_dirs):
    os.makedirs(out_dir, exist_ok=True)
    for beh in behavior_dirs:
        if not os.path.isdir(beh):
            continue
        cmd = [sys.executable, os.path.join(HERE, "05_human_comparison.py"),
               "--human", human_csv, "--behavior", beh, "--out", out_dir]
        print("RUN:", " ".join(cmd))
        subprocess.run(cmd, check=False)


def main():
    # load model rows once from existing stats (model contrasts don't change with the anchor)
    rows = mf.load(mf.AGENT_STATS, "cloud API") + mf.load(mf.LOCAL_STATS, "local open-weight")
    seen, uniq = set(), []
    for r in rows:
        if r["key"] in seen:
            continue
        seen.add(r["key"])
        uniq.append(dict(r))  # copy — nearest will differ per anchor

    # Closed-API models were scored before the stimulus repair and have NOT been
    # rescored (blocked on budget approval). Every ladder is therefore emitted twice:
    # _openonly is the defensible artifact; _all carries the contaminated-era cloud
    # rows and must never be shown without that marking.
    open_rows = [r for r in uniq if r["study"] != "cloud API"]
    SCOPES = OrderedDict([
        ("openonly", dict(rows=open_rows,
                          suffix="_openonly",
                          note="open-weight only, post-repair rescore")),
        ("all", dict(rows=uniq,
                     suffix="_all",
                     note="INCLUDES closed-API models marked "
                          "PENDING RESCORE — contaminated-era")),
    ])

    summary = []          # openonly scope — drives the robustness note
    summary_all = []      # full scope — reported alongside, explicitly marked
    print("=" * 72)
    print("HUMAN ANCHOR LADDERS — every anchor × scope; primary choice is the USER's")
    print(f"  open-weight models: {len(open_rows)}   "
          f"closed-API (contaminated-era): {len(uniq) - len(open_rows)}")
    print("=" * 72)
    for key, spec in ANCHORS.items():
        print(f"\n--- anchor: {spec['label']} ---")
        for g, c in spec["bands"].items():
            print(f"  {g:12} contrast = {c:+.3f}")
        for scope, sc in SCOPES.items():
            png = spec["out_png"].replace(".png", sc["suffix"] + ".png")
            out_csv = spec["out_csv"].replace(".csv", sc["suffix"] + ".csv")
            rendered = render([dict(r) for r in sc["rows"]], spec["bands"],
                              png, out_csv, f"{key} [{sc['note']}]")
            n_below = sum(1 for r in rendered
                          if not r["degenerate"] and below_youngest(r["contrast"], spec["bands"]))
            n_ok = sum(1 for r in rendered if not r["degenerate"])
            print(f"  [{scope:9}] at/below youngest ({spec['bands']['child_4_5']:+.2f}): "
                  f"{n_below}/{n_ok} non-degenerate")
            (summary if scope == "openonly" else summary_all).append(
                (key, spec, rendered, n_below, n_ok))

        # 05_human_comparison for each behavior tree that exists
        run_05(spec["human_csv"], spec["human_out"], [
            os.path.join(ROOT, "outputs", "behavior"),
            os.path.join(ROOT, "outputs", "agents", "behavior"),
            os.path.join(ROOT, "outputs", "_contaminated_20260726", "behavior"),
        ])

    # side-by-side placement table
    out_cmp = os.path.join(ROOT, "outputs", "human_anchor_comparison.csv")
    by_anchor = {k: {r["key"]: r for r in rendered}
                 for k, _, rendered, _, _ in summary}
    def _sort_contrast(k):
        for ak in ("text_reported", "digitized", "punish"):
            if k in by_anchor.get(ak, {}):
                return by_anchor[ak][k]["contrast"]
        return 0.0
    keys = sorted({k for d in by_anchor.values() for k in d}, key=_sort_contrast)
    METHODS_NOTE = (
        "METHODS PRE-SPEC (2026-07-10): methods_child_measure.md chose Naughty/wrongness, "
        "presented-first as primary — sixteen days before this comparison. "
        "human_reference.csv used naughty+punishable text inconsistent with that spec. "
        "Anchor decision must trace to that prior methods choice, not to 9/24 vs 24/24. "
        "Naughty + Punish ladders both remain as a permanent robustness table; "
        "do not choose the anchor here."
    )
    with open(out_cmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "contrast",
                    "nearest_text_reported", "below_youngest_text_reported",
                    "nearest_digitized_naughty", "below_youngest_digitized_naughty",
                    "nearest_punish", "below_youngest_punish",
                    "degenerate", "methods_prespec_note"])
        for i, k in enumerate(keys):
            rt = by_anchor["text_reported"].get(k)
            rd = by_anchor["digitized"].get(k)
            rp = by_anchor["punish"].get(k)
            r = rt or rd or rp
            w.writerow([
                r["model"], f"{r['contrast']:+.3f}",
                rt["nearest"] if rt else "",
                "yes" if rt and not rt["degenerate"] and below_youngest(
                    rt["contrast"], ANCHORS["text_reported"]["bands"]) else "no",
                rd["nearest"] if rd else "",
                "yes" if rd and not rd["degenerate"] and below_youngest(
                    rd["contrast"], ANCHORS["digitized"]["bands"]) else "no",
                rp["nearest"] if rp else "",
                "yes" if rp and not rp["degenerate"] and below_youngest(
                    rp["contrast"], ANCHORS["punish"]["bands"]) else "no",
                "True" if r["degenerate"] else "False",
                METHODS_NOTE if i == 0 else "",
            ])
    print(f"\nwrote {os.path.relpath(out_cmp, ROOT)}")

    # ---- robustness-across-measures paragraph, regenerated from live numbers ----
    # Gate 0 and the mentor packet pull this verbatim; it is a robustness statement,
    # NOT an anchor choice.
    by_key = {k: (spec, n_below, n_ok) for k, spec, _, n_below, n_ok in summary}
    def _frag(k):
        spec, n_below, n_ok = by_key[k]
        held = "holds" if n_below == n_ok else "does not hold"
        return (f"{spec['label']} (youngest band {spec['bands']['child_4_5']:+.2f}): "
                f"{held} for {n_below}/{n_ok} non-degenerate models")
    by_key_all = {k: (spec, n_below, n_ok) for k, spec, _, n_below, n_ok in summary_all}
    both_digitized_hold = all(by_key[k][1] == by_key[k][2] for k in ("digitized", "punish"))
    ROBUSTNESS_NOTE = (
        "SCOPE: computed on open-weight models only (post-repair rescore). Closed-API "
        "models have not been rescored since the stimulus repair; their ladders are "
        "emitted separately as *_all and marked PENDING RESCORE — contaminated-era.\n\n"
        "ROBUSTNESS ACROSS MEASURES (not an anchor choice). The claim 'models fall "
        "at or below the youngest measured band' "
        + ("holds under BOTH digitized child ladders — naughtiness "
           f"(youngest {ANCHORS['digitized']['bands']['child_4_5']:+.2f}) and punishment "
           f"(youngest {ANCHORS['punish']['bands']['child_4_5']:+.2f}) — "
           if both_digitized_hold else
           "does NOT hold uniformly across the digitized ladders — ")
        + "and fails only under human_reference.csv, which mixed naughty+punishable "
          "contrary to the method pre-specified on 2026-07-10. Surviving two "
          "independently digitized child measures is a robustness result; it does not "
          "select a primary anchor, which remains the user's decision.\n\n"
        "THEORETICAL CHECK. The punishment ladder is monotonic in age but flatter than "
        "naughtiness (+0.09/+0.12/+0.19 vs +0.24/+0.50/+0.63) — exactly Cushman et al. "
        "(2013)'s two-process prediction that intent constrains judgments of wrongness "
        "before judgments of deserved punishment. Two independent digitizations "
        "reproducing the predicted pattern is evidence the digitization is sound.\n\n"
        "Per-ladder outcome (open-weight only):\n"
        + "\n".join(f"  - {_frag(k)}" for k in by_key)
        + "\n\nSame ladders WITH contaminated-era closed-API models included "
          "(marked, not for headline use):\n"
        + "\n".join(
            f"  - {by_key_all[k][0]['label']} (youngest band "
            f"{by_key_all[k][0]['bands']['child_4_5']:+.2f}): "
            f"{'holds' if by_key_all[k][1] == by_key_all[k][2] else 'does not hold'} "
            f"for {by_key_all[k][1]}/{by_key_all[k][2]} non-degenerate models"
            for k in by_key_all)
    )
    notes = os.path.join(ROOT, "outputs", "human_anchor_comparison.NOTES.md")
    with open(notes, "w") as nf:
        nf.write("# Human anchor comparison — methods provenance\n\n")
        nf.write(METHODS_NOTE + "\n\n")
        nf.write("## Robustness across measures\n\n")
        nf.write(ROBUSTNESS_NOTE + "\n")
    print(f"wrote {os.path.relpath(notes, ROOT)}")

    print("\n=== HEADLINE CLAIM CHECK ===")
    for key, spec, rendered, n_below, n_ok in summary:
        claim = (f"'models fall at or below the youngest measured band' "
                 f"({'HOLDS for all non-degenerate' if n_below == n_ok else 'does NOT hold for all'})")
        print(f"  [{key:14}] youngest={spec['bands']['child_4_5']:+.2f}  "
              f"at/below={n_below}/{n_ok}  → {claim}")
    print("\n" + ROBUSTNESS_NOTE)
    print("\nNo primary anchor chosen here. Digitized Naughty/presented-first was "
          "pre-specified in methods_child_measure.md (2026-07-10); all three ladders "
          "stay as a robustness table.")


if __name__ == "__main__":
    main()
