#!/usr/bin/env python3
"""Build outputs/figures_final/FIGURE_MANIFEST.md — every PNG gets a row or it does not ship."""
import os
import subprocess
from datetime import date

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEST = os.path.join(ROOT, "outputs", "figures_final")
OUT = os.path.join(DEST, "FIGURE_MANIFEST.md")

# filename -> (what it shows, source CSV, producing script, dataset version note)
ENTRIES = [
    ("master_developmental_ladder_digitized_openonly.png",
     "Developmental ladder vs naughty/wrongness digitized child bands; open-weight only",
     "outputs/master_all_models_digitized_openonly.csv",
     "code/29_dual_human_ladders.py", "moral_2x2_master.csv"),
    ("master_developmental_ladder_punish_openonly.png",
     "Same ladder under punish digitized anchor; open-weight only",
     "outputs/master_all_models_punish_openonly.csv",
     "code/29_dual_human_ladders.py", "moral_2x2_master.csv"),
    ("master_developmental_ladder_text_reported_openonly.png",
     "Ladder under the superseded pooled-prose anchor; open-weight only (robustness)",
     "outputs/master_all_models_text_reported_openonly.csv",
     "code/29_dual_human_ladders.py", "moral_2x2_master.csv"),
    ("checkpoint_dissection.png",
     "Three-family checkpoint contrast trajectory with revised stage shares",
     "outputs/experiments/checkpoint_dissection.csv",
     "code/experiments/16_checkpoint_dissection.py", "rescored 7-template"),
    ("gap_over_surface_span_matched.png",
     "Probe−TF-IDF gaps on span-matched baselines; narrowed caption (C2 supporting)",
     "outputs/probe/gap_over_surface_span_matched.csv",
     "code/experiments/33_gap_dissociation_figure.py", "clause_offsets.csv"),
    ("layerwise_curves.png",
     "Layer-wise intent/outcome decoding curves",
     "outputs/probe/*_probe.csv",
     "code/02_probe.py / layerwise plotter", "activations last-token"),
    ("rsa_similarity_heatmap.png",
     "RSA model×model representational similarity heatmap",
     "outputs/rsa/",
     "code/ rsa scripts", "probe features"),
    ("rsa_convergence_scatter.png",
     "RSA convergence vs behavioral contrast (null)",
     "outputs/link/representation_vs_behavior.csv",
     "code/link scripts", "open roster"),
    ("item_level_dissociation.png",
     "Item-level intent margin vs contrast (null)",
     "outputs/link/item_level_dissociation.csv",
     "code/link scripts", "open roster"),
    ("interaction_forest.png",
     "Mixed-effects interaction forest with cell-ordering annotation (0/20 human order)",
     "outputs/stats/mixed_effects_2x2.csv",
     "code/experiments/39_mixed_effects_2x2.py", "behavior item means"),
    ("w3_steering_dose_OLMo.png",
     "W3 dose–response: contrast vs α; intent/outcome/random; OLMo-2-7B-I",
     "outputs/experiments/w3_steering_OLMo-2-1124-7B-Instruct.csv",
     "code/experiments/48_w3_causal_steering.py", "job 19099255"),
    ("w3_steering_dose_Qwen.png",
     "W3 dose–response: contrast vs α; Qwen2.5-7B-I",
     "outputs/experiments/w3_steering_Qwen2.5-7B-Instruct.csv",
     "code/experiments/48_w3_causal_steering.py", "job 19099255"),
    ("w3_layersweep_OLMo.png",
     "W3 M2 layer×direction grid (narrow claim: peak+deeper)",
     "outputs/experiments/w3_layersweep_OLMo-2-1124-7B-Instruct.csv",
     "code/experiments/48_w3_causal_steering.py", "job 19099255"),
    ("w3_layersweep_Qwen.png",
     "W3 M2 layer×direction grid; Qwen",
     "outputs/experiments/w3_layersweep_Qwen2.5-7B-Instruct.csv",
     "code/experiments/48_w3_causal_steering.py", "job 19099255"),
    ("w3_manipulation_OLMo.png",
     "W3 M1: probe-margin displacement vs Δcontrast; OLMo",
     "outputs/experiments/w3_manipulation_OLMo-2-1124-7B-Instruct.csv",
     "code/experiments/48_w3_causal_steering.py", "job 19099255"),
    ("w3_manipulation_Qwen.png",
     "W3 M1: probe-margin displacement vs Δcontrast; Qwen",
     "outputs/experiments/w3_manipulation_Qwen2.5-7B-Instruct.csv",
     "code/experiments/48_w3_causal_steering.py", "job 19099255"),
    ("w3_prose_rating.png",
     "Prose/rating dissociation: cell means among stories naming intent/belief",
     "outputs/experiments/w3_prose_items_*.csv",
     "code/experiments/51_w3_prose_rating.py", "job 19099255"),
    ("tom_vs_contrast.png",
     "BigToM false-belief vs moral contrast scatter (open; closed pending)",
     "outputs/tom_benchmarks/tom_vs_contrast.csv",
     "code/experiments/42_tom_vs_contrast.py", "init_belief=0"),
    ("w4_curriculum.png",
     "W4 prompt curriculum: contrast at each of 5 cumulative in-context levels",
     "outputs/experiments/w4_prompt_curriculum.csv",
     "code/experiments/55_w4_summary.py", "job 19130876"),
    ("w7_bruneau_selectivity.png",
     "W7 (appendix): harm decoding across 3 domains + cross-domain transfer",
     "outputs/experiments/w7_bruneau_probes.csv",
     "code/experiments/56_w7_bruneau.py", "job 19131423, Bruneau 2011 stimuli"),
    ("reasoning_dose_response.png",
     "Roadmap #7: thinking budget vs contrast (closed models)",
     "outputs/closed_reasoning/closed_reasoning_contrasts.csv",
     "code/experiments/52_closed_reasoning_dose.py", "CLOSED_MODEL_SELECTION.md"),
]

# Quarantined in _pending_rescore/ with a STALE_ prefix. These are not deleted because they
# are the only ladders that show the closed roster at all, and they get restored under their
# original names once outputs/closed_reasoning/ replaces the v1-era closed rows.
STALE_ENTRIES = [
    ("STALE_master_developmental_ladder_digitized_all.png",
     "Digitized ladder including closed models",
     "outputs/master_all_models_digitized_all.csv",
     "code/29_dual_human_ladders.py", "v1-era closed rows"),
    ("STALE_master_developmental_ladder_punish_all.png",
     "Punish ladder including closed models",
     "outputs/master_all_models_punish_all.csv",
     "code/29_dual_human_ladders.py", "v1-era closed rows"),
    ("STALE_master_developmental_ladder_text_reported_all.png",
     "Pooled-prose ladder including closed models",
     "outputs/master_all_models_text_reported_all.csv",
     "code/29_dual_human_ladders.py", "v1-era closed rows"),
]
STALE_PREFIX = ("**STALE — contaminated-era closed rows, do not use.** "
                "Closed models here (gpt-4o, gpt-4o-mini, claude-haiku-4-5, "
                "gemini-2.5-flash) were scored 2026-07-21 against master **v1** and have "
                "not been rescored. ")


def commit_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main():
    os.makedirs(DEST, exist_ok=True)
    h = commit_hash()
    lines = [
        f"# Figure manifest — `outputs/figures_final/` ({date.today().isoformat()})",
        "",
        f"Commit at write: `{h}`. No figure ships without a row here. "
        f"Prior locations (`outputs/updated_figures/`, loose root PNGs) are deprecated.",
        "",
        "| file | present | what it shows | source CSV | script | dataset / run |",
        "|---|---|---|---|---|---|",
    ]
    present = {f for f in os.listdir(DEST) if f.endswith(".png")}
    for name, what, csv, script, ds in ENTRIES:
        ok = "yes" if name in present else "**MISSING**"
        lines.append(f"| `{name}` | {ok} | {what} | `{csv}` | `{script}` | {ds} |")

    quar_dir = os.path.join(DEST, "_pending_rescore")
    quarantined = ({f for f in os.listdir(quar_dir) if f.endswith(".png")}
                   if os.path.isdir(quar_dir) else set())
    lines += ["", "## Quarantined — `_pending_rescore/` (NOT talk-safe)", "",
              "Everything in this table is excluded from `figures_final/` proper. "
              "Restore under the original name (drop the `STALE_` prefix) only after the "
              "closed rescore in `outputs/closed_reasoning/` replaces the v1-era rows and "
              "the ladders are regenerated.", "",
              "| file | present | status and what it shows | source CSV | script "
              "| dataset / run |", "|---|---|---|---|---|---|"]
    for name, what, csv, script, ds in STALE_ENTRIES:
        ok = "yes" if name in quarantined else "**MISSING**"
        lines.append(f"| `_pending_rescore/{name}` | {ok} | {STALE_PREFIX}{what} "
                     f"| `{csv}` | `{script}` | {ds} |")

    extras = sorted(present - {e[0] for e in ENTRIES})
    if extras:
        lines += ["", "## Unmanifested PNGs (must be claimed or removed)", ""]
        for e in extras:
            lines.append(f"- `{e}`")
    stale_extras = sorted(quarantined - {e[0] for e in STALE_ENTRIES})
    if stale_extras:
        lines += ["", "## Unmanifested quarantined PNGs", ""]
        for e in stale_extras:
            lines.append(f"- `_pending_rescore/{e}`")
    open(OUT, "w").write("\n".join(lines) + "\n")
    print(f"wrote {os.path.relpath(OUT, ROOT)} "
          f"({sum(1 for e in ENTRIES if e[0] in present)}/{len(ENTRIES)} present)")


if __name__ == "__main__":
    main()
