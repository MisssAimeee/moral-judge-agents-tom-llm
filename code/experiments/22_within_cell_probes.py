#!/usr/bin/env python3
"""
22_within_cell_probes.py -- Phase 1 / Task C3: the control already sitting in the 2x2.

Each factor is isolated with the other held constant. Paired cells share nearly all their
text, so this is the tightest lexical control available without writing new stimuli.

  intent_noharm     neutral + attempted        target=intent   <- HEADLINE
  intent_harm       accidental + intentional   target=intent
  outcome_innocent  neutral + accidental       target=outcome
  outcome_guilty    attempted + intentional    target=outcome

intent_noharm is the headline: NEITHER story contains a harm event, so if intent still
decodes there, "the probe is reading harm words" cannot explain it. It must be compared
against its OWN TF-IDF baseline (from 21_surface_baseline.py), not against 0.5 -- the
restricted sets still carry lexical cues from the belief clause.

Outputs
  outputs/probe/<model>_withincell.csv   layer, probe_name, cv_acc, cv_std, chance, n
"""
import os, csv, glob, argparse, importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def load_probe_module():
    p = os.path.join(ROOT, "code", "02_probe.py")
    spec = importlib.util.spec_from_file_location("probe_02", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# name -> (row filter, target column, positive class)
PROBES = {
    "intent_noharm":    (lambda r: r["outcome_label"] == "no_harm", "intent_label",  "guilty"),
    "intent_harm":      (lambda r: r["outcome_label"] == "harm",    "intent_label",  "guilty"),
    "outcome_innocent": (lambda r: r["intent_label"] == "innocent", "outcome_label", "harm"),
    "outcome_guilty":   (lambda r: r["intent_label"] == "guilty",   "outcome_label", "harm"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "dataset", "master", "moral_2x2_master.csv"))
    ap.add_argument("--acts", default=os.path.join(ROOT, "outputs", "acts"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "probe"))
    ap.add_argument("--pooling", default="last", choices=["last", "mean"])
    ap.add_argument("--only", default=None)
    a = ap.parse_args()

    group_cv_acc = load_probe_module().group_cv_acc
    lab = {r["story_id"]: r for r in csv.DictReader(open(a.csv))}
    os.makedirs(a.out, exist_ok=True)
    suffix = "" if a.pooling == "last" else f"_{a.pooling}"

    npzs = sorted(glob.glob(os.path.join(a.acts, "*.npz")))
    if a.only:
        npzs = [n for n in npzs if a.only in os.path.basename(n)]

    for npz in npzs:
        tag = os.path.basename(npz)[:-4]
        d = np.load(npz, allow_pickle=True)
        acts = d[a.pooling]
        sids = [str(s) for s in d["story_id"]]

        rows = []
        for pname, (filt, tcol, pos) in PROBES.items():
            keep = [i for i, s in enumerate(sids) if s in lab and filt(lab[s])]
            if len(keep) < 20:
                print(f"  {tag} {pname}: only {len(keep)} items, skipped")
                continue
            sk = [sids[i] for i in keep]
            y = np.array([1 if lab[s][tcol] == pos else 0 for s in sk])
            g = np.array([lab[s]["scenario_id"] for s in sk])
            if len(np.unique(y)) < 2:
                continue
            chance = float(max(y.mean(), 1 - y.mean()))
            A = acts[keep]
            for L in range(A.shape[1]):
                acc, sd = group_cv_acc(A[:, L, :], y, g)
                rows.append([L, pname, round(acc, 4), round(sd, 4),
                             round(chance, 4), len(keep)])
            best = max((r for r in rows if r[1] == pname), key=lambda r: r[2])
            print(f"  {tag:26} {pname:17} peak={best[2]:.3f} @L{best[0]:<3} "
                  f"(chance {chance:.3f}, n={len(keep)})", flush=True)

        p = os.path.join(a.out, f"{tag}_withincell{suffix}.csv")
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["layer", "probe_name", "cv_acc", "cv_std", "chance", "n"])
            w.writerows(rows)
        print(f"  -> {p}", flush=True)


if __name__ == "__main__":
    main()
