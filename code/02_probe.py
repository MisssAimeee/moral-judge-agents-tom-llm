#!/usr/bin/env python3
"""
02_probe.py  --  Level 2: layer-wise representational analysis. NO model training.
A linear probe (logistic regression) is a lightweight read-out, not fine-tuning.

For each model and each layer, fit:
  - intent probe  : guilty vs innocent   (intent_label)
  - outcome probe : harm vs no_harm      (outcome_label)
using GROUP-AWARE cross-validation: all 4 cells of a scenario stay together in the
same fold. This is essential -- the 4 cells share background text, so a random split
would leak and inflate accuracy.

Key questions:
  * Does intent become linearly decodable, and in which layers?
  * Does intent peak in LATER layers than outcome (a depth signature)?
  * How separable are intent and outcome (orthogonality of probe directions)?

Outputs: outputs/probe/<model>_probe.csv  (layer, target, cv_acc, chance)
"""
import os, csv, glob, argparse, numpy as np
from collections import defaultdict

def load_labels(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    lab = {r["story_id"]: r for r in rows}
    return lab

def group_cv_acc(X, y, groups, n_splits=5, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    gkf = GroupKFold(n_splits=n_splits)
    accs = []
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(sc.transform(X[tr]), y[tr])
        accs.append(clf.score(sc.transform(X[te]), y[te]))
    return float(np.mean(accs)), float(np.std(accs))

def run(model_npz, lab, pooling="last"):
    d = np.load(model_npz, allow_pickle=True)
    acts = d[pooling]                      # [n, L, H]
    sids = [str(s) for s in d["story_id"]]
    keep = [i for i,s in enumerate(sids) if s in lab]
    acts = acts[keep]; sids = [sids[i] for i in keep]
    intent  = np.array([1 if lab[s]["intent_label"]=="guilty" else 0 for s in sids])
    outcome = np.array([1 if lab[s]["outcome_label"]=="harm"   else 0 for s in sids])
    groups  = np.array([lab[s]["scenario_id"] for s in sids])
    n_layers = acts.shape[1]
    out = []
    for L in range(n_layers):
        X = acts[:, L, :]
        ai, _ = group_cv_acc(X, intent, groups)
        ao, _ = group_cv_acc(X, outcome, groups)
        out.append((L, "intent",  ai, intent.mean().clip(0.5,1) if False else max(intent.mean(),1-intent.mean())))
        out.append((L, "outcome", ao, max(outcome.mean(),1-outcome.mean())))
    return out

if __name__ == "__main__":
    here = os.path.dirname(__file__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(here,"..","dataset","master","moral_2x2_master.csv"))
    ap.add_argument("--acts", default=os.path.join(here,"..","outputs","acts"))
    ap.add_argument("--out", default=os.path.join(here,"..","outputs","probe"))
    ap.add_argument("--pooling", default="last", choices=["last","mean"])
    a = ap.parse_args()
    lab = load_labels(a.csv)
    os.makedirs(a.out, exist_ok=True)
    for npz in sorted(glob.glob(os.path.join(a.acts, "*.npz"))):
        tag = os.path.basename(npz)[:-4]
        res = run(npz, lab, a.pooling)
        p = os.path.join(a.out, f"{tag}_probe.csv")
        with open(p, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["layer","target","cv_acc","chance"])
            w.writerows(res)
        peak_i = max((r for r in res if r[1]=="intent"), key=lambda r:r[2])
        print(f"{tag}: peak intent acc={peak_i[2]:.3f} @ layer {peak_i[0]}  -> {p}")
