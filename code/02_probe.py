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

def _rowspace_project(Xtr, Xte):
    """
    Losslessly shrink d (up to 4096) down to the training rank (<=238 here).

    For an L2-penalised linear model the optimum satisfies w = Xtr^T a (representer
    theorem), so w lies in the row space of the training matrix. Rotating into an
    orthonormal basis V of that row space leaves the L2 penalty unchanged (V is
    orthonormal) and drops only test-vector components orthogonal to w, which
    contribute exactly zero to the decision function. Predictions are therefore
    IDENTICAL, not approximated -- it is purely a speedup (~15x at 4096 dims), which is
    what makes a 1000-rep permutation null affordable.
    """
    if Xtr.shape[1] <= Xtr.shape[0]:
        return Xtr, Xte
    _, S, Vt = np.linalg.svd(Xtr, full_matrices=False)
    V = Vt[S > 1e-10].T
    return Xtr @ V, Xte @ V


def group_cv_acc(X, y, groups, n_splits=5, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    gkf = GroupKFold(n_splits=n_splits)
    accs = []
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = _rowspace_project(sc.transform(X[tr]), sc.transform(X[te]))
        # Layer 0 is the raw token embedding, and the clause-pooled position is nearly always the
        # same token (a sentence-final period), so a fold can contain a single unique row. With no
        # variance there is nothing to project onto and no classifier to fit. That layer genuinely
        # carries no information, so score it at the majority-class rate rather than crashing --
        # which is the honest answer and is what the layer-0 diagnostic is looking for.
        if Xtr.shape[1] == 0 or len(np.unique(y[tr])) < 2:
            accs.append(max(y[te].mean(), 1 - y[te].mean()) if len(te) else 0.5)
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, y[tr])
        accs.append(clf.score(Xte, y[te]))
    return float(np.mean(accs)), float(np.std(accs))

def permute_within_groups(y, groups, rng):
    """
    Shuffle labels WITHIN each scenario, never across scenarios.

    The 4 cells of a scenario share nearly all their text, so a global shuffle would break
    that dependency structure and give a null with the wrong variance. Permuting inside the
    group preserves it, which is what makes the resulting p-value honest.
    """
    yp = np.array(y, copy=True)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        yp[idx] = yp[rng.permutation(idx)]
    return yp


def permutation_null(X, y, groups, n_perm=1000, seed=0, n_jobs=-1):
    """-> (obs_acc, null_mean, null_p95, empirical_p). Costly: only call on chosen layers."""
    obs, _ = group_cv_acc(X, y, groups)
    rng = np.random.default_rng(seed)
    perms = [permute_within_groups(y, groups, rng) for _ in range(n_perm)]

    def one(yp):
        if len(np.unique(yp)) < 2:
            return 0.5
        return group_cv_acc(X, yp, groups)[0]

    try:
        from joblib import Parallel, delayed
        null = np.array(Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(one)(yp) for yp in perms))
    except Exception:
        null = np.array([one(yp) for yp in perms])
    # +1 correction: an empirical p can never legitimately be 0
    p = (np.sum(null >= obs) + 1) / (n_perm + 1)
    return float(obs), float(null.mean()), float(np.percentile(null, 95)), float(p)


def load_clause_mask(offsets_csv):
    """story_ids whose clause spans were found by the belief-verb pattern rather than guessed.

    When the pattern misses, 25_annotate_clauses.py estimates the spans from sentence position.
    Those estimates are not trustworthy enough to pool on: a wrong belief offset can land on the
    outcome sentence, which would smuggle end-of-story information into the belief probe and make
    it look like belief is decodable when we are really reading the outcome. Excluded, not
    defaulted -- a defaulted row is indistinguishable from a real one in the results.
    """
    if not offsets_csv or not os.path.exists(offsets_csv):
        return None
    import csv as _csv
    return {r["story_id"] for r in _csv.DictReader(open(offsets_csv))
            if r.get("method") == "belief_verb"}


def run(model_npz, lab, pooling="last", clause_ok=None):
    d = np.load(model_npz, allow_pickle=True)
    acts = d[pooling]                      # [n, L, H]
    sids = [str(s) for s in d["story_id"]]
    keep = [i for i,s in enumerate(sids) if s in lab]
    if clause_ok is not None and pooling in ("belief_last", "action_last"):
        keep = [i for i in keep if sids[i] in clause_ok]
    acts = acts[keep]; sids = [sids[i] for i in keep]
    intent  = np.array([1 if lab[s]["intent_label"]=="guilty" else 0 for s in sids])
    outcome = np.array([1 if lab[s]["outcome_label"]=="harm"   else 0 for s in sids])
    # group on scenario_group, not scenario_id: the YS2009 items are reprints of the YS2008 ones,
    # so keying on scenario_id would split a duplicated vignette across train and test
    groups  = np.array([lab[s].get("scenario_group") or lab[s]["scenario_id"] for s in sids])
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
    ap.add_argument("--pooling", default="last",
                    choices=["last","mean","belief_last","action_last"])
    ap.add_argument("--permute", type=int, default=0,
                    help="permutation-null reps; run at layer 0 and the peak layer only "
                         "(N=1000 for final numbers, 100 while developing)")
    ap.add_argument("--only", default=None,
                    help="substring filter on the model tag, so permutation runs can be "
                         "split across parallel jobs (one model each)")
    ap.add_argument("--skip-probe", action="store_true",
                    help="reuse the existing probe CSV instead of refitting every layer "
                         "(for permutation-only reruns)")
    ap.add_argument("--clause-offsets",
                    default=os.path.join(here,"..","dataset","master","clause_offsets.csv"),
                    help="used only by the belief_last/action_last poolings, to drop rows whose "
                         "clause spans were position-guessed rather than pattern-matched")
    a = ap.parse_args()
    lab = load_labels(a.csv)
    clause_ok = load_clause_mask(a.clause_offsets)
    if clause_ok is not None and a.pooling in ("belief_last", "action_last"):
        print(f"clause mask: {len(clause_ok)} reliable rows for {a.pooling} pooling", flush=True)
    os.makedirs(a.out, exist_ok=True)
    # keep the historical filename for last-token pooling so existing consumers still resolve
    suffix = "" if a.pooling == "last" else f"_{a.pooling}"
    npzs = sorted(glob.glob(os.path.join(a.acts, "*.npz")))
    if a.only:
        npzs = [n for n in npzs if a.only in os.path.basename(n)]
        if not npzs:
            raise SystemExit(f"--only {a.only!r} matched no .npz in {a.acts}")
    for npz in npzs:
        tag = os.path.basename(npz)[:-4]
        p = os.path.join(a.out, f"{tag}_probe{suffix}.csv")
        if a.skip_probe and os.path.exists(p):
            res = [(int(r["layer"]), r["target"], float(r["cv_acc"]), float(r["chance"]))
                   for r in csv.DictReader(open(p))]
            print(f"{tag}: reusing {p}", flush=True)
        else:
            res = run(npz, lab, a.pooling, clause_ok)
            with open(p, "w", newline="") as f:
                w = csv.writer(f); w.writerow(["layer","target","cv_acc","chance"])
                w.writerows(res)
        peak_i = max((r for r in res if r[1]=="intent"), key=lambda r:r[2])
        print(f"{tag}: peak intent acc={peak_i[2]:.3f} @ layer {peak_i[0]}  -> {p}", flush=True)

        if a.permute > 0:
            d = np.load(npz, allow_pickle=True)
            acts = d[a.pooling]
            sids = [str(s) for s in d["story_id"]]
            keep = [i for i,s in enumerate(sids) if s in lab]
            acts = acts[keep]; sk = [sids[i] for i in keep]
            ys = {"intent":  np.array([1 if lab[s]["intent_label"]=="guilty" else 0 for s in sk]),
                  "outcome": np.array([1 if lab[s]["outcome_label"]=="harm"   else 0 for s in sk])}
            groups = np.array([lab[s].get("scenario_group") or lab[s]["scenario_id"] for s in sk])
            prows = []
            for target, y in ys.items():
                peak_L = max((r for r in res if r[1]==target), key=lambda r:r[2])[0]
                for L in sorted({0, peak_L}):
                    obs, nm, p95, pval = permutation_null(
                        acts[:, L, :], y, groups, n_perm=a.permute, seed=0)
                    prows.append([L, target, round(obs,4), round(nm,4),
                                  round(p95,4), round(pval,5), a.permute])
                    print(f"    perm {target:8} L{L:<3} obs={obs:.3f} "
                          f"null={nm:.3f} p95={p95:.3f} p={pval:.4f}", flush=True)
            pp = os.path.join(a.out, f"{tag}_permnull{suffix}.csv")
            with open(pp, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["layer","target","obs_acc","null_mean","null_p95","empirical_p","n_perm"])
                w.writerows(prows)
            print(f"    -> {pp}", flush=True)
