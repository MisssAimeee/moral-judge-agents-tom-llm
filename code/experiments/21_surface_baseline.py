#!/usr/bin/env python3
"""
21_surface_baseline.py -- Phase 1 / Task C2: what can surface lexis alone achieve?

Fits the SAME classifier and the SAME grouped-CV protocol as 02_probe.py, but on TF-IDF
features of the raw story text instead of model activations. No activations are touched.

This turns "the outcome probe might be lexical" into a number. Every probe accuracy
reported after this point must be quoted alongside its surface baseline: the scientific
claim is the GAP between them, not the raw accuracy.

Outputs
  outputs/probe/surface_baseline.csv   feature_set, target, cv_acc, cv_std, chance, n_features
"""
import os, csv, argparse, importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def load_probe_module():
    """02_probe.py starts with a digit, so it can't be imported normally."""
    p = os.path.join(ROOT, "code", "02_probe.py")
    spec = importlib.util.spec_from_file_location("probe_02", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


FEATURE_SETS = {
    "tfidf_word_1_2": dict(analyzer="word", ngram_range=(1, 2), min_df=2),
    "tfidf_char_3_5": dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2),
}


def run(master_csv, out_csv, subset=None, subset_name=None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    group_cv_acc = load_probe_module().group_cv_acc

    rows = list(csv.DictReader(open(master_csv)))
    if subset is not None:
        rows = [r for r in rows if subset(r)]
    if not rows:
        raise SystemExit("no rows after subsetting")

    texts = [r["text"] for r in rows]
    groups = np.array([r["scenario_id"] for r in rows])
    targets = {
        "intent": np.array([1 if r["intent_label"] == "guilty" else 0 for r in rows]),
        "outcome": np.array([1 if r["outcome_label"] == "harm" else 0 for r in rows]),
    }

    out = []
    for fs_name, kw in FEATURE_SETS.items():
        X = TfidfVectorizer(**kw).fit_transform(texts)
        # group_cv_acc standardises with the mean, which sparse input rejects
        Xd = np.asarray(X.todense())
        for tname, y in targets.items():
            if len(np.unique(y)) < 2:
                continue
            acc, sd = group_cv_acc(Xd, y, groups)
            chance = max(y.mean(), 1 - y.mean())
            out.append({
                "subset": subset_name or "all",
                "feature_set": fs_name,
                "target": tname,
                "cv_acc": round(acc, 4),
                "cv_std": round(sd, 4),
                "chance": round(float(chance), 4),
                "n_items": len(rows),
                "n_features": Xd.shape[1],
            })
            print(f"  {subset_name or 'all':18} {fs_name:16} {tname:8} "
                  f"acc={acc:.3f} (chance {chance:.3f}, n={len(rows)})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "dataset", "master", "moral_2x2_master.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "probe", "surface_baseline.csv"))
    a = ap.parse_args()

    # The four within-cell restrictions from Task C3 each need their own surface baseline,
    # so compute them here in one pass.
    SUBSETS = {
        "all": None,
        "intent_noharm":    lambda r: r["outcome_label"] == "no_harm",
        "intent_harm":      lambda r: r["outcome_label"] == "harm",
        "outcome_innocent": lambda r: r["intent_label"] == "innocent",
        "outcome_guilty":   lambda r: r["intent_label"] == "guilty",
    }

    all_rows = []
    print("=== TF-IDF surface baselines (same LR + GroupKFold as 02_probe.py) ===")
    for name, fn in SUBSETS.items():
        all_rows += run(a.csv, a.out, subset=fn, subset_name=name)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    cols = ["subset", "feature_set", "target", "cv_acc", "cv_std", "chance", "n_items", "n_features"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
