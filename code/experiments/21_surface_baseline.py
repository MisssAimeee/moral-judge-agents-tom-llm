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


def load_offsets(path):
    if not path or not os.path.exists(path):
        return {}
    return {r["story_id"]: r for r in csv.DictReader(open(path))}


def span_text(row, offsets, span):
    """Truncate story text to the clause span used by the matching probe pooling.

    belief_last / action_last probes see only tokens up to that clause end. Comparing
    them against TF-IDF fit on the FULL story inflates the surface baseline (and shrinks
    the gap) whenever the outcome-determining sentence appears after the cut. Span-
    matched baselines close that mismatch.
    """
    text = row["text"]
    if span in (None, "full", "last", "mean"):
        return text
    off = offsets.get(row["story_id"])
    if not off:
        return text
    key = "belief_end" if span == "belief_last" else "action_end" if span == "action_last" else None
    if key is None:
        return text
    try:
        end = int(float(off[key]))
    except (TypeError, ValueError, KeyError):
        return text
    return text[:end]


def run(master_csv, out_csv, subset=None, subset_name=None, offsets=None, span="full"):
    from sklearn.feature_extraction.text import TfidfVectorizer
    group_cv_acc = load_probe_module().group_cv_acc
    offsets = offsets or {}

    rows = list(csv.DictReader(open(master_csv)))
    if subset is not None:
        rows = [r for r in rows if subset(r)]
    if not rows:
        raise SystemExit("no rows after subsetting")

    texts = [span_text(r, offsets, span) for r in rows]
    groups = np.array([r.get("scenario_group") or r["scenario_id"] for r in rows])
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
            acc, sd, deg = group_cv_acc(Xd, y, groups)
            chance = max(y.mean(), 1 - y.mean())
            out.append({
                "subset": subset_name or "all",
                "span": span,
                "feature_set": fs_name,
                "target": tname,
                "cv_acc": round(acc, 4),
                "cv_std": round(sd, 4),
                "chance": round(float(chance), 4),
                "n_items": len(rows),
                "n_features": Xd.shape[1],
                "degenerate": bool(deg),
            })
            print(f"  {subset_name or 'all':18} span={span:12} {fs_name:16} {tname:8} "
                  f"acc={acc:.3f} (chance {chance:.3f}, n={len(rows)}"
                  f"{', DEGENERATE' if deg else ''})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "dataset", "master", "moral_2x2_master.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "probe", "surface_baseline.csv"))
    ap.add_argument("--clause-offsets",
                    default=os.path.join(ROOT, "dataset", "master", "clause_offsets.csv"))
    a = ap.parse_args()

    # The four within-cell restrictions from Task C3 each need their own surface baseline,
    # so compute them here in one pass. Span-matched rows (belief_last / action_last) are
    # required for fair gaps against clause-position probes; full-story rows stay for
    # mean/last pooling and for the historical comparison.
    SUBSETS = {
        "all": None,
        "intent_noharm":    lambda r: r["outcome_label"] == "no_harm",
        "intent_harm":      lambda r: r["outcome_label"] == "harm",
        "outcome_innocent": lambda r: r["intent_label"] == "innocent",
        "outcome_guilty":   lambda r: r["intent_label"] == "guilty",
        "YS2008":           lambda r: r.get("source") == "YS2008",
        "YS2009":           lambda r: r.get("source") == "YS2009",
    }
    SPANS = ("full", "belief_last", "action_last")
    offsets = load_offsets(a.clause_offsets)

    all_rows = []
    print("=== TF-IDF surface baselines (same LR + GroupKFold as 02_probe.py) ===")
    for name, fn in SUBSETS.items():
        for span in SPANS:
            # Within-cell subsets only need the full span; source splits need all three
            # so C2 can be re-evaluated against matched baselines.
            if name not in ("all", "YS2008", "YS2009") and span != "full":
                continue
            all_rows += run(a.csv, a.out, subset=fn, subset_name=name,
                            offsets=offsets, span=span)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    cols = ["subset", "span", "feature_set", "target", "cv_acc", "cv_std", "chance",
            "n_items", "n_features", "degenerate"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
