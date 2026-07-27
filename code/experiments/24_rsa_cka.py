#!/usr/bin/env python3
"""
24_rsa_cka.py -- Phase 2: representational similarity (RSA) + CKA.

Probes answer "is intent decodable in model X", one model at a time. They cannot answer
the actual research question: models that AGREE behaviourally -- did they get there by the
same internal route? Two models can both decode intent at 0.70 while organising their
representational space completely differently. RSA compares the geometry itself.

The RDM is 298x298 no matter the hidden size (896 for Qwen-0.5B, 4096 for OLMo-7B). That
dimensional invariance is exactly why RSA works across architectures where activations
cannot be compared directly.

Tasks implemented
  R1  RDMs (1 - Pearson r) per model per selected layer, cached to outputs/rsa/
  R2  model x model second-order similarity (Spearman on upper triangles) + heatmap
  R3  the convergence test: does representational similarity track behavioural similarity?
  R4  hypothesis RDMs (intent / outcome / scenario) with scenario partialled out
  R5  base vs instruct geometry within family
  R6  linear + RBF CKA as an independent second metric
  R7  permutation nulls, shuffling within scenario as elsewhere in the project

Outputs -> outputs/rsa/
"""
import os, csv, glob, argparse, itertools, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RSA_DIR = os.path.join(ROOT, "outputs", "rsa")

# relative depths sampled in addition to each model's own peak-intent layer, so models with
# 25 / 29 / 33 layers can still be compared at matched depth
REL_DEPTHS = [0.0, 0.25, 0.5, 0.75, 1.0]


# ---------------------------------------------------------------- helpers
def upper(m):
    iu = np.triu_indices_from(m, k=1)
    return m[iu]


def rankdata(x):
    from scipy.stats import rankdata as rd
    return rd(x)


def spearman(a, b):
    from scipy.stats import spearmanr
    r, p = spearmanr(a, b)
    return float(r), float(p)


def partial_spearman(x, y, z):
    """Spearman(x, y) with z partialled out, via residualising the ranks."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    Z = np.column_stack([np.ones_like(rz), rz])
    bx = np.linalg.lstsq(Z, rx, rcond=None)[0]
    by = np.linalg.lstsq(Z, ry, rcond=None)[0]
    ex, ey = rx - Z @ bx, ry - Z @ by
    r = float(np.corrcoef(ex, ey)[0, 1])
    return r


def build_rdm(X):
    """298x298 dissimilarity, 1 - Pearson r between stimulus activation vectors."""
    Xc = X - X.mean(axis=1, keepdims=True)
    n = np.linalg.norm(Xc, axis=1, keepdims=True)
    n[n == 0] = 1e-12
    C = (Xc / n) @ (Xc / n).T
    return 1.0 - np.clip(C, -1.0, 1.0)


def linear_cka(X, Y):
    """Linear CKA. Invariant to orthogonal transform and isotropic scaling."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    hsic = np.linalg.norm(X.T @ Y, "fro") ** 2
    nx = np.linalg.norm(X.T @ X, "fro")
    ny = np.linalg.norm(Y.T @ Y, "fro")
    return float(hsic / (nx * ny)) if nx and ny else float("nan")


def _center_gram(K):
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def rbf_cka(X, Y, sigma_frac=0.5):
    def gram(A):
        sq = np.sum(A ** 2, 1)
        D = sq[:, None] + sq[None, :] - 2 * A @ A.T
        D = np.maximum(D, 0)
        med = np.median(D[D > 0]) if np.any(D > 0) else 1.0
        return np.exp(-D / (2 * (sigma_frac ** 2) * med + 1e-12))
    Kx, Ky = _center_gram(gram(X)), _center_gram(gram(Y))
    num = np.sum(Kx * Ky)
    den = np.sqrt(np.sum(Kx * Kx) * np.sum(Ky * Ky))
    return float(num / den) if den else float("nan")


def permute_within_groups(idx, groups, rng):
    out = np.array(idx, copy=True)
    for g in np.unique(groups):
        w = np.where(groups == g)[0]
        out[w] = out[rng.permutation(w)]
    return out


def rdm_perm_p(rdm_a, vec_b, groups, n_perm, rng):
    """
    Null for corr(RDM_a, hypothesis_b): permute stimulus identity within scenario and
    rebuild the upper triangle of RDM_a each time.
    """
    obs, _ = spearman(upper(rdm_a), vec_b)
    n = rdm_a.shape[0]
    base = np.arange(n)
    null = np.empty(n_perm)
    for i in range(n_perm):
        p = permute_within_groups(base, groups, rng)
        null[i] = spearman(upper(rdm_a[np.ix_(p, p)]), vec_b)[0]
    pval = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return float(obs), float(null.mean()), float(pval)


# ---------------------------------------------------------------- loading
def load_model(npz, lab, pooling):
    d = np.load(npz, allow_pickle=True)
    sids = [str(s) for s in d["story_id"]]
    keep = [i for i, s in enumerate(sids) if s in lab]
    return d[pooling][keep], [sids[i] for i in keep]


def peak_intent_layer(tag, probe_dir, suffix):
    p = os.path.join(probe_dir, f"{tag}_probe{suffix}.csv")
    if not os.path.exists(p):
        return None
    best = None
    for r in csv.DictReader(open(p)):
        if r["target"] == "intent":
            a = float(r["cv_acc"])
            if best is None or a > best[1]:
                best = (int(r["layer"]), a)
    return best[0] if best else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "dataset", "master", "moral_2x2_master.csv"))
    ap.add_argument("--acts", default=os.path.join(ROOT, "outputs", "acts"))
    ap.add_argument("--probe", default=os.path.join(ROOT, "outputs", "probe"))
    ap.add_argument("--out", default=RSA_DIR)
    ap.add_argument("--pooling", default="mean", choices=["last", "mean"],
                    help="mean is the default: last-token pooling badly understates intent "
                         "(OLMo-Instruct 0.77 last vs 0.94 mean)")
    ap.add_argument("--perm", type=int, default=1000)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    suffix = "" if a.pooling == "last" else f"_{a.pooling}"
    lab = {r["story_id"]: r for r in csv.DictReader(open(a.csv))}
    npzs = sorted(glob.glob(os.path.join(a.acts, "*.npz")))
    if not npzs:
        raise SystemExit(f"no activations in {a.acts}")

    if a.dry_run:
        print("=== RSA/CKA PLAN (dry-run) ===")
        print(f"pooling={a.pooling}  perms={a.perm}  models={len(npzs)}")
        for n in npzs:
            tag = os.path.basename(n)[:-4]
            d = np.load(n, allow_pickle=True)
            L = d[a.pooling].shape[1]
            pk = peak_intent_layer(tag, a.probe, suffix)
            print(f"  {tag:26} layers={L:3}  peak_intent_layer={pk}  "
                  f"RDM layers={sorted({*[int(round(r*(L-1))) for r in REL_DEPTHS], *([pk] if pk is not None else [])})}")
        print(f"\npairs for R2/R3/R6: {len(list(itertools.combinations(npzs,2)))}")
        print(f"Launch:\n  JOBNAME=rsa CPUS=16 bash engaging/submit_cpu.sh "
              f"\"python code/experiments/24_rsa_cka.py --pooling {a.pooling} --perm {a.perm}\"")
        return

    rng = np.random.default_rng(0)

    # ---------------- R1: RDMs ------------------------------------------------
    print("=== R1: building RDMs ===", flush=True)
    store = {}   # tag -> dict(layer -> rdm), plus meta
    for npz in npzs:
        tag = os.path.basename(npz)[:-4]
        A, sids = load_model(npz, lab, a.pooling)
        nL = A.shape[1]
        pk = peak_intent_layer(tag, a.probe, suffix)
        layers = sorted({int(round(r * (nL - 1))) for r in REL_DEPTHS} |
                        ({pk} if pk is not None else set()))
        rdms = {}
        for L in layers:
            rdm = build_rdm(A[:, L, :].astype(np.float64))
            rdms[L] = rdm
            np.save(os.path.join(a.out, f"rdm_{tag}_L{L}{suffix}.npy"), rdm.astype(np.float32))
        store[tag] = {"rdms": rdms, "sids": sids, "n_layers": nL, "peak": pk,
                      "acts_path": npz}
        print(f"  {tag:26} layers={nL:3} peak={pk} cached={sorted(rdms)}", flush=True)

    tags = sorted(store)
    sids0 = store[tags[0]]["sids"]
    groups = np.array([lab[s]["scenario_id"] for s in sids0])

    # ---------------- R4: hypothesis RDMs ------------------------------------
    print("\n=== R4: hypothesis RDMs (scenario partialled out) ===", flush=True)
    intent_v = np.array([1 if lab[s]["intent_label"] == "guilty" else 0 for s in sids0])
    outcome_v = np.array([1 if lab[s]["outcome_label"] == "harm" else 0 for s in sids0])
    scen_v = np.array([lab[s]["scenario_id"] for s in sids0])
    H_intent = (intent_v[:, None] != intent_v[None, :]).astype(float)
    H_outcome = (outcome_v[:, None] != outcome_v[None, :]).astype(float)
    H_scen = (scen_v[:, None] != scen_v[None, :]).astype(float)
    u_int, u_out, u_scen = upper(H_intent), upper(H_outcome), upper(H_scen)

    hyp_rows = []
    for tag in tags:
        for L, rdm in sorted(store[tag]["rdms"].items()):
            u = upper(rdm)
            r_i, _ = spearman(u, u_int)
            r_o, _ = spearman(u, u_out)
            pr_i = partial_spearman(u, u_int, u_scen)
            pr_o = partial_spearman(u, u_out, u_scen)
            denom = abs(pr_i) + abs(pr_o)
            hyp_rows.append({
                "model": tag, "layer": L, "is_peak": L == store[tag]["peak"],
                "rel_depth": round(L / max(store[tag]["n_layers"] - 1, 1), 3),
                "r_intent": round(r_i, 4), "r_outcome": round(r_o, 4),
                "partial_r_intent": round(pr_i, 4), "partial_r_outcome": round(pr_o, 4),
                # representational analogue of the behavioural b_intent vs b_outcome split
                "intent_org_ratio": round(abs(pr_i) / denom, 4) if denom > 1e-9 else "",
            })
        pk = store[tag]["peak"]
        pr = next((h for h in hyp_rows if h["model"] == tag and h["layer"] == pk), None)
        if pr:
            print(f"  {tag:26} L{pk:<3} partial r_intent={pr['partial_r_intent']:+.3f} "
                  f"r_outcome={pr['partial_r_outcome']:+.3f} ratio={pr['intent_org_ratio']}", flush=True)

    with open(os.path.join(a.out, "hypothesis_rdm.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hyp_rows[0].keys()))
        w.writeheader(); w.writerows(hyp_rows)

    # ---------------- R7: permutation nulls for the hypothesis fits ----------
    print(f"\n=== R7: permutation nulls (n={a.perm}) ===", flush=True)
    null_rows = []
    for tag in tags:
        pk = store[tag]["peak"]
        if pk is None or pk not in store[tag]["rdms"]:
            continue
        rdm = store[tag]["rdms"][pk]
        for hname, hv in (("intent", u_int), ("outcome", u_out)):
            obs, nm, p = rdm_perm_p(rdm, hv, groups, a.perm, rng)
            null_rows.append({"model": tag, "layer": pk, "hypothesis": hname,
                              "obs_r": round(obs, 4), "null_mean": round(nm, 4),
                              "empirical_p": round(p, 5), "n_perm": a.perm})
            print(f"  {tag:26} {hname:8} r={obs:+.3f} null={nm:+.3f} p={p:.4f}", flush=True)
    if null_rows:
        with open(os.path.join(a.out, "rsa_permutation_null.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(null_rows[0].keys()))
            w.writeheader(); w.writerows(null_rows)

    # ---------------- R2 + R6: model x model similarity ----------------------
    print("\n=== R2/R6: model x model similarity ===", flush=True)
    def layer_at(tag, mode):
        nL = store[tag]["n_layers"]
        if mode == "peak":
            return store[tag]["peak"] if store[tag]["peak"] is not None else int(round(0.75 * (nL - 1)))
        return int(round(0.75 * (nL - 1)))

    sim_rows = []
    acts_cache = {}
    for mode in ("peak", "reldepth0.75"):
        for t1, t2 in itertools.combinations(tags, 2):
            L1, L2 = layer_at(t1, mode), layer_at(t2, mode)
            r1 = store[t1]["rdms"].get(L1)
            r2 = store[t2]["rdms"].get(L2)
            if r1 is None or r2 is None:
                continue
            rho, _ = spearman(upper(r1), upper(r2))
            for t, L in ((t1, L1), (t2, L2)):
                if (t, L) not in acts_cache:
                    A, _ = load_model(store[t]["acts_path"], lab, a.pooling)
                    acts_cache[(t, L)] = A[:, L, :].astype(np.float64)
            ck = linear_cka(acts_cache[(t1, L1)], acts_cache[(t2, L2)])
            ckr = rbf_cka(acts_cache[(t1, L1)], acts_cache[(t2, L2)])
            sim_rows.append({"mode": mode, "model_a": t1, "model_b": t2,
                             "layer_a": L1, "layer_b": L2,
                             "rsa_spearman": round(rho, 4),
                             "cka_linear": round(ck, 4), "cka_rbf": round(ckr, 4)})
        print(f"  {mode}: {sum(1 for s in sim_rows if s['mode']==mode)} pairs", flush=True)

    with open(os.path.join(a.out, "model_similarity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sim_rows[0].keys()))
        w.writeheader(); w.writerows(sim_rows)

    # ---------------- R5: base vs instruct geometry --------------------------
    print("\n=== R5: base vs instruct geometry ===", flush=True)
    r5 = []
    for t in tags:
        if t.endswith("-Instruct"):
            base = t[: -len("-Instruct")]
            if base in store:
                row = next((s for s in sim_rows
                            if s["mode"] == "peak"
                            and {s["model_a"], s["model_b"]} == {t, base}), None)
                if row:
                    r5.append({"family": base, "rsa_spearman": row["rsa_spearman"],
                               "cka_linear": row["cka_linear"], "cka_rbf": row["cka_rbf"]})
                    print(f"  {base:26} base<->instruct  RSA={row['rsa_spearman']:.3f} "
                          f"CKA={row['cka_linear']:.3f}", flush=True)
    if r5:
        with open(os.path.join(a.out, "base_vs_instruct_geometry.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(r5[0].keys()))
            w.writeheader(); w.writerows(r5)

    # ---------------- R3: the convergence test -------------------------------
    print("\n=== R3: convergence test ===", flush=True)
    conv = convergence_test(sim_rows, a.out)

    # ---------------- figures -------------------------------------------------
    make_heatmap(sim_rows, tags, a.out)
    print(f"\n-> {a.out}")
    if conv:
        print(f"R3: r={conv['r']:+.3f} (95% CI {conv['ci_lo']:+.3f}..{conv['ci_hi']:+.3f}), "
              f"n_pairs={conv['n_pairs']} -> {conv['verdict']}")


def load_behavioral_contrast():
    """model tag -> behavioural contrast, from the master ladder CSV."""
    p = os.path.join(ROOT, "outputs", "master_all_models.csv")
    if not os.path.exists(p):
        return {}
    out = {}
    for r in csv.DictReader(open(p)):
        name = (r.get("model") or r.get("model_name") or "").strip()
        if not name:
            continue
        if str(r.get("degenerate", "")).lower() in ("true", "1"):
            continue
        for key in ("contrast", "contrast_mean", "delta_contrast"):
            v = (r.get(key) or "").strip()
            if v:
                try:
                    out[name.split("/")[-1]] = float(v)
                except ValueError:
                    pass
                break
    return out


def convergence_test(sim_rows, outdir, n_boot=1000):
    """Do behaviourally similar pairs have more similar geometry?"""
    beh = load_behavioral_contrast()
    rows = []
    for s in sim_rows:
        if s["mode"] != "peak":
            continue
        a_, b_ = s["model_a"], s["model_b"]
        if a_ in beh and b_ in beh:
            rows.append({"model_a": a_, "model_b": b_,
                         "rsa_spearman": s["rsa_spearman"],
                         "cka_linear": s["cka_linear"],
                         "abs_behavioral_diff": round(abs(beh[a_] - beh[b_]), 4)})
    if len(rows) < 4:
        print(f"  only {len(rows)} pairs have behavioural contrasts on both sides -- "
              f"convergence test underpowered, skipped")
        if rows:
            with open(os.path.join(outdir, "convergence_pairs.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
        return None

    x = np.array([r["rsa_spearman"] for r in rows])
    y = np.array([r["abs_behavioral_diff"] for r in rows])
    from scipy.stats import spearmanr
    r_obs = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        if len(np.unique(x[idx])) < 3:
            continue
        boots.append(spearmanr(x[idx], y[idx]).statistic)
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"),) * 2
    # negative r = more similar geometry goes with smaller behavioural difference
    verdict = ("convergent: similar behaviour tracks similar geometry" if hi < 0 else
               "same answer, different route (null)" if lo < 0 < hi else
               "divergent (unexpected sign)")
    with open(os.path.join(outdir, "convergence_pairs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    out = {"r": r_obs, "ci_lo": float(lo), "ci_hi": float(hi),
           "n_pairs": len(rows), "verdict": verdict}
    with open(os.path.join(outdir, "convergence_test.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def make_heatmap(sim_rows, tags, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for mode, metric in (("peak", "rsa_spearman"), ("peak", "cka_linear")):
            M = np.full((len(tags), len(tags)), np.nan)
            for i, t1 in enumerate(tags):
                M[i, i] = 1.0
                for j, t2 in enumerate(tags):
                    if i >= j:
                        continue
                    row = next((s for s in sim_rows if s["mode"] == mode
                                and {s["model_a"], s["model_b"]} == {t1, t2}), None)
                    if row:
                        M[i, j] = M[j, i] = row[metric]
            fig, ax = plt.subplots(figsize=(8.5, 7))
            im = ax.imshow(M, cmap="viridis", vmin=np.nanmin(M), vmax=1.0)
            ax.set_xticks(range(len(tags))); ax.set_xticklabels(tags, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(tags))); ax.set_yticklabels(tags, fontsize=7)
            for i in range(len(tags)):
                for j in range(len(tags)):
                    if not np.isnan(M[i, j]):
                        ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                                fontsize=6, color="w")
            ax.set_title(f"{metric} at peak-intent layer")
            fig.colorbar(im); fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"model_similarity_heatmap_{metric}.png"), dpi=150)
            plt.close(fig)
        print(f"  heatmaps -> {outdir}")
    except Exception as e:
        print("  heatmap skipped:", e)


if __name__ == "__main__":
    main()
