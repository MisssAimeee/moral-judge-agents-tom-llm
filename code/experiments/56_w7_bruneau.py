#!/usr/bin/env python3
"""W7 (appendix) -- Bruneau, Pluta & Saxe 2011 stimuli: is the harm code domain-selective?

The project's central asymmetry is that outcome/harm is what the rating tracks. That
raises a question the moral 2x2 cannot answer, because every one of its harms is the same
kind of harm: is the harm representation a generic "something bad happened" detector, or
is it differentiated by how the badness arises? Bruneau's stimuli are the right probe for
this because they hold narrative content fixed and vary only the ending, across three
domains of suffering:

  PP  / PPC   physical pain vs matched control   (bodily injury, no mentalizing needed)
  EP  / EPC   emotional pain vs matched control  (social/relational loss)
  FBP / FBC   false-belief pain vs matched control (suffering caused by a false belief)

24 numbered items per condition, matched item-for-item: PPC item 7 is PP item 7 with a
harmless ending. That pairing is the design's strength and it dictates the analysis --
folds and bootstrap resamples are over ITEM PAIRS, never over individual stimuli, or the
harmful and harmless version of the same story land on opposite sides of a split and the
probe scores lexical overlap.

Analyses
  1. Within-domain harm decoding, grouped CV by item pair, at every layer.
  2. The moral-vs-non-moral interaction: acc(EP) - acc(PP) and acc(FBP) - acc(PP),
     bootstrapped over item pairs.
  3. A 3x3 cross-domain transfer matrix: fit on one domain's harm contrast, test on
     another. High off-diagonal transfer = one generic harm code. Low = domain-specific
     codes. This is the part that actually answers the selectivity question; a difference
     in within-domain accuracy alone can come from one contrast being easier.
  4. TF-IDF surface baselines for every cell of 1 and 3, since these endings differ
     lexically by construction and an unbaselined accuracy here means nothing.

LIMITATION, stated wherever a number from this script appears: these stimuli are NOT an
intent x outcome factorial. There is no condition in which a character intends harm and
none occurs. PP/EP hold belief fixed and vary outcome; FBP/FBC vary belief content but
belief and outcome move together, so neither contrast can separate intent from outcome.
W7 therefore tests outcome/harm selectivity only, and cannot speak to intent-weighting.
It is appendix support for the claim that the harm code is rich, not evidence about the
intent code.
"""
import os, re, sys, csv, json, argparse
from collections import defaultdict

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)
import tom_common as tc  # noqa: E402
import numpy as np  # noqa: E402

PDF = os.path.join(tc.ROOT, "BruneauPlutaSaxe2011_Stimuli_0.pdf")
DATA_DIR = os.path.join(tc.ROOT, "dataset", "bruneau")
STIM_CSV = os.path.join(DATA_DIR, "bruneau_stimuli.csv")
OUT_DIR = os.path.join(tc.ROOT, "outputs", "experiments")
ACT_DIR = os.path.join(tc.ROOT, "outputs", "acts_bruneau")
PARSE_MD = os.path.join(OUT_DIR, "W7_PARSE_REPORT.md")
RES_CSV = os.path.join(OUT_DIR, "w7_bruneau_probes.csv")
XFER_CSV = os.path.join(OUT_DIR, "w7_bruneau_transfer.csv")
MD_OUT = os.path.join(OUT_DIR, "W7_BRUNEAU.md")
FIG = os.path.join(tc.ROOT, "outputs", "figures_final", "w7_bruneau_selectivity.png")

SECTIONS = [
    ("PP",  "Physical Pain (PP)",           "physical",     1),
    ("PPC", "Physical Pain Control (PPC)",  "physical",     0),
    ("EP",  "Emotional Pain (EP)",          "emotional",    1),
    ("EPC", "Emotional Pain Control (EPC)", "emotional",    0),
    ("FBP", "False Belief Pain (FBP)",      "false_belief", 1),
    ("FBC", "False Belief Control (FBC)",   "false_belief", 0),
]
DOMAINS = ["physical", "emotional", "false_belief"]
DOMAIN_LABEL = {"physical": "PP/PPC (physical)", "emotional": "EP/EPC (emotional)",
                "false_belief": "FBP/FBC (false belief)"}
# Two W3 models, so the layer indices line up with the steering and manipulation results.
MODELS = ["allenai/OLMo-2-1124-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]


# --------------------------------------------------------------------------- parse

def parse_pdf():
    import pdfplumber
    with pdfplumber.open(PDF) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    full = "\n".join(pages)

    # Cut at the section headers, in document order, so item text that follows a header
    # belongs to that condition and page breaks mid-item are harmless.
    marks = []
    for code, header, domain, harm in SECTIONS:
        i = full.find(header)
        if i < 0:
            raise SystemExit(f"header not found in PDF: {header!r}")
        marks.append((i, code, header, domain, harm))
    marks.sort()

    rows, problems = [], []
    for j, (i, code, header, domain, harm) in enumerate(marks):
        start = i + len(header)
        end = marks[j + 1][0] if j + 1 < len(marks) else len(full)
        body = full[start:end]
        parts = re.split(r"(?m)^\s*(\d{1,2})\.\s+", body)
        # parts = [preamble, num, text, num, text, ...]
        if parts[0].strip():
            problems.append(f"{code}: {len(parts[0].strip())} chars before item 1 dropped")
        nums = [int(parts[k]) for k in range(1, len(parts), 2)]
        if nums != list(range(1, len(nums) + 1)):
            problems.append(f"{code}: item numbers not 1..n sequential: {nums}")
        for k in range(1, len(parts), 2):
            n = int(parts[k])
            text = re.sub(r"\s+", " ", parts[k + 1]).strip()
            text = text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
            rows.append(dict(story_id=f"BRU-{code}-{n:02d}", condition=code,
                             domain=domain, harm=harm, pair=f"{domain}-{n:02d}",
                             item_num=n, word_count=len(text.split()), text=text))

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(STIM_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    # Every pair must have exactly one harm and one control member or the design is broken.
    pair_counts = defaultdict(int)
    for r in rows:
        pair_counts[r["pair"]] += 1
    unpaired = sorted(p for p, c in pair_counts.items() if c != 2)
    with open(PARSE_MD, "w") as f:
        f.write("# W7 parse report — Bruneau, Pluta & Saxe (2011) stimuli\n\n")
        f.write(f"Source: `{os.path.basename(PDF)}` ({len(pages)} pages). Parsed by "
                "`code/experiments/56_w7_bruneau.py --parse` into "
                f"`{os.path.relpath(STIM_CSV, tc.ROOT)}`.\n\n")
        f.write("| condition | domain | harm | n items | median words |\n")
        f.write("|---|---|---:|---:|---:|\n")
        for code, _, domain, harm in SECTIONS:
            rs = by_cond[code]
            med = int(np.median([r["word_count"] for r in rs])) if rs else 0
            f.write(f"| {code} | {domain} | {harm} | {len(rs)} | {med} |\n")
        f.write(f"\nTotal {len(rows)} stimuli, {len(pair_counts)} item pairs. "
                f"Unpaired items: {unpaired if unpaired else 'none'}.\n")
        if problems:
            f.write("\n## Parse warnings\n\n" + "".join(f"- {p}\n" for p in problems))
        f.write("\n## Matched-pair check (verbatim, one pair per domain)\n\n"
                "The two members of a pair share their opening and diverge only at the "
                "ending. This is why folds and bootstrap resamples are over pairs.\n")
        for domain in DOMAINS:
            f.write(f"\n### {DOMAIN_LABEL[domain]}, item 01\n")
            for r in rows:
                if r["pair"] == f"{domain}-01":
                    f.write(f"\n- **{r['condition']}** (harm={r['harm']}): "
                            f"{r['text']}\n")
    print(f"  wrote {os.path.relpath(STIM_CSV, tc.ROOT)} ({len(rows)} stimuli, "
          f"{len(pair_counts)} pairs)")
    print(f"  wrote {os.path.relpath(PARSE_MD, tc.ROOT)}")
    for p in problems:
        print(f"  [warn] {p}")
    return rows


def load_stimuli():
    if not os.path.exists(STIM_CSV):
        raise SystemExit(f"{STIM_CSV} missing — run with --parse first")
    rows = list(csv.DictReader(open(STIM_CSV)))
    for r in rows:
        r["harm"] = int(r["harm"])
        r["item_num"] = int(r["item_num"])
    return rows


# ----------------------------------------------------------------------- activations

def act_path(model):
    return os.path.join(ACT_DIR, re.sub(r"[^\w.-]", "_", model) + "_last.npz")


def extract(model, rows, dtype="float32", batch=1):
    """Last-token hidden state at every layer. CPU by default: 144 short stimuli.

    Kept self-contained rather than reusing 01_extract_activations.py, which hardcodes
    fp16 + device_map="auto" for GPU nodes; W7 is an appendix item that should not have to
    queue behind the GPU jobs.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    td = dict(float32=torch.float32, bfloat16=torch.bfloat16,
              float16=torch.float16)[dtype if dev == "cpu" else "float16"]
    print(f"  loading {model} on {dev} ({td})", flush=True)
    tok = AutoTokenizer.from_pretrained(model)
    mdl = AutoModelForCausalLM.from_pretrained(model, torch_dtype=td,
                                              output_hidden_states=True).to(dev).eval()
    out = []
    with torch.no_grad():
        for i, r in enumerate(rows):
            enc = tok(r["text"], return_tensors="pt").to(dev)
            hs = mdl(**enc).hidden_states                # tuple[L+1] of [1,T,H]
            out.append(np.stack([h[0, -1].float().cpu().numpy() for h in hs]))
            if (i + 1) % 24 == 0:
                print(f"    {i + 1}/{len(rows)}", flush=True)
    X = np.stack(out)                                    # [n, L+1, H]
    os.makedirs(ACT_DIR, exist_ok=True)
    np.savez_compressed(act_path(model), X=X.astype(np.float16),
                        story_id=np.array([r["story_id"] for r in rows]))
    del mdl
    print(f"  wrote {os.path.relpath(act_path(model), tc.ROOT)} {X.shape}")
    return X


def load_acts(model, rows):
    p = act_path(model)
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=False)
    ids = list(z["story_id"])
    want = [r["story_id"] for r in rows]
    if ids != want:
        idx = {s: i for i, s in enumerate(ids)}
        if not all(s in idx for s in want):
            return None
        return z["X"][[idx[s] for s in want]].astype(np.float32)
    return z["X"].astype(np.float32)


# --------------------------------------------------------------------------- probes

def cv_acc(X, y, pairs, seed=0):
    """Grouped CV accuracy, groups = item pairs (a pair never splits across folds)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    n_splits = min(6, len(set(pairs)))
    if n_splits < 2:
        return float("nan")
    accs = []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, pairs):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=3000, C=1.0, random_state=seed)
        clf.fit(sc.transform(X[tr]), y[tr])
        accs.append(clf.score(sc.transform(X[te]), y[te]))
    return float(np.mean(accs))


def transfer_acc(Xa, ya, Xb, yb, seed=0):
    """Fit on all of domain A, score all of domain B. No item overlap across domains."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xa)
    clf = LogisticRegression(max_iter=3000, C=1.0, random_state=seed).fit(sc.transform(Xa), ya)
    return float(clf.score(sc.transform(Xb), yb))


def tfidf_acc(texts, y, pairs, seed=0):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    n_splits = min(6, len(set(pairs)))
    if n_splits < 2:
        return float("nan")
    accs = []
    idx = np.arange(len(y))
    for tr, te in GroupKFold(n_splits=n_splits).split(idx, y, pairs):
        v = TfidfVectorizer(min_df=1, sublinear_tf=True)
        A = v.fit_transform([texts[i] for i in tr])
        B = v.transform([texts[i] for i in te])
        clf = LogisticRegression(max_iter=3000, random_state=seed).fit(A, y[tr])
        accs.append(clf.score(B, y[te]))
    return float(np.mean(accs))


def boot_diff(pairs_a, pairs_b, statfn, B=2000, seed=0):
    """Bootstrap a difference of two accuracies by resampling item pairs in each domain."""
    rng = np.random.default_rng(seed)
    pa, pb = list(pairs_a), list(pairs_b)
    vals = np.empty(B)
    for b in range(B):
        sa = [pa[i] for i in rng.integers(0, len(pa), len(pa))]
        sb = [pb[i] for i in rng.integers(0, len(pb), len(pb))]
        vals[b] = statfn(sa, sb)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def subset(rows, X, domain, pairs_keep=None):
    idx = [i for i, r in enumerate(rows) if r["domain"] == domain
           and (pairs_keep is None or r["pair"] in pairs_keep)]
    y = np.array([rows[i]["harm"] for i in idx])
    g = [rows[i]["pair"] for i in idx]
    return (X[idx] if X is not None else None), y, g, [rows[i]["text"] for i in idx], idx


def run_model(model, rows, layer_frac, boot, seed, dtype):
    X = load_acts(model, rows)
    if X is None:
        X = extract(model, rows, dtype=dtype)
        X = load_acts(model, rows)
    n_layers = X.shape[1]
    layers = sorted({1, n_layers // 4, int(round(layer_frac * (n_layers - 1))),
                     n_layers // 2, (3 * n_layers) // 4, n_layers - 1})
    recs, xfer = [], []
    for domain in DOMAINS:
        _, y, g, texts, idx = subset(rows, None, domain)
        surf = tfidf_acc(texts, y, g, seed)
        for L in layers:
            Xl = X[idx, L, :]
            acc = cv_acc(Xl, y, g, seed)
            recs.append(dict(model=model, domain=domain, layer=L, n_layers=n_layers,
                             n_pairs=len(set(g)), n_items=len(y), probe_acc=acc,
                             tfidf_acc=surf, gap=acc - surf))
    # transfer at the single best-mean layer, chosen on the mean over domains so it is
    # not chosen per contrast (which would inflate every diagonal entry)
    by_layer = defaultdict(list)
    for r in recs:
        by_layer[r["layer"]].append(r["probe_acc"])
    best_L = max(by_layer, key=lambda L: float(np.mean(by_layer[L])))
    for a in DOMAINS:
        Xa, ya, ga, _, ia = subset(rows, X, a)
        for b in DOMAINS:
            Xb, yb, gb, _, ib = subset(rows, X, b)
            if a == b:
                acc = cv_acc(X[ia, best_L, :], ya, ga, seed)
            else:
                acc = transfer_acc(X[ia, best_L, :], ya, X[ib, best_L, :], yb, seed)
            xfer.append(dict(model=model, layer=best_L, fit_on=a, test_on=b, acc=acc,
                             within=(a == b)))
    # interaction: is harm decoding better in the mentalizing domains than the physical one
    inter = []
    for d in ("emotional", "false_belief"):
        rp = [r for r in recs if r["domain"] == "physical" and r["layer"] == best_L][0]
        rd = [r for r in recs if r["domain"] == d and r["layer"] == best_L][0]
        _, _, gp, _, ip = subset(rows, X, "physical")
        _, _, gd, _, idd = subset(rows, X, d)
        # pair -> the (harm, control) row indices it contributes. A resampled pair enters
        # the design twice, which is what a pair-level bootstrap means; deduplicating
        # would turn it into a subsample and understate the spread.
        pitems = defaultdict(list)
        for i, p in zip(ip, gp):
            pitems[p].append(i)
        ditems = defaultdict(list)
        for i, p in zip(idd, gd):
            ditems[p].append(i)

        def acc_of(sample, table, L=best_L):
            idx, grp = [], []
            for rep, p in enumerate(sample):
                for i in table[p]:
                    idx.append(i)
                    grp.append(f"{p}#{rep}")   # a duplicated pair is its own fold group
            y = np.array([rows[i]["harm"] for i in idx])
            if len(set(y)) < 2:
                return float("nan")
            return cv_acc(X[idx, L, :], y, grp, seed)

        def stat(sa, sb):
            return acc_of(sb, ditems) - acc_of(sa, pitems)

        lo, hi = boot_diff(sorted(pitems), sorted(ditems), stat, B=boot, seed=seed)
        inter.append(dict(model=model, layer=best_L, domain=d, ref="physical",
                          d_acc=rd["probe_acc"] - rp["probe_acc"], lo=lo, hi=hi,
                          acc_domain=rd["probe_acc"], acc_ref=rp["probe_acc"],
                          d_tfidf=rd["tfidf_acc"] - rp["tfidf_acc"]))
    return recs, xfer, inter


def write_csv(path, recs):
    if not recs:
        return
    old = list(csv.DictReader(open(path))) if os.path.exists(path) else []
    models = {r["model"] for r in recs}
    merged = [r for r in old if r["model"] not in models] + recs
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]))
        w.writeheader()
        for r in merged:
            w.writerow({k: r.get(k, "") for k in recs[0]})


def plot(recs, xfer):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (no figure: {e})")
        return
    models = sorted({r["model"] for r in recs})
    fig, axes = plt.subplots(1, len(models) + 1, figsize=(5 + 4 * len(models), 4.0))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        for domain in DOMAINS:
            rs = sorted([r for r in recs if r["model"] == m and r["domain"] == domain],
                        key=lambda r: float(r["layer"]))
            if not rs:
                continue
            ax.plot([float(r["layer"]) for r in rs], [float(r["probe_acc"]) for r in rs],
                    "-o", ms=4, label=DOMAIN_LABEL[domain])
            ax.axhline(float(rs[0]["tfidf_acc"]), ls=":", lw=1,
                       color=ax.lines[-1].get_color())
        ax.axhline(0.5, color="k", lw=1)
        ax.set_ylim(0.35, 1.02)
        ax.set_xlabel("layer"); ax.set_ylabel("harm-vs-control accuracy")
        ax.set_title(tc.pretty(m), fontsize=9)
        ax.legend(fontsize=7)
    ax = axes[-1]
    m0 = models[0]
    M = np.full((3, 3), np.nan)
    for r in xfer:
        if r["model"] != m0:
            continue
        M[DOMAINS.index(r["fit_on"]), DOMAINS.index(r["test_on"])] = float(r["acc"])
    im = ax.imshow(M, vmin=0.4, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(3)); ax.set_xticklabels(DOMAINS, rotation=30, ha="right", fontsize=7)
    ax.set_yticks(range(3)); ax.set_yticklabels(DOMAINS, fontsize=7)
    ax.set_xlabel("test on"); ax.set_ylabel("fit on")
    ax.set_title(f"cross-domain transfer\n{tc.pretty(m0)}", fontsize=9)
    for i in range(3):
        for j in range(3):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        color="w", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("W7 (appendix): harm decoding across three domains of suffering "
                 "— dotted lines are TF-IDF baselines. Not an intent × outcome "
                 "factorial: outcome/harm selectivity only.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=170)
    print(f"  wrote {os.path.relpath(FIG, tc.ROOT)}")


def report(rows):
    recs = list(csv.DictReader(open(RES_CSV))) if os.path.exists(RES_CSV) else []
    xfer = list(csv.DictReader(open(XFER_CSV))) if os.path.exists(XFER_CSV) else []
    inter_p = os.path.join(OUT_DIR, "w7_bruneau_interaction.csv")
    inter = list(csv.DictReader(open(inter_p))) if os.path.exists(inter_p) else []
    if not recs:
        print("no results yet"); return
    L = ["# W7 (appendix) — Bruneau, Pluta & Saxe (2011): is the harm code "
         "domain-selective?", "",
         "**Limitation, first because it governs every number below.** These stimuli are "
         "not an intent × outcome factorial. No condition has a character intending harm "
         "that does not occur. PP/EP hold belief fixed and vary the outcome; FBP/FBC vary "
         "belief content but belief and outcome move together. W7 therefore tests "
         "**outcome/harm selectivity only** and says nothing about intent-weighting. It "
         "is support for the claim that the harm representation is rich and structured — "
         "which sharpens, but does not establish, the asymmetry the main results rest on.",
         "",
         f"Stimuli: {len(rows)} items, {len({r['pair'] for r in rows})} matched pairs, "
         "parsed from the published stimulus PDF (`W7_PARSE_REPORT.md`). Folds and "
         "bootstrap resamples are over item PAIRS, so the harmful and harmless version of "
         "one story never land on opposite sides of a split.", "",
         "## Within-domain harm decoding vs surface baseline", "",
         "| model | domain | layer | probe | TF-IDF | gap | pairs |",
         "|---|---|---:|---:|---:|---:|---:|"]
    for r in recs:
        L.append(f"| {tc.pretty(r['model'])} | {DOMAIN_LABEL[r['domain']]} "
                 f"| {r['layer']}/{r['n_layers']} | {float(r['probe_acc']):.3f} "
                 f"| {float(r['tfidf_acc']):.3f} | {float(r['gap']):+.3f} "
                 f"| {r['n_pairs']} |")
    if inter:
        L += ["", "## Moral-vs-non-moral interaction", "",
              "Difference in harm-decoding accuracy between a mentalizing domain and the "
              "physical domain, bootstrapped over item pairs (95% CI). A CI excluding 0 "
              "means the harm code is not equally available across domains. The TF-IDF "
              "column carries the same difference for the surface baseline: if the probe "
              "difference tracks the lexical difference, the effect is in the wording.", "",
              "| model | domain | layer | acc (domain) | acc (physical) | Δ probe [95% CI] "
              "| Δ TF-IDF |", "|---|---|---:|---:|---:|---:|---:|"]
        for r in inter:
            L.append(f"| {tc.pretty(r['model'])} | {DOMAIN_LABEL[r['domain']]} "
                     f"| {r['layer']} | {float(r['acc_domain']):.3f} "
                     f"| {float(r['acc_ref']):.3f} | {float(r['d_acc']):+.3f} "
                     f"[{float(r['lo']):+.3f}, {float(r['hi']):+.3f}] "
                     f"| {float(r['d_tfidf']):+.3f} |")
    if xfer:
        L += ["", "## Cross-domain transfer (the selectivity test)", "",
              "Fit the harm contrast on one domain, score it on another, at a single layer "
              "chosen by mean accuracy across domains (not per contrast, which would "
              "inflate the diagonal). High off-diagonal accuracy means one generic "
              "\"something bad happened\" code; near-chance off-diagonal means "
              "domain-specific harm codes. One asymmetry to keep in mind: the diagonal is "
              "grouped-CV (fit on ~5/6 of the pairs), the off-diagonal is fit on all 48 "
              "items of the source domain and tested on a fully disjoint set. The transfer "
              "cells therefore have MORE training data, so an off-diagonal that still "
              "falls below the diagonal is evidence for selectivity and not an artefact of "
              "sample size; an off-diagonal that matches the diagonal is the weaker "
              "comparison.", ""]
        for m in sorted({r["model"] for r in xfer}):
            rs = [r for r in xfer if r["model"] == m]
            L += [f"**{tc.pretty(m)}** (layer {rs[0]['layer']})", "",
                  "| fit on \\ test on | " + " | ".join(DOMAINS) + " |",
                  "|---|" + "---:|" * 3]
            for a in DOMAINS:
                cells = []
                for b in DOMAINS:
                    v = [float(r["acc"]) for r in rs if r["fit_on"] == a and r["test_on"] == b]
                    cells.append(f"{v[0]:.3f}" + (" _(within)_" if a == b and v else "")
                                 if v else "—")
                L.append(f"| {a} | " + " | ".join(cells) + " |")
            off = [float(r["acc"]) for r in rs if r["fit_on"] != r["test_on"]]
            wit = [float(r["acc"]) for r in rs if r["fit_on"] == r["test_on"]]
            L += ["", f"Mean within-domain {np.mean(wit):.3f}, mean cross-domain "
                  f"{np.mean(off):.3f} (chance 0.500). "
                  + ("Transfer is substantial: one largely shared harm code."
                     if np.mean(off) > 0.7 else
                     "Transfer is weak: the harm codes are domain-specific.")
                  if wit and off else "", ""]
    L += ["", "## Reading", "",
          "Whatever the transfer matrix shows, it is a statement about the harm/outcome "
          "representation only. The main results say the outcome code is what the rating "
          "uses and the intent code is not; W7 adds resolution to the first half of that "
          "sentence. A reviewer should not be allowed to read it as evidence on the "
          "second half, and no figure or caption from this script claims otherwise.", ""]
    with open(MD_OUT, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"  wrote {os.path.relpath(MD_OUT, tc.ROOT)}")
    plot(recs, xfer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parse", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--layer-frac", type=float, default=0.7,
                    help="extra layer to score, as a fraction of depth (peak-intent band)")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.parse:
        parse_pdf()
    rows = load_stimuli()
    if a.run:
        os.makedirs(OUT_DIR, exist_ok=True)
        for m in a.models:
            print(f"\n=== {m} ===", flush=True)
            try:
                recs, xfer, inter = run_model(m, rows, a.layer_frac, a.boot, a.seed,
                                              a.dtype)
            except Exception as e:
                print(f"!! FAILED {m}: {e}")
                import traceback; traceback.print_exc()
                continue
            write_csv(RES_CSV, recs)
            write_csv(XFER_CSV, xfer)
            write_csv(os.path.join(OUT_DIR, "w7_bruneau_interaction.csv"), inter)
            print(f"  wrote probe/transfer/interaction CSVs for {m}")
    if a.report or a.run:
        report(rows)
    if not (a.parse or a.run or a.report):
        print(__doc__)


if __name__ == "__main__":
    main()
