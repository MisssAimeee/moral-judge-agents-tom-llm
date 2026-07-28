#!/usr/bin/env python3
"""W3 -- causal steering of the intent direction. [GPU-L]

WHY THIS EXISTS. Everything reported so far is correlational: intent is linearly
decodable from the residual stream (probes at 0.90-0.98), and the same models weight
outcome far more than intent behaviourally. Three independent nulls (RSA convergence,
item-level, model-level) say the readable intent representation is not being used. All of
that is consistent with "intent is represented but unused" -- and equally consistent with
"the probe reads an epiphenomenal correlate". Only an intervention separates them. This is
the one remaining experiment that upgrades a correlation into a causal claim.

WHY IT REPLACES 17_steering.py. That script (a) computed digit token ids with
`tok.encode(str(d))[0]`, the exact bug that silently collapsed every rating to the scale
midpoint on SentencePiece tokenizers, (b) keyed the contrast on `scenario_of`, which treats
the 24 YS2009 reprints as independent and inflates effective n from 53 to 77, (c) fitted
the direction only by difference-of-means, with no probe-weight comparison, and (d) checked
coherence with perplexity on a single sentence and no refusal or degeneracy check. This
script fixes all four and adds the dose-response sweep.

=============================== PRE-SPECIFIED PREDICTIONS ==============================
Written before any steering run. Recorded verbatim to outputs/experiments/W3_PRESPEC.md on
first --run, and never overwritten, so results cannot be retrofitted to the hypothesis.

  P1 (direction specificity). Adding the INTENT direction raises the contrast
      (attempted - accidental) relative to alpha=0. The OUTCOME direction and the
      matched-norm RANDOM directions do not.
  P2 (dose-response). Within the coherent alpha range, contrast change is monotone in
      alpha for the intent direction, and the sign flips with the sign of alpha.
  P3 (method agreement). The difference-of-means intent direction and the probe-weight
      intent direction produce the same qualitative effect. If they disagree, the effect
      is a property of one estimator, not of the representation, and P1 is not supported.
  P4 (coherence). Any contrast change claimed under P1-P3 occurs at an alpha where the
      model is still coherent, defined in advance as: perplexity ratio <= 1.5x baseline,
      refusal-rate increase <= 0.10 absolute, and degenerate-generation fraction <= 0.10.
      Effects appearing only outside that range are reported as steering damage, not as
      causal evidence about intent.

  FALSIFICATION. P1 fails if the intent direction moves the contrast no more than the
  controls do, or if it moves it only where P4's coherence bound is violated. That outcome
  is reportable: it would mean the decodable intent direction is not causally wired to the
  moral judgment, strengthening the "represented but unused" reading rather than weakening
  the paper.

  NON-TRIVIALITY. A purely uniform additive shift would raise blame in all four cells
  together and leave their DIFFERENCE unchanged. A contrast change therefore requires
  differential movement across cells, so all four cell means are recorded at every alpha.
========================================================================================

Method
  1. Steer at the peak INTENT layer, read from outputs/probe/<tag>_probe.csv (`last`
     pooling, to match the last-token residual the directions are fitted on).
  2. Fit each direction two ways at that layer, on last-token residuals of the scoring
     prompt: difference-of-means (guilty - innocent) and probe weights (logistic
     regression, coefficients mapped back to raw activation space).
  3. Controls: the same two estimators for the OUTCOME contrast, plus N matched-norm
     random directions (all directions are unit-normalised and scaled by the same typical
     residual norm, so every intervention has identical magnitude).
  4. Sweep alpha for dose-response; rescore the contrast with the FIXED logprob-EV digit
     scoring from 03_behavioral.HFBackend.
  5. Coherence at every alpha: perplexity on held-out text, refusal rate, degeneracy, plus
     a dump of ~20 generations per level to outputs/experiments/w3_generations_<tag>.txt
     for manual reading.

Modes
  (default) --dry-run : print the plan and resolved layer; downloads nothing.
  --run               : load the model, fit directions, sweep, write outputs.

Outputs (with --run)
  outputs/experiments/W3_PRESPEC.md                  (written once, never overwritten)
  outputs/experiments/w3_steering_<tag>.csv          (per direction x alpha)
  outputs/experiments/w3_steering_directions_<tag>.csv  (cosines, probe acc, norms)
  outputs/experiments/w3_generations_<tag>.txt       (manual-read dump)
  outputs/experiments/w3_steering_<tag>.png          (dose-response, coherent band shaded)
"""
import argparse
import csv
import glob
import importlib.util
import os
import re
import sys
from collections import defaultdict

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CODE_DIR)
import numpy as np  # noqa: E402
import tom_common as tc  # noqa: E402

MASTER_CSV = os.path.join(tc.ROOT, "dataset", "master", "moral_2x2_master.csv")
PROBE_DIR = os.path.join(tc.ROOT, "outputs", "probe")
OUT_DIR = os.path.join(tc.ROOT, "outputs", "experiments")
PRESPEC = os.path.join(OUT_DIR, "W3_PRESPEC.md")

GUILTY = {"attempted", "intentional"}     # intent present
HARM = {"accidental", "intentional"}      # bad outcome present
CELLS = ["neutral", "accidental", "attempted", "intentional"]

# Held-out text for the coherence check: unrelated to moral content, so a perplexity rise
# reflects damage to general language modelling rather than to the task representation.
PPL_TEXTS = [
    "The teacher explained the lesson to the class and then asked the students to work "
    "quietly on their assignments.",
    "Water boils at one hundred degrees Celsius at sea level, and the boiling point falls "
    "as altitude increases.",
    "She parked the car outside the library, walked up the steps, and returned the books "
    "she had borrowed last month.",
]
REFUSAL_MARKERS = (
    "i can't", "i cannot", "i can not", "i won't", "i will not", "i'm sorry",
    "i am sorry", "i apologize", "as an ai", "i'm unable", "i am unable",
    "i'm not able", "i am not able", "cannot provide", "can't provide",
    "not appropriate", "i must decline",
)

# Pre-specified coherence bounds (P4). Changing these after seeing results would invalidate
# the pre-registration, so they live here as constants rather than as CLI flags.
PPL_RATIO_MAX = 1.5
REFUSAL_DELTA_MAX = 0.10
DEGENERATE_FRAC_MAX = 0.10
# Task compliance: a model that stops emitting a rating digit is not usable for the
# contrast measurement regardless of how fluent it still is. Added in the 2026-07-28
# amendment (see W3_PRESPEC.md) together with the degeneracy-detector fix.
ANSWER_DROP_MAX = 0.10


def _load_behavioral():
    spec = importlib.util.spec_from_file_location(
        "behavioral", os.path.join(CODE_DIR, "03_behavioral.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class _DigitShim:
    """Lets us call HFBackend._digit_token_ids without loading a second copy of the model.

    The fixed digit mapping is the whole point of reusing it -- reimplementing it here is
    how the collapse bug would come back.
    """

    def __init__(self, tok, model_name):
        self.tok = tok
        self.model_name = model_name


def load_dataset(limit=None):
    with open(MASTER_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit:
        # keep whole scenario groups so the attempted/accidental pairing survives
        keep, seen = [], set()
        for r in rows:
            g = tc.scenario_group_of(r["story_id"])
            if g in seen or len(seen) < limit:
                seen.add(g)
                keep.append(r)
        rows = keep
    return rows


def peak_intent_layer(tag, pooling="last"):
    """Peak intent layer from the probe CSV for the pooling we fit directions on."""
    suffix = "_probe.csv" if pooling == "last" else f"_probe_{pooling}.csv"
    cands = [p for p in glob.glob(os.path.join(PROBE_DIR, f"*{tag}*{suffix}"))
             if "_src" not in os.path.basename(p)]
    if not cands:
        return None, f"no probe CSV matching *{tag}*{suffix}"
    best = None
    for r in csv.DictReader(open(cands[0])):
        if r["target"] != "intent":
            continue
        acc = float(r["cv_acc"])
        if best is None or acc > best[1]:
            best = (int(r["layer"]), acc)
    if best is None:
        return None, "probe CSV has no intent rows"
    return best[0], (f"probe peak in {os.path.basename(cands[0])} "
                     f"(pooling={pooling}, cv_acc={best[1]:.3f})")


AMENDMENT = """
## Amendment, 2026-07-28 — coherence instrument fix

The first run (job 19092342) produced NO usable verdict on P1-P3 because P4 gated out every
non-zero alpha, including alpha values where the model was demonstrably fine. That was a
bug in the coherence instrument, not a property of the model, and it was fixed before any
P1-P3 result was accepted or reported. Three changes, none of them to a threshold:

1. **Degeneracy detector.** It flagged any generation shorter than three tokens as
   degenerate. The rating prompt correctly answers with a single digit ("3"), so 100% of
   BASELINE generations were labelled degenerate. Degeneracy is now empty output, or heavy
   repetition (unique-token ratio < 0.35) evaluated only on outputs of at least 8 tokens.
2. **Coherence is now measured on prose.** Refusal and repetition cannot be assessed on a
   single digit, and "manual read of 20 outputs per level" is meaningless when every output
   is one character. Each level now also generates a one-to-two sentence explanation per
   story, which is what the refusal, degeneracy and manual-read checks run on.
3. **Symmetric measurement.** The first version generated only for the intent directions at
   mid alphas, so the control directions passed coherence by not being tested -- a weaker
   standard for the controls than for the direction under test, which is backwards for a
   specificity comparison. Every direction is now measured identically at every alpha.

One criterion was ADDED: task compliance (`answer_rate`), the fraction of rating prompts
whose greedy generation still contains a scale digit, may not drop by more than 0.10. A
model that has stopped emitting ratings cannot support a contrast measurement whatever its
perplexity. The three original thresholds (perplexity ratio 1.5x, refusal delta 0.10,
degenerate fraction 0.10) are unchanged.

P1-P4 themselves are unchanged and were not informed by any result.
"""


def write_prespec():
    """Write the pre-registration once. Never overwrite: that is the point of it."""
    if os.path.exists(PRESPEC):
        cur = open(PRESPEC).read()
        if "## Amendment, 2026-07-28" not in cur:
            with open(PRESPEC, "a") as f:
                f.write("\n" + AMENDMENT.strip() + "\n")
            print(f"  appended amendment to {os.path.relpath(PRESPEC, tc.ROOT)}")
        else:
            print(f"  pre-spec already current, leaving untouched: "
                  f"{os.path.relpath(PRESPEC, tc.ROOT)}")
        return
    doc = __doc__.split("PRE-SPECIFIED PREDICTIONS", 1)[1]
    doc = doc.split("=====", 1)[1] if "=====" in doc else doc
    body = doc.split("Method", 1)[0].rstrip()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(PRESPEC, "w") as f:
        f.write("# W3 causal steering — pre-specified predictions\n\n")
        f.write("Written by `code/experiments/48_w3_causal_steering.py` at the start of "
                "the first `--run`, before any steering result existed. This file is never "
                "overwritten on later runs.\n\n")
        f.write("Coherence bounds (P4), fixed in the script as constants:\n\n")
        f.write(f"- perplexity ratio <= {PPL_RATIO_MAX}x baseline\n")
        f.write(f"- refusal-rate increase <= {REFUSAL_DELTA_MAX:.2f} absolute\n")
        f.write(f"- degenerate-generation fraction <= {DEGENERATE_FRAC_MAX:.2f}\n\n")
        f.write("```\n" + body + "\n```\n")
    print(f"  wrote pre-spec {os.path.relpath(PRESPEC, tc.ROOT)}")


def fit_probe_direction(X, y, groups, seed=0):
    """Logistic-regression direction in RAW activation space, plus grouped CV accuracy.

    02_probe standardises and then row-space projects before fitting; the projection is a
    variance-reduction step for accuracy estimation and would leave the coefficients in a
    projected basis that cannot be added back to the residual stream. Here we standardise
    only, then undo the scaling so the returned vector lives in raw activation space.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed).fit(Xs, y)
    w = clf.coef_[0] / np.maximum(sc.scale_, 1e-8)

    accs = []
    n_splits = min(5, len(set(groups)))
    if n_splits >= 2:
        for tr, te in GroupKFold(n_splits=n_splits).split(Xs, y, groups):
            s = StandardScaler().fit(X[tr])
            c = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
            c.fit(s.transform(X[tr]), y[tr])
            accs.append(c.score(s.transform(X[te]), y[te]))
    return w, (float(np.mean(accs)) if accs else float("nan"))


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def calibrate_alphas(ppl_at, base_ppl, ladder):
    """Size the sweep grid to the model's coherent range, using perplexity ONLY.

    alpha is in units of the typical residual norm at the steering layer, so alpha=1 adds a
    vector as large as the entire residual and destroys the model -- a 0.5B smoke test hit
    perplexity ratios of 2.2-5.5 at alpha=1. A fixed grid would therefore have produced no
    coherent points at all on some models and a wastefully truncated one on others.

    The grid is chosen from perplexity alone, which is independent of the contrast the
    predictions are about, so this does not weaken the pre-registration: it cannot select
    for or against P1-P3. The largest coherent ladder rung becomes a_max; the sweep spans
    fractions of it, plus one deliberately-past-the-bound rung so the writeup can show
    where steering starts breaking the model rather than only asserting that it does.
    """
    a_max, trace = None, []
    for al in sorted(ladder):
        ratio = ppl_at(al) / base_ppl
        trace.append((al, ratio))
        print(f"    calib alpha={al:+.3f}  ppl_ratio={ratio:.2f}"
              f"{'  <= bound' if ratio <= PPL_RATIO_MAX else '  OVER bound'}")
        if ratio <= PPL_RATIO_MAX:
            a_max = al
        else:
            break
    if a_max is None:
        a_max = min(ladder)
        print(f"    no ladder rung met the perplexity bound; falling back to "
              f"a_max={a_max:+.3f} (expect an empty coherent band)")
    grid = sorted({round(f * a_max, 4) for f in (0.125, 0.25, 0.5, 0.75, 1.0, 1.5)})
    grid = sorted({0.0} | set(grid) | {-g for g in grid})
    return grid, a_max, trace


def replot(tag):
    """Rebuild the figure and readout from a finished results CSV, without loading a model."""
    path = os.path.join(OUT_DIR, f"w3_steering_{tag}.csv")
    if not os.path.exists(path):
        print(f"no results CSV at {path}")
        return
    results = []
    for r in csv.DictReader(open(path)):
        row = dict(r)
        for k, v in r.items():
            if k in ("direction",):
                continue
            if k == "coherent":
                row[k] = v == "True"
            else:
                row[k] = float(v) if v not in ("", None) else float("nan")
        results.append(row)
    meta = {}
    dpath = os.path.join(OUT_DIR, f"w3_steering_directions_{tag}.csv")
    if os.path.exists(dpath):
        for r in csv.reader(open(dpath)):
            if len(r) == 2 and r[0] not in ("metric", ""):
                meta[r[0]] = r[1]
    base = next(r for r in results if r["direction"] == "baseline")
    L = int(float(meta.get("layer", 0)))
    base_cells = {c: base.get(f"cell_{c}", float("nan")) for c in CELLS}
    _plot(results, tag, base["contrast"], L)
    _readout(results, tag, base["contrast"], base_cells, L,
             float(meta.get("cos_intent_dom_vs_probe", "nan")),
             float(meta.get("probe_cv_acc_intent", "nan")),
             float(meta.get("probe_cv_acc_outcome", "nan")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-2-1124-7B-Instruct")
    ap.add_argument("--template", default="human_verbatim")
    ap.add_argument("--layer", type=int, default=None,
                    help="steering layer (hidden_states index); default = probe peak")
    ap.add_argument("--alphas", type=float, nargs="+", default=None,
                    help="explicit alpha grid; default = calibrated from perplexity "
                         "(see calibrate_alphas)")
    ap.add_argument("--calibration-ladder", type=float, nargs="+",
                    default=[0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8],
                    help="alphas probed for coherence before the real sweep")
    ap.add_argument("--n-random", type=int, default=2,
                    help="matched-norm random control directions")
    ap.add_argument("--n-gen", type=int, default=20,
                    help="generations per level for refusal rate + manual read")
    ap.add_argument("--gen-tokens", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit-groups", type=int, default=None,
                    help="smoke test: keep only N scenario groups")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--plot-only", action="store_true",
                    help="regenerate figure + readout from an existing results CSV, no GPU")
    a = ap.parse_args()

    tag = a.model.split("/")[-1]
    L_probe, how = peak_intent_layer(tag)

    if a.plot_only:
        replot(tag)
        return

    if not a.run:
        print("\n=== W3 CAUSAL STEERING PLAN (dry-run: no weights downloaded) ===")
        print(f"  model         : {a.model}")
        print(f"  template      : {a.template}")
        print(f"  steer layer   : {a.layer if a.layer is not None else L_probe}  ({how})")
        print(f"  directions    : intent x {{diff-of-means, probe-weights}}, "
              f"outcome x {{diff-of-means, probe-weights}}, "
              f"{a.n_random} matched-norm random")
        print(f"  alpha sweep   : {a.alphas}   (x typical residual norm at the layer)")
        print(f"  metric        : contrast = blame(attempted) - blame(accidental), "
              f"keyed on scenario_group (53 groups)")
        print(f"  cells logged  : {CELLS}  (uniform shift vs differential movement)")
        print(f"  coherence     : ppl on {len(PPL_TEXTS)} held-out texts, refusal rate, "
              f"degeneracy, {a.n_gen} generations/level dumped for manual read")
        print(f"  coherent iff  : ppl_ratio <= {PPL_RATIO_MAX}, "
              f"refusal_delta <= {REFUSAL_DELTA_MAX}, "
              f"degenerate_frac <= {DEGENERATE_FRAC_MAX}")
        print("\n  PRE-SPECIFIED: intent direction raises contrast; outcome and random do "
              "not;\n  effect must be monotone in alpha and inside the coherent band; "
              "both direction\n  estimators must agree. See docstring P1-P4.")
        print("\nLaunch:")
        print(f"  sbatch code/submit_w3_steering.sh")
        return

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    beh = _load_behavioral()
    os.makedirs(OUT_DIR, exist_ok=True)
    write_prespec()
    rows = load_dataset(a.limit_groups)
    print(f"  {len(rows)} stories, "
          f"{len({tc.scenario_group_of(r['story_id']) for r in rows})} scenario groups")

    print(f"Loading {a.model} ...")
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"   # last position is the true last token for every row
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.float16, output_hidden_states=True,
        device_map="auto", trust_remote_code=True)
    model.eval()

    dig = beh.HFBackend._digit_token_ids(_DigitShim(tok, a.model))
    print(f"  digit tokens distinct: {len(set(dig.values()))} of {len(dig)}")

    n_layers = model.config.num_hidden_layers
    L = a.layer if a.layer is not None else L_probe
    if L is None:
        L = int(0.6 * n_layers)
        how = f"fallback 60% depth of {n_layers}"
    print(f"  {n_layers} layers; steering hidden_states[{L}] "
          f"(output of decoder block {L - 1}) -- {how}")

    def fmt(row):
        prompt, s_min, s_max = beh.build_prompt(row["text"], a.template, row["source"])
        text = (tok.apply_chat_template([{"role": "user", "content": prompt}],
                                        tokenize=False, add_generation_prompt=True)
                if tok.chat_template else prompt)
        return text, s_min, s_max

    prepared = [(r, ) + fmt(r) for r in rows]

    # ---------------- 1. residuals at L, then directions ----------------
    print(f"  collecting last-token residuals at L{L} ...")
    feats, labels_i, labels_o, groups, norms = [], [], [], [], []
    with torch.no_grad():
        for i in range(0, len(prepared), a.batch_size):
            chunk = prepared[i:i + a.batch_size]
            enc = tok([c[1] for c in chunk], return_tensors="pt",
                      padding=True).to(model.device)
            hs = model(**enc).hidden_states[L][:, -1, :].float().cpu().numpy()
            for (row, _t, _lo, _hi), h in zip(chunk, hs):
                feats.append(h)
                norms.append(float(np.linalg.norm(h)))
                labels_i.append(int(row["condition"] in GUILTY))
                labels_o.append(int(row["condition"] in HARM))
                groups.append(tc.scenario_group_of(row["story_id"]))
    X = np.array(feats)
    yi, yo = np.array(labels_i), np.array(labels_o)
    groups = np.array(groups)
    typ_norm = float(np.mean(norms))
    print(f"  typical residual norm @L{L} = {typ_norm:.1f}")

    def dom(y):
        return X[y == 1].mean(0) - X[y == 0].mean(0)

    probe_i, acc_i = fit_probe_direction(X, yi, groups)
    probe_o, acc_o = fit_probe_direction(X, yo, groups)
    print(f"  probe cv_acc at L{L}: intent={acc_i:.3f}  outcome={acc_o:.3f}")

    raw_dirs = {
        "intent_dom": dom(yi),
        "intent_probe": probe_i,
        "outcome_dom": dom(yo),
        "outcome_probe": probe_o,
    }
    rng = np.random.default_rng(0)
    for s in range(a.n_random):
        raw_dirs[f"random{s}"] = rng.standard_normal(X.shape[1])
    # Unit-normalise every direction, then scale all by the same typical residual norm, so
    # each intervention has identical magnitude and 'matched-norm random' is exactly that.
    dirs = {k: v / (np.linalg.norm(v) + 1e-8) for k, v in raw_dirs.items()}

    dinfo = [dict(direction=k, raw_norm=round(float(np.linalg.norm(v)), 4),
                  cos_with_intent_dom=round(cosine(v, raw_dirs["intent_dom"]), 4),
                  cos_with_outcome_dom=round(cosine(v, raw_dirs["outcome_dom"]), 4),
                  probe_cv_acc=(round(acc_i, 4) if k == "intent_probe" else
                                round(acc_o, 4) if k == "outcome_probe" else ""))
             for k, v in raw_dirs.items()]
    cos_methods_i = cosine(raw_dirs["intent_dom"], raw_dirs["intent_probe"])
    cos_methods_o = cosine(raw_dirs["outcome_dom"], raw_dirs["outcome_probe"])
    cos_io = cosine(raw_dirs["intent_dom"], raw_dirs["outcome_dom"])
    print(f"  cos(dom, probe): intent={cos_methods_i:+.3f} outcome={cos_methods_o:+.3f}; "
          f"cos(intent_dom, outcome_dom)={cos_io:+.3f}")

    with open(os.path.join(OUT_DIR, f"w3_steering_directions_{tag}.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dinfo[0].keys()))
        w.writeheader()
        w.writerows(dinfo)
        w.writerow({})
        w2 = csv.DictWriter(f, fieldnames=["metric", "value"])
        w2.writeheader()
        for k, v in (("layer", L), ("typical_residual_norm", round(typ_norm, 2)),
                     ("cos_intent_dom_vs_probe", round(cos_methods_i, 4)),
                     ("cos_outcome_dom_vs_probe", round(cos_methods_o, 4)),
                     ("cos_intent_vs_outcome_dom", round(cos_io, 4)),
                     ("probe_cv_acc_intent", round(acc_i, 4)),
                     ("probe_cv_acc_outcome", round(acc_o, 4))):
            w2.writerow({"metric": k, "value": v})

    # ---------------- steering hook ----------------
    state = {"vec": None}
    blocks = model.model.layers

    def hook(_m, _inp, out):
        if state["vec"] is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        h = h + state["vec"].to(h.dtype)
        return ((h,) + tuple(out[1:])) if isinstance(out, tuple) else h

    blocks[max(0, L - 1)].register_forward_hook(hook)

    def set_vec(vec):
        state["vec"] = None if vec is None else torch.tensor(
            vec, device=model.device, dtype=torch.float32)

    # ---------------- measurement ----------------
    def score_cells():
        by_group = defaultdict(dict)
        with torch.no_grad():
            for i in range(0, len(prepared), a.batch_size):
                chunk = prepared[i:i + a.batch_size]
                enc = tok([c[1] for c in chunk], return_tensors="pt",
                          padding=True).to(model.device)
                logits = model(**enc).logits[:, -1, :].float()
                for (row, _t, s_min, s_max), lg in zip(chunk, logits):
                    vals = [d for d in range(int(s_min), int(s_max) + 1) if d in dig]
                    lp = torch.tensor([lg[dig[d]].item() for d in vals])
                    p = torch.softmax(lp, 0).tolist()
                    ev = sum(p[j] * vals[j] for j in range(len(vals)))
                    by_group[tc.scenario_group_of(row["story_id"])][row["condition"]] = \
                        (ev - s_min) / (s_max - s_min)
        cells = {c: float(np.mean([g[c] for g in by_group.values() if c in g]))
                 for c in CELLS if any(c in g for g in by_group.values())}
        diffs = [g["attempted"] - g["accidental"] for g in by_group.values()
                 if "attempted" in g and "accidental" in g]
        return (float(np.mean(diffs)) if diffs else float("nan")), cells, len(diffs)

    def perplexity():
        vals = []
        with torch.no_grad():
            for t in PPL_TEXTS:
                enc = tok(t, return_tensors="pt").to(model.device)
                out = model(**enc, labels=enc["input_ids"])
                vals.append(float(torch.exp(out.loss).item()))
        return float(np.mean(vals))

    # Two generation sets, because the rating prompt answers with a single digit ("3") and
    # single digits can neither be read for coherence nor distinguished from degeneration by
    # any text statistic. PROSE prompts ask for an explanation, so refusal, repetition and
    # manual reading all have something to work on; RATING prompts measure task compliance.
    prose_prompts, rating_prompts = [], []
    for row, text, _lo, _hi in prepared[:a.n_gen]:
        q = ("Briefly describe what happened in this story and say how blameworthy the "
             "person is, in one or two sentences.")
        p = row["text"].strip() + "\n\n" + q
        prose_prompts.append(
            tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                    add_generation_prompt=True)
            if tok.chat_template else p)
        rating_prompts.append(text)

    def generate(prompts, max_new):
        outs = []
        with torch.no_grad():
            for i in range(0, len(prompts), 8):
                enc = tok(prompts[i:i + 8], return_tensors="pt",
                          padding=True).to(model.device)
                g = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
                for j in range(g.shape[0]):
                    new = g[j, enc["input_ids"].shape[1]:]
                    outs.append(tok.decode(new, skip_special_tokens=True).strip())
        return outs

    def coherence_stats(prose, ratings):
        """Refusal, degeneracy (prose) and task compliance (ratings).

        Degeneracy is empty output or heavy repetition. The repetition rule only applies to
        outputs long enough for the ratio to mean anything -- an earlier version flagged any
        output under three tokens, which labelled every correct single-digit rating as
        degenerate and gated out the whole sweep.
        """
        refusals = sum(any(m in t.lower() for m in REFUSAL_MARKERS) for t in prose)
        degen = 0
        for t in prose:
            toks = t.split()
            if not toks:
                degen += 1
            elif len(toks) >= 8 and len(set(toks)) / len(toks) < 0.35:
                degen += 1
        answered = sum(1 for t in ratings if re.search(r"[1-9]", t))
        return (refusals / max(len(prose), 1), degen / max(len(prose), 1),
                answered / max(len(ratings), 1))

    gen_log = open(os.path.join(OUT_DIR, f"w3_generations_{tag}.txt"), "w")
    gen_log.write(f"W3 steering generations — {a.model}, layer {L}, "
                  f"template {a.template}\nGreedy decoding, {a.gen_tokens} new tokens, "
                  f"{a.n_gen} prompts per level. Each entry shows the free-text "
                  f"explanation and, in brackets, the rating the same prompt elicits.\n"
                  f"Read these to confirm the model is still answering the question "
                  f"rather than degrading.\n")

    def dump(label, prose, ratings):
        gen_log.write(f"\n{'=' * 78}\n### {label}\n{'=' * 78}\n")
        for k, (p, r) in enumerate(zip(prose, ratings)):
            gen_log.write(f"\n[{k:02d}] (rating: {r!r})\n{p}\n")
        gen_log.flush()

    # ---------------- baseline ----------------
    set_vec(None)
    base_c, base_cells, n_pairs = score_cells()
    base_ppl = perplexity()
    base_prose = generate(prose_prompts, a.gen_tokens)
    base_rat = generate(rating_prompts, 4)
    base_ref, base_deg, base_ans = coherence_stats(base_prose, base_rat)
    dump("BASELINE (alpha=0, no steering)", base_prose, base_rat)
    print(f"  baseline: contrast={base_c:+.4f} (n_pairs={n_pairs}) ppl={base_ppl:.2f} "
          f"refusal={base_ref:.2f} degen={base_deg:.2f} answer_rate={base_ans:.2f}")
    print(f"    cells: " + "  ".join(f"{c}={base_cells.get(c, float('nan')):.3f}"
                                     for c in CELLS))

    results = [dict(direction="baseline", alpha=0.0, contrast=round(base_c, 4),
                    dcontrast=0.0, ppl=round(base_ppl, 3), ppl_ratio=1.0,
                    refusal_rate=round(base_ref, 3), refusal_delta=0.0,
                    degenerate_frac=round(base_deg, 3),
                    answer_rate=round(base_ans, 3), answer_drop=0.0, coherent=True,
                    **{f"cell_{c}": round(base_cells.get(c, float("nan")), 4)
                       for c in CELLS})]

    # ---------------- calibrate the alpha grid (perplexity only) ----------------
    if a.alphas is not None:
        alphas, a_max = sorted(a.alphas), max(abs(x) for x in a.alphas)
        print(f"  using explicit alpha grid: {alphas}")
    else:
        print("  calibrating alpha grid on perplexity (intent_dom direction) ...")

        def ppl_at(al):
            set_vec((al * typ_norm) * dirs["intent_dom"])
            v = perplexity()
            set_vec(None)
            return v

        alphas, a_max, calib_trace = calibrate_alphas(ppl_at, base_ppl,
                                                      a.calibration_ladder)
        with open(os.path.join(OUT_DIR, f"w3_calibration_{tag}.csv"), "w",
                  newline="") as f:
            w = csv.writer(f)
            w.writerow(["alpha", "ppl_ratio", "within_bound"])
            for al, ratio in calib_trace:
                w.writerow([al, round(ratio, 4), ratio <= PPL_RATIO_MAX])
        print(f"  a_max={a_max:+.4f} (largest rung within {PPL_RATIO_MAX}x ppl); "
              f"sweep grid: {alphas}")

    # ---------------- sweep ----------------
    # Every direction gets the identical coherence measurement at every alpha. An earlier
    # version generated only for the intent directions at mid alphas to save time, which
    # held the controls to a weaker standard than the direction under test -- backwards for
    # a specificity comparison.
    for dname, d in dirs.items():
        for al in alphas:
            if al == 0.0:
                continue
            set_vec((al * typ_norm) * d)
            c, cells, _ = score_cells()
            ppl = perplexity()
            prose = generate(prose_prompts, a.gen_tokens)
            rat = generate(rating_prompts, 4)
            ref, deg, ans = coherence_stats(prose, rat)
            dump(f"{dname}  alpha={al:+.4f}", prose, rat)
            ratio = ppl / base_ppl if base_ppl else float("nan")
            coherent = bool(ratio <= PPL_RATIO_MAX
                            and (ref - base_ref) <= REFUSAL_DELTA_MAX
                            and deg <= DEGENERATE_FRAC_MAX
                            and (base_ans - ans) <= ANSWER_DROP_MAX)
            results.append(dict(
                direction=dname, alpha=al, contrast=round(c, 4),
                dcontrast=round(c - base_c, 4), ppl=round(ppl, 3),
                ppl_ratio=round(ratio, 3), refusal_rate=round(ref, 3),
                refusal_delta=round(ref - base_ref, 3),
                degenerate_frac=round(deg, 3), answer_rate=round(ans, 3),
                answer_drop=round(base_ans - ans, 3), coherent=coherent,
                **{f"cell_{k}": round(cells.get(k, float("nan")), 4) for k in CELLS}))
            print(f"    {dname:14} a={al:+7.4f}  contrast={c:+.4f} "
                  f"(d={c - base_c:+.4f})  ppl_ratio={ratio:.2f} deg={deg:.2f} "
                  f"ans={ans:.2f}  coherent={'Y' if coherent else 'N'}")
    gen_log.close()
    set_vec(None)

    out_csv = os.path.join(OUT_DIR, f"w3_steering_{tag}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nwrote {os.path.relpath(out_csv, tc.ROOT)}")

    # P1-P3 are claims about intent steering, so the headline band is the range where the
    # INTENT interventions stay coherent. Control ranges are reported separately.
    coh_alphas = [r["alpha"] for r in results
                  if r["coherent"] and r["alpha"] != 0.0
                  and r["direction"].startswith("intent")]
    if coh_alphas:
        print(f"  coherent alpha range (intent directions): "
              f"[{min(coh_alphas):+.4f}, {max(coh_alphas):+.4f}] "
              f"— confine all claims to this band")
    else:
        print("  NO non-zero alpha met the coherence bounds — no causal claim supportable")
    _plot(results, tag, base_c, L)
    _readout(results, tag, base_c, base_cells, L, cos_methods_i, acc_i, acc_o)


def _plot(results, tag, base_c, L):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (skip figure: {e})")
        return
    styles = {"intent_dom": ("#1f3f8f", "o", "-"), "intent_probe": ("#4a7fd4", "s", "-"),
              "outcome_dom": ("#c45c26", "^", "--"), "outcome_probe": ("#e0a06a", "v", "--"),
              }
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    # Shade only the alphas coherent for EVERY direction. Shading the union would put
    # individually-incoherent points inside the "coherent" band; incoherent points are also
    # drawn hollow so no reader has to infer coherence from the shading alone.
    dirs_all = sorted({r["direction"] for r in results} - {"baseline"})
    per_alpha = defaultdict(list)
    for r in results:
        if r["alpha"] != 0.0:
            per_alpha[r["alpha"]].append(r["coherent"])
    coh = [al for al, flags in per_alpha.items() if all(flags)]
    if coh:
        ax.axvspan(min(coh), max(coh), color="#eaf2ea", zorder=0,
                   label=f"coherent for all directions [{min(coh):+.3g}, {max(coh):+.3g}]")
    for dname in dirs_all:
        pts = sorted([r for r in results if r["direction"] == dname],
                     key=lambda r: r["alpha"])
        if not pts:
            continue
        col, mk, ls = styles.get(dname, ("#999999", ".", ":"))
        lbl = dname if not dname.startswith("random") else None
        ax.plot([p["alpha"] for p in pts], [p["contrast"] for p in pts],
                ls=ls, color=col, label=lbl, lw=1.5, zorder=3)
        for p in pts:
            ax.plot(p["alpha"], p["contrast"], marker=mk, color=col, ms=5.0,
                    mfc=col if p["coherent"] else "white", mew=1.2, zorder=4)
    rnd = [r for r in results if r["direction"].startswith("random")]
    if rnd:
        ax.plot([], [], marker=".", ls=":", color="#999999",
                label=f"random x{len({r['direction'] for r in rnd})} (matched norm)")
    ax.plot([], [], marker="o", ls="none", color="#555", mfc="white", mew=1.2,
            label="hollow = fails coherence bound")
    ax.axhline(base_c, color="#555", ls="-", lw=0.9, zorder=2)
    ax.annotate(f"unsteered = {base_c:+.3f}", xy=(0.02, base_c), xycoords=("axes fraction",
                "data"), fontsize=8, color="#555", va="bottom")
    ax.axvline(0, color="#bbb", lw=0.8, zorder=1)
    def max_d(pref):
        vals = [abs(r["dcontrast"]) for r in results if r["coherent"]
                and r["alpha"] != 0.0 and r["direction"].startswith(pref)]
        return max(vals, default=float("nan"))

    ax.set_xlabel("steering coefficient α  (× typical residual norm at the layer)")
    ax.set_ylabel("contrast  (attempted − accidental)")
    ax.set_title(
        f"W3 causal steering at L{L}: {tag}\n"
        f"Pre-specified: intent raises the contrast, controls do not. "
        f"Result: NOT SUPPORTED —\nintent from probe weights moves it "
        f"{max_d('intent_probe'):.3f}; the outcome control moves it "
        f"{max_d('outcome'):.3f}", fontsize=10.2)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"w3_steering_{tag}.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"wrote {os.path.relpath(out, tc.ROOT)}")


def _readout(results, tag, base_c, base_cells, L, cos_i, acc_i, acc_o):
    """Evaluate P1-P4 mechanically so the writeup cannot drift from the numbers."""
    def rows_for(d, coherent_only=True):
        return sorted([r for r in results if r["direction"] == d
                       and r["alpha"] != 0.0
                       and (r["coherent"] or not coherent_only)],
                      key=lambda r: r["alpha"])

    def max_abs_d(d):
        rs = rows_for(d)
        return max((abs(r["dcontrast"]) for r in rs), default=float("nan"))

    lines = [f"# W3 causal steering readout — {tag}", "",
             f"Steering layer L{L}; unsteered contrast {base_c:+.4f}; "
             f"probe cv_acc at L{L}: intent {acc_i:.3f}, outcome {acc_o:.3f}; "
             f"cos(dom, probe) for intent {cos_i:+.3f}.", "",
             "Baseline cells: " + ", ".join(f"{c} {base_cells.get(c, float('nan')):.3f}"
                                            for c in CELLS), "",
             "## Effect sizes inside the coherent band", "",
             "| direction | max |Δcontrast| | alphas tested (coherent) |",
             "|---|---:|---|"]
    for d in sorted({r["direction"] for r in results} - {"baseline"}):
        rs = rows_for(d)
        alphas = ", ".join(f"{r['alpha']:+g}" for r in rs) or "none"
        lines.append(f"| {d} | {max_abs_d(d):.4f} | {alphas} |")
    ctrl = max([max_abs_d(d) for d in {r["direction"] for r in results}
                if d.startswith(("outcome", "random"))], default=float("nan"))
    int_d = max([max_abs_d(d) for d in {r["direction"] for r in results}
                 if d.startswith("intent")], default=float("nan"))
    p1 = int_d > ctrl
    lines += ["", "## Pre-specified predictions", "",
              f"- **P1 direction specificity**: intent max |Δ| = {int_d:.4f} vs control "
              f"max |Δ| = {ctrl:.4f} → {'SUPPORTED' if p1 else 'NOT SUPPORTED'}"]
    for d in ("intent_dom", "intent_probe"):
        rs = rows_for(d)
        if len(rs) >= 3:
            xs = [r["alpha"] for r in rs]
            ys = [r["dcontrast"] for r in rs]
            slope = float(np.polyfit(xs, ys, 1)[0]) if len(xs) > 1 else float("nan")
            mono = all(ys[i] <= ys[i + 1] for i in range(len(ys) - 1)) or \
                all(ys[i] >= ys[i + 1] for i in range(len(ys) - 1))
            lines.append(f"- **P2 dose-response ({d})**: slope {slope:+.4f} per unit α, "
                         f"monotone={'yes' if mono else 'no'}")
    lines.append(f"- **P3 method agreement**: cos(intent_dom, intent_probe) = {cos_i:+.3f}; "
                 f"max |Δ| dom {max_abs_d('intent_dom'):.4f} vs probe "
                 f"{max_abs_d('intent_probe'):.4f}")
    coh = [r["alpha"] for r in results if r["coherent"] and r["alpha"] != 0.0
           and r["direction"].startswith("intent")]
    lines.append(f"- **P4 coherence (intent directions)**: coherent band "
                 f"[{min(coh):+.4f}, {max(coh):+.4f}] (α in units of the typical residual "
                 f"norm); all effect sizes above are computed inside it. "
                 f"Manual read: `w3_generations_{tag}.txt`."
                 if coh else "- **P4 coherence**: NO non-zero α met the coherence bounds; "
                 "no causal claim is supportable from this run.")
    lines += ["", "Generated by `code/experiments/48_w3_causal_steering.py`. "
              "Pre-registration: `W3_PRESPEC.md`.", ""]
    p = os.path.join(OUT_DIR, f"W3_STEERING_{tag}.md")
    open(p, "w").write("\n".join(lines))
    print(f"wrote {os.path.relpath(p, tc.ROOT)}")


if __name__ == "__main__":
    main()
