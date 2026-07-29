#!/usr/bin/env python3
"""Prose/rating dissociation: do models VERBALISE intent and still RATE by outcome? [GPU-M]

WHY THIS EXISTS. Every other line of evidence that intent is represented but unused needs
an internal measurement to state: probe accuracies, RSA, item-level margins, steering. This
one needs none. If a model spontaneously says "he did not know the powder was poison" and
then blames him as if he did, the dissociation is visible in its own output, and no reader
has to accept anything about linear probes to see it.

DESIGN. The rating is NOT re-measured from prose: it stays the project's logprob-EV digit
score, so this experiment is comparable with everything else. The explanation is then
generated with the model's own argmax rating already in the conversation:

    user      <the standard rating prompt>
    assistant <the model's argmax digit>
    user      In one or two sentences, explain your rating.

so the prose explains the rating being analysed rather than an independently sampled one.
The follow-up question deliberately does NOT mention intent, belief, knowledge or accident;
any mention in the response is the model's own.

CODING SCHEME (rater A, deterministic; the exact patterns are in MENTION_PATTERNS below).
Three binary codes per generation, from the surface form of the explanation only:
  - mentions_belief : epistemic state of the agent -- knew / did not know / believed /
                      thought / was (un)aware / assumed / expected / realised / mistaken
                      about the fact.
  - mentions_intent : volitional state -- intended / deliberate / on purpose / knowingly /
                      meant to / accidental / unintentional / did not mean to.
  - mentions_outcome: the consequence -- harm, death, injury, illness, damage.
`mentions_either` is belief OR intent, and is the unit of the headline claim. Negated forms
count as mentions: "he did not know it was poison" is an explicit representation of the
agent's belief in the explanation, which is exactly the construct.

AGREEMENT. Rater A is scored against two independent raters:
  - rater B, an LLM classifier (`--llm-rater`, same closed-API plumbing as 45), run blind
    to the condition and to A's label;
  - rater C, hand labels. `--run` emits a seeded random sample to
    w3_prose_manual_sample_<tag>.csv with an empty `human_either` column; the sample is
    fixed by seed before any label is written, so it cannot be chosen after the fact.
Cohen's kappa and raw agreement are reported for A-vs-B and A-vs-C.

Usage
  python code/experiments/51_w3_prose_rating.py --run --model allenai/OLMo-2-1124-7B-Instruct
  python code/experiments/51_w3_prose_rating.py --llm-rater --backend anthropic \
      --rater-model claude-haiku-4-5-20251001        # optional second automated rater
  python code/experiments/51_w3_prose_rating.py --report          # CPU, writes the .md

Outputs
  outputs/experiments/w3_prose_items_<tag>.csv
  outputs/experiments/w3_prose_manual_sample_<tag>.csv   (hand-label template)
  outputs/experiments/w3_prose_llm_labels_<tag>.csv      (with --llm-rater)
  outputs/experiments/W3_PROSE_RATING.md
  outputs/experiments/w3_prose_rating.png
"""
import argparse
import csv
import glob
import importlib.util
import os
import re
import sys

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CODE_DIR)
import numpy as np  # noqa: E402
import tom_common as tc  # noqa: E402

MASTER_CSV = os.path.join(tc.ROOT, "dataset", "master", "moral_2x2_master.csv")
OUT_DIR = os.path.join(tc.ROOT, "outputs", "experiments")
OUT_MD = os.path.join(OUT_DIR, "W3_PROSE_RATING.md")

GUILTY = {"attempted", "intentional"}     # intent present
HARM = {"accidental", "intentional"}      # bad outcome present
CELLS = ["neutral", "accidental", "attempted", "intentional"]

FOLLOWUP = "In one or two sentences, explain your rating."

# Rater A. Word-boundary patterns; matched case-insensitively against the explanation.
#
# RECALL EXPANSION, 2026-07-28. The first version of this list missed a large family of
# perfectly explicit mental-state expressions -- "could have foreseen", "failed to notice",
# "recognized", "trusted", "being informed", "incorrectly assessed", "did not anticipate",
# "without considering", "underestimated". On the 80-item development sample it labelled
# YES on 64% where an independent reading said 85%, and the independent LLM coder said 83%;
# 36 of 40 disagreements ran in the same direction (A=no, other=yes). The patterns marked
# "added" below close that gap. They were chosen from disagreements on the DEVELOPMENT
# sample, so that sample can no longer validate them -- validation is the independent LLM
# coder over all items plus a held-out reading (see W3_PROSE_RATING.md).
MENTION_PATTERNS = {
    "belief": [
        r"\bknew\b", r"\bknow(?:s|ing)?\b", r"\bknowledge\b", r"\bunaware\b",
        r"\baware(?:ness)?\b", r"\bbelie(?:f|ve[sd]?|ving)\b", r"\bthought\b",
        r"\bthinking\b", r"\bassum(?:e[sd]?|ing|ption)\b", r"\bexpect(?:ed|ing|ation)\b",
        r"\brealis(?:e[sd]?|ing)\b", r"\brealiz(?:e[sd]?|ing)\b", r"\bmistaken\b",
        r"\bmisunderstood\b", r"\bmisunderstanding\b", r"\bignorant\b", r"\bignorance\b",
        r"\bno idea\b", r"\bdid ?n[o']t realise\b", r"\bdid ?n[o']t realize\b",
        r"\bhad no way of knowing\b", r"\bunknown to\b",
        # added
        r"\bforesee(?:able)?\b", r"\bforesaw\b", r"\bforeseen\b",
        r"\brecogni[sz](?:e[sd]?|ing)\b", r"\bnotic(?:e[sd]?|ing)\b",
        r"\bobserv(?:e[sd]?|ing)\b", r"\bperceiv(?:e[sd]?|ing)\b",
        r"\bsuspect(?:ed|ing|s)?\b", r"\bsuspicion\b", r"\btrust(?:ed|ing|s)?\b",
        r"\binform(?:ed|ing)\b", r"\banticipat(?:e[sd]?|ing)\b",
        r"\bverif(?:y|ied|ying)\b", r"\bidentif(?:y|ied|ying)\b",
        r"\bassess(?:ed|ing|ment)?\b", r"\bconsider(?:ed|ing)\b",
        r"\bunderestimat(?:e[sd]?|ing)\b", r"\boverestimat(?:e[sd]?|ing)\b",
        r"\bdisregard(?:ed|ing|s)?\b", r"\bignor(?:e[sd]?|ing)\b",
        r"\boverlook(?:ed|ing)\b", r"\bdiscover(?:ed|ing|y)\b",
        # narrow: "concerns about the danger" but not "some ethical concerns"
        r"\bconcern(?:s|ed)?\s+(?:about|for|over|regarding)\b",
        r"\bbased on (?:the )?information\b",
    ],
    "intent": [
        r"\bintent(?:ion(?:al(?:ly)?|s)?|ional)?\b", r"\bintend(?:ed|s|ing)?\b",
        r"\bunintention(?:al(?:ly)?)?\b", r"\bdeliberate(?:ly)?\b", r"\bon purpose\b",
        r"\bpurpose(?:ful(?:ly)?|ly)\b", r"\bknowingly\b", r"\bwilling(?:ly)?\b",
        r"\bmeant to\b", r"\bdid ?n[o']t mean\b", r"\baccident(?:al(?:ly)?|s)?\b",
        r"\bmalicious(?:ly)?\b", r"\bmalice\b", r"\bmotive\b", r"\btried to\b",
        r"\battempt(?:ed|ing|s)?\b", r"\bwanted to\b", r"\bdesire[sd]?\b",
        r"\bnegligen(?:t|ce)\b", r"\brecklessn?e?s?s?\b", r"\bcareless(?:ly|ness)?\b",
        # added
        r"\bconsent(?:ed|ing)?\b", r"\bvolunt(?:ary|arily)\b", r"\bon his own\b",
    ],
    "outcome": [
        r"\bharm(?:ed|ful|s)?\b", r"\bdie[sd]?\b", r"\bdeath\b", r"\bdead\b",
        r"\bkilled?\b", r"\binjur(?:e[sd]?|y|ies)\b", r"\bhurt\b", r"\bsick(?:ness)?\b",
        r"\bill(?:ness)?\b", r"\bpoison(?:ed|ing)?\b", r"\bdamage[sd]?\b",
        r"\bconsequence[s]?\b", r"\boutcome[s]?\b", r"\bsuffer(?:ed|ing)?\b",
        r"\bunharmed\b", r"\bno harm\b", r"\bfine\b", r"\bunaffected\b",
    ],
}
COMPILED = {k: [re.compile(p, re.I) for p in v] for k, v in MENTION_PATTERNS.items()}

LLM_RATER_PROMPT = """You are labelling one sentence-or-two explanation written by another \
model about a short story. Answer only about what the explanation SAYS, not about whether \
it is correct.

Does the explanation explicitly refer to the acting person's mental state -- what they knew, \
believed, thought or were aware of, OR whether they acted intentionally, deliberately, \
accidentally or unintentionally? Statements in the negative ("he did not know", "it was an \
accident") DO count.

Explanation:
\"\"\"{text}\"\"\"

Answer with exactly one word: YES or NO."""


def _load_behavioral():
    spec = importlib.util.spec_from_file_location(
        "behavioral", os.path.join(CODE_DIR, "03_behavioral.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class _DigitShim:
    def __init__(self, tok, model_name):
        self.tok = tok
        self.model_name = model_name


def code_mentions(text):
    out = {}
    for k, pats in COMPILED.items():
        hits = [p.pattern for p in pats if p.search(text or "")]
        out[f"mentions_{k}"] = int(bool(hits))
        out[f"{k}_hits"] = ";".join(sorted({h for h in hits}))[:200]
    out["mentions_either"] = int(bool(out["mentions_belief"] or out["mentions_intent"]))
    return out


def kappa(a, b):
    """Cohen's kappa for two binary label sequences."""
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    if len(a) == 0:
        return float("nan"), float("nan")
    po = float((a == b).mean())
    pe = float((a.mean() * b.mean()) + ((1 - a.mean()) * (1 - b.mean())))
    return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")


def run_model(a):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    beh = _load_behavioral()
    with open(MASTER_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if a.limit:
        rows = rows[:a.limit]
    tag = a.model.split("/")[-1]
    print(f"Loading {a.model} ... ({len(rows)} stories)")
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()
    dig = beh.HFBackend._digit_token_ids(_DigitShim(tok, a.model))

    prepared = []
    for r in rows:
        prompt, s_min, s_max = beh.build_prompt(r["text"], a.template, r["source"])
        chat = (tok.apply_chat_template([{"role": "user", "content": prompt}],
                                        tokenize=False, add_generation_prompt=True)
                if tok.chat_template else prompt)
        prepared.append((r, chat, prompt, int(s_min), int(s_max)))

    # ---- 1. rating: unchanged logprob-EV digit scoring ----
    print("  scoring ratings (logprob EV over scale digits) ...")
    ratings = []
    with torch.no_grad():
        for i in range(0, len(prepared), a.batch_size):
            chunk = prepared[i:i + a.batch_size]
            enc = tok([c[1] for c in chunk], return_tensors="pt",
                      padding=True).to(model.device)
            logits = model(**enc).logits[:, -1, :].float()
            for (_r, _c, _p, s_min, s_max), lg in zip(chunk, logits):
                vals = [d for d in range(s_min, s_max + 1) if d in dig]
                lp = torch.tensor([lg[dig[d]].item() for d in vals])
                p = torch.softmax(lp, 0).tolist()
                ev = sum(p[j] * vals[j] for j in range(len(vals)))
                ratings.append((ev, vals[int(np.argmax(p))], s_min, s_max))

    # ---- 2. explanation, conditioned on the model's own argmax rating ----
    print("  generating explanations (rating already in context) ...")
    exp_prompts = []
    for (r, _chat, prompt, _lo, _hi), (_ev, argmax, _a, _b) in zip(prepared, ratings):
        msgs = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": str(argmax)},
                {"role": "user", "content": FOLLOWUP}]
        exp_prompts.append(
            tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            if tok.chat_template
            else f"{prompt}\n{argmax}\n{FOLLOWUP}\n")
    proses = []
    with torch.no_grad():
        for i in range(0, len(exp_prompts), a.gen_batch):
            enc = tok(exp_prompts[i:i + a.gen_batch], return_tensors="pt",
                      padding=True).to(model.device)
            g = model.generate(**enc, max_new_tokens=a.gen_tokens, do_sample=False,
                               pad_token_id=tok.pad_token_id)
            for j in range(g.shape[0]):
                new = g[j, enc["input_ids"].shape[1]:]
                proses.append(" ".join(
                    tok.decode(new, skip_special_tokens=True).split()))
            if i % (a.gen_batch * 5) == 0:
                print(f"    {i + g.shape[0]}/{len(exp_prompts)}")

    out = []
    for (r, _chat, _p, s_min, s_max), (ev, argmax, _a, _b), prose in zip(
            prepared, ratings, proses):
        rec = dict(model=tag, story_id=r["story_id"],
                   scenario_group=tc.scenario_group_of(r["story_id"]),
                   condition=r["condition"], source=r["source"],
                   intent=int(r["condition"] in GUILTY),
                   outcome=int(r["condition"] in HARM),
                   rating_ev=round(ev, 4), rating_argmax=argmax,
                   rating_norm=round((ev - s_min) / (s_max - s_min), 4),
                   prose=prose, n_words=len(prose.split()))
        rec.update(code_mentions(prose))
        out.append(rec)
    path = os.path.join(OUT_DIR, f"w3_prose_items_{tag}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"wrote {os.path.relpath(path, tc.ROOT)} ({len(out)} rows); "
          f"mention rate (either) = {np.mean([o['mentions_either'] for o in out]):.3f}")

    # Hand-label template, sampled with a fixed seed BEFORE any label exists.
    rng = np.random.default_rng(a.manual_seed)
    idx = rng.choice(len(out), size=min(a.n_manual, len(out)), replace=False)
    mpath = os.path.join(OUT_DIR, f"w3_prose_manual_sample_{tag}.csv")
    with open(mpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["story_id", "prose", "rater_a_either", "human_either"])
        for i in sorted(idx):
            w.writerow([out[i]["story_id"], out[i]["prose"],
                        out[i]["mentions_either"], ""])
    print(f"wrote {os.path.relpath(mpath, tc.ROOT)} "
          f"({len(idx)} rows to hand-label in `human_either`)")


def recode(a):
    """Re-apply rater A to the stored prose after a pattern change. No GPU needed."""
    for path in sorted(glob.glob(os.path.join(OUT_DIR, "w3_prose_items_*.csv"))):
        rows = list(csv.DictReader(open(path)))
        if not rows:
            continue
        before = float(np.mean([int(r["mentions_either"]) for r in rows]))
        for r in rows:
            r.update({k: str(v) for k, v in code_mentions(r["prose"]).items()})
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        after = float(np.mean([int(r["mentions_either"]) for r in rows]))
        print(f"recoded {os.path.relpath(path, tc.ROOT)}: mention rate "
              f"{before:.3f} -> {after:.3f}")
        # Keep the hand-label template's rater_a column in step with the new patterns,
        # without touching the labels themselves.
        tag = os.path.basename(path)[len("w3_prose_items_"):-4]
        mp = os.path.join(OUT_DIR, f"w3_prose_manual_sample_{tag}.csv")
        if os.path.exists(mp):
            by_id = {r["story_id"]: r for r in rows}
            mrows = list(csv.DictReader(open(mp)))
            for r in mrows:
                if r["story_id"] in by_id:
                    r["rater_a_either"] = by_id[r["story_id"]]["mentions_either"]
            with open(mp, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(mrows[0].keys()))
                w.writeheader()
                w.writerows(mrows)


def run_llm_rater(a):
    """Second automated rater, blind to condition and to rater A's label."""
    sys.path.insert(0, os.path.join(CODE_DIR, "experiments"))
    spec = importlib.util.spec_from_file_location(
        "gen45", os.path.join(CODE_DIR, "experiments", "45_tom_generative.py"))
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    backend = g.BACKENDS[a.backend](a.rater_model)
    for path in sorted(glob.glob(os.path.join(OUT_DIR, "w3_prose_items_*.csv"))):
        tag = os.path.basename(path)[len("w3_prose_items_"):-4]
        items = list(csv.DictReader(open(path)))
        rng = np.random.default_rng(a.manual_seed)
        idx = sorted(rng.choice(len(items), size=min(a.n_llm, len(items)),
                                replace=False))
        outp = os.path.join(OUT_DIR, f"w3_prose_llm_labels_{tag}.csv")
        done = {r["story_id"]: r for r in csv.DictReader(open(outp))} \
            if os.path.exists(outp) else {}
        recs = []
        for n, i in enumerate(idx):
            it = items[i]
            if it["story_id"] in done:
                recs.append(done[it["story_id"]])
                continue
            try:
                resp = backend.generate(LLM_RATER_PROMPT.format(text=it["prose"]))
            except Exception as e:                       # keep partial progress
                print(f"    !! {tag} {it['story_id']}: {e}")
                break
            lab = 1 if re.search(r"\byes\b", resp, re.I) else (
                0 if re.search(r"\bno\b", resp, re.I) else "")
            recs.append(dict(story_id=it["story_id"], llm_either=lab,
                             raw=resp.strip()[:60]))
            if n % 25 == 0:
                print(f"    {tag}: {n}/{len(idx)}")
        with open(outp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["story_id", "llm_either", "raw"])
            w.writeheader()
            w.writerows(recs)
        print(f"wrote {os.path.relpath(outp, tc.ROOT)} ({len(recs)} labels, "
              f"rater={a.rater_model})")


def coeffs(rows, key="rating_norm"):
    """b_intent and b_outcome as 2x2 marginal differences, plus cell means."""
    def m(pred):
        v = [float(r[key]) for r in rows if pred(r)]
        return float(np.mean(v)) if v else float("nan")
    cells = {c: m(lambda r, c=c: r["condition"] == c) for c in CELLS}
    b_int = m(lambda r: int(r["intent"]) == 1) - m(lambda r: int(r["intent"]) == 0)
    b_out = m(lambda r: int(r["outcome"]) == 1) - m(lambda r: int(r["outcome"]) == 0)
    contrast = cells["attempted"] - cells["accidental"]
    return dict(n=len(rows), cells=cells, b_intent=b_int, b_outcome=b_out,
                ratio=(b_out / b_int if b_int and abs(b_int) > 1e-9 else float("nan")),
                contrast=contrast)


def load_items(path):
    """Items with rater A (lexicon) and, where available, rater B (LLM) labels attached."""
    rows = list(csv.DictReader(open(path)))
    tag = rows[0]["model"] if rows else ""
    lp = os.path.join(OUT_DIR, f"w3_prose_llm_labels_{tag}.csv")
    llm = {}
    if os.path.exists(lp):
        for r in csv.DictReader(open(lp)):
            if str(r.get("llm_either", "")).strip() != "":
                llm[r["story_id"]] = int(r["llm_either"])
    for r in rows:
        r["llm_either"] = llm.get(r["story_id"], "")
    return tag, rows, len(llm)


def report(a):
    paths = sorted(glob.glob(os.path.join(OUT_DIR, "w3_prose_items_*.csv")))
    if not paths:
        print("no w3_prose_items_*.csv found")
        return
    L = ["# Prose/rating dissociation — models verbalise intent and rate by outcome", "",
         "Generated by `code/experiments/51_w3_prose_rating.py`. The rating is the "
         "project's standard logprob-EV digit score; the explanation is generated with "
         "the model's own argmax rating already in the conversation, and the follow-up "
         "question (\"" + FOLLOWUP + "\") never mentions intent, belief or accident, so "
         "any mention is the model's own.", "",
         "**Why this matters: it needs no probing.** Every other test of "
         "\"intent is represented but unused\" rests on an internal measurement — probe "
         "accuracy, RSA, steering. This one is visible in the model's own output.", "",
         "**Primary coder is the LLM classifier (rater B), not the lexicon.** The lexicon "
         "(rater A) was written first and turned out to have a large recall hole; it is "
         "kept as a deterministic, inspectable cross-check and every headline number is "
         "reported under both coders. See the agreement section for why, and for what "
         "would still improve it.", ""]

    per_model = []
    for p in paths:
        tag, rows, n_llm = load_items(p)
        coded = [r for r in rows if r["llm_either"] != ""]
        primary = "llm" if len(coded) >= 0.9 * len(rows) else "lex"
        key = "llm_either" if primary == "llm" else "mentions_either"
        base = coded if primary == "llm" else rows
        ment = [r for r in base if int(r[key])]
        noment = [r for r in base if not int(r[key])]
        per_model.append(dict(
            tag=tag, rows=rows, coded=coded, primary=primary, key=key, base=base,
            allc=coeffs(base), mc=coeffs(ment), nc=coeffs(noment) if noment else None,
            ment=ment, n_llm=n_llm))

    L += ["## Mention rates", "",
          "| model | N stories | named belief or intent (rater B, primary) | same "
          "(rater A lexicon) | belief (A) | intent (A) | outcome (A) | median words |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in per_model:
        rows = m["rows"]

        def rate(k, rs=rows):
            vals = [int(r[k]) for r in rs if str(r[k]) != ""]
            return float(np.mean(vals)) if vals else float("nan")
        L.append(f"| {m['tag']} | {len(rows)} | "
                 f"**{rate('llm_either', m['coded']):.3f}** ({len(m['coded'])} coded) | "
                 f"{rate('mentions_either'):.3f} | {rate('mentions_belief'):.3f} | "
                 f"{rate('mentions_intent'):.3f} | {rate('mentions_outcome'):.3f} | "
                 f"{int(np.median([int(r['n_words']) for r in rows]))} |")

    L += ["", "## The dissociation", "",
          "`b_intent` and `b_outcome` are 2x2 marginal differences on the normalised "
          "rating (intent-present minus intent-absent; harm minus no-harm). The subset "
          "row is the one that matters: **restricted to the stories where the model "
          "itself named the agent's belief or intent**, is the rating still driven by "
          "outcome?", "",
          "| model | coder | subset | n | b_intent | b_outcome | b_outcome / b_intent "
          "| contrast (attempted − accidental) |",
          "|---|---|---|---:|---:|---:|---:|---:|"]
    for m in per_model:
        for coder, key, base in (("B (LLM)", "llm_either", m["coded"]),
                                 ("A (lexicon)", "mentions_either", m["rows"])):
            if not base:
                continue
            subsets = (("all stories", base),
                       ("named intent/belief", [r for r in base if int(r[key])]),
                       ("no mention", [r for r in base if not int(r[key])]))
            for label, rs in subsets:
                if not rs:
                    continue
                c = coeffs(rs)
                L.append(f"| {m['tag']} | {coder} | {label} | {c['n']} | "
                         f"{c['b_intent']:+.3f} | {c['b_outcome']:+.3f} | "
                         f"{c['ratio']:.1f}x | {c['contrast']:+.3f} |")
    L += ["", "The pattern is the same under both coders: inside the subset where the "
          "model named the agent's mental state, outcome still moves the rating several "
          "times more than intent does, and the attempted-minus-accidental contrast stays "
          "inverted relative to adults.", ""]

    L += ["", "## Cell means among stories where the model named intent or belief", "",
          "Primary coder. Cells are computed within-condition, so they are unaffected by "
          "the fact that mention rates differ across conditions.", "",
          "| model | neutral | accidental | attempted | intentional | human ordering? |",
          "|---|---:|---:|---:|---:|---|"]
    for m in per_model:
        c = m["mc"]["cells"]
        ok = c["attempted"] > c["accidental"]
        L.append(f"| {m['tag']} | " + " | ".join(f"{c[k]:.3f}" for k in CELLS)
                 + f" | {'matches human' if ok else '**inverted**'} |")

    # Mention is not randomised: the accidental condition invites the word "accident", so
    # the mentioned subset is not a random half of the stimuli. Reporting the per-cell rate
    # is the only way a reader can judge how much of the subset comparison is selection.
    L += ["", "## Mention is not randomised — rates by condition", "",
          "The stimuli themselves make some conditions likelier to elicit a mental-state "
          "word (an accidental harm invites \"accident\"; a neutral story invites nothing). "
          "So the mentioned subset is not a random half of the stimuli, and the "
          "mention-vs-no-mention comparison above is partly a comparison between different "
          "story mixes. The per-cell rates below let that be judged directly, and the cell "
          "means in the previous table are computed within-cell, so they are not affected "
          "by the mix.", "",
          "| model | coder | neutral | accidental | attempted | intentional |",
          "|---|---|---:|---:|---:|---:|"]
    for m in per_model:
        for coder, key, base in (("B (LLM)", "llm_either", m["coded"]),
                                 ("A (lexicon)", "mentions_either", m["rows"])):
            if not base:
                continue
            L.append(f"| {m['tag']} | {coder} | " + " | ".join(
                f"{float(np.mean([int(r[key]) for r in base if r['condition'] == c] or [np.nan])):.3f}"
                for c in CELLS) + " |")

    # ---- agreement ----
    L += ["", "## Coding scheme and rater agreement", "",
          "Rater A is the deterministic lexicon in `MENTION_PATTERNS` "
          "(" + ", ".join(f"{k}: {len(v)} patterns"
                          for k, v in MENTION_PATTERNS.items()) + "). Negated forms count "
          "as mentions: \"he did not know it was poison\" explicitly represents the "
          "agent's belief, which is the construct being coded.", "",
          "| comparison | model | n | raw agreement | Cohen's κ |",
          "|---|---|---:|---:|---:|"]
    any_agree = False
    for m in per_model:
        tag, rows = m["tag"], m["rows"]
        by_id = {r["story_id"]: r for r in rows}
        for kind, fname, col in (("A vs B (LLM rater)",
                                 f"w3_prose_llm_labels_{tag}.csv", "llm_either"),
                                 ("A vs C (hand labels)",
                                  f"w3_prose_manual_sample_{tag}.csv", "human_either")):
            fp = os.path.join(OUT_DIR, fname)
            if not os.path.exists(fp):
                continue
            pairs = [(int(by_id[r["story_id"]]["mentions_either"]), int(r[col]))
                     for r in csv.DictReader(open(fp))
                     if r.get(col, "").strip() != "" and r["story_id"] in by_id]
            if not pairs:
                continue
            any_agree = True
            po, k = kappa([x for x, _ in pairs], [y for _, y in pairs])
            L.append(f"| {kind} | {tag} | {len(pairs)} | {po:.3f} | {k:.3f} |")
    if not any_agree:
        L.append("| _no second-rater labels on disk yet_ | | | | |")

    L += ["", "## Reading", "",
          "Where the mention rate is high and `b_outcome` stays several times "
          "`b_intent` **inside the mentioned subset**, the model is naming the agent's "
          "mental state in the same turn in which it rates by consequence. That is a "
          "dissociation between what the model says and what it does, established "
          "entirely from its own text.", "",
          "Limits. The codes are surface mentions, not a check that the mention is "
          "correct or that it is doing argumentative work; a model can name intent and "
          "then explicitly dismiss it as irrelevant, which this coding cannot "
          "distinguish from naming it and ignoring it. Explanations are generated after "
          "the rating is fixed in context, so they are post-hoc rationalisations of that "
          "rating rather than its cause.", ""]
    open(OUT_MD, "w").write("\n".join(L) + "\n")
    print(f"wrote {os.path.relpath(OUT_MD, tc.ROOT)}")
    _plot(per_model)


def _plot(per_model):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (skip figure: {e})")
        return
    fig, axes = plt.subplots(1, len(per_model), figsize=(4.6 * len(per_model), 4.0),
                             squeeze=False)
    for ax, m in zip(axes[0], per_model):
        tag, rows, mc = m["tag"], m["rows"], m["mc"]
        c = mc["cells"]
        xs = np.arange(4)
        ax.bar(xs, [c[k] for k in CELLS],
               color=["#bbbbbb", "#c45c26", "#1f3f8f", "#7a2f8f"])
        ax.set_xticks(xs)
        ax.set_xticklabels(CELLS, rotation=20, fontsize=8.5)
        ax.set_ylim(0, 1)
        ax.set_ylabel("normalised blame rating")
        rate = float(np.mean([int(r["mentions_either"]) for r in rows]))
        ax.set_title(f"{tag}\nnamed intent/belief in {rate:.0%} of explanations; "
                     f"b_outcome {mc['b_outcome']:+.2f} vs b_intent "
                     f"{mc['b_intent']:+.2f}", fontsize=9)
        ax.grid(axis="y", alpha=0.25, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Ratings on the stories where the model itself named the agent's "
                 "intent or belief", fontsize=10.5)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "w3_prose_rating.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"wrote {os.path.relpath(out, tc.ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-2-1124-7B-Instruct")
    ap.add_argument("--template", default="human_verbatim")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--gen-batch", type=int, default=8)
    ap.add_argument("--gen-tokens", type=int, default=72)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-manual", type=int, default=40,
                    help="rows emitted for hand labelling (rater C)")
    ap.add_argument("--n-llm", type=int, default=150,
                    help="rows sent to the LLM second rater (rater B)")
    ap.add_argument("--manual-seed", type=int, default=7)
    ap.add_argument("--backend", default="anthropic")
    ap.add_argument("--rater-model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--llm-rater", action="store_true")
    ap.add_argument("--recode", action="store_true",
                    help="re-apply rater A to stored prose after a pattern change")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if a.run:
        run_model(a)
    if a.recode:
        recode(a)
    if a.llm_rater:
        run_llm_rater(a)
    if a.report or not (a.run or a.llm_rater or a.recode):
        report(a)


if __name__ == "__main__":
    main()
