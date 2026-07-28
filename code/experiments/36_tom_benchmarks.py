#!/usr/bin/env python3
"""J1 -- score open models on standard theory-of-mind benchmarks (BigToM, ToMi).

PURPOSE AND PRE-REGISTERED INTERPRETATION (written before any model was scored):

The project claims a dissociation: models represent intent and pass belief-reasoning
tasks, yet do not weight intent in graded moral judgment. That claim currently rests on
a literature argument. This script replaces it with our own measurement by scoring the
same models on standard ToM benchmarks and correlating that score against the 2x2
intent contrast.

  * A NULL correlation means ToM-benchmark performance does not predict whether a model
    weights intent in graded moral judgment. The two abilities come apart, and the
    dissociation claim is supported by our own data.
  * A POSITIVE correlation means the moral task is partly measuring general ToM
    competence. The dissociation claim WEAKENS and must be restated: models that reason
    about beliefs better also use intent more, and the moral task is not isolating a
    separate failure.
  * A CEILING on the benchmark (accuracy > 0.95, near-zero spread) means the correlation
    is not estimable at all, because a correlation needs variance on both axes. In that
    case report the ceiling and do not fit.

Both directions are publishable; the point is that we do not get to choose after seeing
the number.

SCORING. Every item is reduced to a two-alternative forced choice and scored by
length-normalised log-likelihood of each option continuation, argmax wins. This avoids
free-generation parsing and is the closest analogue to the logprob scoring used for the
moral task.

BENCHMARK SUBSETS.
  BigToM  -- Forward Belief. Each of the 200 stories is scored twice: once with the
             agent perceiving the causal event (true belief) and once without (false
             belief). The false-belief half is the ToM-critical condition.
  ToMi    -- first-order belief questions ("Where will X look for Y?"). Memory and
             reality questions are kept separately as a comprehension control: a model
             failing those is not failing at ToM, it is failing at reading.
"""
import argparse
import csv
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "dataset", "tom_benchmarks")
OUT = os.path.join(ROOT, "outputs", "tom_benchmarks")

CEILING = 0.95


# ---------------------------------------------------------------- loaders
def load_bigtom():
    """Forward Belief: 200 stories x {true belief, false belief} = 400 items.

    Column order and the answer-to-percept pairing follow the generator in the BigToM
    repo (code/src/generate_conditions.py): "Belief Answer Aware" (field 8) is correct
    when the agent perceives the event, "Belief Answer not Aware" (field 11) when it does
    not. Sentence 4 of the story, the explicit statement of the agent's initial belief,
    is dropped -- that is the init_belief=0 variant, which forces the model to infer the
    belief rather than copy it.
    """
    path = os.path.join(DATA, "bigtom.csv")
    items = []
    for i, line in enumerate(open(path, encoding="utf-8")):
        f = line.rstrip("\n").split(";")
        if len(f) < 12:
            continue
        parts = f[0].split(".")
        if len(parts) < 5:
            continue
        story = ".".join([parts[0], parts[1], parts[2], parts[4]]).strip() + "."
        sees, not_sees = f[1].strip(), f[2].strip()
        question = f[5].strip()
        ans_aware, ans_unaware = f[8].strip(), f[11].strip()
        if ans_aware == ans_unaware:
            continue  # no contrast between conditions for this row
        for cond, percept, correct, wrong in (
                ("false_belief", not_sees, ans_unaware, ans_aware),
                ("true_belief", sees, ans_aware, ans_unaware)):
            prompt = f"{story} {percept}\n\nQuestion: {question}\nAnswer:"
            items.append(dict(bench="bigtom", item_id=f"bigtom-{i}-{cond}",
                              subset=cond, prompt=prompt,
                              options=[correct, wrong], correct=0))
    return items


_CONT = re.compile(r"\b(?:in|to|on) the (\w+)")


def load_tomi(max_items=None):
    """ToMi bAbI blocks, paired with the .trace file for question type and condition.

    Every retained item has exactly two candidate containers, the pre-move and post-move
    locations, so the forced choice is the intended one: an agent absent for the move should
    be expected at the pre-move location.

    Caveat on the subset labels. The trace file's true_belief / false_belief field describes
    the STORY-GENERATION condition, not the belief state of whichever agent the question
    happens to ask about, and the two come apart -- there are items tagged true_belief whose
    question asks about an agent who left before the object moved. Use the aggregate `tomi`
    accuracy and the question-type breakdown; do not read the belief tag as a per-item
    condition label. Cells scoring near zero are models answering with the object's current
    location, which is the ordinary reality-bias error rather than a parsing failure.
    """
    txt = os.path.join(DATA, "tomi_balanced_story_types", "fb_all_test.txt")
    trc = os.path.join(DATA, "tomi_balanced_story_types", "fb_all_test.trace")
    if not (os.path.exists(txt) and os.path.exists(trc)):
        return []
    traces = [l.strip().split(",") for l in open(trc) if l.strip()]

    blocks, cur = [], []
    for line in open(txt, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            continue
        if line.startswith("1 ") and cur:
            blocks.append(cur)
            cur = []
        cur.append(line)
    if cur:
        blocks.append(cur)

    items = []
    for bi, blk in enumerate(blocks):
        if bi >= len(traces):
            break
        qtype = traces[bi][-2] if len(traces[bi]) >= 2 else "unknown"
        belief = traces[bi][-1] if traces[bi] else "unknown"
        story_lines, question, gold = [], None, None
        for l in blk:
            body = l.split(" ", 1)[1] if " " in l else l
            if "\t" in body:
                parts = body.split("\t")
                question, gold = parts[0].strip(), parts[1].strip()
            else:
                story_lines.append(body.strip())
        if not question or not gold:
            continue
        # Candidate containers, in order of first mention.
        cands = []
        for s in story_lines:
            for m in _CONT.findall(s):
                if m not in cands:
                    cands.append(m)
        if gold not in cands or len(cands) < 2:
            continue
        wrong = next((c for c in cands if c != gold), None)
        if wrong is None:
            continue
        story = " ".join(story_lines)
        prompt = f"{story}\n\nQuestion: {question}\nAnswer:"
        items.append(dict(bench="tomi", item_id=f"tomi-{bi}",
                          subset=f"{qtype}|{belief}", prompt=prompt,
                          options=[gold, wrong], correct=0))
    # first-order belief questions are the ToM-critical set; keep the rest as control
    def rank(it):
        s = it["subset"]
        return (0 if "first_order" in s else (1 if "second_order" in s else 2))
    items.sort(key=lambda it: (rank(it), it["item_id"]))
    if max_items:
        items = items[:max_items]
    return items


# ---------------------------------------------------------------- scoring
class OptionScorer:
    def __init__(self, model_name):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.name = model_name
        print(f"  loading {model_name} ...", flush=True)
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.mdl = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True)
        self.mdl.eval()

    def _fmt(self, prompt):
        if getattr(self.tok, "chat_template", None):
            return self.tok.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True)
        return prompt

    def score(self, prompt, options):
        """Mean log-probability of each option continuation; returns argmax index."""
        torch = self.torch
        ctx = self._fmt(prompt)
        ctx_ids = self.tok(ctx, return_tensors="pt").input_ids
        scores = []
        for opt in options:
            full = self.tok(ctx + " " + opt.strip(), return_tensors="pt").input_ids
            full = full.to(self.mdl.device)
            n_ctx = ctx_ids.shape[1]
            if full.shape[1] <= n_ctx:
                scores.append(float("-inf"))
                continue
            with torch.no_grad():
                logits = self.mdl(full).logits[0]
            lp = torch.log_softmax(logits[:-1].float(), dim=-1)
            tgt = full[0, 1:]
            tok_lp = lp[torch.arange(tgt.shape[0]), tgt][n_ctx - 1:]
            scores.append(float(tok_lp.mean()))
        return int(max(range(len(scores)), key=lambda i: scores[i])), scores


def run_model(model_name, items, out_dir):
    tag = model_name.replace("/", "_").replace(".", "_")
    per_path = os.path.join(out_dir, f"tom_items_{tag}.csv")
    sc = OptionScorer(model_name)
    rows = []
    for i, it in enumerate(items):
        pred, scores = sc.score(it["prompt"], it["options"])
        rows.append([it["bench"], it["subset"], it["item_id"], pred,
                     it["correct"], int(pred == it["correct"]),
                     round(scores[0], 4), round(scores[1], 4)])
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(items)} scored", flush=True)
    with open(per_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bench", "subset", "item_id", "pred", "correct", "is_correct",
                    "lp_correct", "lp_wrong"])
        w.writerows(rows)

    agg = {}
    for r in rows:
        for key in (r[0], f"{r[0]}|{r[1]}"):
            a = agg.setdefault(key, [0, 0])
            a[0] += r[5]
            a[1] += 1
    del sc
    try:
        import torch, gc
        gc.collect(); torch.cuda.empty_cache()
    except Exception:
        pass
    return agg, per_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--tomi-limit", type=int, default=400)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--gate", action="store_true",
                    help="ceiling gate: report whether accuracy leaves room for a "
                         "correlation, and do not proceed automatically")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    items = load_bigtom() + load_tomi(a.tomi_limit)
    by_b = {}
    for it in items:
        by_b[it["bench"]] = by_b.get(it["bench"], 0) + 1
    print(f"items: {len(items)}  {by_b}", flush=True)

    summary_path = os.path.join(a.out, "tom_accuracy_by_model.csv")
    existing = {}
    if os.path.exists(summary_path):
        for r in csv.DictReader(open(summary_path)):
            existing[(r["model"], r["subset"])] = r

    out_rows = []
    for m in a.models:
        print(f"\n=== {m} ===", flush=True)
        agg, per = run_model(m, items, a.out)
        for subset, (ok, n) in sorted(agg.items()):
            acc = ok / n if n else float("nan")
            out_rows.append([m, subset, ok, n, round(acc, 4)])
            print(f"  {subset:34} {ok:4}/{n:<4} acc={acc:.3f}", flush=True)

    keep = [list(v.values()) for v in existing.values()] if existing else []
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "subset", "n_correct", "n_items", "accuracy"])
        seen = set()
        for r in out_rows:
            seen.add((r[0], r[1]))
            w.writerow(r)
        for r in keep:
            if (r[0], r[1]) not in seen:
                w.writerow(r)
    print(f"\nwrote {summary_path}")

    if a.gate:
        print("\n=== CEILING GATE ===")
        for bench in ("bigtom", "tomi"):
            accs = [r[4] for r in out_rows if r[1] == bench]
            if not accs:
                continue
            spread = max(accs) - min(accs)
            at_ceiling = all(x > CEILING for x in accs) and spread < 0.05
            print(f"  {bench:8} acc={[f'{x:.3f}' for x in accs]}  "
                  f"spread={spread:.3f}  "
                  f"{'AT CEILING - correlation not estimable' if at_ceiling else 'HAS SPREAD - proceed'}")


if __name__ == "__main__":
    main()
