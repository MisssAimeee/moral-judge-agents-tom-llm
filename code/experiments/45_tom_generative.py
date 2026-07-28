#!/usr/bin/env python3
"""Generative ToM scoring (BigToM; ToMi optional) for open HF and closed APIs.

Purpose: score parity with the logprob 2AFC path in 36_tom_benchmarks.py without
reintroducing a method confound when closed models are added. Free generation is
forced to one of the two option strings (same options as the logprob scorer).

Usage:
  # Open models (GPU) — agreement check against existing logprob CSVs
  python code/experiments/45_tom_generative.py --backend hf \\
      --models Qwen/Qwen2.5-7B-Instruct allenai/OLMo-2-1124-7B-Instruct

  # Closed APIs (CPU; source .env_agents first)
  python code/experiments/45_tom_generative.py --backend anthropic \\
      --models claude-haiku-4-5-20251001 claude-sonnet-4-6 claude-opus-4-6
  python code/experiments/45_tom_generative.py --backend openai \\
      --models gpt-4o gpt-4o-mini
  python code/experiments/45_tom_generative.py --backend google \\
      --models gemini-2.5-flash gemini-2.5-pro

BigToM only by default (ToMi dropped from primary claims; see TOMI_SCORING_AUDIT.md).
Closed-model accuracies are reported standalone — do not correlate against v1 contrasts.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "tom36", os.path.join(HERE, "36_tom_benchmarks.py"))
_tom36 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tom36)
load_bigtom = _tom36.load_bigtom
load_tomi = _tom36.load_tomi

OUT = os.path.join(ROOT, "outputs", "tom_benchmarks")

CHOICE_SUFFIX = (
    "\n\nRespond with exactly one of the following answers, copying the text "
    "verbatim. Do not explain.\n"
    "A) {a}\n"
    "B) {b}\n"
    "Answer:"
)


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = s.replace(" ", "_")
    s = re.sub(r"^[^a-z0-9_]+", "", s)
    s = re.sub(r"[^a-z0-9_]+$", "", s)
    return s


def match_option(text: str, options) -> int | None:
    """Return option index if uniquely identifiable; else None."""
    raw = text.strip()
    # Prefer leading A)/B) markers — also accept a bare "A" / "B" (Gemini often
    # returns just the letter with no punctuation).
    m = re.match(r"^\s*([ab])(?:[\).:\-\s].*)?$", raw, flags=re.I | re.S)
    if m and len(raw) <= 8:
        return 0 if m.group(1).lower() == "a" else 1
    m = re.match(r"^\s*([ab])[\).:\s]", raw, flags=re.I)
    if m:
        return 0 if m.group(1).lower() == "a" else 1
    blob = _norm(raw)
    hits = []
    for i, opt in enumerate(options):
        o = _norm(opt)
        if o and (o in blob or o.replace("_", "") in blob.replace("_", "")):
            hits.append(i)
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        # earliest mention wins
        pos = []
        for i in hits:
            o = _norm(options[i])
            pos.append((blob.find(o) if o in blob else 10**9, i))
        pos.sort()
        return pos[0][1]
    return None


def build_prompt(it) -> str:
    a, b = it["options"][0], it["options"][1]
    return it["prompt"].rstrip() + CHOICE_SUFFIX.format(a=a, b=b)


class HFGen:
    def __init__(self, model_name):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        print(f"  loading {model_name} ...", flush=True)
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.mdl = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True)
        self.mdl.eval()
        self.name = model_name

    def generate(self, prompt: str) -> str:
        if getattr(self.tok, "chat_template", None):
            prompt = self.tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
        ids = self.tok(prompt, return_tensors="pt").to(self.mdl.device)
        with self.torch.no_grad():
            out = self.mdl.generate(
                **ids, max_new_tokens=32, do_sample=False,
                pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(
            out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class OpenAIGen:
    def __init__(self, model_name):
        from openai import OpenAI
        self.client = OpenAI()
        self.name = model_name

    def generate(self, prompt: str) -> str:
        r = self.client.chat.completions.create(
            model=self.name, temperature=0, max_tokens=32,
            messages=[{"role": "user", "content": prompt}])
        return (r.choices[0].message.content or "").strip()


class AnthropicGen:
    def __init__(self, model_name):
        import anthropic
        self.client = anthropic.Anthropic()
        self.name = model_name

    def generate(self, prompt: str) -> str:
        r = self.client.messages.create(
            model=self.name, max_tokens=32, temperature=0,
            messages=[{"role": "user", "content": prompt}])
        parts = [b.text for b in r.content if getattr(b, "type", "") == "text"]
        return "".join(parts).strip()


class GoogleGen:
    def __init__(self, model_name):
        import google.generativeai as genai
        key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise SystemExit("GOOGLE_API_KEY / GEMINI_API_KEY not set")
        genai.configure(api_key=key)
        self.mdl = genai.GenerativeModel(model_name)
        self.name = model_name

    def generate(self, prompt: str) -> str:
        # Thinking Gemini models often put the letter answer after a long chain of
        # thought; give enough output budget, then prefer non-thought parts.
        r = self.mdl.generate_content(
            prompt,
            generation_config={"temperature": 0, "max_output_tokens": 256})
        try:
            parts = r.candidates[0].content.parts
            texts, thoughts = [], []
            for p in parts:
                t = getattr(p, "text", None)
                if not t:
                    continue
                if getattr(p, "thought", False):
                    thoughts.append(t)
                else:
                    texts.append(t)
            out = "".join(texts).strip()
            if out:
                return out
            # Fall back to the last short line of thought (often "A" / "A)")
            blob = "\n".join(thoughts).strip()
            for line in reversed(blob.splitlines()):
                line = line.strip()
                if re.match(r"^[ab][\).:\s]?$", line, flags=re.I):
                    return line
            return blob[-80:] if blob else ""
        except Exception:
            return (getattr(r, "text", None) or "").strip()


BACKENDS = {
    "hf": HFGen,
    "openai": OpenAIGen,
    "anthropic": AnthropicGen,
    "google": GoogleGen,
}


def run_model(backend, model_name, items, out_dir, sleep_s=0.0):
    tag = model_name.replace("/", "_").replace(".", "_")
    per_path = os.path.join(out_dir, f"tom_gen_items_{tag}.csv")
    # resume
    done = {}
    if os.path.exists(per_path):
        for r in csv.DictReader(open(per_path)):
            done[r["item_id"]] = r
        print(f"  [resume] {len(done)} items already in {per_path}", flush=True)

    sc = BACKENDS[backend](model_name)
    rows = list(done.values())
    n_new = 0
    for i, it in enumerate(items):
        if it["item_id"] in done:
            continue
        prompt = build_prompt(it)
        try:
            text = sc.generate(prompt)
        except Exception as e:
            print(f"  !! {it['item_id']}: {e}", flush=True)
            text = ""
        pred = match_option(text, it["options"])
        is_correct = int(pred == it["correct"]) if pred is not None else 0
        parsed = int(pred is not None)
        row = dict(
            bench=it["bench"], subset=it["subset"], item_id=it["item_id"],
            pred="" if pred is None else pred, correct=it["correct"],
            is_correct=is_correct, parsed=parsed,
            response=text.replace("\n", " ")[:200],
            opt0=it["options"][0], opt1=it["options"][1],
            method="generative", backend=backend, model=model_name,
        )
        rows.append(row)
        done[it["item_id"]] = row
        n_new += 1
        if n_new % 25 == 0 or (i + 1) == len(items):
            _flush(per_path, rows)
            print(f"    {i+1}/{len(items)}  new={n_new}", flush=True)
        if sleep_s:
            time.sleep(sleep_s)
    _flush(per_path, rows)

    agg = {}
    for r in rows:
        for key in (r["bench"], f"{r['bench']}|{r['subset']}"):
            a = agg.setdefault(key, [0, 0, 0])  # ok, n, parsed
            a[0] += int(r["is_correct"])
            a[1] += 1
            a[2] += int(r.get("parsed", 1))
    return agg, per_path


def _flush(path, rows):
    cols = ["bench", "subset", "item_id", "pred", "correct", "is_correct",
            "parsed", "response", "opt0", "opt1", "method", "backend", "model"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_agreement(out_dir, gen_tag, model_name):
    """Compare generative vs existing logprob item CSV if present."""
    log_path = os.path.join(out_dir, f"tom_items_{gen_tag}.csv")
    gen_path = os.path.join(out_dir, f"tom_gen_items_{gen_tag}.csv")
    if not (os.path.exists(log_path) and os.path.exists(gen_path)):
        return
    log = {r["item_id"]: r for r in csv.DictReader(open(log_path))}
    gen = {r["item_id"]: r for r in csv.DictReader(open(gen_path))}
    both = [i for i in gen if i in log and i.startswith("bigtom")]
    if not both:
        return
    agree = sum(1 for i in both if str(log[i]["pred"]) == str(gen[i]["pred"]))
    both_ok = sum(1 for i in both
                  if int(log[i]["is_correct"]) == 1 and int(gen[i]["is_correct"]) == 1)
    log_acc = sum(int(log[i]["is_correct"]) for i in both) / len(both)
    gen_acc = sum(int(gen[i]["is_correct"]) for i in both) / len(both)
    path = os.path.join(out_dir, "tom_scoring_agreement.csv")
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["model", "n", "logprob_acc", "generative_acc",
                        "pred_agreement", "both_correct"])
        w.writerow([model_name, len(both), round(log_acc, 4), round(gen_acc, 4),
                    round(agree / len(both), 4), round(both_ok / len(both), 4)])
    print(f"  agreement vs logprob: n={len(both)} log={log_acc:.3f} gen={gen_acc:.3f} "
          f"agree={agree/len(both):.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=list(BACKENDS))
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--include-tomi", action="store_true",
                    help="also score ToMi (dropped from primary claims by default)")
    ap.add_argument("--tomi-limit", type=int, default=400)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    items = load_bigtom()
    if a.include_tomi:
        items = items + load_tomi(a.tomi_limit)
    by_b = {}
    for it in items:
        by_b[it["bench"]] = by_b.get(it["bench"], 0) + 1
    print(f"items: {len(items)}  {by_b}  method=generative  backend={a.backend}",
          flush=True)

    summary_path = os.path.join(a.out, "tom_accuracy_by_model_generative.csv")
    existing = {}
    if os.path.exists(summary_path):
        for r in csv.DictReader(open(summary_path)):
            existing[(r["model"], r["subset"])] = r

    out_rows = []
    for m in a.models:
        print(f"\n=== {m} ({a.backend}) ===", flush=True)
        agg, per = run_model(a.backend, m, items, a.out, sleep_s=a.sleep)
        tag = m.replace("/", "_").replace(".", "_")
        write_agreement(a.out, tag, m)
        for subset, (ok, n, parsed) in sorted(agg.items()):
            acc = ok / n if n else float("nan")
            parse_rate = parsed / n if n else float("nan")
            out_rows.append([m, subset, ok, n, round(acc, 4), round(parse_rate, 4),
                             a.backend, "generative"])
            print(f"  {subset:34} {ok:4}/{n:<4} acc={acc:.3f}  parsed={parse_rate:.3f}",
                  flush=True)

    keep = [list(v.values()) for v in existing.values()] if existing else []
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "subset", "n_correct", "n_items", "accuracy",
                    "parse_rate", "backend", "method"])
        seen = set()
        for r in out_rows:
            seen.add((r[0], r[1]))
            w.writerow(r)
        for r in keep:
            if (r[0], r[1]) not in seen:
                w.writerow(r)
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
