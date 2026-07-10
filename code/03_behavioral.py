#!/usr/bin/env python3
"""
03_behavioral.py  --  Elicit moral blame ratings from LLMs on the Saxelab
intent x outcome vignettes, then compute per-model intent-reliance statistics.

Backends
  --backend hf        HuggingFace transformers (open-weight, local GPU).
  --backend openai    OpenAI Chat API (GPT-4o, o1, ...).  Needs OPENAI_API_KEY.
  --backend anthropic Anthropic API (Claude ...).          Needs ANTHROPIC_API_KEY.
  --backend google    Google Gemini API.                   Needs GOOGLE_API_KEY.
  --backend mistral   Mistral API (Mistral-Large ...).     Needs MISTRAL_API_KEY.
  --backend together  Together AI (Llama 70B/405B ...).   Needs TOGETHER_API_KEY.
  --backend mock      Deterministic fake for smoke tests.

Scoring modes
  --scoring logprob   (HF only) E[rating]=sum p(tok=k)*k in one forward pass.
  --scoring sampling  Parse a number from model text; average n_samples runs.
"""
import argparse, csv, os, random, re, time
from collections import defaultdict
from pathlib import Path

MASTER_CSV = "dataset/master/moral_2x2_master.csv"
OUT_DIR    = Path("outputs/behavior")
CELLS      = ["neutral", "accidental", "attempted", "intentional"]
COND_MAP   = {"neutral":(0,0),"accidental":(0,1),"attempted":(1,0),"intentional":(1,1)}

SOURCE_SCALES = {
    "YS2008": (1, 3, "permissibility"),
    "YS2009": (1, 4, "blame"),
    "YS2011": (1, 7, "wrongness"),
    "KPH":    (1, 7, "wrongness"),
    "SW":     (1, 7, "wrongness"),
    "SK":     (1, 7, "wrongness"),
    "BP":     (1, 7, "wrongness"),
    "DEFAULT":(1, 7, "wrongness"),
}

SCALE_QUESTIONS = {
    "permissibility": (
        "How permissible was {agent}'s action? "
        "Rate from 1 (completely permissible) to 3 (completely impermissible). "
        "Respond with a single integer."
    ),
    "blame": (
        "How much blame does {agent} deserve? "
        "Rate from 1 (no blame) to 4 (full blame). "
        "Respond with a single integer."
    ),
    "wrongness": (
        "How morally wrong was {agent}'s action? "
        "Rate from 1 (not at all wrong) to 7 (extremely wrong). "
        "Respond with a single integer."
    ),
}

PARA_TEMPLATES = {
    "para_blame10":  ("How much moral blame does the agent deserve? Rate 1-10. Single integer only.", 1, 10),
    "para_wrong7":   ("How morally wrong was the agent's action? Rate 1 (not wrong) to 7 (extremely wrong). Single integer only.", 1, 7),
    "para_blame4":   ("How much blame? 1=none 2=slight 3=moderate 4=full. Single integer only.", 1, 4),
    "punish7":       ("How severely should the agent be punished? Rate 1-7. Single integer only.", 1, 7),
    "acceptable7":   ("How acceptable was the action? 1=acceptable to 7=unacceptable. Single integer only.", 1, 7),
    "persona_adult7":("As a thoughtful adult: how morally wrong was the agent's action? Rate 1-7. Single integer only.", 1, 7),
}

ALL_TEMPLATES = ["human_verbatim"] + list(PARA_TEMPLATES.keys())


def load_dataset(limit=None):
    rows = []
    with open(MASTER_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
            if limit and len(rows) >= limit:
                break
    return rows


def get_scale(source):
    for key in SOURCE_SCALES:
        if source.startswith(key):
            return SOURCE_SCALES[key]
    return SOURCE_SCALES["DEFAULT"]


def normalize(raw, s_min, s_max):
    return (raw - s_min) / (s_max - s_min)


def extract_agent(text):
    m = re.search(r'\b([A-Z][a-z]{2,})\b', text)
    return m.group(1) if m else "the agent"


def build_prompt(story_text, template, source):
    agent = extract_agent(story_text)
    if template == "human_verbatim":
        s_min, s_max, q_key = get_scale(source)
        q = SCALE_QUESTIONS[q_key].format(agent=agent)
        return story_text.strip() + "\n\n" + q, s_min, s_max
    if template in PARA_TEMPLATES:
        q, s_min, s_max = PARA_TEMPLATES[template]
        return story_text.strip() + "\n\n" + q, s_min, s_max
    raise ValueError(f"Unknown template: {template}")


def model_safe(name):
    return re.sub(r'[^\w\-]', '_', name)


class MockBackend:
    def __init__(self, model_name, scoring, **_):
        self.model_name = model_name
        self.scoring = scoring

    def rate(self, prompt, s_min, s_max, n_samples=1, temperature=0.7):
        random.seed(abs(hash(prompt)) % (2**31))
        if self.scoring == "logprob":
            val = random.uniform(s_min, s_max)
            return [round(val, 4)], round(normalize(val, s_min, s_max), 4)
        ratings = [random.randint(s_min, s_max) for _ in range(n_samples)]
        avg = sum(normalize(r, s_min, s_max) for r in ratings) / len(ratings)
        return ratings, round(avg, 4)


class HFBackend:
    def __init__(self, model_name, scoring, **_):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.model_name = model_name
        self.scoring = scoring
        self.torch = torch
        print(f"  Loading {model_name} ...")
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.mdl = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True
        )
        self.mdl.eval()
        print(f"  Loaded on {next(self.mdl.parameters()).device}")
        self._dig = {}
        for d in range(1, 11):
            toks = self.tok.encode(str(d), add_special_tokens=False)
            if toks:
                self._dig[d] = toks[0]

    def _fmt(self, prompt):
        if self.tok.chat_template:
            return self.tok.apply_chat_template(
                [{"role":"user","content":prompt}], tokenize=False, add_generation_prompt=True
            )
        return prompt

    def rate(self, prompt, s_min, s_max, n_samples=1, temperature=0.7):
        import torch
        inp = self.tok(self._fmt(prompt), return_tensors="pt").to(self.mdl.device)
        if self.scoring == "logprob":
            with torch.no_grad():
                logits = self.mdl(**inp).logits[0, -1, :]
            probs, vals = [], []
            for d in range(int(s_min), int(s_max)+1):
                if d in self._dig:
                    probs.append(logits[self._dig[d]].item())
                    vals.append(d)
            if not probs:
                mid = (s_min + s_max) / 2
                return [mid], normalize(mid, s_min, s_max)
            lp = torch.tensor(probs)
            p = torch.softmax(lp, dim=0).tolist()
            exp = sum(p[i] * vals[i] for i in range(len(vals)))
            return [round(exp, 4)], round(normalize(exp, s_min, s_max), 4)
        ratings = []
        for _ in range(n_samples):
            with torch.no_grad():
                gen = self.mdl.generate(
                    **inp, max_new_tokens=8, do_sample=True,
                    temperature=temperature, pad_token_id=self.tok.eos_token_id
                )
            dec = self.tok.decode(gen[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
            m = re.search(r'\b(\d+(?:\.\d+)?)\b', dec.strip())
            if m:
                ratings.append(max(s_min, min(s_max, float(m.group(1)))))
        if not ratings:
            ratings = [(s_min + s_max) / 2]
        avg = sum(normalize(r, s_min, s_max) for r in ratings) / len(ratings)
        return ratings, round(avg, 4)


class OpenAIBackend:
    def __init__(self, model_name, scoring, **_):
        from openai import OpenAI
        self.model_name = model_name
        self.client = OpenAI()

    def rate(self, prompt, s_min, s_max, n_samples=5, temperature=0.0):
        ratings = []
        for _ in range(n_samples):
            for attempt in range(3):
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role":"user","content":prompt}],
                        max_tokens=16, temperature=temperature,
                    )
                    text = resp.choices[0].message.content.strip()
                    m = re.search(r'\b(\d+(?:\.\d+)?)\b', text)
                    if m:
                        ratings.append(max(s_min, min(s_max, float(m.group(1)))))
                    break
                except Exception as e:
                    if attempt == 2: print(f"    OpenAI error: {e}")
                    time.sleep(2 ** attempt)
        if not ratings: ratings = [(s_min+s_max)/2]
        return ratings, round(sum(normalize(r,s_min,s_max) for r in ratings)/len(ratings), 4)


class AnthropicBackend:
    def __init__(self, model_name, scoring, **_):
        import anthropic
        self.model_name = model_name
        self.client = anthropic.Anthropic()

    def rate(self, prompt, s_min, s_max, n_samples=5, temperature=0.0):
        ratings = []
        for _ in range(n_samples):
            for attempt in range(3):
                try:
                    resp = self.client.messages.create(
                        model=self.model_name, max_tokens=16,
                        messages=[{"role":"user","content":prompt}],
                    )
                    text = resp.content[0].text.strip()
                    m = re.search(r'\b(\d+(?:\.\d+)?)\b', text)
                    if m:
                        ratings.append(max(s_min, min(s_max, float(m.group(1)))))
                    break
                except Exception as e:
                    if attempt == 2: print(f"    Anthropic error: {e}")
                    time.sleep(2 ** attempt)
        if not ratings: ratings = [(s_min+s_max)/2]
        return ratings, round(sum(normalize(r,s_min,s_max) for r in ratings)/len(ratings), 4)


class GoogleBackend:
    def __init__(self, model_name, scoring, **_):
        import google.generativeai as genai
        key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key: raise EnvironmentError("Set GOOGLE_API_KEY or GEMINI_API_KEY")
        genai.configure(api_key=key)
        self.gai = genai
        self.mdl = genai.GenerativeModel(model_name)
        self.model_name = model_name

    def _extract_text(self, response):
        """Extract answer text, skipping internal thinking parts (Gemini 2.5+)."""
        try:
            parts = response.candidates[0].content.parts
            # Filter out thought-only parts (Gemini 2.5 thinking models)
            text_parts = [p.text for p in parts
                          if hasattr(p, "text") and not getattr(p, "thought", False)]
            return " ".join(text_parts).strip()
        except (IndexError, AttributeError):
            pass
        try:
            return response.text.strip()
        except Exception:
            return ""

    def rate(self, prompt, s_min, s_max, n_samples=5, temperature=0.0):
        # Use a large token budget so thinking models have room to reason then answer.
        cfg = self.gai.types.GenerationConfig(max_output_tokens=1024, temperature=temperature)
        ratings = []
        for _ in range(n_samples):
            for attempt in range(5):
                try:
                    resp = self.mdl.generate_content(prompt, generation_config=cfg)
                    text = self._extract_text(resp)
                    m = re.search(r'\b(\d+(?:\.\d+)?)\b', text)
                    if m:
                        ratings.append(max(s_min, min(s_max, float(m.group(1)))))
                    break
                except Exception as e:
                    err_str = str(e)
                    # Daily quota exhausted — raise immediately so fallback values are NOT used.
                    if "per_day" in err_str or "per_model_per_day" in err_str:
                        raise RuntimeError(
                            f"Daily quota exhausted for {self.model_name}. "
                            "Resubmit after midnight UTC when quota resets."
                        ) from e
                    # Rate-limit (per minute/second): honour the retry-after hint.
                    wait = 2 ** attempt
                    import re as _re
                    m_delay = _re.search(r'retry[_ ](?:in|after)[^\d]*(\d+)', err_str)
                    if m_delay:
                        wait = min(int(m_delay.group(1)) + 5, 400)
                    print(f"    Google error (attempt {attempt+1}/5, wait {wait}s): {err_str[:120]}")
                    time.sleep(wait)
        if not ratings: ratings = [(s_min+s_max)/2]
        return ratings, round(sum(normalize(r,s_min,s_max) for r in ratings)/len(ratings), 4)


class MistralBackend:
    def __init__(self, model_name, scoring, **_):
        from mistralai import Mistral
        key = os.environ.get("MISTRAL_API_KEY")
        if not key: raise EnvironmentError("Set MISTRAL_API_KEY")
        self.client = Mistral(api_key=key)
        self.model_name = model_name

    def rate(self, prompt, s_min, s_max, n_samples=5, temperature=0.0):
        ratings = []
        for _ in range(n_samples):
            for attempt in range(3):
                try:
                    resp = self.client.chat.complete(
                        model=self.model_name,
                        messages=[{"role":"user","content":prompt}],
                        max_tokens=16, temperature=temperature,
                    )
                    text = resp.choices[0].message.content.strip()
                    m = re.search(r'\b(\d+(?:\.\d+)?)\b', text)
                    if m:
                        ratings.append(max(s_min, min(s_max, float(m.group(1)))))
                    break
                except Exception as e:
                    if attempt == 2: print(f"    Mistral error: {e}")
                    time.sleep(2 ** attempt)
        if not ratings: ratings = [(s_min+s_max)/2]
        return ratings, round(sum(normalize(r,s_min,s_max) for r in ratings)/len(ratings), 4)


class TogetherBackend:
    def __init__(self, model_name, scoring, **_):
        from openai import OpenAI
        key = os.environ.get("TOGETHER_API_KEY")
        if not key: raise EnvironmentError("Set TOGETHER_API_KEY")
        self.client = OpenAI(api_key=key, base_url="https://api.together.xyz/v1")
        self.model_name = model_name

    def rate(self, prompt, s_min, s_max, n_samples=5, temperature=0.0):
        ratings = []
        for _ in range(n_samples):
            for attempt in range(3):
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role":"user","content":prompt}],
                        max_tokens=16, temperature=temperature,
                    )
                    text = resp.choices[0].message.content.strip()
                    m = re.search(r'\b(\d+(?:\.\d+)?)\b', text)
                    if m:
                        ratings.append(max(s_min, min(s_max, float(m.group(1)))))
                    break
                except Exception as e:
                    if attempt == 2: print(f"    Together error: {e}")
                    time.sleep(2 ** attempt)
        if not ratings: ratings = [(s_min+s_max)/2]
        return ratings, round(sum(normalize(r,s_min,s_max) for r in ratings)/len(ratings), 4)


BACKENDS = {
    "mock": MockBackend, "hf": HFBackend,
    "openai": OpenAIBackend, "anthropic": AnthropicBackend,
    "google": GoogleBackend, "mistral": MistralBackend, "together": TogetherBackend,
}


def ols_2x2(cell_means):
    X, y = [], []
    for cond, (intent, outcome) in COND_MAP.items():
        if cond in cell_means and cell_means[cond] is not None:
            X.append([1.0, float(intent), float(outcome)])
            y.append(cell_means[cond])
    if len(y) < 3:
        return None, None, None
    n, k = len(X), 3
    XtX = [[sum(X[i][a]*X[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
    Xty = [sum(X[i][a]*y[i] for i in range(n)) for a in range(k)]
    def det3(m):
        return (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
               -m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
               +m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))
    d = det3(XtX)
    if abs(d) < 1e-12: return None, None, None
    inv = [[0.0]*k for _ in range(k)]
    for r in range(k):
        for c in range(k):
            minor = [[XtX[i][j] for j in range(k) if j!=c] for i in range(k) if i!=r]
            cof = ((-1)**(r+c)) * (minor[0][0]*minor[1][1] - minor[0][1]*minor[1][0])
            inv[c][r] = cof / d
    beta = [sum(inv[r][c]*Xty[c] for c in range(k)) for r in range(k)]
    b_i, b_o = beta[1], beta[2]
    denom = abs(b_i) + abs(b_o)
    iri = abs(b_i)/denom if denom > 1e-9 else 0.5
    return round(b_i,4), round(b_o,4), round(iri,4)


RAW_FIELDS = ["model","template","sample","story_id","source","condition",
              "intent_label","outcome_label","raw_rating","norm_rating"]

def run_model(backend, rows, templates, n_samples, temperature, raw_path=None):
    """Rate all stories and write rows incrementally so a crashed run can resume."""
    # Load already-completed (template, story_id) pairs from a partial file.
    done_keys = set()
    existing_rows = []
    if raw_path and Path(raw_path).exists():
        try:
            for r in csv.DictReader(open(raw_path, encoding="utf-8")):
                done_keys.add((r["template"], r["story_id"]))
                existing_rows.append(r)
        except Exception:
            pass

    total = len(rows) * len(templates)
    done = len(done_keys) * n_samples   # already-finished story×template combos
    skipped = len(done_keys)

    if skipped:
        print(f"  Resuming: {skipped}/{len(rows)*len(templates)} story×template already done.")

    # Open raw_path in append mode so each completed story is flushed immediately.
    out_f = None
    writer = None
    if raw_path:
        need_header = not Path(raw_path).exists() or Path(raw_path).stat().st_size == 0
        out_f = open(raw_path, "a", newline="", encoding="utf-8")
        writer = csv.DictWriter(out_f, fieldnames=RAW_FIELDS, extrasaction="ignore")
        if need_header:
            writer.writeheader()
            out_f.flush()

    results = list(existing_rows)
    try:
        for tmpl in templates:
            for row in rows:
                key = (tmpl, row["story_id"])
                if key in done_keys:
                    continue
                prompt, s_min, s_max = build_prompt(row["text"], tmpl, row["source"])
                ratings, norm_avg = backend.rate(prompt, s_min, s_max, n_samples, temperature)
                for i, raw_r in enumerate(ratings):
                    rec = {
                        "model": backend.model_name, "template": tmpl, "sample": i,
                        "story_id": row["story_id"], "source": row["source"],
                        "condition": row["condition"], "intent_label": row["intent_label"],
                        "outcome_label": row["outcome_label"],
                        "raw_rating": round(float(raw_r), 4),
                        "norm_rating": norm_avg,
                    }
                    results.append(rec)
                    if writer:
                        writer.writerow(rec)
                        out_f.flush()
                done_keys.add(key)
                done += n_samples
                if (len(done_keys) - skipped) % 20 == 0:
                    print(f"    {len(done_keys)}/{len(rows)*len(templates)} rated ...", flush=True)
    finally:
        if out_f:
            out_f.flush()
            out_f.close()

    return results


def compute_item_means(raw):
    acc, meta = defaultdict(list), {}
    for r in raw:
        key = (r["template"], r["story_id"])
        acc[key].append(float(r["norm_rating"]))
        meta[key] = r
    out = []
    for (tmpl, sid), vals in sorted(acc.items()):
        r = meta[(tmpl, sid)]
        out.append({
            "template": tmpl, "story_id": sid, "source": r["source"],
            "condition": r["condition"], "intent_label": r["intent_label"],
            "outcome_label": r["outcome_label"],
            "mean_norm_blame": round(sum(vals)/len(vals), 4), "n": len(vals),
        })
    return out


def compute_intent_reliance(item_means):
    by_t = defaultdict(lambda: defaultdict(list))
    for r in item_means:
        by_t[r["template"]][r["condition"]].append(float(r["mean_norm_blame"]))
    out = []
    for tmpl, cd in sorted(by_t.items()):
        cm = {c: (sum(v)/len(v) if v else None) for c, v in cd.items()}
        b_i, b_o, iri = ols_2x2(cm)
        out.append({"template": tmpl, "b_intent": b_i, "b_outcome": b_o, "intent_reliance_index": iri})
    return out


def compute_prompt_invariance(intent_rel):
    iris = [r["intent_reliance_index"] for r in intent_rel if r["intent_reliance_index"] is not None]
    metrics = {}
    if iris:
        metrics["intent_reliance_min"]   = round(min(iris), 4)
        metrics["intent_reliance_max"]   = round(max(iris), 4)
        metrics["intent_reliance_range"] = round(max(iris)-min(iris), 4)
    by_t = {r["template"]: r["intent_reliance_index"] for r in intent_rel}
    ref = "human_verbatim"
    if ref in by_t and by_t[ref] is not None:
        for tmpl, val in sorted(by_t.items()):
            if tmpl != ref and val is not None:
                metrics[f"iri_diff_{ref}__{tmpl}"] = round(val - by_t[ref], 4)
    return [{"metric": k, "value": v} for k, v in sorted(metrics.items())]


def write_csv(path, rows, fieldnames):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {path}  ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend",     default="hf", choices=list(BACKENDS))
    ap.add_argument("--models",      nargs="+", required=True)
    ap.add_argument("--scoring",     default="logprob", choices=["logprob","sampling"])
    ap.add_argument("--n_samples",   type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--templates",   nargs="+", default=ALL_TEMPLATES)
    ap.add_argument("--limit",       type=int, default=None)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--out_dir",     default=None,
                    help="Where to save results. Default outputs/behavior. "
                         "Use e.g. outputs/agents/behavior to keep API/agent runs separate.")
    args = ap.parse_args()

    global OUT_DIR
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)

    if args.backend in ("openai","anthropic","google","mistral","together"):
        if args.scoring == "logprob":
            print("  Note: API backend => switching to --scoring sampling")
            args.scoring = "sampling"
        if args.n_samples < 3:
            print(f"  Note: n_samples={args.n_samples} is low; recommend --n_samples 5+")

    rows = load_dataset(args.limit)
    print(f"Dataset: {len(rows)} stories | Templates: {args.templates}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        safe = model_safe(model_name)
        raw_path = OUT_DIR / f"raw_{safe}.csv"

        # --skip_existing: only skip if ALL story×template combos are present.
        if args.skip_existing and raw_path.exists() and raw_path.stat().st_size > 0:
            try:
                existing = list(csv.DictReader(open(raw_path, encoding="utf-8")))
                expected = len(rows) * len(args.templates)
                done_pairs = len({(r["template"], r["story_id"]) for r in existing})
                if done_pairs >= expected:
                    print(f"\n[SKIP] {model_name}  ({done_pairs}/{expected} story×template done)")
                    continue
                # Partial file — will resume inside run_model
                print(f"\n[RESUME] {model_name}  ({done_pairs}/{expected} done, continuing...)")
            except Exception:
                pass

        print(f"\n{'='*60}\n Model : {model_name}\n Backend: {args.backend} | "
              f"Scoring: {args.scoring} | n_samples: {args.n_samples}\n{'='*60}")
        try:
            backend = BACKENDS[args.backend](model_name, args.scoring)
            # raw_path passed so rows are written incrementally as each story finishes.
            raw = run_model(backend, rows, args.templates, args.n_samples, args.temperature,
                            raw_path=raw_path)
            item_means = compute_item_means(raw)
            intent_rel = compute_intent_reliance(item_means)
            prompt_inv = compute_prompt_invariance(intent_rel)

            # raw CSV already written incrementally; write the derived summaries now.
            print(f"  -> {raw_path}  ({len(raw)} rows)")
            write_csv(OUT_DIR/f"item_means_{safe}.csv", item_means,
                ["template","story_id","source","condition",
                 "intent_label","outcome_label","mean_norm_blame","n"])
            write_csv(OUT_DIR/f"intent_reliance_{safe}.csv", intent_rel,
                ["template","b_intent","b_outcome","intent_reliance_index"])
            write_csv(OUT_DIR/f"prompt_invariance_{safe}.csv", prompt_inv,
                ["metric","value"])

            print("\n  Intent-reliance index per template:")
            for r in intent_rel:
                print(f"    {r['template']:30s}  IRI={r['intent_reliance_index']}")
        except Exception as e:
            print(f"!! {model_name} FAILED: {e}")
            import traceback; traceback.print_exc()

        if args.backend == "hf":
            try:
                import torch, gc
                del backend; gc.collect(); torch.cuda.empty_cache()
            except Exception: pass

    print("\nDone. Next: python code/05_human_comparison.py && python code/06_stats.py")


if __name__ == "__main__":
    main()
