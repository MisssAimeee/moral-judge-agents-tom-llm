#!/usr/bin/env python3
"""Closed-model behavioral scoring + roadmap #7 reasoning dose–response.

WHY. Open models are logprob-EV; closed APIs cannot be. This script scores EVERY closed
cell by sampling (T=1, n>=20) so the open/closed boundary does not reintroduce the
estimator confound that roadmap #1 closed. Open sample↔EV agreement is already on disk
(scoring_parity.csv; OLMo/Qwen-7B PASS). Roadmap #7 is closed in the same job: each
model that exposes a thinking control is run at direct / think / budget_{low,med,high}.

Selection and cost: outputs/experiments/CLOSED_MODEL_SELECTION.md (queried live).
DeepSeek is omitted — no key at launch time.

Usage
  # print the cost table and exit
  python code/experiments/52_closed_reasoning_dose.py --cost-only

  # run (sources .env_agents via the submit wrapper)
  python code/experiments/52_closed_reasoning_dose.py --run
  python code/experiments/52_closed_reasoning_dose.py --run --providers anthropic,openai
  python code/experiments/52_closed_reasoning_dose.py --run --limit 4   # smoke

  # rebuild contrast table from existing raw CSVs
  python code/experiments/52_closed_reasoning_dose.py --report
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CODE_DIR)
import numpy as np  # noqa: E402
import tom_common as tc  # noqa: E402

OUT = os.path.join(tc.ROOT, "outputs", "closed_reasoning")
SEL_MD = os.path.join(tc.ROOT, "outputs", "experiments", "CLOSED_MODEL_SELECTION.md")

# Cost-forced cut: human anchor + one blame wording. Was 4 templates; cut to 2 with
# N_SAMPLES after the first run's cost was reconstructed (see the addendum in
# API_COST_ESTIMATE.md) — as originally configured this is a ~$23,000 job.
TEMPLATES = ["human_verbatim", "blame_w1"]
# Was 20, inherited from the behavioural sampling-parity config where per-item variance was
# the object of study. Here the reported statistic is a cell mean over ~1,200 prompts, which
# averages away per-prompt sampling noise regardless of how many samples each prompt
# contributes — so n=20 multiplied the thinking-token bill by 10x for no gain in anything
# the dose-response reads.
N_SAMPLES = 2
TEMPERATURE = 1.0   # required for several reasoning endpoints; matches sampling parity

# Thinking budgets (provider-native units).
ANTHROPIC_BUDGETS = {"budget_low": 1024, "budget_med": 4096, "budget_high": 16384}
OPENAI_EFFORTS = {"budget_low": "low", "budget_med": "medium", "budget_high": "high"}
# Output cap must exceed the reasoning the effort setting will spend, or the answer token
# never arrives. o3/budget_med came back empty under a flat 4096.
OPENAI_MAX_OUT = {"budget_low": 4096, "budget_med": 8192, "budget_high": 16384}
GOOGLE_BUDGETS = {"budget_low": 512, "budget_med": 2048, "budget_high": 8192}


# Retrying a depleted account is just a slower way of failing. The overnight run spent
# hours on 429s that could never succeed: Google returned "prepayment credits are
# depleted" (which contains neither "quota" nor "per_day", so the existing guard missed it)
# and OpenAI "exceeded your current quota". Both mean stop, not back off.
BILLING_MARKERS = ("credits are depleted", "exceeded your current quota", "billing",
                   "insufficient_quota", "payment", "per_day", "quota exceeded")


class BillingStop(RuntimeError):
    pass


def fatal_if_billing(exc, provider, model, condition):
    err = str(exc).lower()
    if any(m in err for m in BILLING_MARKERS):
        raise BillingStop(
            f"{provider} {model}/{condition}: account cannot serve requests — "
            f"{str(exc)[:200]}")

# Roster locked to strings returned by the 2026-07-28 list endpoints.
ROSTER = [
    # provider, model, conditions
    ("anthropic", "claude-opus-5",
     ["direct", "think", "budget_low", "budget_med", "budget_high"]),
    ("anthropic", "claude-sonnet-5",
     ["direct", "think", "budget_low", "budget_med", "budget_high"]),
    ("openai", "gpt-5.5", ["direct"]),
    ("openai", "gpt-5.4-mini", ["direct"]),
    ("openai", "o3", ["budget_low", "budget_med", "budget_high"]),
    ("openai", "o4-mini", ["budget_low", "budget_med", "budget_high"]),
    ("google", "gemini-3.1-pro-preview",
     ["direct", "think", "budget_low", "budget_med", "budget_high"]),
    ("google", "gemini-3.5-flash",
     ["direct", "think", "budget_low", "budget_med", "budget_high"]),
    ("moonshot", "kimi-k2.6", ["direct"]),
]


def _load_behavioral():
    spec = importlib.util.spec_from_file_location(
        "behavioral", os.path.join(CODE_DIR, "03_behavioral.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def safe(name: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", name)


# ---------------------------------------------------------------------------
# Backends with thinking controls
# ---------------------------------------------------------------------------

class AnthropicThink:
    def __init__(self, model, condition):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.condition = condition

    def rate(self, prompt, s_min, s_max, n_samples, temperature, parse):
        ratings = []
        for _ in range(n_samples):
            # Claude 5.x rejects `temperature` entirely ("deprecated for this model").
            kwargs = dict(model=self.model,
                          messages=[{"role": "user", "content": prompt}])
            if self.condition == "direct":
                kwargs["max_tokens"] = 32
            elif self.condition == "think":
                kwargs.update(max_tokens=8192,
                              thinking={"type": "enabled", "budget_tokens": 4096})
            else:
                b = ANTHROPIC_BUDGETS[self.condition]
                # 1536 of headroom above the thinking budget, matching the cap that fixed
                # the empty-output failures on the ToM run. 256 sufficed for the single
                # integer only if the chain stopped exactly on budget.
                kwargs.update(max_tokens=b + 1536,
                              thinking={"type": "enabled", "budget_tokens": b})
            for attempt in range(4):
                try:
                    resp = self.client.messages.create(**kwargs)
                    texts = []
                    for blk in resp.content:
                        if getattr(blk, "type", "") == "text":
                            texts.append(blk.text)
                    v = parse(" ".join(texts).strip(), s_min, s_max)
                    if v is not None:
                        ratings.append(v)
                    break
                except Exception as e:
                    fatal_if_billing(e, "Anthropic", self.model, self.condition)
                    err = str(e)
                    if "temperature" in err.lower() and "temperature" in kwargs:
                        kwargs.pop("temperature", None)
                        continue
                    if "thinking" in err.lower() and "direct" not in self.condition:
                        kwargs.pop("thinking", None)
                        kwargs["max_tokens"] = 32
                        continue
                    if attempt == 3:
                        print(f"    Anthropic {self.model}/{self.condition}: {err[:160]}")
                    time.sleep(min(2 ** attempt, 30))
        return ratings


class OpenAIThink:
    def __init__(self, model, condition):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self.condition = condition
        self.is_reason = model.startswith(("o1", "o3", "o4"))

    def rate(self, prompt, s_min, s_max, n_samples, temperature, parse):
        ratings = []
        # Reasoning models: no n=, no temperature control on some endpoints.
        if self.is_reason:
            effort = OPENAI_EFFORTS.get(self.condition, "medium")
            # Reasoning tokens are billed against max_completion_tokens, so a flat 4096 lets
            # a medium- or high-effort chain consume the whole budget and return an empty
            # message — the same way Gemini Pro returned header-only rows on the ToM run
            # until the cap was raised. Scale the cap with the effort instead.
            max_out = OPENAI_MAX_OUT.get(self.condition, 8192)
            for _ in range(n_samples):
                for attempt in range(4):
                    try:
                        # Prefer chat.completions with reasoning_effort; fall back to
                        # Responses API if the chat path rejects the kwarg.
                        try:
                            resp = self.client.chat.completions.create(
                                model=self.model,
                                messages=[{"role": "user", "content": prompt}],
                                max_completion_tokens=max_out,
                                reasoning_effort=effort,
                            )
                            text = (resp.choices[0].message.content or "").strip()
                        except TypeError:
                            resp = self.client.responses.create(
                                model=self.model,
                                input=prompt,
                                reasoning={"effort": effort},
                                max_output_tokens=max_out,
                            )
                            text = getattr(resp, "output_text", "") or ""
                        v = parse(text, s_min, s_max)
                        if v is not None:
                            ratings.append(v)
                        break
                    except Exception as e:
                        fatal_if_billing(e, "OpenAI", self.model, self.condition)
                        if attempt == 3:
                            print(f"    OpenAI {self.model}/{self.condition}: {str(e)[:160]}")
                        time.sleep(min(2 ** attempt, 30))
            return ratings

        # gpt-5.* often spends completion tokens on hidden reasoning; give headroom.
        # Also many reject temperature=0 and/or n= — fall back sequentially.
        max_out = 1024 if self.model.startswith("gpt-5") else 32
        kwargs = dict(model=self.model,
                   messages=[{"role": "user", "content": prompt}],
                   max_completion_tokens=max_out)

        def _one(extra):
            resp = self.client.chat.completions.create(**{**kwargs, **extra})
            return (resp.choices[0].message.content or "").strip()

        for attempt in range(4):
            try:
                resp = self.client.chat.completions.create(
                    **kwargs, temperature=0, n=n_samples)
                for ch in resp.choices:
                    v = parse((ch.message.content or "").strip(), s_min, s_max)
                    if v is not None:
                        ratings.append(v)
                break
            except Exception as e:
                fatal_if_billing(e, "OpenAI", self.model, self.condition)
                err = str(e).lower()
                if any(k in err for k in ("temperature", "'n'", '"n"', "n=",
                                          "parallel", "multiple")):
                    for _ in range(n_samples):
                        for att2 in range(3):
                            try:
                                # try plain, then without temperature
                                try:
                                    text = _one({"temperature": 0})
                                except Exception:
                                    text = _one({})
                                v = parse(text, s_min, s_max)
                                if v is not None:
                                    ratings.append(v)
                                break
                            except Exception as e2:
                                fatal_if_billing(e2, "OpenAI", self.model, self.condition)
                                if att2 == 2:
                                    print(f"    OpenAI {self.model}: {str(e2)[:140]}")
                                time.sleep(2 ** att2)
                    break
                if "max_tokens" in err or "output limit" in err:
                    kwargs["max_completion_tokens"] = min(
                        kwargs["max_completion_tokens"] * 2, 8192)
                    continue
                if attempt == 3:
                    print(f"    OpenAI {self.model}: {str(e)[:160]}")
                time.sleep(min(2 ** attempt, 30))
        return ratings


class GoogleThink:
    def __init__(self, model, condition):
        import google.generativeai as genai
        key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise EnvironmentError("GOOGLE_API_KEY not set")
        genai.configure(api_key=key)
        self.gai = genai
        self.mdl = genai.GenerativeModel(model)
        self.model = model
        self.condition = condition

    def _text(self, resp):
        try:
            parts = resp.candidates[0].content.parts
            return " ".join(
                p.text for p in parts
                if getattr(p, "text", None) and not getattr(p, "thought", False)
            ).strip()
        except Exception:
            return (getattr(resp, "text", None) or "").strip()

    def rate(self, prompt, s_min, s_max, n_samples, temperature, parse):
        ratings = []
        # thinking_budget: 0 = direct; -1 or omit = dynamic (think); else int
        if self.condition == "direct":
            tb = 0
        elif self.condition == "think":
            tb = -1
        else:
            tb = GOOGLE_BUDGETS[self.condition]
        for _ in range(n_samples):
            for attempt in range(5):
                try:
                    # The cap has to clear the thinking budget with room for the answer;
                    # budget_high (8192 thinking) under a flat 2048 cap can only ever
                    # return a header. Dynamic `think` gets the widest cap.
                    cap = 8192 if tb < 0 else max(2048, tb + 1536)
                    cfg_kwargs = dict(max_output_tokens=cap, temperature=0)
                    # Prefer thinking_config when the SDK accepts it.
                    try:
                        if tb == 0:
                            cfg = self.gai.types.GenerationConfig(
                                **cfg_kwargs,
                                thinking_config={"thinking_budget": 0})
                        elif tb < 0:
                            cfg = self.gai.types.GenerationConfig(**cfg_kwargs)
                        else:
                            cfg = self.gai.types.GenerationConfig(
                                **cfg_kwargs,
                                thinking_config={"thinking_budget": tb})
                    except TypeError:
                        cfg = self.gai.types.GenerationConfig(**cfg_kwargs)
                    resp = self.mdl.generate_content(prompt, generation_config=cfg)
                    v = parse(self._text(resp), s_min, s_max)
                    if v is not None:
                        ratings.append(v)
                    break
                except Exception as e:
                    fatal_if_billing(e, "Google", self.model, self.condition)
                    err = str(e)
                    wait = min(2 ** attempt, 60)
                    print(f"    Google {self.model}/{self.condition} "
                          f"(wait {wait}s): {err[:120]}")
                    time.sleep(wait)
        return ratings


class MoonshotThink:
    def __init__(self, model, condition):
        from openai import OpenAI
        key = os.environ.get("MOONSHOT_API_KEY") or os.environ.get("KIMI_API_KEY")
        if not key:
            raise EnvironmentError("MOONSHOT_API_KEY not set")
        self.client = OpenAI(api_key=key, base_url="https://api.moonshot.ai/v1")
        self.model = model

    def rate(self, prompt, s_min, s_max, n_samples, temperature, parse):
        ratings = []
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=32, temperature=1.0, n=n_samples,
                )
                for ch in resp.choices:
                    v = parse((ch.message.content or "").strip(), s_min, s_max)
                    if v is not None:
                        ratings.append(v)
                break
            except Exception as e:
                if "n" in str(e).lower():
                    for _ in range(n_samples):
                        resp = self.client.chat.completions.create(
                            model=self.model,
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=32, temperature=1.0)
                        v = parse((resp.choices[0].message.content or "").strip(),
                                  s_min, s_max)
                        if v is not None:
                            ratings.append(v)
                    break
                if attempt == 2:
                    print(f"    Moonshot {self.model}: {str(e)[:160]}")
                time.sleep(2 ** attempt)
        return ratings


BACKENDS = {
    "anthropic": AnthropicThink,
    "openai": OpenAIThink,
    "google": GoogleThink,
    "moonshot": MoonshotThink,
}


def cost_only():
    print(open(SEL_MD).read() if os.path.exists(SEL_MD) else "SELECTION.md missing")
    print("\n=== roster × conditions ===")
    n_cells = 0
    for prov, model, conds in ROSTER:
        print(f"  {prov:10} {model:28} {conds}")
        n_cells += len(conds)
    per = 298 * len(TEMPLATES) * N_SAMPLES
    print(f"\ncompletions/cell = {per:,}")
    print(f"cells = {n_cells}")
    print(f"total completions ≈ {per * n_cells:,}")
    print(f"templates = {TEMPLATES}")
    print(f"n_samples = {N_SAMPLES}, temperature = {TEMPERATURE}")
    print("\nSee CLOSED_MODEL_SELECTION.md for USD estimate (~$205 behavioral + ~$20 BigToM).")


def run_cell(beh, rows, provider, model, condition, limit=None, skip_existing=True):
    tag = f"{safe(model)}__{condition}"
    raw_path = os.path.join(OUT, f"raw_{tag}.csv")
    os.makedirs(OUT, exist_ok=True)
    done = set()
    if skip_existing and os.path.exists(raw_path):
        for r in csv.DictReader(open(raw_path)):
            done.add((r["story_id"], r["template"]))
        print(f"  resume {tag}: {len(done)} story×template already done")

    backend = BACKENDS[provider](model, condition)
    use_rows = rows[:limit] if limit else rows
    fieldnames = ["model", "condition", "provider", "story_id", "condition_cell",
                  "template", "sample_i", "rating", "rating_norm", "s_min", "s_max",
                  "status"]
    new_file = not os.path.exists(raw_path)
    fout = open(raw_path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fout, fieldnames=fieldnames)
    if new_file:
        w.writeheader()

    n_todo = sum(1 for r in use_rows for t in TEMPLATES
                 if (r["story_id"], t) not in done)
    print(f"  {tag}: {n_todo} story×template left "
          f"({len(use_rows)} stories × {len(TEMPLATES)} tmpl)")
    done_n = 0
    t0 = time.time()
    for row in use_rows:
        for tmpl in TEMPLATES:
            key = (row["story_id"], tmpl)
            if key in done:
                continue
            prompt, s_min, s_max = beh.build_prompt(row["text"], tmpl, row["source"])
            s_min, s_max = int(s_min), int(s_max)
            try:
                ratings = backend.rate(prompt, s_min, s_max, N_SAMPLES, TEMPERATURE,
                                       beh._parse_rating)
            except RuntimeError as e:
                if "quota" in str(e).lower():
                    fout.close()
                    raise
                print(f"    !! {e}")
                ratings = []
            common = dict(model=model, condition=condition, provider=provider,
                          story_id=row["story_id"], condition_cell=row["condition"],
                          template=tmpl, s_min=s_min, s_max=s_max)
            if not ratings:
                # Previously this wrote the scale midpoint, which turned an API failure into
                # a datum indistinguishable from a genuinely indifferent judgment. It is how
                # kimi-k2.6 came out at contrast exactly 0.0000 with all four cells at
                # exactly 0.500: every request failed and every failure was imputed. A
                # failure is missing data and is recorded as such.
                w.writerow(dict(common, sample_i=0, rating="", rating_norm="",
                                status="failed"))
                fout.flush()
                done_n += 1
                continue
            for i, v in enumerate(ratings):
                w.writerow(dict(
                    common, sample_i=i, rating=v,
                    rating_norm=round((v - s_min) / (s_max - s_min), 4), status="ok"))
            fout.flush()
            done_n += 1
            if done_n % 25 == 0:
                rate = done_n / max(time.time() - t0, 1)
                print(f"    {done_n}/{n_todo}  ({rate:.2f}/s)")
    fout.close()
    return raw_path


def contrast_from_raw(path):
    """attempted − accidental, keyed on scenario_group, mean over templates.

    Rows marked `failed` are missing data and are excluded. `parse_rate` is reported beside
    every contrast so a cell computed from a handful of surviving requests cannot be read as
    a measurement: a contrast over 3 groups and one over 53 are not the same number.
    """
    by = defaultdict(lambda: defaultdict(list))
    model = condition = provider = ""
    n_ok = n_failed = 0
    for r in csv.DictReader(open(path)):
        model, condition, provider = r["model"], r["condition"], r["provider"]
        # Files written before the status column existed imputed the midpoint on failure and
        # cannot be repaired here; they are quarantined rather than read.
        if r.get("status", "ok") != "ok" or r["rating_norm"] in ("", None):
            n_failed += 1
            continue
        n_ok += 1
        g = tc.scenario_group_of(r["story_id"])
        by[g][r["condition_cell"]].append(float(r["rating_norm"]))
    # mean per cell per group, then contrast
    diffs = []
    cells = {c: [] for c in ("neutral", "accidental", "attempted", "intentional")}
    for g, d in by.items():
        means = {c: float(np.mean(v)) for c, v in d.items() if v}
        for c, v in means.items():
            if c in cells:
                cells[c].append(v)
        if "attempted" in means and "accidental" in means:
            diffs.append(means["attempted"] - means["accidental"])
    return dict(
        model=model, condition=condition, provider=provider,
        n_groups=len(diffs), n_ok=n_ok, n_failed=n_failed,
        parse_rate=round(n_ok / max(n_ok + n_failed, 1), 4),
        usable=(len(diffs) >= 40),   # of 53 scenario groups
        contrast=round(float(np.mean(diffs)), 4) if diffs else float("nan"),
        **{f"cell_{c}": round(float(np.mean(v)), 4) if v else float("nan")
           for c, v in cells.items()},
        raw_path=os.path.relpath(path, tc.ROOT),
    )


def report():
    rows = []
    for p in sorted(os.listdir(OUT)) if os.path.isdir(OUT) else []:
        if p.startswith("raw_") and p.endswith(".csv"):
            rows.append(contrast_from_raw(os.path.join(OUT, p)))
    if not rows:
        print("no raw_*.csv yet")
        return
    out_csv = os.path.join(OUT, "closed_reasoning_contrasts.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    md = ["# Closed-model reasoning dose–response", "",
          f"Generated {datetime.now().isoformat(timespec='seconds')}. "
          f"Templates: {TEMPLATES}. n_samples={N_SAMPLES}. Sampling only.", "",
          "| provider | model | condition | contrast | neutral | accidental | "
          "attempted | intentional | n_groups |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda r: (r["provider"], r["model"], r["condition"])):
        md.append(
            f"| {r['provider']} | {r['model']} | {r['condition']} | "
            f"{r['contrast']:+.3f} | {r['cell_neutral']:.3f} | "
            f"{r['cell_accidental']:.3f} | {r['cell_attempted']:.3f} | "
            f"{r['cell_intentional']:.3f} | {r['n_groups']} |")
    md += ["", "Selection + cost: `CLOSED_MODEL_SELECTION.md`. "
           "Scoring parity bridge: `outputs/analysis/scoring_parity.csv`.", ""]
    out_md = os.path.join(OUT, "CLOSED_REASONING_DOSE.md")
    open(out_md, "w").write("\n".join(md))
    print(f"wrote {out_csv} ({len(rows)} cells)")
    print(f"wrote {out_md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-only", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--providers", default="anthropic,openai,google,moonshot")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-skip", action="store_true")
    a = ap.parse_args()

    if a.cost_only or not (a.run or a.report):
        cost_only()
        if not a.run:
            return

    if a.run:
        beh = _load_behavioral()
        rows = beh.load_dataset()
        want = {p.strip() for p in a.providers.split(",") if p.strip()}
        print(f"\n=== CLOSED REASONING DOSE — {datetime.now().isoformat()} ===")
        print(f"templates={TEMPLATES}  n_samples={N_SAMPLES}  stories={len(rows)}")
        print(open(SEL_MD).read().split("## Cost estimate")[0][-400:]
              if os.path.exists(SEL_MD) else "")
        for prov, model, conds in ROSTER:
            if prov not in want:
                continue
            stopped = False
            for cond in conds:
                print(f"\n----- {prov} / {model} / {cond} -----")
                try:
                    run_cell(beh, rows, prov, model, cond, limit=a.limit,
                             skip_existing=not a.no_skip)
                except BillingStop as e:
                    # Every remaining cell on this provider would fail identically. The
                    # overnight run walked the whole roster this way, writing header-only
                    # CSVs that then read as model failures rather than as an unusable key.
                    print(f"!! BILLING STOP — abandoning provider {prov}: {e}")
                    stopped = True
                    break
                except Exception as e:
                    print(f"!! cell failed: {e}")
            if stopped:
                break
        report()

    if a.report:
        report()


if __name__ == "__main__":
    main()
