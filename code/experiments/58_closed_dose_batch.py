#!/usr/bin/env python3
"""58_closed_dose_batch.py -- roadmap #7 (reasoning dose-response) over provider Batch APIs.

Run 1 of this experiment was submitted request-by-request at N_SAMPLES=20 over 4 templates
and cost ~$23,000 at list rates, of which the accounts covered a few hundred dollars before
Google and OpenAI cut off (see the addendum in outputs/API_COST_ESTIMATE.md). This runs the
same design through the batch endpoints instead:

  * 50% discount on every provider that offers one (OpenAI, Anthropic, Google);
  * no rate-limit pressure, which is what produced the Anthropic overload 429s;
  * one submission per (model, condition) cell, so a failed cell is re-submittable alone.

Batch jobs are asynchronous with completion windows up to 24h, so submit and collect are
separate subcommands and both are resumable from `batch_manifest.csv`.

    python code/experiments/58_closed_dose_batch.py plan
    python code/experiments/58_closed_dose_batch.py submit [--providers openai anthropic]
    python code/experiments/58_closed_dose_batch.py poll
    python code/experiments/58_closed_dose_batch.py collect
    python code/experiments/58_closed_dose_batch.py sync --providers moonshot

Moonshot has no batch endpoint; `sync` runs those cells through 52's synchronous path.

Nothing here imputes a missing rating. A request that returns no parseable integer is written
with status=failed, because run 1 wrote the scale midpoint instead and thereby reported
kimi-k2.6 as a model with perfectly flat moral judgments on the basis of 56 failed calls.
"""
import argparse, csv, importlib.util, json, os, sys, time
from collections import defaultdict

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)
import tom_common as tc  # noqa: E402


def _load(fname, name):
    path = os.path.join(CODE_DIR, fname) if os.sep not in fname else fname
    if not os.path.exists(path):
        path = os.path.join(CODE_DIR, "experiments", fname)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


dose = _load("52_closed_reasoning_dose.py", "closed_dose")
OUT = dose.OUT
MANIFEST = os.path.join(OUT, "batch_manifest.csv")
REQ_DIR = os.path.join(OUT, "_batch_requests")
BATCHABLE = ("openai", "anthropic", "google")

FIELDS = ["model", "condition", "provider", "story_id", "condition_cell",
          "template", "sample_i", "rating", "rating_norm", "s_min", "s_max", "status"]


# ---------------------------------------------------------------------------
# Request construction. custom_id has to round-trip everything needed to write the
# raw CSV row, since batch results come back unordered and without the prompt.
# ---------------------------------------------------------------------------

def custom_id(story_id, template, sample_i):
    return f"{story_id}|{template}|{sample_i}"


def parse_custom_id(cid):
    story_id, template, sample_i = cid.split("|")
    return story_id, template, int(sample_i)


def cell_requests(beh, rows, provider, model, condition, limit=None):
    """One entry per (story, template, sample) with the provider-native body."""
    use_rows = rows[:limit] if limit else rows
    out = []
    for row in use_rows:
        for tmpl in dose.TEMPLATES:
            prompt, s_min, s_max = beh.build_prompt(row["text"], tmpl, row["source"])
            s_min, s_max = int(s_min), int(s_max)
            for i in range(dose.N_SAMPLES):
                out.append(dict(
                    cid=custom_id(row["story_id"], tmpl, i),
                    prompt=prompt, s_min=s_min, s_max=s_max,
                    story_id=row["story_id"], condition_cell=row["condition"],
                    template=tmpl, sample_i=i,
                    body=_body(provider, model, condition, prompt)))
    return out


def _body(provider, model, condition, prompt):
    msgs = [{"role": "user", "content": prompt}]
    if provider == "openai":
        is_reason = model.startswith(("o1", "o3", "o4"))
        if is_reason:
            return dict(model=model, messages=msgs,
                        max_completion_tokens=dose.OPENAI_MAX_OUT.get(condition, 8192),
                        reasoning_effort=dose.OPENAI_EFFORTS.get(condition, "medium"))
        # gpt-5.x spends completion tokens on hidden reasoning even in `direct`.
        return dict(model=model, messages=msgs, max_completion_tokens=1024)
    if provider == "anthropic":
        b = dict(model=model, messages=msgs)
        if condition == "direct":
            b["max_tokens"] = 32
        elif condition == "think":
            b.update(max_tokens=8192,
                     thinking={"type": "enabled", "budget_tokens": 4096})
        else:
            bt = dose.ANTHROPIC_BUDGETS[condition]
            b.update(max_tokens=bt + 1536,
                     thinking={"type": "enabled", "budget_tokens": bt})
        return b
    if provider == "google":
        tb = 0 if condition == "direct" else (-1 if condition == "think"
                                              else dose.GOOGLE_BUDGETS[condition])
        cap = 8192 if tb < 0 else max(2048, tb + 1536)
        cfg = dict(max_output_tokens=cap, temperature=0)
        if tb >= 0:
            cfg["thinking_config"] = {"thinking_budget": tb}
        return dict(model=model, contents=prompt, config=cfg)
    raise ValueError(f"{provider} has no batch endpoint")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

MFIELDS = ["provider", "model", "condition", "tag", "batch_id", "n_requests",
           "submitted_at", "status", "collected_at", "n_ok", "n_failed", "note"]


def read_manifest():
    if not os.path.exists(MANIFEST):
        return {}
    return {(r["model"], r["condition"]): r for r in csv.DictReader(open(MANIFEST))}


def write_manifest(recs):
    os.makedirs(OUT, exist_ok=True)
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MFIELDS)
        w.writeheader()
        for r in sorted(recs.values(), key=lambda r: (r["provider"], r["model"],
                                                      r["condition"])):
            w.writerow({k: r.get(k, "") for k in MFIELDS})


# ---------------------------------------------------------------------------
# Provider submit / poll / fetch
# ---------------------------------------------------------------------------

def submit_openai(reqs, tag):
    from openai import OpenAI
    client = OpenAI()
    os.makedirs(REQ_DIR, exist_ok=True)
    path = os.path.join(REQ_DIR, f"{tag}.jsonl")
    with open(path, "w") as f:
        for r in reqs:
            f.write(json.dumps(dict(custom_id=r["cid"], method="POST",
                                    url="/v1/chat/completions", body=r["body"])) + "\n")
    up = client.files.create(file=open(path, "rb"), purpose="batch")
    b = client.batches.create(input_file_id=up.id, endpoint="/v1/chat/completions",
                              completion_window="24h",
                              metadata={"tag": tag, "experiment": "roadmap7_dose"})
    return b.id


def poll_openai(batch_id):
    from openai import OpenAI
    b = OpenAI().batches.retrieve(batch_id)
    done = b.status in ("completed", "failed", "expired", "cancelled")
    return b.status, done


def fetch_openai(batch_id):
    from openai import OpenAI
    client = OpenAI()
    b = client.batches.retrieve(batch_id)
    if not b.output_file_id:
        return {}
    out = {}
    for line in client.files.content(b.output_file_id).text.splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        try:
            txt = d["response"]["body"]["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            txt = ""
        out[d["custom_id"]] = txt
    return out


def submit_anthropic(reqs, tag):
    import anthropic
    client = anthropic.Anthropic()
    b = client.messages.batches.create(
        requests=[dict(custom_id=r["cid"], params=r["body"]) for r in reqs])
    return b.id


def poll_anthropic(batch_id):
    import anthropic
    b = anthropic.Anthropic().messages.batches.retrieve(batch_id)
    return b.processing_status, b.processing_status == "ended"


def fetch_anthropic(batch_id):
    import anthropic
    client = anthropic.Anthropic()
    out = {}
    for r in client.messages.batches.results(batch_id):
        txt = ""
        if getattr(r.result, "type", "") == "succeeded":
            txt = " ".join(blk.text for blk in r.result.message.content
                           if getattr(blk, "type", "") == "text")
        out[r.custom_id] = txt
    return out


def submit_google(reqs, tag):
    from google import genai
    client = genai.Client()
    model = reqs[0]["body"]["model"]
    src = [dict(contents=r["body"]["contents"], config=r["body"]["config"])
           for r in reqs]
    job = client.batches.create(model=model, src=src,
                                config=dict(display_name=f"dose_{tag}"))
    # The custom_id -> position mapping is ours to keep: Google's inline batch preserves
    # request order in dest.inlined_responses and carries no per-request id.
    os.makedirs(REQ_DIR, exist_ok=True)
    with open(os.path.join(REQ_DIR, f"{tag}.order.json"), "w") as f:
        json.dump([r["cid"] for r in reqs], f)
    return job.name


def poll_google(batch_id):
    from google import genai
    job = genai.Client().batches.get(name=batch_id)
    state = str(job.state)
    return state, state.endswith(("SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED"))


def fetch_google(batch_id, tag):
    from google import genai
    job = genai.Client().batches.get(name=batch_id)
    order_path = os.path.join(REQ_DIR, f"{tag}.order.json")
    order = json.load(open(order_path)) if os.path.exists(order_path) else []
    out = {}
    resps = getattr(getattr(job, "dest", None), "inlined_responses", None) or []
    for i, resp in enumerate(resps):
        txt = ""
        try:
            txt = resp.response.text or ""
        except Exception:
            txt = ""
        if i < len(order):
            out[order[i]] = txt
    return out


SUBMIT = dict(openai=submit_openai, anthropic=submit_anthropic, google=submit_google)
POLL = dict(openai=poll_openai, anthropic=poll_anthropic, google=poll_google)


def fetch(provider, batch_id, tag):
    if provider == "google":
        return fetch_google(batch_id, tag)
    return dict(openai=fetch_openai, anthropic=fetch_anthropic)[provider](batch_id)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cells(providers=None):
    for prov, model, conds in dose.ROSTER:
        if providers and prov not in providers:
            continue
        for cond in conds:
            yield prov, model, cond


def cmd_plan(beh, rows, a):
    print(f"templates={dose.TEMPLATES}  n_samples={dose.N_SAMPLES}  "
          f"items={len(rows[:a.limit] if a.limit else rows)}")
    per = len(rows[:a.limit] if a.limit else rows) * len(dose.TEMPLATES) * dose.N_SAMPLES
    n = 0
    for prov, model, cond in cells(a.providers):
        n += 1
        print(f"  {prov:10} {model:24} {cond:12} {per:>6,} requests"
              + ("" if prov in BATCHABLE else "   (no batch endpoint -> sync)"))
    print(f"\n{n} cells x {per:,} = {n * per:,} requests")
    print("Batch discount applies to " + ", ".join(BATCHABLE))


def cmd_submit(beh, rows, a):
    man = read_manifest()
    for prov, model, cond in cells(a.providers):
        if prov not in BATCHABLE:
            print(f"[skip] {model}/{cond}: {prov} has no batch endpoint (use `sync`)")
            continue
        key = (model, cond)
        if key in man and man[key].get("batch_id") and not a.force:
            print(f"[skip] {model}/{cond}: already submitted {man[key]['batch_id']}")
            continue
        tag = f"{dose.safe(model)}__{cond}"
        reqs = cell_requests(beh, rows, prov, model, cond, limit=a.limit)
        try:
            bid = SUBMIT[prov](reqs, tag)
        except Exception as e:
            dose.fatal_if_billing(e, prov, model, cond)
            print(f"!! submit failed {model}/{cond}: {str(e)[:200]}")
            continue
        man[key] = dict(provider=prov, model=model, condition=cond, tag=tag,
                        batch_id=bid, n_requests=len(reqs),
                        submitted_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                        status="submitted", collected_at="", n_ok="", n_failed="",
                        note="")
        write_manifest(man)
        print(f"  submitted {model}/{cond}: {bid} ({len(reqs):,} requests)")


def cmd_poll(beh, rows, a):
    man = read_manifest()
    if not man:
        print("nothing submitted")
        return
    for key, r in sorted(man.items()):
        if not r.get("batch_id"):
            continue
        try:
            status, done = POLL[r["provider"]](r["batch_id"])
        except Exception as e:
            status, done = f"poll-error: {str(e)[:80]}", False
        r["status"] = status
        print(f"  {r['model']:24} {r['condition']:12} {status:28}"
              f"{' READY' if done else ''}")
    write_manifest(man)


def cmd_collect(beh, rows, a):
    """Write raw_<tag>.csv from finished batches, in 52's schema so report() is unchanged."""
    man = read_manifest()
    for key, r in sorted(man.items()):
        if not r.get("batch_id") or (r.get("collected_at") and not a.force):
            continue
        prov = r["provider"]
        try:
            status, done = POLL[prov](r["batch_id"])
        except Exception as e:
            print(f"  {r['model']}/{r['condition']}: poll failed {str(e)[:100]}")
            continue
        if not done:
            print(f"  {r['model']}/{r['condition']}: {status}, not ready")
            continue
        texts = fetch(prov, r["batch_id"], r["tag"])
        reqs = cell_requests(beh, rows, prov, r["model"], r["condition"], limit=a.limit)
        path = os.path.join(OUT, f"raw_{r['tag']}.csv")
        n_ok = n_failed = 0
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for q in reqs:
                txt = texts.get(q["cid"], "")
                v = beh._parse_rating(txt, q["s_min"], q["s_max"]) if txt else None
                common = dict(model=r["model"], condition=r["condition"], provider=prov,
                              story_id=q["story_id"], condition_cell=q["condition_cell"],
                              template=q["template"], sample_i=q["sample_i"],
                              s_min=q["s_min"], s_max=q["s_max"])
                if v is None:
                    n_failed += 1
                    w.writerow(dict(common, rating="", rating_norm="", status="failed"))
                else:
                    n_ok += 1
                    w.writerow(dict(
                        common, rating=v,
                        rating_norm=round((v - q["s_min"])
                                          / (q["s_max"] - q["s_min"]), 4), status="ok"))
        r.update(status=status, collected_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                 n_ok=n_ok, n_failed=n_failed)
        write_manifest(man)
        print(f"  {r['model']:24} {r['condition']:12} {n_ok:,} ok / {n_failed:,} failed "
              f"-> raw_{r['tag']}.csv")
    dose.report()


def cmd_sync(beh, rows, a):
    """Providers without a batch endpoint, through 52's synchronous path."""
    for prov, model, cond in cells(a.providers):
        if prov in BATCHABLE and not a.force:
            print(f"[skip] {model}/{cond}: {prov} is batchable, use `submit`")
            continue
        print(f"\n----- {prov} / {model} / {cond} (sync) -----")
        try:
            dose.run_cell(beh, rows, prov, model, cond, limit=a.limit)
        except dose.BillingStop as e:
            print(f"!! BILLING STOP: {e}")
            return
        except Exception as e:
            print(f"!! cell failed: {e}")
    dose.report()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "submit", "poll", "collect", "sync"])
    ap.add_argument("--providers", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="first N stories only (smoke test)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    beh = _load("03_behavioral.py", "behavioral")
    rows = beh.load_dataset()
    dict(plan=cmd_plan, submit=cmd_submit, poll=cmd_poll, collect=cmd_collect,
         sync=cmd_sync)[a.cmd](beh, rows, a)


if __name__ == "__main__":
    main()
