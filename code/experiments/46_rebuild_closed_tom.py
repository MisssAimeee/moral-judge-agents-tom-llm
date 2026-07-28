#!/usr/bin/env python3
"""Rebuild tom_accuracy_by_model_generative.csv and CLOSED_TOM.md from item CSVs."""
import collections
import csv
import glob
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "outputs", "tom_benchmarks")

# Full BigToM slice per model. A model scored on fewer items was interrupted, and the
# report must say so rather than presenting a partial accuracy as a finished run.
FULL_N = 400


def main():
    rows_out = []
    for path in sorted(glob.glob(os.path.join(OUT, "tom_gen_items_*.csv"))):
        items = list(csv.DictReader(open(path)))
        if not items:
            continue
        model = items[0].get("model") or os.path.basename(path)
        backend = items[0].get("backend", "")
        # A row with no response text was never scored -- an API error, a quota block, or an
        # interrupted run. Counting those as wrong answers turned a 234-item Gemini Pro run
        # into a "complete" 400-item run at 0.570, so drop them and let the partial-run
        # flag report the real denominator.
        n_raw = len(items)
        items = [r for r in items if (r.get("response") or "").strip()]
        if len(items) < n_raw:
            print(f"  {os.path.basename(path)}: dropped {n_raw - len(items)} unanswered "
                  f"rows (no response text)")
        if not items:
            continue
        agg = collections.defaultdict(lambda: [0, 0, 0])
        for r in items:
            for key in (r["bench"], f"{r['bench']}|{r['subset']}"):
                a = agg[key]
                a[0] += int(r["is_correct"])
                a[1] += 1
                a[2] += int(r.get("parsed", 1))
        for subset, (ok, n, parsed) in sorted(agg.items()):
            rows_out.append([model, subset, ok, n, round(ok / n, 4),
                             round(parsed / n, 4), backend, "generative"])

    path = os.path.join(OUT, "tom_accuracy_by_model_generative.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "subset", "n_correct", "n_items", "accuracy",
                    "parse_rate", "backend", "method"])
        w.writerows(rows_out)
    print(f"wrote {path} ({len(rows_out)} rows)")

    closed_backends = {"anthropic", "openai", "google"}
    by_model = collections.OrderedDict()
    for r in rows_out:
        if r[6] not in closed_backends:
            continue
        by_model.setdefault(r[0], {})[r[1]] = r

    lines = [
        "# Closed-model BigToM (generative, standalone)",
        "",
        "Scoring: free generation forced to one of the two Forward-Belief options",
        "(same options as the open-model logprob 2AFC). BigToM uses **init_belief=0**",
        "(initial-belief sentence dropped). ToMi is not scored.",
        "",
        "**Do not correlate** these accuracies against closed-model moral contrasts —",
        "those contrasts are still v1-contaminated. Report ToM standalone only.",
        "",
        "| model | backend | n | BigToM all | BigToM FB | BigToM TB | parse rate | run |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    partials = []
    for model, d in by_model.items():
        allr = d.get("bigtom")
        fb = d.get("bigtom|false_belief")
        tb = d.get("bigtom|true_belief")
        if not allr:
            continue
        fb_a = fb[4] if fb else float("nan")
        tb_a = tb[4] if tb else float("nan")
        complete = allr[3] >= FULL_N
        if not complete:
            partials.append((model, allr[3]))
        lines.append(
            f"| {model} | {allr[6]} | {allr[3]} | {allr[4]:.3f} | "
            f"{fb_a:.3f} | {tb_a:.3f} | {allr[5]:.3f} | "
            f"{'complete' if complete else f'**PARTIAL {allr[3]}/{FULL_N}**'} |"
        )
    if partials:
        lines += [
            "",
            "**Partial runs.** " + "; ".join(f"`{m}` scored {n}/{FULL_N} items"
                                            for m, n in partials) +
            ". These accuracies are computed on the items completed, so they carry wider "
            "sampling error than the full runs and the item mix may not be balanced "
            "across subsets. Treat them as provisional until the run finishes.",
        ]

    agree = os.path.join(OUT, "tom_scoring_agreement.csv")
    if os.path.exists(agree):
        lines += [
            "", "## Open-model logprob vs generative agreement (BigToM)", "",
            "| model | n | logprob acc | generative acc | pred agreement |",
            "|---|---:|---:|---:|---:|",
        ]
        for r in csv.DictReader(open(agree)):
            lines.append(
                f"| {r['model']} | {r['n']} | {float(r['logprob_acc']):.3f} | "
                f"{float(r['generative_acc']):.3f} | {float(r['pred_agreement']):.3f} |"
            )
        lines += [
            "",
            "Qwen agreement is high; use generative for closed models and treat",
            "open logprob BigToM FB as the open roster measure (parity demonstrated,",
            "not perfect on every family).",
            "",
            "## Notes",
            "",
            "- Gemini thinking models often return a bare `A`/`B` or put the letter in",
            "  thought parts; the generative scorer accepts bare letters and falls back",
            "  to short thought lines. Report parse rate alongside accuracy.",
            "- Empty responses count as incorrect; re-fetch if parse rate < 0.9.",
        ]

    out_md = os.path.join(OUT, "CLOSED_TOM.md")
    open(out_md, "w").write("\n".join(lines) + "\n")
    print(f"wrote {out_md}")
    for model, d in by_model.items():
        fb = d.get("bigtom|false_belief")
        if fb:
            print(f"  {model}: FB={fb[4]:.3f} parse={fb[5]:.3f} n={fb[3]}")


if __name__ == "__main__":
    main()
