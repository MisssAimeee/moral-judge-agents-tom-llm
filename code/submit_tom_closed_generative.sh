#!/usr/bin/env bash
# Submit / run generative BigToM for closed APIs + finish open agreement checks.
# BigToM only (400 items). ToMi dropped — see outputs/tom_benchmarks/TOMI_SCORING_AUDIT.md.
#
# Usage (login node / CPU partition is enough):
#   bash code/submit_tom_closed_generative.sh
#   # or just one provider:
#   PROVIDER=openai bash code/submit_tom_closed_generative.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
[ -f .env_agents ] && source .env_agents
mkdir -p outputs/logs outputs/tom_benchmarks

PROVIDER="${PROVIDER:-all}"
LOG="outputs/logs/tom_gen_closed_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "=== tom closed generative $(date) provider=$PROVIDER ==="

run() {
  local backend="$1"; shift
  echo ""
  echo "----- $backend: $* -----"
  python -u code/experiments/45_tom_generative.py \
    --backend "$backend" --models "$@" --sleep 0.05 || echo "!! $backend failed"
}

# Resume / finish Claude (Opus may be partial)
if [[ "$PROVIDER" == "all" || "$PROVIDER" == "anthropic" || "$PROVIDER" == "claude" ]]; then
  if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "[SKIP] Anthropic — ANTHROPIC_API_KEY not set"
  else
    run anthropic \
      claude-haiku-4-5-20251001 \
      claude-sonnet-4-6 \
      claude-opus-4-6
  fi
fi

if [[ "$PROVIDER" == "all" || "$PROVIDER" == "openai" || "$PROVIDER" == "gpt" ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "[SKIP] OpenAI — OPENAI_API_KEY not set"
  else
    run openai gpt-4o-mini gpt-4o
  fi
fi

if [[ "$PROVIDER" == "all" || "$PROVIDER" == "google" || "$PROVIDER" == "gemini" ]]; then
  if [[ -z "${GOOGLE_API_KEY:-}" && -z "${GEMINI_API_KEY:-}" ]]; then
    echo "[SKIP] Google — GOOGLE_API_KEY not set"
  else
    run google gemini-2.5-flash gemini-2.5-pro
  fi
fi

# Rebuild generative summary + CLOSED_TOM.md from all item CSVs on disk
python -u - <<'PY'
import csv, glob, os, collections
OUT = "outputs/tom_benchmarks"
rows_out = []
for path in sorted(glob.glob(os.path.join(OUT, "tom_gen_items_*.csv"))):
    items = list(csv.DictReader(open(path)))
    if not items:
        continue
    model = items[0].get("model") or os.path.basename(path)
    backend = items[0].get("backend", "")
    agg = collections.defaultdict(lambda: [0, 0, 0])
    for r in items:
        for key in (r["bench"], f"{r['bench']}|{r['subset']}"):
            a = agg[key]
            a[0] += int(r["is_correct"]); a[1] += 1; a[2] += int(r.get("parsed", 1))
    for subset, (ok, n, parsed) in sorted(agg.items()):
        rows_out.append([model, subset, ok, n, round(ok/n, 4), round(parsed/n, 4),
                         backend, "generative"])

path = os.path.join(OUT, "tom_accuracy_by_model_generative.csv")
with open(path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "subset", "n_correct", "n_items", "accuracy",
                "parse_rate", "backend", "method"])
    w.writerows(rows_out)
print(f"wrote {path} ({len(rows_out)} rows)")

# CLOSED_TOM.md — standalone closed accuracies only
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
    "| model | backend | n | BigToM all | BigToM FB | BigToM TB | parse rate |",
    "|---|---|---:|---:|---:|---:|---:|",
]
for model, d in by_model.items():
    allr = d.get("bigtom")
    fb = d.get("bigtom|false_belief")
    tb = d.get("bigtom|true_belief")
    if not allr:
        continue
    lines.append(
        f"| {model} | {allr[6]} | {allr[3]} | {allr[4]:.3f} | "
        f"{(fb[4] if fb else float('nan')):.3f} | "
        f"{(tb[4] if tb else float('nan')):.3f} | {allr[5]:.3f} |"
    )
# Open agreement appendix
agree = os.path.join(OUT, "tom_scoring_agreement.csv")
if os.path.exists(agree):
    lines += ["", "## Open-model logprob vs generative agreement (BigToM)", "",
              "| model | n | logprob acc | generative acc | pred agreement |",
              "|---|---:|---:|---:|---:|"]
    for r in csv.DictReader(open(agree)):
        lines.append(
            f"| {r['model']} | {r['n']} | {float(r['logprob_acc']):.3f} | "
            f"{float(r['generative_acc']):.3f} | {float(r['pred_agreement']):.3f} |"
        )
    lines += ["", "Qwen agreement is high; use generative for closed models and treat",
              "open logprob BigToM FB as the open roster measure (parity demonstrated,",
              "not perfect on every family)."]
open(os.path.join(OUT, "CLOSED_TOM.md"), "w").write("\n".join(lines) + "\n")
print("wrote", os.path.join(OUT, "CLOSED_TOM.md"))
PY

echo "=== done $(date); log=$LOG ==="
