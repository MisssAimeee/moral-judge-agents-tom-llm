#!/usr/bin/env bash
# Closed behavioral + roadmap #7 reasoning dose–response + BigToM generative.
# CPU/API only. Cost estimate printed first (see CLOSED_MODEL_SELECTION.md).
# DeepSeek omitted (no key). Templates cut to 4 (documented in SELECTION.md).
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
[ -f .env_agents ] && set -a && source .env_agents && set +a
mkdir -p outputs/logs outputs/closed_reasoning outputs/tom_benchmarks

STAMP=$(date +%Y%m%d_%H%M%S)
echo "=== closed reasoning dose $STAMP ==="
python -u code/experiments/52_closed_reasoning_dose.py --cost-only \
  | tee "outputs/logs/closed_cost_${STAMP}.txt"

# One process per provider so Anthropic sequential does not block OpenAI/Google/Moonshot.
pids=()
for P in anthropic openai google moonshot; do
  LOG="outputs/logs/closed_${P}_${STAMP}.log"
  echo "launching $P -> $LOG"
  python -u code/experiments/52_closed_reasoning_dose.py --run --providers "$P" \
    >"$LOG" 2>&1 &
  pids+=($!)
done

# BigToM generative on the same roster (ToMi dropped — see TOMI_SCORING_AUDIT.md).
LOG_TOM="outputs/logs/closed_bigtom_${STAMP}.log"
echo "launching BigToM generative -> $LOG_TOM"
(
  python -u code/experiments/45_tom_generative.py --backend anthropic \
    --models claude-opus-5 claude-sonnet-5 --sleep 0.05
  python -u code/experiments/45_tom_generative.py --backend openai \
    --models gpt-5.5 gpt-5.4-mini o3 o4-mini --sleep 0.05
  python -u code/experiments/45_tom_generative.py --backend google \
    --models gemini-3.1-pro-preview gemini-3.5-flash --sleep 0.05
  # Moonshot / Kimi via OpenAI-compatible is not wired in 45; skip or extend later.
  python -u code/experiments/46_rebuild_closed_tom.py
) >"$LOG_TOM" 2>&1 &
pids+=($!)

echo "PIDs: ${pids[*]}"
fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo "!! pid $pid failed"
    fail=1
  fi
done
python -u code/experiments/52_closed_reasoning_dose.py --report
echo "=== done $(date); fail=$fail ==="
exit $fail
