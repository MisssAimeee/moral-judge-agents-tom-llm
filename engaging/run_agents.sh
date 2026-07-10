#!/usr/bin/env bash
# run_agents.sh -- Behavioral ToM experiment on the BIG "daily agent" LLMs
# (GPT, Claude, Gemini) + the Llama open-weight ladder, all via cloud APIs.
#
# Results are kept SEPARATE from the local Qwen size-study, under outputs/agents/.
#
# ── Setup (once) ─────────────────────────────────────────────────────────────
#   bash engaging/setup_api_models.sh          # installs openai/anthropic/google/together
#
# ── Provide keys (only the ones you have; missing providers are skipped) ──────
#   export OPENAI_API_KEY="sk-..."
#   export ANTHROPIC_API_KEY="sk-ant-..."
#   export GOOGLE_API_KEY="AIza..."            # or GEMINI_API_KEY
#   export TOGETHER_API_KEY="..."              # runs all Llama sizes (8B/70B/405B/3B)
#
# ── Run ──────────────────────────────────────────────────────────────────────
#   bash engaging/run_agents.sh                # all providers with a key set
#   bash engaging/run_agents.sh openai         # just one provider
#
# Safe to re-run: --skip_existing skips models already saved. Per-model failures
# don't stop the batch. Estimated cost for the full set at n_samples=5 is a few $.

set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
[ -f .env_agents ] && source .env_agents
export TOKENIZERS_PARALLELISM=false

PROVIDER="${1:-all}"
# For fast thinking models (Gemini 2.5) 1 sample at T=0 is already deterministic.
# Bump N_SAMPLES to 3+ only for non-thinking models (GPT, Claude) where variance matters.
N_SAMPLES="${N_SAMPLES:-1}"
TEMP="${TEMP:-0.0}"
# Limit templates to the 3 most diagnostic for speed. Full 7-template run: pass all.
TEMPLATES="${TEMPLATES:-human_verbatim para_wrong7 punish7}"
OUT="outputs/agents/behavior"
mkdir -p "$OUT" outputs/agents/stats outputs/agents/figures outputs/logs

run() {
  local backend="$1"; shift
  echo ""
  echo "======================= $backend ($(date +%H:%M:%S)) ======================="
  python -u code/03_behavioral.py \
    --backend "$backend" --scoring sampling \
    --n_samples "$N_SAMPLES" --temperature "$TEMP" \
    --templates $TEMPLATES \
    --skip_existing --out_dir "$OUT" \
    --models "$@" || echo "!! $backend batch failed — continuing"
}

# ── OpenAI GPT (gpt-4o-mini + gpt-4o — registered models) ────────────────────
if [[ "$PROVIDER" == "all" || "$PROVIDER" == "openai" || "$PROVIDER" == "gpt" ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "[SKIP] OpenAI — OPENAI_API_KEY not set"
  else
    run openai "gpt-4o-mini" "gpt-4o"
  fi
fi
# NOTE: Llama already run in the local GPU study (outputs/behavior/); no Together API run needed.

# ── Anthropic Claude (Haiku / Sonnet / Opus) ─────────────────────────────────
if [[ "$PROVIDER" == "all" || "$PROVIDER" == "anthropic" || "$PROVIDER" == "claude" ]]; then
  if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "[SKIP] Claude/Anthropic — ANTHROPIC_API_KEY not set"
  else
    run anthropic \
      "claude-haiku-4-5-20251001" \
      "claude-sonnet-4-6" \
      "claude-opus-4-6"
  fi
fi

# ── Google Gemini (2.5 Flash + 2.5 Pro — verified working Jun 2026) ──────────
# gemini-1.5-* and gemini-2.0-* are deprecated/removed; use 2.5+
if [[ "$PROVIDER" == "all" || "$PROVIDER" == "google" || "$PROVIDER" == "gemini" ]]; then
  if [[ -z "${GOOGLE_API_KEY:-}" && -z "${GEMINI_API_KEY:-}" ]]; then
    echo "[SKIP] Gemini/Google — GOOGLE_API_KEY not set"
  else
    run google "gemini-2.5-flash"
    run google "gemini-2.5-pro"
  fi
fi

# ── Analysis + figures (pointed at the agents outputs) ───────────────────────
echo ""
echo "======================= analysis + figures ($(date +%H:%M:%S)) ======================="
python -u code/05_human_comparison.py --behavior "$OUT" --out outputs/agents/human \
       --template human_verbatim || true
python -u code/06_stats.py  --behavior "$OUT" --out outputs/agents/stats || true
python -u code/09_agent_figures.py --behavior "$OUT" --stats outputs/agents/stats \
       --out outputs/agents/figures || true
python -u code/08_report.py --behavior "$OUT" --stats outputs/agents/stats \
       --out outputs/agents/report || true

echo ""
echo "DONE ($(date +%H:%M:%S))"
echo "  ratings : $OUT/"
echo "  stats   : outputs/agents/stats/"
echo "  figures : outputs/agents/figures/  (agent_scale.png + 6 comparison figures)"
echo "  report  : outputs/agents/report/summary_table.md"
