#!/usr/bin/env bash
# run_api_models.sh -- Run the behavioral ToM experiment on closed/large-scale API
# models (OpenAI, Anthropic, Google, Mistral, Together AI).
#
# DO NOT run this directly -- it will consume API credits.
# Set your API keys first:
#   export OPENAI_API_KEY="sk-..."
#   export ANTHROPIC_API_KEY="sk-ant-..."
#   export GOOGLE_API_KEY="AIza..."
#   export MISTRAL_API_KEY="..."
#   export TOGETHER_API_KEY="..."
#
# Then run: bash engaging/run_api_models.sh
# Or for a single provider: bash engaging/run_api_models.sh openai
#
# The script automatically skips models whose raw CSV already exists and is
# non-empty, so you can safely re-run after a partial failure.
#
# Cost estimates (n_samples=5, ~298 stories x 7 templates = 2086 calls/model):
#   GPT-4o         ~$2-4   | GPT-4o-mini  ~$0.10
#   Claude Sonnet  ~$3-6   | Claude Haiku ~$0.10
#   Gemini 1.5 Pro ~$1-3   | Gemini Flash ~$0.05
#   Mistral Large  ~$1-2
#   Together Llama-3.1-70B ~$0.30

set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false

PROVIDER="${1:-all}"   # pass a provider name to run only that one
N_SAMPLES=5            # sampling draws per story x template
TEMP=0.0               # deterministic at T=0 (most APIs support this)

run_provider() {
    local backend="$1"
    shift
    echo ""
    echo "========================================================"
    echo " Provider: $backend  ($(date +%H:%M:%S))"
    echo "========================================================"
    python code/03_behavioral.py \
        --backend "$backend" \
        --scoring sampling \
        --n_samples "$N_SAMPLES" \
        --temperature "$TEMP" \
        --skip_existing \
        --models "$@" \
        || echo "!! $backend run failed -- check output above"
}

# ─── OpenAI ──────────────────────────────────────────────────────────────────
if [[ "$PROVIDER" == "all" || "$PROVIDER" == "openai" ]]; then
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
        echo "[SKIP] openai -- OPENAI_API_KEY not set"
    else
        run_provider openai \
            "gpt-4o" \
            "gpt-4o-mini" \
            "gpt-4-turbo" \
            "o1-mini" \
            "o3-mini"
    fi
fi

# ─── Anthropic ───────────────────────────────────────────────────────────────
if [[ "$PROVIDER" == "all" || "$PROVIDER" == "anthropic" ]]; then
    if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
        echo "[SKIP] anthropic -- ANTHROPIC_API_KEY not set"
    else
        run_provider anthropic \
            "claude-3-5-sonnet-20241022" \
            "claude-3-5-haiku-20241022" \
            "claude-3-opus-20240229"
    fi
fi

# ─── Google Gemini ────────────────────────────────────────────────────────────
if [[ "$PROVIDER" == "all" || "$PROVIDER" == "google" ]]; then
    if [[ -z "${GOOGLE_API_KEY:-}" && -z "${GEMINI_API_KEY:-}" ]]; then
        echo "[SKIP] google -- GOOGLE_API_KEY not set"
    else
        run_provider google \
            "gemini-1.5-pro" \
            "gemini-1.5-flash" \
            "gemini-2.0-flash-exp"
    fi
fi

# ─── Mistral ─────────────────────────────────────────────────────────────────
if [[ "$PROVIDER" == "all" || "$PROVIDER" == "mistral" ]]; then
    if [[ -z "${MISTRAL_API_KEY:-}" ]]; then
        echo "[SKIP] mistral -- MISTRAL_API_KEY not set"
    else
        run_provider mistral \
            "mistral-large-latest" \
            "mistral-small-latest"
    fi
fi

# ─── Together AI (large open-weight models via API) ──────────────────────────
if [[ "$PROVIDER" == "all" || "$PROVIDER" == "together" ]]; then
    if [[ -z "${TOGETHER_API_KEY:-}" ]]; then
        echo "[SKIP] together -- TOGETHER_API_KEY not set"
    else
        run_provider together \
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo" \
            "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo" \
            "Qwen/Qwen2.5-72B-Instruct-Turbo" \
            "deepseek-ai/DeepSeek-V3" \
            "mistralai/Mixtral-8x22B-Instruct-v0.1"
    fi
fi

# ─── Analysis (only if we ran at least one model) ────────────────────────────
echo ""
echo "========================================================"
echo " Running analysis pipeline  ($(date +%H:%M:%S))"
echo "========================================================"
python code/05_human_comparison.py --template human_verbatim || true
python code/06_stats.py            || true
python code/07_plots.py            || true
python code/08_report.py           || true

echo ""
echo "ALL DONE  ($(date +%H:%M:%S))"
echo "  ratings    : outputs/behavior/"
echo "  vs-human   : outputs/human/"
echo "  stats      : outputs/stats/"
echo "  figures    : outputs/figures/"
echo "  report     : outputs/REPORT.md"
