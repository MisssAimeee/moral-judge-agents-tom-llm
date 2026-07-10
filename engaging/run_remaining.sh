#!/usr/bin/env bash
# Completes what run_node.sh didn't finish: 7B-Instruct re-run + all extended families.
# No CoT (that's a separate experiment). Logprob battery only.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
source .venv/bin/activate

MODELS=(
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen2.5-14B"              "Qwen/Qwen2.5-14B-Instruct"
  "meta-llama/Llama-3.1-8B"       "meta-llama/Llama-3.1-8B-Instruct"
  "meta-llama/Llama-3.2-3B"       "meta-llama/Llama-3.2-3B-Instruct"
  "mistralai/Mistral-7B-v0.3"     "mistralai/Mistral-7B-Instruct-v0.3"
  "google/gemma-2-9b"             "google/gemma-2-9b-it"
  "microsoft/Phi-3-mini-4k-instruct"
  "allenai/OLMo-2-1124-7B"        "allenai/OLMo-2-1124-7B-Instruct"
)

for m in "${MODELS[@]}"; do
  echo "=================== $m ($(date +%H:%M:%S)) ==================="
  python code/03_behavioral.py --backend hf --models "$m" --scoring logprob \
    || echo "!! $m FAILED — continuing"
done

echo "=================== analysis ($(date +%H:%M:%S)) ==================="
python code/05_human_comparison.py --template human_verbatim
python code/06_stats.py
python code/07_visualize.py
python code/08_report.py
echo "DONE $(date +%H:%M:%S)"
