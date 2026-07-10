#!/usr/bin/env bash
# run_local_gpu.sh -- run the behavioral pipeline DIRECTLY on the GPU node you are
# already sitting on (an interactive SLURM allocation), instead of submitting a
# separate sbatch job. Designed to be launched inside tmux so you can watch progress.
#
#   tmux new -s tom
#   bash engaging/run_local_gpu.sh
#   (detach: Ctrl-b d   reattach: tmux attach -t tom)
#
# It tolerates per-model failures (e.g. a gated model with no access) and keeps going,
# then runs the comparison, stats, plots, and report.
set -uo pipefail
cd "$(dirname "$0")/.."

# HF cache on scratch (home has a quota); reuse the already-cached login token so
# gated models (Llama) work if your account has access.
export HF_HOME="${HF_HOME:-/orcd/scratch/orcd/007/$USER/hf_cache}"
mkdir -p "$HF_HOME" outputs/logs
[ -f "$HOME/.cache/huggingface/token" ] && export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token")}"
export TOKENIZERS_PARALLELISM=false
source .venv/bin/activate

python -c "import torch;print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# Base AND instruct at each size -> separates the size effect from instruction tuning.
MODELS=(
  "Qwen/Qwen2.5-0.5B"          "Qwen/Qwen2.5-0.5B-Instruct"
  "Qwen/Qwen2.5-1.5B"          "Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen/Qwen2.5-3B"            "Qwen/Qwen2.5-3B-Instruct"
  "Qwen/Qwen2.5-7B"            "Qwen/Qwen2.5-7B-Instruct"
  "meta-llama/Llama-3.1-8B"    "meta-llama/Llama-3.1-8B-Instruct"   # gated
)

for m in "${MODELS[@]}"; do
  echo "=================== $m  ($(date +%H:%M:%S)) ==================="
  if python code/03_behavioral.py --backend hf --models "$m" --scoring logprob; then
    echo ">>> done: $m"
  else
    echo ">>> SKIPPED (failed, e.g. no access / OOM): $m"
  fi
done

echo "=================== comparison + stats + plots + report ==================="
python code/05_human_comparison.py --template human_verbatim || true
python code/06_stats.py            || true
python code/07_plots.py            || true
python code/08_report.py           || true

echo "ALL DONE  ($(date +%H:%M:%S))"
echo "  ratings : outputs/behavior/"
echo "  vs-human: outputs/human/"
echo "  stats   : outputs/stats/"
echo "  figures : outputs/figures/"
echo "  REPORT  : outputs/REPORT.md"
