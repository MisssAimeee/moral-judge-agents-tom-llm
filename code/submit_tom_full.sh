#!/bin/bash
#SBATCH --job-name=tom_full
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/logs/gpu_%j.log

# J1 full run. The ceiling gate (job 19026525) found real spread on both benchmarks --
# BigToM 0.520 / 0.882 / 0.850 and ToMi 0.482 / 0.512 / 0.818 across the three probe
# models -- so both axes of the planned correlation have variance and the full roster is
# worth the GPU. The gate took 4 minutes for 3 models, so this is cheap.
#
# The 20 model ids are the ones with behavioural data in outputs/behavior, spelled exactly
# as the raw_*.csv filenames imply, so ToM accuracy joins to the 2x2 contrast without a
# name-matching step.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false

echo "=== Job $SLURM_JOB_ID on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -u code/experiments/36_tom_benchmarks.py \
  --models \
    Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-0.5B-Instruct \
    Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-1.5B-Instruct \
    Qwen/Qwen2.5-3B Qwen/Qwen2.5-3B-Instruct \
    Qwen/Qwen2.5-7B Qwen/Qwen2.5-7B-Instruct \
    Qwen/Qwen2.5-14B Qwen/Qwen2.5-14B-Instruct \
    allenai/OLMo-2-1124-7B allenai/OLMo-2-1124-7B-Instruct \
    allenai/Llama-3.1-Tulu-3-8B \
    HuggingFaceH4/zephyr-7b-beta \
    mistralai/Mistral-7B-v0.3 mistralai/Mistral-7B-Instruct-v0.3 \
    unsloth/gemma-2-9b unsloth/gemma-2-9b-it \
    unsloth/Meta-Llama-3.1-8B unsloth/Meta-Llama-3.1-8B-Instruct

echo "=== done at $(date) ==="
