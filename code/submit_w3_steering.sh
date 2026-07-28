#!/bin/bash
#SBATCH --job-name=w3_steer
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=outputs/logs/gpu_%j.log

# W3 causal steering (roadmap: the only remaining item that upgrades a correlation into a
# causal claim). Two models, both 7B-class, one L40S is enough.
#
# Cost shape: each (direction x alpha) cell rescores all 298 stories, so 6 directions x 10
# non-zero alphas = 60 sweeps per model, plus generations at every intent alpha and at the
# control extremes. Batched at 16, that is a few hours per model; 10h covers both with
# headroom on a preemptable partition.
#
# Predictions are pre-registered in the script docstring and written to
# outputs/experiments/W3_PRESPEC.md on the first run, before any result exists.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

echo "=== Job $SLURM_JOB_ID on $(hostname) at $(date) ==="
nvidia-smi
python -c "import torch;print('CUDA:', torch.cuda.is_available())"

for M in allenai/OLMo-2-1124-7B-Instruct Qwen/Qwen2.5-7B-Instruct; do
  echo ""
  echo "================ $M ================"
  python -u code/experiments/48_w3_causal_steering.py --run --model "$M" \
    || echo "!! $M failed"
done

echo "=== done at $(date) ==="
