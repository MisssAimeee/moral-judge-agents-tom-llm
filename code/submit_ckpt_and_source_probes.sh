#!/bin/bash
#SBATCH --job-name=ckpt_factorial
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/gpu_%j.log

# Two things at once, both blocked on the digit-token fix:
#
# 1. The whole Zephyr checkpoint family read 0.0 at every stage because its base and both
#    tuned stages are Mistral-tokenizer models, so they hit the midpoint collapse. With
#    that fixed, Zephyr becomes a third mechanism family instead of a fabricated null.
# 2. Rescoring on the 6-template factorial (not just human_verbatim) puts the checkpoint
#    dissection on the same measurement basis as the ladder and the factorial analysis,
#    so the figures can be shown together.
#
# --force rescores all three families so every stage shares one scorer version.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

echo "=== Job $SLURM_JOB_ID on $(hostname) at $(date) ==="
nvidia-smi | head -12

python -u code/experiments/16_checkpoint_dissection.py --run --force \
  --templates human_verbatim blame_w1 blame_w2 wrong_w1 wrong_w2 punish_w1 punish_w2

echo "=== checkpoints done at $(date) ==="
