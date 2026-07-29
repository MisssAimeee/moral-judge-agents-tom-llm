#!/bin/bash
#SBATCH --job-name=w4_pass2
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/logs/gpu_%j.log

# W4 pass 2. Two reasons this exists; run 1 is preserved under
# outputs/experiments/_w4_prefix_fewshot_bug/ with both written up.
#
#   1. RESCORE L4 and L5. In run 1 the few-shot block rendered every example under the
#      question built from the TARGET item, and `human_verbatim` interpolates the agent
#      name — so an example about Nadia was followed by "How permissible was Grace's
#      action?". Invalid on 1 of 7 templates, and only at the levels containing the
#      few-shot block, which is exactly where run 1's anomaly sat. --force is used so
#      these two levels are replaced rather than skipped by the resume check.
#
#   2. ADD the non-cumulative ablation, L6-L8. L1-L5 are cumulative, so L5 confounds the
#      explicit principle with repair of whatever L4 did. L6 = worked example alone,
#      L7 = few-shot alone, L8 = principle alone, each against the same L1 baseline.
#      L2 is already "instruction alone" and is not re-run.
#
# L1-L3 are untouched: their prompts never contained the few-shot block. Their intervals
# were corrected offline with --recompute, which rebuilds the summary rows from the stored
# per-cell means after the bootstrap unit was changed from template x group cell to
# scenario group (7 templates over 53 groups is 53 units, not 371).

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export HF_HOME="${HF_HOME:-/orcd/scratch/orcd/007/$USER/hf_cache}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "=== Job $SLURM_JOB_ID on $(hostname) at $(date) ==="
nvidia-smi
python -c "import torch;print('CUDA:', torch.cuda.is_available())"

echo ""
echo "================ L4 label polarity gate ================"
python -u code/experiments/54_w4_prompt_curriculum.py --check-polarity || exit 1

echo ""
echo "================ L4/L5 rescore + L6-L8 ablation ================"
python -u code/experiments/54_w4_prompt_curriculum.py --run --force \
  --levels 4 5 6 7 8 || echo "!! pass 2 failed"

echo ""
echo "================ report ================"
python -u code/experiments/55_w4_summary.py || echo "!! W4 summary failed"

echo "=== done at $(date) ==="
