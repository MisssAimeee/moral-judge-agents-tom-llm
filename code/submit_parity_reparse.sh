#!/bin/bash
#SBATCH --job-name=parity_refix
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=outputs/logs/gpu_%j.log

# Re-run the scoring-parity sampling side after two fixes to 03_behavioral.py.
#
# Why this cannot be left alone: the parity check is what licenses comparing open-weight
# logprob-EV contrasts against closed-model sampled contrasts, and it was run on
# `--template human_verbatim`, whose scale is the SOURCE-NATIVE one -- 1-3 on the 192 YS2008
# items and 1-4 on the 96 YS2009 items. The old parser clamped any out-of-range answer to the
# nearest endpoint, which on a 1-3 scale turns a model answering "6" out of 1-7 habit into
# maximum condemnation, and it imputed the scale midpoint whenever no sample parsed at all.
#
# The pre-fix artifacts show the signature clearly (preserved in
# outputs/analysis/_scoring_parity_clamped/): fraction of items pinned to norm=1.0 on the
# 1-3 YS2008 items vs the 1-4 YS2009 items --
#
#     Qwen2.5-7B-Instruct    45%  vs   4%
#     Mistral-7B-Instruct    41%  vs   7%
#     Qwen2.5-3B-Instruct    76% at norm=0.5 on YS2008  vs 2% on YS2009
#
# Qwen2.5-7B-Instruct is one of the two models that PASSED the r > 0.95 bar, so the bar may
# have been cleared partly on clamped values. It is not possible to settle this from the
# saved artifacts -- the raw response text was never written -- so the only answer is to
# rescore with the corrected parser and compare.
#
# Out-of-range answers are now rejected rather than clamped, and items where nothing parses
# are dropped rather than imputed, so expect n_items below 298 for the weaker models. That
# reduction is the finding, not a defect.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export HF_HOME="${HF_HOME:-/orcd/scratch/orcd/007/$USER/hf_cache}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "=== Job $SLURM_JOB_ID on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv

echo ""
echo "================ parity rescore (fixed parser) ================"
python -u code/analysis/15_scoring_parity.py --run || echo "!! parity failed"

echo "=== done at $(date) ==="
