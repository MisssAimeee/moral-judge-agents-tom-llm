#!/bin/bash
#SBATCH --job-name=tom_gate
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/gpu_%j.log

# J1 ceiling gate. Three models spanning the size range are scored on BigToM forward
# belief and ToMi first-order belief before any GPU is committed to the full roster. A
# correlation between ToM accuracy and the 2x2 intent contrast needs variance on both
# axes; if all three sit above 0.95 with no spread, the correlation is not estimable and
# the full run is pointless.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export HF_HUB_OFFLINE=0
export TOKENIZERS_PARALLELISM=false

echo "=== Job $SLURM_JOB_ID on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -u code/experiments/36_tom_benchmarks.py --gate \
  --models \
    Qwen/Qwen2.5-0.5B-Instruct \
    Qwen/Qwen2.5-14B-Instruct \
    allenai/OLMo-2-1124-7B-Instruct

echo "=== done at $(date) ==="
