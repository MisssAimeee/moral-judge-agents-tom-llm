#!/usr/bin/env bash
# submit_gpu.sh -- Submit an arbitrary GPU command as a disconnect-safe Slurm batch job.
#
# The job runs on a compute node (not your login shell), so it KEEPS RUNNING if your
# SSH session drops. This is the mechanism to use for all priority items A-F.
#
# Usage (from the login node):
#   bash engaging/submit_gpu.sh "python code/01_extract_activations.py --models Qwen/Qwen2.5-7B"
#   bash engaging/submit_gpu.sh "python code/analysis/15_scoring_parity.py --run"
#
# Optional env overrides:
#   PART=mit_preemptable TIME=12:00:00 MEM=96G GPUS=1 JOBNAME=parity \
#     bash engaging/submit_gpu.sh "python code/experiments/16_checkpoint_dissection.py --run"
#
# Monitor:
#   squeue -u $USER
#   tail -f outputs/logs/gpu_<JOBID>.log
# Cancel:
#   scancel <JOBID>

set -uo pipefail
PROJ=/home/aimeeyu/tom_project
CMD="${*:-}"

if [[ -z "$CMD" ]]; then
  echo "ERROR: give the command to run, e.g."
  echo "  bash engaging/submit_gpu.sh \"python code/01_extract_activations.py --models Qwen/Qwen2.5-7B\""
  exit 1
fi

# Guard: refuse to submit a python script that isn't present on the node yet
# (scripts must be pulled from GitHub before they can run — see README "Repo sync").
SCRIPT=$(echo "$CMD" | grep -oE "code/[A-Za-z0-9_./-]+\.py" | head -1 || true)
if [[ -n "$SCRIPT" && ! -f "$PROJ/$SCRIPT" ]]; then
  echo "ERROR: $SCRIPT not found on the node."
  echo "Pull it from GitHub first:  git pull origin main"
  exit 1
fi

PART="${PART:-mit_normal_gpu}"      # GPU partition (see: sinfo -h -o '%R %G' | grep gpu)
TIME="${TIME:-08:00:00}"
MEM="${MEM:-64G}"
CPUS="${CPUS:-8}"
GPUS="${GPUS:-1}"
JOBNAME="${JOBNAME:-gpu_tom}"

mkdir -p "$PROJ/outputs/logs"

sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=$JOBNAME
#SBATCH --partition=$PART
#SBATCH --gres=gpu:$GPUS
#SBATCH --time=$TIME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --output=$PROJ/outputs/logs/gpu_%j.log
#SBATCH --error=$PROJ/outputs/logs/gpu_%j.log

set -uo pipefail
cd $PROJ

echo "=== Job \${SLURM_JOB_ID} on \$(hostname) at \$(date) ==="
echo "CMD: $CMD"
echo ""

# HF cache on scratch (home has a quota). Do NOT force offline: caches are currently
# empty, so weights may need to download (requires network on the GPU node).
export HF_HOME="\${HF_HOME:-/orcd/scratch/orcd/007/$USER/hf_cache}"
mkdir -p "\$HF_HOME"
[ -f "\$HOME/.cache/huggingface/token" ] && export HF_TOKEN="\${HF_TOKEN:-\$(cat "\$HOME/.cache/huggingface/token")}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

module purge 2>/dev/null || true
module load miniforge/24.3.0-0 2>/dev/null || module load anaconda3 2>/dev/null || true
module load cuda/12.4.0 2>/dev/null || module load cuda/12.1 2>/dev/null || true
source .venv/bin/activate

# Load API keys if present (needed for closed-model / OSF-fetch steps)
[ -f .env_agents ] && source .env_agents

nvidia-smi || true
python -c "import torch;print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
echo ""

$CMD
RC=\$?

echo ""
echo "=== Job \${SLURM_JOB_ID} finished (exit \$RC) at \$(date) ==="
exit \$RC
EOF

echo ""
echo "Submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f $PROJ/outputs/logs/gpu_<JOBID>.log"
