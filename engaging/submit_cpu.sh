#!/usr/bin/env bash
# submit_cpu.sh -- Submit a CPU-only analysis command as a disconnect-safe Slurm batch job.
#
# Same contract as submit_gpu.sh but requests NO GPU. Use this for every analysis step that
# does not load model weights: probes, permutation nulls, RSA/CKA, surface baselines, stats.
# Two reasons it matters:
#   1. The GPU QOS allows only 1 GPU per user, so a CPU job that grabs a GPU blocks the
#      extraction jobs that genuinely need one.
#   2. CPU partitions queue faster.
#
# Usage:
#   bash engaging/submit_cpu.sh "python code/02_probe.py --pooling mean"
#   JOBNAME=perm TIME=08:00:00 CPUS=16 bash engaging/submit_cpu.sh "python code/02_probe.py --permute 1000"
#
# Monitor:  squeue -u $USER ; tail -f outputs/logs/cpu_<JOBID>.log
# Cancel:   scancel <JOBID>

set -uo pipefail
PROJ=/home/aimeeyu/tom_project
CMD="${*:-}"

if [[ -z "$CMD" ]]; then
  echo "ERROR: give the command to run, e.g."
  echo "  bash engaging/submit_cpu.sh \"python code/02_probe.py --pooling mean\""
  exit 1
fi

SCRIPT=$(echo "$CMD" | grep -oE "code/[A-Za-z0-9_./-]+\.py" | head -1 || true)
if [[ -n "$SCRIPT" && ! -f "$PROJ/$SCRIPT" ]]; then
  echo "ERROR: $SCRIPT not found on the node."
  exit 1
fi

PART="${PART:-mit_normal}"
TIME="${TIME:-08:00:00}"
MEM="${MEM:-64G}"
CPUS="${CPUS:-16}"
JOBNAME="${JOBNAME:-cpu_tom}"
# DEP=afterok:12345[:67890] chains this job behind others so a failure halts the chain
# instead of letting the next stage run on missing or stale inputs.
DEP="${DEP:-}"
SBATCH_ARGS=()
[ -n "$DEP" ] && SBATCH_ARGS+=(--dependency="$DEP" --kill-on-invalid-dep=yes)
# PARSABLE=1 prints just the job id, so a chain script can capture it
[ -n "${PARSABLE:-}" ] && SBATCH_ARGS+=(--parsable)

mkdir -p "$PROJ/outputs/logs"

sbatch "${SBATCH_ARGS[@]}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=$JOBNAME
#SBATCH --partition=$PART
#SBATCH --time=$TIME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --output=$PROJ/outputs/logs/cpu_%j.log
#SBATCH --error=$PROJ/outputs/logs/cpu_%j.log

set -uo pipefail
cd $PROJ

echo "=== Job \${SLURM_JOB_ID} on \$(hostname) at \$(date) ==="
echo "CMD: $CMD"
echo "CPUS: $CPUS"
echo ""

# Keep BLAS single-threaded: joblib parallelises across permutations, and nested
# threading would oversubscribe the cores and run SLOWER than serial.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export JOBLIB_N_JOBS=$CPUS

module purge 2>/dev/null || true
module load miniforge/24.3.0-0 2>/dev/null || module load anaconda3 2>/dev/null || true
source .venv/bin/activate
[ -f .env_agents ] && source .env_agents

$CMD
RC=\$?

echo ""
echo "=== Job \${SLURM_JOB_ID} finished (exit \$RC) at \$(date) ==="
exit \$RC
EOF

echo ""
echo "Submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f $PROJ/outputs/logs/cpu_<JOBID>.log"
