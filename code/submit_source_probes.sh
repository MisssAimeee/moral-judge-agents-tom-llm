#!/bin/bash
#SBATCH --job-name=src_probes
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/logs/cpu_%j.log

# C2: the pre-outcome reading of belief_last is only valid if the outcome-determining
# sentence comes AFTER the belief clause. That ordering differs by source — YS2008 puts
# the true state of the world before the belief clause, YS2009 does not. If outcome
# decoding at belief_last is high for YS2008 and at chance for YS2009, the "harm not yet
# stated" reading is wrong for 192 of 298 items and the figure caption must stay neutral.
#
# Reads cached activations, so this is CPU only.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

echo "=== Job $SLURM_JOB_ID on $(hostname) at $(date) ==="
for SRC in YS2008 YS2009; do
  for POOL in belief_last action_last; do
    echo "--- source=$SRC pooling=$POOL ---"
    python -u code/02_probe.py --pooling "$POOL" --source "$SRC"
  done
done
echo "=== done at $(date) ==="
