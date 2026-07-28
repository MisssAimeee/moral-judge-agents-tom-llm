#!/bin/bash
#SBATCH --job-name=v3_refresh
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/cpu_%j.log

# Housekeeping: three probe-side outputs were generated against the v2 activations and are
# compared in the write-up against main probes generated against v3.
#
#   outputs/probe/surface_baseline.csv     2026-07-26 23:42
#   outputs/probe/layer0_diagnostic.csv    2026-07-26 23:56
#   outputs/probe/*_withincell.csv         2026-07-26 23:57
#   outputs/acts/*.npz  (v3)               2026-07-27 11:09-11:17   <-- all newer
#   outputs/probe/*_probe.csv  (v3)        2026-07-27 11:20-11:27
#
# The options were to re-run on v3 or to annotate the mismatch. Re-running is correct and
# nearly free: all three read cached activations, so this is CPU only and no model is
# loaded. Annotating would have left a comparison between two activation versions standing
# in the results.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

echo "=== Job $SLURM_JOB_ID on $(hostname) at $(date) ==="
echo "activation mtimes going in:"
ls -la --time-style=+%Y-%m-%d_%H:%M outputs/acts/*.npz | awk '{print "  "$6, $7}'

echo "--- 21_surface_baseline ---"
python -u code/experiments/21_surface_baseline.py

echo "--- 20_layer0_diagnostic ---"
python -u code/experiments/20_layer0_diagnostic.py

echo "--- 22_within_cell_probes ---"
python -u code/experiments/22_within_cell_probes.py

echo "=== done at $(date) ==="
