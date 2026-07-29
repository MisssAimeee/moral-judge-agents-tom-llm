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
# Phases per model:
#   main   -- 6 directions x ~12 alphas at the peak-intent layer, with the manipulation
#             check (probes fitted on unsteered activations, re-run on steered ones)
#   layers -- 5 depths x 4 directions x 4 alphas, each layer's directions re-fitted and
#             its coefficient range re-calibrated to that layer's own residual norm
# Then the prose/rating dissociation (51), which needs no steering.
#
# A 0.5B smoke test runs first and the job aborts if it fails: run 1 of this experiment was
# lost to a coherence-detector bug that only showed up at full scale, and 3 minutes of
# smoke test is cheaper than 25 minutes of sweep. Its outputs are deleted afterwards so the
# 0.5B tag cannot leak into the cross-model summary, which globs w3_steering_*.csv.
#
# Predictions are pre-registered in the script docstring and written to
# outputs/experiments/W3_PRESPEC.md on the first run, before any result exists.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

echo "=== Job $SLURM_JOB_ID on $(hostname) at $(date) ==="
nvidia-smi
python -c "import torch;print('CUDA:', torch.cuda.is_available())"

echo ""
echo "================ SMOKE TEST (Qwen2.5-0.5B-Instruct) ================"
python -u code/experiments/48_w3_causal_steering.py --run \
  --model Qwen/Qwen2.5-0.5B-Instruct --limit-groups 4 --n-gen 4 --gen-tokens 16 \
  --alphas 0.05 --sweep-layers 6 12 --sweep-alpha-fracs 1.0 \
  --calibration-ladder 0.05 --n-random 1
SMOKE=$?
python -u code/experiments/51_w3_prose_rating.py --run \
  --model Qwen/Qwen2.5-0.5B-Instruct --limit 8 --gen-tokens 24 --n-manual 4
SMOKE2=$?
rm -f outputs/experiments/*Qwen2.5-0.5B-Instruct*
if [ $SMOKE -ne 0 ] || [ $SMOKE2 -ne 0 ]; then
  echo "!! smoke test failed (48=$SMOKE 51=$SMOKE2) — aborting before the real sweep"
  exit 1
fi
echo "smoke test passed; 0.5B outputs removed"

for M in allenai/OLMo-2-1124-7B-Instruct Qwen/Qwen2.5-7B-Instruct; do
  echo ""
  echo "================ $M ================"
  python -u code/experiments/48_w3_causal_steering.py --run --model "$M" \
    || echo "!! $M steering failed"
  python -u code/experiments/51_w3_prose_rating.py --run --model "$M" \
    || echo "!! $M prose/rating failed"
done

echo ""
echo "================ reports ================"
python -u code/experiments/50_w3_summary.py || echo "!! summary failed"
python -u code/experiments/51_w3_prose_rating.py --report || echo "!! prose report failed"

echo "=== done at $(date) ==="
