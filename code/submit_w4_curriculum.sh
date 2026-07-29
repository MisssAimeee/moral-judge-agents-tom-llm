#!/bin/bash
#SBATCH --job-name=w4_curric
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/logs/gpu_%j.log

# W4 prompt curriculum -- the companion to W3, run in parallel with the closed-model job.
#
# W3 showed the intent code is decodable and causally inert to residual-stream
# intervention. W4 asks whether it is reachable from the input instead: five cumulative
# levels of in-context scaffolding, no weight updates, contrast measured at each level on
# the same 7-template basis and the same scenario-group averaging as the headline number.
#
# Both readings are pre-registered in the script docstring and written to
# outputs/experiments/W4_PRESPEC.md on the first run, before any result exists.
#
# 10,430 forward passes per model x 6 engaged models. Results are appended per model, so a
# preemption keeps everything already scored and --run resumes at the next (model, level).
# Model order puts the two W3 models first so a short run still yields the W3/W4 pair.
#
# A 0.5B smoke test runs first (2 scenario groups, 1 template) and the job aborts if it
# fails; W3 run 1 was lost to a bug that only appeared at full scale.

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
echo "================ SMOKE TEST (Qwen2.5-0.5B-Instruct) ================"
python -u code/experiments/54_w4_prompt_curriculum.py --run \
  --models Qwen/Qwen2.5-0.5B-Instruct --templates wrong_w1 --limit-groups 2 --boot 200
SMOKE=$?
# 0.5B is far below the engagement floor; drop its rows so it cannot reach the report.
python - <<'PY'
import csv, os
for p in ("outputs/experiments/w4_prompt_curriculum.csv",
          "outputs/experiments/w4_curriculum_cells.csv"):
    if not os.path.exists(p):
        continue
    rows = [r for r in csv.DictReader(open(p)) if "0.5B" not in r["model"]]
    if rows:
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    else:
        os.remove(p)
    print("cleaned", p)
PY
if [ $SMOKE -ne 0 ]; then
  echo "!! smoke test failed ($SMOKE) -- aborting before the real run"
  exit 1
fi
echo "smoke test passed; 0.5B rows removed"

echo ""
echo "================ FULL CURRICULUM ================"
python -u code/experiments/54_w4_prompt_curriculum.py --run || echo "!! curriculum failed"

echo ""
echo "================ report ================"
python -u code/experiments/55_w4_summary.py || echo "!! W4 summary failed"

echo "=== done at $(date) ==="
