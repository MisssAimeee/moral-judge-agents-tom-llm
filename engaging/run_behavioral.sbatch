#!/usr/bin/env bash
#SBATCH --job-name=tom_behavioral
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
# --- EDIT: set the partition/account you actually have on Engaging/ORCD -----
# Find options with:  sinfo -s   and   sacctmgr show assoc user=$USER
#SBATCH --partition=mit_normal_gpu      # e.g. mit_normal_gpu / sched_mit_hill / ou_bcs_low
##SBATCH --account=YOUR_ACCOUNT         # uncomment if your cluster requires it
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Same module + cache settings as setup_engaging.sh
export HF_HOME="${HF_HOME:-/pool001/$USER/hf_cache}"
export HF_HUB_OFFLINE=1                 # models were pre-downloaded by setup script
export TOKENIZERS_PARALLELISM=false
mkdir -p outputs/logs

module purge || true
module load miniforge/24.3.0-0 2>/dev/null || module load anaconda3 2>/dev/null || true
module load cuda/12.4.0        2>/dev/null || module load cuda/12.1 2>/dev/null || true
source .venv/bin/activate

nvidia-smi || true
python -c "import torch;print('CUDA available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# Models to test (one process each so GPU memory is freed between models).
# Base AND instruct at each size -> 06_stats.py can disentangle size vs instruction
# tuning. Base models have no chat template; 03 falls back to a plain prompt for them.
MODELS=(
  "Qwen/Qwen2.5-0.5B"          "Qwen/Qwen2.5-0.5B-Instruct"
  "Qwen/Qwen2.5-1.5B"          "Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen/Qwen2.5-3B"            "Qwen/Qwen2.5-3B-Instruct"
  "Qwen/Qwen2.5-7B"            "Qwen/Qwen2.5-7B-Instruct"
  "meta-llama/Llama-3.1-8B"    "meta-llama/Llama-3.1-8B-Instruct"
)

for m in "${MODELS[@]}"; do
  echo "=================== $m ==================="
  # --scoring logprob: deterministic expected rating from the token distribution
  # (no sampling noise). n_samples is ignored in this mode.
  python code/03_behavioral.py --backend hf --models "$m" --scoring logprob
done

# Model-vs-human comparison (human_verbatim = the paper-exact prompt) + prompt spread.
python code/05_human_comparison.py --template human_verbatim

# Inferential statistics: bootstrap CIs, cross-model + base-vs-instruct tests,
# prompt-invariance, and (if statsmodels is installed) mixed-model p-values.
python code/06_stats.py
# Figures (forest plot of CIs, profiles, prompt-invariance, size-vs-tuning, ...).
python code/07_visualize.py
# Mentor summary table (CSV + Markdown).
python code/08_report.py

echo "DONE. Results in outputs/behavior/, human/, stats/, figures/, report/"
