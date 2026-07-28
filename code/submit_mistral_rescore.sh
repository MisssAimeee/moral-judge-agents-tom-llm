#!/bin/bash
#SBATCH --job-name=mistral_fix
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/gpu_%j.log

# Rescore the three models whose ratings were fabricated by the digit-token collapse
# (all digits mapped to the SentencePiece dummy-prefix token, so every rating came back
# as the scale midpoint). para_blame10 is omitted: these tokenizers have no single-token
# id for "10", so logprob scoring cannot represent that scale point at all.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

echo "=== Job $SLURM_JOB_ID on $(hostname) at $(date) ==="
nvidia-smi
python -c "import torch;print('CUDA:',torch.cuda.is_available())"

python -u code/03_behavioral.py \
  --backend hf --scoring logprob \
  --csv dataset/master/moral_2x2_master.csv \
  --out_dir outputs/behavior \
  --templates human_verbatim blame_w1 blame_w2 wrong_w1 wrong_w2 punish_w1 punish_w2 \
              para_blame4 acceptable7 persona_adult7 \
  --models HuggingFaceH4/zephyr-7b-beta \
           mistralai/Mistral-7B-v0.3 \
           mistralai/Mistral-7B-Instruct-v0.3

echo "=== done at $(date) ==="
