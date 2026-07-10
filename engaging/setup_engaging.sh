#!/usr/bin/env bash
# setup_engaging.sh -- one-time environment build on an MIT Engaging / ORCD LOGIN node.
# Run this on the LOGIN node (it has internet); compute nodes usually do NOT.
#
#   bash engaging/setup_engaging.sh
#
# It (1) loads modules, (2) builds .venv, (3) installs deps, and
# (4) PRE-DOWNLOADS every model into HF_HOME so the GPU job can run fully OFFLINE.
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root

# --- 0. EDIT THESE for your allocation -------------------------------------
# Put the HF cache on scratch (home dirs have small quotas). Check yours.
export HF_HOME="${HF_HOME:-/pool001/$USER/hf_cache}"   # or /nobackup1/$USER/hf_cache
# Gated models (Llama) need a token from https://huggingface.co/settings/tokens
# export HF_TOKEN="hf_xxx"        # uncomment + paste, OR run: huggingface-cli login
# ---------------------------------------------------------------------------

# --- 1. modules (names vary; run `module avail` to confirm on your cluster) -
module purge || true
# Newer ORCD stack, e.g.:
module load miniforge/24.3.0-0 2>/dev/null || module load anaconda3 2>/dev/null || true
module load cuda/12.4.0        2>/dev/null || module load cuda/12.1 2>/dev/null || true

# --- 2. virtual environment -------------------------------------------------
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r code/requirements.txt
# CUDA build of torch is selected automatically by pip on a Linux GPU node.

# --- 3. (re)build the master CSV if it isn't present ------------------------
if [ ! -f dataset/master/moral_2x2_master.csv ]; then
  echo "master CSV missing -> rebuilding from PDFs"
  python code/build_dataset.py
fi

# --- 4. pre-download models so the compute node can run offline ------------
mkdir -p "$HF_HOME"
# Base AND instruct at every size: lets 06_stats.py separate the size effect from
# the instruction-tuning effect (the base-vs-instruct confound).
MODELS=(
  "Qwen/Qwen2.5-0.5B"          "Qwen/Qwen2.5-0.5B-Instruct"
  "Qwen/Qwen2.5-1.5B"          "Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen/Qwen2.5-3B"            "Qwen/Qwen2.5-3B-Instruct"
  "Qwen/Qwen2.5-7B"            "Qwen/Qwen2.5-7B-Instruct"
  "meta-llama/Llama-3.1-8B"    "meta-llama/Llama-3.1-8B-Instruct"   # gated: needs HF_TOKEN
)
for m in "${MODELS[@]}"; do
  echo "==> caching $m"
  python - "$m" <<'PY'
import sys
from huggingface_hub import snapshot_download
m = sys.argv[1]
try:
    snapshot_download(m, allow_patterns=["*.json","*.txt","*.model","*.safetensors","tokenizer*"])
    print("   cached:", m)
except Exception as e:
    print("   SKIP (need access/token?):", m, "->", e)
PY
done
echo "Setup complete. HF_HOME=$HF_HOME"
echo "Next: sbatch engaging/run_behavioral.sbatch"
