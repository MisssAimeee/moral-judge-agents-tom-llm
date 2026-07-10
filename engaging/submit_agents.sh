#!/usr/bin/env bash
# submit_agents.sh -- Submit the agent LLM behavioral pipeline as a Slurm batch job.
#
# Usage (from login node or any node):
#   bash engaging/submit_agents.sh             # runs all providers with keys in .env_agents
#   bash engaging/submit_agents.sh google      # runs only Google Gemini
#   bash engaging/submit_agents.sh openai      # runs only OpenAI
#
# Outputs:
#   outputs/logs/agents_<JOBID>.log  ← full stdout/stderr
#   outputs/agents/behavior/         ← raw ratings CSVs
#   outputs/agents/figures/          ← 7 PNG figures
#   outputs/agents/stats/            ← contrast_by_model.csv
#
# Check job status: squeue -u $USER
# Watch log live:   tail -f outputs/logs/agents_<JOBID>.log

PROVIDER="${1:-all}"
PROJ=/home/aimeeyu/tom_project

# Verify .env_agents exists (it holds API keys; never commit this file)
if [[ ! -f "$PROJ/.env_agents" ]]; then
  echo "ERROR: $PROJ/.env_agents not found."
  echo "Create it with lines like: export GOOGLE_API_KEY=\"AIza...\""
  exit 1
fi

mkdir -p "$PROJ/outputs/logs"

BEGIN="${BEGIN:-}"   # e.g. BEGIN=2026-07-02T07:05:00 to defer start

sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=agents_tom
#SBATCH --partition=mit_normal              # CPU partition, 12h limit, internet access
#SBATCH --time=08:00:00                     # 8h wall time — Flash+Pro takes ~2.5h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=$PROJ/outputs/logs/agents_%j.log
#SBATCH --error=$PROJ/outputs/logs/agents_%j.log
$([ -n "$BEGIN" ] && echo "#SBATCH --begin=$BEGIN")

set -uo pipefail
cd $PROJ

echo "=== Job \${SLURM_JOB_ID} started on \$(hostname) at \$(date) ==="
echo "Provider(s): $PROVIDER"
echo ""

# Load API keys from .env_agents (never hardcode them in this script)
source .env_agents

# Activate Python venv
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

bash engaging/run_agents.sh $PROVIDER

echo ""
echo "=== Job \${SLURM_JOB_ID} finished at \$(date) ==="
EOF

echo ""
echo "Submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f $PROJ/outputs/logs/agents_<JOBID>.log"
