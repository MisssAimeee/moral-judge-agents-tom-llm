#!/usr/bin/env bash
# submit_overnight_chain.sh -- regenerate every downstream result from the repaired stimulus
# master, as an sbatch dependency chain.
#
# Everything produced before 2026-07-26 used the contaminated master (see
# dataset/master/CONTAMINATION_REPAIR.md) and is invalid. The contaminated outputs have been
# moved to outputs/_contaminated_20260726/ so they survive for the before/after comparison and
# so --skip_existing does not silently skip the rerun.
#
# afterok is used throughout: a failed stage halts its branch rather than letting the next stage
# run on missing or stale inputs, which is how bad data cascaded last time.
#
# Usage:  bash engaging/submit_overnight_chain.sh          # submit
#         DRYRUN=1 bash engaging/submit_overnight_chain.sh # print the plan only
set -uo pipefail
cd "$(dirname "$0")/.."
PROJ="$PWD"

DRYRUN="${DRYRUN:-}"
STAMP=$(date +%Y%m%d)
CSV="dataset/master/moral_2x2_master.csv"
OFFS="dataset/master/clause_offsets.csv"

GPU="engaging/submit_gpu.sh"
CPU="engaging/submit_cpu.sh"

# Preemptable gives long walltimes; every script resumes with --skip_existing.
GPU_PART="mit_preemptable"
CPU_PART="mit_normal"

ACT_MODELS="Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-0.5B-Instruct \
Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-1.5B-Instruct \
Qwen/Qwen2.5-7B Qwen/Qwen2.5-7B-Instruct \
allenai/OLMo-2-1124-7B allenai/OLMo-2-1124-7B-Instruct"

# Every open-weight model on the behavioural ladder. Llama-3.1-8B uses the unsloth mirror
# because the Meta repo gate is still pending.
BEH_MODELS="Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-0.5B-Instruct \
Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-1.5B-Instruct \
Qwen/Qwen2.5-3B Qwen/Qwen2.5-3B-Instruct \
Qwen/Qwen2.5-7B Qwen/Qwen2.5-7B-Instruct \
Qwen/Qwen2.5-14B Qwen/Qwen2.5-14B-Instruct \
allenai/OLMo-2-1124-7B allenai/OLMo-2-1124-7B-Instruct \
allenai/Llama-3.1-Tulu-3-8B HuggingFaceH4/zephyr-7b-beta \
mistralai/Mistral-7B-v0.3 mistralai/Mistral-7B-Instruct-v0.3 \
unsloth/gemma-2-9b unsloth/gemma-2-9b-it \
unsloth/Meta-Llama-3.1-8B unsloth/Meta-Llama-3.1-8B-Instruct"

submit() {  # submit <runner> <jobname> <dep> <env-assignments...> -- <command>
  local runner="$1" name="$2" dep="$3"; shift 3
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done
  shift
  if [ -n "$DRYRUN" ]; then
    # plan to stderr, fake id to stdout, so the captured value stays a bare id
    { echo "[$name] dep='${dep}'  ${envs[*]}"; echo "    $*"; } >&2
    echo "0000${name}"
    return
  fi
  DEP="$dep" PARSABLE=1 JOBNAME="$name" env "${envs[@]}" bash "$runner" "$@"
}

echo "=== overnight chain $STAMP ==="

# --- B1: activations, 8 models x 4 pooling variants -------------------------------------
B1=$(submit "$GPU" acts "" PART=$GPU_PART TIME=08:00:00 MEM=96G GPUS=1 -- \
  python -u code/01_extract_activations.py --csv "$CSV" --clause-offsets "$OFFS" \
  --out outputs/acts --models $ACT_MODELS)
echo "B1 activations          = $B1"

# --- B2: behavioural rescore, independent of B1 so they run in parallel -----------------
B2=$(submit "$GPU" rescore "" PART=$GPU_PART TIME=12:00:00 MEM=96G GPUS=1 -- \
  python -u code/03_behavioral.py --backend hf --scoring logprob --csv "$CSV" \
  --out_dir outputs/behavior --skip_existing --models $BEH_MODELS)
echo "B2 behavioural rescore  = $B2"

# --- B6: TF-IDF surface baseline, needs only the repaired dataset -----------------------
B6=$(submit "$CPU" surface "" PART=$CPU_PART TIME=02:00:00 MEM=32G CPUS=8 -- \
  python -u code/experiments/21_surface_baseline.py --csv "$CSV")
echo "B6 surface baseline     = $B6"

# --- B4/B8: layer-wise probes, one job per pooling variant, all after B1 ----------------
B4_IDS=""
for POOL in last mean belief_last action_last; do
  J=$(submit "$CPU" "probe_$POOL" "afterok:$B1" PART=$CPU_PART TIME=08:00:00 MEM=64G CPUS=16 -- \
    python -u code/02_probe.py --csv "$CSV" --acts outputs/acts --out outputs/probe \
    --pooling "$POOL" --clause-offsets "$OFFS")
  echo "B4 probe/$POOL$([ ${#POOL} -lt 8 ] && echo -e '\t')  = $J"
  B4_IDS="${B4_IDS:+$B4_IDS:}$J"
done

# --- B5: layer-0 read-off diagnostic ----------------------------------------------------
B5=$(submit "$CPU" layer0 "afterok:$B4_IDS" PART=$CPU_PART TIME=04:00:00 MEM=64G CPUS=16 -- \
  python -u code/experiments/20_layer0_diagnostic.py --csv "$CSV")
echo "B5 layer-0 diagnostic   = $B5"

# --- B7: within-cell contrast probes ----------------------------------------------------
B7=$(submit "$CPU" withincell "afterok:$B4_IDS" PART=$CPU_PART TIME=08:00:00 MEM=64G CPUS=16 -- \
  python -u code/experiments/22_within_cell_probes.py --csv "$CSV")
echo "B7 within-cell probes   = $B7"

# --- B3: checkpoint dissection ----------------------------------------------------------
B3=$(submit "$GPU" ckpt "afterok:$B2" PART=$GPU_PART TIME=12:00:00 MEM=96G GPUS=1 -- \
  python -u code/experiments/16_checkpoint_dissection.py --run)
echo "B3 checkpoint dissection= $B3"

# --- B9: master ladder ------------------------------------------------------------------
B9=$(submit "$CPU" ladder "afterok:$B2:$B3" PART=$CPU_PART TIME=01:00:00 MEM=16G CPUS=4 -- \
  python -u code/10_master_figure.py)
echo "B9 master ladder        = $B9"

cat <<TXT

dependency graph
  B1 acts ──┬─ B4 probe x4 ──┬─ B5 layer0
            │                └─ B7 within-cell
  B2 rescore ─── B3 ckpt ────── B9 ladder  (B9 also waits on B2)
  B6 surface (independent)

not queued, by instruction: permutation null (C4), closed-API rescoring (needs budget
approval), Phases 5-7 of confound_and_rsa_plan.md
TXT
