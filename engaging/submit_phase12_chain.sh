#!/usr/bin/env bash
# submit_phase12_chain.sh — YS2011 re-extract → force B3 → P1 (C4,C5) ‖ P2 (RSA×4)
set -uo pipefail
cd "$(dirname "$0")/.."
PROJ="$PWD"
GPU="engaging/submit_gpu.sh"
CPU="engaging/submit_cpu.sh"
GPU_PART="mit_preemptable"
CPU_PART="mit_normal"

ACT_MODELS="Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-0.5B-Instruct \
Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-1.5B-Instruct \
Qwen/Qwen2.5-7B Qwen/Qwen2.5-7B-Instruct \
allenai/OLMo-2-1124-7B allenai/OLMo-2-1124-7B-Instruct"

STORIES="YS2011-Poison-accidental YS2011-Parent-accidental"
CSV="dataset/master/moral_2x2_master.csv"
OFFS="dataset/master/clause_offsets.csv"

submit() {
  local runner="$1" name="$2" dep="$3"; shift 3
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done
  shift
  DEP="$dep" PARSABLE=1 JOBNAME="$name" env "${envs[@]}" bash "$runner" "$@"
}

echo "=== phase12 chain $(date +%Y%m%d-%H%M) ==="

# --- surgical YS2011 re-extract (merge into existing npz) ---------------------
Y1=$(submit "$GPU" ys2011_acts "" PART=$GPU_PART TIME=04:00:00 MEM=96G GPUS=1 -- \
  python -u code/01_extract_activations.py --csv "$CSV" --clause-offsets "$OFFS" \
  --out outputs/acts --merge-existing --story-ids $STORIES --models $ACT_MODELS)
echo "Y1 ys2011 acts          = $Y1"

# --- re-probe all poolings after merge ----------------------------------------
P_LAST=$(submit "$GPU" probe_last "afterok:$Y1" PART=$GPU_PART TIME=02:00:00 MEM=64G GPUS=1 -- \
  python -u code/02_probe.py --acts outputs/acts --out outputs/probe --pooling last \
  --clause-offsets "$OFFS" --csv "$CSV")
P_MEAN=$(submit "$GPU" probe_mean "afterok:$Y1" PART=$GPU_PART TIME=02:00:00 MEM=64G GPUS=1 -- \
  python -u code/02_probe.py --acts outputs/acts --out outputs/probe --pooling mean \
  --clause-offsets "$OFFS" --csv "$CSV")
P_BEL=$(submit "$GPU" probe_belief "afterok:$Y1" PART=$GPU_PART TIME=02:00:00 MEM=64G GPUS=1 -- \
  python -u code/02_probe.py --acts outputs/acts --out outputs/probe --pooling belief_last \
  --clause-offsets "$OFFS" --csv "$CSV")
P_ACT=$(submit "$GPU" probe_action "afterok:$Y1" PART=$GPU_PART TIME=02:00:00 MEM=64G GPUS=1 -- \
  python -u code/02_probe.py --acts outputs/acts --out outputs/probe --pooling action_last \
  --clause-offsets "$OFFS" --csv "$CSV")
echo "probes                  = $P_LAST $P_MEAN $P_BEL $P_ACT"
PROBE_DEP="afterok:$P_LAST:$P_MEAN:$P_BEL:$P_ACT"

# --- force B3 checkpoint dissection ------------------------------------------
B3=$(submit "$GPU" ckpt_force "" PART=$GPU_PART TIME=12:00:00 MEM=96G GPUS=1 -- \
  python -u code/experiments/16_checkpoint_dissection.py --run --force)
echo "B3 ckpt --force         = $B3"

# --- P1 C4: permutation null (peak + L0, all poolings) -----------------------
# 02_probe --permute N runs null at layer 0 and peak only
C4A=$(submit "$CPU" perm_last "$PROBE_DEP" PART=$CPU_PART TIME=08:00:00 MEM=64G CPUS=16 -- \
  python -u code/02_probe.py --acts outputs/acts --out outputs/probe --pooling last \
  --permute 1000 --skip-probe --clause-offsets "$OFFS" --csv "$CSV")
C4B=$(submit "$CPU" perm_mean "$PROBE_DEP" PART=$CPU_PART TIME=08:00:00 MEM=64G CPUS=16 -- \
  python -u code/02_probe.py --acts outputs/acts --out outputs/probe --pooling mean \
  --permute 1000 --skip-probe --clause-offsets "$OFFS" --csv "$CSV")
C4C=$(submit "$CPU" perm_belief "$PROBE_DEP" PART=$CPU_PART TIME=08:00:00 MEM=64G CPUS=16 -- \
  python -u code/02_probe.py --acts outputs/acts --out outputs/probe --pooling belief_last \
  --permute 1000 --skip-probe --clause-offsets "$OFFS" --csv "$CSV")
C4D=$(submit "$CPU" perm_action "$PROBE_DEP" PART=$CPU_PART TIME=08:00:00 MEM=64G CPUS=16 -- \
  python -u code/02_probe.py --acts outputs/acts --out outputs/probe --pooling action_last \
  --permute 1000 --skip-probe --clause-offsets "$OFFS" --csv "$CSV")
echo "C4 permnull             = $C4A $C4B $C4C $C4D"

# --- P1 C5: intent_reliance summary + link analysis (no GPU dep) -------------
C5=$(submit "$CPU" c5_link "" PART=$CPU_PART TIME=01:00:00 MEM=16G CPUS=4 -- \
  bash -c 'python -u code/experiments/23_build_intent_reliance_summary.py --behavior outputs/behavior && python -u code/04_link_analysis.py')
echo "C5 intent_reliance+link = $C5"

# --- P2 RSA: all 4 poolings in parallel (after probes) -----------------------
R_MEAN=$(submit "$CPU" rsa_mean "$PROBE_DEP" PART=$CPU_PART TIME=06:00:00 MEM=64G CPUS=16 -- \
  python -u code/experiments/24_rsa_cka.py --pooling mean --perm 1000 --out outputs/rsa)
R_LAST=$(submit "$CPU" rsa_last "$PROBE_DEP" PART=$CPU_PART TIME=06:00:00 MEM=64G CPUS=16 -- \
  python -u code/experiments/24_rsa_cka.py --pooling last --perm 1000 --out outputs/rsa_last)
R_BEL=$(submit "$CPU" rsa_belief "$PROBE_DEP" PART=$CPU_PART TIME=06:00:00 MEM=64G CPUS=16 -- \
  python -u code/experiments/24_rsa_cka.py --pooling belief_last --perm 1000 --out outputs/rsa_belief_last)
R_ACT=$(submit "$CPU" rsa_action "$PROBE_DEP" PART=$CPU_PART TIME=06:00:00 MEM=64G CPUS=16 -- \
  python -u code/experiments/24_rsa_cka.py --pooling action_last --perm 1000 --out outputs/rsa_action_last)
echo "P2 RSA×4                = $R_MEAN $R_LAST $R_BEL $R_ACT"

# --- dissociation figure (CPU, immediate) ------------------------------------
FIG=$(submit "$CPU" gap_fig "" PART=$CPU_PART TIME=00:20:00 MEM=8G CPUS=2 -- \
  python -u code/experiments/33_gap_dissociation_figure.py)
echo "gap figure              = $FIG"

echo "=== monitor: squeue -u \$USER ==="
