#!/bin/bash
# =============================================================================
# Train the wiretype encoder on FAFB (scripts/train.py). The defaults are the
# paper's configuration: the `refined` 31
# columns, split_random, 24,000 steps, supervised on nt_best, k = 64.
#
#   qsub -v TAG=s24k_ntbest hpc/jobs/train.sh                 # the paper model, seed 0
#   qsub -v SEED=1,TAG=s24k_ntbest_seed1 hpc/jobs/train.sh    # more seeds
#   qsub -v VOLUME=mcns,SCOPE=brain_neurons,TAG=s24k_ntbest_brain hpc/jobs/train.sh   # reverse
#   qsub -v STEPS=100,PROBE_EVERY=50,NO_PROBE=1,TAG=smoke hpc/jobs/train.sh
#
# Writes to experiments/train/: checkpoint_fafb_{untrained,displacement}_<set>_
# <split>_<tag>.pt, the report train_fafb_<...>.json, and float16 embeddings per arm.
# The untrained checkpoint is the matched floor for transfer.
#
# Read `oracle` first: the true displacement, probed on the supervised class. If it
# does not clear the untrained floor comfortably, the read-out is broken.
#
# Needs data/processed/fafb_topk64.npz (hpc/jobs/topk.sh) and the feature tables.
#
# Time: about 5 h. probe_all costs about 1 h per probed arm, whatever STEPS is.
# The paper's cluster offered at most 32G of h_rss on its MIG nodes.
# =============================================================================
#$ -N wiretype_train
#$ -cwd
#$ -o logs/
#$ -e logs/
#$ -l h_rt=12:00:00
#$ -q gpu
#$ -l gpu-mig=1
#$ -l h_rss=32G

set -euo pipefail
source hpc/config.sh
. /etc/profile.d/modules.sh
module load "${ANACONDA_MODULE}"
module load "${CUDA_MODULE}" 2>/dev/null || echo "note: '${CUDA_MODULE}' unavailable; using torch's bundled CUDA"
if [ ! -d "${ENV_PREFIX}" ]; then
    echo "ERROR: training env not found: ${ENV_PREFIX}"; exit 1
fi
set +u; source activate "${ENV_PREFIX}"; set -u
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VOLUME="${VOLUME:-fafb}"; K="${K:-64}"; SET="${SET:-refined}"; SPLIT="${SPLIT:-split_random}"
# `qsub -v` splits on commas, so multi-valued FEATURES and ARMS take `+` instead.
FEATURES="${FEATURES:-degree+connection+rwse+flow+local}"; FEATURES="${FEATURES//+/,}"
ARMS="${ARMS:-untrained+oracle+displacement}"; ARMS="${ARMS//+/,}"
SUPERVISE_ON="${SUPERVISE_ON:-nt_best}"; STEPS="${STEPS:-24000}"; BATCH="${BATCH:-1024}"
LR="${LR:-1e-3}"; PROBE_EVERY="${PROBE_EVERY:-1000}"; LOG_EVERY="${LOG_EVERY:-500}"
SEED="${SEED:-0}"; TAG="${TAG:-}"; NO_PROBE="${NO_PROBE:-}"; NO_AMP="${NO_AMP:-}"
SCOPE="${SCOPE:-brain_neurons}"  # brain neurons with at least one connection (src/wiretype/data/scope.py)

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "train volume=${VOLUME} scope=${SCOPE} set=${SET} split=${SPLIT} arms=${ARMS} features=${FEATURES} k=${K} supervise_on=${SUPERVISE_ON} steps=${STEPS} seed=${SEED} tag=${TAG:-<none>}: $(date)"
PYTHONPATH=src python scripts/train.py \
    --volume "${VOLUME}" -k "${K}" --features "${FEATURES}" --column-set "${SET}" --arms "${ARMS}" \
    --supervise-on "${SUPERVISE_ON}" --split-column "${SPLIT}" --scope "${SCOPE}" \
    --steps "${STEPS}" --batch "${BATCH}" --lr "${LR}" \
    --probe-every "${PROBE_EVERY}" --log-every "${LOG_EVERY}" --seed "${SEED}" --tag "${TAG}" \
    ${NO_PROBE:+--no-final-probe} ${NO_AMP:+--no-amp} \
    --processed data/processed --reports experiments/train
echo "finished: $(date)"
