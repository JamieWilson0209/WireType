#!/bin/bash
# =============================================================================
# Apply a trained encoder, frozen, to another volume (scripts/transfer.py).
# CKPT is required.
#
#   C=experiments/train/checkpoint_fafb_
#   qsub -v CKPT=${C}displacement_refined_split_random_s24k_ntbest.pt,DUMP=1,EMB=1 hpc/jobs/transfer.sh
#   qsub -v CKPT=${C}untrained_refined_split_random_s24k_ntbest.pt,DUMP=1 hpc/jobs/transfer.sh   # floor
#   qsub -v CKPT=<ckpt>,STD=own_neurons hpc/jobs/transfer.sh      # scaling ablation
#   qsub -v CKPT=<ckpt>,IN_VOLUME=1 hpc/jobs/transfer.sh          # + in-volume diagnostic, ~1 h more
#   qsub -v CKPT=<mcns ckpt>,SOURCE=mcns,TARGET=fafb hpc/jobs/transfer.sh   # reverse direction
#
# DUMP=1 writes per-cell predictions (scripts/breakdown.py); EMB=1 writes both
# volumes' embeddings (scripts/score_embeddings.py). STD defaults to source.
# RELEASE=1 (with DUMP=1,EMB=1) is the release run: both released transmitter
# probes call every connected target brain neuron, the fitted probes are saved and
# the embeddings are float32 (scripts/release.py). REPORTS sets the output folder
# (default experiments/transfer).
#
# The target needs its feature tables, splits and top-k first; the script names
# any that are missing.
#
# Writes ${REPORTS}/transfer_<source>_to_<target>_<ckpt>_<std>.json.
# The paper's cluster offered at most 32G of h_rss on its MIG nodes.
# =============================================================================
#$ -N wiretype_transfer
#$ -cwd
#$ -o logs/
#$ -e logs/
#$ -l h_rt=08:00:00
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

CKPT="${CKPT:?set CKPT to a training checkpoint .pt}"
SOURCE="${SOURCE:-fafb}"; TARGET="${TARGET:-mcns}"; STD="${STD:-source}"
TARGET_SPLIT="${TARGET_SPLIT:-split_random}"; SEED="${SEED:-0}"; NO_AMP="${NO_AMP:-}"
REPORTS="${REPORTS:-experiments/transfer}"; RELEASE="${RELEASE:-}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "transfer ${CKPT} ${SOURCE}->${TARGET} standardise=${STD} target_split=${TARGET_SPLIT}${RELEASE:+ (release)} -> ${REPORTS}: $(date)"
PYTHONPATH=src python scripts/transfer.py \
    --checkpoint "${CKPT}" --source "${SOURCE}" --target "${TARGET}" \
    --standardise-with "${STD}" --target-split-column "${TARGET_SPLIT}" --seed "${SEED}" \
    ${NO_AMP:+--no-amp} ${DUMP:+--dump-predictions} ${EMB:+--save-embeddings} ${IN_VOLUME:+--in-volume} \
    ${RELEASE:+--predict-all --save-probes --embeddings-dtype float32} \
    --processed data/processed --reports "${REPORTS}"
echo "finished: $(date)"
