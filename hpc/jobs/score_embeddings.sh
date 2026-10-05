#!/bin/bash
# =============================================================================
# Score a FAFB/MCNS embedding pair by the zero-shot protocol. See
# scripts/score_embeddings.py. NAME is required.
#
#   qsub -v NAME=sage_ntbest_s0 hpc/jobs/score_embeddings.sh
#   qsub -v NAME=degree_s0,BUILD=degree hpc/jobs/score_embeddings.sh    # CPU tiers: build first
#   qsub -v NAME=encoder_check,SRC=<fafb.npy>,TGT=<mcns.npy> hpc/jobs/score_embeddings.sh
#   qsub -v NAME=raw_s0_mcns_to_fafb,BUILD=raw,SOURCE=mcns,TARGET=fafb,SCOPE=brain_neurons \
#        hpc/jobs/score_embeddings.sh        # the reverse direction's raw-features floor
#
# Embeddings default to experiments/baselines/<NAME>_embeddings_{fafb,mcns}.npy.
# BUILD=<method> first runs scripts/baselines.py for the tiers that need no
# GPU (degree, raw, composition). Writes experiments/baselines/<NAME>.json and,
# unless DUMP= (empty), <NAME>_predictions.parquet for breakdown.py.
#
# Time: the transfer jobs spent ~2.3 h on this half of their work with one
# thread. Here the logistic fits and neighbour searches get 8 cores.
#
# Uses the training env (ENV_PREFIX): the package imports torch. It needs no GPU.
# =============================================================================
#$ -N wiretype_score
#$ -cwd
#$ -o logs/
#$ -e logs/
#$ -l h_rt=08:00:00
#$ -pe sharedmem 8
#$ -l h_rss=6G

set -euo pipefail
source hpc/config.sh
. /etc/profile.d/modules.sh
module load "${ANACONDA_MODULE}"
if [ ! -d "${ENV_PREFIX}" ]; then
    echo "ERROR: training env not found: ${ENV_PREFIX}"; exit 1
fi
set +u; source activate "${ENV_PREFIX}"; set -u
export OMP_NUM_THREADS="${NSLOTS:-8}"
export OPENBLAS_NUM_THREADS="${NSLOTS:-8}"
export MKL_NUM_THREADS="${NSLOTS:-8}"

NAME="${NAME:?set NAME, e.g. sage_ntbest_s0}"
SOURCE="${SOURCE:-fafb}"; TARGET="${TARGET:-mcns}"; SCOPE="${SCOPE:-brain_neurons}"
SRC="${SRC:-experiments/baselines/${NAME}_embeddings_${SOURCE}.npy}"
TGT="${TGT:-experiments/baselines/${NAME}_embeddings_${TARGET}.npy}"
DUMP="${DUMP-1}"; SEED="${SEED:-0}"; BUILD="${BUILD:-}"

echo "score_embeddings ${NAME} build=${BUILD:-<none>} on ${NSLOTS:-8} cores: $(date)"
if [ -n "${BUILD}" ]; then
    PYTHONPATH=src python scripts/baselines.py --method "${BUILD}" --seed "${SEED}" \
        --source "${SOURCE}" --target "${TARGET}" --scope "${SCOPE}" \
        --processed data/processed --reports experiments/baselines
fi
for f in "${SRC}" "${TGT}"; do
    [ -f "$f" ] || { echo "ERROR: missing $f -- did the build job finish?"; exit 1; }
done
PYTHONPATH=src python scripts/score_embeddings.py --name "${NAME}" \
    --source-emb "${SRC}" --target-emb "${TGT}" --seed "${SEED}" \
    --source "${SOURCE}" --target "${TARGET}" --scope "${SCOPE}" \
    ${DUMP:+--dump-predictions} ${IN_VOLUME:+--in-volume} \
    --processed data/processed --reports experiments/baselines
echo "finished: $(date)"
