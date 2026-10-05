#!/bin/bash
# =============================================================================
# External baselines, GNN tiers: train on FAFB, embed FAFB and MCNS. See
# scripts/baselines.py. METHOD is required. Scoring is a separate CPU job
# (score_embeddings.sh), held on this one, so the GPU is not held through two
# hours of sklearn; hpc/submit_paper.sh chains both.
#
#   qsub -v METHOD=sage hpc/jobs/baseline.sh                  # supervised on nt_best
#   qsub -v METHOD=sage,SUPERVISE_ON=nt_train hpc/jobs/baseline.sh
#   qsub -v METHOD=bgrl,SEED=1 hpc/jobs/baseline.sh
#   qsub -v METHOD=gat,EPOCHS=5,TAG=smoke hpc/jobs/baseline.sh
#
# Writes experiments/baselines/<name>_embeddings_{fafb,mcns}.npy and
# <name>_build.json (training curve, peak GPU memory, and for supervised
# methods their own head scored zero-shot on MCNS). <name> is
# <method>[_<label>]_s<seed>[_<tag>], e.g. sage_ntbest_s0.
#
# Size: whole-graph training, so memory is nodes x width plus the edge lists.
# SAGE/GIN are a few GiB; GAT checkpoints its attention and is the one to watch
# (peak_gpu_gib in the build report). Each epoch is well under a second for
# SAGE; 300-500 epochs plus embedding MCNS is minutes, GAT perhaps an hour.
# =============================================================================
#$ -N wiretype_baseline
#$ -cwd
#$ -o logs/
#$ -e logs/
#$ -l h_rt=04:00:00
#$ -q gpu
#$ -l gpu-mig=1
# The paper's cluster offered at most 32G of h_rss on its MIG nodes.
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
# As train.sh and transfer.sh: BGRL died at 9.22 of 9.5 GiB with 237 MiB reserved
# but unallocated (job 59018081), i.e. of fragmentation. Changes allocation only.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

METHOD="${METHOD:?set METHOD: sage, gin, gat, dgi, bgrl or graphmae (degree/raw/composition run on CPU)}"
SUPERVISE_ON="${SUPERVISE_ON:-nt_best}"; SEED="${SEED:-0}"; TAG="${TAG:-}"
BACKBONE="${BACKBONE:-sage}"; EPOCHS="${EPOCHS:-}"; LR="${LR:-}"; WIDTH="${WIDTH:-512}"
SCOPE="${SCOPE:-brain_neurons}"   # brain neurons with at least one connection (src/wiretype/data/scope.py)

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "baseline ${METHOD} supervise_on=${SUPERVISE_ON} backbone=${BACKBONE} scope=${SCOPE} seed=${SEED} tag=${TAG:-<none>}: $(date)"
PYTHONPATH=src python scripts/baselines.py \
    --method "${METHOD}" --supervise-on "${SUPERVISE_ON}" --backbone "${BACKBONE}" \
    --seed "${SEED}" --width "${WIDTH}" --scope "${SCOPE}" ${TAG:+--tag "${TAG}"} \
    ${EPOCHS:+--epochs "${EPOCHS}"} ${LR:+--lr "${LR}"} \
    --processed data/processed --reports experiments/baselines
echo "finished: $(date)"
