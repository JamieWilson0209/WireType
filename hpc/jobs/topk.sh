#!/bin/bash
# =============================================================================
# Each cell's k strongest partners, the encoder's tokens. See scripts/topk.py.
# Writes data/processed/<volume>_topk<k>.npz.
#
#   qsub hpc/jobs/topk.sh
#   qsub -v VOLUME=mcns hpc/jobs/topk.sh
#   qsub -v K=256 hpc/jobs/topk.sh
#
# Run this before train.sh and transfer.sh. The .npz lives beside the parquets,
# not in git; both scripts exit with the build command if it is missing.
#
# k is a denoising choice, not a budget: k=64 keeps 26.1%
# of connections and truncates 80.7% of cells; k=256 keeps 70.3%. What it
# discards is the weak tail the noise premise is about -- single-synapse edges
# are 42% consistent between brains, edges above 10 synapses exceed 90%.
#
# Time: one lexsort over 30M symmetrised entries. Minutes.
# =============================================================================
#$ -N wiretype_topk
#$ -cwd
#$ -o logs/
#$ -e logs/
#$ -l h_rt=01:00:00
#$ -pe sharedmem 4
#$ -l h_rss=8G

set -euo pipefail
source hpc/config.sh
. /etc/profile.d/modules.sh
module load "${ANACONDA_MODULE}"
set +u; source activate "${ANALYSIS_ENV}"; set -u
export OMP_NUM_THREADS="${NSLOTS:-4}"

VOLUME="${VOLUME:-fafb}"
K="${K:-64}"
echo "topk volume=${VOLUME} k=${K}: $(date)"
PYTHONPATH=src python scripts/topk.py --volume "${VOLUME}" -k "${K}" --processed data/processed
echo "finished: $(date)"
