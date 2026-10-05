#!/bin/bash
# =============================================================================
# Random-walk structural encoding — exact diag(P^k) for k = 1..MAX_K.
# Reads the intake output; writes data/processed/<volume>_rwse.parquet and
# experiments/features/rwse_<volume>.json.
#
# Part of the critical path: the degree features alone carry 1.74 effective
# directions, so without these the encoder reads as collapsed before training starts.
#
#   qsub hpc/jobs/rwse.sh
#   qsub -v MAX_K=4 hpc/jobs/rwse.sh          # half the cost, k=1..4
#   qsub -v VOLUME=mcns hpc/jobs/rwse.sh      # needed before any transfer run
#
# Sizing. Cost scales as n x nnz -- the number of blocks times the cost of each
# sparse product -- so it is markedly different per volume, which the first
# attempt got wrong by benchmarking FAFB and applying it to both:
#   FAFB   139,262 cells, 26.2M non-zeros   ~3.6 h at K = 8
#   MCNS   211,577 cells, 43.5M non-zeros   ~9.3 h at K = 8   (2.5x FAFB)
# The MCNS run was killed at 60% under a 6-hour limit with nothing written.
# Limit is now 24 h, and the script checkpoints every 200 blocks and resumes
# from the partial file, so a kill costs minutes rather than the whole pass.
# Peak memory is 2 x n x block x 4 bytes = 0.87 GB for MCNS at block 512.
# =============================================================================
#$ -N wiretype_rwse
#$ -cwd
#$ -o logs/
#$ -e logs/
#$ -l h_rt=24:00:00
#$ -pe sharedmem 4
#$ -l h_rss=8G

set -euo pipefail
source hpc/config.sh
. /etc/profile.d/modules.sh
module load "${ANACONDA_MODULE}"
set +u; source activate "${ANALYSIS_ENV}"; set -u

export OMP_NUM_THREADS="${NSLOTS:-4}"
export OPENBLAS_NUM_THREADS="${NSLOTS:-4}"
export MKL_NUM_THREADS="${NSLOTS:-4}"

VOLUME="${VOLUME:-fafb}"
MAX_K="${MAX_K:-8}"
BLOCK="${BLOCK:-512}"

echo "rwse volume=${VOLUME} max_k=${MAX_K} block=${BLOCK}: $(date)"
PYTHONPATH=src python scripts/rwse.py \
    --volume "${VOLUME}" --max-k "${MAX_K}" --block "${BLOCK}" \
    --processed data/processed --reports experiments/features
echo "finished: $(date)"
