#!/bin/bash
# =============================================================================
# Build-order step 2: type-blocked train/val/test assignment plus the leak
# measurement against a naive per-neuron split. Reads the node tables written by
# hpc/jobs/intake.sh; writes data/processed/{volume}_splits.parquet and
# splits_report.json.
#
# Submit from the repo root:
#   qsub hpc/jobs/splits.sh
#   qsub -v VOLUME=both,SEED=1 hpc/jobs/splits.sh
#
# Memory: trivial by the standards of this project. It touches the node table
# only — 139k rows for FAFB, 165k for MCNS — and never the edge list, so a
# single core with 8 GB is already generous. It is a job rather than a local
# script because its input lives wherever intake wrote it, not because it is
# heavy; running it locally against a local data/processed is fine.
# =============================================================================
#$ -N wiretype_splits
#$ -cwd
#$ -o logs/
#$ -e logs/
#$ -l h_rt=00:20:00
#$ -pe sharedmem 1
#$ -l h_rss=8G

set -euo pipefail
source hpc/config.sh
. /etc/profile.d/modules.sh
module load "${ANACONDA_MODULE}"
set +u; source activate "${ANALYSIS_ENV}"; set -u

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

VOLUME="${VOLUME:-fafb}"
SEED="${SEED:-0}"

echo "splits volume=${VOLUME} seed=${SEED}: $(date)"
PYTHONPATH=src python scripts/splits.py --volume "${VOLUME}" --seed "${SEED}" --processed data/processed
echo "finished: $(date)"
