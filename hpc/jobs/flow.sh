#!/bin/bash
# =============================================================================
# P1: the flow hierarchy — trophic level and SpringRank.
# Reads the intake output; writes data/processed/<volume>_flow.parquet and
# experiments/features/flow_<volume>.json.
#
#   qsub hpc/jobs/flow.sh                          FAFB, diagnostics only
#   qsub -v PROBE=1 hpc/jobs/flow.sh               ...and the probe arms
#   qsub -v VOLUME=mcns hpc/jobs/flow.sh           needed before any transfer arm
#
# **This also runs locally; it is a job so the pipeline runs in one place.**
# One Jacobi-preconditioned CG solve per configuration: 20 seconds on FAFB at a
# relative residual of 1e-10, seven solves in under two minutes.
#
# What actually takes time is the `--probe` half: four arms at 8, 10, 24 and 26
# dimensions, each across three targets and two scopes, and the 173-class
# hemilineage fit is the long pole in all four. Budget for the probes rather than the solve.
#
# Memory: the symmetrised Laplacian over 15.1M connections is about 30M
# non-zeros in float64 with int32 indices, so roughly 0.5 GB, plus the edge
# frame. MCNS is 25.6M connections and scales from there. 4 x 8G is ample.
# =============================================================================
#$ -N wiretype_flow
#$ -cwd
#$ -o logs/
#$ -e logs/
#$ -l h_rt=04:00:00
#$ -pe sharedmem 4
#$ -l h_rss=8G

set -euo pipefail
source hpc/config.sh
. /etc/profile.d/modules.sh
module load "${ANACONDA_MODULE}"
set +u; source activate "${ANALYSIS_ENV}"; set -u

# BLAS-bound in both halves: the CG matvecs and then the probe fits.
export OMP_NUM_THREADS="${NSLOTS:-4}"
export OPENBLAS_NUM_THREADS="${NSLOTS:-4}"
export MKL_NUM_THREADS="${NSLOTS:-4}"

VOLUME="${VOLUME:-fafb}"
SEED="${SEED:-0}"
PROBE="${PROBE:-}"

echo "flow_hierarchy volume=${VOLUME} probe=${PROBE:-0}: $(date)"
PYTHONPATH=src python scripts/flow.py \
    --volume "${VOLUME}" --seed "${SEED}" \
    --processed data/processed --reports experiments/features \
    ${PROBE:+--probe}
echo "finished: $(date)"
