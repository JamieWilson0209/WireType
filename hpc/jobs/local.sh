#!/bin/bash
# =============================================================================
# Local topology: reciprocity, core number, partner degree, directed triangles.
# See scripts/local.py.
# Writes data/processed/<volume>_local<suffix>.parquet.
#
#   qsub -v GROUPS=cheap hpc/jobs/local.sh                  # reciprocity, k-core, partner degree
#   qsub -v GROUPS=triangles,SUFFIX=_tri hpc/jobs/local.sh  # ~9 min, benchmarked
#   qsub -v VOLUME=mcns,GROUPS=cheap hpc/jobs/local.sh
#
# Every column is isomorphism-invariant and identity-free, so the block transfers
# to MCNS by construction.
#
# **Cyclic and transitive triangles are kept apart.** i->j->k->i is a feedback
# loop; i->j->k with i->k is feed-forward. They mean opposite things in a circuit
# and an undirected count destroys the distinction -- the same mistake the
# existing RWSE makes by symmetrising before it walks.
#
# Verified against networkx (core number, clustering coefficient) and against
# dense references (reciprocity, both directed triangle types).
#
# Time, benchmarked on the real graph: triangles 0.15 h at block 512, via the
# masked product (A[B] @ A) . A[B] -- forming A @ A outright is the densification
# that has killed jobs here twice. The cheap group is dominated by the k-core
# peel, which is an O(V+E) Python loop over 30M edge endpoints.
# =============================================================================
#$ -N wiretype_local
#$ -cwd
#$ -o logs/
#$ -e logs/
#$ -l h_rt=02:00:00
#$ -pe sharedmem 4
#$ -l h_rss=16G

set -euo pipefail
source hpc/config.sh
. /etc/profile.d/modules.sh
module load "${ANACONDA_MODULE}"
set +u; source activate "${ANALYSIS_ENV}"; set -u
export OMP_NUM_THREADS="${NSLOTS:-4}"

VOLUME="${VOLUME:-fafb}"; GROUPS="${GROUPS:-cheap}"; GROUPS="${GROUPS//+/,}"
BLOCK="${BLOCK:-512}"; SUFFIX="${SUFFIX:-}"
echo "local volume=${VOLUME} groups=${GROUPS} block=${BLOCK} suffix=${SUFFIX:-<none>}: $(date)"
PYTHONPATH=src python scripts/local.py \
    --volume "${VOLUME}" --groups "${GROUPS}" --block "${BLOCK}" --suffix "${SUFFIX}" \
    --processed data/processed --reports experiments/features
echo "finished: $(date)"
