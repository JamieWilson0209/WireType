#!/bin/bash
# =============================================================================
# Single-synapse connection shares by super-class, FAFB and the MCNS brain, for the
# Figure 1-figure supplement 1 legend. See scripts/diagnostics/connection_shape.py.
#
#   qsub hpc/jobs/connection_shape.sh
#
# Writes experiments/features/connection_shape.json; the table is also in the log.
# CPU only: two bincount passes over each volume's edge table, a few minutes.
# =============================================================================
#$ -N wiretype_shape
#$ -cwd
#$ -o logs/
#$ -e logs/
#$ -l h_rt=01:00:00
#$ -l h_rss=16G

set -euo pipefail
source hpc/config.sh
. /etc/profile.d/modules.sh
module load "${ANACONDA_MODULE}"
set +u; source activate "${ENV_PREFIX}"; set -u
echo "connection_shape: $(date)"
PYTHONPATH=src python scripts/diagnostics/connection_shape.py --processed data/processed
echo "finished: $(date)"
