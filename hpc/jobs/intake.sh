#!/bin/bash
# =============================================================================
# Build-order step 1: read both raw volumes into the canonical node/edge shape.
# Writes data/processed/{fafb,mcns}_{nodes,edges}.parquet and intake_report.json.
#
# Submit from the repo root:
#   qsub hpc/jobs/intake.sh                     # both volumes
#   qsub -v VOLUME=fafb hpc/jobs/intake.sh      # one at a time, if memory is tight
#   qsub -v DATA=/path/to/raw hpc/jobs/intake.sh  # releases outside the repo
#
# Memory: the binding step is aggregating FAFB's neuropil-split edge table —
# 16.8M rows down to 15.1M directed pairs. pandas needs the frame plus a sort
# buffer plus the grouped result live at once, so budget ~4x the 3-column frame:
#     fafb  ->  ~6 GB   (3 cols x 16.8M rows, plus group-by intermediates)
#     mcns  ->  ~9 GB   (25.6M rows, already one row per pair, so no group-by)
#     both  ->  peak is whichever is larger, not the sum — they are loaded and
#               released in sequence, and `del raw` drops each table promptly
# h_rss is PER CORE on this setup, so total = sharedmem x h_rss. 4 x 8G = 32 GB
# is ample; this is a memory job rather than a parallel one, so the cores are
# there for the allocation and for pyarrow's reader, not for a speedup.
# =============================================================================
#$ -N wiretype_intake
#$ -cwd
#$ -o logs/
#$ -e logs/
#$ -l h_rt=02:00:00
#$ -pe sharedmem 4
#$ -l h_rss=8G

set -euo pipefail
source hpc/config.sh
. /etc/profile.d/modules.sh
module load "${ANACONDA_MODULE}"
set +u; source activate "${ANALYSIS_ENV}"; set -u

# Single-threaded work; keep BLAS from grabbing the whole node for no gain.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

VOLUME="${VOLUME:-both}"
DATA="${DATA:-${WIRETYPE_DATA}}"

echo "intake volume=${VOLUME} from ${DATA}: $(date)"
PYTHONPATH=src python scripts/intake.py --volume "${VOLUME}" --data "${DATA}" --out data/processed
echo "finished: $(date)"
