#!/bin/bash
# =============================================================================
# Record the cluster environments as lock files in envs/. Read-only: it lists
# packages and changes nothing in the environments.
#
#   bash hpc/export_envs.sh            # on a login node, from the repo root
#
# Writes, for each environment that exists (train = ENV_PREFIX, analysis =
# ANALYSIS_ENV, both set in hpc/config.sh):
#   envs/hpc-<name>.txt         pip freeze, the Python packages and versions
#   envs/hpc-<name>-conda.txt   conda list --explicit, the conda layer (python
#                               and its system libraries) with exact build URLs;
#                               CUDA arrives with the pip torch wheel instead
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/config.sh
. /etc/profile.d/modules.sh
module load "${ANACONDA_MODULE}"
mkdir -p envs

for pair in "train:${ENV_PREFIX}" "analysis:${ANALYSIS_ENV}"; do
    name="${pair%%:*}"; prefix="${pair#*:}"
    if [ ! -d "${prefix}" ]; then
        echo "skip ${name}: ${prefix} not found"
        continue
    fi
    {
        echo "# HPC ${name} environment, frozen $(date +%Y-%m-%d) with: pip freeze"
        echo "# $("${prefix}/bin/python" -c 'import platform; print("Python", platform.python_version(), platform.platform())')"
        "${prefix}/bin/python" -m pip freeze --exclude-editable
    } > "envs/hpc-${name}.txt"
    conda list --prefix "${prefix}" --explicit > "envs/hpc-${name}-conda.txt"
    echo "wrote envs/hpc-${name}.txt and envs/hpc-${name}-conda.txt"
done
