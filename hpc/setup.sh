#!/bin/bash
# =============================================================================
# One-time environment setup on an SGE / Grid Engine cluster. Run from anywhere
# on a login node — installs are fine there; never run an analysis there.
#
#     bash hpc/setup.sh              # analysis env (all current jobs need this)
#     bash hpc/setup.sh train        # CUDA torch env, for training later
#     bash hpc/setup.sh all          # both
#     bash hpc/setup.sh check        # verify an existing env + a tiny smoke run
#
# Targets:
#   analysis  ANALYSIS_ENV : python 3.11 + numpy/scipy/pandas/pyarrow/sklearn.
#                            The feature jobs (intake, splits, rwse, flow,
#                            local, topk) activate it and import wiretype via
#                            PYTHONPATH=src. The package is deliberately NOT
#                            installed here: that would drag in torch and
#                            torch-geometric for jobs that are pure BLAS.
#   train     ENV_PREFIX   : python 3.11 + CUDA torch + `pip install -e .`,
#                            which brings torch-geometric and the rest.
#
# Env locations and module names come from hpc/config.sh. Each env is created
# only if its directory is absent, so re-running is safe.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
source "${HERE}/config.sh"

TARGET="${1:-analysis}"
case "${TARGET}" in
    analysis|train|all|check) ;;
    *) echo "usage: bash hpc/setup.sh [analysis|train|all|check]   (default: analysis)" >&2
       exit 2 ;;
esac

. /etc/profile.d/modules.sh
module load "${ANACONDA_MODULE}"

# Some conda activation hooks reference unbound variables; `set -u` would abort
# on them. Activate with nounset off, exactly as the job scripts do.
activate() {
    set +u
    source activate "$1"
    set -u
}

setup_analysis() {
    echo "=== analysis env: ${ANALYSIS_ENV} ==="
    mkdir -p "$(dirname "${ANALYSIS_ENV}")"
    if [ ! -d "${ANALYSIS_ENV}" ]; then
        conda create --yes --prefix "${ANALYSIS_ENV}" python=3.11
    fi
    activate "${ANALYSIS_ENV}"
    python -m pip install --upgrade pip
    # Pinned to the same floors as pyproject.toml so the two envs agree on
    # behaviour. scikit-learn is what supplies randomized_svd and normalize;
    # pyarrow reads the feather dumps; the rest is the usual numeric stack.
    python -m pip install \
        "numpy>=1.24" "scipy>=1.10" "pandas>=2.0" "pyarrow>=14.0" "scikit-learn>=1.3"
    python -c "import numpy, scipy, pandas, pyarrow, sklearn; \
print('  analysis env ready | numpy', numpy.__version__, '| scipy', scipy.__version__, \
'| sklearn', sklearn.__version__)"
}

setup_train() {
    echo "=== training env: ${ENV_PREFIX} ==="
    mkdir -p "$(dirname "${ENV_PREFIX}")"
    if [ ! -d "${ENV_PREFIX}" ]; then
        conda create --yes --prefix "${ENV_PREFIX}" python=3.11
    fi
    activate "${ENV_PREFIX}"
    python -m pip install --upgrade pip
    # Install the CUDA torch wheel first so the editable install below finds it
    # already satisfied and does not pull a CPU build over the top.
    python -m pip install torch --index-url "https://download.pytorch.org/whl/${CUDA_BUILD}"
    python -m pip install -e "${REPO_ROOT}"
    # torch wheels are built against a specific numpy ABI and pip has no way to
    # express that, so a fresh `pip install numpy` can silently produce an env
    # where `torch.from_numpy` raises "expected np.ndarray (got numpy.ndarray)".
    # Every training script converts arrays, so this must be caught at setup
    # rather than an hour into a GPU job. Checked rather than pinned, so it stays
    # true as versions move.
    if ! python -c "import numpy, torch; torch.from_numpy(numpy.zeros(1, dtype=numpy.float32))" 2>/dev/null; then
        echo "  numpy/torch ABI mismatch — pinning numpy<2 and retrying"
        python -m pip install --quiet "numpy<2"
        python -c "import numpy, torch; torch.from_numpy(numpy.zeros(1, dtype=numpy.float32))" \
            || { echo "  ERROR: torch and numpy still cannot interoperate"; exit 1; }
    fi
    python -c "import numpy, torch, wiretype; \
print('  training env ready | torch', torch.__version__, '| numpy', numpy.__version__, \
'| cuda', torch.cuda.is_available())"
}

run_check() {
    echo "=== checking analysis env: ${ANALYSIS_ENV} ==="
    [ -d "${ANALYSIS_ENV}" ] || { echo "  not built yet — run: bash hpc/setup.sh" >&2; exit 1; }
    activate "${ANALYSIS_ENV}"
    python -c "import numpy, scipy, pandas, pyarrow, sklearn; print('  imports OK')"

    local edges="${WIRETYPE_DATA}/fafb_v783/proofread_connections_783.feather"
    if [ ! -f "${edges}" ]; then
        echo "  edge table NOT found at ${edges}"
        echo "  data/raw is gitignored — copy the ~1.6 GB of dumps across, or set WIRETYPE_DATA."
        exit 1
    fi
    echo "  edge table found: ${edges}"
    # The torch scripts need the train env, so the analysis env checks the rest.
    echo "=== smoke: every analysis script parses its arguments ==="
    ( cd "${REPO_ROOT}" && for s in $(grep -L "^import torch" scripts/*.py); do
        PYTHONPATH=src python "$s" --help >/dev/null || { echo "  FAIL $s" >&2; exit 1; }
      done )
    echo "  scripts OK"
}

case "${TARGET}" in
    analysis) setup_analysis ;;
    train)    setup_train ;;
    all)      setup_analysis; setup_train ;;
    check)    run_check ;;
esac

echo "Done (${TARGET})."
