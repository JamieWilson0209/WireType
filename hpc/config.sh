# =============================================================================
# HPC configuration (SGE / Grid Engine). A template: edit it for your cluster,
# then `source hpc/config.sh` before submitting. Every value can be overridden
# from the environment. Jobs run from the repo root (`#$ -cwd`), so paths inside
# the repo are relative.
# =============================================================================

# Root for the conda environments, kept outside the repo: they are large and
# rebuildable. Point it at scratch or group storage.
export WIRETYPE_SCRATCH="${WIRETYPE_SCRATCH:-$HOME}"

# Two environments, created by `bash hpc/setup.sh`:
#   - ANALYSIS_ENV : numpy/scipy/pandas/pyarrow/scikit-learn only, for the
#                    feature jobs (intake, splits, rwse, flow, local, topk).
#   - ENV_PREFIX   : CUDA torch + torch-geometric + the package itself, for
#                    train, transfer, baselines and score_embeddings.
export ANALYSIS_ENV="${ANALYSIS_ENV:-${WIRETYPE_SCRATCH}/conda/envs/wiretype-analysis}"
export ENV_PREFIX="${ENV_PREFIX:-${WIRETYPE_SCRATCH}/conda/envs/wiretype}"

# Module names: check them against `module avail` on your cluster.
export ANACONDA_MODULE="${ANACONDA_MODULE:-anaconda}"
export CUDA_MODULE="${CUDA_MODULE:-cuda}"

# Where the connectome releases live (README, Data): about 1.6 GB, so on a
# cluster this usually points at scratch or group storage rather than the repo.
export WIRETYPE_DATA="${WIRETYPE_DATA:-data/raw}"

# Resource requests are SGE directives, which cannot read shell variables, so
# they sit in each job script's header. The values there are those of the
# cluster the paper used: run time (h_rt), memory (h_rss), CPU slots
# (`-pe sharedmem`), and for the GPU jobs (train, transfer, baseline) a GPU queue
# with one MIG partition of an A100, about 9.5 GiB, per job:
#
#     #$ -q gpu
#     #$ -l gpu-mig=1
#
# Replace them with your cluster's resource names (`qconf -sc` lists them);
# `qstat -j <jobid>` says why a pending job is not scheduled.

# PyTorch CUDA build for the training environment.
export CUDA_BUILD="${CUDA_BUILD:-cu121}"

# Keep ~/.local user-site packages out of the environments.
export PYTHONNOUSERSITE=1
