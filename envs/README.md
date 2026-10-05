# Environments

The exact package versions the paper's results were produced with. `pyproject.toml`
states the minimum versions the code needs; these files record what was installed,
so an environment can be rebuilt to match.

| file | environment | used for |
|---|---|---|
| `hpc-train.txt`, `hpc-train-conda.txt` | `ENV_PREFIX` (CUDA torch) | training, transfer, baselines, scoring |
| `hpc-analysis.txt`, `hpc-analysis-conda.txt` | `ANALYSIS_ENV` | intake, splits and feature construction (random-walk return, flow, local topology, top-k) |
| `local.txt` | a local `.venv` (Python 3.11, macOS, no torch) | diagnostics, label audit, release tables |

The HPC files were written by `bash hpc/export_envs.sh` on a login node;
`hpc/config.sh` names the environments. The conda files hold only the base layer
(Python 3.11.16 and its system libraries), the same for both environments; CUDA
comes in through pip, as the `torch==2.5.1+cu121` wheel and its `nvidia-*-cu12`
libraries in `hpc-train.txt`. The local file was written with
`uv pip freeze --python .venv/bin/python`.

## Rebuilding

- **Local:** `uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r envs/local.txt`.
- **HPC:** `conda create --prefix <env> --file envs/hpc-<name>-conda.txt`, then
  `<env>/bin/python -m pip install -r envs/hpc-<name>.txt` (for `train`, add
  `--extra-index-url https://download.pytorch.org/whl/cu121`, where the CUDA torch
  wheel lives, as `hpc/setup.sh` does) and
  `<env>/bin/python -m pip install -e . --no-deps` from the repo root.

The package itself (`wiretype`) is installed editable in the training environment and
is not part of the freezes; local scripts run it from `src/` with `PYTHONPATH=src`.
