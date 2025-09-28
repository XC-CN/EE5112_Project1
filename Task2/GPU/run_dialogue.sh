#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "Working directory: ${SCRIPT_DIR}"

CONDA_ENV="${CONDA_ENV:-5112Project}"
SKIP_CONDA="${SKIP_CONDA:-0}"

if [[ "${SKIP_CONDA}" != "1" ]]; then
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    echo "Activating conda environment: ${CONDA_ENV}"
    conda activate "${CONDA_ENV}"
  else
    echo "conda command not found. Set SKIP_CONDA=1 to bypass activation." >&2
    exit 1
  fi
else
  echo "Skipping conda activation (SKIP_CONDA=${SKIP_CONDA})."
fi

export LLAMA_CUBLAS="${LLAMA_CUBLAS:-1}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
fi

echo "LLAMA_CUBLAS=${LLAMA_CUBLAS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

echo "============================================"
echo "Launching GPU dialogue system..."
echo "============================================"

python dialogue_system.py "$@"
