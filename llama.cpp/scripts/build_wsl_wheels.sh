#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-$HOME/anaconda3}"
WORK_ROOT="${WORK_ROOT:-$HOME/gguf-build-1.0.21}"
DIST_DIR="${DIST_DIR:-$WORK_ROOT/dist}"

build_target() {
  local target="$1" python_bin cuda_home
  case "$target" in
    py310)
      python_bin="${PY310_PYTHON:-$CONDA_ROOT/bin/python}"
      cuda_home="${PY310_CUDA_HOME:-/usr/local/cuda-12.8}"
      ;;
    py311)
      python_bin="${PY311_PYTHON:-$CONDA_ROOT/envs/py311/bin/python}"
      cuda_home="${PY311_CUDA_HOME:-$CONDA_ROOT/envs/py311/targets/x86_64-linux}"
      ;;
    *) echo "Usage: $0 [all|py310|py311]" >&2; return 1 ;;
  esac
  "$python_bin" "$ROOT_DIR/scripts/build_release.py" \
    --target "$target" --cuda-home "$cuda_home" \
    --work-dir "$WORK_ROOT/linux_$target" --dist-dir "$DIST_DIR" \
    --max-jobs "${MAX_JOBS:-1}" --nvcc-threads "${LLAMACPP_GGUF_CUDA_NVCC_THREADS:-2}"
}

for target in "${@:-all}"; do
  if [[ "$target" == all ]]; then
    build_target py310
    build_target py311
  else
    build_target "$target"
  fi
done
