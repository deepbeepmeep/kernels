#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${DIST_DIR:-$ROOT_DIR/dist}"
BUILD_DIR="$ROOT_DIR/build"
CONDA_BIN="${CONDA_BIN:-$HOME/anaconda3/bin/conda}"
VERSION_FILE="$ROOT_DIR/src/llamacpp_gguf_cuda/version.py"
MAX_JOBS="${MAX_JOBS:-$(nproc)}"

if [[ ! -x "$CONDA_BIN" ]]; then
  echo "Missing conda executable at $CONDA_BIN" >&2
  exit 1
fi

mkdir -p "$DIST_DIR"
ORIGINAL_VERSION="$(cat "$VERSION_FILE" 2>/dev/null || true)"
TEMP_DIRS=()

restore_version_file() {
  if [[ -n "$ORIGINAL_VERSION" ]]; then
    printf '%s\n' "$ORIGINAL_VERSION" > "$VERSION_FILE"
  fi
}

cleanup_temp_dirs() {
  local dir
  for dir in "${TEMP_DIRS[@]}"; do
    [[ -n "$dir" && -d "$dir" ]] || continue
    rm -rf "$dir"
  done
}

cleanup() {
  restore_version_file
  cleanup_temp_dirs
}

trap cleanup EXIT

join_by_colon() {
  local output=""
  local item
  for item in "$@"; do
    [[ -n "$item" && -d "$item" ]] || continue
    if [[ -n "$output" ]]; then
      output="${output}:"
    fi
    output="${output}${item}"
  done
  printf '%s' "$output"
}

collect_site_package_dirs() {
  local env_name="$1"
  local dir_name="$2"
  run_in_env "$env_name" python - <<PY
import site
from pathlib import Path

matches = []
for root in site.getsitepackages():
    nvidia_root = Path(root) / "nvidia"
    if not nvidia_root.is_dir():
        continue
    for candidate in sorted(nvidia_root.glob(f"*/${dir_name}")):
        if candidate.is_dir():
            matches.append(str(candidate))

for match in sorted(set(matches)):
    print(match)
PY
}

stage_linker_symlinks() {
  local shim_dir
  shim_dir="$(mktemp -d "${TMPDIR:-/tmp}/llamacpp-gguf-linker.XXXXXX")"
  TEMP_DIRS+=("$shim_dir")

  local created=0
  local lib_dir
  local entry
  local base_name
  local shim_name

  shopt -s nullglob
  for lib_dir in "$@"; do
    [[ -d "$lib_dir" ]] || continue
    for entry in "$lib_dir"/lib*.so.*; do
      [[ -f "$entry" || -L "$entry" ]] || continue
      base_name="$(basename "$entry")"
      shim_name="${base_name%%.so.*}.so"
      if [[ ! -e "$shim_dir/$shim_name" ]]; then
        ln -s "$entry" "$shim_dir/$shim_name"
        created=1
      fi
    done
  done
  shopt -u nullglob

  if [[ "$created" -eq 0 ]]; then
    local last_index
    last_index=$((${#TEMP_DIRS[@]} - 1))
    rm -rf "$shim_dir"
    unset "TEMP_DIRS[$last_index]"
    return 1
  fi

  printf '%s' "$shim_dir"
}

run_in_env() {
  local env_name="$1"
  shift
  "$CONDA_BIN" run --no-capture-output -n "$env_name" "$@"
}

verify_env() {
  local env_name="$1"
  local expected_python="$2"
  local expected_torch="$3"
  local expected_cuda="$4"

  run_in_env "$env_name" python - <<PY
import sys
import torch

python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
torch_version = torch.__version__
cuda_version = torch.version.cuda

if python_version != "${expected_python}":
    raise SystemExit("Expected Python ${expected_python} in ${env_name}, found " + python_version)
if not torch_version.startswith("${expected_torch}"):
    raise SystemExit("Expected torch ${expected_torch} in ${env_name}, found " + torch_version)
if cuda_version != "${expected_cuda}":
    raise SystemExit("Expected torch CUDA ${expected_cuda} in ${env_name}, found " + str(cuda_version))

print("[ok] ${env_name}: python=" + python_version + " torch=" + torch_version + " cuda=" + str(cuda_version))
PY
}

build_target() {
  local label="$1"
  local env_name="$2"
  local version_suffix="$3"
  local expected_python="$4"
  local expected_torch="$5"
  local expected_cuda="$6"
  local cuda_home_template="$7"

  verify_env "$env_name" "$expected_python" "$expected_torch" "$expected_cuda"

  local env_prefix
  env_prefix="$(run_in_env "$env_name" python -c 'import os; print(os.environ["CONDA_PREFIX"])')"
  local cuda_home="${cuda_home_template//\{env_prefix\}/$env_prefix}"
  if [[ ! -x "$cuda_home/bin/nvcc" ]]; then
    echo "Missing nvcc under $cuda_home for target $label" >&2
    exit 1
  fi

  local include_dirs=()
  local library_dirs=()
  local cicc_path=""
  if [[ -d "$cuda_home/include" ]]; then
    include_dirs+=("$cuda_home/include")
  fi
  if [[ -d "$cuda_home/include/cccl" ]]; then
    include_dirs+=("$cuda_home/include/cccl")
  fi
  while IFS= read -r include_dir; do
    include_dirs+=("$include_dir")
  done < <(collect_site_package_dirs "$env_name" include)

  if [[ -d "$cuda_home/lib64" ]]; then
    library_dirs+=("$cuda_home/lib64")
  fi
  if [[ -d "$cuda_home/lib" ]]; then
    library_dirs+=("$cuda_home/lib")
  fi
  while IFS= read -r library_dir; do
    library_dirs+=("$library_dir")
  done < <(collect_site_package_dirs "$env_name" lib)

  local linker_shim_dir=""
  if linker_shim_dir="$(stage_linker_symlinks "${library_dirs[@]}")"; then
    library_dirs=("$linker_shim_dir" "${library_dirs[@]}")
  fi

  if [[ -x "$env_prefix/nvvm/bin/cicc" ]]; then
    cicc_path="$env_prefix/nvvm/bin"
  elif [[ -x "$cuda_home/nvvm/bin/cicc" ]]; then
    cicc_path="$cuda_home/nvvm/bin"
  fi

  local include_path
  include_path="$(join_by_colon "${include_dirs[@]}")"
  local library_path
  library_path="$(join_by_colon "${library_dirs[@]}")"

  if [[ "$BUILD_DIR" == "$ROOT_DIR/build" && -d "$BUILD_DIR" ]]; then
    rm -rf "$BUILD_DIR"
  fi

  echo "[build] $label"
  echo "        env=$env_name"
  echo "        cuda_home=$cuda_home"
  echo "        version_suffix=$version_suffix"

  run_in_env "$env_name" env \
    CUDA_HOME="$cuda_home" \
    PATH="$cuda_home/bin:$env_prefix/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    C_INCLUDE_PATH="$include_path" \
    CPLUS_INCLUDE_PATH="$include_path" \
    LIBRARY_PATH="$library_path" \
    LD_LIBRARY_PATH="$library_path" \
    LLAMACPP_GGUF_CUDA_INCLUDE_DIRS="$include_path" \
    LLAMACPP_GGUF_CUDA_LIB_DIRS="$library_path" \
    LLAMACPP_GGUF_CUDA_VERSION_SUFFIX="$version_suffix" \
    LLAMACPP_GGUF_CUDA_NVCC_THREADS="${LLAMACPP_GGUF_CUDA_NVCC_THREADS:-0}" \
    CICC_PATH="$cicc_path" \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-}" \
    MAX_JOBS="$MAX_JOBS" \
    python -m pip wheel "$ROOT_DIR" --no-build-isolation -w "$DIST_DIR"
}

TARGETS=("$@")
if [[ ${#TARGETS[@]} -eq 0 ]]; then
  TARGETS=(all)
fi

for target in "${TARGETS[@]}"; do
  case "$target" in
    all)
      build_target \
        "py310-cu128" \
        "base" \
        "+torch271cu128py310" \
        "3.10" \
        "2.7.1" \
        "12.8" \
        "/usr/local/cuda-12.8"
      build_target \
        "py311-cu130" \
        "py311" \
        "+torch210cu13py311" \
        "3.11" \
        "2.10.0" \
        "13.0" \
        "{env_prefix}/targets/x86_64-linux"
      ;;
    py310)
      build_target \
        "py310-cu128" \
        "base" \
        "+torch271cu128py310" \
        "3.10" \
        "2.7.1" \
        "12.8" \
        "/usr/local/cuda-12.8"
      ;;
    py311)
      build_target \
        "py311-cu130" \
        "py311" \
        "+torch210cu13py311" \
        "3.11" \
        "2.10.0" \
        "13.0" \
        "{env_prefix}/targets/x86_64-linux"
      ;;
    *)
      echo "Unknown target: $target" >&2
      echo "Usage: $0 [all|py310|py311]" >&2
      exit 1
      ;;
  esac
done
