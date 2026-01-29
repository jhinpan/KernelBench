#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_RUNNER="python"
if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON_RUNNER="${ROOT_DIR}/.venv/bin/python"
elif command -v uv >/dev/null 2>&1; then
  PYTHON_RUNNER="uv run python"
fi

${PYTHON_RUNNER} - <<'PY' || { echo "Missing deps: install pydra-config (or use uv sync)"; exit 1; }
import pydra  # noqa: F401
PY

LEVEL="${LEVEL:-1}"
RUN_NAME="${RUN_NAME:-mi300_k2_5_level${LEVEL}}"
DATASET_SRC="${DATASET_SRC:-huggingface}"
NUM_GPU_DEVICES="${NUM_GPU_DEVICES:-1}"
GPU_ARCH="${GPU_ARCH:-[\"gfx942\"]}"
BACKEND="${BACKEND:-triton}"
PRECISION="${PRECISION:-fp32}"
NUM_CORRECT_TRIALS="${NUM_CORRECT_TRIALS:-5}"
NUM_PERF_TRIALS="${NUM_PERF_TRIALS:-100}"
TIMEOUT="${TIMEOUT:-180}"
TIMING_METHOD="${TIMING_METHOD:-cuda_event}"

ARGS=(
  "run_name=${RUN_NAME}"
  "dataset_src=${DATASET_SRC}"
  "level=${LEVEL}"
  "num_gpu_devices=${NUM_GPU_DEVICES}"
  "gpu_arch=${GPU_ARCH}"
  "backend=${BACKEND}"
  "precision=${PRECISION}"
  "num_correct_trials=${NUM_CORRECT_TRIALS}"
  "num_perf_trials=${NUM_PERF_TRIALS}"
  "timeout=${TIMEOUT}"
  "timing_method=${TIMING_METHOD}"
)

if [[ -n "${SUBSET_START:-}" || -n "${SUBSET_END:-}" ]]; then
  start="${SUBSET_START:-1}"
  end="${SUBSET_END:-${start}}"
  ARGS+=("subset=${start},${end}")
fi

if [[ "${BUILD_CACHE:-}" == "true" ]]; then
  ARGS+=("build_cache=true")
  ARGS+=("num_cpu_workers=${NUM_CPU_WORKERS:-20}")
fi

if [[ -n "${PROBLEM_IDS:-}" ]]; then
  ARGS+=("problem_ids=${PROBLEM_IDS}")
fi

cd "${ROOT_DIR}"
${PYTHON_RUNNER} scripts/eval_from_generations.py "${ARGS[@]}"
