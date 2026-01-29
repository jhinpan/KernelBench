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
NUM_WORKERS="${NUM_WORKERS:-8}"
SERVER_TYPE="${SERVER_TYPE:-mi300_k2_5}"
MODEL_NAME="${MODEL_NAME:-/data/Kimi-K2.5}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
TEMPERATURE="${TEMPERATURE:-0.0}"
BACKEND="${BACKEND:-triton}"
PRECISION="${PRECISION:-fp32}"
PROMPT_OPTION="${PROMPT_OPTION:-one_shot}"

ARGS=(
  "run_name=${RUN_NAME}"
  "dataset_src=${DATASET_SRC}"
  "level=${LEVEL}"
  "num_workers=${NUM_WORKERS}"
  "server_type=${SERVER_TYPE}"
  "model_name=${MODEL_NAME}"
  "max_tokens=${MAX_TOKENS}"
  "temperature=${TEMPERATURE}"
  "backend=${BACKEND}"
  "precision=${PRECISION}"
  "prompt_option=${PROMPT_OPTION}"
)

if [[ -n "${SUBSET_START:-}" || -n "${SUBSET_END:-}" ]]; then
  start="${SUBSET_START:-1}"
  end="${SUBSET_END:-${start}}"
  ARGS+=("subset=${start},${end}")
fi

if [[ "${INCLUDE_HARDWARE_INFO:-}" == "true" ]]; then
  ARGS+=("include_hardware_info=true")
  ARGS+=("hardware_gpu_name=${HARDWARE_GPU_NAME:-MI300X}")
fi

if [[ -n "${CUSTOM_PROMPT_KEY:-}" ]]; then
  ARGS+=("custom_prompt_key=${CUSTOM_PROMPT_KEY}")
fi

cd "${ROOT_DIR}"
${PYTHON_RUNNER} scripts/generate_samples.py "${ARGS[@]}"
