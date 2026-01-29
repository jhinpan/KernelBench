#!/usr/bin/env bash
set -euo pipefail

# Serial eval runner to isolate ROCm GPU hangs per-problem.
# Usage example:
#   RUN_NAME=mi300_k2_5_level1 LEVEL=1 END_ID=100 DATASET_SRC=huggingface ./scripts/mi300_k2_5_eval_serial.sh

RUN_NAME="${RUN_NAME:-mi300_k2_5_level1}"
LEVEL="${LEVEL:-1}"
DATASET_SRC="${DATASET_SRC:-huggingface}"
START_ID="${START_ID:-1}"
END_ID="${END_ID:-100}"
NUM_GPU_DEVICES="${NUM_GPU_DEVICES:-1}"
GPU_ARCH="${GPU_ARCH:-[\"gfx942\"]}"
BACKEND="${BACKEND:-triton}"
PRECISION="${PRECISION:-fp32}"
NUM_CORRECT_TRIALS="${NUM_CORRECT_TRIALS:-5}"
NUM_PERF_TRIALS="${NUM_PERF_TRIALS:-100}"
TIMEOUT="${TIMEOUT:-180}"
TIMING_METHOD="${TIMING_METHOD:-cuda_event}"

LOG_DIR="logs/mi300_k2_5"
mkdir -p "${LOG_DIR}"
EVAL_LOG="${LOG_DIR}/eval_level${LEVEL}_serial.log"

PYTHON_RUNNER="${PYTHON_RUNNER:-./.venv/bin/python}"

for pid in $(seq "${START_ID}" "${END_ID}"); do
  # Skip if already evaluated
  if "${PYTHON_RUNNER}" - <<PY
import json, os, sys
path = "runs/${RUN_NAME}/eval_results.json"
if not os.path.exists(path):
    sys.exit(1)
with open(path) as f:
    data = json.load(f)
sys.exit(0 if str(${pid}) in data else 1)
PY
  then
    echo "[skip] problem ${pid} already evaluated" | tee -a "${EVAL_LOG}"
    continue
  fi

  echo "[run] problem ${pid}" | tee -a "${EVAL_LOG}"
  set +e
  "${PYTHON_RUNNER}" scripts/eval_from_generations.py \
    run_name="${RUN_NAME}" \
    dataset_src="${DATASET_SRC}" \
    level="${LEVEL}" \
    problem_ids="[${pid}]" \
    num_gpu_devices="${NUM_GPU_DEVICES}" \
    gpu_arch="${GPU_ARCH}" \
    backend="${BACKEND}" \
    precision="${PRECISION}" \
    num_correct_trials="${NUM_CORRECT_TRIALS}" \
    num_perf_trials="${NUM_PERF_TRIALS}" \
    timeout="${TIMEOUT}" \
    timing_method="${TIMING_METHOD}" \
    >> "${EVAL_LOG}" 2>&1
  status=$?
  set -e

  if [ "${status}" -ne 0 ]; then
    echo "[warn] problem ${pid} eval crashed (exit ${status}); marking skipped" | tee -a "${EVAL_LOG}"
    "${PYTHON_RUNNER}" - <<PY
import json, os
from collections import defaultdict

path = "runs/${RUN_NAME}/eval_results.json"
os.makedirs(os.path.dirname(path), exist_ok=True)
if os.path.exists(path):
    with open(path) as f:
        data = json.load(f)
        data = defaultdict(lambda: [], data)
else:
    data = defaultdict(lambda: [])

pid = str(${pid})
if pid not in data:
    data[pid] = []
data[pid].append({
    "sample_id": 0,
    "compiled": False,
    "correctness": False,
    "metadata": {"skipped": True, "skip_reason": "eval_process_crashed"},
    "runtime": -1.0,
    "runtime_stats": {},
})

sorted_data = dict(sorted(data.items(), key=lambda x: int(x[0])))
with open(path, "w") as f:
    json.dump(sorted_data, f, indent=4)
PY
  fi
done
