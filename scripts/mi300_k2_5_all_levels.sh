#!/usr/bin/env bash
set -euo pipefail

LEVELS="${LEVELS:-1}"
RUN_PREFIX="${RUN_PREFIX:-mi300_k2_5}"

for level in ${LEVELS}; do
  export LEVEL="${level}"
  export RUN_NAME="${RUN_PREFIX}_level${level}"
  echo "[MI300-K2.5] Generate level ${level} -> ${RUN_NAME}"
  bash "$(dirname "${BASH_SOURCE[0]}")/mi300_k2_5_generate.sh"
  echo "[MI300-K2.5] Eval level ${level} -> ${RUN_NAME}"
  bash "$(dirname "${BASH_SOURCE[0]}")/mi300_k2_5_eval.sh"
done
