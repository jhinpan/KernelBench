#!/usr/bin/env bash
set -euo pipefail

LEVELS="${LEVELS:-1}"
RUN_PREFIX="${RUN_PREFIX:-glm_5_2}"

for level in ${LEVELS}; do
  export LEVEL="${level}"
  export RUN_NAME="${RUN_PREFIX}_level${level}"
  echo "[MI300-GLM5.2] Generate level ${level} -> ${RUN_NAME}"
  bash "$(dirname "${BASH_SOURCE[0]}")/glm_5_2_generate.sh"
  echo "[MI300-GLM5.2] Eval level ${level} -> ${RUN_NAME}"
  bash "$(dirname "${BASH_SOURCE[0]}")/glm_5_2_eval.sh"
done
