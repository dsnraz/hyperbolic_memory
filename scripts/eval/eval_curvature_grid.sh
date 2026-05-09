#!/bin/bash
set -euo pipefail

# Submit evaluation jobs for three curvature inference outputs.
# Usage:
#   bash scripts/eval/eval_curvature_grid.sh [extra evaluate args...]

ROOT_DIR="/share/home/leiyh5/Memory"
SCRIPT_DIR="${ROOT_DIR}/scripts/eval"
ANN_FILE="${ROOT_DIR}/data/locomo/locomo10.json"
LOCOMO_ROOT="/share/home/leiyh5/locomo"

CURVATURES=("0.05" "0.5" "5")

for CURV in "${CURVATURES[@]}"; do
  SAFE_CURV="${CURV/./p}"
  PRED_FILE="${ROOT_DIR}/data/locomo/locomo10_pred_c${SAFE_CURV}.json"
  SCORED_FILE="${ROOT_DIR}/data/locomo/locomo10_pred_c${SAFE_CURV}_scored.json"
  STATS_FILE="${ROOT_DIR}/data/locomo/locomo10_pred_c${SAFE_CURV}_stats.json"
  LOG_PATH="${ROOT_DIR}/job_eval_c${SAFE_CURV}.out"

  if [[ ! -f "${PRED_FILE}" ]]; then
    echo "[skip] prediction file not found: ${PRED_FILE}"
    continue
  fi

  echo "=================================================="
  echo "Submitting eval job for curvature=${CURV}"
  echo "pred:   ${PRED_FILE}"
  echo "scored: ${SCORED_FILE}"
  echo "stats:  ${STATS_FILE}"
  echo "log:    ${LOG_PATH}"
  echo "=================================================="

  sbatch --output "${LOG_PATH}" "${SCRIPT_DIR}/eval.sh" \
    --ann-file "${ANN_FILE}" \
    --pred-file "${PRED_FILE}" \
    --locomo-root "${LOCOMO_ROOT}" \
    --scored-file "${SCORED_FILE}" \
    --stats-file "${STATS_FILE}" \
    "$@"
done

echo "All available eval jobs submitted."
