#!/bin/bash
set -euo pipefail

# Submit inference jobs for three curvature-trained projector checkpoints.
# Each job gets isolated checkpoint/persist/output/log paths to avoid overwrite.
#
# Usage:
#   bash scripts/main_curvature_grid.sh [extra session_run args...]

ROOT_DIR="/share/home/leiyh5/Memory"
SCRIPT_DIR="${ROOT_DIR}/scripts"
DATA_FILE="${ROOT_DIR}/data/locomo/locomo10.json"
PERSIST_DIR_SHARED="${ROOT_DIR}/data/memory_running_category"

CURVATURES=("0.2" "0.5" "0.8")

for CURV in "${CURVATURES[@]}"; do
  SAFE_CURV="${CURV/./p}"
  CKPT_PATH="${ROOT_DIR}/checkpoints_locomo_category_c${SAFE_CURV}/hyperbolic_projector_final.pt"
  OUT_FILE="${ROOT_DIR}/data/locomo/locomo10_pred_c${SAFE_CURV}.json"
  LOG_PATH="${ROOT_DIR}/job_main_c${SAFE_CURV}.out"

  if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "[skip] checkpoint not found: ${CKPT_PATH}"
    continue
  fi

  echo "=================================================="
  echo "Submitting inference job for curvature=${CURV}"
  echo "checkpoint: ${CKPT_PATH}"
  echo "persist:    ${PERSIST_DIR_SHARED}"
  echo "output:     ${OUT_FILE}"
  echo "log:        ${LOG_PATH}"
  echo "=================================================="

  sbatch --output "${LOG_PATH}" "${SCRIPT_DIR}/main.sh" \
    --data-file "${DATA_FILE}" \
    --persist-directory "${PERSIST_DIR_SHARED}" \
    --projector-checkpoint-path "${CKPT_PATH}" \
    --out-file "${OUT_FILE}" \
    "$@"
done

echo "All available inference jobs submitted."
