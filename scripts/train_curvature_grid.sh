#!/bin/bash
set -euo pipefail

# Grid search for hyperbolic projector initial curvature.
# Runs train.sh three times with isolated output directories:
#   0.01, 1, 10
#
# Usage:
#   bash scripts/train_curvature_grid.sh [extra train args...]
# Example:
#   bash scripts/train_curvature_grid.sh --mixed_total_iterations 50000

ROOT_DIR="/share/home/leiyh5/Memory"
SCRIPT_DIR="${ROOT_DIR}/scripts"

CURVATURES=("0.2" "0.5" "0.8")

for CURV in "${CURVATURES[@]}"; do
  SAFE_CURV="${CURV/./p}"
  OUT_DIR="${ROOT_DIR}/checkpoints_locomo_category_c${SAFE_CURV}"
  LOG_PATH="${ROOT_DIR}/job_train_c${SAFE_CURV}.out"

  echo "=================================================="
  echo "Submitting job with --initial_curvature=${CURV}"
  echo "Output dir: ${OUT_DIR}"
  echo "Log path:  ${LOG_PATH}"
  echo "=================================================="

  sbatch --output "${LOG_PATH}" "${SCRIPT_DIR}/train.sh" \
    --initial_curvature "${CURV}" \
    --output_dir "${OUT_DIR}" \
    "$@"
done

echo "All jobs submitted. Final checkpoints will be at:"
for CURV in "${CURVATURES[@]}"; do
  SAFE_CURV="${CURV/./p}"
  echo "  ${ROOT_DIR}/checkpoints_locomo_category_c${SAFE_CURV}/hyperbolic_projector_final.pt"
done
