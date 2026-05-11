#!/bin/bash
set -euo pipefail

# ============================================================================
# Inference grid for curvature × lambda_centroid checkpoints.
# Submits inference jobs in batches of 3 via sbatch.
#
# Run this AFTER train_lambda_curvature_grid.sh completes.
#
# Usage:
#   bash scripts/main_lambda_curvature_grid.sh [extra session_run args...]
# ============================================================================

ROOT_DIR="/share/home/leiyh5/Memory"
SCRIPT_DIR="${ROOT_DIR}/scripts"
DATA_FILE="${ROOT_DIR}/data/locomo/locomo10.json"
PERSIST_DIR_SHARED="${ROOT_DIR}/data/memory_running_category_morefact"

# ---------------------------------------------------------------------------
# Grid definition — MUST match the train grid
# ---------------------------------------------------------------------------

CURVATURES=("0.01" "0.1" "1.0" "10.0")
LAMBDAS=("0.1" "0.3" "0.5" "1.0")

# Fine grid around 0.1 — uncomment to match fine train grid
# CURVATURES=("0.05" "0.1" "0.15" "0.2")
# LAMBDAS=("0.1" "0.3" "0.5" "1.0")

BATCH_SIZE=3

# ---------------------------------------------------------------------------
# Build job list (only for checkpoints that exist)
# ---------------------------------------------------------------------------
JOB_SPECS=()

for CURV in "${CURVATURES[@]}"; do
  for LAMBDA in "${LAMBDAS[@]}"; do
    SAFE_CURV="${CURV/./p}"
    SAFE_LAMBDA="${LAMBDA/./p}"
    TAG="c${SAFE_CURV}_la${SAFE_LAMBDA}"
    CKPT_PATH="${ROOT_DIR}/checkpoints_locomo_category_${TAG}/hyperbolic_projector_final.pt"
    OUT_FILE="${ROOT_DIR}/data/locomo/locomo10_pred_${TAG}.json"
    LOG_PATH="${ROOT_DIR}/job_main_${TAG}.out"

    if [[ ! -f "${CKPT_PATH}" ]]; then
      echo "[skip] checkpoint not found: ${CKPT_PATH}"
      continue
    fi

    JOB_SPECS+=("${CURV}|${LAMBDA}|${TAG}|${CKPT_PATH}|${OUT_FILE}|${LOG_PATH}")
  done
done

TOTAL_JOBS=${#JOB_SPECS[@]}
if ((TOTAL_JOBS == 0)); then
  echo "No checkpoints found. Run train_lambda_curvature_grid.sh first."
  exit 1
fi

echo "============================================================"
echo "Inference grid: ${TOTAL_JOBS} checkpoints found"
echo "Batch size: ${BATCH_SIZE}"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Submit in batches
# ---------------------------------------------------------------------------
BATCH_NUM=0
for ((i = 0; i < TOTAL_JOBS; i += BATCH_SIZE)); do
  BATCH_NUM=$((BATCH_NUM + 1))
  BATCH_END=$((i + BATCH_SIZE))
  if ((BATCH_END > TOTAL_JOBS)); then
    BATCH_END=${TOTAL_JOBS}
  fi

  echo "--- Batch ${BATCH_NUM} (jobs ${i}–$((BATCH_END - 1))) ---"

  BATCH_JOB_IDS=()

  for ((j = i; j < BATCH_END; j++)); do
    IFS='|' read -r CURV LAMBDA TAG CKPT_PATH OUT_FILE LOG_PATH <<< "${JOB_SPECS[$j]}"

    echo "  Submitting inference for ${TAG}"

    JOB_ID=$(sbatch --parsable \
      --output "${LOG_PATH}" \
      "${SCRIPT_DIR}/main.sh" \
        --data-file "${DATA_FILE}" \
        --persist-directory "${PERSIST_DIR_SHARED}" \
        --projector-checkpoint-path "${CKPT_PATH}" \
        --out-file "${OUT_FILE}" \
        --retriever-type hyperbolic_angular \
        "$@")

    echo "    → job_id=${JOB_ID}"
    BATCH_JOB_IDS+=("${JOB_ID}")
  done

  if ((BATCH_END < TOTAL_JOBS)); then
    echo "  Waiting for batch ${BATCH_NUM} to finish..."
    DEP_STR=$(IFS=:; echo "${BATCH_JOB_IDS[*]}")
    WAIT_JOB=$(sbatch --parsable \
      --dependency "afterok:${DEP_STR}" \
      --wrap "echo 'batch ${BATCH_NUM} done'" \
      --output /dev/null \
      --time 00:01:00)
    while squeue -j "${WAIT_JOB}" 2>/dev/null | grep -q "${WAIT_JOB}"; do
      sleep 10
    done
    echo "  Batch ${BATCH_NUM} complete."
  fi

  echo ""
done

echo "============================================================"
echo "All ${TOTAL_JOBS} inference jobs submitted."
echo ""
echo "Prediction files at:"
for ((i = 0; i < TOTAL_JOBS; i++)); do
  IFS='|' read -r CURV LAMBDA TAG CKPT_PATH OUT_FILE LOG_PATH <<< "${JOB_SPECS[$i]}"
  echo "  ${OUT_FILE}"
done
echo "============================================================"
