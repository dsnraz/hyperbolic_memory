#!/bin/bash
set -euo pipefail

# ============================================================================
# Fine lambda sweep at curvature 0.1 (the known best curvature).
# Submits train jobs in batches of 3 via sbatch.
#
# Usage:
#   bash scripts/train_lambda_sweep.sh [extra train args...]
# ============================================================================

ROOT_DIR="/share/home/leiyh5/Memory"
SCRIPT_DIR="${ROOT_DIR}/scripts"

CURV=0.1
LAMBDAS=("0.0" "0.1" "0.15" "0.2" "0.25" "0.3" "0.35" "0.4")
BATCH_SIZE=3

# ---------------------------------------------------------------------------
JOB_SPECS=()

for LAMBDA in "${LAMBDAS[@]}"; do
  SAFE_LAMBDA="${LAMBDA/./p}"
  TAG="c0p1_la${SAFE_LAMBDA}"
  OUT_DIR="${ROOT_DIR}/checkpoints_locomo_category_${TAG}"
  LOG_PATH="${ROOT_DIR}/job_train_${TAG}.out"
  JOB_SPECS+=("${LAMBDA}|${TAG}|${OUT_DIR}|${LOG_PATH}")
done

TOTAL_JOBS=${#JOB_SPECS[@]}
echo "============================================================"
echo "Fine lambda sweep at curvature ${CURV}"
echo "Lambdas: ${LAMBDAS[*]}"
echo "Total jobs: ${TOTAL_JOBS}"
echo "Batch size: ${BATCH_SIZE}"
echo "============================================================"
echo ""

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
    IFS='|' read -r LAMBDA TAG OUT_DIR LOG_PATH <<< "${JOB_SPECS[$j]}"

    echo "  Submitting curv=${CURV} lambda=${LAMBDA}"

    JOB_ID=$(sbatch --parsable \
      --output "${LOG_PATH}" \
      "${SCRIPT_DIR}/train.sh" \
        --initial_curvature "${CURV}" \
        --lambda_centroid "${LAMBDA}" \
        --output_dir "${OUT_DIR}" \
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
echo "All ${TOTAL_JOBS} train jobs submitted."
echo ""
echo "Checkpoints:"
for ((i = 0; i < TOTAL_JOBS; i++)); do
  IFS='|' read -r LAMBDA TAG OUT_DIR LOG_PATH <<< "${JOB_SPECS[$i]}"
  echo "  ${OUT_DIR}/hyperbolic_projector_final.pt"
done
echo "============================================================"
