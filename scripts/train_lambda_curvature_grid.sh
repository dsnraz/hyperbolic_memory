#!/bin/bash
set -euo pipefail

# ============================================================================
# Grid search: curvature × lambda_centroid
# Submits train jobs in batches of 3 via sbatch.
#
# Output dir naming: checkpoints_locomo_category_c{CURV}_la{LAMBDA}
#   where dot → p (e.g. c0p1_la0p3)
#
# Usage:
#   bash scripts/train_lambda_curvature_grid.sh [extra train args...]
# Example:
#   bash scripts/train_lambda_curvature_grid.sh --embedding_dim 768 --hidden_dim 2048
# ============================================================================

ROOT_DIR="/share/home/leiyh5/Memory"
SCRIPT_DIR="${ROOT_DIR}/scripts"

# ---------------------------------------------------------------------------
# Grid definition — adjust these arrays to control the search space
# ---------------------------------------------------------------------------

# Coarse grid (16 combos)
CURVATURES=("0.01" "0.1" "1.0" "10.0")
LAMBDAS=("0.1" "0.3" "0.5" "1.0")

# Fine grid around 0.1 — uncomment to replace coarse grid
# CURVATURES=("0.05" "0.1" "0.15" "0.2")
# LAMBDAS=("0.1" "0.3" "0.5" "1.0")

# Lightweight grid (9 combos) — uncomment for a quick scan
# CURVATURES=("0.01" "0.1" "1.0")
# LAMBDAS=("0.1" "0.3" "1.0")

BATCH_SIZE=3   # submit N jobs per round, wait, then next batch

# ---------------------------------------------------------------------------
# Build job list
# ---------------------------------------------------------------------------
JOBS=()
JOB_SPECS=()   # "CURV|LAMBDA|SAFE_CURV|SAFE_LAMBDA|OUT_DIR|LOG_PATH"

for CURV in "${CURVATURES[@]}"; do
  for LAMBDA in "${LAMBDAS[@]}"; do
    SAFE_CURV="${CURV/./p}"
    SAFE_LAMBDA="${LAMBDA/./p}"
    TAG="c${SAFE_CURV}_la${SAFE_LAMBDA}"
    OUT_DIR="${ROOT_DIR}/checkpoints_locomo_category_${TAG}"
    LOG_PATH="${ROOT_DIR}/job_train_${TAG}.out"
    JOB_SPECS+=("${CURV}|${LAMBDA}|${SAFE_CURV}|${SAFE_LAMBDA}|${OUT_DIR}|${LOG_PATH}")
  done
done

TOTAL_JOBS=${#JOB_SPECS[@]}
echo "============================================================"
echo "Grid search: ${#CURVATURES[@]} curvatures × ${#LAMBDAS[@]} lambdas"
echo "Total jobs: ${TOTAL_JOBS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Batches:    $(( (TOTAL_JOBS + BATCH_SIZE - 1) / BATCH_SIZE ))"
echo "============================================================"
echo ""
echo "Curvatures:  ${CURVATURES[*]}"
echo "Lambdas:     ${LAMBDAS[*]}"
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
    IFS='|' read -r CURV LAMBDA SAFE_CURV SAFE_LAMBDA OUT_DIR LOG_PATH <<< "${JOB_SPECS[$j]}"

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

  # Wait for this batch to finish before submitting the next
  if ((BATCH_END < TOTAL_JOBS)); then
    echo "  Waiting for batch ${BATCH_NUM} to finish..."
    DEP_STR=$(IFS=:; echo "${BATCH_JOB_IDS[*]}")
    # Submit a no-op placeholder that depends on all batch jobs;
    # wait for it to start, ensuring the batch is done.
    WAIT_JOB=$(sbatch --parsable \
      --dependency "afterok:${DEP_STR}" \
      --wrap "echo 'batch ${BATCH_NUM} done'" \
      --output /dev/null \
      --time 00:01:00)
    # Poll until the wait job completes
    while squeue -j "${WAIT_JOB}" 2>/dev/null | grep -q "${WAIT_JOB}"; do
      sleep 10
    done
    echo "  Batch ${BATCH_NUM} complete."
  fi

  echo ""
done

echo "============================================================"
echo "All ${TOTAL_JOBS} train jobs submitted across ${BATCH_NUM} batches."
echo ""
echo "Final checkpoints will be at:"
for ((i = 0; i < TOTAL_JOBS; i++)); do
  IFS='|' read -r CURV LAMBDA SAFE_CURV SAFE_LAMBDA OUT_DIR LOG_PATH <<< "${JOB_SPECS[$i]}"
  echo "  ${OUT_DIR}/hyperbolic_projector_final.pt"
done
echo "============================================================"
