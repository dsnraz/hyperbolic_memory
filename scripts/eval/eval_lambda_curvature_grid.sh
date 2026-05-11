#!/bin/bash
set -euo pipefail

# ============================================================================
# Evaluate all curvature × lambda_centroid prediction files.
# Stats printed to stdout are captured to per-combo log files.
# No intermediate scored/stats JSON files are generated.
# A final sorted CSV summary is produced.
#
# Usage:
#   bash scripts/eval/eval_lambda_curvature_grid.sh
# ============================================================================

ROOT_DIR="/share/home/leiyh5/Memory"
ANN_FILE="${ROOT_DIR}/data/locomo/locomo10.json"
LOCOMO_ROOT="/share/home/leiyh5/locomo"
PREDICTION_KEY="memory_prediction"
MODEL_KEY="memory"

# ---------------------------------------------------------------------------
# Grid definition — MUST match the train/main grids
# ---------------------------------------------------------------------------

CURVATURES=("0.01" "0.1" "1.0" "10.0")
LAMBDAS=("0.1" "0.3" "0.5" "1.0")

# Fine grid around 0.1 — uncomment to replace above
# CURVATURES=("0.05" "0.1" "0.15" "0.2")
# LAMBDAS=("0.1" "0.3" "0.5" "1.0")

SUMMARY_CSV="${ROOT_DIR}/data/locomo/grid_search_summary.csv"
EVAL_LOG_DIR="${ROOT_DIR}/data/locomo/eval_logs"
mkdir -p "${EVAL_LOG_DIR}"

# ---------------------------------------------------------------------------
# Single Python helper to parse the printed stats from a log file
# ---------------------------------------------------------------------------
parse_log() {
  local log="$1"
  python -c "
import re, sys
with open('${log}') as f:
    text = f.read()

# Extract category F1s in order (1→5) and overall from lines like:
#   1. multi-hop retrieval: count=282, avg_f1=0.275
#   5. adversarial: count=446, avg_f1=0.830
#   overall: count=1986, avg_f1=0.460
cat_f1s = []
for m in re.finditer(r'^\s*\d+\.\s+\S+:.*?avg_f1=([\d.]+)', text, re.MULTILINE):
    cat_f1s.append(float(m.group(1)))
overall_m = re.search(r'^\s*overall:.*?avg_f1=([\d.]+)', text, re.MULTILINE)
overall = float(overall_m.group(1)) if overall_m else 0.0

mh   = cat_f1s[0] if len(cat_f1s) >= 1 else 0.0
temp = cat_f1s[1] if len(cat_f1s) >= 2 else 0.0
od   = cat_f1s[2] if len(cat_f1s) >= 3 else 0.0
sh   = cat_f1s[3] if len(cat_f1s) >= 4 else 0.0
adv  = cat_f1s[4] if len(cat_f1s) >= 5 else 0.0
print(f'{overall:.4f} {mh:.4f} {temp:.4f} {od:.4f} {sh:.4f} {adv:.4f}')
"
}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
echo "============================================================"
echo "Evaluation Grid: curvature × lambda_centroid"
echo "Eval logs: ${EVAL_LOG_DIR}"
echo "============================================================"
echo ""

# CSV header
echo "curvature,lambda_centroid,overall_f1,multi_hop_f1,temporal_f1,open_domain_f1,single_hop_f1,adversarial_f1" > "${SUMMARY_CSV}"

declare -a RESULTS

for CURV in "${CURVATURES[@]}"; do
  for LAMBDA in "${LAMBDAS[@]}"; do
    SAFE_CURV="${CURV/./p}"
    SAFE_LAMBDA="${LAMBDA/./p}"
    TAG="c${SAFE_CURV}_la${SAFE_LAMBDA}"
    PRED_FILE="${ROOT_DIR}/data/locomo/locomo10_pred_${TAG}.json"
    EVAL_LOG="${EVAL_LOG_DIR}/eval_${TAG}.out"

    if [[ ! -f "${PRED_FILE}" ]]; then
      echo "[skip] prediction file not found: ${PRED_FILE}"
      continue
    fi

    echo "--- Evaluating ${TAG} ---"

    # Run evaluation: stdout → log, no scored/stats JSON written
    if ! python -m scripts.eval.evaluate_locomo_predictions \
      --ann-file "${ANN_FILE}" \
      --pred-file "${PRED_FILE}" \
      --prediction-key "${PREDICTION_KEY}" \
      --model-key "${MODEL_KEY}" \
      --locomo-root "${LOCOMO_ROOT}" \
      > "${EVAL_LOG}" 2>&1; then
      echo "  [ERROR] Evaluation failed for ${TAG}, see ${EVAL_LOG}"
      continue
    fi

    # Parse scores from the log
    if [[ -f "${EVAL_LOG}" ]]; then
      read -r OVERALL MH TEMP OD SH ADV <<< "$(parse_log "${EVAL_LOG}")"
      echo "${CURV},${LAMBDA},${OVERALL},${MH},${TEMP},${OD},${SH},${ADV}" >> "${SUMMARY_CSV}"
      RESULTS+=("${OVERALL}|curv=${CURV} lambda=${LAMBDA}|overall=${OVERALL} mh=${MH} temp=${TEMP} od=${OD} sh=${SH} adv=${ADV}")
      echo "  overall=${OVERALL}  sh=${SH}  temp=${TEMP}  adv=${ADV}"
    fi
  done
done

# ---------------------------------------------------------------------------
# Sorted summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "SORTED SUMMARY (by overall F1, descending)"
echo "============================================================"
echo ""

for entry in "${RESULTS[@]}"; do
  echo "${entry}"
done | sort -t'|' -k1 -rn | while IFS='|' read -r score rest; do
  echo "  ${rest}"
done

echo ""
echo "============================================================"
echo "Eval logs: ${EVAL_LOG_DIR}"
echo "CSV saved: ${SUMMARY_CSV}"
echo "============================================================"

# Print a formatted table
echo ""
echo "Compact table:"
echo ""
python -c "
import csv
with open('${SUMMARY_CSV}') as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)
rows.sort(key=lambda r: float(r[2]), reverse=True)
fmt = '{:<10} {:<12} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}'
print(fmt.format(*header))
print('-' * 80)
for r in rows:
    print(fmt.format(*r))
"
