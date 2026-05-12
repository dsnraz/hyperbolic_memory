#!/bin/bash
set -euo pipefail

# ============================================================================
# Evaluate fine lambda sweep predictions at curvature 0.1.
# Run this AFTER main_lambda_sweep.sh completes.
#
# Usage:
#   bash scripts/eval/eval_lambda_sweep.sh
# ============================================================================

ROOT_DIR="/share/home/leiyh5/Memory"
cd "${ROOT_DIR}" || exit 1

ANN_FILE="${ROOT_DIR}/data/locomo/locomo10.json"
LOCOMO_ROOT="/share/home/leiyh5/locomo"
PREDICTION_KEY="memory_prediction"
MODEL_KEY="memory"

CURV=0.1
LAMBDAS=("0.0" "0.1" "0.15" "0.2" "0.25" "0.3" "0.35" "0.4")

SUMMARY_CSV_F1="${ROOT_DIR}/data/locomo/lambda_sweep_f1.csv"
SUMMARY_CSV_BLEU="${ROOT_DIR}/data/locomo/lambda_sweep_bleu1.csv"
EVAL_LOG_DIR="${ROOT_DIR}/data/locomo/eval_logs"
mkdir -p "${EVAL_LOG_DIR}"

# ---------------------------------------------------------------------------
parse_log() {
  local log="$1"
  python -c '
import re, sys
path = sys.argv[1]
with open(path, encoding="utf-8", errors="replace") as f:
    text = f.read()

cat_pat = re.compile(r"^\s*\d+\.\s+.+?:\s*count=\d+,\s*avg_f1=([\d.]+)", re.MULTILINE)
cat_f1s = [float(m.group(1)) for m in cat_pat.finditer(text)]
overall_m = re.search(r"^\s*overall:\s*count=\d+,\s*avg_f1=([\d.]+)", text, re.MULTILINE)
overall = float(overall_m.group(1)) if overall_m else 0.0

bleu_cat_pat = re.compile(r"^\s*\d+\.\s+.+?:\s*count=\d+,\s*avg_bleu1=([\d.]+)", re.MULTILINE)
cat_bleus = [float(m.group(1)) for m in bleu_cat_pat.finditer(text)]
overall_bleu_m = re.search(r"^\s*overall:\s*count=\d+,\s*avg_bleu1=([\d.]+)", text, re.MULTILINE)
overall_bleu = float(overall_bleu_m.group(1)) if overall_bleu_m else 0.0

mh   = cat_f1s[0] if len(cat_f1s)>=1 else 0.0
temp = cat_f1s[1] if len(cat_f1s)>=2 else 0.0
od   = cat_f1s[2] if len(cat_f1s)>=3 else 0.0
sh   = cat_f1s[3] if len(cat_f1s)>=4 else 0.0
adv  = cat_f1s[4] if len(cat_f1s)>=5 else 0.0
bmh   = cat_bleus[0] if len(cat_bleus)>=1 else 0.0
btemp = cat_bleus[1] if len(cat_bleus)>=2 else 0.0
bod   = cat_bleus[2] if len(cat_bleus)>=3 else 0.0
bsh   = cat_bleus[3] if len(cat_bleus)>=4 else 0.0
badv  = cat_bleus[4] if len(cat_bleus)>=5 else 0.0
print(f"{overall:.4f} {mh:.4f} {temp:.4f} {od:.4f} {sh:.4f} {adv:.4f} {overall_bleu:.4f} {bmh:.4f} {btemp:.4f} {bod:.4f} {bsh:.4f} {badv:.4f}")
' "$log"
}

# ---------------------------------------------------------------------------
echo "============================================================"
echo "Fine lambda sweep evaluation (c=0.1)"
echo "============================================================"
echo ""

HEADER_F1="lambda_centroid,overall_f1,multi_hop_f1,temporal_f1,open_domain_f1,single_hop_f1,adversarial_f1"
HEADER_BLEU="lambda_centroid,overall_bleu1,multi_hop_bleu1,temporal_bleu1,open_domain_bleu1,single_hop_bleu1,adversarial_bleu1"
echo "${HEADER_F1}" > "${SUMMARY_CSV_F1}"
echo "${HEADER_BLEU}" > "${SUMMARY_CSV_BLEU}"

for LAMBDA in "${LAMBDAS[@]}"; do
  SAFE_LAMBDA="${LAMBDA/./p}"
  TAG="c0p1_la${SAFE_LAMBDA}"
  PRED_FILE="${ROOT_DIR}/data/locomo/locomo10_pred_${TAG}.json"
  EVAL_LOG="${EVAL_LOG_DIR}/eval_${TAG}.out"

  if [[ ! -f "${PRED_FILE}" ]]; then
    echo "[skip] prediction file not found: ${PRED_FILE}"
    continue
  fi

  echo "--- Evaluating lambda=${LAMBDA} ---"

  if ! python -m scripts.eval.evaluate_locomo_predictions \
    --ann-file "${ANN_FILE}" \
    --pred-file "${PRED_FILE}" \
    --prediction-key "${PREDICTION_KEY}" \
    --model-key "${MODEL_KEY}" \
    --locomo-root "${LOCOMO_ROOT}" \
    > "${EVAL_LOG}" 2>&1; then
    echo "  [ERROR] Evaluation failed for lambda=${LAMBDA}, see ${EVAL_LOG}"
    continue
  fi

  if [[ -f "${EVAL_LOG}" ]]; then
    read -r OVERALL MH TEMP OD SH ADV OB1 BMH BTEMP BOD BSH BADV <<< "$(parse_log "${EVAL_LOG}")"
    echo "${LAMBDA},${OVERALL},${MH},${TEMP},${OD},${SH},${ADV}" >> "${SUMMARY_CSV_F1}"
    echo "${LAMBDA},${OB1},${BMH},${BTEMP},${BOD},${BSH},${BADV}" >> "${SUMMARY_CSV_BLEU}"
    echo "  lambda=${LAMBDA}  overall_f1=${OVERALL}  sh=${SH}  adv=${ADV}  bleu1=${OB1}"
  fi
done

# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "F1  (sorted by overall_f1 desc)"
echo "============================================================"
python -c "
import csv
with open('${SUMMARY_CSV_F1}') as f:
    rows = list(csv.reader(f))
header, data = rows[0], rows[1:]
data.sort(key=lambda r: float(r[1]), reverse=True)
fmt = '{:<10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}'
print(fmt.format(*header))
print('-' * 80)
for r in data:
    print(fmt.format(*[f'{float(x):.4f}' if i>0 else x for i,x in enumerate(r)]))
"

echo ""
echo "============================================================"
echo "BLEU1 (sorted by overall_bleu1 desc)"
echo "============================================================"
python -c "
import csv
with open('${SUMMARY_CSV_BLEU}') as f:
    rows = list(csv.reader(f))
header, data = rows[0], rows[1:]
data.sort(key=lambda r: float(r[1]), reverse=True)
fmt = '{:<10} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}'
print(fmt.format(*header))
print('-' * 80)
for r in data:
    print(fmt.format(*[f'{float(x):.4f}' if i>0 else x for i,x in enumerate(r)]))
"

echo ""
echo "CSV (F1):   ${SUMMARY_CSV_F1}"
echo "CSV (BLEU): ${SUMMARY_CSV_BLEU}"
