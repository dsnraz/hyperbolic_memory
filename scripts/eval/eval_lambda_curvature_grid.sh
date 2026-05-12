#!/bin/bash
set -euo pipefail

# ============================================================================
# Evaluate all curvature × lambda_centroid prediction files.
# Stats printed to stdout are captured to per-combo log files.
# No intermediate scored/stats JSON files are generated.
# Two CSV summaries: F1-only and BLEU1-only columns, each sorted by its own overall (desc).
#
# Usage:
#   bash scripts/eval/eval_lambda_curvature_grid.sh
# ============================================================================

ROOT_DIR="/share/home/leiyh5/Memory"
cd "${ROOT_DIR}" || exit 1

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

SUMMARY_CSV_F1="${ROOT_DIR}/data/locomo/grid_search_summary_f1.csv"
SUMMARY_CSV_BLEU="${ROOT_DIR}/data/locomo/grid_search_summary_bleu1.csv"
EVAL_LOG_DIR="${ROOT_DIR}/data/locomo/eval_logs"
mkdir -p "${EVAL_LOG_DIR}"

# ---------------------------------------------------------------------------
# Single Python helper to parse the printed stats from a log file
# ---------------------------------------------------------------------------
parse_log() {
  local log="$1"
  # 单引号包裹整段 Python，避免 bash 双引号里误解析 f-string 的 {…} 或 $(
  # 日志路径用 sys.argv[1] 传入，路径里含引号也安全。
  python -c '
import re
import sys

path = sys.argv[1]
with open(path, encoding="utf-8", errors="replace") as f:
    text = f.read()

# Category stats 行：类别名里可含空格，用 .+? 到第一个冒号。
cat_pat = re.compile(
    r"^\s*\d+\.\s+.+?:\s*count=\d+,\s*avg_f1=([\d.]+)",
    re.MULTILINE,
)
cat_f1s = [float(m.group(1)) for m in cat_pat.finditer(text)]

overall_m = re.search(
    r"^\s*overall:\s*count=\d+,\s*avg_f1=([\d.]+)",
    text,
    re.MULTILINE,
)
overall = float(overall_m.group(1)) if overall_m else 0.0

mh = cat_f1s[0] if len(cat_f1s) >= 1 else 0.0
temp = cat_f1s[1] if len(cat_f1s) >= 2 else 0.0
od = cat_f1s[2] if len(cat_f1s) >= 3 else 0.0
sh = cat_f1s[3] if len(cat_f1s) >= 4 else 0.0
adv = cat_f1s[4] if len(cat_f1s) >= 5 else 0.0

bleu_cat_pat = re.compile(
    r"^\s*\d+\.\s+.+?:\s*count=\d+,\s*avg_bleu1=([\d.]+)",
    re.MULTILINE,
)
cat_bleus = [float(m.group(1)) for m in bleu_cat_pat.finditer(text)]

overall_bleu_m = re.search(
    r"^\s*overall:\s*count=\d+,\s*avg_bleu1=([\d.]+)",
    text,
    re.MULTILINE,
)
overall_bleu = float(overall_bleu_m.group(1)) if overall_bleu_m else 0.0

bmh = cat_bleus[0] if len(cat_bleus) >= 1 else 0.0
btemp = cat_bleus[1] if len(cat_bleus) >= 2 else 0.0
bod = cat_bleus[2] if len(cat_bleus) >= 3 else 0.0
bsh = cat_bleus[3] if len(cat_bleus) >= 4 else 0.0
badv = cat_bleus[4] if len(cat_bleus) >= 5 else 0.0

# 12 个数：overall_f1 + 5 类 f1；overall_bleu1 + 5 类 bleu1
print(
    f"{overall:.4f} {mh:.4f} {temp:.4f} {od:.4f} {sh:.4f} {adv:.4f} "
    f"{overall_bleu:.4f} {bmh:.4f} {btemp:.4f} {bod:.4f} {bsh:.4f} {badv:.4f}"
)
' "$log"
}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
echo "============================================================"
echo "Evaluation Grid: curvature × lambda_centroid"
echo "Eval logs: ${EVAL_LOG_DIR}"
echo "============================================================"
echo ""

# 两张表：仅 F1 列 / 仅 BLEU-1 列；跑完后各自按 overall（第 3 列）降序排序
HEADER_F1="curvature,lambda_centroid,overall_f1,multi_hop_f1,temporal_f1,open_domain_f1,single_hop_f1,adversarial_f1"
HEADER_BLEU="curvature,lambda_centroid,overall_bleu1,multi_hop_bleu1,temporal_bleu1,open_domain_bleu1,single_hop_bleu1,adversarial_bleu1"
echo "${HEADER_F1}" > "${SUMMARY_CSV_F1}"
echo "${HEADER_BLEU}" > "${SUMMARY_CSV_BLEU}"

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
      read -r OVERALL MH TEMP OD SH ADV OB1 BMH BTEMP BOD BSH BADV <<< "$(parse_log "${EVAL_LOG}")"
      echo "${CURV},${LAMBDA},${OVERALL},${MH},${TEMP},${OD},${SH},${ADV}" >> "${SUMMARY_CSV_F1}"
      echo "${CURV},${LAMBDA},${OB1},${BMH},${BTEMP},${BOD},${BSH},${BADV}" >> "${SUMMARY_CSV_BLEU}"
      RESULTS+=("${OVERALL}|curv=${CURV} lambda=${LAMBDA}|overall=${OVERALL} mh=${MH} temp=${TEMP} od=${OD} sh=${SH} adv=${ADV}")
      echo "  overall=${OVERALL}  sh=${SH}  temp=${TEMP}  adv=${ADV}"
    fi
  done
done

# 各自按第 3 列 overall 降序写回（F1 表 / BLEU 表互不混列）
python -c '
import csv
import sys

def sort_by_col2(path: str) -> None:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return
    header, data = rows[0], rows[1:]

    def key(row):
        try:
            return float(row[2])
        except (ValueError, IndexError):
            return float("-inf")

    data.sort(key=key, reverse=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(data)

sort_by_col2(sys.argv[1])
sort_by_col2(sys.argv[2])
' "${SUMMARY_CSV_F1}" "${SUMMARY_CSV_BLEU}"

# ---------------------------------------------------------------------------
# Sorted summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "SORTED SUMMARY (by overall F1, descending)"
echo "============================================================"
echo ""

# 首列为 overall F1，须按浮点降序；-n 对多位小数不可靠，用 -g（general numeric）
for entry in "${RESULTS[@]}"; do
  echo "${entry}"
done | LC_ALL=C sort -t'|' -k1,1gr | while IFS='|' read -r score rest; do
  echo "  ${rest}"
done

echo ""
echo "============================================================"
echo "Eval logs: ${EVAL_LOG_DIR}"
echo "CSV (F1 only, by overall_f1 desc):   ${SUMMARY_CSV_F1}"
echo "CSV (BLEU only, by overall_bleu1 desc): ${SUMMARY_CSV_BLEU}"
echo "============================================================"

# Print compact tables: F1 与 BLEU 各自按 overall 降序，行顺序可不同
echo ""
echo "Compact tables (read from sorted CSV files):"
echo ""
python -c '
import csv
import sys

path_f1 = sys.argv[1]
path_bleu = sys.argv[2]

def load(path):
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = list(next(reader))
        rows = [r for r in reader if len(r) >= 8]
    return header, rows

h1, r1 = load(path_f1)
h2, r2 = load(path_bleu)

fmt8 = "{:<10} {:<12} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}"

print("F1 only (file already sorted by overall_f1 desc):")
print(fmt8.format(*h1[:8]))
print("-" * 80)
for r in r1:
    print(fmt8.format(*r[:8]))
print()
print("BLEU-1 only (file already sorted by overall_bleu1 desc):")
print(fmt8.format(*h2[:8]))
print("-" * 80)
for r in r2:
    print(fmt8.format(*r[:8]))
' "${SUMMARY_CSV_F1}" "${SUMMARY_CSV_BLEU}"
