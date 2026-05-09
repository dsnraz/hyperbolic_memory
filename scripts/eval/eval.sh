#!/bin/bash
#SBATCH -p gpu_chen
#SBATCH -n 1
#SBATCH -G 1
#SBATCH -o job_result.out
#
# scripts.eval.evaluate_locomo_predictions 路径类参数（与 parse_args 默认一致，可改值覆盖）
#
source ~/miniconda3/etc/profile.d/conda.sh
conda activate memory
cd //share/home/leiyh5/Memory

python -m scripts.eval.evaluate_locomo_predictions \
  --ann-file /share/home/leiyh5/Memory/data/locomo/locomo10.json \
  --pred-file /share/home/leiyh5/Memory/data/locomo/locomo10_category.json \
  --locomo-root /share/home/leiyh5/locomo \
  --scored-file /share/home/leiyh5/Memory/data/locomo/locomo10_category_scored.json \
  --stats-file /share/home/leiyh5/Memory/data/locomo/locomo10_category_stats.json \
  "$@"