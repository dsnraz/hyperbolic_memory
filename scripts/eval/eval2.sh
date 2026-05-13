#!/bin/bash
#SBATCH -p gpu_chen
#SBATCH -n 1
#SBATCH -G 1
#SBATCH -o job_result2.out
#
# A-mem 风格 F1 + BLEU-1：scripts.eval.evaluate_locomo_predictions_amem_style
# 路径类参数与模块内 parse_args 默认一致，可在命令行追加覆盖
#
source ~/miniconda3/etc/profile.d/conda.sh
conda activate memory
cd /share/home/leiyh5/Memory

python -m scripts.eval.evaluate_locomo_predictions_amem_style \
  --ann-file /share/home/leiyh5/Memory/data/locomo/locomo10.json \
  --pred-file /share/home/leiyh5/Memory/data/locomo/locomo10_cosine.json \
  --prediction-key memory_prediction \
  --model-key memory \
  --scored-file /share/home/leiyh5/Memory/data/locomo/locomo10_cosine_amem_style_scored.json \
  --stats-file /share/home/leiyh5/Memory/data/locomo/locomo10_cosine_amem_style_stats.json \
  "$@"
