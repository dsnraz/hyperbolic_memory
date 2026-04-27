#!/bin/bash
#SBATCH -p gpu_chen
#SBATCH -n 1
#SBATCH -G 1
#SBATCH -o job6.out
#
# 使用多父外角加权双曲检索（hyperbolic_angular）跑 model.llm_inference.run。
# 路径默认值在 run.py 的 argparse 中；若需覆盖可追加参数，例如：
#   --max-samples 2 --max-questions 5 --persist-directory /path/to/chroma
#
source ~/miniconda3/etc/profile.d/conda.sh
conda activate memory
cd /share/home/leiyh5/Memory

python -m model.llm_inference.run
