#!/bin/bash
#SBATCH -p gpu_chen
#SBATCH -n 1
#SBATCH -G 1
#SBATCH -o job6.out
#
# model.llm_inference.session_run 路径类参数（与 session_run.parse_args 默认一致，可删行或改值覆盖）
#
source ~/miniconda3/etc/profile.d/conda.sh
conda activate memory
cd /share/home/leiyh5/Memory

python -m model.llm_inference.session_run \
  --data-file /share/home/leiyh5/Memory/data/locomo/locomo10.json \
  --persist-directory /share/home/leiyh5/Memory/data/memory_running_category_morefact \
  --llm-model-path /share/home/leiyh5/models/Qwen2.5-7B-Instruct \
  --projector-checkpoint-path /share/home/leiyh5/Memory/checkpoints_locomo_categorymorefact_c0p1_la0p3/hyperbolic_projector_final.pt \
  --embedding-model sentence-transformers/all-mpnet-base-v2 \
  --out-file /share/home/leiyh5/Memory/data/locomo/locomo10_hyhyibd_c0p1_morefact.json \
  --generation-model-path /share/home/leiyh5/models/Qwen2.5-7B-Instruct \
  --retriever-type hyperbolic_angular \
  "$@"
