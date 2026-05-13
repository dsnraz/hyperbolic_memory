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
  --persist-directory /share/home/leiyh5/Memory/data/memory_running_category_384_subject \
  --llm-model-path /share/home/leiyh5/.cache/modelscope/hub/models/Qwen/Qwen2___5-3B-Instruct \
  --projector-checkpoint-path /share/home/leiyh5/Memory/checkpoints_locomo_category_c0p1_la0p3/hyperbolic_projector_final.pt \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --out-file /share/home/leiyh5/Memory/data/locomo/locomo10_cosine_subject.json \
  --generation-model-path /share/home/leiyh5/.cache/modelscope/hub/models/Qwen/Qwen2___5-3B-Instruct \
  --retriever-type cosine \
  "$@"
