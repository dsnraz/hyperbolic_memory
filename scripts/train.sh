#!/bin/bash
#SBATCH -p gpu_chen
#SBATCH -n 1
#SBATCH -G 1
#SBATCH -o job.out
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate memory
cd //share/home/leiyh5/Memory
python -m model.hyperbolic_utils.train \
    --mixed_training \
    --output_dir ./checkpoints_v2 \
    --mixed_total_iterations 50000