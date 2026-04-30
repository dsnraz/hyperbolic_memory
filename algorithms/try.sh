#!/bin/bash
#SBATCH -p gpu_chen
#SBATCH -n 1
#SBATCH -G 1
#SBATCH -o job7.out
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate memory
cd //share/home/leiyh5/Memory
python  -m algorithms.diagnose