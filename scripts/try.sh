#!/bin/bash
#SBATCH -p gpu_chen
#SBATCH -n 1
#SBATCH -G 1
#SBATCH -o job3.out
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate memory
cd //share/home/leiyh5/Memory
python  -m model.retrievers.try_retriver2