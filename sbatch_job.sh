#!/bin/bash

#SBATCH --job-name=multi_gpu_pytorch
#SBATCH --time=6:00:00
#SBATCH --mem=16000
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2g.20gb:2
#SBATCH --partition=gpu.medium

module add cuda/cudnn/8.4.1 python/3.10.5
module load anaconda

conda activate audio
srun python train.py \
    '+data_dir=../data/data_preprocessed/' \
    'model=m13ser' \
    'train=torch_dry_run_hpc'