#!/bin/bash

#SBATCH --job-name=multi_gpu_pytorch
#SBATCH --time=6:00:00
#SBATCH --mem=16000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4g.32gb:2
#SBATCH --partition=gpu.medium

export SRUN_CPUS_PER_TASK=16

module add cuda/cudnn/8.4.1 python/3.10.5
module load anaconda

conda activate audio
srun python train.py \
    '+data_dir=../data/data_preprocessed/' \
    'model=m13ser' \
    'train=torch_dry_run_hpc'