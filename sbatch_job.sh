#!/bin/bash

#SBATCH --job-name=m13_dry_run
#SBATCH --time=6:00:00
#SBATCH --mem=32000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:7g.80gb:1
#SBATCH --partition=gpu

module add cuda/cudnn/8.4.1 python/3.10.5
module load anaconda

conda activate audio
srun python train.py \
    '+data_dir=../data/data_preprocessed/' \
    'model=m13ser' \
    'train=torch_dry_run_hpc'