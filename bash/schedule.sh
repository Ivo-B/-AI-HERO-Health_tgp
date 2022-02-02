#!/bin/bash

#SBATCH --job-name=AI-HERO_health
#SBATCH --partition=haicore-gpu4
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --output=/hkfs/work/workspace/scratch/im9193-H5/baseline_training.txt

#export CUDA_CACHE_DISABLE=1
#export OMP_NUM_THREADS=8

group_workspace=/hkfs/work/workspace/scratch/im9193-H5

#module load devel/cuda/11.3

source ${group_workspace}/AI-HERO-Health_tgp/.venv/bin/activate
python ${group_workspace}/AI-HERO-Health_tgp/run.py experiment=baseline
