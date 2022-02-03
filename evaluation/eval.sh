#!/bin/bash

#SBATCH --job-name=AI-HERO_health_baseline_evaluation
#SBATCH --partition=haicore-gpu4
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=152
#SBATCH --time=00:30:00
#SBATCH --output=/hkfs/work/workspace/scratch/im9193-H5/baseline_eval.txt

export CUDA_CACHE_DISABLE=1
export OMP_NUM_THREADS=8

group_workspace=/hkfs/work/workspace/scratch/im9193-H5
data=/hkfs/work/workspace/scratch/im9193-health_challenge

source ${group_workspace}/health_env/bin/activate

weights_path=/hkfs/work/workspace/scratch/im9193-H5/logs/experiments/effinet/runs/2022-02-02/18-26-53/checkpoints/epoch_004.ckpt
python -u ${group_workspace}/AI-HERO-Health_tgp/run_eval.py --weights_path $weights_path --save_dir ${group_workspace}/submission --data_dir ${data}
