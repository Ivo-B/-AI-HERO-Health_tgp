#!/bin/bash

#SBATCH --job-name=AI-HERO_health_baseline_score_calculation
#SBATCH --partition=haicore-gpu4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=/hkfs/work/workspace/scratch/im9193-H5/scoring.txt

group_workspace=/hkfs/work/workspace/scratch/im9193-H5

gt_path=/hkfs/work/workspace/scratch/im9193-health_challenge/data/valid.csv
save_path=${group_workspace}/submission_test/

source /hkfs/work/workspace/scratch/im9193-H5/AI-HERO-Health_tgp/.venv/bin/activate
python -u ${group_workspace}/AI-HERO-Health_tgp/evaluation/calc_score.py --preds ${group_workspace}/submission_test/predictions.csv --gt ${gt_path} --save_dir ${save_path}
