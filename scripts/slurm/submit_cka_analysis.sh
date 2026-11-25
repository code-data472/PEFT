#!/bin/bash
#SBATCH --job-name=cka_analysis
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/cka_%A.out

source ~/.bashrc
conda activate peft_lrasr

python -m tools.compute_cka --experiments_dir experiments/ --output_dir analysis/cka/
python -m tools.drift_correlation --cka_dir analysis/cka/ --output_dir analysis/correlation/
