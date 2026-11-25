#!/bin/bash
#SBATCH --job-name=peft_ablation
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/ablation_%A_%a.out
#SBATCH --error=logs/ablation_%A_%a.err

CONFIG_DIR="configs/ablation"
CONFIGS=($(ls ${CONFIG_DIR}/*.yaml | sort))
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

source ~/.bashrc
conda activate peft_lrasr

python -m src.train --config ${CONFIG}
