#!/bin/bash
#SBATCH --job-name=peft_lrasr_core
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-107
#SBATCH --output=logs/core_%A_%a.out
#SBATCH --error=logs/core_%A_%a.err

CONFIG_DIR="configs/sweeps"
CONFIGS=($(ls ${CONFIG_DIR}/*.yaml | sort))
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

echo "Running config: ${CONFIG}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

source ~/.bashrc
conda activate peft_lrasr

python -m src.train --config ${CONFIG}

echo "Finished: $(date)"
