#!/bin/bash
#SBATCH --job-name=peft_seed_rep
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/seed_%A_%a.out
#SBATCH --error=logs/seed_%A_%a.err

CONFIG=$1
SEEDS=(42 123 7)

for SEED in "${SEEDS[@]}"; do
    echo "Seed: ${SEED}"
    python -m src.train --config ${CONFIG} --override seed=${SEED} \
        output_dir=$(dirname $(grep output_dir ${CONFIG} | awk '{print $2}'))/seed_${SEED}
done
