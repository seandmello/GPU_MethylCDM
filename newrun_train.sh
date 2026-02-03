#!/bin/bash
#SBATCH --job-name=methylcdm
#SBATCH -p gpu
#SBATCH -A kumargroup_gpu
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00

# --- EDIT THESE PATHS ---
CONDA_ENV_NAME="methylcdm"          # name of your Conda environment
DATA_ROOT="/cluster/projects/kumargroup/sean/Methylation_Generation/"            # parent directory containing patches/ and rna_data/
SAVE_DIR="/cluster/projects/kumargroup/sean/Methylation_Generation/models"      # where to save model checkpoints
# -------------------------

mkdir -p logs "${SAVE_DIR}"

# Activate Conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME}"

# Run the Python training script
python newnew_main.py \
    --path_to_patches "${DATA_ROOT}/mnist_patches" \
    --path_to_methyl "${DATA_ROOT}/rna_data" \
    --save_dir "${SAVE_DIR}" \
    --batch_size 2 \
    --max_batch_size 128 \
    --num_epochs 20 \
    --timesteps 1000 \
    --dim 128 \
    --lr 1e-4 \
    --num_workers 8 \
    --num_iter_save 500