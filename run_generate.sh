#!/bin/bash
#SBATCH --job-name=methylgen
#SBATCH -p gpu
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH -A kumargroup_gpu

# --- EDIT THESE PATHS ---
CONDA_ENV_NAME="methylcdm"
CONFIG="/cluster/home/t140585uhn/GPU_MethylCDM/model_config.json"
RNA_DIR="/cluster/projects/kumargroup/sean/Methylation_Generation/rna_data"
SAVE_DIR="/cluster/projects/kumargroup/sean/Methylation_Generation/generated_images"

# Generation settings
NUM_IMAGES=5
COND_SCALE=3.0
SEED=42
# -------------------------

mkdir -p logs "${SAVE_DIR}"

# Activate Conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME}"

# Run the generation script
python generate_tiles.py \
    --config "${CONFIG}" \
    --rna_dir "${RNA_DIR}" \
    --save_dir "${SAVE_DIR}" \
    --num_images "${NUM_IMAGES}" \
    --cond_scale "${COND_SCALE}" \
    --seed "${SEED}"
