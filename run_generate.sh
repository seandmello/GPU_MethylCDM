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
CHECKPOINT="/cluster/projects/kumargroup/sean/Methylation_Generation/models/model-step9500.pt.pt"
RNA_DIR="/cluster/projects/kumargroup/sean/Methylation_Generation/rna_data"
SAVE_DIR="/cluster/projects/kumargroup/sean/Methylation_Generation/generated_images"

# Generation settings
NUM_IMAGES=5
COND_SCALE=3.0
SEED=42

# Model architecture (must match training config)
TIMESTEPS=1000
DIM=128
DIM_MULTS="1 2 3 4"
NUM_RESNET_BLOCKS=3
LAYER_ATTNS="0 1 1 1"
LAYER_CROSS_ATTNS="0 1 1 1"
ATTN_HEADS=8
FF_MULT=2.0
LR=1e-4
# -------------------------

mkdir -p logs "${SAVE_DIR}"

# Activate Conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME}"

# Run the generation script
python generate_tiles.py \
    --checkpoint "${CHECKPOINT}" \
    --rna_dir "${RNA_DIR}" \
    --save_dir "${SAVE_DIR}" \
    --num_images "${NUM_IMAGES}" \
    --cond_scale "${COND_SCALE}" \
    --seed "${SEED}" \
    --timesteps "${TIMESTEPS}" \
    --dim "${DIM}" \
    --dim_mults ${DIM_MULTS} \
    --num_resnet_blocks "${NUM_RESNET_BLOCKS}" \
    --layer_attns ${LAYER_ATTNS} \
    --layer_cross_attns ${LAYER_CROSS_ATTNS} \
    --attn_heads "${ATTN_HEADS}" \
    --ff_mult "${FF_MULT}" \
    --lr "${LR}"
