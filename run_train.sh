#!/bin/bash
#SBATCH --job-name=methylcdm
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# --- EDIT THESE PATHS ---
SIF_IMAGE="/path/to/methylcdm.sif"  # Singularity image built from Dockerfile
DATA_ROOT="/path/to/data"            # parent directory containing patches/ and rna_data/
SAVE_DIR="/path/to/checkpoints"      # where to save model checkpoints
# -------------------------

mkdir -p logs "${SAVE_DIR}"

singularity run --nv \
    --bind "${DATA_ROOT}":/data \
    --bind "${SAVE_DIR}":/output \
    "${SIF_IMAGE}" \
    --path_to_patches /data/patches \
    --path_to_methyl /data/rna_data \
    --save_dir /output \
    --batch_size 8 \
    --max_batch_size 128 \
    --num_epochs 100 \
    --timesteps 1000 \
    --dim 128 \
    --lr 1e-4 \
    --num_workers 8 \
    --num_iter_save 500
