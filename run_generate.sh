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
SAVE_DIR="/cluster/projects/kumargroup/sean/Methylation_Generation/generated_images"

# Generation settings
CANCER_TYPE=""     # e.g. "TCGA-BLCA" for a specific type, or leave empty for all types
NUM_IMAGES=5
COND_SCALE=3.0
SEED=42
# -------------------------

mkdir -p logs "${SAVE_DIR}"

# Activate Conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME}"

# Run the generation script
CMD="python generate_tiles.py \
    --config ${CONFIG} \
    --save_dir ${SAVE_DIR} \
    --num_images ${NUM_IMAGES} \
    --cond_scale ${COND_SCALE} \
    --seed ${SEED}"

if [ -n "${CANCER_TYPE}" ]; then
    CMD="${CMD} --cancer_type ${CANCER_TYPE}"
fi

${CMD}
