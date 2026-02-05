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

# --- EDIT THESE ---
CONDA_ENV_NAME="methylcdm"
DATA_ROOT="/cluster/projects/kumargroup/sean/Methylation_Generation"
SAVE_DIR="/cluster/projects/kumargroup/sean/Methylation_Generation/models"
NUM_GPUS=1        # set to >1 for multi-GPU DDP (also update --gres above)
# ------------------

mkdir -p logs "${SAVE_DIR}"

# Activate Conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME}"

# Single GPU: python, Multi-GPU: torchrun
if [ "${NUM_GPUS}" -gt 1 ]; then
    LAUNCH="torchrun --nproc_per_node=${NUM_GPUS}"
else
    LAUNCH="python"
fi

${LAUNCH} newnew_main.py \
    --path_to_patches "${DATA_ROOT}/patches" \
    --cancer_types TCGA-BLCA TCGA-BRCA TCGA-GBM TCGA-HNSC TCGA-KIRC \
    --max_patches_per_wsi 200 \
    --run_name "5cancer_cond_dim128" \
    --save_dir "${SAVE_DIR}" \
    --batch_size 8 \
    --max_batch_size 128 \
    --num_epochs 50 \
    --timesteps 1000 \
    --dim 128 \
    --lr 1e-4 \
    --num_workers 8 \
    --num_iter_save 500
