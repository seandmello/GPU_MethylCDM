#!/bin/bash
#SBATCH --job-name=extract_patches
#SBATCH -p himem
#SBATCH --output=logs/%j_extract.out
#SBATCH --error=logs/%j_extract.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

mkdir -p logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate methylcdm

DATA_ROOT="/cluster/projects/kumargroup/hayden/data/TCGA"
SAVE_ROOT="/cluster/projects/kumargroup/sean/Methylation_Generation/patches"

COHORTS=(
    TCGA-BLCA
    TCGA-BRCA
    TCGA-GBM
    TCGA-HNSC
    TCGA-KIRC
    TCGA-LGG
    TCGA-LIHC
    TCGA-LUAD
    TCGA-LUSC
    TCGA-OV
    TCGA-PAAD
    TCGA-PRAD
    TCGA-SKCM
    TCGA-THCA
    TCGA-UCEC
)

for COHORT in "${COHORTS[@]}"; do
    echo "========================================"
    echo "Processing ${COHORT}"
    echo "========================================"

    python extract_patches.py batch \
        --svs_dir "${DATA_ROOT}/${COHORT}/slides/raw" \
        --h5_dir "${DATA_ROOT}/${COHORT}/slides/trident/20x_256px_0px_overlap/patches" \
        --geojson_dir "${DATA_ROOT}/${COHORT}/slides/trident/contours_geojson" \
        --save_dir "${SAVE_ROOT}/${COHORT}" \
        --patch_size 256 \
        --target_magnification 20 \
        --max_workers 8 \
        --min_tissue 0.9 \
        --skip_errors \
        --clear_cache

    echo ""
done

echo "All cohorts complete."
