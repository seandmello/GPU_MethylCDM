#!/bin/bash
#SBATCH --job-name=gen_digits
#SBATCH -p himem
#SBATCH --output=logs/%j_gen_digits.out
#SBATCH --error=logs/%j_gen_digits.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00

mkdir -p logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate methylcdm

python generate_digit_data.py
